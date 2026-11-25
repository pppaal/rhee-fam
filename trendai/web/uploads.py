"""``POST /api/upload`` internals (spec C.4, E.17): multipart decoding, the zip guard, in-memory parsing and column
detection, the preview, and the in-memory upload cache.

Nothing here writes to disk: the raw bytes are parsed in memory (``data.read_table_bytes``) and only the parsed
``RawTable`` is kept, for 30 minutes, in :class:`UploadCache` (the ``bytes`` object is dropped). Notices are captured
with ``data.collect_notices()`` (never through ``warnings``, E.13).

Bounds beyond the C.4 limits: at most :data:`MAX_MULTIPART_PARTS` form parts (``email.parser`` cost grows with
them), at most :data:`MAX_TABLE_ROWS` rows (detection cost), column / row labels cut to :data:`MAX_LABEL_CHARS`
(they become display names, export names and part of the AI prompt), binary files refused before the text
fallback, and at most :data:`CLIENT_MAX_ENTRIES` / :data:`CLIENT_MAX_BYTES` of the cache per client.
"""

from __future__ import annotations

import datetime as _dt
import email.parser
import email.policy
import io
import math
import re
import threading
import unicodedata
import zipfile
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

from .. import data, messages
from . import viewmodel
from .errors import WebError, error_object, from_data_error
from .store import iso_date

__all__ = ["parse_multipart", "display_filename", "zip_guard", "UploadEntry", "UploadCache", "analyse_upload",
           "stringify_cell", "MAX_TABLE_CELLS", "MAX_LONG_COLUMNS", "MAX_WIDE_ROWS", "MAX_WIDE_DATE_COLUMNS",
           "MAX_HEADER_COLUMNS", "UPLOAD_TTL_S", "MAX_TABLE_ROWS", "MAX_LABEL_CHARS", "MAX_MULTIPART_PARTS",
           "CLIENT_MAX_ENTRIES", "CLIENT_MAX_BYTES", "looks_binary"]

MAX_HEADER_COLUMNS = 5_000
MAX_TABLE_CELLS = 2_000_000
MAX_LONG_COLUMNS = 50
MAX_WIDE_ROWS = 200
MAX_WIDE_DATE_COLUMNS = 5_000
ZIP_MAX_UNCOMPRESSED = 30 * 1024 * 1024
ZIP_MAX_MEMBERS = 1_000
UPLOAD_TTL_S = 30 * 60
CACHE_MAX_ENTRIES = 64
CACHE_MAX_BYTES = 256 * 1024 * 1024
#: one client's share of the upload cache: its own oldest uploads make room first, so a few big uploads from one
#: address cannot push every other user's pending upload out
CLIENT_MAX_ENTRIES = 4
CLIENT_MAX_BYTES = 96 * 1024 * 1024
MAX_TABLE_ROWS = 200_000
MAX_LABEL_CHARS = 80
MAX_MULTIPART_PARTS = 10
_TOMBSTONES = 2_000
_BINARY_SAMPLE = 8192
_ZIP_SIG = b"PK\x03\x04"
_OLE_SIG = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"
_TEXT_BOMS = (b"\xef\xbb\xbf", b"\xff\xfe", b"\xfe\xff")
_PREVIEW_ROWS = 5
_CELL_MAX = 40
_FILENAME_MAX = 100
_CONTROL = re.compile(r"[\x00-\x1f\x7f-\x9f]")
_NUMERIC_SHARE = 0.8                 # the share of cells that must parse as numbers (C.4 "not_numeric": < 80 %)
_FREQ_LABELS = {"D": ("일간", "Daily"), "W": ("주간", "Weekly"), "M": ("월간", "Monthly"), "Q": ("분기", "Quarterly"),
                "Y": ("연간", "Yearly")}
_FORMAT_UNITS = {"D": ("일", "daily"), "W": ("주", "weekly"), "M": ("월", "monthly"), "Q": ("분기", "quarterly"),
                 "Y": ("연", "yearly")}


# --------------------------------------------------------------------------- request decoding

_BOUNDARY_RX = re.compile(r'boundary="?([^";]+)"?', re.IGNORECASE)


def parse_multipart(content_type: str, body: bytes) -> tuple[str, bytes]:
    """The first part with ``name="file"`` and a filename -> ``(filename, bytes)`` (E.17). Python 3.13 has no
    ``cgi``: the body is parsed with ``email.parser.BytesParser(policy=email.policy.HTTP)``. A missing boundary or
    part, or more than :data:`MAX_MULTIPART_PARTS` parts (counted before parsing: a 10 MB body of 200,000 tiny
    parts held a parse slot for ~25 s) -> ``400 bad_request``."""
    m = _BOUNDARY_RX.search(content_type)
    if m is None:
        raise WebError("bad_request")
    try:
        header = content_type.encode("latin-1")
        delimiter = b"--" + m.group(1).strip().encode("latin-1")
    except UnicodeEncodeError:
        raise WebError("bad_request") from None
    if body.count(delimiter) > MAX_MULTIPART_PARTS + 1:      # n parts have n delimiters plus the closing one
        raise WebError("bad_request")
    msg = email.parser.BytesParser(policy=email.policy.HTTP).parsebytes(b"Content-Type: " + header + b"\r\n\r\n" + body)
    if not msg.is_multipart():
        raise WebError("bad_request")
    for part in msg.iter_parts():
        if part.get_param("name", header="content-disposition") != "file":
            continue
        filename = part.get_filename()
        if filename is None:
            continue
        payload = part.get_payload(decode=True)
        return str(filename), payload if isinstance(payload, bytes) else b""
    raise WebError("bad_request")


def display_filename(name: Any, default: str = "upload") -> str:
    """Last path component (split on ``/`` and ``\\``), NFC, control characters removed, at most 100 characters."""
    text = unicodedata.normalize("NFC", str(name or ""))
    text = re.split(r"[/\\]", text)[-1]
    text = _CONTROL.sub("", text).strip()
    return text[:_FILENAME_MAX] or default


def zip_guard(raw: bytes) -> None:
    """Excel files are zip archives: refuse ones that unpack to more than 30 MB or hold more than 1,000 members
    (``413 upload_archive_too_large``); a broken archive, or one that is no spreadsheet (a Word file, a plain zip:
    no ``xl/`` part and no OpenDocument ``content.xml``) is ``422 upload_unreadable``."""
    if not raw.startswith(_ZIP_SIG):
        return
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            infos = zf.infolist()
    except (zipfile.BadZipFile, zipfile.LargeZipFile, OSError, ValueError):
        raise WebError("upload_unreadable", params={"reason": "excel_unreadable"}) from None
    if len(infos) > ZIP_MAX_MEMBERS or sum(max(0, i.file_size) for i in infos) > ZIP_MAX_UNCOMPRESSED:
        raise WebError("upload_archive_too_large")
    names = [i.filename for i in infos]
    if not any(n.startswith("xl/") for n in names) and "content.xml" not in names:
        raise WebError("upload_unreadable", params={"reason": "excel_unreadable"})


def looks_binary(raw: bytes) -> bool:
    """Whether bytes that are no Excel file (neither zip nor OLE) are binary rather than text: a PDF, an image, random
    bytes renamed ``.csv`` / ``.xlsx``. Text with a BOM is text; NUL bytes count as binary unless they sit at every
    other byte (UTF-16 without a BOM, which ``read_table_bytes`` decodes); otherwise more than 2 % control characters
    (other than tab, line breaks, form feed and ESC) mark binary. Without this check such a file was decoded as
    Latin-1 and reached the confirm step as mojibake."""
    if raw.startswith((_ZIP_SIG, _OLE_SIG)) or raw.startswith(_TEXT_BOMS):
        return False
    if raw.startswith(b"%PDF"):
        return True
    head = raw[:_BINARY_SAMPLE]
    if not head:
        return False
    if b"\x00" in head:
        half = max(1, len(head) // 2)
        even, odd = head[0::2].count(0), head[1::2].count(0)
        return not (max(even, odd) >= 0.3 * half and min(even, odd) <= 0.02 * half)
    controls = sum(1 for b in head if b < 32 and b not in (9, 10, 11, 12, 13, 27))
    return controls > 0.02 * len(head)


def _cap_label(text: str, taken: set) -> str:
    """A label cut to :data:`MAX_LABEL_CHARS` (with ``…``) and kept unique among ``taken`` (`` (2)``, …)."""
    base = text if len(text) <= MAX_LABEL_CHARS else text[:MAX_LABEL_CHARS - 1] + "…"
    name, n = base, 2
    while name in taken:
        suffix = f" ({n})"
        name = base[:MAX_LABEL_CHARS - len(suffix)] + suffix
        n += 1
    return name


def _cap_labels(table: Any) -> Any:
    """Cut over-long column labels and first-column texts (wide tables name their series from it) to
    :data:`MAX_LABEL_CHARS`: a 200,000-character header became the series name in every surface, the export file
    names and -- twice -- the AI prompt. Google Trends term names follow their columns."""
    import dataclasses

    frame = table.frame
    labels = [str(c) for c in frame.columns]
    if any(len(c) > MAX_LABEL_CHARS for c in labels):
        taken: set = set()
        renamed: dict[str, str] = {}
        new_labels = []
        for c in labels:
            new = c if len(c) <= MAX_LABEL_CHARS and c not in taken else _cap_label(c, taken)
            taken.add(new)
            renamed[c] = new
            new_labels.append(new)
        frame.columns = new_labels
        gt = table.google_trends
        if gt is not None:
            terms = {renamed.get(col, col): (str(term)[:MAX_LABEL_CHARS], region)
                     for col, (term, region) in gt.terms.items()}
            table = dataclasses.replace(table, google_trends=dataclasses.replace(gt, terms=terms))
    if len(frame.columns) and len(frame):
        first = frame.iloc[:, 0]
        long_cells = first.map(lambda v: isinstance(v, str) and len(v) > MAX_LABEL_CHARS)
        if bool(long_cells.any()):
            taken = {v for v in first if isinstance(v, str) and len(v) <= MAX_LABEL_CHARS}
            values = []
            for v, is_long in zip(first, long_cells):
                if is_long:
                    v = _cap_label(v, taken)
                    taken.add(v)
                values.append(v)
            frame.isetitem(0, values)
    return table


# --------------------------------------------------------------------------- cache

@dataclass
class UploadEntry:
    raw: Any                          # data.RawTable
    created_at: float                 # app clock timestamp
    filename: str
    size_bytes: int
    est_bytes: int
    wide: bool = False
    value_cols: list = field(default_factory=list)    # detected value columns / wide row names (table order)
    date_cols: list = field(default_factory=list)
    client_key: Optional[str] = None  # the uploading client (per-client share of the cache)
    n_points: int = 0                 # distinct dates of the detected series (the analyze work budget, C.5)
    freq: Optional[str] = None        # detected frequency


class UploadCache:
    """``{upload_id: UploadEntry}``: TTL 30 minutes, LRU-evicted above 64 entries or 256 MB (C.13, E.17); one
    client's own oldest entries go first once it holds more than :data:`CLIENT_MAX_ENTRIES` or
    :data:`CLIENT_MAX_BYTES`. Ids evicted for space are remembered for a while (:meth:`evicted`) so the client gets
    "upload again, the server had to drop it" instead of the 30-minute expiry text."""

    def __init__(self, clock: Callable[[], _dt.datetime], *, ttl_s: float = UPLOAD_TTL_S,
                 max_entries: int = CACHE_MAX_ENTRIES, max_bytes: int = CACHE_MAX_BYTES,
                 client_max_entries: int = CLIENT_MAX_ENTRIES, client_max_bytes: int = CLIENT_MAX_BYTES) -> None:
        self._clock = clock
        self._lock = threading.Lock()
        self._items: "OrderedDict[str, UploadEntry]" = OrderedDict()
        self._evicted: "OrderedDict[str, float]" = OrderedDict()
        self.ttl_s = float(ttl_s)
        self.max_entries = int(max_entries)
        self.max_bytes = int(max_bytes)
        self.client_max_entries = int(client_max_entries)
        self.client_max_bytes = int(client_max_bytes)

    def _now(self) -> float:
        return self._clock().timestamp()

    def _drop(self, upload_id: str) -> None:
        old = self._items.pop(upload_id, None)
        if old is not None and self._now() - old.created_at <= self.ttl_s:
            self._evicted[upload_id] = self._now()
            while len(self._evicted) > _TOMBSTONES:
                self._evicted.popitem(last=False)

    def put(self, upload_id: str, entry: UploadEntry) -> None:
        with self._lock:
            self._items[upload_id] = entry
            self._items.move_to_end(upload_id)
            if entry.client_key is not None:
                own = [(uid, e) for uid, e in self._items.items() if e.client_key == entry.client_key]
                own_bytes = sum(e.est_bytes for _uid, e in own)
                for uid, e in own:
                    if uid == upload_id or (len(own) <= self.client_max_entries
                                            and own_bytes <= self.client_max_bytes):
                        break
                    self._drop(uid)
                    own = [(u, x) for u, x in own if u != uid]
                    own_bytes -= e.est_bytes
            total = sum(e.est_bytes for e in self._items.values())
            while self._items and (len(self._items) > self.max_entries or total > self.max_bytes):
                uid = next(iter(self._items))
                total -= self._items[uid].est_bytes
                self._drop(uid)

    def get(self, upload_id: str) -> Optional[UploadEntry]:
        with self._lock:
            entry = self._items.get(upload_id)
            if entry is None:
                return None
            if self._now() - entry.created_at > self.ttl_s:
                del self._items[upload_id]
                return None
            self._items.move_to_end(upload_id)
            return entry

    def evicted(self, upload_id: str) -> bool:
        """Whether ``upload_id`` was dropped for space (not expired) within the TTL."""
        with self._lock:
            at = self._evicted.get(upload_id)
            return at is not None and self._now() - at <= self.ttl_s

    def sweep(self) -> int:
        now = self._now()
        with self._lock:
            old = [k for k, e in self._items.items() if now - e.created_at > self.ttl_s]
            for k in old:
                del self._items[k]
            for k in [k for k, at in self._evicted.items() if now - at > self.ttl_s]:
                del self._evicted[k]
        return len(old)

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)


# --------------------------------------------------------------------------- preview cells

def stringify_cell(value: Any) -> str:
    """A preview cell (C.4): Timestamps -> ``YYYY-MM-DD`` (``YYYY-MM-DD HH:MM`` when not midnight), floats ->
    ``format(v, ".15g")``, NaN/None -> ``""``, everything cut to 40 characters."""
    import pandas as pd

    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    try:
        if value is pd.NaT or (not isinstance(value, (str, bytes)) and pd.isna(value)):
            return ""
    except (TypeError, ValueError):
        pass
    if isinstance(value, (pd.Timestamp, _dt.datetime)):
        t = pd.Timestamp(value)
        text = f"{t.year:04d}-{t.month:02d}-{t.day:02d}"
        if (t.hour, t.minute, t.second) != (0, 0, 0):
            text += f" {t.hour:02d}:{t.minute:02d}"
    elif isinstance(value, _dt.date):
        text = f"{value.year:04d}-{value.month:02d}-{value.day:02d}"
    elif isinstance(value, bool):
        text = str(value)
    elif isinstance(value, float) or type(value).__name__.startswith("float"):
        text = format(float(value), ".15g")
        if text.endswith(".0"):
            text = text[:-2]
    else:
        text = str(value)
    return text[:_CELL_MAX]


def _preview(frame: Any) -> dict:
    columns = [str(c) for c in frame.columns]
    head = [[stringify_cell(v) for v in row] for row in frame.head(_PREVIEW_ROWS).itertuples(index=False, name=None)]
    tail = [[stringify_cell(v) for v in row] for row in frame.tail(_PREVIEW_ROWS).itertuples(index=False, name=None)]
    return {"columns": columns, "head": head, "tail": tail}


# --------------------------------------------------------------------------- detection

def _numeric_share(col: Any, prefer_decimal_comma: bool) -> float:
    try:
        with data.collect_notices():
            vals, n_bad, _ex = data.parse_numbers(col, prefer_decimal_comma=prefer_decimal_comma)
    except Exception:  # noqa: BLE001 - a column that cannot be read at all is simply not numeric
        return 0.0
    n_ok = int(vals.notna().sum())
    total = n_ok + int(n_bad)
    return round(n_ok / total, 3) if total else 0.0


_iso = iso_date


def _detected_dates(series: Any) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """``(freq, first_date, last_date)`` of the first detected series' dates."""
    import pandas as pd

    idx = pd.DatetimeIndex(series.index)
    idx = idx[~idx.isna()]
    if len(idx) == 0:
        return None, None, None
    freq: Optional[str] = None
    uniq = idx.unique().sort_values()
    if len(uniq) >= 2:
        try:
            with data.collect_notices():
                freq = data.infer_freq(uniq)
        except Exception:  # noqa: BLE001 - no frequency is a valid detection outcome
            freq = None
    return freq, _iso(idx.min()), _iso(idx.max())


def _format_label(label: Optional[str], freq: Optional[str], lang: str) -> Optional[str]:
    if not label:
        return None
    text = messages.format_label(label, lang)
    if lang == "en":
        return f"{text} format ({_FORMAT_UNITS[freq][1]})" if freq in _FORMAT_UNITS else f"{text} format"
    return f"{text} 형식 ({_FORMAT_UNITS[freq][0]})" if freq in _FORMAT_UNITS else f"{text} 형식"


def _rownum_cols(notes: Sequence[Warning]) -> list[str]:
    for w in notes:
        m = w.args[0] if getattr(w, "args", None) else None
        if isinstance(m, messages.Msg):
            for code, params in m.parts:
                if code == "rownum_cols_skipped":
                    return [str(c) for c in (params.get("cols") or [])]
    return []


def analyse_upload(raw: bytes, filename: str, *, hl: str, max_bytes: int, max_mb: int, created_at: float,
                   request_id: Optional[str] = None, client_key: Optional[str] = None) -> tuple[UploadEntry, dict]:
    """Parse and detect one upload (E.17). Returns the cache entry and the C.4 body *without* ``upload_id`` /
    ``expires_at`` (the handler adds them). Raises :class:`WebError` for the C.4 error cases."""
    if len(raw) > max_bytes:
        raise WebError("body_too_large", params={"max_mb": max_mb})
    zip_guard(raw)
    if looks_binary(raw):
        raise WebError("upload_unreadable", params={"reason": "file_unreadable"})
    name = display_filename(filename)
    with data.collect_notices() as read_notes:
        try:
            table = data.read_table_bytes(raw, name, max_columns=MAX_HEADER_COLUMNS, max_cells=MAX_TABLE_CELLS)
        except data.DataError as exc:
            raise from_data_error(exc, "upload_parse") from None
    frame = table.frame
    cells = int(frame.size)
    if cells > MAX_TABLE_CELLS:
        raise WebError("upload_too_large_table", params={"n": cells, "max": MAX_TABLE_CELLS})
    if len(frame) > MAX_TABLE_ROWS:     # detection reads every row (a 1M-row table held a parse slot ~27 s)
        raise WebError("upload_too_large_table",
                       params={"n": cells, "max": MAX_TABLE_ROWS * max(1, len(frame.columns))})
    table = _cap_labels(table)
    frame = table.frame

    problem: Optional[WebError] = None
    series: dict = {}
    excluded: list[str] = []
    with data.collect_notices() as detect_notes:
        try:
            series = data.frame_to_series(frame, prefer_decimal_comma=table.decimal_comma)
        except data.DataError as exc:
            code = str(getattr(exc, "code", "") or "")
            if code.startswith("table_empty"):
                raise from_data_error(exc, "upload_parse") from None
            if code in ("no_numeric_col", "only_rownum_cols") or code.startswith("value_"):
                problem = WebError("upload_no_value_col", status=200)
                if code == "only_rownum_cols":
                    excluded = [str(c) for c in (getattr(exc, "params", {}) or {}).get("cols") or []]
            else:
                problem = WebError("upload_no_date_col", status=200)
            series = {}
    excluded = excluded or _rownum_cols(detect_notes)
    first = next(iter(series.values()), None)
    attrs = dict(first.attrs.get("trendai") or {}) if first is not None else {}
    date_cols = [str(c) for c in attrs.get("date_cols") or []]
    wide = first is not None and not date_cols
    value_cols = [str(k) for k in series]

    n_columns = len(frame.columns)
    if wide:
        n_rows = len(frame)
        if n_rows > MAX_WIDE_ROWS:
            raise WebError("upload_too_many_columns", params={"n": n_rows, "max": MAX_WIDE_ROWS})
        if n_columns - 1 > MAX_WIDE_DATE_COLUMNS:
            raise WebError("upload_too_many_columns", params={"n": n_columns - 1, "max": MAX_WIDE_DATE_COLUMNS})
    elif n_columns > MAX_LONG_COLUMNS:
        raise WebError("upload_too_many_columns", params={"n": n_columns, "max": MAX_LONG_COLUMNS})

    columns: list[dict] = []
    if wide:
        for nm, s in series.items():
            share = round(float(s.notna().sum()) / len(s), 3) if len(s) else 0.0
            columns.append({"name": str(nm), "role": "value", "reason": None, "numeric_share": share})
    else:
        for col in frame.columns:
            c = str(col)
            share = _numeric_share(frame[col], table.decimal_comma)
            if c in date_cols:
                role, reason = "date", None
            elif c in value_cols:
                role, reason = "value", None
            elif c in excluded:
                role, reason = "excluded", "row_number"
            elif share >= _NUMERIC_SHARE:
                # numeric but not detected as a value column -- typically because no date column was found: the
                # user may pick it (it is not in detected.value_cols, so it starts unchecked), never "not numeric"
                role, reason = "value", None
            else:
                role, reason = "ignored", "not_numeric"
            columns.append({"name": c, "role": role, "reason": reason, "numeric_share": share})

    freq, first_date, last_date = _detected_dates(first) if first is not None else (None, None, None)
    label = attrs.get("date_format") if first is not None else None
    detected = {
        "wide": bool(wide),
        "date_col": date_cols[0] if date_cols else None,
        "date_cols": date_cols,
        "date_format": label or None,
        "date_format_label": _format_label(label, freq, hl),
        "value_cols": value_cols,
        "freq": freq,
        "freq_label": (_FREQ_LABELS[freq][1 if hl == "en" else 0] if freq in _FREQ_LABELS else None),
        "n_rows": int(len(frame) if first is None else (len(first) if wide else attrs.get("n_rows", len(frame)))),
        "first_date": first_date,
        "last_date": last_date,
    }
    notices = [viewmodel.notice_from_warning(w, hl, series_index=None) for w in [*read_notes, *detect_notes]]
    gt = table.google_trends
    google_trends = None
    if gt is not None:
        google_trends = {"terms": [{"col": str(col), "term": str(term), "region": region}
                                   for col, (term, region) in gt.terms.items()],
                         "lt1_count": int(gt.lt1_count), "category": gt.category}
    try:
        est = int(frame.memory_usage(deep=True).sum())
    except Exception:  # noqa: BLE001 - an estimate only
        est = len(raw)
    if first is None:
        n_points = 0
    elif wide:
        n_points = int(len(first))
    else:
        idx = first.index
        n_points = int(idx.nunique()) if hasattr(idx, "nunique") else int(len(first))
    entry = UploadEntry(raw=table, created_at=created_at, filename=name, size_bytes=len(raw), est_bytes=est,
                        wide=bool(wide), value_cols=value_cols, date_cols=date_cols, client_key=client_key,
                        n_points=n_points, freq=freq)
    body = {
        "filename": name,
        "size_bytes": len(raw),
        "format": table.format,
        "encoding": table.encoding,
        "kind": "google_trends" if gt is not None else "table",
        "columns": columns,
        "detected": detected,
        "problem": error_object(problem, request_id) if problem is not None else None,
        "preview": _preview(frame),
        "notices": notices,
        "google_trends": google_trends,
    }
    return entry, body
