"""``POST /api/upload`` internals (spec C.4, E.17): multipart decoding, the zip guard, in-memory parsing and column
detection, the preview, and the in-memory upload cache.

Nothing here writes to disk: the raw bytes are parsed in memory (``data.read_table_bytes``) and only the parsed
``RawTable`` is kept, for 30 minutes, in :class:`UploadCache` (the ``bytes`` object is dropped). Notices are captured
with ``data.collect_notices()`` (never through ``warnings``, E.13).
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

__all__ = ["parse_multipart", "display_filename", "zip_guard", "UploadEntry", "UploadCache", "analyse_upload",
           "stringify_cell", "MAX_TABLE_CELLS", "MAX_LONG_COLUMNS", "MAX_WIDE_ROWS", "MAX_WIDE_DATE_COLUMNS",
           "MAX_HEADER_COLUMNS", "UPLOAD_TTL_S"]

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
_PREVIEW_ROWS = 5
_CELL_MAX = 40
_FILENAME_MAX = 100
_CONTROL = re.compile(r"[\x00-\x1f\x7f-\x9f]")
_FREQ_LABELS = {"D": ("일간", "Daily"), "W": ("주간", "Weekly"), "M": ("월간", "Monthly"), "Q": ("분기", "Quarterly"),
                "Y": ("연간", "Yearly")}
_FORMAT_UNITS = {"D": ("일", "daily"), "W": ("주", "weekly"), "M": ("월", "monthly"), "Q": ("분기", "quarterly"),
                 "Y": ("연", "yearly")}


# --------------------------------------------------------------------------- request decoding

def parse_multipart(content_type: str, body: bytes) -> tuple[str, bytes]:
    """The first part with ``name="file"`` and a filename -> ``(filename, bytes)`` (E.17). Python 3.13 has no
    ``cgi``: the body is parsed with ``email.parser.BytesParser(policy=email.policy.HTTP)``. A missing boundary or
    part -> ``400 bad_request``."""
    if "boundary=" not in content_type.lower():
        raise WebError("bad_request")
    try:
        header = content_type.encode("latin-1")
    except UnicodeEncodeError:
        raise WebError("bad_request") from None
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
    (``413 upload_archive_too_large``); a broken archive is ``422 upload_unreadable``."""
    if not raw.startswith(b"PK\x03\x04"):
        return
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            infos = zf.infolist()
    except (zipfile.BadZipFile, zipfile.LargeZipFile, OSError, ValueError):
        raise WebError("upload_unreadable", params={"reason": "excel_unreadable"}) from None
    if len(infos) > ZIP_MAX_MEMBERS or sum(max(0, i.file_size) for i in infos) > ZIP_MAX_UNCOMPRESSED:
        raise WebError("upload_archive_too_large")


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


class UploadCache:
    """``{upload_id: UploadEntry}``: TTL 30 minutes, LRU-evicted above 64 entries or 256 MB (C.13, E.17)."""

    def __init__(self, clock: Callable[[], _dt.datetime], *, ttl_s: float = UPLOAD_TTL_S,
                 max_entries: int = CACHE_MAX_ENTRIES, max_bytes: int = CACHE_MAX_BYTES) -> None:
        self._clock = clock
        self._lock = threading.Lock()
        self._items: "OrderedDict[str, UploadEntry]" = OrderedDict()
        self.ttl_s = float(ttl_s)
        self.max_entries = int(max_entries)
        self.max_bytes = int(max_bytes)

    def _now(self) -> float:
        return self._clock().timestamp()

    def put(self, upload_id: str, entry: UploadEntry) -> None:
        with self._lock:
            self._items[upload_id] = entry
            self._items.move_to_end(upload_id)
            total = sum(e.est_bytes for e in self._items.values())
            while self._items and (len(self._items) > self.max_entries or total > self.max_bytes):
                _uid, old = self._items.popitem(last=False)
                total -= old.est_bytes

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

    def sweep(self) -> int:
        now = self._now()
        with self._lock:
            old = [k for k, e in self._items.items() if now - e.created_at > self.ttl_s]
            for k in old:
                del self._items[k]
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


def _iso(ts: Any) -> str:
    return f"{ts.year:04d}-{ts.month:02d}-{ts.day:02d}"


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
                   request_id: Optional[str] = None) -> tuple[UploadEntry, dict]:
    """Parse and detect one upload (E.17). Returns the cache entry and the C.4 body *without* ``upload_id`` /
    ``expires_at`` (the handler adds them). Raises :class:`WebError` for the C.4 error cases."""
    if len(raw) > max_bytes:
        raise WebError("body_too_large", params={"max_mb": max_mb})
    zip_guard(raw)
    name = display_filename(filename)
    with data.collect_notices() as read_notes:
        try:
            table = data.read_table_bytes(raw, name, max_columns=MAX_HEADER_COLUMNS)
        except data.DataError as exc:
            raise from_data_error(exc, "upload_parse") from None
    frame = table.frame
    cells = int(frame.size)
    if cells > MAX_TABLE_CELLS:
        raise WebError("upload_too_large_table", params={"n": cells, "max": MAX_TABLE_CELLS})

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
    entry = UploadEntry(raw=table, created_at=created_at, filename=name, size_bytes=len(raw), est_bytes=est,
                        wide=bool(wide), value_cols=value_cols, date_cols=date_cols)
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
