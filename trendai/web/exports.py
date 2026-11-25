"""Export files (spec C.10, E.15): job-time builders for the stored JSON/HTML exports, and the serializers that turn
stored VMs / stored export JSON into downloadable files.

Owner: **WP1-vm**. Callers: ``analyze.py`` (job-time builders, stage ``series``/``finish``) and the export routes of
``api.py`` (serializers), both WP1-server.

Division of labour with ``api.py``:

* ``api.py`` validates ids/``k``/``fmt``, loads the stored data (``vm-<hl>.json``, ``export-<hl>.json``,
  ``report-<k>.html``/``compare.html``, ``result.json``; or the share snapshot), checks the stored ``available`` flags
  first (D.2.10, D.5: Naver -> ``403 export_naver_raw``, no openpyxl -> ``501 export_xlsx_unavailable``,
  ``values.csv`` of a single-series result -> ``404 not_found``), applies the ``export`` rate limit, then calls one
  serializer here and sends ``ExportFile.body`` with ``ExportFile.headers()`` plus the common headers (E.12;
  for HTML the sandbox CSP in ``headers()`` **replaces** the app CSP).
* The serializers re-check the same rules defensively and raise :class:`ExportUnavailable` (``api.py`` maps
  ``exc.code`` / ``exc.status`` to its ``WebError``).
* ``file_id`` is the validated result id or share id; only its first 8 characters reach a filename.

File rules (C.10, E.15): CSV is UTF-8 with BOM, CRLF, RFC 4180 quoting, two ``# `` comment lines (source and
conditions), numbers as plain decimals of up to 15 significant digits; every text cell that starts with ``=``,
``+``, ``-``, ``@``, tab or CR gets a leading apostrophe. XLSX text cells are written with ``data_type = "s"``
(no apostrophe). Naver files never carry raw ratios (no ``실제값`` column, no ``values.csv``, no HTML report).
"""

from __future__ import annotations

import copy
import csv
import functools
import html as _html
import io
import json
import re
import unicodedata
from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence
from urllib.parse import quote

from . import texts as T

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd

    from ..types import TrendReport

__all__ = [
    "CONTENT_TYPES", "HTML_EXPORT_CSP",
    "ExportFile", "ExportUnavailable",
    # job-time builders
    "build_series_json", "strip_naver_raw", "build_result_json", "build_export_store",
    "insert_source_paragraph", "render_series_html", "render_result_html", "ranking_rows",
    # serializers (request time)
    "xlsx_available", "series_csv", "series_xlsx", "result_csv", "result_xlsx", "values_csv",
    "html_export", "json_export",
    # helpers
    "export_filenames", "content_disposition", "sheet_names", "guard_text",
]

#: MIME types per format (C.10).
CONTENT_TYPES: dict[str, str] = {
    "csv": "text/csv; charset=utf-8",
    "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "html": "text/html; charset=utf-8",
    "json": "application/json; charset=utf-8",
}

#: Content-Security-Policy sent with HTML exports instead of the app CSP (C.10, E.12).
HTML_EXPORT_CSP: str = ("sandbox; default-src 'none'; style-src 'unsafe-inline' https://fonts.googleapis.com; "
                        "font-src https://fonts.gstatic.com; img-src data:")

#: first characters that make a spreadsheet treat a text cell as a formula (C.10)
_FORMULA_STARTS = ("=", "+", "-", "@", "\t", "\r")
#: characters a filename / sheet name must not contain
_FILENAME_BAD = re.compile(r'[\x00-\x1f\x7f/\\:*?"<>|]')
_SHEET_BAD = re.compile(r"[\[\]:*?/\\]")
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_NAME_MAX = 80
_SHEET_MAX = 31
_BODY_RX = re.compile(r"<body\b[^>]*>", re.IGNORECASE)


@dataclass(frozen=True)
class ExportFile:
    """One downloadable file.

    Attributes:
        body: the file bytes (CSV: UTF-8 with BOM and CRLF; XLSX: the workbook; HTML/JSON: UTF-8).
        fmt: ``"csv"`` | ``"xlsx"`` | ``"html"`` | ``"json"``.
        filename: ASCII fallback name ``trendai-<id8>-<k|all>.<fmt>`` (``values.csv`` -> ``…-values.csv``).
        filename_utf8: display name ``trendai-<display name>.<fmt>`` for ``filename*`` (C.10).
    """

    body: bytes
    fmt: str
    filename: str
    filename_utf8: str

    @property
    def content_type(self) -> str:
        """``CONTENT_TYPES[self.fmt]``."""
        return CONTENT_TYPES[self.fmt]

    def headers(self) -> list[tuple[str, str]]:
        """The export-specific response headers (C.10): ``Content-Type``, ``Content-Disposition``
        (:func:`content_disposition`), ``Cache-Control: private, no-store``, ``X-Content-Type-Options: nosniff`` and,
        for HTML, ``Content-Security-Policy: HTML_EXPORT_CSP``. ``Content-Length`` and the E.12 headers are added by
        the server."""
        out = [("Content-Type", self.content_type),
               ("Content-Disposition", content_disposition(self.filename, self.filename_utf8)),
               ("Cache-Control", "private, no-store"),
               ("X-Content-Type-Options", "nosniff")]
        if self.fmt == "html":
            out.append(("Content-Security-Policy", HTML_EXPORT_CSP))
        return out


class ExportUnavailable(Exception):
    """An export that cannot be produced for this data. ``code`` is the web error code (``export_naver_raw``,
    ``export_xlsx_unavailable``, ``not_found``) and ``status`` its HTTP status (403, 501, 404)."""

    def __init__(self, code: str, status: int) -> None:
        super().__init__(code)
        self.code = code
        self.status = status


# ---------------------------------------------------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------------------------------------------------

def _lang(vm_or_lang: Any) -> str:
    lang = vm_or_lang.get("lang") if isinstance(vm_or_lang, Mapping) else vm_or_lang
    return "en" if lang == "en" else "ko"


def _is_naver(vm: Mapping[str, Any]) -> bool:
    return (vm.get("source") or {}).get("kind") == "naver"


def _csv_number(v: Any) -> str:
    """A plain decimal of up to 15 significant digits (``format(v, ".15g")`` written without an exponent);
    ``""`` for ``None`` / non-finite values."""
    if v is None or isinstance(v, bool):
        return ""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return ""
    if f != f or f in (float("inf"), float("-inf")):
        return ""
    s = format(f, ".15g")
    if "e" in s or "E" in s:
        s = format(Decimal(s), "f")
    return "0" if s in ("-0", "-0.0") else s


def _xlsx_number(v: Any) -> Optional[float]:
    if v is None or isinstance(v, bool):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f == f and f not in (float("inf"), float("-inf")) else None


def guard_text(value: str) -> str:
    """CSV formula-injection guard (C.10): prefix ``'`` when the text starts with ``=``, ``+``, ``-``, ``@``, tab or
    CR. Text cells only -- numbers are never passed here; XLSX uses ``data_type = "s"`` instead."""
    text = "" if value is None else str(value)
    return "'" + text if text.startswith(_FORMULA_STARTS) else text


def _comment(text: str) -> str:
    """One ``# `` comment line, written as is (not quoted, so ``pandas.read_csv(comment="#")`` skips it). A
    spreadsheet still splits it at commas, so every piece after a comma gets the formula guard too."""
    flat = re.sub(r"[\r\n]+", " ", str(text))
    pieces = flat.split(",")
    return "# " + ",".join([pieces[0]] + [guard_text(p) for p in pieces[1:]])


def _write_csv(comments: Sequence[str], header: Sequence[str], rows: Sequence[Sequence[str]]) -> bytes:
    buf = io.StringIO()
    for line in comments:
        buf.write(_comment(line) + "\r\n")
    writer = csv.writer(buf, lineterminator="\r\n")
    writer.writerow([guard_text(h) for h in header])
    writer.writerows(rows)
    return buf.getvalue().encode("utf-8-sig")


def _display_name(name: Any) -> str:
    text = unicodedata.normalize("NFC", str(name or ""))
    text = _FILENAME_BAD.sub("_", text).strip()
    return text[:_NAME_MAX] or "trendai"


def export_filenames(file_id: str, k: "int | str | None", display_name: str, fmt: str) -> tuple[str, str]:
    """``(ascii_name, utf8_name)`` = (``trendai-<file_id[:8]>-<k|all>.<fmt>``, ``trendai-<display name>.<fmt>``).
    ``k`` may be ``"values"`` for ``values.csv`` (the display name then ends in ``-values``). The display name is NFC,
    control characters and ``/\\:*?"<>|`` replaced by ``_``, cut to 80 characters. A compare result's display name is
    the series names joined with ``" vs "`` (the same cut applies)."""
    id8 = re.sub(r"[^A-Za-z0-9_-]", "_", str(file_id or ""))[:8] or "export"
    part = "all" if k is None else str(k)
    shown = _display_name(display_name)
    if part == "values":
        shown = shown[:_NAME_MAX - len("-values")] + "-values"
    return f"trendai-{id8}-{part}.{fmt}", f"trendai-{shown}.{fmt}"


def content_disposition(filename: str, filename_utf8: str) -> str:
    """``attachment; filename="<ascii>"; filename*=UTF-8''<percent-encoded utf8 name>`` (RFC 6266 / 5987)."""
    ascii_name = re.sub(r'[^\x20-\x7e]|["\\]', "_", str(filename))
    return f"attachment; filename=\"{ascii_name}\"; filename*=UTF-8''{quote(str(filename_utf8), safe='')}"


def _result_name(vms: Sequence[Mapping[str, Any]], lang: str) -> str:
    return T.t(T.COMPARE_TEXTS, "names_sep", lang).join(str(v.get("name", "")) for v in vms)


def _file(body: bytes, fmt: str, file_id: str, k: "int | str | None", name: str) -> ExportFile:
    ascii_name, utf8_name = export_filenames(file_id, k, name, fmt)
    return ExportFile(body=body, fmt=fmt, filename=ascii_name, filename_utf8=utf8_name)


# ---------------------------------------------------------------------------------------------------------------------
# Condition texts (CSV comment line, XLSX rows)
# ---------------------------------------------------------------------------------------------------------------------

def _series_item(vm: Mapping[str, Any], lang: str) -> str:
    """``검색어 그룹 선크림(선크림, 자외선차단제)`` / ``문서 …`` / ``열 …`` / ``샘플 …``."""
    kind = (vm.get("source") or {}).get("kind")
    name = str(vm.get("name", ""))
    terms = [str(t) for t in vm.get("terms") or []]
    if kind == "naver":
        if len(terms) > 1 or (terms and terms[0] != name):
            return T.t(T.EXPORT_TEXTS, "group", lang, name=name,
                       terms=T.t(T.EXPORT_TEXTS, "terms_sep", lang).join(terms))
        return T.t(T.EXPORT_TEXTS, "group_single", lang, name=name)
    if kind == "wiki":
        return T.t(T.EXPORT_TEXTS, "article", lang, name=name)
    if kind == "sample":
        return T.t(T.EXPORT_TEXTS, "sample", lang, name=name)
    return T.t(T.EXPORT_TEXTS, "column", lang, name=name)


def _filter_parts(vm: Mapping[str, Any], lang: str) -> list[str]:
    """The Naver filters as separate parts (``모바일``, ``여성``, ``25–34세``)."""
    from .viewmodel import _age_label

    src = vm.get("source") or {}
    filters = src.get("filters") or {}
    if src.get("kind") != "naver" or not filters:
        return []
    parts = []
    for key in ("device", "gender"):
        value = filters.get(key)
        if value in T.FILTER_LABELS:
            parts.append(T.t(T.FILTER_LABELS, value, lang))
    ages = _age_label(filters.get("ages") or [], lang)
    if ages:
        parts.append(ages)
    return parts


def _condition_line(vms: Sequence[Mapping[str, Any]], lang: str) -> str:
    """``조건: 검색어 그룹 선크림(선크림, 자외선차단제) · 모바일 · 여성 · 25–34세 · 예측 26주 · <Naver note>``."""
    if not vms:
        return T.t(T.EXPORT_TEXTS, "conditions", lang) + ":"
    first = vms[0]
    parts = [_series_item(v, lang) for v in vms]
    parts += _filter_parts(first, lang)
    horizons = {str((v.get("future") or {}).get("horizon_label") or "") for v in vms}
    horizons.discard("")
    if len(horizons) == 1:
        parts.append(T.t(T.EXPORT_TEXTS, "horizon", lang, span=horizons.pop()))
    if any(_is_naver(v) for v in vms):
        parts.append(T.t(T.EXPORT_TEXTS, "naver_note", lang))
    return T.t(T.EXPORT_TEXTS, "comment_line", lang, label=T.t(T.EXPORT_TEXTS, "conditions", lang),
               text=" · ".join(p for p in parts if p))


def _series_source_line(vm: Mapping[str, Any]) -> str:
    from .viewmodel import source_line

    return source_line(vm.get("source") or {}, _lang(vm), with_filters=False)


def _xlsx_conditions(vm: Mapping[str, Any], lang: str) -> list[tuple[str, str]]:
    src = vm.get("source") or {}
    labels = T.EXPORT_COLUMNS["xlsx_conditions"][1 if lang == "en" else 0]
    value_kind = str(src.get("value_kind_label") or "")
    if _is_naver(vm):
        value_kind = f"{value_kind} · {T.t(T.EXPORT_TEXTS, 'naver_note', lang)}"
    filters = " · ".join(_filter_parts(vm, lang)) or T.t(T.EXPORT_TEXTS, "no_filters", lang)
    values = [_series_source_line(vm), _series_item(vm, lang), str(src.get("range_label") or ""),
              str(src.get("freq_label") or ""), filters, str(src.get("as_of_label") or ""), value_kind]
    return list(zip(labels, values, strict=True))


# ---------------------------------------------------------------------------------------------------------------------
# Job-time builders (called by analyze.py; results are stored, E.7)
# ---------------------------------------------------------------------------------------------------------------------

def build_series_json(report: TrendReport, source_block: Mapping[str, Any], lang: str, *,
                      narrative: Optional[str] = None) -> dict:
    """The per-series JSON export in ``lang`` (C.10): ``pipeline.report_to_dict(report, lang=lang)`` plus a top-level
    ``"source"`` = ``source_block`` (the VM's ``source`` object in ``lang``, ``viewmodel.build_source_block``). For a
    Naver series :func:`strip_naver_raw` is applied. ``narrative`` (the narrative in ``lang``) replaces
    ``report_to_dict``'s ``narrative``, which is always the report's own Korean text; ``None`` keeps it."""
    from ..pipeline import report_to_dict

    body = report_to_dict(report, lang=_lang(lang))
    if narrative is not None:
        body["narrative"] = str(narrative)
    out: dict[str, Any] = {"source": copy.deepcopy(dict(source_block)), **body}
    if source_block.get("kind") == "naver":
        out = strip_naver_raw(out)
    return out


def strip_naver_raw(export: Mapping[str, Any]) -> dict:
    """Return a copy without raw Naver ratios (C.10, A.3): ``history`` removed; ``metrics.last_value``,
    ``metrics.peak_value``, ``metrics.mean`` and every ``anomalies[].value`` removed; ``"raw_excluded": true`` added.
    Model outputs (``trend``, ``forecast``, ``scenarios``, ...) stay."""
    out = copy.deepcopy(dict(export))
    out.pop("history", None)
    metrics = out.get("metrics")
    if isinstance(metrics, dict):
        for key in ("last_value", "peak_value", "mean"):
            metrics.pop(key, None)
    anomalies = out.get("anomalies")
    if isinstance(anomalies, list):
        for an in anomalies:
            if isinstance(an, dict):
                an.pop("value", None)
    out["raw_excluded"] = True
    return out


def build_result_json(series_json: Sequence[Mapping[str, Any]], *, source_line: Optional[str] = None) -> Any:
    """The whole-result JSON export (C.10) from the per-series exports of the successful series (request order).

    Single-series result (``source_line is None``): the one series object itself. Compare result:
    ``{"source_line": source_line, "source": [each series' "source" object], "reports": [the series objects]}``
    -- an object rather than a bare array so the spec's top-level ``source`` array and ``source_line`` have a place
    (C.10 wording; see ``docs/WEB_INTERFACES.md``)."""
    items = [copy.deepcopy(dict(s)) for s in series_json]
    if source_line is None:
        return items[0] if items else {}
    return {"source_line": str(source_line), "source": [s.get("source") for s in items], "reports": items}


def build_export_store(series_json: Mapping[int, Mapping[str, Any]], result_json: Any) -> dict:
    """The content of ``results/<id>/export-<lang>.json`` (E.7): ``{"series": {"<k>": series export, ...},
    "result": whole-result export}`` (string keys; failed series absent). :func:`json_export` reads exactly this."""
    return {"series": {str(int(k)): copy.deepcopy(dict(v)) for k, v in sorted(series_json.items())},
            "result": copy.deepcopy(result_json)}


def insert_source_paragraph(html: str, source_line_ko: str) -> str:
    """Insert ``<p class="source">{html.escape(source_line_ko)}</p>`` right after the first ``<body>`` tag of a
    rendered report (C.10): the result reads ``<body><p class="source">출처: …</p>`` followed by what the renderer
    emitted after ``<body>``. Returns ``html`` unchanged if it has no ``<body>``."""
    m = _BODY_RX.search(html)
    if m is None:
        return html
    para = f'<p class="source">{_html.escape(str(source_line_ko))}</p>'
    return html[:m.end()] + para + html[m.end():]


def render_series_html(report: TrendReport, source_line_ko: str) -> str:
    """``report.render_html(report)`` + :func:`insert_source_paragraph` with the series' Korean ``source.line``
    (non-Naver only; Korean only). Stored as ``report-<k>.html``."""
    from ..report import render_html

    return insert_source_paragraph(render_html(report), source_line_ko)


def render_result_html(reports: Sequence[TrendReport], ranking: pd.DataFrame, source_line_ko: str) -> str:
    """``report.render_compare_html(list(reports), ranking)`` + :func:`insert_source_paragraph` with the Korean
    ``compare.source_line`` (non-Naver, 2+ successful series). Stored as ``compare.html``."""
    from ..report import render_compare_html

    return insert_source_paragraph(render_compare_html(list(reports), ranking), source_line_ko)


def _json_number(v: Any) -> Optional[float]:
    f = _xlsx_number(v)
    return None if f is None else float(f"{f:.15g}")


def ranking_rows(ranking: pd.DataFrame, name_to_index: Mapping[str, int]) -> list[dict]:
    """JSON-safe ``rank_trends`` rows for ``result.json["ranking_rows"]`` (E.7), best first, each with its series
    ``index`` (via ``name_to_index``) and the raw numbers the ranking CSV prints (rank, name, phase,
    expected/recent change %, prob_increase, momentum, score, confidence). Read back by :func:`result_csv`."""
    rows = []
    for rank, rec in ranking.iterrows():
        name = str(rec["name"])
        rows.append({
            "rank": int(rank),
            "index": int(name_to_index[name]),
            "name": name,
            "phase": str(rec["phase"]),
            "expected_change_pct": _json_number(rec.get("expected_change_pct")),
            "prob_increase": _json_number(rec.get("prob_increase")),
            "recent_change_pct": _json_number(rec.get("recent_change_pct")),
            "momentum": _json_number(rec.get("momentum")),
            "score": _json_number(rec.get("score")),
            "confidence": str(rec.get("confidence")),
            "freq": str(rec.get("freq")),
        })
    return rows


# ---------------------------------------------------------------------------------------------------------------------
# Serializers (called by api.py per request). ``vms``/``compare`` are the stored VMs of one language (``hl``);
# the export language is the VM's ``lang``. HTML is always Korean.
# ---------------------------------------------------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def _openpyxl_importable() -> bool:
    try:
        import openpyxl  # noqa: F401
    except Exception:  # noqa: BLE001 - any import failure means "not available"
        return False
    return True


def xlsx_available() -> bool:
    """Whether ``openpyxl`` can be imported (cached after the first call). Also feeds ``/api/config``
    ``features.xlsx_export`` and the stored availability flags."""
    return _openpyxl_importable()


def _series_table(vm: Mapping[str, Any]) -> tuple[list[str], list[list[Any]]]:
    """Header and raw rows (numbers or ``None``) of one series: history rows then forecast rows."""
    lang = _lang(vm)
    header = list(T.EXPORT_COLUMNS["series"][1 if lang == "en" else 0])
    naver = _is_naver(vm)
    chart = vm.get("chart") or {}
    fc = chart.get("forecast") or {}
    rows: list[list[Any]] = []
    actual = chart.get("actual") or []
    trend = chart.get("trend") or []
    for i, date in enumerate(chart.get("dates") or []):
        a = actual[i] if i < len(actual) else None
        tr = trend[i] if i < len(trend) else None
        rows.append([date, a, tr, None, None, None, None, None])
    cols = ("mean", "lo80", "hi80", "lo95", "hi95")
    bands = [list(fc.get(c) or []) for c in cols]
    for i, date in enumerate(fc.get("dates") or []):
        rows.append([date, None, None, *[arr[i] if i < len(arr) else None for arr in bands]])
    if naver:
        header.pop(1)
        rows = [[r[0], *r[2:]] for r in rows]
    return header, rows


def series_csv(vm: Mapping[str, Any], *, file_id: str) -> ExportFile:
    """One series as CSV (C.10, E.15) from its stored series VM: two ``# `` comment lines (source line without
    filters, ``viewmodel.source_line(..., with_filters=False)``; condition line with terms, filters, horizon and, for
    Naver, the ``export.naver_note``), the header row and history rows (``실제값``, ``추세``) then forecast rows
    (``예측값`` and the bands). Naver: no ``실제값``/``actual`` column. Numbers ``format(v, ".15g")``,
    ``None`` -> empty, :func:`guard_text` on every text cell. Name ``k`` = ``vm["index"]``."""
    return _series_csv(vm, file_id=file_id, k=int(vm.get("index", 0)))


def _series_csv(vm: Mapping[str, Any], *, file_id: str, k: "int | str | None") -> ExportFile:
    lang = _lang(vm)
    header, rows = _series_table(vm)
    body = _write_csv([_series_source_line(vm), _condition_line([vm], lang)], header,
                      [[guard_text(r[0])] + [_csv_number(v) for v in r[1:]] for r in rows])
    return _file(body, "csv", file_id, k, str(vm.get("name", "")))


def _ranking_texts(vm: Optional[Mapping[str, Any]], lang: str, row: Mapping[str, Any]) -> tuple[str, str, str]:
    """(phase, outlook word, reliability word) of a ranking row, from the series VM in the export language."""
    if vm is not None:
        phase = str((vm.get("phase") or {}).get("name") or row.get("phase") or "")
        word = str(((vm.get("hero") or {}).get("outlook") or {}).get("word") or "")
        level = str(((vm.get("hero") or {}).get("reliability") or {}).get("level") or row.get("confidence") or "")
    else:
        label = str(row.get("phase") or "")
        phase = T.t(T.PHASE_SHORT, label, lang) if label in T.PHASE_SHORT else label
        word, level = "", str(row.get("confidence") or "")
    conf = T.t(T.EXPORT_TEXTS, f"conf_{level}", lang) if f"conf_{level}" in T.EXPORT_TEXTS else level
    return phase, word, conf


def result_csv(vms: Sequence[Mapping[str, Any]], compare: Optional[Mapping[str, Any]],
               rows: Optional[Sequence[Mapping[str, Any]]], *, file_id: str) -> ExportFile:
    """Whole result as CSV (C.10). Single-series result (``compare is None``): exactly :func:`series_csv` of
    ``vms[0]`` but named ``…-all.csv``. Compare: the ranking table from ``rows`` (:func:`ranking_rows`, stored in
    ``result.json``) with ``compare["source_line"]`` and the condition line of every series as comment lines."""
    if compare is None:
        if not vms:
            raise ExportUnavailable("not_found", 404)
        return _series_csv(vms[0], file_id=file_id, k=None)
    lang = _lang(vms[0]) if vms else "ko"
    by_index = {int(v.get("index", -1)): v for v in vms}
    header = list(T.EXPORT_COLUMNS["ranking"][1 if lang == "en" else 0])
    table = []
    for row in sorted(rows or [], key=lambda r: int(r.get("rank", 0))):
        phase, word, conf = _ranking_texts(by_index.get(int(row.get("index", -1))), lang, row)
        prob = row.get("prob_increase")
        table.append([
            _csv_number(row.get("rank")), guard_text(str(row.get("name", ""))), guard_text(phase),
            guard_text(word), _csv_number(row.get("expected_change_pct")),
            _csv_number(None if prob is None else float(prob) * 100.0),
            _csv_number(row.get("recent_change_pct")), _csv_number(row.get("momentum")),
            _csv_number(row.get("score")), guard_text(conf)])
    body = _write_csv([str(compare.get("source_line") or ""), _condition_line(vms, lang)], header, table)
    return _file(body, "csv", file_id, None, _result_name(vms, lang))


def values_csv(vms: Sequence[Mapping[str, Any]], compare: Optional[Mapping[str, Any]], *,
               file_id: str) -> ExportFile:
    """The compare values table (C.10, F.6.8): ``날짜``/``date`` + one column per successful series (display names,
    request order), history values on the union of dates (empty where absent), with the two comment lines of the
    whole-result CSV. Raises :class:`ExportUnavailable`: ``not_found`` (404) when ``compare is None``,
    ``export_naver_raw`` (403) for Naver."""
    if compare is None or not vms:
        raise ExportUnavailable("not_found", 404)
    if any(_is_naver(v) for v in vms):
        raise ExportUnavailable("export_naver_raw", 403)
    lang = _lang(vms[0])
    ordered = sorted(vms, key=lambda v: int(v.get("index", 0)))
    columns: list[dict[str, Any]] = []
    dates: set[str] = set()
    for vm in ordered:
        chart = vm.get("chart") or {}
        mapping = dict(zip(chart.get("dates") or [], chart.get("actual") or [], strict=False))
        columns.append(mapping)
        dates.update(mapping)
    header = list(T.EXPORT_COLUMNS["values"][1 if lang == "en" else 0]) + [str(v.get("name", "")) for v in ordered]
    rows = [[guard_text(d)] + [_csv_number(col.get(d)) for col in columns] for d in sorted(dates)]
    body = _write_csv([str(compare.get("source_line") or ""), _condition_line(ordered, lang)], header, rows)
    return _file(body, "csv", file_id, "values", _result_name(ordered, lang))


def sheet_names(names: Sequence[str], lang: str) -> list[str]:
    """XLSX data-sheet names for series display names (C.10): cut to 31 characters, ``[]:*?/\\`` replaced by ``_``,
    de-duplicated with `` (2)``, `` (3)`` … (also against the summary sheet name of ``lang``), request order."""
    taken = {T.t(T.EXPORT_TEXTS, "sheet_summary", lang).casefold()}
    out = []
    for raw in names:
        base = _SHEET_BAD.sub("_", _CONTROL.sub("", str(raw or ""))).strip()
        base = re.sub(r"^'|'$", "_", base) or T.t(T.EXPORT_TEXTS, "sheet_data", lang)
        name = base[:_SHEET_MAX]
        n = 2
        while name.casefold() in taken:
            suffix = f" ({n})"
            name = base[:_SHEET_MAX - len(suffix)] + suffix
            n += 1
        taken.add(name.casefold())
        out.append(name)
    return out


def _xlsx_text(cell: Any, text: Any) -> None:
    """A text cell that is always a string (never a formula, no visible apostrophe; C.10)."""
    try:
        from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE
        value = ILLEGAL_CHARACTERS_RE.sub("", "" if text is None else str(text))
    except ImportError:  # pragma: no cover - openpyxl always has it
        value = _CONTROL.sub("", "" if text is None else str(text))
    cell.value = value
    cell.data_type = "s"


def _xlsx_data_sheet(ws: Any, vm: Mapping[str, Any]) -> None:
    lang = _lang(vm)
    for r, (label, value) in enumerate(_xlsx_conditions(vm, lang), start=1):
        _xlsx_text(ws.cell(row=r, column=1), label)
        _xlsx_text(ws.cell(row=r, column=2), value)
    header, rows = _series_table(vm)
    for c, h in enumerate(header, start=1):
        _xlsx_text(ws.cell(row=9, column=c), h)
    index_kind = (vm.get("source") or {}).get("value_kind") in ("relative_index", "relative_index_gt")
    for r, row in enumerate(rows, start=10):
        _xlsx_text(ws.cell(row=r, column=1), row[0])
        for c, v in enumerate(row[1:], start=2):
            num = _xlsx_number(v)
            if num is None:
                continue
            cell = ws.cell(row=r, column=c)
            cell.value = num
            if index_kind:
                cell.number_format = "0.0"
    ws.column_dimensions["A"].width = 26
    ws.column_dimensions["B"].width = 18


def _summary_row(vm: Mapping[str, Any], score_text: str) -> list[str]:
    hero = vm.get("hero") or {}
    tiles = {t.get("key"): t for t in vm.get("tiles") or []}
    return [str(vm.get("name", "")), str((vm.get("phase") or {}).get("name") or ""),
            str((hero.get("outlook") or {}).get("word") or ""),
            str((tiles.get("expected_change") or {}).get("value") or ""),
            str((tiles.get("prob_increase") or {}).get("value") or ""),
            str((hero.get("reliability") or {}).get("label") or ""), str(hero.get("accuracy_line") or ""), score_text]


def _workbook(vms: Sequence[Mapping[str, Any]], compare: Optional[Mapping[str, Any]]) -> bytes:
    if not xlsx_available():
        raise ExportUnavailable("export_xlsx_unavailable", 501)
    from openpyxl import Workbook

    lang = _lang(vms[0]) if vms else "ko"
    wb = Workbook()
    first = wb.active
    names = sheet_names([str(v.get("name", "")) for v in vms], lang)
    for i, (vm, name) in enumerate(zip(vms, names, strict=True)):
        ws = first if i == 0 else wb.create_sheet()
        ws.title = name
        _xlsx_data_sheet(ws, vm)
    summary = wb.create_sheet() if vms else first
    summary.title = T.t(T.EXPORT_TEXTS, "sheet_summary", lang)
    for c, h in enumerate(T.EXPORT_COLUMNS["summary_sheet"][1 if lang == "en" else 0], start=1):
        _xlsx_text(summary.cell(row=1, column=c), h)
    scores = {int(r.get("index", -1)): str((r.get("score") or {}).get("text") or "")
              for r in (compare or {}).get("ranking") or []}
    for r, vm in enumerate(vms, start=2):
        row = _summary_row(vm, scores.get(int(vm.get("index", -1)), "—"))
        for c, text in enumerate(row, start=1):
            _xlsx_text(summary.cell(row=r, column=c), text)
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def series_xlsx(vm: Mapping[str, Any], *, file_id: str) -> ExportFile:
    """One series as XLSX: one data sheet (rows 1-7 conditions, row 8 empty, row 9 header, data from row 10) and the
    ``요약``/``Summary`` sheet with its single row (C.10). Raises :class:`ExportUnavailable`
    (``export_xlsx_unavailable``, 501) without openpyxl."""
    body = _workbook([vm], None)
    return _file(body, "xlsx", file_id, int(vm.get("index", 0)), str(vm.get("name", "")))


def result_xlsx(vms: Sequence[Mapping[str, Any]], compare: Optional[Mapping[str, Any]], *,
                file_id: str) -> ExportFile:
    """Whole result as XLSX (C.10): one data sheet per successful series (names :func:`sheet_names`) and the
    summary sheet (one row per series, pre-formatted VM texts). Raises :class:`ExportUnavailable` without
    openpyxl."""
    if not vms:
        raise ExportUnavailable("not_found", 404)
    ordered = sorted(vms, key=lambda v: int(v.get("index", 0)))
    body = _workbook(ordered, compare)
    return _file(body, "xlsx", file_id, None, _result_name(ordered, _lang(ordered[0])))


def html_export(html: str, *, file_id: str, k: Optional[int], display_name: str) -> ExportFile:
    """Wrap a stored pre-rendered report (``report-<k>.html`` or ``compare.html``, already carrying its source
    paragraph) as an ``ExportFile``; ``k`` ``None`` -> ``…-all.html``."""
    return _file(str(html).encode("utf-8"), "html", file_id, k, display_name)


def json_export(store: Mapping[str, Any], *, file_id: str, k: Optional[int], display_name: str) -> ExportFile:
    """Serialize ``store["series"][str(k)]`` (``k`` given) or ``store["result"]`` (``k`` ``None``) from a stored
    :func:`build_export_store` document with ``json.dumps(..., ensure_ascii=False, allow_nan=False, indent=2)``.
    Raises :class:`ExportUnavailable` (``not_found``, 404) for a missing series."""
    if k is None:
        doc = store.get("result")
    else:
        doc = (store.get("series") or {}).get(str(int(k)))
    if doc is None:
        raise ExportUnavailable("not_found", 404)
    body = json.dumps(doc, ensure_ascii=False, allow_nan=False, indent=2).encode("utf-8")
    return _file(body, "json", file_id, k, display_name)
