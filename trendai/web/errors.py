"""Web errors (spec C.12): the :class:`WebError` exception, the C.12.1 envelope and the mappings of connector
(C.12.3) and data (C.12.4) errors to web codes.

Texts come from ``messages.MESSAGES`` (WP2): ``render_web(code, params, lang)`` for web codes and
``render_web_msg(exc, lang)`` when a ``DataError`` / ``ConnectorError`` is shown as is (the ``params.reason``
cases). Fix objects use ``viewmodel.fix_object`` so notices and errors share one shape.
"""

from __future__ import annotations

import errno as _errno
import logging
from typing import Any, Mapping, Optional, Sequence
from urllib.parse import quote

from .. import messages

__all__ = ["WebError", "STATUS", "DEFAULT_FIXES", "error_object", "error_envelope", "from_connector_error",
           "from_data_error", "series_error", "wiki_search_url", "encode_series", "catalog_code", "store_error"]

LOG = logging.getLogger("trendai.web")

#: HTTP status of every web code (C.12.2). ``upload_no_*`` are only ever ``problem`` objects (status 200).
STATUS: dict[str, int] = {
    "bad_request": 400, "invalid_param": 400, "empty_query": 400, "too_many_series": 400, "too_many_keywords": 400,
    "term_too_long": 400, "wiki_group_unsupported": 400, "unknown_sample": 400, "body_too_large": 413,
    "length_required": 411, "unsupported_media_type": 415, "method_not_allowed": 405, "cross_origin": 403,
    "not_found": 404, "result_not_found": 404, "job_not_found": 404, "share_not_found": 404, "share_forbidden": 403,
    "share_not_needed": 400, "upload_expired": 410, "upload_unreadable": 422, "upload_too_many_columns": 422,
    "upload_too_large_table": 422, "upload_too_many_series": 400, "upload_archive_too_large": 413,
    "upload_excel_needs_pkg": 501, "upload_no_date_col": 200, "upload_no_value_col": 200,
    "upload_bad_date_col": 400, "upload_bad_value_col": 400, "data_too_short": 422, "series_failed_web": 422,
    "rate_limited": 429, "server_busy": 503, "internal_error": 500, "job_timeout": 504, "naver_unconfigured": 503,
    "naver_auth": 502, "naver_rate": 429, "naver_rate_limited": 429, "naver_budget_exhausted": 503,
    "naver_server": 502, "source_timeout": 504, "source_unreachable": 502, "source_bad_response": 502,
    "wiki_not_found": 404, "wiki_rate": 429, "wiki_empty": 404, "export_xlsx_unavailable": 501,
    "export_naver_raw": 403, "ai_unavailable": 503, "ai_naver_blocked": 403, "ai_readonly": 403, "ai_failed": 502,
    "ai_rate_limited": 429, "store_full": 503, "upload_evicted": 410,
}

#: Default fix actions per code (C.12.2), before the context rules of C.12.1.
DEFAULT_FIXES: dict[str, tuple[str, ...]] = {
    "bad_request": ("reload",), "invalid_param": ("go_home",), "empty_query": ("try_sample",),
    "too_many_series": ("remove_series",), "wiki_group_unsupported": ("use_naver",),
    "unknown_sample": ("try_sample",), "body_too_large": ("upload_file",), "length_required": ("reload",),
    "unsupported_media_type": ("reload",), "cross_origin": ("go_home",), "not_found": ("go_home",),
    "result_not_found": ("reanalyze", "go_home"), "job_not_found": ("reanalyze",), "share_not_found": ("go_home",),
    "share_not_needed": ("copy_link",), "upload_expired": ("reupload",),
    "upload_unreadable": ("show_formats", "try_paste"), "upload_too_many_columns": ("show_formats",),
    "upload_too_large_table": ("show_formats",), "upload_too_many_series": ("pick_columns",),
    "upload_archive_too_large": ("show_formats",), "upload_excel_needs_pkg": ("show_formats",),
    "upload_no_date_col": ("pick_columns",), "upload_no_value_col": ("pick_columns",),
    "upload_bad_date_col": ("pick_columns",), "upload_bad_value_col": ("pick_columns",),
    "data_too_short": ("extend_range", "try_sample"), "series_failed_web": ("reanalyze",),
    "rate_limited": ("retry_in",), "server_busy": ("retry_in",), "internal_error": ("retry", "go_home"),
    "job_timeout": ("reanalyze",), "naver_unconfigured": ("use_wiki", "upload_file"),
    "naver_auth": ("retry", "use_wiki"), "naver_rate": ("retry_in",), "naver_rate_limited": ("use_wiki", "retry_in"),
    "naver_budget_exhausted": ("use_wiki", "upload_file"), "naver_server": ("retry",), "source_timeout": ("retry",),
    "source_unreachable": ("retry",), "source_bad_response": ("retry",),
    "wiki_not_found": ("wiki_title_help", "use_naver"), "wiki_rate": ("retry_in",), "wiki_empty": ("extend_range",),
    "export_xlsx_unavailable": ("export_csv",), "export_naver_raw": ("export_csv",), "ai_failed": ("retry",),
    "ai_rate_limited": ("retry_in",),
    "store_full": ("retry_in",), "upload_evicted": ("reupload",),
}

#: Codes whose envelope carries a ``Retry-After`` / ``retry_after`` default (seconds).
_RETRY_DEFAULTS: dict[str, int] = {"server_busy": 10, "naver_rate": 30, "wiki_rate": 60, "store_full": 300}

_UPLOAD_KINDS = ("upload", "google_trends_upload")
_MAX_FIXES = 3
_RETRY_AFTER_MAX = 300
#: a text variant for a context, used when the catalog has it (``messages.MESSAGES``); else the base text is used
#: with the clause cut below. ``wiki_not_found`` ends "…or switch to Naver", which is wrong when Naver is unavailable.
_NO_NAVER_VARIANT = {"wiki_not_found": "wiki_not_found_no_naver"}
_NO_NAVER_CUT = {"wiki_not_found": {"ko": ("인지 확인하거나 네이버로 바꿔 보세요.", "인지 확인해 주세요."),
                                    "en": (" or switch to Naver.", ".")}}


def catalog_code(code: str, fallback: str) -> str:
    """``code`` when the message catalog has a text for it, else ``fallback`` (a code a later catalog adds can be
    raised now without breaking the C.12 rule that every emitted code has a text)."""
    return code if code in messages.MESSAGES or code in getattr(messages, "WEB_OVERRIDES", {}) else fallback


def store_error(exc: BaseException) -> Optional["WebError"]:
    """A full disk or quota while writing the store is not an internal error: ``store_full`` (or ``server_busy``
    until the catalog has it), ``503`` with ``Retry-After: 300``. ``None`` for anything else."""
    if isinstance(exc, OSError) and exc.errno in (_errno.ENOSPC, getattr(_errno, "EDQUOT", _errno.ENOSPC)):
        return WebError(catalog_code("store_full", "server_busy"), 503, retry_after=300)
    return None


class WebError(Exception):
    """An error answered with the C.12.1 envelope.

    Args:
        code: web code (C.12.2); ``status`` defaults to :data:`STATUS`.
        params: template parameters (only those listed for the code; ``reason`` where the spec allows it).
        retry_after: seconds for 429/503 codes (also sent as ``Retry-After``); defaults per code.
        detail: a ``DataError`` / ``ConnectorError`` whose web text replaces the code's text (``params.reason``
            cases, C.12.1).
        fix_params: params of specific fix actions, e.g. ``{"wiki_title_help": {"url": …}}``.
        source_kind: source kind of the result the error belongs to (fix context rules, C.12.1).
        headers: extra response headers.
    """

    def __init__(self, code: str, status: Optional[int] = None, *, params: Optional[Mapping[str, Any]] = None,
                 retry_after: Optional[int] = None, detail: Optional[BaseException] = None,
                 fix_params: Optional[Mapping[str, Mapping[str, Any]]] = None, source_kind: Optional[str] = None,
                 fixes: Optional[Sequence[str]] = None, headers: Optional[Sequence[tuple[str, str]]] = None) -> None:
        super().__init__(code)
        self.code = code
        self.status = int(status if status is not None else STATUS.get(code, 500))
        self.params = dict(params or {})
        if retry_after is None:
            retry_after = _RETRY_DEFAULTS.get(code)
        self.retry_after = None if retry_after is None else max(1, int(retry_after))
        self.detail = detail
        self.fix_params = {k: dict(v) for k, v in (fix_params or {}).items()}
        self.source_kind = source_kind
        self.fixes = tuple(fixes) if fixes is not None else None
        self.headers = list(headers or ())

    def __repr__(self) -> str:   # never includes params (they may echo user input)
        return f"WebError({self.code!r}, {self.status})"


def _fix_actions(err: WebError, *, naver_available: bool) -> list[str]:
    actions = list(err.fixes if err.fixes is not None else DEFAULT_FIXES.get(err.code, ()))
    if err.source_kind in _UPLOAD_KINDS:
        actions = ["reupload" if a == "extend_range" else a for a in actions]
    if not naver_available:
        actions = [a for a in actions if a != "use_naver"]
    return list(dict.fromkeys(actions))[:_MAX_FIXES]


def _fix_object(action: str, params: Mapping[str, Any]) -> dict:
    from .viewmodel import fix_object

    return fix_object(action, params)


def _text(err: WebError, lang: str, *, naver_available: bool = True) -> str:
    if err.detail is not None:
        return messages.render_web_msg(err.detail, lang)
    if not naver_available and err.code in _NO_NAVER_VARIANT:
        variant = _NO_NAVER_VARIANT[err.code]
        if catalog_code(variant, "") == variant:
            return messages.render_web(variant, err.params, lang)
        text = messages.render_web(err.code, err.params, lang)
        old, new = _NO_NAVER_CUT[err.code]["en" if lang == "en" else "ko"]
        return text.replace(old, new) if text.endswith(old) else text
    return messages.render_web(err.code, err.params, lang)


def error_object(err: WebError, request_id: Optional[str] = None, *, naver_available: bool = True) -> dict:
    """The C.12.1 error object (the body *inside* ``{"error": …}``), also used for ``failed`` entries and
    ``problem``."""
    fixes = []
    for action in _fix_actions(err, naver_available=naver_available):
        params = dict(err.fix_params.get(action, {}))
        if action == "retry_in" and "seconds" not in params:
            params["seconds"] = err.retry_after or 5
        fixes.append(_fix_object(action, params))
    return {
        "code": err.code,
        "status": err.status,
        "message_ko": _text(err, "ko", naver_available=naver_available),
        "message_en": _text(err, "en", naver_available=naver_available),
        "params": dict(err.params),
        "fixes": fixes,
        "retry_after": err.retry_after,
        "request_id": request_id,
    }


def error_envelope(err: WebError, request_id: Optional[str] = None, *, naver_available: bool = True) -> dict:
    """``{"error": error_object(...)}``."""
    return {"error": error_object(err, request_id, naver_available=naver_available)}


def wiki_search_url(title: str, lang: str = "ko") -> str:
    """The ``wiki_title_help`` fix URL: ``https://<wl>.wikipedia.org/w/index.php?search=<title>``."""
    wl = lang if isinstance(lang, str) and lang.isalpha() and 2 <= len(lang) <= 3 else "ko"
    return f"https://{wl.lower()}.wikipedia.org/w/index.php?search={quote(str(title), safe='')}"


def encode_series(terms: Sequence[str]) -> str:
    """One series in the URL ``q`` grammar (F.2): every term ``encodeURIComponent``-encoded, joined with ``+``
    (the ``term`` param of the ``reanalyze_single`` fix)."""
    return "+".join(quote(str(t), safe="-_.!~*'()") for t in terms)


# --------------------------------------------------------------------------- connector errors (C.12.3)

def _upstream_retry_after(exc: BaseException, default: int) -> int:
    """The upstream ``Retry-After`` a connector attached to its error (``exc.retry_after`` or ``params.retry_after``,
    seconds), clamped to 1-300; ``default`` when there is none."""
    raw = getattr(exc, "retry_after", None)
    if raw is None:
        raw = (getattr(exc, "params", None) or {}).get("retry_after")
    try:
        value = int(float(raw))
    except (TypeError, ValueError, OverflowError):
        return default
    return max(1, min(_RETRY_AFTER_MAX, value))


_CONN_SIMPLE: dict[str, str] = {
    "conn_naver_unconfigured": "naver_unconfigured",
    "conn_naver_auth": "naver_auth",
    "conn_naver_rate": "naver_rate",
    "conn_naver_server": "naver_server",
    "conn_naver_http": "source_bad_response",
    "conn_naver_bad_response": "source_bad_response",
    "conn_naver_network": "source_unreachable",
    "conn_naver_timeout": "source_timeout",
    "conn_timeout": "source_timeout",
    "conn_wiki_429": "wiki_rate",
    "conn_network": "source_unreachable",
    "conn_wiki_http": "source_bad_response",
    "conn_wiki_error": "source_bad_response",
    "conn_wiki_json": "source_bad_response",
    "conn_wiki_no_valid": "source_bad_response",
    "conn_no_article": "empty_query",
    "conn_no_keyword": "empty_query",
}

#: web codes of connector errors that are the request's fault (they fail the whole job, E.6)
REQUEST_CLASS = frozenset({"invalid_param", "empty_query", "too_many_series", "too_many_keywords",
                           "internal_error"})


def from_connector_error(exc: BaseException, *, title: Optional[str] = None, wiki_lang: str = "ko",
                         source_kind: Optional[str] = None) -> WebError:
    """Map a ``ConnectorError`` to its web code (C.12.3) and log ``code`` / ``log_detail`` at WARNING."""
    code = getattr(exc, "code", None)
    params = dict(getattr(exc, "params", {}) or {})
    LOG.warning("connector error code=%s detail=%s", code or type(exc).__name__,
                getattr(exc, "log_detail", None) or "-")
    simple = _CONN_SIMPLE.get(str(code))
    if simple is not None:
        if simple == "naver_rate":
            wait = _upstream_retry_after(exc, 30)
            return WebError(simple, params={"retry_after": wait}, retry_after=wait, source_kind=source_kind)
        if simple == "wiki_rate":
            return WebError(simple, retry_after=_upstream_retry_after(exc, 60), source_kind=source_kind)
        return WebError(simple, source_kind=source_kind)
    if code == "conn_naver_too_many_groups":
        return WebError("too_many_series", params={"max": params.get("max", 5)})
    if code == "conn_naver_too_many_keywords":
        return WebError("too_many_keywords", params={"max": params.get("max", 5), "group": params.get("group", "")})
    if code == "conn_naver_bad_filter":
        what = str(params.get("what", ""))
        if what == "groupName":
            LOG.error("naver request used an invalid groupName (wire names must be g1..g5)")
            return WebError("internal_error")
        return WebError("invalid_param", params={"param": what or "naver"})
    if code in ("conn_bad_range", "conn_start_after_end") or str(code).startswith("conn_date_"):
        return WebError("invalid_param", params={"param": "range"})
    if code == "conn_bad_lang":
        return WebError("invalid_param", params={"param": "wl"})
    if code in ("conn_wiki_404", "conn_wiki_empty"):
        name = str(title if title is not None else params.get("title", ""))
        web = "wiki_not_found" if code == "conn_wiki_404" else "wiki_empty"
        fix_params = {"wiki_title_help": {"url": wiki_search_url(name, wiki_lang)}} if web == "wiki_not_found" else {}
        return WebError(web, params={"title": name}, fix_params=fix_params, source_kind=source_kind)
    return WebError("source_bad_response", source_kind=source_kind)


# --------------------------------------------------------------------------- data errors (C.12.4)

_SHORT_CODES = frozenset({"too_few_obs", "too_few_after_regularise", "series_empty", "no_valid_values"})


def _quotes_library_text(exc: BaseException) -> bool:
    """Whether a ``DataError`` text interpolates a library exception (``{error}``: pandas' "Error tokenizing data. C
    error: …", openpyxl's "No such keys(s)…"): such a text is not shown, the catalog ``upload_unreadable`` text is."""
    m = exc.args[0] if len(getattr(exc, "args", ()) or ()) == 1 else None
    parts = getattr(m, "parts", None)
    if parts:
        return any("error" in (params or {}) for _code, params in parts)
    return "error" in (getattr(exc, "params", None) or {})


def from_data_error(exc: BaseException, context: str, *, source_kind: Optional[str] = None,
                    n_obs: Optional[int] = None, col: Optional[str] = None) -> WebError:
    """Map a ``DataError`` (C.12.4). ``context``: ``"upload_parse"`` (read_table_bytes), ``"analyze_cols"``
    (``frame_to_series`` with the user's columns) or ``"series"`` (prepare / run of one series). The ``problem``
    cases of upload detection are handled by ``uploads.py``."""
    code = str(getattr(exc, "code", None) or "")
    params = dict(getattr(exc, "params", {}) or {})
    if context == "upload_parse":
        if code == "table_too_many_columns":
            return WebError("upload_too_many_columns", params={"n": params.get("n"), "max": params.get("max")})
        if code == "excel_needs_pkg":
            return WebError("upload_excel_needs_pkg")
        return WebError("upload_unreadable", params={"reason": code or "file_unreadable"},
                        detail=exc if code and not _quotes_library_text(exc) else None)
    if context == "analyze_cols":
        if code in ("date_col_missing", "date_col_unreadable", "no_date_col"):
            return WebError("upload_bad_date_col", params={"col": str(params.get("col", col or ""))},
                            source_kind=source_kind)
        if code in ("value_col_missing", "value_col_unreadable", "no_numeric_col", "only_rownum_cols"):
            return WebError("upload_bad_value_col", params={"col": str(params.get("col", col or ""))},
                            source_kind=source_kind)
        if code in _SHORT_CODES:
            return WebError("data_too_short", params={"n": int(params.get("n", n_obs or 0) or 0)},
                            source_kind=source_kind)
        return WebError("series_failed_web", source_kind=source_kind)
    return series_error(exc, source_kind=source_kind, n_obs=n_obs)


def series_error(exc: BaseException, *, source_kind: Optional[str] = None, n_obs: Optional[int] = None) -> WebError:
    """The web error of one failed series (C.12.4 per-series rows; anything else ``series_failed_web``)."""
    code = str(getattr(exc, "code", None) or "")
    params = dict(getattr(exc, "params", {}) or {}) if code else {}
    if code in _SHORT_CODES:
        n = params.get("n", n_obs if n_obs is not None else 0)
        try:
            n = int(n)
        except (TypeError, ValueError):
            n = int(n_obs or 0)
        return WebError("data_too_short", params={"n": n}, source_kind=source_kind)
    if code.startswith("horizon_"):
        return WebError("invalid_param", params={"param": "h"}, source_kind=source_kind)
    return WebError("series_failed_web", source_kind=source_kind)
