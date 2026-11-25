"""The analyze job (spec C.5, E.6, E.10): request normalisation, display names, cache keys and the job itself.

Stages: ``data`` (fetch / read and prepare every series, capture notices) -> ``series`` (``pipeline.run`` per
series, VMs, exports) -> ``finish`` (ranking, slots, compare VM, files, result cache) -> ``done``. Data notices are
captured with ``data.collect_notices()`` only; everything ``pipeline.run`` emits is collected and discarded (E.6
step 3). Nothing derived from an upload is written before the job finishes (E.7).
"""

from __future__ import annotations

import contextlib
import dataclasses
import datetime as _dt
import hashlib
import json
import logging
import math
import re
import threading
import unicodedata
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterator, Mapping, Optional, Sequence

from .. import __version__, data
from ..connectors import NAVER_EARLIEST, WIKI_EARLIEST, ConnectorError, NaverGroup, naver_cache_key, period_bounds
from . import exports, fmt, texts, viewmodel
from .errors import REQUEST_CLASS, WebError, catalog_code, encode_series, error_object, from_connector_error, \
    from_data_error, series_error
from .jobs import JobCancelled, JobContext, JobTimeout
from .log import log_exception, short_id
from .store import Store, iso_date, parse_iso, utc_iso

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd

__all__ = ["AnalyzeQuery", "normalize", "display_names", "sample_display_names", "result_cache_key", "inflight_key",
           "naver_groups", "naver_key", "canonical_json", "run_analyze_job", "SingleFlight", "HORIZON_MAX",
           "MAX_COMPARE", "MAX_KEYWORDS", "MAX_TERM_LENGTH", "MAX_SERIES_POINTS", "MAX_REQUEST_POINTS",
           "TRANSIENT_CODES"]

LOG = logging.getLogger("trendai.web.jobs")

MAX_COMPARE = 5
MAX_KEYWORDS = 5
MAX_TERM_LENGTH = 50
MAX_UNIT_LABEL = 12
HORIZON_MAX = {"D": 365, "W": 104, "M": 36, "Q": 16, "Y": 10}
_UNIT_FREQ = {"date": "D", "week": "W", "month": "M"}
_GRANULARITY = {"date": "daily", "week": "weekly", "month": "monthly"}
_RANGE_RX = re.compile(r"^(?:1y|3y|5y|all|\d{4}-\d{2}-\d{2}~\d{4}-\d{2}-\d{2})$")
_WL_RX = re.compile(r"^[a-z]{2,3}$")
_CONTROL_RX = re.compile(r"[\x00-\x1f\x7f-\x9f  ]")
_AGES = tuple(str(i) for i in range(1, 12))
_LONG_HORIZON_RATIO = 5
_SHORT_DATA = 8
_UPLOAD_HORIZON_LIMIT = 10_000
_NAVER_FRESH_TTL_S = 24 * 3600.0
_WIKI_TTL_S = {"monthly": 24 * 3600.0, "daily": 6 * 3600.0, "weekly": 6 * 3600.0}
#: work budget of an upload analysis: points of one prepared series, and of all series of one request. Above them the
#: job's run time, memory (two VMs + two JSON exports per series) and stored result size grow without bound, so an
#: upload within the file limits could time out or fill memory/disk (``422 upload_too_large_table``).
MAX_SERIES_POINTS = 20_000
MAX_REQUEST_POINTS = 60_000
_DAYS_PER = {"D": 1.0, "W": 7.0, "M": 30.44, "Q": 91.3, "Y": 365.25}
#: failure codes of a series that a retry can fix: a result holding one is never put in the result cache, so the
#: error card's "retry" re-runs the query instead of serving the same failure for up to 24 h
TRANSIENT_CODES = frozenset({"wiki_rate", "naver_rate", "naver_server", "source_timeout", "source_unreachable",
                             "source_bad_response", "series_failed_web", "job_timeout", "internal_error"})


def canonical_json(obj: Any) -> str:
    """E.10: ``json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)``."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256(obj: Any) -> str:
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- normalisation (C.5)

@dataclass(frozen=True)
class AnalyzeQuery:
    """A validated, normalised analyze request (E.6 step 1)."""

    source: str
    series: tuple[tuple[str, ...], ...]        # terms per series; uploads: ((column,), ...)
    names: tuple[str, ...]                     # display names (E.6.1; the Korean VM, pipeline.run, rank_trends)
    range: str = "5y"
    unit: Optional[str] = None                 # date | week | month (naver, wiki)
    freq: Optional[str] = None                 # D/W/M (naver, wiki), M (sample), upload: requested or None (auto)
    h: Optional[int] = None                    # None = "auto"
    dev: str = ""
    gender: str = ""
    ages: tuple[str, ...] = ()
    wl: str = "ko"
    start: Optional[_dt.date] = None
    end: Optional[_dt.date] = None
    today: Optional[_dt.date] = None           # the date the bounds were computed for (KST Naver, UTC Wiki)
    upload_id: Optional[str] = None
    date_col: Optional[str] = None
    value_cols: Optional[tuple[str, ...]] = None
    unit_label: Optional[str] = None
    filename: Optional[str] = None
    names_en: Optional[tuple[str, ...]] = None  # English display names when they differ (samples), else ``names``

    @property
    def size(self) -> int:
        return len(self.series)

    @property
    def view(self) -> str:
        return "compare" if self.size >= 2 else "detail"

    def resolved_h(self, freq: Optional[str] = None) -> int:
        f = freq or self.freq or "M"
        return int(self.h) if self.h is not None else int(data.default_horizon(f))

    def echo(self, *, h: Optional[int] = None, freq: Optional[str] = None,
             date_col: Optional[str] = None) -> dict:
        """The normalised ``query`` of C.7."""
        if self.source == "upload":
            return {"source": "upload", "filename": self.filename, "date_col": date_col,
                    "value_cols": list(self.names), "freq": freq or self.freq,
                    "h": h if h is not None else self.h, "unit_label": self.unit_label}
        series = [{"terms": list(t)} for t in self.series]
        if self.source == "sample":
            return {"source": "sample", "series": series, "h": self.resolved_h("M")}
        return {"source": self.source, "series": series, "range": self.range, "unit": self.unit,
                "h": self.resolved_h(), "dev": self.dev, "gender": self.gender, "ages": list(self.ages),
                "wl": self.wl, "period": {"start": self.start.isoformat() if self.start else None,
                                          "end": self.end.isoformat() if self.end else None}}


def _bad(param: str) -> WebError:
    return WebError("invalid_param", params={"param": param})


def display_names(series_terms: Sequence[Sequence[str]]) -> list[str]:
    """E.6.1: base = ``terms[0]``, or ``" + ".join(terms)`` when another series of the request has the same first
    term; a base already taken by an earlier series becomes ``"{base} ({n})"`` with the smallest free ``n >= 2``."""
    firsts = [t[0] for t in series_terms]
    bases = [" + ".join(t) if firsts.count(t[0]) > 1 else t[0] for t in series_terms]
    taken: set[str] = set()
    out = []
    for base in bases:
        name, n = base, 2
        while name in taken:
            name, n = f"{base} ({n})", n + 1
        taken.add(name)
        out.append(name)
    return out


def sample_display_names(ids: Sequence[str], lang: str) -> list[str]:
    """E.6.1 for samples: the sample title of G.13 in the VM language (``growing`` -> ``성장형`` / ``Growing``), the
    same text the gallery and ``/api/samples`` show, never the internal id -- the id stays in ``terms`` (URL ``q``,
    cache keys)."""
    table = texts.SAMPLE_TITLES_EN if lang == "en" else texts.SAMPLE_TITLES
    return display_names([(table.get(i, i),) for i in ids])


def _terms(raw_series: Any, source: str) -> list[tuple[str, ...]]:
    if not isinstance(raw_series, list):
        if raw_series is None:
            raise WebError("empty_query")
        raise _bad("series")
    out: list[tuple[str, ...]] = []
    for item in raw_series:
        if not isinstance(item, dict) or not isinstance(item.get("terms"), list):
            raise _bad("series")
        terms: list[str] = []
        for t in item["terms"]:
            if not isinstance(t, str):
                raise _bad("series")
            t = unicodedata.normalize("NFC", t).strip()
            if not t:
                continue
            if _CONTROL_RX.search(t):
                raise _bad("series")
            if len(t) > MAX_TERM_LENGTH:
                raise WebError("term_too_long", params={"max": MAX_TERM_LENGTH})
            if t not in terms:
                terms.append(t)
        if terms and tuple(terms) not in out:
            out.append(tuple(terms))
    if not out:
        raise WebError("empty_query")
    if len(out) > MAX_COMPARE:
        raise WebError("too_many_series", params={"max": MAX_COMPARE})
    return out


def _horizon(value: Any, freq: Optional[str], *, upload_auto: bool = False) -> Optional[int]:
    if value is None or value == "auto":
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise _bad("h")
    limit = _UPLOAD_HORIZON_LIMIT if upload_auto else HORIZON_MAX.get(freq or "M", 36)
    if not 1 <= value <= limit:
        raise _bad("h")
    return int(value)


def normalize(body: Mapping[str, Any], *, today_kst: _dt.date, today_utc: _dt.date, upload_cache: Any = None,
              max_series: int = 20) -> AnalyzeQuery:
    """Validate and normalise a C.5 body (violations -> ``400``, ``410 upload_expired``)."""
    source = body.get("source")
    if source not in ("naver", "wiki", "sample", "upload"):
        raise _bad("source")
    if source == "upload":
        return _normalize_upload(body, upload_cache, max_series)

    series = _terms(body.get("series"), source)
    if source == "naver":
        for terms, name in zip(series, display_names(series), strict=True):
            if len(terms) > MAX_KEYWORDS:
                raise WebError("too_many_keywords", params={"max": MAX_KEYWORDS, "group": name})
    elif source == "wiki":
        if any(len(t) != 1 for t in series):
            raise WebError("wiki_group_unsupported")
    else:
        from ..samples import list_samples

        known = list_samples()
        for terms in series:
            if len(terms) != 1 or terms[0] not in known:
                raise WebError("unknown_sample", params={"name": " + ".join(terms)})
        ids = [t[0] for t in series]
        return AnalyzeQuery(source="sample", series=tuple(series), names=tuple(sample_display_names(ids, "ko")),
                            names_en=tuple(sample_display_names(ids, "en")), freq="M",
                            h=_horizon(body.get("h", "auto"), "M"))

    rng = body.get("range", "5y")
    rng = "5y" if rng is None else rng
    if not isinstance(rng, str) or not _RANGE_RX.match(rng):
        raise _bad("range")
    unit = body.get("unit", "auto")
    unit = "auto" if unit is None else unit
    if unit not in ("auto", "date", "week", "month"):
        raise _bad("unit")
    if unit == "auto":
        unit = "week" if source == "naver" else "month"
    freq = _UNIT_FREQ[unit]
    h = _horizon(body.get("h", "auto"), freq)

    dev = gender = ""
    ages: tuple[str, ...] = ()
    wl = "ko"
    if source == "naver":
        dev = body.get("dev", "") or ""
        gender = body.get("gender", "") or ""
        if dev not in ("", "pc", "mo"):
            raise _bad("dev")
        if gender not in ("", "m", "f"):
            raise _bad("gender")
        raw_ages = body.get("ages", []) or []
        if not isinstance(raw_ages, list) or any(not isinstance(a, str) or a not in _AGES for a in raw_ages):
            raise _bad("ages")
        ages = tuple(sorted(set(raw_ages), key=int))
        today = today_kst
        earliest = NAVER_EARLIEST
    else:
        wl = body.get("wl", "ko") or "ko"
        if not isinstance(wl, str) or not _WL_RX.match(wl):
            raise _bad("wl")
        today = today_utc
        earliest = WIKI_EARLIEST
    try:
        start, end = period_bounds(rng, unit, today, earliest=earliest)
    except ConnectorError:
        raise _bad("range") from None
    return AnalyzeQuery(source=source, series=tuple(series), names=tuple(display_names(series)), range=rng,
                        unit=unit, freq=freq, h=h, dev=dev, gender=gender, ages=ages, wl=wl, start=start, end=end,
                        today=today)


def _normalize_upload(body: Mapping[str, Any], upload_cache: Any, max_series: int) -> AnalyzeQuery:
    uid = body.get("upload_id")
    valid = isinstance(uid, str) and re.match(r"^[A-Za-z0-9_-]{22}$", uid) is not None
    entry = upload_cache.get(uid) if valid else None
    if entry is None:
        if valid and getattr(upload_cache, "evicted", None) is not None and upload_cache.evicted(uid):
            # dropped for space seconds ago, not 30 minutes: say so (until the catalog has the text, the expiry one)
            raise WebError(catalog_code("upload_evicted", "upload_expired"), 410)
        raise WebError("upload_expired")
    frame_cols = [str(c) for c in entry.raw.frame.columns]
    date_col = body.get("date_col")
    if date_col is not None:
        if not isinstance(date_col, str):
            raise _bad("date_col")
        date_col = date_col.strip()
        if date_col not in frame_cols:
            raise WebError("upload_bad_date_col", params={"col": date_col}, source_kind="upload")
    value_cols = body.get("value_cols")
    if value_cols is not None:
        if not isinstance(value_cols, list) or any(not isinstance(c, str) for c in value_cols):
            raise _bad("value_cols")
        cols = list(dict.fromkeys(c.strip() for c in value_cols if c.strip()))
        if not cols:
            raise _bad("value_cols")
        if len(cols) > max_series:
            raise WebError("upload_too_many_series", params={"max": max_series})
        known = set(frame_cols) | set(entry.value_cols)
        for c in cols:
            if c not in known:
                raise WebError("upload_bad_value_col", params={"col": c}, source_kind="upload")
        names = tuple(cols)
    else:
        if len(entry.value_cols) > max_series:
            raise WebError("upload_too_many_series", params={"max": max_series})
        if not entry.value_cols:
            raise WebError("upload_bad_value_col", params={"col": ""}, source_kind="upload")
        names = tuple(entry.value_cols)
    freq = body.get("freq", "auto")
    freq = "auto" if freq is None else freq
    if freq not in ("auto", "D", "W", "M", "Q", "Y"):
        raise _bad("freq")
    freq_req = None if freq == "auto" else freq
    h = _horizon(body.get("h", "auto"), freq_req, upload_auto=freq_req is None)
    unit_label = body.get("unit_label")
    if unit_label is not None:
        if not isinstance(unit_label, str):
            raise _bad("unit_label")
        unit_label = _CONTROL_RX.sub("", unicodedata.normalize("NFC", unit_label)).strip() or None
        if unit_label is not None and len(unit_label) > MAX_UNIT_LABEL:
            raise _bad("unit_label")
    _check_upload_budget(entry, len(names), freq_req, date_col)
    return AnalyzeQuery(source="upload", series=tuple((n,) for n in names), names=names, freq=freq_req, h=h,
                        upload_id=uid, date_col=date_col, value_cols=tuple(value_cols) if value_cols else None,
                        unit_label=unit_label, filename=entry.filename)


def _too_large(n: int, limit: int) -> WebError:
    return WebError("upload_too_large_table", params={"n": int(n), "max": int(limit)}, source_kind="upload")


def _check_upload_budget(entry: Any, n_series: int, freq_req: Optional[str], date_col: Optional[str]) -> None:
    """Refuse, before queueing, an upload analysis whose estimated work is above the budget (C.5): points per series
    (distinct dates of the detected date column, scaled to a coarser requested frequency) x series. The job checks
    the real prepared lengths again (:meth:`_Job._prepare`)."""
    points = int(getattr(entry, "n_points", 0) or 0)
    if points <= 0 or (date_col is not None and date_col not in (getattr(entry, "date_cols", None) or [])):
        return             # nothing detected, or another date column: only the in-job check applies
    detected = getattr(entry, "freq", None)
    if freq_req in _DAYS_PER and detected in _DAYS_PER and _DAYS_PER[freq_req] > _DAYS_PER[detected]:
        points = int(points * _DAYS_PER[detected] / _DAYS_PER[freq_req]) + 1
    limit = min(MAX_REQUEST_POINTS, MAX_SERIES_POINTS * max(1, n_series))
    if points * n_series > limit:
        raise _too_large(points * n_series, limit)


def naver_groups(q: AnalyzeQuery) -> list[NaverGroup]:
    """One group per series with the wire names ``g1``..``g5`` (C.5): display names never reach Naver."""
    return [NaverGroup(name=f"g{k + 1}", keywords=tuple(terms)) for k, terms in enumerate(q.series)]


def naver_key(q: AnalyzeQuery) -> str:
    return naver_cache_key(naver_groups(q), q.start, q.end, q.unit, q.dev or None, q.gender or None, q.ages)


def inflight_key(q: AnalyzeQuery) -> Optional[str]:
    """The in-flight index key (C.5 joins): the result-cache key, and for uploads a key of the upload and the column
    selection -- so repeating one upload analysis while it runs joins the running job instead of queueing copies
    (uploads are still never put in the result cache)."""
    if q.source != "upload":
        return result_cache_key(q)
    return _sha256({"engine": __version__, "vm_schema": viewmodel.WEB_VM_SCHEMA, "source": "upload",
                    "upload_id": q.upload_id, "date_col": q.date_col, "names": list(q.names), "freq": q.freq,
                    "h": q.h, "unit_label": q.unit_label})


def result_cache_key(q: AnalyzeQuery) -> Optional[str]:
    """E.10 result-cache key (``None`` for uploads, which are never cached)."""
    if q.source == "upload":
        return None
    return _sha256({"engine": __version__, "vm_schema": viewmodel.WEB_VM_SCHEMA, "source": q.source,
                    "series": [list(t) for t in q.series],
                    "start": q.start.isoformat() if q.start else None, "end": q.end.isoformat() if q.end else None,
                    "unit": q.unit, "h": q.resolved_h(), "dev": q.dev, "gender": q.gender, "ages": list(q.ages),
                    "wl": q.wl})


# --------------------------------------------------------------------------- Naver single-flight (E.6)

class SingleFlight:
    """Per-key locks: the second job needing the same Naver body waits for the first, then re-reads the cache."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._locks: dict[str, list] = {}

    @contextlib.contextmanager
    def __call__(self, key: str) -> Iterator[None]:
        with self._lock:
            entry = self._locks.setdefault(key, [threading.Lock(), 0])
            entry[1] += 1
        try:
            with entry[0]:
                yield
        finally:
            with self._lock:
                entry[1] -= 1
                if entry[1] == 0:
                    self._locks.pop(key, None)


# --------------------------------------------------------------------------- the job (E.6)

@dataclass
class _Item:
    index: int
    name: str
    terms: tuple[str, ...]
    raw: Any = None                       # raw pd.Series
    error: Optional[WebError] = None
    as_of: str = ""
    filled: int = 0
    kind: str = ""
    value_kind: str = "raw"
    meta_extra: dict = field(default_factory=dict)
    prepared: Any = None
    freq: Optional[str] = None
    h: Optional[int] = None
    notices: dict = field(default_factory=lambda: {"ko": [], "en": []})
    name_en: Optional[str] = None         # English display name when it differs (samples)


def _num(v: Any) -> Optional[float]:
    """A stored data-cache number: 15 significant digits, ``None`` for missing / non-finite values and for bools
    (the same rule as the VM's numbers)."""
    if isinstance(v, bool):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return float(f"{f:.15g}") if math.isfinite(f) else None


def _series_from(dates: Sequence[str], values: Sequence[Any], name: str) -> "pd.Series":
    import numpy as np
    import pandas as pd

    idx = pd.DatetimeIndex(pd.to_datetime(list(dates)), name="date")
    vals = np.array([np.nan if v is None else float(v) for v in values], dtype=float)
    return pd.Series(vals, index=idx, name=name)


def _iso_dates(index: Any) -> list[str]:
    return [iso_date(d) for d in index]


def _renamed(report: Any, name: str) -> Any:
    """A shallow copy of a ``TrendReport`` under another display name (the English sample title): the numbers are
    shared, only ``report.name`` / ``analysis.name`` -- read by the narrative, the VM and the JSON export -- change."""
    return dataclasses.replace(report, name=name, analysis=dataclasses.replace(report.analysis, name=name))


class _Job:
    """One analyze job; ``app`` is the TrendaiApp (config, clock, store, fetchers, caches, budgets)."""

    def __init__(self, app: Any, q: AnalyzeQuery, job_id: str, request_id: Optional[str]) -> None:
        self.app = app
        self.q = q
        self.rid = job_id
        self.request_id = request_id
        self.now = app.clock()
        self.as_of_now = utc_iso(self.now)
        self.result_notices: dict[str, list] = {"ko": [], "en": []}
        self.data_expiry: Optional[_dt.datetime] = None
        self.naver_mode = "-"
        self.naver_endpoint: Optional[str] = None
        self.gt_category: Optional[str] = None
        self.source_kind = q.source
        self.points = 0                    # prepared points so far (upload work budget)

    # ---------------------------------------------------------------- helpers
    def _expiry(self, when: Optional[_dt.datetime]) -> None:
        if when is not None and (self.data_expiry is None or when < self.data_expiry):
            self.data_expiry = when

    def _fail(self, item: _Item, err: WebError) -> None:
        if err.source_kind is None and self.source_kind in ("upload", "google_trends_upload"):
            err.source_kind = self.source_kind
        item.error = err

    def _notice(self, code: str, *, index: Optional[int], params: Optional[dict] = None,
                fixes: Optional[list] = None, lang_params: Optional[Mapping[str, dict]] = None) -> dict:
        return {lang: viewmodel.make_notice(code, lang, params=(lang_params or {}).get(lang, params),
                                            series_index=index, fixes=fixes, source_kind=self.source_kind)
                for lang in ("ko", "en")}

    def _add(self, target: dict, notice: dict) -> None:
        for lang in ("ko", "en"):
            target[lang].append(notice[lang])

    # ---------------------------------------------------------------- stage data
    def _naver(self, ctx: JobContext) -> list[_Item]:
        app, q = self.app, self.q
        if app.fetchers.naver is None:
            raise WebError("naver_unconfigured")
        key = naver_key(q)
        entry = app.store.cache_get("naver", key)
        mode = "cache"
        if entry is None:
            with app.naver_flight(key):
                ctx.check()
                entry = app.store.cache_get("naver", key)
                if entry is None:
                    entry = self._fetch_naver(key)
                    mode = "fresh"
        self.naver_mode = mode
        self.naver_endpoint = str(entry.get("endpoint") or app.fetchers.naver_endpoint or "hub")
        self._expiry(parse_iso(entry.get("expires_at")))
        fetched = str(entry.get("fetched_at") or self.as_of_now)
        if mode == "cache":
            self._add(self.result_notices, self._notice("naver_cached", index=None, lang_params={
                lang: {"fetched": fmt.datetime_label(fetched, lang)} for lang in ("ko", "en")}))
        groups = list(entry.get("groups") or [])
        items = []
        for k, (terms, name) in enumerate(zip(q.series, q.names, strict=True)):
            wire = f"g{k + 1}"
            g = groups[k] if k < len(groups) else None
            if not isinstance(g, dict) or g.get("name") != wire:
                raise WebError("source_bad_response")
            s = _series_from(g.get("dates") or [], g.get("ratios") or [], wire)
            s.attrs["trendai_source"] = {"kind": "naver", "group": wire, "keywords": list(terms),
                                         "time_unit": q.unit, "device": q.dev or None, "gender": q.gender or None,
                                         "ages": list(q.ages), "endpoint": self.naver_endpoint}
            items.append(_Item(index=k, name=name, terms=terms, raw=s, as_of=fetched, filled=int(g.get("filled") or 0),
                               kind="naver", value_kind="relative_index"))
        return items

    def _fetch_naver(self, key: str) -> dict:
        app, q = self.app, self.q
        app.naver_budget.reserve()
        groups = naver_groups(q)
        try:
            result = app.fetchers.naver(groups, q.start, q.end, q.unit, q.dev or None, q.gender or None,
                                        tuple(q.ages))
        except ConnectorError as exc:
            raise from_connector_error(exc, source_kind="naver") from None
        except OSError as exc:
            LOG.warning("naver fetch failed exc=%s", type(exc).__name__)
            raise WebError("source_unreachable") from None
        now = app.clock()
        latest_end = period_bounds("1y", q.unit, q.today or q.end, earliest=NAVER_EARLIEST)[1]
        ttl = _NAVER_FRESH_TTL_S if q.end == latest_end else app.config.naver_cache_days * 86400.0
        out_groups = []
        for g in groups:
            s = result.series.get(g.name)
            if s is None:
                raise WebError("source_bad_response")
            out_groups.append({"name": g.name, "keywords": list(g.keywords), "dates": _iso_dates(s.index),
                               "ratios": [_num(v) for v in s.to_numpy()],
                               "filled": int((getattr(result, "filled", {}) or {}).get(g.name, 0))})
        entry = {"key": key, "stored_at": utc_iso(now), "expires_at": utc_iso(now + _dt.timedelta(seconds=ttl)),
                 "fetched_at": utc_iso(now), "endpoint": str(getattr(result, "endpoint", None) or "hub"),
                 "time_unit": q.unit, "start": q.start.isoformat(), "end": q.end.isoformat(), "groups": out_groups,
                 "filters": {"device": q.dev or None, "gender": q.gender or None, "ages": list(q.ages)}}
        app.store.cache_put("naver", key, entry)
        return entry

    def _wiki(self, ctx: JobContext) -> list[_Item]:
        app, q = self.app, self.q
        granularity = _GRANULARITY[q.unit]
        items = []
        for k, (terms, name) in enumerate(zip(q.series, q.names, strict=True)):
            title = terms[0]
            item = _Item(index=k, name=name, terms=terms, kind="wiki", value_kind="pageviews")
            items.append(item)
            key = _sha256({"title": title, "lang": q.wl, "granularity": granularity,
                           "start": q.start.isoformat(), "end": q.end.isoformat()})
            entry = app.store.cache_get("wiki", key)
            if entry is None:
                ctx.check()
                try:
                    s = app.fetchers.wiki(title, q.wl, q.start, q.end, granularity)
                except ConnectorError as exc:
                    err = from_connector_error(exc, title=title, wiki_lang=q.wl, source_kind="wiki")
                    if err.code in REQUEST_CLASS:
                        raise err from None
                    self._fail(item, err)
                    continue
                except OSError as exc:
                    LOG.warning("wiki fetch failed exc=%s", type(exc).__name__)
                    self._fail(item, WebError("source_unreachable"))
                    continue
                now = app.clock()
                entry = {"key": key, "stored_at": utc_iso(now),
                         "expires_at": utc_iso(now + _dt.timedelta(seconds=_WIKI_TTL_S[granularity])),
                         "fetched_at": utc_iso(now), "title": title, "lang": q.wl, "granularity": granularity,
                         "dates": _iso_dates(s.index), "values": [_num(v) for v in s.to_numpy()]}
                app.store.cache_put("wiki", key, entry)
            self._expiry(parse_iso(entry.get("expires_at")))
            item.raw = _series_from(entry.get("dates") or [], entry.get("values") or [], title)
            item.as_of = str(entry.get("fetched_at") or self.as_of_now)
            item.meta_extra = {"wiki_lang": q.wl, "wiki_title": title}
        return items

    def _sample(self) -> list[_Item]:
        from ..samples import sample

        names_en = self.q.names_en or self.q.names
        return [_Item(index=k, name=name, terms=terms, raw=sample(terms[0]), as_of=self.as_of_now, kind="sample",
                      value_kind="raw", meta_extra={"synthetic": True}, name_en=names_en[k])
                for k, (terms, name) in enumerate(zip(self.q.series, self.q.names, strict=True))]

    def _upload(self) -> list[_Item]:
        q = self.q
        entry = self.app.uploads.get(q.upload_id)
        if entry is None:
            raise WebError("upload_expired")
        table = entry.raw
        gt = table.google_trends
        self.source_kind = "google_trends_upload" if gt is not None else "upload"
        self.gt_category = gt.category if gt is not None else None
        with data.collect_notices() as notes:
            try:
                if q.date_col is None and entry.wide:
                    got = data.frame_to_series(table.frame, prefer_decimal_comma=table.decimal_comma)
                else:
                    cols = list(q.value_cols) if q.value_cols else None
                    got = data.frame_to_series(table.frame, q.date_col, cols, prefer_decimal_comma=table.decimal_comma)
            except data.DataError as exc:
                col = q.date_col if "date" in str(getattr(exc, "code", "")) else ", ".join(q.value_cols or ())
                raise from_data_error(exc, "analyze_cols", source_kind=self.source_kind, col=col) from None
        for w in notes:
            notice = {lang: viewmodel.notice_from_warning(w, lang, series_index=None) for lang in ("ko", "en")}
            self._add(self.result_notices, notice)
        first = next(iter(got.values()), None)
        date_cols = first.attrs.get("trendai", {}).get("date_cols") if first is not None else None
        self.used_date_col = date_cols[0] if date_cols else None
        items = []
        for k, name in enumerate(q.names):
            s = got.get(name)
            item = _Item(index=k, name=name, terms=(name,), kind=self.source_kind,
                         value_kind="relative_index_gt" if gt is not None else "raw", as_of=self.as_of_now)
            items.append(item)
            if s is None:
                self._fail(item, WebError("upload_bad_value_col", params={"col": name}))
                continue
            extra: dict = {"filename": q.filename, "unit_label": q.unit_label}
            if gt is not None:
                term, region = gt.terms.get(name, (name, None))
                s.attrs["trendai"] = {**s.attrs.get("trendai", {}), "source": "google_trends_upload", "term": term,
                                      "region": region, "value_kind": "relative_index_gt"}
                extra.update(gt_term=str(term), gt_region=region, gt_category=gt.category)
                item.terms = (str(term),)
            item.raw = s
            item.meta_extra = extra
        return items

    def _prepare(self, item: _Item) -> None:
        """``data.prepare`` inside ``collect_notices`` + the attrs copy + the extra series notices (E.6 step 2)."""
        import numpy as np

        q = self.q
        with data.collect_notices() as notes:
            try:
                prepared, f = data.prepare(item.raw, freq=q.freq, name=item.name)
            except data.DataError as exc:
                n_obs = int(np.isfinite(np.asarray(item.raw, dtype=float)).sum()) if item.raw is not None else 0
                self._fail(item, series_error(exc, n_obs=n_obs))
                return
        prepared.attrs.update(item.raw.attrs)
        n_points = int(len(prepared))
        if q.source == "upload":         # the work budget (C.5): the real lengths after regularisation
            if n_points > MAX_SERIES_POINTS:
                self._fail(item, _too_large(n_points, MAX_SERIES_POINTS))
                return
            if self.points + n_points > MAX_REQUEST_POINTS:
                self._fail(item, _too_large(self.points + n_points, MAX_REQUEST_POINTS))
                return
            self.points += n_points
        item.prepared, item.freq = prepared, f
        k = item.index
        for w in notes:
            self._add(item.notices, {lang: viewmodel.notice_from_warning(w, lang, series_index=k)
                                     for lang in ("ko", "en")})
        values = np.asarray(prepared, dtype=float)
        n = int(len(values))
        h = q.resolved_h(f)
        if q.source == "upload" and q.h is not None and q.freq is None and h > HORIZON_MAX.get(f, 36):
            h = HORIZON_MAX.get(f, 36)
            self._add(item.notices, self._notice("horizon_clamped", index=k, lang_params={
                lang: {"span": fmt.span_label(h, f, lang)} for lang in ("ko", "en")}))
        item.h = h
        if n < _SHORT_DATA:
            self._add(item.notices, self._notice("data_short", index=k, params={"n": n}))
        finite = values[np.isfinite(values)]
        if item.kind in ("naver", "wiki") and n and float(np.mean(finite == 0)) >= 0.5:
            self._add(item.notices, self._notice("mostly_zero_volume", index=k))
        if item.kind == "naver" and item.filled > 0:
            self._add(item.notices, self._notice("naver_missing_filled", index=k, params={"n": item.filled}))
        actual_max = float(finite.max()) if finite.size else None
        if item.kind == "naver" and viewmodel.is_low_resolution("naver", actual_max, q.size):
            fix = viewmodel.fix_object("reanalyze_single", {"term": encode_series(item.terms)})
            self._add(item.notices, self._notice("naver_low_resolution", index=k, fixes=[fix]))
        if h > _LONG_HORIZON_RATIO * n:
            self._add(item.notices, self._notice("horizon_long", index=k,
                                                 params={"horizon": h, "n": n, "ratio": _LONG_HORIZON_RATIO}))

    def _meta(self, item: _Item, report: Any) -> viewmodel.SourceMeta:
        idx = report.analysis.series.index
        q = self.q
        extra = dict(item.meta_extra)
        if item.kind == "naver":
            extra.update(naver_endpoint=self.naver_endpoint, device=q.dev or None, gender=q.gender or None,
                         ages=tuple(q.ages))
        return viewmodel.SourceMeta(kind=item.kind, value_kind=item.value_kind, freq=item.freq,
                                    start=_iso_dates(idx[:1])[0], end=_iso_dates(idx[-1:])[0], as_of=item.as_of,
                                    terms=tuple(item.terms), **extra)

    # ---------------------------------------------------------------- run
    def run(self, ctx: JobContext) -> None:
        from .. import compare as compare_mod

        app, q = self.app, self.q
        n_total = q.size
        ctx.set_total(n_total, q.view)
        ctx.set_stage("data")
        ctx.check()
        if q.source == "naver":
            items = self._naver(ctx)
        elif q.source == "wiki":
            items = self._wiki(ctx)
        elif q.source == "sample":
            items = self._sample()
        else:
            items = self._upload()
        ctx.job.log_info["naver"] = self.naver_mode
        for item in items:
            if item.error is None:
                ctx.check()
                self._prepare(item)
        ctx.set_notices(self.result_notices)

        done: dict[int, dict] = {}
        failed_errors: dict[int, WebError] = {}
        failed_entries: list[dict] = []
        timed_out = False
        for item in items:
            k = item.index
            if item.error is None and timed_out:
                self._fail(item, WebError(ctx.job.timeout_code))
            elif item.error is None:
                try:
                    self._series(ctx, item, n_total, done)
                except JobTimeout:
                    if not done:
                        raise
                    # some series are finished: keep them (finish below) and fail this one and the rest with
                    # job_timeout instead of throwing every finished series away (the timeout is from the start)
                    timed_out = True
                    ctx.allow_overrun()
                    self._fail(item, WebError(ctx.job.timeout_code))
            if item.error is not None:
                if item.error.source_kind is None and self.source_kind in ("upload", "google_trends_upload"):
                    item.error.source_kind = self.source_kind
                entry = viewmodel.build_failed_entry(
                    k, item.name, error_object(item.error, self.request_id, naver_available=app.naver_available()))
                failed_errors[k] = item.error
                failed_entries.append(entry)
                ctx.add_failed(entry)
            ctx.series_finished()

        if done:
            ctx.allow_overrun()     # stage finish is bounded work: never lose finished series to the time limit
        ctx.check()
        ctx.set_stage("finish")
        if not done:
            raise failed_errors[min(failed_errors)] if failed_errors else WebError("data_too_short",
                                                                                      params={"n": 0})
        self._finish(ctx, done, failed_entries, failed_errors, compare_mod)

    def _series(self, ctx: JobContext, item: _Item, n_total: int, done: dict) -> None:
        """Stage ``series`` of one prepared series: ``pipeline.run``, the VMs, exports, LLM payload and HTML. A
        failure is recorded on the item (``JobCancelled`` / ``JobTimeout`` propagate)."""
        from .. import narrative, pipeline

        app = self.app
        k = item.index
        ctx.check()
        current = {"index": k, "name": item.name, "step": "analyze"}
        if item.name_en and item.name_en != item.name:
            current["name_en"] = item.name_en          # the progress line in English (jobs.analyze_snapshot)
        ctx.set_stage("series", current)
        is_naver = self.source_kind == "naver"
        try:
            with data.collect_notices():   # everything run() emits is discarded by code (E.6 step 3)
                report = pipeline.run(item.prepared, name=item.name, freq=item.freq, horizon=item.h,
                                      lang="ko", on_stage=lambda st: (ctx.check(), ctx.set_step(st)))
                # the English surfaces use the English display name when it differs (sample titles, E.6.1)
                report_en = report if not item.name_en or item.name_en == item.name else _renamed(report,
                                                                                                    item.name_en)
                en_text = narrative.generate_narrative(report_en.analysis, report_en.forecast, report_en.outlook,
                                                       lang="en")[0]
            meta = self._meta(item, report)
            inp = viewmodel.SeriesInput(
                index=k, name=item.name, report=report, source=meta,
                narratives={"ko": report.narrative, "en": en_text}, notices=item.notices,
                request_size=n_total, ai_enabled=app.ai_enabled(), xlsx_available=exports.xlsx_available(),
                slot=viewmodel.provisional_slot(k, n_total))
            inp_en = inp if report_en is report else dataclasses.replace(inp, name=report_en.name, report=report_en)
            vm_ko = viewmodel.build_series_vm(inp, "ko")
            vm_en = viewmodel.build_series_vm(inp_en, "en")
            json_ko = exports.build_series_json(report, vm_ko["source"], "ko")
            json_en = exports.build_series_json(report_en, vm_en["source"], "en", narrative=en_text)
            # stored as encoded bytes right away: far smaller in memory than the object graph until stage finish
            llm = None if is_naver else Store.encode(narrative.build_llm_payload(report.analysis, report.forecast,
                                                                                 report.outlook))
            html = None if is_naver else exports.render_series_html(report, vm_ko["source"]["line"])
        except (JobCancelled, JobTimeout):
            raise
        except data.DataError as exc:
            self._fail(item, series_error(exc, n_obs=len(item.prepared)))
        except Exception as exc:  # noqa: BLE001 - one bad series must not fail the others
            log_exception(LOG, f"job={short_id(self.rid)} series={k} failed", exc, rid=self.request_id,
                          code="series_failed_web")
            self._fail(item, WebError("series_failed_web"))
        else:
            done[k] = {"item": item, "report": report, "input": inp, "input_en": inp_en,
                       "vm": {"ko": vm_ko, "en": vm_en}, "json": {"ko": json_ko, "en": json_en}, "llm": llm,
                       "html": html}
            ctx.add_series(k, {"ko": vm_ko, "en": vm_en})

    def _finish(self, ctx: JobContext, done: dict, failed_entries: list, failed_errors: Mapping[int, WebError],
                compare_mod: Any) -> None:
        app, q = self.app, self.q
        n_total = q.size
        view = q.view
        order = sorted(done)
        ranking = None
        name_to_index: dict[str, int] = {}
        if view == "compare":
            reports = {done[k]["item"].name: done[k]["report"] for k in order}
            name_to_index = {done[k]["item"].name: k for k in order}
            ranking = compare_mod.rank_trends(reports)
            ranked = [name_to_index[str(nm)] for nm in ranking["name"]]
            slots = viewmodel.assign_slots(ranked, n_total)
        else:
            slots = {k: viewmodel.provisional_slot(k, n_total) for k in order}
        for k in order:
            for lang in ("ko", "en"):
                viewmodel.set_slot(done[k]["vm"][lang], slots.get(k))
        compare_vm: dict[str, Optional[dict]] = {"ko": None, "en": None}
        if view == "compare":
            for lang in ("ko", "en"):
                # ranking rows are looked up by index (name_to_index), so the English members carry English names
                members = [done[k]["input_en" if lang == "en" else "input"] for k in order]
                compare_vm[lang] = viewmodel.build_compare_vm(members, {k: done[k]["vm"][lang] for k in order},
                                                              ranking, name_to_index, slots, failed_entries, lang)
        is_naver = self.source_kind == "naver"
        compare_html = None
        if view == "compare" and not is_naver and len(order) >= 2:
            compare_html = exports.render_result_html([done[k]["report"] for k in order], ranking,
                                                      compare_vm["ko"]["source_line"])
        av = viewmodel.export_availability(self.source_kind, xlsx=exports.xlsx_available(), view=view,
                                           succeeded=len(order))
        now = app.clock()
        expires = now + _dt.timedelta(seconds=app.config.result_ttl_s)
        # every file is encoded as soon as it is built (one document in memory at a time, not all of them)
        files: dict[str, Any] = {}
        for lang in ("ko", "en"):
            series_json = {k: done[k]["json"][lang] for k in order}
            result_json = exports.build_result_json(
                [series_json[k] for k in order],
                source_line=compare_vm[lang]["source_line"] if compare_vm[lang] is not None else None)
            files[f"export-{lang}.json"] = Store.encode(exports.build_export_store(series_json, result_json))
            del series_json, result_json
            for k in order:
                done[k]["json"][lang] = None
            files[f"vm-{lang}.json"] = Store.encode(
                {"series": [done[k]["vm"][lang] for k in order], "compare": compare_vm[lang],
                 "notices": self.result_notices[lang], "exports": viewmodel.export_block(av, lang, level="result")})
        html_series: list[Optional[str]] = [None] * n_total
        for k in order:
            if done[k]["html"] is not None:
                html_series[k] = f"report-{k}.html"
                files[f"report-{k}.html"] = done[k]["html"]
            if done[k]["llm"] is not None:
                files[f"llm-{k}.json"] = done[k]["llm"]
        if compare_html is not None:
            files["compare.html"] = compare_html
        date_col = getattr(self, "used_date_col", None)
        first = done[order[0]]["item"]
        result = {
            "schema": viewmodel.WEB_VM_SCHEMA,
            "engine": __version__,
            "result_id": self.rid,
            "created_at": utc_iso(now),
            "expires_at": utc_iso(expires),
            "query": q.echo(h=first.h, freq=first.freq, date_col=date_col),
            "source_kind": self.source_kind,
            "view": view,
            "failed": {"ko": failed_entries, "en": failed_entries},
            "available": {"series": [dict(av.series) if k in done else None for k in range(n_total)],
                          "result": dict(av.result)},
            "ranking_rows": exports.ranking_rows(ranking, name_to_index) if ranking is not None else [],
            "html": {"series": html_series, "compare": "compare.html" if compare_html is not None else None},
        }
        files["result.json"] = result
        ctx.check()
        app.store.save_dir("results", self.rid, files)
        del files
        key = result_cache_key(q)
        transient = any(err.code in TRANSIENT_CODES for err in failed_errors.values())
        if key is not None and not transient:
            # a result with a series that failed for a passing reason (a 429, a timeout) is not cached: the same
            # query must run again, or the error card's "retry" would serve the same failure until the cache expires
            until = expires if self.data_expiry is None else min(expires, self.data_expiry)
            app.store.cache_put("results", key, {"key": key, "result_id": self.rid, "expires_at": utc_iso(until)})
        ctx.set_final([{"index": k, "vm": done[k]["vm"]} for k in order],
                      compare_vm if view == "compare" else None)


def run_analyze_job(app: Any, q: AnalyzeQuery, job_id: str, request_id: Optional[str]):
    """The job function for ``JobManager.submit`` (E.6)."""
    def fn(ctx: JobContext) -> None:
        _Job(app, q, job_id, request_id).run(ctx)

    return fn
