"""View-model builder (spec section D): ``TrendReport`` objects -> the JSON the SPA renders.

Owner: **WP1-vm**. Callers: ``analyze.py`` (job time), ``narrative_ai.py`` (AI rewrites) and ``api.py``
(per-response link building and share views), all owned by WP1-server. This module is the only place that knows the
D.2 / D.4 shapes; ``vm.schema.json`` is their machine-readable twin and the two must agree.

The public names below are the agreed contract (see ``docs/WEB_INTERFACES.md``). Rules that hold for every builder:

* One call builds **one language** (``lang`` is exactly ``"ko"`` or ``"en"``). The job calls each builder twice.
* Returned dicts are JSON-ready: plain ``dict``/``list``/``str``/``int``/``float``/``bool``/``None``, floats rounded to
  15 significant digits and NaN/inf replaced by ``None`` (``pipeline._clean`` rule, spec section 0). Callers dump them
  with ``json.dumps(..., ensure_ascii=False, allow_nan=False)``.
* Stored VMs contain **no URLs and no ids** (E.7): ``exports`` blocks are built with :func:`export_block` (all links
  ``None``) and :func:`link_exports` fills the links for each response.
* Builders never mutate their inputs unless the docstring says "in place".
* Every display string is pre-formatted with :mod:`trendai.web.fmt` and the tables in :mod:`trendai.web.texts`.
  Undefined values are worded exactly like the HTML report and the narrative, through the private report helpers
  that spec E.18 allows (``report._ChartData``, ``_change_basis(_en)``, ``_change_undefined``, ``_pct_why``,
  ``_rate_why``, ``_recent_lag``).
"""

from __future__ import annotations

import copy
import math
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence

from . import fmt
from . import texts as T

if TYPE_CHECKING:  # pragma: no cover - typing only; numpy/pandas are imported where used, so that
    import pandas as pd  # ``import trendai.web.viewmodel`` stays cheap (constants, notices, export blocks)

    from ..types import TrendReport

__all__ = [
    # constants
    "WEB_VM_SCHEMA", "LANGS", "SOURCE_KINDS", "VALUE_KINDS", "FREQS", "MAX_SLOTS", "SLOT_ICONS",
    "EXPORT_FORMATS", "RESULT_EXPORT_KEYS", "NOTICE_LEVELS", "NOTICE_DEFAULTS",
    # inputs
    "SourceMeta", "SeriesInput", "ExportAvailability",
    # notices and fixes
    "fix_object", "make_notice", "notice_from_warning",
    # source block and lines
    "build_source_block", "source_line", "compare_source_line",
    # series VM
    "build_series_vm", "build_failed_entry", "is_low_resolution",
    # slots
    "provisional_slot", "assign_slots", "set_slot",
    # compare VM
    "build_compare_vm",
    # narrative helpers (rule and AI)
    "split_caveats", "localize_dates", "build_ai_record", "attach_ai", "ai_job_narrative",
    # exports blocks and response-time views
    "export_availability", "export_block", "link_exports", "make_shared_vm",
]

# ---------------------------------------------------------------------------------------------------------------------
# Constants (real values: they are part of the contract)
# ---------------------------------------------------------------------------------------------------------------------

#: VM schema version (spec section 0). Part of every result-cache key (E.10) and stored in results/shares (E.7).
#: WP1-vm bumps it whenever the output of any builder in this module changes in any way.
WEB_VM_SCHEMA: int = 2

LANGS: tuple[str, ...] = ("ko", "en")

#: Source kinds (section 0 glossary).
SOURCE_KINDS: tuple[str, ...] = ("naver", "wiki", "upload", "google_trends_upload", "sample")

#: Value kinds (section 0 glossary): Naver, Google Trends upload, Wikipedia, uploads and samples.
VALUE_KINDS: tuple[str, ...] = ("relative_index", "relative_index_gt", "pageviews", "raw")

#: Engine frequency codes the web app handles.
FREQS: tuple[str, ...] = ("D", "W", "M", "Q", "Y")

#: Comparison colour/icon slots (D.4.2, F.8).
MAX_SLOTS: int = 5
SLOT_ICONS: dict[int, str] = {1: "circle", 2: "diamond", 3: "hexagon", 4: "square", 5: "cross"}

#: Per-series export formats (D.2.10) and the result-level keys (D.5).
EXPORT_FORMATS: tuple[str, ...] = ("csv", "xlsx", "html", "json")
RESULT_EXPORT_KEYS: tuple[str, ...] = ("csv", "xlsx", "html", "json", "values_csv")

#: Notice levels (C.12.5).
NOTICE_LEVELS: tuple[str, ...] = ("notice", "warning", "info")

#: Level and default fix actions of the notices WP1 creates itself (C.12.5 "new notice codes" + ``horizon_long``).
#: Data-layer notices captured through ``data.collect_notices()`` take their level from the warning class instead
#: (DataNotice -> ``notice``, DataWarning -> ``warning``) and have no fixes.
NOTICE_DEFAULTS: dict[str, tuple[str, tuple[str, ...]]] = {
    "naver_missing_filled": ("warning", ()),
    "naver_low_resolution": ("warning", ("reanalyze_single",)),
    "mostly_zero_volume": ("warning", ("unit_month", "group_terms")),
    "data_short": ("warning", ("extend_range", "try_sample")),
    "horizon_clamped": ("notice", ()),
    "horizon_long": ("warning", ()),
    "gt_detected": ("notice", ()),
    "gt_lt1_half": ("notice", ()),
    "index_unavailable": ("info", ()),
    "log_hint": ("info", ("use_log",)),
    "naver_cached": ("info", ()),
    "gaps_filled": ("warning", ()),
}

#: source kinds whose series of one request already share one scale (D.4.3: no ``index`` scale)
_SAME_SCALE_KINDS = ("naver", "google_trends_upload")
_UPLOAD_KINDS = ("upload", "google_trends_upload")
_INDEX_KINDS = ("relative_index", "relative_index_gt")
#: ratio of the largest to the smallest series maximum above which a log axis is suggested (D.4.3)
_LOG_HINT_RATIO = 20
#: the narrative's weekday names (``narrative._seasonal_pattern``), plural in English ("peaks on Saturdays")
_WEEKDAYS_KO = ("월요일", "화요일", "수요일", "목요일", "금요일", "토요일", "일요일")
_WEEKDAYS_EN = ("Mondays", "Tuesdays", "Wednesdays", "Thursdays", "Fridays", "Saturdays", "Sundays")
#: Naver age codes -> (lowest, highest or None for "and over"), the base of the merged filter labels (D.2.2)
_AGE_RANGES: dict[str, tuple[int, Optional[int]]] = {
    "1": (0, 12), "2": (13, 18), "3": (19, 24), "4": (25, 29), "5": (30, 34), "6": (35, 39), "7": (40, 44),
    "8": (45, 49), "9": (50, 54), "10": (55, 59), "11": (60, None),
}
#: narrative dates (``narrative.format_date``): D/W ``2024-03-05``, Q ``2024-Q1``, M ``2024-03`` (D.2.9). ASCII word
#: boundaries, as in JS: ``2016-01부터`` must match (Python's Unicode ``\b`` sees no boundary before Hangul)
_DATE_RX = re.compile(r"\b\d{4}-\d{2}-\d{2}\b|\b\d{4}-Q[1-4]\b|\b\d{4}-\d{2}\b", re.ASCII)
#: a trailing parenthetical with test statistics in a phase reason (D.2.5)
_STATS_RX = re.compile(r"\s*\(([^()]*(?:τ|p\s*[<=>]|t\s*=|MK|Mann)[^()]*)\)\.?$")
#: changepoint magnitudes (relative to the series scale) of a large / medium shift (D.2.6)
_STRENGTH_EDGES = ((0.5, "large"), (0.2, "medium"))
#: how many changepoints get a badge on the chart (the largest by magnitude, D.2.6)
_CHART_CHANGEPOINTS = 5
#: momentum (%p per period) below this size is float noise or invisibly small: printed ``0%p`` on the tile and the
#: ranking (the narrative prints it ``0.00%p``), never ``−1.5×10⁻¹⁵%p`` (UX-5, CON-6)
_MOMENTUM_ZERO = 0.005
#: an anomaly whose actual and expected values differ by less than this share of the level is worded "slightly"
#: (``평소보다 약간``) -- a near noise-free series flags sub-percent deviations (UX-23)
_ANOMALY_SMALL_REL = 0.02
#: a flat outlook whose raw forecast window moves this much (%) against the current level gets the seasonal context
_SEASON_FLAT_PCT = 10.0
#: raw table values of this size are printed compact (만/억/조, K/M/B) in the forecast table, like the scenario boxes
_TABLE_COMPACT_FROM = 1e6
#: "최근 39주 동안 추세가 …" (the phase-judgement window, analysis.phase_window) and its English translation
_REASON_WINDOW_KO = re.compile(r"^(최근 [\d,]+(?:일|주|개월|분기|년)) (동안 추세)")
_REASON_WINDOW_EN = re.compile(r"^(Over the last [\d,]+ (?:days?|weeks?|months?|quarters?|years?))(,? the trend)")
#: rule-based narrative post-processing (UX-13, CON-9): the phase certainty is a word, never ``신뢰도 95%`` next to
#: ``상승 확률``/``예측 신뢰도``; MASE notes, model ids and test statistics live in the 자세히 blocks, not the body
_NARR_TENTATIVE_KO = re.compile(r"잠정 판단입니다\(신뢰도 [<>]?\d+%\)")
_NARR_TENTATIVE_KO2 = re.compile(r", 신뢰도 [<>]?\d+%\)")
_NARR_CONF_KO = re.compile(r"\((?:판단 )?신뢰도 [<>]?\d+%\)")
_NARR_TENTATIVE_EN = re.compile(r"tentative call \(confidence [<>]?\d+%\)")
_NARR_TENTATIVE_EN2 = re.compile(r"; confidence [<>]?\d+%\)")
_NARR_CONF_EN = re.compile(r"\(confidence [<>]?\d+%\)")
_NARR_DROP = (
    re.compile(r"\s*\(기술 참고: 백테스트 MASE[^()]*\)"),
    re.compile(r"\s*\(Technical note: backtest MASE[^()]*\)"),
    re.compile(r"\s*\(모델 ID: [^()]*\)"),
    re.compile(r"\s*\(model ids: [^()]*\)"),
    re.compile(r"\s*\([^()]*(?:τ\s*=|\bp\s*[<=>]\s*\d)[^()]*\)"),
)
_NARR_WORDS = (
    ("(잔차의 변동계수)", ""), (" (residual coefficient of variation)", ""),
    ("Probability of increase", "Chance of rise"), ("forecast confidence", "forecast reliability"),
    ("Forecast confidence", "Forecast reliability"),
)
#: 합니다체 endings of the engine's undefined-change sentences -> 해요체 (D.2.3: hero sentences are 해요체; CON-14)
_HAEYO = (("않았습니다.", "않았어요."), ("없습니다.", "없어요."), ("있습니다.", "있어요."), ("됩니다.", "돼요."),
          ("했습니다.", "했어요."), ("합니다.", "해요."))


# ---------------------------------------------------------------------------------------------------------------------
# Inputs built by analyze.py (WP1-server)
# ---------------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class SourceMeta:
    """Everything the D.2.2 ``source`` block, the source lines (VM, compare, CSV/XLSX, HTML paragraph, PNG footer)
    and the JSON export's ``source`` object need about where one series came from. Built by ``analyze.py`` once per
    series (language independent); the builders turn it into localized text.

    Attributes:
        kind: source kind, one of :data:`SOURCE_KINDS`. A Google Trends file uploaded through ``/api/upload`` is
            ``"google_trends_upload"`` (the series' ``attrs['trendai']['source']``), any other upload ``"upload"``.
        value_kind: one of :data:`VALUE_KINDS` -- ``naver`` -> ``relative_index``, ``google_trends_upload`` ->
            ``relative_index_gt``, ``wiki`` -> ``pageviews``, ``upload``/``sample`` -> ``raw``.
        freq: engine frequency of the prepared series, one of :data:`FREQS`.
        start: first **data** date ``YYYY-MM-DD`` of the prepared series (weekly data: the first bucket's Monday).
            Not the request bound (D.2.2).
        end: last data date ``YYYY-MM-DD`` (weekly data: the last bucket's start; ``fmt.range_label`` adds 6 days).
        as_of: UTC ISO-8601 with ``Z``. Naver/Wikipedia: the fetch time (a cache hit keeps the stored
            ``fetched_at``); uploads/samples: the analysis time (job start, app clock).
        terms: the D.2.1 ``terms`` -- Naver keywords of the group / ``(title,)`` / ``(column,)`` /
            ``(sample name,)`` / ``(Google Trends term,)``.
        naver_endpoint: ``"hub"`` | ``"legacy"`` for Naver (selects the label suffix, D.2.2), else ``None``.
        device: Naver filter ``"pc"`` | ``"mo"`` or ``None`` (no filter).
        gender: Naver filter ``"m"`` | ``"f"`` or ``None``.
        ages: Naver age codes ``"1"``..``"11"``, sorted numerically; ``()`` for no filter.
        filename: display filename of an upload (E.17: last path component, NFC, <= 100 chars), else ``None``.
        wiki_lang: Wikipedia language code (``wl``, ``^[a-z]{2,3}$``) for ``wiki``, else ``None``.
        wiki_title: the article title as requested (equals ``terms[0]``) for ``wiki``, else ``None``.
        unit_label: the user's unit label for uploads (<= 12 chars, axes/tooltips/``원래 단위 (…)``), else ``None``.
        gt_term: Google Trends term of this column (``GoogleTrendsInfo.terms[col][0]``), else ``None``.
        gt_region: Google Trends region of this column or ``None`` (worldwide / not stated / not GT).
        gt_category: Google Trends category text when the file stated one, else ``None``.
        synthetic: ``True`` for built-in samples (``samples.SYNTHETIC_ATTR``); drives the sample label and the
            ``가상 데이터예요…`` value-kind note. Must agree with ``narrative.is_synthetic(report.analysis)``.
    """

    kind: str
    value_kind: str
    freq: str
    start: str
    end: str
    as_of: str
    terms: tuple[str, ...]
    naver_endpoint: Optional[str] = None
    device: Optional[str] = None
    gender: Optional[str] = None
    ages: tuple[str, ...] = ()
    filename: Optional[str] = None
    wiki_lang: Optional[str] = None
    wiki_title: Optional[str] = None
    unit_label: Optional[str] = None
    gt_term: Optional[str] = None
    gt_region: Optional[str] = None
    gt_category: Optional[str] = None
    synthetic: bool = False


@dataclass(frozen=True)
class SeriesInput:
    """One successfully analysed series, as ``analyze.py`` hands it to the builders (E.6 step 3).

    Attributes:
        index: series index ``k`` (0-based request position, section 0).
        name: display name (E.6.1; unique within the request). Also ``report.name``.
        report: the ``TrendReport`` from ``pipeline.run(prepared, name=name, freq=f, horizon=h, lang="ko")``. Its
            ``analysis.series`` carries the raw series' ``attrs`` (E.6 step 2), so ``is_synthetic`` works.
        source: the series' :class:`SourceMeta`.
        narratives: rule-based narrative per language: ``{"ko": report.narrative, "en":
            narrative.generate_narrative(a, f, o, lang="en")[0]}``. Both keys required.
        notices: series-level notice objects per language, ``{"ko": [...], "en": [...]}``, already built with
            :func:`make_notice` / :func:`notice_from_warning` (``series_index == index``) -- the captured data-layer
            notices plus ``data_short``, ``mostly_zero_volume``, ``naver_missing_filled``, ``naver_low_resolution``
            and ``horizon_long`` (E.6 step 2). Copied into the VM's ``notices`` unchanged, in the given order.
        request_size: number of series in the normalised request (failed ones included). It is ``N`` for the slot
            rule (D.4.2) and the "multi-series request" condition of ``flags.low_resolution`` (D.2.1).
        ai_enabled: ``features.ai`` at job time (``fetchers.llm is not None`` and AI not disabled, C.2). With the
            source kind it decides ``narrative.ai_allowed`` / ``ai_block_reason`` (D.2.9).
        xlsx_available: whether XLSX export is possible (``exports.xlsx_available()`` at job time); decides the
            ``exports.unavailable.xlsx`` entry (D.2.10).
        slot: provisional slot, normally ``provisional_slot(index, request_size)`` (``None`` while ``N > 5``). The
            final slot is written with :func:`set_slot` at stage ``finish``.
    """

    index: int
    name: str
    report: TrendReport
    source: SourceMeta
    narratives: Mapping[str, str]
    notices: Mapping[str, Sequence[dict]]
    request_size: int
    ai_enabled: bool
    xlsx_available: bool
    slot: Optional[int] = None


@dataclass(frozen=True)
class ExportAvailability:
    """Which exports exist for a result (stored in ``result.json`` as ``available``, E.7) and why the others do not.

    Attributes:
        series: per-series flags ``{"csv", "xlsx", "html", "json"} -> bool`` (the same for every series of a result,
            since they depend only on the source kind and openpyxl).
        result: result-level flags ``{"csv", "xlsx", "html", "json", "values_csv"} -> bool`` (D.5).
        reasons: message code per unavailable key, e.g. ``{"html": "export_naver_raw", "xlsx":
            "export_xlsx_unavailable", "values_csv": "export_naver_raw"}``. A key that is unavailable *without* a
            reason (``values_csv`` of a single-series result) is absent here.
    """

    series: Mapping[str, bool]
    result: Mapping[str, bool]
    reasons: Mapping[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------------------------------------------------

def _lang(lang: str) -> str:
    return "en" if lang == "en" else "ko"


#: ``float(x)`` for a finite real number (bools excluded), else ``None`` -- one rule with the formatters (A11Y-24)
_finite = fmt._finite


def _num(x: Any) -> Optional[float]:
    """A JSON number: 15 significant digits (the ``pipeline._clean`` rule), ``None`` for missing / non-finite."""
    v = _finite(x)
    return None if v is None else float(f"{v:.15g}")


def _nums(values: Any) -> list[Optional[float]]:
    import numpy as np

    return [_num(v) for v in np.asarray(values, dtype=float).ravel()]


def _iso(ts: Any) -> str:
    """``YYYY-MM-DD`` of a timestamp without ``strftime`` (years after 9999 are possible for long horizons)."""
    import pandas as pd

    t = pd.Timestamp(ts)
    return f"{t.year:04d}-{t.month:02d}-{t.day:02d}"


def _isos(index: Any) -> list[str]:
    return [_iso(d) for d in index]


def _render(code: str, params: Optional[Mapping[str, Any]], lang: str) -> str:
    from .. import messages

    return messages.render_web(code, dict(params or {}), _lang(lang))


def _tile(key: str, lang: str, value: str, *, sub: str, tone: str = "neutral", icon: Optional[str] = None,
          tooltip: Optional[str] = None, undefined_reason: Optional[str] = None,
          value_raw: Optional[float] = None) -> dict:
    return {"key": key, "label": T.t(T.TILE_LABELS, key, lang), "value": value, "icon": icon, "tone": tone,
            "sub": sub, "tooltip": tooltip if tooltip is not None else T.t(T.TILE_HELP, key, lang),
            "undefined_reason": undefined_reason or None, "value_raw": _num(value_raw)}


# ---------------------------------------------------------------------------------------------------------------------
# Notices and fix objects (shared by analyze.py, errors.py and the compare builder; one shape everywhere)
# ---------------------------------------------------------------------------------------------------------------------

def fix_object(action: str, params: Optional[Mapping[str, Any]] = None) -> dict:
    """A fix action object as used in error envelopes (C.12.1) and notices (C.12.5):
    ``{"action": action, "label_ko": messages.fix_label(action, "ko"), "label_en": messages.fix_label(action, "en"),
    "params": dict(params or {})}``. Raises ``KeyError`` for an action without a ``fix_<action>`` message."""
    from .. import messages

    return {"action": action, "label_ko": messages.fix_label(action, "ko"),
            "label_en": messages.fix_label(action, "en"), "params": dict(params or {})}


def make_notice(code: str, lang: str, *, params: Optional[Mapping[str, Any]] = None, level: Optional[str] = None,
                series_index: Optional[int] = None, fixes: Optional[Sequence[dict]] = None,
                source_kind: Optional[str] = None) -> dict:
    """A notice WP1 creates itself (C.12.5): ``{"code", "level", "text", "series_index", "fixes"}``.

    Args:
        code: message code (must exist in ``messages.MESSAGES``).
        lang: ``"ko"`` | ``"en"``; ``text = messages.render_web(code, params, lang)``.
        params: template parameters (only those listed for the code in C.12.5).
        level: overrides ``NOTICE_DEFAULTS[code][0]``; required for codes not in :data:`NOTICE_DEFAULTS`.
        series_index: ``k`` for a series-level notice, ``None`` for a result-level one.
        fixes: explicit fix objects (:func:`fix_object`); ``None`` -> the defaults of ``NOTICE_DEFAULTS[code][1]``
            (actions that need params, e.g. ``reanalyze_single`` with ``term``, must be passed explicitly).
        source_kind: when ``"upload"``/``"google_trends_upload"``, a default ``extend_range`` fix is replaced by
            ``reupload`` (the period of an upload cannot change, C.12.1).
    """
    default_level, default_actions = NOTICE_DEFAULTS.get(code, (None, ()))
    level = level or default_level
    if level not in NOTICE_LEVELS:
        raise ValueError(f"notice {code!r} needs a level (one of {NOTICE_LEVELS})")
    if fixes is None:
        actions = list(default_actions)
        if source_kind in _UPLOAD_KINDS:
            actions = ["reupload" if a == "extend_range" else a for a in actions]
        actions = list(dict.fromkeys(actions))
        fix_list = [fix_object(a) for a in actions]
    else:
        fix_list = [copy.deepcopy(dict(f)) for f in fixes]
    return {"code": code, "level": level, "text": _notice_text(code, params, lang),
            "series_index": series_index, "fixes": fix_list[:3]}


def _notice_text(code: str, params: Optional[Mapping[str, Any]], lang: str) -> str:
    """``messages.render_web`` text; ``horizon_long`` with a ``freq`` param names the horizon with the page's span
    word (``예측 기간(36개월)``) instead of the catalog's ``36기간`` / ``36 periods`` (D.1.6; CON-14)."""
    text = _render(code, params, lang)
    p = dict(params or {})
    if code == "horizon_long" and p.get("freq") in FREQS and _finite(p.get("horizon")) is not None:
        h = int(p["horizon"])
        generic = f"({h:,} periods)" if _lang(lang) == "en" else f"({h:,}기간)"
        text = text.replace(generic, f"({fmt.span_label(h, str(p['freq']), _lang(lang))})", 1)
    return text


def notice_from_warning(warning: Warning, lang: str, *, series_index: Optional[int] = None) -> dict:
    """A data-layer notice captured by ``data.collect_notices()`` -> notice object (C.12.5).

    ``text = messages.render_web_msg(warning.args[0], lang)`` (every part of a joined message), ``code`` = the code of
    the first part, ``level`` = ``"warning"`` for a DataWarning, ``"notice"`` for a DataNotice, ``fixes = []``."""
    from .. import messages
    from ..data import DataNotice

    m = warning.args[0] if len(getattr(warning, "args", ())) >= 1 else None
    code = getattr(m, "code", None) if isinstance(m, messages.Msg) else None
    return {"code": str(code or "data_notice"),
            "level": "notice" if isinstance(warning, DataNotice) else "warning",
            "text": messages.render_web_msg(m if isinstance(m, messages.Msg) else warning, _lang(lang)),
            "series_index": series_index, "fixes": []}


# ---------------------------------------------------------------------------------------------------------------------
# Source block and lines (D.2.2, D.4.1)
# ---------------------------------------------------------------------------------------------------------------------

def _source_label(meta: SourceMeta, lang: str) -> str:
    kind = meta.kind
    if kind == "naver":
        return T.t(T.SOURCE_LABELS, "naver_legacy" if meta.naver_endpoint == "legacy" else "naver", lang)
    if kind == "wiki":
        code = meta.wiki_lang or "ko"
        name = T.t(T.WIKI_LANG_NAMES, code, lang) if code in T.WIKI_LANG_NAMES else code
        return T.t(T.SOURCE_LABELS, "wiki", lang, wiki_lang=name)
    if kind in _UPLOAD_KINDS:
        if meta.filename:
            return T.t(T.SOURCE_LABELS, kind, lang, filename=meta.filename)
        return T.t(T.SOURCE_LABELS, f"{kind}_unnamed", lang)
    return T.t(T.SOURCE_LABELS, "sample", lang)


def _value_kind_label(value_kind: str, unit_label: Optional[str], lang: str) -> str:
    label = T.t(T.VALUE_KIND_LABELS, value_kind if value_kind in T.VALUE_KIND_LABELS else "raw", lang)
    if value_kind == "raw" and unit_label:
        label += T.t(T.SOURCE_LINE_TEXTS, "unit_suffix", lang, unit_label=unit_label)
    return label


def _value_kind_note(meta: SourceMeta, lang: str) -> str:
    if meta.kind == "sample" or (meta.synthetic and meta.value_kind == "raw"):
        return T.t(T.VALUE_KIND_NOTES, "sample", lang)
    if meta.value_kind == "relative_index":
        unit_word = T.t(T.INDEX_UNIT_WORDS, meta.freq if meta.freq in T.INDEX_UNIT_WORDS else "W", lang)
        return T.t(T.VALUE_KIND_NOTES, "relative_index", lang, unit_word=unit_word)
    return T.t(T.VALUE_KIND_NOTES, meta.value_kind if meta.value_kind in T.VALUE_KIND_NOTES else "raw", lang)


def _age_label(codes: Sequence[str], lang: str) -> str:
    """Age filter label with consecutive codes merged (``4,5`` -> ``25–34세``; ``2,5`` -> ``13–18세·30–34세``)."""
    known = sorted({str(c) for c in codes if str(c) in _AGE_RANGES}, key=int)
    runs: list[list[str]] = []
    for c in known:
        if runs and int(c) == int(runs[-1][-1]) + 1:
            runs[-1].append(c)
        else:
            runs.append([c])
    parts = []
    for run in runs:
        if len(run) == 1:
            parts.append(T.t(T.NAVER_AGE_LABELS, run[0], lang))
            continue
        lo, hi = _AGE_RANGES[run[0]][0], _AGE_RANGES[run[-1]][1]
        if lang == "en":
            parts.append(f"{lo}+" if hi is None else f"{lo}–{hi}")
        else:
            parts.append(f"{lo}세 이상" if hi is None else f"{lo}–{hi}세")
    return T.t(T.SOURCE_LINE_TEXTS, "filters_sep", lang).join(parts)


def _filters_label(meta: SourceMeta, lang: str) -> str:
    if meta.kind != "naver":
        return ""
    parts = []
    if meta.device in T.FILTER_LABELS:
        parts.append(T.t(T.FILTER_LABELS, meta.device, lang))
    if meta.gender in T.FILTER_LABELS:
        parts.append(T.t(T.FILTER_LABELS, meta.gender, lang))
    ages = _age_label(meta.ages, lang)
    if ages:
        parts.append(ages)
    return T.t(T.SOURCE_LINE_TEXTS, "filters_sep", lang).join(parts)


def _as_of_part(kind: str, as_of_label: str, lang: str) -> str:
    if kind in ("naver", "wiki"):
        return T.t(T.SOURCE_LINE_TEXTS, "as_of_data", lang, as_of=as_of_label)
    if kind in _UPLOAD_KINDS:
        return T.t(T.SOURCE_LINE_TEXTS, "as_of_analysed", lang, as_of=as_of_label)
    return ""


def _line_from_parts(kind: str, label: str, freq_label: str, filters_label: str, range_label: str,
                     value_kind_label: str, as_of_label: str, lang: str, with_filters: bool = True) -> str:
    """The D.2.2 parts in order: label · freq · filters (Naver) · range · value kind · as-of (samples: label ·
    freq · range)."""
    parts = [label, freq_label]
    if kind == "naver" and with_filters:
        parts.append(filters_label)
    parts.append(range_label)
    if kind != "sample":
        parts.append(value_kind_label)
        parts.append(_as_of_part(kind, as_of_label, lang))
    sep = T.t(T.SOURCE_LINE_TEXTS, "parts_sep", lang)
    return T.t(T.SOURCE_LINE_TEXTS, "prefix", lang) + sep.join(p for p in parts if p)


def build_source_block(meta: SourceMeta, lang: str) -> dict:
    """The D.2.2 ``source`` object in ``lang``: ``kind, label, short, line, value_kind, value_kind_label,
    value_kind_note, unit_label, freq, freq_label, start, end, range_label, as_of, as_of_label, filters, filters_label,
    filename, attribution``. ``filters`` is ``{"device", "gender", "ages"}`` for Naver (``None``/``[]`` when unset)
    and ``None`` for other kinds. The same object is the JSON export's top-level ``source`` (C.10)."""
    lang = _lang(lang)
    label = _source_label(meta, lang)
    freq_label = fmt.freq_label(meta.freq, lang)
    range_text = fmt.range_label(meta.start, meta.end, meta.freq, lang)
    as_of_label = fmt.datetime_label(meta.as_of, lang)
    vk_label = _value_kind_label(meta.value_kind, meta.unit_label, lang)
    filters_label = _filters_label(meta, lang)
    prefix = T.t(T.SOURCE_LINE_TEXTS, "prefix", lang)
    return {
        "kind": meta.kind,
        "label": label,
        "short": T.t(T.SOURCE_SHORT, meta.kind if meta.kind in T.SOURCE_SHORT else "upload", lang),
        "line": _line_from_parts(meta.kind, label, freq_label, filters_label, range_text, vk_label, as_of_label,
                                 lang),
        "value_kind": meta.value_kind,
        "value_kind_label": vk_label,
        "value_kind_note": _value_kind_note(meta, lang),
        "unit_label": meta.unit_label,
        "freq": meta.freq,
        "freq_label": freq_label,
        "start": meta.start,
        "end": meta.end,
        "range_label": range_text,
        "as_of": meta.as_of,
        "as_of_label": as_of_label,
        "filters": ({"device": meta.device, "gender": meta.gender, "ages": list(meta.ages)}
                    if meta.kind == "naver" else None),
        "filters_label": filters_label,
        "filename": meta.filename,
        "attribution": prefix + label,
    }


def source_line(block: Mapping[str, Any], lang: str, *, with_filters: bool = True) -> str:
    """``"출처: "``/``"Source: "`` + the D.2.2 parts of a built source block, joined with ``" · "``.

    ``with_filters=True`` gives the VM's ``source.line``. ``with_filters=False`` drops the filters part: that is the
    first comment line of a series CSV and the XLSX ``출처`` row, where the filters go to the condition line instead
    (C.10 example)."""
    lang = _lang(lang)
    return _line_from_parts(str(block.get("kind")), str(block.get("label") or ""),
                            str(block.get("freq_label") or ""), str(block.get("filters_label") or ""),
                            str(block.get("range_label") or ""), str(block.get("value_kind_label") or ""),
                            str(block.get("as_of_label") or ""), lang, with_filters=with_filters)


def compare_source_line(metas: Sequence[SourceMeta], lang: str) -> str:
    """``compare.source_line`` (D.4.1): the label and parts shared by the compared (successful) series in the D.2.2
    order -- label · freq · filters · range (earliest start ~ latest end) · value kind · as-of (the **oldest**
    ``as_of``). ``metas`` are in request order. Used under the compare chart, as the first comment line of the
    whole-result CSVs, in the compare HTML source paragraph and in the compare PNG footer."""
    lang = _lang(lang)
    if not metas:
        return T.t(T.SOURCE_LINE_TEXTS, "prefix", lang).rstrip()
    first = metas[0]

    def shared(values: Sequence[str]) -> str:
        return values[0] if len(set(values)) == 1 else ""

    labels = [_source_label(m, lang) for m in metas]
    freqs = [m.freq for m in metas]
    freq = shared(freqs)
    range_freq = freq or first.freq
    start = min(m.start for m in metas)
    end = max(m.end for m in metas)
    as_of = min(metas, key=lambda m: _as_of_key(m.as_of)).as_of
    return _line_from_parts(
        first.kind if len({m.kind for m in metas}) == 1 else "upload",
        shared(labels),
        fmt.freq_label(freq, lang) if freq else "",
        shared([_filters_label(m, lang) for m in metas]),
        fmt.range_label(start, end, range_freq, lang),
        shared([_value_kind_label(m.value_kind, m.unit_label, lang) for m in metas]),
        fmt.datetime_label(as_of, lang),
        lang,
    )


def _as_of_key(as_of: str) -> str:
    """Sort key of a UTC ``…Z`` timestamp (ISO strings of one format sort chronologically)."""
    return str(as_of).replace("z", "Z")


# ---------------------------------------------------------------------------------------------------------------------
# Series facts shared by the series VM, the ranking rows and the exports
# ---------------------------------------------------------------------------------------------------------------------

class _Facts:
    """Everything derived once from a report for one language (engine wording via the E.18 helpers)."""

    def __init__(self, report: TrendReport, meta: SourceMeta, lang: str) -> None:
        from .. import narrative as N
        from .. import report as R
        from ..forecast import outlook_levels

        self.report, self.meta, self.lang = report, meta, lang
        a, f, o = report.analysis, report.forecast, report.outlook
        self.a, self.f, self.o = a, f, o
        self.freq = str(a.freq)
        self.kind = meta.kind
        self.value_kind = meta.value_kind
        self.d = R._ChartData(report)
        fut_b, cur_b = R._change_basis_en(report, self.d) if lang == "en" else R._change_basis(report, self.d)
        self.fut_b, self.cur_b = fut_b, cur_b
        self.basis = T.t(T.HERO_TEXTS, "basis", lang, fut=fut_b, cur=cur_b)
        try:
            levels = outlook_levels(a, f)
        except (AttributeError, TypeError, ValueError):
            levels = None
        undefined = getattr(levels, "undefined", None)
        self.word = N.direction_label(o.direction, o.expected_change_pct, lang, undefined, levels=levels)
        word_ko = N.direction_label(o.direction, o.expected_change_pct, "ko", undefined, levels=levels)
        self.uncertain = word_ko == "방향 불확실"
        self.flat_word = word_ko == "보합"
        self.change = _finite(o.expected_change_pct)
        self.nochange = R._change_undefined(report, self.d, lang) if self.change is None else None
        self.prob_why = N.prob_undefined(a)
        self.prob = None if self.prob_why is not None else _finite(o.prob_increase)
        reason = N.prob_reason(a, lang) if self.prob is None else None
        self.prob_reason = reason.short if reason is not None else (
            T.t(T.TILE_TEXTS, "not_computed", lang) if self.prob is None else None)
        self.skill = _finite(f.skill)
        self.mase = _finite(f.backtest_mase)
        if self.skill is not None and self.skill >= 1.0:
            self.display = "bands_only"
        elif o.confidence == "low" or self.skill is None:
            self.display = "weak"
        else:
            self.display = "normal"
        self.tentative = bool(N.phase_is_tentative(a))
        self.peak = _peak_season(a, lang)
        self.n = int(len(a.series))
        self.horizon = int(len(f.mean)) if len(f.mean) else int(f.horizon)
        self.h_label = fmt.span_label(self.horizon, self.freq, lang)
        self.fc_dates = _isos(f.mean.index)
        self.phase_name = T.t(T.PHASE_SHORT, a.phase.label, lang) if a.phase.label in T.PHASE_SHORT \
            else str(a.phase.label)
        self.adjust_note, self.season_shift = self._seasonal_adjustment()

    # ---- seasonally adjusted outlook (FU-3) ------------------------------------------------------------------------
    def _seasonal_adjustment(self) -> tuple[Optional[str], Optional[str]]:
        """``(adjust_note, season_shift)``.

        ``adjust_note``: why ``계절 보정한`` stands in the basis (a window covers only part of a season, so the engine
        removed that window's average seasonal swing; ``report._adjustment_note`` in 해요체 / English), else ``None``.
        ``season_shift``: ``"low"`` / ``"high"`` when the comparison is adjusted and the *raw* forecast window moves
        against the call shown -- an up call while the drawn forecast falls into the off-season (``"low"``), a down
        call while it climbs into the season (``"high"``), or a flat call while it moves by 10 % or more -- else
        ``None``. The hero, the summary line and the rank card then say so, so the headline and the falling dashed
        line no longer contradict each other."""
        import numpy as np

        from .. import report as R

        try:
            k_fut, k_cur, _h, period, adj_fut, adj_cur = R._outlook_windows(self.report, self.d)
        except Exception:  # noqa: BLE001 - hand-made / partial reports: no adjustment to explain
            return None, None
        if period is None or not (adj_fut or adj_cur):
            return None, None
        lang = self.lang
        def span(k: int) -> str:            # ko 26주; en 26-week (an adjective before "window")
            if lang != "en":
                return fmt.span_label(k, self.freq, lang)
            return fmt.span_label(1, self.freq, "en").replace("1 ", f"{k:,}-", 1)

        spans = list(dict.fromkeys(span(k) for k, adj in ((k_fut, adj_fut), (k_cur, adj_cur)) if adj))
        note = T.t(T.SEASON_TEXTS, "adjust_note", lang,
                   spans=T.t(T.SEASON_TEXTS, "spans_sep", lang).join(spans),
                   window=T.t(T.SEASON_TEXTS, "window_one" if len(spans) == 1 else "window_many", lang),
                   period=fmt.span_label(period, self.freq, lang))
        if self.change is None or k_fut < 1 or k_cur < 1:
            return note, None
        fut = np.asarray(self.f.mean, dtype=float)[-k_fut:]
        cur = np.asarray(self.a.series, dtype=float)[-k_cur:]
        fut, cur = fut[np.isfinite(fut)], cur[np.isfinite(cur)]
        if not len(fut) or not len(cur):
            return note, None
        fut_raw, cur_raw = float(np.mean(fut)), float(np.mean(cur))
        if not cur_raw > 0:
            return note, None
        raw_pct = (fut_raw - cur_raw) / cur_raw * 100.0
        direction = str(self.o.direction)
        shift = None
        if direction == "up" and not (self.flat_word or self.uncertain) and raw_pct < 0:
            shift = "low"
        elif direction == "down" and not (self.flat_word or self.uncertain) and raw_pct > 0:
            shift = "high"
        elif (self.flat_word or self.uncertain or direction == "flat") and abs(raw_pct) >= _SEASON_FLAT_PCT:
            shift = "low" if raw_pct < 0 else "high"
        return note, shift

    def season_context(self) -> Optional[str]:
        """The second hero sentence for :attr:`season_shift` (``None`` without one)."""
        if self.season_shift is None:
            return None
        lang = self.lang
        if self.flat_word:
            flow = "flow_flat"
        elif self.uncertain or str(self.o.direction) not in ("up", "down"):
            flow = "flow_uncertain"
        else:
            flow = "flow_up" if str(self.o.direction) == "up" else "flow_down"
        peak = ""
        if self.peak is not None and not self.peak["weekday"]:
            peak = T.t(T.SEASON_TEXTS, "peak", lang, peak=self.peak["peak"])
        return T.t(T.SEASON_TEXTS, f"context_{self.season_shift}", lang, h=self.h_label,
                   topic=_topic_particle(self.h_label), peak=peak, flow=T.t(T.SEASON_TEXTS, flow, lang))

    def season_short(self) -> str:
        """`` · 앞으로 계절적 비수기`` / `` · seasonal low ahead`` for the summary line and the rank card, or ``""``."""
        if self.season_shift is None:
            return ""
        return T.t(T.SEASON_TEXTS, f"short_{self.season_shift}", self.lang)

    # ---- tile 2 / ranking: recent change -------------------------------------------------------------------------
    def recent_basis(self) -> str:
        """``전년 동기 대비`` (M, Q, W with a full-season lag), ``전년 대비`` (Y), else ``{lag}{unit} 전 대비``."""
        from .. import report as R
        from ..data import season_length

        lag = R._recent_lag(self.report, self.n)
        season = season_length(self.freq) if self.freq in FREQS else 0
        if self.freq in ("M", "Q", "W") and lag == season:
            return T.t(T.TILE_TEXTS, "recent_yoy", self.lang)
        if self.freq == "Y" and lag == 1:
            return T.t(T.TILE_TEXTS, "recent_prev_year", self.lang)
        return T.t(T.TILE_TEXTS, "recent_lag", self.lang, span=fmt.span_label(lag, self.freq, self.lang))

    def recent_span(self) -> str:
        from .. import report as R

        return fmt.span_label(R._recent_lag(self.report, self.n), self.freq, self.lang)

    def recent_undefined(self) -> Optional[str]:
        from .. import narrative as N
        from .. import report as R

        if _finite(self.a.metrics.recent_change_pct) is not None:
            return None
        code = N.metric_pct_undefined(self.a).get("recent_change_pct")
        return R._pct_why(code, "recent_change_pct", self.lang) or T.t(T.TILE_TEXTS, "not_computed", self.lang)

    # ---- tile 3 / ranking: momentum --------------------------------------------------------------------------------
    def momentum_undefined(self) -> Optional[str]:
        from .. import narrative as N
        from .. import report as R

        code = N.momentum_pct_undefined(self.a)
        if code is None and _finite(self.a.metrics.momentum) is not None:
            return None
        return R._rate_why(self.report, code, self.lang) or T.t(T.TILE_TEXTS, "not_computed", self.lang)

    def momentum_text(self) -> str:
        if self.momentum_undefined() is not None:
            return fmt.DASH
        unit = T.t(T.MOMENTUM_UNITS, self.freq, self.lang) if self.freq in T.MOMENTUM_UNITS else ""
        mom = _finite(self.a.metrics.momentum)
        if mom is not None and abs(mom) < _MOMENTUM_ZERO:     # float noise / invisibly small: 0%p, as the narrative
            mom = 0.0
        return fmt.pp(mom, self.lang) + unit

    def momentum_word(self) -> str:
        import numpy as np

        from ..data import periods_per_year

        if self.tentative:
            return T.t(T.MOMENTUM_WORDS, "tentative", self.lang)
        if self.a.phase.label in ("stable", "volatile"):
            return T.t(T.MOMENTUM_WORDS, "unclear", self.lang)
        mom = _finite(self.a.metrics.momentum) or 0.0
        try:
            eps = 1.0 / periods_per_year(self.freq)
        except Exception:  # noqa: BLE001 - unknown freq code: monthly
            eps = 1.0 / 12.0
        trend = np.asarray(self.a.decomposition.trend, dtype=float)
        n = len(trend)
        recent = 0.0
        if n >= 2:
            season = max(1, int(self.a.season_length or 1))
            r = max(1, min(max(3, min(season, n // 4)), n - 1))
            diff = trend[-1] - trend[-1 - r]
            recent = float(np.sign(diff)) if math.isfinite(diff) else 0.0
        if mom > eps:
            key = "accelerating" if recent > 0 else "decline_easing"
        elif mom < -eps:
            key = "falling_faster" if recent < 0 else "slowing"
        else:
            key = "steady"
        return T.t(T.MOMENTUM_WORDS, key, self.lang)

    # ---- expected change -------------------------------------------------------------------------------------------
    def change_reason(self) -> Optional[str]:
        if self.change is not None:
            return None
        if self.nochange is not None and self.nochange.short:
            return self.nochange.short
        return T.t(T.TILE_TEXTS, "not_computed", self.lang)

    def likelihood(self) -> Optional[str]:
        return fmt.likelihood(self.prob, self.lang) if self.prob is not None else None

    def arrow_tone(self) -> tuple[str, str]:
        direction = str(self.o.direction)
        if direction == "up":
            return "▲", "up"
        if direction == "down":
            return "▼", "down"
        return "▶", "neutral"


# ---------------------------------------------------------------------------------------------------------------------
# Peak season (D.2.4): the narrative's ``_seasonal_pattern`` condition and grouping, plus the rise start
# ---------------------------------------------------------------------------------------------------------------------

def _peak_season(a: Any, lang: str) -> Optional[dict]:
    """``{"peak", "trough", "start", "weekday"}`` names, or ``None`` when no peak season is shown."""
    import numpy as np
    import pandas as pd

    dec = a.decomposition
    try:
        period = int(dec.period)
        strength = float(a.metrics.seasonal_strength)
    except (AttributeError, TypeError, ValueError):
        return None
    if period <= 1 or not math.isfinite(strength) or strength < 0.3:
        return None
    s = dec.seasonal
    if len(s) < 2 or not isinstance(s.index, pd.DatetimeIndex):
        return None
    vals = pd.Series(np.asarray(s, dtype=float), index=s.index)
    vals = vals[np.isfinite(vals.to_numpy())]
    if len(vals) < 2:
        return None
    freq = str(a.freq)
    if freq in ("M", "W"):
        key, kind = vals.index.month, "month"
    elif freq == "Q":
        key, kind = vals.index.quarter, "quarter"
    elif freq == "D" and period == 7:
        key, kind = vals.index.dayofweek, "weekday"
    else:
        return None
    g = vals.groupby(key).mean()
    prof = g.to_numpy(dtype=float)
    if len(g) < 2 or not np.isfinite(prof).all() or float(prof.max() - prof.min()) <= 0:
        return None
    hi, lo = int(np.argmax(prof)), int(np.argmin(prof))
    start = hi
    for _ in range(len(prof) - 1):          # walk back while the previous group is strictly lower
        prev = (start - 1) % len(prof)
        if prof[prev] < prof[start]:
            start = prev
        else:
            break
    keys = [int(k) for k in g.index]

    def name(pos: int) -> str:
        k = keys[pos]
        if kind == "month":
            return T.t(T.PEAK_TEXTS, "month", lang, n=k, abbr=fmt.MONTHS_EN[k - 1])
        if kind == "quarter":
            return T.t(T.PEAK_TEXTS, "quarter", lang, n=k)
        return (_WEEKDAYS_EN if lang == "en" else _WEEKDAYS_KO)[k]

    return {"peak": name(hi), "trough": name(lo), "start": name(start), "weekday": kind == "weekday"}


# ---------------------------------------------------------------------------------------------------------------------
# Hero (D.2.3)
# ---------------------------------------------------------------------------------------------------------------------

def _outlook_line(direction: str, word: str, prob: Optional[float], reason: Optional[str], lang: str) -> str:
    """The hero/ranking outlook line (D.2.3): the number in brackets is always P(rise) and says so."""
    lang = _lang(lang)
    arrow = {"up": "▲", "down": "▼"}.get(direction, "▶")
    word = _cap(word, lang)                 # English: the first word after the arrow starts the line (CON-14)
    if prob is None:
        return T.t(T.HERO_TEXTS, "line_not_computed", lang, arrow=arrow, word=word, reason=reason or "")
    like = fmt.likelihood(prob, lang)
    side = "up" if prob >= 0.55 else ("down" if prob <= 0.45 else None)
    labelled = fmt.prob_labelled(prob, lang)
    if side is not None and side == direction:
        return T.t(T.HERO_TEXTS, "line_matching", lang, arrow=arrow, likelihood=like, prob=labelled)
    return T.t(T.HERO_TEXTS, "line_full", lang, arrow=arrow, word=word, likelihood=like, prob=labelled)


def _accuracy_line(skill: Optional[float], mase: Optional[float], lang: str) -> str:
    """The narrative's accuracy thresholds as one short line (D.2.3)."""
    if skill is not None:
        skill = max(0.0, skill)
        if skill <= 0.95:
            gain = (1.0 - skill) * 100.0
            if gain > 99.5:
                return T.t(T.ACCURACY_TEXTS, "better_99", lang)
            return T.t(T.ACCURACY_TEXTS, "better", lang, g=str(fmt.round_half_away(gain, 0)))
        if skill <= 1.0:
            return T.t(T.ACCURACY_TEXTS, "similar", lang)
        return T.t(T.ACCURACY_TEXTS, "worse", lang)
    if mase is not None:
        return T.t(T.ACCURACY_TEXTS, "no_naive", lang)
    return T.t(T.ACCURACY_TEXTS, "too_short", lang)


def _hero(fx: _Facts) -> dict:
    lang = fx.lang
    arrow, tone = fx.arrow_tone()
    direction = str(fx.o.direction)
    prob = fx.prob
    outlook = {
        "direction": direction, "arrow": arrow, "tone": tone, "word": fx.word,
        "likelihood": fx.likelihood(),
        "prob": _num(prob),
        "prob_text": fmt.prob_text(prob, lang),
        "prob_exact": fmt.prob_exact(prob, lang),
        "prob_undefined_reason": fx.prob_reason if prob is None else None,
        "line": _outlook_line(direction, fx.word, prob, fx.prob_reason, lang),
    }
    change_undefined = None
    if fx.change is not None:
        more = T.t(T.HERO_TEXTS, "higher" if fx.change > 0 else "lower", lang)
        if fx.flat_word:
            sentence = T.t(T.HERO_TEXTS, "sentence_flat", lang, fut=fx.fut_b, cur=fx.cur_b, pct=fmt.pct(fx.change))
        elif fx.uncertain:
            sentence = T.t(T.HERO_TEXTS, "sentence_uncertain", lang, fut=fx.fut_b, cur=fx.cur_b,
                           pct=fmt.pct_abs(fx.change), more=more)
        else:
            if direction in ("up", "down"):
                more = T.t(T.HERO_TEXTS, "higher" if direction == "up" else "lower", lang)
            sentence = T.t(T.HERO_TEXTS, "sentence_change", lang, fut=fx.fut_b, cur=fx.cur_b,
                           pct=fmt.pct_abs(fx.change), more=more)
    else:
        nc = fx.nochange
        if nc is not None:
            head = _haeyo(nc.sentence) if lang == "ko" else nc.sentence
            sentence = head + (f" ({nc.levels})" if nc.levels else "")
            change_undefined = {"code": nc.code, "reason": nc.short, "levels": nc.levels or None}
        else:   # pragma: no cover - _change_undefined always explains a missing change
            sentence = fx.change_reason() or ""
            change_undefined = {"code": None, "reason": fx.change_reason(), "levels": None}
    context = fx.season_context()
    if context:                             # FU-3: the drawn forecast moves against the adjusted call
        sentence = f"{sentence} {context}"
    confidence = str(fx.o.confidence) if fx.o.confidence in T.RELIABILITY_LABELS else "low"
    return {
        "title": fx.report.name,
        "outlook": outlook,
        "sentence": sentence,
        "basis": fx.basis,
        "change_undefined": change_undefined,
        "reliability": {"level": confidence, "label": T.t(T.RELIABILITY_LABELS, confidence, lang)},
        "accuracy_line": _accuracy_line(fx.skill, fx.mase, lang),
        "reference_note": (T.t(T.REFERENCE_NOTES, _reference_key(fx), lang) if fx.display != "normal" else None),
    }


def _reference_key(fx: _Facts) -> str:
    """``REFERENCE_NOTES`` key: ``bands_only``; ``weak`` (low reliability) or ``weak_unverified`` (no backtest skill
    although the reliability is not low -- "신뢰도가 낮아" would contradict the chip, CON-5)."""
    if fx.display == "weak" and fx.skill is None and str(fx.o.confidence) != "low":
        return "weak_unverified"
    return fx.display


# ---------------------------------------------------------------------------------------------------------------------
# Phase (D.2.5)
# ---------------------------------------------------------------------------------------------------------------------

def _reason_item(text: str, freq: str, lang: str, kind: str = "support") -> dict:
    """A phase reason: ISO dates rewritten (no links), trailing test statistics moved to ``detail``, the window of
    the "최근 N 동안 추세가 …" reason labelled as the phase-judgement window (FU-7) and ``kind`` ``"support"`` or
    ``"caveat"`` (a reason that argues against the call, rendered without the ✓; UX-7)."""
    text, _ = localize_dates(text, freq, lang, ())
    label = T.t(T.PHASE_TEXTS, "window_label", lang)
    if lang == "en":
        text = _REASON_WINDOW_EN.sub(lambda m: m.group(1) + label + m.group(2), text, count=1)
    else:
        text = _REASON_WINDOW_KO.sub(lambda m: m.group(1) + label + " " + m.group(2), text, count=1)
    m = _STATS_RX.search(text)
    if m is None:
        return {"text": text, "detail": None, "kind": kind}
    head = text[:m.start()].rstrip()
    if m.group(0).endswith(".") and not head.endswith("."):
        head += "."
    return {"text": head, "detail": m.group(1).strip(), "kind": kind}


def _reason_kind(text: str, label: str) -> str:
    """``"caveat"`` for a phase reason that qualifies the call instead of supporting it (D.2.5, UX-7): the engine's
    rule (``narrative.phase_reason_kind``) -- the random-walk or short-data sentence beside a directed phase, a
    trend test with no (or the opposite) direction for a growing / declining phase. Judged on the Korean original,
    so the English translation inherits it."""
    from .. import narrative as N

    return N.phase_reason_kind(str(text), label)


def _phase(fx: _Facts, hero: dict) -> dict:
    from .. import narrative as N
    from ..types import PHASE_KO

    lang = fx.lang
    ph = fx.a.phase
    label = str(ph.label)
    if lang == "en":
        pairs = [(r2, _reason_kind(r, label)) for r2, r in ((N.translate_reason(r), r) for r in ph.reasons) if r2]
        items = [_reason_item(r, fx.freq, lang, kind) for r, kind in pairs] or [
            {"text": T.t(T.PHASE_TEXTS, "reasons_fallback", "en"), "detail": None, "kind": "support"}]
        name_long = N.PHASE_EN.get(label, label)
    else:
        items = [_reason_item(str(r), fx.freq, lang, _reason_kind(r, label)) for r in ph.reasons]
        name_long = PHASE_KO.get(label, label)
    # supporting reasons first, caveats after them (stable within each group)
    items = [x for x in items if x["kind"] == "support"] + [x for x in items if x["kind"] == "caveat"]
    conf = _finite(ph.confidence) or 0.0
    if fx.tentative:
        level, cert_key = "low", "tentative"
    else:
        level = "high" if conf >= 0.75 else ("medium" if conf >= 0.5 else "low")
        cert_key = level
    strength = _finite(fx.a.metrics.seasonal_strength) or 0.0
    return {
        "label": label,
        "name": fx.phase_name,
        "name_long": name_long,
        "icon": label,
        "seasonal_note": T.t(T.HERO_TEXTS, "seasonal_note", lang) if label == "stable" and strength >= 0.3 else None,
        "certainty": {"level": level, "label": T.t(T.CERTAINTY_LABELS, cert_key, lang), "value": _num(conf),
                      "pct_text": fmt.prob_exact(conf, lang)},
        "tentative": fx.tentative,
        "reasons": items,
        "rules_url": "/method#phase",
    }


# ---------------------------------------------------------------------------------------------------------------------
# Tiles (D.2.4)
# ---------------------------------------------------------------------------------------------------------------------

def _tiles(fx: _Facts, hero: dict, phase: dict) -> list[dict]:
    from .. import narrative as N
    from .. import report as R

    lang = fx.lang
    m = fx.a.metrics
    weak = fx.display != "normal"
    ref_suffix = T.t(T.TILE_TEXTS, "for_reference", lang) if weak else ""

    # 1 phase
    value = phase["name"] + (T.t(T.TILE_TEXTS, "tentative_suffix", lang) if fx.tentative else "")
    tiles = [_tile("phase", lang, value, sub=phase["certainty"]["label"], icon=phase["icon"])]

    # 2 recent change
    recent = _finite(m.recent_change_pct)
    why = fx.recent_undefined()
    sub = T.t(T.TILE_TEXTS, "recent_sub", lang, basis=fx.recent_basis())
    help_text = T.t(T.TILE_HELP, "recent_change", lang, span=fx.recent_span())
    if why is None and recent is not None:
        tone = "neutral" if fmt.pct_arrow(recent) == "0%" else ("up" if recent > 0 else "down")
        tiles.append(_tile("recent_change", lang, fmt.pct_arrow(recent), sub=sub, tone=tone, tooltip=help_text,
                           value_raw=recent))
    else:
        tiles.append(_tile("recent_change", lang, fmt.DASH, sub=sub, tooltip=help_text, undefined_reason=why))

    # 3 momentum
    mom_why = fx.momentum_undefined()
    if mom_why is None:
        tiles.append(_tile("momentum", lang, fx.momentum_word(), sub=fx.momentum_text(), value_raw=m.momentum))
    else:
        tiles.append(_tile("momentum", lang, fmt.DASH, sub="", undefined_reason=mom_why))

    # 4 expected change (the tooltip explains a seasonally adjusted basis, FU-3)
    basis = hero["basis"]
    help_text = T.t(T.TILE_HELP, "expected_change", lang)
    if fx.adjust_note:
        help_text += " " + fx.adjust_note
    if fx.change is not None:
        if fx.uncertain or fx.flat_word:
            value, tone = fmt.pct(fx.change), "neutral"
            sub = (T.t(T.TILE_TEXTS, "direction_uncertain", lang) if fx.uncertain else "") + basis
        else:
            value = fmt.pct_arrow(fx.change)
            tone = "neutral" if value == "0%" else ("up" if fx.change > 0 else "down")
            sub = basis
        tiles.append(_tile("expected_change", lang, value, sub=sub + ref_suffix, tone=tone, tooltip=help_text,
                           value_raw=fx.change))
    else:
        tiles.append(_tile("expected_change", lang, fmt.DASH, sub=basis + ref_suffix, tooltip=help_text,
                           undefined_reason=fx.change_reason()))

    # 5 chance of rise
    help_text = T.t(T.TILE_HELP, "prob_increase", lang)
    if fx.adjust_note:
        help_text += " " + T.t(T.TILE_TEXTS, "prob_adjusted", lang)
    if fx.prob is not None:
        tooltip = help_text + " " + T.t(T.TILE_TEXTS, "prob_exact_suffix", lang,
                                        prob_exact=fmt.prob_exact(fx.prob, lang))
        tiles.append(_tile("prob_increase", lang, fmt.prob_text(fx.prob, lang),
                           sub=T.t(T.TILE_TEXTS, "same_basis", lang, likelihood=fx.likelihood()) + ref_suffix,
                           tooltip=tooltip, value_raw=fx.prob))
    else:
        tiles.append(_tile("prob_increase", lang, fmt.DASH,
                           sub=T.t(T.TILE_TEXTS, "same_basis_only", lang) + ref_suffix, tooltip=help_text,
                           undefined_reason=fx.prob_reason))

    # 6 peak season or volatility
    pk = fx.peak
    if pk is not None:
        if pk["weekday"]:
            sub = T.t(T.PEAK_TEXTS, "sub_weekday", lang, trough=pk["trough"])
        elif pk["start"] == pk["peak"]:
            sub = T.t(T.PEAK_TEXTS, "sub_no_start", lang, peak=pk["peak"])
        else:
            sub = T.t(T.PEAK_TEXTS, "sub", lang, peak=pk["peak"], start=pk["start"])
        tiles.append(_tile("peak_season", lang, pk["peak"], sub=sub))
    else:
        cv = _finite(m.volatility)
        vol_why = R._rate_why(fx.report, N.volatility_undefined(fx.a), lang)
        if not vol_why and cv is not None:
            level = "low" if cv < 0.05 else ("medium" if cv < 0.15 else "high")
            sub = T.t(T.TILE_TEXTS, "volatility_sub", lang, cv=f"{fmt.round_half_away(cv * 100.0, 1):,.1f}")
            tiles.append(_tile("volatility", lang, T.t(T.VOLATILITY_WORDS, level, lang), sub=sub, value_raw=cv))
        else:
            tiles.append(_tile("volatility", lang, fmt.DASH, sub="",
                               undefined_reason=vol_why or T.t(T.TILE_TEXTS, "not_computed", lang)))
    return tiles


# ---------------------------------------------------------------------------------------------------------------------
# Chart (D.2.6)
# ---------------------------------------------------------------------------------------------------------------------

def _strength(magnitude: Any) -> str:
    v = _finite(magnitude) or 0.0
    for edge, key in _STRENGTH_EDGES:
        if v >= edge:
            return key
    return "small"


def _chart(fx: _Facts, hero: dict) -> dict:
    import numpy as np
    import pandas as pd

    from .. import narrative as N
    from ..analysis import changepoint_label

    lang, a, f, freq, kind = fx.lang, fx.a, fx.f, fx.freq, fx.value_kind
    series = a.series
    dates = _isos(series.index)
    actual = np.asarray(series, dtype=float)
    trend = a.decomposition.trend
    if isinstance(trend, pd.Series) and len(trend) != len(series):
        trend = trend.reindex(series.index)
    trend_vals = np.asarray(trend, dtype=float)
    frame = f.to_frame()
    band = {col: (_nums(frame[col]) if col in frame.columns else [None] * len(frame))
            for col in ("mean", "lo80", "hi80", "lo95", "hi95")}

    fin = np.isfinite(actual)
    last_pos = int(np.flatnonzero(fin)[-1]) if fin.any() else None
    last_actual = None
    if last_pos is not None:
        last_actual = {"date": dates[last_pos], "value": _num(actual[last_pos]),
                       "label": T.t(T.CHART_TEXTS, "last_actual", lang,
                                    date=fmt.date_label(dates[last_pos], freq, lang))}

    pool = [actual[fin]]
    for col in ("mean", "lo80", "hi80", "lo95", "hi95"):
        if col in frame.columns:
            arr = np.asarray(frame[col], dtype=float)
            pool.append(arr[np.isfinite(arr)])
    allv = np.concatenate(pool) if pool else np.zeros(0)
    act_max = float(np.max(actual[fin])) if fin.any() else None
    act_min = float(np.min(actual[fin])) if fin.any() else None
    if kind in _INDEX_KINDS:
        include_zero = True
    else:
        include_zero = bool(act_min is not None and act_min >= 0 and act_min <= 0.5 * (act_max or 0.0))
    y_hint = {"min": _num(np.min(allv)) if len(allv) else None, "max": _num(np.max(allv)) if len(allv) else None,
              "include_zero": include_zero}
    y_note = (T.t(T.CHART_TEXTS, "y_note", lang)
              if kind == "relative_index" and act_max is not None and act_max < 50 else None)

    cps = list(a.changepoints)
    order = sorted(range(len(cps)), key=lambda i: -(_finite(cps[i].magnitude) or 0.0))
    badged = set(order[:_CHART_CHANGEPOINTS])
    changepoints = []
    for i, cp in enumerate(cps):
        iso = _iso(cp.date)
        date_lab = fmt.date_short(iso, freq, lang)
        strength = _strength(cp.magnitude)
        label = changepoint_label(cp, lang)
        strength_label = T.t(T.STRENGTH_WORDS, strength, lang)
        changepoints.append({
            "n": i + 1, "date": iso, "date_label": date_lab, "label": label, "kind": str(cp.kind),
            "strength": strength, "strength_label": strength_label, "on_chart": i in badged,
            "text": T.t(T.CHART_TEXTS, "changepoint_text", lang, date=date_lab, label=label,
                        strength=strength_label)})

    intermittent = bool(N.is_intermittent(a))
    anomalies = []
    for an in a.anomalies:
        iso = _iso(an.date)
        z = _finite(an.zscore) or 0.0
        direction = "up" if z >= 0 else "down"
        v_text, e_text = fmt.value_text(an.value, kind, lang), fmt.value_text(an.expected, kind, lang)
        anomalies.append({
            "date": iso, "date_label": fmt.date_short(iso, freq, lang), "value": _num(an.value),
            "expected": _num(an.expected), "direction": direction,
            "text": T.t(T.CHART_TEXTS, "anomaly_text_small" if _small_anomaly(an, v_text, e_text) else
                        "anomaly_text", lang,
                        direction=T.t(T.CHART_TEXTS, f"anomaly_{direction}", lang), v=v_text,
                        expected_word=T.t(T.CHART_TEXTS, "usual_size" if intermittent else "expected", lang),
                        e=e_text)})

    start, end = (dates[0], dates[-1]) if dates else (fx.meta.start, fx.meta.end)
    end_shown = _iso(pd.Timestamp(end) + pd.Timedelta(days=6)) if freq == "W" else end
    range_freq = freq if freq in ("Q", "Y") else "M"
    x_label = T.t(T.CHART_TEXTS, "x_label", lang, freq=fmt.freq_label(freq, lang),
                  range=fmt.range_label(start, end_shown, range_freq, lang))
    return {
        "title": T.t(T.CHART_TEXTS, "title", lang, name=fx.report.name),
        "desc": _chart_desc(fx, start, end_shown),
        "freq": freq,
        "value_kind": kind,
        "x_label": x_label,
        "y_label": _value_kind_label(kind, fx.meta.unit_label, lang),
        "dates": dates,
        "actual": _nums(actual),
        "trend": _nums(trend_vals),
        "forecast": {"dates": list(fx.fc_dates), **band},
        "forecast_start": fx.fc_dates[0] if fx.fc_dates else None,
        "last_actual": last_actual,
        "levels": [int(lv) for lv in f.levels],
        "display": fx.display,
        "nonnegative": bool(f.nonnegative),
        "y_hint": y_hint,
        "y_note": y_note,
        "changepoints": changepoints,
        "anomalies": anomalies,
    }


def _small_anomaly(an: Any, v_text: str, e_text: str) -> bool:
    """An anomaly whose actual and expected values print alike or differ by under 2 % of the level: worded
    ``평소보다 약간 …`` instead of ``크게`` (a near noise-free series flags tiny deviations; UX-23)."""
    if v_text == e_text:
        return True
    v, e = _finite(an.value), _finite(an.expected)
    if v is None or e is None:
        return False
    level = max(abs(v), abs(e))
    return level > 0 and abs(v - e) < _ANOMALY_SMALL_REL * level


def _chart_desc(fx: _Facts, start: str, end: str) -> str:
    lang = fx.lang
    y0, y1 = start[:4].lstrip("0") or "0", end[:4].lstrip("0") or "0"
    years = y0 if y0 == y1 else f"{y0}–{y1}"
    parts = [T.t(T.CHART_TEXTS, "desc_head", lang, name=fx.report.name, freq_adj=fmt.freq_adjective(fx.freq, lang),
                 value_word=T.t(T.VALUE_WORDS, fx.value_kind if fx.value_kind in T.VALUE_WORDS else "raw", lang),
                 years=years)]
    if fx.peak is not None:
        parts.append(_peak_phrase(fx.peak, lang, "desc"))
    if fx.change is not None:
        parts.append(T.t(T.CHART_TEXTS, "desc_change", lang, fut=_cap(fx.fut_b, lang), pct=fmt.pct(fx.change)))
    else:
        parts.append(T.t(T.CHART_TEXTS, "desc_word", lang, word=fx.word))
    return " ".join(parts)


def _cap(text: str, lang: str) -> str:
    """Capitalise the first letter of an English phrase that starts a sentence."""
    return text[:1].upper() + text[1:] if lang == "en" and text else text


def _topic_particle(word: str) -> str:
    """Korean topic particle after ``word``: ``은`` after a final consonant (``6개월은``, ``5년은``), else ``는``
    (``26주는``, ``8분기는``). Non-Hangul endings (digits) take ``는``."""
    last = word[-1:] if word else ""
    if "가" <= last <= "힣":
        return "은" if (ord(last) - 0xAC00) % 28 else "는"
    return "는"


def _haeyo(sentence: str) -> str:
    """The engine's 합니다체 sentence endings rewritten to 해요체 (``…계산하지 않았습니다.`` -> ``…않았어요.``)."""
    out = sentence
    for old, new in _HAEYO:
        out = out.replace(old, new)
    return out


def _peak_phrase(peak: Mapping[str, Any], lang: str, where: str) -> str:
    """``보통 7월 정점`` in the chart desc / summary line (English: ``in Jul`` or ``on Saturdays``)."""
    prep = "on" if peak["weekday"] else "in"
    if where == "desc":
        if lang == "en":
            return f"Usually peaks {prep} {peak['peak']}."
        return T.t(T.CHART_TEXTS, "desc_peak", lang, peak=peak["peak"])
    if lang == "en":
        return f" · usually peaks {prep} {peak['peak']}"
    return T.t(T.SUMMARY_TEXTS, "peak", lang, peak=peak["peak"])


# ---------------------------------------------------------------------------------------------------------------------
# Past (D.2.7) and future (D.2.8)
# ---------------------------------------------------------------------------------------------------------------------

def _past(fx: _Facts) -> dict:
    lang = fx.lang
    n_cp, n_an = len(fx.a.changepoints), len(fx.a.anomalies)
    if lang == "en":
        summary = T.t(T.PAST_TEXTS, "summary", lang, n=n_cp, m=n_an,
                      cp_word=T.t(T.PAST_TEXTS, "changepoint_one" if n_cp == 1 else "changepoint_many", lang),
                      an_word=T.t(T.PAST_TEXTS, "anomaly_one" if n_an == 1 else "anomaly_many", lang))
    else:
        summary = T.t(T.PAST_TEXTS, "summary", lang, n=n_cp, m=n_an)
    s = _finite(fx.a.metrics.seasonal_strength) or 0.0
    s_text = str(fmt.round_half_away(s, 2))
    if s >= 0.6:
        text = T.t(T.PAST_TEXTS, "seasonality_strong", lang, s=s_text)
    elif s >= 0.3:
        text = T.t(T.PAST_TEXTS, "seasonality_clear", lang, s=s_text)
    else:
        text = T.t(T.PAST_TEXTS, "seasonality_none", lang)
    if fx.peak is not None:
        key = "peak_trough_weekday" if fx.peak["weekday"] else "peak_trough"
        text += T.t(T.PAST_TEXTS, key, lang, peak=fx.peak["peak"], trough=fx.peak["trough"])
    return {"summary": summary, "seasonality": {"strength": _num(s), "text": text}}


def _scenarios(fx: _Facts) -> Optional[dict]:
    from .. import narrative as N

    lang, kind, freq = fx.lang, fx.value_kind, fx.freq
    if not fx.fc_dates:
        return None
    end_iso = fx.fc_dates[-1]
    date_lab = fmt.date_label(end_iso, freq, lang)
    ref = N.scenario_reference(fx.a, fx.f)
    reference = None
    ref_value = None
    if ref is not None:
        ref_iso = _iso(ref.date)
        if ref.kind == "season":
            label = T.t(T.SCENARIO_TEXTS, "ref_season", lang, date_label=fmt.date_label(ref_iso, freq, lang))
        elif ref.kind == "demand":
            label = T.t(T.SCENARIO_TEXTS, "ref_demand", lang, span=fmt.span_label(ref.span, freq, lang))
        else:
            label = T.t(T.SCENARIO_TEXTS, "ref_latest", lang, date_label=fmt.date_label(ref_iso, freq, lang))
        vtext = fmt.value_text(ref.value, kind, lang, compact=True)
        reference = {"label": label, "value_text": vtext,
                     "text": T.t(T.SCENARIO_TEXTS, "reference_text", lang, label=label, value_text=vtext)}
        ref_value = ref.value
    if fx.change is None:          # every surface drops the scenario percentages then
        ref_value = None
    floor = bool(N.scenario_floor(fx.a, fx.f))
    floor_text = T.t(T.SCENARIO_TEXTS, "floor", lang)
    rows = []
    for key in ("pessimistic", "base", "optimistic"):
        v = (fx.o.scenarios or {}).get(key)
        text, pct_raw = N.scenario_value(v, ref_value, nonnegative=floor, lang=lang)
        vtext = text if text == floor_text else fmt.value_text(v, kind, lang, compact=True)
        pct_text = fmt.pct(pct_raw) if _finite(pct_raw) is not None else None
        label = T.t(T.SCENARIO_TEXTS, key, lang)
        row_text = (T.t(T.SCENARIO_TEXTS, "row_with_pct", lang, label=label, value_text=vtext, pct_text=pct_text)
                    if pct_text is not None else
                    T.t(T.SCENARIO_TEXTS, "row_without_pct", lang, label=label, value_text=vtext))
        rows.append({"key": key, "label": label, "value_text": vtext, "pct_text": pct_text, "text": row_text})
    level = _interval_level(fx.f)
    note = T.t(T.SCENARIO_TEXTS, "note", lang, level=level)
    if fx.display == "bands_only":          # the base case is the centre line the chart withholds (CON-4)
        note += T.t(T.SCENARIO_TEXTS, "note_bands_only", lang)
    return {
        "date_label": date_lab,
        "title": T.t(T.SCENARIO_TEXTS, "title", lang, date_label=date_lab),
        "reference": reference,
        "rows": rows,
        "note": note,
    }


def _interval_level(forecast: Any) -> int:
    """The band behind ``Outlook.scenarios`` (``forecast.interval_level``: 80 by default)."""
    from ..forecast import interval_level

    lv = interval_level(forecast)
    return 80 if lv is None else int(lv)


def _bound_text(v: Any, kind: str, lang: str, nonnegative: bool, compact: bool = False) -> str:
    fv = _finite(v)
    if fv is None:
        return fmt.DASH
    if nonnegative and fv <= 0.0:
        return "0"
    return fmt.value_text(fv, kind, lang, compact=compact)


def _future(fx: _Facts) -> dict:
    import pandas as pd

    from .. import narrative as N

    import numpy as np

    lang, f, freq, kind = fx.lang, fx.f, fx.freq, fx.value_kind
    frame = f.to_frame()
    nonneg = bool(f.nonnegative)
    sep = T.t(T.SCENARIO_TEXTS, "range_sep", lang)
    # large raw values (won, page views): compact like the scenario boxes, never 4.22×10¹² next to 4.3조 (CON-13)
    cols = [c for c in ("mean", "lo80", "hi80", "lo95", "hi95") if c in frame.columns]
    vals = np.asarray(frame[cols], dtype=float).ravel() if cols else np.zeros(0)
    vals = vals[np.isfinite(vals)]
    compact = kind not in _INDEX_KINDS and bool(len(vals)) and float(np.max(np.abs(vals))) >= _TABLE_COMPACT_FROM
    hide_mean = fx.display == "bands_only"  # the chart and its table view withhold the centre line (CON-4)
    rows = []
    def band(rec: Any, level: int) -> str:
        lo, hi = rec.get(f"lo{level}"), rec.get(f"hi{level}")
        if lo is None or hi is None:
            return fmt.DASH
        return (f"{_bound_text(lo, kind, lang, nonneg, compact)}{sep}"
                f"{_bound_text(hi, kind, lang, nonneg, compact)}")

    for i, iso in enumerate(fx.fc_dates):
        rec = frame.iloc[i]
        rows.append({"date": iso, "date_label": fmt.date_label(iso, freq, lang),
                     "mean": fmt.DASH if hide_mean else _bound_text(rec.get("mean"), kind, lang, nonneg, compact),
                     "range80": band(rec, 80), "range95": band(rec, 95)})

    if fx.mase is not None:
        m_text = str(fmt.round_half_away(fx.mase, 2))
        details = T.t(T.ACCURACY_TEXTS, "details_high" if fx.mase > 1 else "details_low", lang, m=m_text)
    else:
        details = None

    members = [(s.name, _finite(s.weight) or 0.0) for s in f.model_scores if (_finite(s.weight) or 0.0) > 0.0]
    members.sort(key=lambda t: -t[1])
    total = sum(w for _, w in members) or 1.0
    models = [{"id": str(nm), "name": N.model_label(nm, lang), "weight": _num(w),
               "weight_text": f"{fmt.round_half_away(w / total * 100.0, 0)}%"} for nm, w in members]

    o = fx.o
    fut = str(o.future_phase)
    fut_name = T.t(T.PHASE_SHORT, fut, lang) if fut in T.PHASE_SHORT else fut
    status = N.future_phase_status(fx.a, o)
    if status == "undecided":
        hedge, text = (T.t(T.FUTURE_TEXTS, "hedge_undecided", lang),
                       T.t(T.FUTURE_TEXTS, "undecided", lang, h=fx.h_label))
    elif status == "tentative":
        hedge, text = T.t(T.FUTURE_TEXTS, "hedge_tentative", lang), T.t(T.FUTURE_TEXTS, "tentative", lang)
    elif status == "low_confidence":
        hedge, text = (T.t(T.FUTURE_TEXTS, "hedge_low_confidence", lang),
                       T.t(T.FUTURE_TEXTS, "low_confidence", lang))
    else:
        hedge = None
        key = "stay" if fut == str(fx.a.phase.label) else "become"
        text = T.t(T.FUTURE_TEXTS, key, lang, h=fx.h_label, name=fut_name)
    peak_date = o.forecast_peak_date
    if peak_date is not None and not pd.isna(peak_date):
        p_iso = _iso(peak_date)
        peak = {"date": p_iso, "text": T.t(T.FUTURE_TEXTS, "peak", lang,
                                           date_label=fmt.date_label(p_iso, freq, lang))}
    else:
        peak = {"date": None, "text": T.t(T.FUTURE_TEXTS, "no_peak", lang)}
    return {
        "horizon": fx.horizon,
        "horizon_label": fx.h_label,
        "end_label": fmt.date_label(fx.fc_dates[-1], freq, lang) if fx.fc_dates else None,
        "scenarios": _scenarios(fx),
        "table": {"rows": rows},
        "accuracy": {"line": _accuracy_line(fx.skill, fx.mase, lang), "skill": _num(fx.skill), "details": details},
        "method": N.describe_method(f, lang),
        "models": models,
        "phase": {"label": fut, "name": fut_name, "hedge": hedge, "text": text},
        "peak": peak,
    }


# ---------------------------------------------------------------------------------------------------------------------
# Narrative helpers (D.2.9) -- also used by narrative_ai.py for AI texts
# ---------------------------------------------------------------------------------------------------------------------

def split_caveats(markdown: str, lang: str) -> tuple[str, str]:
    """Split a narrative at the line ``## {narrative.SECTION_HEADINGS[lang][3]}``: returns ``(body, caveats)`` where
    ``body`` is everything before that heading line (trailing blank lines stripped) and ``caveats`` the lines after
    it (``""`` when the heading is absent). Applies to rule-based and AI narratives alike."""
    from ..narrative import SECTION_HEADINGS

    heading = "## " + SECTION_HEADINGS[_lang(lang)][3]
    lines = str(markdown or "").replace("\r\n", "\n").split("\n")
    for i, line in enumerate(lines):
        if line.strip() == heading:
            body = "\n".join(lines[:i]).rstrip()
            caveats = "\n".join(lines[i + 1:]).strip("\n").rstrip()
            return body, caveats
    return str(markdown or "").rstrip(), ""


def localize_dates(text: str, freq: str, lang: str, chart_dates: Sequence[str]) -> tuple[str, dict[str, str]]:
    """Rewrite the engine's narrative dates to display dates and record the links (D.2.9).

    Every match of ``\\b\\d{4}-\\d{2}-\\d{2}\\b``, ``\\b\\d{4}-\\d{2}\\b`` (only when ``freq == "M"``) and
    ``\\b\\d{4}-Q[1-4]\\b`` (only when ``freq == "Q"``) becomes ``fmt.date_short(…, freq, lang)``.

    Args:
        text: Markdown-lite text (narrative body, caveats or an AI text).
        freq: the series frequency.
        lang: ``"ko"`` | ``"en"``.
        chart_dates: ``chart.dates + chart.forecast.dates`` (ISO, ascending).

    Returns ``(rewritten text, date_links)`` with ``date_links[display text] = chart date``: D/W exact match of a
    chart date; M the chart date starting with that ``YYYY-MM``; Q the quarter's start date; Y never linked. Dates
    without a matching chart date are rewritten but not linked.
    """
    lang = _lang(lang)
    exact = set(chart_dates)
    by_month: dict[str, str] = {}
    by_quarter: dict[str, str] = {}
    for iso in chart_dates:
        by_month.setdefault(iso[:7], iso)
        try:
            q = (int(iso[5:7]) - 1) // 3 + 1
        except ValueError:
            continue
        by_quarter.setdefault(f"{iso[:4]}-Q{q}", iso)
    links: dict[str, str] = {}

    def repl(m: re.Match) -> str:
        s = m.group(0)
        try:
            if "-Q" in s:
                if freq != "Q":
                    return s
                year, q = int(s[:4]), int(s[-1])
                iso = f"{year:04d}-{(q - 1) * 3 + 1:02d}-01"
                shown = fmt.date_short(iso, "Q", lang)
                target = by_quarter.get(s)
            elif len(s) == 7:
                if freq != "M":
                    return s
                iso = f"{s}-01"
                fmt.date_short(iso, "M", lang)        # validates the month
                shown = fmt.date_short(iso, "M", lang)
                target = by_month.get(s)
            else:
                shown = fmt.date_short(s, freq, lang)
                target = s if s in exact and freq in ("D", "W") else None
        except ValueError:                             # not a real date (2024-13-45): leave it alone
            return s
        if target is not None and freq != "Y":
            links.setdefault(shown, target)
        return shown

    return _DATE_RX.sub(repl, str(text or "")), links


def _narrative_parts(text: str, freq: str, lang: str, chart_dates: Sequence[str]) -> tuple[str, str, dict]:
    body, caveats = split_caveats(text, lang)
    body, links = localize_dates(body, freq, lang, chart_dates)
    caveats, links2 = localize_dates(caveats, freq, lang, chart_dates)
    for k, v in links2.items():
        links.setdefault(k, v)
    return body, caveats, links


def build_ai_record(markdown: str, vm: Mapping[str, Any], *, model: str, created_at: str) -> dict:
    """The stored AI narrative for one series and language (``results/<id>/ai/<k>-<lang>.json``, D.2.9).

    Args:
        markdown: the LLM's Markdown answer (``fetchers.llm(payload, lang)``).
        vm: the stored series VM in the AI text's language (gives ``lang``, ``chart.freq`` and the chart dates).
        model: model id reported by the fetcher (e.g. ``"claude-opus-5-5"``).
        created_at: UTC ISO ``Z`` timestamp.

    Returns ``{"markdown", "caveats_markdown", "badge", "model", "note", "created_at", "date_links"}`` -- the text
    split with :func:`split_caveats` and dated with :func:`localize_dates`, badge ``AI 해석 · Claude`` /
    ``AI · Claude`` and the fixed note (``texts``). ``date_links`` is storage-only: :func:`attach_ai` merges it into
    ``narrative.date_links`` and leaves it out of ``narrative.ai`` (whose shape is exactly D.2.9's).
    """
    lang = _lang(str(vm.get("lang", "ko")))
    chart = vm.get("chart") or {}
    dates = list(chart.get("dates") or []) + list((chart.get("forecast") or {}).get("dates") or [])
    body, caveats, links = _narrative_parts(markdown, str(chart.get("freq") or "M"), lang, dates)
    return {"markdown": body, "caveats_markdown": caveats, "badge": T.t(T.NARRATIVE_TEXTS, "badge_ai", lang),
            "model": str(model), "note": T.t(T.NARRATIVE_TEXTS, "ai_note", lang), "created_at": str(created_at),
            "date_links": links}


def attach_ai(vm: Mapping[str, Any], ai: Optional[Mapping[str, Any]]) -> dict:
    """Return a copy of a stored series VM with ``narrative.ai`` set from a stored AI record (:func:`build_ai_record`)
    or ``None``: ``ai`` -> ``{markdown, caveats_markdown, badge, model, note, created_at}`` and its ``date_links``
    merged into ``narrative.date_links``. Used by ``api.py`` for ``GET /api/results`` and shares (C.7, C.9)."""
    out = copy.deepcopy(dict(vm))
    nar = out.setdefault("narrative", {})
    if ai is None:
        nar["ai"] = None
        return out
    nar["ai"] = {k: ai.get(k) for k in ("markdown", "caveats_markdown", "badge", "model", "note", "created_at")}
    links = dict(nar.get("date_links") or {})
    for k, v in dict(ai.get("date_links") or {}).items():
        links.setdefault(k, v)
    nar["date_links"] = links
    return out


def ai_job_narrative(ai: Mapping[str, Any]) -> dict:
    """The ``narrative`` object of a finished narrative job snapshot (C.6): ``{"markdown", "caveats_markdown",
    "source": "llm", "badge", "model", "note"}`` from a stored AI record."""
    return {"markdown": ai.get("markdown"), "caveats_markdown": ai.get("caveats_markdown"), "source": "llm",
            "badge": ai.get("badge"), "model": ai.get("model"), "note": ai.get("note")}


def _certainty_word(fx: _Facts) -> str:
    """``판단 확신도 높음`` / ``phase certainty: high`` -- the phase tile's wording (D.2.5), for the narrative."""
    conf = _finite(fx.a.phase.confidence) or 0.0
    level = "high" if conf >= 0.75 else ("medium" if conf >= 0.5 else "low")
    label = T.t(T.CERTAINTY_LABELS, level, fx.lang)
    return label[:1].lower() + label[1:] if fx.lang == "en" else label


def _tidy_rule_narrative(text: str, fx: _Facts) -> str:
    """The web's post-processing of the frozen rule-based narrative (UX-13, CON-9, CON-7; spec D.2.9 amendment).

    * Phase certainty as the tile's word, never a percentage beside ``상승 확률``/``예측 신뢰도`` (brief §4.6, R4):
      ``(신뢰도 57%)`` / ``(판단 신뢰도 95%)`` -> ``(판단 확신도 보통)``; tentative calls drop the number;
      English ``(confidence 57%)`` -> ``(phase certainty: medium)``, ``Probability of increase`` -> ``Chance of
      rise``, ``forecast confidence`` -> ``forecast reliability`` (the cards' terms).
    * The MASE note, the model ids and test statistics (``τ = 0.95, p < 0.001``) are dropped from the body: they
      live in the 자세히 blocks (``future.accuracy.details``, ``future.models``, ``phase.reasons[].detail``).
    * Weekly data: the analysed span ``X부터 Y까지`` / ``from X to Y`` ends on the last week's Sunday, like the
      source line (D.1.6), instead of its Monday.
    """
    lang, out = fx.lang, str(text or "")
    if fx.freq == "W" and len(fx.a.series):
        import pandas as pd

        last = _iso(fx.a.series.index[-1])
        sunday = _iso(pd.Timestamp(last) + pd.Timedelta(days=6))
        rx = (r"(from \d{4}-\d{2}-\d{2} to )" if lang == "en" else r"(\d{4}-\d{2}-\d{2}부터 )") + re.escape(last) + \
            (r"\b" if lang == "en" else r"(?=까지)")
        out = re.sub(rx, lambda m: m.group(1) + sunday, out, count=1)
    word = _certainty_word(fx)
    if lang == "en":
        out = _NARR_TENTATIVE_EN.sub("tentative call", out)
        out = _NARR_TENTATIVE_EN2.sub(")", out)
        out = _NARR_CONF_EN.sub(f"({word})", out)
    else:
        out = _NARR_TENTATIVE_KO.sub("잠정 판단입니다", out)
        out = _NARR_TENTATIVE_KO2.sub(")", out)
        out = _NARR_CONF_KO.sub(f"({word})", out)
    for rx_drop in _NARR_DROP:
        out = rx_drop.sub("", out)
    for old, new in _NARR_WORDS:
        out = out.replace(old, new)
    return out


def _narrative(fx: _Facts, inp: SeriesInput, chart: dict) -> tuple[dict, dict]:
    lang = fx.lang
    dates = list(chart["dates"]) + list(chart["forecast"]["dates"])
    text = _tidy_rule_narrative(str(inp.narratives.get(lang) or ""), fx)
    body, caveats, links = _narrative_parts(text, fx.freq, lang, dates)
    if fx.kind == "naver":
        reason, block = "naver_terms", _render("ai_naver_blocked", {}, lang)
    elif not inp.ai_enabled:
        reason, block = "ai_unavailable", _render("ai_unavailable", {}, lang)
    else:
        reason, block = None, None
    narrative = {
        "markdown": body,
        "date_links": links,
        "source": "rule",
        "badge": T.t(T.NARRATIVE_TEXTS, "badge_rule", lang),
        "ai": None,
        "ai_allowed": reason is None,
        "ai_block_reason": reason,
        "ai_block_text": block,
        "ai_privacy_note": T.t(T.NARRATIVE_TEXTS, "ai_privacy_note", lang),
    }
    caveats_obj = {"markdown": caveats, "disclaimer": T.t(T.NARRATIVE_TEXTS, "disclaimer", lang)}
    return narrative, caveats_obj


# ---------------------------------------------------------------------------------------------------------------------
# Summary line (D.2.10)
# ---------------------------------------------------------------------------------------------------------------------

def _summary_line(fx: _Facts, source: Mapping[str, Any]) -> str:
    lang = fx.lang
    line = T.t(T.SUMMARY_TEXTS, "head", lang, name=fx.report.name, short=source["short"],
               freq_adj=fmt.freq_adjective(fx.freq, lang), phase=fx.phase_name)
    if fx.change is not None:
        line += T.t(T.SUMMARY_TEXTS, "change", lang, fut=fx.fut_b, pct=fmt.pct(fx.change))
    else:
        line += T.t(T.SUMMARY_TEXTS, "word", lang, word=fx.word)
    if fx.prob is not None:
        line += T.t(T.SUMMARY_TEXTS, "prob", lang, prob=fmt.prob_text(fx.prob, lang))
    if fx.peak is not None:
        line += _peak_phrase(fx.peak, lang, "summary")
    line += fx.season_short()
    if fx.display in ("bands_only", "weak"):
        line += T.t(T.SUMMARY_TEXTS, fx.display, lang)
    # D.2.2/D.4.1 wording of as_of: fetch time for Naver/Wikipedia, analysis time for uploads, none for samples
    if fx.kind in ("naver", "wiki"):
        line += T.t(T.SUMMARY_TEXTS, "as_of", lang, as_of=source["as_of_label"])
    elif fx.kind in _UPLOAD_KINDS:
        line += T.t(T.SUMMARY_TEXTS, "as_of_analysed", lang, as_of=source["as_of_label"])
    return line


# ---------------------------------------------------------------------------------------------------------------------
# Series VM (D.2)
# ---------------------------------------------------------------------------------------------------------------------

def build_series_vm(inp: SeriesInput, lang: str) -> dict:
    """The complete D.2 series VM in ``lang``.

    Top-level keys, in this order: ``schema`` (:data:`WEB_VM_SCHEMA`), ``lang``, ``index``, ``name``, ``terms``,
    ``slot``, ``icon``, ``flags``, ``source``, ``hero``, ``phase``, ``tiles`` (exactly 6), ``chart``, ``past``,
    ``future``, ``narrative``, ``caveats``, ``summary_line``, ``exports``, ``notices``.

    * ``slot``/``icon`` from ``inp.slot`` (``SLOT_ICONS``; both ``None`` when the slot is ``None``).
    * ``narrative.markdown``/``caveats.markdown`` from ``inp.narratives[lang]`` via :func:`split_caveats` and
      :func:`localize_dates`; ``narrative.ai`` is ``None`` (AI texts are attached per response, :func:`attach_ai`).
    * ``narrative.ai_allowed`` = not Naver and ``inp.ai_enabled`` (shares are applied later, :func:`make_shared_vm`).
    * ``exports`` = :func:`export_block` of ``export_availability(inp.source.kind, xlsx=inp.xlsx_available, ...)``
      at series level: every link ``None`` plus ``unavailable`` reasons. No URL, no id.
    * ``notices`` = ``list(inp.notices[lang])``.

    Raises nothing for odd data: undefined values become ``"—"`` with an ``undefined_reason`` (D.2.4); any exception
    that does escape is a bug and is turned into a ``series_failed_web`` entry by the caller.
    """
    import numpy as np

    from .. import narrative as N

    lang = _lang(lang)
    fx = _Facts(inp.report, inp.source, lang)
    a = fx.a
    source = build_source_block(inp.source, lang)
    hero = _hero(fx)
    phase = _phase(fx, hero)
    tiles = _tiles(fx, hero, phase)
    chart = _chart(fx, hero)
    narrative, caveats = _narrative(fx, inp, chart)
    actual = np.asarray(a.series, dtype=float)
    actual_max = float(np.max(actual[np.isfinite(actual)])) if np.isfinite(actual).any() else None
    slot = inp.slot if inp.slot in SLOT_ICONS else None
    av = export_availability(inp.source.kind, xlsx=inp.xlsx_available,
                             view="compare" if inp.request_size > 1 else "detail", succeeded=1)
    return {
        "schema": WEB_VM_SCHEMA,
        "lang": lang,
        "index": int(inp.index),
        "name": inp.name,
        "terms": [str(t) for t in inp.source.terms],
        "slot": slot,
        "icon": SLOT_ICONS.get(slot) if slot is not None else None,
        "flags": {
            "synthetic": bool(N.is_synthetic(a)),
            "tentative": fx.tentative,
            "intermittent": bool(N.is_intermittent(a)),
            "all_zero": bool(N.is_all_zero(a)),
            "low_resolution": is_low_resolution(inp.source.kind, actual_max, inp.request_size),
            "reference_only": fx.display != "normal",
            "seasonal": fx.peak is not None,
        },
        "source": source,
        "hero": hero,
        "phase": phase,
        "tiles": tiles,
        "chart": chart,
        "past": _past(fx),
        "future": _future(fx),
        "narrative": narrative,
        "caveats": caveats,
        "summary_line": _summary_line(fx, source),
        "exports": export_block(av, lang, level="series"),
        "notices": [copy.deepcopy(dict(n)) for n in inp.notices.get(lang, ())],
    }


def build_failed_entry(index: int, name: str, error: Mapping[str, Any]) -> dict:
    """A failed series entry (C.6, C.7, D.4): ``{"index": index, "name": name, "error": dict(error)}`` where ``error``
    is the C.12.1 error object (the body *inside* ``{"error": …}``) built by ``errors.py``. Language independent:
    the error object carries both ``message_ko`` and ``message_en``."""
    return {"index": int(index), "name": str(name), "error": copy.deepcopy(dict(error))}


def is_low_resolution(source_kind: str, actual_max: Optional[float], request_size: int) -> bool:
    """The one rule behind ``flags.low_resolution`` (D.2.1) and the ``naver_low_resolution`` notice (E.6 step 2):
    ``source_kind == "naver"`` and ``request_size >= 2`` and ``actual_max < 5`` (``None``/NaN -> ``False``)."""
    v = _finite(actual_max)
    return bool(source_kind == "naver" and int(request_size) >= 2 and v is not None and v < 5)


# ---------------------------------------------------------------------------------------------------------------------
# Slots (D.4.2)
# ---------------------------------------------------------------------------------------------------------------------

def provisional_slot(index: int, request_size: int) -> Optional[int]:
    """Slot known before ranking: ``index + 1`` when ``request_size <= MAX_SLOTS``, else ``None`` (D.4.2)."""
    return int(index) + 1 if int(request_size) <= MAX_SLOTS else None


def assign_slots(ranked_indexes: Sequence[int], request_size: int) -> dict[int, Optional[int]]:
    """Final slots for the successful series.

    Args:
        ranked_indexes: series indexes best first, i.e. the ``rank_trends`` row order mapped through the
            ``{display name: index}`` dict (D.4.1). Ties already follow ``rank_trends``' stable order.
        request_size: ``N`` of the request (failed series included).

    Returns ``{index: slot}`` for every index in ``ranked_indexes``: ``N <= 5`` -> ``index + 1``; ``N > 5`` -> the
    first five of ``ranked_indexes`` get slots 1..5 **in request (index) order**, all others ``None``.
    """
    ranked = [int(i) for i in ranked_indexes]
    if int(request_size) <= MAX_SLOTS:
        return {i: i + 1 for i in ranked}
    top = sorted(ranked[:MAX_SLOTS])
    out: dict[int, Optional[int]] = {i: None for i in ranked}
    for slot, i in enumerate(top, start=1):
        out[i] = slot
    return out


def set_slot(vm: dict, slot: Optional[int]) -> dict:
    """Write the final ``slot`` and matching ``icon`` into a series VM **in place** and return it (stage ``finish``,
    once per language)."""
    slot = slot if slot in SLOT_ICONS else None
    vm["slot"] = slot
    vm["icon"] = SLOT_ICONS.get(slot) if slot is not None else None
    return vm


# ---------------------------------------------------------------------------------------------------------------------
# Compare VM (D.4)
# ---------------------------------------------------------------------------------------------------------------------

def _scale(members: Sequence[SeriesInput], lang: str) -> tuple[dict, list[dict]]:
    """The D.4.3 ``scale`` object and the scale notices (``log_hint``, ``index_unavailable``)."""
    import numpy as np

    kinds = {m.source.kind for m in members}
    kind = members[0].source.kind if len(kinds) == 1 else "upload"
    freq = members[0].source.freq
    notices: list[dict] = []

    series = [m.report.analysis.series for m in members]
    positive = True
    for m in members:
        y = np.asarray(m.report.analysis.series, dtype=float)
        lo95 = m.report.forecast.lower.get(95) if m.report.forecast.lower else None
        lo = np.asarray(lo95, dtype=float) if lo95 is not None else np.zeros(0)
        vals = np.concatenate([y[np.isfinite(y)], lo[np.isfinite(lo)]])
        if not len(vals) or float(np.min(vals)) <= 0.0:
            positive = False
    log_reason = None if positive else T.t(T.COMPARE_TEXTS, "log_disabled", lang)

    index_base = None
    index_ok = False
    if kind not in _SAME_SCALE_KINDS:
        index_base, index_ok = _index_base(series, freq, lang)

    if kind in _SAME_SCALE_KINDS:
        index_reason: Optional[str] = T.t(T.COMPARE_TEXTS, "index_disabled_same_request", lang)
    elif not index_ok:
        index_reason = _render("index_unavailable", {}, lang)
    else:
        index_reason = None

    default = "raw"
    if kind == "upload":
        if index_ok:
            default = "index"
        else:
            notices.append(make_notice("index_unavailable", lang))

    log_hint = False
    if kind == "wiki" and positive:
        maxima = [float(np.nanmax(np.asarray(s, dtype=float))) for s in series if len(s)]
        maxima = [v for v in maxima if math.isfinite(v)]
        if len(maxima) >= 2 and min(maxima) > 0 and max(maxima) / min(maxima) > _LOG_HINT_RATIO:
            log_hint = True
            notices.append(make_notice("log_hint", lang, params={"ratio": _LOG_HINT_RATIO}))

    scale = {
        "default": default,
        "allowed": {"raw": True, "index": bool(index_ok), "log": positive},
        "disabled_reason": {"log": log_reason, "index": index_reason},
        "index_base": index_base,
        "log_hint": log_hint,
    }
    return scale, notices


def _index_base(series: Sequence[pd.Series], freq: str, lang: str) -> tuple[Optional[dict], bool]:
    """``index_base`` (D.4.3): window = the first ``season_length(freq)`` periods from the latest series start
    (1 for Y); base = mean of the actual values in the window per series."""
    import numpy as np
    import pandas as pd

    from ..data import season_length

    if not series or any(len(s) == 0 for s in series):
        return None, False
    try:
        width = 1 if freq == "Y" else max(1, int(season_length(freq)))
    except Exception:  # noqa: BLE001 - unknown freq: one season of 12
        width = 12
    starts = [pd.Timestamp(s.index[0]) for s in series]
    latest = max(range(len(series)), key=lambda i: starts[i])
    ref = series[latest]
    window = ref.index[:width]
    w_start, w_end = pd.Timestamp(window[0]), pd.Timestamp(window[-1])
    values: dict[str, Optional[float]] = {}
    ok = True
    for i, s in enumerate(series):
        y = np.asarray(s, dtype=float)
        idx = pd.DatetimeIndex(s.index)
        sel = y[(idx >= w_start) & (idx <= w_end)]
        sel = sel[np.isfinite(sel)]
        base = float(np.mean(sel)) if len(sel) else float("nan")
        fin = y[np.isfinite(y)]
        scale = float(np.max(np.abs(fin))) if len(fin) else 0.0
        if not (math.isfinite(base) and base > 0 and base > 1e-9 * scale):
            ok = False
        values[str(i)] = _num(base)
    start_iso, end_iso = _iso(w_start), _iso(w_end)
    if len(window) == 1:
        label = T.t(T.COMPARE_TEXTS, "index_base_label_one", lang, date=fmt.date_short(start_iso, freq, lang))
    else:
        label = T.t(T.COMPARE_TEXTS, "index_base_label", lang, range=fmt.range_label(start_iso, end_iso, freq, lang))
    return {"start": start_iso, "end": end_iso, "label": label, "values": values}, ok


def _ranking_row(rank: int, row: Mapping[str, Any], inp: SeriesInput, vm: Mapping[str, Any],
                 slot: Optional[int], lang: str) -> dict:
    import numpy as np

    fx = _Facts(inp.report, inp.source, lang)
    a = fx.a
    hero = vm["hero"]
    outlook = hero["outlook"]
    recent = _finite(a.metrics.recent_change_pct)
    recent_why = fx.recent_undefined()
    mom_why = fx.momentum_undefined()
    score = _finite(row.get("score"))
    season = max(1, int(a.season_length or 1))
    hist = np.asarray(a.series, dtype=float)
    k = min(len(hist), 2 * season, 104)
    display = vm["chart"]["display"]
    fc_mean = np.asarray(inp.report.forecast.mean, dtype=float)[:52]
    return {
        "rank": int(rank),
        "index": int(inp.index),
        "name": inp.name,
        "slot": slot,
        "icon": SLOT_ICONS.get(slot) if slot is not None else None,
        "phase": {"label": vm["phase"]["label"], "name": vm["phase"]["name"], "icon": vm["phase"]["icon"],
                  "tentative": vm["phase"]["tentative"]},
        "outlook": {"arrow": outlook["arrow"], "tone": outlook["tone"], "word": outlook["word"],
                    "likelihood": outlook["likelihood"], "prob_text": outlook["prob_text"], "line": outlook["line"]},
        "expected_change": {"text": fmt.pct(fx.change) if fx.change is not None else fmt.DASH,
                            "basis": hero["basis"] + fx.season_short(), "undefined_reason": fx.change_reason()},
        "recent_change": {"text": fmt.pct(recent) if recent_why is None and recent is not None else fmt.DASH,
                          "basis": fx.recent_basis(), "undefined_reason": recent_why},
        "momentum": {"text": fx.momentum_text(), "word": fx.momentum_word() if mom_why is None else fmt.DASH,
                     "undefined_reason": mom_why},
        "score": {"value": _num(score), "text": str(fmt.round_half_away(score, 1)) if score is not None else fmt.DASH},
        "confidence": {"level": hero["reliability"]["level"], "label": hero["reliability"]["label"]},
        "sort": {"score": _num(score), "prob": outlook["prob"], "expected": _num(fx.change),
                 "recent": _num(recent) if recent_why is None else None, "level": _num(a.metrics.last_value)},
        "spark": {"history": _nums(hist[len(hist) - k:]) if k > 0 else [],
                  "forecast": [] if display == "bands_only" else _nums(fc_mean)},
        "display": display,
        "flags": {"low_resolution": vm["flags"]["low_resolution"], "reference_only": vm["flags"]["reference_only"],
                  "tentative": vm["flags"]["tentative"], "synthetic": vm["flags"]["synthetic"]},
    }


def build_compare_vm(members: Sequence[SeriesInput], vms: Mapping[int, dict], ranking: pd.DataFrame,
                     name_to_index: Mapping[str, int], slots: Mapping[int, Optional[int]],
                     failed: Sequence[dict], lang: str) -> dict:
    """The complete D.4 compare VM in ``lang`` (only for views with 2+ requested series).

    Args:
        members: the successful series in request order.
        vms: ``{index: series VM in lang}`` for the same series (final slots already set); ranking row texts are
            taken from these VMs so card and detail never disagree (D.4.1).
        ranking: ``compare.rank_trends({display name: report})`` (best first).
        name_to_index: the ``{display name: index}`` dict built together with ``ranking``'s input (never re-parse
            names or rely on positions).
        slots: :func:`assign_slots` output.
        failed: :func:`build_failed_entry` objects of the failed series, in index order.
        lang: ``"ko"`` | ``"en"``.

    Keys: ``count`` (successful series), ``slots`` (``{"<index>": slot}``, string keys), ``value_kind``,
    ``source_line`` (:func:`compare_source_line`), ``axis_label``, ``axis_note``, ``scale`` (D.4.3), ``ranking``
    (D.4.1 rows), ``failed``, ``score_note``, ``spark_note``, ``notices``.

    ``notices`` holds only the scale notices this builder creates (``log_hint`` with ``ratio = 20``,
    ``index_unavailable``), result-level (``series_index = None``). They are **not** repeated in the result's
    top-level ``notices``; the client renders both lists.
    """
    lang = _lang(lang)
    by_index = {int(m.index): m for m in members}
    value_kinds = {m.source.value_kind for m in members}
    value_kind = members[0].source.value_kind if len(value_kinds) == 1 and members else "raw"
    freq = members[0].source.freq if members else "M"
    if value_kind in _INDEX_KINDS:
        axis_label = _value_kind_label(value_kind, None, lang)
    elif value_kind == "pageviews":
        axis_label = T.t(T.VALUE_KIND_LABELS, "pageviews", lang)
    else:
        units = {m.source.unit_label for m in members if m.source.unit_label}
        axis_label = units.pop() if len(units) == 1 else T.t(T.COMPARE_TEXTS, "axis_value", lang)
    if value_kind == "relative_index":
        unit_word = T.t(T.INDEX_UNIT_WORDS, freq if freq in T.INDEX_UNIT_WORDS else "W", lang)
        axis_note: Optional[str] = T.t(T.COMPARE_TEXTS, "axis_note_relative", lang, unit_word=unit_word)
    elif value_kind == "relative_index_gt":
        axis_note = T.t(T.COMPARE_TEXTS, "axis_note_gt", lang)
    else:
        axis_note = None

    if members:
        scale, notices = _scale(members, lang)
    else:
        scale = {"default": "raw", "allowed": {"raw": True, "index": False, "log": False},
                 "disabled_reason": {"log": None, "index": None}, "index_base": None, "log_hint": False}
        notices = []

    rows = []
    for rank, rec in ranking.iterrows():
        idx = int(name_to_index[str(rec["name"])])
        inp = by_index[idx]
        rows.append(_ranking_row(int(rank), rec, inp, vms[idx], slots.get(idx), lang))

    return {
        "count": len(members),
        "slots": {str(int(m.index)): slots.get(int(m.index)) for m in members},
        "value_kind": value_kind,
        "source_line": compare_source_line([m.source for m in members], lang),
        "axis_label": axis_label,
        "axis_note": axis_note,
        "scale": scale,
        "ranking": rows,
        "failed": [copy.deepcopy(dict(f)) for f in failed],
        "score_note": T.t(T.COMPARE_TEXTS, "score_note", lang),
        "spark_note": T.t(T.COMPARE_TEXTS, "spark_note", lang),
        "notices": notices,
    }


# ---------------------------------------------------------------------------------------------------------------------
# Export blocks (D.2.10, D.5) and response-time views (C.7, C.9)
# ---------------------------------------------------------------------------------------------------------------------

def export_availability(source_kind: str, *, xlsx: bool, view: str, succeeded: int) -> ExportAvailability:
    """Availability of every export for a result (D.2.10, D.5, C.10).

    Args:
        source_kind: one of :data:`SOURCE_KINDS`.
        xlsx: ``exports.xlsx_available()`` at job time.
        view: ``"detail"`` | ``"compare"``.
        succeeded: number of successful series.

    Rules: Naver -> ``html`` and ``values_csv`` unavailable (``export_naver_raw``); no openpyxl -> ``xlsx``
    unavailable (``export_xlsx_unavailable``); ``values_csv`` needs ``view == "compare"`` (otherwise unavailable
    without a reason); the result-level ``html`` of a compare view with a single successful series is that series'
    report (still available).
    """
    naver = source_kind == "naver"
    compare = view == "compare"
    has_series = int(succeeded) > 0
    series = {"csv": True, "xlsx": bool(xlsx), "html": not naver, "json": True}
    result = {"csv": has_series, "xlsx": bool(xlsx) and has_series, "html": (not naver) and has_series,
              "json": has_series, "values_csv": compare and not naver and has_series}
    reasons: dict[str, str] = {}
    if naver:
        reasons["html"] = "export_naver_raw"
        if compare:
            reasons["values_csv"] = "export_naver_raw"
    if not xlsx:
        reasons["xlsx"] = "export_xlsx_unavailable"
    return ExportAvailability(series=series, result=result, reasons=reasons)


def export_block(availability: ExportAvailability, lang: str, *, level: str) -> dict:
    """The **stored** ``exports`` object: ``level="series"`` -> keys ``csv, xlsx, html, json, unavailable`` (D.2.10);
    ``level="result"`` -> ``csv, xlsx, html, json, values_csv, unavailable`` (D.5). Every link is ``None``;
    ``unavailable[key] = {"code": code, "text": messages.render_web(code, {}, lang)}`` for each reason."""
    keys = EXPORT_FORMATS if level == "series" else RESULT_EXPORT_KEYS
    block: dict[str, Any] = {k: None for k in keys}
    flags = availability.series if level == "series" else availability.result
    block["unavailable"] = {k: {"code": code, "text": _render(code, {}, lang)}
                            for k, code in availability.reasons.items() if k in keys and not flags.get(k, False)}
    return block


def link_exports(block: Mapping[str, Any], flags: Mapping[str, bool], base: str, k: Optional[int] = None) -> dict:
    """Return a copy of a stored ``exports`` block with links filled for one response (E.7).

    Args:
        block: a stored block (:func:`export_block`).
        flags: ``ExportAvailability.series`` (with ``k``) or ``.result`` (without), as stored in ``result.json``.
        base: ``"/api/export/<result_id>"`` or ``"/api/export/s/<share_id>"`` (validated ids only).
        k: series index for a series block; ``None`` for the result block.

    Links: series ``f"{base}/{k}.{fmt}"``; result ``f"{base}.{fmt}"`` and ``values_csv`` ``f"{base}/values.csv"``;
    ``None`` where the flag is false. ``unavailable`` is copied unchanged.
    """
    out: dict[str, Any] = {}
    for key in block:
        if key == "unavailable":
            continue
        if not flags.get(key, False):
            out[key] = None
        elif k is not None:
            out[key] = f"{base}/{int(k)}.{key}"
        elif key == "values_csv":
            out[key] = f"{base}/values.csv"
        else:
            out[key] = f"{base}.{key}"
    out["unavailable"] = copy.deepcopy(dict(block.get("unavailable") or {}))
    return out


def make_shared_vm(vm: Mapping[str, Any]) -> dict:
    """Return a copy of a series VM as served from a share (C.9): ``narrative.ai_allowed = False``,
    ``ai_block_reason = "shared_readonly"``, ``ai_block_text`` = ``ai_readonly`` text in ``vm["lang"]``. Everything
    else (including an attached ``narrative.ai``) is unchanged."""
    out = copy.deepcopy(dict(vm))
    nar = out.setdefault("narrative", {})
    nar["ai_allowed"] = False
    nar["ai_block_reason"] = "shared_readonly"
    nar["ai_block_text"] = _render("ai_readonly", {}, str(out.get("lang", "ko")))
    return out
