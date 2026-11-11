"""Narrative generation: the "AI interpreter" of a trend.

Turns the numbers of an :class:`~trendai.types.AnalysisResult`,
:class:`~trendai.types.ForecastResult` and :class:`~trendai.types.Outlook`
into a readable past -> present -> future story.

Two engines are available:

* :func:`rule_based_narrative` -- deterministic, offline, Korean (default) or
  English text built from templates with concrete numbers.  Always works.
* :func:`llm_narrative` / :func:`ask` -- Claude (official ``anthropic`` SDK,
  imported lazily) writes the story, or answers a free-form question, using
  *only* a compact JSON of facts produced by :func:`build_llm_payload`.

:func:`generate_narrative` picks one and falls back to the rule-based text
whenever Claude is unavailable.

Output is "Markdown-lite": ``## `` headings, ``- `` bullets (sub-items of a
line ending in ``:`` are nested with a two-space indent, ``  - ``) and ``**bold**``.
"""

from __future__ import annotations

import json
import math
import re
import warnings
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd

from .data import periods_per_year
from .types import (
    PHASE_KO, AnalysisResult, Anomaly, Changepoint, ForecastResult, Outlook, TrendReport,
)

__all__ = [
    "DEFAULT_MODEL", "SYSTEM_PROMPT", "SECTION_HEADINGS", "NarrativeUnavailable",
    "rule_based_narrative", "build_llm_payload", "llm_narrative", "generate_narrative", "ask",
    "format_number", "format_pct", "format_date", "format_prob", "period_unit", "translate_reason",
]

DEFAULT_MODEL = "claude-opus-5-5"
_FALLBACK_BETA = "server-side-fallback-2026-07-01"
_MAX_TOKENS = 16000

#: Section headings of a narrative, per language (past, present, future, caveats).
SECTION_HEADINGS: dict[str, tuple[str, str, str, str]] = {
    "ko": ("과거: 어떻게 여기까지 왔나", "현재: 지금 어디에 있나", "미래 전망", "주의할 점"),
    "en": ("Past: how it got here", "Present: where it stands now", "Outlook", "Caveats"),
}

SYSTEM_PROMPT = """\
You are a senior trend analyst working inside "trendai", a tool that studies the past, the present \
and the likely future of a time series (search interest, page views, sales, prices, KPIs and so on). \
Each request gives you a JSON object of facts computed by trendai's statistical engine: a downsampled \
history, a robust trend decomposition, growth metrics, a Mann-Kendall trend test, structural \
changepoints, anomalies, a lifecycle-phase classification with its reasons, and a backtested forecast \
with 80% / 95% prediction intervals, a model leaderboard and an outlook summary. Your job is to turn \
those facts into a clear, honest explanation for a smart reader who is not a statistician.

Ground rules
- Use only the facts in the JSON. Do not present outside knowledge about the subject (news, companies, \
products, events, prices, causes) as fact. If a possible explanation for a change would help the \
reader, you may offer at most one or two, each clearly labelled as an unverified hypothesis \
("가설:" in Korean, "Hypothesis:" in English).
- Quantify. Cite concrete numbers, dates and units from the JSON: levels, total change and CAGR, \
slopes before and after each changepoint, peak and drawdown, recent change, momentum, the expected \
change at the horizon, interval bounds, the probability of increase. Round sensibly and never invent \
numbers that cannot be read or directly derived from the JSON.
- Be honest about uncertainty. Describe the future as a range built from the prediction intervals and \
the probability of increase, not as one certain number. State the forecast confidence and judge accuracy \
by forecast.skill: the backtest error of the final method divided by that of the best naive benchmark \
(last value, seasonal repeat or drift) on the same backtest windows, so skill 0.6 means about 40% lower \
error than naive and skill above 1 means it did not beat naive; if skill is null, do not claim it beats or \
trails a naive forecast. Never read the raw MASE against 1: it is scaled by the in-sample one-step \
seasonal-naive error, so multi-step forecasts often exceed 1 while beating naive; mention it at most as a \
technical detail. Never present a probability as a certainty: write >99% or <1% instead of 100% or 0%. If \
the series is short, noisy, recently broken by a changepoint, or the confidence is low, say so plainly.
- Dates in the JSON are already formatted for the series frequency; reuse them as given. Percentages \
are in percent units. The history is downsampled (older points are block averages), so do not \
over-interpret individual points in it.

Format of a full narrative
- Write in the language requested in the user message, in natural, fluent prose.
- Markdown-lite only: "## " headings, "- " bullets (sub-items under a line ending in ":" indented by \
two spaces, "  - ") and **bold** for the key numbers. No tables, code blocks, HTML or "# " titles.
- Start with a one- or two-sentence bottom line, then exactly four sections in this order, using the \
headings given in the user message: (1) past: data range, overall growth, peak and drawdown, each \
structural break with its date and what changed, notable anomalies, seasonality; (2) present: current \
lifecycle phase with its confidence and reasons, recent change, momentum, volatility; (3) outlook: \
expected change at the horizon, the 80% scenario range (pessimistic / base / optimistic), probability \
of increase, any forecast peak, the projected phase, forecast confidence and method; (4) caveats: \
uncertainty that widens with the horizon, the risk of structural breaks the model cannot foresee, \
data-quality limits, and a clear statement that this is not financial or investment advice.
- Aim for roughly 250-450 words; prefer short, information-dense bullets over long paragraphs.

Answering a question
- If the user message contains a question instead of a narrative request, answer that question \
directly in the first sentence, in the requested language, grounded in the same facts, then add only \
the supporting numbers and context that matter (a few bullets at most). The section structure is not \
required for answers. If the facts cannot answer the question, say what is missing instead of \
guessing. When the answer concerns the future, end with a one-line reminder of the uncertainty."""


class NarrativeUnavailable(RuntimeError):
    """An LLM narrative could not be produced (SDK missing, auth, network, refusal, ...)."""


# --------------------------------------------------------------------------- formatting helpers

_NA = {"ko": "해당 없음", "en": "n/a"}
_HANGUL = re.compile(r"[가-힣]")
_MINUS = "−"
_SCI_SMALL, _SCI_LARGE = 1e-4, 1e15
_SUPERSCRIPT = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")

_UNIT = {
    "ko": {"D": "/일", "W": "/주", "M": "/월", "Q": "/분기", "Y": "/년"},
    "en": {"D": "/day", "W": "/wk", "M": "/mo", "Q": "/qtr", "Y": "/yr"},
}
_PERIOD_KO = {"D": "일", "W": "주", "M": "개월", "Q": "분기", "Y": "년"}
_PERIOD_EN = {"D": ("day", "days"), "W": ("week", "weeks"), "M": ("month", "months"),
              "Q": ("quarter", "quarters"), "Y": ("year", "years")}
_FREQ_ADJ = {
    "ko": {"D": "일별", "W": "주별", "M": "월별", "Q": "분기별", "Y": "연도별"},
    "en": {"D": "daily", "W": "weekly", "M": "monthly", "Q": "quarterly", "Y": "yearly"},
}
PHASE_EN: dict[str, str] = {
    "emerging": "Emerging", "growing": "Growing", "peaking": "Peaking / maturing",
    "declining": "Declining", "stable": "Stable", "reviving": "Reviving", "volatile": "Volatile",
}
_DIR = {"ko": {"up": "상승", "down": "하락", "flat": "보합"},
        "en": {"up": "upward", "down": "downward", "flat": "flat"}}
# nominal form for terse lists ("예측 신뢰도 높음") ...
_CONF = {"ko": {"high": "높음", "medium": "보통", "low": "낮음"},
         "en": {"high": "high", "medium": "medium", "low": "low"}}
# ... and the predicate form for full sentences ("예측 신뢰도는 **높은 편**입니다")
_CONF_PRED = {"ko": {"high": "높은 편", "medium": "보통 수준", "low": "낮은 편"},
              "en": {"high": "high", "medium": "medium", "low": "low"}}
#: the benchmarks behind ForecastResult.skill, as the reader should picture them
_NAIVE_KO = "단순 기준 예측(직전 값 유지·계절 반복·추세 연장 중 가장 나은 것)"
_NAIVE_EN = "the best simple benchmark (repeat the last value, repeat the last season, or extend the straight line)"
#: stop reasons after which the text was cut off by a length limit
_TRUNCATED_STOPS = ("max_tokens", "model_context_window_exceeded")
_MONTH_EN = ("January", "February", "March", "April", "May", "June", "July", "August",
             "September", "October", "November", "December")
_WEEKDAY_KO = ("월요일", "화요일", "수요일", "목요일", "금요일", "토요일", "일요일")
_WEEKDAY_EN = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")


def _norm_lang(lang: str) -> str:
    key = str(lang or "ko").strip().lower()
    if key in ("ko", "kr", "kor", "korean", "ko-kr", "한국어"):
        return "ko"
    if key in ("en", "eng", "english", "en-us", "en-gb"):
        return "en"
    raise ValueError(f"unsupported language {lang!r}; use 'ko' or 'en'")


def _finite(x: Any) -> Optional[float]:
    """``float(x)`` if it is a finite number, else ``None``."""
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def format_number(x: Any, decimals: Optional[int] = None, lang: str = "ko",
                  signed: bool = False) -> str:
    """Human-friendly number: thousands separators, sensible decimals, typographic minus.

    ``decimals=None`` picks the precision from the magnitude (``1,234`` /
    ``45.2`` / ``1.20`` / ``0.02``); magnitudes below 1e-4 or from 1e15 up use
    ``1.05×10⁻⁹`` notation so tiny values never collapse to ``0``.  ``signed=True`` prefixes positive
    values with ``+``.  ``None`` / NaN / inf become ``'해당 없음'`` (ko) or
    ``'n/a'`` (en).
    """
    lang = _norm_lang(lang)
    v = _finite(x)
    if v is None:
        return _NA[lang]
    a = abs(v)
    if decimals is None and (0 < a < _SCI_SMALL or a >= _SCI_LARGE):
        # fixed notation would print tiny values as "0" (or 20-digit integers): use ×10ⁿ
        mant, exp = f"{a:.2e}".split("e")
        mant = mant.rstrip("0").rstrip(".")
        e = int(exp)
        sup = ("⁻" if e < 0 else "") + str(abs(e)).translate(_SUPERSCRIPT)
        sign = _MINUS if v < 0 else ("+" if signed else "")
        return f"{sign}{mant}×10{sup}"
    strip = False
    if decimals is not None:
        d = max(0, int(decimals))
    elif a >= 1000:
        d = 0
    elif a >= 10:
        d = 1
    elif a >= 0.1:
        d = 2
    elif a == 0:
        d = 0
    else:  # small numbers: two significant digits, trailing zeros removed
        d = min(8, -int(math.floor(math.log10(a))) + 1)
        strip = True
    r = round(v, d)
    body = f"{abs(r):,.{d}f}"
    if strip and "." in body:
        body = body.rstrip("0").rstrip(".")
    if r == 0 or body.strip("0.,") == "":
        return "0" if d == 0 or strip else f"{0:.{d}f}"
    sign = _MINUS if r < 0 else ("+" if signed else "")
    return sign + body


def format_pct(x: Any, lang: str = "ko", decimals: int = 1, signed: bool = True) -> str:
    """A percentage in percent units, signed by default: ``+12.3%`` / ``−4.0%``."""
    lang = _norm_lang(lang)
    if _finite(x) is None:
        return _NA[lang]
    return format_number(x, decimals=decimals, lang=lang, signed=signed) + "%"


def format_prob(p: Any, lang: str = "ko") -> str:
    """A 0..1 probability / confidence as a whole percent: ``0.8`` -> ``'80%'``.

    Values that would round to a certainty are never shown as one: ``p >= 0.995``
    becomes ``'>99%'`` and ``p <= 0.005`` becomes ``'<1%'`` -- including exactly 1
    and 0, because these are model estimates (a normal approximation that
    saturates in floating point, or a degenerate zero-width interval), never
    actual certainty.  Same symbols as the report's KPI tiles.  ``None`` / NaN
    become ``'해당 없음'`` / ``'n/a'``.
    """
    lang = _norm_lang(lang)
    v = _finite(p)
    if v is None:
        return _NA[lang]
    v = min(1.0, max(0.0, v))
    if v >= 0.995:
        return ">99%"
    if v <= 0.005:
        return "<1%"
    return f"{v * 100:.0f}%"


def format_date(ts: Any, freq: str, lang: str = "ko") -> str:
    """Format a timestamp for ``freq``: D/W ``2024-03-05``, M ``2024-03``, Q ``2024-Q1``, Y ``2024``."""
    lang = _norm_lang(lang)
    if ts is None:
        return _NA[lang]
    try:
        t = pd.Timestamp(ts)
    except (TypeError, ValueError):
        return str(ts)
    if pd.isna(t):
        return _NA[lang]
    if freq == "M":
        return f"{t.year:04d}-{t.month:02d}"
    if freq == "Q":
        return f"{t.year:04d}-Q{t.quarter}"
    if freq == "Y":
        return f"{t.year:04d}"
    return f"{t.year:04d}-{t.month:02d}-{t.day:02d}"


def period_unit(freq: str, lang: str = "ko") -> str:
    """Per-period unit suffix: ``/월`` ``/주`` ``/일`` ``/분기`` ``/년`` (or ``/mo`` ... in English)."""
    return _UNIT[_norm_lang(lang)].get(freq, "")


def _span(k: int, freq: str, lang: str) -> str:
    """``12개월`` / ``12 months`` style duration of ``k`` periods."""
    k = int(k)
    if lang == "ko":
        return f"{k:,}{_PERIOD_KO.get(freq, '기간')}"
    one, many = _PERIOD_EN.get(freq, ("period", "periods"))
    return f"{k:,} {one if k == 1 else many}"


def _span_adj(k: int, freq: str) -> str:
    """English compound adjective for ``k`` periods: ``26-week``, ``12-month``."""
    return f"{int(k):,}-{_PERIOD_EN.get(freq, ('period', 'periods'))[0]}"


def _ppy(freq: str) -> float:
    try:
        return periods_per_year(freq)
    except Exception:  # unknown freq code: assume monthly
        return 12.0


# --------------------------------------------------------------------------- shared derived facts


def _current_level(analysis: AnalysisResult) -> Optional[float]:
    """Mean of the last ``min(season_length, n)`` observations (as in ``compute_outlook``)."""
    vals = np.asarray(analysis.series, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return None
    k = min(max(1, int(analysis.season_length)), len(vals))
    return float(np.mean(vals[-k:]))


def _scale(analysis: AnalysisResult) -> float:
    """Typical magnitude of the series (mean absolute value; 0 for an all-zero series)."""
    vals = np.asarray(analysis.series, dtype=float)
    vals = vals[np.isfinite(vals)]
    return float(np.mean(np.abs(vals))) if len(vals) else 0.0


def _interval_level(forecast: ForecastResult) -> Optional[int]:
    levels = [lv for lv in forecast.levels if lv in forecast.lower and lv in forecast.upper]
    if not levels:
        return None
    return 80 if 80 in levels else int(min(levels))  # plain int: it goes into the JSON payload


def _level_shift_jump(analysis: AnalysisResult, cp: Changepoint) -> Optional[float]:
    """Size of the level jump at a changepoint, in series units.

    Uses short deseasonalised windows on each side of the break and moves each
    window mean to the break position with the changepoint's own segment
    slopes, so that an ongoing trend is not mistaken for (or does not hide) a
    jump: e.g. a drop of 30 inside a +2/period uptrend is reported as about
    -30, not as the trend-contaminated difference of the window means.
    """
    y = np.asarray(analysis.series, dtype=float)
    seas = np.asarray(analysis.decomposition.seasonal, dtype=float)
    if len(seas) == len(y):
        y = y - np.where(np.isfinite(seas), seas, 0.0)
    i = int(cp.index)
    if not 0 < i < len(y):
        return None
    w = max(2, min(6, max(1, int(analysis.season_length)) // 2 or 3))
    b_idx = np.arange(max(0, i - w), i)
    a_idx = np.arange(i, min(len(y), i + w))
    b_idx, a_idx = b_idx[np.isfinite(y[b_idx])], a_idx[np.isfinite(y[a_idx])]
    if len(b_idx) == 0 or len(a_idx) == 0:
        return None
    s_before = _finite(cp.slope_before) or 0.0
    s_after = _finite(cp.slope_after) or 0.0
    level_before = float(np.mean(y[b_idx])) + s_before * (i - float(np.mean(b_idx)))
    level_after = float(np.mean(y[a_idx])) - s_after * (float(np.mean(a_idx)) - i)
    jump = level_after - level_before
    return jump if math.isfinite(jump) else None


@dataclass
class _Ctx:
    a: AnalysisResult
    f: ForecastResult
    o: Outlook
    lang: str

    @property
    def freq(self) -> str:
        return self.a.freq

    @property
    def m(self):  # TrendMetrics
        return self.a.metrics

    def t(self, ko: str, en: str) -> str:
        return ko if self.lang == "ko" else en

    def num(self, x: Any, decimals: Optional[int] = None, signed: bool = False) -> str:
        return format_number(x, decimals=decimals, lang=self.lang, signed=signed)

    def pct(self, x: Any, decimals: int = 1, signed: bool = True) -> str:
        return format_pct(x, lang=self.lang, decimals=decimals, signed=signed)

    def date(self, ts: Any) -> str:
        """Date for prose: :func:`format_date`, plus the counter ``년`` for yearly data in
        Korean (``2023년부터``, ``2028년 예측값``; a bare ``2023부터`` reads unnaturally)."""
        s = format_date(ts, self.freq, self.lang)
        if self.lang == "ko" and self.freq == "Y" and s.isdigit():
            return s + "년"
        return s

    @property
    def unit(self) -> str:
        return period_unit(self.freq, self.lang)

    def span(self, k: int) -> str:
        return _span(k, self.freq, self.lang)

    def phase_name(self, label: str) -> str:
        return PHASE_KO.get(label, label) if self.lang == "ko" else PHASE_EN.get(label, label)

    def direction(self, d: str) -> str:
        return _DIR[self.lang].get(d, d)

    def recent_window(self) -> int:
        return _recent_window(self.freq, int(self.a.season_length), int(self.m.n_obs))

    def name(self) -> str:
        """Series name on one line (a newline in the name would break the Markdown-lite structure)."""
        return " ".join(str(self.a.name or "series").split()) or "series"


def _recent_window(freq: str, season_len: int, n_obs: int) -> int:
    """Periods covered by ``TrendMetrics.recent_change_pct``: one season (1 for yearly
    data), capped at ``n_obs - 1`` like ``analysis.compute_metrics``."""
    lag = 1 if freq == "Y" else max(1, int(season_len))
    return max(1, min(lag, int(n_obs) - 1))


def _undefined_pct_reason(base: Optional[float], lang: str) -> str:
    """Why a % change is undefined: the base is negative/zero, or close to zero."""
    if base is not None and base < 0:
        return "기준 수준이 음수여서" if lang == "ko" else "the base level is negative"
    return "기준 수준이 0에 가까워" if lang == "ko" else "the base level is close to zero"


def _usable_season(analysis: AnalysisResult) -> Optional[int]:
    """Seasonal period whose component ``forecast.compute_outlook`` uses for its
    partial-season adjustment (same conditions as ``forecast._seasonal_component``)."""
    dec = analysis.decomposition
    try:
        p = int(dec.period)
        seas = np.asarray(dec.seasonal, dtype=float)
    except (TypeError, ValueError):
        return None
    if (p <= 1 or seas.ndim != 1 or len(seas) != len(analysis.series) or len(seas) < p
            or not np.all(np.isfinite(seas)) or not np.any(seas)):
        return None
    return p


@dataclass
class _OutlookWindows:
    """The two windows behind ``Outlook.expected_change_pct`` (see ``forecast.compute_outlook``)."""

    k_fut: int            # last min(season, horizon) forecast points
    k_cur: int            # last min(season, n) observations
    horizon: int
    period: Optional[int]  # seasonal period of the adjustment (None: no usable seasonal component)
    adj_fut: bool         # forecast window is a partial season -> its seasonal effect is removed
    adj_cur: bool         # current window is a partial season -> its seasonal effect is removed


def _outlook_windows(analysis: AnalysisResult, forecast: ForecastResult) -> _OutlookWindows:
    """Mirror of the window logic of ``forecast.compute_outlook``.

    current level = mean of the last ``min(m, n)`` observations, future level =
    mean of the last ``min(m, H)`` forecast means (``m`` = season length); a
    window that is not a whole number of seasonal periods has the average
    seasonal effect over it removed (for the forecast window scaled by how much
    seasonality the forecast contains).
    """
    m = max(1, int(analysis.season_length))
    vals = np.asarray(analysis.series, dtype=float)
    n = max(1, int(np.isfinite(vals).sum()))
    h = len(forecast.mean) or max(1, int(forecast.horizon))
    k_fut, k_cur = min(m, h), min(m, n)
    p = _usable_season(analysis)
    return _OutlookWindows(k_fut=k_fut, k_cur=k_cur, horizon=h, period=p,
                           adj_fut=p is not None and k_fut % p != 0,
                           adj_cur=p is not None and k_cur % p != 0)


def _trend_drawdown(analysis: AnalysisResult) -> Optional[float]:
    """Latest trend value versus the trend maximum, in % (<= 0); ``None`` if undefined."""
    t = np.asarray(analysis.decomposition.trend, dtype=float)
    t = t[np.isfinite(t)]
    if len(t) < 2:
        return None
    top = float(np.max(t))
    if not top > 0:
        return None
    dd = (float(t[-1]) - top) / top * 100.0
    return min(0.0, dd) if math.isfinite(dd) else None


def _drawdown_words(dd: float) -> tuple[str, str]:
    """(Korean predicate, English phrase) for a drawdown of ``dd`` % (<= 0) from the peak."""
    if dd > -5:
        return "여전히 고점 부근입니다", "still close to the high"
    if dd > -20:
        return "다소 내려와 있습니다", "somewhat below the high"
    if dd > -50:
        return "상당히 내려와 있습니다", "well below the high"
    if dd > -80:
        return "고점의 절반 이하로 떨어져 있습니다", "less than half of the peak"
    if dd > -90:
        return "고점의 1/5 이하로 급감했습니다", "less than a fifth of the peak"
    return "고점의 1/10 이하로 급감했습니다", "less than a tenth of the peak"


# --------------------------------------------------------------------------- phase reasons in English
#
# analysis.lifecycle_phase returns Korean reason sentences (the TrendMetrics /
# LifecyclePhase contract).  For English narratives they are translated with the
# templates below, so the English evidence states the same reasons as the
# analysis (and the report's 현재 진단 card) instead of unrelated metrics.  A
# sentence that matches no template is left out.

_RX_NUM = r"[+−-]?[\d,]*\.?\d+(?:[eE][+−-]?\d+)?"
_RX_PCT = _RX_NUM + "%"
_RX_DATE = r"\d{4}(?:-Q[1-4]|-\d{2}(?:-\d{2})?)?"
_RX_UNIT = r"(?:개월|주|일|분기|년|기간)"
_RX_P = r"p (?:<|=) [\d.]+"
_UNIT_EN = {"개월": ("month", "months"), "주": ("week", "weeks"), "일": ("day", "days"),
            "분기": ("quarter", "quarters"), "년": ("year", "years"), "기간": ("period", "periods")}
_KIND_EN = {"가속": "acceleration", "감속": "deceleration", "상승 전환": "turn upward",
            "하락 전환": "turn downward", "수준 변화": "level shift"}


def _en_span(k: str, unit_ko: str) -> str:
    one, many = _UNIT_EN.get(unit_ko, ("period", "periods"))
    return f"{k} {one if k.replace(',', '') == '1' else many}"


def _en_when(date: str) -> str:
    """``on 2024-03-05`` for a day, ``in 2024-03`` / ``in 2024-Q1`` / ``in 2024`` otherwise."""
    return f"on {date}" if re.fullmatch(r"\d{4}-\d{2}-\d{2}", date) else f"in {date}"


def _en_scale_amount(txt: str) -> str:
    """``+25.0%`` stays; ``추세 규모의 12.0%만큼`` -> ``by 12.0% of the trend's scale``."""
    mt = re.fullmatch(rf"추세 규모의 ({_RX_NUM})%만큼", txt.strip())
    return f"by {mt.group(1)}% of the trend's scale" if mt else txt.strip()


_REASON_EN: tuple[tuple[re.Pattern, Any], ...] = tuple((re.compile(p), fn) for p, fn in (
    (r"관측치가 ([\d,]+)개뿐이라 추세를 판단하기에 데이터가 부족합니다\.?",
     lambda g: f"Only {g[0]} observations: too few to judge the trend."),
    (rf"관측 기간 동안 값이 ({_RX_NUM})에서 ({_RX_NUM})로 ({_RX_PCT}) 변했습니다\.?",
     lambda g: f"Over the observed period the value moved from {g[0]} to {g[1]} ({g[2]})."),
    (rf"현재 값은 ({_RX_NUM})입니다\.?", lambda g: f"The current value is {g[0]}."),
    (rf"직선 추세 대비 잡음 비율은 ({_RX_NUM})%입니다\.?",
     lambda g: f"Noise relative to a straight-line trend is {g[0]}%."),
    (rf"최근 ([\d,]+)({_RX_UNIT}) 동안 추세가 ({_RX_PCT}) 변했습니다\.?",
     lambda g: f"Over the last {_en_span(g[0], g[1])} the trend changed by {g[2]}."),
    (rf"최근 ([\d,]+)({_RX_UNIT}) 동안 추세 기울기는 ({_RX_NUM})/({_RX_UNIT})입니다\.?",
     lambda g: f"Over the last {_en_span(g[0], g[1])} the trend slope was {g[2]} per "
               f"{_UNIT_EN.get(g[3], ('period',))[0]}."),
    (rf"현재 추세는 고점\(({_RX_DATE})\)보다 ({_RX_NUM})% 낮은 수준입니다\.?",
     lambda g: f"The trend is {g[1]}% below its peak ({g[0]})."),
    (rf"추세 고점\(({_RX_DATE})\) 대비 현재 수준이 추세 규모\(최대 절댓값\)의 ({_RX_NUM})%만큼 낮습니다\.?",
     lambda g: f"The trend is below its peak ({g[0]}) by {g[1]}% of the trend's scale (its largest absolute value)."),
    (r"추세는 저점 대비 (.+?) 올라 현재 역대 최고 수준입니다\.?",
     lambda g: f"The trend has risen {_en_scale_amount(g[0])} from its low and is at an all-time high."),
    (rf"추세는 저점 대비 (.+?) 오른 뒤 ({_RX_DATE})에 고점을 형성했습니다\.?",
     lambda g: f"The trend rose {_en_scale_amount(g[0])} from its low and peaked {_en_when(g[1])}."),
    (rf"Mann-Kendall 검정에서 전체 기간의 (상승|하락) 추세가 유의합니다 \(τ = ({_RX_NUM}), ({_RX_P})\)\.?",
     lambda g: f"The Mann-Kendall test finds a significant {'upward' if g[0] == '상승' else 'downward'} "
               f"trend over the whole period (τ = {g[1]}, {g[2]})."),
    (rf"Mann-Kendall 검정에서 전체 기간의 뚜렷한 추세가 확인되지 않습니다 \(({_RX_P})\)\.?",
     lambda g: f"The Mann-Kendall test finds no clear trend over the whole period ({g[0]})."),
    (r"변화가 누적되는 무작위 행보\(random walk\)에 가까운 움직임이라, 추세처럼 보이는 등락도 우연일 수 있습니다\.?",
     lambda g: "Changes accumulate like a random walk, so trend-like swings may be due to chance."),
    (rf"({_RX_DATE})에 수준이 한 단계 (높아졌습니다|낮아졌습니다) \(약 ([^)]+)\)\.?",
     lambda g: f"{_en_when(g[0]).capitalize()} the level stepped {'up' if g[1] == '높아졌습니다' else 'down'} (about {g[2]})."),
    (rf"그 이후 ([\d,]+)({_RX_UNIT}) 동안은 뚜렷한 추세 없이 새 수준에서 안정되어 있습니다 "
     rf"\(해당 구간 Mann-Kendall ({_RX_P})\)\.?",
     lambda g: f"Since then it has held at the new level for {_en_span(g[0], g[1])} with no clear trend "
               f"(Mann-Kendall on that stretch: {g[2]})."),
    (rf"잡음이 커서\(추정 잡음 수준이 평균의 ({_RX_NUM})%\) (?:뚜렷한 방향성이 없습니다|방향성이 불분명합니다)\.?",
     lambda g: f"Noise is large (estimated at {g[0]}% of the mean), so there is no clear direction."),
    (rf"추세의 변동 폭\(최고-최저\)이 최고 수준 대비 ({_RX_NUM})%에 그칩니다\.?",
     lambda g: f"The trend's range (high minus low) is only {g[0]}% of its highest level."),
    (rf"계절성 강도가 ({_RX_NUM})로 주기적 변동이 뚜렷합니다\.?",
     lambda g: f"Seasonal strength is {g[0]}: a clear recurring cycle."),
    (rf"고점\(({_RX_DATE})\) 이후 하락해 ({_RX_DATE})에 저점을 찍은 뒤 (.+?) 반등했습니다\.?",
     lambda g: f"After the peak ({g[0]}) it fell to a low {_en_when(g[1])}, then rebounded {_en_scale_amount(g[2])}."),
    (rf"다만 최근 ([\d,]+)({_RX_UNIT}) 동안 비슷한 수준에서 안정되어 있습니다\.?",
     lambda g: f"However, it has held at a similar level for the last {_en_span(g[0], g[1])}."),
    (rf"최근 ([\d,]+)({_RX_UNIT}) 동안은 낮은 수준에서 정체되어 있습니다\.?",
     lambda g: f"It has stagnated at a low level for the last {_en_span(g[0], g[1])}."),
    (rf"최근 기울기가 과거 상승기 기울기\(상위 10%\)의 ({_RX_NUM})% 수준으로 둔화되었습니다\.?",
     lambda g: f"The recent slope has slowed to {g[0]}% of the steepest past growth (top 10% of slopes)."),
    (rf"최근 기울기가 과거 상승기 기울기\(상위 10%\)의 ({_RX_NUM})% 수준을 유지하고 있습니다\.?",
     lambda g: f"The recent slope is still {g[0]}% of the steepest past growth (top 10% of slopes)."),
    (rf"현재 추세 수준이 과거 기반 수준의 ({_RX_NUM})배로, 낮은 기반에서 빠르게 성장 중입니다\.?",
     lambda g: f"The trend level is {g[0]}× its earlier base: fast growth from a low base."),
    (rf"최근 상승 속도가 전체 기간 평균 속도\(추세선 처음→끝\)의 ({_RX_NUM})배로 가속되고 있습니다\.?",
     lambda g: f"The recent rise is {g[0]}× the average pace of the whole period (trend start to end): accelerating."),
    (rf"현재 수준은 추세 고점의 ({_RX_NUM})% 수준입니다\.?",
     lambda g: f"The current level is {g[0]}% of the trend peak."),
    (rf"({_RX_DATE})에 구조 변화\(([^)]+)\)가 감지되었습니다\.?",
     lambda g: f"A structural break ({_KIND_EN.get(g[1], 'trend change')}) was detected {_en_when(g[0])}."),
))


def translate_reason(reason: str) -> Optional[str]:
    """English version of one Korean phase reason from ``analysis.lifecycle_phase``.

    Returns the sentence unchanged if it contains no Hangul, the translation if
    it matches a known template, and ``None`` otherwise.
    """
    s = " ".join(str(reason or "").split())
    if not s:
        return None
    if not _HANGUL.search(s):
        return s
    for rx, fn in _REASON_EN:
        mt = rx.fullmatch(s)
        if mt:
            out = fn(mt.groups())
            return None if _HANGUL.search(out) else out
    return None


# --------------------------------------------------------------------------- rule-based narrative


def _pct0(x: Any, lang: str) -> str:
    """A 0..1 fraction as a whole percent, never a false certainty (see :func:`format_prob`)."""
    return format_prob(x, lang)


def _flat_despite_change(c: _Ctx, chg: Optional[float]) -> bool:
    """Direction rated 'flat' although the expected change is visibly non-zero
    (the change is within the forecast noise band) -- needs explaining, not just
    "+33% (flat)"."""
    return c.o.direction == "flat" and chg is not None and abs(chg) >= 2.0


def _summary_line(c: _Ctx) -> str:
    name = c.name()
    conf = _pct0(c.a.phase.confidence, c.lang)
    prob = _pct0(c.o.prob_increase, c.lang)
    h = c.span(c.f.horizon)
    chg = _finite(c.o.expected_change_pct)
    conf_word = _CONF[c.lang].get(c.o.confidence, c.o.confidence)
    dir_word = c.direction(c.o.direction)
    if c.lang == "ko":
        head = f"**한눈에 보기** — {name}: 현재 **{c.phase_name(c.a.phase.label)}** 단계(신뢰도 {conf})이며, "
        if _flat_despite_change(c, chg):
            head += (f"향후 {h} 동안 **{c.pct(chg)}** 변화가 예상되지만 예측 불확실성에 비해 크지 않아 "
                     f"방향은 **{dir_word}**으로 봅니다.")
        elif chg is not None:
            head += f"향후 {h} 동안 **{c.pct(chg)}** 변화({dir_word})가 예상됩니다."
        else:
            head += f"향후 {h} 전망은 **{dir_word}**입니다."
        return head + f" 상승 확률 {prob}, 예측 신뢰도 {conf_word}."
    head = (f"**At a glance** — {name}: currently **{c.phase_name(c.a.phase.label)}** "
            f"(confidence {conf}); ")
    if _flat_despite_change(c, chg):
        head += (f"over the next {h} the level is expected to change by **{c.pct(chg)}**, but that is small "
                 f"relative to the forecast uncertainty, so the direction is rated **{dir_word}**.")
    elif chg is not None:
        head += f"over the next {h} the level is expected to change by **{c.pct(chg)}** ({dir_word})."
    else:
        head += f"the outlook for the next {h} is **{dir_word}**."
    return head + f" Probability of increase {prob}, forecast confidence {conf_word}."


def _changepoint_line(c: _Ctx, cp: Changepoint) -> str:
    b, a = float(cp.slope_before), float(cp.slope_after)
    flat_tol = 0.1 * max(abs(a), abs(b), 1e-12)
    slopes = f"{c.num(b)} → {c.num(a)}{c.unit}"
    mag = c.num(cp.magnitude, decimals=2)
    d = c.date(cp.date)
    kind = cp.kind
    if kind == "level_shift":
        jump = _level_shift_jump(c.a, cp)
        up = jump is None or jump >= 0
        size = "" if jump is None else c.t(f" (약 {c.num(jump, signed=True)})",
                                             f" (about {c.num(jump, signed=True)})")
        desc = c.t(f"수준이 한 단계 {'높아졌습니다' if up else '낮아졌습니다'}{size}",
                   f"the level {'jumped up' if up else 'dropped'}{size}")
        return c.t(f"- **{d}**에 {desc} — 기울기 {slopes}, 변화 강도 {mag}",
                   f"- In **{d}** {desc} — slope {slopes}, strength {mag}")
    if kind == "acceleration":
        ko, en = ("성장 가속", "growth accelerated") if a >= 0 else ("하락 가속", "the decline accelerated")
    elif kind == "deceleration":
        ko, en = ("성장 둔화", "growth slowed") if b > 0 else ("하락세 둔화", "the decline eased")
    elif kind == "reversal_up":
        ko, en = (("하락에서 상승으로 반전", "turned from decline to growth") if b < -flat_tol
                  else ("횡보에서 상승으로 전환", "turned from flat to growth"))
    elif kind == "reversal_down":
        ko, en = (("상승에서 하락으로 반전", "turned from growth to decline") if b > flat_tol
                  else ("횡보에서 하락으로 전환", "turned from flat to decline"))
    else:
        ko, en = ("추세 변화", "the trend changed")
    return c.t(f"- **{d}**부터 {ko} (기울기 {slopes}, 변화 강도 {mag})",
               f"- From **{d}**: {en} (slope {slopes}, strength {mag})")


def _anomaly_line(c: _Ctx, an: Anomaly) -> str:
    value, expected = _finite(an.value), _finite(an.expected)
    diff = ""
    if value is not None and expected is not None and abs(expected) > 1e-6 * _scale(c.a):
        diff = f"{c.pct((value - expected) / abs(expected) * 100)}, "
    z = c.num(an.zscore, decimals=1, signed=True)
    spike = float(an.zscore) >= 0
    return c.t(
        f"- {c.date(an.date)}: 실제 {c.num(value)}, 예상 {c.num(expected)} ({diff}z = {z}) — "
        f"{'예상보다 크게 높았던 일시적 급등' if spike else '예상보다 크게 낮았던 일시적 급락'}",
        f"- {c.date(an.date)}: actual {c.num(value)} vs. expected {c.num(expected)} ({diff}z = {z}) — "
        f"{'an unusual spike' if spike else 'an unusual dip'}",
    )


def _seasonal_pattern(c: _Ctx) -> Optional[tuple[str, str, str]]:
    """(high name, low name, English preposition) of the seasonal cycle, if meaningful."""
    dec = c.a.decomposition
    if dec.period <= 1 or float(c.m.seasonal_strength) < 0.3:
        return None
    s = dec.seasonal
    if len(s) < 2 or not isinstance(s.index, pd.DatetimeIndex):
        return None
    vals = pd.Series(np.asarray(s, dtype=float), index=s.index)
    vals = vals[np.isfinite(vals.to_numpy())]
    if len(vals) < 2:
        return None
    if c.freq in ("M", "W"):
        key = vals.index.month
        names = [f"{i}월" for i in range(1, 13)] if c.lang == "ko" else list(_MONTH_EN)
        offset, prep = 1, "in"
    elif c.freq == "Q":
        key = vals.index.quarter
        names = [f"{i}분기" for i in range(1, 5)] if c.lang == "ko" else [f"Q{i}" for i in range(1, 5)]
        offset, prep = 1, "in"
    elif c.freq == "D" and dec.period == 7:
        key = vals.index.dayofweek
        names = list(_WEEKDAY_KO) if c.lang == "ko" else [f"{w}s" for w in _WEEKDAY_EN]
        offset, prep = 0, "on"
    else:
        return None
    g = vals.groupby(key).mean()
    if len(g) < 2 or not np.isfinite(g.to_numpy()).all() or float(g.max() - g.min()) <= 0:
        return None
    hi, lo = int(g.idxmax()) - offset, int(g.idxmin()) - offset
    return names[hi], names[lo], prep


def _seasonal_drawdown(c: _Ctx, dd: float) -> Optional[float]:
    """Trend-line drawdown (%) when the raw drawdown ``dd`` is distorted by seasonality.

    For a clearly seasonal series (strength >= 0.5) the latest observation may sit
    in a seasonal low or high, so "x % below the peak" mostly measures the
    seasonal swing.  Returns the drawdown of the trend component when it differs
    from the raw one by at least 5 percentage points, else ``None``.
    """
    if c.a.decomposition.period <= 1 or (_finite(c.m.seasonal_strength) or 0.0) < 0.5:
        return None
    tdd = _trend_drawdown(c.a)
    if tdd is None or abs(dd - tdd) < 5.0:
        return None
    return tdd


def _past_section(c: _Ctx) -> list[str]:
    m = c.m
    lines: list[str] = []
    # 1. data range
    years = float(m.n_obs) / _ppy(c.freq)
    if years >= 1:
        yr = f"{years:.1f}".rstrip("0").rstrip(".")
        dur = c.t(f"{yr}년", f"{yr} year{'s' if yr != '1' else ''}")
    else:
        dur = c.span(m.n_obs)
    adj = _FREQ_ADJ[c.lang].get(c.freq, "")
    lines.append(c.t(
        f"- {c.date(m.start)}부터 {c.date(m.end)}까지 약 {dur} 동안의 {adj} 데이터 {m.n_obs:,}개를 분석했습니다.",
        f"- Analysed {m.n_obs:,} {adj} observations from {c.date(m.start)} to {c.date(m.end)} (about {dur}).",
    ))
    # 2. overall change on the trend line
    trend = np.asarray(c.a.decomposition.trend, dtype=float)
    trend = trend[np.isfinite(trend)]
    t0 = c.num(trend[0]) if len(trend) else c.num(None)
    t1 = c.num(trend[-1]) if len(trend) else c.num(None)
    total, cagr = _finite(m.total_change_pct), _finite(m.cagr_pct)
    if total is not None:
        if c.lang == "ko":
            s = f"- 추세선 기준으로 {t0}에서 {t1}까지 **{c.pct(total)}** 변했"
            s += f"고, 연평균 성장률(CAGR)은 **{c.pct(cagr)}**입니다." if cagr is not None else "습니다."
        else:
            s = f"- On the trend line the level moved from {t0} to {t1}, a total change of **{c.pct(total)}**"
            s += f" (**{c.pct(cagr)}** per year, CAGR)." if cagr is not None else "."
        lines.append(s)
    else:
        # % change undefined: say why (negative base vs. ~0 base) and give the absolute change
        base = float(trend[0]) if len(trend) else None
        diff = c.num(float(trend[-1] - trend[0]), signed=True) if len(trend) else c.num(None)
        reason = _undefined_pct_reason(base, c.lang)
        lines.append(c.t(
            f"- 추세선 기준으로 {t0}에서 {t1}까지 {diff} 변했지만, 시작 시점의 {reason} 변화율(%)은 정의되지 않습니다.",
            f"- On the trend line the level moved from {t0} to {t1} ({diff}); the total % change is undefined "
            f"because {reason.replace('the base level', 'the starting level')}.",
        ))
    # 3. significance + long-run slope
    p = _finite(m.mk_pvalue)
    pstr = "< 0.001" if p is not None and p < 0.001 else f"= {c.num(p, decimals=3)}"
    tau = c.num(m.mk_tau, decimals=2)
    slope = f"{c.num(m.slope, signed=True)}{c.unit}"
    slope_pct = f"{c.pct(m.slope_pct, decimals=2)}{c.unit}"
    if c.lang == "ko":
        verdict = {"up": "통계적으로 유의한 **상승** 추세가 확인됩니다",
                   "down": "통계적으로 유의한 **하락** 추세가 확인됩니다"}.get(
            m.direction, "통계적으로 유의한 장기 추세는 확인되지 않습니다")
        lines.append(f"- Mann-Kendall 검정에서 {verdict} (τ = {tau}, p {pstr}). "
                     f"장기 평균 기울기는 {slope}(평균 수준 대비 {slope_pct})입니다.")
    else:
        verdict = {"up": "a statistically significant **upward** trend",
                   "down": "a statistically significant **downward** trend"}.get(
            m.direction, "no statistically significant long-run trend")
        lines.append(f"- The Mann-Kendall test finds {verdict} (τ = {tau}, p {pstr}); "
                     f"the long-run slope is {slope} ({slope_pct} of the mean level).")
    # 4. peak and drawdown
    dd = _finite(m.drawdown_pct) or 0.0
    pdate, pval, last = c.date(m.peak_date), c.num(m.peak_value), c.num(m.last_value)
    at_peak = dd > -0.5 or pd.Timestamp(m.peak_date) >= pd.Timestamp(m.end)
    obs = np.asarray(c.a.series, dtype=float)
    obs = obs[np.isfinite(obs)]
    constant = len(obs) > 0 and float(np.ptp(obs)) <= 1e-12 * max(1.0, float(np.max(np.abs(obs))))
    if constant:  # "at its all-time high" would be technically true but misleading
        lines.append(c.t(f"- 관측 기간 내내 값이 변하지 않았습니다 (항상 {c.num(obs[-1])}).",
                         f"- The value stayed at {c.num(obs[-1])} over the whole period."))
    elif at_peak:
        lines.append(c.t(f"- 최고치는 {pdate}의 {pval}이며, 현재가 사실상 역대 최고 수준입니다.",
                         f"- The peak was {pval} {_en_when(pdate)} — the series is at (or right next to) its all-time high now."))
    else:
        below = c.pct(abs(dd), signed=False)
        seasonal_dd = _seasonal_drawdown(c, dd)
        if seasonal_dd is not None:
            # a strongly seasonal series: much of the gap to the peak is the seasonal swing
            # (e.g. December, the seasonal low, versus a March high), so judge the trend line
            tdd = seasonal_dd
            t_ko, t_en = _drawdown_words(tdd)
            ss = float(m.seasonal_strength)
            if dd < tdd:
                why_ko = "최근 시점이 계절적으로 낮은 시기여서 이 차이의 상당 부분은 계절적 등락입니다"
                why_en = "the latest period falls in a seasonal low, which exaggerates the gap"
            else:
                why_ko = "최근 시점이 계절적으로 높은 시기여서 실제 하락 폭이 일부 가려져 있습니다"
                why_en = "the latest period falls in a seasonal high, which hides part of the gap"
            lines.append(c.t(
                f"- 최고치는 {pdate}의 {pval}이며, 최근 값({last})은 고점보다 {below} 낮습니다. 다만 계절성이 "
                f"뚜렷하고(강도 {ss:.2f}) {why_ko}. 계절 요인을 뺀 추세선 기준으로는 고점 대비 {c.pct(tdd)}로 {t_ko}.",
                f"- The peak was {pval} {_en_when(pdate)}; the latest value ({last}) is {below} below it, but the series is "
                f"strongly seasonal (strength {ss:.2f}) and {why_en}: on the deseasonalised trend line the level is "
                f"{c.pct(tdd)} from its high — {t_en}.",
            ))
        else:
            ko, en = _drawdown_words(dd)
            lines.append(c.t(
                f"- 최고치는 {pdate}의 {pval}이며, 최근 값({last})은 고점보다 {below} 낮아 {ko}.",
                f"- The peak was {pval} {_en_when(pdate)}; the latest value ({last}) is {below} below it — {en}.",
            ))
    # 5. structural changepoints
    cps = sorted(c.a.changepoints, key=lambda x: pd.Timestamp(x.date))
    if not cps:
        lines.append(c.t("- 뚜렷한 구조 변화(추세 전환점)는 감지되지 않아, 비교적 일관된 흐름을 이어 왔습니다.",
                         "- No structural breaks were detected — the trend has followed a fairly consistent path."))
    else:
        shown = cps
        extra_ko = extra_en = ""
        if len(cps) > 5:
            shown = sorted(sorted(cps, key=lambda x: -float(x.magnitude))[:5], key=lambda x: pd.Timestamp(x.date))
            extra_ko, extra_en = " (영향이 큰 5건)", " (the 5 largest shown)"
        lines.append(c.t(f"- 구조 변화(추세 전환점)가 {len(cps)}번 감지되었습니다{extra_ko}:",
                         f"- {len(cps)} structural break{'s' if len(cps) != 1 else ''} detected{extra_en}:"))
        # sub-items of the line ending in ':' are nested bullets (two-space indent)
        lines.extend("  " + _changepoint_line(c, cp) for cp in shown)
        since = int(m.n_obs) - 1 - int(cps[-1].index)
        if since > 0:
            lines.append(c.t(
                f"- 가장 최근 구조 변화({c.date(cps[-1].date)}) 이후 지금까지 {c.span(since)}의 데이터가 쌓였습니다.",
                f"- {c.span(since)} of data {'has' if since == 1 else 'have'} accumulated since the latest "
                f"break ({c.date(cps[-1].date)}).",
            ))
    # 6. anomalies (max 3, the strongest)
    ans = list(c.a.anomalies)
    if not ans:
        lines.append(c.t("- 통계적 이상치는 발견되지 않았습니다.", "- No statistical anomalies were detected."))
    else:
        top = sorted(sorted(ans, key=lambda x: -abs(float(x.zscore)))[:3], key=lambda x: pd.Timestamp(x.date))
        more_ko = " (영향이 큰 3건)" if len(ans) > 3 else ""
        more_en = " (the 3 strongest shown)" if len(ans) > 3 else ""
        lines.append(c.t(f"- 통계적 이상치가 {len(ans)}건 있었습니다{more_ko}:",
                         f"- {len(ans)} statistical anomal{'ies' if len(ans) != 1 else 'y'} detected{more_en}:"))
        lines.extend("  " + _anomaly_line(c, an) for an in top)
    # 7. seasonality and trend consistency
    ts_ = float(c.m.trend_strength)
    t_ko, t_en = ((("낮습니다", "low") if ts_ < 0.3 else ("보통입니다", "moderate")) if ts_ < 0.6
                  else (("높습니다", "high") if ts_ < 0.85 else ("매우 높습니다", "very high")))
    if c.a.decomposition.period <= 1:
        lines.append(c.t(
            f"- 계절 주기가 없거나 데이터가 짧아 계절성은 추정하지 않았습니다. 추세의 일관성(추세 강도 {ts_:.2f})은 {t_ko}.",
            f"- No seasonal cycle was estimated (no period or too little data); trend consistency is {t_en} (trend strength {ts_:.2f}).",
        ))
    else:
        ss = float(c.m.seasonal_strength)
        for thr, ko, en in ((0.2, "거의 없습니다", "negligible"), (0.4, "약한 편입니다", "weak"),
                            (0.64, "보통 수준입니다", "moderate"), (0.8, "뚜렷합니다", "clear"),
                            (math.inf, "매우 뚜렷합니다", "very strong")):
            if ss < thr:
                s_ko, s_en = ko, en
                break
        pat = _seasonal_pattern(c)
        pat_ko = f" 보통 {pat[0]}에 높고 {pat[1]}에 낮은 패턴이 반복됩니다." if pat else ""
        pat_en = f", typically highest {pat[2]} {pat[0]} and lowest {pat[2]} {pat[1]}" if pat else ""
        lines.append(c.t(
            f"- 계절성은 {s_ko} (강도 {ss:.2f}).{pat_ko} 추세의 일관성(추세 강도 {ts_:.2f})은 {t_ko}.",
            f"- Seasonality is {s_en} (strength {ss:.2f}){pat_en}; trend consistency is {t_en} (trend strength {ts_:.2f}).",
        ))
    return lines


def _evidence(c: _Ctx) -> list[str]:
    """Metric-based evidence for the phase (used when no usable reasons are available).

    Full sentences, and only facts that support the phase: the distance from the
    peak is evidence for declining / peaking / reviving, not for a stable or
    growing call (and it is measured on the trend line when seasonality distorts
    the raw value).
    """
    m = c.m
    out: list[str] = []
    p = _finite(m.mk_pvalue)
    pstr = "< 0.001" if p is not None and p < 0.001 else f"= {c.num(p, decimals=3)}"
    if m.direction in ("up", "down"):
        out.append(c.t(
            f"Mann-Kendall 검정에서 전체 기간의 {c.direction(m.direction)} 추세가 유의합니다 (p {pstr}).",
            f"The Mann-Kendall test finds a significant {c.direction(m.direction)} trend over the whole period "
            f"(p {pstr})."))
    else:
        out.append(c.t(f"Mann-Kendall 검정에서 전체 기간의 뚜렷한 추세가 확인되지 않습니다 (p {pstr}).",
                       f"The Mann-Kendall test finds no clear trend over the whole period (p {pstr})."))
    if _finite(m.recent_change_pct) is not None:
        win = c.span(c.recent_window())
        out.append(c.t(f"최근 {win} 동안 추세가 {c.pct(m.recent_change_pct)} 변했습니다.",
                       f"Over the last {win} the trend changed by {c.pct(m.recent_change_pct)}."))
    mom = c.num(m.momentum, decimals=2, signed=True)
    out.append(c.t(f"최근 기울기와 장기 기울기의 차이(모멘텀)는 평균 수준 대비 {mom}%p{c.unit}입니다.",
                   f"The recent slope differs from the long-run slope (momentum) by {mom} pp{c.unit} "
                   f"of the mean level."))
    dd = _finite(m.drawdown_pct)
    if dd is not None and c.a.phase.label in ("declining", "peaking", "reviving"):
        tdd = _seasonal_drawdown(c, dd)
        basis = dd if tdd is None else tdd
        if basis <= -5:
            gap = c.pct(abs(basis), signed=False)
            if tdd is None:
                out.append(c.t(f"최근 값이 고점({c.date(m.peak_date)})보다 {gap} 낮습니다.",
                               f"The latest value is {gap} below the peak ({c.date(m.peak_date)})."))
            else:
                out.append(c.t(f"계절 요인을 뺀 추세선이 고점보다 {gap} 낮습니다.",
                               f"The deseasonalised trend line is {gap} below its peak."))
    return out[:4]


def _momentum_phrase(c: _Ctx) -> tuple[str, str]:
    m = c.m
    mom = _finite(m.momentum) or 0.0
    long_run = _finite(m.slope_pct) or 0.0
    recent = long_run + mom
    eps = 1.0 / _ppy(c.freq)  # 1 %-point per year, expressed per period
    if mom > eps:
        if recent > 0 and long_run > 0:
            return "장기 평균보다 빠른 속도로 성장하고 있습니다 (가속)", "growing faster than its long-run pace (accelerating)"
        if recent > 0:
            return ("장기적인 하락·횡보 흐름과 달리 최근에는 상승하고 있습니다",
                    "rising recently, unlike its long-run flat or falling path")
        return "하락 속도가 둔화되고 있습니다", "the decline is slowing"
    if mom < -eps:
        if recent < 0 and long_run < 0:
            return "장기 평균보다 빠른 속도로 하락하고 있습니다", "falling faster than its long-run pace"
        if recent < 0:
            return "장기 흐름과 달리 최근에는 하락하고 있습니다", "falling recently, unlike its long-run path"
        return "성장 속도가 장기 평균보다 느려지고 있습니다 (둔화)", "growth is slower than its long-run pace (decelerating)"
    return "장기 평균과 비슷한 속도를 유지하고 있습니다", "moving at about its long-run pace"


def _present_section(c: _Ctx) -> list[str]:
    m, ph = c.m, c.a.phase
    conf = _pct0(ph.confidence, c.lang)
    lines = [c.t(f"- 현재 국면은 **{c.phase_name(ph.label)}**입니다 (판단 신뢰도 {conf}).",
                 f"- Current lifecycle phase: **{c.phase_name(ph.label)}** (confidence {conf}).")]
    reasons = [str(r).strip() for r in (ph.reasons or []) if str(r).strip()]
    if c.lang == "en":
        # the same reasons as the analysis (and the report's diagnosis card), translated
        reasons = [t for t in (translate_reason(r) for r in reasons) if t]
    if not reasons:
        reasons = _evidence(c)
    lines.extend(c.t(f"- 근거: {r}", f"- Evidence: {r}") for r in reasons[:4])
    win = c.span(c.recent_window())
    if _finite(m.recent_change_pct) is not None:
        lines.append(c.t(f"- 최근 {win} 동안 추세는 **{c.pct(m.recent_change_pct)}** 변했습니다.",
                         f"- Over the last {win} the trend changed by **{c.pct(m.recent_change_pct)}**."))
    else:
        tr = np.asarray(c.a.decomposition.trend, dtype=float)
        k_r = c.recent_window()
        base = float(tr[-1 - k_r]) if len(tr) > k_r and math.isfinite(tr[-1 - k_r]) else None
        reason = _undefined_pct_reason(base, c.lang)
        lines.append(c.t(f"- 최근 {win} 변화율은 {reason} 계산하지 않았습니다.",
                         f"- The % change over the last {win} is undefined ({reason})."))
    ko, en = _momentum_phrase(c)
    mom = c.num(m.momentum, decimals=2, signed=True)
    lines.append(c.t(f"- 모멘텀(최근 기울기 − 장기 기울기)은 {mom}%p{c.unit}이며, {ko}.",
                     f"- Momentum (recent minus long-run slope) is {mom} pp{c.unit}: {en}."))
    vol = _finite(m.volatility) or 0.0
    if vol < 0.05:
        v_ko, v_en = "낮은 편입니다", "low"
    elif vol < 0.15:
        v_ko, v_en = "보통 수준입니다", "moderate"
    elif vol < 0.3:
        v_ko, v_en = "높은 편입니다", "high"
    else:
        v_ko, v_en = "매우 높습니다", "very high"
    lines.append(c.t(f"- 변동성(잔차의 변동계수)은 {c.num(vol * 100, decimals=1)}%로 {v_ko}.",
                     f"- Volatility (residual coefficient of variation) is {c.num(vol * 100, decimals=1)}% — {v_en}."))
    cur = _current_level(c.a)
    k = min(max(1, int(c.a.season_length)), int(m.n_obs))
    if k > 1:
        lines.append(c.t(
            f"- 가장 최근 값은 {c.num(m.last_value)}({c.date(m.end)})이며, 최근 {c.span(k)} 평균은 {c.num(cur)}입니다.",
            f"- Latest value: {c.num(m.last_value)} ({c.date(m.end)}); average over the last {c.span(k)}: {c.num(cur)}.",
        ))
    else:
        lines.append(c.t(f"- 가장 최근 값은 {c.num(m.last_value)}({c.date(m.end)})입니다.",
                         f"- Latest value: {c.num(m.last_value)} ({c.date(m.end)})."))
    return lines


def _accuracy_line(c: _Ctx) -> str:
    """How accurate the forecast was in the backtest, judged by ``ForecastResult.skill``.

    ``skill`` = backtest MAE of the final method / MAE of the best naive benchmark
    (naive, seasonal naive, drift) on the *same* windows, so ``1 - skill`` is the
    error reduction versus naive.  The raw MASE is NOT compared with 1: it is
    scaled by the in-sample *one-step* seasonal-naive error, so multi-step
    forecasts routinely exceed 1 while beating naive (and fall below 1 without
    beating it); it is only given as a technical note.  Without ``skill`` no
    naive comparison is claimed.
    """
    skill, mase = _finite(c.f.skill), _finite(c.f.backtest_mase)
    if skill is not None:
        skill = max(0.0, skill)
        if skill <= 0.95:
            gain = (1.0 - skill) * 100.0
            g = "99% 이상" if gain > 99.5 else f"약 {gain:.0f}%"
            g_en = "more than 99%" if gain > 99.5 else f"about {gain:.0f}%"
            ko = f"과거 구간 백테스트에서 {_NAIVE_KO}보다 오차가 **{g}** 작았습니다."
            en = f"in backtests on past data the error was **{g_en}** lower than that of {_NAIVE_EN}."
        elif skill <= 1.0:  # "about 0-5 % lower error" would oversell a tie
            ko = (f"과거 구간 백테스트에서 오차가 {_NAIVE_KO}과 비슷해(차이 5% 이내), 단순 기준보다 "
                  f"뚜렷하게 낫다고 보기는 어렵습니다.")
            en = (f"in backtests on past data the error was about the same as that of {_NAIVE_EN} "
                  f"(within 5%), so the forecast is not clearly better than naive.")
        else:
            worse = (skill - 1.0) * 100.0
            if skill >= 2.0:
                w_ko, w_en = f" (오차가 약 {skill:.1f}배)", f" (about {skill:.1f}× its error)"
            elif worse >= 1.0:
                w_ko, w_en = f" (오차가 약 {worse:.0f}% 더 큼)", f" (error about {worse:.0f}% higher)"
            else:
                w_ko = w_en = ""
            ko = f"과거 구간 백테스트에서 {_NAIVE_KO}보다 나은 성과를 보이지 못했습니다{w_ko}."
            en = f"in backtests on past data it did not beat {_NAIVE_EN}{w_en}."
    elif mase is not None:
        ko = "같은 백테스트 구간에서 단순 기준 예측과 비교한 결과가 없어, 단순 기준보다 나은지는 판단하지 않았습니다."
        en = ("no comparison with simple benchmarks on the same backtest windows is available, so whether it "
              "beats a naive forecast is not assessed.")
    else:
        ko = "데이터가 짧아 백테스트로 정확도를 검증하지 못했습니다."
        en = "the series is too short for a backtest, so the accuracy could not be checked."
    if mase is not None:
        what_ko = f"기술 참고: 백테스트 MASE {mase:.2f} — 학습 구간에서 한 기 앞을 계절 단순 예측한 오차를 1로 둔 상대 오차"
        what_en = (f"Technical note: backtest MASE {mase:.2f}, which scales the errors by the in-sample one-step "
                   f"seasonal-naive error")
        if mase > 1:
            ko += f" ({what_ko}라, 여러 기 앞을 내다보는 예측은 단순 기준보다 정확해도 1을 넘기 쉽습니다.)"
            en += f" ({what_en}; multi-step forecasts often score above 1 even when they beat naive.)"
        else:
            ko += f" ({what_ko}로, 그 자체로 같은 구간의 단순 기준 예측과의 우열을 뜻하지는 않습니다.)"
            en += f" ({what_en}; on its own it does not say whether the forecast beats naive on the same windows.)"
    return c.t(f"- 정확도: {ko}", f"- Accuracy: {en}")


def _future_section(c: _Ctx) -> list[str]:
    f, o = c.f, c.o
    lines: list[str] = []
    h = c.span(f.horizon)
    fc_end = c.date(f.mean.index[-1]) if len(f.mean) else c.num(None)
    chg = _finite(o.expected_change_pct)
    dir_word = c.direction(o.direction)
    win = _outlook_windows(c.a, f)
    if chg is not None:
        # exactly the levels of forecast.compute_outlook: the last min(season, horizon)
        # forecast points versus the last min(season, n) observations, each with its
        # average seasonal effect removed when it is not a whole number of seasons
        fw, cw = win.k_fut, win.k_cur
        if fw <= 1:
            fut_ko, fut_en = f"{fc_end} 예측값", f"the forecast for {fc_end}"
        elif fw >= win.horizon:
            fut_ko, fut_en = f"향후 {h} 예측 평균", f"the average forecast over the next {h}"
        else:
            fut_ko, fut_en = (f"예측 마지막 {c.span(fw)} 평균",
                              f"the average of the last {c.span(fw)} of the forecast")
        cur_ko, cur_en = (("최근 값", "the latest value") if cw <= 1
                          else (f"최근 {c.span(cw)} 평균", f"the average of the last {c.span(cw)}"))
        adj_ko, adj_en = "계절 효과를 보정한 ", "seasonally adjusted "
        if win.adj_fut:
            fut_ko, fut_en = adj_ko + fut_ko, fut_en.replace("the ", "the " + adj_en, 1)
        if win.adj_cur:
            cur_ko, cur_en = adj_ko + cur_ko, cur_en.replace("the ", "the " + adj_en, 1)
        flat_ko = flat_en = ""
        if _flat_despite_change(c, chg):
            flat_ko, flat_en = "예측 불확실성 대비 크지 않아 ", "within the forecast uncertainty, so "
        note_ko = note_en = ""
        if (win.adj_fut or win.adj_cur) and win.period is not None:
            part = [c.span(k) for k, adj in ((fw, win.adj_fut), (cw, win.adj_cur)) if adj]
            part_ko = "·".join(dict.fromkeys(part))
            note_ko = (f" {part_ko} 구간은 계절 주기({c.span(win.period)})를 다 채우지 못해, 그 구간의 평균적인 "
                       f"계절 높낮이를 빼고 비교했습니다.")
            part_en = [_span_adj(k, c.freq) for k, adj in ((fw, win.adj_fut), (cw, win.adj_cur)) if adj]
            note_en = (f" A {' / '.join(dict.fromkeys(part_en))} window covers only part of the "
                       f"{_span_adj(win.period, c.freq)} seasonal cycle, so its typical seasonal effect is removed "
                       f"before comparing.")
        lines.append(c.t(
            f"- {fut_ko}은 {cur_ko}보다 **{c.pct(chg)}** 변할 것으로 예상됩니다 "
            f"({flat_ko}**{dir_word}** 전망, 예측 종료 {fc_end}).{note_ko}",
            f"- {fut_en[0].upper() + fut_en[1:]} is expected to be **{c.pct(chg)}** versus {cur_en} "
            f"({flat_en}**{dir_word}**; forecast to {fc_end}).{note_en}",
        ))
    else:
        lines.append(c.t(
            f"- 향후 {h}({fc_end}까지) 전망은 **{dir_word}**입니다 (현재 수준이 0에 가까워 변화율은 계산하지 않음).",
            f"- The outlook for the next {h} (to {fc_end}) is **{dir_word}** (no % change: current level near zero).",
        ))
    # scenarios (end-of-horizon values; the % in brackets is versus the current level,
    # which is named explicitly because it is an average, not the latest value)
    cur = _current_level(c.a)
    lv = _interval_level(f)
    cw = min(max(1, int(c.a.season_length)), int(c.m.n_obs))
    scale = _scale(c.a)
    with_pct = cur is not None and scale > 0 and abs(cur) > 1e-6 * scale

    def scen(key: str) -> str:
        v = _finite(o.scenarios.get(key)) if o.scenarios else None
        if v is None:
            return c.num(None)
        if with_pct:
            return f"{c.num(v)} ({c.pct((v - cur) / abs(cur) * 100)})"
        return c.num(v)

    band95 = ""
    if 95 in f.lower and 95 in f.upper and len(f.mean):
        lo95, hi95 = c.num(f.lower[95].iloc[-1]), c.num(f.upper[95].iloc[-1])
        band95 = c.t(f" 95% 구간은 {lo95} ~ {hi95}입니다.", f"; 95% interval {lo95} – {hi95}")
    ref_ko, ref_en = (("최근 값", "the latest value") if cw <= 1
                      else (f"최근 {c.span(cw)} 평균", f"the average of the last {c.span(cw)}"))
    paren_ko, paren_en = [], []
    if lv is not None:
        paren_ko.append(f"{lv}% 예측구간")
        paren_en.append(f"{lv}% interval")
    if with_pct:
        paren_ko.append(f"괄호는 {ref_ko} {c.num(cur)} 대비")
        paren_en.append(f"% vs. {ref_en}, {c.num(cur)}")
    sp_ko = f"({', '.join(paren_ko)})" if paren_ko else ""
    sp_en = f" ({'; '.join(paren_en)})" if paren_en else ""
    seasonal_note = ""
    if (with_pct and cw > 1 and c.a.decomposition.period > 1
            and (_finite(c.m.seasonal_strength) or 0.0) >= 0.3):
        seasonal_note = c.t(
            " 시나리오 값은 종료 시점의 계절적 높낮이를 포함하므로 위의 평균 기준 변화율과 다를 수 있습니다.",
            " These end-point values include the seasonal swing at that date, so they can differ from the "
            "average-based change above.",
        )
    elif with_pct and chg is not None and win.k_fut > 1:
        seasonal_note = c.t(
            " 시나리오는 종료 시점 한 값이므로, 기간 평균끼리 비교한 위의 변화율과 다를 수 있습니다.",
            " These are single end-point values, so they can differ from the average-based change above.",
        )
    lines.append(c.t(
        f"- {fc_end} 시점 시나리오{sp_ko}: 비관 {scen('pessimistic')} · 기준 {scen('base')} · "
        f"낙관 {scen('optimistic')}.{band95}{seasonal_note}",
        f"- Scenarios for {fc_end}{sp_en}: pessimistic {scen('pessimistic')} · base {scen('base')} · "
        f"optimistic {scen('optimistic')}{band95}.{seasonal_note}",
    ))
    # probability of increase (symmetric wording around 50 %)
    p_raw = _finite(o.prob_increase)
    if p_raw is None:
        lines.append(c.t("- 상승 확률은 계산되지 않았습니다.", "- The probability of increase is not available."))
    else:
        p = min(1.0, max(0.0, p_raw))
        if p >= 0.9:
            p_ko, p_en = "상승 가능성이 매우 높습니다", "a rise is very likely"
        elif p >= 0.7:
            p_ko, p_en = "상승 쪽이 우세합니다", "a rise is more likely than not"
        elif p >= 0.55:
            p_ko, p_en = "상승 쪽이 약간 우세합니다", "a rise is slightly more likely"
        elif p > 0.45:
            p_ko, p_en = "상승과 하락 가능성이 엇비슷합니다", "a rise and a fall are about equally likely"
        elif p > 0.3:
            p_ko, p_en = "하락 쪽이 약간 우세합니다", "a fall is slightly more likely"
        elif p > 0.1:
            p_ko, p_en = "하락 쪽이 우세합니다", "a fall is more likely than not"
        else:
            p_ko, p_en = "하락 가능성이 매우 높습니다", "a fall is very likely"
        ptxt = format_prob(p, c.lang)
        approx_ko, approx_en = ("", "") if ptxt[0] in "<>" else ("약 ", "about ")
        lines.append(c.t(f"- 현재 수준보다 높아질 확률은 {approx_ko}**{ptxt}**로, {p_ko}.",
                         f"- Probability of ending above the current level: {approx_en}**{ptxt}** — {p_en}."))
    # forecast peak
    if o.forecast_peak_date is not None:
        pk = pd.Timestamp(o.forecast_peak_date)
        pv = f.mean.loc[pk] if pk in f.mean.index else (f.mean.max() if len(f.mean) else None)
        lines.append(c.t(f"- 예측 기간 중 {c.date(pk)}경 정점(약 {c.num(pv)})을 찍은 뒤 하락할 것으로 보입니다.",
                         f"- The forecast peaks around {c.date(pk)} (about {c.num(pv)}) and then declines."))
    else:
        lines.append(c.t("- 예측 기간 안에 뚜렷한 정점(고점 후 하락)은 예상되지 않습니다.",
                         "- No distinct forecast peak (rise, then fall) within the horizon."))
    # projected phase
    cur_ph, fut_ph = c.a.phase.label, o.future_phase
    if fut_ph == cur_ph:
        lines.append(c.t(f"- {h} 뒤에도 **{c.phase_name(fut_ph)}** 국면이 이어질 것으로 전망됩니다.",
                         f"- Projected phase at the horizon: still **{c.phase_name(fut_ph)}**."))
    else:
        lines.append(c.t(
            f"- {h} 안에 현재 **{c.phase_name(cur_ph)}**에서 **{c.phase_name(fut_ph)}** 국면으로 넘어갈 가능성이 있습니다.",
            f"- Projected phase shift: **{c.phase_name(cur_ph)}** → **{c.phase_name(fut_ph)}**.",
        ))
    # confidence and accuracy (skill against the naive benchmarks; MASE only as a technical note)
    conf_word = _CONF_PRED[c.lang].get(o.confidence, o.confidence)
    lines.append(c.t(f"- 예측 신뢰도는 **{conf_word}**입니다.", f"- Forecast confidence: **{conf_word}**."))
    lines.append(_accuracy_line(c))
    method = str(f.method or "").strip() or c.num(None)
    ens = method.lower().startswith("ensemble")
    method_note = c.t(" — 백테스트 정확도에 따라 여러 모델을 가중 결합한 앙상블" if ens else "",
                      " — a weighted ensemble of the best backtested models" if ens else "")
    top = [s for s in f.model_scores if _finite(s.mase) is not None][:3]
    top_txt = ", ".join(f"{s.name} (MASE {float(s.mase):.2f})" for s in top)
    line = c.t(f"- 예측 방법: {method}{method_note}.", f"- Method: {method}{method_note}.")
    if top_txt:
        line += c.t(f" 백테스트 상위 모델: {top_txt}.", f" Top backtested models: {top_txt}.")
    lines.append(line)
    return lines


def _caveats_section(c: _Ctx) -> list[str]:
    f, o, m = c.f, c.o, c.m
    lines: list[str] = []
    # widening uncertainty
    lv = _interval_level(f)
    widened = False
    if lv is not None and len(f.mean) >= 2:
        w = (np.asarray(f.upper[lv], dtype=float) - np.asarray(f.lower[lv], dtype=float)) / 2.0
        # widths that are practically 0 (e.g. a constant series: ±0.0000064 -> ±0.000022, "3.5×")
        # would dramatise numerical noise; require at least 0.01 % of the typical level
        vals = np.concatenate([np.asarray(c.a.series, dtype=float), np.asarray(f.mean, dtype=float)])
        vals = vals[np.isfinite(vals)]
        scale = float(np.mean(np.abs(vals))) if len(vals) else 0.0
        tiny = 1e-4 * (scale if scale > 0 else 1.0)
        if (len(w) >= 2 and np.isfinite(w[[0, -1]]).all() and w[0] > tiny and w[-1] / w[0] > 1.05
                and c.num(w[0]) != "0"):
            ratio = f"{w[-1] / w[0]:.1f}"
            lines.append(c.t(
                f"- 예측은 멀리 갈수록 불확실해집니다: {lv}% 예측구간의 폭이 첫 시점 ±{c.num(w[0])}에서 "
                f"마지막 시점 ±{c.num(w[-1])}까지 약 {ratio}배 넓어집니다. 단일 값보다 구간을 기준으로 판단하세요.",
                f"- Uncertainty grows with the horizon: the {lv}% interval widens from ±{c.num(w[0])} at the first "
                f"step to ±{c.num(w[-1])} at the last (about {ratio}×). Judge by the range, not a single number.",
            ))
            widened = True
    if not widened:
        lines.append(c.t("- 먼 미래일수록 예측 불확실성이 커지므로, 단일 값보다 예측구간을 기준으로 판단하세요.",
                         "- Forecast uncertainty grows with the horizon; judge by the interval, not a single number."))
    # structural-break risk
    k = len(c.a.changepoints)
    if k:
        lines.append(c.t(
            f"- 과거에 {k}번의 구조 변화가 있었던 것처럼, 시장·정책·기술 변화 같은 외부 충격이 생기면 추세가 "
            f"갑자기 바뀔 수 있으며 모델은 이를 미리 알 수 없습니다.",
            f"- Just as the past contains {k} structural break{'s' if k != 1 else ''}, an external shock (market, "
            f"policy, technology) can change the trend abruptly — the model cannot foresee that.",
        ))
        last_cp = max(c.a.changepoints, key=lambda x: int(x.index))
        since = int(m.n_obs) - 1 - int(last_cp.index)
        if since < max(int(c.a.season_length), int(m.n_obs) // 6):
            lines.append(c.t(
                f"- 가장 최근 구조 변화({c.date(last_cp.date)}) 이후 관측치가 {max(since, 0)}개뿐이라 새 흐름이 "
                f"지속될지는 아직 검증이 부족합니다.",
                f"- Only {max(since, 0)} observation{'' if since == 1 else 's'} follow"
                f"{'s' if since == 1 else ''} the latest break ({c.date(last_cp.date)}), so whether "
                f"the new regime lasts is not yet well tested.",
            ))
    else:
        lines.append(c.t(
            "- 과거 데이터에는 뚜렷한 구조 변화가 없었지만, 예기치 못한 외부 충격(시장·정책·기술 변화 등)은 "
            "예측에 반영되어 있지 않습니다.",
            "- The history shows no clear structural break, but unexpected external shocks (market, policy, "
            "technology) are not part of the forecast.",
        ))
    # data quality
    n, sl = int(m.n_obs), int(c.a.season_length)
    if sl > 1 and n < 2 * sl:
        lines.append(c.t(
            f"- 관측치가 {n}개로 계절 주기({c.span(sl)})의 2배에 못 미쳐 계절성·추세 추정이 불안정할 수 있습니다.",
            f"- Only {n} observations — less than two seasonal cycles ({c.span(sl)}) — so seasonality and "
            f"trend estimates may be unstable.",
        ))
    elif n < 24:
        lines.append(c.t(f"- 관측치가 {n}개로 적어 결과의 신뢰도가 제한적입니다.",
                         f"- With only {n} observations the results are of limited reliability."))
    ph_conf = _finite(c.a.phase.confidence)
    if ph_conf is not None and ph_conf < 0.5:
        lines.append(c.t(
            f"- 국면 판단의 신뢰도가 {format_prob(ph_conf)}로 낮아, 여러 국면의 경계에 있을 수 있습니다.",
            f"- The phase call has low confidence ({format_prob(ph_conf, 'en')}); the series may sit between phases.",
        ))
    # "did not beat naive" only from the skill (MASE > 1 does not mean that, see _accuracy_line)
    skill = _finite(f.skill)
    beaten_by_naive = skill is not None and skill > 1.0
    if o.confidence == "low":
        if beaten_by_naive:
            lines.append(c.t(
                "- 예측 신뢰도가 낮습니다: 백테스트에서 단순 기준 예측보다 나은 성과를 보이지 못했으므로 "
                "방향성 정도만 참고하세요.",
                "- Forecast confidence is low (in backtests it did not beat a simple naive benchmark): treat it "
                "as a rough direction only.",
            ))
        else:
            lines.append(c.t(
                "- 예측 신뢰도가 낮습니다: 데이터가 짧거나 백테스트 검증이 부족하거나 예측구간이 넓으므로 "
                "방향성 정도만 참고하세요.",
                "- Forecast confidence is low (little data, a weak backtest or a wide interval): treat it as a "
                "rough direction only.",
            ))
    elif beaten_by_naive:
        lines.append(c.t("- 백테스트에서 단순 기준 예측보다 나은 성과를 보이지 못했습니다.",
                         "- In backtests the model did not beat a simple naive benchmark."))
    if (_finite(m.volatility) or 0.0) >= 0.3:
        lines.append(c.t("- 변동성이 커서 단기 수치는 크게 출렁일 수 있습니다.",
                         "- Volatility is high, so short-term values can swing widely."))
    if c.a.anomalies:
        na = len(c.a.anomalies)
        lines.append(c.t(
            f"- 이상치 {na}건은 일시적 이벤트(캠페인, 사건, 집계 오류 등)일 수 있으니 원인을 확인해 보세요.",
            f"- The {na} anomal{'ies' if na != 1 else 'y'} may reflect one-off events (campaigns, news, data "
            f"errors) — worth checking.",
        ))
    if f.nonnegative and 95 in f.lower and len(f.lower[95]) and float(np.nanmin(np.asarray(f.lower[95], dtype=float))) <= 0:
        lines.append(c.t("- 값이 음수가 될 수 없어 예측 하한을 0으로 제한했습니다.",
                         "- Values cannot be negative, so the lower forecast bound is clipped at 0."))
    lines.append(c.t(
        "- 이 분석은 과거 데이터의 통계적 패턴에 기반한 추정일 뿐 미래를 보장하지 않으며, 투자·재무 조언이 아닙니다.",
        "- This is a statistical extrapolation of past patterns, not a guarantee — and not financial or investment advice.",
    ))
    return lines


def rule_based_narrative(analysis: AnalysisResult, forecast: ForecastResult, outlook: Outlook,
                         lang: str = "ko") -> str:
    """Deterministic past -> present -> future story with concrete numbers.

    Sections (Korean): ``## 과거: 어떻게 여기까지 왔나``, ``## 현재: 지금 어디에 있나``,
    ``## 미래 전망``, ``## 주의할 점`` (English: Past / Present / Outlook / Caveats).
    """
    lang = _norm_lang(lang)
    c = _Ctx(analysis, forecast, outlook, lang)
    h_past, h_now, h_future, h_caveats = SECTION_HEADINGS[lang]
    parts: list[str] = [_summary_line(c), "", f"## {h_past}", *_past_section(c), "",
                        f"## {h_now}", *_present_section(c), "",
                        f"## {h_future}", *_future_section(c), "",
                        f"## {h_caveats}", *_caveats_section(c)]
    return "\n".join(parts).strip() + "\n"


# --------------------------------------------------------------------------- LLM payload


def _r(x: Any, sig: int = 4) -> Optional[float]:
    """Round to ``sig`` significant digits (integers above 10**sig); ``None`` if not finite."""
    v = _finite(x)
    if v is None:
        return None
    if v == 0:
        return 0.0
    a = abs(v)
    if a >= 10 ** sig:
        return float(round(v))
    d = sig - 1 - int(math.floor(math.log10(a)))
    return float(round(v, d))


def _pick(n: int, max_n: int) -> list[int]:
    """Up to ``max_n`` evenly spaced indices of ``range(n)`` (always incl. first and last)."""
    if n <= max_n:
        return list(range(n))
    return sorted({int(i) for i in np.round(np.linspace(0, n - 1, max_n))})


def _downsample_history(series: pd.Series, freq: str, max_points: int = 60,
                        recent: int = 24) -> tuple[list[list[Any]], str]:
    vals = np.asarray(series, dtype=float)
    idx = series.index
    n = len(vals)
    if n <= max_points:
        rows = [[format_date(idx[i], freq), _r(vals[i])] for i in range(n) if np.isfinite(vals[i])]
        return rows, "all observations (no downsampling)"
    old_n = n - recent
    block = int(math.ceil(old_n / (max_points - recent)))
    rows: list[list[Any]] = []
    for start in range(0, old_n, block):
        chunk = vals[start:min(start + block, old_n)]
        chunk = chunk[np.isfinite(chunk)]
        if len(chunk):
            rows.append([format_date(idx[start], freq), _r(chunk.mean())])
    rows.extend([format_date(idx[i], freq), _r(vals[i])] for i in range(old_n, n) if np.isfinite(vals[i]))
    note = (f"downsampled from {n} observations: older points are means of {block} consecutive periods "
            f"(dated by the first period of each block); the last {recent} points are raw")
    return rows, note


def _expected_change_basis(analysis: AnalysisResult, forecast: ForecastResult) -> str:
    """Plain-English definition of ``expected_change_pct`` as ``forecast.compute_outlook`` computes it."""
    w = _outlook_windows(analysis, forecast)
    txt = (f"mean of the last {w.k_fut} forecast periods vs. current_level (mean of the last {w.k_cur} "
           f"observed periods)")
    adj = [name for name, flag in (("forecast", w.adj_fut), ("current", w.adj_cur)) if flag]
    if adj and w.period is not None:
        txt += (f"; the {' and '.join(adj)} window{'s cover' if len(adj) > 1 else ' covers'} only part of the "
                f"{w.period}-period seasonal cycle, so the average seasonal effect over "
                f"{'them' if len(adj) > 1 else 'it'} is removed first (for the forecast window scaled by how much "
                f"seasonality the forecast contains)")
    if w.k_fut > 1:
        txt += "; this compares averages, not the horizon-end value (see scenarios_at_horizon_end)"
    return txt


def build_llm_payload(analysis: AnalysisResult, forecast: ForecastResult, outlook: Outlook) -> dict:
    """Compact, JSON-serialisable dict of facts for the LLM.

    Timestamps become frequency-formatted strings, numpy scalars become
    rounded Python floats, the history is downsampled to at most ~60 points
    (recent detail kept) and the forecast table to at most 36 rows.
    """
    freq = analysis.freq
    m = analysis.metrics

    def fd(ts: Any) -> Optional[str]:
        return None if ts is None else format_date(ts, freq, "en")

    history, note = _downsample_history(analysis.series, freq)
    trend = analysis.decomposition.trend
    tv = np.asarray(trend, dtype=float)
    finite_t = np.flatnonzero(np.isfinite(tv))
    trend_info: dict[str, Any] = {}
    if len(finite_t):
        i0, i1 = int(finite_t[0]), int(finite_t[-1])
        trend_info = {"start_date": fd(trend.index[i0]), "start_value": _r(tv[i0]),
                      "end_date": fd(trend.index[i1]), "end_value": _r(tv[i1])}

    lv80 = _interval_level(forecast)
    levels = sorted(lv for lv in forecast.levels if lv in forecast.lower and lv in forecast.upper)
    table = []
    for i in _pick(len(forecast.mean), 36):
        row: dict[str, Any] = {"date": fd(forecast.mean.index[i]), "mean": _r(forecast.mean.iloc[i])}
        for lv in levels:
            row[f"lo{lv}"] = _r(forecast.lower[lv].iloc[i])
            row[f"hi{lv}"] = _r(forecast.upper[lv].iloc[i])
        table.append(row)

    top_cps = sorted(analysis.changepoints, key=lambda x: -float(x.magnitude))[:10]
    changepoints = [{"date": fd(cp.date), "kind": str(cp.kind),
                     "slope_before_per_period": _r(cp.slope_before),
                     "slope_after_per_period": _r(cp.slope_after),
                     "magnitude": _r(cp.magnitude)}
                    for cp in sorted(top_cps, key=lambda x: pd.Timestamp(x.date))]
    top_an = sorted(analysis.anomalies, key=lambda x: -abs(float(x.zscore)))[:10]
    anomalies = [{"date": fd(a.date), "value": _r(a.value), "expected": _r(a.expected),
                  "robust_zscore": _r(a.zscore, 3)}
                 for a in sorted(top_an, key=lambda x: pd.Timestamp(x.date))]
    peak_date = outlook.forecast_peak_date
    payload: dict[str, Any] = {
        "series": {
            "name": str(analysis.name),
            "frequency": freq,
            "frequency_label": _FREQ_ADJ["en"].get(freq, freq),
            "season_length": int(analysis.season_length),
            "start": fd(m.start), "end": fd(m.end), "n_obs": int(m.n_obs),
        },
        "conventions": {
            "percentages": "percent units (12.5 means +12.5%)",
            "slopes": "series units per period",
            "momentum": "recent trend slope minus long-run slope, in % of mean level per period",
            "skill": ("forecast.skill = backtest MAE of the final method / MAE of the best naive benchmark "
                      "(naive, seasonal naive, drift) on the same backtest windows; < 1 = better than naive "
                      "(0.6 = about 40% lower error), > 1 = did not beat naive; null = not available"),
            "mase": ("mean absolute scaled error, scaled by the in-sample ONE-STEP seasonal-naive MAE; multi-step "
                     "forecasts often exceed 1 even when they beat naive, so do not compare it with 1 -- use skill"),
            "probabilities": "0..1; never present one as a certainty (write >99% / <1%, not 100% / 0%)",
        },
        "history": history,
        "history_note": note,
        "trend": trend_info,
        "current_level": _r(_current_level(analysis)),
        "metrics": {
            "last_value": _r(m.last_value), "mean": _r(m.mean),
            "total_change_pct": _r(m.total_change_pct), "cagr_pct": _r(m.cagr_pct),
            "recent_change_pct": _r(m.recent_change_pct),
            "recent_window_periods": _recent_window(freq, int(analysis.season_length), int(m.n_obs)),
            "slope_per_period": _r(m.slope), "slope_pct_per_period": _r(m.slope_pct),
            "momentum": _r(m.momentum), "acceleration": _r(m.acceleration),
            "volatility_cv": _r(m.volatility), "trend_strength": _r(m.trend_strength, 3),
            "seasonal_strength": _r(m.seasonal_strength, 3),
            "mann_kendall_tau": _r(m.mk_tau, 3), "mann_kendall_p": _r(m.mk_pvalue, 3),
            "direction": str(m.direction),
            "peak_date": fd(m.peak_date), "peak_value": _r(m.peak_value),
            "drawdown_pct": _r(m.drawdown_pct),
        },
        "phase": {
            "label": str(analysis.phase.label),
            "label_ko": analysis.phase.label_ko,
            "confidence": _r(analysis.phase.confidence, 3),
            "reasons": [str(r) for r in analysis.phase.reasons][:6],
        },
        "changepoints": changepoints,
        "changepoints_total": len(analysis.changepoints),
        "anomalies": anomalies,
        "anomalies_total": len(analysis.anomalies),
        "forecast": {
            "horizon": int(forecast.horizon),
            "horizon_end": fd(forecast.mean.index[-1]) if len(forecast.mean) else None,
            "method": str(forecast.method),
            "backtest_mase": _r(forecast.backtest_mase, 3),
            "skill": _r(getattr(forecast, "skill", None), 3),
            "nonnegative": bool(forecast.nonnegative),
            "levels": [int(lv) for lv in levels],
            "table": table,
            "table_note": ("all horizon steps" if len(table) == len(forecast.mean)
                           else f"{len(table)} of {len(forecast.mean)} steps, evenly spaced"),
        },
        "model_leaderboard": [
            {"model": str(s.name), "mase": _r(s.mase, 3), "smape": _r(s.smape, 3), "mae": _r(s.mae),
             "rmse": _r(s.rmse), "ensemble_weight": _r(s.weight, 3)}
            for s in forecast.model_scores[:5]
        ],
        "outlook": {
            "direction": str(outlook.direction),
            "expected_change_pct": _r(outlook.expected_change_pct),
            "expected_change_basis": _expected_change_basis(analysis, forecast),
            "prob_increase": _r(outlook.prob_increase, 3),
            "scenarios_at_horizon_end": {str(k): _r(v) for k, v in (outlook.scenarios or {}).items()},
            "scenario_interval_level": lv80,
            "forecast_peak_date": fd(peak_date) if peak_date is not None else None,
            "future_phase": str(outlook.future_phase),
            "future_phase_ko": PHASE_KO.get(outlook.future_phase, outlook.future_phase),
            "confidence": str(outlook.confidence),
        },
    }
    return payload


# --------------------------------------------------------------------------- Claude


def _facts_block(payload: dict) -> str:
    return "<facts>\n" + json.dumps(payload, ensure_ascii=False, indent=1) + "\n</facts>"


def _narrative_request(payload: dict, lang: str) -> str:
    heads = "\n".join(f"## {h}" for h in SECTION_HEADINGS[lang])
    name = payload.get("series", {}).get("name", "")
    if lang == "ko":
        return (f"다음은 trendai가 시계열 「{name}」을 분석해 계산한 사실(JSON)입니다.\n\n"
                f"{_facts_block(payload)}\n\n"
                "이 사실만 사용해 한국어로 트렌드 해설을 작성하세요. 한두 문장의 핵심 요약으로 시작한 뒤, "
                f"아래 네 개의 섹션 제목을 정확히 이 순서대로 사용하세요:\n{heads}")
    return (f"Below are the facts (JSON) that trendai computed for the time series \"{name}\".\n\n"
            f"{_facts_block(payload)}\n\n"
            "Using only these facts, write the trend narrative in English. Start with a one- or "
            f"two-sentence bottom line, then use exactly these four section headings, in this order:\n{heads}")


def _question_request(payload: dict, question: str, lang: str) -> str:
    if lang == "ko":
        return (f"{_facts_block(payload)}\n\n"
                f"위 사실을 근거로 다음 질문에 한국어로 답하세요.\n\n질문: {question}")
    return (f"{_facts_block(payload)}\n\n"
            f"Answer the following question in English, based on the facts above.\n\nQuestion: {question}")


def _call_claude(content: str, model: str, client: Any, lang: str = "ko") -> str:
    """One ``beta.messages.create`` call; every failure becomes :class:`NarrativeUnavailable`.

    A response cut off by a length limit (``stop_reason`` ``max_tokens``) is still
    returned, but with a ``RuntimeWarning`` and a short note appended in ``lang``
    so the reader knows the end may be missing.
    """
    try:
        import anthropic
    except ImportError as exc:
        raise NarrativeUnavailable("anthropic 패키지가 설치되어 있지 않습니다: pip install anthropic") from exc
    try:
        client = client or anthropic.Anthropic()
        resp = client.beta.messages.create(
            model=model,
            max_tokens=_MAX_TOKENS,
            betas=[_FALLBACK_BETA],
            fallbacks="default",
            output_config={"effort": "medium"},
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": content}],
        )
    except anthropic.AuthenticationError as exc:
        raise NarrativeUnavailable(
            "Claude API 인증에 실패했습니다. ANTHROPIC_API_KEY 환경 변수(또는 `ant auth login`)를 확인하세요."
        ) from exc
    except anthropic.PermissionDeniedError as exc:
        raise NarrativeUnavailable(
            f"이 API 키로는 모델 '{model}' 또는 요청한 기능을 사용할 권한이 없습니다. 키 권한과 모델 이름을 확인하세요."
        ) from exc
    except anthropic.RateLimitError as exc:
        raise NarrativeUnavailable(
            "Claude API 사용량 한도(rate limit)에 도달했습니다. 잠시 후 다시 시도하세요."
        ) from exc
    except anthropic.APIStatusError as exc:
        raise NarrativeUnavailable(
            f"Claude API 오류가 발생했습니다 (HTTP {getattr(exc, 'status_code', '?')}). "
            "잠시 후 다시 시도하거나 요청 내용을 확인하세요."
        ) from exc
    except anthropic.APIConnectionError as exc:
        raise NarrativeUnavailable(
            "Claude API에 연결할 수 없습니다 (네트워크 오류 또는 시간 초과). 인터넷 연결과 프록시 설정을 확인하세요."
        ) from exc
    except anthropic.AnthropicError as exc:
        raise NarrativeUnavailable(f"Claude API 호출에 실패했습니다: {exc}") from exc
    except TypeError as exc:
        msg = str(exc)
        # the SDK raises TypeError when no credentials can be resolved at all
        if "auth" in msg.lower():
            raise NarrativeUnavailable(
                "Claude API 인증 정보를 찾을 수 없습니다. ANTHROPIC_API_KEY 환경 변수를 설정하거나 `ant auth login`을 실행하세요."
            ) from exc
        # an older SDK that does not know `fallbacks` / `output_config` yet: the optional
        # LLM feature must degrade to the rule-based text, not crash the whole pipeline
        if "unexpected keyword" in msg:
            raise NarrativeUnavailable(
                f"설치된 anthropic SDK가 너무 오래되어 필요한 매개변수를 지원하지 않습니다 ({msg}). "
                "pip install -U anthropic 으로 업데이트하세요."
            ) from exc
        raise NarrativeUnavailable(f"Claude API 호출에 실패했습니다: {msg}") from exc
    if getattr(resp, "stop_reason", None) == "refusal":
        raise NarrativeUnavailable(
            "Claude가 이 요청에 대한 응답을 거절했습니다(stop_reason=refusal). 규칙 기반 해설을 사용하세요."
        )
    text = "".join(str(getattr(block, "text", "") or "")
                   for block in (getattr(resp, "content", None) or [])
                   if getattr(block, "type", None) == "text").strip()
    if not text:
        raise NarrativeUnavailable("Claude 응답에 텍스트가 없습니다.")
    stop = getattr(resp, "stop_reason", None)
    if stop in _TRUNCATED_STOPS:
        warnings.warn(f"Claude 응답이 길이 제한에 걸려 잘렸을 수 있습니다 (stop_reason={stop}).",
                      RuntimeWarning, stacklevel=3)
        text += ("\n\n※ 응답이 최대 길이에 도달해 끝부분이 잘렸을 수 있습니다." if lang == "ko"
                 else "\n\nNote: the response reached the maximum length, so the end may be cut off.")
    return text


def llm_narrative(analysis: AnalysisResult, forecast: ForecastResult, outlook: Outlook,
                  lang: str = "ko", model: str = DEFAULT_MODEL, client: Any = None) -> str:
    """Ask Claude to write the narrative from :func:`build_llm_payload` facts.

    ``client`` defaults to ``anthropic.Anthropic()`` (credentials from the
    environment).  Raises :class:`NarrativeUnavailable` on any failure.
    """
    lang = _norm_lang(lang)
    payload = build_llm_payload(analysis, forecast, outlook)
    return _call_claude(_narrative_request(payload, lang), model=model, client=client, lang=lang)


def generate_narrative(analysis: AnalysisResult, forecast: ForecastResult, outlook: Outlook,
                       use_llm: bool = False, lang: str = "ko", model: str = DEFAULT_MODEL,
                       client: Any = None) -> tuple[str, str]:
    """Return ``(text, source)`` with source ``'llm'`` or ``'rule'``.

    With ``use_llm=True`` Claude is tried first; if it is unavailable a
    ``RuntimeWarning`` explains why and the rule-based narrative is returned.
    """
    lang = _norm_lang(lang)
    if use_llm:
        try:
            return llm_narrative(analysis, forecast, outlook, lang=lang, model=model, client=client), "llm"
        except NarrativeUnavailable as exc:
            warnings.warn(f"LLM 해설을 사용할 수 없어 규칙 기반 해설로 대체합니다: {exc}",
                          RuntimeWarning, stacklevel=2)
    return rule_based_narrative(analysis, forecast, outlook, lang=lang), "rule"


def ask(report: TrendReport, question: str, lang: str = "ko", model: str = DEFAULT_MODEL,
        client: Any = None) -> str:
    """Answer a free-form question about ``report`` with Claude, grounded in its facts.

    Raises :class:`NarrativeUnavailable` if Claude cannot be used and
    ``ValueError`` for an empty question.
    """
    lang = _norm_lang(lang)
    q = str(question or "").strip()
    if not q:
        raise ValueError("question must not be empty")
    payload = build_llm_payload(report.analysis, report.forecast, report.outlook)
    return _call_claude(_question_request(payload, q, lang), model=model, client=client, lang=lang)
