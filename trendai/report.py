"""Presentation: self-contained HTML reports, inline SVG charts and terminal summaries.

Public API (see ``docs/ARCHITECTURE.md``)::

    sparkline(values, width=None) -> str          # ▁▂▃▄▅▆▇█
    svg_chart(report, width=960, height=380) -> str
    render_text(report, width=78, lang='ko') -> str   # 'en': English terminal summary
    render_html(report, path=None) -> str
    render_compare_html(reports, ranking, path=None, links=None) -> str

Design notes
------------
* Everything is a pure function of its inputs (no clock, no randomness), so
  rendering the same report twice yields byte-identical output.
* Every user-controlled string (series name, narrative, model names, method,
  reasons ...) is HTML-escaped before it is placed in HTML or SVG -- including
  ``<title>`` elements.  The terminal summary neutralises ``<``/``>`` and control
  characters as well, so its output is safe to paste anywhere.
* HTML pages contain no scripts at all: colours are CSS custom properties with
  a dark palette under ``prefers-color-scheme: dark``; the chart's hover
  read-out is pure CSS (``:hover`` on SVG groups) with ``<title>`` fallbacks.
* SVG text scales with its ``viewBox``.  To keep chart labels readable on any
  screen, :func:`render_html` embeds three renditions of the chart (wide / mid /
  narrow) and shows the one matching the viewport via media queries.
"""

from __future__ import annotations

import hashlib
import html
import math
import re
import sys
import unicodedata
from pathlib import Path
from collections.abc import Mapping
from typing import Any, Iterable, Optional, Sequence, Union
from urllib.parse import quote, urlsplit

import numpy as np
import pandas as pd

from .analysis import changepoint_label as _changepoint_label
from .analysis import pct_change_undefined as _pct_change_undefined
from .data import FREQ_KO
from .data import periods_per_year as _periods_per_year
from .data import season_length as _season_length
from .narrative import MODEL_KO as _NARRATIVE_MODEL_KO
from .narrative import _SCI_LARGE, _SCI_SMALL, _SUPERSCRIPT
from .narrative import PHASE_EN as _PHASE_EN
from .narrative import describe_method as _describe_method
from .narrative import direction_label as _direction_label
from .narrative import format_number as _format_number
from .narrative import format_prob as _format_prob
from .narrative import is_intermittent as _narrative_is_intermittent
from .narrative import model_label as _model_label
from .narrative import period_unit as _period_unit
from .narrative import phase_is_tentative as _phase_is_tentative
from .narrative import scenario_value as _scenario_value
from .narrative import translate_reason as _translate_reason
from .types import PHASE_KO, TrendReport

__all__ = ["sparkline", "svg_chart", "render_text", "render_html", "render_compare_html"]

PathLike = Union[str, Path]

# --------------------------------------------------------------------------- vocabulary

CHANGEPOINT_KO: dict[str, str] = {
    "acceleration": "가속",
    "deceleration": "감속",
    "reversal_up": "상승 반전",
    "reversal_down": "하락 반전",
    "level_shift": "수준 변화",
}
# plain words of Outlook.direction; the pages and the terminal print narrative.direction_label instead
# (see _direction_word: a 'flat' call on a visible change reads "방향 불확실", not "보합")
DIRECTION_KO: dict[str, str] = {"up": "상승", "down": "하락", "flat": "보합"}
DIRECTION_ARROW: dict[str, str] = {"up": "▲", "down": "▼", "flat": "▶"}
CONFIDENCE_KO: dict[str, str] = {"high": "높음", "medium": "보통", "low": "낮음"}
PERIOD_KO: dict[str, str] = {"D": "일", "W": "주", "M": "개월", "Q": "분기", "Y": "년"}
# "<x> 단위" labels (data granularity): yearly data is "연 단위" (두음법칙), not "년 단위"
UNIT_KO: dict[str, str] = {"D": "일", "W": "주", "M": "월", "Q": "분기", "Y": "연"}
# per-period unit suffix, as used by narrative.period_unit ("%p/월")
_PER_UNIT: dict[str, str] = {"D": "/일", "W": "/주", "M": "/월", "Q": "/분기", "Y": "/년"}
SCENARIOS_KO: tuple[tuple[str, str], ...] = (
    ("pessimistic", "비관"), ("base", "기준"), ("optimistic", "낙관"),
)
NARRATIVE_SOURCE_KO: dict[str, str] = {"llm": "AI(Claude) 해석", "rule": "규칙 기반 해석"}

#: model id -> Korean name: the narrative's table (``narrative.MODEL_KO``), so the page, the terminal
#: summary and the narrative name a model alike; kept here as an alias for compatibility
MODEL_KO: dict[str, str] = _NARRATIVE_MODEL_KO

# phase -> accent colour of the badge dot (the label text always carries the meaning)
PHASE_COLOR: dict[str, str] = {
    "emerging": "#1baf7a",
    "growing": "#0ca30c",
    "peaking": "#eda100",
    "declining": "#d03b3b",
    "stable": "#2a78d6",
    "reviving": "#6250d6",
    "volatile": "#eb6834",
}
KIND_COLOR: dict[str, str] = {
    "acceleration": "#0ca30c",
    "deceleration": "#eda100",
    "reversal_up": "#2a78d6",
    "reversal_down": "#d03b3b",
    "level_shift": "#6250d6",
}
# colour of a changepoint by its directed label (analysis.changepoint_label, the one classifier
# behind the chart, the table, the terminal, the narrative and the phase reasons): coloured by
# what it means for the series (a decline accelerating is red, a decline easing green)
_CP_COLOR: dict[str, str] = {
    "성장 가속": "#0ca30c",
    "하락 가속": "#d03b3b",
    "성장 둔화": "#eda100",
    "하락세 둔화": "#1baf7a",
    "하락→상승 반전": "#2a78d6",
    "횡보→상승 전환": "#2a78d6",
    "상승→하락 반전": "#d03b3b",
    "횡보→하락 전환": "#d03b3b",
}

#: the benchmarks behind ForecastResult.skill
_NAIVE_NAMES: tuple[str, ...] = ("naive", "seasonal_naive", "drift")
_SKILL_LABEL = "단순 예측 대비 오차"
_SKILL_TIP = ("같은 백테스트 구간에서 가장 나은 단순 기준 예측(직전값 반복·계절 반복·평균 기울기 연장)의 "
              "평균 절대 오차와 비교한 최종 예측의 오차 증감입니다. 음수일수록 단순 예측보다 정확합니다.")
_MASE_TIP = ("MASE: 예측 오차를 학습 구간에서 한 기 앞을 계절 단순 예측한 오차로 나눈 값(낮을수록 좋음). "
             "여러 기 앞을 내다보는 예측은 단순 예측보다 정확해도 1을 넘기 쉬우므로, 단순 예측과의 비교는 "
             "‘단순 대비’ 열을 보세요.")
#: momentum (TrendMetrics.momentum): recent trend slope minus the long-run slope, in % of the
#: current (recent) trend level per period -- not of the mean level
_MOMENTUM_TIP = "최근 추세 기울기 − 장기 기울기 (현재 추세 수준 대비 %p, 기간당)"
# anomalies are left out of the y range when they stretch it by more than 1 / _SPIKE_RATIO
_SPIKE_RATIO = 0.5

_BLOCKS = "▁▂▃▄▅▆▇█"
_MINUS = "−"
_DASH = "—"

# --------------------------------------------------------------------------- small helpers


# characters that are illegal (C0 controls) or junk (DEL, C1 controls, non-characters) in XML/HTML
_XML_BAD = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f￾￿]")
_SURROGATE = re.compile("[\ud800-\udfff]")


def _clean_text(s: str) -> str:
    """Drop XML-illegal control characters and replace lone surrogates with U+FFFD.

    Lone surrogates appear e.g. in names decoded from ``argv`` with
    ``surrogateescape``; they cannot be encoded as UTF-8 and would make the
    page impossible to write or the SVG impossible to parse.
    """
    return _SURROGATE.sub("�", _XML_BAD.sub("", s))


def _esc(x: Any) -> str:
    """HTML/XML-escape any value (``None`` -> empty string), dropping illegal characters."""
    return html.escape(_clean_text("" if x is None else str(x)), quote=True)


def _finite(x: Any) -> Optional[float]:
    """``float(x)`` if it is a finite number, else ``None``."""
    if x is None or isinstance(x, bool):
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _ts(x: Any) -> Optional[pd.Timestamp]:
    """A tz-naive ``pd.Timestamp`` or ``None``."""
    if x is None:
        return None
    try:
        t = pd.Timestamp(x)
    except (TypeError, ValueError):
        return None
    if pd.isna(t):
        return None
    if t.tzinfo is not None:
        t = t.tz_localize(None)
    try:  # common resolution: ns arithmetic overflows for spans beyond ~292 years
        t = t.as_unit("us")
    except (AttributeError, ValueError, OverflowError):  # pragma: no cover - defensive
        pass
    return t


def _fmt_date(ts: Any, freq: str) -> str:
    """Date formatted per frequency: D/W ``2024-03-05``, M ``2024-03``, Q ``2024-Q1``, Y ``2024``."""
    t = _ts(ts)
    if t is None:
        return _DASH
    if freq == "Y":
        return f"{t.year:04d}"
    if freq == "Q":
        return f"{t.year:04d}-Q{(t.month - 1) // 3 + 1}"
    if freq == "M":
        return f"{t.year:04d}-{t.month:02d}"
    return f"{t.year:04d}-{t.month:02d}-{t.day:02d}"


def _sci(a: float, digits: int = 3) -> str:
    """``×10ⁿ`` notation of a non-negative number, as :func:`narrative.format_number` writes it:
    ``digits`` significant digits, trailing zeros dropped (``1.23×10⁻⁹``, ``4×10¹⁶``)."""
    mant, ex = f"{a:.{max(1, int(digits)) - 1}e}".split("e")
    if "." in mant:
        mant = mant.rstrip("0").rstrip(".")
    e = int(ex)
    if e == 0:
        return mant
    return f"{mant}×10" + ("⁻" if e < 0 else "") + str(abs(e)).translate(_SUPERSCRIPT)


def _fmt_num(x: Any, decimals: Optional[int] = None, signed: bool = False) -> str:
    """Thousands separators, magnitude-dependent decimals, typographic minus; ``—`` for None.

    With automatic decimals this is exactly :func:`narrative.format_number` (``1,234`` /
    ``45.2`` / ``1.20`` / ``0.02``, and ``1.05×10⁻⁹`` / ``6.99×10¹⁶`` below 1e-4 and from
    1e12), so the page, the terminal summary and the narrative print a value alike.
    With fixed ``decimals`` magnitudes from 1e12 still use ``×10ⁿ`` (never 13+-digit
    integers); smaller values keep the given decimals.
    """
    v = _finite(x)
    if v is None:
        return _DASH
    if decimals is None:
        return _format_number(v, signed=signed)
    a = abs(v)
    if a >= _SCI_LARGE:
        body = _sci(a)
    else:
        body = f"{a:,.{max(0, int(decimals))}f}"
    if body.strip("0.,") == "":  # rounds to zero -> no sign
        return body
    if v < 0:
        return _MINUS + body
    return ("+" + body) if signed else body


def _fmt_pct(x: Any, signed: bool = True, decimals: int = 1) -> str:
    """``+12.3%`` style percentage (input already in percent units)."""
    v = _finite(x)
    if v is None:
        return _DASH
    d = 0 if abs(v) >= 1000 else decimals
    return _fmt_num(v, decimals=d, signed=signed) + "%"


def _fmt_prob(p: Any) -> str:
    """Probability (0..1) as a whole percentage, exactly as :func:`narrative.format_prob` writes it.

    ``>99%`` / ``<1%`` at *and* near the bounds (exactly 1.0 or 0.0 included: these
    are model estimates, never certainty), so the page, the terminal summary, the
    CLI ranking and the narrative always print the same text.  ``—`` when missing.
    """
    if _finite(p) is None:
        return _DASH
    return _format_prob(p)


def _fmt_short(x: Any) -> str:
    """Like :func:`_fmt_num` but abbreviates magnitudes ``>= 1e6`` (``1.23M``, ``45.6B``).

    Used for big-number tiles whose width is limited (KPI tiles, scenario boxes);
    the full value is still shown in tables and ``title`` tooltips.  From about
    1e12 -- where :func:`narrative.format_number` switches -- it is the same
    ``×10ⁿ`` notation as the narrative (``6.99×10¹⁶``, never ``69,922T``).
    """
    v = _finite(x)
    if v is None:
        return _DASH
    a = abs(v)
    sign = _MINUS if v < 0 else ""
    if a >= _SCI_LARGE * 0.9995:   # (999.7B would round to "1,000B")
        return sign + _sci(a)
    for unit, suffix in ((1e9, "B"), (1e6, "M")):
        # 0.9995 so that e.g. 999.7M becomes "1.00B" instead of "1,000M"
        if a >= unit * 0.9995:
            q = a / unit
            body = f"{q:,.{2 if q < 9.995 else 1 if q < 99.95 else 0}f}{suffix}"
            return sign + body
    return _fmt_num(v)


def _sign_class(x: Any, eps: float = 0.05) -> str:
    v = _finite(x)
    if v is None or abs(v) < eps:
        return ""
    return "pos" if v > 0 else "neg"


def _hex_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha:g})"


def _phase_ko(label: Any) -> str:
    return PHASE_KO.get(str(label), str(label))


def _phase_name(label: Any, lang: str = "ko") -> str:
    """Phase name in ``lang``: ``성장기`` / ``Growing`` (the narrative's names)."""
    return _PHASE_EN.get(str(label), str(label)) if lang == "en" else _phase_ko(label)


#: phases that describe a move (narrative._phase_undecided): a 'flat' outlook that keeps one of
#: them cannot tell whether the move goes on
_MOVING_PHASES: tuple[str, ...] = ("emerging", "growing", "peaking", "declining", "reviving")


def _tentative(report: TrendReport) -> bool:
    """:func:`narrative.phase_is_tentative`: too few observations for a lifecycle verdict."""
    try:
        return bool(_phase_is_tentative(report.analysis))
    except Exception:  # noqa: BLE001 - hand-made / partial analyses
        return False


def _future_phase_hedge(report: TrendReport) -> Optional[str]:
    """How far ``Outlook.future_phase`` can be stated, by the rule of the narrative's projected-phase
    line (``narrative._phase_projection`` / the payload's ``future_phase_basis``):

    * ``'undecided'``: a 'flat' outlook only *kept* a moving phase -- whether the
      move goes on is hard to judge ("이어질지 바뀔지는 아직 판단하기 어렵습니다");
    * ``'tentative'``: too few observations for a phase call at all
      (:func:`narrative.phase_is_tentative`);
    * ``'low_confidence'``: a phase change implied by a low-confidence forecast
      ("국면이 바뀐다고 보기는 어렵습니다");
    * ``None``: the projected phase as it stands.

    The terminal and the page then never state a phase the narrative declines to project.
    """
    a, o = report.analysis, report.outlook
    cur, fut = str(getattr(a.phase, "label", "")), str(getattr(o, "future_phase", ""))
    if str(getattr(o, "direction", "")) == "flat" and fut == cur and cur in _MOVING_PHASES:
        return "undecided"
    if _tentative(report):
        return "tentative"
    if fut != cur and str(getattr(o, "confidence", "")) == "low":
        return "low_confidence"
    return None


def _future_phase_parts(report: TrendReport, lang: str = "ko") -> tuple[str, str]:
    """``(value, note)`` of the projected phase: ``('성장기', '')``, ``('판단 보류', '예측에 뚜렷한 방향 없음')``,
    ``('쇠퇴기', '신뢰도 낮아 참고용')``, ``('안정기', '관측치가 적어 잠정')`` (see :func:`_future_phase_hedge`)."""
    fut = _phase_name(getattr(report.outlook, "future_phase", ""), lang)
    hedge = _future_phase_hedge(report)
    if lang == "en":
        return {"undecided": ("undecided", "no clear forecast direction"),
                "tentative": (fut, "tentative: few observations"),
                "low_confidence": (fut, "low confidence, indicative only")}.get(hedge, (fut, ""))
    return {"undecided": ("판단 보류", "예측에 뚜렷한 방향 없음"),
            "tentative": (fut, "관측치가 적어 잠정"),
            "low_confidence": (fut, "신뢰도 낮아 참고용")}.get(hedge, (fut, ""))


def _future_phase_tip(report: TrendReport) -> str:
    """The narrative's sentence behind a hedged projected phase (tooltip); ``''`` when not hedged."""
    hedge = _future_phase_hedge(report)
    cur = _phase_ko(getattr(report.analysis.phase, "label", ""))
    fut = _phase_ko(getattr(report.outlook, "future_phase", ""))
    if hedge == "undecided":
        return f"예측에 뚜렷한 방향이 없어, 현재 {cur} 국면이 이어질지 바뀔지는 아직 판단하기 어렵습니다."
    if hedge == "tentative":
        return ("관측치가 적어 국면은 잠정 판단입니다." if fut == cur else
                f"예측대로라면 {fut} 국면에 해당하지만, 현재 국면({cur})이 관측치가 적어 잠정 판단이므로 "
                "국면 변화로 단정하기는 어렵습니다.")
    if hedge == "low_confidence":
        return f"예측대로라면 {cur}에서 {fut} 국면으로 넘어가지만, 예측 신뢰도가 낮아 국면이 바뀐다고 보기는 어렵습니다."
    return ""


def _model_desc(name: str) -> str:
    """Korean name of a model id (``narrative.model_label``), ``''`` for an id it does not know."""
    raw = str(name).strip()
    label = _model_label(raw, "ko")
    return "" if label == raw else label


def _cp_kind(cp: Any) -> tuple[str, str]:
    """Korean label and colour of a changepoint, directed by its slopes.

    The label is :func:`analysis.changepoint_label` -- the one classifier behind
    the phase reasons (``구조 변화(하락→상승 반전)``), the narrative, the chart,
    the table and the terminal: ``acceleration`` with a falling slope is a
    *decline* accelerating (red), ``deceleration`` of a decline is the decline
    easing, and a ``reversal_up`` from a flat stretch is a turn
    (``횡보→상승 전환``), not a reversal.  Slopes that are missing give the
    undirected labels of :data:`CHANGEPOINT_KO`.
    """
    kind = str(getattr(cp, "kind", ""))
    label = _changepoint_label(cp, "ko")
    return label, _CP_COLOR.get(label, KIND_COLOR.get(kind, "#898781"))


def _skill_value(fc: Any) -> Optional[float]:
    """``forecast.skill`` (MAE ratio to the best naive benchmark) as a finite float >= 0, else ``None``."""
    v = _finite(getattr(fc, "skill", None))
    return None if v is None else max(0.0, v)


def _rel_error_text(ratio: Optional[float]) -> str:
    """Error ratio -> signed percent change of the error (``0.65`` -> ``−35%``); ``—`` for None."""
    if ratio is None:
        return _DASH
    return _fmt_pct((ratio - 1.0) * 100.0, decimals=0)


def _rel_error_class(ratio: Optional[float]) -> str:
    """``pos`` when clearly better than naive (lower error), ``neg`` when the shown change of
    the error (:func:`_rel_error_text`) is an increase, else ``''`` (a ``0%`` is never red)."""
    if ratio is None:
        return ""
    if ratio <= 0.95:
        return "pos"
    return "neg" if _rel_error_text(ratio).startswith("+") else ""


def _skill_verdict(skill: Optional[float]) -> str:
    """Short verdict of ``forecast.skill`` for the terminal, with the narrative's thresholds
    (``narrative._accuracy_line``): within 5 % better is a tie, and an error less than 1 %
    higher is "not better", not "less accurate" (its size rounds to ``0%``)."""
    if skill is None:
        return ""
    if skill <= 0.95:
        return "단순 예측보다 정확"
    if skill <= 1.0:
        return "단순 예측과 비슷"
    return "단순 예측보다 부정확" if (skill - 1.0) * 100.0 >= 1.0 else "단순 예측보다 낫지 않음"


def _skill_sentence(skill: Optional[float]) -> str:
    """Plain-Korean verdict of the final method versus the naive benchmarks (same thresholds as
    ``narrative._accuracy_line``: no size for an error less than 1 % higher)."""
    if skill is None:
        return "같은 백테스트 구간에서 단순 기준 예측과 비교한 결과가 없어, 단순 예측보다 나은지는 판단하지 않았습니다."
    if skill <= 0.95:
        gain = (1.0 - skill) * 100.0
        g = "99% 이상" if gain > 99.5 else f"약 {gain:.0f}%"
        return f"최종 방법의 백테스트 오차는 가장 나은 단순 기준 예측보다 {g} 작습니다."
    if skill <= 1.0:
        return "최종 방법의 백테스트 오차가 가장 나은 단순 기준 예측과 비슷합니다(차이 5% 이내)."
    worse = (skill - 1.0) * 100.0
    if skill >= 2.0:
        size = f"(오차 약 {skill:.1f}배)"
    elif worse >= 1.0:
        size = f"(오차 약 {worse:.0f}% 더 큼)"
    else:   # "약 0% 더 큼" would contradict itself: the narrative gives no size either
        size = ""
    return f"최종 방법은 백테스트에서 단순 기준 예측보다 낫지 않았습니다{size}."


def _best_naive(scores: Sequence[Any], metric: str = "mae") -> tuple[Optional[str], Optional[float]]:
    """Name and error (``metric``: ``'mae'`` or ``'rmse'``) of the most accurate naive benchmark
    among the leaderboard rows; ``(None, None)`` when no benchmark row is listed."""
    best: tuple[Optional[str], Optional[float]] = (None, None)
    for s in scores:
        name = str(getattr(s, "name", ""))
        err = _finite(getattr(s, metric, None))
        if name in _NAIVE_NAMES and err is not None and err >= 0 and (best[1] is None or err < best[1]):
            best = (name, err)
    return best


#: models that forecast.default_candidates offers for intermittent demand only (Forecaster.offered_for)
_INTERMITTENT_ONLY: tuple[str, ...] = ("sba", "croston")


def _intermittent_selection(scores: Sequence[Any]) -> bool:
    """Whether the candidates were ranked as intermittent demand (by RMSE, not MASE).

    ``forecast.auto_forecast`` offers ``sba`` only for intermittent demand and then
    ranks, weights and picks the candidates by backtest RMSE, while
    ``model_scores`` stays sorted by MASE (the types contract).
    """
    return any(str(getattr(s, "name", "")).partition("@")[0] in _INTERMITTENT_ONLY for s in scores)


def _intermittent(report: TrendReport) -> bool:
    """Intermittent (mostly-zero) demand: :func:`narrative.is_intermittent` (the rule of the
    analysis and the forecast), or candidates that were ranked as such (an ``sba`` row).

    For such series the forecast measures ``ForecastResult.skill`` in RMSE (the metric
    that ranked the candidates) and ``Anomaly.expected`` is the typical demand size.
    """
    if _intermittent_selection(getattr(report.forecast, "model_scores", None) or []):
        return True
    try:
        return bool(_narrative_is_intermittent(report.analysis))
    except Exception:  # noqa: BLE001 - hand-made / partial analyses
        return False


def _skill_tip(intermittent: bool) -> str:
    """Tooltip of the skill figure: MAE, or RMSE for intermittent demand (``forecast.auto_forecast``)."""
    if not intermittent:
        return _SKILL_TIP
    return _SKILL_TIP.replace("평균 절대 오차", "제곱 오차(RMSE, 간헐 수요라 모델 선택과 같은 기준)")


def _method_kind(fc: Any) -> str:
    """The final method in a few words for the page subtitle: ``단일 모델 예측``,
    ``3개 모델 앙상블 예측`` or, for the fallback of very short series, ``백테스트 없이 만든 기본 예측``
    (members = models with a positive ensemble weight, as :func:`narrative.describe_method`)."""
    method = str(getattr(fc, "method", "") or "").strip()
    if method.lower().startswith("fallback"):
        return "백테스트 없이 만든 기본 예측"
    n = sum(1 for s in (getattr(fc, "model_scores", None) or []) if (_finite(getattr(s, "weight", None)) or 0.0) > 0)
    if n == 0 and method.lower().startswith("ensemble("):
        n = method.count(",") + 1
    if n == 0 and not method:
        return "통계 모델 예측"
    return f"{n}개 모델 앙상블 예측" if n >= 2 else "단일 모델 예측"


def _model_html(name: Any) -> str:
    """Escaped model name with line-break opportunities after ``_`` and before ``@`` only."""
    e = _esc(name)
    return e.replace("_", "_<wbr>").replace("@", "<wbr>@")


def _outlook_levels(report: TrendReport, d: "_ChartData") -> Any:
    """:func:`forecast.outlook_levels` of ``report`` (computed once per render, cached on ``d``).

    The single source of the two levels behind ``Outlook.expected_change_pct`` and
    ``prob_increase`` -- their windows, partial-season adjustments and the reason a
    percentage is undefined (``'zero'``, ``'mostly_zero'``, ``'sign_change'``,
    ``'near_zero'``) -- shared with ``compute_outlook`` itself and the narrative.
    ``None`` for reports it cannot handle (empty or hand-made partial ones).
    """
    if not d.levels_done:
        try:
            from .forecast import outlook_levels

            with np.errstate(all="ignore"):   # sums of |y| ~ 1e306 overflow: no levels, no warning
                d.levels = outlook_levels(report.analysis, report.forecast)
        except Exception:  # noqa: BLE001 - hand-made / partial analyses: windows only, no levels
            d.levels = None
        d.levels_done = True
    return d.levels


def _outlook_windows(report: TrendReport, d: "_ChartData") -> tuple[int, int, int, Optional[int], bool, bool]:
    """Windows of ``forecast.compute_outlook``: ``(k_fut, k_cur, horizon, period, adj_fut, adj_cur)``.

    Future level = mean of the last ``k_fut = min(season, horizon)`` forecast
    points, current level = mean of the last ``k_cur = min(season, n)``
    observations.  A window that is not a whole number of seasonal periods has
    its average seasonal effect removed (``adj_*``; e.g. the default 26 of 52
    weeks) when the decomposition has a usable seasonal component.  Taken from
    :func:`forecast.outlook_levels` (see :func:`_outlook_levels`), so the page,
    ``compute_outlook`` and the narrative cannot drift apart; a report it cannot
    handle gets the plain windows without adjustment.
    """
    mean = getattr(report.forecast, "mean", None)
    h = len(mean) if isinstance(mean, pd.Series) and len(mean) else max(1, int(_finite(report.forecast.horizon) or 1))
    lv = _outlook_levels(report, d)
    if lv is not None:
        p = _finite(lv.period)
        period = int(p) if p is not None and p > 1 else None
        return int(lv.k_fut), int(lv.k_cur), h, period, bool(lv.adj_fut), bool(lv.adj_cur)
    m = max(1, int(_finite(getattr(report.analysis, "season_length", 1)) or 1))
    n = max(1, int(np.isfinite(d.hist_y).sum()))
    return min(m, h), min(m, n), h, None, False, False


def _direction_word(report: TrendReport, d: "_ChartData", lang: str = "ko") -> str:
    """The outlook's direction word, exactly as the narrative names it (:func:`narrative.direction_label`):
    ``상승`` / ``하락`` / ``보합`` / ``방향 불확실`` (a 'flat' call on a visible change, or on a level
    expected to cross zero); ``upward`` / ``downward`` / ``flat`` / ``direction uncertain`` for ``'en'``."""
    lv = _outlook_levels(report, d)
    o = report.outlook
    return _direction_label(str(getattr(o, "direction", "flat")), getattr(o, "expected_change_pct", None), lang,
                            getattr(lv, "undefined", None))


_UNCERTAIN_KO = _direction_label("flat", 100.0, "ko")   # '방향 불확실'


#: prefix of a level whose partial-season window was seasonally adjusted (narrative: "계절 효과를 보정한")
_ADJ_KO = "계절 보정한 "


def _change_basis(report: TrendReport, d: "_ChartData") -> tuple[str, str]:
    """The two levels behind ``Outlook.expected_change_pct`` (see ``forecast.compute_outlook``).

    Returns Korean phrases such as ``('향후 12개월 평균', '최근 12개월 평균')``, with
    ``계절 보정한`` in front of a level whose window covers only part of a season
    (``('계절 보정한 향후 26주 평균', '최근 52주 평균')``; see :func:`_outlook_windows`).
    Both end in a consonant, so they take the particles 을/과/이.
    """
    freq = d.freq
    unit = PERIOD_KO.get(freq, "기간")
    k_fut, k_cur, h, _, adj_fut, adj_cur = _outlook_windows(report, d)
    if k_fut <= 1:
        end = _fmt_date(d.fc_dates[-1], freq) if len(d.fc_dates) else ""
        fut = f"{end} 예측값".strip()
    elif k_fut >= h:
        fut = f"향후 {h:,}{unit} 평균"
    else:
        fut = f"예측 마지막 {k_fut:,}{unit} 평균"
    cur = "최근 값" if k_cur <= 1 else f"최근 {k_cur:,}{unit} 평균"
    return (_ADJ_KO + fut if adj_fut else fut), (_ADJ_KO + cur if adj_cur else cur)


def _change_basis_en(report: TrendReport, d: "_ChartData") -> tuple[str, str]:
    """:func:`_change_basis` in English: ``('next 12-month avg', 'last 12-month avg')``,
    ``('seasonally adj. next 26-week avg', 'last 52-week avg')``, ``('2026 forecast', 'latest value')``."""
    k_fut, k_cur, h, _, adj_fut, adj_cur = _outlook_windows(report, d)
    if k_fut <= 1:
        end = _fmt_date(d.fc_dates[-1], d.freq) if len(d.fc_dates) else ""
        fut = f"{end} forecast".strip()
    elif k_fut >= h:
        fut = f"next {_span_adj_en(h, d.freq)} avg"
    else:
        fut = f"forecast's last {_span_adj_en(k_fut, d.freq)} avg"
    cur = "latest value" if k_cur <= 1 else f"last {_span_adj_en(k_cur, d.freq)} avg"
    adj = "seasonally adj. "
    return (adj + fut if adj_fut else fut), (adj + cur if adj_cur else cur)


def _adjustment_note(report: TrendReport, d: "_ChartData") -> str:
    """``'26주 구간은 계절 주기(52주)를 다 채우지 못해 … 빼고 비교했습니다.'``; ``''`` when nothing was adjusted."""
    k_fut, k_cur, _, period, adj_fut, adj_cur = _outlook_windows(report, d)
    if period is None or not (adj_fut or adj_cur):
        return ""
    spans = list(dict.fromkeys(_span_text(k, d.freq) for k, adj in ((k_fut, adj_fut), (k_cur, adj_cur)) if adj))
    return (f"{'·'.join(spans)} 구간은 계절 주기({_span_text(period, d.freq)})를 다 채우지 못해, 그 구간의 평균적인 "
            "계절 높낮이를 빼고 비교했습니다.")


def _prob_basis(report: TrendReport, d: "_ChartData") -> str:
    """What ``Outlook.prob_increase`` is: ``'향후 12개월 평균이 최근 12개월 평균보다 높을 확률'``.

    The same two levels as ``expected_change_pct`` (see :func:`_change_basis`),
    *not* the chance that the horizon-end value (the scenarios) ends above today.
    """
    fut, cur = _change_basis(report, d)
    return f"{fut}이 {cur}보다 높을 확률"


class _NoChange:
    """Why ``Outlook.expected_change_pct`` is ``None`` (see :func:`_change_undefined`)."""

    def __init__(self, code: Optional[str], short: str, sentence: str, levels: str) -> None:
        self.code = code          # forecast.outlook_levels(...).undefined
        self.short = short        # tile / terminal phrase: '0을 지나 음수로 바뀔 전망'
        self.sentence = sentence  # full sentence: '수준이 0을 지나 … 의미가 없습니다.'
        self.levels = levels      # '최근 12개월 평균 +4.72 → 향후 12개월 평균 −3.90' ('' if unknown)


def _change_undefined(report: TrendReport, d: "_ChartData", lang: str = "ko") -> Optional[_NoChange]:
    """Why the outlook states no percentage, worded like the narrative's outlook line.

    ``None`` when ``expected_change_pct`` is a number.  Otherwise the reason of
    :func:`forecast.outlook_levels` -- ``'sign_change'`` (the level is expected to
    cross zero), ``'near_zero'`` (a level close to 0 for data with negative values),
    ``'mostly_zero'`` (intermittent demand) or ``'zero'`` -- with the two compared
    levels, so a ``—`` is never left unexplained.  ``lang='en'``: the same in English
    (the terminal summary only uses ``short`` and ``levels``).
    """
    if _finite(getattr(report.outlook, "expected_change_pct", None)) is not None:
        return None
    lv = _outlook_levels(report, d)
    code = getattr(lv, "undefined", None)
    fut_b, cur_b = _change_basis_en(report, d) if lang == "en" else _change_basis(report, d)
    cur, fut = (_finite(lv.current), _finite(lv.future)) if lv is not None else (None, None)
    signed = code in ("sign_change", "near_zero")
    scale = max(abs(cur or 0.0), abs(fut or 0.0))
    levels = (f"{cur_b} {_format_number(cur, signed=signed, scale=scale)} → "
              f"{fut_b} {_format_number(fut, signed=signed, scale=scale)}"
              if cur is not None and fut is not None else "")
    if lang == "en":
        if code == "sign_change" and fut is not None:
            to = "negative" if fut < 0 else "positive"
            return _NoChange(code, f"expected to cross zero to {to}",
                             "The level is expected to cross zero, so a % change is not meaningful.", levels)
        if code in ("near_zero", "zero"):
            return _NoChange(code, "current level near zero",
                             "The current level is close to zero, so a % change is not meaningful.", levels)
        if code == "mostly_zero":
            k_cur = _outlook_windows(report, d)[1]
            what = (f"at least half of the last {_span_en(k_cur, d.freq)} are zero" if k_cur > 1
                    else "the latest value is zero")
            return _NoChange(code, what, f"No % change: {what}.", levels)
        return _NoChange(code, "no % change", "No % change was computed.", levels)
    if code == "sign_change" and fut is not None:
        to = "음수로" if fut < 0 else "양수로"
        return _NoChange(code, f"0을 지나 {to} 바뀔 전망",
                         f"수준이 0을 지나 {to} 바뀔 것으로 예상되어 변화율(%)은 의미가 없습니다.", levels)
    if code == "near_zero":
        return _NoChange(code, "현재 수준이 0에 가까움",
                         f"{cur_b}이 값의 평소 변동 폭에 비해 0에 가까워 변화율(%)은 의미가 없습니다.", levels)
    if code == "mostly_zero":
        k_cur = _outlook_windows(report, d)[1]
        what = f"최근 {_span_text(k_cur, d.freq)} 중 절반 이상이 0" if k_cur > 1 else "최근 값이 0"
        return _NoChange(code, what, f"{what}이라 변화율은 계산하지 않았습니다.", levels)
    if code == "zero":
        return _NoChange(code, "현재 수준이 0에 가까움", "현재 수준이 0에 가까워 변화율은 계산하지 않았습니다.",
                         levels)
    return _NoChange(code, "변화율 없음", "변화율은 계산하지 않았습니다.", levels)


def _horizon_steps(report: TrendReport) -> int:
    """Number of forecast steps (length of the forecast mean, else ``forecast.horizon``); 0 if unknown."""
    mean = getattr(report.forecast, "mean", None)
    if isinstance(mean, pd.Series) and len(mean):
        return len(mean)
    return max(0, int(_finite(getattr(report.forecast, "horizon", None)) or 0))


def _recent_lag(report: TrendReport, n: int) -> int:
    """Span (periods) of ``metrics.recent_change_pct``: as analysis.compute_metrics, one
    season (1 for yearly data), capped at ``n - 1``."""
    freq = str(getattr(report.analysis, "freq", "M"))
    try:
        season = _season_length(freq)
    except ValueError:  # unknown freq code: fall back to what the analysis recorded
        season = int(getattr(report.analysis, "season_length", 1) or 1)
    return max(1, min(1 if freq == "Y" else season, n - 1))


def _span_text(periods: int, freq: str) -> str:
    """``12개월``, ``30일``, ``8분기``, ``5년``; ``''`` for a non-positive span."""
    return f"{periods:,}{PERIOD_KO.get(freq, '기간')}" if periods > 0 else ""


_PERIOD_EN: dict[str, tuple[str, str]] = {"D": ("day", "days"), "W": ("week", "weeks"), "M": ("month", "months"),
                                          "Q": ("quarter", "quarters"), "Y": ("year", "years")}
_FREQ_EN: dict[str, str] = {"D": "daily", "W": "weekly", "M": "monthly", "Q": "quarterly", "Y": "yearly"}


def _span_en(periods: int, freq: str) -> str:
    """``12 months``, ``1 week``; ``''`` for a non-positive span."""
    one, many = _PERIOD_EN.get(freq, ("period", "periods"))
    return f"{periods:,} {one if periods == 1 else many}" if periods > 0 else ""


def _span_adj_en(periods: int, freq: str) -> str:
    """Compound adjective ``12-month``, ``26-week`` (as the English narrative)."""
    return f"{periods:,}-{_PERIOD_EN.get(freq, ('period', 'periods'))[0]}"


def _excluded_spike(report: TrendReport, d: "_ChartData") -> Optional[tuple[float, pd.Timestamp]]:
    """The observed maximum when ``metrics.peak_value`` left it out as a one-off spike.

    analysis.compute_metrics ignores upward spikes flagged by ``detect_anomalies``
    for the peak and the drawdown.  Returns ``(value, date)`` of the raw maximum
    if it is higher than ``peak_value`` *and* flagged as an upward anomaly, else ``None``.
    """
    peak = _finite(getattr(report.analysis.metrics, "peak_value", None))
    fin = np.isfinite(d.hist_y)
    if peak is None or not fin.any():
        return None
    i = int(np.flatnonzero(fin)[int(np.argmax(d.hist_y[fin]))])
    top = float(d.hist_y[i])
    if not top > peak + 1e-9 * max(abs(top), abs(peak), 1e-300):
        return None
    day = round(float(d.hist_t[i]), 6)
    for an in getattr(report.analysis, "anomalies", None) or []:
        t = d.day_of(getattr(an, "date", None))
        z = _finite(getattr(an, "zscore", None))
        if t is not None and round(t, 6) == day and z is not None and z > 0:
            return top, d.hist_dates[i]
    return None


#: seasonal strength from which the drawdown may be shown on the trend (a seasonal low is not a fall)
_SEASONAL_DD_STRENGTH = 0.5
#: ... and only when the raw and the trend drawdown differ by at least this many points
#: (the rule of narrative._seasonal_drawdown, so the tile and the narrative use the same basis)
_SEASONAL_DD_GAP = 5.0
#: seasonal strength from which horizon-end values are compared with the same point of the cycle
_SEASONAL_REF_STRENGTH = 0.3


def _trend_drawdown(report: TrendReport, d: "_ChartData") -> Optional[float]:
    """Latest trend value versus the trend maximum, in % (<= 0); ``None`` if undefined
    (fewer than two values, a maximum that is not positive, or a trend that crossed zero
    -- :func:`analysis.pct_change_undefined` from the maximum to the latest value) -- as
    ``narrative._trend_drawdown``."""
    tr = d.trend_y[np.isfinite(d.trend_y)]
    if tr.size < 2:
        return None
    pk, last = float(tr.max()), float(tr[-1])
    if not pk > 0:
        return None
    if _pct_change_undefined(pk, last, d.hist_y) is not None:
        return None   # the trend crossed zero (or peaked near 0 among negative values)
    dd = (last - pk) / pk * 100.0
    return min(0.0, dd) if math.isfinite(dd) else None


def _drawdown_undefined(report: TrendReport, d: "_ChartData") -> Optional[str]:
    """Why ``metrics.drawdown_pct`` is no meaningful percentage (``None``: it is one).

    ``drawdown_pct`` is always a number, but for a series that crossed zero (a P&L
    from 10 to −8 reads "−179 %"), whose peak is not positive, or whose peak is
    near 0 relative to the spread of data with negative values it means nothing:
    :func:`analysis.pct_change_undefined` from the peak to the latest value, the
    rule of ``narrative._drawdown_undefined``.
    """
    m = report.analysis.metrics
    pk, last = _finite(getattr(m, "peak_value", None)), _finite(getattr(m, "last_value", None))
    if pk is None or last is None:
        return None
    return _pct_change_undefined(pk, last, d.hist_y)


def _seasonal_trend_drawdown(report: TrendReport, d: "_ChartData") -> Optional[float]:
    """The trend drawdown when it, not the raw one, describes the distance from the peak.

    With a strong seasonal pattern (``seasonal_strength >= 0.5``) the raw drawdown
    may mostly measure where the season is now (e.g. −25.6 % in a stable series
    that simply sits in its low season).  Exactly the rule of the narrative
    (``narrative._seasonal_drawdown``): the trend drawdown is used when the series
    is seasonal *and* it differs from the raw drawdown by at least 5 points;
    otherwise ``None`` (the raw drawdown is used, as in the narrative's "최근
    값은 고점보다 X% 낮아" sentence).  Never for a drawdown that is no percentage
    (:func:`_drawdown_undefined`).
    """
    m = report.analysis.metrics
    period = int(_finite(getattr(report.analysis.decomposition, "period", 1)) or 1)
    ss = _finite(getattr(m, "seasonal_strength", None))
    if period <= 1 or ss is None or ss < _SEASONAL_DD_STRENGTH or _drawdown_undefined(report, d) is not None:
        return None
    tdd = _trend_drawdown(report, d)
    dd = _finite(m.drawdown_pct)
    if tdd is None or abs((dd if dd is not None else 0.0) - tdd) < _SEASONAL_DD_GAP:
        return None
    return tdd


def _drawdown_sub(report: TrendReport, d: "_ChartData") -> str:
    """The "현재 값" tile's peak phrase: ``최고점 대비 −4.9%`` / ``추세 고점 대비 −1.3%`` / ``현재 최고 수준``.

    The basis is the narrative's (see :func:`_seasonal_trend_drawdown`): the
    observed drawdown, next to the observed current value, unless a seasonal
    low or high distorts it -- then the trend's drawdown, labelled ``추세 고점``.
    On the observation basis a flagged one-off spike that the peak leaves out is
    mentioned (``일회성 급등 제외``).  As in the narrative, no percentage where it
    means nothing: intermittent demand names its largest demand instead
    (``최대 12.0 (2023-02-19)``), and a series that crossed zero (or whose peak
    is not positive) gives the gap in its own units (``최고점보다 18.2 낮음``).
    """
    m = report.analysis.metrics
    spike = " (일회성 급등 제외)" if _excluded_spike(report, d) else ""
    if _intermittent(report) and _finite(m.peak_value) is not None:
        return (f"최대 {_fmt_short(m.peak_value)} ({_fmt_date(m.peak_date, d.freq)}"
                + (", 일회성 급등 제외)" if spike else ")"))
    dd = _finite(m.drawdown_pct)
    if _drawdown_undefined(report, d) is not None and dd is not None and dd < -0.05:
        gap = abs(float(m.peak_value) - float(m.last_value))
        return f"최고점보다 {_fmt_short(gap)} 낮음{spike}"
    tdd = _seasonal_trend_drawdown(report, d)
    if tdd is not None:
        return f"추세 고점 대비 {_fmt_pct(tdd)}" if tdd < -0.05 else "추세 최고 수준"
    if dd is None or dd >= -0.05:
        return "현재 최고 수준"
    return f"최고점 대비 {_fmt_pct(dd)}{spike}"


#: why a growth metric is None, as a short tile / terminal phrase (analysis.pct_change_undefined codes,
#: plus 'intermittent' and 'short_span'); {base} is 시작 수준 for the whole-period change and the CAGR,
#: 기준 수준 for the recent change -- the narrative's subjects
_PCT_WHY: dict[str, dict[str, str]] = {
    "ko": {
        "sign_change": "0을 지나 음수로 바뀌어 변화율 없음",
        "negative": "{base}이 음수라 변화율 없음",
        "near_zero": "{base}이 0에 가까워 변화율 없음",
        "zero": "{base}이 0에 가까워 변화율 없음",
        "intermittent": "간헐 수요라 계산하지 않음",
        "short_span": "기간이 1년 미만이라 연 환산하지 않음",
    },
    "en": {
        "sign_change": "crossed zero, so no % change",
        "negative": "the {base} is negative, so no % change",
        "near_zero": "the {base} is near zero, so no % change",
        "zero": "the {base} is near zero, so no % change",
        "intermittent": "intermittent demand, not computed",
        "short_span": "less than a year of data, not annualised",
    },
}
_PCT_BASE: dict[str, tuple[str, str]] = {"ko": ("시작 수준", "기준 수준"), "en": ("starting level", "base level")}

#: analysis.compute_metrics computes a CAGR only over at least this many years
_CAGR_MIN_YEARS = 0.95


def _pct_why(code: Optional[str], key: str, lang: str = "ko") -> str:
    """The :data:`_PCT_WHY` phrase of ``code`` for the growth metric ``key`` (``''`` for no reason)."""
    table = _PCT_WHY["en" if lang == "en" else "ko"]
    if code not in table:
        return ""
    start, base = _PCT_BASE["en" if lang == "en" else "ko"]
    return table[code].format(base=base if key == "recent_change_pct" else start)


def _metric_undefined(report: TrendReport, d: "_ChartData", key: str) -> Optional[str]:
    """Why the growth metric ``key`` (``total_change_pct``, ``cagr_pct``, ``recent_change_pct``) is ``None``.

    ``'intermittent'`` for mostly-zero demand (the analysis computes no growth
    rates for it); ``'short_span'`` for a CAGR over less than a year (the analysis
    annualises only spans of at least 0.95 years: ``(n - 1) / periods_per_year``);
    else :func:`analysis.pct_change_undefined` of the trend from the metric's base
    to the latest trend value (``'sign_change'``, ``'negative'``, ``'near_zero'``,
    ``'zero'``) -- the rule of the metrics themselves and of the narrative.
    ``None`` when the metric is a number or no reason applies, so a ``—`` is
    explained wherever the analysis' own rule can say why.
    """
    if _finite(getattr(report.analysis.metrics, key, None)) is not None:
        return None
    if _intermittent(report):
        return "intermittent"
    if key == "cagr_pct":
        try:
            years = (len(d.hist_y) - 1) / _periods_per_year(d.freq)
        except (ValueError, KeyError, ZeroDivisionError):
            years = None
        if years is not None and years < _CAGR_MIN_YEARS:
            return "short_span"
    tr = d.trend_y[np.isfinite(d.trend_y)]
    if tr.size < 2:
        return None
    if key == "recent_change_pct":
        lag = _recent_lag(report, len(d.hist_y))
        if tr.size <= lag:
            return None
        base = float(tr[-1 - lag])
    else:
        base = float(tr[0])
    return _pct_change_undefined(base, float(tr[-1]), d.hist_y)


def _nonnegative(report: TrendReport) -> bool:
    return bool(getattr(report.forecast, "nonnegative", False))


def _scen_text(report: TrendReport, value: Any, short: bool = False, lang: str = "ko",
               scale: Optional[float] = None) -> str:
    """A horizon-end scenario value: ``0(하한)`` / ``0 (floor)`` at the 0 floor of a nonnegative
    forecast (:func:`narrative.scenario_value`: the band was cut at 0, the value is a floor, not a
    forecast of zero), else the number (abbreviated with ``short``); ``—`` if missing.  ``scale``: the
    largest magnitude printed next to it (:func:`narrative.format_number`), so one line keeps one notation."""
    v = _finite(value)
    if v is None:
        return _DASH
    text, _ = _scenario_value(v, None, _nonnegative(report), lang)
    if _nonnegative(report) and v <= 0.0:
        return text
    return _fmt_short(v) if short else _format_number(v, scale=scale)


def _floor_note(report: TrendReport) -> str:
    """The narrative's note when a scenario sits at the 0 floor of a nonnegative forecast; ``''`` otherwise."""
    if not _nonnegative(report):
        return ""
    scen = getattr(report.outlook, "scenarios", None) or {}

    def floored(k: str) -> bool:
        v = _finite(scen.get(k))
        return v is not None and v <= 0.0

    if floored("base"):
        return "기준 예측이 하한인 0까지 내려갑니다(값은 0 아래로 내려갈 수 없음)."
    if floored("pessimistic"):
        return "예측구간이 넓어 비관 시나리오는 하한인 0에 닿습니다(값은 0 아래로 내려갈 수 없음)."
    return ""


def _scenario_ref(report: TrendReport, d: "_ChartData") -> Optional[tuple[float, pd.Timestamp, bool]]:
    """The observation the horizon-end scenarios can be compared with: ``(value, date, seasonal)``.

    The scenarios are single values at the last forecast date and carry that
    date's seasonal effect, so with a noticeable seasonal pattern
    (``seasonal_strength >= 0.3``, as the narrative's seasonal note) they are
    compared with the latest observation at the *same position in the seasonal
    cycle* (``seasonal=True``; for monthly data with a 12-month horizon that is
    the latest month) -- not with a season mean, which is a different quantity.
    Otherwise the latest observation (the "현재 값" tile) is the reference
    (``seasonal=False``).
    """
    fin = np.isfinite(d.hist_y)
    if not fin.any():
        return None
    n = len(d.hist_y)
    last = int(np.flatnonzero(fin)[-1])
    period = int(_finite(getattr(report.analysis.decomposition, "period", 1)) or 1)
    ss = _finite(getattr(report.analysis.metrics, "seasonal_strength", None)) or 0.0
    h = _horizon_steps(report)
    if period > 1 and ss >= _SEASONAL_REF_STRENGTH and h > 0 and last == n - 1:
        j = n - 1 + h - period * math.ceil(h / period)
        if 0 <= j < n and np.isfinite(d.hist_y[j]):
            return float(d.hist_y[j]), d.hist_dates[j], True
    return float(d.hist_y[last]), d.hist_dates[last], False


def _is_multiplicative(report: TrendReport) -> bool:
    """Whether the analysis decomposed the series in log space (seasonal swing scaling with the level)."""
    try:
        from .analysis import _is_multiplicative as _mult

        return bool(_mult(report.analysis.decomposition))
    except Exception:  # noqa: BLE001 - hand-made / partial decompositions: treat as additive
        return False


_SAFE_SCHEMES = ("http", "https")


def _safe_href(href: Any) -> Optional[str]:
    """A link target that is safe to put in ``href``: relative or http(s) only, else ``None``.

    Rejects ``javascript:``, ``data:``, ``vbscript:``, ``file:`` ... and any value
    with control characters (browsers drop tabs/newlines, so ``java\\tscript:``
    would still run).  Spaces and non-ASCII characters (e.g. Korean file names)
    are percent-encoded; the result still needs HTML escaping.
    """
    if href is None:
        return None
    s = _clean_text(str(href))
    if s != str(href) or any(unicodedata.category(c) == "Cc" for c in s):
        return None
    s = s.strip()
    if not s:
        return None
    try:
        parts = urlsplit(s)
    except ValueError:
        return None
    if parts.scheme and parts.scheme.lower() not in _SAFE_SCHEMES:
        return None
    return quote(s, safe="/?#[]@!$&'()*+,;=%~:")


def _text_width(s: str, size: float) -> float:
    """Rough rendered width of ``s`` in px for a sans-serif font of ``size`` px."""
    w = 0.0
    for ch in s:
        if unicodedata.east_asian_width(ch) in ("W", "F"):
            w += 1.0
        elif ch == " ":
            w += 0.3
        elif ch in ".,:;|!il'`()[]":
            w += 0.32
        elif ch.isdigit():
            w += 0.6
        elif ch.isupper() or ch in "%MW@":
            w += 0.72
        else:
            w += 0.56
    return w * size


def _f1(v: float) -> str:
    """Compact coordinate formatting for SVG."""
    s = f"{v:.1f}".removesuffix(".0")
    return "0" if s == "-0" else s


def _uid(*parts: Any) -> str:
    """Deterministic id suffix (unique per report/size) for SVG ``id`` attributes."""
    # surrogatepass: names may contain lone surrogates (see _clean_text)
    return hashlib.md5("|".join(map(str, parts)).encode("utf-8", "surrogatepass")).hexdigest()[:8]


# --------------------------------------------------------------------------- sparkline


def _unit_pos(v: Any, lo: float, hi: float) -> Any:
    """``(v - lo) / (hi - lo)``, also when ``hi - lo`` overflows (axes spanning ±1e308)."""
    arr = np.asarray(v, dtype=float)
    if math.isfinite(hi - lo):
        return (arr - lo) / (hi - lo)
    with np.errstate(over="ignore", invalid="ignore"):
        return (arr * 0.5 - lo * 0.5) / (hi * 0.5 - lo * 0.5)


def _resample(arr: np.ndarray, width: int) -> np.ndarray:
    """Bin-average down to ``width`` points, or linearly interpolate up to it."""
    n = len(arr)
    if width == n or n == 0:
        return arr
    if width < n:
        return np.array([float(c.mean()) for c in np.array_split(arr, width)])
    if n == 1:
        return np.repeat(arr, width)
    return np.interp(np.linspace(0.0, n - 1.0, width), np.arange(n, dtype=float), arr)


def _spark_chars(arr: np.ndarray, lo: float, hi: float) -> str:
    span = hi - lo
    if len(arr) == 0:
        return ""
    # relative flatness test (an absolute floor would flatten genuinely tiny-valued data)
    if not span > 1e-9 * max(abs(lo), abs(hi)):
        return _BLOCKS[3] * len(arr)
    idx = np.clip(np.rint(_unit_pos(arr, lo, hi) * (len(_BLOCKS) - 1)), 0, len(_BLOCKS) - 1).astype(int)
    return "".join(_BLOCKS[i] for i in idx)


def _as_float_array(values: Any) -> np.ndarray:
    if values is None or isinstance(values, (str, bytes)):
        return np.zeros(0)
    try:
        if isinstance(values, (np.ndarray, pd.Series, pd.Index)):
            arr = np.asarray(values, dtype=float)
        else:
            arr = np.asarray(list(values), dtype=float)
    except (TypeError, ValueError):
        return np.zeros(0)
    arr = arr.ravel()
    return arr[np.isfinite(arr)]


def sparkline(values: Iterable[float], width: Optional[int] = None) -> str:
    """Unicode block sparkline (``▁▂▃▄▅▆▇█``) of ``values``.

    Non-finite values are dropped.  With ``width`` the data is resampled to
    exactly ``width`` characters (bin averages when shrinking, linear
    interpolation when stretching).  A flat input gives a row of middle blocks;
    empty input (or ``width <= 0``) gives ``''``.
    """
    arr = _as_float_array(values)
    if arr.size == 0:
        return ""
    if width is not None:
        w = int(width)
        if w <= 0:
            return ""
        arr = _resample(arr, w)
    return _spark_chars(arr, float(arr.min()), float(arr.max()))


# --------------------------------------------------------------------------- axis maths


#: magnitude from which ticks are computed in units of 10**k (spans and 10**exp must stay finite)
_TICK_RESCALE = 1e300


def _nice_ticks(lo: float, hi: float, min_n: int = 4, max_n: int = 6,
                limit: float = sys.float_info.max) -> list[float]:
    """4–6 "nice" ticks covering ``[lo, hi]``.

    Steps are 1, 2, 2.5, 5 × 10^k (3 and 4 × 10^k only when they fit much tighter);
    the step that wastes the least vertical space wins.  Any finite range works:
    from ``1e300`` up the ticks are computed in units of ``10**k`` (so neither the
    span nor ``10**exp`` overflows) and never go beyond ``±limit`` (the largest
    float), falling back to evenly spaced ticks if no nice step fits there; a
    subnormal range whose step underflows to 0 (``[0, 5e-324]``) gets its two ends
    (or, when flat, the value and the floats next to it) -- at least two distinct ticks.
    """
    if not (math.isfinite(lo) and math.isfinite(hi)):
        lo, hi = 0.0, 1.0
    if hi < lo:
        lo, hi = hi, lo
    mag = max(abs(lo), abs(hi))
    if mag >= _TICK_RESCALE and limit == sys.float_info.max:
        scale = 10.0 ** math.floor(math.log10(mag))
        ticks = _nice_ticks(lo / scale, hi / scale, min_n, max_n, limit=sys.float_info.max / scale)
        out: list[float] = []
        for t in ticks:
            v = t * scale
            if math.isfinite(v) and (not out or v > out[-1]):
                out.append(v + 0.0)
        return out
    if hi - lo <= 1e-12 * mag:  # flat (relative test; also lo == hi == 0)
        pad = abs(hi) * 0.1 if hi != 0 else 1.0
        if hi + pad > limit:        # next to the largest float: widen downwards only
            lo, hi = lo - 2 * pad, hi
        elif lo - pad < -limit:
            lo, hi = lo, hi + 2 * pad
        else:
            lo, hi = lo - pad, hi + pad
    span = hi - lo
    unit = span / max(1, max_n - 1)
    if not (span > 0 and unit > 0 and math.isfinite(unit)):
        # subnormal ranges (|values| ~ 5e-324): the span or the step underflows to 0, so no
        # 10**k step exists: the range itself, widened to the neighbouring floats when flat
        if hi > lo:
            return [lo + 0.0, hi + 0.0]
        return [math.nextafter(lo, -math.inf) + 0.0, lo + 0.0, math.nextafter(hi, math.inf) + 0.0]
    e = math.floor(math.log10(unit))
    best: Optional[tuple[tuple[float, float], float, float, int]] = None
    for exp in range(e - 1, e + 3):
        for mant, penalty in ((1.0, 0.0), (2.0, 0.0), (2.5, 0.03), (3.0, 0.1), (4.0, 0.1), (5.0, 0.0)):
            step = mant * 10.0 ** exp
            a = math.floor(lo / step + 1e-9) * step
            b = math.ceil(hi / step - 1e-9) * step
            n = int(round((b - a) / step)) + 1
            if n < min_n or n > max_n or a < -limit or b > limit:
                continue
            key = (span / (b - a) - penalty, -abs(n - 5))
            if best is None or key > best[0]:
                best = (key, a, step, n)
    if best is None:  # only next to the largest float: evenly spaced ticks from lo to hi
        return [lo + (hi - lo) * i / 4.0 for i in range(4)] + [hi]
    _, a, step, n = best
    dec = max(0, -math.floor(math.log10(step)) + 2)
    return [round(a + i * step, dec) + 0.0 for i in range(n)]


def _axis_labels(ticks: Sequence[float]) -> list[str]:
    """Tick labels: thousands separators below 10,000, k/M/B abbreviations above.

    Labels are always distinct.  Ticks that reach 1e12 or stay below 1e-4 -- where
    :func:`narrative.format_number` switches -- use its ``×10ⁿ`` notation
    (``5×10⁻⁸``, ``1.25×10¹⁷``), with as many digits as the ticks need; so do
    ticks that would need more than 6 decimals.
    """
    if not ticks:
        return []
    m = max(abs(t) for t in ticks)
    out: Optional[list[str]] = None
    if _SCI_SMALL <= m < _SCI_LARGE:
        unit, suffix = 1.0, ""
        for u, sfx in ((1e9, "B"), (1e6, "M"), (1e4, "k")):
            if m >= u:
                unit, suffix = (1e3 if sfx == "k" else u), sfx
                break
        vals = [t / unit for t in ticks]
        diffs = [abs(b - a) for a, b in zip(vals, vals[1:]) if b != a]
        tol = 1e-6 * (min(diffs) if diffs else max(1.0, abs(vals[0])))
        for dec in range(7):
            if all(abs(round(v, dec) - v) <= tol for v in vals):
                out = []
                for v in vals:
                    body = f"{abs(v):,.{dec}f}"
                    if body.strip("0.,") == "":
                        out.append(body + (suffix if body != "0" else ""))
                    else:
                        out.append((_MINUS if v < 0 else "") + body + suffix)
                break
    if out is None or len(set(out)) < len(out):
        steps = [abs(b - a) for a, b in zip(ticks, ticks[1:]) if b != a]
        sci_tol = 1e-6 * (min(steps) if steps else max(abs(t) for t in ticks))
        for digits in range(2, 18):
            exact = all(t == 0 or abs(float(f"{abs(t):.{digits - 1}e}") - abs(t)) <= sci_tol for t in ticks)
            out = ["0" if t == 0 else (_MINUS if t < 0 else "") + _sci(abs(t), digits) for t in ticks]
            if exact and len(set(out)) == len(out):
                break
    return out


def _fmt_axis_value(v: float, ref: float) -> str:
    """A single value formatted like the axis whose largest magnitude is ``ref``: abbreviated
    (k/M/B) when the axis is, and from 1e12 in the narrative's ``×10ⁿ`` notation, as
    :func:`narrative.format_number` writes a number next to such a value (``scale``)."""
    a = abs(ref)
    if a >= _SCI_LARGE:
        return _format_number(v, scale=a)
    if a >= 1e9:
        return _fmt_num(v / 1e9, decimals=2) + "B"
    if a >= 1e6:
        return _fmt_num(v / 1e6, decimals=2) + "M"
    if a >= 1e4:
        return _fmt_num(v / 1e3, decimals=1) + "k"
    return _fmt_num(v)


def _month_add(year: int, month: int, k: int) -> tuple[int, int]:
    m0 = year * 12 + (month - 1) + k
    return m0 // 12, m0 % 12 + 1


def _date_ticks(t0: pd.Timestamp, t1: pd.Timestamp, freq: str, x_of, plot_left: float,
                plot_right: float, fs: float) -> list[tuple[float, str]]:
    """Choose date ticks for ``[t0, t1]`` whose labels do not overlap.

    ``x_of`` maps a Timestamp to an x pixel.  Candidates go from fine to coarse
    and the finest one whose labels fit (with a gap) wins.
    """
    span_days = max((t1 - t0) / pd.Timedelta(days=1), 0.0)
    cands: list[tuple[str, int]] = []
    if freq == "D":
        cands += [("D", 1), ("D", 2), ("D", 7), ("D", 14)]
    elif freq == "W":
        cands += [("D", 7), ("D", 14)]
    if freq in ("D", "W", "M"):
        cands += [("M", 1), ("M", 2)]
    if freq in ("D", "W", "M", "Q"):
        cands += [("M", 3), ("M", 6)]
    cands += [("Y", k) for k in (1, 2, 5, 10, 20, 25, 50, 100, 250, 500, 1000)]

    def label(t: pd.Timestamp, kind: str) -> str:
        if kind == "Y":
            return f"{t.year:04d}"
        if kind == "M":
            return f"{t.year:04d}-Q{(t.month - 1) // 3 + 1}" if freq == "Q" else f"{t.year:04d}-{t.month:02d}"
        return f"{t.year:04d}-{t.month:02d}-{t.day:02d}"

    def generate(kind: str, step: int) -> Optional[list[pd.Timestamp]]:
        out: list[pd.Timestamp] = []
        if kind == "D":
            if span_days / step > 400:
                return None
            start = t0.normalize()
            if start < t0:
                start += pd.Timedelta(days=1)
            t = start
            while t <= t1:
                out.append(t)
                t += pd.Timedelta(days=step)
        elif kind == "M":
            if span_days / 30.4 / step > 400:
                return None
            y, m = t0.year, t0.month
            while True:
                t = pd.Timestamp(year=y, month=m, day=1)
                if t > t1:
                    break
                if t >= t0 and (m - 1) % step == 0:
                    out.append(t)
                y, m = _month_add(y, m, 1)
        else:
            if span_days / 365.25 / step > 400:
                return None
            y0 = int(math.ceil(t0.year / step) * step)
            for y in range(y0, t1.year + 1, step):
                if y < 1 or y > 9999:
                    continue
                t = pd.Timestamp(year=y, month=1, day=1)
                if t0 <= t <= t1:
                    out.append(t)
        return out

    gap = fs * 1.1
    chosen: Optional[list[tuple[float, str]]] = None
    for kind, step in cands:
        ts = generate(kind, step)
        if ts is None or len(ts) < 2:
            if ts is not None and len(ts) < 2 and kind == "Y":
                break  # coarser year steps only get sparser
            continue
        items = [(x_of(t), label(t, kind)) for t in ts]
        widths = [_text_width(s, fs) for _, s in items]
        ok = all(items[i + 1][0] - items[i][0] >= (widths[i] + widths[i + 1]) / 2 + gap
                 for i in range(len(items) - 1))
        if ok:
            chosen = items
            break
    if chosen is None:
        # fall back to the two end points (always fit unless the plot is tiny)
        kind = "D" if freq in ("D", "W") else ("M" if freq in ("M", "Q") else "Y")
        a = label(t0, kind)
        b = label(t1, kind)
        chosen = [(x_of(t0), a)]
        if x_of(t1) - x_of(t0) >= _text_width(a, fs) + _text_width(b, fs) + gap and b != a:
            chosen.append((x_of(t1), b))
    return chosen


# --------------------------------------------------------------------------- chart data


class _ChartData:
    """Numeric arrays (x in days since the first observation) shared by all charts."""

    def __init__(self, report: TrendReport) -> None:
        a = report.analysis
        fc = report.forecast
        self.freq = str(getattr(a, "freq", "M"))
        # forecast.outlook_levels of the report, computed on first use (see _outlook_levels)
        self.levels: Any = None
        self.levels_done = False
        hist = a.series
        hist_idx = self._index(hist)
        self.t0 = hist_idx[0] if len(hist_idx) else pd.Timestamp("2000-01-01")
        self.hist_dates = hist_idx
        self.hist_t = self._days(hist_idx)
        self.hist_y = np.asarray(hist, dtype=float).ravel() if len(hist) else np.zeros(0)
        if len(self.hist_y) != len(self.hist_t):  # pragma: no cover - defensive
            k = min(len(self.hist_y), len(self.hist_t))
            self.hist_y, self.hist_t, self.hist_dates = self.hist_y[:k], self.hist_t[:k], hist_idx[:k]

        trend = getattr(getattr(a, "decomposition", None), "trend", None)
        if isinstance(trend, pd.Series) and len(trend):
            self.trend_t = self._days(self._index(trend))
            self.trend_y = np.asarray(trend, dtype=float).ravel()
            if len(self.trend_t) != len(self.trend_y):  # pragma: no cover - defensive
                self.trend_t = self.trend_y = np.zeros(0)
        else:
            self.trend_t = self.trend_y = np.zeros(0)
        self.trend_map = {float(t): float(y) for t, y in zip(self.trend_t, self.trend_y)}

        mean = getattr(fc, "mean", None)
        if isinstance(mean, pd.Series) and len(mean):
            self.fc_dates = self._index(mean)
            self.fc_t = self._days(self.fc_dates)
            self.fc_y = np.asarray(mean, dtype=float).ravel()
        else:
            self.fc_dates = pd.DatetimeIndex([])
            self.fc_t = self.fc_y = np.zeros(0)
        # keep only finite forecast points
        ok = np.isfinite(self.fc_y) & np.isfinite(self.fc_t)
        self.fc_dates, self.fc_t, self.fc_y = self.fc_dates[ok], self.fc_t[ok], self.fc_y[ok]

        # prediction bands, widest first so that narrower ones are painted on top
        self.bands: list[tuple[int, np.ndarray, np.ndarray]] = []
        lower = getattr(fc, "lower", None) or {}
        upper = getattr(fc, "upper", None) or {}
        levels = [lv for lv in (getattr(fc, "levels", None) or tuple(lower)) if lv in lower and lv in upper]
        for lv in sorted(set(levels), reverse=True):
            try:
                lo = np.asarray(lower[lv], dtype=float).ravel()
                hi = np.asarray(upper[lv], dtype=float).ravel()
            except (TypeError, ValueError):
                continue
            if len(lo) != len(ok) or len(hi) != len(ok):
                continue
            lo, hi = lo[ok], hi[ok]
            if not lo.size:   # no finite forecast point (e.g. a subnormal series): no band to draw
                continue
            lo = np.where(np.isfinite(lo), lo, self.fc_y)
            hi = np.where(np.isfinite(hi), hi, self.fc_y)
            self.bands.append((int(lv), np.minimum(lo, hi), np.maximum(lo, hi)))

        fin = np.isfinite(self.hist_y) & np.isfinite(self.hist_t)
        if fin.any():
            j = int(np.flatnonzero(fin)[-1])
            self.last_t, self.last_y = float(self.hist_t[j]), float(self.hist_y[j])
            self.last_date = self.hist_dates[j]
        else:
            self.last_t, self.last_y, self.last_date = 0.0, float("nan"), self.t0
        ends = [self.last_t]
        if len(self.fc_t):
            ends.append(float(self.fc_t.max()))
        if len(self.hist_t):
            ends.append(float(np.nanmax(self.hist_t)))
        self.t_end = max(ends)
        # end date taken from the real timestamps: ``t0 + pd.Timedelta(days=t_end)`` would
        # overflow (ns resolution, ~292 years) for long histories such as yearly data since 1700
        cands = [self.last_date]
        if len(self.fc_dates):
            cands.append(self.fc_dates.max())
        if len(self.hist_dates):
            cands.append(self.hist_dates.max())
        self.end_date = max(cands)
        if not self.t_end > 0:
            self.t_end = 1.0  # single point: avoid a zero-width axis
            self.end_date = self.t0 + pd.Timedelta(days=1)

        # history points flagged as anomalies (the last observation, which anchors the
        # "현재" dot and the forecast line, is never treated as one for the y range)
        self.anom_mask = np.zeros(len(self.hist_y), dtype=bool)
        anoms = list(getattr(a, "anomalies", None) or [])
        if anoms and len(self.hist_t):
            days = [self.day_of(getattr(an, "date", None)) for an in anoms]
            days_arr = np.asarray([v for v in days if v is not None], dtype=float)
            if days_arr.size:
                self.anom_mask = np.isin(np.round(self.hist_t, 6), np.round(days_arr, 6))
            if fin.any():
                self.anom_mask[int(np.flatnonzero(fin)[-1])] = False

    @staticmethod
    def _index(s: pd.Series) -> pd.DatetimeIndex:
        idx = s.index
        if not isinstance(idx, pd.DatetimeIndex):
            try:
                idx = pd.DatetimeIndex(pd.to_datetime(idx))
            except (TypeError, ValueError):
                idx = pd.date_range("2000-01-01", periods=len(s), freq="D")
        if idx.tz is not None:
            idx = idx.tz_localize(None)
        try:  # same resolution everywhere (see _ts)
            idx = idx.as_unit("us")
        except (AttributeError, ValueError, OverflowError):  # pragma: no cover - defensive
            pass
        return idx

    def _days(self, idx: pd.DatetimeIndex) -> np.ndarray:
        if len(idx) == 0:
            return np.zeros(0)
        return np.asarray((idx - self.t0) / pd.Timedelta(days=1), dtype=float)

    def day_of(self, ts: Any) -> Optional[float]:
        t = _ts(ts)
        if t is None:
            return None
        v = float((t - self.t0) / pd.Timedelta(days=1))
        return v if math.isfinite(v) else None

    def y_extent(self) -> tuple[float, float]:
        """Value range to show: data (history, trend, forecast) plus clipped bands.

        * A flat series (history/trend/forecast vary by less than 1e-6 of their
          level) gets a +-10 % window around its level, so the axis never zooms
          into numerical noise of the prediction bands.
        * Very wide bands are clipped to about one data range beyond the data.
        * Non-negative data (history, forecast and bands all >= 0) never shows
          a negative axis because of a slight dip of the smoothed trend line.
        * Points flagged as anomalies (``analysis.anomalies``) are left out of
          the range when they more than double it (e.g. one-day news spikes of
          page views), so the level, the seasonal pattern and the forecast stay
          readable; :func:`svg_chart` then marks them at the plot edge with
          their values.  Every other observation always stays inside the range.
        """
        full = self._extent_of(self.hist_y)
        if not self.anom_mask.any() or int((~self.anom_mask & np.isfinite(self.hist_y)).sum()) < 2:
            return full
        robust = self._extent_of(np.where(self.anom_mask, np.nan, self.hist_y))
        if robust[1] - robust[0] < _SPIKE_RATIO * (full[1] - full[0]):
            return robust
        return full

    def spikes_clipped(self) -> bool:
        """Whether :meth:`y_extent` leaves flagged anomalies out of the value range."""
        return bool(self.anom_mask.any()) and self.y_extent() != self._extent_of(self.hist_y)

    def _extent_of(self, hist_y: np.ndarray) -> tuple[float, float]:
        """:meth:`y_extent` computed with ``hist_y`` (NaN = ignored) as the observed values."""
        core = [arr[np.isfinite(arr)] for arr in (hist_y, self.trend_y, self.fc_y)]
        core = [c for c in core if c.size]
        if not core:
            return 0.0, 1.0
        allc = np.concatenate(core)
        dlo, dhi = float(allc.min()), float(allc.max())
        mag = max(abs(dlo), abs(dhi))
        if dhi - dlo <= 1e-6 * mag:  # flat (relative test; also the all-zero series)
            c = 0.5 * (dlo + dhi)
            pad = abs(c) * 0.1 if c != 0 else 1.0
            lo, hi = c - pad, c + pad
        else:
            lo, hi = dlo, dhi
        if self.bands:
            blo = min(float(np.min(b[1])) for b in self.bands)
            bhi = max(float(np.max(b[2])) for b in self.bands)
            ext = max(dhi - dlo, 0.5 * mag, 1e-300)
            lo = min(lo, max(blo, dlo - ext))
            hi = max(hi, min(bhi, dhi + ext))
        if hi - lo <= 1e-12 * max(abs(lo), abs(hi)):  # pragma: no cover - guarded above
            pad = abs(hi) * 0.1 if hi != 0 else 1.0
            lo, hi = lo - pad, hi + pad
        others = [self.hist_y, self.fc_y] + [b[1] for b in self.bands]
        nonneg = all(bool(np.all(arr[np.isfinite(arr)] >= 0)) for arr in others)
        if nonneg and lo < 0 and (-lo <= 0.05 * (hi - lo) or dhi - dlo <= 1e-6 * mag):
            lo = 0.0  # only the smoothed trend dips (slightly) below zero
        if lo >= 0 and lo <= 0.35 * hi:
            lo = 0.0
        return lo, hi


# --------------------------------------------------------------------------- SVG chart

_SVG_STYLE = (
    ".tai-chart text{fill:var(--ink-2,#52514e)}"
    ".tai-chart .tai-ax{fill:var(--ink-3,#6f6d67);font-variant-numeric:tabular-nums}"
    ".tai-chart .tai-strong{fill:var(--ink,#111110);font-weight:600}"
    ".tai-grid{stroke:var(--grid,#e8e7e1);stroke-width:1}"
    ".tai-zero{stroke:var(--axis,#c3c2b7);stroke-width:1}"
    ".tai-base{stroke:var(--axis,#c3c2b7);stroke-width:1}"
    ".tai-region{fill:var(--fc-region,rgba(235,104,52,.045))}"
    ".tai-hist{fill:none;stroke:var(--c-hist,#2a78d6);stroke-width:2;stroke-linejoin:round;stroke-linecap:round}"
    ".tai-trend{fill:none;stroke:var(--c-trend,#76746d);stroke-width:1.5;stroke-dasharray:6 4;stroke-linecap:round}"
    ".tai-fc{fill:none;stroke:var(--c-fc,#eb6834);stroke-width:2;stroke-linejoin:round;stroke-linecap:round}"
    ".tai-band{fill:var(--c-fc,#eb6834)}.tai-bo{fill-opacity:var(--band-o,.11)}.tai-bi{fill-opacity:var(--band-i,.18)}"
    ".tai-now{stroke:var(--ink-3,#6f6d67);stroke-width:1;stroke-dasharray:3 3}"
    ".tai-cp{stroke:var(--c-cp,#1baf7a);stroke-width:1.25;stroke-opacity:.75}"
    ".tai-cpm{fill:var(--c-cp,#1baf7a)}"
    ".tai-cphit{stroke:transparent;stroke-width:10;pointer-events:stroke}"
    ".tai-cpb{fill:var(--c-cp,#1baf7a);stroke:var(--surface,#fff);stroke-width:1.5}"
    ".tai-chart .tai-cpn{fill:#fff;font-weight:700;font-variant-numeric:tabular-nums}"
    ".tai-anom{fill:var(--surface,#fff);stroke:var(--c-anom,#e34948);stroke-width:2}"
    ".tai-spike{fill:var(--c-anom,#e34948);stroke:var(--surface,#fff);stroke-width:1}"
    ".tai-chart .tai-spkl{fill:var(--c-anom,#e34948);font-weight:600}"
    ".tai-dot-h{fill:var(--c-hist,#2a78d6);stroke:var(--surface,#fff);stroke-width:2}"
    ".tai-dot-f{fill:var(--c-fc,#eb6834);stroke:var(--surface,#fff);stroke-width:2}"
    ".tai-chart .tai-lbl{paint-order:stroke;stroke:var(--surface,#fff);stroke-width:4px;stroke-linejoin:round}"
    ".tai-hv .tai-hs{visibility:hidden}.tai-hv:hover .tai-hs{visibility:visible}"
    ".tai-hit{fill:transparent;pointer-events:all}.tai-hs{pointer-events:none}"
    ".tai-xh{stroke:var(--ink-3,#6f6d67);stroke-width:1}"
    ".tai-tip{fill:var(--surface,#fff);stroke:var(--border-strong,rgba(11,11,11,.16));stroke-width:1}"
    ".tai-chart .tai-tipt{fill:var(--ink,#111110)}"
    ".tai-mini .tai-hist,.tai-mini .tai-fc{stroke-width:1.75}"
)


def _path(xs: np.ndarray, ys: np.ndarray) -> str:
    """SVG path data; non-finite points break the line."""
    parts: list[str] = []
    pen = False
    for x, y in zip(xs, ys):
        if not (math.isfinite(x) and math.isfinite(y)):
            pen = False
            continue
        parts.append(("L" if pen else "M") + _f1(x) + "," + _f1(y))
        pen = True
    return "".join(parts)


def _decimate(xs: np.ndarray, ys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Min/max decimation of a polyline given in pixel coordinates.

    Keeps the first, lowest, highest and last point of every half-pixel column,
    so the drawn line is visually identical while very long series (e.g. 200k
    daily points) no longer produce multi-megabyte SVG paths.  Lines with
    gaps (non-finite points) or non-increasing x are returned unchanged.
    """
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    n = len(xs)
    if n < 2:
        return xs, ys
    span = float(xs[-1] - xs[0])
    if (not math.isfinite(span) or n <= 4 * max(span, 1.0) or not np.all(np.isfinite(ys))
            or not np.all(np.diff(xs) >= 0)):
        return xs, ys
    bucket = np.floor((xs - xs[0]) * 2.0).astype(np.int64)
    starts = np.flatnonzero(np.r_[True, bucket[1:] != bucket[:-1]])
    ends = np.r_[starts[1:], n]
    keep: list[int] = []
    for s, e in zip(starts.tolist(), ends.tolist()):
        seg = ys[s:e]
        keep.extend(sorted({s, s + int(np.argmin(seg)), s + int(np.argmax(seg)), e - 1}))
    idx = np.asarray(keep, dtype=np.int64)
    return xs[idx], ys[idx]


def _band_path(xs: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> str:
    pts = [f"{_f1(x)},{_f1(y)}" for x, y in zip(xs, hi)]
    pts += [f"{_f1(x)},{_f1(y)}" for x, y in zip(xs[::-1], lo[::-1])]
    return "M" + "L".join(pts) + "Z" if pts else ""


def _layout_legend(items: list[tuple[str, str, str]], x0: float, max_w: float, fs: float,
                   row_h: float) -> tuple[list[tuple[float, int, tuple[str, str, str]]], int]:
    """Greedy left-to-right legend layout with wrapping. Returns placements and row count."""
    placed = []
    x, row = x0, 0
    for it in items:
        w = 22 + 6 + _text_width(it[2], fs) + 18
        if x > x0 and x + w - 18 > x0 + max_w:
            row += 1
            x = x0
        placed.append((x, row, it))
        x += w
    return placed, (row + 1 if items else 0)


def _legend_svg(placed, top: float, fs: float, row_h: float, band_class: dict[int, str]) -> str:
    out = []
    for x, row, (kind, key, text) in placed:
        cy = top + row * row_h + row_h / 2
        sw = ""
        if kind == "line":
            cls = "tai-hist" if key == "hist" else "tai-fc"
            sw = f'<path class="{cls}" d="M{_f1(x)},{_f1(cy)}h22"/>'
        elif kind == "dash":
            sw = f'<path class="tai-trend" d="M{_f1(x)},{_f1(cy)}h22"/>'
        elif kind == "band":
            bc = band_class.get(int(key), "tai-bi")
            sw = f'<rect class="tai-band {bc}" x="{_f1(x)}" y="{_f1(cy - 5)}" width="22" height="10" rx="2"/>'
        elif kind == "vline":
            sw = (f'<path class="tai-cp" d="M{_f1(x + 11)},{_f1(cy - 7)}v14"/>'
                  f'<circle class="tai-cpm" cx="{_f1(x + 11)}" cy="{_f1(cy - 7)}" r="2.5"/>')
        elif kind == "ring":
            sw = f'<circle class="tai-anom" cx="{_f1(x + 11)}" cy="{_f1(cy)}" r="4"/>'
        out.append(sw + f'<text x="{_f1(x + 28)}" y="{_f1(cy + fs * 0.36)}">{_esc(text)}</text>')
    return '<g class="tai-legend">' + "".join(out) + "</g>"


def svg_chart(report: TrendReport, width: int = 960, height: int = 380) -> str:
    """Inline SVG of the past -> present -> future chart of ``report``.

    Draws the observed history, the trend component (dashed), the forecast mean
    with layered 95 % / 80 % prediction bands, a dashed "현재" divider at the
    last observation, changepoint lines with numbered badges (the numbers of
    the HTML changepoint table) and date labels, anomaly rings, nice y-axis
    ticks, non-overlapping date ticks and a legend.  Top labels never overlap
    each other and are never crossed by another vertical line; a changepoint
    label that does not fit is left out (its badge still identifies the line).
    Anomalies that :meth:`_ChartData.y_extent` leaves outside the value range
    (spikes that would squash the rest of the chart) are drawn as triangles on
    the plot edge with their values.  Wide charts (``width >= 700``) also carry
    a CSS-only hover read-out per date.  The SVG uses a ``viewBox`` with
    ``width="100%"`` so it scales to its container.
    """
    width = max(240, int(width))
    height = max(160, int(height))
    d = _ChartData(report)
    freq = d.freq
    compact = width < 700
    fs = 12.0 if width >= 700 else (12.5 if width >= 500 else 13.0)
    row_h = fs + 9
    uid = _uid(report.name, width, height, "c")
    a = report.analysis
    cps = list(getattr(a, "changepoints", None) or [])
    anoms = list(getattr(a, "anomalies", None) or [])

    # ---- legend (top)
    items: list[tuple[str, str, str]] = [("line", "hist", "실제값")]
    if len(d.trend_y) and np.isfinite(d.trend_y).any():
        items.append(("dash", "trend", "추세"))
    if len(d.fc_y):
        items.append(("line", "fc", "예측"))
    band_class: dict[int, str] = {}
    n_b = len(d.bands)
    for i, (lv, _, _) in enumerate(d.bands):  # widest first; the narrowest band is the strongest
        band_class[lv] = "tai-bo" if i < n_b - 1 else "tai-bi"
    for lv, _, _ in sorted(d.bands, key=lambda b: b[0]):
        items.append(("band", str(lv), f"{lv}% 구간"))
    if cps:
        items.append(("vline", "cp", "변곡점"))
    if anoms:
        items.append(("ring", "anom", "이상치"))

    # ---- y scale
    lo, hi = d.y_extent()
    ticks = _nice_ticks(lo, hi)
    tick_labels = _axis_labels(ticks)
    ymin, ymax = ticks[0], ticks[-1]

    left = max(34.0, max(_text_width(s, fs) for s in tick_labels) + 14.0)
    end_val: Optional[float] = None
    end_t: Optional[float] = None
    end_cls = "tai-dot-f"
    if len(d.fc_y):
        end_val, end_t = float(d.fc_y[-1]), float(d.fc_t[-1])
    elif math.isfinite(d.last_y):
        end_val, end_t, end_cls = d.last_y, d.last_t, "tai-dot-h"
    end_label = _fmt_axis_value(end_val, max(abs(ymin), abs(ymax))) if end_val is not None else ""
    right = (_text_width(end_label, fs) + 16.0) if end_label else 16.0
    right = max(right, 16.0)
    legend_x0 = 8.0 if compact else left
    placed, n_rows = _layout_legend(items, legend_x0, width - legend_x0 - 8, fs, row_h)
    top = 6.0 + n_rows * row_h + 10.0
    bottom = fs + 16.0
    plot_l, plot_r = left, width - right
    plot_t, plot_b = top, height - bottom
    if plot_r - plot_l < 40:
        plot_r = plot_l + 40
    if plot_b - plot_t < 40:
        plot_b = plot_t + 40
    plot_w, plot_h = plot_r - plot_l, plot_b - plot_t

    def X(t: Any) -> Any:
        return plot_l + np.asarray(t, dtype=float) / d.t_end * plot_w

    def Y(v: Any) -> Any:
        return plot_b - _unit_pos(v, ymin, ymax) * plot_h

    def xs(t: float) -> float:
        return float(X(t))

    def ys(v: float) -> float:
        return float(Y(v))

    name = str(report.name)
    title = f"{name} — 과거·현재·미래 추세 차트"
    desc = (f"{_fmt_date(d.t0, freq)}부터 {_fmt_date(d.last_date, freq)}까지의 실제값과 "
            f"{len(d.fc_y)}{PERIOD_KO.get(freq, '기간')} 예측")
    parts: list[str] = [
        f'<svg class="tai-chart" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="100%" font-size="{fs:g}" role="img" aria-labelledby="t-{uid} d-{uid}" '
        f'preserveAspectRatio="xMidYMid meet" style="display:block;height:auto;max-width:100%">',
        f'<title id="t-{uid}">{_esc(title)}</title>',
        f'<desc id="d-{uid}">{_esc(desc)}</desc>',
        f"<style>{_SVG_STYLE}</style>",
        f'<defs><clipPath id="clip-{uid}"><rect x="{_f1(plot_l)}" y="{_f1(plot_t)}" '
        f'width="{_f1(plot_w)}" height="{_f1(plot_h)}"/></clipPath>'
        # lines get a few px of slack so a series lying on the axis limit keeps its full stroke
        f'<clipPath id="clipl-{uid}"><rect x="{_f1(plot_l - 3)}" y="{_f1(plot_t - 3)}" '
        f'width="{_f1(plot_w + 6)}" height="{_f1(plot_h + 6)}"/></clipPath></defs>',
    ]

    x_now = xs(d.last_t)
    # forecast region shading
    if len(d.fc_y) and plot_r - x_now > 0.5:
        parts.append(f'<rect class="tai-region" x="{_f1(x_now)}" y="{_f1(plot_t)}" '
                     f'width="{_f1(plot_r - x_now)}" height="{_f1(plot_h)}"/>')

    # y grid + labels
    g = []
    for tv, tl in zip(ticks, tick_labels):
        yy = ys(tv)
        cls = "tai-zero" if (tv == 0 and ymin < 0 < ymax) else "tai-grid"
        if abs(yy - plot_b) > 0.5:
            g.append(f'<path class="{cls}" d="M{_f1(plot_l)},{_f1(yy)}H{_f1(plot_r)}"/>')
        g.append(f'<text class="tai-ax" x="{_f1(plot_l - 8)}" y="{_f1(yy + fs * 0.35)}" '
                 f'text-anchor="end">{_esc(tl)}</text>')
    parts.append("<g>" + "".join(g) + "</g>")
    parts.append(f'<path class="tai-base" d="M{_f1(plot_l)},{_f1(plot_b)}H{_f1(plot_r)}"/>')

    # x ticks
    def x_of_ts(t: pd.Timestamp) -> float:
        dd = d.day_of(t)
        return xs(dd if dd is not None else 0.0)

    xt = _date_ticks(d.t0, d.end_date, freq, x_of_ts, plot_l, plot_r, fs)
    g = []
    for x, lab in xt:
        w = _text_width(lab, fs)
        anchor = "middle"
        tx = x
        if x - w / 2 < 2:
            anchor, tx = "start", max(2.0, x - 2)
        elif x + w / 2 > width - 2:
            anchor, tx = "end", min(width - 2.0, x + 2)
        g.append(f'<path class="tai-base" d="M{_f1(x)},{_f1(plot_b)}v4"/>')
        g.append(f'<text class="tai-ax" x="{_f1(tx)}" y="{_f1(plot_b + fs + 6)}" '
                 f'text-anchor="{anchor}">{_esc(lab)}</text>')
    parts.append("<g>" + "".join(g) + "</g>")

    # changepoint lines (behind the data); each carries a numbered badge keyed to the
    # changepoint table of the HTML page, so a line whose date label does not fit stays identifiable
    cp_marks: list[tuple[float, list[str]]] = []
    cp_hits: list[str] = []
    events: dict[float, list[tuple[str, str]]] = {}  # day -> extra hover rows
    for num, cp in enumerate(cps, 1):
        dd = d.day_of(getattr(cp, "date", None))
        if dd is None:
            try:
                dd = float(d.hist_t[int(cp.index)])
            except (TypeError, ValueError, IndexError, AttributeError):
                continue
        if not (0 <= dd <= d.t_end):
            continue
        x = xs(dd)
        kind_ko, _ = _cp_kind(cp)
        date_s = _fmt_date(getattr(cp, "date", None), freq)
        sb, sa = _fmt_num(getattr(cp, "slope_before", None), signed=True), _fmt_num(
            getattr(cp, "slope_after", None), signed=True)
        tip = f"변곡점 {num}: {date_s} · {kind_ko} (기울기 {sb} → {sa})"
        parts.append(f'<path class="tai-cp" d="M{_f1(x)},{_f1(plot_t)}V{_f1(plot_b)}"/>')
        badge_r = 7.0 if num < 10 else 8.0
        badge_fs = 10.0 if num < 10 else 9.0
        cp_hits.append(f'<g><title>{_esc(tip)}</title>'
                       f'<path class="tai-cphit" d="M{_f1(x)},{_f1(plot_t)}V{_f1(plot_b)}"/>'
                       f'<circle class="tai-cpb" cx="{_f1(x)}" cy="{_f1(plot_t)}" r="{badge_r:g}"/>'
                       f'<text class="tai-cpn" x="{_f1(x)}" y="{_f1(plot_t + badge_fs * 0.36)}" '
                       f'font-size="{badge_fs:g}" text-anchor="middle">{num}</text></g>')
        # label candidates, most informative first (the badge number is the last resort)
        texts = [date_s] if compact else [f"{date_s} {kind_ko}", date_s]
        if compact and freq in ("D", "W"):
            texts.append(_fmt_date(getattr(cp, "date", None), "M"))
        cp_marks.append((x, texts))
        events.setdefault(round(dd, 3), []).append(("변곡점", f"{num} · {kind_ko}"))

    # "현재" divider
    if math.isfinite(d.last_y) or len(d.hist_t):
        parts.append(f'<path class="tai-now" d="M{_f1(x_now)},{_f1(plot_t)}V{_f1(plot_b)}"/>')

    # bands + lines (clipped to the plot area)
    clip = f' clip-path="url(#clip-{uid})"'
    data_g: list[str] = []
    if len(d.fc_y):
        bx = np.concatenate([[d.last_t], d.fc_t]) if math.isfinite(d.last_y) else d.fc_t
        for lv, blo, bhi in d.bands:
            if math.isfinite(d.last_y):
                blo = np.concatenate([[d.last_y], blo])
                bhi = np.concatenate([[d.last_y], bhi])
            bc = band_class.get(lv, "tai-bi")
            data_g.append(f'<path class="tai-band {bc}" d="{_band_path(X(bx), Y(blo), Y(bhi))}">'
                          f'<title>{lv}% 예측 구간</title></path>')
    if data_g:
        parts.append(f"<g{clip}>" + "".join(data_g) + "</g>")
    clip = f' clip-path="url(#clipl-{uid})"'
    data_g = []
    if len(d.trend_y):
        data_g.append(f'<path class="tai-trend" d="{_path(*_decimate(X(d.trend_t), Y(d.trend_y)))}"/>')
    if len(d.hist_y):
        data_g.append(f'<path class="tai-hist" d="{_path(*_decimate(X(d.hist_t), Y(d.hist_y)))}"/>')
    if len(d.fc_y):
        fx = np.concatenate([[d.last_t], d.fc_t]) if math.isfinite(d.last_y) else d.fc_t
        fy = np.concatenate([[d.last_y], d.fc_y]) if math.isfinite(d.last_y) else d.fc_y
        data_g.append(f'<path class="tai-fc" d="{_path(*_decimate(X(fx), Y(fy)))}"/>')
    parts.append(f"<g{clip}>" + "".join(data_g) + "</g>")

    if cp_hits:
        parts.append("<g>" + "".join(cp_hits) + "</g>")

    # anomalies: rings inside the axis range; points outside it (left out of the range by
    # _ChartData.y_extent) become triangles on the plot edge, grouped when they are close
    g = []
    clipped: list[tuple[float, float, str]] = []  # (x, value, tooltip)
    for an in anoms:
        dd = d.day_of(getattr(an, "date", None))
        v = _finite(getattr(an, "value", None))
        if dd is None or v is None or not (0 <= dd <= d.t_end):
            continue
        events.setdefault(round(dd, 3), []).append(
            ("이상치", f"z {_fmt_num(getattr(an, 'zscore', None), 1, signed=True)}"))
        tip = (f"이상치 {_fmt_date(getattr(an, 'date', None), freq)}: 값 {_fmt_num(v)} "
               f"(예상 {_fmt_num(getattr(an, 'expected', None))}, z={_fmt_num(getattr(an, 'zscore', None), 1)})")
        if not (ymin <= v <= ymax):
            clipped.append((xs(dd), v, tip + " — 축 범위 밖"))
            continue
        g.append(f'<circle class="tai-anom" cx="{_f1(xs(dd))}" cy="{_f1(ys(v))}" r="{4.5 if not compact else 4:g}">'
                 f'<title>{_esc(tip)}</title></circle>')
    spikes: list[tuple[float, float, bool]] = []  # (x, most extreme value, above) per group
    for above in (True, False):
        side = sorted((c for c in clipped if (c[1] > ymax) == above), key=lambda c: c[0])
        groups: list[list[tuple[float, float, str]]] = []
        for c in side:
            if groups and c[0] - groups[-1][-1][0] <= 10.0:
                groups[-1].append(c)
            else:
                groups.append([c])
        for grp in groups:
            x_s, v_s, _ = max(grp, key=lambda c: c[1]) if above else min(grp, key=lambda c: c[1])
            tri = (f"M{_f1(x_s - 5)},{_f1(plot_t + 8)}L{_f1(x_s)},{_f1(plot_t)}L{_f1(x_s + 5)},{_f1(plot_t + 8)}Z"
                   if above else
                   f"M{_f1(x_s - 5)},{_f1(plot_b - 8)}L{_f1(x_s)},{_f1(plot_b)}L{_f1(x_s + 5)},{_f1(plot_b - 8)}Z")
            g.append(f'<path class="tai-spike" d="{tri}"><title>{_esc(" / ".join(c[2] for c in grp))}</title></path>')
            spikes.append((x_s, v_s, above))
    if g:
        parts.append("<g>" + "".join(g) + "</g>")

    # end-point dots + direct label of the final value
    end_box: Optional[tuple[float, float]] = None  # vertical extent of the end label (right margin)
    if math.isfinite(d.last_y) and ymin <= d.last_y <= ymax:
        parts.append(f'<circle class="tai-dot-h" cx="{_f1(x_now)}" cy="{_f1(ys(d.last_y))}" r="4"/>')
    if end_val is not None and end_t is not None and ymin <= end_val <= ymax:
        ex, ey = xs(end_t), ys(end_val)
        if end_cls == "tai-dot-f":
            parts.append(f'<circle class="tai-dot-f" cx="{_f1(ex)}" cy="{_f1(ey)}" r="4"/>')
        parts.append(f'<text class="tai-strong" x="{_f1(ex + 8)}" y="{_f1(ey + fs * 0.35)}">{_esc(end_label)}</text>')
        end_box = (ey - fs, ey + fs)

    # top labels: "현재", changepoint dates and clipped-spike values, stacked into rows.
    # A label never overlaps another label and is never crossed by a vertical line
    # (changepoint, "현재" divider or clipped spike) other than its own; a changepoint
    # label that cannot be placed is dropped -- its numbered badge still identifies it.
    max_rows = 2 if compact else 3
    lab_fs = fs - 1 if not compact else fs - 0.5
    rows: list[list[tuple[float, float]]] = [[] for _ in range(max_rows)]
    vlines = [m[0] for m in cp_marks] + [x_now] + [s[0] for s in spikes if s[2]]
    lbl_g: list[str] = []

    def row_y(r: int) -> float:
        return plot_t + lab_fs + 4 + r * (lab_fs + 6)

    def right_limit(baseline: float) -> float:
        """A label may run into the right margin unless the end-value label sits at its height."""
        if end_box is None or baseline + 4 < end_box[0] or baseline - lab_fs - 2 > end_box[1]:
            return width - 2.0
        return plot_r - 2.0

    def place(x: float, texts: Sequence[str], offset: float, anchors: Sequence[str] = ("start", "end"),
              avoid_lines: bool = True) -> Optional[tuple[int, str, str, float]]:
        """First (text, row, anchor) whose box fits; returns ``(row, text, anchor, text_x)``."""
        # the preferred side first (any text, any row), so labels sit consistently on one side of their line
        for anchor in anchors:
            for text in texts:
                w = _text_width(text, lab_fs)
                for r in range(max_rows):
                    margin = 9.0 if r == 0 else 4.0  # row 0 also clears the badges on the top edge
                    x0 = x + offset if anchor == "start" else x - offset - w
                    x1 = x0 + w
                    if x0 < plot_l + 2 or x1 > right_limit(row_y(r)):
                        continue
                    if any(not (x1 + 6 < a0 or x0 > a1 + 6) for a0, a1 in rows[r]):
                        continue
                    if avoid_lines and any(x0 - margin < vx < x1 + margin for vx in vlines if abs(vx - x) > 0.5):
                        continue
                    rows[r].append((x0, x1))
                    return r, text, anchor, (x0 if anchor == "start" else x1)
        return None

    if math.isfinite(d.last_y) or len(d.hist_t):
        # "현재" sits just left of the divider at the top (right of it when there is no room);
        # boxed in by changepoint lines there, it moves to the bottom of the divider, and only
        # as a last resort is it drawn across a line -- it is too important to drop
        got = place(x_now, ["현재"], 5.0, ("end", "start"))
        now_y = row_y(got[0]) if got is not None else None
        if got is None:
            w_now = _text_width("현재", lab_fs)
            for anchor_now in ("end", "start"):
                x0 = x_now - 5 - w_now if anchor_now == "end" else x_now + 5
                if (x0 >= plot_l + 2 and x0 + w_now <= right_limit(plot_b - 6)
                        and not any(x0 - 4 < vx < x0 + w_now + 4 for vx in vlines if abs(vx - x_now) > 0.5)):
                    got = (0, "현재", anchor_now, x0 if anchor_now == "start" else x0 + w_now)
                    now_y = plot_b - 6
                    break
        if got is None:
            got = place(x_now, ["현재"], 5.0, ("end", "start"), avoid_lines=False)
            now_y = row_y(got[0]) if got is not None else None
        if got is not None and now_y is not None:
            _, _, anchor_now, tx = got
            lbl_g.append(f'<text class="tai-lbl tai-strong" x="{_f1(tx)}" y="{_f1(now_y)}" '
                         f'font-size="{lab_fs:g}" text-anchor="{anchor_now}">현재</text>')
    for x, texts in sorted(cp_marks, key=lambda m: m[0]):
        got = place(x, texts, 10.0)
        if got is None:
            continue
        r, text, anchor, tx = got
        lbl_g.append(f'<text class="tai-lbl" x="{_f1(tx)}" y="{_f1(row_y(r))}" '
                     f'font-size="{lab_fs:g}" text-anchor="{anchor}">{_esc(text)}</text>')
    ref_v = max(abs(ymin), abs(ymax))
    for x, v, above in sorted(spikes, key=lambda s: -abs(s[1])):
        txt = _fmt_axis_value(v, max(abs(v), ref_v))
        if above:
            got = place(x, [txt], 8.0)
            if got is None:
                continue
            r, _, anchor, tx = got
            ty = row_y(r)
        else:
            w = _text_width(txt, lab_fs)
            anchor, tx = ("start", x + 8) if x + 8 + w <= plot_r - 2 else ("end", x - 8)
            ty = plot_b - 4
        lbl_g.append(f'<text class="tai-lbl tai-spkl" x="{_f1(tx)}" y="{_f1(ty)}" '
                     f'font-size="{lab_fs:g}" text-anchor="{anchor}">{_esc(txt)}</text>')
    parts.append("<g>" + "".join(lbl_g) + "</g>")

    # legend
    parts.append(_legend_svg(placed, 6.0, fs, row_h, band_class))

    # hover layer (wide charts only, painted last so read-outs sit above every label)
    if not compact:
        parts.append(_hover_layer(d, X, Y, plot_l, plot_r, plot_t, plot_b, fs,
                                  max(abs(ymin), abs(ymax)), events))
    parts.append("</svg>")
    return "".join(parts)


def _hover_layer(d: _ChartData, X, Y, plot_l: float, plot_r: float, plot_t: float, plot_b: float,
                 fs: float, ref: float, events: Optional[dict[float, list[tuple[str, str]]]] = None,
                 max_cols: int = 260) -> str:
    """Pure-CSS hover read-out: invisible column per date showing crosshair + values.

    ``events`` maps a day offset to extra rows (changepoint / anomaly notes).
    """
    events = events or {}
    # (day, box heading, rows, value, dot class, plain date for the native <title>)
    pts: list[tuple[float, str, list[tuple[str, str]], float, str, str]] = []
    freq = d.freq
    for i in range(len(d.hist_t)):
        v = float(d.hist_y[i])
        if not (math.isfinite(v) and math.isfinite(d.hist_t[i])):
            continue
        rows = [("실제", _fmt_axis_value(v, ref))]
        tv = d.trend_map.get(float(d.hist_t[i]))
        if tv is not None and math.isfinite(tv):
            rows.append(("추세", _fmt_axis_value(tv, ref)))
        rows.extend(events.get(round(float(d.hist_t[i]), 3), []))
        date_s = _fmt_date(d.hist_dates[i], freq)
        pts.append((float(d.hist_t[i]), date_s, rows, v, "tai-dot-h", date_s))
    for i in range(len(d.fc_t)):
        v = float(d.fc_y[i])
        rows = [("예측", _fmt_axis_value(v, ref))]
        for lv, blo, bhi in sorted(d.bands, key=lambda b: b[0]):
            rows.append((f"{lv}%", f"{_fmt_axis_value(blo[i], ref)} ~ {_fmt_axis_value(bhi[i], ref)}"))
        date_s = _fmt_date(d.fc_dates[i], freq)
        pts.append((float(d.fc_t[i]), date_s + " · 예측", rows, v, "tai-dot-f", date_s))
    if not pts:
        return ""
    pts.sort(key=lambda p: p[0])
    if len(pts) > max_cols:
        keep = np.unique(np.linspace(0, len(pts) - 1, max_cols).round().astype(int))
        pts = [pts[k] for k in keep]
    xs = [float(X(p[0])) for p in pts]
    out = ['<g class="tai-hover">']
    line_h = fs + 5
    for k, (_t, date_s, rows, v, dot_cls, plain_date) in enumerate(pts):
        x = xs[k]
        x0 = plot_l if k == 0 else (xs[k - 1] + x) / 2
        x1 = plot_r if k == len(pts) - 1 else (x + xs[k + 1]) / 2
        label_w = max(_text_width(lab, fs) for lab, _ in rows)
        val_w = max(_text_width(val, fs) for _, val in rows)
        bw = max(_text_width(date_s, fs), label_w + 10 + val_w) + 20
        bh = (len(rows) + 1) * line_h + 10
        bx = x + 12 if x + 12 + bw <= plot_r else x - 12 - bw
        yv = float(Y(v))
        # keep the box away from the hovered point: top half -> box at the bottom
        by = plot_b - bh - 4 if yv < plot_t + (plot_b - plot_t) * 0.45 else plot_t + 4
        # the native tooltip (also what assistive tech reads) names the forecast once, through
        # its "예측" row: "2026-01 예측 70.9, 80% 66.0 ~ 75.8" (not "2026-01 · 예측 예측 70.9")
        tip_txt = plain_date + " " + ", ".join(f"{a} {b}" for a, b in rows)
        seg = [f'<g class="tai-hv"><rect class="tai-hit" x="{_f1(x0)}" y="{_f1(plot_t)}" '
               f'width="{_f1(max(x1 - x0, 1.0))}" height="{_f1(plot_b - plot_t)}"><title>{_esc(tip_txt)}</title></rect>'
               f'<g class="tai-hs"><path class="tai-xh" d="M{_f1(x)},{_f1(plot_t)}V{_f1(plot_b)}"/>']
        if plot_t <= yv <= plot_b:
            seg.append(f'<circle class="{dot_cls}" cx="{_f1(x)}" cy="{_f1(yv)}" r="4.5"/>')
        seg.append(f'<rect class="tai-tip" x="{_f1(bx)}" y="{_f1(by)}" width="{_f1(bw)}" height="{_f1(bh)}" rx="6"/>')
        ty = by + 6 + fs
        seg.append(f'<text class="tai-tipt" x="{_f1(bx + 10)}" y="{_f1(ty)}" font-weight="600">{_esc(date_s)}</text>')
        for lab, val in rows:
            ty += line_h
            seg.append(f'<text x="{_f1(bx + 10)}" y="{_f1(ty)}">{_esc(lab)}</text>'
                       f'<text class="tai-tipt" x="{_f1(bx + bw - 10)}" y="{_f1(ty)}" text-anchor="end">{_esc(val)}</text>')
        seg.append("</g></g>")
        out.append("".join(seg))
    out.append("</g>")
    return "".join(out)


def _mini_chart(report: TrendReport, width: int = 320, height: int = 112) -> str:
    """Small-multiple SVG (history + forecast with its narrowest band), no axes."""
    d = _ChartData(report)
    uid = _uid(report.name, width, height, "m")
    lo, hi = d.y_extent()
    pad_x, pad_t, pad_b = 4.0, 8.0, 8.0
    pl, pr, pt, pb = pad_x, width - pad_x - 4, pad_t, height - pad_b
    top = hi if hi > lo else lo + 1.0

    def X(t: Any) -> Any:
        return pl + np.asarray(t, dtype=float) / d.t_end * (pr - pl)

    def Y(v: Any) -> Any:
        return pb - _unit_pos(v, lo, top) * (pb - pt)

    parts = [
        f'<svg class="tai-chart tai-mini" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="100%" role="img" aria-labelledby="t-{uid}" style="display:block;height:auto">',
        f'<title id="t-{uid}">{_esc(str(report.name) + " — 추세와 예측")}</title>',
        f'<defs><clipPath id="clip-{uid}"><rect x="0" y="0" width="{width}" height="{height}"/></clipPath></defs>',
    ]
    x_now = float(X(d.last_t))
    if len(d.fc_y):
        parts.append(f'<rect class="tai-region" x="{_f1(x_now)}" y="0" width="{_f1(width - x_now)}" '
                     f'height="{height}" rx="0"/>')
    if lo < 0 < hi:
        parts.append(f'<path class="tai-zero" d="M{_f1(pl)},{_f1(float(Y(0.0)))}H{_f1(pr)}"/>')
    g = []
    if len(d.fc_y) and d.bands:
        _, blo, bhi = d.bands[-1]  # narrowest band
        bx, blo2, bhi2 = d.fc_t, blo, bhi
        if math.isfinite(d.last_y):
            bx = np.concatenate([[d.last_t], d.fc_t])
            blo2 = np.concatenate([[d.last_y], blo])
            bhi2 = np.concatenate([[d.last_y], bhi])
        g.append(f'<path class="tai-band tai-bi" d="{_band_path(X(bx), Y(blo2), Y(bhi2))}"/>')
    if len(d.hist_y):
        g.append(f'<path class="tai-hist" d="{_path(*_decimate(X(d.hist_t), Y(d.hist_y)))}"/>')
    if len(d.fc_y):
        fx = np.concatenate([[d.last_t], d.fc_t]) if math.isfinite(d.last_y) else d.fc_t
        fy = np.concatenate([[d.last_y], d.fc_y]) if math.isfinite(d.last_y) else d.fc_y
        g.append(f'<path class="tai-fc" d="{_path(*_decimate(X(fx), Y(fy)))}"/>')
    parts.append(f'<g clip-path="url(#clip-{uid})">' + "".join(g) + "</g>")
    parts.append(f'<path class="tai-now" d="M{_f1(x_now)},0V{height}"/>')
    if math.isfinite(d.last_y):
        parts.append(f'<circle class="tai-dot-h" cx="{_f1(x_now)}" cy="{_f1(float(Y(d.last_y)))}" r="3.5"/>')
    if len(d.fc_y):
        parts.append(f'<circle class="tai-dot-f" cx="{_f1(float(X(d.fc_t[-1])))}" '
                     f'cy="{_f1(float(Y(d.fc_y[-1])))}" r="3.5"/>')
    parts.append("</svg>")
    return "".join(parts)


# --------------------------------------------------------------------------- terminal text


def _char_width(ch: str) -> int:
    if unicodedata.combining(ch):
        return 0
    return 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1


def _dw(s: str) -> int:
    """Display width of ``s`` in terminal columns (East Asian wide chars count 2)."""
    return sum(_char_width(c) for c in s)


def _fit(s: str, width: int) -> str:
    """Truncate ``s`` to ``width`` display columns (with an ellipsis)."""
    if _dw(s) <= width:
        return s
    out, w = [], 0
    for ch in s:
        cw = _char_width(ch)
        if w + cw > width - 1:
            break
        out.append(ch)
        w += cw
    return "".join(out) + "…"


def _term(s: Any) -> str:
    """Terminal-safe text: drop control characters, neutralise angle brackets."""
    txt = "" if s is None else str(s)
    txt = "".join(ch for ch in txt if unicodedata.category(ch)[0] != "C" or ch == " ")
    return txt.replace("<", "‹").replace(">", "›")


def _wrap_segments(segments: list[str], width: int, sep: str = " · ", indent: str = "") -> list[str]:
    lines: list[str] = []
    cur = ""
    for seg in segments:
        cand = seg if not cur else cur + sep + seg
        if cur and _dw(indent + cand) > width:
            lines.append(indent + cur)
            cur = seg
        else:
            cur = cand
    if cur:
        lines.append(indent + cur)
    return [_fit(ln, width) for ln in lines]


#: labels of the terminal summary per language (render_text); the narrative supplies the phase names,
#: direction words, changepoint labels, method description and scenario floor text in the same language
_TEXT: dict[str, dict[str, str]] = {
    "ko": {
        "conf": "신뢰도 {p}%", "tentative": "잠정", "obs": "{unit} 단위 {n:,}개", "range": "{a} ~ {b}",
        "current": "현재 값", "total": "전체 변화", "cagr": "CAGR", "recent": "최근 변화",
        "momentum": "모멘텀", "mom_unit": "%p", "volatility": "변동성", "why_sep": "·",
        "outlook": "전망", "change": "예측 변화", "change_basis": "{fut}, {cur} 대비",
        "prob": "상승 확률 {p} (같은 기준)", "confidence": "신뢰도 {c}",
        "scen": "시나리오", "scen_at": "시나리오({end} 시점)", "scen_ref_season": "(비교: {date} 실제 {v})",
        "scen_ref_now": "(비교: 현재 {v})", "future_phase": "미래 단계", "peak": "예측 정점",
        "cps": "변곡점: ", "cps_more": "외 {n}개", "method": "방법: ", "accuracy": "정확도: ",
        "skill": _SKILL_LABEL, "top": "상위 모델({metric}):",
    },
    "en": {
        "conf": "confidence {p}%", "tentative": "tentative", "obs": "{n:,} {unit} obs", "range": "{a} to {b}",
        "current": "Current", "total": "Total change", "cagr": "CAGR", "recent": "Recent change",
        "momentum": "Momentum", "mom_unit": " pp", "volatility": "Volatility", "why_sep": ", ",
        "outlook": "Outlook", "change": "Exp. change", "change_basis": "{fut} vs {cur}",
        "prob": "P(up) {p} (same basis)", "confidence": "Confidence {c}",
        "scen": "Scenarios", "scen_at": "Scenarios (at {end})", "scen_ref_season": "(vs. {date} actual {v})",
        "scen_ref_now": "(vs. now {v})", "future_phase": "Future phase", "peak": "Forecast peak",
        "cps": "Changepoints: ", "cps_more": "+{n} more", "method": "Method: ", "accuracy": "Accuracy: ",
        "skill": "error vs naive", "top": "Top models ({metric}):",
    },
}
_CONFIDENCE_EN: dict[str, str] = {"high": "high", "medium": "medium", "low": "low"}
_SKILL_VERDICT_EN: dict[str, str] = {
    "단순 예측보다 정확": "better than naive", "단순 예측과 비슷": "about the same as naive",
    "단순 예측보다 부정확": "less accurate than naive", "단순 예측보다 낫지 않음": "not better than naive",
}


def _wrap_method(text: str, width: int) -> list[str]:
    """The method line, wrapped at its `` · `` separators (a trailing ``·`` marks the break; continuation
    lines indented by two), or -- when a part is still too wide, e.g. the English fallback
    description -- at spaces, so the description is never cut off."""
    segs = text.split(" · ")
    if all(_dw(seg) + 2 <= width - 2 for seg in segs):
        mlines = _wrap_segments(segs, width - 2)
        return ([ln + " ·" if len(mlines) > 1 else ln for ln in mlines[:1]]
                + [_fit("  " + ln + (" ·" if i < len(mlines) - 2 else ""), width)
                   for i, ln in enumerate(mlines[1:])])
    out: list[str] = []
    cur = ""
    for word in text.split(" "):
        cand = word if not cur else cur + " " + word
        if cur and _dw(("  " if out else "") + cand) > width:
            out.append(("  " if out else "") + cur)
            cur = word
        else:
            cur = cand
    if cur:
        out.append(("  " if out else "") + cur)
    return [_fit(ln, width) for ln in out]


def render_text(report: TrendReport, width: int = 78, lang: str = "ko") -> str:
    """Compact terminal summary of ``report`` (every line fits in ``width`` columns).

    Shows the name, lifecycle phase, key metrics, a sparkline of the history
    and the forecast separated by ``│``, the outlook and the top models.
    Widths below 10 are treated as 10.  ``lang='en'`` gives the same summary in
    English (no Hangul apart from what the data itself contains, e.g. the series
    name), with the narrative's English phase names, direction words,
    changepoint labels and method description; any other value gives Korean.
    """
    lang = "en" if str(lang).lower().startswith("en") else "ko"
    tx = _TEXT[lang]
    en = lang == "en"
    width = max(10, int(width))
    a, fc, o = report.analysis, report.forecast, report.outlook
    m = a.metrics
    freq = str(a.freq)
    d = _ChartData(report)
    lines: list[str] = []
    heavy, light = "═" * width, "─" * width

    lines.append(heavy)
    lines.append(_fit(_term(report.name), width))
    phase = a.phase
    conf = _finite(getattr(phase, "confidence", None))
    # a phase called on too few observations is a tentative call, as the narrative says
    ph = _term(_phase_name(phase.label, lang)) + ((", " if en else "·") + tx["tentative"] if _tentative(report)
                                                  else "")
    unit = _FREQ_EN.get(freq, freq) if en else UNIT_KO.get(freq, FREQ_KO.get(freq, freq))
    head = [f"[{ph}]" + (" " + tx["conf"].format(p=f"{conf * 100:.0f}") if conf is not None else ""),
            tx["obs"].format(unit=unit, n=len(d.hist_y)),
            tx["range"].format(a=_fmt_date(m.start, freq), b=_fmt_date(m.end, freq))]
    lines += _wrap_segments(head, width)
    lines.append(light)

    vol = _finite(m.volatility)
    metrics = [
        f"{tx['current']} {_fmt_num(m.last_value)}",
        f"{tx['total']} {_fmt_pct(m.total_change_pct)}",
        f"{tx['cagr']} {_fmt_pct(m.cagr_pct)}",
        f"{tx['recent']} {_fmt_pct(m.recent_change_pct)}",
        f"{tx['momentum']} {_fmt_num(m.momentum, 2, signed=True)}"
        + (f"{tx['mom_unit']}{_period_unit(freq, lang) or _PER_UNIT.get(freq, '')}"
           if _finite(m.momentum) is not None else ""),
        f"{tx['volatility']} {_fmt_pct(vol * 100 if vol is not None else None, signed=False)}",
    ]
    lines += _wrap_segments(metrics, width)
    # a "—" is never left unexplained: why the growth rates are missing (crossed zero, intermittent,
    # a CAGR over less than a year ...)
    why: dict[str, list[str]] = {}
    for key, label in (("total_change_pct", tx["total"]), ("cagr_pct", tx["cagr"]),
                       ("recent_change_pct", tx["recent"])):
        text = _pct_why(_metric_undefined(report, d, key), key, lang)
        if text:
            why.setdefault(text, []).append(label)
    for text, labels in why.items():
        lines += _wrap_segments([f"※ {tx['why_sep'].join(labels)}: {text}"], width)
    lines.append(light)

    # sparkline: history │ forecast on a shared scale
    hist = d.hist_y[np.isfinite(d.hist_y)]
    fut = d.fc_y[np.isfinite(d.fc_y)]
    if hist.size:
        avail = width - (1 if fut.size else 0)
        n, h = hist.size, fut.size
        if n + h <= avail:
            hw, fw = n, h
        else:
            fw = int(round(avail * h / (n + h))) if h else 0
            fw = min(h, max(fw, min(h, 8)), avail - 1)  # leave >= 1 column for the history
            hw = min(max(1, avail - fw), n)
        both = np.concatenate([hist, fut]) if fut.size else hist
        lo, hi = float(both.min()), float(both.max())
        sp = _spark_chars(_resample(hist, hw), lo, hi)
        if fut.size and fw > 0:
            sp += "│" + _spark_chars(_resample(fut, fw), lo, hi)
        lines.append(sp)
        # date labels under the sparkline
        canvas = [" "] * width
        def put(pos: int, text: str) -> bool:
            if pos < 0 or pos + len(text) > width or any(c != " " for c in canvas[max(0, pos - 1):pos + len(text) + 1]):
                return False
            canvas[pos:pos + len(text)] = list(text)
            return True
        s_lab = _fmt_date(d.t0, freq)
        put(0, s_lab)
        if fut.size and fw > 0:
            e_lab = _fmt_date(d.fc_dates[-1], freq)
            put(hw + 1 + fw - len(e_lab), e_lab)
            now_lab = _fmt_date(d.last_date, freq)
            put(max(0, hw - len(now_lab)), now_lab)
        else:
            e_lab = _fmt_date(d.last_date, freq)
            put(hw - len(e_lab), e_lab)
        lines.append("".join(canvas).rstrip())
        lines.append(light)

    # outlook
    direction = str(getattr(o, "direction", "flat"))
    fut_basis, cur_basis = _change_basis_en(report, d) if en else _change_basis(report, d)
    # the narrative's direction word (방향 불확실 for a 'flat' call on a visible change)
    word = _term(_direction_word(report, d, lang))
    nochange = _change_undefined(report, d, lang)
    if nochange is None:
        chg_seg = [f"{tx['change']} {_fmt_pct(o.expected_change_pct)} "
                   f"({tx['change_basis'].format(fut=fut_basis, cur=cur_basis)})"]
    else:  # no percentage: say why and give the two levels instead
        chg_seg = [f"{tx['change']} {_DASH} ({nochange.short})"] + ([nochange.levels] if nochange.levels else [])
    conf_word = (_CONFIDENCE_EN if en else CONFIDENCE_KO).get(str(o.confidence), _term(o.confidence))
    out_seg = [f"{tx['outlook']} {DIRECTION_ARROW.get(direction, '')} {word}".replace("  ", " "),
               *chg_seg,
               tx["prob"].format(p=_fmt_prob(o.prob_increase)),
               tx["confidence"].format(c=conf_word)]
    lines += _wrap_segments(out_seg, width)
    scen = o.scenarios or {}
    if scen:
        # single horizon-end values: compared with the observation at the same point of the season
        end_s = _fmt_date(d.fc_dates[-1], freq) if len(d.fc_dates) else ""
        names = (("pessimistic", "pessimistic"), ("base", "base"), ("optimistic", "optimistic")) if en \
            else SCENARIOS_KO
        ref = _scenario_ref(report, d)
        # one notation for the whole line (as the narrative: 4.61×10¹¹ next to 1.51×10¹⁵, not 461,…)
        big = max([abs(v) for v in (_finite(x) for x in list(scen.values()) + [ref[0] if ref else None])
                   if v is not None] + [0.0])
        sc = [f"{nm} {_scen_text(report, scen.get(k), lang=lang, scale=big)}" for k, nm in names if k in scen]
        if ref is not None:
            rv = _format_number(ref[0], scale=big)
            sc.append(tx["scen_ref_season"].format(date=_fmt_date(ref[1], freq), v=rv) if ref[2]
                      else tx["scen_ref_now"].format(v=rv))
        lines += _wrap_segments([tx["scen_at"].format(end=end_s) if end_s else tx["scen"]] + sc, width, sep="  ")
    # the projected phase only as far as the narrative projects it (판단 보류 / 참고용 / 잠정)
    ph_val, ph_note = _future_phase_parts(report, lang)
    ph_txt = _term(ph_val) + ((" (" if en else "(") + ph_note + ")" if ph_note else "")
    extra = [f"{tx['future_phase']} {ph_txt}"]
    if _ts(o.forecast_peak_date) is not None:
        extra.append(f"{tx['peak']} {_fmt_date(o.forecast_peak_date, freq)}")
    lines += _wrap_segments(extra, width)
    cps = list(a.changepoints or [])
    if cps:
        segs = [f"{_fmt_date(cp.date, freq)} "
                f"{_term(_changepoint_label(cp, 'en') if en else _cp_kind(cp)[0])}" for cp in cps[:4]]
        if len(cps) > 4:
            segs.append(tx["cps_more"].format(n=len(cps) - 4))
        segs[0] = tx["cps"] + segs[0]
        lines += _wrap_segments(segs, width)
    lines.append(light)
    # the method in words (narrative.describe_method), wrapped at its " · " separators
    method = _term(_describe_method(fc, lang))
    bm = _finite(fc.backtest_mase)
    lines += _wrap_method(f"{tx['method']}{method}", width)
    skill = _skill_value(fc)
    acc = [f"{tx['accuracy']}{tx['skill']} {_rel_error_text(skill)}"]
    if skill is not None:
        verdict = _skill_verdict(skill)
        acc.append(_SKILL_VERDICT_EN.get(verdict, verdict) if en else verdict)
    elif bm is None:   # the "—" is explained, as in the narrative's accuracy line
        acc.append("too short to backtest" if en else "데이터가 짧아 백테스트하지 못함")
    else:
        acc.append("no naive benchmark to compare" if en else "단순 기준 예측과 비교한 결과 없음")
    if bm is not None:
        acc.append(f"MASE {bm:.2f}")
    lines += _wrap_segments(acc, width)
    all_scores = list(fc.model_scores or [])
    if _intermittent(report):
        # intermittent demand: the candidates were ranked by RMSE (see _intermittent)
        ranked = sorted(all_scores, key=lambda s: (_finite(s.rmse) is None, _finite(s.rmse) or 0.0))[:3]
        segs = [f"{_term(s.name)} {_fmt_num(s.rmse, 2)}" for s in ranked]
        lines += _wrap_segments([tx["top"].format(metric="RMSE")] + segs, width, sep="  ")
    elif all_scores:
        segs = [f"{_term(s.name)} {_fmt_num(s.mase, 2)}" for s in all_scores[:3]]
        lines += _wrap_segments([tx["top"].format(metric="MASE")] + segs, width, sep="  ")
    lines.append(heavy)
    return "\n".join(_fit(ln, width) for ln in lines)


# --------------------------------------------------------------------------- HTML helpers

_LIGHT: dict[str, str] = {
    "bg": "#f6f6f3", "surface": "#ffffff", "surface-2": "#f2f2ee", "track": "#ebeae4",
    "border": "rgba(11,11,11,.08)", "border-strong": "rgba(11,11,11,.16)",
    "ink": "#141413", "ink-2": "#4e4d49", "ink-3": "#6b6963",
    "grid": "#ebeae5", "axis": "#c3c2b7", "accent": "#2a78d6", "accent-soft": "rgba(42,120,214,.10)",
    "c-hist": "#2a78d6", "c-trend": "#7a7871", "c-fc": "#eb6834", "c-cp": "#1baf7a", "c-anom": "#e34948",
    "fc-region": "rgba(235,104,52,.045)", "band-o": ".11", "band-i": ".2",
    "pos": "#0b7a33", "neg": "#c0362c", "pos-soft": "rgba(12,163,12,.10)", "neg-soft": "rgba(208,59,59,.10)",
    "shadow": "0 1px 2px rgba(20,20,19,.04), 0 2px 8px rgba(20,20,19,.04)",
}
_DARK: dict[str, str] = {
    "bg": "#0e0e0d", "surface": "#1a1a19", "surface-2": "#242422", "track": "#2e2e2b",
    "border": "rgba(255,255,255,.08)", "border-strong": "rgba(255,255,255,.16)",
    "ink": "#f0efec", "ink-2": "#c3c2b7", "ink-3": "#9c9a92",
    "grid": "#2a2a28", "axis": "#46463f", "accent": "#5a9cea", "accent-soft": "rgba(57,135,229,.16)",
    "c-hist": "#3987e5", "c-trend": "#a3a199", "c-fc": "#e2683a", "c-cp": "#199e70", "c-anom": "#e66767",
    "fc-region": "rgba(226,104,58,.07)", "band-o": ".2", "band-i": ".34",
    "pos": "#45c46c", "neg": "#f07a7a", "pos-soft": "rgba(12,163,12,.16)", "neg-soft": "rgba(230,103,103,.16)",
    "shadow": "none",
}

_FONT_STACK = ('"Pretendard", "Pretendard Variable", "Noto Sans KR", -apple-system, BlinkMacSystemFont, '
               '"Apple SD Gothic Neo", "Malgun Gothic", "Segoe UI", system-ui, sans-serif')
_FONT_LINK = ('<link rel="preconnect" href="https://fonts.googleapis.com">'
              '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
              '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;500;600;700&amp;display=swap">')


def _tokens(d: dict[str, str]) -> str:
    return "".join(f"--{k}:{v};" for k, v in d.items())


_BASE_CSS = """
*,*::before,*::after{box-sizing:border-box}
html{-webkit-text-size-adjust:100%;text-size-adjust:100%}
body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--font);font-size:15px;line-height:1.6;
-webkit-font-smoothing:antialiased;-moz-osx-font-smoothing:grayscale;word-break:keep-all;overflow-wrap:break-word}
.nw{white-space:nowrap}
.wrap{max-width:1100px;margin:0 auto;padding:40px 24px 56px}
h1,h2,h3,h4{color:var(--ink);letter-spacing:-.01em}
.eyebrow{display:flex;align-items:center;gap:8px;font-size:13px;font-weight:600;color:var(--accent);margin:0 0 10px}
.eyebrow .logo{width:18px;height:18px;border-radius:5px;background:var(--accent);display:inline-block;position:relative}
.eyebrow .logo::after{content:"";position:absolute;left:4px;right:4px;bottom:5px;height:7px;border-left:2px solid #fff;
border-bottom:2px solid #fff;transform:skewY(-28deg)}
h1{font-size:clamp(24px,3.6vw,34px);line-height:1.25;margin:0 0 14px;font-weight:700;overflow-wrap:anywhere}
.meta{display:flex;flex-wrap:wrap;gap:8px;align-items:center}
.chip{display:inline-flex;align-items:center;gap:6px;padding:4px 11px;border-radius:999px;background:var(--surface);
border:1px solid var(--border);font-size:13px;color:var(--ink-2);white-space:nowrap}
.badge{display:inline-flex;align-items:center;gap:7px;padding:4px 12px;border-radius:999px;font-size:13px;font-weight:600;
color:var(--ink);white-space:nowrap}
.badge .dot{width:8px;height:8px;border-radius:50%;flex:none}
.badge small{font-weight:500;color:var(--ink-2);font-size:12px}
.gen{font-size:12.5px;color:var(--ink-3);margin:12px 0 0}
.hero{margin:0 0 24px}
.kpis{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:12px;margin:0 0 20px}
@media (max-width:1000px){.kpis{grid-template-columns:repeat(3,minmax(0,1fr))}}
.kpi{background:var(--surface);border:1px solid var(--border);border-radius:14px;padding:14px 16px;box-shadow:var(--shadow);min-width:0}
.kpi .lbl{font-size:13px;color:var(--ink-2);font-weight:500;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.kpi .val{font-size:24px;font-weight:700;margin-top:4px;line-height:1.25;white-space:nowrap;letter-spacing:-.015em;
overflow:hidden;text-overflow:ellipsis}
.kpi .sub{font-size:12px;color:var(--ink-3);margin-top:4px;line-height:1.45}
.pos{color:var(--pos)}.neg{color:var(--neg)}
.arrow{font-size:.55em;vertical-align:.2em;margin-right:4px}
.meter{height:6px;border-radius:999px;background:var(--track);margin-top:9px;overflow:hidden}
.meter>span{display:block;height:100%;background:var(--accent);border-radius:999px}
.card{background:var(--surface);border:1px solid var(--border);border-radius:16px;padding:22px 24px;margin:0 0 20px;
box-shadow:var(--shadow);min-width:0}
.card h2{font-size:17px;margin:0 0 2px;font-weight:700;display:flex;align-items:center;gap:10px;flex-wrap:wrap}
.card h3.sub{font-size:14.5px;margin:0 0 10px;font-weight:700}
.desc{font-size:13px;color:var(--ink-3);margin:0 0 16px}
.chart-v{display:none}
@media (min-width:860px){.chart-wide{display:block}}
@media (min-width:520px) and (max-width:859.98px){.chart-mid{display:block}}
@media (max-width:519.98px){.chart-narrow{display:block}}
.grid-2{display:grid;grid-template-columns:minmax(0,1.55fr) minmax(0,1fr);gap:20px;margin:0 0 20px;align-items:start}
.grid-2>.card,.grid-2>.stack>.card{margin:0}
.stack{display:grid;gap:20px;min-width:0}
.grid-even{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:28px}
@media (max-width:860px){.grid-2,.grid-even{grid-template-columns:minmax(0,1fr)}}
.narr h3{font-size:15px;margin:20px 0 6px;font-weight:700}.narr h3:first-child,.narr h4:first-child{margin-top:0}
.narr h4{font-size:14px;margin:16px 0 4px}
.narr ul,.narr ol{margin:0 0 10px;padding-left:20px}.narr ol{padding-left:24px}.narr li{margin:4px 0}.narr p{margin:0 0 10px}
.narr li>ul,.narr li>ol{margin:4px 0 2px}.narr li ul{list-style:circle}.narr p+ul,.narr p+ol{margin-top:-4px}
.narr strong{font-weight:700;color:var(--ink)}.narr em{font-style:italic}
.src{display:inline-flex;align-items:center;gap:6px;font-size:12px;font-weight:600;padding:2px 9px;border-radius:999px;
background:var(--surface-2);color:var(--ink-2);border:1px solid var(--border)}
.src.llm{background:var(--accent-soft);color:var(--accent);border-color:transparent}
table{width:100%;border-collapse:collapse;font-size:14px}
th{font-size:12px;font-weight:600;color:var(--ink-3);text-align:left;padding:8px 10px;border-bottom:1px solid var(--border-strong);
white-space:nowrap;background:var(--surface)}
td{padding:9px 10px;border-bottom:1px solid var(--border);vertical-align:middle}
tbody tr:last-child td{border-bottom:none}
th.num,td.num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
.table-wrap{overflow-x:auto;-webkit-overflow-scrolling:touch}
.table-wrap.tall{max-height:520px;overflow-y:auto}
.table-wrap.tall th{position:sticky;top:0;z-index:1}
.muted{color:var(--ink-3)}
.empty{font-size:14px;color:var(--ink-3);padding:14px 0 4px;margin:0}
.wbar{display:flex;align-items:center;gap:8px;min-width:120px}
.wbar .track{flex:1;height:6px;background:var(--track);border-radius:999px;overflow:hidden;position:relative}
.wbar .fill{height:100%;background:var(--accent);border-radius:999px}
.wbar .pct{min-width:42px;text-align:right;font-variant-numeric:tabular-nums;font-size:13px;color:var(--ink-2)}
.wbar .mid{position:absolute;left:50%;top:-2px;bottom:-2px;width:1px;background:var(--ink-3);opacity:.5}
.kind{display:inline-flex;align-items:center;gap:6px;font-size:12.5px;font-weight:600;white-space:nowrap}
.kind .dot{width:7px;height:7px;border-radius:50%}
.mname{font-weight:600;overflow-wrap:break-word}.mdesc{display:block;font-size:12px;color:var(--ink-3);font-weight:400}
.sname{font-weight:600;overflow-wrap:anywhere}
.cpn{display:inline-flex;align-items:center;justify-content:center;min-width:18px;height:18px;padding:0 5px;
border-radius:999px;background:var(--c-cp);color:#fff;font-size:11px;font-weight:700;margin-right:8px;vertical-align:1px}
th abbr,.facts dt abbr{text-decoration:underline dotted;text-underline-offset:3px;cursor:help}
td .unit{font-size:11.5px;color:var(--ink-3);font-weight:400;margin-left:1px}
tr.inens td:first-child{box-shadow:inset 3px 0 0 var(--accent)}
code{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:12.5px;background:var(--surface-2);
padding:2px 6px;border-radius:6px;overflow-wrap:anywhere}
.facts{display:grid;grid-template-columns:auto 1fr;gap:8px 16px;margin:0;font-size:14px}
.facts dt{color:var(--ink-3)}.facts dd{margin:0;text-align:right;font-weight:600;font-variant-numeric:tabular-nums}
.facts .dnote{display:block;font-size:12px;font-weight:400;color:var(--ink-3)}
.dir{display:flex;flex-wrap:wrap;align-items:baseline;gap:4px 10px;margin:2px 0 14px}
.dir .big{font-size:26px;font-weight:700;letter-spacing:-.02em;white-space:nowrap}
.dir .lab{font-size:14px;color:var(--ink-2)}
.scen{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px;margin:0 0 12px}
.scen>div{background:var(--surface-2);border-radius:10px;padding:10px 12px;min-width:0}
.scen .k{font-size:12px;color:var(--ink-3)}.scen .v{font-size:16px;font-weight:700;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.scen .base{background:var(--accent-soft)}
.range{position:relative;height:36px;margin:6px 4px 16px}
.range .rt{position:absolute;left:0;right:0;top:12px;height:6px;border-radius:999px;background:var(--track)}
.range .rb{position:absolute;top:12px;height:6px;border-radius:999px;background:var(--c-fc);opacity:.45}
.range .rm{position:absolute;top:8px;width:4px;height:14px;margin-left:-2px;border-radius:2px;background:var(--c-fc)}
.range .rc{position:absolute;top:5px;width:2px;height:20px;margin-left:-1px;background:var(--ink)}
.range .rl{position:absolute;top:24px;font-size:11px;color:var(--ink-3);transform:translateX(-50%);white-space:nowrap}
.reasons{margin:10px 0 0;padding-left:18px;font-size:14px;color:var(--ink-2)}.reasons li{margin:4px 0}
.notes{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px 32px;margin:6px 0 0}
@media (max-width:640px){.notes{grid-template-columns:minmax(0,1fr)}}
.notes div{min-width:0}.notes h3{font-size:14px;margin:0 0 4px;font-weight:700}.notes p{margin:0;font-size:13.5px;color:var(--ink-2)}
.disclaimer{font-size:12.5px;color:var(--ink-3);border-top:1px solid var(--border);margin:18px 0 0;padding:14px 0 0}
footer{font-size:12px;color:var(--ink-3);text-align:center;margin:28px 0 0}
.cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(min(300px,100%),1fr));gap:16px}
.mcard{background:var(--surface);border:1px solid var(--border);border-radius:16px;padding:16px 18px;box-shadow:var(--shadow);min-width:0;
display:flex;flex-direction:column;gap:10px}
.mcard .top{display:flex;align-items:flex-start;justify-content:space-between;gap:10px}
.mcard h3{font-size:15.5px;margin:0;line-height:1.35;overflow-wrap:anywhere}
.mcard .rank{font-size:12px;font-weight:700;color:var(--ink-3);white-space:nowrap}
.mcard .stats{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:6px}
.mcard .stats .k{font-size:11.5px;color:var(--ink-3)}.mcard .stats .v{font-size:15px;font-weight:700;white-space:nowrap}
.mcard .foot{display:flex;justify-content:space-between;font-size:12px;color:var(--ink-3);gap:8px;flex-wrap:wrap}
.score{display:inline-block;min-width:34px;font-weight:700;font-variant-numeric:tabular-nums}
.th-score{min-width:150px}
@media (max-width:640px){
 .wrap{padding:24px 16px 40px}
 .card{padding:18px 16px;border-radius:14px}
 .card.chart-card{padding:16px 12px}
 .kpis{grid-template-columns:repeat(2,minmax(0,1fr));gap:10px}
 .kpi{padding:12px 14px}
 .kpi .val{font-size:21px}
 td,th{padding:8px 7px}
 table{font-size:13.5px}
 .wbar{min-width:76px;gap:6px}.wbar .pct{min-width:32px;font-size:12px}
 .mdesc{font-size:11.5px}
 .hide-sm{display:none}
 .th-score{min-width:0}
 .score-cell .wbar{min-width:0}.score-cell .track{display:none}
 .wcell .wbar{min-width:0}.wcell .track{display:none}
}
"""


def _page(title: str, body: str, extra_css: str = "") -> str:
    dark = _tokens(_DARK)
    css = (
        ":root{color-scheme:light;" + _tokens(_LIGHT) + "--font:" + _FONT_STACK + "}"
        "@media (prefers-color-scheme: dark){:root:not([data-theme=\"light\"]){color-scheme:dark;" + dark + "}}"
        ":root[data-theme=\"dark\"]{color-scheme:dark;" + dark + "}"
        + _BASE_CSS + extra_css
    )
    return (
        "<!DOCTYPE html>\n<html lang=\"ko\">\n<head>\n<meta charset=\"utf-8\">\n"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
        "<meta name=\"color-scheme\" content=\"light dark\">\n"
        f"<title>{_esc(title)}</title>\n{_FONT_LINK}\n<style>{css}</style>\n</head>\n"
        f"<body>\n<main class=\"wrap\">\n{body}\n</main>\n</body>\n</html>\n"
    )


def _write(text: str, path: Optional[PathLike]) -> None:
    if path is None:
        return
    p = Path(path)
    if p.parent and not p.parent.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def _badge(label: str, conf: Optional[float] = None, tentative: bool = False) -> str:
    """Phase badge; ``tentative`` (:func:`narrative.phase_is_tentative`) marks a call on too few
    observations as ``잠정``, as the narrative words it."""
    color = PHASE_COLOR.get(label, "#898781")
    small = (["잠정"] if tentative else []) + ([f"신뢰도 {conf * 100:.0f}%"] if conf is not None else [])
    conf_html = f" <small>{' · '.join(small)}</small>" if small else ""
    return (f'<span class="badge" style="background:{_hex_rgba(color, 0.13)};'
            f'border:1px solid {_hex_rgba(color, 0.35)}"><span class="dot" style="background:{color}"></span>'
            f'{_esc(_phase_ko(label))}{conf_html}</span>')


def _signed_html(x: Any, text: str) -> str:
    cls = _sign_class(x)
    if not cls:
        return f"<span>{_esc(text)}</span>"
    arrow = "▲" if cls == "pos" else "▼"
    return f'<span class="{cls}"><span class="arrow" aria-hidden="true">{arrow}</span>{_esc(text)}</span>'


_MD_BOLD = re.compile(r"\*\*(?=\S)(.+?)(?<=\S)\*\*")
# single-asterisk emphasis: not inside a word on the left (so "a*b*c" and "2 * 3" stay literal),
# but Korean particles may follow directly ("*강조*입니다")
_MD_EM = re.compile(r"(?<![\w*])\*(?=[^\s*])([^*]+?)(?<=[^\s*])\*(?!\*)")
_MD_CODE = re.compile(r"`([^`]+)`")
_MD_ITEM = re.compile(r"^([-*•+]|(\d{1,3})[.)])\s+(.*)$")
_MD_HEAD = re.compile(r"^(#{1,6})\s+(.*)$")
_MD_RULE = re.compile(r"^(?:-{3,}|\*{3,}|_{3,})$")


def _md_inline(s: str) -> str:
    """Escape ``s``, then render ``**bold**``, ``*italic*`` and ```code``` (markup only from the escaped text)."""
    e = _esc(s)
    e = _MD_CODE.sub(r"<code>\1</code>", e)
    e = _MD_BOLD.sub(r"<strong>\1</strong>", e)
    return _MD_EM.sub(r"<em>\1</em>", e)


def _markdown_lite(text: Any) -> str:
    """Render the narrative's Markdown-lite as safe HTML.

    Supported: ``#``–``######`` headings, ``-``/``*``/``•``/``+`` bullets and
    ``1.``/``1)`` numbered items (``<ul>`` / ``<ol>``), nested lists by
    indentation (2+ spaces deeper than the parent item; tabs count 4), indented
    continuation lines of an item, paragraphs, ``**bold**``, ``*italic*`` and
    ```code```.  Every piece of text is HTML-escaped *before* any markup is
    added, so the input can never inject tags or attributes.
    """
    src = "" if text is None else str(text)
    if not src.strip():
        return '<p class="empty">해석이 제공되지 않았습니다.</p>'
    out: list[str] = []
    para: list[str] = []
    stack: list[list[Any]] = []  # open lists: [indent, tag]; each has an open <li>

    def flush_para() -> None:
        if para:
            out.append("<p>" + " ".join(para) + "</p>")
            para.clear()

    def close_level() -> None:
        _, tag = stack.pop()
        out.append(f"</li></{tag}>")

    def close_lists() -> None:
        while stack:
            close_level()

    def open_list(indent: int, tag: str, num: Optional[str]) -> None:
        start = f' start="{int(num)}"' if tag == "ol" and num is not None and int(num) != 1 else ""
        out.append(f"<{tag}{start}><li>")
        stack.append([indent, tag])

    for raw in src.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        expanded = raw.expandtabs(4)
        line = expanded.strip()
        indent = len(expanded) - len(expanded.lstrip(" "))
        if not line:
            flush_para()
            close_lists()
            continue
        if _MD_RULE.match(line):
            flush_para()
            close_lists()
            continue
        mh = _MD_HEAD.match(line)
        if mh:
            flush_para()
            close_lists()
            tag = "h3" if len(mh.group(1)) <= 2 else "h4"
            out.append(f"<{tag}>{_md_inline(mh.group(2))}</{tag}>")
            continue
        mb = _MD_ITEM.match(line)
        if mb:
            flush_para()
            tag = "ol" if mb.group(2) is not None else "ul"
            if not stack:
                open_list(indent, tag, mb.group(2))
            else:
                # dedent: close the levels this item is not part of (an item still indented
                # past its grandparent's marker stays in the current sub-list)
                while len(stack) > 1 and indent < stack[-1][0] and indent < stack[-2][0] + 2:
                    close_level()
                if indent >= stack[-1][0] + 2:          # deeper: a sub-list inside the open item
                    open_list(indent, tag, mb.group(2))
                elif stack[-1][1] != tag:               # same level, other list type
                    close_level()
                    open_list(indent, tag, mb.group(2))
                else:                                   # next item of the same list
                    stack[-1][0] = min(stack[-1][0], indent)
                    out.append("</li><li>")
            out.append(_md_inline(mb.group(3)))
            continue
        if stack and indent > stack[-1][0]:            # indented continuation of the open item
            out.append(" " + _md_inline(line))
            continue
        close_lists()
        para.append(_md_inline(line))
    flush_para()
    close_lists()
    return "".join(out)


def _kpi(label: str, value_html: str, sub: str, extra: str = "", title: Optional[str] = None) -> str:
    """One KPI tile. ``title`` (plain text, escaped here) defaults to the value's text."""
    if title is None:
        tip = re.sub(r"<[^>]+>", "", value_html)  # already-escaped text, safe for an attribute
        tip = tip.replace("▲", "").replace("▼", "")
    else:
        tip = _esc(title)
    return (f'<div class="kpi"><div class="lbl">{_esc(label)}</div><div class="val" title="{tip}">{value_html}</div>'
            f'{extra}<div class="sub">{sub}</div></div>')


# --------------------------------------------------------------------------- HTML sections


def _header_html(report: TrendReport, d: _ChartData) -> str:
    a = report.analysis
    m = a.metrics
    freq = d.freq
    conf = _finite(getattr(a.phase, "confidence", None))
    src = NARRATIVE_SOURCE_KO.get(str(report.narrative_source), "규칙 기반 해석")
    from . import __version__

    chips = [
        _badge(str(a.phase.label), conf, _tentative(report)),
        f'<span class="chip">기간 {_esc(_fmt_date(m.start, freq))} ~ {_esc(_fmt_date(m.end, freq))}</span>',
        f'<span class="chip">{_esc(UNIT_KO.get(freq, FREQ_KO.get(freq, freq)))} 단위 · 관측 {len(d.hist_y):,}개</span>',
        f'<span class="chip">예측 {int(report.forecast.horizon):,}{_esc(PERIOD_KO.get(freq, "기간"))}</span>',
    ]
    return (
        '<header class="hero">'
        '<div class="eyebrow"><span class="logo" aria-hidden="true"></span>TrendAI 트렌드 리포트 · 과거 → 현재 → 미래</div>'
        f"<h1>{_esc(report.name)}</h1>"
        f'<div class="meta">{"".join(chips)}</div>'
        f'<p class="gen">TrendAI {_esc(__version__)} 버전으로 생성 · {_esc(_method_kind(report.forecast))} · '
        f'{_esc(src)}</p>'
        "</header>"
    )


def _kpis_html(report: TrendReport, d: _ChartData) -> str:
    m, o = report.analysis.metrics, report.outlook
    freq = d.freq
    unit = PERIOD_KO.get(freq, "기간")
    fut_basis, cur_basis = _change_basis(report, d)
    lag = _recent_lag(report, len(d.hist_y))
    recent_span = f"최근 {lag}{unit} 추세 변화"
    peak_sub = _drawdown_sub(report, d)
    p = _finite(o.prob_increase)
    prob_basis = _prob_basis(report, d)
    meter = (f'<div class="meter" role="img" aria-label="{_esc(prob_basis)} {_esc(_fmt_prob(p))}">'
             f'<span style="width:{(min(1.0, max(0.0, p)) * 100) if p is not None else 0:.0f}%"></span></div>')

    # a growth rate that is None gets its reason (crossed zero, intermittent demand ...) next to the "—"
    why = {key: _metric_undefined(report, d, key) for key in ("total_change_pct", "cagr_pct", "recent_change_pct")}

    def why_text(key: str) -> str:
        return _esc(_pct_why(why[key], key))

    total_sub = f'<span class="nw">{_esc(_fmt_date(m.start, freq))} 이후</span> (추세 기준)'
    tr = d.trend_y[np.isfinite(d.trend_y)]
    if why_text("total_change_pct"):
        if why["total_change_pct"] != "intermittent" and tr.size >= 2:
            # no percentage: the change of the trend in its own units, as the narrative states it
            total_sub = (f'<span class="nw">{_esc(_fmt_date(m.start, freq))} 이후</span> 추세 '
                         f'<span class="nw">{_esc(_fmt_num(tr[0]))} → {_esc(_fmt_num(tr[-1]))}</span>')
        total_sub += f" · {why_text('total_change_pct')}"
    # the expected change: no arrow / colour when the narrative calls the outlook 보합 or 방향 불확실
    # (a green ▲ next to "방향 불확실" would claim the rise the narrative does not), and a reason
    # instead of a bare "—" when there is no percentage
    word = _direction_word(report, d)
    nochange = _change_undefined(report, d)
    chg_sub = f'{_esc(fut_basis)} · <span class="nw">{_esc(cur_basis)} 대비</span>'
    if nochange is not None:
        chg_val = f"<span>{_DASH}</span>"
        chg_sub = _esc(nochange.short) + (f" · {_esc(nochange.levels)}" if nochange.levels else "")
    elif str(o.direction) == "flat":
        chg_val = f"<span>{_esc(_fmt_pct(o.expected_change_pct))}</span>"
        if word == _UNCERTAIN_KO:
            chg_sub += f' · <span class="nw">{_esc(word)}</span>'
    else:
        chg_val = _signed_html(o.expected_change_pct, _fmt_pct(o.expected_change_pct))
    tiles = [
        _kpi("현재 값", f"<span>{_esc(_fmt_short(m.last_value))}</span>",
             f'<span class="nw">{_esc(_fmt_date(m.end, freq))} 기준</span> · {_esc(peak_sub)}',
             title=_fmt_num(m.last_value)),
        _kpi("전체 변화", _signed_html(m.total_change_pct, _fmt_pct(m.total_change_pct)), total_sub),
        _kpi("연평균 성장률 CAGR", _signed_html(m.cagr_pct, _fmt_pct(m.cagr_pct)),
             why_text("cagr_pct") or "연 환산 복리 성장률"),
        _kpi("최근 변화", _signed_html(m.recent_change_pct, _fmt_pct(m.recent_change_pct)),
             _esc(recent_span) + (f" · {why_text('recent_change_pct')}" if why_text("recent_change_pct") else "")),
        _kpi("예측 변화", chg_val, chg_sub,
             title=(nochange.sentence + (f" ({nochange.levels})" if nochange.levels else "")) if nochange else None),
        _kpi("상승 확률", f"<span>{_esc(_fmt_prob(p))}</span>",
             f"{_esc(prob_basis)} · 예측 신뢰도 {_esc(CONFIDENCE_KO.get(str(o.confidence), str(o.confidence)))}",
             extra=meter, title=f"{prob_basis}: {_fmt_prob(p)}"),
    ]
    return f'<section class="kpis" aria-label="핵심 지표">{"".join(tiles)}</section>'


def _chart_card(report: TrendReport, d: _ChartData) -> str:
    variants = [("chart-wide", 960, 380), ("chart-mid", 600, 360), ("chart-narrow", 380, 340)]
    charts = "".join(f'<div class="chart-v {cls}">{svg_chart(report, w, h)}</div>' for cls, w, h in variants)
    n_b = len(d.bands)
    band_txt = ", ".join(f"{lv}%" for lv, _, _ in sorted(d.bands, key=lambda b: b[0]))
    desc = "실선은 실제값, 점선은 추세, 주황선은 예측 평균"
    desc += f"이며 음영은 {band_txt} 예측 구간입니다." if n_b else "입니다."
    desc += " 세로 점선이 현재 시점이고, 번호 표시는 아래 변곡점 표의 번호입니다." if report.analysis.changepoints \
        else " 세로 점선이 현재 시점입니다."
    if d.spikes_clipped():
        desc += " 축 범위를 크게 벗어난 이상치는 위·아래 가장자리에 ▲와 값으로 표시했습니다."
    return (f'<section class="card chart-card"><h2>과거 · 현재 · 미래</h2><p class="desc">{_esc(desc)}</p>'
            f"{charts}</section>")


def _narrative_card(report: TrendReport) -> str:
    src_key = "llm" if str(report.narrative_source) == "llm" else "rule"
    src = NARRATIVE_SOURCE_KO[src_key]
    return (f'<section class="card narr"><h2>AI 트렌드 해석 <span class="src {src_key}">{_esc(src)}</span></h2>'
            f'<p class="desc">과거의 흐름, 현재의 국면, 미래 전망을 문장으로 정리했습니다.</p>'
            f"{_markdown_lite(report.narrative)}</section>")


def _outlook_card(report: TrendReport, d: _ChartData) -> str:
    """'미래 전망' card.

    Two different kinds of numbers live here and are labelled as such: the
    expected change and the probability of a rise compare two *season means*
    (future vs current level, see :func:`_change_basis`), while the scenarios are
    single *horizon-end values*.  The scenario bar's reference tick is therefore
    an observed single value at the same point of the seasonal cycle
    (:func:`_scenario_ref`), never the current season mean.
    """
    o, fc = report.outlook, report.forecast
    freq = d.freq
    unit = PERIOD_KO.get(freq, "기간")
    direction = str(o.direction)
    dir_ko = _direction_word(report, d)   # the narrative's word: 보합 / 방향 불확실 for a 'flat' call
    cls = {"up": "pos", "down": "neg"}.get(direction, "")
    end_fc = _fmt_date(d.fc_dates[-1], freq) if len(d.fc_dates) else _DASH
    scen = o.scenarios or {}
    # a bound at the 0 floor of a nonnegative forecast reads "0(하한)" (narrative.scenario_value)
    sc_html = "".join(
        f'<div class="{"base" if k == "base" else ""}"><div class="k">{ko}</div>'
        f'<div class="v" title="{_esc(_scen_text(report, scen.get(k)))}">'
        f'{_esc(_scen_text(report, scen.get(k), short=True))}</div></div>'
        for k, ko in SCENARIOS_KO)
    # range visual: pessimistic..optimistic band, base marker, tick at the comparable observation
    ref = _scenario_ref(report, d)
    ref_v = ref[0] if ref is not None else None
    if ref is None:
        ref_lab = ref_cap = ""
    elif ref[2]:
        ref_date = _fmt_date(ref[1], freq)
        ref_lab = f"{ref_date} 실제 {_fmt_short(ref_v)}"
        ref_cap = f"세로선은 같은 계절 시점인 {ref_date}의 실제값입니다"
    else:
        ref_lab = f"현재 {_fmt_short(ref_v)}"
        ref_cap = f"세로선은 현재 값({_fmt_date(ref[1], freq)})입니다"
    vals = [v for v in (_finite(scen.get("pessimistic")), _finite(scen.get("optimistic")),
                        _finite(scen.get("base")), ref_v) if v is not None]
    range_html = ""
    if len(vals) >= 2 and max(vals) > min(vals):
        lo, hi = min(vals), max(vals)
        pad = (hi - lo) * 0.08
        lo, hi = lo - pad, hi + pad

        def pos(v: float) -> float:
            return (v - lo) / (hi - lo) * 100.0

        segs = ['<div class="rt"></div>']
        ps, op = _finite(scen.get("pessimistic")), _finite(scen.get("optimistic"))
        has_band = ps is not None and op is not None
        if has_band:
            a, b = sorted((pos(ps), pos(op)))
            segs.append(f'<div class="rb" style="left:{a:.1f}%;width:{max(b - a, 0.5):.1f}%"></div>')
        bs = _finite(scen.get("base"))
        if bs is not None:
            segs.append(f'<div class="rm" style="left:{pos(bs):.1f}%" title="기준 시나리오"></div>')
        if ref_v is not None:
            pc = pos(ref_v)
            segs.append(f'<div class="rc" style="left:{pc:.1f}%" title="{_esc(ref_lab)}"></div>')
            lab_pos = min(85.0, max(15.0, pc))
            segs.append(f'<div class="rl" style="left:{lab_pos:.1f}%">{_esc(ref_lab)}</div>')
        cap = (f"막대는 {end_fc} 시점의 비관~낙관 범위" if has_band else f"표시는 {end_fc} 시점의 기준 시나리오")
        cap += f"이고, {ref_cap}." if ref_cap else "입니다."
        floor = _floor_note(report)
        if floor:
            cap += " " + floor
        range_html = (f'<div class="range" role="img" aria-label="{_esc(cap)}">{"".join(segs)}</div>'
                      f'<p class="desc" style="margin:0 0 14px">{_esc(cap)}</p>')
    peak = _ts(o.forecast_peak_date)
    skill = _skill_value(fc)
    fut_basis, cur_basis = _change_basis(report, d)
    adj_note = _adjustment_note(report, d)
    nochange = _change_undefined(report, d)
    # the projected phase as far as the narrative projects it (판단 보류 / 참고용 / 잠정, see _future_phase_hedge)
    ph_val, ph_note = _future_phase_parts(report)
    facts: list[tuple[str, str, str, str]] = [  # (label, value, tooltip, value class)
        ("미래 단계", ph_val, _future_phase_tip(report), "ph-hedged" if ph_note else ""),
        ("예측 신뢰도", CONFIDENCE_KO.get(str(o.confidence), str(o.confidence)), "", ""),
        (_SKILL_LABEL, _rel_error_text(skill), _skill_tip(_intermittent(report)), _rel_error_class(skill)),
        ("예측 정점", _fmt_date(peak, freq) if peak is not None else "예측 기간 내 없음", "", ""),
    ]
    facts_html = "".join(
        (f'<dt><abbr title="{_esc(tip)}">{_esc(k)}</abbr></dt>' if tip else f"<dt>{_esc(k)}</dt>")
        + (f'<dd>{_esc(v)}<span class="dnote">{_esc(ph_note)}</span></dd>' if vc == "ph-hedged"
           else f'<dd class="{vc}">{_esc(v)}</dd>' if vc else f"<dd>{_esc(v)}</dd>")
        for k, v, tip, vc in facts)
    arrow = DIRECTION_ARROW.get(direction, "")
    prob_txt = _fmt_prob(o.prob_increase)
    if nochange is None:
        chg_html = f'<span class="nw">{_esc(_fmt_pct(o.expected_change_pct))} 예상</span>'
        basis = f"변화율과 상승 확률은 모두 {fut_basis}을 {cur_basis}과 비교한 값입니다."
    else:  # no percentage: the reason and the two compared levels, worded like the narrative
        chg_html = '<span class="nw">변화율 없음</span>'   # the reason follows in the sentence below
        basis = (f"{nochange.sentence} 상승 확률은 {fut_basis}을 {cur_basis}과 비교한 값입니다"
                 + (f"({nochange.levels})." if nochange.levels else "."))
    return (
        '<section class="card"><h2>미래 전망</h2>'
        f'<p class="desc">{_esc(end_fc)}까지 · 향후 {int(fc.horizon):,}{_esc(unit)}</p>'
        f'<div class="dir"><span class="big {cls}">{_esc(arrow)} {_esc(dir_ko)}</span>'
        f'<span class="lab">{chg_html} · '
        f'<span class="nw" title="{_esc(_prob_basis(report, d))}">상승 확률 {_esc(prob_txt)}</span></span></div>'
        f'<p class="desc" style="margin:-8px 0 14px">{_esc(basis)}{(" " + _esc(adj_note)) if adj_note else ""}</p>'
        f'<h3 class="sub">시나리오 ({_esc(end_fc)} 시점 값)</h3><div class="scen">{sc_html}</div>{range_html}'
        f'<dl class="facts">{facts_html}</dl></section>'
    )


def _diagnosis_card(report: TrendReport, d: _ChartData) -> str:
    a = report.analysis
    m = a.metrics
    freq = d.freq
    conf = _finite(getattr(a.phase, "confidence", None))
    reasons = [str(r) for r in (a.phase.reasons or []) if str(r).strip()]
    # the rule-based narrative quotes the reasons as "근거:" bullets in the card next to this one
    # (an English narrative quotes their translation as "Evidence:" bullets)
    narr = " ".join(str(report.narrative or "").split())

    def quoted(r: str) -> bool:
        if " ".join(r.split()) in narr:
            return True
        en = _translate_reason(r)
        return bool(en) and " ".join(str(en).split()) in narr

    shown = [r for r in reasons if not quoted(r)]
    reasons_html = ("<ul class=\"reasons\">" + "".join(f"<li>{_esc(r)}</li>" for r in shown) + "</ul>"
                    if shown else "")
    desc = "수명주기 단계와 판단 근거"
    if reasons and len(shown) < len(reasons):
        desc = ("수명주기 단계와 주요 지표 · 판단 근거는 ‘AI 트렌드 해석’의 현재 항목에 있습니다" if not shown
                else "수명주기 단계와 판단 근거 · 나머지 근거는 ‘AI 트렌드 해석’에 있습니다")
    vol = _finite(m.volatility)
    mom = _fmt_num(m.momentum, 2, signed=True)
    peak_html = _esc(f"{_fmt_short(m.peak_value)} ({_fmt_date(m.peak_date, freq)})")
    spike = _excluded_spike(report, d)
    if spike is not None:  # metrics.peak_value leaves a flagged one-off spike out
        peak_html += (f'<span class="dnote">일회성 급등 {_esc(_fmt_short(spike[0]))}'
                      f'({_esc(_fmt_date(spike[1], freq))}) 제외</span>')
    facts = [
        # momentum = recent trend slope - long-run (Theil-Sen) slope, in % of the *current* trend
        # level per period (analysis.compute_metrics; the narrative's "현재 추세 수준 대비")
        ("모멘텀", _esc(mom + (f"%p{_PER_UNIT.get(freq, '')}" if mom != _DASH else ""))),
        ("추세 강도", _esc(_fmt_num(m.trend_strength, 2))),
        ("계절성 강도", _esc(_fmt_num(m.seasonal_strength, 2))),
        ("변동성", _esc(_fmt_pct(vol * 100 if vol is not None else None, signed=False))),
        ("최고점", peak_html),
    ]
    tips = {"모멘텀": _MOMENTUM_TIP}
    facts_html = "".join((f'<dt><abbr title="{_esc(tips[k])}">{_esc(k)}</abbr></dt>' if k in tips
                          else f"<dt>{_esc(k)}</dt>") + f"<dd>{v}</dd>" for k, v in facts)
    return (
        '<section class="card"><h2>현재 진단</h2>'
        f'<p class="desc">{_esc(desc)}</p>{_badge(str(a.phase.label), conf, _tentative(report))}{reasons_html}'
        f'<dl class="facts" style="margin-top:16px">{facts_html}</dl></section>'
    )


def _events_card(report: TrendReport, d: _ChartData) -> str:
    a = report.analysis
    freq = d.freq
    unit = {"D": "일", "W": "주", "M": "월", "Q": "분기", "Y": "년"}.get(freq, "기간")
    cps = list(a.changepoints or [])
    if cps:
        rows = []
        for num, cp in enumerate(cps, 1):
            kind_ko, color = _cp_kind(cp)
            rows.append(
                f'<tr><td class="num" style="text-align:left"><span class="cpn" aria-label="변곡점 {num}">{num}</span>'
                f"{_esc(_fmt_date(cp.date, freq))}</td>"
                f'<td><span class="kind"><span class="dot" style="background:{color}"></span>'
                f"{_esc(kind_ko)}</span></td>"
                f"<td class=\"num\">{_esc(_fmt_num(cp.slope_before, signed=True))} → "
                f"{_esc(_fmt_num(cp.slope_after, signed=True))}</td>"
                f"<td class=\"num hide-sm\">{_esc(_fmt_num(cp.magnitude, 2))}</td></tr>")
        cp_html = (f'<div class="table-wrap"><table><thead><tr><th>날짜</th><th>유형</th>'
                   f'<th class="num">기울기 (/{_esc(unit)}) 전 → 후</th><th class="num hide-sm">크기</th></tr></thead>'
                   f'<tbody>{"".join(rows)}</tbody></table></div>')
    else:
        cp_html = '<p class="empty">뚜렷한 구조 변화(변곡점)가 감지되지 않았습니다.</p>'
    anoms = list(a.anomalies or [])
    if anoms:
        rows = []
        for an in anoms:
            z = _finite(an.zscore)
            rows.append(
                f"<tr><td class=\"num\" style=\"text-align:left\">{_esc(_fmt_date(an.date, freq))}</td>"
                f"<td class=\"num\">{_esc(_fmt_num(an.value))}</td>"
                f"<td class=\"num\">{_esc(_fmt_num(an.expected))}</td>"
                f"<td class=\"num\">{_signed_html(z, _fmt_num(z, 1, signed=True))}</td></tr>")
        # intermittent demand: an anomaly is an unusually large demand and Anomaly.expected the
        # typical (median non-zero) demand size, not a trend expectation (narrative.is_intermittent)
        exp_head = ('<abbr title="0이 잦은 간헐 수요라, 0이 아닌 수요의 평소(중앙값) 크기와 비교했습니다">평소 규모</abbr>'
                    if _intermittent(report) else "예상")
        an_html = ('<div class="table-wrap"><table><thead><tr><th>날짜</th><th class="num">실제</th>'
                   f'<th class="num">{exp_head}</th><th class="num">z-점수</th></tr></thead>'
                   f'<tbody>{"".join(rows)}</tbody></table></div>')
    else:
        an_html = '<p class="empty">이상치가 감지되지 않았습니다.</p>'
    return (
        '<section class="card"><h2>과거의 사건</h2>'
        '<p class="desc">추세의 기울기가 바뀐 변곡점과, 추세·계절성으로 설명되지 않는 이상치</p>'
        f'<div class="grid-even"><div><h3 class="sub">변곡점 {len(cps)}개</h3>{cp_html}</div>'
        f'<div><h3 class="sub">이상치 {len(anoms)}개</h3>{an_html}</div></div></section>'
    )


def _models_card(report: TrendReport) -> str:
    fc = report.forecast
    scores = list(fc.model_scores or [])
    skill = _skill_value(fc)
    # ForecastResult.skill compares MAEs -- RMSEs for intermittent demand, the metric that ranked the
    # candidates (forecast.auto_forecast) -- so the per-model column uses the same metric
    intermittent = _intermittent(report)
    metric, metric_ko = ("rmse", "제곱 오차(RMSE)") if intermittent else ("mae", "평균 절대 오차")
    naive_name: Optional[str] = None
    if scores:
        max_w = max([_finite(s.weight) or 0.0 for s in scores] + [0.0])
        # per-model error versus the best naive benchmark on the same backtest windows
        # (the same ratio as ForecastResult.skill, row by row)
        naive_name, naive_err = _best_naive(scores, metric)
        no_bench = ('<span class="muted" title="단순 기준 예측이 비교 후보에 없어 모델별로는 비교하지 않았습니다">'
                    f"{_DASH}</span>")
        rows = []
        for i, s in enumerate(scores, 1):
            w = _finite(s.weight) or 0.0
            desc = _model_desc(s.name)
            fill = (w / max_w * 100) if max_w > 0 else 0.0
            err = _finite(getattr(s, metric, None))
            if naive_err is None:
                rel_html = no_bench
            elif err is None or err < 0:
                rel_html = _DASH
            elif str(s.name) == naive_name:
                rel_html = '<span class="muted">기준</span>'
            elif naive_err > 0:
                ratio = err / naive_err
                rel_cls = _rel_error_class(ratio)
                rel_txt = _esc(_rel_error_text(ratio))
                rel_html = f'<span class="{rel_cls}">{rel_txt}</span>' if rel_cls else rel_txt
            else:
                rel_html = _DASH
            rows.append(
                f'<tr class="{"inens" if w > 0 else ""}"><td class="num" style="text-align:left">{i}</td>'
                f'<td><span class="mname">{_model_html(s.name)}</span>'
                f'{f"<span class=mdesc>{_esc(desc)}</span>" if desc else ""}</td>'
                f'<td class="num">{_esc(_fmt_num(s.mase, 3))}</td>'
                f'<td class="num">{rel_html}</td>'
                f'<td class="num hide-sm">{_esc(_fmt_pct(s.smape, signed=False))}</td>'
                f'<td class="wcell"><div class="wbar"><div class="track"><div class="fill" style="width:{fill:.1f}%"></div></div>'
                f'<span class="pct">{w * 100:.0f}%</span></div></td></tr>')
        rel_tip = ("같은 백테스트 구간에서 가장 나은 단순 기준 예측"
                   + (f"({_model_label(naive_name, 'ko')})" if naive_name else "")
                   + f"보다 {metric_ko}가 몇 % 작은지(−) 큰지(+)")
        table = ('<div class="table-wrap"><table><thead><tr><th>#</th><th>모델</th>'
                 f'<th class="num"><abbr title="{_esc(_MASE_TIP)}">MASE</abbr></th>'
                 f'<th class="num"><abbr title="{_esc(rel_tip)}">단순 대비</abbr></th>'
                 '<th class="num hide-sm">sMAPE</th><th><span class="hide-sm">앙상블 </span>가중치</th></tr></thead>'
                 f'<tbody>{"".join(rows)}</tbody></table></div>')
        if naive_name is None:
            # no benchmark rows (intermittent demand offers level models only; explicit candidates):
            # say why the column is empty instead of leaving a bare "—"
            why = ("0이 잦은 간헐 수요라 수요 수준을 추정하는 모델만 후보로 써서" if intermittent
                   else "후보 목록에 단순 기준 예측(직전값·계절 반복·평균 기울기)이 없어")
            table += (f'<p class="desc" style="margin:8px 0 0">‘단순 대비’: {_esc(why)} 모델별로는 비교하지 '
                      "않았습니다. 최종 방법은 같은 구간에서 단순 기준 예측과 따로 비교했습니다(아래 "
                      "‘단순 예측 대비 오차’).</p>")
    else:
        table = '<p class="empty">백테스트 점수가 없습니다 (데이터가 짧아 단순 모델을 사용).</p>'
    bm = _finite(fc.backtest_mase)
    if bm is None and skill is None:
        acc_txt = "데이터가 짧아 백테스트로 정확도를 검증하지 못했습니다."
    else:
        acc_txt = _skill_sentence(skill)
    skill_html = (f'<abbr title="{_esc(_skill_tip(intermittent))}">{_SKILL_LABEL}</abbr> '
                  f'<strong class="{_rel_error_class(skill)}">{_esc(_rel_error_text(skill))}</strong>')
    mase_note = ""
    if bm is not None:
        judge = "위의 ‘단순 대비’" if naive_name else "‘단순 예측 대비 오차’"
        mase_note = (f"기술 참고: 최종 방법의 백테스트 MASE {bm:.2f}. MASE는 학습 구간에서 한 기 앞을 계절 단순 예측한 "
                     "오차를 1로 둔 상대 오차라, 여러 기 앞을 내다보는 예측은 단순 예측보다 정확해도 1을 넘기 쉬우므로 "
                     f"단순 예측과의 우열은 {judge}로 판단합니다.")
    lead = "롤링 원점 백테스트 성적(MASE 낮을수록 좋음). 왼쪽 강조선은 최종 앙상블에 포함된 모델입니다."
    if intermittent and scores:
        lead += (" 0이 잦은 간헐 수요라 모델 순위·가중치·최종 선택과 ‘단순 대비’는 제곱 오차(RMSE) 기준이며, "
                 "표는 MASE 순서입니다.")
    # the method in words; no tooltip with the raw method string ("ensemble(linear_trend 0.35, …)":
    # raw ids, Python floats, English) -- the ids and the weights are the table rows above
    method_txt = _describe_method(fc, "ko")
    return (
        '<section class="card"><h2>모델 리더보드</h2>'
        f'<p class="desc">{_esc(lead)}</p>'
        f'{table}<p class="desc" style="margin:14px 0 0">최종 방법: {_esc(method_txt)}</p>'
        f'<p style="margin:8px 0 0;font-size:14px">{skill_html} · {_esc(acc_txt)}</p>'
        + (f'<p class="desc" style="margin:6px 0 0">{_esc(mase_note)}</p>' if mase_note else "")
        + "</section>"
    )


def _forecast_table_card(report: TrendReport, d: _ChartData) -> str:
    freq = d.freq
    levels = sorted({lv for lv, _, _ in d.bands})
    band_by = {lv: (blo, bhi) for lv, blo, bhi in d.bands}
    if not len(d.fc_y):
        return ('<section class="card"><h2>예측 표</h2>'
                '<p class="empty">예측 결과가 없습니다.</p></section>')
    head = "<th>날짜</th><th class=\"num\">예측값</th>" + "".join(
        f'<th class="num">{lv}% 구간</th>' for lv in levels)
    rows = []
    for i in range(len(d.fc_y)):
        cells = [f'<td class="num" style="text-align:left">{_esc(_fmt_date(d.fc_dates[i], freq))}</td>',
                 f'<td class="num"><strong>{_esc(_fmt_num(d.fc_y[i]))}</strong></td>']
        for lv in levels:
            blo, bhi = band_by[lv]
            cells.append(f'<td class="num">{_esc(_fmt_num(blo[i]))} ~ {_esc(_fmt_num(bhi[i]))}</td>')
        rows.append("<tr>" + "".join(cells) + "</tr>")
    tall = " tall" if len(rows) > 24 else ""
    note = " 음수가 될 수 없는 지표라 0 미만은 잘라냈습니다." if getattr(report.forecast, "nonnegative", False) else ""
    return (
        '<section class="card"><h2>예측 표</h2>'
        f'<p class="desc">{len(rows)}개 시점의 예측 평균과 예측 구간.{_esc(note)}</p>'
        f'<div class="table-wrap{tall}"><table><thead><tr>{head}</tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table></div></section>'
    )


def _notes_card(report: TrendReport, d: _ChartData) -> str:
    a, fc = report.analysis, report.forecast
    freq = d.freq
    unit = PERIOD_KO.get(freq, "기간")
    period = int(getattr(a.decomposition, "period", 1) or 1)
    if period <= 1:
        decomp = "계절 주기가 없거나 데이터가 짧아 계절 성분 없이, 강건한 국소 선형 평활로 추세를 추정했습니다."
    elif _is_multiplicative(report):
        decomp = (f"계절 변동 폭이 수준에 비례하므로 로그 공간에서 승법 분해(STL 유사)해 계절 주기 {period}{unit}의 "
                  "계절 성분을 분리하고 국소 선형 평활로 추세를 추정했습니다.")
    else:
        decomp = (f"강건한 가법 분해(STL 유사)로 계절 주기 {period}{unit}의 계절 성분을 분리하고 "
                  "국소 선형 평활로 추세를 추정했습니다.")
    if any(str(getattr(cp, "kind", "")) == "level_shift" for cp in (a.changepoints or [])):
        decomp += " 수준이 한 번에 바뀐 시점에서는 추세를 끊어서 추정했습니다."
    levels = ", ".join(f"{lv}%" for lv in sorted({lv for lv, _, _ in d.bands})) or "—"
    n_models = len(fc.model_scores or [])
    if n_models:
        with_naive = _best_naive(fc.model_scores or [])[0] is not None
        if _intermittent(report):
            # forecast offers intermittent demand level models only (ses, theta, holt_damped, sba, ...)
            ens = (f"0이 잦은 간헐 수요라 수요 수준을 추정하는 모델(SBA 등) {n_models}개를 "
                   "롤링 원점 백테스트로 평가하고, 제곱 오차(RMSE)가 작은 상위 모델을 가중 평균한 앙상블(또는 확실히 "
                   "우수한 단일 모델)을 최종 예측으로 씁니다. 단순 기준 예측은 같은 구간에서 따로 계산해 정확도 비교에만 "
                   "씁니다.")
        else:
            ens = (f"{'단순 기준 예측을 포함한 ' if with_naive else ''}{n_models}개 모델을 롤링 원점 백테스트로 평가해 "
                   "MASE 순으로 정렬하고, 상위 모델을 역-MASE 가중 평균한 앙상블(또는 확실히 우수한 단일 모델)을 최종 "
                   "예측으로 씁니다.")
        bands = (f"최종 방법의 백테스트 예측 오차를 시점별로 모아 {levels} 예측 구간을 추정합니다. 백테스트가 짧으면 "
                 "t-분포로 구간을 넓히고, 예측 시점이 멀수록 구간이 넓어집니다.")
    else:
        ens = "데이터가 짧아 백테스트로 모델을 비교하지 못해, 단순 모델로 예측했습니다."
        bands = f"백테스트 없이 추정한 {levels} 예측 구간이라 참고용이며, 예측 시점이 멀수록 넓어집니다."
    notes = [
        ("시계열 분해", decomp + " 증감률·CAGR 같은 성장 지표는 잡음을 줄이기 위해 추세 성분으로 계산합니다."),
        ("변곡점 · 이상치", "계절 조정된 시계열에 구간별 선형 회귀(BIC 유사 벌점)를 적용해 기울기·수준이 바뀐 시점을 찾고, "
                      "잔차의 강건 z-점수(중앙값·MAD)가 큰 점을 이상치로 표시합니다."),
        ("앙상블 백테스트", ens),
        ("예측 구간", bands),
    ]
    items = "".join(f"<div><h3>{_esc(h)}</h3><p>{_esc(t)}</p></div>" for h, t in notes)
    return (
        '<section class="card"><h2>분석 방법</h2>'
        f'<div class="notes">{items}</div>'
        '<p class="disclaimer">이 리포트는 과거 데이터의 통계적 패턴을 바탕으로 한 추정이며 미래를 보장하지 않습니다. '
        '외부 사건, 정책, 시장 변화 등 데이터에 담기지 않은 요인은 반영되지 않으므로, 중요한 의사결정에는 '
        '도메인 지식과 추가 자료를 함께 검토하세요.</p></section>'
    )


def render_html(report: TrendReport, path: Optional[PathLike] = None) -> str:
    """Render ``report`` as one self-contained, responsive Korean HTML page.

    The page has no scripts (inline CSS + inline SVG only; the Noto Sans KR
    Google Fonts link is optional and the system font stack is used without
    it).  When ``path`` is given the page is also written there (UTF-8).
    Returns the HTML text.
    """
    d = _ChartData(report)
    body = "\n".join([
        _header_html(report, d),
        _kpis_html(report, d),
        _chart_card(report, d),
        '<div class="grid-2">' + _narrative_card(report)
        + '<div class="stack">' + _outlook_card(report, d) + _diagnosis_card(report, d) + "</div></div>",
        _events_card(report, d),
        _models_card(report),
        _forecast_table_card(report, d),
        _notes_card(report, d),
        '<footer>TrendAI · 과거에서 현재, 미래까지 — 시계열 트렌드 분석·예측 AI</footer>',
    ])
    page = _page(f"{report.name} — 트렌드 리포트", body)
    _write(page, path)
    return page


# --------------------------------------------------------------------------- compare page


def _cell(row: Any, key: str) -> Any:
    try:
        v = row[key]
    except (KeyError, IndexError, TypeError):
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    return v


def _change_spans(r: Optional[TrendReport]) -> tuple[str, str]:
    """``(forecast horizon, recent-change span)`` of a report as ``'12개월'``-style text."""
    if r is None:
        return "", ""
    freq = str(getattr(r.analysis, "freq", ""))
    n = int(np.isfinite(np.asarray(r.analysis.series, dtype=float)).sum())
    return _span_text(_horizon_steps(r), freq), _span_text(_recent_lag(r, n), freq)


def _link(text_html: str, href: Optional[str]) -> str:
    """``text_html`` wrapped in ``<a>`` when ``href`` is safe (see :func:`_safe_href`)."""
    safe = _safe_href(href)
    return f'<a href="{_esc(safe)}">{text_html}</a>' if safe else text_html


_COMPARE_CSS = (
    ".sname a,.mcard h3 a{color:inherit;text-decoration:underline;text-decoration-thickness:1px;"
    "text-decoration-color:var(--border-strong);text-underline-offset:3px}"
    ".sname a:hover,.mcard h3 a:hover{color:var(--accent);text-decoration-color:currentColor}"
    "td .hz,th .hz{font-size:11.5px;color:var(--ink-3);font-weight:400;margin-left:4px}"
    ".dl{display:block;font-size:11.5px;line-height:1.3;color:var(--ink-3);font-weight:400;white-space:normal}"
    # label / value / note rows shared by the three stats (subgrid), so a value with a note under it
    # ("보합", "방향 불확실") or a label that wraps does not shift the other columns
    ".mcard .stats{grid-template-rows:auto auto auto;row-gap:0;align-items:end}"
    ".mcard .stats>div{display:grid;grid-row:span 3;grid-template-rows:subgrid}"
    ".mcard .stats .v,.mcard .stats .dl{align-self:start}"
)
#: the two levels behind expected_change_pct / prob_increase (forecast.outlook_levels): the last season
#: of the forecast -- the whole horizon, seasonally adjusted, when it is shorter (weekly data: 26 of 52
#: weeks); the last value for yearly data -- against the last observed season
_FUT_LEVEL_KO = ("예측 마지막 한 주기의 평균(예측 기간이 한 주기보다 짧으면 예측 기간 전체를 계절 보정한 평균, "
                 "연 단위 데이터는 마지막 예측값)")
_CUR_LEVEL_KO = "최근 한 주기의 평균(연 단위는 최근 값)"
_EC_TIP = (f"{_FUT_LEVEL_KO}이 {_CUR_LEVEL_KO}보다 몇 % 높은지(낮은지)입니다. "
           "시리즈마다 예측 기간이 다르면 값 옆에 그 기간을 적었습니다.")
_RC_TIP = "추세 성분이 최근 한 주기(연 단위 데이터는 1년) 동안 변한 비율입니다."
_PI_TIP = (f"{_FUT_LEVEL_KO}이 {_CUR_LEVEL_KO}보다 높을 확률입니다(예측 변화와 같은 기준). "
           "예측 기간 끝 시점의 값이 지금보다 높을 확률이 아닙니다.")
#: what the rising score (compare.rising_score) is: the outlook *and* the recent momentum, not the
#: outlook alone -- "higher = stronger expected rise" overclaimed
_SCORE_TIP = ("상승 점수: 전망(예측 변화·상승 확률)과 최근 기세(최근 변화·모멘텀)를 합친 0–100 점수, 50 = 중립. "
              "안정기·변동성 장세에서는 전망만 반영합니다.")


def render_compare_html(reports: list[TrendReport], ranking: pd.DataFrame,
                        path: Optional[PathLike] = None,
                        links: Optional[Mapping[str, str]] = None) -> str:
    """Comparison page: ranking table + a grid of small-multiple cards.

    ``ranking`` has the columns ``name, phase, momentum, recent_change_pct,
    expected_change_pct, prob_increase, score`` (best first); extra columns are
    ignored and missing values are shown as ``—``.  Cards follow the ranking
    order; reports not present in ``ranking`` are appended at the end.

    The changes are shown as reported (each in its own series' time span) with
    the span always visible: in the column header when every row shares it,
    otherwise next to each value -- a 30-day change of a daily series is never
    shown as if it were comparable to a 12-month change of a monthly one.

    ``links`` maps a series name to the URL of its own report (e.g. the
    per-series pages the CLI writes next to this one).  Card titles and ranking
    names become links; only relative and ``http(s)`` URLs are used (anything
    else, such as ``javascript:`` or ``data:``, is ignored) and they are escaped.
    """
    if isinstance(reports, Mapping):  # tolerate {name: report} (e.g. pipeline.run_many output)
        reports = list(reports.values())
    reports = list(reports or [])
    links = dict(links) if isinstance(links, Mapping) else {}
    by_name: dict[str, TrendReport] = {}
    for r in reports:
        by_name.setdefault(str(r.name), r)
    rank_df = ranking if isinstance(ranking, pd.DataFrame) else pd.DataFrame(ranking)

    names = []
    for _, row in rank_df.iterrows():
        nm = _cell(row, "name")
        names.append("" if nm is None else str(nm))
    spans = {nm: _change_spans(by_name.get(nm)) for nm in names}
    fc_spans = {sp[0] for sp in spans.values()}
    rc_spans = {sp[1] for sp in spans.values()}
    fc_shared = next(iter(fc_spans)) if len(fc_spans) == 1 else ""
    rc_shared = next(iter(rc_spans)) if len(rc_spans) == 1 else ""

    def hz(text: str) -> str:
        return f'<span class="hz">{_esc(text)}</span>' if text else ""

    def paren(text: str) -> str:
        return f"({text})" if text else ""

    # chart data per report (also caches forecast.outlook_levels for the direction words and reasons)
    dds: dict[int, _ChartData] = {}

    def dd_of(r: TrendReport) -> _ChartData:
        if id(r) not in dds:
            dds[id(r)] = _ChartData(r)
        return dds[id(r)]

    def change_html(r: Optional[TrendReport], ec: Any) -> tuple[str, str]:
        """``(value, note)`` of an expected change as the narrative calls it: no arrow / colour for a
        'flat' outlook, whose word (보합 / 방향 불확실) is the note; without a percentage the note is
        the reason instead of leaving a bare "—"."""
        if r is None:
            return _signed_html(ec, _fmt_pct(ec)), ""
        rd = dd_of(r)
        nochange = _change_undefined(r, rd)
        if nochange is not None:
            return (f'<span title="{_esc(nochange.sentence)}">{_DASH}</span>',
                    f'<span class="dl">{_esc(nochange.short)}</span>')
        if str(r.outlook.direction) != "flat":
            return _signed_html(ec, _fmt_pct(ec)), ""
        return f'<span>{_esc(_fmt_pct(ec))}</span>', f'<span class="dl">{_esc(_direction_word(r, rd))}</span>'

    rows_html = []
    scores: dict[str, Optional[float]] = {}
    for pos, ((_, row), nm) in enumerate(zip(rank_df.iterrows(), names), 1):
        phase = _cell(row, "phase")
        phase_s = str(phase) if phase is not None else ""
        score = _finite(_cell(row, "score"))
        scores[nm] = score
        mom = _cell(row, "momentum")
        mom_txt = _fmt_num(mom, 2, signed=True)
        r_freq = str(getattr(getattr(by_name.get(nm), "analysis", None), "freq", ""))
        mom_unit = f"%p{_PER_UNIT.get(r_freq, '')}" if mom_txt != _DASH else ""
        rc = _cell(row, "recent_change_pct")
        ec = _cell(row, "expected_change_pct")
        pi = _cell(row, "prob_increase")
        sw = min(100.0, max(0.0, score)) if score is not None else 0.0
        badge = _badge(phase_s) if phase_s else _DASH
        fc_sp, rc_sp = spans[nm]
        ec_html, ec_note = change_html(by_name.get(nm), ec)
        rows_html.append(
            f'<tr><td class="num" style="text-align:left"><strong>{pos}</strong></td>'
            f'<td><span class="sname">{_link(_esc(nm), links.get(nm))}</span></td><td class="hide-sm">{badge}</td>'
            f'<td class="num hide-sm">{_signed_html(mom, mom_txt)}'
            f'{f"<span class=unit>{_esc(mom_unit)}</span>" if mom_unit else ""}</td>'
            f'<td class="num hide-sm">{_signed_html(rc, _fmt_pct(rc))}{"" if rc_shared else hz(rc_sp)}</td>'
            f'<td class="num">{ec_html}{"" if fc_shared else hz(fc_sp)}{ec_note}</td>'
            f'<td class="num hide-sm">{_esc(_fmt_prob(pi))}</td>'
            f'<td class="score-cell"><div class="wbar" title="{_esc(_SCORE_TIP)}"><div class="track"><div class="fill" style="width:{sw:.1f}%"></div>'
            f'<span class="mid"></span></div><span class="pct score">{_esc(_fmt_num(score, 0))}</span></div></td></tr>')
    if rows_html:
        table = ('<div class="table-wrap"><table><thead><tr><th>순위</th><th>이름</th><th class="hide-sm">단계</th>'
                 f'<th class="num hide-sm"><abbr title="{_esc(_MOMENTUM_TIP)}">모멘텀 (%p/기간)</abbr></th>'
                 f'<th class="num hide-sm"><abbr title="{_esc(_RC_TIP)}">최근 변화</abbr>{hz(paren(rc_shared))}</th>'
                 f'<th class="num"><abbr title="{_esc(_EC_TIP)}">예측 변화</abbr>{hz(paren(fc_shared))}</th>'
                 f'<th class="num hide-sm"><abbr title="{_esc(_PI_TIP)}">상승 확률</abbr></th>'
                 f'<th class="th-score"><abbr title="{_esc(_SCORE_TIP)}">상승 점수</abbr></th></tr></thead>'
                 f'<tbody>{"".join(rows_html)}</tbody></table></div>')
    else:
        table = '<p class="empty">순위 데이터가 없습니다.</p>'

    ordered: list[TrendReport] = []
    seen: set[int] = set()
    for nm in names:
        r = by_name.get(nm)
        if r is not None and id(r) not in seen:
            ordered.append(r)
            seen.add(id(r))
    for r in reports:
        if id(r) not in seen:
            ordered.append(r)
            seen.add(id(r))

    cards = []
    rank_of = {nm: i for i, nm in enumerate(names, 1)}
    for r in ordered:
        a, o, m = r.analysis, r.outlook, r.analysis.metrics
        freq = str(a.freq)
        dd = dd_of(r)
        rank = rank_of.get(str(r.name))
        score = scores.get(str(r.name))
        end_fc = _fmt_date(dd.fc_dates[-1], freq) if len(dd.fc_dates) else _fmt_date(m.end, freq)
        fc_sp, rc_sp = _change_spans(r)
        rc_k = f'최근 변화 <span class="nw">({_esc(rc_sp)})</span>' if rc_sp else "최근 변화"
        fc_k = f'예측 변화 <span class="nw">({_esc(fc_sp)})</span>' if fc_sp else "예측 변화"
        ec_html, ec_note = change_html(r, o.expected_change_pct)
        cards.append(
            '<article class="mcard"><div class="top">'
            f'<div><h3>{_link(_esc(r.name), links.get(str(r.name)))}</h3></div>'
            f'<span class="rank">{"#" + str(rank) if rank else ""}</span></div>'
            f'<div>{_badge(str(a.phase.label), _finite(a.phase.confidence), _tentative(r))}</div>'
            f"{_mini_chart(r)}"
            '<div class="stats">'
            f'<div><div class="k">{rc_k}</div><div class="v">{_signed_html(m.recent_change_pct, _fmt_pct(m.recent_change_pct))}</div></div>'
            f'<div><div class="k">{fc_k}</div><div class="v">{ec_html}</div>{ec_note}</div>'
            f'<div title="{_esc(_PI_TIP)}"><div class="k">상승 확률</div><div class="v">{_esc(_fmt_prob(o.prob_increase))}</div></div>'
            "</div>"
            f'<div class="foot"><span>{_esc(_fmt_date(m.start, freq))} ~ {_esc(end_fc)}</span>'
            f'<span title="{_esc(_SCORE_TIP)}">상승 점수 <strong>{_esc(_fmt_num(score, 0))}</strong></span></div>'
            "</article>")
    cards_html = f'<div class="cards">{"".join(cards)}</div>' if cards else '<p class="empty">리포트가 없습니다.</p>'

    n = len(ordered) if ordered else len(names)
    freqs = {str(getattr(r.analysis, "freq", "")) for r in ordered}
    rank_desc = ("상승 점수는 앞으로의 전망(예측 변화·상승 확률)과 최근 기세(최근 변화·모멘텀)를 합친 0–100 점수로, "
                 "50이 중립입니다. 안정기·변동성 장세에서는 전망만 반영합니다.")
    if len(freqs) > 1:
        rank_desc += (" 주기가 다른 시계열이 섞여 있어, 점수는 변화율을 연 단위(모멘텀은 월 단위)로 환산해 계산했고 "
                      "표의 변화율은 값 옆에 적힌 각 시리즈의 기간 기준입니다.")
    if links:
        rank_desc += " 이름을 누르면 시리즈별 상세 리포트가 열립니다."
    from . import __version__
    body = "\n".join([
        '<header class="hero"><div class="eyebrow"><span class="logo" aria-hidden="true"></span>'
        'TrendAI 비교 리포트 · 어떤 트렌드가 뜨고 있나</div>'
        f"<h1>트렌드 비교 · {n}개 시리즈</h1>"
        '<div class="meta"><span class="chip">상승 점수 0–100 (50 = 중립)</span>'
        '<span class="chip">전망(예측 변화 · 상승 확률) + 최근 기세(최근 변화 · 모멘텀)</span></div>'
        f'<p class="gen">TrendAI {_esc(__version__)} 버전으로 생성</p></header>',
        f'<section class="card"><h2>상승 순위</h2><p class="desc">{_esc(rank_desc)}</p>{table}</section>',
        '<section><h2 style="font-size:17px;margin:4px 0 12px">시리즈별 요약</h2>'
        '<p class="desc">파란 선은 실제값, 주황 선과 음영은 예측과 예측 구간입니다.</p>'
        f"{cards_html}</section>",
        '<footer>이 비교는 과거 데이터의 통계적 패턴에 기반한 추정이며 미래를 보장하지 않습니다.</footer>',
    ])
    page = _page(f"트렌드 비교 — {n}개 시리즈", body, extra_css=_SVG_STYLE + _COMPARE_CSS)
    _write(page, path)
    return page
