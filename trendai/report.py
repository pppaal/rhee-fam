"""Presentation: self-contained HTML reports, inline SVG charts and terminal summaries.

Public API (see ``docs/ARCHITECTURE.md``)::

    sparkline(values, width=None) -> str          # ▁▂▃▄▅▆▇█
    svg_chart(report, width=960, height=380) -> str
    render_text(report, width=78) -> str
    render_html(report, path=None) -> str
    render_compare_html(reports, ranking, path=None) -> str

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
import unicodedata
from pathlib import Path
from collections.abc import Mapping
from typing import Any, Iterable, Optional, Sequence, Union

import numpy as np
import pandas as pd

from .data import FREQ_KO
from .data import season_length as _season_length
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

MODEL_KO: dict[str, str] = {
    "naive": "단순 예측(직전값)",
    "seasonal_naive": "계절 단순 예측",
    "drift": "드리프트(평균 기울기)",
    "ses": "단순 지수평활",
    "holt": "Holt 선형 추세",
    "holt_damped": "감쇠 추세 Holt",
    "holt_winters": "Holt-Winters 계절 모형",
    "theta": "Theta 방법",
    "linear_trend": "선형 추세 + 계절 항",
    "ar": "자기회귀(AR)",
    "lag_ridge": "릿지 회귀(머신러닝)",
    "analog_knn": "과거 유사 패턴(kNN)",
    "logistic_growth": "로지스틱 S-커브",
}

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
# acceleration / deceleration only say "steeper" / "flatter": the direction comes from the
# slopes.  Same wording as narrative._changepoint_line, coloured by meaning for the series.
_CP_DIRECTED: dict[tuple[str, bool], tuple[str, str]] = {
    ("acceleration", True): ("성장 가속", "#0ca30c"),
    ("acceleration", False): ("하락 가속", "#d03b3b"),
    ("deceleration", True): ("성장 둔화", "#eda100"),
    ("deceleration", False): ("하락세 둔화", "#1baf7a"),
}

#: the benchmarks behind ForecastResult.skill
_NAIVE_NAMES: tuple[str, ...] = ("naive", "seasonal_naive", "drift")
_SKILL_LABEL = "단순 예측 대비 오차"
_SKILL_TIP = ("같은 백테스트 구간에서 가장 나은 단순 기준 예측(직전값 반복·계절 반복·평균 기울기 연장)의 "
              "평균 절대 오차와 비교한 최종 예측의 오차 증감입니다. 음수일수록 단순 예측보다 정확합니다.")
_MASE_TIP = ("MASE: 예측 오차를 학습 구간에서 한 기 앞을 계절 단순 예측한 오차로 나눈 값(낮을수록 좋음). "
             "여러 기 앞을 내다보는 예측은 단순 예측보다 정확해도 1을 넘기 쉬우므로, 단순 예측과의 비교는 "
             "‘단순 대비’ 열을 보세요.")
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


def _auto_decimals(a: float) -> int:
    if a >= 100:
        return 0 if a >= 1000 else 1
    if a >= 10:
        return 1
    if a >= 1:
        return 2
    if a == 0:
        return 0
    return int(min(8, max(2, 1 - math.floor(math.log10(a)))))


def _sci(a: float, digits: int = 3) -> str:
    """Compact scientific notation of a non-negative number: ``1.23e−9``, ``4e16``."""
    s = f"{a:.{digits}g}"
    if "e" in s:
        mant, ex = s.split("e")
        s = f"{mant}e{int(ex)}".replace("-", _MINUS)
    return s


def _fmt_num(x: Any, decimals: Optional[int] = None, signed: bool = False) -> str:
    """Thousands separators, magnitude-dependent decimals, typographic minus; ``—`` for None.

    Magnitudes ``>= 1e15`` and (with automatic decimals) tiny magnitudes
    ``< 1e-6`` use scientific notation, so tiny values never print as ``0.00000000``.
    """
    v = _finite(x)
    if v is None:
        return _DASH
    a = abs(v)
    if a >= 1e15 or (decimals is None and 0 < a < 1e-6):
        body = _sci(a)
    else:
        d = _auto_decimals(a) if decimals is None else max(0, int(decimals))
        body = f"{a:,.{d}f}"
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
    """Probability (0..1) as a whole percentage; ``>99%`` / ``<1%`` near (but not at) the bounds."""
    v = _finite(p)
    if v is None:
        return _DASH
    v = min(1.0, max(0.0, v))
    if 0.995 <= v < 1.0:
        return ">99%"
    if 0.0 < v < 0.005:
        return "<1%"
    return f"{v * 100:.0f}%"


def _fmt_short(x: Any) -> str:
    """Like :func:`_fmt_num` but abbreviates magnitudes ``>= 1e6`` (``1.23M``, ``45.6B``, ``1.50T``).

    Used for big-number tiles whose width is limited (KPI tiles, scenario boxes);
    the full value is still shown in tables and ``title`` tooltips.
    """
    v = _finite(x)
    if v is None:
        return _DASH
    a = abs(v)
    for unit, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M")):
        # 0.9995 so that e.g. 999.7M becomes "1.00B" instead of "1,000M"
        if a >= unit * 0.9995 and (a < 1e15 or suffix == "T"):
            q = a / unit
            body = f"{q:,.{2 if q < 9.995 else 1 if q < 99.95 else 0}f}{suffix}"
            return (_MINUS if v < 0 else "") + body
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


def _model_desc(name: str) -> str:
    base, _, variant = str(name).partition("@")
    desc = MODEL_KO.get(base, "")
    if desc and variant == "log":
        desc += " (로그 공간)"
    return desc


def _cp_kind(cp: Any) -> tuple[str, str]:
    """Korean label and colour of a changepoint, with the direction for (de)acceleration.

    ``acceleration`` with a falling slope is a *decline* accelerating (red), not
    growth; ``deceleration`` of a decline is the decline easing.  Matches the
    wording of the rule-based narrative.  Other kinds use :data:`CHANGEPOINT_KO`.
    """
    kind = str(getattr(cp, "kind", ""))
    before = _finite(getattr(cp, "slope_before", None))
    after = _finite(getattr(cp, "slope_after", None))
    if kind == "acceleration" and after is not None:
        return _CP_DIRECTED[(kind, after >= 0)]
    if kind == "deceleration" and before is not None:
        return _CP_DIRECTED[(kind, before > 0)]
    return CHANGEPOINT_KO.get(kind, kind), KIND_COLOR.get(kind, "#898781")


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
    """``pos`` when clearly better than naive (lower error), ``neg`` when worse, else ``''``."""
    if ratio is None:
        return ""
    return "pos" if ratio <= 0.95 else ("neg" if ratio > 1.0 else "")


def _skill_sentence(skill: Optional[float]) -> str:
    """Plain-Korean verdict of the final method versus the naive benchmarks (same thresholds as the narrative)."""
    if skill is None:
        return "같은 백테스트 구간에서 단순 기준 예측과 비교한 결과가 없어, 단순 예측보다 나은지는 판단하지 않았습니다."
    if skill <= 0.95:
        gain = (1.0 - skill) * 100.0
        g = "99% 이상" if gain > 99.5 else f"약 {gain:.0f}%"
        return f"최종 방법의 백테스트 오차는 가장 나은 단순 기준 예측보다 {g} 작습니다."
    if skill <= 1.0:
        return "최종 방법의 백테스트 오차가 가장 나은 단순 기준 예측과 비슷합니다(차이 5% 이내)."
    worse = (skill - 1.0) * 100.0
    size = f"약 {skill:.1f}배" if skill >= 2.0 else f"약 {worse:.0f}% 더 큼"
    return f"최종 방법은 백테스트에서 단순 기준 예측보다 낫지 않았습니다(오차 {size})."


def _best_naive_mae(scores: Sequence[Any]) -> tuple[Optional[str], Optional[float]]:
    """Name and MAE of the most accurate naive benchmark among the leaderboard rows."""
    best: tuple[Optional[str], Optional[float]] = (None, None)
    for s in scores:
        name = str(getattr(s, "name", ""))
        mae = _finite(getattr(s, "mae", None))
        if name in _NAIVE_NAMES and mae is not None and mae >= 0 and (best[1] is None or mae < best[1]):
            best = (name, mae)
    return best


def _model_html(name: Any) -> str:
    """Escaped model name with line-break opportunities after ``_`` and before ``@`` only."""
    e = _esc(name)
    return e.replace("_", "_<wbr>").replace("@", "<wbr>@")


def _change_basis(report: TrendReport, d: "_ChartData") -> tuple[str, str]:
    """The two levels behind ``Outlook.expected_change_pct`` (see ``forecast.compute_outlook``).

    Future level = mean of the last ``min(season, horizon)`` forecast points,
    current level = mean of the last ``min(season, n)`` observations.  Returns
    Korean phrases such as ``('향후 12개월 평균', '최근 12개월 평균')``; both end in
    a consonant, so they take the particles 을/과.
    """
    freq = d.freq
    unit = PERIOD_KO.get(freq, "기간")
    m = max(1, int(_finite(getattr(report.analysis, "season_length", 1)) or 1))
    n = max(1, int(np.isfinite(d.hist_y).sum()))
    mean = getattr(report.forecast, "mean", None)
    h = len(mean) if isinstance(mean, pd.Series) and len(mean) else max(1, int(_finite(report.forecast.horizon) or 1))
    k_fut, k_cur = min(m, h), min(m, n)
    if k_fut <= 1:
        end = _fmt_date(d.fc_dates[-1], freq) if len(d.fc_dates) else ""
        fut = f"{end} 예측값".strip()
    elif k_fut >= h:
        fut = f"향후 {h:,}{unit} 평균"
    else:
        fut = f"예측 마지막 {k_fut:,}{unit} 평균"
    cur = "최근 값" if k_cur <= 1 else f"최근 {k_cur:,}{unit} 평균"
    return fut, cur


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
    idx = np.clip(np.rint((arr - lo) / span * (len(_BLOCKS) - 1)), 0, len(_BLOCKS) - 1).astype(int)
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


def _nice_ticks(lo: float, hi: float, min_n: int = 4, max_n: int = 6) -> list[float]:
    """4–6 "nice" ticks covering ``[lo, hi]``.

    Steps are 1, 2, 2.5, 5 × 10^k (3 and 4 × 10^k only when they fit much tighter);
    the step that wastes the least vertical space wins.
    """
    if not (math.isfinite(lo) and math.isfinite(hi)):
        lo, hi = 0.0, 1.0
    if hi < lo:
        lo, hi = hi, lo
    # keep 10**exp and (hi - lo) representable
    lo, hi = max(lo, -1e300), min(hi, 1e300)
    if hi - lo <= 1e-12 * max(abs(lo), abs(hi)):  # flat (relative test; also lo == hi == 0)
        pad = abs(hi) * 0.1 if hi != 0 else 1.0
        lo, hi = lo - pad, hi + pad
    span = hi - lo
    e = math.floor(math.log10(span / max(1, max_n - 1)))
    best: Optional[tuple[tuple[float, float], float, float, int]] = None
    for exp in range(e - 1, e + 3):
        for mant, penalty in ((1.0, 0.0), (2.0, 0.0), (2.5, 0.03), (3.0, 0.1), (4.0, 0.1), (5.0, 0.0)):
            step = mant * 10.0 ** exp
            a = math.floor(lo / step + 1e-9) * step
            b = math.ceil(hi / step - 1e-9) * step
            n = int(round((b - a) / step)) + 1
            if n < min_n or n > max_n:
                continue
            key = (span / (b - a) - penalty, -abs(n - 5))
            if best is None or key > best[0]:
                best = (key, a, step, n)
    if best is None:  # pragma: no cover - the grid above always contains a fit
        step = span / 4.0
        best = ((0.0, 0.0), lo, step, 5)
    _, a, step, n = best
    dec = max(0, -math.floor(math.log10(step)) + 2)
    return [round(a + i * step, dec) + 0.0 for i in range(n)]


def _axis_labels(ticks: Sequence[float]) -> list[str]:
    """Tick labels: thousands separators below 10,000, k/M/B/T abbreviations above.

    Labels are always distinct: when the ticks would need more than 6 decimals
    (tiny magnitudes) they switch to compact scientific notation (``5e−8``).
    """
    if not ticks:
        return []
    m = max(abs(t) for t in ticks)
    unit, suffix = 1.0, ""
    for u, sfx in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e4, "k")):
        if m >= u:
            unit, suffix = (1e3 if sfx == "k" else u), sfx
            break
    vals = [t / unit for t in ticks]
    diffs = [abs(b - a) for a, b in zip(vals, vals[1:]) if b != a]
    tol = 1e-6 * (min(diffs) if diffs else max(1.0, abs(vals[0])))
    out: Optional[list[str]] = None
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
        for digits in range(2, 16):
            out = ["0" if t == 0 else (_MINUS if t < 0 else "") + _sci(abs(t), digits) for t in ticks]
            if len(set(out)) == len(out):
                break
    return out


def _fmt_axis_value(v: float, ref: float) -> str:
    """A single value formatted like the axis (abbreviated when the axis is)."""
    a = abs(ref)
    if a >= 1e12:
        return _fmt_num(v / 1e12, decimals=2) + "T"
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
        return plot_b - (np.asarray(v, dtype=float) - ymin) / (ymax - ymin) * plot_h

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
    pts: list[tuple[float, str, list[tuple[str, str]], float, str]] = []
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
        pts.append((float(d.hist_t[i]), _fmt_date(d.hist_dates[i], freq), rows, v, "tai-dot-h"))
    for i in range(len(d.fc_t)):
        v = float(d.fc_y[i])
        rows = [("예측", _fmt_axis_value(v, ref))]
        for lv, blo, bhi in sorted(d.bands, key=lambda b: b[0]):
            rows.append((f"{lv}%", f"{_fmt_axis_value(blo[i], ref)} ~ {_fmt_axis_value(bhi[i], ref)}"))
        pts.append((float(d.fc_t[i]), _fmt_date(d.fc_dates[i], freq) + " · 예측", rows, v, "tai-dot-f"))
    if not pts:
        return ""
    pts.sort(key=lambda p: p[0])
    if len(pts) > max_cols:
        keep = np.unique(np.linspace(0, len(pts) - 1, max_cols).round().astype(int))
        pts = [pts[k] for k in keep]
    xs = [float(X(p[0])) for p in pts]
    out = ['<g class="tai-hover">']
    line_h = fs + 5
    for k, (_t, date_s, rows, v, dot_cls) in enumerate(pts):
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
        tip_txt = date_s + " " + ", ".join(f"{a} {b}" for a, b in rows)
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
    span = hi - lo if hi > lo else 1.0

    def X(t: Any) -> Any:
        return pl + np.asarray(t, dtype=float) / d.t_end * (pr - pl)

    def Y(v: Any) -> Any:
        return pb - (np.asarray(v, dtype=float) - lo) / span * (pb - pt)

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


def render_text(report: TrendReport, width: int = 78) -> str:
    """Compact terminal summary of ``report`` (every line fits in ``width`` columns).

    Shows the name, lifecycle phase, key metrics, a sparkline of the history
    and the forecast separated by ``│``, the outlook and the top models.
    Widths below 10 are treated as 10.
    """
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
    head = [f"[{_term(_phase_ko(phase.label))}]" + (f" 신뢰도 {conf * 100:.0f}%" if conf is not None else ""),
            f"{UNIT_KO.get(freq, FREQ_KO.get(freq, freq))} 단위 {len(d.hist_y)}개",
            f"{_fmt_date(m.start, freq)} ~ {_fmt_date(m.end, freq)}"]
    lines += _wrap_segments(head, width)
    lines.append(light)

    metrics = [
        f"현재 값 {_fmt_num(m.last_value)}",
        f"전체 변화 {_fmt_pct(m.total_change_pct)}",
        f"CAGR {_fmt_pct(m.cagr_pct)}",
        f"최근 변화 {_fmt_pct(m.recent_change_pct)}",
        f"모멘텀 {_fmt_num(m.momentum, 2, signed=True)}"
        + (f"%p{_PER_UNIT.get(freq, '')}" if _finite(m.momentum) is not None else ""),
        f"변동성 {_fmt_pct((_finite(m.volatility) or 0.0) * 100 if _finite(m.volatility) is not None else None, signed=False)}",
    ]
    lines += _wrap_segments(metrics, width)
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
    fut_basis, cur_basis = _change_basis(report, d)
    out_seg = [f"전망 {DIRECTION_ARROW.get(direction, '')} {DIRECTION_KO.get(direction, _term(direction))}".replace("  ", " "),
               f"예측 변화 {_fmt_pct(o.expected_change_pct)} ({fut_basis}, {cur_basis} 대비)",
               f"상승 확률 {_fmt_prob(o.prob_increase)}",
               f"신뢰도 {CONFIDENCE_KO.get(str(o.confidence), _term(o.confidence))}"]
    lines += _wrap_segments(out_seg, width)
    scen = o.scenarios or {}
    if scen:
        end_s = _fmt_date(d.fc_dates[-1], freq) if len(d.fc_dates) else ""
        sc = [f"{ko} {_fmt_num(scen.get(k))}" for k, ko in SCENARIOS_KO if k in scen]
        lines += _wrap_segments([f"시나리오({end_s})" if end_s else "시나리오"] + sc, width, sep="  ")
    extra = [f"미래 단계 {_term(_phase_ko(o.future_phase))}"]
    if _ts(o.forecast_peak_date) is not None:
        extra.append(f"예측 정점 {_fmt_date(o.forecast_peak_date, freq)}")
    lines += _wrap_segments(extra, width)
    cps = list(a.changepoints or [])
    if cps:
        segs = [f"{_fmt_date(cp.date, freq)} {_term(_cp_kind(cp)[0])}" for cp in cps[:4]]
        if len(cps) > 4:
            segs.append(f"외 {len(cps) - 4}개")
        segs[0] = "변곡점: " + segs[0]
        lines += _wrap_segments(segs, width)
    lines.append(light)
    method = _term(fc.method)
    bm = _finite(fc.backtest_mase)
    lines.append(_fit(f"방법: {method}", width))
    skill = _skill_value(fc)
    acc = [f"정확도: {_SKILL_LABEL} {_rel_error_text(skill)}"]
    if skill is not None:
        acc.append("단순 예측보다 정확" if skill <= 0.95 else ("단순 예측과 비슷" if skill <= 1.0
                                                             else "단순 예측보다 부정확"))
    if bm is not None:
        acc.append(f"MASE {bm:.2f}")
    lines += _wrap_segments(acc, width)
    scores = list(fc.model_scores or [])[:3]
    if scores:
        segs = [f"{_term(s.name)} {_fmt_num(s.mase, 2)}" for s in scores]
        lines += _wrap_segments(["상위 모델(MASE):"] + segs, width, sep="  ")
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
.dir{display:flex;align-items:baseline;gap:10px;margin:2px 0 14px}
.dir .big{font-size:26px;font-weight:700;letter-spacing:-.02em}
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


def _badge(label: str, conf: Optional[float] = None) -> str:
    color = PHASE_COLOR.get(label, "#898781")
    conf_html = f" <small>신뢰도 {conf * 100:.0f}%</small>" if conf is not None else ""
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


def _current_level(report: TrendReport) -> Optional[float]:
    y = np.asarray(report.analysis.series, dtype=float)
    y = y[np.isfinite(y)]
    if y.size == 0:
        return None
    k = max(1, min(int(getattr(report.analysis, "season_length", 1) or 1), y.size))
    return float(y[-k:].mean())


# --------------------------------------------------------------------------- HTML sections


def _header_html(report: TrendReport, d: _ChartData) -> str:
    a = report.analysis
    m = a.metrics
    freq = d.freq
    conf = _finite(getattr(a.phase, "confidence", None))
    src = NARRATIVE_SOURCE_KO.get(str(report.narrative_source), "규칙 기반 해석")
    from . import __version__

    chips = [
        _badge(str(a.phase.label), conf),
        f'<span class="chip">기간 {_esc(_fmt_date(m.start, freq))} ~ {_esc(_fmt_date(m.end, freq))}</span>',
        f'<span class="chip">{_esc(UNIT_KO.get(freq, FREQ_KO.get(freq, freq)))} 단위 · 관측 {len(d.hist_y):,}개</span>',
        f'<span class="chip">예측 {int(report.forecast.horizon):,}{_esc(PERIOD_KO.get(freq, "기간"))}</span>',
    ]
    return (
        '<header class="hero">'
        '<div class="eyebrow"><span class="logo" aria-hidden="true"></span>TrendAI 트렌드 리포트 · 과거 → 현재 → 미래</div>'
        f"<h1>{_esc(report.name)}</h1>"
        f'<div class="meta">{"".join(chips)}</div>'
        f'<p class="gen">TrendAI {_esc(__version__)} 버전으로 생성 · 통계 모델 앙상블 예측 · {_esc(src)}</p>'
        "</header>"
    )


def _kpis_html(report: TrendReport, d: _ChartData) -> str:
    m, o = report.analysis.metrics, report.outlook
    freq = d.freq
    unit = PERIOD_KO.get(freq, "기간")
    fut_basis, cur_basis = _change_basis(report, d)
    # analysis.compute_metrics: lag = data.season_length(freq) (1 for yearly), capped at n - 1
    try:
        season = _season_length(freq)
    except ValueError:  # unknown freq code: fall back to what the analysis recorded
        season = int(getattr(report.analysis, "season_length", 1) or 1)
    lag = max(1, min(1 if freq == "Y" else season, len(d.hist_y) - 1))
    recent_span = f"최근 {lag}{unit} 추세 변화"
    dd = _finite(m.drawdown_pct)
    peak_sub = (f"최고점 대비 {_fmt_pct(dd)}" if dd is not None and dd < -0.05
                else "현재 최고 수준")
    p = _finite(o.prob_increase)
    meter = (f'<div class="meter" role="img" aria-label="상승 확률 {_fmt_prob(p)}">'
             f'<span style="width:{(min(1.0, max(0.0, p)) * 100) if p is not None else 0:.0f}%"></span></div>')
    tiles = [
        _kpi("현재 값", f"<span>{_esc(_fmt_short(m.last_value))}</span>",
             f'<span class="nw">{_esc(_fmt_date(m.end, freq))} 기준</span> · {_esc(peak_sub)}',
             title=_fmt_num(m.last_value)),
        _kpi("전체 변화", _signed_html(m.total_change_pct, _fmt_pct(m.total_change_pct)),
             f'<span class="nw">{_esc(_fmt_date(m.start, freq))} 이후</span> (추세 기준)'),
        _kpi("연평균 성장률 CAGR", _signed_html(m.cagr_pct, _fmt_pct(m.cagr_pct)), "연 환산 복리 성장률"),
        _kpi("최근 변화", _signed_html(m.recent_change_pct, _fmt_pct(m.recent_change_pct)), _esc(recent_span)),
        _kpi("예측 변화", _signed_html(o.expected_change_pct, _fmt_pct(o.expected_change_pct)),
             f'{_esc(fut_basis)} · <span class="nw">{_esc(cur_basis)} 대비</span>'),
        _kpi("상승 확률", f"<span>{_esc(_fmt_prob(p))}</span>",
             f"예측 신뢰도 {_esc(CONFIDENCE_KO.get(str(o.confidence), str(o.confidence)))}", extra=meter),
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
    o, fc = report.outlook, report.forecast
    freq = d.freq
    unit = PERIOD_KO.get(freq, "기간")
    direction = str(o.direction)
    dir_ko = DIRECTION_KO.get(direction, direction)
    cls = {"up": "pos", "down": "neg"}.get(direction, "")
    end_fc = _fmt_date(d.fc_dates[-1], freq) if len(d.fc_dates) else _DASH
    scen = o.scenarios or {}
    sc_html = "".join(
        f'<div class="{"base" if k == "base" else ""}"><div class="k">{ko}</div>'
        f'<div class="v" title="{_esc(_fmt_num(scen.get(k)))}">{_esc(_fmt_short(scen.get(k)))}</div></div>'
        for k, ko in SCENARIOS_KO)
    # range visual: pessimistic..optimistic band, base marker, current level tick
    cur = _current_level(report)
    vals = [v for v in (_finite(scen.get("pessimistic")), _finite(scen.get("optimistic")),
                        _finite(scen.get("base")), cur) if v is not None]
    range_html = ""
    if len(vals) >= 2 and max(vals) > min(vals):
        lo, hi = min(vals), max(vals)
        pad = (hi - lo) * 0.08
        lo, hi = lo - pad, hi + pad

        def pos(v: float) -> float:
            return (v - lo) / (hi - lo) * 100.0

        segs = ['<div class="rt"></div>']
        ps, op = _finite(scen.get("pessimistic")), _finite(scen.get("optimistic"))
        if ps is not None and op is not None:
            a, b = sorted((pos(ps), pos(op)))
            segs.append(f'<div class="rb" style="left:{a:.1f}%;width:{max(b - a, 0.5):.1f}%"></div>')
        bs = _finite(scen.get("base"))
        if bs is not None:
            segs.append(f'<div class="rm" style="left:{pos(bs):.1f}%" title="기준 시나리오"></div>')
        if cur is not None:
            pc = pos(cur)
            segs.append(f'<div class="rc" style="left:{pc:.1f}%" title="현재 수준"></div>')
            lab_pos = min(88.0, max(12.0, pc))
            segs.append(f'<div class="rl" style="left:{lab_pos:.1f}%">현재 수준 {_esc(_fmt_short(cur))}</div>')
        range_html = (f'<div class="range" role="img" aria-label="시나리오 범위와 현재 수준">{"".join(segs)}</div>')
    peak = _ts(o.forecast_peak_date)
    skill = _skill_value(fc)
    fut_basis, cur_basis = _change_basis(report, d)
    facts: list[tuple[str, str, str, str]] = [  # (label, value, tooltip, value class)
        ("상승 확률", _fmt_prob(o.prob_increase), "", ""),
        ("예측 변화", _fmt_pct(o.expected_change_pct), f"{fut_basis}을 {cur_basis}과 비교한 변화율", ""),
        ("미래 단계", _phase_ko(o.future_phase), "", ""),
        ("예측 신뢰도", CONFIDENCE_KO.get(str(o.confidence), str(o.confidence)), "", ""),
        (_SKILL_LABEL, _rel_error_text(skill), _SKILL_TIP, _rel_error_class(skill)),
        ("예측 정점", _fmt_date(peak, freq) if peak is not None else "예측 기간 내 없음", "", ""),
    ]
    facts_html = "".join(
        (f'<dt><abbr title="{_esc(tip)}">{_esc(k)}</abbr></dt>' if tip else f"<dt>{_esc(k)}</dt>")
        + (f'<dd class="{vc}">{_esc(v)}</dd>' if vc else f"<dd>{_esc(v)}</dd>")
        for k, v, tip, vc in facts)
    arrow = DIRECTION_ARROW.get(direction, "")
    return (
        '<section class="card"><h2>미래 전망</h2>'
        f'<p class="desc">{_esc(end_fc)}까지 · 향후 {int(fc.horizon):,}{_esc(unit)}</p>'
        f'<div class="dir"><span class="big {cls}">{_esc(arrow)} {_esc(dir_ko)}</span>'
        f'<span class="lab">{_esc(_fmt_pct(o.expected_change_pct))} 예상</span></div>'
        f'<p class="desc" style="margin:-8px 0 14px">변화율은 {_esc(fut_basis)}을 {_esc(cur_basis)}과 비교한 값입니다.</p>'
        f'<h3 class="sub">시나리오 ({_esc(end_fc)})</h3><div class="scen">{sc_html}</div>{range_html}'
        f'<dl class="facts">{facts_html}</dl></section>'
    )


def _diagnosis_card(report: TrendReport, d: _ChartData) -> str:
    a = report.analysis
    m = a.metrics
    freq = d.freq
    conf = _finite(getattr(a.phase, "confidence", None))
    reasons = [str(r) for r in (a.phase.reasons or []) if str(r).strip()]
    # the rule-based narrative quotes the reasons as "근거:" bullets in the card next to this one
    narr = " ".join(str(report.narrative or "").split())
    shown = [r for r in reasons if " ".join(r.split()) not in narr]
    reasons_html = ("<ul class=\"reasons\">" + "".join(f"<li>{_esc(r)}</li>" for r in shown) + "</ul>"
                    if shown else "")
    desc = "수명주기 단계와 판단 근거"
    if reasons and len(shown) < len(reasons):
        desc = ("수명주기 단계와 주요 지표 · 판단 근거는 ‘AI 트렌드 해석’의 현재 항목에 있습니다" if not shown
                else "수명주기 단계와 판단 근거 · 나머지 근거는 ‘AI 트렌드 해석’에 있습니다")
    vol = _finite(m.volatility)
    mom = _fmt_num(m.momentum, 2, signed=True)
    facts = [
        # momentum = recent slope - long-run slope, both in % of the mean level per period
        ("모멘텀", mom + (f"%p{_PER_UNIT.get(freq, '')}" if mom != _DASH else "")),
        ("추세 강도", _fmt_num(m.trend_strength, 2)),
        ("계절성 강도", _fmt_num(m.seasonal_strength, 2)),
        ("변동성", _fmt_pct(vol * 100 if vol is not None else None, signed=False)),
        ("최고점", f"{_fmt_short(m.peak_value)} ({_fmt_date(m.peak_date, freq)})"),
    ]
    facts_html = "".join(f"<dt>{_esc(k)}</dt><dd>{_esc(v)}</dd>" for k, v in facts)
    return (
        '<section class="card"><h2>현재 진단</h2>'
        f'<p class="desc">{_esc(desc)}</p>{_badge(str(a.phase.label), conf)}{reasons_html}'
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
        an_html = ('<div class="table-wrap"><table><thead><tr><th>날짜</th><th class="num">실제</th>'
                   '<th class="num">예상</th><th class="num">z-점수</th></tr></thead>'
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
    if scores:
        max_w = max([_finite(s.weight) or 0.0 for s in scores] + [0.0])
        # per-model error versus the best naive benchmark on the same backtest windows
        # (the same ratio as ForecastResult.skill, row by row)
        naive_name, naive_mae = _best_naive_mae(scores)
        rows = []
        for i, s in enumerate(scores, 1):
            w = _finite(s.weight) or 0.0
            desc = _model_desc(s.name)
            fill = (w / max_w * 100) if max_w > 0 else 0.0
            mae = _finite(s.mae)
            if naive_mae is None or mae is None or mae < 0:
                rel_html = _DASH
            elif str(s.name) == naive_name:
                rel_html = '<span class="muted">기준</span>'
            elif naive_mae > 0:
                ratio = mae / naive_mae
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
                f'<td><div class="wbar"><div class="track"><div class="fill" style="width:{fill:.1f}%"></div></div>'
                f'<span class="pct">{w * 100:.0f}%</span></div></td></tr>')
        naive_ko = MODEL_KO.get(naive_name or "", "")
        rel_tip = ("같은 백테스트 구간에서 가장 나은 단순 기준 예측"
                   + (f"({naive_name}: {naive_ko})" if naive_name else "")
                   + "보다 평균 절대 오차가 몇 % 작은지(−) 큰지(+)")
        table = ('<div class="table-wrap"><table><thead><tr><th>#</th><th>모델</th>'
                 f'<th class="num"><abbr title="{_esc(_MASE_TIP)}">MASE</abbr></th>'
                 f'<th class="num"><abbr title="{_esc(rel_tip)}">단순 대비</abbr></th>'
                 '<th class="num hide-sm">sMAPE</th><th>앙상블 가중치</th></tr></thead>'
                 f'<tbody>{"".join(rows)}</tbody></table></div>')
    else:
        table = '<p class="empty">백테스트 점수가 없습니다 (데이터가 짧아 단순 모델을 사용).</p>'
    bm = _finite(fc.backtest_mase)
    if bm is None and skill is None:
        acc_txt = "데이터가 짧아 백테스트로 정확도를 검증하지 못했습니다."
    else:
        acc_txt = _skill_sentence(skill)
    skill_html = (f'<abbr title="{_esc(_SKILL_TIP)}">{_SKILL_LABEL}</abbr> '
                  f'<strong class="{_rel_error_class(skill)}">{_esc(_rel_error_text(skill))}</strong>')
    mase_note = ""
    if bm is not None:
        mase_note = (f"기술 참고: 최종 방법의 백테스트 MASE {bm:.2f}. MASE는 학습 구간에서 한 기 앞을 계절 단순 예측한 "
                     "오차를 1로 둔 상대 오차라, 여러 기 앞을 내다보는 예측은 단순 예측보다 정확해도 1을 넘기 쉬우므로 "
                     "단순 예측과의 우열은 위의 ‘단순 대비’로 판단합니다.")
    return (
        '<section class="card"><h2>모델 리더보드</h2>'
        '<p class="desc">롤링 원점 백테스트 성적(MASE 낮을수록 좋음). 왼쪽 강조선은 최종 앙상블에 포함된 모델입니다.</p>'
        f'{table}<p class="desc" style="margin:14px 0 0">최종 방법 <code>{_esc(fc.method)}</code></p>'
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
    season_txt = (f"계절 주기 {period}{unit}의 계절 성분을 분리하고" if period > 1
                  else "계절 주기가 없거나 데이터가 짧아 계절 성분 없이")
    levels = ", ".join(f"{lv}%" for lv in sorted({lv for lv, _, _ in d.bands})) or "—"
    n_models = len(fc.model_scores or [])
    notes = [
        ("시계열 분해", f"강건한 가법 분해(STL 유사)로 {season_txt} 국소 선형 평활로 추세를 추정했습니다. "
                    "증감률·CAGR 같은 성장 지표는 잡음을 줄이기 위해 추세 성분으로 계산합니다."),
        ("변곡점 · 이상치", "계절 조정된 시계열에 구간별 선형 회귀(BIC 유사 벌점)를 적용해 기울기·수준이 바뀐 시점을 찾고, "
                      "잔차의 강건 z-점수(중앙값·MAD)가 큰 점을 이상치로 표시합니다."),
        ("앙상블 백테스트", f"후보 {n_models}개 모델을 롤링 원점 백테스트로 평가해 MASE 순으로 정렬하고, "
                     "상위 모델을 역-MASE 가중 평균한 앙상블(또는 확실히 우수한 단일 모델)을 최종 예측으로 씁니다."),
        ("예측 구간", f"백테스트에서 관측된 예측 오차의 경험적 분포로 시점별 {levels} 예측 구간을 추정하며, "
                  "예측 시점이 멀수록 구간이 넓어집니다."),
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


def render_compare_html(reports: list[TrendReport], ranking: pd.DataFrame,
                        path: Optional[PathLike] = None) -> str:
    """Comparison page: ranking table + a grid of small-multiple cards.

    ``ranking`` has the columns ``name, phase, momentum, recent_change_pct,
    expected_change_pct, prob_increase, score`` (best first); extra columns are
    ignored and missing values are shown as ``—``.  Cards follow the ranking
    order; reports not present in ``ranking`` are appended at the end.
    """
    if isinstance(reports, Mapping):  # tolerate {name: report} (e.g. pipeline.run_many output)
        reports = list(reports.values())
    reports = list(reports or [])
    by_name: dict[str, TrendReport] = {}
    for r in reports:
        by_name.setdefault(str(r.name), r)
    rank_df = ranking if isinstance(ranking, pd.DataFrame) else pd.DataFrame(ranking)

    rows_html = []
    order: list[str] = []
    scores: dict[str, Optional[float]] = {}
    for pos, (_, row) in enumerate(rank_df.iterrows(), 1):
        nm = _cell(row, "name")
        nm = "" if nm is None else str(nm)
        order.append(nm)
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
        rows_html.append(
            f'<tr><td class="num" style="text-align:left"><strong>{pos}</strong></td>'
            f'<td><span class="sname">{_esc(nm)}</span></td><td class="hide-sm">{badge}</td>'
            f'<td class="num hide-sm">{_signed_html(mom, mom_txt)}'
            f'{f"<span class=unit>{_esc(mom_unit)}</span>" if mom_unit else ""}</td>'
            f'<td class="num hide-sm">{_signed_html(rc, _fmt_pct(rc))}</td>'
            f'<td class="num">{_signed_html(ec, _fmt_pct(ec))}</td>'
            f'<td class="num hide-sm">{_esc(_fmt_prob(pi))}</td>'
            f'<td class="score-cell"><div class="wbar" title="50 = 중립"><div class="track"><div class="fill" style="width:{sw:.1f}%"></div>'
            f'<span class="mid"></span></div><span class="pct score">{_esc(_fmt_num(score, 0))}</span></div></td></tr>')
    if rows_html:
        table = ('<div class="table-wrap"><table><thead><tr><th>순위</th><th>이름</th><th class="hide-sm">단계</th>'
                 '<th class="num hide-sm"><abbr title="최근 추세 기울기 − 장기 기울기 (평균 수준 대비 %p, 기간당)">'
                 '모멘텀 (%p/기간)</abbr></th>'
                 '<th class="num hide-sm">최근 변화</th><th class="num">예측 변화</th>'
                 '<th class="num hide-sm">상승 확률</th><th class="th-score">상승 점수</th></tr></thead>'
                 f'<tbody>{"".join(rows_html)}</tbody></table></div>')
    else:
        table = '<p class="empty">순위 데이터가 없습니다.</p>'

    ordered: list[TrendReport] = []
    seen: set[int] = set()
    for nm in order:
        r = by_name.get(nm)
        if r is not None and id(r) not in seen:
            ordered.append(r)
            seen.add(id(r))
    for r in reports:
        if id(r) not in seen:
            ordered.append(r)
            seen.add(id(r))

    cards = []
    rank_of = {nm: i for i, nm in enumerate(order, 1)}
    for r in ordered:
        a, o, m = r.analysis, r.outlook, r.analysis.metrics
        freq = str(a.freq)
        dd = _ChartData(r)
        rank = rank_of.get(str(r.name))
        score = scores.get(str(r.name))
        end_fc = _fmt_date(dd.fc_dates[-1], freq) if len(dd.fc_dates) else _fmt_date(m.end, freq)
        cards.append(
            '<article class="mcard"><div class="top">'
            f'<div><h3>{_esc(r.name)}</h3></div>'
            f'<span class="rank">{"#" + str(rank) if rank else ""}</span></div>'
            f'<div>{_badge(str(a.phase.label), _finite(a.phase.confidence))}</div>'
            f"{_mini_chart(r)}"
            '<div class="stats">'
            f'<div><div class="k">최근 변화</div><div class="v">{_signed_html(m.recent_change_pct, _fmt_pct(m.recent_change_pct))}</div></div>'
            f'<div><div class="k">예측 변화</div><div class="v">{_signed_html(o.expected_change_pct, _fmt_pct(o.expected_change_pct))}</div></div>'
            f'<div><div class="k">상승 확률</div><div class="v">{_esc(_fmt_prob(o.prob_increase))}</div></div>'
            "</div>"
            f'<div class="foot"><span>{_esc(_fmt_date(m.start, freq))} ~ {_esc(end_fc)}</span>'
            f'<span>상승 점수 <strong>{_esc(_fmt_num(score, 0))}</strong></span></div>'
            "</article>")
    cards_html = f'<div class="cards">{"".join(cards)}</div>' if cards else '<p class="empty">리포트가 없습니다.</p>'

    n = len(ordered) if ordered else len(order)
    from . import __version__
    body = "\n".join([
        '<header class="hero"><div class="eyebrow"><span class="logo" aria-hidden="true"></span>'
        'TrendAI 비교 리포트 · 어떤 트렌드가 뜨고 있나</div>'
        f"<h1>트렌드 비교 · {n}개 시리즈</h1>"
        '<div class="meta"><span class="chip">상승 점수 0–100 (50 = 중립)</span>'
        '<span class="chip">예측 변화 · 상승 확률 · 최근 변화 · 모멘텀 종합</span></div>'
        f'<p class="gen">TrendAI {_esc(__version__)} 버전으로 생성</p></header>',
        f'<section class="card"><h2>상승 순위</h2><p class="desc">점수가 높을수록 앞으로 더 강하게 오를 것으로 예상되는 트렌드입니다.</p>{table}</section>',
        '<section><h2 style="font-size:17px;margin:4px 0 12px">시리즈별 요약</h2>'
        '<p class="desc">파란 선은 실제값, 주황 선과 음영은 예측과 예측 구간입니다.</p>'
        f"{cards_html}</section>",
        '<footer>이 비교는 과거 데이터의 통계적 패턴에 기반한 추정이며 미래를 보장하지 않습니다.</footer>',
    ])
    page = _page(f"트렌드 비교 — {n}개 시리즈", body, extra_css=_SVG_STYLE)
    _write(page, path)
    return page
