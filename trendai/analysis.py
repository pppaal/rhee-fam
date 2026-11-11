"""Past & present: understand where a trend has been and where it is now.

This module turns a regular series (see :mod:`trendai.data`) into an
:class:`~trendai.types.AnalysisResult`:

* :func:`decompose` -- robust, STL-like additive decomposition
  ``series = trend + seasonal + resid`` with no NaNs anywhere (edges included),
* :func:`mann_kendall` / :func:`theil_sen_slope` -- non-parametric trend test
  and robust slope,
* :func:`detect_changepoints` -- structural breaks via optimal piecewise-linear
  segmentation (dynamic programming + BIC-like penalty),
* :func:`detect_anomalies` -- robust z-scores of the (studentised) residual,
* :func:`compute_metrics` -- growth, momentum, strength, drawdown, ...,
* :func:`lifecycle_phase` -- explainable rule-based lifecycle classification
  (emerging / growing / peaking / declining / stable / reviving / volatile),
* :func:`analyze` -- all of the above in one call.

Only ``numpy`` and ``pandas`` are used; everything is deterministic and every
returned float is finite.
"""

from __future__ import annotations

import math
import warnings
from typing import Optional

import numpy as np
import pandas as pd

from . import data as _data
from .types import (
    AnalysisResult,
    Anomaly,
    Changepoint,
    Decomposition,
    LifecyclePhase,
    TrendMetrics,
)

__all__ = [
    "decompose",
    "theil_sen_slope",
    "mann_kendall",
    "detect_changepoints",
    "detect_anomalies",
    "compute_metrics",
    "lifecycle_phase",
    "analyze",
    "pct_change_undefined",
    "changepoint_label",
    "CHANGEPOINT_LABEL_EN",
    "MIN_PHASE_OBS",
]

_TINY = 1e-12

# Korean unit words used in explanations ("최근 12개월", "최근 8주", ...).
_UNIT_KO: dict[str, str] = {"D": "일", "W": "주", "M": "개월", "Q": "분기", "Y": "년"}
# Korean per-period units of slopes ("−0.69/월"), as narrative.period_unit writes them
_PER_KO: dict[str, str] = {"D": "일", "W": "주", "M": "월", "Q": "분기", "Y": "년"}

#: changepoint labels without a direction (used only when the slopes are missing)
_KIND_KO: dict[str, str] = {
    "acceleration": "가속",
    "deceleration": "감속",
    "reversal_up": "상승 반전",
    "reversal_down": "하락 반전",
    "level_shift": "수준 변화",
}
_KIND_EN: dict[str, str] = {
    "acceleration": "acceleration",
    "deceleration": "deceleration",
    "reversal_up": "upward reversal",
    "reversal_down": "downward reversal",
    "level_shift": "level shift",
}
#: directed changepoint labels (see :func:`changepoint_label`): (kind, flag) -> (Korean, English)
_CP_LABELS: dict[tuple[str, bool], tuple[str, str]] = {
    ("acceleration", True): ("성장 가속", "growth accelerating"),
    ("acceleration", False): ("하락 가속", "decline accelerating"),
    ("deceleration", True): ("성장 둔화", "growth slowing"),
    ("deceleration", False): ("하락세 둔화", "decline easing"),
    # reversal_* only says that the slope changed sign or started: from a flat stretch it is
    # a turn, not a reversal
    ("reversal_up", True): ("하락→상승 반전", "reversal from decline to growth"),
    ("reversal_up", False): ("횡보→상승 전환", "turn from flat to growth"),
    ("reversal_down", True): ("상승→하락 반전", "reversal from growth to decline"),
    ("reversal_down", False): ("횡보→하락 전환", "turn from flat to decline"),
}
#: a slope smaller in size than this share of the larger of the two slopes counts as flat
_CP_FLAT_TOL = 0.1
#: English for every Korean label :func:`changepoint_label` can return (directed and
#: undirected), e.g. for translating the ``구조 변화(...)`` phase reason
CHANGEPOINT_LABEL_EN: dict[str, str] = {
    **{_KIND_KO[k]: _KIND_EN[k] for k in _KIND_KO},
    **{ko: en for ko, en in _CP_LABELS.values()},
    "추세 변화": "trend change",
}

# The lifecycle rules use these (deliberately round) thresholds.  Relative sizes
# are fractions of the series' trend scale (max |trend|).
_MIN_SHORT_N = 8          # below this many observations: no lifecycle verdict
#: public name of :data:`_MIN_SHORT_N`: with fewer observations ``lifecycle_phase`` gives no
#: lifecycle verdict -- its label ('stable' or 'volatile') is a tentative placeholder that only
#: says whether noise dominates (the trend test may still be significant)
MIN_PHASE_OBS = _MIN_SHORT_N
_MOVE = 0.15              # a "meaningful" move is >= 15% of the trend scale
_FAR_BELOW_PEAK = 0.30    # >= 30% below the trend peak counts as "well below"
_STALL_RATIO = 0.35       # recent slope < 35% of the best past slope = stalled
_EMERGING_MULTIPLE = 2.5  # now >= 2.5x the long-run base level = small base
_VOLATILE_CV = 0.20       # unstructured series with CV >= 20% is "volatile"
_T_SIG = 2.5              # |t| of the recent slope needed to call it a real move
_PACE_CAP = 20.0          # "recent pace is X times the average pace": beyond this, "more than 20x"
_EDGE_SHRINK = 4.0        # edge LOESS keeps the narrow one-sided fit only if (D / sd_D)^2 >> this
#: with a weak (but significant) interaction slope, the log decomposition must leave less
#: than this share of the additive remainder's sum of squares (see _prefer_multiplicative)
_MULT_WEAK_SS = 0.75


# =========================================================================== helpers


def _finite(x: object, default: float = 0.0) -> float:
    """``float(x)`` if finite, else ``default``."""
    try:
        v = float(x)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float(default)
    return v if math.isfinite(v) else float(default)


def _clean_values(series: pd.Series) -> np.ndarray:
    """Float values of ``series`` with interior/edge NaNs filled (warns if any)."""
    y = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    if len(y) and not np.all(np.isfinite(y)):
        warnings.warn("series contains non-finite values; they were interpolated",
                      RuntimeWarning, stacklevel=3)
        s = pd.Series(np.where(np.isfinite(y), y, np.nan))
        s = s.interpolate(limit_direction="both")
        y = s.fillna(0.0).to_numpy(dtype=float)
    return y


def _unit(*arrays: object) -> float:
    """Positive normalising constant: the largest finite ``|value|`` (1.0 if none / all 0).

    The public functions divide their inputs by this before applying any numeric
    tolerance and scale data-unit outputs back, so that results do not depend on
    the units of the data (frequencies of 1e-9 behave like revenues of 1e12).
    """
    m = 0.0
    for a in arrays:
        v = np.asarray(a, dtype=float).ravel()
        v = v[np.isfinite(v)]
        if len(v):
            m = max(m, float(np.max(np.abs(v))))
    return m if (m > 0.0 and math.isfinite(m)) else 1.0


def _robust_sd(r: np.ndarray) -> float:
    """Robust standard deviation: 1.4826 MAD, guarded against point masses.

    The MAD collapses to (nearly) 0 when more than half of the values are tied,
    e.g. zero-inflated counts or a long constant stretch; every other value would
    then look like an extreme outlier.  The scale is therefore floored at half of
    the Gaussian-consistent 90%-quantile estimate (``0.5 * q90 / 1.645``) and,
    for > 90% ties (an isolated spike among zeros), at a tenth of the mean
    absolute deviation.  Neither floor binds for Gaussian noise (~0.5 and
    ~0.08 sigma) nor with a few percent of outliers.  Falls back to the std.
    """
    r = np.asarray(r, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 2:
        return 0.0
    med = float(np.median(r))
    dev = np.abs(r - med)
    s = max(1.4826 * float(np.median(dev)), 0.304 * float(np.quantile(dev, 0.9)),
            0.1 * float(np.mean(dev)))
    if s > _TINY:
        return s
    return float(np.std(r))


def _heavy_ties(y: np.ndarray) -> bool:
    """True for zero-inflated / heavily tied data (intermittent counts, step series).

    At least 30% of the values share one value, yet at least 10% (and >= 5
    points) differ from it.  Median/MAD-type estimators converge onto that mode
    for such data (the non-tied values all look like outliers), so the callers
    switch to quantile-based, more lenient scales.  An isolated spike among
    zeros is *not* heavily tied (too few differing points).
    """
    v = np.asarray(y, dtype=float)
    v = v[np.isfinite(v)]
    n = len(v)
    if n < 10:
        return False
    top = int(np.unique(v, return_counts=True)[1].max())
    return top >= 0.3 * n and (n - top) >= max(5.0, 0.1 * n)


_INTERMITTENT_ZEROS = 0.3   # share of exact zeros (overall and recently): intermittent demand


def _is_intermittent(y: np.ndarray, m: int) -> bool:
    """Intermittent demand: non-negative, ``>= 30 %`` exact zeros overall *and* recently.

    The rule of ``forecast._is_intermittent`` on the same history (the last
    ``max(400, 8 m)`` points; recently = the last ``max(m, 12)``; at least two
    non-zero values), so that the analysis and the forecast agree on which
    series are mostly zeros.  Such a series has no meaningful trend level: the
    smoothed demand rate wanders between ~0 and a fraction of the typical order
    size, so %-changes of it (+3,192 %), a CAGR, a drawdown "from the peak" and
    residual z-scores of ordinary orders say nothing.
    """
    y = np.asarray(y, dtype=float)
    y = y[-max(400, 8 * max(1, int(m))):]
    if len(y) < 8 or not np.all(np.isfinite(y)) or np.any(y < 0):
        return False
    zero = y == 0
    if int((~zero).sum()) < 2:
        return False
    k = min(len(y), max(int(m), 12))
    return bool(zero.mean() >= _INTERMITTENT_ZEROS and zero[-k:].mean() >= _INTERMITTENT_ZEROS)


def _season_of(series: pd.Series, dec: Optional[Decomposition] = None) -> int:
    """Season length of ``series``' frequency (inferred from its index), else ``dec.period``."""
    try:
        return int(_data.season_length(_data.infer_freq(pd.DatetimeIndex(series.index))))
    except Exception:  # not a datetime index, too short, ...
        return max(1, int(dec.period)) if dec is not None else 1


def _winsorized_sd(r: np.ndarray) -> float:
    """Std of ``r`` after clipping at +-3 robust SDs (robust yet efficient on small samples)."""
    r = np.asarray(r, dtype=float)
    r = r[np.isfinite(r)]
    s0 = _robust_sd(r)
    if len(r) < 2 or s0 <= _TINY:
        return s0
    med = float(np.median(r))
    s = float(np.std(np.clip(r - med, -3.0 * s0, 3.0 * s0)))
    return s if s > _TINY else s0


def _ols(y: np.ndarray, x: Optional[np.ndarray] = None) -> tuple[float, float]:
    """OLS ``(intercept, slope)`` of ``y`` on ``x`` (default ``0..n-1``)."""
    y = np.asarray(y, dtype=float)
    n = len(y)
    if x is None:
        x = np.arange(n, dtype=float)
    if n == 0:
        return 0.0, 0.0
    if n == 1:
        return float(y[0]), 0.0
    xm = float(x.mean())
    ym = float(y.mean())
    sxx = float(np.sum((x - xm) ** 2))
    if sxx <= _TINY:
        return ym, 0.0
    slope = float(np.sum((x - xm) * (y - ym)) / sxx)
    return ym - slope * xm, slope


def _scale(values: np.ndarray) -> float:
    """A positive "size" of a series: max(|mean|, std), falling back to 1."""
    v = np.asarray(values, dtype=float)
    if len(v) == 0:
        return 1.0
    s = max(abs(float(np.mean(v))), float(np.std(v)))
    if s <= _TINY:
        s = float(np.max(np.abs(v))) if len(v) else 0.0
    return s if s > _TINY else 1.0


def _level(values: np.ndarray) -> float:
    """Mean level used to express slopes in %; |mean|, else mean |x|, else 1."""
    v = np.asarray(values, dtype=float)
    if len(v) == 0:
        return 1.0
    m = abs(float(np.mean(v)))
    if m > _TINY and m >= 0.05 * float(np.mean(np.abs(v))):
        return m
    m = float(np.mean(np.abs(v)))
    return m if m > _TINY else 1.0


#: a series that takes negative values (net profit, temperature, a growth rate) has no natural
#: zero for percentages: a change in % is stated only when the base is at least this many
#: standard deviations of the observations away from 0 (forecast._PCT_MIN_LEVEL_SD, the rule
#: behind Outlook.expected_change_pct)
_PCT_MIN_LEVEL_SD = 0.5


def _pct_reason(new: float, base: float, ref: float, floor: float = 0.0) -> Optional[str]:
    """Why ``(new - base) / base * 100`` is no meaningful percentage (``None``: it is).

    * ``'zero'``: ``base`` (or ``new``) is not finite, or ``|base|`` is at most
      1 % of ``ref`` (a typical absolute level of the series; the test is
      relative, so it works in any units);
    * ``'negative'``: ``base < 0`` (a percentage of a negative base reads backwards);
    * ``'sign_change'``: ``new < 0 < base`` -- the level crossed zero (a P&L
      going from 10 to -10 is no "-200 %" change);
    * ``'near_zero'``: ``base < floor`` -- for a series with negative values
      ``floor`` is half the standard deviation of the observations
      (:func:`_pct_floor`), 0 otherwise.
    """
    thr = 0.01 * abs(ref) if math.isfinite(ref) else 0.0
    if not (math.isfinite(new) and math.isfinite(base)) or abs(base) <= thr or base == 0.0:
        return "zero"
    if base < 0.0:
        return "negative"
    if new < 0.0:
        return "sign_change"
    if base < floor:
        return "near_zero"
    return None


def _pct_floor(y: np.ndarray) -> float:
    """Smallest base of a meaningful percentage for data ``y``: half the standard deviation
    of ``y`` when it takes negative values, 0 for non-negative (ratio-scale) data."""
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]
    if len(y) == 0 or float(np.min(y)) >= 0.0:
        return 0.0
    return _PCT_MIN_LEVEL_SD * float(np.std(y))


def _pct_change(new: float, base: float, ref: float, floor: float = 0.0) -> Optional[float]:
    """``(new - base) / base * 100``, or ``None`` when :func:`_pct_reason` gives a reason.

    ``None`` if ``base`` is <= 0 or ~0 (at most 1 % of ``ref``), if ``new`` has
    crossed zero (``new < 0 < base``) or if ``base < floor`` (``floor``: see
    :func:`_pct_floor`; 0 keeps non-negative series as they are).
    """
    if _pct_reason(new, base, ref, floor) is not None:
        return None
    v = (new - base) / base * 100.0
    if not math.isfinite(v):
        return None
    return 0.0 if abs(v) < 1e-9 else float(v)


def pct_change_undefined(base: float, new: float, values: object) -> Optional[str]:
    """Why a change from ``base`` to ``new`` has no meaningful percentage in a series with
    observations ``values`` (``None``: ``(new - base) / base * 100`` is meaningful).

    The rule of ``TrendMetrics.total_change_pct`` / ``recent_change_pct`` /
    ``cagr_pct`` and of the phase reasons -- the analysis' counterpart of
    ``forecast._pct_undefined`` -- so that a narrative or a report can tell
    *why* such a number is ``None`` and apply the same rule to numbers the
    metrics cannot leave out (``drawdown_pct`` is typed ``float``: call
    ``pct_change_undefined(metrics.peak_value, metrics.last_value,
    analysis.series)`` before printing it).  Reasons:

    * ``'zero'``: ``base`` is ~0 (at most 1 % of the mean absolute value of
      ``values``) or a number is not finite;
    * ``'negative'``: ``base`` is negative;
    * ``'sign_change'``: ``new`` is negative while ``base`` is positive (the
      level crossed zero: 10 -> -10 is no "-200 %" change);
    * ``'near_zero'``: ``values`` take negative values and ``base`` is below
      half their standard deviation (an interval-scale series near its zero).

    Non-negative series only ever get ``'zero'`` (and ``'negative'`` /
    ``'sign_change'`` for negative inputs they cannot produce themselves).
    """
    y = np.asarray(values, dtype=float).ravel()
    y = y[np.isfinite(y)]
    try:
        b, v = float(base), float(new)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return "zero"
    if not (math.isfinite(b) and math.isfinite(v)):
        return "zero"
    c = _unit(y, b, v)   # tolerances on a unit scale: no overflow at |y| ~ 1e300
    yn = y / c
    ref = float(np.mean(np.abs(yn))) if len(yn) else abs(b / c)
    return _pct_reason(v / c, b / c, ref, _pct_floor(yn))


def changepoint_label(cp: Changepoint, lang: str = "ko") -> str:
    """Directed name of a structural break, read off its slopes: ``'하락→상승 반전'``.

    * ``acceleration``: ``성장 가속`` (slope after >= 0) / ``하락 가속``;
    * ``deceleration``: ``성장 둔화`` (slope before > 0) / ``하락세 둔화``;
    * ``reversal_up``: ``하락→상승 반전`` when the slope before was clearly
      negative (below -10 % of the larger absolute slope), otherwise
      ``횡보→상승 전환``; ``reversal_down`` likewise (``상승→하락 반전`` /
      ``횡보→하락 전환``);
    * ``level_shift``: ``수준 변화``; missing slopes: an undirected label
      (``가속``, ``감속``, ``상승 반전``, ``하락 반전``).

    ``lang='en'``: ``growth accelerating``, ``decline easing``, ``reversal
    from decline to growth``, ``turn from flat to growth``, ``level shift``, ...
    The one classifier behind the phase reasons, the narrative and the report.
    """
    en = str(lang or "ko").strip().lower().startswith("en")
    kind = str(getattr(cp, "kind", ""))

    def slope(name: str) -> Optional[float]:
        try:
            v = float(getattr(cp, name, None))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None
        return v if math.isfinite(v) else None

    before, after = slope("slope_before"), slope("slope_after")
    key: Optional[tuple[str, bool]] = None
    if kind == "acceleration" and after is not None:
        key = (kind, after >= 0)
    elif kind == "deceleration" and before is not None:
        key = (kind, before > 0)
    elif kind in ("reversal_up", "reversal_down") and before is not None and after is not None:
        tol = _CP_FLAT_TOL * max(abs(before), abs(after), 1e-12)
        key = (kind, before < -tol if kind == "reversal_up" else before > tol)
    if key is not None:
        ko, en_txt = _CP_LABELS[key]
        return en_txt if en else ko
    table = _KIND_EN if en else _KIND_KO
    return table.get(kind, "trend change" if en else "추세 변화")


def _fmt_date(ts: pd.Timestamp, freq: str) -> str:
    ts = pd.Timestamp(ts)
    if freq == "M":
        return ts.strftime("%Y-%m")
    if freq == "Q":
        return f"{ts.year}-Q{(ts.month - 1) // 3 + 1}"
    if freq == "Y":
        return f"{ts.year}"
    return ts.strftime("%Y-%m-%d")


def _when_ko(ts: pd.Timestamp, freq: str) -> str:
    """Date for Korean prose before a particle: ``2024-03에``, but ``2016년에`` for yearly data."""
    s = _fmt_date(ts, freq)
    return s + "년" if freq == "Y" else s


def _ramp(x: float, lo: float, hi: float) -> float:
    """Linear ramp: 0 at ``lo``, 1 at ``hi`` (clipped)."""
    if hi == lo:
        return 1.0 if x >= hi else 0.0
    return float(min(1.0, max(0.0, (x - lo) / (hi - lo))))


# =========================================================================== decomposition


def _trend_window(n: int, period: int) -> int:
    """Odd LOESS window for the trend: ~n/10, at least min(1.5*period, n/8), >= 5."""
    p = max(1, int(period))
    w = max(n / 10.0, min(1.5 * p, n / 8.0), 5.0)
    w = int(round(w))
    if w % 2 == 0:
        w += 1
    return int(max(1, min(w, n)))


def _kernel_noise_factors(window: int) -> tuple[float, float]:
    """``(l0, v)`` of the interior LOESS equivalent kernel for ``window``.

    For white noise with variance s^2, the smoothed trend has variance
    ``v * s^2`` and the residual ``(1 - 2*l0 + v) * s^2`` (``l0`` = weight of
    the centre point, ``v`` = sum of squared weights).
    """
    q = max(1, int(window))
    hw = (q - 1) // 2
    d = np.abs(np.arange(-hw, hw + 1, dtype=float))
    k = (1.0 - (d / (hw + 1.0)) ** 3) ** 3
    tot = float(k.sum())
    return 1.0 / tot, float(np.sum(k * k)) / tot ** 2


def _local_fit(y: np.ndarray, w_all: np.ndarray, i: np.ndarray, q: int, want_weights: bool = False
               ) -> tuple[np.ndarray, np.ndarray, Optional[tuple[np.ndarray, np.ndarray]]]:
    """Local-linear (tricube x robustness weights) fit at evaluation points ``i``.

    The window holds the ``q`` nearest points (one-sided at the edges).
    Returns ``(fit, ok, W)``: the fitted values, a mask of well-posed fits (a
    degenerate design falls back to the weighted, else plain, window mean) and,
    if ``want_weights``, ``W = (idx, L)``: the window positions and the
    equivalent-kernel weights over them (``fit = (L * y[idx]).sum(axis=1)``),
    used for variance calculations.
    """
    n = len(y)
    lo = np.clip(i - (q - 1) // 2, 0, n - q)
    idx = lo[:, None] + np.arange(q)[None, :]
    dx = (idx - i[:, None]).astype(float)
    d = np.abs(dx)
    h = d.max(axis=1, keepdims=True) + 1.0
    k = (1.0 - (d / h) ** 3) ** 3 * w_all[idx]
    yy = y[idx]
    s0 = k.sum(axis=1)
    s1 = (k * dx).sum(axis=1)
    s2 = (k * dx * dx).sum(axis=1)
    det = s0 * s2 - s1 * s1
    ok = (det > 1e-10 * np.maximum(s0 * s2, _TINY)) & (s0 > _TINY)
    with np.errstate(divide="ignore", invalid="ignore"):
        lw = np.where(ok[:, None], k * (s2[:, None] - s1[:, None] * dx)
                      / np.where(ok, det, 1.0)[:, None],
                      k / np.where(s0 > _TINY, s0, 1.0)[:, None])
    fit = (lw * yy).sum(axis=1)
    bad = ~np.isfinite(fit) | (s0 <= _TINY)
    if bad.any():  # all weights vanished: plain mean of the window
        fit[bad] = yy[bad].mean(axis=1)
        lw[bad] = 1.0 / q
    return fit, ok, ((idx, lw) if want_weights else None)


def _loess(y: np.ndarray, window: int, weights: Optional[np.ndarray] = None) -> np.ndarray:
    """Local-linear LOESS with tricube kernel over ``window`` nearest points.

    At the edges the window becomes one-sided, i.e. the trend is extrapolated
    with a local linear fit -- no NaNs anywhere.  A one-sided local-linear fit
    is very noisy at the very end (its variance is ~3x that of the interior
    fit: the end value follows the last few observations), so near the edges
    it is shrunk towards a three times wider one-sided local-linear fit.  The
    shrinkage is adaptive (positive-part James-Stein): the difference ``D`` of
    the two fits is compared with its noise variance ``V`` (white noise of the
    size of the second differences of ``y``), the narrow fit is kept with
    weight ``max(0, 1 - c V / D^2)`` -- i.e. genuine curvature at the end (an
    S-curve take-off, a fad's collapse) is kept, while a noise-driven end
    hook is replaced by the steadier wide fit -- and the shrinkage fades out
    with the asymmetry of the window (0 in the interior, full at the end
    point).  The James-Stein factor is applied with a 4x noise variance
    (:data:`_EDGE_SHRINK`): the narrow fit only starts to count once ``D``
    exceeds 2 noise SDs.  Straight lines stay exact.  As in STL, the local fit is
    evaluated at every ``max(1, window // 10)``-th point (and both ends) and
    linearly interpolated in between, which keeps long series fast.
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n <= 2:
        return y.copy()
    q = int(max(2, min(window, n)))
    q_wide = int(min(n, 3 * q - 2))
    hw = (q - 1) / 2.0
    w_all = np.ones(n) if weights is None else np.asarray(weights, dtype=float)
    jump = max(1, q // 10)
    at = np.arange(n) if jump == 1 else np.unique(np.r_[np.arange(0, n, jump), n - 1])
    out = np.empty(len(at))
    chunk = max(1, 2_000_000 // q)
    for start in range(0, len(at), chunk):
        pos = np.arange(start, min(len(at), start + chunk))
        fit, _, _ = _local_fit(y, w_all, at[pos], q)
        out[pos] = fit
    if q_wide > q and n >= 5:
        lo = np.clip(at - (q - 1) // 2, 0, n - q)
        asym = np.clip(np.abs(at - (lo + hw)) / max(hw, 1.0), 0.0, 1.0)
        edge = np.flatnonzero(asym > 1e-9)
        if len(edge):
            ie = at[edge]
            f1, ok1, W1 = _local_fit(y, w_all, ie, q, want_weights=True)
            f3, ok3, W3 = _local_fit(y, w_all, ie, q_wide, want_weights=True)
            # white-noise variance from second differences (robust to trend and spikes)
            sig2 = (_robust_sd(np.diff(y, 2)) ** 2) / 6.0 if n >= 5 else 0.0
            dlt = f1 - f3
            # the narrow (one-sided) window is nested in the wide one at an edge
            idx1, lw1 = W1  # type: ignore[misc]
            idx3, lw3 = W3  # type: ignore[misc]
            diff_w = -lw3.copy()
            np.put_along_axis(diff_w, np.clip(idx1 - idx3[:, :1], 0, q_wide - 1),
                              np.take_along_axis(diff_w, np.clip(idx1 - idx3[:, :1], 0, q_wide - 1),
                                                 axis=1) + lw1, axis=1)
            var_d = sig2 * np.sum(diff_w ** 2, axis=1)
            d2 = dlt * dlt
            with np.errstate(divide="ignore", invalid="ignore"):
                g = np.where(d2 > _TINY * _TINY,
                             np.clip(1.0 - _EDGE_SHRINK * var_d / np.where(d2 > 0, d2, 1.0), 0.0, 1.0),
                             1.0)
            lam = 1.0 - asym[edge] * (1.0 - g)
            good = ok1 & ok3 & np.isfinite(f3) & np.isfinite(lam)
            out[edge] = np.where(good, f3 + lam * dlt, out[edge])
    if jump == 1:
        return out
    return np.interp(np.arange(n, dtype=float), at.astype(float), out)


def _bisquare(r: np.ndarray, wide: bool = False) -> np.ndarray:
    """STL robustness weights from residuals ``r`` (floored, never all zero).

    ``h = 6 * median|r|`` as in STL, but the median is floored at a quarter of
    the 90% quantile of ``|r|`` and at a tenth of the mean ``|r|`` (neither
    binds for Gaussian residuals: ~0.41 and ~0.08 sigma vs. 0.67 sigma).
    Without the first floor, series whose values are mostly tied collapse onto
    the mode.  The second one covers > 90% ties (an isolated spike among
    zeros), where tiny leak residuals next to the spike would otherwise be
    declared outliers too.  ``wide=True`` (heavily tied data, see
    :func:`_heavy_ties`) uses ``h = 6 * q90|r|``: for zero-inflated counts every
    non-zero value is far from the median, and a median-based ``h`` would bias
    the trend towards 0; only gross outliers are down-weighted then.
    """
    a = np.abs(r)
    q90 = float(np.quantile(a, 0.9))
    h = 6.0 * max(float(np.median(a)), (1.0 if wide else 0.25) * q90,
                  0.1 * float(np.mean(a)))
    if h <= _TINY:  # (almost) all of the residuals are exactly 0
        h = 6.0 * float(np.mean(a))
    if h <= _TINY:
        return np.ones_like(r)
    u = r / h
    w = np.where(np.abs(u) < 1.0, (1.0 - u * u) ** 2, 0.0)
    return np.maximum(w, 1e-4)


def _initial_trend(y: np.ndarray, p: int) -> np.ndarray:
    """Centred (2x)p moving average, edges extended by local linear fits.

    Series shorter than ``p + 2`` (possible for a segment between two level
    shifts) get an OLS line instead.
    """
    n = len(y)
    if n < p + 2:
        a, b = _ols(y)
        return a + b * np.arange(n, dtype=float)
    if p % 2 == 0:
        kern = np.r_[0.5, np.ones(p - 1), 0.5] / p
    else:
        kern = np.ones(p) / p
    m = len(kern)
    half = m // 2
    ma = np.convolve(y, kern, mode="valid")
    out = np.empty(n)
    out[half:half + len(ma)] = ma
    k = int(max(2, min(len(ma), p)))
    xs = np.arange(half, half + len(ma), dtype=float)
    a, b = _ols(ma[:k], xs[:k])
    out[:half] = a + b * np.arange(half)
    a, b = _ols(ma[-k:], xs[-k:])
    tail = np.arange(half + len(ma), n, dtype=float)
    out[half + len(ma):] = a + b * tail
    return out


def _seasonal_pattern(d: np.ndarray, p: int, w: np.ndarray, median: bool = False) -> np.ndarray:
    """Per-phase robust average of ``d``, centred to sum to zero.

    ``median=True`` (first pass, before robustness weights exist) uses
    per-phase medians; otherwise bisquare-weighted means.

    The pattern is shrunk towards zero by a positive-part (method-of-moments)
    factor ``1 - E[noise energy] / energy`` whose noise estimate is the
    cycle-to-cycle disagreement of each phase.  A
    genuine seasonal pattern repeats in every cycle and is left (almost)
    untouched; noise and trend misfit that differ between cycles -- the main
    failure mode with only 2-3 cycles -- are pushed back to trend/residual.
    """
    n = len(d)
    phase = np.arange(n) % p
    if median:
        pat = np.array([float(np.median(d[phase == j])) if np.any(phase == j) else 0.0
                        for j in range(p)])
        level = float(np.median(d - pat[phase]))
    else:
        num = np.bincount(phase, weights=w * d, minlength=p)
        den = np.bincount(phase, weights=w, minlength=p)
        cnt = np.bincount(phase, minlength=p).astype(float)
        raw = np.bincount(phase, weights=d, minlength=p) / np.maximum(cnt, 1.0)
        pat = np.where(den > _TINY, num / np.where(den > _TINY, den, 1.0), raw)
        level = float(np.sum(w * d) / max(np.sum(w), _TINY))
    pat = pat - pat.mean()
    if p >= 4:
        dev = d - (pat + level)[phase]
        if median:
            within = _robust_sd(dev) ** 2
        else:
            within = float(np.sum(w * dev * dev) / max(np.sum(w), _TINY))
        k = max(n / p, 1.0)
        within *= k / max(k - 1.0, 1.0)            # unbiased within-phase variance
        var_mean = within / k                      # variance of a phase mean
        energy = float(np.mean(pat * pat))
        if energy > _TINY:
            # (p - 1) / p * var_mean is the expected energy of a centred pattern of
            # pure noise: subtracting it is the method-of-moments estimate of the
            # signal energy (stronger than James-Stein's (p - 3) / p, which left
            # visible noise-only patterns for p = 4 or 2-3 cycles)
            shrink = max(0.0, 1.0 - (p - 1) / p * var_mean / energy)
            pat = pat * shrink
    return pat


def _segment_bounds(n: int, breaks: Optional[list[int]]) -> list[tuple[int, int]]:
    """``[(start, stop), ...]`` of the stretches between ``breaks`` (each >= 4 points)."""
    cuts = sorted({int(b) for b in (breaks or []) if 4 <= int(b) <= n - 4})
    out: list[tuple[int, int]] = []
    a = 0
    for b in cuts:
        if b - a >= 4:
            out.append((a, b))
            a = b
    if out and n - a < 4:          # merge a too-short tail into the previous stretch
        a = out.pop()[0]
    out.append((a, n))
    return out


def _trend_fit(y: np.ndarray, window: int, w: Optional[np.ndarray],
               bounds: list[tuple[int, int]]) -> np.ndarray:
    """LOESS trend, fitted separately on every stretch of ``bounds`` (level shifts)."""
    if len(bounds) == 1:
        return _loess(y, window, w)
    out = np.empty(len(y))
    for a, b in bounds:
        ww = None if w is None else w[a:b]
        out[a:b] = _loess(y[a:b], min(window, b - a), ww)
    return out


def _nonneg_floor(trend: np.ndarray, z: np.ndarray, window: int,
                  bounds: list[tuple[int, int]]) -> np.ndarray:
    """Keep the trend of a non-negative series >= 0 and inside the data at the edges.

    A local-linear extrapolation of a steep fall can end below zero (a short
    fad: trend -1 while the last value is 8).  In the interior the LOESS is a
    positive-weight average and cannot do that; within half a window of each
    end (of every stretch between level shifts) the trend is floored at the
    lowest deseasonalised value ``z`` of the edge window (and at 0).
    """
    out = np.maximum(trend, 0.0)
    for a, b in bounds:
        m = b - a
        q = int(max(2, min(window, m)))
        hw = (q - 1) // 2
        if m < 3 or hw < 1:
            continue
        zz = np.maximum(z[a:b], 0.0)
        # every point within hw of an end has the same (one-sided) window
        out[a:a + hw] = np.maximum(out[a:a + hw], float(np.min(zz[:q])))
        out[b - hw:b] = np.maximum(out[b - hw:b], float(np.min(zz[m - q:])))
    return out


def _decompose_core(yn: np.ndarray, p: int, use_season: bool, window: int, wide: bool,
                    bounds: list[tuple[int, int]]) -> tuple[np.ndarray, np.ndarray]:
    """``(trend, seasonal)`` of ``yn`` (additive; see :func:`decompose`)."""
    n = len(yn)
    if not use_season:
        trend = _trend_fit(yn, window, None, bounds)
        for _ in range(2):
            w = _bisquare(yn - trend, wide)
            trend = _trend_fit(yn, window, w, bounds)
        return trend, np.zeros(n)
    if len(bounds) == 1:
        trend = _initial_trend(yn, p)
    else:
        trend = np.empty(n)
        for a, b in bounds:
            trend[a:b] = _initial_trend(yn[a:b], p)
    w = np.ones(n)
    seasonal = np.zeros(n)
    for outer in range(3):
        for _ in range(2):
            pattern = _seasonal_pattern(yn - trend, p, w, median=(outer == 0))
            seasonal = pattern[np.arange(n) % p]
            trend = _trend_fit(yn - seasonal, window, w, bounds)
        if outer < 2:
            w = _bisquare(yn - trend - seasonal, wide)
    return trend, seasonal


def _prefer_multiplicative(yn: np.ndarray, trend_a: np.ndarray, seas_a: np.ndarray,
                           t_log: Optional[np.ndarray] = None,
                           s_log: Optional[np.ndarray] = None) -> bool:
    """Does the seasonal swing scale with the level (use the log-space decomposition)?

    Seasonality proportional to the level (sales, page views) cannot be
    described by a fixed additive pattern: on a growing level the pattern is
    too weak at the end and too strong at the start, and the misfit leaks into
    trend and residual (false anomalies at every December peak, a trend that
    rises at the end of a declining series).  If the data are multiplicative,
    the additive residual is ``r ~ s * (T - mean T) / mean T`` (``s`` the
    additive seasonal pattern), so ``r`` is regressed on that interaction
    term: the slope ``beta`` is ~1 for multiplicative and ~0 for additive
    seasonality.  The log-space decomposition is used when ``t >= 3.5``, the
    level moves enough (max/min of the trend >= 1.25) and its (winsorised)
    remainder is smaller than the additive one: with ``beta >= 0.35`` any
    reduction counts, with ``0 < beta < 0.35`` it must be clear (below
    :data:`_MULT_WEAK_SS` of the additive one).  A small ``beta`` with a large
    ``t`` is typical of long periods with few cycles (weekly data, period 52,
    3-4 years): the trend window is shorter than the period, so the additive
    trend soaks up part of the growing swing and the interaction left in the
    residual is small, although the log fit removes most of the misfit.
    Without ``t_log`` / ``s_log`` only the first, cheap part is checked.
    """
    n = len(yn)
    if np.min(trend_a) <= 0 or float(np.max(trend_a) / np.min(trend_a)) < 1.25:
        return False
    r_a = yn - trend_a - seas_a
    sd = _robust_sd(r_a)
    if sd <= _TINY * float(np.max(np.abs(yn))):
        return False
    med = float(np.median(r_a))
    rw = np.clip(r_a - med, -4.0 * sd, 4.0 * sd)
    tm = float(np.mean(trend_a))
    x = seas_a * (trend_a - tm) / tm
    sxx = float(np.sum(x * x))
    if sxx <= _TINY * _TINY:
        return False
    beta = float(np.sum(x * rw)) / sxx
    res = rw - beta * x
    se = math.sqrt(float(np.sum(res * res)) / max(n - 2, 1) / sxx)
    if not (beta > 0 and se > 0 and beta / se >= 3.5):
        return False
    if t_log is None or s_log is None:     # evidence only (the log fit is not done yet)
        return True
    r_m = yn - np.exp(t_log + s_log)
    if not np.all(np.isfinite(r_m)):
        return False
    rm = np.clip(r_m - float(np.median(r_m)), -4.0 * sd, 4.0 * sd)
    # a weak interaction slope needs a clearly better log fit (see the docstring)
    need = 1.0 if beta >= 0.35 else _MULT_WEAK_SS
    return bool(float(np.sum(rm * rm)) < need * float(np.sum(rw * rw)))


def _is_multiplicative(dec: Decomposition) -> bool:
    """True if ``dec`` came from the multiplicative (log-space) branch of :func:`decompose`.

    The additive seasonal component is exactly periodic; the multiplicative
    one is ``trend * (exp(s) - 1)`` with a periodic ``s``, i.e. ``seasonal /
    trend`` is periodic while ``seasonal`` itself is not.
    """
    p = int(dec.period)
    if p <= 1:
        return False
    tr = dec.trend.to_numpy(dtype=float)
    se = dec.seasonal.to_numpy(dtype=float)
    n = len(tr)
    if n < 2 * p or np.any(tr <= 0) or np.any(tr + se <= 0):
        return False
    big = float(np.max(np.abs(se)))
    if big <= 0 or float(np.max(np.abs(se[p:] - se[:-p]))) <= 1e-9 * max(big, float(np.max(tr))):
        return False
    ratio = se / tr
    return bool(float(np.max(np.abs(ratio[p:] - ratio[:-p]))) <= 1e-8 * (1.0 + float(np.max(np.abs(ratio)))))


def decompose(series: pd.Series, period: int, breaks: Optional[list[int]] = None) -> Decomposition:
    """Robust decomposition ``series = trend + seasonal + resid`` (exact, no NaNs).

    STL-like: the seasonal component is a per-phase robust average of the
    detrended series (a fixed pattern centred to sum to 0 over each period,
    shrunk towards 0 when the cycles disagree -- see
    :func:`_seasonal_pattern`); the trend is a local-linear LOESS of the
    deseasonalised series (window ~n/10, at least min(1.5*period, n/8)).  Two
    bisquare robustness iterations down-weight outliers.  The trend is
    extrapolated at both ends with one-sided local-linear fits (stabilised
    towards a wider fit when the end hook is not significant, see
    :func:`_loess`), so there are no NaNs anywhere.

    *Multiplicative seasonality*: for a strictly positive series with at
    least three full periods whose seasonal swing grows/shrinks with the
    level, the decomposition is done on ``log(series)`` and transformed back:
    ``trend = k * exp(t)`` with ``k`` the mean of ``exp(s)`` over one period
    (so the trend is the season-average level, not the geometric mean),
    ``seasonal = exp(t + s) - trend`` (the seasonal swing scales with the
    trend; ``seasonal / trend`` is periodic and averages 0 over a period),
    ``resid = series - trend - seasonal`` (see :func:`_prefer_multiplicative`).

    *Non-negative series* keep a non-negative trend (see :func:`_nonneg_floor`).

    ``breaks`` (optional): positions where the level may jump (level shifts
    found by :func:`detect_changepoints`); the trend is then fitted separately
    on each stretch so that it does not smooth across the step (which would
    otherwise turn the observations next to the step into false anomalies).

    If ``period <= 1`` or the series holds fewer than two full periods, the
    seasonal component is zero and ``Decomposition.period`` is 1.
    """
    if not isinstance(series, pd.Series):
        series = pd.Series(series)
    y = _clean_values(series)
    n = len(y)
    idx = series.index
    if n == 0:
        raise ValueError("cannot decompose an empty series")
    p = int(period) if period else 1
    use_season = p > 1 and n >= 2 * p
    window = _trend_window(n, p if use_season else 1)
    bounds = _segment_bounds(n, breaks)
    # work in units of max|y| so that the numeric tolerances are scale-free
    c = _unit(y)
    yn = y / c
    wide = _heavy_ties(yn)        # zero-inflated / step data: lenient robustness

    trend, seasonal = _decompose_core(yn, p, use_season, window, wide, bounds)
    trend = np.where(np.isfinite(trend), trend, yn)
    seasonal = np.where(np.isfinite(seasonal), seasonal, 0.0)
    multiplicative = False
    if (use_season and n >= 3 * p and bool(np.all(yn > 0))
            and _prefer_multiplicative(yn, trend, seasonal)):
        with np.errstate(all="ignore"):
            ly = np.log(yn)
            t_log, s_log = _decompose_core(ly, p, True, window, _heavy_ties(ly), bounds)
        if (np.all(np.isfinite(t_log)) and np.all(np.isfinite(s_log))
                and _prefer_multiplicative(yn, trend, seasonal, t_log, s_log)):
            multiplicative = True
            # exp(s) averages above 1 over a season (Jensen): scale the trend by that average so
            # that it is the season-average level (seasonal / trend averages 0 over a season, as
            # an additive pattern sums to 0) -- exp(t) alone is the geometric mean, ~4 % below the
            # data's level for a +-40 % swing, which every level comparison would then inherit
            k = float(np.mean(np.exp(s_log[:p])))
            trend = np.exp(t_log) * (k if math.isfinite(k) and k > 0 else 1.0)
            seasonal = np.exp(t_log + s_log) - trend
    if not multiplicative and float(np.min(yn)) >= 0.0:
        trend = _nonneg_floor(trend, yn - seasonal, window, bounds)

    trend = trend * c
    seasonal = seasonal * c
    resid = y - trend - seasonal
    name = series.name
    return Decomposition(
        trend=pd.Series(trend, index=idx, name=name),
        seasonal=pd.Series(seasonal, index=idx, name=name),
        resid=pd.Series(resid, index=idx, name=name),
        period=int(p if use_season else 1),
    )


# =========================================================================== trend tests


def theil_sen_slope(y: np.ndarray) -> float:
    """Theil-Sen slope (median of pairwise slopes) per period of ``y``.

    ``y`` is assumed regularly spaced (x = 0, 1, 2, ...); non-finite values are
    ignored.  Long inputs (> 1000 points) use a deterministic, evenly spaced
    subset of 1000 points to bound the O(n^2) cost.
    """
    y = np.asarray(y, dtype=float).ravel()
    x = np.arange(len(y), dtype=float)
    ok = np.isfinite(y)
    x, y = x[ok], y[ok]
    n = len(y)
    if n < 2:
        return 0.0
    if n > 1000:
        sel = np.unique(np.round(np.linspace(0, n - 1, 1000)).astype(int))
        x, y = x[sel], y[sel]
    i, j = np.triu_indices(len(y), 1)
    slopes = (y[j] - y[i]) / (x[j] - x[i])
    return _finite(np.median(slopes))


def _kendall_s(y: np.ndarray) -> int:
    """Mann-Kendall ``S = sum_{i<j} sign(y_j - y_i)`` (exact, ties count 0).

    Short inputs use the direct pairwise sum.  Long inputs use a bottom-up
    merge scheme: at block width ``w`` every pair whose members fall into the
    left / right halves of the same ``2w`` block is counted once with
    ``searchsorted`` on the sorted left halves -- O(n log^2 n) instead of
    O(n^2), so daily series of decades stay fast.
    """
    n = len(y)
    if n <= 1000:
        d = y[None, :] - y[:, None]
        return int(np.sign(d[np.triu_indices(n, 1)]).sum())
    _, rank = np.unique(y, return_inverse=True)
    rank = rank.astype(np.int64).ravel()
    big = np.int64(int(rank.max()) + 2)
    pos = np.arange(n, dtype=np.int64)
    s = 0
    w = 1
    while w < n:
        grp = pos // (2 * w)
        is_right = (pos // w) % 2 == 1
        if is_right.any():
            left_keys = np.sort(grp[~is_right] * big + rank[~is_right])
            g_r = grp[is_right]
            k_r = g_r * big + rank[is_right]
            g_start = np.searchsorted(left_keys, g_r * big, side="left")
            g_end = np.searchsorted(left_keys, (g_r + 1) * big, side="left")
            n_less = np.searchsorted(left_keys, k_r, side="left") - g_start
            n_more = g_end - np.searchsorted(left_keys, k_r, side="right")
            s += int(np.sum(n_less) - np.sum(n_more))
        w *= 2
    return s


def _mk_z(y: np.ndarray) -> tuple[float, float, int]:
    """``(tau, z, n)`` of the Mann-Kendall test (tie- and continuity-corrected ``z``)."""
    y = np.asarray(y, dtype=float).ravel()
    y = y[np.isfinite(y)]
    n = len(y)
    if n < 3:
        return 0.0, 0.0, n
    s = _kendall_s(y)
    _, counts = np.unique(y, return_counts=True)
    counts = counts[counts > 1].astype(float)
    tie = float(np.sum(counts * (counts - 1) * (2 * counts + 5)))
    var = (n * (n - 1) * (2 * n + 5) - tie) / 18.0
    tau = s / (n * (n - 1) / 2.0)
    if var <= 0:
        return _finite(tau), 0.0, n
    if s > 0:
        z = (s - 1) / math.sqrt(var)
    elif s < 0:
        z = (s + 1) / math.sqrt(var)
    else:
        z = 0.0
    return _finite(tau), _finite(z), n


def mann_kendall(y: np.ndarray) -> tuple[float, float]:
    """Mann-Kendall trend test with tie correction.

    Returns ``(tau, p)`` where ``tau = S / (n(n-1)/2)`` (Kendall's tau-a
    against time) and ``p`` is the two-sided p-value of the normal
    approximation with continuity correction.  Degenerate input (n < 3 or all
    values tied) gives ``(0.0, 1.0)``.  This is the classical test that
    assumes independent observations; :func:`compute_metrics` reports the
    serial-dependence-corrected version (:func:`_trend_test`).
    """
    tau, z, _ = _mk_z(y)
    p = math.erfc(abs(z) / math.sqrt(2.0))
    return _finite(tau), float(min(1.0, max(0.0, _finite(p, 1.0))))


# =========================================================================== serial dependence
#
# Real series (page views, prices, KPIs) rarely have white noise around their
# trend: deviations persist (AR-like) or accumulate (random walks).  Tests
# that assume independence then call noise a significant trend.  The noise
# around the trend is modelled as AR(1) with coefficient ``phi`` in [0, 1]
# (``phi = 1``: random walk); the trend tests below use it.


def _variogram_phi(rc: np.ndarray, ok: Optional[np.ndarray] = None) -> float:
    """Raw ``V(2) / V(1) - 1`` of a residual; > 1 means smooth (positively correlated) increments.

    Only pairs whose both points are flagged ``ok`` (not outlying) are used,
    so the misfit around a sharp feature does not pass for persistent noise.
    """
    if ok is None:
        ok = np.ones(len(rc), dtype=bool)
    m1 = ok[1:] & ok[:-1]
    m2 = ok[2:] & ok[:-2]
    if m1.sum() < 4 or m2.sum() < 4:
        return 0.0
    v1 = float(np.mean((rc[1:] - rc[:-1])[m1] ** 2))
    v2 = float(np.mean((rc[2:] - rc[:-2])[m2] ** 2))
    if v1 <= 0.0:
        return 0.0
    return v2 / v1 - 1.0


def _ar1_phi(r: np.ndarray, max_block: Optional[int] = None) -> float:
    """AR(1) coefficient of the noise ``r`` from its variogram: ``phi = V(2) / V(1) - 1``.

    For AR(1) noise ``V(k) = E[(r_t - r_{t-k})^2] ~ 1 - phi^k``, so the ratio
    is ``1 + phi``; for a random walk ``V(k) ~ k`` (ratio 2, ``phi = 1``).
    Variogram increments are insensitive to a slowly varying misfit of the
    trend (it adds ~k^2 terms that are tiny at lags 1-2), unlike the lag-1
    autocorrelation.  Points beyond 3.5 robust SDs (spikes, the misfit of a
    sharp feature) are left out of the variogram.  The estimate is
    significance-gated: below ~1.6 standard errors of the white-
    noise estimate (``sd ~ 1.8 / sqrt(n)``) it fades to 0, so white noise
    keeps ``phi = 0`` and the classical tests.

    *Smooth* short-range noise (interpolated gaps, rolling averages,
    oversampled data) looks like a random walk at the finest scale.  Two
    guards: (1) a ratio clearly above 2 (positively correlated increments)
    thins the residual (every m-th point, m = 2, 4, ... 16) until its
    increments look rough, with ``phi >= (m-1)/(m+1)`` (an effective sample
    size of ``n / m``); (2) a near-unit-root estimate (>= 0.8) is checked at
    coarse scales: the long-run variance inflation ``I(b) = b var(block means
    of size b) / var`` keeps growing for a random walk, but saturates for
    noise with a finite memory -- then ``phi = (I - 1) / (I + 1)`` (the AR(1)
    with the same inflation).  ``max_block`` caps the block size (the scale
    up to which the caller's residual retains persistent noise).
    Clipped to [0, 1].
    """
    r = np.asarray(r, dtype=float)
    r = r[np.isfinite(r)]
    n = len(r)
    if n < 12:
        return 0.0
    sd = _winsorized_sd(r)
    if sd <= _TINY * max(1.0, float(np.max(np.abs(r)))):
        return 0.0
    dev = r - float(np.median(r))
    ok = np.abs(dev) <= 3.5 * sd
    rc = np.clip(dev, -3.5 * sd, 3.5 * sd)
    if float(np.mean(np.diff(rc) ** 2)) <= _TINY * sd * sd:
        return 0.0
    m = 1
    raw = _variogram_phi(rc, ok)
    while raw > 1.4 and m < 16 and len(rc[::2 * m]) >= 24:
        m *= 2
        raw = _variogram_phi(rc[::m], ok[::m])
    phi = float(np.clip(raw, 0.0, 1.0))
    gate = 3.0 / math.sqrt(len(rc[::m]))
    phi *= _ramp(phi, 0.5 * gate, gate)
    if m > 1:
        phi = max(phi ** (1.0 / m), (m - 1.0) / (m + 1.0))
    if phi >= 0.8:
        cap = n // 16 if max_block is None else int(min(n // 16, max(1, max_block)))
        var0 = float(np.var(rc))
        infl = []
        b = 1
        while b <= cap and var0 > 0:
            k = n // b
            bm = rc[:k * b].reshape(k, b).mean(axis=1)
            infl.append(b * float(np.var(bm)) / var0)
            b *= 2
        if len(infl) >= 3 and infl[-1] <= 1.3 * infl[-2]:
            i_lr = max(infl[-1], 1.0)
            phi = min(phi, (i_lr - 1.0) / (i_lr + 1.0))
    return float(min(1.0, max(0.0, phi)))


def _unit_root_push(phi: float) -> float:
    """Push an estimated ``phi`` towards 1 when it is already close to a unit root.

    Near ``phi = 1`` every correction explodes (``(1 + phi) / (1 - phi)``) and
    the estimate is too imprecise to tell AR(0.9) from a random walk, which
    has no trend at all; erring on the random-walk side keeps random walks
    from being reported as significant trends.  Linear ramp: unchanged below
    0.6, 1.0 from 0.85 on (the estimate's sampling SD is ~0.1-0.2 at n = 100-300).
    """
    return float(min(1.0, phi + (1.0 - phi) * _ramp(phi, 0.6, 0.85)))


def _noise_model(z: np.ndarray, period: int,
                 resid: Optional[np.ndarray] = None) -> tuple[float, float]:
    """``(phi, d1_var)`` of the noise around the trend of the deseasonalised ``z``.

    The noise is the residual of a robust LOESS with three times the
    decomposition's trend window: a narrow smoother absorbs persistent noise
    into the trend and biases ``phi`` down.  A wide smoother, however, leaves
    the misfit of sharp features (a spike-shaped fad near an edge) in its
    residual, which looks like persistent noise too; so when the narrow
    residual ``resid`` (the decomposition's) shows less than half that
    persistence, its own estimate is used.  ``phi`` is then pushed towards a
    unit root by :func:`_unit_root_push`.  ``d1_var`` is the (winsorised) mean
    squared first difference of the noise (``2 sigma^2 (1 - phi)`` for AR(1)
    noise of variance ``sigma^2``; the innovation variance of a random walk).
    """
    z = np.asarray(z, dtype=float)
    n = len(z)
    if n < 12:
        return 0.0, 0.0
    key = (int(period), z.tobytes(), None if resid is None else np.asarray(resid, dtype=float).tobytes())
    hit = _NOISE_CACHE.get(key)
    if hit is not None:
        return hit
    out = _noise_model_uncached(z, period, resid)
    if len(_NOISE_CACHE) >= 8:
        _NOISE_CACHE.pop(next(iter(_NOISE_CACHE)))
    _NOISE_CACHE[key] = out
    return out


_NOISE_CACHE: dict = {}


def _noise_model_uncached(z: np.ndarray, period: int,
                          resid: Optional[np.ndarray]) -> tuple[float, float]:
    """Implementation of :func:`_noise_model` (which memoises the last few calls)."""
    n = len(z)
    q = _trend_window(n, period)
    qw = min(n, 3 * q)
    tw = _loess(z, qw)
    for _ in range(2):                    # robust: isolated outliers are not "noise"
        tw = _loess(z, qw, _bisquare(z - tw))
    r = z - tw
    phi = _ar1_phi(r, max_block=max(2, q // 2))
    if resid is not None and phi > 0.0:
        rn = np.asarray(resid, dtype=float)
        phi_n = _ar1_phi(rn, max_block=max(1, q // 3))
        if phi_n < 0.5 * phi:
            phi, r = phi_n, rn
    phi = _unit_root_push(phi)
    sd = _winsorized_sd(r)
    rc = np.clip(r - float(np.median(r)), -3.5 * sd, 3.5 * sd) if sd > _TINY else r
    return phi, float(np.mean(np.diff(rc) ** 2))


def _ar_gamma_like(phi: float, k: np.ndarray) -> np.ndarray:
    """``(1 - phi^k) / (1 - phi^2)`` (``k / 2`` at ``phi = 1``): minus the AR(1)
    autocovariance (unit innovations) up to a constant -- well defined at the
    unit root and enough for contrasts (weights summing to 0)."""
    k = np.asarray(k, dtype=float)
    if phi >= 1.0 - 1e-9:
        return k / 2.0
    return (1.0 - phi ** k) / (1.0 - phi * phi)


def _ols_slope_var(h: int, phi: float, sig_e2: float) -> float:
    """Variance of the OLS slope over ``h`` consecutive points under AR(1) noise.

    ``sig_e2`` is the innovation variance; ``phi = 0`` gives the textbook
    ``sigma^2 / sxx`` and ``phi = 1`` the random-walk value.  Exact:
    ``a' Gamma a / sxx^2`` with ``a = x - mean(x)`` (computed with the
    contrast form of :func:`_ar_gamma_like`, O(h)).
    """
    if h < 3 or sig_e2 <= 0.0:
        return 0.0
    m = (h - 1) / 2.0
    k = np.arange(1, h, dtype=float)
    nk = h - k                                    # number of pairs at lag k
    # c_k = sum_{i=0}^{h-1-k} (i - m)(i + k - m)
    s_sq = (nk - 1) * nk * (2 * nk - 1) / 6.0 - 2 * m * (nk - 1) * nk / 2.0 + nk * m * m
    s_lin = (nk - 1) * nk / 2.0 - nk * m
    ck = s_sq + k * s_lin
    sxx = h * (h * h - 1) / 12.0
    quad = -2.0 * float(np.sum(ck * _ar_gamma_like(phi, k)))
    return float(max(quad, 0.0) * sig_e2 / (sxx * sxx))


def _ar_path_var(phi: float, n: int) -> float:
    """Expected sample variance of an AR(1) path of length ``n`` (unit innovations).

    ``(2 / n^2) sum_k (n - k) g(k)`` with ``g`` from :func:`_ar_gamma_like`:
    ``~1 / (1 - phi^2)`` for a short memory, ``~n / 6`` for a random walk.
    """
    if n < 2:
        return 0.0
    k = np.arange(1, n, dtype=float)
    return float(2.0 / (n * n) * np.sum((n - k) * _ar_gamma_like(phi, k)))


def _mk_variance_ratio(phi: float, n: int) -> float:
    """Inflation of Var(S) of Mann-Kendall under AR(1) noise (Hamed & Rao 1998).

    ``1 + 2 / (n (n-1) (n-2)) sum_k (n-k)(n-k-1)(n-k-2) rho_k`` with the rank
    autocorrelation of Gaussian AR(1) noise ``rho_k = (6 / pi) asin(phi^k / 2)``.
    """
    if phi <= 0.0 or n < 4:
        return 1.0
    k = np.arange(1, n - 2, dtype=float)
    rho = (6.0 / math.pi) * np.arcsin(np.clip(phi ** k, 0.0, 1.0) / 2.0)
    w = (n - k) * (n - k - 1) * (n - k - 2)
    return float(max(1.0, 1.0 + 2.0 / (n * (n - 1.0) * (n - 2.0)) * float(np.sum(w * rho))))


def _drift_pvalue(z: np.ndarray) -> float:
    """Two-sided p-value of a non-zero drift of a random walk ``z``.

    t-test of the mean first difference (winsorised at 4 robust SDs) with a
    Newey-West (Bartlett) long-run variance, bandwidth ``4 (n/100)^(2/9)``.
    """
    d = np.diff(np.asarray(z, dtype=float))
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 5:
        return 1.0
    sd = _robust_sd(d)
    if sd <= _TINY * max(1.0, float(np.max(np.abs(z)))):
        mu = float(np.mean(d))
        return 0.0 if abs(mu) > _TINY else 1.0
    med = float(np.median(d))
    dc = med + np.clip(d - med, -4.0 * sd, 4.0 * sd)
    mu = float(np.mean(dc))
    e = dc - mu
    lag = int(min(n - 1, max(1, math.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))))
    lrv = float(np.mean(e * e))
    for j in range(1, lag + 1):
        lrv += 2.0 * (1.0 - j / (lag + 1.0)) * float(np.mean(e[j:] * e[:-j]))
    lrv = max(lrv, 0.25 * float(np.mean(e * e)))
    if lrv <= 0:
        return 1.0
    t = mu / math.sqrt(lrv / n)
    return float(min(1.0, max(0.0, math.erfc(abs(t) / math.sqrt(2.0)))))


def _trend_test(z: np.ndarray, phi: float) -> tuple[float, float]:
    """Mann-Kendall ``(tau, p)`` of ``z`` corrected for AR(1) noise ``phi``.

    The variance of S is inflated by :func:`_mk_variance_ratio` (Hamed-Rao).
    Near a unit root (``phi >= 0.9``) a trend can only be a drift, so the
    smaller of that p-value and the drift test on the first differences
    (:func:`_drift_pvalue`) is used (never below the classical p-value).
    ``phi = 0`` gives the classical :func:`mann_kendall`.
    """
    tau, zc, n = _mk_z(z)
    p = float(min(1.0, max(0.0, math.erfc(abs(zc) / math.sqrt(2.0)))))
    if phi <= 0.0 or n < 4:
        return tau, p
    za = abs(zc) / math.sqrt(_mk_variance_ratio(phi, n))
    p_ar = math.erfc(za / math.sqrt(2.0))
    if phi >= 0.9:
        y = np.asarray(z, dtype=float)
        p_ar = min(p_ar, _drift_pvalue(y[np.isfinite(y)]))
    return tau, float(min(1.0, max(p, p_ar)))


# =========================================================================== changepoints


def _despike(z: np.ndarray, k: float = 4.5) -> np.ndarray:
    """Replace isolated 1-2 point spikes by the 5-point running median.

    A running median preserves level shifts and kinks (unlike a smoother), so
    only genuine spikes -- residuals beyond ``k`` robust SDs -- are touched.
    """
    z = np.asarray(z, dtype=float)
    n = len(z)
    if n < 5:
        return z
    med = np.empty(n)
    med[2:n - 2] = np.median(np.lib.stride_tricks.sliding_window_view(z, 5), axis=1)
    # edges: robust (Theil-Sen) line through the first / last 5 points
    x5 = np.arange(5, dtype=float)
    for sl, pos in ((slice(0, 5), (0, 1)), (slice(n - 5, n), (3, 4))):
        seg = z[sl]
        b = theil_sen_slope(seg)
        a = float(np.median(seg - b * x5))
        for j in pos:
            med[sl.start + j] = a + b * j
    r = z - med
    # A running median equals the centre value whenever that value is the window
    # median (~20% of points for noise, all points of a constant stretch).  Those
    # exact zeros carry no information about the noise but would drag the MAD
    # down and make ordinary noise look like spikes, so they are left out.
    nz = r[np.abs(r) > 1e-9 * _scale(z)]
    if len(nz) < 3:
        return z
    sig = _winsorized_sd(nz)
    if sig <= _TINY:
        return z
    return np.where(np.abs(r - np.median(nz)) > k * sig, med, z)


def _segment_cost_matrix(z: np.ndarray, min_seg: int) -> np.ndarray:
    """``C[a, b]`` = RSS of an OLS line on ``z[a:b]`` (inf if shorter than ``min_seg``)."""
    n = len(z)
    x = np.arange(n, dtype=float) / max(n, 1)
    zeros = np.zeros(1)
    c1 = np.r_[zeros, np.cumsum(np.ones(n))]
    cx = np.r_[zeros, np.cumsum(x)]
    cxx = np.r_[zeros, np.cumsum(x * x)]
    cy = np.r_[zeros, np.cumsum(z)]
    cxy = np.r_[zeros, np.cumsum(x * z)]
    cyy = np.r_[zeros, np.cumsum(z * z)]
    a = np.arange(n + 1)[:, None]
    b = np.arange(n + 1)[None, :]
    cnt = c1[b] - c1[a]
    with np.errstate(divide="ignore", invalid="ignore"):
        sx = cx[b] - cx[a]
        sy = cy[b] - cy[a]
        sxx = cxx[b] - cxx[a] - sx * sx / cnt
        sxy = cxy[b] - cxy[a] - sx * sy / cnt
        syy = cyy[b] - cyy[a] - sy * sy / cnt
        rss = np.where(sxx > _TINY, syy - sxy * sxy / np.where(sxx > _TINY, sxx, 1.0), syy)
    rss = np.maximum(rss, 0.0)
    rss[(b - a) < min_seg] = np.inf
    rss[~np.isfinite(rss)] = np.inf
    return rss


def _segment(z: np.ndarray, max_cp: int, min_seg: int, penalty_factor: float = 4.0,
             abs_floor: float = 0.0, extra_dof: int = 0) -> list[int]:
    """Optimal piecewise-linear segmentation; returns break positions (segment starts).

    Segment-neighbourhood dynamic programming gives the exact best RSS for
    every number of breaks ``k <= max_cp``; ``k`` is then chosen by the
    BIC-like criterion ``n log(RSS_k / (n - p_k)) + penalty_factor * k * log(n)``
    with ``p_k = 3k + 2`` fitted parameters.  Each break costs ~3 parameters
    (location, intercept, slope), i.e. plain BIC would use 3; 4 is slightly
    stricter, which keeps the false-break rate on pure noise / straight lines
    near zero (residuals of real series are rarely white).  The degrees-of-
    freedom correction ``n - p_k`` matters for short series only: with
    ``min_seg = 4`` a 4-point line fits noise very well and plain BIC accepts
    spurious breaks in ~15% of white-noise series of length 8-20.
    ``extra_dof`` counts parameters already fitted to ``z`` before (``p - 1``
    for a deseasonalised series); a ``k`` that leaves no degrees of freedom is
    not considered.  With serially dependent noise (AR(1) ``phi`` of the
    residual of the most flexible fit, see :func:`_ar1_phi`) the likelihood
    term is scaled by the effective-sample-size factor ``(1 - phi) / (1 + phi)``
    -- otherwise random walks and persistent noise get the maximum number of
    breaks.  ``abs_floor`` is a noise-variance floor in the units of
    ``z`` so that float dust on a (nearly) noise-free series is not segmented.
    """
    n = len(z)
    if n < 2 * min_seg or max_cp < 1:
        return []
    sd = float(np.std(z))
    if sd <= _TINY:
        return []
    zs = (z - float(np.mean(z))) / sd
    cost = _segment_cost_matrix(zs, min_seg)
    k_max = int(min(max_cp, n // min_seg - 1))
    if k_max < 1:
        return []
    d_prev = cost[0].copy()           # best cost of [0, b) with 1 segment
    rss = [d_prev[n]]
    backs: list[np.ndarray] = []
    for _ in range(k_max):
        tot = d_prev[:, None] + cost
        arg = np.argmin(tot, axis=0)
        d_cur = tot[arg, np.arange(n + 1)]
        backs.append(arg)
        rss.append(d_cur[n])
        d_prev = d_cur
    def breaks_for(k: int) -> list[int]:
        out: list[int] = []
        end = n
        for j in range(k, 0, -1):
            start = int(backs[j - 1][end])
            out.append(start)
            end = start
        return sorted(out)

    # Serially dependent noise: n correlated points carry the information of
    # ~n (1 - phi) / (1 + phi) independent ones, so the likelihood term is
    # scaled down (penalty unchanged).  phi comes from the residual of the
    # most flexible fit (k_max breaks), which is ~pure noise when the breaks
    # are real; for a random walk it stays ~1 and no break survives.
    k_top = max((k for k, r in enumerate(rss) if math.isfinite(r)), default=0)
    eff = 1.0
    if k_top >= 1 and n >= 24:
        bnd = [0] + breaks_for(k_top) + [n]
        res = np.empty(n)
        for a, b in zip(bnd[:-1], bnd[1:], strict=True):
            xs = np.arange(a, b, dtype=float)
            ic, sl = _ols(zs[a:b], xs)
            res[a:b] = zs[a:b] - (ic + sl * xs)
        phi = _unit_root_push(_ar1_phi(res, max_block=max(2, min_seg // 4)))
        eff = (1.0 - phi) / (1.0 + phi)
    # zs has unit variance: ignore improvements below 0.1% noise (or abs_floor)
    floor = n * max(1e-6, abs_floor / (sd * sd))
    best_k, best_crit = 0, math.inf
    for k, r in enumerate(rss):
        if not math.isfinite(r):
            continue
        dof = n - (3 * k + 2) - max(0, int(extra_dof))
        if dof < 1:
            continue
        crit = eff * n * math.log(max(r, floor) / dof) + penalty_factor * k * math.log(n)
        if crit < best_crit - 1e-9:
            best_k, best_crit = k, crit
    return breaks_for(best_k)


def _classify_break(s1: float, s2: float, se1: float, se2: float, jump: float,
                    len1: int, len2: int, noise: float) -> str:
    """Kind of a break from neighbouring slopes and the level jump."""
    slope_effect = abs(s2 - s1) * min(len1, len2)
    if abs(jump) > slope_effect and abs(jump) >= 2.5 * noise:
        return "level_shift"
    a = 0 if abs(s1) <= 2.0 * se1 else (1 if s1 > 0 else -1)
    b = 0 if abs(s2) <= 2.0 * se2 else (1 if s2 > 0 else -1)
    if b > 0 and a <= 0:
        return "reversal_up"
    if b < 0 and a >= 0:
        return "reversal_down"
    if a == 0 and b == 0:
        if abs(jump) >= slope_effect:
            return "level_shift"
        if s2 >= s1:
            return "reversal_up" if s2 > 0 >= s1 else ("acceleration" if s2 > 0 else "deceleration")
        return "reversal_down" if s2 < 0 <= s1 else ("acceleration" if s2 < 0 else "deceleration")
    if b == 0:
        return "deceleration"
    return "acceleration" if abs(s2) > abs(s1) else "deceleration"


def _changepoints_from(z: np.ndarray, index: pd.DatetimeIndex, period: int,
                       max_changepoints: int, min_segment: Optional[int]) -> list[Changepoint]:
    """Changepoints of a deseasonalised array ``z`` (core of :func:`detect_changepoints`)."""
    z = np.asarray(z, dtype=float)
    n = len(z)
    c = _unit(z)            # work in units of max|z|; slopes are scaled back below
    z = z / c
    abs_floor = (1e-4 * _scale(z)) ** 2
    if min_segment is None:
        min_seg = max(4, int(period) // 2, n // 12)
    else:
        min_seg = max(3, int(min_segment))
    if n < 2 * min_seg or max_changepoints < 1 or n < 8:
        return []
    # robustify against spikes so single outliers cannot create breaks
    z = _despike(z)
    if float(np.std(z)) <= max(_TINY, 1e-6 * _scale(z)):
        return []

    # long series: segment block means, then map back
    max_points = 800
    if n > max_points:
        f = int(math.ceil(n / 600.0))
        nb = int(math.ceil(n / f))
        pad = nb * f - n
        zb = np.r_[z, np.full(pad, np.nan)].reshape(nb, f)
        zb = np.nanmean(zb, axis=1)
        seg_min = max(3, int(math.ceil(min_seg / f)))
        breaks = [b * f for b in _segment(zb, max_changepoints, seg_min, abs_floor=abs_floor)]
        breaks = [b for b in breaks if min_seg <= b <= n - min_seg]
    else:
        # deseasonalising (same rule as decompose) used up p - 1 degrees of freedom
        p = int(period) if period else 1
        season_dof = p - 1 if (p > 1 and n >= 2 * p) else 0
        breaks = _segment(z, max_changepoints, min_seg, abs_floor=abs_floor,
                          extra_dof=season_dof)
    if not breaks:
        return []

    bounds = [0] + list(breaks) + [n]
    fits = []
    for a, b in zip(bounds[:-1], bounds[1:], strict=True):
        xs = np.arange(a, b, dtype=float)
        icpt, slope = _ols(z[a:b], xs)
        res = z[a:b] - (icpt + slope * xs)
        dof = max(1, (b - a) - 2)
        s_res = math.sqrt(float(np.sum(res * res)) / dof)
        sxx = float(np.sum((xs - xs.mean()) ** 2))
        se = s_res / math.sqrt(sxx) if sxx > _TINY else math.inf
        fits.append((icpt, slope, se, b - a))
    all_res = np.concatenate([
        z[a:b] - (fits[j][0] + fits[j][1] * np.arange(a, b))
        for j, (a, b) in enumerate(zip(bounds[:-1], bounds[1:], strict=True))
    ])
    noise = max(_robust_sd(all_res), _TINY)
    scale = _scale(z)

    out: list[Changepoint] = []
    for j, k in enumerate(breaks):
        i1, s1, se1, l1 = fits[j]
        i2, s2, se2, l2 = fits[j + 1]
        jump = (i2 + s2 * k) - (i1 + s1 * k)
        se1 = se1 if math.isfinite(se1) else abs(s1) + 1.0
        se2 = se2 if math.isfinite(se2) else abs(s2) + 1.0
        kind = _classify_break(s1, s2, se1, se2, jump, l1, l2, noise)
        if kind == "level_shift":
            mag = abs(jump) / scale
        else:
            mag = abs(s2 - s1) * l2 / scale
        out.append(Changepoint(
            date=pd.Timestamp(index[k]), index=int(k),
            slope_before=_finite(s1 * c), slope_after=_finite(s2 * c),
            kind=kind, magnitude=_finite(abs(mag)),
        ))
    out.sort(key=lambda c: c.index)
    return out


def detect_changepoints(series: pd.Series, period: int = 1,
                        max_changepoints: int = 5,
                        min_segment: Optional[int] = None) -> list[Changepoint]:
    """Structural breaks of the (deseasonalised) trend.

    The series is deseasonalised with :func:`decompose` (if ``period > 1``),
    isolated spikes are replaced by a running median, and an optimal
    piecewise-linear segmentation (independent OLS line per segment) is found
    by dynamic programming with a BIC-like penalty, so a straight
    line plus noise yields no (or very few) breaks.  Each break carries the
    slopes of the neighbouring segments, a ``kind`` from
    :data:`trendai.types.CHANGEPOINT_KINDS` and a unitless ``magnitude``
    (|slope change| x following segment length / series scale, or level jump /
    scale for level shifts).  Default ``min_segment = max(4, period // 2, n // 12)``.
    Sorted by date.
    """
    if not isinstance(series, pd.Series):
        series = pd.Series(series)
    n = len(series)
    if n < 8:
        return []
    dec = decompose(series, period)
    z = (dec.trend + dec.resid).to_numpy(dtype=float)
    return _changepoints_from(z, pd.DatetimeIndex(series.index), period,
                              max_changepoints, min_segment)


# =========================================================================== anomalies


def _trend_rows(w: np.ndarray, q: int) -> tuple[np.ndarray, np.ndarray]:
    """``(L_ii, sum_{j != i} L_ij^2)`` of the trend LOESS (:func:`_loess`) rows.

    ``L`` is the equivalent kernel of the local-linear fit with robustness
    weights ``w`` over ``q`` points; near the ends it is blended with the
    three times wider one-sided fit as :func:`_loess` does for a noise-driven
    end hook (full shrinkage, the white-noise case).  As in :func:`_loess`
    the rows are evaluated at every ``max(1, q // 10)``-th point only: the
    own weight is ``w_i`` times a smooth factor and the other weights' sum of
    squares is smooth, so both are interpolated in between.
    """
    n = len(w)
    jump = max(1, q // 10)
    at = np.arange(n) if jump == 1 else np.unique(np.r_[np.arange(0, n, jump), n - 1])
    _, _, kern = _local_fit(np.zeros(n), w, at, q, want_weights=True)
    idx, lw = kern  # type: ignore[misc]
    own = np.take_along_axis(lw, (at - idx[:, 0])[:, None], axis=1)[:, 0]
    ss = np.sum(lw * lw, axis=1)
    q3 = int(min(n, 3 * q - 2))
    if q3 > q and n >= 5:
        hw = (q - 1) / 2.0
        lo = np.clip(at - (q - 1) // 2, 0, n - q)
        asym = np.clip(np.abs(at - (lo + hw)) / max(hw, 1.0), 0.0, 1.0)
        edge = np.flatnonzero(asym > 1e-9)
        if len(edge):
            _, _, kern3 = _local_fit(np.zeros(n), w, at[edge], q3, want_weights=True)
            idx3, lw3 = kern3  # type: ignore[misc]
            lam = (1.0 - asym[edge])[:, None]
            row = (1.0 - lam) * lw3
            pos = np.clip(idx[edge] - idx3[:, :1], 0, q3 - 1)   # narrow window nested in the wide one
            np.put_along_axis(row, pos, np.take_along_axis(row, pos, axis=1) + lam * lw[edge], axis=1)
            own[edge] = np.take_along_axis(row, (at[edge] - idx3[:, 0])[:, None], axis=1)[:, 0]
            ss[edge] = np.sum(row * row, axis=1)
    factor = own / np.maximum(w[at], _TINY)
    others = np.maximum(ss - own * own, 0.0)
    if jump > 1:
        grid = np.arange(n, dtype=float)
        factor = np.interp(grid, at.astype(float), factor)
        others = np.interp(grid, at.astype(float), others)
    return w * factor, others


def _resid_variance_factor(rn: np.ndarray, dn: np.ndarray, sn: np.ndarray, period: int,
                           wide: bool, breaks: Optional[list[int]] = None) -> np.ndarray:
    """Variance of each residual relative to the noise variance, ``F_i``.

    The decomposition is (approximately) a linear smoother with the robustness
    weights ``w``: fitted = ``L y`` with ``L`` = trend LOESS + per-phase
    seasonal average (shrunk as in :func:`_seasonal_pattern`).  For white
    noise the residual ``r_i`` then has variance ``F_i * s^2`` with
    ``F_i = (1 - L_ii)^2 + sum_{j != i} L_ij^2``.  Ordinary points (weight ~1)
    are partly absorbed by their own fit -- with 3 cycles a weekly phase mean
    contains a third of each point, a 5-point trend window ~30 % -- so their
    residuals shrink (``F < 1``), while a point the robust fit down-weights
    keeps its full deviation plus the noise of the fit (``F > 1``).  Scaling
    the z-scores by the median residual alone therefore over-states every
    large deviation (pure noise: ~2.5 flags per weekly series of 3 years);
    ``r_i / sqrt(F_i)`` puts all points on one scale (studentised residuals).

    ``rn``: residual (log residual for a multiplicative fit), ``dn``: the
    detrended values and ``sn``: the seasonal component in the same units.
    """
    n = len(rn)
    w = _bisquare(rn - float(np.median(rn)), wide)
    p = int(period)
    window = _trend_window(n, p)
    h_t, v_t = np.zeros(n), np.zeros(n)
    for a, b in _segment_bounds(n, breaks):
        h_t[a:b], v_t[a:b] = _trend_rows(w[a:b], int(max(2, min(window, b - a))))
    h_s, v_s = np.zeros(n), np.zeros(n)
    if p > 1 and n >= 2 * p:
        ph = np.arange(n) % p
        sw = np.bincount(ph, weights=w, minlength=p)
        sw2 = np.bincount(ph, weights=w * w, minlength=p)
        safe = np.maximum(sw, _TINY)
        raw = np.bincount(ph, weights=w * dn, minlength=p) / safe
        raw = raw - raw.mean()
        pat = np.bincount(ph, weights=sn, minlength=p) / np.maximum(np.bincount(ph, minlength=p), 1)
        pat = pat - pat.mean()
        den = float(raw @ raw)
        # how much of the weighted phase means the (shrunk) seasonal pattern kept
        shrink = float(np.clip((pat @ raw) / den, 0.0, 1.0)) if den > _TINY else 0.0
        h_s = shrink * w / safe[ph]
        v_s = shrink * shrink * np.maximum(sw2[ph] - w * w, 0.0) / safe[ph] ** 2
    h = h_t + h_s - h_t * h_s
    f = (1.0 - h) ** 2 + v_t + v_s
    return np.where(np.isfinite(f), np.clip(f, 1e-3, 4.0), 1.0)


def _intermittent_anomalies(series: pd.Series, y: np.ndarray, threshold: float) -> list[Anomaly]:
    """Unusually *large* demands of an intermittent series (see :func:`_is_intermittent`).

    Every non-zero value of mostly-zero demand is far from the smoothed demand
    rate, so residual z-scores flag ordinary orders (39 of 40 iid demand series
    had "anomalies" of +1,000 %).  What can be unusual is the size of a
    demand: the robust z-score of ``log(size)`` among the non-zero values
    (``expected`` = their median, the typical demand size).  Only upward
    deviations count (a small order, or a zero, is ordinary), and at least 5
    non-zero values are needed.
    """
    nz = np.flatnonzero(y > 0)
    if len(nz) < 5:
        return []
    lv = np.log(y[nz])
    med = float(np.median(lv))
    sd = _robust_sd(lv)
    if sd <= 1e-9 * max(1.0, abs(med)):
        return []
    zs = (lv - med) / sd
    typical = float(np.exp(med))
    return [Anomaly(date=pd.Timestamp(series.index[i]), index=int(i), value=_finite(y[i]),
                    expected=_finite(typical), zscore=_finite(z))
            for i, z in zip(nz, zs) if z >= threshold]


def detect_anomalies(series: pd.Series, decomposition: Decomposition,
                     threshold: float = 3.5, breaks: Optional[list[int]] = None) -> list[Anomaly]:
    """Points whose residual has a robust z-score with ``|z| >= threshold``.

    z = (studentised resid - median) / robust scale, where the scale is the
    MAD-based SD.  The residuals are first divided by ``sqrt(F_i)``, their
    relative variance under pure noise (see :func:`_resid_variance_factor`):
    with few seasonal cycles or a short trend window the fit absorbs part of
    every ordinary point, which used to make the median residual -- and the
    scale -- too small for the points the robust fit left out (white noise:
    2.5 flags per weekly series of 156 points, 1.6 per 36 months).  ``breaks``
    are the level shifts of a piecewise decomposition (the trend windows stop
    there; see :func:`decompose`).
    For longer series (n >= 30) a *local* MAD (rolling window of
    ``max(25, 2 x trend window)`` points) is used, floored at a quarter of the
    global value, so that series whose noise grows with their level (page
    views, sales: multiplicative noise) do not flag every point of a busy
    period.  The local value is shrunk towards the global one in log space
    by its sampling noise (positive-part James-Stein with a 4x variance, as
    the edge LOESS): a window's MAD of ~30 points is itself ±20 % noisy, which
    otherwise added flags (and hid spikes) on homogeneous noise.  The scale
    never collapses onto a point mass: it is at least
    ``0.5 * q90 / 1.645`` of the absolute deviations (the full ``q90 / 1.645``
    for heavily tied data such as zero-inflated counts, where otherwise every
    ordinary non-zero count would be flagged).  For a multiplicative
    decomposition (see :func:`decompose`) the residual is taken in log space,
    ``log(value / expected)``, i.e. the z-scores are relative.  Intermittent
    demand (mostly zeros, :func:`_is_intermittent`) is judged by the size of
    each demand instead (:func:`_intermittent_anomalies`).  Sorted by date.
    """
    y = _clean_values(series)
    r = decomposition.resid.to_numpy(dtype=float)
    n = min(len(y), len(r))
    if n < 3:
        return []
    y, r = y[:n], r[:n]
    if _is_intermittent(y, _season_of(series, decomposition)):
        return _intermittent_anomalies(series, y, threshold)
    r = np.where(np.isfinite(r), r, 0.0)
    trend = decomposition.trend.to_numpy(dtype=float)[:n]
    seasonal = decomposition.seasonal.to_numpy(dtype=float)[:n]
    expected = trend + seasonal
    c = _unit(y, r)                       # scale-free tolerances
    rn = r / c
    dn, sn = (y - trend) / c, seasonal / c
    if len(decomposition.trend) == n and _is_multiplicative(decomposition) and np.all(y > 0):
        with np.errstate(all="ignore"):
            rel = np.log(y / expected)
            d_log, s_log = np.log(y / trend), np.log(expected / trend)
        if np.all(np.isfinite(rel)):
            rn = rel
            if np.all(np.isfinite(d_log)) and np.all(np.isfinite(s_log)):
                dn, sn = d_log, s_log
    heavy = _heavy_ties(y)
    if np.all(np.isfinite(dn)) and np.all(np.isfinite(sn)):
        rn = (rn - float(np.median(rn))) / np.sqrt(
            _resid_variance_factor(rn, dn, sn, int(decomposition.period), heavy, breaks))
    med = float(np.median(rn))
    sd = _robust_sd(rn)
    # point-mass guard (see _robust_sd): scale >= q_coef * q90(|dev|).  For
    # heavily tied data (zero-inflated counts) the full Gaussian-consistent
    # q90 / 1.645 is used, otherwise every ordinary non-zero count is flagged.
    q_coef = 0.608 if heavy else 0.304
    sd = max(sd, q_coef * float(np.quantile(np.abs(rn - med), 0.9)))
    if sd <= max(_TINY, 1e-9 * _scale(y / c)):
        return []
    scale = np.full(n, sd)
    if n >= 30:
        win = int(min(n, max(25, 2 * _trend_window(n, decomposition.period))))
        dev = pd.Series(np.abs(rn - med))
        roll = dev.rolling(win, center=True, min_periods=max(5, win // 3))
        local = np.maximum(1.4826 * roll.median().to_numpy(dtype=float),
                           q_coef * roll.quantile(0.9).to_numpy(dtype=float))
        local = pd.Series(local).bfill().ffill().to_numpy(dtype=float)
        local = np.where(np.isfinite(local) & (local > 0), local, sd)
        # shrink log(local / global) by its sampling variance (~1.36 / m for a MAD of m points)
        m = pd.Series(roll.count().to_numpy(dtype=float)).replace(0.0, np.nan).bfill().ffill()
        se2 = 1.36 / np.maximum(m.fillna(float(win)).to_numpy(dtype=float), 1.0)
        ell = np.log(local / sd)
        g = np.clip(1.0 - _EDGE_SHRINK * se2 / np.maximum(ell * ell, 1e-300), 0.0, 1.0)
        scale = np.maximum(sd * np.exp(g * ell), 0.25 * sd)
    zs = (rn - med) / scale
    out = []
    for i in np.flatnonzero(np.abs(zs) >= threshold):
        out.append(Anomaly(date=pd.Timestamp(series.index[i]), index=int(i),
                           value=_finite(y[i]), expected=_finite(expected[i]),
                           zscore=_finite(zs[i])))
    return out


# =========================================================================== metrics


def _recent_window(n: int, freq: str) -> int:
    """Window (in periods) used for 'recent' slopes in the momentum metric."""
    sl = _data.season_length(freq)
    return int(max(1, min(max(3, min(sl, n // 4)), n - 1)))


def _winsorize(r: np.ndarray, k: float = 3.5) -> np.ndarray:
    """``r`` clipped at ``median +- k`` robust SDs (the anomaly threshold).

    Summary statistics computed from the residual (volatility, trend and
    seasonal strength) must not be dominated by the very spikes that are
    reported as anomalies: four one-day news spikes used to turn a clear
    weekly pattern into "almost no seasonality" and 11% noise into 63%.
    For Gaussian residuals only ~0.05% of the points are touched.
    """
    r = np.asarray(r, dtype=float)
    if len(r) < 3:
        return r
    sd = _robust_sd(r)
    med = float(np.median(r))
    if sd <= _TINY * max(1.0, float(np.max(np.abs(r)))):
        return r
    return med + np.clip(r - med, -k * sd, k * sd)


def _paulson_z(F: float, df1: float, df2: float) -> float:
    """Normal approximation (Paulson 1942) of the upper tail of an ``F(df1, df2)`` statistic."""
    a = 2.0 / (9.0 * df1)
    b = 2.0 / (9.0 * df2)
    f3 = F ** (1.0 / 3.0)
    return ((1.0 - b) * f3 - (1.0 - a)) / math.sqrt(b * f3 * f3 + a)


def _acf_at(x: np.ndarray, lag: int) -> float:
    """Sample autocorrelation of ``x`` at ``lag`` (0 for a constant or too short series)."""
    xc = x - float(np.mean(x))
    den = float(np.sum(xc * xc))
    if den <= 0 or lag >= len(x):
        return 0.0
    return float(np.sum(xc[lag:] * xc[:-lag])) / den


def _seasonality_evidence(y: np.ndarray, p: int) -> float:
    """How firmly the data show a repeating pattern of period ``p``: a factor in [0, 1].

    The same test the forecasting models use to decide on seasonal adjustment
    (``models._is_seasonal``), as a smooth score: with at least 2.5 seasons the
    series is detrended by a centred (2x)p moving average and the remainder
    must show between-phase differences (one-way ANOVA F, degrees of freedom
    deflated for serial correlation of the within-phase residual, Paulson's
    normal approximation ``z``) that repeat from one season to the next (lag-p
    autocorrelation ``r_p`` of the remainder).  The score is
    ``ramp(z, 1.645, 2.326) * ramp(r_p, 0, 0.1)``: 0 below the 95 % level, 1
    from the 99 % level with ``r_p >= 0.1`` (``_is_seasonal``'s rule).  With 2
    to 2.5 seasons (too short for the moving average) an ANCOVA F-test of the
    phase effects given one common slope is used.  Pure noise with 3
    cycles otherwise reads as a "clear" pattern (strength 0.67) about once in
    four.
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    if p <= 1 or n < 2 * p or not np.all(np.isfinite(y)):
        return 0.0
    ref = float(np.max(np.abs(y)))
    if ref <= 0 or float(np.ptp(y)) <= 1e-12 * ref:
        return 0.0
    y = y / ref
    if 2 * n >= 5 * p:
        w = np.r_[0.5, np.ones(p - 1), 0.5] / p if p % 2 == 0 else np.ones(p) / p
        ma = np.convolve(y, w, mode="valid")
        off = len(w) // 2
        x = y[off:off + len(ma)] - ma
        ph = (np.arange(len(ma)) + off) % p
        cnt = np.bincount(ph, minlength=p).astype(float)
        k = int((cnt > 0).sum())
        df1, df2 = k - 1, len(x) - k
        if df1 < 1 or df2 < 2:
            return 0.0
        means = np.bincount(ph, weights=x, minlength=p) / np.maximum(cnt, 1.0)
        ssb = float(np.sum(cnt * (means - float(np.mean(x))) ** 2))
        e = x - means[ph]
        ssw = float(np.sum(e * e))
    else:
        # 2-2.5 seasons: too short for the moving average (models._is_seasonal falls back to
        # a lag-p autocorrelation test that misses even a clean sine here).  ANCOVA instead:
        # phase means plus one common slope, F-test of the phase effects given the slope
        # (a plain straight line would absorb part of the pattern over two cycles)
        t = np.arange(n, dtype=float)
        ph = np.arange(n) % p
        cnt = np.bincount(ph, minlength=p).astype(float)
        yw = y - (np.bincount(ph, weights=y, minlength=p) / cnt)[ph]
        tw = t - (np.bincount(ph, weights=t, minlength=p) / cnt)[ph]
        stt = float(np.sum(tw * tw))
        b = float(np.sum(yw * tw)) / stt if stt > _TINY else 0.0
        e = yw - b * tw
        ssw = float(np.sum(e * e))
        a0, b0 = _ols(y)
        ssr = float(np.sum((y - a0 - b0 * t) ** 2))
        ssb = max(ssr - ssw, 0.0)
        df1, df2 = p - 1, n - p - 1
        if df2 < 2:
            return 0.0
        x = y - b * t
    if ssb <= 0:
        return 0.0
    if ssw <= 1e-12 * ssb:
        z = math.inf
    else:
        r1 = min(max(float(np.sum(e[1:] * e[:-1])) / ssw, 0.0), 0.98)
        kappa = (1.0 + r1 * r1) / (1.0 - r1 * r1)
        z = _paulson_z((ssb / df1) / (ssw / df2), max(df1 / kappa, 1.0), max(df2 / kappa, 2.0))
    return _ramp(z, 1.645, 2.326) * _ramp(_acf_at(x, p), 0.0, 0.1)


def compute_metrics(series: pd.Series, freq: str, decomposition: Decomposition,
                    anomalies: Optional[list[Anomaly]] = None) -> TrendMetrics:
    """Descriptive statistics of the past and present (see :class:`TrendMetrics`).

    Growth numbers use the trend component; ``None`` when the base is <= 0 or
    ~0 (below 1% of the mean absolute level), when the trend crossed zero
    (end < 0 < base), when a series with negative values has a base below
    half the standard deviation of its observations (see
    :func:`pct_change_undefined`), and for intermittent demand (mostly zeros,
    :func:`_is_intermittent`).  ``cagr_pct`` additionally
    needs a span of about a year (>= 0.95 years).  ``momentum`` is the recent
    trend slope (OLS over the last ``max(3, min(season_length, n//4))``
    periods) minus the long-run Theil-Sen ``slope``, in % of the *recent* trend
    level (mean ``|trend|`` over that window, at least a tenth of the mean
    level) per period -- so it can be read against the current growth rate.
    ``acceleration`` is the change of the recent slope between the previous and
    the latest window, in % of the mean level.  ``slope``, ``mk_*`` and
    ``direction`` are computed on the deseasonalised series.

    * ``seasonal_strength`` is the usual ``1 - var(resid) / var(seasonal +
      resid)``, scaled by how firmly a significance test of the repeating
      pattern supports it (:func:`_seasonality_evidence`, 0 below the 95 %
      level): with 2-3 cycles noise alone leaves a sizeable pattern.

    * ``mk_pvalue`` is corrected for serial dependence of the noise around the
      trend (AR(1) variance inflation, drift test near a unit root; see
      :func:`_trend_test`): persistent noise and random walks are not reported
      as significant trends.  ``mk_tau`` is the plain Kendall tau.
    * ``volatility`` and the strengths use the residual winsorised at 3.5
      robust SDs (:func:`_winsorize`), so isolated spikes do not dominate them;
      for a multiplicative decomposition the strengths are computed on the log
      components (``log trend``, ``log(1 + seasonal / trend)``, log residual).
    * ``peak_value`` / ``peak_date`` / ``drawdown_pct`` ignore isolated upward
      spikes flagged by :func:`detect_anomalies` (a one-day spike is reported
      as an anomaly, not as the level the series fell from) -- from 16
      observations on, and never the first or last observation, which cannot
      be confirmed as one-off.  ``anomalies``: the already detected anomalies
      of ``series`` (detected here if omitted).  ``drawdown_pct`` is always a
      number (typed ``float``); for a series that crossed zero or peaked near
      zero it is no meaningful percentage --
      ``pct_change_undefined(peak_value, last_value, series)`` tells.
    """
    y_raw = _clean_values(series)
    n = len(y_raw)
    if n == 0:
        raise ValueError("empty series")
    # ratios / tolerances are computed in units of max|y| (scale-free); the
    # data-unit outputs (values, slope) are taken from / scaled back to raw units
    c = _unit(y_raw)
    y = y_raw / c
    trend = decomposition.trend.to_numpy(dtype=float) / c
    seas = decomposition.seasonal.to_numpy(dtype=float) / c
    resid = decomposition.resid.to_numpy(dtype=float) / c
    z = trend + resid
    idx = series.index
    mean = float(np.mean(y_raw))
    level = _level(y)
    abs_ref = float(np.mean(np.abs(y)))

    # growth numbers (trend based); for data with negative values a base closer to 0 than
    # half a standard deviation, or a level that crossed zero, gives no percentage
    floor = _pct_floor(y)
    total = _pct_change(trend[-1], trend[0], abs_ref, floor) if n >= 2 else None
    cagr: Optional[float] = None
    years = (n - 1) / _data.periods_per_year(freq)
    thr = 0.01 * abs_ref
    if (years >= 0.95 and trend[0] > thr and trend[-1] > thr and trend[0] > 0 and trend[-1] > 0
            and trend[0] >= floor):
        lg = math.log(trend[-1] / trend[0]) / years
        if lg < 50:
            cagr = _finite((math.exp(lg) - 1.0) * 100.0, default=float("nan"))
            if not math.isfinite(cagr):
                cagr = None
            elif abs(cagr) < 1e-9:
                cagr = 0.0
    lag = 1 if freq == "Y" else _data.season_length(freq)
    lag = min(lag, n - 1)
    recent = _pct_change(trend[-1], trend[-1 - lag], abs_ref, floor) if lag >= 1 else None
    if _is_intermittent(y_raw, _data.season_length(freq)):
        # mostly zeros: the trend is a demand rate near 0 that wanders with the timing of
        # a few orders -- its %-changes (+3,192 %, CAGR +240 %) are meaningless
        total = cagr = recent = None

    slope_n = theil_sen_slope(z)
    slope = slope_n * c
    slope_pct = slope_n / level * 100.0

    r = _recent_window(n, freq)
    level_recent = level
    if n >= 2:
        seg_r = trend[-(r + 1):]
        _, s_recent = _ols(seg_r)
        # momentum is expressed against the *current* trend level: against the mean of
        # the whole history a fast riser read "+15.7 %p per month" while its trend grew
        # ~4 % a month.  Floored at a tenth of the mean level (a trend dying out to ~0).
        level_recent = max(float(np.mean(np.abs(seg_r))), 0.1 * level)
    else:
        s_recent = 0.0
    momentum = (s_recent - slope_n) / level_recent * 100.0
    if n >= 2 * r + 1:
        _, s_prev = _ols(trend[-(2 * r + 1):-r])
        accel = (s_recent - s_prev) / level * 100.0
    else:
        accel = 0.0

    resid_w = _winsorize(resid)
    vol = float(np.std(resid_w)) / level if n >= 2 else 0.0
    tt, ss, rr = trend, seas, resid_w
    if _is_multiplicative(decomposition) and np.all(y > 0):
        with np.errstate(all="ignore"):
            lt = np.log(trend)
            ls = np.log1p(seas / trend)
            lr = np.log(y / (trend + seas))
        if np.all(np.isfinite(lt)) and np.all(np.isfinite(ls)) and np.all(np.isfinite(lr)):
            tt, ss, rr = lt, ls, _winsorize(lr)
    # the series the seasonality test sees: spikes winsorised like the strengths' residual
    y_season = tt + ss + rr
    var_r = float(np.var(rr))
    var_tr = float(np.var(tt + rr))
    var_sr = float(np.var(ss + rr))
    t_strength = max(0.0, 1.0 - var_r / var_tr) if var_tr > _TINY else 0.0
    s_strength = (max(0.0, 1.0 - var_r / var_sr)
                  if (var_sr > _TINY and decomposition.period > 1) else 0.0)
    if s_strength > 0.0:
        # with few cycles noise leaves a sizeable "pattern" (strength 0.4-0.7 for a quarter
        # of non-seasonal 36-month series): count it only as far as a significance test
        # of the repeating pattern supports it (the forecasting models' own test)
        s_strength *= _seasonality_evidence(y_season, int(decomposition.period))

    phi, _ = _noise_model(z, decomposition.period, resid)
    tau, p = _trend_test(z, phi)
    direction = "flat"
    if p < 0.05:
        direction = "up" if tau > 0 else ("down" if tau < 0 else "flat")

    # peak / drawdown on the observations, ignoring isolated upward spikes -- but only
    # where a spike can be told from structure: not on short series (a 5-point trend
    # window on 8-15 points cannot follow a real peak, which then looks like a spike),
    # and never at the first or last observation (data on one side only: the latest
    # value of an accelerating series may be a new level, not a one-off)
    cand = np.ones(n, dtype=bool)
    if n >= 16:
        found = detect_anomalies(series, decomposition) if anomalies is None else anomalies
        spikes = [a.index for a in found if a.zscore > 0]
        cand[[i for i in spikes if 0 < i < n - 1]] = False
        if not cand.any():
            cand[:] = True
    i_peak = int(np.flatnonzero(cand)[int(np.argmax(y[cand]))])
    peak_n = float(y[i_peak])
    denom = abs(peak_n) if abs(peak_n) > _TINY else (abs_ref if abs_ref > _TINY else 1.0)
    drawdown = min(0.0, (float(y[-1]) - peak_n) / denom * 100.0)
    peak = float(y_raw[i_peak])
    last = float(y_raw[-1])

    return TrendMetrics(
        n_obs=int(n), freq=freq, start=pd.Timestamp(idx[0]), end=pd.Timestamp(idx[-1]),
        last_value=_finite(last), mean=_finite(mean),
        total_change_pct=total, cagr_pct=cagr, recent_change_pct=recent,
        slope=_finite(slope), slope_pct=_finite(slope_pct), momentum=_finite(momentum),
        acceleration=_finite(accel), volatility=_finite(vol),
        trend_strength=_finite(min(1.0, t_strength)),
        seasonal_strength=_finite(min(1.0, s_strength)),
        mk_tau=_finite(tau), mk_pvalue=_finite(p, 1.0), direction=direction,
        peak_date=pd.Timestamp(idx[i_peak]), peak_value=_finite(peak),
        drawdown_pct=_finite(drawdown),
    )


# =========================================================================== lifecycle


def _noise_reference(y: np.ndarray, z: np.ndarray) -> tuple[float, str]:
    """``(scale, basis)``: what the noise level of a series is compared with.

    For ratio-scale data (no negative values) the mean absolute deseasonalised
    level ``mean|z|`` (basis ``'mean'``): noise of 20 around a level of 100 is
    large, the classic coefficient of variation.  A level is meaningless for
    interval-scale data or data that cross zero (temperatures, profits): the
    same seasonal series would be 'volatile' at a mean of 0.3 and 'stable' at
    12.5.  The scale is therefore at least a shift-invariant spread of the
    observations, half their 5-95 % range (basis ``'spread'``), and only that
    when the series has negative values.  For positive data the level almost
    always exceeds the spread, so their verdicts do not change.
    """
    y = np.asarray(y, dtype=float)
    lvl = float(np.mean(np.abs(z))) if len(z) else 0.0
    spread = 0.0
    if len(y) >= 2:
        q05, q95 = np.quantile(y, [0.05, 0.95])
        spread = 0.5 * float(q95 - q05)
    if len(y) and float(np.min(y)) < 0.0:
        lvl = 0.0
    if spread > lvl:
        return spread, "spread"
    return lvl, "mean"


def _phase_features(series: pd.Series, dec: Decomposition) -> dict:
    """Shape and significance features used by :func:`lifecycle_phase`.

    Computed in units of ``max|y|`` (key ``unit`` holds that constant) so that
    every threshold is scale-free; level/slope features are in those units.
    """
    y = _clean_values(series)
    n = len(y)
    c = _unit(y)
    y = y / c
    T = dec.trend.to_numpy(dtype=float) / c
    R = dec.resid.to_numpy(dtype=float) / c
    z = T + R
    sigma = _winsorized_sd(R)
    zc = T + np.clip(R, -3.5 * sigma, 3.5 * sigma) if sigma > _TINY else z
    ref = float(np.max(np.abs(T)))
    if ref <= _TINY:
        ref = max(float(np.max(np.abs(y))), 1.0)
    # the residual under-states the noise: the trend smoother and the fitted
    # seasonal pattern (p - 1 free values) absorb part of it -> de-bias
    l0, v = _kernel_noise_factors(_trend_window(n, dec.period))
    absorbed = 2.0 * l0 - v + (dec.period - 1) / max(n, 1)
    noise_sd = sigma / math.sqrt(max(1.0 - absorbed, 0.25))
    # That de-biasing is approximate (interior kernel only, no trend/seasonal
    # interplay); on short series (narrow trend window, 2-3 seasonal cycles) the
    # residual can hold only ~1/3 of the noise, which would make noise look like
    # a significant trend.  A model-free check: robust SD of the first
    # differences of the deseasonalised series (/ sqrt 2), corrected for the
    # p - 1 values the seasonal pattern absorbed.
    if n >= 8:
        d_sd = _robust_sd(np.diff(z)) / math.sqrt(2.0)
        d_sd /= math.sqrt(max(1.0 - (dec.period - 1) / max(n, 1), 0.25))
        noise_sd = max(noise_sd, d_sd)

    # serial dependence of the noise (AR(1) phi; 1 = random walk) -> innovation
    # variance; phi = 0 reproduces the white-noise formulas exactly
    phi, d1 = _noise_model(z, dec.period, R) if n >= 12 else (0.0, 0.0)
    sig_e2 = max(noise_sd ** 2 * (1.0 - phi * phi), phi * d1 * (1.0 + phi) / 2.0)
    # spread of the noise process over the sample (random walks wander far)
    path_var = _ar_path_var(phi, n)
    marg_var = min(1.0 / (1.0 - phi * phi) if phi < 1.0 else math.inf, path_var)
    noise_eff = max(noise_sd, math.sqrt(max(sig_e2 * marg_var, 0.0)))

    h = int(min(n - 1, max(3, min(6, n // 3), round(0.15 * n))))
    xs = np.arange(h, dtype=float)
    icpt, s_rec = _ols(zc[-h:], xs)
    sxx = float(np.sum((xs - xs.mean()) ** 2))
    var_slope = _ols_slope_var(h, phi, sig_e2) if phi > 0.0 else (
        noise_sd ** 2 / sxx if sxx > _TINY else 0.0)
    se_rec = math.sqrt(var_slope) if (sxx > _TINY and noise_sd > _TINY and var_slope > 0) else math.inf
    t_rec = s_rec / se_rec if math.isfinite(se_rec) and se_rec > 0 else (
        math.copysign(99.0, s_rec) if abs(s_rec) > _TINY else 0.0)
    chg_rec = s_rec * (h - 1)
    start_level = icpt
    # the same window on the trend line (what the chart and the narrative show): the
    # reasons quote this change -- the OLS fit above only drives the significance rules
    # (its intercept sits far below the trend's start on a convex rise: +1,077% vs +293%)
    t_rec0 = float(T[-1 - h]) if n > h else float(T[0])

    i_max = int(np.argmax(T))
    t_max = float(T[i_max])
    t_last = float(T[-1])
    t_min = float(np.min(T))
    amp = t_max - t_min
    drop_from_peak = (t_max - t_last) / ref
    t_low = float(np.min(T[:i_max + 1]))       # lowest point before the peak
    rise_to_peak = (t_max - t_low) / ref
    i_tr = i_max + int(np.argmin(T[i_max:]))
    t_tr = float(T[i_tr])
    decline_before_trough = (t_max - t_tr) / ref
    rebound = (t_last - t_tr) / ref

    # trailing run of points that stay within 10% of the range of the last level
    band = 0.10 * amp
    if amp > _TINY:
        outside = np.flatnonzero(np.abs(T - t_last) > band)
        settled = n - 1 - int(outside[-1]) if len(outside) else n
    else:
        settled = n

    if n > h:
        s_roll = (T[h - 1:] - T[:n - h + 1]) / max(h - 1, 1)
        # 90th percentile rather than max: robust to noise-inflated extremes
        s_max_up = float(np.quantile(s_roll, 0.9)) if len(s_roll) >= 10 else float(np.max(s_roll))
        s_rec_t = float(s_roll[-1])
    else:
        s_max_up = s_rec_t = (T[-1] - T[0]) / max(n - 1, 1)
    stall_ratio = s_rec_t / s_max_up if s_max_up > _TINY else 0.0
    avg_slope = (T[-1] - T[0]) / max(n - 1, 1)

    base = float(np.median(T[: max(1, int(math.ceil(0.6 * n)))]))
    # a multiple of a base near 0 says nothing for data with negative values (a sine wave
    # around +1 is no "2.9x growth from a low base"): same floor as the percentages
    pct_floor = _pct_floor(y)
    multiple = t_last / base if (base > 0.01 * ref and t_last > 0 and base >= pct_floor) else 0.0

    # noise relative to the level ("coefficient of variation" of the noise);
    # for persistent noise its spread over the sample, not the one-step noise
    lvl, basis = _noise_reference(y, z)
    cv = noise_eff / lvl if lvl > _TINY else (0.0 if noise_eff <= _TINY else 1e6)

    # variance ratio: trend variance vs. what smoothed noise would give
    # (white: v * sigma^2; AR(1): v * sigma_e^2 / (1 - phi)^2, at most the
    # spread of the noise path itself -- a random walk is all "trend")
    var_t = float(np.var(T))
    if noise_sd > _TINY:
        smooth_var = v / (1.0 - phi) ** 2 if phi < 1.0 else math.inf
        f_ratio = var_t / (sig_e2 * min(smooth_var, path_var)) if phi > 0.0 else var_t / (v * noise_sd ** 2)
    else:
        f_ratio = 1e6 if var_t > _TINY * max(ref, 1.0) ** 2 else 0.0

    return dict(
        unit=c, f_ratio=float(min(f_ratio, 1e6)), noise_sd=noise_sd, phi=phi, sig_e2=sig_e2,
        n=n, h=h, sigma=sigma, ref=ref, s_rec=s_rec, t_rec=float(t_rec), chg_rec=chg_rec,
        rel_rec=chg_rec / ref, start_level=start_level, t_rec0=t_rec0,
        i_max=i_max, t_max=t_max, t_last=t_last, t_min=t_min, amp=amp,
        rel_amp=amp / ref, t_low=t_low,
        drop_from_peak=drop_from_peak, rise_to_peak=rise_to_peak,
        i_tr=i_tr, t_tr=t_tr, decline_before_trough=decline_before_trough, rebound=rebound,
        settled=settled, s_max_up=s_max_up, s_rec_t=s_rec_t, stall_ratio=stall_ratio,
        avg_slope=avg_slope, base=base, multiple=multiple, cv=cv, cv_basis=basis,
        g_tot=(t_last - float(T[0])) / ref, pct_floor=pct_floor,
    )


_MINUS = "\u2212"   # typographic minus used in all user-facing numbers


def _txt(x: float, spec: str) -> str:
    """``format(x, spec)`` with the typographic minus (U+2212) instead of ``-``."""
    return format(_finite(x), spec).replace("-", _MINUS)


def _pct_txt(pct: float, decimals: int = 1) -> str:
    """Signed percentage with thousands separators, e.g. ``+12.5%`` / ``\u221242.3%`` / ``+1,076.8%``."""
    return _txt(pct, f"+,.{decimals}f") + "%"


_SUPERSCRIPT = str.maketrans("0123456789", "\u2070\u00b9\u00b2\u00b3\u2074\u2075\u2076\u2077\u2078\u2079")


def _num_txt(x: float, signed: bool = False) -> str:
    """A number for Korean prose, formatted like ``narrative.format_number``.

    Thousands separators and a precision chosen from the magnitude
    (``2,392,153`` / ``45.2`` / ``1.20`` / ``0.023``); below 1e-4 or from 1e12 up
    ``1.05\u00d710\u207b\u2079`` notation (the bounds of ``format_number``: no 13-digit
    integers) -- never Python's ``2.392e+06``.  Typographic minus;
    ``signed=True`` prefixes positive values with ``+``.
    """
    v = _finite(x)
    a = abs(v)
    sign = _MINUS if v < 0 else ("+" if signed and v > 0 else "")
    if 0 < a < 1e-4 or a >= 1e12:
        mant, exp = f"{a:.2e}".split("e")
        mant = mant.rstrip("0").rstrip(".")
        e = int(exp)
        sup = ("\u207b" if e < 0 else "") + str(abs(e)).translate(_SUPERSCRIPT)
        return sign + mant + "\u00d710" + sup
    if a >= 1000:
        body = f"{a:,.0f}"
    elif a >= 10:
        body = f"{a:,.1f}"
    elif a >= 0.1:
        body = f"{a:.2f}"
    elif a == 0:
        return "0"
    else:   # two significant digits, trailing zeros removed
        d = min(8, -int(math.floor(math.log10(a))) + 1)
        body = f"{a:.{d}f}".rstrip("0").rstrip(".")
    if body.strip("0.,") == "":
        return "0"
    return sign + body


def _p_txt(p: float) -> str:
    """p-value as in the narrative: ``p < 0.001`` or ``p = 0.034``."""
    p = _finite(p, 1.0)
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def _post_shift(series: pd.Series, dec: Decomposition, changepoints: list[Changepoint],
                f: dict, freq: str) -> Optional[dict]:
    """The stretch after the latest level shift, if it is long and flat.

    A step (policy change, re-basing, a one-off jump) followed by a flat
    stretch is a *stable* series at a new level, not a growing one, even
    though the whole-period Mann-Kendall test is significant.  Returns
    ``None`` unless the latest structural break is a level shift, at least
    ``max(8, h, min(season, n // 5))`` points have passed since, and the
    stretch since has no significant trend (serial-dependence-corrected
    Mann-Kendall p >= 0.05 and |t| < 2 for its OLS slope).  Otherwise a dict
    with the p-value, the length and two Korean reasons.
    """
    if not changepoints:
        return None
    last = max(changepoints, key=lambda c: c.index)
    if last.kind != "level_shift":
        return None
    n = int(f["n"])
    k = int(last.index)
    length = n - k
    if length < max(8, int(f["h"]), min(_data.season_length(freq), n // 5)) or k < 4:
        return None
    c = f["unit"]
    z = (dec.trend.to_numpy(dtype=float) + dec.resid.to_numpy(dtype=float)) / c
    seg = z[k:]
    phi = float(f["phi"])
    _, p = _trend_test(seg, phi)
    xs = np.arange(length, dtype=float)
    _, slope = _ols(seg, xs)
    var_s = _ols_slope_var(length, phi, f["sig_e2"]) if phi > 0 else f["noise_sd"] ** 2 / float(
        np.sum((xs - xs.mean()) ** 2))
    t = slope / math.sqrt(var_s) if var_s > 0 else 0.0
    if p < 0.05 or abs(t) >= 2.0:
        return None
    unit = _UNIT_KO.get(freq, "기간")
    # "그 이후 N개월": the periods after the break (not counting the break period itself), as the
    # narrative's "가장 최근 구조 변화(…) 이후 지금까지 N개월" counts them
    after_n = length - 1
    before = float(np.median(z[max(0, k - length):k]))
    after = float(np.median(seg))
    pct = _pct_change(after, before, f["ref"], f["pct_floor"])
    up = after >= before
    jump = (f" (약 {_pct_txt(pct)})" if pct is not None
            else f" (약 {_num_txt((after - before) * c, signed=True)})")
    when = _when_ko(last.date, freq)
    return dict(
        p=float(p), length=int(length),
        shift_sentence=f"{when}에 수준이 한 단계 {'높아졌습니다' if up else '낮아졌습니다'}{jump}.",
        flat_sentence=(f"그 이후 {after_n}{unit} 동안은 뚜렷한 추세 없이 새 수준에서 안정되어 있습니다 "
                       f"(해당 구간 Mann-Kendall {_p_txt(p)})."),
    )


def lifecycle_phase(series: pd.Series, freq: str, decomposition: Decomposition,
                    metrics: TrendMetrics, changepoints: list[Changepoint]) -> LifecyclePhase:
    """Classify the current lifecycle phase with explainable rules.

    Uses the shape of the trend (position and size of the peak, drawdown,
    rebound from the trough, recent slope vs. the best past slope, growth
    multiple over the long-run base) plus significance (t-statistic of the
    recent slope against the residual noise, Mann-Kendall on the whole
    series).  "Recent" means the last ~15% of the observations.  Returns the
    label, a confidence in [0, 1] and 2-4 short Korean reasons with numbers.
    """
    n = int(metrics.n_obs)
    unit = _UNIT_KO.get(freq, "기간")
    idx = series.index

    if n < _MIN_SHORT_N:
        # too short for a lifecycle verdict: only say whether noise dominates
        y = _clean_values(series)
        yn = y / _unit(y)                 # scale-free noise ratio
        lvl = float(np.mean(np.abs(yn))) if len(yn) else 0.0
        icpt, slope = _ols(yn)
        res = yn - (icpt + slope * np.arange(len(yn)))
        ref, _ = _noise_reference(yn, yn)   # level, or the spread of data that cross zero
        noise_cv = float(np.sqrt(np.mean(res * res))) / ref if ref > _TINY else 0.0
        label = "volatile" if noise_cv >= _VOLATILE_CV else "stable"
        # "국면", not "추세": the Mann-Kendall test can still find a significant trend in 5 points
        reasons = [f"관측치가 {n}개뿐이라 국면을 판단하기에 데이터가 부족합니다."]
        pct = (_pct_change(float(yn[-1]), float(yn[0]), lvl, _pct_floor(yn))
               if len(yn) >= 2 else None)
        if pct is not None:
            # "...에서 6까지": no particle whose form depends on how the number is read (6으로 / 5로)
            reasons.append(f"관측 기간 동안 값이 {_num_txt(y[0])}에서 {_num_txt(y[-1])}까지 "
                           f"{_pct_txt(pct)} 변했습니다.")
        else:
            reasons.append(f"현재 값은 {_num_txt(metrics.last_value)}입니다.")
        reasons.append(f"직선 추세 대비 잡음 비율은 {noise_cv * 100:.1f}%입니다.")
        return LifecyclePhase(label=label, confidence=float(min(0.3, 0.1 + 0.025 * n)),
                              reasons=reasons)

    f = _phase_features(series, decomposition)
    h = f["h"]
    length_factor = min(1.0, 0.55 + n / 50.0)
    mk_up = metrics.mk_pvalue < 0.05 and metrics.mk_tau > 0
    mk_down = metrics.mk_pvalue < 0.05 and metrics.mk_tau < 0
    rising = f["t_rec"] >= _T_SIG and f["rel_rec"] >= 0.03
    falling = f["t_rec"] <= -_T_SIG and f["rel_rec"] <= -0.03
    # is there any trend structure beyond smoothed noise?
    structured = (f["f_ratio"] >= 4.0 and f["rel_amp"] >= 0.05) or (
        metrics.mk_pvalue < 0.01 and abs(f["g_tot"]) >= 0.10)

    peak_date = _fmt_date(idx[f["i_max"]], freq)        # in brackets: "고점(2016)"
    peak_when = _when_ko(idx[f["i_max"]], freq)          # before a particle: "2016년에"
    trough_when = _when_ko(idx[f["i_tr"]], freq)

    def recent_sentence() -> str:
        # measured on the trend line over the last h periods (T[-1-h] -> T[-1]), like
        # TrendMetrics.recent_change_pct over its window -- not from the OLS intercept
        pct = _pct_change(f["t_last"], f["t_rec0"], f["ref"], f["pct_floor"])
        if pct is not None:
            return f"최근 {h}{unit} 동안 추세가 {_pct_txt(pct)} 변했습니다."
        slope = (f["t_last"] - f["t_rec0"]) / max(h, 1) * f["unit"]
        return (f"최근 {h}{unit} 동안 추세 기울기는 {_num_txt(slope, signed=True)}"
                f"/{_PER_KO.get(freq, unit)}입니다.")

    def peak_sentence() -> str:
        pct = _pct_change(f["t_last"], f["t_max"], f["ref"], f["pct_floor"])
        if pct is not None:
            return f"현재 추세는 고점({peak_date})보다 {abs(pct):,.1f}% 낮은 수준입니다."
        if f["t_max"] > 0.0 > f["t_last"]:
            # crossed zero: "183.6% below the peak" is no percentage -- give the two levels
            return (f"추세는 고점({peak_date}, {_num_txt(f['t_max'] * f['unit'])}) 이후 0 아래로 "
                    f"내려가 현재 {_num_txt(f['t_last'] * f['unit'])}입니다.")
        return (f"추세 고점({peak_date}) 대비 현재 수준이 추세 규모(최대 절댓값)의 "
                f"{f['drop_from_peak'] * 100:.1f}%만큼 낮습니다.")

    at_top = f["i_max"] >= n - 1

    def rise_sentence() -> str:
        pct = _pct_change(f["t_max"], f["t_low"], f["ref"], f["pct_floor"])
        rise = (f"저점 대비 {_pct_txt(pct, 0)}" if pct is not None
                else f"저점 대비 추세 규모의 {f['rise_to_peak'] * 100:.0f}%만큼")
        if at_top:   # the maximum is the latest point: no peak has "formed" yet
            return f"추세는 {rise} 올라 현재 역대 최고 수준입니다."
        return f"추세는 {rise} 오른 뒤 {peak_when}에 고점을 형성했습니다."

    def mk_sentence() -> str:
        if metrics.mk_pvalue < 0.05:
            word = "상승" if metrics.mk_tau > 0 else "하락"
            return (f"Mann-Kendall 검정에서 전체 기간의 {word} 추세가 유의합니다 "
                    f"(τ = {_txt(metrics.mk_tau, '.2f')}, {_p_txt(metrics.mk_pvalue)}).")
        return (f"Mann-Kendall 검정에서 전체 기간의 뚜렷한 추세가 확인되지 않습니다 "
                f"({_p_txt(metrics.mk_pvalue)}).")

    def noise_sentence(tail: str) -> str:
        """'잡음이 커서(...)' with what the noise is compared with (see _noise_reference)."""
        ref_ko = "평균" if f["cv_basis"] == "mean" else "값의 변동 폭"
        return f"잡음이 커서(추정 잡음 수준이 {ref_ko}의 {f['cv'] * 100:,.0f}%) {tail}"

    label: str
    strength: float
    reasons: list[str]

    # random-walk-like noise (phi ~ 1): moves accumulate, so a large swing is
    # not evidence of a trend by itself (see _noise_model)
    walk = f["phi"] >= 0.9
    walk_sentence = ("변화가 누적되는 무작위 행보에 가까운 움직임이라, "
                     "추세처럼 보이는 등락도 우연일 수 있습니다.")

    # a level shift followed by a flat stretch: judge the period since the step
    post = _post_shift(series, decomposition, changepoints, f, freq)

    if post is not None:
        label = "stable"
        strength = 0.5 * _ramp(post["p"], 0.05, 0.5) + 0.5 * _ramp(post["length"] / n, 0.15, 0.4)
        reasons = [post["shift_sentence"], post["flat_sentence"]]
    elif not structured:
        if f["cv"] >= _VOLATILE_CV:
            label = "volatile"
            strength = 0.5 * _ramp(f["cv"], _VOLATILE_CV, 2 * _VOLATILE_CV) + 0.5 * _ramp(
                metrics.mk_pvalue, 0.01, 0.3)
            reasons = [noise_sentence("뚜렷한 방향성이 없습니다."), mk_sentence()]
        else:
            label = "stable"
            strength = 0.5 * _ramp(metrics.mk_pvalue, 0.01, 0.3) + 0.5 * _ramp(
                _VOLATILE_CV - f["cv"], 0.0, _VOLATILE_CV)
            reasons = [mk_sentence()]
            if f["rel_amp"] < 0.3:
                # "only x %" is evidence of stability only while x is small
                reasons.append(f"추세의 변동 폭(최고-최저)이 최고 수준 대비 "
                               f"{f['rel_amp'] * 100:.1f}%에 그칩니다.")
            else:   # a sizeable range that the noise could have produced (f_ratio < 4)
                reasons.append("추세선의 등락이 잡음으로 생길 수 있는 범위를 크게 넘지 않습니다.")
            if metrics.seasonal_strength >= 0.5:
                # no particle right after the number ("0.96로" would need to be "0.96으로")
                reasons.append(f"계절성 강도({metrics.seasonal_strength:.2f})가 높아 "
                               f"주기적 변동이 뚜렷합니다.")
    elif (f["decline_before_trough"] >= 0.20 and f["i_tr"] <= n - 1 - max(2, h // 3)
          and f["rebound"] >= 0.10 and f["rebound"] >= 0.25 * f["decline_before_trough"]
          and rising and f["drop_from_peak"] >= 0.05):
        label = "reviving"
        strength = 0.5 * _ramp(f["rebound"] / max(f["decline_before_trough"], _TINY), 0.25, 0.6) \
            + 0.5 * _ramp(f["t_rec"], _T_SIG, 3 * _T_SIG)
        reb_pct = _pct_change(f["t_last"], f["t_tr"], f["ref"], f["pct_floor"])
        reb_txt = (_pct_txt(reb_pct) if reb_pct is not None
                   else f"추세 규모의 {f['rebound'] * 100:.1f}%만큼")
        reasons = [
            f"고점({peak_date}) 이후 하락해 {trough_when}에 저점을 찍은 뒤 {reb_txt} 반등했습니다.",
            recent_sentence(),
            peak_sentence(),
        ]
    elif f["drop_from_peak"] >= _FAR_BELOW_PEAK and not rising:
        if f["settled"] >= 0.5 * n and not falling:
            # the fall happened long ago; it has been flat for most of the history
            label = "stable"
            strength = 0.5 * _ramp(f["settled"] / n, 0.5, 0.8)
            reasons = [
                peak_sentence(),
                f"다만 최근 {f['settled']}{unit} 동안 비슷한 수준에서 안정되어 있습니다.",
            ]
        else:
            label = "declining"
            strength = 0.6 * _ramp(f["drop_from_peak"], _FAR_BELOW_PEAK, 0.7) + 0.4 * _ramp(
                -f["t_rec"], -1.0, _T_SIG)
            reasons = [peak_sentence(), recent_sentence()]
            if f["settled"] >= 0.25 * n and not falling and abs(f["rel_rec"]) < 0.05:
                reasons.append(f"최근 {f['settled']}{unit} 동안은 낮은 수준에서 정체되어 있습니다.")
            elif mk_down:
                reasons.append(mk_sentence())
    elif rising:
        if f["stall_ratio"] < _STALL_RATIO and f["rel_rec"] < 0.15 and f["rise_to_peak"] >= 0.25:
            label = "peaking"
            strength = 0.5 * _ramp(_STALL_RATIO - f["stall_ratio"], 0.0, _STALL_RATIO) + 0.3
            reasons = [
                f"최근 기울기가 과거 상승기 기울기(상위 10%)의 {f['stall_ratio'] * 100:.0f}% 수준으로 둔화되었습니다.",
                recent_sentence(),
                rise_sentence(),
            ]
        elif (f["multiple"] >= _EMERGING_MULTIPLE and f["s_rec_t"] >= 1.5 * max(f["avg_slope"], _TINY)):
            label = "emerging"
            strength = 0.5 * _ramp(f["multiple"], _EMERGING_MULTIPLE, 2 * _EMERGING_MULTIPLE) \
                + 0.5 * _ramp(f["s_rec_t"] / max(f["avg_slope"], _TINY), 1.5, 3.0)
            pace = f["s_rec_t"] / f["avg_slope"] if f["avg_slope"] > _TINY else math.inf
            reasons = [
                f"현재 추세 수준이 과거 기반 수준의 {f['multiple']:.1f}배로, 낮은 기반에서 빠르게 성장 중입니다.",
                recent_sentence(),
                # a trend that ends near where it started has an average pace of ~0: no "1.1e11배"
                (f"최근 상승 속도가 전체 기간 평균 속도(추세선 처음→끝)의 {pace:.1f}배로 가속되고 있습니다."
                 if pace <= _PACE_CAP else
                 f"최근 상승 속도가 전체 기간 평균 속도(추세선 처음→끝)의 {_PACE_CAP:.0f}배를 넘습니다."),
            ]
        else:
            label = "growing"
            strength = 0.5 * _ramp(f["t_rec"], _T_SIG, 3 * _T_SIG) + 0.5 * _ramp(
                f["stall_ratio"], _STALL_RATIO, 0.8)
            reasons = [recent_sentence()]
            if mk_up:
                reasons.append(mk_sentence())
            share = 100 * (1 - f["drop_from_peak"])
            # "추세 고점의 100% 수준" reads oddly: at (or within rounding of) the high, say so
            reasons.append(f"현재 추세는 고점({peak_date}) 수준입니다." if share >= 99.5
                           else f"현재 수준은 추세 고점의 {share:.0f}% 수준입니다.")
    elif (f["i_max"] >= 0.6 * (n - 1) and f["rise_to_peak"] >= 0.25
          and f["rise_to_peak"] * f["ref"] >= 3.0 * f["sigma"]
          and f["stall_ratio"] >= _STALL_RATIO and f["drop_from_peak"] < 0.05
          and f["chg_rec"] > 0 and (mk_up or f["t_rec"] > 1.0)):
        # still at the high and not slowing down, recent move just not significant
        label = "growing"
        strength = 0.3 * _ramp(f["stall_ratio"], _STALL_RATIO, 0.8)
        reasons = [recent_sentence(), mk_sentence(),
                   f"최근 기울기가 과거 상승기 기울기(상위 10%)의 {f['stall_ratio'] * 100:.0f}% 수준을 유지하고 있습니다."]
    elif (f["i_max"] >= 0.6 * (n - 1) and f["rise_to_peak"] >= 0.25
          and f["rise_to_peak"] * f["ref"] >= 3.0 * f["sigma"]):
        label = "peaking"
        strength = 0.4 * _ramp(f["rise_to_peak"], 0.25, 0.6) + 0.4 * _ramp(
            _STALL_RATIO - f["stall_ratio"], 0.0, _STALL_RATIO) + 0.2
        reasons = [
            rise_sentence(),
            recent_sentence(),
            f"최근 기울기가 과거 상승기 기울기(상위 10%)의 {max(f['stall_ratio'], 0.0) * 100:.0f}% 수준으로 둔화되었습니다.",
        ]
    elif (f["drop_from_peak"] >= _MOVE or f["g_tot"] <= -_MOVE) and (
            falling or (mk_down and f["chg_rec"] <= 0)):
        label = "declining"
        strength = 0.5 * _ramp(f["drop_from_peak"], _MOVE, _FAR_BELOW_PEAK) + 0.5 * _ramp(
            -f["t_rec"], 0.0, 2 * _T_SIG)
        reasons = [peak_sentence(), recent_sentence(), mk_sentence()]
    elif mk_up and f["g_tot"] >= _MOVE and not falling and f["drop_from_peak"] < _MOVE:
        # slow but significant long-run growth, no significant recent fall
        label = "growing"
        strength = 0.3 * _ramp(-math.log10(max(metrics.mk_pvalue, 1e-300)), 1.3, 4.0) + 0.1
        reasons = [mk_sentence(), recent_sentence()]
    else:
        label = "volatile" if f["cv"] >= _VOLATILE_CV else "stable"
        strength = 0.2
        reasons = [mk_sentence(), recent_sentence()]
        if label == "volatile":
            reasons.append(noise_sentence("방향성이 불분명합니다."))

    if walk and label in ("volatile", "stable") and len(reasons) < 4:
        reasons.append(walk_sentence)
    # a recent structural break supports the explanation
    recent_cps = [c for c in changepoints if c.index >= n - max(h, 2 * _data.season_length(freq))]
    if recent_cps and len(reasons) < 4 and post is None:
        c = recent_cps[-1]
        reasons.append(f"{_when_ko(c.date, freq)}에 구조 변화({changepoint_label(c)})가 감지되었습니다.")
    if walk and label not in ("volatile", "stable") and len(reasons) < 4:
        reasons.append(walk_sentence)
    if len(reasons) < 2 and mk_sentence() not in reasons:
        reasons.append(mk_sentence())
    reasons = reasons[:4]

    confidence = (0.45 + 0.5 * float(min(1.0, max(0.0, strength)))) * length_factor
    # random-walk-like series: any phase verdict is much less reliable
    confidence *= 1.0 - 0.4 * _ramp(f["phi"], 0.8, 1.0)
    return LifecyclePhase(label=label, confidence=float(min(1.0, max(0.0, confidence))),
                          reasons=reasons)


# =========================================================================== orchestration


def _decompose_with_shifts(s: pd.Series, period: int, dec: Decomposition,
                           shifts: list[int]) -> tuple[Decomposition, list[int]]:
    """Re-decompose with the trend allowed to jump at the level shifts that need it.

    A break is kept only if the piecewise trend fits the neighbourhood of the
    step clearly better (residual sum of squares within one trend window on
    either side < 70% of the smooth decomposition's): a genuine step leaves a
    large S-shaped residual around it in the smooth fit, whereas a "level
    shift" found on a smooth curve (piecewise-linear segmentation of an
    exponential decay) does not need, and would be hurt by, a jump.
    Returns the decomposition and the breaks it uses.
    """
    n = len(s)
    win = _trend_window(n, dec.period)
    r0 = dec.resid.to_numpy(dtype=float)
    keep = list(shifts)
    for _ in range(2):
        if not keep:
            return dec, []
        cand = decompose(s, period, breaks=keep)
        r1 = cand.resid.to_numpy(dtype=float)
        good = []
        for k in keep:
            a, b = max(0, k - win), min(n, k + win)
            if float(np.sum(r1[a:b] ** 2)) < 0.7 * float(np.sum(r0[a:b] ** 2)):
                good.append(k)
        if good == keep:
            return cand, keep
        keep = good
    return dec, []


def analyze(series: pd.Series, freq: Optional[str] = None,
            name: Optional[str] = None) -> AnalysisResult:
    """Full past -> present analysis of a (possibly raw) series.

    The input is cleaned/regularised with :func:`trendai.data.prepare`, then
    decomposed; changepoints, anomalies, metrics and the lifecycle phase are
    computed.  If level shifts are found, the series is decomposed again with
    the trend allowed to jump there (``decompose(..., breaks=...)``, see
    :func:`_decompose_with_shifts`), and that decomposition is the one
    returned and used for anomalies, metrics and the phase.  Works for as few as 3 observations (degrading gracefully).
    """
    s, freq = _data.prepare(series, freq=freq, name=name)
    n = len(s)
    sl = _data.season_length(freq)
    dec = decompose(s, sl)
    z = (dec.trend + dec.resid).to_numpy(dtype=float)
    cps = _changepoints_from(z, pd.DatetimeIndex(s.index), sl, 5, None) if n >= 8 else []
    shifts = [c.index for c in cps if c.kind == "level_shift"]
    kept: list[int] = []
    if shifts:
        # let the trend jump at level shifts (no smoothing across the step, which
        # would make the points next to it look like spikes/dips)
        dec, kept = _decompose_with_shifts(s, sl, dec, shifts)
        # a "level shift" that the data do not need as a jump (a kink in a smooth
        # curve) is reported by its slope change instead
        for c in cps:
            if c.kind == "level_shift" and c.index not in kept:
                s1, s2 = c.slope_before, c.slope_after
                if s2 > 0 >= s1:
                    c.kind = "reversal_up"
                elif s2 < 0 <= s1:
                    c.kind = "reversal_down"
                else:
                    c.kind = "acceleration" if abs(s2) > abs(s1) else "deceleration"
    anomalies = detect_anomalies(s, dec, breaks=kept) if n >= 8 else []
    metrics = compute_metrics(s, freq, dec, anomalies=anomalies)
    phase = lifecycle_phase(s, freq, dec, metrics, cps)
    nm = name if name is not None else (str(s.name) if s.name is not None else "series")
    return AnalysisResult(name=nm, series=s, freq=freq, season_length=int(sl),
                          decomposition=dec, metrics=metrics, changepoints=cps,
                          anomalies=anomalies, phase=phase)
