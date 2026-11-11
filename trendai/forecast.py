"""Future: automatic forecasting with model selection, ensembling and calibrated intervals.

Pipeline of :func:`auto_forecast`::

    prepared series ──► history cap (the SAME recent history is used for the
                        backtest and for the final fit)
                    ──► isolated spikes replaced by their expected value
                    ──► candidates: level models only for intermittent demand, no
                        trend extrapolators for a driftless random walk
                        (``random_walk``, see :func:`is_random_walk`), plus
                        ``drift@log`` for steadily compounding positive series
                    ──► rolling-origin backtest of every candidate model and of the
                        naive benchmarks (same windows for all, MASE on the training
                        scale, recent windows weighted more, every forecast passed
                        through the plausibility guard)
                    ──► rank by MASE ──► inverse-MASE ensemble of the top-k
                        (kept only if it is not clearly worse than the best model)
                    ──► refit the chosen model(s) on the history, plausibility guard
                    ──► prediction intervals from the chosen method's own backtest
                        errors per horizon step (σ_k ≈ s₁·k^b), rescaled to the
                        current level when the errors grow with the level, shrunk
                        towards a prior (random-walk scale at lead 1, growing like
                        the backtest errors) and widened with Student-t
                        quantiles when the backtest is small, and wide enough for
                        the 95 % band to contain every ensemble member
                    ──► skill = backtest MAE / MAE of the best naive benchmark
                        (RMSE for intermittent demand, the metric that ranked it)

:func:`compute_outlook` then turns a forecast into an interpretation relative
to the present (direction, expected change, probability of an increase,
scenarios, projected lifecycle phase, confidence).

Everything is numpy/pandas only and fully deterministic.
"""

from __future__ import annotations

import functools
import math
import warnings
from dataclasses import dataclass
from typing import NamedTuple, Optional, Sequence

import numpy as np
import pandas as pd

from . import models as _models
from .analysis import _is_intermittent as _analysis_is_intermittent
from .analysis import _is_multiplicative, _pct_reference, pct_change_undefined
from .data import default_horizon, future_index, prepare
from .data import season_length as _freq_season_length
from .types import AnalysisResult, ForecastResult, ModelScore, Outlook

__all__ = [
    "mase_scale",
    "backtest",
    "auto_forecast",
    "is_random_walk",
    "compute_outlook",
    "outlook_levels",
    "OutlookLevels",
    "interval_level",
    "fit_history",
    "norm_cdf",
    "norm_ppf",
    "t_cdf",
    "t_ppf",
]

# --------------------------------------------------------------------------- tuning constants

#: minimum training length of a backtest window when seasonality is not modelled
_MIN_TRAIN_NONSEASONAL = 6
#: at least this many backtest windows are required (otherwise: heuristic fallback)
_MIN_WINDOWS = 2
#: history cap (most recent points) for the backtest AND the final fit; at least 8
#: seasons are always kept.  Using one cap for both matters: several models depend on
#: the length of their input (linear_trend fits the last n//2 points, drift goes from
#: the first to the last value, theta uses the whole-history slope...), so a final fit
#: on a longer history than the backtest scored produced forecasts the backtest never
#: saw (e.g. breakout, daily n=730: test MAE x20, 0 % interval coverage).
_BT_MAX_HISTORY = 400
#: keep the best single model instead of the ensemble if its MASE < margin * ensemble MASE
_ENSEMBLE_MARGIN = 0.9
#: ... but only if the backtest has at least this many errors (W * h) per model
_MIN_ERRORS_FOR_SINGLE = 6
#: weight of the most recent backtest window relative to the oldest one (geometric in
#: between) when averaging window scores: how the models did lately counts more than a
#: regime that may have ended (testbed of 483 series: equal weights 1.6 % worse in MAE,
#: ratio 6 0.8 % worse)
_RECENCY_RATIO = 3.0
#: bounds of the horizon exponent b in sigma_k = s1 * k**b.  The lower bound keeps
#: intervals (mildly) widening with the horizon even when the backtest errors do not
#: grow; 0.1 rather than 0.3 because simulations showed 0.3 under-covers at h = 1 and
#: over-covers at long horizons for series with stationary noise.  (0 gave flatter
#: per-lead coverage but a pooled, wider h = 1 band: step-1 coverage 0.86 instead of 0.78
#: on daily noise, and fewer direction calls through the h = 1 noise threshold.)
_B_MIN, _B_MAX = 0.1, 1.0
#: horizon exponent used when it cannot be estimated (random-walk like growth)
_B_DEFAULT = 0.5
#: minimum horizon exponent beyond the backtest horizon: with a short backtest
#: (e.g. 2 steps for n = 10) a fitted b says nothing about how errors grow at step 12
_B_EXTRAPOLATE = 0.5
#: multiplier of the robust in-sample noise scale used as the floor of s1
_FLOOR_FACTOR = 0.5
#: small-sample / model-selection inflation of the backtest error scale.  (Dropping it for
#: backtests of >= 30 distinct targets brought well-specified long series from 84-86 / 96-97 %
#: to 82-84 / 95-97 % coverage of the 80 / 95 % bands, but a mixed testbed of 483 series,
#: whose 95 % coverage is already short at 0.933 through misspecified mean forecasts, fell to
#: 0.924; kept.)
_SIGMA_INFLATION_BASE = 1.05
#: the random-walk prior of the interval scale takes the growth with the lead from the
#: backtest errors whenever the backtest covers at least 3 leads, unless the series is a
#: driftless random walk (see auto_forecast)
_PRIOR_FOLLOWS_BACKTEST = True
#: prior degrees of freedom of the random-walk error scale the backtest scale is shrunk
#: towards (see _shrink_sigma); it dominates tiny backtests and vanishes for long ones
_PRIOR_DOF = 2.0
#: at most this many Student-t degrees of freedom per distinct backtest target (errors of
#: overlapping multi-step windows are correlated, so they carry less than one each; fewer
#: still when the errors are autocorrelated, see _error_dof_factor)
_DOF_PER_TARGET = 0.5
#: selection optimism: the chosen method's backtest error scale is inflated by
#: sqrt(1 + c * log(number of candidates) / distinct targets)
_SELECTION_PENALTY = 1.0
#: widening of the heuristic intervals used when the series is too short to backtest
_FALLBACK_WIDEN = 1.5
#: rescale backtest errors to the current level for positive series (see _level_adjust)
_LEVEL_ADJUST = True
#: floor of a backtest window's MASE scale, relative to the scale of the history before
#: the latest window origin (keeps MASE unit-free when a training window is constant)
_SCALE_FLOOR_FRAC = 0.1
#: plausibility guard: a forecast may move k steps ahead by at most this multiple of the
#: largest k-step move in the history (or the recent pace times k, whichever is larger)
_GUARD_SLACK = 1.5
#: robust |z| of a residual from which an isolated point counts as a spike
_SPIKE_Z = 5.0
#: at most this share of points may be spikes (otherwise the noise is heavy-tailed and
#: nothing is replaced); 4 news spikes in 3 years of monthly data are 11 %
_SPIKE_MAX_SHARE = 0.15
#: the only candidates for intermittent demand (those that are registered): they estimate a
#: level (a demand rate) instead of following the last observations; 'croston' / 'sba' are
#: picked up automatically if the models module registers them
_INTERMITTENT_MODELS = ("ses", "theta", "holt_damped", "croston", "sba", "tsb")
#: backtest metric that ranks and weights the candidates for intermittent demand.  MAE (and so
#: MASE) is minimised by the median, which for mostly-zero data is 0 or close to it, so it
#: prefers forecasts that under-state the demand rate; squared error is minimised by the mean
#: rate.  Simulations of intermittent demand (W/M/D, 15-60 % non-zero, 60 seeds x 6 settings):
#: mean relative error of the forecast rate 0.280 with MASE ranking, 0.266 with RMSE
_INTERMITTENT_METRIC = "rmse"
#: models that extrapolate a fitted (global or local) trend; for a driftless random walk the
#: fitted trend is the noise path of the sample, which they project forward as a direction
#: (40 daily geometric random walks: p_up 0.95 / +16 % from linear_trend@log + linear_trend,
#: Brier 0.270 against 0.250 for a constant 0.5), so they are not offered there
_TREND_EXTRAPOLATORS = ("linear_trend", "linear_trend@log", "logistic_growth", "holt",
                        "holt_winters", "holt_winters@log")
#: noise-model persistence (AR(1) phi of the noise around the trend, see analysis) from which a
#: series without a significant trend counts as a driftless random walk
_RW_PHI = 0.9
#: candidate added for strictly positive series whose growth is multiplicative (see
#: _multiplicative_growth): level-space drift / holt / ar project *linear* growth, so on
#: compounding series (GDP-like annual data) every forecast falls behind (median bias -2.2 % at
#: h = 5, 73 % of the forecasts below the outcome)
_LOG_DRIFT = "drift@log"
#: ... when the average log growth of the two halves of the history differs by less than this
#: (log of the ratio): compounding at a steady rate, not an S-curve
_STEADY_GROWTH_LOG = math.log(1.5)


# --------------------------------------------------------------------------- normal / t

# Acklam's rational approximation of the inverse normal CDF (rel. error ~1e-9),
# followed by one Halley refinement step using math.erfc.
_PPF_A = (-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
          1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00)
_PPF_B = (-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
          6.680131188771972e01, -1.328068155288572e01)
_PPF_C = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
          -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00)
_PPF_D = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
          3.754408661907416e00)
_SQRT2 = math.sqrt(2.0)


def norm_cdf(x: float) -> float:
    """Standard normal CDF ``Φ(x)`` (via ``math.erfc``, accurate in both tails)."""
    return 0.5 * math.erfc(-float(x) / _SQRT2)


def norm_ppf(p: float) -> float:
    """Inverse standard normal CDF ``Φ⁻¹(p)`` for ``0 < p < 1``.

    Acklam's algorithm refined by one Halley step; accurate to ~1e-12 in the
    central region used for prediction intervals.
    """
    p = float(p)
    if not 0.0 < p < 1.0:
        raise ValueError(f"p must be in (0, 1), got {p}")
    a, b, c, d = _PPF_A, _PPF_B, _PPF_C, _PPF_D
    p_low = 0.02425
    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        x = ((((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
             / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0))
    elif p <= 1.0 - p_low:
        q = p - 0.5
        r = q * q
        x = ((((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
             / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0))
    else:
        q = math.sqrt(-2.0 * math.log1p(-p))
        x = -((((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
              / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0))
    # Halley refinement
    e = norm_cdf(x) - p
    u = e * math.sqrt(2.0 * math.pi) * math.exp(0.5 * x * x)
    x = x - u / (1.0 + 0.5 * x * u)
    return float(x)


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction of the incomplete beta function (modified Lentz)."""
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for i in range(1, 400):
        i2 = 2 * i
        aa = i * (b - i) * x / ((qam + i2) * (a + i2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + i) * (qab + i) * x / ((a + i2) * (qap + i2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    return h


def _betainc(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta function ``I_x(a, b)`` for ``a, b > 0``."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbt = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
           + a * math.log(x) + b * math.log1p(-x))
    bt = math.exp(lbt)
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def t_cdf(x: float, df: float) -> float:
    """CDF of Student's t distribution with ``df > 0`` (real) degrees of freedom."""
    x, df = float(x), float(df)
    if not df > 0:
        raise ValueError(f"df must be > 0, got {df}")
    if math.isinf(df):
        return norm_cdf(x)
    tail = 0.5 * _betainc(0.5 * df, 0.5, df / (df + x * x))
    return 1.0 - tail if x > 0 else tail


@functools.lru_cache(maxsize=512)
def _t_ppf_cached(p: float, df: float) -> float:
    if p == 0.5:
        return 0.0
    if p < 0.5:
        return -_t_ppf_cached(1.0 - p, df)
    if df > 1e7:
        return norm_ppf(p)
    lo, hi = 0.0, max(1.0, norm_ppf(p))
    while t_cdf(hi, df) < p:
        lo, hi = hi, 2.0 * hi
        if hi > 1e12:  # pragma: no cover - only for absurd (p, df)
            return hi
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if t_cdf(mid, df) < p:
            lo = mid
        else:
            hi = mid
        if hi - lo <= 1e-13 * max(1.0, hi):
            break
    return 0.5 * (lo + hi)


def t_ppf(p: float, df: float) -> float:
    """Inverse CDF of Student's t distribution (``0 < p < 1``, ``df > 0``; ``inf`` = normal)."""
    p, df = float(p), float(df)
    if not 0.0 < p < 1.0:
        raise ValueError(f"p must be in (0, 1), got {p}")
    if not df > 0:
        raise ValueError(f"df must be > 0, got {df}")
    if math.isinf(df):
        return norm_ppf(p)
    return float(_t_ppf_cached(p, df))


def _level_z(level: float) -> float:
    """Two-sided normal quantile for a central ``level`` % interval (80 -> 1.2816)."""
    return norm_ppf(0.5 + float(level) / 200.0)


def _level_q(level: float, dof: float) -> float:
    """Two-sided Student-t quantile for a central ``level`` % interval (normal for inf dof)."""
    return t_ppf(0.5 + float(level) / 200.0, dof)


# --------------------------------------------------------------------------- scale helpers


def _robust_scale(x: np.ndarray) -> float:
    """Robust standard deviation (1.4826·MAD; mean absolute deviation if MAD is 0)."""
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return 0.0
    med = float(np.median(x))
    dev = np.abs(x - med)
    mad = float(np.median(dev))
    if mad > 0:
        return 1.4826 * mad
    return 1.2533 * float(np.mean(dev))


def _tiny_scale(y: np.ndarray) -> float:
    """A strictly positive, negligible scale for ``y`` (guards divisions)."""
    ref = float(np.max(np.abs(y))) if len(y) else 0.0
    return 1e-6 * ref if ref > 0 else 1e-9


def _rms(x: np.ndarray, axis: Optional[int] = None):
    """Root mean square that cannot overflow (``|x|`` is rescaled by its maximum first).

    A plain ``sqrt(mean(x**2))`` overflows to ``inf`` once ``|x| > ~1e154``.
    Returns a float for ``axis=None``, else an array reduced along ``axis``.
    """
    a = np.abs(np.asarray(x, dtype=float))
    if a.size == 0:
        return 0.0 if axis is None else np.zeros(np.delete(a.shape, axis))
    s = np.max(a, axis=axis, keepdims=True)
    safe = np.where(s > 0, s, 1.0)
    r = np.sqrt(np.mean((a / safe) ** 2, axis=axis, keepdims=True)) * np.where(s > 0, s, 0.0)
    return float(r.ravel()[0]) if axis is None else np.squeeze(r, axis=axis)


def _naive_mae(y: np.ndarray, season_length: int) -> float:
    """Seasonal-naive in-sample MAE (naive as fallback); ``0.0`` when it is undefined or 0."""
    y = np.asarray(y, dtype=float).ravel()
    y = y[np.isfinite(y)]
    m = max(1, int(season_length))
    n = len(y)
    if m > 1 and n > m:
        s = float(np.mean(np.abs(y[m:] - y[:-m])))
        if s > 0 and math.isfinite(s):
            return s
    if n > 1:
        s = float(np.mean(np.abs(np.diff(y))))
        if s > 0 and math.isfinite(s):
            return s
    return 0.0


def mase_scale(y: np.ndarray, season_length: int) -> float:
    """In-sample MAE of the seasonal-naive forecast (naive as fallback); always ``> 0``.

    This is the MASE denominator: ``mean |y_t - y_{t-m}|`` with
    ``m = season_length`` when the series is longer than one season and that
    MAE is positive, otherwise ``mean |y_t - y_{t-1}|``.  For a constant (or
    single-point) series a tiny positive scale is returned so callers never
    divide by zero.
    """
    s = _naive_mae(y, season_length)
    if s > 0:
        return s
    y = np.asarray(y, dtype=float).ravel()
    y = y[np.isfinite(y)]
    if len(y) == 0:
        return 1.0
    return max(_tiny_scale(y), 1e-3 * float(np.mean(np.abs(y)))) if np.any(y) else 1.0


def _window_scales(y: np.ndarray, origins: Sequence[int], m: int) -> np.ndarray:
    """MASE denominators of the backtest windows: ``mase_scale`` of each training part.

    A training part that is constant (e.g. all zeros before a new trend takes
    off) has no usable scale: :func:`mase_scale` then returns an arbitrary tiny
    value, or ``1.0`` in absolute units for all zeros.  That window would then
    dominate the average MASE, and model ranking, ``backtest_mase`` and
    ``confidence`` would depend on the units of the data.  Each window's scale
    is therefore floored at ``_SCALE_FLOOR_FRAC`` times the scale of the
    history *before the latest window origin* (``y[:max(origins)]``): this is
    unit-free, leaves ordinary series unchanged and never looks at the target
    period of the latest window (scores cannot depend on how the last ``h``
    observations turned out other than through the errors themselves).
    """
    o_last = int(max(origins))
    floor = _SCALE_FLOOR_FRAC * mase_scale(y[:o_last], m)
    return np.array([max(_naive_mae(y[:o], m), floor) for o in origins], dtype=float)


def _noise_floor(y: np.ndarray, m: int) -> float:
    """Lower bound for the one-step forecast error scale (keeps intervals non-degenerate).

    Robust noise level from first differences and (if possible) seasonal
    differences -- for white noise of std σ both have std √2·σ, while trends
    (constant shift) and seasonality (removed by the seasonal difference) do
    not inflate the MAD-based estimate much.  A fraction of it is used.
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    tiny = _tiny_scale(y)
    if n < 3:
        return tiny
    scales = [_robust_scale(np.diff(y))]
    if m > 1 and n >= m + 3:
        scales.append(_robust_scale(y[m:] - y[:-m]))
    pos = [s for s in scales if s > 0]
    if not pos:
        return tiny
    return max(_FLOOR_FACTOR * min(pos) / _SQRT2, tiny)


def _prior_sigma(y: np.ndarray, m: int, horizon: int) -> np.ndarray:
    """Per-step random-walk error scale used as the prior of the interval width.

    ``s·√k`` with ``s`` the robust scale of the first differences (for a
    random walk exactly its step std; for trend + white noise about √2 times
    the noise std), or -- when the seasonal differences are less variable,
    i.e. the series is seasonal -- ``s_m·√⌈k/m⌉`` with the robust scale of the
    seasonal differences (a seasonal random walk).  This is what an analyst
    would assume without a backtest: deliberately conservative, since it only
    matters when the backtest is too small to be trusted.
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    k = np.arange(1, horizon + 1, dtype=float)
    tiny = _tiny_scale(y)
    if n < 3:
        return np.full(horizon, tiny)
    s1 = _robust_scale(np.diff(y))
    out = s1 * np.sqrt(k)
    if m > 1 and n >= m + 3:
        sm = _robust_scale(y[m:] - y[:-m])
        if 0 < sm < s1:
            out = sm * np.sqrt(np.ceil(k / m))
    return np.maximum(out, tiny)


# --------------------------------------------------------------------------- data conditioning


def _running_median(y: np.ndarray, w: int) -> np.ndarray:
    """Centred running median of odd width ``w`` (the window shrinks at the edges)."""
    n = len(y)
    half = w // 2
    if n == 0:
        return y.copy()
    pad = np.r_[np.full(half, np.nan), y, np.full(half, np.nan)]
    win = np.lib.stride_tricks.sliding_window_view(pad, 2 * half + 1)
    return np.nanmedian(win, axis=1)


def _clean_spikes(y: np.ndarray, m: int) -> tuple[np.ndarray, int]:
    """Replace isolated one- or two-point spikes by their expected value.

    One-off events (a news spike in page views, a data glitch) are not part
    of the signal, but seasonal models would otherwise project them as a
    recurring seasonal peak and every model's level would jump.  Expected
    value = centred running median (width ``m + 1`` or at least 5) plus, for
    ``m > 1``, the per-phase median of the remainder -- which needs at least
    three seasons (a median of two values is their mean, so a spike or a
    late level shift would leak into the seasonal pattern); with ``m > 1``
    and fewer than three seasons nothing is replaced, because a genuine
    seasonal peak cannot be told from a spike there.  Strictly positive
    series are examined in log space (spikes and seasonality are
    multiplicative there).  A point is a spike when its robust residual
    ``|z| >= 5`` (MAD scale) and it lies in a burst of at most two
    consecutive elevated points (``|z| >= 2.5``) that ends before the last
    observation (a jump at the very end may be the start of a new regime;
    a longer elevated run is a level shift or a ramp, not a spike).  Only
    the ``|z| >= 5`` points are replaced.  Nothing is replaced when more
    than 15 % of the points would be (heavy-tailed noise) or the series has
    fewer than 12 points; intermittent demand never gets here.  Returns
    ``(cleaned copy, number of replaced points)``.
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n < 12 or (m > 1 and n < 3 * m):
        return y, 0
    log = bool(np.all(y > 0))
    x = np.log(y) if log else y
    w = max(5, m + 1 if m % 2 == 0 else m + 2) if m > 1 else 5
    w = min(w, n if n % 2 else n - 1)
    trend = _running_median(x, w)
    rem = x - trend
    expected = trend.copy()
    if m > 1:
        ph = np.arange(n) % m
        seas = np.array([np.median(rem[ph == p]) for p in range(m)])
        seas -= seas.mean()
        expected = expected + seas[ph]
    r = x - expected
    scale = _robust_scale(r)
    if not scale > (1e-12 if log else _tiny_scale(y) * 1e-3):
        return y, 0
    z = np.abs(r - np.median(r)) / scale
    flag = z >= _SPIKE_Z
    if not flag[:-1].any():
        return y, 0
    if flag.sum() > max(2, int(_SPIKE_MAX_SHARE * n)):
        return y, 0
    elevated = z >= 0.5 * _SPIKE_Z
    out = y.copy()
    replaced = 0
    i = 0
    while i < n:
        if not elevated[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and elevated[j + 1]:
            j += 1
        # a short burst (<= 2 elevated points) that is over before the last observation
        if j - i + 1 <= 2 and j < n - 1:
            idx = np.arange(i, j + 1)[flag[i:j + 1]]
            ev = expected[idx]
            out[idx] = np.exp(ev) if log else ev
            replaced += len(idx)
        i = j + 1
    return out, replaced


def fit_history(y: np.ndarray, season_length: int) -> np.ndarray:
    """The observations :func:`auto_forecast` backtests and fits on: the last
    ``max(400, 8 * season_length)`` of ``y`` (see ``_BT_MAX_HISTORY``).

    The one definition of the forecast's history: when it is all zeros the forecast
    and its bands are exactly 0 (no scale at all), which ``narrative.forecast_is_zero``
    -- and with it the scenario floor, the probability of a rise and the zero-width
    caveat on every surface -- reads through this function.
    """
    return np.asarray(y)[-max(_BT_MAX_HISTORY, 8 * max(1, int(season_length))):]


def _is_intermittent(y: np.ndarray, m: int) -> bool:
    """Intermittent demand: non-negative, many exact zeros overall *and* recently.

    ``>= 30 %`` zeros in the history and in the last ``max(m, 12)`` points
    (a series that was 0 before a new trend took off is not intermittent any
    more), with at least two non-zero values -- :func:`analysis._is_intermittent`,
    the one rule of the analysis, the forecast and every surface (``y`` here is
    already the capped history, which the analysis rule caps the same way).
    """
    return _analysis_is_intermittent(y, m)


def _multiplicative_growth(y: np.ndarray) -> bool:
    """Strictly positive series whose changes scale with the level (compounding growth or decay).

    The level moved by at least 50 % between the first and the last quarter
    of the history in one direction (the average change per period has the
    same sign in both halves), and that average change differs between the
    first and the second half less in log terms than in levels, and by less
    than a factor 1.5 in log terms: steady percentage growth has a steady
    average log change and level changes that grow with the level; a
    straight line (or a random walk with drift) is the other way round, and
    an S-curve's growth rate is not steady.  Average changes are differences of the half's end
    points, so period-to-period noise hardly matters.
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n < 12 or not np.all(y > 0):
        return False
    q = max(2, n // 4)
    lv1, lv2 = float(np.mean(y[:q])), float(np.mean(y[-q:]))
    if max(lv1, lv2) < 1.5 * min(lv1, lv2):
        return False
    half = n // 2
    ly = np.log(y)
    # mean change of each half, from the mean of its first and last few points
    k = max(2, half // 4)

    def avg_change(v: np.ndarray, lo: int, hi: int) -> float:
        return (float(np.mean(v[hi - k:hi])) - float(np.mean(v[lo:lo + k]))) / (hi - lo - k)

    a1, a2 = avg_change(y, 0, half), avg_change(y, n - half, n)
    b1, b2 = avg_change(ly, 0, half), avg_change(ly, n - half, n)
    if not (a1 * a2 > 0 and b1 * b2 > 0):
        return False
    lb, la = abs(math.log(b2 / b1)), abs(math.log(a2 / a1))
    # steady: the growth *rate* of the two halves is similar (an S-curve accelerates, then
    # stalls -- its log growth rate collapses, and log drift would extrapolate the early pace)
    return lb < la and lb < _STEADY_GROWTH_LOG


def _max_moves(x: np.ndarray, J: int) -> tuple[np.ndarray, np.ndarray]:
    """Largest rise and drop of ``x`` over any span of ``<= j`` steps, ``j = 1..J < len(x)``."""
    up = np.empty(J)
    dn = np.empty(J)
    for i in range(1, J + 1):
        d = x[i:] - x[:-i]
        up[i - 1] = d.max()
        dn[i - 1] = -d.min()
    return (np.maximum.accumulate(np.maximum(up, 0.0)),
            np.maximum.accumulate(np.maximum(dn, 0.0)))


def _move_envelope(y: np.ndarray, m: int, h: int, nonnegative: bool = False
                   ) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Plausibility envelope ``(up, down)``: how far a forecast may move ``k`` steps ahead.

    ``up[k-1] = max(1.5 · largest rise over <= k steps anywhere in the
    history, recent pace · k)`` where the recent pace is the largest rise
    within the last ``max(m, n // 4, 8)`` points divided by that window's
    span; ``down`` likewise for drops.  So a forecast may continue the recent
    average pace linearly and may exceed every historical move of the same
    span by 50 %, but it cannot compound a short burst of growth into an
    explosion (e.g. a log-linear trend fitted to 20 days of an S-curve: x16
    the observed maximum within 30 days, or holt extending the speed of the
    very last step for 30 steps).  For ``nonnegative``
    series drops are not limited (``down = inf``): they are bounded by 0
    anyway, and a crash that has just begun is faster than anything in the
    history.  ``None`` for fewer than 3 points (no guard).
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n < 3:
        return None
    steps = np.arange(1, h + 1, dtype=float)
    Ja = min(h, n - 1)
    up_all, dn_all = _max_moves(y, Ja)
    W = min(n, max(m, n // 4, 8))
    rec = y[-W:]
    span = max(W - 1, 1)
    pace_up = float(np.max(rec - np.minimum.accumulate(rec))) / span
    pace_dn = float(np.max(np.maximum.accumulate(rec) - rec)) / span
    ja = np.minimum(np.arange(h), Ja - 1)
    eps = 1e-9 * max(float(np.max(np.abs(y))), 1e-300)
    up = np.maximum(_GUARD_SLACK * up_all[ja], pace_up * steps) + eps
    if nonnegative:
        dn = np.full(h, np.inf)
    else:
        dn = np.maximum(_GUARD_SLACK * dn_all[ja], pace_dn * steps) + eps
    return up, dn


def _guard(f: np.ndarray, env: Optional[tuple[float, np.ndarray, np.ndarray]]) -> np.ndarray:
    """Clip a forecast path into the plausibility envelope ``(y_last, up, down)``."""
    if env is None:
        return f
    y_last, up, dn = env
    h = len(f)
    return np.clip(f, y_last - dn[:h], y_last + up[:h])


def _make_guard(y: np.ndarray, m: int, h: int, nonnegative: bool
                ) -> Optional[tuple[float, np.ndarray, np.ndarray]]:
    """Guard of forecasts made after history ``y``: ``(last value, up, down)`` or ``None``."""
    env = _move_envelope(y, m, h, nonnegative)
    return None if env is None else (float(y[-1]), env[0], env[1])


# --------------------------------------------------------------------------- backtest plan


@dataclass(frozen=True)
class _Plan:
    """Rolling-origin backtest layout shared by all candidate models."""

    horizon: int                 # forecast steps per window
    origins: tuple[int, ...]     # training end positions (exclusive), oldest first
    seasonal: bool               # True if every window trains on >= 2 seasons

    @property
    def n_windows(self) -> int:
        return len(self.origins)

    @property
    def min_train(self) -> int:
        return self.origins[0]

    @property
    def n_targets(self) -> int:
        """Number of distinct observations that are forecast in some window."""
        return self.origins[-1] + self.horizon - self.origins[0]


def _default_n_windows(n: int) -> int:
    return 3 if n < 40 else (5 if n < 100 else 8)


def _step(h: int) -> int:
    return max(1, h // 2)


def _largest_h(avail: int, n_windows: int, horizon: int) -> int:
    """Largest ``h <= horizon`` with ``h + (W-1)·step(h) <= avail`` (0 if none)."""
    for h in range(horizon, 0, -1):
        if h + (n_windows - 1) * _step(h) <= avail:
            return h
    return 0


def _make_plan(n: int, m: int, horizon: int, n_windows: int,
               min_windows: int = _MIN_WINDOWS, prefer_seasonal: bool = False) -> Optional[_Plan]:
    """Choose window count / backtest horizon / training length for a series of length ``n``.

    Seasonal layout first (every window trains on ``>= max(2m, 8)`` points),
    then a non-seasonal one (``>= 6`` points).  Within a layout the largest
    window count whose feasible horizon reaches ``max(1, horizon // 2)`` wins;
    if no layout reaches it, the feasible combination with the most errors
    (``W·h``) is used.  With ``prefer_seasonal`` the seasonal layout is kept
    even when its horizon is shorter (as long as it has 2 windows) --
    otherwise a series of 2 to ~2.4 seasons falls through to the
    non-seasonal layout, whose short training windows exclude every seasonal
    model (weekly n = 120: a flat forecast 3x worse than seasonal naive).
    ``auto_forecast`` sets it whenever ``n >= 2m``: in simulations of 2-2.5
    season series this cut the error of seasonal processes by ~20 % and cost
    non-seasonal ones ~0.5 % (their models still compete in the seasonal
    layout).  Returns ``None`` when not even ``min_windows`` windows of one
    step fit.
    """
    horizon = max(1, int(horizon))
    n_windows = max(1, int(n_windows))
    min_windows = max(1, min(int(min_windows), n_windows))
    h_need = max(1, horizon // 2)
    layouts: list[tuple[int, bool]] = []
    if m > 1:
        layouts.append((max(2 * m, 8), True))
    layouts.append((_MIN_TRAIN_NONSEASONAL, False))
    best: Optional[tuple[int, _Plan]] = None
    for min_train, seasonal in layouts:
        avail = n - min_train
        if avail < 1:
            continue
        layout_best: Optional[tuple[int, _Plan]] = None
        for W in range(n_windows, min_windows - 1, -1):
            h = _largest_h(avail, W, horizon)
            if h < 1:
                continue
            step = _step(h)
            origins = tuple(n - h - (W - 1 - i) * step for i in range(W))
            plan = _Plan(horizon=h, origins=origins, seasonal=seasonal)
            if h >= h_need:
                return plan
            if layout_best is None or W * h > layout_best[0]:
                layout_best = (W * h, plan)
        if layout_best is None:
            continue
        if seasonal and prefer_seasonal and layout_best[1].n_windows >= min(2, n_windows):
            return layout_best[1]
        if best is None or layout_best[0] > best[0]:
            best = layout_best
    return best[1] if best is not None else None


# --------------------------------------------------------------------------- backtest core


def _window_forecasts(y: np.ndarray, name: str, m: int, plan: _Plan,
                      envelopes: Optional[Sequence] = None,
                      factory=None) -> np.ndarray:
    """``(W, h)`` forecasts of model ``name`` for every window of ``plan`` (may raise).

    ``envelopes`` (one per window, see :func:`_move_envelope`) apply the
    plausibility guard; ``factory`` overrides the registry lookup.
    """
    h = plan.horizon
    out = np.empty((plan.n_windows, h))
    for i, o in enumerate(plan.origins):
        model = factory() if factory is not None else _models.make_model(name)
        model.fit(y[:o], season_length=m)
        f = np.asarray(model.predict(h), dtype=float)
        if f.shape != (h,) or not np.all(np.isfinite(f)):
            raise ValueError(f"model {name!r} returned an invalid forecast")
        if envelopes is not None:
            f = _guard(f, envelopes[i])
        out[i] = f
    return out


def _actuals(y: np.ndarray, plan: _Plan) -> np.ndarray:
    return np.stack([y[o:o + plan.horizon] for o in plan.origins])


def _recency_weights(plan: _Plan) -> np.ndarray:
    """Window weights growing geometrically with the origin (newest/oldest = ``_RECENCY_RATIO``)."""
    o = np.asarray(plan.origins, dtype=float)
    if len(o) < 2 or o[-1] <= o[0]:
        return np.full(len(o), 1.0 / len(o))
    pos = (o - o[0]) / (o[-1] - o[0])
    w = _RECENCY_RATIO ** pos
    return w / w.sum()


def _scores(forecasts: np.ndarray, actuals: np.ndarray, scales: np.ndarray,
            weights: Optional[np.ndarray] = None) -> dict:
    """Accuracy metrics of ``(W, h)`` forecasts; MASE uses the per-window training scale.

    Every metric is a (``weights``-weighted, default equal) average over the
    windows of the per-window metric; RMSE averages the per-window MSE.
    """
    W = forecasts.shape[0]
    wts = np.full(W, 1.0 / W) if weights is None else np.asarray(weights, dtype=float)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        err = actuals - forecasts
        abs_e = np.abs(err)
        denom = np.abs(actuals) + np.abs(forecasts)
        smape_terms = np.where(denom > 0, 200.0 * abs_e / np.where(denom > 0, denom, 1.0), 0.0)
        smax = float(np.max(abs_e)) if abs_e.size else 0.0
        if smax > 0 and math.isfinite(smax):
            rmse = math.sqrt(float(wts @ np.mean((abs_e / smax) ** 2, axis=1))) * smax
        else:
            rmse = smax
        return {
            "mae": float(wts @ abs_e.mean(axis=1)),
            "rmse": float(rmse),
            "smape": float(wts @ smape_terms.mean(axis=1)),
            "mase": float(wts @ (abs_e.mean(axis=1) / scales)),
        }


def _finite_scores(sc: dict) -> bool:
    return all(math.isfinite(sc[k]) for k in ("mae", "rmse", "smape", "mase"))


def backtest(y: np.ndarray, model_name: str, season_length: int, horizon: int,
             n_windows: int = 3) -> dict:
    """Rolling-origin backtest of one registered model.

    Windows end at the end of ``y`` and step back by ``max(1, h // 2)``; the
    oldest window keeps at least ``max(2·season_length, 8)`` training points
    when possible (else 6).  For short series fewer windows and/or a shorter
    horizon than requested are used -- the shape of the returned arrays tells
    which.  MASE is averaged over windows, each scaled by :func:`mase_scale`
    of its own training part (floored at 10 % of the scale of the history
    before the latest window origin, so that a constant training window
    cannot make MASE depend on the units of the data).  Windows are weighted
    equally and the raw model forecasts are scored (``auto_forecast`` scores
    guarded forecasts with recency weights, see there).

    Returns ``{'mae', 'rmse', 'smape', 'mase', 'forecasts': (W, h) array,
    'actuals': (W, h) array, 'origins': list of training lengths}``.

    Raises ``ValueError`` if the series is too short for even one window, and
    propagates any exception raised by the model.
    """
    y = np.asarray(y, dtype=float).ravel()
    if not np.all(np.isfinite(y)):
        raise ValueError("y contains non-finite values")
    if int(horizon) < 1:
        raise ValueError("horizon must be >= 1")
    m = max(1, int(season_length))
    plan = _make_plan(len(y), m, int(horizon), int(n_windows), min_windows=1)
    if plan is None:
        raise ValueError(f"series too short to backtest (n={len(y)})")
    fc = _window_forecasts(y, model_name, m, plan)
    act = _actuals(y, plan)
    scales = _window_scales(y, plan.origins, m)
    out = _scores(fc, act, scales)
    out.update({"forecasts": fc, "actuals": act, "origins": list(plan.origins)})
    return out


# --------------------------------------------------------------------------- intervals


def _fit_sigma(errors: np.ndarray, horizon: int, floor: float) -> tuple[np.ndarray, float, float]:
    """Per-step error scale ``σ_k = s₁·k^b`` (k = 1..horizon) from ``(W, h)`` backtest errors.

    ``b`` comes from a log-log fit of the per-step RMS errors (clipped to
    ``[0.1, 1]``); ``s₁`` is then re-estimated from *all* errors pooled
    (``e_{w,k} / k^b``), slightly inflated for the optimism of choosing the
    best model on the same windows, and floored by ``floor``.  Beyond the
    backtest horizon ``h`` the scale grows at least like ``√k``
    (``σ_k = σ_h·(k/h)^max(b, 0.5)``): a short backtest cannot show that
    errors stop growing.  Returns ``(sigma, s1, b)``.
    """
    errors = np.asarray(errors, dtype=float)
    W, h = errors.shape
    k = np.arange(1, h + 1, dtype=float)
    rms_k = _rms(errors, axis=0)
    b = _B_DEFAULT
    if h >= 3 and np.all(rms_k > 0):
        slope = float(np.polyfit(np.log(k), np.log(rms_k), 1)[0])
        if math.isfinite(slope):
            b = slope
    elif h == 2 and np.all(rms_k > 0):
        b = float(math.log(rms_k[1] / rms_k[0]) / math.log(2.0))
    b = float(np.clip(b, _B_MIN, _B_MAX))
    s1 = float(_rms(errors / k ** b))
    if not math.isfinite(s1):
        s1 = 0.0
    # few (overlapping) windows + best-of-N selection => optimistic errors
    s1 *= _SIGMA_INFLATION_BASE * math.sqrt(1.0 + 1.0 / max(W, 1))
    s1 = max(s1, floor)
    kk = np.arange(1, horizon + 1, dtype=float)
    sigma = s1 * kk ** b
    if horizon > h:
        b_ext = max(b, _B_EXTRAPOLATE)
        sigma = np.where(kk <= h, sigma, s1 * h ** b * (kk / h) ** b_ext)
    return sigma, s1, b


def _error_dof_factor(errors: np.ndarray) -> float:
    """Degrees of freedom per distinct backtest target, from the errors' autocorrelation.

    ``(1 - ρ) / (1 + ρ)`` (the AR(1) effective-sample-size factor, at least
    0.05) with ``ρ`` the pooled lag-1 autocorrelation of the errors within the
    windows: independent errors carry one degree of freedom each, while
    random-walk-like multi-step errors (``ρ ≈ 0.9``) come from only a few
    independent paths.
    """
    e = np.asarray(errors, dtype=float)
    if e.ndim != 2 or e.shape[1] < 2:
        return 1.0
    ref = float(np.max(np.abs(e)))
    if not ref > 0:
        return 1.0
    e = e / ref
    den = 0.5 * float(np.sum(e[:, :-1] ** 2 + e[:, 1:] ** 2))
    if not den > 0:
        return 1.0
    rho = min(max(float(np.sum(e[:, :-1] * e[:, 1:])) / den, 0.0), 0.95)
    return max((1.0 - rho) / (1.0 + rho), 0.05)


def _shrink_sigma(sigma_bt: np.ndarray, prior: np.ndarray, dof_bt: float,
                  dof_factor: float = _DOF_PER_TARGET) -> tuple[np.ndarray, float]:
    """Pool the backtest error scale with the random-walk prior (conjugate-style).

    ``σ² = (ν·σ_bt² + ν₀·σ_prior²) / (ν + ν₀)`` with ``ν`` the number of
    distinct forecast targets minus one and ``ν₀ = 2``; the result is used
    with Student-t quantiles on ``dof_factor·ν + ν₀`` degrees of freedom
    (``dof_factor <= 0.5``, see :func:`_error_dof_factor`).  A backtest of
    3-5 target points (n ≈ 8-12) is thus dominated by the prior and the
    t-quantiles, one of 50 independent points barely changes.  Computed in
    units of the largest scale (no overflow of squares).
    """
    nu = max(float(dof_bt), 1.0)
    w = nu / (nu + _PRIOR_DOF)
    ref = max(float(np.max(sigma_bt)), float(np.max(prior)), 1e-300)
    a = np.asarray(sigma_bt, dtype=float) / ref
    p = np.asarray(prior, dtype=float) / ref
    return np.sqrt(w * a * a + (1.0 - w) * p * p) * ref, dof_factor * nu + _PRIOR_DOF


def _member_spread(members: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    """Per-step scale that makes a 95 % band span every ensemble member's forecast.

    ``(max - min of the members' forecasts) / (2·z_0.975)``: when the members
    disagree by more than the backtest-based interval, the interval widens
    just enough for its 95 % band to contain all of them (an explosive member
    can no longer drag the mean far away under a narrow band).
    """
    names = [nm for nm in weights if nm in members and weights[nm] > 0]
    if len(names) < 2:
        return np.zeros(len(next(iter(members.values()))))
    F_ = np.stack([members[nm] for nm in names])
    return (F_.max(axis=0) - F_.min(axis=0)) / (2.0 * _level_z(95))


def _local_level(y: np.ndarray, end: int, m: int) -> float:
    """Mean absolute level of the ``max(m, 4)`` observations before position ``end``."""
    w = min(end, max(m, 4))
    return float(np.mean(np.abs(y[end - w:end]))) if w > 0 else 0.0


def _level_adjust(errors: np.ndarray, y: np.ndarray, origins: Sequence[int], m: int
                  ) -> tuple[np.ndarray, float]:
    """Rescale backtest errors to the current level when their size depends on the level.

    For strictly positive series whose level changed noticeably across the
    backtest windows (max/min of the window levels ``>= 1.25``), the elasticity
    ``γ`` of the per-window RMS error with respect to the window's level is
    estimated by a log-log regression and clipped to ``[0, 1]``; every error
    is then multiplied by ``(level_now / level_window)^γ``.  ``γ ≈ 1`` means
    multiplicative behaviour (errors scale with the level: exponential growth,
    a collapsed fad), ``γ ≈ 0`` additive (nothing changes).  Returns
    ``(errors, γ)``.
    """
    errors = np.asarray(errors, dtype=float)
    W = errors.shape[0]
    if W < 3 or not np.all(y > 0):
        return errors, 0.0
    lv_w = np.array([_local_level(y, o, m) for o in origins])
    lv_now = _local_level(y, len(y), m)
    if np.any(lv_w <= 0) or lv_now <= 0:
        return errors, 0.0
    allv = np.r_[lv_w, lv_now]
    if float(allv.max() / allv.min()) < 1.25:
        return errors, 0.0
    rms_w = _rms(errors, axis=1)
    ok = rms_w > 0
    if ok.sum() < 3:
        return errors, 0.0
    x = np.log(lv_w[ok])
    yv = np.log(rms_w[ok])
    vx = float(np.var(x))
    if vx <= 1e-12:
        return errors, 0.0
    gamma = float(np.mean((x - x.mean()) * (yv - yv.mean())) / vx)
    if not math.isfinite(gamma):
        return errors, 0.0
    gamma = float(np.clip(gamma, 0.0, 1.0))
    if gamma == 0.0:
        return errors, 0.0
    return errors * ((lv_now / lv_w) ** gamma)[:, None], gamma


def _bands(mean: np.ndarray, sigma: np.ndarray, levels: Sequence[int], nonnegative: bool,
           index: pd.DatetimeIndex, dof: float = math.inf
           ) -> tuple[pd.Series, dict[int, pd.Series], dict[int, pd.Series]]:
    """Bands ``mean ± q·σ`` (Student-t ``q`` on ``dof``), clipped at 0 if ``nonnegative``."""
    mean = np.asarray(mean, dtype=float)
    if nonnegative:
        mean = np.maximum(mean, 0.0)
    lower: dict[int, pd.Series] = {}
    upper: dict[int, pd.Series] = {}
    for lv in levels:
        q = _level_q(lv, dof)
        lo = mean - q * sigma
        hi = mean + q * sigma
        if nonnegative:
            lo = np.maximum(lo, 0.0)
            hi = np.maximum(hi, 0.0)
        lower[lv] = pd.Series(lo, index=index, name=f"lo{lv}")
        upper[lv] = pd.Series(hi, index=index, name=f"hi{lv}")
    return pd.Series(mean, index=index, name="mean"), lower, upper


# --------------------------------------------------------------------------- argument checks


def _check_levels(levels: Sequence[float]) -> tuple[int, ...]:
    out: list[int] = []
    for lv in levels:
        f = float(lv)
        iv = int(round(f))
        if abs(f - iv) > 1e-9 or not 0 < iv < 100:
            raise ValueError(f"interval levels must be integers in (0, 100), got {lv!r}")
        if iv not in out:
            out.append(iv)
    return tuple(out)


def _describe(weights: dict[str, float], prefix: str = "ensemble") -> str:
    if len(weights) == 1:
        return next(iter(weights))
    parts = ", ".join(f"{nm} {w:.2f}" for nm, w in weights.items())
    return f"{prefix}({parts})"


# --------------------------------------------------------------------------- fallback (very short series)


def _significant_trend(y: np.ndarray) -> bool:
    """Whether the OLS slope of ``y`` on time is significant (two-sided 5 % t-test, n - 2 dof).

    An exact line (no residual) with a non-zero slope counts as significant;
    fewer than 3 points or a zero slope do not.
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n < 3 or not np.all(np.isfinite(y)):
        return False
    x = np.arange(n, dtype=float) - (n - 1) / 2.0
    c = float(np.max(np.abs(y))) or 1.0          # unit scale: no overflow at |y| ~ 1e200
    z = y / c
    sxx = float(np.sum(x * x))
    b = float(np.sum(x * (z - z.mean())) / sxx)
    resid = z - z.mean() - b * x
    s2 = float(np.sum(resid * resid)) / (n - 2)
    if b == 0.0:
        return False
    if s2 <= (1e-12 * max(abs(b) * n, 1e-300)) ** 2:
        return True                               # an exact line
    return abs(b) / math.sqrt(s2 / sxx) >= t_ppf(0.975, n - 2)


def _fallback_forecast(y: np.ndarray, m: int, horizon: int, reason: str
                       ) -> tuple[np.ndarray, np.ndarray, dict[str, float], str]:
    """Equal-weight drift + naive with wide heuristic intervals; naive alone when there is no
    trend to extrapolate.

    The drift is used only when the series has at least 3 points and a
    significant linear trend (:func:`_significant_trend`: OLS slope, two-sided
    5 % t-test).  Without that test 7 daily points of flat noise
    (Mann-Kendall p = 0.37) had their end-to-end slope extrapolated over the
    30-day default horizon: "down, −50.7 %, P(up) = 22 %" -- 30 to 70 % of
    such noise series got a direction call.  Uses the built-in classes
    directly (not the registry) so it cannot be broken by a re-registered
    model.  No model scores are produced: a handful of one-step errors is no
    backtest, and the numbers it gave (e.g. a naive "MASE" of 10 for n = 2)
    only looked like evidence.
    """
    n = len(y)
    factories: dict[str, type[_models.Forecaster]] = {"naive": _models.Naive}
    if n >= 3 and _significant_trend(y):
        factories = {"drift": _models.Drift, "naive": _models.Naive}
    elif n >= 3:
        reason = f"{reason}; no significant trend"
    w = 1.0 / len(factories)
    weights = {nm: w for nm in factories}
    mean = np.zeros(horizon)
    for cls in factories.values():
        mean += w * cls().fit(y, m).predict(horizon)

    # heuristic wide intervals: RMS of first differences, random-walk growth, widened
    level = float(np.mean(np.abs(y)))
    s1 = float(_rms(np.diff(y))) if n >= 2 else 0.0
    s1 = max(s1, 0.05 * level, _tiny_scale(y))
    sigma = _FALLBACK_WIDEN * s1 * np.sqrt(np.arange(1, horizon + 1, dtype=float))
    method = f"{_describe(weights, 'fallback')} [{reason}]"
    if len(weights) == 1:
        method = f"fallback({next(iter(weights))}) [{reason}]"
    return mean, sigma, weights, method


# --------------------------------------------------------------------------- auto_forecast


_BENCHMARKS: dict[str, type[_models.Forecaster]] = {
    "naive": _models.Naive,
    "seasonal_naive": _models.SeasonalNaive,
    "drift": _models.Drift,
}


#: backtest errors below this share of the series' mean absolute level are rounding noise
#: (an exact line fitted by drift, holt and linear_trend: errors ~1e-16 of the level)
_EXACT_FRAC = 1e-9


def _skill(final_err: float, bench_err: dict[str, float], scale: float = 0.0) -> Optional[float]:
    """``final_err / min(bench_err)`` (finite, capped at 100); ``None`` without benchmarks.

    Both errors are floored at ``1e-9 · scale`` (``scale`` = mean absolute level
    of the series): on exact data every model is exact up to rounding, and
    ``1e-16 / 0`` used to read as "100 times worse than naive".  Both exact
    gives ``1.0`` (as good as the benchmark).
    """
    vals = [v for v in bench_err.values() if math.isfinite(v)]
    if not vals or not math.isfinite(final_err):
        return None
    best = min(vals)
    eps = max(_EXACT_FRAC * float(scale) if math.isfinite(scale) else 0.0,
              1e-12 * max(best, final_err), 1e-300)
    return float(min(max(final_err, eps) / max(best, eps), 100.0))


def is_random_walk(analysis: AnalysisResult) -> bool:
    """Whether the analysis found a driftless random walk (for ``auto_forecast(random_walk=)``).

    The noise around the trend is (near) a unit root -- the analysis' AR(1)
    noise model gives ``phi >= 0.9`` -- and the dependence-corrected trend test
    finds no significant trend (``metrics.direction == 'flat'``): the series
    wanders, and any slope fitted to it is that sample's noise path.
    """
    try:
        from .analysis import _noise_model, _unit
        if getattr(analysis.metrics, "direction", "flat") != "flat":
            return False
        dec = analysis.decomposition
        # the same scaling as analysis.compute_metrics, so its noise-model cache is reused
        c = _unit(np.asarray(analysis.series.to_numpy(dtype=float)))
        T = np.asarray(pd.Series(dec.trend).to_numpy(dtype=float)) / c
        R = np.asarray(pd.Series(dec.resid).to_numpy(dtype=float)) / c
        z = T + R
        if len(z) < 12 or len(T) != len(R) or not np.all(np.isfinite(z)):
            return False
        phi = float(_noise_model(z, int(dec.period), R)[0])
    except (AttributeError, ImportError, TypeError, ValueError):
        return False
    return bool(phi >= _RW_PHI)


def auto_forecast(series: pd.Series, horizon: int | None = None, freq: str | None = None,
                  levels: tuple[int, ...] = (80, 95), candidates: list[str] | None = None,
                  top_k: int = 3, n_windows: int | None = None,
                  nonnegative: bool | None = None, random_walk: bool = False) -> ForecastResult:
    """Forecast ``series`` ``horizon`` steps ahead with automatic model selection.

    Parameters
    ----------
    series:
        Time series (raw input is fine: it goes through :func:`data.prepare`).
    horizon:
        Number of future periods (default :func:`data.default_horizon`).
    freq:
        Canonical frequency code; inferred when ``None``.
    levels:
        Central prediction-interval levels in percent.
    candidates:
        Registry names to try (default :func:`models.default_candidates`).
        Unknown or failing models are skipped with a warning.
    top_k:
        Size of the inverse-MASE weighted ensemble.
    n_windows:
        Target number of backtest windows (default 3 / 5 / 8 for fewer than
        40 / 100 / more observations; fewer are used if the series is short).
    nonnegative:
        Clip mean and bounds at 0 (default: iff all observations are >= 0).
    random_walk:
        The series is a driftless random walk (see :func:`is_random_walk`):
        the trend-extrapolating models (``linear_trend``, ``logistic_growth``,
        ``holt``, ``holt_winters`` and their log variants) are not default
        candidates, since the slope they would fit is the noise path of the
        sample.  Ignored when ``candidates`` is given.

    Notes
    -----
    * History: backtest and final fit both use the last
      ``max(400, 8·season_length)`` observations.
    * Spikes: isolated one/two-point outliers (robust ``|z| >= 5`` around a
      running median + seasonal medians, never the last point) are replaced
      by their expected value before anything is fitted or scored, so a
      one-off event is not projected as recurring seasonality (see
      :func:`_clean_spikes`).
    * Series of at least two seasons always keep the seasonal backtest
      layout (with a shorter backtest horizon if need be) instead of losing
      every seasonal model at 2 to ~2.4 seasons (see :func:`_make_plan`).
    * Scores: ``ModelScore`` metrics are averages over the backtest windows
      weighted towards the recent windows (newest 3x the oldest), computed on
      the guarded forecasts; models are ranked by that MASE (by the RMSE for
      intermittent demand, see below).  ``model_scores`` is always listed by
      MASE.
    * The ensemble (inverse-MASE weights over the top-k) is used unless the
      best single model's backtest MASE is below 0.9 x the ensemble's (and the
      backtest has at least 6 errors per model); ``ModelScore.weight`` holds
      the final weights (1 for a single chosen model).
    * Plausibility guard: every model forecast (backtest windows and final
      fit) is clipped so that ``k`` steps ahead it moves at most
      ``max(1.5 x the largest k-step move in its training history, the
      strongest recent pace x k)`` away from the last observation (see
      :func:`_move_envelope`).  Steady trends and seasonal swings pass
      unchanged; compounding a short burst into an explosion does not.
    * Driftless random walk (``random_walk=True``): ``linear_trend``,
      ``logistic_growth``, ``holt``, ``holt_winters`` and their log variants
      are left out of the default candidates (their slope would be the noise
      path of the sample), and the interval prior keeps its ``√k`` growth.
    * Compounding growth (strictly positive, steady percentage growth or
      decay, see :func:`_multiplicative_growth`): ``drift@log`` joins the
      default candidates (level-space models project linear growth and fall
      behind a compounding series).
    * Intermittent demand (``>= 30 %`` zeros overall and recently): only
      level-type models (``ses``, ``theta``, ``holt_damped``, ``croston``,
      ``sba``, ``tsb`` -- those registered; ``sba`` and ``tsb`` are offered
      for intermittent demand only; ``tsb`` lowers its rate through a run of
      zeros, so a product that stopped selling is not forecast at last
      year's rate) are default candidates (MAE-based ranking would otherwise
      pick naive/drift and anchor the forecast on the last observation).  They
      are ranked and weighted by backtest RMSE rather than MASE: MAE is
      minimised by the median, about 0 for mostly-zero data, so it favours
      forecasts that under-state the demand rate.  Spikes are not cleaned.
      ``compute_outlook`` compares average demand rates (the mean of the
      last season, zeros included, with the mean of the forecast's last
      season) and reports no percentage change only when at least half of
      the current window is zero (``outlook_levels(...).undefined ==
      'mostly_zero'``); the analysis' trend-based percentages
      (``recent_change_pct`` etc.) are ``None`` for all intermittent demand.
    * Intervals: ``mean ± q·σ_k``.  ``σ_k = s₁·k^b`` is fitted to the final
      method's backtest errors (``b`` in ``[0.1, 1]``, at least ``√k`` growth
      beyond the backtest horizon, ``s₁`` floored by a robust in-sample noise
      level, slightly inflated for few windows); for positive series the
      errors are first rescaled to the current level (see
      :func:`_level_adjust`).  It is then pooled
      with a prior scale -- the random-walk scale at lead 1, growing with
      the lead like the backtest errors when the backtest covers at least 3
      leads (like ``√k`` otherwise, and for a driftless random walk) -- with
      weight ``ν/(ν+2)`` (``ν`` = distinct backtest targets - 1) and ``q``
      is the Student-t quantile on
      ``f·ν + 2`` degrees of freedom (``f = min(0.5, (1-ρ)/(1+ρ))``, ``ρ`` the
      lag-1 autocorrelation of the backtest errors), so tiny backtests (short
      series) and few independent long-horizon paths get honest, wide
      intervals; finally ``σ_k`` is large enough for the 95 % band to contain
      every ensemble member's forecast and is made non-decreasing in ``k``.  With
      ``nonnegative`` the bands are centred on the clipped mean, then clipped.
      An all-zero history has no scale and gets zero-width bands (any other
      constant keeps a negligible, level-relative band).
    * ``skill`` = backtest MAE of the final method / MAE of the best of the
      naive benchmarks (naive, drift, seasonal naive when ``m > 1`` and the
      windows hold a season) on the same windows -- always backtested, even
      when they are not candidates.  ``< 1`` means better than naive.  For
      intermittent demand both are RMSEs (the metric the models were chosen
      by: MAE prefers the all-zero naive forecast).  Errors below ``1e-9`` of
      the mean level count as exact (both exact: ``1.0``).
    * Series too short for a 2-window backtest (fewer than ~8 points), or for
      which every candidate fails, get an equal-weight drift + naive forecast
      with wide heuristic intervals, ``backtest_mase=None``, ``skill=None``
      and no model scores.
    """
    s, freq = prepare(series, freq=freq)
    H = default_horizon(freq) if horizon is None else int(horizon)
    if H < 1:
        raise ValueError("horizon must be >= 1")
    if int(top_k) < 1:
        raise ValueError("top_k must be >= 1")
    if n_windows is not None and int(n_windows) < 1:
        raise ValueError("n_windows must be >= 1")
    lvls = _check_levels(levels)
    y_full = s.to_numpy(dtype=float)
    m = _freq_season_length(freq)
    if nonnegative is None:
        nonnegative = bool(np.min(y_full) >= 0)
    nonnegative = bool(nonnegative)
    index = future_index(s.index[-1], freq, H)

    y_raw = fit_history(y_full, m)
    intermittent = _is_intermittent(y_raw, m)
    y = y_raw if intermittent else _clean_spikes(y_raw, m)[0]
    n = len(y)
    W_target = int(n_windows) if n_windows is not None else _default_n_windows(n)
    plan = _make_plan(n, m, H, W_target, prefer_seasonal=m > 1 and n >= 2 * m)
    # an all-zero history has no scale at all: every band would be float noise in absolute
    # units (a '낙관 5.28e−9' scenario for a series that never moved), so it gets none
    no_scale = not bool(np.any(y_raw))

    def fallback(reason: str) -> ForecastResult:
        mean, sigma, _, method = _fallback_forecast(y, m, H, reason)
        if no_scale:
            sigma = np.zeros_like(sigma)
        mean_s, lower, upper = _bands(mean, sigma, lvls, nonnegative, index)
        return ForecastResult(mean=mean_s, lower=lower, upper=upper, horizon=H, levels=lvls,
                              model_scores=[], method=method, backtest_mase=None,
                              nonnegative=nonnegative, skill=None)

    if plan is None:
        return fallback("too short to backtest")

    # ---- candidates
    positive = bool(np.all(y > 0))
    if candidates is None:
        names = list(_models.default_candidates(plan.min_train, m, positive,
                                                intermittent=intermittent))
        if intermittent:
            level_models = [nm for nm in names if nm in _INTERMITTENT_MODELS]
            if level_models:
                names = level_models
        if random_walk:
            no_trend = [nm for nm in names if nm not in _TREND_EXTRAPOLATORS]
            if no_trend:
                names = no_trend
        elif (not intermittent and _LOG_DRIFT in _models.MODEL_REGISTRY and _LOG_DRIFT not in names
              and plan.min_train >= _models.make_model(_LOG_DRIFT).min_obs
              and _multiplicative_growth(y)):
            names.append(_LOG_DRIFT)
    else:
        names = [str(c) for c in candidates]
    names = list(dict.fromkeys(names))  # dedupe, keep order

    # ---- backtest every candidate (and the naive benchmarks) on the same windows
    actuals = _actuals(y, plan)
    scales = _window_scales(y, plan.origins, m)
    wts = _recency_weights(plan)
    envelopes = [_make_guard(y[:o], m, plan.horizon, nonnegative) for o in plan.origins]

    def clip0(f: np.ndarray) -> np.ndarray:
        return np.maximum(f, 0.0) if nonnegative else f

    results: dict[str, tuple[np.ndarray, dict]] = {}
    for nm in names:
        try:
            fc = _window_forecasts(y, nm, m, plan, envelopes)
        except Exception as exc:  # noqa: BLE001 - a broken model must never break the forecast
            warnings.warn(f"forecast: model {nm!r} skipped in backtest "
                          f"({type(exc).__name__}: {exc})", RuntimeWarning, stacklevel=2)
            continue
        fc = clip0(fc)
        sc = _scores(fc, actuals, scales, wts)
        if not _finite_scores(sc):
            warnings.warn(f"forecast: model {nm!r} skipped (non-finite backtest error)",
                          RuntimeWarning, stacklevel=2)
            continue
        results[nm] = (fc, sc)
    if not results:
        warnings.warn("forecast: no candidate model could be backtested; using fallback",
                      RuntimeWarning, stacklevel=2)
        return fallback("no usable candidate model")

    # selection metric: MASE, except RMSE for intermittent demand (see _INTERMITTENT_METRIC);
    # the skill is measured in the same terms (MAE, or RMSE for intermittent demand)
    key = _INTERMITTENT_METRIC if intermittent else "mase"
    skill_key = _INTERMITTENT_METRIC if intermittent else "mae"
    bench_err: dict[str, float] = {}
    for nm, cls in _BENCHMARKS.items():
        if nm == "seasonal_naive" and not (m > 1 and plan.min_train >= m):
            continue
        try:
            fc = clip0(_window_forecasts(y, nm, m, plan, factory=cls))
        except Exception:  # noqa: BLE001 - benchmarks are built-ins; never fatal
            continue
        bench_err[nm] = _scores(fc, actuals, scales, wts)[skill_key]

    order = {nm: i for i, nm in enumerate(names)}
    ranked = sorted(results, key=lambda nm: (results[nm][1][key], order[nm]))

    # ---- ensemble vs. best single model
    best = ranked[0]
    chosen: dict[str, float] = {best: 1.0}
    k = min(int(top_k), len(ranked))
    if k >= 2:
        top = ranked[:k]
        vals = [results[nm][1][key] for nm in top]
        vmax = max(vals)
        if not vmax > 0:
            # every member was exact in the backtest (e.g. a run of zeros forecast as zeros): equal
            # weights -- 1 / max(0, 1e-312) overflowed to inf and inf / inf gave NaN weights
            inv = np.ones(len(vals))
        else:
            floor = 1e-9 if key == "mase" else 1e-12 * vmax
            inv = np.array([1.0 / max(v, floor) for v in vals])
        ens_w = inv / inv.sum()
        ens_fc = np.tensordot(ens_w, np.stack([results[nm][0] for nm in top]), axes=1)
        ens_score = _scores(ens_fc, actuals, scales, wts)[key]
        enough = plan.n_windows * plan.horizon >= _MIN_ERRORS_FOR_SINGLE
        if not (enough and results[best][1][key] < _ENSEMBLE_MARGIN * ens_score):
            chosen = {nm: float(w) for nm, w in zip(top, ens_w)}

    # ---- final fit on the same history the backtest used
    env_final = _make_guard(y, m, H, nonnegative)

    def final_fit(nm: str) -> Optional[np.ndarray]:
        try:
            f = np.asarray(_models.make_model(nm).fit(y, season_length=m).predict(H), dtype=float)
            if f.shape != (H,) or not np.all(np.isfinite(f)):
                raise ValueError("invalid forecast")
        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"forecast: model {nm!r} failed on the full history "
                          f"({type(exc).__name__}: {exc})", RuntimeWarning, stacklevel=3)
            return None
        return clip0(_guard(f, env_final))

    fitted: dict[str, np.ndarray] = {}
    for nm in chosen:
        f = final_fit(nm)
        if f is not None:
            fitted[nm] = f
    if not fitted:
        # every chosen model failed on the full history: use the best-ranked one that works
        for nm in ranked:
            if nm in chosen:
                continue
            f = final_fit(nm)
            if f is not None:
                fitted[nm] = f
                chosen = {nm: 1.0}
                break
    if not fitted:
        return fallback("final fit failed")
    tot = sum(chosen[nm] for nm in fitted)
    weights = {nm: chosen[nm] / tot for nm in fitted}

    mean = np.zeros(H)
    bt_fc = np.zeros_like(actuals)
    for nm, w in weights.items():
        mean += w * fitted[nm]
        bt_fc += w * results[nm][0]
    final_scores = _scores(bt_fc, actuals, scales, wts)
    with np.errstate(over="ignore"):
        level = float(np.mean(np.abs(y)))
    skill = _skill(final_scores[skill_key], bench_err, level)

    # ---- prediction intervals from the final method's backtest errors
    errors = actuals - bt_fc
    if _LEVEL_ADJUST:
        errors, _ = _level_adjust(errors, y, plan.origins, m)
    sigma_bt, _, _ = _fit_sigma(errors, H, _noise_floor(y, m))
    # best-of-N selection on the same windows makes the chosen method's errors optimistic
    sigma_bt = sigma_bt * math.sqrt(1.0 + _SELECTION_PENALTY * math.log(max(len(results), 1))
                                    / plan.n_targets)
    dof_factor = min(_DOF_PER_TARGET, _error_dof_factor(errors))
    prior = _prior_sigma(y, m, H)
    if _PRIOR_FOLLOWS_BACKTEST and plan.horizon >= 3 and sigma_bt[0] > 0 and not random_walk:
        # the backtest shows how the errors grow with the lead (at least 3 leads): the prior
        # only sets the level, with that growth -- a random-walk sqrt(k) prior widened the
        # long leads of stationary series (white noise D400: sigma at lead 30 x1.25 from a
        # prior weighted 1.5 %)
        prior = prior[0] * (sigma_bt / sigma_bt[0])
    sigma, dof = _shrink_sigma(sigma_bt, prior, plan.n_targets - 1, dof_factor)
    sigma = np.maximum(sigma, _member_spread(fitted, weights))
    sigma = np.maximum.accumulate(sigma)
    if no_scale:
        sigma = np.zeros_like(sigma)
    mean_s, lower, upper = _bands(mean, sigma, lvls, nonnegative, index, dof)

    # reported best (lowest MASE) first, whatever metric selected the models
    model_scores = [
        ModelScore(name=nm, mae=results[nm][1]["mae"], rmse=results[nm][1]["rmse"],
                   smape=results[nm][1]["smape"], mase=results[nm][1]["mase"],
                   weight=float(weights.get(nm, 0.0)))
        for nm in sorted(results, key=lambda nm: (results[nm][1]["mase"], order[nm]))
    ]
    return ForecastResult(mean=mean_s, lower=lower, upper=upper, horizon=H, levels=lvls,
                          model_scores=model_scores, method=_describe(weights),
                          backtest_mase=float(final_scores["mase"]), nonnegative=nonnegative,
                          skill=skill)


# --------------------------------------------------------------------------- outlook


def interval_level(forecast: ForecastResult, target: int = 80) -> Optional[int]:
    """The available interval level closest to ``target`` (``None`` if there is none; ties: the
    lower level).

    With the default ``target`` this is the band behind ``Outlook.scenarios`` and the
    uncertainty of :func:`compute_outlook`, so every surface that labels the scenarios
    ("80% 예측구간") names the band they came from -- also for custom levels such as
    ``(50, 95)``, where it is the 95 % band.
    """
    avail = [int(lv) for lv in forecast.levels if lv in forecast.lower and lv in forecast.upper]
    if not avail:
        return None
    return min(avail, key=lambda lv: (abs(lv - target), lv))


_interval_level = interval_level


def _has_inner_peak(f: np.ndarray, min_rise: float = 0.0, anchor: Optional[float] = None,
                    min_fall: float = 0.05) -> bool:
    """True if ``f`` rises to a maximum strictly inside it and then falls ``>= min_fall``.

    * the maximum ``f[i]`` must satisfy ``0 < i < len(f) - 1``;
    * afterwards ``f`` must fall by at least ``min_fall · |f[i]|`` (5 %);
    * before it, ``f`` must have *risen* by more than ``min_rise`` from its lowest
      earlier point (or from ``anchor``, the current level, if that is lower) --
      without this a noise-sized wiggle in the first steps of a declining
      forecast would be reported as a "peak".
    """
    f = np.asarray(f, dtype=float)
    if len(f) < 3 or not np.all(np.isfinite(f)):
        return False
    i = int(np.argmax(f))
    if not 0 < i < len(f) - 1:
        return False
    peak = float(f[i])
    fall = peak - float(np.min(f[i + 1:]))
    if not (fall > 0 and fall >= min_fall * abs(peak)):
        return False
    base = float(np.min(f[:i]))
    if anchor is not None and math.isfinite(anchor):
        base = min(base, float(anchor))
    rise = peak - base
    return bool(rise > 0 and rise > min_rise)


def _smooth(x: np.ndarray, w: int) -> np.ndarray:
    """Centred moving average of odd width ``w`` (windows shrink at the edges)."""
    x = np.asarray(x, dtype=float)
    w = int(w)
    if w <= 1 or len(x) < 3:
        return x.copy()
    # odd width no longer than x (np.convolve 'same' returns max(len) elements)
    w = min(w if w % 2 else w + 1, len(x) if len(x) % 2 else len(x) - 1)
    kernel = np.ones(w)
    num = np.convolve(x, kernel, mode="same")
    cnt = np.convolve(np.ones(len(x)), kernel, mode="same")
    return num / cnt


def _seasonal_component(analysis: AnalysisResult) -> tuple[Optional[np.ndarray], int]:
    """``(seasonal values aligned with analysis.series, period)``, or ``(None, 1)`` if unusable."""
    dec = getattr(analysis, "decomposition", None)
    if dec is None or getattr(dec, "seasonal", None) is None:
        return None, 1
    try:
        p = int(dec.period)
        seas = np.asarray(pd.Series(dec.seasonal).to_numpy(dtype=float))
    except (TypeError, ValueError):
        return None, 1
    if p <= 1 or seas.ndim != 1 or len(seas) != len(analysis.series) or len(seas) < p:
        return None, 1
    if not np.all(np.isfinite(seas)) or not np.any(seas):
        return None, 1
    return seas, p


def _future_seasonal(analysis: AnalysisResult, horizon: int,
                     forecast_mean: Optional[np.ndarray] = None) -> np.ndarray:
    """The analysis' seasonal component continued over the forecast horizon (zeros if none).

    Additive decompositions: the last season repeated.  Multiplicative ones
    (``analysis._is_multiplicative``: ``seasonal = trend · r`` with a
    periodic ratio ``r``) keep the *ratio* instead, applied to the level of
    the forecast: with ``forecast_mean`` given, step ``k`` gets
    ``r_k · f_k / (1 + r_k)``, i.e. the forecast split into trend and
    seasonal parts in the decomposition's proportions.  Repeating the last
    season's values would keep its swing at the current level, too small for
    a level that has doubled by the end of the horizon and too large after a
    collapse.  Without ``forecast_mean`` the last season is repeated.
    """
    seas, p = _seasonal_component(analysis)
    if seas is None:
        return np.zeros(horizon)
    phase = np.arange(horizon) % p
    last = seas[-p:]  # phases of observations n-p .. n-1; step k (0-based) is t = n + k
    if forecast_mean is not None and len(forecast_mean) == horizon:
        dec = analysis.decomposition
        try:
            mult = _is_multiplicative(dec)
        except (AttributeError, TypeError, ValueError):
            mult = False
        trend = np.asarray(pd.Series(dec.trend).to_numpy(dtype=float)) if mult else None
        if trend is not None and len(trend) == len(seas):
            r = (last / trend[-p:])[phase]
            f = np.asarray(forecast_mean, dtype=float)
            out = r * f / (1.0 + r)
            if np.all(np.isfinite(out)) and np.all(1.0 + r > 0):
                return out
    return last[phase]


def _seasonal_share(f: np.ndarray, s: np.ndarray, period: int) -> float:
    """Share (0..1) of the seasonal pattern ``s`` that the forecast ``f`` contains.

    Ratio of the detrended spread of the forecast to that of the (lightly
    smoothed) seasonal pattern over the horizon, capped at 1: ``≈ 1`` when the
    models reproduce the seasonality, ``0`` for a straight-line / flat forecast.
    Used so that removing seasonality from the forecast never *creates* a
    trend change when the chosen models did not forecast any seasonality
    (e.g. analysis.decompose always fits a -- possibly pure-noise -- seasonal
    component once two full periods are available).  An amplitude ratio is
    used rather than a regression coefficient because the seasonal estimate
    is noisy for long periods (52 weekly phases), which would bias a
    regression towards 0.  ``1`` is returned when it cannot be judged
    (fewer than 4 steps, or a pattern that is a straight line over the horizon).
    """
    f = np.asarray(f, dtype=float)
    s = np.asarray(s, dtype=float)
    k = len(f)
    if k < 4 or len(s) != k or not np.any(s):
        return 1.0
    X = np.column_stack([np.ones(k), np.arange(k, dtype=float)])

    def detrended_std(v: np.ndarray) -> float:
        coef = np.linalg.lstsq(X, v, rcond=None)[0]
        return float(np.std(v - X @ coef))

    d_s = detrended_std(_smooth(s, 2 * (int(period) // 8) + 1))
    if not (d_s > 1e-12 * max(float(np.max(np.abs(s))), 1e-300)):
        return 1.0
    share = detrended_std(f) / d_s
    return float(min(1.0, max(0.0, share))) if math.isfinite(share) else 1.0


#: relative half-width of the 80 % band at the horizon end above which confidence is at
#: most 'medium' / is 'low' (positive series only: a band of ±36 % is no 'high' confidence)
_CONF_WIDTH_MEDIUM, _CONF_WIDTH_LOW = 0.25, 1.0


def _confidence(skill: Optional[float], backtest_mase: Optional[float], n: int, m: int,
                rel_width: Optional[float] = None) -> str:
    """Forecast reliability label from the skill, the sample size and the interval width.

    * ``'low'``: no backtest (``skill`` and ``backtest_mase`` both ``None``),
      ``skill > 1.05`` (no better than a naive benchmark), ``n < 12``, or an
      80 % band at the horizon end wider than ± the level itself.
    * ``'high'``: ``skill <= 0.8`` (at least 20 % better than the best naive
      benchmark on the same windows), enough data (``n >= 3`` seasons, or
      36 points without seasonality) and an 80 % half-width of at most 25 %
      of the level.
    * ``'medium'`` otherwise.

    The skill is used rather than the raw MASE because multi-step MASE of a
    trending series is routinely > 1 even for a forecast far better than
    naive (the MASE scale is the *one-step* naive error).  Forecasts built by
    hand without a ``skill`` but with a ``backtest_mase`` use the former MASE
    rule (``> 1.5`` low, ``< 0.8`` high).  ``rel_width`` is ``None`` when it
    cannot be judged (series not strictly positive).
    """
    if n < 12:
        return "low"
    enough = n >= (3 * m if m > 1 else 36)
    if skill is not None and math.isfinite(skill):
        if skill > 1.05:
            return "low"
        good = skill <= 0.8
    elif backtest_mase is not None and math.isfinite(backtest_mase):
        if backtest_mase > 1.5:
            return "low"
        good = backtest_mase < 0.8
    else:
        return "low"
    if rel_width is not None and math.isfinite(rel_width):
        if rel_width > _CONF_WIDTH_LOW:
            return "low"
        if rel_width > _CONF_WIDTH_MEDIUM:
            return "medium"
    return "high" if (good and enough) else "medium"


#: a 'flat' forecast whose level moves by at most this share of the current level is flat
#: in itself (the base case stops the current move); a larger move rated 'flat' only for
#: its uncertainty says nothing about whether the move stops
_FLAT_POINT_FRAC = 0.02


def _future_phase(current: str, direction: str, strong: bool, change_pct: Optional[float],
                  confidence: str, settled: Optional[bool] = None) -> str:
    """Project the lifecycle phase at the horizon end from the current phase + direction.

    * ``'volatile'`` stays ``'volatile'``: a mean forecast never carries the
      noise, so its smoothness cannot show the noise going away.
    * ``'up'`` / ``'down'``: the phase the move leads to (e.g. growing ->
      declining only for a strong fall, otherwise peaking).
    * ``'flat'``: the end of the current move (growing / emerging -> peaking,
      declining / reviving -> stable) is projected only when the flat forecast
      is informative -- ``settled``: the base case itself barely moves
      (``|change| <= 2 %``), the confidence is not ``'low'`` and the band is
      not too wide to tell a continuing move from a stop (see
      :func:`compute_outlook`).  Otherwise the forecast cannot tell whether
      the current phase ends, and the current phase is kept (a +75 % base case
      rated 'flat' for its uncertainty is no peak).  Without ``settled`` it is
      derived from ``change_pct`` and ``confidence`` alone.
    """
    if current == "volatile":
        return "volatile"
    if direction == "up":
        if current == "emerging":
            return "emerging" if (change_pct is None or change_pct >= 50.0) else "growing"
        if current in ("declining", "reviving"):
            return "reviving"
        if current == "peaking":
            return "growing" if strong else "peaking"
        if current in ("stable", "volatile"):
            return "growing" if strong else "stable"
        return "growing"
    if direction == "down":
        if current in ("growing", "peaking", "emerging"):
            return "declining" if strong else "peaking"
        return "declining"
    # flat
    if settled is None:
        settled = (confidence != "low"
                   and (change_pct is None or abs(change_pct) <= 100.0 * _FLAT_POINT_FRAC))
    if not settled:
        return current
    if current in ("growing", "peaking", "emerging"):
        return "peaking"
    return "stable"


#: share of the future window whose forecast means sit at the 0 floor of a nonnegative forecast from
#: which the future level is the floor rather than a forecast ('floor', see _pct_undefined): the
#: mirror of 'mostly_zero' for the current window
_FLOOR_SHARE = 0.5


def _pct_undefined(y: np.ndarray, current: float, future: float, k_cur: int,
                   fut_window: Optional[np.ndarray] = None, nonnegative: bool = False) -> Optional[str]:
    """Why ``(future - current) / current`` is no meaningful percentage (``None``: it is).

    The analysis' percentage rule (:func:`analysis.pct_change_undefined`, the rule
    behind ``TrendMetrics.total_change_pct`` / ``recent_change_pct`` / ``cagr_pct``
    and every surface that prints such a number), applied to the two outlook
    levels, plus outlook-specific refinements:

    * ``'zero'``: the current level is ~0 (at most 1 % of the mean absolute
      level of the history) or a level is not finite;
    * ``'mostly_zero'``: at least half of the current window is exactly 0 for
      a non-negative series (intermittent demand: a percentage of a mean of
      mostly zeros is meaningless);
    * ``'floor'``: at least half of the future window (``fut_window``, the
      forecast means) sits at the 0 floor of a ``nonnegative`` forecast while the
      current level is positive.  Those means were clipped at 0 -- they are the
      floor, not an expected value (the band above them stays wide) -- so the
      "−100.0 %" (or −99.4 %) they give says only that the forecast hit the floor,
      next to a probability of a rise taken from the whole band;
    * ``'sign_change'``: the future level has the opposite sign of the current
      one -- in either direction (the analysis rule names a negative base by
      its sign or its closeness to 0; a level expected to cross zero *upwards*
      is named for the crossing, like a downward one);
    * ``'negative'``: the current level is negative, at least 0.5 standard
      deviations of the history away from 0 (a percentage of a negative base
      reads backwards: −25.9 → −34.2 is no "−32 %"; the two levels are
      compared in units instead);
    * ``'near_zero'``: the series takes negative values and the current level
      is closer to 0 than 0.5 standard deviations of the history (either side).

    Non-negative series only get ``'zero'``, ``'mostly_zero'`` or ``'floor'``.
    """
    why = pct_change_undefined(current, future, y)
    if why == "zero":
        return why
    if len(y) and bool(np.all(y >= 0)) and np.mean(y[-k_cur:] == 0) >= 0.5:
        return "mostly_zero"
    if (nonnegative and current > 0.0 and fut_window is not None and len(fut_window)
            and float(np.mean(np.asarray(fut_window, dtype=float) <= 0.0)) >= _FLOOR_SHARE):
        return "floor"
    if why in ("negative", "near_zero") and current < 0.0 < future:
        return "sign_change"
    return why


#: partial-season adjustment (see compute_outlook): only for seasonality the analysis found to be
#: more than negligible -- metrics.seasonal_strength >= 0.2, below which the narrative calls it
#: "거의 없습니다" -- and only when the removed effect is material: at least 0.5 % of the window's
#: mean absolute level.  A constant weekly series (seasonal component ~1e-14) was "seasonally
#: adjusted" next to "계절성은 거의 없습니다 (강도 0.00)".
_ADJ_MIN_STRENGTH = 0.2
_ADJ_MIN_SHARE = 0.005


def _seasonality_matters(analysis: AnalysisResult) -> bool:
    """Whether the analysis found seasonality worth adjusting for (unknown strength: yes)."""
    ss = getattr(getattr(analysis, "metrics", None), "seasonal_strength", None)
    try:
        ss = float(ss)
    except (TypeError, ValueError):
        return True
    return not math.isfinite(ss) or ss >= _ADJ_MIN_STRENGTH


def _material(effect: float, window: np.ndarray) -> bool:
    """A seasonal effect worth removing from a window: ``|effect|`` >= 0.5 % of its mean |level|."""
    if not math.isfinite(effect) or effect == 0.0:
        return False
    with np.errstate(over="ignore"):
        ref = float(np.mean(np.abs(window))) if len(window) else 0.0
    return not (math.isfinite(ref) and abs(effect) < _ADJ_MIN_SHARE * ref)


class OutlookLevels(NamedTuple):
    """The two levels :func:`compute_outlook` compares (see :func:`outlook_levels`)."""

    current: float           # current level (mean of the last k_cur observations, adjusted)
    future: float            # future level (mean of the last k_fut forecast means, adjusted)
    k_cur: int               # observations in the current window: min(season length, n)
    k_fut: int               # forecast steps in the future window: min(season length, horizon)
    period: Optional[int]    # period of the usable seasonal component (None: there is none)
    adj_cur: bool            # current window is a partial season and its (material, not
                             # negligible) seasonal effect was removed
    adj_fut: bool            # future window likewise: the (scaled) seasonal effect was removed
    undefined: Optional[str]  # why expected_change_pct is None ('zero', 'mostly_zero', 'floor',
                              # 'sign_change', 'negative', 'near_zero'), None when it is defined
    scale: float = 0.0       # typical absolute level of the history (the reference of the % rule,
                             # analysis._pct_reference): what a move of a level at or near 0 is
                             # measured against where no % applies (narrative.direction_label)


def _levels(analysis: AnalysisResult, forecast: ForecastResult
            ) -> tuple[OutlookLevels, np.ndarray]:
    """:func:`outlook_levels` plus the scaled seasonal continuation over the horizon."""
    y_all = np.asarray(analysis.series.to_numpy(dtype=float))
    finite = np.isfinite(y_all)
    y = y_all[finite]
    n = len(y)
    m = max(1, int(analysis.season_length))
    fm = np.asarray(forecast.mean.to_numpy(dtype=float))
    H = len(fm)
    if n == 0 or H == 0:
        raise ValueError("analysis series and forecast must be non-empty")
    k_cur, k_fut = min(m, n), min(m, H)
    current = float(np.mean(y[-k_cur:]))
    future = float(np.mean(fm[-k_fut:]))
    seas, p = _seasonal_component(analysis)
    # the analysis' seasonal pattern carried forward, scaled by how much of it the
    # forecast actually contains (0 if the chosen models forecast no seasonality)
    seas_fut = _future_seasonal(analysis, H, fm)
    seas_fut = _seasonal_share(fm, seas_fut, p) * seas_fut
    adj_cur = adj_fut = False
    if seas is not None and _seasonality_matters(analysis):
        # compare like with like: remove the seasonal effect of partial-season windows (only a
        # material one: removing ~0 would only add a "seasonally adjusted" note to plain means)
        if k_fut % p:
            eff = float(np.mean(seas_fut[-k_fut:]))
            if _material(eff, fm[-k_fut:]):
                future -= eff
                adj_fut = True
        if k_cur % p:
            eff = float(np.mean(seas[finite][-k_cur:]))
            if _material(eff, y[-k_cur:]):
                current -= eff
                adj_cur = True
    levels = OutlookLevels(current=current, future=future, k_cur=k_cur, k_fut=k_fut,
                           period=p if seas is not None else None, adj_cur=adj_cur,
                           adj_fut=adj_fut,
                           undefined=_pct_undefined(y, current, future, k_cur, fm[-k_fut:],
                                                    bool(getattr(forecast, "nonnegative", False))),
                           scale=_pct_reference(y)[0])
    return levels, seas_fut


def outlook_levels(analysis: AnalysisResult, forecast: ForecastResult) -> OutlookLevels:
    """The current and future levels behind ``Outlook.expected_change_pct`` / ``prob_increase``.

    Exactly the levels, windows, partial-season adjustments and percentage rule
    of :func:`compute_outlook` (which uses this function), so a narrative can
    name and print them without re-implementing the rules.
    ``expected_change_pct == (future - current) / |current| * 100`` whenever
    ``undefined`` is ``None``; otherwise it is ``None`` for the reason given.
    """
    return _levels(analysis, forecast)[0]


#: |future - current| must exceed this many σ_H for a direction to be stated: Φ⁻¹(2/3), i.e.
#: prob_increase outside [1/3, 2/3] -- the change must be 'likely' (odds of at least 2:1).
#: A random-walk forecast whose change only comes from the last value sitting below the
#: recent mean is 'flat', not 'down'
_DIRECTION_Z = 0.4307


def compute_outlook(analysis: AnalysisResult, forecast: ForecastResult) -> Outlook:
    """Interpret ``forecast`` relative to the present state in ``analysis``.

    Levels (shared by ``expected_change_pct``, ``prob_increase`` and
    ``direction``):

    * **current level** = mean of the last ``k_cur = min(m, n)``
      observations (``m`` = ``analysis.season_length``, so one full season:
      the last 12 months, 52 weeks, 7 days, 4 quarters; the last value for
      yearly data);
    * **future level** = mean of the last ``k_fut = min(m, horizon)``
      forecast means, i.e. the last season *of the forecast horizon* (with
      the default horizons: the 12 months / 4 quarters / last 7 of 30 days
      ending at the horizon end; for weekly data the 26 weeks of the default
      horizon; the horizon-end value for yearly data);
    * **partial-season adjustment**: when such a window does not cover a
      whole number of seasons (e.g. weekly data, 26 < 52), the analysis'
      seasonal component averaged over that window is subtracted from it --
      for the forecast window scaled by the share of the seasonal pattern the
      forecast actually contains (0 for a flat / straight-line forecast) --
      so that a seasonal high or low is not reported as a trend change.  For
      a multiplicative decomposition the continued seasonal component scales
      with the forecast level (see :func:`_future_seasonal`).  Nothing is
      removed -- and ``OutlookLevels.adj_cur`` / ``adj_fut`` stay ``False`` --
      when the analysis found negligible seasonality
      (``metrics.seasonal_strength < 0.2``) or the window's seasonal effect is
      below 0.5 % of its mean absolute level.  For windows of whole seasons
      (the default for D/M/Q/Y) nothing changes and the two levels are plain
      means.

    Fields:

    * ``expected_change_pct`` = (future level − current level) / |current
      level| × 100: the change of the *seasonally comparable level* from the
      last season to the last season of the horizon -- not the change of the
      horizon-end value (that is ``scenarios['base']``, a single date that
      carries its own seasonal effect).  ``None`` when the analysis'
      percentage rule refuses the change (:func:`analysis.pct_change_undefined`:
      a current level ~0, a negative current level, a level crossing zero, a
      level within 0.5 standard deviations of 0 for data with negative values)
      when at least half of the current window is exactly 0 for a
      non-negative series (intermittent demand: a percentage of a mean of
      mostly zeros is meaningless), and when at least half of the future
      window sits at the 0 floor of a nonnegative forecast ('floor': a mean
      clipped at 0 is no expected value, its "−100 %" only says that the
      forecast hit the floor); see :func:`_pct_undefined`, and
      :func:`outlook_levels` gives the reason.  The direction, the
      probability and the phase are still decided on the levels.
      Intermittent demand whose current window is less than half zeros (it
      has ``>= 30 %`` zeros overall and recently) keeps a percentage: the two
      levels are then average demand rates (demand per period, zeros
      included) over a season each.
    * ``prob_increase`` = P(future level > current level) =
      ``Φ((future − current) / σ_H)``, σ_H recovered from the 80 % interval
      at the horizon end (the wider side when a bound was clipped at 0).
      σ_H is the uncertainty of one horizon-end value, which is at least that
      of the season-mean, so the probability is conservative (closer to 0.5).
      It is consistent in sign with ``expected_change_pct`` and is NOT the
      probability that the horizon-end value exceeds the current level.
    * ``direction`` is ``'flat'`` unless ``|future − current|`` exceeds
      ``max(2 % of the current level, half the 80 % half-width at h = 1,
      0.4307·σ_H)`` -- the last term is ``prob_increase`` outside
      ``[1/3, 2/3]`` (the change is 'likely', odds >= 2:1), so a change
      smaller than the horizon uncertainty (e.g. a random walk's last value
      sitting below its recent mean) is not called a trend.
    * ``scenarios`` = lo80 / mean / hi80 at the horizon end (single values
      at that date).
    * ``forecast_peak_date``: date of the forecast maximum when it lies
      strictly inside the horizon, the forecast later falls ≥ 5 % below it and
      it rose to it by more than the noise threshold used for ``direction``
      (checked on the forecast and on its seasonally adjusted version).
    * ``future_phase`` from the current lifecycle phase + direction (+ size of
      the change); ``'volatile'`` is kept, and a ``'flat'`` outlook keeps the
      current phase unless the base case itself moves by at most 2 % with a
      confidence above ``'low'`` and an 80 % half-width of at most 25 % of the
      level (see :func:`_future_phase`); ``confidence`` from
      ``forecast.skill``, the sample size and the relative width of the 80 %
      band (see :func:`_confidence`).
    """
    lv_, seas_fut = _levels(analysis, forecast)
    y = np.asarray(analysis.series.to_numpy(dtype=float))
    y = y[np.isfinite(y)]
    n = len(y)
    m = max(1, int(analysis.season_length))
    fm = np.asarray(forecast.mean.to_numpy(dtype=float))
    current, future = lv_.current, lv_.future
    diff = future - current
    expected_change_pct: Optional[float] = None
    if lv_.undefined is None:
        expected_change_pct = float(diff / abs(current) * 100.0)

    # uncertainty from the (closest to) 80 % interval
    lv = _interval_level(forecast, 80)
    sigma_h = sigma_1 = 0.0
    if lv is not None:
        z = _level_z(lv)
        lo = forecast.lower[lv].to_numpy(dtype=float)
        hi = forecast.upper[lv].to_numpy(dtype=float)
        sigma_h = max(hi[-1] - fm[-1], fm[-1] - lo[-1], 0.0) / z
        sigma_1 = max(hi[0] - fm[0], fm[0] - lo[0], 0.0) / z
        scenarios = {"pessimistic": float(lo[-1]), "base": float(fm[-1]),
                     "optimistic": float(hi[-1])}
    else:
        scenarios = {"pessimistic": float(fm[-1]), "base": float(fm[-1]),
                     "optimistic": float(fm[-1])}

    if sigma_h > 0 and math.isfinite(sigma_h):
        prob_increase = norm_cdf(diff / sigma_h)
    else:
        prob_increase = 1.0 if diff > 0 else (0.0 if diff < 0 else 0.5)
    prob_increase = float(min(1.0, max(0.0, prob_increase)))

    # direction: change must exceed the noise level and the horizon uncertainty
    half80_h1 = _level_z(80) * sigma_1
    noise_thr = max(0.02 * abs(current), 0.5 * half80_h1)
    sig_h = sigma_h if math.isfinite(sigma_h) else 0.0
    threshold = max(noise_thr, _DIRECTION_Z * sig_h)
    if abs(diff) <= threshold:
        direction = "flat"
    else:
        direction = "up" if diff > 0 else "down"
    strong = abs(diff) >= max(0.10 * abs(current), 2.0 * noise_thr)

    # peak strictly inside the horizon, reached by a rise above the noise level and
    # followed by a >= 5 % fall -- of the forecast itself *and* of its seasonally
    # adjusted, lightly smoothed version (a seasonal high, or a wiggle left over
    # from noisy seasonal estimates -- e.g. weekly data with 52 phases -- is no peak)
    peak_date: Optional[pd.Timestamp] = None
    adjusted = _smooth(fm - seas_fut, 2 * (m // 8) + 1)
    if (_has_inner_peak(fm, noise_thr, current)
            and _has_inner_peak(adjusted, noise_thr, current)):
        peak_date = pd.Timestamp(forecast.mean.index[int(np.argmax(fm))])

    rel_width: Optional[float] = None
    if lv is not None and np.all(y > 0):
        level_ref = max(abs(current), abs(float(fm[-1])))
        if level_ref > 0:
            rel_width = (_level_z(80) * sigma_h) / level_ref
    confidence = _confidence(getattr(forecast, "skill", None), forecast.backtest_mase, n, m,
                             rel_width)
    # a flat forecast projects the end of the current move only when it is flat in itself and
    # precise enough to tell a continuing move from a stop
    settled = (abs(diff) <= _FLAT_POINT_FRAC * abs(current) and confidence != "low"
               and (rel_width is None or rel_width <= _CONF_WIDTH_MEDIUM))
    future_phase = _future_phase(analysis.phase.label, direction, strong, expected_change_pct,
                                 confidence, settled)
    return Outlook(direction=direction, expected_change_pct=expected_change_pct,
                   prob_increase=prob_increase, scenarios=scenarios,
                   forecast_peak_date=peak_date, future_phase=future_phase,
                   confidence=confidence)
