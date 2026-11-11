"""Past & present: understand where a trend has been and where it is now.

This module turns a regular series (see :mod:`trendai.data`) into an
:class:`~trendai.types.AnalysisResult`:

* :func:`decompose` -- robust, STL-like additive decomposition
  ``series = trend + seasonal + resid`` with no NaNs anywhere (edges included),
* :func:`mann_kendall` / :func:`theil_sen_slope` -- non-parametric trend test
  and robust slope,
* :func:`detect_changepoints` -- structural breaks via optimal piecewise-linear
  segmentation (dynamic programming + BIC-like penalty),
* :func:`detect_anomalies` -- robust z-scores of the residual,
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
]

_TINY = 1e-12

# Korean unit words used in explanations ("최근 12개월", "최근 8주", ...).
_UNIT_KO: dict[str, str] = {"D": "일", "W": "주", "M": "개월", "Q": "분기", "Y": "년"}

_KIND_KO: dict[str, str] = {
    "acceleration": "가속",
    "deceleration": "감속",
    "reversal_up": "상승 전환",
    "reversal_down": "하락 전환",
    "level_shift": "수준 변화",
}

# The lifecycle rules use these (deliberately round) thresholds.  Relative sizes
# are fractions of the series' trend scale (max |trend|).
_MIN_SHORT_N = 8          # below this many observations: no lifecycle verdict
_MOVE = 0.15              # a "meaningful" move is >= 15% of the trend scale
_FAR_BELOW_PEAK = 0.30    # >= 30% below the trend peak counts as "well below"
_STALL_RATIO = 0.35       # recent slope < 35% of the best past slope = stalled
_EMERGING_MULTIPLE = 2.5  # now >= 2.5x the long-run base level = small base
_VOLATILE_CV = 0.20       # unstructured series with CV >= 20% is "volatile"
_T_SIG = 2.5              # |t| of the recent slope needed to call it a real move


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


def _pct_change(new: float, base: float, ref: float) -> Optional[float]:
    """``(new - base) / base * 100`` or ``None`` if ``base`` is <= 0 or ~0.

    "~0" means at most 1% of ``ref`` (a typical absolute level of the series);
    the test is relative, so it works in any units.
    """
    thr = 0.01 * abs(ref) if math.isfinite(ref) else 0.0
    if not (math.isfinite(new) and math.isfinite(base)) or base <= thr or base <= 0.0:
        return None
    v = (new - base) / base * 100.0
    if not math.isfinite(v):
        return None
    return 0.0 if abs(v) < 1e-9 else float(v)


def _fmt_date(ts: pd.Timestamp, freq: str) -> str:
    ts = pd.Timestamp(ts)
    if freq == "M":
        return ts.strftime("%Y-%m")
    if freq == "Q":
        return f"{ts.year}-Q{(ts.month - 1) // 3 + 1}"
    if freq == "Y":
        return f"{ts.year}"
    return ts.strftime("%Y-%m-%d")


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


def _loess(y: np.ndarray, window: int, weights: Optional[np.ndarray] = None) -> np.ndarray:
    """Local-linear LOESS with tricube kernel over ``window`` nearest points.

    At the edges the window becomes one-sided, i.e. the trend is extrapolated
    with a local linear fit -- no NaNs anywhere.  As in STL, the local fit is
    evaluated at every ``max(1, window // 10)``-th point (and both ends) and
    linearly interpolated in between, which keeps long series fast.
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n <= 2:
        return y.copy()
    q = int(max(2, min(window, n)))
    w_all = np.ones(n) if weights is None else np.asarray(weights, dtype=float)
    jump = max(1, q // 10)
    at = np.arange(n) if jump == 1 else np.unique(np.r_[np.arange(0, n, jump), n - 1])
    out = np.empty(len(at))
    offs = np.arange(q)
    chunk = max(1, 2_000_000 // q)
    for start in range(0, len(at), chunk):
        pos = np.arange(start, min(len(at), start + chunk))
        i = at[pos]
        lo = np.clip(i - (q - 1) // 2, 0, n - q)
        idx = lo[:, None] + offs[None, :]
        dx = (idx - i[:, None]).astype(float)
        d = np.abs(dx)
        h = d.max(axis=1, keepdims=True) + 1.0
        k = (1.0 - (d / h) ** 3) ** 3 * w_all[idx]
        yy = y[idx]
        s0 = k.sum(axis=1)
        s1 = (k * dx).sum(axis=1)
        s2 = (k * dx * dx).sum(axis=1)
        t0 = (k * yy).sum(axis=1)
        t1 = (k * dx * yy).sum(axis=1)
        det = s0 * s2 - s1 * s1
        ok = (det > 1e-10 * np.maximum(s0 * s2, _TINY)) & (s0 > _TINY)
        with np.errstate(divide="ignore", invalid="ignore"):
            fit = np.where(ok, (s2 * t0 - s1 * t1) / np.where(ok, det, 1.0),
                           t0 / np.where(s0 > _TINY, s0, 1.0))
        bad = ~np.isfinite(fit) | (s0 <= _TINY)
        if bad.any():  # all weights vanished: plain mean of the window
            fit[bad] = yy[bad].mean(axis=1)
        out[pos] = fit
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
    """Centred (2x)p moving average, edges extended by local linear fits."""
    n = len(y)
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

    The pattern is shrunk towards zero by a positive-part James-Stein factor
    whose noise estimate is the cycle-to-cycle disagreement of each phase.  A
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
            shrink = max(0.0, 1.0 - (p - 3) / p * var_mean / energy)
            pat = pat * shrink
    return pat


def decompose(series: pd.Series, period: int) -> Decomposition:
    """Robust additive decomposition ``series = trend + seasonal + resid``.

    STL-like: the seasonal component is a per-phase robust average of the
    detrended series (a fixed pattern centred to sum to 0 over each period,
    shrunk towards 0 when the cycles disagree -- see
    :func:`_seasonal_pattern`); the trend is a local-linear LOESS of the
    deseasonalised series (window ~n/10, at least min(1.5*period, n/8)).  Two
    bisquare robustness iterations down-weight outliers.  The trend is
    extrapolated at both ends with one-sided local-linear fits, so there are
    no NaNs anywhere.

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
    # work in units of max|y| so that the numeric tolerances are scale-free
    c = _unit(y)
    yn = y / c
    wide = _heavy_ties(yn)        # zero-inflated / step data: lenient robustness

    if not use_season:
        trend = _loess(yn, window)
        for _ in range(2):
            w = _bisquare(yn - trend, wide)
            trend = _loess(yn, window, w)
        seasonal = np.zeros(n)
        p_used = 1
    else:
        trend = _initial_trend(yn, p)
        w = np.ones(n)
        seasonal = np.zeros(n)
        for outer in range(3):
            for _ in range(2):
                pattern = _seasonal_pattern(yn - trend, p, w, median=(outer == 0))
                seasonal = pattern[np.arange(n) % p]
                trend = _loess(yn - seasonal, window, w)
            if outer < 2:
                w = _bisquare(yn - trend - seasonal, wide)
        p_used = p

    trend = np.where(np.isfinite(trend), trend, yn) * c
    seasonal = np.where(np.isfinite(seasonal), seasonal, 0.0) * c
    resid = y - trend - seasonal
    name = series.name
    return Decomposition(
        trend=pd.Series(trend, index=idx, name=name),
        seasonal=pd.Series(seasonal, index=idx, name=name),
        resid=pd.Series(resid, index=idx, name=name),
        period=int(p_used),
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


def mann_kendall(y: np.ndarray) -> tuple[float, float]:
    """Mann-Kendall trend test with tie correction.

    Returns ``(tau, p)`` where ``tau = S / (n(n-1)/2)`` (Kendall's tau-a
    against time) and ``p`` is the two-sided p-value of the normal
    approximation with continuity correction.  Degenerate input (n < 3 or all
    values tied) gives ``(0.0, 1.0)``.
    """
    y = np.asarray(y, dtype=float).ravel()
    y = y[np.isfinite(y)]
    n = len(y)
    if n < 3:
        return 0.0, 1.0
    s = _kendall_s(y)
    _, counts = np.unique(y, return_counts=True)
    counts = counts[counts > 1].astype(float)
    tie = float(np.sum(counts * (counts - 1) * (2 * counts + 5)))
    var = (n * (n - 1) * (2 * n + 5) - tie) / 18.0
    tau = s / (n * (n - 1) / 2.0)
    if var <= 0:
        return _finite(tau), 1.0
    if s > 0:
        z = (s - 1) / math.sqrt(var)
    elif s < 0:
        z = (s + 1) / math.sqrt(var)
    else:
        z = 0.0
    p = math.erfc(abs(z) / math.sqrt(2.0))
    return _finite(tau), float(min(1.0, max(0.0, _finite(p, 1.0))))


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
    not considered.  ``abs_floor`` is a noise-variance floor in the units of
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
    # zs has unit variance: ignore improvements below 0.1% noise (or abs_floor)
    floor = n * max(1e-6, abs_floor / (sd * sd))
    best_k, best_crit = 0, math.inf
    for k, r in enumerate(rss):
        if not math.isfinite(r):
            continue
        dof = n - (3 * k + 2) - max(0, int(extra_dof))
        if dof < 1:
            continue
        crit = n * math.log(max(r, floor) / dof) + penalty_factor * k * math.log(n)
        if crit < best_crit - 1e-9:
            best_k, best_crit = k, crit
    breaks: list[int] = []
    end = n
    for k in range(best_k, 0, -1):
        start = int(backs[k - 1][end])
        breaks.append(start)
        end = start
    return sorted(breaks)


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


def detect_anomalies(series: pd.Series, decomposition: Decomposition,
                     threshold: float = 3.5) -> list[Anomaly]:
    """Points whose residual has a robust z-score with ``|z| >= threshold``.

    z = (resid - median) / robust scale, where the scale is the MAD-based SD.
    For longer series (n >= 30) a *local* MAD (rolling window of
    ``max(25, 2 x trend window)`` points) is used, floored at a quarter of the
    global value, so that series whose noise grows with their level (page
    views, sales: multiplicative noise) do not flag every point of a busy
    period.  The scale never collapses onto a point mass: it is at least
    ``0.5 * q90 / 1.645`` of the absolute deviations (the full ``q90 / 1.645``
    for heavily tied data such as zero-inflated counts, where otherwise every
    ordinary non-zero count would be flagged).  Sorted by date.
    """
    y = _clean_values(series)
    r = decomposition.resid.to_numpy(dtype=float)
    n = min(len(y), len(r))
    if n < 3:
        return []
    y, r = y[:n], r[:n]
    r = np.where(np.isfinite(r), r, 0.0)
    c = _unit(y, r)                       # scale-free tolerances
    rn = r / c
    med = float(np.median(rn))
    sd = _robust_sd(rn)
    # point-mass guard (see _robust_sd): scale >= q_coef * q90(|dev|).  For
    # heavily tied data (zero-inflated counts) the full Gaussian-consistent
    # q90 / 1.645 is used, otherwise every ordinary non-zero count is flagged.
    q_coef = 0.608 if _heavy_ties(y) else 0.304
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
        local = np.where(np.isfinite(local), local, sd)
        scale = np.maximum(local, 0.25 * sd)
    zs = (rn - med) / scale
    expected = (decomposition.trend.to_numpy(dtype=float)[:n]
                + decomposition.seasonal.to_numpy(dtype=float)[:n])
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


def compute_metrics(series: pd.Series, freq: str, decomposition: Decomposition) -> TrendMetrics:
    """Descriptive statistics of the past and present (see :class:`TrendMetrics`).

    Growth numbers use the trend component; ``None`` when the base is <= 0 or
    ~0 (below 1% of the mean absolute level).  ``cagr_pct`` additionally
    needs a span of about a year (>= 0.95 years).  ``momentum`` is the recent
    trend slope (last ``max(3, min(season_length, n//4))`` periods) in % of the
    mean level per period minus ``slope_pct``.  ``acceleration`` is the change of
    that %-slope between the previous and the latest window.  ``slope``,
    ``mk_*`` and ``direction`` are computed on the deseasonalised series.
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

    # growth numbers (trend based)
    total = _pct_change(trend[-1], trend[0], abs_ref) if n >= 2 else None
    cagr: Optional[float] = None
    years = (n - 1) / _data.periods_per_year(freq)
    thr = 0.01 * abs_ref
    if years >= 0.95 and trend[0] > thr and trend[-1] > thr and trend[0] > 0 and trend[-1] > 0:
        lg = math.log(trend[-1] / trend[0]) / years
        if lg < 50:
            cagr = _finite((math.exp(lg) - 1.0) * 100.0, default=float("nan"))
            if not math.isfinite(cagr):
                cagr = None
            elif abs(cagr) < 1e-9:
                cagr = 0.0
    lag = 1 if freq == "Y" else _data.season_length(freq)
    lag = min(lag, n - 1)
    recent = _pct_change(trend[-1], trend[-1 - lag], abs_ref) if lag >= 1 else None

    slope_n = theil_sen_slope(z)
    slope = slope_n * c
    slope_pct = slope_n / level * 100.0

    r = _recent_window(n, freq)
    if n >= 2:
        _, s_recent = _ols(trend[-(r + 1):])
    else:
        s_recent = 0.0
    momentum = s_recent / level * 100.0 - slope_pct
    if n >= 2 * r + 1:
        _, s_prev = _ols(trend[-(2 * r + 1):-r])
        accel = (s_recent - s_prev) / level * 100.0
    else:
        accel = 0.0

    vol = float(np.std(resid)) / level if n >= 2 else 0.0
    var_r = float(np.var(resid))
    var_tr = float(np.var(trend + resid))
    var_sr = float(np.var(seas + resid))
    t_strength = max(0.0, 1.0 - var_r / var_tr) if var_tr > _TINY else 0.0
    s_strength = (max(0.0, 1.0 - var_r / var_sr)
                  if (var_sr > _TINY and decomposition.period > 1) else 0.0)

    tau, p = mann_kendall(z)
    direction = "flat"
    if p < 0.05:
        direction = "up" if tau > 0 else ("down" if tau < 0 else "flat")

    i_peak = int(np.argmax(y))
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

    h = int(min(n - 1, max(3, min(6, n // 3), round(0.15 * n))))
    xs = np.arange(h, dtype=float)
    icpt, s_rec = _ols(zc[-h:], xs)
    sxx = float(np.sum((xs - xs.mean()) ** 2))
    se_rec = noise_sd / math.sqrt(sxx) if (sxx > _TINY and noise_sd > _TINY) else math.inf
    t_rec = s_rec / se_rec if math.isfinite(se_rec) and se_rec > 0 else (
        math.copysign(99.0, s_rec) if abs(s_rec) > _TINY else 0.0)
    chg_rec = s_rec * (h - 1)
    start_level = icpt

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
    multiple = t_last / base if (base > 0.01 * ref and t_last > 0) else 0.0

    lvl = float(np.mean(np.abs(z)))
    # noise relative to the level ("coefficient of variation" of the noise)
    cv = noise_sd / lvl if lvl > _TINY else (0.0 if noise_sd <= _TINY else 1e6)

    # variance ratio: trend variance vs. what smoothed white noise would give
    var_t = float(np.var(T))
    if noise_sd > _TINY:
        f_ratio = var_t / (v * noise_sd ** 2)
    else:
        f_ratio = 1e6 if var_t > _TINY * max(ref, 1.0) ** 2 else 0.0

    return dict(
        unit=c, f_ratio=float(min(f_ratio, 1e6)), noise_sd=noise_sd,
        n=n, h=h, sigma=sigma, ref=ref, s_rec=s_rec, t_rec=float(t_rec), chg_rec=chg_rec,
        rel_rec=chg_rec / ref, start_level=start_level,
        i_max=i_max, t_max=t_max, t_last=t_last, t_min=t_min, amp=amp,
        rel_amp=amp / ref, t_low=t_low,
        drop_from_peak=drop_from_peak, rise_to_peak=rise_to_peak,
        i_tr=i_tr, t_tr=t_tr, decline_before_trough=decline_before_trough, rebound=rebound,
        settled=settled, s_max_up=s_max_up, s_rec_t=s_rec_t, stall_ratio=stall_ratio,
        avg_slope=avg_slope, base=base, multiple=multiple, cv=cv,
        g_tot=(t_last - float(T[0])) / ref,
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
        noise_cv = float(np.sqrt(np.mean(res * res))) / lvl if lvl > _TINY else 0.0
        label = "volatile" if noise_cv >= _VOLATILE_CV else "stable"
        reasons = [f"관측치가 {n}개뿐이라 추세를 판단하기에 데이터가 부족합니다."]
        pct = _pct_change(float(yn[-1]), float(yn[0]), lvl) if len(yn) >= 2 else None
        if pct is not None:
            reasons.append(f"관측 기간 동안 값이 {y[0]:,.4g}에서 {y[-1]:,.4g}로 {pct:+.1f}% 변했습니다.")
        else:
            reasons.append(f"현재 값은 {metrics.last_value:,.4g}입니다.")
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

    peak_date = _fmt_date(idx[f["i_max"]], freq)
    trough_date = _fmt_date(idx[f["i_tr"]], freq)

    def recent_sentence() -> str:
        pct = _pct_change(f["start_level"] + f["chg_rec"], f["start_level"], f["ref"])
        if pct is not None:
            return f"최근 {h}{unit} 동안 추세가 {pct:+.1f}% 변했습니다."
        return f"최근 {h}{unit} 동안 추세 기울기는 {_finite(f['s_rec'] * f['unit']):+.3g}/{unit}입니다."

    def peak_sentence() -> str:
        pct = _pct_change(f["t_last"], f["t_max"], f["ref"])
        if pct is not None:
            return f"현재 추세는 고점({peak_date})보다 {abs(pct):.1f}% 낮은 수준입니다."
        return (f"추세 고점({peak_date}) 대비 현재 수준이 추세 규모(최대 절댓값)의 "
                f"{f['drop_from_peak'] * 100:.1f}%만큼 낮습니다.")

    def rise_sentence() -> str:
        pct = _pct_change(f["t_max"], f["t_low"], f["ref"])
        if pct is not None:
            return f"추세는 저점 대비 {pct:+.0f}% 오른 뒤 {peak_date}에 고점을 형성했습니다."
        return (f"추세는 저점 대비 추세 규모의 {f['rise_to_peak'] * 100:.0f}%만큼 오른 뒤 "
                f"{peak_date}에 고점을 형성했습니다.")

    def mk_sentence() -> str:
        if metrics.mk_pvalue < 0.05:
            word = "상승" if metrics.mk_tau > 0 else "하락"
            return (f"Mann-Kendall 검정에서 전체 기간의 {word} 추세가 유의합니다 "
                    f"(tau={metrics.mk_tau:.2f}, p={metrics.mk_pvalue:.3g}).")
        return (f"Mann-Kendall 검정에서 전체 기간의 뚜렷한 추세가 확인되지 않습니다 "
                f"(p={metrics.mk_pvalue:.2f}).")

    label: str
    strength: float
    reasons: list[str]

    if not structured:
        if f["cv"] >= _VOLATILE_CV:
            label = "volatile"
            strength = 0.5 * _ramp(f["cv"], _VOLATILE_CV, 2 * _VOLATILE_CV) + 0.5 * _ramp(
                metrics.mk_pvalue, 0.01, 0.3)
            reasons = [
                f"잡음이 커서(변동계수 {f['cv'] * 100:.0f}%) 뚜렷한 방향성이 없습니다.",
                mk_sentence(),
            ]
        else:
            label = "stable"
            strength = 0.5 * _ramp(metrics.mk_pvalue, 0.01, 0.3) + 0.5 * _ramp(
                _VOLATILE_CV - f["cv"], 0.0, _VOLATILE_CV)
            reasons = [mk_sentence(),
                       f"추세의 변동 폭(최고-최저)이 최고 수준 대비 {f['rel_amp'] * 100:.1f}%에 그칩니다."]
            if metrics.seasonal_strength >= 0.5:
                reasons.append(f"계절성 강도가 {metrics.seasonal_strength:.2f}로 "
                               f"주기적 변동이 뚜렷합니다.")
    elif (f["decline_before_trough"] >= 0.20 and f["i_tr"] <= n - 1 - max(2, h // 3)
          and f["rebound"] >= 0.10 and f["rebound"] >= 0.25 * f["decline_before_trough"]
          and rising and f["drop_from_peak"] >= 0.05):
        label = "reviving"
        strength = 0.5 * _ramp(f["rebound"] / max(f["decline_before_trough"], _TINY), 0.25, 0.6) \
            + 0.5 * _ramp(f["t_rec"], _T_SIG, 3 * _T_SIG)
        reb_pct = _pct_change(f["t_last"], f["t_tr"], f["ref"])
        reb_txt = (f"{reb_pct:+.1f}%" if reb_pct is not None
                   else f"추세 규모의 {f['rebound'] * 100:.1f}%만큼")
        reasons = [
            f"고점({peak_date}) 이후 하락해 {trough_date}에 저점을 찍은 뒤 {reb_txt} 반등했습니다.",
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
            if f["settled"] >= 0.25 * n:
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
            reasons = [
                f"현재 추세 수준이 과거 기반 수준의 {f['multiple']:.1f}배로, 낮은 기반에서 빠르게 성장 중입니다.",
                recent_sentence(),
                f"최근 기울기가 장기 평균 기울기의 {f['s_rec_t'] / max(f['avg_slope'], _TINY):.1f}배로 가속되고 있습니다.",
            ]
        else:
            label = "growing"
            strength = 0.5 * _ramp(f["t_rec"], _T_SIG, 3 * _T_SIG) + 0.5 * _ramp(
                f["stall_ratio"], _STALL_RATIO, 0.8)
            reasons = [recent_sentence()]
            if mk_up:
                reasons.append(mk_sentence())
            reasons.append(f"현재 수준은 추세 고점의 {100 * (1 - f['drop_from_peak']):.0f}% 수준입니다.")
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
            reasons.append(f"잡음이 커서(변동계수 {f['cv'] * 100:.0f}%) 방향성이 불분명합니다.")

    # a recent structural break supports the explanation
    recent_cps = [c for c in changepoints if c.index >= n - max(h, 2 * _data.season_length(freq))]
    if recent_cps and len(reasons) < 4:
        c = recent_cps[-1]
        reasons.append(f"{_fmt_date(c.date, freq)}에 구조 변화({_KIND_KO.get(c.kind, c.kind)})가 감지되었습니다.")
    if len(reasons) < 2:
        reasons.append(mk_sentence())
    reasons = reasons[:4]

    confidence = (0.45 + 0.5 * float(min(1.0, max(0.0, strength)))) * length_factor
    return LifecyclePhase(label=label, confidence=float(min(1.0, max(0.0, confidence))),
                          reasons=reasons)


# =========================================================================== orchestration


def analyze(series: pd.Series, freq: Optional[str] = None,
            name: Optional[str] = None) -> AnalysisResult:
    """Full past -> present analysis of a (possibly raw) series.

    The input is cleaned/regularised with :func:`trendai.data.prepare`, then
    decomposed; changepoints, anomalies, metrics and the lifecycle phase are
    computed.  Works for as few as 3 observations (degrading gracefully).
    """
    s, freq = _data.prepare(series, freq=freq, name=name)
    n = len(s)
    sl = _data.season_length(freq)
    dec = decompose(s, sl)
    z = (dec.trend + dec.resid).to_numpy(dtype=float)
    cps = _changepoints_from(z, pd.DatetimeIndex(s.index), sl, 5, None) if n >= 8 else []
    anomalies = detect_anomalies(s, dec) if n >= 8 else []
    metrics = compute_metrics(s, freq, dec)
    phase = lifecycle_phase(s, freq, dec, metrics, cps)
    nm = name if name is not None else (str(s.name) if s.name is not None else "series")
    return AnalysisResult(name=nm, series=s, freq=freq, season_length=int(sl),
                          decomposition=dec, metrics=metrics, changepoints=cps,
                          anomalies=anomalies, phase=phase)
