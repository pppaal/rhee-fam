"""Forecasting models.

All models share the tiny :class:`Forecaster` interface::

    m = make_model("holt_damped").fit(y, season_length=12)
    yhat = m.predict(h=6)          # np.ndarray of length 6

``y`` is a 1-D float ``np.ndarray`` with no NaNs (regular spacing assumed).
Models must never raise on short or constant input once ``len(y) >= min_obs``:
they degrade gracefully (e.g. drop seasonality) instead.  Predictions must be
finite.

The registry maps a name to a zero-argument factory so the backtester can
create fresh, unfitted instances for every rolling window.

Registered models
-----------------
=====================  ==========================================================
``naive``              repeat the last value
``seasonal_naive``     repeat the last season
``drift``              random walk with drift (first -> last line)
``ses``                simple exponential smoothing (optimised alpha) *
``holt``               Holt's linear trend *
``holt_damped``        damped-trend Holt (phi in [0.8, 0.98]) *
``holt_winters``       additive Holt-Winters with damped trend (``needs_season``)
``theta``              classical Theta method (SES + half the OLS drift) *
``linear_trend``       OLS on time (+ Fourier terms) over the recent part of the data
``ar``                 AR(p) on first differences, p by AIC, forced stationary
``lag_ridge``          ridge regression on lagged differences (ML), recursive
``analog_knn``         k nearest historical windows ("past analogues")
``logistic_growth``    S-curve fit, observed transitions only, capped saturation (``requires_positive``)
``*@log``              ``holt_damped`` / ``holt_winters`` / ``linear_trend`` in log space (growth-guarded)
``drift@log``          drift in log space: constant growth *rate* (only when requested by name)
``sba``                Croston's method with the Syntetos-Boylan bias correction (intermittent demand only)
=====================  ==========================================================

``offered_for`` (class attribute) controls :func:`default_candidates`:
``drift@log`` is ``"explicit"`` (never offered automatically) and ``sba`` is
``"intermittent"`` (offered only for intermittent demand).

``*``: like the M4-competition benchmarks, these non-seasonal methods are
applied to the seasonally adjusted series (classical decomposition) when a
seasonality test with period ``season_length`` is positive, and the forecast
is reseasonalised.

Everything here is numpy-only, deterministic (no randomness at all) and
vectorised so that one fit + predict stays well under 50 ms (typically 1-35 ms)
for a few hundred observations: the exponential-smoothing family evaluates whole parameter
grids in a single pass over time, and long series are only partly used for
parameter search (see ``_ES_MAX_WINDOW``).
"""

from __future__ import annotations

import itertools
import math
from abc import ABC, abstractmethod
from collections.abc import Callable

import numpy as np


class Forecaster(ABC):
    """Base class: ``fit(y, season_length) -> self`` then ``predict(h) -> ndarray``."""

    name: str = "base"
    min_obs: int = 2          # minimum len(y) needed to fit
    needs_season: bool = False  # True if the model only makes sense with seasonality
    requires_positive: bool = False  # True if the model needs all y > 0 (log / multiplicative)
    #: which series :func:`default_candidates` offers the model for: ``"regular"`` (every
    #: series, the default), ``"intermittent"`` (only intermittent demand, i.e. when called
    #: with ``intermittent=True``) or ``"explicit"`` (never: only when requested by name)
    offered_for: str = "regular"

    def __init__(self) -> None:
        self._y: np.ndarray | None = None
        self.season_length: int = 1

    def fit(self, y: np.ndarray, season_length: int = 1) -> Forecaster:
        y = np.array(y, dtype=float)  # private copy: later caller mutations cannot leak in
        if y.ndim != 1:
            raise ValueError("y must be 1-D")
        if not np.all(np.isfinite(y)):
            raise ValueError("y contains non-finite values")
        if len(y) < self.min_obs:
            raise ValueError(f"{self.name} needs at least {self.min_obs} observations, got {len(y)}")
        self._y = y
        self.season_length = max(1, int(season_length))
        self._fit(y, self.season_length)
        return self

    def predict(self, h: int) -> np.ndarray:
        if self._y is None:
            raise RuntimeError("call fit() first")
        if h < 1:
            raise ValueError("h must be >= 1")
        out = np.asarray(self._predict(int(h)), dtype=float)
        if out.shape != (h,):
            raise RuntimeError(f"{self.name} returned shape {out.shape}, expected ({h},)")
        if not np.all(np.isfinite(out)):
            # Never leak NaN/inf, but also never silently disguise a broken model as
            # "naive": raise so the backtester skips it (forecast.py catches per model).
            raise FloatingPointError(f"{self.name} produced a non-finite forecast")
        return out

    @abstractmethod
    def _fit(self, y: np.ndarray, m: int) -> None: ...

    @abstractmethod
    def _predict(self, h: int) -> np.ndarray: ...

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.name}>"


# --------------------------------------------------------------------------- simple baselines


class Naive(Forecaster):
    """Repeat the last observation."""

    name = "naive"
    min_obs = 1

    def _fit(self, y: np.ndarray, m: int) -> None:
        self._last = y[-1]

    def _predict(self, h: int) -> np.ndarray:
        return np.full(h, self._last)


class SeasonalNaive(Forecaster):
    """Repeat the last full season (falls back to naive without enough data)."""

    name = "seasonal_naive"
    min_obs = 1
    needs_season = True

    def _fit(self, y: np.ndarray, m: int) -> None:
        self._m = m if (m > 1 and len(y) >= m) else 1
        self._tail = y[-self._m:]

    def _predict(self, h: int) -> np.ndarray:
        reps = int(np.ceil(h / self._m))
        return np.tile(self._tail, reps)[:h]


class Drift(Forecaster):
    """Random walk with drift: extrapolate the line from first to last value."""

    name = "drift"
    min_obs = 2

    def _fit(self, y: np.ndarray, m: int) -> None:
        self._last = y[-1]
        self._slope = (y[-1] - y[0]) / (len(y) - 1)

    def _predict(self, h: int) -> np.ndarray:
        return self._last + self._slope * np.arange(1, h + 1)


# --------------------------------------------------------------------------- shared helpers


def _standardize(y: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Return ``(z, loc, scale)`` with ``y = loc + scale * z``.

    ``scale`` is the standard deviation, or (for constant input) the largest
    absolute value (1 for an all-zero series), so it is always ``> 0`` and the
    transform is exactly invertible.  Working on ``z`` keeps the numerics of
    every model independent of the series' magnitude (1e-6 .. 1e9).  For
    extreme magnitudes (``max|y|`` above 1e150 or below 1e-150, where squares
    overflow or underflow) the moments are computed on ``y / max|y|`` so the
    standard deviation stays exact instead of overflowing to ``inf`` or
    underflowing to 0.
    """
    ref = float(np.max(np.abs(y))) if y.size else 0.0
    if 0.0 < ref and not (1e-150 < ref < 1e150):
        yr = y / ref
        loc = ref * float(np.mean(yr))
        scale = ref * float(np.std(yr))
    else:
        loc = float(np.mean(y))
        scale = float(np.std(y))
    if not np.isfinite(scale) or scale <= 1e-12 * ref or scale == 0.0:
        scale = ref if ref > 0 else 1.0
    return (y - loc) / scale, loc, scale


def _ols_line(y: np.ndarray, t: np.ndarray | None = None) -> tuple[float, float]:
    """Least-squares line ``y ~ a + b t``; returns ``(a, b)`` (intercept at ``t = 0``)."""
    n = len(y)
    if t is None:
        t = np.arange(n, dtype=float)
    if n == 0:
        return 0.0, 0.0
    if n == 1:
        return float(y[0]), 0.0
    tm = float(t.mean())
    ym = float(y.mean())
    stt = float(((t - tm) ** 2).sum())
    if stt <= 0:
        return ym, 0.0
    b = float(((t - tm) * (y - ym)).sum() / stt)
    return ym - b * tm, b


def _phase_means(r: np.ndarray, phases: np.ndarray, m: int) -> np.ndarray:
    """Per-phase mean of ``r`` (phases in ``0..m-1``), centred to sum 0."""
    cnt = np.bincount(phases, minlength=m).astype(float)
    tot = np.bincount(phases, weights=r, minlength=m)
    s = np.where(cnt > 0, tot / np.maximum(cnt, 1.0), 0.0)
    return s - s.mean()


def _bounded(z: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Map unconstrained ``z`` into ``(lo, hi)`` with a logistic function."""
    return lo + (hi - lo) / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


def _unbounded(p: np.ndarray | float, lo: float, hi: float) -> np.ndarray:
    """Inverse of :func:`_bounded`."""
    p = np.clip(np.asarray(p, dtype=float), lo + 1e-9 * (hi - lo), hi - 1e-9 * (hi - lo))
    return np.log((p - lo) / (hi - p))


def _pattern_search(objective: Callable[[np.ndarray], np.ndarray], axes: list[np.ndarray],
                    step: float = 0.5, rounds: int = 4,
                    max_iter: int | None = None) -> tuple[np.ndarray, float]:
    """Minimise a vectorised objective over a coarse grid, then refine locally.

    ``objective`` maps a ``(K, d)`` array of (unconstrained) points to ``K``
    values.  The coarse grid is the Cartesian product of ``axes``; each
    refinement iteration evaluates the full ``3**d`` neighbourhood of the
    current best point (in one vectorised call) and moves to the best
    neighbour if it improves.  During the first ``max_iter`` iterations
    (default ``2 * rounds``) the step is kept after a successful move, so the
    search can walk past the coarse grid (e.g. towards a parameter bound),
    and halved after a failed one; afterwards every iteration halves it.  The
    search ends after ``rounds`` halvings, so it always finishes on a fine
    step (at most ``max_iter + rounds`` iterations).  Deterministic: ties
    resolve to the first point.
    """
    mesh = np.meshgrid(*axes, indexing="ij")
    pts = np.stack([g.ravel() for g in mesh], axis=1)
    vals = objective(pts)
    i = int(np.argmin(vals))
    best, best_val = pts[i].copy(), float(vals[i])
    d = pts.shape[1]
    offsets = np.array(list(itertools.product((-1.0, 0.0, 1.0), repeat=d)))
    max_iter = 2 * rounds if max_iter is None else int(max_iter)
    halvings = iters = 0
    while halvings < rounds:
        cand = best[None, :] + offsets * step
        v = objective(cand)
        j = int(np.argmin(v))
        improved = float(v[j]) < best_val
        if improved:
            best, best_val = cand[j].copy(), float(v[j])
        iters += 1
        if not improved or iters >= max_iter:
            step *= 0.5
            halvings += 1
    return best, best_val


def _drift_forecast(y: np.ndarray, h: int) -> np.ndarray:
    """Random-walk-with-drift forecast (used as the universal fallback)."""
    n = len(y)
    slope = (y[-1] - y[0]) / (n - 1) if n > 1 else 0.0
    return y[-1] + slope * np.arange(1, h + 1)


# --------------------------------------------------------------------------- seasonal adjustment


def _acf(x: np.ndarray, nlags: int) -> np.ndarray:
    """Sample autocorrelations ``r_0..r_nlags`` (r_0 = 1); zeros for a constant series."""
    n = len(x)
    xc = x - x.mean()
    denom = float((xc * xc).sum())
    out = np.zeros(nlags + 1)
    out[0] = 1.0
    if denom <= 0:
        return out
    for k in range(1, min(nlags, n - 1) + 1):
        out[k] = float((xc[k:] * xc[:-k]).sum()) / denom
    return out


def _centred_ma(y: np.ndarray, m: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Centred moving average (2 x m for even ``m``).

    Returns ``(trend, seg, phases)``: the MA over its valid range, the
    matching observations and their phases ``t % m``.
    """
    if m % 2 == 0:
        w = np.r_[0.5, np.ones(m - 1), 0.5] / m
    else:
        w = np.ones(m) / m
    trend = np.convolve(y, w, mode="valid")
    off = len(w) // 2
    seg = y[off: off + len(trend)]
    ph = (np.arange(len(trend)) + off) % m
    return trend, seg, ph


def _seasonal_indices(trend: np.ndarray, seg: np.ndarray, ph: np.ndarray, m: int,
                      multiplicative: bool) -> np.ndarray:
    """Phase means of ``seg/trend`` (normalised to mean 1) or ``seg - trend`` (mean 0)."""
    if multiplicative:
        ratio = seg / np.where(trend > 0, trend, np.nan)
        ok = np.isfinite(ratio)
        cnt = np.bincount(ph[ok], minlength=m).astype(float)
        tot = np.bincount(ph[ok], weights=ratio[ok], minlength=m)
        idx = np.where(cnt > 0, tot / np.maximum(cnt, 1.0), 1.0)
        idx = np.where(idx > 0, idx, 1.0)
        return idx / idx.mean()
    return _phase_means(seg - trend, ph, m)


def _classical_seasonal(y: np.ndarray, m: int, multiplicative: bool) -> np.ndarray:
    """Seasonal indices (length ``m``, by phase ``t % m``) from classical decomposition."""
    trend, seg, ph = _centred_ma(y, m)
    return _seasonal_indices(trend, seg, ph, m, multiplicative)


def _paulson_z(F: float, df1: float, df2: float) -> float:
    """Normal approximation (Paulson 1942) of the upper tail of an F(df1, df2) statistic.

    Non-integer (effective) degrees of freedom are allowed.
    """
    a = 2.0 / (9.0 * df1)
    b = 2.0 / (9.0 * df2)
    f3 = F ** (1.0 / 3.0)
    return ((1.0 - b) * f3 - (1.0 - a)) / math.sqrt(b * f3 * f3 + a)


def _is_seasonal(y: np.ndarray, m: int) -> bool:
    """Is there a repeating pattern of period ``m``?

    With at least 2.5 seasons: the series is detrended by a centred moving
    average and the remainder must show (a) significant between-phase
    differences (one-way ANOVA F-test, 99 %, Paulson's normal approximation)
    that (b) repeat from one season to the next (positive lag-``m``
    autocorrelation of the remainder, > 0.1).  This handles slow non-linear
    trends and long seasons (weekly data, m = 52) much better than the classic
    test.

    The F-test's degrees of freedom are deflated for serial correlation of
    the noise (Satterthwaite / Bartlett effective sample size): with
    ``r1`` the lag-1 autocorrelation of the *within-phase* residual (the
    remainder minus its phase means, so a genuine seasonal pattern does not
    count as correlation), both df are divided by
    ``kappa = (1 + r1^2) / (1 - r1^2)``, the variance inflation of a sum of
    squares of AR(1) noise.  Without it the test assumed white noise, but the
    moving-average remainder of a random walk (or any persistent level) is
    smooth and its phase means are strongly correlated, so their spread is
    far more variable than ``F(m-1, N-m)`` allows: 13-20 % of pure random
    walks with m = 52 and 3-6 seasons were called seasonal (about 3 % now);
    for white-noise remainders (``r1 ~ 0``) nothing changes.

    With 2 to 2.5 seasons (too short for the above) the classic M4
    test is used: ``|r_m| > 1.645 * sqrt((1 + 2 sum_{k<m} r_k^2) / n)`` on the
    linearly detrended series.  The data are divided by ``max|y|`` first, so
    the decision does not depend on the magnitude (no overflow or underflow
    of squares for ``|y| ~ 1e300`` or ``1e-300``).
    """
    n = len(y)
    if m <= 1 or n < 2 * m:
        return False
    ref = float(np.max(np.abs(y)))
    if float(np.ptp(y)) <= 1e-12 * ref:
        return False  # constant (up to rounding)
    y = y / ref
    if 2 * n >= 5 * m:
        trend, seg, ph = _centred_ma(y, m)
        x = seg - trend
        N = len(x)
        cnt = np.bincount(ph, minlength=m).astype(float)
        P = int((cnt > 0).sum())
        df1, df2 = P - 1, N - P
        if df1 >= 1 and df2 >= 2:
            means = np.bincount(ph, weights=x, minlength=m) / np.maximum(cnt, 1.0)
            ssb = float((cnt * (means - x.mean()) ** 2).sum())
            e = x - means[ph]
            ssw = float((e * e).sum())
            if ssb <= 0:
                return False
            if ssw <= 1e-12 * ssb:
                z = np.inf
            else:
                r1 = min(max(float((e[1:] * e[:-1]).sum()) / ssw, 0.0), 0.98)
                kappa = (1.0 + r1 * r1) / (1.0 - r1 * r1)
                z = _paulson_z((ssb / df1) / (ssw / df2), max(df1 / kappa, 1.0), max(df2 / kappa, 2.0))
            return bool(z > 2.326 and _acf(x, m)[m] > 0.1)
    a, b = _ols_line(y)
    x = y - (a + b * np.arange(n))
    r = _acf(x, m)
    if not np.any(r[1:]):
        return False
    se = math.sqrt((1.0 + 2.0 * float((r[1:m] ** 2).sum())) / n)
    return bool(abs(r[m]) > 1.645 * se)


class _SeasonalAdjustment:
    """Classical seasonal adjustment used by the non-seasonal models (M4 style)."""

    def __init__(self, mode: str, idx: np.ndarray, m: int) -> None:
        self.mode = mode  # 'none' | 'mult' | 'add'
        self.idx = idx
        self.m = m

    @classmethod
    def none(cls) -> _SeasonalAdjustment:
        return cls("none", np.zeros(1), 1)

    @classmethod
    def fit(cls, y: np.ndarray, m: int, multiplicative: bool | None = None) -> _SeasonalAdjustment:
        """Detect seasonality (:func:`_is_seasonal`) and estimate the seasonal indices.

        ``multiplicative=None`` (auto): multiplicative for strictly positive
        data unless the additive decomposition leaves a smaller remainder
        (sum of squares around the centred moving average) -- additive
        seasonality on a growing level would otherwise be over-amplified at
        the end of the series.  ``False`` forces additive; ``True`` requests
        multiplicative (only honoured for positive data).
        """
        if not _is_seasonal(y, m):
            return cls.none()
        positive = bool(np.all(y > 0))
        trend, seg, ph = _centred_ma(y, m)
        if not positive or multiplicative is False:
            return cls("add", _seasonal_indices(trend, seg, ph, m, False), m)
        idx_m = _seasonal_indices(trend, seg, ph, m, True)
        if multiplicative is None:
            idx_a = _seasonal_indices(trend, seg, ph, m, False)
            ref = float(np.max(seg))  # compare in units of max(y): squares cannot overflow
            sse_m = float((((seg - trend * idx_m[ph]) / ref) ** 2).sum())
            sse_a = float((((seg - trend - idx_a[ph]) / ref) ** 2).sum())
            if sse_a < sse_m:
                return cls("add", idx_a, m)
        return cls("mult", idx_m, m)

    def adjust(self, y: np.ndarray) -> np.ndarray:
        if self.mode == "none":
            return y
        s = self.idx[np.arange(len(y)) % self.m]
        return y / s if self.mode == "mult" else y - s

    def reseasonalise(self, f: np.ndarray, n: int) -> np.ndarray:
        """Re-apply the indices to forecasts of observations ``n, n+1, ...``."""
        if self.mode == "none":
            return f
        s = self.idx[(n + np.arange(len(f))) % self.m]
        return f * s if self.mode == "mult" else f + s


# --------------------------------------------------------------------------- exponential smoothing core

#: max number of most recent observations used to *optimise* smoothing parameters
_ES_MAX_WINDOW = 300


def _ets_filter(y: np.ndarray, alpha: np.ndarray, beta: np.ndarray, gamma: np.ndarray,
                phi: np.ndarray, l0: float, b0: float, s0: np.ndarray,
                trend: bool, season: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Additive (damped) exponential smoothing in error-correction form.

    Runs ``K`` parameter sets at once (all parameter arrays have shape
    ``(K,)``)::

        yhat_t = l + phi*b + s[t % m]
        e_t    = y_t - yhat_t
        l      = l + phi*b + alpha*e_t
        b      = phi*b + beta*e_t
        s[t%m] = s[t%m] + gamma*e_t

    Returns ``(sse, level, trend, seasonals)`` with shapes ``(K,)``, ``(K,)``,
    ``(K,)`` and ``(m, K)``; the seasonal row ``j`` holds the latest estimate
    for phase ``j`` (phase of observation ``t`` is ``t % m``).
    """
    n = y.shape[0]
    K = alpha.shape[0]
    lvl = np.full(K, float(l0))
    b = np.full(K, float(b0)) if trend else np.zeros(K)
    m = int(s0.shape[0]) if season else 1
    S = np.repeat(np.asarray(s0, dtype=float)[:, None], K, axis=1) if season else np.zeros((1, K))
    sse = np.zeros(K)
    yl = y.tolist()
    with np.errstate(over="ignore", invalid="ignore"):
        if trend and season:
            for t in range(n):
                j = t % m
                pb = phi * b
                base = lvl + pb
                st = S[j]
                e = yl[t] - base - st
                sse += e * e
                lvl = base + alpha * e
                b = pb + beta * e
                S[j] = st + gamma * e
        elif trend:
            for t in range(n):
                pb = phi * b
                base = lvl + pb
                e = yl[t] - base
                sse += e * e
                lvl = base + alpha * e
                b = pb + beta * e
        elif season:
            for t in range(n):
                j = t % m
                st = S[j]
                e = yl[t] - lvl - st
                sse += e * e
                lvl = lvl + alpha * e
                S[j] = st + gamma * e
        else:
            for t in range(n):
                e = yl[t] - lvl
                sse += e * e
                lvl = lvl + alpha * e
    sse = np.where(np.isfinite(sse), sse, np.inf)
    return sse, lvl, b, S


def _ets_init(z: np.ndarray, m: int, trend: bool, season: bool) -> tuple[float, float, np.ndarray]:
    """Heuristic initial states ``(l0, b0, s0)`` (the state *before* ``z[0]``).

    Seasonal: decomposition of the first ``k >= 2`` full seasons (line +
    centred per-phase means of the residual, two passes).  Non-seasonal: a
    line through the first ``<= 10`` points.
    """
    n = len(z)
    if season:
        k = max(2, min(n // m, math.ceil(24 / m)))
        seg = z[: k * m]
        t = np.arange(len(seg), dtype=float)
        ph = np.arange(len(seg)) % m
        s = np.zeros(m)
        for _ in range(2):
            a, b = _ols_line(seg - s[ph], t)
            s = _phase_means(seg - a - b * t, ph, m)
        a, b = _ols_line(seg - s[ph], t)
    else:
        seg = z[: min(n, 10)]
        a, b = _ols_line(seg)
        s = np.zeros(max(m, 1))
    if not trend:
        return a, 0.0, s
    return a - b, b, s


def _es_fit(z: np.ndarray, m: int, trend: bool, damped: bool, season: bool) -> dict:
    """Optimise smoothing parameters by in-sample one-step SSE and filter.

    ``z`` should be standardised.  Only the last ``_ES_MAX_WINDOW`` points
    (at least 6 seasons when seasonal) are used, which keeps the cost bounded
    for long series; the returned states are at the end of ``z``.
    Parameters are searched on a coarse grid then refined by
    :func:`_pattern_search` in logit space, under the usual admissible
    parametrisation ``beta = alpha*beta*``, ``gamma = (1-alpha)*gamma*``.
    """
    n_all = len(z)
    trend = bool(trend and n_all >= 3)
    season = bool(season and m > 1 and n_all >= 2 * m)
    damped = bool(damped and trend)
    W = min(n_all, max(_ES_MAX_WINDOW, 6 * m if season else 0))
    zw = z[-W:]
    l0, b0, s0 = _ets_init(zw, m if season else 1, trend, season)

    A_LO, A_HI = 0.01, 0.99
    B_LO, B_HI = 0.001, 0.99
    P_LO, P_HI = 0.80, 0.98
    G_LO, G_HI = 0.001, 0.99

    axes: list[np.ndarray] = []
    if trend or season:
        axes.append(_unbounded(np.array([0.05, 0.15, 0.3, 0.5, 0.7, 0.9]), A_LO, A_HI))
    else:
        axes.append(_unbounded(np.array([0.02, 0.05, 0.1, 0.2, 0.3, 0.45, 0.6, 0.75, 0.9, 0.98]),
                               A_LO, A_HI))
    if trend:
        axes.append(_unbounded(np.array([0.01, 0.05, 0.15, 0.4]), B_LO, B_HI))
    if damped:
        axes.append(_unbounded(np.array([0.82, 0.9, 0.96]), P_LO, P_HI))
    if season:
        axes.append(_unbounded(np.array([0.02, 0.1, 0.3, 0.6]), G_LO, G_HI))

    def unpack(P: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        K = P.shape[0]
        c = 0
        alpha = _bounded(P[:, c], A_LO, A_HI)
        c += 1
        beta = np.zeros(K)
        phi = np.ones(K)
        gamma = np.zeros(K)
        if trend:
            beta = alpha * _bounded(P[:, c], B_LO, B_HI)
            c += 1
        if damped:
            phi = _bounded(P[:, c], P_LO, P_HI)
            c += 1
        if season:
            gamma = (1.0 - alpha) * _bounded(P[:, c], G_LO, G_HI)
            c += 1
        return alpha, beta, gamma, phi

    def objective(P: np.ndarray) -> np.ndarray:
        a, b, g, p = unpack(P)
        return _ets_filter(zw, a, b, g, p, l0, b0, s0, trend, season)[0]

    rounds = 4 if len(axes) >= 3 else 5
    if len(axes) == 1:
        rounds = 6
    best, _ = _pattern_search(objective, axes, step=0.5, rounds=rounds)
    alpha, beta, gamma, phi = unpack(best[None, :])
    sse, lvl, b, S = _ets_filter(zw, alpha, beta, gamma, phi, l0, b0, s0, trend, season)
    return {
        "alpha": float(alpha[0]), "beta": float(beta[0]), "gamma": float(gamma[0]),
        "phi": float(phi[0]), "level": float(lvl[0]), "trend": float(b[0]),
        "seasonals": S[:, 0].copy() if season else np.zeros(1),
        "n": W, "m": m if season else 1, "use_trend": trend, "use_season": season,
        "sse": float(sse[0]),
    }


def _es_forecast(st: dict, h: int) -> np.ndarray:
    """Point forecast from the state dict returned by :func:`_es_fit` (standardised units)."""
    steps = np.arange(1, h + 1, dtype=float)
    f = np.full(h, st["level"])
    if st["use_trend"]:
        phi = st["phi"]
        damp = steps if phi >= 1.0 else np.cumsum(phi ** steps)
        f = f + damp * st["trend"]
    if st["use_season"]:
        m = st["m"]
        f = f + st["seasonals"][(st["n"] + np.arange(h)) % m]
    return f


class _ExpSmoothing(Forecaster):
    """Shared implementation of the additive exponential-smoothing family.

    The non-seasonal members (``ses``, ``holt``, ``holt_damped``) follow the
    M4-competition benchmark recipe: when the seasonality test
    (:func:`_is_seasonal`) is positive, they are fitted on the seasonally
    adjusted series (classical decomposition, see
    :meth:`_SeasonalAdjustment.fit`) and the forecast is reseasonalised.  ``holt_winters`` models seasonality in its
    own states instead.
    """

    _trend = False
    _damped = False
    _seasonal = False
    #: None = auto (multiplicative for positive data unless additive fits better); False = additive
    adjust_multiplicative: bool | None = None

    def _fit(self, y: np.ndarray, m: int) -> None:
        n = len(y)
        self._n = n
        self._adj = _SeasonalAdjustment.none()
        x = y
        if not self._seasonal:
            self._adj = _SeasonalAdjustment.fit(y, m, self.adjust_multiplicative)
            x = self._adj.adjust(y)
        z, self._loc, self._scale = _standardize(x)
        self._state = _es_fit(z, m, self._trend, self._damped, self._seasonal)
        self.params_ = {k: self._state[k] for k in ("alpha", "beta", "gamma", "phi")}
        self.params_["seasonal_adjustment"] = self._adj.mode

    def _predict(self, h: int) -> np.ndarray:
        f = self._loc + self._scale * _es_forecast(self._state, h)
        return self._adj.reseasonalise(f, self._n)


class SES(_ExpSmoothing):
    """Simple exponential smoothing (flat forecast, optimised alpha)."""

    name = "ses"
    min_obs = 2


class Holt(_ExpSmoothing):
    """Holt's linear-trend method (undamped)."""

    name = "holt"
    min_obs = 3
    _trend = True


class HoltDamped(_ExpSmoothing):
    """Damped-trend Holt; phi optimised in [0.8, 0.98]."""

    name = "holt_damped"
    min_obs = 3
    _trend = True
    _damped = True


class HoltWinters(_ExpSmoothing):
    """Additive Holt-Winters with damped trend.

    Degrades to :class:`HoltDamped` behaviour when fewer than two full seasons
    are available (or ``season_length <= 1``).
    """

    name = "holt_winters"
    min_obs = 3
    needs_season = True
    _trend = True
    _damped = True
    _seasonal = True


# --------------------------------------------------------------------------- theta


class Theta(Forecaster):
    """Classical Theta method (Assimakopoulos & Nikolopoulos; Hyndman & Billah form).

    Forecast = SES forecast of the (seasonally adjusted) series plus half the
    OLS slope: ``l_n + b/2 * (h - 1 + (1 - (1-alpha)^n) / alpha)``.  When the
    seasonality test (:func:`_is_seasonal`: phase ANOVA + lag-``m`` ACF of the
    detrended series, or the classic lag-``m`` ACF test for short series) is positive
    the series is deseasonalised by classical decomposition first --
    multiplicative for positive data unless the additive decomposition leaves
    a smaller remainder, additive otherwise -- and the forecast is
    reseasonalised.
    """

    name = "theta"
    min_obs = 3
    #: None = auto (multiplicative for positive data unless additive fits better); False = additive
    adjust_multiplicative: bool | None = None

    def _fit(self, y: np.ndarray, m: int) -> None:
        n = len(y)
        self._n = n
        self._adj = _SeasonalAdjustment.fit(y, m, self.adjust_multiplicative)
        x = self._adj.adjust(y)
        z, self._loc, self._scale = _standardize(x)
        st = _es_fit(z, 1, trend=False, damped=False, season=False)
        self._alpha = st["alpha"]
        self._level = st["level"]
        self._n_ses = st["n"]
        self._slope = _ols_line(z)[1]

    def _predict(self, h: int) -> np.ndarray:
        a = self._alpha
        drift = 0.5 * self._slope * (np.arange(h) + (1.0 - (1.0 - a) ** self._n_ses) / a)
        f = self._loc + self._scale * (self._level + drift)
        return self._adj.reseasonalise(f, self._n)


# --------------------------------------------------------------------------- intermittent demand


def _ses_level(x: np.ndarray, alpha: float, k_init: int) -> float:
    """Final SES level of ``x`` (smoothing constant ``alpha``).

    The usual heuristic start: the mean of the first ``k_init`` values, then
    the filter runs through all of ``x``.
    """
    lvl = float(x[0]) + float(np.mean(x[:k_init] - x[0]))  # exact for constant x
    for v in x.tolist():
        lvl += alpha * (v - lvl)
    return lvl


class SBA(Forecaster):
    """Syntetos-Boylan approximation (SBA) of Croston's method, for intermittent demand.

    Croston: the non-zero demand sizes and the intervals between demands
    (in periods; the first one counted from the start of the series) are
    smoothed separately by SES, updated only when a demand occurs, and the
    forecast is the demand rate ``size / interval`` per period, flat over the
    horizon.  That ratio is biased upwards (``E[z/p] > E[z]/E[p]``), so SBA
    multiplies it by ``1 - alpha / 2`` (Syntetos & Boylan 2005).  A series
    without any zero has nothing to correct (every interval is 1): the factor
    is then 1 and the model is SES of the values.

    ``alpha = 0.1`` for both parts (the customary SBA constant) and each SES
    starts from the mean of its first ``k_init = 5`` values.  In simulations
    of stationary and changing demand (occurrence 15-60 %, 24-150 periods)
    this was more accurate than choosing alpha per series by in-sample SSE
    (the one-step errors of a few demands mostly measure noise) and than
    starting from the first demand alone.  An all-zero series is forecast as
    exactly 0, a constant one as that constant; exactly scale-equivariant
    (the sizes are scaled by a power of two).

    Unlike SES on the raw series, whose level jumps at every sale and then
    decays through the zeros after it, the rate only changes when a demand
    occurs, so the forecast hardly depends on how long ago the last sale
    was.  On an ordinary series it is only SES without trend, so
    :func:`default_candidates` offers it for intermittent demand only
    (``offered_for = "intermittent"``).
    """

    name = "sba"
    min_obs = 2
    offered_for = "intermittent"
    #: smoothing constant of the demand sizes and of the intervals
    alpha: float = 0.1
    #: the SES levels start from the mean of this many first sizes / intervals
    k_init: int = 5

    def _fit(self, y: np.ndarray, m: int) -> None:
        pos = np.flatnonzero(y != 0)
        self._rate = 0.0
        self.params_ = {"demands": int(pos.size)}
        if pos.size == 0:
            return
        e = math.frexp(float(np.max(np.abs(y[pos]))))[1]
        sizes = np.ldexp(y[pos], -e)  # exact: |sizes| in (0, 1)
        intervals = np.diff(np.r_[-1, pos]).astype(float)
        size = _ses_level(sizes, self.alpha, self.k_init)
        interval = _ses_level(intervals, self.alpha, self.k_init)
        factor = 1.0 if pos.size == len(y) else 1.0 - 0.5 * self.alpha
        self._rate = float(np.ldexp(factor * size / interval, e))
        self.params_.update({"size": float(np.ldexp(size, e)), "interval": interval,
                             "bias_factor": factor})

    def _predict(self, h: int) -> np.ndarray:
        return np.full(h, self._rate)


# --------------------------------------------------------------------------- linear trend


def _fourier_design(t: np.ndarray, m: int, K: int, center: float, span: float) -> np.ndarray:
    """``[1, (t-center)/span, cos/sin(2 pi k t / m) for k = 1..K]`` (sin dropped when zero)."""
    cols = [np.ones_like(t, dtype=float), (t - center) / span]
    for k in range(1, K + 1):
        ang = 2.0 * np.pi * k * t / m
        cols.append(np.cos(ang))
        if 2 * k != m:  # sin(pi t) == 0 on integer t
            cols.append(np.sin(ang))
    return np.column_stack(cols)


class LinearTrend(Forecaster):
    """OLS on ``[1, t]`` plus Fourier seasonal terms, fitted on the recent part.

    Window: the last ``max(n // 2, 2m (if seasonal), min(n, 12))`` points, so
    the line follows the *current* slope rather than the whole-history
    average (important for accelerating / decelerating trends).  Fourier terms
    (``K = min(3, m // 2)`` harmonics) are added when ``m > 1`` and the
    series has at least two full seasons.
    """

    name = "linear_trend"
    min_obs = 2

    def _fit(self, y: np.ndarray, m: int) -> None:
        n = len(y)
        self._n = n
        use_season = m > 1 and n >= 2 * m
        self._K = min(3, m // 2) if use_season else 0
        self._m = m
        L = max(n // 2, min(n, 12))
        if use_season:
            L = max(L, 2 * m)
        L = min(L, n)
        t = np.arange(n - L, n, dtype=float)
        self._center = float(t.mean())
        self._span = float(max(L - 1, 1))
        z, self._loc, self._scale = _standardize(y[-L:])
        X = _fourier_design(t, m, self._K, self._center, self._span)
        coef, *_ = np.linalg.lstsq(X, z, rcond=None)
        self._coef = np.where(np.isfinite(coef), coef, 0.0)

    def _predict(self, h: int) -> np.ndarray:
        t = np.arange(self._n, self._n + h, dtype=float)
        X = _fourier_design(t, self._m, self._K, self._center, self._span)
        return self._loc + self._scale * (X @ self._coef)


# --------------------------------------------------------------------------- AR on differences


def _ridge_solve(X: np.ndarray, y: np.ndarray, rel_lambda: float = 1e-3) -> np.ndarray:
    """Least squares with a small ridge term (relative to the mean diagonal of X'X)."""
    XtX = X.T @ X
    lam = rel_lambda * float(np.mean(np.diag(XtX))) + 1e-12
    return np.linalg.solve(XtX + lam * np.eye(X.shape[1]), X.T @ y)


def _stabilise_ar(phi: np.ndarray, max_root: float = 0.97) -> np.ndarray:
    """Shrink AR coefficients so all companion roots lie inside ``|z| <= max_root``.

    Scaling ``phi_k -> phi_k * c**k`` multiplies every root by ``c``.
    """
    if phi.size == 0:
        return phi
    roots = np.roots(np.r_[1.0, -phi])
    r = float(np.max(np.abs(roots))) if roots.size else 0.0
    if not np.isfinite(r):
        return np.zeros_like(phi)
    if r > max_root:
        c = max_root / r
        phi = phi * c ** np.arange(1, phi.size + 1)
    return phi


class AR(Forecaster):
    """AR(p) on first differences with mean drift; p by AIC; forced stationary.

    ``d_t - mu = sum_k phi_k (d_{t-k} - mu) + e_t`` with ``mu = mean(d)``;
    ``p`` in ``[1, min(12, n // 5)]`` chosen by AIC on a common sample,
    coefficients by ridge-stabilised least squares, then shrunk so the
    companion matrix has spectral radius <= 0.97.  Forecasts integrate the
    predicted differences.  Falls back to drift (p = 0) when ``n < 5``.
    """

    name = "ar"
    min_obs = 3

    def _fit(self, y: np.ndarray, m: int) -> None:
        z, self._loc, self._scale = _standardize(y)
        self._zlast = float(z[-1])
        d = np.diff(z)
        nd = len(d)
        self._mu = float(d.mean()) if nd else 0.0
        dc = d - self._mu
        pmax = min(12, len(y) // 5, (nd - 2) // 2)
        self._phi = np.zeros(0)
        self._hist = np.zeros(0)
        if pmax < 1:
            return
        best_p, best_aic = 1, np.inf
        N = nd - pmax
        for p in range(1, pmax + 1):
            X = np.column_stack([dc[pmax - k: nd - k] for k in range(1, p + 1)])
            tgt = dc[pmax:]
            coef = _ridge_solve(X, tgt)
            rss = float(((tgt - X @ coef) ** 2).sum())
            aic = N * math.log(rss / N + 1e-300) + 2.0 * (p + 1)
            if aic < best_aic - 1e-9:
                best_p, best_aic = p, aic
        p = best_p
        X = np.column_stack([dc[p - k: nd - k] for k in range(1, p + 1)])
        coef = _ridge_solve(X, dc[p:])
        coef = np.where(np.isfinite(coef), coef, 0.0)
        self._phi = _stabilise_ar(coef)
        self._hist = dc[-p:][::-1].copy()  # most recent first

    def _predict(self, h: int) -> np.ndarray:
        p = self._phi.size
        dev = np.zeros(h)
        if p:
            hist = list(self._hist)
            phi = self._phi
            for i in range(h):
                v = float(np.dot(phi, hist[:p]))
                dev[i] = v
                hist.insert(0, v)
        z = self._zlast + np.cumsum(self._mu + dev)
        return self._loc + self._scale * z


# --------------------------------------------------------------------------- lag ridge (ML)


def _lr_features(z: np.ndarray, idx: np.ndarray, p: int, m: int, seas: bool, w: int) -> np.ndarray:
    """Feature rows for predicting ``z[t] - z[t-1]`` at target positions ``idx``.

    * ``z[t-k] - z[t-k-1]`` for ``k = 1..p`` (lagged differences),
    * seasonal: ``z[t-m] - z[t-m-1]`` (same-phase difference last season) and
      ``z[t-m] - z[t-1]`` (seasonal-naive step),
    * level: ``z[t-1] - mean(z[t-w..t-1])`` (deviation from the local mean).
    """
    cols = [z[idx - k] - z[idx - k - 1] for k in range(1, p + 1)]
    if seas:
        cols.append(z[idx - m] - z[idx - m - 1])
        cols.append(z[idx - m] - z[idx - 1])
    if w > 1:
        cs = np.concatenate(([0.0], np.cumsum(z)))
        cols.append(z[idx - 1] - (cs[idx] - cs[idx - w]) / w)
    return np.column_stack(cols)


class LagRidge(Forecaster):
    """Machine-learning forecaster: ridge regression on lagged differences.

    Predicts the next first difference from ``p`` lagged differences, the
    seasonal-lag features (when the series is seasonal with period ``m`` and
    long enough) and a local level feature.  Features are standardised; the
    ridge penalty is chosen by generalised cross-validation over a small grid
    (closed form via SVD).
    Multi-step forecasts are recursive (each predicted difference is fed back)
    and integrated; predicted steps are clipped to a generous band around the
    observed differences so the recursion can never explode.  Falls back to
    drift when the history is too short to train on.
    """

    name = "lag_ridge"
    min_obs = 3

    def _fit(self, y: np.ndarray, m: int) -> None:
        n = len(y)
        z, self._loc, self._scale = _standardize(y)
        self._z = z
        self._mode = "drift"
        self.params_ = {"fallback": "drift"}  # replaced below when the ridge model is fitted
        if n < 8:
            return
        p = min(6, max(1, (n - 1) // 10))
        seas = m > 1 and (n - m - 1) >= max(12, 3 * (p + 3)) and _is_seasonal(y, m)
        w = max(m, 4) if seas else (4 if n >= 16 else 0)
        while True:
            start = max(p + 1, (m + 1) if seas else 0, w)
            rows = n - start
            F = p + (2 if seas else 0) + (1 if w > 1 else 0)
            if rows >= max(6, 2 * F) or (not seas and p == 1 and w == 0):
                break
            if seas:
                seas, w = False, (4 if n >= 16 else 0)
            elif w:
                w = 0
            else:
                p = max(1, p // 2)
        if rows < 5:
            return
        idx = np.arange(start, n)
        X = _lr_features(z, idx, p, m, seas, w)
        target = z[idx] - z[idx - 1]
        mu = X.mean(axis=0)
        sd = X.std(axis=0)
        sd = np.where(sd > 1e-12, sd, 1.0)
        Xs = (X - mu) / sd
        tmu = float(target.mean())
        tc = target - tmu
        try:
            U, s, Vt = np.linalg.svd(Xs, full_matrices=False)
        except np.linalg.LinAlgError:
            return
        Uty = U.T @ tc
        R = Xs.shape[0]
        best_lam, best_gcv = None, np.inf
        s2 = s * s
        for rel in (1e-3, 1e-2, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0):
            lam = rel * R
            f = s2 / (s2 + lam)
            rss = float(((tc - U @ (f * Uty)) ** 2).sum())
            df = float(f.sum())
            denom = max(R - df, 1e-9)
            gcv = R * rss / denom ** 2
            if gcv < best_gcv - 1e-15:
                best_lam, best_gcv = lam, gcv
        lam = best_lam if best_lam is not None else R
        coef = Vt.T @ (s / (s2 + lam) * Uty)
        if not np.all(np.isfinite(coef)):
            return
        self._mode = "ridge"
        self._p, self._seas, self._w = p, seas, w
        self._mu, self._sd, self._coef, self._tmu = mu, sd, coef, tmu
        rng = float(target.max() - target.min())
        self._dlo = float(target.min()) - 0.25 * rng
        self._dhi = float(target.max()) + 0.25 * rng
        self.params_ = {"p": p, "seasonal": seas, "level_window": w, "lambda": lam / R}

    def _predict(self, h: int) -> np.ndarray:
        z = self._z
        if self._mode != "ridge":
            return self._loc + self._scale * _drift_forecast(z, h)
        n = len(z)
        ext = np.concatenate([z, np.zeros(h)])
        m = self.season_length
        for i in range(h):
            t = n + i
            x = _lr_features(ext[: t + 1], np.array([t]), self._p, m, self._seas, self._w)[0]
            d = self._tmu + float(((x - self._mu) / self._sd) @ self._coef)
            d = min(max(d, self._dlo), self._dhi)
            ext[t] = ext[t - 1] + d
        return self._loc + self._scale * ext[n:]


# --------------------------------------------------------------------------- analogue k-NN


class AnalogKNN(Forecaster):
    """"Past analogues": k nearest historical windows, average of what happened next.

    The last ``w = max(m, 6)`` observations (capped at ``n // 3``) are
    z-normalised and compared (Euclidean) with every earlier window of the
    same length that has at least one future point; windows ending within
    ``w // 2`` steps of the present are excluded (trivial self-matches).  The
    forecast is ``last value + sum_i w_i * path_i * std(current) / std(window_i)``
    where ``path_i`` are the cumulative changes that followed analogue ``i``
    and ``w_i`` are inverse-distance weights (the std ratio is clipped to
    [1/3, 3]).  An analogue that runs out of future points continues with
    its average step, damped by 0.9 per step.

    "No good analogue" guard: the weighted analogue distance is compared with
    the distance expected between unrelated z-normalised windows
    (``sqrt(2 w)``); below half of it the analogues are fully trusted, above
    it not at all (the forecast then is the last value), with a linear blend
    in between.  Falls back to drift when the history is too short.

    Numerics: the series is first multiplied by an exact power of two (so
    ``max|y|`` lies in [0.5, 1); this changes no rounding at all) and the
    final change is scaled back, so the model is exactly scale-equivariant
    and its window variances cannot overflow even for ``|y| ~ 1e300``.
    Distances are compared after rounding to 1e-9 (they are O(1) for
    z-normalised windows), so analogues that are equally close up to rounding
    noise -- e.g. every window of a perfect line -- resolve deterministically
    to the earliest ones, which have the most future points.
    """

    name = "analog_knn"
    min_obs = 3
    k = 5

    def _fit(self, y: np.ndarray, m: int) -> None:
        n = len(y)
        self._mode = "drift"
        self.params_ = {"fallback": "drift"}  # replaced below when analogues are found
        ref = float(np.max(np.abs(y)))
        self._exp = math.frexp(ref)[1] if ref > 0 else 0
        z = np.ldexp(y, -self._exp)  # exact: y = z * 2**exp
        self._z = z
        w = min(max(m, 6), n // 3)
        if w < 3:
            return
        win = np.lib.stride_tricks.sliding_window_view(z, w)  # window i = z[i:i+w]
        mu = win.mean(axis=1)
        sd = win.std(axis=1)
        # windows whose spread is below 1e-9 of the series' magnitude count as flat
        floor = max(0.1 * float(np.median(sd)), 1e-9 * float(np.max(np.abs(z))), 1e-300)
        sde = np.maximum(sd, floor)
        Z = (win - mu[:, None]) / sde[:, None]
        excl = max(1, w // 2)
        i_max = n - w - excl  # window end i + w - 1 <= n - 1 - excl
        if i_max < 0:
            return
        cand = np.arange(0, i_max + 1)
        dist = np.sqrt(((Z[cand] - Z[-1]) ** 2).sum(axis=1))
        k = min(self.k, cand.size)
        order = np.argsort(np.round(dist, 9), kind="stable")[:k]
        self._ends = cand[order] + w - 1
        self._dist = dist[order]
        self._ratio = np.clip(sde[-1] / sde[cand[order]], 1.0 / 3.0, 3.0)
        wts = 1.0 / (self._dist + 0.1 * float(self._dist.mean()) + 1e-12)
        self._wts = wts / wts.sum()
        rel = float(self._wts @ self._dist) / math.sqrt(2.0 * w)
        self._trust = float(np.clip(2.0 * (1.0 - rel), 0.0, 1.0))
        self._mode = "knn"
        self._w = w
        self.params_ = {"window": w, "k": k, "trust": self._trust,
                        "analogue_ends": self._ends.tolist()}

    def _predict(self, h: int) -> np.ndarray:
        y, z = self._y, self._z
        n = len(z)
        if self._mode != "knn":
            return y[-1] + np.ldexp(_drift_forecast(z, h) - z[-1], self._exp)
        paths = np.empty((len(self._ends), h))
        damp = np.cumsum(0.9 ** np.arange(1, h + 1))
        for r, e in enumerate(self._ends):
            avail = n - 1 - e
            L = min(h, avail)
            fut = z[e + 1: e + 1 + L] - z[e]
            paths[r, :L] = fut
            if L < h:
                step = (z[e + avail] - z[e]) / avail
                paths[r, L:] = fut[-1] + step * damp[: h - L]
        paths *= self._ratio[:, None]
        return y[-1] + np.ldexp(self._trust * (self._wts @ paths), self._exp)


# --------------------------------------------------------------------------- logistic growth


def _logistic_eval(u: np.ndarray, y: np.ndarray, wt: np.ndarray, t0: np.ndarray, k: np.ndarray,
                   floor: float, cap: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For every ``(t0, k)`` solve the constrained weighted LS for ``(b, L)`` in closed form.

    Model ``y ~ b + L * g``, ``g = 1 / (1 + exp(-k (u - t0)))`` with
    ``L > 0``, ``b >= floor`` and upper asymptote ``b + L <= cap``; ``wt``
    are observation weights summing to 1.  Returns ``(wsse, b, L)``;
    infeasible combinations get ``wsse = inf``.
    """
    G = 1.0 / (1.0 + np.exp(np.clip(-k[:, None] * (u[None, :] - t0[:, None]), -50.0, 50.0)))
    gm = G @ wt
    ym = float(wt @ y)
    Gc = G - gm[:, None]
    sgg = (Gc * Gc) @ wt
    sgy = Gc @ (wt * (y - ym))
    ok = sgg > 1e-12
    L = np.where(ok, sgy / np.where(ok, sgg, 1.0), 0.0)
    b = ym - L * gm
    # floor constraint: b = floor
    low = b < floor
    if np.any(low):
        g2 = (G * G) @ wt
        Lf = (G @ (wt * (y - floor))) / np.maximum(g2, 1e-300)
        L = np.where(low, Lf, L)
        b = np.where(low, floor, b)
    # saturation constraint: b + L = cap
    high = (b + L) > cap
    if np.any(high):
        H = 1.0 - G
        h2 = (H * H) @ wt
        Lc = (H @ (wt * (cap - y))) / np.maximum(h2, 1e-300)
        L = np.where(high, Lc, L)
        b = np.where(high, cap - Lc, b)
    valid = ok & (L > 1e-9) & (b >= floor - 1e-9) & (b + L <= cap * (1 + 1e-9) + 1e-12)
    resid = y[None, :] - b[:, None] - L[:, None] * G
    sse = (resid * resid) @ wt
    sse = np.where(valid & np.isfinite(sse), sse, np.inf)
    return sse, b, L


class LogisticGrowth(Forecaster):
    """S-curve: ``y ~ b + L / (1 + exp(-k (t - t0)))`` (+ additive seasonal indices).

    ``(t0, k)`` by grid search (both signs of ``k``: growth and decline
    curves) with two local refinements; ``(b, L)`` by closed-form constrained
    least squares with ``L > 0``, ``b >= 0`` (for non-negative data) and the
    saturation level ``b + L <= 3 * max(y)``.  With two or more seasons, phase
    means of the residual are removed and the curve refitted once.  Forecasts
    therefore stay bounded for any input; on non-sigmoidal data the fit is
    just poor (and the backtest will rank it down).  Falls back to drift for
    very short or constant input.

    Identification (see :meth:`_admissible`): only curves whose transition is
    actually *observed* are considered -- at least ``min_active`` observations
    must lie in its 10-90 % part and at least ``min_observed`` of the whole
    transition must happen inside the data.  Otherwise a sharp step placed
    just after the last observation could fit that single (recency-weighted)
    point while the asymptote reached by the forecast is constrained by no
    data at all; on white noise this used to send roughly 20-30 % of the
    forecasts more than 10 standard deviations away from the data.
    """

    name = "logistic_growth"
    min_obs = 3
    requires_positive = True
    #: recency weighting: weight of observation at relative time u is exp(recency * (u - 1))
    recency = 3.0
    #: identification: min. number of observations inside the 10-90 % part of the transition
    min_active = 4
    #: identification: min. fraction of the transition that happens inside the observed range
    min_observed = 0.3

    def _fit(self, y: np.ndarray, m: int) -> None:
        n = len(y)
        self._n = n
        self._m = m
        self._mode = "drift"
        self.params_ = {"fallback": "drift"}  # replaced below when the curve is fitted
        ref = float(np.max(np.abs(y)))
        if n < 5 or ref <= 0 or float(np.ptp(y)) <= 1e-12 * ref:
            return
        yn = y / ref
        lo, hi = float(yn.min()), float(yn.max())
        floor = 0.0 if lo >= 0 else lo - (hi - lo)
        cap = 3.0 * hi if hi > 0 else hi + 2.0 * (hi - lo)
        u = np.arange(n, dtype=float) / (n - 1)
        wt = np.exp((u - 1.0) * self.recency)
        wt = wt / wt.sum()
        use_season = m > 1 and n >= 2 * m
        ph = np.arange(n) % m
        s = np.zeros(max(m, 1))
        fit = None
        for it in range(2 if use_season else 1):
            target = yn - s[ph] if use_season else yn
            fit = self._grid_fit(u, target, wt, floor, cap, self.min_active, self.min_observed)
            if fit is None:
                return
            if use_season:
                t0, k, b, L = fit
                curve = b + L / (1.0 + np.exp(np.clip(-k * (u - t0), -50.0, 50.0)))
                s = _phase_means(yn - curve, ph, m)
        t0, k, b, L = fit
        self._t0, self._k, self._b, self._L = t0, k, b, L
        self._s = s if use_season else np.zeros(1)
        self._use_season = use_season
        self._ref = ref
        self._mode = "logistic"
        self.params_ = {"t0": t0, "k": k, "b": b * ref, "L": L * ref}

    @staticmethod
    def _admissible(t0: np.ndarray, k: np.ndarray, n: int, min_active: int,
                    min_observed: float) -> np.ndarray:
        """Which ``(t0, k)`` curves have a transition that the ``n`` observations resolve.

        With ``c(u) = 1 / (1 + exp(-|k| (u - t0)))`` the completed fraction of
        the transition at relative time ``u`` (same formula for growth and
        decline), a curve is admissible iff ``c(1) - c(0) >= min_observed``
        and at least ``min_active`` of the observation times ``i / (n - 1)``
        fall in ``|u - t0| <= ln(9) / |k|`` (10 % <= c <= 90 %).
        """
        ak = np.abs(k)
        c1 = 1.0 / (1.0 + np.exp(np.clip(-ak * (1.0 - t0), -50.0, 50.0)))
        c0 = 1.0 / (1.0 + np.exp(np.clip(ak * t0, -50.0, 50.0)))
        hw = math.log(9.0) / ak
        first = np.ceil(np.maximum(0.0, t0 - hw) * (n - 1) - 1e-9)
        last = np.floor(np.minimum(1.0, t0 + hw) * (n - 1) + 1e-9)
        return ((c1 - c0) >= min_observed) & ((last - first + 1.0) >= min_active)

    @staticmethod
    def _grid_fit(u: np.ndarray, y: np.ndarray, wt: np.ndarray, floor: float, cap: float,
                  min_active: int = 4, min_observed: float = 0.3,
                  ) -> tuple[float, float, float, float] | None:
        """Coarse ``(t0, k)`` grid (on <= ~250 evenly spaced points), then 3 local refinements.

        Only :meth:`_admissible` curves compete; ``None`` if none is feasible.
        """
        n = len(y)
        step = max(1, math.ceil(n / 250))
        sub = np.unique(np.r_[np.arange(0, n, step), n - 1])
        ws = wt[sub] / wt[sub].sum()
        kpos = np.geomspace(1.5, 60.0, 13)
        t0g = np.linspace(-0.5, 2.0, 26)
        kg = np.r_[-kpos[::-1], kpos]
        T0, KK = np.meshgrid(t0g, kg, indexing="ij")
        t0s, ks = T0.ravel(), KK.ravel()
        sse, _, _ = _logistic_eval(u[sub], y[sub], ws, t0s, ks, floor, cap)
        sse = np.where(LogisticGrowth._admissible(t0s, ks, n, min_active, min_observed), sse, np.inf)
        i = int(np.argmin(sse))
        if not np.isfinite(sse[i]):
            return None
        bt0, bk, bsse = float(t0s[i]), float(ks[i]), np.inf
        bb = bL = 0.0
        dt, fk = 0.1, 1.35
        for _ in range(3):
            t0r = bt0 + dt * np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
            kr = bk * fk ** np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
            T0, KK = np.meshgrid(t0r, kr, indexing="ij")
            t0s, ks = T0.ravel(), KK.ravel()
            sse, b, L = _logistic_eval(u, y, wt, t0s, ks, floor, cap)
            sse = np.where(LogisticGrowth._admissible(t0s, ks, n, min_active, min_observed), sse, np.inf)
            j = int(np.argmin(sse))
            if float(sse[j]) < bsse:
                bt0, bk, bsse = float(t0s[j]), float(ks[j]), float(sse[j])
                bb, bL = float(b[j]), float(L[j])
            dt *= 0.5
            fk = math.sqrt(fk)
        if not np.isfinite(bsse):
            return None
        return bt0, bk, bb, bL

    def _predict(self, h: int) -> np.ndarray:
        y = self._y
        if self._mode != "logistic":
            return _drift_forecast(y, h)
        n = self._n
        u = np.arange(n, n + h, dtype=float) / (n - 1)
        curve = self._b + self._L / (1.0 + np.exp(np.clip(-self._k * (u - self._t0), -50.0, 50.0)))
        if self._use_season:
            curve = curve + self._s[np.arange(n, n + h) % self._m]
        return self._ref * curve


# --------------------------------------------------------------------------- log-space wrapper


def _max_log_moves(x: np.ndarray, J: int) -> tuple[np.ndarray, np.ndarray]:
    """Largest rise and largest drop of ``x`` over any span of ``<= j`` steps, for ``j = 1..J``.

    Returns ``(up, down)`` (both ``>= 0``, non-decreasing in ``j``); ``J`` must
    be ``<= len(x) - 1``.
    """
    up = np.empty(J)
    down = np.empty(J)
    for i in range(1, J + 1):
        d = x[i:] - x[:-i]
        up[i - 1] = d.max()
        down[i - 1] = -d.min()
    return (np.maximum.accumulate(np.maximum(up, 0.0)),
            np.maximum.accumulate(np.maximum(down, 0.0)))


def _log_move_bounds(ly: np.ndarray, m: int, h: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-step bounds ``(up, down)`` (length ``h``) on how far a log forecast may move.

    ``up[j-1]`` bounds the rise ``j`` steps ahead: the largest ``j``-step rise
    within the recent window (the last ``max(m, n // 4, 8)`` points), carried
    on at that window's average pace beyond its length, and never more than
    the largest rise over the same span anywhere in the history (which
    stays flat beyond ``n - 1`` steps, so a horizon longer than the history
    cannot add more growth than the whole history showed).  ``down`` is the
    same for drops.  Requires ``len(ly) >= 2``.
    """
    n = len(ly)
    steps = np.arange(1, h + 1)
    Ja = min(h, n - 1)
    up_all, dn_all = _max_log_moves(ly, Ja)
    W = min(n, max(m, n // 4, 8))
    Jr = min(h, W - 1)
    up_rec, dn_rec = _max_log_moves(ly[-W:], Jr)
    ja = np.minimum(steps, Ja) - 1
    jr = np.minimum(steps, Jr) - 1
    ext = np.maximum(steps / Jr, 1.0)
    return (np.minimum(up_rec[jr] * ext, up_all[ja]),
            np.minimum(dn_rec[jr] * ext, dn_all[ja]))


class LogTransformed(Forecaster):
    """Fit ``inner`` on ``log(y)`` and forecast ``exp(.)``.

    ``requires_positive``; ``needs_season`` and ``min_obs`` follow the inner
    model.  If non-positive values are passed anyway it degrades gracefully by
    shifting the series before the log.  A constant series is forecast exactly
    as that constant (no ``exp(log(c))`` rounding residue, e.g. exactly 0 for
    an all-zero series).

    Growth guard: a straight line in log space is exponential growth, and a
    log-slope estimated on a short explosive history (e.g. x8 over the last 6
    points) compounds into absurd forecasts (x300 after 30 steps).  So the
    log forecast may not rise (fall) ``j`` steps ahead by more than the
    largest ``j``-step log rise (fall) seen in the recent past, extended at
    that pace beyond the recent window, and never by more than the largest
    rise (fall) over the same span anywhere in the history -- which caps the
    total growth over a horizon longer than the series at the growth the
    whole series showed (see :func:`_log_move_bounds`).  The moves are
    measured from the last observation, or from the first forecast step if
    that already lies further out (the model's level estimate is never
    clipped, only the trajectory after it).  A steady exponential trend is
    left untouched.  Finally the log forecasts are clipped to a wide band
    around the observed log range, so ``exp`` can never overflow.
    """

    requires_positive = True

    def __init__(self, inner: Forecaster, name: str | None = None) -> None:
        super().__init__()
        self.inner = inner
        self.name = name or f"{inner.name}@log"
        self.min_obs = inner.min_obs
        self.needs_season = inner.needs_season
        if hasattr(inner, "adjust_multiplicative"):
            # seasonal factors of log data are additive
            inner.adjust_multiplicative = False

    def _fit(self, y: np.ndarray, m: int) -> None:
        ymin = float(y.min())
        if ymin > 0:
            self._shift = 0.0
        else:
            rng = float(y.max()) - ymin
            # relative floor (keeps y + shift > 0 after rounding when the range is tiny next to
            # |ymin|); an absolute one broke scale-equivariance below ~1e-7 (1e-300 data with
            # zeros: forecasts of 1e-25)
            ref = max(abs(ymin), abs(float(y.max())))
            self._shift = -ymin + max(0.01 * rng, 1e-9 * ref) if ref > 0 else 1.0  # 1.0: all zero
        ly = np.log(y + self._shift)
        self._ly = ly
        lo, hi = float(ly.min()), float(ly.max())
        span = hi - lo
        self._clip = (lo - 3.0 - span, hi + 3.0 + span)
        self._const = float(y[0]) if float(np.ptp(y)) == 0.0 else None
        self.inner.fit(ly, m)

    def _predict(self, h: int) -> np.ndarray:
        if self._const is not None:
            return np.full(h, self._const)
        lf = self.inner.predict(h)
        ly = self._ly
        if len(ly) >= 2:
            up, down = _log_move_bounds(ly, self.season_length, h)
            last = float(ly[-1])
            a_up = max(last, float(lf[0]) - float(up[0]))
            a_dn = min(last, float(lf[0]) + float(down[0]))
            lf = np.clip(lf, a_dn - down, a_up + up)
        lf = np.clip(lf, *self._clip)
        return np.exp(lf) - self._shift


class _LogHoltDamped(LogTransformed):
    def __init__(self) -> None:
        super().__init__(HoltDamped(), "holt_damped@log")


class _LogHoltWinters(LogTransformed):
    def __init__(self) -> None:
        super().__init__(HoltWinters(), "holt_winters@log")


class _LogLinearTrend(LogTransformed):
    def __init__(self) -> None:
        super().__init__(LinearTrend(), "linear_trend@log")


class _LogDrift(LogTransformed):
    """Random walk with drift in log space: constant average growth *rate*.

    ``log y`` is extrapolated along the line from its first to its last
    value, i.e. the forecast compounds the average growth rate of the
    history (``y_n * g**k`` with ``g = (y_n / y_1) ** (1 / (n - 1))``), at
    most: like every ``*@log`` model it is growth-guarded, so a recent pace
    slower than the average caps it.  For steadily compounding series (GDP, prices, populations) level-space drift
    projects *linear* growth and falls further behind with every step.

    Not offered automatically (``offered_for = "explicit"``): request it by
    name (``auto_forecast(..., candidates=[...])``).  As a default candidate
    it cut the error on simulated annual GDP-like series (MAPE at h = 1..5
    -11 %, median bias at h = 5 -2.2 % -> -1.5 %) but made the general
    accuracy testbed slightly worse (MAE +0.5 % over 966 series, interval
    coverage -0.1 pt): on S-curves cut mid-transition, which start from a
    low base, the average log growth compounds into large overshoots
    (``peaking`` W n=104: x2.6-4 the MAE).
    """

    offered_for = "explicit"

    def __init__(self) -> None:
        super().__init__(Drift(), "drift@log")


# --------------------------------------------------------------------------- registry

MODEL_REGISTRY: dict[str, Callable[[], Forecaster]] = {
    "naive": Naive,
    "seasonal_naive": SeasonalNaive,
    "drift": Drift,
}


def register(name: str, factory: Callable[[], Forecaster]) -> None:
    """Register (or replace) a zero-argument model factory under ``name``."""
    MODEL_REGISTRY[name] = factory


def make_model(name: str) -> Forecaster:
    """Create a fresh, unfitted model by registry name."""
    try:
        return MODEL_REGISTRY[name]()
    except KeyError:
        raise KeyError(f"unknown model {name!r}; available: {sorted(MODEL_REGISTRY)}") from None


def available_models() -> list[str]:
    """Sorted names of all registered models."""
    return sorted(MODEL_REGISTRY)


def default_candidates(n_obs: int, season_length: int, positive: bool,
                       intermittent: bool = False) -> list[str]:
    """Registry names worth trying for a series of this shape.

    Only returns models whose ``min_obs`` is satisfied; seasonal models only
    when at least two full seasons are available.  ``positive`` means all
    observations are > 0 (enables log-space variants if registered).
    ``intermittent`` (many exact zeros, see ``forecast._is_intermittent``)
    adds the models meant only for intermittent demand
    (``offered_for == "intermittent"``, e.g. ``sba``); they are never offered
    for ordinary series.  Models with ``offered_for == "explicit"`` are never
    returned (request them by name).
    """
    names = []
    for nm, factory in MODEL_REGISTRY.items():
        m = factory()
        offered = getattr(m, "offered_for", "regular")
        if offered == "explicit" or (offered == "intermittent" and not intermittent):
            continue
        if n_obs < m.min_obs:
            continue
        if m.needs_season and (season_length <= 1 or n_obs < 2 * season_length):
            continue
        if (m.requires_positive or nm.endswith("@log")) and not positive:
            continue
        names.append(nm)
    return names


for _name, _factory in (
    ("ses", SES),
    ("holt", Holt),
    ("holt_damped", HoltDamped),
    ("holt_winters", HoltWinters),
    ("theta", Theta),
    ("linear_trend", LinearTrend),
    ("ar", AR),
    ("lag_ridge", LagRidge),
    ("analog_knn", AnalogKNN),
    ("logistic_growth", LogisticGrowth),
    ("holt_damped@log", _LogHoltDamped),
    ("holt_winters@log", _LogHoltWinters),
    ("linear_trend@log", _LogLinearTrend),
    ("drift@log", _LogDrift),
    ("sba", SBA),
):
    register(_name, _factory)
del _name, _factory
