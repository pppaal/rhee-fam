"""Future: automatic forecasting with model selection, ensembling and calibrated intervals.

Pipeline of :func:`auto_forecast`::

    prepared series ──► rolling-origin backtest of every candidate model
                        (same windows for all models, MASE on the training scale)
                    ──► rank by MASE ──► inverse-MASE ensemble of the top-k
                        (kept only if it is not clearly worse than the best model)
                    ──► refit the chosen model(s) on the full history
                    ──► prediction intervals from the chosen method's own
                        backtest errors per horizon step (σ_k ≈ s₁·k^b), rescaled
                        to the current level when the errors grow with the level

:func:`compute_outlook` then turns a forecast into an interpretation relative
to the present (direction, expected change, probability of an increase,
scenarios, projected lifecycle phase, confidence).

Everything is numpy/pandas only and fully deterministic.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from . import models as _models
from .data import default_horizon, future_index, prepare
from .data import season_length as _freq_season_length
from .types import AnalysisResult, ForecastResult, ModelScore, Outlook

__all__ = [
    "mase_scale",
    "backtest",
    "auto_forecast",
    "compute_outlook",
    "norm_cdf",
    "norm_ppf",
]

# --------------------------------------------------------------------------- tuning constants

#: minimum training length of a backtest window when seasonality is not modelled
_MIN_TRAIN_NONSEASONAL = 6
#: at least this many backtest windows are required (otherwise: heuristic fallback)
_MIN_WINDOWS = 2
#: backtest history cap (most recent points); at least 8 seasons are always kept
_BT_MAX_HISTORY = 400
#: history cap for the final fit (most recent points)
_FIT_MAX_HISTORY = 1000
#: keep the best single model instead of the ensemble if its MASE < margin * ensemble MASE
_ENSEMBLE_MARGIN = 0.9
#: ... but only if the backtest has at least this many errors (W * h) per model
_MIN_ERRORS_FOR_SINGLE = 6
#: bounds of the horizon exponent b in sigma_k = s1 * k**b.  The lower bound keeps
#: intervals (mildly) widening with the horizon even when the backtest errors do not
#: grow; 0.1 rather than 0.3 because simulations showed 0.3 under-covers at h = 1 and
#: over-covers at long horizons for series with stationary noise.
_B_MIN, _B_MAX = 0.1, 1.0
#: horizon exponent used when it cannot be estimated (random-walk like growth)
_B_DEFAULT = 0.5
#: multiplier of the robust in-sample noise scale used as the floor of s1
_FLOOR_FACTOR = 0.5
#: small-sample / model-selection inflation of the backtest error scale
_SIGMA_INFLATION_BASE = 1.05
#: widening of the heuristic intervals used when the series is too short to backtest
_FALLBACK_WIDEN = 1.5
#: rescale backtest errors to the current level for positive series (see _level_adjust)
_LEVEL_ADJUST = True
#: floor of a backtest window's MASE scale, relative to the scale of the whole backtest
#: history (keeps MASE unit-free when a training window is constant, see _window_scales)
_SCALE_FLOOR_FRAC = 0.1


# --------------------------------------------------------------------------- normal distribution

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


def _level_z(level: float) -> float:
    """Two-sided normal quantile for a central ``level`` % interval (80 -> 1.2816)."""
    return norm_ppf(0.5 + float(level) / 200.0)


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
    whole backtest history; this is unit-free and leaves ordinary series
    unchanged.
    """
    floor = _SCALE_FLOOR_FRAC * mase_scale(y, m)
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
               min_windows: int = _MIN_WINDOWS) -> Optional[_Plan]:
    """Choose window count / backtest horizon / training length for a series of length ``n``.

    Seasonal layout first (every window trains on ``>= max(2m, 8)`` points),
    then a non-seasonal one (``>= 6`` points).  Within a layout the largest
    window count whose feasible horizon reaches ``max(1, horizon // 2)`` wins;
    if no layout reaches it, the feasible combination with the most errors
    (``W·h``) is used.  Returns ``None`` when not even ``min_windows`` windows
    of one step fit.
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
        for W in range(n_windows, min_windows - 1, -1):
            h = _largest_h(avail, W, horizon)
            if h < 1:
                continue
            step = _step(h)
            origins = tuple(n - h - (W - 1 - i) * step for i in range(W))
            plan = _Plan(horizon=h, origins=origins, seasonal=seasonal)
            if h >= h_need:
                return plan
            if best is None or W * h > best[0]:
                best = (W * h, plan)
    return best[1] if best is not None else None


# --------------------------------------------------------------------------- backtest core


def _window_forecasts(y: np.ndarray, name: str, m: int, plan: _Plan) -> np.ndarray:
    """``(W, h)`` forecasts of model ``name`` for every window of ``plan`` (may raise)."""
    h = plan.horizon
    out = np.empty((plan.n_windows, h))
    for i, o in enumerate(plan.origins):
        model = _models.make_model(name)
        model.fit(y[:o], season_length=m)
        f = np.asarray(model.predict(h), dtype=float)
        if f.shape != (h,) or not np.all(np.isfinite(f)):
            raise ValueError(f"model {name!r} returned an invalid forecast")
        out[i] = f
    return out


def _actuals(y: np.ndarray, plan: _Plan) -> np.ndarray:
    return np.stack([y[o:o + plan.horizon] for o in plan.origins])


def _scores(forecasts: np.ndarray, actuals: np.ndarray, scales: np.ndarray) -> dict:
    """Accuracy metrics of ``(W, h)`` forecasts; MASE uses the per-window training scale."""
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        err = actuals - forecasts
        abs_e = np.abs(err)
        denom = np.abs(actuals) + np.abs(forecasts)
        smape_terms = np.where(denom > 0, 200.0 * abs_e / np.where(denom > 0, denom, 1.0), 0.0)
        return {
            "mae": float(np.mean(abs_e)),
            "rmse": float(_rms(err)),
            "smape": float(np.mean(smape_terms)),
            "mase": float(np.mean(abs_e.mean(axis=1) / scales)),
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
    of its own training part (floored at 10 % of the scale of the whole
    series, so that a constant training window cannot make MASE depend on
    the units of the data).

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
    best model on the same windows, and floored by ``floor``.
    Returns ``(sigma, s1, b)``.
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
    sigma = s1 * np.arange(1, horizon + 1, dtype=float) ** b
    return sigma, s1, b


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
           index: pd.DatetimeIndex) -> tuple[pd.Series, dict[int, pd.Series], dict[int, pd.Series]]:
    """Mean and symmetric normal bands, clipped at 0 when ``nonnegative``."""
    mean = np.asarray(mean, dtype=float)
    if nonnegative:
        mean = np.maximum(mean, 0.0)
    lower: dict[int, pd.Series] = {}
    upper: dict[int, pd.Series] = {}
    for lv in levels:
        z = _level_z(lv)
        lo = mean - z * sigma
        hi = mean + z * sigma
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


def _fallback_forecast(y: np.ndarray, m: int, horizon: int, reason: str
                       ) -> tuple[np.ndarray, np.ndarray, dict[str, float], list[ModelScore], str]:
    """Equal-weight drift + naive (naive alone for n < 3) with wide heuristic intervals.

    Uses the built-in classes directly (not the registry) so it cannot be
    broken by a re-registered model.  Model scores come from one-step
    expanding-window errors when there are any.
    """
    n = len(y)
    factories: dict[str, type[_models.Forecaster]] = {"naive": _models.Naive}
    if n >= 3:
        factories = {"drift": _models.Drift, "naive": _models.Naive}
    w = 1.0 / len(factories)
    weights = {nm: w for nm in factories}
    mean = np.zeros(horizon)
    for cls in factories.values():
        mean += w * cls().fit(y, m).predict(horizon)

    # crude one-step expanding-window scores (common origins for all models)
    first = max(cls.min_obs for cls in factories.values())
    origins = list(range(first, n))
    scores: list[ModelScore] = []
    if origins:
        for nm, cls in factories.items():
            fc = np.array([[cls().fit(y[:o], m).predict(1)[0]] for o in origins])
            act = np.array([[y[o]] for o in origins])
            scales = _window_scales(y, origins, m)
            sc = _scores(fc, act, scales)
            if _finite_scores(sc):
                scores.append(ModelScore(nm, sc["mae"], sc["rmse"], sc["smape"], sc["mase"],
                                         weights[nm]))
        scores.sort(key=lambda s: s.mase)

    # heuristic wide intervals: RMS of first differences, random-walk growth, widened
    level = float(np.mean(np.abs(y)))
    s1 = float(_rms(np.diff(y))) if n >= 2 else 0.0
    s1 = max(s1, 0.05 * level, _tiny_scale(y))
    sigma = _FALLBACK_WIDEN * s1 * np.sqrt(np.arange(1, horizon + 1, dtype=float))
    method = f"{_describe(weights, 'fallback')} [{reason}]"
    if len(weights) == 1:
        method = f"fallback({next(iter(weights))}) [{reason}]"
    return mean, sigma, weights, scores, method


# --------------------------------------------------------------------------- auto_forecast


def auto_forecast(series: pd.Series, horizon: int | None = None, freq: str | None = None,
                  levels: tuple[int, ...] = (80, 95), candidates: list[str] | None = None,
                  top_k: int = 3, n_windows: int | None = None,
                  nonnegative: bool | None = None) -> ForecastResult:
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

    Notes
    -----
    * The ensemble (inverse-MASE weights over the top-k) is used unless the
      best single model's backtest MASE is below 0.9 x the ensemble's (and the
      backtest has at least 6 errors per model); ``ModelScore.weight`` holds
      the final weights (1 for a single chosen model).
    * Intervals are normal (``mean ± z·σ_k``) with ``σ_k = s₁·k^b`` fitted to
      the final method's backtest errors (``b`` in ``[0.1, 1]``, ``s₁``
      floored by a robust in-sample noise level).  For positive series the
      errors are first rescaled to the current level with an elasticity
      estimated from the windows (see :func:`_level_adjust`).  With
      ``nonnegative`` the bands are centred on the clipped mean, then clipped.
    * The backtest uses at most the last ``max(400, 8·season_length)``
      observations, the final fit at most the last 1000.
    * Series too short for a 2-window backtest (fewer than ~8 points), or for
      which every candidate fails, get an equal-weight drift + naive forecast
      with wide heuristic intervals and ``backtest_mase=None``.
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

    y = y_full[-_FIT_MAX_HISTORY:]
    y_bt = y[-max(_BT_MAX_HISTORY, 8 * m):]
    W_target = int(n_windows) if n_windows is not None else _default_n_windows(len(y_bt))
    plan = _make_plan(len(y_bt), m, H, W_target)

    def fallback(reason: str) -> ForecastResult:
        mean, sigma, _, scores, method = _fallback_forecast(y, m, H, reason)
        mean_s, lower, upper = _bands(mean, sigma, lvls, nonnegative, index)
        return ForecastResult(mean=mean_s, lower=lower, upper=upper, horizon=H, levels=lvls,
                              model_scores=scores, method=method, backtest_mase=None,
                              nonnegative=nonnegative)

    if plan is None:
        return fallback("too short to backtest")

    # ---- candidates
    positive = bool(np.all(y > 0))
    if candidates is None:
        names = list(_models.default_candidates(plan.min_train, m, positive))
    else:
        names = [str(c) for c in candidates]
    names = list(dict.fromkeys(names))  # dedupe, keep order

    # ---- backtest every candidate on the same windows
    actuals = _actuals(y_bt, plan)
    scales = _window_scales(y_bt, plan.origins, m)
    results: dict[str, tuple[np.ndarray, dict]] = {}
    for nm in names:
        try:
            fc = _window_forecasts(y_bt, nm, m, plan)
        except Exception as exc:  # noqa: BLE001 - a broken model must never break the forecast
            warnings.warn(f"forecast: model {nm!r} skipped in backtest "
                          f"({type(exc).__name__}: {exc})", RuntimeWarning, stacklevel=2)
            continue
        if nonnegative:
            fc = np.maximum(fc, 0.0)
        sc = _scores(fc, actuals, scales)
        if not _finite_scores(sc):
            warnings.warn(f"forecast: model {nm!r} skipped (non-finite backtest error)",
                          RuntimeWarning, stacklevel=2)
            continue
        results[nm] = (fc, sc)
    if not results:
        warnings.warn("forecast: no candidate model could be backtested; using fallback",
                      RuntimeWarning, stacklevel=2)
        return fallback("no usable candidate model")

    order = {nm: i for i, nm in enumerate(names)}
    ranked = sorted(results, key=lambda nm: (results[nm][1]["mase"], order[nm]))

    # ---- ensemble vs. best single model
    best = ranked[0]
    chosen: dict[str, float] = {best: 1.0}
    k = min(int(top_k), len(ranked))
    if k >= 2:
        top = ranked[:k]
        inv = np.array([1.0 / max(results[nm][1]["mase"], 1e-9) for nm in top])
        wts = inv / inv.sum()
        ens_fc = np.tensordot(wts, np.stack([results[nm][0] for nm in top]), axes=1)
        ens_mase = _scores(ens_fc, actuals, scales)["mase"]
        enough = plan.n_windows * plan.horizon >= _MIN_ERRORS_FOR_SINGLE
        if not (enough and results[best][1]["mase"] < _ENSEMBLE_MARGIN * ens_mase):
            chosen = {nm: float(w) for nm, w in zip(top, wts)}

    # ---- final fit on the full (capped) history
    def final_fit(nm: str) -> Optional[np.ndarray]:
        try:
            f = np.asarray(_models.make_model(nm).fit(y, season_length=m).predict(H), dtype=float)
            if f.shape != (H,) or not np.all(np.isfinite(f)):
                raise ValueError("invalid forecast")
        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"forecast: model {nm!r} failed on the full history "
                          f"({type(exc).__name__}: {exc})", RuntimeWarning, stacklevel=3)
            return None
        return f

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
    final_scores = _scores(bt_fc, actuals, scales)

    # ---- prediction intervals from the final method's backtest errors
    errors = actuals - bt_fc
    if _LEVEL_ADJUST:
        errors, _ = _level_adjust(errors, y_bt, plan.origins, m)
    sigma, _, _ = _fit_sigma(errors, H, _noise_floor(y, m))
    mean_s, lower, upper = _bands(mean, sigma, lvls, nonnegative, index)

    model_scores = [
        ModelScore(name=nm, mae=results[nm][1]["mae"], rmse=results[nm][1]["rmse"],
                   smape=results[nm][1]["smape"], mase=results[nm][1]["mase"],
                   weight=float(weights.get(nm, 0.0)))
        for nm in ranked
    ]
    return ForecastResult(mean=mean_s, lower=lower, upper=upper, horizon=H, levels=lvls,
                          model_scores=model_scores, method=_describe(weights),
                          backtest_mase=float(final_scores["mase"]), nonnegative=nonnegative)


# --------------------------------------------------------------------------- outlook


def _interval_level(forecast: ForecastResult, target: int = 80) -> Optional[int]:
    """The available interval level closest to ``target`` (``None`` if there is none)."""
    avail = [lv for lv in forecast.levels if lv in forecast.lower and lv in forecast.upper]
    if not avail:
        return None
    return min(avail, key=lambda lv: (abs(lv - target), lv))


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


def _future_seasonal(analysis: AnalysisResult, horizon: int) -> np.ndarray:
    """The analysis' seasonal component continued over the forecast horizon (zeros if none)."""
    seas, p = _seasonal_component(analysis)
    if seas is None:
        return np.zeros(horizon)
    last = seas[-p:]  # phases of observations n-p .. n-1; step k (0-based) is t = n + k
    return last[np.arange(horizon) % p]


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


def _confidence(backtest_mase: Optional[float], n: int, m: int) -> str:
    if backtest_mase is None or not math.isfinite(backtest_mase) or backtest_mase > 1.5 or n < 12:
        return "low"
    enough = n >= (3 * m if m > 1 else 36)
    if backtest_mase < 0.8 and enough:
        return "high"
    return "medium"


def _future_phase(current: str, direction: str, strong: bool, change_pct: Optional[float],
                  confidence: str) -> str:
    """Project the lifecycle phase at the horizon end from the current phase + direction."""
    if current == "volatile" and confidence == "low":
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
    if current in ("growing", "peaking", "emerging"):
        return "peaking"
    return "stable"


def compute_outlook(analysis: AnalysisResult, forecast: ForecastResult) -> Outlook:
    """Interpret ``forecast`` relative to the present state in ``analysis``.

    * current level = mean of the last ``min(season_length, n)`` observations,
      future level = mean of the last ``min(season_length, horizon)`` forecast
      points; ``expected_change_pct`` compares the two (``None`` if the current
      level is ~0).  When one of these windows covers only *part* of a
      seasonal cycle (e.g. weekly data: default horizon 26 < 52), the
      analysis' seasonal component averaged over that window (scaled by the
      share of it the forecast contains) is subtracted, so that a seasonal
      high or low is not reported as a trend change.  For windows of whole
      seasons (the default for D/M/Q/Y) nothing changes.
    * ``prob_increase`` = ``Φ((future − current) / σ_H)`` with σ_H recovered
      from the 80 % interval at the horizon end (normal approximation; the
      wider side is used when a bound was clipped at 0).
    * ``direction`` is ``'flat'`` when ``|future − current|`` is within
      ``max(2 % of the current level, half the 80 % half-width at h = 1)``.
    * scenarios = lo80 / mean / hi80 at the horizon end.
    * ``forecast_peak_date``: date of the forecast maximum when it lies
      strictly inside the horizon, the forecast later falls ≥ 5 % below it and
      it rose to it by more than the noise threshold used for ``direction``
      (checked on the forecast and on its seasonally adjusted version).
    * ``future_phase`` from the current lifecycle phase + direction (+ size of
      the change); ``confidence`` from the backtest MASE and the sample size.
    """
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
    seas_fut = _future_seasonal(analysis, H)
    seas_fut = _seasonal_share(fm, seas_fut, p) * seas_fut
    if seas is not None:
        # compare like with like: remove the seasonal effect of partial-season windows
        if k_fut % p:
            future -= float(np.mean(seas_fut[-k_fut:]))
        if k_cur % p:
            current -= float(np.mean(seas[finite][-k_cur:]))
    diff = future - current
    ref = float(np.mean(np.abs(y)))
    if abs(current) > 1e-6 * max(ref, 1e-300):
        expected_change_pct: Optional[float] = float(diff / abs(current) * 100.0)
    else:
        expected_change_pct = None

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

    # direction: change must exceed the noise level
    half80_h1 = _level_z(80) * sigma_1
    threshold = max(0.02 * abs(current), 0.5 * half80_h1)
    if abs(diff) <= threshold:
        direction = "flat"
    else:
        direction = "up" if diff > 0 else "down"
    strong = abs(diff) >= max(0.10 * abs(current), 2.0 * threshold)

    # peak strictly inside the horizon, reached by a rise above the noise level and
    # followed by a >= 5 % fall -- of the forecast itself *and* of its seasonally
    # adjusted, lightly smoothed version (a seasonal high, or a wiggle left over
    # from noisy seasonal estimates -- e.g. weekly data with 52 phases -- is no peak)
    peak_date: Optional[pd.Timestamp] = None
    adjusted = _smooth(fm - seas_fut, 2 * (m // 8) + 1)
    if (_has_inner_peak(fm, threshold, current)
            and _has_inner_peak(adjusted, threshold, current)):
        peak_date = pd.Timestamp(forecast.mean.index[int(np.argmax(fm))])

    confidence = _confidence(forecast.backtest_mase, n, m)
    future_phase = _future_phase(analysis.phase.label, direction, strong, expected_change_pct,
                                 confidence)
    return Outlook(direction=direction, expected_change_pct=expected_change_pct,
                   prob_increase=prob_increase, scenarios=scenarios,
                   forecast_peak_date=peak_date, future_phase=future_phase,
                   confidence=confidence)
