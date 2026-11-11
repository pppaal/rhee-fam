"""Tests for trendai.forecast: backtest, model selection, ensemble, intervals, outlook.

Every test must pass both with the three baseline models only and with the
full model zoo registered in :mod:`trendai.models`; tests that depend on the
registry contents either pin ``candidates`` or run under the ``registry``
fixture, which executes them once with the full registry and once with the
baselines only.  ``AnalysisResult`` objects are built by hand (no dependency
on :mod:`trendai.analysis`).
"""

from __future__ import annotations

import dataclasses
import math
import time
import warnings

import numpy as np
import pandas as pd
import pytest

from trendai import forecast as F
from trendai import models as M
from trendai.data import default_horizon, from_values, future_index, prepare, season_length
from trendai.samples import sample
from trendai.types import (
    PHASES, AnalysisResult, Decomposition, ForecastResult, LifecyclePhase, ModelScore, Outlook,
    TrendMetrics,
)

BASELINES = ("naive", "seasonal_naive", "drift")


# --------------------------------------------------------------------------- fixtures / helpers


@pytest.fixture
def clean_registry():
    """Snapshot ``MODEL_REGISTRY`` and restore it afterwards (for toy registrations)."""
    saved = dict(M.MODEL_REGISTRY)
    yield M.MODEL_REGISTRY
    M.MODEL_REGISTRY.clear()
    M.MODEL_REGISTRY.update(saved)


@pytest.fixture(params=["full", "baselines"])
def registry(request, clean_registry):
    """Run the test with the full registry and with the baseline models only."""
    if request.param == "baselines":
        keep = {k: clean_registry[k] for k in BASELINES}
        clean_registry.clear()
        clean_registry.update(keep)
    return request.param


def _lin_ar1(seed: int, n: int, phi: float = 0.5, slope: float = 0.3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    eps = rng.standard_normal(n)
    e = np.zeros(n)
    for i in range(n):
        e[i] = (phi * e[i - 1] if i else 0.0) + eps[i]
    return 20.0 + slope * np.arange(n) + e


def _assert_valid(fc: ForecastResult, horizon: int) -> None:
    assert isinstance(fc, ForecastResult)
    assert fc.horizon == horizon
    assert len(fc.mean) == horizon
    assert np.all(np.isfinite(fc.mean.to_numpy()))
    for lv in fc.levels:
        lo, hi = fc.lower[lv].to_numpy(), fc.upper[lv].to_numpy()
        assert lo.shape == hi.shape == (horizon,)
        assert np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))
        assert np.all(lo <= fc.mean.to_numpy() + 1e-9)
        assert np.all(fc.mean.to_numpy() <= hi + 1e-9)
        assert fc.lower[lv].index.equals(fc.mean.index)
        assert fc.upper[lv].index.equals(fc.mean.index)
    for ms in fc.model_scores:
        assert isinstance(ms, ModelScore)
        for v in (ms.mae, ms.rmse, ms.smape, ms.mase, ms.weight):
            assert isinstance(v, float) and math.isfinite(v)
        assert 0.0 <= ms.smape <= 200.0
        assert ms.mase >= 0.0 and 0.0 <= ms.weight <= 1.0
    mases = [ms.mase for ms in fc.model_scores]
    assert mases == sorted(mases)
    if fc.backtest_mase is not None:
        assert math.isfinite(fc.backtest_mase) and fc.backtest_mase >= 0
    assert isinstance(fc.method, str) and fc.method
    frame = fc.to_frame()
    assert list(frame.columns[:1]) == ["mean"]


def _make_analysis(s: pd.Series, freq: str, phase: str = "growing",
                   seasonal: pd.Series | None = None,
                   seasonal_strength: float | None = None) -> AnalysisResult:
    """A minimal hand-built AnalysisResult (only what compute_outlook needs is meaningful).

    ``seasonal_strength`` defaults to 0.9 when a ``seasonal`` component is given (the
    analysis found that pattern) and 0 otherwise: compute_outlook removes the seasonal
    effect of a partial-season window only for seasonality that is not negligible."""
    if seasonal_strength is None:
        seasonal_strength = 0.9 if seasonal is not None else 0.0
    m = season_length(freq)
    seas = seasonal if seasonal is not None else pd.Series(np.zeros(len(s)), index=s.index)
    trend = (s - seas).rolling(max(m, 3), center=True, min_periods=1).mean()
    dec = Decomposition(trend=trend, seasonal=seas, resid=s - trend - seas, period=m)
    peak_i = int(np.argmax(s.to_numpy()))
    metrics = TrendMetrics(
        n_obs=len(s), freq=freq, start=s.index[0], end=s.index[-1], last_value=float(s.iloc[-1]),
        mean=float(s.mean()), total_change_pct=None, cagr_pct=None, recent_change_pct=None,
        slope=0.0, slope_pct=0.0, momentum=0.0, acceleration=0.0, volatility=0.1,
        trend_strength=0.5, seasonal_strength=float(seasonal_strength), mk_tau=0.0, mk_pvalue=1.0,
        direction="flat", peak_date=s.index[peak_i], peak_value=float(s.iloc[peak_i]), drawdown_pct=0.0,
    )
    return AnalysisResult(name=str(s.name), series=s, freq=freq, season_length=m, decomposition=dec,
                          metrics=metrics, changepoints=[], anomalies=[],
                          phase=LifecyclePhase(label=phase, confidence=0.8, reasons=[]))


def _hand_forecast(s: pd.Series, freq: str, mean: np.ndarray, sd: np.ndarray,
                   backtest_mase: float | None = 0.5) -> ForecastResult:
    idx = future_index(s.index[-1], freq, len(mean))
    mu = pd.Series(mean, index=idx)
    lower = {lv: mu - F.norm_ppf(0.5 + lv / 200) * sd for lv in (80, 95)}
    upper = {lv: mu + F.norm_ppf(0.5 + lv / 200) * sd for lv in (80, 95)}
    return ForecastResult(mean=mu, lower=lower, upper=upper, horizon=len(mean), levels=(80, 95),
                          model_scores=[], method="hand", backtest_mase=backtest_mase)


# --------------------------------------------------------------------------- normal distribution


def test_norm_ppf_known_values_and_inverse():
    assert F.norm_ppf(0.5) == pytest.approx(0.0, abs=1e-12)
    assert F.norm_ppf(0.9) == pytest.approx(1.2815515655446004, abs=1e-9)
    assert F.norm_ppf(0.975) == pytest.approx(1.959963984540054, abs=1e-9)
    assert F.norm_ppf(0.995) == pytest.approx(2.5758293035489004, abs=1e-9)
    assert F.norm_ppf(1e-6) == pytest.approx(-4.753424308822899, abs=1e-7)
    for p in np.linspace(0.001, 0.999, 57):
        assert F.norm_cdf(F.norm_ppf(p)) == pytest.approx(p, abs=1e-12)
        assert F.norm_ppf(1 - p) == pytest.approx(-F.norm_ppf(p), abs=1e-9)
    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            F.norm_ppf(bad)


# --------------------------------------------------------------------------- mase_scale / backtest


def test_mase_scale_definition_and_fallbacks():
    y = np.array([1.0, 3.0, 2.0, 5.0, 4.0, 8.0])
    assert F.mase_scale(y, 2) == pytest.approx(np.mean(np.abs(y[2:] - y[:-2])))
    assert F.mase_scale(y, 1) == pytest.approx(np.mean(np.abs(np.diff(y))))
    # not longer than one season -> naive fallback
    assert F.mase_scale(y, 12) == pytest.approx(np.mean(np.abs(np.diff(y))))
    # perfectly seasonal -> seasonal MAE is 0 -> naive fallback
    per = np.tile([1.0, 5.0, 3.0], 4)
    assert F.mase_scale(per, 3) == pytest.approx(np.mean(np.abs(np.diff(per))))
    for degenerate in (np.full(10, 7.0), np.zeros(10), np.array([3.0]), np.array([])):
        s = F.mase_scale(degenerate, 4)
        assert math.isfinite(s) and s > 0


def test_backtest_shapes_metrics_and_window_layout():
    n, h = 60, 6
    y = 10.0 + 0.5 * np.arange(n) + np.sin(np.arange(n))
    res = F.backtest(y, "naive", season_length=1, horizon=h, n_windows=3)
    assert {"mae", "rmse", "smape", "mase", "forecasts", "actuals"} <= set(res)
    fc, act = res["forecasts"], res["actuals"]
    assert fc.shape == act.shape == (3, h)
    # windows end at the end of the series, stepping back by h // 2
    np.testing.assert_allclose(act[-1], y[-h:])
    assert res["origins"] == [n - h - 2 * (h // 2), n - h - (h // 2), n - h]
    for o, row_f, row_a in zip(res["origins"], fc, act):
        np.testing.assert_allclose(row_f, np.full(h, y[o - 1]))   # naive repeats the last value
        np.testing.assert_allclose(row_a, y[o:o + h])
    err = act - fc
    assert res["mae"] == pytest.approx(np.mean(np.abs(err)))
    assert res["rmse"] == pytest.approx(np.sqrt(np.mean(err ** 2)))
    scales = [F.mase_scale(y[:o], 1) for o in res["origins"]]
    assert res["mase"] == pytest.approx(np.mean(np.abs(err).mean(axis=1) / scales))
    assert 0 <= res["smape"] <= 200


def test_backtest_drift_is_exact_on_a_line_and_short_series_shrink():
    y = 3.0 + 2.0 * np.arange(40)
    res = F.backtest(y, "drift", season_length=1, horizon=5)
    assert res["mae"] == pytest.approx(0.0, abs=1e-9) and res["mase"] == pytest.approx(0.0, abs=1e-9)
    # a short series gets fewer windows / a shorter horizon instead of failing
    short = F.backtest(np.arange(10.0), "naive", season_length=1, horizon=12, n_windows=3)
    W, h = short["forecasts"].shape
    assert 1 <= W <= 3 and 1 <= h <= 12 and short["origins"][0] >= 6
    with pytest.raises(ValueError):
        F.backtest(np.arange(5.0), "naive", season_length=1, horizon=3)


def test_backtest_seasonal_layout_keeps_two_seasons():
    y = sample("seasonal", n=60).to_numpy()
    res = F.backtest(y, "seasonal_naive", season_length=12, horizon=12, n_windows=3)
    assert res["origins"][0] >= 24
    assert res["forecasts"].shape[1] >= 6


def test_make_plan_properties():
    for n in (8, 9, 12, 20, 36, 60, 120, 400):
        for m in (1, 4, 7, 12, 52):
            for h in (1, 5, 12, 30):
                plan = F._make_plan(n, m, h, F._default_n_windows(n))
                assert plan is not None, (n, m, h)
                assert plan.n_windows >= 2 and 1 <= plan.horizon <= h
                assert plan.origins[-1] + plan.horizon == n
                assert list(plan.origins) == sorted(set(plan.origins))
                assert plan.min_train >= (max(2 * m, 8) if plan.seasonal else 6)
    assert F._make_plan(7, 1, 5, 3) is None
    big = F._make_plan(120, 12, 12, 8)
    assert big.seasonal and big.horizon == 12 and big.n_windows == 8


# --------------------------------------------------------------------------- intervals


def test_interval_calibration_linear_trend_ar1_baselines():
    """40 seeds of a linear trend + AR(1) noise: empirical coverage near nominal."""
    n, h = 100, 12
    cover = {80: [], 95: []}
    for seed in range(40):
        y = _lin_ar1(seed, n + h)
        fc = F.auto_forecast(from_values(y[:n], freq="M"), horizon=h, freq="M",
                             candidates=list(BASELINES))
        actual = y[n:]
        for lv in cover:
            lo, hi = fc.lower[lv].to_numpy(), fc.upper[lv].to_numpy()
            cover[lv].append((actual >= lo) & (actual <= hi))
    c80, c95 = float(np.mean(cover[80])), float(np.mean(cover[95]))
    assert 0.65 <= c80 <= 0.95, c80
    assert 0.85 <= c95 <= 0.995, c95


def test_interval_calibration_default_candidates():
    """Same check and the same required bounds with whatever models are registered."""
    n, h = 100, 12
    cover = {80: [], 95: []}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for seed in range(40):
            y = _lin_ar1(seed, n + h)
            fc = F.auto_forecast(from_values(y[:n], freq="M"), horizon=h, freq="M")
            for lv in cover:
                cover[lv].append((y[n:] >= fc.lower[lv].to_numpy())
                                 & (y[n:] <= fc.upper[lv].to_numpy()))
    c80, c95 = float(np.mean(cover[80])), float(np.mean(cover[95]))
    assert 0.65 <= c80 <= 0.95, c80
    assert 0.85 <= c95 <= 0.995, c95


def test_interval_calibration_multiplicative_growth_baselines():
    """Exponential growth with multiplicative noise: errors are rescaled to the current level."""
    n, h = 100, 12
    hits = []
    for seed in range(40):
        rng = np.random.default_rng(500 + seed)
        t = np.arange(n + h)
        y = 10.0 * np.exp(0.03 * t + 0.08 * rng.standard_normal(n + h))
        fc = F.auto_forecast(from_values(y[:n], freq="M"), horizon=h, freq="M",
                             candidates=list(BASELINES))
        hits.append((y[n:] >= fc.lower[80].to_numpy()) & (y[n:] <= fc.upper[80].to_numpy()))
    assert 0.6 <= float(np.mean(hits)) <= 0.95


def test_level_adjust_elasticity():
    y = np.r_[np.full(20, 10.0), np.full(20, 20.0), np.full(20, 40.0), np.full(20, 80.0)]
    origins = [20, 40, 60]
    lv = np.array([10.0, 20.0, 40.0])
    prop = np.outer(lv, np.ones(4)) * 0.1          # errors proportional to the level
    adj, gamma = F._level_adjust(prop, y, origins, 4)
    assert gamma == pytest.approx(1.0)
    np.testing.assert_allclose(adj, np.full((3, 4), 8.0))   # rescaled to the level 80
    const = np.ones((3, 4))                         # additive errors: untouched
    adj2, gamma2 = F._level_adjust(const, y, origins, 4)
    assert gamma2 == 0.0 and np.array_equal(adj2, const)
    adj3, gamma3 = F._level_adjust(prop, y - 50.0, origins, 4)   # not positive: untouched
    assert gamma3 == 0.0 and np.array_equal(adj3, prop)


@pytest.mark.parametrize("kind", ["growing", "seasonal", "declining", "volatile"])
def test_bands_ordered_nested_and_widening(registry, kind):
    s = sample(kind, n=96)
    fc = F.auto_forecast(s, horizon=18, freq="M", levels=(50, 80, 95))
    _assert_valid(fc, 18)
    w = {lv: (fc.upper[lv] - fc.lower[lv]).to_numpy() for lv in fc.levels}
    assert np.all(w[50] > 0)
    assert np.all(w[50] <= w[80] + 1e-9) and np.all(w[80] <= w[95] + 1e-9)
    assert np.all(w[95] > w[80])
    # no clipping happens on these positive, well-above-zero series -> weakly widening
    assert np.all(np.diff(w[80]) >= -1e-9)
    assert np.all(np.diff(w[95]) >= -1e-9)


def test_intervals_never_degenerate_on_constant_or_perfect_series():
    for y in (np.full(40, 5.0), 2.0 + 0.5 * np.arange(40.0)):
        fc = F.auto_forecast(from_values(y, freq="M"), horizon=6, freq="M",
                             candidates=list(BASELINES))
        _assert_valid(fc, 6)
        assert np.all((fc.upper[80] - fc.lower[80]).to_numpy() > 0)


# --------------------------------------------------------------------------- auto_forecast behaviour


def test_nonnegative_clipping():
    t = np.arange(48.0)
    y = np.maximum(60.0 - 1.6 * t + np.sin(t), 0.0) + 0.5   # heading for zero, all >= 0
    s = from_values(y, freq="M")
    fc = F.auto_forecast(s, horizon=24, freq="M", candidates=["drift"])
    assert fc.nonnegative is True
    assert fc.mean.min() >= 0 and float(fc.mean.iloc[-1]) == 0.0
    for lv in fc.levels:
        assert fc.lower[lv].min() >= 0 and fc.upper[lv].min() >= 0
    raw = F.auto_forecast(s, horizon=24, freq="M", candidates=["drift"], nonnegative=False)
    assert raw.nonnegative is False and raw.mean.min() < 0
    # a series with negative values is not clipped by default
    neg = F.auto_forecast(from_values(y - 30.0, freq="M"), horizon=24, freq="M",
                          candidates=["drift"])
    assert neg.nonnegative is False and neg.mean.min() < 0


@pytest.mark.parametrize("n", [2, 3, 5, 7, 8, 9])
def test_short_series_do_not_crash(registry, n):
    rng = np.random.default_rng(n)
    s = from_values(10.0 + np.arange(n) + rng.standard_normal(n), freq="M")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fc = F.auto_forecast(s, horizon=6, freq="M")
    _assert_valid(fc, 6)
    assert fc.index.equals(future_index(s.index[-1], "M", 6))
    if n < 8:
        assert fc.backtest_mase is None and fc.method.startswith("fallback")
    else:
        assert fc.backtest_mase is not None
    if fc.model_scores:
        assert sum(ms.weight for ms in fc.model_scores) == pytest.approx(1.0)


@pytest.mark.parametrize("freq,n", [("D", 120), ("W", 120), ("M", 60), ("Q", 30), ("Y", 15)])
def test_default_horizon_and_future_index(registry, freq, n):
    s = from_values(50 + 0.3 * np.arange(n) + np.sin(np.arange(n)), freq=freq, start="2015-01-01")
    fc = F.auto_forecast(s)
    H = default_horizon(freq)
    _assert_valid(fc, H)
    expected = future_index(s.index[-1], freq, H)
    assert fc.index.equals(expected)
    for lv in fc.levels:
        assert fc.lower[lv].index.equals(expected) and fc.upper[lv].index.equals(expected)
    assert fc.levels == (80, 95)


def test_raw_irregular_input_is_prepared():
    s = sample("growing", n=60)
    raw = s.iloc[::-1].copy()            # unsorted
    raw.iloc[5] = np.nan                 # a gap
    fc = F.auto_forecast(raw, horizon=6)
    _assert_valid(fc, 6)
    assert fc.index[0] > s.index[-1]


def test_determinism(registry):
    s = sample("reviving", n=100, seed=3)
    a = F.auto_forecast(s, horizon=12, freq="M")
    b = F.auto_forecast(s.copy(), horizon=12, freq="M")
    pd.testing.assert_frame_equal(a.to_frame(), b.to_frame())
    assert a.method == b.method and a.backtest_mase == b.backtest_mase
    assert [(m.name, m.mase, m.weight) for m in a.model_scores] == \
        [(m.name, m.mase, m.weight) for m in b.model_scores]


@pytest.mark.parametrize("kind", ["growing", "seasonal", "fad", "emerging"])
def test_ensemble_weights_and_method(registry, kind):
    fc = F.auto_forecast(sample(kind, n=120), horizon=12, freq="M", top_k=3)
    _assert_valid(fc, 12)
    weights = [ms.weight for ms in fc.model_scores]
    assert sum(weights) == pytest.approx(1.0, abs=1e-9)
    used = [ms for ms in fc.model_scores if ms.weight > 0]
    assert 1 <= len(used) <= 3
    if len(used) == 1:
        assert fc.method == used[0].name and used[0].weight == pytest.approx(1.0)
        assert used[0] is fc.model_scores[0]
    else:
        assert fc.method.startswith("ensemble(")
        assert used == fc.model_scores[: len(used)]           # the top-k by MASE
        for ms in used:
            assert ms.name in fc.method
        ws = [ms.weight for ms in used]
        assert ws == sorted(ws, reverse=True)                  # inverse-MASE weights
    assert fc.backtest_mase is not None


def test_top_k_one_gives_single_best_model():
    fc = F.auto_forecast(sample("growing", n=120), horizon=12, freq="M", top_k=1,
                         candidates=list(BASELINES))
    assert fc.method == fc.model_scores[0].name
    assert fc.model_scores[0].weight == 1.0
    assert fc.backtest_mase == pytest.approx(fc.model_scores[0].mase)


def test_registered_perfect_model_is_selected(clean_registry):
    """A toy model that knows the true (noiseless, non-linear) curve must win."""

    def truth(t: np.ndarray) -> np.ndarray:
        return 50.0 + 10.0 * np.sin(2 * np.pi * t / 17.0) + 0.02 * t ** 2

    class Oracle(M.Forecaster):
        name = "oracle"
        min_obs = 1

        def _fit(self, y: np.ndarray, m: int) -> None:
            self._n = len(y)

        def _predict(self, h: int) -> np.ndarray:
            return truth(np.arange(self._n, self._n + h, dtype=float))

    M.register("oracle", Oracle)
    n, h = 150, 12
    s = from_values(truth(np.arange(n, dtype=float)), freq="M")
    fc = F.auto_forecast(s, horizon=h, freq="M")     # default candidates pick it up
    top = fc.model_scores[0]
    assert top.name == "oracle" and top.mase == pytest.approx(0.0, abs=1e-12)
    assert top.weight > 0.99
    np.testing.assert_allclose(fc.mean.to_numpy(), truth(np.arange(n, n + h, dtype=float)),
                               rtol=1e-6)
    assert "oracle" in fc.method


def test_failing_models_are_skipped_with_a_warning(clean_registry):
    class Broken(M.Forecaster):
        name = "broken"
        min_obs = 1

        def _fit(self, y: np.ndarray, m: int) -> None:
            raise RuntimeError("boom")

        def _predict(self, h: int) -> np.ndarray:  # pragma: no cover
            return np.zeros(h)

    class WrongShape(M.Forecaster):
        name = "wrong_shape"
        min_obs = 1

        def _fit(self, y: np.ndarray, m: int) -> None:
            pass

        def _predict(self, h: int) -> np.ndarray:
            return np.zeros(h + 1)

    M.register("broken", Broken)
    M.register("wrong_shape", WrongShape)
    s = sample("growing", n=60)
    with pytest.warns(RuntimeWarning) as record:
        fc = F.auto_forecast(s, horizon=6, freq="M")
    messages = " ".join(str(w.message) for w in record)
    assert "'broken'" in messages and "'wrong_shape'" in messages
    _assert_valid(fc, 6)
    names = {ms.name for ms in fc.model_scores}
    assert "broken" not in names and "wrong_shape" not in names
    # only broken / unknown candidates -> heuristic fallback, still a forecast
    with pytest.warns(RuntimeWarning):
        fb = F.auto_forecast(s, horizon=6, freq="M", candidates=["broken", "no_such_model"])
    _assert_valid(fb, 6)
    assert fb.backtest_mase is None and fb.method.startswith("fallback")


def test_argument_validation():
    s = sample("growing", n=40)
    with pytest.raises(ValueError):
        F.auto_forecast(s, horizon=0)
    for bad in ((0,), (100,), (80.5,), (-5,)):
        with pytest.raises(ValueError):
            F.auto_forecast(s, horizon=3, levels=bad)
    with pytest.raises(ValueError):
        F.auto_forecast(s, horizon=3, top_k=0)
    fc = F.auto_forecast(s, horizon=3, levels=(95, 50, 95), candidates=list(BASELINES))
    assert fc.levels == (95, 50)
    assert list(fc.to_frame().columns) == ["mean", "lo50", "hi50", "lo95", "hi95"]


def test_explicit_n_windows_and_long_history():
    s = sample("seasonal", n=1500, freq="D")
    fc = F.auto_forecast(s, horizon=14, n_windows=3, candidates=list(BASELINES))
    _assert_valid(fc, 14)
    assert fc.index.equals(future_index(s.index[-1], "D", 14))


def test_speed_daily_730(registry):
    s = sample("growing", n=730, freq="D")
    t0 = time.perf_counter()
    fc = F.auto_forecast(s)
    elapsed = time.perf_counter() - t0
    _assert_valid(fc, default_horizon("D"))
    assert elapsed < 5.0, f"auto_forecast took {elapsed:.2f}s"


# --------------------------------------------------------------------------- compute_outlook


def test_outlook_growing_sample():
    s = sample("growing", n=120)
    fc = F.auto_forecast(s, horizon=12, freq="M", candidates=["drift"])
    o = F.compute_outlook(_make_analysis(s, "M", "growing"), fc)
    assert isinstance(o, Outlook)
    assert o.direction == "up"
    assert o.expected_change_pct is not None and o.expected_change_pct > 2
    assert o.prob_increase > 0.6
    assert o.future_phase == "growing"
    assert o.scenarios["pessimistic"] <= o.scenarios["base"] <= o.scenarios["optimistic"]
    assert o.scenarios["base"] == pytest.approx(float(fc.mean.iloc[-1]))
    assert o.scenarios["pessimistic"] == pytest.approx(float(fc.lower[80].iloc[-1]))
    assert o.confidence in ("high", "medium", "low")
    assert o.future_phase in PHASES


def test_outlook_declining_sample():
    s = sample("declining", n=120)
    fc = F.auto_forecast(s, horizon=12, freq="M", candidates=["drift"])
    o = F.compute_outlook(_make_analysis(s, "M", "declining"), fc)
    assert o.direction == "down"
    assert o.expected_change_pct < -2
    assert o.prob_increase < 0.4
    assert o.future_phase == "declining"
    assert o.forecast_peak_date is None


def test_outlook_with_default_candidates_is_well_formed(registry):
    for kind, phase in (("growing", "growing"), ("seasonal", "stable"), ("fad", "declining"),
                        ("declining", "declining")):
        s = sample(kind, n=120)
        fc = F.auto_forecast(s, horizon=12, freq="M")
        o = F.compute_outlook(_make_analysis(s, "M", phase), fc)
        # (with the three baselines only, seasonal_naive cannot extrapolate a trend)
        if registry == "full" and kind == "growing":
            assert o.direction == "up" and o.prob_increase > 0.6, o
        if registry == "full" and kind == "declining":
            assert o.direction == "down" and o.prob_increase < 0.4, o
        assert o.direction in ("up", "down", "flat")
        assert 0.0 <= o.prob_increase <= 1.0
        assert o.future_phase in PHASES and o.confidence in ("high", "medium", "low")
        assert set(o.scenarios) == {"pessimistic", "base", "optimistic"}
        assert all(math.isfinite(v) for v in o.scenarios.values())
        if o.expected_change_pct is not None:
            assert math.isfinite(o.expected_change_pct)
            assert (o.prob_increase > 0.5) == (o.expected_change_pct > 0) or o.prob_increase == 0.5


def test_outlook_semantics_on_hand_built_forecast():
    s = from_values(np.full(48, 100.0) + np.tile([1.0, -1.0], 24), freq="M")
    an = _make_analysis(s, "M", "stable")
    cur = float(s.iloc[-12:].mean())                       # = 100

    # +10 % with sigma_H = 5 at horizon end
    mean = np.linspace(102, 110, 12)
    sd = np.linspace(2.0, 5.0, 12)
    o = F.compute_outlook(an, _hand_forecast(s, "M", mean, sd, backtest_mase=0.5))
    fut = float(mean[-12:].mean())
    assert o.expected_change_pct == pytest.approx((fut - cur) / cur * 100)
    assert o.prob_increase == pytest.approx(F.norm_cdf((fut - cur) / 5.0), abs=1e-9)
    assert o.direction == "up" and o.confidence == "high"
    assert o.scenarios["base"] == pytest.approx(110.0)
    assert o.scenarios["optimistic"] == pytest.approx(110 + 1.2815515655 * 5.0, rel=1e-6)

    # flat forecast -> flat, p = 0.5, stays stable
    flat = F.compute_outlook(an, _hand_forecast(s, "M", np.full(12, cur), np.full(12, 2.0)))
    assert flat.direction == "flat" and flat.prob_increase == pytest.approx(0.5)
    assert flat.future_phase == "stable" and flat.forecast_peak_date is None

    # small change within the noise band -> flat
    small = F.compute_outlook(an, _hand_forecast(s, "M", np.full(12, cur * 1.015), np.full(12, 2.0)))
    assert small.direction == "flat"


def test_outlook_peak_date_and_seasonal_guard():
    s = from_values(np.full(48, 100.0) + np.tile([1.0, -1.0], 24), freq="M")
    # forecast rises to a peak at step 5 and then falls by > 5 %
    bump = 100 + 20 * np.exp(-0.5 * ((np.arange(12) - 5) / 2.0) ** 2)
    fc = _hand_forecast(s, "M", bump, np.full(12, 2.0))
    o = F.compute_outlook(_make_analysis(s, "M", "growing"), fc)
    assert o.forecast_peak_date == fc.mean.index[5]
    # monotone forecast -> no peak
    mono = _hand_forecast(s, "M", np.linspace(101, 120, 12), np.full(12, 2.0))
    assert F.compute_outlook(_make_analysis(s, "M", "growing"), mono).forecast_peak_date is None
    # the same bump explained by the analysis' seasonal pattern is not a peak
    n = len(s)
    phase_pattern = 20 * np.exp(-0.5 * ((np.arange(12) - 5) / 2.0) ** 2)
    phase_pattern -= phase_pattern.mean()
    seas = pd.Series(phase_pattern[(np.arange(n) - n) % 12], index=s.index)
    seasonal_fc = _hand_forecast(s, "M", 100 + phase_pattern, np.full(12, 2.0))
    o2 = F.compute_outlook(_make_analysis(s, "M", "stable", seasonal=seas), seasonal_fc)
    assert o2.forecast_peak_date is None


def test_outlook_confidence_rules():
    s = from_values(100 + np.sin(np.arange(120.0)), freq="M")
    an = _make_analysis(s, "M", "stable")
    mean, sd = np.full(12, 100.0), np.full(12, 1.0)
    conf = lambda m: F.compute_outlook(an, _hand_forecast(s, "M", mean, sd, m)).confidence  # noqa: E731
    assert conf(None) == "low"
    assert conf(0.5) == "high"
    assert conf(1.0) == "medium"
    assert conf(1.6) == "low"
    short = from_values(100 + np.sin(np.arange(10.0)), freq="M")
    assert F.compute_outlook(_make_analysis(short, "M"), _hand_forecast(short, "M", mean, sd, 0.5)
                             ).confidence == "low"
    mid = from_values(100 + np.sin(np.arange(30.0)), freq="M")    # 30 < 3 seasons
    assert F.compute_outlook(_make_analysis(mid, "M"), _hand_forecast(mid, "M", mean, sd, 0.5)
                             ).confidence == "medium"


def test_outlook_zero_base_and_nonnegative_asymmetric_bands():
    s = from_values(np.r_[np.linspace(5, 0.5, 20), np.zeros(12)], freq="M")
    fc = F.auto_forecast(s, horizon=6, freq="M", candidates=["naive"])
    o = F.compute_outlook(_make_analysis(s, "M", "declining"), fc)
    assert o.expected_change_pct is None             # current level is 0
    assert 0.0 <= o.prob_increase <= 1.0
    assert o.direction == "flat" and o.future_phase == "stable"


@pytest.mark.parametrize("current,direction,strong,change,conf,expected", [
    ("emerging", "up", True, 80.0, "high", "emerging"),
    ("emerging", "up", True, 20.0, "high", "growing"),
    ("growing", "up", False, 5.0, "high", "growing"),
    ("declining", "up", True, 15.0, "medium", "reviving"),
    ("reviving", "up", False, 4.0, "medium", "reviving"),
    ("growing", "down", False, -3.0, "high", "peaking"),
    ("peaking", "down", True, -20.0, "high", "declining"),
    ("emerging", "down", True, -15.0, "high", "declining"),
    ("stable", "down", False, -4.0, "high", "declining"),
    ("growing", "flat", False, 0.5, "high", "peaking"),
    ("peaking", "flat", False, 0.5, "high", "peaking"),
    ("declining", "flat", False, -0.5, "high", "stable"),
    ("volatile", "flat", False, 0.0, "low", "volatile"),
    # UX-2: was 'stable' -- a smooth mean forecast cannot show the noise going away, so a
    # volatile series stays volatile whatever the forecast's direction or confidence
    ("volatile", "flat", False, 0.0, "high", "volatile"),
    ("volatile", "up", True, 30.0, "high", "volatile"),
    # UX-2: 'flat' only for its uncertainty (the base case moves +75 %), or with low
    # confidence, says nothing about the current move ending: the phase is kept
    ("emerging", "flat", False, 75.0, "low", "emerging"),
    ("emerging", "flat", False, 75.0, "medium", "emerging"),
    ("growing", "flat", False, 1.0, "low", "growing"),
    ("reviving", "flat", False, -6.0, "medium", "reviving"),
    ("reviving", "flat", False, -1.0, "medium", "stable"),
])
def test_future_phase_rules(current, direction, strong, change, conf, expected):
    assert F._future_phase(current, direction, strong, change, conf) == expected


def test_future_phase_flat_needs_a_settled_forecast():
    """UX-2 end to end: `sample volatile` read '변동성 장세 → 안정기', `sample emerging --freq Y
    --n 12` (+75 % base case, rated flat for its uncertainty) '태동기 → 정점/성숙기' and
    `sample reviving --freq Q --n 40` (80 % band ±41 % of the level) '재부상 → 안정기'."""
    from trendai.pipeline import run

    for name, freq, n in (("volatile", "M", 120), ("emerging", "Y", 12), ("reviving", "Q", 40)):
        rep = run(sample(name, n=n, freq=freq), name=name)
        assert rep.outlook.direction == "flat", (name, rep.outlook)
        assert rep.outlook.future_phase == rep.analysis.phase.label, (name, rep.outlook)
    # an explicit `settled` decides for a flat outlook
    assert F._future_phase("growing", "flat", False, 1.0, "high", settled=False) == "growing"
    assert F._future_phase("growing", "flat", False, 1.0, "high", settled=True) == "peaking"
    # a precise flat forecast of a declining series still projects the decline ending
    s = from_values(np.r_[np.linspace(200, 100, 36), np.full(24, 100.0)]
                    + np.tile([0.5, -0.5], 30), freq="M")
    o = F.compute_outlook(_make_analysis(s, "M", "declining"),
                          _hand_forecast(s, "M", np.full(12, 100.0), np.full(12, 2.0), 0.5))
    assert o.direction == "flat" and o.confidence != "low" and o.future_phase == "stable"
    # ... but not when the band is too wide to tell a continuing decline from a stop
    wide = F.compute_outlook(_make_analysis(s, "M", "declining"),
                             _hand_forecast(s, "M", np.full(12, 100.0), np.full(12, 30.0), 0.5))
    assert wide.direction == "flat" and wide.future_phase == "declining"


# --------------------------------------------------------------------------- adversarial regressions


def _zeros_then_ramp(seed: int = 3) -> np.ndarray:
    """A new trend taking off after a long run of zeros (typical for search interest)."""
    rng = np.random.default_rng(seed)
    ramp = np.maximum(0.0, np.cumsum(rng.uniform(0, 2, 24)) + rng.standard_normal(24))
    return np.r_[np.zeros(60), ramp]


def test_mase_is_unit_free_when_training_windows_are_constant():
    """All-zero training windows used to get the absolute scale 1.0, so MASE (and with it
    model ranking, backtest_mase and confidence) depended on the units of the data."""
    y = _zeros_then_ramp()
    ref = F.backtest(y, "drift", season_length=12, horizon=12, n_windows=5)
    for c in (1e-3, 1e3, 1e6):
        res = F.backtest(y * c, "drift", season_length=12, horizon=12, n_windows=5)
        assert res["mase"] == pytest.approx(ref["mase"], rel=1e-9)
        assert res["smape"] == pytest.approx(ref["smape"], rel=1e-9)
        assert res["mae"] == pytest.approx(ref["mae"] * c, rel=1e-9)
    # ordinary windows are untouched: each is scaled by mase_scale of its training part
    assert ref["mase"] < 50
    fcs = {c: F.auto_forecast(from_values(y * c, freq="M"), horizon=12, freq="M",
                              candidates=list(BASELINES)) for c in (1e-3, 1.0, 1e6)}
    for c, fc in fcs.items():
        assert fc.method == fcs[1.0].method
        assert fc.backtest_mase == pytest.approx(fcs[1.0].backtest_mase, rel=1e-9)
        assert [(m.name, round(m.weight, 12)) for m in fc.model_scores] == \
            [(m.name, round(m.weight, 12)) for m in fcs[1.0].model_scores]
        np.testing.assert_allclose(fc.upper[80].to_numpy(), fcs[1.0].upper[80].to_numpy() * c,
                                   rtol=1e-9)


def test_window_scales_floor_only_binds_for_degenerate_windows():
    y = 10 + np.sin(np.arange(60.0)) + 0.1 * np.arange(60.0)
    origins = [30, 40, 50]
    np.testing.assert_allclose(F._window_scales(y, origins, 1),
                               [F.mase_scale(y[:o], 1) for o in origins])
    flat = np.r_[np.full(40, 5.0), 5 + np.arange(20.0)]
    sc = F._window_scales(flat, [20, 30, 45], 1)
    # the floor comes from the history before the latest origin (no look-ahead into the
    # target period of the latest window; it used to be the whole series)
    floor = F._SCALE_FLOOR_FRAC * F.mase_scale(flat[:45], 1)
    assert sc[0] == pytest.approx(floor) and sc[1] == pytest.approx(floor)
    assert sc[2] == pytest.approx(F.mase_scale(flat[:45], 1))


def test_window_scales_do_not_look_past_the_latest_origin():
    """Changing observations at or after the latest origin must not change any scale
    (it used to rescale every window through the floor: [0.15, 0.22, 0.28] -> 2.6 each)."""
    y = np.r_[np.full(20, 5.0), 5 + np.cumsum(np.random.default_rng(0).standard_normal(20))]
    plan = F._make_plan(len(y), 1, 6, 3)
    ref = F._window_scales(y, plan.origins, 1)
    for i in range(plan.origins[-1], len(y)):
        y2 = y.copy()
        y2[i] += 1000.0
        np.testing.assert_array_equal(F._window_scales(y2, plan.origins, 1), ref)


def test_rms_does_not_overflow():
    x = np.array([[3e200, -4e200], [0.0, 0.0]])
    assert F._rms(x) == pytest.approx(math.sqrt((9 + 16) / 4) * 1e200, rel=1e-12)
    np.testing.assert_allclose(F._rms(x, axis=1), [math.sqrt(12.5) * 1e200, 0.0], rtol=1e-12)
    np.testing.assert_allclose(F._rms(x, axis=0), [3e200 / math.sqrt(2), 4e200 / math.sqrt(2)],
                               rtol=1e-12)
    assert F._rms(np.zeros(3)) == 0.0


@pytest.mark.parametrize("scale", [1e-200, 1e-9, 1e9, 1e200])
def test_extreme_magnitudes_are_finite_and_scale_equivariant(scale):
    """|y| > ~1e154 used to overflow (err**2) into inf scores and inf interval bounds."""
    rng = np.random.default_rng(7)
    y = 100.0 * (1.0 + 0.05 * np.arange(60) / 60 + 0.02 * rng.standard_normal(60))
    base = F.auto_forecast(from_values(y, freq="M"), horizon=6, freq="M",
                           candidates=list(BASELINES))
    fc = F.auto_forecast(from_values(y * scale, freq="M"), horizon=6, freq="M",
                         candidates=list(BASELINES))
    _assert_valid(fc, 6)
    assert fc.method == base.method
    assert fc.backtest_mase == pytest.approx(base.backtest_mase, rel=1e-6)
    for lv in fc.levels:
        np.testing.assert_allclose(fc.upper[lv].to_numpy() / scale, base.upper[lv].to_numpy(),
                                   rtol=1e-6)
        np.testing.assert_allclose(fc.lower[lv].to_numpy() / scale, base.lower[lv].to_numpy(),
                                   rtol=1e-6)
    # too short to backtest -> heuristic fallback, also finite
    short = F.auto_forecast(from_values(y[:5] * scale, freq="M"), horizon=6, freq="M")
    _assert_valid(short, 6)


def test_huge_magnitudes_with_default_candidates_are_finite(registry):
    rng = np.random.default_rng(1)
    y = 1e200 * (1.0 + 0.01 * rng.standard_normal(48))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fc = F.auto_forecast(from_values(y, freq="M"), horizon=6, freq="M")
    _assert_valid(fc, 6)
    o = F.compute_outlook(_make_analysis(from_values(y, freq="M"), "M", "stable"), fc)
    assert math.isfinite(o.prob_increase) and all(math.isfinite(v) for v in o.scenarios.values())


def test_backtest_has_no_lookahead():
    """Window forecasts may only depend on data before their origin."""
    rng = np.random.default_rng(11)
    y = 50 + np.cumsum(rng.standard_normal(80))
    res = F.backtest(y, "drift", season_length=12, horizon=8, n_windows=4)
    for i, o in enumerate(res["origins"]):
        y2 = y.copy()
        y2[o:] = 1e6 + rng.standard_normal(len(y) - o)       # garbage after the origin
        res2 = F.backtest(y2, "drift", season_length=12, horizon=8, n_windows=4)
        assert res2["origins"] == res["origins"]
        np.testing.assert_array_equal(res2["forecasts"][i], res["forecasts"][i])


def test_auto_forecast_does_not_mutate_its_input():
    s = sample("growing", n=60)
    raw = s.copy()
    raw.iloc[3] = np.nan
    before = raw.copy()
    F.auto_forecast(raw, horizon=6, candidates=list(BASELINES))
    pd.testing.assert_series_equal(raw, before)


_EDGE_SERIES = {
    "constant": lambda n: np.full(n, 5.0),
    "zeros": lambda n: np.zeros(n),
    "negative": lambda n: -100.0 + np.sin(np.arange(n, dtype=float)),
    "mixed_sign": lambda n: np.random.default_rng(1).standard_normal(n),
    "step": lambda n: np.r_[np.full(n // 2, 10.0), np.full(n - n // 2, 50.0)],
    "spike": lambda n: np.r_[np.full(n // 2, 10.0), [1000.0], np.full(n - n // 2 - 1, 10.0)],
}


@pytest.mark.parametrize("kind", sorted(_EDGE_SERIES))
@pytest.mark.parametrize("freq", ["D", "W", "M", "Q", "Y"])
def test_edge_series_every_frequency(registry, kind, freq):
    y = _EDGE_SERIES[kind](30)
    s = from_values(y, freq=freq, start="2010-01-01")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fc = F.auto_forecast(s)
    H = default_horizon(freq)
    _assert_valid(fc, H)
    assert fc.index.equals(future_index(s.index[-1], freq, H))
    assert fc.nonnegative == bool(y.min() >= 0)
    if fc.nonnegative:
        assert fc.lower[80].min() >= 0
    assert sum(ms.weight for ms in fc.model_scores) == pytest.approx(1.0)
    o = F.compute_outlook(_make_analysis(s, freq, "stable"), fc)
    assert o.direction in ("up", "down", "flat") and o.future_phase in PHASES
    assert 0.0 <= o.prob_increase <= 1.0
    assert all(math.isfinite(v) for v in o.scenarios.values())
    assert o.expected_change_pct is None or math.isfinite(o.expected_change_pct)
    if kind == "constant":
        assert o.direction == "flat" and o.expected_change_pct == pytest.approx(0.0, abs=1e-6)


def test_final_fit_failure_uses_next_best_model(clean_registry):
    """If the chosen model breaks on the full history, the next-ranked model is used
    (not the crude drift+naive fallback)."""

    def truth(t: np.ndarray) -> np.ndarray:
        return 10.0 + 0.5 * t + 3.0 * np.sin(t)

    class ShortOnly(M.Forecaster):
        name = "short_only"
        min_obs = 1

        def _fit(self, y: np.ndarray, m: int) -> None:
            if len(y) >= 120:
                raise RuntimeError("cannot handle long input")
            self._n = len(y)

        def _predict(self, h: int) -> np.ndarray:
            return truth(np.arange(self._n, self._n + h, dtype=float))

    M.register("short_only", ShortOnly)
    s = from_values(truth(np.arange(120.0)), freq="M")
    with pytest.warns(RuntimeWarning, match="short_only"):
        fc = F.auto_forecast(s, horizon=6, freq="M", candidates=["short_only", "drift", "naive"])
    _assert_valid(fc, 6)
    assert fc.model_scores[0].name == "short_only" and fc.model_scores[0].weight == 0.0
    assert not fc.method.startswith("fallback")
    assert fc.method == "drift"
    used = {ms.name: ms for ms in fc.model_scores}
    assert used["drift"].weight == 1.0
    assert fc.backtest_mase == pytest.approx(used["drift"].mase)


# ---- compute_outlook: partial seasons and peaks


def _weekly_seasonal_analysis(n: int = 156, amp: float = 8.0, noise: float = 0.0, seed: int = 0,
                              level: float = 100.0):
    t = np.arange(n, dtype=float)
    rng = np.random.default_rng(seed)
    y = level + amp * np.sin(2 * np.pi * t / 52) + noise * rng.standard_normal(n)
    s = from_values(y, freq="W")
    if noise > 0:   # a noisy per-week seasonal estimate (3 points per phase), as a decomposition gives
        ph = np.arange(n) % 52
        est = np.array([np.mean(y[ph == k]) for k in range(52)])
        seas = pd.Series((est - est.mean())[ph], index=s.index)
    else:
        seas = pd.Series(amp * np.sin(2 * np.pi * t / 52), index=s.index)
    return s, _make_analysis(s, "W", "stable", seasonal=seas)


def test_outlook_partial_season_is_not_reported_as_a_trend():
    """Weekly data: default horizon 26 < season 52.  The first half of next year's season
    (a seasonal high here) used to be compared with a full-season average: +10 %, 'up'."""
    s, an = _weekly_seasonal_analysis()
    n, H = len(s), 26
    k = np.arange(n, n + H, dtype=float)
    seasonal_fc = 100 + 8 * np.sin(2 * np.pi * k / 52)
    o = F.compute_outlook(an, _hand_forecast(s, "W", seasonal_fc, np.full(H, 1.0)))
    assert abs(o.expected_change_pct) < 0.5
    assert o.direction == "flat" and o.future_phase == "stable" and o.forecast_peak_date is None
    assert 0.4 < o.prob_increase < 0.6
    # the chosen models forecast no seasonality: nothing must be subtracted either
    flat = F.compute_outlook(an, _hand_forecast(s, "W", np.full(H, 100.0), np.full(H, 1.0)))
    assert abs(flat.expected_change_pct) < 0.5 and flat.direction == "flat"
    # a genuine trend on top of the seasonal pattern is measured without the seasonal bias
    trend_fc = seasonal_fc + 0.5 * np.arange(1, H + 1)
    up = F.compute_outlook(an, _hand_forecast(s, "W", trend_fc, np.full(H, 1.0)))
    assert up.expected_change_pct == pytest.approx(0.5 * np.mean(np.arange(1, H + 1)), abs=0.3)
    assert up.direction == "up"
    # whole-season windows (here: monthly, horizon 12) keep the plain contract formula
    sm = from_values(100 + 5 * np.sin(2 * np.pi * np.arange(48) / 12), freq="M")
    seas_m = pd.Series(5 * np.sin(2 * np.pi * np.arange(48) / 12), index=sm.index)
    fc_m = _hand_forecast(sm, "M", np.linspace(101, 112, 12), np.full(12, 1.0))
    om = F.compute_outlook(_make_analysis(sm, "M", "stable", seasonal=seas_m), fc_m)
    expected = (np.mean(np.linspace(101, 112, 12)) - sm.iloc[-12:].mean()) / sm.iloc[-12:].mean()
    assert om.expected_change_pct == pytest.approx(expected * 100, rel=1e-12)


def test_outlook_weekly_seasonal_with_real_forecast_is_flat():
    s, an = _weekly_seasonal_analysis(amp=12.0, noise=2.0, seed=4, level=50.0)
    fc = F.auto_forecast(s, freq="W", candidates=["seasonal_naive"])
    assert fc.horizon == 26
    o = F.compute_outlook(an, fc)
    raw = (fc.mean.mean() - s.iloc[-52:].mean()) / s.iloc[-52:].mean() * 100
    assert raw > 10                                   # what the old code reported
    assert abs(o.expected_change_pct) < 3 and o.direction == "flat", o
    assert o.forecast_peak_date is None
    assert o.future_phase == "stable"


def test_outlook_peak_requires_a_rise_and_handles_negative_values():
    s = from_values(np.full(48, 100.0) + np.tile([1.0, -1.0], 24), freq="M")
    an = _make_analysis(s, "M", "declining")
    # a noise-sized wiggle at the start of a declining forecast is not a peak
    wiggle = np.r_[100.0, 100.3, 100.5, np.linspace(99, 88, 9)]
    o = F.compute_outlook(an, _hand_forecast(s, "M", wiggle, np.full(12, 2.0)))
    assert o.forecast_peak_date is None and o.direction == "down"
    # ... but a clear rise to an inner maximum followed by a fall is
    hump = np.r_[101.0, 104.0, 108.0, 110.0, 107.0, 103.0, np.linspace(100, 95, 6)]
    fc = _hand_forecast(s, "M", hump, np.full(12, 2.0))
    assert F.compute_outlook(an, fc).forecast_peak_date == fc.mean.index[3]
    # negative-valued series (e.g. a balance): the 5 % fall is relative to |peak|
    neg = from_values(np.full(48, -100.0) + np.tile([1.0, -1.0], 24), freq="M")
    neg_hump = -100 + 20 * np.exp(-0.5 * ((np.arange(12) - 5) / 2.0) ** 2)
    fc_neg = _hand_forecast(neg, "M", neg_hump, np.full(12, 2.0))
    o_neg = F.compute_outlook(_make_analysis(neg, "M", "stable"), fc_neg)
    assert o_neg.forecast_peak_date == fc_neg.mean.index[5]
    # round 5 (G1): a negative current level has no percentage ('negative', the analysis rule); the
    # outlook is still decided on the two levels (was: a finite % with |current| in the denominator)
    lv_neg = F.outlook_levels(_make_analysis(neg, "M", "stable"), fc_neg)
    assert o_neg.expected_change_pct is None and lv_neg.undefined == "negative"
    assert o_neg.direction == "up" and o_neg.prob_increase > 0.9 and lv_neg.future > lv_neg.current


def test_outlook_weekly_peak_detection_with_noisy_seasonal_estimate():
    s, an = _weekly_seasonal_analysis(amp=12.0, noise=2.0, seed=5, level=50.0)
    n, H = len(s), 26
    k = np.arange(n, n + H, dtype=float)
    seas_fc = 50 + 12 * np.sin(2 * np.pi * k / 52)
    # seasonal pattern only (the seasonal high is mid-horizon): no peak
    assert F.compute_outlook(an, _hand_forecast(s, "W", seas_fc, np.full(H, 1.5))
                             ).forecast_peak_date is None
    # a genuine trend peak on top of the seasonal pattern is still found
    bump = 15 * np.exp(-0.5 * ((np.arange(H) - 8) / 3.0) ** 2) - 0.6 * np.arange(H)
    fc = _hand_forecast(s, "W", seas_fc + bump, np.full(H, 1.5))
    assert F.compute_outlook(an, fc).forecast_peak_date is not None


def test_seasonal_share_and_smooth_helpers():
    t = np.arange(26, dtype=float)
    s = 8 * np.sin(2 * np.pi * t / 52)
    assert F._seasonal_share(100 + s, s, 52) == pytest.approx(1.0)
    assert F._seasonal_share(np.full(26, 3.0), s, 52) == pytest.approx(0.0, abs=1e-9)
    assert F._seasonal_share(1 + 0.3 * t, s, 52) == pytest.approx(0.0, abs=1e-9)
    assert F._seasonal_share(100 + s, np.zeros(26), 52) == 1.0
    assert F._seasonal_share(np.arange(3.0), np.ones(3), 52) == 1.0
    x = np.array([0.0, 3.0, 0.0, 3.0, 0.0])
    np.testing.assert_allclose(F._smooth(x, 3), [1.5, 1.0, 2.0, 1.0, 1.5])
    np.testing.assert_array_equal(F._smooth(x, 1), x)
    for n in range(1, 30):                  # never changes the length, even for w > n
        for w in (1, 2, 3, 13, 27):
            assert len(F._smooth(np.arange(float(n)), w)) == n


@pytest.mark.parametrize("H", [3, 4, 6, 8, 12, 13])
def test_outlook_short_weekly_horizons(H):
    """Short (even/odd) weekly horizons exercise the smoothing and seasonal-share edge cases."""
    s, an = _weekly_seasonal_analysis(amp=12.0, noise=2.0, seed=4, level=50.0)
    fc = F.auto_forecast(s, horizon=H, freq="W", candidates=list(BASELINES))
    o = F.compute_outlook(an, fc)
    assert o.direction in ("up", "down", "flat") and 0.0 <= o.prob_increase <= 1.0
    assert o.expected_change_pct is not None and math.isfinite(o.expected_change_pct)


# --------------------------------------------------------------------------- fixer regressions
# (skill / confidence, short-series intervals, plausibility guard, intermittent demand,
#  spikes, history cap, seasonal layout, ensemble disagreement, outlook direction)


def test_t_distribution_helpers():
    assert F.t_ppf(0.975, 1) == pytest.approx(12.706204736, rel=1e-8)
    assert F.t_ppf(0.9, 4) == pytest.approx(1.533206274, rel=1e-8)
    assert F.t_ppf(0.975, 30) == pytest.approx(2.042272456, rel=1e-8)
    assert F.t_ppf(0.025, 5) == pytest.approx(-F.t_ppf(0.975, 5))
    assert F.t_ppf(0.975, math.inf) == pytest.approx(F.norm_ppf(0.975))
    assert F.t_ppf(0.975, 1e9) == pytest.approx(F.norm_ppf(0.975), rel=1e-6)
    for df in (1.0, 2.5, 7.0, 40.0):
        for p in (0.6, 0.9, 0.975, 0.995):
            assert F.t_cdf(F.t_ppf(p, df), df) == pytest.approx(p, abs=1e-10)
    for bad in ((0.0, 3), (1.0, 3), (0.5, 0), (0.5, -1)):
        with pytest.raises(ValueError):
            F.t_ppf(*bad)


@pytest.mark.parametrize("n", [2, 3, 5, 7])
def test_fallback_reports_no_fake_scores(registry, n):
    """Too short to backtest: no model scores (a naive 'MASE' of 10-1000 used to be
    reported and narrated), no skill, no backtest MASE."""
    s = from_values(10.0 + np.arange(n) + np.random.default_rng(n).standard_normal(n), freq="M")
    fc = F.auto_forecast(s, horizon=6, freq="M")
    assert fc.method.startswith("fallback")
    assert fc.model_scores == [] and fc.skill is None and fc.backtest_mase is None
    o = F.compute_outlook(_make_analysis(s, "M"), fc)
    assert o.confidence == "low"


def test_skill_definition_and_benchmarks_are_always_backtested():
    # a candidate that equals a benchmark on every window -> skill = its MAE / best benchmark MAE
    rng = np.random.default_rng(3)
    y = 100 + np.cumsum(rng.standard_normal(60))
    fc = F.auto_forecast(from_values(y, freq="Y"), horizon=5, freq="Y", candidates=["naive"])
    plan = F._make_plan(60, 1, 5, F._default_n_windows(60), prefer_seasonal=False)
    act = F._actuals(y, plan)
    scales = F._window_scales(y, plan.origins, 1)
    w = F._recency_weights(plan)
    mae = {nm: F._scores(F._window_forecasts(y, nm, 1, plan), act, scales, w)["mae"]
           for nm in ("naive", "drift")}
    assert fc.skill == pytest.approx(mae["naive"] / min(mae.values()), rel=1e-9)
    assert [ms.name for ms in fc.model_scores] == ["naive"]
    # drift as the only candidate on a purely seasonal series: seasonal naive (not a
    # candidate) is backtested anyway and shows that drift is much worse than naive
    s = sample("seasonal", n=96)
    fc2 = F.auto_forecast(s, horizon=12, freq="M", candidates=["drift"])
    assert fc2.skill is not None and fc2.skill > 1.5
    assert [ms.name for ms in fc2.model_scores] == ["drift"]
    # a good forecast of a trending series beats naive clearly
    fc3 = F.auto_forecast(sample("growing", n=120), horizon=12, freq="M")
    assert fc3.skill is not None and 0 < fc3.skill < 0.8


def test_skill_is_finite_on_degenerate_series(registry):
    for y in (np.full(40, 5.0), np.zeros(40), 2.0 + 0.5 * np.arange(40.0)):
        fc = F.auto_forecast(from_values(y, freq="M"), horizon=6, freq="M")
        assert fc.skill is not None and math.isfinite(fc.skill) and 0 <= fc.skill <= 100


def test_confidence_uses_skill_and_interval_width():
    s = from_values(100 + np.sin(np.arange(120.0)), freq="M")
    an = _make_analysis(s, "M", "stable")
    mean = np.full(12, 100.0)

    def conf(skill, mase=0.5, sd=1.0):
        fc = _hand_forecast(s, "M", mean, np.full(12, sd), mase)
        fc.skill = skill
        return F.compute_outlook(an, fc).confidence

    assert conf(0.5) == "high"
    assert conf(0.5, mase=1.9) == "high"      # multi-step MASE > 1.5 is no reason for 'low'
    assert conf(0.9) == "medium"
    assert conf(1.2, mase=0.3) == "low"       # worse than a naive benchmark
    assert conf(0.5, sd=25.0) == "medium"     # 80 % band of +-32 %: at most medium
    assert conf(0.5, sd=90.0) == "low"        # band wider than the level itself
    assert conf(None, mase=None) == "low"


def test_confidence_labels_on_samples():
    """breakout beats the best naive benchmark by ~50 % (was 'low' because its multi-step
    MASE is > 1.5); volatile is a coin flip with a +-36 % band (was 'high'); the fad's
    collapse is not forecast with 'high' confidence."""
    expect = {"breakout": ("high", "medium"), "volatile": ("medium", "low"),
              "fad": ("medium", "low"), "growing": ("high",)}
    phase = {"breakout": "growing", "volatile": "volatile", "fad": "declining",
             "growing": "growing"}
    for nm, ok in expect.items():
        s = sample(nm)
        fc = F.auto_forecast(s, horizon=12, freq="M")
        o = F.compute_outlook(_make_analysis(s, "M", phase[nm]), fc)
        assert o.confidence in ok, (nm, o.confidence, fc.skill)


def test_short_series_interval_coverage():
    """n = 12..24 with the full registry: 80/95 % bands used to cover ~0.6-0.75 / ~0.72-0.9;
    tiny backtests are now shrunk towards a random-walk prior and widened with t-quantiles."""

    def proc(kind: str, n: int, rng: np.random.Generator) -> np.ndarray:
        t = np.arange(n, dtype=float)
        e = rng.standard_normal(n)
        if kind == "rw":
            return 100 + 2 * np.cumsum(e)
        if kind == "rw_drift":
            return 100 + t + np.cumsum(e)
        if kind == "linear":
            return 50 + 0.8 * t + 3 * e
        if kind == "ar1":
            x = np.zeros(n)
            for i in range(1, n):
                x[i] = 0.7 * x[i - 1] + e[i]
            return 100 + 5 * x
        return 100 + 10 * e

    cover = {80: [], 95: []}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for kind in ("rw", "rw_drift", "linear", "ar1", "white"):
            for ntr in (12, 16, 20, 24):
                for seed in range(4):
                    rng = np.random.default_rng(1000 * ntr + seed + 17 * len(kind))
                    y = proc(kind, ntr + 12, rng)
                    fc = F.auto_forecast(from_values(y[:ntr], freq="M"), horizon=12, freq="M")
                    for lv in cover:
                        cover[lv].append((y[ntr:] >= fc.lower[lv].to_numpy())
                                         & (y[ntr:] <= fc.upper[lv].to_numpy()))
    c80, c95 = float(np.mean(cover[80])), float(np.mean(cover[95]))
    assert 0.75 <= c80 <= 0.97, c80
    assert 0.88 <= c95 <= 0.995, c95


def test_very_short_trend_series_widen_beyond_the_backtest_horizon():
    """Monthly n = 10, h = 12, true model linear trend + noise: the 80/95 % bands at h = 12
    covered 0.58 / 0.73 (the 2-step backtest's b could drop to 0.1)."""
    idx = pd.date_range("2024-01-01", periods=10, freq="MS")
    hit80 = hit95 = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for seed in range(40):
            rng = np.random.default_rng(seed)
            y = 50 + 1.7 * np.arange(22) + rng.normal(0, 2, 22)
            fc = F.auto_forecast(pd.Series(y[:10], index=idx), horizon=12, freq="M")
            hit80 += fc.lower[80].iloc[-1] <= y[-1] <= fc.upper[80].iloc[-1]
            hit95 += fc.lower[95].iloc[-1] <= y[-1] <= fc.upper[95].iloc[-1]
    assert hit80 / 40 >= 0.75 and hit95 / 40 >= 0.9, (hit80, hit95)
    # a 5-year forecast from 8 yearly points is no +-3 % band any more
    s = sample("growing", n=13, freq="Y", seed=0)
    fc = F.auto_forecast(s.iloc[:8], horizon=5, freq="Y")
    assert (fc.upper[95].iloc[-1] - fc.mean.iloc[-1]) / fc.mean.iloc[-1] > 0.1


def test_plausibility_guard_stops_explosive_extrapolation():
    """20 daily points of an S-curve's take-off: forecasts reached 9-100x the observed
    maximum within 30 days (log-linear / Holt compounding the last burst)."""
    for seed in range(5):
        s = sample("emerging", freq="D", n=20, seed=seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fc = F.auto_forecast(s)
        assert fc.mean.max() < 6 * s.max(), (seed, fc.mean.max(), fc.method)
    # envelope properties
    line = 2.0 + 0.5 * np.arange(30.0)
    up, dn = F._move_envelope(line, 1, 40)
    k = np.arange(1, 41)
    assert np.all(up >= 0.5 * k) and np.all(dn > 0)          # continuing the line passes
    np.testing.assert_allclose(F._guard(line[-1] + 0.5 * k, (line[-1], up, dn)), line[-1] + 0.5 * k)
    burst = np.r_[np.full(15, 5.0) + np.sin(np.arange(15.0)), 5 * 1.6 ** np.arange(1, 6)]
    up, dn = F._move_envelope(burst, 1, 30, nonnegative=True)
    explosive = burst[-1] * 1.6 ** np.arange(1, 31)
    g = F._guard(explosive, (burst[-1], up, dn))
    assert g[-1] < 0.01 * explosive[-1] and np.all(np.diff(g) >= 0)
    assert np.all(np.isinf(dn))                              # drops of >= 0 data: bounded by 0
    assert F._move_envelope(np.array([1.0, 2.0]), 1, 5) is None


def test_intermittent_demand_is_not_anchored_on_the_last_observation():
    """Spare-part demand (P(sale) = 0.15/week, 1-5 units, mean 0.45/week): naive/drift won
    on MAE and the 26-week total was ~6 when the last week was 0 and ~28 when it was not
    (truth 11.7); the 95 % band excluded 0 and the outlook claimed +969 %."""
    totals = {True: [], False: []}
    for seed in range(12):
        rng = np.random.default_rng(seed)
        y = np.where(rng.random(150) < 0.15, rng.integers(1, 6, 150), 0).astype(float)
        s = pd.Series(y, index=pd.date_range("2021-01-03", periods=150, freq="W-SUN"))
        fc = F.auto_forecast(s, freq="W")
        _assert_valid(fc, 26)
        totals[bool(y[-1] == 0)].append(float(fc.mean.sum()))
        assert fc.lower[95].iloc[0] == 0.0
        assert not any(nm in fc.method for nm in ("naive", "drift", "analog_knn"))
        o = F.compute_outlook(_make_analysis(s, "W", "stable"), fc)
        assert o.expected_change_pct is None and o.direction == "flat"
    for zero_last, tot in totals.items():
        if tot:
            assert 4.0 <= float(np.median(tot)) <= 25.0, (zero_last, tot)


def test_is_intermittent_and_clean_spikes_helpers():
    rng = np.random.default_rng(0)
    y = np.where(rng.random(100) < 0.2, 3.0, 0.0)
    assert F._is_intermittent(y, 52)
    assert not F._is_intermittent(np.r_[np.zeros(60), 1 + np.arange(40.0)], 12)  # took off
    assert not F._is_intermittent(y - 1.0, 52)                                    # negatives
    assert not F._is_intermittent(np.zeros(50), 12)                               # all zero
    # isolated spikes are replaced; the last point, level shifts and short seasonal series not
    t = np.arange(60.0)
    base = 100 + 10 * np.sin(2 * np.pi * t / 12) + np.random.default_rng(1).normal(0, 1, 60)
    spiky = base.copy()
    spiky[[20, 41]] *= 1.8
    cleaned, k = F._clean_spikes(spiky, 12)
    assert k == 2 and np.max(np.abs(cleaned - base)) < 6
    last = base.copy()
    last[-1] *= 1.8
    assert F._clean_spikes(last, 12)[1] == 0
    shift = np.r_[base[:40], base[40:] + 40]
    assert F._clean_spikes(shift, 12)[1] == 0
    assert F._clean_spikes(spiky[:30], 12)[1] == 0             # < 3 seasons: peaks look like spikes


def test_one_off_spikes_are_not_projected_as_seasonality():
    """Monthly page views with a few one-off news spikes (flagged as anomalies by the
    analysis): seasonal models used to project them into next year (Dec +24 % YoY)."""
    rng = np.random.default_rng(5)
    t = np.arange(48.0)
    clean = 1000 * np.exp(0.01 * t + 0.15 * np.cos(2 * np.pi * (t - 11) / 12)
                          + rng.normal(0, 0.02, 48))
    spiky = clean.copy()
    spiky[[18, 35, 43]] *= (1.8, 1.6, 1.7)               # Jul, Dec, Aug of different years
    fc_c = F.auto_forecast(from_values(clean, freq="M"), horizon=12, freq="M")
    fc_s = F.auto_forecast(from_values(spiky, freq="M"), horizon=12, freq="M")
    rel = np.abs(fc_s.mean.to_numpy() / fc_c.mean.to_numpy() - 1)
    assert rel.max() < 0.06, rel.round(3)


def test_final_fit_uses_the_backtested_history():
    """The backtest used the last 400 points but the final fit up to 1000, so long series
    got forecasts the backtest never scored (breakout, daily 700 points: test MAE 22 vs 1.1
    with the last 400 points, 0 % coverage)."""
    s = sample("breakout", n=500, freq="D", seed=0)
    a = F.auto_forecast(s, horizon=30, freq="D")
    b = F.auto_forecast(s.iloc[-400:], horizon=30, freq="D")
    assert a.method == b.method
    pd.testing.assert_frame_equal(a.to_frame(), b.to_frame())


def test_two_to_two_and_a_half_seasons_keep_the_seasonal_layout():
    """Weekly n = 120 (2.3 seasons) fell through to the non-seasonal backtest layout, whose
    short windows excluded every seasonal model: a flat forecast 3x worse than seasonal naive."""
    plan = F._make_plan(120, 52, 26, F._default_n_windows(120), prefer_seasonal=True)
    assert plan.seasonal and plan.min_train >= 104 and plan.n_windows >= 2
    assert not F._make_plan(120, 52, 26, F._default_n_windows(120)).seasonal
    ratios = []
    for freq, ntr, h, m in (("W", 120, 26, 52), ("D", 28, 30, 7)):
        for seed in (0, 1):
            s = sample("seasonal", n=ntr + h, freq=freq, seed=seed)
            y = s.to_numpy()
            fc = F.auto_forecast(s.iloc[:ntr], horizon=h, freq=freq)
            sn = M.make_model("seasonal_naive").fit(y[:ntr], m).predict(h)
            ratios.append(np.mean(np.abs(y[ntr:] - fc.mean.to_numpy()))
                          / np.mean(np.abs(y[ntr:] - sn)))
            assert np.ptp(fc.mean.to_numpy()) > 0.5 * np.ptp(sn), (freq, seed, fc.method)
    assert float(np.exp(np.mean(np.log(ratios)))) < 1.3, np.round(ratios, 2)


def test_intervals_cover_ensemble_member_disagreement():
    """An explosive member dragged the mean to 4.5x the training maximum under a +-10 %
    band (members at h = 30: 206, 1195, 406; actual 200.6)."""
    t = np.arange(90.0)
    y = 5 + 200 / (1 + np.exp(-(t - 54) / 9)) + np.random.default_rng(1192).standard_normal(90)
    fc = F.auto_forecast(pd.Series(y[:60], pd.date_range("2015-01-05", periods=60, freq="D")),
                         horizon=30, freq="D")
    assert fc.lower[95].iloc[-1] <= y[-1] <= fc.upper[95].iloc[-1]
    members = {"a": np.array([1.0, 2.0, 3.0]), "b": np.array([1.0, 4.0, 9.0])}
    sp = F._member_spread(members, {"a": 0.5, "b": 0.5})
    np.testing.assert_allclose(sp * F.norm_ppf(0.975) * 2, [0.0, 2.0, 6.0])
    assert np.all(F._member_spread(members, {"a": 1.0}) == 0)


def test_direction_needs_a_change_beyond_the_horizon_uncertainty():
    """A flat (random-walk) forecast whose last value sits 2 % below the recent mean was
    called 'down' (prob_increase 0.43) next to an 80 % band of 72..99."""
    s = from_values(np.r_[np.full(47, 100.0) + np.tile([1.0, -1.0], 24)[:47], 96.0], freq="M")
    an = _make_analysis(s, "M", "stable")
    cur = float(s.iloc[-12:].mean())
    flat = _hand_forecast(s, "M", np.full(12, 96.0), np.linspace(2.0, 10.0, 12))
    o = F.compute_outlook(an, flat)
    assert (96.0 - cur) / cur < -0.03                     # beyond the old 2 % threshold
    assert o.direction == "flat" and 0.3 <= o.prob_increase < 0.5
    # the same change with a tight band is a direction
    tight = F.compute_outlook(an, _hand_forecast(s, "M", np.full(12, 96.0), np.full(12, 0.5)))
    assert tight.direction == "down"
    # the reported case: a business-day random walk whose forecast is flat at the last value
    rng = np.random.default_rng(0)
    raw = pd.Series((100 * np.exp(np.cumsum(rng.normal(0.0002, 0.018, 504)))).round(2),
                    index=pd.bdate_range("2023-01-02", periods=504))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")          # business days -> calendar days (DataWarning)
        stock, f = prepare(raw)
        fc = F.auto_forecast(stock, freq=f)
    o = F.compute_outlook(_make_analysis(stock, f, "stable"), fc)
    assert o.direction == "flat" and 1 / 3 <= o.prob_increase <= 2 / 3, o


def test_prob_increase_compares_seasonally_comparable_levels():
    """prob_increase is P(future level > current level) for the same season-mean levels as
    expected_change_pct (documented), not P(horizon-end value > current mean): a forecast
    that repeats last year's pattern ends in a seasonal low but has no level change."""
    t = np.arange(48.0)
    pattern = 10 * np.sin(2 * np.pi * (t + 10) / 12)         # horizon end = seasonal low
    s = from_values(100 + pattern, freq="M")
    seas = pd.Series(pattern, index=s.index)
    fc = _hand_forecast(s, "M", 100 + pattern[-12:], np.full(12, 2.0))
    assert fc.mean.iloc[-1] < 91                              # the end value is a seasonal low
    o = F.compute_outlook(_make_analysis(s, "M", "stable", seasonal=seas), fc)
    assert abs(o.expected_change_pct) < 1e-6 and o.prob_increase == pytest.approx(0.5)
    assert o.direction == "flat"
    assert o.scenarios["base"] == pytest.approx(float(fc.mean.iloc[-1]))


def test_error_dof_factor_reflects_autocorrelated_errors():
    rng = np.random.default_rng(0)
    iid = rng.standard_normal((8, 30))
    walk = np.cumsum(rng.standard_normal((8, 30)), axis=1)    # random-walk multi-step errors
    assert F._error_dof_factor(iid) > 0.8
    assert F._error_dof_factor(walk) < 0.15
    assert F._error_dof_factor(np.ones((3, 1))) == 1.0 and F._error_dof_factor(np.zeros((3, 4))) == 1.0
    assert F._error_dof_factor(1e300 * walk) == pytest.approx(F._error_dof_factor(walk))


# --------------------------------------------------------------------------- intermittent demand: sba


def _spare_parts(seed: int, n: int = 150) -> pd.Series:
    rng = np.random.default_rng(seed)
    y = np.where(rng.random(n) < 0.15, rng.integers(1, 6, n), 0).astype(float)
    return pd.Series(y, index=pd.date_range("2021-01-03", periods=n, freq="W-SUN"))


def test_intermittent_demand_uses_sba_and_squared_error_selection():
    """Spare-part demand (P(sale) = 0.15/week, 1-5 units, mean 0.45/week).  'sba' (Croston /
    Syntetos-Boylan) is a candidate for intermittent series only, and the candidates are
    ranked and weighted by RMSE: MAE is minimised by the median, which for mostly-zero data
    is about 0, so MASE ranking prefers forecasts that under-state the demand rate.  Mean
    |forecast rate - 0.45| over these 30 seeds: 0.159 before (level models ranked by MASE),
    0.132 now."""
    errs = []
    for seed in range(30):
        fc = F.auto_forecast(_spare_parts(seed), freq="W")
        _assert_valid(fc, 26)
        names = [ms.name for ms in fc.model_scores]
        assert "sba" in names and not any(nm in names for nm in ("naive", "drift", "analog_knn"))
        mases = [ms.mase for ms in fc.model_scores]
        assert mases == sorted(mases)                       # still reported best MASE first
        chosen = {ms.name for ms in fc.model_scores if ms.weight > 0}
        by_rmse = [ms.name for ms in sorted(fc.model_scores, key=lambda ms: ms.rmse)]
        assert chosen == set(by_rmse[:len(chosen)]), (seed, fc.method)
        assert fc.lower[95].iloc[0] == 0.0
        errs.append(abs(float(fc.mean.mean()) - 0.45))
    assert float(np.mean(errs)) < 0.145, np.mean(errs)
    # ordinary series never get it (and keep MASE selection)
    for kind in ("growing", "seasonal", "volatile"):
        fc = F.auto_forecast(sample(kind), freq="M")
        assert "sba" not in [ms.name for ms in fc.model_scores]


# --------------------------------------------------------------------------- multiplicative seasonality


def _multiplicative_analysis(freq: str, n: int, H: int, growth: float, ratio):
    """``y = trend * (1 + r)`` with an exact multiplicative decomposition (``seasonal = trend * r``,
    ``r = ratio(t)`` periodic) and the trend growing by ``growth`` per season.

    Returns ``(series, analysis, future values, future trend)``.
    """
    p = season_length(freq)
    t = np.arange(n + H, dtype=float)
    trend = 100 * growth ** (t / p)
    r = ratio(t)
    y = trend * (1 + r)
    s = from_values(y[:n], freq=freq)
    dec = Decomposition(trend=pd.Series(trend[:n], index=s.index),
                        seasonal=pd.Series(trend[:n] * r[:n], index=s.index),
                        resid=pd.Series(np.zeros(n), index=s.index), period=p)
    an = dataclasses.replace(_make_analysis(s, freq, "growing", seasonal_strength=0.9), decomposition=dec)
    return s, an, y[n:], trend[n:]


@pytest.mark.parametrize("growth", [2.0, 0.5])
def test_outlook_multiplicative_seasonality_follows_the_forecast_level(growth):
    """Multiplicative seasonality (decomposed in log space): the seasonal swing scales with the
    level.  Monthly, horizon 6 = half a season (a seasonal high): the partial-season adjustment
    repeated last year's seasonal values, too small once the level has doubled (+87.7 %
    reported, +71.3 % true) and too large once it has halved (-50.4 %, true -43.2 %)."""
    n, H = 60, 6
    s, an, fut, fut_trend = _multiplicative_analysis(
        "M", n, H, growth, lambda t: 0.3 * np.sin(2 * np.pi * t / 12))
    o = F.compute_outlook(an, _hand_forecast(s, "M", fut, 0.02 * fut))
    truth = 100 * (fut_trend.mean() / s.to_numpy()[-12:].mean() - 1)
    assert o.expected_change_pct == pytest.approx(truth, abs=0.05)
    assert o.direction == ("up" if growth > 1 else "down")
    # the continued seasonal component is the forecast's own seasonal part (trend x ratio)
    r = 0.3 * np.sin(2 * np.pi * np.arange(n, n + H) / 12)
    assert np.allclose(F._future_seasonal(an, H, fut), fut_trend * r, rtol=1e-9, atol=1e-9)
    # without a forecast the last season is repeated, as for additive decompositions
    last = an.decomposition.seasonal.to_numpy()[-12:]
    assert np.allclose(F._future_seasonal(an, H), last[:H])
    additive = _make_analysis(s, "M", "growing", seasonal=an.decomposition.seasonal)
    assert np.allclose(F._future_seasonal(additive, H, fut), last[:H])


def test_outlook_multiplicative_growth_has_no_false_peak():
    """Exponential growth (x2 a year) with a 40 % multiplicative seasonal swing has no peak,
    but the forecast minus last year's smaller seasonal values still swung, and a seasonal
    high inside the horizon was reported as the forecast peak (e.g. 2025-08)."""
    for n, H, phases in ((60, 12, (8, 9, 10, 11)), (48, 18, (1, 2, 3, 4, 5))):
        for ph in phases:
            s, an, fut, _ = _multiplicative_analysis(
                "M", n, H, 2.0, lambda t: 0.4 * np.sin(2 * np.pi * (t + ph) / 12))
            o = F.compute_outlook(an, _hand_forecast(s, "M", fut, 0.02 * fut))
            assert o.forecast_peak_date is None, (n, H, ph, o.forecast_peak_date)
            assert o.direction == "up"


# --------------------------------------------------------------------------- round 3: percentages, skill


def test_expected_change_pct_needs_a_level_away_from_zero():
    """F1: a P&L going from 10 to -10 read '향후 26주 동안 −578,464.7% 변화'; a monthly temperature
    with an annual mean of 0.3 ℃ gave +656.8 %.  No percentage across or near zero."""
    rng = np.random.default_rng(1)
    pnl = pd.Series(np.linspace(10, -10, 40) + rng.normal(0, 1, 40),
                    index=pd.date_range("2022-01-02", periods=40, freq="W-SUN"))
    fc = F.auto_forecast(pnl, freq="W")
    an = _make_analysis(prepare(pnl)[0], "W", "declining")
    o = F.compute_outlook(an, fc)
    assert o.expected_change_pct is None
    assert o.direction == "down" and o.prob_increase < 0.2       # the rest of the outlook stands
    lv = F.outlook_levels(an, fc)
    # round 5 (G1): the outlook uses the analysis' rule, which calls a level within 1 % of the mean
    # |value| 'zero' (here 0.003 against values of about ±10; the old outlook-only rule said 'near_zero')
    assert lv.undefined == "zero" and abs(lv.current) < 0.01 * np.mean(np.abs(pnl)) and lv.future < lv.current
    t = np.arange(120)
    temp = from_values(0.3 + 15 * np.sin(2 * np.pi * (t - 3) / 12)
                       + np.random.default_rng(17).normal(0, 1.5, 120), freq="M")
    fc_t = F.auto_forecast(temp, freq="M", candidates=list(BASELINES))
    assert F.compute_outlook(_make_analysis(temp, "M", "stable"), fc_t).expected_change_pct is None
    # crossing zero: no percentage even when the current level is large
    s = from_values(np.linspace(100, 20, 36), freq="M")
    cross = _hand_forecast(s, "M", np.linspace(10, -60, 12), np.full(12, 3.0))
    lv_c = F.outlook_levels(_make_analysis(s, "M", "declining"), cross)
    assert lv_c.undefined == "sign_change"
    assert F.compute_outlook(_make_analysis(s, "M", "declining"), cross).expected_change_pct is None
    # round 5 (G1): a negative series far from 0 no longer gets a % either -- the outlook follows the
    # analysis' rule (pct_change_undefined -> 'negative'; was: (−110 − cur)/|cur|·100), while the
    # direction and the probability are still decided on the levels
    neg = from_values(-100.0 + np.sin(np.arange(36.0)), freq="M")
    an_neg = _make_analysis(neg, "M", "stable")
    fc_neg = _hand_forecast(neg, "M", np.full(12, -110.0), np.full(12, 1.0))
    o_neg = F.compute_outlook(an_neg, fc_neg)
    lv_neg = F.outlook_levels(an_neg, fc_neg)
    assert o_neg.expected_change_pct is None and lv_neg.undefined == "negative"
    assert lv_neg.current == pytest.approx(neg.iloc[-12:].mean()) and lv_neg.future == pytest.approx(-110.0)
    assert o_neg.direction == "down" and o_neg.prob_increase < 0.1


def test_outlook_levels_reproduce_compute_outlook():
    """outlook_levels is the public contract for the two compared levels (narrative mirror)."""
    for name, freq, n in (("growing", "W", 156), ("seasonal", "M", 120), ("fad", "M", 120),
                          ("emerging", "Y", 12)):
        s = sample(name, n=n, freq=freq)
        fc = F.auto_forecast(s, freq=freq)
        an = _make_analysis(s, freq, "growing")
        lv, o = F.outlook_levels(an, fc), F.compute_outlook(an, fc)
        m = season_length(freq)
        assert (lv.k_cur, lv.k_fut) == (min(m, n), min(m, fc.horizon))
        assert lv.undefined is None
        assert o.expected_change_pct == pytest.approx((lv.future - lv.current) / abs(lv.current) * 100)


def test_skill_of_intermittent_demand_uses_the_ranking_metric():
    """F9: the skill was MAE-based although intermittent candidates are ranked by RMSE; MAE favours
    the all-zero naive forecast, so 28 of 40 iid demand series read 'worse than naive'."""
    worse = 0
    for seed in range(12):
        rng = np.random.default_rng(seed)
        occ = rng.random(150) < 0.15
        y = np.where(occ, rng.integers(1, 6, 150), 0).astype(float)
        fc = F.auto_forecast(pd.Series(y, index=pd.date_range("2021-01-03", periods=150, freq="W-SUN")),
                             freq="W")
        assert fc.skill is not None
        worse += fc.skill > 1.05
    assert worse <= 2


def test_skill_on_exact_data_is_not_a_ratio_of_rounding_noise():
    """F10: on y = -5 - t every model is exact; ensemble MAE 7e-16 / drift MAE 0 read as
    'skill 100' ('오차 +9,900%', confidence low)."""
    s = pd.Series(-5 - np.arange(9.0), index=pd.date_range("2024-01-01", periods=9, freq="D"))
    fc = F.auto_forecast(s, freq="D")
    assert fc.skill == pytest.approx(1.0)
    assert F._skill(7e-16, {"drift": 0.0}, scale=9.0) == pytest.approx(1.0)
    assert F._skill(0.0, {"naive": 1e-15}, scale=9.0) == pytest.approx(1.0)
    # real errors are still compared as before
    assert F._skill(0.5, {"naive": 1.0, "drift": 2.0}, scale=9.0) == pytest.approx(0.5)
    assert F._skill(3.0, {"drift": 0.0}, scale=9.0) == 100.0
    assert F._skill(1.0, {}, scale=9.0) is None


def test_driftless_random_walk_gets_no_trend_extrapolation():
    """F6: a daily geometric random walk (seed 2032) got p_up 0.95 / +16.1 % from
    ensemble(linear_trend@log, linear_trend, drift); the analysis knew it was a random walk."""
    from trendai.analysis import analyze

    rng = np.random.default_rng(2032)
    s = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.018, 730))),
                  index=pd.date_range("2023-01-01", periods=730, freq="D"))
    an = analyze(s, freq="D")
    assert F.is_random_walk(an)
    fc = F.auto_forecast(s, freq="D", random_walk=True)
    assert not set(F._TREND_EXTRAPOLATORS) & {ms.name for ms in fc.model_scores}
    o = F.compute_outlook(an, fc)
    assert o.direction == "flat" and 0.3 < o.prob_increase < 0.7, o
    # explicit candidates are honoured; trending and stationary series are not random walks
    assert "linear_trend" in F.auto_forecast(s, freq="D", random_walk=True,
                                             candidates=["linear_trend", "naive"]).method
    assert not F.is_random_walk(analyze(sample("growing")))
    assert not F.is_random_walk(analyze(sample("seasonal")))


def test_stationary_noise_intervals_do_not_widen_like_a_random_walk():
    """F5: the random-walk sqrt(k) prior widened the long leads of stationary series even
    when the backtest showed flat errors: white noise D400 got an 80 % half-width at lead 30
    of 1.44x the true one (review Monte Carlo: 85 / 96 % for nominal 80 / 95 %)."""
    last, first = [], []
    for seed in range(10):
        y = 100 + 10 * np.random.default_rng(seed).standard_normal(400)
        fc = F.auto_forecast(from_values(y, freq="D"), horizon=30, freq="D",
                             candidates=["ses", "naive", "theta", "drift"])
        hw = (fc.upper[80] - fc.lower[80]).to_numpy() / 2 / (F._level_z(80) * 10.0)
        first.append(hw[0])
        last.append(hw[-1])
    assert np.mean(last) < 1.36, np.mean(last)                 # was 1.44; now 1.29
    assert np.mean(first) > 0.75, np.mean(first)       # not narrower than the noise itself


def test_compounding_growth_offers_log_drift(registry):
    """F12: GDP-like annual series (2.5 % a year) were forecast by level-space models only:
    median bias -2.2 % at h = 5, 73 % of the forecasts below the outcome."""
    rng = np.random.default_rng(3)
    y = 5000 * np.exp(np.cumsum(rng.normal(0.03, 0.01, 40)))
    assert F._multiplicative_growth(y)
    assert not F._multiplicative_growth(50 + 3 * np.arange(40.0) + rng.normal(0, 2, 40))
    assert not F._multiplicative_growth(100 + rng.normal(0, 5, 40))
    fc = F.auto_forecast(from_values(y, freq="Y"), horizon=5, freq="Y")
    names = {ms.name for ms in fc.model_scores}
    assert ("drift@log" in names) == ("drift@log" in M.MODEL_REGISTRY)


# --------------------------------------------------------------------------- round 4: short noise, labels, zeros


def test_fallback_extrapolates_only_a_significant_trend():
    """F2: 7 daily points of flat noise (Mann-Kendall p = 0.37) got drift + naive, whose end-to-end
    slope over the 30-day default horizon read 'down, −50.7 %, P(up) = 22 %'; 30-70 % of such
    noise series got a direction call."""
    rng = np.random.default_rng(107)
    rng.normal(0, 0.3, 7)
    s = pd.Series(10 + rng.normal(0, 0.5, 7), index=pd.date_range("2025-12-25", periods=7, freq="D"))
    fc = F.auto_forecast(s)
    assert fc.method.startswith("fallback(naive)") and "no significant trend" in fc.method
    o = F.compute_outlook(_make_analysis(s, "D", "stable"), fc)
    assert o.direction == "flat" and 0.1 < o.prob_increase < 0.9, o
    # flat white noise, 5 points: almost never a direction call (was 43 of 60 seeds daily)
    calls = 0
    for seed in range(60):
        y = from_values(10 + np.random.default_rng(seed).normal(0, 0.5, 5), freq="D")
        calls += F.compute_outlook(_make_analysis(y, "D", "stable"), F.auto_forecast(y)).direction != "flat"
    assert calls <= 6, calls
    # a clear short trend keeps its drift and its direction
    for vals, word in (([10.0, 11.0, 12.0], "up"), ([10.0, 12.0, 14.0, 16.0, 18.0], "up"),
                       (10 + np.arange(6.0) + np.random.default_rng(1).normal(0, 0.3, 6), "up"),
                       ([5.0, 4.0, 3.0, 2.5, 1.5], "down")):
        y = from_values(vals, freq="M")
        fc = F.auto_forecast(y)
        assert fc.method.startswith("fallback(drift 0.50, naive 0.50)"), (vals, fc.method)
        assert F.compute_outlook(_make_analysis(y, "M", "growing"), fc).direction == word


def test_significant_trend_helper():
    assert F._significant_trend(np.array([1.0, 2.0, 3.0]))                      # exact line
    assert F._significant_trend(1e200 * np.array([1.0, 2.0, 3.0, 4.0]))         # no overflow
    assert not F._significant_trend(np.full(6, 4.0))                            # no slope
    assert not F._significant_trend(np.array([1.0, 2.0]))                       # too short
    assert not F._significant_trend(np.array([10.79, 9.78, 9.7, 10.85, 10.66, 9.83, 9.0]))
    assert F._significant_trend(np.array([10.1, 11.2, 11.8, 13.1, 14.0, 14.9]))


def test_partial_season_adjustment_only_for_material_seasonality():
    """F17: a constant weekly series (seasonal component ~1e-14) was compared 'seasonally adjusted'
    (adj_fut True) next to '계절성은 거의 없습니다 (강도 0.00)'.  The adjustment now needs a
    seasonal strength >= 0.2 and an effect of at least 0.5 % of the window's level."""
    n, H = 156, 26
    t = np.arange(n, dtype=float)
    const = from_values(np.full(n, 100.0), freq="W")
    tiny = pd.Series(1e-14 * np.sin(2 * np.pi * t / 52), index=const.index)
    lv = F.outlook_levels(_make_analysis(const, "W", "stable", seasonal=tiny),
                          _hand_forecast(const, "W", np.full(H, 100.0), np.full(H, 1.0)))
    assert (lv.adj_cur, lv.adj_fut) == (False, False) and lv.current == lv.future == 100.0
    # a real seasonal pattern the forecast carries: adjusted (strength 0.9) ...
    s, an = _weekly_seasonal_analysis()
    k = np.arange(n, n + H, dtype=float)
    seasonal_fc = 100 + 8 * np.sin(2 * np.pi * k / 52)
    lv = F.outlook_levels(an, _hand_forecast(s, "W", seasonal_fc, np.full(H, 1.0)))
    assert lv.adj_fut and lv.future == pytest.approx(100.0, abs=0.5)
    # ... but not when the analysis found it negligible (strength < 0.2): plain means
    weak = dataclasses.replace(an, metrics=dataclasses.replace(an.metrics, seasonal_strength=0.1))
    lv = F.outlook_levels(weak, _hand_forecast(s, "W", seasonal_fc, np.full(H, 1.0)))
    assert not lv.adj_fut and lv.future == pytest.approx(float(np.mean(seasonal_fc)))
    # a flat forecast carries no seasonality: nothing is removed, so nothing is labelled
    lv = F.outlook_levels(an, _hand_forecast(s, "W", np.full(H, 100.0), np.full(H, 1.0)))
    assert not lv.adj_fut and lv.future == 100.0


def test_all_zero_history_gets_no_float_noise_band():
    """F18: zeros(36) monthly got scenarios {0, 0, 5.28e-09} ('낙관 5.28×10⁻⁹'); an all-zero
    history has no scale, so its bands have zero width (short series too)."""
    for n in (36, 5):
        s = from_values(np.zeros(n), freq="M")
        fc = F.auto_forecast(s)
        _assert_valid(fc, 12)
        for lv in fc.levels:
            assert np.all(fc.lower[lv].to_numpy() == 0.0) and np.all(fc.upper[lv].to_numpy() == 0.0)
        o = F.compute_outlook(_make_analysis(s, "M", "stable"), fc)
        assert o.scenarios == {"pessimistic": 0.0, "base": 0.0, "optimistic": 0.0}
        assert o.direction == "flat" and o.expected_change_pct is None
    # any other constant keeps a (negligible, level-relative) band
    fc7 = F.auto_forecast(from_values(np.full(36, 7.0), freq="M"))
    assert np.all((fc7.upper[80] - fc7.lower[80]).to_numpy() > 0)


def test_intermittent_demand_percentage_compares_average_demand_rates():
    """R4-REV-3: the documented contract -- intermittent demand gets a percentage of its average
    demand rates unless at least half of the current window is zero ('mostly_zero')."""
    for seed in range(12):
        rng = np.random.default_rng(seed)
        y = (rng.random(60) < 0.6) * rng.integers(1, 20, 60).astype(float)
        if not F._is_intermittent(y, 12):
            continue
        s = from_values(y, freq="M")
        fc = F.auto_forecast(s)
        lv = F.outlook_levels(_make_analysis(s, "M", "stable"), fc)
        o = F.compute_outlook(_make_analysis(s, "M", "stable"), fc)
        if np.mean(y[-12:] == 0) >= 0.5:
            assert lv.undefined == "mostly_zero" and o.expected_change_pct is None
        else:
            assert lv.undefined is None
            assert lv.current == pytest.approx(np.mean(y[-12:]))          # average demand rate
            assert lv.future == pytest.approx(np.mean(fc.mean.to_numpy()[-12:]))
            assert o.expected_change_pct == pytest.approx((lv.future - lv.current) / lv.current * 100)


# --------------------------------------------------------------------------- round 5: one percentage rule


def _pnl_down_series(n: int = 120, freq: str = "MS") -> pd.Series:
    """The sweep's pnl_down case: a P&L from +50 down through 0 to about -30."""
    rng = np.random.default_rng(1)
    t = np.arange(n)
    return pd.Series(50 - 80 * t / (n - 1) + rng.normal(0, 4, n),
                     index=pd.date_range("2016-01-01", periods=n, freq=freq), name="P&L")


def test_negative_current_level_has_no_outlook_percentage():
    """G1 (round-4 F4): compute_outlook printed −32.0 % from a negative current level (−25.9 → −34.2)
    while analysis.pct_change_undefined calls a negative base 'negative'.  One rule now: no %, the
    reason 'negative', and the direction / probability still decided on the levels."""
    from trendai.analysis import analyze

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        an = analyze(_pnl_down_series(), freq="M")
        fc = F.auto_forecast(an.series, freq="M")
    lv, o = F.outlook_levels(an, fc), F.compute_outlook(an, fc)
    assert lv.current < 0 and lv.future < lv.current
    assert lv.undefined == "negative" and o.expected_change_pct is None
    assert o.direction == "down" and o.prob_increase < 0.1
    # a negative level expected to cross zero upwards is named for the crossing, like a downward one
    neg = from_values(-100.0 + np.sin(np.arange(36.0)), freq="M")
    up = F.outlook_levels(_make_analysis(neg, "M", "stable"),
                          _hand_forecast(neg, "M", np.linspace(20, 80, 12), np.full(12, 2.0)))
    assert up.future > 0 > up.current and up.undefined == "sign_change"


def test_outlook_percentage_rule_is_the_analysis_rule():
    """G1: forecast._pct_undefined delegates to analysis.pct_change_undefined (plus 'mostly_zero' and
    an upward zero crossing): the outlook gives a % exactly when the analysis' rule does."""
    from trendai.analysis import pct_change_undefined

    rng = np.random.default_rng(5)
    series = [from_values(100 + rng.normal(0, 5, 48), freq="M"),            # positive
              from_values(-100 + rng.normal(0, 5, 48), freq="M"),           # negative, far from 0
              from_values(rng.normal(0, 10, 48), freq="M"),                 # around 0
              from_values(np.linspace(50, -30, 48), freq="M")]              # crossing zero
    for s in series:
        an = _make_analysis(s, "M", "stable")
        y = s.to_numpy(dtype=float)
        for fut in (-150.0, -60.0, -5.0, 0.5, 5.0, 60.0, 150.0):
            lv = F.outlook_levels(an, _hand_forecast(s, "M", np.full(12, fut), np.full(12, 1.0)))
            want = pct_change_undefined(lv.current, lv.future, y)
            if want in ("negative", "near_zero") and lv.current < 0 < lv.future:
                want = "sign_change"
            assert lv.undefined == want, (float(y[-1]), fut, lv.undefined, want)
            o = F.compute_outlook(an, _hand_forecast(s, "M", np.full(12, fut), np.full(12, 1.0)))
            assert (o.expected_change_pct is None) == (want is not None)


def test_forecast_intermittent_rule_is_the_analysis_rule():
    """G4: forecast._is_intermittent is analysis._is_intermittent (one implementation, no mirror)."""
    from trendai.analysis import _is_intermittent as a_rule

    rng = np.random.default_rng(2)
    for _ in range(200):
        n = int(rng.integers(5, 120))
        y = (rng.random(n) < rng.uniform(0.2, 0.9)) * rng.integers(0, 6, n).astype(float)
        if rng.random() < 0.1:
            y = y - 1.0
        assert F._is_intermittent(y, 12) == a_rule(y, 12)
    assert not hasattr(F, "_INTERMITTENT_ZEROS") and not hasattr(F, "_PCT_MIN_LEVEL_SD")


def test_interval_level_names_the_band_behind_the_scenarios():
    """G4: the narrative labelled the scenarios '50% 예측구간' for levels (50, 95) while compute_outlook
    took them from the 95 % band (its own rule: the level closest to 80).  One public rule now."""
    from trendai import narrative as N

    s = sample("growing")
    fc = F.auto_forecast(s, freq="M", levels=(50, 95))
    assert F.interval_level(fc) == 95 and N._interval_level(fc) == 95
    o = F.compute_outlook(_make_analysis(s, "M", "growing"), fc)
    assert o.scenarios["pessimistic"] == pytest.approx(float(fc.lower[95].iloc[-1]))
    assert F.interval_level(F.auto_forecast(s, freq="M")) == 80


def test_ensemble_of_exact_backtest_members_has_finite_weights():
    """Round-4 review fuzz (decline_to_zero, Y60): an intermittent history ending in a long run of
    zeros gave every top candidate an RMSE of exactly 0, so the inverse-RMSE weights were inf / inf
    (method 'ensemble(ses nan, holt_damped nan, theta nan)', backtest MASE NaN, skill None)."""
    rng = np.random.default_rng(1692021327)          # the fuzz case (decline_to_zero, n=60, yearly)
    y = np.maximum(0.0, 100 - 3.0 * np.arange(60.0) + rng.normal(0, 2, 60))
    assert np.all(y[-25:] == 0)
    s = from_values(y, start="2001-01-01", freq="Y")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fc = F.auto_forecast(s, freq="Y")
    assert F._is_intermittent(y, 1)
    weights = [ms.weight for ms in fc.model_scores]
    assert all(math.isfinite(w) for w in weights) and math.isclose(sum(weights), 1.0)
    assert "nan" not in fc.method
    assert fc.backtest_mase is not None and math.isfinite(fc.backtest_mase)
    assert fc.skill is not None and math.isfinite(fc.skill)
    assert np.all(np.isfinite(fc.mean)) and np.all(fc.mean == 0)


def test_outlook_levels_carry_the_history_scale():
    """V1-4: a move of a level at or near 0 (no %) is judged against the history's typical level."""
    from trendai.analysis import analyze
    from trendai.forecast import auto_forecast, outlook_levels

    y = np.r_[np.arange(1.0, 49.0), np.zeros(12)]
    s = pd.Series(y, index=pd.date_range("2018-01-01", periods=60, freq="MS"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = analyze(s)
        lv = outlook_levels(a, auto_forecast(s, horizon=12))
    assert lv.undefined == "zero" and lv.current == 0.0
    assert lv.scale == pytest.approx(float(np.mean(np.abs(y))), rel=1e-9)


# --------------------------------------------------------------------------- round 5, verify 2


def _stopped_weekly() -> pd.Series:
    """V2-7: weekly demand (P(sale) 0.35, 1-6 units), no sale in the last 34 of 104 weeks
    (the last sale 41 weeks before the end)."""
    r = np.random.default_rng(57)
    n = 104
    z = (r.poisson(2, n) + 1) * (r.random(n) < 0.35)
    z[-(n // 3):] = 0
    return pd.Series(z.astype(float), index=pd.date_range("2024-01-07", periods=n, freq="W-SUN"))


def test_stopped_intermittent_demand_is_not_forecast_at_last_years_rate():
    """V2-7: ensemble(ses, theta, sba) forecast 0.235 a week after 41 weeks without a sale
    (52-week mean 0.15).  TSB joins the intermittent candidates and the backtest picks it."""
    s = _stopped_weekly()
    fc = F.auto_forecast(s, freq="W")
    assert "tsb" in [ms.name for ms in fc.model_scores]
    assert float(fc.mean.max()) < 0.05, (fc.method, fc.mean.iloc[0])
    # regular series never get it
    assert "tsb" not in [ms.name for ms in F.auto_forecast(sample("growing", n=60, freq="M"), freq="M").model_scores]


def test_fit_history_is_the_forecasts_history():
    """R5-REV2-2: one definition of the history the forecast uses (last max(400, 8 m))."""
    y = np.arange(1000.0)
    assert len(F.fit_history(y, 7)) == 400 and F.fit_history(y, 7)[0] == 600.0
    assert len(F.fit_history(y, 52)) == 416
    tail0 = pd.Series(np.r_[np.linspace(5, 50, 100), np.zeros(450)],
                      index=pd.date_range("2020-01-01", periods=550, freq="D"))
    fc = F.auto_forecast(tail0, freq="D")
    assert not np.any(fc.mean.to_numpy()) and not np.any((fc.upper[80] - fc.lower[80]).to_numpy())


def test_outlook_at_the_zero_floor_has_no_percentage():
    """V2-1: 'sample fad -H 24' printed '예측 변화 −100.0%' next to 'P(up) 45%': the forecast means of
    the last season were clipped at 0 (the band above them stays wide), so (0 − 7.74) / 7.74 says
    only that the forecast hit the floor.  Same at −99.4 % (H = 18: 10 of 12 months at 0)."""
    from trendai.analysis import analyze

    s = sample("fad", n=120, freq="M")
    a = analyze(s, freq="M")
    for h, at_floor in ((24, 12), (18, 10)):
        fc = F.auto_forecast(s, horizon=h, freq="M")
        lv = F.outlook_levels(a, fc)
        assert int(np.sum(fc.mean.to_numpy()[-lv.k_fut:] <= 0)) == at_floor
        assert lv.undefined == "floor" and F.compute_outlook(a, fc).expected_change_pct is None
    # a window with no point at the floor keeps its percentage (H = 12: −63 %)
    fc12 = F.auto_forecast(s, horizon=12, freq="M")
    assert F.outlook_levels(a, fc12).undefined is None
    assert F.compute_outlook(a, fc12).expected_change_pct == pytest.approx(-63.3, abs=0.5)
    # positive quarterly profit falling to the floor (40 -> 2)
    r = np.random.default_rng(41)
    t = np.arange(24)
    q = pd.Series(40 - 38 * t / 23 + r.normal(0, 1, 24), index=pd.date_range("2020-01-01", periods=24, freq="QS"))
    aq, fq = analyze(q, freq="Q"), F.auto_forecast(q, horizon=8, freq="Q")
    assert F.outlook_levels(aq, fq).undefined == "floor"
    assert F.compute_outlook(aq, fq).expected_change_pct is None
