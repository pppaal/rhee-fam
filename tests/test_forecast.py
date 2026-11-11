"""Tests for trendai.forecast: backtest, model selection, ensemble, intervals, outlook.

Every test must pass both with the three baseline models only and with the
full model zoo registered in :mod:`trendai.models`; tests that depend on the
registry contents either pin ``candidates`` or run under the ``registry``
fixture, which executes them once with the full registry and once with the
baselines only.  ``AnalysisResult`` objects are built by hand (no dependency
on :mod:`trendai.analysis`).
"""

from __future__ import annotations

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
                   seasonal: pd.Series | None = None) -> AnalysisResult:
    """A minimal hand-built AnalysisResult (only what compute_outlook needs is meaningful)."""
    m = season_length(freq)
    seas = seasonal if seasonal is not None else pd.Series(np.zeros(len(s)), index=s.index)
    trend = (s - seas).rolling(max(m, 3), center=True, min_periods=1).mean()
    dec = Decomposition(trend=trend, seasonal=seas, resid=s - trend - seas, period=m)
    peak_i = int(np.argmax(s.to_numpy()))
    metrics = TrendMetrics(
        n_obs=len(s), freq=freq, start=s.index[0], end=s.index[-1], last_value=float(s.iloc[-1]),
        mean=float(s.mean()), total_change_pct=None, cagr_pct=None, recent_change_pct=None,
        slope=0.0, slope_pct=0.0, momentum=0.0, acceleration=0.0, volatility=0.1,
        trend_strength=0.5, seasonal_strength=0.0, mk_tau=0.0, mk_pvalue=1.0, direction="flat",
        peak_date=s.index[peak_i], peak_value=float(s.iloc[peak_i]), drawdown_pct=0.0,
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
    ("volatile", "flat", False, 0.0, "high", "stable"),
])
def test_future_phase_rules(current, direction, strong, change, conf, expected):
    assert F._future_phase(current, direction, strong, change, conf) == expected


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
    assert o_neg.expected_change_pct is not None and math.isfinite(o_neg.expected_change_pct)


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
