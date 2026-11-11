"""Tests for trendai.models: registry, robustness, determinism, accuracy, speed."""

from __future__ import annotations

import time

import numpy as np
import pytest

from trendai import models as M
from trendai.samples import list_samples, sample

CONTRACT_NAMES = [
    "ses", "holt", "holt_damped", "holt_winters", "theta", "linear_trend", "ar",
    "lag_ridge", "analog_knn", "logistic_growth",
    "holt_damped@log", "holt_winters@log", "linear_trend@log",
]
BASELINES = ["naive", "seasonal_naive", "drift"]
ALL_NAMES = BASELINES + CONTRACT_NAMES
NON_POSITIVE_OK = [nm for nm in ALL_NAMES if not M.make_model(nm).requires_positive]


def _fit_predict(name: str, y: np.ndarray, m: int, h: int) -> np.ndarray:
    f = M.make_model(name).fit(np.asarray(y, dtype=float), m).predict(h)
    assert isinstance(f, np.ndarray)
    assert f.shape == (h,)
    assert np.all(np.isfinite(f)), f"{name} produced non-finite forecasts: {f}"
    return f


def _mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))


# --------------------------------------------------------------------------- registry / API


def test_registry_has_contract_names():
    avail = M.available_models()
    for nm in ALL_NAMES:
        assert nm in avail
        model = M.make_model(nm)
        assert isinstance(model, M.Forecaster)
        assert model.name == nm
        assert model.min_obs <= 3


def test_make_model_returns_fresh_instances_and_unknown_raises():
    a, b = M.make_model("holt_damped"), M.make_model("holt_damped")
    assert a is not b
    with pytest.raises(KeyError):
        M.make_model("does_not_exist")


def test_model_flags():
    assert M.make_model("holt_winters").needs_season
    assert M.make_model("holt_winters@log").needs_season
    assert not M.make_model("holt_damped@log").needs_season
    for nm in ("logistic_growth", "holt_damped@log", "holt_winters@log", "linear_trend@log"):
        assert M.make_model(nm).requires_positive
    for nm in ("ses", "holt", "holt_damped", "theta", "linear_trend", "ar", "lag_ridge", "analog_knn"):
        assert not M.make_model(nm).requires_positive


def test_default_candidates_filters():
    pos = M.default_candidates(120, 12, positive=True)
    for nm in ALL_NAMES:
        assert nm in pos
    nonpos = M.default_candidates(120, 12, positive=False)
    assert "logistic_growth" not in nonpos
    assert not any(nm.endswith("@log") for nm in nonpos)
    assert "holt_winters" in nonpos and "ar" in nonpos
    short = M.default_candidates(20, 12, positive=True)
    assert "holt_winters" not in short and "holt_winters@log" not in short
    assert "holt_damped" in short
    nonseasonal = M.default_candidates(50, 1, positive=True)
    assert "holt_winters" not in nonseasonal and "seasonal_naive" not in nonseasonal


def test_register_custom_model():
    class Const(M.Forecaster):
        name = "const_test"

        def _fit(self, y, m):
            self._c = float(np.mean(y))

        def _predict(self, h):
            return np.full(h, self._c)

    M.register("const_test", Const)
    try:
        f = M.make_model("const_test").fit(np.arange(5.0)).predict(3)
        assert np.allclose(f, 2.0)
    finally:
        del M.MODEL_REGISTRY["const_test"]


def test_base_api_errors():
    model = M.make_model("holt")
    with pytest.raises(RuntimeError):
        model.predict(3)
    with pytest.raises(ValueError):
        M.make_model("holt").fit(np.array([1.0, 2.0]))  # below min_obs
    with pytest.raises(ValueError):
        M.make_model("ses").fit(np.array([1.0, np.nan, 2.0]))
    with pytest.raises(ValueError):
        M.make_model("ses").fit(np.ones((3, 2)))
    model = M.make_model("ses").fit(np.arange(10.0))
    with pytest.raises(ValueError):
        model.predict(0)


# --------------------------------------------------------------------------- every model on every sample


@pytest.mark.parametrize("name", ALL_NAMES)
def test_all_samples_monthly(name):
    for kind in list_samples():
        y = sample(kind).to_numpy()
        f = _fit_predict(name, y, 12, 24)
        # sane magnitude: within a generous band around the data
        span = y.max() - y.min()
        assert np.all(f > y.min() - 10 * span) and np.all(f < y.max() + 10 * span), (name, kind)


@pytest.mark.parametrize("name", ALL_NAMES)
@pytest.mark.parametrize("freq,n,m,h", [("W", 156, 52, 26), ("D", 400, 7, 30), ("Q", 40, 4, 8),
                                        ("Y", 30, 1, 5), ("W", 110, 52, 26), ("D", 730, 7, 30)])
def test_every_frequency_all_samples(name, freq, n, m, h):
    # every archetype at every canonical frequency (W n=110: between 2 and 2.5 seasons)
    for kind in list_samples():
        y = sample(kind, n=n, freq=freq).to_numpy()
        f = _fit_predict(name, y, m, h)
        span = y.max() - y.min()
        assert np.all(np.abs(f - y[-1]) < 50 * span), (name, freq, kind)


# --------------------------------------------------------------------------- edge inputs


@pytest.mark.parametrize("name", ALL_NAMES)
def test_min_obs_and_short_series(name):
    rng = np.random.default_rng(1)
    model = M.make_model(name)
    lengths = sorted({model.min_obs, *range(max(3, model.min_obs), 11)})
    for n in lengths:
        y = 50 + np.cumsum(rng.normal(size=n))
        for m in (1, 4, 12):
            _fit_predict(name, y, m, 12)


@pytest.mark.parametrize("name", ALL_NAMES)
def test_constant_series(name):
    # exact: no rounding residue (e.g. exp(log(c)) or a shift in the @log wrapper)
    values = [7.0, 0.1, 0.0, 1e9, 1e-6, 123.456]
    if not M.make_model(name).requires_positive:
        values.append(-3.0)
    for n in sorted({M.make_model(name).min_obs, 5, 30, 60}):
        for value in values:
            for m in (1, 12):
                y = np.full(n, value)
                f = _fit_predict(name, y, m, 12)
                assert np.array_equal(f, np.full(12, value)), (name, n, value, m, f)


@pytest.mark.parametrize("name", ALL_NAMES)
def test_series_with_zeros(name):
    rng = np.random.default_rng(2)
    y = np.where(rng.random(60) < 0.4, 0.0, rng.poisson(5, 60).astype(float))
    _fit_predict(name, y, 12, 12)
    _fit_predict(name, np.zeros(40), 12, 6)
    _fit_predict(name, np.r_[np.zeros(20), np.arange(1.0, 21.0)], 4, 8)


@pytest.mark.parametrize("name", NON_POSITIVE_OK)
def test_negative_series(name):
    y = -sample("growing").to_numpy()
    f = _fit_predict(name, y, 12, 12)
    assert np.all(f < 0)
    t = np.arange(80.0)
    y2 = 10 * np.sin(2 * np.pi * t / 12) - 0.3 * t  # crosses zero
    _fit_predict(name, y2, 12, 12)


@pytest.mark.parametrize("name", ALL_NAMES)
def test_huge_and_tiny_magnitudes_are_scale_equivariant(name):
    y = sample("growing", seed=3).to_numpy()
    base = _fit_predict(name, y, 12, 12)
    for scale in (1e9, 1e-6):
        f = _fit_predict(name, y * scale, 12, 12)
        assert np.allclose(f / scale, base, rtol=1e-5, atol=1e-6 * np.abs(base).max()), (name, scale)
    # huge level with tiny relative variation
    _fit_predict(name, 1e9 + 1e3 * np.sin(np.arange(48.0)), 12, 12)


@pytest.mark.parametrize("name", ALL_NAMES)
def test_deterministic(name):
    y = sample("breakout", seed=4).to_numpy()
    a = _fit_predict(name, y, 12, 18)
    b = _fit_predict(name, y.copy(), 12, 18)
    assert np.array_equal(a, b)
    model = M.make_model(name).fit(y, 12)
    assert np.array_equal(model.predict(18), model.predict(18))  # predict has no side effects


@pytest.mark.parametrize("name", ALL_NAMES)
def test_refit_same_instance_matches_fresh_model(name):
    long, short = sample("seasonal").to_numpy(), sample("growing").to_numpy()[:6]
    model = M.make_model(name).fit(long, 12)
    model.fit(short, 12)  # refit on a series short enough to trigger fallbacks
    fresh = M.make_model(name).fit(short, 12)
    assert np.array_equal(model.predict(7), fresh.predict(7))
    # diagnostics must describe the latest fit, not a previous one
    assert getattr(model, "params_", None) == getattr(fresh, "params_", None), name


@pytest.mark.parametrize("name", ALL_NAMES)
def test_input_types_and_no_mutation(name):
    s = sample("growing")
    ro = s.to_numpy()  # read-only under pandas copy-on-write
    base = _fit_predict(name, ro, 12, 6)
    assert np.array_equal(M.make_model(name).fit(s, 12).predict(6), base)
    assert np.array_equal(M.make_model(name).fit(list(ro), np.int64(12)).predict(np.int64(6)), base)
    y = ro.copy()
    y0 = y.copy()
    model = M.make_model(name).fit(y, 12)
    model.predict(6)
    assert np.array_equal(y, y0)  # input not mutated
    y[-1] += 1000.0  # later caller mutation does not leak into the fitted model
    assert np.array_equal(model.predict(6), base)


@pytest.mark.parametrize("name", ALL_NAMES)
def test_unusual_season_lengths(name):
    # 0 / negative -> treated as 1; m > n/2 -> seasonality dropped; m == n/2 -> exactly 2 seasons
    y = sample("seasonal", n=40).to_numpy()
    for m in (0, -5, 1, 2, 3, 19, 20, 21, 40, 41, 10**6):
        _fit_predict(name, y, m, 9)


@pytest.mark.parametrize("name", ALL_NAMES)
def test_adversarial_shapes(name):
    rng = np.random.default_rng(11)
    cases = [
        np.r_[np.zeros(30), np.full(30, 100.0)],          # level step
        np.r_[np.ones(30), 1e6, np.ones(29)],              # huge isolated spike
        np.r_[np.ones(59), 1e6],                           # spike at the very end
        np.tile([1.0, 3.0], 30),                           # period-2 alternation
        np.exp(np.linspace(0.0, 23.0, 60)),                # 10 orders of magnitude
        np.array([1e-10, 1e10])[np.arange(8) % 2],         # extreme alternation, n=8
        1e9 + 1e-3 * rng.normal(size=60),                  # tiny variation on a huge level
        np.r_[np.full(59, 5.0), 6.0],                      # constant then a jump
        np.r_[np.zeros(50), 1.0, np.zeros(9)],             # sparse counts
    ]
    for y in cases:
        for m in (1, 2, 12):
            _fit_predict(name, y, m, 13)


def test_extreme_magnitudes_stay_finite():
    import warnings
    rng = np.random.default_rng(12)
    for y in (1e300 * (1 + rng.random(40)), 1e-300 * (1 + rng.random(40)), 1e-200 * (5 + rng.random(40))):
        for nm in ALL_NAMES:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)  # numpy overflow notes for |y| > 1e150
                f = M.make_model(nm).fit(y, 12).predict(6)
            assert np.all(np.isfinite(f)), nm


def test_log_wrapper_handles_non_positive_gracefully():
    y = np.r_[0.0, -1.0, np.arange(1.0, 40.0)]
    for nm in ("holt_damped@log", "holt_winters@log", "linear_trend@log", "logistic_growth"):
        _fit_predict(nm, y, 12, 12)


def test_log_wrapper_matches_manual_log_fit():
    y = sample("emerging", seed=1).to_numpy()
    f = M.make_model("linear_trend@log").fit(y, 12).predict(12)
    manual = np.exp(M.make_model("linear_trend").fit(np.log(y), 12).predict(12))
    assert np.allclose(f, manual)


def test_holt_winters_degrades_to_holt_damped():
    y = sample("growing").to_numpy()[:20]  # fewer than 2 seasons of m=12
    hw = M.make_model("holt_winters").fit(y, 12).predict(12)
    hd = M.make_model("holt_damped").fit(y, 12).predict(12)
    assert np.allclose(hw, hd)


# --------------------------------------------------------------------------- internals


def test_ar_coefficients_are_stationary():
    rng = np.random.default_rng(5)
    for kind in ("growing", "seasonal", "volatile"):
        model = M.make_model("ar").fit(sample(kind).to_numpy(), 12)
        phi = model._phi
        if phi.size:
            roots = np.roots(np.r_[1.0, -phi])
            assert np.max(np.abs(roots)) <= 0.97 + 1e-9
    explosive = np.array([1.5, 0.3])
    stab = M._stabilise_ar(explosive)
    assert np.max(np.abs(np.roots(np.r_[1.0, -stab]))) <= 0.97 + 1e-9
    # random walk input: forecasts stay bounded
    y = np.cumsum(rng.normal(size=200))
    f = _fit_predict("ar", y, 1, 100)
    assert np.abs(f - y[-1]).max() < 20 * np.std(np.diff(y)) * 10


def test_seasonality_detection():
    assert M._is_seasonal(sample("seasonal").to_numpy(), 12)
    assert M._is_seasonal(sample("seasonal", n=156, freq="W").to_numpy(), 52)
    assert M._is_seasonal(sample("growing", n=400, freq="D").to_numpy(), 7)
    assert not M._is_seasonal(sample("fad").to_numpy(), 12)
    assert not M._is_seasonal(sample("fad", n=400, freq="D").to_numpy(), 7)
    rng = np.random.default_rng(0)
    assert not M._is_seasonal(rng.normal(size=120), 12)
    assert not M._is_seasonal(np.full(48, 3.0), 12)
    assert not M._is_seasonal(np.arange(10.0), 12)


def _scalar_ets(y, a, b, g, p, l0, b0, s0, trend, season):
    """Plain-Python additive (damped) ETS recursion, written independently of models.py."""
    lvl, tr, s, sse = l0, (b0 if trend else 0.0), list(s0), 0.0
    for t, yt in enumerate(y):
        j = t % len(s)
        damped = p * tr if trend else 0.0
        e = yt - (lvl + damped + (s[j] if season else 0.0))
        sse += e * e
        lvl = lvl + damped + a * e
        if trend:
            tr = damped + b * e
        if season:
            s[j] += g * e
    return sse, lvl, tr, np.array(s)


@pytest.mark.parametrize("trend", [False, True])
@pytest.mark.parametrize("season", [False, True])
def test_ets_filter_matches_scalar_reference(trend, season):
    rng = np.random.default_rng(21)
    y = np.cumsum(rng.normal(size=53))  # 53: not a multiple of the season
    K = 5
    a = rng.uniform(0.05, 0.95, K)
    b = a * rng.uniform(0, 1, K)
    g = (1 - a) * rng.uniform(0, 1, K)
    p = rng.uniform(0.8, 0.98, K)
    s0 = rng.normal(size=12)
    sse, lvl, tr, S = M._ets_filter(y, a, b, g, p, 0.3, -0.1, s0, trend, season)
    for k in range(K):
        r_sse, r_l, r_b, r_s = _scalar_ets(y, a[k], b[k], g[k], p[k], 0.3, -0.1, s0, trend, season)
        assert sse[k] == pytest.approx(r_sse, rel=1e-10)
        assert lvl[k] == pytest.approx(r_l, rel=1e-10, abs=1e-12)
        assert tr[k] == pytest.approx(r_b, rel=1e-10, abs=1e-12)
        if season:
            assert np.allclose(S[:, k], r_s, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("n", [97, 101, 107, 305, 333])
def test_seasonal_phase_alignment_for_any_length(n):
    # n not a multiple of m, and n > 300 (ES parameter window): forecasts must stay in phase
    t = np.arange(n + 12.0)
    full = 20 + 5 * np.sin(2 * np.pi * t / 12) + 2 * np.cos(4 * np.pi * t / 12)
    y, expected = full[:n], full[n:]
    for nm in ("ses", "holt", "holt_damped", "holt_winters", "theta", "linear_trend", "analog_knn",
               "lag_ridge", "holt_damped@log", "holt_winters@log", "seasonal_naive"):
        assert _mae(_fit_predict(nm, y, 12, 12), expected) < 0.05, (nm, n)


def test_theta_drift_is_half_the_ols_slope():
    rng = np.random.default_rng(22)
    t = np.arange(80.0)
    y = 50 + 0.4 * t + rng.normal(size=80)
    model = M.make_model("theta").fit(y, 1)
    f = model.predict(10)
    b = np.polyfit(t, y, 1)[0]
    assert np.allclose(np.diff(f), b / 2, rtol=1e-9)
    # first step: SES level + b/2 * (1 - (1 - alpha)^n) / alpha  (Hyndman & Billah 2003)
    a = model._alpha
    level = model._loc + model._scale * model._level
    assert f[0] == pytest.approx(level + b / 2 * (1 - (1 - a) ** model._n_ses) / a, rel=1e-9)


def test_ar_recovers_known_coefficients():
    rng = np.random.default_rng(23)
    phi = np.array([0.5, -0.3])
    n = 800
    d = np.zeros(n)
    e = rng.normal(size=n)
    for i in range(2, n):
        d[i] = 0.3 * (1 - phi.sum()) + phi[0] * d[i - 1] + phi[1] * d[i - 2] + e[i]
    y = 100 + np.cumsum(d)
    model = M.make_model("ar").fit(y, 1)
    est = model._phi
    assert est.size >= 2
    assert est[0] == pytest.approx(0.5, abs=0.1) and est[1] == pytest.approx(-0.3, abs=0.1)
    assert np.all(np.abs(est[2:]) < 0.15)
    assert model._mu * model._scale == pytest.approx(d[1:].mean(), rel=1e-6)


@pytest.mark.parametrize("k", [12.0, -12.0])
def test_logistic_recovers_clean_s_curve(k):
    u = np.arange(150.0) / 149
    full = 10 + 80 / (1 + np.exp(-k * (u - 0.5)))
    f = _fit_predict("logistic_growth", full[:120], 1, 30)
    assert np.max(np.abs(f - full[120:])) < 0.5


def test_smoothing_parameters_reach_their_bounds():
    # regression: the refinement used to halve its step every round, so it could not move more than
    # ~0.94 logit units off the coarse grid (phi was capped at 0.9716, trend alpha at 0.953)
    rng = np.random.default_rng(24)
    t = np.arange(200.0)
    y = 10 + 0.5 * t + rng.normal(scale=0.3, size=200)
    assert M.make_model("holt_damped").fit(y, 1).params_["phi"] > 0.975
    rw = np.cumsum(rng.normal(size=300))
    assert M.make_model("holt").fit(rw, 1).params_["alpha"] > 0.96


def test_pattern_search_walks_beyond_the_grid():
    def obj(P):
        return (P[:, 0] - 3.3) ** 2

    best, _ = M._pattern_search(obj, [np.array([-0.5, 0.0, 0.5])], step=0.5, rounds=4)
    assert best[0] == pytest.approx(3.3, abs=0.05)


def test_pattern_search_finds_minimum():
    def obj(P):
        return (P[:, 0] - 0.3) ** 2 + (P[:, 1] + 0.7) ** 2

    best, _ = M._pattern_search(obj, [np.linspace(-2, 2, 5), np.linspace(-2, 2, 5)], step=0.5, rounds=8)
    assert np.allclose(best, [0.3, -0.7], atol=0.02)


def test_es_parameters_in_admissible_range():
    y = sample("seasonal").to_numpy()
    model = M.make_model("holt_winters").fit(y, 12)
    p = model.params_
    assert 0 < p["alpha"] < 1
    assert 0 <= p["beta"] <= p["alpha"]
    assert 0 <= p["gamma"] <= 1 - p["alpha"]
    assert 0.8 <= p["phi"] <= 0.98
    hd = M.make_model("holt_damped").fit(sample("growing").to_numpy(), 12).params_
    assert 0.8 <= hd["phi"] <= 0.98


# --------------------------------------------------------------------------- accuracy on archetypes


def _holdout(kind: str, seed: int, h: int = 12) -> tuple[np.ndarray, np.ndarray]:
    y = sample(kind, seed=seed).to_numpy()
    return y[:-h], y[-h:]


@pytest.mark.parametrize("kind", ["seasonal", "growing"])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_seasonal_models_beat_naive(kind, seed):
    train, test = _holdout(kind, seed)
    naive = _mae(_fit_predict("naive", train, 12, 12), test)
    for nm in ("holt_winters", "linear_trend"):
        assert _mae(_fit_predict(nm, train, 12, 12), test) < naive, (nm, kind, seed)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_holt_damped_beats_naive_on_growing(seed):
    train, test = _holdout("growing", seed)
    naive = _mae(_fit_predict("naive", train, 12, 12), test)
    assert _mae(_fit_predict("holt_damped", train, 12, 12), test) < naive


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_ml_and_analogue_models_beat_naive_on_seasonal(seed):
    train, test = _holdout("seasonal", seed)
    naive = _mae(_fit_predict("naive", train, 12, 12), test)
    for nm in ("analog_knn", "lag_ridge"):
        assert _mae(_fit_predict(nm, train, 12, 12), test) < naive, (nm, seed)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_logistic_growth_saturates_on_peaking(seed):
    y = sample("peaking", seed=seed).to_numpy()
    f = _fit_predict("logistic_growth", y, 12, 24)
    assert np.all(f >= 0.7 * y.max()) and np.all(f <= 1.6 * y.max()), f
    train, _ = _holdout("peaking", seed)
    f2 = _fit_predict("logistic_growth", train, 12, 12)
    assert np.all(f2 >= 0.7 * train.max()) and np.all(f2 <= 1.6 * train.max())


def test_logistic_growth_respects_saturation_cap():
    t = np.arange(60.0)
    y = 5 + 2.0 * t  # linear growth: no saturation visible in the data
    f = _fit_predict("logistic_growth", y, 1, 200)
    assert f.max() <= 3 * y.max() * (1 + 1e-9)


def test_trend_models_follow_linear_trend():
    y = 10 + 0.5 * np.arange(60.0)
    expected = 10 + 0.5 * np.arange(60.0, 72.0)
    for nm in ("holt", "linear_trend", "drift", "ar", "lag_ridge", "analog_knn"):
        f = _fit_predict(nm, y, 1, 12)
        assert np.allclose(f, expected, atol=0.05 * 0.5 * 12), nm


def test_pure_seasonal_pattern_is_reproduced():
    t = np.arange(96.0)
    y = 20 + 5 * np.sin(2 * np.pi * t / 12)
    expected = 20 + 5 * np.sin(2 * np.pi * np.arange(96.0, 108.0) / 12)
    for nm in ("holt_winters", "linear_trend", "theta", "analog_knn", "lag_ridge", "seasonal_naive"):
        f = _fit_predict(nm, y, 12, 12)
        assert _mae(f, expected) < 0.5, nm


# --------------------------------------------------------------------------- speed


@pytest.mark.parametrize("freq,m", [("M", 12), ("W", 52)])
def test_fit_predict_speed(freq, m):
    y = sample("growing", n=500, freq=freq).to_numpy()
    slow = {}
    for nm in ALL_NAMES:
        M.make_model(nm).fit(y, m).predict(12)  # warm-up
        t0 = time.perf_counter()
        M.make_model(nm).fit(y, m).predict(24)
        dt = time.perf_counter() - t0
        if dt >= 0.25:
            slow[nm] = dt
    assert not slow, f"too slow at n=500: {slow}"
