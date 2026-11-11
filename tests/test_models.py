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


def test_extreme_magnitudes_are_exact_scale_equivariant_without_warnings():
    # regression: analog_knn computed raw window variances (inf at |y| ~ 1e300 -> NaN forecast), and
    # _standardize / _is_seasonal squared raw values (overflow -> crude scale; underflow at 1e-200 ->
    # seasonality silently dropped, so forecasts differed by up to 9 % from the rescaled series)
    import warnings
    rng = np.random.default_rng(13)
    t = np.arange(60.0)
    cases = (1e300 * (1 + rng.random(40)), -1e300 * (1 + rng.random(40)), 1e-300 * (1 + rng.random(40)),
             1e-200 * (5 + rng.random(40)), 1e300 * (2 + np.sin(2 * np.pi * t / 12) + 0.1 * rng.random(60)))
    for y in cases:
        ref = np.abs(y).max()
        for nm in ALL_NAMES:
            with warnings.catch_warnings():
                warnings.simplefilter("error")  # no overflow / underflow / invalid notes at all
                f = M.make_model(nm).fit(y, 12).predict(6)
            unit = M.make_model(nm).fit(y / ref, 12).predict(6) * ref
            assert np.all(np.isfinite(f)), nm
            assert np.allclose(f, unit, rtol=1e-9, atol=0.0), (nm, y[0])


def test_analog_knn_ties_resolve_to_earliest_analogues():
    # every window of a perfect line is an equally good analogue; rounding noise must not pick
    # windows near the end (few future points -> damped continuation that bends the line)
    for slope in (0.5, 0.1, 1.0 / 3.0):
        y = 10 + slope * np.arange(60.0)
        f = _fit_predict("analog_knn", y, 1, 12)
        assert np.allclose(f, 10 + slope * np.arange(60.0, 72.0), rtol=1e-12, atol=1e-9), slope


def test_log_wrapper_handles_non_positive_gracefully():
    y = np.r_[0.0, -1.0, np.arange(1.0, 40.0)]
    for nm in ("holt_damped@log", "holt_winters@log", "linear_trend@log", "logistic_growth"):
        _fit_predict(nm, y, 12, 12)


def test_log_wrapper_matches_manual_log_fit():
    y = sample("emerging", seed=1).to_numpy()
    f = M.make_model("linear_trend@log").fit(y, 12).predict(12)
    manual = np.exp(M.make_model("linear_trend").fit(np.log(y), 12).predict(12))
    assert np.allclose(f, manual)


@pytest.mark.parametrize("name", ["holt_damped@log", "holt_winters@log", "linear_trend@log"])
def test_log_models_do_not_explode_on_short_explosive_history(name):
    # regression: emerging D n=20 grows x8 over its last 6 points; the log-slope compounded to ~21000
    # after 30 steps (data max 67) and auto_forecast ended at ~6800
    y = sample("emerging", freq="D", n=20, seed=7).to_numpy()
    f = _fit_predict(name, y, 7, 30)
    max_growth = y.max() / y.min()  # largest growth anywhere in the history (x16)
    first = max(y[-1], f[0])
    assert np.all(f <= first * max_growth * (1 + 1e-9)), f
    assert np.all(f >= y.min()), f
    if name != "holt_winters@log":
        assert f[-1] > y[-1]  # still a growth forecast, just not a compounding one


def test_log_growth_guard_leaves_steady_exponential_trends_alone():
    # the guard only bounds moves larger than anything observed: a clean exponential trend
    # (horizon longer than the guard's recent window) is extrapolated as before
    t = np.arange(40.0)
    y = 10 * np.exp(0.02 * t)
    for name, inner in (("linear_trend@log", "linear_trend"), ("holt_damped@log", "holt_damped")):
        f = _fit_predict(name, y, 7, 30)
        manual = np.exp(M.make_model(inner).fit(np.log(y), 7).predict(30))
        assert np.allclose(f, manual, rtol=1e-9), name
    assert np.allclose(_fit_predict("linear_trend@log", y, 7, 30), 10 * np.exp(0.02 * np.arange(40.0, 70.0)),
                       rtol=1e-6)


def test_log_move_bounds():
    ly = np.log(np.r_[np.full(10, 5.0), 5.0 * 2.0 ** np.arange(1, 6)])  # flat, then doubling 5 times
    up, down = M._log_move_bounds(ly, 1, 30)
    assert up.shape == down.shape == (30,)
    assert np.all(np.diff(up) >= 0) and np.all(np.diff(down) >= 0)
    assert np.allclose(up[:5], np.log(2.0) * np.arange(1, 6))  # recent pace (window = last 8 points)
    assert np.allclose(up[14:], 5 * np.log(2.0))  # never more than the whole history's growth
    assert np.allclose(down, 0.0)  # it never fell


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


def test_weekly_seasonality_test_rejects_random_walks():
    # regression: the phase ANOVA assumed white noise, but the moving-average remainder of a random
    # walk is smooth, so 13-20 % of pure random walks with m = 52 and 3-6 seasons were "seasonal"
    # (about 3 % with the effective-df correction)
    for n in (156, 208, 260, 312):
        hits = sum(M._is_seasonal(100 + np.cumsum(np.random.default_rng(500 + s).normal(size=n)), 52)
                   for s in range(100))
        assert hits <= 7, (n, hits)
    for n in (60, 120):  # monthly random walks: was 4-6 %
        hits = sum(M._is_seasonal(100 + np.cumsum(np.random.default_rng(900 + s).normal(size=n)), 12)
                   for s in range(100))
        assert hits <= 5, (n, hits)


def test_weekly_seasonality_still_detected():
    # real weekly-yearly seasonality (smooth annual cycle) on stable / trending levels, every seed
    for n in (130, 156, 208, 312):
        for kind in ("seasonal", "growing", "declining"):
            for seed in range(5):
                assert M._is_seasonal(sample(kind, n=n, freq="W", seed=seed).to_numpy(), 52), (kind, n, seed)
    rng = np.random.default_rng(7)
    t = np.arange(208.0)
    weak = 100 + 2.0 * np.sin(2 * np.pi * t / 52) + rng.normal(size=208)  # amplitude 2 sd, white noise
    assert M._is_seasonal(weak, 52)


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


def test_logistic_growth_has_no_phantom_step_on_white_noise():
    # regression: a step placed just after the last observation (t0 > 1, |k| ~ 100) fitted that one
    # recency-weighted point while its far asymptote was constrained by no data: on N(100, 1) noise
    # forecasts of 13.9 (seed 30000, n=12) and ~20-30 % of fits beyond 10 sd at every length
    y = 100 + np.random.default_rng(30000).standard_normal(12)
    f = _fit_predict("logistic_growth", y, 1, 4)
    assert np.all(np.abs(f - 100) < 3), f
    for n in (12, 16, 36, 120, 300):
        worst = 0.0
        for s in range(40):
            y = 100 + np.random.default_rng(31000 + s).standard_normal(n)
            model = M.make_model("logistic_growth").fit(y, 1)
            worst = max(worst, float(np.max(np.abs(model.predict(12) - 100))))
            p = model.params_
            if "t0" in p:  # the fitted transition is (at least 30 %) inside the data
                c = 1 / (1 + np.exp(-abs(p["k"]) * (np.array([0.0, 1.0]) - p["t0"])))
                assert c[1] - c[0] >= 0.3 - 1e-9, (n, s, p)
        assert worst < 10, (n, worst)


def test_logistic_admissible_transitions():
    adm = M.LogisticGrowth._admissible
    t0 = np.array([1.0375, 0.5, 1.09, 1.0, 0.5])
    k = np.array([-101.4, 12.0, 10.0, -24.0, 200.0])
    ok = adm(t0, k, 12, 4, 0.3)
    # phantom step just after the end (2 % of it observed) | full S-curve | early growth phase with
    # only 29 % of the transition observed (< 30 %) | step resolved by only 2 observations | sub-step jump
    assert ok.tolist() == [False, True, False, False, False]
    # the same early growth phase is fine once enough of it is observed and resolved
    assert adm(np.array([1.09]), np.array([10.0]), 120, 4, 0.25).tolist() == [True]


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


# --------------------------------------------------------------------------- drift@log and sba


NEW_NAMES = ["drift@log", "sba", "tsb"]
_GENERIC_CHECKS = [
    test_all_samples_monthly, test_min_obs_and_short_series, test_constant_series,
    test_series_with_zeros, test_huge_and_tiny_magnitudes_are_scale_equivariant, test_deterministic,
    test_refit_same_instance_matches_fresh_model, test_input_types_and_no_mutation,
    test_unusual_season_lengths, test_adversarial_shapes,
]


@pytest.mark.parametrize("check", _GENERIC_CHECKS, ids=lambda f: f.__name__)
@pytest.mark.parametrize("name", NEW_NAMES)
def test_new_models_pass_the_generic_checks(name, check):
    # the models outside ALL_NAMES (not default candidates for ordinary series) get the same
    # robustness checks as every registered model
    check(name)


@pytest.mark.parametrize("name", NEW_NAMES)
def test_new_models_every_frequency_negative_and_extreme_inputs(name):
    for args in [("W", 156, 52, 26), ("D", 400, 7, 30), ("Q", 40, 4, 8), ("Y", 30, 1, 5),
                 ("W", 110, 52, 26), ("D", 730, 7, 30)]:
        test_every_frequency_all_samples(name, *args)
    if not M.make_model(name).requires_positive:
        test_negative_series(name)
    import warnings
    rng = np.random.default_rng(14)
    occ = rng.random(40) < 0.4
    for scale in (1e300, -1e300, 1e-300, 1e-200):
        for y in (1 + rng.random(40), np.where(occ, 1 + rng.random(40), 0.0)):
            Y = scale * y
            ref = np.abs(Y).max()
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                f = M.make_model(name).fit(Y, 12).predict(6)
            unit = M.make_model(name).fit(Y / ref, 12).predict(6) * ref
            assert np.all(np.isfinite(f)), (name, scale)
            # (atol: a log model's shift for data with zeros leaves rounding residue near 0)
            assert np.allclose(f, unit, rtol=1e-9, atol=1e-12 * ref), (name, scale)


def test_drift_at_log_compounds_the_average_growth_rate():
    # GDP-like compounding: level-space drift projects linear growth and falls behind every step
    y = 100 * 1.03 ** np.arange(40.0)
    f = _fit_predict("drift@log", y, 1, 5)
    assert np.allclose(f, 100 * 1.03 ** np.arange(40.0, 45.0), rtol=1e-9)
    assert np.all(_fit_predict("drift", y, 1, 5) < f - 1.0)
    # noisy growth: the average rate from the first to the last value, compounded (at most:
    # the LogTransformed growth guard may slow it to the recent pace), above level-space drift
    rng = np.random.default_rng(5)
    z = 5000 * np.exp(np.cumsum(rng.normal(0.025, 0.01, 40)))
    g = (z[-1] / z[0]) ** (1 / 39)
    fz = _fit_predict("drift@log", z, 1, 5)
    assert fz[0] == pytest.approx(z[-1] * g, rel=1e-9)
    assert np.all(fz <= z[-1] * g ** np.arange(1, 6) * (1 + 1e-9))
    assert np.all(fz > _fit_predict("drift", z, 1, 5)) and np.all(np.diff(fz) > 0)
    model = M.make_model("drift@log")
    assert model.requires_positive and not model.needs_season and model.min_obs == 2


def test_drift_at_log_is_registered_but_only_used_on_request():
    # as a default candidate it helped GDP-like series but cost the general accuracy testbed
    # (S-curves cut mid-transition compound into overshoots), so it is opt-in
    assert "drift@log" in M.available_models()
    assert M.make_model("drift@log").offered_for == "explicit"
    for n, m, pos in ((120, 12, True), (40, 1, True), (40, 1, False)):
        for intermittent in (False, True):
            assert "drift@log" not in M.default_candidates(n, m, pos, intermittent=intermittent)


def _sba_reference(y, alpha=0.1, k=5):
    """Scalar Croston / SBA (sizes and intervals smoothed separately, mean-of-first-k start)."""
    pos = [i for i, v in enumerate(y) if v != 0]
    sizes = [float(y[i]) for i in pos]
    intervals = [pos[0] + 1.0] + [float(b - a) for a, b in zip(pos, pos[1:])]

    def ses(x):
        lvl = sum(x[:k]) / len(x[:k])
        for v in x:
            lvl += alpha * (v - lvl)
        return lvl

    factor = 1.0 if len(pos) == len(y) else 1.0 - alpha / 2.0
    return factor * ses(sizes) / ses(intervals)


def test_sba_matches_the_croston_recursion():
    y = np.array([0, 0, 3, 0, 0, 0, 5, 0, 2, 0, 0, 4, 0, 0, 0, 0, 6, 0, 1, 0, 0, 0, 3, 0], float)
    model = M.make_model("sba").fit(y, 1)
    f = model.predict(7)
    assert np.allclose(f, _sba_reference(y), rtol=1e-12) and np.ptp(f) == 0.0
    assert model.params_["demands"] == 7 and model.params_["bias_factor"] == pytest.approx(0.95)
    # SBA = Croston x (1 - alpha/2): below the plain size/interval ratio
    assert f[0] < model.params_["size"] / model.params_["interval"]
    # no zeros at all: nothing to correct, SES of the values
    v = np.array([2.0, 4.0, 3.0, 5.0, 4.0, 6.0, 5.0])
    assert _fit_predict("sba", v, 1, 3) == pytest.approx(np.full(3, _sba_reference(v)), rel=1e-12)
    # one demand, all zeros, a demand at the very start
    assert _fit_predict("sba", np.r_[np.zeros(9), 4.0], 1, 2) == pytest.approx(np.full(2, 0.95 * 4 / 10))
    assert np.array_equal(_fit_predict("sba", np.zeros(10), 12, 4), np.zeros(4))
    assert _fit_predict("sba", np.r_[2.0, np.zeros(5)], 1, 1)[0] == pytest.approx(0.95 * 2.0)


def test_sba_is_offered_for_intermittent_demand_only():
    assert M.make_model("sba").offered_for == "intermittent"
    for pos in (False, True):
        assert "sba" not in M.default_candidates(150, 52, positive=pos)
    inter = M.default_candidates(150, 52, positive=False, intermittent=True)
    assert "sba" in inter and "ses" in inter and "holt_damped" in inter
    assert "sba" not in M.default_candidates(1, 1, positive=False, intermittent=True)  # min_obs
    # the regular candidate list is unchanged by the new keyword
    assert M.default_candidates(120, 12, True) == M.default_candidates(120, 12, True, intermittent=False)


def test_sba_estimates_the_demand_rate_better_than_level_models():
    """Spare-part demand (P(sale) = 0.15/week, 1-5 units, mean 0.45/week).  SES-type models
    jump towards every sale and decay through the zeros, so their forecast depends on when
    the last sale happened; SBA smooths sizes and intervals separately."""
    err = {nm: [] for nm in ("sba", "ses", "holt_damped", "theta")}
    for seed in range(60):
        rng = np.random.default_rng(seed)
        y = np.where(rng.random(150) < 0.15, rng.integers(1, 6, 150), 0).astype(float)
        for nm in err:
            err[nm].append(abs(float(_fit_predict(nm, y, 52, 26).mean()) - 0.45))
    mean = {nm: float(np.mean(v)) for nm, v in err.items()}
    assert mean["sba"] < 0.11, mean
    assert mean["sba"] < 0.9 * min(v for nm, v in mean.items() if nm != "sba"), mean


def test_log_wrapper_shift_is_scale_free():
    # degraded mode (non-positive input): the shift had an absolute floor of 1e-9, so data of
    # magnitude 1e-300 with zeros were forecast at ~1e-25, 1e275 times the data
    rng = np.random.default_rng(3)
    y = np.where(rng.random(48) < 0.6, 1 + rng.random(48), 0.0)
    for nm in ("holt_damped@log", "holt_winters@log", "linear_trend@log", "drift@log"):
        base = _fit_predict(nm, y, 12, 6)
        for scale in (1e-300, 1e-12, 1e9):
            f = _fit_predict(nm, y * scale, 12, 6)
            assert np.allclose(f / scale, base, rtol=1e-6, atol=1e-9), (nm, scale)


# --------------------------------------------------------------------------- tsb (round 5, V2-7)


def _tsb_reference(y, alpha=0.1, beta=0.1, k=5):
    """Scalar TSB: SES of the demand sizes (on demand only) times SES of the 0/1 occurrence
    (every period), each starting from the mean of its first values."""
    sizes = [float(v) for v in y if v != 0]
    occ = [1.0 if v != 0 else 0.0 for v in y]

    def ses(x, a, k0):
        lvl = sum(x[:k0]) / len(x[:k0])
        for v in x:
            lvl += a * (v - lvl)
        return lvl

    return ses(sizes, alpha, k) * ses(occ, beta, max(k, len(y) // 4)) if sizes else 0.0


def test_tsb_matches_the_recursion_and_decays_through_zeros():
    """V2-7: a product without a sale for 41 weeks was forecast at last year's rate (0.235 a week,
    52-week mean 0.15): Croston / SBA only update on a sale.  TSB lowers the demand probability
    every period, so a run of zeros lowers the rate."""
    y = np.array([0, 0, 3, 0, 0, 0, 5, 0, 2, 0, 0, 4, 0, 0, 0, 0, 6, 0, 1, 0, 0, 0, 3, 0], float)
    f = _fit_predict("tsb", y, 1, 5)
    assert np.allclose(f, _tsb_reference(y), rtol=1e-12) and np.ptp(f) == 0.0
    stopped = np.r_[y, np.zeros(40)]
    assert _fit_predict("tsb", stopped, 1, 1)[0] < 0.05 * f[0]        # 0.9 ** 40 = 1.5 %
    assert _fit_predict("sba", stopped, 1, 1)[0] > 0.5 * _fit_predict("sba", y, 1, 1)[0]   # SBA does not
    assert np.array_equal(_fit_predict("tsb", np.zeros(10), 12, 4), np.zeros(4))
    v = np.array([2.0, 4.0, 3.0, 5.0, 4.0, 6.0, 5.0])                    # no zeros: SES of the values
    assert _fit_predict("tsb", v, 1, 2) == pytest.approx(np.full(2, _tsb_reference(v)), rel=1e-12)


def test_tsb_is_offered_for_intermittent_demand_only():
    assert M.make_model("tsb").offered_for == "intermittent"
    for pos in (False, True):
        assert "tsb" not in M.default_candidates(150, 52, positive=pos)
    assert "tsb" in M.default_candidates(150, 52, positive=False, intermittent=True)
