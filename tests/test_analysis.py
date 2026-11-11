"""Tests for trendai.analysis (past & present)."""

from __future__ import annotations

import dataclasses
import math
import re
import time
import warnings

import numpy as np
import pandas as pd
import pytest

from trendai import analysis as A
from trendai import samples
from trendai.data import from_values, season_length
from trendai.types import (
    CHANGEPOINT_KINDS,
    PHASES,
    AnalysisResult,
    Changepoint,
    Decomposition,
    LifecyclePhase,
    TrendMetrics,
)

HANGUL = re.compile(r"[가-힣]")
DIGIT = re.compile(r"\d")
SEEDS = range(5)


def _assert_metrics_finite(m: TrendMetrics) -> None:
    for f in dataclasses.fields(m):
        v = getattr(m, f.name)
        if isinstance(v, float):
            assert math.isfinite(v), f"{f.name} = {v}"
    for name in ("total_change_pct", "cagr_pct", "recent_change_pct"):
        v = getattr(m, name)
        assert v is None or (isinstance(v, float) and math.isfinite(v)), (name, v)
    assert m.drawdown_pct <= 0.0
    assert 0.0 <= m.trend_strength <= 1.0
    assert 0.0 <= m.seasonal_strength <= 1.0
    assert 0.0 <= m.mk_pvalue <= 1.0
    assert -1.0 <= m.mk_tau <= 1.0
    assert m.direction in ("up", "down", "flat")


def _months_apart(a: pd.Timestamp, b: pd.Timestamp) -> int:
    return abs((a.year - b.year) * 12 + (a.month - b.month))


# =========================================================================== decompose


@pytest.mark.parametrize("name,freq,n", [
    ("seasonal", "M", 120), ("growing", "M", 120), ("fad", "W", 156),
    ("reviving", "D", 400), ("emerging", "Q", 40), ("breakout", "M", 60),
])
def test_decompose_reconstructs_exactly_without_nans(name, freq, n):
    s = samples.sample(name, n=n, freq=freq, seed=1)
    p = season_length(freq)
    d = A.decompose(s, p)
    assert isinstance(d, Decomposition)
    for comp in (d.trend, d.seasonal, d.resid):
        assert comp.index.equals(s.index)
        assert np.isfinite(comp.to_numpy()).all()
    np.testing.assert_allclose((d.trend + d.seasonal + d.resid).to_numpy(), s.to_numpy(),
                               rtol=0, atol=1e-9)
    assert d.period == p
    # the seasonal component sums to ~0 over every full period (any phase offset)
    seas = d.seasonal.to_numpy()
    window_sums = np.convolve(seas, np.ones(p), mode="valid")
    assert np.abs(window_sums).max() < 1e-8 * max(1.0, float(np.abs(s).max()))


def test_decompose_edges_extrapolate_trend_and_recover_pattern():
    t = np.arange(96)
    seas = 5 * np.sin(2 * np.pi * t / 12)
    d = A.decompose(from_values(10 + 0.5 * t + seas), 12)
    # local-linear edges: exact on a line, including first/last points
    np.testing.assert_allclose(d.trend.to_numpy(), 10 + 0.5 * t, atol=1e-6)
    np.testing.assert_allclose(d.seasonal.to_numpy(), seas, atol=1e-6)


def test_decompose_edges_have_no_nans_on_curved_noisy_series():
    s = samples.sample("emerging", n=120, freq="M", seed=3)
    d = A.decompose(s, 12)
    tr = d.trend.to_numpy()
    assert np.isfinite(tr).all()
    # edge trend values stay close to the data level (no wild extrapolation)
    assert abs(tr[0] - s.iloc[:6].mean()) < 5
    assert abs(tr[-1] - s.iloc[-3:].mean()) < 15


@pytest.mark.parametrize("seed", SEEDS)
def test_seasonal_strength_high_on_seasonal_low_on_fad(seed):
    for freq, n in (("M", 120), ("D", 400)):
        r_s = A.analyze(samples.sample("seasonal", n=n, freq=freq, seed=seed), freq=freq)
        r_f = A.analyze(samples.sample("fad", n=n, freq=freq, seed=seed), freq=freq)
        assert r_s.metrics.seasonal_strength > 0.6
        assert r_f.metrics.seasonal_strength < 0.4


def test_decompose_period_one_has_zero_seasonal():
    s = samples.sample("growing", n=60, freq="M", seed=0)
    d = A.decompose(s, 1)
    assert d.period == 1
    assert (d.seasonal.to_numpy() == 0).all()
    np.testing.assert_allclose((d.trend + d.resid).to_numpy(), s.to_numpy(), atol=1e-9)
    assert np.isfinite(d.trend.to_numpy()).all()


def test_decompose_fewer_than_two_periods_has_zero_seasonal():
    s = samples.sample("seasonal", n=20, freq="M", seed=0)
    d = A.decompose(s, 12)
    assert d.period == 1
    assert (d.seasonal.to_numpy() == 0).all()
    np.testing.assert_allclose((d.trend + d.resid).to_numpy(), s.to_numpy(), atol=1e-9)


@pytest.mark.parametrize("values", [[3.0], [1.0, 2.0], [1.0, 5.0, 2.0], [7.0] * 4])
def test_decompose_tiny_series(values):
    s = from_values(values)
    d = A.decompose(s, 12)
    for comp in (d.trend, d.seasonal, d.resid):
        assert np.isfinite(comp.to_numpy()).all()
    np.testing.assert_allclose((d.trend + d.seasonal + d.resid).to_numpy(), s.to_numpy(), atol=1e-9)


def test_decompose_is_robust_to_a_spike():
    clean = samples.sample("seasonal", seed=3)
    spiky = clean.copy()
    spiky.iloc[50] += 100
    d0, d1 = A.decompose(clean, 12), A.decompose(spiky, 12)
    assert np.abs((d0.trend - d1.trend).to_numpy()).max() < 1.0
    assert np.abs((d0.seasonal - d1.seasonal).to_numpy()).max() < 1.0
    assert d1.resid.iloc[50] > 90


def test_decompose_spike_in_zeros_stays_in_residual():
    s = from_values([0.0] * 20 + [10.0] + [0.0] * 15)
    d = A.decompose(s, 12)
    # (robustness weights are floored at 1e-4, so a tiny leak is expected)
    np.testing.assert_allclose(d.trend.to_numpy(), 0.0, atol=0.01)
    np.testing.assert_allclose(d.seasonal.to_numpy(), 0.0, atol=0.01)
    assert d.resid.iloc[20] == pytest.approx(10.0, abs=0.01)
    r = A.analyze(s)
    assert [a.index for a in r.anomalies] == [20]
    assert r.changepoints == []


def test_decompose_interpolates_nan_input_with_warning():
    s = samples.sample("growing", n=48, freq="M", seed=0).copy()
    s.iloc[[0, 10, 11, 47]] = np.nan
    with pytest.warns(RuntimeWarning):
        d = A.decompose(s, 12)
    for comp in (d.trend, d.seasonal, d.resid):
        assert np.isfinite(comp.to_numpy()).all()


# =========================================================================== Mann-Kendall / Theil-Sen


def test_mann_kendall_strictly_increasing_and_decreasing():
    tau, p = A.mann_kendall(np.arange(50) ** 1.5)
    assert tau == pytest.approx(1.0)
    assert p < 1e-10
    tau, p = A.mann_kendall(-np.arange(50, dtype=float))
    assert tau == pytest.approx(-1.0)
    assert p < 1e-10


def test_mann_kendall_matches_hand_computation():
    # n=5 increasing: S=10, Var=5*4*15/18, z=(S-1)/sqrt(Var) -> p = 0.0275
    tau, p = A.mann_kendall(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
    z = 9 / math.sqrt(5 * 4 * 15 / 18)
    assert tau == 1.0
    assert p == pytest.approx(math.erfc(z / math.sqrt(2)), rel=1e-12)


def test_mann_kendall_noise_not_significant():
    y = np.random.default_rng(42).normal(size=200)
    _, p = A.mann_kendall(y)
    assert p > 0.2
    # a symmetric (palindromic) sequence has S = 0 exactly
    half = np.random.default_rng(1).normal(size=40)
    tau, p = A.mann_kendall(np.r_[half, half[::-1]])
    assert tau == pytest.approx(0.0)
    assert p == pytest.approx(1.0)


def test_mann_kendall_tie_correction_and_degenerate_inputs():
    tau, p = A.mann_kendall(np.r_[np.ones(30), 2 * np.ones(30)])
    assert math.isfinite(tau) and math.isfinite(p)
    assert tau > 0 and p < 1e-6
    tau, p = A.mann_kendall(np.r_[np.ones(10), np.zeros(5), np.ones(10)])
    assert math.isfinite(tau) and 0 <= p <= 1
    assert A.mann_kendall(np.full(40, 3.0)) == (0.0, 1.0)
    assert A.mann_kendall(np.array([5.0, 5.0, 5.0])) == (0.0, 1.0)
    assert A.mann_kendall(np.array([1.0, 2.0])) == (0.0, 1.0)
    assert A.mann_kendall(np.array([])) == (0.0, 1.0)
    tau, p = A.mann_kendall(np.array([1.0, np.nan, 2.0, 3.0, np.inf, 4.0]))
    assert tau == 1.0 and 0 <= p <= 1


def test_theil_sen_slope_exact_on_line_and_robust():
    t = np.arange(80, dtype=float)
    assert A.theil_sen_slope(3 - 0.7 * t) == pytest.approx(-0.7, abs=1e-12)
    y = 2 + 0.25 * t
    y[[5, 40, 70]] += np.array([500.0, -300.0, 800.0])
    assert A.theil_sen_slope(y) == pytest.approx(0.25, abs=1e-9)
    long = 1.5 * np.arange(3000, dtype=float) + 7
    assert A.theil_sen_slope(long) == pytest.approx(1.5, abs=1e-12)
    assert A.theil_sen_slope(np.array([4.0])) == 0.0
    assert A.theil_sen_slope(np.array([])) == 0.0


# =========================================================================== changepoints


def test_breakout_changepoint_found_monthly():
    bd = samples.breakout_date()
    hits = 0
    for seed in SEEDS:
        cps = A.detect_changepoints(samples.sample("breakout", seed=seed), period=12)
        if any(_months_apart(c.date, bd) <= 4 and c.kind in ("acceleration", "reversal_up")
               for c in cps):
            hits += 1
    assert hits >= 4


def test_breakout_changepoint_found_daily():
    n = 400
    k = int(round(0.6 * (n - 1)))
    hits = 0
    for seed in SEEDS:
        s = samples.sample("breakout", n=n, freq="D", seed=seed)
        cps = A.analyze(s, freq="D").changepoints
        hits += any(abs(c.index - k) <= 10 and c.kind in ("acceleration", "reversal_up") for c in cps)
    assert hits >= 4


@pytest.mark.parametrize("seed", range(10))
def test_noisy_straight_line_has_at_most_one_changepoint(seed):
    rng = np.random.default_rng(seed)
    y = 10 + 0.5 * np.arange(120) + rng.normal(0, 3, 120)
    assert len(A.detect_changepoints(from_values(y), period=12)) <= 1


@pytest.mark.parametrize("seed", SEEDS)
def test_seasonal_sample_has_at_most_one_changepoint(seed):
    assert len(A.detect_changepoints(samples.sample("seasonal", seed=seed), period=12)) <= 1


def test_fad_reversal_down_near_peak():
    peak = int(round(0.55 * 119))
    hits = 0
    for seed in SEEDS:
        cps = A.detect_changepoints(samples.sample("fad", seed=seed), period=12)
        hits += any(c.kind == "reversal_down" and abs(c.index - peak) <= 8 for c in cps)
    assert hits >= 4


def test_level_shift_detected_and_spikes_ignored():
    rng = np.random.default_rng(0)
    y = np.r_[np.full(60, 10.0), np.full(60, 30.0)] + rng.normal(0, 1, 120)
    y[[0, 30, 90, 119]] += np.array([30.0, 40.0, -40.0, 30.0])
    cps = A.detect_changepoints(from_values(y))
    assert len(cps) == 1
    assert cps[0].kind == "level_shift"
    assert abs(cps[0].index - 60) <= 1
    assert cps[0].magnitude > 0.5


def test_noise_free_kink_and_line():
    y = np.r_[np.full(60, 10.0), 10 + 2 * np.arange(1, 61)]
    cps = A.detect_changepoints(from_values(y))
    assert [c.kind for c in cps] == ["reversal_up"]
    assert abs(cps[0].index - 60) <= 1
    assert cps[0].slope_before == pytest.approx(0.0, abs=1e-6)
    assert cps[0].slope_after == pytest.approx(2.0, abs=1e-6)
    assert A.detect_changepoints(from_values(3.0 + 0.5 * np.arange(100))) == []
    assert A.detect_changepoints(from_values(np.full(50, 4.0))) == []


def test_changepoint_fields_and_min_segment():
    s = samples.sample("fad", n=120, freq="M", seed=2)
    cps = A.detect_changepoints(s, period=12, max_changepoints=5)
    assert 1 <= len(cps) <= 5
    idx = [c.index for c in cps]
    assert idx == sorted(idx)
    min_seg = max(4, 12 // 2, 120 // 12)
    bounds = [0] + idx + [len(s)]
    assert min(np.diff(bounds)) >= min_seg
    for c in cps:
        assert isinstance(c, Changepoint)
        assert c.kind in CHANGEPOINT_KINDS
        assert c.date == s.index[c.index]
        assert math.isfinite(c.magnitude) and c.magnitude >= 0
        assert math.isfinite(c.slope_before) and math.isfinite(c.slope_after)
    assert len(A.detect_changepoints(s, period=12, max_changepoints=2)) <= 2
    custom = A.detect_changepoints(s, period=12, min_segment=25)
    assert min(np.diff([0] + [c.index for c in custom] + [len(s)])) >= 25


def test_changepoints_short_series_empty():
    assert A.detect_changepoints(from_values([1.0, 2.0, 3.0, 10.0, 11.0, 12.0])) == []


def test_changepoints_long_series_block_path_fast():
    s = samples.sample("breakout", n=2000, freq="D", seed=0)
    t0 = time.perf_counter()
    cps = A.detect_changepoints(s, period=7)
    assert time.perf_counter() - t0 < 2.0
    k = int(round(0.6 * 1999))
    assert any(abs(c.index - k) <= 30 for c in cps)


# =========================================================================== anomalies


def test_anomalies_planted_spikes_found():
    s = samples.sample("seasonal", seed=0).copy()
    s.iloc[30] += 40
    s.iloc[77] -= 40
    d = A.decompose(s, 12)
    an = A.detect_anomalies(s, d)
    found = {a.index: a for a in an}
    assert 30 in found and 77 in found
    assert found[30].zscore > 3.5 and found[77].zscore < -3.5
    assert found[30].value == pytest.approx(float(s.iloc[30]))
    assert abs(found[30].expected - 50) < 20
    assert len(an) <= 4
    assert [a.index for a in an] == sorted(a.index for a in an)


def test_anomalies_none_on_constant_and_threshold_respected():
    s = from_values([5.0] * 30)
    assert A.detect_anomalies(s, A.decompose(s, 12)) == []
    s = samples.sample("volatile", seed=1)
    d = A.decompose(s, 12)
    assert len(A.detect_anomalies(s, d, threshold=10.0)) <= len(A.detect_anomalies(s, d, threshold=2.0))


def test_anomalies_adapt_to_level_dependent_noise():
    # multiplicative noise: busy periods must not be flagged wholesale
    rng = np.random.default_rng(0)
    t = np.arange(400)
    level = 100 + 5000 * np.exp(-0.5 * ((t - 150) / 30) ** 2)
    y = level * np.exp(rng.normal(0, 0.1, 400))
    y[[50, 300]] *= 6
    s = from_values(y, freq="D")
    r = A.analyze(s, freq="D")
    idx = {a.index for a in r.anomalies}
    assert {50, 300} <= idx
    assert len(r.anomalies) <= 15


# =========================================================================== metrics


def test_metrics_growing():
    r = A.analyze(samples.sample("growing", seed=0))
    m = r.metrics
    _assert_metrics_finite(m)
    assert m.n_obs == 120 and m.freq == "M"
    assert m.start == r.series.index[0] and m.end == r.series.index[-1]
    assert m.direction == "up" and m.mk_tau > 0.8 and m.mk_pvalue < 1e-6
    assert m.total_change_pct == pytest.approx(212.5, rel=0.15)  # 40 -> 125
    assert m.cagr_pct is not None and 10 < m.cagr_pct < 16
    assert m.recent_change_pct is not None and m.recent_change_pct > 5
    assert m.slope > 0 and m.slope_pct > 0
    assert m.momentum > 0                       # convex: recent slope above average
    assert m.trend_strength > 0.8
    assert m.peak_date >= r.series.index[-24]
    assert m.last_value == pytest.approx(float(r.series.iloc[-1]))
    assert m.mean == pytest.approx(float(r.series.mean()))


def test_metrics_declining_and_peaking():
    m = A.analyze(samples.sample("declining", seed=0)).metrics
    _assert_metrics_finite(m)
    assert m.direction == "down"
    assert m.total_change_pct is not None and m.total_change_pct < -60
    assert m.slope < 0 and m.drawdown_pct < -50
    assert m.momentum > 0                       # decline is decelerating
    m = A.analyze(samples.sample("peaking", seed=0)).metrics
    assert m.momentum < 0                       # growth has slowed
    assert m.recent_change_pct is not None and abs(m.recent_change_pct) < 10


@pytest.mark.parametrize("seed", SEEDS)
def test_metrics_direction_flat_on_seasonal(seed):
    m = A.analyze(samples.sample("seasonal", seed=seed)).metrics
    assert m.direction == "flat"
    assert m.seasonal_strength > 0.6


def test_metrics_cagr_exact_exponential():
    t = np.arange(61)
    m = A.analyze(from_values(100 * 1.1 ** (t / 12))).metrics
    assert m.cagr_pct == pytest.approx(10.0, abs=0.05)
    assert m.total_change_pct == pytest.approx(61.05, abs=0.2)
    assert m.recent_change_pct == pytest.approx(10.0, abs=0.2)


def test_metrics_growth_none_when_base_not_positive():
    m = A.analyze(from_values(np.linspace(-10, 10, 48))).metrics
    assert m.total_change_pct is None and m.cagr_pct is None
    m = A.analyze(from_values(np.r_[np.zeros(12), np.linspace(0, 50, 36)])).metrics
    assert m.total_change_pct is None and m.cagr_pct is None
    m = A.analyze(from_values(-50 - 0.5 * np.arange(48))).metrics
    assert m.total_change_pct is None and m.recent_change_pct is None and m.cagr_pct is None
    assert m.direction == "down" and m.drawdown_pct <= 0


def test_metrics_cagr_needs_about_a_year():
    m = A.analyze(from_values(np.arange(1.0, 9.0))).metrics   # 7 months
    assert m.cagr_pct is None
    assert m.total_change_pct is not None


@pytest.mark.parametrize("values", [
    [5.0] * 36,                                   # constant
    [0.0] * 36,                                   # all zeros
    [0.0] * 20 + [10.0] + [0.0] * 15,             # zeros with a spike
    [0, 3, 0, 0, 5, 0, 2, 0, 0, 0, 4, 0] * 3,     # intermittent with zeros
    list(-50 - np.arange(48) * 0.5 + np.sin(np.arange(48))),   # negative
    [1.0, 2.0, 3.0],                              # n = 3
    [2.0, 2.0, 2.0],
    [1.0, 5.0],                                   # n = 2
    [1e9, 1.2e9, 0.9e9, 1.4e9, 1.1e9, 1.6e9, 1.3e9, 1.7e9, 1.5e9, 1.9e9],  # huge values
])
def test_metrics_all_finite_on_edge_series(values):
    s = from_values(values)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        r = A.analyze(s)
    _assert_metrics_finite(r.metrics)
    assert r.metrics.n_obs == len(values)
    d = r.decomposition
    for comp in (d.trend, d.seasonal, d.resid):
        assert np.isfinite(comp.to_numpy()).all()
    assert 0.0 <= r.phase.confidence <= 1.0
    assert r.phase.label in PHASES


def test_metrics_constant_series_values():
    m = A.analyze(from_values([5.0] * 36)).metrics
    assert m.total_change_pct == 0.0 and m.recent_change_pct == 0.0
    assert m.drawdown_pct == 0.0 and m.volatility == pytest.approx(0.0, abs=1e-12)
    assert m.direction == "flat" and m.trend_strength == 0.0


# =========================================================================== lifecycle


@pytest.mark.parametrize("name", list(samples.EXPECTED_PHASE))
def test_lifecycle_monthly_all_archetypes(name):
    expected = samples.EXPECTED_PHASE[name]
    labels = [A.analyze(samples.sample(name, n=120, freq="M", seed=s), freq="M").phase.label
              for s in SEEDS]
    assert sum(lb in expected for lb in labels) >= 4, labels


@pytest.mark.parametrize("freq,n", [("W", 156), ("D", 400)])
@pytest.mark.parametrize("name", list(samples.EXPECTED_PHASE))
def test_lifecycle_weekly_and_daily(name, freq, n):
    expected = samples.EXPECTED_PHASE[name]
    labels = [A.analyze(samples.sample(name, n=n, freq=freq, seed=s), freq=freq).phase.label
              for s in SEEDS]
    assert sum(lb in expected for lb in labels) >= 3, labels


@pytest.mark.parametrize("name", list(samples.EXPECTED_PHASE))
def test_lifecycle_reasons_are_korean_with_numbers(name):
    for freq, n in (("M", 120), ("D", 400)):
        ph = A.analyze(samples.sample(name, n=n, freq=freq, seed=0), freq=freq).phase
        assert isinstance(ph, LifecyclePhase)
        assert ph.label in PHASES
        assert 0.0 <= ph.confidence <= 1.0
        assert 2 <= len(ph.reasons) <= 4
        for sentence in ph.reasons:
            assert HANGUL.search(sentence), sentence
            assert DIGIT.search(sentence), sentence
            assert "nan" not in sentence.lower() and "inf" not in sentence.lower()


def test_lifecycle_confident_on_clear_cases_less_on_short_data():
    clear = A.analyze(samples.sample("growing", seed=0)).phase
    assert clear.confidence >= 0.7
    short = A.analyze(samples.sample("growing", n=10, seed=0)).phase
    assert short.confidence < clear.confidence


def test_lifecycle_long_settled_decline_is_stable_but_recent_fad_is_declining():
    t = np.arange(240)
    old_spike = 10 + 90 * np.exp(-0.5 * ((t - 20) / 6) ** 2)
    rng = np.random.default_rng(0)
    ph = A.analyze(from_values(old_spike + rng.normal(0, 0.5, 240))).phase
    assert ph.label == "stable"
    assert A.analyze(samples.sample("fad", seed=0)).phase.label == "declining"


def test_lifecycle_phase_callable_directly():
    s = samples.sample("reviving", seed=0)
    d = A.decompose(s, 12)
    m = A.compute_metrics(s, "M", d)
    cps = A.detect_changepoints(s, period=12)
    ph = A.lifecycle_phase(s, "M", d, m, cps)
    assert ph.label == "reviving"


# =========================================================================== analyze


def test_analyze_result_structure():
    s = samples.sample("breakout", seed=0)
    r = A.analyze(s, name="AI 에이전트")
    assert isinstance(r, AnalysisResult)
    assert r.name == "AI 에이전트" and r.series.name == "AI 에이전트"
    assert r.freq == "M" and r.season_length == 12
    assert r.decomposition.period == 12
    assert len(r.series) == 120
    assert all(c.kind in CHANGEPOINT_KINDS for c in r.changepoints)
    assert r.phase.label in samples.EXPECTED_PHASE["breakout"]
    assert A.analyze(s).name == "breakout"


def test_analyze_accepts_raw_irregular_unsorted_input():
    rng = np.random.default_rng(0)
    days = np.sort(rng.choice(900, 300, replace=False))
    dates = pd.Timestamp("2020-01-01") + pd.to_timedelta(days, unit="D")
    vals = 100 + 0.2 * np.arange(300) + rng.normal(0, 3, 300)
    vals[[5, 50, 100]] = np.nan
    raw = pd.Series(vals, index=dates).sample(frac=1.0, random_state=1)
    r = A.analyze(raw)
    assert r.freq == "D"
    assert r.series.index.is_monotonic_increasing
    assert np.isfinite(r.series.to_numpy()).all()
    assert r.metrics.direction == "up"
    assert r.phase.label == "growing"
    _assert_metrics_finite(r.metrics)


def test_analyze_string_index_with_nan_and_freq_override():
    raw = pd.Series([3.0, np.nan, 1.0, 2.0, 4.0, 5.0],
                    index=["2021-03-01", "2021-02-01", "2021-01-01", "2021-04-01",
                           "2021-05-01", "2021-06-01"])
    r = A.analyze(raw, freq="M")
    assert r.freq == "M"
    assert len(r.series) == 6
    assert r.changepoints == []
    _assert_metrics_finite(r.metrics)


@pytest.mark.parametrize("values", [[1.0, 2.0, 3.0], [5.0, 1.0, 9.0], [4.0, 4.0, 4.0], [2.0, 3.0]])
def test_analyze_tiny_series_degrades_gracefully(values):
    r = A.analyze(from_values(values))
    assert r.changepoints == []
    assert r.anomalies == []
    assert r.phase.label in ("stable", "volatile")
    assert r.phase.confidence <= 0.3
    assert 2 <= len(r.phase.reasons) <= 4
    assert all(HANGUL.search(x) and DIGIT.search(x) for x in r.phase.reasons)
    _assert_metrics_finite(r.metrics)


def test_analyze_daily_730_is_fast():
    s = samples.sample("growing", n=730, freq="D", seed=0)
    A.analyze(s, freq="D")  # warm-up
    t0 = time.perf_counter()
    r = A.analyze(s, freq="D")
    assert time.perf_counter() - t0 < 2.0
    assert r.phase.label == "growing"


def test_analyze_is_deterministic():
    s = samples.sample("volatile", n=200, freq="W", seed=4)
    a, b = A.analyze(s), A.analyze(s)
    pd.testing.assert_series_equal(a.decomposition.trend, b.decomposition.trend)
    assert a.metrics == b.metrics
    assert a.phase == b.phase
    assert a.changepoints == b.changepoints


def test_analyze_clean_samples_emit_no_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for name in samples.list_samples():
            A.analyze(samples.sample(name, seed=0))
            A.analyze(samples.sample(name, n=60, freq="Q", seed=0), freq="Q")
            A.analyze(samples.sample(name, n=30, freq="Y", seed=0), freq="Y")


# =========================================================================== adversarial regressions
# Each test below failed on the first implementation (see the review report).


def _brute_kendall_s(y: np.ndarray) -> int:
    n, s = len(y), 0
    for start in range(0, n, 400):
        rows = np.arange(start, min(n, start + 400))
        sg = np.sign(y[None, :] - y[rows, None])
        s += int((sg * (np.arange(n)[None, :] > rows[:, None])).sum())
    return s


@pytest.mark.parametrize("k", [1e-9, 1e9])
@pytest.mark.parametrize("name", ["growing", "fad", "seasonal", "breakout", "emerging", "volatile"])
def test_results_do_not_depend_on_units(name, k):
    # Google-Ngram-like frequencies (~1e-9) or revenues (~1e9): only the unit differs.
    # (absolute 1e-12 / 1e-9 tolerances used to zero trend/seasonal strength and the
    # growth numbers, and changed the decomposition, direction and phase)
    s = samples.sample(name, seed=0)
    a, b = A.analyze(s), A.analyze(s * k)
    big = float(np.abs(s).max())
    for ca, cb in ((a.decomposition.trend, b.decomposition.trend),
                   (a.decomposition.seasonal, b.decomposition.seasonal),
                   (a.decomposition.resid, b.decomposition.resid)):
        np.testing.assert_allclose(cb.to_numpy(), k * ca.to_numpy(), rtol=1e-7, atol=1e-9 * k * big)
    ma, mb = a.metrics, b.metrics
    for f in ("total_change_pct", "cagr_pct", "recent_change_pct", "slope_pct", "momentum",
              "acceleration", "volatility", "trend_strength", "seasonal_strength", "mk_tau",
              "mk_pvalue", "drawdown_pct"):
        va, vb = getattr(ma, f), getattr(mb, f)
        assert (va is None) == (vb is None), f
        if va is not None:
            assert vb == pytest.approx(va, rel=1e-6, abs=1e-9), f
    for f in ("slope", "last_value", "mean", "peak_value"):
        assert getattr(mb, f) == pytest.approx(k * getattr(ma, f), rel=1e-6, abs=1e-9 * k * big), f
    assert mb.direction == ma.direction and mb.peak_date == ma.peak_date
    assert [(c.index, c.kind) for c in b.changepoints] == [(c.index, c.kind) for c in a.changepoints]
    for ca, cb in zip(a.changepoints, b.changepoints, strict=True):
        assert cb.slope_after == pytest.approx(k * ca.slope_after, rel=1e-6, abs=1e-9 * k * big)
        assert cb.magnitude == pytest.approx(ca.magnitude, rel=1e-6)
    assert [x.index for x in b.anomalies] == [x.index for x in a.anomalies]
    assert b.phase.label == a.phase.label
    assert b.phase.reasons == a.phase.reasons
    assert b.phase.confidence == pytest.approx(a.phase.confidence, rel=1e-9)


@pytest.mark.parametrize("n,max_hits", [(12, 6), (20, 8), (40, 3)])
def test_short_white_noise_rarely_gets_changepoints(n, max_hits):
    # was 14/40, 24/40 and 7/40 series with spurious breaks (despike collapse +
    # no small-sample correction of the BIC criterion)
    hits = sum(bool(A.detect_changepoints(
        from_values(50 + np.random.default_rng(seed).normal(0, 5, n), freq="D"), period=7))
        for seed in range(40))
    assert hits <= max_hits


def test_despike_ignores_running_median_ties_but_removes_spikes():
    # running-median residuals are exactly 0 for ~20% of noise points and for whole
    # constant stretches; they used to collapse the MAD so ordinary noise was "despiked"
    replaced = 0
    for seed in range(50):
        z = np.random.default_rng(seed).normal(size=20)
        replaced += int(np.sum(A._despike(z) != z))
    assert replaced <= 2
    z = np.r_[np.zeros(30), np.random.default_rng(1).normal(size=30)]
    np.testing.assert_array_equal(A._despike(z), z)
    z = np.random.default_rng(2).normal(size=60)
    z[25] += 30.0
    out = A._despike(z)
    assert out[25] != z[25] and int(np.sum(out != z)) == 1


def test_robust_sd_does_not_collapse_on_near_ties():
    g = np.random.default_rng(0).normal(0, 2, 5000)
    # the guards never bind on Gaussian data: plain 1.4826 * MAD
    assert A._robust_sd(g) == pytest.approx(1.4826 * np.median(np.abs(g - np.median(g))))
    rng = np.random.default_rng(1)
    mixed = np.r_[1e-9 * rng.normal(size=600), rng.normal(0, 1, 400)]  # 60% (near) ties
    assert A._robust_sd(mixed) > 0.25


@pytest.mark.parametrize("seed", range(3))
def test_constant_stretch_next_to_noise_is_stable_without_breaks(seed):
    rng = np.random.default_rng(seed)
    noise = 5 + rng.normal(0, 0.5, 100)
    for y in (np.r_[np.full(100, 5.0), noise], np.r_[noise, np.full(100, 5.0)]):
        r = A.analyze(from_values(y, freq="D"), freq="D")
        assert r.phase.label == "stable"
        assert r.changepoints == []
        assert len(r.anomalies) <= 2


@pytest.mark.parametrize("seed", range(3))
def test_zero_inflated_counts_keep_level_and_few_anomalies(seed):
    # intermittent counts (80% zeros): the robust trend used to collapse onto 0,
    # flagging ~20% of points (|z| up to 1e5) and calling the flat series "declining"
    rng = np.random.default_rng(seed)
    y = np.where(rng.random(365) < 0.2, rng.integers(1, 6, 365), 0).astype(float)
    r = A.analyze(from_values(y, freq="D"), freq="D")
    assert 0.6 * y.mean() < r.decomposition.trend.mean() < 1.4 * y.mean()
    assert r.phase.label in ("volatile", "stable")
    assert len(r.anomalies) <= 0.05 * len(y)
    assert all(abs(a.zscore) < 100 for a in r.anomalies)
    y[200] = 40.0                                    # a genuine burst is still flagged
    assert 200 in [a.index for a in A.analyze(from_values(y, freq="D"), freq="D").anomalies]


def test_lifecycle_short_pure_noise_mostly_stable():
    # short series: the decomposition residual holds only ~1/3 of the noise, which
    # made noise look like a significant trend (model-free noise check added)
    labels = [A.analyze(from_values(50 + np.random.default_rng(s).normal(0, 5, 20), freq="D"),
                        freq="D").phase.label for s in range(40)]
    assert sum(lb in ("stable", "volatile") for lb in labels) >= 35, labels


def test_lifecycle_slow_noisy_count_growth_is_growing():
    # Poisson counts 1 -> 6 per day: an overwhelmingly significant 6x rise must not be
    # called "volatile" because the last weeks' (insignificant) slope is slightly < 0
    ok = 0
    for s in range(10):
        y = np.random.default_rng(s).poisson(np.linspace(1, 6, 365)).astype(float)
        ok += A.analyze(from_values(y, freq="D"), freq="D").phase.label in ("growing", "emerging")
    assert ok >= 7


def test_mann_kendall_long_input_exact_and_fast():
    rng = np.random.default_rng(3)
    for y in (rng.normal(size=2500), rng.integers(0, 6, 3001).astype(float),
              np.arange(1500) + rng.normal(0, 200, 1500)):
        n = len(y)
        s = _brute_kendall_s(y)
        _, cnt = np.unique(y, return_counts=True)
        var = (n * (n - 1) * (2 * n + 5) - float(np.sum(cnt * (cnt - 1) * (2 * cnt + 5)))) / 18
        z = (s - np.sign(s)) / math.sqrt(var)
        tau, p = A.mann_kendall(y)
        assert tau == pytest.approx(s / (n * (n - 1) / 2), abs=1e-15)
        assert p == pytest.approx(math.erfc(abs(z) / math.sqrt(2)), rel=1e-9, abs=1e-300)
    y = rng.normal(size=100_000)
    t0 = time.perf_counter()
    A.mann_kendall(y)
    assert time.perf_counter() - t0 < 1.0          # was O(n^2): ~40 s at this size


def test_analyze_long_daily_series_fast():
    s = samples.sample("growing", n=40_000, freq="D", seed=0)
    t0 = time.perf_counter()
    r = A.analyze(s, freq="D")
    assert time.perf_counter() - t0 < 3.0          # was ~6 s (quadratic Mann-Kendall)
    assert r.phase.label == "growing"
