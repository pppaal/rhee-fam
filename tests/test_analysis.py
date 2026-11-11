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


# =========================================================================== fixer regressions (round 2)
# Known issues A1-A4 and verifier findings; each test failed before the fix.

MINUS = "−"


def _ar1(n: int, phi: float, rng: np.random.Generator, marginal_sd: float = 5.0) -> np.ndarray:
    e = rng.normal(0.0, marginal_sd * math.sqrt(max(1.0 - phi * phi, 1e-12)), n)
    x = np.empty(n)
    x[0] = rng.normal(0.0, marginal_sd)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + e[i]
    return x


def test_short_nonnegative_series_keep_a_nonnegative_trend():
    # A1: the edge extrapolation of a short fad ended at -1.03 (last value 8.29),
    # giving a total/recent change of -123%
    s = samples.sample("fad", n=10, freq="M", seed=0)
    r = A.analyze(s, freq="M")
    tr = r.decomposition.trend.to_numpy()
    assert tr.min() >= 0.0
    assert tr[-1] >= 0.5 * float(s.iloc[-3:].min())          # inside the data at the edge
    assert r.metrics.total_change_pct is not None and r.metrics.total_change_pct > -100.0
    np.testing.assert_allclose((r.decomposition.trend + r.decomposition.seasonal
                                + r.decomposition.resid).to_numpy(), s.to_numpy(), atol=1e-9)
    neg = 0
    for seed in range(5):
        for n in (8, 9, 10, 12, 16):
            for freq in ("M", "Q", "Y"):
                d = A.analyze(samples.sample("fad", n=n, freq=freq, seed=seed), freq=freq).decomposition
                neg += int(d.trend.min() < 0)
    assert neg == 0


def test_reasons_use_typographic_minus_and_narrative_number_format():
    # A2 + finding 6: "-42.3%" (ASCII hyphen), "tau=0.95, p=2.2e-53", "변동계수" that
    # differed from the KPI's 변동계수, "고점을 형성" for a peak at the last point
    seen_minus = False
    for name in ("fad", "declining", "growing", "volatile", "peaking", "emerging"):
        ph = A.analyze(samples.sample(name, seed=0)).phase
        for sentence in ph.reasons:
            assert not re.search(r"(?<![\w])-\d", sentence), sentence
            assert "tau=" not in sentence and "p=" not in sentence, sentence
            assert "변동계수" not in sentence, sentence
            seen_minus |= MINUS in sentence
    assert seen_minus
    growing = " ".join(A.analyze(samples.sample("growing", seed=0)).phase.reasons)
    assert "τ = 0.9" in growing and "p < 0.001" in growing
    # a trend whose maximum is the latest point has not "formed" a peak yet
    for seed in range(5):
        r = A.analyze(samples.sample("peaking", seed=seed))
        if int(np.argmax(r.decomposition.trend.to_numpy())) == len(r.series) - 1:
            assert not any("고점을 형성" in x for x in r.phase.reasons), r.phase.reasons


@pytest.mark.parametrize("freq,n", [("M", 120), ("D", 400)])
def test_flat_ar1_noise_is_not_a_trend(freq, n):
    # A3: flat series with AR(1) phi=0.7 noise got growing/declining labels in ~1/4
    # of the cases (and MK p < 0.05 in ~40%): no correction for serial correlation
    labels, sig = [], 0
    idx = pd.date_range("2015-01-01", periods=n, freq={"M": "MS", "D": "D"}[freq])
    for seed in range(20):
        y = 100 + _ar1(n, 0.7, np.random.default_rng(seed))
        r = A.analyze(pd.Series(y, index=idx), freq=freq)
        labels.append(r.phase.label)
        sig += r.metrics.mk_pvalue < 0.05
    assert sum(lb not in ("stable", "volatile") for lb in labels) <= 1, labels
    assert sig <= 4


def test_white_noise_gets_little_seasonality():
    # A4: noise-only seasonal patterns (James-Stein (p-3)/p shrinkage is weak for
    # p = 4 and for 2-3 cycles)
    for freq, n, max_frac in (("M", 120, 0.0), ("M", 36, 0.15), ("Q", 40, 0.05)):
        ss = []
        for seed in range(20):
            y = np.random.default_rng(seed).normal(100, 10, n)
            idx = pd.date_range("2010-01-01", periods=n, freq={"M": "MS", "Q": "QS"}[freq])
            ss.append(A.analyze(pd.Series(y, index=idx), freq=freq).metrics.seasonal_strength)
        assert np.mean(np.array(ss) >= 0.3) <= max_frac, (freq, n, np.round(ss, 2))


def test_random_walk_is_not_a_confident_trend():
    # finding 1: driftless geometric random walks got MK p ~ 1e-60 in 29/30 runs,
    # exactly 5 (= the cap) changepoints each and growing/declining/peaking at ~0.8
    labels, conf, ncp, sig = [], [], [], 0
    idx = pd.date_range("2023-01-01", periods=730, freq="D")
    for seed in range(20):
        rng = np.random.default_rng(seed)
        y = 100 * np.exp(np.cumsum(rng.normal(0, 0.018, 730)))
        r = A.analyze(pd.Series(y, index=idx))
        labels.append(r.phase.label)
        conf.append(r.phase.confidence)
        ncp.append(len(r.changepoints))
        sig += r.metrics.mk_pvalue < 0.05
    assert sig <= 4
    assert np.median(ncp) <= 1 and max(ncp) <= 4, ncp
    assert sum(lb in ("volatile", "stable") for lb in labels) >= 14, labels
    assert np.mean(conf) < 0.6
    walk = [x for x in A.analyze(pd.Series(100 * np.exp(np.cumsum(
        np.random.default_rng(0).normal(0, 0.018, 730))), index=idx)).phase.reasons]
    assert any("무작위 행보" in x for x in walk), walk


def test_serial_dependence_helpers():
    # exact AR(1) slope variance (incl. the random-walk limit) vs. brute force
    h = 25
    x = np.arange(h) - (h - 1) / 2
    for phi in (0.0, 0.6, 1.0):
        if phi < 1:
            g = phi ** np.abs(np.subtract.outer(np.arange(h), np.arange(h))) / (1 - phi ** 2)
        else:
            g = np.minimum.outer(np.arange(1, h + 1), np.arange(1, h + 1)).astype(float)
        brute = x @ g @ x / np.sum(x * x) ** 2
        assert A._ols_slope_var(h, phi, 1.0) == pytest.approx(brute, rel=1e-10)
    assert A._mk_variance_ratio(0.0, 100) == 1.0
    assert A._mk_variance_ratio(1.0, 100) == pytest.approx(1 + 2 * np.sum(
        (100 - np.arange(1, 98)) * (99 - np.arange(1, 98)) * (98 - np.arange(1, 98))) / (100 * 99 * 98))
    y = np.random.default_rng(3).normal(size=150)
    assert A._trend_test(y, 0.0) == A.mann_kendall(y)
    walks = [A._noise_model(np.cumsum(np.random.default_rng(k).normal(size=500)), 1)[0]
             for k in range(1, 6)]
    assert np.median(walks) > 0.95 and min(walks) > 0.75        # ~ unit root
    assert A._noise_model(np.random.default_rng(2).normal(size=500), 1)[0] == 0.0
    ars = [A._noise_model(_ar1(500, 0.6, np.random.default_rng(k)), 1)[0] for k in range(4, 9)]
    assert 0.45 < np.median(ars) < 0.75


def test_multiplicative_seasonality_is_decomposed_in_log_space():
    # finding 2: (a) a decaying product with +-20% seasonality got a trend that
    # rose over the last 5 months (+15..+72% end bias, recent change -12% vs. a true
    # -26%); (b) every December of growing retail sales was flagged as an anomaly
    idx = pd.date_range("2018-01-01", periods=84, freq="MS")
    t = np.arange(84)
    rng = np.random.default_rng(0)
    lvl = 5000 * np.exp(-0.025 * t)
    y = lvl * (1 + 0.2 * np.sin(2 * np.pi * (np.asarray(idx.month) - 3) / 12)) * np.exp(rng.normal(0, 0.05, 84))
    s = pd.Series(y, index=idx)
    r = A.analyze(s)
    d = r.decomposition
    tr = d.trend.to_numpy()
    assert A._is_multiplicative(d)
    np.testing.assert_allclose((d.trend + d.seasonal + d.resid).to_numpy(), y, rtol=0, atol=1e-9 * y.max())
    assert np.all(np.diff(tr[-6:]) < 0)                         # still falling at the end
    assert abs(tr[-1] / lvl[-1] - 1) < 0.08
    assert r.metrics.recent_change_pct == pytest.approx((math.exp(-0.3) - 1) * 100, abs=5)
    assert not any(c.kind == "level_shift" for c in r.changepoints)
    idx2 = pd.date_range("2017-01-01", periods=96, freq="MS")
    seas = np.array([.85, .82, .95, .97, 1, .98, .99, 1.02, .98, 1.03, 1.15, 1.65])[np.asarray(idx2.month) - 1]
    dec_flagged = 0
    for seed in range(15):
        y2 = 1000 * 1.06 ** (np.arange(96) / 12) * seas * np.exp(np.random.default_rng(seed).normal(0, 0.025, 96))
        an = A.analyze(pd.Series(y2, index=idx2)).anomalies
        dec_flagged += any(x.date.month == 12 for x in an)
    assert dec_flagged <= 3
    # additive samples stay additive (exactly periodic seasonal component)
    assert not A._is_multiplicative(A.decompose(samples.sample("growing", seed=0), 12))
    assert not A._is_multiplicative(A.decompose(samples.sample("seasonal", seed=0), 12))


def test_metrics_are_robust_to_flagged_spikes():
    # finding 3: four one-day spikes turned a clear weekday pattern into "almost no
    # seasonality" (0.60 -> 0.04), volatility 11% -> 63% and a -88% drawdown
    idx = pd.date_range("2022-01-01", "2024-12-31", freq="D")
    n, t = len(idx), np.arange(len(idx))
    dow = np.array([.10, .12, .11, .08, 0, -.25, -.20])[np.asarray(idx.dayofweek)]
    for seed in range(2):
        rng = np.random.default_rng(seed)
        ar = np.zeros(n)
        e = rng.normal(0, 0.08, n)
        for i in range(1, n):
            ar[i] = 0.7 * ar[i - 1] + e[i]
        clean = np.exp(np.log(2000) + 0.35 * t / n + dow + ar)
        spiky = clean.copy()
        spiky[[200, 480, 700, 950]] *= [12, 6, 20, 8]
        mc = A.analyze(pd.Series(clean, index=idx)).metrics
        ra = A.analyze(pd.Series(spiky, index=idx))
        ms = ra.metrics
        assert {200, 480, 700, 950} <= {a.index for a in ra.anomalies}
        assert ms.seasonal_strength > 0.8 * mc.seasonal_strength > 0.3
        assert ms.volatility < 1.3 * mc.volatility
        assert ms.trend_strength > 0.7 * mc.trend_strength
        assert ms.drawdown_pct > mc.drawdown_pct - 10
        assert ms.peak_date not in {idx[i] for i in (200, 480, 700, 950)}


def test_level_shift_has_no_neighbouring_anomalies_and_is_stable_after():
    # finding 4: the smoother pre-empted the step, so ~3 points next to it were
    # flagged as "temporary spikes/dips" (30/30 seeds) and the flat series was
    # called growing/peaking
    near, labels = 0, []
    idx = pd.date_range("2021-01-03", periods=156, freq="W-SUN")
    for seed in range(10):
        y = 200 + np.random.default_rng(seed).normal(0, 6, 156)
        y[100:] += 60
        r = A.analyze(pd.Series(y, index=idx))
        assert any(c.kind == "level_shift" and abs(c.index - 100) <= 2 for c in r.changepoints)
        near += sum(98 <= a.index <= 102 for a in r.anomalies)
        labels.append(r.phase.label)
        tr = r.decomposition.trend.to_numpy()
        assert tr[101] - tr[98] > 45                              # the trend jumps
    assert near <= 1
    assert labels.count("stable") >= 9, labels
    reasons = " ".join(r.phase.reasons)
    assert "수준이 한 단계 높아졌습니다" in reasons and "안정" in reasons
    # annual, GDP-like: a permanent drop must not make the year before a "spike"
    idx_y = pd.date_range("1980-01-01", periods=44, freq="YS")
    flagged = 0
    for seed in range(6):
        rng = np.random.default_rng(seed)
        g = 100 * np.exp(0.025 * np.arange(44) + rng.normal(0, 0.006, 44))
        g[28:] *= 0.92
        flagged += sum(26 <= a.index <= 29 for a in A.analyze(pd.Series(g, index=idx_y), freq="Y").anomalies)
    assert flagged <= 2                                       # was 2 per series


def test_decompose_breaks_argument_lets_the_trend_jump():
    rng = np.random.default_rng(0)
    y = np.r_[np.full(60, 10.0), np.full(60, 30.0)] + rng.normal(0, 1, 120)
    s = from_values(y)
    smooth, piecewise = A.decompose(s, 1), A.decompose(s, 1, breaks=[60])
    np.testing.assert_allclose((piecewise.trend + piecewise.resid).to_numpy(), y, atol=1e-9)
    assert piecewise.trend.iloc[60] - piecewise.trend.iloc[59] > 17
    assert np.abs(piecewise.resid.to_numpy()[55:65]).max() < np.abs(smooth.resid.to_numpy()[55:65]).max()
    # a "level shift" found on a smooth curve does not get a jump (see finding 2)
    assert A.decompose(s, 1, breaks=None).trend.equals(smooth.trend)


def test_noisy_series_edges_do_not_drive_growth_numbers():
    # finding 5: the one-sided edge fit copied edge noise -> total change +128.6% on
    # pure noise (Q, seed 1), recent change +28% / momentum +2.4 while the last 12
    # months were 13% below the previous 12 (M, seed 4)
    m = A.analyze(samples.sample("volatile", n=40, freq="Q", seed=1), freq="Q").metrics
    assert abs(m.total_change_pct) < 70
    m = A.analyze(samples.sample("volatile", seed=4)).metrics
    assert m.recent_change_pct < 20 and m.momentum < 2.0
    for freq, n, tot_max, rec_max in (("M", 120, 30, 20), ("W", 156, 30, 25)):
        tot, rec = [], []
        for seed in range(10):
            mm = A.analyze(samples.sample("volatile", n=n, freq=freq, seed=seed), freq=freq).metrics
            tot.append(abs(mm.total_change_pct))
            rec.append(abs(mm.recent_change_pct))
        assert max(tot) < tot_max and max(rec) < rec_max, (freq, np.round(tot), np.round(rec))
    # one-sided edge variance: ~1/3 lower than the plain local-linear end point
    errs = []
    for seed in range(100):
        y = 50 + 0.1 * np.arange(120) + np.random.default_rng(seed).normal(0, 5, 120)
        errs.append(A.decompose(from_values(y), 1).trend.iloc[-1] - (50 + 0.1 * 119))
    assert np.std(errs) < 2.4                                 # was ~3.0 (sd of the noise: 5)


# =========================================================================== fixer regressions (round 3)
# Each test failed before the round-3 fix (AN1, AN2, AN9).


def test_reason_at_the_trend_peak_does_not_say_100_percent():
    # AN1: "현재 수준은 추세 고점의 100% 수준입니다" read oddly for a series at its high
    for seed in range(3):
        r = A.analyze(samples.sample("growing", seed=seed))
        assert r.phase.label == "growing"
        assert any(re.fullmatch(r"현재 추세는 고점\(\d{4}-\d{2}\) 수준입니다\.", x) for x in r.phase.reasons), \
            r.phase.reasons
        assert not any(re.search(r"고점의 (99\.5|99\.[6-9]|100)\d*% 수준", x) for x in r.phase.reasons)


def test_yearly_reasons_write_the_year_counter():
    # AN2: "2050에 구조 변화", "2042에 저점" -> "2050년에", "2042년에" (bracketed years stay bare)
    reasons = []
    for name in ("emerging", "reviving"):
        reasons += A.analyze(samples.sample(name, n=40, freq="Y"), freq="Y").phase.reasons
    idx = pd.date_range("1990-01-01", periods=30, freq="YS")
    y = 100 + np.random.default_rng(0).normal(0, 2, 30)
    y[18:] += 30
    reasons += A.analyze(pd.Series(y, index=idx), freq="Y").phase.reasons
    text = " ".join(reasons)
    assert re.search(r"\d{4}년에 구조 변화", text)
    assert re.search(r"\d{4}년에 저점을 찍은 뒤", text)
    assert "2008년에 수준이 한 단계 높아졌습니다" in text
    assert not re.search(r"\d{4}에 ", text), text
    assert re.search(r"고점\(\d{4}\)", text)                 # in brackets: no counter needed
    # monthly dates are unchanged
    m = " ".join(A.analyze(samples.sample("reviving")).phase.reasons)
    assert re.search(r"\d{4}-\d{2}에 저점을 찍은 뒤", m)


def _white_noise_flags(freq: str, n: int, seeds) -> int:
    idx = pd.date_range("2015-01-04", periods=n, freq={"W": "W-SUN", "Q": "QS", "M": "MS"}[freq])
    return sum(len(A.analyze(pd.Series(100 + np.random.default_rng(s).normal(0, 5, n), index=idx),
                             freq=freq).anomalies) for s in seeds)


def test_few_cycle_seasonal_noise_is_not_flagged_as_anomalies():
    # AN9: with 3 weekly cycles (p = 52) the seasonal means absorb a third of every ordinary
    # point (and a 5-point trend window ~30 %), so the median residual understated the scale of
    # the points the robust fit leaves out: ~2.5 flags per pure-noise series at W156 (now ~0.45),
    # 1.6 at M36 (~0.5), 1.0 at Q40 (~0.4); over 20 seeds: 42 / 18 / 13 before
    seeds = range(20)
    assert _white_noise_flags("W", 156, seeds) <= 15
    assert _white_noise_flags("M", 36, seeds) <= 10
    assert _white_noise_flags("Q", 40, seeds) <= 10
    assert _white_noise_flags("M", 120, seeds) <= 8


def test_studentised_anomalies_still_find_real_spikes():
    # AN9: the correction must not hide genuine spikes: an 8-sigma one-off spike is found at
    # any position (edges included), and the leverage estimate matches the decomposition's
    for freq, n, pf in (("W", 156, "W-SUN"), ("M", 120, "MS"), ("D", 365, "D")):
        idx = pd.date_range("2015-01-04", periods=n, freq=pf)
        for seed, pos in ((0, 0), (1, n // 3), (2, n - 2), (3, n - 1)):
            y = 100 + np.random.default_rng(seed).normal(0, 5, n)
            y[pos] += (-1) ** seed * 40
            s = pd.Series(y, index=idx)
            found = A.detect_anomalies(s, A.decompose(s, season_length(freq)))
            assert pos in [a.index for a in found], (freq, seed, pos)
    # the relative residual variance follows the decomposition's actual leverage
    idx = pd.date_range("2015-01-04", periods=156, freq="W-SUN")
    y = 100 + np.random.default_rng(3).normal(0, 5, 156)
    dec = A.decompose(pd.Series(y, index=idx), 52)
    r0 = dec.resid.to_numpy()
    lev = []
    for i in range(0, 156, 13):
        y2 = y.copy()
        y2[i] += 0.01
        lev.append(1 - (A.decompose(pd.Series(y2, index=idx), 52).resid.to_numpy()[i] - r0[i]) / 0.01)
    rn = r0 / A._unit(y, r0)
    f = A._resid_variance_factor(rn, (y - dec.trend.to_numpy()) / A._unit(y, r0),
                                 dec.seasonal.to_numpy() / A._unit(y, r0), 52, False)
    est = 1 - np.sqrt(np.clip(f[::13], 0, 1))           # F ~ (1 - h)^2 for ordinary points
    assert abs(np.median(lev) - np.median(est)) < 0.12, (np.median(lev), np.median(est))
    assert 0.15 < np.median(lev) < 0.45                   # the fit absorbs a lot of each point


# --------------------------------------------------------------------------- round 3 (final review)


def _quiet_analyze(series, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return A.analyze(series, **kw)


def test_recent_reason_quotes_the_trend_line_change():
    # UX-3: the reason divided the OLS change by the OLS intercept of the last h points:
    # emerging Y/12 read '+1076.8%' while the trend line rose +293 % over that stretch, and the
    # peaking sample '+0.2 %' over 18 months next to '+2.0 %' over the last 12
    for nm, freq, n in (("emerging", "Y", 12), ("peaking", "M", 120), ("fad", "M", 120),
                        ("declining", "M", 120)):
        a = _quiet_analyze(samples.sample(nm, freq=freq, n=n), freq=freq)
        T = a.decomposition.trend.to_numpy()
        found = 0
        for r in a.phase.reasons:
            mt = re.search(r"최근 ([\d,]+)\S+ 동안 추세가 ([+−][\d,]+\.\d)% 변했습니다", r)
            if mt:
                h = int(mt.group(1).replace(",", ""))
                quoted = float(mt.group(2).replace("−", "-").replace(",", ""))
                assert quoted == pytest.approx((T[-1] / T[-1 - h] - 1) * 100, abs=0.06), (nm, r)
                found += 1
        assert found, (nm, a.phase.reasons)
        if nm == "peaking":       # consistent with the 12-month change on the same trend line
            assert quoted == pytest.approx(a.metrics.recent_change_pct, abs=1.0)
    # thousands separators like everywhere else ("+1,076.8%", not "+1076.8%")
    assert A._pct_txt(1076.84) == "+1,076.8%" and A._pct_txt(-12.5) == "−12.5%"


def test_reason_numbers_take_no_reading_dependent_particle():
    # UX-14: '계절성 강도가 0.96로' (read "...구육" -> must be "0.96으로"); the n < 8 path had
    # '3에서 6로'.  The reasons now put no particle right after a number.
    a = _quiet_analyze(samples.sample("seasonal"))
    assert any("계절성 강도(0.96)가 높아 주기적 변동이 뚜렷합니다." == r for r in a.phase.reasons), a.phase.reasons
    s = pd.Series([3.0, 4.0, 4.5, 5.0, 6.0], index=pd.date_range("2020-01-01", periods=5, freq="MS"))
    short = _quiet_analyze(s)
    assert "관측 기간 동안 값이 3.00에서 6.00까지 +100.0% 변했습니다." in short.phase.reasons
    for r in a.phase.reasons + short.phase.reasons:
        assert not re.search(r"\d(?:로|으로)\b", r), r


def test_level_shift_from_zero_reason_has_no_python_float_format():
    # F11: revenue 0 before a launch, then ~2.4M a month: '(약 +2.392e+06)'
    y = np.r_[np.zeros(20), 2_400_000 + np.random.default_rng(0).normal(0, 60_000, 28)].round(0)
    a = _quiet_analyze(pd.Series(y, index=pd.date_range("2021-01-01", periods=48, freq="MS")))
    shift = next(r for r in a.phase.reasons if "수준이 한 단계" in r)
    assert "e+" not in shift and re.search(r"\(약 \+2,\d{3},\d{3}\)", shift), shift
    assert A._num_txt(1e9, signed=True) == "+1,000,000,000"
    assert A._num_txt(2.5e16) == "2.5×10¹⁶" and A._num_txt(-3.2e-7) == "−3.2×10⁻⁷"


def test_stable_reason_does_not_call_a_full_range_small():
    # F15: zeros then a single 1 -> 'stable' with "변동 폭이 최고 수준 대비 100.0%에 그칩니다"
    a = _quiet_analyze(pd.Series(np.r_[np.zeros(19), 1.0],
                                 index=pd.date_range("2020-01-01", periods=20, freq="MS")))
    joined = " ".join(a.phase.reasons)
    assert not re.search(r"(?:[3-9]\d|100)\.\d%에 그칩니다", joined), a.phase.reasons
    assert len(set(a.phase.reasons)) == len(a.phase.reasons)          # no duplicated reason
    # a genuinely flat series keeps the "only x %" evidence
    flat = _quiet_analyze(pd.Series(100 + np.random.default_rng(1).normal(0, 1, 60),
                                    index=pd.date_range("2020-01-01", periods=60, freq="MS")))
    if flat.phase.label == "stable":
        assert any("에 그칩니다" in r for r in flat.phase.reasons) or any("잡음" in r for r in flat.phase.reasons)


def test_phase_does_not_depend_on_where_zero_sits():
    # F3: the same seasonal series (±15 swing, noise 1.5) was 'volatile' 19/20 at a mean of
    # 0.3 or 5 (noise / mean|z| = 125 %) and 'stable' 20/20 at 12.5
    t = np.arange(120)
    idx = pd.date_range("2015-01-01", periods=120, freq="MS")
    for seed in range(6):
        noise = np.random.default_rng(seed).normal(0, 1.5, 120)
        labels = {off: _quiet_analyze(pd.Series(off + 15 * np.sin(2 * np.pi * (t - 3) / 12) + noise,
                                                index=idx)).phase.label for off in (0.3, 5.0, 12.5)}
        assert len(set(labels.values())) == 1 and "volatile" not in labels.values(), (seed, labels)
    # the reason names its basis when the data cross zero
    y = np.random.default_rng(0).normal(0, 1, 120)
    a = _quiet_analyze(pd.Series(y, index=idx))
    assert a.phase.label == "volatile"
    assert any("값의 변동 폭의" in r for r in a.phase.reasons), a.phase.reasons
    # positive ratio-scale data keep the classic noise / level ratio
    pos = _quiet_analyze(pd.Series(100 + 30 * np.random.default_rng(2).normal(0, 1, 120), index=idx))
    assert pos.phase.label == "volatile" and any("평균의" in r for r in pos.phase.reasons)


def test_noise_is_not_reported_as_seasonal():
    # UX-6: linear trend + white noise, 36 months: 11 of 40 had seasonal strength >= 0.3
    # ("계절성은 뚜렷합니다 (강도 0.67)"); the strength is now discounted by a significance test
    idx = pd.date_range("2021-01-01", periods=36, freq="MS")
    strengths = []
    for seed in range(40):
        y = 1200 + 25 * np.arange(36) + np.random.default_rng(seed).normal(0, 80, 36)
        strengths.append(_quiet_analyze(pd.Series(y, index=idx)).metrics.seasonal_strength)
    assert max(strengths) < 0.3, sorted(strengths)[-5:]
    # genuine seasonality keeps its strength, also with only 2 to 3 cycles
    for n in (24, 28, 36, 120):
        m = _quiet_analyze(samples.sample("seasonal", n=n)).metrics
        assert m.seasonal_strength > 0.8, (n, m.seasonal_strength)
    w = _quiet_analyze(samples.sample("seasonal", freq="W", n=156), freq="W").metrics
    assert w.seasonal_strength > 0.8
    # the evidence factor itself: 0 for noise, 1 for a clean pattern
    rng = np.random.default_rng(5)
    assert A._seasonality_evidence(rng.normal(0, 1, 36), 12) < 1.0
    assert A._seasonality_evidence(np.sin(2 * np.pi * np.arange(24) / 12) + 0.1 * rng.normal(0, 1, 24), 12) == 1.0
    assert A._seasonality_evidence(np.ones(36), 12) == 0.0


def test_momentum_is_relative_to_the_current_level():
    # UX-8: momentum was normalised by the whole-period mean: emerging '+15.73 %p/월' while its
    # trend grew ~4.4 % a month (yearly: +88.6 %p/년 vs +37.9 % in the last year)
    m = _quiet_analyze(samples.sample("emerging")).metrics
    monthly = ((1 + m.recent_change_pct / 100) ** (1 / 12) - 1) * 100
    assert 0 < m.momentum <= 1.2 * monthly, (m.momentum, monthly)
    y = _quiet_analyze(samples.sample("emerging", freq="Y", n=12), freq="Y").metrics
    assert 0 < y.momentum <= 1.2 * y.recent_change_pct, (y.momentum, y.recent_change_pct)
    # signs are unchanged: convex growth accelerates, a fad's collapse decelerates
    assert _quiet_analyze(samples.sample("fad")).metrics.momentum < 0
    assert _quiet_analyze(samples.sample("peaking")).metrics.momentum < 0


def _iid_demand(seed: int, n: int = 150) -> pd.Series:
    rng = np.random.default_rng(seed)
    y = np.where(rng.random(n) < 0.15, rng.integers(1, 6, n), 0).astype(float)
    return pd.Series(y, index=pd.date_range("2021-01-03", periods=n, freq="W-SUN"))


def test_intermittent_demand_gets_no_percent_changes_or_order_anomalies():
    # F2: iid demand (15 % non-zero, sizes 1..5, no trend): total change +3,192 %, CAGR +240 %,
    # '최근 52주 +231 %', and 39 of 40 series had "anomalies" (+1,350 %, z = +12.4) -- ordinary orders
    flagged = 0
    for seed in range(40):
        a = _quiet_analyze(_iid_demand(seed))
        m = a.metrics
        assert m.total_change_pct is None and m.cagr_pct is None and m.recent_change_pct is None, seed
        flagged += bool(a.anomalies)
    assert flagged <= 4, flagged
    # an unusually large order still is one: judged by its size among the non-zero demands
    s = _iid_demand(9)
    big = int(np.flatnonzero(s.to_numpy() > 0)[10])
    s.iloc[big] = 60.0
    a = _quiet_analyze(s)
    assert [an.index for an in a.anomalies] == [big]
    assert a.anomalies[0].expected == pytest.approx(np.exp(np.median(np.log(s[s > 0]))))
    assert A._is_intermittent(s.to_numpy(), 52)
    # the rule is the forecast's: a series that was 0 before a launch is not intermittent
    launch = np.r_[np.zeros(40), np.linspace(5, 50, 60)]
    assert not A._is_intermittent(launch, 12)
    from trendai import forecast as F
    for y in (s.to_numpy(), launch, _iid_demand(3).to_numpy()):
        assert A._is_intermittent(y, 52) == F._is_intermittent(y[-max(400, 8 * 52):], 52)


def test_edge_and_short_series_peaks_are_not_excluded_as_spikes():
    # UX-5 / report note: the latest value of an accelerating series (emerging Y/12: 72.0) was
    # flagged and left out of the peak ("최고치는 42.3(일회성 급등 제외)" while 72.0 is the
    # current value); on 8-point yearly series real peaks and starts were excluded too
    a = _quiet_analyze(samples.sample("emerging", freq="Y", n=12), freq="Y")
    assert a.metrics.peak_value == pytest.approx(float(a.series.iloc[-1]))
    assert a.metrics.drawdown_pct == 0.0
    for nm in ("emerging", "fad", "declining"):
        s = samples.sample(nm, freq="Y", n=8)
        m = _quiet_analyze(s, freq="Y").metrics
        assert m.peak_value == pytest.approx(float(s.max())), nm
    # a one-off spike inside a long series is still left out (AN4 behaviour kept)
    s = samples.sample("growing").copy()
    s.iloc[60] += 150.0
    m = _quiet_analyze(s).metrics
    assert m.peak_value < float(s.max())


# --------------------------------------------------------------------------- round 4


def _pnl(n: int = 48, start: float = 10.0, end: float = -10.0, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    y = np.linspace(start, end, n) + rng.normal(0, 1.0, n)
    return pd.Series(y, index=pd.date_range("2022-01-01", periods=n, freq="MS"))


def test_zero_crossing_series_get_no_percent_changes():
    # U3: a P&L falling from 10 to -10 read 'total change -183.6 %' (and '-196 %' with other
    # seeds); a percentage of a level that crossed zero is meaningless -> None
    for seed in range(3):
        m = _quiet_analyze(_pnl(seed=seed)).metrics
        assert m.total_change_pct is None and m.cagr_pct is None and m.recent_change_pct is None, seed
    # the trend crosses zero inside the recent window (12 months ago it was still positive)
    s = _pnl(n=24, start=10.0, end=-3.0, seed=1)
    a = _quiet_analyze(s)
    tr = a.decomposition.trend.to_numpy()
    assert tr[-13] > 0 > tr[-1]
    assert a.metrics.recent_change_pct is None and a.metrics.total_change_pct is None
    # the drawdown stays a float (typed so) but is flagged as meaningless
    m = a.metrics
    assert isinstance(m.drawdown_pct, float) and m.drawdown_pct < -100
    assert A.pct_change_undefined(m.peak_value, m.last_value, s) == "sign_change"
    # the phase reason gives the two levels instead of "현재 추세는 고점보다 183.6% 낮은 수준"
    assert not any(re.search(r"\d{3,}(\.\d)?% 낮", r) for r in a.phase.reasons), a.phase.reasons
    assert any("0 아래로 내려가 현재 −" in r for r in a.phase.reasons), a.phase.reasons


def test_near_zero_base_of_data_with_negatives_gets_no_percent_change():
    # U3: a series with negative values whose starting trend is within half a standard deviation
    # of 0 (here +1.3 against an SD of ~18) has no meaningful % change (it read +2,625 %, CAGR +133 %)
    t = np.arange(48.0)
    y = np.r_[np.linspace(0.6, -30, 24), np.linspace(-30, 35, 24)] + np.sin(t)
    s = pd.Series(y, index=pd.date_range("2020-01-01", periods=48, freq="MS"))
    a = _quiet_analyze(s)
    tr = a.decomposition.trend.to_numpy()
    assert 0 < tr[0] < 0.5 * np.std(y) and tr[-1] > 0
    assert a.metrics.total_change_pct is None and a.metrics.cagr_pct is None
    assert A.pct_change_undefined(tr[0], tr[-1], s) == "near_zero"
    # far enough from 0 (more than half an SD), a series with negatives keeps its percentages
    assert A.pct_change_undefined(30.0, 33.0, s) is None
    assert A.pct_change_undefined(30.0, -1.0, s) == "sign_change"
    assert A.pct_change_undefined(-30.0, -10.0, s) == "negative"
    assert A.pct_change_undefined(0.0, 5.0, s) == "zero"
    # non-negative (ratio-scale) data: any clearly positive base qualifies, as before
    pos = samples.sample("declining")
    assert A.pct_change_undefined(0.05 * float(pos.mean()), 1.0, pos) is None
    assert A.pct_change_undefined(float("nan"), 1.0, pos) == "zero"


def test_non_negative_samples_keep_their_metrics():
    # U3 must not touch ratio-scale data: the new rule only fires for a level that crosses zero
    # or for data with negative values (none of the archetypes)
    for nm in samples.list_samples():
        s = samples.sample(nm)
        assert float(s.min()) >= 0
        m = _quiet_analyze(s).metrics
        tr = _quiet_analyze(s).decomposition.trend.to_numpy()
        if m.total_change_pct is not None:
            assert m.total_change_pct == pytest.approx((tr[-1] - tr[0]) / tr[0] * 100, rel=1e-12)
        assert A._pct_floor(s.to_numpy()) == 0.0


def test_emerging_multiple_needs_a_base_away_from_zero():
    # U3 (same rule): a sine wave around +1 with negative values was 'emerging' because its trend
    # was "2.9x the base level" (a base near 0) and "113,089,623,815.2x the average pace"
    idx = pd.date_range("2020-01-01", periods=60, freq="MS")
    for seed in (2, 8):
        rng = np.random.default_rng(seed)
        y = 8 * np.sin(np.arange(60) / 5 + seed) + rng.normal(0, 1, 60) + 1.0
        a = _quiet_analyze(pd.Series(y, index=idx))
        assert a.phase.label != "emerging", (seed, a.phase.reasons)
        assert not any(re.search(r"\d{4,}(\.\d)?배", r.replace(",", "")) for r in a.phase.reasons)


@pytest.mark.parametrize("kind,before,after,ko,en", [
    ("reversal_up", -1.0, 1.0, "하락→상승 반전", "reversal from decline to growth"),
    ("reversal_up", -0.05, 1.0, "횡보→상승 전환", "turn from flat to growth"),
    ("reversal_down", 1.0, -1.0, "상승→하락 반전", "reversal from growth to decline"),
    ("reversal_down", 0.05, -1.0, "횡보→하락 전환", "turn from flat to decline"),
    ("acceleration", 1.0, 2.0, "성장 가속", "growth accelerating"),
    ("acceleration", -1.0, -2.0, "하락 가속", "decline accelerating"),
    ("deceleration", 2.0, 1.0, "성장 둔화", "growth slowing"),
    ("deceleration", -2.0, -1.0, "하락세 둔화", "decline easing"),
    ("level_shift", 0.1, 0.1, "수준 변화", "level shift"),
    ("reversal_up", float("nan"), 1.0, "상승 반전", "upward reversal"),
])
def test_changepoint_label_is_directed(kind, before, after, ko, en):
    # U6: one classifier for the phase reasons, the narrative and the report
    cp = Changepoint(date=pd.Timestamp("2024-03-01"), index=10, slope_before=before, slope_after=after,
                     kind=kind, magnitude=1.0)
    assert A.changepoint_label(cp) == ko and A.changepoint_label(cp, "en") == en
    assert A.CHANGEPOINT_LABEL_EN[ko] == en


def test_phase_reason_names_the_changepoint_with_its_direction():
    # U6: the reason read '구조 변화(상승 전환)' for a reversal out of a decline; it now uses the
    # directed vocabulary of the report and the narrative ('하락→상승 반전' / '횡보→상승 전환')
    from trendai.narrative import translate_reason
    seen = 0
    for nm, freq, n in (("reviving", "M", None), ("reviving", "Y", 25), ("breakout", "M", 30),
                        ("reviving", "M", 30), ("fad", "W", 156)):
        for seed in range(3):
            kw = {"freq": freq, "seed": seed, **({"n": n} if n else {})}
            a = _quiet_analyze(samples.sample(nm, **kw), freq=freq)
            for r in a.phase.reasons:
                mt = re.search(r"구조 변화\(([^)]+)\)", r)
                if not mt:
                    continue
                seen += 1
                assert mt.group(1) not in ("상승 전환", "하락 전환", "가속", "감속"), r
                cp = [c for c in a.changepoints if A._when_ko(c.date, freq) in r][-1]
                assert mt.group(1) == A.changepoint_label(cp)
                assert translate_reason(r) and not HANGUL.search(translate_reason(r))
    assert seen >= 10


# --------------------------------------------------------------------------- round 4 (final)


def test_periods_since_a_level_shift_are_counted_like_the_narrative():
    # F10: '가장 최근 구조 변화(2021-01) 이후 지금까지 59개월의 데이터가 쌓였습니다.' next to
    # '근거: 그 이후 60개월 동안은 …': "그 이후" counts the periods after the break, not the break itself
    from trendai.narrative import rule_based_narrative
    from trendai.pipeline import run
    for freq, n, rule in (("M", 120, "MS"), ("W", 156, "W-SUN"), ("Q", 40, "QS")):
        r = np.random.default_rng(1)
        y = 100 + r.normal(0, 2, n)
        y[n // 2:] += 30
        s = pd.Series(y, index=pd.date_range("2016-01-01", periods=n, freq=rule))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rep = run(s, name="ls")
        a = rep.analysis
        cp = max(a.changepoints, key=lambda c: c.index)
        assert cp.kind == "level_shift"
        since = n - 1 - cp.index
        flat = [x for x in a.phase.reasons if x.startswith("그 이후")]
        assert flat and flat[0].startswith(f"그 이후 {since}{A._UNIT_KO[freq]} 동안"), (freq, flat)
        assert f"이후 지금까지 {since}{A._UNIT_KO[freq]}의 데이터" in rule_based_narrative(
            a, rep.forecast, rep.outlook)


def test_slope_reason_uses_the_per_period_unit():
    # F11: '최근 18개월 동안 추세 기울기는 −0.69/개월입니다.' while every other Korean slope reads '/월'
    from trendai.narrative import period_unit, translate_reason
    a = _quiet_analyze(-samples.sample("growing", n=120, freq="M"), name="neg")
    slope = [x for x in a.phase.reasons if "추세 기울기는" in x]
    assert slope and slope[0].startswith("최근 18개월 동안") and slope[0].endswith(f"{period_unit('M')}입니다.")
    assert "/개월" not in " ".join(a.phase.reasons)
    assert translate_reason(slope[0]).endswith("per month.")
    w = _quiet_analyze(-samples.sample("growing", n=156, freq="W"), name="neg", freq="W")
    assert all("/주입니다" in x for x in w.phase.reasons if "추세 기울기는" in x)


def test_korean_reasons_have_no_english_gloss():
    # F12: '변화가 누적되는 무작위 행보(random walk)에 가까운 움직임이라 …' was the only English word in
    # Korean prose across 370 sweep reports
    from trendai.narrative import translate_reason
    seen = 0
    for seed in range(4):
        r = np.random.default_rng(seed)
        s = pd.Series(100 + np.cumsum(r.normal(0, 1, 400)), index=pd.date_range("2016-01-01", periods=400, freq="D"))
        a = _quiet_analyze(s, name="rw")
        for x in a.phase.reasons:
            assert not re.search(r"[A-Za-z]{3,}", x.replace("Mann-Kendall", "")), x
            if "무작위 행보" in x:
                seen += 1
                assert translate_reason(x) == ("Changes accumulate like a random walk, so trend-like swings "
                                               "may be due to chance.")
    assert seen >= 2


def test_short_series_reason_is_about_the_phase_not_the_trend():
    # F3: '관측치가 5개뿐이라 추세를 판단하기에 데이터가 부족합니다' stood next to a significant
    # Mann-Kendall rise (p = 0.027): it is the lifecycle phase that cannot be judged
    r = np.random.default_rng(105)
    a = _quiet_analyze(from_values(10 + np.arange(5) + r.normal(0, 0.3, 5), start="2021-01-01", freq="Y"))
    assert a.metrics.direction == "up" and a.metrics.n_obs < A.MIN_PHASE_OBS == A._MIN_SHORT_N
    assert a.phase.reasons[0] == "관측치가 5개뿐이라 국면을 판단하기에 데이터가 부족합니다."
    assert "MIN_PHASE_OBS" in A.__all__


def _weekly_multiplicative(off: int, n: int = 208, horizon: int = 26, growth: float = 0.004,
                           amp: float = 0.4) -> tuple[pd.Series, np.ndarray]:
    t = np.arange(n + horizon) + off
    rng = np.random.default_rng(off)
    fac = 1 + amp * np.sin(2 * np.pi * t / 52)
    lvl = 100 * (1 + growth) ** np.arange(n + horizon)
    y = (lvl * fac * np.exp(rng.normal(0, 0.02, n + horizon)))[:n]
    return pd.Series(y, index=pd.date_range("2018-01-07", periods=n, freq="W-SUN")), lvl


def test_weekly_multiplicative_seasonality_is_decomposed_in_log_space():
    # R4-REV-4: weekly, 4 years, +0.4 %/week with a 40 % multiplicative season: the additive trend
    # (26-week window < 52-week period) soaks up part of the growing swing, the interaction slope
    # came out at 0.26 (< 0.35) with t = 11.4, so the series stayed additive although the log fit
    # leaves 2.5x less remainder; the outlook then read +29.1 % for a true +16.7 %
    from trendai.pipeline import run
    for off in (0, 13, 39):
        s, lvl = _weekly_multiplicative(off)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rep = run(s, freq="W")
        d = rep.analysis.decomposition
        assert A._is_multiplicative(d), off
        # the trend is the season-average level (seasonal / trend averages ~0 over a period),
        # not the geometric mean ~4 % below it
        tr, se = d.trend.to_numpy(), d.seasonal.to_numpy()
        assert abs(float(np.mean(se[-52:] / tr[-52:]))) < 0.01
        assert abs(tr[-1] / lvl[207] - 1) < 0.02, (off, tr[-1], lvl[207])
        truth = (lvl[208:].mean() / lvl[156:208].mean() - 1) * 100
        assert abs(rep.outlook.expected_change_pct - truth) < 5, (off, rep.outlook.expected_change_pct, truth)
    # additive seasonality on the same growing level stays additive
    for off in (0, 13):
        t = np.arange(208) + off
        rng = np.random.default_rng(off)
        y = 100 * 1.004 ** np.arange(208) + 40 * np.sin(2 * np.pi * t / 52) + rng.normal(0, 2, 208)
        s = pd.Series(y, index=pd.date_range("2018-01-07", periods=208, freq="W-SUN"))
        assert not A._is_multiplicative(_quiet_analyze(s, freq="W").decomposition)


def test_negative_base_near_zero_is_near_zero_not_negative():
    """Round 5 (G1): the outlook now follows pct_change_undefined, so its reasons must also describe a
    negative level close to 0 (−1.5 among values with an SD of ~10) as near zero, not as a 'negative'
    level whose relative move (−1.5 → −0.1, "93 %") would decide the direction word."""
    rng = np.random.default_rng(1)
    s = pd.Series(rng.normal(0, 10, 40), index=pd.date_range("2016-01-01", periods=40, freq="QS"))
    sd = float(np.std(s))
    assert A.pct_change_undefined(-0.15 * sd, -0.01 * sd, s) == "near_zero"
    assert A.pct_change_undefined(-0.15 * sd, 0.3 * sd, s) == "near_zero"
    assert A.pct_change_undefined(-2.0 * sd, -1.0 * sd, s) == "negative"
    assert A.pct_change_undefined(0.15 * sd, 0.3 * sd, s) == "near_zero"
    assert A.pct_change_undefined(0.15 * sd, -0.3 * sd, s) == "sign_change"


# --------------------------------------------------------------------------- round 5, verify 1


def _sweep_intermittent(kind: str, n: int, freq: str) -> pd.Series:
    """The verifiers' mostly-zero cases (sweep ext.extra_cases, seed 21)."""
    r = np.random.default_rng(21)
    t = np.arange(n)
    size = r.poisson(2, n) + 1
    r.random(n)                                          # intsparse
    rising = size * (r.random(n) < 0.15 + 0.6 * t / (n - 1))
    falling = size * (r.random(n) < 0.75 - 0.6 * t / (n - 1))
    lump = r.choice([1, 1, 2, 60], n)
    lumpy = lump * (r.random(n) < 0.3)
    z = np.zeros(n)
    if kind == "dying":
        z[0], z[n // 3] = 5, 2
    elif kind == "launch":
        z[-6:] = size[-6:]
    vals = {"rising": rising, "falling": falling, "lumpy": lumpy}.get(kind, z)
    pf = {"W": "W-SUN", "M": "MS"}[freq]
    return pd.Series(np.asarray(vals, float), index=pd.date_range(end="2025-12-01", periods=n, freq=pf))


def test_pct_reference_ignores_a_viral_spike():
    """R5-REV-1: one value of 1,000,000 among levels of ~100 made every level 'close to zero'
    (mean|y| = 16,788): the outlook lost its % change and the past section said '0에 가까워'."""
    rng = np.random.default_rng(0)
    y = 100 + 0.8 * np.arange(60) + rng.normal(0, 5, 60)
    y[20] = 1_000_000.0
    s = pd.Series(y, index=pd.date_range("2020-01-01", periods=60, freq="MS"))
    assert A.pct_change_undefined(143.3, 149.0, y) is None
    assert A.pct_change_undefined(0.5, 149.0, y) == "zero"          # a real ~0 base still is one
    res = _quiet_analyze(s)
    assert res.metrics.total_change_pct is not None and res.metrics.recent_change_pct is not None
    # the "mean level" of slope_pct / volatility is not the spike's either ("+0.01%/월", CV 510 %)
    assert 0.3 < res.metrics.slope_pct < 1.0 and res.metrics.volatility < 0.1


@pytest.mark.parametrize("kind,n,freq", [("falling", 156, "W"), ("lumpy", 156, "W"), ("dying", 36, "M"),
                                         ("launch", 36, "M"), ("rising", 36, "M")])
def test_intermittent_phase_reasons_state_no_percent_of_the_level(kind, n, freq):
    """V1-2: '현재 추세는 고점보다 100.0% 낮은 수준', '최근 23주 동안 추세가 −100.0% 변했습니다' for
    mostly-zero demand whose metrics leave every % out: the moves are stated in units."""
    s = _sweep_intermittent(kind, n, freq)
    res = _quiet_analyze(s)
    assert A.level_pct_blocked(s.to_numpy(), season_length(freq)) == "intermittent"
    level_pct = re.compile(r"\d% (?:낮은|변했)|%만큼|최고 수준 대비|고점의 \d+% 수준")
    assert not [r for r in res.phase.reasons if level_pct.search(r)], res.phase.reasons
    from trendai.narrative import translate_reason
    assert all(translate_reason(r) for r in res.phase.reasons), res.phase.reasons


def test_all_zero_phase_reasons_name_the_zeros():
    """V1-2 / R5-REV-4: '추세의 변동 폭(최고-최저)이 최고 수준 대비 0.0%에 그칩니다' of a history of zeros."""
    for n in (36, 5):
        res = _quiet_analyze(pd.Series(np.zeros(n), index=pd.date_range("2020-01-01", periods=n, freq="MS")))
        assert "관측값이 모두 0입니다." in res.phase.reasons
        assert not [r for r in res.phase.reasons if "%" in r], res.phase.reasons


def test_intermittent_launch_is_not_declining():
    """V1-5: 30 zero months, then demand: 'declining' (0.78) next to '추세 기울기는 +0.34/월'."""
    z = np.zeros(36)
    z[-6:] = [3, 5, 4, 2, 3, 2]
    res = _quiet_analyze(pd.Series(z, index=pd.date_range("2023-01-01", periods=36, freq="MS")))
    assert res.phase.label != "declining"
    assert res.phase.label == "growing"           # the recent move and the Mann-Kendall test agree


@pytest.mark.parametrize("kind,n,freq", [("dying", 36, "M"), ("lumpy", 156, "W")])
def test_intermittent_seasonality_must_recur(kind, n, freq):
    """V1-8: two orders, both in January, read as 'very strong seasonality (1.00)'; random 60-unit lumps
    over three weekly years as 0.99."""
    res = _quiet_analyze(_sweep_intermittent(kind, n, freq))
    assert res.metrics.seasonal_strength < 0.2


def test_recurring_intermittent_season_is_kept():
    """V1-8 (the other side): demand only around Nov-Dec every year stays seasonal."""
    rng = np.random.default_rng(4)
    t = np.arange(60)
    y = (rng.poisson(3, 60) + 1) * (rng.random(60) < np.where(t % 12 >= 10, 0.95, 0.05))
    res = _quiet_analyze(pd.Series(y.astype(float), index=pd.date_range("2020-01-01", periods=60, freq="MS")))
    assert A.level_pct_blocked(y, 12) == "intermittent"
    assert res.metrics.seasonal_strength >= 0.5


def test_level_rate_undefined_rule():
    """V1-3: the rule behind '평균 수준 대비 …%/월', momentum %p and the CV on every surface."""
    t = np.arange(60)
    assert A.level_rate_undefined(100 + t, 12) is None
    assert A.level_rate_undefined(-20 - t, 12) == "negative"
    assert A.level_rate_undefined(50 - 80 * t / 59, 12) == "near_zero"   # a P&L crossing zero
    assert A.level_rate_undefined(np.zeros(36), 12) == "zero"
    assert A.level_rate_undefined(np.where(t % 3 == 0, 4.0, 0.0), 12) == "intermittent"


# --------------------------------------------------------------------------- round 5, verify 2


def test_single_final_order_is_no_emerging_phase():
    """V2-6: lumpy demand (0-2 a month, orders of 60 in months 4 and 36) read '태동기(신흥) (신뢰도 95%)',
    '현재 추세 수준이 과거 기반 수준의 91.9배' -- the last order pulled the trend's end.  The phase
    follows the demand rate without its single largest order (first-year 5.50, last-year 5.33,
    Mann-Kendall p = 0.28)."""
    v = [2, 0, 1, 60, 1, 0, 1, 0, 1, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         1, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 60]
    s = pd.Series(np.array(v, float), index=pd.date_range("2023-01-01", periods=36, freq="MS"))
    res = _quiet_analyze(s)
    assert res.phase.label not in ("emerging", "growing") and res.phase.confidence <= 0.6, res.phase
    assert "최근의 높은 추세는 2025-12의 큰 수요 한 건(60.0) 때문이며, 이를 빼면 수요가 늘었다고 보기 어렵습니다." \
        in res.phase.reasons
    from trendai.narrative import translate_reason
    assert all(translate_reason(r) for r in res.phase.reasons), res.phase.reasons
    # the mirror: one early order of 60 is no 'declining'
    m = pd.Series(np.array(v[::-1], float), index=s.index)
    assert _quiet_analyze(m).phase.label != "declining"


@pytest.mark.parametrize("freq,n,seed", [("M", 48, 13), ("W", 156, 16)])
def test_intermittent_declining_never_sits_next_to_a_rise(freq, n, seed):
    """R5-REV2-1: 'declining' with the reason '최근 7개월 동안 추세가 0.15에서 0.34까지 높아졌습니다': the guard
    measured the rise against 5 % of the trend's whole range (8.8), not against the level.  One rule
    (_rate_move) decides both the reason's verb and whether the phase contradicts it."""
    rng = np.random.default_rng(seed)
    y = (rng.poisson(np.linspace(0.8, 0.2, n)) * rng.integers(1, 10, n)).astype(float)
    s = pd.Series(y, index=pd.date_range("2020-01-05", periods=n, freq={"M": "MS", "W": "W-SUN"}[freq]))
    res = _quiet_analyze(s)
    # a rise in units ('0.15에서 0.34까지 높아졌습니다') or, as HEAD worded it, in % ('+117.2% 변했습니다')
    rose = any("높아졌습니다" in r or re.search(r"추세가 \+[\d.,]+% 변했습니다", r) for r in res.phase.reasons)
    fell = any("낮아졌습니다" in r or re.search(r"추세가 −[\d.,]+% 변했습니다", r) for r in res.phase.reasons)
    assert not (res.phase.label == "declining" and rose), (res.phase.label, res.phase.reasons)
    assert not (res.phase.label in ("growing", "emerging", "reviving") and fell), res.phase.reasons


def test_a_small_move_of_the_demand_rate_is_no_move():
    f = {"t_rec0": 0.150, "t_last": 0.152}
    assert A._rate_move(f) == 0 and not A._against_recent_move("declining", f)
    assert A._rate_move({"t_rec0": 0.15, "t_last": 0.34}) == 1
    assert A._against_recent_move("declining", {"t_rec0": 0.15, "t_last": 0.34})
    assert A._against_recent_move("growing", {"t_rec0": 0.34, "t_last": 0.15})
    assert A.phase_window(48) == 7 and A.phase_window(156) == 23
