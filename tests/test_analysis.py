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
