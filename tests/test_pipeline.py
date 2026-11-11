"""End-to-end tests: raw series -> TrendReport, ranking and JSON export."""

from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
import pytest

from trendai import compare, pipeline
from trendai.data import future_index
from trendai.samples import EXPECTED_PHASE, list_samples, sample
from trendai.types import PHASES, TrendReport


@pytest.mark.parametrize("name", list_samples())
def test_run_every_sample(name):
    rep = pipeline.run(sample(name), horizon=12)
    assert isinstance(rep, TrendReport)
    assert rep.name == name
    assert rep.analysis.phase.label in PHASES
    assert rep.forecast.horizon == 12
    assert len(rep.forecast.mean) == 12
    assert np.isfinite(rep.forecast.mean.to_numpy()).all()
    exp_idx = future_index(rep.analysis.series.index[-1], "M", 12)
    assert rep.forecast.mean.index.equals(exp_idx)
    for lv in rep.forecast.levels:
        assert (rep.forecast.lower[lv] <= rep.forecast.mean + 1e-9).all()
        assert (rep.forecast.upper[lv] >= rep.forecast.mean - 1e-9).all()
    assert rep.narrative_source == "rule"
    assert len(rep.narrative) > 200
    assert rep.outlook.direction in ("up", "down", "flat")
    assert 0.0 <= rep.outlook.prob_increase <= 1.0


def test_phase_accuracy_end_to_end():
    hits = sum(pipeline.run(sample(nm), horizon=6).analysis.phase.label in exp
               for nm, exp in EXPECTED_PHASE.items())
    assert hits >= len(EXPECTED_PHASE) - 1


def test_directional_outlook():
    up = pipeline.run(sample("growing"), horizon=12).outlook
    down = pipeline.run(sample("declining"), horizon=12).outlook
    assert up.direction == "up" and up.prob_increase > 0.6
    assert down.direction == "down" and down.prob_increase < 0.4


def test_raw_irregular_input():
    s = sample("growing")
    raw = s.sample(frac=1.0, random_state=1)  # shuffled
    raw.iloc[5] = np.nan
    raw = raw.drop(raw.index[10:13])          # gap
    rep = pipeline.run(raw, name="raw")
    assert rep.analysis.freq == "M"
    assert rep.analysis.series.index.is_monotonic_increasing
    assert not rep.analysis.series.isna().any()


@pytest.mark.parametrize("freq,n", [("D", 400), ("W", 156), ("Q", 40), ("Y", 25)])
def test_frequencies(freq, n):
    rep = pipeline.run(sample("growing", freq=freq, n=n))
    assert rep.analysis.freq == freq
    assert rep.forecast.horizon >= 1


@pytest.mark.parametrize("n", [3, 5, 8, 12])
def test_short_series(n):
    s = sample("growing", n=max(n, 8)).iloc[:n]
    rep = pipeline.run(s)
    assert len(rep.forecast.mean) == rep.forecast.horizon
    assert np.isfinite(rep.forecast.mean.to_numpy()).all()


def test_constant_and_negative_series():
    idx = pd.date_range("2020-01-01", periods=36, freq="MS")
    const = pipeline.run(pd.Series(5.0, index=idx), name="const")
    assert np.allclose(const.forecast.mean.to_numpy(), 5.0, atol=1e-6)
    neg = pipeline.run(pd.Series(-50 + np.arange(36.0), index=idx), name="neg")
    assert np.isfinite(neg.forecast.mean.to_numpy()).all()
    assert neg.forecast.nonnegative is False


def test_report_to_json_roundtrip():
    rep = pipeline.run(sample("breakout"), horizon=6)
    d = json.loads(pipeline.report_to_json(rep))
    assert d["name"] == "breakout"
    assert len(d["history"]) == 120 and len(d["forecast"]["table"]) == 6
    assert {"mean", "lo80", "hi80", "lo95", "hi95"} <= set(d["forecast"]["table"][0])
    assert d["phase"]["label"] in PHASES


def test_run_many_and_rank():
    reps = pipeline.run_many({nm: sample(nm) for nm in ["emerging", "growing", "fad", "declining"]},
                             horizon=12)
    assert set(reps) == {"emerging", "growing", "fad", "declining"}
    df = compare.rank_trends(reps)
    assert list(df.columns) == compare.RANK_COLUMNS
    assert df["score"].is_monotonic_decreasing
    assert df["score"].between(0, 100).all()
    top2 = set(df["name"].iloc[:2])
    assert top2 & {"emerging", "growing"}
    assert df["name"].iloc[-1] in ("fad", "declining")


def test_run_many_skips_failures():
    bad = pd.Series([1.0], index=pd.to_datetime(["2020-01-01"]))
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        reps = pipeline.run_many({"ok": sample("growing"), "bad": bad})
    assert list(reps) == ["ok"]


def test_llm_path_with_fake_client():
    from types import SimpleNamespace

    calls = {}

    def create(**kw):
        calls.update(kw)
        return SimpleNamespace(stop_reason="end_turn",
                               content=[SimpleNamespace(type="text", text="## 과거\nLLM 해석")])

    client = SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(create=create)))
    rep = pipeline.run(sample("growing"), use_llm=True, client=client)
    assert rep.narrative_source == "llm"
    assert "LLM 해석" in rep.narrative
    assert calls["model"] == "claude-opus-5-5"


# --------------------------------------------------------------------------- integration fixes


def test_json_export_contains_skill():
    rep = pipeline.run(sample("growing"), horizon=6)
    d = json.loads(pipeline.report_to_json(rep))
    assert "skill" in d["forecast"]
    rep.forecast.skill = 0.42
    assert json.loads(pipeline.report_to_json(rep))["forecast"]["skill"] == 0.42
    rep.forecast.skill = None
    assert json.loads(pipeline.report_to_json(rep))["forecast"]["skill"] is None


@pytest.mark.parametrize("bad", [0, -3, 2.5])
def test_run_rejects_bad_horizon(bad):
    from trendai.data import DataError

    with pytest.raises(DataError, match="horizon"):
        pipeline.run(sample("growing"), horizon=bad)


def test_run_many_reports_errors_and_korean_warning():
    from trendai.data import DataError

    bad = pd.Series([1.0], index=pd.to_datetime(["2020-01-01"]))
    errors: dict = {}
    with pytest.warns(pipeline.SeriesFailedWarning, match="분석 실패") as rec:
        reps = pipeline.run_many({"ok": sample("growing"), "bad": bad}, errors=errors, horizon=3)
    assert list(reps) == ["ok"] and list(errors) == ["bad"]
    assert isinstance(errors["bad"], DataError)
    assert "DataError" not in str(rec[0].message)


@pytest.mark.parametrize("index,freq", [
    (list(range(1990, 2020)), "Y"),
    (pd.period_range("2020-01", periods=36, freq="M"), "M"),
])
def test_run_accepts_year_and_period_indexes(index, freq):
    s = pd.Series(np.linspace(10, 40, len(index)) + np.sin(np.arange(len(index))), index=index)
    rep = pipeline.run(s, horizon=3)
    assert rep.analysis.freq == freq and len(rep.forecast.mean) == 3


def test_weekly_monday_series_forecasts_on_mondays():
    idx = pd.date_range("2022-01-03", periods=104, freq="W-MON")
    rep = pipeline.run(pd.Series(sample("growing", n=104, freq="W").values, index=idx), horizon=2)
    assert rep.analysis.series.index[-1] == idx[-1]
    assert list(rep.forecast.mean.index) == [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-08")]


def test_annualise_pct():
    assert compare.annualise_pct(26.824, 12, 12.0) == pytest.approx(26.824)
    assert compare.annualise_pct(10.0, 24, 12.0) == pytest.approx((1.1 ** 0.5 - 1) * 100)
    assert compare.annualise_pct(2.0, 1, 12.0) == pytest.approx((1.02 ** 12 - 1) * 100)
    assert compare.annualise_pct(-150.0, 3, 12.0) == -100.0
    assert compare.annualise_pct(None, 3, 12.0) is None
    huge = compare.annualise_pct(500.0, 1, 365.25)
    assert np.isfinite(huge) and huge <= 10000.0 + 1e-6


def test_rising_score_is_frequency_neutral():
    """Same +2 %/month process sampled daily and monthly: before the fix 63.3 vs 77.7."""
    rng = np.random.default_rng(0)
    days = pd.date_range("2022-01-01", "2023-12-31", freq="D")
    t = (days - days[0]).days / 30.4375
    daily = pd.Series(100 * 1.02 ** t * (1 + 0.005 * rng.standard_normal(len(days))), index=days)
    monthly = daily.resample("MS").mean()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reps = pipeline.run_many({"daily": daily, "monthly": monthly})
    df = compare.rank_trends(reps).set_index("name")
    assert abs(df.loc["daily", "score"] - df.loc["monthly", "score"]) < 3.0
    assert set(df["freq"]) == {"D", "M"}
    for col in ("recent_change_ann_pct", "expected_change_ann_pct"):
        assert 15.0 < df.loc["daily", col] < 35.0 and 15.0 < df.loc["monthly", col] < 35.0
    # monthly data with the default horizon: annualised values equal the report's own numbers
    m = reps["monthly"]
    assert df.loc["monthly", "expected_change_ann_pct"] == pytest.approx(m.outlook.expected_change_pct)
    assert df.loc["monthly", "recent_change_ann_pct"] == pytest.approx(m.analysis.metrics.recent_change_pct,
                                                                       rel=1e-6, abs=1e-9)


def test_rank_trends_values_are_finite():
    reps = pipeline.run_many({nm: sample(nm) for nm in ["emerging", "fad", "seasonal"]}, horizon=6)
    df = compare.rank_trends(reps)
    for col in ("score", "momentum_per_month", "recent_change_ann_pct", "expected_change_ann_pct"):
        assert np.isfinite(df[col].astype(float)).all(), col


def test_run_on_loaded_formatted_csv(tmp_path):
    from trendai.data import load_table

    idx = pd.date_range("2016-01-01", periods=120, freq="MS")
    v = np.linspace(300, 1500, 120)
    (tmp_path / "t.csv").write_text("date,sales\n" + "".join(f'{d:%Y-%m-%d},"{x:,.0f}"\n'
                                                             for d, x in zip(idx, v)))
    rep = pipeline.run(load_table(tmp_path / "t.csv")["sales"], horizon=3)
    assert len(rep.analysis.series) == 120
    assert rep.analysis.series.index[-1] == pd.Timestamp("2025-12-01")
    assert rep.analysis.metrics.last_value == pytest.approx(1500)
