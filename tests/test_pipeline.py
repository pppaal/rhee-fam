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
