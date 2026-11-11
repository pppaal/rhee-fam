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


# --------------------------------------------------------------------------- round 3


def test_json_export_keeps_relative_precision():
    """F4: round(f, 6) exported a ~1.2e-5 token price as 1.2e-05 (bands merged) and a ~1e-7
    series as all 0.0."""
    rng = np.random.default_rng(0)
    price = pd.Series(1.234e-5 * np.exp(np.cumsum(rng.normal(0, 0.03, 60))),
                      index=pd.date_range("2024-01-01", periods=60, freq="D"))
    d = json.loads(pipeline.report_to_json(pipeline.run(price, horizon=6)))
    assert [h["value"] for h in d["history"][:3]] == pytest.approx(price.iloc[:3].tolist(), rel=1e-11)
    row = d["forecast"]["table"][0]
    assert row["lo80"] < row["mean"] < row["hi80"]
    tiny = pd.Series((np.arange(36.0) + 100) * 1e-9, index=pd.date_range("2021-01-01", periods=36, freq="MS"))
    d = json.loads(pipeline.report_to_json(pipeline.run(tiny, horizon=3)))
    assert [h["value"] for h in d["history"][:3]] == pytest.approx([1e-7, 1.01e-7, 1.02e-7], rel=1e-9)
    assert d["metrics"]["slope"] == pytest.approx(1e-9, rel=1e-6)
    # ordinary values stay short: no float noise in the export
    assert pipeline._clean(0.1 + 0.2) == 0.3 and pipeline._clean(np.float64(61.2)) == 61.2


def test_run_rejects_horizons_beyond_the_date_range():
    """F14: -H 300000 on yearly data raised a raw OutOfBoundsDatetime; 290,000 ran > 2 minutes."""
    from trendai.data import DataError, DataWarning

    s = pd.Series(np.arange(20.0) * 2 + 50, index=pd.date_range("2001-01-01", periods=20, freq="YS"))
    with pytest.raises(DataError, match="예측 기간"):
        pipeline.run(s, horizon=300_000)
    with pytest.raises(DataError, match="최대"):
        pipeline.run(s, horizon=pipeline.MAX_HORIZON + 1)
    ns = pd.Series(s.to_numpy(), index=s.index.as_unit("ns"))     # nanoseconds end in 2262
    with pytest.raises(DataError, match="날짜로 표현할 수 있는 범위"):
        pipeline.run(ns, horizon=400)
    with pytest.warns(DataWarning, match="5배"):
        rep = pipeline.run(s, horizon=150)
    assert len(rep.forecast.mean) == 150
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        pipeline.run(s, horizon=10)                                 # ordinary horizons: silent


def test_rising_score_ignores_noise_momentum():
    """UX-4: `sample all` ranked volatile (▼ −1.4 %, P 48 %, score 54.6) above seasonal (▲ +0.4 %)
    and peaking (▲ +0.9 %): its lead came from momentum the narrative calls 잡음 수준."""
    reps = {nm: pipeline.run(sample(nm), name=nm) for nm in ("volatile", "seasonal", "peaking")}
    df = compare.rank_trends(reps).set_index("name")
    vol = reps["volatile"]
    assert vol.analysis.phase.label in compare.NOISE_PHASES
    assert compare.momentum_per_month(vol) > 0.5                # the noisy upswing is still reported
    assert df.loc["volatile", "score"] < 50.0
    for nm in ("seasonal", "peaking"):
        o = reps[nm].outlook
        if o.expected_change_pct > 0 and o.prob_increase > 0.5:
            assert df.loc[nm, "score"] > df.loc["volatile", "score"], df
    # in a noise phase the score is the outlook alone
    o = vol.outlook
    expected = 50 + 50 * (0.35 * np.tanh(compare.expected_change_annualised(vol) / 50.0)
                          + 0.25 * (2 * o.prob_increase - 1))
    assert compare.rising_score(vol) == pytest.approx(expected)


@pytest.mark.parametrize("freq,n,last", [("M", 120, "2025-12-01"), ("M", 36, "2025-12-01"),
                                         ("Y", 12, "2025-01-01"), ("Y", 20, "2025-01-01"),
                                         ("Q", 40, "2025-10-01"), ("D", 730, "2025-12-31"),
                                         ("W", 156, "2025-12-28")])
def test_samples_end_at_a_fixed_reference_date(freq, n, last):
    """UX-12: samples started at 2016-01-01 whatever the frequency, so `sample emerging --freq Y
    --n 12` ran 2016 ~ 2027 ('가장 최근 값은 72.0(2027년)') and `--n 20` to 2035."""
    from trendai.samples import breakout_date

    s = sample("emerging", n=n, freq=freq)
    assert len(s) == n and s.index[-1] == pd.Timestamp(last)
    assert breakout_date(n, freq) == s.index[int(round(0.6 * (n - 1)))]
    start = "2016-01-03" if freq == "W" else "2016-01-01"            # a Sunday for W-SUN
    assert sample("emerging", n=n, freq=freq, start=start).index[0] == pd.Timestamp(start)
    # the default monthly sample is unchanged: 2016-01 .. 2025-12
    if (freq, n) == ("M", 120):
        assert s.index[0] == pd.Timestamp("2016-01-01")


def test_run_passes_the_random_walk_finding_to_the_forecast():
    """F6 end to end: the driftless random walk of seed 2032 read 'up', p 0.95, +16.1 %."""
    from trendai.forecast import _TREND_EXTRAPOLATORS

    rng = np.random.default_rng(2032)
    s = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.018, 730))),
                  index=pd.date_range("2023-01-01", periods=730, freq="D"))
    rep = pipeline.run(s)
    assert not set(_TREND_EXTRAPOLATORS) & {ms.name for ms in rep.forecast.model_scores}
    assert rep.outlook.direction == "flat" and 0.3 < rep.outlook.prob_increase < 0.7
    # a trending series keeps its trend models
    grow = pipeline.run(sample("growing"))
    assert {"linear_trend", "holt"} & {ms.name for ms in grow.forecast.model_scores}


def test_default_horizon_gets_the_long_horizon_warning():
    """F9: run(s) on a 3-point daily series made a 30-step forecast (10x the history) silently,
    while run(s, horizon=30) made the same forecast with a DataWarning."""
    from trendai.data import DataWarning, from_values

    s = from_values([10.0, 11.0, 12.0], freq="D")
    with pytest.warns(DataWarning, match="5배"):
        rep = pipeline.run(s)
    assert len(rep.forecast.mean) == 30
    with pytest.warns(DataWarning, match="5배"):
        explicit = pipeline.run(s, horizon=30)
    assert np.allclose(rep.forecast.mean.to_numpy(), explicit.forecast.mean.to_numpy())
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        pipeline.run(sample("growing"))                     # an ordinary default stays silent


@pytest.mark.parametrize("bad", [float("inf"), float("-inf"), float("nan")])
def test_non_finite_horizon_is_a_data_error(bad):
    """R4-REV-7: horizon=inf raised OverflowError instead of the documented DataError."""
    from trendai.data import DataError

    with pytest.raises(DataError, match="horizon"):
        pipeline.run(sample("growing"), horizon=bad)


def test_json_outlook_carries_the_direction_word_and_the_compared_levels():
    """F15: the export said direction 'flat' next to −22.8 % where every human surface says
    '방향 불확실'; it now carries the label, the None reason and the two compared levels."""
    from trendai.forecast import outlook_levels
    from trendai.narrative import build_llm_payload

    rep = pipeline.run(sample("fad", n=8, freq="Y"))
    o = json.loads(pipeline.report_to_json(rep))["outlook"]
    assert o["direction"] == "flat" and o["expected_change_pct"] < -20
    assert o["direction_label"] == {"ko": "방향 불확실", "en": "direction uncertain"}
    assert o["expected_change_undefined"] is None
    lv = outlook_levels(rep.analysis, rep.forecast)
    assert o["compared_levels"]["current"] == pytest.approx(lv.current, rel=1e-12)
    assert o["compared_levels"]["future"] == pytest.approx(lv.future, rel=1e-12)
    # the same fields as the LLM payload
    pay = build_llm_payload(rep.analysis, rep.forecast, rep.outlook)["outlook"]
    assert o["direction_label"] == pay["direction_label"]
    # a level expected to cross zero: no %, its reason and both levels
    idx = pd.date_range("2016-01-01", periods=60, freq="MS")
    pnl = pd.Series(np.linspace(30.0, -2.0, 60) + np.random.default_rng(3).normal(0, 0.3, 60), index=idx)
    rep2 = pipeline.run(pnl, horizon=24)
    o2 = pipeline.report_to_dict(rep2)["outlook"]
    lv2 = outlook_levels(rep2.analysis, rep2.forecast)
    assert o2["expected_change_undefined"] == lv2.undefined == "sign_change"
    assert o2["expected_change_pct"] is None
    assert o2["compared_levels"]["current"] > 0 > o2["compared_levels"]["future"]
    assert "보합" not in o2["direction_label"]["ko"]


def test_json_keeps_fifteen_significant_digits():
    """R4-REV-5: 12-digit rounding exported 1,234,567,890,123 (won) as 1,234,567,890,120."""
    vals = 1_234_567_890_123 + np.arange(24) * 1_000_003.0
    rep = pipeline.run(pd.Series(vals, index=pd.date_range("2020-01-01", periods=24, freq="MS")), horizon=3)
    d = json.loads(pipeline.report_to_json(rep))
    assert [h["value"] for h in d["history"]] == vals.tolist()
    assert pipeline._clean(123456789012345.0) == 123456789012345.0
    assert pipeline._clean(0.1 + 0.2) == 0.3


def test_json_export_of_forecast_dates_after_year_9999():
    """R4-REV-6: run() accepts a yearly horizon that ends after 9999, but the JSON export
    crashed in strftime (NotImplementedError)."""
    s = pd.Series(np.arange(20.0) * 2 + 50, index=pd.date_range("2001-01-01", periods=20, freq="YS"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rep = pipeline.run(s, horizon=8000)
    assert rep.forecast.mean.index[-1].year == 10020
    d = json.loads(pipeline.report_to_json(rep))
    assert d["forecast"]["table"][-1]["date"] == "10020-01-01"
    assert d["forecast"]["table"][0]["date"] == "2021-01-01"
    assert d["history"][0]["date"] == "2001-01-01"


def test_rising_score_follows_the_analysis_percentage_rules():
    """R4-REV-2: the score's recent change ignored the analysis' rules: a P&L falling through
    zero scored a '−100 %' recent change, a trend rising from a near-zero base of data with
    negative values +944 to +1,467 %/yr (score ~99), intermittent demand +1,445 %/yr -- while
    the table's recent_change_pct showed '—' for all three."""
    from trendai.analysis import pct_change_undefined

    idx = pd.date_range("2020-01-01", periods=48, freq="MS")
    rng = np.random.default_rng(5)
    pnl = np.r_[np.linspace(20, 12, 36), np.linspace(12, -6, 12)] + rng.normal(0, 0.5, 48)
    near = np.r_[np.linspace(-6, 0.5, 36), np.linspace(0.5, 8, 12)] + np.random.default_rng(0).normal(0, 0.2, 48)
    rng = np.random.default_rng(13)
    interm = ((rng.random(60) < np.linspace(0.2, 0.6, 60)) * rng.integers(1, 20, 60)).astype(float)
    cases = {"pnl": pd.Series(pnl, index=idx), "near": pd.Series(near, index=idx),
             "interm": pd.Series(interm, index=pd.date_range("2020-01-01", periods=60, freq="MS"))}
    reps = {nm: pipeline.run(s, name=nm, freq="M") for nm, s in cases.items()}
    trend = np.asarray(reps["pnl"].analysis.decomposition.trend, dtype=float)
    assert pct_change_undefined(trend[-13], trend[-1], reps["pnl"].analysis.series) == "sign_change"
    trend = np.asarray(reps["near"].analysis.decomposition.trend, dtype=float)
    assert pct_change_undefined(trend[-13], trend[-1], reps["near"].analysis.series) == "near_zero"
    df = compare.rank_trends(reps).set_index("name")
    for nm, r in reps.items():
        assert r.analysis.metrics.recent_change_pct is None, nm
        assert compare.recent_change_annualised(r) is None, nm
        assert df.loc[nm, "recent_change_ann_pct"] is None or np.isnan(df.loc[nm, "recent_change_ann_pct"])
        # the recent change no longer drives the score: what is left is outlook + momentum
        o = r.outlook
        z = (0.35 * np.tanh((compare.expected_change_annualised(r) or 0.0) / 50.0)
             + 0.25 * (2 * o.prob_increase - 1))
        if r.analysis.phase.label not in compare.NOISE_PHASES:
            z += 0.20 * np.tanh(compare.momentum_per_month(r) / 2.0)
        assert compare.rising_score(r) == pytest.approx(50 + 50 * z), nm
    assert df.loc["near", "score"] < 95.0          # was 98.7-99.7 (tanh saturated by +1,000 %/yr)


# --------------------------------------------------------------------------- round 5

_HANGUL = __import__("re").compile(r"[가-힣]")


def test_negative_current_level_in_json_and_ranking():
    """G1: the JSON export and the rising score follow the one percentage rule: a negative current
    level gives expected_change_pct None ('negative'), and the expected-change term drops out."""
    rng = np.random.default_rng(1)
    t = np.arange(120)
    s = pd.Series(50 - 80 * t / 119 + rng.normal(0, 4, 120),
                  index=pd.date_range("2016-01-01", periods=120, freq="MS"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rep = pipeline.run(s, name="P&L")
    o = pipeline.report_to_dict(rep)["outlook"]
    assert o["expected_change_pct"] is None and o["expected_change_undefined"] == "negative"
    assert o["direction_label"] == {"ko": "하락", "en": "downward"}
    assert o["compared_levels"]["current"] < 0
    assert compare.expected_change_annualised(rep) is None
    df = compare.rank_trends({"P&L": rep})
    assert df["expected_change_pct"].isna().all()


def test_series_failed_and_horizon_warnings_carry_catalog_codes():
    """G5: the pipeline's warnings / errors are catalog messages: Korean str(), English .text('en')."""
    from trendai.data import DataError, DataWarning

    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = pipeline.run_many({"a": pd.Series([1.0], index=idx[:1]), "b": pd.Series([1.0, 2.0, 3.0], index=idx)})
    assert list(out) == ["b"]
    failed = next(x.message for x in w if isinstance(x.message, pipeline.SeriesFailedWarning))
    assert failed.code == "series_failed" and failed.params["series"] == "a"
    assert str(failed) == "'a' 분석 실패 — 관측치가 2개 이상 필요합니다 (유효한 값 1개)."
    assert failed.text("en") == "'a' could not be analysed — at least 2 observations are needed (1 valid value(s))."
    long = next(x.message for x in w if isinstance(x.message, DataWarning))
    assert long.code == "horizon_long" and long.params == {"horizon": 30, "n": 3, "ratio": 5}
    assert str(long).startswith("예측 기간(30기간)이 데이터 길이(3개)의 5배를 넘습니다.")
    assert not _HANGUL.search(long.text("en"))
    for bad, code in ((0, "horizon_not_positive"), (2.5, "horizon_not_positive"), ("x", "horizon_not_int"),
                      (20_000, "horizon_too_long"), (float("inf"), "horizon_not_int")):
        with pytest.raises(DataError) as ei:
            pipeline.run(pd.Series([1.0, 2.0, 3.0], index=idx), horizon=bad)
        assert ei.value.code == code and not _HANGUL.search(ei.value.text("en")), bad


def test_json_prob_increase_follows_the_shared_rule():
    """V2-3: report_to_json gave prob_increase 0.5 with no reason for an all-zero history while the
    payload, the ranking, the narrative and the KPI all said there is none."""
    from trendai import narrative

    idx = pd.date_range("2023-01-01", periods=36, freq="MS")
    dead = np.zeros(36)
    dead[3] = 4.0
    for values, why in ((np.zeros(36), "all_zero"), (dead, "zero_current")):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rep = pipeline.run(pd.Series(values, index=idx, name="returns"), freq="M")
        out = json.loads(pipeline.report_to_json(rep))["outlook"]
        assert narrative.prob_undefined(rep.analysis) == why
        assert out["prob_increase"] is None and out["prob_increase_undefined"] == why
        assert narrative.build_llm_payload(rep.analysis, rep.forecast, rep.outlook)["outlook"]["prob_increase"] is None
        assert compare.rank_trends([rep])["prob_increase"].tolist() == [None]
    # a series with a probability keeps it, with no reason
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rep = pipeline.run(pd.Series(np.arange(36.0) + 10, index=idx), freq="M")
    out = json.loads(pipeline.report_to_json(rep))["outlook"]
    assert out["prob_increase"] == pytest.approx(rep.outlook.prob_increase) and out["prob_increase_undefined"] is None
