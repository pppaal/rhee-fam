"""Tests for trendai.narrative (rule-based text, LLM payload, mocked Claude calls)."""

from __future__ import annotations

import copy
import dataclasses
import json
import re
import sys
import warnings
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from trendai import narrative as nv
from trendai.data import future_index, season_length
from trendai.narrative import (
    DEFAULT_MODEL, SYSTEM_PROMPT, NarrativeUnavailable, ask, build_llm_payload, format_date,
    format_number, format_pct, generate_narrative, llm_narrative, period_unit,
    rule_based_narrative,
)
from trendai.samples import sample
from trendai.types import (
    AnalysisResult, Anomaly, Changepoint, Decomposition, ForecastResult, LifecyclePhase,
    ModelScore, Outlook, TrendMetrics, TrendReport,
)

HANGUL = re.compile(r"[가-힣]")
KO_HEADINGS = ["## 과거: 어떻게 여기까지 왔나", "## 현재: 지금 어디에 있나", "## 미래 전망", "## 주의할 점"]


def _parts(report: TrendReport):
    return report.analysis, report.forecast, report.outlook


def _replace_metrics(report: TrendReport, **kw) -> TrendReport:
    metrics = dataclasses.replace(report.analysis.metrics, **kw)
    analysis = dataclasses.replace(report.analysis, metrics=metrics)
    return dataclasses.replace(report, analysis=analysis)


def _make_report(freq: str = "D", n: int = 400, kind: str = "seasonal", horizon: int = 30,
                 changepoints: bool = True, phase_conf: float = 0.4, outlook_conf: str = "low",
                 forecast_peak: bool = True) -> TrendReport:
    """A hand-built report at any frequency (no analysis/forecast code involved)."""
    s = sample(kind, n=n, freq=freq)
    s.name = "test series"
    sl = season_length(freq)
    trend = s.rolling(max(3, sl), center=True, min_periods=1).mean()
    seasonal = (s - trend).groupby(np.arange(n) % max(sl, 1)).transform("mean") if sl > 1 else s * 0
    resid = s - trend - seasonal
    dec = Decomposition(trend=trend, seasonal=seasonal, resid=resid, period=sl)
    cps = []
    if changepoints:
        k = n // 2
        cps = [Changepoint(date=s.index[k], index=k, slope_before=0.5, slope_after=-0.3,
                           kind="reversal_down", magnitude=0.8),
               Changepoint(date=s.index[n - 3], index=n - 3, slope_before=-0.3, slope_after=-0.3,
                           kind="level_shift", magnitude=0.5)]
    peak_i = int(np.argmax(s.to_numpy()))
    metrics = TrendMetrics(
        n_obs=n, freq=freq, start=s.index[0], end=s.index[-1], last_value=float(s.iloc[-1]),
        mean=float(s.mean()), total_change_pct=-12.5, cagr_pct=-3.2, recent_change_pct=-4.0,
        slope=-0.01, slope_pct=-0.02, momentum=-0.5, acceleration=-0.1, volatility=0.35,
        trend_strength=0.2, seasonal_strength=0.85, mk_tau=-0.1, mk_pvalue=0.034, direction="flat",
        peak_date=s.index[peak_i], peak_value=float(s.iloc[peak_i]),
        drawdown_pct=float((s.iloc[-1] - s.iloc[peak_i]) / s.iloc[peak_i] * 100),
    )
    phase = LifecyclePhase(label="stable", confidence=phase_conf, reasons=["추세 없음"])
    analysis = AnalysisResult(name=s.name, series=s, freq=freq, season_length=sl, decomposition=dec,
                              metrics=metrics, changepoints=cps, anomalies=[], phase=phase)
    idx = future_index(s.index[-1], freq, horizon)
    mean = pd.Series(float(s.iloc[-1]) + 3 * np.sin(np.arange(horizon) / 3), index=idx)
    sd = 6.0 * np.sqrt(np.arange(1, horizon + 1))  # wide enough that the 95% band hits 0
    lower = {80: (mean - 1.28 * sd).clip(lower=0), 95: (mean - 1.96 * sd).clip(lower=0)}
    upper = {80: mean + 1.28 * sd, 95: mean + 1.96 * sd}
    scores = [ModelScore("seasonal_naive", 1.0, 1.2, 5.0, 1.1, 1.0)]
    fc = ForecastResult(mean=mean, lower=lower, upper=upper, horizon=horizon, levels=(80, 95),
                        model_scores=scores, method="seasonal_naive", backtest_mase=1.1,
                        nonnegative=True)
    outlook = Outlook(direction="down", expected_change_pct=-2.0, prob_increase=0.4,
                      scenarios={"pessimistic": float(lower[80].iloc[-1]), "base": float(mean.iloc[-1]),
                                 "optimistic": float(upper[80].iloc[-1])},
                      forecast_peak_date=idx[4] if forecast_peak else None, future_phase="declining",
                      confidence=outlook_conf)
    return TrendReport(name=s.name, analysis=analysis, forecast=fc, outlook=outlook)


# --------------------------------------------------------------------------- formatting helpers


def test_format_number_and_pct():
    assert format_number(1234567.8) == "1,234,568"
    assert format_number(45.25) == "45.2" or format_number(45.25) == "45.3"
    assert format_number(1.2) == "1.20"
    assert format_number(0.02) == "0.02"
    assert format_number(0.000123) == "0.00012"
    assert format_number(-0.35) == "−0.35"
    assert format_number(0.0) == "0"
    assert format_number(3.0, signed=True) == "+3.00"
    assert format_number(2.5, decimals=0) in ("2", "3")
    assert format_number(None) == "해당 없음"
    assert format_number(None, lang="en") == "n/a"
    assert format_number(float("nan")) == "해당 없음"
    assert format_number(float("inf"), lang="en") == "n/a"
    assert format_number(np.float64(12.34)) == "12.3"
    assert format_pct(12.345) == "+12.3%"
    assert format_pct(-5) == "−5.0%"
    assert format_pct(0.01) == "0.0%"
    assert format_pct(1234.5) == "+1,234.5%"
    assert format_pct(None) == "해당 없음"
    assert format_pct(None, lang="en") == "n/a"
    assert format_pct(3.14159, decimals=2, signed=False) == "3.14%"


def test_format_date_per_freq():
    ts = pd.Timestamp("2024-03-05")
    assert format_date(ts, "D") == "2024-03-05"
    assert format_date(ts, "W") == "2024-03-05"
    assert format_date(ts, "M") == "2024-03"
    assert format_date(ts, "Q") == "2024-Q1"
    assert format_date(pd.Timestamp("2024-11-01"), "Q") == "2024-Q4"
    assert format_date(ts, "Y") == "2024"
    assert format_date(None, "M") == "해당 없음"
    assert format_date(None, "M", lang="en") == "n/a"


def test_period_units_and_lang_validation():
    assert [period_unit(f) for f in "DWMQY"] == ["/일", "/주", "/월", "/분기", "/년"]
    assert period_unit("M", "en") == "/mo"
    with pytest.raises(ValueError):
        format_number(1.0, lang="fr")


# --------------------------------------------------------------------------- rule-based narrative


def test_rule_based_korean_content(fake_report):
    text = rule_based_narrative(*_parts(fake_report))
    for h in KO_HEADINGS:
        assert h in text
    positions = [text.index(h) for h in KO_HEADINGS]
    assert positions == sorted(positions)
    assert "성장기" in text                       # phase label (PHASE_KO)
    assert "80%" in text                          # phase confidence
    assert "2016-01" in text and "2025-12" in text  # data range, monthly formatting
    assert "+180.0%" in text and "+11.0%" in text   # total change & CAGR
    assert "**2021-12**부터 성장 가속" in text
    assert "0.02 → 1.20/월" in text
    assert "2018-07" in text                      # the anomaly
    assert "최근 기울기가 장기 평균보다 가파름" in text  # phase reasons
    assert "+35.0%" in text                       # recent change
    assert "+15.0%" in text                       # expected change
    assert "90%" in text                          # probability of increase
    assert "2026-12" in text                      # horizon end
    assert "비관" in text and "낙관" in text        # scenario range
    assert "holt_damped" in text and "MASE 0.65" in text
    assert "투자·재무 조언이 아닙니다" in text
    assert "<script>" in text                     # raw name; escaping is the renderer's job
    # deterministic
    assert text == rule_based_narrative(*_parts(fake_report))


def test_rule_based_english(fake_report):
    text = rule_based_narrative(*_parts(fake_report), lang="en")
    for h in ("## Past", "## Present", "## Outlook", "## Caveats"):
        assert h in text
    assert "Growing" in text
    assert "From **2021-12**: growth accelerated" in text
    assert "0.02 → 1.20/mo" in text
    assert "+15.0%" in text and "+180.0%" in text
    assert "not financial or investment advice" in text
    # apart from the series name itself, no Korean leaks into the English text
    assert not HANGUL.search(text.replace(fake_report.name, ""))


def test_none_metrics_are_handled(fake_report):
    rep = _replace_metrics(fake_report, total_change_pct=None, cagr_pct=None, recent_change_pct=None)
    rep = dataclasses.replace(rep, outlook=dataclasses.replace(rep.outlook, expected_change_pct=None))
    for lang in ("ko", "en"):
        text = rule_based_narrative(*_parts(rep), lang=lang)
        assert not re.search(r"\bNone\b|\bnan\b", text, flags=re.IGNORECASE)
    ko = rule_based_narrative(*_parts(rep))
    assert "정의되지 않습니다" in ko
    assert "## 미래 전망" in ko


def test_no_changepoints_no_anomalies(fake_report):
    analysis = dataclasses.replace(fake_report.analysis, changepoints=[], anomalies=[])
    text = rule_based_narrative(analysis, fake_report.forecast, fake_report.outlook)
    assert "구조 변화(추세 전환점)는 감지되지 않아" in text
    assert "이상치는 발견되지 않았습니다" in text
    assert "2021-12" not in text
    en = rule_based_narrative(analysis, fake_report.forecast, fake_report.outlook, lang="en")
    assert "No structural breaks" in en and "No statistical anomalies" in en


def test_at_most_three_anomalies(fake_report):
    s = fake_report.analysis.series
    idxs = [5, 20, 40, 60, 90]
    zs = [3.6, -9.0, 5.0, 3.7, 8.0]
    ans = [Anomaly(date=s.index[i], index=i, value=float(s.iloc[i]), expected=float(s.iloc[i]) - 1,
                   zscore=z) for i, z in zip(idxs, zs)]
    analysis = dataclasses.replace(fake_report.analysis, anomalies=ans)
    text = rule_based_narrative(analysis, fake_report.forecast, fake_report.outlook)
    assert "5건" in text
    shown = [i for i in idxs if f"- {s.index[i]:%Y-%m}: 실제" in text]
    assert shown == [20, 40, 90]                  # the three strongest |z|, in date order
    assert "z = −9.0" in text


def test_daily_report_caveats_and_units():
    rep = _make_report("D", n=400)
    text = rule_based_narrative(*_parts(rep))
    assert "/일" in text
    assert re.search(r"\d{4}-\d{2}-\d{2}", text)          # daily date format
    assert "변동성이 커서" in text                         # high volatility caveat
    assert "국면 판단의 신뢰도가 40%로 낮아" in text      # low phase confidence
    assert "예측 신뢰도가 낮습니다" in text
    assert "관측치가 2개뿐이라" in text                    # very recent level shift
    assert "정점" in text and "경 정점" in text            # forecast peak
    assert "**안정기**에서 **쇠퇴기** 국면으로" in text      # projected phase change
    assert "하락 쪽이 약간 우세" in text                   # prob_increase 0.4
    assert "예측 하한을 0으로 제한" in text
    assert "요일에 높고" in text                           # weekday seasonal pattern
    en = rule_based_narrative(*_parts(rep), lang="en")
    assert "/day" in en and "Stable" in en and "Declining" in en
    assert not HANGUL.search(en)


@pytest.mark.parametrize("freq,n,unit,datepat", [
    ("W", 156, "/주", r"\d{4}-\d{2}-\d{2}"),
    ("Q", 40, "/분기", r"\d{4}-Q[1-4]"),
    # yearly dates are the bare year; Korean prose attaches the counter (2016년부터, 2028년 예측값)
    ("Y", 30, "/년", r"\b\d{4}년"),
])
def test_other_frequencies(freq, n, unit, datepat):
    rep = _make_report(freq, n=n, horizon=4, changepoints=False, forecast_peak=False)
    text = rule_based_narrative(*_parts(rep))
    assert unit in text
    assert re.search(datepat, text)
    for h in KO_HEADINGS:
        assert h in text
    if freq == "Y":
        assert not re.search(r"\d{4}-(?:\d{2}|Q)", text)          # no month / quarter part
        assert not re.search(r"\b\d{4}(?:부터|까지|의|에) ", text)  # no bare year before a particle
        en = rule_based_narrative(*_parts(rep), lang="en")
        assert re.search(r"\b\d{4}\b", en) and "년" not in en


def test_short_series_caveat():
    rep = _make_report("M", n=18, horizon=6, changepoints=False, forecast_peak=False)
    text = rule_based_narrative(*_parts(rep))
    assert "계절 주기(12개월)의 2배에 못 미쳐" in text


# --------------------------------------------------------------------------- payload


def test_payload_json_serialisable_and_compact(fake_report):
    payload = build_llm_payload(*_parts(fake_report))
    dumped = json.dumps(payload, ensure_ascii=False, allow_nan=False)
    assert len(dumped) < 15000
    assert payload["series"]["start"] == "2016-01" and payload["series"]["end"] == "2025-12"
    assert len(payload["history"]) <= 60
    # recent detail is kept raw
    s = fake_report.analysis.series
    assert payload["history"][-1][0] == "2025-12"
    assert payload["history"][-1][1] == pytest.approx(float(s.iloc[-1]), rel=1e-3)
    assert payload["changepoints"][0]["date"] == "2021-12"
    assert payload["changepoints"][0]["kind"] == "acceleration"
    assert payload["anomalies"][0]["date"] == "2018-07"
    assert payload["phase"]["label"] == "growing" and payload["phase"]["reasons"]
    assert payload["metrics"]["total_change_pct"] == 180.0
    assert set(payload["forecast"]["table"][0]) == {"date", "mean", "lo80", "hi80", "lo95", "hi95"}
    assert len(payload["forecast"]["table"]) == 12
    assert len(payload["model_leaderboard"]) <= 5
    assert payload["model_leaderboard"][0]["model"] == "holt_damped"
    assert payload["outlook"]["prob_increase"] == 0.9
    assert payload["trend"]["end_value"] is not None


def _walk(obj):
    if isinstance(obj, dict):
        for v in obj.values():
            yield from _walk(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _walk(v)
    else:
        yield obj


def test_payload_bounded_for_long_series():
    rep = _make_report("D", n=2000, horizon=365)
    payload = build_llm_payload(*_parts(rep))
    for leaf in _walk(payload):
        assert leaf is None or type(leaf) in (str, int, float, bool)
    dumped = json.dumps(payload, ensure_ascii=False, allow_nan=False)
    assert len(dumped) < 20000
    assert len(payload["history"]) <= 60
    assert len(payload["forecast"]["table"]) <= 36
    assert payload["forecast"]["table"][-1]["date"] == format_date(rep.forecast.mean.index[-1], "D")


# --------------------------------------------------------------------------- mocked Claude


class FakeClient:
    """Mimics ``anthropic.Anthropic`` just enough: ``client.beta.messages.create(**kw)``."""

    def __init__(self, response=None, exc: Exception | None = None):
        self.calls: list[dict] = []
        self._response = response
        self._exc = exc
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if self._exc is not None:
            raise self._exc
        return self._response


def _resp(*blocks, stop_reason="end_turn"):
    return SimpleNamespace(stop_reason=stop_reason, content=list(blocks))


def _text(t):
    return SimpleNamespace(type="text", text=t)


def test_llm_narrative_request_shape_and_text_join(fake_report):
    client = FakeClient(_resp(SimpleNamespace(type="thinking", thinking="secret"),
                              _text("## 과거\n성장"), SimpleNamespace(type="fallback"),
                              _text("\n## 미래\n상승")))
    out = llm_narrative(*_parts(fake_report), client=client)
    assert out == "## 과거\n성장\n## 미래\n상승"
    assert len(client.calls) == 1
    kw = client.calls[0]
    assert kw["model"] == DEFAULT_MODEL == "claude-opus-5-5"
    assert kw["max_tokens"] == 16000
    assert kw["betas"] == ["server-side-fallback-2026-07-01"]
    assert kw["fallbacks"] == "default"
    assert kw["output_config"] == {"effort": "medium"}
    assert kw["system"] == SYSTEM_PROMPT
    for banned in ("thinking", "temperature", "top_p", "top_k", "budget_tokens"):
        assert banned not in kw
    msgs = kw["messages"]
    assert len(msgs) == 1 and msgs[0]["role"] == "user"      # no assistant prefill
    content = msgs[0]["content"]
    assert "## 과거: 어떻게 여기까지 왔나" in content
    facts = content.split("<facts>\n", 1)[1].split("\n</facts>", 1)[0]
    data = json.loads(facts)
    assert data["series"]["name"] == fake_report.name
    assert data["changepoints"][0]["date"] == "2021-12"


def test_llm_narrative_english_and_custom_model(fake_report):
    client = FakeClient(_resp(_text("ok")))
    assert llm_narrative(*_parts(fake_report), lang="en", model="claude-sonnet-5-5", client=client) == "ok"
    kw = client.calls[0]
    assert kw["model"] == "claude-sonnet-5-5"
    assert "## Outlook" in kw["messages"][0]["content"]
    assert "English" in kw["messages"][0]["content"]


def test_system_prompt_rules():
    p = SYSTEM_PROMPT.lower()
    assert "only the facts" in p or "use only" in p
    assert "hypothesis" in p
    assert "uncertainty" in p
    assert "financial" in p


def test_refusal_raises(fake_report):
    client = FakeClient(_resp(stop_reason="refusal"))
    with pytest.raises(NarrativeUnavailable):
        llm_narrative(*_parts(fake_report), client=client)


def test_empty_text_raises(fake_report):
    client = FakeClient(_resp(SimpleNamespace(type="thinking", thinking="...")))
    with pytest.raises(NarrativeUnavailable):
        llm_narrative(*_parts(fake_report), client=client)


def _sdk_errors():
    anthropic = pytest.importorskip("anthropic")
    httpx2 = pytest.importorskip("httpx2")
    req = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")

    def status(cls, code):
        return cls("boom", response=httpx2.Response(code, request=req), body=None)

    return anthropic, {
        "auth": status(anthropic.AuthenticationError, 401),
        "perm": status(anthropic.PermissionDeniedError, 403),
        "rate": status(anthropic.RateLimitError, 429),
        "status": status(anthropic.InternalServerError, 500),
        "conn": anthropic.APIConnectionError(request=req),
    }


@pytest.mark.parametrize("key,needle", [
    ("auth", "ANTHROPIC_API_KEY"), ("perm", "권한"), ("rate", "rate limit"),
    ("status", "HTTP 500"), ("conn", "연결할 수 없습니다"),
])
def test_sdk_errors_become_narrative_unavailable(fake_report, key, needle):
    _, errors = _sdk_errors()
    client = FakeClient(exc=errors[key])
    with pytest.raises(NarrativeUnavailable, match=needle):
        llm_narrative(*_parts(fake_report), client=client)


def test_missing_sdk(monkeypatch, fake_report):
    monkeypatch.setitem(sys.modules, "anthropic", None)
    with pytest.raises(NarrativeUnavailable, match="pip install anthropic"):
        llm_narrative(*_parts(fake_report), client=FakeClient(_resp(_text("x"))))


def test_generate_narrative_default_is_rule_and_offline(fake_report):
    client = FakeClient(exc=AssertionError("must not be called"))
    text, source = generate_narrative(*_parts(fake_report), client=client)
    assert source == "rule"
    assert text == rule_based_narrative(*_parts(fake_report))
    assert client.calls == []


def test_generate_narrative_uses_llm(fake_report):
    client = FakeClient(_resp(_text("LLM story")))
    assert generate_narrative(*_parts(fake_report), use_llm=True, client=client) == ("LLM story", "llm")


def test_generate_narrative_falls_back_on_connection_error(fake_report):
    _, errors = _sdk_errors()
    client = FakeClient(exc=errors["conn"])
    with pytest.warns(RuntimeWarning, match="규칙 기반"):
        text, source = generate_narrative(*_parts(fake_report), use_llm=True, lang="en", client=client)
    assert source == "rule"
    assert text == rule_based_narrative(*_parts(fake_report), lang="en")
    assert len(client.calls) == 1


def test_generate_narrative_falls_back_via_monkeypatch(monkeypatch, fake_report):
    def boom(*args, **kwargs):
        raise NarrativeUnavailable("no key")

    monkeypatch.setattr(nv, "llm_narrative", boom)
    with pytest.warns(RuntimeWarning, match="no key"):
        text, source = generate_narrative(*_parts(fake_report), use_llm=True)
    assert source == "rule" and "## 주의할 점" in text


def test_ask_passes_question(fake_report):
    client = FakeClient(_resp(_text("답변: 상승 확률은 약 90%입니다.")))
    question = "내년에도 계속 오를까요?"
    out = ask(fake_report, question, client=client)
    assert out.startswith("답변")
    kw = client.calls[0]
    content = kw["messages"][0]["content"]
    assert question in content
    assert content.index("<facts>") < content.index(question)   # facts first, question last
    assert kw["system"] == SYSTEM_PROMPT
    assert kw["fallbacks"] == "default" and "thinking" not in kw
    with pytest.raises(ValueError):
        ask(fake_report, "   ", client=client)


def test_ask_refusal(fake_report):
    with pytest.raises(NarrativeUnavailable):
        ask(fake_report, "why?", lang="en", client=FakeClient(_resp(stop_reason="refusal")))


# --------------------------------------------------------------------------- adversarial review: regressions



def _report_from_values(values, freq: str = "M", name: str = "custom", changepoints=(),
                        season_len: int | None = None, outlook_kw=None, levels=(80, 95),
                        half_width=None) -> TrendReport:
    """A hand-built report around an arbitrary series with a zero seasonal component."""
    vals = np.asarray(values, dtype=float)
    n = len(vals)
    idx = pd.date_range("2016-01-01", periods=n, freq={"M": "MS", "D": "D", "W": "W-SUN",
                                                       "Q": "QS", "Y": "YS"}[freq])
    s = pd.Series(vals, index=idx, name=name)
    sl = season_length(freq) if season_len is None else season_len
    trend = s.rolling(3, center=True, min_periods=1).mean()
    dec = Decomposition(trend=trend, seasonal=s * 0.0, resid=s - trend, period=sl)
    peak_i = int(np.argmax(vals))
    metrics = TrendMetrics(
        n_obs=n, freq=freq, start=idx[0], end=idx[-1], last_value=float(vals[-1]), mean=float(vals.mean()),
        total_change_pct=None, cagr_pct=None, recent_change_pct=None, slope=0.0, slope_pct=0.0,
        momentum=0.0, acceleration=0.0, volatility=0.01, trend_strength=0.5, seasonal_strength=0.0,
        mk_tau=0.0, mk_pvalue=0.5, direction="flat", peak_date=idx[peak_i], peak_value=float(vals[peak_i]),
        drawdown_pct=0.0)
    phase = LifecyclePhase(label="stable", confidence=0.7, reasons=["테스트"])
    analysis = AnalysisResult(name=name, series=s, freq=freq, season_length=sl, decomposition=dec,
                              metrics=metrics, changepoints=list(changepoints), anomalies=[], phase=phase)
    h = 6
    fidx = future_index(idx[-1], freq, h)
    mean = pd.Series(np.full(h, float(vals[-1])), index=fidx)
    hw = np.linspace(1.0, 2.0, h) if half_width is None else np.asarray(half_width, dtype=float)
    lower = {lv: mean - hw * (1 + (lv == 95)) for lv in levels}
    upper = {lv: mean + hw * (1 + (lv == 95)) for lv in levels}
    fc = ForecastResult(mean=mean, lower=lower, upper=upper, horizon=h, levels=tuple(levels),
                        model_scores=[ModelScore("naive", 1.0, 1.0, 1.0, 0.9, 1.0)], method="naive",
                        backtest_mase=0.9, nonnegative=False)
    okw = dict(direction="flat", expected_change_pct=0.5, prob_increase=0.5,
               scenarios={"pessimistic": float(vals[-1]) - 1, "base": float(vals[-1]),
                          "optimistic": float(vals[-1]) + 1},
               forecast_peak_date=None, future_phase="stable", confidence="medium")
    okw.update(outlook_kw or {})
    return TrendReport(name=name, analysis=analysis, forecast=fc, outlook=Outlook(**okw))


def test_level_shift_direction_is_trend_adjusted():
    # +10 jump inside a -2/period downtrend: the window means alone fall (47 -> 45), which used
    # to be reported as "the level dropped"; the true jump at the break is +10.
    t = np.arange(60.0)
    up = 200 - 2 * t + 10 * (t >= 30)
    cp_up = Changepoint(date=pd.Timestamp("2018-07-01"), index=30, slope_before=-2.0, slope_after=-2.0,
                        kind="level_shift", magnitude=0.3)
    rep = _report_from_values(up, changepoints=[cp_up])
    ko = rule_based_narrative(*_parts(rep))
    assert "**2018-07**에 수준이 한 단계 높아졌습니다 (약 +10.0)" in ko
    en = rule_based_narrative(*_parts(rep), lang="en")
    assert "the level jumped up (about +10.0)" in en
    # -30 drop inside a +2/period uptrend: window means differ by only -18
    down = 100 + 2 * t - 30 * (t >= 30)
    cp_down = dataclasses.replace(cp_up, slope_before=2.0, slope_after=2.0)
    ko = rule_based_narrative(*_parts(_report_from_values(down, changepoints=[cp_down])))
    assert "수준이 한 단계 낮아졌습니다 (약 −30.0)" in ko


def test_negative_base_is_not_called_near_zero():
    t = np.arange(48.0)
    rep = _report_from_values(-100 + t, name="손익")
    ko = rule_based_narrative(*_parts(rep))
    line = next(ln for ln in ko.splitlines() if ln.startswith("- 추세선"))
    assert "음수" in line and "0에 가까워" not in line
    assert "+46.0" in line                       # absolute change on the trend line (−98.5 -> −52.5)
    assert "최근 12개월 변화율은 기준 수준이 음수여서" in ko
    en = rule_based_narrative(*_parts(rep), lang="en")
    assert "undefined because the starting level is negative" in en
    assert "close to zero" not in en
    # a genuinely ~0 base keeps the "close to zero" wording
    ko0 = rule_based_narrative(*_parts(_report_from_values(np.r_[np.zeros(24), np.arange(24.0)])))
    assert "0에 가까워" in ko0


def test_recent_window_is_capped_by_series_length(fake_report):
    # analysis.compute_metrics measures recent_change_pct over min(season, n - 1) periods
    rep = _make_report("M", n=8, horizon=4, changepoints=False, forecast_peak=False)
    text = rule_based_narrative(*_parts(rep))
    assert "최근 7개월 동안 추세는" in text and "최근 12개월 동안 추세" not in text
    assert build_llm_payload(*_parts(rep))["metrics"]["recent_window_periods"] == 7
    assert build_llm_payload(*_parts(fake_report))["metrics"]["recent_window_periods"] == 12


def test_scenario_percentages_name_their_reference(fake_report):
    text = rule_based_narrative(*_parts(fake_report))
    s = fake_report.analysis.series
    cur = float(s.iloc[-12:].mean())                  # current level = mean of the last season
    base = fake_report.outlook.scenarios["base"]
    assert f"괄호는 최근 12개월 평균 {format_number(cur)} 대비" in text
    assert f"기준 {format_number(base)} ({format_pct((base - cur) / cur * 100)})" in text
    en = rule_based_narrative(*_parts(fake_report), lang="en")
    assert f"% vs. the average of the last 12 months, {format_number(cur, lang='en')}" in en
    # fake_report is barely seasonal: no seasonal caveat on the scenarios
    assert "계절적 높낮이" not in text


def test_seasonal_scenario_note():
    rep = _make_report("D", n=400)                   # seasonal_strength 0.85, weekly period
    assert "계절적 높낮이를 포함하므로" in rule_based_narrative(*_parts(rep))
    assert "include the seasonal swing" in rule_based_narrative(*_parts(rep), lang="en")


def test_flat_direction_with_visible_change_is_explained(fake_report):
    rep = dataclasses.replace(fake_report, outlook=dataclasses.replace(
        fake_report.outlook, direction="flat", expected_change_pct=33.3))
    ko = rule_based_narrative(*_parts(rep))
    assert "**+33.3%** 변화가 예상되지만 예측 불확실성에 비해 크지 않아 방향은 **보합**으로 봅니다" in ko
    assert "변화(보합)" not in ko
    assert "예측 불확실성 대비 크지 않아 **보합** 전망" in ko
    en = rule_based_narrative(*_parts(rep), lang="en")
    assert "small relative to the forecast uncertainty" in en
    small = dataclasses.replace(rep, outlook=dataclasses.replace(rep.outlook, expected_change_pct=1.0))
    assert "**+1.0%** 변화(보합)가 예상됩니다" in rule_based_narrative(*_parts(small))


@pytest.mark.parametrize("p,ko,en", [
    (0.95, "상승 가능성이 매우 높습니다", "a rise is very likely"),
    (0.90, "상승 가능성이 매우 높습니다", "a rise is very likely"),
    (0.70, "상승 쪽이 우세합니다", "a rise is more likely than not"),
    (0.55, "상승 쪽이 약간 우세합니다", "a rise is slightly more likely"),
    (0.50, "상승과 하락 가능성이 엇비슷합니다", "about equally likely"),
    (0.45, "하락 쪽이 약간 우세합니다", "a fall is slightly more likely"),
    (0.30, "하락 쪽이 우세합니다", "a fall is more likely than not"),
    (0.10, "하락 가능성이 매우 높습니다", "a fall is very likely"),
    (0.0, "하락 가능성이 매우 높습니다", "a fall is very likely"),
])
def test_probability_wording_is_symmetric(fake_report, p, ko, en):
    rep = dataclasses.replace(fake_report, outlook=dataclasses.replace(fake_report.outlook, prob_increase=p))
    # a model probability of exactly 0 is shown as "<1%", never as a certain "0%" (see format_prob)
    shown = "**<1%**" if p == 0.0 else f"약 **{p * 100:.0f}%**"
    assert f"{shown}로, {ko}" in rule_based_narrative(*_parts(rep))
    assert en in rule_based_narrative(*_parts(rep), lang="en")


def test_missing_probability_is_not_reported_as_zero(fake_report):
    rep = dataclasses.replace(fake_report, outlook=dataclasses.replace(
        fake_report.outlook, prob_increase=float("nan")))
    text = rule_based_narrative(*_parts(rep))
    assert "상승 확률은 계산되지 않았습니다" in text
    assert "하락 가능성이 매우 높습니다" not in text
    assert "상승 확률 해당 없음," in text and "해당 없음%" not in text
    phase = dataclasses.replace(rep.analysis.phase, confidence=float("nan"))
    rep2 = dataclasses.replace(rep, analysis=dataclasses.replace(rep.analysis, phase=phase))
    for lang, na in (("ko", "해당 없음"), ("en", "n/a")):
        t2 = rule_based_narrative(*_parts(rep2), lang=lang)
        assert f"{na}%" not in t2 and na in t2


def test_constant_series_no_fake_widening_or_all_time_high():
    hw = np.linspace(6.4e-6, 2.2e-5, 6)                # numerical-noise interval widths
    rep = _report_from_values(np.full(36, 5.0), half_width=hw)
    ko = rule_based_narrative(*_parts(rep))
    assert "관측 기간 내내 값이 변하지 않았습니다 (항상 5.00)" in ko
    assert "역대 최고" not in ko
    assert "배 넓어집니다" not in ko
    en = rule_based_narrative(*_parts(rep), lang="en")
    assert "stayed at 5.00" in en and "all-time high" not in en and "widens" not in en


def test_name_newline_cannot_inject_headings(fake_report):
    name = "evil\n## 주의할 점\n- fake"
    analysis = dataclasses.replace(fake_report.analysis, name=name)
    text = rule_based_narrative(analysis, fake_report.forecast, fake_report.outlook)
    headings = [ln for ln in text.splitlines() if ln.startswith("## ")]
    assert headings == KO_HEADINGS
    assert "evil ## 주의할 점 - fake" in text.splitlines()[0]


def test_single_interval_level_and_no_levels():
    rep = _report_from_values(np.arange(1.0, 37.0), levels=(80,))
    text = rule_based_narrative(*_parts(rep))
    assert "95% 구간" not in text and "80% 예측구간" in text
    payload = build_llm_payload(*_parts(rep))
    assert set(payload["forecast"]["table"][0]) == {"date", "mean", "lo80", "hi80"}
    assert payload["forecast"]["levels"] == [80]
    rep0 = _report_from_values(np.arange(1.0, 37.0), levels=())
    for lang in ("ko", "en"):
        t0 = rule_based_narrative(*_parts(rep0), lang=lang)
        assert "( " not in t0 and "()" not in t0 and "% interval" not in t0
    assert set(build_llm_payload(*_parts(rep0))["forecast"]["table"][0]) == {"date", "mean"}


def test_inputs_are_not_mutated(fake_report):
    before = copy.deepcopy(fake_report)
    for lang in ("ko", "en"):
        rule_based_narrative(*_parts(fake_report), lang=lang)
    build_llm_payload(*_parts(fake_report))
    a, b = fake_report.analysis, before.analysis
    pd.testing.assert_series_equal(a.series, b.series)
    pd.testing.assert_series_equal(a.decomposition.trend, b.decomposition.trend)
    assert a.changepoints == b.changepoints and a.anomalies == b.anomalies
    assert a.metrics == b.metrics and a.phase == b.phase
    pd.testing.assert_series_equal(fake_report.forecast.mean, before.forecast.mean)
    assert fake_report.outlook == before.outlook


def test_payload_history_downsampling_matches_block_means():
    rep = _make_report("M", n=100, horizon=12)
    s = rep.analysis.series
    payload = build_llm_payload(*_parts(rep))
    hist = payload["history"]
    vals = s.to_numpy()
    old_n, block = 100 - 24, int(np.ceil(76 / 36))   # 3-period blocks for the older part
    expected_old = [[f"{s.index[i]:%Y-%m}", vals[i:min(i + block, old_n)].mean()] for i in range(0, old_n, block)]
    assert len(hist) == len(expected_old) + 24 <= 60
    for (d, v), (ed, ev) in zip(hist, expected_old):
        assert d == ed and v == pytest.approx(ev, rel=1e-3)
    assert [d for d, _ in hist[-24:]] == [f"{t:%Y-%m}" for t in s.index[-24:]]
    assert [v for _, v in hist[-24:]] == pytest.approx(list(vals[-24:]), rel=1e-3)
    assert hist == sorted(hist, key=lambda r: r[0])
    assert "3 consecutive periods" in payload["history_note"]
    assert "last 12 forecast periods" in payload["outlook"]["expected_change_basis"]


def test_llm_default_client_is_constructed_lazily(monkeypatch, fake_report):
    anthropic = pytest.importorskip("anthropic")
    fake = FakeClient(_resp(_text("from default client")))
    made = []

    def factory(*args, **kwargs):
        made.append((args, kwargs))
        return fake

    monkeypatch.setattr(anthropic, "Anthropic", factory)
    assert llm_narrative(*_parts(fake_report)) == "from default client"
    assert made == [((), {})] and len(fake.calls) == 1


def test_old_sdk_type_error_falls_back(fake_report):
    exc = TypeError("create() got an unexpected keyword argument 'fallbacks'")
    with pytest.raises(NarrativeUnavailable, match="pip install -U anthropic"):
        llm_narrative(*_parts(fake_report), client=FakeClient(exc=exc))
    with pytest.warns(RuntimeWarning, match="anthropic"):
        text, source = generate_narrative(*_parts(fake_report), use_llm=True, client=FakeClient(exc=exc))
    assert source == "rule" and text == rule_based_narrative(*_parts(fake_report))


def test_generate_narrative_falls_back_on_refusal(fake_report):
    with pytest.warns(RuntimeWarning, match="refusal"):
        text, source = generate_narrative(*_parts(fake_report), use_llm=True,
                                          client=FakeClient(_resp(stop_reason="refusal")))
    assert source == "rule" and "## 주의할 점" in text


def test_ask_english(fake_report):
    client = FakeClient(_resp(_text("Probably, yes.")))
    assert ask(fake_report, "Will it keep rising?", lang="en", client=client) == "Probably, yes."
    content = client.calls[0]["messages"][0]["content"]
    assert "English" in content and content.rstrip().endswith("Will it keep rising?")


# --------------------------------------------------------------------------- integration with the real engine


_BAD_TOKEN = re.compile(r"\bNone\b|\bnan\b|\binf\b|NaT", flags=re.IGNORECASE)


def _real_cases():
    t = np.arange(60.0)
    idx = pd.date_range("2020-01-01", periods=60, freq="MS")
    return [
        ("breakout-M", sample("breakout", n=120, freq="M")),
        ("fad-W", sample("fad", n=156, freq="W")),
        ("seasonal-D", sample("seasonal", n=400, freq="D")),
        ("reviving-Q", sample("reviving", n=40, freq="Q")),
        ("emerging-Y", sample("emerging", n=25, freq="Y")),
        ("negative", pd.Series(-100 + t + np.sin(t), index=idx)),
        ("zeros", pd.Series(np.zeros(60), index=idx)),
        ("tiny-n", pd.Series([1.0, 2.0, 3.0], index=idx[:3])),
    ]


@pytest.mark.parametrize("label,series", _real_cases(), ids=[c[0] for c in _real_cases()])
def test_real_engine_output_end_to_end(label, series):
    analysis_mod = pytest.importorskip("trendai.analysis")
    forecast_mod = pytest.importorskip("trendai.forecast")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = analysis_mod.analyze(series, name=label)
        f = forecast_mod.auto_forecast(a.series, freq=a.freq)
        o = forecast_mod.compute_outlook(a, f)
    for lang, heads in (("ko", KO_HEADINGS), ("en", ["## " + h for h in nv.SECTION_HEADINGS["en"]])):
        text = rule_based_narrative(a, f, o, lang=lang)
        assert [ln for ln in text.splitlines() if ln.startswith("## ")] == heads
        assert not _BAD_TOKEN.search(text.replace(label, "")), text
        if lang == "en":
            assert not HANGUL.search(text.replace(label, ""))
    payload = build_llm_payload(a, f, o)
    dumped = json.dumps(payload, ensure_ascii=False, allow_nan=False)
    assert len(dumped) < 20000 and len(payload["history"]) <= 60
    assert payload["series"]["end"] == format_date(a.series.index[-1], a.freq)


def test_tiny_and_huge_magnitudes_do_not_collapse():
    assert format_number(1.05e-9) == "1.05×10⁻⁹"
    assert format_number(-2.5e-11, signed=True) == "−2.5×10⁻¹¹"
    assert format_number(3e-5, signed=True) == "+3×10⁻⁵"
    assert format_number(2.5e20) == "2.5×10²⁰"
    assert format_number(999_999_999_999_999.0) == "999,999,999,999,999"
    assert format_number(1e-9, decimals=2) == "0.00"           # explicit decimals are honoured
    t = np.arange(48.0)
    rep = _report_from_values(1e-9 * (1 + 0.01 * t), name="tiny")
    rep = _replace_metrics(rep, total_change_pct=47.0)
    text = rule_based_narrative(*_parts(rep))
    line = next(ln for ln in text.splitlines() if ln.startswith("- 추세선"))
    assert "0에서 0까지" not in line and "×10⁻⁹" in line
    assert "최고치는 2019-12의 1.47×10⁻⁹" in text
    cur = float(np.mean(1e-9 * (1 + 0.01 * t[-12:])))
    assert f"괄호는 최근 12개월 평균 {format_number(cur)} 대비" in text   # % shown: tiny base is not ~0
    assert "×10⁻⁹" in format_number(cur)


def test_payload_with_numpy_scalar_levels_is_json_serialisable():
    rep = _report_from_values(np.arange(1.0, 37.0), levels=(np.int64(90),))
    payload = build_llm_payload(*_parts(rep))
    json.dumps(payload, allow_nan=False)
    assert payload["outlook"]["scenario_interval_level"] == 90
    assert type(payload["outlook"]["scenario_interval_level"]) is int
    assert "90% 예측구간" in rule_based_narrative(*_parts(rep))


def test_mase_close_to_one_is_not_oversold(fake_report):
    # the naive comparison now comes from ForecastResult.skill (same backtest windows); a near-tie
    # must still not be sold as "about 1 % lower error"
    rep = dataclasses.replace(fake_report, forecast=dataclasses.replace(
        fake_report.forecast, backtest_mase=0.99, skill=0.99))
    ko = rule_based_narrative(*_parts(rep))
    assert "비슷해(차이 5% 이내)" in ko and "약 1%" not in ko and "MASE 0.99" in ko
    en = rule_based_narrative(*_parts(rep), lang="en")
    assert "about the same as" in en and "about 1%" not in en
    # without a skill a raw MASE of 0.99 supports no naive comparison at all
    rep0 = dataclasses.replace(fake_report, forecast=dataclasses.replace(fake_report.forecast, backtest_mase=0.99))
    ko0 = rule_based_narrative(*_parts(rep0))
    assert "MASE 0.99" in ko0 and "단순 기준보다 나은지는 판단하지 않았습니다" in ko0
    assert "비슷" not in ko0.split("- 정확도:")[1].split("\n")[0]


# --------------------------------------------------------------------------- fixer round: regressions


def _with_outlook(rep: TrendReport, **kw) -> TrendReport:
    return dataclasses.replace(rep, outlook=dataclasses.replace(rep.outlook, **kw))


def _with_forecast(rep: TrendReport, **kw) -> TrendReport:
    return dataclasses.replace(rep, forecast=dataclasses.replace(rep.forecast, **kw))


@pytest.mark.parametrize("p,shown", [
    (1.0, ">99%"), (0.9987, ">99%"), (0.995, ">99%"), (0.994, "99%"), (0.9, "90%"),
    (0.5, "50%"), (0.006, "1%"), (0.005, "<1%"), (0.0013, "<1%"), (0.0, "<1%"),
])
def test_probabilities_are_never_shown_as_certain(fake_report, p, shown):
    assert nv.format_prob(p) == shown
    rep = _with_outlook(fake_report, prob_increase=p)
    ko = rule_based_narrative(*_parts(rep))
    en = rule_based_narrative(*_parts(rep), lang="en")
    assert f"상승 확률 {shown}," in ko.splitlines()[0]
    assert f"Probability of increase {shown}," in en.splitlines()[0]
    approx = "" if shown[0] in "<>" else "약 "
    assert f"높아질 확률은 {approx}**{shown}**로" in ko
    for text in (ko, en):
        assert "100%" not in text and not re.search(r"(?<![\d.])0%", text)
    assert nv.format_prob(None) == "해당 없음" and nv.format_prob(float("nan"), "en") == "n/a"


def test_sub_items_are_nested_bullets(fake_report):
    s = fake_report.analysis.series
    ans = [Anomaly(date=s.index[i], index=i, value=float(s.iloc[i]), expected=float(s.iloc[i]) - 1, zscore=z)
           for i, z in ((20, 5.0), (40, -6.0))]
    analysis = dataclasses.replace(fake_report.analysis, anomalies=ans)
    for lang in ("ko", "en"):
        lines = rule_based_narrative(analysis, fake_report.forecast, fake_report.outlook, lang=lang).splitlines()
        intro = [i for i, ln in enumerate(lines) if ln.startswith("- ") and ln.endswith(":")]
        assert len(intro) == 2                                    # changepoints + anomalies
        for i in intro:
            assert lines[i + 1].startswith("  - ")                # nested, two-space indent
        nested = [ln for ln in lines if ln.startswith("  - ")]
        assert len(nested) == 3                                   # 1 changepoint + 2 anomalies
        # nothing else is indented, and the nested items follow their intro line directly
        assert all(ln.startswith(("- ", "  - ", "## ", "**")) or not ln for ln in lines)
    ko = rule_based_narrative(analysis, fake_report.forecast, fake_report.outlook)
    assert "  - **2021-12**부터 성장 가속" in ko


@pytest.mark.parametrize("stop,lang,note", [
    ("max_tokens", "ko", "잘렸을 수 있습니다"),
    ("max_tokens", "en", "may be cut off"),
    ("model_context_window_exceeded", "en", "may be cut off"),
])
def test_truncated_llm_response_warns_and_says_so(fake_report, stop, lang, note):
    client = FakeClient(_resp(_text("## 과거\n- 성장 중이며 최근"), stop_reason=stop))
    with pytest.warns(RuntimeWarning, match=stop):
        out = llm_narrative(*_parts(fake_report), lang=lang, client=client)
    assert out.startswith("## 과거\n- 성장 중이며 최근") and out.rstrip().endswith(out.rstrip().splitlines()[-1])
    assert note in out.splitlines()[-1]
    with pytest.warns(RuntimeWarning, match=stop):
        ans = ask(fake_report, "why?", lang=lang, client=FakeClient(_resp(_text("Because"), stop_reason=stop)))
    assert ans.startswith("Because") and note in ans
    with pytest.warns(RuntimeWarning, match=stop):
        text, source = generate_narrative(*_parts(fake_report), use_llm=True, lang=lang, client=FakeClient(
            _resp(_text("partial"), stop_reason=stop)))
    assert source == "llm" and note in text


def test_complete_llm_response_gets_no_truncation_note(fake_report):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = llm_narrative(*_parts(fake_report), client=FakeClient(_resp(_text("완결된 해설"))))
    assert out == "완결된 해설"


@pytest.mark.parametrize("mase,skill,ko,not_ko", [
    # multi-step MASE > 1 while far better than naive (verifier: emerging 1.36 vs. naive benchmarks 2.4-4.4)
    (1.36, 0.56, "보다 오차가 **약 44%** 작았습니다", ("낫지 않", "나은 성과를 보이지 못")),
    # MASE far below 1 while only 3 % better than naive (verifier: 'policy', claimed 71 %)
    (0.29, 0.97, "비슷해(차이 5% 이내)", ("71%", "작았습니다", "나은 성과를 보이지 못")),
    (0.65, 0.65, "보다 오차가 **약 35%** 작았습니다", ("낫지 않", "나은 성과를 보이지 못")),
    (0.80, 1.20, "보다 나은 성과를 보이지 못했습니다 (오차가 약 20% 더 큼)", ("작았습니다",)),
    (0.80, 2.50, "(오차가 약 2.5배)", ("작았습니다",)),
    (0.70, 0.001, "보다 오차가 **99% 이상** 작았습니다", ("100%",)),
])
def test_accuracy_is_judged_by_skill_not_raw_mase(fake_report, mase, skill, ko, not_ko):
    rep = _with_forecast(fake_report, backtest_mase=mase, skill=skill)
    text = rule_based_narrative(*_parts(rep))
    assert ko in text
    for bad in not_ko:
        assert bad not in text
    assert f"MASE {mase:.2f}" in text                              # kept as a technical note ...
    acc = next(ln for ln in text.splitlines() if ln.startswith("- 정확도:"))
    assert "기술 참고" in acc                                      # ... with a one-line explanation
    beaten = "백테스트에서 단순 기준 예측보다 나은 성과를 보이지 못했습니다." in text.split("## 주의할 점")[1]
    assert beaten == (skill > 1)                                   # caveat only when skill > 1
    en = rule_based_narrative(*_parts(rep), lang="en")
    assert ("did not beat" in en) == (skill > 1)
    assert "no better than" not in en and "seasonal-naive baseline" not in en


def test_accuracy_without_skill_makes_no_naive_claim(fake_report):
    for mase in (0.4, 1.6):
        rep = _with_forecast(fake_report, backtest_mase=mase, skill=None)
        ko = rule_based_narrative(*_parts(rep))
        assert "단순 기준보다 나은지는 판단하지 않았습니다" in ko
        for bad in ("작았습니다", "낫지 않", "나은 성과를 보이지 못", "감소"):
            assert bad not in ko
        assert "not assessed" in rule_based_narrative(*_parts(rep), lang="en")
    rep = _with_forecast(fake_report, backtest_mase=None, skill=None)
    assert "백테스트로 정확도를 검증하지 못했습니다" in rule_based_narrative(*_parts(rep))


def test_payload_and_prompt_describe_skill_not_mase_below_one(fake_report):
    payload = build_llm_payload(*_parts(_with_forecast(fake_report, skill=0.6123)))
    assert payload["forecast"]["skill"] == 0.612
    assert "same backtest windows" in payload["conventions"]["skill"]
    assert "< 1 means more accurate" not in json.dumps(payload["conventions"])
    assert "do not compare it with 1" in payload["conventions"]["mase"]
    assert build_llm_payload(*_parts(fake_report))["forecast"]["skill"] is None
    p = SYSTEM_PROMPT.lower()
    assert "forecast.skill" in p and "below 1 = more accurate" not in p
    assert ">99%" in SYSTEM_PROMPT and "never read the raw mase against 1" in p


def test_real_engine_mase_above_one_is_not_called_worse_than_naive():
    analysis_mod = pytest.importorskip("trendai.analysis")
    forecast_mod = pytest.importorskip("trendai.forecast")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = analysis_mod.analyze(sample("emerging"), name="emerging")
        f = forecast_mod.auto_forecast(a.series, freq=a.freq)
        o = forecast_mod.compute_outlook(a, f)
    text = rule_based_narrative(a, f, o)
    assert "낫지 않음" not in text
    if f.skill is not None and f.skill < 1:
        assert "나은 성과를 보이지 못" not in text and "작았습니다" in text
    assert "100%" not in text


def test_expected_change_names_the_seasonal_adjustment():
    # weekly data, 26-step horizon < 52-week season: compute_outlook removes the seasonal effect
    rep = _make_report("W", n=156, horizon=26, changepoints=False, forecast_peak=False)
    ko = rule_based_narrative(*_parts(rep))
    line = next(ln for ln in ko.splitlines() if "변할 것으로 예상됩니다" in ln)
    assert line.startswith("- 계절 효과를 보정한 향후 26주 예측 평균은 최근 52주 평균보다")
    assert "계절 주기(52주)를 다 채우지 못해" in line
    en = rule_based_narrative(*_parts(rep), lang="en")
    assert "The seasonally adjusted average forecast over the next 26 weeks" in en
    assert "A 26-week window covers only part of the 52-week seasonal cycle" in en
    basis = build_llm_payload(*_parts(rep))["outlook"]["expected_change_basis"]
    assert "last 26 forecast periods" in basis and "covers only part of the 52-period seasonal cycle" in basis
    # whole seasons (monthly, 12-step horizon): plain averages, no adjustment wording
    rep_m = _make_report("M", n=120, horizon=12, changepoints=False, forecast_peak=False)
    ko_m = rule_based_narrative(*_parts(rep_m))
    assert "향후 12개월 예측 평균은 최근 12개월 평균보다" in ko_m and "계절 효과를 보정한" not in ko_m
    assert "covers only part" not in build_llm_payload(*_parts(rep_m))["outlook"]["expected_change_basis"]
    # horizon longer than a season: the last season of the forecast
    rep_l = _make_report("M", n=120, horizon=24, changepoints=False, forecast_peak=False)
    assert "예측 마지막 12개월 평균은 최근 12개월 평균보다" in rule_based_narrative(*_parts(rep_l))


def test_korean_wording_confidence_phase_shift_and_drawdown(fake_report):
    ko = rule_based_narrative(*_parts(fake_report))
    assert "예측 신뢰도는 **높은 편**입니다." in ko and "높음입니다" not in ko
    for conf, word in (("medium", "보통 수준"), ("low", "낮은 편")):
        t = rule_based_narrative(*_parts(_with_outlook(fake_report, confidence=conf)))
        assert f"예측 신뢰도는 **{word}**입니다." in t
    shift = rule_based_narrative(*_parts(_with_outlook(fake_report, future_phase="peaking")))
    line = next(ln for ln in shift.splitlines() if "넘어갈 가능성" in ln)
    assert line.count("국면") == 1 and "**성장기**에서 **정점/성숙기** 국면으로" in line
    # drawdown tiers: −95 % is far more than "less than half", and "고점 대비" is not repeated
    t = np.arange(48.0)
    vals = np.r_[np.linspace(10, 100, 24), np.linspace(100, 5, 24)]
    rep = _replace_metrics(_report_from_values(vals), drawdown_pct=-95.0, peak_value=100.0,
                           peak_date=pd.Timestamp("2017-12-01"), last_value=5.0)
    text = rule_based_narrative(*_parts(rep))
    line = next(ln for ln in text.splitlines() if ln.startswith("- 최고치는"))
    assert "고점보다 95.0% 낮아 고점의 1/10 이하로 급감했습니다" in line and "절반" not in line
    assert line.count("고점 대비") == 0
    assert "less than a tenth of the peak" in rule_based_narrative(*_parts(rep), lang="en")
    for dd, word in ((-3.0, "여전히 고점 부근입니다"), (-12.0, "다소 내려와 있습니다"),
                     (-35.0, "상당히 내려와 있습니다"), (-60.0, "고점의 절반 이하로"), (-85.0, "1/5 이하로")):
        r = _replace_metrics(rep, drawdown_pct=dd)
        assert word in rule_based_narrative(*_parts(r))
    del t


def test_yearly_dates_read_naturally_in_korean():
    rep = _make_report("Y", n=8, horizon=5, changepoints=False, forecast_peak=False)
    ko = rule_based_narrative(*_parts(rep))
    y0, y1 = rep.analysis.series.index[0].year, rep.analysis.series.index[-1].year
    assert f"{y0}년부터 {y1}년까지" in ko
    assert re.search(r"최고치는 \d{4}년의 ", ko)
    end = rep.forecast.mean.index[-1].year
    assert f"{end}년 예측값은" in ko and f"예측 종료 {end}년" in ko
    assert format_date(rep.forecast.mean.index[-1], "Y") == str(end)      # public helper unchanged
    assert build_llm_payload(*_parts(rep))["series"]["end"] == str(y1)    # payload keeps bare years


def _seasonal_low_report(lang_reasons=None) -> TrendReport:
    """A flat, strongly seasonal monthly series that ends in its seasonal low (December)."""
    n = 120
    idx = pd.date_range("2016-01-01", periods=n, freq="MS")
    seas = 8.0 * np.cos(2 * np.pi * (idx.month.to_numpy() - 3) / 12)   # high in March, low in September
    seas = seas - seas.mean()
    trend = 50.0 + 0.002 * np.arange(n)
    y = trend + seas
    y[-1] = trend[-1] + seas.min()                                      # the latest value at a seasonal low
    s = pd.Series(y, index=idx, name="seasonal")
    rep = _report_from_values(y, name="seasonal")
    dec = Decomposition(trend=pd.Series(trend, index=idx), seasonal=pd.Series(y - trend, index=idx),
                        resid=pd.Series(np.zeros(n), index=idx), period=12)
    peak_i = int(np.argmax(y))
    metrics = dataclasses.replace(rep.analysis.metrics, seasonal_strength=0.95, peak_date=idx[peak_i],
                                  peak_value=float(y[peak_i]), last_value=float(y[-1]),
                                  drawdown_pct=float((y[-1] - y[peak_i]) / y[peak_i] * 100))
    phase = LifecyclePhase(label="stable", confidence=0.8, reasons=lang_reasons or ["추세 없음"])
    analysis = dataclasses.replace(rep.analysis, series=s, decomposition=dec, metrics=metrics, phase=phase)
    return dataclasses.replace(rep, analysis=analysis)


def test_seasonal_low_is_not_framed_as_a_fall_from_the_peak():
    rep = _seasonal_low_report()
    dd = rep.analysis.metrics.drawdown_pct
    assert dd < -20                                                    # raw: well below the peak
    ko = rule_based_narrative(*_parts(rep))
    line = next(ln for ln in ko.splitlines() if ln.startswith("- 최고치는"))
    assert "계절적으로 낮은 시기" in line and "계절 요인을 뺀 추세선 기준으로는" in line
    assert "여전히 고점 부근입니다" in line and "상당히 내려와" not in line
    en = rule_based_narrative(*_parts(rep), lang="en")
    assert "seasonal low" in en and "still close to the high" in en
    # the English evidence for 'stable' never argues with the distance from the peak
    evidence = [ln for ln in en.splitlines() if ln.startswith("- Evidence:")]
    assert evidence and not any("peak" in ln for ln in evidence)
    assert all(ln.rstrip().endswith(".") for ln in evidence)          # full sentences, not fragments
    # a non-seasonal series keeps the plain drawdown sentence
    plain = _replace_metrics(rep, seasonal_strength=0.1)
    assert "계절 요인을 뺀" not in rule_based_narrative(*_parts(plain))


def test_english_evidence_translates_the_analysis_reasons():
    reasons = ["Mann-Kendall 검정에서 전체 기간의 뚜렷한 추세가 확인되지 않습니다 (p = 0.215).",
               "추세의 변동 폭(최고-최저)이 최고 수준 대비 4.7%에 그칩니다.",
               "계절성 강도가 0.96로 주기적 변동이 뚜렷합니다."]
    rep = _seasonal_low_report(reasons)
    en = rule_based_narrative(*_parts(rep), lang="en")
    evidence = [ln[len("- Evidence: "):] for ln in en.splitlines() if ln.startswith("- Evidence: ")]
    assert evidence == [
        "The Mann-Kendall test finds no clear trend over the whole period (p = 0.215).",
        "The trend's range (high minus low) is only 4.7% of its highest level.",
        "Seasonal strength is 0.96: a clear recurring cycle.",
    ]
    ko = rule_based_narrative(*_parts(rep))
    assert [ln[len("- 근거: "):] for ln in ko.splitlines() if ln.startswith("- 근거: ")] == reasons


@pytest.mark.parametrize("ko,en", [
    ("최근 18개월 동안 추세가 +143.4% 변했습니다.", "Over the last 18 months the trend changed by +143.4%."),
    ("최근 1주 동안 추세 기울기는 −0.5/주입니다.", "Over the last 1 week the trend slope was −0.5 per week."),
    ("현재 추세는 고점(2021-07)보다 94.6% 낮은 수준입니다.", "The trend is 94.6% below its peak (2021-07)."),
    ("2022-12-04에 수준이 한 단계 높아졌습니다 (약 +29.6%).", "On 2022-12-04 the level stepped up (about +29.6%)."),
    ("2024-Q2에 구조 변화(하락 전환)가 감지되었습니다.", "A structural break (turn downward) was detected in 2024-Q2."),
    ("Mann-Kendall 검정에서 전체 기간의 상승 추세가 유의합니다 (τ = 0.69, p < 0.001).",
     "The Mann-Kendall test finds a significant upward trend over the whole period (τ = 0.69, p < 0.001)."),
    ("고점(2021-07) 이후 하락해 2023-01에 저점을 찍은 뒤 추세 규모의 12.0%만큼 반등했습니다.",
     "After the peak (2021-07) it fell to a low in 2023-01, then rebounded by 12.0% of the trend's scale."),
    ("Already English.", "Already English."),
    ("처음 보는 근거 문장입니다.", None),
    ("", None),
])
def test_translate_reason(ko, en):
    assert nv.translate_reason(ko) == en


def test_real_engine_english_evidence_matches_korean_reasons():
    analysis_mod = pytest.importorskip("trendai.analysis")
    for name in ("seasonal", "fad", "emerging", "volatile"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            a = analysis_mod.analyze(sample(name), name=name)
        translated = [nv.translate_reason(r) for r in a.phase.reasons]
        assert all(translated), (name, a.phase.reasons)
        f = ForecastResult(mean=pd.Series([1.0], index=future_index(a.series.index[-1], a.freq, 1)),
                           lower={}, upper={}, horizon=1, levels=(), model_scores=[], method="naive")
        o = Outlook(direction="flat", expected_change_pct=0.0, prob_increase=0.5, scenarios={},
                    forecast_peak_date=None, future_phase=a.phase.label, confidence="low")
        en = rule_based_narrative(a, f, o, lang="en")
        assert [ln[len("- Evidence: "):] for ln in en.splitlines() if ln.startswith("- Evidence: ")] \
            == translated[:4]


def test_fallback_evidence_is_phase_aware_and_complete(fake_report):
    # reasons that cannot be translated -> metric evidence; distance from the peak only backs a
    # declining / peaking / reviving call
    rep = _replace_metrics(fake_report, drawdown_pct=-40.0)
    for label, has_peak in (("declining", True), ("stable", False), ("growing", False)):
        phase = LifecyclePhase(label=label, confidence=0.7, reasons=["번역할 수 없는 근거"])
        r = dataclasses.replace(rep, analysis=dataclasses.replace(rep.analysis, phase=phase))
        en = rule_based_narrative(*_parts(r), lang="en")
        evidence = [ln for ln in en.splitlines() if ln.startswith("- Evidence: ")]
        assert 2 <= len(evidence) <= 4 and all(ln.endswith(".") for ln in evidence)
        assert any("below the peak" in ln for ln in evidence) == has_peak
        assert any("Over the last 12 months the trend changed by +35.0%." in ln for ln in evidence)
        assert not HANGUL.search(en.replace(r.name, ""))
