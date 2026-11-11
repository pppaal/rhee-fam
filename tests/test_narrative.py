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
    # round 4 (U6): the shared changepoint vocabulary (analysis.changepoint_label)
    assert "From **2021-12**: growth accelerating" in text
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
    # round 4 (U7): with lang='en' the fallback warning (and the reason in it) is English
    with pytest.warns(RuntimeWarning, match="rule-based narrative") as rec:
        text, source = generate_narrative(*_parts(fake_report), use_llm=True, lang="en", client=client)
    assert not any(HANGUL.search(str(w.message)) for w in rec)
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
    # round 5 (V2-4): the scenarios are single horizon-end values, so the narrative now compares them
    # with the observation every other surface uses (narrative.scenario_reference: the latest value
    # for a barely seasonal series, as the terminal's "(비교: 현재 …)" and the HTML scenario bar),
    # no longer with the mean of the last season -- the same name-the-reference check, new reference
    text = rule_based_narrative(*_parts(fake_report))
    s = fake_report.analysis.series
    ref = nv.scenario_reference(fake_report.analysis, fake_report.forecast)
    assert ref.kind == "latest" and ref.value == float(s.iloc[-1])
    cur = ref.value
    when = format_date(s.index[-1], fake_report.analysis.freq)
    base = fake_report.outlook.scenarios["base"]
    assert f"괄호는 현재 값({when}) {format_number(cur)} 대비" in text
    assert f"기준 {format_number(base)} ({format_pct((base - cur) / cur * 100)})" in text
    en = rule_based_narrative(*_parts(fake_report), lang="en")
    assert f"% vs. the latest value ({when}), {format_number(cur, lang='en')}" in en
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
    # round 3 (UX-1): a +33 % change rated 'flat' only because of the forecast uncertainty is
    # "방향 불확실", no longer "보합" (= roughly unchanged), which contradicted the number
    assert "**+33.3%** 변화가 예상되지만 예측 불확실성이 그보다 커서 **방향은 불확실**합니다" in ko
    assert "변화(보합)" not in ko and "보합" not in ko
    assert "예측 불확실성이 이 변화보다 커서 **방향 불확실**, 예측 종료" in ko
    en = rule_based_narrative(*_parts(rep), lang="en")
    assert "uncertainty is larger than that, so the **direction is uncertain**" in en
    assert "the **direction is uncertain**; forecast to" in en
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
    # just below the ×10ⁿ bound stays fixed; the bound moved from 1e15 to 1e12 in round 4 (F8:
    # '18,698,199,572,300 → 460,927,814,133,102/월' was unreadable), so test its new edge
    assert format_number(999_999_999_999.0) == "999,999,999,999"
    assert format_number(18_698_199_572_300.0) == "1.87×10¹³"
    assert format_number(1e-9, decimals=2) == "0.00"           # explicit decimals are honoured
    t = np.arange(48.0)
    rep = _report_from_values(1e-9 * (1 + 0.01 * t), name="tiny")
    rep = _replace_metrics(rep, total_change_pct=47.0)
    text = rule_based_narrative(*_parts(rep))
    line = next(ln for ln in text.splitlines() if ln.startswith("- 추세선"))
    assert "0에서 0까지" not in line and "×10⁻⁹" in line
    assert "최고치는 2019-12의 1.47×10⁻⁹" in text
    # round 5 (V2-4): the scenario reference is the latest value (narrative.scenario_reference), not the
    # mean of the last season; the point of this check is unchanged -- a tiny base is not ~0
    cur = float(1e-9 * (1 + 0.01 * t[-1]))
    assert f"괄호는 현재 값(2019-12) {format_number(cur)} 대비" in text   # % shown: tiny base is not ~0
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
    # round 3 (UX-21): this trend line ends at its maximum, so it says so instead of
    # "고점 대비 0.0%로 여전히 고점 부근입니다" (a −x % trend gap keeps that wording, see
    # test_seasonal_gap_names_its_actual_cause)
    assert "추세선 기준으로는 지금이 고점입니다" in line and "상당히 내려와" not in line
    assert "0.0%" not in line
    en = rule_based_narrative(*_parts(rep), lang="en")
    assert "seasonal low" in en and "the level is at its high now" in en
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


# --------------------------------------------------------------------------- round-3 regressions
# Each test failed before the round-3 fix (AN1-AN8).


def _run(series, **kw):
    pipeline = pytest.importorskip("trendai.pipeline")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pipeline.run(series, **kw)


def _intermittent_series() -> pd.Series:
    rng = np.random.default_rng(3)
    idx = pd.date_range("2020-01-01", periods=60, freq="MS")
    return pd.Series(np.where(rng.random(60) < 0.25, rng.poisson(4, 60) + 1, 0).astype(float), index=idx)


def _level_shift_series() -> pd.Series:
    idx = pd.date_range("2021-01-03", periods=156, freq="W-SUN")
    y = 200 + np.random.default_rng(0).normal(0, 6, 156)
    y[100:] += 60
    return pd.Series(y, index=idx)


@pytest.mark.parametrize("ko,en", [
    ("현재 추세는 고점(2025-05) 수준입니다.", "The trend is at its peak (2025-05)."),
    ("현재 추세는 고점(2024년) 수준입니다.", "The trend is at its peak (2024)."),
    ("2050년에 구조 변화(가속)가 감지되었습니다.", "A structural break (acceleration) was detected in 2050."),
    ("2008년에 수준이 한 단계 높아졌습니다 (약 +31.9%).", "In 2008 the level stepped up (about +31.9%)."),
    ("고점(2016) 이후 하락해 2042년에 저점을 찍은 뒤 +68.0% 반등했습니다.",
     "After the peak (2016) it fell to a low in 2042, then rebounded +68.0%."),
    ("추세는 저점 대비 +95% 오른 뒤 2019년에 고점을 형성했습니다.",
     "The trend rose +95% from its low and peaked in 2019."),
])
def test_translate_reason_round3_templates(ko, en):
    # AN1 / AN2: new analysis wording ("추세 고점 수준", yearly "2016년에") stays translatable
    assert nv.translate_reason(ko) == en


def test_real_engine_yearly_and_peak_reasons_translate():
    analysis_mod = pytest.importorskip("trendai.analysis")
    cases = [(sample(nm, n=40, freq="Y"), "Y") for nm in ("emerging", "reviving")] + [(sample("growing"), "M")]
    idx = pd.date_range("1990-01-01", periods=30, freq="YS")
    y = 100 + np.random.default_rng(0).normal(0, 2, 30)
    y[18:] += 30
    cases.append((pd.Series(y, index=idx), "Y"))
    seen = []
    for s, freq in cases:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            a = analysis_mod.analyze(s, freq=freq)
        for r in a.phase.reasons:
            t = nv.translate_reason(r)
            assert t and "년" not in t and not HANGUL.search(t), r
            seen.append(r)
    joined = " ".join(seen)
    assert "년에" in joined and re.search(r"현재 추세는 고점\(\d{4}-\d{2}\) 수준입니다\.", joined)


@pytest.mark.parametrize("name", ["seasonal", "volatile"])
def test_noise_dominated_momentum_is_not_narrated_as_a_trend(name):
    # AN3: momentum of -0.12 %p/month on the stable seasonal sample was "최근에는 하락하고 있습니다"
    rep = _run(sample(name), name=name)
    assert rep.analysis.phase.label in ("stable", "volatile")
    for lang in ("ko", "en"):
        text = rule_based_narrative(*_parts(rep), lang=lang)
        line = next(ln for ln in text.splitlines() if ln.startswith(("- 모멘텀", "- Momentum")))
        for bad in ("최근에는 상승", "최근에는 하락", "(가속)", "(둔화)", "빠른 속도로", "둔화되고",
                    "rising recently", "falling recently", "(accelerating)", "(decelerating)", "is slowing"):
            assert bad not in line, (lang, line)
        assert ("잡음 수준" in line) if lang == "ko" else ("within the noise" in line)


def test_white_noise_and_directional_momentum_phrases():
    # AN3: pure noise reads as noise; genuine trends keep their momentum wording
    idx = pd.date_range("2015-01-01", periods=120, freq="MS")
    trendy = 0
    for seed in range(8):
        s = pd.Series(100 + np.random.default_rng(seed).normal(0, 5, 120), index=idx)
        pytest.importorskip("trendai.analysis")
        from trendai.analysis import analyze
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            a = analyze(s)
        f = ForecastResult(mean=pd.Series([100.0], index=future_index(idx[-1], "M", 1)), lower={}, upper={},
                           horizon=1, levels=(), model_scores=[], method="naive")
        o = Outlook(direction="flat", expected_change_pct=0.0, prob_increase=0.5, scenarios={},
                    forecast_peak_date=None, future_phase=a.phase.label, confidence="low")
        ko = nv._momentum_phrase(nv._Ctx(a, f, o, "ko"))[0]
        trendy += any(w in ko for w in ("(가속)", "최근에는 상승", "최근에는 하락", "(둔화)", "빠른 속도로", "둔화되고"))
    assert trendy == 0                                    # was 4 of 8
    texts = {nm: rule_based_narrative(*_parts(_run(sample(nm), name=nm))) for nm in ("growing", "fad", "peaking")}
    assert "장기 평균보다 빠른 속도로 성장하고 있습니다 (가속)" in texts["growing"]
    assert "장기 흐름과 달리 최근에는 하락하고 있습니다" in texts["fad"]
    assert "성장 속도가 장기 평균보다 느려지고 있습니다 (둔화)" in texts["peaking"]


def test_peak_sentence_names_the_excluded_spike():
    # AN4: metrics.peak_value leaves out flagged one-off spikes; the narrative must say so
    s = sample("growing").copy()
    s.iloc[60] += 150.0                                   # a one-month spike far above everything else
    rep = _run(s, name="spiky")
    m = rep.analysis.metrics
    assert m.peak_value < float(s.max())
    ko = rule_based_narrative(*_parts(rep))
    line = next(ln for ln in ko.splitlines() if ln.startswith("- 최고치는"))
    assert f"{format_number(m.peak_value)}(일회성 급등 제외)" in line
    assert f"{format_date(s.index[60], 'M')}의 일시적 급등({format_number(float(s.max()))})" in line
    en = rule_based_narrative(*_parts(rep), lang="en")
    line_en = next(ln for ln in en.splitlines() if ln.startswith("- The peak"))
    assert "(excluding one-off spikes)" in line_en and f"spike to {format_number(float(s.max()), lang='en')}" in line_en
    # no spike above the peak: plain sentence
    plain = rule_based_narrative(*_parts(_run(sample("growing"), name="g")))
    assert "일회성 급등 제외" not in plain


def test_rise_from_a_level_shift_is_not_called_steady_growth():
    # AN5: stable after a one-off step, but whole-period MK 'up' and CAGR > 0
    rep = _run(_level_shift_series(), name="step")
    a = rep.analysis
    assert a.phase.label == "stable" and a.metrics.direction == "up" and a.metrics.cagr_pct > 0
    ko = rule_based_narrative(*_parts(rep))
    past = ko.split("## 현재")[0]
    assert "다만 이 상승은 꾸준한 추세가 아니라 **2022-12-04**의 한 차례 수준 변화" in past
    assert "꾸준한 성장으로 해석하면 안 됩니다" in past
    mom = next(ln for ln in ko.splitlines() if ln.startswith("- 모멘텀"))
    assert "한 차례의 수준 변화" in mom and "최근에는 하락하고" not in mom
    en = rule_based_narrative(*_parts(rep), lang="en")
    assert "came from a one-off step on **2022-12-04**" in en and "do not mean steady growth" in en
    assert "On **2022-12-04** the level jumped up" in en          # a day takes "On", not "In"
    # a step down reads the same way ("감소로", not "감소으로")
    down = _level_shift_series()
    down.iloc[100:] -= 120.0
    ko_d = rule_based_narrative(*_parts(_run(down, name="drop")))
    assert "다만 이 하락은 꾸준한 추세가 아니라" in ko_d and "꾸준한 감소로 해석하면 안 됩니다" in ko_d
    # a genuinely growing series gets no such caveat
    assert "한 차례 수준 변화" not in rule_based_narrative(*_parts(_run(sample("growing"), name="g")))


def test_probability_describes_the_same_levels_as_the_expected_change():
    # AN6: prob_increase = P(future season-mean > current season-mean), not P(end value > now)
    for rep in (_run(sample("seasonal"), name="m"), _run(_level_shift_series(), name="w")):
        ko = rule_based_narrative(*_parts(rep))
        lines = ko.splitlines()
        i = next(k for k, ln in enumerate(lines) if "변할 것으로 예상됩니다" in ln)
        chg, prob, scen = lines[i], lines[i + 1], lines[i + 2]
        assert "높아질 확률은" in prob and "시점 시나리오" in scen           # change -> probability -> scenarios
        fut, cur = re.match(r"- (.+?)은 (.+?)보다 \*\*", chg).groups()
        assert prob.startswith(f"- {fut}이 {cur}보다 높아질 확률은")
        assert "현재 수준보다 높아질 확률" not in ko
        assert "변화율·확률과 다를 수 있습니다" in scen                      # end values: another quantity
        en = rule_based_narrative(*_parts(rep), lang="en")
        assert "ending above the current level" not in en
        assert re.search(r"- Probability that the (seasonally adjusted )?average forecast over the next \d+ \w+ "
                         r"is higher than the average of the last \d+ \w+: ", en)
        payload = build_llm_payload(*_parts(rep))
        basis = payload["outlook"]["prob_increase_basis"]
        assert "same two levels as expected_change_pct" in basis and "NOT the probability" in basis
    assert "never as the chance that the value at the horizon" in SYSTEM_PROMPT.replace("\n", " ")


def test_undefined_change_and_fallback_forecast_make_no_claims():
    # AN7: intermittent demand -> expected_change_pct None; n = 4 yearly -> fallback (no scores, no skill)
    inter = _run(_intermittent_series(), name="parts")
    assert inter.outlook.expected_change_pct is None
    tiny = _run(pd.Series([10.0, 12.0, 15.0, 14.0], index=pd.date_range("2020", periods=4, freq="YS")),
                name="tiny")
    assert tiny.forecast.model_scores == [] and tiny.forecast.skill is None
    for rep in (inter, tiny):
        for lang in ("ko", "en"):
            text = rule_based_narrative(*_parts(rep), lang=lang)
            assert not _BAD_TOKEN.search(text.replace(rep.name, ""))
        payload = build_llm_payload(*_parts(rep))
        json.dumps(payload, allow_nan=False)
    ko = rule_based_narrative(*_parts(inter))
    future = ko.split("## 미래 전망")[1].split("## 주의할 점")[0]
    first, scen = future.splitlines()[1], next(ln for ln in future.splitlines() if "시점 시나리오" in ln)
    assert "최근 12개월 중 절반 이상이 0이라 변화율은 계산하지 않음" in first and "0에 가까워" not in first
    assert "%" not in first and "%" not in scen.split("95%")[0].replace("80% 예측구간", "")
    en = rule_based_narrative(*_parts(inter), lang="en")
    assert "at least half of the last 12 months are zero" in en
    p_out = build_llm_payload(*_parts(inter))["outlook"]
    assert p_out["expected_change_pct"] is None and "null here because at least half" in p_out["expected_change_basis"]
    assert p_out["compared_levels"]["current"] == pytest.approx(0.5)
    ko_t = rule_based_narrative(*_parts(tiny))
    assert "백테스트로 정확도를 검증하지 못했습니다" in ko_t and "백테스트 상위 모델" not in ko_t
    assert "단순 기준" not in ko_t.split("- 정확도:")[1].splitlines()[0]
    pt = build_llm_payload(*_parts(tiny))
    assert pt["model_leaderboard"] == [] and pt["forecast"]["skill"] is None


@pytest.mark.parametrize("name,freq,n", [
    ("seasonal", "M", 120), ("growing", "W", 156), ("fad", "Q", 40), ("emerging", "Y", 25),
    ("seasonal", "D", 400), ("reviving", "M", 30), ("growing", "W", 60),
])
def test_outlook_windows_reproduce_compute_outlook(name, freq, n):
    # AN8: the narrative's description of the compared levels is exactly compute_outlook's
    forecast_mod = pytest.importorskip("trendai.forecast")
    rep = _run(sample(name, n=n, freq=freq), name=name, freq=freq)
    a, f, o = _parts(rep)
    w = nv._outlook_windows(a, f)
    m = a.season_length
    assert (w.k_cur, w.k_fut) == (min(m, n), min(m, len(f.mean)))
    assert w.current is not None and w.future is not None
    if o.expected_change_pct is None:
        assert w.undefined is not None
    else:
        assert w.undefined is None
        assert (w.future - w.current) / abs(w.current) * 100 == pytest.approx(o.expected_change_pct, rel=1e-9,
                                                                             abs=1e-9)
    # the adjustment flags agree with compute_outlook's own seasonal component
    seas, p = forecast_mod._seasonal_component(a)
    assert w.adj_fut == (seas is not None and w.k_fut % p != 0)
    assert w.adj_cur == (seas is not None and w.k_cur % p != 0)
    if w.future != w.current:                             # same levels -> same sign as prob_increase
        assert (w.future > w.current) == (o.prob_increase > 0.5)
    payload = build_llm_payload(a, f, o)["outlook"]["compared_levels"]
    assert payload == {"current": nv._r(w.current), "future": nv._r(w.future)}


def test_outlook_windows_none_rule_matches_compute_outlook():
    # AN8: the 'mostly zero' rule of compute_outlook (>= half of the current window exactly 0)
    rep = _run(_intermittent_series(), name="parts")
    w = nv._outlook_windows(*_parts(rep)[:2])
    assert rep.outlook.expected_change_pct is None and w.undefined == "mostly_zero"
    zeros = _run(pd.Series(np.r_[np.arange(1.0, 49.0), np.zeros(12)],
                           index=pd.date_range("2018-01-01", periods=60, freq="MS")), name="z")
    wz = nv._outlook_windows(*_parts(zeros)[:2])
    assert zeros.outlook.expected_change_pct is None and wz.undefined == "zero"
    # round 5 (G3): a current window of exact zeros reads "현재 수준이 0이라" (was "0에 가까워", which
    # stays for a level that is only ~0)
    assert "현재 수준이 0이라 변화율은 계산하지 않음" in rule_based_narrative(*_parts(zeros))


# --------------------------------------------------------------------------- round 3 (final review)


def _sample_run(name: str, freq: str = "M", n=None):
    kw = {"freq": freq} if n is None else {"freq": freq, "n": n}
    return _run(sample(name, **kw), name=name, freq=freq)


def test_large_flat_change_is_uncertain_not_boap():
    # UX-1: '+75.3% 변화가 예상되지만 … 방향은 보합' -- 보합 means roughly unchanged
    assert nv.direction_label("flat", 75.3) == "방향 불확실"
    assert nv.direction_label("flat", -2.1, "en") == "direction uncertain"
    assert nv.direction_label("flat", 1.0) == "보합" and nv.direction_label("flat", None) == "보합"
    assert nv.direction_label("up", 30.0) == "상승" and nv.direction_label("down", -9.0, "en") == "downward"
    rep = _sample_run("emerging", "Y", 12)
    assert rep.outlook.direction == "flat" and abs(rep.outlook.expected_change_pct) >= 2
    ko = rep.narrative
    assert "보합" not in ko and "방향은 불확실" in ko.splitlines()[0]
    assert build_llm_payload(*_parts(rep))["outlook"]["direction_label"]["ko"] == "방향 불확실"
    assert "direction_label" in SYSTEM_PROMPT


def test_latest_value_is_not_called_a_one_off_spike():
    # UX-5: '최고치는 2026년의 42.3(일회성 급등 제외)이며, 현재가 사실상 역대 최고 수준입니다.
    # 2027년의 일시적 급등(72.0)은 이상치로 보아 제외했습니다.' -- 72.0 is the latest value
    rep = _sample_run("emerging", "Y", 12)
    ko = rep.narrative
    last_year = f"{rep.analysis.series.index[-1].year}년"           # the latest observation
    peak = next(ln for ln in ko.splitlines() if ln.startswith("- 최고치는"))
    assert "일회성 급등 제외" not in peak and "일시적 급등(72.0)" not in peak
    assert f"{last_year}의 72.0" in peak and "새로운 수준인지는 다음 관측으로 확인해야 합니다" in peak
    an = next(ln for ln in ko.splitlines() if ln.strip().startswith(f"- {last_year}: 실제 72.0"))
    assert "가장 최근 값이라 일시적 급등인지 새로운 수준인지는 아직 알 수 없습니다" in an
    en = rule_based_narrative(*_parts(rep), lang="en")
    assert "one-off or a new level" in en and "treated as an anomaly and left out" not in en


def test_seasonal_gap_names_its_actual_cause():
    # UX-7: the daily sample's last day is a Friday with seasonal effect +4.2 (above trend) but
    # was described as "최근 시점이 계절적으로 낮은 시기"; the gap came from a Saturday peak (+7.9)
    rep = _sample_run("seasonal", "D", 400)
    dec, m = rep.analysis.decomposition, rep.analysis.metrics
    assert float(dec.seasonal.iloc[-1]) > 0
    line = next(ln for ln in rep.narrative.splitlines() if ln.startswith("- 최고치는"))
    assert "최근 시점이 계절적으로 낮은 시기" not in line
    assert f"고점이 계절적으로 높은 시기({nv._WEEKDAY_KO[m.peak_date.dayofweek]})에 찍힌 값이어서" in line
    # a trend gap of a few % keeps "여전히 고점 부근" (only an exact high reads "지금이 고점")
    assert re.search(r"고점 대비 −\d\.\d%로 여전히 고점 부근입니다", line), line
    en = rule_based_narrative(*_parts(rep), lang="en")
    assert "the latest period falls in a seasonal low" not in en and "the peak fell in a seasonal high" in en
    # the latest point in a seasonal low is still said to be one
    low = _seasonal_low_report()
    ko = rule_based_narrative(*_parts(low))
    assert "최근 값은 계절적으로 낮은 시기(12월)에" in ko


def test_insignificant_mk_with_large_tau_is_explained():
    # UX-10: "유의한 장기 추세는 확인되지 않습니다 (τ = 0.70, p = 0.140)" right after "+1,683 %"
    rep = _sample_run("emerging", "Y", 12)
    line = next(ln for ln in rep.narrative.splitlines() if ln.startswith("- Mann-Kendall"))
    assert "자기상관 보정 p = " in line
    assert "관측치가 12개로 적고 값의 움직임이 서로 이어져 있어(자기상관) 통계적으로는 확정하기 어렵습니다" in line
    en = rule_based_narrative(*_parts(rep), lang="en")
    assert "after correcting for autocorrelation" in en and "cannot be confirmed statistically" in en
    # a significant trend or a small tau gets no such clause
    for nm in ("growing", "volatile"):
        assert "확정하기 어렵습니다" not in _sample_run(nm).narrative
    assert nv._mk_classic_p(0.70, 12) < 0.01 < nv._mk_classic_p(0.2, 12)


def test_method_is_described_in_plain_words():
    # UX-11: 'ensemble(holt_damped 0.34, ar 0.33, holt_damped@log 0.33)' and bare
    # 'logistic_growth' in Korean prose
    rep = _sample_run("fad")
    line = next(ln for ln in rep.narrative.splitlines() if ln.startswith("- 예측 방법:"))
    prose = line.split("(모델 ID:")[0]
    assert not re.search(r"[a-z]+_[a-z_@]+|ensemble\(", prose), line
    assert "3개 모델 앙상블(감쇠 추세 Holt" in prose and "자기회귀(AR)" in prose and "(로그 공간)" in prose
    assert "(모델 ID: holt_damped, " in line
    single = next(ln for ln in _sample_run("emerging").narrative.splitlines() if ln.startswith("- 예측 방법:"))
    assert single.startswith("- 예측 방법: 로지스틱 S-커브 단일 모델")
    assert nv.model_label("holt_damped@log") == "감쇠 추세 Holt(로그 공간)"
    assert nv.model_label("holt_winters@log", "en") == "Holt-Winters seasonal (log scale)"
    assert nv.model_label("my_model") == "my_model"
    fb = ForecastResult(mean=pd.Series([1.0], index=pd.date_range("2030", periods=1, freq="YS")), lower={},
                        upper={}, horizon=1, levels=(), model_scores=[],
                        method="fallback(drift 0.50, naive 0.50) [too short to backtest]")
    assert nv.describe_method(fb) == ("데이터가 짧아 백테스트 없이 만든 기본 예측"
                                      "(드리프트(평균 기울기) · 단순 예측(직전값) 동일 비중)")
    # members are the weighted models, not the first MASE rows (intermittent: chosen by RMSE)
    scores = [ModelScore("naive", 1, 1, 1, 0.5, 0.0), ModelScore("sba", 1, 1, 1, 0.9, 0.6),
              ModelScore("ses", 1, 1, 1, 0.95, 0.4)]
    ens = dataclasses.replace(fb, model_scores=scores, method="ensemble(sba 0.60, ses 0.40)")
    assert nv.describe_method(ens) == "2개 모델 앙상블(간헐 수요 Croston(SBA 보정) 60% · 단순 지수평활 40%)"
    payload = build_llm_payload(*_parts(rep))
    assert payload["forecast"]["method_description"]["ko"].startswith("3개 모델 앙상블")


def test_rising_phase_with_a_non_rising_forecast_is_reconciled():
    # UX-19: reviving Q/40 ('최근에는 상승하고 있습니다', +30.4 % in 4 quarters) with a forecast
    # that drops at once and goes flat -- nothing told the reader the models do not extend it
    rep = _sample_run("reviving", "Q", 40)
    assert rep.analysis.phase.label == "reviving" and rep.outlook.expected_change_pct < 2
    ko = rep.narrative
    future = ko.split("## 미래 전망")[1].split("## 주의할 점")[0]
    assert "최근 반등세가 뚜렷하지만, 백테스트로 고른 예측 모델은 이 흐름이 이어진다고 보지 않고" in future
    assert "현재 수준 부근에 머물 것으로 예측합니다" in future       # −2.1 %: not "오히려 낮아질"
    assert "do not extend it" in rule_based_narrative(*_parts(rep), lang="en")
    # a forecast that does continue the move gets no such sentence
    assert "이 흐름이 이어진다고 보지 않고" not in _sample_run("growing").narrative


def test_round3_wording_nits():
    # UX-21: "보통 수준입니다 … 보통 3월에 높고", "expected to be −63.3% versus",
    # "상당히 내려와" for a noise peak in a volatile series
    peaking = _sample_run("peaking").narrative
    assert "보통 3월에" not in peaking and "대체로 3월에 높고 10월에 낮은 패턴이 반복됩니다" in peaking
    assert "고점 대비 0.0%" not in peaking and "추세선 기준으로는 지금이 고점입니다" in peaking
    en = rule_based_narrative(*_parts(_sample_run("fad")), lang="en")
    assert "is expected to be" not in en and "is expected to change by **−" in en
    vol = _sample_run("volatile")
    line = next(ln for ln in vol.narrative.splitlines() if ln.startswith("- 최고치는"))
    assert "상당히 내려와" not in line and "최고치보다" in line and "일시적 등락" in line


def test_weak_seasonality_names_no_months():
    # UX-6: '계절성은 약한 편입니다 (강도 0.34). 보통 8월에 높고 9월에 낮은 패턴이 반복됩니다'
    # on trend + white noise
    idx = pd.date_range("2021-01-01", periods=36, freq="MS")
    hits = 0
    for seed in range(12):
        y = 1200 + 25 * np.arange(36) + np.random.default_rng(seed).normal(0, 80, 36)
        hits += "패턴이 반복됩니다" in _run(pd.Series(y, index=idx), name="x", freq="M").narrative
    assert hits == 0
    rep = _report_from_values(100 + np.random.default_rng(0).normal(0, 1, 36))
    rep = _replace_metrics(rep, seasonal_strength=0.25)
    rep = dataclasses.replace(rep, analysis=dataclasses.replace(rep.analysis, decomposition=dataclasses.replace(
        rep.analysis.decomposition, period=12)))
    assert "뚜렷한 반복 패턴은 확인되지 않습니다" in rule_based_narrative(*_parts(rep))


def test_intermittent_demand_narrative_makes_no_percent_or_drawdown_claims():
    # F2: '+3,192.4% … CAGR +239.9%', '최근 52주 동안 추세는 +231.3%', '고점의 1/10 이하로 급감',
    # and ordinary orders listed as '+1,349.8%, z = +12.4' anomalies
    rng = np.random.default_rng(9)
    y = np.where(rng.random(150) < 0.15, rng.integers(1, 6, 150), 0).astype(float)
    rep = _run(pd.Series(y, index=pd.date_range("2021-01-03", periods=150, freq="W-SUN")), name="주간 판매량")
    ko = rep.narrative
    past_now = ko.split("## 미래 전망")[0]
    assert "급감" not in ko and "CAGR" not in ko and not re.search(r"[+−]\d[\d,]*\.\d%", past_now)
    # round 5 (V2-5): the demand is described by its first / last 52-week average demand, no longer by
    # the trend line's ends ("평균 수요율 A → B" quoted trend end points that sit at 0 for mostly-zero data)
    assert "간헐적 수요여서 추세선의 변화율(%)과 연평균 성장률은 의미가 없어" in past_now
    assert "기간당 평균 수요량은 처음 52주" in past_now and "고점 대비 하락률은 의미가 없습니다" in past_now
    assert "최근 52주 변화율은 계산하지 않았습니다" in past_now
    payload = build_llm_payload(*_parts(rep))
    assert payload["series"]["intermittent"] is True and "intermittent" in payload["conventions"]
    # a large order is told as such, not as "+1,350 %"
    s = pd.Series(y, index=pd.date_range("2021-01-03", periods=150, freq="W-SUN"))
    s.iloc[int(np.flatnonzero(y > 0)[10])] = 60.0
    ko2 = _run(s, name="x").narrative
    assert "평소보다 훨씬 큰 수요가 1건 있었습니다:" in ko2 and "평소 수요 규모(" in ko2
    # an expected value near 0 gets no percentage in an ordinary anomaly line
    an = Anomaly(date=pd.Timestamp("2018-07-01"), index=30, value=4.0, expected=0.01, zscore=9.0)
    c = nv._Ctx(_report_from_values(np.r_[np.zeros(30), 4.0, np.zeros(5)] + 1.0).analysis, *_parts(
        _report_from_values(np.ones(36)))[1:], "ko")
    assert "%" not in nv._anomaly_line(c, an)


def test_momentum_line_names_its_basis():
    # UX-8: '+15.73 %p/월' read as acceleration against the whole-period mean
    rep = _sample_run("emerging")
    line = next(ln for ln in rep.narrative.splitlines() if ln.startswith("- 모멘텀"))
    assert "현재 추세 수준 대비" in line
    mom = float(re.search(r"([+−][\d.]+)%p/월", line).group(1).replace("−", "-"))
    assert 0 < mom < 6                                     # was +15.73 while the trend grew ~4 %/month
    assert "recent (current) trend level" in build_llm_payload(*_parts(rep))["conventions"]["momentum"]


@pytest.mark.parametrize("ko,en", [
    ("관측 기간 동안 값이 3.00에서 6.00까지 +100.0% 변했습니다.",
     "Over the observed period the value moved from 3.00 to 6.00 (+100.0%)."),
    ("계절성 강도(0.96)가 높아 주기적 변동이 뚜렷합니다.", "Seasonal strength is 0.96: a clear recurring cycle."),
    ("잡음이 커서(추정 잡음 수준이 값의 변동 폭의 61%) 뚜렷한 방향성이 없습니다.",
     "Noise is large (estimated at 61% of the typical spread of the values), so there is no clear direction."),
    ("잡음이 커서(추정 잡음 수준이 평균의 1,250%) 방향성이 불분명합니다.",
     "Noise is large (estimated at 1,250% of the mean), so there is no clear direction."),
    ("추세선의 등락이 잡음으로 생길 수 있는 범위를 크게 넘지 않습니다.",
     "The trend's ups and downs stay within what noise alone could produce."),
    ("최근 4년 동안 추세가 +1,076.8% 변했습니다.", "Over the last 4 years the trend changed by +1,076.8%."),
    ("2022-09에 수준이 한 단계 높아졌습니다 (약 +2,392,181).", "In 2022-09 the level stepped up (about +2,392,181)."),
    ("최근 6개월 동안 추세 기울기는 +1.2×10⁻⁷/개월입니다.",
     "Over the last 6 months the trend slope was +1.2×10⁻⁷ per month."),
])
def test_translate_round3_reasons(ko, en):
    assert nv.translate_reason(ko) == en


def test_round3_reasons_from_real_analyses_all_translate():
    # every reason the analysis now produces has an English template
    cases = (
        sample("seasonal"), sample("volatile"), sample("emerging", freq="Y", n=12),
        pd.Series(np.random.default_rng(0).normal(0, 1, 120), index=pd.date_range("2015-01-01", periods=120, freq="MS")),
        pd.Series(np.r_[np.zeros(19), 1.0], index=pd.date_range("2020-01-01", periods=20, freq="MS")),
        pd.Series(np.r_[np.zeros(20), 2_400_000 + np.random.default_rng(0).normal(0, 60_000, 28)].round(0),
                  index=pd.date_range("2021-01-01", periods=48, freq="MS")),
        pd.Series([3.0, 4.0, 4.5, 5.0, 6.0], index=pd.date_range("2020-01-01", periods=5, freq="MS")),
    )
    from trendai.analysis import analyze
    for s in cases:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            a = analyze(s)
        for r in a.phase.reasons:
            assert nv.translate_reason(r), r


# --------------------------------------------------------------------------- round 4


def _mult_growth(n: int = 120, seed: int = 1) -> pd.Series:
    """Monthly multiplicative seasonality on exponential growth (analysis decomposes it in logs)."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    y = 50 * np.exp(0.02 * t) * (1 + 0.35 * np.sin(2 * np.pi * (t + 10) / 12)) * np.exp(0.03 * rng.standard_normal(n))
    return pd.Series(y, index=pd.date_range("2020-01-05", periods=n, freq="MS"))


def _pnl_cross(seed: int) -> pd.Series:
    """A P&L that rose from -20 to +20 and is falling towards 0 (the forecast crosses zero)."""
    rng = np.random.default_rng(seed)
    y = np.r_[np.linspace(-20, 20, 24), np.linspace(20, 0.5, 24)] + rng.normal(0, 1, 48)
    return pd.Series(y, index=pd.date_range("2020-01-01", periods=48, freq="MS"))


@pytest.mark.parametrize("label,series,freq,horizon", [
    ("mult-M-h6", "mult", "M", 6),             # multiplicative + partial-season forecast window
    ("mult-M-h18", "mult", "M", 18),
    ("mult-M-h12", "mult", "M", 12),
    ("seasonal-W-h26", "seasonal", "W", 26),   # 26 of 52 weeks: both windows partial
    ("seasonal-W-h8", "seasonal", "W", 8),
    ("growing-M-h5", "growing", "M", 5),
    ("seasonal-D-h10", "seasonal", "D", 10),
    ("fad-Q-h3", "fad", "Q", 3),
    ("emerging-Y-h4", "emerging", "Y", 4),
    ("reviving-M-h7", "reviving", "M", 7),
    ("pnl-M-h12", "pnl", "M", 12),
])
def test_narrative_levels_are_outlook_levels(label, series, freq, horizon):
    # U1: narrative._outlook_windows mirrored compute_outlook but continued the seasonal pattern
    # without the forecast level, so for a multiplicative decomposition with a partial-season
    # window the narrative and payload printed 576.4 (+19.0 %) next to expected_change_pct
    # +17.5 % (outlook_levels 569.09).  forecast.outlook_levels is now the single source.
    from trendai.forecast import outlook_levels
    if series == "mult":
        s = _mult_growth()
    elif series == "pnl":
        s = _pnl_cross(1)
    else:
        s = sample(series, freq=freq, **({"n": 156} if freq == "W" else {}))
    rep = _run(s, name=label, freq=freq, horizon=horizon)
    a, f, o = _parts(rep)
    lv = outlook_levels(a, f)
    w = nv._outlook_windows(a, f)
    assert (w.current, w.future, w.k_cur, w.k_fut, w.period, w.adj_cur, w.adj_fut, w.undefined) == (
        lv.current, lv.future, lv.k_cur, lv.k_fut, lv.period, lv.adj_cur, lv.adj_fut, lv.undefined)
    p_out = build_llm_payload(a, f, o)["outlook"]
    assert p_out["compared_levels"] == {"current": nv._r(lv.current), "future": nv._r(lv.future)}
    assert p_out["expected_change_undefined"] == lv.undefined
    if lv.undefined is None:
        implied = (lv.future - lv.current) / abs(lv.current) * 100
        assert implied == pytest.approx(o.expected_change_pct, rel=1e-9, abs=1e-9)
        assert format_pct(o.expected_change_pct) in rep.narrative
    else:
        # the narrative prints the two levels themselves
        assert format_number(lv.future, signed=lv.undefined in ("sign_change", "near_zero")) in rep.narrative
    if series == "mult" and horizon == 6:
        from trendai.analysis import _is_multiplicative
        assert _is_multiplicative(a.decomposition) and lv.adj_fut


def test_sign_change_and_near_zero_outlooks_have_their_own_wording():
    # U2: a level expected to cross zero (or a current level near 0 among negative values) was
    # explained as '현재 수준이 0에 가까워' -- wrong for +4.7 -> -3.9
    from trendai.forecast import outlook_levels
    for seed, word_ko, word_en in ((1, "**하락**", "**downward**"), (2, "**방향 불확실**", "**direction uncertain**")):
        rep = _run(_pnl_cross(seed), name="pnl", horizon=12)
        lv = outlook_levels(*_parts(rep)[:2])
        assert lv.undefined == "sign_change" and lv.current > 0 > lv.future
        assert rep.outlook.expected_change_pct is None
        ko = rule_based_narrative(*_parts(rep))
        first = ko.split("## 미래 전망")[1].splitlines()[1]
        cur, fut = format_number(lv.current, signed=True), format_number(lv.future, signed=True)
        assert f"전망은 {word_ko}입니다" in first and "0을 지나 음수로 바뀔 것으로" in first
        assert f"{cur} → 향후 12개월 예측 평균 {fut}" in first and "0에 가까워" not in first
        assert "%" not in first.replace("변화율(%)", "")
        en = rule_based_narrative(*_parts(rep), lang="en")
        first_en = en.split("## Outlook")[1].splitlines()[1]
        assert f"is {word_en}" in first_en and "expected to cross zero" in first_en
        assert f"from {cur} (the average of the last 12 months) to {fut}" in first_en
        assert not HANGUL.search(en.replace("pnl", ""))
        p = build_llm_payload(*_parts(rep))["outlook"]
        assert p["expected_change_undefined"] == "sign_change" and "cross zero" in p["expected_change_basis"]
        assert p["direction_label"]["ko"] == word_ko.strip("*")
    # 'near_zero': data with negatives, current level within half an SD of 0, same sign ahead
    idx = pd.date_range("2020-01-01", periods=60, freq="MS")
    rng = np.random.default_rng(8)
    y = 8 * np.sin(np.arange(60) / 5 + 8) + rng.normal(0, 1, 60) + 1.0
    rep = _run(pd.Series(y, index=idx), name="temp", horizon=6)
    lv = outlook_levels(*_parts(rep)[:2])
    assert lv.undefined == "near_zero"
    ko = rule_based_narrative(*_parts(rep))
    first = ko.split("## 미래 전망")[1].splitlines()[1]
    assert f"({format_number(lv.current, signed=True)})이 값의 평소 변동 폭(표준편차" in first
    assert "0에 가까워 변화율(%)은 의미가 없습니다" in first and "0을 지나" not in first
    en = rule_based_narrative(*_parts(rep), lang="en")
    assert "close to zero relative to the typical spread of the values" in en
    p = build_llm_payload(*_parts(rep))["outlook"]
    assert p["expected_change_undefined"] == "near_zero" and "half a standard deviation" in p["expected_change_basis"]
    assert nv.direction_label("flat", None, "ko", "sign_change") == "방향 불확실"
    assert nv.direction_label("flat", None, "ko", "near_zero") == "보합"


def test_zero_crossing_history_gets_no_percentages_in_the_narrative():
    # U3: P&L 10 -> -10: '추세선 기준으로 10.4에서 −8.69까지 **−183.6%** 변했습니다' and
    # '최근 값(−8.04)은 고점보다 179.4% 낮아 고점의 1/10 이하로 급감했습니다'
    rng = np.random.default_rng(0)
    s = pd.Series(np.linspace(10, -10, 48) + rng.normal(0, 1.0, 48),
                  index=pd.date_range("2022-01-01", periods=48, freq="MS"))
    rep = _run(s, name="P&L", horizon=12)
    m = rep.analysis.metrics
    assert m.total_change_pct is None and m.drawdown_pct < -100
    ko = rule_based_narrative(*_parts(rep))
    past = ko.split("## 과거: 어떻게 여기까지 왔나")[1].split("## 현재")[0]
    trend_line = next(ln for ln in past.splitlines() if ln.startswith("- 추세선"))
    assert "그사이 0을 지나 음수로 바뀌어 변화율(%)은 정의되지 않습니다" in trend_line
    assert "0에 가까워" not in trend_line
    peak_line = next(ln for ln in past.splitlines() if ln.startswith("- 최고치"))
    gap = format_number(m.peak_value - m.last_value)
    assert f"고점보다 {gap} 낮습니다" in peak_line and "하락률(%)은 의미가 없습니다" in peak_line
    assert not re.search(r"\d+(\.\d)?% 낮", ko) and "급감" not in ko
    en = rule_based_narrative(*_parts(rep), lang="en")
    assert "undefined because the level crossed zero" in en
    assert f"is {format_number(m.peak_value - m.last_value, lang='en')} below it. The value has crossed zero" in en
    assert not HANGUL.search(en.replace("P&L", ""))
    metrics = build_llm_payload(*_parts(rep))["metrics"]
    assert metrics["drawdown_pct"] is None and metrics["total_change_pct"] is None
    assert metrics["pct_undefined"]["total_change_pct"] == "sign_change"
    assert metrics["pct_undefined"]["drawdown_pct"] == "sign_change"
    # the trend-based drawdown (seasonal rule, report tile) follows the same rule
    assert nv._trend_drawdown(rep.analysis) is None
    # ratio-scale samples keep their drawdown wording
    dec = _sample_run("declining")
    assert nv._drawdown_undefined(dec.analysis) is None
    assert "고점보다" in dec.narrative and "하락률(%)은 의미가 없습니다" not in dec.narrative


def _with_outlook(rep: TrendReport, **kw) -> TrendReport:
    return dataclasses.replace(rep, outlook=dataclasses.replace(rep.outlook, **kw))


def _future(text: str, lang: str = "ko") -> str:
    head, tail = ("## 미래 전망", "## 주의할 점") if lang == "ko" else ("## Outlook", "## Caveats")
    return text.split(head)[1].split(tail)[0]


def test_flat_outlook_keeping_a_moving_phase_is_not_a_projection():
    # U4: future_phase keeps the current phase when a flat forecast is uninformative, but the
    # narrative said '12개월 뒤에도 성장기 국면이 이어질 것으로 전망됩니다'
    base = _sample_run("growing")
    assert base.analysis.phase.label == "growing"
    rep = _with_outlook(base, direction="flat", expected_change_pct=30.0, future_phase="growing",
                        confidence="medium")
    fut = _future(rule_based_narrative(*_parts(rep)))
    assert "이어질 것으로 전망됩니다" not in fut
    assert "현재 **성장기** 국면이 12개월 뒤에도 이어질지 바뀔지는 아직 판단하기 어렵습니다" in fut
    en = _future(rule_based_narrative(*_parts(rep), lang="en"), "en")
    assert "still **Growing**" not in en and "is hard to judge yet" in en
    # with the reconciliation line (a small change against a rising phase): one message, not two
    rep2 = _with_outlook(rep, expected_change_pct=0.5)
    fut2 = _future(rule_based_narrative(*_parts(rep2)))
    assert fut2.count("판단하기 어려") == 1 and fut2.count("다음 관측치") == 1
    assert "최근 상승세가 뚜렷하지만" in fut2 and "이어질 것으로 전망됩니다" not in fut2
    assert "현재 **성장기** 국면이 이어질지는 아직 판단하기 어려우니" in fut2
    # peaking and declining are moving phases too
    for nm, ko_name in (("peaking", "정점"), ("declining", "쇠퇴기")):
        r = _sample_run(nm)
        lab = r.analysis.phase.label
        r = _with_outlook(r, direction="flat", expected_change_pct=-15.0, future_phase=lab, confidence="medium")
        f3 = _future(rule_based_narrative(*_parts(r)))
        assert "이어질 것으로 전망됩니다" not in f3 and "판단하기 어렵" in f3
    # a directional outlook keeps the projection; stable/volatile phases too
    up = _with_outlook(base, direction="up", expected_change_pct=12.0, future_phase="growing", confidence="medium")
    assert "12개월 뒤에도 **성장기** 국면이 이어질 것으로 전망됩니다" in rule_based_narrative(*_parts(up))
    p = build_llm_payload(*_parts(rep))["outlook"]
    assert "hard to judge" in p["future_phase_basis"]
    assert build_llm_payload(*_parts(up))["outlook"]["future_phase_basis"] is None


def test_low_confidence_phase_change_is_not_called_likely():
    # U4: '…국면으로 넘어갈 가능성이 있습니다' under a low-confidence forecast
    base = _sample_run("growing")
    rep = _with_outlook(base, direction="down", expected_change_pct=-20.0, future_phase="declining",
                        confidence="low")
    fut = _future(rule_based_narrative(*_parts(rep)))
    assert "가능성이 있습니다" not in fut
    assert "예측대로라면 12개월 안에 **성장기**에서 **쇠퇴기** 국면으로 넘어가지만, 예측 신뢰도가 낮아" in fut
    en = _future(rule_based_narrative(*_parts(rep), lang="en"), "en")
    assert "Projected phase shift" not in en and "with low forecast confidence" in en
    assert "never as a likely" in build_llm_payload(*_parts(rep))["outlook"]["future_phase_basis"]
    sure = _with_outlook(rep, confidence="medium")
    assert "국면으로 넘어갈 가능성이 있습니다" in rule_based_narrative(*_parts(sure))


def test_scenario_bound_at_the_zero_floor():
    # U5: a nonnegative forecast clipped at 0 printed '비관 0 (−100.0%)'
    assert nv.format_scenario(0.0, 7.5, True) == "0(하한)"
    assert nv.format_scenario(0.0, 7.5, True, "en") == "0 (floor)"
    assert nv.scenario_value(0.0, 7.5, nonnegative=True) == ("0(하한)", None)
    assert nv.scenario_value(-1e-12, 7.5, True, "en") == ("0 (floor)", None)
    assert nv.format_scenario(9.0, 7.5, True) == "9.00 (+20.0%)"
    assert nv.format_scenario(0.0, 7.5, False) == "0 (−100.0%)"     # not clipped: a real zero
    assert nv.format_scenario(3.0, None, True) == "3.00" and nv.format_scenario(3.0, 0.0) == "3.00"
    assert nv.scenario_value(None) == ("해당 없음", None) and nv.scenario_value(float("nan"), 1, lang="en")[0] == "n/a"
    # end to end: a series falling towards 0 (nonnegative forecast, bound clipped at 0)
    rng = np.random.default_rng(0)
    s = pd.Series(np.linspace(30, 4, 48) + rng.normal(0, 0.8, 48),
                  index=pd.date_range("2020-01-01", periods=48, freq="MS"))
    rep = _run(s, name="parts", horizon=12)
    f, o = rep.forecast, rep.outlook
    assert f.nonnegative and o.scenarios["pessimistic"] == 0.0
    ko = rule_based_narrative(*_parts(rep))
    scen = next(ln for ln in ko.splitlines() if "시점 시나리오" in ln)
    assert "비관 0(하한)" in scen and "−100.0%" not in ko
    assert scen.count("0 아래로 내려갈 수 없음") == 1
    assert "값이 음수가 될 수 없어" not in ko             # one caveat, not two
    en = rule_based_narrative(*_parts(rep), lang="en")
    assert "pessimistic 0 (floor)" in en and "−100.0%" not in en and "cannot go below 0" in en
    assert "Values cannot be negative" not in en
    # a 95 % bound alone at the floor keeps the caveat section's note
    rep95 = _with_outlook(rep, scenarios={"pessimistic": 2.0, "base": 5.0, "optimistic": 8.0})
    ko95 = rule_based_narrative(*_parts(rep95))
    assert "비관 2.00" in ko95 and "값이 음수가 될 수 없어 예측 하한을 0으로 제한했습니다" in ko95


@pytest.mark.parametrize("key,ko_needle,en_needle", [
    ("auth", "ANTHROPIC_API_KEY", "authentication failed"),
    ("perm", "권한", "may not use the model"),
    ("rate", "rate limit", "rate limit was reached"),
    ("status", "HTTP 500", "returned an error (HTTP 500)"),
    ("conn", "연결할 수 없습니다", "Cannot reach the Claude API"),
])
def test_narrative_unavailable_messages_follow_lang(fake_report, key, ko_needle, en_needle):
    # U7: the messages were Korean even with lang='en' (the CLI prints them after an English prefix)
    _, errors = _sdk_errors()
    with pytest.raises(NarrativeUnavailable, match=re.escape(en_needle)) as ei:
        llm_narrative(*_parts(fake_report), lang="en", client=FakeClient(exc=errors[key]))
    assert not HANGUL.search(str(ei.value))
    with pytest.raises(NarrativeUnavailable, match=re.escape(en_needle)) as ei:
        ask(fake_report, "why?", lang="en", client=FakeClient(exc=errors[key]))
    assert not HANGUL.search(str(ei.value))
    with pytest.raises(NarrativeUnavailable, match=re.escape(ko_needle)):
        ask(fake_report, "왜?", client=FakeClient(exc=errors[key]))


def test_other_narrative_unavailable_paths_follow_lang(monkeypatch, fake_report):
    # U7: refusal, empty response, missing SDK, missing credentials, truncation warning
    cases = [FakeClient(_resp(stop_reason="refusal")),
             FakeClient(_resp(SimpleNamespace(type="thinking", thinking="..."))),
             FakeClient(exc=TypeError("Could not resolve authentication method")),
             FakeClient(exc=TypeError("create() got an unexpected keyword argument 'fallbacks'"))]
    for client in cases:
        with pytest.raises(NarrativeUnavailable) as ei:
            llm_narrative(*_parts(fake_report), lang="en", client=client)
        assert not HANGUL.search(str(ei.value)), str(ei.value)
        with pytest.raises(NarrativeUnavailable) as ei:
            llm_narrative(*_parts(fake_report), client=client)
        assert HANGUL.search(str(ei.value))
    with pytest.warns(RuntimeWarning) as rec:
        ask(fake_report, "why?", lang="en", client=FakeClient(_resp(_text("Because"), stop_reason="max_tokens")))
    assert not any(HANGUL.search(str(w.message)) for w in rec)
    monkeypatch.setitem(sys.modules, "anthropic", None)
    with pytest.raises(NarrativeUnavailable, match="not installed: pip install anthropic"):
        ask(fake_report, "why?", lang="en", client=FakeClient(_resp(_text("x"))))


def test_is_intermittent_is_public():
    # U8: report labels the anomaly table '평소 규모' for intermittent demand
    inter = _run(_intermittent_series(), name="parts")
    assert nv.is_intermittent(inter.analysis) is True
    assert nv.is_intermittent(_sample_run("growing").analysis) is False
    assert "is_intermittent" in nv.__all__ and "format_scenario" in nv.__all__


def test_narrative_changepoints_use_the_shared_labels():
    # U6: narrative, phase reasons and report name a break with analysis.changepoint_label
    from trendai.analysis import changepoint_label
    for nm, kw in (("reviving", {}), ("reviving", {"freq": "Y", "n": 25}), ("breakout", {"n": 30})):
        rep = _run(sample(nm, **kw), name=nm, freq=kw.get("freq", "M"))
        cps = [cp for cp in rep.analysis.changepoints if cp.kind != "level_shift"]
        assert cps
        for lang in ("ko", "en"):
            text = rule_based_narrative(*_parts(rep), lang=lang)
            for cp in cps:
                d = nv._Ctx(rep.analysis, rep.forecast, rep.outlook, lang).date(cp.date)
                lead = f"**{d}**부터 " if lang == "ko" else f"From **{d}**: "
                assert lead + changepoint_label(cp, lang) + " (" in text
        payload = build_llm_payload(*_parts(rep))
        assert all(c["label"] and c["label_ko"] for c in payload["changepoints"])
    assert "하락에서 상승으로" not in rule_based_narrative(*_parts(_run(sample("reviving"), name="r")))


# --------------------------------------------------------------------------- round 4 (final): consistency


def test_payload_numbers_print_like_every_other_surface():
    # F1: prob_increase was rounded to 3 significant digits, 0.45452 -> 0.455, which format_prob
    # (and so an LLM told to quote the payload) prints as '46%' while the narrative, the terminal
    # and the HTML report print '45%' (16 of 370 sweep reports)
    rep = _sample_run("growing")
    a, f, o = _parts(rep)
    cases = [(0.45452, 0.5949, 0.94501), (0.5949042, 0.45452, 0.955001), (0.99449, 0.99501, 0.9449),
             (0.0051, 0.00499, 0.125)]
    for prob, conf, strength in cases:
        o2 = dataclasses.replace(o, prob_increase=prob)
        ph = dataclasses.replace(a.phase, confidence=conf)
        m = dataclasses.replace(a.metrics, trend_strength=strength, seasonal_strength=strength, mk_tau=-strength)
        a2 = dataclasses.replace(a, phase=ph, metrics=m)
        pl = build_llm_payload(a2, f, o2)
        assert nv.format_prob(pl["outlook"]["prob_increase"]) == nv.format_prob(prob)
        assert pl["outlook"]["prob_increase_text"] == nv.format_prob(prob)
        assert nv.format_prob(pl["phase"]["confidence"]) == nv.format_prob(conf)
        assert pl["phase"]["confidence_text"] == nv.format_prob(conf)
        for key, v in (("trend_strength", strength), ("seasonal_strength", strength), ("mann_kendall_tau", -strength)):
            assert format_number(pl["metrics"][key], decimals=2) == format_number(v, decimals=2), key
        # the narrative prints exactly those texts
        text = rule_based_narrative(a2, f, o2)
        assert f"상승 확률 {nv.format_prob(prob)}" in text
        assert f"(판단 신뢰도 {nv.format_prob(conf)})" in text
    assert "prob_increase_text" in SYSTEM_PROMPT and "confidence_text" in SYSTEM_PROMPT
    pl = build_llm_payload(*_parts(rep))
    for key, fmt in (("total_change_pct", format_pct), ("recent_change_pct", format_pct)):
        assert fmt(pl["metrics"][key]) == fmt(getattr(rep.analysis.metrics, key))


def _short_rising(n: int, freq: str = "Y") -> pd.Series:
    from trendai.data import from_values
    r = np.random.default_rng(105)
    return from_values(10 + np.arange(n) + r.normal(0, 0.3, n), start="2021-01-01", freq=freq)


def test_short_series_phase_is_a_tentative_call():
    # F3: 5 yearly values 9.98 -> 13.9 (Mann-Kendall p = 0.027, significant rise) were called
    # 'Stable' with "it rose over the long run, but lately it has moved without a clear direction":
    # below analysis.MIN_PHASE_OBS observations the phase is only a placeholder
    from trendai.analysis import MIN_PHASE_OBS
    rep = _run(_short_rising(5), name="short up")
    a = rep.analysis
    assert a.metrics.n_obs < MIN_PHASE_OBS and a.metrics.direction == "up" and a.phase.label == "stable"
    assert nv.phase_is_tentative(a) and "phase_is_tentative" in nv.__all__
    ko = rule_based_narrative(*_parts(rep))
    en = rule_based_narrative(*_parts(rep), lang="en")
    assert "뚜렷한 방향 없이" not in ko and "without a clear direction" not in en
    assert "단계(관측치가 적어 잠정 판단, 신뢰도" in ko.splitlines()[0]
    assert "(tentative: too few observations; confidence" in en.splitlines()[0]
    mom_ko = next(ln for ln in ko.splitlines() if ln.startswith("- 모멘텀"))
    mom_en = next(ln for ln in en.splitlines() if ln.startswith("- Momentum"))
    assert "데이터가 짧아" in mom_ko and "too short" in mom_en
    assert "현재 국면은 **안정기**입니다 (관측치가 적어 잠정 판단" in ko
    assert "국면으로 넘어가" not in ko and "Projected phase shift" not in en
    assert "여러 국면의 경계" not in ko and "between phases" not in en
    assert not HANGUL.search(en.replace("short up", ""))
    pl = build_llm_payload(*_parts(rep))
    assert pl["phase"]["tentative"] is True and "phase.tentative" in SYSTEM_PROMPT
    # every frequency and n = 5..7 (the auditor's short_up_{D,W,M,Q,Y}{5,6,7})
    for freq in ("D", "W", "M", "Q", "Y"):
        for n in (5, 6, 7):
            r = _run(_short_rising(n, freq), name="s")
            text = rule_based_narrative(*_parts(r))
            assert "뚜렷한 방향 없이" not in text and "잠정 판단" in text.splitlines()[0], (freq, n)
    # from MIN_PHASE_OBS observations on the phase is a verdict again
    long = _run(_short_rising(MIN_PHASE_OBS), name="s")
    assert not nv.phase_is_tentative(long.analysis)
    assert build_llm_payload(*_parts(long))["phase"]["tentative"] is False
    assert "잠정" not in rule_based_narrative(*_parts(long)).splitlines()[0]


def test_mann_kendall_direction_against_the_total_change_is_qualified():
    # F6: fad W156 read '6.94에서 5.80까지 −16.5% 변했고 …' followed by 'Mann-Kendall 검정에서
    # 통계적으로 유의한 상승 추세가 확인됩니다': the rank-based test is driven by the bubble
    rep = _sample_run("fad", "W", 156)
    a = rep.analysis
    tr = a.decomposition.trend.to_numpy()
    assert a.metrics.direction == "up" and tr[-1] < tr[0]
    ko = rule_based_narrative(*_parts(rep))
    en = rule_based_narrative(*_parts(rep), lang="en")
    mk_ko = next(ln for ln in ko.splitlines() if "Mann-Kendall" in ln)
    mk_en = next(ln for ln in en.splitlines() if "Mann-Kendall" in ln)
    assert "확인됩니다" not in mk_ko and "중간의 높은 구간" in mk_ko
    assert "꾸준한 상승으로 보기는 어렵습니다" in mk_ko and "같은 이유로 양수입니다" in mk_ko
    assert "finds a statistically significant" not in mk_en and "high stretch in the middle" in mk_en
    assert "no steady rise" in mk_en
    # and the momentum line does not compare with a long-run "rise" either
    mom = next(ln for ln in ko.splitlines() if ln.startswith("- 모멘텀"))
    assert "장기 흐름과 달리" not in mom and "장기적으로는 상승" not in mom
    # a series whose test agrees with its ends keeps the plain sentence
    grow = rule_based_narrative(*_parts(_sample_run("growing")))
    assert "통계적으로 유의한 **상승** 추세가 확인됩니다" in grow and "같은 이유로" not in grow


def test_huge_numbers_use_one_readable_notation():
    # F8: '기울기 18,698,199,572,300 → 460,927,814,133,102/월' and on the next line
    # '460,927,814,133,102 → 1.51×10¹⁵/월': 13+-digit integers switch notation mid-line
    assert format_number(460_927_814_133.1, scale=1.51e15) == "4.61×10¹¹"
    assert format_number(0.0, scale=1e15) == "0" and format_number(12.5, scale=1e15) == "12.5"
    assert format_number(1_234_567.8, scale=1e3) == "1,234,568"
    rep = _run(sample("emerging", n=120, freq="M") * 1e15, name="huge")
    long_int = re.compile(r"\d{1,3}(?:,\d{3}){4,}")      # 13 digits or more
    for lang in ("ko", "en"):
        text = rule_based_narrative(*_parts(rep), lang=lang)
        assert not long_int.search(text), long_int.search(text)
        cps = [ln for ln in text.splitlines() if "→" in ln and ("기울기" in ln or "slope" in ln)]
        assert cps
        for ln in cps:
            pair = re.search(r"([+−]?[\d.,]+(?:×10[⁰¹²³⁴⁵⁶⁷⁸⁹⁻]+)?) → ([+−]?[\d.,]+(?:×10[⁰¹²³⁴⁵⁶⁷⁸⁹⁻]+)?)", ln)
            assert ("×10" in pair.group(1)) == ("×10" in pair.group(2)), ln


def test_random_walk_trend_strength_is_not_called_consistency():
    # F19: a random walk read '추세의 일관성(추세 강도 0.95)은 매우 높습니다' next to '근거: … 무작위
    # 행보에 가까운 움직임이라, 추세처럼 보이는 등락도 우연일 수 있습니다' -- the smoother absorbs the wander
    from trendai.forecast import is_random_walk
    r = np.random.default_rng(0)
    s = pd.Series(100 + np.cumsum(r.normal(0, 1, 400)), index=pd.date_range("2016-01-01", periods=400, freq="D"))
    rep = _run(s, name="rw")
    assert is_random_walk(rep.analysis) and rep.analysis.metrics.trend_strength >= 0.85
    ko = rule_based_narrative(*_parts(rep))
    en = rule_based_narrative(*_parts(rep), lang="en")
    assert "추세의 일관성" not in ko and "무작위 행보에서는 흔한 일이라 꾸준한 추세의 근거는 아닙니다" in ko
    assert "trend consistency" not in en and "typical of a random walk" in en
    assert "추세의 일관성" in rule_based_narrative(*_parts(_sample_run("growing")))


def test_evidence_gives_a_drawdown_across_zero_in_units():
    # R4-REV-8: with no usable phase reasons, _evidence printed '최근 값이 고점(2021-09)보다 184.9%
    # 낮습니다' for a P&L that fell from 10 through zero, where the past section says a % drop is meaningless
    rng = np.random.default_rng(3)
    y = np.r_[np.linspace(2, 10, 20), np.linspace(10, -8, 20)] + rng.normal(0, 0.3, 40)
    rep = _run(pd.Series(y, index=pd.date_range("2020-01-01", periods=40, freq="MS")), name="pnl", freq="M")
    a = rep.analysis
    m = a.metrics
    assert a.phase.label == "declining" and m.drawdown_pct < -100
    a2 = dataclasses.replace(a, phase=dataclasses.replace(a.phase, reasons=[]))
    gap = format_number(m.peak_value - m.last_value)
    for lang, lead in (("ko", "- 근거: "), ("en", "- Evidence: ")):
        text = rule_based_narrative(a2, rep.forecast, rep.outlook, lang=lang)
        ev = [ln for ln in text.splitlines() if ln.startswith(lead)]
        assert ev and not any("%" in ln and ("고점" in ln or "peak" in ln) for ln in ev), ev
        assert any(gap in ln and ("고점" in ln or "peak" in ln) for ln in ev), ev


@pytest.mark.parametrize("ko, en", [
    ("최근 18개월 동안 추세 기울기는 −0.69/월입니다.", "Over the last 18 months the trend slope was −0.69 per month."),
    ("최근 6주 동안 추세 기울기는 +1.20/주입니다.", "Over the last 6 weeks the trend slope was +1.20 per week."),
    ("관측치가 5개뿐이라 국면을 판단하기에 데이터가 부족합니다.",
     "Only 5 observations: too few to judge the lifecycle phase."),
    ("변화가 누적되는 무작위 행보에 가까운 움직임이라, 추세처럼 보이는 등락도 우연일 수 있습니다.",
     "Changes accumulate like a random walk, so trend-like swings may be due to chance."),
])
def test_translate_reason_knows_the_round4_templates(ko, en):
    # F11 ('/월' instead of '/개월'), F3 (a short series' reason is about the phase), F12 (no gloss)
    assert nv.translate_reason(ko) == en


# --------------------------------------------------------------------------- round 5


def _pnl_down() -> pd.Series:
    rng = np.random.default_rng(1)
    t = np.arange(120)
    return pd.Series(50 - 80 * t / 119 + rng.normal(0, 4, 120),
                     index=pd.date_range("2016-01-01", periods=120, freq="MS"), name="P&L")


def test_negative_current_level_is_compared_in_units():
    """G1: a negative current level (−25.9 → −34.2) printed '−32.0% 변할 것' and scenario percentages
    against the negative level.  Now: the reason in words, both levels and their difference, no %."""
    from trendai.forecast import outlook_levels

    for lang in ("ko", "en"):
        rep = _run(_pnl_down(), name="P&L", lang=lang)
        lv = outlook_levels(rep.analysis, rep.forecast)
        assert lv.undefined == "negative" and rep.outlook.expected_change_pct is None
        text = rep.narrative
        summary = text.split("\n", 1)[0]
        out = text.split("## 미래 전망" if lang == "ko" else "## Outlook")[1].split("## ")[0]
        assert not re.search(r"[+−]\d[\d,.]*%", summary + out.split("\n")[1])
        scen = next(ln for ln in out.split("\n") if "시나리오" in ln or "Scenarios" in ln)
        assert not re.search(r"\([+−]\d[\d,.]*%\)", scen)
        cur, fut = format_number(lv.current, signed=True), format_number(lv.future, signed=True)
        if lang == "ko":
            assert "전망은 **하락**입니다" in summary
            assert (f"현재 수준이 음수(최근 12개월 평균 {cur} → 향후 12개월 예측 평균 {fut})여서 "
                    f"변화율(%) 대신 수준으로 비교합니다") in out
            assert "0에 가까워" not in out
        else:
            assert "The current level is negative, so the levels are compared instead of a % change" in out
            assert f"{cur} (the average of the last 12 months) → {fut}" in out
    p = nv.build_llm_payload(rep.analysis, rep.forecast, rep.outlook)["outlook"]
    assert p["expected_change_undefined"] == "negative" and p["expected_change_pct"] is None
    assert "current level is negative" in p["expected_change_basis"]
    assert "'negative' compare the two compared_levels" in nv.SYSTEM_PROMPT


def test_direction_label_of_a_flat_negative_outlook_uses_the_relative_move():
    """G1: with no percentage a 'flat' outlook read 보합; for a negative level the move relative to
    the level decides like a percentage would (보합 within 2 %, else 방향 불확실)."""
    from types import SimpleNamespace as NS

    small = NS(current=-100.0, future=-101.0, undefined="negative")
    large = NS(current=-100.0, future=-130.0, undefined="negative")
    assert nv.direction_label("flat", None, "ko", levels=small) == "보합"
    assert nv.direction_label("flat", None, "ko", levels=large) == "방향 불확실"
    assert nv.direction_label("flat", None, "en", "negative", levels=large) == "direction uncertain"
    assert nv.direction_label("down", None, "ko", levels=large) == "하락"
    # calls without levels are unchanged
    assert nv.direction_label("flat", None, "ko", "negative") == "보합"
    assert nv.direction_label("flat", None, "ko", "sign_change") == "방향 불확실"


def _intermittent_seed3() -> pd.Series:
    rng = np.random.default_rng(3)
    y = (rng.random(60) < 0.6) * rng.integers(1, 20, 60).astype(float)
    return pd.Series(y, index=pd.date_range("2020-01-01", periods=60, freq="MS"), name="parts")


def test_intermittent_outlook_is_worded_as_average_demand():
    """G2 (R4-REV-3): intermittent demand keeps a % of average demand rates, but the narrative first
    said the trend line's % means nothing and then printed '−17.9% 변할 것' as if it were the level.
    Now the two levels are named average demand per period, with their values."""
    from trendai.forecast import outlook_levels

    for lang in ("ko", "en"):
        rep = _run(_intermittent_seed3(), name="parts", lang=lang)
        assert nv.is_intermittent(rep.analysis) and rep.outlook.expected_change_pct is not None
        lv = outlook_levels(rep.analysis, rep.forecast)
        fv, cv = format_number(lv.future), format_number(lv.current)
        out = rep.narrative.split("## 미래 전망" if lang == "ko" else "## Outlook")[1].split("\n")
        line = out[1]
        if lang == "ko":
            assert line.startswith(f"- 향후 12개월 평균 수요량(기간당 {fv})은 최근 12개월 평균 수요량({cv})보다 **")
            assert "추세선 변화율과는 다른 수치입니다" in line
            assert out[2].startswith("- 향후 12개월 평균 수요량이 최근 12개월 평균 수요량보다 높아질 확률은")
            assert "평균 수요량의 **" in rep.narrative.split("\n", 1)[0]
        else:
            assert line.startswith(f"- The average demand per period over the next 12 months ({fv}) is expected "
                                   f"to change by **")
            assert "a different quantity from the trend line" in line
            assert "the average demand per period is expected to change by" in rep.narrative.split("\n", 1)[0]
    p = nv.build_llm_payload(rep.analysis, rep.forecast, rep.outlook)
    assert "average demand per period" in p["outlook"]["expected_change_basis"]
    assert "average demand per period" in p["conventions"]["intermittent"]
    # ordinary series keep their wording
    plain = _run(sample("growing"), name="g").narrative
    assert "수요량" not in plain and "- 향후 12개월 예측 평균은 최근 12개월 평균보다 **" in plain


def test_all_zero_history_is_described_as_zero_throughout():
    """G3 (round-4 F18): an all-zero history read '0에서 0까지 0 변했지만 …', '0(하한)' scenarios and
    '기준 예측이 하한인 0까지 내려갑니다' although its forecast of exactly 0 is no clipped floor."""
    from trendai.data import from_values

    s = from_values(np.zeros(36), start="2023-01-01", freq="M")
    ko = _run(s, name="zero")
    en = _run(s, name="zero", lang="en")
    assert nv.is_all_zero(ko.analysis) and not nv.scenario_floor(ko.analysis, ko.forecast)
    assert nv.floor_note(ko.analysis, ko.forecast, ko.outlook) == ""
    assert "- 관측 기간 내내 0이었습니다." in ko.narrative
    for bad in ("0에서 0까지", "하한", "값이 변하지 않았습니다"):
        assert bad not in ko.narrative, bad
    assert "비관 0 · 기준 0 · 낙관 0." in ko.narrative
    assert "(관측값이 모두 0이라 변화율은 계산하지 않음)" in ko.narrative
    assert "- The value was 0 throughout the observed period." in en.narrative
    for bad in ("from 0 to 0", "floor"):
        assert bad not in en.narrative, bad
    # a positive series whose band reaches 0 keeps its floor wording
    rep = _run(_intermittent_seed3(), name="parts")
    assert nv.scenario_floor(rep.analysis, rep.forecast)
    assert nv.floor_note(rep.analysis, rep.forecast, rep.outlook) in rep.narrative


def test_round5_public_helpers_replace_mirrors():
    """G4: one implementation for rules that report / analysis used to mirror."""
    from trendai import analysis as A
    from trendai import numfmt
    from trendai import report as R

    for name in ("PHASE_EN", "future_phase_status", "is_all_zero", "scenario_floor", "floor_note",
                 "drawdown_undefined", "seasonal_trend_drawdown"):
        assert name in nv.__all__ and hasattr(nv, name), name
    assert "max(season, 12)" in nv.is_intermittent.__doc__
    # one number notation: narrative, analysis phase reasons and report
    for x in (0.0, 1.05e-9, -2.5e-11, 3e-5, 0.0999, 12.345, -1234.5, 999_999_999_999.0, 1.87e13, -6.99e16):
        t = numfmt.number_text(x)
        assert format_number(x) == t and A._num_txt(x) == t
        assert format_number(x, signed=True) == numfmt.number_text(x, signed=True) == A._num_txt(x, signed=True)
    assert R._sci(4.2e16) == numfmt.sci(4.2e16) == "4.2×10¹⁶"
    assert not hasattr(nv, "_SCI_LARGE") and R._SCI_LARGE is numfmt.SCI_LARGE
    # the projected-phase rule: narrative, payload basis and report from one function
    for nm, n, freq in (("peaking", 120, "M"), ("growing", 120, "M"), ("fad", 36, "M"), ("emerging", 8, "Y")):
        rep = _sample_run(nm, freq, n)
        status = nv.future_phase_status(rep.analysis, rep.outlook)
        assert R._future_phase_hedge(rep) == status
        assert (nv._future_phase_basis(rep.analysis, rep.outlook) is None) == (status is None)
    short = _run(pd.Series([10.0, 10.6, 11.1, 12.4, 12.9],
                           index=pd.date_range("2021-01-01", periods=5, freq="YS")), name="s")
    assert nv.future_phase_status(short.analysis, short.outlook) == "tentative"
    assert "tentative" in nv._future_phase_basis(short.analysis, short.outlook)
    # the seasonal drawdown rule: one function behind the narrative and the "현재 값" tile
    for nm in ("seasonal", "peaking", "fad"):
        rep = _sample_run(nm)
        assert R._seasonal_trend_drawdown(rep, R._ChartData(rep)) == nv.seasonal_trend_drawdown(rep.analysis)
    assert A.CAGR_MIN_YEARS == R._CAGR_MIN_YEARS == 0.95


def test_scenarios_name_the_interval_they_come_from():
    """G4: with levels (50, 95) the narrative wrote '50% 예측구간' next to the 95 % band's values."""
    rep = _run(sample("growing"), name="g", levels=(50, 95))
    line = next(ln for ln in rep.narrative.split("\n") if "시점 시나리오" in ln)
    assert "(95% 예측구간" in line and "50% 예측구간" not in line
    assert format_number(rep.forecast.lower[95].iloc[-1]) in line
    p = nv.build_llm_payload(rep.analysis, rep.forecast, rep.outlook)
    assert p["outlook"]["scenario_interval_level"] == 95


# --------------------------------------------------------------------------- round 5, verify 1


def _monthly(vals, end: str = "2025-12-01", freq: str = "MS") -> pd.Series:
    return pd.Series(np.asarray(vals, float), index=pd.date_range(end=end, periods=len(vals), freq=freq))


def _pnl_spike() -> pd.Series:
    r = np.random.default_rng(1)
    t = np.arange(120)
    y = 50 - 80 * t / 119 + r.normal(0, 4, 120)
    y[100] += 35                                     # a spike where the trend is about -17
    return _monthly(y)


def test_anomaly_line_follows_the_series_percentage_rule():
    """V1-1: '실제 15.2, 예상 −16.5 (+191.9%, z = +9.6)' on a P&L that crossed zero."""
    for lang in ("ko", "en"):
        rep = _run(_pnl_spike(), name="P&L", lang=lang)
        lines = [ln for ln in rep.narrative.splitlines() if re.search(r"(실제|actual) .*z = ", ln)]
        assert lines and not [ln for ln in lines if "%" in ln], lines
        assert any(("차이 +" if lang == "ko" else "a gap of +") in ln for ln in lines)


def test_rates_of_an_undefined_level_are_left_out():
    """V1-3: '평균 수준 대비 −66.77%/년' (a P&L crossing zero), '−2.01%/월' (a negative level),
    '0.00%/월' (all zeros), momentum '+20.01%p/월' of intermittent demand; payload slope_pct non-null."""
    r = np.random.default_rng(1)
    t = np.arange(60)
    launch = np.zeros(36)
    launch[-6:] = [3, 5, 4, 2, 3, 2]
    cases = {"near_zero": _monthly(50 - 80 * t / 59 + r.normal(0, 4, 60)),      # a P&L crossing zero
             "negative": _monthly(-20 - 60 * t / 59 + r.normal(0, 2, 60)),
             "zero": _monthly(np.zeros(36)), "intermittent": _monthly(launch)}
    for why, s in cases.items():
        rep = _run(s, name="x")
        assert nv.rate_pct_undefined(rep.analysis) == why
        assert "평균 수준 대비" not in rep.narrative and "%p/" not in rep.narrative
        en = rule_based_narrative(*_parts(rep), lang="en")
        assert "of the mean level" not in en and " pp/" not in en
        pay = build_llm_payload(*_parts(rep))
        assert pay["metrics"]["slope_pct_per_period"] is None and pay["metrics"]["momentum"] is None
        assert pay["metrics"]["pct_undefined"]["slope_pct_per_period"] == why
    # an ordinary positive series keeps them
    rep = _run(_monthly(100 + 0.5 * t + r.normal(0, 1, 60)), name="x")
    assert "평균 수준 대비" in rep.narrative and "%p/" in rep.narrative


def test_flat_label_follows_the_levels_of_mostly_zero_demand():
    """V1-4: '전망 ▶ 보합' above '최근 52주 평균 수요량 0.35 → 향후 26주 평균 수요량 0.069'."""
    from trendai.forecast import outlook_levels
    from trendai.report import render_text

    launch = np.zeros(36)
    launch[-6:] = [3, 2, 4, 3, 5, 2]
    rep = _run(_monthly(launch), name="launch")
    lv = outlook_levels(rep.analysis, rep.forecast)
    assert lv.undefined == "mostly_zero" and rep.outlook.direction == "flat"
    assert nv.direction_label("flat", None, "ko", levels=lv) == "방향 불확실"
    assert "**방향 불확실**" in rep.narrative and "**보합**" not in rep.narrative
    assert "전망 ▶ 방향 불확실" in render_text(rep)
    assert build_llm_payload(*_parts(rep))["outlook"]["direction_label"] == {"ko": "방향 불확실",
                                                                              "en": "direction uncertain"}
    # a level of exactly 0 that stays ~0 relative to the history is still 'roughly unchanged'
    small = SimpleNamespace(current=0.0, future=0.05, undefined="zero", scale=20.0)
    assert nv.direction_label("flat", None, "ko", levels=small) == "보합"
    large = SimpleNamespace(current=0.0, future=2.0, undefined="zero", scale=20.0)
    assert nv.direction_label("flat", None, "ko", levels=large) == "방향 불확실"


def test_phase_outlook_gap_needs_a_recent_move_and_a_slower_forecast():
    """V1-5 / V1-6: '최근 하락세가 뚜렷하지만' next to '하락세가 멈췄습니다'; 'the models do not extend
    it' although the forecast goes on at the history's own pace (2.42 a year forecast to move 2.30)."""
    launch = np.zeros(36)
    launch[-6:] = [3, 5, 4, 2, 3, 2]
    for lang in ("ko", "en"):
        text = _run(_monthly(launch), name="launch", lang=lang).narrative
        assert not ("하락세가 뚜렷" in text and "하락세가 멈췄" in text)
        assert not ("recent decline is clear" in text and "decline has paused" in text)
    r = np.random.default_rng(5)
    t = np.arange(120)
    base = 100 + 0.2 * t + 15 * np.sin(2 * np.pi * t / 12) + r.normal(0, 2, 120)
    for y in (base, -base):
        for lang in ("ko", "en"):
            text = _run(_monthly(y), name="slow", lang=lang).narrative
            assert "이 흐름이 이어진다고 보지 않고" not in text and "do not extend it" not in text


def _all_zero_texts():
    from trendai.report import render_html, render_text

    out = []
    for lang in ("ko", "en"):
        rep = _run(_monthly(np.zeros(36)), name="zero", lang=lang)
        out.append((lang, rep, rep.narrative, render_text(rep, lang=lang)))
    return out, render_html(out[0][1])


def test_all_zero_history_states_no_probability_interval_or_rates():
    """V1-7 / R5-REV-4: '상승 확률 약 50%, 엇비슷', '95% 구간은 0 ~ 0', 'uncertainty grows', '0.0%' /
    '0.00%' of a zero level, and '기준 수준이 0에 가까워' next to the report's '관측값이 모두 0'."""
    texts, page = _all_zero_texts()
    for lang, rep, nar, txt in texts:
        for bad in ("50%", "0 ~ 0", "0 – 0", "0.0%", "0.00%", "불확실성이 커지므로", "uncertainty grows",
                    "0에 가까워", "close to zero", "%p", " pp"):
            assert bad not in nar and bad not in txt, (lang, bad)
        assert ("관측값이 모두 0" if lang == "ko" else "every observation is 0") in nar
        pay = build_llm_payload(*_parts(rep))
        assert pay["outlook"]["prob_increase"] is None and pay["metrics"]["volatility_cv"] is None
    kpi = re.search(r'<div class="lbl">상승 확률</div><div class="val"[^>]*>(.*?)</div>', page).group(1)
    assert "50%" not in kpi and "—" in kpi


def test_floor_note_when_the_level_is_already_zero():
    """V1-7: '기준 예측이 하한인 0까지 내려갑니다' when the last 12 months are already 0."""
    rep = _report_from_values(np.r_[5.0, np.zeros(10), 2.0, np.zeros(24)],
                              outlook_kw=dict(scenarios={"pessimistic": 0.0, "base": 0.0, "optimistic": 0.4}))
    rep = _with_forecast(rep, nonnegative=True)
    note = nv.floor_note(*_parts(rep))
    assert note.startswith("기준 예측은 하한인 0에 머뭅니다") and "내려갑니다" not in note
    assert nv.floor_note(*_parts(rep), lang="en").startswith("The base forecast stays at the floor of 0")


def test_drawdown_of_a_count_at_zero_names_the_peak():
    """V1-9: '최근 값(0)은 고점보다 100.0% 낮아 고점의 1/10 이하로 급감' / HTML '최고점 대비 −100.0%'."""
    from trendai.report import render_html

    r = np.random.default_rng(2)
    y = (r.poisson(3, 60) + 1).astype(float)
    y[-1] = 0.0
    rep = _run(_monthly(y), name="count")
    assert nv.drawdown_undefined(rep.analysis) == "at_zero"
    assert "100.0%" not in rep.narrative and "1/10" not in rep.narrative
    assert "최근 값은 0입니다" in rep.narrative
    assert "−100.0%" not in render_html(rep)
    assert build_llm_payload(*_parts(rep))["metrics"]["drawdown_pct"] is None


def test_zscores_of_an_almost_constant_history_are_capped():
    """V1-10: 'z = +6,239.4' for a lone first order after 155 zero weeks."""
    z = np.zeros(156)
    z[-1] = 5.0
    rep = _run(_monthly(z, freq="W-SUN"), name="first order")
    assert rep.analysis.anomalies and abs(rep.analysis.anomalies[-1].zscore) > 100
    assert "z > +100" in rep.narrative and not re.search(r"z = [+−][\d,]{4,}", rep.narrative)
    pay = build_llm_payload(*_parts(rep))
    assert all(abs(a["robust_zscore"]) <= 100 for a in pay["anomalies"])
    assert nv.zscore_text(9.62) == "z = +9.6" and nv.zscore_text(-250.0) == "z < −100"


def test_exact_zero_current_level_reads_the_same_on_every_surface():
    """R5-REV-2: narrative '현재 수준이 0이라', terminal/KPI/card '현재 수준이 0에 가까움'."""
    from trendai.report import render_html, render_text

    s = pd.Series(np.r_[np.arange(1.0, 49.0), np.zeros(12)],
                  index=pd.date_range("2018-01-01", periods=60, freq="MS"))
    rep = _run(s, name="z")
    assert "(현재 수준이 0이라 변화율은 계산하지 않음)" in rep.narrative
    assert "예측 변화 — (현재 수준이 0)" in render_text(rep)
    assert "Exp. change — (current level is 0)" in render_text(rep, lang="en")
    page = re.sub(r"<[^>]+>", "|", render_html(rep))
    assert "0에 가까" not in page
    assert "현재 수준이 0이라 변화율은 계산하지 않았습니다." in page
    en = _run(s, name="z", lang="en").narrative
    assert "(no % change: current level is 0)" in en


def test_intermittent_single_value_forecast_is_not_called_a_window_average():
    """R5-REV-3: horizon 1 -- '0인 기간까지 포함한 구간 평균끼리' / 'Both are window averages'."""
    r = np.random.default_rng(3)
    y = np.where(r.random(48) < 0.5, 0, r.poisson(4, 48)).astype(float)
    s = pd.Series(y, pd.date_range("2020-01-01", periods=48, freq="MS"))
    ko = _run(s, name="parts", horizon=1).narrative
    en = _run(s, name="parts", horizon=1, lang="en").narrative
    assert "구간 평균끼리" not in ko and "최근 수준은 0인 기간까지 포함한 구간 평균이고 예측은 한 시점의 값이라" in ko
    assert "Both are window averages" not in en and "the forecast a single value" in en
    line = next(ln for ln in en.splitlines() if "is expected to change by" in ln)
    assert ") (" not in line                                    # no stacked parentheticals


# --------------------------------------------------------------------------- round 5, verify 2


def _run_quiet(series, **kw):
    from trendai import pipeline

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pipeline.run(series, **kw)


def _dead_product() -> pd.Series:
    """V2-2: two orders (months 0 and 20), then nothing for 39 months."""
    z = np.zeros(60)
    z[0], z[20] = 5.0, 2.0
    return pd.Series(z, index=pd.date_range("2021-01-01", periods=60, freq="MS"), name="dead product")


def _zero_tail() -> pd.Series:
    """R5-REV2-2: 100 rising days, then 450 zero days (the forecast's history is all zeros)."""
    return pd.Series(np.r_[np.linspace(5, 50, 100), np.zeros(450)],
                     index=pd.date_range("2020-01-01", periods=550, freq="D"), name="tail0")


def test_floor_outlook_states_no_minus_100_percent():
    """V2-1: 'sample fad -H 24' read '향후 24개월 동안 **−100.0%** 변화가 예상되지만 … 상승 확률 45%':
    the forecast means were clipped at the 0 floor.  The outlook names the floor instead, on every
    surface, and the direction reads the fall to 0 (no '보합')."""
    from trendai import report as R
    from trendai.pipeline import report_to_dict

    s = sample("fad", n=120, freq="M")
    for h, clause_ko, clause_en, short in (
            (24, "예측이 하한인 0까지 내려가 변화율(%)은 의미가 없습니다(최근 12개월 평균 7.74 → 예측 마지막 12개월 평균 0)",
             "The forecast falls to the floor of 0: the average of the last 12 months of the forecast is 0",
             "예측 변화 — (예측이 하한 0까지 내려감)"),
            (18, "예측 마지막 12개월 중 10개월이 하한인 0에 닿아 변화율(%)은 의미가 없습니다",
             "10 of the last 12 months of the forecast sit at the floor of 0",
             "예측 변화 — (예측 마지막 12개월 중 10개월이 하한 0)")):
        ko = _run_quiet(s, freq="M", horizon=h, name="fad")
        en = _run_quiet(s, freq="M", horizon=h, name="fad", lang="en")
        assert ko.outlook.expected_change_pct is None
        for text in (ko.narrative, en.narrative, R.render_text(ko), R.render_text(en, lang="en"), R.render_html(ko)):
            assert "−100.0%" not in text and "−99." not in text
        assert clause_ko in ko.narrative and clause_en in en.narrative
        assert "**방향 불확실**" in ko.narrative and "보합" not in ko.narrative.split("## 미래 전망")[1].split("\n")[1]
        assert short in R.render_text(ko)
        payload = build_llm_payload(*_parts(ko))
        assert payload["outlook"]["expected_change_undefined"] == "floor"
        assert "floor of a nonnegative forecast" in payload["outlook"]["expected_change_basis"]
        assert report_to_dict(ko)["outlook"]["expected_change_undefined"] == "floor"


def test_zero_current_level_states_no_probability():
    """V2-2: a current window of exactly 0 for data that never go below 0 read '높아질 확률은 약 **50%**로,
    상승과 하락 가능성이 엇비슷합니다' -- a fall is impossible.  One rule (prob_undefined) on every
    surface: the narrative, the payload, the JSON, the terminal, the KPI and the ranking."""
    from trendai import compare
    from trendai import report as R
    from trendai.pipeline import report_to_dict

    ko = _run_quiet(_dead_product(), freq="M")
    en = _run_quiet(_dead_product(), freq="M", lang="en")
    assert nv.prob_undefined(ko.analysis) == "zero_current"
    assert "- 최근 39개월 동안 값이 모두 0이고 0 아래로는 내려갈 수 없어, 상승·하락 확률은 따지지 않았습니다." \
        in ko.narrative.splitlines()
    assert "The last 39 months are all 0 and values cannot go below 0" in en.narrative
    for text in (ko.narrative, R.render_text(ko)):
        assert "엇비슷" not in text and "상승 확률 50%" not in text
    assert "equally likely" not in en.narrative and "P(up) 50%" not in R.render_text(en, lang="en")
    assert "상승 확률 — (최근 39개월 연속 0)" in R.render_text(ko)
    assert "P(up) — (the last 39 months are all 0)" in R.render_text(en, lang="en")
    payload = build_llm_payload(*_parts(ko))["outlook"]
    assert payload["prob_increase"] is None and payload["prob_increase_undefined"] == "zero_current"
    out = report_to_dict(ko)["outlook"]
    assert out["prob_increase"] is None and out["prob_increase_undefined"] == "zero_current"
    assert compare.rank_trends([ko])["prob_increase"].tolist() == [None]
    html = R.render_html(ko)
    assert "최근 39개월 동안 값이 모두 0이고 0 아래로는 내려갈 수 없어" in html


def test_zero_forecast_is_one_rule_for_probability_floor_and_caveat():
    """R5-REV2-2: history non-zero long ago, the forecast's own history all zeros -> a forecast of
    exactly 0 with a zero-width band.  scenario_floor called it 'forecast is 0', but the probability
    read '50%, a rise and a fall about equally likely' and the caveat '먼 미래일수록 불확실성이 커지므로'."""
    from trendai import compare
    from trendai import report as R

    rep = _run_quiet(_zero_tail(), name="tail0")
    a = rep.analysis
    assert nv.forecast_is_zero(a) and not nv.is_all_zero(a) and not nv.scenario_floor(a, rep.forecast)
    assert nv.prob_undefined(a) == "zero_forecast"
    nar = rep.narrative
    assert "- 최근 450일 동안 값이 모두 0이라 예측도 0(구간 폭 0)이고, 상승·하락 확률은 따지지 않았습니다." in nar
    assert "- 최근 450일 동안 값이 모두 0이라 예측도 0(구간 폭 0)입니다." in nar
    assert "먼 미래일수록" not in nar and "엇비슷" not in nar and "상승 확률 50%" not in nar
    assert "상승 확률 — (최근 450일 연속 0이고 예측도 0)" in R.render_text(rep)
    assert compare.rank_trends([rep])["prob_increase"].tolist() == [None]
    en = _run_quiet(_zero_tail(), name="tail0", lang="en").narrative
    assert "The last 450 days are all 0 and so is the forecast" in en and "grows with the horizon" not in en


def test_scenarios_are_compared_with_one_reference_on_every_surface():
    """V2-4: the narrative compared the horizon-end scenarios with the 12-month mean (7.74; seasonal:
    49.7) while the terminal and the HTML compared them with the latest value (5.17) or the same
    point of the season (2025-12: 46.9).  One rule (scenario_reference) for all."""
    from trendai import report as R

    for name, ko_ref, text_ref, cap in (
            ("fad", "괄호는 현재 값(2025-12) 5.17 대비", "(비교: 현재 5.17)", "세로선은 현재 값(2025-12)입니다"),
            ("seasonal", "괄호는 같은 계절 시점인 2025-12의 실제값 46.9 대비", "(비교: 2025-12 실제 46.9)",
             "세로선은 같은 계절 시점인 2025-12의 실제값입니다")):
        rep = _run_quiet(sample(name, n=120, freq="M"), freq="M", name=name)
        ref = nv.scenario_reference(rep.analysis, rep.forecast)
        assert ko_ref in rep.narrative
        assert text_ref in " ".join(R.render_text(rep, width=200).split())
        assert cap in R.render_html(rep)
        assert build_llm_payload(*_parts(rep))["outlook"]["scenario_reference"]["value"] == pytest.approx(
            ref.value, rel=1e-3)
        base = rep.outlook.scenarios["base"]
        assert f"기준 {format_number(base)} ({format_pct((base - ref.value) / abs(ref.value) * 100)})" in rep.narrative


def test_intermittent_past_quotes_average_demand_not_trend_ends():
    """V2-5: '추세선(평균 수요율 2.81 → 1.06)' for demand whose first-year mean is 0.67 and last-year
    mean 1.17 -- the trend of mostly-zero data follows the edge orders -- then '다만 추세선의 끝은
    시작보다 낮고' next to a significant upward test."""
    v = [4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 5, 2, 0, 0, 5, 3, 0, 4, 2,
         0, 2, 0, 4, 0, 0, 0, 4, 2, 4, 2, 5, 2, 0, 1, 3, 0, 0, 3, 0, 0, 0, 1, 1, 5, 0]
    s = pd.Series(np.array(v, float), index=pd.date_range("2021-01-01", periods=60, freq="MS"), name="demand")
    ko = _run_quiet(s, freq="M")
    assert nv.demand_rates(ko.analysis) == (12, pytest.approx(0.6667, abs=1e-3), pytest.approx(1.1667, abs=1e-3))
    assert "기간당 평균 수요량은 처음 12개월 0.67 → 최근 12개월 1.17입니다." in ko.narrative
    assert "평균 수요율 2.81" not in ko.narrative and "추세선의 끝은 시작보다" not in ko.narrative
    en = _run_quiet(s, freq="M", lang="en").narrative
    assert "went from 0.67 over the first 12 months to 1.17 over the last 12 months" in en
    assert "ends below where it started" not in en


def test_long_zero_run_and_zero_median_slope_are_stated():
    """V2-7: a product without a sale for 41 weeks (a sale every 3.3 weeks before) -- the narrative
    gave only 'last value 0, 52-week mean 0.15'; and '장기 평균 기울기는 0/주' beside a significant
    decline (the median slope of mostly-zero data is 0)."""
    r = np.random.default_rng(57)
    n = 104
    z = (r.poisson(2, n) + 1) * (r.random(n) < 0.35)
    z[-(n // 3):] = 0
    s = pd.Series(z.astype(float), index=pd.date_range("2024-01-07", periods=n, freq="W-SUN"), name="discontinued")
    ko = _run_quiet(s, freq="W")
    assert nv.long_zero_run(ko.analysis)[0] == 41
    assert "- 최근 41주 연속 수요가 없었습니다(마지막 수요: 2025-03-16의 1.00). 그전에는 평균 3.3주마다 수요가 있었습니다." \
        in ko.narrative.splitlines()
    assert "장기 평균 기울기는 0/주" not in ko.narrative
    assert "관측치의 절반 이상이 0이라 0/주로 나와" in ko.narrative
    en = _run_quiet(s, freq="W", lang="en").narrative
    assert "No demand for the last 41 weeks in a row" in en and "only because more than half" in en
    # ordinary gaps are no news: demand every few periods up to the end
    assert nv.long_zero_run(_run_quiet(pd.Series(np.tile([0.0, 0, 3], 20), index=pd.date_range(
        "2020-01-01", periods=60, freq="MS")), freq="M").analysis) is None


def test_recent_pace_phrase_never_contradicts_the_phase_reason():
    """R5-REV2-1 sweep: 'declining' with the reason '최근 23주 동안 추세가 1.42에서 1.25까지 낮아졌습니다' and
    the momentum phrase '최근에는 하락세가 멈췄습니다' (over a 39-week window): two opposite facts."""
    rng = np.random.default_rng(10)
    n = 156
    y = (rng.poisson(np.linspace(0.8, 0.2, n)) * rng.integers(1, 10, n)).astype(float)
    s = pd.Series(y, index=pd.date_range("2020-01-05", periods=n, freq="W-SUN"))
    rep = _run_quiet(s, freq="W", name="x")
    assert rep.analysis.phase.label == "declining"
    assert any("낮아졌습니다" in r for r in rep.analysis.phase.reasons)
    assert "최근에는 하락세가 멈췄습니다" not in rep.narrative
    assert "최근 39주 전체로는 하락세가 멈췄지만, 최근 23주 동안은 다시 낮아졌습니다" in rep.narrative
