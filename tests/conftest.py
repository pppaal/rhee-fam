"""Shared fixtures.

``fake_report`` builds a fully-populated :class:`TrendReport` by hand (no
analysis/forecast code involved) so that narrative/report modules can be
tested in isolation.  The series name deliberately contains HTML to catch
escaping bugs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trendai.data import future_index
from trendai.samples import sample
from trendai.types import (
    AnalysisResult, Anomaly, Changepoint, Decomposition, ForecastResult, LifecyclePhase,
    ModelScore, Outlook, TrendMetrics, TrendReport,
)

XSS_NAME = 'AI 에이전트 <script>alert("x")</script> & co'


def make_fake_report(name: str = XSS_NAME, kind: str = "breakout", horizon: int = 12) -> TrendReport:
    s = sample(kind)
    s.name = name
    n = len(s)
    # crude hand-made decomposition: centered rolling mean as trend
    trend = s.rolling(13, center=True, min_periods=1).mean()
    seasonal = pd.Series(np.zeros(n), index=s.index)
    resid = s - trend - seasonal
    dec = Decomposition(trend=trend, seasonal=seasonal, resid=resid, period=12)
    k = 71
    cps = [Changepoint(date=s.index[k], index=k, slope_before=0.02, slope_after=1.2,
                       kind="acceleration", magnitude=1.18)]
    anomalies = [Anomaly(date=s.index[30], index=30, value=float(s.iloc[30]),
                         expected=float(trend.iloc[30]), zscore=4.2)]
    peak_i = int(np.argmax(s.to_numpy()))
    metrics = TrendMetrics(
        n_obs=n, freq="M", start=s.index[0], end=s.index[-1], last_value=float(s.iloc[-1]),
        mean=float(s.mean()), total_change_pct=180.0, cagr_pct=11.0, recent_change_pct=35.0,
        slope=0.5, slope_pct=1.2, momentum=2.5, acceleration=0.3, volatility=0.05,
        trend_strength=0.95, seasonal_strength=0.1, mk_tau=0.6, mk_pvalue=1e-6, direction="up",
        peak_date=s.index[peak_i], peak_value=float(s.iloc[peak_i]),
        drawdown_pct=float((s.iloc[-1] - s.iloc[peak_i]) / s.iloc[peak_i] * 100),
    )
    phase = LifecyclePhase(label="growing", confidence=0.8,
                           reasons=["최근 기울기가 장기 평균보다 가파름", "Mann-Kendall 상승 추세 유의"])
    analysis = AnalysisResult(name=name, series=s, freq="M", season_length=12, decomposition=dec,
                              metrics=metrics, changepoints=cps, anomalies=anomalies, phase=phase)
    idx = future_index(s.index[-1], "M", horizon)
    last = float(trend.iloc[-1])
    mean = pd.Series(last + 1.2 * np.arange(1, horizon + 1), index=idx)
    sd = 2.0 * np.sqrt(np.arange(1, horizon + 1))
    lower = {80: mean - 1.2816 * sd, 95: mean - 1.96 * sd}
    upper = {80: mean + 1.2816 * sd, 95: mean + 1.96 * sd}
    scores = [
        ModelScore("holt_damped", 2.0, 2.5, 3.0, 0.7, 0.5),
        ModelScore("linear_trend", 2.2, 2.8, 3.3, 0.8, 0.3),
        ModelScore("theta", 2.5, 3.0, 3.6, 0.9, 0.2),
        ModelScore("naive", 6.0, 7.0, 8.0, 2.1, 0.0),
    ]
    fc = ForecastResult(mean=mean, lower=lower, upper=upper, horizon=horizon, levels=(80, 95),
                        model_scores=scores, method="ensemble(holt_damped, linear_trend, theta)",
                        backtest_mase=0.65, nonnegative=True)
    outlook = Outlook(direction="up", expected_change_pct=15.0, prob_increase=0.9,
                      scenarios={"pessimistic": float(lower[80].iloc[-1]), "base": float(mean.iloc[-1]),
                                 "optimistic": float(upper[80].iloc[-1])},
                      forecast_peak_date=None, future_phase="growing", confidence="high")
    return TrendReport(name=name, analysis=analysis, forecast=fc, outlook=outlook,
                       narrative="## 과거\n- 꾸준히 성장\n\n## 미래\n향후 12개월 **상승** 전망 <b>bold</b>",
                       narrative_source="rule")


@pytest.fixture
def fake_report() -> TrendReport:
    return make_fake_report()
