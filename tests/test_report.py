"""Tests for trendai.report (presentation layer).

Reports are built by hand (``conftest.make_fake_report`` and ``build_report``
below) so these tests do not depend on analysis/forecast code.
"""

from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from html.parser import HTMLParser

import numpy as np
import pandas as pd
import pytest

from trendai import report as R
from trendai.data import from_values, future_index
from trendai.types import (
    AnalysisResult, Anomaly, Changepoint, Decomposition, ForecastResult, LifecyclePhase,
    ModelScore, Outlook, TrendMetrics, TrendReport,
)

from conftest import make_fake_report

KPI_LABELS = ["현재 값", "전체 변화", "연평균 성장률 CAGR", "최근 변화", "예측 변화", "상승 확률"]


# --------------------------------------------------------------------------- builders


def build_report(values, freq: str = "M", start: str = "2020-01-01", horizon: int = 6,
                 bands: bool = True, name: str = "테스트 시리즈", phase: str = "stable",
                 changepoints: bool = False, anomalies: bool = False, none_metrics: bool = False,
                 slope: float = 0.0, narrative: str = "## 과거\n- 평평함", source: str = "rule",
                 scores: bool = True) -> TrendReport:
    s = from_values(values, start=start, freq=freq, name=name)
    n = len(s)
    trend = s.rolling(3, center=True, min_periods=1).mean()
    dec = Decomposition(trend=trend, seasonal=pd.Series(np.zeros(n), index=s.index),
                        resid=s - trend, period=1)
    last = float(s.iloc[-1])
    peak_i = int(np.argmax(s.to_numpy()))
    metrics = TrendMetrics(
        n_obs=n, freq=freq, start=s.index[0], end=s.index[-1], last_value=last,
        mean=float(s.mean()),
        total_change_pct=None if none_metrics else 12.5,
        cagr_pct=None if none_metrics else 3.1,
        recent_change_pct=None if none_metrics else -4.0,
        slope=slope, slope_pct=0.0, momentum=0.0, acceleration=0.0, volatility=0.01,
        trend_strength=0.2, seasonal_strength=0.0, mk_tau=0.0, mk_pvalue=0.9, direction="flat",
        peak_date=s.index[peak_i], peak_value=float(s.iloc[peak_i]), drawdown_pct=0.0,
    )
    cps = []
    if changepoints and n >= 3:
        k = n // 2
        cps = [Changepoint(date=s.index[k], index=k, slope_before=-0.5, slope_after=0.7,
                           kind="reversal_up", magnitude=0.8)]
    anoms = []
    if anomalies and n >= 3:
        anoms = [Anomaly(date=s.index[1], index=1, value=float(s.iloc[1]),
                         expected=float(trend.iloc[1]), zscore=-3.9)]
    analysis = AnalysisResult(name=name, series=s, freq=freq, season_length=1, decomposition=dec,
                              metrics=metrics, changepoints=cps, anomalies=anoms,
                              phase=LifecyclePhase(label=phase, confidence=0.55, reasons=["테스트 근거"]))
    idx = future_index(s.index[-1], freq, horizon)
    mean = pd.Series(last + slope * np.arange(1, horizon + 1), index=idx)
    if bands:
        sd = 0.0 if np.ptp(values) == 0 else 1.0 * np.sqrt(np.arange(1, horizon + 1))
        lower = {80: mean - 1.28 * sd, 95: mean - 1.96 * sd}
        upper = {80: mean + 1.28 * sd, 95: mean + 1.96 * sd}
        levels = (80, 95)
        scen = {"pessimistic": float(lower[80].iloc[-1]), "base": float(mean.iloc[-1]),
                "optimistic": float(upper[80].iloc[-1])}
    else:
        lower, upper, levels = {}, {}, ()
        scen = {"base": float(mean.iloc[-1])}
    ms = [ModelScore("naive", 1.0, 1.0, 2.0, 1.0, 1.0)] if scores else []
    fc = ForecastResult(mean=mean, lower=lower, upper=upper, horizon=horizon, levels=levels,
                        model_scores=ms, method="naive", backtest_mase=None if none_metrics else 1.0)
    outlook = Outlook(direction="flat", expected_change_pct=None if none_metrics else 0.0,
                      prob_increase=0.5, scenarios=scen, forecast_peak_date=None,
                      future_phase=phase, confidence="low")
    return TrendReport(name=name, analysis=analysis, forecast=fc, outlook=outlook,
                       narrative=narrative, narrative_source=source)


class _TagBalance(HTMLParser):
    """Counts open/close tags for a few structural elements."""

    TAGS = {"div", "section", "table", "tr", "td", "th", "ul", "ol", "li", "header", "main", "svg", "g", "span",
            "article", "dl", "p", "h1", "h2", "h3", "h4", "em", "strong", "code", "abbr"}

    def __init__(self) -> None:
        super().__init__()
        self.depth: dict[str, int] = {}
        self.bad: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag in self.TAGS:
            self.depth[tag] = self.depth.get(tag, 0) + 1

    def handle_startendtag(self, tag, attrs):
        pass

    def handle_endtag(self, tag):
        if tag in self.TAGS:
            self.depth[tag] = self.depth.get(tag, 0) - 1
            if self.depth[tag] < 0:
                self.bad.append(tag)


def _assert_balanced(html_text: str) -> None:
    p = _TagBalance()
    p.feed(html_text)
    assert not p.bad, f"unbalanced closing tags: {p.bad}"
    assert all(v == 0 for v in p.depth.values()), {k: v for k, v in p.depth.items() if v}


def _numbers_ok(text: str) -> None:
    assert "nan" not in text.lower().replace("nanum", "")
    assert "inf" not in re.sub(r"[A-Za-z]*inf[A-Za-z]+|[A-Za-z]+inf", "", text.lower())
    assert ">None<" not in text and " None" not in text


# --------------------------------------------------------------------------- sparkline


def test_sparkline_basic_ramp():
    sp = R.sparkline([1, 2, 3, 4, 5, 6, 7, 8])
    assert sp == "▁▂▃▄▅▆▇█"


def test_sparkline_flat_gives_middle_blocks():
    sp = R.sparkline([3.0, 3.0, 3.0, 3.0])
    assert len(sp) == 4 and len(set(sp)) == 1 and sp[0] in "▄▅"
    assert R.sparkline([7.5], width=5) == sp[0] * 5


def test_sparkline_empty_and_invalid():
    assert R.sparkline([]) == ""
    assert R.sparkline([float("nan"), float("inf")]) == ""
    assert R.sparkline([1, 2, 3], width=0) == ""
    assert R.sparkline(np.array([])) == ""


def test_sparkline_resampling_to_width():
    down = R.sparkline(range(100), width=10)
    up = R.sparkline([1, 5, 2], width=9)
    assert len(down) == 10 and len(up) == 9
    blocks = "▁▂▃▄▅▆▇█"
    ranks = [blocks.index(c) for c in down]
    assert ranks == sorted(ranks) and ranks[0] == 0 and ranks[-1] == 7
    assert up[0] == "▁" and "█" in up


def test_sparkline_drops_nan_and_accepts_series():
    s = pd.Series([1.0, np.nan, 2.0, 3.0])
    # NaN dropped; 2 sits at (2-1)/(3-1)*7 = 3.5 -> rint -> block index 4
    assert R.sparkline(s) == "▁▅█"
    assert R.sparkline(x for x in [1, 2, 3]) == "▁▅█"


# --------------------------------------------------------------------------- SVG chart


def test_svg_chart_structure(fake_report):
    svg = R.svg_chart(fake_report)
    assert svg.startswith("<svg") and svg.endswith("</svg>")
    assert 'viewBox="0 0 960 380"' in svg and 'width="100%"' in svg
    for label in ("실제값", "추세", "예측", "80% 구간", "95% 구간", "변곡점", "이상치", "현재"):
        assert label in svg, label
    assert "2021-12" in svg  # changepoint label / tooltip
    assert "tai-anom" in svg and "tai-band" in svg and "tai-trend" in svg
    ET.fromstring(svg)  # well-formed XML


def test_svg_chart_escapes_name_everywhere(fake_report):
    for w, h in ((960, 380), (600, 360), (380, 340)):
        svg = R.svg_chart(fake_report, width=w, height=h)
        assert "<script" not in svg
        assert "&lt;script&gt;alert(&quot;x&quot;)&lt;/script&gt;" in svg
        title = re.search(r"<title[^>]*>(.*?)</title>", svg).group(1)
        assert "&lt;script&gt;" in title and "<" not in title
        ET.fromstring(svg)


def test_svg_chart_y_ticks_are_nice(fake_report):
    svg = R.svg_chart(fake_report)
    labels = re.findall(r'<text class="tai-ax" x="[\d.]+" y="[\d.]+" text-anchor="end">([^<]+)</text>', svg)
    assert 4 <= len(labels) <= 6
    vals = [float(v.replace(",", "").replace("−", "-")) for v in labels]
    steps = np.diff(vals)
    assert np.allclose(steps, steps[0])


def test_svg_compact_variant_has_no_hover_layer(fake_report):
    wide = R.svg_chart(fake_report, 960, 380)
    narrow = R.svg_chart(fake_report, 380, 340)
    assert 'class="tai-hv"' in wide and 'class="tai-hv"' not in narrow
    assert 'viewBox="0 0 380 340"' in narrow


def test_nice_ticks_and_axis_labels():
    for lo, hi in [(0, 1), (28, 151), (-37.2, 12.9), (0.0012, 0.0047), (5, 5), (1e6, 4.2e6), (-1, 0)]:
        t = R._nice_ticks(lo, hi)
        assert 4 <= len(t) <= 6
        assert t[0] <= min(lo, hi) + 1e-12 and t[-1] >= max(lo, hi) - 1e-12 or lo == hi
        assert all(np.isfinite(t))
    assert R._axis_labels([0, 25000, 50000, 75000]) == ["0", "25k", "50k", "75k"]
    assert R._axis_labels([0, 1_000_000, 2_000_000]) == ["0", "1M", "2M"]
    assert R._axis_labels([-20, -10, 0, 10]) == ["−20", "−10", "0", "10"]
    assert R._axis_labels([0, 2.5, 5, 7.5]) == ["0.0", "2.5", "5.0", "7.5"]
    assert R._axis_labels([1000, 2000, 3000, 4000]) == ["1,000", "2,000", "3,000", "4,000"]


def test_date_ticks_do_not_overlap():
    t0 = pd.Timestamp("2016-01-01")
    for t1, freq in [(pd.Timestamp("2026-12-01"), "M"), (pd.Timestamp("2016-04-01"), "M"),
                     (pd.Timestamp("2018-01-01"), "D"), (pd.Timestamp("2016-03-01"), "D"),
                     (pd.Timestamp("2025-10-01"), "Q"), (pd.Timestamp("2040-01-01"), "Y"),
                     (pd.Timestamp("2019-06-30"), "W")]:
        for width in (300.0, 900.0):
            span = (t1 - t0) / pd.Timedelta(days=1)

            def x_of(t, span=span, width=width):
                return 40 + (t - t0) / pd.Timedelta(days=1) / span * width

            ticks = R._date_ticks(t0, t1, freq, x_of, 40, 40 + width, 12.0)
            assert ticks, (t1, freq, width)
            for (xa, la), (xb, lb) in zip(ticks, ticks[1:]):
                need = (R._text_width(la, 12) + R._text_width(lb, 12)) / 2
                assert xb - xa >= need, (freq, la, lb)
            if freq == "Q":
                assert any("-Q" in lab for _, lab in ticks) or all(len(lab) == 4 for _, lab in ticks)


# --------------------------------------------------------------------------- HTML page


def test_html_has_no_raw_xss(fake_report):
    html_text = R.render_html(fake_report)
    assert "<script>alert" not in html_text
    assert "<script" not in html_text.lower()  # the page ships no scripts at all
    assert "&lt;script&gt;alert(&quot;x&quot;)&lt;/script&gt; &amp; co" in html_text
    title = re.search(r"<title>(.*?)</title>", html_text).group(1)
    assert "&lt;script&gt;" in title
    # narrative: raw HTML escaped, markdown-lite rendered
    assert "&lt;b&gt;bold&lt;/b&gt;" in html_text and "<b>bold</b>" not in html_text
    assert "<strong>상승</strong>" in html_text
    assert "<h3>과거</h3>" in html_text and "<li>꾸준히 성장</li>" in html_text


def test_html_theming_and_layout(fake_report):
    html_text = R.render_html(fake_report)
    assert html_text.startswith("<!DOCTYPE html>")
    assert '<html lang="ko">' in html_text and 'name="viewport"' in html_text
    assert "@media (prefers-color-scheme: dark)" in html_text
    assert ":root{" in html_text and "--bg:" in html_text and "--c-hist:" in html_text
    assert re.search(r"body\{[^}]*background:var\(--bg\)", html_text)
    assert "max-width:1100px" in html_text
    assert "<svg" in html_text and "viewBox" in html_text
    assert 'src="http' not in html_text
    _assert_balanced(html_text)


def test_html_sections_and_content(fake_report):
    html_text = R.render_html(fake_report)
    for label in KPI_LABELS:
        assert label in html_text, label
    for needle in ("성장기", "80%", "2016-01", "2025-12", "월 단위", "규칙 기반 해석", "TrendAI",
                   "2021-12", "가속", "+0.020", "+1.20", "2018-07", "모델 리더보드", "holt_damped",
                   "MASE", "sMAPE", "ensemble(holt_damped, linear_trend, theta)", "예측 표", "80% 구간",
                   "95% 구간", "2026-12", "비관", "기준", "낙관", "분석 방법", "보장하지 않습니다",
                   "+180.0%", "+11.0%", "+35.0%", "+15.0%", "90%", "미래 단계"):
        assert needle in html_text, needle
    _numbers_ok(html_text)


def test_html_llm_source_label(fake_report):
    fake_report.narrative_source = "llm"
    assert "AI(Claude) 해석" in R.render_html(fake_report)


def test_html_handles_none_metrics():
    rep = build_report(np.linspace(10, 20, 24), none_metrics=True)
    html_text = R.render_html(rep)
    assert html_text.count("—") >= 4
    _numbers_ok(html_text)
    text = R.render_text(rep)
    assert "—" in text and "None" not in text


def test_html_empty_changepoints_and_anomalies(fake_report):
    fake_report.analysis.changepoints = []
    fake_report.analysis.anomalies = []
    html_text = R.render_html(fake_report)
    assert "변곡점)가 감지되지 않았습니다" in html_text
    assert "이상치가 감지되지 않았습니다" in html_text
    _assert_balanced(html_text)


def test_flat_series():
    rep = build_report([5.0] * 30, changepoints=True)
    svg = R.svg_chart(rep)
    ET.fromstring(svg)
    assert "nan" not in svg.lower()
    html_text = R.render_html(rep)
    _numbers_ok(html_text)
    assert "▄" * 10 in R.render_text(rep)


def test_flat_zero_series():
    rep = build_report([0.0] * 12)
    svg = R.svg_chart(rep)
    ET.fromstring(svg)
    assert "nan" not in svg.lower()


def test_negative_values():
    vals = np.concatenate([np.linspace(-60, -10, 20), np.linspace(-10, 35, 16)])
    rep = build_report(vals, slope=2.0, changepoints=True, anomalies=True, name="적자 → 흑자")
    svg = R.svg_chart(rep)
    ET.fromstring(svg)
    assert "tai-zero" in svg  # zero baseline drawn when the range spans 0
    assert "−40" in svg or "−50" in svg or "−60" in svg
    html_text = R.render_html(rep)
    assert "적자 → 흑자" in html_text and "상승 반전" in html_text
    _numbers_ok(html_text)


def test_very_short_series_n3():
    rep = build_report([3.0, 4.0, 3.5], horizon=2, changepoints=True, anomalies=True)
    svg = R.svg_chart(rep)
    ET.fromstring(svg)
    html_text = R.render_html(rep)
    _numbers_ok(html_text)
    text = R.render_text(rep)
    assert "│" in text


def test_forecast_without_bands():
    rep = build_report(np.arange(20.0), bands=False, slope=1.0)
    svg = R.svg_chart(rep)
    assert "% 구간" not in svg and "tai-band" not in svg.split("</style>")[1]
    html_text = R.render_html(rep)
    assert "80% 구간" not in html_text
    _numbers_ok(html_text)


def test_no_model_scores_and_no_backtest():
    rep = build_report(np.arange(10.0), scores=False, none_metrics=True)
    html_text = R.render_html(rep)
    assert "백테스트 점수가 없습니다" in html_text


def test_large_values_abbreviated():
    rep = build_report(np.linspace(1.2e6, 3.4e6, 40), slope=5e4)
    svg = R.svg_chart(rep)
    assert re.search(r">\d(\.\d)?M<", svg)
    assert "3,400,000" in R.render_html(rep) or "3,450,000" in R.render_html(rep)


@pytest.mark.parametrize("freq,n,label", [("D", 120, "-"), ("W", 80, "-"), ("Q", 6, "-Q"), ("Q", 40, "20"), ("Y", 15, "20")])
def test_other_frequencies(freq, n, label):
    rep = build_report(np.linspace(10, 30, n) + np.sin(np.arange(n)), freq=freq, start="2015-01-01",
                       changepoints=True)
    svg = R.svg_chart(rep)
    ET.fromstring(svg)
    axis = re.findall(r'<text class="tai-ax"[^>]*text-anchor="(?:middle|start|end)">([^<]+)</text>', svg)
    assert any(label in a for a in axis)
    html_text = R.render_html(rep)
    _numbers_ok(html_text)
    assert len(R.render_text(rep).splitlines()) > 5


def test_render_html_writes_file(tmp_path, fake_report):
    out = tmp_path / "sub" / "report.html"
    html_text = R.render_html(fake_report, path=out)
    assert out.read_text(encoding="utf-8") == html_text


def test_render_is_deterministic(fake_report):
    assert R.render_html(fake_report) == R.render_html(make_fake_report())
    assert R.render_text(fake_report) == R.render_text(make_fake_report())


# --------------------------------------------------------------------------- terminal text


@pytest.mark.parametrize("width", [24, 40, 60, 78, 100])
def test_render_text_width_and_content(fake_report, width):
    text = R.render_text(fake_report, width=width)
    lines = text.splitlines()
    assert all(R._dw(ln) <= width for ln in lines), max(lines, key=R._dw)
    assert "<script>alert" not in text and "\x1b" not in text
    assert "성장" in text and "│" in text
    if width >= 60:
        assert "+15.0%" in text and "90%" in text
    assert any(ch in text for ch in "▁▂▃▄▅▆▇█")


def test_render_text_mentions_models_and_changepoints(fake_report):
    text = R.render_text(fake_report)
    assert "holt_damped" in text and "2021-12" in text and "가속" in text
    assert "에이전트" in text


def test_render_text_strips_control_characters():
    rep = build_report(np.arange(12.0), name="evil\x1b[31mname\x07")
    text = R.render_text(rep)
    assert "\x1b" not in text and "\x07" not in text and "evil" in text


# --------------------------------------------------------------------------- compare page


def _ranking(reports):
    rows = []
    for i, r in enumerate(reports):
        rows.append({"name": r.name, "phase": r.analysis.phase.label, "momentum": 1.5 - i,
                     "recent_change_pct": r.analysis.metrics.recent_change_pct,
                     "expected_change_pct": r.outlook.expected_change_pct,
                     "prob_increase": r.outlook.prob_increase, "score": 80.0 - 20 * i})
    return pd.DataFrame(rows)


def test_compare_page_three_reports():
    reports = [make_fake_report(),
               build_report(np.linspace(50, 20, 36), name="쇠퇴 <i>중</i>", phase="declining", slope=-1.0),
               build_report([5.0] * 12, name="평탄", phase="stable")]
    ranking = _ranking(reports)
    html_text = R.render_compare_html(reports, ranking)
    assert "<script" not in html_text.lower()
    assert "&lt;script&gt;" in html_text and "&lt;i&gt;중&lt;/i&gt;" in html_text
    assert html_text.count('class="tai-chart tai-mini"') == 3
    for needle in ("상승 순위", "모멘텀", "최근 변화", "예측 변화", "상승 확률", "상승 점수",
                   "성장기", "쇠퇴기", "안정기", "트렌드 비교 · 3개 시리즈"):
        assert needle in html_text, needle
    assert "@media (prefers-color-scheme: dark)" in html_text
    _assert_balanced(html_text)
    _numbers_ok(html_text)
    # cards follow the ranking order
    first = html_text.index("<article")
    assert html_text.index("AI 에이전트", first) < html_text.index("평탄", first)


def test_compare_page_handles_missing_values_and_order(tmp_path):
    reports = [build_report(np.arange(12.0), name="a"), build_report(np.arange(12.0)[::-1], name="b")]
    ranking = pd.DataFrame([
        {"name": "b", "phase": "declining", "momentum": np.nan, "recent_change_pct": None,
         "expected_change_pct": np.nan, "prob_increase": 0.2, "score": 31.0},
        {"name": "a", "phase": "growing", "momentum": 0.4, "recent_change_pct": 5.0,
         "expected_change_pct": 3.0, "prob_increase": 0.7, "score": 64.0},
    ])
    out = tmp_path / "cmp.html"
    html_text = R.render_compare_html(reports, ranking, path=out)
    assert out.exists()
    _numbers_ok(html_text)
    assert html_text.count("—") >= 3
    first = html_text.index("<article")
    assert html_text.index(">b<", first) < html_text.index(">a<", first)


def test_compare_page_empty_inputs():
    html_text = R.render_compare_html([], pd.DataFrame())
    assert "<!DOCTYPE html>" in html_text and "순위 데이터가 없습니다" in html_text
    _assert_balanced(html_text)


def test_compare_page_ships_chart_styles():
    reports = [make_fake_report()]
    html_text = R.render_compare_html(reports, _ranking(reports))
    style = re.search(r"<style>(.*?)</style>", html_text, re.DOTALL).group(1)
    for cls in (".tai-hist", ".tai-fc", ".tai-band", ".tai-region"):
        assert cls in style, cls


def test_probability_formatting_near_bounds():
    # Exactly 1.0 / 0.0 used to print '100%' / '0%'.  They are model estimates (forecast
    # returns exactly 1.0 e.g. for daily breakout), never certainty, and the narrative
    # (narrative.format_prob) prints '>99%' / '<1%' for them: the report now does the same.
    assert R._fmt_prob(0.997) == ">99%" and R._fmt_prob(1.0) == ">99%"
    assert R._fmt_prob(0.001) == "<1%" and R._fmt_prob(0.0) == "<1%"
    assert R._fmt_prob(None) == "—" and R._fmt_prob(float("nan")) == "—"


# --------------------------------------------------------------------------- adversarial-review regressions


_TICK_RE = re.compile(r'<text class="tai-ax" x="[\d.]+" y="([\d.]+)" text-anchor="end">([^<]+)</text>')


def _y_ticks(svg: str) -> list[tuple[float, str]]:
    return [(float(y), lab) for y, lab in _TICK_RE.findall(svg)]


def _tick_value(label: str) -> float:
    s = label.replace(",", "").replace("−", "-")
    mult = {"k": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}.get(s[-1:], 1.0)
    return float(s[:-1] if mult != 1.0 else s) * mult


def _set_forecast(rep: TrendReport, mean: np.ndarray, half_width: np.ndarray) -> None:
    fc = rep.forecast
    fc.mean = pd.Series(np.asarray(mean, dtype=float), index=fc.mean.index)
    fc.lower = {80: fc.mean - 0.65 * half_width, 95: fc.mean - half_width}
    fc.upper = {80: fc.mean + 0.65 * half_width, 95: fc.mean + half_width}
    fc.levels = (80, 95)


def test_nice_ticks_property_across_magnitudes():
    rng = np.random.default_rng(42)
    for _ in range(3000):
        mag = 10 ** rng.uniform(-14, 16)
        lo = rng.uniform(-1, 1) * mag
        hi = lo + abs(rng.normal()) * mag * 10 ** rng.uniform(-4, 1)
        t = R._nice_ticks(lo, hi)
        labels = R._axis_labels(t)
        assert 4 <= len(t) <= 6, (lo, hi, t)
        assert t[0] <= lo + 1e-9 * abs(lo) and t[-1] >= hi - 1e-9 * abs(hi), (lo, hi, t)
        steps = np.diff(t)
        assert np.allclose(steps, steps[0], rtol=1e-6)
        mant = steps[0] / 10 ** math.floor(math.log10(steps[0]))
        assert min(abs(mant - m) for m in (1, 2, 2.5, 3, 4, 5, 10)) < 1e-6, (steps[0], mant)
        assert len(set(labels)) == len(labels), (t, labels)  # never two identical tick labels


def test_nice_ticks_extreme_ranges_do_not_crash():
    for lo, hi in [(-1e308, 1e308), (0, 9e307), (1e-320, 3e-320), (float("nan"), 1.0), (5.0, float("inf"))]:
        t = R._nice_ticks(lo, hi)
        assert 4 <= len(t) <= 6 and all(math.isfinite(v) for v in t)


def test_axis_labels_tiny_and_huge():
    assert R._axis_labels([0, 5e-8, 1e-7, 1.5e-7]) == ["0", "5e−8", "1e−7", "1.5e−7"]
    assert R._axis_labels([0, 5e13, 1e14, 1.5e14]) == ["0", "50T", "100T", "150T"]
    labs = R._axis_labels([1.965e-09, 1.97e-09, 1.975e-09, 1.98e-09])
    assert len(set(labs)) == 4 and all("e−9" in lab for lab in labs)


def test_tiny_values_are_not_printed_as_zero():
    assert R._fmt_num(4.4e-9) == "4.4e−9" and R._fmt_num(-3e-7) == "−3e−7"
    assert R._fmt_num(0.00012) == "0.00012" and R._fmt_num(0.0) == "0"
    rep = build_report(np.linspace(1e-9, 3e-9, 24), slope=1e-11)
    html_text = R.render_html(rep)
    assert "0.00000000" not in html_text and "e−9" in html_text
    labels = [lab for _, lab in _y_ticks(R.svg_chart(rep))]
    assert len(set(labels)) == len(labels) >= 4
    assert "e−9" in R.render_text(rep)


def test_huge_values_abbreviated_in_tiles_with_exact_tooltip():
    assert R._fmt_short(1.5e12) == "1.50T" and R._fmt_short(999_400) == "999,400"
    assert R._fmt_short(999_700) == "1.00M" and R._fmt_short(999.7e6) == "1.00B"
    assert R._fmt_short(-2.345e9) == "−2.35B" and R._fmt_short(None) == "—"
    rep = build_report(np.linspace(1.0e14, 1.2e14, 24), slope=1e11)
    html_text = R.render_html(rep)
    kpi = re.search(r'<div class="kpi"><div class="lbl">현재 값</div><div class="val" title="([^"]+)">(.*?)</div>',
                    html_text)
    assert kpi.group(1) == "120,000,000,000,000"          # exact value on hover
    assert ">120T<" in kpi.group(2)                       # fits the tile
    scen = re.findall(r'<div class="v" title="([^"]+)">([^<]+)</div>', html_text)
    assert len(scen) == 3 and all(v.endswith("T") and "," in t for t, v in scen)
    assert all(lab.endswith("T") or lab == "0" for _, lab in _y_ticks(R.svg_chart(rep)))


def test_constant_series_axis_not_zoomed_into_noise():
    rep = build_report([7.0] * 24)
    _set_forecast(rep, np.full(6, 7.0), np.full(6, 5e-5))  # numerically-noisy bands
    vals = [_tick_value(lab) for _, lab in _y_ticks(R.svg_chart(rep))]
    assert vals[-1] - vals[0] >= 0.1 * 7.0, vals
    assert vals[0] <= 7.0 <= vals[-1]


def test_zero_series_with_tiny_bands():
    rep = build_report([0.0] * 24)
    _set_forecast(rep, np.zeros(6), np.full(6, 6e-9))
    rep.forecast.lower = {k: v.clip(lower=0) for k, v in rep.forecast.lower.items()}  # nonnegative fc
    svg = R.svg_chart(rep)
    ticks = _y_ticks(svg)
    labels = [lab for _, lab in ticks]
    assert len(set(labels)) == len(labels), labels
    assert _tick_value(labels[0]) == 0.0  # non-negative data: axis starts at 0
    ET.fromstring(svg)


def test_nonnegative_series_axis_ignores_small_trend_dip():
    vals = np.r_[np.linspace(60, 200, 8), [150.0, 8.0]]
    rep = build_report(vals)
    trend = rep.analysis.decomposition.trend.copy()
    trend.iloc[-1] = -1.0  # smoothing artefact below zero
    rep.analysis.decomposition.trend = trend
    _set_forecast(rep, np.full(6, 8.0), np.linspace(5, 9, 6))
    rep.forecast.lower = {k: v.clip(lower=0) for k, v in rep.forecast.lower.items()}
    labels = [lab for _, lab in _y_ticks(R.svg_chart(rep))]
    assert _tick_value(labels[0]) == 0.0, labels
    # genuinely negative data still gets a negative axis
    rep2 = build_report(np.linspace(-60, 35, 30), slope=2.0)
    assert _tick_value(_y_ticks(R.svg_chart(rep2))[0][1]) < 0


def test_chart_geometry_matches_values(fake_report):
    """Independent check of the x/y mapping: invert the axis ticks and read values back."""
    svg = R.svg_chart(fake_report, 960, 380)
    ticks = _y_ticks(svg)
    (ya, la), (yb, lb) = ticks[0], ticks[-1]
    ya, yb = ya - 12 * 0.35, yb - 12 * 0.35  # labels sit fs*0.35 below their grid line (fs=12 at 960px)
    va, vb = _tick_value(la), _tick_value(lb)

    def value_at(y: float) -> float:
        return va + (y - ya) / (yb - ya) * (vb - va)

    hist = re.search(r'class="tai-hist" d="([^"]+)"', svg.split("</defs>")[1]).group(1)
    pts = [tuple(map(float, p.split(","))) for p in re.split(r"[ML]", hist) if p]
    assert len(pts) == len(fake_report.analysis.series)
    x_last, y_last = pts[-1]
    assert value_at(y_last) == pytest.approx(fake_report.analysis.metrics.last_value, abs=0.15)
    assert value_at(pts[0][1]) == pytest.approx(float(fake_report.analysis.series.iloc[0]), abs=0.15)
    # forecast end point
    fc = re.search(r'class="tai-fc" d="([^"]+)"', svg.split("</defs>")[1]).group(1)
    fpts = [tuple(map(float, p.split(","))) for p in re.split(r"[ML]", fc) if p]
    assert fpts[0] == pts[-1]  # forecast continues from the last observation
    assert value_at(fpts[-1][1]) == pytest.approx(float(fake_report.forecast.mean.iloc[-1]), abs=0.15)
    # x: the "현재" divider sits on the last observation; changepoint between the year ticks
    now = re.search(r'class="tai-now" d="M([\d.]+),', svg)
    assert float(now.group(1)) == pytest.approx(x_last, abs=0.11)
    xt = dict((lab, float(x)) for x, lab in re.findall(
        r'<text class="tai-ax" x="([\d.]+)" y="[\d.]+" text-anchor="middle">(\d{4})</text>', svg))
    x16, x26 = xt["2016"], xt["2026"]
    days = (pd.Timestamp("2021-12-01") - pd.Timestamp("2016-01-01")).days
    total = (pd.Timestamp("2026-01-01") - pd.Timestamp("2016-01-01")).days
    cp_x = float(re.search(r'class="tai-cp" d="M([\d.]+),', svg).group(1))
    assert cp_x == pytest.approx(x16 + days / total * (x26 - x16), abs=0.25)


def test_long_history_beyond_292_years():
    """ns-resolution Timedelta arithmetic overflowed for spans > ~292 years."""
    rep = build_report(np.linspace(1, 100, 326), freq="Y", start="1700-01-01", horizon=5, changepoints=True)
    svg = R.svg_chart(rep)
    ET.fromstring(svg)
    years = re.findall(r'text-anchor="(?:middle|start|end)">(\d{4})</text>', svg)
    assert years[0] == "1700" and int(years[-1]) >= 2000
    R.render_html(rep)
    assert "1700" in R.render_text(rep)


def test_mixed_timestamp_resolutions():
    rep = build_report(np.arange(40.0), freq="Y", start="1700-01-01", changepoints=True, anomalies=True)
    rep.analysis.changepoints[0].date = pd.Timestamp("1720-01-01").as_unit("ns")
    s = rep.analysis.series.copy()
    rep.forecast.mean.index = rep.forecast.mean.index.as_unit("ns")
    rep.analysis.series = s
    svg = R.svg_chart(rep)
    assert "1720" in svg
    R.render_html(rep)


def test_control_characters_and_surrogates_in_name(tmp_path):
    rep = build_report(np.arange(12.0), name="evil\x1b[31mname\x07\x0b\udcff")
    for w in (960, 380):
        ET.fromstring(R.svg_chart(rep, w, 340))  # standalone SVG stays well-formed XML
    out = tmp_path / "x.html"
    html_text = R.render_html(rep, path=out)    # lone surrogate used to raise UnicodeEncodeError
    assert out.read_text(encoding="utf-8") == html_text
    assert "\x1b" not in html_text and "\x07" not in html_text and "evil" in html_text
    R.render_text(rep).encode("utf-8")
    R.render_compare_html([rep], pd.DataFrame([{"name": rep.name, "score": 50.0}])).encode("utf-8")


def test_xss_in_every_string_field():
    rep = make_fake_report()
    rep.analysis.phase.label = "<img src=x onerror=alert(1)>"
    rep.analysis.phase.reasons = ["<script>alert(2)</script>"]
    rep.analysis.changepoints[0].kind = "<script>alert(3)</script>"
    rep.forecast.model_scores[0].name = "<script>alert(4)</script>"
    rep.forecast.method = "<img src=x onerror=alert(5)>"
    rep.narrative = "## <script>alert(6)</script>\n- **<img src=x onerror=alert(7)>**\n1. <svg onload=alert(8)>"
    rep.narrative_source = "<script>"
    rep.outlook.confidence = "<script>alert(9)</script>"
    rep.outlook.direction = "<script>"
    rep.outlook.future_phase = "<img src=y>"
    ranking = pd.DataFrame([{"name": rep.name, "phase": "<script>alert(10)</script>", "momentum": 1.0,
                             "recent_change_pct": 1.0, "expected_change_pct": 1.0, "prob_increase": 0.5,
                             "score": 55.0}])
    outputs = [R.render_html(rep), R.render_compare_html([rep], ranking)]
    outputs += [R.svg_chart(rep, w, h) for w, h in ((960, 380), (600, 360), (380, 340))]
    for out in outputs:
        low = out.lower()
        assert "<script" not in low and "<img" not in low and "<svg onload" not in low
    text = R.render_text(rep)
    assert "<" not in text and ">" not in text


def test_render_text_respects_narrow_widths(fake_report):
    for w in (10, 12, 15, 19):
        lines = R.render_text(fake_report, width=w).splitlines()
        assert max(R._dw(ln) for ln in lines) <= w
    assert max(R._dw(ln) for ln in R.render_text(fake_report, width=3).splitlines()) <= 10


def test_render_text_long_horizon_sparkline_fits():
    rep = build_report([1.0, 2.0, 3.0], horizon=200, slope=0.1)
    for w in (20, 40, 78):
        spark = [ln for ln in R.render_text(rep, width=w).splitlines() if "│" in ln]
        assert len(spark) == 1
        assert R._dw(spark[0]) <= w and "…" not in spark[0]  # used to overflow and get cut


def test_compare_accepts_mapping_of_reports():
    reps = {"a": build_report(np.arange(12.0), name="a"), "b": build_report(np.arange(12.0)[::-1], name="b")}
    html_text = R.render_compare_html(reps, pd.DataFrame())
    assert html_text.count("<article") == 2


def test_recent_change_caption_matches_available_history():
    short = R.render_html(build_report(np.linspace(1, 2, 10)))   # monthly n=10 -> lag capped at 9
    assert "최근 9개월 추세 변화" in short and "최근 12개월" not in short
    full = R.render_html(make_fake_report())
    assert "최근 12개월 추세 변화" in full
    yearly = R.render_html(build_report(np.linspace(1, 2, 10), freq="Y"))
    assert "최근 1년 추세 변화" in yearly


def test_momentum_shows_unit(fake_report):
    html_text = R.render_html(fake_report)
    assert "+2.50%p/월" in html_text
    assert "모멘텀 +2.50%p/월" in R.render_text(fake_report, width=100)


def test_decimation_preserves_envelope_and_bounds_size():
    x = np.linspace(0, 100, 100_000)
    y = np.sin(x * 7) + (x > 50) * 3
    y[12345] = 50.0
    xd, yd = R._decimate(x, y)
    assert len(xd) < 1000
    assert yd.max() == y.max() and yd.min() == y.min()
    assert (xd[0], yd[0], xd[-1], yd[-1]) == (x[0], y[0], x[-1], y[-1])
    assert np.all(np.diff(xd) >= 0)
    short_x, short_y = np.arange(50.0), np.arange(50.0)
    assert R._decimate(short_x, short_y)[0] is short_x or len(R._decimate(short_x, short_y)[0]) == 50
    # a long daily series renders a reasonably small page
    vals = np.cumsum(np.random.default_rng(0).normal(size=60_000)) + 1000
    rep = build_report(vals, freq="D", start="1850-01-01", horizon=30)
    html_text = R.render_html(rep)
    assert len(html_text) < 1_500_000


def test_inputs_are_not_mutated(fake_report):
    import copy
    before = copy.deepcopy(fake_report)
    R.render_html(fake_report)
    R.render_text(fake_report, width=40)
    R.svg_chart(fake_report, 380, 300)
    R.render_compare_html([fake_report], pd.DataFrame([{"name": fake_report.name, "score": 50.0}]))
    a, b = fake_report, before
    assert a.analysis.series.equals(b.analysis.series)
    assert a.analysis.decomposition.trend.equals(b.analysis.decomposition.trend)
    assert a.forecast.mean.equals(b.forecast.mean)
    assert all(a.forecast.lower[k].equals(b.forecast.lower[k]) for k in a.forecast.lower)
    assert a.analysis.changepoints == b.analysis.changepoints and a.outlook == b.outlook


@pytest.mark.parametrize("w,h", [(1, 1), (-5, 10), (240, 160), (4000, 1500)])
def test_svg_extreme_sizes(fake_report, w, h):
    svg = R.svg_chart(fake_report, w, h)
    ET.fromstring(svg)
    vb = re.search(r'viewBox="0 0 (\d+) (\d+)"', svg)
    assert int(vb.group(1)) >= 240 and int(vb.group(2)) >= 160


# --------------------------------------------------------------------------- fixer round 2 regressions


import html as _html

_LBL_RE = re.compile(r'<text class="tai-lbl[^"]*" x="([\d.]+)" y="([\d.]+)" font-size="([\d.]+)" '
                     r'text-anchor="(\w+)">([^<]+)</text>')


def _label_boxes(svg: str) -> list[tuple[str, float, float, float]]:
    """(text, x0, x1, baseline y) of every top label of a chart."""
    out = []
    for x, y, fs, anchor, txt in _LBL_RE.findall(svg):
        txt = _html.unescape(txt)
        x, fs = float(x), float(fs)
        w = R._text_width(txt, fs)
        x0 = x if anchor == "start" else (x - w if anchor == "end" else x - w / 2)
        out.append((txt, x0, x0 + w, float(y)))
    return out


def _vertical_lines(svg: str) -> list[float]:
    """x of every changepoint line and of the "현재" divider (legend swatches use a lowercase ``v``)."""
    xs_ = [float(v) for v in re.findall(r'class="tai-cp" d="M([\d.]+),[\d.]+V', svg)]
    return xs_ + [float(v) for v in re.findall(r'class="tai-now" d="M([\d.]+),', svg)]


def _with_changepoints(rep: TrendReport, idx: list[int], kinds: list[str] | None = None) -> TrendReport:
    s = rep.analysis.series
    kinds = kinds or ["acceleration"] * len(idx)
    rep.analysis.changepoints = [
        Changepoint(date=s.index[k], index=k, slope_before=0.2, slope_after=0.9, kind=kd, magnitude=0.5)
        for k, kd in zip(idx, kinds)]
    return rep


def test_markdown_nested_and_ordered_lists_and_emphasis():
    out = R._markdown_lite("- 구조 변화가 2번 감지되었습니다:\n  - **2022-04** 가속\n  - **2023-10** 가속\n- 끝")
    assert out == ("<ul><li>구조 변화가 2번 감지되었습니다:<ul><li><strong>2022-04</strong> 가속</li>"
                   "<li><strong>2023-10</strong> 가속</li></ul></li><li>끝</li></ul>")
    # a lead-in line ending with ':' followed by an indented list; numbered items -> <ol>
    out = R._markdown_lite("요약:\n  - a\n  - b\n\n1. 첫째\n2. 둘째\n\n*강조*입니다, a*b*c, 2 * 3, `holt_damped`")
    assert out.startswith("<p>요약:</p><ul><li>a</li><li>b</li></ul><ol><li>첫째</li><li>둘째</li></ol>")
    assert "<em>강조</em>입니다" in out and "a*b*c" in out and "2 * 3" in out
    assert "<code>holt_damped</code>" in out
    assert R._markdown_lite("3. x\n4. y").startswith('<ol start="3">')
    # three levels deep, dedent back to the middle and to the top; tabs count as 4 spaces
    out = R._markdown_lite("- a\n  - b\n    - c\n  - d\n- e\n\t- f")
    assert out == ("<ul><li>a<ul><li>b<ul><li>c</li></ul></li><li>d</li></ul></li>"
                   "<li>e<ul><li>f</li></ul></li></ul>")
    # an indented continuation line stays in its item
    assert R._markdown_lite("- 첫 줄\n  이어지는 줄") == "<ul><li>첫 줄 이어지는 줄</li></ul>"
    # still escaped first: nested items, emphasis and code cannot inject markup
    evil = R._markdown_lite("- x\n  - *<img src=x onerror=alert(1)>*\n  1. `<script>alert(2)</script>`")
    assert "<img" not in evil and "<script" not in evil
    assert "<em>&lt;img src=x onerror=alert(1)&gt;</em>" in evil
    assert "<code>&lt;script&gt;alert(2)&lt;/script&gt;</code>" in evil
    for txt in ("- a\n    - b\n  - c\n- d", "1. a\n   - b\n2. c", "  - x\n- y\n\n## h\n- z\n  - w"):
        _assert_balanced(R._markdown_lite(txt))


def test_narrative_nested_bullets_rendered_in_page(fake_report):
    fake_report.narrative = "## 과거\n- 변곡점 2개:\n  - **2022-04** 성장 가속\n  - **2023-10** 하락 가속\n1. 하나"
    html_text = R.render_html(fake_report)
    assert "<li>변곡점 2개:<ul><li><strong>2022-04</strong> 성장 가속</li>" in html_text
    assert "<ol><li>하나</li></ol>" in html_text
    _assert_balanced(html_text)


def test_skill_shown_in_outlook_leaderboard_and_text(fake_report):
    fake_report.forecast.skill = 0.65
    fake_report.forecast.backtest_mase = 1.36     # multi-step MASE above 1 although it beats naive
    html_text = R.render_html(fake_report)
    assert "계절 단순 예측과 비슷하거나 부정확" not in html_text
    # outlook card fact with an explanatory tooltip
    m = re.search(r'<dt><abbr title="([^"]+)">단순 예측 대비 오차</abbr></dt><dd class="pos">−35%</dd>', html_text)
    assert m and "단순 기준 예측" in m.group(1)
    # leaderboard headline: skill first, MASE only as a technical note
    assert '<strong class="pos">−35%</strong>' in html_text
    assert "가장 나은 단순 기준 예측보다 약 35% 작습니다" in html_text
    assert "백테스트 MASE 1.36" in html_text and "1을 넘기 쉬우므로" in html_text
    # per-model error versus the best naive row (holt_damped MAE 2.0 vs naive 6.0)
    row = re.search(r"holt_<wbr>damped</span>.*?</tr>", html_text).group(0)
    assert "−67%" in row
    naive_row = re.search(r'<span class="mname">naive</span>.*?</tr>', html_text).group(0)
    assert "기준" in naive_row
    text = R.render_text(fake_report, width=100)
    assert "단순 예측 대비 오차 −35%" in text and "MASE 1.36" in text
    # worse than naive
    fake_report.forecast.skill = 1.3
    html_text = R.render_html(fake_report)
    assert '<dd class="neg">+30%</dd>' in html_text and "낫지 않았습니다" in html_text


def test_skill_missing_shows_dash(fake_report):
    fake_report.forecast.skill = None
    html_text = R.render_html(fake_report)
    assert re.search(r"단순 예측 대비 오차</abbr></dt><dd>—</dd>", html_text)
    assert "비교한 결과가 없어" in html_text
    assert "단순 예측 대비 오차 —" in R.render_text(fake_report, width=100)
    _numbers_ok(html_text)
    fake_report.forecast.skill = float("nan")  # non-finite is treated like None
    assert re.search(r"단순 예측 대비 오차</abbr></dt><dd>—</dd>", R.render_html(fake_report))


def test_compare_momentum_has_unit():
    reports = [make_fake_report(), build_report(np.linspace(50, 20, 36), name="주간", freq="W", phase="declining")]
    html_text = R.render_compare_html(reports, _ranking(reports))
    assert "모멘텀 (%p/기간)" in html_text
    assert "+1.50</span><span class=unit>%p/월</span>" in html_text      # monthly report
    assert "+0.50</span><span class=unit>%p/주</span>" in html_text      # weekly report
    # a ranking row without a matching report still shows the percentage-point unit
    rk = pd.DataFrame([{"name": "unknown", "momentum": 2.0, "score": 50.0}])
    assert "+2.00</span><span class=unit>%p</span>" in R.render_compare_html([], rk)


def _spiky_report(spike: float, n: int = 200) -> TrendReport:
    base = 100 + 10 * np.sin(np.arange(n) * 2 * np.pi / 7)
    vals = base.copy()
    spikes = {50: spike, 51: spike * 0.4, 120: spike * 0.75, n - 1: 140.0}  # the last point is flagged too
    for i, v in spikes.items():
        vals[i] = v
    rep = build_report(vals, freq="D", horizon=14, slope=0.0)
    s = rep.analysis.series
    # a robust decomposition's trend does not follow one-day spikes
    rep.analysis.decomposition.trend = pd.Series(np.full(n, 100.0), index=s.index)
    rep.analysis.anomalies = [Anomaly(date=s.index[i], index=i, value=float(vals[i]), expected=100.0,
                                      zscore=9.0) for i in spikes]
    _set_forecast(rep, np.full(14, 105.0), np.linspace(8, 20, 14))
    return rep


def test_flagged_spikes_do_not_squash_the_axis():
    rep = _spiky_report(2000.0)
    svg = R.svg_chart(rep)
    vals = [_tick_value(lab) for _, lab in _y_ticks(svg)]
    assert vals[-1] <= 200, vals          # was 0..2,000: the level and the forecast used <10% of the plot
    d = R._ChartData(rep)
    lo, hi = d.y_extent()
    normal = d.hist_y[~d.anom_mask]
    assert normal.min() >= lo and normal.max() <= hi       # every unflagged observation stays visible
    assert lo <= d.last_y <= hi                            # the last observation (flagged) is kept too
    # out-of-range spikes: an edge marker with a value label each (the 2-day spike is one group)
    assert svg.count('class="tai-spike"') == 2
    labels = [t for t, *_ in _label_boxes(svg)]
    assert "2,000" in labels and "1,500" in labels
    assert "축 범위 밖" in svg                               # tooltip says why it sits on the edge
    assert "축 범위를 크게 벗어난 이상치" in R.render_html(rep)
    ET.fromstring(svg)
    for w, h in ((600, 360), (380, 340)):
        ET.fromstring(R.svg_chart(rep, w, h))


def test_moderate_anomalies_stay_inside_the_axis():
    rep = _spiky_report(135.0)
    svg = R.svg_chart(rep)
    assert 'class="tai-spike"' not in svg and svg.count('class="tai-anom"') >= 4  # rings (legend has one)
    vals = [_tick_value(lab) for _, lab in _y_ticks(svg)]
    assert vals[-1] >= 140
    assert "축 범위를 크게 벗어난" not in R.render_html(rep)


def test_changepoint_labels_are_never_crossed_by_vertical_lines():
    rng = np.random.default_rng(7)
    checked = 0
    for trial in range(40):
        n = int(rng.integers(40, 160))
        freq = ["M", "W", "D", "Q"][trial % 4]
        k = int(rng.integers(2, 8))
        idx = sorted(set(int(i) for i in rng.integers(2, n - 1, size=k)))
        rep = _with_changepoints(build_report(np.linspace(10, 30, n) + rng.normal(0, 1, n), freq=freq,
                                              horizon=int(rng.integers(3, 30)), slope=0.2), idx)
        for w, h in ((960, 380), (600, 360), (380, 340)):
            svg = R.svg_chart(rep, w, h)
            lines = _vertical_lines(svg)
            boxes = _label_boxes(svg)
            assert [b[0] for b in boxes].count("현재") == 1
            for txt, x0, x1, _y in boxes:
                assert 0 <= x0 and x1 <= w, (txt, x0, x1, w)
                if txt != "현재":   # "현재" may only be crossed when boxed in on both sides
                    assert not [x for x in lines if x0 + 1 < x < x1 - 1], (trial, w, txt, x0, x1, lines)
            for i, a in enumerate(boxes):           # labels in the same row never overlap
                for b in boxes[i + 1:]:
                    if a[3] == b[3]:
                        assert a[2] <= b[1] or b[2] <= a[1], (a, b)
            # every changepoint line carries its number (the key to the HTML table)
            nums = re.findall(r'class="tai-cpn"[^>]*>(\d+)</text>', svg)
            assert nums == [str(i) for i in range(1, len(idx) + 1)]
            checked += 1
    assert checked == 120


def test_compact_chart_labels_from_verifier_repro():
    """fad-like monthly series with 5 breaks 1-2 years apart, rendered at phone width."""
    rep = _with_changepoints(build_report(np.linspace(10, 90, 120), slope=0.5), [28, 46, 63, 75, 95])
    svg = R.svg_chart(rep, 380, 340)
    lines = _vertical_lines(svg)
    for txt, x0, x1, _ in _label_boxes(svg):
        assert not [x for x in lines if x0 + 1 < x < x1 - 1], (txt, x0, x1, lines)
    assert re.findall(r'class="tai-cpn"[^>]*>(\d+)</text>', svg) == ["1", "2", "3", "4", "5"]
    html_text = R.render_html(rep)
    assert all(f'<span class="cpn" aria-label="변곡점 {i}">{i}</span>' in html_text for i in range(1, 6))
    # the "현재" label moves to the free side instead of being struck by a break just left of it
    rep2 = _with_changepoints(build_report(np.linspace(10, 90, 120), slope=0.5), [100, 112])
    svg2 = R.svg_chart(rep2, 380, 340)
    now = [b for b in _label_boxes(svg2) if b[0] == "현재"][0]
    assert not [x for x in _vertical_lines(svg2) if now[1] + 1 < x < now[2] - 1]


def test_changepoint_kind_carries_direction():
    rep = build_report(np.linspace(100, 10, 60), slope=-1.0, changepoints=True)
    cp = rep.analysis.changepoints[0]
    cp.kind, cp.slope_before, cp.slope_after = "acceleration", -1.18, -3.38
    svg, html_text, text = R.svg_chart(rep), R.render_html(rep), R.render_text(rep, width=100)
    assert "하락 가속" in svg and "하락 가속" in html_text and "하락 가속" in text
    assert re.search(r'background:#d03b3b"></span>하락 가속', html_text)        # red, not "good" green
    cp.kind, cp.slope_before, cp.slope_after = "deceleration", -3.38, -0.52
    out = R.render_html(rep) + R.render_text(rep, width=100) + R.svg_chart(rep)
    assert "하락세 둔화" in out and "감속" not in out
    cp.kind, cp.slope_before, cp.slope_after = "deceleration", 2.0, 0.5
    assert "성장 둔화" in R.render_text(rep, width=100)
    cp.kind, cp.slope_before, cp.slope_after = "acceleration", 0.5, 2.0
    assert "성장 가속" in R.render_text(rep, width=100)
    cp.slope_after = float("nan")                                  # undefined slope: plain label
    assert "가속" in R.render_text(rep, width=100)


def test_text_change_basis_and_yearly_unit(fake_report):
    text = R.render_text(fake_report, width=100)
    assert "예측 변화 +15.0% (향후 12개월 평균, 최근 12개월 평균 대비)" in text
    assert "개월 후" not in text
    assert "향후 12개월 평균 · <span class=\"nw\">최근 12개월 평균 대비</span>" in R.render_html(fake_report)
    q = build_report(np.linspace(10, 30, 40), freq="Q", horizon=8)
    q.analysis.season_length = 4
    assert "(예측 마지막 4분기 평균, 최근 4분기 평균 대비)" in R.render_text(q, width=100)
    y = build_report(np.linspace(10, 30, 8), freq="Y", horizon=5)
    ty, hy = R.render_text(y, width=100), R.render_html(y)
    assert "연 단위 8개" in ty and "연 단위 · 관측 8개" in hy
    assert "년 단위" not in ty + hy
    assert "(2032 예측값, 최근 값 대비)" in ty        # yearly: season 1 -> horizon-end value vs latest value


def test_diagnosis_card_does_not_repeat_the_narratives_reasons(fake_report):
    reasons = fake_report.analysis.phase.reasons
    fake_report.narrative = "## 현재\n" + "\n".join(f"- 근거: {r}" for r in reasons)
    html_text = R.render_html(fake_report)
    assert all(html_text.count(r) == 1 for r in reasons)
    assert '<ul class="reasons">' not in html_text and "판단 근거는 ‘AI 트렌드 해석’" in html_text
    # an LLM narrative that does not quote them: the card keeps its reasons
    fake_report.narrative = "## 현재\n- 성장 중입니다."
    html_text = R.render_html(fake_report)
    assert '<ul class="reasons">' in html_text and all(r in html_text for r in reasons)


def test_leaderboard_model_names_break_only_at_separators(fake_report):
    fake_report.forecast.model_scores[1].name = "holt_winters@log"
    html_text = R.render_html(fake_report)
    assert '<span class="mname">holt_<wbr>winters<wbr>@log</span>' in html_text
    assert re.search(r"\.mname\{[^}]*overflow-wrap:break-word", html_text)
    assert not re.search(r"\.mname\{[^}]*anywhere", html_text)
    assert '<th class="num hide-sm">sMAPE</th>' in html_text


# --------------------------------------------------------------------------- round 3 regressions


def test_probabilities_never_read_as_certain_anywhere(fake_report):
    """Exactly 1.0 / 0.0 (daily breakout, yearly growing n=8) must read like the narrative."""
    from trendai.narrative import format_prob

    for p, shown, never in ((1.0, ">99%", "100%"), (0.0, "<1%", None)):
        fake_report.outlook.prob_increase = p
        html_text = R.render_html(fake_report)
        text = R.render_text(fake_report, width=100)
        cmp_html = R.render_compare_html([fake_report], _ranking([fake_report]))
        assert R._fmt_prob(p) == format_prob(p) == shown
        assert f"<span>{R._esc(shown)}</span>" in html_text                    # KPI tile
        assert f"상승 확률 {R._esc(shown)}" in html_text                         # outlook card
        assert f"상승 확률 {shown}" in text
        assert f'<td class="num hide-sm">{R._esc(shown)}</td>' in cmp_html     # ranking row
        visible = re.sub(r"<[^>]+>", " ", re.sub(r"<style>.*?</style>", "", html_text + cmp_html, flags=re.S))
        if never:
            assert never not in visible + text
        assert not re.search(r"상승 확률\s+(0|100)%", visible + text)


def _seasonal_report(horizon: int = 6, seasonal_strength: float = 0.96) -> TrendReport:
    """Flat monthly series with a +-10 seasonal swing that ends in its low season."""
    n = 120
    t = np.arange(n)
    seas = 10 * np.sin(2 * np.pi * t / 12)
    trend = np.linspace(50.6, 50.0, n)                       # trend 1.2% below its own high
    rep = build_report(trend + seas, freq="M", horizon=horizon)
    s = rep.analysis.series
    rep.analysis.season_length = 12
    rep.analysis.decomposition = Decomposition(trend=pd.Series(trend, index=s.index),
                                               seasonal=pd.Series(seas, index=s.index),
                                               resid=pd.Series(np.zeros(n), index=s.index), period=12)
    m = rep.analysis.metrics
    m.seasonal_strength = seasonal_strength
    i_pk = int(np.argmax(s.to_numpy()))
    m.peak_value, m.peak_date = float(s.iloc[i_pk]), s.index[i_pk]
    m.drawdown_pct = float((s.iloc[-1] - s.iloc[i_pk]) / s.iloc[i_pk] * 100)   # a seasonal low, ~ -25 %
    # forecast: the seasonal pattern continues; scenarios are horizon-end values
    k = np.arange(n, n + horizon)
    mean = pd.Series(50.0 + 10 * np.sin(2 * np.pi * k / 12), index=rep.forecast.mean.index)
    sd = np.sqrt(np.arange(1, horizon + 1))
    rep.forecast.mean = mean
    rep.forecast.lower = {80: mean - 1.28 * sd, 95: mean - 1.96 * sd}
    rep.forecast.upper = {80: mean + 1.28 * sd, 95: mean + 1.96 * sd}
    rep.outlook.scenarios = {"pessimistic": float(rep.forecast.lower[80].iloc[-1]), "base": float(mean.iloc[-1]),
                             "optimistic": float(rep.forecast.upper[80].iloc[-1])}
    rep.outlook.prob_increase, rep.outlook.expected_change_pct = 0.53, 0.4
    return rep


def test_seasonal_drawdown_is_shown_on_the_trend():
    rep = _seasonal_report()
    assert rep.analysis.metrics.drawdown_pct < -20                      # raw: measured from a seasonal high
    html_text = R.render_html(rep)
    assert "추세 고점 대비 −1.2%" in html_text
    assert "최고점 대비 −2" not in html_text
    rep.analysis.decomposition.trend.iloc[-1] = 50.6                   # trend back at its high
    assert "추세 최고 수준" in R.render_html(rep)
    # weak seasonality: the observation-based drawdown is kept
    weak = _seasonal_report(seasonal_strength=0.2)
    dd = R._fmt_pct(weak.analysis.metrics.drawdown_pct)
    assert f"최고점 대비 {dd}" in R.render_html(weak) and "추세 고점" not in R.render_html(weak)


def test_outlook_labels_say_what_each_number_is():
    rep = _seasonal_report(horizon=6)
    html_text = R.render_html(rep)
    card = re.search(r"<h2>미래 전망</h2>.*?</section>", html_text).group(0)
    # the probability compares the same two season means as the expected change (the 6-month
    # forecast window covers half a season, so compute_outlook removes its seasonal effect: UX-17)
    basis = "계절 보정한 향후 6개월 평균이 최근 12개월 평균보다 높을 확률"
    assert basis in html_text                                            # KPI tile
    assert f'title="{basis}">상승 확률 53%' in card                      # next to the expected change
    assert "변화율과 상승 확률은 모두 계절 보정한 향후 6개월 평균을 최근 12개월 평균과 비교한 값입니다" in card
    # ... and is no longer listed right under the horizon-end scenarios
    assert "<dt>상승 확률</dt>" not in card
    # scenarios: single horizon-end values, compared with the same point of the seasonal cycle
    assert "시나리오 (2030-06 시점 값)" in card
    assert "현재 수준" not in html_text                                   # the old season-mean tick
    assert "2029-06 실제 55.0" in card
    assert "세로선은 같은 계절 시점인 2029-06의 실제값입니다" in card
    # the tick sits at the observed same-month value (55), next to the base scenario (55)
    tick = float(re.search(r'class="rc" style="left:([\d.]+)%"', card).group(1))
    base = float(re.search(r'class="rm" style="left:([\d.]+)%"', card).group(1))
    assert abs(tick - base) < 1.0
    text = R.render_text(rep, width=100)
    assert "상승 확률 53% (같은 기준)" in text
    assert "시나리오(2030-06 시점)" in text and "(비교: 2029-06 실제 55.0)" in text
    # without a noticeable seasonal pattern the latest observation is the reference
    flat = build_report(np.linspace(10, 20, 36), slope=0.5)
    card = re.search(r"<h2>미래 전망</h2>.*?</section>", R.render_html(flat)).group(0)
    assert "현재 20.0" in card and "세로선은 현재 값(2022-12)입니다" in card
    assert "(비교: 현재 20.0)" in R.render_text(flat, width=100)
    _assert_balanced(html_text)


def _spike_report(flag: bool = True) -> TrendReport:
    vals = 50 + 5 * np.sin(np.arange(60) / 3.0)
    vals[30] = 90.0
    rep = build_report(vals, freq="M")
    s = rep.analysis.series
    m = rep.analysis.metrics
    rest = s.drop(s.index[30])
    m.peak_value, m.peak_date = float(rest.max()), rest.idxmax()        # analysis ignores the spike
    m.drawdown_pct = float((s.iloc[-1] - rest.max()) / rest.max() * 100)
    rep.analysis.anomalies = ([Anomaly(date=s.index[30], index=30, value=90.0, expected=50.0, zscore=6.0)]
                              if flag else [])
    return rep


def test_peak_mentions_an_excluded_one_off_spike():
    rep = _spike_report()
    html_text = R.render_html(rep)
    assert "일회성 급등 90.0(2022-07) 제외" in html_text                  # diagnosis card
    assert f"최고점 대비 {R._fmt_pct(rep.analysis.metrics.drawdown_pct)} (일회성 급등 제외)" in html_text
    _assert_balanced(html_text)
    # without a flagged spike there is nothing to explain
    plain = R.render_html(_spike_report(flag=False))
    assert "일회성 급등" not in plain


def test_method_note_names_the_decomposition_that_was_used():
    n = 72
    t = np.arange(n)
    trend = 100 * np.exp(0.02 * t)
    seas = trend * (np.exp(0.2 * np.sin(2 * np.pi * t / 12)) - 1)       # swing grows with the level
    rep = build_report(trend + seas, freq="M")
    s = rep.analysis.series
    rep.analysis.decomposition = Decomposition(trend=pd.Series(trend, index=s.index),
                                               seasonal=pd.Series(seas, index=s.index),
                                               resid=pd.Series(np.zeros(n), index=s.index), period=12)
    notes = re.search(r"<h2>분석 방법</h2>.*?</section>", R.render_html(rep)).group(0)
    assert "로그 공간에서 승법 분해" in notes and "가법" not in notes
    add = _seasonal_report()
    notes = re.search(r"<h2>분석 방법</h2>.*?</section>", R.render_html(add)).group(0)
    assert "강건한 가법 분해(STL 유사)" in notes and "승법" not in notes
    # the interval note matches forecast.py (t quantiles from the method's backtest errors)
    assert "경험적 분포" not in notes and "t-분포" in notes
    none = re.search(r"<h2>분석 방법</h2>.*?</section>", R.render_html(build_report(np.arange(12.0)))).group(0)
    assert "계절 성분 없이" in none and "가법 분해" not in none


def test_compare_links_are_safe_and_escaped():
    reports = [build_report(np.arange(12.0), name="a"), build_report(np.arange(12.0)[::-1], name="검색 량"),
               build_report(np.arange(12.0), name="evil"), build_report(np.arange(12.0), name="tab")]
    ranking = _ranking(reports)
    links = {"a": "cmp_a.html?x=1&y=\"2\"", "검색 량": "cmp_검색 량.html",
             "evil": "javascript:alert(1)", "tab": "java\tscript:alert(1)"}
    html_text = R.render_compare_html(reports, ranking, links=links)
    # card title and ranking name, escaped
    assert html_text.count('<a href="cmp_a.html?x=1&amp;y=%222%22">a</a>') == 2
    assert html_text.count('<a href="cmp_%EA%B2%80%EC%83%89%20%EB%9F%89.html">검색 량</a>') == 2
    assert "javascript" not in html_text and "script:" not in html_text
    assert html_text.count("<a href=") == 4
    _assert_balanced(html_text)
    for bad in ("data:text/html,<b>x</b>", " JavaScript:alert(1)", "vbscript:x", "java\nscript:x", "", None):
        assert R._safe_href(bad) is None, bad
    for good in ("x.html", "sub/x.html#top", "https://example.com/a?b=1", "http://example.com"):
        assert R._safe_href(good) == good
    # links are optional and keep the page unchanged otherwise
    assert "<a href=" not in R.render_compare_html(reports, ranking)


def test_compare_shows_the_span_of_every_change():
    m = build_report(np.linspace(10, 20, 36), name="월간", horizon=12)
    d = build_report(np.linspace(10, 20, 120), name="일간", freq="D", horizon=30)
    ranking = pd.DataFrame([
        {"name": "일간", "phase": "growing", "momentum": 0.1, "recent_change_pct": 3.0,
         "expected_change_pct": 23.2, "prob_increase": 0.9, "score": 80.0, "expected_change_ann_pct": 1169.0},
        {"name": "월간", "phase": "growing", "momentum": 0.2, "recent_change_pct": 8.0,
         "expected_change_pct": 7.9, "prob_increase": 0.8, "score": 60.0, "expected_change_ann_pct": 7.9},
    ])
    html_text = R.render_compare_html([m, d], ranking)
    table = re.search(r"<table>.*?</table>", html_text).group(0)
    # raw values, each with its own horizon (never an extrapolated annual rate)
    assert '+23.2%</span><span class="hz">30일</span>' in table
    assert '+7.9%</span><span class="hz">12개월</span>' in table
    assert '+3.0%</span><span class="hz">7일</span>' in table             # recent change: one week
    assert "1,169" not in html_text
    assert "주기가 다른 시계열이 섞여 있어" in html_text
    assert "예측 변화 <span class=\"nw\">(30일)</span>" in html_text      # card label
    # a shared span goes to the header once
    m2 = build_report(np.linspace(20, 10, 36), name="월간2", horizon=12)
    rk = pd.DataFrame([{"name": "월간", "expected_change_pct": 7.9, "score": 60.0},
                       {"name": "월간2", "expected_change_pct": -3.0, "score": 40.0}])
    html_text = R.render_compare_html([m, m2], rk)
    table = re.search(r"<table>.*?</table>", html_text).group(0)
    assert '예측 변화</abbr><span class="hz">(12개월)</span>' in table
    assert '<span class="hz">12개월</span>' not in table
    assert "주기가 다른 시계열" not in html_text
    # the probability header says what it compares
    assert "예측 변화와 같은 기준" in table


# --------------------------------------------------------------------------- final-round regressions


def test_reversal_labels_follow_the_slope_before():
    """UX-13: 'reversal_up' from a flat stretch is a turn upwards, not a reversal -- same rule
    (and wording) as narrative._changepoint_line, in the chart, the table and the terminal."""
    rep = build_report(np.linspace(10, 60, 60), slope=0.5, changepoints=True)
    cp = rep.analysis.changepoints[0]

    def outputs() -> str:
        return R.svg_chart(rep) + R.render_html(rep) + R.render_text(rep, width=100)

    cp.kind, cp.slope_before, cp.slope_after = "reversal_up", 0.17, 16.6       # flat -> rising
    out = outputs()
    assert "횡보→상승 전환" in out and "반전" not in out
    cp.slope_before, cp.slope_after = -0.5, 0.7                                # falling -> rising
    out = outputs()
    assert "하락→상승 반전" in out and "횡보" not in out
    cp.kind, cp.slope_before, cp.slope_after = "reversal_down", 2.0, -1.0      # rising -> falling
    out = outputs()
    assert "상승→하락 반전" in out and "횡보" not in out
    cp.slope_before = 0.05                                                     # flat -> falling
    out = outputs()
    assert "횡보→하락 전환" in out and "반전" not in out
    assert re.search(r'background:#d03b3b"></span>횡보→하락 전환', R.render_html(rep))
    cp.slope_before = float("nan")                                             # unknown: plain label
    assert "하락 반전" in R.render_text(rep, width=100)


def test_reversal_label_matches_the_narrative_end_to_end():
    from trendai.pipeline import run
    from trendai.samples import sample

    rep = run(sample("emerging", n=12, freq="Y", seed=0), name="emerging")
    ups = [cp for cp in rep.analysis.changepoints if cp.kind == "reversal_up"]
    assert ups, "the repro needs a reversal_up changepoint"
    text = R.render_text(rep, width=100)
    for cp in ups:
        flat = cp.slope_before >= -0.1 * max(abs(cp.slope_before), abs(cp.slope_after))
        label, narr = ("횡보→상승 전환", "횡보에서 상승으로 전환") if flat else ("하락→상승 반전", "하락에서 상승으로 반전")
        assert f"{R._fmt_date(cp.date, 'Y')} {label}" in text and narr in rep.narrative


def test_hover_titles_do_not_repeat_the_row_label(fake_report):
    """UX-15: the native tooltip read '2026-01 · 예측 예측 70.9, 80% …'."""
    svg = R.svg_chart(fake_report)
    titles = re.findall(r'<rect class="tai-hit"[^>]*><title>([^<]*)</title>', svg)
    assert titles and not any("예측 예측" in t for t in titles)
    fc_titles = [t for t in titles if t.startswith(("2026", "2027"))]
    assert fc_titles and all(re.match(r"\d{4}-\d{2} 예측 \S+, 80% \S+ ~ \S+, 95% ", t) for t in fc_titles)
    assert any(re.match(r"\d{4}-\d{2} 실제 \S+, 추세 \S+", t) for t in titles)
    # the visible read-out box keeps marking the forecast dates
    assert re.search(r'font-weight="600">\d{4}-\d{2} · 예측</text>', svg)


def test_partial_season_windows_are_labelled_seasonally_adjusted():
    """UX-17: compute_outlook removes the average seasonal effect of a window that is not a whole
    number of seasons (6 of 12 months here, 26 of 52 weeks by default); the labels say so."""
    rep = _seasonal_report(horizon=6)
    text, html_text = R.render_text(rep, width=120), R.render_html(rep)
    assert "예측 변화 +0.4% (계절 보정한 향후 6개월 평균, 최근 12개월 평균 대비)" in text
    assert "계절 보정한 향후 6개월 평균 · <span class=\"nw\">최근 12개월 평균 대비</span>" in html_text  # KPI
    assert "계절 보정한 향후 6개월 평균이 최근 12개월 평균보다 높을 확률" in html_text
    card = re.search(r"<h2>미래 전망</h2>.*?</section>", html_text).group(0)
    assert "6개월 구간은 계절 주기(12개월)를 다 채우지 못해" in card
    # whole seasons, or no seasonal component: plain means, no adjustment mentioned
    whole = _seasonal_report(horizon=12)
    assert "계절 보정" not in R.render_text(whole, width=120) + R.render_html(whole)
    plain = build_report(np.linspace(10, 20, 36), horizon=6)
    plain.analysis.season_length = 12
    assert "계절 보정" not in R.render_text(plain, width=120) + R.render_html(plain)


def test_weekly_kpi_label_matches_the_narratives_adjustment():
    from trendai.pipeline import run
    from trendai.samples import sample

    rep = run(sample("growing", n=156, freq="W", seed=0), name="growing")
    assert "계절 효과를 보정한 향후 26주 예측 평균" in rep.narrative
    assert "계절 보정한 향후 26주 평균" in R.render_text(rep, width=120)
    assert "계절 보정한 향후 26주 평균" in R.render_html(rep)


def test_drawdown_tile_uses_the_narratives_rule():
    """UX-18: the tile switches to the trend drawdown only when the narrative does
    (seasonal strength >= 0.5 *and* the two drawdowns differ by >= 5 points)."""
    rep = _seasonal_report()
    s, trend = rep.analysis.series, rep.analysis.decomposition.trend
    m = rep.analysis.metrics
    # raw drawdown within 5 points of the trend's (-1.2 %): the observation basis is kept
    m.drawdown_pct = -4.0
    html_text = R.render_html(rep)
    assert "최고점 대비 −4.0%" in html_text and "추세 고점" not in html_text
    m.drawdown_pct = -6.5                                                # 5.3 points apart
    assert "추세 고점 대비 −1.2%" in R.render_html(rep)
    assert len(s) == len(trend)


@pytest.mark.parametrize("name,freq,n", [("reviving", "Q", 40), ("declining", "M", 120),
                                          ("seasonal", "M", 120), ("fad", "M", 120)])
def test_drawdown_tile_agrees_with_the_narrative_end_to_end(name, freq, n):
    from trendai.pipeline import run
    from trendai.samples import sample

    rep = run(sample(name, n=n, freq=freq, seed=0), name=name)
    tile = re.search(r'<div class="lbl">현재 값</div>.*?<div class="sub">(.*?)</div>', R.render_html(rep)).group(1)
    trend_basis = "계절 요인을 뺀 추세선" in rep.narrative          # narrative._seasonal_drawdown applied
    assert ("추세 " in tile) == trend_basis, (tile, rep.narrative)
    if not trend_basis and "최고점 대비" in tile:
        gap = R._fmt_pct(abs(rep.analysis.metrics.drawdown_pct), signed=False)
        assert gap in rep.narrative                                      # same number in both places


def _huge_report(scale: float) -> TrendReport:
    import warnings

    base = 100 + 10 * np.sin(np.arange(48) / 3) + np.arange(48)
    with warnings.catch_warnings():          # the builder's own sums overflow (metrics.mean): not report code
        warnings.simplefilter("ignore", RuntimeWarning)
        return build_report(base * scale, slope=scale, anomalies=True, changepoints=True)


@pytest.mark.parametrize("scale", [1e300, -1e300, 1e306, -1.1e306, 1e250])
def test_charts_render_beyond_1e300(scale):
    """F8: |y| > 1e300 crashed _nice_ticks (math domain error) and values above 1e300 fell off
    the axis; tick labels of huge values printed hundreds of digits."""
    rep = _huge_report(scale)
    html_text = R.render_html(rep)
    _assert_balanced(html_text)
    for w, h in ((960, 380), (600, 360), (380, 340)):
        svg = R.svg_chart(rep, w, h)
        ET.fromstring(svg)
        ticks = _y_ticks(svg)
        labels = [lab for _, lab in ticks]
        assert 4 <= len(ticks) <= 6 and len(set(labels)) == len(labels)
        assert all(len(lab) <= 9 for lab in labels), labels
        vals = sorted(_tick_value(lab) for lab in labels)
        data = rep.analysis.series.to_numpy()
        assert vals[0] <= data.min() * (1 + 1e-9 * np.sign(data.min())) and vals[-1] >= data.max() * (1 - 1e-9 * np.sign(data.max()))
        assert "inf" not in svg and "nan" not in svg
    R.render_text(rep)
    R.render_compare_html([rep], _ranking([rep]))


def test_axis_spanning_the_whole_float_range_renders():
    import warnings

    vals = np.concatenate([np.linspace(-1.5e308, 0, 20), np.linspace(0, 1.5e308, 20)])
    with warnings.catch_warnings():          # the builder's own sums overflow: not report code
        warnings.simplefilter("ignore", RuntimeWarning)
        rep = build_report(vals, slope=1e306)
    with warnings.catch_warnings():
        warnings.simplefilter("error")                      # no overflow on the way either
        svg = R.svg_chart(rep)
        R.render_html(rep), R.render_text(rep), R._mini_chart(rep)
    ET.fromstring(svg)
    vals_t = sorted(_tick_value(lab) for _, lab in _y_ticks(svg))
    assert vals_t[0] <= -1.5e308 and vals_t[-1] >= 1.5e308
    ys = [float(v) for v in re.findall(r'<path class="tai-hist" d="M[\d.]+,([\d.]+)', svg)]
    assert ys and "inf" not in svg and "nan" not in svg


def test_nice_ticks_cover_any_finite_range():
    for lo, hi in [(1e301, 2e301), (1e300, 1.6e308), (-1.6e308, -1e305), (1.2e308, 1.5e308),
                   (1.79e308, 1.797e308), (-1e308, 1e308), (1.7e308, 1.7e308), (-1.79e308, 1.79e308)]:
        t = R._nice_ticks(lo, hi)
        assert 2 <= len(t) <= 6 and all(math.isfinite(v) for v in t), (lo, hi, t)
        assert t[0] <= lo * (1 + 1e-12 * np.sign(lo)) and t[-1] >= hi * (1 - 1e-12 * np.sign(hi)), (lo, hi, t)
        assert t == sorted(t) and len(set(t)) == len(t)
        labels = R._axis_labels(t)
        assert len(set(labels)) == len(labels) and all(len(lab) <= 12 for lab in labels), labels
    assert R._axis_labels([0, 5e15, 1e16, 1.5e16]) == ["0", "5e15", "1e16", "1.5e16"]
    assert R._axis_labels([1.0e300, 1.25e300, 1.5e300, 1.75e300]) == ["1e300", "1.25e300", "1.5e300", "1.75e300"]
    assert R._fmt_axis_value(1.234e300, 2e300) == "1.23e300"


def test_english_narrative_reasons_are_not_repeated_in_korean():
    """UX-20: on --lang en pages the diagnosis card repeated the 'Evidence:' bullets in Korean."""
    from trendai.narrative import translate_reason
    from trendai.pipeline import run
    from trendai.samples import sample

    rep = run(sample("fad", seed=0), name="fad", lang="en")
    reasons = [r for r in rep.analysis.phase.reasons if translate_reason(r)]
    assert reasons and all(f"Evidence: {translate_reason(r)}" in rep.narrative for r in reasons)
    card = re.search(r"<h2>현재 진단</h2>.*?</section>", R.render_html(rep)).group(0)
    assert not any(R._esc(r) in card for r in reasons)
    assert "판단 근거는 ‘AI 트렌드 해석’" in card


def test_leaderboard_fits_phone_width(fake_report):
    """UX-20: on phones the weight bar is hidden (the % stays) so model names get the width."""
    html_text = R.render_html(fake_report)
    phone = re.search(r"@media \(max-width:640px\)\{(.*?)\n\}", html_text, flags=re.S).group(1)
    assert ".wcell .track{display:none}" in phone
    assert re.search(r'<td class="wcell"><div class="wbar"><div class="track">.*?<span class="pct">50%</span>', html_text)
    assert '<th><span class="hide-sm">앙상블 </span>가중치</th>' in html_text


def test_intermittent_models_are_labelled():
    """forecast ranks intermittent-demand candidates (offered 'sba') by RMSE: say so."""
    assert R._model_desc("sba") and R._model_desc("drift@log").endswith("(로그 공간)")
    vals = np.where(np.arange(48) % 3 == 0, 5.0, 0.0)
    rep = build_report(vals)
    rep.forecast.model_scores = [ModelScore("theta", 2.0, 2.3, 150.0, 0.80, 0.33),
                                 ModelScore("ses", 2.1, 2.2, 150.0, 0.84, 0.34),
                                 ModelScore("sba", 2.2, 2.28, 150.0, 0.91, 0.33),
                                 ModelScore("holt_damped", 2.3, 2.34, 150.0, 0.93, 0.0)]
    html_text = R.render_html(rep)
    assert "간헐 수요 Croston(SBA 보정)" in html_text
    assert "제곱 오차(RMSE)" in html_text
    text = R.render_text(rep, width=100)
    assert "상위 모델(RMSE):  ses 2.20  sba 2.28  theta 2.30" in text
    # ordinary series: unchanged
    plain = build_report(np.linspace(10, 20, 36))
    assert "RMSE" not in R.render_html(plain) + R.render_text(plain, width=100)
