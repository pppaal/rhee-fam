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

    TAGS = {"div", "section", "table", "tr", "td", "th", "ul", "li", "header", "main", "svg", "g", "span",
            "article", "dl", "p", "h1", "h2", "h3"}

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
    assert R._fmt_prob(0.997) == ">99%" and R._fmt_prob(1.0) == "100%"
    assert R._fmt_prob(0.001) == "<1%" and R._fmt_prob(0.0) == "0%"
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
