"""Display formatting of the web VM (docs/WEB_SPEC.md D.1): every row of the D.1.7 vectors, ko and en.

The same table is asserted by WP3's ``tests/js/fmt.test.mjs`` against ``static/js/fmt.js``; keep the rows exactly as
in the spec.
"""

from __future__ import annotations

import ast
import math
from decimal import Decimal
from pathlib import Path

import pytest

from trendai.web import fmt
from trendai.web import texts as T

# ---------------------------------------------------------------------------------------------------- D.1.7 compact

COMPACT_VECTORS = [
    (0, "0", "0"),
    (9876, "9,876", "9,876"),
    (45.23, "45.2", "45.2"),
    (1.204, "1.20", "1.20"),
    (0.0234, "0.023", "0.023"),
    (0.00005, "5×10⁻⁵", "5×10⁻⁵"),
    (12345, "1.2만", "12K"),
    (12500, "1.3만", "13K"),
    (99999, "10만", "100K"),
    (340000, "34만", "340K"),
    (3456789, "346만", "3.5M"),
    (12345678, "1,230만", "12M"),
    (98765432, "9,880만", "99M"),
    (99999999, "1억", "100M"),
    (99950000, "1억", "100M"),
    (340000000, "3.4억", "340M"),
    (999999999, "10억", "1B"),
    (1.2e12, "1.2조", "1.2×10¹²"),
    (3.4e16, "3.4×10¹⁶", "3.4×10¹⁶"),
    (-12345, "−1.2만", "−12K"),
    # exact binary ties (half away from zero)
    (0.125, "0.13", "0.13"),
    (2.5, "2.50", "2.50"),
    (12.25, "12.3", "12.3"),
    (1234.5, "1,235", "1,235"),
    (9999.5, "1만", "10K"),
]


@pytest.mark.parametrize("x, ko, en", COMPACT_VECTORS, ids=[repr(v[0]) for v in COMPACT_VECTORS])
def test_compact_vectors(x, ko, en):
    assert fmt.compact(x, "ko") == ko
    assert fmt.compact(x, "en") == en


# ---------------------------------------------------------------------------------------------- D.1.7 other calls

@pytest.mark.parametrize("x, out", [(123456789, "123,456,789"), (1234.5, "1,235"), (0.125, "0.13")])
def test_full_vectors(x, out):
    assert fmt.full(x, "ko") == out
    assert fmt.full(x, "en") == out


@pytest.mark.parametrize("x, out", [(71.34, "71.3"), (71.25, "71.3"), (100, "100.0")])
def test_value_text_relative_index_vectors(x, out):
    for lang in ("ko", "en"):
        assert fmt.value_text(x, "relative_index", lang) == out
        assert fmt.value_text(x, "relative_index", lang, compact=True) == out   # never compacted


@pytest.mark.parametrize("p, out", [(18.2, "+18%"), (4.06, "+4.1%"), (-4.06, "−4.1%"), (0.02, "0%"), (9.96, "+10%"),
                                    (1365.4, "+1,365%"), (2.25, "+2.3%"), (0.05, "+0.1%")])
def test_pct_vectors(p, out):
    assert fmt.pct(p) == out


@pytest.mark.parametrize("p, out", [(12.03, "▲12%"), (-4.06, "▼4.1%"), (0, "0%")])
def test_pct_arrow_vectors(p, out):
    assert fmt.pct_arrow(p) == out


@pytest.mark.parametrize("p, ko, en", [(0.4, "+0.4%p", "+0.4 pp"), (-3.91, "−3.9%p", "−3.9 pp"),
                                       (12.34, "+12.3%p", "+12.3 pp"), (0.042, "+0.042%p", "+0.042 pp"),
                                       (0.25, "+0.3%p", "+0.3 pp")])
def test_pp_vectors(p, ko, en):
    assert fmt.pp(p, "ko") == ko
    assert fmt.pp(p, "en") == en


PROB_TEXT_VECTORS = [
    (0.81, "약 80%", "about 80%"), (0.83, "약 85%", "about 85%"), (0.5, "약 50%", "about 50%"),
    (0.974, "약 95%", "about 95%"), (0.975, "95% 이상", "95% or more"), (0.99, "95% 이상", "95% or more"),
    (0.03, "약 5%", "about 5%"), (0.02, "5% 이하", "5% or less"), (0.825, "약 85%", "about 85%"),
    (0.875, "약 90%", "about 90%"),
]


@pytest.mark.parametrize("p, ko, en", PROB_TEXT_VECTORS)
def test_prob_text_vectors(p, ko, en):
    assert fmt.prob_text(p, "ko") == ko
    assert fmt.prob_text(p, "en") == en


@pytest.mark.parametrize("p, ko, en", [(0.2, "상승 확률 약 20%", "chance of rise about 20%"),
                                       (0.02, "상승 확률 5% 이하", "chance of rise 5% or less")])
def test_prob_labelled_vectors(p, ko, en):
    assert fmt.prob_labelled(p, "ko") == ko
    assert fmt.prob_labelled(p, "en") == en


LIKELIHOOD_ROWS = ["rise_very_likely", "rise_more_likely", "rise_slightly", "even", "fall_slightly",
                   "fall_more_likely", "fall_very_likely"]


@pytest.mark.parametrize("p, row", list(zip((0.9, 0.8, 0.6, 0.5, 0.4, 0.2, 0.05), LIKELIHOOD_ROWS, strict=True)))
def test_likelihood_vectors(p, row):
    ko, en = T.LIKELIHOOD_WORDS[row]
    assert fmt.likelihood(p, "ko") == ko
    assert fmt.likelihood(p, "en") == en


def test_date_vectors():
    assert fmt.date_label("2026-11-17", "W", "ko") == "2026.11.17 주"
    assert fmt.date_label("2026-11-17", "W", "en") == "Week of Nov 17, 2026"
    assert fmt.range_label("2021-11-01", "2026-10-01", "M", "ko") == "2021.11 ~ 2026.10"
    assert fmt.range_label("2021-11-01", "2026-10-01", "M", "en") == "Nov 2021 – Oct 2026"
    assert fmt.range_label("2021-11-29", "2026-11-16", "W", "ko") == "2021.11.29 ~ 2026.11.22"
    assert fmt.range_label("2021-11-29", "2026-11-16", "W", "en") == "Nov 29, 2021 – Nov 22, 2026"
    assert fmt.datetime_label("2026-11-25T00:12:00Z", "ko") == "2026.11.25 09:12"
    assert fmt.datetime_label("2026-11-25T00:12:00Z", "en") == "Nov 25, 2026 09:12 KST"


# ------------------------------------------------------------------------------------------ D.1.6 table and extras

@pytest.mark.parametrize("freq, ko, en", [
    ("D", "2026.11.25.", "Nov 25, 2026"), ("W", "2026.11.25 주", "Week of Nov 25, 2026"),
    ("M", "2026년 11월", "Nov 2026"), ("Q", "2026년 4분기", "Q4 2026"), ("Y", "2026년", "2026")])
def test_date_label_table(freq, ko, en):
    assert fmt.date_label("2026-11-25", freq, "ko") == ko
    assert fmt.date_label("2026-11-25", freq, "en") == en


@pytest.mark.parametrize("freq, ko, en", [
    ("D", "2026.11.25", "Nov 25, 2026"), ("W", "2026.11.25", "Nov 25, 2026"), ("M", "2026.11", "Nov 2026"),
    ("Q", "2026 4분기", "Q4 2026"), ("Y", "2026", "2026")])
def test_date_short_table(freq, ko, en):
    assert fmt.date_short("2026-11-25", freq, "ko") == ko
    assert fmt.date_short("2026-11-25", freq, "en") == en


@pytest.mark.parametrize("start, end, freq, ko, en", [
    ("2021-11-29", "2026-11-16", "D", "2021.11.29 ~ 2026.11.16", "Nov 29, 2021 – Nov 16, 2026"),
    ("2021-01-01", "2026-07-01", "Q", "2021년 1분기 ~ 2026년 3분기", "Q1 2021 – Q3 2026"),
    ("2016-01-01", "2025-01-01", "Y", "2016 ~ 2025", "2016 – 2025"),
    ("2021-11-01", "2026-10-26", "D", "2021.11.01 ~ 2026.10.26", "Nov 1, 2021 – Oct 26, 2026"),
])
def test_range_label_table(start, end, freq, ko, en):
    assert fmt.range_label(start, end, freq, "ko") == ko
    assert fmt.range_label(start, end, freq, "en") == en


def test_datetime_label_crosses_midnight_in_kst():
    assert fmt.datetime_label("2026-11-30T15:00:00Z", "ko") == "2026.12.01 00:00"
    assert fmt.datetime_label("2026-11-30T15:00:00+00:00", "en") == "Dec 1, 2026 00:00 KST"


@pytest.mark.parametrize("n, freq, ko, en", [(26, "W", "26주", "26 weeks"), (12, "M", "12개월", "12 months"),
                                             (30, "D", "30일", "30 days"), (8, "Q", "8분기", "8 quarters"),
                                             (5, "Y", "5년", "5 years"), (1, "W", "1주", "1 week"),
                                             (104, "W", "104주", "104 weeks")])
def test_span_label(n, freq, ko, en):
    assert fmt.span_label(n, freq, "ko") == ko
    assert fmt.span_label(n, freq, "en") == en


def test_freq_labels_and_adjectives():
    assert [fmt.freq_label(f, "ko") for f in "DWMQY"] == ["일 단위", "주 단위", "월 단위", "분기 단위", "연 단위"]
    assert [fmt.freq_label(f, "en") for f in "DWMQY"] == ["daily", "weekly", "monthly", "quarterly", "yearly"]
    assert [fmt.freq_adjective(f, "ko") for f in "DWMQY"] == ["일간", "주간", "월간", "분기", "연간"]
    assert fmt.freq_adjective("W", "en") == "weekly"


@pytest.mark.parametrize("x, d, out", [(0.125, 2, "0.13"), (1234.5, 0, "1235"), (-2.5, 2, "-2.50"),
                                       (71.25, 1, "71.3"), (-0.125, 2, "-0.13"), (2.675, 2, "2.67")])
def test_round_half_away_reference_strings(x, d, out):
    # 2.675 is 2.67499999… as a double: the exact value decides, not the decimal literal
    r = fmt.round_half_away(x, d)
    assert isinstance(r, Decimal)
    assert str(r) == out


@pytest.mark.parametrize("bad", [None, float("nan"), float("inf"), -float("inf")])
def test_missing_values_print_a_dash(bad):
    for fn in (fmt.compact, fmt.full, fmt.pct, fmt.pct_arrow, fmt.pct_abs, fmt.pp, fmt.prob_text, fmt.prob_exact,
               fmt.prob_labelled, fmt.likelihood):
        assert fn(bad) == fmt.DASH
    assert fmt.value_text(bad, "relative_index") == fmt.DASH
    assert fmt.value_text(bad, "raw", compact=True) == fmt.DASH


def test_minus_sign_and_zero():
    assert fmt.full(-45.23) == "−45.2"
    assert fmt.full(-0.0) == "0"
    assert fmt.compact(-3456789, "en") == "−3.5M"
    assert fmt.pp(0, "ko") == "0%p"
    assert fmt.pp(-0.0, "en") == "0 pp"
    assert fmt.pct_abs(-18.2) == "18%"
    assert fmt.pct_abs(0.01) == "0%"
    assert "-" not in fmt.compact(-12345, "ko")


def test_full_and_compact_notation_edges():
    assert fmt.full(1.5e12) == "1.5×10¹²"
    assert fmt.full(0.0001) == "0.0001"
    assert fmt.full(0.09996) == "0.1"           # two significant digits, rounding carries
    assert fmt.full(999.95) == "1,000.0"       # decimals chosen from |x| before rounding (number_text rule)
    # three significant digits for every ko unit, not only 만
    assert fmt.compact(123456789012, "ko") == "1,230억"
    assert fmt.compact(9.9995e15, "ko") == "1×10¹⁶"
    assert fmt.compact(999.95e9, "en") == "1×10¹²"


def test_pp_tiny_values_use_power_notation():
    assert fmt.pp(7.4e-15, "ko") == "+7.4×10⁻¹⁵%p"
    assert fmt.pp(-0.00005, "en") == "−5×10⁻⁵ pp"
    assert fmt.pp(1234.56, "ko") == "+1,234.6%p"


def test_prob_exact_matches_narrative():
    from trendai.narrative import format_prob

    for p in (0.812, 0.5, 0.996, 0.004):
        assert fmt.prob_exact(p, "ko") == format_prob(p, "ko")
        assert fmt.prob_exact(p, "en") == format_prob(p, "en")


def test_likelihood_bin_edges():
    words = [T.LIKELIHOOD_WORDS[k][0] for k in LIKELIHOOD_ROWS]
    assert fmt.likelihood(0.90) == words[0]
    assert fmt.likelihood(0.8999) == words[1]
    assert fmt.likelihood(0.70) == words[1]
    assert fmt.likelihood(0.55) == words[2]
    assert fmt.likelihood(0.5500001) == words[2]
    assert fmt.likelihood(0.5499) == words[3]
    assert fmt.likelihood(0.4501) == words[3]
    assert fmt.likelihood(0.45) == words[4]
    assert fmt.likelihood(0.30) == words[5]
    assert fmt.likelihood(0.10) == words[6]


def test_prob_text_never_prints_a_certainty():
    for p in (0.0, 1.0, 0.9749, 0.0251):
        text = fmt.prob_text(p, "ko")
        assert text not in ("약 100%", "약 0%")
    assert math.isclose(0.825 * 20, 16.5)   # the documented double product behind "약 85%"


def test_fmt_never_imports_numfmt_number_text():
    source = Path(fmt.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert node.module is None or "numfmt" not in node.module, "fmt.py must not import numfmt"
            assert all(alias.name not in ("number_text", "sci") for alias in node.names)
        if isinstance(node, ast.Import):
            assert all("numfmt" not in alias.name for alias in node.names)
        if isinstance(node, ast.Name):
            assert node.id not in ("number_text", "numfmt"), node.id
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            # display rounding goes through round_half_away only: no builtin round()
            assert node.func.id != "round", "use round_half_away, not round()"
