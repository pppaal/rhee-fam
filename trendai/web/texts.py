"""Server-side display texts of the web app (spec G.13, G.14, and the server texts quoted in C.10 and D.1-D.4).

Owner: **WP1-vm**. Message-catalog texts (errors, notices, fix labels) are *not* here: they live in
``trendai.messages`` (WP2) and are rendered with ``messages.render_web`` / ``render_web_msg``. UI chrome lives in the
frontend catalog (``static/js/catalog.js``, WP3) and is never duplicated here.

Unless stated otherwise a value is a ``(ko, en)`` tuple -- the same convention as ``messages.MESSAGES`` -- and
templates use ``str.format`` placeholders named as in the spec (``{name}``, ``{span}``, ...). Korean texts are in
해요체 like the rest of the web UI. :func:`t` picks the language.

Cross-WP names (WP1-server imports them, so they keep their name and shape): ``SAMPLE_TITLES``,
``SAMPLE_TITLES_EN``, ``SAMPLE_DESCRIPTIONS_EN`` (``GET /api/samples``, C.3) and ``NAVER_AGE_LABELS``
(``GET /api/config``, C.2). Every other table is internal to WP1-vm.
"""

from __future__ import annotations

from typing import Any, Mapping

__all__ = [
    "t",
    "SAMPLE_TITLES", "SAMPLE_TITLES_EN", "SAMPLE_DESCRIPTIONS_EN", "NAVER_AGE_LABELS",
    "TILE_LABELS", "TILE_HELP", "TILE_TEXTS", "MOMENTUM_WORDS", "MOMENTUM_UNITS", "VOLATILITY_WORDS", "PEAK_TEXTS",
    "PHASE_SHORT", "LIKELIHOOD_WORDS", "PROB_TEXTS", "CERTAINTY_LABELS", "RELIABILITY_LABELS", "HERO_TEXTS",
    "ACCURACY_TEXTS", "REFERENCE_NOTES", "PHASE_TEXTS",
    "SOURCE_LABELS", "SOURCE_SHORT", "WIKI_LANG_NAMES", "SOURCE_LINE_TEXTS", "VALUE_KIND_LABELS",
    "VALUE_KIND_NOTES", "VALUE_WORDS", "INDEX_UNIT_WORDS", "FILTER_LABELS",
    "CHART_TEXTS", "STRENGTH_WORDS", "PAST_TEXTS", "SCENARIO_TEXTS", "FUTURE_TEXTS",
    "NARRATIVE_TEXTS", "SUMMARY_TEXTS", "COMPARE_TEXTS",
    "EXPORT_TEXTS", "EXPORT_COLUMNS",
]


def t(table: Mapping[str, tuple[str, str]], key: str, lang: str, **params: Any) -> str:
    """``table[key]`` in ``lang`` (``"en"``; anything else Korean), formatted with ``params`` when given."""
    ko, en = table[key]
    text = en if lang == "en" else ko
    return text.format(**params) if params else text


# ---------------------------------------------------------------------------------------------------------------------
# Cross-WP tables
# ---------------------------------------------------------------------------------------------------------------------

#: Sample name -> Korean title (G.13). Keys = ``samples.list_samples()``.
SAMPLE_TITLES: dict[str, str] = {
    "emerging": "태동기형", "growing": "성장형", "peaking": "정점형", "fad": "반짝유행", "declining": "쇠퇴형",
    "seasonal": "계절형", "reviving": "재부상형", "breakout": "급성장형", "volatile": "변동형",
}

#: Sample name -> English title (G.13).
SAMPLE_TITLES_EN: dict[str, str] = {
    "emerging": "Emerging", "growing": "Growing", "peaking": "Peaking", "fad": "Fad", "declining": "Declining",
    "seasonal": "Seasonal", "reviving": "Reviving", "breakout": "Breakout", "volatile": "Volatile",
}

#: Sample name -> English description (G.13, verbatim). The Korean description is ``samples.DESCRIPTIONS[name]``.
SAMPLE_DESCRIPTIONS_EN: dict[str, str] = {
    "emerging": "An emerging trend growing fast from a small base (start of an S-curve)",
    "growing": "Steady growth with seasonality",
    "peaking": "Growth slowing as it nears saturation (end of an S-curve)",
    "fad": "A short-lived fad that spiked and is now falling",
    "declining": "A steadily declining trend",
    "seasonal": "No trend, clear seasonality (stable)",
    "reviving": "Fell, then recently rebounding",
    "breakout": "Flat, then rapid growth after a structural break",
    "volatile": "No clear direction, large noise",
}

#: Naver age code ``"1"``..``"11"`` -> ``(ko, en)`` label exactly as ``/api/config`` ``naver_ages`` (C.2). Also the
#: base of ``filters_label`` (D.2.2), which merges consecutive codes (``4,5`` -> ``25–34세`` / ``25–34``).
NAVER_AGE_LABELS: dict[str, tuple[str, str]] = {
    "1": ("0–12세", "0–12"), "2": ("13–18세", "13–18"), "3": ("19–24세", "19–24"), "4": ("25–29세", "25–29"),
    "5": ("30–34세", "30–34"), "6": ("35–39세", "35–39"), "7": ("40–44세", "40–44"), "8": ("45–49세", "45–49"),
    "9": ("50–54세", "50–54"), "10": ("55–59세", "55–59"), "11": ("60세 이상", "60+"),
}

# ---------------------------------------------------------------------------------------------------------------------
# Tiles (G.14, D.2.4)
# ---------------------------------------------------------------------------------------------------------------------

#: Tile key -> (ko, en) label.
TILE_LABELS: dict[str, tuple[str, str]] = {
    "phase": ("현재 국면", "Current phase"),
    "recent_change": ("최근 변화", "Recent change"),
    "momentum": ("모멘텀", "Momentum"),
    "expected_change": ("예측 변화", "Expected change"),
    "prob_increase": ("상승 가능성", "Chance of rise"),
    "peak_season": ("피크 시즌", "Peak season"),
    "volatility": ("변동성", "Volatility"),
}

#: Tile key -> (ko, en) tooltip (G.14, verbatim; ``recent_change`` has ``{span}``).
TILE_HELP: dict[str, tuple[str, str]] = {
    "phase": ("추세의 모양과 통계 검정으로 정한 수명주기 단계예요. 근거는 ‘왜?’에서 볼 수 있어요.",
              "The lifecycle stage, decided from the trend's shape and statistical tests. See ‘Why?’ for the "
              "evidence."),
    "recent_change": ("최근 한 주기({span}) 동안 추세선이 얼마나 변했는지예요. 계절 변동은 뺀 값이에요.",
                      "How much the trend line changed over the last cycle ({span}), without seasonal swings."),
    "momentum": ("최근 기울기와 장기 기울기의 차이예요(현재 추세 수준 대비 %p). +면 장기 평균보다 빨리 오르는 중이에요.",
                 "Recent slope minus long-run slope, in percentage points of the current trend level. Positive "
                 "means rising faster than usual."),
    "expected_change": ("예측 마지막 한 주기의 평균을 최근 한 주기의 평균과 비교한 값이에요. 특정 시점 한 값의 변화가 아니에요.",
                        "The average of the last forecast cycle compared with the average of the last observed "
                        "cycle — not the change of a single point."),
    "prob_increase": ("같은 두 평균에서 미래 평균이 더 높을 확률이에요.",
                      "The chance that the future average is higher than the recent one (same two averages)."),
    "peak_season": ("계절 성분이 가장 높은 시기와 오르기 시작하는 시기예요. 계절성이 뚜렷할 때만 보여 드려요.",
                    "When the seasonal component peaks and when it starts rising. Shown only for clearly seasonal "
                    "data."),
    "volatility": ("추세와 계절로 설명되지 않는 변동의 크기(잔차의 변동계수)예요.",
                   "The size of swings not explained by trend and season (coefficient of variation of the "
                   "residuals)."),
}

#: Tile ``sub``/``value`` fragments (D.2.4).
TILE_TEXTS: dict[str, tuple[str, str]] = {
    "prob_exact_suffix": ("정확한 값: {prob_exact}", "Exact value: {prob_exact}"),
    "for_reference": (" · 참고용", " · for reference"),
    "same_basis": ("{likelihood} · 예측 변화와 같은 기준", "{likelihood} · same basis as expected change"),
    "same_basis_only": ("예측 변화와 같은 기준", "same basis as expected change"),
    "direction_uncertain": ("방향 불확실 · ", "direction uncertain · "),
    "recent_sub": ("추세 기준 · {basis}", "trend · {basis}"),
    "recent_yoy": ("전년 동기 대비", "YoY"),
    "recent_prev_year": ("전년 대비", "YoY"),
    "recent_lag": ("{span} 전 대비", "vs. {span} ago"),
    "tentative_suffix": ("(잠정)", " (tentative)"),
    "volatility_sub": ("잔차 변동 {cv}%", "residual CV {cv}%"),
    "not_computed": ("계산하지 않음", "not computed"),
}

#: Momentum word key -> (ko, en) (D.2.4).
MOMENTUM_WORDS: dict[str, tuple[str, str]] = {
    "tentative": ("판단 보류", "Too short to tell"),
    "unclear": ("뚜렷하지 않음", "No clear change"),
    "accelerating": ("가속", "Accelerating"),
    "decline_easing": ("하락 둔화", "Decline easing"),
    "falling_faster": ("하락 가속", "Falling faster"),
    "slowing": ("둔화", "Slowing"),
    "steady": ("비슷한 속도", "Steady pace"),
}

#: Freq -> (ko, en) momentum unit suffix.
MOMENTUM_UNITS: dict[str, tuple[str, str]] = {
    "D": ("/일", "/day"), "W": ("/주", "/wk"), "M": ("/월", "/mo"), "Q": ("/분기", "/qtr"), "Y": ("/년", "/yr"),
}

#: Volatility level -> (ko, en).
VOLATILITY_WORDS: dict[str, tuple[str, str]] = {
    "low": ("낮음", "Low"), "medium": ("보통", "Medium"), "high": ("높음", "High"),
}

#: Peak-season templates and group names (D.2.4). Weekday names are the narrative's own (plural in English).
PEAK_TEXTS: dict[str, tuple[str, str]] = {
    "sub": ("보통 {peak} 정점 · {start}부터 상승", "Usually peaks in {peak} · rises from {start}"),
    "sub_no_start": ("보통 {peak} 정점", "Usually peaks in {peak}"),
    "sub_weekday": ("요일 패턴 · 가장 낮은 날 {trough}", "Weekday pattern · lowest on {trough}"),
    "month": ("{n}월", "{abbr}"),
    "quarter": ("{n}분기", "Q{n}"),
}

# ---------------------------------------------------------------------------------------------------------------------
# Hero, phase, outlook (D.1.5, D.2.3, D.2.5)
# ---------------------------------------------------------------------------------------------------------------------

#: Phase label -> (ko, en) short name (D.2.5).
PHASE_SHORT: dict[str, tuple[str, str]] = {
    "emerging": ("태동기", "Emerging"), "growing": ("성장기", "Growing"), "peaking": ("정점", "Peaking"),
    "declining": ("쇠퇴기", "Declining"), "stable": ("안정기", "Stable"), "reviving": ("재부상", "Reviving"),
    "volatile": ("변동성", "Volatile"),
}

#: Likelihood bin -> (ko, en) word (D.1.5). ``fmt.likelihood`` owns the thresholds and its own copy of these words
#: (it is mirrored by ``fmt.js``); this table is the reference the tests compare it with.
LIKELIHOOD_WORDS: dict[str, tuple[str, str]] = {
    "rise_very_likely": ("상승 가능성 매우 높음", "Rise very likely"),
    "rise_more_likely": ("상승 쪽이 우세", "Rise more likely"),
    "rise_slightly": ("상승 쪽이 약간 우세", "Rise slightly more likely"),
    "even": ("상승·하락 엇비슷", "About even"),
    "fall_slightly": ("하락 쪽이 약간 우세", "Fall slightly more likely"),
    "fall_more_likely": ("하락 쪽이 우세", "Fall more likely"),
    "fall_very_likely": ("하락 가능성 매우 높음", "Fall very likely"),
}

#: ``prob_labelled`` and ``prob_text`` forms (D.1.5); the reference for ``fmt.prob_text`` / ``fmt.prob_labelled``.
PROB_TEXTS: dict[str, tuple[str, str]] = {
    "labelled": ("상승 확률 {prob}", "chance of rise {prob}"),
    "high": ("95% 이상", "95% or more"),
    "low": ("5% 이하", "5% or less"),
    "about": ("약 {n}%", "about {n}%"),
}

#: Phase certainty level -> (ko, en) (D.2.5).
CERTAINTY_LABELS: dict[str, tuple[str, str]] = {
    "high": ("판단 확신도 높음", "Phase certainty: high"),
    "medium": ("판단 확신도 보통", "Phase certainty: medium"),
    "low": ("판단 확신도 낮음", "Phase certainty: low"),
    "tentative": ("잠정 판단", "Tentative"),
}

#: Forecast reliability level -> (ko, en) (D.2.3).
RELIABILITY_LABELS: dict[str, tuple[str, str]] = {
    "high": ("예측 신뢰도 높음", "Forecast reliability: high"),
    "medium": ("예측 신뢰도 보통", "Forecast reliability: medium"),
    "low": ("예측 신뢰도 낮음", "Forecast reliability: low"),
}

#: Hero templates (D.2.3).
HERO_TEXTS: dict[str, tuple[str, str]] = {
    "line_not_computed": ("{arrow} {word} · 상승 확률은 계산하지 않음({reason})",
                          "{arrow} {word} · chance of rise not computed ({reason})"),
    "line_matching": ("{arrow} {likelihood} ({prob})", "{arrow} {likelihood} ({prob})"),
    "line_full": ("{arrow} {word} · {likelihood} ({prob})", "{arrow} {word} · {likelihood} ({prob})"),
    "basis": ("{fut}, {cur} 대비", "{fut}, vs. {cur}"),
    "sentence_change": ("{fut}이 {cur}보다 {pct} {more} 것으로 예측해요.",
                        "The {fut} is expected to be {pct} {more} than the {cur}."),
    "sentence_flat": ("{fut}이 {cur}과 비슷할 것으로 예측해요(변화 {pct}).",
                      "The {fut} is expected to stay close to the {cur} (change {pct})."),
    "sentence_uncertain": ("{fut}이 {cur}보다 {pct} {more} 수 있지만, 불확실성에 비해 변화가 작아 방향을 단정하기 어려워요.",
                           "The {fut} may be {pct} {more} than the {cur}, but the change is small relative to the "
                           "uncertainty, so the direction is unclear."),
    "higher": ("높을", "higher"),
    "lower": ("낮을", "lower"),
    "seasonal_note": ("계절성 뚜렷", "strongly seasonal"),
}

#: Accuracy lines and the MASE note (D.2.3, D.2.8).
ACCURACY_TEXTS: dict[str, tuple[str, str]] = {
    "better": ("과거 데이터 검증: 단순 예측보다 오차 약 {g}% 작음",
               "Backtest: about {g}% lower error than a naive forecast"),
    "better_99": ("과거 데이터 검증: 단순 예측보다 오차 99% 이상 작음",
                  "Backtest: more than 99% lower error than a naive forecast"),
    "similar": ("과거 데이터 검증: 단순 예측과 오차가 비슷함", "Backtest: about as accurate as a naive forecast"),
    "worse": ("과거 데이터 검증: 단순 예측보다 나은 성과를 보이지 못함", "Backtest: did not beat a naive forecast"),
    "no_naive": ("과거 데이터 검증: 단순 예측과 비교한 결과 없음", "Backtest: no comparison with a naive forecast"),
    "too_short": ("데이터가 짧아 정확도를 검증하지 못했어요", "Too short to check accuracy"),
    "details_low": ("기술 참고: 백테스트 MASE {m} — 학습 구간에서 한 기 앞을 계절 단순 예측한 오차를 1로 둔 상대 오차예요. "
                    "1과 비교하지 마세요.",
                    "Technical note: backtest MASE {m} — the error relative to an in-sample one-step seasonal-naive "
                    "forecast (= 1). Do not compare it with 1."),
    "details_high": ("기술 참고: 백테스트 MASE {m} — 학습 구간에서 한 기 앞을 계절 단순 예측한 오차를 1로 둔 상대 오차예요. "
                     "여러 기 앞을 내다보는 예측은 단순 예측보다 정확해도 1을 넘기 쉬워요.",
                     "Technical note: backtest MASE {m} — the error relative to an in-sample one-step seasonal-naive "
                     "forecast (= 1). Multi-step forecasts often score above 1 even when they beat naive."),
}

#: Display -> (ko, en) ``hero.reference_note`` (D.2.3).
REFERENCE_NOTES: dict[str, tuple[str, str]] = {
    "bands_only": ("예측력이 낮아 범위만 참고용으로 보여 드려요.",
                   "The forecast did not beat a naive one, so only the range is shown, for reference."),
    "weak": ("예측 신뢰도가 낮아 중앙선은 참고용이에요.", "Low reliability: treat the centre line as a rough guide."),
}

#: Phase texts (D.2.5): the single English reason used when no phase reason survives ``translate_reason``.
PHASE_TEXTS: dict[str, tuple[str, str]] = {
    "reasons_fallback": ("국면 판단 규칙은 방법 페이지에서 볼 수 있어요.", "See the method page for the rules."),
}

# ---------------------------------------------------------------------------------------------------------------------
# Source block (D.2.2)
# ---------------------------------------------------------------------------------------------------------------------

#: Source label templates (D.2.2).
SOURCE_LABELS: dict[str, tuple[str, str]] = {
    "naver": ("네이버 데이터랩 검색어트렌드(NAVER API HUB)", "Naver DataLab Search Trends (NAVER API HUB)"),
    "naver_legacy": ("네이버 데이터랩 검색어트렌드(네이버 오픈API)", "Naver DataLab Search Trends (Naver Open API)"),
    "wiki": ("Wikimedia Pageviews · {wiki_lang} 위키백과 · 사람 조회만(봇 제외)",
             "Wikimedia Pageviews · {wiki_lang} Wikipedia · human views only (bots excluded)"),
    "upload": ("사용자 업로드 ({filename})", "Your upload ({filename})"),
    "upload_unnamed": ("사용자 업로드", "Your upload"),
    "google_trends_upload": ("Google 트렌드(사용자 업로드, {filename}) · Data source: Google Trends "
                             "(https://www.google.com/trends)",
                             "Google Trends (your upload, {filename}) · Data source: Google Trends "
                             "(https://www.google.com/trends)"),
    "google_trends_upload_unnamed": ("Google 트렌드(사용자 업로드) · Data source: Google Trends "
                                     "(https://www.google.com/trends)",
                                     "Google Trends (your upload) · Data source: Google Trends "
                                     "(https://www.google.com/trends)"),
    "sample": ("trendai 내장 샘플 · 가상 데이터", "trendai built-in sample · synthetic data"),
}

#: Source kind -> (ko, en) ``short`` (summary line, D.2.2).
SOURCE_SHORT: dict[str, tuple[str, str]] = {
    "naver": ("네이버", "Naver"), "wiki": ("위키백과", "Wikipedia"), "upload": ("내 파일", "my file"),
    "google_trends_upload": ("Google 트렌드 업로드", "Google Trends upload"),
    "sample": ("샘플·가상 데이터", "sample · synthetic data"),
}

#: Wikipedia language code -> (ko, en) language name inside the wiki label; other codes print the code itself.
WIKI_LANG_NAMES: dict[str, tuple[str, str]] = {
    "ko": ("한국어", "Korean"), "en": ("영어", "English"),
}

#: Source-line fragments (D.2.2).
SOURCE_LINE_TEXTS: dict[str, tuple[str, str]] = {
    "prefix": ("출처: ", "Source: "),
    "as_of_data": ("데이터 기준 {as_of}", "data as of {as_of}"),
    "as_of_analysed": ("분석 {as_of}", "analysed {as_of}"),
    "original_units": ("원래 단위", "original units"),
    "unit_suffix": (" ({unit_label})", " ({unit_label})"),
    "filters_sep": ("·", " · "),
    "parts_sep": (" · ", " · "),
}

#: Value kind -> (ko, en) label (D.2.2).
VALUE_KIND_LABELS: dict[str, tuple[str, str]] = {
    "relative_index": ("상대 지수(조회 기간 최댓값 = 100)", "relative index (period maximum = 100)"),
    "relative_index_gt": ("상대 지수(0–100, 요청마다 다시 맞춰짐)", "relative index (0–100, rescaled per request)"),
    "pageviews": ("조회수", "page views"),
    "raw": ("원래 단위", "original units"),
}

#: Value kind -> (ko, en) ``value_kind_note`` (D.2.2, verbatim); ``relative_index`` has ``{unit_word}``; extra key
#: ``sample`` for the built-in samples (whose value kind is ``raw``).
VALUE_KIND_NOTES: dict[str, tuple[str, str]] = {
    "relative_index": ("상대 지수: 이 조회에서 가장 많이 검색된 {unit_word} = 100이에요. 실제 검색 횟수가 아니에요.",
                       "Relative index: the {unit_word} with the most searches in this query = 100. Not a search "
                       "count."),
    "relative_index_gt": ("Google 트렌드가 요청마다 0–100으로 다시 맞춘 상대 지수예요. 다른 다운로드와 비교할 수 없어요.",
                          "Google Trends rescales each request to 0–100; not comparable with other downloads."),
    "pageviews": ("위키백과 조회수는 관심의 대리 지표예요. 검색량과 다를 수 있어요.",
                  "Wikipedia page views are a proxy for interest and can differ from search volume."),
    "raw": ("", ""),
    "sample": ("가상 데이터예요. 실제 데이터가 아니에요.", "Synthetic data, not real data."),
}

#: Value kind -> (ko, en) noun used in the chart ``desc`` (D.2.6: ``{value word}``).
VALUE_WORDS: dict[str, tuple[str, str]] = {
    "relative_index": ("상대 지수", "relative index"),
    "relative_index_gt": ("상대 지수", "relative index"),
    "pageviews": ("조회수", "page views"),
    "raw": ("값", "values"),
}

#: Freq -> (ko, en) unit word inside the relative-index note: D 날/day, W 주/week, M 달/month (Q/Y for safety).
INDEX_UNIT_WORDS: dict[str, tuple[str, str]] = {
    "D": ("날", "day"), "W": ("주", "week"), "M": ("달", "month"), "Q": ("분기", "quarter"), "Y": ("해", "year"),
}

#: Naver filter value -> (ko, en).
FILTER_LABELS: dict[str, tuple[str, str]] = {
    "pc": ("PC", "PC"), "mo": ("모바일", "mobile"), "m": ("남성", "male"), "f": ("여성", "female"),
}

# ---------------------------------------------------------------------------------------------------------------------
# Chart, past, future (D.2.6 - D.2.8)
# ---------------------------------------------------------------------------------------------------------------------

#: Chart texts (D.2.6).
CHART_TEXTS: dict[str, tuple[str, str]] = {
    "title": ("{name} — 추이와 예측", "{name} — history and forecast"),
    "desc_head": ("{name} {freq_adj} {value_word}, {years}.", "{name}: {freq_adj} {value_word}, {years}."),
    "desc_peak": ("보통 {peak} 정점.", "Usually peaks in {peak}."),
    "desc_change": ("{fut} {pct} 예상.", "{fut} {pct} expected."),
    "desc_word": ("{word} 전망.", "Outlook: {word}."),
    "x_label": ("{freq} · {range}", "{freq} · {range}"),
    "last_actual": ("마지막 실제값 {date}", "Last actual: {date}"),
    "y_note": ("이 조회의 최댓값(100)보다 작은 검색어라 세로축을 확대해 그렸어요.",
               "This term is small relative to the query's maximum (100), so the vertical axis is zoomed in."),
    "anomaly_text": ("평소보다 크게 {direction} · 실제 {v} / {expected_word} {e}",
                     "Far {direction} the usual range · actual {v} / {expected_word} {e}"),
    "anomaly_up": ("높음", "above"),
    "anomaly_down": ("낮음", "below"),
    "expected": ("예상", "expected"),
    "usual_size": ("평소 규모", "usual size"),
    "changepoint_text": ("{date} · {label} · {strength}", "{date} · {label} · {strength}"),
}

#: Changepoint strength -> (ko, en).
STRENGTH_WORDS: dict[str, tuple[str, str]] = {
    "large": ("큰 전환", "major shift"), "medium": ("뚜렷한 전환", "clear shift"), "small": ("작은 전환", "minor shift"),
}

#: Past texts (D.2.7).
PAST_TEXTS: dict[str, tuple[str, str]] = {
    "summary": ("구조 변화 {n}번 · 이상치 {m}건", "{n} {cp_word} · {m} {an_word}"),
    "changepoint_one": ("", "changepoint"),
    "changepoint_many": ("", "changepoints"),
    "anomaly_one": ("", "anomaly"),
    "anomaly_many": ("", "anomalies"),
    "seasonality_strong": ("계절성이 매우 뚜렷해요(강도 {s}).", "Strongly seasonal (strength {s})."),
    "seasonality_clear": ("계절성이 뚜렷해요(강도 {s}).", "Clearly seasonal (strength {s})."),
    "seasonality_none": ("뚜렷한 계절성은 없어요.", "No clear seasonality."),
    "peak_trough": (" 보통 {peak}에 높고 {trough}에 낮아요.", " Usually high in {peak} and low in {trough}."),
    "peak_trough_weekday": (" 보통 {peak}에 높고 {trough}에 낮아요.", " Usually high on {peak} and low on {trough}."),
}

#: Scenario texts (D.2.8).
SCENARIO_TEXTS: dict[str, tuple[str, str]] = {
    "title": ("{date_label} 한 시점의 값", "Single value at {date_label}"),
    "ref_season": ("같은 계절 시점({date_label})의 실제값", "actual at the same point of the season ({date_label})"),
    "ref_latest": ("최근 실제값({date_label})", "latest actual ({date_label})"),
    "ref_demand": ("최근 {span} 평균 수요량", "average demand over the last {span}"),
    "reference_text": ("비교 기준: {label} {value_text}", "Compared with: {label} {value_text}"),
    "row_with_pct": ("{label} {value_text} (기준 대비 {pct_text})", "{label} {value_text} ({pct_text} vs. reference)"),
    "row_without_pct": ("{label} {value_text}", "{label} {value_text}"),
    "pessimistic": ("비관", "Pessimistic"),
    "base": ("기준", "Base"),
    "optimistic": ("낙관", "Optimistic"),
    "note": ("예측 마지막 시점 한 값의 {level}% 범위예요. 위의 ‘예측 변화’(평균 대 평균)와는 다른 양이에요.",
             "The {level}% range of a single value at the end of the forecast — a different quantity from the "
             "‘Expected change’ above (average vs. average)."),
    "floor": ("0(하한)", "0 (floor)"),
    "range_sep": ("~", "–"),
}

#: Future phase and peak texts (D.2.8).
FUTURE_TEXTS: dict[str, tuple[str, str]] = {
    "hedge_undecided": ("판단 보류", "undecided"),
    "hedge_tentative": ("잠정", "tentative"),
    "hedge_low_confidence": ("참고용", "for reference"),
    "undecided": ("흐름이 보합으로 예측돼 {h} 뒤 국면은 판단하기 어려워요.",
                  "The outlook is flat, so the phase in {h} is hard to judge."),
    "tentative": ("관측치가 적어 잠정 판단이에요.", "Too few observations, so this is a tentative call."),
    "low_confidence": ("예측 신뢰도가 낮아 참고용이에요.", "The forecast reliability is low, so this is for reference only."),
    "stay": ("{h} 뒤에도 {name} 국면이 이어질 것으로 보여요.", "In {h} the phase is expected to stay {name}."),
    "become": ("{h} 뒤에는 {name} 국면으로 바뀔 것으로 보여요.", "In {h} the phase is expected to become {name}."),
    "peak": ("예측 기간 안에 {date_label} 무렵 정점이 예상돼요.", "A peak around {date_label} is expected within the horizon."),
    "no_peak": ("예측 기간 안에 뚜렷한 정점은 보이지 않아요.", "No clear peak within the horizon."),
}

# ---------------------------------------------------------------------------------------------------------------------
# Narrative, summary line, compare (D.2.9, D.2.10, D.4)
# ---------------------------------------------------------------------------------------------------------------------

#: Narrative texts (D.2.9); ``disclaimer`` = G.6 ``detail.disclaimer`` = G.7 ``footer.disclaimer``.
NARRATIVE_TEXTS: dict[str, tuple[str, str]] = {
    "badge_rule": ("규칙 기반 해석", "Rule-based"),
    "badge_ai": ("AI 해석 · Claude", "AI · Claude"),
    "ai_note": ("AI가 작성한 해석이에요. 수치는 위 분석 결과를 근거로 해요.",
                "Written by AI. The figures are based on the analysis above."),
    "ai_privacy_note": ("계산된 요약 수치(JSON)만 Anthropic으로 보내요. 원본 데이터는 보내지 않아요.",
                        "Only the computed summary figures (JSON) are sent to Anthropic, never the original data."),
    "disclaimer": ("예측은 과거 패턴이 이어진다고 가정해요. 갑작스러운 사건이나 외부 변수(가격·광고·휴일)는 반영하지 못해요.",
                   "Forecasts assume past patterns continue. Sudden events and outside factors (prices, ads, "
                   "holidays) are not included."),
}

#: Summary-line templates and suffixes (D.2.10).
SUMMARY_TEXTS: dict[str, tuple[str, str]] = {
    "head": ("[trendai] {name}({short}, {freq_adj}): {phase}", "[trendai] {name} ({short}, {freq_adj}): {phase}"),
    "change": (" · {fut} {pct} 예상", " · {fut} {pct} expected"),
    "word": (" · {word} 전망", " · outlook {word}"),
    "prob": ("(상승 가능성 {prob})", " (chance of rise {prob})"),
    "peak": (" · 보통 {peak} 정점", " · usually peaks in {peak}"),
    "bands_only": (" (참고용: 단순 예측보다 낫지 않음)", " (for reference: did not beat a naive forecast)"),
    "weak": (" (참고용 예측)", " (forecast for reference only)"),
    "as_of": (" (데이터 기준 {as_of})", " (data as of {as_of})"),
}

#: Compare texts (D.4).
COMPARE_TEXTS: dict[str, tuple[str, str]] = {
    "score_note": ("상승 점수는 전망(예측 변화·상승 확률)과 최근 기세(최근 변화·모멘텀)를 합친 0–100 값이에요. 50이 중립이고, "
                   "예상 상승률이 아니에요.",
                   "The rising score combines the outlook (expected change, chance of rise) with the recent "
                   "momentum (recent change, momentum) on a 0–100 scale. 50 is neutral; it is not an expected "
                   "growth rate."),
    "spark_note": ("스파크라인은 카드마다 세로축이 달라요.", "Each card's sparkline has its own vertical scale."),
    "axis_note_relative": ("이 조회에서 가장 많이 검색된 {unit_word} = 100 · 다른 조회와 비교할 수 없어요",
                           "The {unit_word} with the most searches in this query = 100 · not comparable with "
                           "other queries"),
    "axis_note_gt": ("Google 트렌드가 요청마다 0–100으로 다시 맞춘 값 · 다른 다운로드와 비교할 수 없어요",
                     "Rescaled by Google Trends to 0–100 per request · not comparable with other downloads"),
    "index_disabled_same_request": ("같은 조회 안에서 이미 같은 척도예요.",
                                    "These series already share one scale (same request)."),
    "log_disabled": ("0 이하 값이 있어 로그 축을 쓸 수 없어요.", "Values of 0 or less: a log axis is not possible."),
    "index_base_label": ("각 시계열의 {range} 평균을 100으로 맞췄어요", "Each series' average over {range} is set to 100"),
    "index_base_label_one": ("각 시계열의 {date} 값을 100으로 맞췄어요", "Each series' value in {date} is set to 100"),
    "axis_value": ("값", "value"),
    "names_sep": (" vs ", " vs "),
}

# ---------------------------------------------------------------------------------------------------------------------
# Exports (G.8 server-side paragraph, C.10)
# ---------------------------------------------------------------------------------------------------------------------

#: Export labels (G.8, C.10).
EXPORT_TEXTS: dict[str, tuple[str, str]] = {
    "source": ("출처", "Source"),
    "conditions": ("조건", "Conditions"),
    "naver_note": ("네이버 원본 값(ratio)은 이용약관상 포함하지 않았어요",
                   "Raw Naver values (ratio) are excluded under its terms of use"),
    "comment_line": ("{label}: {text}", "{label}: {text}"),
    "group": ("검색어 그룹 {name}({terms})", "keyword group {name} ({terms})"),
    "group_single": ("검색어 {name}", "term {name}"),
    "article": ("문서 {name}", "article {name}"),
    "column": ("열 {name}", "column {name}"),
    "sample": ("샘플 {name}", "sample {name}"),
    "horizon": ("예측 {span}", "forecast {span}"),
    "terms_sep": (", ", ", "),
    "no_filters": ("없음", "none"),
    "sheet_data": ("데이터", "Data"),
    "sheet_summary": ("요약", "Summary"),
    "conf_high": ("높음", "high"),
    "conf_medium": ("보통", "medium"),
    "conf_low": ("낮음", "low"),
}

#: Column-header lists, each a ``(ko_list, en_list)`` pair (C.10, G.8).
EXPORT_COLUMNS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "series": (("날짜", "실제값", "추세", "예측값", "80% 하한", "80% 상한", "95% 하한", "95% 상한"),
               ("date", "actual", "trend", "forecast", "lo80", "hi80", "lo95", "hi95")),
    "ranking": (("순위", "이름", "국면", "전망", "예측 변화(%)", "상승 확률(%)", "최근 변화(%)", "모멘텀(%p/기간)",
                 "상승 점수", "예측 신뢰도"),
                ("rank", "name", "phase", "outlook", "expected change (%)", "chance of rise (%)",
                 "recent change (%)", "momentum (pp/period)", "rising score", "reliability")),
    "values": (("날짜",), ("date",)),
    "xlsx_conditions": (("출처", "검색어 그룹/열", "기간", "단위", "조건(기기·성별·연령)", "데이터 기준", "값의 성격"),
                        ("Source", "Groups/columns", "Period", "Unit", "Filters (device · gender · age)",
                         "Data as of", "Value kind")),
    "summary_sheet": (("이름", "국면", "전망", "예측 변화", "상승 확률", "예측 신뢰도", "정확도", "상승 점수"),
                      ("Name", "Phase", "Outlook", "Expected change", "Chance of rise", "Reliability", "Accuracy",
                       "Rising score")),
}
