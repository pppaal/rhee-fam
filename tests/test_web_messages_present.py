"""Every message code the web backend can emit exists in ``messages.MESSAGES`` (C.12; WP2 adds the texts) and
renders in both languages without leftover placeholders."""

from __future__ import annotations

import re

import pytest

from trendai import messages
from trendai.web import errors, viewmodel

#: C.12.3 target codes and C.12.4 sources that WP1 maps (the web codes themselves are in errors.STATUS)
CONNECTOR_CODES = ["conn_naver_unconfigured", "conn_naver_auth", "conn_naver_rate", "conn_naver_server",
                   "conn_naver_http", "conn_naver_bad_response", "conn_naver_network", "conn_naver_timeout",
                   "conn_naver_too_many_groups", "conn_naver_too_many_keywords", "conn_naver_bad_filter",
                   "conn_bad_range", "conn_timeout", "conn_wiki_404", "conn_wiki_429", "conn_wiki_empty",
                   "conn_network"]
DATA_CODES = ["no_date_col", "no_numeric_col", "only_rownum_cols", "date_col_missing", "date_col_unreadable",
              "value_col_missing", "value_col_unreadable", "file_empty", "table_too_many_columns", "excel_needs_pkg",
              "too_few_obs", "too_few_after_regularise", "series_empty", "no_valid_values"]
NOTICE_CODES = ["naver_missing_filled", "naver_low_resolution", "mostly_zero_volume", "data_short", "horizon_clamped",
                "gt_detected", "gt_lt1_half", "index_unavailable", "log_hint", "naver_cached", "horizon_long", "gaps_filled"]
FIX_ACTIONS = ["reload", "retry", "retry_in", "go_home", "reanalyze", "use_wiki", "use_naver", "upload_file",
               "reupload", "try_sample", "extend_range", "unit_month", "group_terms", "remove_series",
               "show_formats", "try_paste", "pick_columns", "wiki_title_help", "use_log", "copy_link", "export_csv",
               "reanalyze_single"]
PARAMS = {"param": "range", "max": 5, "group": "선크림", "name": "x", "max_mb": 10, "n": 3, "col": "날짜",
          "retry_after": 30, "limit": 20, "title": "인공지능", "span": "26주", "terms": "‘a’", "region": "KR",
          "ratio": 20, "fetched": "2026.11.25 09:12", "horizon": 100, "reason": "daily_limit",
          "dates": "2023.06"}


def test_web_error_codes_present():
    missing = [code for code in errors.STATUS if code not in messages.MESSAGES]
    assert missing == []


def test_mapped_connector_and_data_codes_present():
    assert [c for c in CONNECTOR_CODES + DATA_CODES if c not in messages.MESSAGES] == []


def test_notice_codes_present():
    assert [c for c in NOTICE_CODES if c not in messages.MESSAGES] == []
    assert set(viewmodel.NOTICE_DEFAULTS) <= set(NOTICE_CODES) | {"gt_detected", "gt_lt1_half"}


def test_fix_actions_present():
    used = {a for actions in errors.DEFAULT_FIXES.values() for a in actions}
    used |= {a for _lvl, actions in viewmodel.NOTICE_DEFAULTS.values() for a in actions}
    assert used <= set(FIX_ACTIONS)
    assert [a for a in FIX_ACTIONS if f"fix_{a}" not in messages.MESSAGES] == []
    for action in FIX_ACTIONS:
        assert messages.fix_label(action, "ko") and messages.fix_label(action, "en")


@pytest.mark.parametrize("code", sorted(set(errors.STATUS) | set(NOTICE_CODES)))
def test_web_texts_render(code):
    for lang in ("ko", "en"):
        text = messages.render_web(code, PARAMS, lang)
        assert text and not re.search(r"\{[a-z_]+\}", text), (code, lang, text)
        assert "--" not in text and "Python" not in text and "파이썬" not in text
        if lang == "en":
            assert not re.search(r"[가-힣]", re.sub("|".join(map(re.escape, ("선크림", "인공지능", "26주", "날짜"))),
                                                    "", text)), (code, text)


def test_every_error_renders_as_envelope():
    for code in errors.STATUS:
        obj = errors.error_object(errors.WebError(code, params=PARAMS), "abcd1234")
        assert obj["message_ko"] and obj["message_en"] and len(obj["fixes"]) <= 3
