"""Web rendering of the message catalog (docs/WEB_SPEC.md C.12, H.2): new codes, WEB_OVERRIDES,
render_web / render_web_msg, josa, fix labels."""

from __future__ import annotations

import hashlib
import json
import re
import string
from pathlib import Path

import pytest

from trendai import messages as M
from trendai.data import DataError, DataNotice, DataWarning
from trendai.messages import L, join, josa, msg, render, render_web, render_web_msg

_HANGUL = re.compile(r"[가-힣]")
_FIELDS = string.Formatter()
_CLI_HINTS = ("--", "파이썬", "Python", "trendai.")

#: every code docs/WEB_SPEC.md C.12 adds (web errors, connector codes, the DataError code, notices)
NEW_CODES = (
    "bad_request invalid_param empty_query too_many_series too_many_keywords term_too_long wiki_group_unsupported "
    "unknown_sample body_too_large length_required unsupported_media_type method_not_allowed cross_origin not_found "
    "result_not_found job_not_found share_not_found share_forbidden share_not_needed upload_expired upload_unreadable "
    "upload_too_many_columns upload_too_large_table upload_too_many_series upload_archive_too_large "
    "upload_excel_needs_pkg upload_no_date_col upload_no_value_col upload_bad_date_col upload_bad_value_col "
    "data_too_short series_failed_web rate_limited server_busy internal_error job_timeout naver_unconfigured "
    "naver_auth naver_rate naver_rate_limited naver_budget_exhausted naver_server source_timeout source_unreachable "
    "source_bad_response wiki_not_found wiki_rate wiki_empty export_xlsx_unavailable export_naver_raw "
    "ai_unavailable ai_naver_blocked ai_readonly ai_failed ai_rate_limited "
    "conn_naver_unconfigured conn_naver_auth conn_naver_rate conn_naver_server conn_naver_http "
    "conn_naver_bad_response conn_naver_network conn_naver_timeout conn_naver_too_many_groups "
    "conn_naver_too_many_keywords conn_naver_bad_filter conn_bad_range conn_timeout table_too_many_columns "
    "naver_missing_filled naver_low_resolution mostly_zero_volume data_short horizon_clamped gt_detected "
    "gt_lt1_half index_unavailable log_hint naver_cached "
    "store_full upload_evicted wiki_not_found_no_naver gaps_filled"
).split()

#: C.12.6 fix actions -> (ko, en) labels
FIX_LABELS = {
    "reload": ("새로고침", "Reload"), "retry": ("다시 시도", "Try again"), "retry_in": ("지금 다시", "Retry now"),
    "go_home": ("처음으로", "Home"), "reanalyze": ("다시 분석", "Analyse again"),
    "use_wiki": ("위키백과로 보기", "View on Wikipedia"), "use_naver": ("네이버로 보기", "View on Naver"),
    "upload_file": ("파일 올리기", "Upload a file"), "reupload": ("다시 올리기", "Upload again"),
    "try_sample": ("샘플로 체험", "Try a sample"), "extend_range": ("기간 늘리기", "Use a longer period"),
    "unit_month": ("월 단위로", "Switch to months"), "group_terms": ("같은 뜻 묶기", "Group similar terms"),
    "remove_series": ("하나 빼기", "Remove one"), "show_formats": ("지원 형식 보기", "Supported formats"),
    "try_paste": ("붙여넣기로 시도", "Try pasting"), "pick_columns": ("열 고르기", "Pick columns"),
    "wiki_title_help": ("위키백과에서 제목 찾기", "Search Wikipedia titles"), "use_log": ("로그 축으로", "Use log scale"),
    "copy_link": ("주소 복사", "Copy link"), "export_csv": ("CSV로 받기", "Download CSV"),
    "reanalyze_single": ("이 검색어만 다시 분석", "Analyse this term alone"),
}

#: sample parameters of the new codes
NEW_PARAMS = {
    "param": "range", "max": 5, "group": "선크림", "name": "growing", "max_mb": 10, "n": 3, "col": "날짜",
    "retry_after": 30, "limit": 20, "title": "선크림", "status": 401, "error": "URLError: boom", "what": "device",
    "value": "'tv'", "terms": "‘선크림’", "region": "대한민국", "span": L("104주", "104 weeks"), "ratio": 20,
    "fetched": "2026.11.25 09:12", "dates": "2023.06, 2024.07",
}


def _field_names(template: str) -> set[str]:
    return {f for _, f, _, _ in _FIELDS.parse(template) if f}


def _dummy_params(code: str) -> dict:
    """A value for every parameter of ``code`` (library and web templates), typed by its format spec."""
    out: dict = {}
    templates = [*M.MESSAGES[code], *M.WEB_OVERRIDES.get(code, ())]
    for template in templates:
        for _, field, spec, _ in _FIELDS.parse(template):
            if not field:
                continue
            if spec and spec.endswith("%"):
                out[field] = 0.25
            elif spec:
                out[field] = 12345
            elif field in ("cols", "columns"):
                out[field] = ["번호", "date"]
            elif field in ("unit", "own"):
                out[field] = M.unit("M")
            elif field in ("detail",):
                out[field] = msg("no_date_col", columns=["a", "b"])
            elif field in ("label", "date_format"):
                out[field] = "YYYY (연도)"
            elif field == "examples":
                out[field] = "'a', 'b'"
            else:
                out[field] = "x1"
    return out


# --------------------------------------------------------------------------- the new codes


def test_new_codes_render_both_languages():
    missing = [c for c in NEW_CODES if c not in M.MESSAGES]
    assert not missing
    for code in NEW_CODES:
        ko_t, en_t = M.MESSAGES[code]
        params = {k: NEW_PARAMS[k] for k in _field_names(ko_t) | _field_names(en_t)}
        for lang in ("ko", "en"):
            for text in (render(code, params, lang), render_web(code, params, lang)):
                assert "{" not in text and "}" not in text, (code, text)
        assert _HANGUL.search(render(code, params, "ko")), code
        en = render_web(code, params, "en")
        assert not _HANGUL.search(en.replace("선크림", "").replace("날짜", "").replace("대한민국", "")), (code, en)


def test_fix_labels_exist():
    for action, (ko, en) in FIX_LABELS.items():
        assert M.fix_label(action) == ko and M.fix_label(action, "ko") == ko
        assert M.fix_label(action, "en") == en
    assert {c[4:] for c in M.MESSAGES if c.startswith("fix_")} == set(FIX_LABELS)
    with pytest.raises(KeyError):
        M.fix_label("does_not_exist")


def test_catalog_matches_spec_tables():
    """The texts of C.12.2, C.12.3, C.12.5 and C.12.6 are copied verbatim from the spec."""
    spec = Path(__file__).resolve().parents[1] / "docs" / "WEB_SPEC.md"
    if not spec.exists():
        pytest.skip("docs/WEB_SPEC.md not present")
    lines = spec.read_text(encoding="utf-8").splitlines()

    def rows(start: str, stop: str) -> list[list[str]]:
        i = next(k for k, line in enumerate(lines) if line.startswith(start))
        j = next(k for k in range(i + 1, len(lines)) if lines[k].startswith(stop))
        out = [[c.strip() for c in line.strip().strip("|").split("|")]
               for line in lines[i:j] if line.startswith("|") and not line.startswith("|---")]
        return out[1:]

    expected: dict[str, tuple[str, str]] = {}
    for c in rows("#### C.12.2", "#### C.12.3"):
        expected[c[0].strip("`")] = (c[2], c[3])
    for c in rows("#### C.12.3", "Any other"):
        m = re.match(r"`([a-z_]+)`", c[0])
        if m and not c[1].startswith("(existing)"):
            expected[m.group(1)] = (c[1], c[2])
    for c in rows("#### C.12.5", "Existing data-layer"):
        expected[re.match(r"`([a-z_0-9]+)`", c[0]).group(1)] = (c[2], c[3])
    for c in rows("#### C.12.6", "#### C.12.7"):
        expected["fix_" + c[0].strip("`")] = (c[1], c[2])
    assert len(expected) == 100
    for code, texts in expected.items():
        assert M.MESSAGES[code] == texts, code
    # the C.12.7 table: the required web wordings
    for c in rows("#### C.12.7", "### C.13"):
        code = c[0].strip("`")
        if code in M.MESSAGES and "(rewrite" not in c[1]:
            ko = c[1].split(" (rendered:")[0]
            assert M.WEB_OVERRIDES[code] == (ko, c[2]), code


# --------------------------------------------------------------------------- web wording


def test_render_web_has_no_cli_hints():
    for code in M.MESSAGES:
        params = _dummy_params(code)
        for lang in ("ko", "en"):
            text = render_web(code, params, lang)
            assert not any(h in text for h in _CLI_HINTS), (code, lang, text)
            assert "{" not in text.replace("{날짜: 값}", "").replace("{date: value}", ""), (code, text)


def test_web_overrides_use_library_params():
    """Whatever renders the library text renders the web text (no new parameters)."""
    for code, (ko, en) in M.WEB_OVERRIDES.items():
        lib = _field_names(M.MESSAGES[code][0]) | _field_names(M.MESSAGES[code][1])
        assert _field_names(ko) <= lib and _field_names(en) <= lib, code
        assert _HANGUL.search(ko) and not _HANGUL.search(en.replace("�", "")), code
        assert not any(h in ko + en for h in _CLI_HINTS), code


def test_rownum_notice_reads_naturally():
    text = render_web("rownum_cols_skipped", {"cols": ["번호"]})
    assert text == "행 번호/일련번호로 보이는 열 ‘번호’는 분석에서 제외했어요. 포함하려면 확인 단계에서 값 열로 고르세요."
    assert "['" not in text and "은(는)" not in text
    en = render_web("rownum_cols_skipped", {"cols": ["id", "no"]}, "en")
    assert en.startswith("The column(s) ‘id’, ‘no’ look like row numbers")


def test_render_web_msg_joined():
    note = join(msg("date_order_ambiguous", label="기타 날짜 형식 (일/월 순)"),
                msg("dates_unreadable_rows", n=2, examples="'31/02/2024', 'foo'"))
    ko, en = render_web_msg(note), render_web_msg(note, "en")
    assert "일/월 순서가 모호해" in ko and "날짜로 읽을 수 없는 값 2개(예: ‘31/02/2024’, ‘foo’)" in ko
    assert "ambiguous" in en and "2 row(s) whose date could not be read (e.g. ‘31/02/2024’, ‘foo’)" in en
    assert "other date format (day first)" in en and not _HANGUL.search(en)
    # the whole message, also when it arrives as a captured warning instance
    assert render_web_msg(DataNotice(note), "en") == en


def test_render_web_msg_nested_and_L():
    m = msg("series_failed", series="매출", detail=msg("no_date_col", columns=["a", "b"]))
    ko, en = render_web_msg(m), render_web_msg(m, "en")
    assert ko == "‘매출’을 분석하지 못했어요: 날짜 열을 찾지 못했어요. 확인 단계에서 날짜 열을 골라 주세요."
    assert en == "‘매출’ could not be analysed: No date column found. Pick it in the confirm step."
    assert "--date-col" in str(m)          # the library text keeps the CLI hint
    gaps = msg("gaps_interpolated", name="'v': ", unit=M.unit("M"), filled=5, total=20, share=0.25, observed=15)
    assert render_web_msg(gaps).startswith("‘v’: 월 단위로 맞추면서 빈 기간 5개(전체 20개 중 25%)를")
    assert render_web_msg(gaps, "en").startswith("‘v’: on a month grid, 5 empty periods (25% of 20)")
    edge = L(" 시계열의 처음/끝 부분이 잘렸을 수 있습니다.", " The start or end of the series may be cut off.")
    col = msg("col_values_unreadable", col="v", n=1, examples="'abc'", edge=edge)
    assert render_web_msg(col) == ("‘v’ 열: 숫자로 읽을 수 없는 값 1개(예: ‘abc’)를 빈 값으로 처리했어요. "
                                   "시계열의 처음이나 끝이 잘렸을 수 있어요.")
    assert render_web_msg(col, "en").endswith("were treated as missing. The start or end of the series may be cut off.")
    wide = msg("wide_table", n=2, label=L("YYYY.MM", "YYYY.MM"))
    assert render_web_msg(wide, "en").endswith("(date format: YYYY.MM).")


def test_render_web_msg_lists_and_examples():
    assert render_web("only_rownum_cols", {"cols": ["번호", "순번"]}).startswith(
        "분석할 숫자 값 열이 없어요(행 번호 열 ‘번호’, ‘순번’만 있음)")
    assert "(e.g. ‘a’, ‘b’)" in render_web("values_unreadable", {"name": "", "n": 2, "examples": "'a', 'b'"}, "en")
    # not a repr list: kept as it is
    assert "(e.g. a, b)" in render_web("values_unreadable", {"name": "", "n": 2, "examples": "a, b"}, "en")
    assert "Columns: ‘date’, ‘v’" in render_web("date_col_missing", {"col": "x", "columns": ["date", "v"]}, "en")


def test_render_web_msg_exceptions():
    generic_ko, generic_en = M.MESSAGES["series_failed_web"]
    err = DataError(msg("too_few_obs", name="'v': ", n=1))
    assert render_web_msg(err) == "‘v’: 관측치가 2개 이상 필요해요(유효한 값 1개)."
    assert render_web_msg(ValueError("secret detail /etc/passwd")) == generic_ko
    assert render_web_msg(RuntimeError("x"), "en") == generic_en
    assert render_web_msg(DataWarning("free text with --flag")) == generic_ko
    assert render_web_msg("plain text --flag") == generic_ko


def test_render_web_unknown_code_and_lang():
    with pytest.raises(KeyError):
        render_web("no_such_code")
    assert render_web("not_found", lang="fr") == M.MESSAGES["not_found"][0]


def test_horizon_clamped_particle():
    assert render_web("horizon_clamped", {"span": L("104주", "104 weeks")}) == "예측 기간을 최대치인 104주로 줄였어요."
    assert render_web("horizon_clamped", {"span": L("10년", "10 years")}) == "예측 기간을 최대치인 10년으로 줄였어요."
    assert render_web("horizon_clamped", {"span": L("36개월", "36 months")}, "en") == \
        "The forecast horizon was reduced to the maximum, 36 months."


def test_josa():
    assert josa("‘번호’", "은/는") == "‘번호’는"
    assert josa("104주", "으로/로") == "104주로"
    assert josa("36개월", "으로/로") == "36개월로"
    assert josa("10년", "으로/로") == "10년으로"
    assert josa("CP949", "으로/로") == "CP949로"
    assert josa("UTF-16", "으로/로") == "UTF-16으로"
    assert josa("수능", "이/가") == "수능이"
    assert josa("사과", "을/를") == "사과를" and josa("수박", "과/와") == "수박과"
    assert josa("‘매출’) ", "을/를") == "‘매출’) 을"
    assert josa("17", "은/는") == "17은" and josa("2", "이/가") == "2가" and josa("100", "으로/로") == "100으로"
    assert josa("Excel", "은/는") == "Excel는" and josa("", "이/가") == "가"
    with pytest.raises(ValueError):
        josa("x", "에/에서")


# --------------------------------------------------------------------------- library rendering unchanged

#: sha256 of the 83 templates of the catalog before the web app (commit db8b40d)
_OLD_DIGEST = "828ebcc7007c7ba422714b153bdf2fd2326abc5a8a39e1befffafaff67ffb028"
_OLD_CODES = (
    "all_dates_equal bad_freq bad_how col_values_unreadable conn_bad_granularity conn_bad_lang conn_date_format "
    "conn_date_invalid conn_date_nat conn_need_pytrends conn_need_requests conn_network conn_no_article "
    "conn_no_keyword conn_pytrends_broken conn_start_after_end conn_trends_dates conn_trends_empty "
    "conn_trends_failed conn_trends_no_col conn_trends_no_valid conn_wiki_404 conn_wiki_429 conn_wiki_empty "
    "conn_wiki_error conn_wiki_http conn_wiki_json conn_wiki_no_valid csv_ragged date_col_missing "
    "date_col_unreadable date_order_ambiguous date_ym_ambiguous dates_unreadable_rows duplicates_merged "
    "excel_is_html excel_is_text excel_needs_pkg excel_unreadable excel_unreadable_engine excel_wrong_ext "
    "file_empty file_not_found file_unreadable freq_needs_two_dates gaps_interpolated headerless horizon_long "
    "horizon_min horizon_not_int horizon_not_positive horizon_out_of_range horizon_too_long index_numeric "
    "index_unreadable json_array_shape json_empty_array json_empty_object json_shape json_invalid json_to_table "
    "no_date_col no_numeric_col no_valid_values only_rownum_cols path_is_dir rownum_cols_skipped save_path_is_dir "
    "series_empty series_failed series_required table_empty_rows table_empty_values too_few_after_regularise "
    "too_few_obs upsampled utf8_broken value_col_missing value_col_unreadable values_infinite values_unreadable "
    "weekday_only wide_table"
).split()


def test_existing_render_unchanged():
    old = {code: M.MESSAGES[code] for code in _OLD_CODES}
    blob = json.dumps(sorted(old.items()), ensure_ascii=False, separators=(",", ":"))
    assert len(old) == 83 and hashlib.sha256(blob.encode()).hexdigest() == _OLD_DIGEST
    for code in _OLD_CODES:
        params = _dummy_params(code)
        for lang, template in zip(("ko", "en"), M.MESSAGES[code]):
            plain = {k: (v.get(lang) if isinstance(v, L) else v.text(lang) if isinstance(v, M.Msg) else v)
                     for k, v in params.items()}
            assert render(code, params, lang) == template.format(**plain), code
    # Msg.text / str() still give the library text (with the CLI hint)
    m = msg("rownum_cols_skipped", cols=["id"])
    assert str(m) == "행 번호/일련번호로 보이는 열 ['id']은(는) 분석에서 제외했습니다 (포함하려면 --value-col 로 지정)."
