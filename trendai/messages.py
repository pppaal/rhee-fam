"""Message catalog of the data layer: one code + params, rendered in Korean or English.

The texts of :class:`trendai.data.DataError` / :class:`~trendai.data.DataWarning` /
:class:`~trendai.data.DataNotice`, of ``pipeline.SeriesFailedWarning`` and of
``connectors.ConnectorError`` are built from this catalog.  Library users see the Korean text (``str(exc)`` / ``str(w.message)``,
unchanged); the command line renders the same message in the language of
``--lang`` (``exc.text('en')``)::

    >>> m = msg("too_few_obs", name="'sales': ", n=1)
    >>> str(m)
    "'sales': 관측치가 2개 이상 필요합니다 (유효한 값 1개)."
    >>> m.text("en")
    "'sales': at least 2 observations are needed (1 valid value)."

* :func:`msg` builds a :class:`Msg`: a ``str`` (the Korean text, so every existing
  ``str`` use keeps working) that also carries its ``parts`` -- ``(code, params)``
  pairs -- and renders itself with :meth:`Msg.text`.
* A parameter that differs by language is an :class:`L` (``L(ko='월', en='month')``);
  a parameter that is itself a :class:`Msg` (e.g. the reason inside "could not be
  analysed -- <reason>") is rendered in the same language.
* :class:`Localised` is the mixin of the exception / warning classes: ``.code``,
  ``.params`` and ``.text(lang)`` read the :class:`Msg` they were raised with
  (plain-string messages fall back to ``str``).

This module imports nothing from the package (``data`` imports it).
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Mapping, NamedTuple, Optional

__all__ = ["MESSAGES", "L", "Msg", "msg", "join", "render", "localise", "Localised",
           "unit", "format_label", "FREQ_EN"]


class L(NamedTuple):
    """A parameter value that differs by language."""

    ko: str
    en: str

    def get(self, lang: str) -> str:
        return self.en if lang == "en" else self.ko


#: data granularity words ("월 단위" / "a month grid"), the English side of data.FREQ_KO
FREQ_EN: dict[str, str] = {"D": "day", "W": "week", "M": "month", "Q": "quarter", "Y": "year"}
_FREQ_KO: dict[str, str] = {"D": "일", "W": "주", "M": "월", "Q": "분기", "Y": "년"}


def unit(freq: str) -> L:
    """The granularity word of ``freq`` in both languages: ``L('월', 'month')``."""
    f = str(freq)
    return L(_FREQ_KO.get(f, f), FREQ_EN.get(f, f))


# Korean pieces of the date-format labels of data.parse_dates (they are shown in the CLI's
# "date column ... (format: ...)" note and in the notices) -> English
_LABEL_EN: tuple[tuple[str, str], ...] = (
    ("YYYY년 M월 D일", "Korean year-month-day text"),
    ("YYYY년 M월", "Korean year-month text"),
    ("YYYY Qn (분기)", "YYYY Qn (quarter)"),
    ("YYYY-Www (ISO 주)", "YYYY-Www (ISO week)"),
    ("YYYY (연도)", "YYYY (year)"),
    ("Unix 초", "Unix seconds"),
    ("엑셀 날짜 일련번호", "Excel date serial"),
    ("날짜/시간", "date/time"),
    ("기타 날짜 형식", "other date format"),
    (" (일/월 순)", " (day first)"),
    ("YYYY + 월", "YYYY + month"),
    ("YYYY + 분기", "YYYY + quarter"),
    ("(열 머리글)", "(column headers)"),
)
_HANGUL = re.compile(r"[가-힣]")


def format_label(label: Any, lang: str = "ko") -> str:
    """A date-format label of ``data.parse_dates`` (or the wide-table date column placeholder)
    in ``lang``: ``'YYYY (연도)'`` -> ``'YYYY (year)'``.  Korean is returned unchanged."""
    text = "" if label is None else str(label)
    if lang != "en" or not _HANGUL.search(text):
        return text
    for ko, en in _LABEL_EN:
        text = text.replace(ko, en)
    return text


#: code -> (Korean template, English template); ``{name}`` is a series label such as ``'sales': ``
#: (or ``''``), ``{what}`` a file name.  Korean is the default text of every exception / warning.
MESSAGES: dict[str, tuple[str, str]] = {
    # ---- frequencies, horizons
    "bad_freq": ("알 수 없는 주기 {freq!r}입니다. D/W/M/Q/Y 중 하나를 지정하세요.",
                 "Unknown frequency {freq!r}: use one of D/W/M/Q/Y."),
    "freq_needs_two_dates": ("주기를 추정하려면 날짜가 2개 이상 필요합니다.",
                             "At least 2 dates are needed to infer the frequency."),
    "all_dates_equal": ("모든 날짜가 같습니다. 날짜 열이 올바른지 확인하세요 (--date-col).",
                        "All dates are the same. Check that the date column is right (--date-col)."),
    "horizon_min": ("예측 기간(horizon)은 1 이상이어야 합니다.", "The forecast horizon must be at least 1."),
    "bad_how": ("how 는 'mean' 또는 'sum' 이어야 합니다.", "how must be 'mean' or 'sum'."),
    "horizon_not_int": ("예측 기간(horizon)은 정수여야 합니다: {value!r}",
                        "The forecast horizon must be an integer: {value!r}"),
    "horizon_not_positive": ("예측 기간(horizon)은 1 이상의 정수여야 합니다: {value!r}",
                             "The forecast horizon must be a whole number of at least 1: {value!r}"),
    "horizon_too_long": ("예측 기간(horizon)이 너무 깁니다: {horizon:,}기간 (최대 {max:,}기간까지 지원합니다).",
                         "The forecast horizon is too long: {horizon:,} periods (at most {max:,} are "
                         "supported)."),
    "horizon_out_of_range": ("예측 기간(horizon) {horizon:,}기간은 날짜로 표현할 수 있는 범위를 넘습니다. "
                             "예측 기간을 줄여 주세요.",
                             "A forecast horizon of {horizon:,} periods goes beyond the dates that can be "
                             "represented. Use a shorter horizon."),
    "horizon_long": ("예측 기간({horizon:,}기간)이 데이터 길이({n:,}개)의 {ratio}배를 넘습니다. 이렇게 먼 미래는 "
                     "과거 패턴만으로 추정하기 어려워 불확실성이 매우 큽니다.",
                     "The forecast horizon ({horizon:,} periods) is more than {ratio} times the length of "
                     "the data ({n:,} observations). A future that far away is hard to estimate from past "
                     "patterns alone, so the uncertainty is very large."),
    "series_failed": ("'{series}' 분석 실패 — {detail}", "'{series}' could not be analysed — {detail}"),
    # ---- dates
    "date_order_ambiguous": ("날짜의 일/월 순서가 모호해(모든 값이 12 이하) '{label}' 순서로 읽었습니다. "
                             "다르다면 날짜를 YYYY-MM-DD 형식으로 바꿔 주세요.",
                             "The day/month order of the dates is ambiguous (every value is 12 or less), so "
                             "they were read as '{label}'. If that is wrong, write the dates as YYYY-MM-DD."),
    "date_ym_ambiguous": ("날짜 '{value}'의 연/월 순서가 모호해 '{label}'로 읽었습니다.",
                          "The year/month order of dates such as '{value}' is ambiguous, so they were read "
                          "as '{label}'."),
    "dates_unreadable_rows": ("날짜로 읽을 수 없는 값 {n}개(예: {examples})가 있는 행은 제외했습니다.",
                              "{n} row(s) whose date could not be read (e.g. {examples}) were left out."),
    "index_numeric": ("인덱스가 날짜가 아니라 숫자({lo}~{hi})입니다. 날짜(DatetimeIndex)를 붙이거나 "
                      "trendai.from_values(values, start='2020-01-01', freq='M')로 만드세요. "
                      "연도라면 1000~2200 사이 정수여야 합니다.",
                      "The index holds numbers ({lo} to {hi}), not dates. Attach dates (a DatetimeIndex) "
                      "or build the series with trendai.from_values(values, start='2020-01-01', freq='M'). "
                      "Years must be whole numbers between 1000 and 2200."),
    "index_unreadable": ("인덱스를 날짜로 읽을 수 없습니다 (예: {examples}). 날짜 형식의 인덱스가 필요합니다.",
                         "The index cannot be read as dates (e.g. {examples}); a date index is needed."),
    # ---- series cleaning and regularisation
    "series_required": ("pandas Series(날짜 인덱스 + 숫자 값)가 필요합니다. "
                        "숫자 목록이라면 trendai.from_values(...)를 쓰세요.",
                        "A pandas Series (date index + numeric values) is needed. For a plain list of "
                        "numbers use trendai.from_values(...)."),
    "series_empty": ("시계열이 비어 있습니다.", "The series is empty."),
    "values_unreadable": ("{name}숫자로 읽을 수 없는 값 {n}개(예: {examples})를 결측으로 처리했습니다.",
                          "{name}{n} value(s) that are not numbers (e.g. {examples}) were treated as "
                          "missing."),
    "values_infinite": ("{name}무한대(inf) 값 {n}개를 제외했습니다.", "{name}{n} infinite value(s) were left out."),
    "no_valid_values": ("{name}유효한 숫자 값이 하나도 없습니다.", "{name}there is no valid numeric value."),
    "weekday_only": ("{name}평일(영업일)에만 값이 있는 데이터입니다. 실제 관측 {n_obs}개 사이의 주말 {filled}개를 "
                     "선형 보간해 달력일 기준({n_out}개)으로 분석하며, 예측 날짜에도 주말이 포함됩니다. "
                     "주 단위로 보려면 --freq W (파이썬: freq='W')를 지정하세요.",
                     "{name}values exist on weekdays (business days) only. The {filled} weekend days between "
                     "the {n_obs} observations were interpolated linearly, so the data is analysed on "
                     "calendar days ({n_out}) and the forecast dates include weekends. For a weekly view "
                     "use --freq W (Python: freq='W')."),
    "gaps_interpolated": ("{name}{unit} 단위 격자에 맞추면서 빈 기간 {filled}개(전체 {total}개 중 {share:.0%})를 "
                          "선형 보간했습니다. 실제 관측은 {observed}개입니다.",
                          "{name}on a {unit} grid, {filled} empty periods ({share:.0%} of {total}) were "
                          "interpolated linearly; {observed} were actually observed."),
    "upsampled": ("{name}{own} 단위 데이터를 더 촘촘한 {unit} 단위로 바꾸면서 {filled:,}개 기간(전체 {total:,}개 중 "
                  "{share:.0%})을 선형 보간했습니다. 실제 관측은 {observed:,}개뿐이라 정확도와 확률이 실제보다 좋아 "
                  "보일 수 있습니다.",
                  "{name}data with one value per {own} were spread over a finer {unit} grid: {filled:,} of "
                  "{total:,} periods ({share:.0%}) were interpolated linearly and only {observed:,} were observed, "
                  "so accuracy and probabilities can look better than they are."),
    "duplicates_merged": ("{name}같은 {unit} 단위 기간에 값이 여러 개 있어 {how} 합쳤습니다 "
                          "(관측 {n_raw}개 → {observed}개 기간). 다른 주기로 보려면 --freq 를 지정하세요.",
                          "{name}several values fell in the same {unit} period and were {how} "
                          "({n_raw} observations → {observed} periods). Use --freq for another frequency."),
    "too_few_obs": ("{name}관측치가 2개 이상 필요합니다 (유효한 값 {n}개).",
                    "{name}at least 2 observations are needed ({n} valid value(s))."),
    "too_few_after_regularise": ("{name}{unit} 단위로 정리하면 관측치가 {n}개뿐입니다 (2개 이상 필요). "
                                 "더 짧은 주기(--freq)를 지정하거나 데이터를 늘리세요.",
                                 "{name}on a {unit} grid only {n} observation(s) remain (at least 2 are "
                                 "needed). Choose a shorter frequency (--freq) or add data."),
    # ---- tables
    "wide_table": ("날짜가 열 방향으로 놓인 가로형 표로 보고 행마다 시계열 {n}개로 읽었습니다 (날짜 형식: {label}).",
                   "Read as a wide table with the dates across the columns: {n} series, one per row "
                   "(date format: {label})."),
    "table_empty_rows": ("표가 비어 있습니다 (읽을 행이 없습니다).", "The table is empty (no rows to read)."),
    "table_empty_values": ("표가 비어 있습니다 (값이 있는 행이 없습니다).", "The table is empty (no row has a value)."),
    "date_col_missing": ("날짜 열 {col!r}을(를) 찾지 못했습니다. 열 목록: {columns}",
                         "Date column {col!r} not found. Columns: {columns}"),
    "date_col_unreadable": ("{col!r} 열을 날짜로 읽을 수 없습니다 (예: {examples}). 지원 형식: 2024-01-05, "
                            "2024-01, 2024.01, 202401, 20240105, 2024년 1월, 2024 Q1, 2024년 1분기, 2024-W01, "
                            "05.01.2024, 연도(2024), Unix 시간.",
                            "Column {col!r} cannot be read as dates (e.g. {examples}). Supported formats: "
                            "2024-01-05, 2024-01, 2024.01, 202401, 20240105, 2024 Q1, 2024-W01, 05.01.2024, "
                            "years (2024), Unix time and Korean year-month text."),
    "no_date_col": ("날짜 열을 찾지 못했습니다. --date-col 로 날짜 열을 지정하세요 (파이썬: date_col=...). "
                    "열 목록: {columns}",
                    "No date column found. Name it with --date-col (Python: date_col=...). Columns: {columns}"),
    "value_col_missing": ("값 열 {col!r}을(를) 찾지 못했습니다. 열 목록: {columns}",
                          "Value column {col!r} not found. Columns: {columns}"),
    "value_col_unreadable": ("값 열 {col!r}을(를) 숫자로 읽을 수 없습니다 (예: {examples}).",
                             "Value column {col!r} cannot be read as numbers (e.g. {examples})."),
    "only_rownum_cols": ("분석할 숫자 값 열이 없습니다 (행 번호 열 {cols}만 있음). --value-col 로 값 열을 지정하세요.",
                         "There is no numeric value column to analyse (only the row-number column(s) {cols}). "
                         "Name the value column with --value-col."),
    "no_numeric_col": ("숫자 값 열을 찾지 못했습니다. --value-col 로 값 열을 지정하세요 (파이썬: value_cols=[...]). "
                       "열 목록: {columns}",
                       "No numeric value column found. Name it with --value-col (Python: value_cols=[...]). "
                       "Columns: {columns}"),
    "rownum_cols_skipped": ("행 번호/일련번호로 보이는 열 {cols}은(는) 분석에서 제외했습니다 "
                            "(포함하려면 --value-col 로 지정).",
                            "The column(s) {cols} look like row numbers and were left out (name them with "
                            "--value-col to include them)."),
    "col_values_unreadable": ("'{col}' 열: 숫자로 읽을 수 없는 값 {n}개(예: {examples})를 결측으로 처리했습니다.{edge}",
                              "Column '{col}': {n} value(s) that are not numbers (e.g. {examples}) were "
                              "treated as missing.{edge}"),
    # ---- files
    "utf8_broken": ("{what}: UTF-8 파일에 깨진 바이트 {n}개가 있어 '�'로 바꿔 읽었습니다.",
                    "{what}: {n} broken byte(s) in the UTF-8 file were read as '�'."),
    "file_empty": ("{what}: 파일이 비어 있습니다.", "{what}: the file is empty."),
    "headerless": ("{what}: 머리글(열 이름) 행이 없는 파일로 보고 첫 행부터 데이터로 읽었습니다.",
                   "{what}: read as a file without a header row (column names); the first row is data."),
    "csv_ragged": ("{what}: CSV 형식 오류 — 행마다 열 개수가 다릅니다 ({error}).{hint}",
                   "{what}: CSV format error — the rows have different numbers of columns ({error}).{hint}"),
    "json_empty_array": ("{what}: JSON 배열이 비어 있습니다.", "{what}: the JSON array is empty."),
    "json_array_shape": ("{what}: JSON 배열 형식을 이해할 수 없습니다 (객체 또는 [날짜, 값] 목록이 필요).",
                         "{what}: unsupported JSON array (a list of objects or of [date, value] pairs is "
                         "needed)."),
    "json_empty_object": ("{what}: JSON 객체가 비어 있습니다.", "{what}: the JSON object is empty."),
    "json_shape": ("{what}: JSON 구조를 이해할 수 없습니다 (레코드 목록, {{날짜: 값}} 또는 열 객체가 필요).",
                   "{what}: unsupported JSON structure (a list of records, {{date: value}} or an object of "
                   "columns is needed)."),
    "json_invalid": ("{what}: 올바른 JSON이 아닙니다 ({error}, {line}행).", "{what}: invalid JSON ({error}, line {line})."),
    "json_to_table": ("{what}: JSON 구조를 표로 바꿀 수 없습니다 ({error}).",
                      "{what}: the JSON structure cannot be turned into a table ({error})."),
    "excel_is_html": ("{what}: 확장자는 엑셀이지만 실제로는 HTML/XML 표입니다. 엑셀에서 열어 "
                      "CSV(UTF-8) 또는 .xlsx로 다시 저장하세요.",
                      "{what}: the extension says Excel, but the file is an HTML/XML table. Open it in Excel "
                      "and save it again as CSV (UTF-8) or .xlsx."),
    "excel_is_text": ("{what}: 확장자는 엑셀이지만 내용이 텍스트여서 CSV로 읽었습니다.",
                      "{what}: the extension says Excel, but the content is text, so it was read as CSV."),
    "excel_needs_pkg": ("{what}: 이 엑셀 파일을 읽으려면 {pkg} 패키지가 필요합니다 (pip install {pkg}). "
                        "또는 CSV로 저장해 사용하세요.",
                        "{what}: reading this Excel file needs the {pkg} package (pip install {pkg}); or save "
                        "it as CSV."),
    "excel_unreadable": ("{what}: 엑셀 파일을 읽을 수 없습니다 (손상되었거나 형식이 다름: {error}).",
                         "{what}: cannot read the Excel file (damaged or another format: {error})."),
    "excel_unreadable_engine": ("{what}: 엑셀 파일을 읽을 수 없습니다 ({error}).",
                                "{what}: cannot read the Excel file ({error})."),
    "excel_wrong_ext": ("{what}: 엑셀 파일로 보입니다. 확장자를 .xlsx/.xls로 바꾸거나 CSV로 저장하세요.",
                        "{what}: this looks like an Excel file. Rename it to .xlsx/.xls or save it as CSV."),
    "file_not_found": ("파일을 찾을 수 없습니다: {path}", "File not found: {path}"),
    "path_is_dir": ("폴더가 아니라 파일 경로를 지정하세요: {path}", "Give a file path, not a folder: {path}"),
    "file_unreadable": ("{what}: 파일을 읽을 수 없습니다 ({error}).", "{what}: cannot read the file ({error})."),
    "save_path_is_dir": ("저장 경로가 폴더입니다. 파일 이름까지 지정하세요: {path}",
                         "The output path is a folder; give a file name too: {path}"),
    # ---- online connectors (trendai.connectors.ConnectorError)
    "conn_date_nat": ("{what} 날짜가 비어 있습니다 (NaT).", "The {what} date is empty (NaT)."),
    "conn_date_format": ("{what} 날짜 형식이 올바르지 않습니다: {value} ('YYYY-MM-DD' 또는 'YYYYMMDD')",
                         "The {what} date has the wrong format: {value} ('YYYY-MM-DD' or 'YYYYMMDD')"),
    "conn_date_invalid": ("{what} 날짜가 유효하지 않습니다: {value}", "The {what} date is not a valid date: {value}"),
    "conn_no_article": ("위키백과 문서 제목(article)이 비어 있습니다.", "The Wikipedia article title is empty."),
    "conn_bad_lang": ("언어 코드가 올바르지 않습니다: {lang} (예: 'ko', 'en')",
                      "Invalid language code: {lang} (e.g. 'ko', 'en')"),
    "conn_bad_granularity": ("granularity는 {choices} 중 하나여야 합니다: {value}",
                             "granularity must be one of {choices}: {value}"),
    "conn_start_after_end": ("시작일({start})이 종료일({end})보다 늦습니다.",
                             "The start date ({start}) is after the end date ({end})."),
    "conn_need_requests": ("위키백과 조회에는 requests 패키지가 필요합니다: pip install requests",
                           "Fetching Wikipedia page views needs the requests package: pip install requests"),
    "conn_network": ("Wikimedia API에 연결하지 못했습니다. 인터넷 연결(회사망이면 프록시 설정)을 확인한 뒤 다시 "
                     "시도하세요. (자세한 오류: {error})",
                     "Could not connect to the Wikimedia API (network error). Check the internet connection "
                     "(and the proxy settings on a company network), then try again. (Details: {error})"),
    "conn_wiki_404": ("위키백과 문서를 찾을 수 없습니다: '{title}' ({lang}.wikipedia). "
                      "정확한 문서 제목(대소문자·띄어쓰기)을 확인하거나, 조회 기간에 데이터가 있는지 확인하세요.",
                      "Wikipedia article not found: '{title}' ({lang}.wikipedia). Check the exact title "
                      "(capital letters, spaces) and whether the period has data."),
    "conn_wiki_429": ("Wikimedia API 요청 한도를 초과했습니다 (HTTP 429). 잠시 후 다시 시도하세요.",
                      "Wikimedia API rate limit exceeded (HTTP 429). Try again in a moment."),
    "conn_wiki_http": ("Wikimedia API 오류 (HTTP {status}): {url}", "Wikimedia API error (HTTP {status}): {url}"),
    "conn_wiki_error": ("Wikimedia API 오류: {error}", "Wikimedia API error: {error}"),
    "conn_wiki_json": ("Wikimedia API 응답을 JSON으로 해석할 수 없습니다.",
                       "The Wikimedia API response is not valid JSON."),
    "conn_wiki_empty": ("조회 기간에 '{title}' 페이지뷰 데이터가 없습니다.",
                        "No page views of '{title}' in the requested period (empty items)."),
    "conn_wiki_no_valid": ("Wikimedia API 응답에 유효한 페이지뷰 항목이 없습니다.",
                           "The Wikimedia API response has no valid page-view items."),
    "conn_no_keyword": ("검색어(keyword)가 비어 있습니다.", "The search keyword is empty."),
    "conn_need_pytrends": ("Google Trends 조회에는 pytrends 패키지가 필요합니다: pip install pytrends",
                           "Fetching Google Trends needs the pytrends package: pip install pytrends"),
    "conn_pytrends_broken": ("pytrends 패키지를 불러오지 못했습니다 ({kind}: {error}). "
                             "pip install -U pytrends 로 업데이트하세요.",
                             "Could not load the pytrends package ({kind}: {error}). Update it with "
                             "pip install -U pytrends."),
    "conn_trends_failed": ("Google Trends 요청에 실패했습니다: {error}. 요청이 너무 잦으면 차단(429 rate limit)될 수 "
                           "있으니 잠시 후 다시 시도하세요.",
                           "The Google Trends request failed: {error}. Frequent requests can be blocked "
                           "(429 rate limit); try again in a moment."),
    "conn_trends_empty": ("Google Trends에서 '{kw}'에 대한 데이터가 없습니다. 검색량이 너무 적거나 기간·지역 설정을 "
                          "확인하세요.",
                          "Google Trends has no data for '{kw}'. The search volume may be too low; check the "
                          "period and the region."),
    "conn_trends_no_col": ("Google Trends 응답에 '{kw}' 열이 없습니다.",
                           "The Google Trends response has no '{kw}' column."),
    "conn_trends_dates": ("Google Trends 응답의 날짜를 해석할 수 없습니다.",
                          "Cannot read the dates of the Google Trends response."),
    "conn_trends_no_valid": ("Google Trends에서 '{kw}'에 대한 유효한 값이 없습니다.",
                             "Google Trends returned no valid values for '{kw}'."),
}


def _param(value: Any, lang: str) -> Any:
    if isinstance(value, L):
        return value.get(lang)
    if isinstance(value, Msg):
        return value.text(lang)
    return value


def render(code: str, params: Optional[Mapping[str, Any]] = None, lang: str = "ko") -> str:
    """The text of message ``code`` with ``params`` in ``lang`` ('en'; anything else: Korean)."""
    ko, en = MESSAGES[code]
    lang = "en" if lang == "en" else "ko"
    values = {k: _param(v, lang) for k, v in dict(params or {}).items()}
    return (en if lang == "en" else ko).format(**values)


class Msg(str):
    """A catalog message: the Korean text as a ``str``, plus the ``(code, params)`` parts to
    render it in another language (:meth:`text`).  Several parts are joined with a space."""

    parts: tuple

    def __new__(cls, parts: Iterable[tuple[str, Mapping[str, Any]]]) -> "Msg":
        parts = tuple((str(c), dict(p)) for c, p in parts)
        obj = super().__new__(cls, " ".join(render(c, p, "ko") for c, p in parts))
        obj.parts = parts
        return obj

    def __reduce__(self) -> tuple:
        return (Msg, (self.parts,))

    @property
    def code(self) -> Optional[str]:
        """The code of the (first) part."""
        return self.parts[0][0] if self.parts else None

    @property
    def params(self) -> dict:
        """The params of the (first) part."""
        return dict(self.parts[0][1]) if self.parts else {}

    def text(self, lang: str = "ko") -> str:
        """The message in ``lang`` (``'en'``; anything else: the Korean text)."""
        if lang != "en":
            return str(self)
        return " ".join(render(c, p, "en") for c, p in self.parts)


def msg(code: str, **params: Any) -> Msg:
    """A :class:`Msg` for ``code`` (see :data:`MESSAGES`)."""
    if code not in MESSAGES:
        raise KeyError(code)
    return Msg([(code, params)])


def join(*items: Any) -> Optional[Msg]:
    """Several messages as one (``None`` / empty items skipped); ``None`` when nothing is left."""
    parts: list = []
    for it in items:
        if isinstance(it, Msg):
            parts.extend(it.parts)
        elif it:
            raise TypeError("join() takes Msg items")
    return Msg(parts) if parts else None


def localise(obj: Any, lang: str = "ko") -> str:
    """The text of an exception, a warning or a message in ``lang``: a :class:`Msg` (also as the
    first argument of an exception / warning) is rendered in ``lang``; anything else is ``str``."""
    m = obj
    if isinstance(obj, BaseException):
        m = obj.args[0] if len(obj.args) == 1 else None
        if not isinstance(m, Msg):
            return str(obj)
    if isinstance(m, Msg):
        return m.text(lang)
    return str(obj)


class Localised:
    """Mixin of exceptions / warnings raised with a catalog :class:`Msg`: ``code``, ``params`` and
    ``text(lang)``.  ``str(exc)`` stays the Korean text."""

    @property
    def message(self) -> Optional[Msg]:
        m = self.args[0] if len(getattr(self, "args", ())) == 1 else None  # type: ignore[attr-defined]
        return m if isinstance(m, Msg) else None

    @property
    def code(self) -> Optional[str]:
        """The catalog code of the message (``None`` for a free-text message)."""
        m = self.message
        return m.code if m is not None else None

    @property
    def params(self) -> dict:
        """The parameters of the message (``{}`` for a free-text message)."""
        m = self.message
        return m.params if m is not None else {}

    def text(self, lang: str = "ko") -> str:
        """The message in ``lang`` (``'en'``; anything else: Korean, i.e. ``str(self)``)."""
        return localise(self, lang)
