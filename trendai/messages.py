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

The web app (``trendai.web``) adds its own codes (errors, notices and the ``fix_<action>``
button labels) and never shows library text directly: :func:`render_web_msg` renders a whole
message -- every part, with :data:`WEB_OVERRIDES` instead of templates that name command-line
flags or Python, list parameters as ``‘a’, ‘b’`` and Korean particles after a parameter
resolved with :func:`josa` (``열 ‘번호’는``, not ``열 ['번호']은(는)``).  :func:`render` and
:meth:`Msg.text` are unchanged.

This module imports nothing from the package (``data`` imports it).
"""

from __future__ import annotations

import ast
import re
import string
from typing import Any, Iterable, Mapping, NamedTuple, Optional

__all__ = ["MESSAGES", "L", "Msg", "msg", "join", "render", "localise", "Localised",
           "unit", "format_label", "FREQ_EN",
           "WEB_OVERRIDES", "render_web", "render_web_msg", "josa", "fix_label"]


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
    ("날짜", "date"),             # native date cells (Excel); last: the labels above contain the word
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
    "table_too_many_columns": ("표의 열이 너무 많습니다({n}개, 최대 {max}개).",
                               "The table has too many columns ({n}, at most {max})."),
    "table_too_large": ("표가 너무 큽니다(칸 {n:,}개 이상, 최대 {max:,}개). 기간이나 열을 줄여 주세요.",
                        "The table is too large (at least {n:,} cells, at most {max:,}). Use a shorter period or "
                        "fewer columns."),
    "table_out_of_memory": ("{what}: 표가 너무 커서 읽을 수 없습니다. 기간이나 열을 줄여 주세요.",
                            "{what}: the table is too large to read. Use a shorter period or fewer columns."),
    "title_rows_skipped": ("{what}: 맨 위 {n}개 행(표 제목 등)을 건너뛰고 {row}행을 열 이름으로 읽었습니다.",
                           "{what}: skipped {n} row(s) at the top (a title) and read row {row} as the column "
                           "names."),
    "header_row_invalid": ("{row}행은 열 이름으로 쓸 수 없습니다(비어 있거나 표 밖입니다).",
                           "Row {row} cannot be used for the column names (it is empty or outside the table)."),
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
    # ---- web errors (trendai.web; C.12.2 of docs/WEB_SPEC.md)
    "bad_request": ("요청 형식이 올바르지 않아요. 페이지를 새로고침해 주세요.", "The request was malformed. Please reload the page."),
    "invalid_param": ("‘{param}’ 값이 올바르지 않아요.", "The value of ‘{param}’ is not valid."),
    "empty_query": ("분석할 키워드를 입력해 주세요.", "Enter a keyword to analyse."),
    "too_many_series": ("한 번에 {max}개까지 비교할 수 있어요. 하나를 빼고 추가해 주세요.",
                        "You can compare up to {max} at a time. Remove one first."),
    "too_many_keywords": ("한 묶음에는 같은 뜻 검색어를 {max}개까지 넣을 수 있어요(‘{group}’).",
                          "A group can hold up to {max} terms (‘{group}’)."),
    "term_too_long": ("검색어는 {max}자까지 입력할 수 있어요.", "Terms can be at most {max} characters long."),
    "wiki_group_unsupported": ("위키백과는 ‘+’ 묶기를 지원하지 않아요. 문서 제목을 하나씩 입력해 주세요.",
                               "Wikipedia does not support ‘+’ groups. Enter one article title per item."),
    "unknown_sample": ("‘{name}’ 샘플이 없어요.", "There is no sample called ‘{name}’."),
    "body_too_large": ("파일이 너무 커요. {max_mb} MB까지 올릴 수 있어요.", "The file is too large. The limit is {max_mb} MB."),
    "length_required": ("요청 길이를 알 수 없어요. 페이지를 새로고침해 주세요.", "The request has no length. Please reload the page."),
    "unsupported_media_type": ("지원하지 않는 요청 형식이에요.", "This request format is not supported."),
    "method_not_allowed": ("지원하지 않는 요청이에요.", "This request method is not supported."),
    "cross_origin": ("다른 사이트에서 보낸 요청은 받을 수 없어요.", "Requests from other sites are not accepted."),
    "not_found": ("페이지나 데이터를 찾을 수 없어요.", "Not found."),
    "result_not_found": ("결과를 찾을 수 없어요. 보관 기간(24시간)이 지났거나 주소가 잘못됐어요.",
                         "This result was not found. It may have expired (results are kept for 24 hours) or the link "
                         "is wrong."),
    "job_not_found": ("분석 작업을 찾을 수 없어요. 다시 분석해 주세요.", "The analysis job was not found. Please run it again."),
    "share_not_found": ("공유 링크가 만료됐거나 삭제됐어요.", "This shared link has expired or was deleted."),
    "share_forbidden": ("이 링크를 삭제할 권한이 없어요. 링크를 만든 브라우저에서만 삭제할 수 있어요.",
                        "You cannot delete this link. Only the browser that created it can."),
    "share_not_needed": ("공개 데이터 결과는 지금 주소가 곧 공유 링크예요. 주소를 복사해 주세요.",
                         "For public data the page address is the share link. Copy the address."),
    "upload_expired": ("올린 파일의 보관 시간(30분)이 지났어요. 다시 올려 주세요.",
                       "The uploaded file is no longer available (kept for 30 minutes). Please upload it again."),
    "upload_evicted": ("서버 메모리가 부족해 올린 파일을 보관하지 못했어요. 다시 올려 주세요.",
                       "The server ran short of memory and could not keep the uploaded file. Please upload it again."),
    "upload_unreadable": ("파일을 읽지 못했어요. CSV·TSV·엑셀(.xlsx)·JSON 파일을 올려 주세요.",
                          "The file could not be read. Upload a CSV, TSV, Excel (.xlsx) or JSON file."),
    "upload_too_many_columns": ("열이 너무 많아요({n}개). {max}개 이하의 열만 읽을 수 있어요.",
                                "Too many columns ({n}). Up to {max} columns can be read."),
    "upload_too_large_table": ("표가 너무 커요(칸 {n}개). 칸이 {max}개 이하가 되도록 기간이나 열을 줄여 올려 주세요.",
                               "The table is too large ({n} cells). Reduce the period or columns to at most {max} "
                               "cells."),
    "upload_too_many_series": ("값 열은 {max}개까지 고를 수 있어요.", "Choose up to {max} value columns."),
    "upload_archive_too_large": ("압축을 풀면 너무 큰 엑셀 파일이에요. CSV로 저장해 올려 주세요.",
                                 "This Excel file is too large once unpacked. Save it as CSV and upload that."),
    "upload_excel_needs_pkg": ("이 서버에서는 이 엑셀 형식을 읽을 수 없어요. CSV로 저장해 올려 주세요.",
                               "This server cannot read this Excel format. Save it as CSV and upload that."),
    "upload_no_date_col": ("날짜 열을 찾지 못했어요. 아래에서 날짜 열을 골라 주세요.",
                           "No date column was found. Pick the date column below."),
    "upload_no_value_col": ("숫자 값 열을 찾지 못했어요. 아래에서 값 열을 골라 주세요.",
                            "No numeric value column was found. Pick the value columns below."),
    "upload_bad_date_col": ("‘{col}’ 열을 날짜로 읽을 수 없어요. 다른 열을 골라 주세요.",
                            "Column ‘{col}’ cannot be read as dates. Pick another column."),
    "upload_bad_value_col": ("‘{col}’ 열을 숫자로 읽을 수 없어요.", "Column ‘{col}’ cannot be read as numbers."),
    "data_too_short": ("값이 {n}개뿐이라 분석할 수 없어요. 2개 이상 필요해요.",
                       "Only {n} value(s), so it cannot be analysed. At least 2 are needed."),
    "series_failed_web": ("이 시계열은 분석하지 못했어요.", "This series could not be analysed."),
    "rate_limited": ("요청이 너무 많아요. {retry_after}초 뒤에 다시 시도해 주세요.", "Too many requests. Try again in {retry_after} s."),
    "server_busy": ("지금 분석 요청이 몰려 있어요. 잠시 뒤 다시 시도해 주세요.",
                    "The server is busy with other analyses. Please try again shortly."),
    "internal_error": ("예기치 못한 오류가 났어요. 잠시 뒤 다시 시도해 주세요.", "Something went wrong. Please try again shortly."),
    "store_full": ("저장 공간이 부족해 지금은 저장할 수 없어요. 잠시 뒤 다시 시도해 주세요.",
                   "The server is out of storage space right now. Please try again later."),
    "job_timeout": ("분석이 너무 오래 걸려 멈췄어요. 기간이나 비교 개수를 줄여 보세요.",
                    "The analysis took too long and was stopped. Try a shorter period or fewer series."),
    "naver_unconfigured": ("네이버 검색어 트렌드는 아직 연결되지 않았어요.", "Naver search trends are not connected yet."),
    "naver_auth": ("네이버 연결에 문제가 생겼어요. 잠시 뒤 다시 시도하거나 다른 출처를 써 주세요.",
                   "There is a problem with the Naver connection. Try again later or use another source."),
    "naver_rate": ("요청이 몰려 잠시 막혔어요. {retry_after}초 뒤 자동으로 다시 시도해요.",
                   "Naver is busy right now. Retrying automatically in {retry_after} s."),
    "naver_rate_limited": ("네이버 새 조회는 한 시간에 {limit}번까지 할 수 있어요. 잠시 뒤 다시 시도하거나 위키백과로 보세요.",
                           "New Naver lookups are limited to {limit} per hour. Try again later or view Wikipedia "
                           "data."),
    "naver_budget_exhausted": ("이번 달 네이버 조회 한도를 모두 썼어요. 이미 조회된 검색어는 저장된 결과로 볼 수 있어요.",
                               "This month's Naver lookup quota is used up. Terms looked up before can still be "
                               "viewed from the cache."),
    "naver_server": ("네이버 서버에서 오류가 났어요.", "The Naver server returned an error."),
    "source_timeout": ("데이터를 받아오지 못했어요(20초 초과).", "The data could not be fetched (timed out after 20 s)."),
    "source_unreachable": ("데이터 출처에 연결하지 못했어요. 잠시 뒤 다시 시도해 주세요.",
                           "Could not reach the data source. Please try again shortly."),
    "source_bad_response": ("데이터 출처가 알 수 없는 응답을 보냈어요.", "The data source sent an unexpected response."),
    "wiki_not_found": ("위키백과에 ‘{title}’ 문서가 없어요. 정식 문서 제목인지 확인하거나 네이버로 바꿔 보세요.",
                       "Wikipedia has no article titled ‘{title}’. Check that it is the article's exact title, or "
                       "switch to Naver."),
    "wiki_not_found_no_naver": ("위키백과에 ‘{title}’ 문서가 없어요. 정식 문서 제목인지 확인해 주세요.",
                                "Wikipedia has no article titled ‘{title}’. Check that it is the article's exact "
                                "title."),
    "wiki_rate": ("위키백과 조회가 잠시 막혔어요. 잠시 뒤 다시 시도해 주세요.",
                  "Wikipedia lookups are blocked for a moment. Please try again shortly."),
    "wiki_empty": ("이 기간에는 ‘{title}’ 조회수 데이터가 없어요.", "There are no page views of ‘{title}’ in this period."),
    "export_xlsx_unavailable": ("이 서버에서는 엑셀 내보내기를 쓸 수 없어요. CSV로 받아 주세요.",
                                "Excel export is not available on this server. Download CSV instead."),
    "export_naver_raw": ("네이버 원본 데이터는 이용약관상 내려받기를 제공하지 않아요. 원본은 데이터랩(datalab.naver.com)에서 직접 엑셀로 받을 수 있어요.",
                         "Raw Naver data cannot be downloaded under its terms of use. You can download it yourself "
                         "from DataLab (datalab.naver.com)."),
    "ai_unavailable": ("지금은 AI 해석을 쓸 수 없어요.", "AI interpretation is not available right now."),
    "ai_naver_blocked": ("네이버 데이터는 이용약관에 따라 AI로 보내지 않아요.",
                         "Naver data is not sent to AI, following its terms of use."),
    "ai_readonly": ("공유 링크에서는 AI 해석을 새로 만들 수 없어요.", "AI interpretations cannot be created on a shared link."),
    "ai_failed": ("AI 해석을 만들지 못해 규칙 기반 해석을 보여 드려요.",
                  "The AI interpretation could not be created, so the rule-based one is shown."),
    "ai_rate_limited": ("AI 해석은 한 시간에 {limit}번까지 만들 수 있어요.", "AI interpretations are limited to {limit} per hour."),
    # ---- connectors: Naver search trends, periods, timeouts (C.12.3)
    "conn_naver_unconfigured": ("네이버 API 키가 설정되지 않았습니다. NAVER_API_HUB_KEY_ID와 NAVER_API_HUB_KEY(또는 NAVER_CLIENT_ID와 "
                                "NAVER_CLIENT_SECRET)를 설정하세요.",
                                "Naver API keys are not set. Set NAVER_API_HUB_KEY_ID and NAVER_API_HUB_KEY (or "
                                "NAVER_CLIENT_ID and NAVER_CLIENT_SECRET)."),
    "conn_naver_auth": ("네이버 API 인증에 실패했습니다 (HTTP {status}). 키가 맞는지, 앱에서 검색어트렌드 API를 켰는지 확인하세요.",
                        "Naver API authentication failed (HTTP {status}). Check the keys and that the Search Trend "
                        "API is enabled for the app."),
    "conn_naver_rate": ("네이버 API 요청 한도를 초과했습니다 (HTTP 429). 잠시 후 다시 시도하세요.",
                        "Naver API rate limit exceeded (HTTP 429). Try again in a moment."),
    "conn_naver_server": ("네이버 API 서버 오류입니다 (HTTP {status}).", "Naver API server error (HTTP {status})."),
    "conn_naver_http": ("네이버 API 오류입니다 (HTTP {status}).", "Naver API error (HTTP {status})."),
    "conn_naver_bad_response": ("네이버 API 응답을 해석할 수 없습니다.", "The Naver API response could not be read."),
    "conn_naver_network": ("네이버 API에 연결하지 못했습니다. (자세한 오류: {error})",
                           "Could not connect to the Naver API. (Details: {error})"),
    "conn_naver_timeout": ("네이버 API 응답이 20초 안에 오지 않았습니다.", "The Naver API did not answer within 20 s."),
    "conn_naver_too_many_groups": ("네이버 검색어 그룹은 {max}개까지 요청할 수 있습니다.",
                                   "At most {max} Naver keyword groups can be requested."),
    "conn_naver_too_many_keywords": ("그룹 '{group}'의 검색어는 {max}개까지입니다.",
                                     "Group '{group}' can hold at most {max} keywords."),
    "conn_naver_bad_filter": ("네이버 조회 조건 {what} 값이 올바르지 않습니다: {value}", "Invalid Naver filter {what}: {value}"),
    "conn_bad_range": ("조회 기간이 올바르지 않습니다: {value}", "Invalid period: {value}"),
    "conn_timeout": ("Wikimedia API 응답이 20초 안에 오지 않았습니다.", "The Wikimedia API did not answer within 20 s."),
    # ---- web notices (C.12.5)
    "naver_missing_filled": ("네이버가 값을 주지 않은 기간 {n}개는 0으로 처리했어요. 검색량이 매우 적은 기간이에요.",
                             "{n} period(s) without a value from Naver were treated as 0 (very low search volume)."),
    "naver_low_resolution": ("다른 검색어보다 검색량이 매우 적어 해상도가 낮아요.",
                             "Far fewer searches than the other terms, so the resolution is low."),
    "mostly_zero_volume": ("이 기간에는 검색량이 너무 적어 대부분 0으로 나와요. 기간 단위를 ‘월’로 바꾸거나 같은 뜻 검색어를 묶어 보세요.",
                           "Search volume is so low that most values are 0. Switch the unit to months or group "
                           "similar terms."),
    "data_short": ("값이 {n}개뿐이라 국면은 잠정 판단이고 예측 범위가 넓어요. 기간을 늘리면 더 정확해져요.",
                   "Only {n} values, so the phase is tentative and the forecast range is wide. A longer period helps."),
    "horizon_clamped": ("예측 기간을 최대치인 {span}(으)로 줄였어요.", "The forecast horizon was reduced to the maximum, {span}."),
    "gt_detected": ("Google 트렌드 CSV로 읽었어요(검색어 {terms}, 지역 {region}). 값은 요청마다 0–100으로 다시 맞춰진 상대 지수예요.",
                    "Read as a Google Trends CSV (terms {terms}, region {region}). Values are a relative index "
                    "rescaled to 0–100 per request."),
    "gt_lt1_half": ("‘<1’ 값 {n}개를 0.5로 읽었어요. Google 트렌드에서 ‘1 미만’이라는 뜻이에요.",
                    "{n} ‘<1’ value(s) were read as 0.5 (Google Trends' “less than 1”)."),
    "index_unavailable": ("기준값이 0 이하인 시계열이 있어 ‘기준 시점=100’으로 바꿀 수 없어요. 원래 값으로 보여 드려요.",
                          "Some series have a base of 0 or less, so they cannot be indexed to 100. Showing original "
                          "values."),
    "log_hint": ("최댓값 차이가 {ratio}배를 넘어요. 로그 축으로 보면 작은 시계열도 잘 보여요.",
                 "The largest series is more than {ratio}× the smallest. A log axis shows the small ones better."),
    "naver_cached": ("저장된 네이버 조회 결과({fetched} 기준)를 보여 드려요.", "Showing stored Naver data (as of {fetched})."),
    "gaps_filled": ("빈 기간 {n}개({dates})는 앞뒤 값으로 채운 추정값이에요. 실제 관측값이 아니에요.",
                    "{n} empty period(s) ({dates}) were filled in from the values on either side: estimates, not "
                    "observations."),
    # ---- fix actions: button labels of an error's fixes (C.12.6; messages.fix_label)
    "fix_reload": ("새로고침", "Reload"),
    "fix_retry": ("다시 시도", "Try again"),
    "fix_retry_in": ("지금 다시", "Retry now"),
    "fix_go_home": ("처음으로", "Home"),
    "fix_reanalyze": ("다시 분석", "Analyse again"),
    "fix_use_wiki": ("위키백과로 보기", "View on Wikipedia"),
    "fix_use_naver": ("네이버로 보기", "View on Naver"),
    "fix_upload_file": ("파일 올리기", "Upload a file"),
    "fix_reupload": ("다시 올리기", "Upload again"),
    "fix_try_sample": ("샘플로 체험", "Try a sample"),
    "fix_extend_range": ("기간 늘리기", "Use a longer period"),
    "fix_unit_month": ("월 단위로", "Switch to months"),
    "fix_group_terms": ("같은 뜻 묶기", "Group similar terms"),
    "fix_remove_series": ("하나 빼기", "Remove one"),
    "fix_show_formats": ("지원 형식 보기", "Supported formats"),
    "fix_try_paste": ("붙여넣기로 시도", "Try pasting"),
    "fix_pick_columns": ("열 고르기", "Pick columns"),
    "fix_wiki_title_help": ("위키백과에서 제목 찾기", "Search Wikipedia titles"),
    "fix_use_log": ("로그 축으로", "Use log scale"),
    "fix_copy_link": ("주소 복사", "Copy link"),
    "fix_export_csv": ("CSV로 받기", "Download CSV"),
    "fix_reanalyze_single": ("이 검색어만 다시 분석", "Analyse this term alone"),
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


# --------------------------------------------------------------------------- web rendering


def _library_en(code: str, ko: str) -> tuple[str, str]:
    """A web override that only rewrites the Korean text (해요체): the English library text is kept."""
    return ko, MESSAGES[code][1]


#: Web wording of library templates (trendai.web shows these instead of MESSAGES): no command-line
#: flags, no Python hints, Korean in 해요체 like the rest of the web UI.  Each override uses a subset of
#: its library template's parameters, so whatever renders the library text also renders this one.
WEB_OVERRIDES: dict[str, tuple[str, str]] = {
    # ---- templates that name CLI flags or Python (C.12.7 of docs/WEB_SPEC.md)
    "weekday_only": ("{name}평일(영업일)에만 값이 있어, 주말 {filled}개를 선형 보간해 달력일 기준({n_out}개)으로 "
                     "분석했어요. 주 단위로 보려면 단위를 ‘주’로 바꾸세요.",
                     "{name}values exist on weekdays only, so {filled} weekend days were interpolated and the data "
                     "is analysed on calendar days ({n_out}). Switch the unit to weeks for a weekly view."),
    "duplicates_merged": ("{name}같은 {unit} 단위 기간에 값이 여러 개 있어 {how} 합쳤어요 (관측 {n_raw}개 → "
                          "{observed}개 기간).",
                          "{name}several values fell in the same {unit} period and were {how} ({n_raw} observations "
                          "→ {observed} periods)."),
    "rownum_cols_skipped": ("행 번호/일련번호로 보이는 열 {cols}은(는) 분석에서 제외했어요. 포함하려면 확인 단계에서 "
                            "값 열로 고르세요.",
                            "The column(s) {cols} look like row numbers and were left out. Select them as value "
                            "columns in the confirm step to include them."),
    "too_few_after_regularise": ("{name}{unit} 단위로 정리하면 관측치가 {n}개뿐이에요(2개 이상 필요). 더 짧은 주기를 "
                                 "고르거나 데이터를 늘리세요.",
                                 "{name}on a {unit} grid only {n} observation(s) remain (at least 2 are needed). "
                                 "Choose a shorter frequency or add data."),
    "no_date_col": ("날짜 열을 찾지 못했어요. 확인 단계에서 날짜 열을 골라 주세요.",
                    "No date column found. Pick it in the confirm step."),
    "no_numeric_col": ("숫자 값 열을 찾지 못했어요. 확인 단계에서 값 열을 골라 주세요.",
                       "No numeric value column found. Pick one in the confirm step."),
    "only_rownum_cols": ("분석할 숫자 값 열이 없어요(행 번호 열 {cols}만 있음). 확인 단계에서 값 열을 골라 주세요.",
                         "There is no numeric value column (only the row-number column(s) {cols}). Pick one in the "
                         "confirm step."),
    "index_numeric": ("날짜 자리에 숫자({lo}~{hi})가 들어 있어요. 날짜 열이 맞는지 확인해 주세요. 연도라면 "
                      "1000~2200 사이의 정수여야 해요.",
                      "Numbers ({lo} to {hi}) were found where dates are expected. Check the date column. Years "
                      "must be whole numbers between 1000 and 2200."),
    "series_required": ("날짜와 숫자 값으로 된 시계열이 필요해요.", "A time series of dates and numeric values is needed."),
    "all_dates_equal": ("모든 날짜가 같아요. 날짜 열이 맞는지 확인해 주세요.",
                        "All dates are the same. Check that the date column is right."),
    "horizon_min": ("예측 기간은 1 이상이어야 해요.", "The forecast horizon must be at least 1."),
    "horizon_not_int": ("예측 기간은 정수여야 해요(입력값: {value}).",
                        "The forecast horizon must be a whole number (got {value})."),
    "horizon_not_positive": ("예측 기간은 1 이상의 정수여야 해요(입력값: {value}).",
                             "The forecast horizon must be a whole number of at least 1 (got {value})."),
    "horizon_too_long": ("예측 기간이 너무 길어요: {horizon:,}기간(최대 {max:,}기간).",
                         "The forecast horizon is too long: {horizon:,} periods (at most {max:,})."),
    "horizon_out_of_range": ("예측 기간 {horizon:,}기간은 날짜로 나타낼 수 있는 범위를 넘어요. 예측 기간을 줄여 주세요.",
                             "A forecast horizon of {horizon:,} periods goes beyond the dates that can be "
                             "represented. Use a shorter horizon."),
    "horizon_long": _library_en("horizon_long", "예측 기간({horizon:,}기간)이 데이터 길이({n:,}개)의 {ratio}배를 "
                                "넘어요. 이렇게 먼 미래는 과거 패턴만으로 추정하기 어려워 불확실성이 매우 커요."),
    "excel_needs_pkg": ("{what}: 이 서버에서는 이 엑셀 형식을 읽을 수 없어요. CSV로 저장해 올려 주세요.",
                        "{what}: this Excel format cannot be read on this server. Save it as CSV and upload that."),
    # ---- the other data-layer texts the web shows (notices, upload errors), in 해요체
    "freq_needs_two_dates": _library_en("freq_needs_two_dates", "주기를 알아내려면 날짜가 2개 이상 필요해요."),
    "series_failed": ("‘{series}’을(를) 분석하지 못했어요: {detail}", "‘{series}’ could not be analysed: {detail}"),
    "date_order_ambiguous": ("날짜의 일/월 순서가 모호해(모든 값이 12 이하) ‘{label}’ 순서로 읽었어요. 다르다면 날짜를 "
                             "YYYY-MM-DD 형식으로 바꿔 주세요.",
                             "The day/month order of the dates is ambiguous (every value is 12 or less), so they "
                             "were read as ‘{label}’. If that is wrong, write the dates as YYYY-MM-DD."),
    "date_ym_ambiguous": ("‘{value}’ 같은 날짜는 연/월 순서가 모호해 ‘{label}’ 순서로 읽었어요.",
                          "The year/month order of dates such as ‘{value}’ is ambiguous, so they were read as "
                          "‘{label}’."),
    "dates_unreadable_rows": _library_en("dates_unreadable_rows",
                                         "날짜로 읽을 수 없는 값 {n}개(예: {examples})가 있는 행은 제외했어요."),
    "index_unreadable": ("날짜를 읽을 수 없어요(예: {examples}). 날짜 형식의 값이 필요해요.",
                         "The dates cannot be read (e.g. {examples}); dates are needed."),
    "series_empty": _library_en("series_empty", "시계열이 비어 있어요."),
    "values_unreadable": _library_en("values_unreadable",
                                     "{name}숫자로 읽을 수 없는 값 {n}개(예: {examples})를 빈 값으로 처리했어요."),
    "values_infinite": _library_en("values_infinite", "{name}무한대(inf) 값 {n}개를 제외했어요."),
    "no_valid_values": _library_en("no_valid_values", "{name}유효한 숫자 값이 하나도 없어요."),
    "gaps_interpolated": _library_en("gaps_interpolated", "{name}{unit} 단위로 맞추면서 빈 기간 {filled}개(전체 "
                                     "{total}개 중 {share:.0%})를 선형 보간했어요. 실제 관측은 {observed}개예요."),
    "upsampled": _library_en("upsampled", "{name}{own} 단위 데이터를 더 촘촘한 {unit} 단위로 바꾸면서 {filled:,}개 "
                             "기간(전체 {total:,}개 중 {share:.0%})을 선형 보간했어요. 실제 관측은 {observed:,}개뿐이라 "
                             "정확도와 확률이 실제보다 좋아 보일 수 있어요."),
    "too_few_obs": _library_en("too_few_obs", "{name}관측치가 2개 이상 필요해요(유효한 값 {n}개)."),
    "wide_table": _library_en("wide_table", "날짜가 열 머리글에 있는 가로형 표로 보고, 행마다 시계열 하나씩 {n}개로 "
                              "읽었어요(날짜 형식: {label})."),
    "table_empty_rows": _library_en("table_empty_rows", "표가 비어 있어요(읽을 행이 없어요)."),
    "table_empty_values": _library_en("table_empty_values", "표가 비어 있어요(값이 있는 행이 없어요)."),
    "table_too_many_columns": _library_en("table_too_many_columns", "표의 열이 너무 많아요({n}개, 최대 {max}개)."),
    "table_too_large": _library_en("table_too_large", "표가 너무 커요(칸 {n:,}개 이상, 최대 {max:,}개). 기간이나 열을 "
                                   "줄여 주세요."),
    "table_out_of_memory": _library_en("table_out_of_memory", "{what}: 표가 너무 커서 읽을 수 없어요. 기간이나 열을 "
                                       "줄여 주세요."),
    "title_rows_skipped": _library_en("title_rows_skipped", "{what}: 맨 위 {n}개 행(표 제목 등)을 건너뛰고 {row}행을 "
                                      "열 이름으로 읽었어요."),
    "header_row_invalid": _library_en("header_row_invalid", "{row}행은 열 이름으로 쓸 수 없어요(비어 있거나 표 "
                                      "밖이에요)."),
    "date_col_missing": ("날짜 열 ‘{col}’을(를) 찾지 못했어요. 열 목록: {columns}",
                         "Date column ‘{col}’ not found. Columns: {columns}"),
    "date_col_unreadable": ("‘{col}’ 열을 날짜로 읽을 수 없어요(예: {examples}). 지원 형식: 2024-01-05, 2024-01, "
                            "2024.01, 202401, 20240105, 2024년 1월, 2024 Q1, 2024년 1분기, 2024-W01, 05.01.2024, "
                            "연도(2024), Unix 시간.",
                            "Column ‘{col}’ cannot be read as dates (e.g. {examples}). Supported formats: "
                            "2024-01-05, 2024-01, 2024.01, 202401, 20240105, 2024 Q1, 2024-W01, 05.01.2024, "
                            "years (2024), Unix time and Korean year-month text."),
    "value_col_missing": ("값 열 ‘{col}’을(를) 찾지 못했어요. 열 목록: {columns}",
                          "Value column ‘{col}’ not found. Columns: {columns}"),
    "value_col_unreadable": ("값 열 ‘{col}’을(를) 숫자로 읽을 수 없어요(예: {examples}).",
                             "Value column ‘{col}’ cannot be read as numbers (e.g. {examples})."),
    "col_values_unreadable": ("‘{col}’ 열: 숫자로 읽을 수 없는 값 {n}개(예: {examples})를 빈 값으로 처리했어요.{edge}",
                              "Column ‘{col}’: {n} value(s) that are not numbers (e.g. {examples}) were treated "
                              "as missing.{edge}"),
    "utf8_broken": ("{what}: UTF-8 파일에 깨진 바이트 {n}개가 있어 ‘�’로 바꿔 읽었어요.",
                    "{what}: {n} broken byte(s) in the UTF-8 file were read as ‘�’."),
    "file_empty": _library_en("file_empty", "{what}: 파일이 비어 있어요."),
    "headerless": _library_en("headerless", "{what}: 열 이름(머리글) 행이 없는 파일로 보고 첫 행부터 데이터로 읽었어요."),
    "csv_ragged": _library_en("csv_ragged", "{what}: CSV 형식 오류예요. 행마다 열 개수가 달라요({error}).{hint}"),
    "json_empty_array": _library_en("json_empty_array", "{what}: JSON 배열이 비어 있어요."),
    "json_array_shape": _library_en("json_array_shape", "{what}: JSON 배열 형식을 이해할 수 없어요(객체 목록이나 "
                                    "[날짜, 값] 목록이 필요해요)."),
    "json_empty_object": _library_en("json_empty_object", "{what}: JSON 객체가 비어 있어요."),
    "json_shape": _library_en("json_shape", "{what}: JSON 구조를 이해할 수 없어요(레코드 목록, {{날짜: 값}} 또는 열 "
                              "객체가 필요해요)."),
    "json_invalid": _library_en("json_invalid", "{what}: 올바른 JSON이 아니에요({error}, {line}행)."),
    "json_to_table": _library_en("json_to_table", "{what}: JSON 구조를 표로 바꿀 수 없어요({error})."),
    "excel_is_html": _library_en("excel_is_html", "{what}: 확장자는 엑셀이지만 실제로는 HTML/XML 표예요. 엑셀에서 열어 "
                                 "CSV(UTF-8)나 .xlsx로 다시 저장해 주세요."),
    "excel_is_text": _library_en("excel_is_text", "{what}: 확장자는 엑셀이지만 내용이 텍스트여서 CSV로 읽었어요."),
    "excel_unreadable": _library_en("excel_unreadable", "{what}: 엑셀 파일을 읽을 수 없어요(손상됐거나 다른 형식이에요: "
                                    "{error})."),
    "excel_unreadable_engine": _library_en("excel_unreadable_engine", "{what}: 엑셀 파일을 읽을 수 없어요({error})."),
    "excel_wrong_ext": _library_en("excel_wrong_ext", "{what}: 엑셀 파일로 보여요. 확장자를 .xlsx/.xls로 바꾸거나 "
                                   "CSV로 저장해 주세요."),
    "file_unreadable": _library_en("file_unreadable", "{what}: 파일을 읽을 수 없어요({error})."),
}

#: Library sentences that data.py passes as L parameters -> their web wording (ko, en)
_WEB_PARAM_TEXT: dict[str, tuple[str, str]] = {
    " 시계열의 처음/끝 부분이 잘렸을 수 있습니다.": (" 시계열의 처음이나 끝이 잘렸을 수 있어요.",
                                       " The start or end of the series may be cut off."),
    (" 천 단위 쉼표(1,234)가 따옴표 없이 들어 있으면 열 개수가 달라집니다. "
     "값을 따옴표로 감싸거나 탭/세미콜론 구분자로 저장하세요."): (
        " 천 단위 쉼표(1,234)가 따옴표 없이 들어 있으면 열 개수가 달라져요. 값을 따옴표로 감싸거나 탭·세미콜론으로 "
        "구분해 저장해 주세요.",
        " Thousands separators (1,234) without quotes change the number of columns: quote the values or save the "
        "file with tabs or semicolons as separators."),
}

#: particle pair -> (form after a final consonant, form otherwise)
_JOSA_FORMS: dict[str, tuple[str, str]] = {
    "은/는": ("은", "는"), "이/가": ("이", "가"), "을/를": ("을", "를"), "과/와": ("과", "와"), "으로/로": ("으로", "로"),
}
#: particle markers written in templates right after a parameter -> particle pair
_TEMPLATE_PARTICLES: dict[str, str] = {
    "은(는)": "은/는", "이(가)": "이/가", "을(를)": "을/를", "과(와)": "과/와", "(으)로": "으로/로",
}
#: characters skipped at the end of a word before its last sound is looked at
_JOSA_TRAILING = " \t\n'\"’”)]」』"
#: trailing digit -> (has a final consonant, that consonant is ㄹ) when read in Korean
#: (0 영/십/백/천/만, 1 일, 2 이, 3 삼, 4 사, 5 오, 6 육, 7 칠, 8 팔, 9 구)
_DIGIT_SOUND: dict[str, tuple[bool, bool]] = {
    "0": (True, False), "1": (True, True), "2": (False, False), "3": (True, False), "4": (False, False),
    "5": (False, False), "6": (True, False), "7": (True, True), "8": (True, True), "9": (False, False),
}
_PARTICLE_AFTER_FIELD = re.compile(r"^([’”)\]」』'\"]*)(은\(는\)|이\(가\)|을\(를\)|과\(와\)|\(으\)로)")
_SERIES_LABEL = re.compile(r"^'(.*)': $", re.DOTALL)
_FORMATTER = string.Formatter()


def josa(word: str, pair: str) -> str:
    """``word`` followed by the right form of the Korean particle ``pair`` (``'은/는'``, ``'이/가'``,
    ``'을/를'``, ``'과/와'``, ``'으로/로'``): the first form after a final consonant, the second otherwise;
    ``'으로/로'`` also takes ``'로'`` after ㄹ.  Trailing quotes, brackets and spaces are skipped; a trailing
    digit is read in Korean (``104주로``, ``CP949로``, ``UTF-16으로``); Latin letters take the second form.
    ``ValueError`` for an unknown pair."""
    try:
        with_final, without_final = _JOSA_FORMS[pair]
    except KeyError:
        raise ValueError(f"unknown particle pair: {pair!r}") from None
    text = str(word)
    last = text.rstrip(_JOSA_TRAILING)[-1:]
    if "가" <= last <= "힣":
        final = (ord(last) - 0xAC00) % 28
        has_final, rieul = final != 0, final == 8
    else:
        has_final, rieul = _DIGIT_SOUND.get(last, (False, False))
    if pair == "으로/로":
        return text + ("으로" if has_final and not rieul else "로")
    return text + (with_final if has_final else without_final)


def _quoted(items: Iterable[Any]) -> str:
    """``‘a’, ‘b’`` (curly single quotes, ``, `` separator)."""
    return ", ".join(f"‘{x}’" for x in items)


def _requote_examples(text: str) -> str:
    """A pre-joined ``"'x', 'y'"`` repr list (the ``examples`` parameter) as ``‘x’, ‘y’``; unchanged if it is
    not such a list."""
    if len(text) > 10_000:
        return text
    try:
        items = ast.literal_eval("[" + text + "]")
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
        return text
    if not isinstance(items, list) or not items:
        return text
    return _quoted(items)


def _web_param(name: str, value: Any, lang: str) -> Any:
    """One template parameter as the web shows it (see :func:`render_web_msg`)."""
    if isinstance(value, Msg):
        return render_web_msg(value, lang)
    if isinstance(value, L):
        web = _WEB_PARAM_TEXT.get(value.ko)
        value = L(*web).get(lang) if web else value.get(lang)
    if name in ("label", "date_format"):
        return format_label(value, lang)
    if isinstance(value, (list, tuple)):
        return _quoted(render_web_msg(x, lang) if isinstance(x, Msg) else
                       x.get(lang) if isinstance(x, L) else x for x in value)
    if isinstance(value, str):
        if name == "examples":
            return _requote_examples(value)
        if name == "name":   # the series label of data._label: "'sales': " -> "‘sales’: "
            return _SERIES_LABEL.sub(r"‘\1’: ", value)
    return value


def _format_web(template: str, values: Mapping[str, Any]) -> str:
    """``template.format(**values)``, with every particle marker written right after a parameter
    (``{cols}은(는)``, ``‘{col}’을(를)``, ``{span}(으)로``) resolved for the substituted text."""
    out: list[str] = []
    previous: Optional[str] = None   # text of the field that the current literal directly follows
    for literal, field, spec, conversion in _FORMATTER.parse(template):
        if previous is not None and literal:
            m = _PARTICLE_AFTER_FIELD.match(literal)
            if m:
                closers, marker = m.group(1), m.group(2)
                particle = josa(previous + closers, _TEMPLATE_PARTICLES[marker])[len(previous) + len(closers):]
                literal = closers + particle + literal[m.end():]
        out.append(literal)
        previous = None
        if field is None:
            continue
        obj, _ = _FORMATTER.get_field(field, (), values)
        obj = _FORMATTER.convert_field(obj, conversion)
        if spec and "{" in spec:
            spec = _FORMATTER.vformat(spec, (), values)
        previous = _FORMATTER.format_field(obj, spec or "")
        out.append(previous)
    return "".join(out)


def render_web(code: str, params: Optional[Mapping[str, Any]] = None, lang: str = "ko") -> str:
    """The web text of one message ``code``: like :func:`render`, but with :data:`WEB_OVERRIDES` where
    the library text names command-line flags or Python, and the parameter rules of
    :func:`render_web_msg`.  ``KeyError`` for an unknown code (or a missing parameter)."""
    lang = "en" if lang == "en" else "ko"
    ko, en = WEB_OVERRIDES.get(code) or MESSAGES[code]
    values = {k: _web_param(k, v, lang) for k, v in dict(params or {}).items()}
    return _format_web(en if lang == "en" else ko, values)


def render_web_msg(m: Any, lang: str = "ko") -> str:
    """The web text of a whole message: a :class:`Msg`, or an exception / warning raised with one.

    * every part of a joined message is rendered (with :data:`WEB_OVERRIDES`), joined with a space;
    * a parameter that is a :class:`Msg` is rendered the same way; an :class:`L` in ``lang``;
    * ``label`` / ``date_format`` go through :func:`format_label` (no Korean format label in English);
    * list parameters read ``‘a’, ‘b’`` instead of a Python list, and so does ``examples`` (a joined
      ``"'a', 'b'"`` repr string); the series label ``name`` (``"'sales': "``) gets the same quotes;
    * the particles ``은(는)``, ``이(가)``, ``을(를)``, ``과(와)``, ``(으)로`` written right after a parameter
      are resolved with :func:`josa`.

    Anything that carries no catalog message (a plain exception, a free-text warning) gives the generic
    ``series_failed_web`` text: an exception's own text is never shown."""
    if isinstance(m, BaseException):
        inner = m.args[0] if len(m.args) == 1 else None
        m = inner if isinstance(inner, Msg) else None
    if not isinstance(m, Msg) or not m.parts:
        return render_web("series_failed_web", {}, lang)
    return " ".join(render_web(code, params, lang) for code, params in m.parts)


def fix_label(action: str, lang: str = "ko") -> str:
    """The button label of a fix action (``render('fix_' + action, {}, lang)``); ``KeyError`` if unknown."""
    return render("fix_" + action, {}, lang)
