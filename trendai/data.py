"""Loading and normalising time series.

Everything downstream expects a *regular* series: a float ``pd.Series`` with a
sorted ``DatetimeIndex`` on one of the canonical frequencies below, no
duplicates and no NaNs.  :func:`prepare` turns almost any user input into that.

File loading (:func:`load_table`) is built for the files people actually have:

* encodings: UTF-8 (with/without BOM), UTF-16 (Excel "Unicode text"), CP949 /
  EUC-KR (Korean Excel and public-data portals);
* separators: ``,`` ``;`` tab ``|`` (sniffed), quoted fields;
* dates: ISO (``2024-01-05``, ``2024-01``), ``2024.01`` / ``2024. 1. 5.``
  (KOSIS / Korean Excel), ``202401`` / ``20240105`` (ECOS), ``2024년 1월``,
  ``2024년 1분기``, ``2024 Q1`` / ``2024Q1`` / ``Q1 2024``, ISO weeks
  (``2024-W01``), ``01.02.2024`` / ``02/01/2024`` (day/month order inferred
  once for the whole column), plain years in any order, Unix seconds / ms,
  timestamps with UTC offsets (the local wall time is kept);
* numbers: ``1,234``, ``1.234,5`` (decimal comma), ``12%``, ``$1,234``,
  ``1,234원``, ``(1,234)`` (negative), ``1.2억`` / ``3,400만`` / ``12K``;
* wide tables whose dates run across the columns (KOSIS default layout);
* row-number columns (``Unnamed: 0`` written by ``DataFrame.to_csv``, ``id``,
  ``번호`` ...) are ignored.

Problems are never silent: unreadable values, dropped rows, interpolated gaps
and ambiguous date orders are reported with a Korean :class:`DataWarning` /
:class:`DataNotice`; input that cannot be used raises :class:`DataError`
(Korean message that names the CLI flag / argument to use).  Every such text comes
from the message catalog (:mod:`trendai.messages`): the exception / warning carries
its ``code`` and ``params`` and renders itself in English with ``.text('en')`` (the
command line does under ``--lang en``); ``str()`` stays Korean.
"""

from __future__ import annotations

import csv
import io
import json
import re
import warnings
import zipfile
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional, Union

import numpy as np
import pandas as pd

from . import numfmt as _numfmt
from .messages import L, Localised, Msg, format_label, join, msg, unit

# canonical frequency code -> pandas offset alias used for resampling / future dates
PANDAS_FREQ: dict[str, str] = {
    "D": "D",
    "W": "W-SUN",  # default anchor; weekly data keeps its own weekday (see regularize)
    "M": "MS",
    "Q": "QS",
    "Y": "YS",
}

FREQ_KO: dict[str, str] = {"D": "일", "W": "주", "M": "월", "Q": "분기", "Y": "년"}

_SEASON_LENGTH = {"D": 7, "W": 52, "M": 12, "Q": 4, "Y": 1}
_PERIODS_PER_YEAR = {"D": 365.25, "W": 52.1775, "M": 12.0, "Q": 4.0, "Y": 1.0}
_DEFAULT_HORIZON = {"D": 30, "W": 26, "M": 12, "Q": 8, "Y": 5}
_STEP_DAYS = {"D": 1.0, "W": 7.0, "M": 30.436875, "Q": 91.310625, "Y": 365.2425}
_WEEKDAYS = ("MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN")


class DataError(Localised, ValueError):
    """Raised when input data cannot be turned into a usable series (Korean message).

    Raised with a catalog message (:func:`trendai.messages.msg`): ``.code`` / ``.params``
    name it and ``.text('en')`` renders it in English; ``str(exc)`` is the Korean text."""


class DataWarning(Localised, UserWarning):
    """The data was changed in a way the user should know about (values dropped, gaps filled...).

    Like :class:`DataError`: ``code``, ``params`` and ``text(lang)``; ``str()`` is Korean."""


class DataNotice(Localised, UserWarning):
    """Informational note about how the input was interpreted (ignored columns, date order...).

    Like :class:`DataError`: ``code``, ``params`` and ``text(lang)``; ``str()`` is Korean."""


def _warn(message: str, category: type = DataWarning) -> None:
    warnings.warn(message, category, stacklevel=3)


def season_length(freq: str) -> int:
    """Number of periods in one seasonal cycle for ``freq``."""
    return _SEASON_LENGTH[_check_freq(freq)]


def periods_per_year(freq: str) -> float:
    """Average number of periods of ``freq`` in one calendar year."""
    return _PERIODS_PER_YEAR[_check_freq(freq)]


def default_horizon(freq: str) -> int:
    """A sensible default forecast horizon (roughly one year / one cycle)."""
    return _DEFAULT_HORIZON[_check_freq(freq)]


def _check_freq(freq: str) -> str:
    if freq not in PANDAS_FREQ:
        raise DataError(msg("bad_freq", freq=freq))
    return freq


def _days(index: pd.DatetimeIndex) -> np.ndarray:
    """Timestamps as float days (resolution-agnostic; pandas 3 stores ``us``)."""
    return pd.DatetimeIndex(index).to_numpy().astype("datetime64[s]").astype(np.int64) / 86400.0


def infer_freq(index: pd.DatetimeIndex) -> str:
    """Infer the canonical frequency code from the spacing of ``index``."""
    if len(index) < 2:
        raise DataError(msg("freq_needs_two_dates"))
    idx = pd.DatetimeIndex(index).sort_values()
    diffs = np.diff(_days(idx))
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        raise DataError(msg("all_dates_equal"))
    # a low quantile (not the median) so that gappy data still maps to its native spacing
    d = float(np.quantile(diffs, 0.25))
    if d <= 1.5:
        return "D"
    if d <= 10:
        return "W"
    if d <= 45:
        return "M"
    if d <= 120:
        return "Q"
    return "Y"


def _weekly_alias(index: pd.DatetimeIndex) -> str:
    """``W-<DAY>`` anchored on the most common weekday of ``index`` (ties: latest obs)."""
    idx = pd.DatetimeIndex(index)
    if len(idx) == 0:
        return PANDAS_FREQ["W"]
    counts = Counter(int(d) for d in idx.dayofweek)
    top = max(counts.values())
    candidates = {d for d, c in counts.items() if c == top}
    anchor = int(idx.max().dayofweek) if int(idx.max().dayofweek) in candidates else min(candidates)
    return f"W-{_WEEKDAYS[anchor]}"


def _alias_for(freq: str, index: Optional[pd.DatetimeIndex] = None) -> str:
    alias = PANDAS_FREQ[_check_freq(freq)]
    if freq == "W" and index is not None and len(index):
        alias = _weekly_alias(index)
    return alias


def future_index(last_date: pd.Timestamp, freq: str, horizon: int) -> pd.DatetimeIndex:
    """The ``horizon`` timestamps that follow ``last_date`` at ``freq``.

    Weekly series continue on the weekday of ``last_date`` (a Monday-anchored
    history gets Monday forecast dates).
    """
    if int(horizon) < 1:
        raise DataError(msg("horizon_min"))
    last = pd.Timestamp(last_date)
    alias = PANDAS_FREQ[_check_freq(freq)]
    if freq == "W":
        alias = f"W-{_WEEKDAYS[int(last.dayofweek)]}"
    rng = pd.date_range(start=last, periods=horizon + 1, freq=alias)
    if rng[0] != last:  # last_date not on the grid: date_range rolled forward
        rng = pd.date_range(start=last, periods=horizon + 2, freq=alias)
        rng = rng[rng > last]
        return pd.DatetimeIndex(rng[:horizon])
    return pd.DatetimeIndex(rng[1:])


def regularize(series: pd.Series, freq: Optional[str] = None, how: str = "mean") -> pd.Series:
    """Resample ``series`` onto the canonical grid of ``freq`` and fill gaps.

    * duplicates / sub-period observations are aggregated with ``how``
      (``'mean'`` or ``'sum'``),
    * interior gaps are linearly interpolated in time,
    * leading / trailing missing periods are dropped,
    * weekly data keeps the weekday it was recorded on (``W-MON``...).
    """
    s, dates = _clean_with_dates(series)
    if freq is None:
        freq = _infer_freq_with_gaps(s.index, dates)
    alias = _alias_for(freq, s.index)
    if how not in ("mean", "sum"):
        raise DataError(msg("bad_how"))
    r = s.resample(alias)
    out = r.mean() if how == "mean" else r.sum(min_count=1)
    out = out.astype(float)
    if out.isna().any():
        out = out.interpolate(method="time", limit_area="inside")
    out = out.dropna()
    out.name = series.name
    out.index.name = "date"
    try:
        out.index.freq = alias
    except ValueError:  # pragma: no cover - defensive, grid should already be regular
        pass
    return out


# --------------------------------------------------------------------------- numbers

_MISSING_TOKENS = frozenset({
    "", "-", "--", "---", "—", "–", "−", ".", "..", "...", "…", "?", "*", "x",
    "na", "n/a", "n.a.", "nan", "null", "none", "nil", "nd", "n.d.", "missing",
    "#n/a", "#na", "#value!", "#div/0!", "#ref!", "#num!", "#name?", "#null!",
    "없음", "해당없음", "미상", "자료없음", "결측",
})
_MULTIPLIERS = {"천": 1e3, "만": 1e4, "억": 1e8, "조": 1e12, "k": 1e3, "K": 1e3, "M": 1e6,
                "mn": 1e6, "mil": 1e6, "B": 1e9, "bn": 1e9}
_NUM_RE = r"^(-?)(\d[\d.,]*(?:[eE]-?\d+)?|[.,]\d+(?:[eE]-?\d+)?)(.*)$"
_UNIT_RE = re.compile(r"^[A-Za-z가-힣/°℃℉㎏㎡㎞²³µ]{1,8}$")


def _decimal_votes(num: pd.Series) -> tuple[int, int]:
    """(votes for a decimal comma, votes for a decimal point) among number strings."""
    has_c = num.str.contains(",", regex=False)
    has_d = num.str.contains(".", regex=False)
    both = has_c & has_d
    last_c = num.str.rfind(",")
    last_d = num.str.rfind(".")
    comma = int((both & (last_c > last_d)).sum())
    comma += int(num.str.fullmatch(r"-?\d{1,3}(?:\.\d{3}){2,}(?:,\d+)?").sum())
    comma += int(num.str.fullmatch(r"-?\d+,(?:\d{1,2}|\d{4,})").sum())
    dot = int((both & (last_d > last_c)).sum())
    dot += int(num.str.fullmatch(r"-?\d{1,3}(?:,\d{3}){2,}(?:\.\d+)?").sum())
    dot += int(num.str.fullmatch(r"-?\d+\.(?:\d{1,2}|\d{4,})").sum())
    return comma, dot


def parse_numbers(col: pd.Series, decimal: Optional[str] = None,
                  prefer_decimal_comma: bool = False) -> tuple[pd.Series, int, list[str]]:
    """Convert a column of numbers written as text into floats.

    Handles thousands separators (``1,234`` / ``1.234,5`` / ``1 234``), decimal
    commas (auto-detected for the whole column, or ``decimal=','``), ``%`` and
    currency marks (``$ € £ ¥ ₩ 원``), trailing unit words (``명``, ``kg``),
    multipliers (``1.2억``, ``3,400만``, ``12K``, ``1.5M``), accounting
    negatives ``(1,234)`` and the usual missing-value markers (``-``, ``NA``,
    ``#N/A`` ...).  ``prefer_decimal_comma`` breaks ties for files that use
    ``;`` as the separator.

    Returns ``(values, n_unreadable, examples)`` where ``n_unreadable`` counts
    non-empty cells that could not be read (they become NaN).
    """
    if pd.api.types.is_bool_dtype(col):
        return col.astype(float), 0, []
    if pd.api.types.is_numeric_dtype(col):
        return pd.to_numeric(col, errors="coerce").astype(float), 0, []
    raw = col.astype("string").str.strip()
    missing = raw.isna() | raw.str.lower().isin(_MISSING_TOKENS)
    plain = pd.to_numeric(raw.where(~missing), errors="coerce").astype(float)
    if bool((plain.notna() | missing).all()):  # fast path: ordinary numbers only
        out = pd.Series(plain.to_numpy(dtype=float), index=col.index, name=col.name)
        out[~np.isfinite(out.to_numpy())] = np.nan
        bad = (~missing.to_numpy()) & out.isna().to_numpy()
        return out, int(bad.sum()), [str(v) for v in raw[bad].head(3)]
    t = raw.where(~missing)
    t = t.str.replace("\u2212", "-", regex=False)
    t = t.str.replace(r"[\s\u00a0\u202f\u2009'’]", "", regex=True)
    neg = t.str.fullmatch(r"\(.+\)").fillna(False).astype(bool)
    t = t.str.replace(r"^\((.+)\)$", r"\1", regex=True)
    t = t.str.replace(r"(?i)^(?:usd|krw|eur|jpy|cny|rmb|gbp)|(?:usd|krw|eur|jpy|cny|rmb|gbp)$", "",
                      regex=True)
    t = t.str.replace(r"[$€£¥₩￦%‰+]", "", regex=True)
    t = t.str.replace(r"^-(?=\D*$)", "", regex=True)  # lone sign left over
    parts = t.str.extract(_NUM_RE)
    sign = parts[0].fillna("")
    num = parts[1]
    suffix = parts[2].fillna("")
    if decimal not in (",", ".") and not bool(num.str.contains(",", regex=False).any()) \
            and not bool((num.str.count(r"\.") >= 2).any()):
        decimal = "."  # no comma anywhere and no 1.234.567 grouping: a plain decimal point
    if decimal not in (",", "."):
        comma, dot = _decimal_votes(num.dropna())
        if comma > dot:
            decimal = ","
        elif dot > comma:
            decimal = "."
        else:
            decimal = "," if (prefer_decimal_comma and bool(num.str.contains(",", regex=False).any())) else "."
    if decimal == ",":
        num = num.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    else:
        num = num.str.replace(",", "", regex=False)
    val = pd.to_numeric(num, errors="coerce").astype(float)
    mult = pd.Series(1.0, index=col.index)
    has_suffix = suffix.str.len() > 0
    if bool(has_suffix.any()):
        mult = suffix.map(lambda u: _MULTIPLIERS.get(u, 1.0 if (u == "" or _UNIT_RE.match(u)) else np.nan))
        mult = mult.astype(float)
    sgn = np.where(sign.to_numpy() == "-", -1.0, 1.0) * np.where(neg.to_numpy(), -1.0, 1.0)
    out = pd.Series(val.to_numpy(dtype=float) * mult.to_numpy(dtype=float) * sgn, index=col.index,
                    name=col.name)
    out[~np.isfinite(out.to_numpy())] = np.nan
    bad = (~missing.to_numpy()) & out.isna().to_numpy()
    examples = [str(v) for v in raw[bad].head(3)]
    return out, int(bad.sum()), examples


# --------------------------------------------------------------------------- dates

_TIME = r"(?:(?:\s+|T)(\d{1,2}):(\d{2})(?::(\d{2})(?:[.,]\d+)?)?)?"
_TZ_SUFFIX = re.compile(r"(\d{1,2}:\d{2}(?::\d{2}(?:[.,]\d+)?)?)\s*(?:Z|UTC|GMT|[+-]\d{2}(?::?\d{2})?)$",
                        re.IGNORECASE)
_DOW_SUFFIX = r"\s*\((?:[월화수목금토일](?:요일)?|mon|tue|wed|thu|fri|sat|sun)[a-z]*\.?\)$"
_DATE_NAME_EXACT = frozenset({
    "date", "dates", "time", "timestamp", "datetime", "day", "week", "month", "year", "period",
    "quarter", "ts", "dt", "ds", "yyyymm", "yyyymmdd", "ym", "ymd", "yearmonth", "날", "일", "주",
    "월", "년", "연", "분기", "주차",
})
_DATE_NAME_PART = ("date", "time", "period", "날짜", "일자", "일시", "시점", "기간", "연도", "년도",
                   "연월", "년월", "기준일", "기준월", "기준년", "기준연", "조사일", "발표일")


def _name_looks_like_date(name: Any) -> bool:
    n = re.sub(r"[\s_\-./()]+", "", str(name)).lower()
    return n in _DATE_NAME_EXACT or any(p in n for p in _DATE_NAME_PART)


def _to_float(x: pd.Series) -> pd.Series:
    """Numeric float64 Series from strings / nullable dtypes (missing -> NaN, never pd.NA)."""
    return pd.to_numeric(x.astype(object).where(x.notna(), np.nan), errors="coerce").astype(float)


def _from_parts(year: Any, month: Any = 1, day: Any = 1, hour: Any = None, minute: Any = None,
                second: Any = None, index: Optional[pd.Index] = None) -> pd.Series:
    """Vectorised Timestamp construction from (string/int) components; invalid -> NaT."""
    def num(x: Any, default: Optional[float]) -> Any:
        if isinstance(x, pd.Series):
            v = _to_float(x)
            return v if default is None else v.fillna(default)
        return default if x is None else x

    parts = {"year": num(year, None), "month": num(month, 1), "day": num(day, 1)}
    if hour is not None:
        parts.update(hour=num(hour, 0), minute=num(minute, 0), second=num(second, 0))
    frame = pd.DataFrame(parts, index=index if index is not None else getattr(year, "index", None))
    try:
        res = pd.to_datetime(frame, errors="coerce")
    except (ValueError, OverflowError, TypeError):  # pragma: no cover - defensive
        return pd.Series(pd.NaT, index=frame.index, dtype="datetime64[us]")
    return pd.Series(res, index=frame.index)


def _two_digit_year(yy: pd.Series) -> pd.Series:
    y = _to_float(yy)
    return y.where(y >= 100, np.where(y < 69, 2000 + y, 1900 + y))


def _naive(v: Any) -> pd.Timestamp:
    """Timestamp in local wall time (offset dropped); NaT for missing."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return pd.NaT
    ts = pd.Timestamp(v)
    return ts.tz_localize(None) if ts.tzinfo is not None else ts


def _coverage_score(ts: pd.Series) -> float:
    """How plausible a parsed date column is: monotone order + density on its own grid."""
    t = pd.DatetimeIndex(ts.dropna())
    if len(t) < 2:
        return 0.0
    u = t.unique().sort_values()
    if len(u) < 2:
        return 0.0
    try:
        f = infer_freq(u)
    except DataError:
        return 0.0
    span = float(_days(u[-1:])[0] - _days(u[:1])[0])
    grid = span / _STEP_DAYS[f] + 1.0
    cov = min(1.0, len(u) / max(grid, 1.0))
    mono = 0.5 if (t.is_monotonic_increasing or t.is_monotonic_decreasing) else 0.0
    return cov + mono


class _Candidate:
    """One whole-column interpretation of a date column."""

    def __init__(self, label: str, values: pd.Series, kind: str = "normal",
                 ambiguous: Optional[Msg] = None) -> None:
        self.label = label
        self.values = values
        self.kind = kind            # 'normal' or 'weak' (needs a date-like name / monotone order)
        self.ambiguous = ambiguous  # catalog note (Msg) when another order was equally plausible


def _pick_order(options: list[tuple[str, pd.Series]], tie_index: int) -> tuple[str, pd.Series, bool]:
    """Choose between alternative field orders: most parsed, then most plausible.

    Returns ``(label, values, tied)``; on a tie ``options[tie_index]`` wins.
    """
    keys = [(int(v.notna().sum()), round(_coverage_score(v), 6)) for _, v in options]
    top = max(keys)
    tied = keys.count(top) > 1
    i = tie_index if (tied and keys[tie_index] == top) else keys.index(top)
    return options[i][0], options[i][1], tied


def _string_candidates(s: pd.Series) -> Iterator[_Candidate]:
    """Whole-column interpretations of the (normalised, non-missing) strings ``s``, lazily."""
    idx = s.index

    def ext(pattern: str) -> pd.DataFrame:
        return s.str.extract(pattern, flags=re.IGNORECASE)

    # 2024-01-05, 2024/1/5, 2024.01.05, 2024. 1. 5., with optional time
    m = ext(r"^(\d{4})\s*([-/.])\s*(\d{1,2})\s*\2\s*(\d{1,2})\.?" + _TIME + "$")
    if m[0].notna().any():
        sep = m[1].dropna().mode().iloc[0] if m[1].notna().any() else "-"
        label = {"-": "YYYY-MM-DD", "/": "YYYY/MM/DD", ".": "YYYY.MM.DD"}[sep]
        has_time = m[4].notna().any()
        yield (_Candidate(label + (" hh:mm" if has_time else ""),
                              _from_parts(m[0], m[2], m[3], m[4] if has_time else None, m[5], m[6],
                                          index=idx)))
    # 2024-01, 2024.01, 2024/1, 2024. 1.
    m = ext(r"^(\d{4})\s*([-/.])\s*(\d{1,2})\.?$")
    if m[0].notna().any():
        sep = m[1].dropna().mode().iloc[0] if m[1].notna().any() else "-"
        label = {"-": "YYYY-MM", "/": "YYYY/MM", ".": "YYYY.MM"}[sep]
        yield (_Candidate(label, _from_parts(m[0], m[2], 1, index=idx),
                              kind="weak" if sep == "." else "normal"))
    # 2024년 1월 (5일)
    m = ext(r"^(\d{4})\s*년\s*(\d{1,2})\s*월(?:\s*(\d{1,2})\s*일)?\.?$")
    if m[0].notna().any():
        yield (_Candidate("YYYY년 M월" + (" D일" if m[2].notna().any() else ""),
                              _from_parts(m[0], m[1], m[2], index=idx)))
    # quarters: 2024Q1, 2024-Q1, Q1 2024, 2024 1Q, 1Q 2024, 2024년 1분기, 2024 1/4분기, 2024.1/4
    q_year = pd.Series(np.nan, index=idx)
    q_num = pd.Series(np.nan, index=idx)
    for pat, yi, qi in ((r"^(\d{4})\s*[-_./]?\s*Q\s*([1-4])$", 0, 1),
                        (r"^Q([1-4])\s*[-_./]?\s*(\d{4})$", 1, 0),
                        (r"^(\d{4})\s*[-_./]?\s*([1-4])\s*Q$", 0, 1),
                        (r"^([1-4])\s*Q\s*[-_./]?\s*(\d{4})$", 1, 0),
                        (r"^(\d{4})\s*(?:년|\.)?\s*([1-4])\s*(?:/\s*4)?\s*분기$", 0, 1),
                        (r"^(\d{4})\s*[.]?\s*([1-4])\s*/\s*4$", 0, 1)):
        m = ext(pat)
        hit = m[0].notna() & q_year.isna()
        if hit.any():
            q_year[hit] = _to_float(m.loc[hit, yi])
            q_num[hit] = _to_float(m.loc[hit, qi])
    if q_year.notna().any():
        yield (_Candidate("YYYY Qn (분기)", _from_parts(q_year, (q_num - 1) * 3 + 1, 1, index=idx)))
    # ISO weeks: 2024-W01, 2024W01-3, 2024년 1주(차)
    m = ext(r"^(\d{4})\s*-?\s*W\s*(\d{1,2})(?:\s*-?\s*([1-7]))?$")
    m2 = ext(r"^(\d{4})\s*년\s*(\d{1,2})\s*주(?:차)?$")
    if m[0].notna().any() or m2[0].notna().any():
        y = m[0].fillna(m2[0])
        w = m[1].fillna(m2[1])
        d = m[2]
        vals = []
        for yy, ww, dd in zip(y, w, d):
            try:
                vals.append(pd.Timestamp(date.fromisocalendar(int(yy), int(ww), int(dd) if isinstance(dd, str) else 1)))
            except (TypeError, ValueError):
                vals.append(pd.NaT)
        yield (_Candidate("YYYY-Www (ISO 주)", pd.Series(pd.DatetimeIndex(vals), index=idx)))
    # compact: 20240105 / 202401
    m = ext(r"^(\d{4})(\d{2})(\d{2})$")
    if m[0].notna().any():
        yield (_Candidate("YYYYMMDD", _from_parts(m[0], m[1], m[2], index=idx), kind="weak"))
    m = ext(r"^(\d{4})(\d{2})$")
    if m[0].notna().any():
        yield (_Candidate("YYYYMM", _from_parts(m[0], m[1], 1, index=idx), kind="weak"))
    # plain years (2024, 2024년)
    m = ext(r"^(\d{4})\s*년?$")
    if m[0].notna().any():
        y = _to_float(m[0])
        y = y.where((y >= 1000) & (y <= 2200))
        yield (_Candidate("YYYY (연도)", _from_parts(y, 1, 1, index=idx), kind="weak"))
    # Unix seconds (9-10 digits) / milliseconds (12-13 digits)
    m = ext(r"^(\d{9,10}|\d{12,13})$")
    if m[0].notna().any():
        v = _to_float(m[0])
        ms = v > 1e11
        secs = v.where(~ms, v / 1000.0)
        ok = (secs >= 1e8) & (secs <= 4.2e9)
        ts = pd.to_datetime(secs.where(ok), unit="s", errors="coerce")
        yield (_Candidate("Unix " + ("ms" if bool(ms.any()) else "초"), pd.Series(ts, index=idx),
                              kind="epoch"))
    # Excel serial day numbers (45292 = 2024-01-01), only for date-named / chosen columns
    m = ext(r"^(\d{5})(?:\.0+)?$")
    if m[0].notna().any():
        v = _to_float(m[0])
        v = v.where((v >= 20000) & (v <= 80000))
        ts = pd.to_datetime(v, unit="D", origin="1899-12-30", errors="coerce")
        yield (_Candidate("엑셀 날짜 일련번호", pd.Series(ts, index=idx), kind="serial"))
    # day/month order: 05.01.2024 / 01/05/2024 / 5-1-24 (+ time)
    m = ext(r"^(\d{1,2})\s*([-/.])\s*(\d{1,2})\s*\2\s*(\d{4}|\d{2})\.?" + _TIME + "$")
    if m[0].notna().any():
        sep = m[1].dropna().mode().iloc[0] if m[1].notna().any() else "/"
        has_time = m[4].notna().any()
        year = _two_digit_year(m[3])
        tm = (m[4], m[5], m[6]) if has_time else (None, None, None)
        dmy = _from_parts(year, m[2], m[0], *tm, index=idx)
        mdy = _from_parts(year, m[0], m[2], *tm, index=idx)
        s_ = sep
        lab_d = f"DD{s_}MM{s_}YYYY" + (" hh:mm" if has_time else "")
        lab_m = f"MM{s_}DD{s_}YYYY" + (" hh:mm" if has_time else "")
        first, second = m[0].dropna(), m[2].dropna()
        if first.nunique() == 1 and second.nunique() > 1:
            tie = 0  # constant first field = the day of monthly data (01.02.2022, 01.03.2022 ...)
        elif second.nunique() == 1 and first.nunique() > 1:
            tie = 1
        else:
            tie = 0 if sep == "." else 1
        label, vals, tied = _pick_order([(lab_d, dmy), (lab_m, mdy)], tie_index=tie)
        note = None
        if tied and first.nunique() > 1 and second.nunique() > 1:
            note = msg("date_order_ambiguous", label=label)
        yield (_Candidate(label, vals, ambiguous=note))
    # month/year: 01/2024, 1.2024
    m = ext(r"^(\d{1,2})\s*[-/.]\s*(\d{4})$")
    if m[0].notna().any():
        yield (_Candidate("MM/YYYY", _from_parts(m[1], m[0], 1, index=idx)))
    # two-digit year + month: 24-01 (or 01-24)
    m = ext(r"^(\d{2})\s*([-/.])\s*(\d{2})$")
    if m[0].notna().any():
        ym = _from_parts(_two_digit_year(m[0]), m[2], 1, index=idx)
        my = _from_parts(_two_digit_year(m[2]), m[0], 1, index=idx)
        label, vals, tied = _pick_order([("YY-MM", ym), ("MM-YY", my)], tie_index=0)
        note = None
        if tied:
            note = msg("date_ym_ambiguous", value=str(s.iloc[0]), label=label)
        yield (_Candidate(label, vals, kind="weak", ambiguous=note))


def _mixed_candidate(s: pd.Series) -> Optional[_Candidate]:
    """Last resort: pandas' per-element parser, with one day/month order for the column."""
    opts = []
    for dayfirst in (False, True):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                v = pd.to_datetime(s, errors="coerce", format="mixed", dayfirst=dayfirst)
            except (ValueError, TypeError, OverflowError):
                try:
                    v = pd.to_datetime(s, errors="coerce", format="mixed", dayfirst=dayfirst, utc=True)
                    v = v.dt.tz_localize(None)
                except (ValueError, TypeError, OverflowError):
                    continue
        if getattr(v.dt, "tz", None) is not None:
            v = v.dt.tz_localize(None)
        opts.append(("기타 날짜 형식" + (" (일/월 순)" if dayfirst else ""), pd.Series(v, index=s.index)))
    if not opts:
        return None
    label, vals, _ = _pick_order(opts, tie_index=0)
    return _Candidate(label, vals)


def _normalise_date_strings(col: pd.Series) -> pd.Series:
    s = col.astype("string").str.strip()
    s = s.str.replace(r"\s+", " ", regex=True)
    s = s.str.replace(_DOW_SUFFIX, "", regex=True, flags=re.IGNORECASE)
    s = s.str.replace(_TZ_SUFFIX, r"\1", regex=True)  # keep local wall time of offset timestamps
    s = s.str.replace(r"\.$", "", regex=True).str.strip()
    s = s.where(~s.str.lower().isin(_MISSING_TOKENS))
    return s


def _numeric_dates_as_strings(col: pd.Series) -> pd.Series:
    """Numbers from Excel/JSON -> date-like text: 2024 -> '2024', 2021.1 -> '2021.10'."""
    v = pd.to_numeric(col, errors="coerce").astype(float)
    finite = v.notna() & np.isfinite(v)
    out = pd.Series(pd.NA, index=col.index, dtype="string")
    is_int = finite & (v % 1 == 0)
    out[is_int] = v[is_int].map(lambda x: str(int(x)))
    frac = finite & ~is_int
    out[frac] = v[frac].map(lambda x: f"{x:.2f}")  # KOSIS '2021.10' read as 2021.1 -> October
    return out


def parse_dates(col: pd.Series, name_hint: bool = True,
                allow_epoch: bool = True) -> tuple[Optional[pd.Series], str, Optional[str]]:
    """Interpret ``col`` as dates with one format for the whole column.

    Returns ``(timestamps, format_label, note)``; ``timestamps`` is ``None``
    when fewer than 80 % of the non-empty cells can be read with any single
    format.  ``name_hint=False`` (the column name does not look like a date)
    makes the weak interpretations (plain years, ``YYYYMM``, ``YYYY.MM``) also
    require a sorted column; epoch seconds need ``allow_epoch``.  ``note`` is
    a Korean remark (ambiguous order, unreadable cells) or ``None`` -- a
    :class:`trendai.messages.Msg`, so it also renders in English (``note.text('en')``).
    """
    if isinstance(col.dtype, pd.DatetimeTZDtype):
        return pd.Series(col.dt.tz_localize(None), index=col.index), "날짜/시간", None
    if pd.api.types.is_datetime64_any_dtype(col):
        return pd.Series(pd.to_datetime(col), index=col.index), "날짜/시간", None
    if pd.api.types.is_bool_dtype(col):
        return None, "", None
    if col.dtype == object and len(col.dropna()) and all(
            isinstance(v, (pd.Timestamp, date)) for v in col.dropna()):
        return pd.Series(pd.DatetimeIndex([_naive(v) for v in col]), index=col.index), "날짜/시간", None
    from_numbers = pd.api.types.is_numeric_dtype(col)
    s = _numeric_dates_as_strings(col) if from_numbers else _normalise_date_strings(col)
    present = s.notna()
    n = int(present.sum())
    if n == 0:
        return None, "", None
    sp = s[present]
    best: Optional[_Candidate] = None
    best_frac = 0.0
    for cand in _string_candidates(sp):
        vals = cand.values
        frac = float(vals.notna().sum()) / n
        if (cand.kind == "epoch" and not allow_epoch) or (cand.kind == "serial" and not name_hint):
            continue
        if cand.kind in ("weak", "epoch") and not name_hint:
            v = pd.DatetimeIndex(vals.dropna())
            if not (v.is_monotonic_increasing or v.is_monotonic_decreasing):
                continue
        if from_numbers and cand.label not in ("YYYY (연도)", "YYYYMM", "YYYYMMDD", "YYYY.MM",
                                               "엑셀 날짜 일련번호") and not cand.label.startswith("Unix"):
            continue
        if frac > best_frac:
            best, best_frac = cand, frac
        if frac >= 1.0:
            break
    if best is None or best_frac < 0.8:
        pure_numbers = bool(sp.str.fullmatch(r"[-+]?[\d.,]+").all())
        has_letters = float(sp.str.contains(r"[A-Za-z]", regex=True).mean()) >= 0.8
        if not from_numbers and not pure_numbers and (name_hint or has_letters):
            mixed = _mixed_candidate(sp)
            if mixed is not None:
                frac = float(mixed.values.notna().sum()) / n
                if frac > best_frac and frac >= 0.8:
                    best, best_frac = mixed, frac
    if best is None or best_frac < 0.8:
        return None, "", None
    full = pd.Series(pd.NaT, index=col.index, dtype="datetime64[us]")
    full[present] = pd.DatetimeIndex(best.values).as_unit("us")
    note = best.ambiguous
    n_bad = n - int(best.values.notna().sum())
    if n_bad:
        bad_ex = ", ".join(repr(str(x)) for x in sp[best.values.isna()].head(3))
        note = join(note, msg("dates_unreadable_rows", n=n_bad, examples=bad_ex))
    return full, best.label, note


# --------------------------------------------------------------------------- series cleaning


def _coerce_index(index: pd.Index) -> pd.DatetimeIndex:
    """Turn any reasonable index into a naive DatetimeIndex (or raise a Korean DataError)."""
    if isinstance(index, pd.DatetimeIndex):
        return index.tz_localize(None) if index.tz is not None else index
    if isinstance(index, pd.PeriodIndex):
        return index.to_timestamp(how="start")
    if len(index) and pd.api.types.is_numeric_dtype(index.dtype) and not pd.api.types.is_bool_dtype(index.dtype):
        v = pd.to_numeric(pd.Series(index), errors="coerce").astype(float)
        finite = v[np.isfinite(v)]
        if len(finite) and bool(((finite % 1 == 0) & (finite >= 1000) & (finite <= 2200)).all()):
            years = v.where(np.isfinite(v))
            return pd.DatetimeIndex(_from_parts(years, 1, 1, index=pd.RangeIndex(len(v))).to_numpy())
        lo, hi = (float(finite.min()), float(finite.max())) if len(finite) else (0.0, 0.0)
        raise DataError(msg("index_numeric", lo=_index_number(lo), hi=_index_number(hi)))
    parsed, _, note = parse_dates(pd.Series(list(index), dtype=object), name_hint=True)
    if parsed is None:
        ex = ", ".join(repr(str(x)) for x in list(index[:3]))
        raise DataError(msg("index_unreadable", examples=ex))
    if note:
        _warn(note, DataNotice)
    return pd.DatetimeIndex(parsed.to_numpy())


def _index_number(x: float) -> str:
    """A number of a numeric index for a message: integers as such (``36``), others in the shared
    notation (:func:`trendai.numfmt.number_text`; never Python's ``1e+07``)."""
    if float(x).is_integer() and abs(x) < _numfmt.SCI_LARGE:
        return f"{int(x):,}".replace("-", _numfmt.MINUS)
    return _numfmt.number_text(x) or str(x)


def _clean(series: pd.Series) -> pd.Series:
    return _clean_with_dates(series)[0]


def _clean_with_dates(series: pd.Series) -> tuple[pd.Series, pd.DatetimeIndex]:
    """``(finite values sorted by date, dates of every row with a valid date)``.

    The second element keeps the dates of rows whose value is missing: a
    monthly date column with many empty cells still says the data is monthly.
    """
    if not isinstance(series, pd.Series):
        raise DataError(msg("series_required"))
    if series.empty:
        raise DataError(msg("series_empty"))
    s = series.copy()
    s.index = _coerce_index(s.index)
    vals, n_bad, ex = parse_numbers(s)
    if n_bad:
        _warn(msg("values_unreadable", name=_label(series), n=n_bad, examples=", ".join(map(repr, ex))))
    s = pd.Series(vals.to_numpy(dtype=float), index=s.index, name=series.name)
    arr = s.to_numpy()
    n_inf = int(np.isinf(arr).sum())
    if n_inf:
        _warn(msg("values_infinite", name=_label(series), n=n_inf))
    dates = pd.DatetimeIndex(s.index[~s.index.isna()]).sort_values()
    s = s[np.isfinite(arr)]
    s = s[~s.index.isna()]
    s = s.sort_index(kind="mergesort")
    if s.empty:
        raise DataError(msg("no_valid_values", name=_label(series)))
    return s, dates


def _infer_freq_with_gaps(observed: pd.DatetimeIndex, dates: pd.DatetimeIndex) -> str:
    """Frequency of a series whose date column may have rows without a value.

    The spacing of the observed values alone turns a monthly table with many
    empty cells into a coarser one (36 monthly rows with 9 values read as
    quarterly averages).  So the frequency of the full date column is used,
    unless the observed values cover at least 80 % of the periods of their
    own, coarser frequency (e.g. month-start values in a daily calendar, or
    Monday values with a couple of holiday weeks missing), which is then kept:
    a few empty periods of a weekly or monthly series are gaps of that series,
    not a reason to interpolate it into a mostly made-up daily one.
    """
    f_obs = infer_freq(observed)
    if len(dates) <= len(observed) or len(pd.DatetimeIndex(dates).unique()) < 2:
        return f_obs
    try:
        f_all = infer_freq(dates)
    except DataError:
        return f_obs
    if f_all == f_obs:
        return f_obs
    counts = pd.Series(1.0, index=observed).resample(_alias_for(f_obs, observed)).count()
    if len(counts) and float((counts > 0).mean()) >= _OWN_GRID_COVERAGE:
        return f_obs
    return f_all


#: share of the periods of the observed values' own (coarser) frequency that must hold a value
#: for that frequency to be kept over the date column's finer one (_infer_freq_with_gaps)
_OWN_GRID_COVERAGE = 0.8


def _label(series: pd.Series) -> str:
    return f"'{series.name}': " if series.name is not None else ""


def _weekday_only(index: pd.DatetimeIndex) -> bool:
    """Business-day data: no Saturday/Sunday observations over at least two weeks."""
    idx = pd.DatetimeIndex(index)
    if len(idx) < 10:
        return False
    if bool((idx.dayofweek >= 5).any()):
        return False
    span = float(_days(idx[-1:])[0] - _days(idx[:1])[0])
    return span >= 14


def _report_regularisation(raw: pd.Series, out: pd.Series, freq: str, how: str) -> None:
    """Tell the user when regularisation filled or merged a noticeable share of the data."""
    alias = _alias_for(freq, raw.index)
    counts = raw.resample(alias).count()
    counts = counts.reindex(out.index, fill_value=0)
    observed = int((counts > 0).sum())
    filled = len(out) - observed
    merged = int(len(raw) - observed)
    name = _label(raw)
    if freq == "D" and _weekday_only(raw.index):
        _warn(msg("weekday_only", name=name, n_obs=len(raw), filled=filled, n_out=len(out)))
    elif filled >= 3 and filled / max(len(out), 1) >= 0.2:
        _warn(msg("gaps_interpolated", name=name, unit=unit(freq), filled=filled, total=len(out),
                  share=filled / len(out), observed=observed))
    if merged > 0 and merged / max(len(raw), 1) >= 0.05:
        verb = L("평균", "averaged") if how == "mean" else L("합계", "summed")
        _warn(msg("duplicates_merged", name=name, unit=unit(freq), how=verb, n_raw=len(raw),
                  observed=observed))


def prepare(series: pd.Series, freq: Optional[str] = None, name: Optional[str] = None,
            how: str = "mean") -> tuple[pd.Series, str]:
    """Clean + regularise a raw series. Returns ``(series, freq)``.

    Accepts a DatetimeIndex (tz-aware ok), PeriodIndex, integer years
    (1000..2200) or date strings as the index, and numbers or number strings
    (``'1,234'``) as values.  The frequency is inferred from every dated row,
    including rows whose value is missing (see :func:`_infer_freq_with_gaps`).
    When the frequency is inferred, a :class:`DataWarning` reports
    weekday-only data, large interpolated gaps or merged duplicate periods.
    """
    s, dates = _clean_with_dates(series)
    if len(s) < 2:
        raise DataError(msg("too_few_obs", name=_label(series), n=len(s)))
    inferred = freq is None
    if freq is None:
        freq = _infer_freq_with_gaps(s.index, dates)
    out = regularize(s, freq, how=how)
    if inferred:
        _report_regularisation(s, out, freq, how)
    if name is not None:
        out.name = name
    elif out.name is None:
        out.name = "series"
    if len(out) < 2:
        raise DataError(msg("too_few_after_regularise", name=_label(series), unit=unit(freq), n=len(out)))
    return out, freq


# --------------------------------------------------------------------------- tables

_INDEX_NAME_RE = re.compile(
    r"^(?:unnamed:?\s*\d*|index|idx|level_\d+|id|no\.?|#|seq|row|rownum|row_?id|순번|번호|연번|"
    r"일련번호|행번호)$", re.IGNORECASE)


def _is_step1_sequence(v: pd.Series) -> bool:
    x = v.dropna().to_numpy(dtype=float)
    if len(x) < 3 or len(x) != len(v) or not np.all(x % 1 == 0):
        return False
    d = np.diff(x)
    return bool(np.all(d == 1.0) or np.all(d == -1.0))


def _clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [re.sub(r"\s+", " ", str(c).replace("\ufeff", "")).strip() for c in df.columns]
    if not df.empty:
        blank = df.apply(lambda c: c.isna() | c.astype("string").str.strip().fillna("").eq(""))
        df = df.loc[~blank.all(axis=1), ~blank.all(axis=0).to_numpy()]
    return df


def _find_date_column(df: pd.DataFrame) -> Optional[tuple[str, pd.Series, str, Optional[str]]]:
    """First column (date-like names first) that reads as dates with one format."""
    cols = list(df.columns)
    hinted = [c for c in cols if _name_looks_like_date(c)]
    for strict_pass, group in ((False, hinted), (True, [c for c in cols if c not in hinted])):
        for c in group:
            parsed, label, note = parse_dates(df[c], name_hint=not strict_pass,
                                              allow_epoch=(not strict_pass) or c == cols[0])
            if parsed is not None:
                return c, parsed, label, note
    return None


def _wide_to_series(df: pd.DataFrame, prefer_decimal_comma: bool) -> Optional[dict[str, pd.Series]]:
    """Dates across the columns (KOSIS layout): one series per row."""
    names = pd.Series([str(c) for c in df.columns], dtype=object)
    parsed, label, _ = parse_dates(names, name_hint=True, allow_epoch=False)
    if parsed is None:  # some columns are labels (항목, 단위 ...): look only at the ones with digits
        digits = names[names.str.contains(r"\d", regex=True)]
        if len(digits) < 3:
            return None
        sub, label, _ = parse_dates(digits, name_hint=True, allow_epoch=False)
        if sub is None:
            return None
        parsed = pd.Series(pd.NaT, index=names.index, dtype="datetime64[us]")
        parsed[sub.index] = sub
    is_date = parsed.notna().to_numpy()
    if int(is_date.sum()) < 3:
        return None
    date_cols = [c for c, d in zip(df.columns, is_date) if d]
    id_cols = [c for c, d in zip(df.columns, is_date) if not d]
    dates = pd.DatetimeIndex(parsed[is_date].to_numpy())
    out: dict[str, pd.Series] = {}
    for i, (_, row) in enumerate(df.iterrows()):
        label_parts = [str(row[c]).strip() for c in id_cols
                       if pd.notna(row[c]) and str(row[c]).strip() not in ("", "nan")]
        nm = " / ".join(label_parts) or f"행{i + 1}"
        vals, _, _ = parse_numbers(row[date_cols].reset_index(drop=True),
                                   prefer_decimal_comma=prefer_decimal_comma)
        if int(vals.notna().sum()) < 2:
            continue
        base, k = nm, 2
        while nm in out:
            nm, k = f"{base} ({k})", k + 1
        s = pd.Series(vals.to_numpy(dtype=float), index=dates, name=nm)
        s.attrs["trendai"] = {"date_col": "(열 머리글)", "date_format": label, "n_rows": int(len(df))}
        out[nm] = s
    if not out:
        return None
    _warn(msg("wide_table", n=len(out), label=L(label, format_label(label, "en"))), DataNotice)
    return out


_MONTH_COL = frozenset({"월", "month", "mon", "mm", "m"})
_QUARTER_COL = frozenset({"분기", "quarter", "qtr", "q"})


def _combine_year_and_period(df: pd.DataFrame, date_col: str, parsed: pd.Series,
                             label: str) -> Optional[tuple[pd.Series, str, str]]:
    """Long tables with separate year and month/quarter columns (연도 | 월 | 값)."""
    if not label.startswith("YYYY (") or parsed.dropna().is_unique:
        return None
    for c in df.columns:
        if c == date_col:
            continue
        key = re.sub(r"[\s_]+", "", str(c)).lower()
        if key in _MONTH_COL:
            hi, step, what = 12, 1, "월"
        elif key in _QUARTER_COL:
            hi, step, what = 4, 3, "분기"
        else:
            continue
        txt = df[c].astype("string").str.strip().str.replace(r"(?i)^q|q$|월$|분기$", "", regex=True)
        num = _to_float(txt)
        ok = num.notna() & (num % 1 == 0) & (num >= 1) & (num <= hi)
        if float(ok.mean()) < 0.9:
            continue
        combined = _from_parts(parsed.dt.year.where(ok), (num.where(ok) - 1) * step + 1, 1,
                               index=df.index)
        return combined, f"YYYY + {what}", str(c)
    return None


def frame_to_series(df: pd.DataFrame, date_col: Optional[str] = None,
                    value_cols: Optional[Iterable[str]] = None, *,
                    decimal: Optional[str] = None,
                    prefer_decimal_comma: bool = False) -> dict[str, pd.Series]:
    """Split a table (one date column + N value columns) into raw series.

    The date column is detected automatically (column names such as
    ``date``/``날짜``/``기간`` are tried first) unless ``date_col`` is given;
    value columns are all columns whose cells read as numbers (see
    :func:`parse_numbers`), minus row-number columns.  A table whose dates run
    across the columns is read row by row.  Each returned series carries
    ``series.attrs['trendai']`` = ``{'date_col', 'date_format', 'n_rows'}``.
    """
    if df is None or df.empty:
        raise DataError(msg("table_empty_rows"))
    df = _clean_frame(df)
    if df.empty:
        raise DataError(msg("table_empty_values"))
    note: Optional[Msg] = None
    if date_col is not None:
        date_col = str(date_col).strip()
        if date_col not in df.columns:
            raise DataError(msg("date_col_missing", col=date_col, columns=list(df.columns)))
        parsed, label, note = parse_dates(df[date_col], name_hint=True, allow_epoch=True)
        if parsed is None:
            ex = ", ".join(repr(str(x)) for x in df[date_col].dropna().head(3))
            raise DataError(msg("date_col_unreadable", col=date_col, examples=ex))
    else:
        found = _find_date_column(df)
        if found is None:
            wide = _wide_to_series(df, prefer_decimal_comma)
            if wide is not None:
                return wide
            raise DataError(msg("no_date_col", columns=list(df.columns)))
        date_col, parsed, label, note = found
    period_col: Optional[str] = None
    combo = _combine_year_and_period(df, str(date_col), parsed, label)
    if combo is not None:
        parsed, label, period_col = combo
    if note:
        _warn(note, DataNotice)
    dates = pd.DatetimeIndex(parsed.to_numpy())
    if value_cols is not None:
        value_cols = [str(c).strip() for c in value_cols]
        for c in value_cols:
            if c not in df.columns:
                raise DataError(msg("value_col_missing", col=c, columns=list(df.columns)))
        explicit = True
    else:
        explicit = False
    candidates = value_cols if explicit else [c for c in df.columns if c not in (date_col, period_col)]
    parsed_vals: dict[str, tuple[pd.Series, int, list[str]]] = {}
    for c in candidates:
        vals, n_bad, ex = parse_numbers(df[c], decimal=decimal, prefer_decimal_comma=prefer_decimal_comma)
        n_ok = int(vals.notna().sum())
        if explicit:
            if n_ok < 1:
                ex_raw = ", ".join(repr(str(x)) for x in df[c].dropna().head(3))
                raise DataError(msg("value_col_unreadable", col=c, examples=ex_raw))
            parsed_vals[c] = (vals, n_bad, ex)
        elif n_ok >= 1 and n_ok / max(n_ok + n_bad, 1) >= 0.8:
            parsed_vals[c] = (vals, n_bad, ex)
    if not explicit:
        def index_like(c: str) -> bool:
            unnamed = str(c) == "" or str(c).lower().startswith("unnamed")
            if unnamed:  # a blank header over real values is kept; over 0..N-1 it is a row number
                return _is_step1_sequence(parsed_vals[c][0])
            return bool(_INDEX_NAME_RE.match(str(c)))

        by_name = [c for c in parsed_vals if index_like(c)]
        rest = [c for c in parsed_vals if c not in by_name]
        by_seq = [c for c in rest if _is_step1_sequence(parsed_vals[c][0])]
        keep = [c for c in rest if c not in by_seq] or by_seq
        skipped = [c for c in parsed_vals if c not in keep]
        if not keep:
            if skipped:
                raise DataError(msg("only_rownum_cols", cols=skipped))
            raise DataError(msg("no_numeric_col", columns=list(df.columns)))
        if skipped:
            _warn(msg("rownum_cols_skipped", cols=skipped), DataNotice)
        parsed_vals = {c: parsed_vals[c] for c in keep}
    valid_date = ~dates.isna()
    out: dict[str, pd.Series] = {}
    for c, (vals, n_bad, ex) in parsed_vals.items():
        s = pd.Series(vals.to_numpy(dtype=float), index=dates, name=str(c))
        if n_bad:
            pos = np.flatnonzero(vals.isna().to_numpy() & valid_date)
            edge: Any = ""
            ok_pos = np.flatnonzero(vals.notna().to_numpy())
            if len(ok_pos) and len(pos) and (pos.max() > ok_pos.max() or pos.min() < ok_pos.min()):
                edge = L(" 시계열의 처음/끝 부분이 잘렸을 수 있습니다.", " The start or end of the series may be cut off.")
            _warn(msg("col_values_unreadable", col=str(c), n=n_bad, examples=", ".join(map(repr, ex)), edge=edge))
        s = s[valid_date]
        shown = f"{date_col} + {period_col}" if period_col else str(date_col)
        s.attrs["trendai"] = {"date_col": shown, "date_format": label, "n_rows": int(len(df))}
        out[str(c)] = s
    return out


# --------------------------------------------------------------------------- files


def _decode(raw: bytes, what: str) -> tuple[str, str]:
    """Decode file bytes: BOM, then strict UTF-8, CP949, UTF-16 (no BOM), CP1252, Latin-1."""
    if raw.startswith(b"\xef\xbb\xbf"):
        return raw[3:].decode("utf-8", errors="replace"), "utf-8-sig"
    if raw.startswith((b"\xff\xfe\x00\x00", b"\x00\x00\xfe\xff")):
        return raw.decode("utf-32", errors="replace"), "utf-32"
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return raw.decode("utf-16", errors="replace"), "utf-16"
    try:
        return raw.decode("utf-8"), "utf-8"
    except UnicodeDecodeError:
        pass
    sample = raw[:4000]
    if sample.count(b"\x00") > len(sample) // 4:  # UTF-16 without BOM
        enc = "utf-16-le" if sample[1:2] == b"\x00" else "utf-16-be"
        try:
            return raw.decode(enc), enc
        except UnicodeDecodeError:
            pass
    bad_utf8 = raw.decode("utf-8", errors="replace").count("\ufffd")
    multibyte = len(re.findall(rb"[\xc2-\xf4][\x80-\xbf]+", raw))
    if multibyte >= 20 * max(bad_utf8, 1):
        _warn(msg("utf8_broken", what=what, n=bad_utf8))
        return raw.decode("utf-8", errors="replace"), "utf-8"
    for enc in ("cp949", "cp1252"):
        try:
            return raw.decode(enc), enc
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1"), "latin-1"


def _sniff_sep(lines: list[str]) -> Optional[str]:
    best: Optional[tuple[tuple[float, int, int], str]] = None
    for order, sep in enumerate([",", "\t", ";", "|"]):
        try:
            counts = [len(r) for r in csv.reader(lines, delimiter=sep)]
        except csv.Error:
            continue
        counts = [c for c in counts if c > 0]
        if not counts:
            continue
        mode, freq = Counter(counts).most_common(1)[0]
        if mode < 2:
            continue
        key = (freq / len(counts), mode, -order)
        if best is None or key > best[0]:
            best = (key, sep)
    if best is not None:
        return best[1]
    ws = [len(line.split()) for line in lines]
    if ws and min(ws) >= 2 and len(set(ws)) == 1:
        return r"\s+"
    return None


def _header_is_data(columns: list[str]) -> bool:
    """True when the 'header' row is really the first data row (headerless file)."""
    if len(columns) < 2:
        return False
    first, _, _ = parse_dates(pd.Series([columns[0]], dtype=object), name_hint=True, allow_epoch=False)
    if first is None or first.isna().all():
        return False
    vals, _, _ = parse_numbers(pd.Series(columns[1:], dtype=object))
    return bool(vals.notna().all())


def _read_text_table(text: str, what: str) -> tuple[pd.DataFrame, bool]:
    """Parse delimited text into an all-string DataFrame. Returns (df, decimal_comma_hint)."""
    text = text.lstrip("\ufeff")
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        raise DataError(msg("file_empty", what=what))
    sep = _sniff_sep(lines[:50]) or ","
    kw: dict[str, Any] = dict(sep=sep, dtype=str, keep_default_na=False, na_filter=False,
                              skipinitialspace=True, skip_blank_lines=True)
    if sep == r"\s+":
        kw["engine"] = "python"
    try:
        df = pd.read_csv(io.StringIO(text), **kw)
        if _header_is_data([str(c) for c in df.columns]):
            ncol = df.shape[1]
            names = ["date"] + (["value"] if ncol == 2 else [f"value{i}" for i in range(1, ncol)])
            df = pd.read_csv(io.StringIO(text), header=None, names=names, **kw)
            _warn(msg("headerless", what=what), DataNotice)
    except pd.errors.EmptyDataError:
        raise DataError(msg("file_empty", what=what)) from None
    except (pd.errors.ParserError, csv.Error) as exc:
        hint: Any = ""
        if sep == ",":
            hint = L(" 천 단위 쉼표(1,234)가 따옴표 없이 들어 있으면 열 개수가 달라집니다. "
                     "값을 따옴표로 감싸거나 탭/세미콜론 구분자로 저장하세요.",
                     " Thousands separators (1,234) without quotes change the number of columns: quote the "
                     "values or save the file with tabs or semicolons as separators.")
        raise DataError(msg("csv_ragged", what=what, error=str(exc), hint=hint)) from None
    return df, sep == ";"


def _json_to_frame(obj: Any, what: str) -> pd.DataFrame:
    """Accept the common JSON shapes: records, columns/index/split/table orients, {date: value}."""
    if isinstance(obj, dict) and isinstance(obj.get("data"), list) and ("schema" in obj or "columns" in obj):
        if "columns" in obj:
            df = pd.DataFrame(obj["data"], columns=obj["columns"])
            if "index" in obj and len(obj["index"]) == len(df):
                df.insert(0, "date", obj["index"])
            return df
        return pd.DataFrame(obj["data"])
    if isinstance(obj, dict) and len(obj) == 1:
        (only,) = obj.values()
        if isinstance(only, list) and only and all(isinstance(r, dict) for r in only):
            return pd.DataFrame(only)  # e.g. {"items": [...]} (Wikimedia API)
    if isinstance(obj, list):
        if not obj:
            raise DataError(msg("json_empty_array", what=what))
        if all(isinstance(r, dict) for r in obj):
            return pd.DataFrame(obj)
        if all(isinstance(r, (list, tuple)) for r in obj):
            head = obj[0]
            if all(isinstance(h, str) for h in head) and len(obj) > 1:
                return pd.DataFrame(obj[1:], columns=head)
            return pd.DataFrame(obj, columns=["date"] + [f"value{i}" if len(head) > 2 else "value"
                                                         for i in range(1, len(head))])
        raise DataError(msg("json_array_shape", what=what))
    if isinstance(obj, dict):
        if not obj:
            raise DataError(msg("json_empty_object", what=what))
        vals = list(obj.values())
        if all(not isinstance(v, (dict, list)) for v in vals):  # {date: value} (Series.to_json)
            return pd.DataFrame({"date": list(obj.keys()), "value": vals})
        if all(isinstance(v, list) for v in vals):
            return pd.DataFrame(obj)
        if all(isinstance(v, dict) for v in vals):
            keys = pd.Series([str(k) for k in obj.keys()], dtype=object)
            outer_dates, _, _ = parse_dates(keys, name_hint=True)
            if outer_dates is not None:  # orient='index': {date: {col: value}}
                df = pd.DataFrame.from_dict(obj, orient="index")
            else:  # orient='columns': {col: {date: value}}
                df = pd.DataFrame(obj)
            df.index.name = None
            return df.rename_axis("date").reset_index()
    raise DataError(msg("json_shape", what=what))


_EXCEL_SIGNATURES = (b"PK\x03\x04", b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1")


def _read_excel(p: Path, raw: bytes, what: str) -> pd.DataFrame:
    if not raw.startswith(_EXCEL_SIGNATURES):
        head = raw[:2000].lstrip().lower()
        if head.startswith((b"<!doctype html", b"<html", b"<table", b"<?xml")):
            raise DataError(msg("excel_is_html", what=what))
        _warn(msg("excel_is_text", what=what), DataNotice)
        return pd.DataFrame()  # caller falls back to text parsing
    engine_pkg = "xlrd" if raw.startswith(_EXCEL_SIGNATURES[1]) else "openpyxl"
    try:
        df = pd.read_excel(p, sheet_name=0)
    except ImportError:
        raise DataError(msg("excel_needs_pkg", what=what, pkg=engine_pkg)) from None
    except (ValueError, zipfile.BadZipFile, KeyError, OSError, TypeError) as exc:
        raise DataError(msg("excel_unreadable", what=what, error=str(exc))) from None
    except Exception as exc:  # noqa: BLE001 - engine-specific parse errors (xlrd.XLRDError, ...)
        raise DataError(msg("excel_unreadable_engine", what=what, error=f"{type(exc).__name__}: {exc}")) from None
    if _header_is_data([str(c) for c in df.columns]):
        try:
            df = pd.read_excel(p, sheet_name=0, header=None)
        except Exception:  # noqa: BLE001 - already read once; keep the first result
            return df
        ncol = df.shape[1]
        df.columns = ["date"] + (["value"] if ncol == 2 else [f"value{i}" for i in range(1, ncol)])
    return df


def load_table(path: Union[str, Path], date_col: Optional[str] = None,
               value_cols: Optional[Iterable[str]] = None) -> dict[str, pd.Series]:
    """Load a CSV/TSV/TXT/Excel/JSON file into ``{name: raw series}``.

    The encoding, separator, date column and numeric value columns are
    detected automatically unless given explicitly (see the module docstring
    for the supported formats).  Returned series are *raw*; pass them through
    :func:`prepare` (or :func:`trendai.pipeline.run`) before analysis.

    Raises :class:`DataError` with a Korean message for every unreadable input
    (missing / empty / directory path, damaged Excel, unknown JSON shape,
    no date or value column ...).
    """
    p = Path(path)
    what = p.name or str(p)
    if not p.exists():
        raise DataError(msg("file_not_found", path=str(p)))
    if p.is_dir():
        raise DataError(msg("path_is_dir", path=str(p)))
    try:
        raw = p.read_bytes()
    except OSError as exc:
        raise DataError(msg("file_unreadable", what=what, error=str(exc.strerror or exc))) from None
    if not raw.strip():
        raise DataError(msg("file_empty", what=what))
    suffix = p.suffix.lower()
    decimal_comma = False
    df: Optional[pd.DataFrame] = None
    if suffix in (".xlsx", ".xlsm", ".xls", ".ods"):
        df = _read_excel(p, raw, what)
        if df.empty and not raw.startswith(_EXCEL_SIGNATURES):
            df = None  # text content behind an Excel extension: parse as CSV below
    elif suffix == ".json":
        text, _ = _decode(raw, what)
        try:
            obj = json.loads(text.lstrip("\ufeff"))
        except json.JSONDecodeError as exc:
            try:  # JSON Lines
                obj = [json.loads(ln) for ln in text.lstrip("\ufeff").splitlines() if ln.strip()]
            except json.JSONDecodeError:
                raise DataError(msg("json_invalid", what=what, error=exc.msg, line=exc.lineno)) from None
        try:
            df = _json_to_frame(obj, what)
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            if isinstance(exc, DataError):
                raise
            raise DataError(msg("json_to_table", what=what, error=str(exc))) from None
    if df is None:
        if raw.startswith(_EXCEL_SIGNATURES):
            raise DataError(msg("excel_wrong_ext", what=what))
        text, _ = _decode(raw, what)
        df, decimal_comma = _read_text_table(text, what)
    return frame_to_series(df, date_col=date_col, value_cols=value_cols,
                           prefer_decimal_comma=decimal_comma)


def from_values(values: Iterable[float], start: str = "2020-01-01", freq: str = "M",
                name: str = "series") -> pd.Series:
    """Build a regular series from a plain list of numbers."""
    vals = np.asarray(list(values), dtype=float)
    idx = pd.date_range(start=start, periods=len(vals), freq=PANDAS_FREQ[_check_freq(freq)])
    return pd.Series(vals, index=idx, name=name)
