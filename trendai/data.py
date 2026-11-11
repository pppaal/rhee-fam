"""Loading and normalising time series.

Everything downstream expects a *regular* series: a float ``pd.Series`` with a
sorted ``DatetimeIndex`` on one of the canonical frequencies below, no
duplicates and no NaNs.  :func:`prepare` turns almost any user input into that.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Union

import numpy as np
import pandas as pd

# canonical frequency code -> pandas offset alias used for resampling / future dates
PANDAS_FREQ: dict[str, str] = {
    "D": "D",
    "W": "W-SUN",
    "M": "MS",
    "Q": "QS",
    "Y": "YS",
}

FREQ_KO: dict[str, str] = {"D": "일", "W": "주", "M": "월", "Q": "분기", "Y": "년"}

_SEASON_LENGTH = {"D": 7, "W": 52, "M": 12, "Q": 4, "Y": 1}
_PERIODS_PER_YEAR = {"D": 365.25, "W": 52.1775, "M": 12.0, "Q": 4.0, "Y": 1.0}
_DEFAULT_HORIZON = {"D": 30, "W": 26, "M": 12, "Q": 8, "Y": 5}


class DataError(ValueError):
    """Raised when input data cannot be turned into a usable series."""


def season_length(freq: str) -> int:
    """Number of periods in one seasonal cycle for ``freq``."""
    return _SEASON_LENGTH[_check_freq(freq)]


def periods_per_year(freq: str) -> float:
    return _PERIODS_PER_YEAR[_check_freq(freq)]


def default_horizon(freq: str) -> int:
    """A sensible default forecast horizon (roughly one year / one cycle)."""
    return _DEFAULT_HORIZON[_check_freq(freq)]


def _check_freq(freq: str) -> str:
    if freq not in PANDAS_FREQ:
        raise DataError(f"unknown frequency {freq!r}; expected one of {list(PANDAS_FREQ)}")
    return freq


def infer_freq(index: pd.DatetimeIndex) -> str:
    """Infer the canonical frequency code from the median spacing of ``index``."""
    if len(index) < 2:
        raise DataError("need at least 2 timestamps to infer a frequency")
    idx = pd.DatetimeIndex(index).sort_values()
    # resolution-agnostic (pandas >= 2 may store s/ms/us/ns)
    diffs = np.diff(idx.to_numpy().astype("datetime64[s]").astype(np.int64)) / 86400.0  # days
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        raise DataError("all timestamps are identical")
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


def future_index(last_date: pd.Timestamp, freq: str, horizon: int) -> pd.DatetimeIndex:
    """The ``horizon`` timestamps that follow ``last_date`` at ``freq``."""
    if horizon < 1:
        raise DataError("horizon must be >= 1")
    alias = PANDAS_FREQ[_check_freq(freq)]
    rng = pd.date_range(start=pd.Timestamp(last_date), periods=horizon + 1, freq=alias)
    if rng[0] != pd.Timestamp(last_date):  # last_date not on the grid: date_range rolled forward
        rng = pd.date_range(start=pd.Timestamp(last_date), periods=horizon + 2, freq=alias)
        rng = rng[rng > pd.Timestamp(last_date)]
        return pd.DatetimeIndex(rng[:horizon])
    return pd.DatetimeIndex(rng[1:])


def regularize(series: pd.Series, freq: Optional[str] = None, how: str = "mean") -> pd.Series:
    """Resample ``series`` onto the canonical grid of ``freq`` and fill gaps.

    * duplicates / sub-period observations are aggregated with ``how``
      (``'mean'`` or ``'sum'``),
    * interior gaps are linearly interpolated in time,
    * leading / trailing missing periods are dropped.
    """
    s = _clean(series)
    if freq is None:
        freq = infer_freq(s.index)
    alias = PANDAS_FREQ[_check_freq(freq)]
    if how not in ("mean", "sum"):
        raise DataError("how must be 'mean' or 'sum'")
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


def _clean(series: pd.Series) -> pd.Series:
    if not isinstance(series, pd.Series):
        raise DataError("expected a pandas Series")
    s = series.copy()
    if not isinstance(s.index, pd.DatetimeIndex):
        try:
            s.index = pd.to_datetime(s.index)
        except (ValueError, TypeError) as exc:
            raise DataError(f"index is not parseable as dates: {exc}") from exc
    if s.index.tz is not None:
        s.index = s.index.tz_localize(None)
    s = pd.to_numeric(s, errors="coerce").astype(float)
    s = s[np.isfinite(s.to_numpy())]
    s = s[~s.index.isna()]
    s = s.sort_index()
    if s.empty:
        raise DataError("series has no finite values")
    return s


def prepare(series: pd.Series, freq: Optional[str] = None, name: Optional[str] = None,
            how: str = "mean") -> tuple[pd.Series, str]:
    """Clean + regularise a raw series. Returns ``(series, freq)``."""
    s = _clean(series)
    if len(s) < 2:
        raise DataError("need at least 2 observations")
    if freq is None:
        freq = infer_freq(s.index)
    out = regularize(s, freq, how=how)
    if name is not None:
        out.name = name
    elif out.name is None:
        out.name = "series"
    if len(out) < 2:
        raise DataError("need at least 2 observations after regularisation")
    return out, freq


# --------------------------------------------------------------------------- CSV / Excel


def _looks_like_dates(col: pd.Series) -> bool:
    if pd.api.types.is_datetime64_any_dtype(col):
        return True
    if pd.api.types.is_numeric_dtype(col):
        # plain years like 1990, 1991, ... are dates; other numbers are values
        vals = col.dropna()
        return bool(len(vals) > 0 and ((vals % 1 == 0) & (vals >= 1000) & (vals <= 2200)).all()
                    and vals.is_monotonic_increasing and vals.is_unique)
    parsed = pd.to_datetime(col.astype(str), errors="coerce", format="mixed")
    return parsed.notna().mean() >= 0.9


def _to_datetime(col: pd.Series) -> pd.DatetimeIndex:
    if pd.api.types.is_numeric_dtype(col) and not pd.api.types.is_datetime64_any_dtype(col):
        return pd.DatetimeIndex(pd.to_datetime(col.astype(int).astype(str), format="%Y"))
    return pd.DatetimeIndex(pd.to_datetime(col.astype(str), errors="coerce", format="mixed"))


def frame_to_series(df: pd.DataFrame, date_col: Optional[str] = None,
                    value_cols: Optional[Iterable[str]] = None) -> dict[str, pd.Series]:
    """Split a wide DataFrame (one date column + N value columns) into raw series."""
    if df.empty:
        raise DataError("table is empty")
    if date_col is None:
        for c in df.columns:
            if _looks_like_dates(df[c]):
                date_col = c
                break
        if date_col is None:
            raise DataError("could not find a date column; pass date_col explicitly")
    if date_col not in df.columns:
        raise DataError(f"date column {date_col!r} not found; columns: {list(df.columns)}")
    dates = _to_datetime(df[date_col])
    if value_cols is None:
        value_cols = []
        for c in df.columns:
            if c == date_col:
                continue
            num = pd.to_numeric(df[c], errors="coerce")
            if num.notna().mean() >= 0.5:
                value_cols.append(c)
        if not value_cols:
            raise DataError("no numeric value column found")
    out: dict[str, pd.Series] = {}
    for c in value_cols:
        if c not in df.columns:
            raise DataError(f"value column {c!r} not found; columns: {list(df.columns)}")
        s = pd.Series(pd.to_numeric(df[c], errors="coerce").to_numpy(), index=dates, name=str(c))
        s = s[~s.index.isna()]
        out[str(c)] = s
    return out


def load_table(path: Union[str, Path], date_col: Optional[str] = None,
               value_cols: Optional[Iterable[str]] = None) -> dict[str, pd.Series]:
    """Load a CSV/TSV/Excel/JSON file into ``{name: raw series}``.

    The date column and the numeric value columns are detected automatically
    unless given explicitly.  Returned series are *raw*; pass them through
    :func:`prepare` before analysis.
    """
    p = Path(path)
    if not p.exists():
        raise DataError(f"file not found: {p}")
    suffix = p.suffix.lower()
    if suffix in (".xlsx", ".xls"):
        df = pd.read_excel(p)
    elif suffix == ".json":
        df = pd.read_json(p)
    else:
        df = pd.read_csv(p, sep=None, engine="python", encoding_errors="replace")
    return frame_to_series(df, date_col=date_col, value_cols=value_cols)


def from_values(values: Iterable[float], start: str = "2020-01-01", freq: str = "M",
                name: str = "series") -> pd.Series:
    """Build a regular series from a plain list of numbers."""
    vals = np.asarray(list(values), dtype=float)
    idx = pd.date_range(start=start, periods=len(vals), freq=PANDAS_FREQ[_check_freq(freq)])
    return pd.Series(vals, index=idx, name=name)
