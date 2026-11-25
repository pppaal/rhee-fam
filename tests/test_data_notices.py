"""Thread-safe notice capture (collect_notices / emit_notice), the date_cols attr and unique series names
(docs/WEB_SPEC.md B.3.1, E.13, H.2)."""

from __future__ import annotations

import inspect
import linecache
import threading
import warnings

import numpy as np
import pandas as pd
import pytest

from trendai import data
from trendai.data import (
    DataNotice, DataWarning, collect_notices, emit_notice, frame_to_series, load_table_bytes, prepare,
)
from trendai.messages import Msg, msg


def _gappy(name: str, n_gaps: int, n: int = 15) -> pd.Series:
    """A monthly series with ``n_gaps`` (3..11) empty months inside (-> one gaps_interpolated warning:
    at least 3 filled periods and 20 % of the grid)."""
    idx = pd.date_range("2020-01-01", periods=n, freq="MS")
    s = pd.Series(100 + np.arange(float(n)), index=idx, name=name)
    return s.drop(idx[2:2 + n_gaps])


def test_collect_notices_captures_and_suppresses():
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        with collect_notices() as notes:
            prepare(_gappy("a", 10))
    assert seen == []                                   # warnings saw nothing
    assert len(notes) == 1
    w = notes[0]
    assert isinstance(w, DataWarning) and isinstance(w.args[0], Msg)
    assert w.code == "gaps_interpolated" and w.params["filled"] == 10 and w.params["name"] == "'a': "
    assert str(w).startswith("'a': 월 단위 격자에 맞추면서 빈 기간 10개(전체 15개 중 67%)")
    # outside a block: warnings as before
    with pytest.warns(DataWarning, match="빈 기간 10개"):
        prepare(_gappy("a", 10))
    # an exception inside the block still restores the previous state
    with pytest.raises(data.DataError):
        with collect_notices():
            prepare(pd.Series([1.0], index=pd.to_datetime(["2020-01-01"])))
    with pytest.warns(DataWarning):
        prepare(_gappy("a", 10))


def test_collect_notices_nested():
    with collect_notices() as outer:
        emit_notice(msg("series_empty"), DataNotice)
        with collect_notices() as inner:
            prepare(_gappy("inner", 7))
        prepare(_gappy("outer", 9))
    assert [n.code for n in outer] == ["series_empty", "gaps_interpolated"]
    assert isinstance(outer[0], DataNotice) and outer[1].params["filled"] == 9
    assert [n.params["filled"] for n in inner] == [7]


def test_collect_notices_thread_isolation():
    filters_before = list(warnings.filters)
    n_threads, rounds = 8, 5
    barrier = threading.Barrier(n_threads)
    results: dict[int, list] = {}
    errors: list[BaseException] = []

    def worker(k: int) -> None:
        try:
            barrier.wait(10)
            series = _gappy(f"t{k}", 3 + k)      # distinct gap counts: 3..10
            if k % 2 == 0:
                with collect_notices() as notes:
                    for _ in range(rounds):
                        prepare(series)
                results[k] = notes
            else:                                 # no block: plain warnings, never into another list
                for _ in range(rounds):
                    prepare(series)
        except BaseException as exc:  # noqa: BLE001 - reported below
            errors.append(exc)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")           # the unblocked threads' warnings
        threads = [threading.Thread(target=worker, args=(k,)) for k in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(60)
    assert not errors
    assert sorted(results) == [0, 2, 4, 6]
    for k, notes in results.items():
        assert len(notes) == rounds
        assert all(n.code == "gaps_interpolated" and n.params["name"] == f"'t{k}': "
                   and n.params["filled"] == 3 + k for n in notes), k
    assert list(warnings.filters) == filters_before


def _emit_from_helper() -> None:
    emit_notice(msg("series_empty"), DataNotice)      # default stacklevel=2: reported at the helper's caller


def test_emit_notice_stacklevel():
    with pytest.warns(DataNotice) as rec:
        line = inspect.currentframe().f_lineno + 1
        _emit_from_helper()
    assert rec[0].filename == __file__ and rec[0].lineno == line
    assert rec[0].category is DataNotice and rec[0].message.code == "series_empty"
    with pytest.warns(DataWarning) as rec:
        line = inspect.currentframe().f_lineno + 1
        emit_notice(msg("series_empty"), DataWarning, stacklevel=1)
    assert rec[0].filename == __file__ and rec[0].lineno == line
    # the data layer's own warnings point where they always did: the caller of the reporting helper
    # (prepare -> _report_regularisation -> _warn -> emit_notice)
    with pytest.warns(DataWarning) as rec:
        prepare(_gappy("a", 10))
    assert rec[0].filename == data.__file__
    assert "_report_regularisation(" in linecache.getline(rec[0].filename, rec[0].lineno)


def test_date_cols_attr():
    long = pd.DataFrame({"연도": [2020] * 12 + [2021] * 12, "월": list(range(1, 13)) * 2, "값": np.arange(24.0)})
    (s,) = frame_to_series(long).values()
    assert s.attrs["trendai"]["date_cols"] == ["연도", "월"] and s.attrs["trendai"]["date_col"] == "연도 + 월"
    single = pd.DataFrame({"날짜": [f"2024-{m:02d}" for m in range(1, 13)], "v": np.arange(12.0)})
    (s,) = frame_to_series(single).values()
    assert s.attrs["trendai"]["date_cols"] == ["날짜"] and s.attrs["trendai"]["date_col"] == "날짜"
    wide = pd.DataFrame({"항목": ["매출"], **{f"2021.{m:02d}": [100 + m] for m in range(1, 13)}})
    with collect_notices():
        (s,) = frame_to_series(wide).values()
    assert s.attrs["trendai"]["date_cols"] == [] and s.attrs["trendai"]["date_col"] == "(열 머리글)"
    # an explicitly named date column of a year + month table
    (s,) = frame_to_series(long, date_col="연도").values()
    assert s.attrs["trendai"]["date_cols"] == ["연도", "월"]


def test_wide_duplicate_names():
    text = "항목," + ",".join(f"2021.{m:02d}" for m in range(1, 13)) + "\n" + "".join(
        "A," + ",".join(str(base + m) for m in range(12)) + "\n" for base in (100, 200, 300))
    with collect_notices():
        out = load_table_bytes(text.encode(), "wide.csv")
    assert list(out) == ["A", "A (2)", "A (3)"]
    assert [s.name for s in out.values()] == ["A", "A (2)", "A (3)"]
    assert [s.iloc[0] for s in out.values()] == [100.0, 200.0, 300.0]
    # duplicate column labels of a long table (e.g. 'a' and 'a ' after whitespace clean-up) stay apart
    df = pd.DataFrame([[f"2024-{m:02d}", (m * 7) % 5, 10 * m * m] for m in range(1, 13)],
                      columns=["date", "a", "a "])
    out = frame_to_series(df)
    assert list(out) == ["a", "a (2)"] and out["a (2)"].iloc[1] == 40.0 and out["a"].iloc[1] == 4.0
