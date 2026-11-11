"""Data loading / normalisation edge cases (real-world CSV, Excel and JSON exports)."""

from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
import pytest

from trendai.data import (
    DataError, DataNotice, DataWarning, future_index, load_table, parse_dates, parse_numbers,
    prepare,
)
from trendai.samples import list_samples, sample

MONTHLY = 1000 + np.arange(36) * 5 + np.tile([0, -50, 20, 10, 0, 30, 0, 10, -10, 20, 80, 300], 3)


def _write(path, text, encoding="utf-8"):
    path.write_text(text, encoding=encoding)
    return path


def _only(d: dict) -> pd.Series:
    assert len(d) == 1, list(d)
    return next(iter(d.values()))


def _prep(s: pd.Series):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return prepare(s)


# --------------------------------------------------------------------------- date formats


@pytest.mark.parametrize("fmt", [
    lambda y, m: f"{y}.{m:02d}",            # KOSIS: read_csv would turn 2021.10 into 2021.1
    lambda y, m: f"{y}{m:02d}",             # ECOS yyyymm
    lambda y, m: f"{y}년 {m}월",
    lambda y, m: f"{y}-{m:02d}",
    lambda y, m: f"{y}/{m}",
    lambda y, m: f"{y}. {m}. 1.",           # Korean Excel display format
    lambda y, m: f"\"{y}.{m:02d}\"",
])
def test_monthly_formats_load_as_36_monthly_points(tmp_path, fmt):
    rows = "\n".join(f"{fmt(2021 + i // 12, i % 12 + 1)},{v}" for i, v in enumerate(MONTHLY))
    s = _only(load_table(_write(tmp_path / "m.csv", "날짜,매출액\n" + rows + "\n")))
    p, f = _prep(s)
    assert f == "M" and len(p) == 36
    assert p.index[0] == pd.Timestamp("2021-01-01") and p.index[-1] == pd.Timestamp("2023-12-01")
    assert p.loc["2021-10-01"] == MONTHLY[9] and p.loc["2021-01-01"] == MONTHLY[0]


def test_dotted_months_with_explicit_date_col_are_not_collapsed_to_years(tmp_path):
    rows = "\n".join(f"{2021 + i // 12}.{i % 12 + 1:02d},{v}" for i, v in enumerate(MONTHLY))
    f = _write(tmp_path / "k.csv", "날짜,매출액\n" + rows + "\n")
    p, freq = _prep(_only(load_table(f, date_col="날짜")))
    assert freq == "M" and len(p) == 36   # was: yearly, 3 points (2016.01..2016.12 -> 2016)


def test_numeric_non_integer_date_column_is_never_truncated_to_years():
    df = pd.DataFrame({"t": [1.5, 2.25, 3.75, 4.5], "v": [1.0, 2.0, 3.0, 4.0]})
    from trendai.data import frame_to_series

    with pytest.raises(DataError, match="날짜로 읽을 수 없습니다"):
        frame_to_series(df, date_col="t")


def test_excel_style_float_months_are_read_as_months():
    # 2021.1 is how a numeric KOSIS cell "2021.10" (October) arrives from Excel
    from trendai.data import frame_to_series

    vals = [2021 + m / 100 for m in range(1, 13)]
    df = pd.DataFrame({"시점": vals, "v": np.arange(12.0)})
    s = _only(frame_to_series(df))
    assert list(s.index.month) == list(range(1, 13))


@pytest.mark.parametrize("header,cells,freq,n", [
    ("기간", [f"{2020 + i // 4}년 {i % 4 + 1}분기" for i in range(12)], "Q", 12),
    ("period", [f"{2020 + i // 4} Q{i % 4 + 1}" for i in range(12)], "Q", 12),
    ("period", [f"{2020 + i // 4}Q{i % 4 + 1}" for i in range(12)], "Q", 12),
    ("기간", [f"{2020 + i // 4} {i % 4 + 1}/4분기" for i in range(12)], "Q", 12),
    ("week", [f"2024-W{i + 1:02d}" for i in range(20)], "W", 20),
    ("dt", [f"{d:%Y%m%d}" for d in pd.date_range("2024-01-01", periods=40)], "D", 40),
    ("timestamp", [str(1704067200 + 86400 * i) for i in range(30)], "D", 30),
    ("timestamp", [str((1704067200 + 86400 * i) * 1000) for i in range(30)], "D", 30),
    ("year", [str(2020 - i) for i in range(12)], "Y", 12),                       # newest first
    ("m", [f"{d:%y-%m}" for d in pd.date_range("2020-01-01", periods=30, freq="MS")], "M", 30),
    ("일자", [f"{d:%Y-%m-%d} ({'월화수목금토일'[d.dayofweek]})"
              for d in pd.date_range("2024-01-01", periods=20)], "D", 20),
    ("date", [f"{d:%b %Y}" for d in pd.date_range("2020-01-01", periods=24, freq="MS")], "M", 24),
])
def test_other_date_formats(tmp_path, header, cells, freq, n):
    text = f"{header},v\n" + "".join(f"{c},{i + 1}\n" for i, c in enumerate(cells))
    p, f = _prep(_only(load_table(_write(tmp_path / "d.csv", text))))
    assert f == freq and len(p) == n


def test_year_column_with_duplicates_and_blank_trailing_row(tmp_path):
    f = _write(tmp_path / "y.csv", "year,val\n1990,1\n1991,2\n1991,4\n1992,3\n1993,5\n,\n")
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        p, freq = prepare(_only(load_table(f)))
    assert freq == "Y" and list(p.index.year) == [1990, 1991, 1992, 1993]
    assert p.loc["1991-01-01"] == 3.0                      # duplicates averaged ...
    assert any(issubclass(w.category, DataWarning) and "합쳤습니다" in str(w.message) for w in rec)


def test_day_first_monthly_dates(tmp_path):
    idx = pd.date_range("2022-01-01", periods=36, freq="MS")
    text = "Datum,Wert\n" + "".join(f"{d:%d.%m.%Y},{100 + i}\n" for i, d in enumerate(idx))
    p, f = _prep(_only(load_table(_write(tmp_path / "eu.csv", text))))
    assert f == "M" and len(p) == 36                        # was: daily, 742 fabricated points
    assert p.index[1] == pd.Timestamp("2022-02-01") and p.iloc[1] == 101


def test_day_first_daily_dates_keep_their_order(tmp_path):
    dd = pd.date_range("2023-01-01", periods=200)
    text = "date,v\n" + "".join(f"{x:%d/%m/%Y},{i}\n" for i, x in enumerate(dd))
    s = _only(load_table(_write(tmp_path / "d.csv", text)))
    assert s.index.equals(pd.DatetimeIndex(dd))
    p, f = _prep(s)
    assert f == "D" and len(p) == 200 and np.allclose(p.to_numpy(), np.arange(200))


def test_month_first_us_dates(tmp_path):
    dd = pd.date_range("2023-01-01", periods=60)
    text = "date,v\n" + "".join(f"{x:%m/%d/%Y},{i}\n" for i, x in enumerate(dd))
    assert _only(load_table(_write(tmp_path / "us.csv", text))).index.equals(pd.DatetimeIndex(dd))


def test_parse_dates_reports_unreadable_cells():
    col = pd.Series(["2024-01", "2024-02", "2024-03", "2024-04", "합계"])
    parsed, label, note = parse_dates(col)
    assert label == "YYYY-MM" and parsed.isna().sum() == 1
    assert "합계" in note


def test_timestamps_with_dst_offsets_keep_local_dates(tmp_path):
    idx = pd.date_range("2023-01-01", periods=24, freq="MS", tz="Europe/Berlin")
    pd.DataFrame({"date": [d.isoformat() for d in idx], "val": range(24)}).to_csv(tmp_path / "dst.csv",
                                                                                  index=False)
    p, f = _prep(_only(load_table(tmp_path / "dst.csv")))
    assert f == "M" and len(p) == 24 and p.index[3] == pd.Timestamp("2023-04-01") and p.iloc[3] == 3


# --------------------------------------------------------------------------- numbers


def test_thousands_separators_do_not_shorten_the_series(tmp_path):
    idx = pd.date_range("2016-01-01", periods=120, freq="MS")
    v = np.linspace(300, 1500, 120)
    text = "date,sales\n" + "".join(f'{d:%Y-%m-%d},"{x:,.0f}"\n' for d, x in zip(idx, v))
    s = load_table(_write(tmp_path / "t.csv", text))["sales"]
    p, _ = _prep(s)
    assert len(p) == 120 and p.index[-1] == pd.Timestamp("2025-12-01")   # was 70 points ending 2021-10
    assert p.iloc[-1] == 1500


@pytest.mark.parametrize("cells,expected", [
    (["12%", "13%", "15%"], [12, 13, 15]),
    (['"$1,234"', '"$2,000"', '"$3,500.5"'], [1234, 2000, 3500.5]),
    (['"1,234원"', '"2,000원"', '"35,000원"'], [1234, 2000, 35000]),
    (['"(1,234)"', "500", '"-2,000"'], [-1234, 500, -2000]),
    (["1.5억", "2억", '"3,400만"'], [1.5e8, 2e8, 3.4e7]),
    (["12K", "1.5M", "2B"], [12e3, 1.5e6, 2e9]),
    (["1 234", "2 500", "3 000"], [1234, 2500, 3000]),
])
def test_formatted_numbers(tmp_path, cells, expected):
    text = "date,v\n" + "".join(f"2024-{i + 1:02d}-01,{c}\n" for i, c in enumerate(cells))
    s = _only(load_table(_write(tmp_path / "n.csv", text)))
    assert np.allclose(s.to_numpy(), expected)


def test_semicolon_csv_with_decimal_commas(tmp_path):
    text = "Datum;Wert\n" + "".join(f"01.{m:02d}.2022;1.{m:03d},5\n" for m in range(1, 13))
    s = _only(load_table(_write(tmp_path / "semi.csv", text)))
    assert np.allclose(s.to_numpy(), [1000 + m + 0.5 for m in range(1, 13)])
    assert list(s.index.month) == list(range(1, 13))


def test_parse_numbers_detects_decimal_comma_per_column():
    vals, bad, _ = parse_numbers(pd.Series(["1.234,5", "12,25", "7"]))
    assert bad == 0 and np.allclose(vals, [1234.5, 12.25, 7])
    vals, bad, _ = parse_numbers(pd.Series(["1,234.5", "12.25", "7"]))
    assert bad == 0 and np.allclose(vals, [1234.5, 12.25, 7])


def test_missing_markers_are_gaps_not_errors(tmp_path):
    text = "date,v\n2024-01,10\n2024-02,-\n2024-03,N/A\n2024-04,13\n"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        s = _only(load_table(_write(tmp_path / "h.csv", text)))
    assert s.isna().sum() == 2


def test_unreadable_values_are_reported(tmp_path):
    text = "date,v\n" + "".join(f"2024-{m:02d},{m}\n" for m in range(1, 12)) + "2024-12,abc\n"
    with pytest.warns(DataWarning, match="abc"):
        s = _only(load_table(_write(tmp_path / "bad.csv", text)))
    assert s.isna().sum() == 1


# --------------------------------------------------------------------------- encodings & files


def test_cp949_headers_are_decoded(tmp_path):
    f = tmp_path / "k.csv"
    f.write_bytes(("날짜,검색량\n" + "".join(f"2020-{m:02d}-01,{m * 10}\n" for m in range(1, 13)))
                  .encode("cp949"))
    assert list(load_table(f)) == ["검색량"]
    assert list(load_table(f, value_cols=["검색량"])) == ["검색량"]


def test_utf16_tab_separated_export(tmp_path):
    g = sample("growing")
    f = tmp_path / "u16.txt"
    _write(f, "날짜\t검색량\n" + "".join(f"{d:%Y-%m-%d}\t{v:.2f}\n" for d, v in g.items()), "utf-16")
    s = load_table(f)["검색량"]
    assert len(s) == 120


@pytest.mark.parametrize("kind", ["series", "records_bom", "columns", "split"])
def test_json_shapes(tmp_path, kind):
    g = sample("growing")
    f = tmp_path / "s.json"
    if kind == "series":
        g.to_json(f)
    elif kind == "records_bom":
        txt = pd.DataFrame({"date": g.index.strftime("%Y-%m-%d"), "v": g.values}).to_json(orient="records")
        _write(f, txt, "utf-8-sig")
    elif kind == "columns":
        g.to_frame("v").to_json(f)
    else:
        g.to_frame("v").to_json(f, orient="split")
    p, freq = _prep(_only(load_table(f)))
    assert freq == "M" and len(p) == 120 and np.isclose(p.iloc[-1], g.iloc[-1], rtol=1e-6)


@pytest.mark.parametrize("make,needle", [
    (lambda d: (d / "empty.csv").write_text(""), "비어 있습니다"),
    (lambda d: (d / "d.csv").mkdir(), "폴더"),
    (lambda d: (d / "bad.json").write_text("{not json"), "JSON"),
    (lambda d: (d / "nd.csv").write_text("a,b\nx,2\ny,4\n"), "--date-col"),
])
def test_unusable_files_raise_korean_data_errors(tmp_path, make, needle):
    make(tmp_path)
    target = next(tmp_path.iterdir())
    with pytest.raises(DataError, match=needle):
        load_table(target)


def test_missing_file():
    with pytest.raises(DataError, match="찾을 수 없습니다"):
        load_table("/nonexistent/x.csv")


def test_csv_with_excel_extension_is_read_as_text(tmp_path):
    f = _write(tmp_path / "fake.xlsx", "date,v\n" + "".join(f"2024-{m:02d}-01,{m}\n" for m in range(1, 13)))
    with pytest.warns(DataNotice):
        assert len(_only(load_table(f))) == 12


def test_old_xls_without_engine_gives_install_hint(tmp_path):
    pytest.importorskip("openpyxl")  # only the hint for the missing xls engine is tested
    try:
        import xlrd  # noqa: F401
        pytest.skip("xlrd installed")
    except ImportError:
        pass
    f = tmp_path / "o.xls"
    f.write_bytes(bytes.fromhex("D0CF11E0A1B11AE1") + b"\0" * 1000)
    with pytest.raises(DataError, match="xlrd"):
        load_table(f)


def test_headerless_csv(tmp_path):
    f = _write(tmp_path / "nh.csv", "".join(f"2024-{m:02d}-01,{m * 3}\n" for m in range(1, 13)))
    with pytest.warns(DataNotice):
        s = _only(load_table(f))
    assert len(s) == 12 and s.iloc[0] == 3


def test_wide_table_dates_across_columns(tmp_path):
    head = "항목,단위," + ",".join(f"2021.{m:02d}" for m in range(1, 13))
    rows = ["매출,백만원," + ",".join(str(100 + m) for m in range(12)),
            "비용,백만원," + ",".join(str(50 + m) for m in range(12))]
    with pytest.warns(DataNotice):
        d = load_table(_write(tmp_path / "w.csv", "\n".join([head, *rows]) + "\n"))
    assert set(d) == {"매출 / 백만원", "비용 / 백만원"}
    assert list(d["매출 / 백만원"].index.month) == list(range(1, 13))


# --------------------------------------------------------------------------- columns


def test_row_number_column_is_ignored(tmp_path):
    g = sample("growing")
    pd.DataFrame({"date": g.index.strftime("%Y-%m-%d"), "sales": g.round(2).values}).to_csv(tmp_path / "wi.csv")
    with pytest.warns(DataNotice, match="Unnamed: 0"):
        d = load_table(tmp_path / "wi.csv")
    assert list(d) == ["sales"]


def test_id_and_step_sequences_are_ignored_but_explicit_choice_wins(tmp_path):
    text = "id,date,no_name,v\n" + "".join(f"{i + 1},2024-{i + 1:02d},{i},{(i * 7) % 5}\n" for i in range(12))
    f = _write(tmp_path / "ids.csv", text)
    with pytest.warns(DataNotice):
        assert list(load_table(f)) == ["v"]
    assert list(load_table(f, value_cols=["id"])) == ["id"]


def test_attrs_describe_detected_format(tmp_path):
    rows = "\n".join(f"{2021 + i // 12}.{i % 12 + 1:02d},{v}" for i, v in enumerate(MONTHLY))
    s = _only(load_table(_write(tmp_path / "m.csv", "시점,값\n" + rows + "\n")))
    assert s.attrs["trendai"]["date_col"] == "시점" and s.attrs["trendai"]["date_format"] == "YYYY.MM"


# --------------------------------------------------------------------------- series / prepare


def test_weekly_anchor_is_kept():
    idx = pd.date_range("2022-01-03", periods=104, freq="W-MON")
    p, f = prepare(pd.Series(sample("growing", n=104, freq="W").values, index=idx))
    assert f == "W" and p.index[-1] == idx[-1]                      # was shifted to Sunday 2023-12-31
    nxt = future_index(p.index[-1], "W", 2)
    assert list(nxt) == [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-08")]
    assert (nxt.dayofweek == 0).all()


def test_business_day_series_warns_with_real_counts():
    rng = np.random.default_rng(0)
    s = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.018, 504))),
                  index=pd.bdate_range("2023-01-02", periods=504))
    with pytest.warns(DataWarning, match="실제 관측 504개") as rec:
        p, f = prepare(s)
    assert f == "D" and len(p) == 704
    assert "주말 200개" in str(rec[0].message)


@pytest.mark.parametrize("index,freq,n", [
    (list(range(1990, 2020)), "Y", 30),
    (pd.period_range("2020-01", periods=36, freq="M"), "M", 36),
    (pd.period_range("2020Q1", periods=12, freq="Q"), "Q", 12),
    ([f"2024-{m:02d}" for m in range(1, 13)], "M", 12),
])
def test_series_index_kinds(index, freq, n):
    s = pd.Series(np.arange(float(len(index))) + 10, index=index)
    p, f = prepare(s)
    assert f == freq and len(p) == n


def test_range_index_gives_helpful_error():
    with pytest.raises(DataError, match="from_values"):
        prepare(pd.Series(np.arange(50.0)))


def test_string_values_in_series():
    s = pd.Series(["1,000", "2,000", "3,500"], index=pd.date_range("2024-01-01", periods=3, freq="MS"))
    p, _ = prepare(s)
    assert list(p) == [1000.0, 2000.0, 3500.0]


def test_too_short_series_message_is_korean():
    with pytest.raises(DataError, match="2개 이상"):
        prepare(pd.Series([1.0], index=pd.to_datetime(["2020-01-01"])))


def test_clean_regular_samples_emit_no_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for nm in list_samples():
            for freq, n in (("M", 120), ("W", 156), ("D", 400), ("Q", 40), ("Y", 25)):
                prepare(sample(nm, n=n, freq=freq))


def test_load_table_is_fast_for_large_files(tmp_path):
    import time

    idx = pd.date_range("2000-01-01", periods=20000, freq="D")
    rng = np.random.default_rng(1)
    pd.DataFrame({"date": idx.strftime("%Y-%m-%d"), "a": rng.normal(100, 5, 20000).round(2),
                  "b": rng.normal(50, 3, 20000).round(2)}).to_csv(tmp_path / "big.csv", index=False)
    t0 = time.perf_counter()
    d = load_table(tmp_path / "big.csv")
    assert time.perf_counter() - t0 < 5.0
    assert set(d) == {"a", "b"} and len(d["a"]) == 20000


def test_json_records_from_dataframe_with_datetime_column(tmp_path):
    g = sample("growing")
    f = tmp_path / "r.json"
    ms = (g.index - pd.Timestamp("1970-01-01")) // pd.Timedelta(milliseconds=1)
    f.write_text(json.dumps([{"date": int(d), "v": float(v)} for d, v in zip(ms, g.values)]))
    assert json.loads(f.read_text())[0]["date"] > 1e11  # epoch milliseconds (pandas' old default)
    p, freq = _prep(_only(load_table(f)))
    assert freq == "M" and len(p) == 120


@pytest.mark.parametrize("text,freq,n", [
    ("연도,월,값\n" + "".join(f"{2020 + i // 12},{i % 12 + 1},{100 + i}\n" for i in range(36)), "M", 36),
    ("year,quarter,v\n" + "".join(f"{2020 + i // 4},Q{i % 4 + 1},{100 + i}\n" for i in range(12)), "Q", 12),
])
def test_separate_year_and_period_columns(tmp_path, text, freq, n):
    d = load_table(_write(tmp_path / "long.csv", text))
    assert len(d) == 1                                   # the month/quarter column is not a value column
    p, f = _prep(_only(d))
    assert f == freq and len(p) == n


def test_excel_serial_dates_in_date_named_column():
    from trendai.data import frame_to_series

    df = pd.DataFrame({"date": [45292 + 31 * i for i in range(12)], "v": np.arange(12.0)})
    s = _only(frame_to_series(df))
    assert s.index[0] == pd.Timestamp("2024-01-01")
    with pytest.raises(DataError):  # an unnamed 5-digit column is a value column, not dates
        frame_to_series(df.rename(columns={"date": "x"}))


def test_xlsx_roundtrip(tmp_path):
    pytest.importorskip("openpyxl")
    g = sample("growing")
    f = tmp_path / "d.xlsx"
    pd.DataFrame({"날짜": g.index, "검색량": g.values}).to_excel(f, index=False)
    p, freq = _prep(load_table(f)["검색량"])
    assert freq == "M" and len(p) == 120


def test_frequency_is_inferred_from_every_dated_row():
    """F13: a complete monthly date column with 27 of 36 values missing was read as quarterly
    averages ('분기 단위 12개'); its spacing says monthly."""
    import warnings

    from trendai.data import DataWarning, prepare

    idx = pd.date_range("2021-01-01", periods=36, freq="MS")
    keep = [1, 2, 9, 13, 14, 16, 21, 31, 35]
    v = pd.Series(np.nan, index=idx)
    v.iloc[keep] = 100 + np.arange(9.0)
    with pytest.warns(DataWarning, match="월 단위 격자"):
        s, freq = prepare(v)
    assert freq == "M" and len(s) == 35 and s.index[0] == idx[1]
    # values that form a complete coarser series keep their own frequency (month starts in a
    # daily calendar are monthly data, not 97 % interpolated daily data)
    d = pd.date_range("2020-01-01", "2022-12-31", freq="D")
    cal = pd.Series(np.nan, index=d)
    cal[d[d.day == 1]] = np.arange(36.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        s2, f2 = prepare(cal)
    assert f2 == "M" and len(s2) == 36


def test_a_few_empty_periods_keep_the_values_own_frequency():
    """R4-REV-1: Monday values in a full daily calendar with 2 of 104 weeks empty were read as
    daily data (722 points, 620 interpolated, a false 'weekday-only' warning), and month-start
    values with 1 of 24 months empty as 97 % interpolated daily data.  A few empty periods of
    a weekly / monthly series are gaps of that series."""
    import warnings

    from trendai.data import DataWarning, prepare

    days = pd.date_range("2022-01-01", "2023-12-31", freq="D")
    mon = days[days.dayofweek == 0]
    weekly = pd.Series(np.nan, index=days)
    weekly[mon] = 100 + np.cumsum(np.random.default_rng(0).normal(0, 1, len(mon)))
    weekly[mon[[10, 60]]] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("error", DataWarning)
        s, freq = prepare(weekly)
    assert freq == "W" and len(s) == 104

    monthly = pd.Series(np.nan, index=days)
    starts = days[days.day == 1]
    monthly[starts] = 100 + np.arange(len(starts), dtype=float)
    monthly[starts[7]] = np.nan
    s2, f2 = prepare(monthly)
    assert f2 == "M" and len(s2) == 24
    # the gap is filled on the monthly grid (linear interpolation), not on a daily one
    assert s2.iloc[7] == pytest.approx(107.0)

    # sparse values still follow the date column: 7 of 12 quarters (58 %) is no quarterly series
    idx = pd.date_range("2021-01-01", periods=36, freq="MS")
    v = pd.Series(np.nan, index=idx)
    v.iloc[[1, 2, 9, 13, 14, 16, 21, 31, 35]] = 100 + np.arange(9.0)
    with pytest.warns(DataWarning):
        _, f3 = prepare(v)
    assert f3 == "M"


# --------------------------------------------------------------------------- round 5: message catalog

import re as _re

_HANGUL = _re.compile(r"[가-힣]")


def test_message_catalog_renders_every_code_in_both_languages():
    """G5: every data-layer message has a Korean and an English template; English has no Hangul."""
    from trendai import messages as MS

    sample_params = {
        "freq": "X", "value": 0, "horizon": 20_000, "max": 10_000, "n": 3, "ratio": 5, "series": "s",
        "detail": MS.msg("too_few_obs", name="", n=1), "label": MS.L("YYYY (연도)", "YYYY (year)"),
        "examples": "'a'", "lo": "1", "hi": "36", "name": "'v': ", "n_obs": 60, "filled": 22, "n_out": 82,
        "unit": MS.unit("M"), "total": 47, "share": 0.32, "observed": 32, "how": MS.L("평균으로", "averaged"), "own": MS.unit("M"),
        "n_raw": 25, "col": "v", "columns": ["date", "v"], "cols": ["id"], "edge": "", "what": "f.csv",
        "error": "boom", "hint": "", "line": 3, "pkg": "openpyxl", "path": "/x/f.csv",
        # connectors.ConnectorError (R5-REV-5)
        "lang": "'k o'", "choices": "('daily', 'monthly')", "start": "2024-05-01", "end": "2024-01-01",
        "status": 503, "url": "https://example.org/x", "title": "Foo", "kind": "ImportError", "kw": "foo",
    }
    for code, (ko, en) in MS.MESSAGES.items():
        names = {f for t in (ko, en) for _, f, _, _ in __import__("string").Formatter().parse(t) if f}
        params = {k: sample_params[k] for k in names}
        m = MS.msg(code, **params)
        assert str(m) == MS.render(code, params, "ko") and _HANGUL.search(str(m)), code
        assert m.text("en") == MS.render(code, params, "en") and not _HANGUL.search(m.text("en")), code
        assert m.code == code and m.params == params
    assert MS.format_label("YYYY년 M월", "en") == "Korean year-month text"
    assert MS.format_label("(열 머리글)", "en") == "(column headers)"
    assert not _HANGUL.search(" ".join(MS.format_label(k, "en") for k, _ in MS._LABEL_EN))


def test_data_errors_and_warnings_carry_code_params_and_english_text(tmp_path):
    """G5: DataError / DataWarning / DataNotice keep their Korean str() for library users and carry
    the catalog code + params, so the CLI can render them in English."""
    from trendai.data import DataError, DataNotice, DataWarning, load_table, prepare

    with pytest.raises(DataError) as ei:
        prepare(pd.Series([1.0], index=pd.date_range("2024-01-01", periods=1), name="v"))
    e = ei.value
    assert str(e) == "'v': 관측치가 2개 이상 필요합니다 (유효한 값 1개)."      # unchanged Korean text
    assert e.code == "too_few_obs" and e.params == {"name": "'v': ", "n": 1}
    assert e.text("en") == "'v': at least 2 observations are needed (1 valid value(s))."
    with pytest.raises(DataError) as ei:
        load_table(tmp_path / "nope.csv")
    assert ei.value.code == "file_not_found" and ei.value.text("en").startswith("File not found: ")
    # warnings: the instance warnings.warn creates carries the message
    idx = pd.bdate_range("2024-01-01", periods=60)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        prepare(pd.Series(np.arange(60.0) + 100, index=idx, name="v"))
        prepare(pd.Series([1.0, 2.0, 3.0, 4.0], index=pd.to_datetime(
            ["2024-01-01", "2024-01-01", "2024-02-01", "2024-03-01"]), name="d"))
    codes = {x.message.code: x.message for x in w if isinstance(x.message, DataWarning)}
    assert {"weekday_only", "duplicates_merged"} <= set(codes)
    assert str(codes["weekday_only"]).startswith("'v': 평일(영업일)에만 값이 있는 데이터입니다.")
    assert codes["duplicates_merged"].text("en").startswith("'d': several values fell in the same month period "
                                                           "and were averaged")
    p = tmp_path / "rows.csv"
    p.write_text("id,date,value\n" + "".join(f"{i},2024-{i:02d}-01,{10 + i * 1.5}\n" for i in range(1, 13)))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        load_table(p)
    notice = next(x.message for x in w if isinstance(x.message, DataNotice))
    assert notice.code == "rownum_cols_skipped" and notice.params == {"cols": ["id"]}
    assert notice.text("en") == ("The column(s) ['id'] look like row numbers and were left out (name them with "
                                 "--value-col to include them).")
    # parse_dates' note is a catalog message too (ambiguous day/month order + unreadable rows)
    _, label, note = parse_dates(pd.Series(["01/02/2024", "03/04/2024", "05/06/2024", "07/08/2024", "foo"],
                                           dtype=object))
    assert [c for c, _ in note.parts] == ["date_order_ambiguous", "dates_unreadable_rows"]
    assert str(note).startswith("날짜의 일/월 순서가 모호해") and not _HANGUL.search(note.text("en"))


def test_forced_finer_frequency_warns_about_interpolation():
    """U3: '--freq W' on 48 monthly values gave 205 weekly points (157 interpolated) without a warning, and
    a confident accuracy and '상승 확률 >99%'; prepare reported its repairs only when it inferred the frequency."""
    from trendai.data import DataWarning

    s = pd.Series(100 + np.arange(48.0), index=pd.date_range("2021-01-01", periods=48, freq="MS"), name="search")
    with pytest.warns(DataWarning) as rec:
        out, f = prepare(s, freq="W")
    assert f == "W" and len(out) == 205
    w = [r.message for r in rec if "보간" in str(r.message)][0]
    assert "월 단위 데이터를 더 촘촘한 주 단위로 바꾸면서 157개 기간(전체 205개 중 77%)을 선형 보간했습니다" in str(w)
    assert "157 of 205 periods" in w.text("en") and not _HANGUL.search(w.text("en"))
    # the data's own frequency, or an aggregation that was asked for: no such warning
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        prepare(s, freq="M")
        prepare(s, freq="Q")
