"""In-memory table loading (read_table_bytes / load_table_bytes) and Google Trends CSV support
(docs/WEB_SPEC.md B.3.1, E.17, H.2)."""

from __future__ import annotations

import builtins
import io
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trendai import data
from trendai.data import (
    DataError, DataNotice, DataWarning, GoogleTrendsInfo, RawTable, collect_notices, load_table, load_table_bytes,
    read_table_bytes,
)
from trendai.messages import render_web_msg
from trendai.samples import sample

MONTHLY = 1000 + np.arange(36) * 5 + np.tile([0, -50, 20, 10, 0, 30, 0, 10, -10, 20, 80, 300], 3)


def _weeks(n: int, start: str = "2021-11-28") -> list[str]:
    return [f"{d:%Y-%m-%d}" for d in pd.date_range(start, periods=n, freq="W-SUN")]


def _codes(notes) -> list[str]:
    return [n.code for n in notes]


# --------------------------------------------------------------------------- Google Trends


CLASSIC = ("Category: All categories\n\nWeek,sunscreen: (South Korea),parka: (South Korea)\n"
           + "".join(f"{d},{(i * 37) % 90 + 5},{'<1' if i % 4 == 0 else i}\n" for i, d in enumerate(_weeks(12))))


def test_classic_gt_csv_with_category_preamble():
    with collect_notices() as notes:
        table = read_table_bytes(CLASSIC.encode(), "multiTimeline.csv")
    assert isinstance(table, RawTable) and table.format == "csv" and table.encoding == "utf-8"
    assert list(table.frame.columns) == ["Week", "sunscreen", "parka"]
    assert table.google_trends == GoogleTrendsInfo(
        terms={"sunscreen": ("sunscreen", "South Korea"), "parka": ("parka", "South Korea")},
        lt1_count=3, category="All categories")
    assert table.frame["parka"].tolist()[:2] == ["0.5", "1"]
    assert _codes(notes) == ["gt_detected", "gt_lt1_half"]
    assert all(isinstance(n, DataNotice) for n in notes)
    assert notes[1].params == {"n": 3}
    assert render_web_msg(notes[0]) == ("Google 트렌드 CSV로 읽었어요(검색어 ‘sunscreen’, ‘parka’, 지역 South Korea). "
                                        "값은 요청마다 0–100으로 다시 맞춰진 상대 지수예요.")
    assert render_web_msg(notes[1], "en") == "3 ‘<1’ value(s) were read as 0.5 (Google Trends' “less than 1”)."

    with collect_notices() as notes:
        out = load_table_bytes(CLASSIC.encode(), "multiTimeline.csv")
    assert list(out) == ["sunscreen", "parka"]
    parka = out["parka"]
    assert parka.iloc[0] == 0.5 and parka.iloc[4] == 0.5 and parka.iloc[1] == 1.0 and parka.notna().all()
    for name, s in out.items():
        info = s.attrs["trendai"]
        assert info["source"] == "google_trends_upload" and info["value_kind"] == "relative_index_gt"
        assert info["term"] == name and info["region"] == "South Korea"
        assert info["date_col"] == "Week" and info["date_cols"] == ["Week"]
    assert "gt_detected" in _codes(notes) and "col_values_unreadable" not in _codes(notes)
    # load_table reads the same file the same way
    prepared, freq = data.prepare(out["sunscreen"])
    assert freq == "W" and len(prepared) == 12


def test_gt_without_preamble():
    text = "Month,sunscreen: (Worldwide)\n" + "".join(f"2020-{m:02d},{m * 7}\n" for m in range(1, 13))
    with collect_notices() as notes:
        table = read_table_bytes(text.encode(), "gt.csv")
        out = load_table_bytes(text.encode(), "gt.csv")
    assert table.google_trends.category is None and table.google_trends.lt1_count == 0
    assert list(out) == ["sunscreen"] and out["sunscreen"].attrs["trendai"]["region"] == "Worldwide"
    assert "gt_lt1_half" not in _codes(notes)
    assert list(out["sunscreen"].index.month) == list(range(1, 13))


def test_gt_korean_headers():
    text = ("﻿카테고리: 모든 카테고리\n\n주,선크림: (대한민국),패딩: (대한민국)\n"
            + "".join(f"{d},{(i * 37) % 90 + 5},<1\n" for i, d in enumerate(_weeks(10))))
    with collect_notices() as notes:
        table = read_table_bytes(text.encode("utf-8"), "multiTimeline.csv")
        out = load_table_bytes(text.encode("utf-8"), "multiTimeline.csv")
    assert table.google_trends.category == "모든 카테고리"
    assert table.google_trends.terms == {"선크림": ("선크림", "대한민국"), "패딩": ("패딩", "대한민국")}
    assert table.encoding == "utf-8-sig" and list(out) == ["선크림", "패딩"]
    assert out["패딩"].tolist() == [0.5] * 10
    assert out["선크림"].attrs["trendai"]["region"] == "대한민국"
    gt = next(n for n in notes if n.code == "gt_detected")
    assert render_web_msg(gt).startswith("Google 트렌드 CSV로 읽었어요(검색어 ‘선크림’, ‘패딩’, 지역 대한민국).")
    # a quoted preamble with trailing separators (exports opened and saved again in Excel)
    resaved = '"카테고리: 모든 카테고리",,\r\n,,\r\n' + text.split("\n\n", 1)[1].replace("\n", "\r\n")
    with collect_notices():
        table = read_table_bytes(resaved.encode("utf-8"), "gt.csv")
    assert table.google_trends.category == "모든 카테고리" and list(table.frame.columns) == ["주", "선크림", "패딩"]


def test_gt_ispartial_dropped():
    rows = "".join(f"{d},{(i * 37) % 90 + 5},{'True' if i == 7 else 'False'}\n" for i, d in enumerate(_weeks(8)))
    text = "Week,sunscreen: (South Korea),isPartial\n" + rows
    with collect_notices():
        table = read_table_bytes(text.encode(), "gt.csv")
    assert list(table.frame.columns) == ["Week", "sunscreen"]
    with collect_notices():
        out = load_table_bytes(text.encode(), "gt.csv")
    assert list(out) == ["sunscreen"] and len(out["sunscreen"]) == 8
    # pytrends' interest_over_time().to_csv(): no region in the header
    py = "date,python,isPartial\n" + rows
    with collect_notices() as notes:
        out = load_table_bytes(py.encode(), "trends.csv")
    assert list(out) == ["python"] and out["python"].attrs["trendai"]["region"] is None
    assert out["python"].attrs["trendai"]["value_kind"] == "relative_index_gt"
    assert "지역 표시 없음" in render_web_msg(notes[0]) and "region not stated" in render_web_msg(notes[0], "en")


def test_gt_same_term_two_regions():
    text = "Week,sunscreen: (South Korea),sunscreen: (Japan)\n" + "".join(
        f"{d},{(i * 37) % 90 + 5},{(i * 53) % 70 + 3}\n" for i, d in enumerate(_weeks(6)))
    with collect_notices():
        table = read_table_bytes(text.encode(), "gt.csv")
        out = load_table_bytes(text.encode(), "gt.csv")
    assert list(out) == ["sunscreen (South Korea)", "sunscreen (Japan)"]
    assert table.google_trends.terms["sunscreen (Japan)"] == ("sunscreen", "Japan")
    assert out["sunscreen (Japan)"].attrs["trendai"]["term"] == "sunscreen"


def test_non_gt_file_lt1_unchanged():
    text = "date,v\n" + "".join(f"2024-{m:02d},{'<1' if m == 3 else m * 5}\n" for m in range(1, 13))
    with collect_notices() as notes:
        table = read_table_bytes(text.encode(), "plain.csv")
        out = load_table_bytes(text.encode(), "plain.csv")
    assert table.google_trends is None and table.frame["v"].iloc[2] == "<1"
    assert "gt_detected" not in _codes(notes) and "gt_lt1_half" not in _codes(notes)
    assert "col_values_unreadable" in _codes(notes) and np.isnan(out["v"].iloc[2])
    assert "source" not in out["v"].attrs["trendai"]
    # a colon and brackets in a header alone do not make a Google Trends file
    other = "기간,매출: (원)\n" + "".join(f"2024-{m:02d},{m * m}\n" for m in range(1, 13))
    assert read_table_bytes(other.encode(), "sales.csv").google_trends is None


# --------------------------------------------------------------------------- bytes loader == file loader


def _legacy_load(path: Path) -> dict:
    """The reading path of load_table before the bytes refactor: Excel read from the path, no Google Trends
    detection, column labels left to frame_to_series."""
    raw = path.read_bytes()
    what = path.name
    suffix = path.suffix.lower()
    df, decimal_comma = None, False
    if suffix in (".xlsx", ".xlsm", ".xls", ".ods"):
        if raw.startswith(data._EXCEL_SIGNATURES):
            df = pd.read_excel(path, sheet_name=0)
        else:
            data._read_excel(raw, what)        # emits excel_is_text; parsed as text below
    elif suffix == ".json":
        text, _ = data._decode(raw, what)
        df = data._json_to_frame(json.loads(text.lstrip("﻿")), what)
    if df is None:
        text, _ = data._decode(raw, what)
        df, decimal_comma = data._read_text_table(text, what)
    return data.frame_to_series(df, prefer_decimal_comma=decimal_comma)


def _fixtures(d: Path) -> list[Path]:
    """The files tests/test_data.py writes to disk (one or two per test), plus an .xlsx year/month table."""
    files: list[tuple[str, object]] = []
    for k, fmt in enumerate([lambda y, m: f"{y}.{m:02d}", lambda y, m: f"{y}{m:02d}", lambda y, m: f"{y}년 {m}월",
                             lambda y, m: f"{y}-{m:02d}", lambda y, m: f"{y}/{m}", lambda y, m: f"{y}. {m}. 1.",
                             lambda y, m: f"\"{y}.{m:02d}\""]):
        rows = "\n".join(f"{fmt(2021 + i // 12, i % 12 + 1)},{v}" for i, v in enumerate(MONTHLY))
        files.append((f"m{k}.csv", "날짜,매출액\n" + rows + "\n"))
    for k, (header, cells) in enumerate([
        ("기간", [f"{2020 + i // 4}년 {i % 4 + 1}분기" for i in range(12)]),
        ("period", [f"{2020 + i // 4} Q{i % 4 + 1}" for i in range(12)]),
        ("week", [f"2024-W{i + 1:02d}" for i in range(20)]),
        ("dt", [f"{x:%Y%m%d}" for x in pd.date_range("2024-01-01", periods=40)]),
        ("timestamp", [str(1704067200 + 86400 * i) for i in range(30)]),
        ("year", [str(2020 - i) for i in range(12)]),
        ("m", [f"{x:%y-%m}" for x in pd.date_range("2020-01-01", periods=30, freq="MS")]),
        ("일자", [f"{x:%Y-%m-%d} ({'월화수목금토일'[x.dayofweek]})" for x in pd.date_range("2024-01-01", periods=20)]),
        ("date", [f"{x:%b %Y}" for x in pd.date_range("2020-01-01", periods=24, freq="MS")]),
    ]):
        files.append((f"d{k}.csv", f"{header},v\n" + "".join(f"{c},{i + 1}\n" for i, c in enumerate(cells))))
    files.append(("y.csv", "year,val\n1990,1\n1991,2\n1991,4\n1992,3\n1993,5\n,\n"))
    idx = pd.date_range("2022-01-01", periods=36, freq="MS")
    files.append(("eu.csv", "Datum,Wert\n" + "".join(f"{x:%d.%m.%Y},{100 + i}\n" for i, x in enumerate(idx))))
    dd = pd.date_range("2023-01-01", periods=200)
    files.append(("dmy.csv", "date,v\n" + "".join(f"{x:%d/%m/%Y},{i}\n" for i, x in enumerate(dd))))
    files.append(("us.csv", "date,v\n" + "".join(f"{x:%m/%d/%Y},{i}\n" for i, x in enumerate(dd[:60]))))
    tz = pd.date_range("2023-01-01", periods=24, freq="MS", tz="Europe/Berlin")
    files.append(("dst.csv", pd.DataFrame({"date": [x.isoformat() for x in tz], "val": range(24)}).to_csv(index=False)))
    big = pd.date_range("2016-01-01", periods=120, freq="MS")
    files.append(("t.csv", "date,sales\n" + "".join(f'{x:%Y-%m-%d},"{v:,.0f}"\n'
                                                     for x, v in zip(big, np.linspace(300, 1500, 120)))))
    for k, cells in enumerate([["12%", "13%", "15%"], ['"$1,234"', '"$2,000"', '"$3,500.5"'],
                               ['"1,234원"', '"2,000원"', '"35,000원"'], ['"(1,234)"', "500", '"-2,000"'],
                               ["1.5억", "2억", '"3,400만"'], ["12K", "1.5M", "2B"], ["1 234", "2 500", "3 000"]]):
        files.append((f"n{k}.csv", "date,v\n" + "".join(f"2024-{i + 1:02d}-01,{c}\n" for i, c in enumerate(cells))))
    files.append(("semi.csv", "Datum;Wert\n" + "".join(f"01.{m:02d}.2022;1.{m:03d},5\n" for m in range(1, 13))))
    files.append(("h.csv", "date,v\n2024-01,10\n2024-02,-\n2024-03,N/A\n2024-04,13\n"))
    files.append(("bad.csv", "date,v\n" + "".join(f"2024-{m:02d},{m}\n" for m in range(1, 12)) + "2024-12,abc\n"))
    files.append(("k.csv", ("날짜,검색량\n" + "".join(f"2020-{m:02d}-01,{m * 10}\n" for m in range(1, 13))).encode("cp949")))
    g = sample("growing")
    files.append(("u16.txt", ("날짜\t검색량\n" + "".join(f"{x:%Y-%m-%d}\t{v:.2f}\n" for x, v in g.items()))
                  .encode("utf-16")))
    files.append(("s0.json", g.to_json()))
    files.append(("s1.json", b"\xef\xbb\xbf" + pd.DataFrame({"date": g.index.strftime("%Y-%m-%d"), "v": g.values})
                  .to_json(orient="records").encode()))
    files.append(("s2.json", g.to_frame("v").to_json()))
    files.append(("s3.json", g.to_frame("v").to_json(orient="split")))
    ms = (g.index - pd.Timestamp("1970-01-01")) // pd.Timedelta(milliseconds=1)
    files.append(("r.json", json.dumps([{"date": int(x), "v": float(v)} for x, v in zip(ms, g.values)])))
    files.append(("fake.xlsx", "date,v\n" + "".join(f"2024-{m:02d}-01,{m}\n" for m in range(1, 13))))
    files.append(("nh.csv", "".join(f"2024-{m:02d}-01,{m * 3}\n" for m in range(1, 13))))
    head = "항목,단위," + ",".join(f"2021.{m:02d}" for m in range(1, 13))
    files.append(("w.csv", "\n".join([head, "매출,백만원," + ",".join(str(100 + m) for m in range(12)),
                                      "비용,백만원," + ",".join(str(50 + m) for m in range(12))]) + "\n"))
    files.append(("wi.csv", pd.DataFrame({"date": g.index.strftime("%Y-%m-%d"), "sales": g.round(2).values}).to_csv()))
    files.append(("ids.csv", "id,date,no_name,v\n" + "".join(f"{i + 1},2024-{i + 1:02d},{i},{(i * 7) % 5}\n"
                                                             for i in range(12))))
    files.append(("long.csv", "연도,월,값\n" + "".join(f"{2020 + i // 12},{i % 12 + 1},{100 + i}\n" for i in range(36))))
    files.append(("q.csv", "year,quarter,v\n" + "".join(f"{2020 + i // 4},Q{i % 4 + 1},{100 + i}\n" for i in range(12))))
    paths = []
    for name, content in files:
        p = d / name
        p.write_bytes(content if isinstance(content, bytes) else content.encode("utf-8"))
        paths.append(p)
    try:
        import openpyxl  # noqa: F401
    except ImportError:
        return paths
    pd.DataFrame({"날짜": g.index, "검색량": g.values}).to_excel(d / "d.xlsx", index=False)
    years = [2020.0] * 12 + [2021.0] * 12
    years[5] = np.nan                                      # a blank cell: the column is read as floats
    pd.DataFrame({"연도": years, "월": list(range(1, 13)) * 2, "판매량": np.arange(24.0) * 3 + 7}).to_excel(
        d / "ym.xlsx", index=False)
    return paths + [d / "d.xlsx", d / "ym.xlsx"]


def _run(fn):
    with collect_notices() as notes:
        try:
            result = fn()
        except DataError as exc:
            result = ("error", exc.code, str(exc))
    return result, [(type(n).__name__, str(n)) for n in notes]


def _assert_same(a, b, where: str) -> None:
    (res_a, notes_a), (res_b, notes_b) = a, b
    assert notes_a == notes_b, where
    if isinstance(res_a, tuple) or isinstance(res_b, tuple):
        assert res_a == res_b, where
        return
    assert list(res_a) == list(res_b), where
    for name in res_a:
        pd.testing.assert_series_equal(res_a[name], res_b[name], check_freq=True, obj=f"{where}:{name}")
        assert res_a[name].attrs == res_b[name].attrs, where


def test_load_table_bytes_equals_load_table(tmp_path):
    paths = _fixtures(tmp_path)
    assert len(paths) >= 45
    for p in paths:
        by_path = _run(lambda p=p: load_table(p))
        by_bytes = _run(lambda p=p: load_table_bytes(p.read_bytes(), p.name))
        legacy = _run(lambda p=p: _legacy_load(p))
        _assert_same(by_path, by_bytes, p.name)
        _assert_same(legacy, by_bytes, p.name)
    # explicit columns go the same way
    long = tmp_path / "long.csv"
    _assert_same(_run(lambda: load_table(long, date_col="연도", value_cols=["값"])),
                 _run(lambda: load_table_bytes(long.read_bytes(), "long.csv", date_col="연도", value_cols=["값"])),
                 "long.csv explicit")
    # unusable files raise the same errors
    for name, content in (("empty.csv", b""), ("bad.json", b"{not json"), ("nd.csv", b"a,b\nx,2\ny,4\n"),
                          ("o.xls", bytes.fromhex("D0CF11E0A1B11AE1") + b"\0" * 1000),
                          ("zip.csv", b"PK\x03\x04" + b"\0" * 100)):
        p = tmp_path / name
        p.write_bytes(content)
        _assert_same(_run(lambda p=p: load_table(p)),
                     _run(lambda content=content, name=name: load_table_bytes(content, name)), name)


def test_xlsx_frame_keeps_native_types(tmp_path):
    pytest.importorskip("openpyxl")
    paths = {p.name: p for p in _fixtures(tmp_path)}
    raw = paths["ym.xlsx"].read_bytes()
    table = read_table_bytes(raw, "ym.xlsx")
    assert table.format == "xlsx" and table.encoding is None and table.decimal_comma is False
    assert table.frame["연도"].dtype == np.float64 and table.frame["연도"].iloc[0] == 2020.0
    assert np.isnan(table.frame["연도"].iloc[5])
    with collect_notices():
        out = load_table_bytes(raw, "ym.xlsx")
    s = out["판매량"]
    assert s.attrs["trendai"]["date_cols"] == ["연도", "월"] and s.attrs["trendai"]["date_col"] == "연도 + 월"
    assert len(s) == 23 and s.index[0] == pd.Timestamp("2020-01-01")
    # frame_to_series on the RawTable (what the web app does) gives the same result
    with collect_notices():
        again = data.frame_to_series(table.frame, prefer_decimal_comma=table.decimal_comma)
    pd.testing.assert_series_equal(again["판매량"], s)


def test_read_table_bytes_xlsx(tmp_path):
    pytest.importorskip("openpyxl")
    g = sample("growing", n=24)
    buf = io.BytesIO()
    pd.DataFrame({"날짜": g.index, "검색량": g.values, " 메모  열 ": ["x"] * 24}).to_excel(buf, index=False)
    raw = buf.getvalue()
    table = read_table_bytes(raw, "월매출.xlsx")
    assert table.filename == "월매출.xlsx" and table.format == "xlsx" and table.encoding is None
    assert list(table.frame.columns) == ["날짜", "검색량", "메모 열"]
    assert pd.api.types.is_datetime64_any_dtype(table.frame["날짜"])
    assert table.google_trends is None
    assert read_table_bytes(raw, "x.xlsm").format == "xlsx"
    with pytest.raises(DataError) as ei:
        read_table_bytes(raw, "wide.xlsx", max_columns=2)
    assert ei.value.code == "table_too_many_columns" and ei.value.params == {"n": 3, "max": 2}
    with pytest.raises(DataError) as ei:
        read_table_bytes(b"PK\x03\x04garbage" * 10, "broken.xlsx")
    assert ei.value.code == "excel_unreadable"


def test_read_table_bytes_meta():
    rows = "".join(f"2020-{m:02d}-01,{m * 10}\n" for m in range(1, 13))
    t = read_table_bytes(("날짜,검색량\n" + rows).encode("cp949"), "k.csv")
    assert (t.format, t.encoding, t.decimal_comma) == ("csv", "cp949", False)
    assert list(t.frame.columns) == ["날짜", "검색량"] and t.frame["검색량"].iloc[0] == "10"
    assert read_table_bytes(("﻿date,v\n" + rows).encode("utf-8"), "b.csv").encoding == "utf-8-sig"
    t = read_table_bytes(("date\tv\n" + rows.replace(",", "\t")).encode("utf-16"), "u.tsv")
    assert (t.format, t.encoding) == ("tsv", "utf-16")
    t = read_table_bytes(("Datum;Wert\n" + "".join(f"01.{m:02d}.2022;1,5\n" for m in range(1, 13))).encode(), "s.csv")
    assert t.decimal_comma is True and t.format == "csv"
    assert read_table_bytes(("date\tv\n" + rows.replace(",", "\t")).encode(), "x.dat").format == "tsv"
    assert read_table_bytes(("date,v\n" + rows).encode(), "paste").format == "csv"
    assert read_table_bytes(("date v\n" + rows.replace(",", " ")).encode(), "x.log").format == "txt"
    assert read_table_bytes(("date\tv\n" + rows.replace(",", "\t")).encode(), "x.txt").format == "txt"
    t = read_table_bytes(json.dumps({"2020-01-01": 1, "2020-02-01": 2}).encode(), "s.json")
    assert (t.format, t.encoding) == ("json", "utf-8") and list(t.frame.columns) == ["date", "value"]
    with collect_notices() as notes:
        t = read_table_bytes(("date,v\n" + rows).encode(), "fake.xlsx")
    assert t.format == "csv" and _codes(notes) == ["excel_is_text"]
    with pytest.raises(DataError) as ei:
        read_table_bytes(b"  \n ", "empty.csv")
    assert ei.value.code == "file_empty" and ei.value.params == {"what": "empty.csv"}


def test_read_table_bytes_never_writes(monkeypatch, tmp_path):
    xlsx = None
    try:
        import openpyxl  # noqa: F401
        buf = io.BytesIO()
        pd.DataFrame({"date": pd.date_range("2024-01-01", periods=12, freq="MS"), "v": range(12)}).to_excel(
            buf, index=False)
        xlsx = buf.getvalue()
    except ImportError:
        pass
    inputs = [(("date,v\n" + "".join(f"2024-{m:02d},{m}\n" for m in range(1, 13))).encode(), "a.csv"),
              (json.dumps([{"date": f"2024-{m:02d}", "v": m} for m in range(1, 13)]).encode(), "a.json"),
              (CLASSIC.encode(), "gt.csv")]
    if xlsx is not None:
        inputs.append((xlsx, "a.xlsx"))
    with collect_notices():
        expected = [load_table_bytes(raw, name) for raw, name in inputs]     # warm up imports first

    def refuse(*args, **kwargs):
        raise AssertionError("the disk was touched")

    monkeypatch.setattr(builtins, "open", refuse)
    monkeypatch.setattr(io, "open", refuse)
    for name in ("write_bytes", "write_text", "open", "touch", "mkdir"):
        monkeypatch.setattr(Path, name, refuse)
    for name in ("NamedTemporaryFile", "TemporaryFile", "SpooledTemporaryFile", "mkstemp", "mkdtemp"):
        monkeypatch.setattr(tempfile, name, refuse)
    with collect_notices():
        for (raw, name), exp in zip(inputs, expected):
            table = read_table_bytes(raw, name, max_columns=5000)
            got = data.frame_to_series(table.frame, prefer_decimal_comma=table.decimal_comma)
            assert list(got) == list(exp)
            for key in got:
                pd.testing.assert_series_equal(got[key], exp[key])


def test_max_columns_checked_before_parse(monkeypatch):
    header = "date," + ",".join(f"c{i}" for i in range(5999))
    raw = (header + "\n" + "2024-01-01," + ",".join("1" for _ in range(5999)) + "\n").encode()

    def no_parse(*args, **kwargs):
        raise AssertionError("pd.read_csv was called")

    monkeypatch.setattr(pd, "read_csv", no_parse)
    with pytest.raises(DataError) as ei:
        read_table_bytes(raw, "wide.csv", max_columns=5000)
    assert ei.value.code == "table_too_many_columns" and ei.value.params == {"n": 6000, "max": 5000}
    assert render_web_msg(ei.value) == "표의 열이 너무 많아요(6000개, 최대 5000개)."
    # a Google Trends preamble does not hide the header from the check
    with pytest.raises(DataError):
        read_table_bytes(b"Category: All categories\n\n" + raw, "gt.csv", max_columns=5000)
    monkeypatch.undo()
    assert read_table_bytes(raw, "wide.csv", max_columns=6000).frame.shape == (1, 6000)
    # JSON (and Excel) are checked right after reading
    rows = json.dumps([{"date": "2024-01", **{f"c{i}": 1 for i in range(6)}}]).encode()
    with pytest.raises(DataError) as ei:
        read_table_bytes(rows, "x.json", max_columns=5)
    assert ei.value.params == {"n": 7, "max": 5}
    assert read_table_bytes(rows, "x.json").frame.shape == (1, 7)


def test_load_table_emits_gt_notices_as_warnings(tmp_path):
    p = tmp_path / "multiTimeline.csv"
    p.write_text(CLASSIC, encoding="utf-8")
    with pytest.warns(DataNotice) as rec:
        out = load_table(p)
    assert {"gt_detected", "gt_lt1_half"} <= {r.message.code for r in rec if isinstance(r.message, DataNotice)}
    assert out["parka"].attrs["trendai"]["value_kind"] == "relative_index_gt"
    assert not [r for r in rec if isinstance(r.message, DataWarning)]
