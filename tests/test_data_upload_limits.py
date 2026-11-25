"""Upload reading: shape limits before allocation (SEC-1), a title above the header (FU-4), the date-format label of
Excel date cells (FU-13) and the record of interpolated periods (FU-5)."""

from __future__ import annotations

import io
import json
import time
import tracemalloc
import zipfile

import numpy as np
import pandas as pd
import pytest

from trendai import data
from trendai.data import (
    INTERPOLATED_ATTR, DataError, collect_notices, frame_to_series, load_table_bytes, prepare, read_table_bytes,
)
from trendai.messages import format_label, render_web, render_web_msg


def _codes(notes) -> list[str]:
    return [n.code for n in notes]


def _sparse_xlsx(n_rows: int, col: str = "XFD", dimension: str = "") -> bytes:
    """A hand-made .xlsx whose sheet has, per row, one cell in column A and one in ``col`` (a few KB on disk)."""
    ct = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
          '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
          '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
          '<Default Extension="xml" ContentType="application/xml"/>'
          '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.'
          'spreadsheetml.sheet.main+xml"/>'
          '<Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-'
          'officedocument.spreadsheetml.worksheet+xml"/></Types>')
    rels = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/'
            'officeDocument" Target="xl/workbook.xml"/></Relationships>')
    wb = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
          '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
          'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
          '<sheets><sheet name="S" sheetId="1" r:id="rId1"/></sheets></workbook>')
    wbrels = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
              '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
              '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/'
              'worksheet" Target="worksheets/sheet1.xml"/></Relationships>')
    parts = ['<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
             '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">',
             f'<dimension ref="{dimension}"/>' if dimension else "", "<sheetData>"]
    for r in range(1, n_rows + 1):
        parts.append(f'<row r="{r}"><c r="A{r}"><v>{r}</v></c><c r="{col}{r}"><v>{r}</v></c></row>')
    parts.append("</sheetData></worksheet>")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", ct)
        z.writestr("_rels/.rels", rels)
        z.writestr("xl/workbook.xml", wb)
        z.writestr("xl/_rels/workbook.xml.rels", wbrels)
        z.writestr("xl/worksheets/sheet1.xml", "".join(parts))
    return buf.getvalue()


def _peak_mb(fn) -> tuple[object, float, float]:
    """``(result or DataError, peak traced MB, seconds)`` of ``fn()``."""
    tracemalloc.start()
    t0 = time.perf_counter()
    try:
        out: object = fn()
    except DataError as exc:
        out = exc
    finally:
        secs = time.perf_counter() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    return out, peak / 2**20, secs


# --------------------------------------------------------------------------- SEC-1: limits before allocation


def test_sparse_xlsx_far_column_is_refused_while_streaming():
    pytest.importorskip("openpyxl")
    raw = _sparse_xlsx(3000)            # ~30 KB; pd.read_excel padded it to 3000 x 16,384 cells (1.2 GB RSS)
    assert len(raw) < 60_000
    err, peak, secs = _peak_mb(lambda: read_table_bytes(raw, "bomb.xlsx", max_columns=5000))
    assert isinstance(err, DataError) and err.code == "table_too_many_columns"
    assert err.params == {"n": 16384, "max": 5000}
    assert peak < 40 and secs < 5, (peak, secs)
    # a wrong <dimension> does not hide the far cells
    err2, _, _ = _peak_mb(lambda: read_table_bytes(_sparse_xlsx(50, dimension="A1:A1"), "d.xlsx", max_columns=5000))
    assert isinstance(err2, DataError) and err2.code == "table_too_many_columns"


def test_sparse_xlsx_within_the_column_limit_is_bounded_by_cells():
    pytest.importorskip("openpyxl")
    raw = _sparse_xlsx(3000, col="GJH")       # column 5000: allowed width, but 3000 x 5000 = 15M cells
    err, peak, secs = _peak_mb(lambda: read_table_bytes(raw, "wide.xlsx", max_columns=5000))
    assert isinstance(err, DataError) and err.code == "table_too_large"
    assert err.params["max"] == data.UNTRUSTED_MAX_CELLS and err.params["n"] > data.UNTRUSTED_MAX_CELLS
    assert peak < 80 and secs < 5, (peak, secs)
    err, _, _ = _peak_mb(lambda: read_table_bytes(raw, "wide.xlsx", max_columns=5000, max_cells=100_000))
    assert isinstance(err, DataError) and err.code == "table_too_large" and err.params["max"] == 100_000
    assert render_web_msg(err).startswith("표가 너무 커요(칸 ")
    # the library loader (no limits) still reads what it is given
    small = _sparse_xlsx(20, col="J")
    assert read_table_bytes(small, "s.xlsx").frame.shape == (19, 10)     # padded like pd.read_excel


def test_json_records_with_distinct_keys_are_refused_before_the_frame():
    raw = json.dumps([{f"k{i}": i} for i in range(8000)]).encode()      # an 8000 x 8000 frame (1 GB) before
    err, peak, secs = _peak_mb(lambda: read_table_bytes(raw, "sparse.json", max_columns=5000))
    assert isinstance(err, DataError) and err.code == "table_too_large"
    assert peak < 60 and secs < 5, (peak, secs)
    err, _, _ = _peak_mb(lambda: read_table_bytes(raw, "sparse.json", max_columns=100, max_cells=10**9))
    assert isinstance(err, DataError) and err.code == "table_too_many_columns" and err.params["max"] == 100
    # {col: {date: value}} with distinct inner keys: the union of the inner keys is the row count
    cols = {f"c{i}": {f"2020-01-{(i % 28) + 1:02d}T{i}": 1} for i in range(3000)}
    err, _, _ = _peak_mb(lambda: read_table_bytes(json.dumps(cols).encode(), "c.json", max_columns=5000))
    assert isinstance(err, DataError) and err.code == "table_too_large"
    # ragged lists of lists: the widest row sets the width
    rows = [[1]] * 2000 + [list(range(4000))]
    err, _, _ = _peak_mb(lambda: read_table_bytes(json.dumps(rows).encode(), "l.json", max_columns=5000))
    assert isinstance(err, DataError) and err.code == "table_too_large"
    # ordinary JSON is unaffected
    ok = json.dumps([{"date": f"2024-{m:02d}", "v": m} for m in range(1, 13)]).encode()
    assert read_table_bytes(ok, "ok.json", max_columns=5000).frame.shape == (12, 2)


def test_text_cells_are_checked_before_the_parse(monkeypatch):
    ncols, nrows = 41, 1000
    header = "날짜," + ",".join(f"v{i}" for i in range(ncols - 1))
    raw = (header + "\n" + "".join(f"2020-01-01,{','.join('1' for _ in range(ncols - 1))}\n"
                                   for _ in range(nrows)) + "\n\n").encode()

    def no_parse(*args, **kwargs):
        raise AssertionError("pd.read_csv was called")

    monkeypatch.setattr(pd, "read_csv", no_parse)
    with pytest.raises(DataError) as ei:
        read_table_bytes(raw, "cells.csv", max_columns=5000, max_cells=40_000)
    assert ei.value.code == "table_too_large" and ei.value.params == {"n": ncols * nrows, "max": 40_000}
    monkeypatch.undo()
    # text is not capped unless asked: its frame is never larger than the file (the web passes max_cells)
    assert read_table_bytes(raw, "cells.csv", max_columns=5000).frame.shape == (nrows, ncols)


def test_memory_error_is_a_data_error(monkeypatch):
    def boom(*args, **kwargs):
        raise MemoryError

    monkeypatch.setattr(data, "_json_to_frame", boom)
    with pytest.raises(DataError) as ei:
        read_table_bytes(b'[{"date": "2024-01", "v": 1}]', "m.json")
    assert ei.value.code == "table_out_of_memory"
    assert render_web_msg(ei.value) == "m.json: 표가 너무 커서 읽을 수 없어요. 기간이나 열을 줄여 주세요."


def test_styled_empty_cells_beyond_the_limit_are_ignored():
    openpyxl = pytest.importorskip("openpyxl")
    from openpyxl.styles import Font

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["날짜", "매출"])
    for m in range(1, 13):
        ws.append([pd.Timestamp(f"2024-{m:02d}-01").to_pydatetime(), m * 10])
    ws.cell(row=1, column=6000).font = Font(bold=True)          # formatted, no value
    buf = io.BytesIO()
    wb.save(buf)
    table = read_table_bytes(buf.getvalue(), "styled.xlsx", max_columns=5000)
    assert list(table.frame.columns) == ["날짜", "매출"] and table.frame.shape == (12, 2)


def test_streamed_xlsx_equals_pandas_read_excel():
    openpyxl = pytest.importorskip("openpyxl")
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["날짜", "매출", "매출", "메모", "참", "비율"])
    for m in range(1, 13):
        ws.append([pd.Timestamp(f"2024-{m:02d}-01").to_pydatetime(), None if m == 5 else m * 1000,
                   float(m), "행사" if m % 4 == 0 else None, m % 2 == 0, m / 7])
    ws.append([None] * 6)                                          # a blank row inside the table
    ws.append(["합계", 78000, 78.0, None, None, None])
    buf = io.BytesIO()
    wb.save(buf)
    raw = buf.getvalue()
    ref = pd.read_excel(io.BytesIO(raw))
    ref.columns = data._clean_labels(ref.columns)
    got = read_table_bytes(raw, "p.xlsx", max_columns=5000).frame
    pd.testing.assert_frame_equal(got, ref)


# --------------------------------------------------------------------------- FU-4: a title above the header


def _title_xlsx(title_rows: list[list], header: list, rows: list[list]) -> bytes:
    openpyxl = pytest.importorskip("openpyxl")
    wb = openpyxl.Workbook()
    ws = wb.active
    for r in title_rows + [header] + rows:
        ws.append(r)
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


MONTHS = [pd.Timestamp(f"2022-{m:02d}-01").to_pydatetime() for m in range(1, 13)]
SALES = [[d, 8_000_000 + i * 371_000, 2300 + (i * 37) % 500] for i, d in enumerate(MONTHS)]


def test_title_row_above_the_header_is_skipped_in_excel():
    raw = _title_xlsx([["우리가게 월별 매출 현황 (단위: 원)"], []], ["날짜", "매출", "방문수"], SALES)
    plain = _title_xlsx([], ["날짜", "매출", "방문수"], SALES)
    with collect_notices() as notes:
        table = read_table_bytes(raw, "월별매출.xlsx", max_columns=5000)
    assert list(table.frame.columns) == ["날짜", "매출", "방문수"]
    assert (table.header_row, table.title_rows) == (2, 1)
    assert _codes(notes) == ["title_rows_skipped"] and notes[0].params == {"what": "월별매출.xlsx", "n": 1, "row": 3}
    assert render_web_msg(notes[0]) == "월별매출.xlsx: 맨 위 1개 행(표 제목 등)을 건너뛰고 3행을 열 이름으로 읽었어요."
    assert render_web_msg(notes[0], "en") == ("월별매출.xlsx: skipped 1 row(s) at the top (a title) and read row 3 as "
                                              "the column names.")
    pd.testing.assert_frame_equal(table.frame, read_table_bytes(plain, "p.xlsx").frame)
    assert read_table_bytes(plain, "p.xlsx").header_row == 0
    with collect_notices():
        series = frame_to_series(table.frame)
    assert list(series) == ["매출", "방문수"] and not any(str(c).startswith("Unnamed") for c in table.frame.columns)
    # the CLI loader reads the same file the same way
    with collect_notices() as notes:
        assert list(load_table_bytes(raw, "월별매출.xlsx")) == ["매출", "방문수"]
    assert "title_rows_skipped" in _codes(notes)


def test_title_lines_above_the_header_are_skipped_in_csv():
    rows = "".join(f"2024-{m:02d}-01,{m * 10},{m * 3 + 1}\n" for m in range(1, 13))
    for text in ("우리가게 월별 매출 현황 (단위: 원),,\n,,\n날짜,매출,방문수\n" + rows,       # saved by Excel
                 "월별 매출\n\n(단위: 원)\n날짜,매출,방문수\n" + rows):                # typed by hand
        with collect_notices() as notes:
            table = read_table_bytes(text.encode("cp949"), "t.csv", max_columns=5000)
        assert list(table.frame.columns) == ["날짜", "매출", "방문수"], text[:20]
        assert table.frame.shape == (12, 3) and _codes(notes) == ["title_rows_skipped"]
    with collect_notices():
        assert read_table_bytes(text.encode(), "t.csv").title_rows == 2


def test_header_row_can_be_chosen():
    raw = _title_xlsx([["매출 현황"], []], ["날짜", "매출", "방문수"], SALES)
    with collect_notices() as notes:
        table = read_table_bytes(raw, "x.xlsx", header_row=0)
    assert list(table.frame.columns) == ["매출 현황", "열 B", "열 C"] and table.header_row == 0
    assert notes == []                      # the user's choice: no notice
    assert list(read_table_bytes(raw, "x.xlsx", header_row=2).frame.columns) == ["날짜", "매출", "방문수"]
    csv = ("제목\n날짜,매출\n" + "".join(f"2024-{m:02d},{m}\n" for m in range(1, 13))).encode()
    assert list(read_table_bytes(csv, "c.csv", header_row=1).frame.columns) == ["날짜", "매출"]
    for bad in (1, 99, -1, True, "2"):
        with pytest.raises(DataError) as ei:
            read_table_bytes(raw, "x.xlsx", header_row=bad)
        assert ei.value.code == "header_row_invalid"
    assert render_web("header_row_invalid", {"row": 2}) == "2행은 열 이름으로 쓸 수 없어요(비어 있거나 표 밖이에요)."


def test_header_detection_has_no_false_positives():
    # a blank top-left header cell above dates: row 1 is the header (2 columns, one label)
    text = ",매출\n" + "".join(f"2024-{m:02d}-01,{m * 7 % 11 + 1}\n" for m in range(1, 13))
    table = read_table_bytes(text.encode(), "b.csv")
    assert list(table.frame.columns) == ["열 A", "매출"] and table.title_rows == 0
    # a single-cell row followed by data (no header row): nothing is skipped
    text = "매출 현황\n" + "".join(f"2024-{m:02d}-01,{m}\n" for m in range(1, 13))
    with collect_notices() as notes:
        table = read_table_bytes(text.encode(), "d.csv")
    assert table.title_rows == 0 and "title_rows_skipped" not in _codes(notes)
    # a one-column file
    assert read_table_bytes(b"v\n1\n2\n3\n", "one.csv").frame.shape == (3, 1)
    # a headerless file stays headerless
    with collect_notices() as notes:
        table = read_table_bytes("".join(f"2024-{m:02d}-01,{m * 3}\n" for m in range(1, 13)).encode(), "nh.csv")
    assert list(table.frame.columns) == ["date", "value"] and table.header_row is None
    # a KOSIS-style wide table under a title
    head = "항목," + ",".join(f"2021.{m:02d}" for m in range(1, 13))
    text = "\n".join(["지역별 매출", head, "서울," + ",".join(str(100 + m) for m in range(12)),
                      "부산," + ",".join(str(50 + m) for m in range(12))]) + "\n"
    with collect_notices():
        table = read_table_bytes(text.encode(), "w.csv")
        series = frame_to_series(table.frame)
    assert table.title_rows == 1 and list(series) == ["서울", "부산"]


def test_blank_header_cells_get_column_letters_on_the_web_path():
    text = ",매출,\n" + "".join(f"2024-{m:02d}-01,{(m * 7) % 13 + 2},{m}\n" for m in range(1, 13))
    table = read_table_bytes(text.encode(), "b.csv")
    assert list(table.frame.columns) == ["열 A", "매출", "열 C"]
    with collect_notices() as notes:
        series = frame_to_series(table.frame)
    assert list(series) == ["매출"] and notes[0].code == "rownum_cols_skipped"
    assert notes[0].params["cols"] == ["열 C"]          # a blank-headed 1..N column is a row number
    # the library loader keeps pandas' names (scripts and --value-col name them)
    with collect_notices() as notes:
        assert list(load_table_bytes(text.encode(), "b.csv")) == ["매출"]
    assert notes[0].params["cols"] == ["Unnamed: 2"]
    raw = _title_xlsx([], [None, "매출"], [[d, i * 5 % 7 + 1] for i, d in enumerate(MONTHS)])
    assert list(read_table_bytes(raw, "x.xlsx").frame.columns) == ["열 A", "매출"]
    assert data._col_letter(0) == "A" and data._col_letter(25) == "Z" and data._col_letter(26) == "AA"
    assert data._col_letter(16383) == "XFD"


# --------------------------------------------------------------------------- FU-13: date-format labels


def test_excel_date_cells_with_a_total_row_are_labelled_as_dates():
    raw = _title_xlsx([], ["날짜", "매출"], [[d, i + 1.5] for i, d in enumerate(MONTHS)] + [["합계", 90]])
    with collect_notices() as notes:
        series = frame_to_series(read_table_bytes(raw, "t.xlsx").frame)
    s = series["매출"]
    assert s.attrs["trendai"]["date_format"] == "날짜" and len(s) == 12
    assert _codes(notes) == ["dates_unreadable_rows"] and "합계" in notes[0].params["examples"]
    assert format_label("날짜", "en") == "date"
    # the same cells written out as text ("2022-01-01 00:00:00") are dates without a time of day
    text = "날짜,v\n" + "".join(f"2022-{m:02d}-01 00:00:00,{m}\n" for m in range(1, 13))
    with collect_notices():
        got = frame_to_series(read_table_bytes(text.encode(), "s.csv").frame)
    assert got["v"].attrs["trendai"]["date_format"] == "YYYY-MM-DD"
    text = "날짜,v\n" + "".join(f"2022-01-{d:02d} 13:30,{d}\n" for d in range(1, 13))
    with collect_notices():
        got = frame_to_series(read_table_bytes(text.encode(), "t.csv").frame)
    assert got["v"].attrs["trendai"]["date_format"] == "YYYY-MM-DD hh:mm"
    # native timestamps with a time of day keep "날짜/시간"
    ts = pd.Series(pd.date_range("2024-01-01 09:00", periods=5, freq="h"))
    assert data.parse_dates(ts)[1] == "날짜/시간"
    assert data.parse_dates(pd.Series(pd.date_range("2024-01-01", periods=5)))[1] == "날짜"


# --------------------------------------------------------------------------- FU-5: interpolated periods


def test_prepare_lists_every_interpolated_period():
    idx = pd.date_range("2022-01-01", periods=24, freq="MS")
    vals = np.arange(24, dtype=float) + 10
    vals[[5, 17]] = np.nan                      # two blank months: below the gaps_interpolated warning threshold
    raw = pd.Series(vals, index=idx, name="매출")
    with collect_notices() as notes:
        out, freq = prepare(raw)
    assert freq == "M" and notes == []
    assert out.attrs[INTERPOLATED_ATTR] == ["2022-06-01", "2023-06-01"]
    assert out["2022-06-01"] == pytest.approx(15.0, abs=0.05)            # linear in time
    again, _ = prepare(out)                     # a prepared series prepared again keeps the record
    assert again.attrs[INTERPOLATED_ATTR] == ["2022-06-01", "2023-06-01"]
    full, _ = prepare(pd.Series(np.arange(24.0), index=idx))
    assert full.attrs[INTERPOLATED_ATTR] == []
    # missing rows (not blank cells) are filled the same way
    with collect_notices():
        sparse, _ = prepare(raw.dropna())
    assert sparse.attrs[INTERPOLATED_ATTR] == ["2022-06-01", "2023-06-01"]
    assert render_web("gaps_filled", {"n": 2, "dates": "2022.06, 2023.06"}) == (
        "빈 기간 2개(2022.06, 2023.06)는 앞뒤 값으로 채운 추정값이에요. 실제 관측값이 아니에요.")
    assert render_web("gaps_filled", {"n": 1, "dates": "Jun 2023"}, "en") == (
        "1 empty period(s) (Jun 2023) were filled in from the values on either side: estimates, not observations.")


def test_header_row_of_a_google_trends_export_counts_its_preamble():
    text = "Category: All categories\n\nWeek,sunscreen: (South Korea)\n" + "".join(
        f"2024-01-{d:02d},{d}\n" for d in range(1, 15))
    with collect_notices() as notes:
        table = read_table_bytes(text.encode(), "multiTimeline.csv")
    assert (table.header_row, table.title_rows) == (2, 0) and _codes(notes) == ["gt_detected"]
    assert list(table.frame.columns) == ["Week", "sunscreen"]
