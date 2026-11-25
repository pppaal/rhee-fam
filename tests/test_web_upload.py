"""``POST /api/upload`` (C.4, E.17) and analysing uploads (C.5, E.6): detection, limits, guards, the upload cache
and the no-raw-bytes-on-disk promise."""

from __future__ import annotations

import datetime as dt
import io
import logging
import math
import os
import threading
import zipfile

import pytest

from trendai import data
from trendai.web.testing import (FrozenClock, TestClient, analyze_and_wait, assert_valid, make_test_app,
                                 monthly_csv, multipart_body, write_static_fixture)

T0 = dt.datetime(2026, 11, 25, 0, 12, tzinfo=dt.timezone.utc)


@pytest.fixture
def static(tmp_path):
    return write_static_fixture(tmp_path / "static")


@pytest.fixture
def make(tmp_path, static):
    apps = []

    def factory(**kw):
        kw.setdefault("static_dir", static)
        app = make_test_app(tmp_path / f"data{len(apps)}", **kw)
        apps.append(app)
        return app

    yield factory
    for app in apps:
        app.close(timeout=10)


@pytest.fixture
def app(make):
    return make()


@pytest.fixture
def client(app):
    return TestClient(app)


def sales_csv(n: int = 40, sep: str = ",") -> bytes:
    sales = [round(1000 + i * 10 + 50 * math.sin(i), 1) for i in range(n)]
    visits = [int(300 + i * 3 + 20 * math.cos(i)) for i in range(n)]
    memo = ["행사" if i % 7 == 0 else "" for i in range(n)]
    return monthly_csv({"매출": sales, "방문수": visits, "번호": list(range(1, n + 1)), "메모": memo}, sep=sep)


def test_multipart_csv_detection(client):
    resp = client.upload("월매출.csv", sales_csv())
    assert resp.status == 200
    assert resp.headers["cache-control"] == "no-store"
    body = resp.json()
    assert list(body) == ["upload_id", "expires_at", "filename", "size_bytes", "format", "encoding", "kind",
                          "columns", "detected", "problem", "preview", "notices", "google_trends"]
    assert len(body["upload_id"]) == 22
    assert (body["filename"], body["format"], body["kind"], body["problem"]) == ("월매출.csv", "csv", "table", None)
    roles = {c["name"]: (c["role"], c["reason"]) for c in body["columns"]}
    assert roles == {"날짜": ("date", None), "매출": ("value", None), "방문수": ("value", None),
                     "번호": ("excluded", "row_number"), "메모": ("ignored", "not_numeric")}
    shares = {c["name"]: c["numeric_share"] for c in body["columns"]}
    assert shares["매출"] == 1.0 and shares["메모"] == 0.0
    det = body["detected"]
    assert det == {"wide": False, "date_col": "날짜", "date_cols": ["날짜"], "date_format": "YYYY-MM",
                   "date_format_label": "YYYY-MM 형식 (월)", "value_cols": ["매출", "방문수"], "freq": "M",
                   "freq_label": "월간", "n_rows": 40, "first_date": "2022-01-01", "last_date": "2025-04-01"}
    note = [n for n in body["notices"] if n["code"] == "rownum_cols_skipped"][0]
    assert note["level"] == "notice" and note["series_index"] is None
    assert "열 ‘번호’는" in note["text"] and "['번호']" not in note["text"] and "은(는)" not in note["text"]
    prev = body["preview"]
    assert prev["columns"] == ["날짜", "매출", "방문수", "번호", "메모"]
    assert prev["head"][0][:2] == ["2022-01", "1000.0"] and len(prev["head"]) == 5 and len(prev["tail"]) == 5
    assert prev["tail"][-1][3] == "40"
    en = client.upload("월매출.csv", sales_csv(), hl="en").json()
    assert en["detected"]["freq_label"] == "Monthly"
    assert en["detected"]["date_format_label"] == "YYYY-MM format (monthly)"
    assert "row numbers" in en["notices"][0]["text"]


def test_paste_tsv(client):
    tsv = sales_csv(sep="\t").decode()
    resp = client.post("/api/upload", json={"text": tsv, "filename": "붙여넣기.tsv"})
    assert resp.status == 200
    body = resp.json()
    assert body["filename"] == "붙여넣기.tsv" and body["format"] == "tsv"
    assert body["detected"]["value_cols"] == ["매출", "방문수"]
    default = client.post("/api/upload", json={"text": tsv}).json()
    assert default["filename"] == "paste.tsv"
    for i, bad in enumerate(({"text": ""}, {"text": 5}, {"nothing": 1}, {"text": "a", "filename": 3})):
        resp = TestClient(client.app, remote_addr=f"10.3.0.{i}").post("/api/upload", json=bad)
        assert resp.status == 400 and resp.json()["error"]["code"] == "bad_request", bad


def test_cp949_csv(client):
    raw = sales_csv().decode("utf-8").encode("cp949")
    body = client.upload("cp949.csv", raw).json()
    assert body["encoding"] == "cp949"
    assert body["detected"]["date_col"] == "날짜" and "매출" in body["detected"]["value_cols"]


def test_upload_errors(client):
    resp = client.post("/api/upload", data=b"abc", headers={"Content-Type": "text/csv"})
    assert resp.status == 415 and resp.json()["error"]["code"] == "unsupported_media_type"
    body, ctype = multipart_body("x.csv", b"a,b\n1,2\n", field="other")
    resp = client.post("/api/upload", data=body, headers={"Content-Type": ctype})
    assert resp.status == 400 and resp.json()["error"]["code"] == "bad_request"
    resp = client.post("/api/upload", data=b"--x--", headers={"Content-Type": "multipart/form-data"})
    assert resp.status == 400
    resp = client.upload("empty.csv", b"")
    assert resp.status == 422
    err = resp.json()["error"]
    assert err["code"] == "upload_unreadable" and err["params"]["reason"] == "file_empty"
    assert [f["action"] for f in err["fixes"]] == ["show_formats", "try_paste"]


def test_xlsx_upload(client):
    openpyxl = pytest.importorskip("openpyxl")
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["날짜", "매출"])
    for i in range(30):
        ws.append([dt.datetime(2022 + i // 12, i % 12 + 1, 1), 1000 + i * 7 + (i % 5) * 3])
    buf = io.BytesIO()
    wb.save(buf)
    resp = client.upload("월매출.xlsx", buf.getvalue())
    assert resp.status == 200
    body = resp.json()
    assert body["format"] == "xlsx" and body["encoding"] is None
    assert body["detected"]["value_cols"] == ["매출"] and body["detected"]["freq"] == "M"
    assert body["preview"]["head"][0][0] == "2022-01-01"


def test_xlsx_year_month_columns(client):
    openpyxl = pytest.importorskip("openpyxl")
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["연도", "월", "매출"])
    for i in range(30):
        ws.append([2020 + i // 12 if i != 5 else None, i % 12 + 1, 500 + i * 9 + (i % 4) * 5])
    buf = io.BytesIO()
    wb.save(buf)
    body = client.upload("연월.xlsx", buf.getvalue()).json()
    det = body["detected"]
    assert det["date_cols"] == ["연도", "월"] and det["date_col"] == "연도"
    roles = {c["name"]: c["role"] for c in body["columns"]}
    assert roles == {"연도": "date", "월": "date", "매출": "value"}
    assert body["preview"]["head"][0][0] == "2020"
    snap, env = analyze_and_wait(client, {"source": "upload", "upload_id": body["upload_id"], "date_col": "연도",
                                          "value_cols": ["매출"]})
    assert snap["status"] == "done", snap["error"]
    assert env["query"]["date_col"] == "연도"


GT = ("Category: All categories\n\nWeek,sunscreen: (South Korea),parka: (South Korea)\n"
      + "".join(f"{d},{(i * 37) % 90 + 5},{'<1' if i % 4 == 0 else (i * 13) % 60 + 3}\n"
                for i, d in enumerate(f"{x:%Y-%m-%d}" for x in __import__('pandas').date_range(
                    "2023-11-26", periods=40, freq="W-SUN"))))


def test_google_trends_csv(client):
    body = client.upload("multiTimeline.csv", GT.encode()).json()
    assert body["kind"] == "google_trends"
    assert body["google_trends"] == {"terms": [{"col": "sunscreen", "term": "sunscreen", "region": "South Korea"},
                                               {"col": "parka", "term": "parka", "region": "South Korea"}],
                                     "lt1_count": 10, "category": "All categories"}
    assert [n["code"] for n in body["notices"]][:2] == ["gt_detected", "gt_lt1_half"]
    snap, env = analyze_and_wait(client, {"source": "upload", "upload_id": body["upload_id"]})
    assert snap["status"] == "done", snap["error"]
    vm = env["series"][0]
    assert vm["source"]["kind"] == "google_trends_upload" and vm["source"]["value_kind"] == "relative_index_gt"
    assert "Google Trends (https://www.google.com/trends)" in vm["source"]["line"]
    assert env["compare"]["scale"]["allowed"]["index"] is False


def test_no_date_col_problem_200(client):
    raw = b"name,value\nalpha,1\nbeta,2\ngamma,3\ndelta,4\n"
    resp = client.upload("nodate.csv", raw)
    assert resp.status == 200
    body = resp.json()
    prob = body["problem"]
    assert_valid(prob, "error")
    assert prob["code"] == "upload_no_date_col" and prob["status"] == 200 and prob["params"] == {}
    assert prob["fixes"] == [{"action": "pick_columns", "label_ko": "열 고르기", "label_en": "Pick columns",
                              "params": {}}]
    assert prob["retry_after"] is None and len(prob["request_id"]) == 8
    assert body["detected"]["date_col"] is None and body["detected"]["value_cols"] == []
    novalue = client.upload("novalue.csv", b"date,label\n2024-01,a\n2024-02,b\n2024-03,c\n").json()
    assert novalue["problem"]["code"] == "upload_no_value_col"


def test_too_large_413(make):
    app = make()
    c = TestClient(app)
    resp = c.post("/api/upload", data=b"x" * 10, headers={"Content-Type": "multipart/form-data; boundary=x",
                                                         "Content-Length": str(10 * 1024 * 1024 + 65537)})
    assert resp.status == 413
    err = resp.json()["error"]
    assert err["code"] == "body_too_large" and err["params"] == {"max_mb": 10}
    assert err["message_ko"] == "파일이 너무 커요. 10 MB까지 올릴 수 있어요."
    small = make(upload_max_mb=1)
    resp = TestClient(small).upload("big.csv", b"1,2\n" * (1024 * 1024 // 4 + 100))
    assert resp.status == 413 and resp.json()["error"]["params"] == {"max_mb": 1}


def _zip_of_zeros(size: int) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        with zf.open("xl/worksheets/sheet1.xml", "w", force_zip64=True) as fh:
            chunk = b"\0" * (1 << 20)
            for _ in range(size // len(chunk)):
                fh.write(chunk)
    return buf.getvalue()


def test_zip_bomb_guard(client):
    for mb in (200, 40):
        raw = _zip_of_zeros(mb * 1024 * 1024)
        assert len(raw) < 1024 * 1024
        resp = client.upload("bomb.xlsx", raw)
        assert resp.status == 413, mb
        assert resp.json()["error"]["code"] == "upload_archive_too_large"
    broken = client.upload("broken.xlsx", b"PK\x03\x04" + b"garbage" * 10)
    assert broken.status == 422 and broken.json()["error"]["code"] == "upload_unreadable"


def test_too_many_columns(client):
    cols = {f"c{i}": [i + j * 1.5 for j in range(12)] for i in range(59)}
    resp = client.upload("wide.csv", monthly_csv(cols))
    assert resp.status == 422
    err = resp.json()["error"]
    assert err["code"] == "upload_too_many_columns" and err["params"] == {"n": 60, "max": 50}


def test_header_too_wide_422(client):
    header = ",".join(f"c{i}" for i in range(6000))
    resp = client.upload("huge.csv", (header + "\n" + ",".join("1" for _ in range(6000)) + "\n").encode())
    assert resp.status == 422
    err = resp.json()["error"]
    assert err["code"] == "upload_too_many_columns" and err["params"]["max"] == 5000


def test_wide_table_many_periods_ok(client):
    months = [f"{2015 + i // 12}-{i % 12 + 1:02d}" for i in range(120)]
    lines = ["항목," + ",".join(months)]
    for r, name in enumerate(("서울", "부산", "대구")):
        lines.append(name + "," + ",".join(str(100 + r * 10 + i + (i % 12) * 2) for i in range(120)))
    resp = client.upload("kosis.csv", ("\n".join(lines) + "\n").encode())
    assert resp.status == 200
    body = resp.json()
    det = body["detected"]
    assert det["wide"] is True and det["date_col"] is None and det["date_cols"] == []
    assert det["value_cols"] == ["서울", "부산", "대구"]
    assert [c["role"] for c in body["columns"]] == ["value"] * 3


def test_too_large_table(client):
    ncols, nrows = 41, 48_800
    header = "날짜," + ",".join(f"v{i}" for i in range(ncols - 1))
    row = ",".join("1" for _ in range(ncols - 1))
    raw = (header + "\n" + "".join(f"2020-01-01,{row}\n" for _ in range(nrows))).encode()
    resp = client.upload("cells.csv", raw)
    assert resp.status == 422
    err = resp.json()["error"]
    assert err["code"] == "upload_too_large_table" and err["params"] == {"n": ncols * nrows, "max": 2_000_000}


def test_parse_slots_503(client, monkeypatch):
    gate = threading.Event()
    entered = threading.Semaphore(0)
    real = data.read_table_bytes

    def slow(raw, name, **kw):
        entered.release()
        gate.wait(10)
        return real(raw, name, **kw)

    monkeypatch.setattr(data, "read_table_bytes", slow)
    results = []
    threads = [threading.Thread(target=lambda i=i: results.append(
        TestClient(client.app, remote_addr=f"10.2.0.{i}").upload("a.csv", sales_csv()).status)) for i in range(2)]
    for t in threads:
        t.start()
    assert entered.acquire(timeout=5) and entered.acquire(timeout=5)
    third = TestClient(client.app, remote_addr="10.2.0.9").upload("a.csv", sales_csv())
    gate.set()
    for t in threads:
        t.join(10)
    assert third.status == 503
    assert third.headers["retry-after"] == "5" and third.json()["error"]["code"] == "server_busy"
    assert results == [200, 200]


def test_upload_rate_limit(client):
    raw = sales_csv(12)
    assert [client.upload("a.csv", raw).status for _ in range(5)] == [200] * 5
    resp = client.upload("a.csv", raw)
    assert resp.status == 429 and resp.json()["error"]["code"] == "rate_limited"


def test_more_than_20_detected(client):
    cols = {f"값{i}": [round(100 + i * 3 + j * 1.7 + (j % 5), 1) for j in range(14)] for i in range(25)}
    body = client.upload("many.csv", monthly_csv(cols)).json()
    assert len(body["detected"]["value_cols"]) == 25
    resp = client.post("/api/analyze", json={"source": "upload", "upload_id": body["upload_id"], "value_cols": None})
    assert resp.status == 400
    err = resp.json()["error"]
    assert err["code"] == "upload_too_many_series" and err["params"] == {"max": 20}
    resp = client.post("/api/analyze", json={"source": "upload", "upload_id": body["upload_id"],
                                             "value_cols": [f"값{i}" for i in range(21)]})
    assert resp.json()["error"]["code"] == "upload_too_many_series"


def test_duplicate_wide_row_names(client):
    months = [f"2020-{m:02d}" for m in range(1, 13)]
    lines = ["항목," + ",".join(months), "A," + ",".join(str(10 + i) for i in range(12)),
             "A," + ",".join(str(50 + 2 * i) for i in range(12)), "B," + ",".join(str(30 + i % 3) for i in range(12))]
    body = client.upload("dup.csv", ("\n".join(lines) + "\n").encode()).json()
    assert body["detected"]["value_cols"] == ["A", "A (2)", "B"]
    snap, env = analyze_and_wait(client, {"source": "upload", "upload_id": body["upload_id"],
                                          "value_cols": ["A (2)", "A"]})
    assert snap["status"] == "done"
    assert [v["name"] for v in env["series"]] == ["A (2)", "A"]


def test_upload_expired_410(make):
    clock = FrozenClock(T0)
    app = make(clock=clock)
    c = TestClient(app)
    body = c.upload("a.csv", sales_csv()).json()
    assert body["expires_at"] == "2026-11-25T00:42:00Z"
    clock.advance(31 * 60)
    resp = c.post("/api/analyze", json={"source": "upload", "upload_id": body["upload_id"]})
    assert resp.status == 410
    err = resp.json()["error"]
    assert err["code"] == "upload_expired" and [f["action"] for f in err["fixes"]] == ["reupload"]


def test_analyze_upload_selected_cols(client):
    body = client.upload("월매출.csv", sales_csv()).json()
    snap, env = analyze_and_wait(client, {"source": "upload", "upload_id": body["upload_id"], "date_col": None,
                                          "value_cols": ["매출"], "freq": "auto", "unit_label": "원", "h": 6})
    assert snap["status"] == "done" and env["view"] == "detail"
    assert env["query"] == {"source": "upload", "filename": "월매출.csv", "date_col": "날짜", "value_cols": ["매출"],
                            "freq": "M", "h": 6, "unit_label": "원"}
    vm = env["series"][0]
    assert vm["source"]["kind"] == "upload" and vm["source"]["unit_label"] == "원"
    assert vm["source"]["line"].startswith("출처: 사용자 업로드 (월매출.csv)")
    assert vm["narrative"]["ai_allowed"] is True


def test_upload_horizon_clamped(client):
    body = client.upload("월매출.csv", sales_csv()).json()
    snap, env = analyze_and_wait(client, {"source": "upload", "upload_id": body["upload_id"], "value_cols": ["매출"],
                                          "h": 100})
    assert snap["status"] == "done", snap["error"]
    vm = env["series"][0]
    assert vm["future"]["horizon"] == 36
    note = [n for n in vm["notices"] if n["code"] == "horizon_clamped"][0]
    assert note["text"] == "예측 기간을 최대치인 36개월로 줄였어요."
    assert env["query"]["h"] == 36


def test_bad_date_col_400(client):
    body = client.upload("월매출.csv", sales_csv()).json()
    resp = client.post("/api/analyze", json={"source": "upload", "upload_id": body["upload_id"], "date_col": "메모"})
    snap = client.wait_job(resp.json()["job_id"]) if resp.status == 202 else None
    err = resp.json()["error"] if snap is None else snap["error"]
    assert err["code"] == "upload_bad_date_col" and err["params"] == {"col": "메모"}
    assert err["fixes"][0]["action"] == "pick_columns"
    resp = client.post("/api/analyze", json={"source": "upload", "upload_id": body["upload_id"], "date_col": "없음"})
    assert resp.status == 400 and resp.json()["error"]["code"] == "upload_bad_date_col"
    resp = client.post("/api/analyze", json={"source": "upload", "upload_id": body["upload_id"],
                                             "value_cols": ["없는열"]})
    assert resp.status == 400 and resp.json()["error"]["code"] == "upload_bad_value_col"


def test_upload_index_scale_default(client):
    body = client.upload("월매출.csv", sales_csv()).json()
    snap, env = analyze_and_wait(client, {"source": "upload", "upload_id": body["upload_id"]})
    assert env["view"] == "compare"
    assert env["compare"]["scale"]["default"] == "index"
    assert env["exports"]["values_csv"] is not None


def _all_bytes(root: str) -> list[bytes]:
    out = []
    for dirpath, _dirs, files in os.walk(root):
        for name in files:
            with open(os.path.join(dirpath, name), "rb") as fh:
                out.append(fh.read())
    return out


def test_raw_upload_never_written(client, app, caplog):
    caplog.set_level(logging.DEBUG)
    rows = ["날짜,매출,메모"]
    for i in range(30):
        value = "SECRET-MARKER-9b1c" if i == 12 else str(1000 + i * 11 + (i % 3))
        rows.append(f"{2022 + i // 12}-{i % 12 + 1:02d},{value},SECRET-MARKER-7f3a")
    up = client.upload("secret.csv", ("\n".join(rows) + "\n").encode())
    assert up.status == 200
    body = up.json()
    snap, env = analyze_and_wait(client, {"source": "upload", "upload_id": body["upload_id"], "value_cols": ["매출"]})
    assert snap["status"] == "done", snap["error"]
    rid = env["result_id"]
    for fmt in ("csv", "xlsx", "json", "html"):
        client.get(f"/api/export/{rid}/0.{fmt}")
        client.get(f"/api/export/{rid}.{fmt}")
    share = client.post("/api/share", json={"result_id": rid}).json()
    client.get(f"/api/shared/{share['share_id']}")
    malformed = b'a,b\n"SECRET-MARKER-MALFORMED,1\n2,3,4,5,6\n\x00\x00\xff\xfe broken'
    client.upload("broken.csv", malformed)
    client.upload("broken.xlsx", b"PK\x03\x04SECRET-MARKER-MALFORMED")
    blobs = _all_bytes(app.config.data_dir)
    assert blobs and not [b for b in blobs if b"SECRET-MARKER-7f3a" in b]
    logs = "\n".join(r.getMessage() for r in caplog.records)
    for marker in ("SECRET-MARKER-7f3a", "SECRET-MARKER-9b1c", "SECRET-MARKER-MALFORMED", "secret.csv",
                   "broken.csv", rid):
        assert marker not in logs, marker
