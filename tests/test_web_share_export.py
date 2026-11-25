"""Share links (C.8, C.9) and export routes (C.10, E.15) over HTTP."""

from __future__ import annotations

import datetime as dt
import io
import math
import os
import re
from urllib.parse import quote

import pytest

from trendai.web import exports
from trendai.web.testing import (FrozenClock, TestClient, analyze_and_wait, assert_valid, make_test_app,
                                 monthly_csv, write_static_fixture)

T0 = dt.datetime(2026, 11, 25, 0, 12, tzinfo=dt.timezone.utc)
EXPORT_CSP = ("sandbox; default-src 'none'; style-src 'unsafe-inline' https://fonts.googleapis.com; "
              "font-src https://fonts.gstatic.com; img-src data:")


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


def upload_result(client: TestClient, columns: dict | None = None, **extra) -> dict:
    columns = columns or {"매출": [round(1000 + i * 10 + 50 * math.sin(i), 1) for i in range(36)],
                          "방문수": [int(300 + i * 3 + 20 * math.cos(i)) for i in range(36)]}
    up = client.upload("월매출.csv", monthly_csv(columns)).json()
    snap, env = analyze_and_wait(client, {"source": "upload", "upload_id": up["upload_id"], **extra})
    assert snap["status"] == "done", snap["error"]
    return env


def test_share_only_uploads(client):
    snap, env = analyze_and_wait(client, {"source": "sample", "series": [{"terms": ["growing"]}]})
    resp = client.post("/api/share", json={"result_id": env["result_id"]})
    assert resp.status == 400
    err = resp.json()["error"]
    assert err["code"] == "share_not_needed" and [f["action"] for f in err["fixes"]] == ["copy_link"]
    assert client.post("/api/share", json={"result_id": "X" * 22}).json()["error"]["code"] == "result_not_found"
    assert client.post("/api/share", json={"result_id": "../etc"}).status == 404


def test_share_create_get_delete(client):
    env = upload_result(client)
    resp = client.post("/api/share", json={"result_id": env["result_id"]})
    assert resp.status == 201
    share = resp.json()
    assert set(share) == {"share_id", "url", "expires_at", "delete_token"}
    sid = share["share_id"]
    assert share["url"] == f"/s/{sid}" and len(sid) == 22 and len(share["delete_token"]) == 22
    got = client.get(f"/api/shared/{sid}")
    assert got.status == 200
    assert got.headers["cache-control"] == "private, no-store"
    assert got.headers["x-robots-tag"] == "noindex, nofollow"
    body = got.json()
    assert_valid(body, "resultEnvelope")
    assert body["readonly"] is True and body["result_id"] is None
    assert body["shared"] == {"share_id": sid, "created_at": body["shared"]["created_at"],
                              "expires_at": share["expires_at"]}
    assert body["exports"]["csv"] == f"/api/export/s/{sid}.csv"
    assert body["series"][0]["exports"]["csv"] == f"/api/export/s/{sid}/0.csv"
    for vm in body["series"]:
        assert vm["narrative"]["ai_allowed"] is False
        assert vm["narrative"]["ai_block_reason"] == "shared_readonly"
    assert client.delete(f"/api/share/{sid}").status == 403
    wrong = client.delete(f"/api/share/{sid}", headers={"X-Trendai-Delete-Token": "x" * 22})
    assert wrong.status == 403 and wrong.json()["error"]["code"] == "share_forbidden"
    ok = client.delete(f"/api/share/{sid}", headers={"X-Trendai-Delete-Token": share["delete_token"]})
    assert ok.status == 204 and ok.body == b""
    gone = client.get(f"/api/shared/{sid}")
    assert gone.status == 404 and gone.json()["error"]["code"] == "share_not_found"
    assert client.delete(f"/api/share/{sid}", headers={"X-Trendai-Delete-Token": share["delete_token"]}).status == 404


def test_share_never_leaks_result_id(client, app):
    env = upload_result(client)
    rid = env["result_id"]
    client.post("/api/narrative", json={"result_id": rid, "series": 0, "lang": "ko"})
    share = client.post("/api/share", json={"result_id": rid}).json()
    sid = share["share_id"]
    for hl in ("ko", "en"):
        resp = client.get(f"/api/shared/{sid}?hl={hl}")
        assert rid.encode() not in resp.body
        assert rid not in "".join(v for _k, v in resp.header_list)
    share_dir = os.path.join(app.config.data_dir, "shares", sid)
    for dirpath, _dirs, files in os.walk(share_dir):
        for name in files:
            with open(os.path.join(dirpath, name), "rb") as fh:
                assert rid.encode() not in fh.read(), name
    assert not [n for n in os.listdir(share_dir) if n.startswith("llm-")]


def test_share_outlives_result(make):
    clock = FrozenClock(T0)
    app = make(clock=clock)
    c = TestClient(app)
    env = upload_result(c)
    share = c.post("/api/share", json={"result_id": env["result_id"]}).json()
    assert share["expires_at"] == "2026-12-25T00:12:00Z"
    clock.advance(25 * 3600)
    assert c.get(f"/api/results/{env['result_id']}").status == 404
    shared = c.get(f"/api/shared/{share['share_id']}")
    assert shared.status == 200
    assert c.get(f"/api/export/s/{share['share_id']}/0.csv").status == 200


def test_share_expires_30d(make):
    clock = FrozenClock(T0)
    app = make(clock=clock)
    c = TestClient(app)
    env = upload_result(c)
    share = c.post("/api/share", json={"result_id": env["result_id"]}).json()
    clock.advance(30 * 86400 - 60)
    assert c.get(f"/api/shared/{share['share_id']}").status == 200
    clock.advance(120)
    assert c.get(f"/api/shared/{share['share_id']}").json()["error"]["code"] == "share_not_found"
    assert c.get(f"/api/export/s/{share['share_id']}.csv").json()["error"]["code"] == "share_not_found"


def test_csv_format(make):
    app = make(clock=FrozenClock(T0))
    c = TestClient(app)
    snap, env = analyze_and_wait(c, {"source": "wiki", "series": [{"terms": ["인공지능"]}], "unit": "week",
                                     "range": "1y"})
    resp = c.get(f"/api/export/{env['result_id']}/0.csv")
    assert resp.status == 200
    assert resp.headers["content-type"] == "text/csv; charset=utf-8"
    assert resp.headers["cache-control"] == "private, no-store"
    raw = resp.body
    assert raw.startswith(b"\xef\xbb\xbf")
    text = raw.decode("utf-8-sig")
    assert "\r\n" in text and "\n" not in text.replace("\r\n", "")
    lines = text.split("\r\n")
    assert lines[0].startswith("# 출처: Wikimedia Pageviews") and lines[1].startswith("# 조건: ")
    assert " ~ 2026.11.22 · " in lines[0]                         # weekly: the last bucket's Sunday
    assert lines[2] == "날짜,실제값,추세,예측값,80% 하한,80% 상한,95% 하한,95% 상한"
    rows = [ln.split(",") for ln in lines[3:] if ln]
    history = [r for r in rows if r[1]]
    forecast = [r for r in rows if not r[1]]
    assert history and forecast and rows.index(forecast[0]) == len(history)
    assert all(not r[3] for r in history) and all(r[3] for r in forecast)
    for r in history[:5]:
        float(r[1])
        assert "e" not in r[1].lower() and len(r[1].replace(".", "").replace("-", "").lstrip("0")) <= 15
    en = c.get(f"/api/export/{env['result_id']}/0.csv?hl=en").body.decode("utf-8-sig").split("\r\n")
    assert en[2] == "date,actual,trend,forecast,lo80,hi80,lo95,hi95"


def test_csv_formula_guard(client):
    env = upload_result(client, {"=SUM(A1)": [round(10 + i * 1.3 + (i % 4), 1) for i in range(30)],
                                 "plain": [round(50 + i * 0.7 + (i % 3), 1) for i in range(30)]})
    rid = env["result_id"]
    values = client.get(f"/api/export/{rid}/values.csv").body.decode("utf-8-sig").split("\r\n")
    assert values[2].split(",")[1] == "'=SUM(A1)"
    ranking = client.get(f"/api/export/{rid}.csv").body.decode("utf-8-sig")
    assert "'=SUM(A1)" in ranking and ",=SUM(A1)" not in ranking


def test_values_csv(client):
    env = upload_result(client)
    resp = client.get(f"/api/export/{env['result_id']}/values.csv")
    assert resp.status == 200
    lines = resp.body.decode("utf-8-sig").split("\r\n")
    assert lines[0] == "# " + env["compare"]["source_line"]
    assert lines[2] == "날짜,매출,방문수"
    assert lines[3].startswith("2022-01-01,")
    single = upload_result(client, value_cols=["매출"])
    assert client.get(f"/api/export/{single['result_id']}/values.csv").json()["error"]["code"] == "not_found"


def test_content_disposition_utf8(client):
    env = upload_result(client)
    rid = env["result_id"]
    disp = client.get(f"/api/export/{rid}/1.csv").headers["content-disposition"]
    assert disp == (f'attachment; filename="trendai-{rid[:8]}-1.csv"; '
                    f"filename*=UTF-8''{quote('trendai-방문수.csv')}")
    whole = client.get(f"/api/export/{rid}.json").headers["content-disposition"]
    assert f'filename="trendai-{rid[:8]}-all.json"' in whole
    assert quote("매출 vs 방문수") in whole


def test_export_errors(client):
    env = upload_result(client)
    rid = env["result_id"]
    assert client.get(f"/api/export/{rid}/7.csv").json()["error"]["code"] == "not_found"
    assert client.get(f"/api/export/{rid}/01.csv").status == 404
    assert client.get(f"/api/export/{rid}.pdf").status == 404
    assert client.get(f"/api/export/{'Q' * 22}.csv").json()["error"]["code"] == "result_not_found"
    assert client.get(f"/api/export/s/{'Q' * 22}.csv").json()["error"]["code"] == "share_not_found"
    for _ in range(40):
        last = client.get(f"/api/export/{rid}/0.csv")
    assert last.status == 429 and last.json()["error"]["code"] == "rate_limited"


def test_xlsx_structure(client):
    openpyxl = pytest.importorskip("openpyxl")
    env = upload_result(client, {"=SUM(A1)": [round(10 + i * 1.3 + (i % 4), 1) for i in range(30)],
                                 "plain": [round(50 + i * 0.7 + (i % 3), 1) for i in range(30)]})
    resp = client.get(f"/api/export/{env['result_id']}.xlsx")
    assert resp.status == 200
    assert resp.headers["content-type"] == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    wb = openpyxl.load_workbook(io.BytesIO(resp.body))
    assert wb.sheetnames[-1] == "요약"
    ws = wb[wb.sheetnames[0]]
    labels = [ws.cell(row=r, column=1).value for r in range(1, 8)]
    assert labels == ["출처", "검색어 그룹/열", "기간", "단위", "조건(기기·성별·연령)", "데이터 기준", "값의 성격"]
    assert ws.cell(row=8, column=1).value in (None, "")
    assert ws.cell(row=9, column=1).value == "날짜"
    assert ws.cell(row=10, column=1).value is not None
    texts = [c for sheet in wb.worksheets for row in sheet.iter_rows() for c in row
             if isinstance(c.value, str) and "=SUM(A1)" in c.value]
    assert texts and all(c.data_type == "s" and not c.value.startswith("'") for c in texts)
    summary = wb["요약"]
    names = [summary.cell(row=r, column=1).value for r in range(1, summary.max_row + 1)]
    assert "=SUM(A1)" in names


def test_export_xlsx_unavailable_501(client, monkeypatch):
    env = upload_result(client)
    monkeypatch.setattr(exports, "xlsx_available", lambda: False)
    for path in (f"/api/export/{env['result_id']}.xlsx", f"/api/export/{env['result_id']}/0.xlsx"):
        resp = client.get(path)
        assert resp.status == 501
        err = resp.json()["error"]
        assert err["code"] == "export_xlsx_unavailable" and err["fixes"][0]["action"] == "export_csv"
    other = upload_result(client)
    assert other["exports"]["xlsx"] is None
    assert other["exports"]["unavailable"]["xlsx"]["code"] == "export_xlsx_unavailable"
    assert client.get(f"/api/export/{other['result_id']}.xlsx").status == 501


def test_html_export_csp_sandbox(client):
    env = upload_result(client)
    resp = client.get(f"/api/export/{env['result_id']}/0.html")
    assert resp.status == 200
    assert resp.headers["content-type"] == "text/html; charset=utf-8"
    assert resp.headers["content-security-policy"] == EXPORT_CSP
    csp = [v for k, v in resp.header_list if k.lower() == "content-security-policy"]
    assert csp == [EXPORT_CSP]
    assert resp.headers["x-content-type-options"] == "nosniff"
    assert resp.headers["x-frame-options"] == "DENY"


def test_html_export_has_source_paragraph(client):
    env = upload_result(client)
    html = client.get(f"/api/export/{env['result_id']}/0.html").body.decode("utf-8")
    m = re.search(r"<body[^>]*>(.{0,200})", html, re.S)
    assert m and m.group(1).startswith('<p class="source">출처: 사용자 업로드 (월매출.csv)')
    whole = client.get(f"/api/export/{env['result_id']}.html").body.decode("utf-8")
    m = re.search(r"<body[^>]*>(.{0,40})", whole, re.S)
    assert m and m.group(1).startswith('<p class="source">출처: ')


def test_json_export_lang(client):
    env = upload_result(client)
    rid = env["result_id"]
    ko = client.get(f"/api/export/{rid}/0.json").json()
    en = client.get(f"/api/export/{rid}/0.json?hl=en").json()
    assert ko["source"]["line"].startswith("출처: ") and en["source"]["line"].startswith("Source: ")
    assert ko["source"]["kind"] == "upload"
    assert "history" in ko and ko.get("raw_excluded") is None
    whole = client.get(f"/api/export/{rid}.json").json()
    assert whole["source_line"] == env["compare"]["source_line"]
    assert len(whole["source"]) == 2 and len(whole["reports"]) == 2
