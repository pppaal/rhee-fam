"""Naver flows (C.5, C.10, C.13, E.6, E.9, E.10) with ``testing.FakeNaver``: one call per comparison, wire names,
cache and single-flight, budget and per-IP limits, connector error mapping, raw-ratio stripping."""

from __future__ import annotations

import datetime as dt

import pytest

from trendai.web.budget import NaverBudget
from trendai.web.store import Store
from trendai.web.testing import FakeNaver, FrozenClock, TestClient, analyze_and_wait, assert_valid, make_test_app, \
    write_static_fixture

T0 = dt.datetime(2026, 11, 25, 0, 12, tzinfo=dt.timezone.utc)      # Wednesday 09:12 KST


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


def naver(*groups, **extra):
    body = {"source": "naver", "series": [{"terms": list(g)} for g in groups], "range": "1y", "unit": "month"}
    body.update(extra)
    return body


def test_one_call_for_compare(client, app):
    snap, env = analyze_and_wait(client, naver(["선크림", "자외선차단제"], ["패딩"]))
    assert snap["status"] == "done"
    calls = app.fetchers.naver.calls
    assert len(calls) == 1
    assert calls[0]["groups"] == [("g1", ("선크림", "자외선차단제")), ("g2", ("패딩",))]
    assert env["query"]["series"] == [{"terms": ["선크림", "자외선차단제"]}, {"terms": ["패딩"]}]
    assert [v["name"] for v in env["series"]] == ["선크림", "패딩"]
    assert env["series"][0]["source"]["kind"] == "naver"
    assert env["series"][0]["source"]["label"] == "네이버 데이터랩 검색어트렌드(NAVER API HUB)"


def test_long_display_names(client, app):
    groups = [[f"{chr(0xAC00 + g * 40 + k)}" * 50 for k in range(5)] for g in range(5)]
    snap, env = analyze_and_wait(client, naver(*groups))
    assert snap["status"] == "done", snap["error"]
    sent = app.fetchers.naver.calls[0]["groups"]
    assert [name for name, _kw in sent] == ["g1", "g2", "g3", "g4", "g5"]
    assert [list(kw) for _n, kw in sent] == groups
    assert [v["name"] for v in env["series"]] == [g[0] for g in groups]


def test_cache_hit_no_call_and_notice(client, app):
    analyze_and_wait(client, naver(["선크림"]))
    snap, env = analyze_and_wait(client, naver(["선크림"], h=6))      # another result, the same Naver body
    assert snap["status"] == "done"
    assert len(app.fetchers.naver.calls) == 1
    cached = [n for n in env["notices"] if n["code"] == "naver_cached"]
    assert len(cached) == 1 and cached[0]["level"] == "info" and cached[0]["series_index"] is None
    assert "저장된 네이버 조회 결과(" in cached[0]["text"]
    en = client.get(f"/api/results/{env['result_id']}?hl=en").json()
    assert en["notices"][0]["text"].startswith("Showing stored Naver data (as of ")


def test_fetch_single_flight(client, app):
    app.fetchers.naver.delay = 0.6
    a = client.post("/api/analyze", json=naver(["선크림"])).json()
    b = client.post("/api/analyze", json=naver(["선크림"], h=6)).json()
    assert a["job_id"] != b["job_id"]
    assert client.wait_job(a["job_id"])["status"] == "done"
    assert client.wait_job(b["job_id"])["status"] == "done"
    assert len(app.fetchers.naver.calls) == 1
    assert app.naver_budget.state()[0] == 1


def test_cache_ttl_24h_recent_21d_past(make):
    clock = FrozenClock(T0)
    app = make(clock=clock)
    c = TestClient(app)
    recent = naver(["선크림"], range="1y")
    past = naver(["선크림"], range="2022-01-01~2023-12-31")
    analyze_and_wait(c, recent)
    analyze_and_wait(c, past)
    assert len(app.fetchers.naver.calls) == 2
    clock.advance(23 * 3600)
    analyze_and_wait(c, {**recent, "h": 6})
    analyze_and_wait(c, {**past, "h": 6})
    assert len(app.fetchers.naver.calls) == 2                 # both still cached
    clock.advance(2 * 3600)                                   # +25 h: the recent query expired, the past one not
    analyze_and_wait(c, {**recent, "h": 3})
    analyze_and_wait(c, {**past, "h": 3})
    assert len(app.fetchers.naver.calls) == 3
    clock.advance(21 * 86400)                                 # past the 21-day limit
    analyze_and_wait(c, {**past, "h": 12})
    assert len(app.fetchers.naver.calls) == 4


def test_budget_reserve_and_block(make):
    app = make(naver_monthly_limit=10)
    for i in range(9):
        c = TestClient(app, remote_addr=f"10.0.0.{i}")
        snap, _ = analyze_and_wait(c, naver([f"검색어{i}"]))
        assert snap["status"] == "done"
    assert app.naver_budget.state() == (9, 10, "blocked")
    resp = TestClient(app, remote_addr="10.0.1.1").post("/api/analyze", json=naver(["새검색어"]))
    assert resp.status == 503
    err = resp.json()["error"]
    assert err["code"] == "naver_budget_exhausted"
    assert [f["action"] for f in err["fixes"]] == ["use_wiki", "upload_file"]
    snap, env = analyze_and_wait(TestClient(app, remote_addr="10.0.1.2"), naver(["검색어3"], h=6))
    assert snap["status"] == "done"                           # cache hits are still served
    assert len(app.fetchers.naver.calls) == 9
    cfg = TestClient(app).get("/api/config").json()["sources"]["naver"]
    assert cfg == {"available": False, "reason": "budget_exhausted", "endpoint": "hub", "budget_state": "blocked"}


def test_budget_month_rollover_kst(tmp_path):
    clock = FrozenClock(dt.datetime(2026, 11, 30, 14, 59, tzinfo=dt.timezone.utc))   # 23:59 KST, November
    store = Store(tmp_path / "s", clock)
    store.ensure()
    budget = NaverBudget(store, clock, 100)
    for _ in range(85):
        budget.reserve()
    assert budget.state() == (85, 100, "low")
    clock.advance(60)                                          # 15:00Z = 2026-12-01 00:00 KST
    assert budget.state() == (0, 100, "ok")
    budget.reserve()
    assert budget.count() == 1
    assert store.budget_read("naver-2026-11")["count"] == 85
    assert store.budget_read("naver-2026-12") == {"month": "2026-12", "count": 1, "updated_at": "2026-11-30T15:00:00Z"}


def test_naver_ip_limit(client, app):
    first = None
    for i in range(5):
        resp = client.post("/api/analyze", json=naver([f"단어{i}"]))
        assert resp.status == 202
        first = first or resp.json()["job_id"]
    resp = client.post("/api/analyze", json=naver(["단어9"]))
    assert resp.status == 429
    err = resp.json()["error"]
    assert err["code"] == "naver_rate_limited" and err["params"] == {"limit": 20}
    assert int(resp.headers["retry-after"]) >= 1 and err["retry_after"] == int(resp.headers["retry-after"])
    assert [f["action"] for f in err["fixes"]] == ["use_wiki", "retry_in"]
    assert "한 시간에 20번까지" in err["message_ko"]
    assert client.wait_job(first)["status"] == "done"
    cached = client.post("/api/analyze", json=naver(["단어0"], h=6))     # a Naver cache hit costs no token
    assert cached.status == 202


def test_naver_ip_limit_low_budget(make):
    app = make(naver_monthly_limit=100)
    for _ in range(80):
        app.naver_budget.reserve()
    c = TestClient(app)
    for i in range(5):
        assert c.post("/api/analyze", json=naver([f"낮음{i}"])).status == 202
    err = c.post("/api/analyze", json=naver(["낮음9"])).json()["error"]
    assert err["code"] == "naver_rate_limited" and err["params"] == {"limit": 5}


def test_naver_unconfigured_503(make):
    c = TestClient(make(naver=False))
    resp = c.post("/api/analyze", json=naver(["선크림"]))
    assert resp.status == 503
    err = resp.json()["error"]
    assert err["code"] == "naver_unconfigured" and err["retry_after"] is None
    assert [f["action"] for f in err["fixes"]] == ["use_wiki", "upload_file"]
    assert err["message_ko"] == "네이버 검색어 트렌드는 아직 연결되지 않았어요."


@pytest.mark.parametrize("mode,code,status", [
    ("auth", "naver_auth", 502), ("rate", "naver_rate", 429), ("server", "naver_server", 502),
    ("network", "source_unreachable", 502), ("timeout", "source_timeout", 504), ("bad", "source_bad_response", 502),
])
def test_connector_error_mapping(app, mode, code, status):
    app.fetchers.naver.fail_mode = mode
    snap, env = analyze_and_wait(TestClient(app), naver([f"오류{mode}"]))
    assert snap["status"] == "failed" and env is None
    err = snap["error"]
    assert (err["code"], err["status"]) == (code, status)
    assert_valid(err, "error")
    if code == "naver_rate":
        assert err["retry_after"] == 30 and err["params"] == {"retry_after": 30}
    assert "HTTP" not in err["message_ko"] and "URLError" not in err["message_en"]


def test_keyword_triggers_of_fake(client):
    snap, _ = analyze_and_wait(client, naver(["에러"]))
    assert snap["error"]["code"] == "naver_server"
    snap, _ = analyze_and_wait(client, naver(["인증오류"]))
    assert snap["error"]["code"] == "naver_auth"


def test_ai_blocked_for_naver(client):
    snap, env = analyze_and_wait(client, naver(["선크림"]))
    nar = env["series"][0]["narrative"]
    assert nar["ai_allowed"] is False and nar["ai_block_reason"] == "naver_terms"
    assert nar["ai_block_text"] == "네이버 데이터는 이용약관에 따라 AI로 보내지 않아요."
    resp = client.post("/api/narrative", json={"result_id": env["result_id"], "series": 0, "lang": "ko"})
    assert resp.status == 403 and resp.json()["error"]["code"] == "ai_naver_blocked"
    import os

    assert not [n for n in os.listdir(os.path.join(client.app.config.data_dir, "results", env["result_id"]))
                if n.startswith("llm-")]


def test_exports_strip_raw(client):
    snap, env = analyze_and_wait(client, naver(["선크림"], ["패딩"]))
    rid = env["result_id"]
    csv = client.get(f"/api/export/{rid}/0.csv")
    assert csv.status == 200
    text = csv.body.decode("utf-8-sig")
    lines = text.split("\r\n")
    assert lines[2].startswith("날짜,추세,예측값") and "실제값" not in text
    assert lines[1].endswith("네이버 원본 값(ratio)은 이용약관상 포함하지 않았어요")
    data = client.get(f"/api/export/{rid}/0.json").json()
    assert "history" not in data and data["raw_excluded"] is True
    assert "last_value" not in data["metrics"] and "peak_value" not in data["metrics"]
    assert data["source"]["kind"] == "naver"
    whole = client.get(f"/api/export/{rid}.json").json()
    assert all("history" not in r for r in whole["reports"])
    for path in (f"/api/export/{rid}/0.html", f"/api/export/{rid}.html", f"/api/export/{rid}/values.csv"):
        resp = client.get(path)
        assert resp.status == 403 and resp.json()["error"]["code"] == "export_naver_raw", path
    assert env["exports"]["values_csv"] is None and env["exports"]["html"] is None
    assert env["exports"]["unavailable"]["values_csv"]["code"] == "export_naver_raw"
    assert env["series"][0]["exports"]["html"] is None


def test_naver_index_scale_disabled(client):
    snap, env = analyze_and_wait(client, naver(["선크림"], ["패딩"]))
    scale = env["compare"]["scale"]
    assert scale["allowed"]["index"] is False
    assert scale["disabled_reason"]["index"] == "같은 조회 안에서 이미 같은 척도예요."
    assert scale["index_base"] is None and scale["default"] == "raw"


class _FilledNaver(FakeNaver):
    def __call__(self, *args, **kwargs):
        result = super().__call__(*args, **kwargs)
        result.filled = {name: (3 if name == "g1" else 0) for name in result.series}
        return result


def test_low_resolution_and_missing_notices(app):
    app.fetchers.naver = _FilledNaver()
    snap, env = analyze_and_wait(TestClient(app), naver(["선크림"], ["저검색어"]))
    first, low = env["series"]
    filled = [n for n in first["notices"] if n["code"] == "naver_missing_filled"]
    assert len(filled) == 1 and filled[0]["level"] == "warning" and filled[0]["series_index"] == 0
    assert "기간 3개는 0으로 처리했어요" in filled[0]["text"]
    notes = [n for n in low["notices"] if n["code"] == "naver_low_resolution"]
    assert len(notes) == 1 and low["flags"]["low_resolution"] is True
    assert notes[0]["fixes"][0]["action"] == "reanalyze_single"
    assert notes[0]["fixes"][0]["params"] == {"term": "%EC%A0%80%EA%B2%80%EC%83%89%EC%96%B4"}
    assert first["flags"]["low_resolution"] is False


def test_period_snapping_used(make):
    app = make(clock=FrozenClock(T0))
    analyze_and_wait(TestClient(app), {"source": "naver", "series": [{"terms": ["선크림"]}], "range": "5y"})
    call = app.fetchers.naver.calls[0]
    assert call["time_unit"] == "week"
    assert (call["start"], call["end"]) == (dt.date(2021, 11, 29), dt.date(2026, 11, 22))
    assert call["start"].weekday() == 0 and call["end"].weekday() == 6


def test_filters_passed_and_echoed(client, app):
    snap, env = analyze_and_wait(client, naver(["선크림"], dev="mo", gender="f", ages=["5", "4", "4"]))
    call = app.fetchers.naver.calls[0]
    assert (call["device"], call["gender"], call["ages"]) == ("mo", "f", ("4", "5"))
    q = env["query"]
    assert (q["dev"], q["gender"], q["ages"], q["unit"]) == ("mo", "f", ["4", "5"], "month")
    assert env["series"][0]["source"]["filters_label"] == "모바일·여성·25–34세"
