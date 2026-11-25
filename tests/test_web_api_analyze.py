"""``POST /api/analyze``, job polling and the analyze job (C.5, C.6, C.7, E.5, E.6, E.10)."""

from __future__ import annotations

import io
import os
import threading
import time

import numpy as np
import pandas as pd
import pytest

from trendai import pipeline
from trendai.data import collect_notices
from trendai.web import jobs, viewmodel
from trendai.web.testing import (TestClient, analyze_and_wait, assert_valid, iter_strings, make_test_app,
                                 write_static_fixture)

SERIES_KEYS = ["schema", "lang", "index", "name", "terms", "slot", "icon", "flags", "source", "hero", "phase", "tiles",
               "chart", "past", "future", "narrative", "caveats", "summary_line", "exports", "notices"]


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


def sample_body(*names, **extra):
    return {"source": "sample", "series": [{"terms": [n]} for n in names], **extra}


def assert_series_vm(vm: dict) -> None:
    assert list(vm) == SERIES_KEYS
    assert_valid(vm, "seriesVM")
    assert len(vm["tiles"]) == 6
    for text in iter_strings(vm):
        assert text not in ("None", "nan", "NaN", "undefined")


def test_sample_detail_end_to_end(client):
    resp = client.post("/api/analyze", json=sample_body("growing"))
    assert resp.status == 202
    acc = resp.json()
    assert set(acc) == {"job_id", "status", "cached", "joined", "queue_position", "poll_url", "result_url"}
    assert acc["cached"] is False and acc["joined"] is False and acc["status"] in ("queued", "running")
    assert acc["poll_url"] == f"/api/jobs/{acc['job_id']}" and acc["result_url"] == f"/api/results/{acc['job_id']}"
    assert resp.headers["cache-control"] == "no-store"
    snap = client.wait_job(acc["job_id"])
    assert_valid(snap, "analyzeJob")
    assert snap["status"] == "done" and snap["view"] == "detail" and snap["result_url"] == acc["result_url"]
    assert snap["poll_after_ms"] is None and snap["compare"] is None
    env = client.get(acc["result_url"]).json()
    assert_valid(env, "resultEnvelope")
    assert env["result_id"] == acc["job_id"] and env["readonly"] is False and env["view"] == "detail"
    assert env["query"] == {"source": "sample", "series": [{"terms": ["growing"]}], "h": 12}
    vm = env["series"][0]
    assert_series_vm(vm)
    assert vm["flags"]["synthetic"] is True
    assert "가상 데이터" in vm["source"]["line"]
    assert vm["slot"] == 1 and vm["icon"] == "circle"
    assert vm["exports"]["csv"] == f"/api/export/{env['result_id']}/0.csv"
    export = client.get(f"/api/export/{env['result_id']}/0.json")
    assert export.status == 200 and export.json()["synthetic"] is True
    en = client.get(f"{acc['result_url']}?hl=en").json()
    assert en["series"][0]["lang"] == "en"


def test_compare_three_samples(client):
    snap, env = analyze_and_wait(client, sample_body("emerging", "fad", "seasonal"))
    assert snap["status"] == "done" and snap["view"] == "compare"
    assert_valid(snap["compare"], "compareVM")
    cmp_ = env["compare"]
    assert_valid(cmp_, "compareVM")
    assert [r["rank"] for r in cmp_["ranking"]] == [1, 2, 3]
    scores = [r["score"]["value"] for r in cmp_["ranking"]]
    assert scores == sorted(scores, reverse=True)
    assert cmp_["slots"] == {"0": 1, "1": 2, "2": 3}
    assert [v["slot"] for v in env["series"]] == [1, 2, 3]
    by_name = {r["name"]: r["index"] for r in cmp_["ranking"]}
    assert by_name == {"태동기형": 0, "반짝유행": 1, "계절형": 2}           # sample titles (G.13), never the ids
    assert env["exports"]["values_csv"] == f"/api/export/{env['result_id']}/values.csv"


def test_progress_since(client, app, monkeypatch):
    steps: list[tuple[str, str]] = []
    real = jobs.JobContext.set_step

    def slow_step(self, step):
        steps.append((self.job.stage, step))
        time.sleep(0.03)
        real(self, step)

    monkeypatch.setattr(jobs.JobContext, "set_step", slow_step)
    app.fetchers.wiki.delay = 0.2
    resp = client.post("/api/analyze", json={"source": "wiki", "series": [{"terms": [t]} for t in ("가", "나", "다")]})
    job_id = resp.json()["job_id"]
    seen: dict[int, int] = {}
    done_counts, stages, current_steps = [], [], []
    since = 0
    while True:
        resp = client.get(f"/api/jobs/{job_id}?since={since}")
        if resp.status == 429:                   # a client backs off on the poll limit (C.13)
            time.sleep(0.2)
            continue
        snap = resp.json()
        assert_valid(snap, "analyzeJob")
        stages.append(snap["stage"])
        done_counts.append(snap["series_done"])
        if snap["current"]:
            current_steps.append(snap["current"]["step"])
        for entry in snap["series"]:
            assert entry["index"] >= since
            seen[entry["index"]] = seen.get(entry["index"], 0) + 1
        indexes = [e["index"] for e in snap["series"]] + [f["index"] for f in snap["failed"]]
        if indexes:
            since = max(since, max(indexes) + 1)
        if snap["status"] in ("done", "failed"):
            break
        time.sleep(0.1)
    assert snap["status"] == "done"
    assert seen == {0: 1, 1: 1, 2: 1}
    assert done_counts == sorted(done_counts)
    order = ["queued", "data", "series", "finish", "done"]
    ranks = [order.index(s) for s in stages]
    assert ranks == sorted(ranks) and "data" in stages and "done" in stages
    assert set(current_steps) <= {"analyze", "forecast", "narrative"}
    per_series = [s for stage, s in steps if stage == "series"]
    assert per_series == ["prepare", "analyze", "forecast", "outlook", "narrative"] * 3
    assert snap["series_total"] == 3 and snap["series_done"] == 3


def test_since_after_failed_series(client):
    resp = client.post("/api/analyze", json={"source": "wiki", "series": [{"terms": [t]} for t in
                                                                          ("인공지능", "없는문서", "기계학습")]})
    job_id = resp.json()["job_id"]
    received: list[int] = []
    since = 0
    for _ in range(2000):
        resp = client.get(f"/api/jobs/{job_id}?since={since}")
        if resp.status == 429:
            time.sleep(0.2)
            continue
        snap = resp.json()
        received += [e["index"] for e in snap["series"]]
        indexes = [e["index"] for e in snap["series"]] + [f["index"] for f in snap["failed"]]
        if indexes:
            since = max(since, max(indexes) + 1)
        if snap["status"] in ("done", "failed"):
            break
        time.sleep(0.05)
    assert snap["status"] == "done"
    assert sorted(received) == [0, 2] and len(received) == len(set(received))
    assert [f["index"] for f in snap["failed"]] == [1]
    assert snap["failed"][0]["name"] == "없는문서"
    assert snap["failed"][0]["error"]["code"] == "wiki_not_found"


@pytest.mark.parametrize("body,code,param", [
    ({"source": "naver", "series": []}, "empty_query", None),
    ({"source": "naver", "series": [{"terms": ["  ", ""]}]}, "empty_query", None),
    ({"source": "naver", "series": [{"terms": [f"t{i}"]} for i in range(6)]}, "too_many_series", None),
    ({"source": "naver", "series": [{"terms": [f"t{i}" for i in range(6)]}]}, "too_many_keywords", None),
    ({"source": "naver", "series": [{"terms": ["가" * 51]}]}, "term_too_long", None),
    ({"source": "wiki", "series": [{"terms": ["a", "b"]}]}, "wiki_group_unsupported", None),
    ({"source": "sample", "series": [{"terms": ["nope"]}]}, "unknown_sample", None),
    ({"source": "naver", "series": [{"terms": ["a"]}], "range": "10y"}, "invalid_param", "range"),
    ({"source": "naver", "series": [{"terms": ["a"]}], "range": "2025-01-01~2024-01-01"}, "invalid_param", "range"),
    ({"source": "sample", "series": [{"terms": ["growing"]}], "h": 0}, "invalid_param", "h"),
    ({"source": "sample", "series": [{"terms": ["growing"]}], "h": 37}, "invalid_param", "h"),
    ({"source": "sample", "series": [{"terms": ["growing"]}], "h": True}, "invalid_param", "h"),
    ({"source": "naver", "series": [{"terms": ["a"]}], "unit": "year"}, "invalid_param", "unit"),
    ({"source": "naver", "series": [{"terms": ["a"]}], "dev": "tv"}, "invalid_param", "dev"),
    ({"source": "naver", "series": [{"terms": ["a"]}], "ages": ["12"]}, "invalid_param", "ages"),
    ({"source": "wiki", "series": [{"terms": ["a"]}], "wl": "../x"}, "invalid_param", "wl"),
    ({"source": "web", "series": [{"terms": ["a"]}]}, "invalid_param", "source"),
    ({"source": "naver", "series": [{"terms": ["a\u0007b"]}]}, "invalid_param", "series"),
    ({"source": "upload", "upload_id": "A" * 22}, "upload_expired", None),
])
def test_validation_errors(client, body, code, param):
    resp = client.post("/api/analyze", json=body)
    err = resp.json()["error"]
    assert err["code"] == code
    assert resp.status == err["status"] == (410 if code == "upload_expired" else 400)
    if param is not None:
        assert err["params"]["param"] == param
    assert_valid(err, "error")
    assert err["message_ko"] and err["message_en"]


def test_validation_messages_have_params(client):
    err = client.post("/api/analyze", json={"source": "naver", "series": [{"terms": [f"t{i}"]} for i in range(6)]}
                      ).json()["error"]
    assert err["params"] == {"max": 5}
    assert err["message_ko"] == "한 번에 5개까지 비교할 수 있어요. 하나를 빼고 추가해 주세요."
    assert [f["action"] for f in err["fixes"]] == ["remove_series"]
    err = client.post("/api/analyze", json={"source": "naver", "series": [{"terms": [f"k{i}" for i in range(6)]}]}
                      ).json()["error"]
    assert err["params"] == {"max": 5, "group": "k0"}


def test_display_names_unique(client):
    body = {"source": "naver", "series": [{"terms": ["a + b"]}, {"terms": ["a", "b"]}, {"terms": ["a", "c"]}],
            "range": "1y", "unit": "month"}
    snap, env = analyze_and_wait(client, body)
    assert snap["status"] == "done"
    assert [v["name"] for v in env["series"]] == ["a + b", "a + b (2)", "a + c"]
    assert [v["terms"] for v in env["series"]] == [["a + b"], ["a", "b"], ["a", "c"]]
    for row in env["compare"]["ranking"]:
        assert env["series"][row["index"]]["name"] == row["name"]


def test_json_required_415(client):
    resp = client.post("/api/analyze", data=b"source=sample",
                       headers={"Content-Type": "application/x-www-form-urlencoded"})
    assert resp.status == 415 and resp.json()["error"]["code"] == "unsupported_media_type"
    bad = client.post("/api/analyze", data=b"{not json", headers={"Content-Type": "application/json"})
    assert bad.status == 400 and bad.json()["error"]["code"] == "bad_request"
    arr = client.post("/api/analyze", data=b"[1]", headers={"Content-Type": "application/json"})
    assert arr.status == 400


def test_body_limit_413_before_read(app):
    class Exploding(io.BytesIO):
        def read(self, *a):
            raise AssertionError("the body must not be read")

    captured = {}
    environ = {"REQUEST_METHOD": "POST", "PATH_INFO": "/api/analyze", "QUERY_STRING": "", "SERVER_NAME": "t",
               "SERVER_PORT": "80", "REMOTE_ADDR": "127.0.0.1", "HTTP_HOST": "t", "wsgi.url_scheme": "http",
               "wsgi.input": Exploding(b"0123456789"), "CONTENT_TYPE": "application/json", "CONTENT_LENGTH": "70000"}
    body = b"".join(app(environ, lambda status, headers: captured.update(status=status)))
    assert captured["status"].startswith("413")
    import json

    assert json.loads(body)["error"]["code"] == "body_too_large"
    environ.pop("CONTENT_LENGTH")
    environ["wsgi.input"] = io.BytesIO(b"{}")
    body = b"".join(app(environ, lambda status, headers: captured.update(status=status)))
    assert captured["status"].startswith("411") and json.loads(body)["error"]["code"] == "length_required"


def test_cached_result_reuse(client, app):
    body = {"source": "wiki", "series": [{"terms": ["인공지능"]}]}
    snap, env = analyze_and_wait(client, body)
    again = client.post("/api/analyze", json=body)
    assert again.status == 200
    acc = again.json()
    assert acc["cached"] is True and acc["status"] == "done" and acc["job_id"] == env["result_id"]
    assert len(app.fetchers.wiki.calls) == 1


def test_inflight_join(client, app):
    app.fetchers.wiki.delay = 1.0
    body = {"source": "wiki", "series": [{"terms": ["인공지능"]}]}
    first = client.post("/api/analyze", json=body).json()
    _wait_for(lambda: app.fetchers.wiki.calls)          # the job is inside the (slow) fetch
    second = client.post("/api/analyze", json=body)
    assert second.status == 202
    acc = second.json()
    assert acc["joined"] is True and acc["job_id"] == first["job_id"] and acc["cached"] is False
    d1 = client.delete(f"/api/jobs/{first['job_id']}")
    assert d1.status == 202 and d1.json()["status"] in ("queued", "running")
    d2 = client.delete(f"/api/jobs/{first['job_id']}")
    assert d2.status == 202 and d2.json() == {"job_id": first["job_id"], "status": "cancelling"}
    snap = client.wait_job(first["job_id"])
    assert snap["status"] == "cancelled"
    assert len(app.fetchers.wiki.calls) == 1
    assert client.delete(f"/api/jobs/{first['job_id']}").status == 200


def _wait_for(predicate, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "condition not reached"
        time.sleep(0.01)


def test_cancel_running_job(client, app, tmp_path):
    app.fetchers.wiki.delay = 0.6
    acc = client.post("/api/analyze", json={"source": "wiki", "series": [{"terms": ["가"]}, {"terms": ["나"]}]}).json()
    _wait_for(lambda: app.fetchers.wiki.calls)
    resp = client.delete(f"/api/jobs/{acc['job_id']}")
    assert resp.status == 202 and resp.json()["status"] == "cancelling"
    snap = client.wait_job(acc["job_id"])
    assert snap["status"] == "cancelled" and snap["result_url"] is None
    assert not [n for n in os.listdir(os.path.join(app.config.data_dir, "results")) if not n.startswith(".")]
    assert client.get(f"/api/results/{acc['job_id']}").status == 404
    assert client.delete(f"/api/jobs/{'Z' * 22}").status == 404


def test_queue_full_503(make):
    app = make(workers=1, max_queued=1)
    app.fetchers.wiki.delay = 1.0
    c = TestClient(app)
    a = c.post("/api/analyze", json={"source": "wiki", "series": [{"terms": ["가"]}]})
    b = c.post("/api/analyze", json={"source": "wiki", "series": [{"terms": ["나"]}]})
    assert (a.status, b.status) == (202, 202)
    assert b.json()["queue_position"] == 1
    full = c.post("/api/analyze", json={"source": "wiki", "series": [{"terms": ["다"]}]})
    assert full.status == 503
    assert full.headers["retry-after"] == "10"
    err = full.json()["error"]
    assert err["code"] == "server_busy" and err["retry_after"] == 10
    assert err["fixes"][0] == {"action": "retry_in", "label_ko": "지금 다시", "label_en": "Retry now",
                               "params": {"seconds": 10}}
    for acc in (a.json(), b.json()):
        c.delete(f"/api/jobs/{acc['job_id']}")


def test_analyze_rate_limit_429(client):
    statuses = [client.post("/api/analyze", json=sample_body("growing")).status for _ in range(10)]
    assert 429 not in statuses
    resp = client.post("/api/analyze", json=sample_body("growing"))
    assert resp.status == 429
    assert int(resp.headers["retry-after"]) >= 1
    err = resp.json()["error"]
    assert err["code"] == "rate_limited" and err["retry_after"] == int(resp.headers["retry-after"])
    assert err["params"] == {"retry_after": err["retry_after"]}
    assert TestClient(client.app, remote_addr="10.0.0.9").post("/api/analyze", json=sample_body("fad")).status == 202


def test_job_timeout(make):
    app = make(job_timeout_s=0.1)
    app.fetchers.wiki.delay = 0.5
    c = TestClient(app)
    snap, env = analyze_and_wait(c, {"source": "wiki", "series": [{"terms": ["가"]}, {"terms": ["나"]}]})
    assert snap["status"] == "failed" and env is None
    assert snap["error"]["code"] == "job_timeout" and snap["error"]["status"] == 504


def test_job_after_restart_reads_store(tmp_path, static):
    app = make_test_app(tmp_path / "data", static_dir=static)
    c = TestClient(app)
    snap, env = analyze_and_wait(c, sample_body("growing", "fad"))
    app.close(timeout=5)
    again = make_test_app(tmp_path / "data", static_dir=static)
    try:
        resp = TestClient(again).get(f"/api/jobs/{env['result_id']}?since=0&hl=en")
        assert resp.status == 200
        body = resp.json()
        assert_valid(body, "analyzeJob")
        assert body["status"] == "done" and body["view"] == "compare" and body["poll_after_ms"] is None
        assert [e["index"] for e in body["series"]] == [0, 1]
        assert body["series"][0]["vm"]["lang"] == "en"
        assert body["compare"]["count"] == 2
        assert TestClient(again).get(f"/api/jobs/{'Q' * 22}").json()["error"]["code"] == "job_not_found"
    finally:
        again.close(timeout=5)


def test_all_series_fail(client):
    snap, env = analyze_and_wait(client, {"source": "wiki", "series": [{"terms": ["없는문서"]}]})
    assert snap["status"] == "failed" and env is None
    err = snap["error"]
    assert err["code"] == "wiki_not_found" and err["status"] == 404
    assert err["params"] == {"title": "없는문서"}
    assert [f["action"] for f in err["fixes"]] == ["wiki_title_help", "use_naver"]


def test_use_naver_fix_dropped_when_unconfigured(make):
    c = TestClient(make(naver=False))
    snap, _ = analyze_and_wait(c, {"source": "wiki", "series": [{"terms": ["없는문서"]}]})
    assert [f["action"] for f in snap["error"]["fixes"]] == ["wiki_title_help"]


def _gappy_csv() -> bytes:
    rows = ["날짜,갭값"]
    for i in range(30):
        if i in (3, 4, 7, 11, 15, 16, 20, 24):      # 8 of 30 months missing (>= 20 %: gaps_interpolated)
            continue
        rows.append(f"{2022 + i // 12}-{i % 12 + 1:02d},{100 + i * 3 + (i % 4)}")
    return ("\n".join(rows) + "\n").encode()


def _duplicate_csv() -> bytes:
    rows = ["날짜,중복값"]
    for i in range(30):
        rows.append(f"{2022 + i // 12}-{i % 12 + 1:02d},{200 + i * 2 + (i % 3)}")
        if i in (5, 10, 20):                         # 3 of 33 rows repeat a month (>= 5 %: duplicates_merged)
            rows.append(f"{2022 + i // 12}-{i % 12 + 1:02d},{205 + i * 2}")
    return ("\n".join(rows) + "\n").encode()


@pytest.mark.filterwarnings("ignore::trendai.data.DataWarning")
def test_notices_isolated_between_threads(client):
    up_a = client.upload("gaps.csv", _gappy_csv()).json()
    up_b = client.upload("dups.csv", _duplicate_csv()).json()
    stop = threading.Event()
    captured: list[list[str]] = []

    def hammer():
        short = pd.Series(np.linspace(1, 9, 9), index=pd.date_range("2024-01-01", periods=9, freq="MS"))
        while not stop.is_set():
            with collect_notices() as notes:
                pipeline.run(short, horizon=60)
            captured.append([getattr(n, "code", "") for n in notes])
            pipeline.run(short, horizon=60)         # outside a block: a plain warnings.warn

    t = threading.Thread(target=hammer, daemon=True)
    t.start()
    try:
        results = {}

        def run(name, up):
            snap, env = analyze_and_wait(TestClient(client.app, remote_addr=f"10.1.0.{len(name)}"),
                                         {"source": "upload", "upload_id": up["upload_id"]})
            results[name] = env

        threads = [threading.Thread(target=run, args=("a", up_a)), threading.Thread(target=run, args=("bb", up_b))]
        for th in threads:
            th.start()
        for th in threads:
            th.join(60)
    finally:
        stop.set()
        t.join(30)
    codes_a = [n["code"] for n in results["a"]["series"][0]["notices"]]
    codes_b = [n["code"] for n in results["bb"]["series"][0]["notices"]]
    assert "gaps_interpolated" in codes_a and "duplicates_merged" not in codes_a
    assert "duplicates_merged" in codes_b and "gaps_interpolated" not in codes_b
    assert "horizon_long" not in codes_a + codes_b
    assert captured and all("horizon_long" in c for c in captured)


def test_vm_schema_bump_invalidates_cache(client, monkeypatch):
    snap, env = analyze_and_wait(client, sample_body("growing"))
    monkeypatch.setattr(viewmodel, "WEB_VM_SCHEMA", viewmodel.WEB_VM_SCHEMA + 1)
    resp = client.post("/api/analyze", json=sample_body("growing"))
    assert resp.status == 202 and resp.json()["cached"] is False
    assert client.get(f"/api/results/{env['result_id']}").json()["error"]["code"] == "result_not_found"
    new = client.wait_job(resp.json()["job_id"])
    assert new["status"] == "done"


def test_result_not_found_and_headers(client):
    resp = client.get(f"/api/results/{'N' * 22}")
    assert resp.status == 404
    err = resp.json()["error"]
    assert err["code"] == "result_not_found"
    assert [f["action"] for f in err["fixes"]] == ["reanalyze", "go_home"]
    assert resp.headers["cache-control"] == "no-store"
