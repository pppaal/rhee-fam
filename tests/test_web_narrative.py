"""``POST /api/narrative`` and the AI narrative job (C.6, C.11, D.2.9) with ``testing.fake_llm``."""

from __future__ import annotations

import os
import time

import pytest

from trendai.narrative import NarrativeUnavailable
from trendai.web import narrative_ai
from trendai.web.testing import (TestClient, analyze_and_wait, assert_valid, fake_llm, make_test_app, monthly_csv,
                                 write_static_fixture)


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


def sample_result(client, *names):
    snap, env = analyze_and_wait(client, {"source": "sample", "series": [{"terms": [n]} for n in names]})
    assert snap["status"] == "done"
    return env


def ai(client, rid, series=0, lang="ko", **extra):
    return client.post("/api/narrative", json={"result_id": rid, "series": series, "lang": lang, **extra})


def test_ai_job_success(client, app):
    env = sample_result(client, "growing")
    rid = env["result_id"]
    assert env["series"][0]["narrative"]["ai_allowed"] is True
    seen = []
    app.fetchers.llm = lambda payload, lang: (seen.append((payload, lang)), time.sleep(0.2), fake_llm(payload, lang))[2]
    resp = ai(client, rid)
    assert resp.status == 202
    acc = resp.json()
    assert set(acc) == {"job_id", "status", "poll_url"} and acc["poll_url"] == f"/api/jobs/{acc['job_id']}"
    running = client.get(acc["poll_url"]).json()
    assert_valid(running, "narrativeJob")
    assert running["kind"] == "narrative" and running["narrative"] is None
    assert running["status"] in ("queued", "running") and running["poll_after_ms"] == 500
    done = client.wait_job(acc["job_id"])
    assert_valid(done, "narrativeJob")
    assert done["status"] == "done" and done["poll_after_ms"] is None and done["error"] is None
    nar = done["narrative"]
    assert nar["source"] == "llm" and nar["badge"] == "AI 해석 · Claude" and nar["model"] == "claude-opus-5-5"
    assert nar["markdown"].startswith("**AI 테스트 해석** — 성장형")
    assert "## 주의할 점" not in nar["markdown"] and nar["caveats_markdown"] == "- 테스트 주의입니다."
    assert seen[0][1] == "ko" and seen[0][0]["series"]["name"] == "성장형"     # the VM's name, never the id
    assert os.path.isfile(os.path.join(app.config.data_dir, "results", rid, "ai", "0-ko.json"))
    later = client.get(f"/api/results/{rid}").json()
    stored = later["series"][0]["narrative"]["ai"]
    assert stored["markdown"] == nar["markdown"] and stored["created_at"].endswith("Z")
    assert client.get(f"/api/results/{rid}?hl=en").json()["series"][0]["narrative"]["ai"] is None
    again = ai(client, rid)
    assert again.status == 200 and again.json()["status"] == "done"
    assert client.get(again.json()["poll_url"]).json()["narrative"]["markdown"] == nar["markdown"]
    assert len(seen) == 1


def test_ai_unavailable(make):
    app = make(ai=False)
    c = TestClient(app)
    env = sample_result(c, "growing")
    nar = env["series"][0]["narrative"]
    assert nar["ai_allowed"] is False and nar["ai_block_reason"] == "ai_unavailable"
    assert nar["ai_block_text"] == "지금은 AI 해석을 쓸 수 없어요."
    resp = ai(c, env["result_id"])
    assert resp.status == 503 and resp.json()["error"]["code"] == "ai_unavailable"


def test_ai_rate_limit(client):
    env = sample_result(client, "growing", "fad")
    rid = env["result_id"]
    statuses = [ai(client, rid, k, lang).status for k, lang in ((0, "ko"), (0, "en"), (1, "ko"))]
    assert statuses == [202, 202, 202]
    resp = ai(client, rid, 1, "en")
    assert resp.status == 429
    err = resp.json()["error"]
    assert err["code"] == "ai_rate_limited" and err["params"] == {"limit": 10}
    assert int(resp.headers["retry-after"]) >= 1
    assert err["message_ko"] == "AI 해석은 한 시간에 10번까지 만들 수 있어요."


def test_ai_daily_cap(make):
    app = make(ai_daily_limit=1)
    c = TestClient(app)
    env = sample_result(c, "growing")
    assert ai(c, env["result_id"], 0, "ko").status == 202
    resp = ai(c, env["result_id"], 0, "en")
    assert resp.status == 503
    err = resp.json()["error"]
    assert err["code"] == "ai_unavailable" and err["params"] == {"reason": "daily_limit"}


def test_ai_failure_maps_502(client, app):
    env = sample_result(client, "growing")

    def broken(payload, lang):
        raise NarrativeUnavailable("Claude API 인증 정보를 찾을 수 없습니다.")

    app.fetchers.llm = broken
    job = client.wait_job(ai(client, env["result_id"]).json()["job_id"])
    assert job["status"] == "failed" and job["narrative"] is None
    err = job["error"]
    assert (err["code"], err["status"]) == ("ai_failed", 502)
    assert err["message_ko"] == "AI 해석을 만들지 못해 규칙 기반 해석을 보여 드려요."
    assert "인증" not in err["message_ko"]
    assert client.get(f"/api/results/{env['result_id']}").json()["series"][0]["narrative"]["ai"] is None


def test_ai_readonly_on_share(client):
    up = client.upload("a.csv", monthly_csv({"v": [10 + i * 1.5 + (i % 4) for i in range(30)]})).json()
    snap, env = analyze_and_wait(client, {"source": "upload", "upload_id": up["upload_id"]})
    share = client.post("/api/share", json={"result_id": env["result_id"]}).json()
    shared = client.get(f"/api/shared/{share['share_id']}").json()
    nar = shared["series"][0]["narrative"]
    assert (nar["ai_allowed"], nar["ai_block_reason"]) == (False, "shared_readonly")
    assert nar["ai_block_text"] == "공유 링크에서는 AI 해석을 새로 만들 수 없어요."
    resp = ai(client, env["result_id"], share_id=share["share_id"])
    assert resp.status == 403 and resp.json()["error"]["code"] == "ai_readonly"
    resp = client.post("/api/narrative", json={"share_id": None})
    assert resp.status == 403 and resp.json()["error"]["code"] == "ai_readonly"


def test_narrative_request_validation(client, app):
    env = sample_result(client, "growing")
    rid = env["result_id"]
    resp = ai(client, rid, lang="../x")
    assert resp.status == 400
    err = resp.json()["error"]
    assert err["code"] == "invalid_param" and err["params"] == {"param": "lang"}
    for bad_lang in ("KO", "ja", None, ["ko"]):
        assert ai(client, rid, lang=bad_lang).json()["error"]["code"] == "invalid_param"
    for series in (True, 0.0, "0", -1, 99, None):
        resp = ai(client, rid, series=series)
        assert resp.status == 404 and resp.json()["error"]["code"] == "not_found", series
    for bad_id in ("../../etc/passwd", "x" * 22 + "y", 5, None):
        resp = client.post("/api/narrative", json={"result_id": bad_id, "series": 0, "lang": "ko"})
        assert resp.status == 404 and resp.json()["error"]["code"] == "result_not_found"
    assert ai(client, "Q" * 22).json()["error"]["code"] == "result_not_found"
    written = [os.path.relpath(os.path.join(d, f), app.config.data_dir)
               for d, _dirs, files in os.walk(app.config.data_dir) for f in files]
    assert not [p for p in written if p.endswith("x") or "ai" + os.sep in p]     # nothing escaped, nothing stored
    assert not os.path.exists(os.path.join(app.config.data_dir, "results", rid, "ai"))


def test_ai_job_timeout(client, app, monkeypatch):
    env = sample_result(client, "growing")
    monkeypatch.setattr(narrative_ai, "AI_JOB_TIMEOUT_S", 0.2)
    app.fetchers.llm = lambda payload, lang: (time.sleep(0.5), fake_llm(payload, lang))[1]
    job = client.wait_job(ai(client, env["result_id"]).json()["job_id"])
    assert job["status"] == "failed" and job["error"]["code"] == "ai_failed"
    assert client.get(f"/api/results/{env['result_id']}").json()["series"][0]["narrative"]["ai"] is None
