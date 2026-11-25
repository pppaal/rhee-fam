"""Security rules (C.0, C.13, E.3, E.7, E.11, E.12, I): origin check, proxy headers, per-job poll buckets, secrets,
id validation, no tracebacks, and no ``warnings`` state changes in the web package."""

from __future__ import annotations

import datetime as dt
import json
import logging
import pathlib
import re

import pytest

from trendai import connectors
from trendai.web import api, create_app
from trendai.web.config import WebConfig
from trendai.web.http import Request
from trendai.web.testing import FrozenClock, TestClient, analyze_and_wait, make_test_app, write_static_fixture

SAMPLE = {"source": "sample", "series": [{"terms": ["growing"]}]}


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


def test_cross_origin_post_403(make):
    c = TestClient(make())
    for method, path in (("POST", "/api/analyze"), ("POST", "/api/share"), ("POST", "/api/upload"),
                         ("DELETE", f"/api/jobs/{'A' * 22}"), ("DELETE", f"/api/share/{'A' * 22}")):
        resp = c.request(method, path, json=SAMPLE, headers={"Origin": "https://evil.example"})
        assert resp.status == 403, path
        assert resp.json()["error"]["code"] == "cross_origin"
    assert c.post("/api/analyze", json=SAMPLE, headers={"Origin": "null"}).status == 403


def test_same_origin_post_ok(make):
    c = TestClient(make())
    assert c.post("/api/analyze", json=SAMPLE, headers={"Origin": "http://testserver"}).status == 202
    assert c.post("/api/analyze", json=SAMPLE, headers={"Origin": "http://testserver:80"}).status in (200, 202)
    assert c.post("/api/analyze", json=SAMPLE).status in (200, 202)          # no Origin header at all
    assert c.get("/api/config", headers={"Origin": "https://evil.example"}).status == 200   # GET is not checked


def test_origin_behind_proxy(make):
    proxied = {"X-Forwarded-Proto": "https", "X-Forwarded-Host": "trendai.example",
               "Origin": "https://trendai.example"}
    c = TestClient(make(trust_proxy=True))
    assert c.post("/api/analyze", json=SAMPLE, headers=proxied).status == 202
    many = {**proxied, "X-Forwarded-Proto": "http, https", "X-Forwarded-Host": "evil.example, trendai.example"}
    assert c.post("/api/analyze", json=SAMPLE, headers=many).status in (200, 202)
    plain = TestClient(make())
    assert plain.post("/api/analyze", json=SAMPLE, headers=proxied).status == 403
    public = TestClient(make(public_url="https://trendai.example"))
    assert public.post("/api/analyze", json=SAMPLE, headers={"Origin": "https://trendai.example"}).status == 202
    assert public.post("/api/analyze", json=SAMPLE, headers={"Origin": "http://testserver"}).status == 403
    port = TestClient(make(public_url="https://trendai.example:443"))
    assert port.post("/api/analyze", json=SAMPLE, headers={"Origin": "https://trendai.example"}).status == 202


def test_poll_bucket_per_job(make):
    clock = FrozenClock(dt.datetime(2026, 11, 25, 0, 12, tzinfo=dt.timezone.utc))
    app = make(clock=clock)
    c = TestClient(app)
    a = c.post("/api/analyze", json=SAMPLE).json()["job_id"]
    b = c.post("/api/analyze", json={"source": "sample", "series": [{"terms": ["fad"]}]}).json()["job_id"]
    for _ in range(60):                      # 2 polls per second per job for 30 s from one IP
        for jid in (a, a, b, b):
            assert c.get(f"/api/jobs/{jid}").status == 200
        clock.advance(1.0)
    for _ in range(80):                      # a runaway poller is still limited
        last = c.get(f"/api/jobs/{a}")
    assert last.status == 429


class _NaverAuthFailSession:
    def __init__(self) -> None:
        self.calls = []

    def post(self, url, json=None, headers=None, timeout=None):   # noqa: A002 - mirrors UrllibSession
        self.calls.append({"url": url, "headers": dict(headers or {})})
        body = b'{"errorCode": "024", "errorMessage": "Authentication failed (SENTINEL-KEY-123?)"}'
        return connectors.HttpResponse(401, {"content-type": "application/json"}, body)

    def get(self, url, headers=None, timeout=None):
        raise OSError("no network in tests")


def test_secrets_never_leak(tmp_path, static, monkeypatch, caplog):
    caplog.set_level(logging.DEBUG)
    session = _NaverAuthFailSession()
    monkeypatch.setattr(connectors, "UrllibSession", lambda *a, **k: session)
    monkeypatch.setenv("NAVER_API_HUB_KEY_ID", "SENTINEL-ID-456")
    monkeypatch.setenv("NAVER_API_HUB_KEY", "SENTINEL-KEY-123")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "SENTINEL-ANT-789")
    cfg = WebConfig.from_env(environ={}, data_dir=str(tmp_path / "d"), static_dir=str(static), ai="off")
    app = create_app(cfg)
    try:
        assert app.fetchers.naver_endpoint == "hub"
        c = TestClient(app)
        bodies = [c.get("/api/config"), c.get("/api/health")]
        acc = c.post("/api/analyze", json={"source": "naver", "series": [{"terms": ["선크림"]}], "range": "1y",
                                           "unit": "month"})
        bodies.append(acc)
        snap = c.wait_job(acc.json()["job_id"])
        assert snap["status"] == "failed" and snap["error"]["code"] == "naver_auth"
        bodies.append(c.get(f"/api/jobs/{acc.json()['job_id']}"))
        assert session.calls and session.calls[0]["headers"]["X-NCP-APIGW-API-KEY"] == "SENTINEL-KEY-123"
        sentinels = ("SENTINEL-KEY-123", "SENTINEL-ID-456", "SENTINEL-ANT-789")
        for resp in bodies:
            blob = resp.content.decode("utf-8") + json.dumps(resp.header_list)
            for s in sentinels:
                assert s not in blob
        logs = "\n".join(r.getMessage() for r in caplog.records)
        for s in sentinels:
            assert s not in logs
        assert "SENTINEL" not in repr(cfg) and "SENTINEL" not in repr(app.config)
        creds = connectors.NaverCredentials.from_env()
        assert repr(creds) == "NaverCredentials(endpoint='hub', key_id='***', key='***')"
    finally:
        app.close(timeout=5)


def _client_key(environ_extra: dict, trust: bool) -> str:
    env = {"REMOTE_ADDR": "10.0.0.1", "REQUEST_METHOD": "GET", "PATH_INFO": "/"}
    env.update(environ_extra)
    return Request(env, trust_proxy=trust).client_key


def test_x_forwarded_for_ignored_by_default(make):
    assert _client_key({"HTTP_X_FORWARDED_FOR": "1.2.3.4"}, False) == "10.0.0.1"
    c = TestClient(make())
    for i in range(60):
        assert c.get("/api/config", headers={"X-Forwarded-For": f"9.9.9.{i}"}).status == 200
    assert c.get("/api/config", headers={"X-Forwarded-For": "9.9.8.1"}).status == 429


def test_trust_proxy_uses_last_hop(make):
    assert _client_key({"HTTP_X_FORWARDED_FOR": "6.6.6.6, 1.2.3.4"}, True) == "1.2.3.4"
    assert _client_key({"HTTP_X_FORWARDED_FOR": ""}, True) == "10.0.0.1"
    assert _client_key({"REMOTE_ADDR": "2001:db8:1:2:3:4:5:6"}, False) == "2001:db8:1:2::/64"
    assert _client_key({"REMOTE_ADDR": "2001:db8:1:2:ffff::1"}, False) == "2001:db8:1:2::/64"
    c = TestClient(make(trust_proxy=True))
    for i in range(70):
        assert c.get("/api/config", headers={"X-Forwarded-For": f"6.6.6.6, 9.9.9.{i}"}).status == 200


def test_ids_validated(make):
    c = TestClient(make())
    for path in ("/api/results/../../etc", "/api/results/..%2f..%2fetc", "/api/results/" + "A" * 21,
                 "/api/results/" + "A" * 23, "/api/shared/!!!!!!!!!!!!!!!!!!!!!!", "/api/jobs/A%00" + "A" * 19,
                 "/api/export/" + "A" * 22 + "/100.csv", "/api/export/s/../x.csv"):
        resp = c.get(path)
        assert resp.status == 404, path
        assert resp.json()["error"]["code"] in ("not_found", "result_not_found"), path


def test_error_never_contains_traceback(make, monkeypatch, caplog):
    def boom(*_a, **_k):
        raise KeyError("선크림 SECRET-TERM")

    monkeypatch.setattr(api, "get_samples", boom)
    c = TestClient(make())
    resp = c.get("/api/samples")
    assert resp.status == 500
    err = resp.json()["error"]
    assert err["code"] == "internal_error" and err["request_id"] == resp.headers["x-request-id"]
    for needle in ("Traceback", 'File "', "KeyError", "SECRET-TERM", "선크림"):
        assert needle not in resp.text
    logged = "\n".join(r.getMessage() for r in caplog.records)
    assert "KeyError" in logged and resp.headers["x-request-id"] in logged
    assert "SECRET-TERM" not in logged


def test_access_log_has_no_paths_or_terms(make, caplog):
    caplog.set_level(logging.INFO, logger="trendai.web.access")
    c = TestClient(make())
    snap, env = analyze_and_wait(c, SAMPLE)
    c.get(f"/api/results/{env['result_id']}?hl=en")
    lines = [r.getMessage() for r in caplog.records if r.name == "trendai.web.access"]
    assert lines
    assert any(re.match(r"^rid=[0-9a-f]{8} ip=[0-9a-f]{12} GET /api/results/\{id\} 200 \d+ms \d+B$", ln)
               for ln in lines)
    joined = "\n".join(lines)
    assert env["result_id"] not in joined and "hl=en" not in joined and "growing" not in joined
    assert "127.0.0.1" not in joined


def test_no_warnings_state_changes_in_web_package():
    root = pathlib.Path(__file__).resolve().parents[1] / "trendai" / "web"
    offenders = []
    for path in root.glob("*.py"):
        if path.name in ("log.py", "testing.py"):
            continue
        text = path.read_text(encoding="utf-8")
        for needle in ("catch_warnings", "simplefilter", "showwarning", "filterwarnings"):
            if needle in text:
                offenders.append((path.name, needle))
    assert offenders == []
