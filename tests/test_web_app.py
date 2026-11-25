"""App wiring (E.2-E.4, E.12, C.0, C.1, C.14): health, router, SPA fallback, static files, headers, gzip, strict JSON.

Every app here serves ``testing.write_static_fixture`` -- WP1's tests never read WP3's files -- except
``test_package_static_if_present``.
"""

from __future__ import annotations

import gzip
import json
import pathlib

import pytest

from trendai.web import api, viewmodel
from trendai.web.http import APP_CSP
from trendai.web.testing import TestClient, make_test_app, write_static_fixture

SECURITY = {
    "content-security-policy": APP_CSP,
    "x-content-type-options": "nosniff",
    "x-frame-options": "DENY",
    "referrer-policy": "no-referrer",
    "permissions-policy": "camera=(), microphone=(), geolocation=(), payment=(), usb=()",
    "cross-origin-opener-policy": "same-origin",
    "cross-origin-resource-policy": "same-origin",
}
ID = "A" * 22


@pytest.fixture
def static(tmp_path):
    return write_static_fixture(tmp_path / "static")


@pytest.fixture
def app(tmp_path, static):
    a = make_test_app(tmp_path / "data", static_dir=static)
    yield a
    a.close(timeout=5)


@pytest.fixture
def client(app):
    return TestClient(app)


def test_health_ok(client):
    resp = client.get("/api/health")
    assert resp.status == 200
    assert resp.headers["cache-control"] == "no-store"
    body = resp.json()
    assert set(body) == {"status", "version", "uptime_s", "draining", "jobs", "store"}
    assert body["status"] == "ok" and body["draining"] is False
    assert body["jobs"] == {"running": 0, "queued": 0} and body["store"] == {"writable": True}
    assert len(resp.headers["x-request-id"]) == 8


def test_security_headers_everywhere(client, monkeypatch):
    def boom(*_a, **_k):
        raise RuntimeError("forced failure")

    responses = [client.get("/"), client.get("/static/js/app.js"), client.get("/api/config"),
                 client.get("/api/nope")]
    monkeypatch.setattr(api, "get_config", boom)
    forced = client.get("/api/config")
    assert forced.status == 500 and forced.json()["error"]["code"] == "internal_error"
    for resp in [*responses, forced]:
        for name, value in SECURITY.items():
            assert resp.headers.get(name) == value, (resp, name)
        assert "strict-transport-security" not in resp.headers


def test_hsts_when_configured(tmp_path, static):
    app = make_test_app(tmp_path / "d", static_dir=static, hsts=True)
    try:
        assert TestClient(app).get("/").headers["strict-transport-security"] == "max-age=31536000"
    finally:
        app.close(timeout=5)


def test_spa_routes_serve_index(client, static):
    index = (static / "index.html").read_bytes()
    for path in ("/", "/explore", "/upload", "/samples", "/method", "/sources", "/privacy", f"/r/{ID}", f"/s/{ID}"):
        resp = client.get(path)
        assert resp.status == 200, path
        assert resp.body == index, path
        assert resp.headers["content-type"] == "text/html; charset=utf-8"
        assert resp.headers["cache-control"] == "no-cache"
        assert resp.headers["etag"].startswith('"')
        if path.startswith(("/r/", "/s/")):
            assert resp.headers["x-robots-tag"] == "noindex, nofollow"
        else:
            assert "x-robots-tag" not in resp.headers


def test_static_missing_503(tmp_path, caplog):
    empty = tmp_path / "empty"
    empty.mkdir()
    app = make_test_app(tmp_path / "d", static_dir=empty)
    try:
        c = TestClient(app)
        for path in ("/", "/explore", "/nope", "/favicon.ico"):
            resp = c.get(path)
            assert resp.status == 503, path
            assert resp.headers["content-type"] == "text/plain; charset=utf-8"
            assert resp.text == "trendai: static files are missing"
            assert resp.headers["content-security-policy"] == APP_CSP
        assert c.get("/static/js/app.js").status == 404
        assert c.get("/api/health").status == 200
    finally:
        app.close(timeout=5)
    assert any("static files missing" in r.getMessage() for r in caplog.records)


def test_package_static_if_present(tmp_path):
    index = pathlib.Path(__file__).resolve().parents[1] / "trendai" / "web" / "static" / "index.html"
    if not index.is_file():
        pytest.skip("trendai/web/static/index.html is not there yet (WP3)")
    app = make_test_app(tmp_path / "d")
    try:
        assert TestClient(app).get("/").body == index.read_bytes()
    finally:
        app.close(timeout=5)


def test_trailing_slash_redirect(client):
    resp = client.get("/explore/?q=a")
    assert resp.status == 308
    assert resp.headers["location"] == "/explore?q=a"
    assert client.get("/method/").headers["location"] == "/method"


def test_unknown_page_404_index(client, static):
    resp = client.get("/nope")
    assert resp.status == 404
    assert resp.body == (static / "index.html").read_bytes()


def test_unknown_api_404_json(client):
    for path in ("/api/nope", "/api/results", "/api/results/short", f"/api/jobs/{ID}x"):
        resp = client.get(path)
        assert resp.status == 404, path
        err = resp.json()["error"]
        assert err["code"] == "not_found" and err["status"] == 404
        assert resp.headers["cache-control"] == "no-store"


def test_method_not_allowed(client):
    resp = client.request("PUT", "/api/analyze", json={})
    assert resp.status == 405
    assert resp.headers["allow"] == "POST"
    assert resp.json()["error"]["code"] == "method_not_allowed"
    assert client.request("POST", "/explore", data=b"").headers["allow"] == "GET, HEAD"


def test_options_405_no_cors(client):
    for path in ("/api/analyze", "/api/config", "/"):
        resp = client.request("OPTIONS", path, headers={"Origin": "https://evil.example",
                                                       "Access-Control-Request-Method": "POST"})
        assert resp.status == 405
        assert not [h for h in resp.headers if h.startswith("access-control-")]


def test_static_traversal_blocked(client):
    for path in ("/static/../app.py", "/static/%2e%2e/app.py", "/static/..%2fapp.py", "/static/\\app.py",
                 "/static/dev/mock.html", "/static/js/../../web/app.py", "/static/%00js/app.js",
                 "/static/js/app.js%00.html", "/static/nope.js"):
        assert client.get(path).status == 404, path


def test_dev_mode_serves_dev_files(tmp_path, static):
    app = make_test_app(tmp_path / "d", static_dir=static, dev=True)
    try:
        c = TestClient(app)
        assert c.get("/static/dev/mock.html").status == 200
        (static / "js" / "new.js").write_bytes(b"export {};\n")       # dev rescans per request
        assert c.get("/static/js/new.js").status == 200
    finally:
        app.close(timeout=5)


def test_static_etag_304(client, static):
    resp = client.get("/static/js/app.js")
    assert resp.status == 200
    assert resp.body == (static / "js" / "app.js").read_bytes()
    assert resp.headers["content-type"] == "text/javascript; charset=utf-8"
    assert resp.headers["cache-control"] == "no-cache"
    etag = resp.headers["etag"]
    again = client.get("/static/js/app.js", headers={"If-None-Match": etag})
    assert again.status == 304 and again.body == b""
    font = client.get("/static/fonts/pretendard-1.3.9/x.woff2")
    assert font.headers["cache-control"] == "public, max-age=31536000, immutable"
    assert font.headers["content-type"] == "font/woff2"
    css = client.get("/static/css/tokens.css")
    assert css.headers["content-type"] == "text/css; charset=utf-8"


def test_head_has_no_body(client):
    get = client.get("/")
    head = client.head("/")
    assert head.status == 200 and head.body == b""
    assert head.headers["content-length"] == get.headers["content-length"]
    h2 = client.head("/api/health")
    assert h2.status == 200 and h2.body == b""


def test_favicon_and_robots(client, static):
    fav = client.get("/favicon.ico")
    assert fav.status == 200
    assert fav.headers["content-type"] == "image/svg+xml"
    assert fav.body == (static / "favicon.svg").read_bytes()
    robots = client.get("/robots.txt")
    assert robots.status == 200
    assert robots.text == "User-agent: *\nDisallow: /api/\nDisallow: /r/\nDisallow: /s/\n"


def test_gzip_json(client):
    plain = client.get("/api/samples")
    assert "content-encoding" not in plain.headers
    zipped = client.get("/api/samples", headers={"Accept-Encoding": "gzip, br"})
    assert zipped.headers["content-encoding"] == "gzip"
    assert zipped.headers["vary"] == "Accept-Encoding"
    assert json.loads(gzip.decompress(zipped.body)) == plain.json()
    assert int(zipped.headers["content-length"]) == len(zipped.body)
    small = client.get("/api/health", headers={"Accept-Encoding": "gzip"})
    assert "content-encoding" not in small.headers                 # <= 1 KB stays plain
    static = client.get("/static/js/app.js", headers={"Accept-Encoding": "gzip"})
    assert "content-encoding" not in static.headers


def test_samples_endpoint(client):
    resp = client.get("/api/samples")
    assert resp.headers["cache-control"] == "public, max-age=3600"
    samples = resp.json()["samples"]
    assert [s["name"] for s in samples] == ["emerging", "growing", "peaking", "fad", "declining", "seasonal",
                                            "reviving", "breakout", "volatile"]
    first = samples[0]
    assert set(first) == {"name", "title", "description", "expected_phase", "freq", "start", "end", "spark",
                          "synthetic"}
    assert (first["title"], first["freq"], first["synthetic"]) == ("태동기형", "M", True)
    assert first["description"] == "낮은 기반에서 가속 성장 중인 신흥 트렌드 (S-커브 초입)"
    assert (first["start"], first["end"]) == ("2016-01-01", "2025-12-01")
    assert len(first["spark"]) == 60
    en = client.get("/api/samples?hl=en").json()["samples"]
    assert en[1]["title"] == "Growing" and en[1]["description"] == "Steady growth with seasonality"
    from trendai.samples import sample

    s = sample("emerging").to_numpy()
    assert first["spark"][-1] == float(f"{s[119]:.3g}") and first["spark"][0] == float(f"{s[1]:.3g}")


def test_json_allow_nan_false(client, monkeypatch):
    real = viewmodel.build_series_vm

    def nan_vm(inp, lang):
        vm = real(inp, lang)
        vm["hero"]["outlook"]["prob"] = float("nan")
        return vm

    monkeypatch.setattr(viewmodel, "build_series_vm", nan_vm)
    resp = client.post("/api/analyze", json={"source": "sample", "series": [{"terms": ["growing"]}]})
    job_id = resp.json()["job_id"]
    final = None
    import time

    for _ in range(400):
        poll = client.get(f"/api/jobs/{job_id}?since=0")
        body = json.loads(poll.content.decode("utf-8"))      # always strict, parseable JSON
        assert poll.status in (200, 429, 500)
        if poll.status == 500:
            assert body["error"]["code"] == "internal_error"
        late = client.get(f"/api/jobs/{job_id}?since=99")
        if late.status == 200 and late.json()["status"] in ("done", "failed", "cancelled"):
            final = late.json()
            break
        time.sleep(0.1)
    assert final is not None and final["status"] == "failed"
    assert final["error"]["code"] == "internal_error" and final["error"]["status"] == 500
    assert client.get(f"/api/results/{job_id}").status == 404


def test_two_apps_in_one_process(tmp_path, static):
    a = make_test_app(tmp_path / "a", static_dir=static)
    b = make_test_app(tmp_path / "b", static_dir=static)
    try:
        ca, cb = TestClient(a), TestClient(b)
        missing = f"/api/results/{'N' * 22}"
        for _ in range(60):
            ca.get(missing)
        assert ca.get(missing).status == 429
        assert cb.get(missing).status == 404                   # no shared limiter state
    finally:
        a.close(timeout=5)
        b.close(timeout=5)


def test_close_is_idempotent(app):
    app.close(timeout=1)
    app.close(timeout=1)
    resp = TestClient(app).post("/api/analyze", json={"source": "sample", "series": [{"terms": ["growing"]}]})
    assert resp.status == 503 and resp.headers["retry-after"] == "30"
