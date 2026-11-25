"""The stdlib server helpers (B.3.2, E.13, E.14): background serving, threading, draining, signals, lazy WSGI."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import textwrap
import threading
import time
import urllib.error
import urllib.request

import pytest

from trendai.web import api
from trendai.web.server import begin_graceful_shutdown, start_background
from trendai.web.testing import TestClient, make_test_app, write_static_fixture

# never route 127.0.0.1 through the environment's HTTP(S) proxy
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
SLOW_WIKI = {"source": "wiki", "series": [{"terms": ["가"]}]}


def fetch(url: str, *, data: bytes | None = None, method: str = "GET", timeout: float = 10.0) -> tuple[int, dict]:
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json"} if data is not None else {})
    try:
        with OPENER.open(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read() or b"null")
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read() or b"null")


@pytest.fixture
def served(tmp_path):
    app = make_test_app(tmp_path / "data", static_dir=write_static_fixture(tmp_path / "static"))
    server, thread, base = start_background(app)
    yield app, server, base
    server.shutdown()
    server.server_close()
    app.close(timeout=5)


def test_start_background_serves(served):
    app, server, base = served
    assert base.startswith("http://127.0.0.1:") and not base.endswith(":0")
    status, body = fetch(base + "/api/health")
    assert status == 200 and body["status"] == "ok"
    with OPENER.open(base + "/", timeout=10) as resp:
        assert resp.status == 200 and resp.headers["Content-Security-Policy"].startswith("default-src 'self'")


def test_threaded_concurrency(served, monkeypatch):
    app, server, base = served
    real = api.health

    def slow_health(*args, **kwargs):
        time.sleep(0.6)
        return real(*args, **kwargs)

    monkeypatch.setattr(api, "health", slow_health)
    results = []
    started = time.monotonic()
    threads = [threading.Thread(target=lambda: results.append(fetch(base + "/api/health")[0])) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)
    elapsed = time.monotonic() - started
    assert results == [200, 200]
    assert elapsed < 1.1, elapsed                  # the two slow requests overlapped


def test_close_drains(tmp_path):
    app = make_test_app(tmp_path / "data", static_dir=write_static_fixture(tmp_path / "static"))
    app.fetchers.wiki.delay = 0.8
    c = TestClient(app)
    job_id = c.post("/api/analyze", json=SLOW_WIKI).json()["job_id"]
    time.sleep(0.1)
    started = time.monotonic()
    app.close(timeout=10)
    assert time.monotonic() - started >= 0.3       # waited for the running job
    assert c.get(f"/api/jobs/{job_id}").json()["status"] == "done"
    resp = c.post("/api/analyze", json={"source": "sample", "series": [{"terms": ["fad"]}]})
    assert resp.status == 503 and resp.headers["retry-after"] == "30"
    health = c.get("/api/health")
    assert health.status == 503 and health.json()["status"] == "draining"
    app.close(timeout=1)                            # idempotent


def test_close_cancels_queued(tmp_path):
    app = make_test_app(tmp_path / "data", static_dir=write_static_fixture(tmp_path / "static"), workers=1)
    app.fetchers.wiki.delay = 0.6
    c = TestClient(app)
    first = c.post("/api/analyze", json=SLOW_WIKI).json()["job_id"]
    queued = c.post("/api/analyze", json={"source": "wiki", "series": [{"terms": ["나"]}]}).json()["job_id"]
    time.sleep(0.1)
    app.close(timeout=10)
    assert c.get(f"/api/jobs/{queued}").json()["status"] == "cancelled"
    assert c.get(f"/api/jobs/{first}").json()["status"] == "done"


def test_draining_keeps_serving(served):
    app, server, base = served
    app.fetchers.wiki.delay = 1.0
    status, acc = fetch(base + "/api/analyze", data=json.dumps(SLOW_WIKI).encode(), method="POST")
    assert status == 202
    time.sleep(0.1)
    helper = begin_graceful_shutdown(app, server, grace=10)
    status, body = fetch(base + "/api/health")
    assert status == 503 and body["status"] == "draining" and body["draining"] is True
    status, body = fetch(base + "/api/analyze", data=json.dumps({"source": "sample", "series": [
        {"terms": ["fad"]}]}).encode(), method="POST")
    assert status == 503 and body["error"]["code"] == "server_busy" and body["error"]["retry_after"] == 30
    status, body = fetch(base + f"/api/jobs/{acc['job_id']}")
    assert status == 200 and body["status"] in ("queued", "running", "done")
    helper.join(15)
    assert not helper.is_alive()
    assert app.jobs.counts() == (0, 0)
    with pytest.raises(OSError):
        OPENER.open(base + "/api/health", timeout=2)


SERVE_SCRIPT = textwrap.dedent("""
    import os, sys
    from trendai.web import install_warning_baseline
    from trendai.web.server import serve_forever
    from trendai.web.testing import make_test_app, write_static_fixture
    install_warning_baseline()
    app = make_test_app(sys.argv[1], static_dir=write_static_fixture(sys.argv[2]))
    code = serve_forever(app, "127.0.0.1", 0, on_ready=lambda url: print("READY", url, flush=True))
    print("EXIT", code, flush=True)
    os._exit(code)
""")


def test_serve_forever_sigterm(tmp_path):
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([os.getcwd(), os.environ.get("PYTHONPATH", "")])}
    proc = subprocess.Popen([sys.executable, "-c", SERVE_SCRIPT, str(tmp_path / "d"), str(tmp_path / "s")],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    try:
        line = proc.stdout.readline()
        assert line.startswith("READY http://127.0.0.1:"), (line, proc.stderr.read() if proc.poll() else "")
        base = line.split()[1]
        assert fetch(base + "/api/health")[0] == 200
        proc.send_signal(signal.SIGTERM)
        out, _err = proc.communicate(timeout=20)
        assert "EXIT 0" in out and proc.returncode == 0
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(5)


def test_wsgi_import_is_lazy(tmp_path):
    target = tmp_path / "x"
    code = ("import threading, os, sys; before = threading.active_count(); import trendai.web.wsgi; "
            "print(threading.active_count() - before, os.path.exists(sys.argv[1]))")
    env = {**os.environ, "TRENDAI_DATA_DIR": str(target)}
    out = subprocess.run([sys.executable, "-c", code, str(target)], capture_output=True, text=True, env=env,
                         timeout=60, cwd=os.getcwd())
    assert out.returncode == 0, out.stderr
    assert out.stdout.split() == ["0", "False"]


def test_wsgi_application_builds_once(tmp_path):
    code = textwrap.dedent("""
        import io, json, sys
        import trendai.web.wsgi as w
        environ = {"REQUEST_METHOD": "GET", "PATH_INFO": "/api/health", "QUERY_STRING": "", "SERVER_NAME": "t",
                   "SERVER_PORT": "80", "REMOTE_ADDR": "127.0.0.1", "HTTP_HOST": "t", "wsgi.url_scheme": "http",
                   "wsgi.input": io.BytesIO()}
        seen = {}
        body = b"".join(w.application(environ, lambda s, h: seen.update(status=s)))
        first = w._APP
        b"".join(w.application(environ, lambda s, h: None))
        print(seen["status"], json.loads(body)["status"], w._APP is first)
        w._APP.close(timeout=1)
    """)
    env = {**os.environ, "TRENDAI_DATA_DIR": str(tmp_path / "d"), "TRENDAI_HOST": "127.0.0.1"}
    for name in ("TRENDAI_PUBLIC_URL", "TRENDAI_CONTACT"):
        env.pop(name, None)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=60,
                         cwd=os.getcwd())
    assert out.returncode == 0, out.stderr
    assert out.stdout.split() == ["200", "OK", "ok", "True"]
