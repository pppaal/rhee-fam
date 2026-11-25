"""``trendai serve`` (spec B.5, H.4.4): Korean help, start-up messages, exit codes and signal handling.

Every test that starts the server runs it in a subprocess: the command installs the process-wide warnings baseline
and ends with ``os._exit``, neither of which may happen inside the pytest process. The child environment is hermetic
(no ``TRENDAI_*``, ``NAVER_*`` or ``ANTHROPIC_API_KEY`` from the developer's shell), so nothing reaches the network.
"""

from __future__ import annotations

import contextlib
import json
import os
import queue
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterator, Optional

import pytest

from trendai import cli

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(os.name != "posix", reason="signals and process groups are POSIX-only here")

#: a urllib opener that never uses the sandbox's HTTP(S)_PROXY for 127.0.0.1
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))

CONTACT_WARNING_KO = ("경고: TRENDAI_CONTACT가 설정되지 않아 Wikimedia 요청의 User-Agent에 연락처가 없습니다. "
                      "공개 운영 전에 운영자 이메일이나 URL을 설정하세요.")
NEED_CONTACT_KO = ("오류: 공개 주소로 실행하려면 TRENDAI_CONTACT(운영자 이메일이나 URL)를 설정하세요. "
                   "비공개로 쓰려면 TRENDAI_ALLOW_NO_CONTACT=1.")
NEED_CONTACT_EN = ("Error: set TRENDAI_CONTACT (operator e-mail or URL) to run on a public address, or "
                   "TRENDAI_ALLOW_NO_CONTACT=1 for private use.")
NAVER_NOTE_KO = "안내: 네이버 API 키가 없어 네이버 출처는 ‘준비 중’으로 표시됩니다."


def _env(**extra: str) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items()
           if not (k.startswith(("TRENDAI_", "NAVER_")) or k == "ANTHROPIC_API_KEY")}
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(ROOT), env.get("PYTHONPATH")]))
    env["PYTHONUNBUFFERED"] = "0"     # the URL line must be flushed by the command itself, not by the env
    env.update(extra)
    return env


class Server:
    """A ``python -m trendai serve`` child with stdout lines in a queue and stderr collected in the background."""

    def __init__(self, args: list[str], env: dict[str, str], cwd: Path) -> None:
        self.proc = subprocess.Popen([sys.executable, "-m", "trendai", "serve", *args], cwd=cwd, env=env,
                                     stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                     text=True, encoding="utf-8")
        self.lines: "queue.Queue[str]" = queue.Queue()
        self._err: list[str] = []
        self._out = threading.Thread(target=self._pump, args=(self.proc.stdout, self.lines.put), daemon=True)
        self._errt = threading.Thread(target=self._pump, args=(self.proc.stderr, self._err.append), daemon=True)
        self._out.start()
        self._errt.start()

    @staticmethod
    def _pump(stream, sink) -> None:  # type: ignore[no-untyped-def]
        for line in stream:
            sink(line.rstrip("\n"))

    def url(self, timeout: float = 30.0) -> str:
        """The base URL from the start-up line, read from the pipe while the process is still running."""
        line = self.lines.get(timeout=timeout)
        assert self.proc.poll() is None, "the server exited before its URL could be read"
        prefix = "trendai 웹 앱: " if line.startswith("trendai 웹 앱: ") else "trendai web app: "
        assert line.startswith(prefix), line
        return line[len(prefix):].split(" ", 1)[0]

    def wait(self, timeout: float) -> int:
        code = self.proc.wait(timeout=timeout)
        self._errt.join(timeout=5)
        return code

    @property
    def stderr(self) -> str:
        return "\n".join(self._err)

    def stop(self) -> None:
        if self.proc.poll() is None:
            self.proc.kill()
            self.proc.wait(timeout=10)


@contextlib.contextmanager
def serve(tmp_path: Path, *args: str, env: Optional[dict[str, str]] = None) -> Iterator[Server]:
    server = Server(["--port", "0", "--data-dir", str(tmp_path / "data"), *args], env or _env(), tmp_path)
    try:
        yield server
    finally:
        server.stop()


def _get(url: str, timeout: float = 10.0) -> tuple[int, dict]:
    try:
        with _OPENER.open(url, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8") or "{}")


def _post(url: str, body: bytes, content_type: str) -> tuple[int, dict]:
    req = urllib.request.Request(url, data=body, method="POST", headers={"Content-Type": content_type})
    try:
        with _OPENER.open(req, timeout=30) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8") or "{}")


# --------------------------------------------------------------------------- help and argument errors


def test_serve_help_korean(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as info:
        cli.main(["serve", "--help"])
    assert info.value.code == 0
    out = capsys.readouterr().out
    assert out.startswith("사용법: trendai serve")
    for option, text in (("--host", "접속 주소 (기본: TRENDAI_HOST 또는 127.0.0.1)"),
                         ("--port", "포트 (기본: TRENDAI_PORT 또는 8000, 0이면 빈 포트를 고름)"),
                         ("--data-dir", "결과·캐시 저장 폴더 (기본: TRENDAI_DATA_DIR 또는 ./.trendai-data)"),
                         ("--dev", "개발 모드: 정적 파일을 매번 새로 읽고 /static/dev를 제공"),
                         ("--lang", "안내 문구 언어 (ko, en)")):
        assert option in out and text in out, option
    assert "웹 앱 서버 실행 (기본 http://127.0.0.1:8000)" in out


def test_serve_listed_in_main_help(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        cli.main(["--help"])
    assert "serve         웹 앱 서버 실행 (기본 http://127.0.0.1:8000)" in capsys.readouterr().out


@pytest.mark.parametrize("value", ["abc", "-1", "65536"])
def test_serve_bad_port_is_a_usage_error(value: str, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as info:
        cli.main(["serve", "--port", value])
    assert info.value.code == 2
    assert "포트(--port): 0에서 65535 사이의 정수를 입력하세요" in capsys.readouterr().err


# --------------------------------------------------------------------------- start, stop and messages


def test_serve_starts_and_stops(tmp_path: Path) -> None:
    with serve(tmp_path) as server:
        url = server.url()
        assert url.startswith("http://127.0.0.1:")
        status, body = _get(url + "/api/health")
        assert status == 200 and body["status"] == "ok" and body["store"]["writable"] is True
        started = time.monotonic()
        server.proc.send_signal(signal.SIGTERM)
        assert server.wait(timeout=10) == 0, server.stderr
        assert time.monotonic() - started < 10
    assert (tmp_path / "data" / "results").is_dir()


def test_serve_prints_url_flushed(tmp_path: Path) -> None:
    # PYTHONUNBUFFERED is off and stdout is a pipe: the line arrives only because the command flushes it
    with serve(tmp_path) as server:
        url = server.url(timeout=30)
        assert server.proc.poll() is None
        assert _get(url + "/api/health")[0] == 200


def test_serve_warns_without_contact(tmp_path: Path) -> None:
    with serve(tmp_path) as server:
        server.url()
        server.proc.send_signal(signal.SIGTERM)
        assert server.wait(timeout=10) == 0, server.stderr
    assert CONTACT_WARNING_KO in server.stderr
    assert NAVER_NOTE_KO in server.stderr                       # hermetic env: no Naver keys
    assert "TRENDAI_TRUST_PROXY" not in server.stderr


def test_serve_contact_set_no_warning_and_proxy_warning(tmp_path: Path) -> None:
    env = _env(TRENDAI_CONTACT="ops@example.com", TRENDAI_TRUST_PROXY="1")
    with serve(tmp_path, env=env) as server:
        server.url()
        server.proc.send_signal(signal.SIGTERM)
        assert server.wait(timeout=10) == 0, server.stderr
    assert "TRENDAI_CONTACT가 설정되지 않아" not in server.stderr
    assert "경고: TRENDAI_TRUST_PROXY를 쓸 때는 TRENDAI_PUBLIC_URL도 설정하는 것이 안전합니다." in server.stderr


def test_serve_naver_keys_no_note(tmp_path: Path) -> None:
    # fake keys: the server only builds the connector at start-up, it calls nothing
    env = _env(NAVER_API_HUB_KEY_ID="test-id", NAVER_API_HUB_KEY="SENTINEL-KEY-0000")
    with serve(tmp_path, env=env) as server:
        url = server.url()
        status, cfg = _get(url + "/api/config")
        assert status == 200 and cfg["sources"]["naver"]["available"] is True
        server.proc.send_signal(signal.SIGTERM)
        assert server.wait(timeout=10) == 0, server.stderr
    assert "네이버 API 키가 없어" not in server.stderr
    assert "SENTINEL-KEY-0000" not in server.stderr


def test_serve_english_messages(tmp_path: Path) -> None:
    with serve(tmp_path, "--lang", "en") as server:
        line = server.lines.get(timeout=30)
        assert line.startswith("trendai web app: http://127.0.0.1:") and line.endswith(" (Ctrl+C to stop)")
        server.proc.send_signal(signal.SIGINT)
        assert server.wait(timeout=10) == 0, server.stderr
    assert "Warning: TRENDAI_CONTACT is not set" in server.stderr
    assert "Note: no Naver API keys, so the Naver source shows as not available." in server.stderr


def test_serve_flags_override_environment(tmp_path: Path) -> None:
    env = _env(TRENDAI_DATA_DIR=str(tmp_path / "from-env"), TRENDAI_DEV="0")
    with serve(tmp_path, "--dev", env=env) as server:        # serve() passes --data-dir tmp/data
        url = server.url()
        with _OPENER.open(url + "/static/dev/mock.html", timeout=10) as resp:   # --dev wins over TRENDAI_DEV=0
            assert resp.status == 200
        server.proc.send_signal(signal.SIGTERM)
        assert server.wait(timeout=10) == 0, server.stderr
    assert (tmp_path / "data" / "results").is_dir()
    assert not (tmp_path / "from-env").exists()


def test_serve_environment_used_without_flags(tmp_path: Path) -> None:
    env = _env(TRENDAI_DATA_DIR=str(tmp_path / "env-dir"), TRENDAI_PORT="0")
    server = Server([], env, tmp_path)
    try:
        assert _get(server.url() + "/api/health")[0] == 200
        server.proc.send_signal(signal.SIGTERM)
        assert server.wait(timeout=10) == 0, server.stderr
    finally:
        server.stop()
    assert (tmp_path / "env-dir" / "results").is_dir()


# --------------------------------------------------------------------------- bad configuration (2), port errors (1)


def _run(tmp_path: Path, *args: str, env: Optional[dict[str, str]] = None) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", "trendai", "serve", "--data-dir", str(tmp_path / "data"), *args],
                          cwd=tmp_path, env=env or _env(), capture_output=True, text=True, encoding="utf-8",
                          timeout=60)


def test_serve_public_requires_contact(tmp_path: Path) -> None:
    result = _run(tmp_path, "--host", "0.0.0.0", "--port", "0")
    assert result.returncode == 2
    assert result.stderr.strip().splitlines()[-1] == NEED_CONTACT_KO
    assert result.stdout == ""
    english = _run(tmp_path, "--host", "0.0.0.0", "--port", "0", "--lang", "en")
    assert english.returncode == 2 and english.stderr.strip().splitlines()[-1] == NEED_CONTACT_EN
    public_url = _run(tmp_path, "--port", "0", env=_env(TRENDAI_PUBLIC_URL="https://trendai.example"))
    assert public_url.returncode == 2 and NEED_CONTACT_KO in public_url.stderr

    with serve(tmp_path, "--host", "0.0.0.0", env=_env(TRENDAI_ALLOW_NO_CONTACT="1")) as server:
        url = server.url()                                   # 0.0.0.0 is printed as a reachable loopback URL
        assert _get(url + "/api/health")[0] == 200
        server.proc.send_signal(signal.SIGTERM)
        assert server.wait(timeout=10) == 0, server.stderr
    assert "TRENDAI_CONTACT가 설정되지 않아" not in server.stderr   # public with an explicit opt-out


def test_serve_invalid_environment_exits_2(tmp_path: Path) -> None:
    result = _run(tmp_path, "--port", "0", env=_env(TRENDAI_WORKERS="many"))
    assert result.returncode == 2
    assert result.stderr.strip().splitlines()[-1] == "오류: TRENDAI_WORKERS: expected an integer, got 'many'"


def test_serve_port_in_use_exits_1(tmp_path: Path) -> None:
    with socket.socket() as busy:
        busy.bind(("127.0.0.1", 0))
        busy.listen()
        port = busy.getsockname()[1]
        result = _run(tmp_path, "--port", str(port))
        english = _run(tmp_path, "--port", str(port), "--lang", "en")
    assert result.returncode == 1
    assert result.stderr.strip().splitlines()[-1].startswith(f"오류: 127.0.0.1:{port}")
    assert "열 수 없습니다 — 이미 다른 프로그램이 이 포트를 쓰고 있습니다" in result.stderr
    assert english.returncode == 1
    assert f"Error: cannot listen on 127.0.0.1:{port} — the port is already in use" in english.stderr


# --------------------------------------------------------------------------- graceful drain and the second signal


def _slow_upload_csv(columns: int = 12, rows: int = 1100) -> bytes:
    """A daily table whose analysis takes several seconds (12 series of 1,100 days)."""
    import datetime as dt
    import math

    lines = ["날짜," + ",".join(f"v{i}" for i in range(columns))]
    for t in range(rows):
        day = (dt.date(2021, 1, 1) + dt.timedelta(days=t)).isoformat()
        lines.append(day + "," + ",".join(f"{100 + 10 * math.sin(t / (3 + i)) + 0.05 * t:.3f}"
                                          for i in range(columns)))
    return ("\n".join(lines) + "\n").encode("utf-8")


def _start_slow_job(url: str, columns: int = 12) -> str:
    from trendai.web.testing import multipart_body

    body, ctype = multipart_body("slow.csv", _slow_upload_csv(columns))
    status, upload = _post(url + "/api/upload", body, ctype)
    assert status == 200, upload
    request = {"source": "upload", "upload_id": upload["upload_id"], "value_cols": None, "h": "auto"}
    status, job = _post(url + "/api/analyze", json.dumps(request).encode("utf-8"), "application/json")
    assert status == 202, job
    return job["job_id"]


def _wait_running(url: str, job_id: str, timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status, snap = _get(f"{url}/api/jobs/{job_id}")
        assert status in (200, 429), snap
        if status == 200 and snap["status"] == "running":
            return
        assert status == 429 or snap["status"] == "queued", snap
        time.sleep(0.1)
    raise AssertionError("the job never started")


def test_serve_drains_while_serving(tmp_path: Path) -> None:
    with serve(tmp_path) as server:
        url = server.url()
        job_id = _start_slow_job(url, columns=4)
        _wait_running(url, job_id)
        server.proc.send_signal(signal.SIGTERM)
        deadline = time.monotonic() + 10
        while True:                                   # first signal: health 503 draining, the listener keeps serving
            status, health = _get(url + "/api/health")
            if status == 503:
                break
            assert time.monotonic() < deadline, health
            time.sleep(0.05)
        assert health["status"] == "draining"
        assert _get(f"{url}/api/jobs/{job_id}")[0] in (200, 429)    # the running job can still be watched
        assert server.wait(timeout=60) == 0, server.stderr          # exits once the job is over


def test_serve_second_signal_exits_1(tmp_path: Path) -> None:
    with serve(tmp_path) as server:
        url = server.url()
        job_id = _start_slow_job(url)
        _wait_running(url, job_id)
        server.proc.send_signal(signal.SIGTERM)
        deadline = time.monotonic() + 10
        while _get(url + "/api/health")[0] != 503:  # the first handler has run: draining, job still running
            assert time.monotonic() < deadline
            time.sleep(0.05)
        started = time.monotonic()
        server.proc.send_signal(signal.SIGTERM)
        assert server.wait(timeout=10) == 1, server.stderr
        assert time.monotonic() - started < 5
