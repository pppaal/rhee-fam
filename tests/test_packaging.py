"""Packaging of the web app (spec B.5, H.4.4): package data, fonts, dependencies, the lazy WSGI module, the
container files."""

from __future__ import annotations

import importlib.resources
import os
import re
import shlex
import shutil
import socket
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "trendai" / "web" / "static"
FONTS = STATIC / "fonts" / "pretendard-1.3.9"


def _pyproject() -> dict:
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def test_static_index_is_package_data() -> None:
    index = importlib.resources.files("trendai.web") / "static" / "index.html"
    assert index.is_file()
    assert (importlib.resources.files("trendai.web") / "vm.schema.json").is_file()


def test_pyproject_package_data_patterns() -> None:
    tool = _pyproject()["tool"]["setuptools"]
    assert "static/**/*" in tool["package-data"]["trendai.web"]
    assert "vm.schema.json" in tool["package-data"]["trendai.web"]
    assert tool["exclude-package-data"]["trendai.web"] == ["static/dev/**"]
    # namespace discovery would turn static/dev/ into a package and ship make_fixtures.py as a module
    assert tool["packages"]["find"]["namespaces"] is False
    # package data only from the patterns above: the manifest scan (the default) warns once per static sub-directory
    # "Package 'trendai.web.static…' is absent from the `packages` configuration" (OPS-17)
    assert tool["include-package-data"] is False
    pytest_cfg = _pyproject()["tool"]["pytest"]["ini_options"]
    assert any(m.startswith("e2e:") for m in pytest_cfg["markers"])


def test_no_runtime_deps_added() -> None:
    project = _pyproject()["project"]
    names = sorted(dep.split(">")[0].split("=")[0].split("<")[0].strip() for dep in project["dependencies"])
    assert names == ["numpy", "pandas"]
    extras = project["optional-dependencies"]
    assert extras["web"] == ["openpyxl>=3.1", "anthropic>=1.0"]
    assert extras["xlsx"] == ["openpyxl>=3.1"]
    assert project["scripts"]["trendai"] == "trendai.cli:main"


def test_fonts_present_with_license() -> None:
    woff2 = sorted(p.name for p in (FONTS / "woff2-dynamic-subset").glob("*.woff2"))
    assert len(woff2) == 92
    assert (FONTS / "pretendardvariable-dynamic-subset.css").is_file()
    assert "SIL Open Font License" in (FONTS / "LICENSE.txt").read_text(encoding="utf-8")


def test_wsgi_import_is_lazy(tmp_path: Path) -> None:
    target = tmp_path / "x"
    code = ("import threading, os, sys; before = threading.active_count(); import trendai.web.wsgi; "
            "print(threading.active_count() - before, os.path.exists(sys.argv[1]))")
    env = {**os.environ, "TRENDAI_DATA_DIR": str(target),
           "PYTHONPATH": os.pathsep.join(filter(None, [str(ROOT), os.environ.get("PYTHONPATH")]))}
    out = subprocess.run([sys.executable, "-c", code, str(target)], capture_output=True, text=True, env=env,
                         timeout=60, cwd=tmp_path)
    assert out.returncode == 0, out.stderr
    assert out.stdout.split() == ["0", "False"]
    assert not target.exists()


def test_build_ships_static_without_dev(tmp_path: Path) -> None:
    """``build_py`` of a clean copy applies the package-data rules exactly as a wheel build does: every static file
    except ``static/dev/**``, the schema, and no Python module from the static tree."""
    pytest.importorskip("setuptools")
    src = tmp_path / "src"
    src.mkdir()
    for name in ("pyproject.toml", "README.md"):
        shutil.copy2(ROOT / name, src / name)
    shutil.copytree(ROOT / "trendai", src / "trendai",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.egg-info"))
    code = "import sys; from setuptools import setup; sys.argv = ['setup.py', '-q', 'build_py']; setup()"
    result = subprocess.run([sys.executable, "-c", code], cwd=src, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr[-2000:]
    # no "Package would be ignored" block per static sub-directory (OPS-17): it reads like a failed build
    assert "absent from the `packages` configuration" not in result.stdout + result.stderr, result.stderr[-2000:]
    built = src / "build" / "lib" / "trendai" / "web"
    shipped = {p.relative_to(built / "static").as_posix() for p in (built / "static").rglob("*") if p.is_file()}
    expected = {p.relative_to(STATIC).as_posix() for p in STATIC.rglob("*")
                if p.is_file() and not p.relative_to(STATIC).as_posix().startswith("dev/")
                and "__pycache__" not in p.parts}
    assert shipped == expected
    assert "index.html" in shipped and not any(name.startswith("dev/") for name in shipped)
    assert sum(name.endswith(".woff2") for name in shipped) == 92
    assert (built / "vm.schema.json").is_file()
    assert not list((built / "static").rglob("*.py"))
    assert (built / "wsgi.py").is_file() and (built / "server.py").is_file()


def test_gitignore_entries() -> None:
    lines = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert ".trendai-data/" in lines and "node_modules/" in lines


def test_dockerignore_entries() -> None:
    lines = set((ROOT / ".dockerignore").read_text(encoding="utf-8").split())
    for entry in (".git", ".gitignore", ".trendai-data", ".pytest_cache", ".ruff_cache", "**/__pycache__",
                  "*.egg-info", "build", "dist", "out", "reports", "tests", "docs", "node_modules", "read.py",
                  "trendai/web/static/dev"):
        assert entry in lines, entry


def test_dockerfile_contract() -> None:
    text = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert text.startswith("FROM python:3.12-slim\n")
    for needle in ("TRENDAI_HOST=0.0.0.0", "TRENDAI_PORT=8000", "TRENDAI_DATA_DIR=/data",
                   "pip install --no-cache-dir . openpyxl anthropic", "useradd --system --uid 10001",
                   "USER trendai", 'VOLUME ["/data"]', "EXPOSE 8000",
                   "HEALTHCHECK --interval=30s --timeout=5s --start-period=20s",
                   'CMD ["trendai", "serve"]'):
        assert needle in text, needle
    # the health check follows TRENDAI_PORT (8000 when unset or empty) and bypasses any proxy (OPS-17)
    code = _healthcheck_code()
    assert "os.environ.get('TRENDAI_PORT') or '8000'" in code and "ProxyHandler({})" in code
    assert "127.0.0.1:8000" not in code
    # the README is copied because pyproject's readme field needs it at build time
    assert "COPY pyproject.toml README.md ./" in text


def _healthcheck_cmd() -> str:
    """The shell-form command of the Dockerfile's ``HEALTHCHECK ... CMD python -c "..."`` (Docker runs it with
    ``/bin/sh -c``)."""
    text = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    match = re.search(r'^HEALTHCHECK [^\n]*\\\n\s*CMD (python -c "[^"]+")\s*$', text, re.M)
    assert match, "HEALTHCHECK CMD python -c \"...\" not found"
    return match.group(1)


def _healthcheck_code() -> str:
    """The Python source inside the health-check command."""
    return shlex.split(_healthcheck_cmd())[2]


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _run_healthcheck(port: int, proxy_port: int, cwd: Path) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items()
           if k.lower() not in ("http_proxy", "https_proxy", "all_proxy", "no_proxy", "trendai_port")}
    # an operator's proxy (without NO_PROXY) must not swallow the probe of 127.0.0.1: point it at a closed port
    proxy = f"http://127.0.0.1:{proxy_port}"
    env.update({"TRENDAI_PORT": str(port), "HTTP_PROXY": proxy, "http_proxy": proxy, "ALL_PROXY": proxy})
    # through the shell exactly as Docker's shell form does, with this interpreter in place of the image's python
    command = shlex.quote(sys.executable) + _healthcheck_cmd()[len("python"):]
    return subprocess.run(["/bin/sh", "-c", command], env=env, cwd=cwd, capture_output=True, text=True, timeout=30)


@pytest.mark.skipif(not os.path.exists("/bin/sh"), reason="needs a POSIX shell, like the image")
def test_dockerfile_healthcheck_follows_trendai_port(tmp_path: Path) -> None:
    """``docker run -e TRENDAI_PORT=9000`` must not give a permanently unhealthy container, and an ``HTTP_PROXY``
    set for the app's outbound requests must not route the local probe through the proxy (OPS-17)."""
    from trendai.web.server import start_background
    from trendai.web.testing import make_test_app, write_static_fixture

    app = make_test_app(tmp_path / "data", static_dir=write_static_fixture(tmp_path / "static"))
    server, _thread, base = start_background(app)
    try:
        port = int(base.rsplit(":", 1)[1])
        closed = _free_port()
        ok = _run_healthcheck(port, closed, tmp_path)
        assert ok.returncode == 0, ok.stderr[-2000:]
        down = _run_healthcheck(closed, closed, tmp_path)       # nothing listens there: unhealthy
        assert down.returncode != 0
    finally:
        server.shutdown()
        server.server_close()
        app.close(timeout=5)
