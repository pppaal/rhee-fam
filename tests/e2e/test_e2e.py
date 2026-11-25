"""pytest wrapper of the Node suites (spec H.4.1): the browser E2E scenarios (``tests/e2e``) and the frontend unit
tests (``tests/js``). Marker ``e2e``; exempt from the 60 s rule; skipped when ``node`` or the global ``playwright``
module is missing.

One child environment is built once and used for the skip probe and both runs: ``os.environ`` plus
``NODE_PATH=$(npm root -g)`` when Playwright is not resolvable through the current ``NODE_PATH`` (a globally
installed module is found by ``createRequire`` only through ``NODE_PATH``) and ``TRENDAI_E2E_PYTHON`` = this
interpreter, so the E2E server runs with the same Python and packages as pytest.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.e2e


def _probe(node: str, env: dict[str, str]) -> bool:
    try:
        return subprocess.run([node, "-e", "require('playwright')"], env=env, cwd=ROOT, capture_output=True,
                              timeout=60).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


@pytest.fixture(scope="module")
def node_env() -> tuple[str, dict[str, str]]:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    env = dict(os.environ)
    env["TRENDAI_E2E_PYTHON"] = sys.executable
    if not env.get("NODE_PATH") or not _probe(node, env):
        npm = shutil.which("npm")
        if npm is None:
            pytest.skip("npm is not installed (needed to find the global playwright module)")
        root = subprocess.run([npm, "root", "-g"], capture_output=True, text=True, timeout=60).stdout.strip()
        env["NODE_PATH"] = os.pathsep.join(filter(None, [env.get("NODE_PATH"), root]))
    if not _probe(node, env):
        pytest.skip("the Node module 'playwright' is not installed (npm install -g playwright)")
    return node, env


def _run(node: str, env: dict[str, str], *args: str, timeout: float) -> None:
    result = subprocess.run([node, "--test", *args], env=env, cwd=ROOT, capture_output=True, text=True,
                            encoding="utf-8", errors="replace", timeout=timeout)
    if result.returncode != 0:
        pytest.fail(f"node --test {' '.join(args)} exited with {result.returncode}\n"
                    f"--- stdout ---\n{result.stdout[-20000:]}\n--- stderr ---\n{result.stderr[-5000:]}",
                    pytrace=False)


def test_js_unit_tests(node_env: tuple[str, dict[str, str]]) -> None:
    node, env = node_env
    _run(node, env, "tests/js", timeout=300)


def test_browser_scenarios(node_env: tuple[str, dict[str, str]]) -> None:
    node, env = node_env
    _run(node, env, "--test-concurrency=1", "tests/e2e", timeout=1200)
