"""E2E test server (spec H.4.1): the real app with the fake fetchers of ``trendai.web.testing`` on a free port.

    python tests/e2e/server.py --naver on|off --ai on|off [--data-dir DIR]

Prints ``READY <base_url>`` on stdout (flushed: a reader on a pipe would otherwise block) and serves until SIGTERM
or SIGINT arrives or stdin is closed; a temporary data directory is removed on the way out. Logs go to stderr at
INFO, so the tests can count job lines (``naver=fresh``). Nothing here touches the network.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import signal
import sys
import tempfile
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:          # run as a script from a checkout: make the package importable
    sys.path.insert(0, str(ROOT))

from trendai.web import install_warning_baseline   # noqa: E402 - after the sys.path fix above
from trendai.web.server import start_background    # noqa: E402
from trendai.web.testing import make_test_app      # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="trendai E2E server (fake Naver, Wikipedia and LLM)")
    parser.add_argument("--naver", choices=["on", "off"], default="on")
    parser.add_argument("--ai", choices=["on", "off"], default="on")
    parser.add_argument("--data-dir", default=None)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    install_warning_baseline()          # the process owner's warnings baseline (E.13)

    temp_dir = None if args.data_dir else tempfile.mkdtemp(prefix="trendai-e2e-")
    data_dir = args.data_dir or temp_dir
    app = make_test_app(data_dir, naver=args.naver == "on", ai=args.ai == "on")
    server, _thread, url = start_background(app, "127.0.0.1", 0)

    stop = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda _signum, _frame: stop.set())

    def watch_stdin() -> None:          # the parent died or closed our stdin: stop too
        try:
            sys.stdin.read()
        except (OSError, ValueError):
            pass
        stop.set()

    threading.Thread(target=watch_stdin, name="e2e-stdin", daemon=True).start()
    print(f"READY {url}", flush=True)
    try:
        while not stop.wait(0.5):
            pass
    finally:
        server.shutdown()
        server.server_close()
        app.close(timeout=0)
        if temp_dir:
            shutil.rmtree(temp_dir, ignore_errors=True)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)                         # analysis threads must not keep the process alive


if __name__ == "__main__":
    raise SystemExit(main())
