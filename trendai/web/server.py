"""The stdlib HTTP server for ``trendai serve`` and the tests (spec B.3.2, E.13, E.14): ``wsgiref`` with one daemon
thread per connection, a 30 s socket timeout against slow clients, quiet default logging (the app logs access lines
itself) and graceful SIGINT/SIGTERM handling."""

from __future__ import annotations

import logging
import os
import signal
import socket
import socketserver
import sys
import threading
import time
import wsgiref.simple_server
from typing import Any, Callable, Optional

from .app import TrendaiApp

__all__ = ["ThreadingWSGIServer", "make_server", "serve_forever", "start_background", "begin_graceful_shutdown"]

LOG = logging.getLogger("trendai.web")


class ThreadingWSGIServer(socketserver.ThreadingMixIn, wsgiref.simple_server.WSGIServer):
    daemon_threads = True
    allow_reuse_address = True


class _V6Server(ThreadingWSGIServer):
    address_family = socket.AF_INET6


class _QuietHandler(wsgiref.simple_server.WSGIRequestHandler):
    timeout = 30      # seconds of socket inactivity before a slow client is dropped (E.13)

    def log_message(self, format: str, *args: Any) -> None:   # noqa: A002 - signature of the base class
        return            # E.11: the app writes its own access line (no raw paths or query strings)

    def address_string(self) -> str:
        return str(self.client_address[0])


def make_server(app: TrendaiApp, host: str = "127.0.0.1", port: int = 8000) -> ThreadingWSGIServer:
    """A listening (not yet serving) threaded WSGI server. ``OSError`` when the port cannot be opened."""
    server_class = _V6Server if ":" in host else ThreadingWSGIServer
    return wsgiref.simple_server.make_server(host.strip("[]"), int(port), app, server_class=server_class,
                                             handler_class=_QuietHandler)


def _base_url(server: ThreadingWSGIServer) -> str:
    host, port = server.server_address[:2]
    host = str(host)
    if host in ("0.0.0.0", "::", ""):
        host = "127.0.0.1"
    if ":" in host:
        host = f"[{host}]"
    return f"http://{host}:{port}"


def begin_graceful_shutdown(app: TrendaiApp, server: ThreadingWSGIServer, *,
                            grace: Optional[float] = None) -> threading.Thread:
    """E.14 steps 1-2: start draining now (the listener keeps serving polls, results and exports), and in a helper
    thread wait until no job runs or ``grace`` seconds (``shutdown_grace_s``) have passed, then ``app.close(0)`` and
    ``server.shutdown()``. Returns the helper thread."""
    app.begin_drain()
    limit = app.config.shutdown_grace_s if grace is None else float(grace)

    def wait_then_stop() -> None:
        deadline = time.monotonic() + limit
        while app.jobs.running() and time.monotonic() < deadline:
            time.sleep(0.2)
        if app.jobs.running():
            LOG.warning("shutdown grace period over: %d running job(s) abandoned", app.jobs.running())
        app.close(timeout=0)
        server.shutdown()

    helper = threading.Thread(target=wait_then_stop, name="trendai-shutdown", daemon=True)
    helper.start()
    return helper


def _flush_and_exit(code: int) -> None:
    # may run inside a signal handler while the main thread is half-way through a write to the same stream: a
    # nested write raises RuntimeError("reentrant call"), so every flush is best effort
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except Exception:  # noqa: BLE001 - exiting anyway
            pass
    try:
        logging.shutdown()
    except Exception:  # noqa: BLE001 - exiting anyway
        pass
    os._exit(code)


def serve_forever(app: TrendaiApp, host: str, port: int, *, on_ready: Optional[Callable[[str], None]] = None,
                  install_signal_handlers: bool = True) -> int:
    """Blocks. Calls ``on_ready(base_url)`` once listening (port 0 -> the real port). SIGINT/SIGTERM -> graceful
    shutdown (E.14); a second signal -> flush and ``os._exit(1)``. Returns the exit code (0); the caller then exits
    with ``os._exit(code)`` after flushing, so a still-running job thread cannot hold the process."""
    server = make_server(app, host, port)
    state: dict[str, Any] = {"signals": 0, "signum": None, "stopped": False}
    previous: dict[int, Any] = {}
    signalled = threading.Event()

    def on_signal(signum: int, _frame: Any) -> None:
        # Runs in the main thread between two bytecodes, possibly half-way through a write to stderr (a log line,
        # the CLI's start-up notes): no logging, no I/O and no locks the main thread may hold here -- a nested write
        # raises RuntimeError("reentrant call"). The drain itself starts in the watcher thread below.
        state["signals"] += 1
        if state["signals"] > 1:
            _flush_and_exit(1)
        state["signum"] = signum
        signalled.set()

    def watch_signals() -> None:
        signalled.wait()
        if state["stopped"]:
            return
        LOG.info("signal %d: draining", state["signum"])
        begin_graceful_shutdown(app, server)

    can_install = install_signal_handlers and threading.current_thread() is threading.main_thread()
    if can_install:
        threading.Thread(target=watch_signals, name="trendai-signals", daemon=True).start()
        for sig in (signal.SIGINT, signal.SIGTERM):
            previous[sig] = signal.signal(sig, on_signal)
    try:
        if on_ready is not None:
            on_ready(_base_url(server))
        server.serve_forever(poll_interval=0.2)
    finally:
        state["stopped"] = True
        signalled.set()                  # lets the watcher thread end when no signal came
        server.server_close()
        for sig, handler in previous.items():
            try:
                signal.signal(sig, handler)
            except (ValueError, TypeError):
                pass
    return 0


def start_background(app: TrendaiApp, host: str = "127.0.0.1",
                     port: int = 0) -> tuple[ThreadingWSGIServer, threading.Thread, str]:
    """For tests and E2E: serve in a daemon thread; returns ``(server, thread, base_url)``."""
    server = make_server(app, host, port)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.1}, name="trendai-server",
                              daemon=True)
    thread.start()
    return server, thread, _base_url(server)
