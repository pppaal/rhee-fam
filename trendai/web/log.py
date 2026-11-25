"""Logging for the web app (spec E.11) and the process-wide warnings baseline (E.13).

Loggers: ``trendai.web`` (general), ``trendai.web.access`` (one line per request), ``trendai.web.jobs`` (one line
per finished job). ``create_app`` adds no handlers; process owners call ``logging.basicConfig`` (:func:`basic_config`).

Privacy rules (E.11): no API keys or header values, no bodies, no query strings, no search terms, no upload file
names or contents, no full ids or tokens. Client IPs are logged as a 12-hex hash with a per-process salt that rotates
every KST day. Unexpected exceptions are logged with the exception *type* and the traceback frames only -- never
``str(exc)``, which may quote input data (pandas parser errors quote file lines, ``KeyError`` quotes a term).
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import logging
import secrets
import threading
import time
import traceback
import warnings
from typing import Callable, Optional

__all__ = ["install_warning_baseline", "reassert_warning_baseline", "basic_config", "IpHasher",
           "log_exception", "short_id", "LOGGER", "ACCESS_LOGGER", "JOBS_LOGGER"]

LOGGER = logging.getLogger("trendai.web")
ACCESS_LOGGER = logging.getLogger("trendai.web.access")
JOBS_LOGGER = logging.getLogger("trendai.web.jobs")

_KST = _dt.timedelta(hours=9)

# Process-level state of the warnings baseline. It belongs to the *process owner* (``trendai serve``, the lazy WSGI
# entry point, the E2E server), not to an app, so two apps in one process (tests) still share no app state.
_BASELINE_LOCK = threading.Lock()
_baseline_installed = False


def install_warning_baseline() -> None:
    """Process owners only (``trendai serve``, ``trendai.web.wsgi``, ``tests/e2e/server.py``; never ``create_app``):
    ignore every warning category -- the web app reads data notices through ``data.collect_notices()``, never
    through ``warnings`` (E.13) -- and remember that it did, so the sweeper re-asserts the same baseline every run."""
    global _baseline_installed
    with _BASELINE_LOCK:
        warnings.simplefilter("ignore")
        _baseline_installed = True


def reassert_warning_baseline() -> bool:
    """Called by the sweeper: re-install the baseline if (and only if) a process owner installed it, undoing filter
    state that interleaved third-party ``catch_warnings`` blocks may have left behind. Returns whether it did."""
    with _BASELINE_LOCK:
        if not _baseline_installed:
            return False
        warnings.simplefilter("ignore")
        return True


class _UtcFormatter(logging.Formatter):
    converter = time.gmtime


def basic_config(level: str = "INFO") -> None:
    """``logging.basicConfig`` with the E.11 format and UTC timestamps (for process owners)."""
    logging.basicConfig(level=getattr(logging, str(level).upper(), logging.INFO),
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    for handler in logging.getLogger().handlers:
        fmt = handler.formatter
        if fmt is not None and not isinstance(fmt, _UtcFormatter):
            handler.setFormatter(_UtcFormatter(fmt._fmt, fmt.datefmt))   # noqa: SLF001 - keep the format string


class IpHasher:
    """``ip=<12hex>``: the first 12 hex digits of ``sha256(daily_salt + client_key)``. The salt is random per
    process (per app) and replaced when the KST day changes, so hashes cannot be joined across days."""

    def __init__(self, clock: Callable[[], _dt.datetime]) -> None:
        self._clock = clock
        self._lock = threading.Lock()
        self._day: Optional[_dt.date] = None
        self._salt = b""

    def __call__(self, client_key: str) -> str:
        day = (self._clock() + _KST).date()
        with self._lock:
            if day != self._day:
                self._day, self._salt = day, secrets.token_bytes(16)
            salt = self._salt
        return hashlib.sha256(salt + str(client_key).encode("utf-8", "replace")).hexdigest()[:12]


def short_id(value: Optional[str]) -> str:
    """The first 6 characters of an id followed by ``…`` (full ids are never logged)."""
    return f"{str(value)[:6]}…" if value else "-"


def log_exception(logger: logging.Logger, what: str, exc: BaseException, *, rid: Optional[str] = None,
                  code: str = "internal_error") -> None:
    """Log an unexpected exception without its message: ``<what> rid=… error=<code> exc=<TypeName>`` plus the
    traceback frames (file, line and function of each, from ``traceback.extract_tb``; no source text, no local
    variables). Used for every unexpected error, including upload parsing and jobs on upload data, whose exception
    texts may quote cell contents (E.11)."""
    frames = traceback.extract_tb(exc.__traceback__) if exc.__traceback__ is not None else []
    stack = "\n".join(f'  File "{f.filename}", line {f.lineno}, in {f.name}' for f in frames)
    logger.error("%s rid=%s error=%s exc=%s\n%s", what, rid or "-", code, type(exc).__name__, stack)
