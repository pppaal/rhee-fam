"""WSGI entry point for gunicorn / waitress (spec B.3.2):

    gunicorn -w 1 --threads 8 trendai.web.wsgi:application        # never --preload: threads do not survive fork
    waitress-serve --threads=8 trendai.web.wsgi:application

Importing this module creates nothing (no data directory, no threads). The first request, under a module lock,
installs the warnings baseline, validates the configuration, configures logging and builds the app.

* The WSGI server, not ``TRENDAI_HOST``, decides where the app listens, so here the bind is unknown: the deployment
  counts as public (``TRENDAI_CONTACT`` required for the Wikimedia User-Agent, A.5) unless ``TRENDAI_HOST`` is set
  explicitly to a loopback address or ``TRENDAI_ALLOW_NO_CONTACT`` is set.
* An invalid configuration does not leave the app half built: the error is logged once (ERROR) and every request,
  ``/api/health`` included, is answered ``503`` with a JSON ``config_error`` envelope, so health checks fail
  visibly. The E.14 drain on SIGTERM is ``trendai serve``'s; gunicorn/waitress stop on their own terms (use
  gunicorn ``--graceful-timeout``).
"""

from __future__ import annotations

import json
import logging
import os
import threading
from typing import Any, Callable, Iterable, Optional

__all__ = ["application"]

_LOCK = threading.Lock()
_APP: Optional[Callable[..., Iterable[bytes]]] = None
_CONFIG_ERROR: Optional[BaseException] = None
_CONFIG_BODY = json.dumps({"error": {
    "code": "config_error", "status": 503,
    "message_ko": "서버 설정 오류로 지금은 서비스를 쓸 수 없어요. 운영자에게 알려 주세요.",
    "message_en": "The service is unavailable because of a server configuration error. Please tell the operator.",
    "params": {}, "fixes": [], "retry_after": None, "request_id": None}}, ensure_ascii=False).encode("utf-8")


def _build() -> Callable[..., Iterable[bytes]]:
    from .app import create_app
    from .config import WebConfig
    from .log import basic_config, install_warning_baseline

    install_warning_baseline()
    config = WebConfig.from_env()
    host_set = bool(str(os.environ.get("TRENDAI_HOST", "") or "").strip())
    if not host_set and not config.contact and not config.allow_no_contact:
        raise ValueError("TRENDAI_CONTACT must be set (operator e-mail or URL) for the WSGI entry point, whose bind "
                         "address the WSGI server decides; set TRENDAI_HOST=127.0.0.1 for a local-only server or "
                         "TRENDAI_ALLOW_NO_CONTACT=1 for private use")
    config.validate_public()
    basic_config(config.log_level)
    app = create_app(config)
    if not host_set:
        # the bind is unknown, so the loopback-only origin rule (DNS rebinding guard) cannot apply: a server bound to
        # 0.0.0.0 by gunicorn is reached under its own host name
        app.loopback_only = False
    return app


def _get_app() -> Optional[Callable[..., Iterable[bytes]]]:
    global _APP, _CONFIG_ERROR
    if _APP is None and _CONFIG_ERROR is None:
        with _LOCK:
            if _APP is None and _CONFIG_ERROR is None:
                try:
                    _APP = _build()
                except Exception as exc:  # noqa: BLE001 - answered as 503 config_error on every request
                    _CONFIG_ERROR = exc
                    logging.basicConfig()
                    logging.getLogger("trendai.web").error("trendai.web.wsgi: invalid configuration: %s", exc)
    return _APP


def application(environ: dict, start_response: Callable[..., Any]) -> Iterable[bytes]:
    """Lazy WSGI callable: builds the app on first use, then delegates (``503 config_error`` if it cannot)."""
    app = _get_app()
    if app is None:
        start_response("503 Service Unavailable", [("Content-Type", "application/json; charset=utf-8"),
                                                    ("Cache-Control", "no-store"),
                                                    ("Content-Length", str(len(_CONFIG_BODY)))])
        return [b"" if str(environ.get("REQUEST_METHOD", "")).upper() == "HEAD" else _CONFIG_BODY]
    return app(environ, start_response)
