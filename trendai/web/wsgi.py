"""WSGI entry point for gunicorn / waitress (spec B.3.2):

    gunicorn -w 1 --threads 8 trendai.web.wsgi:application        # never --preload: threads do not survive fork
    waitress-serve --threads=8 trendai.web.wsgi:application

Importing this module creates nothing (no data directory, no threads). The first request, under a module lock,
installs the warnings baseline, validates the configuration, configures logging and builds the app.
"""

from __future__ import annotations

import threading
from typing import Any, Callable, Iterable, Optional

__all__ = ["application"]

_LOCK = threading.Lock()
_APP: Optional[Callable[..., Iterable[bytes]]] = None


def _get_app() -> Callable[..., Iterable[bytes]]:
    global _APP
    if _APP is None:
        with _LOCK:
            if _APP is None:
                from .app import create_app
                from .config import WebConfig
                from .log import basic_config, install_warning_baseline

                install_warning_baseline()
                config = WebConfig.from_env()
                config.validate_public()
                basic_config(config.log_level)
                _APP = create_app(config)
    return _APP


def application(environ: dict, start_response: Callable[..., Any]) -> Iterable[bytes]:
    """Lazy WSGI callable: builds the app on first use, then delegates."""
    return _get_app()(environ, start_response)
