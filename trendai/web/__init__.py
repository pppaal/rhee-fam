"""trendai web app: a stdlib WSGI server, JSON API and single-page frontend over the trendai engine.

Binding contract: ``docs/WEB_SPEC.md``. Team working note on the module split: ``docs/WEB_INTERFACES.md``.

Public names (B.3.2): :func:`create_app`, :class:`TrendaiApp`, :class:`WebConfig`, :func:`install_warning_baseline`.
They are resolved lazily, so importing this package (or a light sub-module such as ``trendai.web.fmt``) stays cheap
and side-effect free: no data directory, no threads, no network.
"""

from __future__ import annotations

import importlib
from typing import Any

__all__ = ["create_app", "TrendaiApp", "WebConfig", "install_warning_baseline"]

_LAZY = {
    "create_app": ("app", "create_app"),
    "TrendaiApp": ("app", "TrendaiApp"),
    "WebConfig": ("config", "WebConfig"),
    "install_warning_baseline": ("log", "install_warning_baseline"),
}


def __getattr__(name: str) -> Any:
    try:
        module, attr = _LAZY[name]
    except KeyError:
        raise AttributeError(f"module 'trendai.web' has no attribute {name!r}") from None
    value = getattr(importlib.import_module(f".{module}", __name__), attr)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
