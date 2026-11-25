"""Static files and the SPA shell (spec E.4, C.14).

At start-up the static root is walked once into an index ``{relative posix path: entry}``; only paths in the index
are served and a request path is only ever used as a dict key, never joined to the filesystem, so traversal is
impossible by construction. ``dev/`` is indexed only in dev mode, where the index is rebuilt on every request.
"""

from __future__ import annotations

import hashlib
import importlib.resources
import logging
import os
import threading
from dataclasses import dataclass
from typing import Optional

__all__ = ["StaticIndex", "StaticEntry", "MIME_TYPES", "default_static_root"]

LOG = logging.getLogger("trendai.web")

MIME_TYPES: dict[str, str] = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".woff2": "font/woff2",
    ".txt": "text/plain; charset=utf-8",
    ".webmanifest": "application/manifest+json",
}


@dataclass(frozen=True)
class StaticEntry:
    path: str        # relative posix path
    content: bytes
    etag: str        # strong ETag: quoted sha1 hex of the bytes
    mime: str


def default_static_root() -> str:
    """``trendai/web/static`` of the installed package (a real directory: package data)."""
    return os.fspath(importlib.resources.files("trendai.web") / "static")


class StaticIndex:
    def __init__(self, root: Optional[str], *, dev: bool = False) -> None:
        self.root = os.path.realpath(root) if root else default_static_root()
        self.dev = dev
        self._lock = threading.Lock()
        self._warned = False
        self._entries: dict[str, StaticEntry] = {}
        self.rebuild()

    def rebuild(self) -> None:
        entries: dict[str, StaticEntry] = {}
        if os.path.isdir(self.root):
            for dirpath, dirnames, filenames in os.walk(self.root, followlinks=False):
                rel_dir = os.path.relpath(dirpath, self.root)
                parts = [] if rel_dir == "." else rel_dir.split(os.sep)
                dirnames[:] = sorted(d for d in dirnames if not d.startswith(".")
                                     and not (not parts and d == "dev" and not self.dev))
                for name in sorted(filenames):
                    if name.startswith("."):
                        continue
                    ext = os.path.splitext(name)[1].lower()
                    mime = MIME_TYPES.get(ext)
                    full = os.path.join(dirpath, name)
                    if mime is None or os.path.islink(full) or not os.path.isfile(full):
                        continue
                    rel = "/".join([*parts, name])
                    try:
                        with open(full, "rb") as fh:
                            content = fh.read()
                    except OSError:
                        continue
                    entries[rel] = StaticEntry(rel, content, f'"{hashlib.sha1(content).hexdigest()}"', mime)
        with self._lock:
            self._entries = entries
        if "index.html" not in entries and not self._warned:
            self._warned = True
            LOG.warning("static files missing: %s", self.root)

    def lookup(self, rel: str) -> Optional[StaticEntry]:
        """The entry for a relative path, or ``None``. With ``dev`` the index is rebuilt first."""
        if self.dev:
            self.rebuild()
        with self._lock:
            return self._entries.get(rel)

    @property
    def index_html(self) -> Optional[StaticEntry]:
        return self.lookup("index.html")

    def paths(self) -> list[str]:
        with self._lock:
            return sorted(self._entries)
