"""The on-disk store (spec E.7): results, shares, data caches and budget counters under ``TRENDAI_DATA_DIR``.

* Directories are created ``0700`` and files ``0600``; every write is atomic (``O_EXCL`` temp file, ``fsync``,
  ``os.replace``) and a result or share directory appears in one ``os.replace`` of a fully written temp directory.
* Paths are built only from validated values (ids matching ``^[A-Za-z0-9_-]{22}$``, ``int`` series indexes, the
  literal languages, fixed file names, sha256 hex keys); :meth:`Store._path` also asserts that the final path stays
  under the root (``os.path.commonpath``).
* Reads treat an expired file -- or a result/share stored with another ``WEB_VM_SCHEMA`` -- as absent and delete
  expired ones best-effort. Unreadable files are tolerated (treated as absent) and swept after an hour.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import os
import re
import secrets
import shutil
import threading
import time
from typing import Any, Callable, Iterator, Mapping, Optional

from .http import valid_id

__all__ = ["Store", "utc_iso", "parse_iso", "SUBDIRS"]

LOG = logging.getLogger("trendai.web")

SUBDIRS = ("results", "shares", "cache/naver", "cache/wiki", "cache/results", "budget")
_FILE_RX = re.compile(r"^(?:result\.json|share\.json|vm-(?:ko|en)\.json|export-(?:ko|en)\.json|compare\.html"
                      r"|report-(?:0|[1-9][0-9]?)\.html|llm-(?:0|[1-9][0-9]?)\.json)$")
_KEY_RX = re.compile(r"^[0-9a-f]{64}$")
_BUDGET_RX = re.compile(r"^(?:naver-\d{4}-\d{2}|ai-\d{4}-\d{2}-\d{2})$")
_LANGS = ("ko", "en")
_CACHE_KINDS = ("naver", "wiki")
_STALE_UNREADABLE_S = 3600.0
_BUDGET_KEEP_DAYS = 400


def utc_iso(t: _dt.datetime) -> str:
    """``2026-11-25T00:12:03Z`` (section 0)."""
    if t.tzinfo is None:
        t = t.replace(tzinfo=_dt.timezone.utc)
    return t.astimezone(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_iso(text: Any) -> Optional[_dt.datetime]:
    """Parse a stored UTC timestamp; ``None`` when it is not one."""
    if not isinstance(text, str):
        return None
    s = text.strip()
    if s.endswith(("Z", "z")):
        s = s[:-1] + "+00:00"
    try:
        t = _dt.datetime.fromisoformat(s)
    except ValueError:
        return None
    if t.tzinfo is None:
        t = t.replace(tzinfo=_dt.timezone.utc)
    return t


def _vm_schema() -> int:
    from . import viewmodel   # read at call time: a bumped WEB_VM_SCHEMA invalidates stored results at once

    return int(viewmodel.WEB_VM_SCHEMA)


class Store:
    """All disk access of the web app. Thread-safe: writes are atomic and each file has one writer at a time."""

    def __init__(self, root: "str | os.PathLike", clock: Callable[[], _dt.datetime]) -> None:
        self.root = os.path.realpath(os.fspath(root))
        self.clock = clock
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ basics
    def ensure(self) -> None:
        """Create the root and the E.2 sub-directories with mode 0700 (missing parents of the root are created
        with the default mode, so ``trendai serve --data-dir ./a/b`` works on a fresh checkout)."""
        os.makedirs(os.path.dirname(self.root), exist_ok=True)
        self._mkdir(self.root)
        for sub in SUBDIRS:
            path = self.root
            for part in sub.split("/"):
                path = os.path.join(path, part)
                self._mkdir(path)

    @staticmethod
    def _mkdir(path: str) -> None:
        try:
            os.mkdir(path, 0o700)
        except FileExistsError:
            return
        os.chmod(path, 0o700)

    def writable(self) -> bool:
        """Whether a file can be created in the root (``/api/health``)."""
        probe = os.path.join(self.root, f".probe-{secrets.token_hex(6)}")
        try:
            fd = os.open(probe, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            os.close(fd)
            os.unlink(probe)
            return True
        except OSError:
            return False

    def _path(self, *parts: str) -> str:
        path = os.path.join(self.root, *parts)
        real = os.path.realpath(path)
        if os.path.commonpath([self.root, real]) != self.root:
            raise ValueError("store path escapes the data directory")
        return path

    def now(self) -> _dt.datetime:
        return self.clock()

    def _write_atomic(self, path: str, data: bytes) -> None:
        directory = os.path.dirname(path)
        self._makedirs(directory)
        tmp = os.path.join(directory, f".tmp-{secrets.token_hex(8)}")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(data)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def _makedirs(self, directory: str) -> None:
        rel = os.path.relpath(directory, self.root)
        path = self.root
        if rel == ".":
            return
        for part in rel.split(os.sep):
            path = os.path.join(path, part)
            self._mkdir(path)

    @staticmethod
    def encode(obj: Any) -> bytes:
        """Strict JSON for stored files (section 0)."""
        return json.dumps(obj, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")

    def write_json(self, path: str, obj: Any) -> None:
        self._write_atomic(path, self.encode(obj))

    @staticmethod
    def _read_json_file(path: str) -> Any:
        try:
            with open(path, "rb") as fh:
                return json.loads(fh.read().decode("utf-8"))
        except FileNotFoundError:
            return None
        except (OSError, UnicodeDecodeError, ValueError):
            return None

    def _expired(self, obj: Any) -> bool:
        exp = parse_iso(obj.get("expires_at")) if isinstance(obj, dict) else None
        return exp is None or exp <= self.now()

    @staticmethod
    def _rmtree(path: str) -> None:
        shutil.rmtree(path, ignore_errors=True)

    @staticmethod
    def _unlink(path: str) -> None:
        try:
            os.unlink(path)
        except OSError:
            pass

    # ------------------------------------------------------------------ results and shares
    def _dir(self, area: str, item_id: str) -> str:
        if area not in ("results", "shares") or not valid_id(item_id):
            raise ValueError("invalid id")
        return self._path(area, item_id)

    def _file(self, area: str, item_id: str, name: str) -> str:
        if not _FILE_RX.match(name):
            raise ValueError("invalid store file name")
        return self._path(area, item_id, name)

    def save_dir(self, area: str, item_id: str, files: Mapping[str, Any]) -> None:
        """Write ``files`` (name -> ``bytes`` | ``str`` | JSON object) into a temp directory and move it into
        place in one ``os.replace``: readers see the whole result/share or nothing (E.7)."""
        final = self._dir(area, item_id)
        parent = self._path(area)
        self._makedirs(parent)
        tmp = os.path.join(parent, f".tmp-{item_id}-{secrets.token_hex(4)}")
        self._mkdir(tmp)
        try:
            for name, content in files.items():
                parts = name.split("/")
                if len(parts) == 2 and parts[0] == "ai" and re.match(r"^(?:0|[1-9][0-9]?)-(?:ko|en)\.json$", parts[1]):
                    target = os.path.join(tmp, "ai", parts[1])
                elif _FILE_RX.match(name):
                    target = os.path.join(tmp, name)
                else:
                    raise ValueError("invalid store file name")
                if isinstance(content, bytes):
                    data = content
                elif isinstance(content, str):
                    data = content.encode("utf-8")
                else:
                    data = self.encode(content)
                self._write_atomic(target, data)
            if os.path.exists(final):
                self._rmtree(final)
            os.replace(tmp, final)
        except BaseException:
            self._rmtree(tmp)
            raise

    def _load_meta(self, area: str, item_id: str, meta_name: str) -> Optional[dict]:
        if not valid_id(item_id):
            return None
        path = self._file(area, item_id, meta_name)
        obj = self._read_json_file(path)
        if not isinstance(obj, dict):
            return None
        if self._expired(obj):
            self._rmtree(self._dir(area, item_id))
            return None
        schema = obj.get("schema") if area == "results" else obj.get("schema", (obj.get("result") or {}).get("schema"))
        if schema != _vm_schema():
            return None
        return obj

    def load_result(self, rid: str) -> Optional[dict]:
        """``result.json`` of a fresh result with the current schema, else ``None``."""
        return self._load_meta("results", rid, "result.json")

    def load_share(self, sid: str) -> Optional[dict]:
        """``share.json`` of a fresh share with the current schema, else ``None``."""
        return self._load_meta("shares", sid, "share.json")

    def read_item_json(self, area: str, item_id: str, name: str) -> Any:
        return self._read_json_file(self._file(area, item_id, name))

    def read_item_text(self, area: str, item_id: str, name: str) -> Optional[str]:
        try:
            with open(self._file(area, item_id, name), "rb") as fh:
                return fh.read().decode("utf-8")
        except (OSError, UnicodeDecodeError):
            return None

    def item_files(self, area: str, item_id: str) -> Iterator[tuple[str, bytes]]:
        """``(name, bytes)`` of every top-level file of a result/share (for share snapshots)."""
        directory = self._dir(area, item_id)
        try:
            names = sorted(os.listdir(directory))
        except OSError:
            return
        for name in names:
            if _FILE_RX.match(name):
                try:
                    with open(os.path.join(directory, name), "rb") as fh:
                        yield name, fh.read()
                except OSError:
                    continue

    def delete_item(self, area: str, item_id: str) -> None:
        self._rmtree(self._dir(area, item_id))

    # AI narratives (results/<id>/ai/<k>-<lang>.json)
    def _ai_path(self, rid: str, k: int, lang: str) -> str:
        if isinstance(k, bool) or not isinstance(k, int) or not 0 <= k < 100 or lang not in _LANGS:
            raise ValueError("invalid AI record path")
        return self._path("results", self._checked_id(rid), "ai", f"{k}-{lang}.json")

    @staticmethod
    def _checked_id(item_id: str) -> str:
        if not valid_id(item_id):
            raise ValueError("invalid id")
        return item_id

    def write_ai(self, rid: str, k: int, lang: str, record: Mapping[str, Any]) -> None:
        path = self._ai_path(rid, k, lang)
        if not os.path.isdir(self._dir("results", rid)):
            raise FileNotFoundError("result directory is gone")
        self.write_json(path, dict(record))

    def read_ai(self, area: str, item_id: str, k: int, lang: str) -> Optional[dict]:
        if isinstance(k, bool) or not isinstance(k, int) or not 0 <= k < 100 or lang not in _LANGS:
            return None
        obj = self._read_json_file(self._path(area, self._checked_id(item_id), "ai", f"{k}-{lang}.json"))
        return obj if isinstance(obj, dict) else None

    def ai_files(self, rid: str) -> Iterator[tuple[str, bytes]]:
        directory = os.path.join(self._dir("results", rid), "ai")
        try:
            names = sorted(os.listdir(directory))
        except OSError:
            return
        for name in names:
            if re.match(r"^(?:0|[1-9][0-9]?)-(?:ko|en)\.json$", name):
                try:
                    with open(os.path.join(directory, name), "rb") as fh:
                        yield f"ai/{name}", fh.read()
                except OSError:
                    continue

    # ------------------------------------------------------------------ caches
    def _cache_path(self, kind: str, key: str) -> str:
        if not _KEY_RX.match(key):
            raise ValueError("invalid cache key")
        if kind in _CACHE_KINDS:
            return self._path("cache", kind, key[:2], f"{key}.json")
        if kind == "results":
            return self._path("cache", "results", f"{key}.json")
        raise ValueError("invalid cache kind")

    def cache_get(self, kind: str, key: str) -> Optional[dict]:
        """A fresh cache entry (``expires_at`` in the future), else ``None`` (expired ones are deleted)."""
        path = self._cache_path(kind, key)
        obj = self._read_json_file(path)
        if not isinstance(obj, dict):
            return None
        if self._expired(obj):
            self._unlink(path)
            return None
        return obj

    def cache_put(self, kind: str, key: str, obj: Mapping[str, Any]) -> None:
        self.write_json(self._cache_path(kind, key), dict(obj))

    def cache_delete(self, kind: str, key: str) -> None:
        self._unlink(self._cache_path(kind, key))

    # ------------------------------------------------------------------ budget counters
    def _budget_path(self, name: str) -> str:
        if not _BUDGET_RX.match(name):
            raise ValueError("invalid budget file")
        return self._path("budget", f"{name}.json")

    def budget_read(self, name: str) -> Optional[dict]:
        obj = self._read_json_file(self._budget_path(name))
        return obj if isinstance(obj, dict) else None

    def budget_write(self, name: str, obj: Mapping[str, Any]) -> None:
        self.write_json(self._budget_path(name), dict(obj))

    # ------------------------------------------------------------------ sweeper
    def sweep(self, max_mb: Optional[float] = None) -> dict[str, int]:
        """Delete expired results/shares/cache entries, unreadable files older than an hour, abandoned temp
        files, budget files older than 400 days, and -- above ``max_mb`` -- the oldest result directories until the
        store is under 90 % of it (shares and budget files are never size-evicted). Returns counts."""
        counts = {"results": 0, "shares": 0, "cache": 0, "budget": 0, "evicted": 0}
        wall = time.time()
        for area, meta in (("results", "result.json"), ("shares", "share.json")):
            base = self._path(area)
            for name in self._listdir(base):
                path = os.path.join(base, name)
                if name.startswith(".tmp-"):
                    if self._older(path, wall):
                        self._rmtree(path)
                    continue
                if not valid_id(name):
                    continue
                obj = self._read_json_file(os.path.join(path, meta))
                if not isinstance(obj, dict):
                    if self._older(path, wall):
                        self._rmtree(path)
                        counts[area] += 1
                    continue
                if self._expired(obj):
                    self._rmtree(path)
                    counts[area] += 1
        for kind in _CACHE_KINDS:
            base = self._path("cache", kind)
            for sub in self._listdir(base):
                subdir = os.path.join(base, sub)
                for name in self._listdir(subdir):
                    counts["cache"] += self._sweep_cache_file(os.path.join(subdir, name), wall)
        base = self._path("cache", "results")
        for name in self._listdir(base):
            counts["cache"] += self._sweep_cache_file(os.path.join(base, name), wall)
        counts["budget"] = self._sweep_budget(wall)
        if max_mb is not None:
            counts["evicted"] = self._evict(float(max_mb) * 1024 * 1024)
        return counts

    @staticmethod
    def _listdir(path: str) -> list[str]:
        try:
            return sorted(os.listdir(path))
        except OSError:
            return []

    @staticmethod
    def _older(path: str, wall: float, seconds: float = _STALE_UNREADABLE_S) -> bool:
        try:
            return wall - os.stat(path).st_mtime > seconds
        except OSError:
            return False

    def _sweep_cache_file(self, path: str, wall: float) -> int:
        if os.path.isdir(path):
            return 0
        name = os.path.basename(path)
        if name.startswith(".tmp-"):
            if self._older(path, wall):
                self._unlink(path)
            return 0
        obj = self._read_json_file(path)
        if not isinstance(obj, dict):
            if self._older(path, wall):
                self._unlink(path)
                return 1
            return 0
        if self._expired(obj):
            self._unlink(path)
            return 1
        return 0

    def _sweep_budget(self, wall: float) -> int:
        removed = 0
        base = self._path("budget")
        today = self.now().date()
        for name in self._listdir(base):
            path = os.path.join(base, name)
            stem = name[:-5] if name.endswith(".json") else name
            m = re.match(r"^(?:naver-(\d{4})-(\d{2})|ai-(\d{4})-(\d{2})-(\d{2}))$", stem)
            if not m:
                if name.startswith(".tmp-") and self._older(path, wall):
                    self._unlink(path)
                continue
            try:
                if m.group(1):
                    day = _dt.date(int(m.group(1)), int(m.group(2)), 1)
                else:
                    day = _dt.date(int(m.group(3)), int(m.group(4)), int(m.group(5)))
            except ValueError:
                continue
            if (today - day).days > _BUDGET_KEEP_DAYS:
                self._unlink(path)
                removed += 1
        return removed

    def size_bytes(self) -> int:
        total = 0
        for dirpath, _dirs, files in os.walk(self.root):
            for name in files:
                try:
                    total += os.lstat(os.path.join(dirpath, name)).st_size
                except OSError:
                    continue
        return total

    def _evict(self, max_bytes: float) -> int:
        total = self.size_bytes()
        if total <= max_bytes:
            return 0
        target = 0.9 * max_bytes
        base = self._path("results")
        dirs = []
        for name in self._listdir(base):
            if not valid_id(name):
                continue
            path = os.path.join(base, name)
            obj = self._read_json_file(os.path.join(path, "result.json"))
            created = parse_iso(obj.get("created_at")) if isinstance(obj, dict) else None
            stamp = created.timestamp() if created is not None else self._mtime(path)
            dirs.append((stamp, path))
        evicted = 0
        for _stamp, path in sorted(dirs):
            if total <= target:
                break
            size = sum(os.lstat(os.path.join(dp, f)).st_size for dp, _d, fs in os.walk(path) for f in fs)
            self._rmtree(path)
            total -= size
            evicted += 1
        if evicted:
            LOG.warning("store above %d MB: evicted %d oldest results", int(max_bytes / 1024 / 1024), evicted)
        return evicted

    @staticmethod
    def _mtime(path: str) -> float:
        try:
            return os.stat(path).st_mtime
        except OSError:
            return 0.0
