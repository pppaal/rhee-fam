"""Test support for the web app (spec B.3.2, B.3.3, H.4.2): an in-process WSGI test client, fake fetchers, a frozen
clock, a minimal static tree, a hermetic app factory and the stdlib validator of ``vm.schema.json``.

Used by ``tests/test_web_*.py`` and ``tests/e2e/server.py``. Nothing here touches the network.
"""

from __future__ import annotations

import datetime as _dt
import gzip
import io
import json as _json
import math
import os
import pathlib
import threading
import time
import zlib
from typing import Any, Callable, Iterator, Mapping, Optional, Sequence
from urllib.parse import unquote

__all__ = ["TestResponse", "TestClient", "FakeNaver", "FakeWiki", "fake_llm", "FrozenClock", "write_static_fixture",
           "make_test_app", "multipart_body", "load_schema", "validate_schema", "assert_valid", "iter_strings",
           "analyze_and_wait", "monthly_csv"]


# --------------------------------------------------------------------------- test client

class TestResponse:
    """A response captured by :class:`TestClient`. ``headers`` keys are lower-case; ``body`` is the raw bytes (gzip
    when the response was compressed); :meth:`json` / :attr:`text` decompress first."""

    __test__ = False   # not a pytest test class

    def __init__(self, status: int, headers: dict[str, str], body: bytes, header_list: list[tuple[str, str]]) -> None:
        self.status = status
        self.headers = headers
        self.body = body
        self.header_list = header_list

    @property
    def content(self) -> bytes:
        if self.headers.get("content-encoding") == "gzip" and self.body:
            return gzip.decompress(self.body)
        return self.body

    def json(self) -> Any:
        return _json.loads(self.content.decode("utf-8"))

    @property
    def text(self) -> str:
        return self.content.decode("utf-8")

    def __repr__(self) -> str:
        return f"<TestResponse {self.status} {self.headers.get('content-type', '')}>"


class TestClient:
    """Calls the WSGI app in-process with a realistic environ (``PATH_INFO`` percent-decoded like ``wsgiref``)."""

    __test__ = False

    def __init__(self, app: Any, *, remote_addr: str = "127.0.0.1", host: str = "testserver") -> None:
        self.app = app
        self.remote_addr = remote_addr
        self.host = host

    def request(self, method: str, path: str, *, json: Any = None, data: Optional[bytes] = None,
                headers: Optional[Mapping[str, str]] = None) -> TestResponse:
        raw_path, _, query = path.partition("?")
        hdrs = {k.lower(): str(v) for k, v in (headers or {}).items()}
        body = b""
        if json is not None:
            body = _json.dumps(json, ensure_ascii=False).encode("utf-8")
            hdrs.setdefault("content-type", "application/json")
        elif data is not None:
            body = data
        environ: dict[str, Any] = {
            "REQUEST_METHOD": method.upper(),
            "SCRIPT_NAME": "",
            "PATH_INFO": unquote(raw_path, "iso-8859-1"),
            "QUERY_STRING": query,
            "SERVER_NAME": self.host,
            "SERVER_PORT": "80",
            "SERVER_PROTOCOL": "HTTP/1.1",
            "REMOTE_ADDR": self.remote_addr,
            "wsgi.version": (1, 0),
            "wsgi.url_scheme": "http",
            "wsgi.input": io.BytesIO(body),
            "wsgi.errors": io.StringIO(),
            "wsgi.multithread": True,
            "wsgi.multiprocess": False,
            "wsgi.run_once": False,
            "HTTP_HOST": self.host,
        }
        if body or method.upper() in ("POST", "PUT", "PATCH"):
            environ["CONTENT_LENGTH"] = str(len(body))
        for key, value in hdrs.items():
            name = key.upper().replace("-", "_")
            if name in ("CONTENT_TYPE", "CONTENT_LENGTH"):
                environ[name] = value
            else:
                environ["HTTP_" + name] = value
        captured: dict[str, Any] = {}

        def start_response(status: str, header_list: list, exc_info: Any = None) -> Callable:
            captured["status"] = int(status.split(" ", 1)[0])
            captured["headers"] = list(header_list)
            return lambda chunk: None

        chunks = self.app(environ, start_response)
        out = b"".join(chunks)
        header_list = captured["headers"]
        return TestResponse(captured["status"], {k.lower(): v for k, v in header_list}, out, header_list)

    def get(self, path: str, **kw: Any) -> TestResponse:
        return self.request("GET", path, **kw)

    def post(self, path: str, **kw: Any) -> TestResponse:
        return self.request("POST", path, **kw)

    def delete(self, path: str, **kw: Any) -> TestResponse:
        return self.request("DELETE", path, **kw)

    def head(self, path: str, **kw: Any) -> TestResponse:
        return self.request("HEAD", path, **kw)

    def upload(self, filename: str, content: bytes, *, field: str = "file", hl: str = "ko") -> TestResponse:
        """A multipart ``POST /api/upload`` with one file part."""
        body, ctype = multipart_body(filename, content, field=field)
        suffix = "" if hl == "ko" else f"?hl={hl}"
        return self.request("POST", f"/api/upload{suffix}", data=body, headers={"Content-Type": ctype})

    def analyze(self, body: Mapping[str, Any]) -> TestResponse:
        return self.post("/api/analyze", json=dict(body))

    def wait_job(self, job_id: str, *, timeout: float = 60.0, hl: str = "ko") -> dict:
        """Poll ``GET /api/jobs/<id>`` until the job is terminal; returns the last snapshot."""
        deadline = time.monotonic() + timeout
        while True:
            resp = self.get(f"/api/jobs/{job_id}?since=0&hl={hl}")
            if resp.status == 429:          # back off on the poll limit like the SPA does (C.13)
                time.sleep(0.25)
                continue
            snap = resp.json()
            if resp.status != 200:
                raise AssertionError(f"job poll failed: {resp.status} {snap}")
            if snap["status"] in ("done", "failed", "cancelled"):
                return snap
            if time.monotonic() > deadline:
                raise AssertionError(f"job {job_id} still {snap['status']} after {timeout} s")
            time.sleep(0.05)


def analyze_and_wait(client: TestClient, body: Mapping[str, Any], *, hl: str = "ko",
                     timeout: float = 60.0) -> tuple[dict, Optional[dict]]:
    """POST ``/api/analyze``, poll the job to its end and fetch the result: ``(last snapshot, envelope or None)``.
    A ``200 cached`` answer is returned as a synthetic done snapshot."""
    resp = client.post("/api/analyze", json=dict(body))
    if resp.status not in (200, 202):
        raise AssertionError(f"analyze refused: {resp.status} {resp.json()}")
    job_id = resp.json()["job_id"]
    snap = client.wait_job(job_id, timeout=timeout, hl=hl)
    env = None
    if snap["status"] == "done":
        res = client.get(f"/api/results/{job_id}?hl={hl}")
        if res.status != 200:
            raise AssertionError(f"result fetch failed: {res.status} {res.json()}")
        env = res.json()
    return snap, env


def monthly_csv(columns: Mapping[str, Sequence[Any]], *, start: str = "2022-01", date_col: str = "날짜",
                sep: str = ",") -> bytes:
    """A small monthly CSV: ``date_col`` from ``start`` plus the given columns (equal lengths)."""
    names = list(columns)
    n = len(next(iter(columns.values()))) if columns else 0
    year, month = (int(p) for p in start.split("-")[:2])
    lines = [sep.join([date_col, *names])]
    for i in range(n):
        y, m = year + (month - 1 + i) // 12, (month - 1 + i) % 12 + 1
        lines.append(sep.join([f"{y:04d}-{m:02d}", *(str(columns[c][i]) for c in names)]))
    return ("\n".join(lines) + "\n").encode("utf-8")


def multipart_body(filename: str, content: bytes, *, field: str = "file",
                   extra_fields: Optional[Mapping[str, str]] = None) -> tuple[bytes, str]:
    """``(body, content_type)`` of a multipart/form-data request with one file part."""
    boundary = "----trendaiTestBoundary7f3a9b1c"
    parts = []
    for name, value in (extra_fields or {}).items():
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode())
    head = (f'--{boundary}\r\nContent-Disposition: form-data; name="{field}"; filename="{filename}"\r\n'
            f"Content-Type: application/octet-stream\r\n\r\n").encode("utf-8")
    parts.append(head + content + b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


# --------------------------------------------------------------------------- clock

class FrozenClock:
    """A callable returning an aware UTC datetime that only moves with :meth:`advance` / :meth:`set`."""

    def __init__(self, start: _dt.datetime) -> None:
        if start.tzinfo is None:
            start = start.replace(tzinfo=_dt.timezone.utc)
        self._now = start.astimezone(_dt.timezone.utc)
        self._lock = threading.Lock()

    def __call__(self) -> _dt.datetime:
        with self._lock:
            return self._now

    def advance(self, seconds: float) -> _dt.datetime:
        with self._lock:
            self._now = self._now + _dt.timedelta(seconds=seconds)
            return self._now

    def set(self, when: _dt.datetime) -> None:
        if when.tzinfo is None:
            when = when.replace(tzinfo=_dt.timezone.utc)
        with self._lock:
            self._now = when.astimezone(_dt.timezone.utc)


# --------------------------------------------------------------------------- fake fetchers (H.4.2)

def _grid(time_unit: str, start: _dt.date, end: _dt.date) -> list[_dt.date]:
    out: list[_dt.date] = []
    if time_unit == "week":
        d = start + _dt.timedelta(days=(7 - start.weekday()) % 7)
        while d <= end:
            out.append(d)
            d += _dt.timedelta(days=7)
    elif time_unit == "month":
        d = start.replace(day=1)
        if d < start:
            d = (d.replace(day=28) + _dt.timedelta(days=4)).replace(day=1)
        while d <= end:
            out.append(d)
            d = (d.replace(day=28) + _dt.timedelta(days=4)).replace(day=1)
    else:
        d = start
        while d <= end:
            out.append(d)
            d += _dt.timedelta(days=1)
    return out


class FakeNaver:
    """Deterministic stand-in for ``connectors.fetch_naver_search_trend`` (H.4.2).

    Keyword ``에러`` -> ``conn_naver_server`` (500); ``인증오류`` -> ``conn_naver_auth`` (401); ``느린`` -> sleeps
    ``slow_seconds`` (2 s) first; keywords containing ``저검색`` are scaled by 0.03. ``fail_mode`` (tests) forces one
    of ``auth``, ``rate``, ``server``, ``network``, ``timeout``, ``bad``; ``delay`` sleeps before every call. Group
    names other than ``g1``..``gN`` in order raise ``conn_naver_bad_filter`` (``what="groupName"``)."""

    endpoint = "hub"

    def __init__(self, *, delay: float = 0.0, slow_seconds: float = 2.0, fail_mode: Optional[str] = None) -> None:
        self.calls: list[dict] = []
        self.delay = delay
        self.slow_seconds = slow_seconds
        self.fail_mode = fail_mode
        self._lock = threading.Lock()

    def __call__(self, groups: Sequence[Any], start: _dt.date, end: _dt.date, time_unit: str = "week",
                 device: Optional[str] = None, gender: Optional[str] = None, ages: Sequence[str] = ()) -> Any:
        import pandas as pd

        from ..connectors import ConnectorError, NaverResult
        from ..messages import msg

        with self._lock:
            self.calls.append({"groups": [(g.name, tuple(g.keywords)) for g in groups], "start": start, "end": end,
                               "time_unit": time_unit, "device": device, "gender": gender, "ages": tuple(ages)})
        for i, g in enumerate(groups):
            if g.name != f"g{i + 1}":
                raise ConnectorError(msg("conn_naver_bad_filter", what="groupName", value=repr(g.name)))
        if self.delay:
            time.sleep(self.delay)
        keywords = [k for g in groups for k in g.keywords]
        if any("느린" in k for k in keywords):
            time.sleep(self.slow_seconds)
        mode = self.fail_mode
        if mode is None and any("인증오류" in k for k in keywords):
            mode = "auth"
        if mode is None and any("에러" in k for k in keywords):
            mode = "server"
        if mode is not None:
            raise ConnectorError({
                "auth": msg("conn_naver_auth", status=401),
                "rate": msg("conn_naver_rate"),
                "server": msg("conn_naver_server", status=500),
                "network": msg("conn_naver_network", error="URLError: test"),
                "timeout": msg("conn_naver_timeout"),
                "bad": msg("conn_naver_bad_response"),
            }[mode])
        grid = _grid(time_unit, start, end)
        raw: dict[str, list[float]] = {}
        for g in groups:
            c = zlib.crc32("+".join(g.keywords).encode("utf-8"))
            vals = []
            for t, d in enumerate(grid):
                base = 50 + 30 * math.sin(2 * math.pi * (d.month - 7 + c % 3) / 12)
                v = max(0.5, base * (1 + 0.004 * t * ((c % 5) - 2)))
                if any("저검색" in k for k in g.keywords):
                    v *= 0.03
                vals.append(v)
            raw[g.name] = vals
        top = max((max(v) for v in raw.values() if v), default=1.0) or 1.0
        index = pd.DatetimeIndex([pd.Timestamp(d) for d in grid], name="date")
        series = {}
        for g in groups:
            s = pd.Series([round(v / top * 100.0, 5) for v in raw[g.name]], index=index, name=g.name, dtype=float)
            s.attrs["trendai_source"] = {"kind": "naver", "group": g.name, "keywords": list(g.keywords),
                                         "time_unit": time_unit, "device": device, "gender": gender,
                                         "ages": list(ages), "endpoint": "hub"}
            series[g.name] = s
        return NaverResult(series=series, endpoint="hub", time_unit=time_unit, start=start, end=end,
                           filled={g.name: 0 for g in groups}, raw_max=100.0)


class FakeWiki:
    """Deterministic stand-in for the Wikipedia fetcher (H.4.2): title ``없는문서`` -> ``conn_wiki_404``; ``느린문서``
    sleeps ``slow_seconds`` (2 s) first. ``delay`` sleeps before every call."""

    def __init__(self, *, delay: float = 0.0, slow_seconds: float = 2.0) -> None:
        self.calls: list[dict] = []
        self.delay = delay
        self.slow_seconds = slow_seconds
        self._lock = threading.Lock()

    def __call__(self, title: str, lang: str = "ko", start: Any = None, end: Any = None,
                 granularity: str = "monthly") -> Any:
        import pandas as pd

        from ..connectors import ConnectorError
        from ..messages import msg

        with self._lock:
            self.calls.append({"title": title, "lang": lang, "start": start, "end": end, "granularity": granularity})
        if self.delay:
            time.sleep(self.delay)
        if title == "느린문서":
            time.sleep(self.slow_seconds)
        if title == "없는문서":
            raise ConnectorError(msg("conn_wiki_404", title=title, lang=lang))
        s0 = pd.Timestamp(start).date() if start is not None else _dt.date(2016, 1, 1)
        e0 = pd.Timestamp(end).date() if end is not None else _dt.date(2026, 10, 31)
        unit = {"monthly": "month", "weekly": "week", "daily": "date"}.get(granularity, "month")
        if unit == "week":
            e0 = e0 - _dt.timedelta(days=(e0.weekday() + 1) % 7)
            grid = [d for d in _grid("week", s0, e0) if d + _dt.timedelta(days=6) <= e0]
        else:
            grid = _grid(unit, s0, e0)
        c = zlib.crc32(title.encode("utf-8")) % 7
        vals = [10000 * (1 + 0.3 * math.sin(2 * math.pi * d.month / 12 + c)) * (1 + 0.003 * t)
                for t, d in enumerate(grid)]
        index = pd.DatetimeIndex([pd.Timestamp(d) for d in grid], name="date")
        return pd.Series(vals, index=index, name=title, dtype=float)


def fake_llm(payload: dict, lang: str) -> str:
    """A canned AI narrative with the four ``SECTION_HEADINGS[lang]`` (H.4.2)."""
    from ..narrative import SECTION_HEADINGS

    h = SECTION_HEADINGS["en" if lang == "en" else "ko"]
    name = (payload.get("series") or {}).get("name", "") if isinstance(payload, dict) else ""
    return (f"**AI 테스트 해석** — {name}\n\n## {h[0]}\n- 테스트 문장입니다.\n\n## {h[1]}\n- …\n\n## {h[2]}\n- …\n\n"
            f"## {h[3]}\n- 테스트 주의입니다.")


# --------------------------------------------------------------------------- static fixture and app factory

_FIXTURE_FILES: dict[str, bytes] = {
    "index.html": (b'<!doctype html>\n<html lang="ko"><head><meta charset="utf-8"><title>trendai</title>\n'
                   b'<link rel="stylesheet" href="/static/css/tokens.css"></head>\n'
                   b'<body><div id="app"></div>'
                   b'<script type="module" src="/static/js/app.js"></script></body></html>\n'),
    "js/app.js": b"export const fixture = true;\n",
    "css/tokens.css": b":root { --bg: #fff; }\n",
    "favicon.svg": b'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16"><circle cx="8" cy="8" r="7"/></svg>\n',
    "fonts/pretendard-1.3.9/x.woff2": b"wOF2\x00\x01fixture",
    "dev/mock.html": b"<!doctype html><title>mock</title>\n",
}


def write_static_fixture(directory: "str | os.PathLike") -> pathlib.Path:
    """Write a minimal static tree for WP1's tests (index.html, js/app.js, css/tokens.css, favicon.svg,
    fonts/pretendard-1.3.9/x.woff2, dev/mock.html) and return the directory."""
    root = pathlib.Path(directory)
    for rel, content in _FIXTURE_FILES.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    return root


def make_test_app(data_dir: "str | os.PathLike", *, naver: bool = True, ai: bool = True,
                  clock: Optional[Callable[[], _dt.datetime]] = None,
                  static_dir: "str | os.PathLike | None" = None, **config_overrides: Any) -> Any:
    """Hermetic: ``WebConfig.from_env(environ={}, data_dir=..., static_dir=..., **config_overrides)`` -- the
    developer's ``TRENDAI_*``, ``NAVER_*`` and ``ANTHROPIC_API_KEY`` never leak in; fetchers are always the fakes
    (``naver=False`` -> ``fetchers.naver`` ``None``; ``ai=False`` -> ``fetchers.llm`` ``None``). ``static_dir`` ``None``
    -> the package static dir (E2E)."""
    from .app import create_app
    from .config import WebConfig
    from .fetchers import Fetchers

    config = WebConfig.from_env(environ={}, data_dir=os.fspath(data_dir),
                                static_dir=os.fspath(static_dir) if static_dir is not None else None,
                                **config_overrides)
    fakes = Fetchers(naver=FakeNaver() if naver else None, naver_endpoint="hub" if naver else None,
                     wiki=FakeWiki(), llm=fake_llm if ai else None)
    return create_app(config, fetchers=fakes, clock=clock)


# --------------------------------------------------------------------------- vm.schema.json validator (B.3.3)

_TYPES: dict[str, Callable[[Any], bool]] = {
    "object": lambda v: isinstance(v, dict),
    "array": lambda v: isinstance(v, list),
    "string": lambda v: isinstance(v, str),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "null": lambda v: v is None,
}


def load_schema() -> dict:
    """``trendai/web/vm.schema.json``."""
    path = pathlib.Path(__file__).with_name("vm.schema.json")
    return _json.loads(path.read_text(encoding="utf-8"))


def validate_schema(instance: Any, schema: Optional[dict] = None, *, entry: Optional[str] = None,
                    root: Optional[dict] = None, path: str = "$") -> list[str]:
    """Errors of ``instance`` against the B.3.3 JSON-Schema subset (empty list: valid). ``entry`` picks
    ``#/$defs/<entry>`` of ``vm.schema.json``. Null rule: a null instance is valid when ``type`` lists ``"null"``."""
    root = root if root is not None else load_schema()
    if schema is None:
        schema = {"$ref": f"#/$defs/{entry}"} if entry else root
    errors: list[str] = []
    types = schema.get("type")
    if types is not None:
        names = [types] if isinstance(types, str) else list(types)
        if instance is None and "null" in names:
            return errors
        if not any(_TYPES[n](instance) for n in names):
            return [f"{path}: expected {names}, got {type(instance).__name__}"]
    if "$ref" in schema:
        name = schema["$ref"].split("/")[-1]
        errors += validate_schema(instance, root["$defs"][name], root=root, path=path)
    if "enum" in schema and instance not in schema["enum"]:
        errors.append(f"{path}: {instance!r} not in {schema['enum']}")
    if isinstance(instance, dict):
        props = schema.get("properties", {})
        for key in schema.get("required", []):
            if key not in instance:
                errors.append(f"{path}: missing {key!r}")
        if schema.get("additionalProperties") is False:
            for key in instance:
                if key not in props:
                    errors.append(f"{path}: unexpected {key!r}")
        for key, sub in props.items():
            if key in instance:
                errors += validate_schema(instance[key], sub, root=root, path=f"{path}.{key}")
    if isinstance(instance, list):
        if "minItems" in schema and len(instance) < schema["minItems"]:
            errors.append(f"{path}: fewer than {schema['minItems']} items")
        if "maxItems" in schema and len(instance) > schema["maxItems"]:
            errors.append(f"{path}: more than {schema['maxItems']} items")
        if "items" in schema:
            for i, item in enumerate(instance):
                errors += validate_schema(item, schema["items"], root=root, path=f"{path}[{i}]")
    return errors


def assert_valid(instance: Any, entry: str) -> None:
    """Assert that ``instance`` validates against ``#/$defs/<entry>`` and is strict JSON."""
    errors = validate_schema(instance, entry=entry)
    assert not errors, "\n".join(errors[:20])
    _json.dumps(instance, ensure_ascii=False, allow_nan=False)


def iter_strings(obj: Any) -> Iterator[str]:
    """Every string inside a JSON-like value."""
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from iter_strings(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from iter_strings(v)
