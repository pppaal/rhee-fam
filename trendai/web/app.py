"""The WSGI application (spec B.3.2, E.2, E.3): app factory, router, static/SPA handlers and the request wrapper
(request id, security headers, error envelopes, HEAD, gzip, access log).

There is no module-level mutable state: every cache, limiter, budget, executor and the sweeper belong to one
:class:`TrendaiApp`, so several apps can live in one process (tests).
"""

from __future__ import annotations

import datetime as _dt
import re
import secrets
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Optional

from . import api
from .analyze import SingleFlight
from .budget import AiBudget, NaverBudget
from .config import WebConfig
from .errors import WebError, error_envelope, store_error
from .fetchers import Fetchers, default_fetchers
from .http import HSTS, SECURITY_HEADERS, Request, Response, json_response, maybe_gzip, text_response
from .jobs import JobManager
from .log import ACCESS_LOGGER, LOGGER, IpHasher, log_exception, reassert_warning_baseline
from .ratelimit import RateLimiter
from .static_files import StaticIndex
from .store import Store
from .uploads import UploadCache

__all__ = ["TrendaiApp", "create_app", "ROUTES", "SWEEP_INTERVAL_S"]

SWEEP_INTERVAL_S = 600.0
_PARSE_SLOTS = 2
#: concurrent heavy reads (stored VMs of results/shares/job snapshots, exports): beyond these a request waits up to
#: :data:`_HEAVY_WAIT_S`, then gets ``503 server_busy`` -- one client cannot occupy every request thread with them
_HEAVY_SLOTS = 4
_HEAVY_WAIT_S = 10.0
_METHOD_RX = re.compile(r"^[A-Z]{1,16}$")
_MISSING_STATIC = "trendai: static files are missing"
_ROBOTS = "User-agent: *\nDisallow: /api/\nDisallow: /r/\nDisallow: /s/\n"
_ID = r"[A-Za-z0-9_-]{22}"
_PAGES = r"(?:explore|upload|samples|method|sources|privacy)"


@dataclass(frozen=True)
class Route:
    methods: tuple[str, ...]          # () = any method
    regex: "re.Pattern[str]"
    handler: str                      # method name on TrendaiApp or a function name in api
    template: str                     # logged instead of the raw path (E.11)


def _r(methods: Iterable[str], pattern: str, handler: str, template: str) -> Route:
    return Route(tuple(methods), re.compile(pattern), handler, template)


GH = ("GET", "HEAD")
#: E.3, in order: the first regex that matches decides; a method mismatch is 405 with ``Allow``.
ROUTES: tuple[Route, ...] = (
    _r(GH, rf"^/{_PAGES}?$", "spa", "/{page}"),
    _r(GH, rf"^/(?:r|s)/{_ID}$", "spa_noindex", "/{r|s}/{id}"),
    _r(GH, rf"^/{_PAGES}/$", "slash_redirect", "/{page}/"),
    _r(GH, r"^/static/(?P<path>.+)$", "static", "/static/{path}"),
    _r(GH, r"^/favicon\.ico$", "favicon", "/favicon.ico"),
    _r(GH, r"^/robots\.txt$", "robots", "/robots.txt"),
    _r(GH, r"^/api/health$", "api.health", "/api/health"),
    _r(("GET",), r"^/api/config$", "api.get_config", "/api/config"),
    _r(("GET",), r"^/api/samples$", "api.get_samples", "/api/samples"),
    _r(("POST",), r"^/api/upload$", "api.upload", "/api/upload"),
    _r(("POST",), r"^/api/analyze$", "api.post_analyze", "/api/analyze"),
    _r(("POST",), r"^/api/narrative$", "api.post_narrative", "/api/narrative"),
    _r(("POST",), r"^/api/share$", "api.post_share", "/api/share"),
    _r(("GET", "DELETE"), rf"^/api/jobs/(?P<id>{_ID})$", "jobs", "/api/jobs/{id}"),
    _r(("GET",), rf"^/api/results/(?P<id>{_ID})$", "api.get_result", "/api/results/{id}"),
    _r(("GET",), rf"^/api/shared/(?P<sid>{_ID})$", "api.get_shared", "/api/shared/{sid}"),
    _r(("DELETE",), rf"^/api/share/(?P<sid>{_ID})$", "api.delete_share", "/api/share/{sid}"),
    _r(("GET",), rf"^/api/export/(?:s/(?P<sid>{_ID})|(?P<id>{_ID}))(?:/(?P<k>0|[1-9][0-9]?))?"
                 r"\.(?P<fmt>csv|xlsx|html|json)$", "api.export_file", "/api/export/{id}[/{k}].{fmt}"),
    _r(("GET",), rf"^/api/export/(?:s/(?P<sid>{_ID})|(?P<id>{_ID}))/values\.csv$", "api.export_values",
       "/api/export/{id}/values.csv"),
    _r((), r"^/api/", "api_not_found", "/api/*"),
    _r(GH, r"", "spa_404", "*"),
)


def _utc_now() -> _dt.datetime:
    return _dt.datetime.now(_dt.timezone.utc)


def _bad_path(path: str) -> bool:
    return (not path.startswith("/") or "\x00" in path or "\\" in path
            or any(seg == ".." for seg in path.split("/")))


class TrendaiApp:
    """The WSGI callable. Build it with :func:`create_app`."""

    def __init__(self, config: WebConfig, fetchers: Fetchers, clock: Callable[[], _dt.datetime]) -> None:
        self.config = config
        self.fetchers = fetchers
        self.clock = clock
        self.started_at = clock()
        self.store = Store(config.data_dir, clock)
        self.store.ensure()
        if not self.store.writable():
            LOGGER.warning("data directory is not writable: results cannot be stored")
        self.static = StaticIndex(config.static_dir, dev=config.dev)
        self.samples = api.build_samples()
        self.limiter = RateLimiter(clock)
        self.naver_budget = NaverBudget(self.store, clock, config.naver_monthly_limit)
        self.ai_budget = AiBudget(self.store, clock, config.ai_daily_limit)
        self.uploads = UploadCache(clock)
        self.parse_slots = threading.BoundedSemaphore(_PARSE_SLOTS)
        self.parse_clients: set[str] = set()          # clients holding a parse slot (one each, SEC: E.17)
        self.parse_lock = threading.Lock()
        self.heavy_slots = threading.BoundedSemaphore(_HEAVY_SLOTS)
        self.heavy_wait_s = _HEAVY_WAIT_S
        # a loopback-only server (no public_url, no proxy) accepts browser POST/DELETE only from loopback origins
        self.loopback_only = not config.is_public() and not config.trust_proxy
        self.jobs = JobManager(clock, workers=config.workers, max_queued=config.max_queued,
                               timeout_s=config.job_timeout_s)
        self.naver_flight = SingleFlight()
        self.ip_hash = IpHasher(clock)
        self._draining = False
        self._closed = False
        self._close_lock = threading.Lock()
        self._stop = threading.Event()
        self._sweeper = threading.Thread(target=self._sweep_loop, name="trendai-sweeper", daemon=True)
        self._sweeper.start()

    # ------------------------------------------------------------------ feature state
    def ai_enabled(self) -> bool:
        """``features.ai`` (C.2): an LLM fetcher exists and ``TRENDAI_AI`` is not ``off``."""
        return self.fetchers.llm is not None and self.config.ai != "off"

    def ai_reason(self) -> Optional[str]:
        if self.config.ai == "off":
            return "disabled"
        return None if self.fetchers.llm is not None else "unconfigured"

    def naver_available(self) -> bool:
        """Naver is configured and the monthly budget is not blocked (C.2)."""
        return self.fetchers.naver is not None and self.naver_budget.state()[2] != "blocked"

    # ------------------------------------------------------------------ lifecycle
    @property
    def draining(self) -> bool:
        return self._draining

    def begin_drain(self) -> None:
        """E.14 step 1: health answers 503, new analyze/narrative/upload requests 503, queued jobs are cancelled;
        everything else (polls, results, exports) keeps working."""
        self._draining = True
        self.jobs.drain()

    def close(self, timeout: float = 30.0) -> None:
        """Drain: refuse new jobs, wait for running ones up to ``timeout``, cancel queued, stop the sweeper.
        Idempotent."""
        with self._close_lock:
            if self._closed:
                return
            self._closed = True
        self.begin_drain()
        self._stop.set()
        self.jobs.close(timeout)
        if self._sweeper.is_alive() and self._sweeper is not threading.current_thread():
            self._sweeper.join(timeout=1.0)

    def sweep_once(self) -> None:
        """One sweeper run (E.7, E.8, E.13): expired files and caches, idle buckets, old jobs, warnings baseline."""
        reassert_warning_baseline()
        self.store.sweep(max_mb=self.config.store_max_mb)
        self.limiter.sweep()
        self.uploads.sweep()
        self.jobs.sweep()

    def _sweep_loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.sweep_once()
            except Exception as exc:  # noqa: BLE001 - the sweeper must keep running
                log_exception(LOGGER, "sweeper run failed", exc)
            self._stop.wait(SWEEP_INTERVAL_S)

    # ------------------------------------------------------------------ WSGI
    def __call__(self, environ: dict, start_response: Callable) -> Iterable[bytes]:
        rid = secrets.token_hex(4)
        started = time.monotonic()
        req = Request(environ, trust_proxy=self.config.trust_proxy, public_url=self.config.public_url,
                      loopback_only=self.loopback_only)
        # the method is the only free-form request text in the access line: control characters (ANSI escapes,
        # BEL) never reach the log
        method = req.method if _METHOD_RX.match(req.method) else "OTHER"
        holder = {"template": "-"}
        try:
            resp = self._dispatch(req, rid, holder)
        except WebError as err:
            resp = self._error(err, rid)
        except Exception as exc:  # noqa: BLE001 - every unexpected error is a 500 envelope, never a traceback
            mapped = store_error(exc)
            log_exception(LOGGER, f"request {method} {holder['template']}", exc, rid=rid,
                          code=mapped.code if mapped is not None else "internal_error")
            resp = self._error(mapped or WebError("internal_error"), rid)
        resp = maybe_gzip(resp, req.header("Accept-Encoding"))
        headers = list(resp.headers)
        if not resp.replaces_csp:      # only HTML exports bring their own (sandbox) CSP (C.10)
            headers = [(k, v) for k, v in headers if k.lower() != "content-security-policy"]
        names = {k.lower() for k, _ in headers}
        headers.extend((name, value) for name, value in SECURITY_HEADERS if name.lower() not in names)
        if self.config.hsts:
            headers.append(HSTS)
        headers.append(("X-Request-Id", rid))
        body = resp.body if resp.status not in (204, 304) else b""
        headers.append(("Content-Length", str(len(body))))
        if req.method == "HEAD":
            body = b""
        start_response(resp.status_line, headers)
        ms = int((time.monotonic() - started) * 1000)
        ACCESS_LOGGER.info("rid=%s ip=%s %s %s %d %dms %dB", rid, self.ip_hash(req.client_key), method,
                           holder["template"], resp.status, ms, len(body))
        return [body]

    def _error(self, err: WebError, rid: str) -> Response:
        resp = json_response(error_envelope(err, rid, naver_available=self._naver_available_safe()), err.status,
                             cache="no-store", headers=err.headers)
        if err.retry_after is not None and err.status in (429, 503):
            resp.set_header("Retry-After", str(err.retry_after))
        return resp

    def _naver_available_safe(self) -> bool:
        try:
            return self.naver_available()
        except Exception:  # noqa: BLE001 - an unreadable budget file must not break error rendering
            return self.fetchers.naver is not None

    def _dispatch(self, req: Request, rid: str, holder: dict) -> Response:
        path = req.path
        if _bad_path(path):
            holder["template"] = "(rejected)"
            raise WebError("not_found")
        for route in ROUTES:
            m = route.regex.match(path)
            if m is None:
                continue
            holder["template"] = route.template
            if route.methods and req.method not in route.methods:
                raise WebError("method_not_allowed", headers=[("Allow", ", ".join(route.methods))])
            if req.method in ("POST", "DELETE") and not req.origin_allowed():
                raise WebError("cross_origin")
            params = {k: v for k, v in m.groupdict().items() if v is not None}
            if route.handler.startswith("api."):
                return getattr(api, route.handler[4:])(self, req, rid, **params)
            return getattr(self, "_h_" + route.handler)(req, rid, **params)
        raise WebError("not_found")   # pragma: no cover - the last route matches everything

    # ------------------------------------------------------------------ non-API handlers
    def _missing_static(self) -> Response:
        return text_response(_MISSING_STATIC, 503, cache="no-store")

    def _index(self, req: Request, *, status: int = 200, noindex: bool = False) -> Response:
        entry = self.static.index_html
        if entry is None:
            return self._missing_static()
        headers = [("Content-Type", entry.mime), ("Cache-Control", "no-cache"), ("ETag", entry.etag)]
        if noindex:
            headers.append(("X-Robots-Tag", "noindex, nofollow"))
        if status == 200 and req.header("If-None-Match").strip() == entry.etag:
            return Response(304, b"", headers)
        return Response(status, entry.content, headers)

    def _h_spa(self, req: Request, rid: str) -> Response:
        return self._index(req)

    def _h_spa_noindex(self, req: Request, rid: str) -> Response:
        return self._index(req, noindex=True)

    def _h_spa_404(self, req: Request, rid: str) -> Response:
        return self._index(req, status=404)

    def _h_slash_redirect(self, req: Request, rid: str) -> Response:
        location = req.path.rstrip("/") + (f"?{req.query_string}" if req.query_string else "")
        return Response(308, b"", [("Location", location), ("Cache-Control", "no-store")])

    def _h_static(self, req: Request, rid: str, path: str) -> Response:
        entry = self.static.lookup(path)
        if entry is None:
            raise WebError("not_found")
        cache = "public, max-age=31536000, immutable" if path.startswith("fonts/") else "no-cache"
        headers = [("Content-Type", entry.mime), ("Cache-Control", cache), ("ETag", entry.etag)]
        if req.header("If-None-Match").strip() == entry.etag:
            return Response(304, b"", headers)
        return Response(200, entry.content, headers)

    def _h_favicon(self, req: Request, rid: str) -> Response:
        if self.static.index_html is None:
            return self._missing_static()
        entry = self.static.lookup("favicon.svg")
        if entry is None:
            raise WebError("not_found")
        return Response(200, entry.content, [("Content-Type", "image/svg+xml"), ("Cache-Control", "no-cache"),
                                             ("ETag", entry.etag)])

    def _h_robots(self, req: Request, rid: str) -> Response:
        return text_response(_ROBOTS, cache="no-cache")

    def _h_jobs(self, req: Request, rid: str, id: str) -> Response:   # noqa: A002 - route parameter name
        if req.method == "DELETE":
            return api.delete_job(self, req, rid, id)
        return api.get_job(self, req, rid, id)

    def _h_api_not_found(self, req: Request, rid: str) -> Response:
        raise WebError("not_found")


def create_app(config: "WebConfig | Mapping[str, Any] | None" = None, *, fetchers: Optional[Fetchers] = None,
               clock: Optional[Callable[[], _dt.datetime]] = None) -> TrendaiApp:
    """config ``None`` -> ``WebConfig.from_env()``; a Mapping -> ``WebConfig.from_env(**mapping)``. fetchers ``None``
    -> ``fetchers.default_fetchers(config)``. clock returns an aware UTC datetime (default ``datetime.now(utc)``).
    Creates the data dir, starts the sweeper thread and the (lazily threaded) executors. A static dir without
    ``index.html`` is not an error: the app logs a WARNING and answers SPA routes with 503 (E.4)."""
    if config is None:
        cfg = WebConfig.from_env()
    elif isinstance(config, WebConfig):
        cfg = config
    else:
        cfg = WebConfig.from_env(**dict(config))
    return TrendaiApp(cfg, fetchers if fetchers is not None else default_fetchers(cfg), clock or _utc_now)
