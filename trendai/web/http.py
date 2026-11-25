"""Request/response plumbing of the stdlib WSGI app: parsing the environ, body limits (C.0), the client key and the
request's own origin (C.0, E.8), JSON encoding (section 0), gzip (C.0) and the security headers (E.12)."""

from __future__ import annotations

import gzip
import ipaddress
import json
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional
from urllib.parse import parse_qs

from .errors import WebError

__all__ = ["Request", "Response", "dumps", "json_response", "text_response", "SECURITY_HEADERS", "APP_CSP",
           "JSON_TYPE", "ID_RX", "valid_id"]

JSON_TYPE = "application/json; charset=utf-8"

#: E.12, verbatim.
APP_CSP = ("default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data: blob:; font-src 'self'; "
           "connect-src 'self'; object-src 'none'; base-uri 'none'; form-action 'self'; frame-ancestors 'none'")
SECURITY_HEADERS: tuple[tuple[str, str], ...] = (
    ("Content-Security-Policy", APP_CSP),
    ("X-Content-Type-Options", "nosniff"),
    ("X-Frame-Options", "DENY"),
    ("Referrer-Policy", "no-referrer"),
    ("Permissions-Policy", "camera=(), microphone=(), geolocation=(), payment=(), usb=()"),
    ("Cross-Origin-Opener-Policy", "same-origin"),
    ("Cross-Origin-Resource-Policy", "same-origin"),
)
HSTS = ("Strict-Transport-Security", "max-age=31536000")

#: ids of results, shares, jobs and uploads (section 0)
ID_RX = re.compile(r"^[A-Za-z0-9_-]{22}$")
_GZIP_MIN = 1024
_REASONS = {200: "OK", 201: "Created", 202: "Accepted", 204: "No Content", 304: "Not Modified",
            308: "Permanent Redirect", 400: "Bad Request", 403: "Forbidden", 404: "Not Found",
            405: "Method Not Allowed", 410: "Gone", 411: "Length Required", 413: "Payload Too Large",
            415: "Unsupported Media Type", 422: "Unprocessable Entity", 429: "Too Many Requests",
            500: "Internal Server Error", 501: "Not Implemented", 502: "Bad Gateway", 503: "Service Unavailable",
            504: "Gateway Timeout"}


def valid_id(value: Any) -> bool:
    """``value`` is a string id of the section 0 form (validated before any store lookup)."""
    return isinstance(value, str) and ID_RX.match(value) is not None


def dumps(obj: Any) -> bytes:
    """Strict JSON (section 0): ``ensure_ascii=False, allow_nan=False`` -- a stray NaN raises ``ValueError``
    (answered as ``500 internal_error``) instead of producing JSON that ``JSON.parse`` rejects."""
    return json.dumps(obj, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")


@dataclass
class Response:
    """One response: always a single ``bytes`` body with ``Content-Length`` (E.3)."""

    status: int
    body: bytes = b""
    headers: list[tuple[str, str]] = field(default_factory=list)
    compressible: bool = False      # JSON bodies may be gzip-compressed (C.0); static files and exports never
    replaces_csp: bool = False      # HTML exports carry their own sandbox CSP (C.10)

    def header(self, name: str) -> Optional[str]:
        low = name.lower()
        for k, v in self.headers:
            if k.lower() == low:
                return v
        return None

    def set_header(self, name: str, value: str) -> None:
        low = name.lower()
        self.headers = [(k, v) for k, v in self.headers if k.lower() != low]
        self.headers.append((name, value))

    @property
    def status_line(self) -> str:
        return f"{self.status} {_REASONS.get(self.status, 'Unknown')}"


def json_response(obj: Any, status: int = 200, *, cache: str = "no-store",
                  headers: Iterable[tuple[str, str]] = ()) -> Response:
    """A JSON response with ``Cache-Control: cache`` (C.14)."""
    hdrs = [("Content-Type", JSON_TYPE), ("Cache-Control", cache), *headers]
    return Response(status, dumps(obj), hdrs, compressible=True)


def text_response(text: str, status: int = 200, *, cache: str = "no-store",
                  content_type: str = "text/plain; charset=utf-8") -> Response:
    return Response(status, text.encode("utf-8"), [("Content-Type", content_type), ("Cache-Control", cache)])


def maybe_gzip(resp: Response, accept_encoding: str) -> Response:
    """gzip JSON bodies larger than 1 KB when the client accepts it (C.0)."""
    if not resp.compressible or len(resp.body) <= _GZIP_MIN or "gzip" not in accept_encoding.lower():
        return resp
    resp.body = gzip.compress(resp.body, compresslevel=6)
    resp.set_header("Content-Encoding", "gzip")
    resp.set_header("Vary", "Accept-Encoding")
    return resp


def _client_key(addr: str) -> str:
    """The rate-limit key of an address: IPv6 addresses are keyed by their /64 (E.8)."""
    text = addr.strip().strip("[]")
    try:
        ip = ipaddress.ip_address(text)
    except ValueError:
        return text or "unknown"
    if ip.version == 6:
        if ip.ipv4_mapped is not None:
            return str(ip.ipv4_mapped)
        return str(ipaddress.ip_network(f"{ip}/64", strict=False))
    return str(ip)


def _origin_tuple(scheme: str, host: str) -> tuple[str, str]:
    scheme = scheme.lower().strip()
    host = host.lower().strip()
    default = {"http": ":80", "https": ":443"}.get(scheme)
    if default and host.endswith(default):
        host = host[: -len(default)]
    return scheme, host


class Request:
    """A parsed WSGI request. ``path`` is ``PATH_INFO`` (already percent-decoded by the server) re-read as UTF-8."""

    def __init__(self, environ: dict, *, trust_proxy: bool = False, public_url: Optional[str] = None) -> None:
        self.environ = environ
        self.method = str(environ.get("REQUEST_METHOD", "GET")).upper()
        raw_path = str(environ.get("PATH_INFO", "") or "")
        try:
            self.path = raw_path.encode("latin-1").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            self.path = raw_path
        self.query_string = str(environ.get("QUERY_STRING", "") or "")
        self._query: Optional[dict[str, list[str]]] = None
        self.trust_proxy = trust_proxy
        self.public_url = public_url
        self._body: Optional[bytes] = None

    # ------------------------------------------------------------------ headers and query
    def header(self, name: str, default: str = "") -> str:
        key = name.upper().replace("-", "_")
        if key in ("CONTENT_TYPE", "CONTENT_LENGTH"):
            return str(self.environ.get(key, default) or default)
        return str(self.environ.get("HTTP_" + key, default) or default)

    @property
    def query(self) -> dict[str, list[str]]:
        if self._query is None:
            self._query = parse_qs(self.query_string, keep_blank_values=True)
        return self._query

    def arg(self, name: str, default: Optional[str] = None) -> Optional[str]:
        values = self.query.get(name)
        return values[0] if values else default

    @property
    def hl(self) -> str:
        """``?hl=en`` -> ``"en"``; anything else ``"ko"`` (section 0)."""
        return "en" if self.arg("hl") == "en" else "ko"

    # ------------------------------------------------------------------ client and origin
    @property
    def client_key(self) -> str:
        """``REMOTE_ADDR``, or with ``trust_proxy`` the last ``X-Forwarded-For`` entry (E.8)."""
        addr = str(self.environ.get("REMOTE_ADDR", "") or "")
        if self.trust_proxy:
            xff = self.header("X-Forwarded-For")
            if xff.strip():
                last = [p.strip() for p in xff.split(",") if p.strip()]
                if last:
                    addr = last[-1]
        return _client_key(addr)

    def own_origin(self) -> tuple[str, str]:
        """The request's own origin (C.0): ``public_url``; else with ``trust_proxy`` the last
        ``X-Forwarded-Proto`` / ``X-Forwarded-Host`` values; else ``wsgi.url_scheme`` + ``Host``."""
        if self.public_url:
            scheme, _, rest = self.public_url.partition("://")
            return _origin_tuple(scheme, rest.split("/", 1)[0])
        scheme = str(self.environ.get("wsgi.url_scheme", "http") or "http")
        host = self.header("Host") or str(self.environ.get("SERVER_NAME", "") or "")
        if not self.header("Host") and self.environ.get("SERVER_PORT"):
            host = f"{host}:{self.environ['SERVER_PORT']}"
        if self.trust_proxy:
            proto = [p.strip() for p in self.header("X-Forwarded-Proto").split(",") if p.strip()]
            fhost = [p.strip() for p in self.header("X-Forwarded-Host").split(",") if p.strip()]
            if proto:
                scheme = proto[-1]
            if fhost:
                host = fhost[-1]
        return _origin_tuple(scheme, host)

    def origin_allowed(self) -> bool:
        """No ``Origin`` header, or one whose ``scheme://host[:port]`` equals :meth:`own_origin` (C.0)."""
        origin = self.header("Origin").strip()
        if not origin:
            return True
        if origin == "null":
            return False
        scheme, sep, rest = origin.partition("://")
        if not sep:
            return False
        return _origin_tuple(scheme, rest.split("/", 1)[0]) == self.own_origin()

    # ------------------------------------------------------------------ body
    @property
    def content_type(self) -> str:
        return self.header("Content-Type").split(";", 1)[0].strip().lower()

    def content_length(self, limit: int, *, max_mb: Any) -> int:
        """Validate ``Content-Length`` before reading (C.0): missing -> 411, above ``limit`` -> 413."""
        raw = self.header("Content-Length").strip()
        if not raw:
            raise WebError("length_required")
        try:
            length = int(raw)
        except ValueError:
            raise WebError("bad_request") from None
        if length < 0:
            raise WebError("bad_request")
        if length > limit:
            raise WebError("body_too_large", params={"max_mb": max_mb})
        return length

    def read_body(self, limit: int, *, max_mb: Any) -> bytes:
        """Read exactly ``Content-Length`` bytes (checked against ``limit`` first)."""
        if self._body is not None:
            return self._body
        length = self.content_length(limit, max_mb=max_mb)
        stream = self.environ.get("wsgi.input")
        chunks: list[bytes] = []
        remaining = length
        while remaining > 0 and stream is not None:
            chunk = stream.read(min(remaining, 1 << 16))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        if remaining > 0:
            raise WebError("bad_request")
        self._body = b"".join(chunks)
        return self._body

    def json_body(self, limit: int = 65536) -> dict:
        """The JSON object body of a POST (C.0): ``415`` unless ``application/json``; ``411``/``413`` per
        :meth:`read_body`; ``400 bad_request`` for invalid UTF-8/JSON or a non-object."""
        if self.content_type != "application/json":
            raise WebError("unsupported_media_type")
        body = self.read_body(limit, max_mb=round(limit / (1024 * 1024), 2))
        try:
            obj = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            raise WebError("bad_request") from None
        if not isinstance(obj, dict):
            raise WebError("bad_request")
        return obj
