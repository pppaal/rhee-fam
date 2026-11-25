"""Online data connectors (optional).

* :func:`fetch_naver_search_trend` -- Naver DataLab search trends (NAVER API HUB, or the legacy
  Naver Developers endpoint): up to 5 keyword groups in one request, relative ratios (max = 100).
* :func:`fetch_wikipedia_pageviews` -- Wikimedia REST pageviews API (no key needed; daily, weekly
  or monthly; uses ``requests`` when installed, else the stdlib :class:`UrllibSession`, or any object
  with a compatible ``.get``).
* :func:`fetch_google_trends` -- Google Trends interest over time via the
  optional ``pytrends`` package.

They return raw float ``pd.Series`` with a sorted ``DatetimeIndex``; pass them
through :func:`trendai.data.prepare` (or straight into ``analyze``/``run``).
Every failure (missing package or key, HTTP error, network error, timeout, empty result) is
raised as :class:`ConnectorError` with a message that says what to do: a
catalog message (:mod:`trendai.messages`), Korean as ``str(exc)`` and English
as ``exc.text('en')`` (the CLI follows ``--lang``).  Upstream details for operator logs (never
keys or headers) are in ``exc.log_detail``.

:func:`period_bounds` turns a range chip (``'5y'``, ``'all'``, ``'2024-01-01~2024-12-31'``) into the
first and last day of whole, completed days / Monday-Sunday weeks / months; :func:`user_agent`
builds the descriptive User-Agent (version, public URL, operator contact) that Wikimedia asks for.
Requests carry no personal data.
"""

from __future__ import annotations

import datetime as _dt
import email.utils
import hashlib
import http.client
import json as _json
import math
import os
import re
import ssl
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence
from urllib.parse import quote, urlsplit

import numpy as np
import pandas as pd

from . import __version__
from .messages import Localised, msg

__all__ = [
    "ConnectorError", "USER_AGENT", "wikipedia_pageviews_url", "fetch_wikipedia_pageviews",
    "fetch_google_trends",
    # web app (docs/WEB_SPEC.md B.3.1)
    "NAVER_HUB_URL", "NAVER_LEGACY_URL", "NAVER_EARLIEST", "WIKI_EARLIEST", "NAVER_MAX_GROUPS",
    "NAVER_MAX_KEYWORDS", "NAVER_AGE_CODES", "TIMEOUT", "HttpResponse", "UrllibSession", "user_agent",
    "period_bounds", "NaverCredentials", "NaverGroup", "NaverResult", "build_naver_body", "naver_cache_key",
    "fetch_naver_search_trend",
]

USER_AGENT = "trendai/0.1 (time-series trend analysis tool)"
WIKIMEDIA_PAGEVIEWS = "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article"
DEFAULT_START = "20160101"
TIMEOUT = 20                       # seconds, every outbound request
_GRANULARITIES = ("daily", "monthly")   # what the Wikimedia API serves (weekly = summed daily data)

NAVER_HUB_URL = "https://naverapihub.apigw.ntruss.com/search-trend/v1/search"
NAVER_LEGACY_URL = "https://openapi.naver.com/v1/datalab/search"
NAVER_EARLIEST = _dt.date(2016, 1, 1)
WIKI_EARLIEST = _dt.date(2016, 1, 1)
NAVER_MAX_GROUPS = 5
NAVER_MAX_KEYWORDS = 5             # the HUB docs say both 5 and 20; the safe value
NAVER_AGE_CODES = ("1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11")
_NAVER_TIME_UNITS = ("date", "week", "month")
_NAVER_DEVICES = ("pc", "mo")
_NAVER_GENDERS = ("m", "f")
_NAVER_GROUP_NAME_MAX = 100


class ConnectorError(Localised, RuntimeError):
    """Fetching data from an online source failed.

    Raised with a catalog message (:func:`trendai.messages.msg`, codes ``conn_*``): ``.code`` /
    ``.params`` name it, ``.text('en')`` renders it in English; ``str(exc)`` is the Korean text.
    ``log_detail`` (``None`` or at most 200 characters) carries the upstream error code / message for
    operator logs only -- never request headers or keys."""

    log_detail: Optional[str] = None
    #: seconds from the upstream ``Retry-After`` header of a 429 answer (``conn_wiki_429``, ``conn_naver_rate``),
    #: or ``None`` when the source gave none
    retry_after: Optional[int] = None


def _retry_after(resp: Any) -> Optional[int]:
    """The ``Retry-After`` header of ``resp`` in whole seconds (delta-seconds or an HTTP date), else ``None``."""
    headers = getattr(resp, "headers", None)
    if headers is None:
        return None
    try:
        raw = headers.get("retry-after") or headers.get("Retry-After")
    except Exception:  # noqa: BLE001 - an odd headers object is the same as no header
        return None
    if raw is None:
        return None
    text = str(raw).strip()
    if text.isdigit():
        return int(text)
    try:
        when = email.utils.parsedate_to_datetime(text)
    except (TypeError, ValueError, IndexError):
        return None
    if when is None:
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=_dt.timezone.utc)
    return max(0, math.ceil((when - _dt.datetime.now(_dt.timezone.utc)).total_seconds()))


def _connector_error(code: str, log_detail: Optional[str] = None, **params: Any) -> ConnectorError:
    exc = ConnectorError(msg(code, **params))
    if log_detail:
        exc.log_detail = log_detail[:200]
    return exc


def _is_timeout(exc: BaseException) -> bool:
    """A timeout from any HTTP client: TimeoutError (= socket.timeout), a class named *Timeout*
    (requests' ConnectTimeout / ReadTimeout), or a URLError whose reason is one of those."""
    for e in (exc, getattr(exc, "reason", None)):
        if e is not None and (isinstance(e, TimeoutError) or "Timeout" in type(e).__name__):
            return True
    return False


# --------------------------------------------------------------------------- stdlib HTTP


class HttpResponse:
    """What :class:`UrllibSession` returns: ``status_code``, ``headers`` (lower-case keys), ``content``."""

    __slots__ = ("status_code", "headers", "content")

    def __init__(self, status_code: int, headers: Mapping[str, str], content: bytes) -> None:
        self.status_code = int(status_code)
        self.headers = {str(k).lower(): str(v) for k, v in headers.items()}
        self.content = bytes(content)

    @property
    def text(self) -> str:
        """The body decoded as UTF-8 (undecodable bytes replaced)."""
        return self.content.decode("utf-8", errors="replace")

    def json(self) -> Any:
        """The body parsed as JSON; ``ValueError`` when it is not JSON."""
        return _json.loads(self.content)

    def raise_for_status(self) -> None:
        """``RuntimeError('HTTP <status>')`` for an error status (>= 400)."""
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def __repr__(self) -> str:
        return f"HttpResponse(status_code={self.status_code}, {len(self.content)} bytes)"


class UrllibSession:
    """Stdlib HTTP client with the ``.get`` / ``.post`` subset the connectors use (no ``requests`` needed).

    HTTP error statuses are *returned* (not raised); network failures raise ``OSError`` subclasses
    (``urllib.error.URLError``, ``TimeoutError`` -- a connect timeout too --, ``ConnectionError`` ...).
    Uses urllib's default proxy handling (``HTTPS_PROXY`` etc.) and ``ssl.create_default_context()``.
    A response larger than ``max_bytes`` raises ``OSError('response too large')``.  Only ``http`` and
    ``https`` URLs are accepted (``ValueError`` otherwise).

    Stateless and thread-safe: nothing changes after ``__init__`` (no cookie jar, no connection pool);
    every call builds its own opener and request, so one instance may be shared by concurrent threads.
    """

    def __init__(self, *, max_bytes: int = 10_000_000) -> None:
        if int(max_bytes) < 1:
            raise ValueError("max_bytes must be at least 1")
        self.max_bytes = int(max_bytes)
        self._ssl_context = ssl.create_default_context()

    def get(self, url: str, headers: Optional[Mapping[str, str]] = None,
            timeout: float = TIMEOUT) -> HttpResponse:
        return self._request("GET", url, None, headers, timeout)

    def post(self, url: str, json: Any = None, headers: Optional[Mapping[str, str]] = None,
             timeout: float = TIMEOUT) -> HttpResponse:
        """POST ``json`` (serialised as UTF-8 JSON, ``Content-Type: application/json`` unless given)."""
        hdrs = dict(headers or {})
        data = b""
        if json is not None:
            data = _json.dumps(json, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            if not any(str(k).lower() == "content-type" for k in hdrs):
                hdrs["Content-Type"] = "application/json"
        return self._request("POST", url, data, hdrs, timeout)

    def _request(self, method: str, url: str, data: Optional[bytes], headers: Optional[Mapping[str, str]],
                 timeout: float) -> HttpResponse:
        scheme = urlsplit(str(url)).scheme.lower()
        if scheme not in ("http", "https"):
            raise ValueError(f"unsupported URL scheme: {scheme!r}")
        request = urllib.request.Request(str(url), data=data, method=method)
        for key, value in (headers or {}).items():
            request.add_header(str(key), str(value))
        opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=self._ssl_context))
        try:
            try:
                resp = opener.open(request, timeout=timeout)
            except urllib.error.HTTPError as exc:   # an error status is a response, not a failure
                resp = exc
            except urllib.error.URLError as exc:
                if isinstance(exc.reason, TimeoutError):
                    raise TimeoutError(f"timed out: {exc.reason}") from exc
                raise
            with resp:
                status = getattr(resp, "status", None) or resp.getcode()
                headers_out: dict[str, str] = {}
                for key, value in resp.headers.items():
                    k = key.lower()
                    headers_out[k] = f"{headers_out[k]}, {value}" if k in headers_out else value
                declared = headers_out.get("content-length", "")
                if declared.isdigit() and int(declared) > self.max_bytes:
                    raise OSError("response too large")
                content = self._read_limited(resp)
        except http.client.HTTPException as exc:   # IncompleteRead, BadStatusLine ...: a network failure
            raise OSError(f"{type(exc).__name__}: {exc}") from exc
        return HttpResponse(int(status), headers_out, content)

    def _read_limited(self, resp: Any) -> bytes:
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = resp.read(65536)
            if not chunk:
                return b"".join(chunks)
            total += len(chunk)
            if total > self.max_bytes:
                raise OSError("response too large")
            chunks.append(chunk)


def user_agent(public_url: Optional[str] = None, contact: Optional[str] = None) -> str:
    """The web app's User-Agent: ``'trendai-web/0.1.0 (+https://trendai.example; ops@example.com)'``.

    Parts that are ``None`` or empty are left out (neither -> ``'trendai-web/0.1.0'``); characters
    outside printable ASCII (including line breaks) are dropped."""
    def printable(value: Optional[str]) -> str:
        return "".join(ch for ch in str(value or "") if " " <= ch <= "~").strip()

    parts = [p for p in ("+" + printable(public_url) if printable(public_url) else "", printable(contact)) if p]
    base = f"trendai-web/{printable(__version__)}"
    return f"{base} ({'; '.join(parts)})" if parts else base


# --------------------------------------------------------------------------- periods

_PERIOD_UNITS = ("date", "week", "month")
_DAY = _dt.timedelta(days=1)


def _as_date(value: Any) -> Optional[_dt.date]:
    """A date from a date / datetime / pandas Timestamp / 'YYYY-MM-DD' string; ``None`` otherwise."""
    if isinstance(value, _dt.datetime):   # also pd.Timestamp (NaT is not a datetime)
        return value.date()
    if isinstance(value, _dt.date):
        return value
    if isinstance(value, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", value.strip()):
        try:
            return _dt.date.fromisoformat(value.strip())
        except ValueError:
            return None
    return None


def _last_completed_end(unit: str, today: _dt.date) -> _dt.date:
    """Last day of the last period that ended before ``today``."""
    if unit == "week":
        return today - _dt.timedelta(days=today.weekday() + 1)   # the last Sunday strictly before today
    if unit == "month":
        return today.replace(day=1) - _DAY
    return today - _DAY


def _snap_back_to_end(unit: str, d: _dt.date) -> _dt.date:
    """``d`` if it ends a period, else the end of the previous period."""
    if unit == "week":
        return d - _dt.timedelta(days=(d.weekday() + 1) % 7)       # back to a Sunday
    if unit == "month":
        return d if (d + _DAY).day == 1 else d.replace(day=1) - _DAY
    return d


def _snap_forward_to_start(unit: str, d: _dt.date) -> _dt.date:
    """``d`` if it starts a period, else the start of the next period."""
    if unit == "week":
        return d + _dt.timedelta(days=(7 - d.weekday()) % 7)       # forward to a Monday
    if unit == "month":
        if d.day == 1:
            return d
        return (d.replace(day=28) + _dt.timedelta(days=4)).replace(day=1)
    return d


def _years_before(d: _dt.date, years: int) -> _dt.date:
    try:
        return d.replace(year=d.year - years)
    except ValueError:   # 29 February -> 28 February
        return d.replace(year=d.year - years, day=28)


def period_bounds(range_spec: str, unit: str, today: _dt.date,
                  earliest: _dt.date = NAVER_EARLIEST) -> tuple[_dt.date, _dt.date]:
    """Whole, *completed* periods for a range chip: ``(first day, last day)``.

    ``unit``: ``'date'`` | ``'week'`` | ``'month'``.  ``range_spec``: ``'1y'`` | ``'3y'`` | ``'5y'`` (any
    ``'<N>y'``, N = 1..99) | ``'all'`` | ``'YYYY-MM-DD~YYYY-MM-DD'``.

    * end = the last completed period's last day: date -> yesterday; week -> the last Sunday strictly
      before ``today``; month -> the last day of the previous month.  A custom end is clamped to this
      and then snapped back to a period end.
    * start = ``'<N>y'``: end - N years + 1 day; ``'all'``: ``earliest``; custom: as given; then clamped
      to ``>= earliest`` and snapped *forward* to a period start (Monday / 1st of the month).

    Raises :class:`ConnectorError` (``conn_bad_range``) for an unparsable spec or unit, or when no whole
    period is left (start > end)."""
    spec = str(range_spec if range_spec is not None else "").strip()
    shown = spec if len(spec) <= 60 else spec[:59] + "…"
    if unit not in _PERIOD_UNITS:
        raise ConnectorError(msg("conn_bad_range", value=f"{shown} ({unit!r})"))
    day = _as_date(today)
    floor = _as_date(earliest)
    if day is None or floor is None:
        raise ConnectorError(msg("conn_bad_range", value=shown))
    latest_end = _last_completed_end(unit, day)
    years = re.fullmatch(r"([1-9][0-9]?)y", spec)
    custom = re.fullmatch(r"(\d{4}-\d{2}-\d{2})\s*~\s*(\d{4}-\d{2}-\d{2})", spec)
    if spec == "all" or years:
        end = latest_end
        start = floor if spec == "all" else _years_before(end, int(years.group(1))) + _DAY
    elif custom:
        start, end = _as_date(custom.group(1)), _as_date(custom.group(2))
        if start is None or end is None:
            raise ConnectorError(msg("conn_bad_range", value=shown))
        end = _snap_back_to_end(unit, min(end, latest_end))
    else:
        raise ConnectorError(msg("conn_bad_range", value=shown))
    start = _snap_forward_to_start(unit, max(start, floor))
    if start > end:
        raise ConnectorError(msg("conn_bad_range", value=shown))
    return start, end


# --------------------------------------------------------------------------- Naver search trends


@dataclass(frozen=True, repr=False)
class NaverCredentials:
    """API keys of the Naver search-trend endpoint: ``endpoint`` ``'hub'`` (NAVER API HUB key id + key)
    or ``'legacy'`` (Naver Developers client id + secret).  The repr never shows the keys."""

    endpoint: str
    key_id: str
    key: str

    def __post_init__(self) -> None:
        if self.endpoint not in ("hub", "legacy"):
            raise ValueError("endpoint must be 'hub' or 'legacy'")

    def __repr__(self) -> str:
        return f"NaverCredentials(endpoint={self.endpoint!r}, key_id='***', key='***')"

    @classmethod
    def from_env(cls, environ: Mapping[str, str] = os.environ) -> Optional["NaverCredentials"]:
        """HUB when ``NAVER_API_HUB_KEY_ID`` and ``NAVER_API_HUB_KEY`` are both non-empty; else legacy when
        ``NAVER_CLIENT_ID`` and ``NAVER_CLIENT_SECRET`` are; else ``None``.  No runtime fallback between
        the endpoints (a HUB key that fails is reported, not retried on the legacy endpoint)."""
        def get(name: str) -> str:
            return str(environ.get(name) or "").strip()

        hub_id, hub_key = get("NAVER_API_HUB_KEY_ID"), get("NAVER_API_HUB_KEY")
        if hub_id and hub_key:
            return cls("hub", hub_id, hub_key)
        client_id, secret = get("NAVER_CLIENT_ID"), get("NAVER_CLIENT_SECRET")
        if client_id and secret:
            return cls("legacy", client_id, secret)
        return None


@dataclass(frozen=True)
class NaverGroup:
    """One keyword group of a request: ``name`` (the API's ``groupName``; the web app sends the wire
    names ``'g1'``..``'g5'``) and 1..:data:`NAVER_MAX_KEYWORDS` keywords whose searches are summed."""

    name: str
    keywords: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.keywords, tuple):
            kws = (self.keywords,) if isinstance(self.keywords, str) else tuple(self.keywords)
            object.__setattr__(self, "keywords", kws)


@dataclass
class NaverResult:
    """The parsed answer of one request (see :func:`fetch_naver_search_trend`)."""

    series: dict[str, pd.Series]   # group name -> float ratio series (request order), complete bucket grid
    endpoint: str                  # 'hub' | 'legacy'
    time_unit: str                 # 'date' | 'week' | 'month'
    start: _dt.date
    end: _dt.date
    filled: dict[str, int]         # group name -> missing buckets filled with 0
    raw_max: float                 # largest ratio in the response (normally 100.0)


def _short_repr(value: Any) -> str:
    text = repr(value)
    return text if len(text) <= 60 else text[:59] + "…"


def _bad_filter(what: str, value: Any) -> ConnectorError:
    return ConnectorError(msg("conn_naver_bad_filter", what=what, value=_short_repr(value)))


def build_naver_body(groups: Sequence[NaverGroup], start: _dt.date, end: _dt.date, time_unit: str,
                     device: Optional[str] = None, gender: Optional[str] = None,
                     ages: Sequence[str] = ()) -> dict:
    """The JSON body of a search-trend request.

    Validates (:class:`ConnectorError`): 1..5 groups (more: ``conn_naver_too_many_groups``; none:
    ``conn_naver_bad_filter`` with ``what='keywordGroups'``); group names 1..100 characters and unique
    (``conn_naver_bad_filter``, ``what='groupName'``); 1..5 non-empty keywords per group (more:
    ``conn_naver_too_many_keywords``; none or an empty one: ``conn_naver_bad_filter``,
    ``what='keywords'``); ``time_unit`` in date/week/month, ``device`` in {None, 'pc', 'mo'}, ``gender`` in
    {None, 'm', 'f'}, ``ages`` a subset of :data:`NAVER_AGE_CODES` (``conn_naver_bad_filter``);
    :data:`NAVER_EARLIEST` <= start <= end (``conn_bad_range``).  Optional keys are left out when empty;
    keywords are stripped; ages are de-duplicated and sorted numerically."""
    groups = list(groups)
    if not groups:
        raise _bad_filter("keywordGroups", [])
    if len(groups) > NAVER_MAX_GROUPS:
        raise ConnectorError(msg("conn_naver_too_many_groups", max=NAVER_MAX_GROUPS))
    seen: set[str] = set()
    keyword_groups = []
    for g in groups:
        name = g.name
        if (not isinstance(name, str) or not name.strip() or len(name) > _NAVER_GROUP_NAME_MAX
                or name in seen):
            raise _bad_filter("groupName", name)
        seen.add(name)
        if len(g.keywords) > NAVER_MAX_KEYWORDS:
            raise ConnectorError(msg("conn_naver_too_many_keywords", max=NAVER_MAX_KEYWORDS, group=name))
        if not g.keywords or not all(isinstance(k, str) and k.strip() for k in g.keywords):
            raise _bad_filter("keywords", list(g.keywords))
        keyword_groups.append({"groupName": name, "keywords": [k.strip() for k in g.keywords]})
    if time_unit not in _NAVER_TIME_UNITS:
        raise _bad_filter("timeUnit", time_unit)
    if device not in (None, "", *_NAVER_DEVICES):
        raise _bad_filter("device", device)
    if gender not in (None, "", *_NAVER_GENDERS):
        raise _bad_filter("gender", gender)
    if isinstance(ages, str):
        raise _bad_filter("ages", ages)
    age_codes = {str(a).strip() for a in ages}
    if not age_codes <= set(NAVER_AGE_CODES):
        raise _bad_filter("ages", sorted(age_codes - set(NAVER_AGE_CODES)))
    first, last = _as_date(start), _as_date(end)
    if first is None or last is None or first < NAVER_EARLIEST or first > last:
        raise ConnectorError(msg("conn_bad_range", value=f"{start}~{end}"))
    body: dict[str, Any] = {"startDate": first.isoformat(), "endDate": last.isoformat(), "timeUnit": time_unit,
                            "keywordGroups": keyword_groups}
    if device:
        body["device"] = device
    if gender:
        body["gender"] = gender
    if age_codes:
        body["ages"] = sorted(age_codes, key=int)
    return body


def naver_cache_key(groups: Sequence[NaverGroup], start: _dt.date, end: _dt.date, time_unit: str,
                    device: Optional[str] = None, gender: Optional[str] = None, ages: Sequence[str] = ()) -> str:
    """sha256 hex of the canonical JSON of :func:`build_naver_body` (``sort_keys``, compact separators,
    ``ensure_ascii=False``): no credentials, no endpoint.  Group order matters; age order does not."""
    body = build_naver_body(groups, start, end, time_unit, device, gender, ages)
    canonical = _json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _scrub(text: str, credentials: NaverCredentials) -> str:
    """``text`` with the credentials' key id and key masked (defence in depth for error texts)."""
    for secret in (credentials.key_id, credentials.key):
        if secret and len(secret) >= 4:
            text = text.replace(secret, "***")
    return text


def _naver_error_fields(payload: Any) -> tuple[Optional[str], Optional[str]]:
    """(errorCode, message) of a HUB (``{"error": {"errorCode", "message"}}``) or legacy
    (``{"errorCode", "errorMessage"}``) error body."""
    if not isinstance(payload, dict):
        return None, None
    err = payload.get("error") if isinstance(payload.get("error"), dict) else payload
    code = err.get("errorCode")
    message = err.get("message", err.get("errorMessage"))
    return (None if code is None else str(code)), (None if message is None else str(message))


#: error codes in the body that mean "authentication failed / the Search Trend API is not enabled"
#: (HUB 300 and 401, legacy 024)
_NAVER_AUTH_CODES = frozenset({"300", "401", "024"})


def _naver_failure(status: int, payload: Any, credentials: NaverCredentials) -> Optional[ConnectorError]:
    """The ConnectorError for an error answer (status >= 300 or an error object in the body), else None."""
    code, message = _naver_error_fields(payload)
    if status < 300 and code is None:
        return None
    detail = f"HTTP {status}"
    if code is not None:
        detail += f" errorCode={code}"
    if message:
        detail += f" message={message}"
    detail = _scrub(detail, credentials)
    shown = status if status >= 400 or code is None or not code.isdigit() else int(code)
    if status in (401, 403) or code in _NAVER_AUTH_CODES:
        return _connector_error("conn_naver_auth", detail, status=shown)
    if status == 429 or code == "429":
        return _connector_error("conn_naver_rate", detail)
    if status >= 500:
        return _connector_error("conn_naver_server", detail, status=status)
    if status >= 300:
        return _connector_error("conn_naver_http", detail, status=status)
    return _connector_error("conn_naver_bad_response", detail)


def _parse_ratio(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ValueError("ratio")
    ratio = float(value.strip()) if isinstance(value, str) else float(value)
    if not math.isfinite(ratio):
        raise ValueError("ratio")
    return ratio


def _parse_keywords(value: Any) -> list[str]:
    """``keywords`` of a result: a list (documented) or a comma-separated string (also documented)."""
    if value is None:
        return []
    if isinstance(value, str):
        return [k.strip() for k in value.split(",") if k.strip()]
    if isinstance(value, (list, tuple)):
        return [str(k).strip() for k in value if str(k).strip()]
    raise ValueError("keywords")


def _month_start(d: _dt.date) -> _dt.date:
    return d.replace(day=1)


def _month_end(d: _dt.date) -> _dt.date:
    return (d.replace(day=28) + _dt.timedelta(days=4)).replace(day=1) - _DAY


def _bucket_grid(time_unit: str, start: _dt.date, end: _dt.date, periods: list[_dt.date]) -> list[_dt.date]:
    """Every bucket start from ``start`` (or an earlier returned bucket) to the last complete bucket.

    Weekly buckets follow the weekday of the returned periods (Monday when ``start`` is one); a returned
    period off that 7-day grid is a bad response.  Monthly buckets are month starts."""
    first = min([start, *periods])
    if time_unit == "date":
        return [first + _dt.timedelta(days=i) for i in range((end - first).days + 1)]
    if time_unit == "month":
        grid, m = [], _month_start(first)
        while _month_end(m) <= end:
            grid.append(m)
            m = _snap_forward_to_start("month", m + _DAY)
        return grid
    anchor = periods[0] if periods else start
    if any((p - anchor).days % 7 for p in periods):
        raise ValueError("weekly periods are not 7 days apart")
    if anchor >= start:
        first = anchor - _dt.timedelta(days=7 * ((anchor - start).days // 7))
    else:
        first = anchor
    grid, d = [], first
    while d + _dt.timedelta(days=6) <= end:
        grid.append(d)
        d += _dt.timedelta(days=7)
    return grid


def fetch_naver_search_trend(groups: Sequence[NaverGroup], start: _dt.date, end: _dt.date,
                             time_unit: str = "week", device: Optional[str] = None,
                             gender: Optional[str] = None, ages: Sequence[str] = (), *,
                             credentials: NaverCredentials, session: Any = None,
                             timeout: float = TIMEOUT) -> NaverResult:
    """Naver DataLab search trends of up to 5 keyword groups in **one** request.

    One POST to the HUB (``X-NCP-APIGW-API-KEY-ID`` / ``X-NCP-APIGW-API-KEY``) or legacy endpoint
    (``X-Naver-Client-Id`` / ``X-Naver-Client-Secret``) with the body of :func:`build_naver_body`.
    ``session``: an object with ``.post(url, json=, headers=, timeout=)`` (default :class:`UrllibSession`).

    ``results[i]`` belongs to ``groups[i]`` (its ``title`` must equal the group name); ``ratio`` is read
    as a float (number or numeric string) and ``keywords`` as a list or a comma-separated string.  A
    trailing weekly bucket whose ``period + 6 days > end`` (an incomplete week) is dropped; each series is
    re-gridded and buckets Naver left out are filled with 0 (counted in ``filled``).  Series carry
    ``attrs['trendai_source'] = {'kind': 'naver', 'group', 'keywords', 'time_unit', 'device', 'gender',
    'ages', 'endpoint'}``.

    Errors are :class:`ConnectorError` (``conn_naver_*``, ``conn_bad_range``); ``exc.log_detail`` holds the
    upstream status / errorCode / message for operator logs (never headers or keys)."""
    groups = list(groups)
    body = build_naver_body(groups, start, end, time_unit, device, gender, ages)
    if credentials is None:
        raise ConnectorError(msg("conn_naver_unconfigured"))
    if credentials.endpoint == "hub":
        url = NAVER_HUB_URL
        auth = {"X-NCP-APIGW-API-KEY-ID": credentials.key_id, "X-NCP-APIGW-API-KEY": credentials.key}
    else:
        url = NAVER_LEGACY_URL
        auth = {"X-Naver-Client-Id": credentials.key_id, "X-Naver-Client-Secret": credentials.key}
    headers = {**auth, "Content-Type": "application/json", "Accept": "application/json",
               "User-Agent": user_agent()}
    if session is None:
        session = UrllibSession()
    try:
        resp = session.post(url, json=body, headers=headers, timeout=timeout)
    except Exception as exc:  # urllib / requests / socket errors
        detail = _scrub(_short_error(exc), credentials)
        if _is_timeout(exc):
            raise _connector_error("conn_naver_timeout", detail) from None
        raise _connector_error("conn_naver_network", detail, error=detail) from None

    status = getattr(resp, "status_code", None)
    if not isinstance(status, int):
        raise _connector_error("conn_naver_bad_response", "no HTTP status")
    try:
        payload = resp.json()
    except Exception:  # noqa: BLE001 - any parse failure of the body
        payload = None
    failure = _naver_failure(status, payload, credentials)
    if failure is not None:
        if failure.code == "conn_naver_rate":
            failure.retry_after = _retry_after(resp)
        raise failure
    if not isinstance(payload, dict):
        raise _connector_error("conn_naver_bad_response", f"HTTP {status}: body is not a JSON object")
    try:
        return _parse_naver_payload(payload, groups, body, credentials.endpoint)
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        detail = _scrub(f"HTTP {status}: {type(exc).__name__}: {exc}", credentials)
        raise _connector_error("conn_naver_bad_response", detail) from None


def _parse_naver_payload(payload: dict, groups: list[NaverGroup], body: dict, endpoint: str) -> NaverResult:
    start = _dt.date.fromisoformat(body["startDate"])
    end = _dt.date.fromisoformat(body["endDate"])
    time_unit = body["timeUnit"]
    results = payload.get("results")
    if not isinstance(results, list) or len(results) != len(groups):
        raise ValueError(f"expected {len(groups)} results")
    parsed: list[dict[_dt.date, float]] = []
    raw_max = 0.0
    for g, item in zip(groups, results):
        if not isinstance(item, dict) or item.get("title") != g.name:
            raise ValueError("result title does not match the group name")
        _parse_keywords(item.get("keywords"))   # shape check only: the requested keywords are kept in attrs
        data = item.get("data") or []
        if not isinstance(data, list):
            raise ValueError("data is not a list")
        values: dict[_dt.date, float] = {}
        for point in data:
            if not isinstance(point, dict) or not isinstance(point.get("period"), str):
                raise ValueError("bad data point")
            period = _dt.date.fromisoformat(point["period"].strip()[:10])
            ratio = _parse_ratio(point.get("ratio"))
            raw_max = max(raw_max, ratio)
            if time_unit == "month":
                period = _month_start(period)
                if _month_end(period) > end:
                    continue
            elif time_unit == "week" and period + _dt.timedelta(days=6) > end:
                continue    # incomplete trailing week
            elif period > end:
                continue
            values[period] = ratio
        parsed.append(values)
    periods = sorted({p for values in parsed for p in values})
    grid = _bucket_grid(time_unit, start, end, periods)
    if not grid:
        raise ConnectorError(msg("conn_bad_range", value=f"{start}~{end}"))
    index = pd.DatetimeIndex(pd.to_datetime(grid), name="date")
    ages = list(body.get("ages", []))
    series: dict[str, pd.Series] = {}
    filled: dict[str, int] = {}
    for g, values in zip(groups, parsed):
        s = pd.Series([values.get(d, 0.0) for d in grid], index=index, name=g.name, dtype=float)
        s.attrs["trendai_source"] = {"kind": "naver", "group": g.name, "keywords": list(g.keywords),
                                     "time_unit": time_unit, "device": body.get("device"),
                                     "gender": body.get("gender"), "ages": ages, "endpoint": endpoint}
        series[g.name] = s
        filled[g.name] = sum(1 for d in grid if d not in values)
    return NaverResult(series=series, endpoint=endpoint, time_unit=time_unit, start=start, end=end,
                       filled=filled, raw_max=float(raw_max))


# --------------------------------------------------------------------------- Wikipedia


def _today() -> pd.Timestamp:
    """Today's date (UTC, midnight).  Separate function so tests can pin it."""
    return pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()


def _to_timestamp(value: Any, what: str) -> pd.Timestamp:
    """Parse ``'YYYY-MM-DD'``, ``'YYYYMMDD'`` or a date-like object."""
    if isinstance(value, (pd.Timestamp, _dt.datetime, _dt.date, np.datetime64)):
        ts = pd.Timestamp(value)
        if pd.isna(ts):  # NaT is a datetime subclass but has no usable date
            raise ConnectorError(msg("conn_date_nat", what=what))
        if ts.tzinfo is not None:
            ts = ts.tz_convert("UTC").tz_localize(None)
        return ts.normalize()
    if isinstance(value, str):
        v = value.strip()
        if re.fullmatch(r"\d{8}", v):
            fmt = "%Y%m%d"
        elif re.fullmatch(r"\d{4}-\d{2}-\d{2}", v):
            fmt = "%Y-%m-%d"
        else:
            raise ConnectorError(msg("conn_date_format", what=what, value=repr(value)))
        try:
            return pd.to_datetime(v, format=fmt)
        except (ValueError, TypeError) as exc:
            raise ConnectorError(msg("conn_date_invalid", what=what, value=repr(value))) from exc
    raise ConnectorError(msg("conn_date_format", what=what, value=repr(value)))


def _default_end(granularity: str) -> pd.Timestamp:
    """Last complete month (monthly), Monday-Sunday week (weekly: its Sunday) or day (daily) before today."""
    today = _today()
    if granularity == "monthly":
        return today.replace(day=1) - pd.Timedelta(days=1)
    if granularity == "weekly":
        return today - pd.Timedelta(days=int(today.dayofweek) + 1)
    return today - pd.Timedelta(days=1)


def _weekly_bounds(start: Any, end: Any) -> tuple[pd.Timestamp, pd.Timestamp]:
    """``start`` snapped forward to a Monday and ``end`` back to a Sunday (whole weeks only)."""
    s = _to_timestamp(start, "start") if start is not None else pd.Timestamp(DEFAULT_START)
    e = _to_timestamp(end, "end") if end is not None else _default_end("weekly")
    s = s + pd.Timedelta(days=(7 - int(s.dayofweek)) % 7)
    e = e - pd.Timedelta(days=(int(e.dayofweek) + 1) % 7)
    if s > e:
        raise ConnectorError(msg("conn_start_after_end", start=f"{s:%Y-%m-%d}", end=f"{e:%Y-%m-%d}"))
    return s, e


def _weekly_sums(daily: pd.Series, start: pd.Timestamp, end: pd.Timestamp, title: str) -> pd.Series:
    """Monday-indexed sums of the Monday-Sunday weeks of ``daily`` that lie inside [start, end] *and*
    inside the span of the returned days (a week cut off by the first / last returned day is dropped).
    Days missing inside that span count as 0 views: the API leaves out days without views."""
    d = daily[(daily.index >= start) & (daily.index <= end)]
    if d.empty:
        raise ConnectorError(msg("conn_wiki_empty", title=title))
    first, last = d.index.min(), d.index.max()
    week_start = first + pd.Timedelta(days=(7 - int(first.dayofweek)) % 7)
    week_end = last - pd.Timedelta(days=(int(last.dayofweek) + 1) % 7)
    if week_start > week_end:
        raise ConnectorError(msg("conn_wiki_empty", title=title))
    days = pd.date_range(week_start, week_end, freq="D")
    values = d.reindex(days, fill_value=0.0).to_numpy(dtype=float).reshape(-1, 7).sum(axis=1)
    return pd.Series(values, index=pd.DatetimeIndex(days[::7], name="date"), name=daily.name, dtype=float)


def _check_args(article: str, lang: str, granularity: str) -> tuple[str, str, str]:
    if not isinstance(article, str) or not article.strip():
        raise ConnectorError(msg("conn_no_article"))
    lang_ = str(lang or "").strip().lower()
    if not re.fullmatch(r"[a-z][a-z0-9-]{1,20}", lang_):
        raise ConnectorError(msg("conn_bad_lang", lang=repr(lang)))
    gran = str(granularity or "").strip().lower()
    if gran not in _GRANULARITIES:
        raise ConnectorError(msg("conn_bad_granularity", choices=str(_GRANULARITIES), value=repr(granularity)))
    return article.strip(), lang_, gran


def wikipedia_pageviews_url(article: str, lang: str = "ko", start: Optional[str] = None,
                            end: Optional[str] = None, granularity: str = "monthly") -> str:
    """The Wikimedia REST URL :func:`fetch_wikipedia_pageviews` would request.

    Spaces in the title become underscores and the title is fully
    percent-encoded (UTF-8, ``/`` included).  ``start`` defaults to
    2016-01-01 and ``end`` to the last complete month/day; for monthly data
    ``start`` is snapped to the first day of its month.
    """
    title, lang_, gran = _check_args(article, lang, granularity)
    s = _to_timestamp(start, "start") if start is not None else pd.Timestamp(DEFAULT_START)
    e = _to_timestamp(end, "end") if end is not None else _default_end(gran)
    if gran == "monthly":
        s = s.replace(day=1)
    if s > e:
        raise ConnectorError(msg("conn_start_after_end", start=f"{s:%Y-%m-%d}", end=f"{e:%Y-%m-%d}"))
    encoded = quote(title.replace(" ", "_"), safe="")
    return (f"{WIKIMEDIA_PAGEVIEWS}/{lang_}.wikipedia/all-access/user/{encoded}/{gran}/"
            f"{s:%Y%m%d}00/{e:%Y%m%d}00")


def _short_error(exc: BaseException) -> str:
    """``exc`` as a short detail for a message: its class and the first 120 characters of its first line
    (a urllib3 dump with the whole URL is no help to the user)."""
    text = str(exc).strip()
    text = text.splitlines()[0] if text else ""
    if len(text) > 120:
        text = text[:119] + "…"
    return f"{type(exc).__name__}: {text}" if text else type(exc).__name__


def fetch_wikipedia_pageviews(article: str, lang: str = "ko", start: Optional[str] = None,
                              end: Optional[str] = None, granularity: str = "monthly",
                              session: Any = None, *, user_agent: Optional[str] = None) -> pd.Series:
    """Page views of a Wikipedia article (human users, all access methods).

    Parameters
    ----------
    article:
        Exact article title, e.g. ``"인공지능"`` or ``"Machine learning"``.
    lang:
        Wikipedia language edition (``"ko"``, ``"en"``, ...).
    start, end:
        ``'YYYY-MM-DD'``, ``'YYYYMMDD'`` or a date-like object; default 2016-01-01 to the
        last complete month (monthly) / week (weekly) / day (daily).
    granularity:
        ``"monthly"``, ``"daily"`` or ``"weekly"``: daily data from ``start`` (snapped
        forward to a Monday) to ``end`` (snapped back to a Sunday), returned as
        Monday-indexed sums of complete Monday-Sunday weeks.
    session:
        Object with ``.get(url, headers=..., timeout=...)`` (e.g. a
        ``requests.Session`` or a :class:`UrllibSession`); defaults to the
        ``requests`` module when it is installed, else a :class:`UrllibSession`.
    user_agent:
        The User-Agent header (default :data:`USER_AGENT`); the web app sends
        :func:`trendai.connectors.user_agent` with its public URL and contact.

    Returns a float series named after the article, indexed by date (sorted).
    Raises :class:`ConnectorError` on HTTP / network errors (``conn_timeout`` for a
    timeout) or empty data.
    """
    weekly = str(granularity or "").strip().lower() == "weekly"
    if weekly:
        title, lang_, _ = _check_args(article, lang, "daily")
        week_start, week_end = _weekly_bounds(start, end)
        url = wikipedia_pageviews_url(article, lang=lang, start=week_start, end=week_end, granularity="daily")
    else:
        url = wikipedia_pageviews_url(article, lang=lang, start=start, end=end, granularity=granularity)
        title, lang_, _ = _check_args(article, lang, granularity)
    if session is None:
        try:
            import requests
        except ImportError:
            session = UrllibSession()
        else:
            session = requests
    try:
        resp = session.get(url, headers={"User-Agent": user_agent or USER_AGENT}, timeout=TIMEOUT)
    except Exception as exc:  # requests.RequestException, socket errors, timeouts, ...
        if _is_timeout(exc):
            raise ConnectorError(msg("conn_timeout")) from exc
        raise ConnectorError(msg("conn_network", error=_short_error(exc))) from exc

    status = getattr(resp, "status_code", None)
    if status == 404:
        raise ConnectorError(msg("conn_wiki_404", title=title, lang=lang_))
    if status == 429:
        exc = ConnectorError(msg("conn_wiki_429"))
        exc.retry_after = _retry_after(resp)
        raise exc
    if isinstance(status, int) and status >= 400:
        raise ConnectorError(msg("conn_wiki_http", status=status, url=url))
    raise_for_status = getattr(resp, "raise_for_status", None)
    if callable(raise_for_status):
        try:
            raise_for_status()
        except Exception as exc:
            raise ConnectorError(msg("conn_wiki_error", error=str(exc))) from exc
    try:
        data = resp.json()
    except Exception as exc:
        raise ConnectorError(msg("conn_wiki_json")) from exc

    items = data.get("items") if isinstance(data, dict) else None
    if not items:
        raise ConnectorError(msg("conn_wiki_empty", title=title))
    dates: list[pd.Timestamp] = []
    views: list[float] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        ts = str(item.get("timestamp", ""))
        if len(ts) < 8 or not ts[:8].isdigit():
            continue
        try:
            date = pd.to_datetime(ts[:8], format="%Y%m%d")
            value = float(item.get("views"))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            dates.append(date)
            views.append(value)
    if not dates:
        raise ConnectorError(msg("conn_wiki_no_valid"))
    s = pd.Series(views, index=pd.DatetimeIndex(dates, name="date"), name=title.replace("_", " "),
                  dtype=float)
    s = s[~s.index.duplicated(keep="last")].sort_index()
    if weekly:
        return _weekly_sums(s, week_start, week_end, title)
    return s


def fetch_google_trends(keyword: str, timeframe: str = "today 5-y", geo: str = "") -> pd.Series:
    """Google Trends interest over time (0-100) for ``keyword`` via ``pytrends``.

    ``timeframe`` uses pytrends syntax (``"today 5-y"``, ``"2020-01-01 2024-12-31"``, ...);
    ``geo`` is a country code such as ``"KR"`` (``""`` = worldwide).
    Raises :class:`ConnectorError` if pytrends is missing, the request fails
    (often rate limiting) or no data comes back.
    """
    if not isinstance(keyword, str) or not keyword.strip():
        raise ConnectorError(msg("conn_no_keyword"))
    kw = keyword.strip()
    try:
        from pytrends.request import TrendReq
    except ImportError as exc:
        raise ConnectorError(msg("conn_need_pytrends")) from exc
    except Exception as exc:  # installed but broken (e.g. incompatible with the installed pandas)
        raise ConnectorError(msg("conn_pytrends_broken", kind=type(exc).__name__, error=str(exc))) from exc
    try:
        trends = TrendReq(hl="ko-KR", tz=540)
        trends.build_payload([kw], timeframe=timeframe, geo=geo)
        df = trends.interest_over_time()
    except Exception as exc:  # pytrends raises a variety of request/response errors
        raise ConnectorError(msg("conn_trends_failed", error=str(exc))) from exc
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        raise ConnectorError(msg("conn_trends_empty", kw=kw))
    df = df.drop(columns=["isPartial"], errors="ignore")
    if df.shape[1] == 0:
        raise ConnectorError(msg("conn_trends_no_col", kw=kw))
    col = kw if kw in df.columns else df.columns[0]
    s = pd.to_numeric(df[col], errors="coerce").astype(float)
    try:
        s.index = pd.DatetimeIndex(pd.to_datetime(s.index))
    except (TypeError, ValueError) as exc:
        raise ConnectorError(msg("conn_trends_dates")) from exc
    s.index.name = "date"
    s = s[s.notna()].sort_index()
    if s.empty:
        raise ConnectorError(msg("conn_trends_no_valid", kw=kw))
    s.name = kw
    return s
