"""Online data connectors (optional).

* :func:`fetch_wikipedia_pageviews` -- Wikimedia REST pageviews API
  (no key needed; uses ``requests`` or any object with a compatible ``.get``).
* :func:`fetch_google_trends` -- Google Trends interest over time via the
  optional ``pytrends`` package.

Both return a raw float ``pd.Series`` with a sorted ``DatetimeIndex``; pass it
through :func:`trendai.data.prepare` (or straight into ``analyze``/``run``).
Every failure (missing package, HTTP error, network error, empty result) is
raised as :class:`ConnectorError` with a message that says what to do.
Requests carry a generic User-Agent and no personal data.
"""

from __future__ import annotations

import datetime as _dt
import math
import re
from typing import Any, Optional
from urllib.parse import quote

import numpy as np
import pandas as pd

__all__ = [
    "ConnectorError", "USER_AGENT", "wikipedia_pageviews_url", "fetch_wikipedia_pageviews",
    "fetch_google_trends",
]

USER_AGENT = "trendai/0.1 (time-series trend analysis tool)"
WIKIMEDIA_PAGEVIEWS = "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article"
DEFAULT_START = "20160101"
TIMEOUT = 20
_GRANULARITIES = ("daily", "monthly")


class ConnectorError(RuntimeError):
    """Fetching data from an online source failed."""


def _today() -> pd.Timestamp:
    """Today's date (UTC, midnight).  Separate function so tests can pin it."""
    return pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()


def _to_timestamp(value: Any, what: str) -> pd.Timestamp:
    """Parse ``'YYYY-MM-DD'``, ``'YYYYMMDD'`` or a date-like object."""
    if isinstance(value, (pd.Timestamp, _dt.datetime, _dt.date, np.datetime64)):
        ts = pd.Timestamp(value)
        if pd.isna(ts):  # NaT is a datetime subclass but has no usable date
            raise ConnectorError(f"{what} 날짜가 비어 있습니다 (NaT).")
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
            raise ConnectorError(f"{what} 날짜 형식이 올바르지 않습니다: {value!r} ('YYYY-MM-DD' 또는 'YYYYMMDD')")
        try:
            return pd.to_datetime(v, format=fmt)
        except (ValueError, TypeError) as exc:
            raise ConnectorError(f"{what} 날짜가 유효하지 않습니다: {value!r}") from exc
    raise ConnectorError(f"{what} 날짜 형식이 올바르지 않습니다: {value!r} ('YYYY-MM-DD' 또는 'YYYYMMDD')")


def _default_end(granularity: str) -> pd.Timestamp:
    """Last complete month (monthly) or day (daily) before today."""
    today = _today()
    if granularity == "monthly":
        return today.replace(day=1) - pd.Timedelta(days=1)
    return today - pd.Timedelta(days=1)


def _check_args(article: str, lang: str, granularity: str) -> tuple[str, str, str]:
    if not isinstance(article, str) or not article.strip():
        raise ConnectorError("위키백과 문서 제목(article)이 비어 있습니다.")
    lang_ = str(lang or "").strip().lower()
    if not re.fullmatch(r"[a-z][a-z0-9-]{1,20}", lang_):
        raise ConnectorError(f"언어 코드가 올바르지 않습니다: {lang!r} (예: 'ko', 'en')")
    gran = str(granularity or "").strip().lower()
    if gran not in _GRANULARITIES:
        raise ConnectorError(f"granularity는 {_GRANULARITIES} 중 하나여야 합니다: {granularity!r}")
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
        raise ConnectorError(f"시작일({s:%Y-%m-%d})이 종료일({e:%Y-%m-%d})보다 늦습니다.")
    encoded = quote(title.replace(" ", "_"), safe="")
    return (f"{WIKIMEDIA_PAGEVIEWS}/{lang_}.wikipedia/all-access/user/{encoded}/{gran}/"
            f"{s:%Y%m%d}00/{e:%Y%m%d}00")


def fetch_wikipedia_pageviews(article: str, lang: str = "ko", start: Optional[str] = None,
                              end: Optional[str] = None, granularity: str = "monthly",
                              session: Any = None) -> pd.Series:
    """Page views of a Wikipedia article (human users, all access methods).

    Parameters
    ----------
    article:
        Exact article title, e.g. ``"인공지능"`` or ``"Machine learning"``.
    lang:
        Wikipedia language edition (``"ko"``, ``"en"``, ...).
    start, end:
        ``'YYYY-MM-DD'`` or ``'YYYYMMDD'``; default 2016-01-01 to the last
        complete month (monthly) / day (daily).
    granularity:
        ``"monthly"`` or ``"daily"``.
    session:
        Object with ``.get(url, headers=..., timeout=...)`` (e.g. a
        ``requests.Session``); defaults to the ``requests`` module.

    Returns a float series named after the article, indexed by date (sorted).
    Raises :class:`ConnectorError` on HTTP / network errors or empty data.
    """
    url = wikipedia_pageviews_url(article, lang=lang, start=start, end=end, granularity=granularity)
    title, lang_, _ = _check_args(article, lang, granularity)
    if session is None:
        try:
            import requests
        except ImportError as exc:
            raise ConnectorError("위키백과 조회에는 requests 패키지가 필요합니다: pip install requests") from exc
        session = requests
    try:
        resp = session.get(url, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT)
    except Exception as exc:  # requests.RequestException, socket errors, timeouts, ...
        raise ConnectorError(f"Wikimedia API에 연결하지 못했습니다 (network error): {exc}") from exc

    status = getattr(resp, "status_code", None)
    if status == 404:
        raise ConnectorError(
            f"위키백과 문서를 찾을 수 없습니다 (article not found): '{title}' ({lang_}.wikipedia). "
            "정확한 문서 제목(대소문자·띄어쓰기)을 확인하거나, 조회 기간에 데이터가 있는지 확인하세요."
        )
    if status == 429:
        raise ConnectorError("Wikimedia API 요청 한도를 초과했습니다 (HTTP 429). 잠시 후 다시 시도하세요.")
    if isinstance(status, int) and status >= 400:
        raise ConnectorError(f"Wikimedia API 오류 (HTTP {status}): {url}")
    raise_for_status = getattr(resp, "raise_for_status", None)
    if callable(raise_for_status):
        try:
            raise_for_status()
        except Exception as exc:
            raise ConnectorError(f"Wikimedia API 오류: {exc}") from exc
    try:
        data = resp.json()
    except Exception as exc:
        raise ConnectorError("Wikimedia API 응답을 JSON으로 해석할 수 없습니다.") from exc

    items = data.get("items") if isinstance(data, dict) else None
    if not items:
        raise ConnectorError(f"조회 기간에 '{title}' 페이지뷰 데이터가 없습니다 (empty items).")
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
        raise ConnectorError("Wikimedia API 응답에 유효한 페이지뷰 항목이 없습니다.")
    s = pd.Series(views, index=pd.DatetimeIndex(dates, name="date"), name=title.replace("_", " "),
                  dtype=float)
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s


def fetch_google_trends(keyword: str, timeframe: str = "today 5-y", geo: str = "") -> pd.Series:
    """Google Trends interest over time (0-100) for ``keyword`` via ``pytrends``.

    ``timeframe`` uses pytrends syntax (``"today 5-y"``, ``"2020-01-01 2024-12-31"``, ...);
    ``geo`` is a country code such as ``"KR"`` (``""`` = worldwide).
    Raises :class:`ConnectorError` if pytrends is missing, the request fails
    (often rate limiting) or no data comes back.
    """
    if not isinstance(keyword, str) or not keyword.strip():
        raise ConnectorError("검색어(keyword)가 비어 있습니다.")
    kw = keyword.strip()
    try:
        from pytrends.request import TrendReq
    except ImportError as exc:
        raise ConnectorError("Google Trends 조회에는 pytrends 패키지가 필요합니다: pip install pytrends") from exc
    except Exception as exc:  # installed but broken (e.g. incompatible with the installed pandas)
        raise ConnectorError(
            f"pytrends 패키지를 불러오지 못했습니다 ({type(exc).__name__}: {exc}). "
            "pip install -U pytrends 로 업데이트하세요."
        ) from exc
    try:
        trends = TrendReq(hl="ko-KR", tz=540)
        trends.build_payload([kw], timeframe=timeframe, geo=geo)
        df = trends.interest_over_time()
    except Exception as exc:  # pytrends raises a variety of request/response errors
        raise ConnectorError(
            f"Google Trends 요청에 실패했습니다: {exc}. 요청이 너무 잦으면 차단(429 rate limit)될 수 있으니 "
            "잠시 후 다시 시도하세요."
        ) from exc
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        raise ConnectorError(
            f"Google Trends에서 '{kw}'에 대한 데이터가 없습니다. 검색량이 너무 적거나 기간·지역 설정을 확인하세요."
        )
    df = df.drop(columns=["isPartial"], errors="ignore")
    if df.shape[1] == 0:
        raise ConnectorError(f"Google Trends 응답에 '{kw}' 열이 없습니다.")
    col = kw if kw in df.columns else df.columns[0]
    s = pd.to_numeric(df[col], errors="coerce").astype(float)
    try:
        s.index = pd.DatetimeIndex(pd.to_datetime(s.index))
    except (TypeError, ValueError) as exc:
        raise ConnectorError("Google Trends 응답의 날짜를 해석할 수 없습니다.") from exc
    s.index.name = "date"
    s = s[s.notna()].sort_index()
    if s.empty:
        raise ConnectorError(f"Google Trends에서 '{kw}'에 대한 유효한 값이 없습니다.")
    s.name = kw
    return s
