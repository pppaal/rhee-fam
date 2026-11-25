"""Tests for trendai.connectors (all network access is mocked)."""

from __future__ import annotations

import datetime as dt
import http.server
import json
import sys
import re
import threading
import time
import types

import numpy as np
import pandas as pd
import pytest

from trendai import connectors
from trendai.connectors import (
    USER_AGENT, ConnectorError, fetch_google_trends, fetch_wikipedia_pageviews,
    wikipedia_pageviews_url,
)
from trendai.data import prepare

BASE = "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article"


class FakeResponse:
    def __init__(self, status_code: int = 200, payload=None, json_exc: Exception | None = None):
        self.status_code = status_code
        self._payload = payload
        self._json_exc = json_exc

    def json(self):
        if self._json_exc is not None:
            raise self._json_exc
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class FakeSession:
    def __init__(self, response: FakeResponse | None = None, exc: Exception | None = None):
        self.response = response
        self.exc = exc
        self.calls: list[dict] = []

    def get(self, url, headers=None, timeout=None):
        self.calls.append({"url": url, "headers": headers, "timeout": timeout})
        if self.exc is not None:
            raise self.exc
        return self.response


def _items(*pairs):
    return {"items": [{"project": "ko.wikipedia", "article": "x", "granularity": "monthly",
                       "timestamp": ts, "access": "all-access", "agent": "user", "views": v}
                      for ts, v in pairs]}


@pytest.fixture
def pinned_today(monkeypatch):
    monkeypatch.setattr(connectors, "_today", lambda: pd.Timestamp("2026-11-11"))


# --------------------------------------------------------------------------- URL building


def test_url_korean_title_is_encoded(pinned_today):
    url = wikipedia_pageviews_url("인공 지능", lang="ko", start="2020-01-01", end="20201231")
    assert url == (f"{BASE}/ko.wikipedia/all-access/user/%EC%9D%B8%EA%B3%B5_%EC%A7%80%EB%8A%A5/"
                   "monthly/2020010100/2020123100")


def test_url_defaults_to_last_complete_period(pinned_today):
    monthly = wikipedia_pageviews_url("Python (programming language)", lang="en")
    assert monthly.endswith("/monthly/2016010100/2026103100")
    assert "/en.wikipedia/all-access/user/Python_%28programming_language%29/" in monthly
    daily = wikipedia_pageviews_url("Python", lang="en", granularity="daily")
    assert daily.endswith("/daily/2016010100/2026111000")


def test_url_slash_and_monthly_start_snap(pinned_today):
    url = wikipedia_pageviews_url("AC/DC", lang="en", start="2021-03-15", end="2021-12-31")
    assert "/AC%2FDC/" in url
    assert url.endswith("/monthly/2021030100/2021123100")
    daily = wikipedia_pageviews_url("AC/DC", lang="en", start="2021-03-15", end="2021-03-20",
                                    granularity="daily")
    assert daily.endswith("/daily/2021031500/2021032000")


@pytest.mark.parametrize("kwargs", [
    {"start": "2020/01/01"}, {"start": "20201301"}, {"end": "yesterday"},
    {"start": "2021-01-01", "end": "2020-01-01"}, {"granularity": "weekly"}, {"lang": "k o"},
])
def test_url_invalid_arguments(kwargs, pinned_today):
    with pytest.raises(ConnectorError):
        wikipedia_pageviews_url("인공지능", **kwargs)


def test_empty_article_rejected():
    with pytest.raises(ConnectorError):
        fetch_wikipedia_pageviews("   ", session=FakeSession(FakeResponse(200, _items())))


# --------------------------------------------------------------------------- fetching


def test_fetch_parses_and_sorts(pinned_today):
    payload = _items(("2020030100", 300), ("2020010100", 100), ("2020020100", 200))
    session = FakeSession(FakeResponse(200, payload))
    s = fetch_wikipedia_pageviews("인공 지능", start="2020-01-01", end="2020-03-31", session=session)
    assert len(session.calls) == 1
    call = session.calls[0]
    assert call["url"].startswith(f"{BASE}/ko.wikipedia/all-access/user/%EC%9D%B8")
    assert call["headers"] == {"User-Agent": USER_AGENT}
    assert USER_AGENT == "trendai/0.1 (time-series trend analysis tool)"
    assert call["timeout"] == 20
    assert s.name == "인공 지능"
    assert s.dtype == float
    assert list(s.index) == [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-02-01"),
                             pd.Timestamp("2020-03-01")]
    assert s.tolist() == [100.0, 200.0, 300.0]
    assert s.index.is_monotonic_increasing
    # result feeds straight into the pipeline's data preparation
    prepared, freq = prepare(s)
    assert freq == "M" and len(prepared) == 3


def test_fetch_daily_and_skips_malformed_items():
    payload = {"items": [{"timestamp": "2024010300", "views": 7},
                         {"timestamp": "2024010100", "views": 5},
                         {"timestamp": "garbage", "views": 1},
                         {"timestamp": "2024010200"},
                         "not-a-dict",
                         {"timestamp": "2024010200", "views": "6"}]}
    s = fetch_wikipedia_pageviews("Python", lang="en", start="20240101", end="20240103",
                                  granularity="daily", session=FakeSession(FakeResponse(200, payload)))
    assert s.tolist() == [5.0, 6.0, 7.0]
    assert s.index[0] == pd.Timestamp("2024-01-01")


def test_404_means_article_not_found():
    session = FakeSession(FakeResponse(404, {"type": "not_found"}))
    # final round (U13): the Korean text no longer carries an English parenthetical; the English
    # text (exc.text('en')) still says "article not found"
    with pytest.raises(ConnectorError, match="위키백과 문서를 찾을 수 없습니다") as ei:
        fetch_wikipedia_pageviews("없는문서제목xyz", start="2020-01-01", end="2020-12-31", session=session)
    assert "article not found" in ei.value.text("en")


@pytest.mark.parametrize("status", [400, 429, 500, 503])
def test_other_http_errors(status):
    session = FakeSession(FakeResponse(status, {}))
    with pytest.raises(ConnectorError):
        fetch_wikipedia_pageviews("Python", lang="en", start="2020-01-01", end="2020-12-31", session=session)


def test_network_error():
    session = FakeSession(exc=ConnectionError("connection reset"))
    # final round (U13): Korean advice instead of "(network error)"; the English text keeps the words
    with pytest.raises(ConnectorError, match="연결하지 못했습니다. 인터넷 연결") as ei:
        fetch_wikipedia_pageviews("Python", lang="en", start="2020-01-01", end="2020-12-31", session=session)
    assert "network error" in ei.value.text("en") and "ConnectionError: connection reset" in str(ei.value)
    long = FakeSession(exc=ConnectionError("HTTPSConnectionPool(host='wikimedia.org', port=443): Max retries "
                                           "exceeded with url: /api/rest_v1/metrics/pageviews/" + "x" * 300))
    with pytest.raises(ConnectorError) as ei:
        fetch_wikipedia_pageviews("Python", lang="en", start="2020-01-01", end="2020-12-31", session=long)
    assert len(str(ei.value)) < 260 and "x" * 200 not in str(ei.value)


@pytest.mark.parametrize("payload", [{"items": []}, {}, [], None])
def test_empty_items(payload):
    session = FakeSession(FakeResponse(200, payload))
    with pytest.raises(ConnectorError):
        fetch_wikipedia_pageviews("Python", lang="en", start="2020-01-01", end="2020-12-31", session=session)


def test_invalid_json():
    session = FakeSession(FakeResponse(200, json_exc=ValueError("no json")))
    with pytest.raises(ConnectorError, match="JSON"):
        fetch_wikipedia_pageviews("Python", lang="en", start="2020-01-01", end="2020-12-31", session=session)


def test_only_malformed_items():
    session = FakeSession(FakeResponse(200, {"items": [{"timestamp": "bad", "views": 1}]}))
    with pytest.raises(ConnectorError):
        fetch_wikipedia_pageviews("Python", lang="en", start="2020-01-01", end="2020-12-31", session=session)


def test_wiki_falls_back_to_urllib_without_requests(monkeypatch):
    """docs/WEB_SPEC.md B.3.1: without ``requests`` the stdlib UrllibSession is used (it used to raise
    conn_need_requests)."""
    monkeypatch.setitem(sys.modules, "requests", None)
    used: list[str] = []

    class RecordingSession(connectors.UrllibSession):
        def get(self, url, headers=None, timeout=connectors.TIMEOUT):
            used.append(url)
            return connectors.HttpResponse(200, {}, json.dumps(_items(("2020010100", 5))).encode())

    monkeypatch.setattr(connectors, "UrllibSession", RecordingSession)
    s = fetch_wikipedia_pageviews("Python", lang="en", start="2020-01-01", end="2020-01-31")
    assert len(used) == 1 and s.tolist() == [5.0]


# --------------------------------------------------------------------------- Google Trends


def _install_fake_pytrends(monkeypatch, frame=None, exc: Exception | None = None):
    calls: dict = {}

    class TrendReq:
        def __init__(self, hl=None, tz=None, **kwargs):
            calls["init"] = {"hl": hl, "tz": tz, **kwargs}

        def build_payload(self, kw_list, timeframe=None, geo=None, **kwargs):
            calls["payload"] = {"kw_list": kw_list, "timeframe": timeframe, "geo": geo}

        def interest_over_time(self):
            if exc is not None:
                raise exc
            return frame

    pkg = types.ModuleType("pytrends")
    request = types.ModuleType("pytrends.request")
    request.TrendReq = TrendReq
    pkg.request = request
    monkeypatch.setitem(sys.modules, "pytrends", pkg)
    monkeypatch.setitem(sys.modules, "pytrends.request", request)
    return calls


def test_google_trends_with_fake_pytrends(monkeypatch):
    idx = pd.date_range("2024-01-07", periods=4, freq="W-SUN")
    frame = pd.DataFrame({"챗GPT": [10, 40, 25, 100], "isPartial": [False, False, False, True]},
                         index=pd.Index(idx, name="date"))
    calls = _install_fake_pytrends(monkeypatch, frame)
    s = fetch_google_trends("챗GPT", timeframe="today 12-m", geo="KR")
    assert calls["init"] == {"hl": "ko-KR", "tz": 540}
    assert calls["payload"] == {"kw_list": ["챗GPT"], "timeframe": "today 12-m", "geo": "KR"}
    assert isinstance(s, pd.Series)
    assert s.name == "챗GPT"
    assert s.dtype == float
    assert s.tolist() == [10.0, 40.0, 25.0, 100.0]
    assert isinstance(s.index, pd.DatetimeIndex) and s.index.is_monotonic_increasing
    prepared, freq = prepare(s)
    assert freq == "W"


def test_google_trends_defaults(monkeypatch):
    frame = pd.DataFrame({"python": [1.0, 2.0]}, index=pd.date_range("2024-01-01", periods=2, freq="MS"))
    calls = _install_fake_pytrends(monkeypatch, frame)
    s = fetch_google_trends("python")
    assert calls["payload"] == {"kw_list": ["python"], "timeframe": "today 5-y", "geo": ""}
    assert s.tolist() == [1.0, 2.0]


def test_google_trends_empty_result(monkeypatch):
    _install_fake_pytrends(monkeypatch, pd.DataFrame())
    with pytest.raises(ConnectorError):
        fetch_google_trends("아무도검색안하는말")
    _install_fake_pytrends(monkeypatch, None)
    with pytest.raises(ConnectorError):
        fetch_google_trends("아무도검색안하는말")


def test_google_trends_request_error_mentions_rate_limit(monkeypatch):
    _install_fake_pytrends(monkeypatch, exc=RuntimeError("The request failed: Google returned 429"))
    with pytest.raises(ConnectorError, match="rate limit"):
        fetch_google_trends("python")


def test_google_trends_missing_package(monkeypatch):
    monkeypatch.setitem(sys.modules, "pytrends", None)
    monkeypatch.delitem(sys.modules, "pytrends.request", raising=False)
    with pytest.raises(ConnectorError, match="pip install pytrends"):
        fetch_google_trends("python")


def test_google_trends_empty_keyword():
    with pytest.raises(ConnectorError):
        fetch_google_trends("  ")


# --------------------------------------------------------------------------- adversarial review: regressions



@pytest.mark.parametrize("value", [pd.NaT, np.datetime64("NaT", "D")])
def test_nat_dates_raise_connector_error(value, pinned_today):
    with pytest.raises(ConnectorError):
        wikipedia_pageviews_url("Python", lang="en", start=value)
    with pytest.raises(ConnectorError):
        wikipedia_pageviews_url("Python", lang="en", end=value)


@pytest.mark.parametrize("value", [dt.date(2020, 1, 5), dt.datetime(2020, 1, 5, 13, 30),
                                   pd.Timestamp("2020-01-05"), np.datetime64("2020-01-05")])
def test_date_like_objects(value, pinned_today):
    url = wikipedia_pageviews_url("Python", lang="en", start=value, end=value, granularity="daily")
    assert url.endswith("/daily/2020010500/2020010500")


@pytest.mark.parametrize("today,end", [
    ("2026-03-01", "2026022800"),   # first of the month: previous month is the last complete one
    ("2024-03-15", "2024022900"),   # leap year
    ("2026-01-10", "2025123100"),   # year boundary
])
def test_default_monthly_end_edges(monkeypatch, today, end):
    monkeypatch.setattr(connectors, "_today", lambda: pd.Timestamp(today))
    assert wikipedia_pageviews_url("Python", lang="en").endswith(f"/monthly/2016010100/{end}")


def test_url_special_characters_are_encoded(pinned_today):
    url = wikipedia_pageviews_url("C++ (프로그래밍 언어)?#&", lang="ko", start="2020-01-01", end="2020-12-31")
    seg = url.split("/user/", 1)[1].split("/monthly/", 1)[0]
    assert seg == "C%2B%2B_%28%ED%94%84%EB%A1%9C%EA%B7%B8%EB%9E%98%EB%B0%8D_%EC%96%B8%EC%96%B4%29%3F%23%26"


def test_duplicate_timestamps_keep_last_and_name_restores_spaces():
    payload = _items(("2020010100", 1), ("2020010100", 9), ("2020020100", 2))
    s = fetch_wikipedia_pageviews("Machine_learning", lang="en", start="2020-01-01", end="2020-02-29",
                                  session=FakeSession(FakeResponse(200, payload)))
    assert s.tolist() == [9.0, 2.0]
    assert s.name == "Machine learning"
    assert s.index.name == "date"


class _BadStatusResponse(FakeResponse):
    def raise_for_status(self):
        raise RuntimeError("HTTP 418 teapot")


def test_raise_for_status_failure_is_wrapped():
    session = FakeSession(_BadStatusResponse(200, _items(("2020010100", 1))))
    with pytest.raises(ConnectorError, match="teapot"):
        fetch_wikipedia_pageviews("Python", lang="en", start="2020-01-01", end="2020-12-31", session=session)


def test_items_not_a_list():
    session = FakeSession(FakeResponse(200, {"items": "2020010100"}))
    with pytest.raises(ConnectorError):
        fetch_wikipedia_pageviews("Python", lang="en", start="2020-01-01", end="2020-12-31", session=session)


def test_broken_pytrends_install_is_a_connector_error(monkeypatch):
    class Broken(types.ModuleType):
        def __getattr__(self, name):
            raise ValueError("pandas 3 incompatible")   # not an ImportError

    monkeypatch.setitem(sys.modules, "pytrends", types.ModuleType("pytrends"))
    monkeypatch.setitem(sys.modules, "pytrends.request", Broken("pytrends.request"))
    with pytest.raises(ConnectorError, match="pytrends"):
        fetch_google_trends("python")


def test_google_trends_keeps_keyword_column_and_drops_nan(monkeypatch):
    idx = pd.date_range("2024-01-07", periods=3, freq="W-SUN")
    frame = pd.DataFrame({"other": [1, 2, 3], "python": [5.0, np.nan, 7.0], "isPartial": [False] * 3},
                         index=idx)
    _install_fake_pytrends(monkeypatch, frame)
    s = fetch_google_trends(" python ")
    assert s.name == "python"
    assert s.tolist() == [5.0, 7.0]


def test_connector_errors_are_catalog_messages():
    """R5-REV-5: ConnectorError texts were free Korean text, printed as such under --lang en."""
    with pytest.raises(ConnectorError) as ei:
        wikipedia_pageviews_url("Foo", start="2024-13-01")
    assert ei.value.code == "conn_date_invalid"
    assert str(ei.value) == "start 날짜가 유효하지 않습니다: '2024-13-01'"          # Korean unchanged
    assert ei.value.text("en") == "The start date is not a valid date: '2024-13-01'"
    with pytest.raises(ConnectorError) as ei:
        wikipedia_pageviews_url("Foo", start="2024-05-01", end="2024-01-01", granularity="daily")
    assert ei.value.code == "conn_start_after_end"
    assert ei.value.text("en") == "The start date (2024-05-01) is after the end date (2024-01-01)."
    with pytest.raises(ConnectorError) as ei:
        fetch_wikipedia_pageviews("Foo", session=types.SimpleNamespace(
            get=lambda url, **kw: FakeResponse(status_code=404)))
    assert "article not found" in ei.value.text("en") and not re.search("[가-힣]", ei.value.text("en"))


# --------------------------------------------------------------------------- web app: stdlib HTTP, weekly, UA


class _EchoHandler(http.server.BaseHTTPRequestHandler):
    def _send(self, status: int, body: bytes, content_type: str = "application/json") -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("X-Test-Header", "Yes")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _echo(self, body: bytes = b"") -> bytes:
        return json.dumps({"method": self.command, "path": self.path, "body": body.decode("utf-8"),
                           "headers": {k.lower(): v for k, v in self.headers.items()}}, ensure_ascii=False).encode()

    def do_GET(self):  # noqa: N802 - http.server API
        if self.path == "/missing":
            self._send(404, b'{"type": "not_found"}')
        elif self.path == "/slow":
            time.sleep(1.5)
            self._send(200, b"late")
        elif self.path == "/big":
            self._send(200, b"x" * 5000, "text/plain")
        else:
            self._send(200, self._echo())

    def do_POST(self):  # noqa: N802 - http.server API
        body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        self._send(201, self._echo(body))

    def log_message(self, *args):  # silence
        pass


class _QuietServer(http.server.ThreadingHTTPServer):
    daemon_threads = True
    block_on_close = False

    def handle_error(self, request, client_address):  # the /slow client hangs up early
        pass


@pytest.fixture
def local_server(monkeypatch):
    for var in ("http_proxy", "HTTP_PROXY", "https_proxy", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"):
        monkeypatch.delenv(var, raising=False)
    server = _QuietServer(("127.0.0.1", 0), _EchoHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


def test_urllib_session_get_post(local_server):
    session = connectors.UrllibSession()
    resp = session.get(local_server + "/hello?x=1", headers={"User-Agent": "ua-test", "X-Custom": "v"})
    assert isinstance(resp, connectors.HttpResponse) and resp.status_code == 200
    assert resp.headers["x-test-header"] == "Yes" and resp.headers["content-type"] == "application/json"
    assert all(k == k.lower() for k in resp.headers)
    echo = resp.json()
    assert echo["method"] == "GET" and echo["path"] == "/hello?x=1"
    assert echo["headers"]["user-agent"] == "ua-test" and echo["headers"]["x-custom"] == "v"
    assert json.loads(resp.text) == echo and resp.content == resp.text.encode()
    resp.raise_for_status()
    resp = session.post(local_server + "/api", json={"키": "값", "n": [1, 2]}, headers={"X-A": "1"}, timeout=5)
    echo = resp.json()
    assert resp.status_code == 201 and echo["method"] == "POST"
    assert json.loads(echo["body"]) == {"키": "값", "n": [1, 2]}
    assert echo["headers"]["content-type"] == "application/json" and echo["headers"]["x-a"] == "1"
    with pytest.raises(ValueError):
        connectors.HttpResponse(200, {}, b"<html>").json()
    with pytest.raises(ValueError):
        session.get("file:///etc/passwd")


def test_urllib_session_http_error_returned(local_server):
    resp = connectors.UrllibSession().get(local_server + "/missing")
    assert resp.status_code == 404 and resp.json() == {"type": "not_found"}
    with pytest.raises(RuntimeError, match="HTTP 404"):
        resp.raise_for_status()


def test_urllib_session_timeout_raises(local_server):
    t0 = time.perf_counter()
    with pytest.raises(TimeoutError):
        connectors.UrllibSession().get(local_server + "/slow", timeout=0.2)
    assert time.perf_counter() - t0 < 1.4


def test_urllib_session_size_limit_and_network_error(local_server):
    with pytest.raises(OSError, match="response too large"):
        connectors.UrllibSession(max_bytes=1000).get(local_server + "/big")
    assert len(connectors.UrllibSession(max_bytes=5000).get(local_server + "/big").content) == 5000
    import socket

    with socket.socket() as sock:   # a port that was free a moment ago: nothing listens there
        sock.bind(("127.0.0.1", 0))
        free_port = sock.getsockname()[1]
    with pytest.raises(OSError):    # connection refused -> URLError, an OSError
        connectors.UrllibSession().get(f"http://127.0.0.1:{free_port}/", timeout=2)


def _daily(start: str, end: str, value=lambda d: d.day) -> dict:
    days = pd.date_range(start, end, freq="D")
    return {"items": [{"timestamp": f"{d:%Y%m%d}00", "views": value(d)} for d in days]}


def test_wiki_weekly_aggregation(pinned_today):
    # the API answers daily data from Wednesday 2024-01-03 to Wednesday 2024-01-24: only the complete
    # Monday-Sunday weeks 01-08..01-14 and 01-15..01-21 are kept, indexed by their Monday
    session = FakeSession(FakeResponse(200, _daily("2024-01-03", "2024-01-24")))
    s = fetch_wikipedia_pageviews("Python", lang="en", start="2024-01-01", end="2024-01-28",
                                  granularity="weekly", session=session)
    url = session.calls[0]["url"]
    assert "/daily/2024010100/2024012800" in url   # start snapped to Monday 01-01, end to Sunday 01-28
    assert list(s.index) == [pd.Timestamp("2024-01-08"), pd.Timestamp("2024-01-15")]
    assert (s.index.dayofweek == 0).all() and s.index.name == "date" and s.name == "Python"
    assert s.tolist() == [float(sum(range(8, 15))), float(sum(range(15, 22)))]
    # request bounds are snapped: Wednesday..Wednesday asks for Monday 01-08 .. Sunday 01-21
    session = FakeSession(FakeResponse(200, _daily("2024-01-08", "2024-01-21", lambda d: 1)))
    s = fetch_wikipedia_pageviews("Python", lang="en", start="2024-01-03", end="2024-01-24",
                                  granularity="weekly", session=session)
    assert "/daily/2024010800/2024012100" in session.calls[0]["url"]
    assert s.tolist() == [7.0, 7.0]
    # a day without views inside a week is missing from the API answer: it counts as 0
    payload = _daily("2024-01-08", "2024-01-21", lambda d: 1)
    del payload["items"][3]
    s = fetch_wikipedia_pageviews("Python", lang="en", start="2024-01-08", end="2024-01-21",
                                  granularity="weekly", session=FakeSession(FakeResponse(200, payload)))
    assert s.tolist() == [6.0, 7.0]
    # default end = the last Sunday before today (2026-11-11 is a Wednesday)
    session = FakeSession(FakeResponse(200, _daily("2026-10-26", "2026-11-08")))
    fetch_wikipedia_pageviews("Python", lang="en", start="2026-10-26", granularity="weekly", session=session)
    assert session.calls[0]["url"].endswith("/daily/2026102600/2026110800")
    # less than one complete week
    with pytest.raises(ConnectorError) as ei:
        fetch_wikipedia_pageviews("Python", lang="en", start="2024-01-03", end="2024-01-06",
                                  granularity="weekly", session=FakeSession(FakeResponse(200, _daily(
                                      "2024-01-03", "2024-01-06"))))
    assert ei.value.code == "conn_start_after_end"
    with pytest.raises(ConnectorError) as ei:
        fetch_wikipedia_pageviews("Python", lang="en", start="2024-01-01", end="2024-01-14",
                                  granularity="weekly", session=FakeSession(FakeResponse(200, _daily(
                                      "2024-01-03", "2024-01-09"))))
    assert ei.value.code == "conn_wiki_empty"


def test_wiki_weekly_accepts_dates_from_period_bounds(pinned_today):
    start, end = connectors.period_bounds("all", "week", dt.date(2026, 11, 11), connectors.WIKI_EARLIEST)
    session = FakeSession(FakeResponse(200, _daily("2016-01-04", "2016-01-17")))
    s = fetch_wikipedia_pageviews("Python", lang="en", start=start, end=end, granularity="weekly", session=session)
    assert session.calls[0]["url"].endswith("/daily/2016010400/2026110800") and len(s) == 2


def test_wiki_user_agent_param(pinned_today):
    session = FakeSession(FakeResponse(200, _items(("2020010100", 1))))
    ua = connectors.user_agent("https://trendai.example", "ops@example.com")
    fetch_wikipedia_pageviews("Python", lang="en", start="2020-01-01", end="2020-01-31", session=session,
                              user_agent=ua)
    assert session.calls[0]["headers"] == {"User-Agent": "trendai-web/0.1.0 (+https://trendai.example; "
                                                         "ops@example.com)"}
    fetch_wikipedia_pageviews("Python", lang="en", start="2020-01-01", end="2020-01-31", session=session)
    assert session.calls[1]["headers"] == {"User-Agent": USER_AGENT}


def test_wiki_timeout_code():
    class ReadTimeout(Exception):   # requests.exceptions.ReadTimeout-like
        pass

    for exc in (TimeoutError("timed out"), ReadTimeout("read timed out"),
                connectors.urllib.error.URLError(TimeoutError("timed out"))):
        with pytest.raises(ConnectorError) as ei:
            fetch_wikipedia_pageviews("Python", lang="en", start="2020-01-01", end="2020-12-31",
                                      session=FakeSession(exc=exc))
        assert ei.value.code == "conn_timeout"
        assert ei.value.text("en") == "The Wikimedia API did not answer within 20 s."
    with pytest.raises(ConnectorError) as ei:
        fetch_wikipedia_pageviews("Python", lang="en", start="2020-01-01", end="2020-12-31",
                                  session=FakeSession(exc=ConnectionResetError("reset")))
    assert ei.value.code == "conn_network"


def test_user_agent_format():
    assert connectors.user_agent() == "trendai-web/0.1.0"
    assert connectors.user_agent("https://trendai.example", "ops@example.com") == \
        "trendai-web/0.1.0 (+https://trendai.example; ops@example.com)"
    assert connectors.user_agent(public_url="https://t.example") == "trendai-web/0.1.0 (+https://t.example)"
    assert connectors.user_agent(contact="ops@example.com") == "trendai-web/0.1.0 (ops@example.com)"
    assert connectors.user_agent("", "  ") == "trendai-web/0.1.0"
    # non-printable / non-ASCII characters (header injection, Hangul) are dropped
    assert connectors.user_agent("https://t.example\r\nX-Evil: 1", "운영 ops@example.com") == \
        "trendai-web/0.1.0 (+https://t.exampleX-Evil: 1; ops@example.com)"
    import trendai
    assert connectors.user_agent().endswith(trendai.__version__)
