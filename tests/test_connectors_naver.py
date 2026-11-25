"""Naver search-trend connector, stdlib HTTP session thread safety, periods (docs/WEB_SPEC.md B.3.1, H.2).

No test touches the network: the connector gets a fake session; UrllibSession talks to a local
http.server on 127.0.0.1."""

from __future__ import annotations

import datetime as dt
import http.server
import json
import threading
import urllib.error

import pandas as pd
import pytest

from trendai.connectors import (
    NAVER_HUB_URL, NAVER_LEGACY_URL, ConnectorError, HttpResponse, NaverCredentials, NaverGroup, NaverResult,
    UrllibSession, build_naver_body, fetch_naver_search_trend, naver_cache_key, period_bounds,
)

KEY_ID = "SECRET-KEY-ID-7f3a"
KEY = "SECRET-KEY-9b1c"
HUB = NaverCredentials("hub", KEY_ID, KEY)
LEGACY = NaverCredentials("legacy", KEY_ID, KEY)
G1 = NaverGroup("g1", ("선크림", "자외선차단제"))
G2 = NaverGroup("g2", ("패딩",))
START, END = dt.date(2026, 10, 5), dt.date(2026, 11, 1)   # Monday .. Sunday: 4 complete weeks
WEEKS = ["2026-10-05", "2026-10-12", "2026-10-19", "2026-10-26"]


class FakeSession:
    def __init__(self, response=None, exc: BaseException | None = None):
        self.response = response
        self.exc = exc
        self.calls: list[dict] = []

    def post(self, url, json=None, headers=None, timeout=None):
        self.calls.append({"url": url, "json": json, "headers": dict(headers or {}), "timeout": timeout})
        if self.exc is not None:
            raise self.exc
        return self.response


def _response(payload, status: int = 200) -> HttpResponse:
    content = payload if isinstance(payload, bytes) else json.dumps(payload, ensure_ascii=False).encode()
    return HttpResponse(status, {"Content-Type": "application/json"}, content)


def _payload(*groups: tuple[str, list], keywords=None) -> dict:
    return {"startDate": START.isoformat(), "endDate": END.isoformat(), "timeUnit": "week",
            "results": [{"title": name, "keywords": keywords if keywords is not None else ["k"],
                         "data": [{"period": p, "ratio": r} for p, r in data]} for name, data in groups]}


def _fetch(payload, groups=(G1,), **kw):
    session = FakeSession(_response(payload))
    result = fetch_naver_search_trend(list(groups), kw.pop("start", START), kw.pop("end", END),
                                      kw.pop("time_unit", "week"), credentials=kw.pop("credentials", HUB),
                                      session=session, **kw)
    return result, session


# --------------------------------------------------------------------------- request body


def test_build_body_minimal_and_full():
    body = build_naver_body([G1], START, END, "week")
    assert body == {"startDate": "2026-10-05", "endDate": "2026-11-01", "timeUnit": "week",
                    "keywordGroups": [{"groupName": "g1", "keywords": ["선크림", "자외선차단제"]}]}
    full = build_naver_body([G1, G2], START, END, "month", device="mo", gender="f", ages=["10", "4", "2", "4"])
    assert full["device"] == "mo" and full["gender"] == "f"
    assert full["ages"] == ["2", "4", "10"]                       # numeric order, de-duplicated
    assert [g["groupName"] for g in full["keywordGroups"]] == ["g1", "g2"]
    # empty optional filters are omitted
    assert "device" not in build_naver_body([G1], START, END, "date", device="", gender=None, ages=())


@pytest.mark.parametrize("kwargs,code,what", [
    ({"groups": [NaverGroup(f"g{i}", ("a",)) for i in range(6)]}, "conn_naver_too_many_groups", None),
    ({"groups": [NaverGroup("g1", tuple("abcdef"))]}, "conn_naver_too_many_keywords", None),
    ({"groups": [NaverGroup("g1", ("a",)), NaverGroup("g1", ("b",))]}, "conn_naver_bad_filter", "groupName"),
    ({"groups": [NaverGroup("", ("a",))]}, "conn_naver_bad_filter", "groupName"),
    ({"groups": [NaverGroup("x" * 101, ("a",))]}, "conn_naver_bad_filter", "groupName"),
    ({"groups": [NaverGroup("g1", ())]}, "conn_naver_bad_filter", "keywords"),
    ({"groups": [NaverGroup("g1", ("a", " "))]}, "conn_naver_bad_filter", "keywords"),
    ({"groups": []}, "conn_naver_bad_filter", "keywordGroups"),
    ({"time_unit": "day"}, "conn_naver_bad_filter", "timeUnit"),
    ({"device": "tv"}, "conn_naver_bad_filter", "device"),
    ({"gender": "x"}, "conn_naver_bad_filter", "gender"),
    ({"ages": ["4", "12"]}, "conn_naver_bad_filter", "ages"),
    ({"ages": ["40"]}, "conn_naver_bad_filter", "ages"),
    ({"start": dt.date(2015, 12, 31)}, "conn_bad_range", None),
    ({"start": dt.date(2026, 11, 2), "end": dt.date(2026, 11, 1)}, "conn_bad_range", None),
])
def test_validation_errors(kwargs, code, what):
    args = {"groups": [G1], "start": START, "end": END, "time_unit": "week", "device": None, "gender": None,
            "ages": (), **kwargs}
    with pytest.raises(ConnectorError) as ei:
        build_naver_body(args.pop("groups"), args.pop("start"), args.pop("end"), args.pop("time_unit"), **args)
    assert ei.value.code == code
    if what is not None:
        assert ei.value.params["what"] == what
    if code == "conn_naver_too_many_groups":
        assert ei.value.params == {"max": 5}
    if code == "conn_naver_too_many_keywords":
        assert ei.value.params == {"max": 5, "group": "g1"}
    # the fetch validates before any request
    session = FakeSession(_response({}))
    with pytest.raises(ConnectorError):
        fetch_naver_search_trend(kwargs.get("groups", [G1]), kwargs.get("start", START), kwargs.get("end", END),
                                 kwargs.get("time_unit", "week"), kwargs.get("device"), kwargs.get("gender"),
                                 kwargs.get("ages", ()), credentials=HUB, session=session)
    assert session.calls == []


def test_cache_key_canonical():
    key = naver_cache_key([G1, G2], START, END, "week", "mo", "f", ["4", "10", "5"])
    assert len(key) == 64 and int(key, 16) >= 0
    assert key == naver_cache_key([G1, G2], START, END, "week", "mo", "f", ["10", "5", "4"])   # age order
    assert key != naver_cache_key([G2, G1], START, END, "week", "mo", "f", ["4", "10", "5"])   # group order
    assert key != naver_cache_key([G1, G2], START, END, "month", "mo", "f", ["4", "10", "5"])
    body = build_naver_body([G1, G2], START, END, "week", "mo", "f", ["4", "10", "5"])
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    import hashlib
    assert key == hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    # credentials / endpoint are not part of it (there is no parameter for them at all)
    assert KEY not in canonical and "hub" not in canonical


# --------------------------------------------------------------------------- credentials


def test_credentials_from_env_precedence():
    env = {"NAVER_API_HUB_KEY_ID": "hid", "NAVER_API_HUB_KEY": "hkey", "NAVER_CLIENT_ID": "cid",
           "NAVER_CLIENT_SECRET": "csec"}
    assert NaverCredentials.from_env(env) == NaverCredentials("hub", "hid", "hkey")
    assert NaverCredentials.from_env({**env, "NAVER_API_HUB_KEY": " "}) == NaverCredentials("legacy", "cid", "csec")
    assert NaverCredentials.from_env({"NAVER_API_HUB_KEY_ID": "hid", "NAVER_CLIENT_ID": "cid"}) is None
    assert NaverCredentials.from_env({"NAVER_CLIENT_SECRET": "csec"}) is None
    assert NaverCredentials.from_env({}) is None
    assert NaverCredentials.from_env({"NAVER_CLIENT_ID": " cid ", "NAVER_CLIENT_SECRET": "csec"}).key_id == "cid"
    with pytest.raises(ValueError):
        NaverCredentials("both", "a", "b")


def test_credentials_repr_masks():
    for creds in (HUB, LEGACY):
        for text in (repr(creds), str(creds), f"{creds}", repr([creds])):
            assert KEY not in text and KEY_ID not in text
    assert repr(HUB) == "NaverCredentials(endpoint='hub', key_id='***', key='***')"


# --------------------------------------------------------------------------- requests


def test_hub_headers_and_url():
    result, session = _fetch(_payload(("g1", [(p, 50.0) for p in WEEKS])))
    (call,) = session.calls
    assert call["url"] == NAVER_HUB_URL == "https://naverapihub.apigw.ntruss.com/search-trend/v1/search"
    h = call["headers"]
    assert h["X-NCP-APIGW-API-KEY-ID"] == KEY_ID and h["X-NCP-APIGW-API-KEY"] == KEY
    assert "X-Naver-Client-Id" not in h and h["Content-Type"] == "application/json"
    assert h["User-Agent"].startswith("trendai-web/")
    assert call["json"] == build_naver_body([G1], START, END, "week") and call["timeout"] == 20
    assert result.endpoint == "hub" and result.series["g1"].attrs["trendai_source"]["endpoint"] == "hub"


def test_legacy_headers_and_url():
    result, session = _fetch(_payload(("g1", [(p, 50.0) for p in WEEKS])), credentials=LEGACY, timeout=5)
    (call,) = session.calls
    assert call["url"] == NAVER_LEGACY_URL == "https://openapi.naver.com/v1/datalab/search"
    h = call["headers"]
    assert h["X-Naver-Client-Id"] == KEY_ID and h["X-Naver-Client-Secret"] == KEY
    assert "X-NCP-APIGW-API-KEY" not in h and call["timeout"] == 5
    assert result.endpoint == "legacy"


def test_result_shape_and_attrs():
    payload = _payload(("g1", [(p, r) for p, r in zip(WEEKS, (10, 100, 40.5, 20))]),
                       ("g2", [(p, r) for p, r in zip(WEEKS, (1, 2, 3, 4))]))
    result, _ = _fetch(payload, groups=(G1, G2), device="mo", gender="f", ages=["5", "4"])
    assert isinstance(result, NaverResult)
    assert list(result.series) == ["g1", "g2"] and result.time_unit == "week"
    assert (result.start, result.end) == (START, END) and result.raw_max == 100.0
    assert result.filled == {"g1": 0, "g2": 0}
    s = result.series["g1"]
    assert s.dtype == float and s.name == "g1" and s.index.name == "date"
    assert list(s.index) == list(pd.to_datetime(WEEKS)) and s.tolist() == [10.0, 100.0, 40.5, 20.0]
    assert s.attrs["trendai_source"] == {"kind": "naver", "group": "g1", "keywords": ["선크림", "자외선차단제"],
                                         "time_unit": "week", "device": "mo", "gender": "f", "ages": ["4", "5"],
                                         "endpoint": "hub"}
    json.dumps(s.attrs)    # JSON-serialisable (the web app caches it)


def test_parse_ratio_variants():
    payload = _payload(("g1", [(WEEKS[0], 12), (WEEKS[1], 12.5), (WEEKS[2], "33.25"), (WEEKS[3], " 7 ")]))
    result, _ = _fetch(payload)
    assert result.series["g1"].tolist() == [12.0, 12.5, 33.25, 7.0]
    for keywords in (["선크림", "자외선차단제"], "선크림,자외선차단제", None):
        result, _ = _fetch(_payload(("g1", [(p, 1) for p in WEEKS]), keywords=keywords))
        assert result.series["g1"].attrs["trendai_source"]["keywords"] == ["선크림", "자외선차단제"]
    for bad in ("n/a", None, True, float("nan"), [1]):
        with pytest.raises(ConnectorError) as ei:
            _fetch(_payload(("g1", [(WEEKS[0], bad)])))
        assert ei.value.code == "conn_naver_bad_response"


def test_missing_buckets_filled():
    payload = _payload(("g1", [(WEEKS[0], 10), (WEEKS[2], 30), (WEEKS[3], 40)]), ("g2", []))
    result, _ = _fetch(payload, groups=(G1, G2))
    assert result.series["g1"].tolist() == [10.0, 0.0, 30.0, 40.0]
    assert result.series["g2"].tolist() == [0.0] * 4
    assert result.filled == {"g1": 1, "g2": 4}
    # a gap at the start is filled too: the grid starts at the requested start
    result, _ = _fetch(_payload(("g1", [(WEEKS[2], 30), (WEEKS[3], 40)])))
    assert result.series["g1"].tolist() == [0.0, 0.0, 30.0, 40.0] and result.filled == {"g1": 2}


def test_partial_trailing_week_dropped():
    # end on a Wednesday: the bucket of 2026-11-02 would cover 11-02..11-08 > end -> dropped
    payload = _payload(("g1", [(p, 10) for p in WEEKS] + [("2026-11-02", 100)]))
    result, _ = _fetch(payload, end=dt.date(2026, 11, 4))
    s = result.series["g1"]
    assert list(s.index) == list(pd.to_datetime(WEEKS)) and s.tolist() == [10.0] * 4
    assert result.raw_max == 100.0 and result.filled == {"g1": 0}


def test_month_and_date_grids():
    payload = {"results": [{"title": "g1", "keywords": ["a"],
                            "data": [{"period": "2026-01-01", "ratio": 5}, {"period": "2026-03-01", "ratio": 7}]}]}
    result, _ = _fetch(payload, start=dt.date(2026, 1, 1), end=dt.date(2026, 3, 31), time_unit="month")
    s = result.series["g1"]
    assert list(s.index) == list(pd.to_datetime(["2026-01-01", "2026-02-01", "2026-03-01"]))
    assert s.tolist() == [5.0, 0.0, 7.0] and result.filled == {"g1": 1}
    payload = {"results": [{"title": "g1", "keywords": ["a"], "data": [{"period": "2026-01-02", "ratio": 1}]}]}
    result, _ = _fetch(payload, start=dt.date(2026, 1, 1), end=dt.date(2026, 1, 3), time_unit="date")
    assert result.series["g1"].tolist() == [0.0, 1.0, 0.0]


def test_week_shorter_than_one_bucket_is_bad_range():
    with pytest.raises(ConnectorError) as ei:
        _fetch(_payload(("g1", [])), start=dt.date(2026, 10, 5), end=dt.date(2026, 10, 9))
    assert ei.value.code == "conn_bad_range"


def test_title_mismatch_bad_response():
    with pytest.raises(ConnectorError) as ei:
        _fetch(_payload(("선크림", [(p, 1) for p in WEEKS])))
    assert ei.value.code == "conn_naver_bad_response"
    with pytest.raises(ConnectorError) as ei:      # fewer results than groups
        _fetch(_payload(("g1", [(p, 1) for p in WEEKS])), groups=(G1, G2))
    assert ei.value.code == "conn_naver_bad_response"
    with pytest.raises(ConnectorError) as ei:      # results in the wrong order
        _fetch(_payload(("g2", []), ("g1", [])), groups=(G1, G2))
    assert ei.value.code == "conn_naver_bad_response"


@pytest.mark.parametrize("status,body,exc,code", [
    (401, {"error": {"errorCode": "401", "message": "Authentication failed"}}, None, "conn_naver_auth"),
    (400, {"error": {"errorCode": "300", "message": "API not enabled", "details": "x"}}, None, "conn_naver_auth"),
    (403, {"errorMessage": "Scope Status Invalid", "errorCode": "024"}, None, "conn_naver_auth"),
    (429, {"error": {"errorCode": "429", "message": "Quota Exceeded"}}, None, "conn_naver_rate"),
    (500, {"error": {"errorCode": "500", "message": "Internal Server Error"}}, None, "conn_naver_server"),
    (503, b"<html>gateway</html>", None, "conn_naver_server"),
    (400, {"errorMessage": "Incorrect query request", "errorCode": "SE01"}, None, "conn_naver_http"),
    (200, b"{not json", None, "conn_naver_bad_response"),
    (200, {"error": {"errorCode": "999", "message": "odd"}}, None, "conn_naver_bad_response"),
    (None, None, urllib.error.URLError(ConnectionRefusedError(111, "refused")), "conn_naver_network"),
    (None, None, TimeoutError("timed out"), "conn_naver_timeout"),
    (None, None, urllib.error.URLError(TimeoutError("timed out")), "conn_naver_timeout"),
])
def test_error_mapping(status, body, exc, code):
    session = FakeSession(_response(body, status) if status is not None else None, exc=exc)
    with pytest.raises(ConnectorError) as ei:
        fetch_naver_search_trend([G1], START, END, credentials=HUB, session=session)
    e = ei.value
    assert e.code == code
    if code in ("conn_naver_auth", "conn_naver_server", "conn_naver_http"):
        assert e.params["status"] == status
    if isinstance(body, dict):
        error = body.get("error", body)
        assert error["errorCode"] in e.log_detail and len(e.log_detail) <= 200
    for text in (str(e), e.text("en"), e.log_detail or "", repr(e)):
        assert KEY not in text and KEY_ID not in text


def test_no_secret_in_exception_text():
    # an exception text or an error body that echoes the keys never reaches the message or the log detail
    session = FakeSession(exc=ConnectionError(f"proxy rejected header X-NCP-APIGW-API-KEY: {KEY} ({KEY_ID})"))
    with pytest.raises(ConnectorError) as ei:
        fetch_naver_search_trend([G1], START, END, credentials=HUB, session=session)
    body = {"error": {"errorCode": "401", "message": f"invalid key {KEY_ID}/{KEY}"}}
    with pytest.raises(ConnectorError) as ei2:
        fetch_naver_search_trend([G1], START, END, credentials=HUB, session=FakeSession(_response(body, 401)))
    for e in (ei.value, ei2.value):
        for text in (str(e), e.text("en"), e.log_detail or "", repr(e), repr(e.params)):
            assert KEY not in text and KEY_ID not in text
    assert ei.value.code == "conn_naver_network" and "***" in ei.value.log_detail
    with pytest.raises(ConnectorError) as ei3:
        fetch_naver_search_trend([G1], START, END, credentials=None, session=FakeSession())
    assert ei3.value.code == "conn_naver_unconfigured"


# --------------------------------------------------------------------------- periods

TODAY = dt.date(2026, 11, 25)   # a Wednesday


@pytest.mark.parametrize("spec,unit,expected", [
    ("5y", "week", (dt.date(2021, 11, 29), dt.date(2026, 11, 22))),
    ("5y", "month", (dt.date(2021, 11, 1), dt.date(2026, 10, 31))),
    ("1y", "date", (dt.date(2025, 11, 25), dt.date(2026, 11, 24))),
    ("3y", "week", (dt.date(2023, 11, 27), dt.date(2026, 11, 22))),
    ("all", "week", (dt.date(2016, 1, 4), dt.date(2026, 11, 22))),
    ("all", "month", (dt.date(2016, 1, 1), dt.date(2026, 10, 31))),
    ("all", "date", (dt.date(2016, 1, 1), dt.date(2026, 11, 24))),
    ("2015-06-10~2030-01-01", "week", (dt.date(2016, 1, 4), dt.date(2026, 11, 22))),
    ("2024-02-15~2024-06-15", "month", (dt.date(2024, 3, 1), dt.date(2024, 5, 31))),
    ("2024-01-03~2024-01-28", "week", (dt.date(2024, 1, 8), dt.date(2024, 1, 28))),
    (" 2024-01-01 ~ 2024-01-31 ", "date", (dt.date(2024, 1, 1), dt.date(2024, 1, 31))),
])
def test_period_bounds(spec, unit, expected):
    start, end = period_bounds(spec, unit, TODAY)
    assert (start, end) == expected
    if unit == "week":
        assert start.weekday() == 0 and end.weekday() == 6
    if unit == "month":
        assert start.day == 1 and (end + dt.timedelta(days=1)).day == 1


@pytest.mark.parametrize("spec", ["5x", "2024-13-01~2025-01-01", "2025-01-01~2024-01-01", "", "5", "y",
                                  "2026-11-23~2026-11-30"])
def test_period_bounds_errors(spec):
    with pytest.raises(ConnectorError) as ei:
        period_bounds(spec, "week", TODAY)
    assert ei.value.code == "conn_bad_range"
    with pytest.raises(ConnectorError):
        period_bounds("5y", "day", TODAY)


def test_period_bounds_week_edges():
    monday, sunday = dt.date(2026, 11, 23), dt.date(2026, 11, 29)
    assert period_bounds("1y", "week", monday)[1] == dt.date(2026, 11, 22)   # yesterday was Sunday
    assert period_bounds("1y", "week", sunday)[1] == dt.date(2026, 11, 22)   # this week is not complete
    assert period_bounds("1y", "month", dt.date(2024, 3, 1))[1] == dt.date(2024, 2, 29)
    assert period_bounds("1y", "date", dt.date(2024, 3, 1)) == (dt.date(2023, 3, 1), dt.date(2024, 2, 29))
    assert period_bounds("all", "week", TODAY, earliest=dt.date(2020, 1, 1))[0] == dt.date(2020, 1, 6)


# --------------------------------------------------------------------------- UrllibSession thread safety


class _Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802 - http.server API
        body = self.path.encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):  # silence
        pass


class _Server(http.server.ThreadingHTTPServer):
    daemon_threads = True
    block_on_close = False


@pytest.fixture
def local_server(monkeypatch):
    for var in ("http_proxy", "HTTP_PROXY", "https_proxy", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"):
        monkeypatch.delenv(var, raising=False)
    server = _Server(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


def test_urllib_session_thread_safe(local_server):
    session = UrllibSession()
    errors: list[str] = []

    def worker(t: int) -> None:
        for i in range(20):
            path = f"/t{t}/r{i}"
            resp = session.get(local_server + path, timeout=10)
            if resp.status_code != 200 or resp.text != path:
                errors.append(f"{path}: {resp.status_code} {resp.text!r}")

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(8)]
    for th in threads:
        th.start()
    for th in threads:
        th.join(30)
    assert not errors and not any(th.is_alive() for th in threads)
