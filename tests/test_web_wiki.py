"""Wikipedia flows (C.5, D.2.2, D.4.3, E.6, E.10) with ``testing.FakeWiki`` and, for the User-Agent, the default
fetchers over a fake HTTP session."""

from __future__ import annotations

import datetime as dt
import json

import pandas as pd
import pytest

from trendai import connectors
from trendai.web.config import WebConfig
from trendai.web.fetchers import default_fetchers
from trendai.web.testing import FrozenClock, TestClient, analyze_and_wait, make_test_app, write_static_fixture

T0 = dt.datetime(2026, 11, 25, 0, 12, tzinfo=dt.timezone.utc)


@pytest.fixture
def static(tmp_path):
    return write_static_fixture(tmp_path / "static")


@pytest.fixture
def make(tmp_path, static):
    apps = []

    def factory(**kw):
        kw.setdefault("static_dir", static)
        app = make_test_app(tmp_path / f"data{len(apps)}", **kw)
        apps.append(app)
        return app

    yield factory
    for app in apps:
        app.close(timeout=10)


def wiki(*titles, **extra):
    return {"source": "wiki", "series": [{"terms": [t]} for t in titles], **extra}


def test_wiki_monthly_detail(make):
    app = make(clock=FrozenClock(T0))
    snap, env = analyze_and_wait(TestClient(app), wiki("인공지능"))
    assert snap["status"] == "done"
    vm = env["series"][0]
    assert "Wikimedia Pageviews" in vm["source"]["line"]
    assert vm["source"]["value_kind"] == "pageviews" and vm["source"]["freq"] == "M"
    assert vm["source"]["as_of"] == "2026-11-25T00:12:00Z"
    call = app.fetchers.wiki.calls[0]
    assert (call["title"], call["lang"], call["granularity"]) == ("인공지능", "ko", "monthly")
    assert (call["start"], call["end"]) == (dt.date(2021, 11, 1), dt.date(2026, 10, 31))
    assert env["query"]["unit"] == "month" and env["query"]["wl"] == "ko"
    assert env["query"]["period"] == {"start": "2021-11-01", "end": "2026-10-31"}
    assert vm["narrative"]["ai_allowed"] is True


def test_wiki_404_series_card(make):
    snap, env = analyze_and_wait(TestClient(make()), wiki("인공지능", "없는문서"))
    assert snap["status"] == "done" and env["view"] == "compare"
    assert len(env["series"]) == 1 and env["series"][0]["index"] == 0
    assert len(env["failed"]) == 1
    failed = env["failed"][0]
    assert (failed["index"], failed["name"]) == (1, "없는문서")
    err = failed["error"]
    assert err["code"] == "wiki_not_found" and err["status"] == 404
    assert err["message_ko"].startswith("위키백과에 ‘없는문서’ 문서가 없어요.")
    help_fix = err["fixes"][0]
    assert help_fix["action"] == "wiki_title_help"
    assert help_fix["params"]["url"] == ("https://ko.wikipedia.org/w/index.php?search="
                                         "%EC%97%86%EB%8A%94%EB%AC%B8%EC%84%9C")
    assert env["compare"]["failed"] == env["failed"]
    assert env["compare"]["count"] == 1


def test_wiki_other_language_fix_url(make):
    snap, env = analyze_and_wait(TestClient(make()), wiki("인공지능", "없는문서", wl="en"))
    assert env["failed"][0]["error"]["fixes"][0]["params"]["url"].startswith("https://en.wikipedia.org/")


def test_wiki_weekly_unit(make):
    app = make(clock=FrozenClock(T0))
    snap, env = analyze_and_wait(TestClient(app), wiki("인공지능", unit="week", range="1y"))
    assert snap["status"] == "done"
    vm = env["series"][0]
    assert vm["source"]["freq"] == "W" and vm["chart"]["freq"] == "W"
    call = app.fetchers.wiki.calls[0]
    assert call["granularity"] == "weekly"
    assert call["start"].weekday() == 0 and call["end"] == dt.date(2026, 11, 22)
    assert env["query"]["unit"] == "week" and env["query"]["h"] == 26


def test_wiki_cache_ttls(make):
    clock = FrozenClock(T0)
    app = make(clock=clock)
    c = TestClient(app)
    analyze_and_wait(c, wiki("월간"))
    analyze_and_wait(c, wiki("주간", unit="week", range="1y"))
    assert len(app.fetchers.wiki.calls) == 2
    clock.advance(5 * 3600)
    analyze_and_wait(c, wiki("월간", h=6))
    analyze_and_wait(c, wiki("주간", unit="week", range="1y", h=13))
    assert len(app.fetchers.wiki.calls) == 2                  # both cached
    clock.advance(2 * 3600)                                   # +7 h: weekly (6 h) expired, monthly (24 h) not
    analyze_and_wait(c, wiki("월간", h=3))
    analyze_and_wait(c, wiki("주간", unit="week", range="1y", h=4))
    assert [x["title"] for x in app.fetchers.wiki.calls] == ["월간", "주간", "주간"]
    clock.advance(18 * 3600)                                  # +25 h
    analyze_and_wait(c, wiki("월간", h=24))
    assert [x["title"] for x in app.fetchers.wiki.calls][-1] == "월간"
    assert len(app.fetchers.wiki.calls) == 4


class _FakeSession:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def get(self, url, headers=None, timeout=None):
        self.calls.append({"url": url, "headers": dict(headers or {})})
        items = [{"timestamp": f"{y}{m:02d}0100", "views": 1000 + 10 * i}
                 for i, (y, m) in enumerate((y, m) for y in (2024, 2025) for m in range(1, 13))]
        return connectors.HttpResponse(200, {"content-type": "application/json"},
                                       json.dumps({"items": items}).encode())

    def post(self, url, json=None, headers=None, timeout=None):   # noqa: A002 - mirrors UrllibSession
        raise AssertionError("no POST expected")


def test_user_agent_contains_contact(monkeypatch):
    session = _FakeSession()
    monkeypatch.setattr(connectors, "UrllibSession", lambda *a, **k: session)
    for name in ("NAVER_API_HUB_KEY", "NAVER_API_HUB_KEY_ID", "NAVER_CLIENT_ID", "NAVER_CLIENT_SECRET"):
        monkeypatch.delenv(name, raising=False)
    cfg = WebConfig.from_env(environ={}, public_url="https://trendai.example", contact="ops@example.com", ai="off")
    fetchers = default_fetchers(cfg)
    assert fetchers.naver is None and fetchers.naver_endpoint is None and fetchers.llm is None
    s = fetchers.wiki("인공지능", "ko", dt.date(2024, 1, 1), dt.date(2025, 12, 31), "monthly")
    assert isinstance(s, pd.Series) and len(s) == 24
    ua = session.calls[0]["headers"]["User-Agent"]
    assert ua == "trendai-web/0.1.0 (+https://trendai.example; ops@example.com)"
    assert "wikimedia.org" in session.calls[0]["url"]


def test_log_hint(make):
    app = make()
    real = app.fetchers.wiki

    def scaled(title, lang, start, end, granularity):
        s = real(title, lang, start, end, granularity)
        return s * (40.0 if title == "큰문서" else 1.0)

    app.fetchers.wiki = scaled
    snap, env = analyze_and_wait(TestClient(app), wiki("큰문서", "작은문서"))
    scale = env["compare"]["scale"]
    assert scale["log_hint"] is True and scale["allowed"]["log"] is True
    hints = [n for n in env["compare"]["notices"] if n["code"] == "log_hint"]
    assert len(hints) == 1 and "20배" in hints[0]["text"]
    assert hints[0]["fixes"][0]["action"] == "use_log"
