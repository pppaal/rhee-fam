"""``GET /api/config`` (C.2) and ``WebConfig`` (E.1, B.3.2)."""

from __future__ import annotations

import datetime as dt

import pytest

from trendai.web import create_app, exports
from trendai.web.config import WebConfig
from trendai.web.fetchers import ai_status
from trendai.web.testing import FakeNaver, FrozenClock, TestClient, make_test_app, write_static_fixture


@pytest.fixture
def make(tmp_path):
    apps = []

    def factory(**kw):
        kw.setdefault("static_dir", write_static_fixture(tmp_path / "static"))
        app = make_test_app(tmp_path / f"data{len(apps)}", **kw)
        apps.append(app)
        return app

    yield factory
    for app in apps:
        app.close(timeout=5)


def test_config_naver_unconfigured(make):
    body = TestClient(make(naver=False)).get("/api/config").json()
    assert body["sources"]["naver"] == {"available": False, "reason": "unconfigured", "endpoint": None,
                                        "budget_state": "ok"}
    assert body["default_source"] == "wiki"


def test_config_naver_hub(make):
    resp = TestClient(make()).get("/api/config")
    assert resp.status == 200
    assert resp.headers["cache-control"] == "no-cache"
    body = resp.json()
    assert body["sources"]["naver"] == {"available": True, "reason": None, "endpoint": "hub", "budget_state": "ok"}
    assert body["default_source"] == "naver"
    assert body["version"] == "0.1.0"
    assert body["limits"] == {
        "max_compare": 5, "max_keywords_per_group": 5, "max_term_length": 50,
        "horizon_choices": {"D": [14, 30, 90], "W": [13, 26, 52], "M": [6, 12, 24], "Q": [4, 8], "Y": [3, 5]},
        "horizon_default": {"D": 30, "W": 26, "M": 12, "Q": 8, "Y": 5},
        "horizon_max": {"D": 365, "W": 104, "M": 36, "Q": 16, "Y": 10}}
    assert body["naver_ages"][0] == {"code": "1", "ko": "0–12세", "en": "0–12"}
    assert body["naver_ages"][-1] == {"code": "11", "ko": "60세 이상", "en": "60+"}
    assert body["retention"] == {"result_hours": 24, "share_days": 30, "upload_minutes": 30}
    assert body["sources"]["upload"]["max_mb"] == 10 and body["sources"]["upload"]["max_series"] == 20
    assert body["sources"]["sample"] == {"available": True}
    assert body["sources"]["wiki"] == {"available": True, "reason": None}


@pytest.mark.parametrize("count,state", [(79, "ok"), (80, "low"), (90, "blocked")])
def test_config_budget_states(make, count, state):
    app = make(naver_monthly_limit=100)
    for _ in range(count):
        app.naver_budget.reserve()
    naver = TestClient(app).get("/api/config").json()["sources"]["naver"]
    assert naver["budget_state"] == state
    if state == "blocked":
        assert (naver["available"], naver["reason"]) == (False, "budget_exhausted")
    else:
        assert (naver["available"], naver["reason"]) == (True, None)


def test_config_features_xlsx_ai(make, monkeypatch):
    body = TestClient(make()).get("/api/config").json()
    assert body["features"]["ai"] is True and body["features"]["ai_reason"] is None
    assert body["features"]["html_report"] is True
    assert body["features"]["xlsx_export"] == exports.xlsx_available()

    monkeypatch.setattr(exports, "xlsx_available", lambda: False)
    body = TestClient(make(ai=False)).get("/api/config").json()
    assert body["features"]["xlsx_export"] is False
    assert "xlsx" not in body["sources"]["upload"]["formats"]
    assert body["features"]["ai"] is False and body["features"]["ai_reason"] == "unconfigured"

    # TRENDAI_AI=off wins over an available LLM (make_test_app's own ``ai`` flag only picks the fake LLM)
    app = make()
    off = create_app(WebConfig.from_env(environ={}, data_dir=app.config.data_dir, static_dir=app.config.static_dir,
                                        ai="off"), fetchers=app.fetchers)
    try:
        body = TestClient(off).get("/api/config").json()
    finally:
        off.close(timeout=5)
    assert body["features"]["ai"] is False and body["features"]["ai_reason"] == "disabled"


def test_config_last_complete_dates(make):
    app = make(clock=FrozenClock(dt.datetime(2026, 11, 25, 0, 12, tzinfo=dt.timezone.utc)))
    periods = TestClient(app).get("/api/config").json()["periods"]
    expected = {"date": "2026-11-24", "week": "2026-11-22", "month": "2026-10-31"}
    assert periods["naver"] == {"earliest": "2016-01-01", "last_complete": expected}
    assert periods["wiki"] == {"earliest": "2016-01-01", "last_complete": expected}


def test_config_last_complete_kst_vs_utc(make):
    # 2026-11-30T16:00Z is already 2026-12-01 in KST: Naver's last complete month is November, Wikipedia's October
    app = make(clock=FrozenClock(dt.datetime(2026, 11, 30, 16, 0, tzinfo=dt.timezone.utc)))
    periods = TestClient(app).get("/api/config").json()["periods"]
    assert periods["naver"]["last_complete"]["month"] == "2026-11-30"
    assert periods["wiki"]["last_complete"]["month"] == "2026-10-31"


@pytest.mark.parametrize("env,name", [
    ({"TRENDAI_PORT": "abc"}, "TRENDAI_PORT"),
    ({"TRENDAI_PORT": "70000"}, "TRENDAI_PORT"),
    ({"TRENDAI_DEV": "maybe"}, "TRENDAI_DEV"),
    ({"TRENDAI_WORKERS": "0"}, "TRENDAI_WORKERS"),
    ({"TRENDAI_AI": "sometimes"}, "TRENDAI_AI"),
    ({"TRENDAI_LOG_LEVEL": "LOUD"}, "TRENDAI_LOG_LEVEL"),
    ({"TRENDAI_JOB_TIMEOUT": "-1"}, "TRENDAI_JOB_TIMEOUT"),
    ({"TRENDAI_PUBLIC_URL": "trendai.example"}, "TRENDAI_PUBLIC_URL"),
])
def test_webconfig_from_env_invalid_raises(env, name):
    with pytest.raises(ValueError, match=name):
        WebConfig.from_env(environ=env)


def test_webconfig_from_env_values_and_overrides():
    cfg = WebConfig.from_env(environ={"TRENDAI_PORT": "9000", "TRENDAI_DEV": "YES", "TRENDAI_NAVER_CACHE_DAYS": "40",
                                      "TRENDAI_PUBLIC_URL": "https://trendai.example/", "TRENDAI_AI": "Off"},
                             port="0", workers=3)
    assert (cfg.port, cfg.dev, cfg.naver_cache_days, cfg.workers, cfg.ai) == (0, True, 21, 3, "off")
    assert cfg.public_url == "https://trendai.example"
    assert WebConfig.from_env(environ={}) == WebConfig()
    with pytest.raises(TypeError):
        WebConfig.from_env(environ={}, no_such_setting=1)
    assert "KEY" not in repr(cfg)


def test_validate_public():
    with pytest.raises(ValueError, match="TRENDAI_CONTACT"):
        WebConfig(host="0.0.0.0").validate_public()
    with pytest.raises(ValueError):
        WebConfig(public_url="https://trendai.example").validate_public()
    WebConfig(host="0.0.0.0", contact="ops@example.com").validate_public()
    WebConfig(host="0.0.0.0", allow_no_contact=True).validate_public()
    for host in ("127.0.0.1", "127.8.9.10", "::1", "localhost"):
        assert not WebConfig(host=host).is_public()
        WebConfig(host=host).validate_public()
    assert WebConfig(host="example.org").is_public()


def test_make_test_app_hermetic(make, monkeypatch):
    monkeypatch.setenv("TRENDAI_WORKERS", "7")
    monkeypatch.setenv("NAVER_API_HUB_KEY", "x")
    monkeypatch.setenv("NAVER_API_HUB_KEY_ID", "y")
    monkeypatch.setenv("TRENDAI_AI", "off")
    app = make()
    assert app.config.workers == 2
    assert app.config.ai == "auto"
    assert isinstance(app.fetchers.naver, FakeNaver)


def test_ai_status(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert ai_status(WebConfig(ai="off")) == (False, "disabled")
    assert ai_status(WebConfig(ai="auto"), environ={}) == (False, "unconfigured")
    pytest.importorskip("anthropic")
    assert ai_status(WebConfig(ai="auto"), environ={"ANTHROPIC_API_KEY": "k"}) == (True, None)
    assert ai_status(WebConfig(ai="on"), environ={}) == (True, None)
