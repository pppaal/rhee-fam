"""The on-disk store (E.7): atomic 0600 writes, 0700 directories, expiry, the sweeper, size eviction and tolerance
of corrupt files."""

from __future__ import annotations

import datetime as dt
import json
import os
import stat
import time

import pytest

from trendai.web import viewmodel
from trendai.web.store import Store, utc_iso
from trendai.web.testing import FrozenClock, TestClient, analyze_and_wait, make_test_app, write_static_fixture

T0 = dt.datetime(2026, 11, 25, 0, 12, tzinfo=dt.timezone.utc)
RID = "R" * 22
SID = "S" * 22
KEY = "ab" + "0" * 62


@pytest.fixture
def clock():
    return FrozenClock(T0)


@pytest.fixture
def store(tmp_path, clock):
    s = Store(tmp_path / "data", clock)
    s.ensure()
    return s


def result_meta(clock, hours=24, schema=None):
    now = clock()
    return {"schema": viewmodel.WEB_VM_SCHEMA if schema is None else schema, "result_id": RID,
            "created_at": utc_iso(now), "expires_at": utc_iso(now + dt.timedelta(hours=hours))}


def mode(path) -> int:
    return stat.S_IMODE(os.stat(path).st_mode)


def test_atomic_write_and_modes(store, clock, tmp_path):
    root = tmp_path / "data"
    assert mode(root) == 0o700
    for sub in ("results", "shares", "cache", "cache/naver", "cache/wiki", "cache/results", "budget"):
        assert mode(root / sub) == 0o700, sub
    store.save_dir("results", RID, {"result.json": result_meta(clock), "vm-ko.json": {"series": []},
                                    "report-0.html": "<html></html>", "ai/0-ko.json": {"markdown": "x"}})
    store.cache_put("naver", KEY, {"key": KEY, "expires_at": utc_iso(clock() + dt.timedelta(hours=1))})
    store.budget_write("naver-2026-11", {"month": "2026-11", "count": 1})
    files = [os.path.join(d, f) for d, _ds, fs in os.walk(root) for f in fs]
    dirs = [os.path.join(d, x) for d, ds, _fs in os.walk(root) for x in ds]
    assert files and all(mode(f) == 0o600 for f in files)
    assert all(mode(d) == 0o700 for d in dirs)
    assert not [f for f in files if os.path.basename(f).startswith(".tmp-")]
    assert store.load_result(RID)["result_id"] == RID
    assert store.read_item_json("results", RID, "vm-ko.json") == {"series": []}
    assert store.read_ai("results", RID, 0, "ko") == {"markdown": "x"}
    assert json.loads((root / "cache" / "naver" / "ab" / f"{KEY}.json").read_text())["key"] == KEY


def test_paths_validated(store, clock):
    for bad in ("../x", "short", "R" * 21 + "/", "R" * 23):
        assert store.load_result(bad) is None
        with pytest.raises(ValueError):
            store.save_dir("results", bad, {"result.json": {}})
    with pytest.raises(ValueError):
        store.save_dir("results", RID, {"../evil.json": {}})
    with pytest.raises(ValueError):
        store.read_item_json("results", RID, "passwd")
    with pytest.raises(ValueError):
        store.cache_get("naver", "../../etc")
    with pytest.raises(ValueError):
        store.write_ai(RID, True, "ko", {})
    with pytest.raises(ValueError):
        store.write_ai(RID, 0, "../x", {})
    assert store.read_ai("results", RID, 0, "../x") is None


def test_lazy_expiry(store, clock):
    store.save_dir("results", RID, {"result.json": result_meta(clock, hours=1)})
    store.cache_put("wiki", KEY, {"key": KEY, "expires_at": utc_iso(clock() + dt.timedelta(minutes=30))})
    assert store.load_result(RID) is not None and store.cache_get("wiki", KEY) is not None
    clock.advance(31 * 60)
    assert store.cache_get("wiki", KEY) is None
    assert not os.path.exists(store._cache_path("wiki", KEY))
    clock.advance(30 * 60)
    assert store.load_result(RID) is None
    assert not os.path.exists(os.path.join(store.root, "results", RID))


def test_schema_mismatch_is_absent(store, clock):
    store.save_dir("results", RID, {"result.json": result_meta(clock, schema=viewmodel.WEB_VM_SCHEMA + 1)})
    assert store.load_result(RID) is None
    assert os.path.exists(os.path.join(store.root, "results", RID))     # not expired: left for the sweeper


def test_sweeper_removes_expired(store, clock):
    store.save_dir("results", RID, {"result.json": result_meta(clock, hours=1)})
    store.save_dir("results", "T" * 22, {"result.json": {**result_meta(clock, hours=48), "result_id": "T" * 22}})
    store.save_dir("shares", SID, {"share.json": {"schema": viewmodel.WEB_VM_SCHEMA, "share_id": SID,
                                                  "expires_at": utc_iso(clock() + dt.timedelta(hours=2))}})
    store.cache_put("naver", KEY, {"key": KEY, "expires_at": utc_iso(clock() + dt.timedelta(minutes=10))})
    store.cache_put("results", "cd" + "1" * 62, {"key": "x", "result_id": RID,
                                                 "expires_at": utc_iso(clock() + dt.timedelta(minutes=10))})
    store.budget_write("naver-2025-01", {"month": "2025-01", "count": 3})
    store.budget_write("ai-2026-11-24", {"day": "2026-11-24", "count": 3})
    clock.advance(3 * 3600)
    counts = store.sweep()
    assert counts["results"] == 1 and counts["shares"] == 1 and counts["cache"] == 2 and counts["budget"] == 1
    assert sorted(os.listdir(os.path.join(store.root, "results"))) == ["T" * 22]
    assert os.listdir(os.path.join(store.root, "shares")) == []
    assert os.listdir(os.path.join(store.root, "budget")) == ["ai-2026-11-24.json"]


def test_store_size_eviction(store, clock):
    ids = [c * 22 for c in "ABCDE"]
    for rid in ids:
        meta = {**result_meta(clock), "result_id": rid}
        store.save_dir("results", rid, {"result.json": meta, "report-0.html": "x" * 300_000})
        clock.advance(60)
    store.save_dir("shares", SID, {"share.json": {"schema": viewmodel.WEB_VM_SCHEMA, "share_id": SID,
                                                  "expires_at": utc_iso(clock() + dt.timedelta(days=30))},
                                   "report-0.html": "y" * 300_000})
    counts = store.sweep(max_mb=1.0)               # 6 x 300 KB = 1.8 MB -> the 3 oldest go (900 KB < 0.9 MiB)
    assert counts["evicted"] == 3
    assert sorted(os.listdir(os.path.join(store.root, "results"))) == ["D" * 22, "E" * 22]   # the newest stay
    assert os.listdir(os.path.join(store.root, "shares")) == [SID]          # shares are never size-evicted
    assert store.size_bytes() <= 0.9 * 1024 * 1024


def test_corrupt_file_tolerated(store, clock):
    store.save_dir("results", RID, {"result.json": result_meta(clock)})
    path = os.path.join(store.root, "results", RID, "result.json")
    with open(path, "wb") as fh:
        fh.write(b"{not json")
    assert store.load_result(RID) is None
    cache = store._cache_path("naver", KEY)
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    with open(cache, "wb") as fh:
        fh.write(b"\xff\xfe")
    assert store.cache_get("naver", KEY) is None
    store.budget_write("naver-2026-11", {"month": "2026-11", "count": 1})
    with open(os.path.join(store.root, "budget", "naver-2026-11.json"), "w") as fh:
        fh.write("[]")
    assert store.budget_read("naver-2026-11") is None
    store.sweep()                                             # fresh unreadable files stay for an hour
    assert os.path.exists(path) and os.path.exists(cache)
    old = time.time() - 7200
    for p in (path, os.path.dirname(path), cache):
        os.utime(p, (old, old))
    store.sweep()
    assert not os.path.exists(os.path.dirname(path)) and not os.path.exists(cache)


def test_corrupt_result_over_http(tmp_path):
    app = make_test_app(tmp_path / "d", static_dir=write_static_fixture(tmp_path / "s"))
    try:
        c = TestClient(app)
        snap, env = analyze_and_wait(c, {"source": "sample", "series": [{"terms": ["growing"]}]})
        vm_path = os.path.join(app.config.data_dir, "results", env["result_id"], "vm-ko.json")
        with open(vm_path, "w") as fh:
            fh.write("{oops")
        resp = c.get(f"/api/results/{env['result_id']}")
        assert resp.status == 404 and resp.json()["error"]["code"] == "result_not_found"
        assert c.get(f"/api/results/{env['result_id']}?hl=en").status == 200
    finally:
        app.close(timeout=5)


def test_health_degraded_when_not_writable(tmp_path, monkeypatch):
    app = make_test_app(tmp_path / "d", static_dir=write_static_fixture(tmp_path / "s"))
    try:
        monkeypatch.setattr(app.store, "writable", lambda: False)
        resp = TestClient(app).get("/api/health")
        assert resp.status == 503
        assert resp.json()["status"] == "degraded" and resp.json()["store"] == {"writable": False}
    finally:
        app.close(timeout=5)
