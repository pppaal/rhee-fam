"""JSON API handlers (spec C.1-C.11). Each handler takes ``(app, req, rid, **path_params)`` and returns a
:class:`~trendai.web.http.Response` or raises :class:`~trendai.web.errors.WebError`."""

from __future__ import annotations

import copy
import datetime as _dt
import hashlib
import hmac
import importlib.util
import json
import secrets
from typing import Any, Optional

from .. import __version__
from ..connectors import NAVER_EARLIEST, WIKI_EARLIEST, period_bounds
from . import analyze, exports, narrative_ai, texts, uploads, viewmodel
from .budget import kst_now
from .errors import WebError
from .http import Request, Response, json_response, valid_id
from .jobs import analyze_snapshot, narrative_snapshot
from .log import LOGGER, log_exception
from .store import utc_iso

__all__ = ["health", "get_config", "get_samples", "upload", "post_analyze", "get_job", "delete_job", "get_result",
           "post_share", "delete_share", "get_shared", "export_file", "export_values", "post_narrative",
           "build_samples", "HORIZON_CHOICES", "HORIZON_DEFAULT"]

HORIZON_CHOICES = {"D": [14, 30, 90], "W": [13, 26, 52], "M": [6, 12, 24], "Q": [4, 8], "Y": [3, 5]}
HORIZON_DEFAULT = {"D": 30, "W": 26, "M": 12, "Q": 8, "Y": 5}
_JSON_LIMIT = 65_536
_MULTIPART_OVERHEAD = 65_536
_UPLOAD_KINDS = ("upload", "google_trends_upload")


def _number(x: float) -> Any:
    return int(x) if float(x).is_integer() else float(x)


def _rate(app: Any, bucket: str, key: Any) -> None:
    retry = app.limiter.take(bucket, key)
    if retry is not None:
        raise WebError("rate_limited", params={"retry_after": retry}, retry_after=retry)


def _refuse_when_draining(app: Any) -> None:
    if app.draining:
        raise WebError("server_busy", retry_after=30)


# --------------------------------------------------------------------------- C.1 health

def health(app: Any, req: Request, rid: str) -> Response:
    writable = app.store.writable()
    draining = app.draining
    running, queued = app.jobs.counts()
    status = "draining" if draining else ("ok" if writable else "degraded")
    body = {"status": status, "version": __version__,
            "uptime_s": int(max(0.0, (app.clock() - app.started_at).total_seconds())),
            "draining": draining, "jobs": {"running": running, "queued": queued}, "store": {"writable": writable}}
    return json_response(body, 200 if status == "ok" else 503, cache="no-store")


# --------------------------------------------------------------------------- C.2 config

def _formats() -> list[str]:
    formats = ["csv", "tsv", "txt"]
    if exports.xlsx_available():
        formats.append("xlsx")
    if importlib.util.find_spec("xlrd") is not None:
        formats.append("xls")
    formats.append("json")
    return formats


def get_config(app: Any, req: Request, rid: str) -> Response:
    _rate(app, "read", req.client_key)
    configured = app.fetchers.naver is not None
    _count, _limit, budget_state = app.naver_budget.state()
    if not configured:
        naver = {"available": False, "reason": "unconfigured", "endpoint": None, "budget_state": budget_state}
    elif budget_state == "blocked":
        naver = {"available": False, "reason": "budget_exhausted", "endpoint": app.fetchers.naver_endpoint,
                 "budget_state": budget_state}
    else:
        naver = {"available": True, "reason": None, "endpoint": app.fetchers.naver_endpoint,
                 "budget_state": budget_state}
    today_kst = kst_now(app.clock).date()
    today_utc = app.clock().astimezone(_dt.timezone.utc).date()

    def last_complete(today: _dt.date, earliest: _dt.date) -> dict:
        return {unit: period_bounds("1y", unit, today, earliest=earliest)[1].isoformat()
                for unit in ("date", "week", "month")}

    ai_on = app.ai_enabled()
    body = {
        "version": __version__,
        "default_source": "naver" if naver["available"] else "wiki",
        "sources": {
            "naver": naver,
            "wiki": {"available": True, "reason": None},
            "upload": {"available": True, "max_mb": app.config.upload_max_mb,
                       "max_series": app.config.upload_max_series, "formats": _formats()},
            "sample": {"available": True},
        },
        "features": {"xlsx_export": bool(exports.xlsx_available()), "ai": ai_on,
                     "ai_reason": None if ai_on else app.ai_reason(), "html_report": True},
        "limits": {"max_compare": analyze.MAX_COMPARE, "max_keywords_per_group": analyze.MAX_KEYWORDS,
                   "max_term_length": analyze.MAX_TERM_LENGTH, "horizon_choices": HORIZON_CHOICES,
                   "horizon_default": HORIZON_DEFAULT, "horizon_max": dict(analyze.HORIZON_MAX)},
        "naver_ages": [{"code": code, "ko": ko, "en": en} for code, (ko, en) in texts.NAVER_AGE_LABELS.items()],
        "periods": {"naver": {"earliest": NAVER_EARLIEST.isoformat(),
                              "last_complete": last_complete(today_kst, NAVER_EARLIEST)},
                    "wiki": {"earliest": WIKI_EARLIEST.isoformat(),
                             "last_complete": last_complete(today_utc, WIKI_EARLIEST)}},
        "retention": {"result_hours": _number(app.config.result_ttl_h), "share_days": _number(app.config.share_ttl_d),
                      "upload_minutes": int(uploads.UPLOAD_TTL_S // 60)},
    }
    return json_response(body, cache="no-cache")


# --------------------------------------------------------------------------- C.3 samples

def build_samples() -> dict[str, list[dict]]:
    """The C.3 list in both languages, computed once at app start."""
    from ..samples import DESCRIPTIONS, EXPECTED_PHASE, list_samples, sample

    out: dict[str, list[dict]] = {"ko": [], "en": []}
    for name in list_samples():
        s = sample(name)
        values = s.to_numpy(dtype=float)
        spark = [float(f"{float(values[i]):.3g}") for i in range(1, len(values), 2)]
        start, end = s.index[0], s.index[-1]
        common = {"name": name, "expected_phase": EXPECTED_PHASE[name][0], "freq": "M",
                  "start": f"{start.year:04d}-{start.month:02d}-{start.day:02d}",
                  "end": f"{end.year:04d}-{end.month:02d}-{end.day:02d}", "spark": spark, "synthetic": True}
        out["ko"].append({"name": name, "title": texts.SAMPLE_TITLES[name], "description": DESCRIPTIONS[name],
                          **{k: v for k, v in common.items() if k != "name"}})
        out["en"].append({"name": name, "title": texts.SAMPLE_TITLES_EN[name],
                          "description": texts.SAMPLE_DESCRIPTIONS_EN[name],
                          **{k: v for k, v in common.items() if k != "name"}})
    return out


def get_samples(app: Any, req: Request, rid: str) -> Response:
    _rate(app, "read", req.client_key)
    return json_response({"samples": app.samples[req.hl]}, cache="public, max-age=3600")


# --------------------------------------------------------------------------- C.4 upload

def upload(app: Any, req: Request, rid: str) -> Response:
    _refuse_when_draining(app)
    _rate(app, "upload", req.client_key)
    cfg = app.config
    ctype = req.content_type
    if ctype not in ("multipart/form-data", "application/json"):
        raise WebError("unsupported_media_type")
    limit = cfg.upload_max_bytes + _MULTIPART_OVERHEAD
    # 411 / 413 before reading (C.0); the body is read before a parse slot is taken, so a slow client cannot hold
    # one of the two slots while its bytes trickle in (E.17 bounds the *parse*: multipart decode -> detection)
    body = req.read_body(limit, max_mb=cfg.upload_max_mb)
    if not app.parse_slots.acquire(blocking=False):
        raise WebError("server_busy", retry_after=5)
    try:
        if ctype == "multipart/form-data":
            filename, raw = uploads.parse_multipart(req.header("Content-Type"), body)
        else:
            try:
                obj = json.loads(body.decode("utf-8"))
            except (UnicodeDecodeError, ValueError):
                raise WebError("bad_request") from None
            text = obj.get("text") if isinstance(obj, dict) else None
            name = obj.get("filename", "paste.tsv") if isinstance(obj, dict) else None
            if not isinstance(text, str) or not text or not isinstance(name, (str, type(None))):
                raise WebError("bad_request")
            raw = text.encode("utf-8")
            filename = name or "paste.tsv"
        created = app.clock()
        try:
            entry, result = uploads.analyse_upload(raw, filename, hl=req.hl, max_bytes=cfg.upload_max_bytes,
                                                   max_mb=cfg.upload_max_mb, created_at=created.timestamp(),
                                                   request_id=rid)
        except WebError:
            raise
        except Exception as exc:  # noqa: BLE001 - never a 500 with input-quoting text for an odd file (E.11)
            log_exception(LOGGER, "upload parse failed", exc, rid=rid, code="upload_unreadable")
            raise WebError("upload_unreadable", params={"reason": "file_unreadable"}) from None
        del raw, body
    finally:
        app.parse_slots.release()
    upload_id = secrets.token_urlsafe(16)
    app.uploads.put(upload_id, entry)
    expires = created + _dt.timedelta(seconds=app.uploads.ttl_s)
    return json_response({"upload_id": upload_id, "expires_at": utc_iso(expires), **result}, cache="no-store")


# --------------------------------------------------------------------------- C.5 analyze

def _accepted(job: Any, *, status: str, cached: bool, joined: bool, position: int, code: int) -> Response:
    return json_response({"job_id": job if isinstance(job, str) else job.id, "status": status, "cached": cached,
                          "joined": joined, "queue_position": position,
                          "poll_url": f"/api/jobs/{job if isinstance(job, str) else job.id}",
                          "result_url": f"/api/results/{job if isinstance(job, str) else job.id}"},
                         code, cache="no-store")


def post_analyze(app: Any, req: Request, rid: str) -> Response:
    _refuse_when_draining(app)
    body = req.json_body(_JSON_LIMIT)
    now = app.clock()
    q = analyze.normalize(body, today_kst=kst_now(app.clock).date(),
                          today_utc=now.astimezone(_dt.timezone.utc).date(), upload_cache=app.uploads,
                          max_series=app.config.upload_max_series)
    client = req.client_key
    _rate(app, "analyze", client)
    key = analyze.result_cache_key(q)
    if key is not None:
        cached = app.store.cache_get("results", key)
        if cached is not None:
            rid_cached = cached.get("result_id")
            if valid_id(rid_cached) and app.store.load_result(rid_cached) is not None:
                return _accepted(rid_cached, status="done", cached=True, joined=False, position=0, code=200)
            app.store.cache_delete("results", key)
        joined = app.jobs.join(key)
        if joined is not None:
            return _accepted(joined, status=joined.status, cached=False, joined=True,
                             position=app.jobs.queue_position(joined), code=202)
    if q.source == "naver":
        if app.fetchers.naver is None:
            raise WebError("naver_unconfigured")
        if app.store.cache_get("naver", analyze.naver_key(q)) is None:
            _c, _l, state = app.naver_budget.state()
            if state == "blocked":
                raise WebError("naver_budget_exhausted")
            refill = 5 if state != "ok" else app.config.naver_ip_per_hour
            retry = app.limiter.take("naver", client, refill=refill)
            if retry is not None:
                raise WebError("naver_rate_limited", params={"limit": refill}, retry_after=retry)
    job_id = app.jobs.new_id()
    job, joined = app.jobs.submit_or_join(
        "analyze", analyze.run_analyze_job(app, q, job_id, rid), cache_key=key, job_id=job_id, request_id=rid,
        log_info={"source": q.source, "series": q.size, "naver": "-"}, view=q.view, series_total=q.size)
    return _accepted(job, status=job.status, cached=False, joined=joined, position=app.jobs.queue_position(job),
                     code=202)


# --------------------------------------------------------------------------- C.6 jobs

def _since(req: Request) -> int:
    try:
        return max(0, int(req.arg("since", "0") or 0))
    except ValueError:
        return 0


def _stored_snapshot(app: Any, result_id: str, hl: str) -> Optional[dict]:
    result = app.store.load_result(result_id)
    vms = app.store.read_item_json("results", result_id, f"vm-{hl}.json") if result is not None else None
    if result is None or not isinstance(vms, dict):
        return None
    total = len((result.get("available") or {}).get("series") or [])
    return {"job_id": result_id, "kind": "analyze", "status": "done", "stage": "done", "series_total": total,
            "series_done": total, "current": None,
            "series": [{"index": v["index"], "status": "done", "vm": v} for v in vms.get("series") or []],
            "failed": (result.get("failed") or {}).get(hl, []), "notices": vms.get("notices") or [],
            "compare": vms.get("compare") if result.get("view") == "compare" else None,
            "view": result.get("view", "detail"), "error": None, "result_url": f"/api/results/{result_id}",
            "elapsed_ms": 0, "poll_after_ms": None}


def get_job(app: Any, req: Request, rid: str, id: str) -> Response:   # noqa: A002 - route parameter name
    retry = app.limiter.take("poll", (req.client_key, id))
    if retry is not None:
        raise WebError("rate_limited", params={"retry_after": retry}, retry_after=retry)
    job = app.jobs.get(id)
    now = app.clock().timestamp()
    if job is None:
        snap = _stored_snapshot(app, id, req.hl)
        if snap is None:
            raise WebError("job_not_found")
        return json_response(snap, cache="no-store")
    if job.kind == "narrative":
        return json_response(narrative_snapshot(job, now=now, naver_available=app.naver_available()),
                             cache="no-store")
    return json_response(analyze_snapshot(job, since=_since(req), hl=req.hl, now=now,
                                          naver_available=app.naver_available()), cache="no-store")


def delete_job(app: Any, req: Request, rid: str, id: str) -> Response:   # noqa: A002
    res = app.jobs.unsubscribe(id)
    if res is None:
        if app.store.load_result(id) is not None:
            return json_response({"job_id": id, "status": "done"}, 200, cache="no-store")
        raise WebError("job_not_found")
    job, status, terminal = res
    return json_response({"job_id": job.id, "status": status}, 200 if terminal else 202, cache="no-store")


# --------------------------------------------------------------------------- C.7 / C.9 results and shares

def _ai_flags(vm: dict, source_kind: str, ai_on: bool) -> dict:
    """Bring ``narrative.ai_allowed`` in line with the AI feature of this process (D.2.9: not Naver and
    ``features.ai``); Naver stays blocked whatever happens."""
    nar = vm.get("narrative") or {}
    if source_kind == "naver" or nar.get("ai_block_reason") == "naver_terms":
        return vm
    lang = vm.get("lang", "ko")
    if ai_on and not nar.get("ai_allowed"):
        nar.update(ai_allowed=True, ai_block_reason=None, ai_block_text=None)
    elif not ai_on and nar.get("ai_allowed"):
        from ..messages import render_web

        nar.update(ai_allowed=False, ai_block_reason="ai_unavailable",
                   ai_block_text=render_web("ai_unavailable", {}, "en" if lang == "en" else "ko"))
    return vm


def _envelope(app: Any, area: str, item_id: str, result: dict, hl: str, *, shared: Optional[dict]) -> dict:
    vms = app.store.read_item_json(area, item_id, f"vm-{hl}.json")
    if not isinstance(vms, dict):
        raise WebError("share_not_found" if shared else "result_not_found")
    base = f"/api/export/s/{item_id}" if shared else f"/api/export/{item_id}"
    available = result.get("available") or {}
    flags_series = available.get("series") or []
    ai_on = app.ai_enabled()
    series = []
    for vm in vms.get("series") or []:
        k = int(vm.get("index", 0))
        out = viewmodel.attach_ai(vm, app.store.read_ai(area, item_id, k, hl))
        if shared:
            out = viewmodel.make_shared_vm(out)
        else:
            out = _ai_flags(out, result.get("source_kind", ""), ai_on)
        flags = flags_series[k] if k < len(flags_series) and flags_series[k] else {}
        out["exports"] = viewmodel.link_exports(out.get("exports") or {}, flags, base, k)
        series.append(out)
    return {
        "result_id": None if shared else item_id,
        "created_at": result.get("created_at"),
        "expires_at": shared["expires_at"] if shared else result.get("expires_at"),
        "readonly": bool(shared),
        "view": result.get("view", "detail"),
        "query": copy.deepcopy(result.get("query") or {}),
        "series": series,
        "failed": copy.deepcopy((result.get("failed") or {}).get(hl, [])),
        "compare": vms.get("compare"),
        "notices": vms.get("notices") or [],
        "exports": viewmodel.link_exports(vms.get("exports") or {}, available.get("result") or {}, base),
    }


def get_result(app: Any, req: Request, rid: str, id: str) -> Response:   # noqa: A002
    _rate(app, "read", req.client_key)
    result = app.store.load_result(id)
    if result is None:
        raise WebError("result_not_found")
    return json_response(_envelope(app, "results", id, result, req.hl, shared=None), cache="no-store")


def post_share(app: Any, req: Request, rid: str) -> Response:
    body = req.json_body(_JSON_LIMIT)
    _rate(app, "share", req.client_key)
    result_id = body.get("result_id")
    if not valid_id(result_id):
        raise WebError("result_not_found")
    result = app.store.load_result(result_id)
    if result is None:
        raise WebError("result_not_found")
    if result.get("source_kind") not in _UPLOAD_KINDS:
        raise WebError("share_not_needed")
    share_id = secrets.token_urlsafe(16)
    token = secrets.token_urlsafe(16)
    now = app.clock()
    expires = now + _dt.timedelta(seconds=app.config.share_ttl_s)
    snapshot = {k: v for k, v in result.items() if k != "result_id"}
    files: dict[str, Any] = {}
    for name, content in app.store.item_files("results", result_id):
        if name.startswith("llm-") or name == "result.json":
            continue       # no LLM payloads in a share (C.8); result.json lives inside share.json
        files[name] = content
    for name, content in app.store.ai_files(result_id):
        files[name] = content
    files["share.json"] = {"schema": result.get("schema"), "share_id": share_id, "created_at": utc_iso(now),
                           "expires_at": utc_iso(expires),
                           "delete_token_sha256": hashlib.sha256(token.encode("ascii")).hexdigest(),
                           "result": snapshot}
    app.store.save_dir("shares", share_id, files)
    return json_response({"share_id": share_id, "url": f"/s/{share_id}", "expires_at": utc_iso(expires),
                          "delete_token": token}, 201, cache="no-store")


def delete_share(app: Any, req: Request, rid: str, sid: str) -> Response:
    _rate(app, "share", req.client_key)
    share = app.store.load_share(sid)
    if share is None:
        raise WebError("share_not_found")
    token = req.header("X-Trendai-Delete-Token").strip()
    given = hashlib.sha256(token.encode("utf-8", "replace")).hexdigest()
    stored = str(share.get("delete_token_sha256") or "")
    if not token or not stored or not hmac.compare_digest(given, stored):
        raise WebError("share_forbidden")
    app.store.delete_item("shares", sid)
    return Response(204, b"", [("Cache-Control", "no-store")])


def get_shared(app: Any, req: Request, rid: str, sid: str) -> Response:
    _rate(app, "read", req.client_key)
    share = app.store.load_share(sid)
    if share is None or not isinstance(share.get("result"), dict):
        raise WebError("share_not_found")
    shared = {"share_id": sid, "created_at": share.get("created_at"), "expires_at": share.get("expires_at")}
    body = _envelope(app, "shares", sid, share["result"], req.hl, shared=shared)
    body["shared"] = shared
    return json_response(body, cache="private, no-store", headers=[("X-Robots-Tag", "noindex, nofollow")])


# --------------------------------------------------------------------------- C.10 exports

def _export_source(app: Any, id: Optional[str], sid: Optional[str]) -> tuple[str, str, dict]:   # noqa: A002
    if sid is not None:
        share = app.store.load_share(sid)
        if share is None or not isinstance(share.get("result"), dict):
            raise WebError("share_not_found")
        return "shares", sid, share["result"]
    result = app.store.load_result(id)
    if result is None:
        raise WebError("result_not_found")
    return "results", id, result


def _unavailable(result: dict, fmt: str) -> WebError:
    if result.get("source_kind") == "naver" and fmt in ("html", "values_csv"):
        return WebError("export_naver_raw")
    if fmt == "xlsx":
        return WebError("export_xlsx_unavailable")
    return WebError("not_found")


def _file_response(f: Any) -> Response:
    return Response(200, f.body, list(f.headers()), compressible=False, replaces_csp=f.fmt == "html")


def _vm_doc(app: Any, area: str, item: str, hl: str) -> dict:
    doc = app.store.read_item_json(area, item, f"vm-{hl}.json")
    if not isinstance(doc, dict):
        raise WebError("not_found")
    return doc


def _result_name(vms: list) -> str:
    names = [str(v.get("name", "")) for v in sorted(vms, key=lambda v: int(v.get("index", 0)))]
    return " vs ".join(names)[:80]


def export_file(app: Any, req: Request, rid: str, fmt: str, id: Optional[str] = None,   # noqa: A002
                sid: Optional[str] = None, k: Optional[str] = None) -> Response:
    _rate(app, "export", req.client_key)
    area, item, result = _export_source(app, id, sid)
    hl = req.hl
    available = result.get("available") or {}
    try:
        if k is not None:
            index = int(k)
            flags_list = available.get("series") or []
            flags = flags_list[index] if index < len(flags_list) else None
            if not flags:
                raise WebError("not_found")
            if not flags.get(fmt):
                raise _unavailable(result, fmt)
            doc = _vm_doc(app, area, item, hl)
            vm = next((v for v in doc.get("series") or [] if int(v.get("index", -1)) == index), None)
            if vm is None:
                raise WebError("not_found")
            if fmt == "csv":
                return _file_response(exports.series_csv(vm, file_id=item))
            if fmt == "xlsx":
                if not exports.xlsx_available():
                    raise WebError("export_xlsx_unavailable")
                return _file_response(exports.series_xlsx(vm, file_id=item))
            if fmt == "html":
                names = (result.get("html") or {}).get("series") or []
                name = names[index] if index < len(names) else None
                html = app.store.read_item_text(area, item, name) if name else None
                if html is None:
                    raise WebError("not_found")
                return _file_response(exports.html_export(html, file_id=item, k=index, display_name=vm.get("name", "")))
            store_doc = app.store.read_item_json(area, item, f"export-{hl}.json") or {}
            return _file_response(exports.json_export(store_doc, file_id=item, k=index,
                                                      display_name=vm.get("name", "")))
        flags = available.get("result") or {}
        if not flags.get(fmt):
            raise _unavailable(result, fmt)
        doc = _vm_doc(app, area, item, hl)
        vms = list(doc.get("series") or [])
        compare = doc.get("compare")
        if fmt == "csv":
            return _file_response(exports.result_csv(vms, compare, result.get("ranking_rows") or [], file_id=item))
        if fmt == "xlsx":
            if not exports.xlsx_available():
                raise WebError("export_xlsx_unavailable")
            return _file_response(exports.result_xlsx(vms, compare, file_id=item))
        if fmt == "html":
            html_info = result.get("html") or {}
            name = html_info.get("compare") or next((n for n in html_info.get("series") or [] if n), None)
            html = app.store.read_item_text(area, item, name) if name else None
            if html is None:
                raise WebError("not_found")
            return _file_response(exports.html_export(html, file_id=item, k=None, display_name=_result_name(vms)))
        store_doc = app.store.read_item_json(area, item, f"export-{hl}.json") or {}
        return _file_response(exports.json_export(store_doc, file_id=item, k=None, display_name=_result_name(vms)))
    except exports.ExportUnavailable as exc:
        raise WebError(exc.code, exc.status) from None


def export_values(app: Any, req: Request, rid: str, id: Optional[str] = None,   # noqa: A002
                  sid: Optional[str] = None) -> Response:
    _rate(app, "export", req.client_key)
    area, item, result = _export_source(app, id, sid)
    if result.get("view") != "compare":
        raise WebError("not_found")
    if not (result.get("available") or {}).get("result", {}).get("values_csv"):
        raise _unavailable(result, "values_csv")
    doc = _vm_doc(app, area, item, req.hl)
    try:
        return _file_response(exports.values_csv(list(doc.get("series") or []), doc.get("compare"), file_id=item))
    except exports.ExportUnavailable as exc:
        raise WebError(exc.code, exc.status) from None


# --------------------------------------------------------------------------- C.11 narrative

def post_narrative(app: Any, req: Request, rid: str) -> Response:
    _refuse_when_draining(app)
    body = req.json_body(_JSON_LIMIT)
    status, job = narrative_ai.start_narrative(app, body, client_key=req.client_key, request_id=rid)
    return json_response({"job_id": job.id, "status": job.status, "poll_url": f"/api/jobs/{job.id}"}, status,
                         cache="no-store")
