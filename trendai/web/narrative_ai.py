"""``POST /api/narrative`` and the AI narrative job (spec C.11).

The job sends only the stored LLM payload (computed facts, ``results/<id>/llm-<k>.json``) to ``fetchers.llm``,
splits the caveats, stores the record as ``results/<id>/ai/<k>-<lang>.json`` and finishes with the C.6 narrative
body. Naver results never have a payload and are refused (``403 ai_naver_blocked``). Any failure -> job ``failed``
with ``502 ai_failed``; a job still running after :data:`AI_JOB_TIMEOUT_S` fails with ``ai_failed`` at its next check.
"""

from __future__ import annotations

import copy
import json
import logging
from typing import Any, Mapping, Optional

from . import viewmodel
from .errors import WebError
from .http import valid_id
from .jobs import Job, JobContext
from .log import log_exception, short_id
from .store import utc_iso

__all__ = ["start_narrative", "AI_JOB_TIMEOUT_S", "MAX_NAME_CHARS", "MAX_PAYLOAD_CHARS", "payload_for"]

LOG = logging.getLogger("trendai.web.jobs")

#: a narrative job still running after this many seconds fails with ``ai_failed`` at its next check (C.11)
AI_JOB_TIMEOUT_S = 120.0
#: the series name sent to the model (uploads: a column header of any length -- each call's cost and the room for
#: instructions hidden in a header must not grow with it)
MAX_NAME_CHARS = 80
#: payloads above this size (canonical JSON characters; a normal one is ~10 k) are refused before calling the model:
#: the operator's per-day call limit must also bound each call's cost
MAX_PAYLOAD_CHARS = 40_000


def payload_for(payload: Mapping[str, Any], vm: Mapping[str, Any]) -> dict:
    """The stored LLM payload as sent for one language: ``series.name`` is the VM's display name in that language
    (sample titles, E.6.1), cut to :data:`MAX_NAME_CHARS`."""
    out = copy.deepcopy(dict(payload))
    series = out.get("series")
    if isinstance(series, dict):
        name = " ".join(str(vm.get("name") or series.get("name") or "").split())
        series["name"] = name if len(name) <= MAX_NAME_CHARS else name[:MAX_NAME_CHARS - 1] + "…"
    return out


def _series_index(value: Any) -> Optional[int]:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return int(value)


def start_narrative(app: Any, body: Mapping[str, Any], *, client_key: str,
                    request_id: Optional[str]) -> tuple[int, Job]:
    """Run the C.11 checks in order and return ``(HTTP status, job)``: 202 for a new job, 200 for an AI narrative
    that already exists (a job that is already ``done``)."""
    if "share_id" in body:
        raise WebError("ai_readonly")
    lang = body.get("lang")
    if lang not in ("ko", "en") or not isinstance(lang, str):
        raise WebError("invalid_param", params={"param": "lang"})
    rid = body.get("result_id")
    if not valid_id(rid):
        raise WebError("result_not_found")
    result = app.store.load_result(rid)
    if result is None:
        raise WebError("result_not_found")
    k = _series_index(body.get("series"))
    available = (result.get("available") or {}).get("series") or []
    if k is None or not 0 <= k < len(available) or available[k] is None:
        raise WebError("not_found")
    if result.get("source_kind") == "naver":
        raise WebError("ai_naver_blocked")
    if not app.ai_enabled():
        raise WebError("ai_unavailable")
    existing = app.store.read_ai("results", rid, k, lang)
    if existing is not None:
        return 200, app.jobs.add_done("narrative", narrative=viewmodel.ai_job_narrative(existing),
                                      request_id=request_id)
    retry = app.limiter.take("ai", client_key, refill=app.config.ai_ip_per_hour)
    if retry is not None:
        raise WebError("ai_rate_limited", params={"limit": app.config.ai_ip_per_hour}, retry_after=retry)
    app.ai_budget.reserve()
    try:
        job = app.jobs.submit("narrative", _job_fn(app, rid, k, lang), timeout_s=AI_JOB_TIMEOUT_S,
                              timeout_code="ai_failed", request_id=request_id, client_key=client_key,
                              log_info={"source": result.get("source_kind", "-"), "series": 1})
    except WebError:
        app.ai_budget.release()
        raise
    return 202, job


def _job_fn(app: Any, rid: str, k: int, lang: str):
    def fn(ctx: JobContext) -> None:
        ctx.check()
        payload = app.store.read_item_json("results", rid, f"llm-{k}.json")
        vms = app.store.read_item_json("results", rid, f"vm-{lang}.json")
        if not isinstance(payload, dict) or not isinstance(vms, dict):
            raise WebError("ai_failed")
        vm = next((v for v in vms.get("series") or [] if v.get("index") == k), None)
        llm = app.fetchers.llm
        if vm is None or llm is None:
            raise WebError("ai_failed")
        payload = payload_for(payload, vm)
        size = len(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
        if size > MAX_PAYLOAD_CHARS:
            LOG.warning("job=%s AI payload of %d characters refused (limit %d)", short_id(ctx.job.id), size,
                        MAX_PAYLOAD_CHARS)
            raise WebError("ai_failed")
        try:
            markdown = llm(payload, lang)
        except Exception as exc:  # noqa: BLE001 - NarrativeUnavailable and anything else -> ai_failed (C.11)
            if type(exc).__name__ != "NarrativeUnavailable":
                log_exception(LOG, f"job={short_id(ctx.job.id)} AI narrative failed", exc, rid=ctx.job.request_id,
                              code="ai_failed")
            raise WebError("ai_failed") from None
        ctx.check()
        if not isinstance(markdown, str) or not markdown.strip():
            raise WebError("ai_failed")
        record = viewmodel.build_ai_record(markdown, vm, model=app.config.ai_model, created_at=utc_iso(app.clock()))
        try:
            app.store.write_ai(rid, k, lang, record)
        except FileNotFoundError:     # the result expired while the model was writing
            raise WebError("ai_failed") from None
        ctx.set_narrative(viewmodel.ai_job_narrative(record))

    return fn
