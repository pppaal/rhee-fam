"""Asynchronous jobs (spec E.5, C.5, C.6): the registry, two bounded executors, in-flight joining and cancellation.

* ``analyze`` jobs run on ``ThreadPoolExecutor(max_workers=workers, thread_name_prefix="trendai-job")`` with
  ``workers + max_queued`` slots; ``narrative`` jobs on a pool of 2 with 2 + 4 slots. A full pool answers
  ``503 server_busy`` (``Retry-After: 10``); a draining manager ``503 server_busy`` (``Retry-After: 30``).
* :meth:`JobManager.join` lets identical queued/running analyze jobs share one job (subscriber count, C.5/C.6).
* The job function gets a :class:`JobContext`; ``check()`` (called at every stage boundary, including from
  ``pipeline.run``'s ``on_stage`` hook) raises :class:`JobCancelled` or :class:`JobTimeout`.
* Terminal jobs stay in memory 10 minutes; the result itself is on disk.
"""

from __future__ import annotations

import copy
import datetime as _dt
import itertools
import logging
import secrets
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .errors import WebError, error_object
from .log import JOBS_LOGGER, log_exception, short_id

__all__ = ["Job", "JobContext", "JobManager", "JobCancelled", "JobTimeout", "STEP_MAP", "TERMINAL",
           "analyze_snapshot", "narrative_snapshot", "POLL_AFTER_MS"]

LOG = logging.getLogger("trendai.web.jobs")

TERMINAL = ("done", "failed", "cancelled")
#: ``pipeline.run``'s ``on_stage`` names -> ``current.step`` (C.6)
STEP_MAP = {"prepare": "analyze", "analyze": "analyze", "forecast": "forecast", "outlook": "forecast",
            "narrative": "narrative"}
POLL_AFTER_MS = 500
_KEEP_TERMINAL_S = 600.0


class JobCancelled(Exception):
    """The job's last subscriber cancelled it (raised by :meth:`JobContext.check`)."""


class JobTimeout(Exception):
    """The job ran past its time limit (raised by :meth:`JobContext.check`)."""


@dataclass(eq=False)
class Job:
    id: str
    kind: str                                   # 'analyze' | 'narrative'
    created: float
    status: str = "queued"                      # queued | running | done | failed | cancelled
    stage: str = "queued"                       # queued | data | series | finish | done
    started: Optional[float] = None
    finished: Optional[float] = None
    series_total: int = 0
    series_done: int = 0
    current: Optional[dict] = None              # {'index', 'name', 'step'}
    vms: list = field(default_factory=list)     # finished series: {'index', 'vm': {'ko': ..., 'en': ...}}
    failed: list = field(default_factory=list)  # failed entries (C.6; language independent)
    notices: dict = field(default_factory=lambda: {"ko": [], "en": []})
    compare: Optional[dict] = None              # {'ko': ..., 'en': ...}
    error: Optional[WebError] = None
    cancel_requested: bool = False
    subscribers: int = 1                        # creator + C.5 joins - DELETEs; cancel only at 0 (C.6)
    cache_key: Optional[str] = None             # result-cache key (None for uploads and narrative jobs)
    view: str = "detail"
    narrative: Optional[dict] = None            # narrative jobs: the C.6 ``narrative`` body when done
    request_id: Optional[str] = None
    timeout_s: float = 180.0
    timeout_code: str = "job_timeout"
    log_info: dict = field(default_factory=dict)   # source, series count, naver fresh|cache (E.11 job line)
    seq: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def terminal(self) -> bool:
        return self.status in TERMINAL


class JobContext:
    """What a job function may do with its job (E.5)."""

    def __init__(self, job: Job, manager: "JobManager") -> None:
        self.job = job
        self._manager = manager

    def now(self) -> float:
        return self._manager.now()

    def check(self) -> None:
        """Raise :class:`JobCancelled` when cancellation was requested, :class:`JobTimeout` past the time limit."""
        job = self.job
        if job.cancel_requested:
            raise JobCancelled()
        if job.started is not None and self.now() - job.started > job.timeout_s:
            raise JobTimeout()

    def set_stage(self, stage: str, current: Optional[dict] = None) -> None:
        with self.job.lock:
            self.job.stage = stage
            self.job.current = dict(current) if current is not None else None

    def set_step(self, step: str) -> None:
        """Map an ``on_stage`` name to ``current.step`` (C.6) under the job lock."""
        mapped = STEP_MAP.get(step)
        if mapped is None:
            return
        with self.job.lock:
            if self.job.current is not None:
                self.job.current = {**self.job.current, "step": mapped}

    def set_total(self, total: int, view: str) -> None:
        with self.job.lock:
            self.job.series_total = int(total)
            self.job.view = view

    def add_series(self, index: int, vm_by_lang: dict) -> None:
        with self.job.lock:
            self.job.vms.append({"index": int(index), "vm": vm_by_lang})
            self.job.vms.sort(key=lambda e: e["index"])

    def add_failed(self, entry: dict) -> None:
        with self.job.lock:
            self.job.failed.append(entry)
            self.job.failed.sort(key=lambda e: e["index"])

    def series_finished(self) -> None:
        with self.job.lock:
            self.job.series_done += 1

    def set_notices(self, notices: dict) -> None:
        with self.job.lock:
            self.job.notices = {"ko": list(notices.get("ko", [])), "en": list(notices.get("en", []))}

    def set_final(self, vms: list, compare: Optional[dict]) -> None:
        with self.job.lock:
            self.job.vms = sorted(vms, key=lambda e: e["index"])
            self.job.compare = compare

    def set_narrative(self, body: dict) -> None:
        with self.job.lock:
            self.job.narrative = body


class _Pool:
    def __init__(self, kind: str, workers: int, max_queued: int, prefix: str) -> None:
        self.kind = kind
        self.workers = int(workers)
        self.capacity = int(workers) + int(max_queued)
        self.executor = ThreadPoolExecutor(max_workers=self.workers, thread_name_prefix=prefix)


class JobManager:
    """The in-memory job registry (one per app)."""

    def __init__(self, clock: Callable[[], _dt.datetime], *, workers: int, max_queued: int, timeout_s: float,
                 ai_workers: int = 2, ai_queued: int = 4) -> None:
        self._clock = clock
        self._lock = threading.Lock()
        self._jobs: dict[str, Job] = {}
        self._inflight: dict[str, str] = {}
        self._seq = itertools.count()
        self._closed = False
        self.timeout_s = float(timeout_s)
        self._pools = {"analyze": _Pool("analyze", workers, max_queued, "trendai-job"),
                       "narrative": _Pool("narrative", ai_workers, ai_queued, "trendai-ai")}

    def now(self) -> float:
        return self._clock().timestamp()

    # ------------------------------------------------------------------ registry
    def new_id(self) -> str:
        while True:
            jid = secrets.token_urlsafe(16)
            with self._lock:
                if jid not in self._jobs:
                    return jid

    def _active(self, kind: str) -> list[Job]:
        return [j for j in self._jobs.values() if j.kind == kind and j.status in ("queued", "running")]

    def counts(self, kind: Optional[str] = None) -> tuple[int, int]:
        """``(running, queued)`` over one kind or all jobs."""
        with self._lock:
            jobs = [j for j in self._jobs.values() if kind is None or j.kind == kind]
            return (sum(1 for j in jobs if j.status == "running"), sum(1 for j in jobs if j.status == "queued"))

    def check_capacity(self, kind: str) -> None:
        """``WebError(server_busy)`` when a new job of ``kind`` could not be queued right now."""
        with self._lock:
            self._check_capacity_locked(kind)

    def _check_capacity_locked(self, kind: str) -> None:
        if self._closed:
            raise WebError("server_busy", retry_after=30)
        if len(self._active(kind)) >= self._pools[kind].capacity:
            raise WebError("server_busy", retry_after=10)

    def submit(self, kind: str, fn: Callable[[JobContext], Any], *, cache_key: Optional[str] = None,
               job_id: Optional[str] = None, timeout_s: Optional[float] = None, timeout_code: str = "job_timeout",
               request_id: Optional[str] = None, log_info: Optional[dict] = None, **attrs: Any) -> Job:
        """Register and queue a job; ``WebError(server_busy)`` when the pool is full or the app is draining."""
        with self._lock:
            return self._submit_locked(kind, fn, cache_key=cache_key, job_id=job_id, timeout_s=timeout_s,
                                       timeout_code=timeout_code, request_id=request_id, log_info=log_info, **attrs)

    def _submit_locked(self, kind: str, fn: Callable[[JobContext], Any], *, cache_key: Optional[str],
                       job_id: Optional[str], timeout_s: Optional[float], timeout_code: str,
                       request_id: Optional[str], log_info: Optional[dict], **attrs: Any) -> Job:
        self._check_capacity_locked(kind)
        jid = job_id or secrets.token_urlsafe(16)
        job = Job(id=jid, kind=kind, created=self.now(), cache_key=cache_key, request_id=request_id,
                  timeout_s=float(timeout_s if timeout_s is not None else self.timeout_s),
                  timeout_code=timeout_code, log_info=dict(log_info or {}), seq=next(self._seq), **attrs)
        self._jobs[jid] = job
        if cache_key is not None:
            self._inflight[cache_key] = jid
        self._pools[kind].executor.submit(self._run, job, fn)
        return job

    def submit_or_join(self, kind: str, fn: Callable[[JobContext], Any], *, cache_key: Optional[str] = None,
                       job_id: Optional[str] = None, timeout_s: Optional[float] = None,
                       timeout_code: str = "job_timeout", request_id: Optional[str] = None,
                       log_info: Optional[dict] = None, **attrs: Any) -> tuple[Job, bool]:
        """:meth:`submit`, unless an identical job (same result-cache key) is queued or running: then that job is
        joined (subscriber count + 1) and ``(job, True)`` is returned. One lock covers the check and the submit, so
        two simultaneous identical requests always share one job (C.5)."""
        with self._lock:
            jid = self._inflight.get(cache_key) if cache_key is not None else None
            job = self._jobs.get(jid) if jid else None
            if job is not None:
                with job.lock:
                    if job.status in ("queued", "running") and not job.cancel_requested:
                        job.subscribers += 1
                        return job, True
            return self._submit_locked(kind, fn, cache_key=cache_key, job_id=job_id, timeout_s=timeout_s,
                                       timeout_code=timeout_code, request_id=request_id, log_info=log_info,
                                       **attrs), False

    def add_done(self, kind: str, *, narrative: Optional[dict] = None, **attrs: Any) -> Job:
        """Register a job that is already done (an existing AI narrative, C.11)."""
        now = self.now()
        job = Job(id=self.new_id(), kind=kind, created=now, status="done", stage="done", started=now, finished=now,
                  narrative=narrative, seq=next(self._seq), **attrs)
        with self._lock:
            self._jobs[job.id] = job
        return job

    def join(self, cache_key: str) -> Optional[Job]:
        """The queued/running analyze job with this result-cache key (subscriber count + 1), or ``None``."""
        with self._lock:
            jid = self._inflight.get(cache_key)
            job = self._jobs.get(jid) if jid else None
            if job is None:
                return None
            with job.lock:
                if job.status not in ("queued", "running") or job.cancel_requested:
                    return None
                job.subscribers += 1
            return job

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is not None and job.terminal and job.finished is not None \
                and self.now() - job.finished > _KEEP_TERMINAL_S:
            return None
        return job

    def queue_position(self, job: Job) -> int:
        """Jobs ahead of ``job`` (0 = starts now or already running, C.5)."""
        with self._lock:
            if job.status != "queued":
                return 0
            pool = self._pools[job.kind]
            running = sum(1 for j in self._active(job.kind) if j.status == "running")
            ahead = sum(1 for j in self._active(job.kind) if j.status == "queued" and j.seq < job.seq)
            free = max(0, pool.workers - running)
            return 0 if ahead < free else ahead - free + 1

    def unsubscribe(self, job_id: str) -> Optional[tuple[Job, str, bool]]:
        """``DELETE /api/jobs/<id>`` (C.6): ``(job, status to report, terminal_before)`` or ``None`` if unknown.
        Cancels only when the last subscriber leaves: queued -> ``cancelled`` now; running -> ``cancelling``."""
        job = self.get(job_id)
        if job is None:
            return None
        with job.lock:
            if job.terminal:
                return job, job.status, True
            job.subscribers -= 1
            if job.subscribers > 0:
                return job, job.status, False
            job.cancel_requested = True
            queued = job.status == "queued"
        if queued:
            self._finish(job, "cancelled")
        return job, "cancelling", False

    # ------------------------------------------------------------------ running
    def _run(self, job: Job, fn: Callable[[JobContext], Any]) -> None:
        with job.lock:
            if job.status != "queued":       # cancelled while queued
                return
            job.status = "running"
            job.started = self.now()
        ctx = JobContext(job, self)
        status = "done"
        try:
            if job.cancel_requested:
                raise JobCancelled()
            fn(ctx)
        except JobCancelled:
            status = "cancelled"
        except JobTimeout:
            status = "failed"
            job.error = WebError(job.timeout_code)
        except WebError as err:
            status = "failed"
            job.error = err
        except Exception as exc:  # noqa: BLE001 - a job must always end in a terminal state
            status = "failed"
            job.error = WebError("internal_error")
            log_exception(LOG, f"job={short_id(job.id)} kind={job.kind} failed", exc)
        self._finish(job, status)

    def _finish(self, job: Job, status: str) -> None:
        with job.lock:
            if job.terminal:
                return
            job.status = status
            job.finished = self.now()
            if status == "done":
                job.stage = "done"
                job.current = None
        with self._lock:
            if job.cache_key is not None and self._inflight.get(job.cache_key) == job.id:
                del self._inflight[job.cache_key]
        info = job.log_info
        ms = int(max(0.0, (job.finished or 0) - (job.started or job.created)) * 1000)
        JOBS_LOGGER.info("job=%s kind=%s source=%s series=%s naver=%s status=%s ms=%d error=%s",
                         short_id(job.id), job.kind, info.get("source", "-"), info.get("series", "-"),
                         info.get("naver", "-"), status, ms, job.error.code if job.error is not None else "-")

    # ------------------------------------------------------------------ housekeeping and shutdown
    def sweep(self) -> int:
        """Drop terminal jobs finished more than 10 minutes ago."""
        now = self.now()
        with self._lock:
            old = [jid for jid, j in self._jobs.items()
                   if j.terminal and j.finished is not None and now - j.finished > _KEEP_TERMINAL_S]
            for jid in old:
                del self._jobs[jid]
        return len(old)

    def drain(self) -> None:
        """Refuse new jobs and cancel the queued ones (E.14 step 1)."""
        with self._lock:
            self._closed = True
            queued = [j for j in self._jobs.values() if j.status == "queued"]
        for job in queued:
            with job.lock:
                job.cancel_requested = True
            self._finish(job, "cancelled")

    @property
    def draining(self) -> bool:
        return self._closed

    def running(self) -> int:
        return self.counts()[0]

    def close(self, timeout: float = 30.0) -> None:
        """Drain, wait for running jobs up to ``timeout`` seconds, then stop the executors (idempotent)."""
        self.drain()
        deadline = time.monotonic() + max(0.0, float(timeout))
        while self.running() and time.monotonic() < deadline:
            time.sleep(0.05)
        if self.running():
            LOG.warning("shutdown: %d running job(s) abandoned", self.running())
        for pool in self._pools.values():
            pool.executor.shutdown(wait=False, cancel_futures=True)


# --------------------------------------------------------------------------- snapshots (C.6)

def _elapsed_ms(job: Job, now: float) -> int:
    end = job.finished if job.finished is not None else now
    return int(max(0.0, end - job.created) * 1000)


def analyze_snapshot(job: Job, *, since: int, hl: str, now: float, naver_available: bool = True) -> dict:
    """The C.6 body of an analyze job in ``hl``, taken under the job lock."""
    with job.lock:
        status = job.status
        series = [{"index": e["index"], "status": "done", "vm": copy.deepcopy(e["vm"][hl])}
                  for e in job.vms if e["index"] >= since]
        body = {
            "job_id": job.id,
            "kind": "analyze",
            "status": status,
            "stage": job.stage,
            "series_total": job.series_total,
            "series_done": job.series_done,
            "current": dict(job.current) if job.current is not None and status == "running" else None,
            "series": series,
            "failed": copy.deepcopy(job.failed),
            "notices": copy.deepcopy(job.notices.get(hl, [])),
            "compare": (copy.deepcopy(job.compare[hl]) if status == "done" and job.view == "compare"
                        and job.compare is not None else None),
            "view": job.view,
            "error": (error_object(job.error, job.request_id, naver_available=naver_available)
                      if status == "failed" and job.error is not None else None),
            "result_url": f"/api/results/{job.id}" if status == "done" else None,
            "elapsed_ms": _elapsed_ms(job, now),
            "poll_after_ms": None if status in TERMINAL else POLL_AFTER_MS,
        }
    return body


def narrative_snapshot(job: Job, *, now: float, naver_available: bool = True) -> dict:
    """The C.6 body of a narrative job."""
    with job.lock:
        status = job.status
        return {
            "job_id": job.id,
            "kind": "narrative",
            "status": status,
            "error": (error_object(job.error, job.request_id, naver_available=naver_available)
                      if status == "failed" and job.error is not None else None),
            "elapsed_ms": _elapsed_ms(job, now),
            "poll_after_ms": None if status in TERMINAL else POLL_AFTER_MS,
            "narrative": copy.deepcopy(job.narrative) if status == "done" else None,
        }
