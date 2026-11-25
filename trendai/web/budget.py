"""Monthly Naver budget and daily AI cap (spec C.13, E.9), counted in files so they survive restarts.

* :class:`NaverBudget`: per KST calendar month. ``count >= 80 %`` -> state ``low`` (the per-IP ``naver`` bucket refill
  drops to 5/h); ``count >= 90 %`` -> ``blocked`` (fresh calls refused with ``503 naver_budget_exhausted``; cache hits
  still served). :meth:`NaverBudget.reserve` increments *before* the HTTP call, so failed calls count.
* :class:`AiBudget`: per KST day against ``TRENDAI_AI_DAILY_LIMIT``; blocks at 100 % with ``503 ai_unavailable``
  (``params.reason = "daily_limit"``).
"""

from __future__ import annotations

import datetime as _dt
import threading
from typing import Callable

from .errors import WebError
from .store import Store, utc_iso

__all__ = ["NaverBudget", "AiBudget", "kst_now"]

_KST = _dt.timezone(_dt.timedelta(hours=9))


def kst_now(clock: Callable[[], _dt.datetime]) -> _dt.datetime:
    """The app clock in fixed KST (UTC+9, no tzdata, section 0)."""
    return clock().astimezone(_KST)


class _Counter:
    def __init__(self, store: Store, clock: Callable[[], _dt.datetime], limit: int) -> None:
        self.store = store
        self.clock = clock
        self.limit = int(limit)
        self._lock = threading.Lock()

    def _period(self) -> tuple[str, str]:   # (file name, period label)
        raise NotImplementedError

    def _key(self) -> str:
        raise NotImplementedError

    def count(self) -> int:
        name, label = self._period()
        obj = self.store.budget_read(name)
        if not obj or obj.get(self._key()) != label:
            return 0
        try:
            return max(0, int(obj.get("count", 0)))
        except (TypeError, ValueError):
            return 0

    def _write(self, count: int) -> None:
        name, label = self._period()
        self.store.budget_write(name, {self._key(): label, "count": int(count), "updated_at": utc_iso(self.clock())})


class NaverBudget(_Counter):
    """``NaverBudget(store, clock, limit)`` (E.9)."""

    def _period(self) -> tuple[str, str]:
        month = kst_now(self.clock).strftime("%Y-%m")
        return f"naver-{month}", month

    def _key(self) -> str:
        return "month"

    def state(self) -> tuple[int, int, str]:
        """``(count, limit, "ok" | "low" | "blocked")``."""
        count = self.count()
        return count, self.limit, self._state_of(count)

    def _state_of(self, count: int) -> str:
        if count >= 0.9 * self.limit:
            return "blocked"
        if count >= 0.8 * self.limit:
            return "low"
        return "ok"

    def reserve(self) -> int:
        """Count one fresh Naver call before making it; ``WebError(naver_budget_exhausted)`` at >= 90 %."""
        with self._lock:
            count = self.count()
            if self._state_of(count) == "blocked":
                raise WebError("naver_budget_exhausted")
            self._write(count + 1)
            return count + 1


class AiBudget(_Counter):
    """``AiBudget(store, clock, limit)``: AI narratives per KST day (no 90 % rule: blocks at 100 %)."""

    def _period(self) -> tuple[str, str]:
        day = kst_now(self.clock).strftime("%Y-%m-%d")
        return f"ai-{day}", day

    def _key(self) -> str:
        return "day"

    def reserve(self) -> int:
        with self._lock:
            count = self.count()
            if count >= self.limit:
                raise WebError("ai_unavailable", params={"reason": "daily_limit"})
            self._write(count + 1)
            return count + 1

    def release(self) -> None:
        """Give back a reservation whose job could not be queued (``server_busy``)."""
        with self._lock:
            count = self.count()
            if count > 0:
                self._write(count - 1)
