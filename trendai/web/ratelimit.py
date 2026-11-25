"""Per-client token buckets (spec C.13, E.8).

``tokens = min(burst, tokens + elapsed x refill / 3600)``; a request needs one token; a refusal returns
``retry_after = ceil((1 - tokens) x 3600 / refill)``. One lock per limiter; every bucket name keeps its own
least-recently-used table of at most 50,000 keys, so one kind of key can never push another kind out (a flood of
per-job ``poll`` keys used to evict -- and so refill -- the same client's ``analyze``/``naver``/``upload`` buckets).
Idle buckets are dropped by the sweeper after an hour. The clock is the app clock.
"""

from __future__ import annotations

import datetime as _dt
import math
import threading
from collections import OrderedDict
from typing import Callable, Hashable, Optional

__all__ = ["RateLimiter", "BUCKETS"]

#: bucket -> (burst, refill per hour) (C.13). ``naver`` and ``ai`` refills come from the config.
BUCKETS: dict[str, tuple[int, float]] = {
    "analyze": (10, 60.0),
    "upload": (5, 30.0),
    "naver": (5, 20.0),
    "ai": (3, 10.0),
    "share": (5, 20.0),
    "export": (30, 300.0),
    "poll": (60, 21_600.0),
    "read": (60, 1_200.0),
    # /api/config and /api/samples: small, identical for everyone and needed to boot the SPA, so a shared address
    # (office NAT, carrier-grade NAT) must not lock everybody out of the app with the stricter "read" bucket
    "config": (300, 6_000.0),
}
_MAX_KEYS = 50_000
_IDLE_S = 3600.0


class RateLimiter:
    """Token buckets keyed by ``(bucket, client key)`` (the ``poll`` bucket by ``(client key, job id)``)."""

    def __init__(self, clock: Callable[[], _dt.datetime], *, max_keys: int = _MAX_KEYS) -> None:
        self._clock = clock
        self._lock = threading.Lock()
        self._tables: dict[str, "OrderedDict[Hashable, list[float]]"] = {}
        self._max_keys = max_keys

    def _now(self) -> float:
        return self._clock().timestamp()

    @staticmethod
    def _limits(bucket: str, refill: Optional[float], burst: Optional[int]) -> tuple[float, float]:
        default_burst, default_refill = BUCKETS[bucket]
        return (float(burst if burst is not None else default_burst),
                float(refill if refill is not None else default_refill))

    def take(self, bucket: str, key: Hashable, *, refill: Optional[float] = None,
             burst: Optional[int] = None) -> Optional[int]:
        """Take one token. Returns ``None`` when allowed, else the ``Retry-After`` seconds (>= 1)."""
        cap, rate = self._limits(bucket, refill, burst)
        now = self._now()
        with self._lock:
            table = self._tables.setdefault(bucket, OrderedDict())
            state = table.get(key)
            if state is None:
                tokens = cap
            else:
                tokens = min(cap, state[0] + max(0.0, now - state[1]) * rate / 3600.0)
                table.move_to_end(key)
            if tokens >= 1.0:
                table[key] = [tokens - 1.0, now]
                self._trim(table)
                return None
            table[key] = [tokens, now]
            self._trim(table)
            return max(1, int(math.ceil((1.0 - tokens) * 3600.0 / rate)))

    def give_back(self, bucket: str, key: Hashable, *, burst: Optional[int] = None) -> None:
        """Return one token taken by :meth:`take` for a request that was then refused for a reason that is not the
        client's (``503 server_busy``): following the 503's ``Retry-After`` must not run the client into a 429."""
        cap, _rate = self._limits(bucket, None, burst)
        with self._lock:
            state = (self._tables.get(bucket) or {}).get(key)
            if state is not None:
                state[0] = min(cap, state[0] + 1.0)

    def _trim(self, table: "OrderedDict[Hashable, list[float]]") -> None:
        while len(table) > self._max_keys:
            table.popitem(last=False)

    def sweep(self, idle_s: float = _IDLE_S) -> int:
        """Drop buckets idle for more than ``idle_s`` seconds (they would be full again anyway)."""
        now = self._now()
        removed = 0
        with self._lock:
            for table in self._tables.values():
                stale = [k for k, (_t, at) in table.items() if now - at > idle_s]
                for k in stale:
                    del table[k]
                removed += len(stale)
        return removed

    def __len__(self) -> int:
        with self._lock:
            return sum(len(t) for t in self._tables.values())
