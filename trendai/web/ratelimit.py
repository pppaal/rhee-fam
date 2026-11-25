"""Per-client token buckets (spec C.13, E.8).

``tokens = min(burst, tokens + elapsed x refill / 3600)``; a request needs one token; a refusal returns
``retry_after = ceil((1 - tokens) x 3600 / refill)``. One lock per limiter, at most 50,000 keys (least recently used
dropped first), idle buckets dropped by the sweeper after an hour. The clock is the app clock.
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
}
_MAX_KEYS = 50_000
_IDLE_S = 3600.0


class RateLimiter:
    """Token buckets keyed by ``(bucket, client key)`` (the ``poll`` bucket by ``(client key, job id)``)."""

    def __init__(self, clock: Callable[[], _dt.datetime], *, max_keys: int = _MAX_KEYS) -> None:
        self._clock = clock
        self._lock = threading.Lock()
        self._buckets: "OrderedDict[tuple[str, Hashable], list[float]]" = OrderedDict()
        self._max_keys = max_keys

    def _now(self) -> float:
        return self._clock().timestamp()

    def take(self, bucket: str, key: Hashable, *, refill: Optional[float] = None,
             burst: Optional[int] = None) -> Optional[int]:
        """Take one token. Returns ``None`` when allowed, else the ``Retry-After`` seconds (>= 1)."""
        default_burst, default_refill = BUCKETS[bucket]
        cap = float(burst if burst is not None else default_burst)
        rate = float(refill if refill is not None else default_refill)
        now = self._now()
        ident = (bucket, key)
        with self._lock:
            state = self._buckets.get(ident)
            if state is None:
                tokens = cap
            else:
                tokens = min(cap, state[0] + max(0.0, now - state[1]) * rate / 3600.0)
                self._buckets.move_to_end(ident)
            if tokens >= 1.0:
                self._buckets[ident] = [tokens - 1.0, now]
                self._trim()
                return None
            self._buckets[ident] = [tokens, now]
            self._trim()
            return max(1, int(math.ceil((1.0 - tokens) * 3600.0 / rate)))

    def _trim(self) -> None:
        while len(self._buckets) > self._max_keys:
            self._buckets.popitem(last=False)

    def sweep(self, idle_s: float = _IDLE_S) -> int:
        """Drop buckets idle for more than ``idle_s`` seconds (they would be full again anyway)."""
        now = self._now()
        with self._lock:
            stale = [k for k, (_t, at) in self._buckets.items() if now - at > idle_s]
            for k in stale:
                del self._buckets[k]
        return len(stale)

    def __len__(self) -> int:
        with self._lock:
            return len(self._buckets)
