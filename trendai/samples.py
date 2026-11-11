"""Deterministic synthetic sample series (clearly synthetic, not real data).

Each archetype represents a typical trend lifecycle and comes with a known
"ground truth" (expected lifecycle phase and, where applicable, the dates of
planted structural breaks).  They power the demo (``trendai sample ...``) and
the accuracy tests.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .data import PANDAS_FREQ, season_length

DESCRIPTIONS: dict[str, str] = {
    "emerging": "낮은 기반에서 가속 성장 중인 신흥 트렌드 (S-커브 초입)",
    "growing": "꾸준한 성장 + 계절성",
    "peaking": "성장이 둔화되며 포화(정점)에 접근하는 트렌드 (S-커브 후반)",
    "fad": "급등 후 급락한 일시적 유행 (현재 하락 중)",
    "declining": "지속적으로 감소하는 트렌드",
    "seasonal": "추세 없이 계절성만 뚜렷한 안정형",
    "reviving": "하락 후 최근 반등하는 재부상 트렌드",
    "breakout": "횡보하다 특정 시점 이후 급성장 (구조 변화)",
    "volatile": "방향성 없이 잡음이 큰 변동형",
}

# Expected lifecycle phase at the end of each sample (acceptable alternatives listed).
EXPECTED_PHASE: dict[str, tuple[str, ...]] = {
    "emerging": ("emerging", "growing"),
    "growing": ("growing",),
    "peaking": ("peaking",),
    "fad": ("declining",),
    "declining": ("declining",),
    "seasonal": ("stable",),
    "reviving": ("reviving",),
    "breakout": ("growing", "emerging"),
    "volatile": ("volatile", "stable"),
}


def list_samples() -> list[str]:
    return list(DESCRIPTIONS)


def _seasonal(t: np.ndarray, period: int, amp: float) -> np.ndarray:
    if period <= 1:
        return np.zeros_like(t, dtype=float)
    return amp * (np.sin(2 * np.pi * t / period) + 0.35 * np.cos(4 * np.pi * t / period))


def sample(name: str, n: int = 120, freq: str = "M", start: str = "2016-01-01",
           seed: int = 0, noise: float = 1.0) -> pd.Series:
    """Generate the synthetic archetype ``name`` with ``n`` observations.

    ``noise`` scales the random component (0 gives a noiseless curve).
    Values are always positive.
    """
    if name not in DESCRIPTIONS:
        raise KeyError(f"알 수 없는 샘플 {name!r} (unknown sample); 사용 가능: {list_samples()}")
    if isinstance(n, bool) or int(n) != n or n < 8:
        raise ValueError(f"관측치 수 n은 8 이상의 정수여야 합니다 (n must be an integer >= 8): {n!r}")
    n = int(n)
    rng = np.random.default_rng(seed + sum(map(ord, name)))
    t = np.arange(n, dtype=float)
    u = t / (n - 1)  # 0..1
    p = season_length(freq)
    eps = rng.standard_normal(n)

    if name == "emerging":
        base = 5 + 120 / (1 + np.exp(-(u - 0.98) / 0.09))
        y = base * (1 + _seasonal(t, p, 0.04)) + noise * 1.2 * eps
    elif name == "growing":
        base = 40 + 60 * u + 25 * u ** 2
        y = base + _seasonal(t, p, 6.0) + noise * 2.5 * eps
    elif name == "peaking":
        base = 10 + 90 / (1 + np.exp(-(u - 0.55) / 0.08))
        y = base + _seasonal(t, p, 2.0) + noise * 1.5 * eps
    elif name == "fad":
        base = 8 + 92 * np.exp(-0.5 * ((u - 0.55) / 0.12) ** 2)
        y = base + noise * 2.0 * eps
    elif name == "declining":
        base = 100 * np.exp(-1.6 * u) + 5
        y = base + _seasonal(t, p, 3.0) + noise * 1.8 * eps
    elif name == "seasonal":
        y = 50 + _seasonal(t, p, 12.0) + noise * 2.0 * eps
    elif name == "reviving":
        base = 30 + 50 * (u - 0.72) ** 2 / 0.72 ** 2
        base = np.where(u > 0.72, 30 + 260 * (u - 0.72) ** 2, base)
        y = base + _seasonal(t, p, 2.0) + noise * 1.5 * eps
    elif name == "breakout":
        k = int(round(0.6 * (n - 1)))
        base = 30 + 0.02 * t
        base = base + np.where(t > k, 1.2 * (t - k), 0.0)
        y = base + _seasonal(t, p, 2.0) + noise * 1.5 * eps
    elif name == "volatile":
        y = 50 + noise * 12.0 * eps + 3.0 * np.sin(t / 3.7)
    else:  # pragma: no cover
        raise KeyError(name)

    y = np.maximum(y, 0.5)
    idx = pd.date_range(start=start, periods=n, freq=PANDAS_FREQ[freq])
    s = pd.Series(y.astype(float), index=idx, name=name)
    s.index.name = "date"
    return s


def breakout_date(n: int = 120, freq: str = "M", start: str = "2016-01-01") -> pd.Timestamp:
    """The planted changepoint date of the ``breakout`` sample."""
    k = int(round(0.6 * (n - 1)))
    return pd.date_range(start=start, periods=n, freq=PANDAS_FREQ[freq])[k]


def sample_set(names: Optional[list[str]] = None, **kwargs) -> dict[str, pd.Series]:
    """Several samples at once, e.g. for the ``compare`` demo."""
    names = names or ["emerging", "growing", "peaking", "fad", "declining", "seasonal"]
    return {nm: sample(nm, **kwargs) for nm in names}
