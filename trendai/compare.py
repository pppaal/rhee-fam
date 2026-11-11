"""Compare several trends and rank them by how strongly they are rising.

The *rising score* (0–100, 50 = neutral) blends where a trend is heading
(forecast change and its probability) with where it is right now (recent
change, momentum) so that early, accelerating trends float to the top::

    score = 50 + 50 * (0.35·tanh(expected_change/50%) + 0.25·(2·P(up) − 1)
                       + 0.20·tanh(recent_change/50%) + 0.20·tanh(momentum/2))
"""

from __future__ import annotations

import math
from typing import Iterable, Mapping, Optional, Union

import pandas as pd

from .types import PHASE_KO, TrendReport

RANK_COLUMNS = ["name", "phase", "phase_ko", "momentum", "recent_change_pct",
                "expected_change_pct", "prob_increase", "confidence", "score"]


def _f(x: Optional[float]) -> float:
    """None/NaN-safe float (treated as 0 for scoring)."""
    if x is None:
        return 0.0
    try:
        v = float(x)
    except (TypeError, ValueError):
        return 0.0
    return v if math.isfinite(v) else 0.0


def rising_score(report: TrendReport) -> float:
    """Composite 0–100 score: higher = stronger expected rise (50 = neutral)."""
    m, o = report.analysis.metrics, report.outlook
    z = (0.35 * math.tanh(_f(o.expected_change_pct) / 50.0)
         + 0.25 * (2.0 * _f(o.prob_increase) - 1.0)
         + 0.20 * math.tanh(_f(m.recent_change_pct) / 50.0)
         + 0.20 * math.tanh(_f(m.momentum) / 2.0))
    return float(min(100.0, max(0.0, 50.0 + 50.0 * z)))


def rank_trends(reports: Union[Mapping[str, TrendReport], Iterable[TrendReport]]) -> pd.DataFrame:
    """Return a ranking table (best first) with the columns in ``RANK_COLUMNS``."""
    items = list(reports.values()) if isinstance(reports, Mapping) else list(reports)
    rows = []
    for r in items:
        m, o = r.analysis.metrics, r.outlook
        rows.append({
            "name": r.name,
            "phase": r.analysis.phase.label,
            "phase_ko": PHASE_KO.get(r.analysis.phase.label, r.analysis.phase.label),
            "momentum": m.momentum,
            "recent_change_pct": m.recent_change_pct,
            "expected_change_pct": o.expected_change_pct,
            "prob_increase": o.prob_increase,
            "confidence": o.confidence,
            "score": rising_score(r),
        })
    df = pd.DataFrame(rows, columns=RANK_COLUMNS)
    if not df.empty:
        df = df.sort_values("score", ascending=False, kind="mergesort").reset_index(drop=True)
        df.index = df.index + 1
        df.index.name = "rank"
    return df
