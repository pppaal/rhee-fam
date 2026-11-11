"""Compare several trends and rank them by how strongly they are rising.

The *rising score* (0–100, 50 = neutral) blends where a trend is heading
(forecast change and its probability) with where it is right now (recent
change, momentum) so that early, accelerating trends float to the top::

    score = 50 + 50 * (0.35·tanh(expected_ann/50%) + 0.25·(2·P(up) − 1)
                       + 0.20·tanh(recent_ann/50%) + 0.20·tanh(momentum_pm/2))

All inputs are put on a **common time basis** first, so a daily and a monthly
file of the same process get about the same score:

* ``expected_ann`` – the outlook's expected change, annualised (compounded)
  over the time that actually separates the outlook's *current level* (mean of
  the last season) from its *future level* (mean of the last season of the
  horizon),
* ``recent_ann`` – year-over-year change of the trend component (last
  ``periods_per_year`` periods; annualised when the series is shorter),
* ``momentum_pm`` – the analysis' momentum converted from % per period to
  % per month,
* ``P(up)`` – the outlook's probability of a rise at the end of the forecast
  horizon (used as is).

For monthly data with the default 12-month horizon the annualised numbers equal
the per-report ones, so the score is unchanged there.  The raw per-report
values are kept in the ranking (``momentum``, ``recent_change_pct``,
``expected_change_pct``) next to the normalised ones (``momentum_per_month``,
``recent_change_ann_pct``, ``expected_change_ann_pct``).
"""

from __future__ import annotations

import math
from typing import Iterable, Mapping, Optional, Union

import numpy as np
import pandas as pd

from .data import periods_per_year
from .types import PHASE_KO, TrendReport

RANK_COLUMNS = ["name", "phase", "phase_ko", "momentum", "recent_change_pct",
                "expected_change_pct", "prob_increase", "confidence", "score",
                "freq", "momentum_per_month", "recent_change_ann_pct", "expected_change_ann_pct"]

_MAX_LOG_GROWTH = math.log(101.0)  # cap annualised growth at +10,000 % (keeps every float finite)


def _f(x: Optional[float]) -> float:
    """None/NaN-safe float (treated as 0 for scoring)."""
    if x is None:
        return 0.0
    try:
        v = float(x)
    except (TypeError, ValueError):
        return 0.0
    return v if math.isfinite(v) else 0.0


def _opt(x: Optional[float]) -> Optional[float]:
    """Finite float or None."""
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def annualise_pct(pct: Optional[float], periods: float, ppy: float) -> Optional[float]:
    """Compound a change of ``pct`` % over ``periods`` periods to a per-year rate (in %).

    ``ppy`` is the number of periods per year.  A fall of 100 % or more stays
    −100 %; growth is capped at +10,000 % per year.  ``None`` when undefined.
    """
    v = _opt(pct)
    if v is None or not (periods > 0) or not (ppy > 0):
        return None
    ratio = 1.0 + v / 100.0
    if ratio <= 0.0:
        return -100.0
    lg = math.log(ratio) * (ppy / periods)
    lg = min(lg, _MAX_LOG_GROWTH)
    return float(math.expm1(max(lg, -50.0)) * 100.0)


def _ppy(report: TrendReport) -> float:
    try:
        return float(periods_per_year(report.analysis.freq))
    except Exception:  # noqa: BLE001 - unknown freq: treat as monthly
        return 12.0


def expected_change_annualised(report: TrendReport) -> Optional[float]:
    """The outlook's expected change as a per-year rate (see the module docstring)."""
    a, fc = report.analysis, report.forecast
    m = max(1, int(a.season_length or 1))
    n = max(1, len(a.series))
    h = max(1, int(fc.horizon or len(fc.mean) or 1))
    k_cur, k_fut = min(m, n), min(m, h)
    gap = h - (k_fut - 1) / 2.0 + (k_cur - 1) / 2.0
    return annualise_pct(report.outlook.expected_change_pct, gap, _ppy(report))


def recent_change_annualised(report: TrendReport) -> Optional[float]:
    """Year-over-year change of the trend component in % (annualised if shorter than a year)."""
    a = report.analysis
    trend = getattr(a.decomposition, "trend", None)
    y = np.asarray(trend if trend is not None and len(trend) == len(a.series) else a.series,
                   dtype=float)
    n = len(y)
    if n < 2 or not np.isfinite(y[-1]):
        return None
    ppy = _ppy(report)
    k = int(min(max(1, round(ppy)), n - 1))
    base, last = float(y[-1 - k]), float(y[-1])
    scale = float(np.nanmax(np.abs(y))) if np.isfinite(y).any() else 0.0
    if not math.isfinite(base) or base <= 1e-9 * max(scale, 1e-300):
        return None
    return annualise_pct((last / base - 1.0) * 100.0, k, ppy)


def momentum_per_month(report: TrendReport) -> float:
    """Momentum (% per period, see analysis) expressed in % per month."""
    return _f(report.analysis.metrics.momentum) * _ppy(report) / 12.0


def rising_score(report: TrendReport) -> float:
    """Composite 0–100 score: higher = stronger expected rise (50 = neutral).

    Frequency-neutral: the change components are annualised and momentum is
    per month (see the module docstring), so daily, weekly, monthly and
    quarterly series can be ranked together.
    """
    o = report.outlook
    z = (0.35 * math.tanh(_f(expected_change_annualised(report)) / 50.0)
         + 0.25 * (2.0 * min(1.0, max(0.0, _f(o.prob_increase))) - 1.0)
         + 0.20 * math.tanh(_f(recent_change_annualised(report)) / 50.0)
         + 0.20 * math.tanh(momentum_per_month(report) / 2.0))
    return float(min(100.0, max(0.0, 50.0 + 50.0 * z)))


def rank_trends(reports: Union[Mapping[str, TrendReport], Iterable[TrendReport]]) -> pd.DataFrame:
    """Return a ranking table (best first, index = rank from 1) with the columns in ``RANK_COLUMNS``.

    ``momentum`` / ``recent_change_pct`` / ``expected_change_pct`` are the
    per-report values (each in its own frequency's units); the ``*_ann_pct``
    and ``momentum_per_month`` columns are the frequency-neutral values the
    score is computed from.  Ties keep the input order.
    """
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
            "freq": r.analysis.freq,
            "momentum_per_month": momentum_per_month(r),
            "recent_change_ann_pct": recent_change_annualised(r),
            "expected_change_ann_pct": expected_change_annualised(r),
        })
    df = pd.DataFrame(rows, columns=RANK_COLUMNS)
    if not df.empty:
        df = df.sort_values("score", ascending=False, kind="mergesort").reset_index(drop=True)
        df.index = df.index + 1
        df.index.name = "rank"
    return df
