"""Compare several trends and rank them by how strongly they are rising.

The *rising score* (0–100, 50 = neutral) blends where a trend is heading
(forecast change and its probability) with where it is right now (recent
change, momentum) so that early, accelerating trends float to the top::

    score = 50 + 50 * (0.35·tanh(expected_ann/50%) + 0.25·(2·P(up) − 1)
                       + 0.20·tanh(recent_ann/50%) + 0.20·tanh(momentum_pm/2))

The two backward-looking terms (recent change, momentum) count only where the
analysis found a reliable recent trend: in a ``'stable'`` or ``'volatile'``
phase its rules found the recent slope to be within the noise (the narrative
calls such a difference "잡음 수준"), so both terms are 0 there and the score
rests on the outlook alone -- a noise series cannot outrank a positive
outlook on the strength of a random recent upswing.

All inputs are put on a **common time basis** first, so a daily and a monthly
file of the same process get about the same score:

* ``expected_ann`` – the outlook's expected change, annualised (compounded)
  over the time that actually separates the outlook's *current level* (mean of
  the last season) from its *future level* (mean of the last season of the
  horizon),
* ``recent_ann`` – year-over-year change of the trend component (last
  ``periods_per_year`` periods; annualised when the series is shorter).
  It follows the analysis' percentage rules: ``None`` (no contribution)
  when the analysis states no recent change in % (``metrics.recent_change_pct``
  is ``None``: intermittent demand, a level that crossed zero, a base near
  zero), and when :func:`trendai.analysis.pct_change_undefined` refuses the
  year-over-year change itself (a P&L that fell through zero is no "−100 %") --
  except for a trend that fell to about 0 (``'at_zero'``): no % is printed for
  it, but it scores as the strongest fall (−100 %), as an outlook whose
  forecast falls to the 0 floor (``'floor'``) does for ``expected_ann``,
* ``momentum_pm`` – the analysis' momentum converted from % per period to
  % per month; 0 (no contribution) where a %p of the level means nothing
  (:func:`momentum_undefined`: a level near 0, a negative level, intermittent
  demand, an all-zero history -- the table shows ``—`` there),
* ``P(up)`` – the outlook's ``prob_increase``: the probability that the
  future level (mean of the last season of the horizon) exceeds the current
  level (mean of the last season), i.e. the same two levels as the expected
  change (used as is); 0.5 (no contribution) where
  :func:`trendai.narrative.prob_undefined` gives no probability (a forecast of
  exactly 0, a current level of 0 that cannot fall -- the table shows ``—``).

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

from .analysis import _is_intermittent, momentum_undefined as _momentum_undefined, pct_change_undefined
from .data import periods_per_year
from .forecast import outlook_levels
from .narrative import prob_undefined
from .types import PHASE_KO, TrendReport

RANK_COLUMNS = ["name", "phase", "phase_ko", "momentum", "recent_change_pct",
                "expected_change_pct", "prob_increase", "confidence", "score",
                "freq", "momentum_per_month", "recent_change_ann_pct", "expected_change_ann_pct"]

_MAX_LOG_GROWTH = math.log(101.0)  # cap annualised growth at +10,000 % (keeps every float finite)

#: lifecycle phases in which the analysis found no reliable recent trend: the recent change
#: and the momentum are noise there and do not enter the rising score
NOISE_PHASES = ("stable", "volatile")


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
    pct = report.outlook.expected_change_pct
    if pct is None and _outlook_undefined(report) == "floor":
        # the forecast falls to the 0 floor: no "−100 %" is printed, but the score keeps the fall
        pct = -100.0
    return annualise_pct(pct, gap, _ppy(report))


def _outlook_undefined(report: TrendReport) -> Optional[str]:
    """``forecast.outlook_levels(...).undefined`` (``None`` for a hand-made / partial report)."""
    try:
        return outlook_levels(report.analysis, report.forecast).undefined
    except Exception:  # noqa: BLE001 - hand-made / partial reports
        return None


def recent_change_annualised(report: TrendReport) -> Optional[float]:
    """Year-over-year change of the trend component in % (annualised if shorter than a year).

    ``None`` -- the score then leaves the recent change out -- wherever the
    analysis gives no percentage for it: ``metrics.recent_change_pct`` is
    ``None`` (intermittent demand, a trend that crossed zero, a base near
    zero), the series is intermittent demand, or
    :func:`trendai.analysis.pct_change_undefined` refuses the change over this
    window (base ~0 or negative, a crossing of zero, a base within half a
    standard deviation of 0 for data with negative values).  A table showing
    ``—`` for the recent change never sits next to a score driven by it.
    """
    a = report.analysis
    values = np.asarray(a.series, dtype=float)
    if _is_intermittent(values, int(a.season_length or 1)):
        return None
    trend = getattr(a.decomposition, "trend", None)
    y = np.asarray(trend if trend is not None and len(trend) == len(a.series) else a.series,
                   dtype=float)
    n = len(y)
    if n < 2 or not np.isfinite(y[-1]):
        return None
    ppy = _ppy(report)
    k = int(min(max(1, round(ppy)), n - 1))
    base, last = float(y[-1 - k]), float(y[-1])
    why = pct_change_undefined(base, last, values)
    if why == "at_zero":
        # a trend that fell to (about) 0: the surfaces print no "−100 %" ('at_zero'), but the
        # score keeps the direction -- the strongest fall, not a neutral 0 (a stopped product
        # scored 38.4, level with an ordinary declining series)
        return -100.0
    if why is not None or getattr(a.metrics, "recent_change_pct", 0.0) is None:
        return None
    return annualise_pct((last / base - 1.0) * 100.0, k, ppy)


def _values(report: TrendReport) -> np.ndarray:
    y = np.asarray(report.analysis.series, dtype=float)
    return y[np.isfinite(y)]


def momentum_undefined(report: TrendReport) -> Optional[str]:
    """Why the momentum (%p of the current trend level) means nothing for ``report``:
    :func:`trendai.analysis.momentum_undefined` (a level near 0, a negative level, intermittent
    demand, an all-zero history, or a trend that fell to about 0: ``'at_zero'``) -- the rule every
    surface uses before printing it (``—``)."""
    a = report.analysis
    trend = getattr(getattr(a, "decomposition", None), "trend", None)
    return _momentum_undefined(_values(report), int(a.season_length or 1),
                               None if trend is None else np.asarray(trend, dtype=float))


def momentum_per_month(report: TrendReport) -> float:
    """Momentum (% per period, see analysis) expressed in % per month; 0 where it means nothing
    (:func:`momentum_undefined`: the score then leaves it out, as the table shows ``—``)."""
    if momentum_undefined(report) is not None:
        return 0.0
    return _f(report.analysis.metrics.momentum) * _ppy(report) / 12.0


def rising_score(report: TrendReport) -> float:
    """Composite 0–100 score: the outlook plus the recent momentum (50 = neutral).

    Higher means a stronger rising picture overall -- the forecast's expected
    change and ``P(up)`` blended with the recent change and momentum (see the
    module docstring) -- not an expected rise by itself.

    Frequency-neutral: the change components are annualised and momentum is
    per month (see the module docstring), so daily, weekly, monthly and
    quarterly series can be ranked together.  In a ``'stable'`` /
    ``'volatile'`` phase the recent change and momentum are noise and are
    left out (the outlook alone decides).
    """
    o = report.outlook
    # no probability where the table shows "—" (prob_undefined): neutral, like a missing change
    p_up = 0.5 if prob_undefined(report.analysis) is not None else _f(o.prob_increase)
    z = (0.35 * math.tanh(_f(expected_change_annualised(report)) / 50.0)
         + 0.25 * (2.0 * min(1.0, max(0.0, p_up)) - 1.0))
    if report.analysis.phase.label not in NOISE_PHASES:
        z += (0.20 * math.tanh(_f(recent_change_annualised(report)) / 50.0)
              + 0.20 * math.tanh(momentum_per_month(report) / 2.0))
    return float(min(100.0, max(0.0, 50.0 + 50.0 * z)))


def rank_trends(reports: Union[Mapping[str, TrendReport], Iterable[TrendReport]]) -> pd.DataFrame:
    """Return a ranking table (best first, index = rank from 1) with the columns in ``RANK_COLUMNS``.

    ``momentum`` / ``recent_change_pct`` / ``expected_change_pct`` are the
    per-report values (each in its own frequency's units); the ``*_ann_pct``
    and ``momentum_per_month`` columns are the frequency-neutral values the
    score is computed from.  ``momentum`` / ``momentum_per_month`` are ``None``
    where a %p of the level means nothing (:func:`momentum_undefined`) and
    ``prob_increase`` where :func:`trendai.narrative.prob_undefined` gives a reason (a
    forecast of exactly 0, a current level of 0 that cannot fall).  Ties keep the input order.
    """
    items = list(reports.values()) if isinstance(reports, Mapping) else list(reports)
    rows = []
    for r in items:
        m, o = r.analysis.metrics, r.outlook
        mom_ok = momentum_undefined(r) is None
        rows.append({
            "name": r.name,
            "phase": r.analysis.phase.label,
            "phase_ko": PHASE_KO.get(r.analysis.phase.label, r.analysis.phase.label),
            "momentum": m.momentum if mom_ok else None,
            "recent_change_pct": m.recent_change_pct,
            "expected_change_pct": o.expected_change_pct,
            # a forecast of exactly 0, or a level of 0 that cannot fall: no probability (narrative.prob_undefined)
            "prob_increase": None if prob_undefined(r.analysis) is not None else o.prob_increase,
            "confidence": o.confidence,
            "score": rising_score(r),
            "freq": r.analysis.freq,
            "momentum_per_month": momentum_per_month(r) if mom_ok else None,
            "recent_change_ann_pct": recent_change_annualised(r),
            "expected_change_ann_pct": expected_change_annualised(r),
        })
    df = pd.DataFrame(rows, columns=RANK_COLUMNS)
    if not df.empty:
        df = df.sort_values("score", ascending=False, kind="mergesort").reset_index(drop=True)
        df.index = df.index + 1
        df.index.name = "rank"
    return df
