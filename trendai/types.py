"""Shared result types for trendai.

Every module communicates through these dataclasses so that analysis,
forecasting, narrative generation and reporting stay decoupled.

Conventions
-----------
* A *series* is a ``pd.Series`` of floats with a sorted, regular
  ``pd.DatetimeIndex`` (see :mod:`trendai.data`).
* ``freq`` is one of the canonical codes ``'D' 'W' 'M' 'Q' 'Y'``.
* Percentages are expressed in percent units (``12.5`` means +12.5%).
* Slopes are "per period" in the series' own units unless noted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

# Lifecycle phases a trend can be in (see analysis.lifecycle_phase).
PHASES: tuple[str, ...] = (
    "emerging",   # small base, accelerating growth (early adoption)
    "growing",    # sustained, significant upward trend
    "peaking",    # was growing, growth has slowed/stalled near the high
    "declining",  # sustained downward trend / well below the peak
    "stable",     # no significant trend (possibly seasonal)
    "reviving",   # declined in the past, now turning up again
    "volatile",   # noise dominates, no reliable direction
)

PHASE_KO: dict[str, str] = {
    "emerging": "태동기(신흥)",
    "growing": "성장기",
    "peaking": "정점/성숙기",
    "declining": "쇠퇴기",
    "stable": "안정기",
    "reviving": "재부상",
    "volatile": "변동성 장세",
}

# Changepoint kinds (see analysis.detect_changepoints).
CHANGEPOINT_KINDS: tuple[str, ...] = (
    "acceleration",   # same direction, slope got steeper
    "deceleration",   # same direction, slope got flatter
    "reversal_up",    # slope went from <=0 to >0
    "reversal_down",  # slope went from >=0 to <0
    "level_shift",    # abrupt jump/drop in level with similar slope
)


@dataclass
class Decomposition:
    """Additive decomposition ``series = trend + seasonal + resid``."""

    trend: pd.Series
    seasonal: pd.Series
    resid: pd.Series
    period: int  # season length used (1 means no seasonal component)


@dataclass
class Changepoint:
    """A structural break in the trend."""

    date: pd.Timestamp
    index: int             # integer position in the series
    slope_before: float    # trend slope per period before the break
    slope_after: float     # trend slope per period after the break
    kind: str              # one of CHANGEPOINT_KINDS
    magnitude: float       # size of the change, relative to the series scale (unitless, >= 0)


@dataclass
class Anomaly:
    """A point that deviates strongly from trend + seasonality."""

    date: pd.Timestamp
    index: int
    value: float
    expected: float
    zscore: float          # robust z-score of the residual (signed)


@dataclass
class TrendMetrics:
    """Descriptive statistics of the past and present state of a series."""

    n_obs: int
    freq: str
    start: pd.Timestamp
    end: pd.Timestamp
    last_value: float
    mean: float
    total_change_pct: Optional[float]   # smoothed first -> last, None if base ~ 0
    cagr_pct: Optional[float]           # annualised growth rate in %, None if undefined
    recent_change_pct: Optional[float]  # change of the trend over the last season (YoY-like)
    slope: float                        # robust (Theil-Sen) slope per period
    slope_pct: float                    # slope relative to mean level, % per period
    momentum: float                     # recent slope vs. long-run slope, see analysis docs
    acceleration: float                 # normalised 2nd derivative of trend near the end
    volatility: float                   # residual std / mean level (coefficient of variation)
    trend_strength: float               # 0..1
    seasonal_strength: float            # 0..1
    mk_tau: float                       # Mann-Kendall tau
    mk_pvalue: float                    # Mann-Kendall two-sided p-value
    direction: str                      # 'up' | 'down' | 'flat'
    peak_date: pd.Timestamp
    peak_value: float
    drawdown_pct: float                 # (last - peak) / peak * 100, <= 0


@dataclass
class LifecyclePhase:
    label: str                          # one of PHASES
    confidence: float                   # 0..1
    reasons: list[str] = field(default_factory=list)  # short Korean explanations

    @property
    def label_ko(self) -> str:
        return PHASE_KO.get(self.label, self.label)


@dataclass
class AnalysisResult:
    """Past -> present analysis of one series."""

    name: str
    series: pd.Series
    freq: str
    season_length: int
    decomposition: Decomposition
    metrics: TrendMetrics
    changepoints: list[Changepoint]
    anomalies: list[Anomaly]
    phase: LifecyclePhase


@dataclass
class ModelScore:
    """Rolling-origin backtest score of one candidate model."""

    name: str
    mae: float
    rmse: float
    smape: float     # symmetric MAPE in %, 0..200
    mase: float      # mean absolute scaled error (scaled by seasonal-naive in-sample MAE)
    weight: float    # ensemble weight, 0 if not part of the ensemble


@dataclass
class ForecastResult:
    """Present -> future forecast of one series."""

    mean: pd.Series                        # indexed by future dates
    lower: dict[int, pd.Series]            # level (e.g. 80, 95) -> lower bound series
    upper: dict[int, pd.Series]            # level -> upper bound series
    horizon: int
    levels: tuple[int, ...]
    model_scores: list[ModelScore]         # sorted best (lowest MASE) first; recency-weighted
                                           # backtest scores (benchmarks naive / seasonal_naive /
                                           # drift included); empty when there was no backtest
    method: str                            # human readable description of the final method
    backtest_mase: Optional[float] = None  # estimated MASE of the final (ensemble) method
    nonnegative: bool = False              # whether the forecast was clipped at 0
    skill: Optional[float] = None          # backtest MAE of the final method / MAE of the best
                                           # naive benchmark (naive, seasonal_naive, drift) on the
                                           # same windows; < 1 means better than naive, None if n/a

    @property
    def index(self) -> pd.DatetimeIndex:
        return self.mean.index

    def to_frame(self) -> pd.DataFrame:
        cols = {"mean": self.mean}
        for lv in sorted(self.levels):
            cols[f"lo{lv}"] = self.lower[lv]
            cols[f"hi{lv}"] = self.upper[lv]
        return pd.DataFrame(cols)


@dataclass
class Outlook:
    """Interpretation of the forecast relative to the present."""

    direction: str                    # 'up' | 'down' | 'flat'
    expected_change_pct: Optional[float]  # future level vs current level, in %: mean of the
                                      # forecast's last season vs mean of the last observed
                                      # season (partial seasons adjusted); None when undefined
                                      # (zero / mostly-zero current level)
    prob_increase: float              # 0..1, P(future level > current level) for the same two
                                      # season-mean levels (NOT P(horizon-end value > now))
    scenarios: dict[str, float]       # {'pessimistic', 'base', 'optimistic'} values at horizon end
    forecast_peak_date: Optional[pd.Timestamp]  # date of max forecast mean if it peaks inside horizon
    future_phase: str                 # projected lifecycle phase at horizon end (one of PHASES)
    confidence: str                   # 'high' | 'medium' | 'low' -- from forecast skill vs naive,
                                      # relative 80% band width and sample size


@dataclass
class TrendReport:
    """Everything known about one series: past, present and future."""

    name: str
    analysis: AnalysisResult
    forecast: ForecastResult
    outlook: Outlook
    narrative: str = ""
    narrative_source: str = "rule"    # 'rule' | 'llm'
