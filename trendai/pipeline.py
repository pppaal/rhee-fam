"""End-to-end pipeline: raw series -> past/present analysis -> forecast -> story.

    >>> from trendai import run, sample
    >>> report = run(sample("emerging"), horizon=12)
    >>> report.analysis.phase.label, report.outlook.direction
"""

from __future__ import annotations

import json
import math
from typing import Any, Iterable, Mapping, Optional

import numpy as np
import pandas as pd

from .analysis import analyze
from .data import prepare
from .forecast import auto_forecast, compute_outlook
from .narrative import generate_narrative
from .types import PHASE_KO, TrendReport

DEFAULT_LLM_MODEL = "claude-opus-5-5"


def run(series: pd.Series, name: Optional[str] = None, freq: Optional[str] = None,
        horizon: Optional[int] = None, levels: tuple[int, ...] = (80, 95),
        use_llm: bool = False, lang: str = "ko", model: str = DEFAULT_LLM_MODEL,
        candidates: Optional[list[str]] = None, client: Any = None) -> TrendReport:
    """Analyse one series end to end and return a :class:`TrendReport`.

    Parameters
    ----------
    series:
        Raw time series (any datetime-like index; irregular / unsorted / NaN ok).
    name:
        Display name (defaults to ``series.name``).
    freq:
        Force a frequency code ``'D' 'W' 'M' 'Q' 'Y'`` (auto-detected otherwise).
    horizon:
        Number of future periods to forecast (default: about one year/cycle).
    use_llm:
        Ask Claude to write the interpretation (falls back to the rule-based
        narrative when the SDK/credentials are unavailable).
    """
    s, f = prepare(series, freq=freq, name=name)
    display_name = str(name if name is not None else (s.name if s.name is not None else "series"))
    analysis = analyze(s, freq=f, name=display_name)
    forecast = auto_forecast(analysis.series, horizon=horizon, freq=f, levels=tuple(levels),
                             candidates=candidates)
    outlook = compute_outlook(analysis, forecast)
    text, source = generate_narrative(analysis, forecast, outlook, use_llm=use_llm, lang=lang,
                                      model=model, client=client)
    return TrendReport(name=display_name, analysis=analysis, forecast=forecast, outlook=outlook,
                       narrative=text, narrative_source=source)


def run_many(series_map: Mapping[str, pd.Series], **kwargs: Any) -> dict[str, TrendReport]:
    """Run :func:`run` for every series in ``series_map`` (name -> series).

    Series that fail (e.g. too short) are skipped with a warning instead of
    aborting the whole batch.
    """
    import warnings

    out: dict[str, TrendReport] = {}
    for nm, s in series_map.items():
        try:
            out[nm] = run(s, name=nm, **kwargs)
        except Exception as exc:  # noqa: BLE001 - batch mode must keep going
            warnings.warn(f"{nm}: 분석 실패 ({type(exc).__name__}: {exc})", stacklevel=2)
    return out


# --------------------------------------------------------------------------- JSON export


def _clean(obj: Any) -> Any:
    """Recursively convert numpy/pandas objects into JSON-serialisable values."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (pd.Timestamp,)):
        return obj.strftime("%Y-%m-%d")
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return round(f, 6) if math.isfinite(f) else None
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def _series_records(s: pd.Series) -> list[dict[str, Any]]:
    return [{"date": d.strftime("%Y-%m-%d"), "value": float(v)} for d, v in s.items()]


def report_to_dict(report: TrendReport) -> dict[str, Any]:
    """Full machine-readable export of a report (history, trend, forecast, findings)."""
    a, fc, o = report.analysis, report.forecast, report.outlook
    m = a.metrics
    fc_frame = fc.to_frame()
    forecast_rows = []
    for d, row in fc_frame.iterrows():
        rec = {"date": d.strftime("%Y-%m-%d")}
        rec.update({k: float(v) for k, v in row.items()})
        forecast_rows.append(rec)
    payload = {
        "name": report.name,
        "freq": a.freq,
        "season_length": a.season_length,
        "history": _series_records(a.series),
        "trend": _series_records(a.decomposition.trend),
        "metrics": {k: getattr(m, k) for k in m.__dataclass_fields__},
        "phase": {"label": a.phase.label, "label_ko": PHASE_KO.get(a.phase.label, a.phase.label),
                  "confidence": a.phase.confidence, "reasons": list(a.phase.reasons)},
        "changepoints": [cp.__dict__ for cp in a.changepoints],
        "anomalies": [an.__dict__ for an in a.anomalies],
        "forecast": {
            "horizon": fc.horizon,
            "levels": list(fc.levels),
            "method": fc.method,
            "backtest_mase": fc.backtest_mase,
            "nonnegative": fc.nonnegative,
            "table": forecast_rows,
            "models": [ms.__dict__ for ms in fc.model_scores],
        },
        "outlook": {k: getattr(o, k) for k in o.__dataclass_fields__},
        "narrative": report.narrative,
        "narrative_source": report.narrative_source,
    }
    return _clean(payload)


def report_to_json(report: TrendReport, indent: int = 2) -> str:
    return json.dumps(report_to_dict(report), ensure_ascii=False, indent=indent)


def reports_to_json(reports: Iterable[TrendReport], indent: int = 2) -> str:
    return json.dumps([report_to_dict(r) for r in reports], ensure_ascii=False, indent=indent)
