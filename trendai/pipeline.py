"""End-to-end pipeline: raw series -> past/present analysis -> forecast -> story.

    >>> from trendai import run, sample
    >>> report = run(sample("emerging"), horizon=12)
    >>> report.analysis.phase.label, report.outlook.direction
"""

from __future__ import annotations

import datetime as _dt
import json
import math
import warnings
from typing import Any, Iterable, Mapping, Optional

import numpy as np
import pandas as pd
from pandas.errors import OutOfBoundsDatetime

from .analysis import analyze
from .data import DataError, DataWarning, default_horizon, future_index, prepare
from .forecast import auto_forecast, compute_outlook, is_random_walk, outlook_levels
from .narrative import direction_label, generate_narrative
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

    The forecast is told when the analysis found a driftless random walk
    (:func:`trendai.forecast.is_random_walk`), so trend-extrapolating models
    are not offered there.

    Raises :class:`trendai.data.DataError` (Korean message) when the input
    cannot be used (no date index, fewer than 2 observations, a horizon that
    is not a whole number >= 1 (``inf`` and ``nan`` included), ``horizon >
    MAX_HORIZON`` or forecast dates beyond pandas' datetime range).
    Data issues that were worked around (interpolated gaps, weekday-only data,
    unreadable values) and a horizon of more than 5x the history -- the
    default horizon included -- are reported as :class:`trendai.data.DataWarning`.
    """
    if horizon is not None:
        try:
            h_int = int(horizon)
        except (TypeError, ValueError, OverflowError):
            raise DataError(f"예측 기간(horizon)은 정수여야 합니다: {horizon!r}") from None
        if h_int != horizon or h_int < 1:
            raise DataError(f"예측 기간(horizon)은 1 이상의 정수여야 합니다: {horizon!r}")
        if h_int > MAX_HORIZON:
            raise DataError(f"예측 기간(horizon)이 너무 깁니다: {h_int:,}기간 "
                            f"(최대 {MAX_HORIZON:,}기간까지 지원합니다).")
        horizon = h_int
    s, f = prepare(series, freq=freq, name=name)
    if horizon is None:
        # resolve the default here, so that it gets the same checks and warnings as an explicit
        # horizon (a 3-point daily series silently got a 30-day forecast, 10x its history)
        horizon = default_horizon(f)
    _check_horizon(s, f, horizon)
    display_name = str(name if name is not None else (s.name if s.name is not None else "series"))
    analysis = analyze(s, freq=f, name=display_name)
    forecast = auto_forecast(analysis.series, horizon=horizon, freq=f, levels=tuple(levels),
                             candidates=candidates, random_walk=is_random_walk(analysis))
    outlook = compute_outlook(analysis, forecast)
    text, source = generate_narrative(analysis, forecast, outlook, use_llm=use_llm, lang=lang,
                                      model=model, client=client)
    return TrendReport(name=display_name, analysis=analysis, forecast=forecast, outlook=outlook,
                       narrative=text, narrative_source=source)


#: longest horizon run() accepts (forecasting time grows faster than linearly with it)
MAX_HORIZON = 10_000
#: a horizon longer than this many times the history gets a DataWarning
_LONG_HORIZON_RATIO = 5


def _check_horizon(s: pd.Series, freq: str, horizon: int) -> None:
    """Korean DataError if the forecast dates leave pandas' datetime range; DataWarning if the
    horizon is far longer than the history."""
    try:
        end = future_index(s.index[-1], freq, horizon)[-1]
    except (OutOfBoundsDatetime, OverflowError, ValueError):
        end = None
    if end is None or pd.isna(end):
        raise DataError(f"예측 기간(horizon) {horizon:,}기간은 날짜로 표현할 수 있는 범위를 넘습니다. "
                        f"예측 기간을 줄여 주세요.")
    n = len(s)
    if horizon > _LONG_HORIZON_RATIO * n:
        warnings.warn(f"예측 기간({horizon:,}기간)이 데이터 길이({n:,}개)의 {_LONG_HORIZON_RATIO}배를 "
                      f"넘습니다. 이렇게 먼 미래는 과거 패턴만으로 추정하기 어려워 불확실성이 매우 큽니다.",
                      DataWarning, stacklevel=3)


class SeriesFailedWarning(UserWarning):
    """A series in :func:`run_many` could not be analysed and was skipped."""


def run_many(series_map: Mapping[str, pd.Series], errors: Optional[dict[str, Exception]] = None,
             **kwargs: Any) -> dict[str, TrendReport]:
    """Run :func:`run` for every series in ``series_map`` (name -> series).

    Series that fail (e.g. too short) are skipped with a
    :class:`SeriesFailedWarning` instead of aborting the whole batch.  Pass a
    dict as ``errors`` to also receive ``{name: exception}`` for the skipped
    series.  Keyword arguments are forwarded to :func:`run`.
    """
    out: dict[str, TrendReport] = {}
    for nm, s in series_map.items():
        try:
            out[nm] = run(s, name=nm, **kwargs)
        except Exception as exc:  # noqa: BLE001 - batch mode must keep going
            if errors is not None:
                errors[str(nm)] = exc
            detail = str(exc) if isinstance(exc, DataError) else f"{type(exc).__name__}: {exc}"
            prefix = f"'{nm}': "
            if detail.startswith(prefix):
                detail = detail[len(prefix):]
            warnings.warn(f"'{nm}' 분석 실패 — {detail}", SeriesFailedWarning, stacklevel=2)
    return out


# --------------------------------------------------------------------------- JSON export


#: significant digits of the floats in the JSON export: every decimal input of up to 15
#: significant digits (a 13-digit amount in won) round-trips exactly, while float noise such
#: as 0.30000000000000004 still becomes 0.3
_JSON_DIGITS = 15


def _date_str(d: Any) -> str:
    """``YYYY-MM-DD`` of a timestamp / date without ``strftime`` (which refuses years after 9999,
    dates that run() accepts for long yearly horizons)."""
    return f"{d.year:04d}-{d.month:02d}-{d.day:02d}"


def _clean(obj: Any) -> Any:
    """Recursively convert numpy/pandas objects into JSON-serialisable values.

    Floats keep 15 significant digits (relative precision, so values of any
    magnitude survive); NaN / inf become ``None``; dates become ``YYYY-MM-DD``.
    """
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, np.ndarray)):
        return [_clean(v) for v in obj]
    if obj is None or obj is pd.NaT:
        return None
    if isinstance(obj, (pd.Timestamp, _dt.date)):
        return _date_str(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        # significant digits, not decimals: round(f, 6) zeroed a 1e-7 series and merged the
        # bands of a 1e-5 price; 15 digits keep every value while dropping float noise
        return float(f"{f:.{_JSON_DIGITS}g}") if math.isfinite(f) else None
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def _series_records(s: pd.Series) -> list[dict[str, Any]]:
    return [{"date": _date_str(d), "value": float(v)} for d, v in s.items()]


def _outlook_extras(report: TrendReport) -> dict[str, Any]:
    """What the human surfaces say about the outlook beyond the raw ``Outlook`` fields.

    * ``direction_label``: ``{'ko', 'en'}`` word of :func:`narrative.direction_label`
      (``'flat'`` is ``보합`` or ``방향 불확실`` depending on the change);
    * ``expected_change_undefined``: why ``expected_change_pct`` is ``None``
      (``forecast.outlook_levels(...).undefined``), ``None`` when it is a number;
    * ``compared_levels``: ``{'current', 'future'}``, the two levels the expected
      change and ``prob_increase`` compare.

    The same fields and rules as ``narrative.build_llm_payload``'s outlook.
    """
    o = report.outlook
    try:
        lv = outlook_levels(report.analysis, report.forecast)
    except (AttributeError, TypeError, ValueError):   # hand-made or partial report
        lv = None
    undefined = lv.undefined if lv is not None else None
    return {
        "direction_label": {lg: direction_label(o.direction, o.expected_change_pct, lg, undefined)
                            for lg in ("ko", "en")},
        "expected_change_undefined": undefined,
        "compared_levels": {"current": lv.current if lv is not None else None,
                            "future": lv.future if lv is not None else None},
    }


def report_to_dict(report: TrendReport) -> dict[str, Any]:
    """Full machine-readable export of a report (history, trend, forecast, findings).

    ``outlook`` holds the :class:`~trendai.types.Outlook` fields plus
    ``direction_label`` (``{'ko', 'en'}``: the word every human surface shows,
    e.g. ``방향 불확실`` for a ``'flat'`` outlook with a large but uncertain
    change), ``expected_change_undefined`` (why ``expected_change_pct`` is
    ``None``) and ``compared_levels`` (``{'current', 'future'}``, see
    :func:`trendai.forecast.outlook_levels`).
    """
    a, fc, o = report.analysis, report.forecast, report.outlook
    m = a.metrics
    fc_frame = fc.to_frame()
    forecast_rows = []
    for d, row in fc_frame.iterrows():
        rec = {"date": _date_str(d)}
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
            "skill": getattr(fc, "skill", None),
            "nonnegative": fc.nonnegative,
            "table": forecast_rows,
            "models": [ms.__dict__ for ms in fc.model_scores],
        },
        "outlook": {**{k: getattr(o, k) for k in o.__dataclass_fields__}, **_outlook_extras(report)},
        "narrative": report.narrative,
        "narrative_source": report.narrative_source,
    }
    return _clean(payload)


def report_to_json(report: TrendReport, indent: int = 2) -> str:
    """:func:`report_to_dict` as a UTF-8-friendly JSON string (NaN/inf become ``null``)."""
    return json.dumps(report_to_dict(report), ensure_ascii=False, indent=indent)


def reports_to_json(reports: Iterable[TrendReport], indent: int = 2) -> str:
    """A JSON array of :func:`report_to_dict` for several reports."""
    return json.dumps([report_to_dict(r) for r in reports], ensure_ascii=False, indent=indent)
