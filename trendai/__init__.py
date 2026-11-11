"""trendai — 과거에서 현재, 미래까지: 시계열 트렌드 분석·예측 AI.

Quick start::

    from trendai import run, sample, load_table
    report = run(sample("emerging"), horizon=12)
    print(report.analysis.phase.label_ko, report.outlook.expected_change_pct)
    print(report.narrative)

Problems with the input raise :class:`DataError` (Korean message); data that
was repaired (interpolated gaps, unreadable cells, weekday-only data) is
reported with a :class:`DataWarning`, interpretation notes with a
:class:`DataNotice`.

Public names are imported lazily so that ``import trendai`` stays fast and
optional pieces (e.g. the Claude SDK) are only needed when used.
"""

from __future__ import annotations

import importlib
from typing import Any

__version__ = "0.1.0"

_LAZY = {
    # pipeline
    "run": ("pipeline", "run"),
    "run_many": ("pipeline", "run_many"),
    "report_to_dict": ("pipeline", "report_to_dict"),
    "report_to_json": ("pipeline", "report_to_json"),
    "reports_to_json": ("pipeline", "reports_to_json"),
    # building blocks
    "analyze": ("analysis", "analyze"),
    "auto_forecast": ("forecast", "auto_forecast"),
    "compute_outlook": ("forecast", "compute_outlook"),
    "generate_narrative": ("narrative", "generate_narrative"),
    "ask": ("narrative", "ask"),
    "rank_trends": ("compare", "rank_trends"),
    "rising_score": ("compare", "rising_score"),
    "render_html": ("report", "render_html"),
    "render_text": ("report", "render_text"),
    "render_compare_html": ("report", "render_compare_html"),
    # data
    "load_table": ("data", "load_table"),
    "prepare": ("data", "prepare"),
    "from_values": ("data", "from_values"),
    "DataError": ("data", "DataError"),
    "DataWarning": ("data", "DataWarning"),
    "DataNotice": ("data", "DataNotice"),
    "sample": ("samples", "sample"),
    "sample_set": ("samples", "sample_set"),
    "list_samples": ("samples", "list_samples"),
    # types
    "TrendReport": ("types", "TrendReport"),
}

__all__ = ["__version__", *_LAZY]


def __getattr__(name: str) -> Any:
    try:
        mod_name, attr = _LAZY[name]
    except KeyError:
        raise AttributeError(f"module 'trendai' has no attribute {name!r}") from None
    value = getattr(importlib.import_module(f".{mod_name}", __name__), attr)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(__all__)
