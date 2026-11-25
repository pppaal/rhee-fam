"""E.16 hooks in existing modules: ``pipeline.run(on_stage=...)``, ``horizon_long`` through ``data.emit_notice`` and
``narrative.llm_narrative_from_payload``."""

from __future__ import annotations

import types
import warnings

import numpy as np
import pandas as pd
import pytest

from trendai import narrative, pipeline
from trendai.data import DataWarning, collect_notices
from trendai.samples import sample


def _short_series(n: int = 12) -> pd.Series:
    idx = pd.date_range("2024-01-01", periods=n, freq="MS")
    return pd.Series(np.linspace(10.0, 20.0, n) + np.sin(np.arange(n)), index=idx, name="x")


def test_run_on_stage_order():
    seen: list[str] = []
    pipeline.run(sample("growing"), horizon=6, on_stage=seen.append)
    assert seen == ["prepare", "analyze", "forecast", "outlook", "narrative"]


def test_run_on_stage_exception_propagates():
    class Stop(Exception):
        pass

    def hook(name: str) -> None:
        if name == "forecast":
            raise Stop(name)

    with pytest.raises(Stop):
        pipeline.run(sample("growing"), horizon=6, on_stage=hook)


def test_run_output_unchanged_without_hook():
    plain = pipeline.report_to_dict(pipeline.run(sample("seasonal"), horizon=6))
    hooked = pipeline.report_to_dict(pipeline.run(sample("seasonal"), horizon=6, on_stage=lambda _n: None))
    assert plain == hooked


def test_horizon_long_unchanged_outside_collect():
    with pytest.warns(DataWarning) as record:
        pipeline.run(_short_series(), horizon=100)
    hits = [w for w in record if getattr(w.message, "code", None) == "horizon_long"]
    assert len(hits) == 1
    w = hits[0]
    assert str(w.message).startswith("예측 기간(100기간)이 데이터 길이(12개)의 5배를 넘습니다.")
    assert w.filename == __file__        # the reported caller line is the caller of run(), as before


def test_horizon_long_collected_inside_collect():
    with warnings.catch_warnings(record=True) as escaped:
        warnings.simplefilter("always")
        with collect_notices() as notes:
            pipeline.run(_short_series(), horizon=100)
    assert not [w for w in escaped if issubclass(w.category, DataWarning)]
    assert "horizon_long" in [getattr(n, "code", None) for n in notes]


class _FakeClient:
    """Records the ``beta.messages.create`` call of ``narrative._call_claude``."""

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.beta = types.SimpleNamespace(messages=types.SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        block = types.SimpleNamespace(type="text", text="## 과거: 어떻게 여기까지 왔나\n- 테스트")
        return types.SimpleNamespace(content=[block], stop_reason="end_turn")


@pytest.mark.parametrize("lang", ["ko", "en"])
def test_llm_narrative_from_payload_matches(lang):
    pytest.importorskip("anthropic")
    report = pipeline.run(sample("growing"), horizon=6)
    a, f, o = report.analysis, report.forecast, report.outlook
    direct, stored = _FakeClient(), _FakeClient()
    text1 = narrative.llm_narrative(a, f, o, lang=lang, model="claude-test", client=direct)
    payload = narrative.build_llm_payload(a, f, o)
    text2 = narrative.llm_narrative_from_payload(payload, lang, model="claude-test", client=stored)
    assert text1 == text2
    assert direct.calls == stored.calls
    assert direct.calls[0]["model"] == "claude-test"
    assert "llm_narrative_from_payload" in narrative.__all__
