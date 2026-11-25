"""Series and compare view models (docs/WEB_SPEC.md D.2-D.5) built from real ``TrendReport`` objects.

Every VM built here is validated against ``trendai/web/vm.schema.json`` with a small stdlib validator. B.3.3 puts
the shared validator in ``trendai.web.testing`` (WP1-server); this file keeps a private copy of the same subset so it
runs on its own.
"""

from __future__ import annotations

import copy
import dataclasses
import functools
import json
import re
from pathlib import Path
from typing import Any, Iterator, Optional

import numpy as np
import pandas as pd
import pytest

from trendai import narrative as N
from trendai import pipeline, samples
from trendai.compare import rank_trends
from trendai.data import DataNotice
from trendai.messages import join, msg, render_web
from trendai.web import fmt
from trendai.web import texts as T
from trendai.web import viewmodel as vm

AS_OF = "2026-11-25T00:12:00Z"
SCHEMA = json.loads(Path(vm.__file__).with_name("vm.schema.json").read_text(encoding="utf-8"))

SERIES_KEYS = ["schema", "lang", "index", "name", "terms", "slot", "icon", "flags", "source", "hero", "phase",
               "tiles", "chart", "past", "future", "narrative", "caveats", "summary_line", "exports", "notices"]
TILE_FIELDS = ["key", "label", "value", "icon", "tone", "sub", "tooltip", "undefined_reason", "value_raw"]
COMPARE_KEYS = ["count", "slots", "value_kind", "source_line", "axis_label", "axis_note", "scale", "ranking",
                "failed", "score_note", "spark_note", "notices"]


# ------------------------------------------------------------------------------------------------ schema validator

_TYPES = {
    "object": lambda v: isinstance(v, dict),
    "array": lambda v: isinstance(v, list),
    "string": lambda v: isinstance(v, str),
    "boolean": lambda v: isinstance(v, bool),
    "null": lambda v: v is None,
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
}


def validate(instance: Any, schema: dict, root: dict = SCHEMA, path: str = "$") -> list[str]:
    """Errors of ``instance`` against the B.3.3 JSON-Schema subset (empty list: valid). Null rule: a null instance
    is valid when ``type`` lists ``"null"``, and no other keyword is checked."""
    errors: list[str] = []
    types = schema.get("type")
    if types is not None:
        names = [types] if isinstance(types, str) else list(types)
        if instance is None and "null" in names:
            return errors
        if not any(_TYPES[n](instance) for n in names):
            return [f"{path}: expected {names}, got {type(instance).__name__}"]
    if "$ref" in schema:
        name = schema["$ref"].split("/")[-1]
        errors += validate(instance, root["$defs"][name], root, path)
    if "enum" in schema and instance not in schema["enum"]:
        errors.append(f"{path}: {instance!r} not in {schema['enum']}")
    if isinstance(instance, dict):
        props = schema.get("properties", {})
        for key in schema.get("required", []):
            if key not in instance:
                errors.append(f"{path}: missing {key!r}")
        if schema.get("additionalProperties") is False:
            for key in instance:
                if key not in props:
                    errors.append(f"{path}: unexpected {key!r}")
        for key, sub in props.items():
            if key in instance:
                errors += validate(instance[key], sub, root, f"{path}.{key}")
    if isinstance(instance, list):
        if "minItems" in schema and len(instance) < schema["minItems"]:
            errors.append(f"{path}: fewer than {schema['minItems']} items")
        if "maxItems" in schema and len(instance) > schema["maxItems"]:
            errors.append(f"{path}: more than {schema['maxItems']} items")
        if "items" in schema:
            for i, item in enumerate(instance):
                errors += validate(item, schema["items"], root, f"{path}[{i}]")
    return errors


def assert_valid(instance: Any, entry: str) -> None:
    errors = validate(instance, {"$ref": f"#/$defs/{entry}"})
    assert not errors, "\n".join(errors[:20])
    json.dumps(instance, ensure_ascii=False, allow_nan=False)   # strict JSON, as every response is dumped


def strings(obj: Any) -> Iterator[str]:
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from strings(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from strings(v)


# ------------------------------------------------------------------------------------------------ report builders

def _weekly_pattern(n: int = 200, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-01-01", periods=n, freq="D")
    return pd.Series(100 + 10 * np.sin(2 * np.pi * np.arange(n) / 7) + rng.normal(0, 1, n), index=idx)


def _naver_like(peak: float, seed: int, n: int = 156) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-11-27", periods=n, freq="W-MON")
    base = 0.5 + 0.5 * np.sin(2 * np.pi * np.arange(n) / 52.0) + np.linspace(0, 0.4, n) + rng.normal(0, 0.05, n)
    base = np.clip(base, 0.01, None)
    return pd.Series(base / base.max() * peak, index=idx)


SERIES = {
    "growing": lambda: samples.sample("growing"),
    "seasonal": lambda: samples.sample("seasonal"),
    "fad": lambda: samples.sample("fad"),
    "emerging": lambda: samples.sample("emerging"),
    "peaking": lambda: samples.sample("peaking"),
    "declining": lambda: samples.sample("declining"),
    "reviving": lambda: samples.sample("reviving"),
    "allzero": lambda: pd.Series(np.zeros(36), index=pd.date_range("2020-01-01", periods=36, freq="MS")),
    "negative": lambda: pd.Series(-50 + np.cumsum(np.random.default_rng(1).normal(0, 1, 36)),
                                  index=pd.date_range("2020-01-01", periods=36, freq="MS")),
    "daily_weekly": _weekly_pattern,
    "daily_short": lambda: pd.Series(np.arange(13, dtype=float) + 5,
                                     index=pd.date_range("2025-01-01", periods=13, freq="D")),
    "naver_big": lambda: _naver_like(100.0, 1),
    "naver_small": lambda: _naver_like(8.0, 2),
    "naver_tiny": lambda: _naver_like(4.0, 3),
    "quarterly": lambda: pd.Series(100 + np.arange(24) * 2.0 + np.tile([5, -3, 8, -10], 6),
                                   index=pd.date_range("2018-01-01", periods=24, freq="QS")),
}


@functools.lru_cache(maxsize=None)
def report_of(key: str, name: Optional[str] = None):
    return pipeline.run(SERIES[key](), name=name or key)


@functools.lru_cache(maxsize=None)
def english(key: str, name: Optional[str] = None) -> str:
    r = report_of(key, name)
    return N.generate_narrative(r.analysis, r.forecast, r.outlook, lang="en")[0]


def meta_of(report, kind: str = "sample", *, value_kind: Optional[str] = None, as_of: str = AS_OF,
            terms: Optional[tuple] = None, **kw) -> vm.SourceMeta:
    a = report.analysis
    kinds = {"naver": "relative_index", "wiki": "pageviews", "google_trends_upload": "relative_index_gt"}
    vk = value_kind or kinds.get(kind, "raw")
    extra = dict(kw)
    if kind == "naver":
        extra.setdefault("naver_endpoint", "hub")
    if kind == "wiki":
        extra.setdefault("wiki_lang", "ko")
        extra.setdefault("wiki_title", report.name)
    if kind == "sample":
        extra.setdefault("synthetic", True)
    return vm.SourceMeta(kind=kind, value_kind=vk, freq=a.freq, start=vm._iso(a.series.index[0]),
                         end=vm._iso(a.series.index[-1]), as_of=as_of, terms=terms or (report.name,), **extra)


def make_input(report, meta: vm.SourceMeta, *, index: int = 0, request_size: int = 1, ai: bool = True,
               xlsx: bool = True, notices: Optional[dict] = None, en_text: Optional[str] = None) -> vm.SeriesInput:
    if en_text is None:
        en_text = N.generate_narrative(report.analysis, report.forecast, report.outlook, lang="en")[0]
    return vm.SeriesInput(index=index, name=report.name, report=report, source=meta,
                          narratives={"ko": report.narrative, "en": en_text},
                          notices=notices or {"ko": [], "en": []}, request_size=request_size, ai_enabled=ai,
                          xlsx_available=xlsx, slot=vm.provisional_slot(index, request_size))


def build(inp: vm.SeriesInput, lang: str = "ko") -> dict:
    out = vm.build_series_vm(inp, lang)
    assert_valid(out, "seriesVM")
    return out


def sample_vm(key: str, lang: str = "ko") -> dict:
    r = report_of(key)
    return build(make_input(r, meta_of(r, "sample"), en_text=english(key)), lang)


def compare_of(inputs: list, lang: str = "ko") -> tuple[dict, dict]:
    reports = {i.name: i.report for i in inputs}
    ranking = rank_trends(reports)
    name_to_index = {i.name: i.index for i in inputs}
    ranked = [name_to_index[n] for n in ranking["name"]]
    slots = vm.assign_slots(ranked, inputs[0].request_size)
    vms = {i.index: vm.set_slot(build(i, lang), slots[i.index]) for i in inputs}
    out = vm.build_compare_vm(inputs, vms, ranking, name_to_index, slots, [], lang)
    assert_valid(out, "compareVM")
    return out, vms


def with_forecast(report, **changes):
    return dataclasses.replace(report, forecast=dataclasses.replace(report.forecast, **changes))


def with_outlook(report, **changes):
    return dataclasses.replace(report, outlook=dataclasses.replace(report.outlook, **changes))


# ------------------------------------------------------------------------------------------------------------ tests

def test_vm_matches_schema():
    for key in ("growing", "seasonal", "fad", "allzero", "negative", "daily_weekly", "daily_short", "quarterly"):
        for lang in vm.LANGS:
            v = build(make_input(report_of(key), meta_of(report_of(key), "upload", filename="data.csv")), lang)
            assert list(v) == SERIES_KEYS
            assert v["schema"] == vm.WEB_VM_SCHEMA and v["lang"] == lang
    inputs = [make_input(report_of(k), meta_of(report_of(k), "wiki"), index=i, request_size=3)
              for i, k in enumerate(("growing", "fad", "seasonal"))]
    for lang in vm.LANGS:
        cmp, _ = compare_of(inputs, lang)
        assert list(cmp) == COMPARE_KEYS


def test_schema_rejects_wrong_shapes():
    v = sample_vm("growing")
    broken = copy.deepcopy(v)
    broken["tiles"] = broken["tiles"][:5]
    assert validate(broken, {"$ref": "#/$defs/seriesVM"})
    broken = copy.deepcopy(v)
    broken["hero"]["extra"] = 1
    assert validate(broken, {"$ref": "#/$defs/seriesVM"})
    broken = copy.deepcopy(v)
    broken["chart"]["display"] = "strong"
    assert validate(broken, {"$ref": "#/$defs/seriesVM"})
    broken = copy.deepcopy(v)
    broken["index"] = True                      # a bool is never an integer
    assert validate(broken, {"$ref": "#/$defs/seriesVM"})


def test_tiles_exact_keys_order():
    seasonal = sample_vm("seasonal")
    assert [t["key"] for t in seasonal["tiles"]] == ["phase", "recent_change", "momentum", "expected_change",
                                                     "prob_increase", "peak_season"]
    fad = sample_vm("fad", "en")
    assert [t["key"] for t in fad["tiles"]] == ["phase", "recent_change", "momentum", "expected_change",
                                                "prob_increase", "volatility"]
    for tile in seasonal["tiles"] + fad["tiles"]:
        assert list(tile) == TILE_FIELDS
        assert tile["tone"] in ("up", "down", "neutral")
        assert tile["label"] == T.t(T.TILE_LABELS, tile["key"], "en" if tile in fad["tiles"] else "ko")
    tips = {t["key"]: t["tooltip"] for t in seasonal["tiles"]}
    assert tips["recent_change"] == "최근 한 주기(12개월) 동안 추세선이 얼마나 변했는지예요. 계절 변동은 뺀 값이에요."
    assert tips["prob_increase"].startswith(T.TILE_HELP["prob_increase"][0] + " 정확한 값: ")
    assert seasonal["flags"]["seasonal"] is True and fad["flags"]["seasonal"] is False


def test_peak_season_matches_narrative():
    seasonal = sample_vm("seasonal")
    peak = seasonal["tiles"][5]
    trough = re.search(r"보통 \S+에 높고 (\S+)에 낮아요", seasonal["past"]["seasonality"]["text"]).group(1)
    assert f"대체로 {peak['value']}에 높고 {trough}에 낮은" in report_of("seasonal").narrative
    assert peak["sub"].startswith(f"보통 {peak['value']} 정점")
    en = sample_vm("seasonal", "en")
    month = fmt.MONTHS_EN.index(en["tiles"][5]["value"])
    full = ("January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
            "November", "December")[month]
    assert f"highest in {full}" in english("seasonal")

    r = report_of("daily_weekly")
    assert r.analysis.decomposition.period == 7
    d = build(make_input(r, meta_of(r, "upload", filename="d.csv")))
    tile = d["tiles"][5]
    assert tile["key"] == "peak_season"
    m = re.fullmatch(r"요일 패턴 · 가장 낮은 날 (\S+)", tile["sub"])
    assert m, tile["sub"]
    assert f"대체로 {tile['value']}에 높고 {m.group(1)}에 낮은" in r.narrative
    d_en = build(make_input(r, meta_of(r, "upload", filename="d.csv")), "en")
    assert d_en["tiles"][5]["value"].endswith("days")                          # "Fridays", as the narrative says
    assert d_en["tiles"][5]["sub"].startswith("Weekday pattern · lowest on ")

    # daily data with another period (or none) -> no peak season, the volatility tile instead
    other = dataclasses.replace(r, analysis=dataclasses.replace(
        r.analysis, decomposition=dataclasses.replace(r.analysis.decomposition, period=30)))
    assert build(make_input(other, meta_of(other, "upload")))["tiles"][5]["key"] == "volatility"
    short = report_of("daily_short")
    assert short.analysis.decomposition.period != 7
    assert build(make_input(short, meta_of(short, "upload")))["tiles"][5]["key"] == "volatility"


def test_volatility_tile_without_season():
    r = report_of("fad")
    assert r.analysis.metrics.seasonal_strength < 0.3
    tile = sample_vm("fad")["tiles"][5]
    assert tile["key"] == "volatility" and tile["label"] == "변동성"
    cv = r.analysis.metrics.volatility
    assert tile["value"] == ("낮음" if cv < 0.05 else "보통" if cv < 0.15 else "높음")
    assert tile["sub"] == f"잔차 변동 {fmt.round_half_away(cv * 100, 1)}%"
    assert tile["value_raw"] == pytest.approx(cv)
    assert sample_vm("fad", "en")["tiles"][5]["sub"].startswith("residual CV ")


@pytest.mark.parametrize("key, affected", [
    ("allzero", ("recent_change", "momentum", "expected_change", "prob_increase", "volatility")),
    ("negative", ("recent_change", "momentum", "expected_change", "volatility")),
])
def test_undefined_reasons(key, affected):
    r = report_of(key)
    for lang in vm.LANGS:
        v = build(make_input(r, meta_of(r, "upload", filename="x.csv")), lang)
        tiles = {t["key"]: t for t in v["tiles"]}
        for k in affected:
            assert tiles[k]["value"] == "—", (k, tiles[k])
            assert tiles[k]["undefined_reason"], (k, tiles[k])
            assert tiles[k]["value_raw"] is None
        for s in strings(v):
            assert s not in ("None", "nan", "NaN", "null", "undefined", "inf")
            assert not re.search(r"\bNone\b|\bnan\b|\bNaN\b|\binf\b|해당 없음|\['|'\]", s), s
    v = build(make_input(r, meta_of(r, "upload")), "ko")
    if key == "allzero":
        assert v["hero"]["outlook"]["prob"] is None
        assert v["hero"]["outlook"]["line"] == "▶ 보합 · 상승 확률은 계산하지 않음(관측값이 모두 0이고 예측도 0)"
        assert v["flags"]["all_zero"] is True
    assert v["hero"]["change_undefined"]["reason"] == {t["key"]: t for t in v["tiles"]}["expected_change"][
        "undefined_reason"]
    assert v["hero"]["sentence"].endswith(")")       # the compared levels are named


def test_display_rule():
    r = report_of("growing")
    assert r.forecast.skill < 1 and r.outlook.confidence != "low"
    meta = meta_of(r, "sample")
    assert build(make_input(r, meta))["chart"]["display"] == "normal"
    bands = build(make_input(with_forecast(r, skill=1.2), meta))
    assert bands["chart"]["display"] == "bands_only"
    assert bands["hero"]["reference_note"] == "예측력이 낮아 범위만 참고용으로 보여 드려요."
    assert bands["flags"]["reference_only"] is True
    weak = build(make_input(with_outlook(r, confidence="low"), meta))
    assert weak["chart"]["display"] == "weak"
    assert weak["hero"]["reference_note"] == "예측 신뢰도가 낮아 중앙선은 참고용이에요."
    assert build(make_input(with_forecast(r, skill=None), meta))["chart"]["display"] == "weak"
    assert build(make_input(with_forecast(r, skill=1.0), meta))["chart"]["display"] == "bands_only"


def test_weak_rule_everywhere():
    good = report_of("growing", "좋은")
    bad = with_forecast(report_of("fad", "나쁜"), skill=1.2)
    inputs = [make_input(good, meta_of(good, "upload", filename="f.csv"), index=0, request_size=2),
              make_input(bad, meta_of(bad, "upload", filename="f.csv"), index=1, request_size=2)]
    for lang, suffix, marker in (("ko", "· 참고용", "참고용"), ("en", "· for reference", "for reference")):
        cmp, vms = compare_of(inputs, lang)
        weak = vms[1]
        assert weak["chart"]["display"] == "bands_only"
        tiles = {t["key"]: t for t in weak["tiles"]}
        assert tiles["expected_change"]["sub"].endswith(suffix)
        assert tiles["prob_increase"]["sub"].endswith(suffix)
        assert marker in weak["summary_line"]
        assert not vms[0]["tiles"][3]["sub"].endswith(suffix)
        rows = {row["index"]: row for row in cmp["ranking"]}
        assert rows[1]["display"] == "bands_only" and rows[1]["spark"]["forecast"] == []
        assert rows[1]["flags"]["reference_only"] is True
        assert 0 < len(rows[0]["spark"]["forecast"]) <= 52
    assert vms[1]["summary_line"].count("(for reference: did not beat a naive forecast)") == 1


def test_likelihood_and_prob_text_boundaries():
    words = {k: v for k, v in T.LIKELIHOOD_WORDS.items()}
    for p, key in ((0.9, "rise_very_likely"), (0.7, "rise_more_likely"), (0.55, "rise_slightly"),
                   (0.45000001, "even"), (0.45, "fall_slightly"), (0.3, "fall_more_likely"),
                   (0.1, "fall_very_likely")):
        assert fmt.likelihood(p, "ko") == words[key][0]
        assert fmt.likelihood(p, "en") == words[key][1]
    assert fmt.prob_text(0.975) == "95% 이상" and fmt.prob_text(0.97499) == "약 95%"
    assert fmt.prob_text(0.025) == "5% 이하" and fmt.prob_text(0.02501) == "약 5%"
    for key in ("growing", "fad", "seasonal"):
        r = report_of(key)
        v = sample_vm(key)
        p = r.outlook.prob_increase
        tile = v["tiles"][4]
        assert tile["value"] == fmt.prob_text(p, "ko")
        assert v["hero"]["outlook"]["likelihood"] == fmt.likelihood(p, "ko")   # the narrative's raw-P bins
        suffix = " · 참고용" if v["chart"]["display"] != "normal" else ""
        assert tile["sub"] == f"{fmt.likelihood(p, 'ko')} · 예측 변화와 같은 기준{suffix}"
        assert "상승 확률" in v["hero"]["outlook"]["line"]


def test_outlook_line_rules():
    line = vm._outlook_line
    assert line("up", "상승", 0.812, None, "ko") == "▲ 상승 쪽이 우세 (상승 확률 약 80%)"
    assert line("down", "하락", 0.2, None, "ko") == "▼ 하락 쪽이 우세 (상승 확률 약 20%)"
    assert line("down", "downward", 0.2, None, "en") == "▼ Fall more likely (chance of rise about 20%)"
    assert line("down", "하락", 0.02, None, "ko") == "▼ 하락 가능성 매우 높음 (상승 확률 5% 이하)"
    assert line("down", "downward", 0.02, None, "en") == "▼ Fall very likely (chance of rise 5% or less)"
    assert line("flat", "보합", 0.5, None, "ko") == "▶ 보합 · 상승·하락 엇비슷 (상승 확률 약 50%)"
    assert line("flat", "flat", 0.5, None, "en") == "▶ flat · About even (chance of rise about 50%)"
    # the likelihood's side disagrees with the direction: the word stays in front
    assert line("up", "상승", 0.35, None, "ko") == "▲ 상승 · 하락 쪽이 약간 우세 (상승 확률 약 35%)"
    assert line("up", "상승", 0.3, None, "ko") == "▲ 상승 · 하락 쪽이 우세 (상승 확률 약 30%)"     # bin edge: > 0.30
    assert line("flat", "보합", None, "현재 수준이 0", "ko") == "▶ 보합 · 상승 확률은 계산하지 않음(현재 수준이 0)"
    assert line("flat", "flat", None, "current level is 0", "en") == \
        "▶ flat · chance of rise not computed (current level is 0)"
    growing = sample_vm("growing")["hero"]["outlook"]
    assert growing["line"] == f"▲ {growing['likelihood']} (상승 확률 {growing['prob_text']})"
    assert (growing["arrow"], growing["tone"], growing["word"]) == ("▲", "up", "상승")


def test_scenario_rows(monkeypatch):
    calls = []
    real = N.scenario_value

    def spy(value, reference=None, nonnegative=False, lang="ko"):
        calls.append((reference, nonnegative))
        return real(value, reference, nonnegative, lang)

    r = report_of("fad")
    floor = N.scenario_floor(r.analysis, r.forecast)
    monkeypatch.setattr(N, "scenario_value", spy)
    v = sample_vm("fad")
    assert calls and all(nn == floor for _, nn in calls)
    sc = v["future"]["scenarios"]
    last = v["chart"]["forecast"]["dates"][-1]
    assert sc["date_label"] == fmt.date_label(last, "M", "ko")
    assert sc["title"] == f"{sc['date_label']} 한 시점의 값"
    ref = sc["reference"]
    assert ref["text"] == f"비교 기준: {ref['label']} {ref['value_text']}"
    rows = {row["key"]: row for row in sc["rows"]}
    assert [row["label"] for row in sc["rows"]] == ["비관", "기준", "낙관"]
    if floor and r.outlook.scenarios["pessimistic"] <= 0:
        assert rows["pessimistic"]["value_text"] == "0(하한)" and rows["pessimistic"]["pct_text"] is None
        assert rows["pessimistic"]["text"] == "비관 0(하한)"
    base = rows["base"]
    assert base["pct_text"] is not None
    assert base["text"] == f"기준 {base['value_text']} (기준 대비 {base['pct_text']})"
    assert sc["note"].startswith("예측 마지막 시점 한 값의 80% 범위예요.")

    # no expected change -> reference=None for scenario_value, no percentages anywhere
    calls.clear()
    neg = report_of("negative")
    assert neg.outlook.expected_change_pct is None
    nv = build(make_input(neg, meta_of(neg, "upload")))
    assert calls and all(ref is None for ref, _ in calls)
    for row in nv["future"]["scenarios"]["rows"]:
        assert row["pct_text"] is None and "기준 대비" not in row["text"]
        assert row["text"] == f"{row['label']} {row['value_text']}"
    en = sample_vm("fad", "en")["future"]["scenarios"]
    assert en["title"].startswith("Single value at ")
    assert en["reference"]["text"].startswith("Compared with: ")
    assert any(row["text"].endswith("vs. reference)") for row in en["rows"])


def test_reasons_detail_and_dates():
    v = sample_vm("growing")
    reasons = v["phase"]["reasons"]
    mk = [x for x in reasons if x["text"].startswith("Mann-Kendall")]
    assert mk and mk[0]["detail"] == "τ = 0.95, p < 0.001"
    assert mk[0]["text"] == "Mann-Kendall 검정에서 전체 기간의 상승 추세가 유의합니다."
    assert {"text": "현재 추세는 고점(2025.12) 수준입니다.", "detail": None} in reasons
    for x in reasons:
        assert not re.search(r"\d{4}-\d{2}", x["text"])
    en = sample_vm("growing", "en")["phase"]["reasons"]
    assert {"text": "The trend is at its peak (Dec 2025).", "detail": None} in en
    assert any(x["detail"] == "τ = 0.95, p < 0.001" for x in en)
    item = vm._reason_item("2023-03-06에 구조 변화(성장 가속)가 감지되었습니다.", "W", "ko")
    assert item == {"text": "2023.03.06에 구조 변화(성장 가속)가 감지되었습니다.", "detail": None}


def test_narrative_date_links():
    text, links = vm.localize_dates("- **2023-03-06**부터 성장 가속 (2021-11-29 시작)", "W", "ko",
                                    ["2023-03-06", "2023-03-13"])
    assert text == "- **2023.03.06**부터 성장 가속 (2021.11.29 시작)"
    assert links == {"2023.03.06": "2023-03-06"}           # 2021-11-29 is not a chart date: rewritten only
    text, links = vm.localize_dates("2016-01부터 2024-03까지, 정점 2024-03", "M", "ko", ["2016-01-01", "2024-03-01"])
    assert text == "2016.01부터 2024.03까지, 정점 2024.03"
    assert links == {"2016.01": "2016-01-01", "2024.03": "2024-03-01"}
    text, links = vm.localize_dates("2024-Q1에 정점", "Q", "ko", ["2023-10-01", "2024-01-01"])
    assert text == "2024 1분기에 정점" and links == {"2024 1분기": "2024-01-01"}
    text, links = vm.localize_dates("peak in 2024-Q1", "Q", "en", ["2024-01-01"])
    assert text == "peak in Q1 2024" and links == {"Q1 2024": "2024-01-01"}
    text, links = vm.localize_dates("2024년 최고, 2021-01-01 기준", "Y", "ko", ["2021-01-01", "2024-01-01"])
    assert links == {} and text == "2024년 최고, 2021 기준"
    assert vm.localize_dates("2024-03 그대로, 2024-13-45도", "W", "ko", [])[0] == "2024-03 그대로, 2024-13-45도"
    assert vm.localize_dates("Nov 2025-06-30", "D", "en", ["2025-06-30"]) == (
        "Nov Jun 30, 2025", {"Jun 30, 2025": "2025-06-30"})
    v = sample_vm("seasonal")
    dates = set(v["chart"]["dates"]) | set(v["chart"]["forecast"]["dates"])
    assert v["narrative"]["date_links"] and set(v["narrative"]["date_links"].values()) <= dates
    assert not re.search(r"\b\d{4}-\d{2}\b", v["narrative"]["markdown"], re.ASCII)
    for shown in v["narrative"]["date_links"]:
        assert shown in v["narrative"]["markdown"] or shown in v["caveats"]["markdown"]


def test_summary_line():
    v = sample_vm("growing")
    assert v["source"]["short"] == "샘플·가상 데이터"
    assert v["summary_line"].startswith("[trendai] growing(샘플·가상 데이터, 월간): 성장기 · 향후 12개월 평균 +")
    assert "예상(상승 가능성 95% 이상)" in v["summary_line"]
    assert v["summary_line"].endswith(" (데이터 기준 2026.11.25 09:12)")
    seasonal = sample_vm("seasonal")
    assert f" · 보통 {seasonal['tiles'][5]['value']} 정점 (데이터 기준" in seasonal["summary_line"]
    en = sample_vm("growing", "en")
    assert en["summary_line"].startswith("[trendai] growing (sample · synthetic data, monthly): Growing · next ")
    assert en["summary_line"].endswith(" (data as of Nov 25, 2026 09:12 KST)")
    neg = build(make_input(report_of("negative"), meta_of(report_of("negative"), "upload")))
    assert " · 보합 전망" in neg["summary_line"] or " 전망" in neg["summary_line"]
    weak = build(make_input(with_outlook(report_of("growing"), confidence="low"), meta_of(report_of("growing"))))
    assert "(참고용 예측) (데이터 기준" in weak["summary_line"]


def test_y_hint_relative_index():
    r = report_of("naver_small")
    v = build(make_input(r, meta_of(r, "naver")))
    actual = np.asarray(r.analysis.series, dtype=float)
    assert actual.max() == pytest.approx(8.0)
    band_max = max(actual.max(), float(np.max(r.forecast.upper[95])))
    assert v["chart"]["y_hint"]["max"] == pytest.approx(band_max)
    assert v["chart"]["y_hint"]["include_zero"] is True
    assert v["chart"]["y_note"] == "이 조회의 최댓값(100)보다 작은 검색어라 세로축을 확대해 그렸어요."
    big = report_of("naver_big")
    assert build(make_input(big, meta_of(big, "naver")))["chart"]["y_note"] is None
    assert build(make_input(r, meta_of(r, "naver")), "en")["chart"]["y_note"].startswith("This term is small")
    assert v["tiles"][1]["sub"] == "추세 기준 · 전년 동기 대비"


def _without_ko_twins(obj: Any) -> Any:
    """Drop ``label_ko`` / ``message_ko``: fix and error objects carry both languages by design (C.12.1)."""
    if isinstance(obj, dict):
        return {k: _without_ko_twins(v) for k, v in obj.items() if k not in ("label_ko", "message_ko")}
    if isinstance(obj, list):
        return [_without_ko_twins(v) for v in obj]
    return obj


def _assert_no_hangul(obj: Any, names: list[str]) -> None:
    for s in strings(_without_ko_twins(obj)):
        for nm in sorted(names, key=len, reverse=True):
            s = s.replace(nm, "X")
        assert not re.search(r"[가-힣]", s), s


def test_en_vm_has_no_hangul_outside_names():
    _assert_no_hangul(sample_vm("growing", "en"), ["growing"])

    sun = report_of("naver_big", "선크림")
    pad = report_of("naver_small", "패딩")
    filters = dict(device="mo", gender="f", ages=("4", "5"))
    inputs = [make_input(sun, meta_of(sun, "naver", terms=("선크림", "자외선차단제"), **filters), index=0,
                         request_size=2, en_text=english("naver_big", "선크림")),
              make_input(pad, meta_of(pad, "naver", terms=("패딩",), **filters), index=1, request_size=2,
                         en_text=english("naver_small", "패딩"))]
    cmp, vms = compare_of(inputs, "en")
    names = ["선크림", "자외선차단제", "패딩"]
    _assert_no_hangul(cmp, names)
    for v in vms.values():
        _assert_no_hangul(v, names)
        assert v["source"]["filters_label"] == "mobile · female · 25–34"

    sales = report_of("growing", "매출")
    note = DataNotice(join(msg("date_order_ambiguous", label="DD/MM/YYYY (일/월 순)"),
                           msg("dates_unreadable_rows", n=2, examples="'2024-13-01', '없음'")))
    notices = {lang: [vm.notice_from_warning(note, lang, series_index=0),
                      vm.make_notice("data_short", lang, params={"n": 5}, series_index=0, source_kind="upload")]
               for lang in vm.LANGS}
    up = build(make_input(sales, meta_of(sales, "upload", filename="월매출.xlsx", terms=("매출",), unit_label="원"),
                          notices=notices, en_text=english("growing", "매출")), "en")
    assert up["notices"][0]["code"] == "date_order_ambiguous"
    _assert_no_hangul(up, ["매출", "월매출.xlsx", "원", "2024-13-01", "없음"])


def test_phase_reasons_en_translated():
    r = report_of("growing")
    en = sample_vm("growing", "en")["phase"]
    translated = [N.translate_reason(x) for x in r.analysis.phase.reasons]
    assert len(en["reasons"]) == len([x for x in translated if x])
    for item in en["reasons"]:
        assert not re.search(r"[가-힣]", item["text"] + (item["detail"] or ""))
    assert en["name"] == "Growing" and en["name_long"] == N.PHASE_EN["growing"]
    odd = dataclasses.replace(r, analysis=dataclasses.replace(
        r.analysis, phase=dataclasses.replace(r.analysis.phase, reasons=["번역 규칙이 없는 이유입니다."])))
    out = build(make_input(odd, meta_of(odd)), "en")["phase"]["reasons"]
    assert out == [{"text": "See the method page for the rules.", "detail": None}]
    ko = build(make_input(odd, meta_of(odd)), "ko")["phase"]["reasons"]
    assert ko == [{"text": "번역 규칙이 없는 이유입니다.", "detail": None}]


def _shifted(key: str, months: int, name: str, scale: float = 1.0):
    s = SERIES[key]().iloc[months:] * scale
    return pipeline.run(s, name=name)


def test_compare_scale_index_base():
    a = report_of("growing", "A")
    b = _shifted("seasonal", 6, "B", 3.0)
    inputs = [make_input(a, meta_of(a, "upload", filename="t.csv"), index=0, request_size=2),
              make_input(b, meta_of(b, "upload", filename="t.csv"), index=1, request_size=2)]
    cmp, _ = compare_of(inputs)
    scale = cmp["scale"]
    assert scale["default"] == "index" and scale["allowed"]["index"] is True
    base = scale["index_base"]
    start = vm._iso(b.analysis.series.index[0])                       # the latest series start
    end = vm._iso(b.analysis.series.index[11])                        # one season (12 months) from it
    assert (base["start"], base["end"]) == (start, end)
    assert base["label"] == f"각 시계열의 {fmt.range_label(start, end, 'M', 'ko')} 평균을 100으로 맞췄어요"
    win_a = a.analysis.series[(a.analysis.series.index >= start) & (a.analysis.series.index <= end)]
    assert base["values"]["0"] == pytest.approx(float(win_a.mean()))
    assert base["values"]["1"] == pytest.approx(float(b.analysis.series.iloc[:12].mean()))
    assert scale["disabled_reason"]["index"] is None and cmp["notices"] == []

    zero = report_of("allzero", "Z")
    inputs = [make_input(a, meta_of(a, "upload"), index=0, request_size=2),
              make_input(zero, meta_of(zero, "upload"), index=1, request_size=2)]
    cmp, _ = compare_of(inputs)
    assert cmp["scale"]["default"] == "raw" and cmp["scale"]["allowed"]["index"] is False
    assert cmp["scale"]["disabled_reason"]["index"] == render_web("index_unavailable", {}, "ko")
    assert cmp["scale"]["allowed"]["log"] is False
    assert cmp["scale"]["disabled_reason"]["log"] == "0 이하 값이 있어 로그 축을 쓸 수 없어요."
    assert [n["code"] for n in cmp["notices"]] == ["index_unavailable"]
    assert cmp["notices"][0]["series_index"] is None and cmp["notices"][0]["level"] == "info"

    big, small = report_of("naver_big", "N1"), report_of("naver_tiny", "N2")
    inputs = [make_input(big, meta_of(big, "naver"), index=0, request_size=2),
              make_input(small, meta_of(small, "naver"), index=1, request_size=2)]
    for lang, reason in (("ko", "같은 조회 안에서 이미 같은 척도예요."),
                         ("en", "These series already share one scale (same request).")):
        cmp, vms = compare_of(inputs, lang)
        assert cmp["scale"]["index_base"] is None and cmp["scale"]["allowed"]["index"] is False
        assert cmp["scale"]["disabled_reason"]["index"] == reason and cmp["scale"]["default"] == "raw"
        assert cmp["scale"]["log_hint"] is False
    assert cmp["axis_label"] == "relative index (period maximum = 100)"
    assert cmp["axis_note"].startswith("The week with the most searches in this query = 100")
    assert vms[1]["flags"]["low_resolution"] is True and vms[0]["flags"]["low_resolution"] is False


def test_log_hint_for_wiki():
    a = report_of("growing", "Big")
    b = _shifted("seasonal", 0, "Small", 0.01)
    inputs = [make_input(a, meta_of(a, "wiki"), index=0, request_size=2),
              make_input(b, meta_of(b, "wiki"), index=1, request_size=2)]
    cmp, _ = compare_of(inputs)
    assert cmp["scale"]["log_hint"] is True and cmp["scale"]["allowed"]["log"] is True
    assert cmp["scale"]["default"] == "raw" and cmp["scale"]["index_base"] is not None
    hint = cmp["notices"][0]
    assert hint["code"] == "log_hint" and hint["text"] == render_web("log_hint", {"ratio": 20}, "ko")
    assert [f["action"] for f in hint["fixes"]] == ["use_log"]
    assert cmp["axis_label"] == "조회수" and cmp["axis_note"] is None


def test_compare_source_line():
    a = vm.SourceMeta(kind="wiki", value_kind="pageviews", freq="M", start="2018-01-01", end="2025-10-01",
                      as_of="2026-11-25T03:00:00Z", terms=("A",), wiki_lang="ko", wiki_title="A")
    b = dataclasses.replace(a, start="2016-05-01", end="2025-09-01", as_of="2026-11-24T23:30:00Z", terms=("B",))
    assert vm.compare_source_line([a, b], "ko") == (
        "출처: Wikimedia Pageviews · 한국어 위키백과 · 사람 조회만(봇 제외) · 월 단위 · 2016.05 ~ 2025.10 · 조회수 · "
        "데이터 기준 2026.11.25 08:30")
    assert vm.compare_source_line([a, b], "en") == (
        "Source: Wikimedia Pageviews · Korean Wikipedia · human views only (bots excluded) · monthly · "
        "May 2016 – Oct 2025 · page views · data as of Nov 25, 2026 08:30 KST")
    r1, r2 = report_of("growing", "W1"), _shifted("seasonal", 10, "W2")
    inputs = [make_input(r1, meta_of(r1, "wiki", as_of="2026-11-25T01:00:00Z"), index=0, request_size=2),
              make_input(r2, meta_of(r2, "wiki", as_of="2026-11-24T01:00:00Z"), index=1, request_size=2)]
    cmp, _ = compare_of(inputs)
    assert cmp["source_line"].endswith("2016.01 ~ 2025.12 · 조회수 · 데이터 기준 2026.11.24 10:00")


def test_slots_over_five():
    keys = ["emerging", "growing", "peaking", "fad", "declining", "seasonal", "reviving"]
    reports = [report_of(k, f"col{i}") for i, k in enumerate(keys)]
    inputs = [make_input(r, meta_of(r, "upload", filename="wide.csv"), index=i, request_size=7)
              for i, r in enumerate(reports)]
    assert all(i.slot is None for i in inputs)                          # provisional: unknown while N > 5
    assert all(build(i)["slot"] is None for i in inputs[:1])
    cmp, vms = compare_of(inputs)
    ranking = rank_trends({i.name: i.report for i in inputs})
    ranked = [int(n[3:]) for n in ranking["name"]]
    top = sorted(ranked[:5])
    assert cmp["slots"] == {str(i): (top.index(i) + 1 if i in top else None) for i in range(7)}
    assert sum(1 for s in cmp["slots"].values() if s is None) == 2
    for row in cmp["ranking"]:
        assert row["slot"] == cmp["slots"][str(row["index"])]
        assert row["icon"] == (vm.SLOT_ICONS[row["slot"]] if row["slot"] else None)
        assert vms[row["index"]]["slot"] == row["slot"]
    assert [row["rank"] for row in cmp["ranking"]] == list(range(1, 8))
    assert [row["index"] for row in cmp["ranking"]] == ranked
    scores = [row["sort"]["score"] for row in cmp["ranking"]]
    assert scores == sorted(scores, reverse=True)


def test_ranking_rows_follow_series_vm():
    inputs = [make_input(report_of(k, n), meta_of(report_of(k, n), "upload"), index=i, request_size=3)
              for i, (k, n) in enumerate((("growing", "가"), ("fad", "나"), ("negative", "다")))]
    cmp, vms = compare_of(inputs)
    for row in cmp["ranking"]:
        v = vms[row["index"]]
        assert row["name"] == v["name"]
        assert row["outlook"]["line"] == v["hero"]["outlook"]["line"]
        assert row["expected_change"]["basis"] == v["hero"]["basis"]
        assert row["phase"]["name"] == v["phase"]["name"]
        assert row["score"]["text"] == str(fmt.round_half_away(row["score"]["value"], 1))
        hist = len(row["spark"]["history"])
        assert hist == min(len(v["chart"]["actual"]), 2 * 12, 104)
        assert row["recent_change"]["basis"] == "전년 동기 대비"
    neg = next(row for row in cmp["ranking"] if row["name"] == "다")
    assert neg["expected_change"]["text"] == "—" and neg["expected_change"]["undefined_reason"]
    assert neg["momentum"]["text"] == "—" and neg["momentum"]["undefined_reason"]
    grow = next(row for row in cmp["ranking"] if row["name"] == "가")
    assert grow["expected_change"]["text"] == fmt.pct(report_of("growing", "가").outlook.expected_change_pct)
    assert grow["momentum"]["text"].endswith("%p/월")


def test_narrative_caveats_split():
    r = report_of("fad")
    v = sample_vm("fad")
    assert "## 주의할 점" not in v["narrative"]["markdown"]
    body, caveats = vm.split_caveats(r.narrative, "ko")
    assert caveats and caveats.startswith("- ")
    assert v["caveats"]["markdown"] == vm.localize_dates(caveats, "M", "ko", [])[0]
    assert v["narrative"]["markdown"] == vm.localize_dates(body, "M", "ko", v["chart"]["dates"])[0]
    assert v["caveats"]["disclaimer"] == T.NARRATIVE_TEXTS["disclaimer"][0]
    en = sample_vm("fad", "en")
    assert "## Caveats" not in en["narrative"]["markdown"] and en["caveats"]["markdown"].startswith("- ")
    assert vm.split_caveats("no heading here\n", "ko") == ("no heading here", "")
    assert vm.split_caveats("a\n\n## Caveats\n\n- one\n- two\n\n", "en") == ("a", "- one\n- two")
    assert v["narrative"]["source"] == "rule" and v["narrative"]["badge"] == "규칙 기반 해석"


def test_source_block_naver_example():
    meta = vm.SourceMeta(kind="naver", value_kind="relative_index", freq="W", start="2021-11-29", end="2026-11-16",
                         as_of=AS_OF, terms=("선크림", "자외선차단제"), naver_endpoint="hub", device="mo", gender="f",
                         ages=("4", "5"))
    block = vm.build_source_block(meta, "ko")
    assert list(block) == ["kind", "label", "short", "line", "value_kind", "value_kind_label", "value_kind_note",
                           "unit_label", "freq", "freq_label", "start", "end", "range_label", "as_of", "as_of_label",
                           "filters", "filters_label", "filename", "attribution"]
    assert block["line"] == ("출처: 네이버 데이터랩 검색어트렌드(NAVER API HUB) · 주 단위 · 모바일·여성·25–34세 · "
                             "2021.11.29 ~ 2026.11.22 · 상대 지수(조회 기간 최댓값 = 100) · 데이터 기준 2026.11.25 09:12")
    assert block["value_kind_note"] == "상대 지수: 이 조회에서 가장 많이 검색된 주 = 100이에요. 실제 검색 횟수가 아니에요."
    assert block["filters"] == {"device": "mo", "gender": "f", "ages": ["4", "5"]}
    assert block["attribution"] == "출처: 네이버 데이터랩 검색어트렌드(NAVER API HUB)"
    assert (block["short"], block["range_label"], block["as_of_label"]) == ("네이버", "2021.11.29 ~ 2026.11.22",
                                                                            "2026.11.25 09:12")
    assert vm.source_line(block, "ko", with_filters=False) == block["line"].replace(" · 모바일·여성·25–34세", "")
    legacy = vm.build_source_block(dataclasses.replace(meta, naver_endpoint="legacy"), "en")
    assert legacy["label"] == "Naver DataLab Search Trends (Naver Open API)"
    assert legacy["value_kind_note"].startswith("Relative index: the week with the most searches")
    assert vm._age_label(["2", "5"], "ko") == "13–18세·30–34세"
    assert vm._age_label(["5", "2"], "en") == "13–18 · 30–34"
    assert vm._age_label(["10", "11"], "ko") == "55세 이상" and vm._age_label(["10", "11"], "en") == "55+"
    assert vm._age_label(["11"], "ko") == "60세 이상" and vm._age_label([], "ko") == ""
    none = vm.build_source_block(dataclasses.replace(meta, device=None, gender=None, ages=()), "ko")
    assert none["filters_label"] == "" and " · 2021.11.29" in none["line"]


def test_source_lines_by_kind():
    base = dict(freq="M", start="2021-01-01", end="2025-12-01", as_of=AS_OF)
    wiki = vm.SourceMeta(kind="wiki", value_kind="pageviews", terms=("BTS",), wiki_lang="ja", wiki_title="BTS", **base)
    assert vm.build_source_block(wiki, "ko")["line"] == (
        "출처: Wikimedia Pageviews · ja 위키백과 · 사람 조회만(봇 제외) · 월 단위 · 2021.01 ~ 2025.12 · 조회수 · "
        "데이터 기준 2026.11.25 09:12")
    up = vm.SourceMeta(kind="upload", value_kind="raw", terms=("매출",), filename="월매출.xlsx", unit_label="원",
                       **base)
    block = vm.build_source_block(up, "ko")
    assert block["line"] == ("출처: 사용자 업로드 (월매출.xlsx) · 월 단위 · 2021.01 ~ 2025.12 · 원래 단위 (원) · "
                             "분석 2026.11.25 09:12")
    assert block["value_kind_note"] == "" and block["filters"] is None and block["short"] == "내 파일"
    assert vm.build_source_block(up, "en")["line"] == (
        "Source: Your upload (월매출.xlsx) · monthly · Jan 2021 – Dec 2025 · original units (원) · "
        "analysed Nov 25, 2026 09:12 KST")
    gt = vm.SourceMeta(kind="google_trends_upload", value_kind="relative_index_gt", terms=("kpop",),
                       filename="multiTimeline.csv", **base)
    assert vm.build_source_block(gt, "ko")["line"] == (
        "출처: Google 트렌드(사용자 업로드, multiTimeline.csv) · Data source: Google Trends "
        "(https://www.google.com/trends) · 월 단위 · 2021.01 ~ 2025.12 · 상대 지수(0–100, 요청마다 다시 맞춰짐) · "
        "분석 2026.11.25 09:12")
    smp = vm.SourceMeta(kind="sample", value_kind="raw", terms=("fad",), synthetic=True, **base)
    block = vm.build_source_block(smp, "ko")
    assert block["line"] == "출처: trendai 내장 샘플 · 가상 데이터 · 월 단위 · 2021.01 ~ 2025.12"
    assert block["value_kind_note"] == "가상 데이터예요. 실제 데이터가 아니에요."
    assert "가상 데이터" in sample_vm("fad")["source"]["line"]


def test_notices_and_fixes():
    fix = vm.fix_object("wiki_title_help", {"url": "https://ko.wikipedia.org/w/index.php?search=x"})
    assert fix == {"action": "wiki_title_help", "label_ko": "위키백과에서 제목 찾기", "label_en": "Search Wikipedia titles",
                   "params": {"url": "https://ko.wikipedia.org/w/index.php?search=x"}}
    with pytest.raises(KeyError):
        vm.fix_object("no_such_action")
    n = vm.make_notice("data_short", "ko", params={"n": 5}, series_index=2)
    assert n["code"] == "data_short" and n["level"] == "warning" and n["series_index"] == 2
    assert n["text"] == render_web("data_short", {"n": 5}, "ko")
    assert [f["action"] for f in n["fixes"]] == ["extend_range", "try_sample"]
    up = vm.make_notice("data_short", "en", params={"n": 5}, series_index=0, source_kind="upload")
    assert [f["action"] for f in up["fixes"]] == ["reupload", "try_sample"]
    custom = vm.make_notice("naver_low_resolution", "ko", series_index=1,
                            fixes=[vm.fix_object("reanalyze_single", {"term": "패딩"})])
    assert custom["fixes"][0]["params"] == {"term": "패딩"}
    assert vm.make_notice("naver_cached", "ko", params={"fetched": "2026.11.25 09:12"})["series_index"] is None
    with pytest.raises(ValueError):
        vm.make_notice("gaps_interpolated", "ko")
    assert vm.make_notice("index_unavailable", "ko", level="warning")["level"] == "warning"
    joined = DataNotice(join(msg("date_order_ambiguous", label="DD/MM/YYYY (일/월 순)"),
                             msg("dates_unreadable_rows", n=2, examples="'x', 'y'")))
    nt = vm.notice_from_warning(joined, "en", series_index=0)
    assert nt["code"] == "date_order_ambiguous" and nt["level"] == "notice" and nt["fixes"] == []
    assert "day first" in nt["text"] and "2 row(s)" in nt["text"]
    from trendai.data import DataWarning
    assert vm.notice_from_warning(DataWarning(msg("horizon_long", horizon=24, n=3, ratio=5)), "ko")["level"] == \
        "warning"
    for level, actions in vm.NOTICE_DEFAULTS.values():
        assert level in vm.NOTICE_LEVELS
        for action in actions:
            vm.fix_object(action)


def test_export_blocks_and_links():
    av = vm.export_availability("naver", xlsx=True, view="compare", succeeded=2)
    assert av.series == {"csv": True, "xlsx": True, "html": False, "json": True}
    assert av.result["values_csv"] is False and av.result["html"] is False
    assert av.reasons == {"html": "export_naver_raw", "values_csv": "export_naver_raw"}
    series = vm.export_block(av, "ko", level="series")
    assert list(series) == ["csv", "xlsx", "html", "json", "unavailable"]
    assert all(series[k] is None for k in vm.EXPORT_FORMATS)
    assert series["unavailable"] == {"html": {"code": "export_naver_raw",
                                              "text": render_web("export_naver_raw", {}, "ko")}}
    result = vm.export_block(av, "en", level="result")
    assert list(result) == ["csv", "xlsx", "html", "json", "values_csv", "unavailable"]
    assert set(result["unavailable"]) == {"html", "values_csv"}
    linked = vm.link_exports(series, av.series, "/api/export/Zx8kP2mQ9rT4vW6yA1bC3d", 0)
    assert linked["csv"] == "/api/export/Zx8kP2mQ9rT4vW6yA1bC3d/0.csv" and linked["html"] is None
    assert linked["unavailable"] == series["unavailable"]
    assert series["csv"] is None                                  # the stored block is not mutated

    av2 = vm.export_availability("upload", xlsx=False, view="compare", succeeded=3)
    share_base = "/api/export/s/pQ3rS5tU7vW9xY1zA2bC4d"
    res = vm.link_exports(vm.export_block(av2, "ko", level="result"), av2.result, share_base)
    assert res == {"csv": "/api/export/s/pQ3rS5tU7vW9xY1zA2bC4d.csv", "xlsx": None,
                   "html": "/api/export/s/pQ3rS5tU7vW9xY1zA2bC4d.html",
                   "json": "/api/export/s/pQ3rS5tU7vW9xY1zA2bC4d.json",
                   "values_csv": "/api/export/s/pQ3rS5tU7vW9xY1zA2bC4d/values.csv",
                   "unavailable": {"xlsx": {"code": "export_xlsx_unavailable",
                                            "text": render_web("export_xlsx_unavailable", {}, "ko")}}}
    single = vm.export_availability("wiki", xlsx=True, view="detail", succeeded=1)
    assert single.result["values_csv"] is False and "values_csv" not in single.reasons
    naver_single = vm.export_availability("naver", xlsx=True, view="detail", succeeded=1)
    assert "values_csv" not in naver_single.reasons            # single series: 404 not_found, no reason shown
    v = sample_vm("growing")
    assert v["exports"] == {"csv": None, "xlsx": None, "html": None, "json": None, "unavailable": {}}
    r = report_of("naver_big")
    nv = build(make_input(r, meta_of(r, "naver"), xlsx=False))
    assert set(nv["exports"]["unavailable"]) == {"html", "xlsx"}
    assert not any("/api/" in s for s in strings(nv))              # stored VMs carry no URLs


def test_ai_fields_shares_and_records():
    nv_report = report_of("naver_big")
    naver = build(make_input(nv_report, meta_of(nv_report, "naver")))["narrative"]
    assert (naver["ai_allowed"], naver["ai_block_reason"]) == (False, "naver_terms")
    assert naver["ai_block_text"] == render_web("ai_naver_blocked", {}, "ko")
    off = build(make_input(report_of("growing"), meta_of(report_of("growing")), ai=False), "en")["narrative"]
    assert (off["ai_allowed"], off["ai_block_reason"]) == (False, "ai_unavailable")
    assert off["ai_block_text"] == render_web("ai_unavailable", {}, "en")
    v = sample_vm("growing")
    assert (v["narrative"]["ai_allowed"], v["narrative"]["ai_block_reason"], v["narrative"]["ai"]) == (True, None,
                                                                                                         None)
    shared = vm.make_shared_vm(v)
    assert shared["narrative"]["ai_allowed"] is False
    assert shared["narrative"]["ai_block_reason"] == "shared_readonly"
    assert shared["narrative"]["ai_block_text"] == render_web("ai_readonly", {}, "ko")
    assert v["narrative"]["ai_allowed"] is True                     # input untouched
    assert_valid(shared, "seriesVM")

    text = "**한눈에 보기** — 2024-03에 정점이에요.\n\n## 미래 전망\n- 2026-06까지\n\n## 주의할 점\n- 2025-12 이후 주의"
    record = vm.build_ai_record(text, v, model="claude-opus-5-5", created_at="2026-11-25T01:00:00Z")
    assert record["markdown"] == "**한눈에 보기** — 2024.03에 정점이에요.\n\n## 미래 전망\n- 2026.06까지"
    assert record["caveats_markdown"] == "- 2025.12 이후 주의"
    assert record["badge"] == "AI 해석 · Claude" and record["note"] == T.NARRATIVE_TEXTS["ai_note"][0]
    assert record["date_links"] == {"2024.03": "2024-03-01", "2026.06": "2026-06-01", "2025.12": "2025-12-01"}
    attached = vm.attach_ai(v, record)
    assert list(attached["narrative"]["ai"]) == ["markdown", "caveats_markdown", "badge", "model", "note",
                                                 "created_at"]
    assert attached["narrative"]["date_links"]["2026.06"] == "2026-06-01"
    assert v["narrative"]["ai"] is None
    assert_valid(attached, "seriesVM")
    assert_valid(vm.make_shared_vm(attached), "seriesVM")
    assert vm.attach_ai(v, None)["narrative"]["ai"] is None
    job = vm.ai_job_narrative(record)
    assert job == {"markdown": record["markdown"], "caveats_markdown": record["caveats_markdown"], "source": "llm",
                   "badge": "AI 해석 · Claude", "model": "claude-opus-5-5", "note": record["note"]}
    snapshot = {"job_id": "x" * 22, "kind": "narrative", "status": "done", "error": None, "elapsed_ms": 9120,
                "poll_after_ms": None, "narrative": job}
    assert_valid(snapshot, "narrativeJob")


def test_slot_helpers_and_flags():
    assert [vm.provisional_slot(i, 5) for i in range(5)] == [1, 2, 3, 4, 5]
    assert vm.provisional_slot(0, 6) is None
    assert vm.assign_slots([2, 0, 1], 3) == {2: 3, 0: 1, 1: 2}
    assert vm.assign_slots([6, 3, 0, 5, 1, 2, 4], 7) == {0: 1, 1: 2, 3: 3, 5: 4, 6: 5, 2: None, 4: None}
    v = {"slot": None, "icon": None}
    assert vm.set_slot(v, 3) is v and v == {"slot": 3, "icon": "hexagon"}
    assert vm.set_slot(v, None) == {"slot": None, "icon": None}
    assert vm.is_low_resolution("naver", 4.9, 2) is True
    assert vm.is_low_resolution("naver", 4.9, 1) is False
    assert vm.is_low_resolution("wiki", 1.0, 3) is False
    assert vm.is_low_resolution("naver", float("nan"), 3) is False
    assert vm.is_low_resolution("naver", None, 3) is False
    err = {"code": "wiki_not_found", "status": 404, "message_ko": "…", "message_en": "…", "params": {"title": "x"},
           "fixes": [vm.fix_object("use_naver")], "retry_after": None, "request_id": "a1b2c3d4"}
    entry = vm.build_failed_entry(2, "없는문서", err)
    assert entry == {"index": 2, "name": "없는문서", "error": err} and entry["error"] is not err
    assert_valid(entry, "failedEntry")
    synthetic = sample_vm("growing")["flags"]
    assert synthetic["synthetic"] is True and synthetic["tentative"] is False


def test_compare_with_failed_series_and_envelope_schema():
    r1, r2 = report_of("growing", "a"), report_of("fad", "b")
    inputs = [make_input(r1, meta_of(r1, "wiki"), index=0, request_size=3),
              make_input(r2, meta_of(r2, "wiki"), index=2, request_size=3)]
    reports = {i.name: i.report for i in inputs}
    ranking = rank_trends(reports)
    n2i = {"a": 0, "b": 2}
    slots = vm.assign_slots([n2i[n] for n in ranking["name"]], 3)
    vms = {i.index: vm.set_slot(build(i), slots[i.index]) for i in inputs}
    err = {"code": "wiki_not_found", "status": 404, "message_ko": "없음", "message_en": "missing",
           "params": {"title": "없는문서"}, "fixes": [], "retry_after": None, "request_id": None}
    failed = [vm.build_failed_entry(1, "없는문서", err)]
    cmp = vm.build_compare_vm(inputs, vms, ranking, n2i, slots, failed, "ko")
    assert_valid(cmp, "compareVM")
    assert cmp["count"] == 2 and cmp["slots"] == {"0": 1, "2": 3} and cmp["failed"] == failed
    av = vm.export_availability("wiki", xlsx=True, view="compare", succeeded=2)
    envelope = {
        "result_id": "Zx8kP2mQ9rT4vW6yA1bC3d", "created_at": AS_OF, "expires_at": "2026-11-26T00:12:00Z",
        "readonly": False, "view": "compare",
        "query": {"source": "wiki", "series": [{"terms": ["a"]}, {"terms": ["없는문서"]}, {"terms": ["b"]}],
                  "range": "5y", "unit": "month", "h": 12, "wl": "ko",
                  "period": {"start": "2021-11-01", "end": "2026-10-31"}},
        "series": [vm.link_exports(v["exports"], av.series, "/api/export/Zx8kP2mQ9rT4vW6yA1bC3d", k) and
                   {**v, "exports": vm.link_exports(v["exports"], av.series, "/api/export/Zx8kP2mQ9rT4vW6yA1bC3d", k)}
                   for k, v in sorted(vms.items())],
        "failed": failed, "compare": cmp, "notices": [],
        "exports": vm.link_exports(vm.export_block(av, "ko", level="result"), av.result,
                                   "/api/export/Zx8kP2mQ9rT4vW6yA1bC3d"),
    }
    assert_valid(envelope, "resultEnvelope")
    job = {"job_id": "Zx8kP2mQ9rT4vW6yA1bC3d", "kind": "analyze", "status": "running", "stage": "series",
           "series_total": 3, "series_done": 1, "current": {"index": 1, "name": "없는문서", "step": "forecast"},
           "series": [{"index": 0, "status": "done", "vm": vms[0]}], "failed": [], "notices": [], "compare": None,
           "view": "compare", "error": None, "result_url": None, "elapsed_ms": 1840, "poll_after_ms": 500}
    assert_valid(job, "analyzeJob")
