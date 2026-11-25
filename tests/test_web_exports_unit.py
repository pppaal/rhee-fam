"""Export files at unit level (docs/WEB_SPEC.md C.10, E.15): CSV/XLSX/JSON/HTML built from real VMs and reports.

The HTTP-level export routes (status codes, headers on the wire, share snapshots) are tested by WP1-server in
``tests/test_web_share_export.py``; this file checks the bytes ``trendai.web.exports`` produces.
"""

from __future__ import annotations

import csv
import functools
import io
import json
from typing import Optional

import numpy as np
import pandas as pd
import pytest

from trendai import narrative as N
from trendai import pipeline, samples
from trendai.compare import rank_trends
from trendai.web import exports as ex
from trendai.web import viewmodel as vm

AS_OF = "2026-11-25T00:12:00Z"
FILE_ID = "Zx8kP2mQ9rT4vW6yA1bC3d"


# --------------------------------------------------------------------------------------------------------- builders

def _weekly(peak: float, seed: int, end: str = "2026-11-16", n: int = 156) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.date_range(end=end, periods=n, freq="W-MON")
    base = 0.5 + 0.5 * np.sin(2 * np.pi * np.arange(n) / 52.0) + np.linspace(0, 0.4, n) + rng.normal(0, 0.05, n)
    base = np.clip(base, 0.01, None)
    return pd.Series(base / base.max() * peak, index=idx)


SERIES = {
    "growing": lambda: samples.sample("growing"),
    "seasonal": lambda: samples.sample("seasonal"),
    "fad": lambda: samples.sample("fad"),
    "late": lambda: samples.sample("seasonal").iloc[30:] * 2.0,
    "naver_a": lambda: _weekly(100.0, 1),
    "naver_b": lambda: _weekly(40.0, 2),
}


@functools.lru_cache(maxsize=None)
def report_of(key: str, name: Optional[str] = None):
    return pipeline.run(SERIES[key](), name=name or key)


def meta_of(report, kind: str, **kw) -> vm.SourceMeta:
    a = report.analysis
    vk = {"naver": "relative_index", "wiki": "pageviews"}.get(kind, "raw")
    if kind == "naver":
        kw.setdefault("naver_endpoint", "hub")
    if kind == "sample":
        kw.setdefault("synthetic", True)
    kw.setdefault("terms", (report.name,))
    return vm.SourceMeta(kind=kind, value_kind=vk, freq=a.freq, start=vm._iso(a.series.index[0]),
                         end=vm._iso(a.series.index[-1]), as_of=AS_OF, **kw)


def input_of(report, meta, index: int = 0, request_size: int = 1) -> vm.SeriesInput:
    en = N.generate_narrative(report.analysis, report.forecast, report.outlook, lang="en")[0]
    return vm.SeriesInput(index=index, name=report.name, report=report, source=meta,
                          narratives={"ko": report.narrative, "en": en}, notices={"ko": [], "en": []},
                          request_size=request_size, ai_enabled=True, xlsx_available=True,
                          slot=vm.provisional_slot(index, request_size))


def compare_of(inputs, lang: str = "ko"):
    ranking = rank_trends({i.name: i.report for i in inputs})
    n2i = {i.name: i.index for i in inputs}
    slots = vm.assign_slots([n2i[n] for n in ranking["name"]], len(inputs))
    vms = [vm.set_slot(vm.build_series_vm(i, lang), slots[i.index]) for i in inputs]
    cmp = vm.build_compare_vm(inputs, {v["index"]: v for v in vms}, ranking, n2i, slots, [], lang)
    return vms, cmp, ranking, n2i


@functools.lru_cache(maxsize=None)
def naver_case(lang: str = "ko"):
    a, b = report_of("naver_a", "선크림"), report_of("naver_b", "패딩")
    filters = dict(device="mo", gender="f", ages=("4", "5"))
    inputs = [input_of(a, meta_of(a, "naver", terms=("선크림", "자외선차단제"), **filters), 0, 2),
              input_of(b, meta_of(b, "naver", terms=("패딩",), **filters), 1, 2)]
    return inputs, compare_of(inputs, lang)


@functools.lru_cache(maxsize=None)
def upload_case(lang: str = "ko"):
    a, b = report_of("growing", "=SUM(A1)"), report_of("late", "방문자,=HYPERLINK(1)")
    inputs = [input_of(a, meta_of(a, "upload", filename="sales.csv"), 0, 2),
              input_of(b, meta_of(b, "upload", filename="sales.csv"), 1, 2)]
    return inputs, compare_of(inputs, lang)


def csv_lines(f: ex.ExportFile) -> list[str]:
    assert f.body.startswith(b"\xef\xbb\xbf")
    text = f.body.decode("utf-8-sig")
    assert "\r\n" in text and "\n" not in text.replace("\r\n", "")
    return text.split("\r\n")[:-1]


# ------------------------------------------------------------------------------------------------------------- CSV

def test_csv_format_naver_weekly():
    inputs, (vms, cmp, _, _) = naver_case()
    v = vms[0]
    f = ex.series_csv(v, file_id=FILE_ID)
    lines = csv_lines(f)
    assert lines[0] == ("# 출처: 네이버 데이터랩 검색어트렌드(NAVER API HUB) · 주 단위 · 2023.11.27 ~ 2026.11.22 · "
                        "상대 지수(조회 기간 최댓값 = 100) · 데이터 기준 2026.11.25 09:12")
    assert lines[1] == ("# 조건: 검색어 그룹 선크림(선크림, 자외선차단제) · 모바일 · 여성 · 25–34세 · 예측 26주 · "
                        "네이버 원본 값(ratio)은 이용약관상 포함하지 않았어요")
    assert lines[2] == "날짜,추세,예측값,80% 하한,80% 상한,95% 하한,95% 상한"     # no 실제값 for Naver
    chart = v["chart"]
    rows = list(csv.reader(lines[3:]))
    n_hist, n_fc = len(chart["dates"]), len(chart["forecast"]["dates"])
    assert len(rows) == n_hist + n_fc
    assert rows[0] == [chart["dates"][0], format(chart["trend"][0], ".15g"), "", "", "", "", ""]
    first_fc = rows[n_hist]
    fc = chart["forecast"]
    assert first_fc == [fc["dates"][0], ""] + [format(fc[k][0], ".15g") for k in ("mean", "lo80", "hi80", "lo95",
                                                                                  "hi95")]
    assert f.filename == "trendai-Zx8kP2mQ-0.csv" and f.filename_utf8 == "trendai-선크림.csv"
    en = ex.series_csv(naver_case("en")[1][0][0], file_id=FILE_ID)
    en_lines = csv_lines(en)
    assert en_lines[2] == "date,trend,forecast,lo80,hi80,lo95,hi95"
    assert en_lines[1].endswith("Raw Naver values (ratio) are excluded under its terms of use")
    assert "Nov 22, 2026" in en_lines[0]


def test_csv_history_has_actual_and_reads_back():
    r = report_of("growing")
    v = vm.build_series_vm(input_of(r, meta_of(r, "sample")), "ko")
    f = ex.series_csv(v, file_id=FILE_ID)
    lines = csv_lines(f)
    assert lines[0] == "# 출처: trendai 내장 샘플 · 가상 데이터 · 월 단위 · 2016.01 ~ 2025.12"
    assert lines[1] == "# 조건: 샘플 growing · 예측 12개월"
    assert lines[2] == "날짜,실제값,추세,예측값,80% 하한,80% 상한,95% 하한,95% 상한"
    df = pd.read_csv(io.BytesIO(f.body), comment="#", encoding="utf-8-sig")
    assert len(df) == len(r.analysis.series) + len(r.forecast.mean)
    assert df["실제값"].iloc[:120].tolist() == pytest.approx(np.asarray(r.analysis.series).tolist())
    assert df["예측값"].iloc[120:].tolist() == pytest.approx(np.asarray(r.forecast.mean).tolist())
    assert df["실제값"].iloc[120:].isna().all() and df["예측값"].iloc[:120].isna().all()


def test_csv_formula_guard():
    assert [ex.guard_text(s) for s in ("=1+1", "+1", "-1", "@SUM(A1)", "\tx", "\rx", "−4.1%", "a=b", "")] == \
        ["'=1+1", "'+1", "'-1", "'@SUM(A1)", "'\tx", "'\rx", "−4.1%", "a=b", ""]
    inputs, (vms, cmp, ranking, n2i) = upload_case()
    rows = ex.ranking_rows(ranking, n2i)
    lines = csv_lines(ex.result_csv(vms, cmp, rows, file_id=FILE_ID))
    assert lines[2] == "순위,이름,국면,전망,예측 변화(%),상승 확률(%),최근 변화(%),모멘텀(%p/기간),상승 점수,예측 신뢰도"
    names = [r[1] for r in csv.reader(lines[3:])]
    # a quoted cell that merely contains "=" later on starts with "방": nothing to guard
    assert set(names) == {"'=SUM(A1)", "방문자,=HYPERLINK(1)"}
    assert lines[1].startswith("# 조건: 열 =SUM(A1) · 열 방문자,'=HYPERLINK(1) · ")   # comment pieces guarded too
    values = csv_lines(ex.values_csv(vms, cmp, file_id=FILE_ID))
    header = next(csv.reader([values[2]]))
    assert header == ["날짜", "'=SUM(A1)", "방문자,=HYPERLINK(1)"]
    for line in values[:2]:
        assert line.startswith("# ")


def test_values_csv_union_of_dates():
    inputs, (vms, cmp, _, _) = upload_case()
    f = ex.values_csv(vms, cmp, file_id=FILE_ID)
    assert f.filename == "trendai-Zx8kP2mQ-values.csv"
    assert f.filename_utf8.endswith("-values.csv")
    lines = csv_lines(f)
    assert lines[0] == "# " + cmp["source_line"]
    rows = list(csv.reader(lines[3:]))
    a, b = (inp.report.analysis.series for inp in inputs)
    dates = sorted(set(vm._iso(d) for d in a.index) | set(vm._iso(d) for d in b.index))
    assert [r[0] for r in rows] == dates
    first = rows[0]
    assert first[1] == format(float(a.iloc[0]), ".15g") and first[2] == ""      # b starts later
    last = rows[-1]
    assert last[1] and last[2]
    with pytest.raises(ex.ExportUnavailable) as e:
        ex.values_csv(vms[:1], None, file_id=FILE_ID)
    assert (e.value.code, e.value.status) == ("not_found", 404)
    _, (nvms, ncmp, _, _) = naver_case()
    with pytest.raises(ex.ExportUnavailable) as e:
        ex.values_csv(nvms, ncmp, file_id=FILE_ID)
    assert (e.value.code, e.value.status) == ("export_naver_raw", 403)


def test_result_csv_ranking_and_single():
    inputs, (vms, cmp, ranking, n2i) = upload_case("en")
    rows = ex.ranking_rows(ranking, n2i)
    f = ex.result_csv(vms, cmp, rows, file_id=FILE_ID)
    assert f.filename == "trendai-Zx8kP2mQ-all.csv"
    assert f.filename_utf8 == "trendai-=SUM(A1) vs 방문자,=HYPERLINK(1).csv"
    lines = csv_lines(f)
    assert lines[0] == "# " + cmp["source_line"]          # "Nov 25, 2026": a comma, nothing to guard after it
    assert "," in cmp["source_line"]
    assert lines[2].startswith("rank,name,phase,outlook,expected change (%)")
    table = list(csv.reader(lines[3:]))
    assert [int(r[0]) for r in table] == [1, 2]
    by_name = {r[1]: r for r in table}
    for row in rows:
        out = by_name[ex.guard_text(row["name"])]
        v = next(x for x in vms if x["index"] == row["index"])
        assert out[2] == v["phase"]["name"] and out[3] == v["hero"]["outlook"]["word"]
        assert out[5] == ("" if row["prob_increase"] is None else format(row["prob_increase"] * 100, ".15g"))
        assert out[8] == format(row["score"], ".15g")
        assert out[9] in ("high", "medium", "low")
    single = vms[:1]
    a = ex.result_csv(single, None, None, file_id=FILE_ID)
    b = ex.series_csv(single[0], file_id=FILE_ID)
    assert a.body == b.body and a.filename.endswith("-all.csv")


def test_csv_numbers_are_plain_decimals():
    assert [ex._csv_number(v) for v in (1e-05, 1.5e20, 0.1 + 0.2, -0.0, None, float("nan"), 12.5, 3)] == \
        ["0.00001", "150000000000000000000", "0.3", "0", "", "", "12.5", "3"]


# ------------------------------------------------------------------------------------------------------------ JSON

def test_json_series_export_and_naver_strip():
    a = report_of("naver_a", "선크림")
    meta = meta_of(a, "naver", terms=("선크림",))
    block = vm.build_source_block(meta, "en")
    en_text = N.generate_narrative(a.analysis, a.forecast, a.outlook, lang="en")[0]
    out = ex.build_series_json(a, block, "en", narrative=en_text)
    assert out["source"] == block and out["narrative"] == en_text
    assert "history" not in out and out["raw_excluded"] is True
    for key in ("last_value", "peak_value", "mean"):
        assert key not in out["metrics"]
    assert all("value" not in an for an in out["anomalies"])
    assert out["trend"] and out["forecast"]["table"] and "scenarios" in out["outlook"]
    json.dumps(out, ensure_ascii=False, allow_nan=False)

    r = report_of("growing")
    sblock = vm.build_source_block(meta_of(r, "sample"), "ko")
    plain = ex.build_series_json(r, sblock, "ko")
    assert plain["history"] and "raw_excluded" not in plain and plain["synthetic"] is True
    assert plain["narrative"] == r.narrative and list(plain)[0] == "source"
    stripped = ex.strip_naver_raw(plain)
    assert "history" in plain and "history" not in stripped              # copy, input untouched


def test_json_result_store_and_export():
    r1, r2 = report_of("growing"), report_of("fad")
    s1 = ex.build_series_json(r1, vm.build_source_block(meta_of(r1, "sample"), "ko"), "ko")
    s2 = ex.build_series_json(r2, vm.build_source_block(meta_of(r2, "sample"), "ko"), "ko")
    assert ex.build_result_json([s1]) == s1
    whole = ex.build_result_json([s1, s2], source_line="출처: x")
    assert list(whole) == ["source_line", "source", "reports"]
    assert whole["source"] == [s1["source"], s2["source"]] and whole["reports"] == [s1, s2]
    store = ex.build_export_store({2: s2, 0: s1}, whole)
    assert list(store["series"]) == ["0", "2"] and store["result"] == whole
    f = ex.json_export(store, file_id=FILE_ID, k=2, display_name="fad")
    assert json.loads(f.body) == s2 and f.filename == "trendai-Zx8kP2mQ-2.json"
    assert f.body.decode().startswith('{\n  "source"')                 # indent=2
    g = ex.json_export(store, file_id=FILE_ID, k=None, display_name="growing vs fad")
    assert json.loads(g.body) == whole and g.filename_utf8 == "trendai-growing vs fad.json"
    assert ("Content-Type", "application/json; charset=utf-8") in g.headers()
    with pytest.raises(ex.ExportUnavailable) as e:
        ex.json_export(store, file_id=FILE_ID, k=1, display_name="x")
    assert (e.value.code, e.value.status) == ("not_found", 404)


def test_ranking_rows_are_json_safe():
    inputs, (vms, cmp, ranking, n2i) = upload_case()
    rows = ex.ranking_rows(ranking, n2i)
    assert [r["rank"] for r in rows] == [1, 2]
    assert {r["index"] for r in rows} == {0, 1}
    for r in rows:
        assert r["index"] == n2i[r["name"]]
        assert set(r) == {"rank", "index", "name", "phase", "expected_change_pct", "prob_increase",
                          "recent_change_pct", "momentum", "score", "confidence", "freq"}
    json.dumps(rows, allow_nan=False)


# ------------------------------------------------------------------------------------------------------------ HTML

def test_html_source_paragraph_and_headers():
    html = "<!doctype html>\n<html><head></head><body>\n<main class=\"wrap\">x</main></body></html>"
    out = ex.insert_source_paragraph(html, "출처: 사용자 업로드 (<b>a.csv</b>)")
    assert '<body><p class="source">출처: 사용자 업로드 (&lt;b&gt;a.csv&lt;/b&gt;)</p>\n<main class="wrap">' in out
    assert ex.insert_source_paragraph("<p>no body</p>", "x") == "<p>no body</p>"
    r = report_of("growing")
    line = vm.build_source_block(meta_of(r, "sample"), "ko")["line"]
    page = ex.render_series_html(r, line)
    assert '<body><p class="source">출처: trendai 내장 샘플 · 가상 데이터' in page
    assert page.count('<p class="source">') == 1
    reports = [report_of("growing"), report_of("fad")]
    cpage = ex.render_result_html(reports, rank_trends({x.name: x for x in reports}), "출처: 비교")
    assert '<p class="source">출처: 비교</p>' in cpage
    f = ex.html_export(page, file_id=FILE_ID, k=None, display_name="growing")
    assert f.filename == "trendai-Zx8kP2mQ-all.html" and f.body == page.encode("utf-8")
    headers = dict(f.headers())
    assert headers["Content-Security-Policy"] == ex.HTML_EXPORT_CSP
    assert headers["Cache-Control"] == "private, no-store" and headers["X-Content-Type-Options"] == "nosniff"
    assert headers["Content-Type"] == "text/html; charset=utf-8"
    assert "Content-Security-Policy" not in dict(ex.json_export({"result": {}}, file_id=FILE_ID, k=None,
                                                               display_name="x").headers())


# -------------------------------------------------------------------------------------------------------- filenames

def test_filenames_and_content_disposition():
    assert ex.export_filenames(FILE_ID, 0, "선크림", "csv") == ("trendai-Zx8kP2mQ-0.csv", "trendai-선크림.csv")
    assert ex.export_filenames(FILE_ID, None, "a vs b", "xlsx") == ("trendai-Zx8kP2mQ-all.xlsx", "trendai-a vs b.xlsx")
    assert ex.export_filenames(FILE_ID, "values", "a vs b", "csv")[0] == "trendai-Zx8kP2mQ-values.csv"
    assert ex.export_filenames(FILE_ID, 1, 'a/b\\c:d*e?f"g<h>i|j\x01', "csv")[1] == "trendai-a_b_c_d_e_f_g_h_i_j_.csv"
    long_name = "가" * 120
    assert ex.export_filenames(FILE_ID, 1, long_name, "json")[1] == "trendai-" + "가" * 80 + ".json"
    assert ex.export_filenames(FILE_ID, 1, "é", "csv")[1] == "trendai-é.csv"          # NFC
    cd = ex.content_disposition("trendai-Zx8kP2mQ-0.csv", "trendai-선크림.csv")
    assert cd == ("attachment; filename=\"trendai-Zx8kP2mQ-0.csv\"; "
                  "filename*=UTF-8''trendai-%EC%84%A0%ED%81%AC%EB%A6%BC.csv")
    assert "%20" in ex.content_disposition("a.csv", "trendai-a vs b.csv")


def test_sheet_names():
    assert ex.sheet_names(["a[b]:c*d?e/f\\g", "x" * 40, "x" * 40, "요약", ""], "ko") == \
        ["a_b__c_d_e_f_g", "x" * 31, "x" * 27 + " (2)", "요약 (2)", "데이터"]
    assert ex.sheet_names(["Summary", "summary"], "en") == ["Summary (2)", "summary (3)"]
    assert ex.sheet_names(["'quoted'"], "en") == ["_quoted_"]


# ------------------------------------------------------------------------------------------------------------ XLSX

def test_xlsx_structure():
    openpyxl = pytest.importorskip("openpyxl")
    inputs, (vms, cmp, _, _) = upload_case()
    f = ex.result_xlsx(vms, cmp, file_id=FILE_ID)
    assert f.fmt == "xlsx" and f.filename == "trendai-Zx8kP2mQ-all.xlsx"
    wb = openpyxl.load_workbook(io.BytesIO(f.body))
    assert wb.sheetnames == ["=SUM(A1)", "방문자,=HYPERLINK(1)", "요약"]
    ws = wb["=SUM(A1)"]
    assert [ws.cell(r, 1).value for r in range(1, 8)] == ["출처", "검색어 그룹/열", "기간", "단위", "조건(기기·성별·연령)",
                                                         "데이터 기준", "값의 성격"]
    assert ws.cell(1, 2).value.startswith("출처: 사용자 업로드 (sales.csv) · 월 단위")
    assert ws.cell(2, 2).value == "열 =SUM(A1)" and ws.cell(2, 2).data_type == "s"
    assert ws.cell(8, 1).value is None
    assert [ws.cell(9, c).value for c in range(1, 9)] == ["날짜", "실제값", "추세", "예측값", "80% 하한", "80% 상한",
                                                         "95% 하한", "95% 상한"]
    assert ws.cell(10, 1).value == vms[0]["chart"]["dates"][0]
    assert isinstance(ws.cell(10, 2).value, float)
    assert ws.cell(10, 2).value == pytest.approx(vms[0]["chart"]["actual"][0])
    summary = wb["요약"]
    assert [c.value for c in summary[1]] == ["이름", "국면", "전망", "예측 변화", "상승 확률", "예측 신뢰도", "정확도",
                                             "상승 점수"]
    names = [summary.cell(r, 1) for r in (2, 3)]
    assert [c.value for c in names] == ["=SUM(A1)", "방문자,=HYPERLINK(1)"]
    for cell in names:
        assert cell.data_type == "s" and not cell.value.startswith("'")
    scores = {row["index"]: row["score"]["text"] for row in cmp["ranking"]}
    assert summary.cell(2, 8).value == scores[0]
    assert summary.cell(2, 4).value == vms[0]["tiles"][3]["value"]


def test_xlsx_naver_rules_and_number_format():
    openpyxl = pytest.importorskip("openpyxl")
    _, (vms, cmp, _, _) = naver_case("en")
    f = ex.series_xlsx(vms[0], file_id=FILE_ID)
    assert f.filename == "trendai-Zx8kP2mQ-0.xlsx"
    wb = openpyxl.load_workbook(io.BytesIO(f.body))
    assert wb.sheetnames == ["선크림", "Summary"]
    ws = wb["선크림"]
    assert [ws.cell(r, 1).value for r in range(1, 8)] == ["Source", "Groups/columns", "Period", "Unit",
                                                         "Filters (device · gender · age)", "Data as of",
                                                         "Value kind"]
    assert ws.cell(5, 2).value == "mobile · female · 25–34"
    assert ws.cell(7, 2).value.endswith("Raw Naver values (ratio) are excluded under its terms of use")
    assert [ws.cell(9, c).value for c in range(1, 8)] == ["date", "trend", "forecast", "lo80", "hi80", "lo95",
                                                         "hi95"]
    assert ws.cell(10, 2).number_format == "0.0"
    assert all("actual" != ws.cell(9, c).value for c in range(1, 9))
    assert wb["Summary"].cell(2, 8).value == "—"                          # a single series has no score


def test_xlsx_unavailable(monkeypatch):
    monkeypatch.setattr(ex, "xlsx_available", lambda: False)
    _, (vms, cmp, _, _) = upload_case()
    for call in (lambda: ex.series_xlsx(vms[0], file_id=FILE_ID), lambda: ex.result_xlsx(vms, cmp, file_id=FILE_ID)):
        with pytest.raises(ex.ExportUnavailable) as e:
            call()
        assert (e.value.code, e.value.status) == ("export_xlsx_unavailable", 501)


def test_xlsx_available_is_cached_bool():
    assert isinstance(ex.xlsx_available(), bool)
    assert ex.xlsx_available() is ex.xlsx_available()
