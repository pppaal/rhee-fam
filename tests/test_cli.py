"""CLI smoke tests (no network)."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from trendai.cli import main
from trendai.samples import sample


def test_help(capsys):
    assert main([]) == 0
    assert "trendai" in capsys.readouterr().out


def test_list_samples(capsys):
    assert main(["list-samples"]) == 0
    assert "emerging" in capsys.readouterr().out


def test_sample_with_outputs(tmp_path, capsys):
    html, js = tmp_path / "r.html", tmp_path / "r.json"
    assert main(["sample", "fad", "--html", str(html), "--json", str(js), "-H", "6"]) == 0
    out = capsys.readouterr().out
    assert "합성" in out
    assert html.read_text(encoding="utf-8").lstrip().lower().startswith("<!doctype html")
    assert json.loads(js.read_text(encoding="utf-8"))["forecast"]["horizon"] == 6


def test_sample_all_compare(tmp_path, capsys):
    html = tmp_path / "cmp.html"
    assert main(["sample", "all", "--html", str(html), "-q"]) == 0
    out = capsys.readouterr().out
    assert "순위" in out
    assert html.exists()
    assert len(list(tmp_path.glob("cmp_*.html"))) >= 5
    # every card title and ranking name links to its own per-series page, and the links resolve
    import re
    from urllib.parse import unquote

    page = html.read_text(encoding="utf-8")
    hrefs = re.findall(r'<a href="([^"]+)">([^<]+)</a>', page)
    names = {nm for _, nm in hrefs}
    assert len(names) == 9 and len(hrefs) == 18
    for href, nm in hrefs:
        target = tmp_path / unquote(href)
        assert target.exists(), href
        assert f"<h1>{nm}</h1>" in target.read_text(encoding="utf-8")


def test_unknown_sample(capsys):
    assert main(["sample", "nope"]) == 2


def test_analyze_csv_multi_column(tmp_path, capsys):
    df = pd.DataFrame({"날짜": sample("growing").index.strftime("%Y-%m-%d"),
                       "검색량": sample("growing").to_numpy(),
                       "판매량": sample("declining").to_numpy()})
    f = tmp_path / "data.csv"
    df.to_csv(f, index=False)
    assert main(["analyze", str(f), "--json", str(tmp_path / "o.json")]) == 0
    data = json.loads((tmp_path / "o.json").read_text(encoding="utf-8"))
    assert {d["name"] for d in data} == {"검색량", "판매량"}
    assert main(["analyze", str(f), "--value-col", "검색량", "-q"]) == 0


def test_analyze_missing_file(capsys):
    assert main(["analyze", "/nonexistent/file.csv"]) == 2


def test_compare_samples(capsys):
    assert main(["compare", "--samples", "emerging,declining", "-q"]) == 0
    assert "순위" in capsys.readouterr().out


# --------------------------------------------------------------------------- UX / robustness


def _display_width(s: str) -> int:
    import unicodedata

    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in s)


@pytest.mark.parametrize("argv", [
    ["sample", "growing", "-H", "0"],
    ["sample", "growing", "-H", "-3"],
    ["sample", "growing", "--n", "5"],
    ["analyze", "x.csv", "--horizon", "abc"],
])
def test_bad_numeric_arguments_rejected_by_argparse(argv, capsys):
    with pytest.raises(SystemExit) as exc:
        main(argv)
    assert exc.value.code == 2


def test_compare_unknown_sample_name(capsys):
    assert main(["compare", "--samples", "emerging,foo"]) == 2
    err = capsys.readouterr().err
    assert "foo" in err and "Traceback" not in err


def test_too_short_data_exits_2_with_clean_warning(tmp_path, capsys):
    f = tmp_path / "one.csv"
    f.write_text("date,val\n2024-01-01,5\n")
    assert main(["analyze", str(f)]) == 2
    err = capsys.readouterr().err
    assert "경고:" in err and "2개 이상" in err
    assert ".py:" not in err and "UserWarning" not in err


@pytest.mark.parametrize("content,needle", [
    ("", "비어 있습니다"),
    ("a,b\nx,1\ny,2\n", "--date-col"),
])
def test_input_errors_are_one_korean_line(tmp_path, capsys, content, needle):
    f = tmp_path / "bad.csv"
    f.write_text(content)
    assert main(["analyze", str(f)]) == 2
    err = capsys.readouterr().err.strip()
    assert needle in err and "Traceback" not in err and len(err.splitlines()) == 1


def test_directory_and_json_series(tmp_path, capsys):
    (tmp_path / "d.csv").mkdir()
    assert main(["analyze", str(tmp_path / "d.csv")]) == 2
    sample("growing").to_json(tmp_path / "s.json")
    assert main(["analyze", str(tmp_path / "s.json"), "-q", "-H", "3"]) == 0


def test_row_number_column_gives_single_series(tmp_path, capsys):
    g = sample("growing")
    pd.DataFrame({"date": g.index.strftime("%Y-%m-%d"), "sales": g.round(2).values}).to_csv(tmp_path / "wi.csv")
    js = tmp_path / "o.json"
    assert main(["analyze", str(tmp_path / "wi.csv"), "-H", "3", "--json", str(js)]) == 0
    captured = capsys.readouterr()
    assert "순위" not in captured.out                 # no fake comparison
    assert "## " in captured.out                      # narrative printed for the single series
    assert "Unnamed: 0" in captured.err               # ... and the user is told what was ignored
    assert isinstance(json.loads(js.read_text(encoding="utf-8")), dict)


def test_html_names_do_not_collide(tmp_path, capsys):
    g, d = sample("growing"), sample("declining")
    pd.DataFrame({"date": g.index.strftime("%Y-%m"), "sales ($)": g.values, "sales (%)": d.values}
                 ).to_csv(tmp_path / "slug.csv", index=False)
    out = tmp_path / "o" / "cmp.html"
    assert main(["analyze", str(tmp_path / "slug.csv"), "-q", "-H", "3", "--html", str(out)]) == 0
    files = sorted(p.name for p in (tmp_path / "o").glob("cmp_*.html"))
    assert len(files) == 2
    titles = {(tmp_path / "o" / f).read_text(encoding="utf-8").count("sales (%)") > 0 for f in files}
    assert titles == {True, False}


def test_compare_page_gets_links_when_supported(tmp_path, capsys, monkeypatch):
    import trendai.report as report

    seen = {}

    def fake_compare(reports, ranking, path=None, links=None):
        seen["links"] = links
        Path(path).write_text("<!doctype html>", encoding="utf-8")
        return "<!doctype html>"

    from pathlib import Path

    monkeypatch.setattr(report, "render_compare_html", fake_compare)
    out = tmp_path / "all.html"
    assert main(["compare", "--samples", "growing,fad", "-q", "-H", "3", "--html", str(out)]) == 0
    assert set(seen["links"]) == {"growing", "fad"}
    for href in seen["links"].values():
        assert (tmp_path / href).exists()


def test_ranking_columns_align_with_korean_text(capsys):
    assert main(["compare", "--samples", "emerging,fad,seasonal", "-q", "-H", "3"]) == 0
    out = capsys.readouterr().out.splitlines()
    i = next(k for k, line in enumerate(out) if "순위" in line)
    rows = [line for line in out[i + 1:] if line.strip()[:2].rstrip(".").isdigit()]
    assert len(rows) == 3
    assert len({_display_width(r) for r in rows}) == 1
    assert len({_display_width(r.split("%")[0]) for r in rows}) == 1


def test_multi_series_without_quiet_explains_hidden_narratives(tmp_path, capsys):
    assert main(["compare", "--samples", "emerging,fad", "-H", "3"]) == 0
    assert "해석문은 생략" in capsys.readouterr().out


def test_lang_en_cli_chrome(capsys, tmp_path):
    js = tmp_path / "r.json"
    assert main(["sample", "fad", "-H", "3", "--lang", "en", "--json", str(js)]) == 0
    out = capsys.readouterr().out
    assert "Synthetic sample data" in out and "JSON saved" in out
    assert "합성(가상)" not in out


def test_business_day_warning_is_a_clean_line(tmp_path, capsys):
    import numpy as np

    idx = pd.bdate_range("2023-01-02", periods=120)
    vals = 100 + np.cumsum(np.random.default_rng(0).normal(0, 1, 120))
    pd.DataFrame({"date": idx.strftime("%Y-%m-%d"), "close": vals}).to_csv(tmp_path / "b.csv", index=False)
    assert main(["analyze", str(tmp_path / "b.csv"), "-q", "-H", "5"]) == 0
    err = capsys.readouterr().err
    assert "경고: " in err and "평일" in err and "DataWarning" not in err


def test_broken_pipe_keeps_writing_files(tmp_path, monkeypatch):
    import sys

    class ClosedPipe:
        encoding = "utf-8"

        def write(self, s):
            raise BrokenPipeError

        def flush(self):
            raise BrokenPipeError

    monkeypatch.setattr(sys, "stdout", ClosedPipe())
    html = tmp_path / "r.html"
    assert main(["sample", "growing", "-H", "3", "--html", str(html)]) == 0
    assert html.exists()


def test_korean_formats_end_to_end(tmp_path, capsys):
    import numpy as np

    v = 1000 + np.arange(36) * 5
    rows = "\n".join(f"{2021 + i // 12}.{i % 12 + 1:02d},\"{x * 1000:,}\"" for i, x in enumerate(v))
    f = tmp_path / "kosis.csv"
    f.write_bytes(("날짜,매출액\n" + rows + "\n").encode("cp949"))
    js = tmp_path / "k.json"
    assert main(["analyze", str(f), "--date-col", "날짜", "-q", "-H", "3", "--json", str(js)]) == 0
    d = json.loads(js.read_text(encoding="utf-8"))
    assert d["name"] == "매출액" and d["freq"] == "M" and len(d["history"]) == 36
    assert d["history"][-1]["value"] == 1175000


# --------------------------------------------------------------------------- round 3 regressions


def test_ranking_probabilities_match_the_reports():
    from trendai.cli import _fmt_prob, _format_ranking
    from trendai.narrative import format_prob

    df = pd.DataFrame([
        {"name": "a", "phase": "growing", "phase_ko": "성장기", "expected_change_pct": 23.2,
         "prob_increase": 1.0, "score": 90.0, "freq": "D"},
        {"name": "b", "phase": "declining", "phase_ko": "쇠퇴기", "expected_change_pct": -50.0,
         "prob_increase": 0.0, "score": 10.0, "freq": "D"},
    ], index=[1, 2])
    out = _format_ranking(df)
    assert "100%" not in out and " 0%" not in out and "100.0%" not in out
    assert ">99%" in out and "<1%" in out
    for p in (1.0, 0.0, 0.997, 0.5, 0.003):
        assert _fmt_prob(p) == format_prob(p)
    assert _fmt_prob(None) == "—"


def test_ranking_shows_each_change_with_its_horizon():
    from trendai.cli import _format_ranking

    rows = [{"name": "daily", "phase": "growing", "phase_ko": "성장기", "expected_change_pct": 23.2,
             "expected_change_ann_pct": 1169.0, "prob_increase": 0.99, "score": 90.0, "freq": "D"},
            {"name": "monthly", "phase": "stable", "phase_ko": "안정기", "expected_change_pct": 7.9,
             "expected_change_ann_pct": 7.9, "prob_increase": 0.6, "score": 60.0, "freq": "M"},
            {"name": "quarterly", "phase": "growing", "phase_ko": "성장기", "expected_change_pct": 18.0,
             "expected_change_ann_pct": 8.6, "prob_increase": 0.8, "score": 55.0, "freq": "Q"}]
    df = pd.DataFrame(rows, index=[1, 2, 3])
    out = _format_ranking(df, horizons={"daily": 30, "monthly": 12, "quarterly": 8})
    lines = out.splitlines()
    assert "예측 변화" in lines[2] and "기간" in lines[2] and "연환산" not in out
    assert "+23.2%  30일" in lines[3] and "+7.9%  12개월" in lines[4] and "+18.0%  8분기" in lines[5]
    assert "1,169" not in out                                   # no extrapolated annual rate
    assert len({_display_width(r.split("%")[0]) for r in lines[3:]}) == 1
    en = _format_ranking(df, lang="en", horizons={"daily": 30, "monthly": 12, "quarterly": 8})
    assert "Horizon" in en and "30 d" in en and "12 mo" in en
    # a shared horizon goes to the header once
    same = _format_ranking(df.iloc[1:2], horizons={"monthly": 12})
    assert "예측 변화(12개월)" in same and "기간" not in same


def test_mixed_frequency_ranking_end_to_end(tmp_path, capsys):
    d, m = sample("growing", n=120, freq="D"), sample("seasonal")
    pd.DataFrame({"date": d.index.strftime("%Y-%m-%d"), "daily": d.values}).to_csv(tmp_path / "d.csv", index=False)
    pd.DataFrame({"date": m.index.strftime("%Y-%m-%d"), "monthly": m.values}).to_csv(tmp_path / "m.csv", index=False)
    assert main(["compare", str(tmp_path / "d.csv"), str(tmp_path / "m.csv"), "-q"]) == 0
    out = capsys.readouterr().out
    assert "30일" in out and "12개월" in out and "주기가 다른 시계열" in out
    assert "예측 변화와 상승확률은" in out


def _run_into_closed_pipe(args, lines_to_read=0):
    """Run the CLI in a subprocess and close its stdout after ``lines_to_read`` lines (like ``| head``)."""
    import os
    import subprocess
    import sys

    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    proc = subprocess.Popen([sys.executable, "-m", "trendai", *args], stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, env=env)
    for _ in range(lines_to_read):
        proc.stdout.readline()
    proc.stdout.close()
    err = proc.stderr.read().decode("utf-8", "replace")
    proc.stderr.close()
    return proc.wait(timeout=120), err


@pytest.mark.parametrize("args,k", [
    (["sample", "growing", "--freq", "Y", "--n", "8"], 4),
    (["sample", "growing", "--freq", "Y", "--n", "8"], 0),
    (["sample", "all", "-H", "3"], 3),
])
def test_real_broken_pipe_exits_quietly(args, k):
    code, err = _run_into_closed_pipe(args, k)
    assert code == 0, err
    assert "Traceback" not in err and "BrokenPipe" not in err and "Exception ignored" not in err


# --------------------------------------------------------------------------- final-round regressions


def test_lang_en_column_notes_are_english(tmp_path, capsys, monkeypatch):
    """UX-16: --lang en left the column-detection and first-column notes in Korean."""
    import re

    g, d = sample("growing"), sample("declining")
    f = tmp_path / "kr.csv"
    pd.DataFrame({"날짜": g.index.strftime("%Y.%m"), "검색량": g.values, "판매량(개)": d.values}
                 ).to_csv(f, index=False)
    assert main(["analyze", str(f), "--lang", "en", "-q", "-H", "3"]) == 0
    err = capsys.readouterr().err
    note = [ln for ln in err.splitlines() if "kr.csv" in ln]
    assert note and note[0].startswith("Note: kr.csv — date column '날짜' (format: YYYY.MM), 2 value columns:")
    assert not any(re.search(r"[가-힣]", re.sub(r"'[^']*'|: .*$", "", ln)) for ln in note)
    # Korean stays the default
    assert main(["analyze", str(f), "-q", "-H", "3"]) == 0
    assert "안내: kr.csv — 날짜 열 '날짜'" in capsys.readouterr().err
    # ask: the "first column only" note and the Claude error follow --lang as well
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert main(["ask", str(f), "why?", "--lang", "en", "-H", "3"]) == 1
    err = capsys.readouterr().err
    assert "Note: using only the first column '검색량'" in err and "Error: cannot ask Claude" in err
    assert "안내:" not in err and "오류:" not in err


def test_korean_rank_title_has_no_english():
    from trendai.cli import _format_ranking

    df = pd.DataFrame([{"name": "a", "phase": "growing", "phase_ko": "성장기", "expected_change_pct": 3.0,
                        "prob_increase": 0.7, "score": 60.0, "freq": "M"}], index=[1])
    out = _format_ranking(df, horizons={"a": 12})
    # round 4 (D7): the score blends the outlook with the recent momentum (compare.rising_score), so the
    # title no longer says "높을수록 강한 상승 전망" (higher = a stronger expected rise)
    assert "■ 상승 트렌드 순위 (상승 점수: 전망 + 최근 기세, 50=중립)" in out
    assert "rising" not in out
    assert "Rising-trend ranking (rising score" in _format_ranking(df, lang="en", horizons={"a": 12})


def test_lang_en_errors_are_english(tmp_path, capsys):
    assert main(["sample", "nope", "--lang", "en"]) == 2
    err = capsys.readouterr().err
    assert err.startswith("Error: unknown sample 'nope'") and "오류" not in err
    assert main(["analyze", str(tmp_path / "missing.csv"), "--lang", "en"]) == 2
    assert capsys.readouterr().err.startswith("Data error: ")       # the data module's own text follows


def test_huge_values_end_to_end(tmp_path, capsys):
    """F8: |y| > 1e300 crashed the chart (math domain error), the CLI exited 1 and lost --json."""
    import numpy as np

    idx = pd.date_range("2020-01-01", periods=48, freq="MS")
    base = 100 + 10 * np.sin(np.arange(48) / 3) + np.arange(48)
    f = tmp_path / "huge.csv"
    pd.DataFrame({"date": idx.strftime("%Y-%m-%d"), "v": base * 1e300}).to_csv(f, index=False)
    js, html = tmp_path / "huge.json", tmp_path / "huge.html"
    assert main(["analyze", str(f), "-q", "--json", str(js), "--html", str(html)]) == 0
    assert json.loads(js.read_text(encoding="utf-8"))["name"] == "v"
    assert html.read_text(encoding="utf-8").lstrip().lower().startswith("<!doctype html")


def test_json_is_kept_when_the_html_report_fails(tmp_path, capsys, monkeypatch):
    import trendai.report as report

    def boom(*a, **k):
        raise ValueError("render failed")

    monkeypatch.setattr(report, "render_html", boom)
    js, html = tmp_path / "r.json", tmp_path / "r.html"
    assert main(["sample", "growing", "-q", "-H", "3", "--json", str(js), "--html", str(html)]) == 1
    assert json.loads(js.read_text(encoding="utf-8"))["name"] == "growing"
    assert not html.exists()
    assert "render failed" in capsys.readouterr().err


# --------------------------------------------------------------------------- round 4


def test_rank_title_does_not_overclaim_in_either_language():
    """D7: the rising score blends the outlook with the recent momentum (compare.rising_score), so the
    title says so instead of "higher = stronger expected rise" / "높을수록 강한 상승 전망"."""
    from trendai.cli import _format_ranking

    df = pd.DataFrame([{"name": "a", "phase": "growing", "phase_ko": "성장기", "expected_change_pct": 3.0,
                        "prob_increase": 0.7, "score": 60.0, "freq": "M"}], index=[1])
    ko, en = _format_ranking(df, horizons={"a": 12}), _format_ranking(df, lang="en", horizons={"a": 12})
    assert "높을수록" not in ko and "전망 + 최근 기세" in ko
    assert "stronger expected rise" not in en
    assert "■ Rising-trend ranking (rising score: outlook + recent momentum, 50 = neutral)" in en


def test_ask_error_line_is_english_under_lang_en(tmp_path, capsys, monkeypatch):
    """U7 / D (cli): NarrativeUnavailable follows lang, so 'Error: cannot ask Claude — …' has no Korean."""
    import re

    g = sample("growing")
    f = tmp_path / "g.csv"
    pd.DataFrame({"date": g.index.strftime("%Y-%m-%d"), "v": g.values}).to_csv(f, index=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert main(["ask", str(f), "why?", "--lang", "en", "-H", "3"]) == 1
    line = [ln for ln in capsys.readouterr().err.splitlines() if ln.startswith("Error: cannot ask Claude")]
    assert line and not re.search(r"[가-힣]", line[0]), line


# --------------------------------------------------------------------------- round 4, final fixer


def test_rank_basis_names_the_levels_that_were_compared():
    """F16: the note said 'the last forecast cycle vs the last observed cycle' for every ranking, but
    with weekly data the default 26-week horizon is shorter than the 52-week season (the future level
    is the seasonally adjusted 26-week mean) and yearly data compare single values."""
    from trendai.cli import _rank_basis
    from trendai.pipeline import run

    m = run(sample("growing", n=36), name="m")
    w = run(sample("growing", n=156, freq="W"), name="w")
    y = run(sample("growing", n=12, freq="Y"), name="y")
    assert len(w.forecast.mean) == 26 and w.analysis.season_length == 52
    ko_m = _rank_basis([m])
    assert ko_m == "※ 예측 변화와 상승확률은 예측 마지막 한 주기의 평균을 최근 한 주기의 평균과 비교한 값입니다."
    ko_w = _rank_basis([w, m])
    assert "예측 마지막 한 주기의 평균(예측 기간이 한 주기보다 짧으면 예측 기간 전체를 계절 보정한 평균)을" in ko_w
    assert _rank_basis([y]) == "※ 예측 변화와 상승확률은 마지막 예측값을 최근 값과 비교한 값입니다."
    assert _rank_basis([m, y]).endswith(" 연 단위 데이터는 마지막 예측값을 최근 값과 비교합니다.")
    en_w = _rank_basis([w], "en")
    assert "the forecast's last season (the whole horizon, seasonally adjusted, when it is shorter than a " \
           "season)" in en_w and "cycle" not in en_w
    assert _rank_basis([y], "en") == ("※ Exp. change and P(up) compare the last forecast value with the latest "
                                      "observation.")


def test_weekly_ranking_note_and_english_header(tmp_path, capsys):
    """F16: under a weekly ranking the note names the seasonally adjusted 26-week level; the English
    header reads 'Exp. change (26 wk)', not 'Exp. change(26 wk)'."""
    files = []
    for nm in ("growing", "fad"):
        s = sample(nm, n=156, freq="W")
        f = tmp_path / f"{nm}.csv"
        pd.DataFrame({"date": s.index.strftime("%Y-%m-%d"), "v": s.values}).to_csv(f, index=False)
        files.append(str(f))
    assert main(["compare", *files, "-q"]) == 0
    out = capsys.readouterr().out
    assert "예측 변화(26주)" in out
    assert "예측 기간이 한 주기보다 짧으면 예측 기간 전체를 계절 보정한 평균" in out
    assert main(["compare", *files, "-q", "--lang", "en"]) == 0
    out = capsys.readouterr().out
    assert "Exp. change (26 wk)" in out and "change(" not in out
    assert "(the whole horizon, seasonally adjusted, when it is shorter than a season)" in out


def test_lang_en_output_has_no_hangul(tmp_path, capsys):
    """F20: the terminal summaries stayed Korean under --lang en ('[태동기(신흥)] 신뢰도 95% · 월 단위
    120개 …'); render_text now follows --lang, so English output carries no Hangul."""
    import re

    assert main(["sample", "emerging", "--lang", "en", "-H", "6"]) == 0
    out = capsys.readouterr().out
    assert "[Emerging] confidence" in out and "Outlook ▲ upward" in out and "Method: " in out
    assert not re.search(r"[가-힣]", out), [ln for ln in out.splitlines() if re.search(r"[가-힣]", ln)]
    assert main(["compare", "--samples", "emerging,fad,seasonal,peaking", "--lang", "en", "-q", "-H", "3"]) == 0
    out = capsys.readouterr().out
    assert out.count("Future phase ") == 4 and "Rising-trend ranking" in out
    assert not re.search(r"[가-힣]", out), [ln for ln in out.splitlines() if re.search(r"[가-힣]", ln)]
    # Korean stays the default
    assert main(["sample", "emerging", "-q", "-H", "6"]) == 0
    assert "[태동기(신흥)] 신뢰도" in capsys.readouterr().out
