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
