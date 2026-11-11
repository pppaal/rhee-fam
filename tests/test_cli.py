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
