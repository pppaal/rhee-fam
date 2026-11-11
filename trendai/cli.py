"""Command line interface.

Examples::

    trendai sample emerging --html out/emerging.html
    trendai sample all --html out/compare.html
    trendai analyze data.csv --horizon 12 --html report.html --json report.json
    trendai analyze data.csv --value-col 검색량 --llm
    trendai wiki "인공지능" --html ai.html
    trendai trends "생성형 AI" --geo KR
    trendai compare a.csv b.csv --html compare.html
    trendai ask data.csv "앞으로 6개월 동안 어떻게 될까?"
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

from . import __version__
from .data import DataError, load_table
from .types import TrendReport

FREQ_CHOICES = ["D", "W", "M", "Q", "Y"]


# --------------------------------------------------------------------------- helpers


def _setup_stdout() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except (AttributeError, ValueError):  # pragma: no cover - non-reconfigurable stream
            pass


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--horizon", "-H", type=int, default=None,
                   help="예측 기간(주기 수). 기본: 약 1년/1주기")
    p.add_argument("--freq", choices=FREQ_CHOICES, default=None,
                   help="데이터 주기 강제 지정 (D/W/M/Q/Y). 기본: 자동 감지")
    p.add_argument("--html", metavar="PATH", default=None, help="HTML 리포트 저장 경로")
    p.add_argument("--json", metavar="PATH", default=None, help="JSON 결과 저장 경로")
    p.add_argument("--llm", action="store_true",
                   help="Claude로 해석문 생성 (ANTHROPIC_API_KEY 필요, 실패 시 규칙 기반으로 대체)")
    p.add_argument("--model", default="claude-opus-5-5", help="--llm 사용 시 Claude 모델 ID")
    p.add_argument("--lang", choices=["ko", "en"], default="ko", help="해석문 언어")
    p.add_argument("--quiet", "-q", action="store_true", help="해석문 출력 생략 (요약만)")


def _run_kwargs(args: argparse.Namespace) -> dict:
    return dict(freq=args.freq, horizon=args.horizon, use_llm=args.llm, lang=args.lang,
                model=args.model)


def _ensure_parent(path: str) -> Path:
    p = Path(path)
    if p.parent and not p.parent.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _print_report(report: TrendReport, quiet: bool) -> None:
    from .report import render_text

    print(render_text(report))
    if not quiet:
        print()
        print(report.narrative)
        print()


def _emit(reports: dict[str, TrendReport], args: argparse.Namespace) -> int:
    """Print reports and write HTML/JSON outputs. Returns an exit code."""
    from .compare import rank_trends
    from .pipeline import report_to_json, reports_to_json
    from .report import render_compare_html, render_html

    if not reports:
        print("오류: 분석에 성공한 시계열이 없습니다.", file=sys.stderr)
        return 1
    many = len(reports) > 1
    for rep in reports.values():
        _print_report(rep, args.quiet or many)
    ranking = None
    if many:
        ranking = rank_trends(reports)
        print(_format_ranking(ranking))
    if args.html:
        out = _ensure_parent(args.html)
        if many:
            render_compare_html(list(reports.values()), ranking, path=out)
            stem = out.with_suffix("")
            for nm, rep in reports.items():
                render_html(rep, path=f"{stem}_{_slug(nm)}.html")
            print(f"HTML 비교 리포트 저장: {out} (+ 개별 리포트 {len(reports)}개)")
        else:
            render_html(next(iter(reports.values())), path=out)
            print(f"HTML 리포트 저장: {out}")
    if args.json:
        out = _ensure_parent(args.json)
        text = reports_to_json(reports.values()) if many else report_to_json(next(iter(reports.values())))
        out.write_text(text, encoding="utf-8")
        print(f"JSON 저장: {out}")
    return 0


def _slug(name: str) -> str:
    keep = [c if (c.isalnum() or c in "-_") else "_" for c in str(name)]
    s = "".join(keep).strip("_")
    return s[:60] or "series"


def _format_ranking(df: pd.DataFrame) -> str:
    lines = ["", "■ 상승 트렌드 순위 (rising score: 50=중립, 높을수록 강한 상승 전망)"]
    for rank, row in df.iterrows():
        exp = row["expected_change_pct"]
        exp_s = "  n/a " if exp is None or pd.isna(exp) else f"{exp:+6.1f}%"
        lines.append(f"  {rank:>2}. {str(row['name'])[:28]:<28} {row['phase_ko']:<10} "
                     f"예상변화 {exp_s}  상승확률 {row['prob_increase'] * 100:5.1f}%  "
                     f"점수 {row['score']:5.1f}")
    return "\n".join(lines)


def _load(path: str, date_col: Optional[str], value_cols: Optional[list[str]]) -> dict[str, pd.Series]:
    return load_table(path, date_col=date_col, value_cols=value_cols)


# --------------------------------------------------------------------------- commands


def cmd_analyze(args: argparse.Namespace) -> int:
    from .pipeline import run_many

    series = _load(args.file, args.date_col, args.value_col)
    reports = run_many(series, **_run_kwargs(args))
    return _emit(reports, args)


def cmd_sample(args: argparse.Namespace) -> int:
    from .pipeline import run_many
    from .samples import DESCRIPTIONS, list_samples, sample

    names = list_samples() if args.name == "all" else [args.name]
    for nm in names:
        if nm not in DESCRIPTIONS:
            print(f"오류: 알 수 없는 샘플 {nm!r}. 사용 가능: {', '.join(list_samples())}", file=sys.stderr)
            return 2
    freq = args.freq or "M"
    series = {nm: sample(nm, n=args.n, freq=freq, seed=args.seed) for nm in names}
    print("※ 합성(가상) 샘플 데이터입니다 — 실제 데이터가 아닙니다.")
    reports = run_many(series, **_run_kwargs(args))
    return _emit(reports, args)


def cmd_list_samples(args: argparse.Namespace) -> int:
    from .samples import DESCRIPTIONS

    for nm, desc in DESCRIPTIONS.items():
        print(f"  {nm:<10} {desc}")
    return 0


def cmd_wiki(args: argparse.Namespace) -> int:
    from .connectors import ConnectorError, fetch_wikipedia_pageviews
    from .pipeline import run_many

    series = {}
    for article in args.article:
        try:
            s = fetch_wikipedia_pageviews(article, lang=args.wiki_lang, start=args.start, end=args.end,
                                          granularity=args.granularity)
        except ConnectorError as exc:
            print(f"오류: {exc}", file=sys.stderr)
            return 1
        series[f"{article} (위키백과 조회수)"] = s
    reports = run_many(series, **_run_kwargs(args))
    return _emit(reports, args)


def cmd_trends(args: argparse.Namespace) -> int:
    from .connectors import ConnectorError, fetch_google_trends
    from .pipeline import run_many

    series = {}
    for kw in args.keyword:
        try:
            series[f"{kw} (Google Trends)"] = fetch_google_trends(kw, timeframe=args.timeframe, geo=args.geo)
        except ConnectorError as exc:
            print(f"오류: {exc}", file=sys.stderr)
            return 1
    reports = run_many(series, **_run_kwargs(args))
    return _emit(reports, args)


def cmd_compare(args: argparse.Namespace) -> int:
    from .pipeline import run_many
    from .samples import sample

    series: dict[str, pd.Series] = {}
    for f in args.files:
        for nm, s in _load(f, args.date_col, args.value_col).items():
            key = nm if nm not in series else f"{Path(f).stem}:{nm}"
            series[key] = s
    if args.samples:
        for nm in args.samples.split(","):
            series[nm.strip()] = sample(nm.strip(), freq=args.freq or "M")
    if len(series) < 1:
        print("오류: 비교할 시계열이 없습니다 (파일 또는 --samples 지정).", file=sys.stderr)
        return 2
    reports = run_many(series, **_run_kwargs(args))
    return _emit(reports, args)


def cmd_ask(args: argparse.Namespace) -> int:
    from .narrative import NarrativeUnavailable, ask
    from .pipeline import run

    series = _load(args.file, args.date_col, args.value_col)
    nm, s = next(iter(series.items()))
    if len(series) > 1:
        print(f"※ 여러 열 중 첫 번째 열 '{nm}'만 사용합니다 (--value-col로 지정 가능).", file=sys.stderr)
    report = run(s, name=nm, freq=args.freq, horizon=args.horizon, lang=args.lang)
    try:
        answer = ask(report, args.question, lang=args.lang, model=args.model)
    except NarrativeUnavailable as exc:
        print(f"오류: Claude에 질문할 수 없습니다 — {exc}", file=sys.stderr)
        return 1
    print(answer)
    return 0


# --------------------------------------------------------------------------- parser


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="trendai",
        description="과거→현재→미래: 시계열 트렌드를 분석하고 예측하는 AI",
    )
    p.add_argument("--version", action="version", version=f"trendai {__version__}")
    sub = p.add_subparsers(dest="command", metavar="COMMAND")

    a = sub.add_parser("analyze", help="CSV/Excel/JSON 파일의 시계열 분석·예측")
    a.add_argument("file")
    a.add_argument("--date-col", default=None, help="날짜 열 이름 (기본: 자동 감지)")
    a.add_argument("--value-col", action="append", default=None,
                   help="값 열 이름 (여러 번 지정 가능, 기본: 숫자 열 전체)")
    _add_common(a)
    a.set_defaults(func=cmd_analyze)

    s = sub.add_parser("sample", help="내장 합성 샘플로 데모 실행 (all = 전체 비교)")
    s.add_argument("name", nargs="?", default="emerging")
    s.add_argument("--n", type=int, default=120, help="관측치 수")
    s.add_argument("--seed", type=int, default=0)
    _add_common(s)
    s.set_defaults(func=cmd_sample)

    ls = sub.add_parser("list-samples", help="내장 샘플 목록")
    ls.set_defaults(func=cmd_list_samples)

    w = sub.add_parser("wiki", help="위키백과 문서 조회수 트렌드 (인터넷 필요)")
    w.add_argument("article", nargs="+", help="문서 제목 (여러 개면 비교)")
    w.add_argument("--wiki-lang", default="ko", help="위키백과 언어 (ko, en, ...)")
    w.add_argument("--start", default=None, help="시작일 YYYY-MM-DD (기본 2016-01-01)")
    w.add_argument("--end", default=None, help="종료일 YYYY-MM-DD (기본: 지난달 말)")
    w.add_argument("--granularity", choices=["monthly", "daily"], default="monthly")
    _add_common(w)
    w.set_defaults(func=cmd_wiki)

    t = sub.add_parser("trends", help="Google Trends 검색 관심도 (pytrends 필요)")
    t.add_argument("keyword", nargs="+")
    t.add_argument("--timeframe", default="today 5-y")
    t.add_argument("--geo", default="", help="국가 코드 (예: KR)")
    _add_common(t)
    t.set_defaults(func=cmd_trends)

    c = sub.add_parser("compare", help="여러 시계열을 분석해 상승 트렌드 순위 매기기")
    c.add_argument("files", nargs="*")
    c.add_argument("--samples", default=None, help="쉼표로 구분한 내장 샘플 이름")
    c.add_argument("--date-col", default=None)
    c.add_argument("--value-col", action="append", default=None)
    _add_common(c)
    c.set_defaults(func=cmd_compare)

    q = sub.add_parser("ask", help="분석 결과를 바탕으로 Claude에게 질문")
    q.add_argument("file")
    q.add_argument("question")
    q.add_argument("--date-col", default=None)
    q.add_argument("--value-col", action="append", default=None)
    q.add_argument("--horizon", "-H", type=int, default=None)
    q.add_argument("--freq", choices=FREQ_CHOICES, default=None)
    q.add_argument("--model", default="claude-opus-5-5")
    q.add_argument("--lang", choices=["ko", "en"], default="ko")
    q.set_defaults(func=cmd_ask)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    _setup_stdout()
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return 0
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("default")
            return int(args.func(args) or 0)
    except DataError as exc:
        print(f"데이터 오류: {exc}", file=sys.stderr)
        return 2
    except FileNotFoundError as exc:
        print(f"파일 없음: {exc}", file=sys.stderr)
        return 2
