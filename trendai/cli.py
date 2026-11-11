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

Exit codes: 0 success, 1 runtime failure (network, Claude, unexpected error),
2 input error (bad file / data / arguments).  Data notes and warnings are
printed to stderr as single ``안내:`` / ``경고:`` lines -- ``Note:`` / ``Warning:``
with ``--lang en``, where the data layer's messages (``DataError`` /
``DataWarning`` / ``DataNotice``, the catalog of :mod:`trendai.messages`) are
rendered in English too, like the CLI's own notes and errors.  ``--json`` is
written before ``--html``, so a rendering problem never loses the JSON result.
"""

from __future__ import annotations

import argparse
import inspect
import os
import sys
import unicodedata
import warnings
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import pandas as pd

from . import __version__
from .data import DataError, DataNotice, load_table
from .messages import format_label, localise, msg
from .types import TrendReport

FREQ_CHOICES = ["D", "W", "M", "Q", "Y"]

# CLI chrome per language (the terminal summary follows --lang too; the HTML reports are Korean).  The data
# module's messages (DataError / DataWarning / DataNotice) are rendered in the same language (messages.localise).
_MSG: dict[str, dict[str, str]] = {
    "ko": {
        "synthetic": "※ 합성(가상) 샘플 데이터입니다 — 실제 데이터가 아닙니다.",
        # the score blends the outlook with the recent momentum (compare.rising_score): "higher = a
        # stronger expected rise" overclaimed
        "rank_title": "■ 상승 트렌드 순위 (상승 점수: 전망 + 최근 기세, 50=중립)",
        "rank_cols": "예측 변화",
        "rank_span": "기간",
        "rank_prob": "상승확률",
        "rank_score": "점수",
        "mixed_freq": ("※ 주기가 다른 시계열이 섞여 있어, 점수는 변화율을 연 단위(모멘텀은 월 단위)로 환산해 "
                       "매겼습니다. 예측 변화는 각 시계열의 기간 기준입니다."),
        # the levels of forecast.outlook_levels: the forecast's last season -- the whole horizon,
        # seasonally adjusted, when it is shorter (weekly default: 26 of 52 weeks) -- versus the
        # last observed season; yearly data compare single values (see _rank_basis)
        "rank_cols_span": "{cols}({span})",
        "rank_basis": "※ 예측 변화와 상승확률은 예측 마지막 한 주기의 평균{partial}을 최근 한 주기의 평균과 비교한 값입니다.",
        "rank_basis_partial": "(예측 기간이 한 주기보다 짧으면 예측 기간 전체를 계절 보정한 평균)",
        "rank_basis_yearly": " 연 단위 데이터는 마지막 예측값을 최근 값과 비교합니다.",
        "rank_basis_only_yearly": "※ 예측 변화와 상승확률은 마지막 예측값을 최근 값과 비교한 값입니다.",
        # why a row shows "—" (report._change_undefined: the terminal summary's own reason text)
        "rank_undefined": "※ {cols} — ({why}): {names}",
        "rank_undefined_prob": "예측 변화·상승확률",
        "narr_hidden": ("※ 시계열이 여러 개라 개별 해석문은 생략했습니다 — HTML(--html)·JSON(--json) "
                        "리포트 또는 --value-col 로 하나씩 실행해 확인하세요."),
        "html_one": "HTML 리포트 저장: {path}",
        "html_many": "HTML 비교 리포트 저장: {path} (+ 개별 리포트 {n}개)",
        "json": "JSON 저장: {path}",
        "none_ok": "오류: 분석에 성공한 시계열이 없습니다.",
        "columns": "안내: {file} — 날짜 열 '{date_col}' (형식: {date_format}), 값 열 {n}개: {names}",
        "first_col": "안내: 여러 열 중 첫 번째 열 '{name}'만 사용합니다 (--value-col 로 지정 가능).",
        "ask_failed": "오류: Claude에 질문할 수 없습니다 — {exc}",
        "error": "오류: {exc}",
        "unknown_sample": "오류: 알 수 없는 샘플 {bad}. 사용 가능: {names}",
        "no_series": "오류: 비교할 시계열이 없습니다 (파일 또는 --samples 지정).",
        "save_failed": "오류: 파일을 저장할 수 없습니다 — {exc}",
        "data_error": "데이터 오류: {exc}",
        "not_found": "파일 없음: {exc}",
        "interrupted": "중단되었습니다.",
        "unexpected": ("오류: 예기치 못한 문제가 발생했습니다 ({kind}: {exc}). "
                       "자세한 내용은 TRENDAI_DEBUG=1 로 다시 실행해 확인하세요."),
        "warn_notice": "안내",
        "warn_warning": "경고",
        "wiki_series": "{article} (위키백과 조회수)",
        "arg_int": "{what}: 정수를 입력하세요 (입력: {text!r})",
        "arg_min": "{what}: {lo} 이상이어야 합니다 (입력: {v})",
        "what_horizon": "예측 기간",
        "what_n": "관측치 수(--n)",
    },
    "en": {
        "synthetic": "※ Synthetic sample data — not real data.",
        "rank_title": "■ Rising-trend ranking (rising score: outlook + recent momentum, 50 = neutral)",
        "rank_cols": "Exp. change",
        "rank_span": "Horizon",
        "rank_prob": "P(up)",
        "rank_score": "Score",
        "mixed_freq": ("※ Series of different frequencies: changes were annualised (momentum per month) before "
                       "scoring. Expected changes are shown over each series' own horizon."),
        "rank_cols_span": "{cols} ({span})",
        "rank_basis": ("※ Exp. change and P(up) compare the mean of the forecast's last season{partial} with the "
                       "mean of the last observed season."),
        "rank_basis_partial": " (the whole horizon, seasonally adjusted, when it is shorter than a season)",
        "rank_basis_yearly": " For yearly data they compare the last forecast value with the latest observation.",
        "rank_basis_only_yearly": ("※ Exp. change and P(up) compare the last forecast value with the latest "
                                   "observation."),
        "rank_undefined": "※ {cols} — ({why}): {names}",
        "rank_undefined_prob": "Exp. change, P(up)",
        "narr_hidden": ("※ Several series: individual narratives are not printed — see the HTML (--html) / "
                        "JSON (--json) output or run one column at a time with --value-col."),
        "html_one": "HTML report saved: {path}",
        "html_many": "HTML comparison saved: {path} (+ {n} individual reports)",
        "json": "JSON saved: {path}",
        "none_ok": "Error: no series could be analysed.",
        "columns": "Note: {file} — date column '{date_col}' (format: {date_format}), {n} value column{s}: {names}",
        "first_col": "Note: using only the first column '{name}' of several (choose one with --value-col).",
        "ask_failed": "Error: cannot ask Claude — {exc}",
        "error": "Error: {exc}",
        "unknown_sample": "Error: unknown sample {bad}. Available: {names}",
        "no_series": "Error: nothing to compare (give files or --samples).",
        "save_failed": "Error: cannot save the file — {exc}",
        "data_error": "Data error: {exc}",
        "not_found": "File not found: {exc}",
        "interrupted": "Interrupted.",
        "unexpected": ("Error: unexpected problem ({kind}: {exc}). "
                       "Run again with TRENDAI_DEBUG=1 for details."),
        "warn_notice": "Note",
        "warn_warning": "Warning",
        "wiki_series": "{article} (Wikipedia page views)",
        "arg_int": "{what} must be an integer: {text!r}",
        "arg_min": "{what} must be at least {lo}: {v}",
        "what_horizon": "the forecast horizon",
        "what_n": "the number of observations (--n)",
    },
}


def _t(args: Any, key: str, **kw: Any) -> str:
    """Message ``key`` in the language of ``args.lang`` (or ``args`` itself when it is a language code)."""
    lang = args if isinstance(args, str) else getattr(args, "lang", "ko")
    return _MSG.get(lang, _MSG["ko"])[key].format(**kw)


# --------------------------------------------------------------------------- output helpers


class _Out:
    """stdout writer that survives a closed pipe (``trendai ... | head``)."""

    broken = False

    @classmethod
    def say(cls, *parts: Any, file: Any = None) -> None:
        stream = file if file is not None else sys.stdout
        if cls.broken and stream is sys.stdout:
            return
        try:
            print(*parts, file=stream)
            if stream is sys.stdout:
                stream.flush()
        except BrokenPipeError:
            cls._silence()

    @classmethod
    def _silence(cls) -> None:
        cls.broken = True
        try:  # point fd 1 at /dev/null so the interpreter's final flush does not fail again
            devnull = os.open(os.devnull, os.O_WRONLY)
            os.dup2(devnull, sys.stdout.fileno())
            os.close(devnull)
        except (OSError, ValueError, AttributeError):  # pragma: no cover - not a real fd (tests)
            pass


def _say(*parts: Any) -> None:
    _Out.say(*parts)


def _err(*parts: Any) -> None:
    _Out.say(*parts, file=sys.stderr)


def _setup_stdout() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except (AttributeError, ValueError):  # pragma: no cover - non-reconfigurable stream
            pass


_SEEN_WARNINGS: set[str] = set()
#: language of the warning lines (set from --lang by main(); the warnings hook has no args)
_LANG = "ko"


def _show_warning(message: Any, category: type, filename: str, lineno: int, file: Any = None,
                  line: Optional[str] = None) -> None:
    """Print warnings as one clean line (no file path / source line), each text once per run.

    ``안내:`` / ``경고:`` (``Note:`` / ``Warning:`` with ``--lang en``); a catalog message of the
    data layer is rendered in that language (:func:`trendai.messages.localise`)."""
    prefix = _t(_LANG, "warn_notice" if issubclass(category, DataNotice) else "warn_warning")
    text = f"{prefix}: {localise(message, _LANG)}"
    if text in _SEEN_WARNINGS:
        return
    _SEEN_WARNINGS.add(text)
    _err(text)


def _install_warning_filters() -> None:
    warnings.simplefilter("ignore")
    warnings.filterwarnings("default", category=UserWarning)
    warnings.filterwarnings("default", category=RuntimeWarning)
    # numpy floating-point notes are internal details, not user-facing problems
    for pat in ("overflow encountered", "invalid value encountered", "divide by zero",
                "Mean of empty slice", "Degrees of freedom", "underflow encountered",
                "All-NaN slice"):
        warnings.filterwarnings("ignore", message=pat, category=RuntimeWarning)
    warnings.showwarning = _show_warning


# --------------------------------------------------------------------------- argument helpers


def _int_at_least(lo: int, what: str) -> Callable[[str], int]:
    """argparse type: an integer >= ``lo``; ``what`` is a ``_MSG`` key naming the option (the
    message follows --lang, read from the command line before parsing, see :func:`main`)."""
    def conv(text: str) -> int:
        try:
            v = int(text)
        except (TypeError, ValueError):
            raise argparse.ArgumentTypeError(_t(_LANG, "arg_int", what=_t(_LANG, what), text=text)) from None
        if v < lo:
            raise argparse.ArgumentTypeError(_t(_LANG, "arg_min", what=_t(_LANG, what), lo=lo, v=v))
        return v
    return conv


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--horizon", "-H", type=_int_at_least(1, "what_horizon"), default=None,
                   help="예측 기간(주기 수, 1 이상). 기본: 약 1년/1주기")
    p.add_argument("--freq", choices=FREQ_CHOICES, default=None,
                   help="데이터 주기 강제 지정 (D/W/M/Q/Y). 기본: 자동 감지")
    p.add_argument("--html", metavar="PATH", default=None, help="HTML 리포트 저장 경로")
    p.add_argument("--json", metavar="PATH", default=None, help="JSON 결과 저장 경로")
    p.add_argument("--llm", action="store_true",
                   help="Claude로 해석문 생성 (ANTHROPIC_API_KEY 필요, 실패 시 규칙 기반으로 대체)")
    p.add_argument("--model", default="claude-opus-5-5", help="--llm 사용 시 Claude 모델 ID")
    p.add_argument("--lang", choices=["ko", "en"], default="ko",
                   help="해석문·터미널 요약·CLI 안내와 데이터 오류·경고의 언어 (HTML 리포트와 도움말은 한국어)")
    p.add_argument("--quiet", "-q", action="store_true", help="해석문 출력 생략 (요약만)")


def _run_kwargs(args: argparse.Namespace) -> dict:
    return dict(freq=args.freq, horizon=args.horizon, use_llm=args.llm, lang=args.lang,
                model=args.model)


def _ensure_parent(path: str) -> Path:
    p = Path(path)
    if p.is_dir():
        raise DataError(msg("save_path_is_dir", path=str(p)))
    if p.parent and not p.parent.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
    return p


# --------------------------------------------------------------------------- formatting


def _char_width(ch: str) -> int:
    if unicodedata.combining(ch):
        return 0
    return 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1


def _dw(s: str) -> int:
    """Terminal display width (East-Asian wide characters count 2)."""
    return sum(_char_width(c) for c in s)


def _pad(s: Any, width: int, align: str = "<") -> str:
    """Truncate/pad ``s`` to exactly ``width`` display columns."""
    txt = str(s)
    if _dw(txt) > width:
        out, w = "", 0
        for c in txt:
            cw = _char_width(c)
            if w + cw > width - 1:
                break
            out, w = out + c, w + cw
        txt = out + "…"
    fill = " " * max(0, width - _dw(txt))
    return fill + txt if align == ">" else txt + fill


def _fmt_pct(x: Any) -> str:
    """``+12.3%`` / ``−4.0%`` exactly as the reports print it (``report._fmt_pct``); ``—`` when missing."""
    from .report import _fmt_pct as fmt

    return fmt(x)


def _fmt_prob(p: Any) -> str:
    """Probability as a whole percentage, the same text as the reports and the narrative
    (``>99%`` / ``<1%`` at and near the bounds, exactly 1.0 / 0.0 included; ``—`` when missing)."""
    from .report import _fmt_prob as fmt

    return fmt(p)


# forecast-horizon units per language ("12개월" / "12 mo")
_SPAN_UNIT: dict[str, dict[str, str]] = {
    "ko": {"D": "일", "W": "주", "M": "개월", "Q": "분기", "Y": "년"},
    "en": {"D": " d", "W": " wk", "M": " mo", "Q": " qtr", "Y": " yr"},
}


def _span(h: Any, freq: Any, lang: str) -> str:
    """Forecast horizon as text (``12개월``, ``30일`` / ``12 mo``); ``''`` when unknown."""
    try:
        k = int(h)
    except (TypeError, ValueError):
        return ""
    if k <= 0:
        return ""
    units = _SPAN_UNIT.get(lang, _SPAN_UNIT["ko"])
    return f"{k:,}{units.get(str(freq), '')}"


def _format_ranking(df: pd.DataFrame, lang: str = "ko",
                    horizons: Optional[dict[str, int]] = None,
                    tentative: Optional[Sequence[str]] = None) -> str:
    """Ranking table aligned by display width (Korean names/phases count double).

    The expected change is shown as reported (``expected_change_pct``), always
    with its forecast horizon: in the header when every series shares it
    (``예측 변화(12개월)``), otherwise in a ``기간`` column next to the value, so a
    30-day change of a daily series is never read as a 12-month one.
    ``horizons`` maps a series name to its number of forecast steps; the phases
    of the series named in ``tentative`` (too few observations for a lifecycle
    verdict, :func:`narrative.phase_is_tentative`) are marked ``·잠정`` / ``(tentative)``.
    """
    msg = _MSG.get(lang, _MSG["ko"])
    horizons = horizons or {}
    names = [str(n) for n in df["name"]]
    freqs = list(df["freq"]) if "freq" in df.columns else [None] * len(names)
    spans = [_span(horizons.get(nm), fq, lang) for nm, fq in zip(names, freqs)]
    shared = spans[0] if spans and len(set(spans)) == 1 else None
    name_w = min(28, max([_dw(n) for n in names] + [4]))
    marked = {str(n) for n in (tentative or ())}
    phases = [str(p) + ((" (tentative)" if lang == "en" else "·잠정") if nm in marked else "")
              for p, nm in zip(df["phase_ko"] if lang == "ko" else df["phase"], names)]
    phase_w = max([_dw(p) for p in phases] + [4])
    exp_head = msg["rank_cols_span"].format(cols=msg["rank_cols"], span=shared) if shared else msg["rank_cols"]
    exps = [_fmt_pct(v) for v in df["expected_change_pct"]]
    exp_w = max([_dw(exp_head)] + [_dw(e) for e in exps])
    span_w = 0 if shared is not None else max([_dw(msg["rank_span"])] + [_dw(sp) for sp in spans])
    span_head = f"  {_pad(msg['rank_span'], span_w)}" if span_w else ""
    head = (f"  {'#':>3}  {_pad('', name_w)}  {_pad('', phase_w)}  {_pad(exp_head, exp_w, '>')}{span_head}  "
            f"{_pad(msg['rank_prob'], 8, '>')}  {_pad(msg['rank_score'], 6, '>')}")
    lines = ["", msg["rank_title"], head.rstrip()]
    for (rank, row), nm, ph, ex, sp in zip(df.iterrows(), names, phases, exps, spans):
        score = "{:.1f}".format(float(row["score"]))
        span_cell = f"  {_pad(sp, span_w)}" if span_w else ""
        lines.append(f"  {str(rank) + '.':>3}  {_pad(nm, name_w)}  {_pad(ph, phase_w)}  "
                     f"{_pad(ex, exp_w, '>')}{span_cell}  {_pad(_fmt_prob(row['prob_increase']), 8, '>')}  "
                     f"{_pad(score, 6, '>')}")
    return "\n".join(lines)


def _rank_basis(reports: Any, lang: str = "ko") -> str:
    """The note under the ranking: which two levels the expected change and P(up) compare.

    The levels of :func:`forecast.outlook_levels`: the mean of the forecast's last
    season versus the mean of the last observed season.  Only what applies to the
    ranked series is said: a horizon shorter than the season (weekly data by
    default: 26 of 52 weeks) uses the whole horizon, seasonally adjusted, and
    yearly data (a one-period season) compare the last forecast value with the
    latest observation.
    """
    from .forecast import outlook_levels

    partial = yearly = seasonal = False
    for rep in reports:
        m = int(getattr(rep.analysis, "season_length", 1) or 1)
        if m <= 1:
            yearly = True
            continue
        seasonal = True
        try:
            k_fut = int(outlook_levels(rep.analysis, rep.forecast).k_fut)
        except Exception:  # noqa: BLE001 - hand-made reports: the forecast length decides
            k_fut = min(m, len(getattr(rep.forecast, "mean", ())) or m)
        partial = partial or k_fut < m
    if yearly and not seasonal:
        return _t(lang, "rank_basis_only_yearly")
    text = _t(lang, "rank_basis", partial=_t(lang, "rank_basis_partial") if partial else "")
    return text + (_t(lang, "rank_basis_yearly") if yearly else "")


def _rank_undefined_notes(reports: Any, lang: str = "ko") -> list[str]:
    """One note per reason a ranked series shows ``—`` as its expected change (``※ 예측 변화 —
    (현재 수준이 음수): loss``), worded as the series' own terminal summary words it
    (``report._change_undefined``); an all-zero history also has no ``상승확률``."""
    from .narrative import prob_undefined
    from .report import _change_undefined_short

    groups: dict[tuple[str, bool], list[str]] = {}
    for rep in reports:
        why = _change_undefined_short(rep, lang)
        if why:
            no_prob = prob_undefined(rep.analysis) is not None
            groups.setdefault((why, no_prob), []).append(str(rep.name))
    out = []
    for (why, no_prob), names in groups.items():
        cols = _t(lang, "rank_undefined_prob") if no_prob else _t(lang, "rank_cols")
        out.append(_t(lang, "rank_undefined", cols=cols, why=why, names=", ".join(names)))
    return out


def _unique_slugs(names: Sequence[str]) -> dict[str, str]:
    """File-name-safe slug per name, made unique (case-insensitively) with -2, -3 ..."""
    out: dict[str, str] = {}
    used: set[str] = set()
    for nm in names:
        base = _slug(nm)
        slug, k = base, 2
        while slug.lower() in used:
            slug, k = f"{base}-{k}", k + 1
        used.add(slug.lower())
        out[nm] = slug
    return out


def _slug(name: str) -> str:
    keep = [c if (c.isalnum() or c in "-_") else "_" for c in str(name)]
    s = "".join(keep)
    while "__" in s:
        s = s.replace("__", "_")
    return s.strip("_")[:60] or "series"


# --------------------------------------------------------------------------- core


def _print_report(report: TrendReport, quiet: bool, lang: str = "ko") -> None:
    from .report import render_text

    # the terminal summary in the language of the narrative and the CLI notes (--lang en: no Hangul)
    _say(render_text(report, lang=lang))
    if not quiet:
        _say()
        _say(report.narrative)
        _say()


def _analyse(series: dict[str, pd.Series], args: argparse.Namespace) -> tuple[dict[str, TrendReport], int]:
    """Run the pipeline on every series. Returns (reports, exit code if nothing succeeded)."""
    from .pipeline import run_many

    errors: dict[str, Exception] = {}
    reports = run_many(series, errors=errors, **_run_kwargs(args))
    if reports:
        return reports, 0
    data_only = bool(errors) and all(isinstance(e, DataError) for e in errors.values())
    return reports, (2 if data_only or not errors else 1)


def _emit(reports: dict[str, TrendReport], args: argparse.Namespace, fail_code: int = 1) -> int:
    """Print reports and write HTML/JSON outputs. Returns an exit code."""
    from .compare import rank_trends
    from .pipeline import report_to_json, reports_to_json
    from .report import render_compare_html, render_html

    if not reports:
        _err(_t(args, "none_ok"))
        return fail_code
    many = len(reports) > 1
    for rep in reports.values():
        _print_report(rep, args.quiet or many, getattr(args, "lang", "ko"))
    ranking = None
    if many:
        ranking = rank_trends(reports)
        from .narrative import phase_is_tentative

        horizons = {str(rep.name): len(rep.forecast.mean) for rep in reports.values()}
        tentative = [str(rep.name) for rep in reports.values() if phase_is_tentative(rep.analysis)]
        _say(_format_ranking(ranking, getattr(args, "lang", "ko"), horizons, tentative))
        _say(_rank_basis(reports.values(), getattr(args, "lang", "ko")))
        for note in _rank_undefined_notes(reports.values(), getattr(args, "lang", "ko")):
            _say(note)
        if ranking["freq"].nunique() > 1:
            _say(_t(args, "mixed_freq"))
        if not args.quiet:
            _say(_t(args, "narr_hidden"))
    try:
        # JSON first: it does not depend on the (much larger) HTML rendering, so a rendering
        # problem never costs the user the machine-readable result
        if args.json:
            out = _ensure_parent(args.json)
            text = reports_to_json(reports.values()) if many else report_to_json(next(iter(reports.values())))
            out.write_text(text, encoding="utf-8")
            _say(_t(args, "json", path=out))
        if args.html:
            out = _ensure_parent(args.html)
            if many:
                slugs = _unique_slugs(list(reports))
                stem = out.with_suffix("")
                links = {nm: f"{stem.name}_{slugs[nm]}.html" for nm in reports}
                kwargs: dict[str, Any] = {}
                if "links" in inspect.signature(render_compare_html).parameters:
                    kwargs["links"] = links
                render_compare_html(list(reports.values()), ranking, path=out, **kwargs)
                for nm, rep in reports.items():
                    render_html(rep, path=out.parent / links[nm])
                _say(_t(args, "html_many", path=out, n=len(reports)))
            else:
                render_html(next(iter(reports.values())), path=out)
                _say(_t(args, "html_one", path=out))
    except OSError as exc:
        _err(_t(args, "save_failed", exc=exc))
        return 1
    return 0


def _load(path: str, date_col: Optional[str], value_cols: Optional[list[str]],
          lang: str = "ko") -> dict[str, pd.Series]:
    series = load_table(path, date_col=date_col, value_cols=value_cols)
    info = next(iter(series.values())).attrs.get("trendai", {}) if series else {}
    if info:
        names = ", ".join(list(series)[:6]) + (" …" if len(series) > 6 else "")
        # the format label and the wide-table placeholder are the data module's words: in --lang
        date_col = info.get("date_col")
        if date_col == "(열 머리글)":
            date_col = format_label(date_col, lang)
        _err(_t(lang, "columns", file=Path(path).name, date_col=date_col,
                date_format=format_label(info.get("date_format"), lang), n=len(series), names=names,
                s="" if len(series) == 1 else "s"))
    return series


# --------------------------------------------------------------------------- commands


def cmd_analyze(args: argparse.Namespace) -> int:
    series = _load(args.file, args.date_col, args.value_col, args.lang)
    reports, code = _analyse(series, args)
    return _emit(reports, args, fail_code=code)


def _check_samples(names: Sequence[str], lang: str = "ko") -> Optional[str]:
    from .samples import DESCRIPTIONS, list_samples

    bad = [nm for nm in names if nm not in DESCRIPTIONS]
    if bad:
        return _t(lang, "unknown_sample", bad=", ".join(map(repr, bad)), names=", ".join(list_samples()))
    return None


def cmd_sample(args: argparse.Namespace) -> int:
    from .samples import list_samples, sample

    names = list_samples() if args.name == "all" else [args.name]
    problem = _check_samples(names, args.lang)
    if problem:
        _err(problem)
        return 2
    freq = args.freq or "M"
    series = {nm: sample(nm, n=args.n, freq=freq, seed=args.seed) for nm in names}
    _say(_t(args, "synthetic"))
    reports, code = _analyse(series, args)
    return _emit(reports, args, fail_code=code)


def cmd_list_samples(args: argparse.Namespace) -> int:
    from .samples import DESCRIPTIONS

    width = max(_dw(nm) for nm in DESCRIPTIONS)
    for nm, desc in DESCRIPTIONS.items():
        _say(f"  {_pad(nm, width)}  {desc}")
    return 0


def cmd_wiki(args: argparse.Namespace) -> int:
    from .connectors import ConnectorError, fetch_wikipedia_pageviews

    series = {}
    for article in args.article:
        try:
            s = fetch_wikipedia_pageviews(article, lang=args.wiki_lang, start=args.start, end=args.end,
                                          granularity=args.granularity)
        except ConnectorError as exc:   # a catalog message: rendered in --lang
            _err(_t(args, "error", exc=localise(exc, args.lang)))
            return 1
        series[_t(args, "wiki_series", article=article)] = s
    reports, code = _analyse(series, args)
    return _emit(reports, args, fail_code=code)


def cmd_trends(args: argparse.Namespace) -> int:
    from .connectors import ConnectorError, fetch_google_trends

    series = {}
    for kw in args.keyword:
        try:
            series[f"{kw} (Google Trends)"] = fetch_google_trends(kw, timeframe=args.timeframe, geo=args.geo)
        except ConnectorError as exc:   # a catalog message: rendered in --lang
            _err(_t(args, "error", exc=localise(exc, args.lang)))
            return 1
    reports, code = _analyse(series, args)
    return _emit(reports, args, fail_code=code)


def cmd_compare(args: argparse.Namespace) -> int:
    from .samples import sample

    sample_names = [nm.strip() for nm in (args.samples or "").split(",") if nm.strip()]
    problem = _check_samples(sample_names, args.lang)
    if problem:
        _err(problem)
        return 2
    series: dict[str, pd.Series] = {}
    for f in args.files:
        for nm, s in _load(f, args.date_col, args.value_col, args.lang).items():
            key = nm if nm not in series else f"{Path(f).stem}:{nm}"
            series[key] = s
    for nm in sample_names:
        series[nm if nm not in series else f"sample:{nm}"] = sample(nm, freq=args.freq or "M")
    if len(series) < 1:
        _err(_t(args, "no_series"))
        return 2
    reports, code = _analyse(series, args)
    return _emit(reports, args, fail_code=code)


def cmd_ask(args: argparse.Namespace) -> int:
    from .narrative import NarrativeUnavailable, ask
    from .pipeline import run

    series = _load(args.file, args.date_col, args.value_col, args.lang)
    nm, s = next(iter(series.items()))
    if len(series) > 1:
        _err(_t(args, "first_col", name=nm))
    report = run(s, name=nm, freq=args.freq, horizon=args.horizon, lang=args.lang)
    try:
        answer = ask(report, args.question, lang=args.lang, model=args.model)
    except NarrativeUnavailable as exc:
        _err(_t(args, "ask_failed", exc=exc))
        return 1
    except ValueError as exc:
        _err(_t(args, "error", exc=exc))
        return 2
    _say(answer)
    return 0


# --------------------------------------------------------------------------- parser


def build_parser() -> argparse.ArgumentParser:
    """The ``trendai`` argument parser (subcommands analyze, sample, list-samples, wiki, trends, compare, ask)."""
    p = argparse.ArgumentParser(
        prog="trendai",
        description="과거→현재→미래: 시계열 트렌드를 분석하고 예측하는 AI",
    )
    p.add_argument("--version", action="version", version=f"trendai {__version__}")
    sub = p.add_subparsers(dest="command", metavar="COMMAND")

    a = sub.add_parser("analyze", help="CSV/TSV/Excel/JSON 파일의 시계열 분석·예측")
    a.add_argument("file")
    a.add_argument("--date-col", default=None, help="날짜 열 이름 (기본: 자동 감지)")
    a.add_argument("--value-col", action="append", default=None,
                   help="값 열 이름 (여러 번 지정 가능, 기본: 숫자 열 전체)")
    _add_common(a)
    a.set_defaults(func=cmd_analyze)

    s = sub.add_parser("sample", help="내장 합성 샘플로 데모 실행 (all = 전체 비교)")
    s.add_argument("name", nargs="?", default="emerging")
    s.add_argument("--n", type=_int_at_least(8, "what_n"), default=120, help="관측치 수 (8 이상)")
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

    q = sub.add_parser("ask", help="분석 결과를 바탕으로 Claude에게 질문 (ANTHROPIC_API_KEY 필요)")
    q.add_argument("file")
    q.add_argument("question")
    q.add_argument("--date-col", default=None)
    q.add_argument("--value-col", action="append", default=None)
    q.add_argument("--horizon", "-H", type=_int_at_least(1, "what_horizon"), default=None)
    q.add_argument("--freq", choices=FREQ_CHOICES, default=None)
    q.add_argument("--model", default="claude-opus-5-5")
    q.add_argument("--lang", choices=["ko", "en"], default="ko")
    q.set_defaults(func=cmd_ask)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point of ``python -m trendai`` / ``trendai``. Returns the exit code."""
    global _LANG
    _setup_stdout()
    _Out.broken = False
    _SEEN_WARNINGS.clear()
    argv_list = list(sys.argv[1:] if argv is None else argv)
    # --lang decides the language of argparse type errors too, so read it before parsing
    _LANG = "ko"
    for i, a in enumerate(argv_list):
        if a == "--lang" and i + 1 < len(argv_list) and argv_list[i + 1] in ("ko", "en"):
            _LANG = argv_list[i + 1]
        elif a in ("--lang=en", "--lang=ko"):
            _LANG = a.split("=", 1)[1]
    parser = build_parser()
    args = parser.parse_args(argv_list)
    _LANG = getattr(args, "lang", _LANG) or "ko"
    if not getattr(args, "command", None):
        parser.print_help()
        return 0
    try:
        with warnings.catch_warnings():
            _install_warning_filters()
            return int(args.func(args) or 0)
    except DataError as exc:
        _err(_t(args, "data_error", exc=localise(exc, getattr(args, "lang", "ko"))))
        return 2
    except FileNotFoundError as exc:
        _err(_t(args, "not_found", exc=exc))
        return 2
    except BrokenPipeError:
        _Out._silence()
        return 0
    except KeyboardInterrupt:
        _err(_t(args, "interrupted"))
        return 130
    except Exception as exc:  # noqa: BLE001 - one clean line instead of a traceback
        if os.environ.get("TRENDAI_DEBUG"):
            raise
        _err(_t(args, "unexpected", kind=type(exc).__name__, exc=exc))
        return 1
