# trendai — architecture & module contracts

Goal: an AI that understands a trend's **past** (history, structural breaks),
its **present** (lifecycle phase, momentum) and predicts its **future**
(forecast with uncertainty + interpretation), for *any* time series
(search interest, page views, sales, prices, KPIs...). The user-facing text
is Korean first; on request (`lang='en'`, `--lang en`) the narrative, the
terminal summary, the CLI's own messages and the data-layer and connector
messages (`DataError` / `DataWarning` / `DataNotice` / `SeriesFailedWarning` /
`ConnectorError`, built from the catalog in `messages.py`) are English. The
HTML pages stay Korean, and `str(exc)` of those exceptions stays Korean
(`exc.text('en')` gives the English text).

Runtime deps: `numpy`, `pandas` only. Optional: `anthropic` (Claude narrative /
Q&A), `requests` / `pytrends` (online connectors), `openpyxl` / `xlrd` (Excel
files). Python >= 3.10, pandas 3.x compatible (pandas 3 may store datetimes at
`us` resolution: never use `.asi8` or assume ns). Everything is deterministic:
no randomness, no clock in the analysis, forecasting or rendering paths.

This document describes the code as it is. When a statement here and a
docstring disagree, the docstring of the function is the authority.

```
CSV/TSV/TXT/Excel/JSON ─ data.load_table() ─► {name: raw pd.Series}
samples.sample() · connectors.fetch_*() ────► raw pd.Series
        │  data.prepare()  → regular float pd.Series + freq code ('D','W','M','Q','Y')
        ▼
analysis.analyze()          ── past & present ─► AnalysisResult
        │   decomposition (additive or log-space; piecewise after level shifts),
        │   metrics, changepoints, anomalies, lifecycle phase
forecast.auto_forecast()    ── future ─────────► ForecastResult
        │   models.py candidates, rolling-origin backtest, weighted ensemble,
        │   plausibility guard, Student-t prediction intervals, skill vs. naive
forecast.compute_outlook()  ───────────────────► Outlook
narrative.generate_narrative() ────────────────► Korean / English story (rule-based or Claude)
pipeline.run() = all of the above ─────────────► TrendReport
report.render_text() / render_html() ──────────► terminal summary / self-contained HTML page
compare.rank_trends() + report.render_compare_html() ► ranking table / comparison page
cli.py exposes everything as `python -m trendai <command>` (also the `trendai` script)
```

---

## types.py — shared dataclasses

All modules communicate through the dataclasses in `trendai/types.py`. Their
fields are frozen (helper methods/properties may be added; only the lead edits
the file). Conventions: a *series* is a float `pd.Series` with a sorted,
regular `DatetimeIndex`; `freq` is one of `'D' 'W' 'M' 'Q' 'Y'`; percentages are
in percent units (`12.5` = +12.5 %); slopes are per period in the series' units.

| type | contents and semantics |
|---|---|
| `Decomposition` | `trend`, `seasonal`, `resid` (`series == trend + seasonal + resid` exactly, no NaNs, edges included) and `period` (1 = no seasonal component). The seasonal part is a periodic pattern summing to 0 per period only for an **additive** decomposition. A **multiplicative** (log-space) one has `trend = k·exp(t)` (`k` = mean of `exp(s)` over one period, so the trend is the season-average level) and `seasonal = exp(t + s) − trend`, which scales with the level; `seasonal / trend` is periodic and averages 0 per period. Tell them apart with `analysis._is_multiplicative(dec)`. After a kept level shift the trend jumps there (piecewise decomposition). |
| `Changepoint` | `date`, `index`, `slope_before`, `slope_after`, `kind` (one of `CHANGEPOINT_KINDS`: acceleration, deceleration, reversal_up, reversal_down, level_shift) and a unitless `magnitude >= 0`. Every surface names it with `analysis.changepoint_label(cp, lang)` (directed by the slopes, see the analysis section). |
| `Anomaly` | `date`, `index`, `value`, `expected`, signed robust `zscore` (see `detect_anomalies`). For intermittent demand `expected` is the typical (median non-zero) demand size. |
| `TrendMetrics` | descriptive past/present statistics; see the analysis section for the rules behind `total_change_pct`, `cagr_pct`, `recent_change_pct` (`None` when no meaningful percentage exists), `momentum`, `volatility`, strengths, `mk_pvalue`, `peak_value` and `drawdown_pct`. `drawdown_pct` is always a number: check `narrative.drawdown_undefined(analysis)` (`analysis.pct_change_undefined(m.peak_value, m.last_value, analysis.series)`, plus `'at_zero'` for a non-negative series whose latest value is 0) before printing it as a percentage. `slope_pct`, `momentum` and `volatility` are numbers too: check `analysis.level_rate_undefined(series, season_length)` first. |
| `LifecyclePhase` | `label` (one of `PHASES`: emerging, growing, peaking, declining, stable, reviving, volatile), `confidence` in [0, 1], `reasons` (2–4 short Korean sentences), property `label_ko` (`PHASE_KO`). Below `analysis.MIN_PHASE_OBS` (8) observations the label is only a tentative placeholder (`narrative.phase_is_tentative`). |
| `AnalysisResult` | `name`, the prepared `series`, `freq`, `season_length`, `decomposition`, `metrics`, `changepoints`, `anomalies`, `phase`. |
| `ModelScore` | `name`, `mae`, `rmse`, `smape` (0..200 %), `mase` (scaled by the in-sample seasonal-naive MAE), `weight` (final ensemble weight, 0 if not used). |
| `ForecastResult` | `mean`, `lower` / `upper` (`{level: series}`), `horizon`, `levels`, `model_scores`, `method`, `backtest_mase`, `nonnegative`, `skill`; `index` property and `to_frame()` (`mean, lo80, hi80, lo95, hi95`). |
| `Outlook` | `direction`, `expected_change_pct`, `prob_increase`, `scenarios`, `forecast_peak_date`, `future_phase`, `confidence` (semantics below). |
| `TrendReport` | `name`, `analysis`, `forecast`, `outlook`, `narrative`, `narrative_source` (`'rule'` or `'llm'`). |

`ForecastResult` semantics:

* `model_scores` — every candidate that could be backtested, sorted by MASE
  (lowest first; ties keep the candidate order), with **recency-weighted**
  metrics computed on guarded forecasts. With the default candidates of an
  ordinary series this includes the benchmarks `naive`, `drift` and (for
  `m > 1` with two seasons) `seasonal_naive`; with custom candidates or for
  intermittent demand only the candidates are listed. **Empty** when there
  was no backtest (fallback forecast). For intermittent demand the models
  were ranked and weighted by RMSE, so the members (`weight > 0`) are not
  necessarily the first rows.
* `skill` — backtest MAE of the final method / MAE of the best naive benchmark
  (naive, drift, seasonal naive when `m > 1` and the windows hold a season)
  on the same windows (RMSEs for intermittent demand). `< 1` = better than
  naive. Finite, capped at 100; `None` iff there was no backtest.
* `backtest_mase` — recency-weighted backtest MASE of the final method;
  `None` without a backtest. MASE is scaled by the *one-step* naive error, so
  multi-step forecasts of trending series routinely exceed 1 even when they
  beat naive: judge accuracy by `skill`.
* `nonnegative` — mean and bounds were clipped at 0.

`Outlook` semantics (all from `forecast.compute_outlook`):

* `expected_change_pct` compares the **future level** (mean of the forecast's
  last season, i.e. the last `min(m, horizon)` forecast means) with the
  **current level** (mean of the last observed season, the last `min(m, n)`
  observations). A partial-season window is seasonally adjusted only when the
  seasonality matters (see `outlook_levels` below). It is not the change of
  the horizon-end value. `None` when undefined, and
  `forecast.outlook_levels(...).undefined` gives the reason
  (`forecast._pct_undefined` = `analysis.pct_change_undefined(current,
  future, series)` plus `'mostly_zero'`, `'floor'` and an upward
  `'sign_change'`; `'zero'` wins, then `'mostly_zero'`, then `'floor'`):
  * `'zero'`: the current level is about 0 (at most 1 % of the history's
    typical absolute level, the mean `|value|` after clipping outliers);
  * `'mostly_zero'`: at least half of the current window is exactly 0;
  * `'floor'`: at least half of the future window sits at the 0 floor of a
    nonnegative forecast while the current level is positive (means clipped
    at 0 are the floor, not an expected value: no "−100.0 %");
  * `'sign_change'`: the future level has the opposite sign (either
    direction);
  * `'negative'`: a negative current level at least 0.5 SD from 0 (the two
    levels are compared in units: −25.9 → −34.2 is no "−32 %");
  * `'near_zero'`: a series with negative values whose current level is
    within 0.5 SD of 0 (either side) and does not cross 0.

  Intermittent demand whose current window is less than half zeros keeps a
  percentage: the two levels are then average demand rates (every surface
  calls them `평균 수요량` / average demand).
* `prob_increase` — P(future level > current level) for **the same two
  levels** (sign-consistent with `expected_change_pct`); NOT P(horizon-end
  value > today's value). Always a float here; every surface leaves it out
  where `narrative.prob_undefined` gives a reason (`'all_zero'`,
  `'zero_forecast'`, `'zero_current'`): the JSON export nulls it with
  `prob_increase_undefined`.
* `scenarios` — `{'pessimistic', 'base', 'optimistic'}` = lo80 / mean / hi80
  **at the horizon end** (single values that carry that date's seasonal
  effect); the band is `forecast.interval_level(forecast)`, the level closest
  to 80 (the 95 % band for custom levels `(50, 95)`), and every surface names
  it. For a `nonnegative` forecast a value at 0 is the floor of the band and
  is printed as `0(하한)` / `0 (floor)` without a percentage
  (`narrative.scenario_value`); an all-zero history prints a plain `0`.
* `direction` `'up' | 'down' | 'flat'`, `future_phase` (one of `PHASES`),
  `confidence` `'high' | 'medium' | 'low'` (from skill, sample size and the
  relative 80 % band width), `forecast_peak_date` or `None`. Every surface
  that names the direction uses `narrative.direction_label(direction,
  expected_change_pct, lang, levels=outlook_levels(analysis, forecast))`:
  `'flat'` reads `보합` or `방향 불확실` (see the narrative section).

---

## data.py — loading and regularising

```python
class DataError(Localised, ValueError)     # input cannot be used (message naming the flag/argument)
class DataWarning(Localised, UserWarning)  # data was repaired (values dropped, gaps filled, merged...)
class DataNotice(Localised, UserWarning)   # how the input was interpreted (ignored columns, date order...)

def load_table(path, date_col=None, value_cols=None) -> dict[str, pd.Series]       # raw series
def frame_to_series(df, date_col=None, value_cols=None, *, decimal=None,
                    prefer_decimal_comma=False) -> dict[str, pd.Series]
def parse_dates(col, name_hint=True, allow_epoch=True) -> tuple[pd.Series | None, str, str | None]
def parse_numbers(col, decimal=None, prefer_decimal_comma=False) -> tuple[pd.Series, int, list[str]]
def prepare(series, freq=None, name=None, how="mean") -> tuple[pd.Series, str]
def regularize(series, freq=None, how="mean") -> pd.Series
def infer_freq(index) -> str
def future_index(last_date, freq, horizon) -> pd.DatetimeIndex
def from_values(values, start="2020-01-01", freq="M", name="series") -> pd.Series
def season_length(freq) -> int;  periods_per_year(freq) -> float;  default_horizon(freq) -> int
PANDAS_FREQ, FREQ_KO
```

| freq | season length | periods / year | default horizon | grid |
|---|---|---|---|---|
| `D` | 7 | 365.25 | 30 | `D` |
| `W` | 52 | 52.1775 | 26 | `W-<weekday of the data>` |
| `M` | 12 | 12 | 12 | `MS` (month start) |
| `Q` | 4 | 4 | 8 | `QS` |
| `Y` | 1 | 1 | 5 | `YS` |

* `load_table` reads `.csv` `.tsv` `.txt` (separator `,` `;` tab `|` sniffed),
  `.xlsx` (openpyxl), `.xls` (xlrd) and `.json` (records, `{date: value}`,
  the `DataFrame.to_json` orients, JSON Lines). Encodings: UTF-8 (± BOM),
  UTF-16, CP949/EUC-KR. Dates: ISO, `2024.01` / `2024. 1. 5.`, `202401` /
  `20240105`, `2024년 1월`, `2024년 1분기`, `2024 Q1` / `Q1 2024`, ISO weeks,
  `dd.mm.yyyy` / `mm/dd/yyyy` (order decided once per column, ambiguous orders
  reported), month names, plain years, Unix s/ms, timestamps with UTC offsets
  (local wall time kept). Numbers: thousands separators, decimal commas, `%`,
  currency marks, unit words, `1.2억` / `3,400만` / `12K` / `1.5M`, accounting
  negatives, the usual missing markers. Tables: long (date + N value columns),
  `연도` + `월`/`분기` columns, wide (dates across the columns, KOSIS layout),
  headerless files. Row-number columns (`Unnamed: 0`, `id`, `번호`, integer
  runs from 0/1) are skipped unless requested with `value_cols`. Returned
  series are **raw** and carry `series.attrs['trendai'] = {'date_col',
  'date_format', 'n_rows'}`. Every unreadable input raises `DataError`.
* `prepare` accepts a DatetimeIndex (tz-aware ok), PeriodIndex, integer years
  (1000..2200) or date strings as the index and numbers or number strings as
  values; it sorts, aggregates duplicates (`how='mean'|'sum'`), interpolates
  interior gaps and drops leading/trailing missing periods (`regularize`). It
  needs at least 2 observations (`DataError` otherwise). Unreadable and
  infinite values are always reported (`DataWarning`); the regularisation
  reports (weekday-only data, interpolated gaps of >= 20 % and at least 3
  periods, >= 5 % merged duplicates) are emitted only when `prepare` infers
  the frequency itself, so clean regular input emits nothing.
* Frequency inference uses the 25 % quantile of the spacing (gappy data keeps
  its native frequency). `prepare` / `regularize` infer it from every dated
  row, empty values included (`_infer_freq_with_gaps`), so a monthly table
  with many empty cells stays monthly. The observed values keep their own,
  coarser frequency when they fill at least 80 % of its periods
  (`_OWN_GRID_COVERAGE`), e.g. Monday values in a daily calendar with a few
  holiday weeks missing stay weekly. There is no business-day code: weekday-only daily
  data is interpolated over weekends (with a `DataWarning` suggesting
  `--freq W`), and sub-daily data is averaged to days.
* Every message is a catalog message (`messages.msg(code, **params)`, see
  `messages.py`): `str(exc)` is the Korean text, `exc.code` / `exc.params`
  name it and `exc.text('en')` renders it in English. `parse_dates`' note is a
  `messages.Msg` (a `str` subclass). Numbers inside the messages use
  `numfmt` (thousands separators, U+2212 minus; the numeric-index bounds no
  longer use Python's `:g` format).
* Weekly series keep the weekday they were recorded on, and
  `future_index(last, 'W', h)` continues on `last.weekday()`.
  `PANDAS_FREQ['W']` is only the default anchor (`W-SUN`): build weekly dates
  with `future_index`, not with `PANDAS_FREQ`.

## samples.py — synthetic archetypes

`sample(name, n=120, freq="M", start=None, seed=0, noise=1.0)`,
`sample_set(names=None, **kw)`, `list_samples()`, `breakout_date(...)`,
`DESCRIPTIONS` (Korean), `EXPECTED_PHASE`. Nine clearly synthetic archetypes:
emerging, growing, peaking, fad, declining, seasonal, reviving, breakout,
volatile. Values are always positive; without `start` a sample ends at the
last full period on or before 2025-12-31. Unknown names raise `KeyError` and
`n < 8` raises `ValueError` (Korean + English message); an unknown `freq` is a
`DataError`.

---

## analysis.py — past & present

```python
def decompose(series, period, breaks=None) -> Decomposition
def theil_sen_slope(y) -> float
def mann_kendall(y) -> tuple[float, float]          # (tau, two-sided p), classical, tie-corrected
def detect_changepoints(series, period=1, max_changepoints=5, min_segment=None) -> list[Changepoint]
def detect_anomalies(series, decomposition, threshold=3.5, breaks=None) -> list[Anomaly]
def compute_metrics(series, freq, decomposition, anomalies=None) -> TrendMetrics
def lifecycle_phase(series, freq, decomposition, metrics, changepoints) -> LifecyclePhase
def analyze(series, freq=None, name=None) -> AnalysisResult
def pct_change_undefined(base, new, values) -> str | None     # why a % change means nothing
def level_pct_blocked(values, season_length=1) -> str | None   # 'zero' (all 0) / 'intermittent'
def level_rate_undefined(values, season_length=1) -> str | None  # rates in % of the mean level
def changepoint_label(cp, lang="ko") -> str                    # '하락→상승 반전', 'growth slowing'
CHANGEPOINT_LABEL_EN: dict[str, str]                           # every Korean label -> English
def phase_window(n) -> int                                     # the phase reasons' recent window
MIN_PHASE_OBS = 8                                              # fewer: placeholder phase
CAGR_MIN_YEARS = 0.95                                          # shortest span with a CAGR
```

* `pct_change_undefined(base, new, values)` is the one percentage rule of the
  analysis (`total_change_pct`, `recent_change_pct`, `cagr_pct`, the phase
  reasons) and of every surface that prints such a number. It returns `None`
  when `(new − base)/base·100` is meaningful, otherwise the reason:
  * `'zero'`: `|base|` is at most 1 % of mean `|values|`, or a number is not
    finite;
  * `'negative'`: `base < 0`, and for data with negative values at least half
    their standard deviation from 0;
  * `'sign_change'`: `new < 0 < base` (the level crossed zero: a P&L going from
    10 to −10 is no "−200 %");
  * `'near_zero'`: `values` contain negatives and `|base|` is below half their
    standard deviation (an interval-scale series near its zero).

  The reference level and the SD are taken after clipping `|values|` at ten
  times their 90 % quantile (from 10 values on, `_pct_reference`): one viral
  spike of 1,000,000 among levels of ~100 does not make every level "close to
  zero". Non-negative series only ever get `'zero'` from their own numbers. A
  drawdown is checked with it (`pct_change_undefined(m.peak_value,
  m.last_value, analysis.series)`; `narrative.drawdown_undefined` adds
  `'at_zero'`). The outlook uses the same rule (`forecast._pct_undefined` calls
  it and adds `'mostly_zero'`, `'floor'` and an upward `'sign_change'`); there
  is no second percentage rule in the forecast.
* `level_pct_blocked(values, season_length)`: `'zero'` for an all-zero
  history, `'intermittent'` for intermittent demand, else `None` -- the gate
  under which no percentage of the level is stated at all: the growth metrics
  are `None` and the phase reasons give moves in units.
* `level_rate_undefined(values, season_length)`: `level_pct_blocked`, else
  `pct_change_undefined(mean, mean, values)` -- why `slope_pct` ("평균 수준 대비
  …%/월"), `momentum` (%p) and `volatility` (a CV) mean nothing (a mean near 0,
  a negative mean, a P&L crossing zero). The metrics stay numbers; every
  surface asks this first (`narrative.rate_pct_undefined`).
* `changepoint_label(cp, lang)` names a break by its slopes; the phase
  reasons, the narrative, the chart, the tables and the terminal all use it:
  * acceleration: `성장 가속` (slope after >= 0) / `하락 가속` (growth
    accelerating / decline accelerating);
  * deceleration: `성장 둔화` (slope before > 0) / `하락세 둔화` (growth
    slowing / decline easing);
  * reversal_up: `하락→상승 반전` when the slope before is below −10 % of the
    larger absolute slope, else `횡보→상승 전환` (reversal from decline to
    growth / turn from flat to growth); reversal_down likewise (`상승→하락 반전`
    / `횡보→하락 전환`);
  * level_shift: `수준 변화` (level shift);
  * missing slopes: `가속`, `감속`, `상승 반전`, `하락 반전`; an unknown kind:
    `추세 변화` (trend change).

  `CHANGEPOINT_LABEL_EN` maps every Korean label above to its English one.

* `analyze` calls `data.prepare` first (raw input is fine), decomposes,
  finds changepoints on the deseasonalised series (`trend + resid`), and, if
  there are level shifts, re-decomposes with the trend allowed to jump at the
  shifts that the data need as a jump (`decompose(..., breaks=...)`; kept only
  if the residual sum of squares within one trend window of the step drops
  below 70 % of the smooth fit's). A "level shift" that is not kept is
  re-labelled acceleration / deceleration / reversal by its slopes. The
  returned decomposition (piecewise if shifts were kept) is the one used for
  anomalies (`breaks=kept`), metrics (with those anomalies) and the phase.
  Changepoints and anomalies need `n >= 8`; it works from 3 observations.
* `decompose`: STL-like and robust: the seasonal component is a per-phase
  robust average of the detrended series (centred, shrunk towards 0 when
  cycles disagree), the trend a local-linear LOESS of the deseasonalised
  series (window ~n/10, at least min(1.5·period, n/8)), two bisquare
  robustness iterations. Edges are extrapolated with one-sided local-linear
  fits, shrunk towards a wider fit when the end hook is not significant, so
  there are no NaNs. No seasonal component if `period <= 1` or fewer than two
  full periods (`Decomposition.period == 1`).
  * **Multiplicative mode**: a strictly positive series with >= 3 full periods
    whose seasonal swing scales with the level is decomposed in log space
    (`_prefer_multiplicative`). The test regresses the additive residual on
    `seasonal × (trend − mean)/mean` and needs a slope `beta > 0` with
    t >= 3.5 and a trend max/min >= 1.25. With `beta >= 0.35` the log fit's
    remainder must be smaller than the additive one; with `0 < beta < 0.35`
    (typical of weekly data with few cycles) it must be below 0.75 × the
    additive sum of squares (`_MULT_WEAK_SS`). The components are
    `trend = k·exp(t)` with `k` = mean of `exp(s)` over one period (the
    season-average level, not the geometric mean),
    `seasonal = exp(t+s) − trend` and `resid = y − trend − seasonal`.
  * Non-negative series keep a trend >= 0; within half a window of each end
    it is also floored at the lowest deseasonalised value of the edge window.
  * `breaks`: positions where the level may jump; the trend is fitted
    separately on each stretch (each >= 4 points).
* `mann_kendall` is the classical test (independent observations).
  `TrendMetrics.mk_pvalue` uses the **dependence-corrected** test: Var(S)
  inflated for AR(1) noise around the trend (Hamed–Rao), and near a unit
  root (`phi >= 0.9`) the smaller of that and a drift test on the first
  differences — never below the classical p-value. `mk_tau` stays the plain
  Kendall tau; `direction` = sign of tau if the corrected p < 0.05.
* `detect_changepoints`: deseasonalise, replace isolated spikes by a running
  median, then an optimal piecewise-linear segmentation by dynamic programming
  (exact best RSS for every `k <= max_changepoints`), choosing `k` by
  `eff·n·log(RSS_k / dof_k) + 4·k·log(n)` with `dof_k = n − (3k + 2) −
  (period − 1 if deseasonalised)`. `eff = (1 − φ)/(1 + φ)` is the
  effective-sample-size factor of the AR(1) coefficient φ of the most flexible
  fit's residual (from n >= 24; pushed towards 1 near a unit root), so random
  walks and persistent noise get few or no breaks. Series longer than 800
  points are segmented on block means. `magnitude` = |slope change| ×
  following segment length / series scale (level jump / scale for level
  shifts). Default `min_segment = max(4, period // 2, n // 12)`; sorted by date.
* `detect_anomalies`: robust z of the **studentised** residual: each residual
  is divided by `sqrt(F_i)`, its relative variance under pure noise given the
  smoother's leverage (few seasonal cycles or a short trend window absorb
  part of every ordinary point), then `z = (r − median) / robust SD`,
  `|z| >= threshold`. From `n >= 30` the scale is a local rolling MAD
  (window `max(25, 2 × trend window)`) floored at a quarter of the global one
  and **shrunk towards the global scale** in log space by its own sampling
  noise (positive-part James–Stein). The scale never collapses on tied data.
  On an almost constant history (a lone order after zeros) the scale is ~0
  and |z| runs into the thousands; every surface prints |z| > 100 as
  `z > +100` (`numfmt.zscore_text`) and the payload caps it at ±100.
  Multiplicative decompositions use `log(value / expected)`; intermittent
  demand (>= 30 % zeros overall and recently) is judged by the size of each
  demand (`expected` = median non-zero demand). `breaks` are the level shifts
  of a piecewise decomposition. The honest z-scores mean far fewer false
  flags on white noise, but a single 4–5σ spike in a very short or few-cycle
  series (e.g. Q40, M36, Y30) is also flagged less often than without the
  studentisation; spikes of ~8σ and news-style ×6–×20 spikes are still found.
* `compute_metrics` (field docs in `types.TrendMetrics`):
  * growth numbers use the trend: `total_change_pct` first → last,
    `recent_change_pct` over the last `season_length` periods (1 for yearly),
    `cagr_pct` needs a span >= 0.95 years. They are `None` when
    `pct_change_undefined` gives a reason (base <= 0 or below 1 % of the mean
    absolute level, a trend that crossed zero, a base below 0.5 SD for data
    with negative values) and under `level_pct_blocked` (all zeros,
    intermittent demand). Non-negative series are unaffected by the
    zero-crossing and SD rules;
  * `slope_pct`, `acceleration` and `volatility` are relative to the mean
    level of the outlier-clipped values (`_mean_level`, the clipping of
    `_pct_reference`);
  * `momentum` = recent OLS trend slope (last `max(3, min(season_length,
    n//4))` periods) minus the Theil–Sen `slope`, in % of the *recent* trend
    level per period; `acceleration` = change of the recent slope between
    the previous and the latest window, in % of the mean level;
  * `volatility` and the strengths use the residual winsorised at 3.5 robust
    SDs (log components for multiplicative decompositions);
    `seasonal_strength` is additionally scaled by the evidence of a
    significance test of the repeating pattern (0 below the 95 % level); for
    intermittent demand the timing of the non-zero periods must recur too
    (`_intermittent_season_evidence`: demand in at least 3 cycles, and the 0/1
    occurrence series passes the same test), so two orders in two Januaries
    are no "very strong seasonality";
  * `peak_value` / `peak_date` / `drawdown_pct` skip upward spikes flagged by
    `detect_anomalies` — only from 16 observations on and never at the first
    or last observation; `anomalies` passes the already detected ones
    (`analyze` does, so peak/drawdown exclude exactly the reported spikes).
    `drawdown_pct` is always a number, even for a series that crossed zero
    (peak 10, last −8 reads −180 %): the narrative, the report and the
    payload test it with `narrative.drawdown_undefined` and then give the gap
    in units (or, for a count at 0, the peak) instead.
* `lifecycle_phase`: explainable rules over the trend's shape (position and
  size of the peak, drawdown, rebound from the trough, recent slope vs. the
  best past slope, growth multiple over the base) plus significance
  (t-statistic of the recent slope, dependence-corrected Mann–Kendall).
  "Recent" = the last ~15 % of the observations. It can return `'stable'`
  after a level shift followed by a flat stretch, and confidence drops for
  random-walk-like series. With fewer than `MIN_PHASE_OBS` (8) observations
  there is no lifecycle verdict: the label (`'stable'` or `'volatile'`) only
  says whether noise dominates, and the first reason reads `관측치가 N개뿐이라
  국면을 판단하기에 데이터가 부족합니다.` The growth multiple behind
  `'emerging'` needs a base of at least the percentage floor (0.5 SD for data
  with negative values). The reasons' percentages follow
  `pct_change_undefined` with the metrics' reference; under
  `level_pct_blocked` (all zeros, intermittent demand) no reason states a %
  of the level: moves are two levels in units (`최근 23주 동안 추세가 0.33에서
  0까지 낮아졌습니다.`, `추세는 고점(…, 2.44) 이후 낮아져 현재 0입니다.`), and the
  stable-range reason becomes `관측값이 모두 0입니다.` or the noise-range one.
  For intermittent demand a moving phase must agree with the recent move of
  the demand rate (`_against_recent_move`): 'declining' while the rate rose
  (a just-launched product) becomes 'growing' when the whole-period
  Mann–Kendall test agrees with that move, else 'volatile' / 'stable'.
  "Moved" is one rule (`_rate_move`: more than 5 % of the level, the larger
  end), shared with the reason's verb — `… 높아졌습니다` / `… 낮아졌습니다`, or
  `최근 N개월 동안 추세는 A에서 B까지 거의 변하지 않았습니다.` — so no phase sits next
  to a reason that says the opposite (5 % of the trend's whole *range* let a
  +127 % rise pass as no move next to 'declining'). 'emerging' / 'growing' /
  'declining' of intermittent demand must also be backed by the demand rate
  (`_single_order_move`): the average demand of the recent window (the last
  `max(h, min(m, n // 3))` periods) against the earlier history, each with its
  single largest demand counted at the median non-zero size, must move by at
  least 15 % in the phase's direction and either Mann–Kendall agrees or the
  rate changed 2.5×; otherwise 'volatile' / 'stable' (strength 0.2) with the
  reason `최근의 높은 추세는 2025-12의 큰 수요 한 건(60.0) 때문이며, 이를 빼면 수요가
  늘었다고 보기 어렵습니다.` (or the `과거의 … 줄었다고` mirror, or `기간당 평균
  수요량(이전 24개월 A, 최근 12개월 B)만으로는 수요가 늘었다고 보기 어렵습니다.`).
  `phase_window(n)` (public) is the reasons' recent window `h`.
* Phase reasons are Korean with numbers (U+2212 minus, `p < 0.001`, numbers
  formatted like `narrative.format_number`, ×10ⁿ from 1e12). Dates before a
  particle use `2016년에` for yearly data and bracketed dates stay bare
  (`고점(2016)`). Some templates:
  * at >= 99.5 % of the trend peak: `현재 추세는 고점(YYYY-MM) 수준입니다.`;
  * a trend that crossed zero: `추세는 고점(YYYY-MM, X) 이후 0 아래로 내려가
    현재 Y입니다.`;
  * no recent percentage: `최근 18개월 동안 추세 기울기는 −0.69/월입니다.`
    (per-period unit);
  * a pace ratio above 20 (`_PACE_CAP`): `최근 상승 속도가 전체 기간 평균
    속도(추세선 처음→끝)의 20배를 넘습니다.`;
  * a recent break: `…에 구조 변화(하락→상승 반전)가 감지되었습니다.`
    (`changepoint_label`);
  * levels in units (all zeros / intermittent): `최근 N주 동안 추세가 A에서
    B까지 높아졌습니다/낮아졌습니다.`, `… 추세는 A 수준에 머물렀습니다.`,
    `추세는 저점 A에서 B까지 올라 …` / `… 오른 뒤 …에 고점을 형성했습니다.`,
    `관측값이 모두 0입니다.`;
  * after a level shift: `그 이후 N개월 동안은 …` counts the periods after the
    break.

  `narrative.translate_reason` must know every template.
* Tests require the archetypes of `samples.EXPECTED_PHASE` to be classified
  correctly for >= 4 of 5 seeds (monthly n=120) and >= 3 of 5 (weekly n=156,
  daily n=400).

## models.py — forecasters

```python
class Forecaster:          # fit(y, season_length=1) -> self;  predict(h) -> np.ndarray (h,)
    name: str; min_obs: int; needs_season: bool; requires_positive: bool
    offered_for: str = "regular"     # 'regular' | 'intermittent' | 'explicit'
MODEL_REGISTRY: dict[str, Callable[[], Forecaster]]
def register(name, factory) -> None
def make_model(name) -> Forecaster
def available_models() -> list[str]
def default_candidates(n_obs, season_length, positive, intermittent=False) -> list[str]
```

`fit` copies `y`, rejects non-1-D / non-finite input and `len(y) < min_obs`;
`predict` raises if the model produced a non-finite forecast (the backtester
skips that model) — models never mask a failure as "naive". Models never raise
for valid input of at least `min_obs` points (they degrade, e.g. drop
seasonality), are deterministic and fast (one fit + predict typically
1–35 ms for a few hundred points).

| name | model | notes |
|---|---|---|
| `naive` | repeat the last value | benchmark |
| `seasonal_naive` | repeat the last season | benchmark, `needs_season` |
| `drift` | random walk with drift (first → last line) | benchmark |
| `ses` | simple exponential smoothing (optimised α) | * |
| `holt` | Holt linear trend | * |
| `holt_damped` | damped-trend Holt (φ in [0.8, 0.98]) | * |
| `holt_winters` | additive Holt-Winters, damped trend | `needs_season` |
| `theta` | Theta method (SES + ½ OLS drift) | * (multiplicative or additive classical decomposition) |
| `linear_trend` | OLS on time + Fourier terms over the recent part (last `max(n//2, 2m, min(n,12))` points) | |
| `ar` | AR(p) on first differences, p by AIC, forced stationary | |
| `lag_ridge` | ML: ridge on lagged differences (+ seasonal lag), GCV penalty, recursive | |
| `analog_knn` | "past analogues": k nearest z-normalised windows, "no good analogue" guard | |
| `logistic_growth` | S-curve fit; only curves whose transition is observed (`min_active=4` points in its 10–90 % part, `min_observed=0.3` of it inside the data); saturation <= 3·max(y) | `requires_positive` |
| `holt_damped@log`, `holt_winters@log`, `linear_trend@log` | the same models in log space, **growth-guarded** (a j-step log move is at most the largest recent j-step move, extended at that pace, and never more than the largest move over the same span in the whole history) | `requires_positive` |
| `drift@log` | drift in log space (constant growth rate) | `requires_positive`, `offered_for='explicit'`: `default_candidates` never returns it; `auto_forecast` adds it for compounding growth (see forecast) or on request |
| `sba` | Croston's method with the Syntetos–Boylan correction (α = 0.1; `params_` holds demands, size, interval, bias_factor) | `offered_for='intermittent'` |
| `tsb` | Teunter–Syntetos–Babai: SES of the demand sizes (on demand, α = 0.1) × SES of the 0/1 demand occurrence updated every period (β = 0.1), so a run of zeros lowers the rate (a product without a sale for 41 weeks: 0.005 instead of SBA's last-year rate); `params_` holds demands, size, probability | `offered_for='intermittent'` |

`*`: when a seasonality test with period `season_length` is positive, these
non-seasonal methods run on the seasonally adjusted series (classical
decomposition, M4 style) and the forecast is reseasonalised.

`default_candidates` returns the registered models whose `min_obs` is met;
seasonal ones only with two full seasons; `requires_positive` / `@log` ones
only when `positive`; `offered_for='intermittent'` ones only with
`intermittent=True`; `'explicit'` ones never. Explicit models can still be
requested through `auto_forecast(candidates=...)` / `pipeline.run(candidates=...)`.
`LogTransformed` shifts non-positive input by a floor relative to `max|y|`
(1.0 for all-zero input) and forecasts a constant series exactly.

## forecast.py — future

```python
def mase_scale(y, season_length) -> float     # in-sample seasonal-naive MAE (naive fallback), > 0
def backtest(y, model_name, season_length, horizon, n_windows=3) -> dict
    # {'mae','rmse','smape','mase','forecasts': (W,h),'actuals': (W,h),'origins': [...]}
def auto_forecast(series, horizon=None, freq=None, levels=(80, 95), candidates=None,
                  top_k=3, n_windows=None, nonnegative=None, random_walk=False) -> ForecastResult
def is_random_walk(analysis) -> bool
def compute_outlook(analysis, forecast) -> Outlook
def outlook_levels(analysis, forecast) -> OutlookLevels
class OutlookLevels(NamedTuple): current, future, k_cur, k_fut, period, adj_cur, adj_fut, undefined, scale
def interval_level(forecast, target=80) -> int | None
def fit_history(y, season_length) -> ndarray   # the last max(400, 8·m): the forecast's history
def norm_cdf(x); norm_ppf(p); t_cdf(x, df); t_ppf(p, df)
```

`auto_forecast` (raw input goes through `data.prepare`; `horizon=None` →
`data.default_horizon(freq)`; `horizon`, `top_k`, `n_windows` must be >= 1):

1. **History cap** — the backtest *and* the final fit use the same last
   `max(400, 8·m)` observations (`fit_history`, the one definition: when it is
   all zeros the forecast and its bands are exactly 0, which
   `narrative.forecast_is_zero` reads).
2. **Spike cleaning** — isolated one/two-point outliers (robust `|z| >= 5`
   around a running median + per-phase medians, in log space for positive
   series; never the last point; needs >= 12 points and, if seasonal, three
   seasons; skipped if more than 15 % of the points would be replaced) are
   replaced by their expected value before anything is fitted or scored.
3. **Candidates** (unless `candidates` is given, which overrides all of this):
   * ordinary series: `models.default_candidates(...)`, plus `drift@log` when
     the history shows compounding growth or decay (strictly positive, >= 12
     points, the level moved >= 50 % between the first and the last quarter,
     and the period-to-period changes are steadier in logs than in levels:
     `_multiplicative_growth`);
   * `random_walk=True` — which `pipeline.run` sets from
     `is_random_walk(analysis)`: no significant trend (`metrics.direction ==
     'flat'`) and AR(1) noise persistence `phi >= 0.9` — drops the trend
     extrapolators (`linear_trend`, `linear_trend@log`, `logistic_growth`,
     `holt`, `holt_winters`, `holt_winters@log`) and adds no `drift@log`;
   * **intermittent demand** (non-negative, >= 30 % exact zeros overall and
     in the last `max(m, 12)` points): no spike cleaning, and only the level
     models `ses`, `theta`, `holt_damped`, `sba`, `tsb` (those registered). They are
     ranked, weighted and chosen single-vs-ensemble by backtest **RMSE**
     (`_INTERMITTENT_METRIC`) and the skill is an RMSE ratio; ordinary series
     use MASE (skill: MAE ratio).
4. **Rolling-origin backtest** of every candidate and of the naive benchmarks
   on the same windows (default 3 / 5 / 8 windows for fewer than 40 / 100 /
   more points; windows step back by `max(1, h // 2)`; series with two seasons
   keep the seasonal layout with a shorter backtest horizon if needed). Window
   scores are weighted towards recent windows (newest 3× the oldest); MASE
   uses each window's training scale, floored at 10 % of the scale of the
   history before the latest origin. Failing or non-finite models are skipped
   with a `RuntimeWarning`.
5. **Plausibility guard** on every forecast (backtest windows and final fit):
   `k` steps ahead a forecast may move at most `max(1.5 × the largest k-step
   move in its history, recent pace × k)` from the last observation (drops
   unlimited for non-negative series). Steady trends and seasonal swings pass;
   compounding a short burst into an explosion does not.
6. **Selection** — rank by the metric; inverse-metric weighted ensemble of the
   top-k, kept unless the best single model scores below 0.9 × the ensemble
   with at least 6 backtest errors; members that were all exact in the
   backtest (a run of zeros forecast as zeros) get equal weights. Refit the
   chosen model(s) on the capped history (if all fail, the best-ranked model
   that works), guard, combine.
   `method` reads e.g. `ensemble(holt_damped 0.45, theta 0.35, ar 0.20)`.
7. **Intervals** `mean ± q·σ_k`: `σ_k = s₁·k^b` fitted to the final method's
   backtest errors (`b` in [0.1, 1], at least `√k` growth beyond the backtest
   horizon, `s₁` floored by an in-sample noise level), errors first rescaled to
   the current level when they grow with it (positive series); inflated for
   best-of-N selection by `sqrt(1 + log(#candidates)/targets)`; pooled with a
   random-walk prior scale with weight `ν/(ν+2)` (`ν` = distinct backtest
   targets − 1); `q` = Student-t quantile on `f·ν + 2` dof with
   `f = min(0.5, (1−ρ)/(1+ρ))`, `ρ` the lag-1 autocorrelation of the errors;
   floored so the 95 % band spans every ensemble member (member-disagreement
   floor); made non-decreasing in `k`. With `nonnegative` (default: all
   observations >= 0) bands are centred on the clipped mean, then clipped.
   An all-zero history (the capped last `max(400, 8·m)` points) has no scale
   and gets zero-width bands; any other constant keeps a tiny, level-relative
   band.
8. **Skill** — see `ForecastResult.skill`; errors below 1e-9 of the mean level
   count as exact (both exact: 1.0).
9. **Fallback** — fewer than ~8 points (no 2-window backtest) or every
   candidate failing: equal-weight drift + naive only when the series has a
   significant linear trend (`_significant_trend`: OLS slope, two-sided 5 %
   t-test on n − 2 dof; an exact line counts), otherwise naive alone with
   method `fallback(naive) [<reason>; no significant trend]`; naive alone for
   n < 3. Wide heuristic intervals, `model_scores == []`, `skill = None`,
   `backtest_mase = None`.

The public `backtest()` weights windows equally and scores raw forecasts
(no guard); it raises `ValueError` if the series is too short for one window.

`compute_outlook` / `outlook_levels`:

* **Levels** (shared by `expected_change_pct`, `prob_increase`, `direction`):
  current = mean of the last `k_cur = min(m, n)` observations; future = mean
  of the last `k_fut = min(m, horizon)` forecast means. A window that is not a
  whole number of seasons (e.g. weekly 26 < 52) has the analysis' average
  seasonal effect over it subtracted — for the forecast window scaled by the
  share of the seasonal pattern the forecast contains; for a multiplicative
  decomposition the continued seasonal component keeps the seasonal *ratio*
  applied to the forecast level (`_future_seasonal(analysis, h, forecast_mean)`).
  The adjustment (and `adj_cur` / `adj_fut = True`) needs all of:
  * a partial-season window;
  * `metrics.seasonal_strength >= 0.2` or unknown (`_ADJ_MIN_STRENGTH`,
    `_seasonality_matters`);
  * a removed effect of at least 0.5 % of the window's mean `|level|`
    (`_ADJ_MIN_SHARE`, `_material`).

  Otherwise, and with whole-season windows (default horizons for D/M/Q/Y),
  the levels are plain means.
* `outlook_levels(analysis, forecast) -> OutlookLevels(current, future, k_cur,
  k_fut, period, adj_cur, adj_fut, undefined, scale)` returns exactly these
  levels (`period` is `None` without a usable seasonal component), `undefined`
  in `{None, 'zero', 'mostly_zero', 'floor', 'sign_change', 'negative', 'near_zero'}`
  (`_pct_undefined` = `analysis.pct_change_undefined` + `'mostly_zero'` +
  `'floor'` (at least half of the future window at the 0 floor of a
  nonnegative forecast, current level positive; `_FLOOR_SHARE`) + an upward
  `'sign_change'` for `current < 0 < future`; a negative current level gets no
  percentage, and `'zero'` means at most 1 % of the mean `|value|`)
  and `scale`, the history's typical absolute level
  (the % rule's reference), against which a level move without a % is judged.
  It is the single source for `compute_outlook`, the narrative, the payload,
  the report, the CLI ranking notes and the JSON export.
* `interval_level(forecast, target=80)`: the band behind the scenarios and
  `prob_increase` (the available level closest to 80, ties to the lower one;
  `_interval_level` is an alias); the narrative and the payload
  (`scenario_interval_level`) name it, and the narrative does not repeat the
  95 % band when it is that band.
* `expected_change_pct = (future − current)/|current| × 100`, `None` when
  `undefined` is set. For intermittent demand the levels are average demand
  rates (zeros included), and the percentage is given unless at least half of
  the current window is zero (`'mostly_zero'`); the analysis' trend-based
  percentages are `None` for all intermittent demand.
* `prob_increase = Φ((future − current)/σ_H)` with σ_H from the 80 % band at
  the horizon end (wider side if a bound was clipped). σ_H is the uncertainty
  of one end value, so the probability is conservative. It may be exactly 0
  or 1; displays use `narrative.format_prob` (`>99%` / `<1%`).
* `direction` is `'flat'` unless `|future − current|` exceeds
  `max(2 % of the current level, half the 80 % half-width at h=1, 0.4307·σ_H)`;
  the last term means `prob_increase` must lie outside [1/3, 2/3].
* `scenarios` = lo80 / mean / hi80 at the horizon end.
* `forecast_peak_date`: forecast maximum strictly inside the horizon, reached
  by a rise above the noise threshold and followed by a >= 5 % fall, on the
  forecast and on its seasonally adjusted, smoothed version.
* `confidence`: `'low'` without a backtest, for `n < 12`, `skill > 1.05` or an
  80 % half-width at the horizon end above the level itself; `'high'` for
  `skill <= 0.8` with `n >= 3` seasons (36 points without seasonality) and a
  half-width of at most 25 % of the level; `'medium'` otherwise (hand-made
  forecasts without `skill` use the old MASE rule: > 1.5 low, < 0.8 high).
* `future_phase`: `'volatile'` stays; up/down lead to the phase the move
  implies (e.g. growing → declining only for a strong fall, else peaking);
  a `'flat'` outlook projects the end of the current move only if the base
  case moves by <= 2 %, confidence is not low and the band is not too wide;
  otherwise the current phase is kept.

## narrative.py — the "AI interpreter"

```python
DEFAULT_MODEL = "claude-opus-5-5";  SYSTEM_PROMPT;  SECTION_HEADINGS;  MODEL_KO;  MODEL_EN
class NarrativeUnavailable(RuntimeError)
def rule_based_narrative(analysis, forecast, outlook, lang="ko") -> str
def build_llm_payload(analysis, forecast, outlook) -> dict
def llm_narrative(analysis, forecast, outlook, lang="ko", model=DEFAULT_MODEL, client=None) -> str
def generate_narrative(analysis, forecast, outlook, use_llm=False, lang="ko",
                       model=DEFAULT_MODEL, client=None) -> tuple[str, str]   # (text, 'llm'|'rule')
def ask(report, question, lang="ko", model=DEFAULT_MODEL, client=None) -> str
def format_number(x, decimals=None, lang="ko", signed=False, scale=None) -> str
def format_pct(x, lang="ko", decimals=1, signed=True) -> str
def format_prob(p, lang="ko") -> str
def format_date(ts, freq, lang="ko") -> str        # D/W 2024-03-05, M 2024-03, Q 2024-Q1, Y 2024
def period_unit(freq, lang="ko") -> str            # /월 /주 /일 /분기 /년 (/mo ... in English)
def translate_reason(reason) -> str | None
def direction_label(direction, expected_change_pct=None, lang="ko", undefined=None, levels=None) -> str
def model_label(name, lang="ko") -> str;  describe_method(forecast, lang="ko") -> str
def scenario_value(value, reference=None, nonnegative=False, lang="ko") -> tuple[str, float | None]
def format_scenario(value, reference=None, nonnegative=False, lang="ko") -> str   # '12.3 (+4.5%)'
def is_intermittent(analysis) -> bool;  phase_is_tentative(analysis) -> bool
def is_all_zero(analysis) -> bool;  scenario_floor(analysis, forecast) -> bool
def floor_note(analysis, forecast, outlook, lang="ko") -> str
def future_phase_status(analysis, outlook) -> str | None   # 'undecided' | 'tentative' | 'low_confidence'
def drawdown_undefined(analysis) -> str | None;  seasonal_trend_drawdown(analysis) -> float | None
def rate_pct_undefined(analysis) -> str | None;  volatility_undefined(analysis) -> str | None
def rate_reason_clause(analysis, code, lang="ko") -> str
def zero_level_reason(analysis, current, lang="ko") -> ZeroLevelReason   # .short / .clause / .sentence
def prob_undefined(analysis) -> str | None         # 'all_zero' | 'zero_forecast' | 'zero_current'
def prob_reason(analysis, lang="ko") -> ZeroLevelReason | None   # why, in short / clause / sentence form
def forecast_is_zero(analysis) -> bool             # forecast.fit_history is all zeros: forecast exactly 0
def trailing_zero_run(analysis) -> int;  long_zero_run(analysis) -> (run, last, gap) | None
def floor_change_reason(analysis, forecast, lang="ko") -> ZeroLevelReason | None   # 'floor' wording
def scenario_reference(analysis, forecast) -> ScenarioRef | None   # (value, date, kind, span)
def demand_rates(analysis) -> (w, first, last) | None   # intermittent: first / last year (season) means
changepoint_label                                   # re-exported from analysis
zscore_text                                         # re-exported from numfmt
PHASE_EN                                            # English phase names
```

* Imports: `analysis` at module level (`CHANGEPOINT_LABEL_EN`,
  `MIN_PHASE_OBS`, `changepoint_label`, `pct_change_undefined`,
  `level_rate_undefined`, `phase_window`, `_is_intermittent`, `_mean_level`),
  `data.periods_per_year` and `numfmt` (`number_text`, `zscore_text`,
  `Z_SHOWN_MAX`); `forecast.outlook_levels`, `interval_level`, `is_random_walk`
  and `fit_history` lazily. `analysis` imports only `data`, `numfmt` and
  `types`, so there is no cycle.
* **Format**: Markdown-lite — `## ` headings, `- ` bullets, sub-items of a
  line ending in `:` nested with a two-space indent (`  - `), `**bold**`.
  A bottom line (`**한눈에 보기** — …`) is followed by four sections:
  과거 / 현재 / 미래 전망 / 주의할 점 (Past / Present / Outlook / Caveats).
  Numbers use thousands separators and the U+2212 minus; Korean prose writes
  yearly dates as `2024년` while `format_date` and the payload keep `2024`.
* `format_number` is `numfmt.number_text`: magnitudes below 1e-4 or from 1e12 up use `1.05×10⁻⁹` /
  `1.87×10¹³` notation (no 13-digit integers, no `e+06`). `scale` is the
  largest magnitude printed next to the number: from 1e12 up, numbers of
  1,000 or more next to it switch to ×10ⁿ as well (`4.61×10¹¹ → 1.51×10¹⁵`).
* `format_prob`: whole percent, `>99%` for p >= 0.995 and `<1%` for p <= 0.005
  (exact 1 and 0 included), `해당 없음` / `n/a` for None/NaN. The report, the
  terminal summary and the CLI ranking print the same text.
* `direction_label`: `'flat'` reads `보합` when `|expected_change_pct| < 2 %`
  and `방향 불확실` / `direction uncertain` otherwise. With no percentage the
  levels (`levels=outlook_levels(...)`, which also supply `undefined`) decide
  the same way: a move relative to `|current|` for `'negative'`, `'floor'` and
  a non-zero `'mostly_zero'` level, relative to `levels.scale` for `'zero'` /
  `'near_zero'` / a mostly-zero window averaging 0 (mostly-zero demand falling
  from 0.35 to 0.07 is no "보합"); `'sign_change'` is always `방향 불확실`.
  Without `levels` a missing percentage reads `보합` (`방향 불확실` for
  `undefined='sign_change'`). The pipeline, the report and the payload call it
  with `levels=`.
* Rates in % of the level: `rate_pct_undefined(analysis)` =
  `analysis.level_rate_undefined`. Where it is set the narrative drops the
  slope's `(평균 수준 대비 …%/월)`, gives momentum in words only (`모멘텀(…)은
  평균 수준이 음수여서 %p로 나타내지 않았으며, …`), states the volatility as the
  residual SD in units (except for intermittent demand, whose CV stays:
  `volatility_undefined`), and the payload nulls `slope_pct_per_period`,
  `momentum` and `volatility_cv` with `metrics.pct_undefined` naming the
  reason. `rate_reason_clause` is the wording the report uses too.
* No probability (`prob_undefined`, one rule for the narrative, the payload,
  the JSON export, the terminal, the HTML KPI / outlook card and the ranking):
  `'all_zero'` (an all-zero history), `'zero_forecast'` (the forecast's own
  history is all zeros: `forecast_is_zero`, the rule `scenario_floor` and the
  zero-width caveat use too — `최근 450일 동안 값이 모두 0이라 예측도 0(구간 폭
  0)이고, …`) and `'zero_current'` (the current window is exactly 0 for data
  that never go below 0: a fall is impossible, so no "rise and fall about
  equally likely" — `최근 39개월 동안 값이 모두 0이고 0 아래로는 내려갈 수 없어,
  상승·하락 확률은 따지지 않았습니다.`); `prob_reason` words it once (the terminal:
  `상승 확률 — (최근 39개월 연속 0)`). The payload carries
  `outlook.prob_increase_undefined`.
* `'floor'` outlook (`floor_change_reason`): `예측이 하한인 0까지 내려가
  변화율(%)은 의미가 없습니다(최근 12개월 평균 7.74 → 예측 마지막 12개월 평균 0).`
  or `예측 마지막 12개월 중 10개월이 하한인 0에 닿아 …`; `direction_label` judges the
  fall relative to `|current|` (a fall to 0 is no `보합`).
* Scenario reference (`scenario_reference`, the one rule of the narrative's
  `괄호는 … 대비` percentages, the payload's `outlook.scenario_reference`, the
  terminal's `(비교: …)` and the HTML scenario bar): the same point of the
  seasonal cycle for `seasonal_strength >= 0.3` (`'season'`), else the latest
  observation (`'latest'`); intermittent demand compares the forecast rate with
  the average demand of the last season (`'demand'`). A scenario's % follows
  `pct_change_undefined(reference, value, series)`.
* Intermittent past: `0인 기간까지 포함한 기간당 평균 수요량은 처음 12개월 0.67 → 최근
  12개월 1.17입니다.` (`demand_rates`: the first / last year, else season; the
  trend's ends only for a series shorter than two seasons). The Mann–Kendall
  line judges "against the ends" on these means, and a Theil–Sen slope of
  exactly 0 beside a significant test is explained (`관측치의 절반 이상이 0이라
  0/주로 나와 …`). The present section states an unusually long trailing run
  of zeros (`long_zero_run`: at least 3 and 3× the usual gap): `최근 41주 연속
  수요가 없었습니다(마지막 수요: 2025-03-16의 1.00). 그전에는 평균 3.3주마다 수요가
  있었습니다.`
* The recent-pace phrase never contradicts the phase reason over the phase's
  own window (`analysis.phase_window`): `최근 39주 전체로는 하락세가 멈췄지만, 최근
  23주 동안은 다시 낮아졌습니다` instead of a bare `최근에는 하락세가 멈췄습니다`.
* All-zero history (`is_all_zero`): the past reads `관측 기간 내내 0이었습니다.`,
  the scenarios are a plain `0` (no `0(하한)`: nothing was clipped), no
  probability (`prob_undefined`; the payload's `prob_increase` is null), no
  95 % band of zero width, no "uncertainty grows" caveat (instead `관측값이 모두
  0이라 예측도 0(구간 폭 0)입니다.`), momentum and volatility "not computed",
  every reason reads `관측값이 모두 0`. `floor_note` says the base case "stays
  at" the floor of 0 when the current level is already 0. The `'zero'`
  outlook reason has one wording on every surface (`zero_level_reason`: all
  zeros `관측값이 모두 0이라`, exactly 0 `현재 수준이 0이라`, near 0; short,
  clause and sentence forms).
* Anomaly lines print the % deviation only where `pct_change_undefined(expected,
  value, series)` allows it, else the gap in units (`차이 +31.7`); z-scores
  via `zscore_text` (`z > +100` beyond 100). A count whose latest value is 0
  has `drawdown_undefined == 'at_zero'`: the peak is named, never −100 %.
* `scenario_value` / `format_scenario`: a scenario or interval bound at or
  below 0 of a `nonnegative` forecast is `0(하한)` / `0 (floor)` without a
  percentage; otherwise `format_number(value)` and the percentage
  `(value − reference)/|reference|·100` (none without a reference or for a
  reference of 0); `해당 없음` / `n/a` for None/NaN.
* `is_intermittent(analysis)`: `analysis._is_intermittent` on the series (the
  rule of the analysis and the forecast). `phase_is_tentative(analysis)`:
  fewer than `analysis.MIN_PHASE_OBS` observations.
* `translate_reason`: English for one Korean phase reason (input returned
  unchanged if it has no Hangul, `None` if no template matches; untranslated
  reasons are dropped from English text). It knows the old and the new
  templates.
* The outlook levels come from `forecast.outlook_levels` (`_outlook_windows`
  wraps it), so the narrative and the payload print exactly the levels behind
  `expected_change_pct` and `prob_increase`.
* **Outlook section order**: expected change (with the two compared windows;
  partial-season adjustment named) → probability (same two levels named) →
  horizon-end scenarios (80 % band, flagged as a different quantity that
  carries the end date's seasonal effect) → forecast peak → projected phase →
  confidence → `정확도:` / `Accuracy:` bullet from `ForecastResult.skill`
  (MASE only as a technical note; "did not beat naive" only when skill > 1) →
  method in plain words. With `expected_change_pct = None` no percentage is
  stated (scenario percentages omitted too); the first line gives the reason
  in words (a level expected to cross zero, a negative level compared in
  units, a level near 0 relative to the spread, mostly zeros, a forecast at
  the 0 floor, a level of 0 or about 0, an all-zero history) and names both
  levels. A negative current level: `향후 12개월(2021-12까지) 전망은
  **하락**입니다. 현재 수준이 음수(최근 12개월 평균 −27.8 → 향후 12개월 예측 평균
  −30.4)여서 변화율(%) 대신 수준으로 비교합니다(차이 −2.55).`
  * Scenarios at the 0 floor read `0(하한)` with one floor note; the caveats
    section then drops its own floor line. The 95 % band is added only when
    it is not the scenarios' own band (`interval_level`).
  * Projected phase: a moving phase (emerging, growing, peaking, declining,
    reviving) that a `'flat'` outlook only kept is "hard to judge"; a phase
    change from a low-confidence forecast is only what the forecast would
    imply; a tentative phase is called tentative.
  * "The models do not extend the recent rise/decline" (`_phase_outlook_gap`)
    only when the recent trend slope has the phase's sign and the forecast
    moves less than half of what that pace gives between the two compared
    windows (and less than 2 % in the phase's direction).
  * Intermittent levels are named average demand with their values
    (`향후 12개월 평균 수요량(기간당 1.65)은 최근 12개월 평균 수요량(2.00)보다
    **−17.5%** 변할 것으로 예상됩니다 (…). 0인 기간까지 포함한 구간 평균끼리 비교한
    값이라, 계산하지 않은 추세선 변화율과는 다른 수치입니다.`, or without a
    percentage `향후 12개월 평균 수요량은 1.16, 최근 12개월 평균 수요량은
    1.25입니다.`); the note says which side is a window average and which a
    single value.
* Past/present details:
  * peak "(일회성 급등 제외)" when a flagged spike was left out;
  * no percentage where `pct_change_undefined` refuses one: the reason gets its
    own wording and a zero-crossing change or drawdown is given in units
    (`최근 값(−9.40)은 고점보다 21.0 낮습니다`); `_trend_drawdown` is `None` for a
    trend that crossed zero;
  * changepoint lines use `changepoint_label`;
  * a tentative phase (too few observations) is presented as such in the
    summary, present, momentum, projection and caveats lines;
  * a Mann–Kendall direction that contradicts the trend's endpoints is
    qualified, and the momentum phrase follows it; random-walk-like series get
    their own trend-strength wording;
  * after a level shift followed by a flat stretch (phase stable/volatile
    while the whole-period trend test and CAGR still count the step) a caveat
    says the rise/fall came from a one-off step, not a steady trend;
  * momentum differences inside the noise are called noise-level.
* `build_llm_payload`: JSON-serialisable facts — `series`, `conventions`
  (units; `skill`; `mase` "do not compare with 1"; `probabilities` pointing to
  `prob_increase_basis`; `momentum`, `slope_pct_per_period` and
  `robust_zscore` definitions; seasonal-strength notes; `intermittent`, which
  also says that the compared levels of intermittent demand are average
  demand per period, zeros included; `pct_undefined`), a history downsampled
  to ~60 points (recent detail kept)
  with a note, trend ends, `current_level`, `metrics`, `phase`,
  `changepoints`, `anomalies`, `forecast` (method + plain-language
  description, `backtest_mase`, `skill`, table of at most 36 rows),
  `model_leaderboard`, and `outlook`. Notable fields:
  * `metrics.pct_undefined`: why a percentage metric is null, e.g.
    `{'total_change_pct': 'sign_change', 'drawdown_pct': 'sign_change'}`, also
    for `slope_pct_per_period` / `momentum` / `volatility_cv`
    (`rate_pct_undefined`) and `'at_zero'` for a drawdown at the 0 floor;
    `metrics.drawdown_pct` is null when `drawdown_undefined` is set;
  * `anomalies[].robust_zscore` capped at ±100;
  * `phase.confidence_text` (`format_prob` text) and `phase.tentative`;
  * `changepoints[].label` (English) and `changepoints[].label_ko`;
  * `outlook`: `direction_label {ko, en}`, `expected_change_pct`,
    `expected_change_basis` (also explains a null for every reason — for
    `negative` "compare the two levels in compared_levels (in units)" — and,
    for intermittent demand with a percentage, that both levels are average
    demand per period, not the trend), `expected_change_undefined` (`None | 'zero' |
    'mostly_zero' | 'floor' | 'sign_change' | 'negative' | 'near_zero'`), `compared_levels {current,
    future}`, `prob_increase`, `prob_increase_text`, `prob_increase_basis`,
    `prob_increase_undefined` (`prob_undefined`; the three probability fields are
    null / say why when it is set), `scenarios_at_horizon_end`,
    `scenario_interval_level` (`interval_level`), `scenario_reference {value,
    date, basis}` (`scenario_reference`), `forecast_peak_date`, `future_phase`,
    `future_phase_ko`, `future_phase_basis` (`future_phase_status` in words for
    `'undecided'`, `'tentative'` and `'low_confidence'`, else `None`),
    `confidence`;
  * numbers are rounded so that they print as the surfaces print them
    (`prob_increase`, `phase.confidence`, the strengths, `mk_tau`, momentum,
    `slope_pct`, `volatility_cv`, the % changes).
* `SYSTEM_PROMPT` tells the model to use only the JSON facts (outside
  explanations at most as labelled hypotheses), quantify, state no
  percentage when `expected_change_pct` or a percentage metric is null (for
  `sign_change` say that the level crosses zero and give both levels; for
  `negative` compare the two levels in units; for `floor` say that the
  forecast falls to the floor of 0, never −100 %) and no probability when
  `prob_increase` is null, write a 0 bound of a nonnegative forecast as
  "0 (floor)" without a percentage, describe `prob_increase` as the chance
  that the future level exceeds the current one (never as P(horizon-end
  value > now)), say that the scenarios are a different quantity, call the
  compared levels of intermittent demand average demand per period (not the
  trend), use `direction_label`, follow
  `future_phase_basis`, present a tentative phase as tentative, quote
  `prob_increase_text` and `confidence_text` verbatim, name models in plain
  words, judge accuracy by `skill` and never read MASE against 1, write
  `>99%` / `<1%` instead of certainties, and follow the four-section
  Markdown-lite format (nested `  - ` bullets allowed); questions are
  answered directly first.
* **Claude call** (`anthropic` SDK imported lazily):
  `client.beta.messages.create(model=..., max_tokens=16000,
  betas=["server-side-fallback-2026-07-01"], fallbacks="default",
  output_config={"effort": "medium"}, system=SYSTEM_PROMPT, messages=[user])`
  — no `thinking`, `temperature`, `top_p`, `budget_tokens` or prefill.
  `stop_reason == "refusal"` is checked before reading content; only `text`
  blocks are joined. A response stopped by `max_tokens` or
  `model_context_window_exceeded` is returned with a `RuntimeWarning` and a
  truncation note in the request language. SDK errors are caught
  most-specific-first (`AuthenticationError`, `PermissionDeniedError`,
  `RateLimitError`, `APIStatusError`, `APIConnectionError`, `AnthropicError`,
  plus `TypeError` for missing credentials / an SDK too old for the
  parameters) and raised as `NarrativeUnavailable` with a message in the
  request language (English for `lang='en'`, Korean otherwise); the
  truncation and fallback `RuntimeWarning`s follow the language too.
  `generate_narrative(use_llm=True)` falls back to the rule-based text with a
  `RuntimeWarning` (source `'rule'`). Tests mock the client — no network.

## connectors.py — online data (optional)

```python
class ConnectorError(Localised, RuntimeError);  USER_AGENT;  WIKIMEDIA_PAGEVIEWS;  DEFAULT_START = "20160101"
def wikipedia_pageviews_url(article, lang="ko", start=None, end=None, granularity="monthly") -> str
def fetch_wikipedia_pageviews(article, lang="ko", start=None, end=None, granularity="monthly",
                              session=None) -> pd.Series
def fetch_google_trends(keyword, timeframe="today 5-y", geo="") -> pd.Series
```

Wikimedia REST pageviews (`user` agent type, all access, `monthly`/`daily`,
default 2016-01-01 to the last complete month/day), via `requests` or any
object with a compatible `.get`; Google Trends via the optional `pytrends`.
Generic User-Agent, no personal data, 20 s timeout; every failure is a
`ConnectorError` that says what to do, raised with a catalog message (codes
`conn_*` in `messages.MESSAGES`): Korean `str(exc)`, English `exc.text('en')`
(the CLI's `wiki` / `trends` follow `--lang`). Both return raw series.

## report.py — presentation

```python
def sparkline(values, width=None) -> str
def svg_chart(report, width=960, height=380) -> str
def render_text(report, width=78, lang="ko") -> str      # 'en': English summary; else Korean
def render_html(report, path=None) -> str
def render_compare_html(reports, ranking, path=None, links=None) -> str
CHANGEPOINT_KO, DIRECTION_KO, DIRECTION_ARROW, CONFIDENCE_KO, PERIOD_KO, UNIT_KO (Y = '연'),
MODEL_KO, PHASE_COLOR, KIND_COLOR ...
```

* Pure functions (byte-identical output for the same report). HTML pages are
  self-contained and script-free (inline CSS + SVG; the Noto Sans KR Google
  Fonts link is optional), light/dark themes via CSS custom properties,
  responsive (the chart is embedded in wide/mid/narrow renditions chosen by
  media queries; CSS-only hover read-out). Every user string is HTML-escaped
  (the terminal summary neutralises `<`/`>` and control characters). The HTML
  pages are **Korean only** (`<html lang="ko">`); an English narrative is
  shown as is, and the diagnosis card recognises reasons quoted in
  translation (`narrative.translate_reason`). `render_text(lang='en')` is an
  English terminal summary with no Hangul apart from the data itself (e.g.
  the series name).
* One vocabulary with the narrative: `MODEL_KO` **is** `narrative.MODEL_KO`
  (`@log` ids read `…(로그 공간)` via `narrative.model_label`); direction words
  come from `narrative.direction_label` (with `outlook_levels(...).undefined`),
  changepoint labels from `analysis.changepoint_label` (colour by label,
  `_CP_COLOR`), the method from `narrative.describe_method`, scenario floors
  from `narrative.scenario_value`, phase names from `PHASE_KO` /
  `narrative.PHASE_EN`, a tentative phase from `narrative.phase_is_tentative`
  (badge `잠정`) and intermittent demand from `narrative.is_intermittent` (or an
  intermittent-only leaderboard row: `_intermittent_only()` reads the registry's
  `offered_for == 'intermittent'` models, `sba` / `tsb`, plus `croston`). The outlook levels and their `undefined`
  reason come from `forecast.outlook_levels` (imported lazily, computed once
  per render and cached on `_ChartData.levels`).
* Numbers: `_fmt_num(x)` with automatic decimals **is**
  `narrative.format_number(x, signed=…)`. `_sci`, `_fmt_short`, `_axis_labels`
  and `_fmt_axis_value` use the ×10ⁿ notation of `numfmt` from 1e12 and below
  1e-4 (`_fmt_short` abbreviates 1e6–1e12 as `M` / `B`; no `T` and no
  e-notation labels). `_fmt_prob` returns exactly `narrative.format_prob(p)`
  and `_fmt_pct` is `_fmt_num(…) + '%'` (both `—` for None/NaN; the CLI
  ranking calls them). Dates are `narrative.format_date` (`_fmt_date`).
  Anomaly z-scores (chart tooltips, the events table) use
  `numfmt.zscore_text` (`z > +100` beyond 100).
* Shared rules (no mirrors): `_drawdown_undefined` / `_seasonal_trend_drawdown`
  call `narrative.drawdown_undefined` / `seasonal_trend_drawdown` (the drawdown
  basis), `_floor_note` `narrative.floor_note` and the scenario floors
  `narrative.scenario_floor`, the moving-phase hedge
  (`_future_phase_hedge`) `narrative.future_phase_status`, an all-zero history
  (`_all_zero`) `narrative.is_all_zero`, average-demand level names
  (`_demand_levels`) `narrative.is_intermittent`,
  `_CAGR_MIN_YEARS` is `analysis.CAGR_MIN_YEARS`, the `'zero'` outlook reason
  `narrative.zero_level_reason`, the rate gates `narrative.rate_pct_undefined`
  / `volatility_undefined` / `rate_reason_clause`, the missing probability
  `narrative.prob_undefined` / `prob_reason` (KPI `—` with its reason, the
  terminal's `상승 확률 — (최근 39개월 연속 0)`, the outlook card's sentence), the
  `'floor'` outlook reason `narrative.floor_change_reason` (terminal `예측 변화 —
  (예측이 하한 0까지 내려감)`, the levels end in `0(하한)`) and the scenario
  reference `narrative.scenario_reference` (`_scenario_ref`).
* A `—` is never left unexplained: a growth rate that is `None` carries its
  reason (`0을 지나 음수로 바뀌어 변화율 없음`, `간헐 수요라 계산하지 않음`,
  `기간이 1년 미만이라 연 환산하지 않음`, …; `_metric_undefined` / `_pct_why`), and
  an expected change without a percentage gives the reason and both levels
  (`_change_undefined`).
* `render_html` page: header (name, phase badge, range, frequency, horizon;
  the subtitle names the method kind: `단일 모델 예측`, `N개 모델 앙상블 예측` or
  `백테스트 없이 만든 기본 예측`), KPI tiles (현재 값, 전체 변화, 연평균 성장률
  CAGR, 최근 변화, 예측 변화, 상승 확률), chart, "AI 트렌드 해석" (the
  narrative), "미래 전망" card, "현재 진단", "과거의 사건" (numbered
  changepoints, anomalies; for intermittent demand the `expected` column reads
  `평소 규모`), "모델 리더보드" (with a `단순 대비` column: error / smallest
  error of the benchmark rows − 1, in MAE, or RMSE for intermittent demand,
  with a note when there are no benchmark rows; the final method in words),
  "예측 표", "분석 방법".
* KPI "현재 값" peak phrase: with `seasonal_strength >= 0.5` (and a seasonal
  decomposition) the **trend** drawdown is shown (`추세 고점 대비 X%`) when it
  differs from the raw drawdown by >= 5 points — the same rule as the
  narrative; otherwise the observed drawdown, with `(일회성 급등 제외)` when a
  flagged spike was left out of the peak. A drawdown that is no percentage
  (`pct_change_undefined`, e.g. a series that crossed zero) is given in units
  (`최고점보다 18.2 낮음`); intermittent demand names its largest demand, a
  count whose latest value is 0 its peak (`최고 8.00 (2024-03)`, never −100 %),
  and an all-zero history reads `관측 기간 내내 0`.
* "현재 진단" card: momentum and volatility read `—` with the narrative's reason
  where a % of the level means nothing (`평균 수준이 음수여서 %로 나타내지 않음`;
  `관측값이 모두 0이라 계산하지 않음`). KPI "상승 확률" and the outlook card give
  no probability where `narrative.prob_undefined` gives a reason (all zeros, a
  forecast of exactly 0, a current window of 0 that cannot fall), with that
  reason.
* KPI "예측 변화": a `'flat'` outlook gets a neutral value (no arrow or colour)
  and, for `방향 불확실`, that word; without a percentage the tile shows the
  reason and the two levels.
* "미래 전망" card: change and 상승 확률 both described as the future window
  average vs. the current window average (prefix `계절 보정한` and a note when
  `outlook_levels` adjusted a window; for intermittent demand
  `향후 12개월 평균 수요량` / `최근 12개월 평균 수요량`, `_demand_levels` =
  `narrative.is_intermittent`); the scenario values read `0(하한)` at
  the floor, with the narrative's floor note; the scenario bar compares the
  horizon-end scenarios with the observation at the same point of the seasonal
  cycle when `seasonal_strength >= 0.3`, else with the latest value, and for
  intermittent demand with the average demand of the last season
  (`_scenario_ref` = `narrative.scenario_reference`, the narrative's reference
  too; labelled: `세로선은 최근 12개월 평균 수요량입니다`, terminal `(비교: 최근
  12개월 평균 수요량 0.15)`). "미래 단계" is hedged by the narrative's rule
  (`_future_phase_hedge` returns `narrative.future_phase_status`): `판단 보류`
  for a moving phase only kept by a `'flat'` outlook, `신뢰도 낮아 참고용` for a
  phase change under low confidence, `관측치가 적어 잠정` for a tentative phase.
* Narrative rendering (`_markdown_lite`) accepts a superset of the narrative
  format: `#`–`######` headings, `-`/`*`/`•`/`+` bullets and `1.`/`1)` ordered
  lists, nesting by indentation (2+ spaces deeper; tab = 4), continuation
  lines, paragraphs, `**bold**`, `*italic*`, `` `code` ``; `---` rules are
  skipped; text is escaped before any markup is added.
* Chart: actuals, trend (dashed), forecast mean, 80 %/95 % bands, "현재"
  divider, changepoint lines with **numbered badges** (1..n in the order of
  `analysis.changepoints`, the same numbers as the HTML table), anomaly rings,
  nice ticks, legend. **Anomaly-aware y-range**: flagged anomalies are left
  out of the value range when they more than double it and are drawn as edge
  triangles with their values; the last observation always stays; the compare
  page's mini charts use the same range. Changepoint labels and colours come
  from `analysis.changepoint_label` (`성장 가속`, `하락세 둔화`,
  `하락→상승 반전`, `횡보→상승 전환`, …).
* `render_text`: name, phase (`잠정` / `tentative` when tentative), key metrics
  (momentum and volatility `—` under `rate_pct_undefined` /
  `volatility_undefined`) with a `※` line for every unexplained `—`, a sparkline of history │
  forecast, `예측 변화 X% (<future window>, <current window> 대비)` (or the
  reason and the two levels), 상승 확률 (same basis), confidence, horizon-end
  scenarios (`0(하한)` at the floor) with their reference, the projected phase
  hedged like the page, changepoints, the method in words, an `정확도:` line
  (skill-based, MASE last) and the top models (by RMSE for intermittent
  demand) in both languages as the readable name with the id merged into its
  parenthesis (`과거 유사 패턴(kNN; analog_knn) 2.15`, `logistic S-curve
  (logistic_growth) 1.34`; `_model_seg`), continuation lines indented by two
  spaces (`_wrap_segments(..., cont="  ")`). An all-zero history has no
  상승 확률 segment.
* `render_compare_html`: ranking table + small-multiple cards in ranking order
  (reports missing from `ranking` appended); a phase called on too few
  observations carries the `잠정` badge in the table and on its card, as on
  the report page. Changes are shown as reported,
  each **with its span always visible** — in the column header when every row
  shares it, otherwise next to each value (no annualised values; the score is
  unchanged). A `'flat'` outlook's expected change is neutral with its word
  (`보합` / `방향 불확실`) as a note; a missing one carries its reason. The
  rising score is described as the outlook (expected change, P(up)) plus the
  recent momentum (recent change, momentum). Card stats share label / value /
  note rows through a CSS subgrid. `links` (series name → href) turns card
  titles and ranking names into links; only relative and `http(s)` hrefs are
  used (others such as `javascript:` / `data:` are ignored), percent-encoded
  and HTML-escaped.

## compare.py — ranking

```python
RANK_COLUMNS = ["name", "phase", "phase_ko", "momentum", "recent_change_pct",
                "expected_change_pct", "prob_increase", "confidence", "score",
                "freq", "momentum_per_month", "recent_change_ann_pct", "expected_change_ann_pct"]
NOISE_PHASES = ("stable", "volatile")
def rising_score(report) -> float          # 0..100, 50 = neutral
def rank_trends(reports) -> pd.DataFrame   # best first, index = rank from 1, ties keep input order
def annualise_pct(pct, periods, ppy);  expected_change_annualised(report)
def recent_change_annualised(report);  momentum_per_month(report);  momentum_undefined(report)
```

Frequency-neutral rising score:

```
score = 50 + 50·(0.35·tanh(expected_ann/50) + 0.25·(2·P(up) − 1)
                 + 0.20·tanh(recent_ann/50) + 0.20·tanh(momentum_pm/2))
```

`expected_ann` compounds `expected_change_pct` to a yearly rate over the gap
between the outlook's two levels (`h − (k_fut−1)/2 + (k_cur−1)/2` periods);
`recent_ann` is the year-over-year change of the trend (annualised if
shorter); `momentum_pm` converts momentum to % per month; `P(up)` is
`prob_increase` as is. In `'stable'` / `'volatile'` phases the recent-change
and momentum terms are dropped (the outlook alone decides). Growth is capped
at +10,000 %/yr; for monthly data with a 12-month horizon the annualised
values equal the raw ones. The score blends the outlook with the recent
momentum, so it is not an expected rise by itself (the CLI and the compare
page call it "전망 + 최근 기세").

`recent_change_annualised` is `None` (the term then contributes nothing, and
the table never shows `—` next to a score driven by it) when
`metrics.recent_change_pct` is `None`, when the series is intermittent demand
(`analysis._is_intermittent`), or when
`analysis.pct_change_undefined(trend[-1-k], trend[-1], series)` gives a
reason (a P&L that fell through zero is no "−100 %"). Likewise the momentum
term drops out (`momentum_per_month` = 0) where `momentum_undefined`
(`analysis.level_rate_undefined`) is set, and `rank_trends` then puts `None` in
`momentum` / `momentum_per_month`; `prob_increase` is `None` where
`narrative.prob_undefined` gives a reason (a forecast of exactly 0, a current
level of 0 that cannot fall), and the score then counts P(up) as 0.5 (no
contribution). `compare` imports `analysis` and `narrative.prob_undefined` at
module level (no cycle: `analysis` imports only `data`, `numfmt` and `types`;
`narrative` does not import `compare`).

## pipeline.py — glue

```python
def run(series, name=None, freq=None, horizon=None, levels=(80, 95), use_llm=False, lang="ko",
        model="claude-opus-5-5", candidates=None, client=None) -> TrendReport
def run_many(series_map, errors=None, **kwargs) -> dict[str, TrendReport]
class SeriesFailedWarning(Localised, UserWarning)   # code 'series_failed'
def report_to_dict(report) -> dict;  report_to_json(report, indent=2);  reports_to_json(reports, indent=2)
MAX_HORIZON = 10_000
```

`run`: prepare → analyze → auto_forecast (of `analysis.series`, with
`random_walk=is_random_walk(analysis)`) → compute_outlook →
generate_narrative. The default horizon (`data.default_horizon(freq)`) is
resolved before the horizon checks, so it gets the same checks and warnings
as an explicit one. `DataError` for a horizon that is not a whole number
>= 1 (`inf` and `nan` included), a horizon above `MAX_HORIZON` or forecast
dates beyond pandas' range; `DataWarning` when the horizon (default included)
exceeds 5× the history (catalog messages). `run_many` skips failing series
with a `SeriesFailedWarning` (`Localised`: code `series_failed`, params
`series` / `detail`; and fills `errors` if given).

JSON exports (`report_to_dict` / `report_to_json`) contain history, trend,
metrics, phase, changepoints, anomalies, forecast (incl. `skill`, model
scores, table), outlook and narrative. `outlook` holds the `Outlook` fields
plus `direction_label {ko, en}` (the word every human surface shows),
`expected_change_undefined` (why `expected_change_pct` is `None`),
`compared_levels {current, future}` (from `forecast.outlook_levels`; `None`
for hand-made reports it cannot handle) and `prob_increase_undefined`
(`narrative.prob_undefined`): where it is set `prob_increase` is `null`, as on
every other surface (the `Outlook` object keeps its raw value). Floats keep 15 significant digits
(`_JSON_DIGITS`), NaN/inf become `null`, and dates are written as
`YYYY-MM-DD` without `strftime` (`_date_str`), so years after 9999 work.
`pipeline` imports `narrative.direction_label` / `prob_undefined` and
`forecast.outlook_levels` at module level.

## cli.py — command line

Subcommands: `analyze FILE`, `sample [NAME|all]` (`--n` >= 8, default 120;
`--seed`; monthly unless `--freq`), `list-samples`, `wiki ARTICLE...`
(`--wiki-lang`, `--start`, `--end`, `--granularity monthly|daily`),
`trends KEYWORD...` (`--timeframe`, `--geo`), `compare [FILES...]`
(`--samples a,b`), `ask FILE QUESTION`. Options of every analysing command
(all but `list-samples` and `ask`): `--horizon/-H` (>= 1), `--freq D|W|M|Q|Y`,
`--html`, `--json`, `--llm`, `--model` (default `claude-opus-5-5`),
`--lang ko|en`, `--quiet/-q`; `analyze` and `compare` also take `--date-col` /
`--value-col` (repeatable). `ask` takes `--date-col`, `--value-col`,
`--horizon`, `--freq`, `--model`, `--lang`, uses the first value column (with
a note) and answers with Claude (exit 1 if Claude is unavailable).

* One series: summary + narrative, one HTML report. Several: summaries and a
  ranking (`_format_ranking(df, lang, horizons, tentative)`: raw `expected_change_pct`
  with its horizon — in the header when shared, otherwise a `기간` column —
  then 상승확률 and 점수 (title: "상승 점수: 전망 + 최근 기세, 50=중립"), plus
  one note line from `_rank_basis(reports, lang)` that says which two levels
  예측 변화 / 상승확률 compare: the mean of the forecast's last season vs. the
  mean of the last observed season, adding "the whole horizon, seasonally
  adjusted" only when some horizon is shorter than its season and the yearly
  rule (last forecast value vs. latest value) only when yearly series are
  ranked; one note per reason a row shows `—` as its expected change, worded
  as that series' terminal summary (`_rank_undefined_notes`, via
  `report._change_undefined_short`: `※ 예측 변화 — (현재 수준이 음수): loss`,
  `※ 예측 변화·상승확률 — (관측값이 모두 0): zeros`); and a note when
  frequencies are mixed). Phases too short to judge read `·잠정` /
  `(tentative)`. `--html out.html` writes
  the compare page (with `links=`) and `out_<slug>.html` per series.
  `--json` (written before the HTML) is an object for one series and an
  array for several.
* `--lang` changes the narrative, the terminal summary
  (`render_text(report, lang=…)`, via `_print_report(report, quiet, lang)`),
  the CLI's own messages (`_MSG`), argparse type errors (`--lang` is read
  before parsing) and every catalog message (`messages.localise(exc, lang)`:
  `DataError` / `DataWarning` / `DataNotice` / `SeriesFailedWarning` /
  `ConnectorError`). The HTML pages stay Korean.
* Warnings are printed once each to stderr as `안내:` / `Note:` (`DataNotice`)
  or `경고:` / `Warning:` (other warnings) by `--lang`; data errors as
  `데이터 오류:` / `Data error:`; numpy floating-point notes are hidden.
* Exit codes: `0` success (also when stdout is a closed pipe), `1` runtime
  failure (network/connector, Claude, file save, unexpected error; set
  `TRENDAI_DEBUG=1` for the traceback), `2` input error (`DataError`, missing
  file, bad sample name, nothing to compare, every series failing with a
  `DataError`, argparse errors), `130` interrupted.

## numfmt.py — number notation

```python
MINUS = "−";  SCI_SMALL, SCI_LARGE = 1e-4, 1e12;  Z_SHOWN_MAX = 100.0
def superscript(e) -> str;  sci(a, digits=3) -> str;  uses_sci(a, scale=None) -> bool
def number_text(x, decimals=None, signed=False, scale=None) -> str | None
def zscore_text(z, label="z") -> str    # 'z = +9.6' / 'z > +100'; label='' for a table cell
```

The one notation of every surface (imports nothing of the package):
thousands separators, a precision chosen from the magnitude, ×10ⁿ below 1e-4
and from 1e12 up (never `e+06`, never a 13-digit integer), U+2212 minus.
`narrative.format_number`, `analysis._num_txt` (phase reasons), the report's
`_sci` and the data messages use it.

## messages.py — the message catalog

```python
MESSAGES: dict[str, tuple[str, str]]       # code -> (Korean template, English template)
class L(NamedTuple): ko; en                # a parameter that differs by language
class Msg(str)                             # the Korean text + .parts [(code, params)]; .text(lang)
def msg(code, **params) -> Msg;  join(*msgs) -> Msg | None;  render(code, params, lang) -> str
def localise(obj, lang="ko") -> str        # an exception / warning / Msg in lang
class Localised                            # mixin: .code, .params, .text(lang); str() stays Korean
def unit(freq) -> L;  format_label(label, lang="ko") -> str;  FREQ_EN
```

The texts of `DataError` / `DataWarning` / `DataNotice` (data.py), of
`pipeline.SeriesFailedWarning` and of `connectors.ConnectorError` (codes
`conn_*`) come from this catalog (about 80 codes). Library users see the
Korean text (`str(exc)`); the CLI renders the same message in `--lang`
(`localise(exc, lang)` / `exc.text('en')`). A parameter that is itself a `Msg`
(the reason inside "could not be analysed — …") is rendered in the same
language. Imports nothing of the package.

## Cross-module private dependencies

These private names are imported across modules and are therefore part of
the de-facto contract (rename only together):

* `forecast` imports from `analysis` at module level: `_is_intermittent`
  (`forecast._is_intermittent` **is** that rule: >= 30 % zeros overall and in
  the last `max(m, 12)` points; there is no second threshold),
  `_is_multiplicative` (periodic seasonal/trend ratio ⇒ log-space branch;
  `report` imports it lazily), `_pct_reference` (the % rule's reference level,
  `OutlookLevels.scale`) and the public `pct_change_undefined` (the outlook's
  percentage rule). `forecast.is_random_walk` lazily imports
  `analysis._noise_model` and `analysis._unit` (and reproduces compute_metrics'
  unit scaling to reuse its cache).
* `narrative` and `compare` import `analysis._is_intermittent` at module
  level; `narrative` also imports `analysis._mean_level` (the outlier-clipped
  mean level behind `volatility`, to state it in units).
* `cli` imports `report._fmt_pct`, `report._fmt_prob` (so `cli._fmt_pct` /
  `cli._fmt_prob` print exactly the reports' text) and
  `report._change_undefined_short` (the ranking notes), all lazily.
* `compare.expected_change_annualised` follows compute_outlook's window rule
  (`k_cur = min(m, n)`, `k_fut = min(m, h)`).

Public cross-module imports (the shared rules; `report` keeps no copy of a
`narrative` / `analysis` rule: its former `_MOVING_PHASES`, `_trend_drawdown`,
`_SEASONAL_DD_*` and local `_CAGR_MIN_YEARS` / `_SCI_*` constants are gone or
now imported): `report` imports from `narrative` `MODEL_KO`, `PHASE_EN`, `describe_method`,
`direction_label`, `drawdown_undefined`, `floor_change_reason`, `floor_note`,
`format_date`, `format_number`, `format_prob`, `future_phase_status`,
`is_all_zero`, `is_intermittent`, `model_label`, `period_unit`,
`phase_is_tentative`, `prob_reason`, `prob_undefined`, `rate_pct_undefined`,
`rate_reason_clause`, `scenario_floor`, `scenario_reference`,
`scenario_value`, `seasonal_trend_drawdown`, `translate_reason`,
`volatility_undefined` and `zero_level_reason`; from `analysis`
`CAGR_MIN_YEARS`, `changepoint_label` and `pct_change_undefined`; from `numfmt`
`SCI_SMALL`, `SCI_LARGE`, `sci` and `zscore_text`; and `models.MODEL_REGISTRY`
lazily. `narrative` imports `analysis.phase_window` and, lazily,
`forecast.outlook_levels`, `interval_level`, `is_random_walk` and
`fit_history`. `pipeline` imports `narrative.direction_label` /
`prob_undefined` and `forecast.outlook_levels`; `compare` imports
`narrative.prob_undefined`; `cli` imports `narrative.prob_undefined` /
`phase_is_tentative` and `forecast.outlook_levels` lazily.

Module-level imports (besides `types`): `numfmt` and `messages` import
nothing of the package; `models` imports nothing of the package; `data`
imports `messages` and `numfmt`; `analysis` imports `data` and `numfmt`;
`forecast` imports `models`, `analysis` and `data`; `narrative` imports
`analysis`, `data` and `numfmt`; `compare` imports `analysis`, `data` and
`narrative`; `report` imports `analysis`, `data`, `narrative` and `numfmt`;
`connectors` imports `messages`; `pipeline` imports `analysis`, `data`,
`forecast`, `messages` and `narrative`; `cli` imports `data` and `messages`
(the rest lazily). `forecast` is imported lazily by `narrative`, `report` and
`cli`, so there is no cycle.

## Known gaps

* The HTML pages are Korean only (`--lang en` changes the narrative inside them,
  not the page chrome). The argparse help texts (`--help`) and the generated
  name `행N` of an unlabelled row of a wide table are Korean too.
* `TrendMetrics.slope_pct`, `momentum` and `volatility` are typed `float` and
  stay numbers where a % of the level means nothing; every surface asks
  `analysis.level_rate_undefined` before printing them.
* `Outlook.prob_increase` is typed `float` and stays a number where
  `narrative.prob_undefined` says it is no finding; every surface asks it first
  (the JSON export nulls it).
* A nonnegative series whose trend falls to ~0 still reports a whole-period
  `total_change_pct` of −100 % (e.g. a daily series rising from 5 to 50 and
  then 0 for 450 days: `추세선 기준으로 5.00에서 5.33×10⁻¹⁸까지 **−100.0%**
  변했습니다.`; the drawdown names the peak instead, `'at_zero'`).
* The probability of a rise uses a normal approximation also for intermittent
  demand: with a wide band a forecast rate of 0.0034 against 0.038 a week
  reads "about 48 %".
* No business-day (`'B'`) or sub-daily frequency; daily data models only the
  weekly cycle (season length 7), not the yearly one.
* Prediction intervals are calibrated on synthetic tests (tests require
  80/95 % coverage within [0.65, 0.95] / [0.85, 0.995] at n=100 and
  [0.75, 0.97] / [0.88, 0.995] for n = 12–24) but can still under-cover on
  very short series and across regime changes; external shocks are not modelled.

## Tests

`python -m pytest -o addopts="" -q` (about 1,400 tests, ~3 min). Each module
has its own test file (`tests/test_<module>.py`, plus `test_connectors.py`);
`tests/conftest.py` provides a hand-built `fake_report` fixture. Tests never
touch the network (the Claude client and HTTP sessions are mocked).

## File ownership (historical)

The modules were first written in parallel by separate tasks; this table
records that split and is no longer binding.

| Module | Owner task | Tests |
|---|---|---|
| `trendai/types.py`, `data.py`, `samples.py`, `models.py` (base), `tests/conftest.py` | foundation (lead) | `tests/test_data.py` |
| `trendai/analysis.py` | analysis | `tests/test_analysis.py` |
| `trendai/models.py` (extend; keep base API) | models | `tests/test_models.py` |
| `trendai/forecast.py` | forecast | `tests/test_forecast.py` |
| `trendai/narrative.py`, `trendai/connectors.py` | narrative | `tests/test_narrative.py`, `tests/test_connectors.py` |
| `trendai/report.py` | report | `tests/test_report.py` |
| `trendai/pipeline.py`, `compare.py`, `cli.py`, `__main__.py`, `README.md` | integration | `tests/test_pipeline.py`, `tests/test_cli.py` |
