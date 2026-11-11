# trendai — architecture & module contracts

Goal: an AI that understands a trend's **past** (history, structural breaks),
its **present** (lifecycle phase, momentum) and predicts its **future**
(forecast with uncertainty + interpretation), for *any* time series
(search interest, page views, sales, prices, KPIs...). The user-facing text
is Korean first (English for the narrative and the CLI chrome on request).

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
| `Decomposition` | `trend`, `seasonal`, `resid` (`series == trend + seasonal + resid` exactly, no NaNs, edges included) and `period` (1 = no seasonal component). The seasonal part is a periodic pattern summing to 0 per period only for an **additive** decomposition; a **multiplicative** (log-space) one has `seasonal = exp(t + s) − exp(t)`, which scales with the level (tell them apart with `analysis._is_multiplicative(dec)`). After a kept level shift the trend jumps there (piecewise decomposition). |
| `Changepoint` | `date`, `index`, `slope_before`, `slope_after`, `kind` (one of `CHANGEPOINT_KINDS`: acceleration, deceleration, reversal_up, reversal_down, level_shift) and a unitless `magnitude >= 0`. |
| `Anomaly` | `date`, `index`, `value`, `expected`, signed robust `zscore` (see `detect_anomalies`). |
| `TrendMetrics` | descriptive past/present statistics; see the analysis section for the rules behind `total_change_pct`, `cagr_pct`, `recent_change_pct`, `momentum`, `volatility`, strengths, `mk_pvalue`, `peak_value` and `drawdown_pct`. |
| `LifecyclePhase` | `label` (one of `PHASES`: emerging, growing, peaking, declining, stable, reviving, volatile), `confidence` in [0, 1], `reasons` (2–4 short Korean sentences), property `label_ko` (`PHASE_KO`). |
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
  observations); partial-season windows are seasonally adjusted. It is not the
  change of the horizon-end value. `None` when undefined (current level ~0,
  sign change, at least half of the current window exactly 0, or a series with
  negative values whose level is within 0.5 SD of 0).
* `prob_increase` — P(future level > current level) for **the same two
  levels** (sign-consistent with `expected_change_pct`); NOT P(horizon-end
  value > today's value).
* `scenarios` — `{'pessimistic', 'base', 'optimistic'}` = lo80 / mean / hi80
  **at the horizon end** (single values that carry that date's seasonal effect).
* `direction` `'up' | 'down' | 'flat'`, `future_phase` (one of `PHASES`),
  `confidence` `'high' | 'medium' | 'low'` (from skill, sample size and the
  relative 80 % band width), `forecast_peak_date` or `None`.

---

## data.py — loading and regularising

```python
class DataError(ValueError)        # input cannot be used (Korean message naming the flag/argument)
class DataWarning(UserWarning)     # data was repaired (values dropped, gaps filled, merged...)
class DataNotice(UserWarning)      # how the input was interpreted (ignored columns, date order...)

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
  its native frequency). There is no business-day code: weekday-only daily
  data is interpolated over weekends (with a `DataWarning` suggesting
  `--freq W`), and sub-daily data is averaged to days.
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
```

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
    whose seasonal swing scales with the level (regression of the additive
    residual on `seasonal × (trend − mean)/mean`: slope >= 0.35 with t >= 3.5,
    trend max/min >= 1.25, and a smaller remainder than the additive fit) is
    decomposed in log space: `trend = exp(t)`, `seasonal = exp(t+s) − exp(t)`,
    `resid = y − trend − seasonal`.
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
    `cagr_pct` needs a span >= 0.95 years; `None` when the base is <= 0 or
    below 1 % of the mean absolute level, and for intermittent demand;
  * `momentum` = recent OLS trend slope (last `max(3, min(season_length,
    n//4))` periods) minus the Theil–Sen `slope`, in % of the *recent* trend
    level per period; `acceleration` = change of the recent slope between
    the previous and the latest window, in % of the mean level;
  * `volatility` and the strengths use the residual winsorised at 3.5 robust
    SDs (log components for multiplicative decompositions);
    `seasonal_strength` is additionally scaled by the evidence of a
    significance test of the repeating pattern (0 below the 95 % level);
  * `peak_value` / `peak_date` / `drawdown_pct` skip upward spikes flagged by
    `detect_anomalies` — only from 16 observations on and never at the first
    or last observation; `anomalies` passes the already detected ones
    (`analyze` does, so peak/drawdown exclude exactly the reported spikes).
* `lifecycle_phase`: explainable rules over the trend's shape (position and
  size of the peak, drawdown, rebound from the trough, recent slope vs. the
  best past slope, growth multiple over the base) plus significance
  (t-statistic of the recent slope, dependence-corrected Mann–Kendall).
  "Recent" = the last ~15 % of the observations. It can return `'stable'`
  after a level shift followed by a flat stretch, and confidence drops for
  random-walk-like series. Reasons are Korean with numbers (U+2212 minus,
  `p < 0.001`); dates before a particle use `2016년에` for yearly data and
  bracketed dates stay bare (`고점(2016)`); at >= 99.5 % of the trend peak the
  reason reads `현재 추세는 고점(YYYY-MM) 수준입니다.`
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
class OutlookLevels(NamedTuple): current, future, k_cur, k_fut, period, adj_cur, adj_fut, undefined
def norm_cdf(x); norm_ppf(p); t_cdf(x, df); t_ppf(p, df)
```

`auto_forecast` (raw input goes through `data.prepare`; `horizon=None` →
`data.default_horizon(freq)`; `horizon`, `top_k`, `n_windows` must be >= 1):

1. **History cap** — the backtest *and* the final fit use the same last
   `max(400, 8·m)` observations.
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
     models `ses`, `theta`, `holt_damped`, `sba` (those registered). They are
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
   with at least 6 backtest errors. Refit the chosen model(s) on the capped
   history (if all fail, the best-ranked model that works), guard, combine.
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
8. **Skill** — see `ForecastResult.skill`; errors below 1e-9 of the mean level
   count as exact (both exact: 1.0).
9. **Fallback** — fewer than ~8 points (no 2-window backtest) or every
   candidate failing: equal-weight drift + naive (naive alone for n < 3),
   wide heuristic intervals, `model_scores == []`, `skill = None`,
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
  With whole-season windows (default horizons for D/M/Q/Y) the levels are
  plain means. `outlook_levels` returns exactly these levels and
  `undefined` (`'zero'`, `'mostly_zero'`, `'sign_change'`, `'near_zero'`).
* `expected_change_pct = (future − current)/|current| × 100`, `None` when
  `undefined` is set.
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
def format_number(x, decimals=None, lang="ko", signed=False) -> str
def format_pct(x, lang="ko", decimals=1, signed=True) -> str
def format_prob(p, lang="ko") -> str
def format_date(ts, freq, lang="ko") -> str        # D/W 2024-03-05, M 2024-03, Q 2024-Q1, Y 2024
def period_unit(freq, lang="ko") -> str            # /월 /주 /일 /분기 /년 (/mo ... in English)
def translate_reason(reason) -> str | None
def direction_label(direction, expected_change_pct=None, lang="ko") -> str
def model_label(name, lang="ko") -> str;  describe_method(forecast, lang="ko") -> str
```

* **Format**: Markdown-lite — `## ` headings, `- ` bullets, sub-items of a
  line ending in `:` nested with a two-space indent (`  - `), `**bold**`.
  A bottom line (`**한눈에 보기** — …`) is followed by four sections:
  과거 / 현재 / 미래 전망 / 주의할 점 (Past / Present / Outlook / Caveats).
  Numbers use thousands separators and the U+2212 minus; Korean prose writes
  yearly dates as `2024년` while `format_date` and the payload keep `2024`.
* `format_prob`: whole percent, `>99%` for p >= 0.995 and `<1%` for p <= 0.005
  (exact 1 and 0 included), `해당 없음` / `n/a` for None/NaN. The report, the
  terminal summary and the CLI ranking print the same text.
* `direction_label`: `'flat'` reads `보합` when `|expected_change_pct| < 2 %`
  (or no percentage) and `방향 불확실` / `direction uncertain` otherwise.
* `translate_reason`: English for one Korean phase reason (input returned
  unchanged if it has no Hangul, `None` if no template matches; untranslated
  reasons are dropped from English text).
* **Outlook section order**: expected change (with the two compared windows;
  partial-season adjustment named) → probability (same two levels named) →
  horizon-end scenarios (80 % band, flagged as a different quantity that
  carries the end date's seasonal effect) → forecast peak → projected phase →
  confidence → `정확도:` / `Accuracy:` bullet from `ForecastResult.skill`
  (MASE only as a technical note; "did not beat naive" only when skill > 1) →
  method in plain words. With `expected_change_pct = None` no percentage is
  stated (scenario percentages omitted too).
* Past/present details: peak "(일회성 급등 제외)" when a flagged spike was left
  out; after a level shift followed by a flat stretch (phase stable/volatile
  while the whole-period trend test and CAGR still count the step) a caveat
  says the rise/fall came from a one-off step, not a steady trend; momentum
  differences inside the noise are called noise-level.
* `build_llm_payload`: JSON-serialisable facts — `series`, `conventions`
  (units; `skill`; `mase` "do not compare with 1"; `probabilities` pointing to
  `prob_increase_basis`; intermittent and seasonal-strength notes), a history
  downsampled to ~60 points (recent detail kept) with a note, trend ends,
  `current_level`, `metrics`, `phase`, `changepoints`, `anomalies`,
  `forecast` (method + plain-language description, `backtest_mase`, `skill`,
  table of at most 36 rows), `model_leaderboard`, and `outlook` with
  `expected_change_pct`, `expected_change_basis` (also explains a null),
  `compared_levels {current, future}`, `prob_increase`,
  `prob_increase_basis`, `scenarios_at_horizon_end`, `forecast_peak_date`,
  `future_phase`, `confidence`.
* `SYSTEM_PROMPT` tells the model to use only the JSON facts (outside
  explanations at most as labelled hypotheses), quantify, state no
  percentage when `expected_change_pct` is null, describe `prob_increase` as
  the chance that the future level exceeds the current one (never as
  P(horizon-end value > now)), say that the scenarios are a different
  quantity, use `direction_label`, name models in plain words, judge accuracy
  by `skill` and never read MASE against 1, write `>99%` / `<1%` instead of
  certainties, and follow the four-section Markdown-lite format (nested
  `  - ` bullets allowed); questions are answered directly first.
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
  parameters) and raised as `NarrativeUnavailable` with a Korean message.
  `generate_narrative(use_llm=True)` falls back to the rule-based text with a
  `RuntimeWarning` (source `'rule'`). Tests mock the client — no network.

## connectors.py — online data (optional)

```python
class ConnectorError(RuntimeError);  USER_AGENT;  WIKIMEDIA_PAGEVIEWS;  DEFAULT_START = "20160101"
def wikipedia_pageviews_url(article, lang="ko", start=None, end=None, granularity="monthly") -> str
def fetch_wikipedia_pageviews(article, lang="ko", start=None, end=None, granularity="monthly",
                              session=None) -> pd.Series
def fetch_google_trends(keyword, timeframe="today 5-y", geo="") -> pd.Series
```

Wikimedia REST pageviews (`user` agent type, all access, `monthly`/`daily`,
default 2016-01-01 to the last complete month/day), via `requests` or any
object with a compatible `.get`; Google Trends via the optional `pytrends`.
Generic User-Agent, no personal data, 20 s timeout; every failure is a
`ConnectorError` that says what to do. Both return raw series.

## report.py — presentation

```python
def sparkline(values, width=None) -> str
def svg_chart(report, width=960, height=380) -> str
def render_text(report, width=78) -> str
def render_html(report, path=None) -> str
def render_compare_html(reports, ranking, path=None, links=None) -> str
CHANGEPOINT_KO, DIRECTION_KO, CONFIDENCE_KO, PERIOD_KO, UNIT_KO (Y = '연'), MODEL_KO, PHASE_COLOR ...
```

* Pure functions (byte-identical output for the same report). HTML pages are
  self-contained and script-free (inline CSS + SVG; the Noto Sans KR Google
  Fonts link is optional), light/dark themes via CSS custom properties,
  responsive (the chart is embedded in wide/mid/narrow renditions chosen by
  media queries; CSS-only hover read-out). Every user string is HTML-escaped
  (the terminal summary neutralises `<`/`>` and control characters). The page
  and the terminal summary are **Korean only** (`<html lang="ko">`); an English
  narrative is shown as is, and the diagnosis card recognises reasons quoted
  in translation (`narrative.translate_reason`).
* `render_html` page: header (name, phase badge, range, frequency, horizon),
  KPI tiles (현재 값, 전체 변화, 연평균 성장률 CAGR, 최근 변화, 예측 변화,
  상승 확률), chart, "AI 트렌드 해석" (the narrative), "미래 전망" card,
  "현재 진단", "과거의 사건" (numbered changepoints, anomalies), "모델 리더보드"
  (with a `단순 대비` column: MAE / min MAE of the benchmark rows − 1, `—`
  without them), "예측 표", "분석 방법".
* KPI "현재 값" peak phrase: with `seasonal_strength >= 0.5` (and a seasonal
  decomposition) the **trend** drawdown is shown (`추세 고점 대비 X%`) when it
  differs from the raw drawdown by >= 5 points — the same rule as the
  narrative; otherwise the observed drawdown, with `(일회성 급등 제외)` when a
  flagged spike was left out of the peak.
* "미래 전망" card: change and 상승 확률 both described as the future window
  average vs. the current window average; the scenario bar compares the
  horizon-end scenarios with the observation at the same point of the seasonal
  cycle when `seasonal_strength >= 0.3`, else with the latest value
  (`_scenario_ref`; labelled).
* `_fmt_prob` returns exactly `narrative.format_prob(p)` (`—` for None/NaN).
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
  page's mini charts use the same range. Changepoint labels are
  direction-aware (`성장 가속`, `하락 가속`, `성장 둔화`, `하락세 둔화`).
* `render_text`: name, phase, key metrics, a sparkline of history │ forecast,
  `예측 변화 X% (<future window>, <current window> 대비)`, 상승 확률 (same basis),
  confidence, horizon-end scenarios with their reference, projected phase,
  method, an `정확도:` line (skill-based, MASE last) and the top models.
* `render_compare_html`: ranking table + small-multiple cards in ranking order
  (reports missing from `ranking` appended). Changes are shown as reported,
  each **with its span always visible** — in the column header when every row
  shares it, otherwise next to each value (no annualised values; the score is
  unchanged). `links` (series name → href) turns card titles and ranking names
  into links; only relative and `http(s)` hrefs are used (others such as
  `javascript:` / `data:` are ignored), percent-encoded and HTML-escaped.

## compare.py — ranking

```python
RANK_COLUMNS = ["name", "phase", "phase_ko", "momentum", "recent_change_pct",
                "expected_change_pct", "prob_increase", "confidence", "score",
                "freq", "momentum_per_month", "recent_change_ann_pct", "expected_change_ann_pct"]
NOISE_PHASES = ("stable", "volatile")
def rising_score(report) -> float          # 0..100, 50 = neutral
def rank_trends(reports) -> pd.DataFrame   # best first, index = rank from 1, ties keep input order
def annualise_pct(pct, periods, ppy);  expected_change_annualised(report)
def recent_change_annualised(report);  momentum_per_month(report)
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
values equal the raw ones.

## pipeline.py — glue

```python
def run(series, name=None, freq=None, horizon=None, levels=(80, 95), use_llm=False, lang="ko",
        model="claude-opus-5-5", candidates=None, client=None) -> TrendReport
def run_many(series_map, errors=None, **kwargs) -> dict[str, TrendReport]
class SeriesFailedWarning(UserWarning)
def report_to_dict(report) -> dict;  report_to_json(report, indent=2);  reports_to_json(reports, indent=2)
MAX_HORIZON = 10_000
```

`run`: prepare → analyze → auto_forecast (of `analysis.series`, with
`random_walk=is_random_walk(analysis)`) → compute_outlook → generate_narrative. `DataError` for a non-integer or `< 1` horizon, a horizon
above `MAX_HORIZON` or forecast dates beyond pandas' range; `DataWarning` when
the horizon exceeds 5× the history. `run_many` skips failing series with a
`SeriesFailedWarning` (and fills `errors` if given). JSON exports contain
history, trend, metrics, phase, changepoints, anomalies, forecast (incl.
`skill`, model scores, table), outlook and narrative; NaN/inf become `null`.

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
  ranking (`_format_ranking(df, lang, horizons)`: raw `expected_change_pct`
  with its horizon — in the header when shared, otherwise a `기간` column —
  then 상승확률 and 점수, plus one note line defining 예측 변화 / 상승확률 as
  "last forecast cycle mean vs. last observed cycle mean", and a note when
  frequencies are mixed); `--html out.html` writes the compare page (with
  `links=`) and `out_<slug>.html` per series. `--json` (written before the
  HTML) is an object for one series and an array for several.
* `--lang` changes the narrative and the CLI's own messages; the terminal
  summary and the HTML pages stay Korean.
* Warnings are printed once each to stderr as `안내:` (`DataNotice`) or
  `경고:` (other warnings); numpy floating-point notes are hidden.
* Exit codes: `0` success (also when stdout is a closed pipe), `1` runtime
  failure (network/connector, Claude, file save, unexpected error; set
  `TRENDAI_DEBUG=1` for the traceback), `2` input error (`DataError`, missing
  file, bad sample name, nothing to compare, every series failing with a
  `DataError`, argparse errors), `130` interrupted.

## Cross-module private dependencies

These private names are imported across modules and are therefore part of
the de-facto contract (rename only together):

* `forecast` imports `analysis._is_multiplicative` (periodic seasonal/trend
  ratio ⇒ log-space branch); `report` imports it lazily.
  `forecast.is_random_walk` lazily imports `analysis._noise_model` and
  `analysis._unit`.
* `narrative._outlook_windows` reproduces the outlook levels with
  `forecast._seasonal_component`, `_future_seasonal` and `_seasonal_share`
  (imported read-only); `forecast.outlook_levels` is the public equivalent.
  `narrative` also uses `analysis._is_intermittent`.
* `report` imports `narrative.format_prob` and `narrative.translate_reason`
  at module level (narrative imports only `data` / `types` at module level,
  so there is no cycle).
* `compare.expected_change_annualised` follows compute_outlook's window rule
  (`k_cur = min(m, n)`, `k_fut = min(m, h)`).

## Known gaps

* `narrative._outlook_windows` calls `forecast._future_seasonal(analysis, h)`
  without the forecast mean, while `compute_outlook` passes it. For a
  multiplicative decomposition with a partial-season forecast window (e.g.
  monthly data, horizon 6) the narrative's `future` level — and the payload's
  `compared_levels.future` — then differ from the level behind
  `expected_change_pct` (switching to `forecast.outlook_levels` fixes this).
* No business-day (`'B'`) or sub-daily frequency; daily data models only the
  weekly cycle (season length 7), not the yearly one.
* Prediction intervals are calibrated on synthetic tests (tests require
  80/95 % coverage within [0.65, 0.95] / [0.85, 0.995] at n=100 and
  [0.75, 0.97] / [0.88, 0.995] for n = 12–24) but can still under-cover on
  very short series and across regime changes; external shocks are not modelled.

## Tests

`python -m pytest -o addopts="" -q` (about 1,000+ tests, ~90 s). Each module
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
