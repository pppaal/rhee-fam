# trendai — architecture & module contracts

Goal: an AI that understands a trend's **past** (history, structural breaks),
its **present** (lifecycle phase, momentum) and predicts its **future**
(forecast with uncertainty + interpretation), for *any* time series
(search interest, page views, sales, prices, KPIs...).

Runtime deps: `numpy`, `pandas` only. Optional: `anthropic` (LLM narrative /
Q&A), `requests` / `pytrends` (online connectors). Python ≥ 3.10, pandas 3.x
compatible (note: pandas 3 stores datetimes at `us` resolution — never use
`.asi8` / assume ns).

```
CSV / samples / connectors
        │  data.prepare()  → regular float pd.Series + freq code ('D','W','M','Q','Y')
        ▼
analysis.analyze()   ── past & present ──►  AnalysisResult
        │                                   (decomposition, metrics, changepoints,
        │                                    anomalies, lifecycle phase)
forecast.auto_forecast() ── future ──────►  ForecastResult
        │   (models.py candidates, rolling-origin backtest, weighted ensemble,
        │    empirical prediction intervals)
forecast.compute_outlook() ─────────────►  Outlook
narrative.generate_narrative() ─────────►  Korean/English story (rule-based or Claude)
report.render_html()/render_text() ─────►  self-contained HTML report / terminal summary
pipeline.run() glues it together → TrendReport; cli.py exposes it.
```

All shared dataclasses live in `trendai/types.py` — **do not change their
fields** (adding helper methods/properties is fine). Foundation modules
already written: `types.py`, `data.py`, `samples.py`, `models.py` (base class,
baselines, registry), `tests/conftest.py` (`fake_report` fixture).

## File ownership (parallel implementation)

| Module | Owner task | Tests |
|---|---|---|
| `trendai/analysis.py` | analysis | `tests/test_analysis.py` |
| `trendai/models.py` (extend; keep base API) | models | `tests/test_models.py` |
| `trendai/forecast.py` | forecast | `tests/test_forecast.py` |
| `trendai/narrative.py`, `trendai/connectors.py` | narrative | `tests/test_narrative.py`, `tests/test_connectors.py` |
| `trendai/report.py` | report | `tests/test_report.py` |
| `trendai/pipeline.py`, `compare.py`, `cli.py`, `__main__.py`, `README.md` | integration (later) | `tests/test_pipeline.py`, `tests/test_cli.py` |

Only edit files you own. If you need something from a module you do not own,
code against the contract below (and, in tests, build inputs by hand or use
`tests/conftest.py`). Never touch `.git` state (no commits).

---

## analysis.py — past & present

```python
def decompose(series: pd.Series, period: int) -> Decomposition
def theil_sen_slope(y: np.ndarray) -> float
def mann_kendall(y: np.ndarray) -> tuple[float, float]          # (tau, two-sided p) with tie correction
def detect_changepoints(series: pd.Series, period: int = 1,
                        max_changepoints: int = 5,
                        min_segment: int | None = None) -> list[Changepoint]
def detect_anomalies(series: pd.Series, decomposition: Decomposition,
                     threshold: float = 3.5) -> list[Anomaly]
def compute_metrics(series: pd.Series, freq: str, decomposition: Decomposition) -> TrendMetrics
def lifecycle_phase(series: pd.Series, freq: str, decomposition: Decomposition,
                    metrics: TrendMetrics, changepoints: list[Changepoint]) -> LifecyclePhase
def analyze(series: pd.Series, freq: str | None = None, name: str | None = None) -> AnalysisResult
```

* `analyze` calls `data.prepare` first (so raw input is fine).
* `decompose`: robust additive decomposition with **no NaNs anywhere, including
  the edges** (extrapolate the trend at the ends with a local linear fit).
  STL-like (seasonal = per-phase robust average, centred to sum 0; trend =
  LOESS/local-linear smoother of the deseasonalised series; 1–2 robustness
  iterations). If `period <= 1` or fewer than 2 full periods: seasonal = 0.
* `detect_changepoints`: piecewise-linear segmentation of the deseasonalised
  series (binary segmentation or DP) with a BIC-like penalty, so pure
  linear+noise yields no (or very few, weak) breaks. Each break gets
  slope_before/after (fit on the neighbouring segments), `kind` from
  `types.CHANGEPOINT_KINDS`, and `magnitude` = |slope change| × segment length
  / series scale (or level jump / scale for level shifts). Default
  `min_segment = max(4, period // 2, n // 12)`. Sorted by date.
* `compute_metrics`: see field docs in `types.TrendMetrics`. Use the trend
  component for growth numbers (`total_change_pct`, `cagr_pct`,
  `recent_change_pct` over the last `season_length` periods, or 1 period for
  yearly data); `None` when the base is ≤ 0 or ~0. `momentum` = recent trend
  slope (last `max(3, min(season_length, n//4))` periods) relative to mean
  level in % per period **minus** `slope_pct` (positive ⇒ growing faster than
  its long-run average). `trend_strength = max(0, 1 - var(R)/var(T+R))`,
  `seasonal_strength = max(0, 1 - var(R)/var(S+R))`. `direction` from
  Mann-Kendall (p<0.05). All floats finite.
* `lifecycle_phase`: rule/score based over metrics (+ shape of the trend: where
  the peak is, curvature, recent vs long-run slope, drawdown). Must classify
  the archetypes in `samples.EXPECTED_PHASE` correctly (monthly n=120; also
  weekly n=156, daily n=400 at reasonable accuracy) across seeds. `reasons`: 2–4
  short Korean sentences with numbers. `confidence` in [0, 1].
* `detect_anomalies`: robust z (median/MAD) of the residual; |z| ≥ threshold.

## models.py — forecasters

Keep `Forecaster`, `Naive`, `SeasonalNaive`, `Drift`, `MODEL_REGISTRY`,
`register`, `make_model`, `available_models`, `default_candidates` working
as-is. Add and register (names are part of the contract):

| name | model |
|---|---|
| `ses` | simple exponential smoothing (optimised α) |
| `holt` | Holt linear trend |
| `holt_damped` | damped-trend Holt (φ optimised in ~[0.8, 0.98]) |
| `holt_winters` | additive Holt-Winters, damped trend (`needs_season`) |
| `theta` | Theta method (SES + ½ drift), seasonal adjustment when seasonal |
| `linear_trend` | OLS on time + Fourier seasonal terms |
| `ar` | autoregression on the differenced series, order by AIC, stable |
| `lag_ridge` | ML: ridge regression on lagged differences + seasonal lag, recursive multi-step |
| `analog_knn` | "past analogues": k nearest historical windows (z-normalised), average of what happened next |
| `logistic_growth` | S-curve (logistic) fit, `requires_positive`, saturation capped |
| `holt_damped@log`, `holt_winters@log`, `linear_trend@log` | same models in log space (`requires_positive`) |

Requirements: never raise for `len(y) >= min_obs` (degrade gracefully), finite
output, deterministic, fast (any single fit+predict ≤ ~50 ms at n=500; the
backtester fits each model several times). Reasonable accuracy on the
`samples` archetypes (e.g. `holt_winters` beats `naive` on `seasonal`/`growing`).

## forecast.py — future

```python
def mase_scale(y: np.ndarray, season_length: int) -> float     # in-sample seasonal-naive MAE (fallback naive), > 0
def backtest(y: np.ndarray, model_name: str, season_length: int, horizon: int,
             n_windows: int = 3) -> dict   # {'mae','rmse','smape','mase', 'forecasts': (W,h) array, 'actuals': (W,h) array}
def auto_forecast(series: pd.Series, horizon: int | None = None, freq: str | None = None,
                  levels: tuple[int, ...] = (80, 95), candidates: list[str] | None = None,
                  top_k: int = 3, n_windows: int | None = None,
                  nonnegative: bool | None = None) -> ForecastResult
def compute_outlook(analysis: AnalysisResult, forecast: ForecastResult) -> Outlook
```

* Rolling-origin backtest; candidates default to `models.default_candidates`.
  Rank by MASE; inverse-MASE weighted ensemble of the top-k; evaluate the
  ensemble on the same windows and keep the best single model instead if it
  is clearly better. `method` describes the result (e.g.
  `"ensemble(holt_damped 0.45, theta 0.35, linear_trend 0.20)"`).
* Prediction intervals from empirical backtest errors per horizon step
  (smoothed, e.g. σ_h ≈ σ₁·h^b, floor at in-sample one-step residual std),
  calibrated: ~80/95 % empirical coverage on synthetic data.
* `horizon=None` → `data.default_horizon(freq)`. Works for `n >= 3` (fallback
  to naive/drift with wide intervals when too short to backtest).
* `nonnegative=None` → True iff all observations ≥ 0; then clip mean/bounds at 0.
* Long series: may restrict backtest history (e.g. last ~6 seasons / 600 pts)
  for speed. Target: `auto_forecast` ≤ ~5 s for daily n=730.
* `compute_outlook`: current level = mean of the last `min(season_length, n)`
  observations; future level = mean of the last `min(season_length, horizon)`
  forecast points; `expected_change_pct` from these; `prob_increase` via a
  normal approximation from the 80 % interval at horizon end; scenarios =
  lo80 / mean / hi80 at horizon end; `forecast_peak_date` if the forecast
  mean peaks strictly inside the horizon and then falls ≥ 5 %;
  `future_phase` projected from current phase + direction; `confidence`
  from backtest MASE and sample size ('high' | 'medium' | 'low').

## narrative.py — the "AI interpreter"

```python
class NarrativeUnavailable(RuntimeError)
def rule_based_narrative(analysis, forecast, outlook, lang: str = "ko") -> str
def build_llm_payload(analysis, forecast, outlook) -> dict          # compact JSON-serialisable facts
def llm_narrative(analysis, forecast, outlook, lang="ko", model="claude-opus-5-5", client=None) -> str
def generate_narrative(analysis, forecast, outlook, use_llm: bool = False, lang="ko",
                       model="claude-opus-5-5", client=None) -> tuple[str, str]   # (text, 'llm'|'rule')
def ask(report: TrendReport, question: str, lang="ko", model="claude-opus-5-5", client=None) -> str
```

* Narrative is Markdown-lite (`## ` headings, `- ` bullets, `**bold**`),
  sections: 과거 / 현재 / 미래 전망 / 주의할 점 (English equivalents for `en`).
  Dates formatted per freq (`2024-03` for M, `2024-Q1`, `2024`, `2024-03-05`).
* LLM via the official `anthropic` SDK only (imported lazily): use
  `client.beta.messages.create(model="claude-opus-5-5", max_tokens=16000,
  betas=["server-side-fallback-2026-07-01"], fallbacks="default",
  output_config={"effort": "medium"}, system=..., messages=[...])`. Do **not**
  send `thinking`, `temperature`, `top_p`, or `budget_tokens`, and no
  assistant prefill. Check `stop_reason == "refusal"` before reading content;
  join `text` blocks only. Catch SDK errors most-specific-first
  (`AuthenticationError`, `RateLimitError`, `APIStatusError`,
  `APIConnectionError`) and raise `NarrativeUnavailable`.
  `generate_narrative(use_llm=True)` falls back to the rule-based text on
  `NarrativeUnavailable` (returning source `'rule'`). The model is told to
  only use the provided facts and to state uncertainty.
* Tests must mock the client — no network.

## connectors.py — online data (optional)

```python
class ConnectorError(RuntimeError)
def fetch_wikipedia_pageviews(article: str, lang: str = "ko", start: str | None = None,
                              end: str | None = None, granularity: str = "monthly",
                              session=None) -> pd.Series
def fetch_google_trends(keyword: str, timeframe: str = "today 5-y", geo: str = "") -> pd.Series
```

Wikimedia REST pageviews API (`/metrics/pageviews/per-article/{lang}.wikipedia/all-access/user/{article}/{granularity}/{YYYYMMDD00}/{YYYYMMDD00}`),
generic User-Agent (no personal data), timeouts, clear errors. Google Trends
via optional `pytrends` (clear message if missing). Tests mock everything.

## report.py — presentation

```python
def sparkline(values, width: int | None = None) -> str
def svg_chart(report: TrendReport, width: int = 960, height: int = 380) -> str
def render_text(report: TrendReport, width: int = 78) -> str
def render_html(report: TrendReport, path=None) -> str
def render_compare_html(reports: list[TrendReport], ranking: pd.DataFrame, path=None) -> str
```

Self-contained HTML (no external scripts; Google Fonts optional), Korean UI,
light + dark theme via CSS custom properties, responsive (SVG `viewBox`,
`width:100%`), all user strings HTML-escaped. Chart: actuals, trend
(dashed), forecast mean, 80 %/95 % bands, "현재" divider, changepoint markers,
anomaly dots, axis ticks with formatted dates/values, legend. Page: header
(name, phase badge, data range), KPI tiles, chart, narrative, changepoints,
model leaderboard, forecast table, method notes. `ranking` columns for
compare: `name, phase, momentum, recent_change_pct, expected_change_pct,
prob_increase, score`.
