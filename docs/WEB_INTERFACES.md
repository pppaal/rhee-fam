# Web app: interface split (team working note)

Status: working note for the four engineers splitting WP1 and WP3 (2026-11-25). The binding contract is still
`docs/WEB_SPEC.md`. This note only adds the seams *inside* WP1 and *inside* WP3. **The stub files are the source of
truth**: their signatures, docstrings and JSDoc are the agreed interface. Each half fills in its own files and
codes against the other half's stubs.

## 1. Who owns which files

| Half | Owns | Does not touch |
|---|---|---|
| **WP1-vm** (view model) | `trendai/web/viewmodel.py`, `fmt.py`, `texts.py`, `exports.py`, `vm.schema.json` | everything else in `trendai/web/` |
| **WP1-server** | every other `trendai/web/*.py` (`__init__`, `config`, `app`, `http`, `errors`, `api`, `jobs`, `analyze`, `uploads`, `store`, `ratelimit`, `budget`, `fetchers`, `narrative_ai`, `static_files`, `log`, `server`, `wsgi`, `testing`) and the E.16 hooks in `trendai/pipeline.py` and `trendai/narrative.py` | the five WP1-vm files |
| **WP3-chart** | `static/js/chart/*.js`, `static/css/chart.css`, `static/js/fmt.js`, `tests/js/fmt.test.mjs`, `tests/js/scale.test.mjs`, `tests/js/chart*.test.mjs` | everything else under `static/` and `tests/js/` |
| **WP3-app** | everything else under `trendai/web/static/` (incl. `js/dom.js`, `js/catalog.js`, `css/tokens.css`, `dev/fixtures/`) and `tests/js/` | the WP3-chart files |

Proposed test split for WP1 (the orchestrator's assignment wins if it differs): WP1-vm writes
`tests/test_web_viewmodel.py`, `tests/test_web_fmt.py` and, if useful, a unit-level `tests/test_web_exports.py`;
WP1-server writes every other `tests/test_web_*.py` from H.1, including the HTTP-level `test_web_share_export.py`.
`tests/js/fixtures/round_vectors.json` belongs with `fmt.test.mjs` (WP3-chart).

`js/dom.js` is already complete (written with these stubs). From now on WP3-app maintains it and WP3-chart only
imports it.

## 2. Python: WP1-server → WP1-vm

Stubs: `trendai/web/viewmodel.py`, `trendai/web/exports.py`, `trendai/web/fmt.py`, `trendai/web/texts.py`.
Every builder returns JSON-ready dicts for **one language**. The job calls each one twice (`"ko"`, `"en"`).

### 2.1 The analyze job (`analyze.py`, E.6), call by call

| Stage | WP1-server does | Calls into WP1-vm |
|---|---|---|
| data | fetches/prepares series, captures notices | `viewmodel.notice_from_warning(w, lang, series_index=k)` for each captured notice; `viewmodel.make_notice(code, lang, params=…, series_index=k, source_kind=…)` for `data_short`, `mostly_zero_volume`, `naver_missing_filled`, `naver_low_resolution` (condition: `viewmodel.is_low_resolution`), `horizon_long`, `horizon_clamped` (`span` = `fmt.span_label`), `naver_cached` (`fetched` = `fmt.datetime_label`); builds one `viewmodel.SourceMeta` per series |
| series | `pipeline.run`, English narrative, LLM payload | `SeriesInput(...)` → `viewmodel.build_series_vm(inp, lang)`; `viewmodel.build_source_block(meta, lang)` → `exports.build_series_json(report, block, lang, narrative=…)`; non-Naver: `exports.render_series_html(report, vm_ko["source"]["line"])` |
| series (failure) | maps the exception to a C.12.1 error object (`errors.py`) | `viewmodel.build_failed_entry(k, name, error_obj)` |
| finish | `compare.rank_trends`, `{display name: index}` | `viewmodel.assign_slots(ranked_indexes, request_size)` → `viewmodel.set_slot(vm, slot)` on every stored VM; `viewmodel.build_compare_vm(members, vms, ranking, name_to_index, slots, failed, lang)`; `exports.build_result_json(...)`, `exports.build_export_store(...)`, `exports.ranking_rows(...)`; non-Naver 2+: `exports.render_result_html(reports, ranking, compare_ko["source_line"])`; `viewmodel.export_availability(kind, xlsx=exports.xlsx_available(), view=…, succeeded=…)` → `result.json["available"]` (its `.series` / `.result`) and `viewmodel.export_block(av, lang, level="result")` |

Progressive VMs: `SeriesInput.slot = viewmodel.provisional_slot(k, request_size)` (`None` while N > 5).

### 2.2 Per response (`api.py`) and the AI job (`narrative_ai.py`)

* Export links: `viewmodel.link_exports(vm["exports"], available["series"][k], base, k)` for each series and
  `viewmodel.link_exports(stored["exports"], available["result"], base)` for the envelope; `base` is
  `/api/export/<result_id>` or `/api/export/s/<share_id>`.
* AI text: `viewmodel.attach_ai(vm, ai_record_or_None)`; shares: `viewmodel.make_shared_vm(vm)`.
* AI job: `record = viewmodel.build_ai_record(markdown, stored_vm_in_lang, model=…, created_at=…)` is stored as
  `ai/<k>-<lang>.json`; the done snapshot's `narrative` is `viewmodel.ai_job_narrative(record)`.
* `GET /api/samples`: `texts.SAMPLE_TITLES`, `texts.SAMPLE_TITLES_EN`, `texts.SAMPLE_DESCRIPTIONS_EN`.
  `GET /api/config`: `texts.NAVER_AGE_LABELS`, `features.xlsx_export = exports.xlsx_available()`.
* Error envelopes should build their `fixes` with `viewmodel.fix_object(action, params)`, so notices and errors
  share one fix shape.

### 2.3 Export routes (C.10, E.15)

`api.py` validates the path, loads the stored data, checks the stored `available` flags (Naver → 403
`export_naver_raw`, no openpyxl → 501 `export_xlsx_unavailable`, single-series `values.csv` → 404 `not_found`), and
then calls one serializer. It sends `ExportFile.body` with `ExportFile.headers()` plus the E.12 headers. For HTML,
the sandbox CSP in `headers()` replaces the app CSP. Serializers raise `exports.ExportUnavailable(code, status)` as
a second line of defence.

| Route | Serializer (inputs are the stored data of language `hl`) |
|---|---|
| `/<id>/<k>.csv` · `.xlsx` | `exports.series_csv(vm, file_id=id)` · `exports.series_xlsx(vm, file_id=id)` |
| `/<id>.csv` · `.xlsx` | `exports.result_csv(vms, compare, result_json["ranking_rows"], file_id=id)` · `exports.result_xlsx(vms, compare, file_id=id)` |
| `/<id>/values.csv` | `exports.values_csv(vms, compare, file_id=id)` |
| `/<id>[/<k>].html` | `exports.html_export(stored_html, file_id=id, k=k, display_name=…)` |
| `/<id>[/<k>].json` | `exports.json_export(export_store, file_id=id, k=k, display_name=…)` |
| `/s/<sid>…` | the same functions with `file_id=sid` and the share snapshot's files |

### 2.4 Storage additions (internal to WP1, both halves rely on them)

* `vm-<lang>.json` = `{series, compare, notices, exports}`: `exports` is the stored result-level block (D.5, links
  `None`). E.7 lists only the first three keys.
* `export-<lang>.json` = `exports.build_export_store(...)` = `{"series": {"<k>": …}, "result": …}`.
* `ai/<k>-<lang>.json` = `build_ai_record(...)`, i.e. the D.2.9 `ai` object plus `date_links`.

## 3. JavaScript: WP3-app → WP3-chart

Stubs: `static/js/fmt.js`, `static/js/chart/{scale,axes,fanchart,comparechart,sparkline,crosshair,png,describe}.js`.
Pages call only `fanchart`, `comparechart`, `sparkline`, `png`, `describe` and `fmt`. `scale`, `axes` and
`crosshair` are internal to the charts (`scale.js` and the pure helpers in `crosshair.js` are unit-tested).

| Caller (WP3-app) | API |
|---|---|
| `views/detail.js` | `createFanChart(container, seriesVM, {lang, t, band, trend, anomalies, announce?})` returns `{update, setBand, setToggles, highlightDate, highlightChangepoint, onChangepoint, onPointFocus, getSvg, destroy}` |
| `views/compare.js` | `createCompareChart(container, compareVM, seriesVMs, {lang, t, scale, bands, hidden, focus, announce?})` returns `{update, setScale, setBands, setFocus, setHidden, onLegendToggle, onFocusChange, highlightDate, getSvg, destroy}` |
| `rankcards.js`, `samples.js` | `sparkline(spark, {slot, grey, display})` returns `SVGSVGElement` |
| `sharemenu.js`, compare `⋯` menu | `exportPng(chart.getSvg(), {title, footer, filename: pngFilename(name)})` returns `Promise<Blob>` (the page shows `export.png.saving` / `err.png_failed`) |
| `narrative.js` (date links) | `chart.highlightDate(iso)` (the page scrolls the chart card into view) |
| `past.js` (changepoint list) | `chart.highlightChangepoint(n)` / `chart.onChangepoint(cb)` (two-way, F.7.4) |
| everyone | `fmt.js`: `valueText`, `dateLabel`, `datetimeLabel` (expiry dates), `pct`, … |

**i18n in charts**: data labels come from the VM (titles, descriptions, axis labels, changepoint/anomaly texts,
names). The few UI words come from `opts.t(key, params)` with these catalog keys: `chart.forecast`,
`chart.readout_actual`, `chart.readout_forecast`, `chart.index_axis`, `common.reference_only`, `a11y.chart_keys`,
`compare.title`, `compare.grey_hint`. Two new keys are needed in `catalog.js` (WP3-app adds them):

| key | ko | en |
|---|---|---|
| `chart.describe.compare` | {n}개 시계열 비교: {names}. | Comparison of {n} series: {names}. |
| `chart.describe.top` | 상승 점수 1위 {name}. | Highest rising score: {name}. |

**DOM and CSS**: charts build everything with `dom.svg()` / `dom.h()` inside the container they are given; they
own its children (and empty it on `destroy()`). `chart.css` owns the classes the charts emit, prefixed `chart-`
and `spark-`. WP3-app never styles those classes, and WP3-chart styles nothing outside its container. Colours come
only from the `tokens.css` variables (`--slot-N`, `--c-hist`, …).

**`dom.js` conventions** (complete): `h(tag, attrs, ...children)`, `svg(tag, attrs, ...children)`, `clear(el)`,
`focusFirst(el, {preventScroll})`, `focusables(el)`, `FOCUSABLE_SELECTOR`. Events go only through
`attrs.on = {click: fn}`; any other `on*` key throws. `style` must be an object (it is applied with
`setProperty`); `class` takes a string or an array; `dataset` and `props` take objects. `true` gives an empty
attribute, and `false`/`null` leave the attribute out. Text children always become text nodes.

## 4. Decisions taken where the spec is silent or ambiguous

1. **Whole-result JSON of a comparison** (C.10): `{"source_line", "source": [...], "reports": [...]}`. The spec
   asks for a top-level `source` array and a `source_line`, which a bare array cannot hold. A single series stays
   one object.
2. **JSON export narrative** is the narrative in the export's language (`build_series_json(..., narrative=)`).
   `report_to_dict` would otherwise always carry the Korean text.
3. **Notices**: `viewmodel` owns the notice and fix shapes. `analyze.py` builds the series and result notices, and
   `build_compare_vm` adds only the scale notices (`log_hint`, `index_unavailable`) to `compare.notices`. Those are
   not repeated in the envelope's `notices`, and the client renders both lists.
4. **Slots** are assigned from the ranking *order* (`assign_slots(ranked_indexes, N)`), so ties follow
   `rank_trends`' stable order.
5. **`SourceMeta`** carries data dates only (D.2.2). The request bounds (`query.period`) stay in WP1-server's
   normalised query.
6. **Export file names**: a comparison's display name is the series names joined with `" vs "`, cut to 80
   characters. The PNG name uses the same join.
7. JS `roundHalfAway` returns the digit string of Python's `str(round_half_away(x, d))`. The fourth parameter of
   `valueText` is called `isCompact` so it does not shadow `compact()`.
