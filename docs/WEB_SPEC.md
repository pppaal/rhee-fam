# trendai web app — implementation spec

Date 2026-11-25 (rev. 2, after the feasibility and design reviews) · Status: **binding implementation contract**
for the four work packages (WP1–WP4) · Inputs: `docs/WEB_DESIGN_RESEARCH.md` (the design brief, "the brief", cited as
§x.y), `docs/ARCHITECTURE.md`, `README.md`, the code as of commit `0b7f1ce` (re-checked at `88c3492`: Python 3.13,
pandas 3.0.5, Node 22). Review findings that were *not* applied, and partial applications, are explained in the
**Decisions log (J)**.

How to read this document:

* **MUST / MUST NOT / SHOULD** are normative. Anything not specified is the owning WP's choice, as long as it
  does not change a contract in sections B.3, C or D.
* Every JSON example is complete in its *shape*. Long arrays are shortened with `…` and comments are written
  `// like this`; real responses contain neither.
* This spec supersedes the brief's §8 endpoint sketch (SSE, server SVG, `[web]` extra) wherever they differ (A.3).
* Four engineers implement this in parallel without talking. The only coupling points are the contracts in
  **B.3** (Python signatures), **C** (HTTP API), **D** (view model) and **C.12 / G** (message codes and copy).
  When a WP needs something another WP delivers, it codes against the contract here and uses an
  *uncommitted local stub* until the other WP's code lands.

Contents: [0 Conventions](#0-conventions-and-glossary) · [A Scope](#a-scope) ·
[B Files & ownership](#b-file-layout-and-ownership) · [C HTTP API](#c-http-api-contract) · [D View model](#d-view-model) ·
[E Backend internals](#e-backend-internals-wp1-unless-stated) · [F Frontend](#f-frontend-architecture-wp3) ·
[G Copy](#g-copy-ui-catalog-staticjscatalogjs-wp3-server-texts-noted) · [H Tests](#h-test-plan) ·
[I Definition of Done](#i-acceptance-checklist-definition-of-done) · [J Decisions log](#j-decisions-log)

---

## 0. Conventions and glossary

| Term | Meaning |
|---|---|
| **result** | The stored outcome of one analysis job: 1–20 analysed series plus, for 2+, a comparison. Identified by `result_id`. |
| **job** | Asynchronous work item. `kind = "analyze"` (produces a result; **`job_id == result_id`**) or `kind = "narrative"` (AI rewrite of one series' narrative). |
| **series index `k`** | 0-based position of a series in the request order (stable for the life of a result). |
| **VM** | View model: the JSON the frontend renders (section D). A result stores one VM per language (`ko`, `en`); the API returns the one asked for with `hl`. |
| **slot** | Comparison colour/icon slot 1–5 (section D.4, F.8). |
| **source kind** | `naver` · `wiki` · `upload` · `google_trends_upload` · `sample`. |
| **value kind** | `relative_index` (Naver) · `relative_index_gt` (Google Trends upload) · `pageviews` (Wikipedia) · `raw` (uploads, samples). |

* **IDs**: `secrets.token_urlsafe(16)` → 22 characters `[A-Za-z0-9_-]`. Every route/store lookup validates
  `^[A-Za-z0-9_-]{22}$` before touching the disk. Share delete tokens are generated the same way but are never
  part of a URL.
* **Time**: stored and transported as UTC ISO-8601 with `Z` (`2026-11-25T00:12:03Z`). Data dates are
  `YYYY-MM-DD`. **Displayed** times use fixed KST (UTC+09:00, no DST; do not depend on tzdata).
* **Numbers in JSON**: raw floats rounded to 15 significant digits (the `pipeline._clean` rule); NaN/inf → `null`.
  Every JSON response and stored file is encoded with `json.dumps(..., ensure_ascii=False, allow_nan=False)` *after*
  that cleaning, so a stray NaN raises (→ `500 internal_error`, logged) instead of producing JSON that
  `JSON.parse` rejects. Display strings are pre-formatted by the server where section D.6 says so.
* **Languages**: `ko` (default) and `en`. API GET endpoints take `?hl=ko|en` (anything else → `ko`).
* **Engine version** `ENGINE = trendai.__version__` and **VM schema** `WEB_VM_SCHEMA` (an int constant in
  `trendai/web/viewmodel.py`, starting at `1`, bumped by WP1 whenever the VM builder's output changes in any way) are
  part of every result-cache key (section E.10); stored results and shares carry `schema` (E.7).

---

## A. Scope

### A.1 MVP coverage (brief §6.1)

| ID | Feature | Status | Where | Notes |
|---|---|---|---|---|
| M1 | Search-first home, source switch, example keywords, default-value chips (period, unit, horizon, Naver device/gender/age) | **In** | WP3 (UI), WP1 (`/api/config`) | Chips row on results; bottom sheet on mobile. |
| M2 | All state in the URL, back-safe, `hl` | **In** | WP3 | URL grammar F.2. View-only params use `replaceState`. |
| M3 | Compare ≤ 5, chips add/remove, fixed colour+icon per series, Naver in one request, axis rule raw/index/log, no dual axis | **In** | WP1 (one Naver call, scale defaults), WP3 | Slot 3 icon changed (A.3). |
| M4 | Compare chart + ranking cards + table view | **In** | WP1 (compare VM), WP3 | |
| M5 | Detail hero + 6 KPI tiles (incl. peak season) + fan chart | **In** | WP1 (VM incl. peak season), WP3 | Peak season algorithm D.2.4. |
| M6 | Phase badge + "왜 이 국면인가요?" (`phase.reasons`, certainty, rules link) | **In** | WP1, WP3 | Structured reasons are S1 (not MVP). |
| M7 | Per-series accuracy line + weak-forecast display rule | **In (partial wording)** | WP1, WP3 | The engine does not expose the number of backtest windows, so the line reads "과거 데이터 검증: …" without "N개 시점" (A.2). |
| M8 | P(rise) as words + number + basis, 5 % rounding | **In** | WP1 (`fmt`), WP3 | Word bins = the narrative's (D.1.5). |
| M9 | Changepoint/anomaly annotations (≤ 5 on chart + list, two-way highlight, anomalies in words) | **In** | WP1, WP3 | |
| M10 | Rule-based narrative by default, links in text, AI on click, AI label, AI off for Naver | **In (partial)** | WP1, WP3 | Dates in the narrative link to the chart through a server-built map (D.2.9); number→tile links deferred (A.2). AI result arrives by job polling, not streaming (A.3). |
| M11 | Upload: drop / select / paste → confirm step → results | **In** | WP1 (`/api/upload`, in-memory cache), WP2 (bytes loader), WP3 | Naver DataLab Excel uploads are *not* promoted or specially handled (A.2). |
| M12 | Google Trends CSV guide + loader support (preamble, `<1`, `term: (region)`) | **In** | WP2 (loader), WP3 (guide card) | |
| M13 | First run: 9 samples (synthetic badge), example keywords, upload card, recent analyses (this browser) | **In** | WP1 (`/api/samples`), WP3 | |
| M14 | Loading: skeleton < 1 s, staged progress, progressive fill, cancel | **In (4 steps)** | WP1 (job stages), WP3 | 4 honest steps instead of 5 (A.3). |
| M15 | Error/empty-state catalog via `messages.py` codes (ko/en) | **In** | WP2 (texts), WP1 (mapping), WP3 (rendering) | Catalog C.12. |
| M16 | Naver HUB connector + legacy fallback + server cache ≤ 21 d + per-IP limit + monthly budget guard | **In** | WP2 (connector), WP1 (cache, limits, budget) | "Delay queue at 80 %" replaced (A.3). |
| M17 | Wikipedia connector: UA with contact, 404 guidance | **In** | WP2, WP1 | |
| M18 | Source line + value kind + data-as-of on every result and export | **In** | WP1, WP3 | Detail: `source.line`; compare: `compare.source_line` (D.4); CSV/XLSX: comment/condition rows; JSON: `source` block; HTML: a source paragraph inserted into the rendered report; PNG: footer (C.10, F.7.6). |
| M19 | Export CSV (BOM), Excel (condition rows), PNG, HTML report, summary copy; no raw Naver ratio | **In** | WP1 (files), WP3 (PNG, copy) | HTML report disabled for Naver results; Naver JSON stripped of raw values (A.3). |
| M20 | Share: public sources = URL; uploads private by default + unguessable read-only link with expiry & delete | **In** | WP1, WP3 | Expiry *extension* deferred (A.2). |
| M21 | Korean number/date notation (만/억/조, range first, %p) | **In** | WP1 (`fmt.py`), WP3 (`fmt.js`, same vectors D.1) | |
| M22 | Up/down colours: ko = domestic (red up), en = international, user setting; anomalies/decline never red/blue | **In** | WP3 | |
| M23 | Accessibility basics (table view, keyboard crosshair, focus/tap tooltips, 3:1, 24 px, `<desc>`) | **In** | WP3 | F.9. |
| M24 | Mobile: one column, bottom-sheet chips, press-and-drag scrub, fixed read-out bar, 2-col tiles, card tables | **In** | WP3 | |
| M25 | ko/en UI switch | **In** | WP3 (UI catalog), WP1 (VM per language), WP2 (messages) | The "HTML 리포트" export stays Korean (existing renderer). |
| M26 | Theme system/light/dark with report tokens + §4.2 additions | **In** | WP3 | Tokens F.8. |
| M27 | `/method`, `/sources`, `/privacy` | **In** | WP3 | Content outline G.10–G.12. |

### A.2 Deferred or partial MVP items (and why)

| Item | Deferred part | Why |
|---|---|---|
| M7 | "과거 **N개 시점** 검증" count | `ForecastResult` does not expose the backtest window count; adding it changes `forecast.py`, outside every WP. Wording without N is equally true. |
| M10 | Linking numbers in the narrative to tiles | Matching free-text numbers to tiles is ambiguous (the narrative prints 1-decimal %, tiles print rounded %). Dates are linked (exact match on `YYYY-MM[-DD]`). |
| M20 | Extending a share link's expiry | Not in the lead decisions (30 days, deletable). A new share can be created instead. |
| §3.4 | Showing > 5 coloured series is MVP; *grey line click-to-highlight* is in, *per-series colour editing* is not | Not in the brief either. |
| §3.9 | OG preview image | S7 (not MVP). |
| §3.3 | Full-screen mobile search view with recent searches | Kept simple: the search box expands to full width and shows examples + recent below it (no separate view). |
| §5.1 / K6 | Uploading a **Naver DataLab Excel export** as a Naver fallback (`upload_datalab` fix, `/upload#datalab` guide) | No verified sample of the current DataLab export layout (condition rows above the table, paired columns) exists in the repo, so WP2 cannot build or test a detector. Without one such a file would also be classed as a plain `upload` (value kind "원래 단위", index scale, AI allowed), which misstates a relative ratio and bypasses the Naver restrictions. The guide, the fix action and the catalog keys are removed; S-scope item: a `naver_datalab_upload` source with the Naver rules, built from a real export fixture. |
| §3.5 | Phase reasons in plain words instead of test statistics | The engine's reasons are free text (`analysis` phase rules); rewriting them in the VM would duplicate the phase logic. The VM moves the trailing statistics into a `detail` sub-line and reformats dates (D.2.5); plain-word structured reasons stay S1. |

### A.3 Brief items overridden by lead decisions or by this spec

| Brief says | This spec does | Why |
|---|---|---|
| SSE `GET /api/jobs/<id>/events`, AI narrative streamed (§3.5, §3.7, §8) | Job + polling (`GET /api/jobs/<id>` every ~500 ms); AI rewrite is a `narrative` job | Lead decision 2. Works through any proxy and the stdlib server. |
| First render reuses `report.py` SVG (§0 #12, §8) | Client-side SVG chart from JSON; `render_html` only for the "HTML 리포트" export | Lead decision 3 (interaction, theming, i18n). |
| Web deps in an optional `[web]` extra (§8) | No web dependencies at all (stdlib WSGI) | Lead decision 1. |
| 5 progress steps (§3.7) | 4 steps: ① 데이터 확인 ② 추세·구조 분석 ③ 예측 모델 검증 (k/N) ④ 정리 | `analysis.analyze()` does decomposition, changepoints and anomalies in one call; showing them as separate steps would be fake progress (§3.7 forbids fake timing). |
| Paste → "임시 .tsv로 저장해 load_table" (§3.6) | Pasted text and uploaded bytes are parsed in memory; the parsed table lives in an in-process cache for 30 min; nothing raw is written to disk | Lead decision 5. |
| Naver budget: "80 % → delay queue" (§5.1.4) | ≥ 80 %: per-IP fresh-call limit drops from 20/h to 5/h; ≥ 90 %: no fresh calls | A queue that holds HTTP requests is fragile on a stdlib server; a lower rate gives the same protection. |
| HTML report & JSON export for every source (§3.10) | Naver: no HTML report (it embeds raw ratios in SVG/tooltips/tables); JSON export drops raw observations | Same rule as the CSV/Excel "no raw ratio" default (§3.10, §5.1.6), applied consistently. The working legal judgement (to be confirmed, docs/WEB.md legal checklist): **displaying** Naver ratios on screen (chart, read-out, table views, the API responses that feed them) is allowed; **downloadable files and copy buttons** must not carry raw ratios. Model outputs on the ratio scale (trend, forecast, bands, scenarios, expected values) are derived results and may be exported. |
| One example-keyword list (§3.3) | Per-source example lists (`home.example_terms.naver` / `.wiki`) | Without Naver keys the default source is Wikipedia, where most keyword examples are not article titles (e.g. no `선크림` article): the first click would be a 404. |
| `index` scale allowed for every compare (§3.4) | Not offered for `naver` / `google_trends_upload` | One request is already one scale; rebasing each term to its own first year erases the cross-term size comparison and may count as modifying the provider's results (§5.1.6). |
| Likelihood word next to "약 N%" (§4.6) | The number is always named: `(상승 확률 약 20%)` | A bare number next to a fall word ("하락 쪽이 우세 (약 20%)") reads as the chance of the fall. |
| Slot 3 icon ▲ triangle (§4.2) | Slot 3 icon = **hexagon ⬢** | ▲ means "rise" next to every number (§4.4); a teal ▲ before a name reads as a rise. |
| Probability "약 80%" only | Plus explicit extremes "95% 이상" / "5% 이하" | 5 % rounding would otherwise print "약 100%" (a certainty, which §4.5 forbids). |
| `/r/<id>` lists "only in the creating browser" | Same, implemented with `localStorage` (`trendai.recent`) | No accounts in MVP. |

### A.4 Non-goals (MVP)

Accounts, watch lists and alerts (S9) · BYOK (S10) · Shopping Insight (S11) · AI Q&A panel (S5) · structured phase
reasons/counterfactuals (S1) · seasonality bar panel (S2) · quick-compare chips (S3) · absolute-volume conversion
(S4) · event notes (S6) · OG images (S7) · Wikipedia autocomplete/redirects (S8) · period brush (S12) · live
Google Trends, pytrends in the server, any scraping · ads · multi-process deployment (E.13) · a JS/CSS build step or
any JS/CSS library · server-side PNG rendering.

### A.5 Unverified facts and the safe choice taken

| Unverified (brief §7) | Choice |
|---|---|
| HUB keywords per group: 5 or 20 | **5** (`NAVER_MAX_KEYWORDS = 5`). |
| Meaning of missing Naver buckets | Re-grid, fill with 0, notice `naver_missing_filled` with the count. |
| Weekday a Naver weekly bucket starts on | Request `startDate` snapped forward to a **Monday** and `endDate` back to a **Sunday**; use returned `period` values as given; drop a trailing bucket whose `period + 6 d > endDate`. Both interpretations then give complete weeks. |
| HUB monthly quota reset day | Budget counted per **KST calendar month**. |
| Applicability of the 2026-09 Naver terms to DataLab | Assume they apply: no AI, cache ≤ 21 d, no raw ratio export, text attribution, no ads. |
| Wikimedia UA policy wording | UA always carries version + public URL + contact when configured. `TRENDAI_CONTACT` is **required** for a public deployment: start-up fails (exit 2) when it is unset and the app is public (non-loopback bind or `TRENDAI_PUBLIC_URL` set), unless `TRENDAI_ALLOW_NO_CONTACT=1` (E.1, `WebConfig.validate_public`); on loopback it only warns. |
| Current Google Trends CSV layout (incl. Korean UI) | Detection by structure (`term: (region)` headers; optional `Category:`/`카테고리:` preamble), not by fixed strings. |
| Pretendard file names | Verified: npm `pretendard@1.3.9`, `dist/web/variable/pretendardvariable-dynamic-subset.css` + 92 files in `woff2-dynamic-subset/` (3.1 MB), `dist/LICENSE.txt` (OFL-1.1). Tarball sha512 `PaQAADyLY5v4kYFwkpSJHbSSYIkiriY/1xXw75TKoZ9UQQqeU+tvP05yTdZAWibiIYoo8ZKtRv8PM7w0IaywSw==`. |

---
## B. File layout and ownership

### B.1 Tree (new `+`, changed `~`)

```
~ pyproject.toml                         WP4
~ .gitignore                             WP4   (+ .trendai-data/, node_modules/)
+ Dockerfile                             WP4
+ .dockerignore                          WP4
~ README.md                              WP4   (new section "웹 앱")
~ docs/ARCHITECTURE.md                   WP4   (new "web/" section + cross-module private deps)
+ docs/WEB.md                            WP4   (operator guide)
~ trendai/cli.py                         WP4   (serve subcommand)
~ trendai/pipeline.py                    WP1   (run(..., on_stage=None))
~ trendai/narrative.py                   WP1   (llm_narrative_from_payload)
~ trendai/connectors.py                  WP2
~ trendai/data.py                        WP2
~ trendai/messages.py                    WP2
+ trendai/web/__init__.py                WP1
+ trendai/web/config.py                  WP1
+ trendai/web/app.py                     WP1
+ trendai/web/http.py                    WP1
+ trendai/web/errors.py                  WP1
+ trendai/web/api.py                     WP1
+ trendai/web/jobs.py                    WP1
+ trendai/web/analyze.py                 WP1
+ trendai/web/viewmodel.py               WP1
+ trendai/web/texts.py                   WP1
+ trendai/web/fmt.py                     WP1
+ trendai/web/exports.py                 WP1
+ trendai/web/uploads.py                 WP1
+ trendai/web/store.py                   WP1
+ trendai/web/ratelimit.py               WP1
+ trendai/web/budget.py                  WP1
+ trendai/web/fetchers.py                WP1
+ trendai/web/narrative_ai.py            WP1
+ trendai/web/static_files.py            WP1
+ trendai/web/log.py                     WP1
+ trendai/web/server.py                  WP1
+ trendai/web/wsgi.py                    WP1
+ trendai/web/testing.py                 WP1
+ trendai/web/vm.schema.json             WP1   (machine-readable VM contract, D.0; read by WP1 and WP3 tests)
+ trendai/web/static/**                  WP3   (see F.1 for the full list)
+ tests/test_web_*.py                    WP1   (list in H.1)
~ tests/test_connectors.py               WP2
+ tests/test_connectors_naver.py         WP2
~ tests/test_data.py                     WP2   (only if an existing assertion must change; none expected)
+ tests/test_data_gtrends.py             WP2
+ tests/test_data_notices.py             WP2   (collect_notices / emit_notice, wide de-duplication, max_columns)
+ tests/test_messages_web.py             WP2
+ tests/js/**                            WP3   (Node unit tests of pure ES modules)
+ tests/e2e/**                           WP4   (Playwright E2E, pytest wrapper, fixtures)
+ tests/test_cli_serve.py                WP4
+ tests/test_packaging.py                WP4
```

No file appears in two packages. Nobody edits `analysis.py`, `forecast.py`, `models.py`, `report.py`,
`compare.py`, `samples.py`, `types.py`, `numfmt.py`, `tests/conftest.py`.

### B.2 Work packages

| WP | Owns | Delivers |
|---|---|---|
| **WP1 backend** | `trendai/web/*.py`, `trendai/web/vm.schema.json`, `trendai/pipeline.py` (hook + `emit_notice`), `trendai/narrative.py` (one public function), `tests/test_web_*.py` | WSGI app, router, API (C), VM builder (D), jobs, store, limits, budget, caches, exports, AI job, static serving + SPA fallback, server helpers, test client and fakes (`trendai.web.testing`). |
| **WP2 data & connectors** | `trendai/connectors.py`, `trendai/data.py`, `trendai/messages.py`, `tests/test_connectors*.py`, `tests/test_data*.py`, `tests/test_messages_web.py` | Naver connector (HUB + legacy), stdlib HTTP session, period snapping, Wikipedia weekly + UA + timeout code, bytes-based table loader with Google Trends support, thread-safe notice capture (`collect_notices`), every new message code with ko/en text (C.12), web-safe rendering of existing codes and multi-part messages (`render_web_msg`). |
| **WP3 frontend** | `trendai/web/static/**` (incl. fonts, dev fixtures), `tests/js/**` | SPA (vanilla ES modules + CSS, no build), chart components, UI catalog ko/en, fmt.js, all pages, PNG export, accessibility. |
| **WP4 integration** | `trendai/cli.py`, `pyproject.toml`, `.gitignore`, `Dockerfile`, `.dockerignore`, `docs/WEB.md`, `README.md`, `docs/ARCHITECTURE.md`, `tests/e2e/**`, `tests/test_cli_serve.py`, `tests/test_packaging.py` | `trendai serve`, package data, container, operator guide, docs, E2E suite + pytest wrapper. |

### B.3 Cross-WP contracts

#### B.3.1 WP2 → WP1 (Python)

All names below are added to the module's `__all__`. WP1 imports them at module level.

```python
# ---------------------------------------------------------------- trendai/connectors.py (WP2)
import datetime as dt
from dataclasses import dataclass, field

NAVER_HUB_URL = "https://naverapihub.apigw.ntruss.com/search-trend/v1/search"
NAVER_LEGACY_URL = "https://openapi.naver.com/v1/datalab/search"
NAVER_EARLIEST = dt.date(2016, 1, 1)
WIKI_EARLIEST = dt.date(2016, 1, 1)
NAVER_MAX_GROUPS = 5
NAVER_MAX_KEYWORDS = 5            # HUB docs say 5 and 20; the safe value
NAVER_AGE_CODES = ("1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11")
TIMEOUT = 20                       # seconds, every outbound request (existing constant, unchanged)

class HttpResponse:                # what UrllibSession returns
    status_code: int
    headers: dict[str, str]        # lower-case keys
    content: bytes
    @property
    def text(self) -> str: ...     # content decoded as UTF-8 (errors="replace")
    def json(self) -> Any: ...     # json.loads(content); raises ValueError on bad JSON
    def raise_for_status(self) -> None: ...   # RuntimeError("HTTP <status>") for >= 400

class UrllibSession:
    """Stdlib HTTP client with the `.get` / `.post` subset the connectors use (no `requests` needed).
    HTTP error statuses are *returned* (not raised); network failures raise OSError subclasses
    (urllib.error.URLError, TimeoutError, ...). Uses urllib's default proxy handling (HTTPS_PROXY etc.)
    and ssl.create_default_context(). Responses larger than max_bytes raise OSError("response too large").
    Stateless and thread-safe: no mutable state after __init__ (no cookie jar, no connection pool); every call
    builds its own opener/request, so one instance may be shared by concurrent job threads (E.2)."""
    def __init__(self, *, max_bytes: int = 10_000_000) -> None: ...
    def get(self, url: str, headers: Optional[Mapping[str, str]] = None, timeout: float = TIMEOUT) -> HttpResponse: ...
    def post(self, url: str, json: Any = None, headers: Optional[Mapping[str, str]] = None,
             timeout: float = TIMEOUT) -> HttpResponse: ...

def user_agent(public_url: Optional[str] = None, contact: Optional[str] = None) -> str:
    """'trendai-web/0.1.0 (+https://trendai.example; ops@example.com)'. Parts that are None are left out;
    with neither -> 'trendai-web/0.1.0'. Characters outside printable ASCII are dropped."""

def period_bounds(range_spec: str, unit: str, today: dt.date,
                  earliest: dt.date = NAVER_EARLIEST) -> tuple[dt.date, dt.date]:
    """Whole, *completed* periods for a range chip.
    unit: 'date' | 'week' | 'month'.  range_spec: '1y' | '3y' | '5y' | 'all' | 'YYYY-MM-DD~YYYY-MM-DD'.
    end   = last completed period's last day: date -> today-1; week -> the last Sunday strictly before
            today; month -> last day of the previous month.  A custom end is clamped to this and then
            snapped back to a period end.
    start = for 'Ny': end - N years + 1 day; 'all': earliest; custom: as given; then clamped to >= earliest
            and snapped *forward* to a period start (Monday / 1st of month).
    Raises ConnectorError(msg('conn_bad_range', value=...)) for an unparsable spec or start > end."""

@dataclass(frozen=True)
class NaverCredentials:
    endpoint: str                  # 'hub' | 'legacy'
    key_id: str                    # HUB key id  | legacy client id
    key: str                       # HUB key     | legacy client secret
    def __repr__(self) -> str: ... # never shows key_id/key: "NaverCredentials(endpoint='hub', key_id='***', key='***')"
    @classmethod
    def from_env(cls, environ: Mapping[str, str] = os.environ) -> Optional["NaverCredentials"]:
        """HUB when NAVER_API_HUB_KEY_ID and NAVER_API_HUB_KEY are both non-empty; else legacy when
        NAVER_CLIENT_ID and NAVER_CLIENT_SECRET are; else None.  No runtime fallback between endpoints."""

@dataclass(frozen=True)
class NaverGroup:
    name: str                      # groupName, 1..100 chars, unique within a request (the web app sends the wire
                                   # names 'g1'..'g5', never display names — E.6)
    keywords: tuple[str, ...]      # 1..NAVER_MAX_KEYWORDS non-empty strings

@dataclass
class NaverResult:
    series: dict[str, pd.Series]   # group name (wire name) -> float ratio series, request order; DatetimeIndex of bucket
                                   # starts (name 'date'), complete grid start..end, missing buckets = 0.0;
                                   # s.attrs['trendai_source'] = {'kind': 'naver', 'group', 'keywords',
                                   # 'time_unit', 'device', 'gender', 'ages', 'endpoint'}
    endpoint: str                  # 'hub' | 'legacy'
    time_unit: str                 # 'date' | 'week' | 'month'
    start: dt.date
    end: dt.date
    filled: dict[str, int]         # group name -> buckets filled with 0
    raw_max: float                 # largest ratio in the response (normally 100.0)

def build_naver_body(groups: Sequence[NaverGroup], start: dt.date, end: dt.date, time_unit: str,
                     device: Optional[str] = None, gender: Optional[str] = None,
                     ages: Sequence[str] = ()) -> dict:
    """The JSON body (brief §5.1.2). Validates: 1..5 groups (conn_naver_too_many_groups), names 1..100 chars and
    unique (else conn_naver_bad_filter with what='groupName'),
    1..5 keywords each (conn_naver_too_many_keywords), time_unit in date/week/month, device in
    {None,'pc','mo'}, gender in {None,'m','f'}, ages subset of NAVER_AGE_CODES (conn_naver_bad_filter),
    start >= NAVER_EARLIEST, start <= end (conn_bad_range). Optional keys are omitted when empty;
    ages are sorted numerically."""

def naver_cache_key(groups: Sequence[NaverGroup], start: dt.date, end: dt.date, time_unit: str,
                    device: Optional[str] = None, gender: Optional[str] = None, ages: Sequence[str] = ()) -> str:
    """sha256 hex of the canonical JSON of build_naver_body(...) (sort_keys, separators=(',',':'),
    ensure_ascii=False) — no credentials, no endpoint. Group order matters; age order does not."""

def fetch_naver_search_trend(groups: Sequence[NaverGroup], start: dt.date, end: dt.date,
                             time_unit: str = "week", device: Optional[str] = None,
                             gender: Optional[str] = None, ages: Sequence[str] = (), *,
                             credentials: NaverCredentials, session: Any = None,
                             timeout: float = TIMEOUT) -> NaverResult:
    """One POST to the HUB (X-NCP-APIGW-API-KEY-ID / X-NCP-APIGW-API-KEY) or legacy endpoint
    (X-Naver-Client-Id / X-Naver-Client-Secret), Content-Type application/json, User-Agent user_agent().
    session: object with .post(url, json=, headers=, timeout=) (default UrllibSession()).
    Parses results[i] by position (title must equal the group name; else conn_naver_bad_response):
    'ratio' as float (number or numeric string), 'keywords' as list or comma string; drops a trailing
    weekly bucket whose period + 6 days > end; re-grids and fills missing buckets with 0.
    Errors -> ConnectorError (codes C.12.3); exc.log_detail: str holds the upstream errorCode/message
    (<= 200 chars, never headers or keys) for operator logs only."""

def fetch_wikipedia_pageviews(article, lang="ko", start=None, end=None, granularity="monthly",
                              session=None, *, user_agent: Optional[str] = None) -> pd.Series:
    """Existing function, extended:
    * granularity 'weekly': fetches daily data from start (snapped forward to Monday) to end (snapped back
      to Sunday) and returns Monday-indexed sums of complete Monday–Sunday weeks;
    * user_agent: sent as the User-Agent header (default: the existing USER_AGENT constant);
    * session=None and `requests` not importable -> UrllibSession() (no conn_need_requests any more);
    * a timeout (exception class name contains 'Timeout', or TimeoutError / socket.timeout) raises
      ConnectorError code 'conn_timeout' instead of 'conn_network'."""

# ---------------------------------------------------------------- trendai/data.py (WP2)
@dataclass
class GoogleTrendsInfo:
    terms: dict[str, tuple[str, Optional[str]]]   # final column name -> (term, region or None)
    lt1_count: int                                 # '<1' cells read as 0.5
    category: Optional[str]                        # text after 'Category:' / '카테고리:' if present

@dataclass
class RawTable:
    frame: pd.DataFrame            # the table *as read*: str cells for text formats (csv/tsv/txt/json), pandas'
                                   # native cell types for Excel (Timestamps, floats incl. 2020.0 for integer
                                   # columns with blanks) -- exactly what load_table hands frame_to_series today;
                                   # column labels as str (header row applied, GT preamble skipped, GT columns
                                   # renamed, '<1' -> '0.5', 'isPartial' dropped). WP1 stringifies cells only for
                                   # the upload preview (E.17), never before frame_to_series.
    filename: str
    format: str                    # 'csv' | 'tsv' | 'txt' | 'xlsx' | 'xls' | 'json'
    encoding: Optional[str]        # e.g. 'utf-8-sig', 'cp949'; None for Excel
    decimal_comma: bool            # pass to frame_to_series(prefer_decimal_comma=...)
    google_trends: Optional[GoogleTrendsInfo]

def read_table_bytes(raw: bytes, filename: str, *, max_columns: Optional[int] = None) -> RawTable:
    """Parse file bytes in memory (never touches the disk). Format from the extension of `filename`
    (.csv .tsv .txt .xlsx .xlsm .xls .ods .json; unknown -> text sniffing), Excel via io.BytesIO.
    Emits the same DataNotice/DataWarning as load_table (+ gt_detected, gt_lt1_half). Raises DataError.
    max_columns: for text formats the header row is split with the sniffed delimiter *before* the full parse and
    more than max_columns fields raise DataError(msg('table_too_many_columns', n=..., max=...)); for Excel/JSON the
    check runs right after reading. None = no limit (load_table's behaviour)."""

def load_table_bytes(raw: bytes, filename: str, date_col: Optional[str] = None,
                     value_cols: Optional[Iterable[str]] = None) -> dict[str, pd.Series]:
    """read_table_bytes + frame_to_series. For a Google Trends file every series gets
    attrs['trendai'] |= {'source': 'google_trends_upload', 'term': ..., 'region': ...,
    'value_kind': 'relative_index_gt'}  (the same value-kind name as the VM, section 0)."""

# load_table(path, ...) keeps its signature and behaviour; it now reads the bytes and delegates to
# load_table_bytes(raw, Path(path).name, ...). frame_to_series(...) keeps its signature and is public (WP1 calls
# it), with two additive changes:
#  * attrs['trendai'] gains 'date_cols': list[str] -- the real column name(s) the dates came from
#    (['연도', '월'] for a year + period combination, ['날짜'] for one column, [] for a wide table). The existing
#    'date_col' display string ('연도 + 월') is unchanged.
#  * a wide table with duplicate row names (and any other source of duplicate series names) gets unique keys:
#    the 2nd, 3rd ... occurrence is renamed '<name> (2)', '<name> (3)' (series.name too) instead of silently
#    overwriting the earlier series in the returned dict.

@contextmanager
def collect_notices() -> Iterator[list[Warning]]:
    """Capture the data layer's notices for the *current thread* (contextvars.ContextVar) instead of emitting them
    through `warnings`: while the block is active, every DataNotice/DataWarning that trendai code would emit is
    appended -- as an instance whose args[0] is the Msg -- to the yielded list, and warnings.warn is NOT called.
    Nested blocks: the innermost list receives. Process-global warnings state (filters, showwarning) is never
    read or changed, so concurrent threads cannot see or lose each other's notices (H.2 test)."""

def emit_notice(message: str, category: type = DataWarning, stacklevel: int = 2) -> None:
    """The single emission point (data._warn delegates here; WP1's pipeline.py calls it for 'horizon_long'):
    appends to the active collect_notices() list if there is one, else warnings.warn(message, category,
    stacklevel=stacklevel + 1) -- i.e. outside a collect block the behaviour, category and reported caller line
    are exactly today's (the CLI and every existing warning test are unchanged)."""

# ---------------------------------------------------------------- trendai/messages.py (WP2)
MESSAGES: dict[str, tuple[str, str]]   # + every code in C.12 (texts given there, copy verbatim)
WEB_OVERRIDES: dict[str, tuple[str, str]]   # web wording of existing codes whose text names CLI flags/Python
def render_web(code: str, params: Optional[Mapping[str, Any]] = None, lang: str = "ko") -> str:
    """One code: like render(), but uses WEB_OVERRIDES[code] when present and applies the web parameter rules of
    render_web_msg. Never returns text containing '--', '파이썬', 'Python' or 'trendai.' (tested)."""
def render_web_msg(m: "Msg | Localised | BaseException", lang: str = "ko") -> str:
    """The web text of a whole message -- what WP1 uses for EVERY captured notice and every DataError /
    ConnectorError it shows (never Msg.text(), never render_web(m.code, m.params), which drop parts or bypass
    WEB_OVERRIDES). Rules:
      * every part of a joined Msg is rendered (with WEB_OVERRIDES) and the parts are joined with a space;
      * a param that is itself a Msg is rendered with render_web_msg (web wording); an L is resolved for lang;
      * params named 'label' or 'date_format' go through format_label(value, lang) (Korean format labels such as
        'YYYY-MM-DD (일/월 순)' never reach English text);
      * list/tuple params are written as ‘a’, ‘b’ (curly single quotes, ', ' separator) instead of a Python repr;
        a param named 'examples' (a pre-joined "'x', 'y'" repr string) is re-quoted the same way (parse with
        ast.literal_eval('[' + s + ']'); on failure keep the string);
      * Korean particles written in the template right after a parameter -- '은(는)', '이(가)', '을(를)',
        '과(와)', '(으)로' -- are resolved with josa() on the substituted text.
    For a non-Localised exception: the generic text of 'series_failed_web' (the exception text is never shown)."""
def josa(word: str, pair: str) -> str:
    """word + the right particle for pair in {'은/는', '이/가', '을/를', '과/와', '으로/로'}: looks at the last
    Hangul syllable of word after stripping trailing quotes/brackets/spaces (’ ” ) ] 」 』); final consonant ->
    first form ('으로' also takes '로' after ㄹ); a trailing digit is read in Korean: 0 (영/십/백/천/만), 1 일,
    3 삼, 6 육, 7 칠, 8 팔 have a final consonant (ㄹ for 1, 7, 8), 2 4 5 9 do not; Latin letters / other -> the
    second form ('로', '는', ...). Tested with ‘번호’, 104주, 36개월, 10년, CP949, UTF-16."""
def fix_label(action: str, lang: str = "ko") -> str:
    """Label of a fix action: render('fix_' + action, {}, lang). KeyError for unknown actions."""
```

#### B.3.2 WP1 → WP4 (Python)

```python
# trendai/web/__init__.py
from .app import TrendaiApp, create_app
from .config import WebConfig
from .log import install_warning_baseline
__all__ = ["create_app", "TrendaiApp", "WebConfig", "install_warning_baseline"]

# trendai/web/config.py
@dataclass(frozen=True)
class WebConfig:            # every field in E.1; defaults = the E.1 defaults
    ...
    static_dir: Optional[str] = None     # override only (no env var): None -> the package's static/ directory
    @classmethod
    def from_env(cls, environ: Mapping[str, str] = os.environ, **overrides) -> "WebConfig": ...
    def is_public(self) -> bool: ...     # host not loopback (127.0.0.0/8, ::1, localhost) or public_url set
    def validate_public(self) -> None:
        """ValueError('TRENDAI_CONTACT ...') when is_public() and not contact and not allow_no_contact.
        Called by `trendai serve` and by trendai.web.wsgi before the app is built (A.5, E.1)."""

# trendai/web/log.py
def install_warning_baseline() -> None:
    """Process owners only (`trendai serve`, trendai.web.wsgi, tests/e2e/server.py; never create_app):
    warnings.simplefilter('ignore') for every category -- the web app reads data notices through
    data.collect_notices(), never through `warnings` (E.13) -- and remembers that it did, so the sweeper re-asserts
    the same baseline every run (undoing any filter state left behind by third-party catch_warnings interleavings)."""

# trendai/web/app.py
def create_app(config: "WebConfig | Mapping[str, Any] | None" = None, *,
               fetchers: "Fetchers | None" = None,
               clock: "Callable[[], datetime.datetime] | None" = None) -> "TrendaiApp":
    """config None -> WebConfig.from_env(); a Mapping -> WebConfig.from_env(**mapping).
    fetchers None -> fetchers.default_fetchers(config). clock returns an aware UTC datetime
    (default: datetime.now(timezone.utc)). Creates the data dir, starts the sweeper thread and executors.
    A static dir without index.html is not an error: the app starts, logs a WARNING and answers SPA routes and
    /favicon.ico with 503 text/plain 'trendai: static files are missing' (E.4)."""

class TrendaiApp:
    config: WebConfig
    def __call__(self, environ: dict, start_response: Callable) -> Iterable[bytes]: ...   # the WSGI callable
    def close(self, timeout: float = 30.0) -> None: ...   # drain: refuse new jobs, wait for running ones up to
                                                         # timeout, cancel queued, stop sweeper; idempotent

# trendai/web/server.py
class ThreadingWSGIServer(socketserver.ThreadingMixIn, wsgiref.simple_server.WSGIServer):
    daemon_threads = True
def make_server(app: TrendaiApp, host: str = "127.0.0.1", port: int = 8000) -> ThreadingWSGIServer: ...
def serve_forever(app: TrendaiApp, host: str, port: int, *,
                  on_ready: Optional[Callable[[str], None]] = None,
                  install_signal_handlers: bool = True) -> int:
    """Blocks. Calls on_ready(base_url) once listening (port 0 -> the real port in base_url).
    SIGINT/SIGTERM -> graceful shutdown (E.14). Returns the exit code (0); the caller then exits with
    os._exit(code) after flushing stdout/stderr/logging (so a still-running job thread cannot hold the process)."""
def start_background(app: TrendaiApp, host: str = "127.0.0.1",
                     port: int = 0) -> tuple[ThreadingWSGIServer, threading.Thread, str]:
    """For tests/E2E: serves in a daemon thread, returns (server, thread, base_url)."""

# trendai/web/wsgi.py   (for gunicorn/waitress)
def application(environ, start_response):
    """Lazy: importing the module creates nothing (no data dir, no threads). The first call, under a module lock,
    runs install_warning_baseline(), WebConfig.from_env().validate_public(), logging.basicConfig (E.11) and
    create_app(), then delegates. docs/WEB.md: never use gunicorn --preload (threads do not survive fork)."""

# trendai/web/fetchers.py
@dataclass
class Fetchers:
    naver: Optional[Callable[..., Any]]        # (groups, start, end, time_unit, device, gender, ages) -> NaverResult-like;
                                               # None = Naver not configured
    naver_endpoint: Optional[str]              # 'hub' | 'legacy' | None (shown in /api/config)
    wiki: Callable[..., pd.Series]             # (title, lang, start, end, granularity) -> pd.Series
    llm: Optional[Callable[[dict, str], str]]  # (llm_payload, lang) -> markdown; None = AI unavailable
def default_fetchers(config: WebConfig) -> Fetchers: ...

# trendai/web/testing.py   (used by tests/test_web_*.py and tests/e2e/server.py)
class TestResponse:
    status: int; headers: dict[str, str]  # lower-case keys
    body: bytes
    def json(self) -> Any: ...
    @property
    def text(self) -> str: ...
class TestClient:
    def __init__(self, app: TrendaiApp, *, remote_addr: str = "127.0.0.1") -> None: ...
    def request(self, method: str, path: str, *, json: Any = None, data: Optional[bytes] = None,
                headers: Optional[Mapping[str, str]] = None) -> TestResponse: ...   # path may carry a query string
    def get(self, path, **kw) -> TestResponse: ...
    def post(self, path, **kw) -> TestResponse: ...
    def delete(self, path, **kw) -> TestResponse: ...
    def upload(self, filename: str, content: bytes, *, field: str = "file") -> TestResponse: ...  # multipart POST /api/upload
    def wait_job(self, job_id: str, *, timeout: float = 60.0, hl: str = "ko") -> dict: ...  # polls until terminal
class FakeNaver: ...                       # behaviour in H.4.2; .calls: list[dict]; .endpoint = 'hub'
class FakeWiki: ...                        # behaviour in H.4.2; .calls: list[dict]
def fake_llm(payload: dict, lang: str) -> str: ...
class FrozenClock:                         # callable returning an aware UTC datetime; .advance(seconds)
    def __init__(self, start: datetime.datetime) -> None: ...
def write_static_fixture(directory: "str | os.PathLike") -> pathlib.Path:
    """Writes a minimal static tree for WP1's tests (index.html, js/app.js, css/tokens.css, favicon.svg,
    fonts/pretendard-1.3.9/x.woff2, dev/mock.html) and returns the directory. WP1 tests never depend on WP3's files."""
def make_test_app(data_dir: "str | os.PathLike", *, naver: bool = True, ai: bool = True,
                  clock: Optional[Callable[[], datetime.datetime]] = None,
                  static_dir: "str | os.PathLike | None" = None, **config_overrides) -> TrendaiApp:
    """Hermetic: WebConfig.from_env(environ={}, data_dir=..., static_dir=..., **config_overrides) -- the developer's
    TRENDAI_*, NAVER_* and ANTHROPIC_API_KEY variables never leak in; fetchers are always the fakes (naver=False ->
    fetchers.naver None; ai=False -> fetchers.llm None). static_dir None -> the package static dir (E2E)."""
```

#### B.3.3 WP1 ↔ WP3

* The HTTP API (section C) and the VM (section D) are the whole contract. WP3 never parses Python output.
* WP1 serves `<static_dir>/index.html` (default `trendai/web/static/index.html`) **byte for byte** (no templating)
  for every SPA route (E.4). `index.html` references only `/static/...` URLs.
* WP3 must not rely on any response field not in C/D. WP1 must not rename fields without updating this spec.
* **Machine-readable VM contract**: WP1 writes `trendai/web/vm.schema.json` (JSON Schema subset: `type` incl. type
  arrays for nullable fields, `required`, `properties`, `additionalProperties: false`, `items`, `enum`,
  `minItems`/`maxItems`, `$defs`/`$ref`) describing the series VM (D.2), compare VM (D.4), the C.6 job snapshot and
  the C.7 envelope. WP1's tests validate every real VM and response against it with a small stdlib validator in
  `testing.py`; WP3's `tests/js/schema.test.mjs` validates every file in `static/dev/fixtures/` against it with an
  equivalent ~100-line validator. Until WP1 lands, WP3 writes fixtures from the D examples; the merge of WP3 (after
  WP1, B.4) requires the fixture test to pass.

#### B.3.4 Shared constants (one value everywhere)

| Constant | Value | Used by |
|---|---|---|
| Max series per compare (search) | 5 | WP1 validation, WP3 chips |
| Max keywords per Naver group | 5 | WP2, WP1, WP3 chip editor |
| Max value columns per upload | 20 | WP1, WP3 confirm step |
| Upload table limits | header ≤ 5,000 columns (read-time check); long table ≤ 50 columns; wide table ≤ 200 rows (series) and ≤ 5,000 date columns; ≤ 2,000,000 cells | WP1 (E.17), WP2 (`max_columns`) |
| Concurrent upload parses | 2 (more → `503 server_busy`) | WP1 |
| Upload size | 10 MB (10 × 1024 × 1024 bytes) | WP1, WP3 (client pre-check) |
| JSON body size | 64 KB (65,536 bytes) | WP1 |
| Term length | 1–50 characters after trim | WP1, WP3 |
| Horizon choices | D 14/30/90 · W 13/26/52 · M 6/12/24 · Q 4/8 · Y 3/5 | WP1 (`/api/config`), WP3 |
| Horizon defaults | D 30 · W 26 · M 12 · Q 8 · Y 5 (`data.default_horizon`) | WP1 |
| Horizon maxima | D 365 · W 104 · M 36 · Q 16 · Y 10 | WP1 |
| Poll interval | server `poll_after_ms` (500) | WP1, WP3 |
| Result TTL / share TTL / upload cache TTL | 24 h / 30 d / 30 min | WP1, WP3 copy |

### B.4 Working in parallel

* **WP1** develops against local stubs of the B.3.1 names (not committed) and its own fakes (`testing.py`), so its
  tests run before WP2 lands — except `test_web_messages_present.py`, which asserts every code WP1 can emit exists in
  `messages.MESSAGES` and is expected to fail until WP2 is merged. WP1's tests never read WP3's files: they run
  against `testing.write_static_fixture(tmp_path / "static")` (`make_test_app(..., static_dir=...)`), so they are green
  at WP1's merge, before `trendai/web/static/` exists. The only test that reads the real static tree
  (`test_package_static_if_present`) skips while `trendai/web/static/index.html` is absent.
* **WP2** has no dependency on the others.
* **WP3** develops against the JSON examples of C/D saved as fixtures in `trendai/web/static/dev/fixtures/` and a
  dev page `trendai/web/static/dev/mock.html` that installs a `fetch` shim (`dev/mock.js`). `/static/dev/**` is served
  only when `TRENDAI_DEV=1` and is excluded from the wheel by WP4's `pyproject.toml` (E.4, H.5). Once WP1 is available,
  WP3 runs `TRENDAI_DEV=1 trendai serve` or `python -c "from trendai.web.testing import *; ..."` with fakes.
* **WP4** builds `cli serve`, packaging and docs from B.3.2 and E.1; E2E tests run once WP1+WP3 land.
* Merge order: WP2 → WP1 → WP3 → WP4. Each merge must keep `python -m pytest` green for what is merged.

### B.5 WP4 deliverables in detail

**`trendai serve`** (`trendai/cli.py`; imports `trendai.web` lazily inside the command):

```
trendai serve [--host HOST] [--port PORT] [--data-dir DIR] [--dev] [--lang ko|en]
```

| Option | Korean help text |
|---|---|
| (command) | 웹 앱 서버 실행 (기본 http://127.0.0.1:8000) |
| `--host` | 접속 주소 (기본: TRENDAI_HOST 또는 127.0.0.1) |
| `--port` | 포트 (기본: TRENDAI_PORT 또는 8000, 0이면 빈 포트를 고름) |
| `--data-dir` | 결과·캐시 저장 폴더 (기본: TRENDAI_DATA_DIR 또는 ./.trendai-data) |
| `--dev` | 개발 모드: 정적 파일을 매번 새로 읽고 /static/dev를 제공 |
| `--lang` | 안내 문구 언어 (ko, en) |

* Precedence: flags > environment > defaults (`WebConfig.from_env(**{k: v for flags given})`).
* Start-up, in this order (the command body runs inside `cli.main`'s `warnings.catch_warnings()` block, i.e. *after*
  `_install_warning_filters()`): `trendai.web.install_warning_baseline()` (overrides the CLI filters for this
  process; E.13); build the config and call `config.validate_public()` (missing contact on a public bind → exit 2
  with `오류: 공개 주소로 실행하려면 TRENDAI_CONTACT(운영자 이메일이나 URL)를 설정하세요. 비공개로 쓰려면
  TRENDAI_ALLOW_NO_CONTACT=1.` / `Error: set TRENDAI_CONTACT (operator e-mail or URL) to run on a public address, or
  TRENDAI_ALLOW_NO_CONTACT=1 for private use.`); `logging.basicConfig` (E.11); build the app; print to stdout with
  `flush=True` `trendai 웹 앱: {url} (Ctrl+C로 종료)` / `trendai web app: {url} (Ctrl+C to stop)`; to stderr, when
  `TRENDAI_CONTACT` is unset on loopback: `경고: TRENDAI_CONTACT가 설정되지 않아 Wikimedia 요청의 User-Agent에 연락처가
  없습니다. 공개 운영 전에 운영자 이메일이나 URL을 설정하세요.` / `Warning: TRENDAI_CONTACT is not set, so Wikimedia
  requests carry no contact in their User-Agent. Set an operator e-mail or URL before running publicly.`; when Naver
  keys are missing: `안내: 네이버 API 키가 없어 네이버 출처는 ‘준비 중’으로 표시됩니다.` / `Note: no Naver API keys, so the
  Naver source shows as not available.`; when `trust_proxy` is on without `public_url`: `경고: TRENDAI_TRUST_PROXY를 쓸
  때는 TRENDAI_PUBLIC_URL도 설정하는 것이 안전합니다.` / `Warning: set TRENDAI_PUBLIC_URL as well when using
  TRENDAI_TRUST_PROXY.`
* Runs `server.serve_forever(app, host, port, on_ready=print_url)`, then flushes and calls `os._exit(code)` (E.14).
  Exit codes: `0` after a graceful stop, `2` for an invalid configuration (`ValueError` → `오류: {message}` /
  `Error: {message}`), `1` when the port cannot be opened (`오류: {host}:{port}를 열 수 없습니다 — {reason}` /
  `Error: cannot listen on {host}:{port} — {reason}`) and after a second SIGINT/SIGTERM.

**`pyproject.toml`** (additions; `dependencies` unchanged):

```toml
[project.optional-dependencies]
llm = ["anthropic>=1.0"]
connectors = ["requests>=2.28", "pytrends>=4.9"]
xlsx = ["openpyxl>=3.1"]
web = ["openpyxl>=3.1", "anthropic>=1.0"]    # optional features of the web app (it runs without them)
dev = ["pytest>=7"]

[tool.setuptools.package-data]
"trendai.web" = ["static/**/*"]

[tool.setuptools.exclude-package-data]
"trendai.web" = ["static/dev/**"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q"
markers = ["e2e: browser end-to-end tests (need node and playwright)"]
```

**`Dockerfile`**:

```dockerfile
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    TRENDAI_HOST=0.0.0.0 TRENDAI_PORT=8000 TRENDAI_DATA_DIR=/data
# binds 0.0.0.0, so the container refuses to start without -e TRENDAI_CONTACT=... (or TRENDAI_ALLOW_NO_CONTACT=1)
WORKDIR /app
COPY pyproject.toml README.md ./
COPY trendai ./trendai
RUN pip install --no-cache-dir . openpyxl anthropic \
 && useradd --system --uid 10001 --home-dir /nonexistent --shell /usr/sbin/nologin trendai \
 && mkdir -p /data && chown trendai /data
USER trendai
VOLUME ["/data"]
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=3)"
CMD ["trendai", "serve"]
```

**`.dockerignore`**: `.git`, `.gitignore`, `.trendai-data`, `.pytest_cache`, `.ruff_cache`, `**/__pycache__`,
`*.egg-info`, `build`, `dist`, `out`, `reports`, `tests`, `docs`, `node_modules`, `read.py`, `trendai/web/static/dev`.

**`.gitignore`**: add `.trendai-data/` and `node_modules/`.

**`docs/WEB.md`** (Korean, operator guide) sections: 빠른 시작 (`pip install -e .` → `trendai serve`) · 환경 변수 (every
E.1 row) · 네이버 연결 (HUB 키 발급, 앱에서 검색어트렌드 켜기, 레거시 키는 2027-06-30까지, 월 50,000건과 예산 가드) ·
Wikimedia 연락처 (`TRENDAI_CONTACT` 필수 규칙, `TRENDAI_ALLOW_NO_CONTACT`, `TRENDAI_PUBLIC_URL`) · AI 해석
(`ANTHROPIC_API_KEY`, `TRENDAI_AI`, 한도, 90초 제한) · 배포 (단일 프로세스 규칙, `gunicorn -w 1 --threads 8
trendai.web.wsgi:application` — **`--preload` 금지**, `waitress-serve --threads=8 --inbuf-overflow=12582912
trendai.web.wsgi:application`, 리버스 프록시·TLS: nginx는 `client_max_body_size 11m;` + `proxy_request_buffering off;`
(아니면 업로드 본문이 nginx 임시 파일로 디스크에 쓰여 ‘원본 미저장’ 약속이 깨짐), `proxy_set_header Host $host;`,
`X-Forwarded-For`/`X-Forwarded-Proto`/`X-Forwarded-Host`, `TRENDAI_TRUST_PROXY` + `TRENDAI_PUBLIC_URL`, `TRENDAI_HSTS`,
Docker 실행 예 `-e TRENDAI_CONTACT=…`) · 처리량 (한 프로세스 기준 대략: 주간 5개 비교 분석이 GIL 시간 약 3.6초를 써서 분당
15–20건, 단일 시계열은 0.4–1.0초; `TRENDAI_WORKERS`를 늘려도 처리량은 거의 늘지 않고 긴 작업 뒤에 짧은 작업이 막히지 않게
해 줄 뿐; 같은 조회의 동시 요청은 한 작업으로 합쳐짐(C.5); 넘치면 `503 server_busy`) · 요청 한도와 NAT (IP당 한도는 통신사
CGNAT·사무실 NAT 뒤 사용자끼리 나눠 씀, 프록시 뒤에서 `TRENDAI_TRUST_PROXY`를 안 켜면 모든 사용자가 한 IP로 보임) · 데이터 보관
(폴더 구조, TTL, 용량 한도, 백업은 공유 링크 폴더만 의미 있음, `WEB_VM_SCHEMA`가 바뀌는 업그레이드는 기존 결과·공유 링크를
무효화함) · 보안 메모 (CSP, 로그 정책, 키 관리) · 정식 공개 전 법무 확인 항목 (brief §5.6, §7; 그리고: 네이버 비율 값을 화면과
API 응답에 *표시*하는 것은 허용, 파일 내려받기·복사는 불가라는 판단(A.3); 네이버 결과에 ‘기준 시점=100’ 변환을 막은 것이
충분한지; Google 트렌드 업로드 출처 표기) · 문제 해결 (프록시 뒤 TLS 오류 시 `SSL_CERT_FILE`, 포트 사용 중, 네이버 401, POST가
`403 cross_origin`이면 `TRENDAI_PUBLIC_URL` 확인) · 테스트 실행 (`python -m pytest`, E2E는 전역 playwright를 `NODE_PATH=$(npm root -g)`로 찾음, H.5).

**README** — new section `## 웹 앱` (≤ 15 lines): what it is, `trendai serve`, link to `docs/WEB.md`, the privacy one-liner
(앱은 원본 파일을 디스크에 쓰지 않음 — 프록시 설정은 docs/WEB.md, 결과 24시간), the Naver/AI notes.

**`docs/ARCHITECTURE.md`** — new section `## web/ — web app` summarising sections C–E of this spec (modules, job flow,
store, limits, notice capture with `data.collect_notices`) and the E.18 private imports added to "Cross-module private
dependencies"; mention the `on_stage` hook, `emit_notice` and `llm_narrative_from_payload` in the
pipeline/data/narrative sections.

---
## C. HTTP API contract

### C.0 General rules

* Base path `/api`. All bodies are UTF-8 JSON (`Content-Type: application/json; charset=utf-8`,
  `ensure_ascii=False`) except `/api/upload` (multipart or JSON) and `/api/export/*` (files).
* **Requests with a body** must send `Content-Type: application/json` (or `multipart/form-data` for `/api/upload`),
  else `415 unsupported_media_type`. Bodies need `Content-Length` (`411 length_required` otherwise) and are rejected
  with `413 body_too_large` **before reading** when `Content-Length` exceeds the limit (JSON 64 KB; upload 10 MB +
  64 KB multipart overhead; paste JSON 10 MB).
* **Origin check**: for `POST`/`DELETE`, if an `Origin` header is present and its `scheme://host[:port]` differs
  from the request's own origin → `403 cross_origin`. The own origin is, in this order: `TRENDAI_PUBLIC_URL` when set;
  else, with `trust_proxy`, the last `X-Forwarded-Proto` value + the last `X-Forwarded-Host` value (each falling back
  to `wsgi.url_scheme` / `HTTP_HOST` when absent); else `wsgi.url_scheme` + `HTTP_HOST`. Default ports (`:80` for
  http, `:443` for https) are dropped before comparing. There is no CORS: no `Access-Control-*` headers, `OPTIONS` → `405`.
* **Compression**: JSON responses larger than 1 KB are gzip-compressed (`gzip.compress(level=6)`) when the request's
  `Accept-Encoding` contains `gzip`; such responses carry `Content-Encoding: gzip` and `Vary: Accept-Encoding`.
  Static files and exports are never compressed (their ETags stay byte-exact).
* Unknown `/api/...` path → `404 not_found`; known path, wrong method → `405 method_not_allowed` with `Allow`.
* Every response carries `X-Request-Id: <8 hex>` and the security headers of E.12.
* Errors always use the envelope of C.12.1. Successful responses never contain an `error` key at the top level.
* Rate limits (C.13) answer `429` with `Retry-After: <seconds>` and also `retry_after` in the envelope.
* Path parameters `<id>`, `<share_id>` must match `^[A-Za-z0-9_-]{22}$`, `<k>` must match `^(0|[1-9][0-9]?)$`;
  otherwise `404 not_found` (never 400: do not reveal the format).
* `hl` query parameter on GETs selects the VM / message language (`ko` default).

### C.1 `GET /api/health` (also `HEAD`)

`200` (or `503` when the data dir is not writable or the app is draining):

```json
{"status": "ok", "version": "0.1.0", "uptime_s": 1234, "draining": false,
 "jobs": {"running": 1, "queued": 0}, "store": {"writable": true}}
```

`status` is `"ok"` | `"draining"` | `"degraded"` (store not writable). No secrets, no config values.
Headers: `Cache-Control: no-store`. Not rate limited.

### C.2 `GET /api/config`

```json
{
  "version": "0.1.0",
  "default_source": "naver",
  "sources": {
    "naver":  {"available": true, "reason": null, "endpoint": "hub", "budget_state": "ok"},
    "wiki":   {"available": true, "reason": null},
    "upload": {"available": true, "max_mb": 10, "max_series": 20,
               "formats": ["csv", "tsv", "txt", "xlsx", "json"]},
    "sample": {"available": true}
  },
  "features": {"xlsx_export": true, "ai": true, "ai_reason": null, "html_report": true},
  "limits": {
    "max_compare": 5, "max_keywords_per_group": 5, "max_term_length": 50,
    "horizon_choices": {"D": [14, 30, 90], "W": [13, 26, 52], "M": [6, 12, 24], "Q": [4, 8], "Y": [3, 5]},
    "horizon_default": {"D": 30, "W": 26, "M": 12, "Q": 8, "Y": 5},
    "horizon_max": {"D": 365, "W": 104, "M": 36, "Q": 16, "Y": 10}
  },
  "naver_ages": [
    {"code": "1", "ko": "0–12세", "en": "0–12"}, {"code": "2", "ko": "13–18세", "en": "13–18"},
    {"code": "3", "ko": "19–24세", "en": "19–24"}, {"code": "4", "ko": "25–29세", "en": "25–29"},
    {"code": "5", "ko": "30–34세", "en": "30–34"}, {"code": "6", "ko": "35–39세", "en": "35–39"},
    {"code": "7", "ko": "40–44세", "en": "40–44"}, {"code": "8", "ko": "45–49세", "en": "45–49"},
    {"code": "9", "ko": "50–54세", "en": "50–54"}, {"code": "10", "ko": "55–59세", "en": "55–59"},
    {"code": "11", "ko": "60세 이상", "en": "60+"}
  ],
  "periods": {
    "naver": {"earliest": "2016-01-01",
              "last_complete": {"date": "2026-11-24", "week": "2026-11-22", "month": "2026-10-31"}},
    "wiki":  {"earliest": "2016-01-01",
              "last_complete": {"date": "2026-11-24", "week": "2026-11-22", "month": "2026-10-31"}}
  },
  "retention": {"result_hours": 24, "share_days": 30, "upload_minutes": 30}
}
```

* `sources.naver.reason`: `null` | `"unconfigured"` | `"budget_exhausted"`; `available` is false for both.
  `budget_state`: `"ok"` (< 80 %), `"low"` (80–90 %), `"blocked"` (≥ 90 %); `endpoint` `null` when unconfigured.
* `default_source` = `"naver"` when Naver is available, else `"wiki"`.
* `formats` adds `"xls"` only when `xlrd` is importable, and drops `"xlsx"` when `openpyxl` is not.
* `features.ai_reason`: `null` | `"unconfigured"` (no `anthropic` SDK or no `ANTHROPIC_API_KEY`) | `"disabled"`
  (`TRENDAI_AI=off`). `xlsx_export` = `openpyxl` importable. `html_report` is always true (per-result availability
  is in the VM `exports`).
* `last_complete.*` = `period_bounds(..., today)[1]` per unit (Naver with the KST date, Wikipedia with the UTC date).
* Headers: `Cache-Control: no-cache`. Rate limit: `read` bucket.

### C.3 `GET /api/samples?hl=ko`

```json
{"samples": [
  {"name": "emerging", "title": "태동기형", "description": "낮은 기반에서 가속 성장 중인 신흥 트렌드 (S-커브 초입)",
   "expected_phase": "emerging", "freq": "M", "start": "2016-01-01", "end": "2025-12-01",
   "spark": [0.9, 1.0, …, 69.9], "synthetic": true},
  …  // 9 entries, in samples.list_samples() order
]}
```

* `title`/`description` in `hl`: Korean from `texts.SAMPLE_TITLES` / `samples.DESCRIPTIONS`, English from
  `texts.SAMPLE_TITLES_EN` / `texts.SAMPLE_DESCRIPTIONS_EN` (G.13). `spark` = the default sample (`n=120`, `M`)
  downsampled to exactly 60 points: the values at positions 1, 3, 5, …, 119 (0-based; the last value is therefore
  always included), rounded to 3 significant digits.
* Computed once at app start. `Cache-Control: public, max-age=3600`. Rate limit: `read`.

### C.4 `POST /api/upload`

Query: `?hl=ko|en` (language of `notices[].text` and labels; default `ko`). **Request A — file**: `multipart/form-data` with exactly one part `name="file"` that has a `filename`. Other parts
are ignored. **Request B — paste**: `application/json` `{"text": "<tab-separated text>", "filename": "붙여넣기.tsv"}`
(`filename` optional, default `"paste.tsv"`; `text` 1 byte–10 MB).

Processing (E.17): rate limit → parse slot (at most 2 concurrent parses, else `503 server_busy`, `Retry-After: 5`)
→ size checks → Excel zip guard → `data.read_table_bytes(raw, name, max_columns=5000)` → table limits →
`data.frame_to_series` with automatic detection → store the `RawTable` in the in-memory upload cache (30 min).
Notices of both calls are captured with `data.collect_notices()`. Nothing is written to disk by the app.

**`200`** (also when detection failed — the confirm step lets the user pick columns):

```json
{
  "upload_id": "Qm1vYXJ0ZmFrZWlkMTIzNA",
  "expires_at": "2026-11-25T00:42:03Z",
  "filename": "월매출.xlsx",
  "size_bytes": 18234,
  "format": "xlsx",
  "encoding": null,
  "kind": "table",
  "columns": [
    {"name": "날짜",   "role": "date",     "reason": null,         "numeric_share": 0.0},
    {"name": "매출",   "role": "value",    "reason": null,         "numeric_share": 1.0},
    {"name": "방문수", "role": "value",    "reason": null,         "numeric_share": 1.0},
    {"name": "번호",   "role": "excluded", "reason": "row_number", "numeric_share": 1.0},
    {"name": "메모",   "role": "ignored",  "reason": "not_numeric","numeric_share": 0.0}
  ],
  "detected": {
    "wide": false,
    "date_col": "날짜",
    "date_cols": ["날짜"],
    "date_format": "YYYY-MM",
    "date_format_label": "2024.01 형식 (월)",
    "value_cols": ["매출", "방문수"],
    "freq": "M",
    "freq_label": "월간",
    "n_rows": 94,
    "first_date": "2019-01-01",
    "last_date": "2026-10-01"
  },
  "problem": null,
  "preview": {
    "columns": ["날짜", "매출", "방문수", "번호", "메모"],
    "head": [["2019-01", "1,234,000", "320", "1", ""], …],      // first 5 rows, cells as strings ≤ 40 chars
    "tail": [["2026-10", "2,345,000", "512", "94", "행사"], …]   // last 5 rows
  },
  "notices": [
    {"code": "rownum_cols_skipped", "level": "notice", "series_index": null, "fixes": [],
     "text": "행 번호/일련번호로 보이는 열 ‘번호’는 분석에서 제외했어요. 포함하려면 확인 단계에서 값 열로 고르세요."}
  ],
  "google_trends": null
}
```

* `role`: `date` | `value` (detected value column) | `excluded` (numeric but skipped: `reason: "row_number"`) |
  `ignored` (`reason: "not_numeric"`: < 80 % of cells parse as numbers). `numeric_share` = parsed share of non-empty cells.
* `detected.freq` = `data.infer_freq` of the detected date column's parsed dates (non-null), or `null`;
  `freq_label` from G.5 (`일간`/`주간`/`월간`/`분기`/`연간`). `date_format_label` = `messages.format_label(label, hl)`
  plus the unit word (`형식 (월)` / `format (monthly)`).
* `detected.date_cols` = `attrs['trendai']['date_cols']` (B.3.1): every real column the dates come from, all with
  `role: "date"` in `columns` (`["연도", "월"]` for a year + month combination). `detected.date_col` =
  `date_cols[0]` — a real column name that may be posted back (`frame_to_series` re-detects the period column from
  it); never the display string `연도 + 월`. The client sends `date_col: null` in C.5 unless the user changed the
  date `<select>`, so detection is kept as is.
* `detected.value_cols`: all detected value columns, in table order. The confirm step pre-checks the **first 20**;
  C.5 with `value_cols: null` and more than 20 detected columns → `400 upload_too_many_series`.
* **Wide tables** (dates across columns): `detected.wide = true`, `detected.date_col = null`, `date_cols = []`,
  `detected.value_cols` = the series (row) names (unique, B.3.1), `columns` lists those names with `role: "value"`.
* **Table limits** (B.3.4), checked after `read_table_bytes` and detection: more than 2,000,000 cells
  (`frame.size`) → `422 upload_too_large_table`; a long table with more than 50 columns → `422 upload_too_many_columns`;
  a wide table with more than 200 rows or more than 5,000 date columns → `422 upload_too_many_columns` (`n` = the
  offending count, `max` = its limit). The column limit thus never rejects a wide table for having many periods.
* `preview` cells are WP1's stringification of `RawTable.frame` (E.17): Timestamps → `YYYY-MM-DD` (or
  `YYYY-MM-DD HH:MM` when the time is not midnight), floats → `format(v, ".15g")` with a trailing `.0` dropped,
  NaN/None → `""`, everything cut to 40 characters.
* `problem` (detection failed): the body of a C.12.1 error object — `{"code": "upload_no_date_col" | "upload_no_value_col",
  "status": 200, "message_ko", "message_en", "params": {}, "fixes": [{"action": "pick_columns", "label_ko": "열 고르기",
  "label_en": "Pick columns", "params": {}}], "retry_after": null, "request_id"}` — with `detected.date_col` /
  `value_cols` `null` / `[]`.
* `kind = "google_trends"` when `RawTable.google_trends` is set; then
  `"google_trends": {"terms": [{"col": "선크림", "term": "선크림", "region": "대한민국"}], "lt1_count": 3, "category": "모든 카테고리"}`.
* `notices`: captured `DataNotice`/`DataWarning` (C.12.5), texts via `messages.render_web_msg` in `hl`.

**Errors**: `413 body_too_large`, `413 upload_archive_too_large`, `415 unsupported_media_type`,
`400 bad_request` (no `file` part / empty text), `422 upload_unreadable`, `422 upload_too_many_columns`,
`422 upload_too_large_table`, `501 upload_excel_needs_pkg`, `429 rate_limited` (bucket `upload`),
`503 server_busy` (both parse slots taken).

### C.5 `POST /api/analyze`

Request (JSON, ≤ 64 KB):

```json
{
  "source": "naver",
  "series": [{"terms": ["선크림", "자외선차단제"]}, {"terms": ["패딩"]}],
  "range": "5y",
  "unit": "auto",
  "h": "auto",
  "dev": "", "gender": "", "ages": [],
  "wl": "ko",
  "upload_id": null, "date_col": null, "value_cols": null, "freq": "auto", "unit_label": null
}
```

| Field | Sources | Rule (violations → `400`, code in brackets) |
|---|---|---|
| `source` | all | `naver` \| `wiki` \| `sample` \| `upload` [`invalid_param`] |
| `series` | naver, wiki, sample | 1–5 items [`empty_query`, `too_many_series`]; each `terms` 1–5 strings after trim + NFC; empty strings dropped; a series left empty is dropped; duplicate terms in a series dropped; duplicate series (same term tuple) dropped; term length ≤ 50 [`term_too_long`], no control characters [`invalid_param`]. Naver: ≤ 5 terms [`too_many_keywords`]. Wiki: exactly 1 term [`wiki_group_unsupported`]. Sample: exactly 1 term ∈ `samples.list_samples()` [`unknown_sample`]. |
| `range` | naver, wiki | `1y`\|`3y`\|`5y`\|`all`\|`YYYY-MM-DD~YYYY-MM-DD`, default `5y` [`invalid_param`] |
| `unit` | naver, wiki | `auto`\|`date`\|`week`\|`month`; `auto` = `week` (naver) / `month` (wiki) |
| `h` | all | `"auto"` or an integer 1…`horizon_max[freq]` (freq from `unit`; for samples `M`). Upload with automatic freq: values above the maximum are clamped with notice `horizon_clamped`. |
| `dev` / `gender` / `ages` | naver | `""`\|`pc`\|`mo` · `""`\|`m`\|`f` · subset of `"1"`…`"11"` |
| `wl` | wiki | `^[a-z]{2,3}$`, default `ko` |
| `upload_id` | upload | an id in the upload cache, else `410 upload_expired` |
| `date_col` | upload | a column of the table, or `null` for wide tables / automatic detection |
| `value_cols` | upload | 1–20 column (or wide-row) names, or `null` = detected [`upload_too_many_series`] |
| `freq` | upload | `auto`\|`D`\|`W`\|`M`\|`Q`\|`Y` |
| `unit_label` | upload | ≤ 12 characters, shown on axes/tooltips, or `null` |

Fields that do not apply to the source are ignored. Display names are unique within a request (E.6.1). The Naver
`groupName` is the wire name `g1` … `g5` (series index + 1), never the display name; results are mapped back by
index (E.6), so display names of any length or content are safe.

**Pre-checks before queueing** (all synchronous): validation; rate limit `analyze`; result-cache lookup (E.10) →
`200 cached`; **in-flight lookup** — a queued or running analyze job with the same result-cache key → `202` with
that job's id and `"joined": true` (its subscriber count is incremented; no Naver token, budget or executor slot is
used); for Naver: configured (`503 naver_unconfigured`), cache lookup by `naver_cache_key`; on a miss: budget
(`503 naver_budget_exhausted`) and per-IP Naver bucket (`429 naver_rate_limited`); executor capacity (`503 server_busy`).
Uploads have no result-cache key and are never joined.

**`202`** (queued, or joined) or **`200`** (an identical result exists and is fresh — `cached: true`):

```json
{"job_id": "Zx8kP2mQ9rT4vW6yA1bC3d", "status": "queued", "cached": false, "joined": false, "queue_position": 0,
 "poll_url": "/api/jobs/Zx8kP2mQ9rT4vW6yA1bC3d", "result_url": "/api/results/Zx8kP2mQ9rT4vW6yA1bC3d"}
```

`status` is `"queued"`, `"running"` (joined) or (cached) `"done"`. `queue_position` = number of jobs ahead (0 = starts
now or already running).

### C.6 `GET /api/jobs/<id>?since=0&hl=ko` and `DELETE /api/jobs/<id>`

GET `200` while queued/running (analyze job):

```json
{
  "job_id": "Zx8kP2mQ9rT4vW6yA1bC3d",
  "kind": "analyze",
  "status": "running",
  "stage": "series",
  "series_total": 3,
  "series_done": 1,
  "current": {"index": 1, "name": "패딩", "step": "forecast"},
  "series": [ {"index": 0, "status": "done", "vm": { /* D.2 series VM in hl */ }} ],
  "failed": [],
  "notices": [],
  "compare": null,
  "view": "compare",
  "error": null,
  "result_url": null,
  "elapsed_ms": 1840,
  "poll_after_ms": 500
}
```

* `status`: `queued` | `running` | `done` | `failed` | `cancelled`.
* `stage`: `queued` → `data` (fetching sources, preparing series) → `series` (per series; `current.step` is
  `analyze` | `forecast` | `narrative`) → `finish` (ranking, comparison, exports, storing) → `done`.
  `pipeline.run`'s `on_stage` names map to `current.step` as `prepare`→`analyze`, `analyze`→`analyze`,
  `forecast`→`forecast`, `outlook`→`forecast`, `narrative`→`narrative` (via `JobContext.set_step`, E.5).
* `series` contains only finished (`done`) entries with `index >= since`; a failed series goes to
  `failed`: `[{"index": 2, "name": "없는문서", "error": {/* C.12.1 envelope body */}}]` (always the full list).
  Series finish in index order. The client passes `since` = 1 + the highest index it has seen **in either
  `series` or `failed`** (0 at first) and de-duplicates by `index` anyway.
* Progressive VMs are for rendering while the job runs. **On `done` the client fetches `result_url` once**
  (`GET /api/results/<id>?hl=`): that envelope is authoritative — final `slot`s (N > 5 assigns them only at the end,
  D.4.2), `exports`, `query` (incl. `period`), `expires_at` and the compare VM. The client takes slots from
  `compare.slots`, never from progressive VMs.
* `compare`: the D.4 compare VM once `status == "done"` and `view == "compare"`, else `null`.
* `view`: `"detail"` (1 series requested) or `"compare"` (2+).
* `error`: the C.12.1 error object when `status == "failed"` (e.g. `naver_auth`, `data_too_short`,
  `job_timeout`); `null` otherwise. A job fails when the data stage fails or **every** series fails.
* `result_url` set when `done`. `poll_after_ms`: 500 while queued/running, `null` when terminal.
* After a restart, or 10 min after completion, the job is no longer in memory: GET then synthesises
  `{"status": "done", …}` from the stored result (full `series` list, `compare`), or `404 job_not_found`.

Narrative job (`kind: "narrative"`) while queued or running:

```json
{"job_id": "…", "kind": "narrative", "status": "running", "error": null, "elapsed_ms": 2140,
 "poll_after_ms": 500, "narrative": null}
```

and when done:

```json
{"job_id": "…", "kind": "narrative", "status": "done", "error": null, "elapsed_ms": 9120, "poll_after_ms": null,
 "narrative": {"markdown": "**한눈에 보기** — …\n\n## 과거: 어떻게 여기까지 왔나\n- …",
               "caveats_markdown": "- …", "source": "llm", "badge": "AI 해석 · Claude",
               "model": "claude-opus-5-5", "note": "AI가 작성한 해석이에요. 수치는 위 분석 결과를 근거로 해요."}}
```

`DELETE /api/jobs/<id>` decrements the job's subscriber count (C.5 joins increment it; the creator counts as 1). Only
when it reaches 0 is the job cancelled: `202 {"job_id": "…", "status": "cancelling"}` for queued/running jobs
(queued jobs become `cancelled` immediately; running jobs at their next stage boundary). While other subscribers
remain the job keeps running and the answer is `202 {"job_id": "…", "status": "<current status>"}` — one tab's
cancel never cancels another tab's shared job. Terminal jobs → `200 {"job_id": "…", "status": "<terminal>"}`,
unknown ids → `404 job_not_found`. A cancelled analyze job stores nothing.

Rate limit: bucket `poll` (GET, keyed by client IP **and** job id); DELETE is not rate limited.
`Cache-Control: no-store`.

### C.7 `GET /api/results/<id>?hl=ko`

```json
{
  "result_id": "Zx8kP2mQ9rT4vW6yA1bC3d",
  "created_at": "2026-11-25T00:12:03Z",
  "expires_at": "2026-11-26T00:12:03Z",
  "readonly": false,
  "view": "compare",
  "query": {"source": "naver", "series": [{"terms": ["선크림", "자외선차단제"]}, {"terms": ["패딩"]}],
            "range": "5y", "unit": "week", "h": 26, "dev": "", "gender": "", "ages": [], "wl": "ko",
            "period": {"start": "2021-11-29", "end": "2026-11-22"}},
  "series": [ /* D.2 series VMs in hl, index order; failed series are absent */ ],
  "failed": [ {"index": 2, "name": "플리스", "error": { /* C.12.1 */ }} ],
  "compare": { /* D.4, or null for view "detail" */ },
  "notices": [ /* C.12.5, result-level */ ],
  "exports": { /* D.5 result-level export links */ }
}
```

`query` is the **normalised** query (resolved `unit`, integer `h`, sorted `ages`; `period` = the request bounds from
`connectors.period_bounds`, used by the client to write an absolute `range` when copying a link, F.6.12). For
uploads: `{"source": "upload", "filename": "월매출.xlsx", "date_col": "날짜", "value_cols": ["매출"], "freq": "M", "h": 12, "unit_label": "원"}`;
samples have no `period`.
Errors: `404 result_not_found` (unknown, expired, or stored with another `WEB_VM_SCHEMA`). `Cache-Control: no-store`.
Rate limit: `read`.

### C.8 `POST /api/share` and `DELETE /api/share/<share_id>`

POST `{"result_id": "…"}` → **`201`**:

```json
{"share_id": "pQ3rS5tU7vW9xY1zA2bC4d", "url": "/s/pQ3rS5tU7vW9xY1zA2bC4d",
 "expires_at": "2026-12-25T00:15:00Z", "delete_token": "kL8mN0oP2qR4sT6uV8wX0y"}
```

* Only results whose source kind is `upload` or `google_trends_upload`; others → `400 share_not_needed` (the page URL
  is the link). Unknown/expired result → `404 result_not_found`.
* Creates `shares/<share_id>/` with a snapshot of the result (VMs in both languages, export data, pre-rendered HTML;
  **no** LLM payloads) so the share outlives the 24 h result. Stored VMs never contain URLs or the `result_id`
  (E.7: `exports` are stored as availability flags and URLs are built per response), so the snapshot cannot leak the
  result's id — the result is a writable capability for 24 h (AI rewrites, more shares). The delete token is returned
  once; only its SHA-256 is stored.
* Rate limit: bucket `share`.

DELETE with header `X-Trendai-Delete-Token: <token>` → `204` (no body). Missing/wrong token → `403 share_forbidden`;
unknown/expired → `404 share_not_found`. Constant-time comparison (`hmac.compare_digest`).

### C.9 `GET /api/shared/<share_id>?hl=ko`

Same body as C.7 with `"readonly": true`, `"result_id": null`, export links under `/api/export/s/<share_id>…`, every
series VM with `narrative.ai_allowed = false` (`ai_block_reason: "shared_readonly"`), plus:

```json
"shared": {"share_id": "pQ3rS5tU7vW9xY1zA2bC4d", "created_at": "2026-11-25T00:15:00Z", "expires_at": "2026-12-25T00:15:00Z"}
```

The original `result_id` string never appears in any byte of this response (tested, H.1).
Errors: `404 share_not_found` (also for a snapshot stored with another `WEB_VM_SCHEMA`). Headers: `Cache-Control:
private, no-store`, `X-Robots-Tag: noindex, nofollow`.

### C.10 Exports

| Path | Content |
|---|---|
| `GET /api/export/<id>.<fmt>` | Whole result. `csv`: compare → ranking table; single → same as series 0. `xlsx`: conditions + one data sheet per series + summary sheet. `html`: `render_compare_html` page (compare) or `render_html` (single). `json`: array of `report_to_dict` (compare) or one object (single). |
| `GET /api/export/<id>/<k>.<fmt>` | One series: `csv`, `xlsx`, `html`, `json`. |
| `GET /api/export/<id>/values.csv` | Compare results only: the date × series values table of the compare table view (F.6.8): `날짜` + one column per successful series (display names, request order) with the history values on the union of dates (empty where a series has no value). **Naver: `403 export_naver_raw`** (the values are raw ratios). Single-series results → `404 not_found`. |
| `GET /api/export/s/<share_id>.<fmt>` · `/api/export/s/<share_id>/<k>.<fmt>` · `/api/export/s/<share_id>/values.csv` | The same from a share snapshot. |

`<fmt>` ∈ `csv` | `xlsx` | `html` | `json`; `?hl=ko|en` selects header/label language (HTML is always Korean).

**CSV (per series)** — UTF-8 **with BOM**, CRLF, comma separator, RFC 4180 quoting:

```
# 출처: 네이버 데이터랩 검색어트렌드(NAVER API HUB) · 주 단위 · 2021.11.29 ~ 2026.11.22 · 상대 지수(조회 기간 최댓값 = 100) · 데이터 기준 2026.11.25 09:12
# 조건: 검색어 그룹 선크림(선크림, 자외선차단제) · 모바일 · 여성 · 25–34세 · 예측 26주 · 네이버 원본 값(ratio)은 이용약관상 포함하지 않았어요
날짜,추세,예측값,80% 하한,80% 상한,95% 하한,95% 상한
2021-11-29,12.1,,,,,
…
2026-11-23,,63.4,58.1,68.7,55.2,71.6
```

* Columns: `날짜, 실제값, 추세, 예측값, 80% 하한, 80% 상한, 95% 하한, 95% 상한` (en: `date, actual, trend, forecast,
  lo80, hi80, lo95, hi95`). **Naver: no `실제값`/`actual` column** and the second comment line ends with the
  "원본 값 제외" note (G.8 `export.naver_note`). History rows fill `실제값`, `추세`; forecast rows fill the rest.
* Values: plain decimal numbers (`repr`-free, up to 15 significant digits, `.` decimal point, no thousands
  separators); dates `YYYY-MM-DD`. Comment lines start with `# ` (pandas `comment="#"` reads the file).
* **Formula-injection guard (CSV)**: any *text* cell (names, labels, comment lines' leading cell) that starts with
  `=`, `+`, `-`, `@`, tab or CR is prefixed with `'`. Numbers are never prefixed.
* Whole-result CSV (compare): ranking columns `순위, 이름, 국면, 전망, 예측 변화(%), 상승 확률(%), 최근 변화(%),
  모멘텀(%p/기간), 상승 점수, 예측 신뢰도` from `rank_trends`; the first comment line is `compare.source_line` (D.4),
  the second the condition line listing every series. `values.csv` uses the same two comment lines.

**XLSX** (only when `openpyxl` is importable, else `501 export_xlsx_unavailable`): sheet `데이터`/`Data` per series
(name = series display name cut to 31 characters, `[]:*?/\` replaced by `_`, de-duplicated with ` (2)`):
rows 1–7 = conditions (`출처`, `검색어 그룹/열`, `기간`, `단위`, `조건(기기·성별·연령)`, `데이터 기준`, `값의 성격`; label in
column A, value in B), row 8 empty, row 9 header (same columns as CSV), data from row 10; numbers as numbers
(`0.0` format for relative indexes). Sheet `요약`/`Summary`: one row per series with `이름, 국면, 전망, 예측 변화,
상승 확률, 예측 신뢰도, 정확도, 상승 점수` (pre-formatted texts from the VM). Same Naver rules as CSV. **Formula
guard (XLSX)**: openpyxl stores any string starting with `=` as a formula, and a leading `'` would show as a visible
character, so text cells are written as strings explicitly (`cell.value = text; cell.data_type = "s"`) — no
apostrophe is added in XLSX.

**HTML**: the file pre-rendered at job completion (`report.render_html` / `render_compare_html`; Korean only), with
one post-processing step because the frozen renderer prints only the name: right after the first `<body>` (the
renderer emits `<body>\n<main class="wrap">`), WP1 inserts `<p class="source">{html.escape(source.line ko)}</p>`
(compare: `compare.source_line`). **Naver results: `403 export_naver_raw`** (the report embeds raw ratios).
Response headers for HTML:
`Content-Security-Policy: sandbox; default-src 'none'; style-src 'unsafe-inline' https://fonts.googleapis.com; font-src https://fonts.gstatic.com; img-src data:`.

**JSON**: `report_to_dict(report, lang=hl)` stored at job time in both languages, plus a top-level `"source"` object
= the series VM's `source` block in that language (kind, label, line, value_kind, value_kind_label, freq, start, end,
as_of, filters_label, filename, attribution; compare: an array of such objects plus `"source_line"`). **Naver**: the
`history` key is removed, `metrics.last_value`, `metrics.peak_value`, `metrics.mean` and every `anomalies[].value`
are removed, and `"raw_excluded": true` is added (model outputs such as `trend`, `forecast`, `scenarios` stay, A.3).

All exports: `Content-Disposition: attachment; filename="trendai-<id8>-<k|all>.<fmt>"; filename*=UTF-8''<pct-encoded "trendai-<display name>.<fmt>">`,
`Cache-Control: private, no-store`, `X-Content-Type-Options: nosniff`. MIME: `text/csv; charset=utf-8`,
`application/vnd.openxmlformats-officedocument.spreadsheetml.sheet`, `text/html; charset=utf-8`,
`application/json; charset=utf-8`. Errors: `404 result_not_found` / `share_not_found` / `not_found` (bad `k`).
Rate limit: bucket `export`.

### C.11 `POST /api/narrative`

Request `{"result_id": "…", "series": 0, "lang": "ko"}` → `202 {"job_id": "…", "status": "queued", "poll_url": "/api/jobs/…"}`.

Checks in order: a `share_id` key in the body (any value) → `403 ai_readonly` (AI cannot be created from a shared
link; the SPA never offers it there, but the code path exists and is tested); `lang` must be exactly `"ko"` or `"en"`
(else `400 invalid_param`, `params.param = "lang"`); `result_id` must be a string matching the id pattern (else
`404 result_not_found`); result exists (`404 result_not_found`); `series` must be an `int` that is not a `bool` with
`0 ≤ series < number of series` and a successful series at that index (else `404 not_found`; `true`, `0.0`, `"0"`
are rejected); source kind is not `naver` (`403 ai_naver_blocked`), AI available (`503 ai_unavailable`), per-IP
bucket `ai` (`429 ai_rate_limited`), daily cap `TRENDAI_AI_DAILY_LIMIT` (`503 ai_unavailable` with
`params.reason = "daily_limit"`), narrative executor capacity (`503 server_busy`). If an AI narrative for
`(result, k, lang)` already exists, respond `200` with a job that is already `done`.

The job calls `fetchers.llm(llm_payload, lang)` (default: `narrative.llm_narrative_from_payload`), splits the
caveats section (D.2.9), stores it in the result (`ai/<k>-<lang>.json`) so later `GET /api/results` shows it under
`series[k].narrative.ai`, and finishes with the C.6 narrative body. Any `NarrativeUnavailable` or exception →
job `failed` with `502 ai_failed` (the rule-based narrative stays on screen). The default LLM fetcher passes an
explicit client timeout of 90 s (`anthropic.Anthropic(timeout=90.0, max_retries=1)`), and a narrative job that is
still running after 120 s is failed with `ai_failed` at its next check. Only the stored LLM payload (computed facts)
is sent; no raw upload bytes ever exist to send.

### C.12 Error and notice catalog

#### C.12.1 Error envelope

```json
{"error": {
  "code": "wiki_not_found",
  "status": 404,
  "message_ko": "위키백과에 ‘선크림’ 문서가 없어요. 정식 문서 제목인지 확인하거나 네이버로 바꿔 보세요.",
  "message_en": "Wikipedia has no article titled ‘선크림’. Check that it is the article's exact title, or switch to Naver.",
  "params": {"title": "선크림"},
  "fixes": [
    {"action": "wiki_title_help", "label_ko": "위키백과에서 제목 찾기", "label_en": "Search Wikipedia titles",
     "params": {"url": "https://ko.wikipedia.org/w/index.php?search=%EC%84%A0%ED%81%AC%EB%A6%BC"}},
    {"action": "use_naver", "label_ko": "네이버로 보기", "label_en": "View on Naver", "params": {}}
  ],
  "retry_after": null,
  "request_id": "a1b2c3d4"
}}
```

* `message_ko`/`message_en` = `messages.render_web(code, params, lang)` for web codes, and
  `messages.render_web_msg(exc, lang)` when the message comes from a `DataError`/`ConnectorError` that is shown as
  is (`params.reason` cases); `label_*` = `messages.fix_label(action, lang)`.
* `fixes` are the C.12.2 defaults adjusted to the context by WP1: `extend_range` is dropped for `upload` /
  `google_trends_upload` results (their period cannot change) and `reupload` is added instead; `use_naver` is
  dropped when Naver is unavailable; at most 3 fixes are sent.
* `params` holds only the template parameters listed in C.12.2 (user-supplied values such as a title are echoed;
  upstream details, exception texts, paths and keys never are).
* `retry_after` (seconds, int) for 429/503 codes that have one; mirrors the `Retry-After` header.
* `500 internal_error` is used for every unexpected exception (traceback logged with the request id, never returned).

#### C.12.2 Web error codes (WP2 adds to `MESSAGES`, WP1 raises; texts verbatim)

| code | HTTP | ko | en | fixes |
|---|---|---|---|---|
| `bad_request` | 400 | 요청 형식이 올바르지 않아요. 페이지를 새로고침해 주세요. | The request was malformed. Please reload the page. | reload |
| `invalid_param` | 400 | ‘{param}’ 값이 올바르지 않아요. | The value of ‘{param}’ is not valid. | go_home |
| `empty_query` | 400 | 분석할 키워드를 입력해 주세요. | Enter a keyword to analyse. | try_sample |
| `too_many_series` | 400 | 한 번에 {max}개까지 비교할 수 있어요. 하나를 빼고 추가해 주세요. | You can compare up to {max} at a time. Remove one first. | remove_series |
| `too_many_keywords` | 400 | 한 묶음에는 같은 뜻 검색어를 {max}개까지 넣을 수 있어요(‘{group}’). | A group can hold up to {max} terms (‘{group}’). | — |
| `term_too_long` | 400 | 검색어는 {max}자까지 입력할 수 있어요. | Terms can be at most {max} characters long. | — |
| `wiki_group_unsupported` | 400 | 위키백과는 ‘+’ 묶기를 지원하지 않아요. 문서 제목을 하나씩 입력해 주세요. | Wikipedia does not support ‘+’ groups. Enter one article title per item. | use_naver |
| `unknown_sample` | 400 | ‘{name}’ 샘플이 없어요. | There is no sample called ‘{name}’. | try_sample |
| `body_too_large` | 413 | 파일이 너무 커요. {max_mb} MB까지 올릴 수 있어요. | The file is too large. The limit is {max_mb} MB. | upload_file |
| `length_required` | 411 | 요청 길이를 알 수 없어요. 페이지를 새로고침해 주세요. | The request has no length. Please reload the page. | reload |
| `unsupported_media_type` | 415 | 지원하지 않는 요청 형식이에요. | This request format is not supported. | reload |
| `method_not_allowed` | 405 | 지원하지 않는 요청이에요. | This request method is not supported. | — |
| `cross_origin` | 403 | 다른 사이트에서 보낸 요청은 받을 수 없어요. | Requests from other sites are not accepted. | go_home |
| `not_found` | 404 | 페이지나 데이터를 찾을 수 없어요. | Not found. | go_home |
| `result_not_found` | 404 | 결과를 찾을 수 없어요. 보관 기간(24시간)이 지났거나 주소가 잘못됐어요. | This result was not found. It may have expired (results are kept for 24 hours) or the link is wrong. | reanalyze, go_home |
| `job_not_found` | 404 | 분석 작업을 찾을 수 없어요. 다시 분석해 주세요. | The analysis job was not found. Please run it again. | reanalyze |
| `share_not_found` | 404 | 공유 링크가 만료됐거나 삭제됐어요. | This shared link has expired or was deleted. | go_home |
| `share_forbidden` | 403 | 이 링크를 삭제할 권한이 없어요. 링크를 만든 브라우저에서만 삭제할 수 있어요. | You cannot delete this link. Only the browser that created it can. | — |
| `share_not_needed` | 400 | 공개 데이터 결과는 지금 주소가 곧 공유 링크예요. 주소를 복사해 주세요. | For public data the page address is the share link. Copy the address. | copy_link |
| `upload_expired` | 410 | 올린 파일의 보관 시간(30분)이 지났어요. 다시 올려 주세요. | The uploaded file is no longer available (kept for 30 minutes). Please upload it again. | reupload |
| `upload_unreadable` | 422 | 파일을 읽지 못했어요. CSV·TSV·엑셀(.xlsx)·JSON 파일을 올려 주세요. | The file could not be read. Upload a CSV, TSV, Excel (.xlsx) or JSON file. | show_formats, try_paste |
| `upload_too_many_columns` | 422 | 열이 너무 많아요({n}개). {max}개 이하의 열만 읽을 수 있어요. | Too many columns ({n}). Up to {max} columns can be read. | show_formats |
| `upload_too_large_table` | 422 | 표가 너무 커요(칸 {n}개). 칸이 {max}개 이하가 되도록 기간이나 열을 줄여 올려 주세요. | The table is too large ({n} cells). Reduce the period or columns to at most {max} cells. | show_formats |
| `upload_too_many_series` | 400 | 값 열은 {max}개까지 고를 수 있어요. | Choose up to {max} value columns. | pick_columns |
| `upload_archive_too_large` | 413 | 압축을 풀면 너무 큰 엑셀 파일이에요. CSV로 저장해 올려 주세요. | This Excel file is too large once unpacked. Save it as CSV and upload that. | show_formats |
| `upload_excel_needs_pkg` | 501 | 이 서버에서는 이 엑셀 형식을 읽을 수 없어요. CSV로 저장해 올려 주세요. | This server cannot read this Excel format. Save it as CSV and upload that. | show_formats |
| `upload_no_date_col` | (problem) | 날짜 열을 찾지 못했어요. 아래에서 날짜 열을 골라 주세요. | No date column was found. Pick the date column below. | pick_columns |
| `upload_no_value_col` | (problem) | 숫자 값 열을 찾지 못했어요. 아래에서 값 열을 골라 주세요. | No numeric value column was found. Pick the value columns below. | pick_columns |
| `upload_bad_date_col` | 400 | ‘{col}’ 열을 날짜로 읽을 수 없어요. 다른 열을 골라 주세요. | Column ‘{col}’ cannot be read as dates. Pick another column. | pick_columns |
| `upload_bad_value_col` | 400 | ‘{col}’ 열을 숫자로 읽을 수 없어요. | Column ‘{col}’ cannot be read as numbers. | pick_columns |
| `data_too_short` | 422 | 값이 {n}개뿐이라 분석할 수 없어요. 2개 이상 필요해요. | Only {n} value(s), so it cannot be analysed. At least 2 are needed. | extend_range, try_sample |
| `series_failed_web` | 422 | 이 시계열은 분석하지 못했어요. | This series could not be analysed. | reanalyze |
| `rate_limited` | 429 | 요청이 너무 많아요. {retry_after}초 뒤에 다시 시도해 주세요. | Too many requests. Try again in {retry_after} s. | retry_in |
| `server_busy` | 503 | 지금 분석 요청이 몰려 있어요. 잠시 뒤 다시 시도해 주세요. | The server is busy with other analyses. Please try again shortly. | retry_in |
| `internal_error` | 500 | 예기치 못한 오류가 났어요. 잠시 뒤 다시 시도해 주세요. | Something went wrong. Please try again shortly. | retry, go_home |
| `job_timeout` | 504 | 분석이 너무 오래 걸려 멈췄어요. 기간이나 비교 개수를 줄여 보세요. | The analysis took too long and was stopped. Try a shorter period or fewer series. | reanalyze |
| `naver_unconfigured` | 503 | 네이버 검색어 트렌드는 아직 연결되지 않았어요. | Naver search trends are not connected yet. | use_wiki, upload_file |
| `naver_auth` | 502 | 네이버 연결에 문제가 생겼어요. 잠시 뒤 다시 시도하거나 다른 출처를 써 주세요. | There is a problem with the Naver connection. Try again later or use another source. | retry, use_wiki |
| `naver_rate` | 429 | 요청이 몰려 잠시 막혔어요. {retry_after}초 뒤 자동으로 다시 시도해요. | Naver is busy right now. Retrying automatically in {retry_after} s. | retry_in (automatic **once** per query; after that the button only) |
| `naver_rate_limited` | 429 | 네이버 새 조회는 한 시간에 {limit}번까지 할 수 있어요. 잠시 뒤 다시 시도하거나 위키백과로 보세요. | New Naver lookups are limited to {limit} per hour. Try again later or view Wikipedia data. | use_wiki, retry_in |
| `naver_budget_exhausted` | 503 | 이번 달 네이버 조회 한도를 모두 썼어요. 이미 조회된 검색어는 저장된 결과로 볼 수 있어요. | This month's Naver lookup quota is used up. Terms looked up before can still be viewed from the cache. | use_wiki, upload_file |
| `naver_server` | 502 | 네이버 서버에서 오류가 났어요. | The Naver server returned an error. | retry |
| `source_timeout` | 504 | 데이터를 받아오지 못했어요(20초 초과). | The data could not be fetched (timed out after 20 s). | retry |
| `source_unreachable` | 502 | 데이터 출처에 연결하지 못했어요. 잠시 뒤 다시 시도해 주세요. | Could not reach the data source. Please try again shortly. | retry |
| `source_bad_response` | 502 | 데이터 출처가 알 수 없는 응답을 보냈어요. | The data source sent an unexpected response. | retry |
| `wiki_not_found` | 404 | 위키백과에 ‘{title}’ 문서가 없어요. 정식 문서 제목인지 확인하거나 네이버로 바꿔 보세요. | Wikipedia has no article titled ‘{title}’. Check that it is the article's exact title, or switch to Naver. | wiki_title_help, use_naver |
| `wiki_rate` | 429 | 위키백과 조회가 잠시 막혔어요. 잠시 뒤 다시 시도해 주세요. | Wikipedia lookups are blocked for a moment. Please try again shortly. | retry_in |
| `wiki_empty` | 404 | 이 기간에는 ‘{title}’ 조회수 데이터가 없어요. | There are no page views of ‘{title}’ in this period. | extend_range |
| `export_xlsx_unavailable` | 501 | 이 서버에서는 엑셀 내보내기를 쓸 수 없어요. CSV로 받아 주세요. | Excel export is not available on this server. Download CSV instead. | export_csv |
| `export_naver_raw` | 403 | 네이버 원본 데이터는 이용약관상 내려받기를 제공하지 않아요. 원본은 데이터랩(datalab.naver.com)에서 직접 엑셀로 받을 수 있어요. | Raw Naver data cannot be downloaded under its terms of use. You can download it yourself from DataLab (datalab.naver.com). | export_csv |
| `ai_unavailable` | 503 | 지금은 AI 해석을 쓸 수 없어요. | AI interpretation is not available right now. | — |
| `ai_naver_blocked` | 403 | 네이버 데이터는 이용약관에 따라 AI로 보내지 않아요. | Naver data is not sent to AI, following its terms of use. | — |
| `ai_readonly` | 403 | 공유 링크에서는 AI 해석을 새로 만들 수 없어요. | AI interpretations cannot be created on a shared link. | — |
| `ai_failed` | 502 | AI 해석을 만들지 못해 규칙 기반 해석을 보여 드려요. | The AI interpretation could not be created, so the rule-based one is shown. | retry |
| `ai_rate_limited` | 429 | AI 해석은 한 시간에 {limit}번까지 만들 수 있어요. | AI interpretations are limited to {limit} per hour. | retry_in |

#### C.12.3 Connector codes (WP2 adds; library/CLI texts) and their web mapping (WP1)

| ConnectorError code (params) | ko (library text) | en | → web code |
|---|---|---|---|
| `conn_naver_unconfigured` | 네이버 API 키가 설정되지 않았습니다. NAVER_API_HUB_KEY_ID와 NAVER_API_HUB_KEY(또는 NAVER_CLIENT_ID와 NAVER_CLIENT_SECRET)를 설정하세요. | Naver API keys are not set. Set NAVER_API_HUB_KEY_ID and NAVER_API_HUB_KEY (or NAVER_CLIENT_ID and NAVER_CLIENT_SECRET). | `naver_unconfigured` |
| `conn_naver_auth` (`status`) | 네이버 API 인증에 실패했습니다 (HTTP {status}). 키가 맞는지, 앱에서 검색어트렌드 API를 켰는지 확인하세요. | Naver API authentication failed (HTTP {status}). Check the keys and that the Search Trend API is enabled for the app. | `naver_auth` |
| `conn_naver_rate` | 네이버 API 요청 한도를 초과했습니다 (HTTP 429). 잠시 후 다시 시도하세요. | Naver API rate limit exceeded (HTTP 429). Try again in a moment. | `naver_rate` (`retry_after` 30) |
| `conn_naver_server` (`status`) | 네이버 API 서버 오류입니다 (HTTP {status}). | Naver API server error (HTTP {status}). | `naver_server` |
| `conn_naver_http` (`status`) | 네이버 API 오류입니다 (HTTP {status}). | Naver API error (HTTP {status}). | `source_bad_response` |
| `conn_naver_bad_response` | 네이버 API 응답을 해석할 수 없습니다. | The Naver API response could not be read. | `source_bad_response` |
| `conn_naver_network` (`error`) | 네이버 API에 연결하지 못했습니다. (자세한 오류: {error}) | Could not connect to the Naver API. (Details: {error}) | `source_unreachable` |
| `conn_naver_timeout` | 네이버 API 응답이 20초 안에 오지 않았습니다. | The Naver API did not answer within 20 s. | `source_timeout` |
| `conn_naver_too_many_groups` (`max`) | 네이버 검색어 그룹은 {max}개까지 요청할 수 있습니다. | At most {max} Naver keyword groups can be requested. | `too_many_series` |
| `conn_naver_too_many_keywords` (`max`, `group`) | 그룹 '{group}'의 검색어는 {max}개까지입니다. | Group '{group}' can hold at most {max} keywords. | `too_many_keywords` |
| `conn_naver_bad_filter` (`what`, `value`) | 네이버 조회 조건 {what} 값이 올바르지 않습니다: {value} | Invalid Naver filter {what}: {value} | `invalid_param` (`param` = what); `what = "groupName"` (empty, > 100 chars or duplicate names) cannot happen in the web app (wire names, C.5) and maps to `internal_error` + an ERROR log |
| `conn_bad_range` (`value`) | 조회 기간이 올바르지 않습니다: {value} | Invalid period: {value} | `invalid_param` (`param` = "range") |
| `conn_timeout` | Wikimedia API 응답이 20초 안에 오지 않았습니다. | The Wikimedia API did not answer within 20 s. | `source_timeout` |
| `conn_wiki_404` | (existing) | | `wiki_not_found` (`title`) |
| `conn_wiki_429` | (existing) | | `wiki_rate` (`retry_after` 60) |
| `conn_wiki_empty` | (existing) | | `wiki_empty` (`title`) |
| `conn_network` | (existing) | | `source_unreachable` |
| `conn_wiki_http`, `conn_wiki_error`, `conn_wiki_json`, `conn_wiki_no_valid` | (existing) | | `source_bad_response` |
| `conn_no_article`, `conn_no_keyword` | (existing) | | `empty_query` |
| `conn_bad_lang` | (existing) | | `invalid_param` (`param` = "wl") |
| `conn_start_after_end`, `conn_date_*` | (existing) | | `invalid_param` (`param` = "range") |

Any other `ConnectorError` → `source_bad_response`. WP1 logs `exc.code`, `exc.log_detail` (if any) at WARNING.

#### C.12.4 DataError mapping (WP1)

| DataError code(s) | Context | → web code |
|---|---|---|
| `no_date_col` | upload detection | `problem: upload_no_date_col` (200) |
| `no_numeric_col`, `only_rownum_cols` | upload detection | `problem: upload_no_value_col` (200) |
| `date_col_missing`, `date_col_unreadable` | analyze with user `date_col` | `upload_bad_date_col` (`col`) |
| `value_col_missing`, `value_col_unreadable` | analyze with user `value_cols` | `upload_bad_value_col` (`col`) |
| `file_empty`, `csv_ragged`, `json_*`, `excel_is_html`, `excel_unreadable*`, `excel_wrong_ext`, `file_unreadable`, `table_empty_*` | upload parsing | `upload_unreadable` (`params.reason` = original code) |
| `table_too_many_columns` (new, WP2: `표의 열이 너무 많습니다({n}개, 최대 {max}개).` / `The table has too many columns ({n}, at most {max}).`) | upload parsing (`max_columns`) | `upload_too_many_columns` (`n`, `max`) |
| `excel_needs_pkg` | upload parsing | `upload_excel_needs_pkg` |
| `too_few_obs`, `too_few_after_regularise`, `series_empty`, `no_valid_values` | per series | `data_too_short` (`n`) |
| `horizon_*` | per series | `invalid_param` (`param` = "h") |
| anything else | per series | `series_failed_web` |

#### C.12.5 Notices

Notice object (in VMs, job/result `notices`, upload `notices`):

```json
{"code": "gaps_interpolated", "level": "warning", "text": "…", "series_index": 0, "fixes": []}
```

`level`: `"notice"` (DataNotice / interpretation), `"warning"` (DataWarning / data changed), `"info"` (web hints).
`text` in the VM's language via `messages.render_web_msg(captured.args[0], lang)` — every part of a joined message
(e.g. `date_order_ambiguous` + `dates_unreadable_rows`, data.py's `join(...)`) is rendered; `code` = the first part's
code. Notices WP1 creates itself use `render_web(code, params, lang)`. `series_index` is `null` for result-level notices.

New notice codes (WP2 adds texts; WP1 emits):

| code (params) | level | ko | en | fixes |
|---|---|---|---|---|
| `naver_missing_filled` (`n`) | warning | 네이버가 값을 주지 않은 기간 {n}개는 0으로 처리했어요. 검색량이 매우 적은 기간이에요. | {n} period(s) without a value from Naver were treated as 0 (very low search volume). | — |
| `naver_low_resolution` | warning | 다른 검색어보다 검색량이 매우 적어 해상도가 낮아요. | Far fewer searches than the other terms, so the resolution is low. | reanalyze_single |
| `mostly_zero_volume` | warning | 이 기간에는 검색량이 너무 적어 대부분 0으로 나와요. 기간 단위를 ‘월’로 바꾸거나 같은 뜻 검색어를 묶어 보세요. | Search volume is so low that most values are 0. Switch the unit to months or group similar terms. | unit_month, group_terms |
| `data_short` (`n`) | warning | 값이 {n}개뿐이라 국면은 잠정 판단이고 예측 범위가 넓어요. 기간을 늘리면 더 정확해져요. | Only {n} values, so the phase is tentative and the forecast range is wide. A longer period helps. | extend_range, try_sample |
| `horizon_clamped` (`span`) | notice | 예측 기간을 최대치인 {span}(으)로 줄였어요. | The forecast horizon was reduced to the maximum, {span}. | — |
| `gt_detected` (`terms`, `region`) | notice | Google 트렌드 CSV로 읽었어요(검색어 {terms}, 지역 {region}). 값은 요청마다 0–100으로 다시 맞춰진 상대 지수예요. | Read as a Google Trends CSV (terms {terms}, region {region}). Values are a relative index rescaled to 0–100 per request. | — |
| `gt_lt1_half` (`n`) | notice | ‘<1’ 값 {n}개를 0.5로 읽었어요. Google 트렌드에서 ‘1 미만’이라는 뜻이에요. | {n} ‘<1’ value(s) were read as 0.5 (Google Trends' “less than 1”). | — |
| `index_unavailable` | info | 기준값이 0 이하인 시계열이 있어 ‘기준 시점=100’으로 바꿀 수 없어요. 원래 값으로 보여 드려요. | Some series have a base of 0 or less, so they cannot be indexed to 100. Showing original values. | — |
| `log_hint` (`ratio`) | info | 최댓값 차이가 {ratio}배를 넘어요. 로그 축으로 보면 작은 시계열도 잘 보여요. | The largest series is more than {ratio}× the smallest. A log axis shows the small ones better. | use_log |
| `naver_cached` (`fetched`) | info | 저장된 네이버 조회 결과({fetched} 기준)를 보여 드려요. | Showing stored Naver data (as of {fetched}). | — |

Existing data-layer codes (`values_unreadable`, `values_infinite`, `gaps_interpolated`, `weekday_only`,
`duplicates_merged`, `upsampled`, `rownum_cols_skipped`, `headerless`, `utf8_broken`, `excel_is_text`,
`wide_table`, `date_order_ambiguous`, `date_ym_ambiguous`, `dates_unreadable_rows`, `col_values_unreadable`) are
passed through with `render_web_msg` (DataNotice → `notice`, DataWarning → `warning`). `horizon_long` is created by
WP1 itself (E.6 step 2); the engine's own `horizon_long` from `pipeline.run` is discarded by code (E.6 step 3).
`span` in `horizon_clamped` is the D.1.6 span word (`104주` / `104 weeks`), so English plurals are right.

#### C.12.6 Fix actions (`fix_<action>` codes in `MESSAGES`)

| action | ko | en | client behaviour (WP3) | params |
|---|---|---|---|---|
| `reload` | 새로고침 | Reload | `location.reload()` | |
| `retry` | 다시 시도 | Try again | repeat the failed request | |
| `retry_in` | 지금 다시 | Retry now | show a countdown from `retry_after`, retry at 0 on click; automatic retry at 0 only for `naver_rate` and only **once** per query (each retry may reserve Naver budget) | `seconds` |
| `go_home` | 처음으로 | Home | navigate `/` | |
| `reanalyze` | 다시 분석 | Analyse again | re-POST the current query | |
| `use_wiki` | 위키백과로 보기 | View on Wikipedia | navigate `/?src=wiki&q=<first term of each series>`: home with the Wikipedia source selected and the search box **pre-filled, not run**, plus the hint `search.hint.wiki_title` — keyword terms are rarely article titles | |
| `use_naver` | 네이버로 보기 | View on Naver | same URL with `src=naver` (dropped by WP1 when Naver is unavailable) | |
| `upload_file` | 파일 올리기 | Upload a file | navigate `/upload` | |
| `reupload` | 다시 올리기 | Upload again | navigate `/upload` | |
| `try_sample` | 샘플로 체험 | Try a sample | navigate `/samples` | |
| `extend_range` | 기간 늘리기 | Use a longer period | set `range=all` (never offered for upload results, C.12.1) | |
| `unit_month` | 월 단위로 | Switch to months | set `unit=month` | |
| `group_terms` | 같은 뜻 묶기 | Group similar terms | open the chip editor of the first series | |
| `remove_series` | 하나 빼기 | Remove one | focus the chips | |
| `show_formats` | 지원 형식 보기 | Supported formats | navigate `/upload#formats` | |
| `try_paste` | 붙여넣기로 시도 | Try pasting | open the paste dialog | |
| `pick_columns` | 열 고르기 | Pick columns | focus the confirm step's date dropdown | |
| `wiki_title_help` | 위키백과에서 제목 찾기 | Search Wikipedia titles | open `params.url` in a new tab (`rel="noopener noreferrer"`) | `url` |
| `use_log` | 로그 축으로 | Use log scale | set `scale=log` | |
| `copy_link` | 주소 복사 | Copy link | copy `location.href` | |
| `export_csv` | CSV로 받기 | Download CSV | download the CSV export | |
| `reanalyze_single` | 이 검색어만 다시 분석 | Analyse this term alone | navigate `/explore` with every analysis-relevant param of the current state kept (`src range unit h dev gender ages wl`, and `hl`) and only `q` replaced by that series, so segment and period stay the same | `term` (encoded series) |

#### C.12.7 `WEB_OVERRIDES` (WP2)

Every existing `MESSAGES` template whose ko or en text contains `--`, `파이썬`, `Python` or `trendai.` gets a web
wording without CLI/Python hints. Required at least:

| code | ko (web) | en (web) |
|---|---|---|
| `weekday_only` | {name}평일(영업일)에만 값이 있어, 주말 {filled}개를 선형 보간해 달력일 기준({n_out}개)으로 분석했어요. 주 단위로 보려면 단위를 ‘주’로 바꾸세요. | {name}values exist on weekdays only, so {filled} weekend days were interpolated and the data is analysed on calendar days ({n_out}). Switch the unit to weeks for a weekly view. |
| `duplicates_merged` | {name}같은 {unit} 단위 기간에 값이 여러 개 있어 {how} 합쳤어요 (관측 {n_raw}개 → {observed}개 기간). | {name}several values fell in the same {unit} period and were {how} ({n_raw} observations → {observed} periods). |
| `rownum_cols_skipped` | 행 번호/일련번호로 보이는 열 {cols}은(는) 분석에서 제외했어요. 포함하려면 확인 단계에서 값 열로 고르세요. (rendered: `열 ‘번호’는 …` — list param and particle resolved by `render_web_msg`) | The column(s) {cols} look like row numbers and were left out. Select them as value columns in the confirm step to include them. |
| `too_few_after_regularise` | {name}{unit} 단위로 정리하면 관측치가 {n}개뿐이에요(2개 이상 필요). 더 짧은 주기를 고르거나 데이터를 늘리세요. | {name}on a {unit} grid only {n} observation(s) remain (at least 2 are needed). Choose a shorter frequency or add data. |
| `no_date_col` | 날짜 열을 찾지 못했어요. 확인 단계에서 날짜 열을 골라 주세요. | No date column found. Pick it in the confirm step. |
| `no_numeric_col` | 숫자 값 열을 찾지 못했어요. 확인 단계에서 값 열을 골라 주세요. | No numeric value column found. Pick one in the confirm step. |
| `only_rownum_cols` | 분석할 숫자 값 열이 없어요(행 번호 열 {cols}만 있음). 확인 단계에서 값 열을 골라 주세요. | There is no numeric value column (only the row-number column(s) {cols}). Pick one in the confirm step. |
| `index_numeric`, `series_required`, `horizon_*` and any other matching code | (rewrite without the flag/Python sentence) | |

### C.13 Rate limits and the Naver budget

Token buckets per client IP (E.8). `burst` = capacity, `refill` = tokens per hour.

| Bucket | Endpoints | burst | refill/h | Error |
|---|---|---|---|---|
| `analyze` | `POST /api/analyze` | 10 | 60 | `429 rate_limited` |
| `upload` | `POST /api/upload` | 5 | 30 | `429 rate_limited` |
| `naver` | fresh Naver calls only (cache hits are free) | 5 | 20 (5 when budget ≥ 80 %) | `429 naver_rate_limited` (`limit`) |
| `ai` | `POST /api/narrative` | 3 | 10 | `429 ai_rate_limited` (`limit`) |
| `share` | `POST /api/share`, `DELETE /api/share/*` | 5 | 20 | `429 rate_limited` |
| `export` | `GET /api/export/*` | 30 | 300 | `429 rate_limited` |
| `poll` | `GET /api/jobs/*` — keyed by `(client IP, job id)`, so two tabs, an analyze plus an AI job, or several users behind one NAT never share a bucket | 60 | 21,600 (6/s; the client polls at ≤ 2/s) | `429 rate_limited` |
| `read` | `GET /api/results/*`, `/api/shared/*`, `/api/config`, `/api/samples` | 60 | 1,200 | `429 rate_limited` |

Global guards:

* **Naver monthly budget** (E.9): `TRENDAI_NAVER_MONTHLY_LIMIT` (50,000). Before each fresh call the counter for the
  current KST month is incremented (failed calls count). `count ≥ 0.8 × limit` → `budget_state "low"` and the
  `naver` bucket refill drops to 5/h; `count ≥ 0.9 × limit` → `"blocked"`: fresh calls refused with
  `503 naver_budget_exhausted`, cache hits still served, `/api/config` reports `available: false, reason: "budget_exhausted"`.
* **Analysis executor**: `TRENDAI_WORKERS` (2) running + `TRENDAI_MAX_QUEUED` (8) waiting; more → `503 server_busy`,
  `Retry-After: 10`. Identical queries in flight share one job (C.5) and identical Naver bodies share one fetch (E.6),
  so a burst of clicks on the same example chip costs one slot and one Naver call.
* **AI executor**: 2 running + 4 waiting; daily cap `TRENDAI_AI_DAILY_LIMIT` (200, KST day).
* **Upload parsing**: at most 2 concurrent parses (`threading.BoundedSemaphore(2)`, non-blocking acquire); more →
  `503 server_busy`, `Retry-After: 5`.
* **Upload cache**: 64 entries / 256 MB (estimated `frame.memory_usage(deep=True)`); oldest evicted first.
* Per-IP limits are shared by every user behind one address (carrier CGNAT, office NAT, or a proxy without
  `TRENDAI_TRUST_PROXY`); docs/WEB.md says so.

### C.14 Caching headers

| Response | `Cache-Control` | Other |
|---|---|---|
| `index.html` (all SPA routes) | `no-cache` | `ETag`; `X-Robots-Tag: noindex, nofollow` on `/r/*`, `/s/*` |
| `/static/fonts/**` | `public, max-age=31536000, immutable` | path is versioned (`pretendard-1.3.9`) |
| other `/static/**` | `no-cache` | strong `ETag` (sha1 of bytes) → `304` on `If-None-Match` |
| `/api/config` | `no-cache` | |
| `/api/samples` | `public, max-age=3600` | |
| `/api/health`, `/api/jobs/*`, `/api/results/*`, `/api/analyze`, `/api/upload`, `/api/narrative`, `/api/share*` | `no-store` | `Vary: Accept-Encoding` when gzip-compressed (C.0) |
| `/api/shared/*`, `/api/export/*` | `private, no-store` | |
| errors | `no-store` | |

---
## D. View model

The VM is built by `trendai/web/viewmodel.py` (WP1) from the `TrendReport` objects at job time, once per language,
and stored. The frontend renders it without recomputing anything except chart geometry, axis ticks, crosshair
read-outs, index/log transforms and table views (D.6). The machine-readable form of this section is
`trendai/web/vm.schema.json` (B.3.3); prose and schema must agree.

### D.1 Formatting rules (`trendai/web/fmt.py`, mirrored by `static/js/fmt.js`)

**One canonical rounding algorithm, implemented identically in both languages**: `roundHalfAway(x, d)` rounds the
**exact decimal value of the IEEE-754 double** `x` to `d` decimals, ties away from zero. Python:
`Decimal(x).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP)` (`Decimal(float)` is exact). JS: take the exact
expansion `Math.abs(x).toFixed(100)` (exact for the magnitudes formatted here, 1e-6 ≤ |x| < 1e21), round that
digit string at `d` decimals with the half-away rule, re-apply the sign. Never `Math.round`, `toFixed(d)`,
`toPrecision`, Python `round()` or `format(x, ".Nf")` for display rounding (they round exact binary ties
differently: `toFixed` up, Python half-to-even). Consequently **`fmt.py` does not call `numfmt.number_text`/`numfmt.sci`**;
it reimplements their *decimal-place rules* with `roundHalfAway` (decimals by magnitude of `|x|` before rounding:
≥ 1,000 → 0, ≥ 10 → 1, ≥ 0.1 → 2, 0 → `0`, smaller → 2 significant digits with trailing zeros stripped; ×10ⁿ below
1e-4 and from 1e12 — the ko compact rule uses 조 up to 1e16 instead). The minus sign is U+2212 `−`. Results differ
from the engine's own texts (narrative, HTML report, which use `number_text`) only at exact binary ties such as
`1234.5` or `71.25`.

#### D.1.1 `compact(x, lang)` — tiles, scenario boxes, axes

* `roundHalfAway(|x|, 0) < 10,000`: `full(x)` (so `9999.5` is *not* printed as `10,000` but goes to the unit branch).
* ko: unit 만 (1e4), 억 (1e8), 조 (1e12); en: K (1e3, only from 1e4), M (1e6), B (1e9). `v = |x| / unit`:
  if `round1(v) < 10` → one decimal with a trailing `.0` stripped; else if `round0(v) < 1,000` → `round0(v)`;
  else (ko 만 only: 1,000만–9,999만) → `v` rounded to 3 significant digits (`round0(v / 10) · 10`, e.g.
  `1,230만`) with thousands separators — at most 3 significant digits everywhere (brief §4.4); if any rounding reaches
  10,000 (ko) / 1,000 (en) move up one unit and repeat.
* ko `|x| ≥ 1e16`, en `|x| ≥ 1e12`: ×10ⁿ with 3 significant digits, trailing zeros stripped (`3.4×10¹⁶`), sign prepended.
* `None`/NaN → `—`.

#### D.1.2 `full(x, lang)` — tables, tooltips, anomaly texts

The `number_text` decimal rules with `roundHalfAway` (`123,456,789`, `45.2`), both languages. `None` → `—`.

#### D.1.3 Value-kind wrapper `value_text(x, kind, lang, compact=False)`

`relative_index` / `relative_index_gt`: always one fixed decimal (`71.3`, `100.0`), never compacted.
Otherwise `compact()` when `compact=True`, else `full()`. (Axis *ticks* of a relative index are not value texts: they
use the tick step's own decimals, i.e. integers `0 20 40 … 100`, F.7.1.)

#### D.1.4 Percentages

* `pct(p)`: decimals = 0 if `round1(|p|) ≥ 10` else 1; signed (`+18%`, `+4.1%`, `−4.1%`, `+1,365%`);
  `round1(|p|) == 0` → `0%`.
* `pct_arrow(p)`: same digits, `▲` for > 0, `▼` for < 0, no sign character (`▲12%`, `▼4.1%`); 0 → `0%`.
* `pct_abs(p)`: digits only, no sign (`18%`) — inside sentences that say 높을/낮을.
* `pp(p)`: signed; one fixed decimal when `round1(|p|) ≥ 0.1` (thousands separators from 1,000), `0` when
  `p == 0`, else 2 significant digits with trailing zeros stripped; + `%p` (ko) / ` pp` (en): `+0.4%p`, `−3.9%p`,
  `+12.3%p`, `+0.042%p`.

#### D.1.5 Probabilities and likelihood words

* `prob_text(p, lang)`: `p ≥ 0.975` → `95% 이상` / `95% or more`; `p ≤ 0.025` → `5% 이하` / `5% or less`;
  else `약 {round0(p·20)·5}%` / `about {…}%`. `None` → `—`.
* `prob_exact(p, lang)` = `narrative.format_prob(p, lang)` (`81%`, `>99%`, `<1%`) — tooltips only.
* `prob_labelled(p, lang)` = `상승 확률 {prob_text}` / `chance of rise {prob_text}` — the form used wherever the
  number stands next to a likelihood word (hero and ranking lines, D.2.3), so it can never be read as the chance of
  the word beside it.
* `likelihood(p, lang)` — the narrative's bins (`narrative._probability_line`) on the **raw** P, so the card and the
  narrative of the same series always use the same word (J), short forms:

| P(rise) | ko | en |
|---|---|---|
| ≥ 0.90 | 상승 가능성 매우 높음 | Rise very likely |
| ≥ 0.70 | 상승 쪽이 우세 | Rise more likely |
| ≥ 0.55 | 상승 쪽이 약간 우세 | Rise slightly more likely |
| > 0.45 | 상승·하락 엇비슷 | About even |
| > 0.30 | 하락 쪽이 약간 우세 | Fall slightly more likely |
| > 0.10 | 하락 쪽이 우세 | Fall more likely |
| else | 하락 가능성 매우 높음 | Fall very likely |

#### D.1.6 Dates (all from `YYYY-MM-DD`; KST for timestamps)

| Function | D | W | M | Q | Y |
|---|---|---|---|---|---|
| `date_label` ko | `2026.11.25.` | `2026.11.17 주` | `2026년 11월` | `2026년 4분기` | `2026년` |
| `date_label` en | `Nov 25, 2026` | `Week of Nov 17, 2026` | `Nov 2026` | `Q4 2026` | `2026` |
| `date_short` ko (lists) | `2026.11.25` | `2026.11.17` | `2026.11` | `2026 4분기` | `2026` |
| `date_short` en | `Nov 25, 2026` | `Nov 17, 2026` | `Nov 2026` | `Q4 2026` | `2026` |
| `range_label` ko | `2021.11.29 ~ 2026.11.16` | `2021.11.29 ~ 2026.11.22` | `2021.11 ~ 2026.10` | `2021년 1분기 ~ 2026년 3분기` | `2016 ~ 2025` |
| `range_label` en | `Nov 1, 2021 – Oct 26, 2026` | `Nov 29, 2021 – Nov 22, 2026` | `Nov 2021 – Oct 2026` | `Q1 2021 – Q3 2026` | `2016 – 2025` |

For **W**, `range_label(start, end)` prints `start` (the first bucket's Monday) and `end + 6 days` (the last
bucket's Sunday), because the data really runs to that day; the W example above has `end = 2026-11-16`. The same
rule applies wherever a weekly period end is shown (source line, CSV comment line, XLSX `기간`).

`datetime_label(utc_iso)`: ko `2026.11.25 09:12`, en `Nov 25, 2026 09:12 KST`. Month abbreviations
`Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec`. Span words: ko `26주`, `12개월`, `30일`, `8분기`, `5년`;
en `26 weeks`, `12 months`, `1 week`. Frequency labels: ko `일 단위`/`주 단위`/`월 단위`/`분기 단위`/`연 단위`, en
`daily`/`weekly`/`monthly`/`quarterly`/`yearly`; adjectives for the summary line ko `일간`/`주간`/`월간`/`분기`/`연간`.

Axis ticks (client only, F.7.1): span ≥ 3 years → yearly ticks `2022`; 6 months–3 years → month ticks every
1/2/3/6 months labelled ko `4월` / en `Apr`, with the first tick and every January as ko `2026.01` / en `Jan 2026`;
< 6 months → day/week ticks ko `11.25` / en `Nov 25`, first tick and year changes ko `2026.11.25` / en `Nov 25, 2026`.

#### D.1.7 Test vectors (WP1 `tests/test_web_fmt.py` and WP3 `tests/js/fmt.test.mjs` both assert every row)

| input | `compact` ko | `compact` en |
|---|---|---|
| 0 | `0` | `0` |
| 9876 | `9,876` | `9,876` |
| 45.23 | `45.2` | `45.2` |
| 1.204 | `1.20` | `1.20` |
| 0.0234 | `0.023` | `0.023` |
| 0.00005 | `5×10⁻⁵` | `5×10⁻⁵` |
| 12345 | `1.2만` | `12K` |
| 12500 | `1.3만` | `13K` |
| 99999 | `10만` | `100K` |
| 340000 | `34만` | `340K` |
| 3456789 | `346만` | `3.5M` |
| 12345678 | `1,230만` | `12M` |
| 98765432 | `9,880만` | `99M` |
| 99999999 | `1억` | `100M` |
| 99950000 | `1억` | `100M` |
| 340000000 | `3.4억` | `340M` |
| 999999999 | `10억` | `1B` |
| 1.2e12 | `1.2조` | `1.2×10¹²` |
| 3.4e16 | `3.4×10¹⁶` | `3.4×10¹⁶` |
| −12345 | `−1.2만` | `−12K` |
| **exact binary ties** (half away from zero) | | |
| 0.125 | `0.13` | `0.13` |
| 2.5 | `2.50` | `2.50` |
| 12.25 | `12.3` | `12.3` |
| 1234.5 | `1,235` | `1,235` |
| 9999.5 | `1만` | `10K` |

| call | ko | en |
|---|---|---|
| `full(123456789)` · `full(1234.5)` · `full(0.125)` | `123,456,789` · `1,235` · `0.13` | same |
| `value_text(71.34, "relative_index")` · `value_text(71.25, "relative_index")` | `71.3` · `71.3` | same |
| `value_text(100, "relative_index")` | `100.0` | `100.0` |
| `pct(18.2)` · `pct(4.06)` · `pct(-4.06)` · `pct(0.02)` · `pct(9.96)` · `pct(1365.4)` · `pct(2.25)` · `pct(0.05)` | `+18%` · `+4.1%` · `−4.1%` · `0%` · `+10%` · `+1,365%` · `+2.3%` · `+0.1%` | same |
| `pct_arrow(12.03)` · `pct_arrow(-4.06)` · `pct_arrow(0)` | `▲12%` · `▼4.1%` · `0%` | same |
| `pp(0.4)` · `pp(-3.91)` · `pp(12.34)` · `pp(0.042)` · `pp(0.25)` | `+0.4%p` · `−3.9%p` · `+12.3%p` · `+0.042%p` · `+0.3%p` | `+0.4 pp` · `−3.9 pp` · `+12.3 pp` · `+0.042 pp` · `+0.3 pp` |
| `prob_text(0.81)` · `(0.83)` · `(0.5)` · `(0.974)` · `(0.975)` · `(0.99)` · `(0.03)` · `(0.02)` · `(0.825)` · `(0.875)` | `약 80%` · `약 85%` · `약 50%` · `약 95%` · `95% 이상` · `95% 이상` · `약 5%` · `5% 이하` · `약 85%` · `약 90%` | `about 80%` · `about 85%` · `about 50%` · `about 95%` · `95% or more` · `95% or more` · `about 5%` · `5% or less` · `about 85%` · `about 90%` |
| `prob_labelled(0.2)` · `prob_labelled(0.02)` | `상승 확률 약 20%` · `상승 확률 5% 이하` | `chance of rise about 20%` · `chance of rise 5% or less` |
| `likelihood(0.9)` · `(0.8)` · `(0.6)` · `(0.5)` · `(0.4)` · `(0.2)` · `(0.05)` | per D.1.5 rows 1,2,3,4,5,6,7 | same |
| `date_label("2026-11-17","W")` | `2026.11.17 주` | `Week of Nov 17, 2026` |
| `range_label("2021-11-01","2026-10-01","M")` | `2021.11 ~ 2026.10` | `Nov 2021 – Oct 2026` |
| `range_label("2021-11-29","2026-11-16","W")` | `2021.11.29 ~ 2026.11.22` | `Nov 29, 2021 – Nov 22, 2026` |
| `datetime_label("2026-11-25T00:12:00Z")` | `2026.11.25 09:12` | `Nov 25, 2026 09:12 KST` |

### D.2 Series VM

#### D.2.0 Full example (Naver, weekly, `hl=ko`)

```jsonc
{
  "schema": 1,
  "lang": "ko",
  "index": 0,
  "name": "선크림",
  "terms": ["선크림", "자외선차단제"],
  "slot": 1,                         // 1..5, or null (more than 5 series and not in the top 5; or before ranking)
  "icon": "circle",                  // circle | diamond | hexagon | square | cross | null
  "flags": {"synthetic": false, "tentative": false, "intermittent": false, "all_zero": false,
            "low_resolution": false, "reference_only": false, "seasonal": true},
  "source": {
    "kind": "naver",
    "label": "네이버 데이터랩 검색어트렌드(NAVER API HUB)",
    "short": "네이버",
    "line": "출처: 네이버 데이터랩 검색어트렌드(NAVER API HUB) · 주 단위 · 모바일·여성·25–34세 · 2021.11.29 ~ 2026.11.22 · 상대 지수(조회 기간 최댓값 = 100) · 데이터 기준 2026.11.25 09:12",
    "value_kind": "relative_index",
    "value_kind_label": "상대 지수(조회 기간 최댓값 = 100)",
    "value_kind_note": "상대 지수: 이 조회에서 가장 많이 검색된 주 = 100이에요. 실제 검색 횟수가 아니에요.",
    "unit_label": null,
    "freq": "W",
    "freq_label": "주 단위",
    "start": "2021-11-29",
    "end": "2026-11-16",
    "range_label": "2021.11.29 ~ 2026.11.22",
    "as_of": "2026-11-25T00:12:00Z",
    "as_of_label": "2026.11.25 09:12",
    "filters": {"device": "mo", "gender": "f", "ages": ["4", "5"]},
    "filters_label": "모바일·여성·25–34세",
    "filename": null,
    "attribution": "출처: 네이버 데이터랩 검색어트렌드(NAVER API HUB)"
  },
  "hero": {
    "title": "선크림",
    "outlook": {"direction": "up", "arrow": "▲", "tone": "up", "word": "상승",
                "likelihood": "상승 쪽이 우세", "prob": 0.812, "prob_text": "약 80%", "prob_exact": "81%",
                "prob_undefined_reason": null, "line": "▲ 상승 쪽이 우세 (상승 확률 약 80%)"},
    "sentence": "향후 26주 평균이 최근 26주 평균보다 18% 높을 것으로 예측해요.",
    "basis": "향후 26주 평균, 최근 26주 평균 대비",
    "change_undefined": null,
    "reliability": {"level": "medium", "label": "예측 신뢰도 보통"},
    "accuracy_line": "과거 데이터 검증: 단순 예측보다 오차 약 23% 작음",
    "reference_note": null
  },
  "phase": {
    "label": "growing", "name": "성장기", "name_long": "성장기", "icon": "growing",
    "seasonal_note": "계절성 뚜렷",
    "certainty": {"level": "high", "label": "판단 확신도 높음", "value": 0.95, "pct_text": "95%"},
    "tentative": false,
    "reasons": [{"text": "최근 39주 동안 추세가 +13.4% 변했습니다.", "detail": null},
                {"text": "Mann-Kendall 검정에서 전체 기간의 상승 추세가 유의합니다.", "detail": "τ = 0.41, p < 0.001"},
                {"text": "현재 추세는 고점(2026.11.16) 수준입니다.", "detail": null}],
    "rules_url": "/method#phase"
  },
  "tiles": [
    {"key": "phase", "label": "현재 국면", "value": "성장기", "icon": "growing", "tone": "neutral",
     "sub": "판단 확신도 높음", "tooltip": "추세의 모양과 통계 검정으로 정한 수명주기 단계예요. 근거는 ‘왜?’에서 볼 수 있어요.",
     "undefined_reason": null, "value_raw": null},
    {"key": "recent_change", "label": "최근 변화", "value": "▲12%", "icon": null, "tone": "up",
     "sub": "추세 기준 · 전년 동기 대비", "tooltip": "최근 한 주기(52주) 동안 추세선이 얼마나 변했는지예요. 계절 변동은 뺀 값이에요.",
     "undefined_reason": null, "value_raw": 12.03},
    {"key": "momentum", "label": "모멘텀", "value": "가속", "icon": null, "tone": "neutral",
     "sub": "+0.4%p/주", "tooltip": "최근 기울기와 장기 기울기의 차이예요(현재 추세 수준 대비 %p). +면 장기 평균보다 빨리 오르는 중이에요.",
     "undefined_reason": null, "value_raw": 0.4},
    {"key": "expected_change", "label": "예측 변화", "value": "▲18%", "icon": null, "tone": "up",
     "sub": "향후 26주 평균, 최근 26주 평균 대비", "tooltip": "예측 마지막 한 주기의 평균을 최근 한 주기의 평균과 비교한 값이에요. 특정 시점 한 값의 변화가 아니에요.",
     "undefined_reason": null, "value_raw": 18.2},
    {"key": "prob_increase", "label": "상승 가능성", "value": "약 80%", "icon": null, "tone": "neutral",
     "sub": "상승 쪽이 우세 · 예측 변화와 같은 기준", "tooltip": "같은 두 평균에서 미래 평균이 더 높을 확률이에요. 정확한 값: 81%",
     "undefined_reason": null, "value_raw": 0.812},
    {"key": "peak_season", "label": "피크 시즌", "value": "7월", "icon": null, "tone": "neutral",
     "sub": "보통 7월 정점 · 5월부터 상승", "tooltip": "계절 성분이 가장 높은 달과 오르기 시작하는 달이에요. 계절성이 뚜렷할 때만 보여 드려요.",
     "undefined_reason": null, "value_raw": null}
  ],
  "chart": {
    "title": "선크림 — 추이와 예측",
    "desc": "선크림 주간 상대 지수, 2021–2026. 보통 7월 정점. 향후 26주 평균 +18% 예상.",
    "freq": "W",
    "value_kind": "relative_index",
    "x_label": "주 단위 · 2021.11 ~ 2026.11",
    "y_label": "상대 지수(조회 기간 최댓값 = 100)",
    "dates": ["2021-11-29", "2021-12-06", …, "2026-11-16"],
    "actual": [12.37, 11.9, …, 61.2],
    "trend": [12.1, 12.0, …, 58.8],
    "forecast": {"dates": ["2026-11-23", …, "2027-05-17"],
                 "mean": [63.4, …], "lo80": [58.1, …], "hi80": [68.7, …], "lo95": [55.2, …], "hi95": [71.6, …]},
    "forecast_start": "2026-11-23",
    "last_actual": {"date": "2026-11-16", "value": 61.2, "label": "마지막 실제값 2026.11.16 주"},
    "levels": [80, 95],
    "display": "normal",              // normal | weak | bands_only
    "nonnegative": true,
    "y_hint": {"min": 0, "max": 71.6, "include_zero": true},
    "y_note": null,
    "changepoints": [{"n": 1, "date": "2023-03-06", "date_label": "2023.03.06", "label": "성장 가속",
                      "kind": "acceleration", "strength": "medium", "strength_label": "뚜렷한 전환",
                      "on_chart": true, "text": "2023.03.06 · 성장 가속 · 뚜렷한 전환"}],
    "anomalies": [{"date": "2025-07-06", "date_label": "2025.07.06", "value": 59.3, "expected": 52.4,
                   "direction": "up", "text": "평소보다 크게 높음 · 실제 59.3 / 예상 52.4"}]
  },
  "past": {
    "summary": "구조 변화 1번 · 이상치 1건",
    "seasonality": {"strength": 0.84, "text": "계절성이 매우 뚜렷해요(강도 0.84). 보통 7월에 높고 12월에 낮아요."}
  },
  "future": {
    "horizon": 26, "horizon_label": "26주", "end_label": "2027.05.17 주",
    "scenarios": {
      "date_label": "2027.05.17 주",
      "title": "2027.05.17 주 한 시점의 값",
      "reference": {"label": "같은 계절 시점(2026.05.18 주)의 실제값", "value_text": "52.4",
                    "text": "비교 기준: 같은 계절 시점(2026.05.18 주)의 실제값 52.4"},
      "rows": [{"key": "pessimistic", "label": "비관", "value_text": "48.1", "pct_text": "−8.2%",
                "text": "비관 48.1 (기준 대비 −8.2%)"},
               {"key": "base", "label": "기준", "value_text": "55.0", "pct_text": "+5.0%",
                "text": "기준 55.0 (기준 대비 +5.0%)"},
               {"key": "optimistic", "label": "낙관", "value_text": "60.8", "pct_text": "+16%",
                "text": "낙관 60.8 (기준 대비 +16%)"}],
      "note": "예측 마지막 시점 한 값의 80% 범위예요. 위의 ‘예측 변화’(평균 대 평균)와는 다른 양이에요."
    },
    "table": {"rows": [{"date": "2026-11-23", "date_label": "2026.11.23 주", "mean": "63.4",
                        "range80": "58.1~68.7", "range95": "55.2~71.6"}, …]},
    "accuracy": {"line": "과거 데이터 검증: 단순 예측보다 오차 약 23% 작음", "skill": 0.77,
                 "details": "기술 참고: 백테스트 MASE 0.84 — 학습 구간에서 한 기 앞을 계절 단순 예측한 오차를 1로 둔 상대 오차예요. 1과 비교하지 마세요."},
    "method": "3개 모델 앙상블(Holt-Winters 계절 모형 36% · 선형 추세 + 계절 항 34% · 감쇠 Holt 30%)",
    "models": [{"id": "holt_winters", "name": "Holt-Winters 계절 모형", "weight": 0.36, "weight_text": "36%"}, …],
    "phase": {"label": "growing", "name": "성장기", "hedge": null,
              "text": "26주 뒤에도 성장기 국면이 이어질 것으로 보여요."},
    "peak": {"date": null, "text": "예측 기간 안에 뚜렷한 정점은 보이지 않아요."}
  },
  "narrative": {
    "markdown": "**한눈에 보기** — …\n\n## 과거: 어떻게 여기까지 왔나\n- …\n\n## 현재: 지금 어디에 있나\n- …\n\n## 미래 전망\n- …",
    "date_links": {"2023.03.06": "2023-03-06", "2025.07.06": "2025-07-06"},
    "source": "rule",
    "badge": "규칙 기반 해석",
    "ai": null,
    "ai_allowed": false,
    "ai_block_reason": "naver_terms",
    "ai_block_text": "네이버 데이터는 이용약관에 따라 AI로 보내지 않아요.",
    "ai_privacy_note": "계산된 요약 수치(JSON)만 Anthropic으로 보내요. 원본 데이터는 보내지 않아요."
  },
  "caveats": {
    "markdown": "- 예측 구간은 먼 미래일수록 넓어집니다.\n- …",
    "disclaimer": "예측은 과거 패턴이 이어진다고 가정해요. 갑작스러운 사건이나 외부 변수(가격·광고·휴일)는 반영하지 못해요."
  },
  "summary_line": "[trendai] 선크림(네이버, 주간): 성장기 · 향후 26주 평균 +18% 예상(상승 가능성 약 80%) · 보통 7월 정점 (데이터 기준 2026.11.25 09:12)",
  "exports": {                       // URLs built per response from stored availability flags (E.7)
    "csv": "/api/export/Zx8kP2mQ9rT4vW6yA1bC3d/0.csv",
    "xlsx": "/api/export/Zx8kP2mQ9rT4vW6yA1bC3d/0.xlsx",
    "html": null,
    "json": "/api/export/Zx8kP2mQ9rT4vW6yA1bC3d/0.json",
    "unavailable": {"html": {"code": "export_naver_raw", "text": "네이버 원본 데이터는 이용약관상 내려받기를 제공하지 않아요. …"}}
  },
  "notices": []
}
```

#### D.2.1 Identity and flags

* `name` = display name (C.5). `terms`: Naver keywords / `[title]` / `[column]` / `[sample name]` / `[term]` (GT).
* `slot`/`icon`: D.4.2. Icons by slot: 1 `circle` ●, 2 `diamond` ◆, 3 `hexagon` ⬢, 4 `square` ■, 5 `cross` ✚.
* `flags.synthetic` = `narrative.is_synthetic(analysis)` — true for samples only because E.6 copies the raw series'
  `attrs` (incl. `samples.SYNTHETIC_ATTR`) onto the prepared series before `pipeline.run` (`data.prepare` returns a
  series with empty attrs); `tentative` = `narrative.phase_is_tentative`;
  `intermittent` = `narrative.is_intermittent`; `all_zero` = `narrative.is_all_zero`; `low_resolution` = Naver and
  `max(actual) < 5` within a multi-series request; `reference_only` = `chart.display != "normal"`;
  `seasonal` = peak season shown (D.2.4).

#### D.2.2 Source

`line` = `"출처: "`/`"Source: "` + parts joined by ` · `, in this order (empty parts skipped):

| kind | label (ko / en) | parts after the label |
|---|---|---|
| `naver` | 네이버 데이터랩 검색어트렌드(NAVER API HUB) / Naver DataLab Search Trends (NAVER API HUB) — legacy endpoint: `(네이버 오픈API)` / `(Naver Open API)` | freq · filters · range · value kind · `데이터 기준 {as_of}` / `data as of {as_of}` |
| `wiki` | Wikimedia Pageviews · {한국어/영어/`{wl}`} 위키백과 · 사람 조회만(봇 제외) / Wikimedia Pageviews · {Korean/English/`{wl}`} Wikipedia · human views only (bots excluded) | freq · range · 조회수 / page views · data as of |
| `upload` | 사용자 업로드 ({filename}) / Your upload ({filename}) | freq · range · `원래 단위` (+ `({unit_label})`) / original units · `분석 {as_of}` / analysed {as_of} |
| `google_trends_upload` | Google 트렌드(사용자 업로드, {filename}) · Data source: Google Trends (https://www.google.com/trends) / Google Trends (your upload, {filename}) · Data source: Google Trends (https://www.google.com/trends) | freq · range · `상대 지수(0–100, 요청마다 다시 맞춰짐)` / relative index (0–100, rescaled per request) · analysed |
| `sample` | trendai 내장 샘플 · 가상 데이터 / trendai built-in sample · synthetic data | freq · range |

* `start`/`end` are the first and last **data** dates (for weekly data: bucket start dates), not the request bounds;
  `range_label` = `range_label(start, end, freq)` (W shows the last bucket's Sunday, D.1.6).
* `short` (summary line, D.2.10), per kind: `naver` 네이버 / Naver · `wiki` 위키백과 / Wikipedia · `upload` 내 파일 /
  my file · `google_trends_upload` Google 트렌드 업로드 / Google Trends upload · `sample` 샘플·가상 데이터 / sample ·
  synthetic data.
* `as_of` = the fetch time (cached data keep their original fetch time); uploads/samples: analysis time.
* `filters_label`: device `PC`/`모바일` (`PC`/`mobile`), gender `남성`/`여성` (`male`/`female`), ages from
  `/api/config` labels with consecutive codes merged (`4,5` → `25–34세` / `25–34`; `2,5` → `13–18세·30–34세`).
  Joined with `·` (ko) / ` · ` (en). Empty when no filter.
* `value_kind_note`: `relative_index` → "상대 지수: 이 조회에서 가장 많이 검색된 {날/주/달} = 100이에요. 실제 검색 횟수가
  아니에요." / "Relative index: the {day/week/month} with the most searches in this query = 100. Not a search
  count."; `relative_index_gt` → "Google 트렌드가 요청마다 0–100으로 다시 맞춘 상대 지수예요. 다른 다운로드와 비교할 수
  없어요." / "Google Trends rescales each request to 0–100; not comparable with other downloads.";
  `pageviews` → "위키백과 조회수는 관심의 대리 지표예요. 검색량과 다를 수 있어요." / "Wikipedia page views are a proxy
  for interest and can differ from search volume."; `raw` → `""` (samples: "가상 데이터예요. 실제 데이터가 아니에요." /
  "Synthetic data, not real data.").

#### D.2.3 Hero

* `outlook.direction` = `Outlook.direction`; `word` = `report_to_dict(...)["outlook"]["direction_label"][lang]`;
  `arrow`/`tone`: `up` → `▲`/`up`, `down` → `▼`/`down`, `flat` (보합 or 방향 불확실) → `▶`/`neutral`.
* `prob` = JSON-export `prob_increase` (null when `prob_increase_undefined`); `prob_text`/`prob_exact`/`likelihood`
  per D.1.5. `prob_undefined_reason` = `narrative.prob_reason(analysis, lang).short` when undefined.
* `line`: if `prob` is null → `"{arrow} {word} · 상승 확률은 계산하지 않음({reason})"` / `"{arrow} {word} · chance of
  rise not computed ({reason})"`; if the likelihood's side (P ≥ 0.55 → rise, ≤ 0.45 → fall) matches a non-flat direction →
  `"{arrow} {likelihood} ({prob_labelled})"`; otherwise `"{arrow} {word} · {likelihood} ({prob_labelled})"`. The number in
  brackets is always P(rise) and always says so (`prob_labelled`, D.1.5). Required vectors
  (`test_outlook_line_rules`): direction up, P = 0.812 → `▲ 상승 쪽이 우세 (상승 확률 약 80%)`; direction down,
  P = 0.2 → `▼ 하락 쪽이 우세 (상승 확률 약 20%)` / `▼ Fall more likely (chance of rise about 20%)`; direction down,
  P = 0.02 → `▼ 하락 가능성 매우 높음 (상승 확률 5% 이하)` / `▼ Fall very likely (chance of rise 5% or less)`;
  direction flat (보합), P = 0.5 → `▶ 보합 · 상승·하락 엇비슷 (상승 확률 약 50%)`.
* `basis`: `(fut_b, cur_b) = report._change_basis(report, report._ChartData(report))` (en `_change_basis_en`) →
  `"{fut_b}, {cur_b} 대비"` / `"{fut_b}, vs. {cur_b}"`.
* `sentence` (해요체):
  * change defined, direction up/down: ko `"{fut_b}이 {cur_b}보다 {pct_abs} {높을|낮을} 것으로 예측해요."`,
    en `"The {fut_b} is expected to be {pct_abs} {higher|lower} than the {cur_b}."`;
  * `word` = 보합: ko `"{fut_b}이 {cur_b}과 비슷할 것으로 예측해요(변화 {pct})."`, en `"The {fut_b} is expected to stay
    close to the {cur_b} (change {pct})."`;
  * `word` = 방향 불확실: ko `"{fut_b}이 {cur_b}보다 {pct_abs} {높을|낮을} 수 있지만, 불확실성에 비해 변화가 작아 방향을
    단정하기 어려워요."`, en `"The {fut_b} may be {pct_abs} {higher|lower} than the {cur_b}, but the change is small
    relative to the uncertainty, so the direction is unclear."`;
  * change undefined: `nc = report._change_undefined(report, d, lang)`; `sentence = nc.sentence` + `" ({nc.levels})"`
    when `nc.levels`; `change_undefined = {"code": nc.code, "reason": nc.short, "levels": nc.levels}`.
* `reliability`: `Outlook.confidence` → `예측 신뢰도 높음/보통/낮음` / `Forecast reliability: high/medium/low`.
* `accuracy_line` from `ForecastResult.skill` / `backtest_mase` (the narrative's thresholds):
  `skill ≤ 0.95` → `과거 데이터 검증: 단순 예측보다 오차 약 {(1−skill)·100:.0f}% 작음` (`99% 이상` beyond 99.5) /
  `Backtest: about {g}% lower error than a naive forecast`; `0.95 < skill ≤ 1` → `과거 데이터 검증: 단순 예측과 오차가
  비슷함` / `Backtest: about as accurate as a naive forecast`; `skill > 1` → `과거 데이터 검증: 단순 예측보다 나은 성과를
  보이지 못함` / `Backtest: did not beat a naive forecast`; skill null, MASE present → `과거 데이터 검증: 단순 예측과 비교한
  결과 없음` / `Backtest: no comparison with a naive forecast`; both null → `데이터가 짧아 정확도를 검증하지 못했어요` /
  `Too short to check accuracy`.
* `reference_note`: `display == "bands_only"` → `예측력이 낮아 범위만 참고용으로 보여 드려요.` / `The forecast did not beat
  a naive one, so only the range is shown, for reference.`; `"weak"` → `예측 신뢰도가 낮아 중앙선은 참고용이에요.` /
  `Low reliability: treat the centre line as a rough guide.`; else `null`.

#### D.2.4 Tiles (always exactly 6, in this order)

`tone` ∈ `up` | `down` | `neutral` (colour via `--up`/`--down`, always with an arrow or sign). An undefined value
has `value: "—"` and `undefined_reason` (a phrase); `sub` keeps the basis. **Weak-forecast rule** (brief §4.3 rule
8, applied on every surface, not only the fan chart): when `chart.display != "normal"`, tiles 4 and 5 append
` · 참고용` / ` · for reference` to their `sub`.

| # | key | label ko / en | value | sub | undefined_reason |
|---|---|---|---|---|---|
| 1 | `phase` | 현재 국면 / Current phase | `phase.name` (+ `(잠정)`/` (tentative)`) | `phase.certainty.label` | — |
| 2 | `recent_change` | 최근 변화 / Recent change | `pct_arrow(metrics.recent_change_pct)`, tone by sign | `추세 기준 · {basis}` where basis = `전년 동기 대비` (M, Q, W with lag 52), `전년 대비` (Y), `7일 전 대비` (D), or `{lag}{unit} 전 대비` when the lag is shorter than a season (lag rule of `report._recent_lag`); en `trend · YoY` / `vs. {lag} {units} ago` | `report._pct_why(code, "recent_change_pct", lang)` with `code = metric_pct_undefined(analysis).get("recent_change_pct")` |
| 3 | `momentum` | 모멘텀 / Momentum | word (below) | `pp(metrics.momentum)` + `/주`, `/월`, `/일`, `/분기`, `/년` (en `/wk`, `/mo`, `/day`, `/qtr`, `/yr`) | `report._rate_why(report, narrative.momentum_pct_undefined(analysis), lang)` |
| 4 | `expected_change` | 예측 변화 / Expected change | `pct_arrow(expected_change_pct)`; for 보합/방향 불확실: `pct(…)` with tone neutral | `hero.basis`; for 방향 불확실 prefix `방향 불확실 · ` / `direction uncertain · ` | `hero.change_undefined.reason` |
| 5 | `prob_increase` | 상승 가능성 / Chance of rise | `prob_text` | `{likelihood} · 예측 변화와 같은 기준` / `{likelihood} · same basis as expected change` | `prob_undefined_reason` |
| 6 | `peak_season` *or* `volatility` | 피크 시즌 / Peak season · 변동성 / Volatility | see below | see below | volatility: `report._rate_why(report, narrative.volatility_undefined(analysis), lang)` |

**Momentum word**: tentative → `판단 보류`/`Too short to tell`; phase stable/volatile → `뚜렷하지 않음`/`No clear
change`; else with `eps = 1 / periods_per_year(freq)` and `recent` = sign of `trend[-1] − trend[-1−r]`,
`r = max(1, min(max(3, min(season_length, n // 4)), n − 1))`: `momentum > eps` → `가속`/`Accelerating` if
recent > 0 else `하락 둔화`/`Decline easing`; `momentum < −eps` → `하락 가속`/`Falling faster` if recent < 0 else
`둔화`/`Slowing`; otherwise `비슷한 속도`/`Steady pace`.

**Peak season** — exactly the narrative's `_seasonal_pattern` (narrative.py) condition and grouping, so the tile and
the narrative always name the same peak: shown when `decomposition.period > 1`, `metrics.seasonal_strength ≥ 0.3`,
the seasonal component has ≥ 2 finite values on a DatetimeIndex, and either `freq ∈ {M, W}` (group by calendar
month), `freq == Q` (by quarter) or **`freq == D` and `decomposition.period == 7`** (by weekday); any other case
(Y, D with another period) → no peak season. Profile = group means of the finite seasonal values; shown only if every
group mean is finite and `max − min > 0` (this replaces a separate all-zero check). Peak = argmax; trough = argmin;
rise start = walk back cyclically from the peak while the previous group is strictly lower (at most
`len(profile) − 1` steps); the earliest group of that run. Names: months `7월` / `Jul` (D.1.6 abbreviations; the
narrative's `July` names the same month), quarters `3분기` / `Q3`, weekdays the narrative's own names `토요일` /
`Saturdays` (plural, as in "peaks on Saturdays"). Sub ko `보통 {peak} 정점 · {start}부터 상승` (`· …부터 상승` omitted
when start == peak; for D: `요일 패턴 · 가장 낮은 날 {trough}`), en `Usually peaks in {peak} · rises from {start}`
(D: `Weekday pattern · lowest on {trough}`).
Otherwise tile 6 is **volatility**: `metrics.volatility` (CV) → value `낮음` (< 0.05) / `보통` (< 0.15) / `높음`
(en `Low`/`Medium`/`High`), sub `잔차 변동 {cv·100:.1f}%` / `residual CV {…}%`.

Tooltips (server texts, `texts.TILE_HELP`, G.14) are fixed strings per key and language; the prob tile appends
`정확한 값: {prob_exact}` / `Exact value: {prob_exact}`.

#### D.2.5 Phase

* `label` = `phase.label`; `name` short: 태동기/성장기/정점/쇠퇴기/안정기/재부상/변동성 (en Emerging/Growing/Peaking/
  Declining/Stable/Reviving/Volatile); `name_long` = `PHASE_KO` / `narrative.PHASE_EN`; `icon` = label.
* `seasonal_note`: stable phase and `seasonal_strength ≥ 0.3` → `계절성 뚜렷` / `strongly seasonal`.
* `certainty.level`: tentative → `low` with label `잠정 판단`/`Tentative`; else `confidence ≥ 0.75` → `high`,
  `≥ 0.5` → `medium`, else `low`; label `판단 확신도 높음/보통/낮음` / `Phase certainty: high/medium/low`;
  `pct_text` = `format_prob(confidence)` — shown **only inside the `help.certainty` popover**, never as visible text
  next to `상승 가능성 약 80%` or `예측 신뢰도` (three percent-like numbers side by side were the R4 naming collision).
* `reasons`: a list of `{text, detail}` built from ko = `phase.reasons`, en = `[translate_reason(r) for r in reasons
  if translate_reason(r)]` (untranslated reasons dropped, as in the English narrative; if none remain:
  `[{"text": "See the method page for the rules.", "detail": null}]`). For each reason: ISO dates are rewritten to
  `date_short(…, freq, lang)` (same regexes as D.2.9), and a trailing parenthetical holding test statistics
  (regex `\s*\(([^()]*(?:τ|p\s*[<=>]|t\s*=|MK|Mann)[^()]*)\)\.?$`, the final full stop kept on `text`) is moved to
  `detail` (shown under a `자세히` toggle); otherwise `detail` is `null`. The wording itself is the engine's (S1, A.2).

#### D.2.6 Chart

* `dates`/`actual` = `analysis.series` (prepared, regular, no gaps); `trend` = `decomposition.trend`;
  `forecast` = `ForecastResult.to_frame()` columns `mean`, `lo80`, `hi80`, `lo95`, `hi95` (levels as returned).
  Arrays are the same length as their `dates`; `null` only for non-finite values.
* `display`: `skill is not None and skill ≥ 1.0` → `bands_only`; elif `outlook.confidence == "low"` or `skill is None`
  → `weak`; else `normal` (brief §4.3 rule 8).
* `y_hint` (detail fan chart): `min`/`max` = data extremes incl. the 95 % band; `include_zero` = `true` for
  `relative_index` / `relative_index_gt`, otherwise nonnegative series and `min(actual) ≤ 0.5 × max(actual)`. A
  relative index is therefore drawn on its own range (0 to its max), not forced to 0–100, so a small term opened
  from a comparison is not flattened near zero; the comparison chart alone uses 0–100 (F.7.2). The client rounds to
  nice ticks (F.7).
* `y_note`: for `relative_index` when `max(actual) < 50` → `이 조회의 최댓값(100)보다 작은 검색어라 세로축을 확대해
  그렸어요.` / `This term is small relative to the query's maximum (100), so the vertical axis is zoomed in.`;
  else `null`. Shown as a caption under the chart.
* `changepoints`: numbered 1..n in `analysis.changepoints` order; `label` = `analysis.changepoint_label(cp, lang)`;
  `strength` by `magnitude`: ≥ 0.5 `large` (`큰 전환`/`major shift`), ≥ 0.2 `medium` (`뚜렷한 전환`/`clear shift`), else
  `small` (`작은 전환`/`minor shift`); `on_chart` = among the 5 largest magnitudes.
* `anomalies`: `direction` = sign of `zscore`; `text` ko `평소보다 크게 {높음|낮음} · 실제 {v} / 예상 {e}`, en `Far
  {above|below} the usual range · actual {v} / expected {e}` (values `value_text(…, kind, compact=False)`;
  intermittent: `예상` → `평소 규모` / `usual size`).
* `desc` (for `<desc>`): `"{name} {freq_adj} {value word}, {start_year}–{end_year}. {peak sentence}. {fut_b} {pct} 예상."`
  (en analogous); `title` = `"{name} — 추이와 예측"` / `"{name} — history and forecast"`.

#### D.2.7 Past

`summary` = `구조 변화 {n}번 · 이상치 {m}건` / `{n} changepoint(s) · {m} anomaly(ies)`. `seasonality.text`: strength
≥ 0.6 `계절성이 매우 뚜렷해요(강도 {s:.2f}).`, ≥ 0.3 `계절성이 뚜렷해요(강도 {s:.2f}).`, else `뚜렷한 계절성은 없어요.`; when
peak season is shown append ` 보통 {peak}에 높고 {trough}에 낮아요.` (en `Strongly seasonal (strength 0.84). Usually
high in Jul and low in Dec.`). The changepoint and anomaly lists are `chart.changepoints` / `chart.anomalies`.

#### D.2.8 Future

* `scenarios`: `Outlook.scenarios` at the last forecast date; reference = `narrative.scenario_reference(analysis,
  forecast)`: kind `season` → `같은 계절 시점({date_label})의 실제값` / `actual at the same point of the season
  ({date_label})`; `latest` → `최근 실제값({date_label})` / `latest actual ({date_label})`; `demand` → `최근 {span} 평균
  수요량` / `average demand over the last {span}`. Rows: `narrative.scenario_value(v, reference,
  nonnegative=narrative.scenario_floor(analysis, forecast), lang=lang)` with `reference = None` when
  `Outlook.expected_change_pct` is `None` (every other surface does the same) → text replaced by
  `value_text(v, kind, compact=True)` unless it is the floor text (`0(하한)` / `0 (floor)`); `pct_text = pct(...)` or
  `null`.
* Scenario labelling (the hero % and the scenario % are different quantities): `title` = `{date_label} 한 시점의 값` /
  `Single value at {date_label}`; `reference.text` = `비교 기준: {reference.label} {value_text}` / `Compared with:
  {label} {value_text}` (rendered as a header row above the three cases); each row's `text` = `{label} {value_text}
  (기준 대비 {pct_text})` / `{label} {value_text} ({pct_text} vs. reference)`, or `{label} {value_text}` without a
  percentage.
* `table.rows`: every forecast step: `mean` = `value_text(…, compact=False)`, `range80` = `"{lo}~{hi}"` (en
  `"{lo}–{hi}"`), `range95` likewise; bounds at the floor of a nonnegative forecast print `0`.
* `accuracy.details` = the narrative's technical MASE note in 해요체 (`기술 참고: 백테스트 MASE {m:.2f} — …`), `null`
  without MASE.
* `method` = `narrative.describe_method(forecast, lang)`; `models` = `forecast.model_scores` with `weight > 0`,
  `name = narrative.model_label(id, lang)`, `weight_text = pct-free "36%"`.
* `phase` from `Outlook.future_phase` + `narrative.future_phase_status(analysis, outlook)`: `undecided` → hedge
  `판단 보류`/`undecided`, text `흐름이 보합으로 예측돼 {h} 뒤 국면은 판단하기 어려워요.`; `tentative` → `잠정`/
  `tentative`, `관측치가 적어 잠정 판단이에요.`; `low_confidence` → `참고용`/`for reference`, `예측 신뢰도가 낮아 참고용이에요.`;
  none → `{h} 뒤에도 {name} 국면이 이어질 것으로 보여요.` (same label) or `{h} 뒤에는 {name} 국면으로 바뀔 것으로 보여요.`
  (en `In {h} the phase is expected to stay {name}.` / `… to become {name}.`).
* `peak`: `Outlook.forecast_peak_date` → `예측 기간 안에 {date_label} 무렵 정점이 예상돼요.` / `A peak around {date_label}
  is expected within the horizon.`; else `예측 기간 안에 뚜렷한 정점은 보이지 않아요.` / `No clear peak within the horizon.`

#### D.2.9 Narrative and caveats

* `markdown` = the report's narrative in the VM language (job runs `run(..., lang="ko")` then
  `narrative.generate_narrative(a, f, o, lang="en")` for the English VM), **cut before** the line
  `## {SECTION_HEADINGS[lang][3]}`; `caveats.markdown` = the lines after that heading (empty if absent).
  The same split applies to AI narratives.
* **Dates in the narrative** (`markdown`, `caveats.markdown`, AI text alike): the engine prints them with
  `narrative.format_date` (D/W `2024-03-05`, M `2024-03`, Q `2024-Q1`, Y `2024`). WP1 rewrites every match of
  `\b\d{4}-\d{2}-\d{2}\b`, `\b\d{4}-\d{2}\b` (only for M) and `\b\d{4}-Q[1-4]\b` (only for Q) to
  `date_short(…, freq, lang)` (`2024.03.05`, `2024.03`, `2024 1분기` / `Q1 2024`) and records
  `date_links[display text] = chart date`: D/W exact match of a `chart.dates`/`forecast.dates` entry; M the date
  starting with that `YYYY-MM`; Q the quarter's start date; Y never linked. Strings without a matching chart date are
  rewritten but not linked. md.js links exact occurrences of the `date_links` keys (F.6.14), so the client needs no
  per-frequency date logic.
* `badge`: `규칙 기반 해석` / `Rule-based`; AI: `AI 해석 · Claude` / `AI · Claude`.
* `ai`: stored AI narrative for this language or `null` (`{markdown, caveats_markdown, badge, model, note, created_at}`).
* `ai_allowed` = not Naver and `features.ai` and not a share; `ai_block_reason` ∈ `naver_terms` | `ai_unavailable` |
  `shared_readonly` | `null`; `ai_block_text` = text of `ai_naver_blocked` / `ai_unavailable` / `ai_readonly`.
* `caveats.disclaimer` fixed (G.6 `detail.disclaimer`).

#### D.2.10 Summary line, exports, notices

* `summary_line`: ko `[trendai] {name}({source.short}, {freq_adj}): {phase.name} · {fut_b} {pct} 예상(상승 가능성
  {prob_text})` + ` · 보통 {peak} 정점` when seasonal; undefined change → `{word} 전망`; en `[trendai] {name} ({short},
  {freq_en}): {phase} · {fut_b} {pct} expected (chance of rise {prob_text}) · usually peaks in {peak}`. Then, when
  `chart.display == "bands_only"`: ` (참고용: 단순 예측보다 낫지 않음)` / ` (for reference: did not beat a naive
  forecast)`; when `"weak"`: ` (참고용 예측)` / ` (forecast for reference only)`. Then always
  ` (데이터 기준 {as_of_label})` / ` (data as of {as_of_label})`. Samples always say `샘플·가상 데이터` through
  `source.short`. The client appends ` — {URL}` (public sources, with the absolute range of F.6.12) or
  ` — {share URL}` when one exists.
* `exports`: per-series links (C.10); `null` when unavailable with the reason in `unavailable[fmt] = {code, text}`
  (`html` for Naver → `export_naver_raw`; `xlsx` without openpyxl → `export_xlsx_unavailable`). Stored as
  availability flags; the URLs (`/api/export/<id>/<k>.<fmt>` or `/api/export/s/<share_id>/<k>.<fmt>`) are built for
  each response (E.7).
* `notices`: series-level notices (C.12.5) — data-layer notices captured while preparing this series,
  `data_short` (n < 8), `naver_low_resolution`, `mostly_zero_volume` (≥ 50 % zeros in a Naver/Wiki series),
  `naver_missing_filled` (its own count).

### D.3 Result envelope

C.7 is the envelope; `series` holds D.2 VMs, `compare` the D.4 VM, `exports` D.5.

### D.4 Compare VM

```jsonc
{
  "count": 3,
  "slots": {"0": 1, "1": 2, "2": 3},
  "value_kind": "relative_index",
  "source_line": "출처: 네이버 데이터랩 검색어트렌드(NAVER API HUB) · 주 단위 · 모바일·여성·25–34세 · 2021.11.29 ~ 2026.11.22 · 상대 지수(조회 기간 최댓값 = 100) · 데이터 기준 2026.11.25 09:12",
  "axis_label": "상대 지수(조회 기간 최댓값 = 100)",
  "axis_note": "이 조회에서 가장 많이 검색된 주 = 100 · 다른 조회와 비교할 수 없어요",
  "scale": {
    "default": "raw",
    "allowed": {"raw": true, "index": false, "log": true},
    "disabled_reason": {"log": null, "index": "같은 조회 안에서 이미 같은 척도예요."},
    "index_base": {"start": "2021-11-29", "end": "2022-11-21",
                   "label": "각 시계열의 2021.11.29 ~ 2022.11.21 평균을 100으로 맞췄어요",
                   "values": {"0": 23.4, "1": 41.2, "2": 18.0}},
    "log_hint": false
  },
  "ranking": [
    {"rank": 1, "index": 1, "name": "패딩", "slot": 2, "icon": "diamond",
     "phase": {"label": "growing", "name": "성장기", "icon": "growing", "tentative": false},
     "outlook": {"arrow": "▲", "tone": "up", "word": "상승", "likelihood": "상승 쪽이 우세", "prob_text": "약 80%",
                 "line": "▲ 상승 쪽이 우세 (상승 확률 약 80%)"},
     "expected_change": {"text": "+18%", "basis": "향후 26주 평균, 최근 26주 평균 대비", "undefined_reason": null},
     "recent_change": {"text": "+12%", "basis": "전년 동기 대비", "undefined_reason": null},
     "momentum": {"text": "+0.4%p/주", "word": "가속", "undefined_reason": null},
     "score": {"value": 82.4, "text": "82.4"},
     "confidence": {"level": "medium", "label": "예측 신뢰도 보통"},
     "sort": {"score": 82.4, "prob": 0.81, "expected": 18.2, "recent": 12.0, "level": 48.3},
     "spark": {"history": [41.0, …], "forecast": [52.1, …]},
     "display": "normal",
     "flags": {"low_resolution": false, "reference_only": false, "tentative": false, "synthetic": false}}
  ],
  "failed": [{"index": 2, "name": "플리스", "error": { /* C.12.1 */ }}],
  "score_note": "상승 점수는 전망(예측 변화·상승 확률)과 최근 기세(최근 변화·모멘텀)를 합친 0–100 값이에요. 50이 중립이고, 예상 상승률이 아니에요.",
  "spark_note": "스파크라인은 카드마다 세로축이 달라요.",
  "notices": []
}
```

#### D.4.1 Ranking

`compare.rank_trends({display name: report})` (best first). Display names are unique (E.6.1) and are also each
report's `name`, which `rank_trends` uses for its `name` column; WP1 maps every ranking row back to the series index
through a `{display name: index}` dict built at the same time, never by position or by re-parsing names. Row texts
come from the series VM (same rules as the tiles: `expected_change.text = pct(...)`, `recent_change.text = pct(...)`;
`outlook.line` per D.2.3, with the number always labelled). `score.text` = one decimal.
`sort.level` = last actual value (`metrics.last_value`). `spark.history` = the last `min(len, 2·season_length, 104)`
actual values; `spark.forecast` = the forecast mean (≤ 52 values), **empty** when `display == "bands_only"` (the
weak-forecast rule; `display` = the series' `chart.display`). Up to 20 rows (uploads); search compare ≤ 5.

`source_line` = `"출처: "`/`"Source: "` + the parts shared by the compared series, in the D.2.2 order: label · freq ·
filters · range · value kind · as-of. `range` = earliest successful series start ~ latest end (`range_label`);
as-of = the **oldest** `as_of` among the series (`데이터 기준` for Naver/Wikipedia, `분석` for uploads/samples), so
mixed cache times never overstate freshness. Uploads use the upload label with the filename; samples the sample
label (`가상 데이터` included). It is rendered under the compare chart (F.6.8), is the first comment line of the
compare CSVs (C.10), the HTML source paragraph, and the footer of a compare PNG (F.7.6).

#### D.4.2 Slots

N ≤ 5: `slot = index + 1` (request order — colours follow series, never rank; removing a series keeps the others'
colours because the client keeps the remaining series' original order in the URL). N > 5 (uploads): the 5 best
scores get slots 1–5 assigned in request order; the rest `null` (drawn grey). Series VMs delivered while the job is
running carry `slot: null` when N > 5; the final stored VMs and `slots` carry the assignment, and the client always
takes slots from `compare.slots` of the final `GET /api/results` (C.6).

#### D.4.3 Scale rule (brief §3.4)

| source kind | default | notes |
|---|---|---|
| `naver`, `google_trends_upload` | `raw` | one request = one scale; **`index` is not allowed** (`disabled_reason.index` = `같은 조회 안에서 이미 같은 척도예요.` / `These series already share one scale (same request).`); `log` per the rule below (A.3) |
| `wiki` | `raw` | `log_hint = true` (+ notice `log_hint` with `ratio` = 20) when log is allowed and max(series max) / min(series max) > 20 |
| `upload` | `index` | falls back to `raw` with notice `index_unavailable` when index is not allowed |
| `sample` | `raw` | |

* **index base**: common start = latest series start; window = the first `season_length(freq)` periods from it
  (1 for Y); base = mean of `actual` in the window per series. Allowed iff the source kind permits it (table above)
  and every base is finite and `> 1e-9 × max|actual|` and > 0 (`index_base` is still computed for `wiki`, `upload`,
  `sample`; `null` for `naver`/`google_trends_upload`). Label ko `각 시계열의 {range_label} 평균을 100으로 맞췄어요` (one-period window:
  `각 시계열의 {date_short} 값을 100으로 맞췄어요`), en `Each series' average over {range} is set to 100`.
  `disabled_reason.index` = `index_unavailable` text when not allowed.
* **log** allowed iff every `actual` and every `lo95` > 0; else `disabled_reason.log` = `0 이하 값이 있어 로그 축을
  쓸 수 없어요.` / `Values of 0 or less: a log axis is not possible.`
* The client applies the transform (`value / base × 100`, or log axis). No dual axes, ever.
* `axis_label`: `relative_index` → the value kind label; `pageviews` → `조회수` / `page views`; `raw` → `unit_label`
  or `값` / `value`. Under `index` the client shows `기준 시점 = 100` / `base period = 100`.

### D.5 Result-level exports

```json
"exports": {"csv": "/api/export/<id>.csv", "xlsx": "/api/export/<id>.xlsx", "html": "/api/export/<id>.html",
            "json": "/api/export/<id>.json", "values_csv": "/api/export/<id>/values.csv", "unavailable": {}}
```

Same availability rules as D.2.10; `values_csv` is `null` for single-series results and for Naver (with
`unavailable.values_csv = {code: "export_naver_raw", text}`). Share envelopes use `/api/export/s/<share_id>…`. Stored
as flags; URLs built per response (E.7).

### D.6 What is pre-formatted (server) vs raw (client formats)

| Data | Server sends | Client does |
|---|---|---|
| Hero, tiles, phase, reasons, scenarios, forecast table, accuracy, method, models, future phase/peak, past summary, seasonality, changepoint/anomaly texts, source line, notices, errors, summary line, ranking row texts, score/spark notes | final strings in the VM language | renders as text (never as HTML) |
| `value_raw`, `prob`, `sort.*`, scale `index_base.values` | raw numbers | sorting, transforms |
| Chart arrays (`actual`, `trend`, `forecast.*`), dates, `y_hint`, spark arrays | raw numbers / ISO dates | geometry, axis ticks (D.1.6), crosshair read-out (`value_text` rules mirrored in fmt.js: `full` + `value_kind`), table view cells |
| Narrative | Markdown-lite string | safe rendering (F.6.14), date links |
| UI chrome (buttons, headings, help, empty states, client-side errors) | — | `catalog.js` (G) |

---
## E. Backend internals (WP1 unless stated)

### E.1 Configuration (`WebConfig.from_env`)

| Environment variable | Field | Default | Meaning |
|---|---|---|---|
| `TRENDAI_HOST` | `host` | `127.0.0.1` | bind address of `trendai serve` |
| `TRENDAI_PORT` | `port` | `8000` | port (`0` = random; the real one is printed) |
| `TRENDAI_DATA_DIR` | `data_dir` | `./.trendai-data` | store root (created `0700`) |
| `TRENDAI_PUBLIC_URL` | `public_url` | `None` | e.g. `https://trendai.example`; Wikimedia UA, Origin check |
| `TRENDAI_CONTACT` | `contact` | `None` | operator e-mail/URL for the Wikimedia UA; **required when public** (`validate_public`, A.5), a warning on loopback |
| `TRENDAI_ALLOW_NO_CONTACT` | `allow_no_contact` | `false` | explicit opt-out of the contact requirement for private deployments |
| `TRENDAI_TRUST_PROXY` | `trust_proxy` | `false` | take the client IP from the last `X-Forwarded-For` entry and the own origin from `X-Forwarded-Proto`/`X-Forwarded-Host` (C.0) |
| — | `static_dir` | `None` | override only (tests): directory served as `/static`; `None` = the package's `static/` |
| `TRENDAI_DEV` | `dev` | `false` | serve `/static/dev/**`, rescan static files per request |
| `TRENDAI_LOG_LEVEL` | `log_level` | `INFO` | |
| `TRENDAI_WORKERS` | `workers` | `2` | concurrent analysis jobs |
| `TRENDAI_MAX_QUEUED` | `max_queued` | `8` | waiting analysis jobs before `503 server_busy` |
| `TRENDAI_JOB_TIMEOUT` | `job_timeout_s` | `180` | job is failed with `job_timeout` at the next stage boundary after this |
| `TRENDAI_UPLOAD_MAX_MB` | `upload_max_mb` | `10` | |
| `TRENDAI_UPLOAD_MAX_SERIES` | `upload_max_series` | `20` | |
| `TRENDAI_RESULT_TTL_H` | `result_ttl_h` | `24` | |
| `TRENDAI_SHARE_TTL_D` | `share_ttl_d` | `30` | |
| `TRENDAI_NAVER_MONTHLY_LIMIT` | `naver_monthly_limit` | `50000` | |
| `TRENDAI_NAVER_IP_PER_HOUR` | `naver_ip_per_hour` | `20` | |
| `TRENDAI_NAVER_CACHE_DAYS` | `naver_cache_days` | `21` | values > 21 are clamped to 21 |
| `NAVER_API_HUB_KEY_ID`, `NAVER_API_HUB_KEY`, `NAVER_CLIENT_ID`, `NAVER_CLIENT_SECRET` | — | — | read only by `connectors.NaverCredentials.from_env`; never stored in `WebConfig`, never logged |
| `TRENDAI_AI` | `ai` | `auto` | `auto` (on iff `anthropic` importable and `ANTHROPIC_API_KEY` set) · `on` · `off` |
| `TRENDAI_AI_MODEL` | `ai_model` | `claude-opus-5-5` | |
| `TRENDAI_AI_IP_PER_HOUR` | `ai_ip_per_hour` | `10` | |
| `TRENDAI_AI_DAILY_LIMIT` | `ai_daily_limit` | `200` | per KST day |
| `ANTHROPIC_API_KEY` | — | — | read only by the SDK |
| `TRENDAI_HSTS` | `hsts` | `false` | send HSTS (only behind TLS) |
| `TRENDAI_STORE_MAX_MB` | `store_max_mb` | `2048` | sweeper evicts oldest results above this |
| `TRENDAI_SHUTDOWN_GRACE` | `shutdown_grace_s` | `30` | |

Booleans accept `1/true/yes/on` and `0/false/no/off` (case-insensitive). Invalid values raise `ValueError` naming
the variable at start-up. `WebConfig` is frozen; `repr()` contains no secrets because it holds none.

### E.2 App factory

`create_app(config, fetchers=None, clock=None)`:

1. Resolve `WebConfig`; create `data_dir` and sub-dirs (`results`, `shares`, `cache/naver`, `cache/wiki`,
   `cache/results`, `budget`) with mode `0700`; check writability (health reports it).
2. `fetchers = fetchers or default_fetchers(config)`: Naver = `NaverCredentials.from_env()` → `None` when unset, else
   `functools.partial(connectors.fetch_naver_search_trend, credentials=..., session=UrllibSession())`;
   Wiki = `fetch_wikipedia_pageviews(..., session=UrllibSession(), user_agent=connectors.user_agent(public_url, contact))`
   (one shared `UrllibSession` is fine: it is stateless, B.3.1);
   LLM = `None` unless AI is on, else `lambda payload, lang: narrative.llm_narrative_from_payload(payload, lang,
   model=config.ai_model, client=anthropic.Anthropic(timeout=90.0, max_retries=1))` (client built once, C.11).
3. Build the static index (E.4), samples cache (C.3), store, rate limiter, Naver budget, AI budget, upload cache,
   upload parse semaphore, job manager (two executors, in-flight index), Naver fetch single-flight table, sweeper
   thread (`trendai-sweeper`, daemon, every 600 s and once at start).
4. Return `TrendaiApp`. Creating two apps in one process (tests) must work: no module-level mutable state at all
   (notice capture is per thread, E.13; the warnings baseline is installed by process owners, not by `create_app`).

### E.3 Router

`PATH_INFO` (already percent-decoded by the WSGI server) is rejected with `404` if it contains NUL, `\`, a `..`
segment or does not start with `/`. Routing is an ordered list of `(methods, regex, handler)`; the first regex
that matches the path decides; method mismatch → `405` with `Allow`.

| Methods | Path regex | Handler |
|---|---|---|
| GET, HEAD | `^/(explore\|upload\|samples\|method\|sources\|privacy)?$` | SPA `index.html` |
| GET, HEAD | `^/(r\|s)/[A-Za-z0-9_-]{22}$` | SPA `index.html` + `X-Robots-Tag: noindex, nofollow` |
| GET, HEAD | `^/(explore\|upload\|samples\|method\|sources\|privacy)/$` | `308` to the path without the slash (query kept) |
| GET, HEAD | `^/static/(.+)$` | static file (E.4) |
| GET, HEAD | `^/favicon\.ico$` | `/static/favicon.svg` content (`image/svg+xml`) |
| GET, HEAD | `^/robots\.txt$` | `User-agent: *\nDisallow: /api/\nDisallow: /r/\nDisallow: /s/\n` |
| GET, HEAD | `^/api/health$` | C.1 |
| GET | `^/api/config$` · `^/api/samples$` | C.2 · C.3 |
| POST | `^/api/upload$` · `^/api/analyze$` · `^/api/narrative$` · `^/api/share$` | C.4 · C.5 · C.11 · C.8 |
| GET, DELETE | `^/api/jobs/(?P<id>[A-Za-z0-9_-]{22})$` | C.6 |
| GET | `^/api/results/(?P<id>…)$` · `^/api/shared/(?P<sid>…)$` | C.7 · C.9 |
| DELETE | `^/api/share/(?P<sid>…)$` | C.8 |
| GET | `^/api/export/(?:s/(?P<sid>…)\|(?P<id>…))(?:/(?P<k>0\|[1-9][0-9]?))?\.(?P<fmt>csv\|xlsx\|html\|json)$` | C.10 |
| GET | `^/api/export/(?:s/(?P<sid>…)\|(?P<id>…))/values\.csv$` | C.10 values table |
| any | `^/api/` (anything else) | `404 not_found` (JSON) |
| GET, HEAD | anything else | `index.html` with status **404** (the SPA renders its not-found page) |

`TrendaiApp.__call__` wraps every handler: request id, security headers, `HEAD` → body dropped, `WebError` →
envelope, any other exception → `500 internal_error` + `logger.exception`, access log line. Responses are always a
single `bytes` chunk (no streaming), `Content-Length` set.

### E.4 Static files and SPA fallback

* Root = `config.static_dir` when set, else `importlib.resources.files("trendai.web") / "static"` (a real directory;
  installed as package data).
* At start-up the app walks the root and builds an index `{relative posix path: (absolute path, size, sha1 etag,
  mime)}` excluding `dev/` (unless `dev`). **Only paths in the index are served**; the request path is looked up
  as a dict key, never joined to the filesystem — traversal is impossible by construction. With `dev` the index is
  rebuilt on every static request.
* A missing root or a root without `index.html` is not fatal: the index is empty (or lacks `index.html`), a WARNING
  `static files missing: <root>` is logged once, SPA routes, the 404 page and `/favicon.ico` answer `503`
  `text/plain; charset=utf-8` `trendai: static files are missing` (with the security headers), `/static/*` → `404`,
  and the API works normally.
* MIME by extension: `.html text/html; charset=utf-8`, `.js text/javascript; charset=utf-8`, `.css text/css;
  charset=utf-8`, `.json application/json; charset=utf-8`, `.svg image/svg+xml`, `.png image/png`, `.woff2 font/woff2`,
  `.txt text/plain; charset=utf-8`, `.webmanifest application/manifest+json`. Other extensions are not indexed.
* `If-None-Match` equal to the ETag → `304` (no body). Caching per C.14.
* `index.html` is served from the same index for SPA routes (C.14 headers).

### E.5 Jobs (`jobs.py`)

```python
@dataclass
class Job:
    id: str; kind: str                      # 'analyze' | 'narrative'
    status: str = "queued"                  # queued | running | done | failed | cancelled
    stage: str = "queued"                   # queued | data | series | finish | done
    created: float; started: float | None; finished: float | None   # clock timestamps
    series_total: int = 0; series_done: int = 0
    current: dict | None = None             # {'index', 'name', 'step'}
    vms: list[dict]                         # finished series: {'index', 'vm': {'ko': ..., 'en': ...}}
    failed: list[dict]; notices: list[dict] # notices stored per language: {'ko': [...], 'en': [...]}
    compare: dict | None                    # {'ko': ..., 'en': ...}
    error: WebError | None
    cancel_requested: bool = False
    subscribers: int = 1                    # creator + C.5 joins − DELETEs; cancel only at 0 (C.6)
    cache_key: str | None = None            # result-cache key (None for uploads and narrative jobs)
    lock: threading.Lock
```

* `JobManager.submit(kind, fn, cache_key=None)` raises `WebError(server_busy, retry_after=10)` when `queued + running ≥
  workers + max_queued` (AI: 2 + 4). Jobs run on `ThreadPoolExecutor(max_workers=workers,
  thread_name_prefix="trendai-job")`. `JobManager.join(cache_key)` returns the queued/running analyze job with that
  key (subscriber count +1) or `None`; the in-flight index `{cache_key: job_id}` is updated under the manager lock
  and an entry is removed when its job becomes terminal.
* The job function receives a `JobContext` with `check()` (raises `JobCancelled` when `cancel_requested`, raises
  `JobTimeout` past `job_timeout_s`), `set_stage(stage, current=None)`, `set_step(step)` (maps the `on_stage` names to
  `current.step` per C.6 and sets it under the job lock) and `add_series(...)`. `check()` is called at every stage
  boundary, including from the `on_stage` callback passed to `pipeline.run`.
* GET snapshots are taken under the job lock and filtered by `since`/`hl`.
* Terminal jobs stay in memory 10 minutes, then are dropped (the result stays on disk).
* `close()`: no new submissions (`server_busy`, `Retry-After: 30`), queued jobs → `cancelled`, wait for running ones.

### E.6 The analyze job (`analyze.py`)

1. **Normalise** the request into a frozen `AnalyzeQuery` (C.5 rules): resolved `unit` → `freq` (`date`→`D`,
   `week`→`W`, `month`→`M`; samples `M`; uploads from `freq` or detection), integer `h` (`auto` →
   `data.default_horizon(freq)`), `(start, end) = connectors.period_bounds(range, unit, today)` with `today` = KST
   date (Naver) / UTC date (Wiki), display names (E.6.1).
2. **Stage `data`**:
   * Naver: one `NaverGroup(name=f"g{k+1}", keywords=terms)` per series (wire names, C.5); cache lookup (E.10) → hit:
     rebuild the series from JSON (attrs included), `as_of` = stored `fetched_at`, notice `naver_cached`; miss:
     enter the **fetch single-flight** for that `naver_cache_key` (a per-key lock in the app's in-flight table; the
     second job for the same key waits for the first and then re-reads the cache instead of calling), re-check the
     cache, then `budget.reserve()` (E.9) and `fetchers.naver(groups, start, end, unit, dev or None, gender or None,
     tuple(ages))`, store the cache entry. Results are mapped back **by index**: `result.series[f"g{k+1}"]` is
     series `k`, `result.filled[f"g{k+1}"]` its fill count. A `ConnectorError` fails the job (mapped per C.12.3).
     The per-IP `naver` token is consumed at submission.
   * Wiki: per series `fetchers.wiki(title, wl, start, end, granularity)` with cache (E.10); a `ConnectorError`
     fails **that series** (`failed` entry with mapped error); `invalid_param`-class errors fail the job.
   * Sample: `samples.sample(name)` (n = 120, monthly).
   * Upload: upload-cache entry (`410 upload_expired` if gone) → `frame_to_series(entry.raw.frame, date_col,
     value_cols, prefer_decimal_comma=entry.raw.decimal_comma)` inside `with data.collect_notices() as notes:`
     (C.12.4 mapping); for wide tables `date_col=None` and `value_cols` filters the returned dict.
   * Each raw series: `with data.collect_notices() as notes: prepared, f = data.prepare(raw, freq=freq or None,
     name=display)` (`prepare` returns a `(series, freq)` tuple); then **`prepared.attrs.update(raw.attrs)`** so that
     `samples.SYNTHETIC_ATTR` (and the Naver/GT source attrs) survive — `prepare` returns a series with empty
     `attrs`, and `pipeline.run` marks a sample as synthetic only when the series it receives carries the attr.
     `notes` → series notices (`render_web_msg`, both languages). `DataError` → series failure (C.12.4).
   * Extra series notices: `data_short` (n < 8), `mostly_zero_volume` (Naver/Wiki, ≥ 50 % zeros),
     `naver_missing_filled` (`filled > 0`), `naver_low_resolution` (multi-series Naver, `max < 5`),
     `horizon_long` when `h > 5·n` (created here with `render_web`).
3. **Stage `series`**, for each prepared series in index order: `ctx.check()`;
   `with data.collect_notices(): report = pipeline.run(prepared, name=display, freq=f, horizon=h, lang="ko",
   on_stage=lambda st: (ctx.check(), ctx.set_step(st)))` — the collected list is **discarded by code**: everything
   `run` emits (its internal `prepare` on already-prepared data, and `horizon_long`, which WP1 routes through
   `data.emit_notice` in `pipeline.py`, E.16) is dropped, independent of timing or other threads;
   English narrative `narrative.generate_narrative(a, f, o, lang="en")[0]`; VMs ko/en (D.2; `slot` provisional);
   JSON exports ko/en (`report_to_dict` + `source` block, Naver stripping C.10); LLM payload (non-Naver)
   `narrative.build_llm_payload(a, f, o)`; HTML (non-Naver) `report.render_html(report)` + the source paragraph
   (C.10). Any exception → `failed` entry (`DataError` → C.12.4, else `series_failed_web` + an exception log line per
   E.11). `series_done += 1` after each series (failed or not).
4. **Stage `finish`**: if every series failed → job `failed` with the first series' error (or `data_too_short`).
   If 2+ were requested: `compare.rank_trends({display name: report})` with the `{display name: index}` map (D.4.1),
   slots (D.4.2), compare VMs ko/en (incl. `source_line`), and (non-Naver, 2+ succeeded)
   `report.render_compare_html(list(reports.values()), ranking)` + source paragraph. Write the result (E.7), register
   the result-cache key (non-upload), drop the `TrendReport` objects, status `done`.
5. `JobCancelled` → status `cancelled`, nothing written; `JobTimeout` → `failed` with `job_timeout`.

**E.6.1 Display names** (unique within a request): base name = `terms[0]`, or `" + ".join(terms)` when another series
of the request has the same first term; uploads use the column (or wide-row) name (already unique, B.3.1); samples
the sample name. Then, in request order, a base name already taken by an earlier series becomes `"{base} ({n})"` with
the smallest `n ≥ 2` that is free (e.g. a literal term `a + b` next to `a`+`b` → `a + b`, `a + b (2)`). The display
name is the VM `name`, the `pipeline.run` name, the `rank_trends` key and the export names; it never goes to Naver.

### E.7 Store (`store.py`)

```
$TRENDAI_DATA_DIR/                        mode 0700, files 0600
  results/<id>/result.json                {schema, engine, result_id, created_at, expires_at, query, source_kind, view,
                                           failed: {ko, en}, available: {series: [{csv, xlsx, html, json}…],
                                           result: {csv, xlsx, html, json, values_csv}}, ranking_rows: [...],
                                           html: {series: ["report-0.html" | null, ...], compare: "compare.html" | null}}
  results/<id>/vm-ko.json · vm-en.json    {series, compare, notices} per language (GET reads only the one asked for)
  results/<id>/export-ko.json · export-en.json   JSON exports (report_to_dict + source) per language
  results/<id>/report-<k>.html            render_html (non-Naver)
  results/<id>/compare.html               render_compare_html (non-Naver, 2+ series)
  results/<id>/llm-<k>.json               build_llm_payload (non-Naver)
  results/<id>/ai/<k>-<lang>.json         AI narratives
  shares/<sid>/share.json                 {schema, share_id, created_at, expires_at, delete_token_sha256,
                                           result: <result.json with result_id removed>}
  shares/<sid>/vm-ko.json, vm-en.json, export-ko.json, export-en.json, report-<k>.html, compare.html
  cache/naver/<key[:2]>/<key>.json        {key, stored_at, expires_at, fetched_at, endpoint, time_unit, start, end,
                                           groups: [{name, keywords, dates, ratios, filled}], filters}
  cache/wiki/<key[:2]>/<key>.json         {key, stored_at, expires_at, fetched_at, title, lang, granularity, dates, values}
  cache/results/<key>.json                {key, result_id, expires_at}
  budget/naver-YYYY-MM.json               {month, count, updated_at}       (KST month)
  budget/ai-YYYY-MM-DD.json               {day, count, updated_at}         (KST day)
```

* Writes are atomic: `os.open(tmp, O_WRONLY|O_CREAT|O_EXCL, 0o600)` → write → `fsync` → `os.replace`. Directories
  `os.makedirs(..., mode=0o700)`. Reads treat a file with `expires_at < now` — or a result/share whose `schema` ≠
  `WEB_VM_SCHEMA` — as absent and delete expired ones best-effort.
* **Paths are built only from validated values**: ids that matched `^[A-Za-z0-9_-]{22}$`, series indexes that are
  `int` (not `bool`) within range, the literal languages `ko`/`en`, format literals from a fixed set, and sha256 hex
  cache keys. No request string (a `lang`, a filename, a term) is ever joined into a path; `store.py` asserts the
  final path stays under its root (`os.path.commonpath`) as a second line of defence.
* **Stored VMs contain no URLs and no ids**: `exports` are stored as availability flags (`available` above, with the
  `unavailable` reasons in the VM), and the API builds `/api/export/<id>/…` or `/api/export/s/<share_id>/…` links for
  each response. A share snapshot therefore never carries the `result_id`.
* **Nothing derived from an upload is written before the job finishes**, and raw upload bytes are never written by
  the app.
* Sweeper: deletes expired results/shares/cache entries and unreadable files older than 1 h; budget files older than
  400 days; when the store exceeds `store_max_mb`, deletes the oldest result dirs until under 90 % (shares and
  budget files are never size-evicted).

### E.8 Rate limiter (`ratelimit.py`)

Token bucket per `(bucket, client_key)`: `tokens = min(burst, tokens + elapsed × refill/3600)`; a request needs 1
token; refusal → `retry_after = ceil((1 − tokens) × 3600 / refill)`. `client_key` = `REMOTE_ADDR`, or with
`trust_proxy` the last `X-Forwarded-For` entry; IPv6 addresses are keyed by their /64; the `poll` bucket key is
`(client_key, job_id)` (C.13). One lock per limiter; at most
50,000 keys (the least recently used dropped); the sweeper drops buckets idle > 1 h. The clock is the app clock.
The `naver` bucket refill switches to 5/h while `budget_state != "ok"`.

### E.9 Naver budget (`budget.py`)

`NaverBudget(store, clock, limit)`: `state() -> (count, limit, "ok"|"low"|"blocked")`; `reserve()` under a process
lock: reload the current KST month file, raise `WebError(naver_budget_exhausted)` when `count ≥ 0.9 × limit`, else
`count += 1` and write atomically **before** the HTTP call (failed calls count). A new month starts at 0.
`AiBudget` is the same per KST day against `ai_daily_limit` (no 90 % rule: blocks at 100 %).

### E.10 Cache keys and TTLs

| Cache | Key | TTL |
|---|---|---|
| Naver data | `connectors.naver_cache_key(groups, start, end, unit, dev, gender, ages)` | 24 h when `end` equals `period_bounds(...)[1]` for today (the usual case), else `naver_cache_days` (≤ 21 d) |
| Wikipedia data | sha256 of canonical JSON `{"title", "lang", "granularity", "start", "end"}` | monthly 24 h; daily/weekly 6 h |
| Result | sha256 of canonical JSON `{"engine", "vm_schema", "source", "series": [[terms]…], "start", "end", "unit", "h", "dev", "gender", "ages", "wl"}` (`vm_schema` = `WEB_VM_SCHEMA`, so a VM-builder change never serves stale cached VMs even while `trendai.__version__` stays `0.1.0`) | until `min(result expires_at, data cache entry expires_at)`; never for uploads. Also the key of the in-flight index (C.5). |

Canonical JSON = `json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)`, dates as ISO strings.

### E.11 Logging (`log.py`)

* Loggers `trendai.web`, `trendai.web.access`, `trendai.web.jobs`. `create_app` adds no handlers; `trendai serve`
  calls `logging.basicConfig(level=config.log_level, format="%(asctime)s %(levelname)s %(name)s %(message)s")` with UTC times.
* Access line: `rid=<8hex> ip=<12hex> <METHOD> <route template> <status> <ms>ms <bytes>B` — the route template is
  the router pattern name (`/api/results/{id}`), never the raw path or query string. `ip` = first 12 hex of
  `sha256(daily_salt + client_key)`, the salt random per process and rotated per KST day.
* Job line: `job=<id[:6]>… kind=analyze source=naver series=3 naver=fresh|cache|- status=done ms=2810 error=-`.
* Never logged: API keys or any header value, request/response bodies, query strings, search terms, upload
  filenames or content, full ids or tokens. Connector failures log `code` and `log_detail` only.
* Tracebacks of unexpected errors are logged with the request id (no local variables). **Exception messages are not
  logged for upload parsing or for jobs on upload data** (pandas `ParserError` quotes input lines, `ValueError` quotes
  cell text): those lines carry `type(exc).__name__`, the mapped web code and the frames from
  `traceback.format_tb(exc.__traceback__)` only — never `str(exc)` or `logger.exception`'s formatted message.

### E.12 Security headers

Every response:

```
Content-Security-Policy: default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data: blob:; font-src 'self'; connect-src 'self'; object-src 'none'; base-uri 'none'; form-action 'self'; frame-ancestors 'none'
X-Content-Type-Options: nosniff
X-Frame-Options: DENY
Referrer-Policy: no-referrer
Permissions-Policy: camera=(), microphone=(), geolocation=(), payment=(), usb=()
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Resource-Policy: same-origin
```

plus `Strict-Transport-Security: max-age=31536000` when `hsts`. HTML exports replace the CSP with the sandboxed one
of C.10. No inline scripts or styles exist in the app (F.1 rules), so no `unsafe-inline` is ever needed.

### E.13 Threading model

* `trendai serve` = `wsgiref` + `ThreadingWSGIServer` (one daemon thread per connection, request handler
  `timeout = 30` s against slow clients, default request logging silenced — E.11 logs instead). For public
  deployment `docs/WEB.md` recommends a reverse proxy (TLS, connection limits) or waitress/gunicorn.
* **One process only**: the job registry, upload cache and rate limiter are in memory. gunicorn
  `-w 1 --threads 8 trendai.web.wsgi:application`; waitress `--threads=8 trendai.web.wsgi:application`.
* Analysis jobs: thread pool of `workers` (2); AI jobs: thread pool of 2. Request threads never run analyses;
  they parse uploads (bounded to 2 at a time, C.13).
* **Capacity**: the engine is CPU-bound Python under the GIL. Measured on 4 CPUs: 8 weekly series × 260 points take
  ≈ 5.8 s sequentially and *not less* with 2 or 4 threads; one analysis takes 0.4–1.0 s up to 3,900 points. So one
  process handles roughly 15–20 five-series weekly comparisons per minute. `workers = 2` buys fairness (a short job
  is not stuck behind a long one), not throughput. Mitigations in this spec: result cache, in-flight joining of
  identical queries (C.5), Naver fetch single-flight (E.6), upload parse limit; the numbers go in docs/WEB.md. A
  process pool is deliberately not used in the MVP (J).
* **Data notices are captured per thread, never through `warnings`.** `warnings.catch_warnings` swaps
  process-global state on Python 3.12/3.13 (and on 3.14 unless the `context_aware_warnings` flag is on, which it is
  not by default outside free-threaded builds), so any design that records warnings while other threads run can mix
  one user's notices into another's, or leave a recorder/filter installed for good when third-party code (pandas
  calls `catch_warnings` in ~40 places) interleaves. Therefore: every web call that must capture notices
  (`read_table_bytes`, `frame_to_series`, `prepare`) runs inside `data.collect_notices()` (a `ContextVar`-backed sink in
  `data.emit_notice`, B.3.1), which bypasses `warnings` entirely; `pipeline.run` runs inside a discarded
  `collect_notices()` block (E.6 step 3). WP1 never calls `warnings.catch_warnings`, `simplefilter` or
  `showwarning` from request or job threads. There is no `CAPTURE_LOCK`.
* **Warnings baseline**: `install_warning_baseline()` (B.3.2) is run by the process owner — `trendai serve` (after
  `cli._install_warning_filters`, B.5), the lazy `trendai.web.wsgi.application` and `tests/e2e/server.py` — never by
  `create_app` (tests must keep pytest's warning capture). It ignores every warning category (nothing reaches
  stderr; data notices do not use `warnings` anyway), and the sweeper re-asserts it on every run, so filter state
  left behind by interleaved third-party `catch_warnings` blocks (including `data.py`'s own `catch_warnings` around
  pandas date parsing) lasts at most one sweeper period and can never route notices anywhere.

### E.14 Graceful shutdown

`serve_forever` installs SIGINT/SIGTERM handlers. First signal, in this order:
1. `draining = true`: health answers `503` with `status: "draining"`, new analyze/narrative/upload requests →
   `503 server_busy` with `Retry-After: 30`; queued jobs are cancelled; **the listener keeps serving** (polls, result
   GETs, exports) so running jobs can still be watched and fetched.
2. A helper thread waits until no job is running or `shutdown_grace_s` has passed (polling every 0.2 s), then calls
   `app.close(timeout=0)` (stops the sweeper; running jobs, if any, are abandoned and logged at WARNING) and
   `server.shutdown()`.
3. `serve_forever` returns 0; the caller flushes stdout/stderr/logging and calls `os._exit(0)`, so non-daemon executor
   threads (a pandas computation, an AI call bounded by its 90 s timeout) cannot hold the process past the grace
   period or until Docker's SIGKILL.
A second signal at any time → flush and `os._exit(1)` immediately (a normal `sys.exit` would wait for executor
threads). The budget files are already on disk (written per call), so nothing needs flushing.

### E.15 Exports (`exports.py`)

* CSV via `csv.writer(lineterminator="\r\n")` into a `StringIO`, encoded `utf-8-sig`. Numbers: `format(v, ".15g")`;
  `None` → empty cell. Apostrophe formula guard (C.10) on every text cell. Source + condition comment lines from the
  VM (series) or `compare.source_line` (whole result, `values.csv`).
* XLSX via `openpyxl.Workbook()`; built in memory (`BytesIO`). Same data; text cells written with
  `data_type = "s"` (no apostrophe, C.10); sheet names per C.10.
* JSON from the stored `exports.json[hl]`; HTML from the stored files; share variants read the share snapshot.

### E.16 Hooks in existing modules (WP1)

```python
# trendai/pipeline.py
def run(series, name=None, freq=None, horizon=None, levels=(80, 95), use_llm=False, lang="ko",
        model=DEFAULT_LLM_MODEL, candidates=None, client=None,
        on_stage: Optional[Callable[[str], None]] = None) -> TrendReport:
    # on_stage(name) is called immediately before each step: "prepare", "analyze", "forecast", "outlook",
    # "narrative". Exceptions it raises propagate unchanged. Default None: behaviour and output unchanged.
    # _check_horizon's `warnings.warn(msg("horizon_long", ...), DataWarning, stacklevel=3)` becomes
    # `data.emit_notice(msg("horizon_long", ...), DataWarning, stacklevel=3)`: identical outside a
    # collect_notices() block (same category, text and reported caller line), collected/discarded inside one.

# trendai/narrative.py
def llm_narrative_from_payload(payload: dict, lang: str = "ko", model: str = DEFAULT_MODEL,
                               client: Any = None) -> str:
    """Claude narrative from a stored build_llm_payload(...) dict: exactly the prompt and call of llm_narrative
    (llm_narrative now builds the payload and delegates here). Raises NarrativeUnavailable."""
```

Both are added to the modules' `__all__`/docs; all existing tests must still pass unchanged.

### E.17 Uploads (`uploads.py`)

* Multipart: Python 3.13 has no `cgi`; parse with `email.parser.BytesParser(policy=email.policy.HTTP).parsebytes(
  b"Content-Type: " + header + b"\r\n\r\n" + body)`; take the first part whose `Content-Disposition` has
  `name="file"` and a filename; `get_payload(decode=True)` gives the bytes. Missing boundary/part → `400 bad_request`.
* Filename for display: last path component (split on `/` and `\`), NFC, control characters removed, ≤ 100 chars;
  the lower-cased extension selects the parser (unknown → text).
* Concurrency: the whole parse (multipart decode → detection) runs while holding one of the 2 slots of the app's
  `BoundedSemaphore(2)` (`acquire(blocking=False)`; none free → `503 server_busy`, `Retry-After: 5`). Peak memory is
  therefore bounded to about two 10 MB parses (≈ 180 MB each for a dense CSV).
* Zip guard (bytes starting `PK\x03\x04`): `zipfile.ZipFile(BytesIO(raw))`; total uncompressed size > **30 MB** or
  > 1,000 members → `413 upload_archive_too_large`; `BadZipFile` → `422 upload_unreadable`.
* `read_table_bytes(raw, name, max_columns=5000)`: a header wider than 5,000 fields is refused before the full
  parse (`table_too_many_columns` → `422 upload_too_many_columns`). Then the table limits of C.4 by orientation:
  `frame.size > 2,000,000` → `422 upload_too_large_table`; after detection, long tables > 50 columns, wide tables
  > 200 rows or > 5,000 date columns → `422 upload_too_many_columns`.
* Detection: `frame_to_series(raw.frame, prefer_decimal_comma=raw.decimal_comma)` inside `collect_notices()`; date
  columns from the first series' `attrs['trendai']['date_cols']` (all marked `date`; `detected.date_col` =
  `date_cols[0]`, C.4); excluded columns from a captured `rownum_cols_skipped` notice (`params['cols']`);
  `numeric_share` via `data.parse_numbers` (it accepts the native Excel cell types too); `freq` via `data.infer_freq`
  on the parsed dates.
* Preview: cells of `raw.frame.head(5)` / `.tail(5)` stringified per C.4 (`preview` bullet); `RawTable.frame` itself
  is never converted (Excel keeps native types, B.3.1).
* Upload cache: `{upload_id: (RawTable, created_at, filename, size_bytes, est_bytes)}`, TTL 30 min, LRU-evicted
  above 64 entries or 256 MB. The `bytes` object is not kept.

### E.18 Private names WP1 may import (one source of truth with the HTML report)

The VM must word undefined values exactly like the report and the narrative. WP1 therefore imports these existing
private helpers (read-only use; WP4 lists them in `docs/ARCHITECTURE.md` › "Cross-module private dependencies"):

| Name | Used for |
|---|---|
| `report._ChartData(report)` | argument of the helpers below |
| `report._change_basis(report, d)` / `report._change_basis_en(report, d)` | `hero.basis`, sentence level names (D.2.3) |
| `report._change_undefined(report, d, lang)` → `.code`, `.short`, `.sentence`, `.levels` | undefined expected change (D.2.3, tile 4) |
| `report._pct_why(code, key, lang)` | undefined recent change (tile 2) |
| `report._rate_why(report, code, lang)` | undefined momentum / volatility (tiles 3, 6) |
| `report._recent_lag(report, n)` | recent-change basis wording (tile 2) |

Everything else comes from public functions (`narrative.direction_label`, `format_prob`, `prob_reason`,
`metric_pct_undefined`, `momentum_pct_undefined`, `volatility_undefined`, `phase_is_tentative`, `is_intermittent`,
`is_all_zero`, `is_synthetic`, `scenario_reference`, `scenario_value`, `scenario_floor`, `format_date`,
`describe_method`, `model_label`,
`future_phase_status`, `translate_reason`, `PHASE_EN`, `SECTION_HEADINGS`, `build_llm_payload`,
`generate_narrative`; `analysis.changepoint_label`; `data.periods_per_year`, `season_length`, `default_horizon`;
`types.PHASE_KO`; `pipeline.report_to_dict`; `compare.rank_trends`; `report.render_html`, `render_compare_html`).

---
## F. Frontend architecture (WP3)

### F.1 Files and hard rules

```
trendai/web/static/
  index.html                 the only HTML page (SPA shell; static, no templating)
  favicon.svg
  img/icons.svg              SVG sprite: <symbol> ph-emerging … ph-volatile, slot-circle … slot-cross, ui-* icons
  css/tokens.css             F.8 (exact values)
  css/base.css               reset, typography, layout grid, utilities, focus ring, skip link
  css/components.css         every component
  css/chart.css              chart classes (lines, bands, axes, badges, crosshair)
  js/theme-init.js           classic script in <head>: applies stored theme/updown before first paint
  js/app.js                  entry module: loads config, mounts topbar/footer, starts router
  js/router.js               History API router, link interception, title + focus management
  js/state.js                URL ⇄ state (pure; F.2)
  js/api.js                  fetch wrapper, ApiError, job polling, cancel (F.3)
  js/i18n.js                 t(), hl resolution, <html lang> (pure except setLang)
  js/catalog.js              export const CATALOG = {ko: {...}, en: {...}} (G)
  js/fmt.js                  D.1 mirror (pure)
  js/store.js                localStorage wrappers with try/catch (F.11)
  js/dom.js                  h(tag, attrs, ...children) builder, svg(tag, attrs), clear(), focusFirst()
  js/md.js                   Markdown-lite → DOM nodes (pure builder returning a DocumentFragment)
  js/ime.js                  shouldCommit(event): IME-safe commit test for the search box (pure)
  js/pages/home.js · explore.js · result.js (/r, /s) · upload.js · samples.js · info.js (/method /sources /privacy) · notfound.js
  js/views/detail.js · compare.js
  js/components/topbar.js · footer.js · searchbox.js · chipeditor.js · querybar.js · sheet.js · popover.js ·
                progress.js · errorcard.js · notices.js · hero.js · tiles.js · phasewhy.js · past.js · future.js ·
                narrative.js · caveats.js · rankcards.js · datatable.js · sharemenu.js · sharedialog.js · toast.js ·
                help.js · sourceline.js · settings.js · recent.js · gtguide.js
  js/chart/scale.js (pure) · axes.js · fanchart.js · comparechart.js · sparkline.js · crosshair.js · png.js · describe.js
  fonts/pretendard-1.3.9/pretendardvariable-dynamic-subset.css   (from the npm tarball, unmodified)
  fonts/pretendard-1.3.9/woff2-dynamic-subset/PretendardVariable.subset.0.woff2 … .91.woff2 (92 files)
  fonts/pretendard-1.3.9/LICENSE.txt                              (dist/LICENSE.txt, SIL OFL 1.1)
  dev/mock.html · dev/mock.js · dev/fixtures/*.json               (dev only; never shipped or served in production)
```

Hard rules (reviewed in I):

1. No build step, no third-party JS/CSS, no CDN, no inline `<script>`/`<style>`, no `on*=` attributes, no
   `setAttribute('style', …)`, no `innerHTML`/`outerHTML`/`insertAdjacentHTML` with anything but the empty string.
   Text goes in via `textContent`; styles via classes or `el.style.setProperty()` (CSSOM is CSP-safe).
2. Modules that `tests/js` imports (`state.js`, `fmt.js`, `i18n.js`, `catalog.js`, `md.js`, `ime.js`, `chart/scale.js`)
   touch no DOM/`window`/`localStorage` at import time and export pure functions.
3. Every user-visible string comes from `catalog.js` or the VM; no hard-coded copy in components.
4. Every request goes through `api.js`; no other `fetch` call sites.
5. Fonts: obtained from the npm registry tarball `https://registry.npmjs.org/pretendard/-/pretendard-1.3.9.tgz`
   (verify the sha512 in A.5), copied unmodified from `package/dist/web/variable/` and `package/dist/LICENSE.txt`.

`index.html` (complete):

```html
<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>trendai</title>
  <meta name="description" content="과거를 풀고, 지금 국면을 진단하고, 앞으로를 범위로 예측해요.">
  <meta name="color-scheme" content="light dark">
  <link rel="icon" href="/static/favicon.svg" type="image/svg+xml">
  <link rel="stylesheet" href="/static/fonts/pretendard-1.3.9/pretendardvariable-dynamic-subset.css">
  <link rel="stylesheet" href="/static/css/tokens.css">
  <link rel="stylesheet" href="/static/css/base.css">
  <link rel="stylesheet" href="/static/css/components.css">
  <link rel="stylesheet" href="/static/css/chart.css">
  <script src="/static/js/theme-init.js"></script>
  <script type="module" src="/static/js/app.js"></script>
</head>
<body>
  <a class="skip-link" href="#main">본문으로 건너뛰기</a>
  <div id="app">
    <header id="topbar" class="topbar"></header>
    <main id="main" tabindex="-1"></main>
    <footer id="footer" class="footer"></footer>
  </div>
  <div id="live-polite" class="sr-only" aria-live="polite"></div>
  <div id="live-assertive" class="sr-only" aria-live="assertive"></div>
  <noscript><p class="noscript">trendai는 JavaScript가 있어야 동작해요. · trendai needs JavaScript.</p></noscript>
</body>
</html>
```

### F.2 Router and URL state (`router.js`, `state.js`)

Routes: `/` home · `/explore` results (1 series → detail, 2+ → compare) · `/upload` · `/r/<id>` · `/s/<id>` ·
`/samples` · `/method` · `/sources` · `/privacy` · anything else → not found. `/explore` without `q` → replace with `/`.
`/` accepts `src` and `q` as a **pre-fill** of the source switch and the search box (chips), never starting an
analysis (used by the `use_wiki` fix, C.12.6).

**Query parameters** (brief §3.1): `src q range unit h dev gender ages wl hl scale band view focus`.

| Param | Values | Default (omitted from the URL when default) |
|---|---|---|
| `src` | `naver` `wiki` `sample` | `config.default_source` (written explicitly when the user chose it) |
| `q` | grammar below | — (required on `/explore`) |
| `range` | `1y` `3y` `5y` `all` `YYYY-MM-DD~YYYY-MM-DD` | `5y` |
| `unit` | `auto` `date` `week` `month` | `auto` |
| `h` | `auto` or integer | `auto` |
| `dev` / `gender` / `ages` | `pc`/`mo` · `m`/`f` · comma list of `1`–`11` | empty |
| `wl` | `[a-z]{2,3}` | `ko` |
| `hl` | `ko` `en` | not written unless the user switched language |
| `scale` | `raw` `index` `log` | the compare VM's `scale.default` |
| `band` | `80` `95` `both` `off` | `both` |
| `view` | `chart` `table` | `chart` |
| `focus` | `encodeURIComponent(series name)` | — |

**`q` grammar**: `q = series *("," series)`, `series = term *("+" term)`, each `term` written with
`encodeURIComponent` (so literal `,` `+` `&` `#` and spaces inside a term are `%2C` `%2B` `%26` `%23` `%20`).
Parsing reads `location.search` **raw** (not `URLSearchParams`, which turns `+` into a space): split on `&`, take
`q=`, split on `,` then `+`, `decodeURIComponent` each piece, trim, drop empties, keep at most 5 series. All other
params are parsed with `URLSearchParams` (their values never contain `+`). Serialisation order:
`src, q, range, unit, h, dev, gender, ages, wl, hl, scale, band, view, focus`.

`state.js` exports `parseLocation(pathname, search) → State`, `serialize(State) → string` (path + query),
`queryKey(State) → string` (canonical key of the analysis-relevant fields: `src, q, range, unit, h, dev, gender,
ages, wl`), `toAnalyzeBody(State) → C.5 body`. Round-trip `parse(serialize(s)) == s` is tested (H.3).

**History**: changing an analysis-relevant field → `pushState` + new analysis; `focus` → `pushState` (back returns
to the comparison); `scale`, `band`, `view`, `hl` → `replaceState` (no new history entry, no new analysis).
`popstate` re-renders from the URL; finished results are cached in memory by `queryKey` + `hl`, so back/forward
never re-polls. On every route change: scroll to top, set `document.title` (`{name} · trendai`, `{a} vs {b} · trendai`,
page titles from G), move focus to the page `<h1>` (`tabindex="-1"`), announce the title in `#live-polite`.
Internal links are `<a href>` elements; the router intercepts same-origin left clicks without modifier keys.

### F.3 API client (`api.js`)

* `request(method, path, {json, body, signal})` → parsed JSON; sends `Accept: application/json` and, with `json`,
  `Content-Type: application/json`. Non-2xx → throws `ApiError {code, status, message, fixes, retryAfter, params}`
  with `message`/fix labels picked for the current `hl` from the envelope. Network failure / non-JSON → `ApiError`
  with `code: "client_network"` (text from the UI catalog `err.client_network`).
* `analyze(state, {signal, onUpdate})`: `POST /api/analyze` then polls `GET /api/jobs/<id>?since=<n>&hl=<hl>` every
  `poll_after_ms` (min 300, max 2,000) with `n` = 1 + the highest index seen in `series` or `failed` (C.6), calling
  `onUpdate(snapshot)` with the accumulated VMs (de-duplicated by `index`); on `429` waits `retryAfter`; when
  `status == "done"` it fetches `result_url` once and resolves with that envelope (authoritative slots, exports,
  query, expiry); rejects with `ApiError` on `failed`, and with `{cancelled: true}` on `cancelled`. Aborting the
  signal stops polling and sends `DELETE /api/jobs/<id>` (`keepalive: true`) if the job is not terminal (a joined job
  keeps running for its other subscribers, C.6). Leaving the page (route change) aborts. A `naver_rate` failure is
  retried automatically at most once per query (C.12.6).
* `upload(file | text)`, `result(id, hl)`, `shared(id, hl)`, `share(resultId)`, `deleteShare(id, token)`,
  `narrative(resultId, k, lang)` (+ polling like `analyze`), `config()`, `samples(hl)` — thin wrappers.
* Client-side pre-checks (same limits as the server): file > 10 MB → error card `body_too_large` without
  uploading; 6th series → toast `too_many_series`; term > 50 chars → inline error `term_too_long`.

### F.4 i18n (`i18n.js`, `catalog.js`)

`t(key, params?)` looks up `CATALOG[hl][key]`, falls back to `CATALOG.ko[key]`, then to the key (and
`console.warn` once). Placeholders `{name}`. `hl` resolution: URL `hl` → `localStorage['trendai.hl']` → any
`navigator.languages` entry starting with `ko` → `ko`, else `en`. Switching language: store it, `replaceState`
with `hl`, set `<html lang>`, set `data-updown` default (F.8) when the user has no explicit up/down setting,
re-render chrome, re-fetch the current result in the new language (`GET /api/results/<id>?hl=…`, cached per hl).
Server texts (VM, errors) arrive already in the chosen language (`message_ko`/`message_en` picked in `api.js`).

### F.5 Formatting (`fmt.js`)

Exports `roundHalfAway(x, d)`, `compact(x, lang)`, `full(x, lang)`, `valueText(x, kind, lang, compact)`, `pct(p)`,
`pctArrow(p)`, `pp(p, lang)`, `probText(p, lang)`, `probLabelled(p, lang)`, `dateLabel(iso, freq, lang)`,
`dateShort(iso, freq, lang)`, `rangeLabel(a, b, freq, lang)`, `axisLabel(ms, step, first, lang)`. Exactly D.1, with
all display rounding through `roundHalfAway` on the exact decimal expansion (`toFixed(100)` digit string), never
`toFixed(d)`/`Math.round`; `tests/js/fmt.test.mjs` asserts every D.1.7 row, the tie rows included.

### F.6 Components and pages

**F.6.1 Top bar** (`topbar.js`): wordmark `trendai` (link `/`), nav `탐색` (`/`), `내 파일` (`/upload`), `샘플`
(`/samples`), `방법` (`/method`); right side: language toggle `KO | EN` (two buttons, `aria-pressed`) and the theme
button (cycles system → light → dark; label shows the current mode; `aria-label` from G). Below 640 px the nav
collapses into a `메뉴` button opening a sheet (F.6.6).

**F.6.2 Home** (`home.js`, brief §3.3): `<h1>` hero text, sub text, search box (F.6.3), source switch
`[네이버 검색] [위키백과] [내 파일] [샘플]` as a radio group (`role="radiogroup"`; `내 파일` navigates to `/upload`,
`샘플` switches the input to sample chips), hint line, example chips **per source** — `home.example_terms.naver`
(`선크림 패딩 수능 김장 다이어트`) when Naver is selected, `home.example_terms.wiki` (`자외선 차단제 대학수학능력시험 김장
장마 인공지능`, all verified ko.wikipedia article titles with real page views) when Wikipedia is selected; the chips
swap when the source changes (click → explore with the current source), upload card (drop zone + `파일 선택` +
`붙여넣기`, → `/upload` with the file/text handed over in memory), samples card (9 chips with the `가상 데이터` badge
→ `/explore?src=sample&q=<name>`), recent analyses (F.6.20). Naver unavailable: its switch button stays visible with label suffix `준비 중` and a `?` tooltip with the reason.
`reason: "unconfigured"` → `aria-disabled="true"` (not selectable; tooltip `source.naver.unconfigured`);
`reason: "budget_exhausted"` → still selectable (cached queries keep working; a cache miss returns
`503 naver_budget_exhausted`), tooltip `source.naver.budget`.

**F.6.3 Search box with chips** (`searchbox.js`): a text input plus chips for committed series. `Enter` or `,`
commits the current text as a series — **but never during IME composition**: key events with `e.isComposing ===
true` or `e.keyCode === 229` are ignored (Korean input fires `keydown` for a half-built syllable), and a `,` typed
during composition is handled on `compositionend`, so `선크림` is never committed as `선크` + `림`. `+` inside the text
splits it into terms of one series (shown in the chip
as `선크림 + 자외선차단제`); `Backspace` on an empty input focuses the last chip; chip `×` removes it. Clicking a chip
opens the chip editor (`chipeditor.js`, a dialog): one input "같은 뜻 검색어 (쉼표로 구분)" pre-filled with the
terms joined by `, `, max 5 terms (Naver), Save/Cancel — DataLab's comma = group rule lives only here.
`분석` button (or `Enter` on an empty input with ≥ 1 chip) navigates to `/explore`. A 6th series is refused with a
toast `too_many_series`. For `wiki`, `+` is not allowed (inline hint `wiki_group_unsupported`). For `sample`, the
input is replaced by the 9 sample chips (multi-select up to 5). Below 640 px focusing the input expands it to full
width and shows examples + recent analyses beneath (A.2).

**F.6.4 Query bar** (`querybar.js`, results pages, sticky): series chips (colour swatch + slot icon + name + `×`)
and `+ 비교` (opens an inline input; disabled at 5 with tooltip), then the default chips row:
`출처 ▾` `기간 ▾` (1년/3년/5년/전체/직접 입력 with two `<input type="date">` limited to `periods.<src>.earliest` …
`last_complete`) `단위 ▾` (자동/일/주/월) `예측 기간 ▾` (`limits.horizon_choices[freq]` + 자동) and, for Naver,
`상세 조건 ▾` (device radio 전체/PC/모바일, gender radio 전체/남성/여성, age checkboxes from `naver_ages`). Each chip shows
its current value (`기간: 최근 5년`). Changing a value updates the URL (pushState) and re-runs the analysis. Samples:
`기간`/`단위` chips are disabled with tooltip `chip.sample_fixed`. Results from uploads (`/r`, `/s`): chips are
read-only labels (period, unit, horizon) plus a `다시 분석` link to `/upload`. ≥ 1024 px: popovers
(`popover.js`, anchored, `Esc` closes, focus returns); < 640 px: every chip opens a bottom sheet (F.6.6), and once the
page scrolls past the bar's natural position the sticky bar **collapses to one line** (≤ 56 px tall, e.g.
`●선크림 ◆패딩 · 5년 · 주 ▾`) that opens a bottom sheet holding the full bar; it expands again at the top of the page.

**F.6.5 Progress** (`progress.js`, brief §3.7): during a job, the result layout is drawn as skeletons of the final
size (hero, 6 tiles, chart, cards). After 1 s a status line appears (`progress.fetching_naver` etc. by stage).
After 3 s, or immediately when `series_total ≥ 3`, a 4-step panel appears: ① `데이터 확인` (stage `data`) ②
`추세·구조 분석` (`current.step == "analyze"`) ③ `예측 모델 검증 (k/N)` (`forecast`/`narrative`, k =
`series_done`) ④ `정리` (`finish`); states pending/active/done with icons + text (not colour only); the panel is
`role="status"`. A `취소` button appears after 1 s; it aborts (F.3) and shows the `progress.cancelled` empty state
with `다시 분석`. Each finished series VM renders immediately (compare: its card and line; detail: the whole page).
`prefers-reduced-motion: reduce` → no shimmer. No fake progress bars.

**F.6.6 Sheet and popover** (`sheet.js`, `popover.js`): `<div role="dialog" aria-modal="true">` with a labelled
title, focus trap, `Esc` and backdrop close, focus returns to the opener; the sheet slides from the bottom on
< 640 px (no animation with reduced motion).

**F.6.7 Error and empty states** (`errorcard.js`): card with an icon, the message (`ApiError.message`), and up to
3 fix buttons (C.12.6 behaviours). Per-series failures in a comparison render as a card in the ranking grid with
the series name. Job-level errors replace the result area. Empty states use the G.9 copy (no results, cancelled,
not found, share expired).

**F.6.8 Compare view** (`views/compare.js`, brief §3.4): card `한눈에 보기` with header (title, `?` help, `⋯` menu
= CSV · 이미지 저장 · 링크 복사), segmented control `원래 값 | 기준 시점=100 | 로그` (disabled options show
`scale.disabled_reason` as tooltip; selection → `scale`), toggle `예측 범위 보기` (shows the 80 % band of the focused /
hovered series only), toggle `표로 보기` (`view=table`), the comparison chart (F.7.2), the source line
(`compare.source_line`, `sourceline.js`), the axis note (`compare.axis_note` with a `?` that opens `help.naver_index`
for Naver; index label under `index`) and result-level notices. Then `무엇이 뜨나요?` with sort select
(`상승 점수` default, `상승 가능성`, `예측 변화`, `최근 변화`, `현재 수준`; client-side using `sort.*`), the score note
with `?`, the spark note, and ranking cards (`rankcards.js`): rank, swatch+icon+name, phase badge (icon + text),
outlook line, `예측 변화 {text} ({basis})`, `최근 변화 {text} ({basis})`, sparkline, reliability chip,
`reference_only` chip `참고용`, low-resolution warning with the `이 검색어만 다시 분석` button, `자세히 ›` (→ `focus`).
Grid: 1 col < 640, 2 cols 640–1023, 3 cols ≥ 1024. Table view (`datatable.js`): ranking table (rank, name, phase,
expected change, P(rise), recent change, momentum, score) + a values table (date × series). The values table's
`CSV로 받기` is a plain link to the **server** export `exports.values_csv` (C.10: formula guard, source comment line);
the browser never builds a CSV. When `exports.values_csv` is `null` (Naver) the link is absent and the
`unavailable.values_csv.text` (`export_naver_raw`) is shown in its place. No table view offers a copy-to-clipboard
control for Naver data (A.3).

**F.6.9 Detail view** (`views/detail.js`, brief §3.5) — fixed card order:
header (`← 비교로` when `focus` is set, swatch+name, `synthetic` badge `가상 데이터`, `공유`, `내보내기 ▾`, `표`) ·
source line (`sourceline.js`, with `value_kind_note` under the chart; for Naver a `?` next to it opens
`help.naver_index`) · hero (`hero.js`: big phase badge with
`왜?` toggle, outlook line, sentence, reliability chip + accuracy line, `reference_note` chip) · tiles (`tiles.js`:
6 tiles, label + `?` (help.js) + value + sub; `undefined_reason` shown under `—`; 6 cols ≥ 1024, 3 cols 640–1023,
2 cols < 640) · chart card (`추이와 예측` + toggles `80%` `95%` (map to `band`) `추세선` `이상치` + fan chart F.7.1 +
legend line `chart.legend_bands` (`진한 띠: 10번 중 8번은 이 안 · 연한 띠: 20번 중 19번은 이 안`) + caption
`chart.band_caption` + `chart.y_note` when set) · `현재 — 왜 {phase}인가요?`
(`phasewhy.js`: reasons list ✓ with each `detail` under a `자세히` toggle, certainty label (its % only in the
`help.certainty` popover), link `국면 판단 규칙 보기 ›` to `/method#phase`) · `과거 — 무슨 일이
있었나요?` (`past.js`: numbered changepoint list ①②③ linked both ways with the chart badges, anomaly list,
seasonality line) · `미래 — 얼마나 확실한가요?` (`future.js`: scenario box headed by `scenarios.title`, the
`reference.text` row, then the three `rows[].text` and the `note`; collapsible forecast table, accuracy block with
`자세히` revealing `details` and the model list, method sentence, future phase, peak) · `해석` (`narrative.js`: badge,
Markdown-lite body, date links from `date_links`; `AI로 다시 쓰기` button when `ai_allowed`, else
disabled with `ai_block_text` visible under it; while the AI job runs the rule text stays and a status line shows
`narrative.ai_running`; result replaces the body with the AI badge + `note`; `규칙 기반 해석 보기` toggles back) ·
`주의할 점` (`caveats.js`: caveat markdown + fixed disclaimer).
Every card — compare card, chart card, tiles block, `현재`/`과거`/`미래`/`해석`/`주의할 점` — uses one card-head component
(`title · ? · ⋯`, brief §3.2). Prose blocks (narrative, reasons, caveats, `/method`, `/sources`, `/privacy`) are capped
at `max-width: 40em` (≈ 40 Hangul characters per line, brief §4.1).
The **first Naver result in either view** (detail or compare) shows a one-time dismissible explainer card
(`naver_explainer.*`, dismissal stored in `trendai.dismiss.naverIndex`); after dismissal the same text stays
reachable through the `?` next to `value_kind_note` / `compare.axis_note` (`help.naver_index`).

**F.6.10 Upload page** (`upload.js`, brief §3.6): step 1 — drop zone (keyboard-accessible button `파일 선택`,
`<input type="file" accept=".csv,.tsv,.txt,.xlsx,.xls,.json">`), `엑셀에서 붙여넣기` (dialog with a textarea, `붙여넣기`
`읽기` buttons), privacy note, supported-formats block (`id="formats"`) and the Google Trends guide card
(`gtguide.js`). There is no DataLab guide (A.2). Step 2 — confirm ("이렇게 읽었어요"): the encoding sentence from
`confirm.encoding.<utf8|cp949|utf16|other>` (`한국어 엑셀(CP949)로 읽었어요`, never a raw codec name with `(으)로`),
date-column `<select>` (all columns, `detected.date_col` selected; for a combination the `detected.date_cols` are
shown together, e.g. `연도 + 월`; wide tables show `날짜가 열 머리글에 있어요` instead; the request sends `date_col: null`
unless the user changed it, C.4), value-column checkboxes (the first 20 detected ones checked, excluded ones
unchecked with their reason, ≤ 20 selectable), frequency `<select>` (자동 감지/일간/주간/월간/분기/연간), optional unit
label input (≤ 12 chars), preview tables (first/last 5 rows), notices, `problem` banner with `열 고르기`, buttons
`분석하기` (POST analyze, then navigate to `/r/<job_id>`) and `다시 올리기`. Expired upload (`410`) → error card.

**F.6.11 Result page** (`result.js`): `/r/<id>` polls the job if running (progress as F.6.5) or loads the result; adds
the entry to recent analyses (F.6.20, with `expires_at`). For upload results it shows a notice line under the header,
`result.expiry` (`이 결과는 {date}까지 보관돼요 · 오래 두려면 공유 링크를 만드세요`) with a `공유` button. `/s/<id>` loads
`GET /api/shared/<id>`: read-only banner `읽기 전용 공유 링크 · {date}까지` (no share/AI buttons; exports available).

**F.6.12 Share and export menu** (`sharemenu.js`, `sharedialog.js`): `공유` — public sources: copy the page URL
**with an absolute period** (`range=` replaced by `{query.period.start}~{query.period.end}` from the result envelope,
C.7), so the boss who opens the link next week sees the same period and, while the Naver cache lasts, the same
numbers (toast `share.copied`); uploads: `POST /api/share` → dialog with the absolute URL (`location.origin + url`),
copy button, expiry date, `링크 삭제` (DELETE with the stored token, confirm step), stored in
`trendai.shares[result_id]`; a second click reuses the stored share. `요약 복사`: `summary_line` (which already ends
with the data date, D.2.10) + ` — ` + (share URL or the absolute-period page URL).
`내보내기 ▾`: `CSV`, `엑셀(.xlsx)` (hidden when `exports.xlsx` is null; the unavailable text as tooltip), `이미지(PNG)`
(F.7.6), `HTML 리포트` (or disabled with the `export_naver_raw` text), `JSON`. Downloads use `<a href download>` to
the export URL + `?hl=`.

**F.6.13 Notices** (`notices.js`): list under the source line; `warning` with a ⚠ icon, `notice`/`info` with ⓘ;
fix buttons as F.6.7.

**F.6.14 Markdown-lite** (`md.js`): input → escaped text nodes only. Supports `## ` headings (rendered as `<h3>`),
`- ` bullets with nesting by 2-space indentation, blank-line paragraphs, `**bold**`. `md(text, dateLinks)` turns every
exact occurrence of a `dateLinks` key (the VM's `narrative.date_links`, D.2.9: display text → chart date, longest keys
matched first) into `<button class="datelink" data-date="{chart date}">` which highlights that date on the chart
(crosshair moves there) and scrolls the chart into view. md.js has no date regex and no frequency logic of its own.
Nothing else is interpreted.

**F.6.15 Samples page** (`samples.js`): 9 cards (title, description, sparkline from `spark`, `가상 데이터` badge);
checkboxes for multi-select (max 5) and a sticky `선택한 샘플 비교 (n)` button → `/explore?src=sample&q=a,b,c`.

**F.6.16 Info pages** (`info.js`): `/method`, `/sources`, `/privacy` with content from G.10–G.12; headings have ids
(`#phase`, `#bands`, `#probability`, `#accuracy`, `#metrics`, `#scores`, `#index`) for deep links.

**F.6.17 Settings** (`settings.js`, a dialog opened from `설정` in the footer and in the mobile menu): theme (system/light/dark), up/down colours (국내식 ▲빨강 ▼파랑 / 해외식 ▲초록 ▼빨강 /
언어에 맞춤), stored in `trendai.theme`, `trendai.updown`.

**F.6.18 Footer** (`footer.js`): disclaimers (G.7), links `방법` `데이터 출처` `개인정보` `설정`, `광고 없음`, version from config.

**F.6.19 Toasts** (`toast.js`): `role="status"`, 4 s, also announced via `#live-polite`.

**F.6.20 Recent analyses** (`recent.js`): `trendai.recent` (max 8, newest first, de-duplicated by URL):
`{title, url, at, expires_at}` (`expires_at` = the result's, `null` for public-source URLs, which re-run); titles
`선크림`, `패딩 vs 플리스`, `월매출.xlsx`. Shown on home; each has a `×` remove button. Entries whose `expires_at` has
passed are shown greyed with `recent.expired` (`보관 기간이 지났어요`) and no link, and are pruned after 7 days.

### F.7 Charts (`chart/*`)

All charts are SVG built with `dom.svg()`; colours via classes from `chart.css` (CSS variables), never inline
styles. Geometry is recomputed on container resize (`ResizeObserver`, debounced 100 ms).

#### F.7.1 Fan chart (detail)

* Size: width = container; height 320 px (≥ 640 px wide) / 240 px (< 640). Margins: left = widest y label + 8,
  right 16 (48 when the forecast region label needs it), top 24, bottom 28.
* Layers, bottom to top: forecast region rect (`--fc-region`) with label `예측` at its top-left; grid lines
  (`--grid`); 95 % band, 80 % band (fill `--c-fc`, opacities `--band95-op`, `--band80-op`); trend (1.5 px `--c-trend`,
  `stroke-dasharray: 1 3`, toggle); history (2 px solid `--c-hist`); forecast mean (2 px `--c-fc`, dash `6 4`);
  vertical "now" line (1 px `--ink-3`, dash `3 3`) at the last actual date with label `last_actual.label`;
  changepoint lines (1 px `--ink-3`) with numbered badges ① (circle r 9, fill `--c-cp-bg`, number `--c-cp-fg`) on top;
  anomaly rings (r 4, 2 px `--c-anom`, fill `--surface`; toggle); last actual point (r 4, `--c-hist`, 2 px `--surface`
  stroke); crosshair layer.
* **Seam**: the forecast mean and both bands start at `last_actual` (band width 0 there) — prepend
  `(last_actual.date, last_actual.value)` to mean/lo/hi.
* **Display rule**: `normal` as above; `weak` → mean 1.5 px at opacity .6 + chip `참고용`; `bands_only` → no mean
  line, bands only + chip `참고용` (+ `hero.reference_note`).
* **Y range** (brief §4.3 rule 9): from `y_hint`; anomalies whose inclusion would more than double the range of the
  other values are left out of the range and drawn as small triangles at the top/bottom edge with their value
  text; the last actual always stays in range. Nice ticks: 4–6 (≥ 640 px) / 3–4 (< 640) via `scale.js` (1-2-5
  steps); tick labels use the step's own decimals (a relative index gets integer ticks `0 20 40 …`; only the read-out
  shows one decimal). Relative index in the detail chart: from 0 to the data/band max (`y_hint`, D.2.6), with
  `chart.y_note` as a caption when the term is small; the 0–100 frame is used only by the comparison chart. Never a
  broken axis.
* **X axis**: time scale over history + forecast; ticks per D.1.6; the x label `chart.x_label` under the axis.
* `band` param: `both` (default), `80`, `95`, `off`.
* Accessibility: `<svg role="img" aria-labelledby="t d">` with `<title id=t>` = `chart.title`, `<desc id=d>` =
  `chart.desc`; the chart wrapper is focusable (F.7.4); a `표로 보기` toggle shows `datatable.js` with every
  history and forecast row.

#### F.7.2 Comparison chart

* Same frame as F.7.1 (height 340 / 260). One history line per series (2 px, slot colour `--slot-N`); forecast
  dashed in the same colour; no bands unless a series is focused (legend hover/focus, line hover, card hover) and
  `예측 범위 보기` is on → that series' 80 % band (opacity `--band80-op`). Relative-index compares use a fixed 0–100
  frame (extended above 100 only if a band exceeds it).
* **Weak-forecast rule per series** (the series VM's `chart.display`, also `ranking[].display`): `normal` → dashed
  2 px forecast mean; `weak` → dashed mean at 1 px, opacity .6; `bands_only` → **no forecast mean line at all** (its
  80 % band appears only when the series is focused and `예측 범위 보기` is on), and the legend chip gets `참고용`.
* Legend chips above the plot (always): swatch + slot icon + name, `button[aria-pressed]` toggles visibility.
  **Every coloured (slotted) visible series gets a direct label at its line end** (up to 5), nudged ≥ 14 px apart
  vertically, with a 1 px `--ink-3` leader line when nudged more than 6 px; and each line ends in its slot icon as an
  end marker (r 4, slot colour, 1.5 px `--surface` stroke). This is the mitigation that allows light-mode slot
  strokes below 3:1 (F.8). Slot `null` series: 1 px `--ink-3` at opacity .5, no label; clicking it highlights it
  (2 px, `--ink-2`, with a direct label) until clicked again.
* Scale: `raw`, `index` (client divides by `scale.index_base.values[k]` × 100; axis label `기준 시점 = 100`), `log`
  (log10 axis with 1-2-5 decade ticks; disabled when not allowed). No trend lines, no anomaly rings.
* Read-out (F.7.4) lists every visible series' value at the crosshair date, sorted descending, with swatch+icon.

#### F.7.3 Sparkline

120×32 px, no axes: history polyline 1.5 px slot colour (or `--c-hist` outside comparisons), forecast tail dashed
(`weak`: opacity .6; `bands_only`: no tail — `spark.forecast` is empty, D.4.1), last-point dot r 2.5. Each card
scales its own y. `aria-hidden="true"` (the card text carries the information).

#### F.7.4 Crosshair and read-out (shared by F.7.1/F.7.2)

* A fixed **read-out bar** above the plot (not a floating tooltip) shows: date label (`dateLabel`), actual value or
  forecast `약 {mean} ({lo80}~{hi80})` (en `~{mean} ({lo80}–{hi80})`), and changepoint/anomaly text when the point has
  one. Values via `fmt.valueText(…, value_kind)` + `unit_label`.
* Pointer: moving over the plot snaps the vertical crosshair to the nearest date (history or forecast); leaving
  hides it. Touch: press-and-drag horizontally scrubs (`touch-action: pan-y` on the plot so vertical page scroll
  still works); lifting the finger keeps the last read-out.
* Keyboard: the chart wrapper (`tabindex="0"`, `role="group"`, `aria-label` = chart title + `· 화살표 키로 탐색`)
  shows the crosshair on focus at the last actual point; `←`/`→` one point, `Shift+←/→` 10 points, `Home`/`End`
  first/last point, `Esc` hides. The read-out text is mirrored into `#live-polite` at most every 250 ms.
* Changepoint badges are buttons (`aria-label` = changepoint text); activating one highlights the list item and vice
  versa (two-way, brief §3.5).

#### F.7.5 Auto description (`describe.js`)

Uses the VM `chart.desc` for single series; for comparisons builds `"{n}개 시계열 비교: {a}, {b}, {c}. 상승 점수 1위
{name}."` / `"Comparison of {n} series: …. Highest rising score: {name}."`.

#### F.7.6 PNG export (`png.js`)

Clone the visible chart SVG; replace every CSS-variable colour with its computed value as presentation attributes
(`fill`, `stroke`, `opacity`) because external CSS does not apply inside an `<img>`; set the font-family attribute to
the system stack; serialise (`XMLSerializer`) → `Blob` (`image/svg+xml`) → object URL → `Image` → canvas at 2× scale
with `--surface` background, padding 24 px, chart title above, the source line (wrapped) — the series VM's
`source.line` for a detail chart, **`compare.source_line`** for a comparison chart, so filters such as
`모바일·여성·25–34세` always travel with the image — and the watermark `trendai` below → `canvas.toBlob('image/png')` →
download `trendai-{name}-{YYYYMMDD}.png`. No network access, so the canvas is never tainted.

### F.8 CSS tokens (`css/tokens.css`, exact values; brief §4.2)

```css
:root {
  --font: "Pretendard Variable", Pretendard, -apple-system, BlinkMacSystemFont, "Apple SD Gothic Neo",
          "Malgun Gothic", "Noto Sans KR", system-ui, sans-serif;
  --bg: #f6f6f3; --surface: #ffffff; --surface-2: #f2f2ee;
  --ink: #141413; --ink-2: #4e4d49; --ink-3: #6b6963;
  --grid: #ebeae5; --axis: #c3c2b7;
  --border: rgba(20, 20, 19, .10); --border-strong: rgba(20, 20, 19, .18);
  --accent: #2a78d6; --accent-ink: #ffffff; --focus: #2a78d6;
  --c-hist: #2a78d6; --c-fc: #eb6834; --c-trend: #7a7871; --c-anom: #b4418e;
  --c-cp-bg: var(--ink); --c-cp-fg: #ffffff;
  --fc-region: rgba(235, 104, 52, .045);
  --band95-op: .11; --band80-op: .20;
  --up-kr: #d22d3d; --down-kr: #1e5fc2; --up-intl: #0b7a33; --down-intl: #c0362c;
  --up: var(--up-kr); --down: var(--down-kr);
  --slot-1: #257de0; --slot-2: #f47230; --slot-3: #00b894; --slot-4: #f2a600; --slot-5: #ee7fac;
  --ph-emerging: #0f8a7e; --ph-growing: #2f8a3a; --ph-peaking: #b26b00; --ph-declining: #8c5a3c;
  --ph-stable: #6b6963; --ph-reviving: #6741d9; --ph-volatile: #c2255c;
  --warn: #b26b00; --danger: #b4418e;
  --r-card: 12px; --r-chip: 999px;
  --s-1: 4px; --s-2: 8px; --s-3: 12px; --s-4: 16px; --s-5: 24px; --s-6: 32px;
  --fs-body: 16px; --fs-2: 14px; --fs-cap: 13px; --fs-tile: 28px; --fs-hero: clamp(28px, 4vw, 32px);
  --lh-body: 1.65; --lh-head: 1.3;
  --content-max: 1120px;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #0e0e0d; --surface: #1a1a19; --surface-2: #242422;
    --ink: #f0efec; --ink-2: #c3c2b7; --ink-3: #9c9a92;
    --grid: #2a2a28; --axis: #46463f;
    --border: rgba(240, 239, 236, .12); --border-strong: rgba(240, 239, 236, .22);
    --accent: #5a9cea; --accent-ink: #0e0e0d; --focus: #5a9cea;
    --c-hist: #3987e5; --c-fc: #e2683a; --c-trend: #a3a199; --c-anom: #c94f98;
    --c-cp-fg: #0e0e0d;
    --fc-region: rgba(226, 104, 58, .07);
    --band95-op: .20; --band80-op: .34;
    --up-kr: #ff6b6b; --down-kr: #6ea8fe; --up-intl: #45c46c; --down-intl: #f07a7a;
    --slot-1: #338ced; --slot-2: #e76624; --slot-3: #00a888; --slot-4: #d28e05; --slot-5: #e24d8b;
    --ph-emerging: #2fbfaf; --ph-growing: #5cc46a; --ph-peaking: #e0a030; --ph-declining: #c7916b;
    --ph-stable: #9c9a92; --ph-reviving: #9b87f5; --ph-volatile: #f06595;
    --warn: #e0a030; --danger: #c94f98;
  }
}
:root[data-theme="dark"] { /* the same dark block, repeated verbatim */ }
:root[data-updown="intl"] { --up: var(--up-intl); --down: var(--down-intl); }
@media (max-width: 639px) { :root { --fs-tile: 24px; } }
```

* `theme-init.js`: reads `trendai.theme` (`light`/`dark`; absent = system) → `document.documentElement.dataset.theme`;
  reads `trendai.updown` (`kr`/`intl`; absent → by language: `ko` → `kr`, else `intl`) → `dataset.updown`. Wrapped in
  try/catch (storage may throw).
* Body: `background: var(--bg); color: var(--ink); font: 400 var(--fs-body)/var(--lh-body) var(--font);
  word-break: keep-all; overflow-wrap: anywhere;` numbers in tiles/tables/axes/read-outs `font-variant-numeric:
  tabular-nums`. No italics, no uppercase emphasis; emphasis = weight 600.
* Phase colours only on the phase icon/dot; badge background `--surface-2`, text `--ink`. Never colour-only legends.
* Light-mode slot colours 2–5 are below 3:1 against `--surface` (brief §4.2 note). They are allowed for chart strokes
  **only** together with the F.7.2 mitigations: direct line-end labels for every slotted series, slot-icon end
  markers, legend chips with icon + name, and the table view.
* `--up`/`--down` are used only together with `▲`/`▼` or a sign; anomalies and the declining phase never use them.
* Naver green and Naver/Google logos are never used.

### F.9 Accessibility (WCAG 2.2 AA / KWCAG 2.2; brief §4.3 rule 13)

1. Landmarks (`header`, `nav`, `main`, `footer`), one `<h1>` per page, skip link, logical heading order.
2. Visible focus everywhere: `:focus-visible { outline: 2px solid var(--focus); outline-offset: 2px; }`.
3. Targets ≥ 24×24 px (WCAG 2.5.8); primary buttons, chips and tabs ≥ 44 px tall below 640 px.
4. Colour is never the only carrier: phase = icon + text; direction = arrow + sign/word; series = colour + icon +
   name; notices = icon + text; required non-text contrast ≥ 3:1 (lines, markers, focus ring), with the F.8 exception
   for mitigated slot strokes. Glyphs are not read aloud: `▲ ▼ ▶`, slot icons and ⚠/ⓘ are rendered `aria-hidden="true"` and paired with
   visually hidden text from the catalog (`a11y.up` `{value} 상승` / `up {value}`, `a11y.down` `{value} 하락` /
   `down {value}`, `a11y.flat` `보합` / `flat`; slot icons → the series name; notices → `a11y.warning` `주의` /
   `a11y.info` `안내`). The server texts keep the glyphs; `tiles.js`, `hero.js`, `rankcards.js` split a leading arrow
   off a value (`▲12%` → hidden-arrow + `12%` + sr-only `12% 상승`).
5. Charts: `role="img"` + `<title>` + `<desc>`; keyboard crosshair (F.7.4); table view for every chart.
6. Help popovers (`?`) open on click/tap/Enter, stay while hovered or focused, close on `Esc` (WCAG 1.4.13); the
   trigger is a `button` with `aria-expanded` and `aria-controls`.
7. Dialogs/sheets trap focus, close on `Esc`, restore focus; `aria-modal="true"` and labelled.
8. Live regions: progress (`role="status"`), toasts, crosshair read-out (throttled), route title announcements;
   errors in `#live-assertive`.
9. Forms: every input has a `<label>`; errors tied with `aria-describedby`; radio groups for source/device/gender.
10. `lang` on `<html>` follows `hl`; data names keep their own language (no `lang` override needed).
11. `prefers-reduced-motion`: no shimmer, no transitions on charts/sheets.
12. Text zoom to 200 % and 320 px width without loss (reflow), no horizontal page scroll at ≥ 320 px.

### F.10 Responsive breakpoints

| Width | Layout |
|---|---|
| < 640 px (mobile; tested at 390 px) | single column, 16 px side gutter, nav in a sheet, chips → bottom sheets, tiles 2 cols, ranking cards 1 col, tables as stacked cards (`datatable.js` card mode), chart 240/260 px tall, 3–4 y ticks |
| 640–1023 px | tiles 3 cols, ranking 2 cols, popovers for chips, chart 320/340 px |
| ≥ 1024 px | content max 1120 px centred, tiles 6 cols, ranking 3 cols, upload confirm in two columns (controls \| preview) |

No element may cause horizontal page scroll at 390 px (tables scroll inside their own container if wider).

### F.11 `localStorage` keys (all access wrapped in try/catch; the app works without storage)

| Key | Value |
|---|---|
| `trendai.hl` | `ko` \| `en` |
| `trendai.theme` | `light` \| `dark` (absent = system) |
| `trendai.updown` | `kr` \| `intl` (absent = by language) |
| `trendai.recent` | JSON array ≤ 8 of `{title, url, at, expires_at}` |
| `trendai.shares` | JSON object `{result_id: {share_id, url, delete_token, expires_at}}` (expired entries pruned on read) |
| `trendai.dismiss.naverIndex` | `1` once the Naver explainer was dismissed |

---
## G. Copy (UI catalog `static/js/catalog.js`, WP3; server texts noted)

Tone (brief §4.5): UI, buttons, empty states and errors in 해요체; analysis text (narrative, reasons, caveats) keeps
the engine's 합니다체. No exaggeration (`폭발`, `대박`, exclamation marks), no certainty words. Keys are flat, dotted.
WP3 copies every row below into `CATALOG.ko` / `CATALOG.en`; `tests/js/i18n.test.mjs` checks both languages have the
same keys and no empty values. Texts produced by the server (VM, errors, notices) are listed in C.12, D and G.13–G.14
and are **not** duplicated in the catalog.

### G.1 Common, navigation, settings, accessibility

| key | ko | en |
|---|---|---|
| `app.name` | trendai | trendai |
| `nav.explore` | 탐색 | Explore |
| `nav.upload` | 내 파일 | My file |
| `nav.samples` | 샘플 | Samples |
| `nav.method` | 방법 | Method |
| `nav.menu` | 메뉴 | Menu |
| `lang.label` | 화면 언어 | Display language |
| `theme.button` | 테마 바꾸기 (지금: {mode}) | Change theme (now: {mode}) |
| `theme.system` · `theme.light` · `theme.dark` | 시스템 설정 · 라이트 · 다크 | System · Light · Dark |
| `settings.title` | 설정 | Settings |
| `settings.theme` | 테마 | Theme |
| `settings.updown` | 등락 색상 | Up/down colours |
| `settings.updown.kr` | 국내식 (▲ 빨강 · ▼ 파랑) | Korean style (▲ red · ▼ blue) |
| `settings.updown.intl` | 해외식 (▲ 초록 · ▼ 빨강) | International (▲ green · ▼ red) |
| `settings.updown.auto` | 언어에 맞춤 | Follow the language |
| `common.close` · `common.cancel` · `common.save` · `common.apply` | 닫기 · 취소 · 저장 · 적용 | Close · Cancel · Save · Apply |
| `common.details` | 자세히 | Details |
| `common.help` | 도움말 | Help |
| `common.synthetic` | 가상 데이터 | Synthetic data |
| `common.reference_only` | 참고용 | For reference |
| `common.tentative` | 잠정 | Tentative |
| `common.dash_reason` | 값 없음: {reason} | No value: {reason} |
| `a11y.skip` | 본문으로 건너뛰기 | Skip to content |
| `a11y.chart_keys` | 화살표 키로 시점을 옮길 수 있어요 | Use the arrow keys to move along the chart |
| `a11y.remove` | {name} 빼기 | Remove {name} |
| `a11y.new_tab` | 새 창에서 열림 | Opens in a new tab |
| `title.home` | trendai — 트렌드 분석과 예측 | trendai — trend analysis and forecasts |
| `title.upload` · `title.samples` · `title.method` · `title.sources` · `title.privacy` · `title.notfound` | 내 파일 분석 · 샘플 갤러리 · 방법과 용어 · 데이터 출처 · 개인정보 처리 · 페이지를 찾을 수 없어요 | Analyse your file · Sample gallery · Method and terms · Data sources · Privacy · Page not found |

### G.2 Home

| key | ko | en |
|---|---|---|
| `home.title` | 이 트렌드, 뜨고 있을까요? | Is this trend on the rise? |
| `home.sub` | 과거를 풀고, 지금 국면을 진단하고, 앞으로를 범위로 예측해요. | We unpack the past, diagnose the current phase and forecast the future as a range. |
| `home.examples` | 예시 | Examples |
| `home.example_terms.naver` | 선크림,패딩,수능,김장,다이어트 | 선크림,패딩,수능,김장,다이어트 |
| `home.example_terms.wiki` | 자외선 차단제,대학수학능력시험,김장,장마,인공지능 | 자외선 차단제,대학수학능력시험,김장,장마,인공지능 |
| `home.upload.title` | 내 데이터 분석 | Analyse your data |
| `home.upload.body` | 엑셀·CSV를 끌어다 놓거나 | Drop an Excel or CSV file here, or |
| `home.upload.choose` | 파일 선택 | Choose a file |
| `home.upload.paste` | 붙여넣기 | Paste |
| `home.samples.title` | 샘플로 체험하기 · 가상 데이터 | Try a sample · synthetic data |
| `home.recent` | 최근 분석(이 브라우저) | Recent analyses (this browser) |
| `home.recent.remove` | 목록에서 지우기 | Remove from the list |

### G.3 Search box, sources, chip editor

| key | ko | en |
|---|---|---|
| `search.placeholder.naver` | 키워드를 입력하세요 (예: 선크림, 패딩) | Enter a keyword (e.g. 선크림, 패딩) |
| `search.placeholder.wiki` | 위키백과 문서 제목을 입력하세요 (예: 인공지능) | Enter a Wikipedia article title (e.g. 인공지능) |
| `search.placeholder.sample` | 샘플을 고르세요 (최대 5개) | Choose samples (up to 5) |
| `search.label` | 분석할 키워드 | Keywords to analyse |
| `search.submit` | 분석 | Analyse |
| `search.hint` | 쉼표·Enter로 비교 추가(최대 5개) · ‘+’로 같은 뜻 묶기 | Comma or Enter adds a comparison (up to 5) · ‘+’ groups terms with the same meaning |
| `search.hint.wiki` | 쉼표·Enter로 비교 추가(최대 5개) · 위키백과는 ‘+’ 묶기를 지원하지 않아요 | Comma or Enter adds a comparison (up to 5) · Wikipedia does not support ‘+’ groups |
| `search.hint.wiki_title` | 위키백과는 문서 제목으로 찾아요. 검색어를 정식 문서 제목으로 고친 뒤 분석해 주세요(예: 선크림 → 자외선 차단제). | Wikipedia is searched by article title. Change the terms to exact article titles, then analyse (e.g. 선크림 → 자외선 차단제). |
| `search.too_many` | 한 번에 5개까지 비교할 수 있어요. 하나를 빼고 추가해 주세요. | You can compare up to 5 at a time. Remove one first. |
| `search.too_long` | 검색어는 50자까지 입력할 수 있어요. | Terms can be at most 50 characters long. |
| `source.label` | 데이터 출처 | Data source |
| `source.naver` · `source.wiki` · `source.upload` · `source.sample` | 네이버 검색 · 위키백과 · 내 파일 · 샘플 | Naver search · Wikipedia · My file · Samples |
| `source.unavailable` | 준비 중 | Not available |
| `source.naver.unconfigured` | 운영자가 네이버 API 키를 설정하지 않았어요. 위키백과나 내 파일로 분석해 보세요. | The operator has not set up Naver API keys. Try Wikipedia or your own file. |
| `source.naver.budget` | 이번 달 네이버 조회 한도를 모두 썼어요. 이미 조회된 검색어는 저장된 결과로 볼 수 있어요. | This month's Naver quota is used up. Terms looked up before can still be viewed. |
| `chipeditor.title` | 같은 뜻 검색어 | Terms with the same meaning |
| `chipeditor.label` | 같은 뜻 검색어 (쉼표로 구분, 최대 5개) | Terms with the same meaning (comma-separated, up to 5) |
| `chipeditor.help` | 네이버 데이터랩처럼 여러 검색어의 검색량을 더해 하나의 선으로 봐요. | As in Naver DataLab, the searches for these terms are added up into one line. |

### G.4 Query bar chips

| key | ko | en |
|---|---|---|
| `chip.compare_add` | + 비교 | + Compare |
| `chip.compare_full` | 5개까지 비교할 수 있어요 | Up to 5 can be compared |
| `chip.source` | 출처: {value} | Source: {value} |
| `chip.range` | 기간: {value} | Period: {value} |
| `chip.unit` | 단위: {value} | Unit: {value} |
| `chip.h` | 예측: {value} | Forecast: {value} |
| `chip.filters` | 상세 조건 | Filters |
| `chip.filters_on` | 상세 조건: {value} | Filters: {value} |
| `range.1y` · `range.3y` · `range.5y` · `range.all` · `range.custom` | 최근 1년 · 최근 3년 · 최근 5년 · 전체(2016년부터) · 직접 입력 | Last year · Last 3 years · Last 5 years · All (since 2016) · Custom |
| `range.start` · `range.end` | 시작일 · 종료일 | Start · End |
| `unit.auto` · `unit.date` · `unit.week` · `unit.month` | 자동 · 일 · 주 · 월 | Auto · Day · Week · Month |
| `h.auto` | 자동 ({value}) | Auto ({value}) |
| `h.value` | {n}{unit} | {n} {unit} |
| `filter.device` · `filter.all` · `filter.pc` · `filter.mo` | 기기 · 전체 · PC · 모바일 | Device · All · PC · Mobile |
| `filter.gender` · `filter.m` · `filter.f` | 성별 · 남성 · 여성 | Gender · Male · Female |
| `filter.ages` | 연령 | Age |
| `chip.sample_fixed` | 샘플은 기간과 단위가 정해져 있어요. | Samples have a fixed period and unit. |
| `chip.readonly` | 올린 파일의 결과는 조건을 바꿀 수 없어요. | Results from an uploaded file cannot be changed here. |
| `chip.reanalyze_upload` | 다시 분석 | Analyse again |

### G.5 Upload and confirm step

| key | ko | en |
|---|---|---|
| `upload.title` | 내 파일 분석 | Analyse your file |
| `upload.drop` | 파일을 여기에 끌어다 놓으세요 | Drop a file here |
| `upload.or` | 또는 | or |
| `upload.choose` | 파일 선택 | Choose a file |
| `upload.formats` | CSV·TSV·TXT·엑셀(.xlsx)·JSON · 최대 10MB · 값 열 최대 20개 | CSV, TSV, TXT, Excel (.xlsx), JSON · up to 10 MB · up to 20 value columns |
| `upload.paste` | 엑셀에서 붙여넣기 | Paste from Excel |
| `upload.paste.label` | 엑셀에서 표를 복사해 여기에 붙여넣으세요 (첫 줄은 열 이름) | Copy a table in Excel and paste it here (first row = column names) |
| `upload.paste.read` | 읽기 | Read |
| `upload.privacy` | 개인정보(이름·연락처 등)가 든 파일은 올리지 마세요. 원본 파일은 저장하지 않아요. | Do not upload files with personal data (names, contacts). The original file is never stored. |
| `upload.reading` | 파일을 읽는 중… | Reading the file… |
| `upload.formats.title` | 읽을 수 있는 파일 | Supported files |
| `upload.formats.body` | CSV·TSV·TXT(구분자 자동 감지), 엑셀 .xlsx, JSON. UTF-8·CP949(한국어 엑셀)·UTF-16. 날짜 2024-01-05, 2024.01, 202401, 2024년 1월, 2024 Q1 등. 숫자 1,234 · 1.2억 · 3,400만 · 12% 등. | CSV, TSV, TXT (separator detected), Excel .xlsx, JSON. UTF-8, CP949 (Korean Excel), UTF-16. Dates like 2024-01-05, 2024.01, 202401, 2024 Q1. Numbers like 1,234 · 1.2억 · 12%. |
| `gt.title` | Google 트렌드 데이터 올리기 | Upload Google Trends data |
| `gt.steps` | Google 트렌드(trends.google.com)에서 조회 → 차트 오른쪽 위 다운로드(CSV) → 받은 CSV를 여기에 올려요. | Run a query on Google Trends (trends.google.com) → download (CSV) at the chart's top right → upload the CSV here. |
| `gt.note` | 값은 Google 트렌드의 상대 지수(0–100)로 표시하고 출처 문구를 자동으로 붙여요. | Values are shown as Google Trends' relative index (0–100) and the attribution is added automatically. |
| `confirm.title` | 이렇게 읽었어요 | Here is how we read it |
| `confirm.date_col` | 날짜 열 | Date column |
| `confirm.date_format` | 형식: {value} | Format: {value} |
| `confirm.wide` | 날짜가 열 머리글에 있는 표예요. 행마다 시계열 하나로 읽었어요. | The dates are in the column headers; each row was read as one series. |
| `confirm.value_cols` | 값 열 | Value columns |
| `confirm.excluded.row_number` | 제외됨: 순번으로 보여요 | Excluded: looks like row numbers |
| `confirm.excluded.not_numeric` | 제외됨: 숫자가 아니에요 | Excluded: not numeric |
| `confirm.max_cols` | 값 열은 20개까지 고를 수 있어요. | Choose up to 20 value columns. |
| `confirm.freq` | 주기 | Frequency |
| `confirm.freq.detected` | (자동 감지: {value}) | (detected: {value}) |
| `freq.auto` · `freq.D` · `freq.W` · `freq.M` · `freq.Q` · `freq.Y` | 자동 감지 · 일간 · 주간 · 월간 · 분기 · 연간 | Auto-detect · Daily · Weekly · Monthly · Quarterly · Yearly |
| `confirm.unit_label` | 단위(선택) | Unit (optional) |
| `confirm.unit_placeholder` | 예: 원, 개, 명 | e.g. KRW, units, people |
| `confirm.preview` | 미리보기: 처음 5행 · 마지막 5행 | Preview: first 5 rows · last 5 rows |
| `confirm.encoding.utf8` (`utf-8`, `utf-8-sig`) | UTF-8로 읽었어요 | Read as UTF-8 |
| `confirm.encoding.cp949` (`cp949`, `euc-kr`) | 한국어 엑셀(CP949)로 읽었어요 | Read as Korean Excel text (CP949) |
| `confirm.encoding.utf16` (`utf-16*`) | UTF-16으로 읽었어요 | Read as UTF-16 |
| `confirm.encoding.other` | 문자 인코딩: {value} | Text encoding: {value} |
| `confirm.analyze` | 분석하기 | Analyse |
| `confirm.reupload` | 다시 올리기 | Upload again |
| `confirm.need_value` | 값 열을 하나 이상 골라 주세요. | Choose at least one value column. |

### G.6 Results (detail and compare)

| key | ko | en |
|---|---|---|
| `detail.back` | ← 비교로 | ← Back to comparison |
| `detail.share` · `detail.export` · `detail.table` | 공유 · 내보내기 · 표 | Share · Export · Table |
| `detail.why` | 왜? | Why? |
| `detail.chart.title` | 추이와 예측 | History and forecast |
| `detail.toggle.trend` · `detail.toggle.anomalies` | 추세선 · 이상치 | Trend · Anomalies |
| `chart.forecast` | 예측 | Forecast |
| `chart.legend_bands` | 진한 띠: 10번 중 8번은 이 안 · 연한 띠: 20번 중 19번은 이 안 | Dark band: 8 times in 10 inside · light band: 19 times in 20 inside |
| `chart.band_caption` | 범위가 넓어지는 건 먼 미래일수록 덜 확실하기 때문이에요. 값이 더 출렁인다는 뜻은 아니에요. | The range widens because the far future is less certain, not because values will swing more. |
| `chart.readout_forecast` | 예측 약 {mean} ({lo}~{hi}) | Forecast ~{mean} ({lo}–{hi}) |
| `chart.readout_actual` | 실제 {value} | Actual {value} |
| `chart.index_axis` | 기준 시점 = 100 | Base period = 100 |
| `detail.now.title` | 현재 — 왜 {phase}인가요? | Now — why {phase}? |
| `detail.now.rules` | 국면 판단 규칙 보기 › | How phases are decided › |
| `detail.past.title` | 과거 — 무슨 일이 있었나요? | Past — what happened? |
| `detail.past.changepoints` · `detail.past.none_cp` | 구조 변화 · 구조 변화가 감지되지 않았어요 | Changepoints · No changepoints were detected |
| `detail.past.anomalies` · `detail.past.none_an` | 이상치 · 이상치가 없어요 | Anomalies · No anomalies |
| `detail.future.title` | 미래 — 얼마나 확실한가요? | Future — how certain is it? |
| `detail.future.scenarios` | 시나리오 ({date}) | Scenarios ({date}) |
| `detail.future.table` | 예측표 | Forecast table |
| `detail.future.cols` | 기간,예측값,80% 범위,95% 범위 | Period,Forecast,80% range,95% range |
| `detail.future.accuracy` · `detail.future.models` · `detail.future.method` | 정확도 · 쓰인 모델 · 예측 방법 | Accuracy · Models used · Method |
| `detail.narrative.title` | 해석 | Interpretation |
| `detail.narrative.ai` | AI로 다시 쓰기 | Rewrite with AI |
| `detail.narrative.ai_running` | AI가 해석을 쓰는 중이에요… 규칙 기반 해석을 먼저 보여 드려요. | The AI is writing… showing the rule-based interpretation meanwhile. |
| `detail.narrative.show_rule` · `detail.narrative.show_ai` | 규칙 기반 해석 보기 · AI 해석 보기 | Show rule-based · Show AI version |
| `detail.caveats.title` | 주의할 점 | Caveats |
| `naver_explainer.title` | 네이버 상대 지수 읽는 법 | How to read Naver's relative index |
| `naver_explainer.body` | 표시된 값은 실제 검색 횟수가 아닌 상대값이에요. 조회 기간 중 가장 많이 검색된 시점을 100으로 두고 나머지를 비율로 나타내요. 같은 조회 안의 검색어끼리만 비교할 수 있고, 기간을 바꾸면 100의 기준도 바뀌어요. trendai의 변화율·국면·예측은 척도와 무관해서 상대 지수여도 그대로 의미가 있어요. | The values are relative, not search counts: the busiest point of the period is 100 and everything else is a share of it. Only terms in the same query can be compared, and changing the period moves the 100. trendai's changes, phases and forecasts do not depend on the scale, so they stay meaningful. |
| `naver_explainer.ok` | 알겠어요 | Got it |
| `help.naver_index` | (= `naver_explainer.body`) | (= `naver_explainer.body`) |
| `result.expiry` | 이 결과는 {date}까지 보관돼요 · 오래 두려면 공유 링크를 만드세요 | This result is kept until {date} · create a share link to keep it longer |
| `recent.expired` | 보관 기간이 지났어요 | Expired |
| `detail.reason_detail` | 자세히 | Details |
| `a11y.up` · `a11y.down` · `a11y.flat` | {value} 상승 · {value} 하락 · 보합 | up {value} · down {value} · flat |
| `a11y.warning` · `a11y.info` | 주의 · 안내 | Warning · Note |
| `compare.title` | 한눈에 보기 | At a glance |
| `compare.scale.raw` · `compare.scale.index` · `compare.scale.log` | 원래 값 · 기준 시점=100 · 로그 | Original · Base period = 100 · Log |
| `compare.bands` | 예측 범위 보기 | Show forecast range |
| `compare.view.table` · `compare.view.chart` | 표로 보기 · 차트로 보기 | Table view · Chart view |
| `compare.rank.title` | 무엇이 뜨나요? | What is rising? |
| `compare.sort` | 정렬 | Sort |
| `compare.sort.score` · `.prob` · `.expected` · `.recent` · `.level` | 상승 점수 · 상승 가능성 · 예측 변화 · 최근 변화 · 현재 수준 | Rising score · Chance of rise · Expected change · Recent change · Current level |
| `compare.score_short` | 점수는 예상 상승률이 아니에요 | The score is not an expected growth rate |
| `compare.card.expected` | 예측 변화 {text} ({basis}) | Expected change {text} ({basis}) |
| `compare.card.recent` | 최근 변화 {text} ({basis}) | Recent change {text} ({basis}) |
| `compare.card.more` | 자세히 › | Details › |
| `compare.card.score` | 상승 점수 {value} | Rising score {value} |
| `compare.table.cols` | 순위,이름,국면,예측 변화,상승 가능성,최근 변화,모멘텀,상승 점수 | Rank,Name,Phase,Expected change,Chance of rise,Recent change,Momentum,Rising score |
| `compare.values_table` | 값 표 | Values table |
| `compare.download_csv` | CSV로 받기 | Download CSV |
| `compare.grey_hint` | 회색 선을 누르면 강조돼요 | Click a grey line to highlight it |

### G.7 Footer

| key | ko | en |
|---|---|---|
| `footer.disclaimer` | 예측은 과거 패턴이 이어진다고 가정해요. 갑작스러운 사건이나 외부 변수(가격·광고·휴일)는 반영하지 못해요. | Forecasts assume past patterns continue. Sudden events and outside factors (prices, ads, holidays) are not included. |
| `footer.not_advice` | 투자·사업 판단의 유일한 근거로 쓰지 마세요. | Do not use this as the sole basis for investment or business decisions. |
| `footer.no_ads` | 광고 없음 | No ads |
| `footer.links.method` · `.sources` · `.privacy` · `.settings` | 방법 · 데이터 출처 · 개인정보 · 설정 | Method · Data sources · Privacy · Settings |
| `footer.version` | trendai {version} | trendai {version} |

### G.8 Share and export

| key | ko | en |
|---|---|---|
| `share.copy_link` | 링크 복사 | Copy link |
| `share.copied` | 링크를 복사했어요 | Link copied |
| `share.dialog.title` | 공유 링크 | Share link |
| `share.dialog.body` | 이 링크를 가진 사람은 누구나 결과를 볼 수 있어요(읽기 전용). {date}까지 보관해요. | Anyone with this link can view the result (read-only). Kept until {date}. |
| `share.delete` | 링크 삭제 | Delete link |
| `share.delete.confirm` | 링크를 삭제할까요? 삭제하면 이 링크로 더는 볼 수 없어요. | Delete the link? It will stop working. |
| `share.deleted` | 링크를 삭제했어요 | Link deleted |
| `share.summary` | 요약 복사 | Copy summary |
| `share.summary.copied` | 요약을 복사했어요 | Summary copied |
| `share.readonly` | 읽기 전용 공유 링크 · {date}까지 | Read-only shared link · until {date} |
| `export.csv` · `export.xlsx` · `export.png` · `export.html` · `export.json` | CSV · 엑셀(.xlsx) · 이미지(PNG) · HTML 리포트(한국어) · JSON | CSV · Excel (.xlsx) · Image (PNG) · HTML report (Korean) · JSON |
| `export.png.saving` | 이미지를 만드는 중… | Creating the image… |

Server-side export texts (WP1 `texts.py`): CSV/XLSX comment and condition labels — `export.source` `출처`/`Source`,
`export.conditions` `조건`/`Conditions`, `export.naver_note` `네이버 원본 값(ratio)은 이용약관상 포함하지 않았어요` /
`Raw Naver values (ratio) are excluded under its terms of use`, column headers per C.10, XLSX condition labels
`출처`, `검색어 그룹/열`, `기간`, `단위`, `조건(기기·성별·연령)`, `데이터 기준`, `값의 성격` / `Source`, `Groups/columns`,
`Period`, `Unit`, `Filters (device · gender · age)`, `Data as of`, `Value kind`; summary sheet headers `이름, 국면, 전망,
예측 변화, 상승 확률, 예측 신뢰도, 정확도, 상승 점수` / `Name, Phase, Outlook, Expected change, Chance of rise,
Reliability, Accuracy, Rising score`.

### G.9 Progress, empty states, client-side errors

| key | ko | en |
|---|---|---|
| `progress.step.data` · `.analyze` · `.forecast` · `.finish` | 데이터 확인 · 추세·구조 분석 · 예측 모델 검증 ({k}/{n}) · 정리 | Checking data · Trend and structure · Validating forecast models ({k}/{n}) · Wrapping up |
| `progress.queued` | 순서를 기다리는 중… | Waiting in line… |
| `progress.fetching_naver` · `progress.fetching_wiki` · `progress.preparing` | 네이버에서 받아오는 중… · 위키백과에서 받아오는 중… · 데이터를 확인하는 중… | Fetching from Naver… · Fetching from Wikipedia… · Checking the data… |
| `progress.series` | {name} 분석 중 ({k}/{n}) | Analysing {name} ({k}/{n}) |
| `progress.finishing` | 결과를 정리하는 중… | Putting the results together… |
| `progress.cancel` | 취소 | Cancel |
| `progress.cancelled` | 분석을 취소했어요. | The analysis was cancelled. |
| `empty.no_query` | 분석할 키워드를 입력해 주세요. | Enter a keyword to analyse. |
| `empty.notfound.title` | 페이지를 찾을 수 없어요 | Page not found |
| `empty.notfound.body` | 주소를 확인하거나 처음 화면에서 다시 시작해 주세요. | Check the address or start again from the home page. |
| `empty.recent_none` | 아직 분석한 데이터가 없어요. 샘플로 먼저 둘러보세요. | Nothing analysed yet. Start with a sample. |
| `err.client_network` | 서버에 연결하지 못했어요. 인터넷 연결을 확인하고 다시 시도해 주세요. | Could not reach the server. Check your connection and try again. |
| `err.file_too_large` | 파일이 너무 커요. 10MB까지 올릴 수 있어요. | The file is too large. The limit is 10 MB. |
| `err.png_failed` | 이미지를 만들지 못했어요. CSV나 HTML 리포트로 받아 주세요. | The image could not be created. Download CSV or the HTML report instead. |
| `err.clipboard` | 복사하지 못했어요. 주소를 직접 복사해 주세요. | Could not copy. Please copy the address manually. |

Samples page: `samples.title` 샘플 갤러리 / Sample gallery · `samples.intro` 가상 데이터로 9가지 트렌드 유형을 체험해 보세요.
실제 데이터가 아니에요. / Explore 9 trend types with synthetic data. This is not real data. · `samples.compare` 선택한 샘플
비교 ({n}) / Compare selected ({n}) · `samples.max` 최대 5개까지 고를 수 있어요 / Pick up to 5.

Help popovers (`?`): `help.compare` 같은 조회의 시계열을 한 축에 그렸어요. 예측은 점선이에요. / Series from the same
query on one axis. Forecasts are dashed. · `help.score` 상승 점수는 전망(예측 변화·상승 확률)과 최근 기세(최근 변화·모멘텀)를
합친 0–100 값이에요. 50이 중립이고, 예상 상승률이 아니에요. / The rising score combines the outlook (expected change, chance
of rise) and recent momentum into 0–100. 50 is neutral; it is not an expected growth rate. · `help.reliability` 예측
신뢰도(높음·보통·낮음)는 백테스트 정확도, 데이터 길이, 예측 범위 폭으로 정해요. / Forecast reliability (high/medium/low)
comes from backtest accuracy, data length and the width of the forecast range. · `help.accuracy` 과거 구간에서 실제로
예측해 보고, 직전 값 유지·계절 반복·추세 연장 중 가장 나은 단순 예측과 오차를 비교했어요. / We forecast past periods and
compared the error with the best simple forecast (repeat last value, repeat last season, extend the line). ·
`help.certainty` 판단 확신도는 국면 규칙이 얼마나 분명하게 들어맞았는지예요. 통계의 신뢰수준이나 예측 신뢰도와는 다른
숫자예요. / Phase certainty says how clearly the phase rules matched. It is neither a statistical confidence level nor
the forecast reliability. · `help.bands` = `chart.band_caption`.

### G.10 `/method` content outline (WP3 writes the prose in both languages from these points)

1. `#overview` 어떻게 분석하나요 — 과거(추세·계절 분해, 구조 변화, 이상치) → 현재(국면, 모멘텀) → 미래(여러 모델 백테스트, 앙상블,
   80/95 % 구간, 상승 확률) → 해석(규칙 기반, 선택 시 AI). All computed from the one series only.
2. `#phase` 국면 판단 규칙 — seven phases with icon and one line each (태동기: 작은 기반에서 가속 성장, 최근 수준이 장기 기반의
   2.5배 이상이고 최근 기울기가 평균의 1.5배 이상 · 성장기: 최근 기울기가 통계적으로 유의(|t| ≥ 2.5)하고 최근 3 % 이상 상승 ·
   정점: 오르다가 최근 기울기가 과거 최대 기울기의 35 % 미만으로 꺾이고 고점 근처 · 쇠퇴기: 추세 고점보다 30 % 이상 낮고 오르지
   않음, 또는 의미 있는 하락(추세 규모의 15 % 이상) · 안정기: 뚜렷한 추세 없음(계절성은 있을 수 있음) · 재부상: 20 % 이상 하락
   뒤 저점에서 10 % 이상 반등하며 상승 중 · 변동성: 구조 없이 잡음이 큼(변동계수 20 % 이상)); "최근" = 마지막 약 15 % 구간;
   관측치 8개 미만은 잠정 판단; 판단 확신도 ≠ 통계 신뢰수준 ≠ 예측 신뢰도; rules are heuristics validated on synthetic
   archetypes (참고용).
3. `#bands` 예측 범위(80 %·95 %) — 10번 중 8번 / 20번 중 19번은 이 안에 들어올 것으로 보는 범위; 먼 미래일수록 넓어짐(불확실성,
   출렁임 아님); 백테스트 오차로 보정하지만 짧은 데이터·구조 변화 직후엔 실제 포함률이 낮을 수 있음.
4. `#probability` 상승 확률 — 예측 마지막 한 주기 평균이 최근 한 주기 평균보다 높을 확률(특정 시점 한 값이 아님); 말로 된 척도
   표(D.1.5 bins); 5 % 단위 반올림, 95 % 이상/5 % 이하 표기; 간헐 수요는 주문 발생 확률과 크기로 계산.
5. `#accuracy` 정확도 — 롤링 원점 백테스트; 단순 예측 3종 중 가장 나은 것 대비 오차; 단순 예측보다 못하면 중앙선을 지우고
   범위만 '참고용'; MASE는 기술 참고값이며 1과 비교하지 않음.
6. `#metrics` 지표 정의 — 최근 변화(추세 기준, 한 주기), 모멘텀(%p/기간), 변동성(잔차 변동계수), 예측 변화(평균 대 평균),
   피크 시즌(계절 성분 최고 달, 계절성 강도 0.3 이상일 때), 변화율을 내지 않는 경우(기준값 0 근처·음수·0 통과·0까지 하락).
7. `#scores` 상승 점수 — 공식 개요(전망 60 %: 예측 변화 35 % + 상승 확률 25 %, 최근 기세 40 %: 최근 변화 20 % + 모멘텀 20 %;
   연 환산; 안정기·변동성 국면은 전망만), 50 = 중립, 예상 상승률 아님.
8. `#index` 상대 지수와 축 — 네이버 상대 지수 설명(naver_explainer), 비교 축 규칙(원래 값 / 기준 시점=100 / 로그), 이중 축을
   쓰지 않는 이유, 위키백과 조회수는 관심의 대리 지표.
9. `#limits` 한계 — README '한계와 주의' 요약(외부 충격 미반영, 일별 데이터는 요일 주기만, 영업일 주기 없음, 짧은 데이터).
10. `#ai` AI 해석 — 누를 때만 실행, 계산된 사실(JSON)만 Anthropic(Claude)으로 전송, 네이버 출처는 제외, AI가 쓴 글은 항상 표시.

### G.11 `/sources` content

One section per source with: what it is, value kind, period/limits, attribution text exactly as in D.2.2, retention.
Naver: NAVER API HUB 검색어트렌드, 2016-01-01부터, 상대 지수, 같은 조회 안에서만 비교, 서버 캐시 최대 21일, AI로 보내지 않음,
원본 값 내려받기 미제공, 로고·브랜드 미사용(텍스트 표기만), 운영자 키가 있을 때만 · Wikimedia: per-article pageviews,
사람 조회만, 2016-01부터, 서버 캐시 24시간(월)/6시간(일·주), 출처 표기 · Google 트렌드: 실시간 연동 안 함, 사용자가 내려받은
CSV만, "Data source: Google Trends (https://www.google.com/trends)", 요청마다 0–100으로 다시 맞춰짐 · 내 파일: 원래 단위,
비공개 기본, 24시간 보관 · 샘플: 가상 데이터, 실제 아님. Footer of the page: 법무 확인 전 보수적 기본값이라는 문장.

### G.12 `/privacy` content

1. 올린 파일: trendai는 원본 바이트를 메모리에서만 읽고 디스크에 쓰지 않음; 확인 단계를 위해 읽은 표를 서버 메모리에 최대
   30분 보관. 운영자 앞단의 프록시·서버 설정에 따라 전송 중 임시 버퍼가 생길 수 있음(운영 가이드 docs/WEB.md에 이를 막는 설정이
   있음) — the page states this qualification instead of an unconditional promise.
2. 분석 결과: 추측할 수 없는 주소(128비트 무작위)로 24시간 보관 후 자동 삭제; 목록은 이 브라우저(localStorage)에만.
3. 공유 링크: 만든 경우에만, 30일 보관, 만든 브라우저에서 언제든 삭제 가능; 결과 사본을 따로 보관.
4. 로그: 요청 종류·상태·시간, 하루 단위로 바뀌는 IP 해시만; 검색어·파일 이름·파일 내용·전체 주소는 기록하지 않음.
5. AI: 누를 때만; 계산된 요약 수치(JSON)만 Anthropic으로 전송; 원본 파일은 보내지 않음; 네이버 데이터는 보내지 않음.
6. 쿠키·추적·광고 없음; 브라우저 저장소(localStorage)는 언어·테마·최근 분석·공유 링크 삭제 토큰에만 사용.
7. 개인정보가 든 파일을 올리지 말라는 안내; 문의: 운영자 연락처(`TRENDAI_CONTACT`가 있으면 그 값을 표시 — `/api/config` does not
   expose it, so the page says "운영자에게 문의하세요" / "Contact the operator of this site").

### G.13 Server texts in `trendai/web/texts.py` (WP1)

* Scenario title/reference/row templates, `y_note`, the summary-line weak-forecast suffixes, `prob_labelled`, the
  `index` disabled reason for single-request sources, and source `short` names — exactly as given in D.1–D.4.
* `SAMPLE_TITLES` ko/en: emerging 태동기형/Emerging · growing 성장형/Growing · peaking 정점형/Peaking · fad 반짝유행/Fad ·
  declining 쇠퇴형/Declining · seasonal 계절형/Seasonal · reviving 재부상형/Reviving · breakout 급성장형/Breakout ·
  volatile 변동형/Volatile.
* `SAMPLE_DESCRIPTIONS_EN`: emerging "An emerging trend growing fast from a small base (start of an S-curve)" ·
  growing "Steady growth with seasonality" · peaking "Growth slowing as it nears saturation (end of an S-curve)" ·
  fad "A short-lived fad that spiked and is now falling" · declining "A steadily declining trend" · seasonal "No
  trend, clear seasonality (stable)" · reviving "Fell, then recently rebounding" · breakout "Flat, then rapid growth
  after a structural break" · volatile "No clear direction, large noise". Korean = `samples.DESCRIPTIONS`.
* Phase short names, likelihood words, certainty/reliability labels, source labels and parts, value-kind labels and
  notes, sentence templates, strength words, momentum words, peak/seasonality templates, future-phase texts,
  accuracy lines, reference notes, summary-line template — exactly as given in D.1–D.2.
* Export labels (G.8 server-side paragraph).

### G.14 Tile labels and help (`texts.TILE_HELP`, WP1)

| key | label ko / en | help ko | help en |
|---|---|---|---|
| `phase` | 현재 국면 / Current phase | 추세의 모양과 통계 검정으로 정한 수명주기 단계예요. 근거는 ‘왜?’에서 볼 수 있어요. | The lifecycle stage, decided from the trend's shape and statistical tests. See ‘Why?’ for the evidence. |
| `recent_change` | 최근 변화 / Recent change | 최근 한 주기({span}) 동안 추세선이 얼마나 변했는지예요. 계절 변동은 뺀 값이에요. | How much the trend line changed over the last cycle ({span}), without seasonal swings. |
| `momentum` | 모멘텀 / Momentum | 최근 기울기와 장기 기울기의 차이예요(현재 추세 수준 대비 %p). +면 장기 평균보다 빨리 오르는 중이에요. | Recent slope minus long-run slope, in percentage points of the current trend level. Positive means rising faster than usual. |
| `expected_change` | 예측 변화 / Expected change | 예측 마지막 한 주기의 평균을 최근 한 주기의 평균과 비교한 값이에요. 특정 시점 한 값의 변화가 아니에요. | The average of the last forecast cycle compared with the average of the last observed cycle — not the change of a single point. |
| `prob_increase` | 상승 가능성 / Chance of rise | 같은 두 평균에서 미래 평균이 더 높을 확률이에요. | The chance that the future average is higher than the recent one (same two averages). |
| `peak_season` | 피크 시즌 / Peak season | 계절 성분이 가장 높은 시기와 오르기 시작하는 시기예요. 계절성이 뚜렷할 때만 보여 드려요. | When the seasonal component peaks and when it starts rising. Shown only for clearly seasonal data. |
| `volatility` | 변동성 / Volatility | 추세와 계절로 설명되지 않는 변동의 크기(잔차의 변동계수)예요. | The size of swings not explained by trend and season (coefficient of variation of the residuals). |

---
## H. Test plan

General: no test touches the network (fetchers injected, HTTP sessions faked, local `http.server` on 127.0.0.1 only
for `UrllibSession`). Every new Python test file runs in < 60 s, **except** `tests/e2e/test_e2e.py` (marker `e2e`:
14 browser scenarios + the Node unit tests). Tests are hermetic: apps come from `make_test_app` (empty environ, B.3.2),
WP1 tests use `write_static_fixture`. Existing tests (`python -m pytest`, ~1,450) must stay green.

### H.1 WP1 — `tests/test_web_*.py` (use `trendai.web.testing`; `tmp_path` data dirs; `FrozenClock` where time matters)

| File | Tests (name → assertion) |
|---|---|
| `test_web_app.py` (all with `static_dir=write_static_fixture(...)`) | `test_health_ok` → 200, keys of C.1, `Cache-Control: no-store` · `test_security_headers_everywhere` → exact CSP string of E.12 and the other 6 headers on `/`, `/static/js/app.js`, `/api/config`, a 404 and a 500 (forced via a handler that raises) · `test_spa_routes_serve_index` → `/`, `/explore`, `/upload`, `/samples`, `/method`, `/sources`, `/privacy`, `/r/<22>`, `/s/<22>` all 200 with bytes identical to `<static_dir>/index.html`; `/r/`, `/s/` carry `X-Robots-Tag` · `test_static_missing_503` (empty static dir → app starts, `/` 503 text/plain, `/api/health` 200) · `test_package_static_if_present` (skipped while `trendai/web/static/index.html` is absent; else `/` serves it byte for byte) · `test_trailing_slash_redirect` → `/explore/?q=a` → 308 `Location: /explore?q=a` · `test_unknown_page_404_index` → `/nope` 404 with index body · `test_unknown_api_404_json` → envelope code `not_found` · `test_method_not_allowed` → `PUT /api/analyze` 405 with `Allow: POST` · `test_options_405_no_cors` → no `Access-Control-*` header · `test_static_traversal_blocked` → `/static/../app.py`, `/static/%2e%2e/app.py`, `/static/..%2fapp.py`, `/static/\\app.py`, `/static/dev/mock.html` (dev off), `/static/js/../../web/app.py` all 404 · `test_static_etag_304` · `test_head_has_no_body` · `test_favicon_and_robots` · `test_gzip_json` (`Accept-Encoding: gzip` on a > 1 KB JSON → `Content-Encoding: gzip`, `Vary`; static never compressed) · `test_json_allow_nan_false` (a VM builder patched to return NaN → 500 `internal_error`, never invalid JSON) |
| `test_web_config.py` | `test_config_naver_unconfigured` (fetchers.naver None → `available false`, `reason "unconfigured"`, `default_source "wiki"`) · `test_config_naver_hub` · `test_config_budget_states` (count at 79 %/80 %/90 % → ok/low/blocked) · `test_config_features_xlsx_ai` (monkeypatch import of openpyxl/AI on/off) · `test_config_last_complete_dates` (FrozenClock 2026-11-25T00:12Z → `date` `2026-11-24`, `week` `2026-11-22`, `month` `2026-10-31`) · `test_webconfig_from_env_invalid_raises` · `test_validate_public` (host 0.0.0.0 without contact → ValueError; with contact ok; `allow_no_contact` ok; 127.0.0.1 ok) · `test_make_test_app_hermetic` (monkeypatched `TRENDAI_WORKERS=7`, `NAVER_API_HUB_KEY=x` in `os.environ` do not reach the app) |
| `test_web_api_analyze.py` | `test_sample_detail_end_to_end` → job done, `view "detail"`, VM validates against `vm.schema.json` and the D.2 key set (helper `assert_series_vm(vm)`), `flags.synthetic` true, 6 tiles, source line contains `가상 데이터`, and **the JSON export has `"synthetic": true`** (the attrs copy of E.6) · `test_compare_three_samples` → `view "compare"`, 3 ranking rows sorted by score, slots 1–3 in request order · `test_progress_since` → polling with `since` returns only new VMs; `series_done` monotonic; stages and steps (`analyze`/`forecast`/`narrative` only) observed in order (use a slow fake) · `test_since_after_failed_series` (wiki series 0 ok, 1 `없는문서`, 2 ok: a client following C.6's rule never receives index 2 twice) · `test_validation_errors` (parametrised: empty series → `empty_query`; 6 series → `too_many_series`; 6 terms Naver → `too_many_keywords`; 51-char term → `term_too_long`; wiki `a+b` → `wiki_group_unsupported`; unknown sample → `unknown_sample`; bad range → `invalid_param` with `params.param == "range"`; h=0 → `invalid_param`) · `test_display_names_unique` (series `["a + b"]`, `["a","b"]`, `["a","c"]` → names `a + b`, `a + b (2)`, `a + c`; ranking rows map to the right indexes) · `test_json_required_415` · `test_body_limit_413_before_read` (Content-Length 70,000 with a 10-byte body → 413 without reading) · `test_cached_result_reuse` (same query twice → second `200 cached true`, same job_id, fetcher called once) · `test_inflight_join` (two identical posts while the first runs with a slow fake → second `202 joined true` with the same job_id; one fetch; one DELETE leaves the job running, the second cancels it) · `test_cancel_running_job` (slow fake → DELETE → status `cancelled`, nothing in `results/`) · `test_queue_full_503` (workers=1, max_queued=1, 3 slow *different* jobs → third `503 server_busy` + `Retry-After: 10`) · `test_analyze_rate_limit_429` (11 rapid posts → 429 `rate_limited`, `Retry-After` ≥ 1) · `test_job_timeout` (job_timeout_s=0.1 with slow fake → `job_timeout`) · `test_job_after_restart_reads_store` (new app on the same data dir → GET job synthesises done) · `test_all_series_fail` (wiki 404 for every title → job failed `wiki_not_found`) · `test_notices_isolated_between_threads` (two concurrent upload jobs whose data emit different notices, plus a third thread calling `pipeline.run` with `horizon_long` in a loop: each result has exactly its own notices) · `test_vm_schema_bump_invalidates_cache` (monkeypatched `WEB_VM_SCHEMA` → no cache reuse, old result `404`) |
| `test_web_naver.py` | `test_one_call_for_compare` (2 series → FakeNaver called once with 2 groups named `g1`, `g2`) · `test_long_display_names` (5 groups of 5 × 50-char terms → works; Naver never sees a display name) · `test_cache_hit_no_call_and_notice` · `test_fetch_single_flight` (two different-`h` queries with the same Naver body submitted together → one FakeNaver call, one budget unit) · `test_cache_ttl_24h_recent_21d_past` (FrozenClock advance) · `test_budget_reserve_and_block` (limit 10: 9th call ok, 10th → `naver_budget_exhausted`; cache hits still served) · `test_budget_month_rollover_kst` (2026-11-30T15:00Z is December in KST → counter resets) · `test_naver_ip_limit` (6th fresh call in a burst → `429 naver_rate_limited`) · `test_naver_unconfigured_503` · `test_connector_error_mapping` (FakeNaver fail modes auth/rate/server/network/timeout/bad → C.12.3 codes and statuses) · `test_ai_blocked_for_naver` (VM `ai_allowed false`, `POST /api/narrative` → 403 `ai_naver_blocked`) · `test_exports_strip_raw` (CSV has no `실제값` column + note; JSON has no `history`, no `metrics.last_value`, `raw_excluded true`, has `source`; HTML → 403 `export_naver_raw`; `values.csv` → 403 `export_naver_raw`; compare `exports.values_csv` null) · `test_naver_index_scale_disabled` (`scale.allowed.index false` with the D.4.3 reason) · `test_low_resolution_and_missing_notices` · `test_period_snapping_used` (FakeNaver receives Monday start / Sunday end) |
| `test_web_wiki.py` | `test_wiki_monthly_detail` (source line contains `Wikimedia Pageviews`) · `test_wiki_404_series_card` (`없는문서` in a 2-series compare → one VM + one `failed` with `wiki_not_found`, fix `wiki_title_help` URL `https://ko.wikipedia.org/w/index.php?search=…`) · `test_wiki_weekly_unit` · `test_wiki_cache_ttls` · `test_user_agent_contains_contact` (default fetchers with a fake session capture the header) · `test_log_hint` (two series with max ratio > 20 → `scale.log_hint true`) |
| `test_web_upload.py` | `test_multipart_csv_detection` (columns roles incl. `번호` excluded/row_number, `메모` ignored; the notice text reads `열 ‘번호’는 …`, no `['번호']`, no `은(는)`) · `test_paste_tsv` · `test_cp949_csv` · `test_xlsx_upload` (skip without openpyxl) · `test_xlsx_year_month_columns` (skip without openpyxl: separate 연도/월 columns with a blank → detected, `date_cols ["연도","월"]`, `date_col "연도"`, preview cells `2020`, not `2020.0`; posting `date_col: "연도"` back works) · `test_google_trends_csv` (kind `google_trends`, notices `gt_detected`, `gt_lt1_half`, value kind `relative_index_gt`) · `test_no_date_col_problem_200` · `test_too_large_413` · `test_zip_bomb_guard` (zip with a 200 MB zero file, compressed small → 413 `upload_archive_too_large`; a 40 MB one → 413 too) · `test_too_many_columns` (long table with 60 columns → 422) · `test_wide_table_many_periods_ok` (wide table, 3 rows × 120 monthly columns → 200, `detected.wide`) · `test_too_large_table` (> 2,000,000 cells → 422 `upload_too_large_table`) · `test_parse_slots_503` (2 parses blocked on a patched slow `read_table_bytes` → a third upload gets `503 server_busy`) · `test_more_than_20_detected` (25 value columns → `value_cols` lists 25; analyze with `null` → 400 `upload_too_many_series`) · `test_duplicate_wide_row_names` (two rows named `A` → series `A` and `A (2)`) · `test_upload_expired_410` (FrozenClock +31 min) · `test_analyze_upload_selected_cols` · `test_bad_date_col_400` · `test_upload_index_scale_default` (2 value columns → compare `scale.default "index"`) · **`test_raw_upload_never_written`** (CSV with an ignored text column containing `SECRET-MARKER-7f3a` and a value column with one unparsable cell `SECRET-MARKER-9b1c`, plus a malformed second file that makes pandas raise with input text in its message; after upload + analyze + share + export: no file under the data dir contains `SECRET-MARKER-7f3a`, and captured logs contain neither marker nor the malformed file's text — `SECRET-MARKER-9b1c` may appear in the stored result's own notices, which are part of the user's result) |
| `test_web_viewmodel.py` | `test_vm_matches_schema` (every VM produced in this file validates against `vm.schema.json`) · `test_tiles_exact_keys_order` · `test_peak_season_matches_narrative` (monthly seasonal sample and a daily series with a weekly pattern (`period == 7`): tile peak equals the narrative's; a daily series with another period → volatility tile) · `test_volatility_tile_without_season` · `test_undefined_reasons` (all-zero series and a negative-level series: `—` with non-empty `undefined_reason` for the affected tiles; no `None`/`nan` strings anywhere in the VM) · `test_display_rule` (skill 1.2 → `bands_only`; low confidence → `weak`) · `test_weak_rule_everywhere` (`bands_only` series: tiles 4–5 `sub` end with `· 참고용`, `summary_line` contains `참고용`, ranking `spark.forecast == []`, `ranking[].display == "bands_only"`) · `test_likelihood_and_prob_text_boundaries` · `test_outlook_line_rules` (the four D.2.3 vectors, ko and en) · `test_scenario_rows` (`scenario_floor` passed as `nonnegative`; `reference=None` when `expected_change_pct` is None → no `pct_text`; `title`/`reference.text`/`rows[].text` per D.2.8) · `test_reasons_detail_and_dates` (Mann-Kendall reason → statistics in `detail`; ISO dates rewritten) · `test_narrative_date_links` (W: `2023.03.06` → `2023-03-06`; M: `2024.03` → the chart date starting `2024-03`; Q: `2024 1분기` → quarter start; Y: no links) · `test_summary_line` (data-as-of suffix; sample `short` = `샘플·가상 데이터`) · `test_y_hint_relative_index` (focused Naver series with max 8 → `y_hint.max` ≈ band max, `y_note` set) · `test_en_vm_has_no_hangul_outside_names` (regex `[가-힣]` absent from every string after replacing every occurrence of the data names — series names, terms, column names, `source.filename` — with `X`; run for a sample, a Naver compare and an upload with Korean column names and a `date_order_ambiguous` notice) · `test_phase_reasons_en_translated` · `test_compare_scale_index_base` · `test_compare_source_line` (wiki compare: earliest start ~ latest end, oldest as-of) · `test_slots_over_five` (7 upload columns → 5 slots by score in request order, 2 `null`) · `test_narrative_caveats_split` |
| `test_web_fmt.py` | every row of D.1.7 (parametrised), including the exact-tie rows; plus a check that `fmt.py` never imports `numfmt.number_text` |
| `test_web_share_export.py` | `test_share_only_uploads` (sample → 400 `share_not_needed`) · `test_share_create_get_delete` (201, GET shared readonly, DELETE wrong token 403, right token 204, GET → 404 `share_not_found`) · `test_share_never_leaks_result_id` (the `result_id` string occurs in no byte of `GET /api/shared/<sid>` in either language, nor in `shares/<sid>/*`) · `test_share_outlives_result` (FrozenClock +25 h: result 404, share 200) · `test_share_expires_30d` · `test_csv_format` (BOM, CRLF, two `# ` lines, header, history then forecast rows, `.15g` numbers; weekly range end is the Sunday) · `test_csv_formula_guard` (column named `=SUM(A1)` → `'=SUM(A1)`, also in `values.csv`) · `test_values_csv` (upload compare → date × series, source comment line) · `test_content_disposition_utf8` · `test_xlsx_structure` (skip without openpyxl: rows 1–7 labels, header at row 9, summary sheet; a text cell `=SUM(A1)` has `data_type == "s"` and no leading apostrophe) · `test_export_xlsx_unavailable_501` (openpyxl hidden) · `test_html_export_csp_sandbox` · `test_html_export_has_source_paragraph` (`<p class="source">출처: 사용자 업로드 …` right after `<body>`) · `test_json_export_lang` (and has a `source` block) |
| `test_web_narrative.py` | `test_ai_job_success` (fake_llm → narrative job done, `source "llm"`, stored, later `GET /api/results` shows `narrative.ai`; the running snapshot matches C.6's narrative shape) · `test_ai_unavailable` · `test_ai_rate_limit` · `test_ai_daily_cap` · `test_ai_failure_maps_502` (llm raises `NarrativeUnavailable`) · `test_ai_readonly_on_share` (VM flags; a body with `share_id` → 403 `ai_readonly`) · `test_narrative_request_validation` (`lang: "../x"` → 400 `invalid_param` and nothing written outside `results/<id>/ai/`; `series: true`, `0.0`, `"0"`, `-1`, `99` → 404 `not_found`) · `test_ai_job_timeout` (llm sleeping past a patched 0.2 s limit → `ai_failed`) |
| `test_web_store.py` | `test_atomic_write_and_modes` (files 0600, dirs 0700) · `test_lazy_expiry` · `test_sweeper_removes_expired` · `test_store_size_eviction` · `test_corrupt_file_tolerated` |
| `test_web_security.py` | `test_cross_origin_post_403` · `test_same_origin_post_ok` · `test_origin_behind_proxy` (`trust_proxy` + `X-Forwarded-Proto: https` + `X-Forwarded-Host: trendai.example` + `Origin: https://trendai.example` → allowed; same without `trust_proxy` → 403; `public_url` set → compared with it) · `test_poll_bucket_per_job` (two jobs polled at 2/s each from one IP for 30 s → no 429) · `test_secrets_never_leak` (env `NAVER_API_HUB_KEY=SENTINEL-KEY-123` etc. with default fetchers over a fake session; run a Naver flow incl. an auth failure; assert the sentinel is absent from every response body/header and from captured logs; `repr(NaverCredentials)` masked) · `test_x_forwarded_for_ignored_by_default` / `test_trust_proxy_uses_last_hop` · `test_ids_validated` (`/api/results/../../etc` → 404) · `test_error_never_contains_traceback` |
| `test_web_pipeline_hooks.py` | `test_run_on_stage_order` (`["prepare","analyze","forecast","outlook","narrative"]`) · `test_run_on_stage_exception_propagates` · `test_run_output_unchanged_without_hook` (report_to_dict equal with/without a no-op hook) · `test_horizon_long_unchanged_outside_collect` (`pytest.warns(DataWarning)` with the same text and the caller's filename as before) · `test_horizon_long_collected_inside_collect` (no warning escapes a `collect_notices()` block) · `test_llm_narrative_from_payload_matches` (mocked client: same request content as `llm_narrative`) |
| `test_web_messages_present.py` | every code in WP1's emit tables (C.12.2, C.12.3 targets, C.12.4 incl. `table_too_many_columns`, C.12.5) and every fix action exists in `messages.MESSAGES` (fails until WP2 is merged) |
| `test_web_server.py` | `test_start_background_serves` (real socket on port 0; urllib GET `/api/health`) · `test_threaded_concurrency` (two slow requests overlap) · `test_close_drains` · `test_draining_keeps_serving` (after the first signal handler runs with a slow job: health `503 draining`, new analyze `503`, job GET still `200`; listener closes after the job ends) |

### H.2 WP2

| File | Tests |
|---|---|
| `tests/test_connectors_naver.py` | `test_build_body_minimal_and_full` (keys omitted when empty; ages sorted numerically) · `test_validation_errors` (6 groups, 6 keywords, duplicate names / empty name / 101-char name → `conn_naver_bad_filter` with `what="groupName"`, bad device/gender/age, start < 2016-01-01, start > end → the C.12.3 codes) · `test_urllib_session_thread_safe` (one instance, 8 threads × 20 GETs against the local server → all correct) · `test_hub_headers_and_url` / `test_legacy_headers_and_url` (fake session records url, headers incl. User-Agent, JSON body) · `test_credentials_from_env_precedence` (HUB over legacy; partial pairs ignored) · `test_credentials_repr_masks` · `test_parse_ratio_variants` (number, numeric string, keywords list or comma string) · `test_missing_buckets_filled` (gap → 0.0, `filled` count) · `test_partial_trailing_week_dropped` · `test_title_mismatch_bad_response` · `test_error_mapping` (401, 300 in HUB body, 429, 500, 400, invalid JSON, `URLError`, `TimeoutError` → codes; `log_detail` carries errorCode, never keys) · `test_no_secret_in_exception_text` · `test_cache_key_canonical` (age order irrelevant, group order relevant, credentials irrelevant) · `test_period_bounds` (parametrised, today = Wednesday 2026-11-25: week 5y → `(2021-11-29, 2026-11-22)` (end = last Sunday strictly before today; start 2021-11-23 snapped forward to Monday); month 5y → `(2021-11-01, 2026-10-31)`; date 1y → `(2025-11-25, 2026-11-24)`; all → `2016-01-04` (week) / `2016-01-01` (month, date); custom `2015-06-10~2030-01-01` (week) → `(2016-01-04, 2026-11-22)`; `5x`, `2024-13-01~2025-01-01`, start > end → `conn_bad_range`) |
| `tests/test_connectors.py` (extend) | `test_urllib_session_get_post` (local `http.server` thread: status, headers lower-cased, body, JSON echo) · `test_urllib_session_http_error_returned` (404 returned, not raised) · `test_urllib_session_timeout_raises` · `test_wiki_weekly_aggregation` (daily fake from Wed to Wed → only complete Mon–Sun weeks, Monday index, sums) · `test_wiki_user_agent_param` · `test_wiki_timeout_code` (`conn_timeout`) · `test_wiki_falls_back_to_urllib_without_requests` (monkeypatch `sys.modules['requests'] = None`) · `test_user_agent_format` |
| `tests/test_data_gtrends.py` | `test_classic_gt_csv_with_category_preamble` (series names = terms, attrs region/source/`value_kind == "relative_index_gt"`, `<1` → 0.5, notices `gt_detected` + `gt_lt1_half` with n) · `test_gt_without_preamble` · `test_gt_korean_headers` (`카테고리: 모든 카테고리`, `주,선크림: (대한민국)`) · `test_gt_ispartial_dropped` · `test_non_gt_file_lt1_unchanged` (generic CSV with `<1` still reported unreadable, no GT notices) · `test_load_table_bytes_equals_load_table` (every fixture used by `tests/test_data.py` that is written to disk, **plus an `.xlsx` fixture with separate year and month columns and a blank cell** (skip without openpyxl): identical series and attrs; the xlsx frame keeps native types) · `test_read_table_bytes_xlsx` (skip without openpyxl) · `test_read_table_bytes_meta` (format, encoding `cp949`, decimal_comma) · `test_read_table_bytes_never_writes` (monkeypatch `open`/`Path.write_bytes` to fail → still works) · `test_max_columns_checked_before_parse` (6,000-field header with `max_columns=5000` → `table_too_many_columns` without calling `pd.read_csv`) |
| `tests/test_data_notices.py` | `test_collect_notices_captures_and_suppresses` (inside the block: list filled, `warnings` sees nothing; outside: unchanged warnings) · `test_collect_notices_nested` · `test_collect_notices_thread_isolation` (8 threads, each preparing a series with distinct gaps inside its own block while others run without a block: each list holds exactly its own notices; `warnings.filters` identical before and after) · `test_emit_notice_stacklevel` (outside a block the warning's filename is the caller's, as before) · `test_date_cols_attr` (year + month → `['연도','월']`; single → `['날짜']`; wide → `[]`) · `test_wide_duplicate_names` (`A`, `A (2)`, `A (3)`) |
| `tests/test_messages_web.py` | `test_new_codes_render_both_languages` (each C.12 code with sample params renders ko and en without `KeyError`/leftover `{`) · `test_render_web_has_no_cli_hints` (for every code in `MESSAGES`, `render_web` output contains none of `--`, `파이썬`, `Python`, `trendai.`; params filled with dummies) · `test_render_web_msg_joined` (`join(date_order_ambiguous, dates_unreadable_rows)` → both parts in ko and en; en contains no Hangul from the `label` param) · `test_render_web_msg_nested_and_L` (`series_failed` with a nested Msg `detail` uses web wording; `gaps_interpolated.unit` and `col_values_unreadable.edge` (L) resolved) · `test_render_web_msg_lists_and_examples` (`cols=['번호','순번']` → `‘번호’, ‘순번’`; `examples="'a', 'b'"` → `‘a’, ‘b’`) · `test_josa` (‘번호’+은/는 → `‘번호’는`; 104주+으로/로 → `104주로`; 36개월 → `36개월로`; 10년 → `10년으로`; CP949 → `CP949로`; UTF-16 → `UTF-16으로`; 수능+이/가 → `수능이`) · `test_fix_labels_exist` (every action of C.12.6) · `test_existing_render_unchanged` (`render` of old codes byte-identical to before) |


### H.3 WP3 — `tests/js/*.test.mjs` (Node 22 built-in runner, no browser; imports the pure modules by relative path)

| File | Tests |
|---|---|
| `fmt.test.mjs` | every D.1.7 row, the exact-tie rows included (`0.125` → `0.13`, `1234.5` → `1,235`, `71.25` → `71.3`, `9999.5` → `1만`/`10K`); `roundHalfAway` against 1,000 random doubles compared with the same values' expected strings exported from Python (`tests/js/fixtures/round_vectors.json`, written by WP3 with the D.1 Python recipe); `dateLabel`/`rangeLabel` for all five frequencies in both languages (W end = Sunday); axis label rules |
| `state.test.mjs` | `q` grammar: `q=%EC%84%A0+%EC%9E%90,%ED%8C%A8` → `[["선","자"],["패"]]`; literal `+` and `,` inside terms round-trip; `parse(serialize(s)) == s` for 30 generated states; defaults omitted; at most 5 series kept; `toAnalyzeBody` matches C.5; `/` with `src`+`q` parses as a pre-fill (no analysis); `reanalyze_single` keeps every analysis param but `q` |
| `md.test.mjs` | `<script>` in text stays text; nested bullets; `**bold**`; headings; date links exactly for the keys of the given `dateLinks` map (incl. `2024.03` and `2024 1분기` keys, longest match first) and nothing else |
| `schema.test.mjs` | every `static/dev/fixtures/*.json` validates against `../../trendai/web/vm.schema.json` with the ~100-line validator (B.3.3); skipped with a message while the schema file does not exist |
| `ime.test.mjs` | `shouldCommit(event)` from the pure `js/ime.js` returns false for `isComposing: true` and for `keyCode: 229`, true for `Enter`/`,` otherwise |
| `scale.test.mjs` | nice ticks (1-2-5) counts 4–6 / 3–4; zero inclusion rule; relative index 0–100 (extends above 100); log decade ticks; anomaly range exclusion rule |
| `i18n.test.mjs` | ko and en key sets identical; no empty strings; every `{placeholder}` present in both languages; every key referenced in `js/**` (regex `t\('([^']+)'`) exists |

### H.4 WP4 — E2E and integration

#### H.4.1 Infrastructure

* `tests/e2e/server.py`: `python tests/e2e/server.py --naver on|off --ai on|off [--data-dir DIR]` →
  `logging.basicConfig(level=logging.INFO, stream=sys.stderr)`, the E.13 warning filters,
  `trendai.web.install_warning_baseline()`, `make_test_app(data_dir or a temp dir, naver=…, ai=…)` (package static
  dir) + `start_background(app, "127.0.0.1", 0)`; prints `READY <base_url>` on stdout with `flush=True` (a reader on a
  pipe would otherwise block); exits on SIGTERM or when stdin closes (temp dir removed).
* `tests/e2e/helpers.mjs`: `startServer(opts)` spawns it with `process.env.TRENDAI_E2E_PYTHON || "python3"`,
  waits for `READY` (timeout 30 s), returns `{url, stop}`; `newPage(browser, {viewport, colorScheme})`; download helper.
* `tests/e2e/*.test.mjs` use `node:test` and `require('playwright')` via `createRequire(import.meta.url)`
  (`chromium.launch()`). Checked in this environment: Node 22 with the global `playwright` 1.63 module and Chromium
  under `/opt/pw-browsers` launch fine; the Python `playwright` package is **not** installed, so E2E is written in Node.
  A globally installed module is found by `createRequire` **only through `NODE_PATH`** (verified: without it,
  `MODULE_NOT_FOUND`).
* `tests/e2e/test_e2e.py` (pytest wrapper, marker `e2e`, exempt from the 60 s rule): builds the child environment
  once — `os.environ` plus, when `NODE_PATH` is unset, `NODE_PATH = $(npm root -g)` (skipped if `npm` is missing) and
  `TRENDAI_E2E_PYTHON=sys.executable` — and uses that **same environment** for the skip probe `node -e
  "require('playwright')"` and for the runs `node --test tests/js` and `node --test --test-concurrency=1 tests/e2e`;
  skips when `node` is missing or the probe fails; asserts exit code 0 (stdout/stderr attached on failure).
  docs/WEB.md's test section documents `NODE_PATH`.
* Fixtures: `tests/e2e/fixtures/sales.csv` (월, `날짜,매출,방문수,번호,메모`, 60 rows, `메모` text column),
  `gtrends.csv` (classic preamble, 2 terms, a few `<1`).

#### H.4.2 Fake fetcher behaviour (implemented by WP1 in `trendai/web/testing.py`, relied on by WP4)

* **FakeNaver** `(groups, start, end, time_unit, device, gender, ages)`: builds the full grid (week: Mondays from
  `start`; month: month starts; date: days). Per group, with `c = zlib.crc32("+".join(keywords).encode())` (the
  keywords, not the wire name `g1`…, so each term keeps its own shape whatever its position):
  `base = 50 + 30·sin(2π·(month − 7 + c % 3)/12)` (peak in July ± c % 3 months) times growth
  `1 + 0.004·t·((c % 5) − 2)` (t = position), floored at 0.5; groups with any keyword containing `저검색` are
  multiplied by 0.03; all groups scaled together so the overall max is exactly 100, rounded to 5 decimals. Keyword
  `에러` → raises `ConnectorError(msg("conn_naver_server", status=500))`; `인증오류` → `conn_naver_auth` (401);
  `느린` → sleeps 2 s first. Returns a `NaverResult`-like object (`series`, `endpoint="hub"`, `time_unit`, `start`,
  `end`, `filled` all 0, `raw_max=100.0`; `series` keyed by the received group names), appends `{groups, start, end,
  time_unit, device, gender, ages}` to `.calls`. It raises `ConnectorError(msg("conn_naver_bad_filter",
  what="groupName", value=…))` if a group name is not `g1`…`g5` in order, so a regression to display names fails tests.
* **FakeWiki** `(title, lang, start, end, granularity)`: monthly/daily/weekly grid; values
  `10000·(1 + 0.3·sin(2π·month/12 + c)) · (1 + 0.003·t)`; title `없는문서` → `ConnectorError(msg("conn_wiki_404",
  title=title, lang=lang))`; `느린문서` → sleeps 2 s; records `.calls`.
* **fake_llm** `(payload, lang)`: returns `"**AI 테스트 해석** — {name}\n\n## {h1}\n- 테스트 문장입니다.\n\n## {h2}\n- …\n\n## {h3}\n- …\n\n## {h4}\n- 테스트 주의입니다."` with the
  four `SECTION_HEADINGS[lang]`.
* `make_test_app(..., naver=False)` sets `fetchers.naver = None`; `ai=False` sets `fetchers.llm = None`.

#### H.4.3 E2E scenarios (each an independent `test(...)`, Chromium, `hl=ko` unless stated)

| # | Name | Steps → assertions |
|---|---|---|
| 1 | home → sample detail | open `/`; click sample chip `성장형` → URL `/explore?src=sample&q=growing`; hero phase badge text `성장기`; 6 tiles; `[role=img] desc` non-empty; badge `가상 데이터` visible; source line contains `가상 데이터`; no console errors |
| 2 | compare 3 samples | `/samples`; check `emerging`, `fad`, `seasonal`; click `선택한 샘플 비교 (3)` → compare view; 3 legend chips with distinct slot icons; 3 ranking cards with ranks 1–3; toggle `표로 보기` → URL has `view=table`, table with 3 rows; back to chart; click card 1 `자세히` → URL has `focus=`, detail hero visible; browser back → compare view restored without a new `POST /api/analyze` (request counter) |
| 3 | upload CSV → confirm → results | `/upload`; set input file `sales.csv`; confirm step shows `날짜` selected, `번호` unchecked with `순번으로 보여요`, `메모` excluded `숫자가 아니에요`; uncheck `방문수`; `분석하기` → URL `/r/<22 chars>`; detail of `매출`; home shows `sales.csv` under recent analyses |
| 4 | wiki with fake fetcher | `/explore?src=wiki&q=인공지능` → detail; source line contains `Wikimedia Pageviews`; then `/explore?src=wiki&q=인공지능,없는문서` → compare with one card and one error card `위키백과에 ‘없는문서’ 문서가 없어요…` and buttons `위키백과에서 제목 찾기`, `네이버로 보기` |
| 5 | Naver unconfigured | server `--naver off`; home: Naver switch shows `준비 중` and is `aria-disabled`; tooltip text = `source.naver.unconfigured`; the example chips are the wiki list (first chip `자외선 차단제`), clicking it → detail with a Wikipedia source line; `/explore?src=naver&q=선크림` → error card `네이버 검색어 트렌드는 아직 연결되지 않았어요.` with buttons `위키백과로 보기` and `파일 올리기`; clicking `위키백과로 보기` → URL `/?src=wiki&q=선크림`, Wikipedia selected, a `선크림` chip in the search box, hint `search.hint.wiki_title` visible, and **no** `POST /api/analyze` sent (request counter) |
| 6 | Naver configured compare | server `--naver on`; `/explore?src=naver&q=선크림,패딩,저검색어` → compare; exactly one fresh Naver call (the server's stderr contains exactly one job log line with `naver=fresh`; `tests/e2e/server.py` configures INFO logging to stderr and `helpers.mjs` captures it); the compare source line is visible under the chart; every ranking card's outlook line contains `상승 확률`; the `기준 시점=100` scale option is disabled; card for `저검색어` shows the low-resolution warning + `이 검색어만 다시 분석` (its URL keeps `range`/`unit` and has `q=저검색어`); `표로 보기` → the values table has **no** `CSV로 받기` control and shows the `export_naver_raw` text; focus `선크림` → `AI로 다시 쓰기` disabled with `네이버 데이터는 이용약관에 따라 AI로 보내지 않아요.`; export menu: `HTML 리포트` disabled |
| 7 | share link | upload `sales.csv`, analyse; `공유` → dialog with `/s/…` URL; open that URL in a new browser context → banner `읽기 전용 공유 링크`, same hero sentence, no `공유`/`AI` buttons; back in the first context `링크 삭제` → confirm; the second context reloads → `공유 링크가 만료됐거나 삭제됐어요.` |
| 8 | export CSV | sample detail; `내보내기 ▾` → `CSV` → download; bytes start with `EF BB BF`; line 3 header starts `날짜,실제값,추세,예측값`; contains a row with an empty `실제값` and a forecast value; for the Naver detail (scenario 6 server) the header starts `날짜,추세,예측값` |
| 9 | mobile 390 px, no horizontal scroll | viewport 390×844 (`isMobile`, `hasTouch`); for `/`, sample detail, 3-sample compare, `/upload`, `/method`: `document.documentElement.scrollWidth <= 390`; tapping chip `기간` opens a bottom sheet (`role=dialog`), `Esc` closes it and focus returns to the chip; tiles are in 2 columns (two tiles share the same `offsetTop`); after scrolling 600 px on the compare view the sticky query bar's height is ≤ 56 px and tapping it opens the full bar in a bottom sheet |
| 9b | Korean IME in the search box | home; focus the search input; dispatch (via `page.evaluate`) `keydown` events with `key: 'Enter'`, `isComposing: true` and with `keyCode: 229` while the input holds `선크` → no chip is created; then a normal `Enter` with `선크림` → exactly one chip `선크림` |
| 10 | dark mode | `colorScheme: 'dark'`; body computed background `rgb(14, 14, 13)`; open settings → `라이트` → background `rgb(246, 246, 243)`; reload → still light (`trendai.theme`) |
| 11 | keyboard crosshair | sample detail; Tab until the chart wrapper is focused (`document.activeElement` has `role=group`); read-out shows the last actual date; `ArrowLeft` ×3 → read-out date changed and `#live-polite` text contains it; `Home` → first date; `End` → last forecast date; `Escape` → crosshair hidden |
| 12 | progress and cancel | `/explore?src=wiki&q=느린문서,인공지능,기계학습` → the 4-step panel appears with `role=status`; click `취소` → `분석을 취소했어요.` and a `다시 분석` button; `GET /api/jobs/<id>` reports `cancelled` |
| 13 | English UI | `/explore?src=sample&q=fad&hl=en` → `<html lang="en">`; hero/tiles/chrome contain no Hangul (regex over `main` text excluding the series name); up/down colours follow `intl` (`data-updown`) |
| 14 | AI rewrite (non-Naver) | server `--ai on`; sample detail; `AI로 다시 쓰기` → status line, then badge `AI 해석 · Claude` and text `AI 테스트 해석`; toggle back to `규칙 기반 해석` |

#### H.4.4 Other WP4 tests

* `tests/test_cli_serve.py`: `test_serve_help_korean` (`trendai serve --help` lists `--host`, `--port`, `--data-dir`,
  `--dev` with Korean help) · `test_serve_starts_and_stops` (subprocess `python -m trendai serve --port 0 --data-dir
  tmp` → reads the printed URL, GET `/api/health` 200, SIGTERM → exit 0 within 10 s) · `test_serve_warns_without_contact`
  (stderr contains the TRENDAI_CONTACT warning) · `test_serve_public_requires_contact` (`--host 0.0.0.0` without
  `TRENDAI_CONTACT` → exit 2 with the B.5 message; with `TRENDAI_ALLOW_NO_CONTACT=1` it starts) ·
  `test_serve_prints_url_flushed` (the URL line is readable from a pipe before the process exits) ·
  `test_serve_second_signal_exits_1`.
* `tests/test_packaging.py`: `test_static_index_is_package_data` (`importlib.resources.files("trendai.web") / "static"
  / "index.html"` exists) · `test_pyproject_package_data_patterns` (tomllib: `trendai.web` includes `static/**/*`,
  excludes `static/dev/**`) · `test_fonts_present_with_license` (92 `.woff2` files + `LICENSE.txt` mentioning `SIL
  Open Font License`) · `test_no_runtime_deps_added` (`[project].dependencies` still exactly numpy and pandas) ·
  `test_wsgi_import_is_lazy` (importing `trendai.web.wsgi` in a subprocess with `TRENDAI_DATA_DIR=<tmp>/x` creates
  no directory and starts no thread).

### H.5 Commands

```bash
python -m pytest                       # everything incl. E2E when node + playwright exist (marker e2e)
python -m pytest -m "not e2e"          # Python only
python -m pytest tests/test_web_*.py   # WP1
node --test tests/js                   # WP3 unit tests
NODE_PATH="$(npm root -g)" node --test --test-concurrency=1 tests/e2e   # E2E directly (global playwright)
```

---

## I. Acceptance checklist (Definition of Done)

Reviewers tick every item; an unticked item blocks the merge of the owning WP (owner in brackets).

**Scope & contracts**
- [ ] Every MVP item M1–M27 behaves as in A.1; deferrals only as listed in A.2 [all]
- [ ] Each WP touched only its files in B.1 [all]
- [ ] B.3 signatures exist exactly (names, parameters, return shapes) [WP1, WP2]
- [ ] Every endpoint of C answers with the documented status codes, bodies and headers; every error uses the C.12.1
      envelope with ko/en text and fixes [WP1]
- [ ] VMs contain exactly the D.2/D.4 keys and validate against `trendai/web/vm.schema.json`; WP3's fixtures validate
      too; no `None`, `nan`, `undefined` or Python repr strings (e.g. `['번호']`) in any text field [WP1, WP3]
- [ ] D.1.7 vectors pass in Python and JS, the exact-tie rows included [WP1, WP3]
- [ ] Every likelihood word next to a number says `상승 확률 …`; the weak-forecast rule holds on every surface (fan
      chart, comparison chart, sparklines, tiles 4–5, summary line) [WP1, WP3]

**Data & connectors**
- [ ] Naver: HUB headers/URL, legacy fallback by configuration, ≤ 5 groups, ≤ 5 keywords, completed periods only,
      Monday/Sunday snapping, missing buckets filled with a notice, one call per comparison [WP2, WP1]
- [ ] Naver cache ≤ 21 days, per-IP 20/h (5/h when low), budget blocks at 90 % of 50,000 per KST month; cache hits
      still work when blocked [WP1]
- [ ] Naver data never reaches Claude (VM, API 403, no LLM payload stored for Naver results) [WP1]
- [ ] Naver raw ratios absent from CSV/XLSX/JSON exports and `values.csv`; HTML export refused; no client-built
      file or copy control carries them; attribution is text only [WP1, WP3]
- [ ] Naver requests use the wire names `g1`…`g5`; display names are unique and never sent [WP1]
- [ ] Wikipedia UA includes version, public URL and contact; 404 gives the title-search fix; weekly unit sums complete
      weeks [WP2, WP1]
- [ ] Google Trends CSV (classic and Korean headers): preamble skipped, `<1` → 0.5 with notice, `term: (region)`
      split, source/value kind recorded [WP2]
- [ ] No web text contains CLI flags or Python hints (`render_web`, `render_web_msg`); joined messages render every
      part; list params and Korean particles read naturally [WP2]
- [ ] Notices are captured with `data.collect_notices()` only; no `warnings.catch_warnings`/`simplefilter`/
      `showwarning` call in `trendai/web/**` except `log.install_warning_baseline`; thread-isolation test passes [WP2, WP1]
- [ ] Sample results say "synthetic" everywhere (VM flag, source line, JSON export `synthetic: true`) [WP1]

**Security & privacy**
- [ ] CSP and the other headers of E.12 on every response; no inline script/style anywhere in `static/` (grep for
      `<script>` without `src`, `style=`, ` on[a-z]+=`, `innerHTML`) [WP1, WP3]
- [ ] Body limits enforced before reading; JSON content type required; Origin check on POST/DELETE; no CORS [WP1]
- [ ] Static serving only from the start-up index (traversal tests pass); `/static/dev` unavailable unless dev [WP1]
- [ ] API keys never appear in responses, logs, URLs or exception texts (sentinel test) [WP1, WP2]
- [ ] Raw upload bytes never written; marker test passes; logs carry no terms, filenames, contents, full ids [WP1]
- [ ] IDs are 128-bit `token_urlsafe(16)`; share delete tokens stored hashed; constant-time comparison; a share never
      reveals its `result_id`; store paths are built only from validated values [WP1]
- [ ] CSV (apostrophe) and XLSX (`data_type "s"`) formula-injection guards; zip-bomb guard (30 MB); upload table
      limits and 2 concurrent parses [WP1, WP2]
- [ ] Rate limits and executor caps answer 429/503 with `Retry-After`; identical in-flight queries are joined [WP1]

**Frontend & UX**
- [ ] No build step, no third-party JS/CSS, fonts self-hosted from `pretendard@1.3.9` with `LICENSE.txt` [WP3]
- [ ] All pages of F.2 work with deep links and back/forward; URL grammar round-trips (tests) [WP3]
- [ ] Home: search with chips, `+` grouping, chip editor, source switch (Naver `준비 중` state), examples, upload
      card, samples, recent analyses [WP3]
- [ ] Detail card order and content as F.6.9; compare as F.6.8; upload confirm step as F.6.10 [WP3]
- [ ] Fan chart per F.7.1 (seam, band opacities, trend dots vs forecast dashes, now line, ≤ 5 numbered badges,
      anomaly rings, display rule, y-range rule); comparison chart per F.7.2; sparklines per F.7.3 [WP3]
- [ ] Progress: skeleton < 1 s, status line, 4 honest steps, cancel, progressive fill [WP3, WP1]
- [ ] Error/empty states show the cause and working fix buttons (C.12.6 behaviours) [WP3]
- [ ] Share (uploads), copy link (public), summary copy, CSV/XLSX/PNG/HTML/JSON exports work as specified [WP3, WP1]
- [ ] ko/en switch changes chrome, VM and messages; English pages contain no Hangul except data names [WP3, WP1]
- [ ] Tokens exactly as F.8; theme system/light/dark persists; up/down colours by language with override; anomalies
      and the declining phase never use `--up`/`--down` [WP3]
- [ ] Accessibility F.9 items 1–12 verified (keyboard-only walkthrough of home → compare → detail → export; axe-like
      manual check of contrast and labels) [WP3]
- [ ] 390 px: no horizontal scroll on every page; bottom sheets; 2-column tiles [WP3]

**Integration & ops**
- [ ] `trendai serve` (Korean help, `--host/--port/--data-dir/--dev`), graceful SIGTERM (drain while serving, then
      `os._exit`), public bind refused without `TRENDAI_CONTACT` (warning on loopback), warning baseline installed
      after the CLI filters; `trendai.web.wsgi` is lazy [WP4, WP1]
- [ ] `pip install .` ships `trendai/web/static/**` minus `static/dev/**`; dependencies still only numpy + pandas [WP4]
- [ ] `Dockerfile`: `python:3.12-slim`, non-root user, `pip install --no-cache-dir . openpyxl anthropic`,
      `ENV TRENDAI_DATA_DIR=/data TRENDAI_HOST=0.0.0.0`, `VOLUME /data`, `EXPOSE 8000`, `HEALTHCHECK` with
      `python -c "import urllib.request,sys; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=3)"`,
      `CMD ["trendai", "serve"]`; `.dockerignore` excludes `.git`, `.trendai-data`, `tests`, `out`, caches [WP4]
- [ ] `docs/WEB.md` covers: quick start, every E.1 variable, Naver HUB/legacy key setup, Wikimedia contact,
      single-process rule with gunicorn/waitress commands, reverse proxy + TLS + `TRENDAI_TRUST_PROXY`/`TRENDAI_HSTS`,
      data dir and retention, budget behaviour, backups (none needed beyond shares), legal checklist (brief §5.6/§7) [WP4]
- [ ] README has a short "웹 앱" section linking `docs/WEB.md`; ARCHITECTURE has a `web/` section and the E.18 private
      dependencies [WP4]
- [ ] `python -m pytest` green (Python suites + E2E scenarios 1–14 and 9b + `tests/js`) [all]

---

## J. Decisions log

Review findings (feasibility review and design review of rev. 1) that were **not** applied, or applied only in
part, and why. Everything else from both reviews was applied in the sections named in parentheses.

| # | Finding | Decision | Why |
|---|---|---|---|
| J1 | Feasibility, blocking (warnings capture): "Option A, preferred: run each series' prepare + run + VM/export build in a `ProcessPoolExecutor` (spawn)". Also feasibility, non-blocking (capacity): "use a process pool for the per-series compute". | **Not adopted.** The blocking defect is fixed with a third variant of option B that touches no global state at all: a `ContextVar`-backed `data.collect_notices()` sink in the data layer's single emission point (`data.emit_notice`), plus `pipeline.run` inside a discarded block and a process-owner warnings baseline (B.3.1, E.6, E.13). | The ContextVar sink removes the cross-thread mixing and the leaked-recorder risk by construction, without `showwarning` hooks. A process pool would add, for the MVP: pickling of reports/VMs, cross-process progress (`current.step`) and cancellation plumbing, no way to stop a hung series short of killing the pool, ~1–2 s spawn cost and ~150–200 MB per worker, and slow, flaky unit tests (every test app would spawn workers). The throughput limit is real but bounded and now documented (E.13, docs/WEB.md: ≈ 15–20 five-series comparisons per minute per process), and the in-flight joining, Naver single-flight, result cache and upload limits remove the launch-day duplicate load the review described. A process pool remains the scale-out path once traffic shows it is needed. |
| J2 | Feasibility, non-blocking (uploads): "run parsing in the job or worker pool instead of the request thread". | **Not adopted**; the other parts of that finding (2 concurrent parses → `503`, header-first column check, cell cap, 30 MB zip guard) are applied (C.4, C.13, E.17). | `POST /api/upload` answers synchronously with the detection result that the confirm step needs; moving it into a job would add a second polling flow for a ≤ 3.5 s operation. With at most 2 concurrent parses the memory (≈ 2 × 180 MB) and the GIL share are bounded, and notice capture no longer serialises jobs behind uploads (there is no lock any more). |
| J3 | Design, non-blocking (likelihood bins): "pick the bin from the rounded value, or move bin edges to rounding midpoints". | **Not adopted.** Words keep the narrative's raw-P bins (D.1.5). | The rule-based narrative on the same page prints the exact probability with raw-P words (`narrative._probability_line`, frozen code outside every WP). Binning the card on the rounded value would make the card and the narrative of the *same series* disagree at 0.69/0.70 (card "우세", narrative "약간 우세"), which is worse than two *different* series near a boundary sharing a number with adjacent words. Since the number is now always labelled `상승 확률 약 70%`, the reader sees one measured quantity with two neighbouring descriptions, both of which are true. |
| J4 | Design, non-blocking (jargon in reasons): "map test statistics to plain words (통계적으로 뚜렷한 상승 추세)". | **Applied in part**: dates are reformatted and trailing statistics move to a `detail` sub-line (D.2.5); the sentences themselves stay the engine's. | Rewriting free-text reasons would duplicate and could contradict the phase rules in `analysis.py` (outside every WP). Structured plain-word reasons are S1 (A.2). |
| J5 | Design, blocking (DataLab upload): "drop the guidance, *or* spec DataLab support (detector, `naver_datalab_upload` source, real fixture)". | **The "drop" alternative was taken** (A.2, C.12.2, C.12.6, F.6.10, G.5). | There is no verified sample of the current DataLab export in the repo; a detector written from a description could not be tested and would misclassify silently. Recorded as an S-scope item with the restrictions it must carry. |
| J6 | Feasibility, blocking (rounding): "simplest: adopt Python's half-to-even behaviour because `number_text` is frozen; alternatively WP1 stops using `number_text`". | **The alternative was taken**: `fmt.py` reimplements the `number_text` decimal rules with exact-value half-away-from-zero rounding, identical to `fmt.js` (D.1). | Half-away is what readers expect (`12,500 → 1.3만`, the existing D.1.7 vectors) and is portable to JS through the exact decimal expansion; half-to-even would have changed several existing vectors (`1.2만`, `12K`). The only cost is that the engine's own narrative/HTML texts may differ from the VM at exact binary ties (e.g. `71.25`), which is rare and stated in D.1. |
| J7 | Feasibility, non-blocking (result cache key): "add `WEB_VM_SCHEMA` and/or a hash of the viewmodel module". | **Only the explicit constant** is used (0, E.10). | A source hash would invalidate caches and shares on comment-only edits and differ between installs; an explicit constant bumped by WP1 on any VM change is reviewable and testable (`test_vm_schema_bump_invalidates_cache`). |
| J8 | Design, non-blocking (compact numbers): "accept `1,235만` in A.3, or round to `1,200만`". | **Neither exactly**: at most 3 significant digits (`1,230만`) (D.1.1). | Matches the brief's "2–3 significant digits" while losing less than the 2-digit `1,200만`; English K/M/B already never exceed 3 digits. |
| J9 | Design, blocking (outlook line): "either `(상승 확률 약 20%)`, or show the fall probability `(하락 확률 약 80%)`". | **The first form** (D.1.5 `prob_labelled`, D.2.3). | One quantity everywhere (tile 5, ranking sort, `/method`) is easier to learn than a number whose meaning flips with the direction. |
| J10 | Feasibility, non-blocking (Naver raw-ratio policy): "either accept that the API is machine-readable, or round/strip in the API". | **Accepted as machine-readable display data** and recorded as the working legal judgement (A.3, docs/WEB.md legal checklist). | The SPA needs the values to draw the chart and read-outs; rounding would degrade the display without changing what can be scraped from the page. Files and copy controls stay free of raw ratios. |

