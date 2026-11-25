// URL ⇄ state (spec F.2). Owner: WP3-app. Pure: no DOM, no window, no storage (F.1 rule 2).
//
// State = {route, id, path, src, q, range, unit, h, dev, gender, ages, wl, hl, scale, band, view, focus}.
//   route  'home' | 'explore' | 'upload' | 'result' (/r/<id>) | 'shared' (/s/<id>) | 'samples' | 'method' |
//          'sources' | 'privacy' | 'notfound'
//   id     result id (/r) or share id (/s), else null;  path: the raw pathname for 'notfound', else null
//   q      string[][]: series of terms (F.2 grammar)
//   src    'naver' | 'wiki' | 'sample' | null (null = config.default_source, not written to the URL)
//   h      'auto' | integer;  ages: sorted array of '1'…'11';  hl / scale / focus: null when not set
// Defaults are omitted from the URL. The `q` grammar is parsed from the raw query string, because
// URLSearchParams would turn the term separator '+' into a space.

export const MAX_SERIES = 5;
export const MAX_TERMS = 5;
export const MAX_TERM_LENGTH = 50;
export const SOURCES = Object.freeze(['naver', 'wiki', 'sample']);
export const PARAM_ORDER = Object.freeze(['src', 'q', 'range', 'unit', 'h', 'dev', 'gender', 'ages', 'wl', 'hl',
  'scale', 'band', 'view', 'focus']);

export const DEFAULTS = Object.freeze({
  src: null, q: Object.freeze([]), range: '5y', unit: 'auto', h: 'auto', dev: '', gender: '', ages: Object.freeze([]),
  wl: 'ko', hl: null, scale: null, band: 'both', view: 'chart', focus: null,
});

const STATIC_ROUTES = Object.freeze({
  '/': 'home', '/explore': 'explore', '/upload': 'upload', '/samples': 'samples', '/method': 'method',
  '/sources': 'sources', '/privacy': 'privacy',
});
const ROUTE_PATHS = Object.freeze(Object.fromEntries(Object.entries(STATIC_ROUTES).map(([p, r]) => [r, p])));
const ID_RE = /^[A-Za-z0-9_-]{22}$/;
const RANGE_RE = /^(?:1y|3y|5y|all|\d{4}-\d{2}-\d{2}~\d{4}-\d{2}-\d{2})$/;
const AGE_RE = /^(?:[1-9]|1[01])$/;
// eslint-disable-next-line no-control-regex
const CONTROL_RE = /[\u0000-\u001f\u007f]/g;

/** A fresh state for `route` with every field at its default. */
export function emptyState(route = 'home') {
  return {
    route, id: null, path: null, src: null, q: [], range: '5y', unit: 'auto', h: 'auto', dev: '', gender: '',
    ages: [], wl: 'ko', hl: null, scale: null, band: 'both', view: 'chart', focus: null,
  };
}

function safeDecode(piece) {
  try {
    return decodeURIComponent(piece);
  } catch {
    return piece;
  }
}

/** NFC + trim + no control characters (the C.5 normalisation the server applies too). */
export function cleanTerm(term) {
  return String(term == null ? '' : term).replace(CONTROL_RE, '').normalize('NFC').trim();
}

/**
 * Normalise a list of series the way the server does (C.5): terms cleaned, empty terms dropped, duplicate terms in a
 * series dropped, empty series dropped, duplicate series (same term tuple) dropped, at most 5 series kept.
 * @param {Array<Array<string>>} series
 * @returns {string[][]}
 */
export function normalizeSeries(series) {
  const out = [];
  const seen = new Set();
  for (const terms of series || []) {
    const clean = [];
    for (const term of terms || []) {
      const c = cleanTerm(term);
      if (c && !clean.includes(c)) clean.push(c);
    }
    if (!clean.length) continue;
    const key = JSON.stringify(clean);
    if (seen.has(key)) continue;
    seen.add(key);
    out.push(clean);
    if (out.length >= MAX_SERIES) break;
  }
  return out;
}

/** One series of the `q` grammar ('a+b', each term percent-decoded) → terms (not de-duplicated). */
export function parseSeries(raw) {
  return String(raw || '').split('+').map((p) => cleanTerm(safeDecode(p))).filter(Boolean);
}

/** The raw `q` value ('a+b,c') → normalised series. */
export function parseQ(raw) {
  if (!raw) return [];
  return normalizeSeries(String(raw).split(',').map(parseSeries));
}

/** Terms → one series of the `q` grammar: every term encodeURIComponent-encoded, joined with '+'. */
export function encodeSeries(terms) {
  return (terms || []).map((term) => encodeURIComponent(term)).join('+');
}

/** Series → the `q` value. */
export function encodeQ(series) {
  return (series || []).map(encodeSeries).join(',');
}

/** Display name of a series as the server builds it (C.5, E.6.1 before de-duplication): terms joined by ' + '. */
export function seriesName(terms) {
  return (terms || []).join(' + ');
}

/** The raw (still percent-encoded) value of the first `q=` parameter of a query string. */
function rawQ(search) {
  const s = String(search || '').replace(/^\?/, '');
  for (const piece of s.split('&')) {
    if (piece.startsWith('q=')) return piece.slice(2);
  }
  return '';
}

function routeOf(pathname) {
  let path = String(pathname || '/');
  if (path.length > 1 && path.endsWith('/')) path = path.slice(0, -1);
  if (STATIC_ROUTES[path]) return { route: STATIC_ROUTES[path], id: null, path: null };
  const m = /^\/(r|s)\/([^/]+)$/.exec(path);
  if (m && ID_RE.test(m[2])) return { route: m[1] === 'r' ? 'result' : 'shared', id: m[2], path: null };
  return { route: 'notfound', id: null, path: String(pathname || '/') };
}

/** Sorted, de-duplicated list of valid Naver age codes. */
export function normalizeAges(list) {
  const set = new Set((list || []).map(String).filter((a) => AGE_RE.test(a)));
  return [...set].sort((a, b) => Number(a) - Number(b));
}

/**
 * Parse a location into a State (F.2). Invalid parameter values fall back to their defaults.
 * @param {string} pathname
 * @param {string} search  location.search (with or without the leading '?')
 * @returns {Object} State
 */
export function parseLocation(pathname, search) {
  const state = emptyState();
  Object.assign(state, routeOf(pathname));
  const params = new URLSearchParams(String(search || '').replace(/^\?/, ''));
  const get = (k) => params.get(k);

  const src = get('src');
  state.src = SOURCES.includes(src) ? src : null;
  state.q = parseQ(rawQ(search));
  const range = get('range');
  state.range = range && RANGE_RE.test(range) ? range : '5y';
  const unit = get('unit');
  state.unit = ['auto', 'date', 'week', 'month'].includes(unit) ? unit : 'auto';
  const h = get('h');
  state.h = h && /^[1-9][0-9]{0,3}$/.test(h) ? Number(h) : 'auto';
  const dev = get('dev');
  state.dev = dev === 'pc' || dev === 'mo' ? dev : '';
  const gender = get('gender');
  state.gender = gender === 'm' || gender === 'f' ? gender : '';
  state.ages = normalizeAges((get('ages') || '').split(','));
  const wl = get('wl');
  state.wl = wl && /^[a-z]{2,3}$/.test(wl) ? wl : 'ko';
  const hl = get('hl');
  state.hl = hl === 'ko' || hl === 'en' ? hl : null;
  const scale = get('scale');
  state.scale = ['raw', 'index', 'log'].includes(scale) ? scale : null;
  const band = get('band');
  state.band = ['80', '95', 'both', 'off'].includes(band) ? band : 'both';
  const view = get('view');
  state.view = view === 'table' ? 'table' : 'chart';
  const focus = get('focus');
  state.focus = focus && focus.trim() ? focus : null;
  return state;
}

/** The pathname of a state's route. */
export function pathOf(state) {
  if (state.route === 'result') return `/r/${state.id}`;
  if (state.route === 'shared') return `/s/${state.id}`;
  if (state.route === 'notfound') return state.path || '/';
  return ROUTE_PATHS[state.route] || '/';
}

/**
 * State → path + query (F.2 order, defaults omitted).
 * @param {Object} state
 * @returns {string}
 */
export function serialize(state) {
  const parts = [];
  const s = { ...emptyState(state.route), ...state };
  if (s.src) parts.push(`src=${s.src}`);
  if (s.q && s.q.length) parts.push(`q=${encodeQ(s.q)}`);
  if (s.range !== '5y') parts.push(`range=${encodeURIComponent(s.range)}`);
  if (s.unit !== 'auto') parts.push(`unit=${s.unit}`);
  if (s.h !== 'auto' && s.h != null) parts.push(`h=${s.h}`);
  if (s.dev) parts.push(`dev=${s.dev}`);
  if (s.gender) parts.push(`gender=${s.gender}`);
  if (s.ages && s.ages.length) parts.push(`ages=${s.ages.join(',')}`);
  if (s.wl !== 'ko') parts.push(`wl=${s.wl}`);
  if (s.hl) parts.push(`hl=${s.hl}`);
  if (s.scale) parts.push(`scale=${s.scale}`);
  if (s.band !== 'both') parts.push(`band=${s.band}`);
  if (s.view !== 'chart') parts.push(`view=${s.view}`);
  if (s.focus) parts.push(`focus=${encodeURIComponent(s.focus)}`);
  return pathOf(s) + (parts.length ? `?${parts.join('&')}` : '');
}

/** The source in effect: the URL's `src`, else the config default. */
export function effectiveSource(state, defaultSource = 'naver') {
  return state.src || (SOURCES.includes(defaultSource) ? defaultSource : 'naver');
}

/**
 * Canonical key of the analysis-relevant fields (`src q range unit h dev gender ages wl`); fields that do not apply
 * to the source are left out, so equivalent URLs share one in-memory cache entry.
 * @param {Object} state
 * @param {{defaultSource?: string}} [opts]
 * @returns {string}
 */
export function queryKey(state, opts = {}) {
  const src = effectiveSource(state, opts.defaultSource);
  const key = { src, q: state.q, h: state.h };
  if (src !== 'sample') Object.assign(key, { range: state.range, unit: state.unit });
  if (src === 'naver') Object.assign(key, { dev: state.dev, gender: state.gender, ages: state.ages });
  if (src === 'wiki') key.wl = state.wl;
  return JSON.stringify(key);
}

/**
 * The C.5 request body of a search state (naver / wiki / sample).
 * @param {Object} state
 * @param {{defaultSource?: string}} [opts]
 * @returns {Object}
 */
export function toAnalyzeBody(state, opts = {}) {
  return {
    source: effectiveSource(state, opts.defaultSource),
    series: state.q.map((terms) => ({ terms: [...terms] })),
    range: state.range,
    unit: state.unit,
    h: state.h,
    dev: state.dev,
    gender: state.gender,
    ages: [...state.ages],
    wl: state.wl,
    upload_id: null,
    date_col: null,
    value_cols: null,
    freq: 'auto',
    unit_label: null,
  };
}

/** A copy of `state` with `patch` applied (arrays copied). */
export function withState(state, patch) {
  const next = { ...state, ...patch };
  next.q = (next.q || []).map((terms) => [...terms]);
  next.ages = [...(next.ages || [])];
  return next;
}

/** Keep only the language of `state` (navigation to a page that has no parameters of its own). */
export function pageState(route, from) {
  const s = emptyState(route);
  if (from && from.hl) s.hl = from.hl;
  return s;
}

/**
 * The `reanalyze_single` fix (C.12.6): /explore with every analysis parameter of `state` (src range unit h dev gender
 * ages wl, and hl) kept and only `q` replaced by the series in `term` (one series in the q grammar).
 * @param {Object} state
 * @param {string} term
 * @returns {Object} State
 */
export function reanalyzeSingleState(state, term) {
  const next = emptyState('explore');
  for (const k of ['src', 'range', 'unit', 'h', 'dev', 'gender', 'ages', 'wl', 'hl']) next[k] = state[k];
  next.ages = [...(state.ages || [])];
  next.q = normalizeSeries([parseSeries(term)]);
  return next;
}

/**
 * The `use_wiki` / `use_naver` fixes (C.12.6): home with that source selected and the search box pre-filled with the
 * first term of each series (never run).
 * @param {Object} state
 * @param {'wiki'|'naver'} src
 * @returns {Object} State
 */
export function switchSourceState(state, src) {
  const next = pageState('home', state);
  next.src = src;
  next.q = normalizeSeries((state.q || []).map((terms) => terms.slice(0, 1)));
  return next;
}
