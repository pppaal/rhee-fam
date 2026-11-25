// localStorage wrappers (spec F.11). Owner: WP3-app.
//
// Every access is wrapped in try/catch: storage can be missing, full or blocked (private windows, previews), and the
// app must work without it. Nothing is read at import time.

export const KEYS = Object.freeze({
  hl: 'trendai.hl',
  theme: 'trendai.theme',
  updown: 'trendai.updown',
  recent: 'trendai.recent',
  shares: 'trendai.shares',
  dismissNaver: 'trendai.dismiss.naverIndex',
});

export const RECENT_MAX = 8;
const RECENT_PRUNE_MS = 7 * 24 * 3600 * 1000;

function storage() {
  try {
    return globalThis.localStorage || null;
  } catch {
    return null;
  }
}

/** Raw string value, or null. */
export function getItem(key) {
  try {
    const s = storage();
    return s ? s.getItem(key) : null;
  } catch {
    return null;
  }
}

/** Store a string (null removes the key). Returns false when storage is unavailable. */
export function setItem(key, value) {
  try {
    const s = storage();
    if (!s) return false;
    if (value == null) s.removeItem(key);
    else s.setItem(key, String(value));
    return true;
  } catch {
    return false;
  }
}

/** Parsed JSON value, or `fallback` when absent or unreadable. */
export function getJSON(key, fallback) {
  const raw = getItem(key);
  if (raw == null) return fallback;
  try {
    return JSON.parse(raw);
  } catch {
    return fallback;
  }
}

/** Store a JSON value. */
export function setJSON(key, value) {
  try {
    return setItem(key, JSON.stringify(value));
  } catch {
    return false;
  }
}

const pick = (value, allowed) => (allowed.includes(value) ? value : null);

export const getHl = () => pick(getItem(KEYS.hl), ['ko', 'en']);
export const setHl = (hl) => setItem(KEYS.hl, pick(hl, ['ko', 'en']));
/** 'light' | 'dark' | null (null = follow the system). */
export const getTheme = () => pick(getItem(KEYS.theme), ['light', 'dark']);
export const setTheme = (theme) => setItem(KEYS.theme, pick(theme, ['light', 'dark']));
/** 'kr' | 'intl' | null (null = by language). */
export const getUpdown = () => pick(getItem(KEYS.updown), ['kr', 'intl']);
export const setUpdown = (updown) => setItem(KEYS.updown, pick(updown, ['kr', 'intl']));
export const isNaverExplainerDismissed = () => getItem(KEYS.dismissNaver) === '1';
export const dismissNaverExplainer = () => setItem(KEYS.dismissNaver, '1');

function timeOf(iso) {
  const ms = Date.parse(iso || '');
  return Number.isFinite(ms) ? ms : null;
}

/**
 * Recent analyses (F.6.20): `{title, url, at, expires_at}`, newest first. Entries whose `expires_at` passed more
 * than 7 days ago are pruned on read.
 * @param {number} [now=Date.now()]
 * @returns {Array<{title: string, url: string, at: string, expires_at: string|null}>}
 */
export function getRecent(now = Date.now()) {
  const list = getJSON(KEYS.recent, []);
  if (!Array.isArray(list)) return [];
  const kept = list.filter((e) => e && typeof e.url === 'string' && typeof e.title === 'string'
    && !(timeOf(e.expires_at) != null && timeOf(e.expires_at) + RECENT_PRUNE_MS < now)).slice(0, RECENT_MAX);
  if (kept.length !== list.length) setJSON(KEYS.recent, kept);
  return kept;
}

/** Add (or move to the top) a recent analysis, de-duplicated by URL, at most 8 kept. */
export function addRecent(entry, now = Date.now()) {
  if (!entry || !entry.url) return;
  const item = {
    title: String(entry.title || ''), url: String(entry.url), at: entry.at || new Date(now).toISOString(),
    expires_at: entry.expires_at || null,
  };
  const list = getRecent(now).filter((e) => e.url !== item.url);
  list.unshift(item);
  setJSON(KEYS.recent, list.slice(0, RECENT_MAX));
}

/** Remove a recent analysis by URL. */
export function removeRecent(url) {
  setJSON(KEYS.recent, getRecent().filter((e) => e.url !== url));
}

/**
 * Share links created in this browser (F.11): `{result_id: {share_id, url, delete_token, expires_at}}`; expired
 * entries are pruned on read.
 */
export function getShares(now = Date.now()) {
  const all = getJSON(KEYS.shares, {});
  if (!all || typeof all !== 'object' || Array.isArray(all)) return {};
  const kept = {};
  let changed = false;
  for (const [rid, info] of Object.entries(all)) {
    const exp = info && timeOf(info.expires_at);
    if (info && info.share_id && info.delete_token && (exp == null || exp > now)) kept[rid] = info;
    else changed = true;
  }
  if (changed) setJSON(KEYS.shares, kept);
  return kept;
}

/** The stored share of a result, or null. */
export function getShare(resultId) {
  return getShares()[resultId] || null;
}

/** Remember a share created for `resultId`. */
export function saveShare(resultId, info) {
  const all = getShares();
  all[resultId] = {
    share_id: info.share_id, url: info.url, delete_token: info.delete_token, expires_at: info.expires_at,
  };
  setJSON(KEYS.shares, all);
}

/** Forget the share of `resultId`. */
export function forgetShare(resultId) {
  const all = getShares();
  delete all[resultId];
  setJSON(KEYS.shares, all);
}
