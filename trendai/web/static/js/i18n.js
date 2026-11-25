// i18n (spec F.4). Owner: WP3-app.
//
// `t(key, params)` looks up CATALOG[hl][key], falls back to CATALOG.ko[key], then to the key itself (with one
// console.warn per missing key). Placeholders are `{name}`. Pure except setLang(), which also sets <html lang> when a
// document exists; nothing touches the DOM at import time (F.1 rule 2), so Node tests can import this module.

import { CATALOG } from './catalog.js';

export const LANGS = Object.freeze(['ko', 'en']);

let current = 'ko';
const warned = new Set();

/** Normalise anything to 'ko' | 'en' | null. */
export function normalizeLang(value) {
  return value === 'ko' || value === 'en' ? value : null;
}

/**
 * hl resolution (F.4): URL `hl` → stored `trendai.hl` → any navigator.languages entry starting with `ko` → 'ko',
 * else 'en'. Pure: the caller passes the three inputs.
 * @param {{url?: string|null, stored?: string|null, languages?: readonly string[]|null}} inputs
 * @returns {'ko'|'en'}
 */
export function resolveLang({ url = null, stored = null, languages = null } = {}) {
  const fromUrl = normalizeLang(url);
  if (fromUrl) return fromUrl;
  const fromStore = normalizeLang(stored);
  if (fromStore) return fromStore;
  const list = Array.isArray(languages) ? languages : [];
  return list.some((l) => typeof l === 'string' && l.toLowerCase().startsWith('ko')) ? 'ko' : 'en';
}

/** The current UI language. */
export function getLang() {
  return current;
}

/**
 * Set the current language; also sets `<html lang>` when running in a browser (F.9 item 10).
 * @param {'ko'|'en'} lang
 * @returns {'ko'|'en'} the language now in effect
 */
export function setLang(lang) {
  current = normalizeLang(lang) || 'ko';
  if (typeof document !== 'undefined' && document.documentElement) document.documentElement.lang = current;
  return current;
}

/**
 * Replace `{name}` placeholders. Unknown placeholders are left as they are; null/undefined params become ''.
 * @param {string} template
 * @param {Object} [params]
 * @returns {string}
 */
export function format(template, params) {
  if (!params) return template;
  return template.replace(/\{([A-Za-z0-9_]+)\}/g, (whole, name) => {
    if (!Object.prototype.hasOwnProperty.call(params, name)) return whole;
    const v = params[name];
    return v == null ? '' : String(v);
  });
}

/**
 * Catalog lookup in the current language (or `lang` when given).
 * @param {string} key
 * @param {Object} [params]
 * @param {'ko'|'en'} [lang]
 * @returns {string}
 */
export function t(key, params, lang) {
  const hl = normalizeLang(lang) || current;
  let text = CATALOG[hl] && CATALOG[hl][key];
  if (text == null) text = CATALOG.ko[key];
  if (text == null) {
    if (!warned.has(key)) {
      warned.add(key);
      if (typeof console !== 'undefined') console.warn(`i18n: missing catalog key "${key}"`);
    }
    return key;
  }
  return format(text, params);
}

/** True when the catalog has `key` (in Korean, the reference language). */
export function has(key) {
  return Object.prototype.hasOwnProperty.call(CATALOG.ko, key);
}
