// Error and empty-state cards with fix buttons (spec F.6.7, C.12.6). Owner: WP3-app.
//
// A card shows an icon, the message (ApiError.text() in the current language) and up to 3 fix controls. Fix actions
// run against a context object the page provides (navigation, retry, the current state, page-specific focus helpers);
// a fix whose behaviour the context cannot provide is not shown.

import { h } from '../dom.js';
import { getLang, t } from '../i18n.js';
import { reanalyzeSingleState, serialize, switchSourceState, withState } from '../state.js';
import { copyText, exportHref, icon, srOnly } from '../ui.js';
import { toast } from './toast.js';

/**
 * @typedef {Object} FixContext
 * @property {(url: string, opts?: {replace?: boolean}) => void} navigate
 * @property {Object} [state]                       current F.2 state (needed by state-changing fixes)
 * @property {() => void} [retry]                   repeat the failed request
 * @property {() => void} [reanalyze]               re-run the current query
 * @property {() => void} [openChipEditor]          group_terms
 * @property {() => void} [focusChips]              remove_series
 * @property {() => void} [openPaste]               try_paste
 * @property {() => void} [focusDateSelect]         pick_columns
 * @property {string|null} [csvUrl]                 export_csv
 */

const RUNNERS = {
  reload: () => () => window.location.reload(),
  retry: (ctx) => ctx.retry || null,
  retry_in: (ctx) => ctx.retry || null,
  go_home: (ctx) => () => ctx.navigate('/'),
  reanalyze: (ctx) => ctx.reanalyze || (ctx.state && ctx.state.route !== 'explore' ? () => ctx.navigate('/upload') : null),
  use_wiki: (ctx) => () => ctx.navigate(serialize(switchSourceState(ctx.state || { q: [] }, 'wiki'))),
  use_naver: (ctx) => () => ctx.navigate(serialize(switchSourceState(ctx.state || { q: [] }, 'naver'))),
  upload_file: (ctx) => () => ctx.navigate('/upload'),
  reupload: (ctx) => () => ctx.navigate('/upload'),
  try_sample: (ctx) => () => ctx.navigate('/samples'),
  extend_range: (ctx) => (ctx.state && ctx.state.route === 'explore'
    ? () => ctx.navigate(serialize(withState(ctx.state, { range: 'all' }))) : null),
  unit_month: (ctx) => (ctx.state && ctx.state.route === 'explore'
    ? () => ctx.navigate(serialize(withState(ctx.state, { unit: 'month' }))) : null),
  group_terms: (ctx) => ctx.openChipEditor || null,
  remove_series: (ctx) => ctx.focusChips || null,
  show_formats: (ctx) => () => ctx.navigate('/upload#formats'),
  try_paste: (ctx) => ctx.openPaste || (() => ctx.navigate('/upload#paste')),
  pick_columns: (ctx) => ctx.focusDateSelect || null,
  use_log: (ctx) => (ctx.state ? () => ctx.navigate(serialize(withState(ctx.state, { scale: 'log' })), { replace: true }) : null),
  copy_link: () => async () => {
    const ok = await copyText(window.location.href);
    toast(ok ? t('share.copied') : t('err.clipboard'), { kind: ok ? 'info' : 'error' });
  },
  reanalyze_single: (ctx, fix) => (ctx.state && fix.params && fix.params.term
    ? () => ctx.navigate(serialize(reanalyzeSingleState(ctx.state, fix.params.term))) : null),
};

/**
 * One fix control, or null when the context cannot perform it.
 * @param {{action: string, label: string, params?: Object}} fix  from ApiError.fixList() or a notice (label resolved)
 * @param {FixContext} ctx
 * @param {{autoRetry?: boolean, retryAfter?: number|null, primary?: boolean}} [opts]
 * @returns {HTMLElement|null}
 */
export function fixControl(fix, ctx, opts = {}) {
  const cls = ['btn', opts.primary ? 'btn-primary' : 'btn-secondary'];
  if (fix.action === 'wiki_title_help') {
    const url = fix.params && fix.params.url;
    if (!url || !/^https:\/\//.test(url)) return null;
    return h('a', { class: cls, href: url, target: '_blank', rel: 'noopener noreferrer' },
      fix.label, icon('ui-external'), srOnly(` (${t('a11y.new_tab')})`));
  }
  if (fix.action === 'export_csv') {
    if (!ctx.csvUrl) return null;
    return h('a', { class: cls, href: exportHref(ctx.csvUrl, getLang()), download: '' }, fix.label);
  }
  const factory = RUNNERS[fix.action];
  const run = factory ? factory(ctx, fix) : null;
  if (!run) return null;
  const button = h('button', { type: 'button', class: cls, on: { click: () => run() } }, fix.label);
  if (fix.action === 'retry_in') startCountdown(button, fix, run, opts);
  return button;
}

/** retry_in: count down from `retry_after`; at 0 retry automatically only when allowed (naver_rate, once per query). */
function startCountdown(button, fix, run, opts) {
  let left = Math.ceil(Number((fix.params && fix.params.seconds) || opts.retryAfter || 0));
  if (!(left > 0)) return;
  button.textContent = t('fix.retry_in_seconds', { seconds: left });
  const timer = setInterval(() => {
    if (!button.isConnected) {
      clearInterval(timer);
      return;
    }
    left -= 1;
    if (left > 0) {
      button.textContent = t('fix.retry_in_seconds', { seconds: left });
      return;
    }
    clearInterval(timer);
    button.textContent = fix.label;
    if (opts.autoRetry) run();
  }, 1000);
}

/**
 * A row of fix controls (at most 3; the first is primary).
 * @param {Array<{action: string, label: string, params?: Object}>} fixes
 * @param {FixContext} ctx
 * @param {Object} [opts]  passed to fixControl
 * @returns {HTMLElement|null}
 */
export function fixRow(fixes, ctx, opts = {}) {
  const controls = (fixes || []).slice(0, 3)
    .map((fix, i) => fixControl(fix, ctx, { ...opts, primary: opts.primary !== false && i === 0 }))
    .filter(Boolean);
  return controls.length ? h('div', { class: 'fixes' }, controls) : null;
}

/** Notice/fix objects carry both labels (label_ko/label_en): resolve them for the current language. */
export function localFixes(fixes, lang = getLang()) {
  return (fixes || []).map((f) => ({
    action: f.action, params: f.params || {},
    label: (lang === 'en' ? f.label_en : f.label_ko) || f.label || f.label_ko || f.action,
  }));
}

/**
 * An error card.
 * @param {import('../api.js').ApiError} err
 * @param {FixContext} ctx
 * @param {{name?: string, compact?: boolean, autoRetry?: boolean, headingLevel?: number}} [opts]
 *        name: series name shown as the card title (per-series failures in a comparison, F.6.7)
 * @returns {HTMLElement}
 */
export function errorCard(err, ctx, opts = {}) {
  const fixes = err && typeof err.fixList === 'function' ? err.fixList() : [];
  const message = err && typeof err.text === 'function' ? err.text() : String((err && err.message) || '');
  const head = opts.name ? h(`h${opts.headingLevel || 3}`, { class: 'error-card-title' }, opts.name) : null;
  return h('section', { class: ['card', 'error-card', opts.compact && 'error-card-compact'], dataset: { code: err && err.code } },
    h('div', { class: 'error-card-icon' }, icon('ui-error')),
    h('div', { class: 'error-card-body' },
      head,
      h('p', { class: 'error-card-message' }, message),
      fixRow(fixes, ctx, { autoRetry: opts.autoRetry, retryAfter: err && err.retryAfter })));
}

/**
 * An empty state (G.9 copy): title, optional body and actions.
 * @param {{title: string, body?: string, actions?: Node[], icon?: string, level?: number, focusable?: boolean}} o
 */
export function emptyState(o) {
  return h('section', { class: 'empty-state' },
    h('div', { class: 'empty-icon' }, icon(o.icon || 'ui-info')),
    h(`h${o.level || 2}`, { class: 'empty-title', tabindex: o.focusable ? '-1' : null }, o.title),
    o.body ? h('p', { class: 'empty-body' }, o.body) : null,
    o.actions && o.actions.length ? h('div', { class: 'fixes' }, o.actions) : null);
}
