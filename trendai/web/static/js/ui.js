// Small shared UI helpers used by components and pages: sprite icons, slot swatches, phase badges, arrow glyphs with
// screen-reader text (F.9 item 4), card heads (F.6.9 "one card-head component"), live-region announcements,
// clipboard copy and layout queries. Owner: WP3-app. Needs a DOM when called; nothing runs at import time.

import { h, svg } from './dom.js';
import { t } from './i18n.js';

export const SPRITE = '/static/img/icons.svg';
export const SLOT_ICONS = Object.freeze(['circle', 'diamond', 'hexagon', 'square', 'cross']);
const ARROWS = { '▲': 'up', '▼': 'down', '▶': 'flat' };

/**
 * A sprite icon (decorative unless `label` is given).
 * @param {string} id  symbol id in img/icons.svg, e.g. 'ui-help', 'ph-growing', 'slot-diamond'
 * @param {{cls?: string|string[], label?: string}} [opts]
 * @returns {SVGSVGElement}
 */
export function icon(id, opts = {}) {
  const attrs = { class: ['icon', ...[].concat(opts.cls || [])], focusable: 'false' };
  if (opts.label) Object.assign(attrs, { role: 'img', 'aria-label': opts.label });
  else attrs['aria-hidden'] = 'true';
  return svg('svg', attrs, svg('use', { href: `${SPRITE}#${id}` }));
}

/** The slot icon name of slot 1–5, or null. */
export function slotIconName(slot) {
  return Number.isInteger(slot) && slot >= 1 && slot <= 5 ? SLOT_ICONS[slot - 1] : null;
}

/**
 * Series swatch: slot colour + slot icon (aria-hidden; the series name next to it carries the meaning, F.9).
 * Grey (no slot) series get a small grey dot.
 */
export function swatch(slot) {
  const name = slotIconName(slot);
  return h('span', { class: ['swatch', name ? `slot-${slot}` : 'slot-none'], 'aria-hidden': 'true' },
    icon(name ? `slot-${name}` : 'ui-dot'));
}

/** Visually hidden text for screen readers. */
export function srOnly(text) {
  return h('span', { class: 'sr-only' }, text);
}

/**
 * Split a leading ▲ / ▼ / ▶ off a server text (F.9 item 4).
 * @param {string} text
 * @returns {{dir: 'up'|'down'|'flat'|null, glyph: string|null, rest: string}}
 */
export function splitArrow(text) {
  const s = String(text == null ? '' : text);
  const dir = ARROWS[s[0]];
  if (!dir) return { dir: null, glyph: null, rest: s };
  return { dir, glyph: s[0], rest: s.slice(1).trim() };
}

/**
 * A value such as `▲12%`: hidden arrow glyph + visible `12%` + sr-only `12% 상승` (tiles, hero, ranking, F.9).
 * Values without an arrow are returned as plain text.
 * @param {string} text
 * @returns {Node[]}
 */
export function arrowValue(text) {
  const { dir, glyph, rest } = splitArrow(text);
  if (!dir) return [document.createTextNode(String(text == null ? '' : text))];
  const spoken = dir === 'flat' ? t('a11y.flat') : t(dir === 'up' ? 'a11y.up' : 'a11y.down', { value: rest });
  return [
    h('span', { class: 'glyph', 'aria-hidden': 'true' }, glyph),
    h('span', { 'aria-hidden': 'true' }, rest),
    srOnly(spoken),
  ];
}

/**
 * A line that starts with an arrow and continues in words (hero / ranking outlook lines): the glyph is hidden, the
 * words are read as they are.
 */
export function arrowLine(text) {
  const { dir, glyph, rest } = splitArrow(text);
  if (!dir) return [document.createTextNode(String(text == null ? '' : text))];
  return [h('span', { class: 'glyph', 'aria-hidden': 'true' }, glyph), ` ${rest}`];
}

/** CSS class for a tone ('up' | 'down' | 'neutral'). */
export function toneClass(tone) {
  return tone === 'up' ? 'tone-up' : tone === 'down' ? 'tone-down' : 'tone-neutral';
}

/**
 * Phase badge: phase icon (colour on the icon only, F.8) + name (+ 잠정).
 * @param {{label: string, name: string, icon?: string, tentative?: boolean}} phase
 * @param {{big?: boolean, tag?: string}} [opts]
 */
export function phaseBadge(phase, opts = {}) {
  const label = (phase && (phase.icon || phase.label)) || 'stable';
  return h(opts.tag || 'span', { class: ['phase-badge', opts.big && 'phase-badge-big', `ph-${label}`] },
    icon(`ph-${label}`, { cls: 'phase-icon' }),
    h('span', { class: 'phase-name' }, phase ? phase.name : ''),
    phase && phase.tentative ? h('span', { class: 'phase-tentative' }, t('common.tentative')) : null);
}

/** A small pill (synthetic badge, 참고용, reliability). */
export function pill(text, cls) {
  return h('span', { class: ['pill', cls] }, text);
}

/**
 * The shared card head (F.6.9): title, optional `?` help, optional extra controls on the right.
 * @param {{title: string|Node, level?: number, id?: string, help?: Node|null, actions?: Node[]}} opts
 */
export function cardHead(opts) {
  const tag = `h${opts.level || 2}`;
  return h('div', { class: 'card-head' },
    h('div', { class: 'card-title-row' },
      h(tag, { class: 'card-title', id: opts.id || null }, opts.title),
      opts.help || null),
    opts.actions && opts.actions.length ? h('div', { class: 'card-actions' }, opts.actions) : null);
}

/** Announce text in a live region (`#live-polite` / `#live-assertive`, F.9 item 8). */
export function announce(text, assertive = false) {
  const el = document.getElementById(assertive ? 'live-assertive' : 'live-polite');
  if (!el) return;
  el.textContent = '';
  // a new text node after a tick makes screen readers repeat identical messages
  setTimeout(() => {
    el.textContent = text;
  }, 30);
}

/**
 * Copy text to the clipboard (async Clipboard API, with a hidden-textarea fallback).
 * @returns {Promise<boolean>}
 */
export async function copyText(text) {
  try {
    if (navigator.clipboard && window.isSecureContext) {
      await navigator.clipboard.writeText(text);
      return true;
    }
  } catch {
    // fall through to the legacy path
  }
  try {
    const area = h('textarea', { class: 'clipboard-proxy', readonly: true, 'aria-hidden': 'true' });
    area.value = text;
    document.body.appendChild(area);
    area.select();
    const ok = document.execCommand('copy');
    area.remove();
    return ok;
  } catch {
    return false;
  }
}

/** True below the 640 px breakpoint (F.10). */
export function isNarrow() {
  return window.matchMedia('(max-width: 639px)').matches;
}

/** True when the user asked for reduced motion (F.9 item 11). */
export function reducedMotion() {
  return window.matchMedia('(prefers-reduced-motion: reduce)').matches;
}

/** Bytes as a short size ('18 KB', '2.4 MB'). */
export function formatBytes(n) {
  if (!Number.isFinite(n)) return '';
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${Math.round(n / 1024)} KB`;
  return `${(n / (1024 * 1024)).toFixed(1)} MB`;
}

/** An absolute URL on this origin. */
export function absoluteUrl(path) {
  return new URL(path, window.location.origin).href;
}

/** Download link attributes for a server export (`?hl=` appended, F.6.12). */
export function exportHref(url, lang) {
  return `${url}${url.includes('?') ? '&' : '?'}hl=${lang}`;
}
