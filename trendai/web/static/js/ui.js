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
 * Split a trailing parenthetical off a line: '▲ 상승 가능성 매우 높음 (상승 확률 95% 이상)' →
 * ['▲ 상승 가능성 매우 높음 ', '(상승 확률 95% 이상)'] (the second part never wraps inside, UX-28). Pure.
 * @param {string} text
 * @returns {[string, string|null]}
 */
export function splitParenthetical(text) {
  const s = String(text == null ? '' : text);
  const m = /^(.*\S)(\s+)(\([^()]*\))$/.exec(s);
  return m ? [m[1] + m[2], m[3]] : [s, null];
}

function keepParenthetical(text) {
  const [head, tail] = splitParenthetical(text);
  return tail ? [head, h('span', { class: 'nowrap' }, tail)] : [head];
}

/**
 * A line that starts with an arrow and continues in words (hero / ranking outlook lines): the glyph is hidden, the
 * words are read as they are; a closing parenthetical stays on one line.
 */
export function arrowLine(text) {
  const { dir, glyph, rest } = splitArrow(text);
  if (!dir) return keepParenthetical(text);
  return [h('span', { class: 'glyph', 'aria-hidden': 'true' }, glyph), ' ', ...keepParenthetical(rest)];
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
 * Copy text to the clipboard (async Clipboard API, with a hidden-textarea fallback). The fallback has to focus its
 * textarea; focus goes back to the control that was focused before (A11Y-4), never to <body>.
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
  const before = document.activeElement;
  try {
    const area = h('textarea', { class: 'clipboard-proxy', readonly: true, 'aria-hidden': 'true', tabindex: '-1' });
    area.value = text;
    document.body.appendChild(area);
    area.select();
    const ok = document.execCommand('copy');
    area.remove();
    return ok;
  } catch {
    return false;
  } finally {
    if (before && before !== document.body && before.isConnected && typeof before.focus === 'function') {
      before.focus({ preventScroll: true });
    }
  }
}

/**
 * A disclosure: `button` toggles `panel` (hidden attribute), with aria-expanded / aria-controls kept in sync.
 * @param {HTMLElement} button
 * @param {HTMLElement} panel  needs an id
 * @param {(open: boolean) => void} [onToggle]
 * @returns {HTMLElement} the button
 */
export function disclosure(button, panel, onToggle) {
  button.setAttribute('aria-expanded', String(!panel.hidden));
  if (panel.id) button.setAttribute('aria-controls', panel.id);
  button.addEventListener('click', () => {
    panel.hidden = !panel.hidden;
    button.setAttribute('aria-expanded', String(!panel.hidden));
    if (onToggle) onToggle(!panel.hidden);
  });
  return button;
}

let radioCounter = 0;

/**
 * A list of native radios (one name, so the browser groups them and handles the arrow keys): settings, query-bar
 * popovers. Put it in a labelled fieldset or a labelled dialog.
 * @param {string} name
 * @param {Array<{value: string|number, label: string, hint?: string|null, disabled?: boolean}>} options
 * @param {string|number} value  the checked option
 * @param {(value: string|number) => void} onPick
 * @returns {HTMLElement}
 */
export function radioList(name, options, value, onPick) {
  const id = `${name}-${++radioCounter}`;
  return h('div', { class: 'radio-list' }, options.map((o) => h('label', { class: ['radio', o.disabled && 'is-disabled'] },
    h('input', {
      type: 'radio', name: id, value: String(o.value), disabled: o.disabled ? true : null, props: { checked: o.value === value },
      on: { change: (e) => { if (e.target.checked) onPick(o.value); } },
    }),
    h('span', null, o.label, o.hint ? h('span', { class: 'radio-hint' }, o.hint) : null))));
}

/**
 * Roving arrow-key navigation for a role="radiogroup" of buttons: Left/Up and Right/Down move to the previous / next
 * usable button (wrapping), focus it and choose it.
 * @param {HTMLElement[]} buttons
 * @param {(button: HTMLElement) => void} onChoose
 * @param {{usable?: (b: HTMLElement) => boolean, vertical?: boolean}} [opts]
 */
export function rovingRadios(buttons, onChoose, opts = {}) {
  const keys = opts.vertical === false ? { ArrowRight: 1, ArrowLeft: -1 }
    : { ArrowRight: 1, ArrowDown: 1, ArrowLeft: -1, ArrowUp: -1 };
  const usable = opts.usable || ((b) => b.getAttribute('aria-disabled') !== 'true');
  for (const b of buttons) {
    b.addEventListener('keydown', (e) => {
      const step = keys[e.key];
      if (!step) return;
      e.preventDefault();
      const list = buttons.filter(usable);
      const i = list.indexOf(e.currentTarget);
      const next = list[(i + step + list.length) % list.length];
      if (!next) return;
      next.focus();
      onChoose(next);
    });
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
