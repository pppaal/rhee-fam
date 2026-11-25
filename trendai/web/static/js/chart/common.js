// Plumbing shared by the fan and comparison charts (spec F.7, F.7.4). Owner: WP3-chart. Internal module.
// Nothing runs at import time; the functions need a DOM.

import { h, svg, clear } from '../dom.js';
import { slotIconPath } from './geom.js';

/** Container width from which the charts use the wide layout (F.7.1, F.10). */
export const WIDE_PX = 640;

let seq = 0;
/** A document-unique id for <title>/<desc>/<clipPath> references. */
export function uid(prefix) {
  seq += 1;
  return `${prefix}-${seq.toString(36)}`;
}

/** Content-box width of `el` in whole px (the charts fill their container's content box). */
export function contentWidth(el) {
  if (!el || !el.isConnected) return 0;
  const cs = getComputedStyle(el);
  const pad = (parseFloat(cs.paddingLeft) || 0) + (parseFloat(cs.paddingRight) || 0);
  return Math.max(0, Math.floor(el.clientWidth - pad));
}

/**
 * Call `cb(width)` when the content width of `el` changes, debounced (F.7: ResizeObserver, 100 ms).
 * @param {Element} el
 * @param {(width: number) => void} cb
 * @param {number} [delay=100]
 * @returns {() => void} stop observing
 */
export function observeWidth(el, cb, delay = 100) {
  if (typeof ResizeObserver === 'undefined') return () => {};
  let last = Math.floor(el.clientWidth);
  let timer = null;
  const ro = new ResizeObserver((entries) => {
    const w = Math.floor(entries[entries.length - 1].contentRect.width);
    if (w === last) return;
    last = w;
    clearTimeout(timer);
    timer = setTimeout(() => cb(w), delay);
  });
  ro.observe(el);
  return () => {
    clearTimeout(timer);
    ro.disconnect();
  };
}

/** Default live-region writer for the crosshair read-out (index.html's #live-polite). */
export function defaultAnnounce(text) {
  const el = typeof document !== 'undefined' && document.getElementById('live-polite');
  if (el) el.textContent = text;
}

/** Join read-out parts with the middle dot used across the product. */
export function joinParts(parts) {
  return parts.filter((p) => p != null && p !== '').join(' · ');
}

/**
 * A small line key for read-outs and legends: `solid` (history), `dashed` (forecast) or `band`, in the colour of
 * `cls` (e.g. 'chart-key-hist', 'chart-slot-2').
 * @param {'solid'|'dashed'|'band'} style
 * @param {string|string[]} cls
 * @returns {SVGSVGElement}
 */
export function lineKey(style, cls) {
  const shape = style === 'band'
    ? svg('rect', { class: 'chart-key-band', x: 1, y: 1, width: 16, height: 8, rx: 2 })
    : svg('line', { class: ['chart-key-line', style === 'dashed' && 'is-dashed'], x1: 1.5, x2: 16.5, y1: 5, y2: 5 });
  return svg('svg', {
    class: ['chart-key', cls], width: 18, height: 10, viewBox: '0 0 18 10', 'aria-hidden': 'true', focusable: 'false',
  }, shape);
}

/**
 * A slot icon glyph (●◆⬢■✚ drawn as paths, so it also survives PNG export) for legends and read-outs.
 * @param {string|null} icon  circle | diamond | hexagon | square | cross | null (null → a small dot)
 * @param {string|string[]} cls  colour class (chart-slot-N / chart-grey)
 * @param {number} [size=12]
 * @returns {SVGSVGElement}
 */
export function slotGlyph(icon, cls, size = 12) {
  const r = icon ? size * 0.36 : size * 0.22;
  return svg('svg', {
    class: ['chart-glyph', cls], width: size, height: size, viewBox: `0 0 ${size} ${size}`,
    'aria-hidden': 'true', focusable: 'false',
  }, svg('path', { class: 'chart-glyph-shape', d: slotIconPath(icon || 'circle', size / 2, size / 2, r) }));
}

/**
 * Fill the fixed read-out bar (F.7.4).
 * @param {HTMLElement} el      .chart-readout
 * @param {{date: string, rows: Array<{key?: Node|null, name?: string, text: string, cls?: string}>, note?: string|null,
 *          idle: boolean}} content
 */
export function renderReadout(el, content) {
  el.classList.toggle('is-idle', Boolean(content.idle));
  clear(el).append(
    h('span', { class: 'chart-readout-date' }, content.date),
    ...content.rows.map((row) => h('span', { class: ['chart-readout-row', row.cls] }, row.key || null,
      row.name ? h('span', { class: 'chart-readout-name' }, row.name) : null,
      h('span', { class: 'chart-readout-value' }, row.text))),
  );
  if (content.note) el.append(h('span', { class: 'chart-readout-note' }, content.note));
}

/**
 * A rounded chip drawn inside the SVG (e.g. the 참고용 chip of a weak forecast), right-aligned at `right`.
 * Returns the group and its width.
 * @param {string} text
 * @param {number} right
 * @param {number} top
 * @param {number} textWidth  measured width of `text` at the chip's font
 * @returns {{node: SVGGElement, width: number}}
 */
export function svgChip(text, right, top, textWidth) {
  const width = Math.ceil(textWidth) + 16;
  const node = svg('g', { class: 'chart-ref-chip', transform: `translate(${right - width},${top})` },
    svg('rect', { class: 'chart-ref-chip-bg', x: 0, y: 0, width, height: 18, rx: 9 }),
    svg('text', { class: 'chart-ref-chip-text', x: width / 2, y: 9, dy: '0.34em', 'text-anchor': 'middle' }, text));
  return { node, width };
}
