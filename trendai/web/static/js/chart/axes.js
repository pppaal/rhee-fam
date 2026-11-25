// Axes and grid for the fan and comparison charts (spec F.7.1 "X axis", "Y range"; D.1.6 axis ticks).
// Owner: WP3-chart. Internal to the chart modules: pages never import it.
//
// All nodes are built with dom.svg(); colours come from chart.css classes (.chart-grid-line, .chart-axis-line,
// .chart-tick, .chart-axis-label). Coordinates are absolute SVG pixels: scales map into
// [margin.left, width − margin.right] (x) and [height − margin.bottom, margin.top] (y).

import { svg } from '../dom.js';
import { axisLabel } from '../fmt.js';
import { estimateTextWidth } from './geom.js';

/** Font size of tick labels (chart.css `.chart-tick`); used for estimates before text can be measured. */
export const TICK_FONT_PX = 12;
/** Height reserved under the plot for x tick labels, and for the caption line below them (F.7.1). */
export const X_TICK_SPACE = 28;
export const X_CAPTION_SPACE = 18;

/**
 * @typedef {Object} Frame
 * @property {number} width   full SVG width (px)
 * @property {number} height  full SVG height (px)
 * @property {{top: number, right: number, bottom: number, left: number}} margin
 */

/**
 * Width in px of the widest label, measured for the left margin (F.7.1: left = widest label + 8).
 * Uses a hidden measuring <text> in `svgEl` (or a character-width estimate when not attached).
 * @param {SVGSVGElement} svgEl
 * @param {string[]} labels
 * @param {string} [cls='chart-tick']  class of the text being measured (font size comes from chart.css)
 * @returns {number}
 */
export function measureLabels(svgEl, labels, cls = 'chart-tick') {
  return Math.max(0, ...textWidths(svgEl, labels, cls));
}

/**
 * Widths in px of several labels rendered with class `cls` inside `svgEl` (estimates when `svgEl` is not laid
 * out, e.g. detached or hidden).
 * @param {SVGSVGElement} svgEl
 * @param {string[]} labels
 * @param {string} [cls='chart-tick']
 * @param {number} [fontPx=TICK_FONT_PX]  font size for the estimate
 * @returns {number[]}
 */
export function textWidths(svgEl, labels, cls = 'chart-tick', fontPx = TICK_FONT_PX) {
  if (!labels.length) return [];
  const probe = svgEl && svgEl.isConnected
    ? svg('text', { class: [cls, 'chart-measure'], x: 0, y: -100, 'aria-hidden': 'true' }) : null;
  if (probe) svgEl.appendChild(probe);
  try {
    return labels.map((label) => {
      if (probe) {
        probe.textContent = label;
        const w = probe.getComputedTextLength();
        if (w > 0) return Math.ceil(w);
      }
      return Math.ceil(estimateTextWidth(label, fontPx));
    });
  } finally {
    if (probe) probe.remove();
  }
}

/**
 * Horizontal grid lines and y tick labels into `layer` (an SVG <g>). A zero line inside the domain is drawn in
 * the axis colour so the baseline reads as one.
 * @param {SVGGElement} layer
 * @param {Frame} frame
 * @param {import("./scale.js").LinearScale} y
 * @param {number[]} values  tick values
 * @param {(v: number, i: number) => string} label  tick label formatter (step decimals or log ticks)
 * @returns {void}
 */
export function drawYAxis(layer, frame, y, values, label) {
  const { width, height, margin } = frame;
  const x1 = margin.left;
  const x2 = width - margin.right;
  const bottom = height - margin.bottom;
  values.forEach((v, i) => {
    const py = Math.round(y.map(v)) + 0.5;        // crisp 1 px lines
    if (!Number.isFinite(py)) return;
    const isBase = Math.abs(py - bottom - 0.5) < 1;
    if (!isBase) {
      layer.appendChild(svg('line', {
        class: ['chart-grid-line', v === 0 && 'chart-zero-line'], x1, x2, y1: py, y2: py,
      }));
    }
    layer.appendChild(svg('text', {
      class: 'chart-tick chart-tick-y', x: margin.left - 8, y: py, dy: '0.32em', 'text-anchor': 'end',
    }, label(v, i)));
  });
}

/**
 * X axis line, time ticks (labels via fmt.axisLabel) and the axis caption (`chart.x_label`) into `layer`.
 * Labels that would overlap are dropped (the long, year-carrying form moves to the next kept label), and the
 * first/last labels are kept inside the SVG.
 * @param {SVGGElement} layer
 * @param {Frame} frame
 * @param {import("./scale.js").LinearScale} x  time scale (UTC ms → px)
 * @param {{step: 'year'|'month'|'day', ticks: import("./scale.js").TimeTick[]}} ticks  scale.timeTicks() output
 * @param {{lang: 'ko'|'en', caption?: string|null, svgEl?: SVGSVGElement}} opts
 * @returns {void}
 */
export function drawXAxis(layer, frame, x, ticks, opts) {
  const { width, height, margin } = frame;
  const bottom = Math.round(height - margin.bottom) + 0.5;
  layer.appendChild(svg('line', {
    class: 'chart-axis-line', x1: margin.left, x2: width - margin.right, y1: bottom, y2: bottom,
  }));
  const svgEl = opts.svgEl || layer.ownerSVGElement;

  // Decide labels left to right; a dropped long label hands its long form to the next label that is kept.
  const items = ticks.ticks.map((t) => ({ ms: t.ms, first: t.first, x: x.map(t.ms) }));
  const keptLabels = [];
  let carry = false;
  let lastRight = -Infinity;
  const short = textWidths(svgEl, items.map((t) => axisLabel(t.ms, ticks.step, false, opts.lang)));
  const long = textWidths(svgEl, items.map((t) => axisLabel(t.ms, ticks.step, true, opts.lang)));
  items.forEach((t, i) => {
    const first = t.first || carry;
    const w = first ? long[i] : short[i];
    const left = Math.max(0, Math.min(t.x - w / 2, width - 1 - w));   // keep the label inside the SVG
    if (left < lastRight + 12) {
      carry = first;
      return;
    }
    carry = false;
    lastRight = left + w;
    keptLabels.push({ t, text: axisLabel(t.ms, ticks.step, first, opts.lang), left, w });
  });

  for (const { t, text, left, w } of keptLabels) {
    const tx = Math.round(t.x) + 0.5;
    layer.appendChild(svg('line', { class: 'chart-tick-mark', x1: tx, x2: tx, y1: bottom, y2: bottom + 4 }));
    layer.appendChild(svg('text', {
      class: 'chart-tick chart-tick-x', x: left + w / 2, y: bottom + 17, 'text-anchor': 'middle',
    }, text));
  }
  if (opts.caption) {
    layer.appendChild(svg('text', {
      class: 'chart-axis-label', x: margin.left, y: height - 5, 'text-anchor': 'start',
    }, opts.caption));
  }
}
