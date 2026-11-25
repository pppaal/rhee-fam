// Sparkline (spec F.7.3): 120×32 px, no axes, own y scale per card, aria-hidden.
// Owner: WP3-chart. Called by components/rankcards.js and pages/samples.js (WP3-app).

import { svg } from '../dom.js';
import { linePath } from './geom.js';

const PAD = 3;                                 // keeps the r 2.5 end dot and 1.5 px strokes inside the box
const isNum = (v) => typeof v === 'number' && Number.isFinite(v);

/**
 * @typedef {Object} SparklineOptions
 * @property {number|null} [slot=null]  1..5 → slot colour class; null → --c-hist (outside comparisons) — a grey
 *                                      (unslotted) comparison series passes `grey: true`
 * @property {boolean} [grey=false]     unslotted comparison series (--ink-3)
 * @property {'normal'|'weak'|'bands_only'} [display='normal']  weak → forecast tail at opacity .6; bands_only → no
 *                                      tail (spark.forecast is empty then anyway, D.4.1)
 * @property {number} [width=120]
 * @property {number} [height=32]
 */

/**
 * Build a sparkline SVG: history polyline (1.5 px), dashed forecast tail continuing from the last history point,
 * last-history-point dot (r 2.5). Null values break the line. `aria-hidden="true"`, `focusable="false"`.
 * @param {{history: Array<number|null>, forecast?: Array<number|null>} | Array<number|null>} spark
 *        a ranking row's `spark` (D.4.1) or a plain array (the samples' `spark`, C.3 — history only)
 * @param {SparklineOptions} [opts]
 * @returns {SVGSVGElement}
 */
export function sparkline(spark, opts = {}) {
  const width = opts.width || 120;
  const height = opts.height || 32;
  const display = opts.display || 'normal';
  const history = (Array.isArray(spark) ? spark : (spark && spark.history) || []).map((v) => (isNum(v) ? v : null));
  const forecast = display === 'bands_only' || Array.isArray(spark) ? []
    : ((spark && spark.forecast) || []).map((v) => (isNum(v) ? v : null));

  const slot = Number.isInteger(opts.slot) && opts.slot >= 1 && opts.slot <= 5 ? opts.slot : null;
  const colour = opts.grey ? 'spark-grey' : (slot ? `spark-slot-${slot}` : null);
  const root = svg('svg', {
    class: ['spark', colour, display === 'weak' && 'spark-weak'],
    width, height, viewBox: `0 0 ${width} ${height}`, 'aria-hidden': 'true', focusable: 'false',
  });

  const all = [...history, ...forecast].filter(isNum);
  if (!all.length) return root;
  let lo = Math.min(...all);
  let hi = Math.max(...all);
  if (lo === hi) {                              // a flat series sits in the middle
    lo -= 1;
    hi += 1;
  }
  const n = history.length + forecast.length;
  const xAt = (i) => (n > 1 ? PAD + (i / (n - 1)) * (width - 2 * PAD) : width / 2);
  const yAt = (v) => height - PAD - ((v - lo) / (hi - lo)) * (height - 2 * PAD);

  root.appendChild(svg('path', {
    class: 'spark-line', d: linePath(history.map((v, i) => (v === null ? null : [xAt(i), yAt(v)]))),
  }));

  let lastIdx = -1;
  for (let i = history.length - 1; i >= 0; i -= 1) {
    if (history[i] !== null) {
      lastIdx = i;
      break;
    }
  }
  if (forecast.length) {
    // the tail starts at the last history point (no gap at the seam)
    const tail = [lastIdx >= 0 ? [xAt(lastIdx), yAt(history[lastIdx])] : null,
      ...forecast.map((v, j) => (v === null ? null : [xAt(history.length + j), yAt(v)]))];
    root.appendChild(svg('path', { class: 'spark-line spark-fc', d: linePath(tail) }));
  }
  if (lastIdx >= 0) {
    root.appendChild(svg('circle', { class: 'spark-dot', cx: xAt(lastIdx), cy: yAt(history[lastIdx]), r: 2.5 }));
  }
  return root;
}
