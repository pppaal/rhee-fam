// Scales, nice ticks and y-range rules for the charts (spec F.7.1, F.7.2, D.1.6 axis ticks).
// Owner: WP3-chart. Pure module (F.1 rule 2): no DOM/window access; imported by tests/js/scale.test.mjs.
//
// Times are UTC milliseconds of a date at 00:00Z (isoToMs); never local time.
// Tick *values* are geometry and are cleaned of float noise with toPrecision; tick *labels* are display text and go
// through fmt.js (roundHalfAway) in tickLabels().

import { compact, compactTicks, roundHalfAway, MINUS } from '../fmt.js';

/**
 * @typedef {Object} Ticks
 * @property {number} min       domain minimum: rounded out to the tick grid (the first tick)
 * @property {number} max       domain maximum: the data maximum or the last tick, whichever is higher (niceTicks)
 * @property {number} step      tick step (1-2-5 × 10ⁿ, or 25 × 10ⁿ)
 * @property {number} decimals  decimals of the step itself (tick labels use exactly these; a relative index gets 0)
 * @property {number[]} values  tick values, ascending, min..max inclusive
 */

/**
 * @typedef {Object} LinearScale
 * @property {(v: number) => number} map     value → pixel
 * @property {(px: number) => number} invert pixel → value
 * @property {[number, number]} domain
 * @property {[number, number]} range
 */

/**
 * @typedef {Object} TimeTick
 * @property {number} ms      tick time (UTC ms)
 * @property {boolean} first  use the long label (first tick, every January for month ticks, year changes for day
 *                            ticks) — passed to fmt.axisLabel()
 */

const DAY_MS = 86400000;
const YEAR_MS = 365.25 * DAY_MS;
const INDEX_KINDS = new Set(['relative_index', 'relative_index_gt']);
/** Linear tick mantissas; 2.5 only from 25 up (25, 250 …), so tick labels stay integers (a relative index keeps
 * `0 20 40 …`-style integer ticks) while 0–125 or 0–250 ranges get a grid line above the data (UX-15). */
const MANTISSAS = [1, 2, 2.5, 5];
/** Log tick mantissas per decade, densest last: used when 1-2-5 leaves fewer than `minTicks` ticks (a range inside
 * one decade, e.g. 7,600–13,000 page views, would otherwise get only `5,000 · 1만`). */
const LOG_MANTISSAS = [[1, 2, 5], [1, 1.5, 2, 3, 5, 7], [1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 7, 8, 9]];

/** Remove binary float noise from a computed tick value (3 × 0.1 → 0.3) and normalise −0 to 0. */
const clean = (v) => {
  const c = Number(v.toPrecision(12));
  return c === 0 ? 0 : c;
};

const isNum = (v) => typeof v === 'number' && Number.isFinite(v);

/**
 * 'YYYY-MM-DD' → UTC milliseconds at 00:00Z.
 * @param {string} iso
 * @returns {number}
 */
export function isoToMs(iso) {
  const s = String(iso);
  return Date.UTC(Number(s.slice(0, 4)), Number(s.slice(5, 7)) - 1, Number(s.slice(8, 10)));
}

/**
 * UTC milliseconds → 'YYYY-MM-DD'.
 * @param {number} ms
 * @returns {string}
 */
export function msToIso(ms) {
  const t = new Date(ms);
  const p2 = (n) => String(n).padStart(2, '0');
  return `${String(t.getUTCFullYear()).padStart(4, '0')}-${p2(t.getUTCMonth() + 1)}-${p2(t.getUTCDate())}`;
}

/**
 * Number of y ticks for a plot width (F.7.1, F.10): `{min: 4, max: 6}` at >= 640 px, `{min: 3, max: 4}` below.
 * @param {number} widthPx
 * @returns {{min: number, max: number}}
 */
export function tickCount(widthPx) {
  return widthPx >= 640 ? { min: 4, max: 6 } : { min: 3, max: 4 };
}

/** Decimals of a tick step (0.5 → 1, 2 → 0, 0.02 → 2, 25 → 0). */
function stepDecimals(step) {
  for (let d = 0; d < 12; d += 1) {
    const scaled = step * 10 ** d;
    if (Math.abs(scaled - Math.round(scaled)) <= 1e-9 * Math.max(1, scaled)) return d;
  }
  return 12;
}

/** Log ticks: snap the top tick up to the next one when that leaves at most this share of the axis empty. */
const SNAP_UP = 0.06;
/** Weight of data above the top tick (unlabelled) against empty headroom above the data in niceTicks(). */
const UNLABELLED_WEIGHT = 1.5;

/**
 * Nice linear ticks with a 1-2-5 step (or 25 × 10ⁿ) whose count lies in [count.min, count.max]. `lo == hi` widens
 * the domain symmetrically (or to [0, 1] at 0).
 *
 * The bottom is rounded out to the tick grid (the baseline is always a labelled tick). For the top, every step
 * offers two grids: the last grid line at or below `hi` (the data above it is unlabelled) and the first one at or
 * above `hi` (the plot gets empty headroom). The grid that wastes the least share of the axis wins — unlabelled data
 * counts 1.5 × as much as headroom, since the peak cannot be read off the axis — and among equal grids the denser
 * one. So 0–103 keeps 0, 20 … 100 (3 % of the axis unlabelled) instead of stretching to 0–125, while 0–140 gets
 * 0, 50, 100, 150 instead of 0, 50, 100 with the top 29 % unlabelled (UX-15), and 0–225 gets 0, 50 … 250.
 * The domain is [first tick, max(hi, last tick)]. When no step gives a count inside the bounds, the densest grid
 * that does not exceed count.max is used (labels never crowd).
 * @param {number} lo
 * @param {number} hi
 * @param {{min: number, max: number}} count
 * @returns {Ticks}
 */
export function niceTicks(lo, hi, count) {
  let a = Number(lo);
  let b = Number(hi);
  if (!isNum(a) || !isNum(b)) { a = 0; b = 1; }
  if (a > b) [a, b] = [b, a];
  if (a === b) {
    if (a === 0) b = 1;
    else {
      const pad = Math.abs(a) * 0.1;
      a -= pad;
      b += pad;
    }
  }
  const maxN = Math.max(2, count.max);
  const minN = Math.max(2, Math.min(count.min, maxN));
  const span = b - a;
  const eps = 1e-9;
  const grids = [];
  const e0 = Math.floor(Math.log10(span / (maxN - 1))) - 1;
  for (let e = e0; e <= e0 + 3; e += 1) {
    for (const m of MANTISSAS) {
      if (m === 2.5 && e < 1) continue;
      const step = clean(m * 10 ** e);
      const i0 = Math.floor(a / step + eps);
      const below = Math.max(i0 + 1, Math.floor(b / step + eps));
      const above = Math.max(i0 + 1, Math.ceil(b / step - eps));
      for (const i1 of below === above ? [below] : [below, above]) {
        const top = i1 * step;
        const bottom = i0 * step;
        const axis = Math.max(b, top) - bottom;
        const waste = ((top < b ? UNLABELLED_WEIGHT * (b - top) : top - b) + (a - bottom)) / axis;
        grids.push({ step, i0, i1, n: i1 - i0 + 1, waste });
      }
    }
  }
  const better = (g, best) => !best || g.waste < best.waste - 1e-9
    || (Math.abs(g.waste - best.waste) <= 1e-9 && g.n > best.n);
  let chosen = null;
  for (const g of grids) if (g.n >= minN && g.n <= maxN && better(g, chosen)) chosen = g;
  if (!chosen) {
    for (const g of grids) {
      if (g.n <= maxN && (!chosen || g.n > chosen.n || (g.n === chosen.n && g.waste < chosen.waste))) chosen = g;
    }
  }
  const { step, i0, i1 } = chosen;
  const values = [];
  for (let i = i0; i <= i1; i += 1) values.push(clean(i * step));
  const top = values[values.length - 1];
  return { min: values[0], max: Math.max(b, top), step, decimals: stepDecimals(step), values };
}

/**
 * Log10 axis ticks with 1-2-5 per decade (F.7.2 `log` scale); lo must be > 0. When that leaves fewer than
 * `minTicks` ticks (a range inside one decade), denser mantissas are used (1-1.5-2-3-5-7, then 1-1.2-1.5-2-2.5-3-4
 * … 9), the first set that gives `minTicks` (UX-15: `0.7만 · 0.8만 · 0.9만 · 1만 · 1.2만` instead of `5,000 · 1만`).
 * When more than `maxTicks` values would result, only the decades (1 × 10ⁿ) are kept (then every k-th decade). The
 * bottom is rounded out to a tick and the top keeps `hi` unless the next tick is within 6 % of the axis (in log
 * units).
 * @param {number} lo
 * @param {number} hi
 * @param {number} [maxTicks=9]
 * @param {number} [minTicks=3]
 * @returns {{min: number, max: number, values: number[]}}
 */
export function logTicks(lo, hi, maxTicks = 9, minTicks = 3) {
  let a = Number(lo);
  let b = Number(hi);
  if (!(a > 0) || !isNum(a)) a = Math.min(1, b > 0 ? b : 1);
  if (!(b > 0) || !isNum(b)) b = a * 10;
  if (a > b) [a, b] = [b, a];
  if (a === b) { a /= 2; b *= 2; }
  const tol = 1e-9;
  const k0 = Math.floor(Math.log10(a)) - 1;
  const k1 = Math.ceil(Math.log10(b)) + 1;
  const build = (mantissas) => {
    const all = [];
    for (let k = k0; k <= k1; k += 1) for (const m of mantissas) all.push(clean(m * 10 ** k));
    const min = Math.max(...all.filter((v) => v <= a * (1 + tol)));
    let top = Math.max(...all.filter((v) => v <= b * (1 + tol)));
    const next = all.find((v) => v > top * (1 + tol));
    const axis = Math.log10(b) - Math.log10(min);
    if (next && (top <= min || Math.log10(next) - Math.log10(b) <= SNAP_UP * axis)) top = next;
    return { min, max: Math.max(b, top), values: all.filter((v) => v >= min && v <= top) };
  };
  let ticks = build(LOG_MANTISSAS[0]);
  for (const set of LOG_MANTISSAS.slice(1)) {
    if (ticks.values.length >= minTicks) break;
    const denser = build(set);
    if (denser.values.length > maxTicks) break;
    ticks = denser;
  }
  if (ticks.values.length > maxTicks) ticks = build([1]);
  if (ticks.values.length > maxTicks) {         // many decades: every k-th decade from the bottom
    const k = Math.ceil((ticks.values.length - 1) / (maxTicks - 1));
    ticks = { ...ticks, values: ticks.values.filter((_, i) => i % k === 0) };
  }
  return ticks;
}

/**
 * Linear scale from a value domain to a pixel range (range may be inverted for y: [bottom, top]).
 * @param {[number, number]} domain
 * @param {[number, number]} range
 * @returns {LinearScale}
 */
export function linearScale(domain, range) {
  const [d0, d1] = domain;
  const [r0, r1] = range;
  const span = d1 - d0;
  const map = span === 0 ? () => (r0 + r1) / 2 : (v) => r0 + ((v - d0) / span) * (r1 - r0);
  const invert = r1 === r0 ? () => d0 : (px) => d0 + ((px - r0) / (r1 - r0)) * span;
  return { map, invert, domain: [d0, d1], range: [r0, r1] };
}

/**
 * Log10 scale (domain values > 0).
 * @param {[number, number]} domain
 * @param {[number, number]} range
 * @returns {LinearScale}
 */
export function logScale(domain, range) {
  const lin = linearScale([Math.log10(domain[0]), Math.log10(domain[1])], range);
  return {
    map: (v) => (v > 0 ? lin.map(Math.log10(v)) : NaN),
    invert: (px) => 10 ** lin.invert(px),
    domain: [domain[0], domain[1]],
    range: [range[0], range[1]],
  };
}

/** UTC calendar parts of a time. */
function ymd(ms) {
  const t = new Date(ms);
  return { y: t.getUTCFullYear(), m: t.getUTCMonth(), d: t.getUTCDate() };
}

/** Time `n` calendar months after `ms` (same day of month, clamped by Date.UTC rollover rules). */
function addMonths(ms, n) {
  const { y, m, d } = ymd(ms);
  return Date.UTC(y, m + n, d);
}

/**
 * Time ticks per D.1.6: span >= 3 years → yearly ('year'); 6 months–3 years → every 1/2/3/6 months ('month', the
 * smallest interval whose tick count fits `maxTicks`); < 6 months → days/weeks ('day'). Year ticks thin out to
 * every 2/5/10… years and day ticks use 1, 2, 7, 14 or 28 days (7 and more aligned to Mondays, the weekly bucket
 * start) when needed to fit.
 * @param {number} startMs
 * @param {number} endMs
 * @param {number} maxTicks
 * @returns {{step: 'year'|'month'|'day', ticks: TimeTick[]}}
 */
export function timeTicks(startMs, endMs, maxTicks) {
  const lo = Math.min(startMs, endMs);
  const hi = Math.max(startMs, endMs);
  const limit = Math.max(1, Math.floor(maxTicks));
  const pick = (intervals, make) => {
    let ticks = [];
    for (const n of intervals) {
      ticks = make(n);
      if (ticks.length <= limit) return ticks;
    }
    return ticks.filter((_, i) => i % Math.ceil(ticks.length / limit) === 0);
  };

  if (hi >= addMonths(lo, 36)) {
    const y0 = ymd(lo).y;
    const y1 = ymd(hi).y;
    const ticks = pick([1, 2, 5, 10, 20, 50, 100], (n) => {
      const out = [];
      for (let y = y0; y <= y1; y += 1) {
        const ms = Date.UTC(y, 0, 1);
        if (ms >= lo && ms <= hi && y % n === 0) out.push(ms);
      }
      return out;
    });
    return { step: 'year', ticks: ticks.map((ms, i) => ({ ms, first: i === 0 })) };
  }

  if (hi >= addMonths(lo, 6)) {
    const first = ymd(lo);
    const ticks = pick([1, 2, 3, 6, 12], (n) => {
      const out = [];
      for (let k = 0; ; k += 1) {
        const ms = Date.UTC(first.y, first.m + k, 1);
        if (ms > hi) break;
        if (ms >= lo && new Date(ms).getUTCMonth() % n === 0) out.push(ms);
      }
      return out;
    });
    return {
      step: 'month',
      ticks: ticks.map((ms, i) => ({ ms, first: i === 0 || new Date(ms).getUTCMonth() === 0 })),
    };
  }

  const day0 = Math.ceil(lo / DAY_MS);
  const day1 = Math.floor(hi / DAY_MS);
  const MONDAY = 4;                          // 1970-01-05, epoch day 4, was a Monday
  let ticks = pick([1, 2, 7, 14, 28, 56], (n) => {
    const out = [];
    for (let d = day0; d <= day1; d += 1) {
      const aligned = n >= 7 ? (d - MONDAY) % n === 0 : (d - day0) % n === 0;
      if (aligned) out.push(d * DAY_MS);
    }
    return out;
  });
  if (!ticks.length) ticks = [day0 * DAY_MS];
  return {
    step: 'day',
    ticks: ticks.map((ms, i) => ({ ms, first: i === 0 || ymd(ms).y !== ymd(ticks[i - 1]).y })),
  };
}

/** Share of the time axis below which a narrow chart windows its history (narrowWindow). */
const NARROW_FC_SHARE = 0.2;

/**
 * Default x window of a narrow chart (< 640 px, UX-4). On a phone a 12-month forecast after 10 years of history,
 * or 26 weeks after 5 years, gets ~20 px of the plot and the fan cannot be read. When the forecast takes less than a
 * fifth of the time axis, the chart shows only the last part of the history: whole years for W/M/Q/Y data
 * (at least 1, about three forecast spans: 26 weeks → 2 years, 12 months → 3 years) and days for daily data (at
 * least 28, three forecast spans), so the forecast gets about a quarter of the plot. The scale stays uniform (no
 * stretched forecast region, which would flatten its slope); the charts offer a "full period" toggle and the table
 * and exports keep every row. Null when there is no forecast, the forecast already has room or the window would show
 * the whole history anyway.
 * @param {number} startMs  first time on the chart (UTC ms)
 * @param {number} lastMs   last actual time — the seam
 * @param {number} endMs    last forecast time
 * @param {'D'|'W'|'M'|'Q'|'Y'} freq
 * @returns {{startMs: number, n: number, unit: 'D'|'Y'}|null}  window start, and its length for the caption
 *          (fmt.spanLabel(n, unit))
 */
export function narrowWindow(startMs, lastMs, endMs, freq) {
  if (![startMs, lastMs, endMs].every(isNum)) return null;
  const fc = endMs - lastMs;
  const hist = lastMs - startMs;
  if (!(fc > 0) || !(hist > 0) || fc / (hist + fc) >= NARROW_FC_SHARE) return null;
  let out;
  if (freq === 'D') {
    const n = Math.max(28, Math.ceil((3 * fc) / DAY_MS - 1e-9));
    out = { startMs: lastMs - n * DAY_MS, n, unit: 'D' };
  } else {
    const n = Math.max(1, Math.ceil((3 * fc) / YEAR_MS - 0.1));
    const { y, m, d } = ymd(lastMs);
    out = { startMs: Date.UTC(y - n, m, d), n, unit: 'Y' };
  }
  return out.startMs > startMs ? out : null;
}

/**
 * Group line-end labels whose anchors coincide: same end time and pixel y within `tol`. Such lines end in one spot,
 * so one label `A · B` with one leader reads better than two nudged labels whose leaders converge (UX-16). Returns
 * groups of input indexes, each in input order, the groups ordered by their first member.
 * Added by WP3-chart (pure).
 * @param {Array<{ms: number, y: number}>} anchors
 * @param {number} tol
 * @returns {number[][]}
 */
export function groupAnchors(anchors, tol) {
  const order = anchors.map((a, i) => i).sort((i, j) => anchors[i].ms - anchors[j].ms
    || anchors[i].y - anchors[j].y || i - j);
  const groups = [];
  let cur = null;
  for (const i of order) {
    const prev = cur && anchors[cur[cur.length - 1]];
    if (prev && prev.ms === anchors[i].ms && Math.abs(anchors[i].y - prev.y) <= tol) cur.push(i);
    else {
      cur = [i];
      groups.push(cur);
    }
  }
  return groups.map((g) => g.slice().sort((i, j) => i - j)).sort((g, h) => g[0] - h[0]);
}

/**
 * Y domain of the detail fan chart (F.7.1 "Y range", D.2.6 `y_hint`).
 * Starts from `yHint` (min/max incl. the 95 % band; 0 included when `include_zero`); anomalies whose inclusion
 * would more than double the range of the other values are excluded (drawn as edge triangles); the last actual
 * value always stays in range. Never a broken axis.
 *
 * `rest` (optional, added by WP3-chart) is the min/max of every plotted value *except* the anomalies (history
 * without the anomaly points, forecast mean and bands). The exclusion rule needs it, because `y_hint` already
 * contains the anomalies; without it no anomaly is excluded and the domain is `y_hint`.
 * @param {{min: number, max: number, include_zero: boolean}} yHint  chart.y_hint
 * @param {Array<{date: string, value: number}>} anomalies  chart.anomalies
 * @param {{date: string, value: number}|null} lastActual  chart.last_actual
 * @param {{min: number, max: number}} count  tickCount(width)
 * @param {{min: number, max: number}|null} [rest]  range of the non-anomaly values
 * @returns {{ticks: Ticks, excluded: number[]}}  nice ticks and the indexes of excluded anomalies
 */
export function detailYDomain(yHint, anomalies, lastActual, count, rest = null) {
  const hint = yHint || {};
  const hasRest = rest && isNum(rest.min) && isNum(rest.max);
  let lo = hasRest ? rest.min : hint.min;
  let hi = hasRest ? rest.max : hint.max;
  if (!isNum(lo) || !isNum(hi)) {
    lo = isNum(hint.min) ? hint.min : 0;
    hi = isNum(hint.max) ? hint.max : 1;
  }
  const last = lastActual && isNum(lastActual.value) ? lastActual.value : null;
  if (last !== null) {
    lo = Math.min(lo, last);
    hi = Math.max(hi, last);
  }
  if (hint.include_zero) {
    lo = Math.min(lo, 0);
    hi = Math.max(hi, 0);
  }
  const width = hi - lo;
  const excluded = [];
  let outLo = lo;
  let outHi = hi;
  (anomalies || []).forEach((a, i) => {
    const v = a && a.value;
    if (!isNum(v) || (v >= lo && v <= hi)) return;
    const grown = Math.max(hi, v) - Math.min(lo, v);
    if (hasRest && width > 0 && grown > 2 * width) {
      excluded.push(i);
    } else {
      outLo = Math.min(outLo, v);
      outHi = Math.max(outHi, v);
    }
  });
  if (!excluded.length) {
    if (isNum(hint.min)) outLo = Math.min(outLo, hint.min);
    if (isNum(hint.max)) outHi = Math.max(outHi, hint.max);
  }
  return { ticks: niceTicks(outLo, outHi, count), excluded };
}

/**
 * Y domain of the comparison chart (F.7.2): relative indexes (`relative_index`, `relative_index_gt`) under the
 * `raw` scale use a fixed 0–100 frame, extended above 100 only when a visible value or band exceeds it; other
 * kinds take the visible data (and visible bands), with 0 included for non-negative data whose minimum is at most
 * half the maximum (brief §4.3 rule 9); `index` scale works on value / base × 100; `log` returns log ticks.
 * @param {Object} args
 * @param {string} args.valueKind      compare.value_kind
 * @param {'raw'|'index'|'log'} args.scale
 * @param {number} args.dataMin        min of the visible (transformed) values
 * @param {number} args.dataMax        max of the visible (transformed) values
 * @param {number|null} args.bandMax   max of the visible focused band, or null
 * @param {number|null} [args.bandMin] min of the visible focused band, or null (optional)
 * @param {{min: number, max: number}} args.count
 * @returns {{ticks: Ticks|{min: number, max: number, values: number[]}, log: boolean}}
 */
export function compareYDomain(args) {
  const { valueKind, scale, count } = args;
  let lo = isNum(args.dataMin) ? args.dataMin : null;
  let hi = isNum(args.dataMax) ? args.dataMax : null;
  if (isNum(args.bandMax)) hi = hi === null ? args.bandMax : Math.max(hi, args.bandMax);
  if (isNum(args.bandMin)) lo = lo === null ? args.bandMin : Math.min(lo, args.bandMin);

  if (scale === 'log') {
    const ticks = logTicks(lo !== null && lo > 0 ? lo : 1, hi !== null && hi > 0 ? hi : 10,
      Math.max(5, count.max + 3), count.min);
    return { ticks, log: true };
  }
  if (scale !== 'index' && INDEX_KINDS.has(valueKind)) {
    const top = hi === null ? 100 : Math.max(100, hi);
    const bottom = lo === null ? 0 : Math.min(0, lo);
    if (top <= 100 && bottom === 0) return { ticks: niceTicks(0, 100, count), log: false };
    return { ticks: niceTicks(bottom, top, count), log: false };
  }
  if (lo === null || hi === null) return { ticks: niceTicks(0, scale === 'index' ? 100 : 1, count), log: false };
  if (lo >= 0 && lo <= 0.5 * hi) lo = 0;
  return { ticks: niceTicks(lo, hi, count), log: false };
}

/**
 * Index transform for the `index` scale (F.7.2): value / base × 100 (null stays null; an unusable base gives
 * nulls).
 * @param {Array<number|null>} values
 * @param {number} base  compare.scale.index_base.values[k]
 * @returns {Array<number|null>}
 */
export function indexValues(values, base) {
  const ok = isNum(base) && base > 0;
  return (values || []).map((v) => (ok && isNum(v) ? (v / base) * 100 : null));
}

/**
 * Y tick labels (F.7.1): the step's own decimals (a relative index gets integers `0 20 40 …`); an axis whose ticks
 * reach 10,000 writes every tick in one compact unit (fmt.compactTicks: `0 · 0.5만 · 1만 · 1.5만`, en
 * `0 · 5K · 10K · 15K`, UX-15), else — log ticks over several decades — fmt.compact per tick
 * (`100 · 1,000 · 1만 · 10만`), falling back to full digits when compact labels would repeat. Log ticks use each
 * tick's own decimals. Never compacted for relative indexes. Added by WP3-chart (pure; shared by the fan and
 * comparison charts).
 * @param {number[]} values
 * @param {number} decimals  Ticks.decimals (ignored for log ticks)
 * @param {'ko'|'en'} lang
 * @param {{kind?: string, log?: boolean}} [opts]
 * @returns {string[]}
 */
export function tickLabels(values, decimals, lang, opts = {}) {
  const fixed = (v, d) => {
    const s = roundHalfAway(Math.abs(v), d);
    const dot = s.indexOf('.');
    const int = dot < 0 ? s : s.slice(0, dot);
    const body = int.replace(/\B(?=(\d{3})+(?!\d))/g, ',') + (dot < 0 ? '' : s.slice(dot));
    return v < 0 && /[1-9]/.test(s) ? MINUS + body : body;
  };
  const own = (v) => (v === 0 ? 0 : Math.max(0, -Math.floor(Math.log10(Math.abs(v)) + 1e-9)));
  const plain = values.map((v) => (opts.log ? fixed(v, own(v)) : fixed(v, decimals)));
  const reaches = values.some((v) => Math.abs(v) >= 9999.5);
  if (INDEX_KINDS.has(opts.kind) || !reaches) return plain;
  const unit = compactTicks(values, lang);
  if (unit) return unit;
  // compact() prints full digits below 10,000 with its own decimal rule; keep the tick decimals there instead
  const short = values.map((v, i) => (Math.abs(v) >= 9999.5 ? compact(v, lang) : plain[i]));
  return new Set(short).size === short.length ? short : plain;
}

/**
 * Spread label positions so neighbours are at least `gap` apart, moving each overlapping group as little as
 * possible (groups are centred on the mean of their wanted positions) and keeping everything inside [lo, hi].
 * Used for the line-end labels (F.7.2: ≥ 14 px apart) and changepoint badges. Returns positions in input order.
 * Added by WP3-chart (pure).
 * @param {number[]} wanted
 * @param {number} gap
 * @param {number} lo
 * @param {number} hi
 * @returns {number[]}
 */
export function dodge(wanted, gap, lo, hi) {
  const order = wanted.map((v, i) => ({ i, v })).sort((a, b) => a.v - b.v || a.i - b.i);
  const place = (c) => {
    const span = (c.items.length - 1) * gap;
    const mean = c.items.reduce((s, it) => s + it.v, 0) / c.items.length;
    let top = mean - span / 2;
    top = Math.min(top, hi - span);
    top = Math.max(top, lo);
    c.top = top;
    c.bottom = top + span;
  };
  const clusters = [];
  for (const item of order) {
    const c = { items: [item] };
    place(c);
    clusters.push(c);
    while (clusters.length > 1) {
      const last = clusters[clusters.length - 1];
      const prev = clusters[clusters.length - 2];
      if (prev.bottom + gap <= last.top + 1e-9) break;
      prev.items.push(...last.items);
      clusters.pop();
      place(prev);
    }
  }
  const out = new Array(wanted.length);
  for (const c of clusters) c.items.forEach((it, k) => { out[it.i] = c.top + k * gap; });
  return out;
}
