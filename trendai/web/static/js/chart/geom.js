// Pure geometry and text helpers shared by the chart modules (spec F.7). Owner: WP3-chart.
// Pure module: no DOM/window access at import or call time; unit-tested by tests/js/chart-geom.test.mjs.
// Coordinates here are pixels (geometry, not display text), so they are trimmed to 0.1 px for compact paths.

const px = (v) => {
  const r = Math.round(v * 10) / 10;           // geometry precision, not display rounding
  return Object.is(r, -0) ? 0 : r;
};

const isNum = (v) => typeof v === 'number' && Number.isFinite(v);

/**
 * SVG path of a polyline through `points` ([x, y] pairs); a null point or a non-finite coordinate breaks the
 * line (a gap). A segment of one point is drawn as a zero-length line so round caps still show a dot.
 * @param {Array<[number, number]|null>} points
 * @returns {string}  '' when nothing is drawable
 */
export function linePath(points) {
  let d = '';
  let open = false;
  let runStart = '';
  let runLen = 0;
  const close = () => {
    if (open && runLen === 1) d += `${runStart.replace('M', 'L')}`;
  };
  for (const p of points) {
    if (!p || !isNum(p[0]) || !isNum(p[1])) {
      close();
      open = false;
      runLen = 0;
      continue;
    }
    const seg = `${px(p[0])},${px(p[1])}`;
    if (!open) {
      runStart = `M${seg}`;
      d += runStart;
      open = true;
      runLen = 1;
    } else {
      d += `L${seg}`;
      runLen += 1;
    }
  }
  close();
  return d;
}

/**
 * SVG path of a band (area between `lo` and `hi`) over `xs`; points where either bound is missing split the
 * band into separate closed shapes.
 * @param {number[]} xs
 * @param {Array<number|null>} lo  pixel y of the lower bound
 * @param {Array<number|null>} hi  pixel y of the upper bound
 * @returns {string}
 */
export function areaPath(xs, lo, hi) {
  let d = '';
  let run = [];
  const flush = () => {
    if (run.length >= 2) {
      const top = run.map((i, k) => `${k ? 'L' : 'M'}${px(xs[i])},${px(hi[i])}`).join('');
      const bottom = run.slice().reverse().map((i) => `L${px(xs[i])},${px(lo[i])}`).join('');
      d += `${top}${bottom}Z`;
    }
    run = [];
  };
  for (let i = 0; i < xs.length; i += 1) {
    if (isNum(xs[i]) && isNum(lo[i]) && isNum(hi[i])) run.push(i);
    else flush();
  }
  flush();
  return d;
}

/**
 * The forecast with the seam prepended (F.7.1 "Seam"): the mean and every band start at the last actual point
 * with zero width, so the forecast continues the history without a gap or a jump.
 * @param {{date: string, value: number}|null} lastActual  chart.last_actual
 * @param {{dates: string[], mean?: Array<number|null>, lo80?: Array<number|null>, hi80?: Array<number|null>,
 *          lo95?: Array<number|null>, hi95?: Array<number|null>}} forecast  chart.forecast
 * @returns {{dates: string[], mean: Array<number|null>, lo80: Array<number|null>, hi80: Array<number|null>,
 *            lo95: Array<number|null>, hi95: Array<number|null>, seam: boolean}}
 */
export function seamForecast(lastActual, forecast) {
  const fc = forecast || {};
  const dates = (fc.dates || []).slice();
  const col = (key) => (Array.isArray(fc[key]) ? fc[key].slice(0, dates.length) : dates.map(() => null));
  const out = { dates, mean: col('mean'), lo80: col('lo80'), hi80: col('hi80'), lo95: col('lo95'), hi95: col('hi95'),
    seam: false };
  if (!lastActual || !isNum(lastActual.value) || !lastActual.date || !dates.length) return out;
  if (lastActual.date >= dates[0]) return out;
  const v = lastActual.value;
  out.dates.unshift(lastActual.date);
  for (const key of ['mean', 'lo80', 'hi80', 'lo95', 'hi95']) out[key].unshift(v);
  out.seam = true;
  return out;
}

/**
 * Path of a slot icon (D.2.1: 1 circle ●, 2 diamond ◆, 3 hexagon ⬢, 4 square ■, 5 cross ✚) centred on
 * (cx, cy) with "radius" r; shapes are scaled to look equally heavy. Unknown icons give a circle.
 * @param {string|null} icon
 * @param {number} cx
 * @param {number} cy
 * @param {number} r
 * @returns {string}
 */
export function slotIconPath(icon, cx, cy, r) {
  const P = (x, y) => `${px(cx + x)},${px(cy + y)}`;
  switch (icon) {
    case 'diamond': {
      const k = r * 1.3;
      return `M${P(0, -k)}L${P(k, 0)}L${P(0, k)}L${P(-k, 0)}Z`;
    }
    case 'hexagon': {
      const k = r * 1.12;
      const pts = [0, 1, 2, 3, 4, 5].map((i) => {
        const a = (Math.PI / 3) * i - Math.PI / 2;
        return P(k * Math.cos(a), k * Math.sin(a));
      });
      return `M${pts.join('L')}Z`;
    }
    case 'square': {
      const k = r * 0.9;
      return `M${P(-k, -k)}L${P(k, -k)}L${P(k, k)}L${P(-k, k)}Z`;
    }
    case 'cross': {
      const k = r * 1.15;
      const w = k * 0.38;
      return `M${P(-w, -k)}L${P(w, -k)}L${P(w, -w)}L${P(k, -w)}L${P(k, w)}L${P(w, w)}L${P(w, k)}L${P(-w, k)}`
        + `L${P(-w, w)}L${P(-k, w)}L${P(-k, -w)}L${P(-w, -w)}Z`;
    }
    default:
      return `M${P(-r, 0)}A${px(r)},${px(r)} 0 1,0 ${P(r, 0)}A${px(r)},${px(r)} 0 1,0 ${P(-r, 0)}Z`;
  }
}

/**
 * Rough text width in px for layout before the text can be measured (or where it cannot): Hangul and other wide
 * characters ≈ 0.92 em, digits ≈ 0.6 em (tabular), narrow punctuation ≈ 0.3 em, others ≈ 0.56 em.
 * @param {string} text
 * @param {number} fontPx
 * @returns {number}
 */
export function estimateTextWidth(text, fontPx) {
  let em = 0;
  for (const ch of String(text)) {
    const c = ch.codePointAt(0);
    if (c >= 0x1100 && (c <= 0x115f || (c >= 0x2e80 && c <= 0xa4cf) || (c >= 0xac00 && c <= 0xd7a3)
      || (c >= 0xf900 && c <= 0xfaff) || (c >= 0xff00 && c <= 0xff60))) em += 0.92;
    else if (c >= 0x2460 && c <= 0x24ff) em += 0.9;
    else if (ch >= '0' && ch <= '9') em += 0.6;
    else if (' .,:;·|!\'()[]'.includes(ch)) em += 0.3;
    else em += 0.56;
  }
  return em * fontPx;
}

/**
 * Cut `text` with an ellipsis so that `measure(text) <= maxPx` (binary search over code points). Returns the text
 * unchanged when it fits.
 * @param {string} text
 * @param {number} maxPx
 * @param {(s: string) => number} measure
 * @returns {string}
 */
export function truncateToWidth(text, maxPx, measure) {
  const s = String(text);
  if (measure(s) <= maxPx) return s;
  const chars = Array.from(s);
  let lo = 0;
  let hi = chars.length;
  while (lo < hi) {
    const mid = Math.ceil((lo + hi) / 2);
    if (measure(`${chars.slice(0, mid).join('').trimEnd()}…`) <= maxPx) lo = mid;
    else hi = mid - 1;
  }
  return lo ? `${chars.slice(0, lo).join('').trimEnd()}…` : '…';
}

/**
 * A value text with the VM's unit label (F.7.4 "+ unit_label"): Korean units and % attach directly (`1,234원`),
 * other units after a space (`1,234 kg`). No unit → the text unchanged.
 * @param {string} text
 * @param {string|null|undefined} unit
 * @returns {string}
 */
export function withUnit(text, unit) {
  const u = unit == null ? '' : String(unit).trim();
  if (!u || text === '—') return text;
  return /^[ㄱ-ㆎ가-힣%‰°]/.test(u) ? `${text}${u}` : `${text} ${u}`;
}

/**
 * Circled number for changepoint badges and texts: 1 → '①' … 20 → '⑳'; larger numbers as '(21)'.
 * @param {number} n
 * @returns {string}
 */
export function circled(n) {
  return n >= 1 && n <= 20 ? String.fromCodePoint(0x245f + n) : `(${n})`;
}

/**
 * Min and max over several arrays of numbers, skipping non-finite values. Null when there is no finite value.
 * @param {...Array<number|null|undefined>} arrays
 * @returns {{min: number, max: number}|null}
 */
export function extent(...arrays) {
  let min = Infinity;
  let max = -Infinity;
  for (const arr of arrays) {
    for (const v of arr || []) {
      if (isNum(v)) {
        if (v < min) min = v;
        if (v > max) max = v;
      }
    }
  }
  return min <= max ? { min, max } : null;
}
