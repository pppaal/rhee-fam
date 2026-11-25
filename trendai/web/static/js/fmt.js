// Display formatting (spec D.1, F.5): the JS mirror of trendai/web/fmt.py, function for function.
// Owner: WP3-chart. Pure module (F.1 rule 2): no DOM, window or storage access; imported by tests/js/fmt.test.mjs.
//
// Rules for every function:
//   * All display rounding goes through roundHalfAway() on the exact decimal expansion of the double, ties away
//     from zero. Never Math.round, toFixed(d), toPrecision or Intl rounding for display digits.
//   * The minus sign is U+2212 (MINUS); a missing value (null, undefined, NaN, ±Infinity) prints DASH ('—').
//   * `lang` is 'ko' | 'en'; anything else is treated as 'ko'.
//   * Every row of the D.1.7 test vectors holds (tests/js/fmt.test.mjs), identical to fmt.py.
//
// Value kinds: 'relative_index' | 'relative_index_gt' | 'pageviews' | 'raw'.
// Frequencies: 'D' | 'W' | 'M' | 'Q' | 'Y'. Dates are 'YYYY-MM-DD' strings; timestamps UTC ISO with 'Z'.
//
// Exact decimals: a number is held as {c, q} = c × 10^q with a non-negative BigInt coefficient `c`, the same model
// as Python's Decimal. The coefficient of a double is derived from its IEEE-754 bits (mantissa × 5^k / 10^k), which
// is the exact expansion `Math.abs(x).toFixed(100)` gives wherever that is exact, and stays exact beyond it.

/** U+2212 MINUS SIGN, used for every negative number. */
export const MINUS = '−';
/** EM DASH printed for missing values. */
export const DASH = '—';
/** English month abbreviations (D.1.6). */
export const MONTHS_EN = Object.freeze(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']);

const INDEX_KINDS = new Set(['relative_index', 'relative_index_gt']);
const SUPERSCRIPT = { 0: '⁰', 1: '¹', 2: '²', 3: '³', 4: '⁴', 5: '⁵', 6: '⁶', 7: '⁷', 8: '⁸', 9: '⁹', '-': '⁻' };
const UNITS = { ko: [[4, '만'], [8, '억'], [12, '조']], en: [[3, 'K'], [6, 'M'], [9, 'B']] };

const FREQ_LABELS = {
  D: ['일 단위', 'daily'], W: ['주 단위', 'weekly'], M: ['월 단위', 'monthly'],
  Q: ['분기 단위', 'quarterly'], Y: ['연 단위', 'yearly'],
};
const FREQ_ADJ_KO = { D: '일간', W: '주간', M: '월간', Q: '분기', Y: '연간' };
const SPAN_KO = { D: '일', W: '주', M: '개월', Q: '분기', Y: '년' };
const SPAN_EN = {
  D: ['day', 'days'], W: ['week', 'weeks'], M: ['month', 'months'], Q: ['quarter', 'quarters'], Y: ['year', 'years'],
};

// Likelihood bins on the raw P(rise) (D.1.5; the narrative's _probability_line thresholds).
const LIKELIHOOD = [
  [(p) => p >= 0.90, '상승 가능성 매우 높음', 'Rise very likely'],
  [(p) => p >= 0.70, '상승 쪽이 우세', 'Rise more likely'],
  [(p) => p >= 0.55, '상승 쪽이 약간 우세', 'Rise slightly more likely'],
  [(p) => p > 0.45, '상승·하락 엇비슷', 'About even'],
  [(p) => p > 0.30, '하락 쪽이 약간 우세', 'Fall slightly more likely'],
  [(p) => p > 0.10, '하락 쪽이 우세', 'Fall more likely'],
  [() => true, '하락 가능성 매우 높음', 'Fall very likely'],
];

const isEn = (lang) => lang === 'en';

/** `x` as a finite number (booleans excluded), else null. */
function finite(x) {
  if (x == null || typeof x === 'boolean') return null;
  const v = typeof x === 'number' ? x : Number(x);
  return Number.isFinite(v) ? v : null;
}

// ------------------------------------------------------------------------------------------------ exact decimals

const TEN = 10n;
const pow10 = (n) => TEN ** BigInt(n);

/** Parse a decimal literal ('0.1', '1e-4', '10000') into {c, q}. Internal constants only. */
function decimal(text) {
  const [, intPart, frac = '', exp = '0'] = /^(\d+)(?:\.(\d+))?(?:e([+-]?\d+))?$/i.exec(text);
  return { c: BigInt(intPart + frac), q: Number(exp) - frac.length };
}

const D_ONE = decimal('1');
const D_TEN = decimal('10');
const D_TENTH = decimal('0.1');
const D_THOUSAND = decimal('1000');
const D_10K = decimal('10000');
const SCI_SMALL = decimal('1e-4');
const SCI_LARGE = decimal('1e12');
const COMPACT_SCI = { ko: decimal('1e16'), en: decimal('1e12') };
const UNIT_LIMIT = { ko: D_10K, en: D_THOUSAND };

const F64 = new DataView(new ArrayBuffer(8));

/** Exact decimal value of |x| for a finite double (Decimal(x) without the sign). */
function exactAbs(x) {
  F64.setFloat64(0, x);
  const hi = F64.getUint32(0);
  const lo = F64.getUint32(4);
  const biased = (hi >>> 20) & 0x7ff;
  let mant = (BigInt(hi & 0xfffff) << 32n) | BigInt(lo);
  let e2;
  if (biased === 0) {
    e2 = -1074;                       // subnormal
  } else {
    mant |= 1n << 52n;
    e2 = biased - 1075;
  }
  if (mant === 0n) return { c: 0n, q: 0 };
  if (e2 >= 0) return { c: mant << BigInt(e2), q: 0 };
  // mant / 2^k == mant × 5^k / 10^k
  return { c: mant * 5n ** BigInt(-e2), q: e2 };
}

/** Compare two exact decimals: -1, 0 or 1. */
function cmp(a, b) {
  let ac = a.c;
  let bc = b.c;
  if (a.q > b.q) ac *= pow10(a.q - b.q);
  else if (b.q > a.q) bc *= pow10(b.q - a.q);
  return ac < bc ? -1 : (ac > bc ? 1 : 0);
}

const lt = (a, b) => cmp(a, b) < 0;
const ge = (a, b) => cmp(a, b) >= 0;
const isZero = (a) => a.c === 0n;

/** a × 10^n (Decimal.scaleb). */
const scaleb = (a, n) => ({ c: a.c, q: a.q + n });

/** Exponent of the most significant digit (Decimal.adjusted) of a non-zero decimal. */
const adjusted = (a) => a.c.toString().length - 1 + a.q;

/**
 * Round to `places` decimals. mode 'away': ties away from zero (ROUND_HALF_UP on the magnitude, the D.1 rule);
 * mode 'even': ties to even (Python's float formatting, used only by probExact to mirror narrative.format_prob).
 */
function quantize(a, places, mode = 'away') {
  const target = -places;
  if (a.q >= target) return { c: a.c * pow10(a.q - target), q: target };
  const div = pow10(target - a.q);
  let quot = a.c / div;
  const twice = (a.c % div) * 2n;
  if (twice > div || (twice === div && (mode === 'away' || quot % 2n === 1n))) quot += 1n;
  return { c: quot, q: target };
}

/** Digits with exactly max(0, -q) decimals (Python f"{d:f}"). */
function plain(a) {
  if (a.q >= 0) return (a.c * pow10(a.q)).toString();
  const places = -a.q;
  const s = a.c.toString().padStart(places + 1, '0');
  return `${s.slice(0, -places)}.${s.slice(-places)}`;
}

/** Insert thousands separators into the integer part of a plain digit string. */
function groupDigits(s) {
  const dot = s.indexOf('.');
  const int = dot < 0 ? s : s.slice(0, dot);
  const frac = dot < 0 ? '' : s.slice(dot);
  return int.replace(/\B(?=(\d{3})+(?!\d))/g, ',') + frac;
}

/** An already rounded decimal with thousands separators and exactly `places` decimals (f"{d:,.{places}f}"). */
function grouped(a, places) {
  return groupDigits(plain(quantize(a, places)));
}

/** Strip trailing fractional zeros and a dangling point ('0.0230' → '0.023', '10.0' → '10'). */
function stripZeros(s) {
  return s.includes('.') ? s.replace(/0+$/, '').replace(/\.$/, '') : s;
}

function superscript(n) {
  return String(n).split('').map((ch) => SUPERSCRIPT[ch]).join('');
}

/** ×10ⁿ notation of a positive magnitude with `digits` significant digits, trailing zeros stripped. No sign. */
function sciText(a, digits = 3) {
  let e = adjusted(a);
  let mant = quantize(scaleb(a, -e), digits - 1);
  if (ge(mant, D_TEN)) {               // 9.995 → 10.0: one more power of ten
    e += 1;
    mant = quantize(scaleb(a, -e), digits - 1);
  }
  const body = stripZeros(plain(mant));
  return e === 0 ? body : `${body}×10${superscript(e)}`;
}

function signed(negative, body, plus = false) {
  if (negative) return MINUS + body;
  return plus ? `+${body}` : body;
}

// ------------------------------------------------------------------------------------------------ rounding

/**
 * Canonical display rounding (D.1): round the exact decimal value of the double `x` to `d` decimals, ties away
 * from zero. Returns the digit string Python's `str(round_half_away(x, d))` gives — exactly `d` fractional digits,
 * ASCII '-' for negatives (e.g. '0.13', '1235', '-2.50'); display functions apply MINUS themselves.
 * Valid for finite 1e-6 <= |x| < 1e21 and x = 0.
 * @param {number} x
 * @param {number} d  decimals, 0..20
 * @returns {string}
 */
export function roundHalfAway(x, d) {
  const v = Number(x);
  if (!Number.isFinite(v)) throw new RangeError(`roundHalfAway: ${x} is not finite`);
  const negative = v < 0 || Object.is(v, -0);
  return (negative ? '-' : '') + plain(quantize(exactAbs(v), Math.trunc(d)));
}

// ------------------------------------------------------------------------------------------------ numbers (D.1.1-D.1.3)

/** full() of a non-negative exact decimal (no sign). */
function fullAbs(a) {
  if (isZero(a)) return '0';
  if (lt(a, SCI_SMALL) || ge(a, SCI_LARGE)) return sciText(a);
  let places;
  if (ge(a, D_THOUSAND)) places = 0;
  else if (ge(a, D_TEN)) places = 1;
  else if (ge(a, D_TENTH)) places = 2;
  else return stripZeros(plain(quantize(a, -adjusted(a) + 1)));  // two significant digits
  return grouped(quantize(a, places), places);
}

/**
 * Full number for tables, tooltips and read-outs (D.1.2): number_text decimal rules with roundHalfAway and
 * thousands separators (123,456,789 · 45.2 · 0.13). Same digits in both languages.
 * @param {number|null|undefined} x
 * @param {'ko'|'en'} [lang='ko']
 * @returns {string}
 */
export function full(x, lang = 'ko') {
  const v = finite(x);
  if (v === null) return DASH;
  const body = fullAbs(exactAbs(v));
  return signed(v < 0 && body !== '0', body);
}

/**
 * Compact number for tiles, scenario boxes and axes (D.1.1): full() below 10,000 after rounding; ko 만/억/조,
 * en K (from 1e4)/M/B; at most 3 significant digits; ×10ⁿ from 1e16 (ko) / 1e12 (en).
 * compact(12345,'ko') → '1.2만'; compact(12345,'en') → '12K'; compact(9999.5,'ko') → '1만'.
 * @param {number|null|undefined} x
 * @param {'ko'|'en'} [lang='ko']
 * @returns {string}
 */
export function compact(x, lang = 'ko') {
  const v = finite(x);
  if (v === null) return DASH;
  const key = isEn(lang) ? 'en' : 'ko';
  const a = exactAbs(v);
  if (lt(quantize(a, 0), D_10K)) return full(v, lang);
  const neg = v < 0;
  if (ge(a, COMPACT_SCI[key])) return signed(neg, sciText(a));
  const units = UNITS[key];
  const limit = UNIT_LIMIT[key];
  let i = 0;
  while (i + 1 < units.length && ge(a, scaleb(D_ONE, units[i + 1][0]))) i += 1;
  for (; i < units.length; i += 1) {
    const [power, suffix] = units[i];
    const u = scaleb(a, -power);
    const r1 = quantize(u, 1);
    if (lt(r1, D_TEN)) {
      const body = plain(r1);
      return signed(neg, (body.endsWith('.0') ? body.slice(0, -2) : body) + suffix);
    }
    const r0 = quantize(u, 0);
    if (lt(r0, D_THOUSAND)) return signed(neg, plain(r0) + suffix);
    if (key === 'ko') {                // 1,000만–9,999만 (and 억, 조): three significant digits
      const r3 = scaleb(quantize(scaleb(u, -1), 0), 1);
      if (lt(r3, limit)) return signed(neg, grouped(r3, 0) + suffix);
    }
    // the rounding reached the next unit
  }
  return signed(neg, sciText(a));
}

/**
 * A data value by value kind (D.1.3): relative indexes → one fixed decimal ('71.3', '100.0'), never compacted;
 * other kinds → compact() when `isCompact`, else full(). Used by the crosshair read-out and table views
 * (append the VM's `source.unit_label` yourself).
 * @param {number|null|undefined} x
 * @param {string} kind  value kind
 * @param {'ko'|'en'} [lang='ko']
 * @param {boolean} [isCompact=false]
 * @returns {string}
 */
export function valueText(x, kind, lang = 'ko', isCompact = false) {
  const v = finite(x);
  if (v === null) return DASH;
  if (INDEX_KINDS.has(kind)) {
    const r = quantize(exactAbs(v), 1);
    return signed(v < 0 && !isZero(r), grouped(r, 1));
  }
  return isCompact ? compact(v, lang) : full(v, lang);
}

// ------------------------------------------------------------------------------------------------ percentages (D.1.4)

/** Digits of a percentage (no sign, no '%'); null when it rounds to 0 at one decimal. */
function pctDigits(v) {
  const a = exactAbs(v);
  const r1 = quantize(a, 1);
  if (isZero(r1)) return null;
  if (ge(r1, D_TEN)) return grouped(quantize(a, 0), 0);
  return grouped(r1, 1);
}

/**
 * Signed percentage: '+18%', '+4.1%', '−4.1%', '+1,365%'; |p| rounding to 0 → '0%'.
 * @param {number|null|undefined} p  percent (18.2 means 18.2 %)
 * @returns {string}
 */
export function pct(p) {
  const v = finite(p);
  if (v === null) return DASH;
  const digits = pctDigits(v);
  if (digits === null) return '0%';
  return `${signed(v < 0, digits, true)}%`;
}

/**
 * Like pct() with ▲/▼ instead of the sign: '▲12%', '▼4.1%', zero → '0%'.
 * @param {number|null|undefined} p
 * @returns {string}
 */
export function pctArrow(p) {
  const v = finite(p);
  if (v === null) return DASH;
  const digits = pctDigits(v);
  if (digits === null) return '0%';
  return `${v < 0 ? '▼' : '▲'}${digits}%`;
}

/**
 * Digits of pct() without a sign: '18%'.
 * @param {number|null|undefined} p
 * @returns {string}
 */
export function pctAbs(p) {
  const v = finite(p);
  if (v === null) return DASH;
  return `${pctDigits(v) || '0'}%`;
}

/**
 * Signed percentage points: '+0.4%p' (ko) / '+0.4 pp' (en), '−3.9%p', '+0.042%p'; 0 → '0%p' / '0 pp'.
 * @param {number|null|undefined} p
 * @param {'ko'|'en'} [lang='ko']
 * @returns {string}
 */
export function pp(p, lang = 'ko') {
  const v = finite(p);
  if (v === null) return DASH;
  const suffix = isEn(lang) ? ' pp' : '%p';
  if (v === 0) return `0${suffix}`;
  const a = exactAbs(v);
  const r1 = quantize(a, 1);
  let body;
  if (ge(r1, D_TENTH)) body = grouped(r1, 1);
  else if (lt(a, SCI_SMALL)) body = sciText(a, 2);   // float noise such as 7.4e-15
  else body = stripZeros(plain(quantize(a, -adjusted(a) + 1)));
  return signed(v < 0, body, true) + suffix;
}

// ------------------------------------------------------------------------------------------------ probabilities (D.1.5)

/**
 * '약 80%' / 'about 80%' in 5-point steps; p >= 0.975 → '95% 이상' / '95% or more'; p <= 0.025 → '5% 이하' /
 * '5% or less'.
 * @param {number|null|undefined} p  probability 0..1
 * @param {'ko'|'en'} [lang='ko']
 * @returns {string}
 */
export function probText(p, lang = 'ko') {
  const v = finite(p);
  if (v === null) return DASH;
  const en = isEn(lang);
  if (v >= 0.975) return en ? '95% or more' : '95% 이상';
  if (v <= 0.025) return en ? '5% or less' : '5% 이하';
  const n = Number(roundHalfAway(v * 20, 0)) * 5;    // p·20 is a double product, as in fmt.py
  return en ? `about ${n}%` : `약 ${n}%`;
}

/**
 * narrative.format_prob mirror: '81%', '>99%', '<1%' (tooltips only).
 * @param {number|null|undefined} p
 * @param {'ko'|'en'} [lang='ko']
 * @returns {string}
 */
export function probExact(p, lang = 'ko') {
  let v = finite(p);
  if (v === null) return DASH;
  v = Math.min(1, Math.max(0, v));
  if (v >= 0.995) return '>99%';
  if (v <= 0.005) return '<1%';
  // Python f"{v * 100:.0f}" rounds the exact binary value half to even
  return `${plain(quantize(exactAbs(v * 100), 0, 'even'))}%`;
}

/**
 * '상승 확률 {probText}' / 'chance of rise {probText}'.
 * @param {number|null|undefined} p
 * @param {'ko'|'en'} [lang='ko']
 * @returns {string}
 */
export function probLabelled(p, lang = 'ko') {
  if (finite(p) === null) return DASH;
  const text = probText(p, lang);
  return isEn(lang) ? `chance of rise ${text}` : `상승 확률 ${text}`;
}

/**
 * Likelihood word on the raw P(rise) (D.1.5 table), e.g. likelihood(0.8,'ko') → '상승 쪽이 우세'.
 * @param {number} p
 * @param {'ko'|'en'} [lang='ko']
 * @returns {string}
 */
export function likelihood(p, lang = 'ko') {
  const v = finite(p);
  if (v === null) return DASH;
  const row = LIKELIHOOD.find(([test]) => test(v));
  return isEn(lang) ? row[2] : row[1];
}

// ------------------------------------------------------------------------------------------------ dates (D.1.6)

const pad2 = (n) => String(n).padStart(2, '0');
const pad4 = (n) => String(n).padStart(4, '0');

/** 'YYYY-MM-DD' (a longer ISO string is cut to its date part) → {y, m, d}. */
function parts(iso) {
  const s = String(iso).trim();
  return { y: Number(s.slice(0, 4)), m: Number(s.slice(5, 7)), d: Number(s.slice(8, 10)) };
}

/** {y, m, d} shifted by `days` (UTC calendar arithmetic). */
function addDays({ y, m, d }, days) {
  const t = new Date(Date.UTC(y, m - 1, d + days));
  return { y: t.getUTCFullYear(), m: t.getUTCMonth() + 1, d: t.getUTCDate() };
}

const koDay = ({ y, m, d }) => `${pad4(y)}.${pad2(m)}.${pad2(d)}`;
const enDay = ({ y, m, d }) => `${MONTHS_EN[m - 1]} ${d}, ${y}`;
const quarter = ({ m }) => Math.floor((m - 1) / 3) + 1;

function dateLabelParts(p, freq, lang) {
  if (isEn(lang)) {
    if (freq === 'W') return `Week of ${enDay(p)}`;
    if (freq === 'M') return `${MONTHS_EN[p.m - 1]} ${p.y}`;
    if (freq === 'Q') return `Q${quarter(p)} ${p.y}`;
    if (freq === 'Y') return `${p.y}`;
    return enDay(p);
  }
  if (freq === 'W') return `${koDay(p)} 주`;
  if (freq === 'M') return `${p.y}년 ${p.m}월`;
  if (freq === 'Q') return `${p.y}년 ${quarter(p)}분기`;
  if (freq === 'Y') return `${p.y}년`;
  return `${koDay(p)}.`;
}

function dateShortParts(p, freq, lang) {
  if (isEn(lang)) {
    if (freq === 'M') return `${MONTHS_EN[p.m - 1]} ${p.y}`;
    if (freq === 'Q') return `Q${quarter(p)} ${p.y}`;
    if (freq === 'Y') return `${p.y}`;
    return enDay(p);
  }
  if (freq === 'M') return `${pad4(p.y)}.${pad2(p.m)}`;
  if (freq === 'Q') return `${p.y} ${quarter(p)}분기`;
  if (freq === 'Y') return `${p.y}`;
  return koDay(p);
}

/**
 * Long date label: D '2026.11.25.' / 'Nov 25, 2026'; W '2026.11.17 주' / 'Week of Nov 17, 2026'; M '2026년 11월' /
 * 'Nov 2026'; Q '2026년 4분기' / 'Q4 2026'; Y '2026년' / '2026'. Used for the read-out date.
 * @param {string} iso  'YYYY-MM-DD'
 * @param {'D'|'W'|'M'|'Q'|'Y'} freq
 * @param {'ko'|'en'} [lang='ko']
 * @returns {string}
 */
export function dateLabel(iso, freq, lang = 'ko') {
  return dateLabelParts(parts(iso), freq, lang);
}

/**
 * Short date for lists: D/W '2026.11.25' / 'Nov 25, 2026'; M '2026.11' / 'Nov 2026'; Q '2026 4분기' / 'Q4 2026';
 * Y '2026'.
 * @param {string} iso
 * @param {'D'|'W'|'M'|'Q'|'Y'} freq
 * @param {'ko'|'en'} [lang='ko']
 * @returns {string}
 */
export function dateShort(iso, freq, lang = 'ko') {
  return dateShortParts(parts(iso), freq, lang);
}

/**
 * Period label '{a} ~ {b}' (ko) / '{a} – {b}' (en); for W the end shown is end + 6 days (the last bucket's Sunday).
 * @param {string} start  'YYYY-MM-DD'
 * @param {string} end    'YYYY-MM-DD'
 * @param {'D'|'W'|'M'|'Q'|'Y'} freq
 * @param {'ko'|'en'} [lang='ko']
 * @returns {string}
 */
export function rangeLabel(start, end, freq, lang = 'ko') {
  const a = parts(start);
  let b = parts(end);
  if (freq === 'W') b = addDays(b, 6);
  if (isEn(lang)) return `${dateShortParts(a, freq, 'en')} – ${dateShortParts(b, freq, 'en')}`;
  if (freq === 'Q') return `${dateLabelParts(a, 'Q', 'ko')} ~ ${dateLabelParts(b, 'Q', 'ko')}`;
  return `${dateShortParts(a, freq, 'ko')} ~ ${dateShortParts(b, freq, 'ko')}`;
}

const ISO_TIME = /^(\d{4})-(\d{2})-(\d{2})(?:[T ](\d{2}):(\d{2})(?::(\d{2})(?:\.\d+)?)?)?\s*(Z|[+-]\d{2}:?\d{2})?$/i;

/** UTC milliseconds of an ISO timestamp; a missing zone means UTC (as fmt.py treats naive times). */
function utcMs(iso) {
  const m = ISO_TIME.exec(String(iso).trim());
  if (!m) throw new RangeError(`datetimeLabel: not an ISO timestamp: ${iso}`);
  const [, y, mo, d, hh = '0', mm = '0', ss = '0', zone] = m;
  let offsetMin = 0;
  if (zone && zone.toUpperCase() !== 'Z') {
    const sign = zone[0] === '-' ? -1 : 1;
    const digits = zone.slice(1).replace(':', '');
    offsetMin = sign * (Number(digits.slice(0, 2)) * 60 + Number(digits.slice(2, 4)));
  }
  return Date.UTC(Number(y), Number(mo) - 1, Number(d), Number(hh), Number(mm), Number(ss)) - offsetMin * 60000;
}

/**
 * A UTC timestamp in fixed KST (UTC+9, no Intl time zones): '2026.11.25 09:12' / 'Nov 25, 2026 09:12 KST'.
 * Also used for share/result expiry dates in the UI.
 * @param {string} utcIso  e.g. '2026-11-25T00:12:00Z'
 * @param {'ko'|'en'} [lang='ko']
 * @returns {string}
 */
export function datetimeLabel(utcIso, lang = 'ko') {
  const k = new Date(utcMs(utcIso) + 9 * 3600000);
  const day = { y: k.getUTCFullYear(), m: k.getUTCMonth() + 1, d: k.getUTCDate() };
  const hm = `${pad2(k.getUTCHours())}:${pad2(k.getUTCMinutes())}`;
  return isEn(lang) ? `${enDay(day)} ${hm} KST` : `${koDay(day)} ${hm}`;
}

/** Integer with thousands separators (Python f"{k:,}"). */
function groupInt(k) {
  return (k < 0 ? '-' : '') + groupDigits(String(Math.abs(k)));
}

/**
 * Span words: ko '26주', '12개월', '30일', '8분기', '5년'; en '26 weeks', '12 months', '1 week'.
 * @param {number} n
 * @param {'D'|'W'|'M'|'Q'|'Y'} freq
 * @param {'ko'|'en'} [lang='ko']
 * @returns {string}
 */
export function spanLabel(n, freq, lang = 'ko') {
  const k = Math.trunc(Number(n));
  if (isEn(lang)) {
    const [one, many] = SPAN_EN[freq] || ['period', 'periods'];
    return `${groupInt(k)} ${k === 1 ? one : many}`;
  }
  return `${groupInt(k)}${SPAN_KO[freq] || '기간'}`;
}

/**
 * Frequency label: ko '일 단위'/'주 단위'/'월 단위'/'분기 단위'/'연 단위'; en 'daily'/'weekly'/'monthly'/'quarterly'/
 * 'yearly'.
 * @param {'D'|'W'|'M'|'Q'|'Y'} freq
 * @param {'ko'|'en'} [lang='ko']
 * @returns {string}
 */
export function freqLabel(freq, lang = 'ko') {
  const [ko, en] = FREQ_LABELS[freq] || [freq, freq];
  return isEn(lang) ? en : ko;
}

/**
 * Frequency adjective: ko '일간'/'주간'/'월간'/'분기'/'연간'; en as freqLabel().
 * @param {'D'|'W'|'M'|'Q'|'Y'} freq
 * @param {'ko'|'en'} [lang='ko']
 * @returns {string}
 */
export function freqAdjective(freq, lang = 'ko') {
  if (isEn(lang)) return freqLabel(freq, 'en');
  return FREQ_ADJ_KO[freq] || freq;
}

/**
 * Axis tick label (client only, D.1.6 "Axis ticks"; JS-only, no fmt.py twin).
 *   step 'year'  → '2022'
 *   step 'month' → ko '4월' / en 'Apr'; when `first` (first tick or a January) ko '2026.01' / en 'Jan 2026'
 *   step 'day'   → ko '11.25' / en 'Nov 25'; when `first` (first tick or a year change) ko '2026.11.25' /
 *                  en 'Nov 25, 2026'   (weekly ticks use 'day')
 * @param {number} ms  tick time, UTC milliseconds of the tick's date at 00:00Z
 * @param {'year'|'month'|'day'} step  the tick unit chosen by scale.timeTicks()
 * @param {boolean} first  long form (see above)
 * @param {'ko'|'en'} [lang='ko']
 * @returns {string}
 */
export function axisLabel(ms, step, first, lang = 'ko') {
  const t = new Date(ms);
  const p = { y: t.getUTCFullYear(), m: t.getUTCMonth() + 1, d: t.getUTCDate() };
  const en = isEn(lang);
  if (step === 'year') return String(p.y);
  if (step === 'month') {
    if (first) return en ? `${MONTHS_EN[p.m - 1]} ${p.y}` : `${pad4(p.y)}.${pad2(p.m)}`;
    return en ? MONTHS_EN[p.m - 1] : `${p.m}월`;
  }
  if (first) return en ? enDay(p) : koDay(p);
  return en ? `${MONTHS_EN[p.m - 1]} ${p.d}` : `${pad2(p.m)}.${pad2(p.d)}`;
}
