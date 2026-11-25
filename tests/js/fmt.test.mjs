// fmt.js: every D.1.7 vector (the exact-tie rows included), the roundHalfAway reference strings exported from
// Python (fixtures/round_vectors.json), D.1.6 date tables in both languages and the axis label rules.
// Run: node --test tests/js
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

import * as fmt from '../../trendai/web/static/js/fmt.js';

const fixture = JSON.parse(readFileSync(new URL('./fixtures/round_vectors.json', import.meta.url), 'utf8'));

// ------------------------------------------------------------------------------------------------ D.1.7 compact

const COMPACT = [
  [0, '0', '0'],
  [9876, '9,876', '9,876'],
  [45.23, '45.2', '45.2'],
  [1.204, '1.20', '1.20'],
  [0.0234, '0.023', '0.023'],
  [0.00005, '5×10⁻⁵', '5×10⁻⁵'],
  [12345, '1.2만', '12K'],
  [12500, '1.3만', '13K'],
  [99999, '10만', '100K'],
  [340000, '34만', '340K'],
  [3456789, '346만', '3.5M'],
  [12345678, '1,230만', '12M'],
  [98765432, '9,880만', '99M'],
  [99999999, '1억', '100M'],
  [99950000, '1억', '100M'],
  [340000000, '3.4억', '340M'],
  [999999999, '10억', '1B'],
  [1.2e12, '1.2조', '1.2×10¹²'],
  [3.4e16, '3.4×10¹⁶', '3.4×10¹⁶'],
  [-12345, '−1.2만', '−12K'],
  // exact binary ties (half away from zero)
  [0.125, '0.13', '0.13'],
  [2.5, '2.50', '2.50'],
  [12.25, '12.3', '12.3'],
  [1234.5, '1,235', '1,235'],
  [9999.5, '1만', '10K'],
];

test('compact: every D.1.7 row in ko and en', () => {
  for (const [x, ko, en] of COMPACT) {
    assert.equal(fmt.compact(x, 'ko'), ko, `compact(${x}, ko)`);
    assert.equal(fmt.compact(x, 'en'), en, `compact(${x}, en)`);
  }
});

// ------------------------------------------------------------------------------------------------ D.1.7 other calls

test('full: D.1.7 rows, both languages', () => {
  for (const [x, out] of [[123456789, '123,456,789'], [1234.5, '1,235'], [0.125, '0.13']]) {
    assert.equal(fmt.full(x, 'ko'), out);
    assert.equal(fmt.full(x, 'en'), out);
  }
});

test('valueText: relative indexes print one fixed decimal and are never compacted', () => {
  for (const [x, out] of [[71.34, '71.3'], [71.25, '71.3'], [100, '100.0']]) {
    for (const lang of ['ko', 'en']) {
      assert.equal(fmt.valueText(x, 'relative_index', lang), out);
      assert.equal(fmt.valueText(x, 'relative_index', lang, true), out);
      assert.equal(fmt.valueText(x, 'relative_index_gt', lang, true), out);
    }
  }
  assert.equal(fmt.valueText(12345, 'pageviews', 'ko'), '12,345');
  assert.equal(fmt.valueText(12345, 'pageviews', 'ko', true), '1.2만');
  assert.equal(fmt.valueText(12345, 'raw', 'en', true), '12K');
  assert.equal(fmt.valueText(-0.04, 'relative_index'), '0.0');
});

test('pct, pctArrow, pctAbs: D.1.7 rows', () => {
  const pct = [[18.2, '+18%'], [4.06, '+4.1%'], [-4.06, '−4.1%'], [0.02, '0%'], [9.96, '+10%'],
    [1365.4, '+1,365%'], [2.25, '+2.3%'], [0.05, '+0.1%']];
  for (const [p, out] of pct) assert.equal(fmt.pct(p), out, `pct(${p})`);
  for (const [p, out] of [[12.03, '▲12%'], [-4.06, '▼4.1%'], [0, '0%']]) assert.equal(fmt.pctArrow(p), out);
  assert.equal(fmt.pctAbs(-18.2), '18%');
  assert.equal(fmt.pctAbs(0.01), '0%');
});

test('pp: D.1.7 rows and edges', () => {
  const rows = [[0.4, '+0.4%p', '+0.4 pp'], [-3.91, '−3.9%p', '−3.9 pp'], [12.34, '+12.3%p', '+12.3 pp'],
    [0.042, '+0.042%p', '+0.042 pp'], [0.25, '+0.3%p', '+0.3 pp']];
  for (const [p, ko, en] of rows) {
    assert.equal(fmt.pp(p, 'ko'), ko);
    assert.equal(fmt.pp(p, 'en'), en);
  }
  assert.equal(fmt.pp(0, 'ko'), '0%p');
  assert.equal(fmt.pp(-0, 'en'), '0 pp');
  assert.equal(fmt.pp(7.4e-15, 'ko'), '+7.4×10⁻¹⁵%p');
  assert.equal(fmt.pp(-0.00005, 'en'), '−5×10⁻⁵ pp');
  assert.equal(fmt.pp(1234.56, 'ko'), '+1,234.6%p');
});

test('probText, probLabelled: D.1.7 rows', () => {
  const rows = [[0.81, '약 80%', 'about 80%'], [0.83, '약 85%', 'about 85%'], [0.5, '약 50%', 'about 50%'],
    [0.974, '약 95%', 'about 95%'], [0.975, '95% 이상', '95% or more'], [0.99, '95% 이상', '95% or more'],
    [0.03, '약 5%', 'about 5%'], [0.02, '5% 이하', '5% or less'], [0.825, '약 85%', 'about 85%'],
    [0.875, '약 90%', 'about 90%']];
  for (const [p, ko, en] of rows) {
    assert.equal(fmt.probText(p, 'ko'), ko, `probText(${p})`);
    assert.equal(fmt.probText(p, 'en'), en);
  }
  assert.equal(fmt.probLabelled(0.2, 'ko'), '상승 확률 약 20%');
  assert.equal(fmt.probLabelled(0.02, 'ko'), '상승 확률 5% 이하');
  assert.equal(fmt.probLabelled(0.2, 'en'), 'chance of rise about 20%');
  assert.equal(fmt.probLabelled(0.02, 'en'), 'chance of rise 5% or less');
});

test('probExact mirrors narrative.format_prob (half-even on the exact value)', () => {
  assert.equal(fmt.probExact(0.812), '81%');
  assert.equal(fmt.probExact(0.5), '50%');
  assert.equal(fmt.probExact(0.996), '>99%');
  assert.equal(fmt.probExact(0.004), '<1%');
  assert.equal(fmt.probExact(1), '>99%');
  assert.equal(fmt.probExact(0.125), '12%');     // 12.5 exactly: Python's .0f rounds to even
  assert.equal(fmt.probExact(0.135), '14%');
});

test('likelihood: D.1.5 rows and bin edges', () => {
  const ko = ['상승 가능성 매우 높음', '상승 쪽이 우세', '상승 쪽이 약간 우세', '상승·하락 엇비슷', '하락 쪽이 약간 우세',
    '하락 쪽이 우세', '하락 가능성 매우 높음'];
  const en = ['Rise very likely', 'Rise more likely', 'Rise slightly more likely', 'About even',
    'Fall slightly more likely', 'Fall more likely', 'Fall very likely'];
  [0.9, 0.8, 0.6, 0.5, 0.4, 0.2, 0.05].forEach((p, i) => {
    assert.equal(fmt.likelihood(p, 'ko'), ko[i]);
    assert.equal(fmt.likelihood(p, 'en'), en[i]);
  });
  const edges = [[0.90, 0], [0.8999, 1], [0.70, 1], [0.55, 2], [0.5499, 3], [0.4501, 3], [0.45, 4], [0.30, 5],
    [0.10, 6]];
  for (const [p, i] of edges) assert.equal(fmt.likelihood(p), ko[i], `likelihood(${p})`);
});

test('D.1.7 date rows', () => {
  assert.equal(fmt.dateLabel('2026-11-17', 'W', 'ko'), '2026.11.17 주');
  assert.equal(fmt.dateLabel('2026-11-17', 'W', 'en'), 'Week of Nov 17, 2026');
  assert.equal(fmt.rangeLabel('2021-11-01', '2026-10-01', 'M', 'ko'), '2021.11 ~ 2026.10');
  assert.equal(fmt.rangeLabel('2021-11-01', '2026-10-01', 'M', 'en'), 'Nov 2021 – Oct 2026');
  assert.equal(fmt.rangeLabel('2021-11-29', '2026-11-16', 'W', 'ko'), '2021.11.29 ~ 2026.11.22');
  assert.equal(fmt.rangeLabel('2021-11-29', '2026-11-16', 'W', 'en'), 'Nov 29, 2021 – Nov 22, 2026');
  assert.equal(fmt.datetimeLabel('2026-11-25T00:12:00Z', 'ko'), '2026.11.25 09:12');
  assert.equal(fmt.datetimeLabel('2026-11-25T00:12:00Z', 'en'), 'Nov 25, 2026 09:12 KST');
});

// ------------------------------------------------------------------------------------------------ roundHalfAway

test('roundHalfAway matches the Python reference strings (fixtures/round_vectors.json)', () => {
  assert.ok(fixture.rows.length >= 1000);
  for (const { x, d, out } of fixture.rows) {
    assert.equal(fmt.roundHalfAway(x, d), out, `roundHalfAway(${x}, ${d})`);
  }
});

test('roundHalfAway: exact binary value decides, ties go away from zero', () => {
  assert.equal(fmt.roundHalfAway(0.125, 2), '0.13');
  assert.equal(fmt.roundHalfAway(1234.5, 0), '1235');
  assert.equal(fmt.roundHalfAway(-2.5, 2), '-2.50');
  assert.equal(fmt.roundHalfAway(71.25, 1), '71.3');
  assert.equal(fmt.roundHalfAway(2.675, 2), '2.67');   // 2.67499999… as a double
  assert.equal(fmt.roundHalfAway(1.005, 2), '1.00');   // 1.00499999…
  assert.throws(() => fmt.roundHalfAway(NaN, 2), RangeError);
});

// ------------------------------------------------------------------------------------------------ edges shared with fmt.py

test('missing values print a dash', () => {
  for (const bad of [null, undefined, NaN, Infinity, -Infinity]) {
    for (const fn of [fmt.compact, fmt.full, fmt.pct, fmt.pctArrow, fmt.pctAbs, fmt.pp, fmt.probText, fmt.probExact,
      fmt.probLabelled, fmt.likelihood]) {
      assert.equal(fn(bad), fmt.DASH);
    }
    assert.equal(fmt.valueText(bad, 'relative_index'), fmt.DASH);
    assert.equal(fmt.valueText(bad, 'raw', 'ko', true), fmt.DASH);
  }
});

test('minus sign, zero and notation edges (same as tests/test_web_fmt.py)', () => {
  assert.equal(fmt.full(-45.23), '−45.2');
  assert.equal(fmt.full(-0), '0');
  assert.equal(fmt.compact(-3456789, 'en'), '−3.5M');
  assert.ok(!fmt.compact(-12345, 'ko').includes('-'));
  assert.equal(fmt.full(1.5e12), '1.5×10¹²');
  assert.equal(fmt.full(0.0001), '0.0001');
  assert.equal(fmt.full(0.09996), '0.1');
  assert.equal(fmt.full(999.95), '1,000.0');
  assert.equal(fmt.compact(123456789012, 'ko'), '1,230억');
  assert.equal(fmt.compact(9.9995e15, 'ko'), '1×10¹⁶');
  assert.equal(fmt.compact(999.95e9, 'en'), '1×10¹²');
});

// ------------------------------------------------------------------------------------------------ D.1.6 tables

test('dateLabel and dateShort for all five frequencies', () => {
  const label = { D: ['2026.11.25.', 'Nov 25, 2026'], W: ['2026.11.25 주', 'Week of Nov 25, 2026'],
    M: ['2026년 11월', 'Nov 2026'], Q: ['2026년 4분기', 'Q4 2026'], Y: ['2026년', '2026'] };
  const short = { D: ['2026.11.25', 'Nov 25, 2026'], W: ['2026.11.25', 'Nov 25, 2026'], M: ['2026.11', 'Nov 2026'],
    Q: ['2026 4분기', 'Q4 2026'], Y: ['2026', '2026'] };
  for (const f of 'DWMQY') {
    assert.equal(fmt.dateLabel('2026-11-25', f, 'ko'), label[f][0]);
    assert.equal(fmt.dateLabel('2026-11-25', f, 'en'), label[f][1]);
    assert.equal(fmt.dateShort('2026-11-25', f, 'ko'), short[f][0]);
    assert.equal(fmt.dateShort('2026-11-25', f, 'en'), short[f][1]);
  }
  assert.equal(fmt.dateLabel('2026-03-05', 'M', 'ko'), '2026년 3월');
  assert.equal(fmt.dateLabel('2026-03-05', 'D', 'en'), 'Mar 5, 2026');
});

test('rangeLabel for all five frequencies (W end = Sunday)', () => {
  const rows = [
    ['2021-11-29', '2026-11-16', 'D', '2021.11.29 ~ 2026.11.16', 'Nov 29, 2021 – Nov 16, 2026'],
    ['2021-11-01', '2026-10-26', 'D', '2021.11.01 ~ 2026.10.26', 'Nov 1, 2021 – Oct 26, 2026'],
    ['2021-11-29', '2026-11-16', 'W', '2021.11.29 ~ 2026.11.22', 'Nov 29, 2021 – Nov 22, 2026'],
    ['2025-12-22', '2026-12-28', 'W', '2025.12.22 ~ 2027.01.03', 'Dec 22, 2025 – Jan 3, 2027'],
    ['2021-11-01', '2026-10-01', 'M', '2021.11 ~ 2026.10', 'Nov 2021 – Oct 2026'],
    ['2021-01-01', '2026-07-01', 'Q', '2021년 1분기 ~ 2026년 3분기', 'Q1 2021 – Q3 2026'],
    ['2016-01-01', '2025-01-01', 'Y', '2016 ~ 2025', '2016 – 2025'],
  ];
  for (const [a, b, f, ko, en] of rows) {
    assert.equal(fmt.rangeLabel(a, b, f, 'ko'), ko);
    assert.equal(fmt.rangeLabel(a, b, f, 'en'), en);
  }
});

test('datetimeLabel crosses midnight in KST and accepts offsets', () => {
  assert.equal(fmt.datetimeLabel('2026-11-30T15:00:00Z', 'ko'), '2026.12.01 00:00');
  assert.equal(fmt.datetimeLabel('2026-11-30T15:00:00+00:00', 'en'), 'Dec 1, 2026 00:00 KST');
  assert.equal(fmt.datetimeLabel('2026-11-30T23:30:00+09:00', 'ko'), '2026.11.30 23:30');
  assert.equal(fmt.datetimeLabel('2026-11-25T00:12:03.123Z', 'ko'), '2026.11.25 09:12');
});

test('spanLabel, freqLabel, freqAdjective', () => {
  const spans = [[26, 'W', '26주', '26 weeks'], [12, 'M', '12개월', '12 months'], [30, 'D', '30일', '30 days'],
    [8, 'Q', '8분기', '8 quarters'], [5, 'Y', '5년', '5 years'], [1, 'W', '1주', '1 week'],
    [104, 'W', '104주', '104 weeks'], [1500, 'D', '1,500일', '1,500 days']];
  for (const [n, f, ko, en] of spans) {
    assert.equal(fmt.spanLabel(n, f, 'ko'), ko);
    assert.equal(fmt.spanLabel(n, f, 'en'), en);
  }
  assert.deepEqual([...'DWMQY'].map((f) => fmt.freqLabel(f, 'ko')), ['일 단위', '주 단위', '월 단위', '분기 단위', '연 단위']);
  assert.deepEqual([...'DWMQY'].map((f) => fmt.freqLabel(f, 'en')), ['daily', 'weekly', 'monthly', 'quarterly', 'yearly']);
  assert.deepEqual([...'DWMQY'].map((f) => fmt.freqAdjective(f, 'ko')), ['일간', '주간', '월간', '분기', '연간']);
  assert.equal(fmt.freqAdjective('W', 'en'), 'weekly');
});

test('axisLabel rules (D.1.6 axis ticks)', () => {
  const ms = (iso) => Date.parse(`${iso}T00:00:00Z`);
  assert.equal(fmt.axisLabel(ms('2022-01-01'), 'year', true, 'ko'), '2022');
  assert.equal(fmt.axisLabel(ms('2022-01-01'), 'year', false, 'en'), '2022');
  assert.equal(fmt.axisLabel(ms('2026-04-01'), 'month', false, 'ko'), '4월');
  assert.equal(fmt.axisLabel(ms('2026-04-01'), 'month', false, 'en'), 'Apr');
  assert.equal(fmt.axisLabel(ms('2026-01-01'), 'month', true, 'ko'), '2026.01');
  assert.equal(fmt.axisLabel(ms('2026-01-01'), 'month', true, 'en'), 'Jan 2026');
  assert.equal(fmt.axisLabel(ms('2026-11-25'), 'day', false, 'ko'), '11.25');
  assert.equal(fmt.axisLabel(ms('2026-11-25'), 'day', false, 'en'), 'Nov 25');
  assert.equal(fmt.axisLabel(ms('2026-11-25'), 'day', true, 'ko'), '2026.11.25');
  assert.equal(fmt.axisLabel(ms('2026-11-25'), 'day', true, 'en'), 'Nov 25, 2026');
});

test('any lang other than en is Korean', () => {
  assert.equal(fmt.compact(12345, 'fr'), '1.2만');
  assert.equal(fmt.dateLabel('2026-11-17', 'W', undefined), '2026.11.17 주');
});
