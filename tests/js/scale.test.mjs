// chart/scale.js (pure): nice 1-2-5 ticks with 4–6 / 3–4 counts, the zero-inclusion rule, the relative-index 0–100
// frame of the comparison chart (extended above 100), log decade ticks, the anomaly range exclusion rule, time ticks
// (D.1.6), y tick labels and label dodging. Run: node --test tests/js
import { test } from 'node:test';
import assert from 'node:assert/strict';

import {
  compareYDomain, detailYDomain, dodge, indexValues, isoToMs, linearScale, logScale, logTicks, msToIso, niceTicks,
  tickCount, tickLabels, timeTicks,
} from '../../trendai/web/static/js/chart/scale.js';

const WIDE = { min: 4, max: 6 };
const NARROW = { min: 3, max: 4 };
const isNice = (step) => {
  const m = step / 10 ** Math.floor(Math.log10(step) + 1e-9);
  return [1, 2, 5].some((k) => Math.abs(m - k) < 1e-9);
};

test('tickCount follows the 640 px breakpoint', () => {
  assert.deepEqual(tickCount(1120), WIDE);
  assert.deepEqual(tickCount(640), WIDE);
  assert.deepEqual(tickCount(639), NARROW);
  assert.deepEqual(tickCount(358), NARROW);
});

test('isoToMs / msToIso are UTC and inverse', () => {
  assert.equal(isoToMs('2026-11-16'), Date.UTC(2026, 10, 16));
  assert.equal(msToIso(isoToMs('2024-02-29')), '2024-02-29');
  assert.equal(msToIso(isoToMs('1999-12-31')), '1999-12-31');
});

test('niceTicks: 1-2-5 steps and counts within 4–6 (wide) / 3–4 (narrow)', () => {
  const ranges = [[0, 103], [0, 63], [0, 71.6], [0, 100], [88, 112], [0, 8.3], [29, 62.96], [0, 98.5], [1234, 98765],
    [-3.2, 5.1], [0.012, 0.087], [1e6, 3.4e6], [0, 1], [5, 1234567], [17.999, 47.0003], [0, 4.2]];
  for (const [lo, hi] of ranges) {
    for (const count of [WIDE, NARROW]) {
      const t = niceTicks(lo, hi, count);
      assert.ok(isNice(t.step), `step ${t.step} for [${lo}, ${hi}]`);
      assert.ok(t.values.length <= count.max, `${t.values.length} ticks > ${count.max} for [${lo}, ${hi}]`);
      assert.ok(t.values.length >= Math.min(count.min, 3) - 1, `too few ticks for [${lo}, ${hi}]`);
      assert.ok(t.min <= lo + 1e-9 && t.max >= hi - 1e-9, `domain covers [${lo}, ${hi}]`);
      assert.equal(t.min, t.values[0], 'the baseline is a tick');
      for (let i = 1; i < t.values.length; i += 1) {
        assert.ok(Math.abs(t.values[i] - t.values[i - 1] - t.step) < t.step * 1e-6, 'evenly spaced');
      }
    }
  }
});

test('niceTicks: expected grids for the common ranges', () => {
  assert.deepEqual(niceTicks(0, 100, WIDE).values, [0, 20, 40, 60, 80, 100]);
  assert.deepEqual(niceTicks(0, 100, NARROW).values, [0, 50, 100]);
  // the top tick stays at or below the data maximum: 0–103 is not stretched to 0–150
  const t = niceTicks(0, 103, WIDE);
  assert.deepEqual(t.values, [0, 20, 40, 60, 80, 100]);
  assert.equal(t.max, 103);
  // … unless the next grid line is very close (within 6 % of the axis)
  assert.deepEqual(niceTicks(0, 98.5, WIDE).values, [0, 20, 40, 60, 80, 100]);
  assert.deepEqual(niceTicks(0, 63, NARROW).values, [0, 20, 40, 60]);
  assert.deepEqual(niceTicks(0.012, 0.087, WIDE).values, [0, 0.02, 0.04, 0.06, 0.08]);
});

test('niceTicks: decimals are the step decimals; degenerate ranges widen', () => {
  assert.equal(niceTicks(0, 100, WIDE).decimals, 0);
  assert.equal(niceTicks(0, 1, WIDE).decimals, 1);
  assert.equal(niceTicks(0.012, 0.087, WIDE).decimals, 2);
  assert.deepEqual(niceTicks(0, 0, WIDE).values, [0, 0.2, 0.4, 0.6, 0.8, 1]);
  const flat = niceTicks(5, 5, NARROW);
  assert.ok(flat.min < 5 && flat.max > 5);
  assert.ok(!Object.is(niceTicks(-1, 1, WIDE).values.find((v) => v === 0), -0), 'no negative zero');
});

test('detailYDomain: y_hint with include_zero; last actual always in range', () => {
  const { ticks, excluded } = detailYDomain({ min: 29, max: 62.96, include_zero: true }, [], null, WIDE);
  assert.equal(ticks.min, 0);
  assert.ok(ticks.max >= 62.96);
  assert.deepEqual(excluded, []);
  const noZero = detailYDomain({ min: 88, max: 112, include_zero: false }, [], null, WIDE).ticks;
  assert.ok(noZero.min > 0 && noZero.min <= 88);
  const withLast = detailYDomain({ min: 88, max: 112, include_zero: false }, [], { date: 'x', value: 140 }, WIDE).ticks;
  assert.ok(withLast.max >= 140);
});

test('detailYDomain: anomalies that would more than double the range are excluded', () => {
  const hint = { min: 0, max: 348, include_zero: true };
  const anomalies = [{ date: '2021-01-01', value: 348 }, { date: '2021-04-01', value: 44.6 }];
  const rest = { min: 20, max: 110 };
  const { ticks, excluded } = detailYDomain(hint, anomalies, { date: '2025-12-01', value: 26 }, WIDE, rest);
  assert.deepEqual(excluded, [0]);
  assert.ok(ticks.max < 200, `domain ${ticks.max} ignores the spike`);
  // a moderate anomaly (does not double the range) stays in range
  const mild = detailYDomain({ min: 0, max: 150, include_zero: true }, [{ date: 'a', value: 150 }], null, WIDE,
    { min: 20, max: 110 });
  assert.deepEqual(mild.excluded, []);
  assert.ok(mild.ticks.max >= 150);
  // a low outlier below a non-zero range is excluded the same way
  const low = detailYDomain({ min: -500, max: 110, include_zero: false }, [{ date: 'a', value: -500 }], null, WIDE,
    { min: 90, max: 110 });
  assert.deepEqual(low.excluded, [0]);
  assert.ok(low.ticks.min > -100);
  // without the non-anomaly range the hint is used and nothing is excluded
  assert.deepEqual(detailYDomain(hint, anomalies, null, WIDE).excluded, []);
});

test('compareYDomain: relative indexes use a 0–100 frame, extended above 100 only when needed', () => {
  const base = { valueKind: 'relative_index', scale: 'raw', bandMax: null, count: WIDE };
  assert.deepEqual(compareYDomain({ ...base, dataMin: 0.3, dataMax: 64 }).ticks.values, [0, 20, 40, 60, 80, 100]);
  assert.deepEqual(compareYDomain({ ...base, dataMin: 2, dataMax: 100 }).ticks.values, [0, 20, 40, 60, 80, 100]);
  const ext = compareYDomain({ ...base, dataMin: 2, dataMax: 100, bandMax: 131 }).ticks;
  assert.ok(ext.max >= 131 && ext.values[0] === 0);
  const gt = compareYDomain({ ...base, valueKind: 'relative_index_gt', dataMin: 10, dataMax: 40, count: NARROW });
  assert.deepEqual(gt.ticks.values, [0, 50, 100]);
});

test('compareYDomain: zero inclusion rule for other kinds, index scale and log', () => {
  const raw = { valueKind: 'raw', scale: 'raw', bandMax: null, count: WIDE };
  assert.equal(compareYDomain({ ...raw, dataMin: 40, dataMax: 120 }).ticks.min, 0);    // 40 <= 60
  assert.ok(compareYDomain({ ...raw, dataMin: 90, dataMax: 120 }).ticks.min > 0);      // 90 > 60
  const idx = compareYDomain({ ...raw, valueKind: 'relative_index', scale: 'index', dataMin: 80, dataMax: 260 });
  assert.equal(idx.log, false);
  assert.ok(idx.ticks.max >= 260);
  const log = compareYDomain({ ...raw, scale: 'log', dataMin: 3, dataMax: 400 });
  assert.equal(log.log, true);
  assert.ok(log.ticks.values.every((v) => v > 0));
});

test('logTicks: 1-2-5 per decade, decades only when crowded', () => {
  assert.deepEqual(logTicks(1, 10).values, [1, 2, 5, 10]);
  const wide = logTicks(0.03, 105, 7);
  assert.deepEqual(wide.values, [0.01, 0.1, 1, 10, 100]);
  assert.equal(wide.max, 105);
  const many = logTicks(1e-3, 1e9, 7);
  assert.ok(many.values.length <= 7);
  assert.ok(many.values.every((v) => Math.abs(Math.log10(v) - Math.round(Math.log10(v))) < 1e-9));
});

test('linear and log scales map and invert', () => {
  const y = linearScale([0, 100], [300, 20]);
  assert.equal(y.map(0), 300);
  assert.equal(y.map(100), 20);
  assert.ok(Math.abs(y.invert(y.map(37.5)) - 37.5) < 1e-9);
  const flat = linearScale([5, 5], [0, 100]);
  assert.equal(flat.map(5), 50);
  const lg = logScale([1, 1000], [300, 0]);
  assert.equal(lg.map(10), 200);
  assert.ok(Number.isNaN(lg.map(0)));
  assert.ok(Math.abs(lg.invert(100) - 100) < 1e-9);
});

test('indexValues divides by the base and keeps nulls', () => {
  assert.deepEqual(indexValues([50, null, 75], 50), [100, null, 150]);
  assert.deepEqual(indexValues([1, 2], 0), [null, null]);
  assert.deepEqual(indexValues([1, 2], null), [null, null]);
});

test('timeTicks: yearly from 3 years, months from 6 months, days/weeks below (D.1.6)', () => {
  const years = timeTicks(isoToMs('2021-11-29'), isoToMs('2027-05-17'), 8);
  assert.equal(years.step, 'year');
  assert.deepEqual(years.ticks.map((t) => msToIso(t.ms)), ['2022-01-01', '2023-01-01', '2024-01-01', '2025-01-01',
    '2026-01-01', '2027-01-01']);
  const thinned = timeTicks(isoToMs('2016-01-01'), isoToMs('2026-12-01'), 3);
  assert.equal(thinned.step, 'year');
  assert.ok(thinned.ticks.length <= 3);

  const months = timeTicks(isoToMs('2025-01-01'), isoToMs('2025-08-18'), 10);
  assert.equal(months.step, 'month');
  assert.equal(msToIso(months.ticks[0].ms), '2025-01-01');
  assert.equal(months.ticks[0].first, true);
  assert.equal(months.ticks[1].first, false);
  const crossing = timeTicks(isoToMs('2025-07-01'), isoToMs('2026-06-30'), 6);
  assert.ok(crossing.ticks.length <= 6);
  assert.ok(crossing.ticks.some((t) => msToIso(t.ms) === '2026-01-01' && t.first), 'January is a long label');
  assert.ok(crossing.ticks.every((t) => new Date(t.ms).getUTCMonth() % 2 === 0), 'aligned to the interval');

  const days = timeTicks(isoToMs('2025-01-01'), isoToMs('2025-02-10'), 6);
  assert.equal(days.step, 'day');
  assert.ok(days.ticks.length <= 6);
  assert.ok(days.ticks.every((t) => new Date(t.ms).getUTCDay() === 1), 'weekly ticks fall on Mondays');
  const year = timeTicks(isoToMs('2025-12-01'), isoToMs('2026-01-20'), 10);
  const jan = year.ticks.find((t) => msToIso(t.ms).startsWith('2026'));
  assert.equal(jan.first, true, 'a year change gets the long label');
});

test('tickLabels: step decimals, compact from 10,000 (one style per axis), never compact for indexes', () => {
  assert.deepEqual(tickLabels([0, 20, 40, 60, 80, 100], 0, 'ko', { kind: 'relative_index' }),
    ['0', '20', '40', '60', '80', '100']);
  assert.deepEqual(tickLabels([0, 0.5, 1, 1.5], 1, 'ko'), ['0.0', '0.5', '1.0', '1.5']);
  assert.deepEqual(tickLabels([0, 5000, 10000, 15000], 0, 'ko'), ['0', '5,000', '1만', '1.5만']);
  assert.deepEqual(tickLabels([0, 5000, 10000, 15000], 0, 'en'), ['0', '5,000', '10K', '15K']);
  assert.deepEqual(tickLabels([-4, -2, 0, 2], 0, 'ko'), ['−4', '−2', '0', '2']);
  // compact labels that would repeat fall back to full digits
  assert.deepEqual(tickLabels([1000000, 1020000, 1040000], 0, 'en'), ['1,000,000', '1,020,000', '1,040,000']);
  assert.deepEqual(tickLabels([0.01, 0.1, 1, 10, 100, 1000, 100000], 0, 'ko', { log: true }),
    ['0.01', '0.1', '1', '10', '100', '1,000', '10만']);
});

test('dodge: labels at least `gap` apart, in bounds, minimal moves, input order kept', () => {
  assert.deepEqual(dodge([10, 50, 90], 14, 0, 100), [10, 50, 90]);
  const close = dodge([50, 52, 51], 14, 0, 200);
  const sorted = close.slice().sort((a, b) => a - b);
  assert.ok(sorted[1] - sorted[0] >= 14 - 1e-9 && sorted[2] - sorted[1] >= 14 - 1e-9);
  assert.ok(Math.abs(close.reduce((s, v) => s + v, 0) / 3 - 51) < 1e-9, 'the group stays centred');
  assert.ok(close[0] < close[2] && close[2] < close[1], 'order follows the wanted positions');
  const edge = dodge([98, 99, 100], 14, 0, 100);
  assert.ok(Math.max(...edge) <= 100 && Math.min(...edge) >= 0);
  assert.deepEqual(dodge([], 14, 0, 100), []);
});
