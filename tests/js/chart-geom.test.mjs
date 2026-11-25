// chart/geom.js (pure geometry and text helpers): line paths with gaps, band areas, the forecast seam at the last
// actual (F.7.1), slot icon paths, text width estimates, truncation, unit labels and circled numbers.
// Run: node --test tests/js
import { test } from 'node:test';
import assert from 'node:assert/strict';

import {
  areaPath, circled, estimateTextWidth, extent, linePath, seamForecast, slotIconPath, truncateToWidth, withUnit,
} from '../../trendai/web/static/js/chart/geom.js';

test('linePath: polyline, gaps on null points, single points kept as dots', () => {
  assert.equal(linePath([[0, 0], [10, 5.04], [20, 10]]), 'M0,0L10,5L20,10');
  assert.equal(linePath([[0, 0], [10, 5], null, [30, 1], [40, 2]]), 'M0,0L10,5M30,1L40,2');
  assert.equal(linePath([[0, 0], null, [20, 3], null]), 'M0,0L0,0M20,3L20,3');
  assert.equal(linePath([[0, NaN], null]), '');
  assert.equal(linePath([]), '');
});

test('areaPath: closed band, split where a bound is missing', () => {
  assert.equal(areaPath([0, 10, 20], [5, 6, 7], [1, 2, 3]), 'M0,1L10,2L20,3L20,7L10,6L0,5Z');
  const split = areaPath([0, 10, 20, 30, 40], [5, 6, null, 8, 9], [1, 2, 3, 4, 5]);
  assert.equal(split, 'M0,1L10,2L10,6L0,5ZM30,4L40,5L40,9L30,8Z');
  assert.equal(areaPath([0], [1], [2]), '');
});

test('seamForecast: mean and bands start at the last actual with zero width', () => {
  const fc = { dates: ['2026-11-23', '2026-11-30'], mean: [63.4, 64], lo80: [58.1, 57], hi80: [68.7, 70],
    lo95: [55.2, 54], hi95: [71.6, 73] };
  const s = seamForecast({ date: '2026-11-16', value: 61.2 }, fc);
  assert.equal(s.seam, true);
  assert.deepEqual(s.dates, ['2026-11-16', '2026-11-23', '2026-11-30']);
  for (const key of ['mean', 'lo80', 'hi80', 'lo95', 'hi95']) assert.equal(s[key][0], 61.2);
  assert.deepEqual(s.mean.slice(1), [63.4, 64]);
  assert.deepEqual(fc.dates, ['2026-11-23', '2026-11-30'], 'input not mutated');
  assert.equal(seamForecast(null, fc).seam, false);
  assert.equal(seamForecast({ date: '2026-11-16', value: null }, fc).seam, false);
  assert.equal(seamForecast({ date: '2026-11-16', value: 1 }, { dates: [] }).dates.length, 0);
  const partial = seamForecast({ date: '2026-11-16', value: 1 }, { dates: ['2026-11-23'] });
  assert.deepEqual(partial.lo95, [1, null]);
});

test('slotIconPath: one closed path per icon, centred on the point', () => {
  for (const icon of ['circle', 'diamond', 'hexagon', 'square', 'cross', null, 'unknown']) {
    const d = slotIconPath(icon, 50, 20, 4);
    assert.match(d, /^M[\d.,]+.*Z$/);
    const nums = (d.match(/-?\d+(\.\d+)?/g) || []).map(Number);
    assert.ok(nums.every((n) => Number.isFinite(n)));
  }
  assert.equal(slotIconPath('square', 10, 10, 4), 'M6.4,6.4L13.6,6.4L13.6,13.6L6.4,13.6Z');
  assert.equal(slotIconPath('diamond', 0, 0, 4), 'M0,-5.2L5.2,0L0,5.2L-5.2,0Z');
  // UX-29: the hexagon is a quarter taller than the circle, so it does not read as a dot at 1×
  const ys = (d) => [...d.matchAll(/-?[\d.]+,(-?[\d.]+)/g)].map((m) => Number(m[1]));
  assert.equal(Math.max(...ys(slotIconPath('hexagon', 0, 0, 4))) - Math.min(...ys(slotIconPath('hexagon', 0, 0, 4))), 10);
  assert.ok(slotIconPath('hexagon', 0, 0, 4).startsWith('M0,-5L'));
});

test('estimateTextWidth: Hangul wider than digits, scales with font size', () => {
  assert.ok(estimateTextWidth('선크림', 12) > estimateTextWidth('123', 12));
  assert.equal(estimateTextWidth('ab', 24), 2 * estimateTextWidth('ab', 12));
  assert.equal(estimateTextWidth('', 12), 0);
});

test('truncateToWidth: unchanged when it fits, ellipsis otherwise', () => {
  const measure = (s) => s.length * 10;
  assert.equal(truncateToWidth('패딩', 100, measure), '패딩');
  assert.equal(truncateToWidth('abcdefghij', 50, measure), 'abcd…');
  assert.equal(truncateToWidth('abcdefghij', 5, measure), '…');
  assert.equal(truncateToWidth('ab cd', 40, measure), 'ab…');
});

test('withUnit: Korean units attach, others after a space, dash untouched', () => {
  assert.equal(withUnit('1,234', '원'), '1,234원');
  assert.equal(withUnit('1,234', 'kg'), '1,234 kg');
  assert.equal(withUnit('12.5', '%'), '12.5%');
  assert.equal(withUnit('1,234', null), '1,234');
  assert.equal(withUnit('1,234', '  '), '1,234');
  assert.equal(withUnit('—', '원'), '—');
});

test('circled numbers and extent', () => {
  assert.equal(circled(1), '①');
  assert.equal(circled(5), '⑤');
  assert.equal(circled(20), '⑳');
  assert.equal(circled(21), '(21)');
  assert.deepEqual(extent([3, null, 1], [NaN, 7], undefined), { min: 1, max: 7 });
  assert.equal(extent([null], []), null);
});
