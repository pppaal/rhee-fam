// chart/describe.js (F.7.5) and png.js's pure file-name helper (F.7.6). Run: node --test tests/js
import { test } from 'node:test';
import assert from 'node:assert/strict';

import { describeCompare, describeSeries } from '../../trendai/web/static/js/chart/describe.js';
import { pngFilename } from '../../trendai/web/static/js/chart/png.js';

const CATALOG = {
  ko: { 'chart.describe.compare': '{n}개 시계열 비교: {names}.', 'chart.describe.top': '상승 점수 1위 {name}.' },
  en: { 'chart.describe.compare': 'Comparison of {n} series: {names}.', 'chart.describe.top': 'Highest rising score: {name}.' },
};
const tFor = (lang) => (key, params = {}) => CATALOG[lang][key].replace(/\{(\w+)\}/g, (_, k) => String(params[k]));

test('describeSeries returns chart.desc unchanged', () => {
  const desc = '선크림 주간 상대 지수, 2021–2026. 보통 7월 정점. 향후 26주 평균 +18% 예상.';
  assert.equal(describeSeries({ chart: { desc } }), desc);
  assert.equal(describeSeries({}), '');
});

test('describeCompare: names in request order and the top rising score (F.7.5)', () => {
  const series = [{ index: 2, name: '플리스' }, { index: 0, name: '선크림' }, { index: 1, name: '패딩' }];
  const compare = { ranking: [{ rank: 1, index: 1, name: '패딩' }, { rank: 2, index: 0, name: '선크림' }] };
  assert.equal(describeCompare(compare, series, tFor('ko')), '3개 시계열 비교: 선크림, 패딩, 플리스. 상승 점수 1위 패딩.');
  assert.equal(describeCompare(compare, series, tFor('en')),
    'Comparison of 3 series: 선크림, 패딩, 플리스. Highest rising score: 패딩.');
  assert.equal(describeCompare({ ranking: [] }, series.slice(0, 2), tFor('ko')), '2개 시계열 비교: 선크림, 플리스.');
  assert.equal(describeCompare(null, series.slice(1, 2), tFor('ko')), '1개 시계열 비교: 선크림.');
});

test('pngFilename: KST date and file-name-safe characters', () => {
  assert.equal(pngFilename('선크림', new Date('2026-11-25T03:00:00Z')), 'trendai-선크림-20261125.png');
  // 15:00 UTC is already the next day in KST
  assert.equal(pngFilename('패딩 vs 플리스', new Date('2026-11-24T15:00:00Z')), 'trendai-패딩 vs 플리스-20261125.png');
  assert.equal(pngFilename('a/b\\c:d*e?f"g<h>i|j\u0001', new Date('2026-01-02T00:00:00Z')),
    'trendai-a_b_c_d_e_f_g_h_i_j_-20260102.png');
  assert.equal(pngFilename('', new Date('2026-01-02T00:00:00Z')), 'trendai-chart-20260102.png');
});
