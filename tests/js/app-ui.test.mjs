// Pure helpers of the SPA components fixed after the web-app review (frontend_app): popover placement (UX-2),
// progress step states (A11Y-6), retry countdown (UX-11), client fixes (FU-1), provisional ranking rows (CON-15),
// upload preview / limits (UX-25, OPS-12), tile sizing (UX-28), outlook parentheticals (UX-28), the Wikipedia
// language of home (FU-1) and the comparison's summary line (FU-15). Node 22 built-in runner, no DOM: every module
// imported here builds nodes only when called.
import { test } from 'node:test';
import assert from 'node:assert/strict';

import { setLang } from '../../trendai/web/static/js/i18n.js';
import { CATALOG } from '../../trendai/web/static/js/catalog.js';
import { panelPosition } from '../../trendai/web/static/js/components/popover.js';
import { mergeStates, stepStates } from '../../trendai/web/static/js/components/progress.js';
import { clientFixes, countdownMessage, otherWikiLang } from '../../trendai/web/static/js/components/errorcard.js';
import { provisionalRow } from '../../trendai/web/static/js/components/rankcards.js';
import { compareSummaryLine } from '../../trendai/web/static/js/components/sharemenu.js';
import { isLongValue } from '../../trendai/web/static/js/components/tiles.js';
import { previewRows, uploadLimits } from '../../trendai/web/static/js/pages/upload.js';
import { initialWikiLang } from '../../trendai/web/static/js/pages/home.js';
import { splitParenthetical } from '../../trendai/web/static/js/ui.js';
import { isTransient, ApiError } from '../../trendai/web/static/js/api.js';
import { DATELINK_HINT_ID, md } from '../../trendai/web/static/js/md.js';

setLang('ko');

// ------------------------------------------------------------------------------------------------ UX-2
test('UX-2: a panel near the right edge keeps its own width and is moved left into the viewport', () => {
  // a 320 px popover opened from a "?" 20 px from the right edge of a 390 px phone
  const pos = panelPosition({ left: 350, top: 600, bottom: 624 }, 320, 120, 390, 844);
  assert.equal(pos.left, 390 - 320 - 8);
  assert.equal(pos.top, 630);
});

test('UX-2: a tall panel flips above the anchor, or is kept on screen when neither side has room', () => {
  assert.deepEqual(panelPosition({ left: 10, top: 700, bottom: 724 }, 300, 200, 390, 844), { left: 10, top: 494 });
  const tall = panelPosition({ left: 10, top: 300, bottom: 324 }, 300, 700, 390, 844);
  assert.equal(tall.top, 844 - 700 - 8, 'never below the viewport (it is position: fixed and cannot be scrolled to)');
});

// ------------------------------------------------------------------------------------------------ A11Y-6
test('A11Y-6: step ② is done once a series has passed it, and never goes back', () => {
  const s1 = stepStates({ stage: 'series', current: { step: 'analyze' }, series_done: 0 });
  assert.deepEqual(s1, ['done', 'active', 'pending', 'pending']);
  const s2 = stepStates({ stage: 'series', current: { step: 'analyze' }, series_done: 1 });
  assert.equal(s2[1], 'done', 'the second series analysing does not reopen step ②');
  assert.deepEqual(mergeStates(['done', 'done', 'active', 'pending'], ['done', 'active', 'pending', 'pending']),
    ['done', 'done', 'active', 'pending']);
  assert.deepEqual(mergeStates(null, s1), s1);
});

// ------------------------------------------------------------------------------------------------ UX-11
test('UX-11: the retry message counts down with the button, and drops the number at 0', () => {
  const msg = '요청이 너무 많아요. 28초 뒤에 다시 시도해 주세요.';
  assert.equal(countdownMessage(msg, 28, 5), '요청이 너무 많아요. 5초 뒤에 다시 시도해 주세요.');
  assert.equal(countdownMessage(msg, 28, 0), '요청이 너무 많아요.');
  assert.equal(countdownMessage('Too many requests. Try again in 28 s.', 28, 0), 'Too many requests.');
  assert.equal(countdownMessage('no number here', 28, 3), 'no number here');
  assert.ok(CATALOG.ko['fix.retry_wait'].includes('{seconds}'));
  assert.ok(!('fix.retry_in_seconds' in CATALOG.ko), 'the "retry now (27 s)" label is gone');
});

// ------------------------------------------------------------------------------------------------ FU-1
test('FU-1: a missing Wikipedia title offers the other wiki language', () => {
  const state = { route: 'explore', src: 'wiki', wl: 'ko', q: [['JavaScript']] };
  const err = { code: 'wiki_not_found' };
  const fixes = clientFixes(err, { state, source: 'wiki' });
  assert.equal(fixes.length, 1);
  assert.equal(fixes[0].action, 'switch_wiki_lang');
  assert.deepEqual(fixes[0].params, { wl: 'en' });
  assert.equal(fixes[0].label, '영어 위키백과에서 찾기');
  assert.equal(clientFixes({ code: 'data_too_short' }, { state, source: 'wiki' }).length, 1, 'a near-empty redirect stub too');
  assert.equal(clientFixes({ code: 'data_too_short' }, { state, source: 'naver' }).length, 0);
  assert.equal(clientFixes(err, { state: { ...state, route: 'result' } }).length, 0);
  assert.equal(otherWikiLang('en'), 'ko');
});

test('FU-1: home starts with the URL wl, else the UI language', () => {
  assert.equal(initialWikiLang('?src=wiki&wl=en', 'ko'), 'en');
  assert.equal(initialWikiLang('?src=wiki', 'en'), 'en');
  assert.equal(initialWikiLang('', 'ko'), 'ko');
  assert.equal(initialWikiLang('?wl=EN!', 'ko'), 'ko');
  for (const lang of ['ko', 'en']) {
    for (const key of ['wiki.lang.ko', 'wiki.lang.en', 'wiki.lang.label', 'source.wiki_lang', 'fix.wiki_lang',
      'search.placeholder.wiki_en', 'home.example_terms.wiki_en']) assert.ok(CATALOG[lang][key], `${lang}.${key}`);
  }
  assert.doesNotMatch(CATALOG.en['home.example_terms.wiki_en'], /[가-힣]/, 'English Wikipedia examples are English titles');
});

// ------------------------------------------------------------------------------------------------ CON-15
test('CON-15: provisional ranking rows mirror the server rows (signed %, 2 × season history)', () => {
  const actual = Array.from({ length: 120 }, (_, i) => i);
  const vm = {
    index: 1, name: 'x', slot: 2, icon: 'diamond',
    phase: { label: 'growing', name: '성장기', icon: 'growing', tentative: false },
    hero: { outlook: { line: '▲ …', tone: 'up' }, basis: '향후 12개월 평균', reliability: { level: 'high', label: '높음' } },
    tiles: [
      { key: 'expected_change', value: '▲45%', value_raw: 45.4, undefined_reason: null },
      { key: 'recent_change', value: '▼0.9%', value_raw: -0.945, sub: '추세 기준 · 전년 동기 대비', undefined_reason: null },
    ],
    chart: { freq: 'M', actual, display: 'normal', forecast: { mean: [1, 2, 3] } },
    flags: {},
  };
  const row = provisionalRow(vm);
  assert.equal(row.expected_change.text, '+45%');
  assert.equal(row.recent_change.text, '−0.9%');
  assert.equal(row.recent_change.basis, '전년 동기 대비');
  assert.equal(row.spark.history.length, 24, 'monthly: min(len, 2 × 12, 104)');
  assert.deepEqual(row.spark.history.slice(-1), [119]);
  const undef = provisionalRow({ ...vm, tiles: [{ key: 'expected_change', value: '—', value_raw: null, undefined_reason: '기준값이 0' }] });
  assert.equal(undef.expected_change.text, '—');
});

// ------------------------------------------------------------------------------------------------ UX-25, OPS-12
test('UX-25: a short file previews every row once, without the ⋯ gap', () => {
  const rows = (n, from = 0) => Array.from({ length: n }, (_, i) => [String(from + i)]);
  const short = previewRows({ head: rows(5), tail: rows(5) }, 5);
  assert.deepEqual(short.rows.map((r) => r[0]), ['0', '1', '2', '3', '4']);
  assert.equal(short.gapAt, null);
  const seven = previewRows({ head: rows(5), tail: rows(5, 2) }, 7);
  assert.deepEqual(seven.rows.map((r) => r[0]), ['0', '1', '2', '3', '4', '5', '6']);
  const long = previewRows({ head: rows(5), tail: rows(5, 55) }, 60);
  assert.equal(long.rows.length, 10);
  assert.equal(long.gapAt, 5);
});

test('OPS-12: the upload limits come from /api/config, with 10 MB / 20 columns as fallback', () => {
  assert.deepEqual(uploadLimits({ sources: { upload: { max_mb: 50, max_series: 5 } } }), { mb: 50, cols: 5, bytes: 50 * 1024 * 1024 });
  assert.deepEqual(uploadLimits(null), { mb: 10, cols: 20, bytes: 10 * 1024 * 1024 });
  for (const key of ['upload.formats', 'err.file_too_large']) assert.match(CATALOG.ko[key], /\{mb\}/, key);
  assert.match(CATALOG.ko['confirm.max_cols'], /\{n\}/);
});

// ------------------------------------------------------------------------------------------------ UX-28
test('UX-28: word tile values and long numbers are set smaller; the outlook parenthetical stays together', () => {
  assert.ok(isLongValue('뚜렷하지 않음'));
  assert.ok(isLongValue('Not clear'));
  assert.ok(!isLongValue('가속'));
  assert.ok(!isLongValue('▲12%'));
  assert.ok(isLongValue('95% 이상 확실'));
  assert.deepEqual(splitParenthetical('상승 가능성 매우 높음 (상승 확률 95% 이상)'), ['상승 가능성 매우 높음 ', '(상승 확률 95% 이상)']);
  assert.deepEqual(splitParenthetical('보합'), ['보합', null]);
});

// ------------------------------------------------------------------------------------------------ FU-15
test('FU-15: a comparison has a one-line summary for 요약 복사', () => {
  const compare = {
    ranking: [
      { rank: 2, name: '패딩', phase: { name: '쇠퇴기' }, outlook: { prob_text: '약 10%' } },
      { rank: 1, name: '선크림', phase: { name: '성장기' }, outlook: { prob_text: '약 95%' } },
    ],
  };
  assert.equal(compareSummaryLine(compare, ['패딩', '선크림']),
    '[trendai] 패딩 vs 선크림: 1위 선크림(성장기, 상승 확률 약 95%) · 2위 패딩(쇠퇴기, 상승 확률 약 10%)');
  assert.equal(compareSummaryLine({ ...compare, summary_line: 'server line' }, []), 'server line');
});

// ------------------------------------------------------------------------------------------------ A11Y-20
test('A11Y-20: proxy 502/503/504 and network failures are transient for polling; a 404 is not', () => {
  assert.ok(isTransient(ApiError.client('client_network', 'err.client_network')));
  assert.ok(isTransient(ApiError.fromObject({ code: 'internal_error' }, 502)));
  assert.ok(isTransient(new ApiError({ code: 'server_busy', status: 503 })));
  assert.ok(!isTransient(new ApiError({ code: 'job_not_found', status: 404 })));
  assert.ok(!isTransient(new Error('x')));
});

// ------------------------------------------------------------------------------------------------ A11Y-21
test('A11Y-21: date links are described by the one page hint', () => {
  const doc = {
    createElement: (n) => ({ nodeName: n.toUpperCase(), attributes: {}, childNodes: [], setAttribute(k, v) { this.attributes[k] = v; }, appendChild(c) { this.childNodes.push(c); } }),
    createTextNode: (t) => ({ data: t }),
    createDocumentFragment: () => ({ childNodes: [], appendChild(c) { this.childNodes.push(c); } }),
  };
  const btn = md('2024.03', { '2024.03': '2024-03-01' }, doc).childNodes[0].childNodes[0];
  assert.equal(btn.attributes['aria-describedby'], DATELINK_HINT_ID);
  assert.ok(CATALOG.ko['a11y.datelink'] && CATALOG.en['a11y.datelink']);
});
