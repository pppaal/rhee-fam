// Browser end-to-end scenarios (spec H.4.3): 14 scenarios plus 9b, each an independent test with its own server
// (fresh data dir and rate limits) and browser context, Chromium, hl=ko unless stated. One extra scenario covers the
// Google Trends upload guide path (fixture gtrends.csv, H.4.1).
//
// Run: NODE_PATH="$(npm root -g)" node --test --test-concurrency=1 tests/e2e   (or through tests/e2e/test_e2e.py)

import assert from 'node:assert/strict';
import { after, before, test } from 'node:test';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

import { FIXTURES, STATIC_JS, download, jobLines, newPage, nextResult, playwright, withServer } from './helpers.mjs';

const { CATALOG } = await import(pathToFileURL(path.join(STATIC_JS, 'catalog.js')).href);
const fmt = await import(pathToFileURL(path.join(STATIC_JS, 'fmt.js')).href);
const KO = CATALOG.ko;
const SALES = path.join(FIXTURES, 'sales.csv');
const GTRENDS = path.join(FIXTURES, 'gtrends.csv');
const HANGUL = /[가-힣]/;
const T = { timeout: 120_000 };

let browser;
before(async () => { browser = await playwright().chromium.launch(); });
after(async () => { if (browser) await browser.close(); });

const q = (...terms) => terms.map(encodeURIComponent).join(',');
const text = async (page, selector) => ((await page.textContent(selector)) || '').trim();

async function uploadSales(page, url) {
  await page.goto(`${url}/upload`);
  await page.setInputFiles('#upload-file', SALES);
  await page.waitForSelector('#confirm-date');
}

// ------------------------------------------------------------------------------------------------ 1
test('1 home → sample detail', T, () => withServer({}, async ({ url }) => {
  const { context, page, errors } = await newPage(browser);
  await page.goto(`${url}/`);
  await page.click('a.chip:has-text("성장형")');
  await page.waitForSelector('.hero .phase-badge');
  assert.equal(new URL(page.url()).pathname + new URL(page.url()).search, '/explore?src=sample&q=growing');
  assert.match(await text(page, '.hero .phase-badge'), /성장기/);
  assert.equal(await page.locator('.tile').count(), 6);
  const desc = await page.$eval('[role=img] desc', (d) => d.textContent.trim());
  assert.ok(desc.length > 0, 'chart <desc> is empty');
  const badge = page.locator('.pill-synthetic').first();
  assert.ok(await badge.isVisible());
  assert.match(await badge.textContent(), /가상 데이터/);
  assert.match(await text(page, '.source-line'), /가상 데이터/);
  assert.deepEqual(errors, []);
  await context.close();
}));

// ------------------------------------------------------------------------------------------------ 2
test('2 compare 3 samples', T, () => withServer({}, async ({ url }) => {
  const { context, page, errors, posts } = await newPage(browser);
  await page.goto(`${url}/samples`);
  await page.waitForSelector('input[value=emerging]');
  for (const name of ['emerging', 'fad', 'seasonal']) await page.check(`input[value=${name}]`);
  await page.click('button:has-text("선택한 샘플 비교 (3)")');
  await page.waitForSelector('.rank-card .rank-num');

  const chips = page.locator('.chart-legend-chip');
  assert.equal(await chips.count(), 3);
  const icons = await chips.evaluateAll((els) => els.map((el) => el.querySelector('svg.chart-glyph path').getAttribute('d')));
  assert.equal(new Set(icons).size, 3, 'legend chips need three distinct slot icons');
  const ranks = await page.locator('.rank-card .rank-num').allTextContents();
  assert.deepEqual(ranks.map((r) => r.trim().replace(/위$/, '')), ['1', '2', '3']);   // rendered 1위 2위 3위

  const postsBefore = posts.length;
  await page.click('button:has-text("표로 보기")');
  await page.waitForURL(/[?&]view=table/);
  await page.waitForSelector('.ranking-table tbody tr');
  assert.equal(await page.locator('.ranking-table tbody tr').count(), 3);
  await page.click('button:has-text("표로 보기")');                       // back to the chart
  await page.waitForSelector('.chart-legend-chip');
  assert.ok(!new URL(page.url()).searchParams.has('view'));

  await page.locator('.rank-card').first().locator('.more-link').click();
  await page.waitForSelector('.hero');
  assert.ok(new URL(page.url()).searchParams.has('focus'), 'focus= missing');
  await page.goBack();
  await page.waitForSelector('.rank-card .rank-num');
  assert.equal(await page.locator('.rank-card').count(), 3);
  assert.equal(posts.length, postsBefore, 'browser back must not POST /api/analyze again');
  assert.deepEqual(errors, []);
  await context.close();
}));

// ------------------------------------------------------------------------------------------------ 3
test('3 upload CSV → confirm → results', T, () => withServer({}, async ({ url }) => {
  const { context, page, errors } = await newPage(browser);
  await uploadSales(page, url);
  assert.equal(await page.$eval('#confirm-date', (s) => s.value), '날짜');
  const rowNumber = page.locator('input[value="번호"]');
  assert.equal(await rowNumber.isChecked(), false);
  const confirm = await text(page, '.confirm');
  assert.match(confirm, /순번으로 보여요/);
  assert.match(confirm, /숫자가 아니에요/);
  assert.equal(await page.locator('input[value="메모"]').isChecked().catch(() => false), false);
  await page.uncheck('input[value="방문수"]');
  await page.click('button:has-text("분석하기")');
  await page.waitForURL(/\/r\/[A-Za-z0-9_-]{22}$/);
  await page.waitForSelector('.hero');
  assert.match(await text(page, 'h1'), /매출/);
  assert.equal(await page.locator('.rank-card').count(), 0, 'one value column gives the detail view');
  await page.goto(`${url}/`);
  await page.waitForSelector('.recent-list');
  assert.match(await text(page, '.recent-list'), /sales\.csv/);
  assert.deepEqual(errors, []);
  await context.close();
}));

// ------------------------------------------------------------------------------------------------ 4
test('4 Wikipedia with the fake fetcher', T, () => withServer({}, async ({ url }) => {
  const { context, page, errors } = await newPage(browser);
  await page.goto(`${url}/explore?src=wiki&q=${q('인공지능')}`);
  await page.waitForSelector('.hero');
  assert.match(await text(page, '.source-line'), /Wikimedia Pageviews/);

  await page.goto(`${url}/explore?src=wiki&q=${q('인공지능', '없는문서')}`);
  await page.waitForSelector('.error-card');
  assert.equal(await page.locator('.rank-card:not(.error-card)').count(), 1);
  const card = page.locator('.error-card');
  assert.equal(await card.count(), 1);
  assert.match(await card.textContent(), /위키백과에 ‘없는문서’ 문서가 없어요/);
  const search = card.locator('a:has-text("위키백과에서 제목 찾기")');
  assert.match(await search.getAttribute('href'), /^https:\/\/ko\.wikipedia\.org\/w\/index\.php\?search=/);
  assert.equal(await card.locator('button:has-text("네이버로 보기")').count(), 1);
  assert.deepEqual(errors, []);
  await context.close();
}));

// ------------------------------------------------------------------------------------------------ 5
test('5 Naver unconfigured', T, () => withServer({ naver: false }, async ({ url }) => {
  const { context, page, errors, posts } = await newPage(browser);
  await page.goto(`${url}/`);
  const naver = page.locator('[data-source=naver]');
  await naver.waitFor();
  assert.equal(await naver.getAttribute('aria-disabled'), 'true');
  assert.match(await naver.textContent(), /준비 중/);
  await page.click('.source-help .help-btn');
  assert.equal(await text(page, '.help-pop'), KO['source.naver.unconfigured']);
  const firstChip = page.locator('.examples a.chip').first();
  assert.equal((await firstChip.textContent()).trim(), '자외선 차단제');
  await firstChip.click();
  await page.waitForSelector('.hero');
  assert.match(await text(page, '.source-line'), /Wikimedia Pageviews/);

  await page.goto(`${url}/explore?src=naver&q=${q('선크림')}`);
  await page.waitForSelector('.error-card');
  const card = page.locator('.error-card');
  assert.match(await card.textContent(), /네이버 검색어 트렌드는 아직 연결되지 않았어요\./);
  assert.equal(await card.locator('button:has-text("위키백과로 보기")').count(), 1);
  assert.equal(await card.locator(':is(a, button):has-text("파일 올리기")').count(), 1);
  const postsBefore = posts.length;
  await card.locator('button:has-text("위키백과로 보기")').click();
  await page.waitForSelector('.home-title');
  const at = new URL(page.url());
  assert.equal(at.pathname, '/');
  assert.equal(at.searchParams.get('src'), 'wiki');
  assert.equal(at.searchParams.get('q'), '선크림');
  assert.equal(await page.getAttribute('[data-source=wiki]', 'aria-checked'), 'true');
  assert.deepEqual((await page.locator('.search-chips li').allTextContents()).map((s) => s.replace(/\s*(삭제|×|✕).*$/, '').trim()).filter(Boolean).length, 1);
  assert.match(await text(page, '.search-chips'), /선크림/);
  assert.ok(await page.locator('.search-hint', { hasText: KO['search.hint.wiki_title'] }).isVisible());
  assert.equal(posts.length, postsBefore, 'use_wiki must pre-fill the search box, not run an analysis');
  assert.deepEqual(errors, []);
  await context.close();
}));

// ------------------------------------------------------------------------------------------------ 6
test('6 Naver configured compare', T, () => withServer({ naver: true }, async (server) => {
  const { context, page, errors } = await newPage(browser);
  // range=3y (not the default) so that the reanalyze_single URL has a range to keep
  await page.goto(`${server.url}/explore?src=naver&q=${q('선크림', '패딩', '저검색어')}&range=3y`);
  await page.waitForSelector('.rank-card .rank-num');
  assert.equal(await page.locator('.rank-card .rank-num').count(), 3);

  // exactly one fresh Naver call for the whole comparison (the job line is written when the job ends)
  for (let i = 0; i < 40 && jobLines(server, /kind=analyze/).length === 0; i += 1) await page.waitForTimeout(100);
  assert.equal(jobLines(server, /kind=analyze .*naver=fresh/).length, 1, server.stderr());

  assert.ok(await page.locator('.compare-card .source-line').first().isVisible());
  for (const line of await page.locator('.rank-outlook').allTextContents()) assert.match(line, /상승 확률/);
  const index = page.locator('[data-scale=index]');
  assert.ok((await index.getAttribute('aria-disabled')) === 'true' || (await index.isDisabled()));

  const low = page.locator('.rank-card', { hasText: '저검색어' });
  assert.match(await low.textContent(), /다른 검색어보다 검색량이 매우 적어 해상도가 낮아요\./);
  for (const other of ['선크림', '패딩']) {
    assert.doesNotMatch(await page.locator('.rank-card', { hasText: other }).textContent(), /해상도가 낮아요/);
  }
  assert.equal(await low.locator('button:has-text("이 검색어만 다시 분석")').count(), 1);

  await page.click('button:has-text("표로 보기")');
  await page.waitForSelector('.values-table');
  assert.equal(await page.locator('.values-table :is(a, button):has-text("CSV로 받기")').count(), 0);
  assert.match(await text(page, '.values-table'), /네이버 원본 데이터는/);
  await page.click('button:has-text("표로 보기")');

  await page.locator('.rank-card', { hasText: '선크림' }).locator('.more-link').click();
  await page.waitForSelector('.hero');
  const ai = page.locator('#narrative button:has-text("AI로 다시 쓰기")');
  assert.ok(await ai.isDisabled());
  assert.match(await text(page, '#narrative'), /네이버 데이터는 이용약관에 따라 AI로 보내지 않아요\./);
  await page.click('.view-actions button:has-text("내보내기")');
  const html = page.locator('.menu .menu-item', { hasText: 'HTML 리포트' });
  assert.equal(await html.getAttribute('aria-disabled'), 'true');
  await page.keyboard.press('Escape');

  await page.goBack();
  await page.waitForSelector('.rank-card .rank-num');
  await page.locator('.rank-card', { hasText: '저검색어' }).locator('button:has-text("이 검색어만 다시 분석")').click();
  await page.waitForURL((u) => new URL(u).searchParams.get('q') === '저검색어');
  const re = new URL(page.url());
  assert.equal(re.searchParams.get('src'), 'naver');
  assert.equal(re.searchParams.get('range'), '3y');
  assert.equal(re.searchParams.get('unit'), null);                        // unit was the default and stays so
  await page.waitForSelector('.hero');
  assert.deepEqual(errors, []);
  await context.close();
}));

// ------------------------------------------------------------------------------------------------ 7
test('7 share link', T, () => withServer({}, async ({ url }) => {
  const first = await newPage(browser);
  const { page } = first;
  await uploadSales(page, url);
  await page.uncheck('input[value="방문수"]');
  await page.click('button:has-text("분석하기")');
  await page.waitForSelector('.hero');
  const sentence = await text(page, '.hero-sentence');
  await page.click('.view-actions button:has-text("공유")');
  await page.waitForSelector('#share-url');
  const shareUrl = await page.inputValue('#share-url');
  assert.match(shareUrl, /^http:\/\/127\.0\.0\.1:\d+\/s\/[A-Za-z0-9_-]{22}$/);

  const second = await newPage(browser);
  await second.page.goto(shareUrl);
  await second.page.waitForSelector('.hero');
  assert.match(await text(second.page, '.banner-readonly'), /읽기 전용 공유 링크/);
  assert.equal(await text(second.page, '.hero-sentence'), sentence);
  assert.equal(await second.page.locator('.view-actions button:has-text("공유")').count(), 0);
  assert.equal(await second.page.locator('button:has-text("AI로 다시 쓰기")').count(), 0);

  await page.click('button:has-text("링크 삭제")');
  await page.click('.sheet-small .btn-danger');
  await page.waitForSelector('#share-url', { state: 'detached' }).catch(() => {});
  await second.page.waitForTimeout(300);
  await second.page.reload();
  await second.page.waitForSelector('.error-card');
  assert.match(await text(second.page, '.error-card'), /공유 링크가 만료됐거나 삭제됐어요\./);
  assert.deepEqual(first.errors, []);
  assert.deepEqual(second.errors, []);
  await second.context.close();
  await first.context.close();
}));

// ------------------------------------------------------------------------------------------------ 8
function csvLines(bytes) {
  assert.deepEqual([...bytes.subarray(0, 3)], [0xef, 0xbb, 0xbf], 'CSV must start with a UTF-8 BOM');
  return bytes.subarray(3).toString('utf8').split('\r\n');
}

test('8 export CSV', T, () => withServer({ naver: true }, async ({ url }) => {
  const { context, page, errors } = await newPage(browser);
  await page.goto(`${url}/explore?src=sample&q=growing`);
  await page.waitForSelector('.hero');
  await page.click('.view-actions button:has-text("내보내기")');
  const sample = await download(page, () => page.click('.menu a:has-text("CSV")'));
  const lines = csvLines(sample.bytes);
  assert.ok(lines[0].startsWith('# ') && lines[1].startsWith('# '));
  assert.ok(lines[2].startsWith('날짜,실제값,추세,예측값'), lines[2]);
  const rows = lines.slice(3).filter(Boolean).map((l) => l.split(','));
  assert.ok(rows.some((r) => r[1] === '' && r[3] !== ''), 'a forecast row with an empty 실제값 is missing');

  await page.goto(`${url}/explore?src=naver&q=${q('선크림')}`);
  await page.waitForSelector('.hero');
  await page.click('.view-actions button:has-text("내보내기")');
  const naver = await download(page, () => page.click('.menu a:has-text("CSV")'));
  const header = csvLines(naver.bytes)[2];
  assert.ok(header.startsWith('날짜,추세,예측값'), header);
  assert.ok(!header.includes('실제값'));
  assert.deepEqual(errors, []);
  await context.close();
}));

// ------------------------------------------------------------------------------------------------ 9
test('9 mobile 390 px, no horizontal scroll', T, () => withServer({}, async ({ url }) => {
  const { context, page, errors } = await newPage(browser, { viewport: { width: 390, height: 844 }, isMobile: true, hasTouch: true });
  const pages = [['/', '.home-title'], ['/explore?src=sample&q=growing', '.hero'],
    [`/explore?src=sample&q=${q('emerging', 'fad', 'seasonal')}`, '.rank-card .rank-num'], ['/upload', '#upload-file'],
    ['/method', 'h1']];
  for (const [p, ready] of pages) {
    await page.goto(url + p);
    await page.waitForSelector(ready, { state: 'attached' });
    await page.waitForTimeout(300);
    const width = await page.evaluate(() => document.documentElement.scrollWidth);
    assert.ok(width <= 390, `${p}: scrollWidth ${width}`);
  }

  await page.goto(`${url}/explore?src=sample&q=growing`);
  await page.waitForSelector('.tile');
  const tops = await page.locator('.tile').evaluateAll((els) => els.map((e) => e.offsetTop));
  assert.equal(tops[0], tops[1], 'tiles 1 and 2 share a row');
  assert.notEqual(tops[1], tops[2], 'tile 3 starts the second row');

  await page.goto(`${url}/explore?src=wiki&q=${q('인공지능', '기계학습')}`);
  await page.waitForSelector('.rank-card .rank-num');
  const range = page.locator('.qb-full [data-chip=range]');
  await range.tap();
  await page.waitForSelector('[role=dialog]');
  await page.keyboard.press('Escape');
  await page.waitForSelector('[role=dialog]', { state: 'detached' });
  assert.equal(await page.evaluate(() => document.activeElement && document.activeElement.dataset.chip), 'range');

  await page.evaluate(() => window.scrollTo(0, 600));
  await page.waitForTimeout(400);
  const height = await page.$eval('.querybar', (e) => e.getBoundingClientRect().height);
  assert.ok(height <= 56, `collapsed query bar is ${height}px`);
  await page.tap('.qb-summary');
  await page.waitForSelector('.querybar-sheet[role=dialog], [role=dialog] .querybar-sheet, .querybar-sheet');
  assert.ok(await page.locator('[role=dialog]').first().isVisible());
  assert.deepEqual(errors, []);
  await context.close();
}));

// ------------------------------------------------------------------------------------------------ 9b
test('9b Korean IME in the search box', T, () => withServer({}, async ({ url }) => {
  const { context, page, errors } = await newPage(browser);
  await page.goto(`${url}/`);
  await page.focus('.search-input');
  await page.fill('.search-input', '선크');
  await page.evaluate(() => {
    const input = document.querySelector('.search-input');
    input.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', isComposing: true, bubbles: true, cancelable: true }));
    input.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', keyCode: 229, bubbles: true, cancelable: true }));
  });
  assert.equal(await page.locator('.search-chips li').count(), 0, 'a composing Enter must not create a chip');
  await page.fill('.search-input', '선크림');
  await page.press('.search-input', 'Enter');
  const chips = await page.locator('.search-chips li').allTextContents();
  assert.equal(chips.length, 1);
  assert.match(chips[0], /선크림/);
  assert.deepEqual(errors, []);
  await context.close();
}));

// ------------------------------------------------------------------------------------------------ 10
test('10 dark mode', T, () => withServer({}, async ({ url }) => {
  const { context, page, errors } = await newPage(browser, { colorScheme: 'dark' });
  const background = () => page.evaluate(() => getComputedStyle(document.body).backgroundColor);
  await page.goto(`${url}/`);
  await page.waitForSelector('.home-title');
  assert.equal(await background(), 'rgb(14, 14, 13)');
  await page.click('.footer-links button');
  await page.click('label:has-text("라이트")');
  assert.equal(await background(), 'rgb(246, 246, 243)');
  await page.reload();
  await page.waitForSelector('.home-title');
  assert.equal(await background(), 'rgb(246, 246, 243)');
  assert.equal(await page.evaluate(() => localStorage.getItem('trendai.theme')), 'light');
  assert.deepEqual(errors, []);
  await context.close();
}));

// ------------------------------------------------------------------------------------------------ 11
test('11 keyboard crosshair', T, () => withServer({}, async ({ url }) => {
  const { context, page, errors } = await newPage(browser);
  const result = nextResult(page);
  await page.goto(`${url}/explore?src=sample&q=growing`);
  const chart = (await result).series[0].chart;
  const label = (iso) => fmt.dateLabel(iso, chart.freq, 'ko');
  await page.waitForSelector('.chart-wrap');

  let focused = false;
  for (let i = 0; i < 120 && !focused; i += 1) {
    await page.keyboard.press('Tab');
    focused = await page.evaluate(() => document.activeElement && document.activeElement.getAttribute('role') === 'group'
      && document.activeElement.classList.contains('chart-wrap'));
  }
  assert.ok(focused, 'Tab never reached the chart wrapper');
  const readout = () => text(page, '.chart-fan .chart-readout-date');
  assert.equal(await readout(), label(chart.dates.at(-1)));

  for (let i = 0; i < 3; i += 1) await page.keyboard.press('ArrowLeft');
  const moved = await readout();
  assert.notEqual(moved, label(chart.dates.at(-1)));
  assert.ok(chart.dates.slice(-4, -1).map(label).includes(moved), `read-out after 3 × ArrowLeft: ${moved}`);
  await page.waitForFunction((d) => (document.getElementById('live-polite') || {}).textContent?.includes(d), moved);

  await page.keyboard.press('Home');
  assert.equal(await readout(), label(chart.dates[0]));
  await page.keyboard.press('End');
  assert.equal(await readout(), label(chart.forecast.dates.at(-1)));
  await page.keyboard.press('Escape');
  assert.equal(await page.locator('.chart-fan .chart-xh-line').first().getAttribute('visibility'), 'hidden');
  assert.ok(await page.locator('.chart-fan .chart-readout.is-idle').count() === 1);
  assert.deepEqual(errors, []);
  await context.close();
}));

// ------------------------------------------------------------------------------------------------ 12
test('12 progress and cancel', T, () => withServer({}, async ({ url }) => {
  const { context, page, errors } = await newPage(browser);
  const posted = page.waitForResponse((r) => new URL(r.url()).pathname === '/api/analyze' && r.request().method() === 'POST');
  await page.goto(`${url}/explore?src=wiki&q=${q('느린문서', '인공지능', '기계학습')}`);
  const job = await (await posted).json();
  await page.waitForSelector('.progress-steps[role=status]:not([hidden])', { timeout: 5000 });
  assert.equal(await page.locator('.progress-steps[role=status] li').count(), 4);
  await page.waitForSelector('.progress-cancel:not([hidden])');
  await page.click('.progress-cancel');
  await page.waitForSelector('.empty-state');
  assert.match(await text(page, '.empty-state'), /분석을 취소했어요\./);
  assert.ok(await page.locator('.empty-state button:has-text("다시 분석")').isVisible());
  let status = null;
  for (let i = 0; i < 50 && status !== 'cancelled'; i += 1) {
    const res = await fetch(`${url}/api/jobs/${job.job_id}`);
    if (res.status === 200) status = (await res.json()).status;
    if (status !== 'cancelled') await new Promise((r) => setTimeout(r, 100));
  }
  assert.equal(status, 'cancelled');
  assert.deepEqual(errors, []);
  await context.close();
}));

// ------------------------------------------------------------------------------------------------ 13
test('13 English UI', T, () => withServer({}, async ({ url }) => {
  const { context, page, errors } = await newPage(browser);
  await page.goto(`${url}/explore?src=sample&q=fad&hl=en`);
  await page.waitForSelector('.hero');
  assert.equal(await page.evaluate(() => document.documentElement.lang), 'en');
  for (const region of ['main', 'header', 'footer']) {
    const content = await page.evaluate((sel) => {
      const el = document.querySelector(sel);
      if (!el) return '';
      const clone = el.cloneNode(true);
      // the language switch names each language in its own script (한국어 / English): not UI copy
      for (const node of clone.querySelectorAll('[lang]:not([lang=en])')) node.remove();
      return clone.innerText || clone.textContent;
    }, region);
    const hangul = content.match(/[가-힣]+/g);   // the sample is named 'Fad' in English too (E.6.1)
    assert.equal(hangul, null, `${region} contains Hangul: ${(hangul || []).slice(0, 8).join(' ')}`);
  }
  assert.equal(await page.evaluate(() => document.documentElement.dataset.updown), 'intl');
  assert.ok(!HANGUL.test(await page.title()));
  assert.deepEqual(errors, []);
  await context.close();
}));

// ------------------------------------------------------------------------------------------------ 14
test('14 AI rewrite (non-Naver)', T, () => withServer({ ai: true }, async ({ url }) => {
  const { context, page, errors } = await newPage(browser);
  await page.goto(`${url}/explore?src=sample&q=growing`);
  // the progressive view draws the card from the poll VM with the AI button disabled (no result id yet); the final
  // render replaces the card, so observe the status line of the final card only
  await page.waitForSelector('#narrative button:has-text("AI로 다시 쓰기"):not([disabled])');
  await page.evaluate(() => {
    window.__statusTexts = [];
    const status = document.querySelector('#narrative .narrative-status');
    new MutationObserver(() => window.__statusTexts.push(status.textContent))
      .observe(status, { childList: true, characterData: true, subtree: true });
  });
  await page.click('#narrative button:has-text("AI로 다시 쓰기")');
  await page.waitForSelector('#narrative .pill-ai', { timeout: 20_000 });
  const statuses = await page.evaluate(() => window.__statusTexts);
  assert.ok(statuses.includes(KO['detail.narrative.ai_running']), `status line never shown: ${statuses}`);
  assert.match(await text(page, '#narrative .pill-ai'), /AI 해석 · Claude/);
  assert.match(await text(page, '#narrative'), /AI 테스트 해석/);
  await page.click(`#narrative button:has-text("${KO['detail.narrative.show_rule']}")`);
  assert.match(await text(page, '#narrative .narrative-badge'), /규칙 기반 해석/);
  assert.doesNotMatch(await text(page, '#narrative .narrative-body'), /AI 테스트 해석/);
  assert.deepEqual(errors, []);
  await context.close();
}));

// ------------------------------------------------------------------------------------------------ extra
test('extra: Google Trends CSV upload', T, () => withServer({}, async ({ url }) => {
  const { context, page, errors } = await newPage(browser);
  await page.goto(`${url}/upload`);
  await page.setInputFiles('#upload-file', GTRENDS);
  await page.waitForSelector('#confirm-date');
  const confirm = await text(page, '.confirm');
  assert.match(confirm, /Google 트렌드/);
  assert.match(confirm, /<1/);
  await page.click('button:has-text("분석하기")');
  await page.waitForURL(/\/r\/[A-Za-z0-9_-]{22}$/);
  await page.waitForSelector('.rank-card .rank-num');                 // two terms → comparison
  assert.match(await text(page, '.compare-card .source-line'), /Google 트렌드/);
  const index = page.locator('[data-scale=index]');
  assert.ok((await index.getAttribute('aria-disabled')) === 'true' || (await index.isDisabled()));
  assert.deepEqual(errors, []);
  await context.close();
}));
