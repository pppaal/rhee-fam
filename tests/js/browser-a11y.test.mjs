// Browser regression tests of the SPA fixes from the web-app review (frontend_app): keyboard focus kept across
// re-renders (A11Y-2, A11Y-4, A11Y-14), menus and help popovers (A11Y-7, A11Y-9), overlays closed on Back (A11Y-10),
// the theme button (A11Y-11), announcements (A11Y-3, A11Y-6, A11Y-12), layout (UX-2, UX-3, UX-18), sample chips
// (UX-8), the AI default (UX-9), unknown links (UX-10), the Wikipedia language (FU-1), all-failed comparisons (FU-2)
// and tab close (OPS-7).
//
// They drive the real app (tests/e2e/server.py: fake Naver / Wikipedia / LLM, no network) in Chromium through the
// globally installed Playwright. When Playwright or its browser is missing the tests are skipped, so
// `node --test tests/js` still works anywhere; TRENDAI_SKIP_BROWSER=1 skips them on purpose.

import { after, before, test } from 'node:test';
import assert from 'node:assert/strict';
import { execSync } from 'node:child_process';
import { createRequire } from 'node:module';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const SALES = path.join(HERE, '..', 'e2e', 'fixtures', 'sales.csv');

function loadPlaywright() {
  if (process.env.TRENDAI_SKIP_BROWSER) return null;
  const require = createRequire(import.meta.url);
  try {
    return require('playwright');
  } catch {
    try {
      const root = execSync('npm root -g', { encoding: 'utf8', stdio: ['ignore', 'pipe', 'ignore'] }).trim();
      return createRequire(path.join(root, 'noop.js'))('playwright');
    } catch {
      return null;
    }
  }
}

const pw = loadPlaywright();
let browser = null;
let server = null;          // one fresh server per test (own rate limits, like tests/e2e)
let startServer = null;
let skip = pw ? false : 'Playwright not found (npm root -g) or TRENDAI_SKIP_BROWSER set';

before(async () => {
  if (skip) return;
  try {
    if (!process.env.TRENDAI_E2E_PYTHON) process.env.TRENDAI_E2E_PYTHON = 'python3';
    ({ startServer } = await import('../e2e/helpers.mjs'));
    browser = await pw.chromium.launch();
    const probe = await startServer({ naver: true, ai: true });
    await probe.stop();
  } catch (err) {
    skip = `browser or server unavailable: ${err.message.split('\n')[0]}`;
  }
});
after(async () => {
  if (browser) await browser.close();
});

const T = { timeout: 90_000 };
const q = (...terms) => terms.map(encodeURIComponent).join(',');

async function page(opts = {}) {
  const context = await browser.newContext({
    viewport: { width: 1280, height: 900 }, locale: 'ko-KR', timezoneId: 'Asia/Seoul', ...opts,
  });
  const p = await context.newPage();
  const errors = [];
  p.on('pageerror', (e) => errors.push(e.message));
  return { context, p, errors };
}
/** Wait for the final result (exports and share become available with it; the progressive view comes earlier). */
const ready = (p) => p.waitForSelector('.view-actions button:has-text("내보내기"):not([disabled])');
const active = (p) => p.evaluate(() => {
  const a = document.activeElement;
  return a ? { tag: a.tagName, cls: a.className, text: (a.getAttribute('aria-label') || a.textContent || '').trim() } : null;
});

/** Run a browser test, or skip it with the reason. */
function btest(name, fn) {
  test(name, T, async (t) => {
    if (skip) {
      t.skip(skip);
      return;
    }
    server = await startServer({ naver: true, ai: true });
    try {
      await fn();
    } finally {
      await server.stop();
      server = null;
    }
  });
}

btest('A11Y-2: focus inside the results survives progressive re-renders; a chip popover survives the result', async () => {
  const { context, p, errors } = await page();
  await p.goto(`${server.url}/explore?src=naver&q=${q('감자', '고구마', '옥수수', '호박')}&range=all&unit=date`);
  await p.waitForSelector('.qb-full [data-chip=range]');
  await p.locator('.qb-full [data-chip=range]').focus();
  await p.keyboard.press('Enter');
  await p.waitForSelector('.popover');
  await p.waitForSelector('.rank-num', { timeout: 30_000 });
  assert.equal(await p.locator('.popover').count(), 1, 'the open chip popover is still anchored to a live chip');
  await p.keyboard.press('Escape');
  assert.equal((await active(p)).text, '기간: 전체(2016년부터)');

  await p.goto(`${server.url}/explore?src=naver&q=${q('선크림', '패딩', '수능', '김장', '다이어트')}&range=all&unit=date`);
  await p.waitForSelector('.segment[data-scale=raw]');
  await p.locator('.segment[data-scale=raw]').focus();
  const seen = new Set();
  for (let i = 0; i < 80 && !(await p.locator('.rank-num').count()); i += 1) {
    seen.add((await active(p)).tag);
    await p.waitForTimeout(100);
  }
  await p.waitForTimeout(300);
  seen.add((await active(p)).tag);
  assert.deepEqual([...seen], ['BUTTON'], 'focus never fell to <body>');
  assert.match(await p.evaluate(() => document.getElementById('live-polite').textContent), /분석을 마쳤어요/, 'completion announced (A11Y-6)');
  assert.deepEqual(errors, []);
  await context.close();
});

btest('A11Y-2 / A11Y-6: 취소 moves focus to the cancelled message; the steps are updated in place', async () => {
  const { context, p } = await page();
  await p.goto(`${server.url}/explore?src=wiki&q=${q('느린문서', '인공지능', '기계학습')}`);
  await p.waitForSelector('.progress-steps[role=status]:not([hidden])');
  assert.equal(await p.locator('.progress-steps[role=status] li').count(), 4);
  assert.equal(await p.locator('.progress-steps ol').count(), 1, 'the steps stay a list');
  const first = await p.$('.progress-steps li');
  await p.waitForTimeout(1200);
  assert.ok(await first.evaluate((el) => el.isConnected), 'polls do not rebuild the steps');
  await p.locator('.progress-cancel').focus();
  await p.keyboard.press('Enter');
  await p.waitForSelector('.empty-state');
  assert.equal((await active(p)).text, '분석을 취소했어요.');
  await context.close();
});

btest('A11Y-7 / A11Y-9: menus return focus on Tab; help popovers do not capture Esc after leaving', async () => {
  const { context, p } = await page();
  await p.goto(`${server.url}/explore?src=sample&q=growing`);
  await p.waitForSelector('.chart-wrap');
  // exports exist once the final result is in (the progressive view shows the chart a little earlier)
  await ready(p);
  const exportBtn = p.locator('.view-actions button:has-text("내보내기")');
  assert.equal(await exportBtn.getAttribute('aria-controls'), null, 'no aria-controls while closed');
  await exportBtn.focus();
  await p.keyboard.press('Enter');
  const labelled = await p.$eval('.menu', (m) => Boolean(document.getElementById(m.getAttribute('aria-labelledby'))));
  assert.ok(labelled, 'the menu is labelled by its button');
  const outline = await p.$eval('.menu-item:focus', (el) => getComputedStyle(el).outlineStyle);
  assert.equal(outline, 'solid', 'the focused menu item has a ring');
  await p.keyboard.press('Shift+Tab');
  assert.equal((await active(p)).text, '공유', 'Shift+Tab continues from the menu button');

  await p.locator('#now .help-btn').focus();
  await p.keyboard.press('Enter');
  await p.locator('#now .more-link').focus();
  assert.equal(await p.locator('#now .help-pop:not([hidden])').count(), 0, 'closes when focus leaves');
  await p.locator('#now .help-btn').focus();
  await p.keyboard.press('Enter');
  await p.locator('#now .more-link').click();
  await p.waitForURL(/\/method/);
  await p.goBack();
  await p.waitForSelector('.chart-fan .chart-wrap');
  await p.locator('.chart-fan .chart-wrap').focus();
  await p.keyboard.press('ArrowLeft');
  await p.keyboard.press('Escape');
  assert.equal(await p.locator('.chart-fan .chart-readout.is-idle').count(), 1, 'one Esc reaches the chart');
  await context.close();
});

btest('A11Y-10 / A11Y-11: Back closes an open dialog; a theme chosen in 설정 updates the top bar', async () => {
  const { context, p } = await page();
  await p.goto(`${server.url}/`);
  await p.waitForSelector('.home-title');
  await p.click('.topnav a[href="/method"]');
  await p.waitForSelector('.info-page');
  await p.click('.footer-links button');
  await p.waitForSelector('[role=dialog]');
  await p.goBack();
  await p.waitForSelector('.home-title');
  assert.equal(await p.locator('[role=dialog]').count(), 0);
  assert.equal(await p.evaluate(() => document.documentElement.classList.contains('has-modal')), false);
  await p.click('.footer-links button');
  await p.click('label:has-text("다크")');
  assert.match(await p.$eval('#topbar .theme-btn', (b) => b.getAttribute('aria-label')), /다크/);
  assert.deepEqual(await p.$$eval('.lang-btn', (els) => els.map((e) => e.getAttribute('aria-label'))), ['KO 한국어', 'EN English']);
  await context.close();
});

btest('A11Y-4: + 비교 Esc, the copy fallback and the mobile menu keep focus', async () => {
  const { context, p } = await page();
  await p.goto(`${server.url}/explore?src=wiki&q=${q('김장')}`);
  await ready(p);
  await p.locator('.chip-add').focus();
  await p.keyboard.press('Enter');
  await p.keyboard.press('Escape');
  assert.equal((await active(p)).text, '+ 비교');
  await p.evaluate(() => Object.defineProperty(navigator, 'clipboard', { value: { writeText: () => Promise.reject(new Error('no')) } }));
  await p.locator('.view-actions button:has-text("공유")').focus();
  await p.keyboard.press('Enter');
  await p.waitForTimeout(200);
  assert.equal((await active(p)).text, '공유');
  await context.close();

  const m = await page({ viewport: { width: 390, height: 844 }, isMobile: true, hasTouch: true });
  await m.p.goto(`${server.url}/`);
  await m.p.waitForSelector('.home-title');
  await m.p.locator('.menu-btn').click();
  await m.p.locator('.menu-theme [data-theme-mode=dark]').click();
  assert.equal(await m.p.getAttribute('.menu-theme [data-theme-mode=dark]', 'aria-pressed'), 'true');
  await m.p.keyboard.press('Escape');
  assert.match((await active(m.p)).cls, /menu-btn/);
  await m.context.close();
});

btest('A11Y-3 / A11Y-12 / FU-9: upload errors are announced and focused; /r names its loading heading; 다시 분석 keeps the file', async () => {
  const { context, p } = await page();
  await p.goto(`${server.url}/upload`);
  await p.setInputFiles('#upload-file', { name: 'garbage.xlsx', mimeType: 'application/octet-stream', buffer: Buffer.from('not an xlsx') });
  await p.waitForSelector('.error-card');
  await p.waitForTimeout(100);
  assert.match((await active(p)).cls, /error-card/);
  assert.ok((await p.evaluate(() => document.getElementById('live-assertive').textContent)).length > 0);

  await p.setInputFiles('#upload-file', SALES);
  await p.waitForSelector('#confirm-h');
  await p.uncheck('input[value="방문수"]');
  await p.selectOption('#confirm-h', '6');
  await p.click('button:has-text("분석하기")');
  await p.waitForURL(/\/r\//);
  const h1 = await p.$eval('h1', (h) => [h.textContent, Boolean(h.closest('[aria-busy]'))]);
  assert.deepEqual(h1, ['결과를 불러오는 중…', false]);
  await ready(p);
  assert.match(await p.textContent('.qb-full'), /예측: 6개월/);
  await p.click('.qb-full button:has-text("다시 분석")');
  await p.waitForSelector('#confirm-date');
  assert.deepEqual(await p.$$eval('.check-list input:checked', (els) => els.map((e) => e.value)), ['매출']);
  assert.equal(await p.$eval('#confirm-h', (s) => s.value), '6');
  await context.close();
});

btest('A11Y-14 / UX-8 / UX-9: back to the comparison focuses the card; sample chips; AI is never the default', async () => {
  const { context, p } = await page();
  await p.goto(`${server.url}/explore?src=sample&q=${q('emerging', 'fad', 'seasonal')}`);
  await p.waitForSelector('.rank-num');
  await p.locator('.rank-card').nth(1).locator('.more-link').focus();
  const index = await p.locator('.rank-card').nth(1).getAttribute('data-index');
  await p.keyboard.press('Enter');
  await p.waitForSelector('.hero');
  await p.goBack();
  await p.waitForSelector('.rank-num');
  await p.waitForTimeout(200);
  const back = await p.evaluate(() => {
    const a = document.activeElement;
    const r = a.getBoundingClientRect();
    return [a.className, a.closest('.rank-card') && a.closest('.rank-card').dataset.index, r.top >= 0 && r.bottom <= innerHeight];
  });
  assert.deepEqual(back, ['more-link', index, true]);

  await p.goto(`${server.url}/explore?src=sample&q=growing`);
  await ready(p);
  const chips = await p.$$eval('.qb-full [data-chip], .qb-full .chip-series', (els) => els.map((e) => e.textContent.trim()));
  assert.ok(chips.includes('기간: 2016.01 ~ 2025.12'), chips.join(' | '));
  assert.ok(chips.includes('단위: 월간'));
  assert.ok(chips.some((c) => c.startsWith('성장형')), 'the sample title, not its id');
  await p.click('#narrative button:has-text("AI로 다시 쓰기")');
  await p.waitForSelector('#narrative .pill-ai', { timeout: 20_000 });
  assert.match((await active(p)).text, /규칙 기반 해석 보기/);
  await context.close();

  const fresh = await page();
  await fresh.p.goto(`${server.url}/explore?src=sample&q=growing`);
  await ready(fresh.p);
  assert.match(await fresh.p.textContent('#narrative .narrative-badge'), /규칙 기반/);
  assert.equal(await fresh.p.locator('#narrative button:has-text("AI 해석 보기")').count(), 1);
  await fresh.context.close();
});

btest('UX-10 / FU-1 / FU-2: unknown links, the wiki language and all-failed comparisons', async () => {
  const { context, p } = await page();
  await p.goto(`${server.url}/s/AAAAAAAAAAAAAAAAAAAAAA`);
  await p.waitForSelector('.error-card');
  assert.equal(await p.textContent('h1'), '공유 링크를 열 수 없어요');
  assert.equal(await p.locator('.querybar-wrap[hidden]').count(), 1);

  await p.goto(`${server.url}/explore?src=wiki&q=${q('인공지능', '없는문서')}&range=2026-10-01~2026-10-31&unit=month`);
  await p.waitForSelector('.error-card');
  await p.waitForTimeout(200);
  const cards = await p.$$eval('.error-card .error-card-title', (els) => els.map((e) => e.textContent));
  assert.deepEqual(cards, ['인공지능', '없는문서'], 'one card per failed series');
  assert.equal(await p.locator('.error-card button:has-text("영어 위키백과에서 찾기")').count(), 1);
  assert.equal(await p.textContent('.qb-full [data-chip=source]'), '출처: 위키백과 · 한국어');
  await p.click('.error-card button:has-text("영어 위키백과에서 찾기")');
  await p.waitForURL((u) => new URL(u).searchParams.get('wl') === 'en');
  await context.close();

  const en = await page({ locale: 'en-US' });
  await en.p.goto(`${server.url}/?hl=en`);
  await en.p.waitForSelector('.home-title');
  await en.p.click('[data-source=wiki]');
  assert.equal(await en.p.getAttribute('.wiki-lang [aria-checked=true]', 'data-wl'), 'en');
  assert.match(await en.p.getAttribute('.examples a', 'href'), /wl=en/);
  await en.context.close();
});

btest('UX-2 / UX-3 / UX-18: phone layout — help popovers are readable, the answer is in the first screen, 적용 is visible', async () => {
  const { context, p } = await page({ viewport: { width: 390, height: 844 }, isMobile: true, hasTouch: true });
  await p.goto(`${server.url}/explore?src=naver&q=${q('선크림')}`);
  await p.waitForSelector('.hero');
  const hero = await p.$eval('.hero-outlook', (e) => e.getBoundingClientRect().bottom);
  assert.ok(hero < 844, `outlook line in the first screen (bottom ${hero})`);
  const help = p.locator('.tile .help-btn').nth(3);
  await help.scrollIntoViewIfNeeded();
  await help.click();
  const box = await p.$eval('.help-pop:not([hidden])', (e) => { const r = e.getBoundingClientRect(); return [r.width, r.top, r.bottom]; });
  assert.ok(box[0] >= 300, `help popover is ${box[0]} px wide`);
  assert.ok(box[1] >= 0 && box[2] <= 844, 'inside the viewport');
  await p.keyboard.press('Escape');
  await p.evaluate(() => window.scrollTo(0, 0));
  await p.locator('.qb-full [data-chip=filters]').click();
  await p.waitForSelector('[role=dialog]');
  await p.waitForTimeout(300);
  const apply = await p.$eval('[role=dialog] .sheet-foot .btn-primary', (b) => b.getBoundingClientRect().bottom);
  assert.ok(apply <= 844, `적용 at ${apply}`);
  await context.close();
});

btest('OPS-7: closing the tab cancels its running job', async () => {
  const { context, p } = await page();
  const posted = p.waitForResponse((r) => new URL(r.url()).pathname === '/api/analyze');
  await p.goto(`${server.url}/explore?src=wiki&q=${q('느린문서', '자외선 차단제')}`);
  const job = await (await posted).json();
  await p.waitForTimeout(300);
  await p.close({ runBeforeUnload: true });
  let status = null;
  for (let i = 0; i < 50 && status !== 'cancelled' && status !== 'done'; i += 1) {
    const res = await fetch(`${server.url}/api/jobs/${job.job_id}`);
    if (res.status === 200) status = (await res.json()).status;
    await new Promise((r) => setTimeout(r, 100));
  }
  assert.equal(status, 'cancelled');
  await context.close();
});
