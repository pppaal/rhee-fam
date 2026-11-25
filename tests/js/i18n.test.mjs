// UI catalog and i18n (spec F.4, G, H.3): ko and en have the same keys, no empty values, the same {placeholders};
// every t('…') key used in js/** exists; every G row is present; hl resolution and fallbacks.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, readdirSync, statSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { CATALOG } from '../../trendai/web/static/js/catalog.js';
import { format, resolveLang, setLang, t } from '../../trendai/web/static/js/i18n.js';

const HERE = dirname(fileURLToPath(import.meta.url));
const JS = join(HERE, '../../trendai/web/static/js');
const SPEC = join(HERE, '../../docs/WEB_SPEC.md');

const placeholders = (s) => [...s.matchAll(/\{([A-Za-z0-9_]+)\}/g)].map((m) => m[1]).sort();

test('ko and en have identical key sets', () => {
  const ko = Object.keys(CATALOG.ko).sort();
  const en = Object.keys(CATALOG.en).sort();
  assert.deepEqual(ko.filter((k) => !en.includes(k)), [], 'missing in en');
  assert.deepEqual(en.filter((k) => !ko.includes(k)), [], 'missing in ko');
});

test('no empty values', () => {
  for (const lang of ['ko', 'en']) {
    for (const [k, v] of Object.entries(CATALOG[lang])) {
      assert.equal(typeof v, 'string', `${lang}.${k}`);
      assert.ok(v.trim().length > 0, `${lang}.${k} is empty`);
    }
  }
});

test('every {placeholder} exists in both languages', () => {
  for (const k of Object.keys(CATALOG.ko)) {
    assert.deepEqual(placeholders(CATALOG.en[k]), placeholders(CATALOG.ko[k]), k);
  }
});

function walk(dir) {
  return readdirSync(dir).flatMap((name) => {
    const p = join(dir, name);
    return statSync(p).isDirectory() ? walk(p) : (p.endsWith('.js') ? [p] : []);
  });
}

test("every t('…') key referenced in js/** exists", () => {
  const missing = [];
  for (const file of walk(JS).filter((f) => !f.endsWith('catalog.js'))) {
    const src = readFileSync(file, 'utf8');
    for (const m of src.matchAll(/\bt\('([^']+)'/g)) {
      const key = m[1];
      if (key.includes('${')) continue;
      if (!(key in CATALOG.ko)) missing.push(`${file.slice(JS.length + 1)}: ${key}`);
    }
  }
  assert.deepEqual(missing, []);
});

test('dynamic key families used by components exist', () => {
  const families = {
    'source.': ['naver', 'wiki', 'upload', 'sample'],
    'range.': ['1y', '3y', '5y', 'all', 'custom'],
    'range.short.': ['1y', '3y', '5y', 'all', 'custom'],
    'unit.': ['auto', 'date', 'week', 'month'],
    'freq.': ['auto', 'D', 'W', 'M', 'Q', 'Y'],
    'filter.': ['all', 'pc', 'mo', 'm', 'f'],
    'theme.': ['system', 'light', 'dark'],
    'phase.': ['emerging', 'growing', 'peaking', 'declining', 'stable', 'reviving', 'volatile'],
    'compare.sort.': ['score', 'prob', 'expected', 'recent', 'level'],
    'compare.scale.': ['raw', 'index', 'log'],
    'progress.state.': ['pending', 'active', 'done'],
    'lang.': ['ko', 'en'],
    'search.placeholder.': ['naver', 'wiki', 'sample'],
    'home.example_terms.': ['naver', 'wiki'],
    'fix.': ['retry', 'reload', 'go_home'],
  };
  for (const [prefix, items] of Object.entries(families)) {
    for (const item of items) assert.ok(`${prefix}${item}` in CATALOG.ko, `${prefix}${item}`);
  }
  for (const id of ['overview', 'phase', 'bands', 'probability', 'accuracy', 'metrics', 'scores', 'index', 'limits', 'ai']) {
    assert.ok(`method.${id}.title` in CATALOG.ko, id);
    if (id !== 'phase') assert.ok(`method.${id}.body` in CATALOG.ko, id);
  }
  for (const id of ['naver', 'wiki', 'gt', 'upload', 'sample']) assert.ok(`sources.${id}.body` in CATALOG.ko, id);
  for (const id of ['upload', 'results', 'share', 'logs', 'ai', 'storage', 'contact']) assert.ok(`privacy.${id}.body` in CATALOG.ko, id);
});

/**
 * Spec G rows whose catalog text deliberately differs from the spec until docs/WEB_SPEC.md G is updated (the web-app
 * review fixes, reported as contract changes): the key must still exist, and the expected text is checked here.
 */
const PENDING_SPEC_TEXT = {
  'theme.system': '시스템 테마',                                                         // UX-19
  'home.samples.title': '샘플로 체험하기',                                                // UX-27
  'source.naver.unconfigured': '네이버 검색어 트렌드는 아직 연결되지 않았어요. 위키백과나 내 파일로 분석해 보세요.', // UX-26
  'upload.formats': 'CSV·TSV·TXT·엑셀(.xlsx)·JSON · 최대 {mb}MB · 값 열 최대 {n}개',      // OPS-12
  'confirm.max_cols': '값 열은 {n}개까지 고를 수 있어요.',                                // OPS-12
  'err.file_too_large': '파일이 너무 커요. {mb}MB까지 올릴 수 있어요.',                    // OPS-12
};

test('spec G rows with a pending spec update have the agreed text', () => {
  for (const [key, text] of Object.entries(PENDING_SPEC_TEXT)) assert.equal(CATALOG.ko[key], text, key);
});

test('every key of the spec G tables (G.1–G.9) is in the catalog with the spec text', () => {
  const spec = readFileSync(SPEC, 'utf8');
  const start = spec.indexOf('### G.1 Common');
  const end = spec.indexOf('### G.10');
  const rows = spec.slice(start, end).split('\n').filter((l) => /^\| `/.test(l));
  let checked = 0;
  for (const row of rows) {
    const cells = row.split(' | ').map((c) => c.replace(/^\|\s*/, '').replace(/\s*\|$/, ''));
    const keys = [...cells[0].matchAll(/`([a-z0-9_.]+)`/gi)].map((m) => m[1]).filter((k) => k.includes('.'));
    // `compare.sort.score` · `.prob` …: relative keys continue the first key's prefix
    const first = keys[0];
    const base = first.slice(0, first.lastIndexOf('.'));
    const full = keys.map((k) => (k.startsWith('.') ? base + k : k));
    for (const k of full) {
      assert.ok(k in CATALOG.ko, `missing ${k}`);
      checked += 1;
    }
    const koParts = cells[1].split(' · ');
    const specText = (k, text) => {
      if (k in PENDING_SPEC_TEXT) return;
      assert.equal(CATALOG.ko[k], text, `ko text of ${k}`);
    };
    if (full.length === 1 && !cells[1].startsWith('(')) {
      specText(full[0], cells[1].replace(/\\\|/g, '|'));
    } else if (full.length === koParts.length && full.length > 1) {
      full.forEach((k, i) => specText(k, koParts[i]));
    }
  }
  assert.ok(checked > 200, `checked ${checked}`);
});

test('t(): current language, ko fallback, key fallback and placeholders', () => {
  setLang('en');
  assert.equal(t('nav.explore'), 'Explore');
  assert.equal(t('samples.compare', { n: 3 }), 'Compare selected (3)');
  assert.equal(t('nav.explore', null, 'ko'), '탐색');
  assert.equal(t('no.such.key'), 'no.such.key');
  setLang('ko');
  assert.equal(t('samples.compare', { n: 2 }), '선택한 샘플 비교 (2)');
  assert.equal(format('{a} {b} {a}', { a: 1 }), '1 {b} 1');
});

test('hl resolution: URL → stored → navigator ko* → ko, else en', () => {
  assert.equal(resolveLang({ url: 'en', stored: 'ko', languages: ['ko-KR'] }), 'en');
  assert.equal(resolveLang({ url: 'jp', stored: 'en', languages: ['ko-KR'] }), 'en');
  assert.equal(resolveLang({ stored: null, languages: ['en-US', 'ko'] }), 'ko');
  assert.equal(resolveLang({ languages: ['en-US', 'de'] }), 'en');
  assert.equal(resolveLang({}), 'en');
});

test('copy tone: no exclamation marks or hype words in Korean UI text', () => {
  for (const [k, v] of Object.entries(CATALOG.ko)) {
    assert.ok(!/[!！]/.test(v), `${k} has an exclamation mark`);
    assert.ok(!/(폭발|대박)/.test(v), `${k} uses hype`);
  }
});
