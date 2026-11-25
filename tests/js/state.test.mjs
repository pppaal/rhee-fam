// URL grammar and state (spec F.2, H.3). Node 22 built-in runner; imports the pure module by relative path.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  emptyState, encodeQ, parseLocation, parseQ, queryKey, reanalyzeSingleState, serialize, switchSourceState,
  toAnalyzeBody,
} from '../../trendai/web/static/js/state.js';

test('q grammar: + joins terms, comma separates series (raw parsing, not URLSearchParams)', () => {
  const s = parseLocation('/explore', '?q=%EC%84%A0+%EC%9E%90,%ED%8C%A8');
  assert.deepEqual(s.q, [['선', '자'], ['패']]);
  assert.equal(s.route, 'explore');
});

test('literal + , & # and spaces inside a term round-trip', () => {
  const state = emptyState('explore');
  state.q = [['a+b', 'c,d'], ['e & f #1'], ['50% off']];
  const url = serialize(state);
  assert.match(url, /q=a%2Bb\+c%2Cd,e%20%26%20f%20%231,50%25%20off/);
  const back = parseLocation(url.split('?')[0], `?${url.split('?')[1]}`);
  assert.deepEqual(back.q, state.q);
});

test('defaults are omitted from the URL and restored on parse', () => {
  const state = emptyState('explore');
  state.q = [['선크림']];
  assert.equal(serialize(state), '/explore?q=%EC%84%A0%ED%81%AC%EB%A6%BC');
  const back = parseLocation('/explore', '?q=%EC%84%A0%ED%81%AC%EB%A6%BC');
  assert.deepEqual(back, state);
});

test('serialisation order is src, q, range, unit, h, dev, gender, ages, wl, hl, scale, band, view, focus', () => {
  const s = emptyState('explore');
  Object.assign(s, {
    src: 'naver', q: [['a']], range: '1y', unit: 'week', h: 13, dev: 'mo', gender: 'f', ages: ['4', '5'], wl: 'en',
    hl: 'en', scale: 'log', band: '80', view: 'table', focus: 'a',
  });
  const keys = serialize(s).split('?')[1].split('&').map((p) => p.split('=')[0]);
  assert.deepEqual(keys, ['src', 'q', 'range', 'unit', 'h', 'dev', 'gender', 'ages', 'wl', 'hl', 'scale', 'band', 'view', 'focus']);
});

test('at most 5 series are kept, empties and duplicates dropped (C.5 normalisation)', () => {
  assert.deepEqual(parseQ('a,b,c,d,e,f,g'), [['a'], ['b'], ['c'], ['d'], ['e']]);
  assert.deepEqual(parseQ('a,,+,b+b, a ,c'), [['a'], ['b'], ['c']]);
  assert.deepEqual(parseQ('%ED%95%9C%EA%B8%80+%20x%20'), [['한글', 'x']]);
});

test('malformed percent escapes do not throw', () => {
  assert.deepEqual(parseQ('%E0%A4%A,b'), [['%E0%A4%A'], ['b']]);
});

test('invalid parameter values fall back to defaults', () => {
  const s = parseLocation('/explore', '?q=a&src=google&range=2y&unit=year&h=0&dev=tv&gender=x&ages=0,4,12,4,2&wl=EN&hl=jp&scale=sqrt&band=50&view=list');
  assert.equal(s.src, null);
  assert.equal(s.range, '5y');
  assert.equal(s.unit, 'auto');
  assert.equal(s.h, 'auto');
  assert.equal(s.dev, '');
  assert.equal(s.gender, '');
  assert.deepEqual(s.ages, ['2', '4']);
  assert.equal(s.wl, 'ko');
  assert.equal(s.hl, null);
  assert.equal(s.scale, null);
  assert.equal(s.band, 'both');
  assert.equal(s.view, 'chart');
});

test('routes: /r and /s ids, static pages, trailing slash, not found', () => {
  const id = 'Zx8kP2mQ9rT4vW6yA1bC3d';
  assert.deepEqual([parseLocation(`/r/${id}`, '').route, parseLocation(`/r/${id}`, '').id], ['result', id]);
  assert.deepEqual([parseLocation(`/s/${id}`, '').route, parseLocation(`/s/${id}`, '').id], ['shared', id]);
  assert.equal(parseLocation('/r/short', '').route, 'notfound');
  for (const r of ['upload', 'samples', 'method', 'sources', 'privacy', 'explore']) assert.equal(parseLocation(`/${r}`, '').route, r);
  assert.equal(parseLocation('/method/', '').route, 'method');
  assert.equal(parseLocation('/', '').route, 'home');
  const nf = parseLocation('/nope', '?hl=en');
  assert.equal(nf.route, 'notfound');
  assert.equal(serialize(nf), '/nope?hl=en');
});

test('/ with src + q parses as a pre-fill of home (no analysis route)', () => {
  const s = parseLocation('/', '?src=wiki&q=%EC%84%A0%ED%81%AC%EB%A6%BC');
  assert.equal(s.route, 'home');
  assert.equal(s.src, 'wiki');
  assert.deepEqual(s.q, [['선크림']]);
});

function rng(seed) {
  let x = seed;
  return () => {
    x = (x * 1103515245 + 12345) % 2147483648;
    return x / 2147483648;
  };
}

test('parse(serialize(s)) == s for 30 generated states', () => {
  const r = rng(7);
  const pick = (arr) => arr[Math.floor(r() * arr.length)];
  const terms = ['선크림', '자외선차단제', '패딩', 'a+b', 'x,y', '50%', 'New York', 'ÄÖ', '#1', '&co', '김장 김치'];
  const routes = ['explore', 'home', 'result', 'shared', 'samples', 'method'];
  for (let i = 0; i < 30; i += 1) {
    const s = emptyState(pick(routes));
    if (s.route === 'result' || s.route === 'shared') s.id = 'Abcdefghij_klmnopq-rst';
    const nSeries = Math.floor(r() * 6);
    const used = new Set();
    for (let k = 0; k < nSeries; k += 1) {
      const group = [];
      const nTerms = 1 + Math.floor(r() * 3);
      for (let j = 0; j < nTerms; j += 1) {
        const t = pick(terms);
        if (!group.includes(t)) group.push(t);
      }
      const key = JSON.stringify(group);
      if (!used.has(key) && s.q.length < 5) { used.add(key); s.q.push(group); }
    }
    s.src = pick([null, 'naver', 'wiki', 'sample']);
    s.range = pick(['5y', '1y', '3y', 'all', '2021-01-04~2024-12-29']);
    s.unit = pick(['auto', 'date', 'week', 'month']);
    s.h = pick(['auto', 13, 26, 52]);
    s.dev = pick(['', 'pc', 'mo']);
    s.gender = pick(['', 'm', 'f']);
    s.ages = pick([[], ['1'], ['4', '5'], ['2', '10', '11']]);
    s.wl = pick(['ko', 'en', 'ja']);
    s.hl = pick([null, 'ko', 'en']);
    s.scale = pick([null, 'raw', 'index', 'log']);
    s.band = pick(['both', '80', '95', 'off']);
    s.view = pick(['chart', 'table']);
    s.focus = pick([null, '선크림 + 자외선차단제', 'a+b (2)', 'x']);
    const url = serialize(s);
    const [path, query = ''] = url.split('?');
    assert.deepEqual(parseLocation(path, query ? `?${query}` : ''), s, url);
  }
});

test('toAnalyzeBody matches the C.5 request shape', () => {
  const s = parseLocation('/explore', '?src=naver&q=%EC%84%A0%ED%81%AC%EB%A6%BC+%EC%9E%90%EC%99%B8%EC%84%A0%EC%B0%A8%EB%8B%A8%EC%A0%9C,%ED%8C%A8%EB%94%A9');
  assert.deepEqual(toAnalyzeBody(s), {
    source: 'naver',
    series: [{ terms: ['선크림', '자외선차단제'] }, { terms: ['패딩'] }],
    range: '5y', unit: 'auto', h: 'auto', dev: '', gender: '', ages: [], wl: 'ko',
    upload_id: null, date_col: null, value_cols: null, freq: 'auto', unit_label: null,
  });
  const d = parseLocation('/explore', '?q=x&h=13&ages=5,4');
  const body = toAnalyzeBody(d, { defaultSource: 'wiki' });
  assert.equal(body.source, 'wiki');
  assert.equal(body.h, 13);
  assert.deepEqual(body.ages, ['4', '5']);
});

test('queryKey covers only analysis-relevant fields', () => {
  const a = parseLocation('/explore', '?src=naver&q=a&scale=log&band=80&view=table&focus=a&hl=en');
  const b = parseLocation('/explore', '?src=naver&q=a');
  assert.equal(queryKey(a), queryKey(b));
  assert.notEqual(queryKey(b), queryKey(parseLocation('/explore', '?src=naver&q=a&range=1y')));
  // wiki ignores Naver filters; samples ignore range and unit
  assert.equal(queryKey(parseLocation('/explore', '?src=wiki&q=a&dev=pc')), queryKey(parseLocation('/explore', '?src=wiki&q=a')));
  assert.equal(queryKey(parseLocation('/explore', '?src=sample&q=fad&range=1y')), queryKey(parseLocation('/explore', '?src=sample&q=fad')));
  // the default source resolves
  assert.equal(queryKey(parseLocation('/explore', '?q=a'), { defaultSource: 'wiki' }), queryKey(parseLocation('/explore', '?src=wiki&q=a')));
});

test('reanalyze_single keeps every analysis param but q (and drops view params)', () => {
  const s = parseLocation('/explore', '?src=naver&q=a,%EC%A0%80%EA%B2%80%EC%83%89%EC%96%B4&range=1y&unit=month&h=12&dev=mo&gender=f&ages=4,5&hl=en&scale=log&view=table&focus=a');
  const next = reanalyzeSingleState(s, '%EC%A0%80%EA%B2%80%EC%83%89%EC%96%B4');
  assert.equal(serialize(next), '/explore?src=naver&q=%EC%A0%80%EA%B2%80%EC%83%89%EC%96%B4&range=1y&unit=month&h=12&dev=mo&gender=f&ages=4,5&hl=en');
  assert.deepEqual(reanalyzeSingleState(s, 'a+b%2Bc').q, [['a', 'b+c']]);
});

test('use_wiki pre-fills home with the first term of each series', () => {
  const s = parseLocation('/explore', '?src=naver&q=%EC%84%A0%ED%81%AC%EB%A6%BC+%EC%9E%90%EC%99%B8%EC%84%A0,%ED%8C%A8%EB%94%A9&range=1y&hl=en');
  const next = switchSourceState(s, 'wiki');
  assert.equal(next.route, 'home');
  assert.equal(serialize(next), `/?src=wiki&q=${encodeQ([['선크림'], ['패딩']])}&hl=en`);
});
