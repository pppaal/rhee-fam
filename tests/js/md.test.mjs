// Markdown-lite (spec F.6.14, H.3): text only, nested bullets, **bold**, ## headings, date links for exactly the
// keys of the given map (longest match first) and nothing else.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { md, parseMarkdown } from '../../trendai/web/static/js/md.js';

/** A tiny DOM stand-in: enough for md() (createElement/TextNode/DocumentFragment, setAttribute, appendChild). */
function fakeDocument() {
  class Node {
    constructor(name) { this.nodeName = name; this.childNodes = []; this.attributes = {}; }
    appendChild(c) { this.childNodes.push(c); return c; }
    setAttribute(k, v) { this.attributes[k] = String(v); }
    get textContent() { return this.childNodes.map((c) => c.textContent).join(''); }
  }
  class Text { constructor(t) { this.nodeName = '#text'; this.data = t; } get textContent() { return this.data; } }
  return {
    createElement: (n) => new Node(n.toUpperCase()),
    createTextNode: (t) => new Text(t),
    createDocumentFragment: () => new Node('#fragment'),
  };
}

const text = (nodes) => nodes.map((n) => (n.type === 'text' || n.type === 'datelink' ? n.text : text(n.children || []))).join('');

test('<script> and other markup in the input stay text', () => {
  const tree = parseMarkdown('Hello <script>alert(1)</script> & <b>x</b>');
  assert.equal(tree.children.length, 1);
  assert.equal(tree.children[0].type, 'p');
  assert.deepEqual(tree.children[0].children, [{ type: 'text', text: 'Hello <script>alert(1)</script> & <b>x</b>' }]);
  const frag = md('<img src=x onerror=alert(1)>', null, fakeDocument());
  const p = frag.childNodes[0];
  assert.equal(p.nodeName, 'P');
  assert.equal(p.childNodes[0].nodeName, '#text');
  assert.equal(p.textContent, '<img src=x onerror=alert(1)>');
});

test('headings, paragraphs and nested bullets', () => {
  const tree = parseMarkdown('**한눈에 보기** — 요약\n둘째 줄\n\n## 과거\n- 하나\n  - 하나의 하위\n  - 또 하나\n- 둘\n\n끝');
  assert.deepEqual(tree.children.map((n) => n.type), ['p', 'h3', 'ul', 'p']);
  const [p, h3, ul, last] = tree.children;
  assert.equal(p.children[0].type, 'strong');
  assert.equal(text(p.children), '한눈에 보기 — 요약 둘째 줄');
  assert.equal(text(h3.children), '과거');
  assert.equal(ul.children.length, 2);
  const nested = ul.children[0].children.find((c) => c.type === 'ul');
  assert.ok(nested, 'nested list');
  assert.equal(nested.children.length, 2);
  assert.equal(text(nested.children[1].children), '또 하나');
  assert.equal(text(last.children), '끝');
});

test('**bold** pairs; an unpaired ** stays literal', () => {
  const tree = parseMarkdown('a **b** c **d');
  const kids = tree.children[0].children;
  assert.deepEqual(kids.map((k) => k.type), ['text', 'strong', 'text']);
  assert.equal(text(kids), 'a b c **d');
});

test('date links exactly for the keys of dateLinks, longest match first', () => {
  const links = { '2024.03': '2024-03-01', '2024.03.05': '2024-03-05', '2024 1분기': '2024-01-01' };
  const tree = parseMarkdown('- 2024.03.05에 급등, 2024.03 평균, **2024 1분기** 이후. 2023.03은 링크 없음', links);
  const li = tree.children[0].children[0];
  const found = [];
  const walk = (nodes) => nodes.forEach((n) => { if (n.type === 'datelink') found.push([n.text, n.date]); if (n.children) walk(n.children); });
  walk(li.children);
  assert.deepEqual(found, [['2024.03.05', '2024-03-05'], ['2024.03', '2024-03-01'], ['2024 1분기', '2024-01-01']]);
  const frag = md('2024.03.05', links, fakeDocument());
  const btn = frag.childNodes[0].childNodes[0];
  assert.equal(btn.nodeName, 'BUTTON');
  assert.equal(btn.attributes.class, 'datelink');
  assert.equal(btn.attributes['data-date'], '2024-03-05');
  assert.equal(btn.attributes.type, 'button');
});

test('no date logic of its own: without a map nothing is linked', () => {
  const tree = parseMarkdown('2024-03-05 and 2024.03.05');
  assert.deepEqual(tree.children[0].children, [{ type: 'text', text: '2024-03-05 and 2024.03.05' }]);
});

test('keys with regex characters are matched literally', () => {
  const tree = parseMarkdown('Q1 2024 (x) and Q1 2024', { 'Q1 2024 (x)': '2024-01-01' });
  const kinds = tree.children[0].children.map((n) => n.type);
  assert.deepEqual(kinds, ['datelink', 'text']);
});

test('CRLF input and trailing spaces', () => {
  const tree = parseMarkdown('## 제목  \r\n- a\r\n- b\r\n');
  assert.deepEqual(tree.children.map((n) => n.type), ['h3', 'ul']);
});
