// Markdown-lite (spec F.6.14). Owner: WP3-app. Pure: no DOM access at import time (F.1 rule 2).
//
// Supported, nothing else: `## ` headings (rendered as <h3>), `- ` bullets nested by 2-space indentation, blank-line
// paragraphs, `**bold**`. Every exact occurrence of a `dateLinks` key (the VM's narrative.date_links, D.2.9: display
// text → chart date; longest keys first) becomes <button class="datelink" data-date="…">. Text only ever becomes text
// nodes, so markup in the input (e.g. `<script>`) stays visible text. md.js has no date regex of its own.
//
// parseMarkdown() builds a plain tree (testable without a DOM); md() renders it with a document.

const BULLET_RE = /^( *)- (.*)$/;
const HEADING_RE = /^## (.*)$/;

function escapeRegExp(s) {
  return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

/** Split a text run into text and datelink nodes. */
function linkDates(text, matcher, links) {
  if (!matcher || !text) return text ? [{ type: 'text', text }] : [];
  const out = [];
  let last = 0;
  matcher.lastIndex = 0;
  for (let m = matcher.exec(text); m; m = matcher.exec(text)) {
    if (m.index > last) out.push({ type: 'text', text: text.slice(last, m.index) });
    out.push({ type: 'datelink', text: m[0], date: links[m[0]] });
    last = m.index + m[0].length;
  }
  if (last < text.length) out.push({ type: 'text', text: text.slice(last) });
  return out;
}

/** Inline nodes of one line: **bold** runs and date links. An unpaired ** stays literal. */
function inline(text, matcher, links) {
  const parts = text.split('**');
  if (parts.length % 2 === 0) {           // odd number of markers: keep the last one as text
    const tail = parts.pop();
    parts[parts.length - 1] += `**${tail}`;
  }
  const out = [];
  parts.forEach((part, i) => {
    if (!part) return;
    if (i % 2 === 1) out.push({ type: 'strong', children: linkDates(part, matcher, links) });
    else out.push(...linkDates(part, matcher, links));
  });
  return out;
}

/**
 * Parse Markdown-lite into a plain tree:
 * {type: 'root', children: [{type: 'h3'|'p', children}, {type: 'ul', children: [{type: 'li', children}]}]}
 * where inline children are {type: 'text', text} | {type: 'strong', children} | {type: 'datelink', text, date};
 * a nested list is a 'ul' child of its parent 'li'.
 * @param {string} text
 * @param {Object<string, string>} [dateLinks]
 * @returns {{type: 'root', children: Object[]}}
 */
export function parseMarkdown(text, dateLinks) {
  const links = {};
  for (const [k, v] of Object.entries(dateLinks || {})) {
    if (typeof k === 'string' && k && typeof v === 'string' && v) links[k] = v;
  }
  const keys = Object.keys(links).sort((a, b) => b.length - a.length);
  const matcher = keys.length ? new RegExp(keys.map(escapeRegExp).join('|'), 'g') : null;

  const root = { type: 'root', children: [] };
  let para = null;          // current paragraph lines
  let lists = [];           // stack of {level, ul}
  const closePara = () => {
    if (para) root.children.push({ type: 'p', children: inline(para.join(' '), matcher, links) });
    para = null;
  };
  const closeLists = () => {
    lists = [];
  };

  for (const rawLine of String(text == null ? '' : text).replace(/\r\n?/g, '\n').split('\n')) {
    const line = rawLine.replace(/\s+$/, '');
    if (!line.trim()) {
      closePara();
      closeLists();
      continue;
    }
    const heading = HEADING_RE.exec(line);
    if (heading) {
      closePara();
      closeLists();
      root.children.push({ type: 'h3', children: inline(heading[1].trim(), matcher, links) });
      continue;
    }
    const bullet = BULLET_RE.exec(line);
    if (bullet) {
      closePara();
      const level = Math.floor(bullet[1].length / 2);
      const item = { type: 'li', children: inline(bullet[2].trim(), matcher, links) };
      while (lists.length && lists[lists.length - 1].level > level) lists.pop();
      const top = lists[lists.length - 1];
      if (top && top.level === level) {
        top.ul.children.push(item);
      } else if (top && top.level < level) {
        const parentItem = top.ul.children[top.ul.children.length - 1];
        const ul = { type: 'ul', children: [item] };
        parentItem.children.push(ul);
        lists.push({ level, ul });
      } else {
        const ul = { type: 'ul', children: [item] };
        root.children.push(ul);
        lists.push({ level, ul });
      }
      continue;
    }
    if (lists.length && /^\s+/.test(line)) {
      // an indented continuation line belongs to the last list item
      const ul = lists[lists.length - 1].ul;
      const li = ul.children[ul.children.length - 1];
      li.children.push({ type: 'text', text: ' ' }, ...inline(line.trim(), matcher, links));
      continue;
    }
    closeLists();
    if (!para) para = [];
    para.push(line.trim());
  }
  closePara();
  return root;
}

function render(node, doc) {
  switch (node.type) {
    case 'text':
      return doc.createTextNode(node.text);
    case 'datelink': {
      const btn = doc.createElement('button');
      btn.setAttribute('type', 'button');
      btn.setAttribute('class', 'datelink');
      btn.setAttribute('data-date', node.date);
      btn.appendChild(doc.createTextNode(node.text));
      return btn;
    }
    default: {
      const el = doc.createElement(node.type === 'root' ? 'div' : node.type);
      for (const child of node.children || []) el.appendChild(render(child, doc));
      return el;
    }
  }
}

/**
 * Render Markdown-lite to a DocumentFragment of safe nodes (text nodes, h3, p, ul, li, strong, button.datelink).
 * @param {string} text
 * @param {Object<string, string>} [dateLinks]  display text → chart date (ISO)
 * @param {Document} [doc=globalThis.document]
 * @returns {DocumentFragment}
 */
export function md(text, dateLinks, doc = globalThis.document) {
  const frag = doc.createDocumentFragment();
  for (const child of parseMarkdown(text, dateLinks).children) frag.appendChild(render(child, doc));
  return frag;
}
