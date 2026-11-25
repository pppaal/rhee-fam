// DOM builders (spec F.1). Owner: WP3-app; used by every component, page and chart module.
//
// Hard rules enforced here (F.1 rule 1): text always becomes text nodes (never parsed as HTML), no inline
// `on*=` handlers, no `style` attribute strings. Nothing runs at import time, so the module can be imported
// anywhere; the functions themselves need a DOM (`document`).
//
// Attribute conventions for h() and svg():
//   * `class`        string, or an array whose falsy entries are dropped:  {class: ['chip', on && 'is-on']}
//   * `dataset`      object -> element.dataset (camelCase keys):          {dataset: {date: '2026-11-16'}}
//   * `style`        object of CSS properties set with style.setProperty (custom properties allowed):
//                    {style: {'--slot': 'var(--slot-2)', width: '120px'}}. A string throws (CSP rule).
//   * `on`           object of event listeners, added with addEventListener: {on: {click: fn, keydown: fn}}.
//                    A null/undefined listener is skipped (conditional handlers: {click: editable ? fn : null}).
//                    Any other key starting with "on" throws, so an inline handler can never be created.
//   * `props`        object assigned as DOM properties (form state): {props: {value: 'x', checked: true}}.
//   * true           -> attribute present with an empty value (`hidden`, `disabled`, `aria-*` need strings).
//   * false / null / undefined -> attribute omitted.
//   * anything else  -> String(value) via setAttribute (aria-*, role, href, viewBox, d, ...).
//   * `innerHTML`, `outerHTML`, `insertAdjacentHTML` keys (also inside `props`) throw.
// Children: strings and numbers become text nodes; Nodes are appended; arrays are flattened; null, undefined,
// true and false are skipped (so `cond && h(...)` works).

const SVG_NS = 'http://www.w3.org/2000/svg';
const XLINK_NS = 'http://www.w3.org/1999/xlink';
const FORBIDDEN = new Set(['innerHTML', 'outerHTML', 'insertAdjacentHTML']);

/** Selector of elements that can take keyboard focus (used by focusFirst and focus traps). */
export const FOCUSABLE_SELECTOR = [
  'a[href]', 'area[href]', 'button:not([disabled])', 'input:not([disabled]):not([type="hidden"])',
  'select:not([disabled])', 'textarea:not([disabled])', 'iframe', 'summary',
  '[contenteditable=""]', '[contenteditable="true"]', '[tabindex]',
].join(',');

function applyAttrs(el, attrs, isSvg) {
  if (attrs == null) return;
  if (typeof attrs !== 'object' || attrs instanceof Node || Array.isArray(attrs)) {
    throw new TypeError('dom: attrs must be a plain object or null');
  }
  for (const [key, value] of Object.entries(attrs)) {
    if (FORBIDDEN.has(key)) throw new TypeError(`dom: "${key}" is not allowed`);
    if (key === 'on') {
      if (value == null) continue;
      for (const [type, handler] of Object.entries(value)) {
        if (handler == null) continue;
        if (typeof handler !== 'function') throw new TypeError(`dom: on.${type} must be a function`);
        el.addEventListener(type, handler);
      }
      continue;
    }
    if (/^on/i.test(key)) {
      throw new TypeError(`dom: "${key}" is not allowed; use {on: {${key.slice(2).toLowerCase()}: fn}}`);
    }
    if (key === 'class' || key === 'className') {
      const cls = Array.isArray(value) ? value.filter(Boolean).join(' ') : value;
      if (cls) el.setAttribute('class', String(cls));
      continue;
    }
    if (key === 'style') {
      if (value == null) continue;
      if (typeof value !== 'object') throw new TypeError('dom: style must be an object (CSP: no style strings)');
      for (const [prop, v] of Object.entries(value)) {
        if (v != null && v !== false) el.style.setProperty(prop, String(v));
      }
      continue;
    }
    if (key === 'dataset') {
      if (value == null) continue;
      for (const [k, v] of Object.entries(value)) {
        if (v != null && v !== false) el.dataset[k] = String(v);
      }
      continue;
    }
    if (key === 'props') {
      if (value == null) continue;
      for (const [k, v] of Object.entries(value)) {
        if (FORBIDDEN.has(k) || /^on/i.test(k)) throw new TypeError(`dom: props.${k} is not allowed`);
        el[k] = v;
      }
      continue;
    }
    if (value == null || value === false) continue;
    const text = value === true ? '' : String(value);
    if (isSvg && key.startsWith('xlink:')) el.setAttributeNS(XLINK_NS, key, text);
    else el.setAttribute(key, text);
  }
}

function appendChildren(el, children) {
  for (const child of children) {
    if (child == null || child === false || child === true) continue;
    if (Array.isArray(child)) appendChildren(el, child);
    else if (child instanceof Node) el.appendChild(child);
    else el.appendChild(document.createTextNode(String(child)));
  }
}

/**
 * Create an HTML element.
 * @param {string} tag  Element name, e.g. 'button'.
 * @param {Object|null} [attrs]  Attributes per the conventions at the top of this file.
 * @param {...(Node|string|number|null|undefined|boolean|Array)} children
 * @returns {HTMLElement}
 */
export function h(tag, attrs, ...children) {
  const el = document.createElement(tag);
  applyAttrs(el, attrs, false);
  appendChildren(el, children);
  return el;
}

/**
 * Create an SVG element in the SVG namespace (same conventions as h(); `xlink:href` keys use the XLink namespace,
 * plain `href` is preferred). Use for every chart node (F.7).
 * @param {string} tag  e.g. 'svg', 'g', 'path', 'title'.
 * @param {Object|null} [attrs]
 * @param {...(Node|string|number|null|undefined|boolean|Array)} children
 * @returns {SVGElement}
 */
export function svg(tag, attrs, ...children) {
  const el = document.createElementNS(SVG_NS, tag);
  applyAttrs(el, attrs, true);
  appendChildren(el, children);
  return el;
}

/**
 * Remove every child of `el` and return it.
 * @template {Element} T
 * @param {T} el
 * @returns {T}
 */
export function clear(el) {
  while (el.firstChild) el.removeChild(el.firstChild);
  return el;
}

/**
 * The visible, keyboard-focusable descendants of `root`, in document order (tabindex="-1" excluded).
 * @param {Element} root
 * @returns {HTMLElement[]}
 */
export function focusables(root) {
  return Array.from(root.querySelectorAll(FOCUSABLE_SELECTOR)).filter((el) =>
    el.getAttribute('tabindex') !== '-1' && !el.closest('[hidden], [inert]') && el.getClientRects().length > 0);
}

/**
 * Focus the first focusable descendant of `root`; when there is none, focus `root` itself if it is focusable
 * (e.g. a page `<h1 tabindex="-1">`, a dialog with tabindex="-1").
 * @param {Element} root
 * @param {{preventScroll?: boolean}} [opts]
 * @returns {HTMLElement|null} the element that received focus, or null.
 */
export function focusFirst(root, opts = {}) {
  if (!root) return null;
  const target = focusables(root)[0] || (root.hasAttribute('tabindex') ? root : null);
  if (!target) return null;
  target.focus({ preventScroll: Boolean(opts.preventScroll) });
  return target;
}
