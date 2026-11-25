// Modal dialog / bottom sheet (spec F.6.6, F.9 item 7). Owner: WP3-app.
//
// <div role="dialog" aria-modal="true" aria-labelledby> with a title, focus trap, Esc and backdrop close; focus returns
// to the opener. Below 640 px it slides up from the bottom (CSS; no animation with reduced motion), otherwise it is
// a centred dialog. Dialogs stack: Esc closes only the top one.

import { clear, focusables, focusFirst, h } from '../dom.js';
import { t } from '../i18n.js';
import { icon } from '../ui.js';

const stack = [];
let counter = 0;

/**
 * Keep Tab / Shift+Tab inside `container`.
 * @param {HTMLElement} container
 * @param {KeyboardEvent} event
 */
export function trapFocus(container, event) {
  if (event.key !== 'Tab') return;
  const items = focusables(container);
  if (!items.length) {
    event.preventDefault();
    container.focus();
    return;
  }
  const first = items[0];
  const last = items[items.length - 1];
  const active = document.activeElement;
  if (event.shiftKey && (active === first || !container.contains(active))) {
    event.preventDefault();
    last.focus();
  } else if (!event.shiftKey && (active === last || !container.contains(active))) {
    event.preventDefault();
    first.focus();
  }
}

function onDocumentKey(event) {
  const top = stack[stack.length - 1];
  if (!top) return;
  if (event.key === 'Escape') {
    event.preventDefault();
    event.stopPropagation();
    top.close();
  } else {
    trapFocus(top.dialog, event);
  }
}

/** True while any modal dialog, sheet or modal popover is open. */
export function modalOpen() {
  return stack.length > 0;
}

/**
 * Register a modal surface (also used by popover.js) so Esc and focus trapping work for the top one.
 * @param {{dialog: HTMLElement, close: () => void}} entry
 * @returns {() => void} unregister
 */
export function registerModal(entry) {
  if (!stack.length) document.addEventListener('keydown', onDocumentKey, true);
  stack.push(entry);
  return () => {
    const i = stack.indexOf(entry);
    if (i >= 0) stack.splice(i, 1);
    if (!stack.length) document.removeEventListener('keydown', onDocumentKey, true);
  };
}

/**
 * Open a modal sheet.
 * @param {{title: string, content: Node|Node[], footer?: Node|Node[], opener?: HTMLElement|null,
 *          onClose?: () => void, className?: string, initialFocus?: HTMLElement|null}} opts
 * @returns {{el: HTMLElement, dialog: HTMLElement, body: HTMLElement, close: () => void, setContent: (n) => void}}
 */
export function openSheet(opts) {
  const id = `sheet-title-${++counter}`;
  const opener = opts.opener || document.activeElement;
  const body = h('div', { class: 'sheet-body' }, opts.content);
  const dialog = h('div', {
    class: ['sheet', opts.className], role: 'dialog', 'aria-modal': 'true', 'aria-labelledby': id, tabindex: '-1',
  },
  h('div', { class: 'sheet-handle', 'aria-hidden': 'true' }),
  h('div', { class: 'sheet-head' },
    h('h2', { class: 'sheet-title', id }, opts.title),
    h('button', {
      type: 'button', class: 'icon-btn sheet-close', 'aria-label': t('common.close'), on: { click: () => close() },
    }, icon('ui-close'))),
  body,
  opts.footer ? h('div', { class: 'sheet-foot' }, opts.footer) : null);
  const layer = h('div', { class: 'sheet-layer' },
    h('div', { class: 'sheet-backdrop', on: { click: () => close() } }),
    dialog);
  document.body.appendChild(layer);
  document.documentElement.classList.add('has-modal');

  let closed = false;
  const entry = { dialog, close: () => close() };
  const unregister = registerModal(entry);
  function close() {
    if (closed) return;
    closed = true;
    unregister();
    layer.remove();
    if (!modalOpen()) document.documentElement.classList.remove('has-modal');
    if (opts.onClose) opts.onClose();
    if (opener && opener.isConnected && typeof opener.focus === 'function') opener.focus({ preventScroll: true });
  }
  requestAnimationFrame(() => layer.classList.add('is-open'));
  if (opts.initialFocus) opts.initialFocus.focus();
  else if (!focusFirst(body)) dialog.focus();
  return {
    el: layer,
    dialog,
    body,
    close,
    setContent(content) {
      clear(body);
      body.append(...[].concat(content));
    },
  };
}

/**
 * A small confirm dialog: resolves true when the confirm button was pressed.
 * @param {{title: string, body: string, confirm: string, cancel?: string, danger?: boolean, opener?: HTMLElement}} o
 * @returns {Promise<boolean>}
 */
export function confirmDialog(o) {
  return new Promise((resolve) => {
    let answer = false;
    const ok = h('button', {
      type: 'button', class: ['btn', o.danger ? 'btn-danger' : 'btn-primary'],
      on: { click: () => { answer = true; sheet.close(); } },
    }, o.confirm);
    const cancel = h('button', { type: 'button', class: 'btn', on: { click: () => sheet.close() } },
      o.cancel || t('common.cancel'));
    const sheet = openSheet({
      title: o.title, content: h('p', null, o.body), footer: [cancel, ok], opener: o.opener,
      className: 'sheet-small', initialFocus: cancel, onClose: () => resolve(answer),
    });
  });
}
