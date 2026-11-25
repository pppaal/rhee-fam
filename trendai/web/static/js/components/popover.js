// Anchored popover and menu button (spec F.6.4, F.6.6). Owner: WP3-app.
//
// openPopover(): from 640 px a panel anchored under its opener (role="dialog", aria-modal, labelled, focus trap, Esc
// and outside click close, focus returns to the opener); below 640 px the same content opens as a bottom sheet.
// menuButton(): a button with aria-haspopup="menu" opening a role="menu" list (arrow keys, Home/End, Esc).

import { focusFirst, h } from '../dom.js';
import { t } from '../i18n.js';
import { icon, isNarrow } from '../ui.js';
import { openSheet, registerModal } from './sheet.js';

let counter = 0;

/** Position a fixed panel under (or above) `anchor`, kept inside the viewport. */
export function placeUnder(panel, anchor, gap = 6) {
  const r = anchor.getBoundingClientRect();
  const vw = document.documentElement.clientWidth;
  const vh = window.innerHeight;
  const pw = panel.offsetWidth;
  const ph = panel.offsetHeight;
  let left = Math.min(Math.max(8, r.left), Math.max(8, vw - pw - 8));
  let top = r.bottom + gap;
  if (top + ph > vh - 8 && r.top - gap - ph > 8) top = r.top - gap - ph;
  if (!Number.isFinite(left)) left = 8;
  panel.style.setProperty('left', `${Math.round(left)}px`);
  panel.style.setProperty('top', `${Math.round(top)}px`);
}

/**
 * Open an anchored popover (bottom sheet below 640 px).
 * @param {HTMLElement} anchor
 * @param {{title: string, content: Node|Node[], footer?: Node|Node[], onClose?: () => void, className?: string}} opts
 * @returns {{close: () => void, el: HTMLElement}}
 */
export function openPopover(anchor, opts) {
  if (isNarrow()) {
    const sheet = openSheet({ ...opts, opener: anchor });
    return { close: sheet.close, el: sheet.dialog };
  }
  const id = `pop-title-${++counter}`;
  const panel = h('div', {
    class: ['popover', opts.className], role: 'dialog', 'aria-modal': 'true', 'aria-labelledby': id, tabindex: '-1',
  },
  h('div', { class: 'popover-head' },
    h('h2', { class: 'popover-title', id }, opts.title),
    h('button', { type: 'button', class: 'icon-btn', 'aria-label': t('common.close'), on: { click: () => close() } },
      icon('ui-close'))),
  h('div', { class: 'popover-body' }, opts.content),
  opts.footer ? h('div', { class: 'popover-foot' }, opts.footer) : null);
  document.body.appendChild(panel);
  anchor.setAttribute('aria-expanded', 'true');
  placeUnder(panel, anchor);

  let closed = false;
  const reposition = () => placeUnder(panel, anchor);
  const onPointer = (e) => {
    if (!panel.contains(e.target) && !anchor.contains(e.target)) close(false);
  };
  const unregister = registerModal({ dialog: panel, close: () => close() });
  window.addEventListener('resize', reposition);
  window.addEventListener('scroll', reposition, true);
  document.addEventListener('pointerdown', onPointer, true);

  function close(restoreFocus = true) {
    if (closed) return;
    closed = true;
    unregister();
    window.removeEventListener('resize', reposition);
    window.removeEventListener('scroll', reposition, true);
    document.removeEventListener('pointerdown', onPointer, true);
    panel.remove();
    anchor.setAttribute('aria-expanded', 'false');
    if (opts.onClose) opts.onClose();
    if (restoreFocus && anchor.isConnected) anchor.focus({ preventScroll: true });
  }
  if (!focusFirst(panel.querySelector('.popover-body'))) panel.focus();
  return { close, el: panel };
}

/**
 * A menu button.
 * @param {{label: string|Node[], ariaLabel?: string, icon?: string, className?: string, items: Array<{
 *   label: string, href?: string, download?: boolean|string, onSelect?: () => void, disabled?: boolean,
 *   hint?: string, hidden?: boolean}>}} opts
 * @returns {HTMLButtonElement}
 */
export function menuButton(opts) {
  const id = `menu-${++counter}`;
  const button = h('button', {
    type: 'button', class: ['btn', opts.className], 'aria-haspopup': 'menu', 'aria-expanded': 'false',
    'aria-controls': id, 'aria-label': opts.ariaLabel || null, on: { click: () => (open ? close() : show()) },
  }, opts.icon ? icon(opts.icon) : null, opts.label);
  let open = null;

  function close(restore = true) {
    if (!open) return;
    const m = open;
    open = null;
    m.cleanup();
    m.menu.remove();
    button.setAttribute('aria-expanded', 'false');
    if (restore) button.focus({ preventScroll: true });
  }

  function show() {
    const items = (typeof opts.items === 'function' ? opts.items() : opts.items).filter((it) => it && !it.hidden);
    const nodes = items.map((it) => {
      const content = [h('span', { class: 'menu-label' }, it.label), it.hint ? h('span', { class: 'menu-hint' }, it.hint) : null];
      if (it.href && !it.disabled) {
        return h('a', {
          role: 'menuitem', class: 'menu-item', href: it.href, download: it.download === true ? '' : it.download || null,
          tabindex: '-1', on: { click: () => close(false) },
        }, content);
      }
      return h('button', {
        type: 'button', role: 'menuitem', class: 'menu-item', tabindex: '-1', 'aria-disabled': it.disabled ? 'true' : null,
        on: {
          click: () => {
            if (it.disabled) return;
            close();
            if (it.onSelect) it.onSelect(button);
          },
        },
      }, content);
    });
    const menu = h('div', { class: 'menu', role: 'menu', id, 'aria-label': opts.ariaLabel || null }, nodes);
    document.body.appendChild(menu);
    button.setAttribute('aria-expanded', 'true');
    placeUnder(menu, button, 4);
    const focusAt = (i) => {
      if (!nodes.length) return;
      const k = (i + nodes.length) % nodes.length;
      nodes[k].focus();
    };
    const onKey = (e) => {
      const i = nodes.indexOf(document.activeElement);
      if (e.key === 'ArrowDown') { e.preventDefault(); focusAt(i + 1); }
      else if (e.key === 'ArrowUp') { e.preventDefault(); focusAt(i - 1); }
      else if (e.key === 'Home') { e.preventDefault(); focusAt(0); }
      else if (e.key === 'End') { e.preventDefault(); focusAt(nodes.length - 1); }
      else if (e.key === 'Escape') { e.preventDefault(); e.stopPropagation(); close(); }
      else if (e.key === 'Tab') close(false);
    };
    const onPointer = (e) => {
      if (!menu.contains(e.target) && !button.contains(e.target)) close(false);
    };
    const reposition = () => placeUnder(menu, button, 4);
    menu.addEventListener('keydown', onKey);
    document.addEventListener('pointerdown', onPointer, true);
    window.addEventListener('resize', reposition);
    window.addEventListener('scroll', reposition, true);
    open = {
      menu,
      cleanup() {
        document.removeEventListener('pointerdown', onPointer, true);
        window.removeEventListener('resize', reposition);
        window.removeEventListener('scroll', reposition, true);
      },
    };
    focusAt(0);
  }
  return button;
}
