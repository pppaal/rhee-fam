// Anchored popover and menu button (spec F.6.4, F.6.6). Owner: WP3-app.
//
// openPopover(): from 640 px a panel anchored under its opener (role="dialog", aria-modal, labelled, focus trap, Esc
// and outside click close, focus returns to the opener); below 640 px the same content opens as a bottom sheet.
// menuButton(): a button with aria-haspopup="menu" opening a role="menu" list (arrow keys, Home/End, Esc; Tab closes
// the menu and continues from the button). Every open panel registers with sheet.js so a route change closes it.

import { focusFirst, h } from '../dom.js';
import { t } from '../i18n.js';
import { icon, isNarrow } from '../ui.js';
import { openSheet, registerModal, registerTransient, resolveOpener } from './sheet.js';

let counter = 0;
const GUTTER = 8;

/**
 * The position of a fixed panel of size `pw` × `ph` under (or above) an anchor rect, kept inside the viewport.
 * Pure (exported for tests).
 * @param {{left: number, top: number, bottom: number}} r  anchor rect
 * @param {number} pw
 * @param {number} ph
 * @param {number} vw  viewport width
 * @param {number} vh  viewport height
 * @param {number} [gap=6]
 * @returns {{left: number, top: number}}
 */
export function panelPosition(r, pw, ph, vw, vh, gap = 6) {
  let left = Math.min(Math.max(GUTTER, r.left), Math.max(GUTTER, vw - pw - GUTTER));
  if (!Number.isFinite(left)) left = GUTTER;
  let top = r.bottom + gap;
  if (top + ph > vh - GUTTER) {
    if (r.top - gap - ph >= GUTTER) top = r.top - gap - ph;              // room above: flip
    else top = Math.max(GUTTER, vh - ph - GUTTER);                       // neither: keep it on screen
  }
  return { left: Math.round(left), top: Math.round(top) };
}

/**
 * Position a fixed panel under (or above) `anchor`, kept inside the viewport. The panel is moved to 0,0 before it is
 * measured, so its width is its natural width and not the space left at its static position (UX-2).
 * @param {HTMLElement} panel
 * @param {HTMLElement} anchor
 * @param {number} [gap=6]
 * @param {{gutters?: boolean}} [opts]  gutters: below 640 px span the page between the 16 px gutters
 */
export function placeUnder(panel, anchor, gap = 6, opts = {}) {
  if (!anchor || !anchor.isConnected) return;
  const r = anchor.getBoundingClientRect();
  const vw = document.documentElement.clientWidth;
  const vh = window.innerHeight;
  panel.style.setProperty('left', '0px');
  panel.style.setProperty('top', '0px');
  if (opts.gutters && isNarrow()) {
    panel.style.setProperty('width', `${Math.max(0, vw - 32)}px`);
    panel.style.setProperty('max-width', 'none');
  } else {
    panel.style.removeProperty('width');
    panel.style.removeProperty('max-width');
  }
  const pos = panelPosition(r, panel.offsetWidth, panel.offsetHeight, vw, vh, gap);
  if (opts.gutters && isNarrow()) pos.left = 16;
  panel.style.setProperty('left', `${pos.left}px`);
  panel.style.setProperty('top', `${pos.top}px`);
}

/**
 * Open an anchored popover (bottom sheet below 640 px).
 * @param {HTMLElement} anchor
 * @param {{title: string, content: Node|Node[], footer?: Node|Node[], onClose?: () => void, className?: string,
 *          opener?: HTMLElement|(() => HTMLElement|null)}} opts
 *        opener: where focus returns (default: the anchor); a function is resolved at close time
 * @returns {{close: (restoreFocus?: boolean) => void, el: HTMLElement}}
 */
export function openPopover(anchor, opts) {
  const opener = opts.opener || anchor;
  if (isNarrow()) {
    const sheet = openSheet({ ...opts, opener });
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
  const reposition = () => {
    if (!anchor.isConnected) close(false);
    else placeUnder(panel, anchor);
  };
  const onPointer = (e) => {
    if (!panel.contains(e.target) && !anchor.contains(e.target)) close(false);
  };
  const unregister = registerModal({ dialog: panel, close: (restore) => close(restore) });
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
    const target = restoreFocus ? resolveOpener(opener) : null;
    if (target) target.focus({ preventScroll: true });
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
  const n = ++counter;
  const id = `menu-${n}`;
  const buttonId = `menu-btn-${n}`;
  const button = h('button', {
    type: 'button', id: buttonId, class: ['btn', opts.className], 'aria-haspopup': 'menu', 'aria-expanded': 'false',
    'aria-label': opts.ariaLabel || null, on: { click: () => (open ? close() : show()) },
  }, opts.icon ? icon(opts.icon) : null, opts.label);
  let open = null;

  function close(restore = true) {
    if (!open) return;
    const m = open;
    open = null;
    m.cleanup();
    m.menu.remove();
    button.setAttribute('aria-expanded', 'false');
    button.removeAttribute('aria-controls');
    if (restore && button.isConnected) button.focus({ preventScroll: true });
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
    const menu = h('div', { class: 'menu', role: 'menu', id, 'aria-labelledby': buttonId }, nodes);
    document.body.appendChild(menu);
    button.setAttribute('aria-expanded', 'true');
    button.setAttribute('aria-controls', id);
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
      // Tab: focus goes back to the button first, so the browser's own Tab move continues from there (A11Y-7)
      else if (e.key === 'Tab') close(true);
    };
    const onPointer = (e) => {
      if (!menu.contains(e.target) && !button.contains(e.target)) close(false);
    };
    const reposition = () => {
      if (!button.isConnected) close(false);
      else placeUnder(menu, button, 4);
    };
    menu.addEventListener('keydown', onKey);
    document.addEventListener('pointerdown', onPointer, true);
    window.addEventListener('resize', reposition);
    window.addEventListener('scroll', reposition, true);
    const unregister = registerTransient({ close: (restore) => close(restore) });
    open = {
      menu,
      cleanup() {
        unregister();
        document.removeEventListener('pointerdown', onPointer, true);
        window.removeEventListener('resize', reposition);
        window.removeEventListener('scroll', reposition, true);
      },
    };
    focusAt(0);
  }
  return button;
}
