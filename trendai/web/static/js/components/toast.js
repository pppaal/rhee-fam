// Toasts (spec F.6.19). Owner: WP3-app.
// A role="status" message shown for 4 s. It is announced once through #live-polite; the visible toast itself has
// aria-live="off" so screen readers do not read it twice.

import { h } from '../dom.js';
import { announce, icon } from '../ui.js';

const DURATION_MS = 4000;
let region = null;

function ensureRegion() {
  if (region && region.isConnected) return region;
  region = h('div', { class: 'toasts' });
  document.body.appendChild(region);
  return region;
}

/**
 * Show a toast.
 * @param {string} text
 * @param {{kind?: 'info'|'error'}} [opts]
 */
export function toast(text, opts = {}) {
  const el = h('div', { class: ['toast', opts.kind === 'error' && 'toast-error'], role: 'status', 'aria-live': 'off' },
    opts.kind === 'error' ? icon('ui-error') : icon('ui-check'), h('span', null, text));
  const box = ensureRegion();
  box.appendChild(el);
  announce(text, opts.kind === 'error');
  setTimeout(() => {
    el.classList.add('is-leaving');
    setTimeout(() => el.remove(), 250);
  }, DURATION_MS);
}
