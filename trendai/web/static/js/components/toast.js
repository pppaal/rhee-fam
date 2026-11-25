// Toasts (spec F.6.19). Owner: WP3-app.
// A role="status" message shown for 4 s. It is announced once through #live-polite; the visible toast itself has
// aria-live="off" so screen readers do not read it twice. Kinds: 'info' (✓, a finished action such as "link
// copied"), 'warning' (ⓘ, a refused action such as a 6th series, UX-12) and 'error' (announced assertively).

import { h } from '../dom.js';
import { t } from '../i18n.js';
import { announce, copyText, icon } from '../ui.js';

const DURATION_MS = 4000;
const ICONS = { info: 'ui-check', warning: 'ui-info', error: 'ui-error' };
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
 * @param {{kind?: 'info'|'warning'|'error'}} [opts]
 */
export function toast(text, opts = {}) {
  const kind = ICONS[opts.kind] ? opts.kind : 'info';
  const el = h('div', { class: ['toast', `toast-${kind}`], role: 'status', 'aria-live': 'off' },
    icon(ICONS[kind]), h('span', null, text));
  const box = ensureRegion();
  box.appendChild(el);
  announce(text, kind === 'error');
  setTimeout(() => {
    el.classList.add('is-leaving');
    setTimeout(() => el.remove(), 250);
  }, DURATION_MS);
}

/**
 * Copy `text` and say so: `okKey` (default share.copied) on success, err.clipboard otherwise.
 * @param {string} text
 * @param {string} [okKey]
 * @returns {Promise<boolean>}
 */
export async function copyWithToast(text, okKey = 'share.copied') {
  const ok = await copyText(text);
  toast(ok ? t(okKey) : t('err.clipboard'), { kind: ok ? 'info' : 'error' });
  return ok;
}
