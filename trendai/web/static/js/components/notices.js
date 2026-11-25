// Notice list (spec F.6.13, C.12.5). Owner: WP3-app.
// `warning` notices get a ⚠ icon, `notice` / `info` an ⓘ icon; icons are hidden from screen readers and paired with
// visually hidden text (a11y.warning / a11y.info, F.9 item 4). Fix buttons behave as in error cards.

import { h } from '../dom.js';
import { t } from '../i18n.js';
import { icon, srOnly } from '../ui.js';
import { fixRow, localFixes } from './errorcard.js';

/**
 * @param {Array<{code: string, level: string, text: string, fixes?: Array}>} notices
 * @param {import('./errorcard.js').FixContext} ctx
 * @param {{className?: string}} [opts]
 * @returns {HTMLElement|null}
 */
export function noticeList(notices, ctx, opts = {}) {
  const list = (notices || []).filter((n) => n && n.text);
  if (!list.length) return null;
  return h('ul', { class: ['notices', opts.className] }, list.map((n) => {
    const warn = n.level === 'warning';
    return h('li', { class: ['notice', warn ? 'notice-warning' : 'notice-info'], dataset: { code: n.code } },
      icon(warn ? 'ui-warning' : 'ui-info', { cls: 'notice-icon' }),
      h('div', { class: 'notice-body' },
        h('p', null, srOnly(`${t(warn ? 'a11y.warning' : 'a11y.info')}: `), n.text),
        fixRow(localFixes(n.fixes), ctx, { primary: false })));
  }));
}
