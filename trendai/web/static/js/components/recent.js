// Recent analyses of this browser (spec F.6.20): trendai.recent, newest first, each with a × remove button; entries
// whose expires_at has passed are greyed with `recent.expired` and no link. Owner: WP3-app. After a removal focus moves
// to the next entry's ×, else the previous one, else the section heading (A11Y-4).

import { clear, h } from '../dom.js';
import { t } from '../i18n.js';
import * as store from '../store.js';
import { icon } from '../ui.js';

/**
 * Render the recent list into `root` (re-renders itself after a removal).
 * @param {HTMLElement} root
 * @param {{headingLevel?: number}} [opts]
 */
export function renderRecent(root, opts = {}) {
  clear(root);
  const list = store.getRecent();
  const tag = `h${opts.headingLevel || 2}`;
  root.appendChild(h(tag, { class: 'section-title' }, t('home.recent')));
  if (!list.length) {
    root.appendChild(h('p', { class: 'muted' }, t('empty.recent_none'), ' ',
      h('a', { href: '/samples' }, t('nav.samples'))));
    return;
  }
  const now = Date.now();
  root.appendChild(h('ul', { class: 'recent-list' }, list.map((entry, i) => {
    const expired = entry.expires_at && Date.parse(entry.expires_at) < now;
    const remove = h('button', {
      type: 'button', class: 'icon-btn recent-x', 'aria-label': `${t('home.recent.remove')}: ${entry.title}`,
      on: {
        click: () => {
          store.removeRecent(entry.url);
          renderRecent(root, opts);
          const xs = root.querySelectorAll('.recent-x');
          const target = xs[Math.min(i, xs.length - 1)] || root.querySelector('.section-title');
          if (target) {
            if (!xs.length) target.setAttribute('tabindex', '-1');
            target.focus();
          }
        },
      },
    }, icon('ui-close'));
    return h('li', { class: ['recent-item', expired && 'is-expired'] },
      expired
        ? h('span', { class: 'recent-title' }, entry.title, h('span', { class: 'recent-note' }, t('recent.expired')))
        : h('a', { class: 'recent-title', href: entry.url }, entry.title),
      remove);
  })));
}
