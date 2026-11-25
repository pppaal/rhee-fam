// Footer (spec F.6.18, G.7): disclaimers, links, 광고 없음, version. Owner: WP3-app.

import { clear, h } from '../dom.js';
import { t } from '../i18n.js';

/**
 * @param {HTMLElement} root  #footer
 * @param {{version: string, onSettings: (opener: HTMLElement) => void}} opts
 */
export function renderFooter(root, opts) {
  clear(root);
  const settings = h('button', { type: 'button', class: 'link-btn', on: { click: (e) => opts.onSettings(e.currentTarget) } },
    t('footer.links.settings'));
  root.appendChild(h('div', { class: 'footer-inner' },
    h('div', { class: 'footer-text' },
      h('p', null, t('footer.disclaimer')),
      h('p', null, t('footer.not_advice'))),
    h('nav', { class: 'footer-links', 'aria-label': t('footer.nav') },
      h('a', { href: '/method' }, t('footer.links.method')),
      h('a', { href: '/sources' }, t('footer.links.sources')),
      h('a', { href: '/privacy' }, t('footer.links.privacy')),
      settings),
    h('p', { class: 'footer-meta' },
      h('span', null, t('footer.no_ads')),
      opts.version ? h('span', null, t('footer.version', { version: opts.version })) : null)));
}
