// Not-found page (spec F.2: "anything else → not found"; G.9 empty state). Owner: WP3-app.

import { h } from '../dom.js';
import { t } from '../i18n.js';
import { emptyState } from '../components/errorcard.js';

export function create(ctx) {
  ctx.setTitle(`${t('title.notfound')} · trendai`);
  return {
    el: h('div', { class: 'page page-narrow' },
      emptyState({
        title: t('empty.notfound.title'), body: t('empty.notfound.body'), level: 1, icon: 'ui-search',
        actions: [h('a', { class: 'btn btn-primary', href: '/' }, t('fix.go_home'))],
      })),
  };
}
