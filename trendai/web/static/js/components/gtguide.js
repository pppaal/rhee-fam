// Google Trends guide card (spec F.6.10, G.5, M12). Owner: WP3-app.

import { h } from '../dom.js';
import { t } from '../i18n.js';

/** The guide card: title, the three steps and the note on values and attribution. */
export function gtGuide() {
  return h('section', { class: 'card gt-guide', 'aria-labelledby': 'gt-guide-title' },
    h('h2', { class: 'card-title', id: 'gt-guide-title' }, t('gt.title')),
    h('p', null, t('gt.steps')),
    h('p', { class: 'muted' }, t('gt.note')));
}
