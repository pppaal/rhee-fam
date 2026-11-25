// Detail hero (spec F.6.9, D.2.3, brief §3.5): big phase badge with a 왜? toggle, outlook line, sentence, reliability
// chip + accuracy line, and the reference-note chip of weak forecasts. Owner: WP3-app.

import { h } from '../dom.js';
import { t } from '../i18n.js';
import { arrowLine, disclosure, icon, phaseBadge, pill, toneClass } from '../ui.js';
import { helpButton } from './help.js';

let counter = 0;

/**
 * @param {Object} vm  D.2 series VM
 * @returns {HTMLElement}
 */
export function renderHero(vm) {
  const id = `why-${++counter}`;
  const { hero, phase } = vm;
  const why = h('div', { class: 'hero-why', id, hidden: true },
    h('ul', { class: 'reason-list' }, (phase.reasons || []).map((r) => h('li', null,
      icon('ui-check', { cls: 'reason-icon' }), h('span', null, r.text)))),
    h('p', { class: 'muted small' }, phase.certainty.label, ' · ',
      h('a', { href: phase.rules_url || '/method#phase' }, t('detail.now.rules'))));
  const whyBtn = disclosure(h('button', { type: 'button', class: 'link-btn hero-why-btn' },
    t('detail.why'), icon('ui-chevron-down', { cls: 'chev' })), why);

  return h('section', { class: 'card hero', 'aria-label': hero.title },
    h('div', { class: 'hero-top' },
      phaseBadge({ ...phase, tentative: phase.tentative }, { big: true }),
      whyBtn,
      phase.seasonal_note ? pill(phase.seasonal_note, 'pill-muted') : null),
    why,
    h('p', { class: ['hero-outlook', toneClass(hero.outlook.tone)] }, arrowLine(hero.outlook.line)),
    h('p', { class: 'hero-sentence' }, hero.sentence),
    h('div', { class: 'hero-meta' },
      h('span', { class: 'hero-meta-item' },
        pill(hero.reliability.label, `pill-rel-${hero.reliability.level}`),
        helpButton(hero.reliability.label, t('help.reliability'))),
      h('span', { class: 'hero-meta-item' },
        h('span', null, hero.accuracy_line),
        helpButton(t('detail.future.accuracy'), t('help.accuracy'))),
      hero.reference_note ? pill(hero.reference_note, 'pill-warn') : null));
}
