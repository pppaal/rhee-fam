// "현재 — 왜 {phase}인가요?" card (spec F.6.9, D.2.5): reasons ✓ with each `detail` under a 자세히 toggle, the
// certainty label (its % only inside the help.certainty popover), and the rules link. Owner: WP3-app.

import { h } from '../dom.js';
import { t } from '../i18n.js';
import { cardHead, icon, pill } from '../ui.js';
import { helpButton } from './help.js';

let counter = 0;

/**
 * @param {Object} vm  D.2 series VM
 * @returns {HTMLElement}
 */
export function renderPhaseWhy(vm) {
  const { phase } = vm;
  const reasons = (phase.reasons || []).map((r) => {
    let detail = null;
    let toggle = null;
    if (r.detail) {
      const id = `reason-detail-${++counter}`;
      detail = h('p', { class: 'reason-detail', id, hidden: true }, r.detail);
      toggle = h('button', {
        type: 'button', class: 'link-btn small', 'aria-expanded': 'false', 'aria-controls': id,
        on: {
          click: () => {
            detail.hidden = !detail.hidden;
            toggle.setAttribute('aria-expanded', String(!detail.hidden));
          },
        },
      }, t('detail.reason_detail'));
    }
    return h('li', null, icon('ui-check', { cls: 'reason-icon' }),
      h('div', null, h('span', null, r.text), toggle ? [' ', toggle] : null, detail));
  });
  const certaintyHelp = helpButton(phase.certainty.label, () => [
    t('help.certainty'),
    phase.certainty.pct_text ? t('help.certainty_value', { pct: phase.certainty.pct_text }) : null,
  ]);
  return h('section', { class: 'card', id: 'now', 'aria-labelledby': 'now-title' },
    cardHead({ title: t('detail.now.title', { phase: phase.name }), id: 'now-title' }),
    h('ul', { class: 'reason-list prose' }, reasons),
    h('p', { class: 'certainty' },
      pill(phase.certainty.label, `pill-cert-${phase.certainty.level}`), certaintyHelp,
      phase.seasonal_note ? pill(phase.seasonal_note, 'pill-muted') : null),
    h('p', null, h('a', { class: 'more-link', href: phase.rules_url || '/method#phase' }, t('detail.now.rules'))));
}
