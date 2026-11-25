// "미래 — 얼마나 확실한가요?" card (spec F.6.9, D.2.8): scenario box headed by scenarios.title (reference row, the three
// cases, note), collapsible forecast table, accuracy block with 자세히 (details + model list), method sentence,
// future phase and peak. Owner: WP3-app.

import { h } from '../dom.js';
import { t } from '../i18n.js';
import { cardHead, pill } from '../ui.js';
import { helpButton } from './help.js';

let counter = 0;

/**
 * @param {Object} vm  D.2 series VM
 * @returns {HTMLElement}
 */
export function renderFuture(vm) {
  const f = vm.future || {};
  const sc = f.scenarios;
  const scenarios = sc ? h('div', { class: 'scenarios' },
    h('h3', { class: 'sub-title' }, sc.title),
    sc.reference ? h('p', { class: 'scenario-ref' }, sc.reference.text) : null,
    h('ul', { class: 'scenario-rows' }, (sc.rows || []).map((r) => h('li', { class: ['scenario', `scenario-${r.key}`] },
      h('span', { class: 'scenario-label', 'aria-hidden': 'true' }, r.label),
      h('span', { class: 'scenario-value', 'aria-hidden': 'true' }, r.value_text),
      h('span', { class: 'scenario-pct', 'aria-hidden': 'true' }, r.pct_text || ''),
      h('span', { class: 'sr-only' }, r.text)))),
    sc.note ? h('p', { class: 'muted small' }, sc.note) : null) : null;

  const cols = t('detail.future.cols').split(',');
  const rows = (f.table && f.table.rows) || [];
  const table = rows.length ? h('details', { class: 'disclosure' },
    h('summary', null, t('detail.future.table')),
    h('div', { class: 'table-scroll', tabindex: '0', role: 'region', 'aria-label': t('detail.future.table') },
      h('table', { class: 'data-table data-table-cards' },
        h('thead', null, h('tr', null, cols.map((c) => h('th', { scope: 'col' }, c)))),
        h('tbody', null, rows.map((r) => h('tr', null,
          h('th', { scope: 'row', 'data-label': cols[0] }, r.date_label),
          h('td', { 'data-label': cols[1] }, r.mean),
          h('td', { 'data-label': cols[2] }, r.range80),
          h('td', { 'data-label': cols[3] }, r.range95))))))) : null;

  const acc = f.accuracy || {};
  const detailsId = `acc-details-${++counter}`;
  const models = (f.models || []).length ? [
    h('h4', { class: 'sub-sub-title' }, t('detail.future.models')),
    h('ul', { class: 'model-list' }, f.models.map((m) => h('li', null, h('span', null, m.name),
      h('span', { class: 'model-weight' }, m.weight_text)))),
  ] : null;
  const accDetails = h('div', { class: 'acc-details', id: detailsId, hidden: true },
    acc.details ? h('p', { class: 'muted small' }, acc.details) : null, models);
  const accToggle = (acc.details || models) ? h('button', {
    type: 'button', class: 'link-btn small', 'aria-expanded': 'false', 'aria-controls': detailsId,
    on: {
      click: () => {
        accDetails.hidden = !accDetails.hidden;
        accToggle.setAttribute('aria-expanded', String(!accDetails.hidden));
      },
    },
  }, t('common.details')) : null;

  return h('section', { class: 'card', id: 'future', 'aria-labelledby': 'future-title' },
    cardHead({ title: t('detail.future.title'), id: 'future-title' }),
    scenarios,
    table,
    h('div', { class: 'future-block' },
      h('h3', { class: 'sub-title' }, t('detail.future.accuracy'), helpButton(t('detail.future.accuracy'), t('help.accuracy'))),
      h('p', null, acc.line || vm.hero.accuracy_line, accToggle ? [' ', accToggle] : null),
      accDetails),
    f.method ? h('div', { class: 'future-block' },
      h('h3', { class: 'sub-title' }, t('detail.future.method')),
      h('p', null, f.method)) : null,
    f.phase ? h('p', { class: 'future-line' }, f.phase.hedge ? [pill(f.phase.hedge, 'pill-muted'), ' '] : null, f.phase.text) : null,
    f.peak ? h('p', { class: 'future-line' }, f.peak.text) : null);
}
