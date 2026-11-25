// "미래 — 얼마나 확실한가요?" card (spec F.6.9, D.2.8): scenario box headed by scenarios.title (reference row, the three
// cases, note), collapsible forecast table, accuracy block with 자세히 (details + model list), method sentence,
// future phase and peak. Owner: WP3-app.

import { h } from '../dom.js';
import { t } from '../i18n.js';
import { cardHead, disclosure, pill } from '../ui.js';
import { cardCell, cardRow, table as dataTable } from './datatable.js';
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

  // bands_only (D.2.8): the centre value is withheld ('—' in every row), so the column is left out, as in the
  // chart's data table
  const noMean = vm.chart && vm.chart.display === 'bands_only';
  const allCols = t('detail.future.cols').split(',');
  const cols = noMean ? [allCols[0], allCols[2], allCols[3]] : allCols;
  const rows = (f.table && f.table.rows) || [];
  const table = rows.length ? h('details', { class: 'disclosure' },
    h('summary', null, t('detail.future.table')),
    dataTable(t('detail.future.table'), cols, rows.map((r) => cardRow(noMean
      ? [cardCell('th', cols[0], r.date_label), cardCell('td', cols[1], r.range80), cardCell('td', cols[2], r.range95)]
      : [cardCell('th', cols[0], r.date_label), cardCell('td', cols[1], r.mean), cardCell('td', cols[2], r.range80),
        cardCell('td', cols[3], r.range95)])))) : null;

  const acc = f.accuracy || {};
  const detailsId = `acc-details-${++counter}`;
  const models = (f.models || []).length ? [
    h('h4', { class: 'sub-sub-title' }, t('detail.future.models')),
    h('ul', { class: 'model-list' }, f.models.map((m) => h('li', null, h('span', null, m.name),
      h('span', { class: 'model-weight' }, m.weight_text)))),
  ] : null;
  const accDetails = h('div', { class: 'acc-details', id: detailsId, hidden: true },
    acc.details ? h('p', { class: 'muted small' }, acc.details) : null, models);
  const accToggle = (acc.details || models)
    ? disclosure(h('button', { type: 'button', class: 'link-btn small' }, t('common.details')), accDetails) : null;

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
