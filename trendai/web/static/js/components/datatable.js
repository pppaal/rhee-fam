// Table views (spec F.6.8, F.7.1, F.9 item 5): every chart has one. Below 640 px the tables render as stacked cards
// (CSS card mode, each cell labelled through data-label); wider tables scroll inside their own container, never the
// page. Values are formatted client-side with fmt.js (D.6). No copy-to-clipboard control exists here (A.3).
// Owner: WP3-app.

import { h } from '../dom.js';
import { dateLabel, valueText } from '../fmt.js';
import { t } from '../i18n.js';
import { exportHref, swatch } from '../ui.js';

const isNum = (v) => typeof v === 'number' && Number.isFinite(v);

function table(caption, cols, rows, opts = {}) {
  return h('div', { class: 'table-scroll', tabindex: '0', role: 'region', 'aria-label': caption },
    h('table', { class: ['data-table', opts.cards !== false && 'data-table-cards', opts.className] },
      h('caption', { class: 'sr-only' }, caption),
      h('thead', null, h('tr', null, cols.map((c) => h('th', { scope: 'col' }, c)))),
      h('tbody', null, rows)));
}

/**
 * Every history and forecast row of one series (the fan chart's table view).
 * @param {Object} vm  D.2 series VM
 * @param {'ko'|'en'} lang
 */
export function seriesTable(vm, lang) {
  const c = vm.chart;
  const kind = c.value_kind || vm.source.value_kind;
  const unit = vm.source.unit_label;
  const fmt = (v) => (isNum(v) ? valueText(v, kind, lang) + (unit ? ` ${unit}` : '') : '—');
  const range = (lo, hi) => (isNum(lo) && isNum(hi) ? `${valueText(lo, kind, lang)}${lang === 'en' ? '–' : '~'}${valueText(hi, kind, lang)}` : '—');
  const cols = t('detail.table.cols').split(',');
  const rows = [];
  (c.dates || []).forEach((iso, i) => {
    rows.push(h('tr', null,
      h('th', { scope: 'row', 'data-label': cols[0] }, dateLabel(iso, c.freq, lang)),
      h('td', { 'data-label': cols[1] }, fmt(c.actual[i])),
      h('td', { 'data-label': cols[2] }, fmt(c.trend ? c.trend[i] : null)),
      h('td', { 'data-label': cols[3] }, '—'),
      h('td', { 'data-label': cols[4] }, '—'),
      h('td', { 'data-label': cols[5] }, '—')));
  });
  const f = c.forecast || {};
  const bandsOnly = c.display === 'bands_only';
  (f.dates || []).forEach((iso, j) => {
    rows.push(h('tr', { class: 'is-forecast' },
      h('th', { scope: 'row', 'data-label': cols[0] }, dateLabel(iso, c.freq, lang)),
      h('td', { 'data-label': cols[1] }, '—'),
      h('td', { 'data-label': cols[2] }, '—'),
      h('td', { 'data-label': cols[3] }, bandsOnly ? '—' : fmt(f.mean[j])),
      h('td', { 'data-label': cols[4] }, range(f.lo80[j], f.hi80[j])),
      h('td', { 'data-label': cols[5] }, range(f.lo95[j], f.hi95[j]))));
  });
  return table(c.title || vm.name, cols, rows, { className: 'series-table' });
}

/**
 * The ranking table of a comparison (rank, name, phase, expected change, chance of rise, recent change, momentum,
 * score), in the given row order.
 * @param {Array<Object>} rows  compare.ranking rows (already sorted)
 */
export function rankingTable(rows) {
  const cols = t('compare.table.cols').split(',');
  return table(t('compare.title'), cols, rows.map((r) => h('tr', null,
    h('td', { 'data-label': cols[0] }, String(r.rank)),
    h('th', { scope: 'row', 'data-label': cols[1] }, h('span', { class: 'name-cell' }, swatch(r.slot), r.name)),
    h('td', { 'data-label': cols[2] }, r.phase.name),
    h('td', { 'data-label': cols[3] }, r.expected_change.text),
    h('td', { 'data-label': cols[4] }, r.outlook.prob_text),
    h('td', { 'data-label': cols[5] }, r.recent_change.text),
    h('td', { 'data-label': cols[6] }, r.momentum.text),
    h('td', { 'data-label': cols[7] }, r.score.text))), { className: 'ranking-table' });
}

/**
 * The values table of a comparison: date × series history values on the union of dates (empty where a series has
 * no value). Its CSV link points to the server export (`exports.values_csv`); for Naver the link is absent and the
 * unavailable text is shown instead (F.6.8).
 * @param {Object[]} vms  successful series VMs, request order
 * @param {{values_csv?: string|null, unavailable?: Object}} exports  result-level exports (D.5)
 * @param {Object} slots  compare.slots
 * @param {'ko'|'en'} lang
 */
export function valuesTable(vms, exports, slots, lang) {
  const dates = new Set();
  const maps = vms.map((vm) => {
    const m = new Map();
    (vm.chart.dates || []).forEach((iso, i) => {
      dates.add(iso);
      m.set(iso, vm.chart.actual[i]);
    });
    return m;
  });
  const sorted = [...dates].sort();
  const freq = vms[0] ? vms[0].chart.freq : 'D';
  const cols = [t('compare.values_date'), ...vms.map((vm) => vm.name)];
  const rows = sorted.map((iso) => h('tr', null,
    h('th', { scope: 'row', 'data-label': cols[0] }, dateLabel(iso, freq, lang)),
    vms.map((vm, k) => {
      const v = maps[k].get(iso);
      return h('td', { 'data-label': vm.name }, isNum(v) ? valueText(v, vm.chart.value_kind, lang) : '');
    })));
  const csv = exports && exports.values_csv;
  const unavailable = exports && exports.unavailable && exports.unavailable.values_csv;
  const headCells = cols.map((c, i) => (i === 0 ? c : h('span', { class: 'name-cell' }, swatch(slots ? slots[String(vms[i - 1].index)] : null), c)));
  return h('div', { class: 'values-table' },
    h('div', { class: 'values-table-head' },
      h('h3', { class: 'sub-title' }, t('compare.values_table')),
      csv ? h('a', { class: 'btn btn-secondary btn-small', href: exportHref(csv, lang), download: '' }, t('compare.download_csv'))
        : (unavailable ? h('p', { class: 'muted small unavailable-note' }, unavailable.text) : null)),
    h('div', { class: 'table-scroll table-scroll-tall', tabindex: '0', role: 'region', 'aria-label': t('compare.values_table') },
      h('table', { class: 'data-table values' },
        h('thead', null, h('tr', null, headCells.map((c) => h('th', { scope: 'col' }, c)))),
        h('tbody', null, rows))));
}
