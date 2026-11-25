// Table views (spec F.6.8, F.7.1, F.9 item 5): every chart has one. Below 640 px the tables render as stacked cards
// (CSS card mode); wider tables scroll inside their own container, never the page. Values are formatted client-side
// with fmt.js (D.6). No copy-to-clipboard control exists here (A.3). Owner: WP3-app.
//
// Card mode keeps the table semantics (A11Y-15): every element carries its explicit ARIA role (table / rowgroup / row
// / columnheader / rowheader / cell), because restyled table elements lose them in WebKit, and the per-cell label
// shown in card mode is a real aria-hidden span (not CSS generated content), so it is never read twice with the
// column header. Empty cells ('—') are hidden in card mode (UX-32).

import { h } from '../dom.js';
import { DASH, dateLabel, valueText } from '../fmt.js';
import { t } from '../i18n.js';
import { exportHref, swatch } from '../ui.js';

const isNum = (v) => typeof v === 'number' && Number.isFinite(v);

/**
 * A body cell of a card-mode table: a row header (`th`) or a data cell (`td`) with its column label.
 * @param {'th'|'td'} tag
 * @param {string} label  the column header (shown before the value in card mode only)
 * @param {Node|string} content
 * @param {Object} [attrs]
 */
export function cardCell(tag, label, content, attrs = {}) {
  const empty = content == null || content === '' || content === DASH;
  return h(tag, {
    ...attrs, role: tag === 'th' ? 'rowheader' : 'cell', scope: tag === 'th' ? 'row' : null,
    class: [attrs.class, empty && 'is-empty'],
  }, h('span', { class: 'cell-label', 'aria-hidden': 'true' }, label), h('span', { class: 'cell-value' }, content));
}

/** A row of a card-mode table. */
export function cardRow(cells, attrs = {}) {
  return h('tr', { ...attrs, role: 'row' }, cells);
}

/**
 * A table (card mode below 640 px unless opts.cards === false) in its own scroll region.
 * @param {string} caption
 * @param {string[]} cols
 * @param {Node[]} rows  cardRow()s
 * @param {{cards?: boolean, className?: string, regionLabel?: string}} [opts]
 */
export function table(caption, cols, rows, opts = {}) {
  return h('div', { class: 'table-scroll', tabindex: '0', role: 'region', 'aria-label': opts.regionLabel || caption },
    h('table', { class: ['data-table', opts.cards !== false && 'data-table-cards', opts.className], role: 'table' },
      h('caption', { class: 'sr-only' }, caption),
      h('thead', { role: 'rowgroup' }, h('tr', { role: 'row' }, cols.map((c) => h('th', { scope: 'col', role: 'columnheader' }, c)))),
      h('tbody', { role: 'rowgroup' }, rows)));
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
  const fmt = (v) => (isNum(v) ? valueText(v, kind, lang) + (unit ? ` ${unit}` : '') : DASH);
  const range = (lo, hi) => (isNum(lo) && isNum(hi) ? `${valueText(lo, kind, lang)}${lang === 'en' ? '–' : '~'}${valueText(hi, kind, lang)}` : DASH);
  const cols = t('detail.table.cols').split(',');
  const rows = [];
  (c.dates || []).forEach((iso, i) => {
    rows.push(cardRow([
      cardCell('th', cols[0], dateLabel(iso, c.freq, lang)),
      cardCell('td', cols[1], fmt(c.actual[i])),
      cardCell('td', cols[2], fmt(c.trend ? c.trend[i] : null)),
      cardCell('td', cols[3], DASH),
      cardCell('td', cols[4], DASH),
      cardCell('td', cols[5], DASH)]));
  });
  const f = c.forecast || {};
  const bandsOnly = c.display === 'bands_only';
  (f.dates || []).forEach((iso, j) => {
    rows.push(cardRow([
      cardCell('th', cols[0], dateLabel(iso, c.freq, lang)),
      cardCell('td', cols[1], DASH),
      cardCell('td', cols[2], DASH),
      cardCell('td', cols[3], bandsOnly ? DASH : fmt(f.mean[j])),
      cardCell('td', cols[4], range(f.lo80[j], f.hi80[j])),
      cardCell('td', cols[5], range(f.lo95[j], f.hi95[j]))], { class: 'is-forecast' }));
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
  return table(t('compare.title'), cols, rows.map((r) => cardRow([
    cardCell('td', cols[0], String(r.rank)),
    cardCell('th', cols[1], h('span', { class: 'name-cell' }, swatch(r.slot), r.name)),
    cardCell('td', cols[2], r.phase.name),
    cardCell('td', cols[3], r.expected_change.text),
    cardCell('td', cols[4], r.outlook.prob_text),
    cardCell('td', cols[5], r.recent_change.text),
    cardCell('td', cols[6], r.momentum.text),
    cardCell('td', cols[7], r.score.text)])), { className: 'ranking-table' });
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
    h('th', { scope: 'row' }, dateLabel(iso, freq, lang)),
    vms.map((vm, k) => {
      const v = maps[k].get(iso);
      return h('td', null, isNum(v) ? valueText(v, vm.chart.value_kind, lang) : '');
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
