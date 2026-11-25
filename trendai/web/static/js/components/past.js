// "과거 — 무슨 일이 있었나요?" card (spec F.6.9, D.2.7): numbered changepoint list ①②③ linked both ways with the chart
// badges (F.7.4), anomaly list and the seasonality line. Owner: WP3-app.

import { h } from '../dom.js';
import { t } from '../i18n.js';
import { DATELINK_HINT_ID } from '../md.js';
import { cardHead } from '../ui.js';
import { circled } from '../chart/geom.js';

/**
 * @param {Object} vm  D.2 series VM
 * @param {{onChangepoint?: (n: number) => void, onDate?: (iso: string) => void}} [opts]
 * @returns {{el: HTMLElement, highlight: (n: number|null) => void}}
 */
export function createPast(vm, opts = {}) {
  const cps = (vm.chart && vm.chart.changepoints) || [];
  const anomalies = (vm.chart && vm.chart.anomalies) || [];
  const items = new Map();
  const cpList = cps.length
    ? h('ol', { class: 'cp-list' }, cps.map((cp) => {
      const btn = h('button', {
        type: 'button', class: 'cp-item', dataset: { n: cp.n }, 'aria-pressed': 'false',
        on: { click: () => { highlight(cp.n); if (opts.onChangepoint) opts.onChangepoint(cp.n); } },
      }, h('span', { class: 'cp-num', 'aria-hidden': 'true' }, circled(cp.n)), h('span', null, cp.text));
      items.set(cp.n, btn);
      return h('li', null, btn);
    }))
    : h('p', { class: 'muted' }, t('detail.past.none_cp'));
  const anList = anomalies.length
    ? h('ul', { class: 'anomaly-list' }, anomalies.map((a) => h('li', null,
      h('button', {
        type: 'button', class: 'link-btn anomaly-date', dataset: { date: a.date }, 'aria-describedby': DATELINK_HINT_ID,
        on: { click: () => opts.onDate && opts.onDate(a.date) },
      }, a.date_label),
      h('span', null, ` ${a.text}`))))
    : h('p', { class: 'muted' }, t('detail.past.none_an'));

  function highlight(n) {
    for (const [k, btn] of items) {
      const on = k === n;
      btn.classList.toggle('is-active', on);
      btn.setAttribute('aria-pressed', String(on));
    }
  }

  const el = h('section', { class: 'card', id: 'past', 'aria-labelledby': 'past-title' },
    cardHead({ title: t('detail.past.title'), id: 'past-title' }),
    vm.past && vm.past.summary ? h('p', { class: 'muted' }, vm.past.summary) : null,
    h('div', { class: 'past-grid' },
      h('div', null, h('h3', { class: 'sub-title' }, t('detail.past.changepoints')), cpList),
      h('div', null, h('h3', { class: 'sub-title' }, t('detail.past.anomalies')), anList)),
    vm.past && vm.past.seasonality ? h('p', { class: 'seasonality' }, vm.past.seasonality.text) : null);
  return {
    el,
    highlight(n) {
      highlight(n);
      const btn = items.get(n);
      if (btn) btn.scrollIntoView({ block: 'nearest' });
    },
  };
}
