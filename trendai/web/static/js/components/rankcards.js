// Ranking cards (spec F.6.8, D.4.1): rank, swatch + icon + name, phase badge (icon + text), outlook line,
// `예측 변화 {text} ({basis})`, `최근 변화 {text} ({basis})`, sparkline, reliability chip, 참고용 chip, the Naver
// low-resolution warning with `이 검색어만 다시 분석`, and `자세히 ›` (→ focus). Per-series failures render as error
// cards in the same grid. Hover / focus on a card focuses the series on the chart. Owner: WP3-app.

import { h } from '../dom.js';
import { DASH, pct } from '../fmt.js';
import { t } from '../i18n.js';
import { sparkline } from '../chart/sparkline.js';
import { arrowLine, phaseBadge, pill, swatch, toneClass } from '../ui.js';
import { ApiError } from '../api.js';
import { errorCard, fixRow, localFixes } from './errorcard.js';

/** Client-side sorts of the ranking rows (`sort.*`, nulls last, highest first). */
export const SORT_KEYS = Object.freeze(['score', 'prob', 'expected', 'recent', 'level']);

export function sortRows(rows, key) {
  if (key === 'score') return rows.slice();
  const val = (r) => (r.sort && typeof r.sort[key] === 'number' && Number.isFinite(r.sort[key]) ? r.sort[key] : null);
  return rows.slice().sort((a, b) => {
    const va = val(a);
    const vb = val(b);
    if (va == null && vb == null) return a.rank - b.rank;
    if (va == null) return 1;
    if (vb == null) return -1;
    return vb - va || a.rank - b.rank;
  });
}

/** Season length per data frequency (the sparkline history of D.4.1 is min(len, 2 × season, 104) points). */
const SEASON = { D: 7, W: 52, M: 12, Q: 4, Y: 1 };

/** The ranking text of a change tile: signed % like the server's ranking rows (D.4.1), '—' when undefined. */
function changeText(tile) {
  if (!tile || tile.undefined_reason) return DASH;
  return Number.isFinite(tile.value_raw) ? pct(tile.value_raw) : String(tile.value || DASH).replace(/^▲/, '+').replace(/^▼/, '−');
}

/**
 * A provisional row built from a series VM while the job still runs (no rank and no score yet). It mirrors the
 * server's ranking rows (D.4.1), so a card does not change when the final ranking replaces it (CON-15): signed %
 * texts and a sparkline of min(len, 2 × season, 104) points.
 */
export function provisionalRow(vm) {
  const tile = (key) => (vm.tiles || []).find((x) => x.key === key) || {};
  const actual = vm.chart.actual || [];
  const season = SEASON[vm.chart.freq] || 12;
  const k = Math.min(actual.length, 2 * season, 104);
  const history = k > 0 ? actual.slice(actual.length - k) : [];
  const display = vm.chart.display || 'normal';
  return {
    rank: null, index: vm.index, name: vm.name, slot: vm.slot, icon: vm.icon,
    phase: { label: vm.phase.label, name: vm.phase.name, icon: vm.phase.icon, tentative: vm.phase.tentative },
    outlook: { ...vm.hero.outlook },
    expected_change: { text: changeText(tile('expected_change')), basis: vm.hero.basis, undefined_reason: tile('expected_change').undefined_reason },
    recent_change: { text: changeText(tile('recent_change')), basis: (tile('recent_change').sub || '').replace(/^[^·]*·\s*/, ''), undefined_reason: tile('recent_change').undefined_reason },
    confidence: vm.hero.reliability,
    spark: { history, forecast: display === 'bands_only' ? [] : ((vm.chart.forecast && vm.chart.forecast.mean) || []).slice(0, 52) },
    display,
    flags: { low_resolution: vm.flags.low_resolution, reference_only: vm.flags.reference_only, tentative: vm.flags.tentative, synthetic: vm.flags.synthetic },
    provisional: true,
  };
}

/**
 * @param {Object} row  a ranking row (or provisionalRow)
 * @param {{vm?: Object, slot: number|null, fixCtx: Object, onMore?: (row: Object, e: Event) => void,
 *          moreHref?: string, onFocus?: (index: number|null) => void}} opts
 */
export function rankCard(row, opts) {
  const slot = opts.slot;
  const changeLine = (key, item) => {
    if (!item) return null;
    const basis = item.undefined_reason ? `${item.basis ? `${item.basis} · ` : ''}${item.undefined_reason}` : item.basis;
    return h('p', { class: 'rank-line' }, t(key, { text: item.text, basis }));
  };
  const lowRes = row.flags && row.flags.low_resolution && opts.vm
    ? (opts.vm.notices || []).find((n) => n.code === 'naver_low_resolution') : null;
  const focus = () => opts.onFocus && opts.onFocus(row.index);
  const blur = () => opts.onFocus && opts.onFocus(null);
  return h('article', {
    class: ['card', 'rank-card', slot ? `slot-${slot}` : 'slot-none', row.provisional && 'is-provisional'],
    dataset: { index: row.index }, 'aria-labelledby': `rank-name-${row.index}`,
    on: { mouseenter: focus, mouseleave: blur, focusin: focus, focusout: blur },
  },
  h('div', { class: 'rank-head' },
    row.rank != null ? h('span', { class: 'rank-num' }, t('compare.card.rank', { rank: row.rank })) : null,
    h('h3', { class: 'rank-name', id: `rank-name-${row.index}` }, swatch(slot), h('span', null, row.name)),
    phaseBadge(row.phase)),
  h('p', { class: ['rank-outlook', toneClass(row.outlook.tone)] }, arrowLine(row.outlook.line)),
  changeLine('compare.card.expected', row.expected_change),
  changeLine('compare.card.recent', row.recent_change),
  h('div', { class: 'rank-spark' }, sparkline(row.spark || [], { slot, grey: !slot, display: row.display })),
  lowRes ? h('div', { class: 'rank-warning' },
    h('p', null, lowRes.text),
    fixRow(localFixes(lowRes.fixes), opts.fixCtx, { primary: false })) : null,
  h('div', { class: 'rank-foot' },
    h('div', { class: 'pill-row' },
      row.confidence ? pill(row.confidence.label, `pill-rel-${row.confidence.level}`) : null,
      row.flags && row.flags.reference_only ? pill(t('common.reference_only'), 'pill-warn') : null,
      row.score ? h('span', { class: 'rank-score' }, t('compare.card.score', { value: row.score.text })) : null),
    row.provisional ? h('span', { class: 'muted small' }, t('compare.card.pending'))
      : h('a', {
        class: 'more-link', href: opts.moreHref || '#',
        on: { click: (e) => { if (opts.onMore) { e.preventDefault(); opts.onMore(row, e); } } },
      }, t('compare.card.more'))));
}

/** The error card of a failed series in the ranking grid (F.6.7). */
export function failedCard(entry, fixCtx) {
  return errorCard(ApiError.fromObject(entry.error), fixCtx, { name: entry.name, headingLevel: 3 });
}
