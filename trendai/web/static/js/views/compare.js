// Compare view (spec F.6.8, brief §3.4): header (names, 공유, 내보내기 ▾) · card 한눈에 보기 (?, ⋯ menu, scale
// segmented control, 예측 범위 보기, 표로 보기, comparison chart or ranking + values tables, source line, axis note,
// notices) · 무엇이 뜨나요? (sort select, score note with ?, spark note, ranking cards; failed series as error cards).
// While the job runs it renders progressively from the series VMs received so far. A first Naver comparison shows
// the one-time explainer after the comparison card, never above the chart (UX-3). 내보내기 has 요약 복사: the
// ranking in one line (FU-15). Owner: WP3-app.

import { h, clear } from '../dom.js';
import { getLang, t } from '../i18n.js';
import { serialize, withState } from '../state.js';
import { cardHead, icon, rovingRadios } from '../ui.js';
import { createCompareChart } from '../chart/comparechart.js';
import { rankingTable, valuesTable } from '../components/datatable.js';
import { helpButton } from '../components/help.js';
import { noticeList } from '../components/notices.js';
import { SORT_KEYS, failedCard, provisionalRow, rankCard, sortRows } from '../components/rankcards.js';
import { chartMoreMenu, compareSummaryLine, exportMenu, shareButton, summaryUrl } from '../components/sharemenu.js';
import { noteLine, sourceLine } from '../components/sourceline.js';
import { naverExplainer } from './detail.js';

const SCALES = ['raw', 'index', 'log'];

/** The scale in effect: the URL's `scale` when allowed, else the VM default (D.4.3). */
export function effectiveScale(compare, wanted) {
  if (!compare || !compare.scale) return 'raw';
  const allowed = compare.scale.allowed || {};
  if (wanted && allowed[wanted]) return wanted;
  return allowed[compare.scale.default] ? compare.scale.default : 'raw';
}

/**
 * @param {{env: Object|null, series: Object[], failed: Array, total: number, names: string[], state: Object,
 *          ctx: Object, fixCtx: Object, banner?: Node|null, title: string}} opts
 *        env: final envelope (null while running: series/failed are the progressive ones)
 * @returns {{el: HTMLElement, update: (state: Object) => void, destroy: () => void}}
 */
export function createCompareView(opts) {
  const { env, ctx, fixCtx } = opts;
  const lang = getLang();
  const compare = env ? env.compare : null;
  const series = (env ? env.series : opts.series) || [];
  const failed = (env ? env.failed : opts.failed) || [];
  const vmByIndex = new Map(series.map((vm) => [vm.index, vm]));
  const slots = compare ? compare.slots : null;
  const slotOf = (index) => {
    if (slots) {
      const s = slots[String(index)];
      return Number.isInteger(s) ? s : null;
    }
    const vm = vmByIndex.get(index);
    return vm && Number.isInteger(vm.slot) ? vm.slot : null;
  };
  const naver = series.some((vm) => vm.source.kind === 'naver');
  let state = opts.state;
  let sortKey = 'score';
  let bands = false;
  let chart = null;

  // ------------------------------------------------------------------ header
  const head = h('header', { class: 'view-head' },
    h('div', { class: 'view-title-wrap' }, h('h1', { class: 'view-title' }, opts.title)),
    h('div', { class: 'view-actions' },
      shareButton(env, ctx),
      exportMenu({
        exports: env ? env.exports : null,
        png: { getChart: () => chart, title: t('compare.title'), footer: compare ? compare.source_line : '', name: opts.names.join(' vs ') },
        summary: compare ? { line: compareSummaryLine(compare, opts.names), url: () => summaryUrl(env, state) } : null,
      })));

  // ------------------------------------------------------------------ controls
  const scaleNote = h('p', { class: 'muted small scale-note', 'aria-live': 'polite' });
  const scaleButtons = SCALES.map((s) => {
    const allowed = compare ? Boolean(compare.scale.allowed[s]) : s === 'raw';
    const reason = compare && compare.scale.disabled_reason ? compare.scale.disabled_reason[s] : null;
    return h('button', {
      type: 'button', role: 'radio', class: 'segment', dataset: { scale: s },
      'aria-disabled': allowed ? null : 'true', title: allowed ? null : reason,
      on: {
        click: () => {
          if (!allowed) {
            scaleNote.textContent = reason || '';
            return;
          }
          scaleNote.textContent = '';
          ctx.patch({ scale: s === compare.scale.default ? null : s }, { replace: true });
        },
      },
    }, t(`compare.scale.${s}`));
  });
  rovingRadios(scaleButtons, (next) => next.click(), { vertical: false });
  const segmented = h('div', { class: 'segmented', role: 'radiogroup', 'aria-label': t('compare.scale.label') }, scaleButtons);
  const bandsBtn = h('button', {
    type: 'button', class: 'chip chip-toggle chip-small', 'aria-pressed': 'false', disabled: compare ? null : true,
    on: { click: () => { bands = !bands; bandsBtn.setAttribute('aria-pressed', String(bands)); if (chart) chart.setBands(bands); } },
  }, t('compare.bands'));
  const tableBtn = h('button', {
    type: 'button', class: 'chip chip-toggle chip-small', 'aria-pressed': String(state.view === 'table'), disabled: compare ? null : true,
    on: { click: () => ctx.patch({ view: state.view === 'table' ? 'chart' : 'table' }, { replace: true }) },
  }, icon('ui-table'), t('compare.view.table'));

  const chartBox = h('div', { class: 'chart-box' });
  const tableBox = h('div', { class: 'table-box', hidden: true });
  const indexLabel = h('p', { class: 'muted small', hidden: true }, compare && compare.scale.index_base ? compare.scale.index_base.label : '');

  const compareCard = h('section', { class: 'card compare-card', 'aria-labelledby': 'compare-title' },
    cardHead({
      title: t('compare.title'), id: 'compare-title', help: helpButton(t('compare.title'), t('help.compare')),
      actions: compare ? [chartMoreMenu({ env, ctx, png: { getChart: () => chart, title: t('compare.title'), footer: compare.source_line, name: opts.names.join(' vs ') } })] : [],
    }),
    h('div', { class: 'compare-controls' }, segmented, h('div', { class: 'toggle-row' }, bandsBtn, tableBtn)),
    scaleNote,
    chartBox,
    tableBox,
    compare ? sourceLine(compare.source_line) : null,
    compare ? noteLine(compare.axis_note, { naver, className: 'muted small' }) : null,
    indexLabel,
    noticeList([...((compare && compare.notices) || []), ...((env && env.notices) || [])], fixCtx));

  // ------------------------------------------------------------------ ranking
  const grid = h('div', { class: 'rank-grid' });
  const sortSelect = h('select', {
    id: 'rank-sort', class: 'input select select-small', disabled: compare ? null : true,
    on: { change: (e) => { sortKey = e.target.value; renderCards(); } },
  }, SORT_KEYS.map((k) => h('option', { value: k }, t(`compare.sort.${k}`))));
  const rankSection = h('section', { class: 'rank-section', 'aria-labelledby': 'rank-title' },
    h('div', { class: 'rank-section-head' },
      h('h2', { class: 'section-title', id: 'rank-title' }, t('compare.rank.title')),
      h('div', { class: 'rank-tools' },
        h('label', { class: 'field-label inline', for: 'rank-sort' }, t('compare.sort')), sortSelect)),
    // two notes, two lines on a phone: no separator left dangling at a line end (UX-31)
    h('div', { class: 'muted small rank-notes' },
      h('p', { class: 'rank-note' }, h('span', null, t('compare.score_short')), helpButton(t('compare.sort.score'), compare ? compare.score_note : t('help.score'))),
      compare && compare.spark_note ? h('p', { class: 'rank-note rank-note-sep' }, compare.spark_note) : null),
    grid);

  function moreHref(row) {
    return serialize(withState(state, { focus: row.name }));
  }

  function renderCards() {
    clear(grid);
    const rows = compare ? sortRows(compare.ranking, sortKey) : series.map(provisionalRow);
    for (const row of rows) {
      grid.appendChild(rankCard(row, {
        vm: vmByIndex.get(row.index), slot: slotOf(row.index), fixCtx,
        moreHref: moreHref(row),
        onMore: (r) => ctx.patch({ focus: r.name }, { state: { fromCompare: true } }),
        onFocus: (index) => { if (chart) chart.setFocus(index); },
      }));
    }
    if (!compare) {
      const pending = Math.max(0, opts.total - series.length - failed.length);
      for (let i = 0; i < pending; i += 1) grid.appendChild(skeletonCard());
    }
    for (const entry of failed) grid.appendChild(failedCard(entry, fixCtx));
  }

  function applyView() {
    const table = state.view === 'table' && compare;
    tableBtn.setAttribute('aria-pressed', String(Boolean(table)));
    chartBox.hidden = Boolean(table);
    tableBox.hidden = !table;
    segmented.hidden = Boolean(table);
    if (table && !tableBox.firstChild) {
      tableBox.append(rankingTable(sortRows(compare.ranking, 'score')),
        valuesTable(series.slice().sort((a, b) => a.index - b.index), env.exports, slots, lang));
    }
    const scale = effectiveScale(compare, state.scale);
    for (const b of scaleButtons) b.setAttribute('aria-checked', String(b.dataset.scale === scale));
    for (const b of scaleButtons) b.tabIndex = b.dataset.scale === scale ? 0 : -1;
    indexLabel.hidden = scale !== 'index';
    if (chart) chart.setScale(scale);
  }

  renderCards();
  const el = h('div', { class: 'view compare-view' },
    head,
    opts.banner || null,
    compareCard,
    naver ? naverExplainer() : null,
    rankSection);

  requestAnimationFrame(() => {
    if (!chartBox.isConnected) return;
    chart = createCompareChart(chartBox, compare, series, {
      lang, t: (key, params) => t(key, params), scale: effectiveScale(compare, state.scale), bands, focus: null,
    });
    chart.onFocusChange((index) => {
      for (const card of grid.querySelectorAll('.rank-card')) {
        card.classList.toggle('is-focused', index != null && Number(card.dataset.index) === index);
      }
    });
  });
  applyView();

  return {
    el,
    update(next) {
      state = next;
      applyView();
    },
    destroy() {
      if (chart) chart.destroy();
    },
  };
}

function skeletonCard() {
  return h('div', { class: 'card skeleton-card', 'aria-hidden': 'true' },
    h('div', { class: 'skeleton skeleton-line w-60' }),
    h('div', { class: 'skeleton skeleton-line w-80' }),
    h('div', { class: 'skeleton skeleton-line w-40' }),
    h('div', { class: 'skeleton skeleton-block h-32' }));
}

