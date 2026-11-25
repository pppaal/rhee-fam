// Detail view of one series (spec F.6.9, brief §3.5) — fixed card order: header (← 비교로 when focused, swatch + name,
// 가상 데이터 badge, 공유, 내보내기 ▾, 표) · source line + notices · hero · tiles · chart card (80% / 95% / 추세선 /
// 이상치 toggles, fan chart or its table, band legend, caption, y_note, value-kind note) · 현재 · 과거 · 미래 · 해석 ·
// 주의할 점. The first Naver result shows a one-time explainer card. Owner: WP3-app.

import { h, clear } from '../dom.js';
import { getLang, t } from '../i18n.js';
import * as store from '../store.js';
import { cardHead, icon, pill, swatch } from '../ui.js';
import { createFanChart } from '../chart/fanchart.js';
import { createCaveats } from '../components/caveats.js';
import { seriesTable } from '../components/datatable.js';
import { renderFuture } from '../components/future.js';
import { renderHero } from '../components/hero.js';
import { createNarrative } from '../components/narrative.js';
import { noticeList } from '../components/notices.js';
import { createPast } from '../components/past.js';
import { renderPhaseWhy } from '../components/phasewhy.js';
import { exportMenu, shareButton, summaryUrl } from '../components/sharemenu.js';
import { noteLine, sourceLine } from '../components/sourceline.js';
import { renderTiles } from '../components/tiles.js';

/** The one-time Naver explainer card (F.6.9); null once dismissed. */
export function naverExplainer() {
  if (store.isNaverExplainerDismissed()) return null;
  const card = h('section', { class: 'card explainer', 'aria-labelledby': 'naver-explainer-title' },
    h('div', { class: 'explainer-icon' }, icon('ui-info')),
    h('div', null,
      h('h2', { class: 'card-title', id: 'naver-explainer-title' }, t('naver_explainer.title')),
      h('p', null, t('naver_explainer.body')),
      h('button', {
        type: 'button', class: 'btn btn-secondary btn-small',
        on: { click: () => { store.dismissNaverExplainer(); card.remove(); } },
      }, t('naver_explainer.ok'))));
  return card;
}

const BANDS = { both: [true, true], 80: [true, false], 95: [false, true], off: [false, false] };
const bandParam = (b80, b95) => (b80 && b95 ? 'both' : b80 ? '80' : b95 ? '95' : 'off');

/**
 * @param {{vm: Object, env: Object|null, state: Object, ctx: Object, fixCtx: Object, onBack?: (() => void)|null,
 *          extraNotices?: Array, banner?: Node|null}} opts
 *        env: the final C.7/C.9 envelope (null while the job still runs: share/export wait for it)
 * @returns {{el: HTMLElement, update: (state: Object) => void, destroy: () => void}}
 */
export function createDetailView(opts) {
  const { vm, env, ctx, fixCtx } = opts;
  const lang = getLang();
  let state = opts.state;
  const naver = vm.source.kind === 'naver';
  const toggles = { trend: true, anomalies: true };
  let chart = null;

  // ------------------------------------------------------------------ header
  const tableBtn = h('button', {
    type: 'button', class: 'btn btn-secondary', 'aria-pressed': String(state.view === 'table'),
    on: { click: () => setView(state.view === 'table' ? 'chart' : 'table') },
  }, icon('ui-table'), t('detail.table'));
  const exportBtn = exportMenu({
    exports: env ? vm.exports : null,
    png: { getChart: () => chart, title: vm.chart.title || vm.name, footer: vm.source.line, name: vm.name },
    summary: { line: vm.summary_line, url: () => summaryUrl(env, state) },
  });
  const head = h('header', { class: 'view-head' },
    h('div', { class: 'view-title-wrap' },
      opts.onBack ? h('button', { type: 'button', class: 'link-btn back-link', on: { click: opts.onBack } }, t('detail.back')) : null,
      h('h1', { class: 'view-title' }, swatch(vm.slot), h('span', null, vm.name)),
      vm.flags.synthetic ? pill(t('common.synthetic'), 'pill-synthetic') : null),
    h('div', { class: 'view-actions' }, shareButton(env, ctx), exportBtn, tableBtn));

  // ------------------------------------------------------------------ chart card
  const chartBox = h('div', { class: 'chart-box' });
  const tableBox = h('div', { class: 'table-box', hidden: true });
  const [b80, b95] = BANDS[state.band] || BANDS.both;
  const toggle = (label, pressed, onToggle) => h('button', {
    type: 'button', class: 'chip chip-toggle chip-small', 'aria-pressed': String(pressed),
    on: { click: (e) => { const on = e.currentTarget.getAttribute('aria-pressed') !== 'true'; e.currentTarget.setAttribute('aria-pressed', String(on)); onToggle(on); } },
  }, label);
  const t80 = toggle(t('detail.toggle.band80'), b80, () => setBand());
  const t95 = toggle(t('detail.toggle.band95'), b95, () => setBand());
  const tTrend = toggle(t('detail.toggle.trend'), true, (on) => { toggles.trend = on; if (chart) chart.setToggles({ trend: on }); });
  const tAnom = toggle(t('detail.toggle.anomalies'), true, (on) => { toggles.anomalies = on; if (chart) chart.setToggles({ anomalies: on }); });
  const chartCard = h('section', { class: 'card chart-card', id: 'chart-card', 'aria-labelledby': 'chart-title' },
    cardHead({ title: t('detail.chart.title'), id: 'chart-title', actions: [h('div', { class: 'toggle-row', role: 'group', 'aria-label': t('detail.chart.title') }, t80, t95, tTrend, tAnom)] }),
    chartBox,
    tableBox,
    h('p', { class: 'chart-legend-line' },
      h('span', { class: 'band-key band-key-80', 'aria-hidden': 'true' }), h('span', { class: 'band-key band-key-95', 'aria-hidden': 'true' }),
      t('chart.legend_bands')),
    h('p', { class: 'chart-caption muted small' }, t('chart.band_caption')),
    vm.chart.y_note ? h('p', { class: 'chart-caption muted small' }, vm.chart.y_note) : null,
    noteLine(vm.source.value_kind_note, { naver, className: 'muted small' }));

  function setBand() {
    const band = bandParam(t80.getAttribute('aria-pressed') === 'true', t95.getAttribute('aria-pressed') === 'true');
    if (chart) chart.setBand(band);
    ctx.patch({ band }, { replace: true });
  }

  function setView(view) {
    ctx.patch({ view }, { replace: true });
  }

  function applyView() {
    const table = state.view === 'table';
    tableBtn.setAttribute('aria-pressed', String(table));
    chartBox.hidden = table;
    tableBox.hidden = !table;
    if (table && !tableBox.firstChild) tableBox.appendChild(seriesTable(vm, lang));
  }

  // ------------------------------------------------------------------ sections
  const past = createPast(vm, {
    onChangepoint: (n) => chart && chart.highlightChangepoint(n),
    onDate: (iso) => showDate(iso),
  });
  const caveats = createCaveats(vm);
  const narrative = createNarrative(vm, {
    resultId: env && !env.readonly ? env.result_id : null,
    onDateLink: (iso) => showDate(iso),
    onModeChange: (mode, caveatsMarkdown) => caveats.setMarkdown(mode === 'ai' ? caveatsMarkdown : null),
    navigate: ctx.navigate,
  });

  function showDate(iso) {
    if (state.view === 'table') setView('chart');
    if (chart) chart.highlightDate(iso);
    chartCard.scrollIntoView({ block: 'center' });
  }

  const notices = [...(vm.notices || []), ...(opts.extraNotices || [])];
  const el = h('div', { class: 'view detail-view' },
    head,
    opts.banner || null,
    naver ? naverExplainer() : null,
    sourceLine(vm.source.line, { naver }),
    noticeList(notices, fixCtx),
    renderHero(vm),
    renderTiles(vm),
    chartCard,
    renderPhaseWhy(vm),
    past.el,
    renderFuture(vm),
    narrative.el,
    caveats.el);

  // the chart needs a laid-out container: create it once the view is in the document
  requestAnimationFrame(() => {
    if (!chartBox.isConnected) return;
    chart = createFanChart(chartBox, vm, {
      lang, t: (key, params) => t(key, params), band: state.band, trend: toggles.trend, anomalies: toggles.anomalies,
    });
    chart.onChangepoint((n) => past.highlight(n));
  });
  applyView();

  return {
    el,
    update(next) {
      state = next;
      const [n80, n95] = BANDS[next.band] || BANDS.both;
      t80.setAttribute('aria-pressed', String(n80));
      t95.setAttribute('aria-pressed', String(n95));
      if (chart) chart.setBand(next.band);
      applyView();
    },
    destroy() {
      narrative.destroy();
      if (chart) chart.destroy();
      clear(chartBox);
    },
  };
}
