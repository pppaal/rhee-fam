// Results pages (spec F.2, F.6.4–F.6.12): /explore (search: 1 series → detail, 2+ → compare) and, through
// createResultsPage(), /r/<id> and /s/<id> (pages/result.js). Owner: WP3-app.
//
// Flow: skeletons of the final layout at once (< 1 s, F.6.5) → progress (status line after 1 s, 4-step panel after
// 3 s or at once for ≥ 3 series, 취소) → progressive VMs as they finish → the authoritative envelope from
// GET /api/results (slots, exports, query, expiry). Finished results are cached in memory per query and language,
// so back/forward and focus changes never re-poll; view-only parameters (scale, band, view) and `focus` are applied
// without a new analysis. Leaving the page aborts the analysis (DELETE for search jobs, F.3).

import * as api from '../api.js';
import { clear, h } from '../dom.js';
import { datetimeLabel } from '../fmt.js';
import { getLang, t } from '../i18n.js';
import { effectiveSource, queryKey, seriesName, serialize, withState } from '../state.js';
import * as store from '../store.js';
import { announce, icon } from '../ui.js';
import { emptyState, errorCard } from '../components/errorcard.js';
import { createProgress } from '../components/progress.js';
import { createQueryBar } from '../components/querybar.js';
import { isUploadResult, shareAction } from '../components/sharemenu.js';
import { createCompareView } from '../views/compare.js';
import { createDetailView } from '../views/detail.js';

const autoRetried = new Set();       // naver_rate: automatic retry at most once per query (C.12.6)

/** Skeleton of the final layout (F.6.5): real title, placeholder hero / tiles / chart / cards. */
function skeleton(kind, title, total) {
  const line = (w) => h('div', { class: ['skeleton', 'skeleton-line', w] });
  const block = (cls) => h('div', { class: ['skeleton', 'skeleton-block', cls] });
  const head = h('header', { class: 'view-head' }, h('div', { class: 'view-title-wrap' },
    h('h1', { class: 'view-title' }, title || h('span', { class: 'skeleton skeleton-line w-40 inline' }))));
  if (kind === 'compare') {
    return h('div', { class: 'view skeleton-view', 'aria-busy': 'true' }, head,
      h('div', { class: 'card' }, line('w-30'), block('h-chart')),
      h('div', { class: 'rank-grid' }, Array.from({ length: Math.max(2, total || 3) }, () => h('div', { class: 'card skeleton-card' },
        line('w-60'), line('w-80'), line('w-40'), block('h-32')))));
  }
  return h('div', { class: 'view skeleton-view', 'aria-busy': 'true' }, head,
    line('w-80 source-skeleton'),
    h('div', { class: 'card skeleton-hero' }, line('w-30 h-tall'), line('w-60'), line('w-80')),
    h('div', { class: 'skeleton-tiles' }, Array.from({ length: 6 }, () => h('div', { class: 'skeleton-tile' }, line('w-60'), line('w-40 h-tall'), line('w-80')))),
    h('div', { class: 'card' }, line('w-30'), block('h-chart')),
    h('div', { class: 'card' }, line('w-40'), line('w-80'), line('w-60')));
}

/**
 * The results page controller shared by /explore, /r/<id> and /s/<id>.
 * @param {Object} ctx  page context (app.js)
 * @param {'explore'|'result'|'shared'} kind
 */
export function createResultsPage(ctx, kind) {
  let state = ctx.state;
  const lang = getLang();
  const config = ctx.config;
  const src = effectiveSource(state, config.default_source);
  const key = kind === 'explore' ? queryKey(state, { defaultSource: config.default_source }) : `${kind}:${state.id}`;
  const cacheKey = `${key}|${lang}`;
  let env = null;
  let view = null;
  let progress = null;
  let controller = null;
  let jobId = kind === 'result' ? state.id : null;
  let progressiveCount = -1;

  const body = h('div', { class: 'results-body' });
  const progressSlot = h('div', { class: 'results-progress' });
  const viewSlot = h('div', { class: 'results-view' });
  body.append(progressSlot, viewSlot);

  const patch = (p, o = {}) => ctx.go(withState(state, p), o);
  const viewCtx = { ...ctx, patch, get state() { return state; } };
  const qb = createQueryBar({
    ctx, state, mode: kind === 'explore' ? 'search' : 'upload',
    onChange: (p) => ctx.go(withState(state, { ...p, scale: p.q ? null : state.scale })),
  });
  const fixCtx = {
    navigate: ctx.navigate,
    get state() { return state; },
    retry: () => restart(),
    reanalyze: kind === 'explore' ? () => restart() : () => ctx.navigate('/upload'),
    openChipEditor: () => qb.openChipEditor(),
    focusChips: () => qb.focusChips(),
    get csvUrl() { return env && env.exports ? env.exports.csv : null; },
  };

  // ------------------------------------------------------------------ titles
  function names() {
    if (env) return env.series.map((vm) => vm.name).concat((env.failed || []).map((f) => f.name));
    return kind === 'explore' ? state.q.map(seriesName) : [];
  }
  function titleText() {
    const list = names();
    if (!list.length) return '';
    return list.length > 1 ? list.join(' vs ') : list[0];
  }
  function setTitle(text) {
    ctx.setTitle(text ? `${text} · trendai` : 'trendai');
  }
  setTitle(titleText());

  function keepHeadingFocus(fn) {
    const hadFocus = document.activeElement && document.activeElement.matches && document.activeElement.matches('h1');
    fn();
    if (hadFocus) {
      const h1 = viewSlot.querySelector('h1');
      if (h1) {
        h1.setAttribute('tabindex', '-1');
        h1.focus({ preventScroll: true });
      }
    }
  }

  function destroyView() {
    if (view && view.destroy) view.destroy();
    view = null;
  }

  function destroyProgress() {
    if (progress) progress.destroy();
    progress = null;
    clear(progressSlot);
  }

  // ------------------------------------------------------------------ banners (result / shared pages)
  function banner() {
    if (!env) return null;
    if (env.readonly && env.shared) {
      return h('p', { class: 'banner banner-readonly' }, icon('ui-lock'), h('span', null, t('share.readonly', { date: datetimeLabel(env.shared.expires_at, lang) })));
    }
    if (kind !== 'explore' && isUploadResult(env) && env.expires_at) {
      return h('div', { class: 'banner' }, icon('ui-clock'),
        h('span', null, t('result.expiry', { date: datetimeLabel(env.expires_at, lang) })),
        h('button', { type: 'button', class: 'btn btn-secondary btn-small', on: { click: (e) => shareAction(env, viewCtx, e.currentTarget) } },
          icon('ui-share'), t('detail.share')));
    }
    return null;
  }

  // ------------------------------------------------------------------ rendering
  function showEnvelope(next) {
    env = next;
    destroyProgress();
    qb.setEnvelope(env);
    keepHeadingFocus(() => {
      destroyView();
      clear(viewSlot);
      const focusVm = state.focus ? env.series.find((vm) => vm.name === state.focus) : null;
      if (env.view === 'compare' && !focusVm) {
        view = createCompareView({ env, state, ctx: viewCtx, fixCtx, names: names(), title: titleText(), banner: banner(), total: env.series.length });
        setTitle(titleText());
      } else {
        const vm = focusVm || env.series[0];
        view = createDetailView({
          vm, env, state, ctx: viewCtx, fixCtx, banner: banner(),
          extraNotices: env.view === 'detail' ? env.notices : [],
          onBack: focusVm && env.view === 'compare' ? () => {
            if (window.history.state && window.history.state.fromCompare) window.history.back();
            else patch({ focus: null });
          } : null,
        });
        setTitle(vm.name);
      }
      viewSlot.appendChild(view.el);
    });
  }

  function onSnapshot(snap) {
    if (progress) progress.update(snap);
    const vms = snap.series || [];
    const failed = snap.failed || [];
    const count = vms.length * 100 + failed.length;
    if (count === progressiveCount || (!vms.length && !failed.length)) return;
    progressiveCount = count;
    keepHeadingFocus(() => {
      destroyView();
      clear(viewSlot);
      if (snap.view === 'compare' || (snap.series_total || 0) > 1) {
        view = createCompareView({
          env: null, series: vms, failed, total: snap.series_total || vms.length + failed.length, state, ctx: viewCtx, fixCtx,
          names: names(), title: titleText() || vms.map((vm) => vm.name).join(' vs '), banner: null,
        });
      } else if (vms.length) {
        view = createDetailView({ vm: vms[0], env: null, state, ctx: viewCtx, fixCtx, banner: null });
      }
      if (view) viewSlot.appendChild(view.el);
    });
  }

  function showError(err) {
    destroyProgress();
    destroyView();
    clear(viewSlot);
    const auto = err.code === 'naver_rate' && !autoRetried.has(key);
    if (auto) autoRetried.add(key);
    viewSlot.append(
      h('header', { class: 'view-head' }, h('div', { class: 'view-title-wrap' }, h('h1', { class: 'view-title' }, titleText() || t('app.name')))),
      errorCard(err, fixCtx, { autoRetry: auto }));
    announce(err.text(), true);
  }

  function showCancelled() {
    destroyProgress();
    destroyView();
    clear(viewSlot);
    viewSlot.append(
      h('header', { class: 'view-head' }, h('div', { class: 'view-title-wrap' }, h('h1', { class: 'view-title' }, titleText()))),
      emptyState({
        title: t('progress.cancelled'), level: 2, icon: 'ui-info',
        actions: [h('button', { type: 'button', class: 'btn btn-primary', on: { click: () => fixCtx.reanalyze() } }, t('chip.reanalyze_upload'))],
      }));
    announce(t('progress.cancelled'));
  }

  function remember(next) {
    ctx.results.set(cacheKey, next);
    if (kind === 'explore' && next.result_id) ctx.resultIds.set(key, next.result_id);
    if (kind === 'shared') return;
    const upload = isUploadResult(next);
    const title = upload && next.query.filename ? next.query.filename : names().join(' vs ');
    const url = kind === 'explore'
      ? serialize(withState(state, { scale: null, band: 'both', view: 'chart', focus: null, hl: null }))
      : `/r/${state.id}`;
    store.addRecent({ title, url, expires_at: upload || kind === 'result' ? next.expires_at : null });
  }

  function startProgress(total) {
    destroyProgress();
    progress = createProgress({ total, source: src, onCancel: () => cancel() });
    progressSlot.appendChild(progress.el);
  }

  function cancel() {
    if (jobId) api.cancelJob(jobId);
    if (controller) controller.abort();
    controller = null;
    showCancelled();
  }

  // ------------------------------------------------------------------ loading
  async function loadExplore() {
    const known = ctx.resultIds.get(key);
    if (known) {
      try {
        const next = await api.result(known, lang, { signal: controller.signal });
        remember(next);
        showEnvelope(next);
        return;
      } catch (err) {
        if (err && err.name === 'AbortError') return;
        ctx.resultIds.delete(key);
      }
    }
    startProgress(state.q.length);
    const next = await api.analyze(state, {
      signal: controller.signal, hl: getLang, defaultSource: config.default_source,
      onStart: (start) => { jobId = start.job_id; },
      onUpdate: onSnapshot,
    });
    remember(next);
    showEnvelope(next);
  }

  async function loadResult() {
    startProgress(1);
    try {
      // a result page only stops polling when left: the upload job keeps running (no DELETE), only 취소 cancels it
      const next = await api.pollJob(state.id, { signal: controller.signal, hl: getLang, onUpdate: onSnapshot, cancelOnAbort: false });
      remember(next);
      showEnvelope(next);
    } catch (err) {
      if (err && err.code === 'job_not_found') {
        const next = await api.result(state.id, lang, { signal: controller.signal });   // throws result_not_found
        remember(next);
        showEnvelope(next);
        return;
      }
      throw err;
    }
  }

  async function loadShared() {
    const next = await api.shared(state.id, lang, { signal: controller.signal });
    remember(next);
    showEnvelope(next);
  }

  /** Load (or reload) the result: cache → known result id → analysis / job polling / share. */
  function run({ fresh = false } = {}) {
    if (controller) controller.abort();
    controller = new AbortController();
    const mine = controller;
    if (fresh) ctx.results.delete(cacheKey);
    progressiveCount = -1;
    env = null;
    destroyProgress();
    destroyView();
    clear(viewSlot);
    const cached = ctx.results.get(cacheKey);
    if (cached) {
      showEnvelope(cached);
      return;
    }
    viewSlot.appendChild(skeleton(kind === 'explore' && state.q.length > 1 ? 'compare' : 'detail', titleText(), state.q.length));
    const load = kind === 'explore' ? loadExplore : kind === 'result' ? loadResult : loadShared;
    load().catch((err) => {
      if (mine !== controller || !err || err.name === 'AbortError') return;
      if (err.cancelled) showCancelled();
      else if (typeof err.text === 'function') showError(err);
      else showError(api.ApiError.client('client_network', 'err.client_network', { fixes: ['retry'] }));
    });
  }

  function restart() {
    run({ fresh: true });
  }

  const el = h('div', { class: 'page results-page' }, qb.el, body);
  run();

  return {
    el,
    update(next) {
      if (next.lang !== lang) return false;
      if (kind === 'explore' && queryKey(next.state, { defaultSource: config.default_source }) !== key) return false;
      if (kind !== 'explore' && next.state.id !== state.id) return false;
      const focusChanged = next.state.focus !== state.focus;
      state = next.state;
      if (focusChanged && env) {
        showEnvelope(env);
        window.scrollTo(0, 0);
        const h1 = viewSlot.querySelector('h1');
        if (h1) {
          h1.setAttribute('tabindex', '-1');
          h1.focus({ preventScroll: true });
        }
        announce(document.title);
      } else if (view && view.update) {
        view.update(state);
      }
      return true;
    },
    destroy() {
      if (controller) controller.abort();
      controller = null;
      destroyProgress();
      destroyView();
      qb.destroy();
    },
  };
}

export function create(ctx) {
  return createResultsPage(ctx, 'explore');
}
