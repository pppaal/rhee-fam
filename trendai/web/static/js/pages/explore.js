// Results pages (spec F.2, F.6.4–F.6.12): /explore (search: 1 series → detail, 2+ → compare) and, through
// createResultsPage(), /r/<id> and /s/<id> (pages/result.js). Owner: WP3-app.
//
// Flow: skeletons of the final layout at once (< 1 s, F.6.5) → progress (status line after 1 s, 4-step panel after
// 3 s or at once for ≥ 3 series, 취소) → progressive VMs as they finish → the authoritative envelope from
// GET /api/results (slots, exports, query, expiry). Finished results are cached in memory per query and language,
// so back/forward and focus changes never re-poll; view-only parameters (scale, band, view) and `focus` are applied
// without a new analysis. Leaving the page aborts the analysis (DELETE for search jobs, F.3); so does closing or
// reloading the tab (pagehide, OPS-7), so a reload's new request and a later 취소 count subscribers correctly.
//
// Keyboard focus survives every re-render (A11Y-2): a control focused inside the results when a new series arrives
// (or the final envelope replaces the progressive view) is found again in the new view, else the view's <h1> gets
// focus. Completion, cancellation and errors are announced (A11Y-6).

import * as api from '../api.js';
import { FOCUSABLE_SELECTOR, clear, h } from '../dom.js';
import { datetimeLabel } from '../fmt.js';
import { getLang, t } from '../i18n.js';
import { effectiveSource, queryKey, seriesName, serialize, withState } from '../state.js';
import * as store from '../store.js';
import { announce, icon } from '../ui.js';
import { emptyState, errorCard } from '../components/errorcard.js';
import { createProgress } from '../components/progress.js';
import { createQueryBar } from '../components/querybar.js';
import { failedCard } from '../components/rankcards.js';
import { isUploadResult, shareAction } from '../components/sharemenu.js';
import { createCompareView } from '../views/compare.js';
import { createDetailView } from '../views/detail.js';
import { loadSamples } from './samples.js';

const autoRetried = new Set();       // naver_rate: automatic retry at most once per query (C.12.6)
const NOT_FOUND = new Set(['result_not_found', 'share_not_found', 'job_not_found']);

/** Skeleton of the final layout (F.6.5): real title, placeholder hero / tiles / chart / cards. */
function skeleton(kind, title, total) {
  const line = (w) => h('div', { class: ['skeleton', 'skeleton-line', w] });
  const block = (cls) => h('div', { class: ['skeleton', 'skeleton-block', cls] });
  // the heading always has a name (the series names, or "결과를 불러오는 중…" on /r and /s, A11Y-12); only the
  // placeholders are busy and hidden from assistive technology
  const head = h('header', { class: 'view-head' }, h('div', { class: 'view-title-wrap' },
    h('h1', { class: 'view-title' }, title || t('progress.loading'))));
  const busy = (...nodes) => h('div', { class: 'skeleton-body', 'aria-busy': 'true', 'aria-hidden': 'true' }, nodes);
  if (kind === 'compare') {
    return h('div', { class: 'view skeleton-view' }, head, busy(
      h('div', { class: 'card' }, line('w-30'), block('h-chart')),
      h('div', { class: 'rank-grid' }, Array.from({ length: Math.max(2, total || 3) }, () => h('div', { class: 'card skeleton-card' },
        line('w-60'), line('w-80'), line('w-40'), block('h-32'))))));
  }
  return h('div', { class: 'view skeleton-view' }, head, busy(
    line('w-80 source-skeleton'),
    h('div', { class: 'card skeleton-hero' }, line('w-30 h-tall'), line('w-60'), line('w-80')),
    h('div', { class: 'skeleton-tiles' }, Array.from({ length: 6 }, () => h('div', { class: 'skeleton-tile' }, line('w-60'), line('w-40 h-tall'), line('w-80')))),
    h('div', { class: 'card' }, line('w-30'), block('h-chart')),
    h('div', { class: 'card' }, line('w-40'), line('w-80'), line('w-60'))));
}

/**
 * A comparable identity of a focusable element, so the "same" control can be found in a re-rendered view: tag, role,
 * name, text and data-* attributes, plus the series index of the card it sits in. Pure apart from DOM reads.
 * @param {Element} el
 * @returns {string}
 */
export function focusSignature(el) {
  const holder = el.parentElement ? el.parentElement.closest('[data-index]') : null;
  return JSON.stringify([el.tagName, el.getAttribute('role'), el.getAttribute('aria-label'),
    (el.textContent || '').replace(/\s+/g, ' ').trim().slice(0, 80), { ...el.dataset }, holder ? holder.dataset.index : null]);
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
  let running = false;            // a search job of this page is still running on the server
  let cancelledOnHide = false;    // pagehide sent DELETE for it (OPS-7)
  let progressiveCount = -1;
  let sampleTitles = null;        // sample id → title, for the heading before the result arrives

  const body = h('div', { class: 'results-body' });
  const progressSlot = h('div', { class: 'results-progress' });
  const viewSlot = h('div', { class: 'results-view' });
  body.append(progressSlot, viewSlot);

  const patch = (p, o = {}) => ctx.go(withState(state, p), o);
  const viewCtx = { ...ctx, patch, get state() { return state; } };
  const qb = createQueryBar({
    ctx, state, mode: kind === 'explore' ? 'search' : 'upload', kind,
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
    if (kind !== 'explore') return [];
    return state.q.map((terms) => (src === 'sample' && sampleTitles && sampleTitles.get(terms[0])) || seriesName(terms));
  }
  function titleText() {
    const list = names();
    if (!list.length) return '';
    return list.length > 1 ? list.join(' vs ') : list[0];
  }
  function setTitle(text) {
    ctx.setTitle(text ? `${text} · trendai` : `${t('progress.loading')} · trendai`);
  }
  setTitle(titleText());

  if (kind === 'explore' && src === 'sample') {
    // the URL carries sample ids (growing, fad, …): show their titles (성장형, …) until the result names them
    loadSamples(lang).then((list) => {
      sampleTitles = new Map(list.map((s) => [s.name, s.title]));
      if (env) return;
      setTitle(titleText());
      const h1 = viewSlot.querySelector('.skeleton-view h1');
      if (h1) h1.textContent = titleText();
    }).catch(() => {});
  }

  /**
   * Run `fn` (which replaces the view) and keep keyboard focus: a control focused inside the view is focused again
   * in the new view (matched by focusSignature, also after the chart is drawn on the next frame), else the <h1>.
   */
  function rerender(fn) {
    const active = document.activeElement;
    const inside = Boolean(active && active !== document.body && viewSlot.contains(active));
    let sig = null;
    let nth = 0;
    if (inside && !active.matches('h1')) {
      sig = focusSignature(active);
      nth = Math.max(0, [...viewSlot.querySelectorAll(FOCUSABLE_SELECTOR)].filter((e) => focusSignature(e) === sig).indexOf(active));
    }
    fn();
    if (!inside) return;
    const find = () => {
      if (!sig) return null;
      const matches = [...viewSlot.querySelectorAll(FOCUSABLE_SELECTOR)].filter((e) => focusSignature(e) === sig);
      return matches[nth] || matches[0] || null;
    };
    const heading = () => {
      const h1 = viewSlot.querySelector('h1');
      if (!h1) return null;
      h1.setAttribute('tabindex', '-1');
      return h1;
    };
    const now = find();
    if (now || !sig) {
      const target = now || heading();
      if (target) target.focus({ preventScroll: true });
      return;
    }
    // the chart (legend chips, chart wrapper) is drawn on the next frame: look again then
    requestAnimationFrame(() => {
      if (document.activeElement && document.activeElement !== document.body) return;
      const target = find() || heading();
      if (target) target.focus({ preventScroll: true });
    });
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
  function renderEnvelope() {
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
  }

  /** Show the final envelope; `done` = it just finished on the server (announce it, A11Y-6). */
  function showEnvelope(next, { done = false } = {}) {
    env = next;
    destroyProgress();
    qb.setEnvelope(env);
    rerender(renderEnvelope);
    if (done) announce(t('progress.done', { name: titleText() }));
  }

  function onSnapshot(snap) {
    if (progress) progress.update(snap);
    const vms = snap.series || [];
    const failed = snap.failed || [];
    const count = vms.length * 100 + failed.length;
    if (count === progressiveCount || (!vms.length && !failed.length)) return;
    progressiveCount = count;
    rerender(() => {
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

  function viewHead(title) {
    return h('header', { class: 'view-head' }, h('div', { class: 'view-title-wrap' }, h('h1', { class: 'view-title' }, title)));
  }

  /** Focus a heading the page just rendered when focus was lost with the replaced content. */
  function focusIfLost(el) {
    const active = document.activeElement;
    if (!el || (active && active !== document.body && document.contains(active))) return;
    el.setAttribute('tabindex', '-1');
    el.focus({ preventScroll: true });
  }

  function showError(err) {
    destroyProgress();
    destroyView();
    clear(viewSlot);
    // an unknown or expired /r or /s link: no query bar, a heading that says what happened (UX-10)
    if (kind !== 'explore' && NOT_FOUND.has(err.code)) {
      qb.el.hidden = true;
      const title = t(kind === 'shared' ? 'share.unavailable' : 'result.unavailable');
      setTitle(title);
      viewSlot.append(viewHead(title), errorCard(err, fixCtx, { extraFixes: [{ action: 'try_sample', label: t('nav.samples') }] }));
      announce(err.text(), true);
      focusIfLost(viewSlot.querySelector('h1'));
      return;
    }
    // every series failed: one card per series with its own fixes, as for partial failures (FU-2)
    if (Array.isArray(err.failed) && err.failed.length > 1) {
      viewSlot.append(viewHead(titleText() || t('app.name')),
        h('p', { class: 'all-failed-note' }, t('compare.all_failed')),
        h('div', { class: 'rank-grid' }, err.failed.map((entry) => failedCard(entry, fixCtx))));
      announce(t('compare.all_failed'), true);
      focusIfLost(viewSlot.querySelector('h1'));
      return;
    }
    const auto = err.code === 'naver_rate' && !autoRetried.has(key);
    if (auto) autoRetried.add(key);
    viewSlot.append(viewHead(titleText() || t('app.name')), errorCard(err, fixCtx, { autoRetry: auto }));
    announce(err.text(), true);
    focusIfLost(viewSlot.querySelector('h1'));
  }

  function showCancelled() {
    destroyProgress();
    destroyView();
    clear(viewSlot);
    const empty = emptyState({
      title: t('progress.cancelled'), level: 2, icon: 'ui-info', focusable: true,
      actions: [h('button', { type: 'button', class: 'btn btn-primary', on: { click: () => fixCtx.reanalyze() } }, t('chip.reanalyze_upload'))],
    });
    viewSlot.append(viewHead(titleText() || t('app.name')), empty);
    announce(t('progress.cancelled'));
    // 취소 itself is gone with the progress panel: focus the message that replaced it (A11Y-2)
    focusIfLost(empty.querySelector('.empty-title'));
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
    // a search job's poll sends the DELETE itself when aborted; a result page's poll does not (cancelOnAbort: false).
    // One DELETE per subscription: a second one would withdraw another tab's subscription to a joined job.
    const pollDeletes = kind === 'explore' && controller && running;
    if (jobId && !pollDeletes) api.cancelJob(jobId);
    running = false;
    if (controller) controller.abort();
    controller = null;
    showCancelled();
  }

  // ------------------------------------------------------------------ closing / reloading the tab (OPS-7)
  function onPageHide() {
    if (kind !== 'explore' || !running || !jobId) return;
    api.cancelJob(jobId);          // keepalive DELETE: the job stops holding a worker for a tab that is gone
    cancelledOnHide = true;
  }
  function onPageShow(e) {
    // back from the bfcache after pagehide cancelled the job: start it again instead of showing "cancelled"
    if (e.persisted && cancelledOnHide) {
      cancelledOnHide = false;
      run();
    }
  }
  window.addEventListener('pagehide', onPageHide);
  window.addEventListener('pageshow', onPageShow);

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
    try {
      const next = await api.analyze(state, {
        signal: controller.signal, hl: getLang, defaultSource: config.default_source,
        onStart: (start) => { jobId = start.job_id; running = start.status !== 'done'; },
        onUpdate: onSnapshot,
      });
      running = false;
      remember(next);
      showEnvelope(next, { done: true });
    } catch (err) {
      running = false;
      throw err;
    }
  }

  async function loadResult() {
    startProgress(1);
    try {
      // a result page only stops polling when left: the upload job keeps running (no DELETE), only 취소 cancels it
      const next = await api.pollJob(state.id, { signal: controller.signal, hl: getLang, onUpdate: onSnapshot, cancelOnAbort: false });
      remember(next);
      showEnvelope(next, { done: true });
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
    qb.el.hidden = false;
    destroyProgress();
    destroyView();
    const cached = ctx.results.get(cacheKey);
    if (cached) {
      showEnvelope(cached);
      return;
    }
    rerender(() => {
      clear(viewSlot);
      viewSlot.appendChild(skeleton(kind === 'explore' && state.q.length > 1 ? 'compare' : 'detail', titleText(), state.q.length));
    });
    const load = kind === 'explore' ? loadExplore : kind === 'result' ? loadResult : loadShared;
    load().catch((err) => {
      if (mine !== controller || !err || err.name === 'AbortError') return;
      if (err.cancelled && cancelledOnHide) return;                  // pageshow restarts it
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

  /** Back from a focused series to the comparison: focus that series' 자세히 › link where it is (A11Y-14). */
  function focusCard(name) {
    const vm = env && env.series.find((x) => x.name === name);
    const link = vm ? viewSlot.querySelector(`.rank-card[data-index="${vm.index}"] .more-link`) : null;
    if (!link) return false;
    link.focus({ preventScroll: true });
    // the browser restores the scroll position of a history entry after this handler: check once it has
    requestAnimationFrame(() => {
      const r = link.getBoundingClientRect();
      if (r.top < 0 || r.bottom > window.innerHeight) link.scrollIntoView({ block: 'center' });
    });
    return true;
  }

  return {
    el,
    update(next) {
      if (next.lang !== lang) return false;
      if (kind === 'explore' && queryKey(next.state, { defaultSource: config.default_source }) !== key) return false;
      if (kind !== 'explore' && next.state.id !== state.id) return false;
      const focusChanged = next.state.focus !== state.focus;
      const previous = state.focus;
      state = next.state;
      if (focusChanged && env) {
        renderEnvelope();
        if (state.focus || !focusCard(previous)) {
          window.scrollTo(0, 0);
          const h1 = viewSlot.querySelector('h1');
          if (h1) {
            h1.setAttribute('tabindex', '-1');
            h1.focus({ preventScroll: true });
          }
        }
        announce(document.title);
      } else if (view && view.update) {
        view.update(state);
      }
      return true;
    },
    destroy() {
      window.removeEventListener('pagehide', onPageHide);
      window.removeEventListener('pageshow', onPageShow);
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
