// Progress of an analysis job (spec F.6.5, C.6, brief §3.7). Owner: WP3-app.
//
// The page draws skeletons of the final layout immediately. After 1 s a status line appears (stage-specific text) and
// the 취소 button; after 3 s — or at once when series_total ≥ 3 — the 4-step panel (role="status"): ① 데이터 확인
// (stage data) ② 추세·구조 분석 (current.step analyze) ③ 예측 모델 검증 (k/N) (forecast / narrative) ④ 정리 (finish),
// each pending / active / done with an icon and a text state (not colour only). No fake progress bars.
//
// The four steps are built once and updated in place (A11Y-6): a poll that changes nothing changes no node, and a
// step that changes says only "<step>: <state>" (aria-atomic="false"), instead of the whole panel on every poll.
// Step states never go backwards (② stays 완료 once the first series has passed it).

import { h, svg } from '../dom.js';
import { t } from '../i18n.js';
import { SPRITE } from '../ui.js';

const STATUS_DELAY_MS = 1000;
const PANEL_DELAY_MS = 3000;
const STATE_ICON = { pending: 'ui-ring', active: 'ui-dot', done: 'ui-check' };
const RANK = { pending: 0, active: 1, done: 2 };

/** Step states for a C.6 snapshot. */
export function stepStates(snap) {
  const stage = snap ? snap.stage : 'queued';
  const step = snap && snap.current ? snap.current.step : null;
  if (stage === 'done') return ['done', 'done', 'done', 'done'];
  if (stage === 'finish') return ['done', 'done', 'done', 'active'];
  if (stage === 'series') {
    const forecasting = step === 'forecast' || step === 'narrative';
    const someDone = (snap.series_done || 0) > 0;
    return ['done', forecasting || someDone ? 'done' : 'active', forecasting || someDone ? 'active' : 'pending', 'pending'];
  }
  return ['active', 'pending', 'pending', 'pending'];
}

/** Monotonic merge: a step never goes back from done to active or from active to pending. */
export function mergeStates(prev, next) {
  if (!prev) return next.slice();
  return next.map((s, i) => (RANK[prev[i]] > RANK[s] ? prev[i] : s));
}

/** The status line text for a snapshot. */
export function statusText(snap, source) {
  const stage = snap ? snap.stage : 'queued';
  if (stage === 'queued') return t('progress.queued');
  if (stage === 'data') {
    if (source === 'naver') return t('progress.fetching_naver');
    if (source === 'wiki') return t('progress.fetching_wiki');
    return t('progress.preparing');
  }
  if (stage === 'series' && snap.current) {
    const n = snap.series_total || 1;
    return t('progress.series', { name: snap.current.name, k: Math.min(n, (snap.series_done || 0) + 1), n });
  }
  if (stage === 'finish' || stage === 'done') return t('progress.finishing');
  return t('progress.preparing');
}

/**
 * @param {{total: number, source?: string, onCancel?: () => void}} opts
 * @returns {{el: HTMLElement, update: (snap: Object) => void, destroy: () => void, cancelButton: HTMLElement|null}}
 */
export function createProgress(opts) {
  let snap = null;
  let total = opts.total || 1;
  let states = null;
  const statusLine = h('span', { class: 'progress-status-text' });
  const status = h('p', { class: 'progress-status', hidden: true }, h('span', { class: 'spinner', 'aria-hidden': 'true' }), statusLine);
  const items = [0, 1, 2, 3].map(() => {
    const use = svg('use');
    const svgIcon = svg('svg', { class: 'icon', 'aria-hidden': 'true', focusable: 'false' }, use);
    const label = h('span', { class: 'progress-step-label', 'aria-hidden': 'true' });
    const spoken = h('span', { class: 'sr-only' });
    const li = h('li', { class: 'progress-step' }, h('span', { class: 'progress-step-icon' }, svgIcon), label, spoken);
    return { li, use, label, spoken, state: null, text: null };
  });
  const steps = h('div', {
    class: 'progress-steps', role: 'status', 'aria-label': t('progress.label'), 'aria-atomic': 'false', hidden: true,
  }, h('ol', { class: 'progress-step-list' }, items.map((it) => it.li)));
  const cancel = opts.onCancel ? h('button', {
    type: 'button', class: 'btn btn-secondary btn-small progress-cancel', hidden: true,
    on: { click: () => opts.onCancel() },
  }, t('progress.cancel')) : null;
  const el = h('div', { class: 'progress' }, h('div', { class: 'progress-row' }, status, cancel), steps);

  function render() {
    const line = statusText(snap, opts.source);
    if (statusLine.textContent !== line) statusLine.textContent = line;
    states = mergeStates(states, stepStates(snap));
    const k = snap ? snap.series_done || 0 : 0;
    const labels = [t('progress.step.data'), t('progress.step.analyze'),
      t('progress.step.forecast', { k, n: total }), t('progress.step.finish')];
    items.forEach((it, i) => {
      const state = states[i];
      if (it.state !== state) {
        it.li.className = `progress-step is-${state}`;
        it.use.setAttribute('href', `${SPRITE}#${STATE_ICON[state]}`);
      }
      const spoken = `${labels[i]}: ${t(`progress.state.${state}`)}`;
      if (it.text !== labels[i]) it.label.textContent = labels[i];
      if (it.state !== state || it.text !== labels[i]) it.spoken.textContent = spoken;
      it.state = state;
      it.text = labels[i];
    });
  }

  render();
  const showStatus = setTimeout(() => {
    status.hidden = false;
    if (cancel) cancel.hidden = false;
  }, STATUS_DELAY_MS);
  const showPanel = setTimeout(() => { steps.hidden = false; }, PANEL_DELAY_MS);
  if (total >= 3) steps.hidden = false;

  return {
    el,
    cancelButton: cancel,
    update(next) {
      snap = next;
      if (next && next.series_total) total = next.series_total;
      if (total >= 3) steps.hidden = false;
      render();
    },
    destroy() {
      clearTimeout(showStatus);
      clearTimeout(showPanel);
    },
  };
}
