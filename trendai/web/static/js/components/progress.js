// Progress of an analysis job (spec F.6.5, C.6, brief §3.7). Owner: WP3-app.
//
// The page draws skeletons of the final layout immediately. After 1 s a status line appears (stage-specific text) and
// the 취소 button; after 3 s — or at once when series_total ≥ 3 — the 4-step panel (role="status"): ① 데이터 확인
// (stage data) ② 추세·구조 분석 (current.step analyze) ③ 예측 모델 검증 (k/N) (forecast / narrative) ④ 정리 (finish),
// each pending / active / done with an icon and a text state (not colour only). No fake progress bars.

import { h, clear } from '../dom.js';
import { t } from '../i18n.js';
import { icon, srOnly } from '../ui.js';

const STATUS_DELAY_MS = 1000;
const PANEL_DELAY_MS = 3000;
const STATE_ICON = { pending: 'ui-ring', active: 'ui-dot', done: 'ui-check' };

/** Step states for a C.6 snapshot. */
export function stepStates(snap) {
  const stage = snap ? snap.stage : 'queued';
  const step = snap && snap.current ? snap.current.step : null;
  if (stage === 'done') return ['done', 'done', 'done', 'done'];
  if (stage === 'finish') return ['done', 'done', 'done', 'active'];
  if (stage === 'series') {
    const forecasting = step === 'forecast' || step === 'narrative';
    return ['done', forecasting ? 'done' : 'active',
      forecasting || (snap.series_done || 0) > 0 ? 'active' : 'pending', 'pending'];
  }
  return ['active', 'pending', 'pending', 'pending'];
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
 * @returns {{el: HTMLElement, update: (snap: Object) => void, destroy: () => void}}
 */
export function createProgress(opts) {
  let snap = null;
  let total = opts.total || 1;
  const status = h('p', { class: 'progress-status', hidden: true },
    h('span', { class: 'spinner', 'aria-hidden': 'true' }), h('span', { class: 'progress-status-text' }));
  const steps = h('ol', { class: 'progress-steps', role: 'status', 'aria-label': t('progress.label'), hidden: true });
  const cancel = opts.onCancel ? h('button', {
    type: 'button', class: 'btn btn-secondary btn-small progress-cancel', hidden: true,
    on: { click: () => opts.onCancel() },
  }, t('progress.cancel')) : null;
  const el = h('div', { class: 'progress' }, h('div', { class: 'progress-row' }, status, cancel), steps);

  function render() {
    status.querySelector('.progress-status-text').textContent = statusText(snap, opts.source);
    const states = stepStates(snap);
    const k = snap ? snap.series_done || 0 : 0;
    const labels = [t('progress.step.data'), t('progress.step.analyze'),
      t('progress.step.forecast', { k, n: total }), t('progress.step.finish')];
    clear(steps);
    labels.forEach((label, i) => {
      steps.appendChild(h('li', { class: ['progress-step', `is-${states[i]}`] },
        h('span', { class: 'progress-step-icon' }, icon(STATE_ICON[states[i]])),
        h('span', { class: 'progress-step-label' }, label),
        srOnly(` · ${t(`progress.state.${states[i]}`)}`)));
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
