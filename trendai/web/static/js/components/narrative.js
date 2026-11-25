// "해석" card (spec F.6.9, D.2.9, C.11): badge, Markdown-lite body with date links (→ chart crosshair), and the AI
// rewrite. `AI로 다시 쓰기` appears when ai_allowed, else a disabled button with ai_block_text under it — except on a
// read-only shared link, which shows no AI button at all, only the ai_block_text (F.6.11). While the AI job
// runs the rule text stays and a status line says so; the result replaces the body with the AI badge + note, and
// `규칙 기반 해석 보기` toggles back. Owner: WP3-app.

import * as api from '../api.js';
import { clear, h } from '../dom.js';
import { getLang, t } from '../i18n.js';
import { md } from '../md.js';
import { cardHead, icon, pill } from '../ui.js';
import { errorCard } from './errorcard.js';

/**
 * @param {Object} vm  D.2 series VM
 * @param {{resultId: string|null, onDateLink?: (iso: string) => void,
 *          onModeChange?: (mode: 'rule'|'ai', caveatsMarkdown: string|null) => void,
 *          navigate?: Function}} opts
 * @returns {{el: HTMLElement, destroy: () => void}}
 */
export function createNarrative(vm, opts) {
  const n = vm.narrative;
  let ai = n.ai || null;
  let mode = ai ? 'ai' : 'rule';
  let controller = null;

  const badgeSlot = h('span', { class: 'narrative-badge' });
  const status = h('p', { class: 'narrative-status', 'aria-live': 'polite' });
  const body = h('div', { class: 'prose narrative-body' });
  const note = h('p', { class: 'muted small narrative-note' });
  const errorSlot = h('div');
  const toggle = h('button', { type: 'button', class: 'btn btn-secondary btn-small', hidden: true, on: { click: () => setMode(mode === 'ai' ? 'rule' : 'ai') } });
  const aiButton = h('button', {
    type: 'button', class: 'btn btn-secondary', disabled: n.ai_allowed ? null : true,
    'aria-describedby': n.ai_allowed ? null : `ai-block-${vm.index}`,
    on: { click: () => runAi() },
  }, icon('ui-sparkle'), t('detail.narrative.ai'));
  const readonly = n.ai_block_reason === 'shared_readonly';
  const aiRow = h('div', { class: 'ai-row' },
    h('div', { class: 'btn-row' }, ai || readonly ? null : aiButton, toggle),
    n.ai_allowed
      ? (n.ai_privacy_note ? h('p', { class: 'muted small' }, n.ai_privacy_note) : null)
      : h('p', { class: 'muted small', id: `ai-block-${vm.index}` }, n.ai_block_text));

  body.addEventListener('click', (e) => {
    const link = e.target instanceof Element ? e.target.closest('button.datelink') : null;
    if (link && opts.onDateLink) opts.onDateLink(link.dataset.date);
  });

  function render() {
    clear(body);
    const showAi = mode === 'ai' && ai;
    body.appendChild(md(showAi ? ai.markdown : n.markdown, n.date_links));
    clear(badgeSlot);
    badgeSlot.appendChild(pill(showAi ? ai.badge : n.badge, showAi ? 'pill-ai' : 'pill-muted'));
    note.textContent = showAi ? (ai.note || '') : '';
    note.hidden = !showAi;
    toggle.hidden = !ai;
    toggle.textContent = t(mode === 'ai' ? 'detail.narrative.show_rule' : 'detail.narrative.show_ai');
    if (opts.onModeChange) opts.onModeChange(showAi ? 'ai' : 'rule', showAi ? ai.caveats_markdown : null);
  }

  function setMode(next) {
    mode = next;
    render();
  }

  async function runAi() {
    if (!n.ai_allowed || !opts.resultId || controller) return;
    controller = new AbortController();
    aiButton.disabled = true;
    clear(errorSlot);
    status.textContent = t('detail.narrative.ai_running');
    try {
      ai = await api.narrative(opts.resultId, vm.index, getLang(), { signal: controller.signal });
      status.textContent = '';
      aiButton.remove();
      mode = 'ai';
      render();
    } catch (err) {
      if (err && err.name === 'AbortError') return;
      status.textContent = '';
      aiButton.disabled = false;
      if (err && typeof err.text === 'function') {
        errorSlot.appendChild(errorCard(err, { navigate: opts.navigate || (() => {}), retry: () => runAi() }, { compact: true }));
      }
    } finally {
      controller = null;
    }
  }

  render();
  return {
    el: h('section', { class: 'card', id: 'narrative', 'aria-labelledby': 'narrative-title' },
      cardHead({ title: t('detail.narrative.title'), id: 'narrative-title', actions: [badgeSlot] }),
      status, body, note, errorSlot, aiRow),
    destroy() {
      if (controller) controller.abort();
    },
  };
}
