// "주의할 점" card (spec F.6.9, D.2.9): the caveat Markdown (rule-based, or the AI version's caveats while the AI
// interpretation is shown) plus the fixed disclaimer. Owner: WP3-app.

import { clear, h } from '../dom.js';
import { t } from '../i18n.js';
import { md } from '../md.js';
import { cardHead } from '../ui.js';

/**
 * @param {Object} vm  D.2 series VM
 * @returns {{el: HTMLElement, setMarkdown: (markdown: string|null) => void}}
 */
export function createCaveats(vm) {
  const body = h('div', { class: 'prose' });
  const fallback = (vm.caveats && vm.caveats.markdown) || '';
  function setMarkdown(markdown) {
    clear(body);
    const text = markdown == null ? fallback : markdown;
    if (text) body.appendChild(md(text, vm.narrative && vm.narrative.date_links));
  }
  setMarkdown(null);
  return {
    el: h('section', { class: 'card', id: 'caveats', 'aria-labelledby': 'caveats-title' },
      cardHead({ title: t('detail.caveats.title'), id: 'caveats-title' }),
      body,
      vm.caveats && vm.caveats.disclaimer ? h('p', { class: 'disclaimer' }, vm.caveats.disclaimer) : null),
    setMarkdown,
  };
}
