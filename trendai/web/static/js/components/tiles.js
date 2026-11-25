// KPI tiles (spec F.6.9, D.2.4): always 6 — label + `?` (the VM tooltip) + value + sub; an undefined value shows `—`
// with its reason under it. 6 columns ≥ 1024 px, 3 at 640–1023, 2 below 640 (CSS). Owner: WP3-app.

import { h } from '../dom.js';
import { t } from '../i18n.js';
import { arrowValue, icon, srOnly, toneClass } from '../ui.js';
import { helpButton } from './help.js';

/**
 * @param {Object} vm  D.2 series VM
 * @returns {HTMLElement}
 */
export function renderTiles(vm) {
  return h('section', { class: 'tiles-block', 'aria-labelledby': `tiles-${vm.index}` },
    h('h2', { class: 'sr-only', id: `tiles-${vm.index}` }, t('detail.tiles.title')),
    h('ul', { class: 'tiles' }, (vm.tiles || []).map((tile) => {
      const undefinedValue = Boolean(tile.undefined_reason);
      return h('li', { class: ['tile', `tile-${tile.key}`], dataset: { key: tile.key } },
        h('div', { class: 'tile-label' },
          h('span', null, tile.label),
          tile.tooltip ? helpButton(tile.label, tile.tooltip) : null),
        h('p', { class: ['tile-value', toneClass(tile.tone), undefinedValue && 'is-undefined', String(tile.value || '').length >= 8 && 'is-long'] },
          tile.icon ? icon(`ph-${tile.icon}`, { cls: ['phase-icon', `ph-${tile.icon}`] }) : null,
          undefinedValue ? [h('span', { 'aria-hidden': 'true' }, tile.value), srOnly(t('common.dash_reason', { reason: tile.undefined_reason }))]
            : arrowValue(tile.value)),
        undefinedValue ? h('p', { class: 'tile-reason' }, tile.undefined_reason) : null,
        tile.sub ? h('p', { class: 'tile-sub' }, tile.sub) : null);
    })));
}
