// Samples page (spec F.6.15): 9 cards (title, description, sparkline, 가상 데이터 badge) with checkboxes for a
// multi-select of up to 5 and a sticky `선택한 샘플 비교 (n)` button → /explore?src=sample&q=a,b,c. Owner: WP3-app.

import * as api from '../api.js';
import { clear, h } from '../dom.js';
import { getLang, t } from '../i18n.js';
import { MAX_SERIES, emptyState, serialize } from '../state.js';
import { pill } from '../ui.js';
import { sparkline } from '../chart/sparkline.js';
import { errorCard } from '../components/errorcard.js';
import { toast } from '../components/toast.js';

const cache = new Map();

/** GET /api/samples?hl= once per language (C.3 is computed once at app start, so it never changes). */
export function loadSamples(hl) {
  if (!cache.has(hl)) {
    const p = api.samples(hl).then((data) => (data && Array.isArray(data.samples) ? data.samples : []));
    p.catch(() => cache.delete(hl));
    cache.set(hl, p);
  }
  return cache.get(hl);
}

export function create(ctx) {
  ctx.setTitle(`${t('title.samples')} · trendai`);
  const selected = new Set();
  const grid = h('div', { class: 'sample-grid', 'aria-busy': 'true' },
    Array.from({ length: 9 }, () => h('div', { class: 'card sample-card skeleton-card' },
      h('div', { class: 'skeleton skeleton-line w-40' }), h('div', { class: 'skeleton skeleton-line w-80' }),
      h('div', { class: 'skeleton skeleton-block h-32' }))));
  const compareBtn = h('button', {
    type: 'button', class: 'btn btn-primary', disabled: true,
    on: {
      click: () => {
        if (!selected.size) return;
        const s = emptyState('explore');
        s.src = 'sample';
        s.q = [...selected].map((name) => [name]);
        s.hl = ctx.state.hl;
        ctx.go(s);
      },
    },
  }, t('samples.compare', { n: 0 }));
  const maxNote = h('p', { class: 'muted small', 'aria-live': 'polite' });
  const bar = h('div', { class: 'sticky-action' }, maxNote, compareBtn);

  function refresh() {
    compareBtn.textContent = t('samples.compare', { n: selected.size });
    compareBtn.disabled = selected.size === 0;
    const full = selected.size >= MAX_SERIES;
    maxNote.textContent = full ? t('samples.max') : '';
    for (const box of grid.querySelectorAll('input[type=checkbox]')) {
      box.disabled = full && !box.checked;
    }
  }

  function card(s) {
    const st = emptyState('explore');
    st.src = 'sample';
    st.q = [[s.name]];
    st.hl = ctx.state.hl;
    const id = `sample-${s.name}`;
    return h('article', { class: 'card sample-card', 'aria-labelledby': id },
      h('div', { class: 'sample-card-head' },
        h('h2', { class: 'sample-title', id }, h('a', { href: serialize(st) }, s.title)),
        pill(t('common.synthetic'), 'pill-synthetic')),
      h('p', { class: 'sample-desc' }, s.description),
      h('div', { class: 'sample-spark' }, sparkline(s.spark || [], { width: 240, height: 48 })),
      h('label', { class: 'check' },
        h('input', {
          type: 'checkbox', value: s.name,
          on: {
            change: (e) => {
              if (e.target.checked) {
                if (selected.size >= MAX_SERIES) {
                  e.target.checked = false;
                  toast(t('samples.max'), { kind: 'warning' });
                  return;
                }
                selected.add(s.name);
              } else selected.delete(s.name);
              refresh();
            },
          },
        }),
        h('span', null, t('samples.select')),
        h('span', { class: 'sr-only' }, ` ${s.title}`)));
  }

  function load() {
    loadSamples(getLang()).then((list) => {
      clear(grid);
      grid.removeAttribute('aria-busy');
      grid.append(...list.map(card));
      refresh();
    }).catch((err) => {
      clear(grid);
      grid.removeAttribute('aria-busy');
      grid.appendChild(errorCard(err, { navigate: ctx.navigate, retry: load }));
    });
  }
  load();

  return {
    el: h('div', { class: 'page samples-page' },
      h('div', { class: 'page-head' },
        h('h1', { class: 'page-title' }, t('samples.title')),
        h('p', { class: 'page-sub' }, t('samples.intro'))),
      grid,
      bar),
  };
}
