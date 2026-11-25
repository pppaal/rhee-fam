// Info pages (spec F.6.16, G.10–G.12): /method, /sources, /privacy. Headings carry ids for deep links (#phase,
// #bands, #probability, #accuracy, #metrics, #scores, #index, …). The prose lives in catalog.js (both languages) as
// Markdown-lite and is rendered with md.js, i.e. as text nodes only. Owner: WP3-app.

import { h } from '../dom.js';
import { t } from '../i18n.js';
import { md } from '../md.js';
import { icon } from '../ui.js';

const PHASES = ['emerging', 'growing', 'peaking', 'declining', 'stable', 'reviving', 'volatile'];

const PAGES = {
  method: {
    title: 'title.method', intro: 'method.intro', toc: true,
    sections: ['overview', 'phase', 'bands', 'probability', 'accuracy', 'metrics', 'scores', 'index', 'limits', 'ai']
      .map((id) => ({ id, prefix: `method.${id}` })),
  },
  sources: {
    title: 'title.sources', intro: 'sources.intro', footer: 'sources.footer',
    sections: ['naver', 'wiki', 'gt', 'upload', 'sample'].map((id) => ({ id, prefix: `sources.${id}` })),
  },
  privacy: {
    title: 'title.privacy', intro: 'privacy.intro',
    sections: ['upload', 'results', 'share', 'logs', 'ai', 'storage', 'contact'].map((id) => ({ id, prefix: `privacy.${id}` })),
  },
};

function phaseSection() {
  return [
    h('p', null, t('method.phase.intro')),
    h('ul', { class: 'phase-rules' }, PHASES.map((p) => h('li', { class: `ph-${p}` },
      h('span', { class: 'phase-badge' }, icon(`ph-${p}`, { cls: 'phase-icon' }), h('span', { class: 'phase-name' }, t(`phase.${p}`))),
      h('span', { class: 'phase-rule' }, t(`method.phase.rule.${p}`))))),
    md(t('method.phase.notes')),
  ];
}

export function create(ctx) {
  const def = PAGES[ctx.state.route] || PAGES.method;
  ctx.setTitle(`${t(def.title)} · trendai`);
  const sections = def.sections.map((s) => h('section', { class: 'info-section', id: s.id, 'aria-labelledby': `${s.id}-h` },
    h('h2', { id: `${s.id}-h` }, t(`${s.prefix}.title`)),
    s.id === 'phase' && ctx.state.route === 'method' ? phaseSection() : md(t(`${s.prefix}.body`))));
  const toc = def.toc ? h('nav', { class: 'info-toc', 'aria-label': t('info.toc') },
    h('p', { class: 'info-toc-title' }, t('info.toc')),
    h('ol', null, def.sections.map((s) => h('li', null, h('a', { href: `#${s.id}` }, t(`${s.prefix}.title`)))))) : null;
  return {
    el: h('div', { class: ['page', 'info-page', def.toc && 'has-toc'] },
      h('div', { class: 'page-head' },
        h('h1', { class: 'page-title' }, t(def.title)),
        h('p', { class: 'page-sub prose' }, t(def.intro))),
      h('div', { class: 'info-layout' },
        toc,
        h('div', { class: 'prose info-body' }, sections,
          def.footer ? h('p', { class: 'info-footer muted' }, t(def.footer)) : null))),
    update(next) {
      // in-page anchors only change the hash; the router scrolls
      return next.state.route === ctx.state.route && next.state.hl === ctx.state.hl;
    },
  };
}
