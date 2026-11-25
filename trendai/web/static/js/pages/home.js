// Home (spec F.6.2, brief §3.3): hero text, search box with chips, source switch, hint line, example chips per
// source, upload card (drop zone + 파일 선택 + 붙여넣기 → /upload with the file or text handed over in memory),
// samples card (9 chips with the 가상 데이터 badge) and recent analyses. `src` and `q` in the URL pre-fill the source
// switch and the search box and never start an analysis (the use_wiki fix, C.12.6). Owner: WP3-app.
//
// Wikipedia has a language choice (한국어 / English → `wl`, FU-1) next to the source switch; it starts at the URL's
// `wl`, else at the UI language, and changes the example titles and the placeholder.

import { h, clear } from '../dom.js';
import { getLang, t } from '../i18n.js';
import { effectiveSource, emptyState, serialize, withState } from '../state.js';
import { pill, rovingRadios } from '../ui.js';
import { createSearchBox } from '../components/searchbox.js';
import { dropZone } from '../components/dropzone.js';
import { renderRecent } from '../components/recent.js';
import { helpButton } from '../components/help.js';
import { loadSamples } from './samples.js';
import { openPasteDialog, setPendingUpload, uploadLimits } from './upload.js';

const SOURCES = ['naver', 'wiki', 'upload', 'sample'];
const WIKI_LANGS = ['ko', 'en'];

/** The Wikipedia language home starts with: the URL's `wl`, else the UI language (FU-1). */
export function initialWikiLang(search, uiLang) {
  const wl = new URLSearchParams(String(search || '').replace(/^\?/, '')).get('wl');
  if (wl && /^[a-z]{2,3}$/.test(wl)) return wl;
  return uiLang === 'en' ? 'en' : 'ko';
}

/** Naver availability from /api/config: {available, selectable, reasonKey}. */
export function naverStatus(config) {
  const n = (config && config.sources && config.sources.naver) || {};
  if (n.available) return { available: true, selectable: true, reasonKey: null };
  if (n.reason === 'budget_exhausted') return { available: false, selectable: true, reasonKey: 'source.naver.budget' };
  return { available: false, selectable: false, reasonKey: 'source.naver.unconfigured' };
}

/** The initially selected source of the switch: the URL's `src` when selectable, else the config default. */
function initialSource(state, config) {
  const naver = naverStatus(config);
  const wanted = state.src || effectiveSource(state, config.default_source);
  if (wanted === 'naver' && !naver.selectable) return 'wiki';
  return wanted;
}

export function create(ctx) {
  const { config, state } = ctx;
  ctx.setTitle(t('title.home'));
  const naver = naverStatus(config);
  let source = initialSource(state, config);
  let wikiLang = initialWikiLang(window.location.search, getLang());
  let blurTimer = null;

  const page = h('div', { class: ['page', 'home', source === 'sample' && 'is-sample'] });
  const box = createSearchBox({
    source,
    wikiLang,
    series: source === 'sample' ? state.q.map((terms) => terms.slice(0, 1)) : state.q,
    onSubmit: (series) => {
      const next = emptyState('explore');
      next.src = source;
      next.q = series;
      next.hl = state.hl;
      if (source === 'wiki') next.wl = wikiLang;
      ctx.go(next);
    },
    onFocusChange: (focused) => {
      clearTimeout(blurTimer);
      if (focused) page.classList.add('is-searching');
      else blurTimer = setTimeout(() => page.classList.remove('is-searching'), 200);
    },
  });

  // ------------------------------------------------------------------ source switch (radiogroup, roving tabindex)
  const radios = SOURCES.map((src) => {
    const disabled = src === 'naver' && !naver.selectable;
    return h('button', {
      type: 'button', role: 'radio', class: 'source-option', dataset: { source: src },
      'aria-checked': String(src === source), 'aria-disabled': disabled ? 'true' : null,
      tabindex: src === source ? '0' : '-1',
      on: { click: () => choose(src) },
    }, t(`source.${src}`), src === 'naver' && !naver.available
      ? h('span', { class: 'source-badge' }, t('source.unavailable')) : null);
  });
  rovingRadios(radios, (next) => choose(next.dataset.source),
    { usable: (r) => r.getAttribute('aria-disabled') !== 'true' && r.dataset.source !== 'upload' });

  // Wikipedia language (FU-1): a small radiogroup shown while 위키백과 is the source
  const langs = WIKI_LANGS.includes(wikiLang) ? WIKI_LANGS : [...WIKI_LANGS, wikiLang];
  const langRadios = langs.map((wl) => h('button', {
    type: 'button', role: 'radio', class: 'segment', dataset: { wl },
    'aria-checked': String(wl === wikiLang), tabindex: wl === wikiLang ? '0' : '-1',
    on: { click: () => chooseWikiLang(wl) },
  }, WIKI_LANGS.includes(wl) ? t(`wiki.lang.${wl}`) : wl));
  rovingRadios(langRadios, (b) => chooseWikiLang(b.dataset.wl));
  const wikiLangEl = h('div', { class: 'wiki-lang', hidden: source !== 'wiki' },
    h('span', { class: 'field-label inline', id: 'wiki-lang-label' }, t('wiki.lang.label')),
    h('div', { class: 'segmented', role: 'radiogroup', 'aria-labelledby': 'wiki-lang-label' }, langRadios));

  const switchEl = h('div', { class: 'source-switch-wrap' },
    h('div', { class: 'source-switch', role: 'radiogroup', 'aria-label': t('source.label') }, radios),
    naver.reasonKey ? helpButton(t('source.naver'), t(naver.reasonKey), { className: 'source-help' }) : null,
    wikiLangEl);

  function chooseWikiLang(wl) {
    wikiLang = wl;
    for (const b of langRadios) {
      const on = b.dataset.wl === wl;
      b.setAttribute('aria-checked', String(on));
      b.tabIndex = on ? 0 : -1;
    }
    box.setWikiLang(wl);
    renderExamples();
    ctx.navigate(serialize(withState(state, { src: source, q: box.getSeries(), wl })), { replace: true, silent: true });
  }

  function choose(src) {
    if (src === 'naver' && !naver.selectable) return;
    if (src === 'upload') {
      ctx.navigate('/upload');
      return;
    }
    source = src;
    for (const r of radios) {
      const on = r.dataset.source === src;
      r.setAttribute('aria-checked', String(on));
      r.tabIndex = on ? 0 : -1;
    }
    box.setSource(src);
    wikiLangEl.hidden = src !== 'wiki';
    page.classList.toggle('is-sample', src === 'sample');
    renderExamples();
    const next = withState(state, { src, q: box.getSeries(), wl: src === 'wiki' ? wikiLang : 'ko' });
    ctx.navigate(serialize(next), { replace: true, silent: true });
  }

  // ------------------------------------------------------------------ examples per source
  const examples = h('div', { class: 'examples' });
  function renderExamples() {
    clear(examples);
    if (source !== 'naver' && source !== 'wiki') return;
    const key = source === 'wiki' && wikiLang === 'en' ? 'home.example_terms.wiki_en' : `home.example_terms.${source}`;
    const terms = t(key).split(',').map((s) => s.trim()).filter(Boolean);
    examples.append(
      h('span', { class: 'examples-label' }, t('home.examples')),
      h('ul', { class: 'chip-row' }, terms.map((term) => {
        const s = emptyState('explore');
        s.src = source;
        s.q = [[term]];
        s.hl = state.hl;
        if (source === 'wiki') s.wl = wikiLang;
        return h('li', null, h('a', { class: 'chip chip-link', href: serialize(s) }, term));
      })));
  }
  renderExamples();

  // ------------------------------------------------------------------ upload card
  function handOver(fileOrText) {
    setPendingUpload(fileOrText);
    ctx.navigate('/upload');
  }
  const limits = uploadLimits(config);
  const uploadCard = h('section', { class: 'card home-card', 'aria-labelledby': 'home-upload-title' },
    h('h2', { class: 'card-title', id: 'home-upload-title' }, t('home.upload.title')),
    dropZone({
      compact: true, text: h('p', null, t('home.upload.body')),
      chooseLabel: t('home.upload.choose'), pasteLabel: t('home.upload.paste'),
      onFile: (file) => handOver(file),
      onPaste: (opener) => openPasteDialog({ opener, onRead: (text) => handOver(text) }),
    }),
    h('p', { class: 'muted small' }, t('upload.formats', { mb: limits.mb, n: limits.cols })));

  // ------------------------------------------------------------------ samples card
  const sampleChips = h('ul', { class: 'chip-row sample-chips' });
  const samplesCard = h('section', { class: 'card home-card home-samples-card', 'aria-labelledby': 'home-samples-title' },
    h('div', { class: 'card-title-row' },
      h('h2', { class: 'card-title', id: 'home-samples-title' }, t('home.samples.title')),
      pill(t('common.synthetic'), 'pill-synthetic')),
    sampleChips);
  loadSamples(getLang()).then((list) => {
    box.setSamples(list.map((s) => ({ name: s.name, title: s.title })));
    sampleChips.append(...list.map((s) => {
      const st = emptyState('explore');
      st.src = 'sample';
      st.q = [[s.name]];
      st.hl = state.hl;
      return h('li', null, h('a', { class: 'chip chip-link', href: serialize(st), title: s.description }, s.title));
    }));
  }).catch(() => {
    sampleChips.append(h('li', null, h('a', { class: 'chip chip-link', href: '/samples' }, t('nav.samples'))));
  });

  const recent = h('section', { class: 'recent' });
  renderRecent(recent);

  page.append(
    h('section', { class: 'home-hero' },
      h('h1', { class: 'home-title' }, t('home.title')),
      h('p', { class: 'home-sub' }, t('home.sub')),
      h('div', { class: 'search-panel' }, box.el, switchEl, examples)),
    h('div', { class: 'home-cards' }, uploadCard, samplesCard),
    recent);

  // The use_wiki / use_naver fixes land here with a pre-filled box: say that article titles are needed.
  if (source === 'wiki' && state.q.length) box.showHint(t('search.hint.wiki_title'));

  return {
    el: page,
    update(next) {
      // only the silent src rewrite reaches here as a replace; anything else re-creates the page
      return next.info.type === 'replace' && next.state.hl === state.hl;
    },
    destroy() {
      clearTimeout(blurTimer);
    },
  };
}
