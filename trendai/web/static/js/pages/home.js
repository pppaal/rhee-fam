// Home (spec F.6.2, brief §3.3): hero text, search box with chips, source switch, hint line, example chips per
// source, upload card (drop zone + 파일 선택 + 붙여넣기 → /upload with the file or text handed over in memory),
// samples card (9 chips with the 가상 데이터 badge) and recent analyses. `src` and `q` in the URL pre-fill the source
// switch and the search box and never start an analysis (the use_wiki fix, C.12.6). Owner: WP3-app.

import { h, clear } from '../dom.js';
import { getLang, t } from '../i18n.js';
import { effectiveSource, emptyState, serialize, withState } from '../state.js';
import { icon, pill } from '../ui.js';
import { createSearchBox } from '../components/searchbox.js';
import { renderRecent } from '../components/recent.js';
import { helpButton } from '../components/help.js';
import { loadSamples } from './samples.js';
import { openPasteDialog, setPendingUpload } from './upload.js';

const SOURCES = ['naver', 'wiki', 'upload', 'sample'];

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
  let blurTimer = null;

  const page = h('div', { class: 'page home' });
  const box = createSearchBox({
    source,
    series: source === 'sample' ? state.q.map((terms) => terms.slice(0, 1)) : state.q,
    onSubmit: (series) => {
      const next = emptyState('explore');
      next.src = source;
      next.q = series;
      next.hl = state.hl;
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
      on: { click: () => choose(src), keydown: onRadioKey },
    }, t(`source.${src}`), src === 'naver' && !naver.available
      ? h('span', { class: 'source-badge' }, t('source.unavailable')) : null);
  });
  const switchEl = h('div', { class: 'source-switch-wrap' },
    h('div', { class: 'source-switch', role: 'radiogroup', 'aria-label': t('source.label') }, radios),
    naver.reasonKey ? helpButton(t('source.naver'), t(naver.reasonKey), { className: 'source-help' }) : null);

  function onRadioKey(e) {
    const keys = { ArrowRight: 1, ArrowDown: 1, ArrowLeft: -1, ArrowUp: -1 };
    if (!(e.key in keys)) return;
    e.preventDefault();
    const usable = radios.filter((r) => r.getAttribute('aria-disabled') !== 'true' && r.dataset.source !== 'upload');
    const i = usable.indexOf(e.currentTarget);
    const next = usable[(i + keys[e.key] + usable.length) % usable.length];
    choose(next.dataset.source);
    next.focus();
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
    renderExamples();
    const next = withState(state, { src, q: box.getSeries() });
    ctx.navigate(serialize(next), { replace: true, silent: true });
  }

  // ------------------------------------------------------------------ examples per source
  const examples = h('div', { class: 'examples' });
  function renderExamples() {
    clear(examples);
    if (source !== 'naver' && source !== 'wiki') return;
    const terms = t(`home.example_terms.${source}`).split(',').map((s) => s.trim()).filter(Boolean);
    examples.append(
      h('span', { class: 'examples-label' }, t('home.examples')),
      h('ul', { class: 'chip-row' }, terms.map((term) => {
        const s = emptyState('explore');
        s.src = source;
        s.q = [[term]];
        s.hl = state.hl;
        return h('li', null, h('a', { class: 'chip chip-link', href: serialize(s) }, term));
      })));
  }
  renderExamples();

  // ------------------------------------------------------------------ upload card
  const fileInput = h('input', {
    type: 'file', class: 'sr-only', tabindex: '-1', 'aria-hidden': 'true', accept: '.csv,.tsv,.txt,.xlsx,.xls,.json',
    on: { change: () => { if (fileInput.files && fileInput.files[0]) handOver(fileInput.files[0]); } },
  });
  function handOver(fileOrText) {
    setPendingUpload(fileOrText);
    ctx.navigate('/upload');
  }
  const drop = h('div', {
    class: 'dropzone dropzone-compact',
    on: {
      dragover: (e) => { e.preventDefault(); drop.classList.add('is-over'); },
      dragleave: () => drop.classList.remove('is-over'),
      drop: (e) => {
        e.preventDefault();
        drop.classList.remove('is-over');
        const file = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
        if (file) handOver(file);
      },
    },
  },
  icon('ui-upload', { cls: 'dropzone-icon' }),
  h('p', null, t('home.upload.body')),
  h('div', { class: 'btn-row' },
    h('button', { type: 'button', class: 'btn btn-secondary', on: { click: () => fileInput.click() } },
      icon('ui-file'), t('home.upload.choose')),
    h('button', {
      type: 'button', class: 'btn btn-secondary',
      on: { click: (e) => openPasteDialog({ opener: e.currentTarget, onRead: (text) => handOver(text) }) },
    }, icon('ui-paste'), t('home.upload.paste'))),
  fileInput);
  const uploadCard = h('section', { class: 'card home-card', 'aria-labelledby': 'home-upload-title' },
    h('h2', { class: 'card-title', id: 'home-upload-title' }, t('home.upload.title')),
    drop,
    h('p', { class: 'muted small' }, t('upload.formats')));

  // ------------------------------------------------------------------ samples card
  const sampleChips = h('ul', { class: 'chip-row sample-chips' });
  const samplesCard = h('section', { class: 'card home-card', 'aria-labelledby': 'home-samples-title' },
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
