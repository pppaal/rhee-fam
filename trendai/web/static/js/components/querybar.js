// Query bar (spec F.6.4, brief §3.2): sticky on results pages. Series chips (swatch + slot icon + name + ×) and
// `+ 비교` (inline input, disabled at 5), then the default chips `출처 ▾ 기간 ▾ 단위 ▾ 예측 기간 ▾` and, for Naver,
// `상세 조건 ▾`, each showing its value. A change updates the URL (pushState) and re-runs the analysis. Samples: 기간 /
// 단위 are fixed (chip.sample_fixed) and show the sample's real period once the result is known (UX-8). Upload
// results (/r, /s): read-only labels plus `다시 분석` (back to the confirm step of the same file while it is still
// held, FU-9; `내 파일로 분석해 보기` on a shared link). ≥ 640 px the chips open anchored popovers, below 640 px bottom
// sheets; once the page scrolls past the bar it collapses to one line (≤ 56 px) that opens the full bar in a bottom
// sheet — below 640 px wide and also on short viewports (≤ 560 px tall, e.g. 200 % zoom, A11Y-5). Its height is
// published as --qb-h so focused elements scroll clear of it. Owner: WP3-app.
//
// The source chip also picks the Wikipedia language (한국어 / English, `wl`, FU-1). When the result arrives, chip
// labels and swatches are updated in place (A11Y-2): no chip is replaced, so focus and open popovers stay put.

import { clear, h } from '../dom.js';
import { rangeLabel, spanLabel } from '../fmt.js';
import { getLang, t } from '../i18n.js';
import { isComposingEvent, shouldCommit } from '../ime.js';
import { MAX_SERIES, MAX_TERM_LENGTH, cleanTerm, effectiveSource, seriesName } from '../state.js';
import { icon, isNarrow, radioList, swatch } from '../ui.js';
import { loadSamples } from '../pages/samples.js';
import { canReconfirm, reconfirmUpload } from '../pages/upload.js';
import { openChipEditor } from './chipeditor.js';
import { openPopover } from './popover.js';
import { openSheet } from './sheet.js';
import { toast } from './toast.js';

const PRESETS = ['1y', '3y', '5y', 'all'];
const UNITS = ['auto', 'date', 'week', 'month'];
const UNIT_FREQ = { date: 'D', week: 'W', month: 'M' };
const WIKI_LANGS = ['ko', 'en'];
const SHORT_QUERY = '(max-height: 560px)';

/** The data frequency a search state will produce (for horizon choices and span words). */
export function freqOf(state, src, env) {
  if (env && env.series && env.series[0]) return env.series[0].chart.freq;
  if (src === 'sample') return 'M';
  if (UNIT_FREQ[state.unit]) return UNIT_FREQ[state.unit];
  return src === 'naver' ? 'W' : 'M';
}

function rangeText(range, lang) {
  if (PRESETS.includes(range)) return t(`range.${range}`);
  const [a, b] = String(range).split('~');
  return a && b ? rangeLabel(a, b, 'D', lang) : range;
}

/** The name of a Wikipedia language code ('ko' → 한국어; other codes as they are). */
export function wikiLangName(wl) {
  return WIKI_LANGS.includes(wl) ? t(`wiki.lang.${wl}`) : String(wl || 'ko');
}

/** True when the bar should collapse once scrolled past: narrow or short viewports (A11Y-5). */
function collapsible() {
  return isNarrow() || window.matchMedia(SHORT_QUERY).matches;
}

let counter = 0;

/**
 * @param {{ctx: Object, state: Object, mode: 'search'|'upload', kind?: 'explore'|'result'|'shared',
 *          env?: Object|null, onChange: (patch: Object) => void}} opts
 * @returns {{el: HTMLElement, setEnvelope: (env: Object) => void, openChipEditor: () => void,
 *            focusChips: () => void, destroy: () => void}}
 */
export function createQueryBar(opts) {
  const { ctx, state } = opts;
  const config = ctx.config;
  const lang = getLang();
  const src = effectiveSource(state, config.default_source);
  let env = opts.env || null;
  let sheet = null;
  let sampleTitles = null;
  const bar = h('div', { class: 'querybar', role: 'region', 'aria-label': t('chip.bar_label') });
  const sentinel = h('div', { class: 'querybar-sentinel', 'aria-hidden': 'true' });
  const wrap = h('div', { class: 'querybar-wrap' }, sentinel, bar);
  // live label nodes, updated in place when the result arrives (entries of the bar and of an open sheet)
  let labelNodes = [];         // {key, span}
  let seriesNodes = [];        // {index, swatchBox, nameSpan}
  let summaryNodes = [];       // {swatches, text}

  const naverInfo = (config.sources && config.sources.naver) || {};
  const naverSelectable = naverInfo.available || naverInfo.reason === 'budget_exhausted';

  // ------------------------------------------------------------------ helpers
  const chipBtn = (labelFn, onOpen, extra = {}) => {
    const span = h('span', null, labelFn());
    labelNodes.push({ key: extra.key, span, labelFn });
    return h('button', {
      type: 'button', class: ['chip', 'chip-select', extra.disabled && 'is-disabled'],
      'aria-haspopup': 'dialog', 'aria-expanded': 'false', 'aria-disabled': extra.disabled ? 'true' : null,
      dataset: { chip: extra.key || null },
      on: { click: (e) => onOpen(e.currentTarget) },
    }, span, icon('ui-chevron-down', { cls: 'chev' }));
  };

  /** Where a chip popover returns focus: its chip while it is shown, else the bar's chip with the same key. */
  const chipNow = (anchor) => () => {
    if (anchor.isConnected && anchor.getClientRects().length) return anchor;
    return bar.querySelector(`.qb-full [data-chip="${anchor.dataset.chip}"]`) || bar.querySelector('.qb-summary');
  };

  const fixedNote = (anchor, text) => openPopover(anchor, {
    title: anchor.querySelector('span').textContent, content: h('p', null, text), opener: chipNow(anchor),
  });

  /** Radios drawn as a segmented row (상세 조건 기기·성별, UX-18): still native radios in a labelled fieldset. */
  function segmentRadios(name, legend, options, value, onPick) {
    const id = `${name}-${++counter}`;
    return h('fieldset', { class: 'field' }, h('legend', { class: 'field-label' }, legend),
      h('div', { class: 'radio-seg' }, options.map((o) => h('label', { class: 'radio-seg-item' },
        h('input', {
          type: 'radio', name: id, value: o.value, props: { checked: o.value === value },
          on: { change: (e) => { if (e.target.checked) onPick(o.value); } },
        }),
        h('span', null, o.label)))));
  }

  function choose(anchor, title, build, extra = {}) {
    let pop = null;
    const done = (patch) => {
      if (pop) pop.close(false);
      if (sheet) { sheet.close(false); sheet = null; }
      opts.onChange(patch);
    };
    const parts = build(done);
    pop = openPopover(anchor, {
      title, content: parts.content || parts, footer: parts.footer || null,
      className: ['chip-popover', extra.className].filter(Boolean).join(' '),
      opener: chipNow(anchor),
    });
  }

  // ------------------------------------------------------------------ series chips (search mode)
  function chipName(i, terms) {
    if (src === 'sample') {
      const vm = env && env.series ? env.series.find((x) => x.index === i) : null;
      return (vm && vm.name) || (sampleTitles && sampleTitles.get(terms[0])) || seriesName(terms);
    }
    return seriesName(terms);
  }
  function slotOf(i) {
    if (env && env.compare) return env.compare.slots[String(i)];
    return state.q.length <= MAX_SERIES ? i + 1 : null;
  }

  function seriesChips() {
    const list = h('ul', { class: 'qb-series', 'aria-label': t('search.label') });
    state.q.forEach((terms, i) => {
      const name = chipName(i, terms);
      const editable = src === 'naver';
      const swatchBox = h('span', { class: 'swatch-box' }, swatch(slotOf(i)));
      const nameSpan = h('span', null, name);
      seriesNodes.push({ index: i, swatchBox, nameSpan, terms });
      const main = editable
        ? h('button', {
          type: 'button', class: 'chip-main', 'aria-label': t('search.edit_chip', { name }), dataset: { seriesIndex: i },
          on: { click: (e) => editSeries(i, e.currentTarget) },
        }, swatchBox, nameSpan)
        : h('span', { class: 'chip-main' }, swatchBox, nameSpan);
      list.appendChild(h('li', { class: 'chip chip-series' }, main,
        h('button', {
          type: 'button', class: 'chip-x', 'aria-label': t('a11y.remove', { name }), dataset: { seriesIndex: i },
          on: { click: () => removeSeries(i) },
        }, icon('ui-close'))));
    });
    const full = state.q.length >= MAX_SERIES;
    const add = src === 'sample' ? null : h('li', null, h('button', {
      type: 'button', class: ['chip', 'chip-add', full && 'is-disabled'], 'aria-disabled': full ? 'true' : null,
      title: full ? t('chip.compare_full') : null,
      on: { click: (e) => (full ? toast(t('chip.compare_full'), { kind: 'warning' }) : openAdd(e.currentTarget)) },
    }, t('chip.compare_add')));
    if (add) list.appendChild(add);
    return list;
  }

  function editSeries(i, opener) {
    openChipEditor({
      terms: state.q[i],
      opener: () => bar.querySelector(`.qb-full .chip-main[data-series-index="${i}"]`) || opener,
      onSave: (terms) => {
        const q = state.q.map((x, j) => (j === i ? terms : x));
        if (sheet) { sheet.close(false); sheet = null; }
        opts.onChange({ q, focus: null });
      },
    });
  }

  function removeSeries(i) {
    const q = state.q.filter((_, j) => j !== i);
    if (sheet) { sheet.close(false); sheet = null; }
    if (!q.length) {
      ctx.navigate('/');
      return;
    }
    opts.onChange({ q, focus: null });
  }

  function openAdd(button) {
    const li = button.parentElement;
    const id = `qb-add-${++counter}`;
    const input = h('input', {
      id, type: 'text', class: 'input input-chip', autocomplete: 'off', enterkeyhint: 'search',
      placeholder: t(src === 'wiki' ? (state.wl === 'en' ? 'search.placeholder.wiki_en' : 'search.placeholder.wiki') : 'search.placeholder.naver'),
      on: {
        keydown: (e) => {
          if (isComposingEvent(e)) return;
          if (e.key === 'Escape') { e.preventDefault(); e.stopPropagation(); restore(true); return; }
          if (shouldCommit(e)) { e.preventDefault(); commit(); }
        },
        blur: () => { if (!input.value.trim()) setTimeout(() => restore(false), 120); },
      },
    });
    const form = h('div', { class: 'qb-add-form' }, h('label', { class: 'sr-only', for: id }, t('chip.compare_input')), input);
    li.replaceChildren(form);
    input.focus();
    /** Put the + 비교 button back; Esc returns focus to it (A11Y-4), a blur leaves focus where it went. */
    function restore(refocus) {
      if (!form.isConnected) return;
      li.replaceChildren(button);
      if (refocus) button.focus();
    }
    function commit() {
      const terms = [];
      for (const part of input.value.split(/[+,]/)) {
        const term = cleanTerm(part);
        if (term && !terms.includes(term)) terms.push(term);
      }
      if (!terms.length) return;
      if (src === 'wiki' && (input.value.includes('+') || terms.length > 1)) { toast(t('search.wiki_no_group'), { kind: 'warning' }); return; }
      if (terms.some((x) => x.length > MAX_TERM_LENGTH)) { toast(t('search.too_long'), { kind: 'warning' }); return; }
      if (sheet) { sheet.close(false); sheet = null; }
      opts.onChange({ q: [...state.q, terms], focus: null });
    }
  }

  // ------------------------------------------------------------------ default chips
  const sourceValue = () => {
    if (src === 'sample') return t('source.sample');
    if (src === 'wiki') return `${t('source.wiki')} · ${wikiLangName(state.wl)}`;
    return t(`source.${src}`);
  };

  function sourceChip() {
    const label = () => t('chip.source', { value: sourceValue() });
    if (src === 'sample') return chipBtn(label, (a) => fixedNote(a, t('chip.sample_fixed')), { disabled: true, key: 'source' });
    const wikiLangs = WIKI_LANGS.includes(state.wl) ? WIKI_LANGS : [...WIKI_LANGS, state.wl];
    const current = src === 'wiki' ? `wiki:${state.wl}` : src;
    return chipBtn(label, (anchor) => choose(anchor, t('source.label'), (done) => radioList('src', [
      { value: 'naver', label: t('source.naver'), disabled: !naverSelectable, hint: naverSelectable ? null : t('source.unavailable') },
      ...wikiLangs.map((wl) => ({ value: `wiki:${wl}`, label: t('source.wiki_lang', { lang: wikiLangName(wl) }) })),
    ], current, (value) => {
      if (value === 'naver') done({ src: 'naver', focus: null });
      else {
        const wl = value.slice(5);
        done({ src: 'wiki', wl, q: src === 'wiki' ? state.q : state.q.map((terms) => terms.slice(0, 1)), focus: null });
      }
    })), { key: 'source' });
  }

  /** A sample's period and unit come from its data, once the result is known (UX-8). */
  const sampleSource = () => (env && env.series && env.series[0] ? env.series[0].source : null);

  function rangeChip() {
    const label = () => {
      if (src === 'sample') {
        const s = sampleSource();
        return t('chip.range', { value: s ? s.range_label : t('chip.fixed') });
      }
      return t('chip.range', { value: rangeText(state.range, lang) });
    };
    if (src === 'sample') return chipBtn(label, (a) => fixedNote(a, t('chip.sample_fixed')), { disabled: true, key: 'range' });
    return chipBtn(label, (anchor) => choose(anchor, t('chip.title.range'), (done) => {
      const periods = (config.periods && config.periods[src]) || {};
      const min = periods.earliest || '2016-01-01';
      const max = (periods.last_complete && periods.last_complete.date) || '';
      const custom = !PRESETS.includes(state.range);
      const [a, b] = custom ? state.range.split('~') : ['', ''];
      const startId = `range-start-${++counter}`;
      const endId = `range-end-${counter}`;
      const start = h('input', { id: startId, type: 'date', class: 'input', min, max, props: { value: a || '' } });
      const end = h('input', { id: endId, type: 'date', class: 'input', min, max, props: { value: b || max } });
      const customBox = h('div', { class: 'range-custom', hidden: !custom },
        h('div', { class: 'range-inputs' },
          h('label', { class: 'field' }, h('span', { class: 'field-label' }, t('range.start')), start),
          h('label', { class: 'field' }, h('span', { class: 'field-label' }, t('range.end')), end)),
        h('button', {
          type: 'button', class: 'btn btn-primary btn-small',
          on: {
            click: () => {
              if (!start.value || !end.value || start.value > end.value) { start.focus(); return; }
              done({ range: `${start.value}~${end.value}` });
            },
          },
        }, t('common.apply')));
      const list = radioList('range', [...PRESETS, 'custom'].map((r) => ({ value: r, label: t(`range.${r}`) })),
        custom ? 'custom' : state.range, (value) => {
          if (value === 'custom') {
            customBox.hidden = false;
            start.focus();
          } else done({ range: value });
        });
      return [list, customBox];
    }), { key: 'range' });
  }

  function unitChip() {
    const label = () => {
      if (src === 'sample') {
        const s = sampleSource();
        return t('chip.unit', { value: s ? t(`freq.${s.freq}`) : t('chip.fixed') });
      }
      return t('chip.unit', { value: t(`unit.${state.unit}`) });
    };
    if (src === 'sample') return chipBtn(label, (a) => fixedNote(a, t('chip.sample_fixed')), { disabled: true, key: 'unit' });
    return chipBtn(label, (anchor) => choose(anchor, t('chip.title.unit'), (done) => radioList('unit',
      UNITS.map((u) => ({ value: u, label: t(`unit.${u}`) })), state.unit, (value) => done({ unit: value, h: 'auto' }))), { key: 'unit' });
  }

  function horizonValue() {
    const freq = freqOf(state, src, env);
    if (state.h === 'auto') return env && env.query && Number.isInteger(env.query.h) ? t('h.auto', { value: spanLabel(env.query.h, freq, lang) }) : t('unit.auto');
    return spanLabel(state.h, freq, lang);
  }

  function horizonChip() {
    return chipBtn(() => t('chip.h', { value: horizonValue() }), (anchor) => choose(anchor, t('chip.title.h'), (done) => {
      const freq = freqOf(state, src, env);
      const choices = (config.limits && config.limits.horizon_choices && config.limits.horizon_choices[freq]) || [];
      const def = config.limits && config.limits.horizon_default ? config.limits.horizon_default[freq] : null;
      return radioList('h', [
        { value: 'auto', label: def ? t('h.auto', { value: spanLabel(def, freq, lang) }) : t('unit.auto') },
        ...choices.map((n) => ({ value: n, label: spanLabel(n, freq, lang) })),
      ], state.h, (value) => done({ h: value }));
    }), { key: 'h' });
  }

  function filtersValue() {
    if (env && env.series && env.series[0] && env.series[0].source.filters_label) return env.series[0].source.filters_label;
    const parts = [];
    if (state.dev) parts.push(t(`filter.${state.dev}`));
    if (state.gender) parts.push(t(`filter.${state.gender}`));
    const ages = (config.naver_ages || []).filter((a) => state.ages.includes(a.code)).map((a) => a[lang] || a.ko);
    if (ages.length) parts.push(ages.join(', '));
    return parts.join(' · ');
  }

  function filtersChip() {
    if (src !== 'naver') return null;
    const label = () => {
      const value = filtersValue();
      return value ? t('chip.filters_on', { value }) : t('chip.filters');
    };
    return chipBtn(label, (anchor) => choose(anchor, t('chip.filters'), (done) => {
      let dev = state.dev;
      let gender = state.gender;
      const ages = new Set(state.ages);
      const boxes = [];
      const ageBoxes = h('fieldset', { class: 'field' }, h('legend', { class: 'field-label' }, t('filter.ages')),
        h('div', { class: 'check-grid check-grid-3' }, (config.naver_ages || []).map((a) => {
          const box = h('input', {
            type: 'checkbox', value: a.code, props: { checked: ages.has(a.code) },
            on: { change: (e) => { if (e.target.checked) ages.add(a.code); else ages.delete(a.code); } },
          });
          boxes.push(box);
          return h('label', { class: 'check' }, box, h('span', null, a[lang] || a.ko));
        })));
      const devGroup = segmentRadios('dev', t('filter.device'), [['', 'filter.all'], ['pc', 'filter.pc'], ['mo', 'filter.mo']].map(([v, k]) => ({ value: v, label: t(k) })), dev, (v) => { dev = v; });
      const genderGroup = segmentRadios('gender', t('filter.gender'), [['', 'filter.all'], ['m', 'filter.m'], ['f', 'filter.f']].map(([v, k]) => ({ value: v, label: t(k) })), gender, (v) => { gender = v; });
      const reset = h('button', {
        type: 'button', class: 'btn btn-secondary',
        on: {
          click: () => {
            dev = '';
            gender = '';
            ages.clear();
            for (const input of [...devGroup.querySelectorAll('input'), ...genderGroup.querySelectorAll('input')]) input.checked = input.value === '';
            for (const box of boxes) box.checked = false;
          },
        },
      }, t('common.reset'));
      const apply = h('button', {
        type: 'button', class: 'btn btn-primary',
        on: { click: () => done({ dev, gender, ages: [...ages].sort((x, y) => Number(x) - Number(y)) }) },
      }, t('common.apply'));
      // the footer sits outside the scrolling body, so 적용 is always visible (UX-18)
      return { content: [devGroup, genderGroup, ageBoxes], footer: [reset, apply] };
    }, { className: 'filters-popover' }), { key: 'filters' });
  }

  // ------------------------------------------------------------------ upload / shared (read-only)
  function reanalyzeControl() {
    if (!env) return null;                                  // nothing to re-run before the result is known (UX-10)
    if (opts.kind === 'shared') {
      // a share recipient has no file: invite them to analyse their own (UX-10, FU-9)
      return h('a', { class: 'chip chip-link', href: '/upload' }, t('chip.analyze_own'));
    }
    const id = state.id;
    if (id && canReconfirm(id)) {
      return h('button', {
        type: 'button', class: 'chip chip-link',
        on: { click: () => { reconfirmUpload(id); ctx.navigate('/upload'); } },
      }, t('chip.reanalyze_upload'));
    }
    return h('a', { class: 'chip chip-link', href: '/upload' }, t('chip.reanalyze_upload'));
  }

  function readonlyRows() {
    const vms = (env && env.series) || [];
    const first = vms[0];
    const names = h('ul', { class: 'qb-series' }, vms.map((vm) => h('li', { class: 'chip chip-series chip-static' },
      h('span', { class: 'chip-main' }, swatch(env.compare ? env.compare.slots[String(vm.index)] : vm.slot), h('span', null, vm.name)))));
    const label = (text) => h('span', { class: 'chip chip-static', title: t('chip.readonly') }, text);
    const freq = first ? first.chart.freq : null;
    return [
      names,
      h('div', { class: 'qb-defaults' },
        first ? label(t('chip.range', { value: first.source.range_label })) : null,
        // the unit word alone: '단위: 월간', not '단위: 월 단위' (UX-26)
        first ? label(t('chip.unit', { value: t(`freq.${first.source.freq || freq}`) })) : null,
        first && env.query && Number.isInteger(env.query.h) ? label(t('chip.h', { value: spanLabel(env.query.h, freq, lang) })) : null,
        reanalyzeControl()),
    ];
  }

  function content() {
    if (opts.mode === 'upload') return readonlyRows();
    return [
      seriesChips(),
      h('div', { class: 'qb-defaults' }, sourceChip(), rangeChip(), unitChip(), horizonChip(), filtersChip()),
    ];
  }

  // ------------------------------------------------------------------ collapsed one-line bar
  function summaryText() {
    const names = opts.mode === 'upload'
      ? ((env && env.series) || []).map((vm) => vm.name)
      : state.q.map((terms, i) => chipName(i, terms));
    const parts = [names.join(' · ')];
    if (opts.mode !== 'upload' && src !== 'sample') parts.push(t(`range.short.${PRESETS.includes(state.range) ? state.range : 'custom'}`), t(`unit.${state.unit}`));
    return parts.filter(Boolean).join(' · ');
  }

  function summarySwatches() {
    return (opts.mode === 'upload' ? ((env && env.series) || []).map((vm) => vm.slot) : state.q.map((_, i) => slotOf(i)))
      .slice(0, 5).map((s) => swatch(s));
  }

  function render() {
    labelNodes = [];
    seriesNodes = [];
    summaryNodes = [];
    clear(bar);
    const swatches = h('span', { class: 'qb-summary-swatches', 'aria-hidden': 'true' }, summarySwatches());
    const text = h('span', { class: 'qb-summary-text' }, summaryText());
    summaryNodes.push({ swatches, text });
    const expand = h('button', {
      type: 'button', class: 'qb-summary', 'aria-haspopup': 'dialog',
      on: {
        click: () => {
          sheet = openSheet({
            title: t('chip.bar_label'), content: h('div', { class: 'querybar-sheet' }, content()),
            opener: () => bar.querySelector('.qb-summary'), onClose: () => { sheet = null; },
          });
        },
      },
    }, swatches, text, h('span', { class: 'sr-only' }, ` · ${t('chip.expand')}`), icon('ui-chevron-down', { cls: 'chev' }));
    bar.append(h('div', { class: 'qb-full' }, content()), expand);
  }

  /** Bring every label, swatch and name up to date without replacing a node (A11Y-2). */
  function refresh() {
    labelNodes = labelNodes.filter((n) => n.span.isConnected);
    for (const n of labelNodes) {
      const text = n.labelFn();
      if (n.span.textContent !== text) n.span.textContent = text;
    }
    seriesNodes = seriesNodes.filter((n) => n.nameSpan.isConnected);
    for (const n of seriesNodes) {
      const name = chipName(n.index, n.terms);
      if (n.nameSpan.textContent !== name) {
        n.nameSpan.textContent = name;
        const main = n.nameSpan.closest('.chip-main');
        if (main && main.tagName === 'BUTTON') main.setAttribute('aria-label', t('search.edit_chip', { name }));
        const x = n.nameSpan.closest('.chip-series') && n.nameSpan.closest('.chip-series').querySelector('.chip-x');
        if (x) x.setAttribute('aria-label', t('a11y.remove', { name }));
      }
      n.swatchBox.replaceChildren(swatch(slotOf(n.index)));
    }
    for (const n of summaryNodes) {
      n.swatches.replaceChildren(...summarySwatches());
      n.text.textContent = summaryText();
    }
  }

  render();
  if (src === 'sample' && opts.mode !== 'upload') {
    loadSamples(lang).then((list) => {
      sampleTitles = new Map(list.map((s) => [s.name, s.title]));
      refresh();
    }).catch(() => {});
  }

  let observer = null;
  let visible = true;
  const applyCollapse = () => bar.classList.toggle('is-collapsed', !visible && collapsible());
  if ('IntersectionObserver' in window) {
    observer = new IntersectionObserver(([entry]) => {
      visible = entry.isIntersecting;
      applyCollapse();
    }, { threshold: 0 });
    requestAnimationFrame(() => { if (sentinel.isConnected) observer.observe(sentinel); });
  }
  window.addEventListener('resize', applyCollapse);
  // publish the bar's height for scroll-padding-top (base.css), so focus never lands under it (A11Y-5)
  let sizer = null;
  const root = document.documentElement;
  const publish = () => {
    if (!bar.isConnected || wrap.hidden) root.style.removeProperty('--qb-h');
    else root.style.setProperty('--qb-h', `${Math.ceil(bar.getBoundingClientRect().height)}px`);
  };
  if ('ResizeObserver' in window) {
    sizer = new ResizeObserver(publish);
    requestAnimationFrame(() => { if (bar.isConnected) sizer.observe(bar); publish(); });
  }

  return {
    el: wrap,
    setEnvelope(next) {
      const before = opts.mode === 'upload' ? Boolean(env) : null;
      env = next;
      // read-only rows exist only once the result is known; everything else is updated in place
      if (opts.mode === 'upload' && before !== Boolean(env)) {
        const active = document.activeElement;
        const hadFocus = active && bar.contains(active);
        render();
        if (hadFocus) {
          const target = bar.querySelector('.qb-full a, .qb-full button');
          if (target) target.focus({ preventScroll: true });
        }
      } else {
        refresh();
      }
      publish();
    },
    openChipEditor() {
      if (src === 'naver' && state.q.length) {
        const btn = bar.querySelector('.qb-full .qb-series .chip-main');
        editSeries(0, btn);
      }
    },
    focusChips() {
      const first = bar.querySelector('.qb-full .qb-series .chip-x');
      if (first) first.focus();
    },
    destroy() {
      if (observer) observer.disconnect();
      if (sizer) sizer.disconnect();
      window.removeEventListener('resize', applyCollapse);
      root.style.removeProperty('--qb-h');
      if (sheet) sheet.close(false);
    },
  };
}
