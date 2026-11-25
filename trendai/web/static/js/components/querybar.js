// Query bar (spec F.6.4, brief §3.2): sticky on results pages. Series chips (swatch + slot icon + name + ×) and
// `+ 비교` (inline input, disabled at 5), then the default chips `출처 ▾ 기간 ▾ 단위 ▾ 예측 기간 ▾` and, for Naver,
// `상세 조건 ▾`, each showing its value. A change updates the URL (pushState) and re-runs the analysis. Samples: 기간 /
// 단위 are fixed (chip.sample_fixed). Upload results (/r, /s): read-only labels plus a `다시 분석` link to /upload.
// ≥ 640 px the chips open anchored popovers, below 640 px bottom sheets; once the page scrolls past the bar it
// collapses to one line (≤ 56 px) that opens the full bar in a bottom sheet. Owner: WP3-app.

import { clear, h } from '../dom.js';
import { rangeLabel, spanLabel } from '../fmt.js';
import { getLang, t } from '../i18n.js';
import { isComposingEvent, shouldCommit } from '../ime.js';
import { MAX_SERIES, MAX_TERM_LENGTH, cleanTerm, effectiveSource, seriesName } from '../state.js';
import { icon, isNarrow, swatch } from '../ui.js';
import { openChipEditor } from './chipeditor.js';
import { openPopover } from './popover.js';
import { openSheet } from './sheet.js';
import { toast } from './toast.js';

const PRESETS = ['1y', '3y', '5y', 'all'];
const UNITS = ['auto', 'date', 'week', 'month'];
const UNIT_FREQ = { date: 'D', week: 'W', month: 'M' };

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

let counter = 0;

/**
 * @param {{ctx: Object, state: Object, mode: 'search'|'upload', env?: Object|null,
 *          onChange: (patch: Object) => void}} opts
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
  const bar = h('div', { class: 'querybar', role: 'region', 'aria-label': t('chip.bar_label') });
  const sentinel = h('div', { class: 'querybar-sentinel', 'aria-hidden': 'true' });
  const wrap = h('div', { class: 'querybar-wrap' }, sentinel, bar);

  const naverInfo = (config.sources && config.sources.naver) || {};
  const naverSelectable = naverInfo.available || naverInfo.reason === 'budget_exhausted';

  // ------------------------------------------------------------------ helpers
  const chipBtn = (label, onOpen, extra = {}) => h('button', {
    type: 'button', class: ['chip', 'chip-select', extra.disabled && 'is-disabled'],
    'aria-haspopup': 'dialog', 'aria-expanded': 'false', 'aria-disabled': extra.disabled ? 'true' : null,
    dataset: { chip: extra.key || null },
    on: { click: (e) => onOpen(e.currentTarget) },
  }, h('span', null, label), icon('ui-chevron-down', { cls: 'chev' }));

  const fixedNote = (anchor, text) => openPopover(anchor, { title: anchor.querySelector('span').textContent, content: h('p', null, text) });

  function radioList(name, options, value, onPick) {
    const id = `${name}-${++counter}`;
    return h('div', { class: 'radio-list', role: 'radiogroup' }, options.map((o) => h('label', { class: ['radio', o.disabled && 'is-disabled'] },
      h('input', {
        type: 'radio', name: id, value: o.value, disabled: o.disabled ? true : null, props: { checked: o.value === value },
        on: { change: (e) => { if (e.target.checked) onPick(o.value); } },
      }),
      h('span', null, o.label, o.hint ? h('span', { class: 'radio-hint' }, o.hint) : null))));
  }

  function choose(anchor, title, build) {
    let pop = null;
    const done = (patch) => {
      if (pop) pop.close();
      if (sheet) { sheet.close(); sheet = null; }
      opts.onChange(patch);
    };
    pop = openPopover(anchor, { title, content: build(done), className: 'chip-popover' });
  }

  // ------------------------------------------------------------------ series chips (search mode)
  function seriesChips() {
    const list = h('ul', { class: 'qb-series', 'aria-label': t('search.label') });
    state.q.forEach((terms, i) => {
      const name = seriesName(terms);
      const editable = src === 'naver';
      const slot = env && env.compare ? env.compare.slots[String(i)] : (state.q.length <= MAX_SERIES ? i + 1 : null);
      const main = editable
        ? h('button', {
          type: 'button', class: 'chip-main', 'aria-label': t('search.edit_chip', { name }),
          on: { click: (e) => editSeries(i, e.currentTarget) },
        }, swatch(slot), h('span', null, name))
        : h('span', { class: 'chip-main' }, swatch(slot), h('span', null, name));
      list.appendChild(h('li', { class: 'chip chip-series' }, main,
        h('button', {
          type: 'button', class: 'chip-x', 'aria-label': t('a11y.remove', { name }),
          on: { click: () => removeSeries(i) },
        }, icon('ui-close'))));
    });
    const full = state.q.length >= MAX_SERIES;
    const add = src === 'sample' ? null : h('li', null, h('button', {
      type: 'button', class: ['chip', 'chip-add', full && 'is-disabled'], 'aria-disabled': full ? 'true' : null,
      title: full ? t('chip.compare_full') : null,
      on: { click: (e) => (full ? toast(t('chip.compare_full')) : openAdd(e.currentTarget)) },
    }, t('chip.compare_add')));
    if (add) list.appendChild(add);
    return list;
  }

  function editSeries(i, opener) {
    openChipEditor({
      terms: state.q[i], opener,
      onSave: (terms) => {
        const q = state.q.map((x, j) => (j === i ? terms : x));
        if (sheet) { sheet.close(); sheet = null; }
        opts.onChange({ q, focus: null });
      },
    });
  }

  function removeSeries(i) {
    const q = state.q.filter((_, j) => j !== i);
    if (sheet) { sheet.close(); sheet = null; }
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
      placeholder: t(`search.placeholder.${src === 'wiki' ? 'wiki' : 'naver'}`),
      on: {
        keydown: (e) => {
          if (isComposingEvent(e)) return;
          if (e.key === 'Escape') { e.preventDefault(); restore(); return; }
          if (shouldCommit(e)) { e.preventDefault(); commit(); }
        },
        blur: () => { if (!input.value.trim()) setTimeout(restore, 120); },
      },
    });
    const form = h('div', { class: 'qb-add-form' }, h('label', { class: 'sr-only', for: id }, t('chip.compare_input')), input);
    li.replaceChildren(form);
    input.focus();
    function restore() {
      if (form.isConnected) li.replaceChildren(button);
    }
    function commit() {
      const terms = [];
      for (const part of input.value.split(/[+,]/)) {
        const term = cleanTerm(part);
        if (term && !terms.includes(term)) terms.push(term);
      }
      if (!terms.length) return;
      if (src === 'wiki' && (input.value.includes('+') || terms.length > 1)) { toast(t('search.wiki_no_group')); return; }
      if (terms.some((x) => x.length > MAX_TERM_LENGTH)) { toast(t('search.too_long')); return; }
      if (sheet) { sheet.close(); sheet = null; }
      opts.onChange({ q: [...state.q, terms], focus: null });
    }
  }

  // ------------------------------------------------------------------ default chips
  function sourceChip() {
    if (src === 'sample') return chipBtn(t('chip.source', { value: t('source.sample') }), (a) => fixedNote(a, t('chip.sample_fixed')), { disabled: true, key: 'source' });
    return chipBtn(t('chip.source', { value: t(`source.${src}`) }), (anchor) => choose(anchor, t('source.label'), (done) => radioList('src', [
      { value: 'naver', label: t('source.naver'), disabled: !naverSelectable, hint: naverSelectable ? null : t('source.unavailable') },
      { value: 'wiki', label: t('source.wiki') },
    ], src, (value) => done({ src: value, q: value === 'wiki' ? state.q.map((terms) => terms.slice(0, 1)) : state.q, focus: null }))), { key: 'source' });
  }

  function rangeChip() {
    const label = t('chip.range', { value: rangeText(state.range, lang) });
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
    const label = t('chip.unit', { value: t(`unit.${state.unit}`) });
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
    return chipBtn(t('chip.h', { value: horizonValue() }), (anchor) => choose(anchor, t('chip.title.h'), (done) => {
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
    const value = filtersValue();
    return chipBtn(value ? t('chip.filters_on', { value }) : t('chip.filters'), (anchor) => choose(anchor, t('chip.filters'), (done) => {
      let dev = state.dev;
      let gender = state.gender;
      const ages = new Set(state.ages);
      const group = (legend, node) => h('fieldset', { class: 'field' }, h('legend', { class: 'field-label' }, legend), node);
      const ageBoxes = h('div', { class: 'check-grid' }, (config.naver_ages || []).map((a) => h('label', { class: 'check' },
        h('input', {
          type: 'checkbox', value: a.code, props: { checked: ages.has(a.code) },
          on: { change: (e) => { if (e.target.checked) ages.add(a.code); else ages.delete(a.code); } },
        }),
        h('span', null, a[lang] || a.ko))));
      return [
        group(t('filter.device'), radioList('dev', [['', 'filter.all'], ['pc', 'filter.pc'], ['mo', 'filter.mo']].map(([v, k]) => ({ value: v, label: t(k) })), dev, (v) => { dev = v; })),
        group(t('filter.gender'), radioList('gender', [['', 'filter.all'], ['m', 'filter.m'], ['f', 'filter.f']].map(([v, k]) => ({ value: v, label: t(k) })), gender, (v) => { gender = v; })),
        group(t('filter.ages'), ageBoxes),
        h('div', { class: 'btn-row' }, h('button', {
          type: 'button', class: 'btn btn-primary btn-small',
          on: { click: () => done({ dev, gender, ages: [...ages].sort((x, y) => Number(x) - Number(y)) }) },
        }, t('common.apply'))),
      ];
    }), { key: 'filters' });
  }

  // ------------------------------------------------------------------ upload / shared (read-only)
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
        first ? label(t('chip.unit', { value: first.source.freq_label })) : null,
        first && env.query && Number.isInteger(env.query.h) ? label(t('chip.h', { value: spanLabel(env.query.h, freq, lang) })) : null,
        h('a', { class: 'chip chip-link', href: '/upload' }, t('chip.reanalyze_upload'))),
    ];
  }

  function content() {
    if (opts.mode === 'upload') return readonlyRows();
    return [
      seriesChips(),
      h('div', { class: 'qb-defaults' }, sourceChip(), rangeChip(), unitChip(), horizonChip(), filtersChip()),
    ];
  }

  // ------------------------------------------------------------------ collapsed one-line bar (< 640 px)
  function summaryText() {
    const names = opts.mode === 'upload'
      ? ((env && env.series) || []).map((vm) => vm.name)
      : state.q.map(seriesName);
    const parts = [names.join(' · ')];
    if (opts.mode !== 'upload' && src !== 'sample') parts.push(t(`range.short.${PRESETS.includes(state.range) ? state.range : 'custom'}`), t(`unit.${state.unit}`));
    return parts.filter(Boolean).join(' · ');
  }

  function render() {
    clear(bar);
    const expand = h('button', {
      type: 'button', class: 'qb-summary', 'aria-haspopup': 'dialog',
      on: {
        click: (e) => {
          sheet = openSheet({ title: t('chip.bar_label'), content: h('div', { class: 'querybar-sheet' }, content()), opener: e.currentTarget, onClose: () => { sheet = null; } });
        },
      },
    }, h('span', { class: 'qb-summary-swatches', 'aria-hidden': 'true' },
      (opts.mode === 'upload' ? ((env && env.series) || []).map((vm) => vm.slot) : state.q.map((_, i) => i + 1)).slice(0, 5).map((s) => swatch(s))),
    h('span', { class: 'qb-summary-text' }, summaryText()),
    h('span', { class: 'sr-only' }, ` · ${t('chip.expand')}`),
    icon('ui-chevron-down', { cls: 'chev' }));
    bar.append(h('div', { class: 'qb-full' }, content()), expand);
  }

  render();
  let observer = null;
  if ('IntersectionObserver' in window) {
    observer = new IntersectionObserver(([entry]) => {
      bar.classList.toggle('is-collapsed', !entry.isIntersecting && isNarrow());
    }, { threshold: 0 });
    requestAnimationFrame(() => { if (sentinel.isConnected) observer.observe(sentinel); });
  }

  return {
    el: wrap,
    setEnvelope(next) {
      env = next;
      render();
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
      if (sheet) sheet.close();
    },
  };
}
