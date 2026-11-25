// Search box with chips (spec F.6.3). Owner: WP3-app.
//
// A text input plus one chip per committed series. `Enter` or `,` commits the text — never during IME composition
// (ime.js); a comma typed while composing, pasted text and mobile keyboards are handled from the input's value.
// `+` inside the text splits it into terms of one series (chip "선크림 + 자외선차단제"). Backspace on an empty input
// focuses the last chip; Backspace/Delete on a chip removes it; × removes it; clicking a Naver chip opens the chip
// editor. 분석 (or Enter on an empty input with at least one chip) submits. A 6th series is refused with a toast;
// Wikipedia refuses '+' groups; terms are at most 50 characters. For `sample` the input is replaced by the sample
// chips (multi-select, up to 5). Refusals are 'warning' toasts (UX-12); the hint line hides while an error says the
// same thing (UX-27). Only Naver chips have a focusable name (it opens the editor); other chips are reached through
// their × button, which also takes Backspace / Delete (A11Y-21).

import { h, clear } from '../dom.js';
import { t } from '../i18n.js';
import { isComposingEvent, shouldCommit, splitCommitted } from '../ime.js';
import { MAX_SERIES, MAX_TERM_LENGTH, MAX_TERMS, cleanTerm, seriesName } from '../state.js';
import { icon } from '../ui.js';
import { openChipEditor } from './chipeditor.js';
import { toast } from './toast.js';

let counter = 0;

/**
 * @param {{source: 'naver'|'wiki'|'sample', wikiLang?: string, series?: string[][],
 *          samples?: Array<{name: string, title: string}>,
 *          onSubmit: (series: string[][]) => void, onChange?: (series: string[][]) => void,
 *          onFocusChange?: (focused: boolean) => void}} opts
 * @returns {{el: HTMLElement, setSource: (s: string) => void, setWikiLang: (wl: string) => void,
 *            setSamples: (list: Array) => void, getSeries: () => string[][], focus: () => void, submit: () => void}}
 */
export function createSearchBox(opts) {
  const id = `search-${++counter}`;
  let source = opts.source;
  let wikiLang = opts.wikiLang || 'ko';
  let series = (opts.series || []).map((terms) => [...terms]);
  let samples = opts.samples || null;
  let composing = false;

  const error = h('p', { class: 'field-error', id: `${id}-err`, 'aria-live': 'polite' });
  const hint = h('p', { class: 'field-help search-hint', id: `${id}-hint` });
  const chipList = h('ul', { class: 'search-chips', 'aria-label': t('search.label') });
  const input = h('input', {
    id, type: 'text', class: 'search-input', autocomplete: 'off', autocapitalize: 'off', spellcheck: 'false',
    enterkeyhint: 'search', 'aria-describedby': `${id}-hint ${id}-err`,
    on: {
      keydown: onKeydown,
      input: onInput,
      compositionstart: () => { composing = true; },
      compositionend: () => { composing = false; onInput(); },
      focus: () => opts.onFocusChange && opts.onFocusChange(true),
      blur: () => opts.onFocusChange && opts.onFocusChange(false),
    },
  });
  const sampleBox = h('div', { class: 'sample-picker', role: 'group', 'aria-label': t('search.placeholder.sample') });
  const submit = h('button', { type: 'submit', class: 'btn btn-primary search-submit' }, t('search.submit'));
  const field = h('div', {
    class: 'search-field',
    on: { click: (e) => { if (e.target === field || e.target === chipList) input.focus(); } },
  }, icon('ui-search', { cls: 'search-icon' }), chipList, input, sampleBox);
  const form = h('form', {
    class: 'searchbox', role: 'search', novalidate: true,
    on: { submit: (e) => { e.preventDefault(); doSubmit(); } },
  },
  h('label', { class: 'sr-only', for: id }, t('search.label')),
  h('div', { class: 'search-row' }, field, submit),
  error,
  hint);

  function setError(message) {
    error.textContent = message || '';
    hint.hidden = Boolean(message);
    if (message) input.setAttribute('aria-invalid', 'true');
    else input.removeAttribute('aria-invalid');
  }

  function changed() {
    renderChips();
    if (opts.onChange) opts.onChange(series.map((s) => [...s]));
  }

  /** Turn one piece of text into a series and add it. Returns false (and shows why) when refused. */
  function commitText(text) {
    const terms = [];
    for (const part of String(text).split('+')) {
      const term = cleanTerm(part);
      if (term && !terms.includes(term)) terms.push(term);
    }
    if (!terms.length) return true;
    if (source === 'wiki' && terms.length > 1) {
      setError(t('search.wiki_no_group'));
      return false;
    }
    if (terms.length > MAX_TERMS) {
      setError(t('chipeditor.too_many'));
      return false;
    }
    if (terms.some((term) => term.length > MAX_TERM_LENGTH)) {
      setError(t('search.too_long'));
      return false;
    }
    if (series.some((s) => JSON.stringify(s) === JSON.stringify(terms))) {
      setError('');
      return true;
    }
    if (series.length >= MAX_SERIES) {
      toast(t('search.too_many'), { kind: 'warning' });
      return false;
    }
    series.push(terms);
    setError('');
    return true;
  }

  function commitInput() {
    const text = input.value;
    if (!text.trim()) return true;
    const ok = commitText(text);
    if (ok) input.value = '';
    changed();
    return ok;
  }

  function onKeydown(e) {
    if (isComposingEvent(e) || composing) return;
    if (shouldCommit(e)) {
      e.preventDefault();
      if (e.key === 'Enter' && !input.value.trim()) {
        if (series.length) doSubmit();
        return;
      }
      commitInput();
      return;
    }
    if (e.key === 'Backspace' && !input.value && series.length) {
      e.preventDefault();
      const last = chipList.querySelector('li:last-child button.chip-main, li:last-child .chip-x');
      if (last) last.focus();
    }
  }

  function onInput() {
    if (composing) return;
    if (input.value.includes(',')) {
      const { commit, rest } = splitCommitted(input.value);
      let ok = true;
      for (const piece of commit) ok = commitText(piece) && ok;
      input.value = ok ? rest : input.value.replace(/,/g, ' ').trim();
      changed();
    } else if (error.textContent && input.value.length <= MAX_TERM_LENGTH) {
      setError('');
    }
  }

  function removeAt(i, refocus = true) {
    series.splice(i, 1);
    changed();
    if (refocus) {
      const chips = chipList.querySelectorAll('li');
      const li = chips.length ? chips[Math.min(i, chips.length - 1)] : null;
      const target = li ? li.querySelector('button.chip-main, .chip-x') : null;
      if (target) target.focus();
      else input.focus();
    }
  }

  function renderChips() {
    clear(chipList);
    series.forEach((terms, i) => {
      const name = seriesName(terms);
      const editable = source === 'naver';
      const onKey = (e) => {
        if (e.key === 'Backspace' || e.key === 'Delete') { e.preventDefault(); removeAt(i); }
      };
      const main = editable
        ? h('button', {
          type: 'button', class: 'chip-main', 'aria-label': t('search.edit_chip', { name }),
          on: { click: (e) => editAt(i, e.currentTarget), keydown: onKey },
        }, name)
        : h('span', { class: 'chip-main' }, name);
      const remove = h('button', {
        type: 'button', class: 'chip-x', 'aria-label': t('a11y.remove', { name }),
        on: { click: () => removeAt(i), keydown: onKey },
      }, icon('ui-close'));
      chipList.appendChild(h('li', { class: 'chip chip-series' }, main, remove));
    });
    form.classList.toggle('has-chips', series.length > 0);
  }

  function editAt(i, opener) {
    openChipEditor({
      terms: series[i], opener,
      onSave: (terms) => {
        if (series.some((s, j) => j !== i && JSON.stringify(s) === JSON.stringify(terms))) series.splice(i, 1);
        else series[i] = terms;
        changed();
      },
    });
  }

  function renderSamples() {
    clear(sampleBox);
    if (!samples) return;
    for (const s of samples) {
      const on = series.some((terms) => terms[0] === s.name);
      sampleBox.appendChild(h('button', {
        type: 'button', class: 'chip chip-toggle', 'aria-pressed': String(on), dataset: { sample: s.name },
        on: {
          click: () => {
            const i = series.findIndex((terms) => terms[0] === s.name);
            if (i >= 0) series.splice(i, 1);
            else if (series.length >= MAX_SERIES) {
              toast(t('samples.max'), { kind: 'warning' });
              return;
            } else series.push([s.name]);
            if (opts.onChange) opts.onChange(series.map((x) => [...x]));
            renderSamples();
          },
        },
      }, s.title));
    }
  }

  function placeholder() {
    if (source !== 'wiki') return t('search.placeholder.naver');
    return t(wikiLang === 'en' ? 'search.placeholder.wiki_en' : 'search.placeholder.wiki');
  }

  function applySource() {
    form.dataset.mode = source;
    const sample = source === 'sample';
    input.hidden = sample;
    chipList.hidden = sample;
    sampleBox.hidden = !sample;
    input.placeholder = placeholder();
    hint.textContent = sample ? t('search.placeholder.sample') : t(source === 'wiki' ? 'search.hint.wiki' : 'search.hint');
    setError('');
    renderChips();
    renderSamples();
  }

  function doSubmit() {
    if (source !== 'sample' && !commitInput()) return;
    if (!series.length) {
      setError(t('empty.no_query'));
      if (source !== 'sample') input.focus();
      return;
    }
    opts.onSubmit(series.map((s) => [...s]));
  }

  applySource();
  return {
    el: form,
    setSource(next) {
      if (next === source) return;
      const wasSample = source === 'sample';
      source = next;
      if (wasSample || next === 'sample') series = [];
      if (next === 'wiki') series = series.map((terms) => terms.slice(0, 1));
      applySource();
      if (opts.onChange) opts.onChange(series.map((s) => [...s]));
    },
    setWikiLang(wl) {
      wikiLang = wl;
      input.placeholder = placeholder();
    },
    setSamples(list) {
      samples = list;
      renderSamples();
    },
    getSeries: () => series.map((s) => [...s]),
    focus: () => (source === 'sample' ? (sampleBox.querySelector('button') || submit).focus() : input.focus()),
    submit: doSubmit,
    showHint(text) {
      hint.textContent = text;
      hint.classList.add('is-emphasis');
    },
  };
}
