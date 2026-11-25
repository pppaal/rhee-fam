// Chip editor (spec F.6.3): edits the terms of one Naver series in a dialog — one input "같은 뜻 검색어 (쉼표로 구분)",
// at most 5 terms, Save / Cancel. DataLab's "comma = group" rule lives only here. Owner: WP3-app.

import { h } from '../dom.js';
import { t } from '../i18n.js';
import { MAX_TERM_LENGTH, MAX_TERMS, cleanTerm } from '../state.js';
import { openSheet } from './sheet.js';

let counter = 0;

/** Split the editor text into unique, cleaned terms. */
export function parseTermList(text) {
  const out = [];
  for (const piece of String(text || '').split(',')) {
    const term = cleanTerm(piece);
    if (term && !out.includes(term)) out.push(term);
  }
  return out;
}

/**
 * Open the chip editor.
 * @param {{terms: string[], opener?: HTMLElement, onSave: (terms: string[]) => void}} opts
 */
export function openChipEditor(opts) {
  const id = `chipedit-${++counter}`;
  const input = h('input', {
    id, type: 'text', class: 'input', autocomplete: 'off', 'aria-describedby': `${id}-help ${id}-err`,
    props: { value: (opts.terms || []).join(', ') },
    on: { keydown: (e) => { if (e.key === 'Enter' && !e.isComposing && e.keyCode !== 229) { e.preventDefault(); save(); } } },
  });
  const error = h('p', { class: 'field-error', id: `${id}-err`, 'aria-live': 'polite' });
  const content = [
    h('label', { class: 'field-label', for: id }, t('chipeditor.label')),
    input,
    h('p', { class: 'field-help', id: `${id}-help` }, t('chipeditor.help')),
    error,
  ];
  const cancel = h('button', { type: 'button', class: 'btn', on: { click: () => sheet.close() } }, t('common.cancel'));
  const ok = h('button', { type: 'button', class: 'btn btn-primary', on: { click: () => save() } }, t('common.save'));
  const sheet = openSheet({
    title: t('chipeditor.title'), content, footer: [cancel, ok], opener: opts.opener, className: 'sheet-small',
    initialFocus: input,
  });

  function save() {
    const terms = parseTermList(input.value);
    let message = '';
    if (!terms.length) message = t('empty.no_query');
    else if (terms.length > MAX_TERMS) message = t('chipeditor.too_many');
    else if (terms.some((term) => term.length > MAX_TERM_LENGTH)) message = t('search.too_long');
    if (message) {
      error.textContent = message;
      input.setAttribute('aria-invalid', 'true');
      input.focus();
      return;
    }
    sheet.close();
    opts.onSave(terms);
  }
  return sheet;
}
