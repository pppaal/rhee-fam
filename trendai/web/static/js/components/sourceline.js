// Source line (spec F.6.8, F.6.9, M18): `source.line` (detail) or `compare.source_line` (comparison), with an optional
// `?` that opens help.naver_index for Naver data. Owner: WP3-app.

import { h } from '../dom.js';
import { t } from '../i18n.js';
import { helpButton } from './help.js';

/**
 * @param {string} text  the server's source line
 * @param {{naver?: boolean}} [opts]
 * @returns {HTMLElement}
 */
export function sourceLine(text, opts = {}) {
  return h('p', { class: 'source-line' },
    h('span', null, text),
    opts.naver ? helpButton(t('naver_explainer.title'), t('help.naver_index')) : null);
}

/** A caption line (value-kind note, axis note) with the Naver `?` when asked. */
export function noteLine(text, opts = {}) {
  if (!text) return null;
  return h('p', { class: ['note-line', opts.className] },
    h('span', null, text),
    opts.naver ? helpButton(t('naver_explainer.title'), t('help.naver_index')) : null);
}
