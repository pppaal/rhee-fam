// File drop zone with a keyboard-accessible 파일 선택 button and 붙여넣기 (spec F.6.2 upload card, F.6.10 step 1).
// One builder for home and /upload (A11Y-22). Owner: WP3-app.

import { h } from '../dom.js';
import { icon } from '../ui.js';

export const ACCEPT = '.csv,.tsv,.txt,.xlsx,.xls,.json';

/**
 * @param {{text: Node|Node[], chooseLabel: string, pasteLabel: string, onFile: (file: File) => void,
 *          onPaste: (opener: HTMLElement) => void, primary?: boolean, compact?: boolean, inputId?: string,
 *          after?: Node|Node[]}} o
 *        text: the lines above the buttons; after: lines under them; primary: 파일 선택 is the primary button
 * @returns {HTMLElement}
 */
export function dropZone(o) {
  const fileInput = h('input', {
    id: o.inputId || null, type: 'file', class: 'sr-only', accept: ACCEPT, tabindex: '-1', 'aria-hidden': 'true',
    on: { change: () => { if (fileInput.files && fileInput.files[0]) o.onFile(fileInput.files[0]); } },
  });
  const drop = h('div', {
    class: ['dropzone', o.compact && 'dropzone-compact'],
    on: {
      dragover: (e) => { e.preventDefault(); drop.classList.add('is-over'); },
      dragleave: () => drop.classList.remove('is-over'),
      drop: (e) => {
        e.preventDefault();
        drop.classList.remove('is-over');
        const file = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
        if (file) o.onFile(file);
      },
    },
  },
  icon('ui-upload', { cls: 'dropzone-icon' }),
  o.text,
  h('div', { class: 'btn-row' },
    h('button', {
      type: 'button', class: ['btn', o.primary ? 'btn-primary' : 'btn-secondary', 'dropzone-choose'],
      on: { click: () => fileInput.click() },
    }, icon('ui-file'), o.chooseLabel),
    h('button', { type: 'button', class: 'btn btn-secondary', on: { click: (e) => o.onPaste(e.currentTarget) } },
      icon('ui-paste'), o.pasteLabel)),
  o.after || null,
  fileInput);
  return drop;
}
