// `?` help popovers (spec F.9 item 6, WCAG 1.4.13). Owner: WP3-app.
//
// The trigger is a <button> with aria-expanded and aria-controls. The popover opens on click / tap / Enter / Space,
// stays while hovered or focused, and closes on Esc (while focus is on the button or in the popover), a second click,
// a click outside, when focus moves elsewhere and on every route change (sheet.js closeAllOverlays). It is a sibling
// of the button (reading order) with fixed positioning so overflow containers never clip it; its width is its own
// (max 320 px), and below 640 px it spans the page between the 16 px gutters.

import { h } from '../dom.js';
import { t } from '../i18n.js';
import { icon } from '../ui.js';
import { placeUnder } from './popover.js';
import { registerTransient } from './sheet.js';

let counter = 0;
let openOne = null;

/**
 * A help button and its popover.
 * @param {string} label  what the help is about (used in the button's accessible name: `help.button`)
 * @param {string|Node|Array<string|Node>|(() => (string|Node|Array))} content  popover content (strings → paragraphs)
 * @param {{className?: string}} [opts]
 * @returns {HTMLElement}  a <span class="help"> holding the button and the (hidden) popover
 */
export function helpButton(label, content, opts = {}) {
  const id = `help-${++counter}`;
  const pop = h('div', { class: 'help-pop', id, role: 'note', 'aria-label': t('common.help'), hidden: true });
  const button = h('button', {
    type: 'button', class: 'help-btn', 'aria-expanded': 'false', 'aria-controls': id,
    'aria-label': t('help.button', { label }),
    on: { click: (e) => { e.stopPropagation(); toggle(); } },
  }, icon('ui-help'));
  const wrap = h('span', {
    class: ['help', opts.className],
    on: {
      // Esc belongs to the help only while focus is on its button or inside it (A11Y-9): no document-wide capture
      keydown: (e) => {
        if (e.key !== 'Escape' || pop.hidden) return;
        e.stopPropagation();
        close();
        if (button.isConnected) button.focus();
      },
      focusout: (e) => {
        const next = e.relatedTarget;
        if (next instanceof Node && !wrap.contains(next)) close();
      },
    },
  }, button, pop);

  function fill() {
    const value = typeof content === 'function' ? content() : content;
    pop.replaceChildren(...[].concat(value).filter((v) => v != null)
      .map((v) => (typeof v === 'string' ? h('p', null, v) : v)));
  }
  function onOutside(e) {
    if (!wrap.contains(e.target)) close();
  }
  const reposition = () => {
    if (!button.isConnected) close();
    else placeUnder(pop, button, 6, { gutters: true });
  };
  let unregister = null;
  function show() {
    if (openOne && openOne !== api) openOne.close();
    fill();
    pop.hidden = false;
    button.setAttribute('aria-expanded', 'true');
    reposition();
    document.addEventListener('pointerdown', onOutside, true);
    window.addEventListener('scroll', reposition, true);
    window.addEventListener('resize', reposition);
    unregister = registerTransient(api);
    openOne = api;
  }
  function close() {
    if (pop.hidden) return;
    pop.hidden = true;
    button.setAttribute('aria-expanded', 'false');
    document.removeEventListener('pointerdown', onOutside, true);
    window.removeEventListener('scroll', reposition, true);
    window.removeEventListener('resize', reposition);
    if (unregister) unregister();
    unregister = null;
    if (openOne === api) openOne = null;
  }
  function toggle() {
    if (pop.hidden) show();
    else close();
  }
  const api = { close };
  return wrap;
}
