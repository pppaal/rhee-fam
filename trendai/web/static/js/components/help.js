// `?` help popovers (spec F.9 item 6, WCAG 1.4.13). Owner: WP3-app.
//
// The trigger is a <button> with aria-expanded and aria-controls. The popover opens on click / tap / Enter / Space,
// stays while hovered or focused, and closes on Esc, a second click or a click outside. It is a sibling of the button
// (reading order) with fixed positioning so overflow containers never clip it.

import { h } from '../dom.js';
import { t } from '../i18n.js';
import { icon } from '../ui.js';
import { placeUnder } from './popover.js';

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
  const pop = h('div', { class: 'help-pop', id, role: 'note', hidden: true });
  const button = h('button', {
    type: 'button', class: 'help-btn', 'aria-expanded': 'false', 'aria-controls': id,
    'aria-label': t('help.button', { label }),
    on: { click: (e) => { e.stopPropagation(); toggle(); } },
  }, icon('ui-help'));
  const wrap = h('span', { class: ['help', opts.className] }, button, pop);

  function fill() {
    const value = typeof content === 'function' ? content() : content;
    pop.replaceChildren(...[].concat(value).filter((v) => v != null)
      .map((v) => (typeof v === 'string' ? h('p', null, v) : v)));
  }
  function onKey(e) {
    if (e.key === 'Escape') {
      e.stopPropagation();
      close();
      button.focus();
    }
  }
  function onOutside(e) {
    if (!wrap.contains(e.target)) close();
  }
  const reposition = () => placeUnder(pop, button, 6);
  function show() {
    if (openOne && openOne !== api) openOne.close();
    fill();
    pop.hidden = false;
    button.setAttribute('aria-expanded', 'true');
    reposition();
    document.addEventListener('keydown', onKey, true);
    document.addEventListener('pointerdown', onOutside, true);
    window.addEventListener('scroll', reposition, true);
    window.addEventListener('resize', reposition);
    openOne = api;
  }
  function close() {
    if (pop.hidden) return;
    pop.hidden = true;
    button.setAttribute('aria-expanded', 'false');
    document.removeEventListener('keydown', onKey, true);
    document.removeEventListener('pointerdown', onOutside, true);
    window.removeEventListener('scroll', reposition, true);
    window.removeEventListener('resize', reposition);
    if (openOne === api) openOne = null;
  }
  function toggle() {
    if (pop.hidden) show();
    else close();
  }
  const api = { close };
  return wrap;
}
