// Settings dialog (spec F.6.17): theme (system / light / dark) and up/down colours (Korean / international / follow
// the language), stored in trendai.theme / trendai.updown and applied immediately. Owner: WP3-app.

import { h } from '../dom.js';
import { getLang, t } from '../i18n.js';
import * as store from '../store.js';
import { openSheet } from './sheet.js';

/** Apply a theme choice ('system' | 'light' | 'dark') to <html> and storage. */
export function applyTheme(mode) {
  const root = document.documentElement;
  if (mode === 'light' || mode === 'dark') root.dataset.theme = mode;
  else delete root.dataset.theme;
  store.setTheme(mode === 'light' || mode === 'dark' ? mode : null);
}

/** Apply an up/down choice ('kr' | 'intl' | 'auto'). */
export function applyUpdown(mode) {
  store.setUpdown(mode === 'kr' || mode === 'intl' ? mode : null);
  const effective = mode === 'kr' || mode === 'intl' ? mode : (getLang() === 'ko' ? 'kr' : 'intl');
  document.documentElement.dataset.updown = effective;
}

/** The current theme choice. */
export function currentTheme() {
  return store.getTheme() || 'system';
}

let counter = 0;

function radioGroup(name, legend, options, value, onChange) {
  const id = `${name}-${++counter}`;
  return h('fieldset', { class: 'field radio-group' },
    h('legend', { class: 'field-label' }, legend),
    h('div', { class: 'radio-list' }, options.map(([v, label]) => h('label', { class: 'radio' },
      h('input', {
        type: 'radio', name: id, value: v, props: { checked: v === value },
        on: { change: (e) => { if (e.target.checked) onChange(v); } },
      }),
      h('span', null, label)))));
}

/**
 * Open the settings dialog.
 * @param {{opener?: HTMLElement, onThemeChange?: (mode: string) => void}} [opts]
 */
export function openSettings(opts = {}) {
  const content = [
    radioGroup('theme', t('settings.theme'),
      [['system', t('theme.system')], ['light', t('theme.light')], ['dark', t('theme.dark')]],
      currentTheme(), (mode) => {
        applyTheme(mode);
        document.dispatchEvent(new CustomEvent('trendai:theme'));
        if (opts.onThemeChange) opts.onThemeChange(mode);
      }),
    radioGroup('updown', t('settings.updown'),
      [['kr', t('settings.updown.kr')], ['intl', t('settings.updown.intl')], ['auto', t('settings.updown.auto')]],
      store.getUpdown() || 'auto', applyUpdown),
  ];
  return openSheet({ title: t('settings.title'), content, opener: opts.opener, className: 'sheet-small' });
}
