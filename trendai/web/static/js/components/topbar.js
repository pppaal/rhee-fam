// Top bar (spec F.6.1): wordmark, main navigation, KO | EN toggle (aria-pressed) and the theme button (cycles
// system → light → dark; the label shows the current mode). Below 640 px the navigation collapses into a 메뉴 button
// that opens a sheet holding the links, the language toggle, the theme choice and 설정. Owner: WP3-app.
//
// A theme change (here, in the menu sheet or in 설정) updates the theme button in place (syncThemeButton, A11Y-11):
// the button is never replaced, so focus and an open sheet's way back to its opener stay intact (A11Y-4).

import { clear, h } from '../dom.js';
import { t } from '../i18n.js';
import { SPRITE, icon } from '../ui.js';
import { applyTheme, currentTheme } from './settings.js';
import { openSheet } from './sheet.js';

const NAV = [['/', 'nav.explore'], ['/upload', 'nav.upload'], ['/samples', 'nav.samples'], ['/method', 'nav.method']];
const THEME_ORDER = ['system', 'light', 'dark'];
const THEME_ICON = { system: 'ui-system', light: 'ui-sun', dark: 'ui-moon' };

/** The nav entry a pathname belongs to ('/' for home and every results page). */
function navFor(pathname) {
  if (pathname.startsWith('/upload')) return '/upload';
  if (pathname.startsWith('/samples')) return '/samples';
  if (pathname.startsWith('/method')) return '/method';
  if (pathname === '/' || pathname.startsWith('/explore') || pathname.startsWith('/r/')) return '/';
  return null;
}

/** Mark the current navigation link with aria-current="page". */
export function markActive(pathname = window.location.pathname) {
  const active = navFor(pathname);
  for (const a of document.querySelectorAll('#topbar .topnav a, .menu-sheet .menu-nav a')) {
    if (a.getAttribute('href') === active) a.setAttribute('aria-current', 'page');
    else a.removeAttribute('aria-current');
  }
}

function langToggle(lang, onLang) {
  return h('div', { class: 'lang-toggle', role: 'group', 'aria-label': t('lang.label') },
    ['ko', 'en'].map((code) => h('button', {
      type: 'button', class: 'lang-btn', dataset: { lang: code }, lang: code,
      // the name starts with the visible code (WCAG 2.5.3): 'KO 한국어', 'EN English'
      'aria-pressed': String(code === lang), 'aria-label': `${t(`lang.${code}`)} ${t(`lang.${code}.name`)}`,
      on: { click: () => onLang(code) },
    }, t(`lang.${code}`))));
}

/** Bring every theme control (top-bar button, menu-sheet chips) up to date with the stored theme, in place. */
export function syncThemeButton() {
  const mode = currentTheme();
  const label = t(`theme.${mode}`);
  for (const btn of document.querySelectorAll('#topbar .theme-btn')) {
    btn.setAttribute('aria-label', t('theme.button', { mode: label }));
    const use = btn.querySelector('svg use');
    if (use) use.setAttribute('href', `${SPRITE}#${THEME_ICON[mode]}`);
    const text = btn.querySelector('.theme-label');
    if (text) text.textContent = label;
  }
  for (const chip of document.querySelectorAll('.menu-theme [data-theme-mode]')) {
    chip.setAttribute('aria-pressed', String(chip.dataset.themeMode === mode));
  }
}

function themeButton() {
  const mode = currentTheme();
  const label = t(`theme.${mode}`);
  return h('button', {
    type: 'button', class: 'icon-btn theme-btn', 'aria-label': t('theme.button', { mode: label }),
    on: {
      click: () => {
        applyTheme(THEME_ORDER[(THEME_ORDER.indexOf(currentTheme()) + 1) % THEME_ORDER.length]);
        syncThemeButton();
      },
    },
  }, icon(THEME_ICON[mode]), h('span', { class: 'theme-label' }, label));
}

/**
 * Render the top bar into `root` (#topbar).
 * @param {HTMLElement} root
 * @param {{lang: 'ko'|'en', onLang: (lang: string) => void, onSettings: (opener: HTMLElement|Function) => void}} opts
 */
export function renderTopbar(root, opts) {
  const menuBtn = h('button', {
    type: 'button', class: 'icon-btn menu-btn', 'aria-haspopup': 'dialog',
    on: { click: () => openMenu(opts) },
  }, icon('ui-menu'), h('span', null, t('nav.menu')));

  clear(root);
  root.appendChild(h('div', { class: 'topbar-inner' },
    h('a', { class: 'wordmark', href: '/' },
      h('span', { class: 'wordmark-mark', 'aria-hidden': 'true' }, icon('ph-growing')), t('app.name')),
    h('nav', { class: 'topnav', 'aria-label': t('nav.label') },
      NAV.map(([href, key]) => h('a', { href }, t(key)))),
    h('div', { class: 'topbar-tools' },
      langToggle(opts.lang, opts.onLang),
      themeButton(),
      menuBtn)));
  markActive();
}

/** The 메뉴 button as it is now (the top bar is re-rendered on a language change). */
const menuButtonNow = () => document.querySelector('#topbar .menu-btn');

function openMenu(opts) {
  const close = () => sheet.close();
  const themeChoices = h('div', { class: 'menu-theme', role: 'group', 'aria-label': t('settings.theme') },
    THEME_ORDER.map((mode) => h('button', {
      type: 'button', class: 'chip chip-toggle', 'aria-pressed': String(currentTheme() === mode), dataset: { themeMode: mode },
      on: {
        click: () => {
          applyTheme(mode);
          syncThemeButton();
        },
      },
    }, icon(THEME_ICON[mode]), t(`theme.${mode}`))));
  const content = [
    h('nav', { class: 'menu-nav', 'aria-label': t('nav.label') },
      NAV.map(([href, key]) => h('a', { href, on: { click: close } }, t(key)))),
    h('div', { class: 'menu-row' }, h('span', { class: 'field-label' }, t('lang.label')),
      langToggle(opts.lang, (code) => { sheet.close(false); opts.onLang(code); })),
    h('div', { class: 'menu-row' }, h('span', { class: 'field-label' }, t('settings.theme')), themeChoices),
    h('button', {
      type: 'button', class: 'btn btn-secondary btn-block',
      on: { click: () => { sheet.close(false); opts.onSettings(menuButtonNow); } },
    }, t('footer.links.settings')),
  ];
  // the opener is looked up when the sheet closes: a language change re-renders the top bar (A11Y-4)
  const sheet = openSheet({ title: t('nav.menu'), content, opener: menuButtonNow, className: 'menu-sheet' });
  markActive();
}
