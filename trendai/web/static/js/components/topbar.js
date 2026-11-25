// Top bar (spec F.6.1): wordmark, main navigation, KO | EN toggle (aria-pressed) and the theme button (cycles
// system → light → dark; the label shows the current mode). Below 640 px the navigation collapses into a 메뉴 button
// that opens a sheet holding the links, the language toggle, the theme choice and 설정. Owner: WP3-app.

import { clear, h } from '../dom.js';
import { t } from '../i18n.js';
import { icon } from '../ui.js';
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
      'aria-pressed': String(code === lang), 'aria-label': t(`lang.${code}.name`),
      on: { click: () => onLang(code) },
    }, t(`lang.${code}`))));
}

function themeButton(onChange) {
  const mode = currentTheme();
  const label = t(`theme.${mode}`);
  return h('button', {
    type: 'button', class: 'icon-btn theme-btn', 'aria-label': t('theme.button', { mode: label }),
    on: {
      click: () => {
        const next = THEME_ORDER[(THEME_ORDER.indexOf(currentTheme()) + 1) % THEME_ORDER.length];
        applyTheme(next);
        onChange();
      },
    },
  }, icon(THEME_ICON[mode]), h('span', { class: 'theme-label' }, label));
}

/**
 * Render the top bar into `root` (#topbar).
 * @param {HTMLElement} root
 * @param {{lang: 'ko'|'en', onLang: (lang: string) => void, onSettings: (opener: HTMLElement) => void}} opts
 */
export function renderTopbar(root, opts) {
  const rerender = (focusTheme) => {
    renderTopbar(root, opts);
    if (focusTheme) {
      const btn = root.querySelector('.theme-btn');
      if (btn) btn.focus();
    }
  };
  const menuBtn = h('button', {
    type: 'button', class: 'icon-btn menu-btn', 'aria-haspopup': 'dialog',
    on: { click: (e) => openMenu(e.currentTarget, opts, () => rerender(false)) },
  }, icon('ui-menu'), h('span', null, t('nav.menu')));

  clear(root);
  root.appendChild(h('div', { class: 'topbar-inner' },
    h('a', { class: 'wordmark', href: '/' },
      h('span', { class: 'wordmark-mark', 'aria-hidden': 'true' }, icon('ph-growing')), t('app.name')),
    h('nav', { class: 'topnav', 'aria-label': t('nav.label') },
      NAV.map(([href, key]) => h('a', { href }, t(key)))),
    h('div', { class: 'topbar-tools' },
      langToggle(opts.lang, opts.onLang),
      themeButton(() => rerender(true)),
      menuBtn)));
  markActive();
}

function openMenu(opener, opts, onThemeChange) {
  const close = () => sheet.close();
  const themeChoices = h('div', { class: 'menu-theme', role: 'group', 'aria-label': t('settings.theme') },
    THEME_ORDER.map((mode) => h('button', {
      type: 'button', class: 'chip', 'aria-pressed': String(currentTheme() === mode),
      on: {
        click: (e) => {
          applyTheme(mode);
          for (const b of e.currentTarget.parentElement.children) b.setAttribute('aria-pressed', String(b === e.currentTarget));
          onThemeChange();
        },
      },
    }, icon(THEME_ICON[mode]), t(`theme.${mode}`))));
  const content = [
    h('nav', { class: 'menu-nav', 'aria-label': t('nav.label') },
      NAV.map(([href, key]) => h('a', { href, on: { click: close } }, t(key)))),
    h('div', { class: 'menu-row' }, h('span', { class: 'field-label' }, t('lang.label')),
      langToggle(opts.lang, (code) => { close(); opts.onLang(code); })),
    h('div', { class: 'menu-row' }, h('span', { class: 'field-label' }, t('settings.theme')), themeChoices),
    h('button', {
      type: 'button', class: 'btn btn-secondary btn-block',
      on: { click: () => { close(); opts.onSettings(opener); } },
    }, t('footer.links.settings')),
  ];
  const sheet = openSheet({ title: t('nav.menu'), content, opener, className: 'menu-sheet' });
  markActive();
}
