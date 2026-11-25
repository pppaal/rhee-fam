// Entry module (spec F.1): resolves the language, mounts the top bar and footer, loads /api/config, starts the router
// and mounts one page module per route. Owner: WP3-app.
//
// Page modules export `create(ctx)` returning `{el, update?(ctx) → boolean, destroy?()}`. A page that can absorb a
// location change (view-only parameters, focus) returns true from update(); otherwise it is destroyed and the new
// route's page is created. On every route change the app closes every open dialog, sheet, popover and menu (so
// browser Back never leaves a modal over the next page, A11Y-10), scrolls to the top, sets the title, moves focus to
// the page <h1> and announces the title (F.2).

import * as api from './api.js';
import { clear, h } from './dom.js';
import { getLang, resolveLang, setLang, t } from './i18n.js';
import { createRouter } from './router.js';
import { emptyState, serialize, withState } from './state.js';
import * as store from './store.js';
import { announce } from './ui.js';
import { renderFooter } from './components/footer.js';
import { openSettings } from './components/settings.js';
import { closeAllOverlays } from './components/sheet.js';
import { markActive, renderTopbar, syncThemeButton } from './components/topbar.js';
import { errorCard } from './components/errorcard.js';
import * as home from './pages/home.js';
import * as explore from './pages/explore.js';
import * as upload from './pages/upload.js';
import * as result from './pages/result.js';
import * as samples from './pages/samples.js';
import * as info from './pages/info.js';
import * as notfound from './pages/notfound.js';

const PAGES = {
  home, explore, upload, result, shared: result, samples, method: info, sources: info, privacy: info, notfound,
};

/** A Map that keeps only its `limit` most recently set entries (A11Y-20: results are hundreds of KB each). */
class BoundedMap extends Map {
  constructor(limit) {
    super();
    this.limit = limit;
  }

  set(key, value) {
    if (this.has(key)) this.delete(key);
    super.set(key, value);
    while (this.size > this.limit) this.delete(this.keys().next().value);
    return this;
  }
}

const RESULTS_KEPT = 20;

const app = {
  config: null,
  router: null,
  state: null,
  page: null,              // {route, instance}
  results: new BoundedMap(RESULTS_KEPT),     // `${queryKey}|${hl}` or `r:${id}|${hl}` → C.7 envelope (back/forward never re-polls)
  resultIds: new BoundedMap(RESULTS_KEPT * 2), // queryKey → result_id (a language switch re-fetches instead of re-analysing)
  focusLang: null,         // after a language switch: focus this language's button once the chrome is re-rendered
};

const main = () => document.getElementById('main');

function applyUpdownDefault(lang) {
  if (!store.getUpdown()) document.documentElement.dataset.updown = lang === 'ko' ? 'kr' : 'intl';
}

/** The settings dialog; a theme change there ('trendai:theme', below) updates the theme button in place (A11Y-11). */
function settings(opener) {
  return openSettings({ opener });
}

function renderChrome(focusLang) {
  renderTopbar(document.getElementById('topbar'), {
    lang: getLang(),
    onLang: switchLang,
    onSettings: settings,
  });
  renderFooter(document.getElementById('footer'), {
    version: app.config ? app.config.version : '',
    onSettings: settings,
  });
  const skip = document.querySelector('.skip-link');
  if (skip) skip.textContent = t('a11y.skip');
  const hint = document.getElementById('datelink-hint');
  if (hint) hint.textContent = t('a11y.datelink');
  if (focusLang) {
    // the language buttons live in the 메뉴 sheet below 640 px: focus the 메뉴 button there (A11Y-4)
    const btn = document.querySelector(`#topbar [data-lang="${focusLang}"]`);
    const target = btn && btn.offsetParent ? btn : document.querySelector('#topbar .menu-btn');
    if (target) target.focus({ preventScroll: true });
  }
}

/** The language switch (F.4): store it, write `hl` with replaceState, update <html lang>, chrome and page. */
function switchLang(lang) {
  if (lang === getLang()) return;
  store.setHl(lang);
  applyUpdownDefault(lang);
  const state = withState(app.state || emptyState('home'), { hl: lang });
  app.focusLang = lang;           // onChange renders the chrome once, then focuses this language's button
  app.router.navigate(serialize(state) + window.location.hash, { replace: true, state: window.history.state });
  if (app.focusLang) {            // the navigation did not reach onChange (no config yet): render it here
    app.focusLang = null;
    renderChrome(lang);
  }
}

function context(state, routeInfo) {
  return {
    state,
    info: routeInfo,
    config: app.config,
    lang: getLang(),
    results: app.results,
    resultIds: app.resultIds,
    navigate: (url, o) => app.router.navigate(url, o),
    go: (s, o) => app.router.navigate(serialize(s), o),
    setTitle: (title) => {
      document.title = title;
    },
    openSettings: settings,
  };
}

function focusHeading(root) {
  const h1 = root.querySelector('h1');
  if (!h1) {
    root.focus({ preventScroll: true });
    return;
  }
  if (!h1.hasAttribute('tabindex')) h1.setAttribute('tabindex', '-1');
  h1.focus({ preventScroll: true });
}

function scrollToHash(hash) {
  if (!hash || hash.length < 2) return false;
  const el = document.getElementById(decodeURIComponent(hash.slice(1)));
  if (!el) return false;
  el.scrollIntoView({ block: 'start' });
  return true;
}

function onChange(state, routeInfo) {
  if (routeInfo.type !== 'initial') closeAllOverlays();
  if (state.route === 'explore' && !state.q.length) {
    const home = emptyState('home');
    home.hl = state.hl;
    app.router.navigate(serialize(home), { replace: true });
    return;
  }
  const lang = state.hl || getLang();
  const langChanged = lang !== getLang();
  const focusLang = app.focusLang;
  app.focusLang = null;
  if (langChanged) {
    setLang(lang);
    applyUpdownDefault(lang);
    renderChrome(focusLang);
  }
  app.state = state;
  markActive(window.location.pathname);
  if (!app.config) return;      // config failed: the error card stays until a reload

  const ctx = context(state, routeInfo);
  if (app.page && app.page.route === state.route && !langChanged && app.page.instance.update
    && app.page.instance.update(ctx) === true) {
    if (routeInfo.hash) scrollToHash(routeInfo.hash);
    return;
  }
  if (app.page && app.page.instance.destroy) app.page.instance.destroy();
  const root = clear(main());
  const module = PAGES[state.route] || notfound;
  const instance = module.create(ctx);
  root.appendChild(instance.el);
  app.page = { route: state.route, instance };

  if (routeInfo.type === 'initial' || routeInfo.type === 'replace') {
    if (routeInfo.hash) requestAnimationFrame(() => scrollToHash(routeInfo.hash));
    return;
  }
  requestAnimationFrame(() => {
    if (!scrollToHash(routeInfo.hash)) window.scrollTo(0, 0);
    focusHeading(root);
    announce(document.title);
  });
}

async function loadConfig() {
  try {
    app.config = await api.config();
    renderChrome();
    return true;
  } catch (err) {
    const root = clear(main());
    root.appendChild(h('div', { class: 'page page-narrow' },
      h('h1', { class: 'sr-only' }, t('app.name')),
      errorCard(err, { navigate: (url) => window.location.assign(url), retry: () => window.location.reload() })));
    return false;
  }
}

document.addEventListener('trendai:theme', () => syncThemeButton());

async function boot() {
  const initial = new URLSearchParams(window.location.search).get('hl');
  setLang(resolveLang({ url: initial, stored: store.getHl(), languages: navigator.languages || [navigator.language] }));
  applyUpdownDefault(getLang());
  renderChrome();
  app.router = createRouter({ onChange });
  await loadConfig();
  app.router.start();
}

boot();
