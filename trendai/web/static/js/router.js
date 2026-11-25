// History API router (spec F.2). Owner: WP3-app.
//
// Intercepts same-origin left clicks on <a href> without modifier keys (and without target/download), keeps all state
// in the URL, re-renders on popstate, and leaves view-only changes to replaceState (the caller decides). The router
// itself is page-agnostic: app.js receives every location change through `onChange`.

import { parseLocation } from './state.js';

/**
 * @param {{onChange: (state: Object, info: {type: 'initial'|'push'|'replace'|'pop', hash: string,
 *          historyState: any}) => void}} opts
 */
export function createRouter(opts) {
  let started = false;

  function current() {
    return parseLocation(window.location.pathname, window.location.search);
  }

  function emit(type) {
    opts.onChange(current(), { type, hash: window.location.hash, historyState: window.history.state });
  }

  /**
   * Go to `url` (same origin, path + query [+ hash]).
   * @param {string} url
   * @param {{replace?: boolean, state?: any, silent?: boolean}} [o]  silent: update the URL without re-rendering
   */
  function navigate(url, o = {}) {
    const target = new URL(url, window.location.href);
    if (target.origin !== window.location.origin) {
      window.location.assign(target.href);
      return;
    }
    const next = target.pathname + target.search + target.hash;
    const same = next === window.location.pathname + window.location.search + window.location.hash;
    if (o.replace || same) window.history.replaceState(o.state ?? null, '', next);
    else window.history.pushState(o.state ?? null, '', next);
    if (!o.silent) emit(o.replace || same ? 'replace' : 'push');
  }

  function onClick(event) {
    if (event.defaultPrevented || event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey
      || event.altKey) return;
    const link = event.target instanceof Element ? event.target.closest('a[href]') : null;
    if (!link || link.hasAttribute('download') || (link.target && link.target !== '_self')) return;
    if (link.getAttribute('rel') === 'external') return;
    const url = new URL(link.href, window.location.href);
    if (url.origin !== window.location.origin) return;
    if (url.pathname.startsWith('/api/') || url.pathname.startsWith('/static/')) return;
    event.preventDefault();
    if (url.pathname === window.location.pathname && url.search === window.location.search && url.hash) {
      // in-page anchor: scroll without a new render
      window.history.pushState(window.history.state, '', url.hash);
      const el = document.getElementById(decodeURIComponent(url.hash.slice(1)));
      if (el) {
        el.scrollIntoView({ block: 'start' });
        if (!el.hasAttribute('tabindex')) el.setAttribute('tabindex', '-1');
        el.focus({ preventScroll: true });
      }
      return;
    }
    navigate(url.pathname + url.search + url.hash);
  }

  return {
    start() {
      if (started) return;
      started = true;
      document.addEventListener('click', onClick);
      window.addEventListener('popstate', () => emit('pop'));
      emit('initial');
    },
    navigate,
    current,
  };
}
