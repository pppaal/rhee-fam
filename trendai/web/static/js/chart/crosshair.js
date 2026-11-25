// Crosshair and read-out plumbing shared by the fan and comparison charts (spec F.7.4).
// Owner: WP3-chart. Internal to the chart modules (pages use FanChart/CompareChart methods instead).
// nearestIndex, keyTarget and throttle are pure (tests/js/chart-crosshair.test.mjs); createCrosshair needs a DOM.

import { svg } from '../dom.js';

/**
 * Index of the point nearest to `ms` in an ascending array (binary search; ties → the earlier point).
 * Pure; unit-testable without a DOM.
 * @param {number[]} sortedMs
 * @param {number} ms
 * @returns {number}  -1 for an empty array
 */
export function nearestIndex(sortedMs, ms) {
  const n = sortedMs.length;
  if (!n) return -1;
  let lo = 0;
  let hi = n - 1;
  while (lo < hi) {                       // first index with value >= ms
    const mid = (lo + hi) >> 1;
    if (sortedMs[mid] < ms) lo = mid + 1;
    else hi = mid;
  }
  if (lo > 0 && ms - sortedMs[lo - 1] <= sortedMs[lo] - ms) return lo - 1;
  return lo;
}

/**
 * Keyboard navigation target (F.7.4): ←/→ one point, Shift+←/→ ten points, Home/End first/last, clamped to
 * [0, length-1]. Returns null for keys the crosshair does not handle (Esc is handled by the caller: hide).
 * When the crosshair is hidden (index -1) an arrow key first shows it at `startIndex`.
 * Pure; unit-testable without a DOM.
 * @param {{key: string, shiftKey?: boolean}} event
 * @param {number} index   current index (-1 when hidden: the first key starts at `startIndex`)
 * @param {number} length
 * @param {number} startIndex  the last actual point
 * @returns {number|null}
 */
export function keyTarget(event, index, length, startIndex) {
  if (!length) return null;
  const clamp = (i) => Math.max(0, Math.min(length - 1, i));
  const step = event.shiftKey ? 10 : 1;
  switch (event.key) {
    case 'Home': return 0;
    case 'End': return length - 1;
    case 'ArrowLeft':
    case 'Left':
      return index < 0 ? clamp(startIndex) : clamp(index - step);
    case 'ArrowRight':
    case 'Right':
      return index < 0 ? clamp(startIndex) : clamp(index + step);
    default:
      return null;
  }
}

/**
 * Call `fn` at most once per `ms` milliseconds, always delivering the latest arguments (trailing call).
 * Used for the #live-polite mirror (250 ms).
 * @template {(...args: any[]) => void} F
 * @param {F} fn
 * @param {number} ms
 * @returns {F & {cancel: () => void}}
 */
export function throttle(fn, ms) {
  let last = -Infinity;
  let timer = null;
  let pending = null;
  const fire = () => {
    timer = null;
    last = Date.now();
    const args = pending;
    pending = null;
    fn(...args);
  };
  const throttled = (...args) => {
    pending = args;
    const wait = last + ms - Date.now();
    if (wait <= 0 && timer === null) fire();
    else if (timer === null) timer = setTimeout(fire, Math.max(0, wait));
  };
  throttled.cancel = () => {
    if (timer !== null) clearTimeout(timer);
    timer = null;
    pending = null;
  };
  return throttled;
}

/**
 * @typedef {Object} CrosshairPoint
 * @property {number} ms     point time (UTC ms)
 * @property {number} x      pixel x in the coordinates of `layer` (the charts use absolute SVG coordinates)
 */

/**
 * @typedef {Object} Crosshair
 * @property {(points: CrosshairPoint[], frame?: {plotTop?: number, plotHeight?: number, startIndex?: number})
 *           => void} setPoints  after a re-render/resize (the optional frame updates the line's extent and the
 *           keyboard start point)
 * @property {(index: number) => void} moveTo
 * @property {() => void} hide
 * @property {() => number} current  current index, -1 when hidden
 * @property {() => void} destroy    remove listeners
 */

/**
 * Attach pointer, touch and keyboard handling. Pointer moves snap the vertical line (drawn in `layer`) to the
 * nearest point and leaving hides it; touch press-and-drag scrubs (the wrapper gets `touch-action: pan-y` via the
 * `.chart-wrap` class, so vertical page scroll still works) and lifting keeps the last read-out; the focused
 * `wrapper` shows the crosshair at `startIndex` and handles keys via keyTarget(); Esc hides. Every change calls
 * `onMove(index | -1, cause)`; the owner renders the read-out text.
 * @param {Object} args
 * @param {HTMLElement} args.wrapper     focusable chart wrapper (tabindex="0")
 * @param {Element} args.plot            element receiving pointer events (the charts pass the wrapper, so pointer
 *                                       events over badges and labels keep the crosshair alive)
 * @param {SVGGElement} args.layer       crosshair layer to draw the line in
 * @param {number} args.plotHeight
 * @param {number} [args.plotTop=0]      y of the plot's top edge in layer coordinates
 * @param {CrosshairPoint[]} args.points
 * @param {number} args.startIndex       index of the last actual point
 * @param {(index: number, cause: 'pointer'|'touch'|'key'|'focus'|'api') => void} args.onMove
 * @returns {Crosshair}
 */
export function createCrosshair(args) {
  const { wrapper, plot, layer, onMove } = args;
  let points = args.points || [];
  let xs = points.map((p) => p.x);
  let plotTop = args.plotTop || 0;
  let plotHeight = args.plotHeight;
  let startIndex = args.startIndex;
  let index = -1;
  let touchActive = false;
  let pointerInside = false;

  const line = svg('line', { class: 'chart-xh-line', x1: 0, x2: 0, y1: 0, y2: 0, visibility: 'hidden' });
  layer.insertBefore(line, layer.firstChild);

  const draw = () => {
    if (index < 0 || index >= points.length) {
      line.setAttribute('visibility', 'hidden');
      return;
    }
    const x = Math.round(points[index].x) + 0.5;
    line.setAttribute('x1', String(x));
    line.setAttribute('x2', String(x));
    line.setAttribute('y1', String(plotTop));
    line.setAttribute('y2', String(plotTop + plotHeight));
    line.setAttribute('visibility', 'visible');
  };

  const set = (i, cause) => {
    const next = i >= 0 && i < points.length ? i : -1;
    if (next === index) return;
    index = next;
    draw();
    onMove(index, cause);
  };

  /** Pointer client x → layer x. */
  const layerX = (clientX, clientY) => {
    const ctm = layer.getScreenCTM && layer.getScreenCTM();
    if (ctm && layer.ownerSVGElement) {
      const pt = layer.ownerSVGElement.createSVGPoint();
      pt.x = clientX;
      pt.y = clientY;
      return pt.matrixTransform(ctm.inverse()).x;
    }
    return clientX - wrapper.getBoundingClientRect().left;
  };

  const snap = (e, cause) => {
    if (!points.length) return;
    set(nearestIndex(xs, layerX(e.clientX, e.clientY)), cause);
  };

  const handlers = {
    pointermove(e) {
      if (e.pointerType === 'touch' || e.pointerType === 'pen') {
        if (touchActive) snap(e, 'touch');
        return;
      }
      pointerInside = true;
      snap(e, 'pointer');
    },
    pointerdown(e) {
      if (e.pointerType !== 'touch' && e.pointerType !== 'pen') return;
      touchActive = true;
      snap(e, 'touch');
    },
    pointerup(e) {
      if (e.pointerType === 'touch' || e.pointerType === 'pen') touchActive = false;   // keep the read-out
    },
    pointercancel() {
      touchActive = false;                  // the browser took over (vertical scroll): keep the read-out
    },
    pointerleave(e) {
      if (e.pointerType === 'touch' || e.pointerType === 'pen') return;
      pointerInside = false;
      if (!keyboardFocused()) set(-1, 'pointer');
    },
  };
  /** True while the wrapper has keyboard focus (then the crosshair belongs to the keyboard, not the mouse). */
  const keyboardFocused = () => {
    if (document.activeElement !== wrapper) return false;
    try {
      return wrapper.matches(':focus-visible');
    } catch {
      return true;
    }
  };
  const onFocus = (e) => {
    if (e.target === wrapper && index < 0) set(startIndex, 'focus');
  };
  const onFocusOut = (e) => {
    if (!wrapper.contains(e.relatedTarget) && !pointerInside && !touchActive) set(-1, 'focus');
  };
  const onKey = (e) => {
    if (e.target !== wrapper) return;
    if (e.key === 'Escape' || e.key === 'Esc') {
      if (index >= 0) {
        set(-1, 'key');
        e.preventDefault();
      }
      return;
    }
    const next = keyTarget(e, index, points.length, startIndex);
    if (next === null) return;
    e.preventDefault();
    set(next, 'key');
  };

  for (const [type, fn] of Object.entries(handlers)) plot.addEventListener(type, fn);
  wrapper.addEventListener('focus', onFocus);
  wrapper.addEventListener('focusout', onFocusOut);
  wrapper.addEventListener('keydown', onKey);

  return {
    setPoints(next, frame = {}) {
      points = next || [];
      xs = points.map((p) => p.x);
      if (frame.plotTop !== undefined) plotTop = frame.plotTop;
      if (frame.plotHeight !== undefined) plotHeight = frame.plotHeight;
      if (frame.startIndex !== undefined) startIndex = frame.startIndex;
      if (index >= points.length) index = -1;
      if (!line.isConnected) layer.insertBefore(line, layer.firstChild);
      draw();
    },
    moveTo(i) {
      set(i, 'api');
    },
    hide() {
      set(-1, 'api');
    },
    current() {
      return index;
    },
    destroy() {
      for (const [type, fn] of Object.entries(handlers)) plot.removeEventListener(type, fn);
      wrapper.removeEventListener('focus', onFocus);
      wrapper.removeEventListener('focusout', onFocusOut);
      wrapper.removeEventListener('keydown', onKey);
      line.remove();
    },
  };
}
