// Comparison chart (spec F.7.2, F.7.4, F.9): one line per series in its slot colour, dashed forecasts, legend
// chips, direct line-end labels with slot-icon end markers, focused-series 80 % band, raw / index / log scales.
// Owner: WP3-chart. Called by views/compare.js (WP3-app).
//
// Slots come from compareVM.slots (authoritative, C.6), never from seriesVM.slot — except while no compare VM
// exists yet (progressive rendering before the first compare snapshot), when the provisional seriesVM.slot is used.
// Series with slot null are drawn grey (1 px --ink-3, opacity .5, no label) and highlighted on click until clicked
// again.
// Text sources: VM labels (series names, compare.axis_label, chart.x_label of the series, values via
// fmt.valueText + source.unit_label). Catalog keys used through `opts.t`: 'chart.index_axis' (axis label under the
// index scale), 'chart.forecast' (forecast region label), 'chart.readout_range' (bands_only values in the
// read-out, with 'chart.readout_forecast' as the fallback), 'common.reference_only' (legend chip of bands_only
// series), 'a11y.chart_keys', 'compare.title' (<title>), 'compare.grey_hint' (shown when grey series exist), plus
// describe.js's keys for <desc>; through common.uiText (fallback text while the catalog lacks them):
// 'compare.bands_hint' (whose band 예측 범위 보기 shows), 'chart.window_note' / 'chart.window_full' (the narrow-width
// history window).
//
// DOM (classes styled by css/chart.css):
//   .chart.chart-compare
//     .chart-legend > button.chart-legend-chip[aria-pressed]   swatch + slot icon + name (+ 참고용)
//     p.chart-hint                                              compare.grey_hint, only with grey series
//     .chart-readout                                            every visible series at the crosshair date
//     .chart-wrap[tabindex=0][role=group] > svg.chart-svg[role=img]
//     p.chart-hint.chart-band-hint                              while 예측 범위 보기 is on: whose band is shown
//     .chart-range                                              below 640 px when the history is windowed (UX-4)

import { h, svg, clear } from '../dom.js';
import * as fmt from '../fmt.js';
import {
  compareYDomain, dodge, groupAnchors, isoToMs, linearScale, logScale, narrowWindow, tickCount, tickLabels, timeTicks,
} from './scale.js';
import { drawXAxis, drawYAxis, measureLabels, textWidths, X_CAPTION_SPACE, X_TICK_SPACE } from './axes.js';
import { createCrosshair, nearestIndex, shouldAnnounce, throttle } from './crosshair.js';
import { areaPath, extent, linePath, seamForecast, slotIconPath, truncateToWidth, withUnit } from './geom.js';
import { describeCompare } from './describe.js';
import {
  WIDE_PX, contentWidth, defaultAnnounce, joinParts, lineKey, observeWidth, rangeControl, renderReadout, slotGlyph,
  uid, uiText,
} from './common.js';

const HEIGHT_WIDE = 340;
const HEIGHT_NARROW = 260;
const MARGIN_TOP = 24;
const LABEL_BAND = 22;
const LABEL_GAP = 14;                       // F.7.2: direct labels ≥ 14 px apart
const LABEL_MERGE_PX = 4;                   // line ends this close (same date) share one label `A · B` (UX-16)
const LABEL_BOTTOM = 7;                     // a label's centre stays this far above the x axis line (UX-16)
const HOVER_PX = 8;                         // mouse: a line within 8 px of the pointer takes the focus
const TOUCH_PX = 16;                        // touch: a line within 16 px of the finger takes the focus (UX-14)
const ICONS = ['circle', 'diamond', 'hexagon', 'square', 'cross'];
const isNum = (v) => typeof v === 'number' && Number.isFinite(v);

/**
 * @typedef {import("./fanchart.js").ChartBaseOptions & {
 *   scale?: 'raw'|'index'|'log',
 *   bands?: boolean,
 *   hidden?: Iterable<number>,
 *   focus?: number|null,
 *   fullRange?: boolean
 * }} CompareChartOptions
 * scale: URL `scale` param (default compareVM.scale.default; a disallowed scale falls back to 'raw');
 * bands: the 예측 범위 보기 toggle (default false); hidden: series indexes switched off in the legend;
 * focus: series index whose band is shown (when `bands`) and whose line is emphasised; fullRange: below 640 px,
 * show the whole history instead of the default window (scale.narrowWindow; flipped by the chart's own toggle).
 */

/**
 * @typedef {Object} CompareChart
 * @property {(compareVM: Object, seriesVMs: Object[], opts?: Partial<CompareChartOptions>) => void} update
 *           re-render (language switch, final result after progressive VMs, option change)
 * @property {(scale: 'raw'|'index'|'log') => void} setScale
 * @property {(on: boolean) => void} setBands
 * @property {(index: number|null) => void} setFocus  from ranking-card hover/focus
 * @property {(indexes: Iterable<number>) => void} setHidden
 * @property {(cb: (index: number, visible: boolean) => void) => void} onLegendToggle  a legend chip was toggled
 *           (the chip's aria-pressed is already updated)
 * @property {(cb: (index: number|null) => void) => void} onFocusChange  line/legend hover or focus changed
 *           (pages highlight the matching card)
 * @property {(iso: string) => boolean} highlightDate  move the crosshair to that date; false when absent
 * @property {() => SVGSVGElement} getSvg  the live chart <svg> (input of png.exportPng)
 * @property {() => void} destroy
 */

/** Per-series data in request order, plus the union of all dates for the crosshair. */
function buildModel(compareVM, seriesVMs) {
  const vms = (seriesVMs || []).filter((vm) => vm && vm.chart).slice().sort((a, b) => a.index - b.index);
  const slots = compareVM && compareVM.slots;
  const displayOf = new Map(((compareVM && compareVM.ranking) || []).map((r) => [r.index, r.display]));
  const series = vms.map((vm) => {
    const c = vm.chart;
    const raw = slots ? slots[String(vm.index)] : vm.slot;
    const slot = Number.isInteger(raw) && raw >= 1 && raw <= 5 ? raw : null;
    const display = c.display || displayOf.get(vm.index) || 'normal';
    const last = c.last_actual && isNum(c.last_actual.value) ? c.last_actual : null;
    return {
      k: vm.index,
      name: vm.name,
      slot,
      icon: slot ? ICONS[slot - 1] : null,
      cls: slot ? `chart-slot-${slot}` : 'chart-grey',
      display: ['weak', 'bands_only'].includes(display) ? display : 'normal',
      unit: vm.source ? vm.source.unit_label : null,
      dates: c.dates || [],
      actual: (c.actual || []).map((v) => (isNum(v) ? v : null)),
      fc: seamForecast(last, c.forecast),
      last,
      xLabel: c.x_label || null,
      byDate: null,
    };
  });
  for (const s of series) {
    s.byDate = new Map();
    s.dates.forEach((iso, i) => s.byDate.set(iso, { part: 'history', v: s.actual[i] }));
    s.fc.dates.forEach((iso, j) => {
      if (s.fc.seam && j === 0) return;            // the seam point is the last actual, already listed
      s.byDate.set(iso, { part: 'forecast', v: s.fc.mean[j], lo: s.fc.lo80[j], hi: s.fc.hi80[j] });
    });
  }
  const isoSet = new Set();
  for (const s of series) for (const iso of s.byDate.keys()) isoSet.add(iso);
  const isos = [...isoSet].sort();
  const points = isos.map((iso) => ({ iso, ms: isoToMs(iso) }));
  const lastIso = series.reduce((m, s) => (s.last && s.last.date > m ? s.last.date : m), '');
  const xLabels = new Set(series.map((s) => s.xLabel));
  const rankOrder = ((compareVM && compareVM.ranking) || []).slice()
    .sort((a, b) => (a.rank ?? Infinity) - (b.rank ?? Infinity)).map((r) => r.index);
  return {
    series,
    points,
    indexByDate: new Map(isos.map((iso, i) => [iso, i])),
    lastIso,
    startIndex: lastIso ? isos.indexOf(lastIso) : Math.max(0, isos.length - 1),
    valueKind: (compareVM && compareVM.value_kind) || (vms[0] && vms[0].chart.value_kind) || 'raw',
    freq: (vms[0] && vms[0].chart.freq) || 'D',
    caption: xLabels.size === 1 ? [...xLabels][0] : null,
    rankOrder,
  };
}

/**
 * Render the comparison chart into `container` (children replaced): legend chips (`button[aria-pressed]`, swatch +
 * slot icon + name), read-out bar, focusable wrapper and `<svg role="img">` (height 340 / 260 px). Relative
 * indexes use the fixed 0–100 frame (scale.compareYDomain). Weak-forecast rule per series from its chart.display.
 * Below 640 px a long history is windowed (scale.narrowWindow, with a 전체 기간 toggle under the plot) and, with
 * three or more direct labels, the labels give way to the legend chips (end markers stay).
 * May be called with a partial set while the job runs; call update() with the final envelope's VMs.
 * @param {HTMLElement} container
 * @param {Object} compareVM   D.4 compare VM (slots, value_kind, axis_label, scale, ranking)
 * @param {Object[]} seriesVMs D.2 series VMs of the successful series, request order (chart.*, name, source)
 * @param {CompareChartOptions} opts
 * @returns {CompareChart}
 */
export function createCompareChart(container, compareVM, seriesVMs, opts) {
  const o = { bands: false, ...opts };
  let cvm = compareVM;
  let svms = seriesVMs;
  let model = buildModel(cvm, svms);
  let hidden = new Set(o.hidden || []);
  let extFocus = o.focus ?? null;
  let localFocus = null;
  const highlighted = new Set();
  const ids = { title: uid('cmp-t'), desc: uid('cmp-d'), clip: uid('cmp-c') };
  const legendListeners = [];
  const focusListeners = [];
  let geo = null;

  const legend = h('div', { class: 'chart-legend' });
  const hint = h('p', { class: 'chart-hint', hidden: true });
  const readout = h('div', { class: 'chart-readout' });
  const svgEl = svg('svg', {
    class: 'chart-svg', role: 'img', 'aria-labelledby': `${ids.title} ${ids.desc}`, focusable: 'false',
  });
  const xhLayer = svg('g', { class: 'chart-xh chart-noexport' });
  const dots = svg('g', { class: 'chart-xh-dots' });
  xhLayer.appendChild(dots);
  const wrapper = h('div', { class: 'chart-wrap', tabindex: '0', role: 'group' }, svgEl);
  // below the plot, so its text changing with the focus never moves the chart
  const bandHint = h('p', { class: 'chart-hint chart-band-hint', hidden: true });
  const range = rangeControl(() => {
    o.fullRange = !o.fullRange;
    render();
  });
  const root = h('div', { class: ['chart', 'chart-compare'] }, legend, hint, readout, wrapper, bandHint, range.el);
  clear(container).appendChild(root);

  const t = (key, params) => o.t(key, params);
  const announce = throttle((text) => (o.announce || defaultAnnounce)(text), 250);

  // The crosshair works on the points inside the x window (geo.off = index of the first one); everything else uses
  // indexes into model.points.
  const offset = () => (geo ? geo.off : 0);
  const currentK = () => {
    const i = crosshair.current();
    return i < 0 ? -1 : i + offset();
  };
  /** Move the crosshair to model point k, showing the full period first when k lies before the window. */
  function moveToK(k) {
    if (!(k >= 0) || !geo) return;
    if (k < geo.off) {
      o.fullRange = true;
      render();
    }
    if (geo) crosshair.moveTo(k - geo.off);
  }

  // ------------------------------------------------------------------------------------------ scale

  function scaleBase(k) {
    const values = cvm && cvm.scale && cvm.scale.index_base && cvm.scale.index_base.values;
    const b = values ? values[String(k)] : null;
    return isNum(b) && b > 0 ? b : null;
  }

  function effectiveScale() {
    const wanted = o.scale || (cvm && cvm.scale && cvm.scale.default) || 'raw';
    if (wanted === 'raw') return 'raw';
    const allowed = cvm && cvm.scale && cvm.scale.allowed && cvm.scale.allowed[wanted] === true;
    if (!allowed) return 'raw';
    if (wanted === 'index' && model.series.some((s) => scaleBase(s.k) === null)) return 'raw';
    return wanted;
  }

  /** Value under the current scale (index: v / base × 100; log: positive values only). */
  function tx(s, v, scale) {
    if (!isNum(v)) return null;
    if (scale === 'index') {
      const b = scaleBase(s.k);
      return b ? (v / b) * 100 : null;
    }
    if (scale === 'log') return v > 0 ? v : null;
    return v;
  }

  // ------------------------------------------------------------------------------------------ focus & legend

  const focusK = () => {
    const f = localFocus ?? extFocus;
    return f !== null && f !== undefined && !hidden.has(f) && model.series.some((s) => s.k === f) ? f : null;
  };

  /**
   * The series whose 80 % band is drawn: the focused one; while 예측 범위 보기 is on and nothing is focused, the
   * best-ranked visible series with a forecast (request order before the ranking exists), so the toggle always
   * shows a band — touch screens have no hover, and a legend tap hides the series (UX-14).
   */
  function bandSeries(f) {
    const hasFc = (s) => s && !hidden.has(s.k) && s.fc.dates.length > 0;
    if (f !== null) {
      const fs = model.series.find((s) => s.k === f);
      return hasFc(fs) ? fs : null;
    }
    if (!o.bands) return null;
    for (const k of model.rankOrder) {
      const s = model.series.find((it) => it.k === k);
      if (hasFc(s)) return s;
    }
    return model.series.find(hasFc) || null;
  }

  function setLocalFocus(k) {
    if (k === localFocus) return;
    localFocus = k;
    drawDynamic();
    for (const cb of focusListeners) cb(k);
  }

  function buildLegend() {
    const active = legend.contains(document.activeElement) ? document.activeElement.getAttribute('data-k') : null;
    clear(legend);
    for (const s of model.series) {
      const chip = h('button', {
        type: 'button', class: ['chart-legend-chip', s.cls], 'aria-pressed': hidden.has(s.k) ? 'false' : 'true',
        'data-k': s.k,
        on: {
          click: () => {
            const visible = hidden.has(s.k);
            if (visible) hidden.delete(s.k);
            else hidden.add(s.k);
            chip.setAttribute('aria-pressed', visible ? 'true' : 'false');
            render();
            for (const cb of legendListeners) cb(s.k, visible);
          },
          mouseenter: () => setLocalFocus(s.k),
          mouseleave: () => setLocalFocus(null),
          focus: () => setLocalFocus(s.k),
          blur: () => setLocalFocus(null),
        },
      },
      lineKey('solid', s.cls),
      s.slot ? slotGlyph(s.icon, s.cls) : null,
      h('span', { class: 'chart-chip-name' }, s.name),
      s.display === 'bands_only' ? h('span', { class: 'chart-chip-ref' }, t('common.reference_only')) : null);
      legend.appendChild(chip);
    }
    const anyGrey = model.series.some((s) => !s.slot);
    hint.hidden = !anyGrey;
    clear(hint);
    if (anyGrey) hint.textContent = t('compare.grey_hint');
    if (active !== null) {
      const again = legend.querySelector(`[data-k="${active}"]`);
      if (again) again.focus({ preventScroll: true });
    }
  }

  // ------------------------------------------------------------------------------------------ read-out

  function valueText(s, v, scale) {
    if (scale === 'index') return fmt.valueText(v, 'relative_index', o.lang);
    return withUnit(fmt.valueText(v, model.valueKind, o.lang), s.unit);
  }

  /**
   * Read-out of every visible series at point k, highest value first (F.7.4). When every row is a forecast the date
   * says so once (`… · 예측`); a forecast row next to history rows (series with different end dates) carries the
   * full forecast text. A bands_only forecast shows its 80 % range only (no mean, the weak-forecast rule).
   */
  function readoutContent(k) {
    const p = model.points[k];
    if (!p) return null;
    const scale = effectiveScale();
    const entries = model.series.filter((s) => !hidden.has(s.k) && s.byDate.has(p.iso))
      .map((s) => ({ s, e: s.byDate.get(p.iso) }));
    const allForecast = entries.length > 0 && entries.every(({ e }) => e.part === 'forecast');
    const rows = entries.map(({ s, e }) => {
      const vt = (v) => valueText(s, tx(s, v, scale), scale);
      if (e.part === 'history') return { s, text: vt(e.v), sortV: tx(s, e.v, scale) };
      const params = { mean: vt(e.v), lo: vt(e.lo), hi: vt(e.hi) };
      if (s.display === 'bands_only') {
        let text = t('chart.readout_range', params);
        if (!text || text === 'chart.readout_range') text = t('chart.readout_forecast', params);
        return { s, text, sortV: isNum(e.lo) && isNum(e.hi) ? tx(s, (e.lo + e.hi) / 2, scale) : null };
      }
      return { s, text: allForecast ? vt(e.v) : t('chart.readout_forecast', params), sortV: tx(s, e.v, scale) };
    });
    rows.sort((a, b) => (isNum(b.sortV) ? b.sortV : -Infinity) - (isNum(a.sortV) ? a.sortV : -Infinity));
    const date = fmt.dateLabel(p.iso, model.freq, o.lang);
    const dateText = allForecast ? joinParts([date, t('chart.forecast')]) : date;
    const view = rows.map((r) => ({ key: slotGlyph(r.s.icon, r.s.cls), name: r.s.name, text: r.text }));
    return { date: dateText, rows: view, text: joinParts([dateText, ...rows.map((r) => `${r.s.name} ${r.text}`)]) };
  }

  function showReadout(k) {
    const idle = k < 0;
    const content = readoutContent(idle ? model.startIndex : k);
    if (!content) {
      clear(readout);
      return null;
    }
    renderReadout(readout, { ...content, idle });
    return content;
  }

  function drawDots(k) {
    clear(dots);
    const p = model.points[k];
    if (!p || !geo) return;
    const x = geo.x.map(p.ms);
    for (const s of model.series) {
      if (hidden.has(s.k)) continue;
      const e = s.byDate.get(p.iso);
      if (!e || (e.part === 'forecast' && s.display === 'bands_only')) continue;
      const v = tx(s, e.v, geo.scale);
      if (v === null) continue;
      dots.appendChild(svg('circle', { class: ['chart-xh-dot', s.cls], cx: x, cy: geo.y.map(v), r: 3.5 }));
    }
  }

  function onMove(i, cause) {
    const k = i < 0 ? -1 : i + offset();
    const content = showReadout(k);
    drawDots(k);
    if (k >= 0 && content && shouldAnnounce(cause)) announce(content.text);
  }

  const crosshair = createCrosshair({
    wrapper, plot: wrapper, layer: xhLayer, plotHeight: 0, points: [], startIndex: model.startIndex, onMove,
  });

  // line hover → focus (mouse: within 8 px of a visible line at the nearest date). Touch has no hover: pressing or
  // scrubbing within 16 px of a line focuses it until the next touch (a touch away from every line clears it), so
  // the 예측 범위 보기 band can be switched on phones too (UX-14).
  /** Series key of the visible line nearest to the client point within `tol` px at the nearest date, or null. */
  function lineAt(clientX, clientY, tol) {
    if (!geo) return null;
    const ctm = svgEl.getScreenCTM();
    if (!ctm) return null;
    const pt = svgEl.createSVGPoint();
    pt.x = clientX;
    pt.y = clientY;
    const { x: px, y: py } = pt.matrixTransform(ctm.inverse());
    const i = nearestIndex(geo.xs, px);
    const p = model.points[i < 0 ? -1 : i + geo.off];
    let best = null;
    let bestD = tol;
    for (const s of model.series) {
      if (hidden.has(s.k) || !p) continue;
      const entry = s.byDate.get(p.iso);
      if (!entry || (entry.part === 'forecast' && s.display === 'bands_only')) continue;
      const v = tx(s, entry.v, geo.scale);
      if (v === null) continue;
      const d = Math.abs(geo.y.map(v) - py);
      if (d < bestD) {
        bestD = d;
        best = s.k;
      }
    }
    return best;
  }

  let hoverFrame = 0;
  let touchDown = false;
  const isTouch = (e) => e.pointerType === 'touch' || e.pointerType === 'pen';
  const onHover = (e) => {
    if (isTouch(e)) {
      if (!touchDown) return;
      const k = lineAt(e.clientX, e.clientY, TOUCH_PX);
      if (k !== null) setLocalFocus(k);                // scrubbing off a line keeps the last one
      return;
    }
    if (e.pointerType !== 'mouse' || hoverFrame) return;
    const { clientX, clientY } = e;
    hoverFrame = requestAnimationFrame(() => {
      hoverFrame = 0;
      setLocalFocus(lineAt(clientX, clientY, HOVER_PX));
    });
  };
  const onDown = (e) => {
    if (!isTouch(e)) return;
    touchDown = true;
    setLocalFocus(lineAt(e.clientX, e.clientY, TOUCH_PX));
  };
  const onUp = (e) => {
    if (isTouch(e)) touchDown = false;
  };
  const onLeave = (e) => {
    if (!isTouch(e)) setLocalFocus(null);             // a lifted finger also "leaves": keep its focus
  };
  wrapper.addEventListener('pointermove', onHover);
  wrapper.addEventListener('pointerdown', onDown);
  wrapper.addEventListener('pointerup', onUp);
  wrapper.addEventListener('pointercancel', onUp);
  wrapper.addEventListener('pointerleave', onLeave);

  // ------------------------------------------------------------------------------------------ render

  /** The 전체 기간 toggle: shown below 640 px whenever a window applies (also while the full period is on). */
  function updateRange(win) {
    range.set({
      shown: Boolean(win),
      full: Boolean(o.fullRange),
      note: win ? uiText(t, o.lang, 'chart.window_note', { span: fmt.spanLabel(win.n, win.unit, o.lang) }) : '',
      label: win ? uiText(t, o.lang, 'chart.window_full') : '',
    });
  }

  /**
   * Where each visible series ends: the end marker sits where its drawn line ends (forecast end, or the last actual
   * for bands_only); the label anchor is the same point, except for bands_only series, whose label is anchored at
   * the middle of their 80 % band at its last date — the seam would pull a long leader across the forecast region
   * (UX-16). Pixel y; times in ms.
   */
  function lineEnds(visible, scale, y) {
    const ends = new Map();
    for (const s of visible) {
      let mark = null;
      if (s.last) {
        const lv = tx(s, s.last.value, scale);
        if (lv !== null) mark = { ms: isoToMs(s.last.date), y: y.map(lv) };
      }
      let anchor = null;
      for (let j = s.fc.dates.length - 1; j >= 0 && !anchor; j -= 1) {
        const ms = isoToMs(s.fc.dates[j]);
        if (s.display === 'bands_only') {
          const lo = tx(s, s.fc.lo80[j], scale);
          const hi = tx(s, s.fc.hi80[j], scale);
          if (lo !== null && hi !== null) anchor = { ms, y: (y.map(lo) + y.map(hi)) / 2 };
        } else {
          const v = tx(s, s.fc.mean[j], scale);
          if (v !== null) {
            anchor = { ms, y: y.map(v) };
            mark = anchor;
          }
        }
      }
      if (mark) ends.set(s.k, { mark, anchor: anchor || mark });
    }
    return ends;
  }

  /** Static layers: frame, axes, lines, markers. Then drawDynamic() for focus, band and labels. */
  function render() {
    const width = contentWidth(container);
    const keepK = geo ? currentK() : -1;
    clear(svgEl);
    clear(dots);
    const title = t('compare.title');
    wrapper.setAttribute('aria-label', joinParts([title, t('a11y.chart_keys')]));
    svgEl.append(svg('title', { id: ids.title }, title),
      svg('desc', { id: ids.desc }, describeCompare(cvm, svms, t)));
    const { series, points } = model;
    if (!width || !points.length) {
      geo = null;
      updateRange(null);
      bandHint.hidden = true;
      showReadout(-1);
      return;
    }
    const wide = width >= WIDE_PX;
    const height = wide ? HEIGHT_WIDE : HEIGHT_NARROW;
    svgEl.setAttribute('width', String(width));
    svgEl.setAttribute('height', String(height));
    svgEl.setAttribute('viewBox', `0 0 ${width} ${height}`);

    const scale = effectiveScale();
    const visible = series.filter((s) => !hidden.has(s.k));

    // y domain: visible history + drawn forecast means (+ every visible 80 % band while 예측 범위 보기 is on, so
    // the axis stays put while the focus moves between series)
    const data = [];
    const band = [];
    for (const s of visible) {
      data.push(s.actual.map((v) => tx(s, v, scale)));
      if (s.display !== 'bands_only') data.push(s.fc.mean.map((v) => tx(s, v, scale)));
      if (o.bands) band.push(s.fc.lo80.map((v) => tx(s, v, scale)), s.fc.hi80.map((v) => tx(s, v, scale)));
    }
    const dataExt = extent(...data);
    const bandExt = extent(...band);
    const count = tickCount(width);
    const dom = compareYDomain({
      valueKind: model.valueKind, scale, count,
      dataMin: dataExt ? dataExt.min : null, dataMax: dataExt ? dataExt.max : null,
      bandMax: bandExt ? bandExt.max : null, bandMin: bandExt ? bandExt.min : null,
    });
    const ticks = dom.ticks;
    const labelKind = scale === 'index' ? 'relative_index' : model.valueKind;
    const yLabels = tickLabels(ticks.values, dom.log ? 0 : ticks.decimals, o.lang, { kind: labelKind, log: dom.log });

    // y first: it does not depend on the right margin, which the direct labels decide
    const caption = model.caption;
    const T = MARGIN_TOP;
    const B = height - (X_TICK_SPACE + (caption ? X_CAPTION_SPACE : 0));
    let y;
    if (dom.log) {
      const top = 10 ** (Math.log10(ticks.max)
        + ((Math.log10(ticks.max) - Math.log10(ticks.min)) * LABEL_BAND) / Math.max(1, B - T - LABEL_BAND));
      y = logScale([ticks.min, top], [B, T]);
    } else {
      const top = ticks.max + ((ticks.max - ticks.min) * LABEL_BAND) / Math.max(1, B - T - LABEL_BAND);
      y = linearScale([ticks.min, top], [B, T]);
    }

    // direct labels: every visible slotted series, plus highlighted grey ones; lines ending in one spot share a
    // label (UX-16); below 640 px three or more labels give way to the legend chips, which name every line with
    // its slot icon, so the forecast keeps its width (UX-4)
    const ends = lineEnds(visible, scale, y);
    for (const e of ends.values()) e.anchor.y = Math.min(B, Math.max(T, e.anchor.y));   // a band beyond the axis
    const labelled = visible.filter((s) => (s.slot || highlighted.has(s.k)) && ends.has(s.k));
    let groups = groupAnchors(labelled.map((s) => ends.get(s.k).anchor), LABEL_MERGE_PX)
      .map((g) => g.map((i) => labelled[i]));
    const dropLabels = !wide && groups.length >= 3;
    if (dropLabels) groups = [];
    const maxLabel = wide ? 140 : 84;
    const measureLabel = (str) => textWidths(svgEl, [str], 'chart-end-label')[0];
    const labelGroups = groups.map((members) => {
      const text = truncateToWidth(members.map((s) => s.name).join(' · '),
        members.length > 1 ? maxLabel + 60 : maxLabel, measureLabel);
      return { keys: members.map((s) => s.k), text, w: measureLabel(text) };
    });
    const labelW = labelGroups.length ? Math.max(...labelGroups.map((g) => g.w)) : 0;

    const margin = {
      top: T,
      right: labelGroups.length ? labelW + 22 : 16,
      bottom: height - B,
      left: Math.max(measureLabels(svgEl, yLabels) + 8, 20),
    };
    const first = points[0].ms;
    const t1 = points[points.length - 1].ms > first ? points[points.length - 1].ms : first + 86400000;
    const seamMs = model.lastIso ? isoToMs(model.lastIso) : t1;
    // below 640 px a long history leaves the forecast ~20 px: show the last years only, unless asked (UX-4)
    const win = wide ? null : narrowWindow(first, seamMs, t1, model.freq);
    const t0 = win && !o.fullRange ? win.startMs : first;
    const off = Math.max(0, points.findIndex((p) => p.ms >= t0));
    const fcLabel = t('chart.forecast');
    const fcLabelW = textWidths(svgEl, [fcLabel], 'chart-region-label')[0];
    let x = linearScale([t0, t1], [margin.left, width - margin.right]);
    if (seamMs < t1 && width - margin.right - x.map(seamMs) < fcLabelW + 12 && margin.right < 48) {
      margin.right = 48;
      x = linearScale([t0, t1], [margin.left, width - margin.right]);
    }
    const L = margin.left;
    const R = width - margin.right;
    const frame = { width, height, margin };
    for (const e of ends.values()) {
      e.mark.x = x.map(e.mark.ms);
      e.anchor.x = x.map(e.anchor.ms);
    }
    geo = {
      x, y, scale, frame, L, R, T, B, off, t0, ends, labelGroups, dropLabels,
      xs: points.slice(off).map((p) => x.map(p.ms)),
    };
    updateRange(win);

    svgEl.appendChild(svg('defs', null, svg('clipPath', { id: ids.clip },
      svg('rect', { x: L - 2, y: T - 3, width: R - L + 4, height: B - T + 6 }))));

    // axis label (top margin)
    const axisLabel = scale === 'index' ? t('chart.index_axis') : (cvm && cvm.axis_label) || '';
    if (axisLabel) {
      const measure = (str) => textWidths(svgEl, [str], 'chart-y-label')[0];
      svgEl.appendChild(svg('text', { class: 'chart-y-label', x: 0, y: 14 }, truncateToWidth(axisLabel, width - 4, measure)));
    }

    // forecast region (where every series is forecast)
    const seamX = x.map(seamMs);
    if (seamMs < t1 && R > seamX) {
      svgEl.appendChild(svg('rect', { class: 'chart-fc-region', x: seamX, y: T, width: R - seamX, height: B - T }));
      svgEl.appendChild(svg('text', { class: 'chart-region-label', x: seamX + 6, y: T + 15 }, fcLabel));
    }

    const gridG = svg('g', { class: 'chart-grid' });
    drawYAxis(gridG, frame, y, ticks.values, (v, i) => yLabels[i]);
    const xTicks = timeTicks(t0, t1, Math.max(2, Math.floor((R - L) / (wide ? 90 : 72))));
    drawXAxis(gridG, frame, x, xTicks, { lang: o.lang, caption, svgEl });
    svgEl.appendChild(gridG);

    // data: the focused band (dynamic), then series lines — grey first, slot 1 on top
    const dataG = svg('g', { class: 'chart-data', 'clip-path': `url(#${ids.clip})` });
    geo.bandLayer = svg('g', { class: 'chart-bands' });
    dataG.appendChild(geo.bandLayer);
    geo.groups = new Map();
    const order = visible.slice().sort((a, b) => (a.slot ? 6 - a.slot : 0) - (b.slot ? 6 - b.slot : 0));
    for (const s of order) {
      const g = svg('g', { class: ['chart-series', s.cls], 'data-k': s.k });
      const histXs = s.dates.map((iso) => x.map(isoToMs(iso)));
      g.appendChild(svg('path', {
        class: 'chart-line chart-line-hist',
        d: linePath(s.actual.map((v, i) => { const tv = tx(s, v, scale); return tv === null ? null : [histXs[i], y.map(tv)]; })),
      }));
      if (s.display !== 'bands_only' && s.fc.dates.length) {
        const pts = s.fc.mean.map((v, j) => { const tv = tx(s, v, scale); return tv === null ? null : [x.map(isoToMs(s.fc.dates[j])), y.map(tv)]; });
        g.appendChild(svg('path', { class: ['chart-line', 'chart-line-fc', s.display === 'weak' && 'is-weak'], d: linePath(pts) }));
      }
      geo.groups.set(s.k, g);
      dataG.appendChild(g);
    }
    svgEl.appendChild(dataG);

    // click targets for grey lines (highlight until clicked again)
    const hits = svg('g', { class: 'chart-hits', 'clip-path': `url(#${ids.clip})` });
    for (const s of visible.filter((v) => !v.slot)) {
      const pts = [...s.byDate.entries()].map(([iso, e]) => {          // history then forecast, by date
        const v = e.part === 'forecast' && s.display === 'bands_only' ? null : tx(s, e.v, scale);
        return v === null ? null : [x.map(isoToMs(iso)), y.map(v)];
      });
      hits.appendChild(svg('path', {
        class: 'chart-hit', d: linePath(pts), 'data-k': s.k,
        on: {
          click: (e) => {
            e.stopPropagation();
            if (highlighted.has(s.k)) highlighted.delete(s.k);
            else highlighted.add(s.k);
            render();
          },
        },
      }));
    }
    svgEl.appendChild(hits);

    geo.markers = svg('g', { class: 'chart-markers' });
    geo.labels = svg('g', { class: 'chart-labels' });
    svgEl.append(geo.markers, geo.labels, xhLayer);
    drawDynamic();

    crosshair.setPoints(points.slice(off).map((p, i) => ({ ms: p.ms, x: geo.xs[i] })),
      { plotTop: T, plotHeight: B - T, startIndex: Math.max(0, model.startIndex - off) });
    if (keepK >= 0 && keepK !== currentK()) {                           // the window moved under the crosshair
      if (keepK >= off) crosshair.moveTo(keepK - off);
      else crosshair.hide();
    }
    const k = currentK();
    showReadout(k);
    drawDots(k);
  }

  /** Focus classes, the shown 80 % band and its hint, end markers and direct labels. */
  function drawDynamic() {
    if (!geo) return;
    const f = focusK();
    const { x, y, scale, R, T, B } = geo;
    for (const [k, g] of geo.groups) {
      g.classList.toggle('is-focused', f === k);
      g.classList.toggle('is-dimmed', f !== null && f !== k);
      g.classList.toggle('is-highlight', highlighted.has(k) || (f === k && !model.series.find((s) => s.k === k).slot));
    }
    // band (예측 범위 보기): the focused series', else the default one (bandSeries)
    clear(geo.bandLayer);
    const fs = o.bands ? bandSeries(f) : null;
    if (fs) {
      const xs = fs.fc.dates.map((iso) => x.map(isoToMs(iso)));
      const py = (arr) => arr.map((v) => { const tv = tx(fs, v, scale); return tv === null ? null : y.map(tv); });
      geo.bandLayer.appendChild(svg('g', { class: ['chart-series', fs.cls] },
        svg('path', { class: 'chart-series-band', d: areaPath(xs, py(fs.fc.lo80), py(fs.fc.hi80)) })));
    }
    bandHint.hidden = !fs;
    bandHint.textContent = fs ? uiText(t, o.lang, 'compare.bands_hint', { name: fs.name }) : '';

    // end markers: every visible slotted series, plus highlighted or focused grey ones
    clear(geo.markers);
    clear(geo.labels);
    const items = model.series.filter((s) => !hidden.has(s.k) && geo.ends.has(s.k)
      && (s.slot || highlighted.has(s.k) || f === s.k));
    for (const s of items) {
      const { mark } = geo.ends.get(s.k);
      geo.markers.appendChild(svg('path', {
        class: ['chart-end-marker', s.cls, f !== null && f !== s.k && 'is-dimmed'],
        d: slotIconPath(s.icon || 'circle', mark.x, mark.y, s.icon ? 4 : 3),
      }));
    }
    if (geo.dropLabels) return;

    // direct labels (groups from render(), plus a focused grey line), nudged ≥ 14 px apart above the x axis
    const measure = (str) => textWidths(svgEl, [str], 'chart-end-label')[0];
    const groups = geo.labelGroups.slice();
    if (f !== null && geo.ends.has(f) && !groups.some((g) => g.keys.includes(f))) {
      const name = model.series.find((s) => s.k === f).name;
      groups.push({ keys: [f], text: truncateToWidth(name, Math.max(40, geo.frame.margin.right - 22), measure) });
    }
    const anchorOf = (g) => {
      const list = g.keys.map((k) => geo.ends.get(k).anchor);
      return { x: list[0].x, y: list.reduce((sum, a) => sum + a.y, 0) / list.length };
    };
    const anchors = groups.map(anchorOf);
    const placed = dodge(anchors.map((a) => a.y), LABEL_GAP, T + 6, B - LABEL_BOTTOM);
    groups.forEach((g, i) => {
      const end = anchors[i];
      const dim = f !== null && !g.keys.includes(f);
      const ly = placed[i];
      const lx = R + 10;
      if (Math.abs(ly - end.y) > 6 || end.x < R - 2) {
        geo.labels.appendChild(svg('path', {
          class: ['chart-leader', dim && 'is-dimmed'], d: `M${end.x + 6},${end.y}L${lx - 3},${ly}`,
        }));
      }
      geo.labels.appendChild(svg('text', {
        class: ['chart-end-label', dim && 'is-dimmed'], x: lx, y: ly, dy: '0.34em',
      }, g.text));
    });
  }

  buildLegend();
  render();
  const stopObserving = observeWidth(container, () => render());

  return {
    update(compareVM2, seriesVMs2, opts2 = {}) {
      const keep = currentK() >= 0 ? model.points[currentK()]?.iso : null;
      if (compareVM2) cvm = compareVM2;
      if (seriesVMs2) svms = seriesVMs2;
      Object.assign(o, opts2);
      if (opts2.hidden) hidden = new Set(opts2.hidden);
      if (opts2.focus !== undefined) extFocus = opts2.focus;
      model = buildModel(cvm, svms);
      crosshair.hide();
      buildLegend();
      render();
      if (keep && model.indexByDate.has(keep)) moveToK(model.indexByDate.get(keep));
    },
    setScale(scale) {
      o.scale = scale;
      render();
    },
    setBands(on) {
      o.bands = Boolean(on);
      render();
    },
    setFocus(index) {
      extFocus = index ?? null;
      drawDynamic();
    },
    setHidden(indexes) {
      hidden = new Set(indexes || []);
      for (const chip of legend.querySelectorAll('[data-k]')) {
        chip.setAttribute('aria-pressed', hidden.has(Number(chip.getAttribute('data-k'))) ? 'false' : 'true');
      }
      render();
    },
    onLegendToggle(cb) {
      legendListeners.push(cb);
    },
    onFocusChange(cb) {
      focusListeners.push(cb);
    },
    highlightDate(iso) {
      const k = model.indexByDate.get(iso);
      if (k === undefined || !geo) return false;
      moveToK(k);
      return true;
    },
    getSvg() {
      return svgEl;
    },
    destroy() {
      stopObserving();
      announce.cancel();
      crosshair.destroy();
      if (hoverFrame) cancelAnimationFrame(hoverFrame);
      wrapper.removeEventListener('pointermove', onHover);
      wrapper.removeEventListener('pointerdown', onDown);
      wrapper.removeEventListener('pointerup', onUp);
      wrapper.removeEventListener('pointercancel', onUp);
      wrapper.removeEventListener('pointerleave', onLeave);
      legendListeners.length = 0;
      focusListeners.length = 0;
      clear(container);
    },
  };
}
