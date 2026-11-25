// Detail fan chart (spec F.7.1, F.7.4, F.9): history, trend, forecast mean, 80/95 % bands, changepoint badges,
// anomaly rings, "now" line, keyboard/pointer crosshair with a fixed read-out bar.
// Owner: WP3-chart. Called by views/detail.js (WP3-app).
//
// Text sources (F.1 rule 3): everything data-related comes from the series VM (chart.title, chart.desc,
// chart.x_label, chart.y_label, chart.y_note is rendered by the page, last_actual.label, changepoints[].text,
// anomalies[].text, source.unit_label). The few UI words a chart needs come from `opts.t` with these catalog keys
// (G.1, G.6): 'chart.forecast' (forecast region label), 'chart.readout_actual', 'chart.readout_forecast',
// 'chart.readout_range' (bands_only read-out; falls back to 'chart.readout_forecast' while the key is missing),
// 'common.reference_only' (weak / bands_only chip), 'a11y.chart_keys' (wrapper aria-label suffix).
//
// DOM (all classes styled by css/chart.css):
//   .chart.chart-fan
//     .chart-readout                      fixed read-out bar (F.7.4)
//     .chart-wrap[tabindex=0][role=group] focusable wrapper; pointer/touch/keyboard crosshair
//       svg.chart-svg[role=img]           the picture (title, desc, every visual layer) — getSvg(), PNG export
//       svg.chart-overlay                 interactive layer: changepoint badge buttons (an <svg role="img"> exposes
//                                         no children to assistive technology, so the buttons live beside it)

import { h, svg, clear } from '../dom.js';
import * as fmt from '../fmt.js';
import { detailYDomain, dodge, isoToMs, linearScale, tickCount, tickLabels, timeTicks } from './scale.js';
import { drawXAxis, drawYAxis, measureLabels, textWidths, X_CAPTION_SPACE, X_TICK_SPACE } from './axes.js';
import { createCrosshair, throttle } from './crosshair.js';
import { areaPath, circled, extent, linePath, seamForecast, truncateToWidth, withUnit } from './geom.js';
import { describeSeries } from './describe.js';
import {
  WIDE_PX, contentWidth, defaultAnnounce, joinParts, lineKey, observeWidth, renderReadout, svgChip, uid,
} from './common.js';

const HEIGHT_WIDE = 320;
const HEIGHT_NARROW = 240;
const MARGIN_TOP = 24;
/** Label band at the top of the plot, kept free of data: forecast/now labels, changepoint badges, edge marks. */
const LABEL_BAND = 24;
const BADGE_R = 9;
const isNum = (v) => typeof v === 'number' && Number.isFinite(v);

/**
 * Options shared by the chart components.
 * @typedef {Object} ChartBaseOptions
 * @property {'ko'|'en'} lang                         number/date formatting language (fmt.js)
 * @property {(key: string, params?: Object) => string} t  catalog lookup (i18n.js `t`), see the key lists
 * @property {(text: string) => void} [announce]      live-region writer for the crosshair read-out (called at most
 *                                                    every 250 ms); default: sets textContent of #live-polite
 */

/**
 * @typedef {ChartBaseOptions & {
 *   band?: 'both'|'80'|'95'|'off',
 *   trend?: boolean,
 *   anomalies?: boolean
 * }} FanChartOptions
 * band: URL `band` param (default 'both'); trend / anomalies: the 추세선 / 이상치 toggles (default true).
 */

/**
 * Payload of a crosshair move (null when the crosshair is hidden).
 * @typedef {Object} PointFocus
 * @property {string} date             ISO date under the crosshair
 * @property {number} index            index into the combined history + forecast point list
 * @property {'history'|'forecast'} part
 * @property {string} text             the read-out text as displayed
 */

/**
 * @typedef {Object} FanChart
 * @property {(vm: Object, opts?: Partial<FanChartOptions>) => void} update  re-render with a new series VM
 *           (e.g. language switch) and/or options; keeps the crosshair date when it still exists
 * @property {(band: 'both'|'80'|'95'|'off') => void} setBand
 * @property {(toggles: {trend?: boolean, anomalies?: boolean}) => void} setToggles
 * @property {(iso: string) => boolean} highlightDate  move the crosshair to that history/forecast date (narrative
 *           date links, F.6.14); false when the date is not on the chart. Does not scroll — the page does.
 * @property {(n: number|null) => void} highlightChangepoint  emphasise badge ① … (from the past.js list); null clears
 * @property {(cb: (n: number) => void) => void} onChangepoint  a changepoint badge was activated (click/Enter/Space)
 * @property {(cb: (p: PointFocus|null) => void) => void} onPointFocus  crosshair moved / hidden
 * @property {() => SVGSVGElement} getSvg  the live chart <svg> (input of png.exportPng)
 * @property {() => void} destroy  remove listeners and the ResizeObserver, empty the container
 */

/** Everything the renderer needs from a series VM, computed once per VM. */
function buildModel(vm) {
  const c = vm.chart || {};
  const dates = c.dates || [];
  const actual = c.actual || [];
  const trend = c.trend || [];
  const fcIn = c.forecast || { dates: [] };
  const fcDates = fcIn.dates || [];
  const anomalies = c.anomalies || [];
  const anomalyByDate = new Map(anomalies.map((a, i) => [a.date, i]));

  const points = [
    ...dates.map((iso, i) => ({ iso, ms: isoToMs(iso), part: 'history', i, v: isNum(actual[i]) ? actual[i] : null })),
    ...fcDates.map((iso, j) => ({ iso, ms: isoToMs(iso), part: 'forecast', i: j,
      mean: isNum(fcIn.mean?.[j]) ? fcIn.mean[j] : null,
      lo: isNum(fcIn.lo80?.[j]) ? fcIn.lo80[j] : null,
      hi: isNum(fcIn.hi80?.[j]) ? fcIn.hi80[j] : null })),
  ];
  const indexByDate = new Map(points.map((p, k) => [p.iso, k]));

  let last = c.last_actual && isNum(c.last_actual.value) ? c.last_actual : null;
  if (!last) {
    for (let i = dates.length - 1; i >= 0; i -= 1) {
      if (isNum(actual[i])) {
        last = { date: dates[i], value: actual[i], label: null };
        break;
      }
    }
  }
  const lastIndex = last && indexByDate.has(last.date) ? indexByDate.get(last.date) : Math.max(0, dates.length - 1);

  const cps = (c.changepoints || []).map((cp) => ({ ...cp, index: indexByDate.get(cp.date) ?? -1 }));
  const cpByDate = new Map(cps.filter((cp) => cp.index >= 0).map((cp) => [cp.date, cp]));

  const nonAnomalous = actual.map((v, i) => (anomalyByDate.has(dates[i]) ? null : v));
  const rest = extent(nonAnomalous, fcIn.mean, fcIn.lo95, fcIn.hi95, fcIn.lo80, fcIn.hi80, last ? [last.value] : []);

  return {
    vm, c, dates, actual, trend, fcDates, anomalies, anomalyByDate, points, indexByDate, last, lastIndex, cps,
    cpByDate, rest,
    seam: seamForecast(last, fcIn),
    display: ['weak', 'bands_only'].includes(c.display) ? c.display : 'normal',
    kind: c.value_kind || (vm.source && vm.source.value_kind) || 'raw',
    unit: vm.source ? vm.source.unit_label : null,
    freq: c.freq || (vm.source && vm.source.freq) || 'D',
  };
}

/**
 * Render the fan chart of one series into `container` (its children are replaced). Builds, in order: the read-out
 * bar (`.chart-readout`, F.7.4), and the focusable wrapper (`tabindex="0"`, `role="group"`, aria-label
 * `chart.title + ' · ' + t('a11y.chart_keys')`) holding `<svg role="img" aria-labelledby>` with `<title>` =
 * chart.title and `<desc>` = chart.desc. Height 320 px (container >= 640 px wide) / 240 px; recomputed on resize
 * (ResizeObserver, debounced 100 ms). Display rule from chart.display (normal / weak / bands_only).
 * @param {HTMLElement} container
 * @param {Object} seriesVM  D.2 series VM (uses chart.*, source.unit_label, source.value_kind, lang-independent data)
 * @param {FanChartOptions} opts
 * @returns {FanChart}
 */
export function createFanChart(container, seriesVM, opts) {
  const o = { band: 'both', trend: true, anomalies: true, ...opts };
  const ids = { title: uid('fan-t'), desc: uid('fan-d'), clip: uid('fan-c') };
  const cpListeners = [];
  const pointListeners = [];
  let model = buildModel(seriesVM);
  let geo = null;
  let activeCp = null;

  const readout = h('div', { class: 'chart-readout' });
  const svgEl = svg('svg', {
    class: 'chart-svg', role: 'img', 'aria-labelledby': `${ids.title} ${ids.desc}`, focusable: 'false',
  });
  const overlay = svg('svg', { class: 'chart-overlay', role: 'presentation', focusable: 'false' });
  const xhLayer = svg('g', { class: 'chart-xh chart-noexport' });
  const dots = svg('g', { class: 'chart-xh-dots' });
  xhLayer.appendChild(dots);
  const wrapper = h('div', { class: 'chart-wrap', tabindex: '0', role: 'group' }, svgEl, overlay);
  const root = h('div', { class: ['chart', 'chart-fan'] }, readout, wrapper);
  clear(container).appendChild(root);

  const t = (key, params) => o.t(key, params);
  const announce = throttle((text) => (o.announce || defaultAnnounce)(text), 250);

  // ------------------------------------------------------------------------------------------ read-out texts

  const value = (v) => withUnit(fmt.valueText(v, model.kind, o.lang), model.unit);

  /** Read-out content of point k (F.7.4): date, actual or forecast with its 80 % range, changepoint/anomaly text. */
  function readoutContent(k) {
    const p = model.points[k];
    if (!p) return null;
    const date = fmt.dateLabel(p.iso, model.freq, o.lang);
    let row;
    if (p.part === 'history') {
      row = { key: lineKey('solid', 'chart-key-hist'), text: t('chart.readout_actual', { value: value(p.v) }) };
    } else {
      const params = { mean: value(p.mean), lo: value(p.lo), hi: value(p.hi) };
      let text;
      if (model.display === 'bands_only') {
        text = t('chart.readout_range', params);
        if (!text || text === 'chart.readout_range') text = t('chart.readout_forecast', params);
      } else {
        text = t('chart.readout_forecast', params);
      }
      row = { key: lineKey(model.display === 'bands_only' ? 'band' : 'dashed', 'chart-key-fc'), text };
    }
    const notes = [];
    const cp = model.cpByDate.get(p.iso);
    if (cp) notes.push(`${circled(cp.n)} ${joinParts([cp.label, cp.strength_label])}`);
    if (p.part === 'history' && model.anomalyByDate.has(p.iso)) notes.push(model.anomalies[model.anomalyByDate.get(p.iso)].text);
    const note = notes.join(' · ') || null;
    return { date, rows: [row], note, text: joinParts([date, row.text, note]) };
  }

  function showReadout(k) {
    const idle = k < 0;
    const content = readoutContent(idle ? model.lastIndex : k);
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
    if (p.part === 'history' && p.v !== null) {
      dots.appendChild(svg('circle', { class: 'chart-xh-dot chart-xh-dot-hist', cx: x, cy: geo.y.map(p.v), r: 4 }));
    } else if (p.part === 'forecast' && p.mean !== null && model.display !== 'bands_only') {
      dots.appendChild(svg('circle', { class: 'chart-xh-dot chart-xh-dot-fc', cx: x, cy: geo.y.map(p.mean), r: 4 }));
    }
  }

  function onMove(k) {
    const content = showReadout(k);
    drawDots(k);
    if (k >= 0 && content) announce(content.text);
    const p = model.points[k];
    const payload = k >= 0 && p && content ? { date: p.iso, index: k, part: p.part, text: content.text } : null;
    for (const cb of pointListeners) cb(payload);
  }

  const crosshair = createCrosshair({
    wrapper, plot: wrapper, layer: xhLayer, plotHeight: 0, points: [], startIndex: model.lastIndex, onMove,
  });

  // ------------------------------------------------------------------------------------------ changepoints

  function setActiveCp(n) {
    activeCp = n ?? null;
    for (const el of root.querySelectorAll('[data-cp]')) {
      el.classList.toggle('is-active', Number(el.getAttribute('data-cp')) === activeCp);
    }
  }

  function activateCp(n) {
    setActiveCp(n);
    const cp = model.cps.find((c) => c.n === n);
    if (cp && cp.index >= 0) crosshair.moveTo(cp.index);
    for (const cb of cpListeners) cb(n);
  }

  // ------------------------------------------------------------------------------------------ render

  function applyToggles() {
    root.dataset.band = ['both', '80', '95', 'off'].includes(String(o.band)) ? String(o.band) : 'both';
    root.classList.toggle('chart-no-trend', o.trend === false);
    root.classList.toggle('chart-no-anoms', o.anomalies === false);
  }

  /** Scales, margins and ticks for the current width (F.7.1 size, margins, y range, label band). */
  function layout(width) {
    const { c, points } = model;
    const wide = width >= WIDE_PX;
    const height = wide ? HEIGHT_WIDE : HEIGHT_NARROW;
    const { ticks, excluded } = detailYDomain(c.y_hint, model.anomalies, model.last, tickCount(width), model.rest);
    const yLabels = tickLabels(ticks.values, ticks.decimals, o.lang, { kind: model.kind });
    const caption = c.x_label || null;
    const margin = {
      top: MARGIN_TOP, right: 16, bottom: X_TICK_SPACE + (caption ? X_CAPTION_SPACE : 0),
      left: Math.max(measureLabels(svgEl, yLabels) + 8, 20),            // widest y label + 8
    };
    const t0 = points[0].ms;
    const t1 = points[points.length - 1].ms > t0 ? points[points.length - 1].ms : t0 + 86400000;
    const hasFc = model.seam.dates.length > 0;
    const seamMs = model.last ? isoToMs(model.last.date) : t1;
    const fcLabel = t('chart.forecast');
    let x = linearScale([t0, t1], [margin.left, width - margin.right]);
    if (hasFc && width - margin.right - x.map(seamMs) < textWidths(svgEl, [fcLabel], 'chart-region-label')[0] + 12) {
      margin.right = 48;                                                 // room for the forecast region label
      x = linearScale([t0, t1], [margin.left, width - margin.right]);
    }
    // keep the data below the label band: extend the domain upwards by the band's share of the plot
    const plotH = height - margin.bottom - margin.top;
    const yTop = ticks.max + ((ticks.max - ticks.min) * LABEL_BAND) / Math.max(1, plotH - LABEL_BAND);
    const y = linearScale([ticks.min, yTop], [height - margin.bottom, margin.top]);
    return {
      width, height, wide, margin, x, y, ticks, excluded, yLabels, caption, t0, t1, hasFc, fcLabel,
      frame: { width, height, margin },
      L: margin.left, R: width - margin.right, T: margin.top, B: height - margin.bottom,
      seamX: x.map(seamMs), chipW: 0, yLabelRight: 0, badgeXs: [],
    };
  }

  /** Clip path, top margin (y label, 참고용 chip), forecast region, grid and axes. */
  function drawFrame(g) {
    const { c } = model;
    const { width, margin, L, R, T, B } = g;
    svgEl.appendChild(svg('defs', null, svg('clipPath', { id: ids.clip },
      svg('rect', { x: L - 2, y: T - 3, width: R - L + 4, height: B - T + 6 }))));
    if (model.display !== 'normal') {
      const text = t('common.reference_only');
      const chip = svgChip(text, width - margin.right, 2, textWidths(svgEl, [text], 'chart-ref-chip-text')[0]);
      g.chipW = chip.width + 8;
      svgEl.appendChild(chip.node);
    }
    if (c.y_label) {
      const measure = (str) => textWidths(svgEl, [str], 'chart-y-label')[0];
      const text = truncateToWidth(c.y_label, width - margin.right - g.chipW - 4, measure);
      g.yLabelRight = measure(text);
      svgEl.appendChild(svg('text', { class: 'chart-y-label', x: 0, y: 14 }, text));
    }
    if (g.hasFc && R > g.seamX) {
      svgEl.appendChild(svg('rect', { class: 'chart-fc-region', x: g.seamX, y: T, width: R - g.seamX, height: B - T }));
      svgEl.appendChild(svg('text', { class: 'chart-region-label', x: g.seamX + 6, y: T + 15 }, g.fcLabel));
    }
    const grid = svg('g', { class: 'chart-grid' });
    drawYAxis(grid, g.frame, g.y, g.ticks.values, (v, i) => g.yLabels[i]);
    const xTicks = timeTicks(g.t0, g.t1, Math.max(2, Math.floor((R - L) / (g.wide ? 90 : 72))));
    drawXAxis(grid, g.frame, g.x, xTicks, { lang: o.lang, caption: g.caption, svgEl });
    svgEl.appendChild(grid);
  }

  /** Bands (95 under 80), trend, history and forecast mean, clipped to the plot; the seam is in model.seam. */
  function drawData(g) {
    const { x, y } = g;
    const data = svg('g', { class: 'chart-data', 'clip-path': `url(#${ids.clip})` });
    const s = model.seam;
    const fcXs = s.dates.map((d) => x.map(isoToMs(d)));
    const py = (arr) => arr.map((v) => (isNum(v) ? y.map(v) : null));
    if (g.hasFc) {
      data.append(
        svg('path', { class: 'chart-band chart-band-95', d: areaPath(fcXs, py(s.lo95), py(s.hi95)) }),
        svg('path', { class: 'chart-band chart-band-80', d: areaPath(fcXs, py(s.lo80), py(s.hi80)) }),
      );
    }
    const histXs = model.dates.map((d) => x.map(isoToMs(d)));
    const line = (xs, values) => linePath(values.map((v, i) => (isNum(v) ? [xs[i], y.map(v)] : null)));
    if (model.trend.length) data.appendChild(svg('path', { class: 'chart-trend', d: line(histXs, model.trend) }));
    data.appendChild(svg('path', { class: 'chart-hist', d: line(histXs, model.actual) }));
    if (g.hasFc && model.display !== 'bands_only') {
      data.appendChild(svg('path', {
        class: ['chart-fc-mean', model.display === 'weak' && 'is-weak'], d: line(fcXs, s.mean),
      }));
    }
    svgEl.appendChild(data);
  }

  /** Changepoint lines and numbered badges in the label band (≥ 2r + 2 apart), with button twins in the overlay. */
  function drawChangepoints(g) {
    const shown = model.cps.filter((cp) => cp.on_chart && cp.index >= 0);
    const trueXs = shown.map((cp) => g.x.map(isoToMs(cp.date)));
    g.badgeXs = dodge(trueXs, BADGE_R * 2 + 2, g.L + BADGE_R, g.R - BADGE_R);
    const cy = g.T + BADGE_R + 2;
    const group = svg('g', { class: 'chart-cps' });
    shown.forEach((cp, i) => {
      const bx = g.badgeXs[i];
      const lx = Math.round(trueXs[i]) + 0.5;
      const top = cy + BADGE_R;
      const d = Math.abs(bx - lx) > 1 ? `M${bx},${top}L${lx},${top + 8}L${lx},${g.B}` : `M${lx},${top}L${lx},${g.B}`;
      group.append(
        svg('path', { class: 'chart-cp-line', d, 'data-cp': cp.n }),
        svg('g', { class: 'chart-cp-badge', 'data-cp': cp.n },
          svg('circle', { class: 'chart-cp-dot', cx: bx, cy, r: BADGE_R }),
          svg('text', { class: 'chart-cp-num', x: bx, y: cy, dy: '0.35em', 'text-anchor': 'middle' }, String(cp.n))),
      );
      overlay.appendChild(svg('g', {
        class: 'chart-cp-hit', role: 'button', tabindex: '0', 'aria-label': cp.text || `${circled(cp.n)} ${cp.label}`,
        'data-cp': cp.n,
        on: {
          click: (e) => {
            e.stopPropagation();
            activateCp(cp.n);
          },
          keydown: (e) => {
            if (e.key === 'Enter' || e.key === ' ' || e.key === 'Spacebar') {
              e.preventDefault();
              activateCp(cp.n);
            }
          },
        },
      },
      svg('circle', { class: 'chart-cp-hit-area', cx: bx, cy, r: BADGE_R + 4 }),
      svg('circle', { class: 'chart-cp-ring', cx: bx, cy, r: BADGE_R + 3 })));
    });
    svgEl.appendChild(group);
  }

  /**
   * The "now" line at the last actual date and its label, right-aligned to the line: in the label band when no
   * badge is in the way, else in the top margin when the y label and chip leave room, else left out (the idle
   * read-out already names the last actual).
   */
  function drawNow(g) {
    if (!model.last || !g.hasFc) return;
    const nx = Math.round(g.seamX) + 0.5;
    svgEl.appendChild(svg('line', { class: 'chart-now', x1: nx, x2: nx, y1: g.T, y2: g.B }));
    const label = model.last.label;
    if (!label) return;
    const right = g.seamX - 6;
    const left = right - textWidths(svgEl, [label], 'chart-now-label')[0];
    const inBand = left >= g.L + 4 && !g.badgeXs.some((bx) => bx + BADGE_R + 4 > left && bx - BADGE_R - 4 < right);
    const inMargin = left >= g.yLabelRight + 12 && right <= g.width - g.margin.right - g.chipW;
    if (inBand || inMargin) {
      svgEl.appendChild(svg('text', {
        class: 'chart-now-label', x: right, y: inBand ? g.T + 15 : 14, 'text-anchor': 'end',
      }, label));
    }
  }

  /** Anomaly rings in range; excluded anomalies as triangles at the top/bottom edge with their value (F.7.1). */
  function drawAnomalies(g) {
    const group = svg('g', { class: 'chart-anoms' });
    const excluded = new Set(g.excluded);
    model.anomalies.forEach((a, i) => {
      if (!isNum(a.value) || !model.indexByDate.has(a.date)) return;
      const ax = g.x.map(isoToMs(a.date));
      if (!excluded.has(i)) {
        group.appendChild(svg('circle', { class: 'chart-anom', cx: ax, cy: g.y.map(a.value), r: 4 }));
        return;
      }
      const up = a.value > g.ticks.max;
      const underBadge = up && g.badgeXs.some((bx) => Math.abs(bx - ax) < BADGE_R + 6);
      const tip = up ? g.T + 1 + (underBadge ? BADGE_R * 2 + 4 : 0) : g.B - 1;
      const base = up ? tip + 8 : tip - 8;
      const text = fmt.valueText(a.value, model.kind, o.lang, true);
      const flip = ax + 8 + textWidths(svgEl, [text], 'chart-anom-edge')[0] > g.R;
      group.append(
        svg('path', { class: 'chart-anom-edge-mark', d: `M${ax - 5},${base}L${ax + 5},${base}L${ax},${tip}Z` }),
        svg('text', {
          class: 'chart-anom-edge', x: flip ? ax - 8 : ax + 8, y: up ? base : tip - 1,
          'text-anchor': flip ? 'end' : 'start',
        }, text),
      );
    });
    svgEl.appendChild(group);
  }

  function render() {
    const width = contentWidth(container);
    const focusedCp = overlay.contains(document.activeElement) ? document.activeElement.getAttribute('data-cp') : null;
    clear(svgEl);
    clear(overlay);
    clear(dots);
    const { c, points } = model;
    wrapper.setAttribute('aria-label', joinParts([c.title, t('a11y.chart_keys')]));
    svgEl.append(svg('title', { id: ids.title }, c.title || ''), svg('desc', { id: ids.desc }, describeSeries(model.vm)));
    if (!width || !points.length) {
      geo = null;
      showReadout(-1);
      return;
    }
    const height = width >= WIDE_PX ? HEIGHT_WIDE : HEIGHT_NARROW;
    for (const el of [svgEl, overlay]) {
      el.setAttribute('width', String(width));
      el.setAttribute('height', String(height));
      el.setAttribute('viewBox', `0 0 ${width} ${height}`);
    }
    const g = layout(width);
    geo = g;
    drawFrame(g);
    drawData(g);
    drawChangepoints(g);
    drawNow(g);
    drawAnomalies(g);
    if (model.last) {
      svgEl.appendChild(svg('circle', { class: 'chart-last', cx: g.seamX, cy: g.y.map(model.last.value), r: 4 }));
    }
    svgEl.appendChild(xhLayer);                                          // crosshair on top
    crosshair.setPoints(points.map((p) => ({ ms: p.ms, x: g.x.map(p.ms) })),
      { plotTop: g.T, plotHeight: g.B - g.T, startIndex: model.lastIndex });
    setActiveCp(activeCp);
    const k = crosshair.current();
    showReadout(k);
    drawDots(k);
    if (focusedCp !== null) {
      const again = overlay.querySelector(`[data-cp="${focusedCp}"]`);
      if (again) again.focus({ preventScroll: true });
    }
  }

  applyToggles();
  render();
  const stopObserving = observeWidth(container, () => render());

  return {
    update(vm, opts2 = {}) {
      const keep = crosshair.current() >= 0 ? model.points[crosshair.current()]?.iso : null;
      if (vm) model = buildModel(vm);
      Object.assign(o, opts2);
      applyToggles();
      crosshair.hide();
      render();
      if (keep && model.indexByDate.has(keep)) crosshair.moveTo(model.indexByDate.get(keep));
    },
    setBand(band) {
      o.band = band;
      applyToggles();
    },
    setToggles(toggles = {}) {
      if (toggles.trend !== undefined) o.trend = toggles.trend;
      if (toggles.anomalies !== undefined) o.anomalies = toggles.anomalies;
      applyToggles();
    },
    highlightDate(iso) {
      const k = model.indexByDate.get(iso);
      if (k === undefined || !geo) return false;
      crosshair.moveTo(k);
      return true;
    },
    highlightChangepoint(n) {
      setActiveCp(n);
      const cp = n == null ? null : model.cps.find((c) => c.n === n);
      if (cp && cp.index >= 0) crosshair.moveTo(cp.index);
    },
    onChangepoint(cb) {
      cpListeners.push(cb);
    },
    onPointFocus(cb) {
      pointListeners.push(cb);
    },
    getSvg() {
      return svgEl;
    },
    destroy() {
      stopObserving();
      announce.cancel();
      crosshair.destroy();
      cpListeners.length = 0;
      pointListeners.length = 0;
      clear(container);
    },
  };
}
