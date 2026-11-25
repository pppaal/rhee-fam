// PNG export of a visible chart (spec F.7.6). Owner: WP3-chart. Called by components/sharemenu.js and the compare
// card menu (WP3-app), which show t('export.png.saving') while the promise runs and t('err.png_failed') on rejection.
// No network access: fonts are the system stack, so the canvas is never tainted.

import { h } from '../dom.js';

/** Font stack used inside the exported image (web fonts are not available to an SVG drawn as an <img>). */
export const SYSTEM_FONT = '-apple-system, BlinkMacSystemFont, "Apple SD Gothic Neo", "Malgun Gothic", '
  + '"Noto Sans KR", "Segoe UI", system-ui, sans-serif';

const SVG_NS = 'http://www.w3.org/2000/svg';
const PAD = 24;
const TITLE_PX = 16;
const TITLE_LH = 22;
const FOOT_PX = 12;
const FOOT_LH = 18;
const GAP = 12;
const CANVAS_2D = '2d';
const WATERMARK = 'trendai';

// Presentation properties copied from the live nodes' computed style (external CSS does not reach the image).
const PAINT = ['fill', 'fill-opacity', 'stroke', 'stroke-opacity', 'stroke-width', 'stroke-dasharray',
  'stroke-linecap', 'stroke-linejoin', 'opacity', 'paint-order', 'shape-rendering', 'visibility'];
const TEXT = ['font-size', 'font-weight'];

/**
 * @typedef {Object} PngOptions
 * @property {string} title     drawn above the chart (series name, or t('compare.title') for a comparison)
 * @property {string} footer    the source line, wrapped: series VM `source.line` (detail) or `compare.source_line`
 *                              (comparison) — so filters always travel with the image
 * @property {number} [scale=2] device-pixel scale of the canvas
 */

/** Clone `svgEl` with every computed colour/stroke/font inlined as presentation attributes. */
function standaloneSvg(svgEl) {
  const clone = svgEl.cloneNode(true);
  const live = [svgEl, ...svgEl.querySelectorAll('*')];
  const copies = [clone, ...clone.querySelectorAll('*')];
  const drop = [];
  live.forEach((el, i) => {
    const copy = copies[i];
    const cs = getComputedStyle(el);
    if (cs.display === 'none' || el.classList.contains('chart-noexport') || el.classList.contains('chart-measure')) {
      drop.push(copy);
      return;
    }
    for (const prop of PAINT) {
      const v = cs.getPropertyValue(prop);
      if (v) copy.setAttribute(prop, prop === 'stroke-dasharray' || prop === 'stroke-width' ? v.replace(/px/g, '') : v);
    }
    if (el.localName === 'text' || el.localName === 'tspan') {
      for (const prop of TEXT) copy.setAttribute(prop, cs.getPropertyValue(prop));
      copy.setAttribute('font-family', SYSTEM_FONT);
    }
    copy.removeAttribute('class');
    copy.removeAttribute('style');
    copy.removeAttribute('tabindex');
    copy.removeAttribute('role');
  });
  for (const node of drop) node.remove();
  clone.setAttribute('xmlns', SVG_NS);
  return clone;
}

/** Split `text` into lines no wider than `max` (words first; a word longer than a line is cut by characters). */
function wrapLines(ctx, text, max) {
  const lines = [];
  let line = '';
  const push = (word) => {
    const next = line ? `${line} ${word}` : word;
    if (ctx.measureText(next).width <= max) {
      line = next;
      return;
    }
    if (line) lines.push(line);
    line = '';
    if (ctx.measureText(word).width <= max) {
      line = word;
      return;
    }
    let part = '';
    for (const ch of Array.from(word)) {
      if (ctx.measureText(part + ch).width > max && part) {
        lines.push(part);
        part = '';
      }
      part += ch;
    }
    line = part;
  };
  for (const word of String(text || '').split(/\s+/).filter(Boolean)) push(word);
  if (line) lines.push(line);
  return lines;
}

function loadImage(url) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error('png: the chart image could not be decoded'));
    img.src = url;
  });
}

/**
 * Render a chart SVG to a PNG blob: clone `svgEl`, inline every CSS-variable colour as presentation attributes
 * (fill, stroke, opacity, from getComputedStyle of the live nodes — so the current theme is used), set the system
 * font stack, serialise → image/svg+xml Blob → object URL → Image → canvas at `scale`× with a --surface background,
 * 24 px padding, `title` above, wrapped `footer` and the watermark 'trendai' below → canvas.toBlob('image/png').
 * @param {SVGSVGElement} svgEl  FanChart.getSvg() / CompareChart.getSvg()
 * @param {PngOptions} opts
 * @returns {Promise<Blob>}  rejects with an Error when any step fails
 */
export async function renderPng(svgEl, opts) {
  if (!svgEl || !svgEl.isConnected) throw new Error('png: the chart is not on the page');
  const width = Number(svgEl.getAttribute('width')) || svgEl.getBoundingClientRect().width;
  const height = Number(svgEl.getAttribute('height')) || svgEl.getBoundingClientRect().height;
  if (!width || !height) throw new Error('png: the chart has no size');
  const scale = opts && opts.scale > 0 ? opts.scale : 2;

  const theme = getComputedStyle(svgEl);
  const colour = (name, fallback) => theme.getPropertyValue(name).trim() || fallback;
  const surface = colour('--surface', '#ffffff');
  const ink = colour('--ink', '#141413');
  const ink3 = colour('--ink-3', '#6b6963');

  const markup = new XMLSerializer().serializeToString(standaloneSvg(svgEl));
  const url = URL.createObjectURL(new Blob([markup], { type: 'image/svg+xml;charset=utf-8' }));
  try {
    const img = await loadImage(url);
    const canvas = h('canvas');
    const ctx = canvas.getContext(CANVAS_2D);
    if (!ctx) throw new Error('png: canvas is not available');
    const titleFont = `600 ${TITLE_PX}px ${SYSTEM_FONT}`;
    const footFont = `400 ${FOOT_PX}px ${SYSTEM_FONT}`;
    ctx.font = titleFont;
    const titleLines = wrapLines(ctx, opts && opts.title, width);
    ctx.font = footFont;
    const footLines = wrapLines(ctx, opts && opts.footer, width);

    const titleH = titleLines.length ? titleLines.length * TITLE_LH + GAP : 0;
    const footH = (footLines.length ? footLines.length * FOOT_LH : 0) + FOOT_LH + 4;
    const cssW = width + PAD * 2;
    const cssH = PAD + titleH + height + GAP + footH + PAD;
    canvas.width = Math.round(cssW * scale);
    canvas.height = Math.round(cssH * scale);
    ctx.scale(scale, scale);
    ctx.fillStyle = surface;
    ctx.fillRect(0, 0, cssW, cssH);

    ctx.textBaseline = 'top';
    let y = PAD;
    ctx.font = titleFont;
    ctx.fillStyle = ink;
    for (const line of titleLines) {
      ctx.fillText(line, PAD, y + (TITLE_LH - TITLE_PX) / 2);
      y += TITLE_LH;
    }
    if (titleLines.length) y += GAP;
    ctx.drawImage(img, PAD, y, width, height);
    y += height + GAP;
    ctx.font = footFont;
    ctx.fillStyle = ink3;
    for (const line of footLines) {
      ctx.fillText(line, PAD, y + (FOOT_LH - FOOT_PX) / 2);
      y += FOOT_LH;
    }
    ctx.font = `700 ${FOOT_PX}px ${SYSTEM_FONT}`;
    ctx.textAlign = 'right';
    ctx.fillText(WATERMARK, PAD + width, y + 4 + (FOOT_LH - FOOT_PX) / 2);

    return await new Promise((resolve, reject) => {
      canvas.toBlob((blob) => (blob ? resolve(blob) : reject(new Error('png: encoding failed'))), 'image/png');
    });
  } finally {
    URL.revokeObjectURL(url);
  }
}

/**
 * renderPng() and download the result as `filename` (via a temporary <a download> and an object URL that is
 * revoked afterwards). Resolves with the Blob.
 * @param {SVGSVGElement} svgEl
 * @param {PngOptions & {filename: string}} opts  filename from pngFilename()
 * @returns {Promise<Blob>}
 */
export async function exportPng(svgEl, opts) {
  const blob = await renderPng(svgEl, opts);
  const url = URL.createObjectURL(blob);
  const link = h('a', { href: url, download: opts.filename || pngFilename('chart'), hidden: true });
  document.body.appendChild(link);
  try {
    link.click();
  } finally {
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);   // the download has started by then
  }
  return blob;
}

/**
 * 'trendai-{name}-{YYYYMMDD}.png' with the KST date of `now`; characters not allowed in file names
 * (/\:*?"<>| and control characters) become '_'.
 * @param {string} name  series display name, or the compared names joined with ' vs '
 * @param {Date} [now=new Date()]
 * @returns {string}
 */
export function pngFilename(name, now = new Date()) {
  const kst = new Date(now.getTime() + 9 * 3600000);
  const ymd = `${kst.getUTCFullYear()}${String(kst.getUTCMonth() + 1).padStart(2, '0')}`
    + `${String(kst.getUTCDate()).padStart(2, '0')}`;
  // eslint-disable-next-line no-control-regex
  const safe = String(name == null ? '' : name).replace(/[\\/:*?"<>|\x00-\x1f\x7f]/g, '_').trim() || 'chart';
  return `trendai-${safe}-${ymd}.png`;
}
