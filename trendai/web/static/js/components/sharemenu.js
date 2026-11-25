// Share and export controls (spec F.6.12, F.7.6, C.10, D.5). Owner: WP3-app.
//
// 공유 — public sources (naver / wiki / sample): copy the page URL with an absolute period (`range` replaced by
// `{query.period.start}~{query.period.end}`); uploads: the share-link dialog. 내보내기 ▾ — CSV, 엑셀 (hidden when the
// export is unavailable), 이미지(PNG), HTML 리포트 (disabled with the export_naver_raw text for Naver), JSON, and
// 요약 복사 (summary_line + ' — ' + the share URL or the absolute-period page URL). Downloads are plain
// <a href download> links to the server exports with `?hl=`; the browser never builds a data file.

import { h } from '../dom.js';
import { getLang, t } from '../i18n.js';
import { serialize, withState } from '../state.js';
import * as store from '../store.js';
import { absoluteUrl, copyText, exportHref, icon } from '../ui.js';
import { exportPng, pngFilename } from '../chart/png.js';
import { menuButton } from './popover.js';
import { openShareDialog } from './sharedialog.js';
import { toast } from './toast.js';

/** True when the result comes from an uploaded file (share links instead of page URLs, C.8). */
export function isUploadResult(env) {
  return Boolean(env && env.query && env.query.source === 'upload');
}

/** The current page URL with the result's absolute period (public sources, F.6.12); `hl` is dropped. */
export function absolutePageUrl(env, state) {
  if (!state) return window.location.href;
  const period = env && env.query && env.query.period;
  const patch = { hl: null };
  if (period && period.start && period.end && state.route === 'explore') patch.range = `${period.start}~${period.end}`;
  return absoluteUrl(serialize(withState(state, patch)));
}

/** The URL that travels with a copied summary, or null when there is none to give (an upload without a share). */
export function summaryUrl(env, state) {
  if (env && env.readonly) return window.location.origin + window.location.pathname;
  if (isUploadResult(env)) {
    const share = env.result_id ? store.getShare(env.result_id) : null;
    return share ? absoluteUrl(share.url) : null;
  }
  return absolutePageUrl(env, state);
}

async function copyLink(env, state) {
  const ok = await copyText(absolutePageUrl(env, state));
  toast(ok ? t('share.copied') : t('err.clipboard'), { kind: ok ? 'info' : 'error' });
}

/** Run the share action for a result (copy link, or the share dialog for uploads). */
export function shareAction(env, ctx, opener) {
  if (isUploadResult(env)) {
    if (env.result_id) openShareDialog({ resultId: env.result_id, opener, navigate: ctx.navigate });
    return;
  }
  copyLink(env, ctx.state);
}

/**
 * The 공유 button, or null on read-only shared pages (no share button there, F.6.11).
 * @param {Object} env  C.7 / C.9 envelope (may be null while the job runs)
 * @param {Object} ctx  page context ({state, navigate})
 */
export function shareButton(env, ctx) {
  if (env && env.readonly) return null;
  return h('button', {
    type: 'button', class: 'btn btn-secondary', disabled: env ? null : true,
    on: { click: (e) => env && shareAction(env, ctx, e.currentTarget) },
  }, icon('ui-share'), t('detail.share'));
}

/** Make the PNG of a chart (F.7.6) with the saving / failure messages. */
export async function savePng(chart, { title, footer, name }) {
  const svgEl = chart && chart.getSvg ? chart.getSvg() : null;
  if (!svgEl) {
    toast(t('err.png_failed'), { kind: 'error' });
    return;
  }
  toast(t('export.png.saving'));
  try {
    await exportPng(svgEl, { title, footer, filename: pngFilename(name) });
  } catch {
    toast(t('err.png_failed'), { kind: 'error' });
  }
}

/**
 * The 내보내기 ▾ menu.
 * @param {{exports: Object|null, png?: {getChart: () => Object, title: string, footer: string, name: string},
 *          summary?: {line: string, url: () => string|null}}} opts
 */
export function exportMenu(opts) {
  const lang = getLang();
  const ex = opts.exports;
  const un = (ex && ex.unavailable) || {};
  const link = (fmt) => (ex && ex[fmt] ? exportHref(ex[fmt], lang) : null);
  const button = menuButton({
    label: [h('span', null, t('detail.export')), icon('ui-chevron-down', { cls: 'chev' })],
    icon: 'ui-download',
    className: 'btn-secondary',
    items: () => [
      { label: t('export.csv'), href: link('csv'), download: true, hidden: !link('csv') },
      { label: t('export.xlsx'), href: link('xlsx'), download: true, hidden: !link('xlsx') },
      opts.png ? {
        label: t('export.png'),
        onSelect: () => savePng(opts.png.getChart(), opts.png),
      } : null,
      link('html') ? { label: t('export.html'), href: link('html'), download: true }
        : { label: t('export.html'), disabled: true, hint: un.html ? un.html.text : null },
      { label: t('export.json'), href: link('json'), download: true, hidden: !link('json') },
      opts.summary ? {
        label: t('share.summary'),
        onSelect: async () => {
          const url = opts.summary.url();
          const ok = await copyText(url ? `${opts.summary.line} — ${url}` : opts.summary.line);
          toast(ok ? t('share.summary.copied') : t('err.clipboard'), { kind: ok ? 'info' : 'error' });
        },
      } : null,
    ],
  });
  if (!ex) button.disabled = true;
  return button;
}

/**
 * The compare card's ⋯ menu (F.6.8): CSV · 이미지 저장 · 링크 복사.
 * @param {{env: Object, ctx: Object, png: Object}} opts
 */
export function compareMoreMenu(opts) {
  const lang = getLang();
  const csv = opts.env && opts.env.exports && opts.env.exports.csv;
  return menuButton({
    label: [], icon: 'ui-more', ariaLabel: t('common.more'), className: 'icon-btn btn-ghost',
    items: () => [
      { label: t('export.csv'), href: csv ? exportHref(csv, lang) : null, download: true, hidden: !csv },
      { label: t('export.png'), onSelect: () => savePng(opts.png.getChart(), opts.png) },
      opts.env && !opts.env.readonly ? {
        label: t('share.copy_link'), onSelect: (opener) => shareAction(opts.env, opts.ctx, opener),
      } : null,
    ],
  });
}
