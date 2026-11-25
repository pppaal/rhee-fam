// Share-link dialog for upload results (spec F.6.12, C.8, M20): POST /api/share (or reuse the share stored for this
// result in trendai.shares), show the absolute /s/… URL with a copy button and its expiry, and `링크 삭제` (DELETE with
// the stored token, after a confirm step). Owner: WP3-app.

import * as api from '../api.js';
import { h } from '../dom.js';
import { datetimeLabel } from '../fmt.js';
import { getLang, t } from '../i18n.js';
import * as store from '../store.js';
import { absoluteUrl, copyText, icon } from '../ui.js';
import { confirmDialog, openSheet } from './sheet.js';
import { errorCard } from './errorcard.js';
import { toast } from './toast.js';

/** The stored share of `resultId`, or a new one (POST /api/share, then remembered). */
export async function ensureShare(resultId) {
  const known = store.getShare(resultId);
  if (known) return known;
  const created = await api.share(resultId);
  store.saveShare(resultId, created);
  return created;
}

/**
 * Open the share dialog.
 * @param {{resultId: string, opener?: HTMLElement, navigate: Function}} opts
 */
export async function openShareDialog(opts) {
  let info;
  try {
    info = await ensureShare(opts.resultId);
  } catch (err) {
    const sheet = openSheet({
      title: t('share.dialog.title'), opener: opts.opener, className: 'sheet-small',
      content: errorCard(err, { navigate: opts.navigate, retry: () => { sheet.close(); openShareDialog(opts); } }, { compact: true }),
    });
    return;
  }
  const url = absoluteUrl(info.url);
  const field = h('input', {
    id: 'share-url', type: 'text', class: 'input share-url', readonly: true, props: { value: url },
    on: { focus: (e) => e.target.select() },
  });
  const copy = h('button', {
    type: 'button', class: 'btn btn-primary',
    on: {
      click: async () => {
        const ok = await copyText(url);
        toast(ok ? t('share.copied') : t('err.clipboard'), { kind: ok ? 'info' : 'error' });
      },
    },
  }, icon('ui-copy'), t('share.copy_link'));
  const del = h('button', {
    type: 'button', class: 'btn btn-danger-quiet',
    on: {
      click: async (e) => {
        const yes = await confirmDialog({
          title: t('share.delete'), body: t('share.delete.confirm'), confirm: t('share.delete'), danger: true,
          opener: e.currentTarget,
        });
        if (!yes) return;
        try {
          await api.deleteShare(info.share_id, info.delete_token);
        } catch (err) {
          if (!(err && err.code === 'share_not_found')) {
            toast(err && typeof err.text === 'function' ? err.text() : t('err.client_network'), { kind: 'error' });
            return;
          }
        }
        store.forgetShare(opts.resultId);
        sheet.close();
        toast(t('share.deleted'));
      },
    },
  }, icon('ui-trash'), t('share.delete'));
  const sheet = openSheet({
    title: t('share.dialog.title'), opener: opts.opener, className: 'sheet-small',
    content: [
      h('p', null, t('share.dialog.body', { date: datetimeLabel(info.expires_at, getLang()) })),
      h('label', { class: 'sr-only', for: 'share-url' }, t('share.dialog.title')),
      h('div', { class: 'share-row' }, field, copy),
    ],
    footer: [del],
    initialFocus: copy,
  });
}
