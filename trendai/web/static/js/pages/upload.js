// Upload page (spec F.6.10, C.4, brief §3.6). Owner: WP3-app.
//
// Step 1: drop zone with a keyboard-accessible 파일 선택 button, 엑셀에서 붙여넣기 (dialog), privacy note, the
// supported-formats block (#formats) and the Google Trends guide. Step 2 ("이렇게 읽었어요"): encoding sentence,
// date-column select (detected column selected; a combination shown as "연도 + 월"; wide tables explained), value
// checkboxes (first N detected checked, excluded ones with their reason), frequency, forecast horizon (FU-9), optional
// unit label, preview, notices, the `problem` banner, 분석하기 (POST /api/analyze → /r/<job_id>) and 다시 올리기.
// A file or pasted text handed over from home (setPendingUpload) is read right away; nothing is uploaded before the
// user picks something. The limits (size, value columns) are the server's (config.sources.upload, OPS-12).
//
// `다시 분석` on an upload result comes back here with the same upload and the choices made (reconfirmUpload), while
// the server still holds the parsed table (its expires_at); else the page starts at step 1 (FU-9).
//
// Errors are announced and focused (A11Y-3): reading a file replaces the button the user pressed, so focus moves to
// the reading line and then to the error card or the confirm heading.

import * as api from '../api.js';
import { clear, h } from '../dom.js';
import { getLang, t } from '../i18n.js';
import { announce, formatBytes, icon } from '../ui.js';
import { dropZone } from '../components/dropzone.js';
import { errorCard, fixRow, localFixes } from '../components/errorcard.js';
import { gtGuide } from '../components/gtguide.js';
import { noticeList } from '../components/notices.js';
import { openSheet } from '../components/sheet.js';
import { spanLabel } from '../fmt.js';

const DEFAULT_MAX_COLS = 20;
const DEFAULT_MAX_MB = 10;
const NUMERIC_SHARE = 0.8;          // preview columns at least this numeric are right-aligned (UX-25)
let pending = null;
/** job id → {res, choices}: the upload behind a result, so 다시 분석 can return to the confirm step (FU-9). */
const submitted = new Map();
const SUBMITTED_MAX = 10;

/** Hand a File or pasted text to the upload page (home's upload card); read once the page mounts. */
export function setPendingUpload(input) {
  pending = { kind: 'input', input };
}

/** True while the upload behind result `jobId` can be analysed again without uploading it again. */
export function canReconfirm(jobId) {
  const entry = submitted.get(jobId);
  if (!entry) return false;
  const exp = Date.parse(entry.res.expires_at || '');
  return !Number.isFinite(exp) || exp > Date.now();
}

/** Open the upload page at the confirm step of the upload behind `jobId` (call before navigating to /upload). */
export function reconfirmUpload(jobId) {
  if (canReconfirm(jobId)) pending = { kind: 'reconfirm', entry: submitted.get(jobId) };
}

/** The upload limits of /api/config (OPS-12), with the defaults as fallback. */
export function uploadLimits(config) {
  const u = (config && config.sources && config.sources.upload) || {};
  const mb = Number.isFinite(u.max_mb) && u.max_mb > 0 ? u.max_mb : DEFAULT_MAX_MB;
  const cols = Number.isInteger(u.max_series) && u.max_series > 0 ? u.max_series : DEFAULT_MAX_COLS;
  return { mb, cols, bytes: mb * 1024 * 1024 };
}

/**
 * The preview rows to show: the head, a gap and the tail — or, when the file is so short that head and tail overlap,
 * every row once and no gap (UX-25). Pure (exported for tests).
 * @param {{head?: Array, tail?: Array}} preview
 * @param {number|null} nRows  rows in the table (detected.n_rows)
 * @returns {{rows: Array, gapAt: number|null}}  gapAt: index in `rows` before which the ⋯ row goes
 */
export function previewRows(preview, nRows) {
  const head = (preview && preview.head) || [];
  const tail = (preview && preview.tail) || [];
  if (!tail.length || !head.length) return { rows: [...head, ...tail], gapAt: null };
  if (Number.isInteger(nRows) && nRows <= head.length + tail.length) {
    const overlap = head.length + tail.length - nRows;
    return { rows: [...head, ...tail.slice(overlap)], gapAt: null };
  }
  return { rows: [...head, ...tail], gapAt: head.length };
}

/**
 * The paste dialog: a textarea and 읽기.
 * @param {{opener?: HTMLElement, onRead: (text: string) => void}} opts
 */
export function openPasteDialog(opts) {
  const area = h('textarea', { id: 'paste-area', class: 'input textarea', rows: '10', spellcheck: 'false' });
  const read = h('button', {
    type: 'button', class: 'btn btn-primary',
    on: {
      click: () => {
        const text = area.value;
        if (!text.trim()) {
          area.focus();
          return;
        }
        sheet.close(false);
        opts.onRead(text);
      },
    },
  }, t('upload.paste.read'));
  const sheet = openSheet({
    title: t('upload.paste'),
    content: [h('label', { class: 'field-label', for: 'paste-area' }, t('upload.paste.label')), area],
    footer: [h('button', { type: 'button', class: 'btn', on: { click: () => sheet.close() } }, t('common.cancel')), read],
    opener: opts.opener, initialFocus: area, className: 'sheet-wide',
  });
  return sheet;
}

function encodingKey(enc) {
  if (!enc) return null;
  const e = String(enc).toLowerCase();
  if (e === 'utf-8' || e === 'utf-8-sig' || e === 'utf8') return ['confirm.encoding.utf8', null];
  if (e === 'cp949' || e === 'euc-kr' || e === 'euckr') return ['confirm.encoding.cp949', null];
  if (e.startsWith('utf-16') || e.startsWith('utf16')) return ['confirm.encoding.utf16', null];
  return ['confirm.encoding.other', { value: enc }];
}

/** '형식: YYYY-MM (월)' — without repeating the word when the server label already says 형식 / format (UX-25). */
function formatLine(label) {
  return /형식|format/i.test(label) ? label : t('confirm.date_format', { value: label });
}

export function create(ctx) {
  ctx.setTitle(`${t('title.upload')} · trendai`);
  const limits = uploadLimits(ctx.config);
  const body = h('div', { class: 'upload-body' });
  const page = h('div', { class: 'page upload-page' },
    h('div', { class: 'page-head' }, h('h1', { class: 'page-title' }, t('upload.title'))),
    body);
  let lastInput = null;
  let controller = null;
  let destroyed = false;

  // ------------------------------------------------------------------------------------------------ step 1
  function pickStep() {
    clear(body);
    body.append(...pickNodes());
  }

  /** The step-1 nodes: drop zone, privacy note, formats block and Google Trends guide. */
  function pickNodes() {
    const drop = dropZone({
      inputId: 'upload-file', primary: true,
      text: [h('p', { class: 'dropzone-title' }, t('upload.drop')), h('p', { class: 'muted' }, t('upload.or'))],
      chooseLabel: t('upload.choose'), pasteLabel: t('upload.paste'),
      onFile: (file) => read(file), onPaste: (opener) => paste(opener),
      after: h('p', { class: 'muted small' }, t('upload.formats', { mb: limits.mb, n: limits.cols })),
    });
    return [
      h('div', { class: 'upload-grid' },
        h('div', { class: 'upload-main' },
          drop,
          h('p', { class: 'privacy-note' }, icon('ui-lock'), h('span', null, t('upload.privacy')))),
        h('div', { class: 'upload-side' },
          h('section', { class: 'card', id: 'formats', 'aria-labelledby': 'formats-title', tabindex: '-1' },
            h('h2', { class: 'card-title', id: 'formats-title' }, t('upload.formats.title')),
            h('p', null, t('upload.formats.body'))),
          gtGuide())),
    ];
  }

  function paste(opener) {
    openPasteDialog({ opener, onRead: (text) => read(text) });
  }

  // ------------------------------------------------------------------------------------------------ reading
  async function read(input) {
    lastInput = input;
    if (controller) controller.abort();
    controller = new AbortController();
    clear(body);
    const status = h('div', { class: 'reading', role: 'status', tabindex: '-1' },
      h('span', { class: 'spinner', 'aria-hidden': 'true' }), t('upload.reading'));
    body.appendChild(status);
    status.focus({ preventScroll: true });          // the pressed button is gone: keep focus on the page
    try {
      const res = await api.upload(input, { hl: getLang(), signal: controller.signal, maxBytes: limits.bytes });
      if (destroyed) return;
      confirmStep(res);
    } catch (err) {
      if ((err && err.name === 'AbortError') || destroyed) return;
      clear(body);
      const card = errorCard(err, fixContext(), { focusable: true });
      body.append(card, ...pickNodes());
      announce(err && typeof err.text === 'function' ? err.text() : t('err.client_network'), true);
      card.focus({ preventScroll: false });
    }
  }

  function fixContext() {
    return {
      navigate: ctx.navigate,
      state: ctx.state,
      retry: () => (lastInput ? read(lastInput) : pickStep()),
      openPaste: () => paste(null),
    };
  }

  // ------------------------------------------------------------------------------------------------ step 2
  /**
   * @param {Object} res  the C.4 upload response
   * @param {{date_col?: string|null, value_cols?: string[], freq?: string, h?: string|number,
   *          unit_label?: string|null}} [prev]  the choices of an earlier 분석하기 (다시 분석)
   */
  function confirmStep(res, prev = null) {
    clear(body);
    const detected = res.detected || {};
    const lang = getLang();
    const maxCols = limits.cols;
    const valueCols = res.columns.filter((c) => c.role === 'value');
    const otherCols = res.columns.filter((c) => c.role === 'excluded' || c.role === 'ignored');
    const firstChecked = new Set(prev && prev.value_cols ? prev.value_cols : (detected.value_cols || []).slice(0, maxCols));
    let dateChanged = Boolean(prev && prev.date_col);

    // date column
    const dateCols = detected.date_cols || [];
    const wantedDate = prev && prev.date_col ? prev.date_col : null;
    const dateSelect = h('select', { id: 'confirm-date', class: 'input select', on: { change: () => { dateChanged = true; syncValues(); } } });
    if (!detected.wide) {
      if (dateCols.length > 1) {
        dateSelect.appendChild(h('option', { value: dateCols[0], props: { selected: !wantedDate } }, dateCols.join(' + ')));
      }
      for (const c of res.columns) {
        if (dateCols.length > 1 && c.name === dateCols[0]) continue;
        const selected = wantedDate ? c.name === wantedDate : dateCols.length === 1 && c.name === detected.date_col;
        dateSelect.appendChild(h('option', { value: c.name, props: { selected } }, c.name));
      }
      if (!detected.date_col && !wantedDate) dateSelect.prepend(h('option', { value: '', props: { selected: true } }, '—'));
    }
    const dateField = detected.wide
      ? h('div', { class: 'field' }, h('p', { class: 'field-label' }, t('confirm.date_col')), h('p', null, t('confirm.wide')))
      : h('div', { class: 'field' },
        h('label', { class: 'field-label', for: 'confirm-date' }, t('confirm.date_col')),
        dateSelect,
        detected.date_format_label ? h('p', { class: 'field-help' }, formatLine(detected.date_format_label)) : null);

    // value columns
    const maxNote = h('p', { class: 'field-help', 'aria-live': 'polite' });
    const needNote = h('p', { class: 'field-error', 'aria-live': 'polite' });
    const boxes = [];
    const valueItem = (c, checked, reasonKey) => {
      const box = h('input', {
        type: 'checkbox', value: c.name, props: { checked }, dataset: { reason: reasonKey ? c.reason : null },
        on: { change: syncValues },
      });
      boxes.push(box);
      return h('li', null, h('label', { class: ['check', reasonKey && 'is-excluded'] }, box,
        h('span', { class: 'check-name' }, c.name),
        reasonKey ? h('span', { class: 'check-reason' }, t(reasonKey)) : null));
    };
    const valueList = h('ul', { class: 'check-list' },
      valueCols.map((c) => valueItem(c, firstChecked.has(c.name), null)),
      otherCols.map((c) => valueItem(c, Boolean(prev) && firstChecked.has(c.name) && c.reason !== 'not_numeric',
        c.reason === 'row_number' ? 'confirm.excluded.row_number' : 'confirm.excluded.not_numeric')));

    // frequency, horizon, unit label
    const freqSelect = h('select', { id: 'confirm-freq', class: 'input select', on: { change: () => fillHorizon() } },
      ['auto', 'D', 'W', 'M', 'Q', 'Y'].map((f) => h('option', { value: f, props: { selected: f === ((prev && prev.freq) || 'auto') } }, t(`freq.${f}`))));
    const hSelect = h('select', { id: 'confirm-h', class: 'input select' });
    let wantedH = prev && prev.h != null ? String(prev.h) : 'auto';
    function fillHorizon() {
      const freq = freqSelect.value === 'auto' ? detected.freq : freqSelect.value;
      const lim = (ctx.config && ctx.config.limits) || {};
      const choices = (freq && lim.horizon_choices && lim.horizon_choices[freq]) || [];
      const def = freq && lim.horizon_default ? lim.horizon_default[freq] : null;
      const current = hSelect.value || wantedH;
      clear(hSelect);
      hSelect.appendChild(h('option', { value: 'auto' }, def ? t('h.auto', { value: spanLabel(def, freq, lang) }) : t('unit.auto')));
      for (const n of choices) hSelect.appendChild(h('option', { value: String(n) }, spanLabel(n, freq, lang)));
      hSelect.value = [...hSelect.options].some((o) => o.value === current) ? current : 'auto';
      wantedH = hSelect.value;
    }
    fillHorizon();
    const unitInput = h('input', {
      id: 'confirm-unit', type: 'text', class: 'input', maxlength: '12', placeholder: t('confirm.unit_placeholder'),
      autocomplete: 'off', props: { value: (prev && prev.unit_label) || '' },
    });

    const analyzeBtn = h('button', { type: 'button', class: 'btn btn-primary', on: { click: () => submit() } }, t('confirm.analyze'));
    const againBtn = h('button', { type: 'button', class: 'btn btn-secondary', on: { click: () => { pickStep(); focusChoose(); } } }, t('confirm.reupload'));
    const submitError = h('div', { class: 'submit-error' });

    function selectedValues() {
      return boxes.filter((b) => b.checked).map((b) => b.value);
    }
    function syncValues() {
      const dateName = detected.wide ? null : dateSelect.value;
      for (const b of boxes) {
        if (b.value === dateName && b.checked) b.checked = false;
      }
      const n = selectedValues().length;
      const full = n >= maxCols;
      for (const b of boxes) {
        b.disabled = b.dataset.reason === 'not_numeric' || (full && !b.checked) || b.value === dateName;
      }
      maxNote.textContent = full ? t('confirm.max_cols', { n: maxCols }) : '';
      needNote.textContent = n === 0 ? t('confirm.need_value') : '';
      analyzeBtn.disabled = n === 0;
    }

    async function submit() {
      const cols = selectedValues();
      if (!cols.length) {
        syncValues();
        return;
      }
      analyzeBtn.disabled = true;
      clear(submitError);
      const hValue = hSelect.value === 'auto' ? 'auto' : Number(hSelect.value);
      const reqBody = {
        source: 'upload', upload_id: res.upload_id,
        date_col: !detected.wide && dateChanged && dateSelect.value ? dateSelect.value : null,
        value_cols: cols, freq: freqSelect.value, h: hValue, unit_label: unitInput.value.trim() || null,
      };
      if (controller) controller.abort();
      controller = new AbortController();
      try {
        const start = await api.startAnalyze(reqBody, { signal: controller.signal });
        if (destroyed) return;                       // the user has left: never yank them back (A11Y-19)
        submitted.set(start.job_id, { res, choices: { ...reqBody, date_col: reqBody.date_col } });
        while (submitted.size > SUBMITTED_MAX) submitted.delete(submitted.keys().next().value);
        ctx.navigate(`/r/${start.job_id}${ctx.state.hl ? `?hl=${ctx.state.hl}` : ''}`);
      } catch (err) {
        if ((err && err.name === 'AbortError') || destroyed) return;
        analyzeBtn.disabled = false;
        const card = errorCard(err, {
          ...fixContext(), retry: () => submit(), focusDateSelect: () => (detected.wide ? boxes[0] : dateSelect).focus(),
        }, { compact: true, focusable: true });
        submitError.appendChild(card);
        announce(err && typeof err.text === 'function' ? err.text() : t('err.client_network'), true);
        card.focus({ preventScroll: false });
      }
    }

    const problem = res.problem ? h('div', { class: 'problem-banner', role: 'alert' },
      icon('ui-warning'),
      h('div', null,
        h('p', null, lang === 'en' ? res.problem.message_en : res.problem.message_ko),
        fixRow(localFixes(res.problem.fixes), {
          navigate: ctx.navigate, focusDateSelect: () => (detected.wide || !dateSelect.options.length ? boxes[0] : dateSelect).focus(),
        }))) : null;

    const enc = encodingKey(res.encoding);
    const shown = previewRows(res.preview, detected.n_rows);
    const preview = previewTable(res.preview, res.columns, shown);
    body.append(h('section', { class: 'confirm', 'aria-labelledby': 'confirm-title' },
      h('div', { class: 'confirm-head' },
        h('h2', { class: 'section-title', id: 'confirm-title' }, t('confirm.title')),
        h('p', { class: 'muted' }, t('upload.file', { name: res.filename, size: formatBytes(res.size_bytes) }),
          enc ? ` · ${t(enc[0], enc[1])}` : '')),
      problem,
      h('div', { class: 'confirm-grid' },
        h('div', { class: 'confirm-controls card' },
          dateField,
          h('fieldset', { class: 'field' },
            h('legend', { class: 'field-label' }, t('confirm.value_cols')),
            valueList, maxNote, needNote),
          h('div', { class: 'field' },
            h('label', { class: 'field-label', for: 'confirm-freq' }, t('confirm.freq')), freqSelect,
            detected.freq ? h('p', { class: 'field-help' }, t('confirm.freq.detected', { value: t(`freq.${detected.freq}`) })) : null),
          h('div', { class: 'field' },
            h('label', { class: 'field-label', for: 'confirm-h' }, t('chip.title.h')), hSelect),
          h('div', { class: 'field' },
            h('label', { class: 'field-label', for: 'confirm-unit' }, t('confirm.unit_label')), unitInput),
          noticeList(res.notices, fixContext()),
          submitError,
          h('div', { class: 'btn-row' }, analyzeBtn, againBtn)),
        h('div', { class: 'confirm-preview card' },
          h('h3', { class: 'card-title' }, shown.gapAt === null ? t('confirm.preview_all', { n: shown.rows.length }) : t('confirm.preview')),
          preview))));
    syncValues();
    const heading = body.querySelector('#confirm-title');
    if (heading) {
      heading.setAttribute('tabindex', '-1');
      heading.focus({ preventScroll: false });
    }
  }

  function focusChoose() {
    const btn = body.querySelector('.dropzone-choose');
    if (btn) btn.focus();
  }

  function previewTable(preview, columns, shown) {
    if (!preview || !preview.columns) return null;
    const share = new Map((columns || []).map((c) => [c.name, c.numeric_share]));
    const numeric = preview.columns.map((name) => (share.get(name) || 0) >= NUMERIC_SHARE);
    const cell = (tag, c, i) => h(tag, { class: numeric[i] ? 'num-cell' : null, scope: tag === 'th' ? 'col' : null }, c);
    const row = (cells) => h('tr', null, cells.map((c, i) => cell('td', c, i)));
    const { rows, gapAt } = shown;
    const gap = () => h('tr', { class: 'gap-row' }, h('td', { colspan: String(preview.columns.length) }, '⋯'));
    const bodyRows = rows.flatMap((r, i) => (i === gapAt ? [gap(), row(r)] : [row(r)]));
    return h('div', { class: 'table-scroll', tabindex: '0', role: 'region', 'aria-label': t('confirm.preview') },
      h('table', { class: 'data-table preview-table' },
        h('thead', null, h('tr', null, preview.columns.map((c, i) => cell('th', c, i)))),
        h('tbody', null, bodyRows)));
  }

  // ------------------------------------------------------------------------------------------------ start
  const start = pending;
  pending = null;
  if (start && start.kind === 'reconfirm') {
    confirmStep(start.entry.res, start.entry.choices);
  } else if (start && start.kind === 'input') {
    read(start.input);
  } else {
    pickStep();
    if (window.location.hash === '#paste') requestAnimationFrame(() => paste(null));
  }

  return {
    el: page,
    update(next) {
      return next.state.route === 'upload' && next.state.hl === ctx.state.hl && next.info.type !== 'push';
    },
    destroy() {
      destroyed = true;
      if (controller) controller.abort();
    },
  };
}
