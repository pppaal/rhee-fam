// API client (spec F.3, C). Owner: WP3-app. The only module that calls fetch() (F.1 rule 4).
//
// request() returns parsed JSON or throws ApiError (C.12.1 envelope → {code, status, message, fixes, retryAfter,
// params}). analyze()/pollJob() implement the C.6 polling contract: `since` = 1 + the highest index seen in `series`
// or `failed`, VMs de-duplicated by index, poll interval = poll_after_ms clamped to 300–2,000 ms, 429 → wait
// Retry-After, `done` → fetch result_url once (the authoritative envelope), `failed` → ApiError, `cancelled` →
// {cancelled: true}. Aborting the signal stops polling and sends DELETE /api/jobs/<id> (keepalive) unless the job is
// already terminal; a joined job keeps running for its other subscribers (C.6).

import { getLang, t } from './i18n.js';
import { toAnalyzeBody } from './state.js';

export const UPLOAD_MAX_BYTES = 10 * 1024 * 1024;
const POLL_MIN_MS = 300;
const POLL_MAX_MS = 2000;

/** Client-side fallback labels for fixes of errors the server never saw (network failures). */
const CLIENT_FIX_KEYS = { retry: 'fix.retry', reload: 'fix.reload', go_home: 'fix.go_home' };

/**
 * An API or client-side error. Server texts are kept in both languages so a card re-renders correctly after a
 * language switch (`text()` / `fixList()` read the current language).
 */
export class ApiError extends Error {
  /**
   * @param {{code: string, status?: number, messages?: {ko?: string, en?: string}, messageKey?: string,
   *          params?: Object, fixes?: Array, retryAfter?: number|null, requestId?: string|null}} info
   */
  constructor(info) {
    super('');
    this.name = 'ApiError';
    this.code = info.code;
    this.status = info.status || 0;
    this.messages = info.messages || {};
    this.messageKey = info.messageKey || null;
    this.params = info.params || {};
    this.rawFixes = Array.isArray(info.fixes) ? info.fixes : [];
    this.retryAfter = Number.isFinite(info.retryAfter) ? info.retryAfter : null;
    this.requestId = info.requestId || null;
    this.message = this.text();
  }

  /** The message in `lang` (default: the current UI language). */
  text(lang = getLang()) {
    if (this.messageKey) return t(this.messageKey, this.params, lang);
    return (lang === 'en' ? this.messages.en : this.messages.ko) || this.messages.ko || this.messages.en || this.code;
  }

  /** Up to 3 fixes as `{action, label, params}` in `lang`. */
  fixList(lang = getLang()) {
    return this.rawFixes.slice(0, 3).map((f) => ({
      action: f.action,
      label: (lang === 'en' ? f.label_en : f.label_ko) || f.label_ko || f.label_en
        || (CLIENT_FIX_KEYS[f.action] ? t(CLIENT_FIX_KEYS[f.action], null, lang) : f.action),
      params: f.params || {},
    }));
  }

  /** Build from a C.12.1 error object (the body inside `{"error": …}`, also used in job snapshots / failed[]). */
  static fromObject(obj, fallbackStatus = 0, headerRetry = null) {
    const e = obj || {};
    return new ApiError({
      code: e.code || 'internal_error',
      status: e.status || fallbackStatus,
      messages: { ko: e.message_ko, en: e.message_en },
      params: e.params || {},
      fixes: e.fixes || [],
      retryAfter: Number.isFinite(e.retry_after) ? e.retry_after : headerRetry,
      requestId: e.request_id || null,
    });
  }

  /** A client-side error whose message comes from the UI catalog. */
  static client(code, messageKey, { fixes = [], params = {}, status = 0 } = {}) {
    return new ApiError({ code, status, messageKey, params, fixes: fixes.map((action) => ({ action })) });
  }
}

/** `{cancelled: true}` rejection of analyze()/pollJob() for a cancelled job. */
export const CANCELLED = Object.freeze({ cancelled: true });

function networkError() {
  return ApiError.client('client_network', 'err.client_network', { fixes: ['retry'] });
}

function abortError() {
  const e = new Error('aborted');
  e.name = 'AbortError';
  return e;
}

/** True for the rejection of an aborted request or poll. */
export function isAbort(err) {
  return Boolean(err) && err.name === 'AbortError';
}

function sleep(ms, signal) {
  return new Promise((resolve, reject) => {
    if (signal && signal.aborted) {
      reject(abortError());
      return;
    }
    const timer = setTimeout(() => {
      if (signal) signal.removeEventListener('abort', onAbort);
      resolve();
    }, ms);
    function onAbort() {
      clearTimeout(timer);
      reject(abortError());
    }
    if (signal) signal.addEventListener('abort', onAbort, { once: true });
  });
}

function withHl(path, hl) {
  return `${path}${path.includes('?') ? '&' : '?'}hl=${encodeURIComponent(hl || getLang())}`;
}

/**
 * One API request.
 * @param {string} method
 * @param {string} path  same-origin path, e.g. '/api/config'
 * @param {{json?: any, body?: BodyInit, signal?: AbortSignal, keepalive?: boolean, headers?: Object}} [opts]
 * @returns {Promise<any>} parsed JSON (null for 204)
 */
export async function request(method, path, opts = {}) {
  const headers = { Accept: 'application/json', ...(opts.headers || {}) };
  let body = opts.body;
  if (opts.json !== undefined) {
    headers['Content-Type'] = 'application/json';
    body = JSON.stringify(opts.json);
  }
  let res;
  try {
    res = await fetch(path, {
      method, headers, body, signal: opts.signal, keepalive: Boolean(opts.keepalive), credentials: 'same-origin',
    });
  } catch (err) {
    if (isAbort(err) || (opts.signal && opts.signal.aborted)) throw abortError();
    throw networkError();
  }
  if (res.status === 204) return null;
  let data = null;
  try {
    data = await res.json();
  } catch (err) {
    if (isAbort(err)) throw abortError();
    throw networkError();
  }
  if (!res.ok) {
    const header = Number(res.headers.get('Retry-After'));
    if (data && data.error) throw ApiError.fromObject(data.error, res.status, Number.isFinite(header) ? header : null);
    throw networkError();
  }
  return data;
}

/** DELETE /api/jobs/<id> with keepalive, errors ignored (cancel on leave, C.6). */
export function cancelJob(jobId) {
  request('DELETE', `/api/jobs/${encodeURIComponent(jobId)}`, { keepalive: true }).catch(() => {});
}

/**
 * Poll a job (C.6) until it is terminal.
 * @param {string} jobId
 * @param {{signal?: AbortSignal, onUpdate?: (snap: Object) => void, hl?: () => string, kind?: 'analyze'|'narrative',
 *          initial?: Object, cancelOnAbort?: boolean}} [opts]
 *        onUpdate receives the snapshot with `series` = every VM seen so far (index order) and `failed` (full list).
 *        cancelOnAbort (default true): send DELETE when the signal aborts before the job is terminal.
 * @returns {Promise<Object>} analyze: the C.7 result envelope; narrative: the snapshot's `narrative` object
 */
export async function pollJob(jobId, opts = {}) {
  const { signal, onUpdate } = opts;
  const lang = typeof opts.hl === 'function' ? opts.hl : () => opts.hl || getLang();
  const kind = opts.kind || 'analyze';
  const seen = new Map();
  let failed = [];
  let since = 0;
  let terminal = false;
  const onAbort = () => {
    if (!terminal && opts.cancelOnAbort !== false) cancelJob(jobId);
  };
  if (signal) {
    if (signal.aborted) throw abortError();
    signal.addEventListener('abort', onAbort, { once: true });
  }
  try {
    if (kind === 'analyze' && opts.initial && opts.initial.status === 'done') {
      terminal = true;
      return await request('GET', withHl(opts.initial.result_url || `/api/results/${jobId}`, lang()), { signal });
    }
    for (;;) {
      let snap;
      try {
        snap = await request('GET', `/api/jobs/${encodeURIComponent(jobId)}?since=${since}&hl=${lang()}`, { signal });
      } catch (err) {
        if (err instanceof ApiError && err.status === 429) {
          await sleep(Math.max(1, err.retryAfter || 1) * 1000, signal);
          continue;
        }
        throw err;
      }
      for (const entry of snap.series || []) {
        if (entry && Number.isInteger(entry.index) && entry.vm) seen.set(entry.index, entry.vm);
      }
      if (Array.isArray(snap.failed)) failed = snap.failed;
      const indexes = [...seen.keys(), ...failed.map((f) => f.index)].filter(Number.isInteger);
      since = indexes.length ? Math.max(...indexes) + 1 : 0;
      if (onUpdate) {
        onUpdate({ ...snap, series: [...seen.entries()].sort((a, b) => a[0] - b[0]).map(([, vm]) => vm), failed });
      }
      if (snap.status === 'done') {
        terminal = true;
        if (kind === 'narrative') return snap.narrative;
        return await request('GET', withHl(snap.result_url || `/api/results/${jobId}`, lang()), { signal });
      }
      if (snap.status === 'failed') {
        terminal = true;
        throw ApiError.fromObject(snap.error, 500);
      }
      if (snap.status === 'cancelled') {
        terminal = true;
        throw CANCELLED;
      }
      const wait = Number.isFinite(snap.poll_after_ms) ? snap.poll_after_ms : 500;
      await sleep(Math.min(POLL_MAX_MS, Math.max(POLL_MIN_MS, wait)), signal);
    }
  } finally {
    if (signal) signal.removeEventListener('abort', onAbort);
  }
}

/**
 * Start an analysis of a search state and poll it to its end (F.3).
 * @param {Object} state  F.2 state (naver / wiki / sample)
 * @param {{signal?: AbortSignal, onUpdate?: Function, onStart?: (start: Object) => void, hl?: () => string,
 *          defaultSource?: string}} [opts]
 * @returns {Promise<Object>} the C.7 result envelope
 */
export async function analyze(state, opts = {}) {
  const body = toAnalyzeBody(state, { defaultSource: opts.defaultSource });
  return analyzeBody(body, opts);
}

/** POST an explicit C.5 body (uploads) and poll it. */
export async function analyzeBody(body, opts = {}) {
  const start = await startAnalyze(body, opts);
  if (opts.onStart) opts.onStart(start);
  return pollJob(start.job_id, { ...opts, initial: start });
}

/** POST /api/analyze only: `{job_id, status, cached, joined, queue_position, poll_url, result_url}`. */
export function startAnalyze(body, opts = {}) {
  return request('POST', '/api/analyze', { json: body, signal: opts.signal });
}

/**
 * POST /api/upload with a File (multipart) or pasted text (JSON) (C.4). Files over 10 MB are refused before
 * uploading with a `body_too_large` error (F.3 pre-check).
 * @param {File|Blob|string} input
 * @param {{signal?: AbortSignal, hl?: string, filename?: string}} [opts]
 */
export function upload(input, opts = {}) {
  const path = withHl('/api/upload', opts.hl);
  if (typeof input === 'string') {
    const json = { text: input };
    if (opts.filename) json.filename = opts.filename;
    return request('POST', path, { json, signal: opts.signal });
  }
  if (input && input.size > UPLOAD_MAX_BYTES) {
    return Promise.reject(ApiError.client('body_too_large', 'err.file_too_large', { status: 413 }));
  }
  const form = new FormData();
  form.append('file', input, input && input.name ? input.name : 'upload');
  return request('POST', path, { body: form, signal: opts.signal });
}

export const config = (opts = {}) => request('GET', '/api/config', opts);
export const samples = (hl, opts = {}) => request('GET', withHl('/api/samples', hl), opts);
export const result = (id, hl, opts = {}) => request('GET', withHl(`/api/results/${encodeURIComponent(id)}`, hl), opts);
export const shared = (id, hl, opts = {}) => request('GET', withHl(`/api/shared/${encodeURIComponent(id)}`, hl), opts);
export const share = (resultId) => request('POST', '/api/share', { json: { result_id: resultId } });
export const deleteShare = (shareId, token) => request('DELETE', `/api/share/${encodeURIComponent(shareId)}`,
  { headers: { 'X-Trendai-Delete-Token': token } });

/**
 * AI rewrite of one series' narrative (C.11) and its polling; resolves with the narrative object of the done job.
 * @param {string} resultId
 * @param {number} k
 * @param {'ko'|'en'} lang
 * @param {{signal?: AbortSignal}} [opts]
 */
export async function narrative(resultId, k, lang, opts = {}) {
  const start = await request('POST', '/api/narrative', {
    json: { result_id: resultId, series: k, lang }, signal: opts.signal,
  });
  return pollJob(start.job_id, { ...opts, kind: 'narrative', hl: lang });
}
