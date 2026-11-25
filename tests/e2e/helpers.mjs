// E2E helpers (spec H.4.1): the test server, browser pages with request/console bookkeeping, downloads.
//
// Playwright is the globally installed Node module: `createRequire` finds it only through NODE_PATH
// (NODE_PATH="$(npm root -g)"; tests/e2e/test_e2e.py sets it). The server is tests/e2e/server.py, run with
// TRENDAI_E2E_PYTHON (default python3).

import { spawn } from 'node:child_process';
import { readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const require = createRequire(import.meta.url);

export const HERE = path.dirname(fileURLToPath(import.meta.url));
export const ROOT = path.resolve(HERE, '..', '..');
export const FIXTURES = path.join(HERE, 'fixtures');
export const STATIC_JS = path.join(ROOT, 'trendai', 'web', 'static', 'js');

/** The global `playwright` module (throws MODULE_NOT_FOUND without NODE_PATH). */
export function playwright() {
  return require('playwright');
}

/**
 * Start tests/e2e/server.py and wait for its `READY <url>` line (30 s).
 * @param {{naver?: boolean, ai?: boolean}} [opts]
 * @returns {Promise<{url: string, stderr: () => string, stop: () => Promise<void>}>}
 */
export function startServer({ naver = true, ai = true } = {}) {
  const python = process.env.TRENDAI_E2E_PYTHON || 'python3';
  const env = { ...process.env, PYTHONPATH: [ROOT, process.env.PYTHONPATH].filter(Boolean).join(path.delimiter) };
  const child = spawn(python, [path.join(HERE, 'server.py'), '--naver', naver ? 'on' : 'off', '--ai', ai ? 'on' : 'off'],
    { cwd: ROOT, env, stdio: ['pipe', 'pipe', 'pipe'] });
  let stdout = '';
  let stderr = '';
  child.stderr.setEncoding('utf8');
  child.stderr.on('data', (chunk) => { stderr += chunk; });
  const exited = new Promise((resolve) => child.once('exit', resolve));

  const stop = async () => {
    if (child.exitCode !== null || child.signalCode !== null) return;
    child.stdin.end();
    child.kill('SIGTERM');
    const timer = setTimeout(() => child.kill('SIGKILL'), 10_000);
    await exited;
    clearTimeout(timer);
  };

  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      stop();
      reject(new Error(`the E2E server did not start within 30 s\n${stderr}`));
    }, 30_000);
    child.stdout.setEncoding('utf8');
    child.stdout.on('data', (chunk) => {
      stdout += chunk;
      const m = stdout.match(/^READY (\S+)$/m);
      if (m) {
        clearTimeout(timer);
        resolve({ url: m[1], stderr: () => stderr, stop });
      }
    });
    child.once('exit', (code) => {
      clearTimeout(timer);
      reject(new Error(`the E2E server exited with ${code}\n${stderr}`));
    });
  });
}

/** Run `fn(server)` against a fresh server (own data dir and rate limits), then stop it. */
export async function withServer(opts, fn) {
  const server = await startServer(opts);
  try {
    return await fn(server);
  } finally {
    await server.stop();
  }
}

/**
 * A new context + page. `errors` collects page errors and console errors (resource-load failures excluded: the app
 * answers some requests with documented 4xx/5xx statuses, which Chromium logs as console errors); `posts` counts
 * POST /api/analyze requests.
 */
export async function newPage(browser, { viewport = { width: 1280, height: 900 }, colorScheme = 'light',
  isMobile = false, hasTouch = false } = {}) {
  const context = await browser.newContext({
    viewport, colorScheme, isMobile, hasTouch, locale: 'ko-KR', timezoneId: 'Asia/Seoul', acceptDownloads: true,
  });
  const page = await context.newPage();
  const errors = [];
  const posts = [];
  page.on('pageerror', (err) => errors.push(`pageerror: ${err.message}`));
  page.on('console', (m) => {
    if (m.type() === 'error' && !/Failed to load resource/.test(m.text())) errors.push(`console: ${m.text()}`);
  });
  page.on('request', (r) => {
    if (r.method() === 'POST' && new URL(r.url()).pathname === '/api/analyze') posts.push(r.url());
  });
  return { context, page, errors, posts };
}

/** Click `trigger` and return the download's suggested name and bytes. */
export async function download(page, trigger) {
  const [dl] = await Promise.all([page.waitForEvent('download'), trigger()]);
  return { name: dl.suggestedFilename(), bytes: readFileSync(await dl.path()) };
}

/** The JSON of the next GET /api/results/<id> response (register before navigating). */
export function nextResult(page) {
  return page.waitForResponse((r) => /\/api\/results\/[A-Za-z0-9_-]{22}(\?|$)/.test(r.url()) && r.status() === 200)
    .then((r) => r.json());
}

/** Count job log lines of the server (E.11) matching `re`. */
export function jobLines(server, re) {
  return server.stderr().split('\n').filter((line) => / trendai\.web\.jobs /.test(line) && re.test(line));
}
