// Entry point for `node --test --test-concurrency=1 tests/e2e` (spec H.4.1, H.5). Node 22 resolves a directory
// argument of --test as a module path, so without this file the command fails with MODULE_NOT_FOUND. It loads
// every *.test.mjs file of this directory in name order. (A runner that expands directories itself never picks this
// file up: index.js does not match node's test-file patterns.)
'use strict';

const { readdirSync } = require('node:fs');
const { join } = require('node:path');
const { pathToFileURL } = require('node:url');

(async () => {
  for (const name of readdirSync(__dirname).filter((f) => f.endsWith('.test.mjs')).sort()) {
    await import(pathToFileURL(join(__dirname, name)).href);
  }
})().catch((err) => {
  console.error(err);
  process.exitCode = 1;
});
