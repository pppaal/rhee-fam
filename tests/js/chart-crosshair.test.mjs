// chart/crosshair.js pure helpers (F.7.4): nearest point snapping, keyboard targets, the 250 ms live-region
// throttle. Run: node --test tests/js
import { test } from 'node:test';
import assert from 'node:assert/strict';

import { keyTarget, nearestIndex, throttle } from '../../trendai/web/static/js/chart/crosshair.js';

test('nearestIndex: binary search, ties go to the earlier point', () => {
  const xs = [0, 10, 20, 30];
  assert.equal(nearestIndex(xs, -5), 0);
  assert.equal(nearestIndex(xs, 4), 0);
  assert.equal(nearestIndex(xs, 5), 0);          // tie → earlier
  assert.equal(nearestIndex(xs, 6), 1);
  assert.equal(nearestIndex(xs, 29), 3);
  assert.equal(nearestIndex(xs, 99), 3);
  assert.equal(nearestIndex([7], 1000), 0);
  assert.equal(nearestIndex([], 3), -1);
  const big = Array.from({ length: 2000 }, (_, i) => i * 3);
  assert.equal(nearestIndex(big, 3000.4), 1000);
});

test('keyTarget: arrows, Shift+arrows, Home/End, clamping, unhandled keys', () => {
  assert.equal(keyTarget({ key: 'ArrowLeft' }, 5, 10, 7), 4);
  assert.equal(keyTarget({ key: 'ArrowRight' }, 5, 10, 7), 6);
  assert.equal(keyTarget({ key: 'ArrowLeft', shiftKey: true }, 15, 30, 7), 5);
  assert.equal(keyTarget({ key: 'ArrowRight', shiftKey: true }, 25, 30, 7), 29);
  assert.equal(keyTarget({ key: 'ArrowLeft' }, 0, 10, 7), 0);
  assert.equal(keyTarget({ key: 'Home' }, 5, 10, 7), 0);
  assert.equal(keyTarget({ key: 'End' }, 5, 10, 7), 9);
  assert.equal(keyTarget({ key: 'Escape' }, 5, 10, 7), null);
  assert.equal(keyTarget({ key: 'a' }, 5, 10, 7), null);
  assert.equal(keyTarget({ key: 'ArrowLeft' }, 0, 0, 0), null);
});

test('keyTarget: when hidden, the first arrow shows the crosshair at the last actual point', () => {
  assert.equal(keyTarget({ key: 'ArrowLeft' }, -1, 10, 7), 7);
  assert.equal(keyTarget({ key: 'ArrowRight', shiftKey: true }, -1, 10, 7), 7);
  assert.equal(keyTarget({ key: 'End' }, -1, 10, 7), 9);
});

test('throttle: leading call, at most one call per window, trailing call with the latest arguments', async () => {
  const calls = [];
  const fn = throttle((v) => calls.push(v), 40);
  fn(1);
  fn(2);
  fn(3);
  assert.deepEqual(calls, [1]);
  await new Promise((r) => setTimeout(r, 70));
  assert.deepEqual(calls, [1, 3]);
  fn(4);
  fn.cancel();
  await new Promise((r) => setTimeout(r, 70));
  assert.deepEqual(calls, [1, 3]);
  fn(5);
  assert.deepEqual(calls, [1, 3, 5]);
  fn.cancel();
});
