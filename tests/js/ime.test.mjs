// IME-safe commit (spec F.6.3, H.3).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { isComposingEvent, shouldCommit, splitCommitted } from '../../trendai/web/static/js/ime.js';

test('never commits during IME composition', () => {
  assert.equal(shouldCommit({ key: 'Enter', isComposing: true }), false);
  assert.equal(shouldCommit({ key: 'Enter', keyCode: 229 }), false);
  assert.equal(shouldCommit({ key: ',', isComposing: true }), false);
  assert.equal(shouldCommit({ key: 'Process', keyCode: 229 }), false);
  assert.equal(isComposingEvent({ keyCode: 229 }), true);
});

test('commits on Enter and comma otherwise', () => {
  assert.equal(shouldCommit({ key: 'Enter', isComposing: false, keyCode: 13 }), true);
  assert.equal(shouldCommit({ key: ',', keyCode: 188 }), true);
  assert.equal(shouldCommit({ key: 'a' }), false);
  assert.equal(shouldCommit({ key: '+' }), false);
  assert.equal(shouldCommit(null), false);
});

test('splitCommitted keeps the text after the last comma', () => {
  assert.deepEqual(splitCommitted('선크림, 패딩,수'), { commit: ['선크림', '패딩'], rest: '수' });
  assert.deepEqual(splitCommitted('선크림'), { commit: [], rest: '선크림' });
  assert.deepEqual(splitCommitted(',, a ,'), { commit: ['a'], rest: '' });
});
