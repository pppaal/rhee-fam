// IME-safe commit rules for the search box (spec F.6.3, H.3). Owner: WP3-app. Pure: no DOM at import time.
//
// Korean (and other) IMEs fire keydown events for half-built syllables: those carry `isComposing === true` or the
// legacy `keyCode === 229`. Committing on them would split 선크림 into 선크 + 림, so they are ignored; a comma typed
// during composition is picked up from the input's value on `compositionend` / `input` instead (splitCommitted).

/** Keys that commit the current text as a series. */
export const COMMIT_KEYS = Object.freeze(['Enter', ',']);

/**
 * True when a keydown event is part of an IME composition and must not trigger any shortcut.
 * @param {{isComposing?: boolean, keyCode?: number}} event
 * @returns {boolean}
 */
export function isComposingEvent(event) {
  return Boolean(event) && (event.isComposing === true || event.keyCode === 229);
}

/**
 * Whether a keydown event commits the current text as a series (F.6.3): `Enter` or `,`, never during composition.
 * @param {{key?: string, isComposing?: boolean, keyCode?: number}} event
 * @returns {boolean}
 */
export function shouldCommit(event) {
  if (!event || isComposingEvent(event)) return false;
  return COMMIT_KEYS.includes(event.key);
}

/**
 * Split an input value at its commas: every complete piece before the last comma is committed, the text after it
 * stays in the input. Used on `input` / `compositionend`, where a comma may arrive without a usable keydown (IME,
 * mobile keyboards, paste).
 * @param {string} value
 * @returns {{commit: string[], rest: string}}  commit: trimmed, non-empty pieces in order
 */
export function splitCommitted(value) {
  const text = String(value == null ? '' : value);
  const at = text.lastIndexOf(',');
  if (at < 0) return { commit: [], rest: text };
  const commit = text.slice(0, at).split(',').map((p) => p.trim()).filter(Boolean);
  return { commit, rest: text.slice(at + 1).replace(/^\s+/, '') };
}
