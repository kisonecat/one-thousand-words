// Whole-story encode/decode: opening (from the caption, no bits) plus a
// continuation that carries the bits. See prompt.js for why.

import { distribution, encode, decode } from './stego.js';
import { OPENING_SENTENCES, subjectPromptIds, storyPromptIds, splitOpening } from './prompt.js';

const OPENING_PARAMS = { temperature: 0.8, topK: 40, topP: 0.9 };
const MAX_OPENING_TOKENS = 120;

// Samples the opening sentences about `subject` (ordinary random sampling:
// nothing is hidden here, so it needn't be reproducible).
export async function writeOpening(lm, tokenizer, subject, random = Math.random) {
  lm.reset();
  let logits = await lm.feed(subjectPromptIds(tokenizer, subject));
  const v = lm.vocab;
  let text = '';
  let sentences = 0;
  for (let n = 0; n < MAX_OPENING_TOKENS; n++) {
    const mask = n === 0 ? v.allowedOpeningFirst : v.allowedOpening;
    const { ids, cum } = distribution(logits, v, OPENING_PARAMS, mask);
    const r = random() * cum[cum.length - 1];
    let i = 0;
    while (cum[i + 1] <= r) i++;
    const piece = v.text([ids[i]]);
    text += piece;
    if ('.!?'.includes(piece[piece.length - 1]) && ++sentences === OPENING_SENTENCES) return text;
    logits = await lm.feed([ids[i]]);
  }
  // Rambled on without finishing: close the sentences ourselves.
  return text + '.'.repeat(OPENING_SENTENCES - sentences);
}

// Returns the full story text. opts.onToken(storySoFar, settledBits) streams.
export async function encodeStory(lm, tokenizer, subject, bits, opts = {}) {
  const opening = await writeOpening(lm, tokenizer, subject, opts.random);
  opts.onOpening?.(opening);
  lm.reset();
  const logits = await lm.feed(storyPromptIds(tokenizer, opening));
  const { text } = await encode(lm, logits, bits, undefined, {
    onToken: opts.onToken && ((t, settled) => opts.onToken(opening + t, settled)),
  });
  return opening + text;
}

// Returns every bit hidden in `story` (already normalized). Errors carry
// .pos (character offset in the story) when the text doesn't decode.
export async function decodeStory(lm, tokenizer, story, opts = {}) {
  const parts = splitOpening(story);
  if (!parts) {
    const err = new Error(`a story needs at least ${OPENING_SENTENCES} sentences`);
    err.pos = story.length;
    throw err;
  }
  lm.reset();
  const logits = await lm.feed(storyPromptIds(tokenizer, parts.opening));
  try {
    return await decode(lm, logits, parts.rest, undefined, { onProgress: opts.onProgress });
  } catch (e) {
    if (e.pos !== undefined) e.pos += parts.opening.length;
    throw e;
  }
}
