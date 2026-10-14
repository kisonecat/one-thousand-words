// Step 1 prototype: hide random bits in a story and recover them.
//
//   node scripts/roundtrip.mjs [nbits] [title] [seed]

import { loadLM, loadTokenizer } from '../src/node.js';
import { subjectPromptIds as promptIds } from '../src/prompt.js';
import { encode, decode, DEFAULTS } from '../src/stego.js';

const MODEL = new URL('../models/smollm2-135m', import.meta.url).pathname;
const nbits = Number(process.argv[2] ?? 512);
const title = process.argv[3] ?? 'An old man feeding pigeons in a snowy square';
let seed = Number(process.argv[4] ?? 1);

function rand() {
  // mulberry32
  seed = (seed + 0x6d2b79f5) | 0;
  let t = seed;
  t = Math.imul(t ^ (t >>> 15), t | 1);
  t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
}

const bits = Array.from({ length: nbits }, () => (rand() < 0.5 ? 1 : 0));

const [lm, tok] = await Promise.all([loadLM(MODEL), loadTokenizer(MODEL)]);
const ids = promptIds(tok, title);
console.error(`prompt: ${ids.length} tokens; params ${JSON.stringify(DEFAULTS)}`);

let t0 = performance.now();
const first = await lm.feed(ids);
const promptState = lm.save();
console.error(`prompt prefill: ${((performance.now() - t0) / 1000).toFixed(1)}s`);

t0 = performance.now();
const enc = await encode(lm, first, bits);
const encSec = (performance.now() - t0) / 1000;
const words = enc.text.trim().split(/\s+/).length;
console.log(`\n# ${title}\n\n${enc.text}\n`);
console.error(
  `encode: ${enc.tokens.length} tokens (${enc.payloadTokens} carry payload), ${words} words, ` +
    `${(nbits / enc.payloadTokens).toFixed(2)} bits/token, ` +
    `${((nbits * enc.tokens.length) / enc.payloadTokens / words).toFixed(2)} bits/word, ` +
    `${encSec.toFixed(1)}s (${(enc.tokens.length / encSec).toFixed(1)} tok/s)`,
);

lm.restore(promptState);
t0 = performance.now();
const out = await decode(lm, first, enc.text);
const decSec = (performance.now() - t0) / 1000;
const got = out.slice(0, nbits);
const errors = got.reduce((n, b, i) => n + (b !== bits[i] ? 1 : 0), 0);
console.error(
  `decode: ${out.length} bits recovered, ${errors} errors in first ${nbits}, ${decSec.toFixed(1)}s`,
);
process.exit(errors === 0 && got.length === nbits ? 0 : 1);
