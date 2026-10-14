// Hide the same photo with several language models; print each story with
// stats and check that it decodes back to the same bits.
//
//   node scripts/compare_lms.mjs PHOTO "Title" [model-dir ...] > out/compare.md

import { loadLM, loadTokenizer } from '../src/node.js';
import { subjectPromptIds as promptIds } from '../src/prompt.js';
import { encode, decode } from '../src/stego.js';
import { TiTok, codesToBits } from '../src/titok.js';
import { loadSquare } from '../src/image-node.js';

const root = new URL('..', import.meta.url).pathname;
const [photo, title, ...dirs] = process.argv.slice(2);
const models = dirs.length ? dirs : ['smollm2-135m', 'smollm2-360m', 'qwen2.5-0.5b'];

const titok = await TiTok.load(`${root}models/titok-b64`, { decoder: false });
const bits = codesToBits(await titok.encode(await loadSquare(photo)));

console.log(`# ${title}\n`);
for (const name of models) {
  const dir = `${root}models/${name}`;
  const [lm, tok] = await Promise.all([loadLM(dir), loadTokenizer(dir)]);
  const ids = promptIds(tok, title);
  const first = await lm.feed(ids);
  const prompt = lm.save();

  let t0 = performance.now();
  const enc = await encode(lm, first, bits);
  const encSec = (performance.now() - t0) / 1000;

  lm.restore(prompt);
  t0 = performance.now();
  const got = await decode(lm, first, enc.text);
  const decSec = (performance.now() - t0) / 1000;
  const ok = bits.every((b, i) => got[i] === b);

  const words = enc.text.trim().split(/\s+/).length;
  const stats =
    `${words} words, ${enc.tokens.length} tokens, ` +
    `${(bits.length / words).toFixed(2)} bits/word, ` +
    `encode ${encSec.toFixed(1)}s (${(enc.tokens.length / encSec).toFixed(1)} tok/s), ` +
    `decode ${decSec.toFixed(1)}s, round trip ${ok ? 'OK' : 'FAILED'}`;
  console.error(`${name}: ${stats}`);
  console.log(`## ${name}\n\n_${stats}_\n\n${enc.text}\n`);
}
