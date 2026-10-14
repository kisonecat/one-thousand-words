// Turn a photo into a story that hides it.
//
//   node scripts/photo2story.mjs PHOTO ["what the story is about"] > story.txt
//
// Without a subject, Florence-2 captions the photo and the caption is used.
// The caption only steers the story; it isn't needed to decode it.

import { env } from '@huggingface/transformers';
import { loadLM, loadTokenizer } from '../src/node.js';
import { encodeStory } from '../src/story.js';
import { TiTok } from '../src/titok.js';
import { pack } from '../src/payload.js';
import { loadSquare } from '../src/image-node.js';
import { Captioner } from '../src/caption.js';

env.cacheDir = new URL('../models/hf-cache', import.meta.url).pathname;

const root = new URL('..', import.meta.url).pathname;
const [photo, givenSubject] = process.argv.slice(2);
if (!photo) {
  console.error('usage: photo2story.mjs PHOTO ["subject"]');
  process.exit(2);
}
let subject = givenSubject;
if (!subject) {
  const t0 = performance.now();
  subject = await (await Captioner.load()).caption(photo);
  console.error(`caption: "${subject}" (${((performance.now() - t0) / 1000).toFixed(1)}s)`);
}

let t0 = performance.now();
const titok = await TiTok.load(`${root}models/titok-b64`, { decoder: false });
const codes = await titok.encode(await loadSquare(photo));
const bits = pack(codes);
console.error(`image -> ${codes.length} codes, ${bits.length} bits with CRC (${((performance.now() - t0) / 1000).toFixed(1)}s)`);

const dir = `${root}models/smollm2-135m`;
const [lm, tok] = await Promise.all([loadLM(dir), loadTokenizer(dir)]);
t0 = performance.now();
const story = await encodeStory(lm, tok, subject, bits);
console.error(`story: ${story.split(/\s+/).length} words (${((performance.now() - t0) / 1000).toFixed(1)}s)`);

console.log(story);
