// Recover the photo hidden in a story written by photo2story.mjs.
//
//   node scripts/story2photo.mjs story.txt out.png

import { readFile } from 'node:fs/promises';
import { loadLM, loadTokenizer } from '../src/node.js';
import { decodeStory } from '../src/story.js';
import { TiTok } from '../src/titok.js';
import { unpack, PAYLOAD_BITS } from '../src/payload.js';
import { savePng } from '../src/image-node.js';
import { normalizeStory } from '../src/document.js';

const root = new URL('..', import.meta.url).pathname;
const [storyPath, outPath] = process.argv.slice(2);
if (!storyPath || !outPath) {
  console.error('usage: story2photo.mjs STORY.txt OUT.png');
  process.exit(2);
}

const story = normalizeStory(await readFile(storyPath, 'utf8'));
const dir = `${root}models/smollm2-135m`;
const [lm, tok] = await Promise.all([loadLM(dir), loadTokenizer(dir)]);
let t0 = performance.now();
let bits;
try {
  bits = await decodeStory(lm, tok, story);
} catch (e) {
  if (e.pos === undefined) throw e;
  console.error(`the story was altered; decoding broke near: "${story.slice(Math.max(0, e.pos - 30), e.pos + 30)}"`);
  process.exit(1);
}
console.error(`story -> ${bits.length} bits (${((performance.now() - t0) / 1000).toFixed(1)}s)`);
if (bits.length < PAYLOAD_BITS) throw new Error(`story too short: ${bits.length} < ${PAYLOAD_BITS} bits`);

const { codes, ok } = unpack(bits);
if (!ok) {
  console.error('CRC mismatch: the story was altered');
  process.exit(1);
}

t0 = performance.now();
const titok = await TiTok.load(`${root}models/titok-b64`, { encoder: false });
await savePng(outPath, await titok.decode(codes));
console.error(`wrote ${outPath} (${((performance.now() - t0) / 1000).toFixed(1)}s)`);
