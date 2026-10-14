// Runs the story LM and the TiTok image codec off the main thread.
//
// Messages in:  { type: 'init', appBase }
//               { type: 'encode', id, rgb: Float32Array, subject }
//               { type: 'decode', id, story }
// Messages out: { type: 'download', name, loaded, total }
//               { type: 'status', id, text }
//               { type: 'token', id, text, bits, totalBits }   (encode)
//               { type: 'progress', id, done, total }          (decode)
//               { type: 'done', id, story | image }
//               { type: 'error', id, message }

import { AutoTokenizer, env as hfEnv } from '@huggingface/transformers';
import { LM, ort } from '../src/lm.js';
import { TiTok } from '../src/titok.js';
import { encodeStory, decodeStory } from '../src/story.js';
import { pack, unpack, PAYLOAD_BITS } from '../src/payload.js';
import { fetchBytes, fetchJson } from './fetch.js';
import { LM as LM_CFG, TITOK } from './config.js';

let appBase;
const once = new Map();
const lazy = (key, fn) => {
  if (!once.has(key)) once.set(key, fn().catch((e) => (once.delete(key), Promise.reject(e))));
  return once.get(key);
};

const download = (name) => (loaded, total) => postMessage({ type: 'download', name, loaded, total });

function threads() {
  // Threads need cross-origin isolation (SharedArrayBuffer); without it,
  // fall back to one thread, which gives identical results, just slower.
  return self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;
}

const loadLM = () =>
  lazy('lm', async () => {
    ort.env.wasm.numThreads = threads();
    hfEnv.allowLocalModels = false;
    const [bytes, config, tokenizerJson, tokenizer] = await Promise.all([
      fetchBytes(LM_CFG.base + LM_CFG.model, download('Story model')),
      fetchJson(LM_CFG.base + 'config.json'),
      fetchJson(LM_CFG.base + 'tokenizer.json'),
      AutoTokenizer.from_pretrained(LM_CFG.repo, { revision: LM_CFG.revision }),
    ]);
    const lm = await LM.create(bytes, config, tokenizerJson);
    return { lm, tokenizer };
  });

const loadTiTok = (part) =>
  lazy(`titok-${part}`, async () => {
    await loadLM(); // sets ort.env.wasm.numThreads first
    const bytes = await fetchBytes(TITOK.base(appBase) + TITOK[part], download(`Image ${part}`));
    return TiTok.create({ [part]: bytes });
  });

async function handleEncode({ id, rgb, subject }) {
  postMessage({ type: 'status', id, text: 'Loading models…' });
  const [titok, { lm, tokenizer }] = await Promise.all([loadTiTok('encoder'), loadLM()]);
  postMessage({ type: 'status', id, text: 'Compressing the photo…' });
  const bits = pack(await titok.encode(rgb));
  postMessage({ type: 'status', id, text: 'Writing the opening…' });
  let last = 0;
  const story = await encodeStory(lm, tokenizer, subject, bits, {
    onOpening: (text) => postMessage({ type: 'token', id, text, bits: 0, totalBits: bits.length }),
    onToken(text, settled) {
      const now = performance.now();
      if (now - last > 50) {
        last = now;
        postMessage({ type: 'token', id, text, bits: settled, totalBits: bits.length });
      }
    },
  });
  postMessage({ type: 'token', id, text: story, bits: bits.length, totalBits: bits.length });
  postMessage({ type: 'done', id, story });
}

async function handleDecode({ id, story }) {
  postMessage({ type: 'status', id, text: 'Loading models…' });
  const [titok, { lm, tokenizer }] = await Promise.all([loadTiTok('decoder'), loadLM()]);
  postMessage({ type: 'status', id, text: 'Reading the story…' });
  let bits;
  try {
    bits = await decodeStory(lm, tokenizer, story, {
      onProgress: (done, total) => postMessage({ type: 'progress', id, done, total }),
    });
  } catch (e) {
    if (e.pos === undefined) throw e;
    const near = story.slice(Math.max(0, e.pos - 40), e.pos + 40);
    throw new Error(`This story has been changed, so the photo can't be recovered. It breaks down here: “…${near}…”`);
  }
  if (bits.length < PAYLOAD_BITS) throw new Error('This story is too short to hold a photo. Was part of it cut off?');
  const { codes, ok } = unpack(bits);
  if (!ok) throw new Error('The hidden photo is damaged: the story was altered.');
  postMessage({ type: 'status', id, text: 'Developing the photo…' });
  const image = await titok.decode(codes);
  postMessage({ type: 'done', id, image }, [image.buffer]);
}

onmessage = async ({ data }) => {
  if (data.type === 'init') {
    appBase = data.appBase;
    return;
  }
  try {
    if (data.type === 'encode') await handleEncode(data);
    else if (data.type === 'decode') await handleDecode(data);
  } catch (e) {
    console.error(e);
    postMessage({ type: 'error', id: data.id, message: e.message || String(e) });
  }
};
