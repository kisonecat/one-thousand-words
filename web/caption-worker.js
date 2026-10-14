// Captions photos with Florence-2 in its own worker, so transformers.js's
// onnxruntime build never touches the story LM's (see lm-worker.js).
//
// In:  { type: 'init', appBase } | { type: 'caption', id, blob }
// Out: { type: 'download', name, loaded, total }
//      | { type: 'status', id, stage: 'loading' | 'captioning' }
//      | { type: 'done', id, title } | { type: 'error', id, message }

import { env, RawImage } from '@huggingface/transformers';
import { Captioner } from '../src/caption.js';

let appBase;
let captioner;

// transformers.js reports progress per file; sum them into one bar.
const files = new Map();
function onProgress(p) {
  if (p.status !== 'progress' && p.status !== 'done') return;
  const f = files.get(p.file) ?? { loaded: 0, total: 0 };
  if (p.status === 'progress') Object.assign(f, { loaded: p.loaded, total: p.total });
  else f.loaded = f.total;
  files.set(p.file, f);
  let loaded = 0;
  let total = 0;
  for (const v of files.values()) {
    loaded += v.loaded;
    total += v.total;
  }
  postMessage({ type: 'download', name: 'Captioner', loaded, total });
}

onmessage = async ({ data }) => {
  if (data.type === 'init') {
    appBase = data.appBase;
    return;
  }
  try {
    postMessage({ type: 'status', id: data.id, stage: 'loading' });
    if (!captioner) {
      env.allowLocalModels = false;
      env.backends.onnx.wasm.wasmPaths = {
        mjs: new URL('ort/ort-wasm-simd-threaded.asyncify.mjs', appBase).href,
        wasm: new URL('ort/ort-wasm-simd-threaded.asyncify.wasm', appBase).href,
      };
      captioner = Captioner.load({ progress_callback: onProgress, device: 'wasm' });
    }
    const c = await captioner;
    postMessage({ type: 'status', id: data.id, stage: 'captioning' });
    const title = await c.caption(await RawImage.fromBlob(data.blob));
    postMessage({ type: 'done', id: data.id, title });
  } catch (e) {
    console.error(e);
    captioner = null;
    postMessage({ type: 'error', id: data.id, message: e.message || String(e) });
  }
};
