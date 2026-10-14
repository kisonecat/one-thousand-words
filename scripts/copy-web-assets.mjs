// Copies runtime files the web app serves itself into web/public/:
// onnxruntime-web's asyncify WASM build for transformers.js (which would
// otherwise fetch it from a CDN) and the cross-origin-isolation service
// worker. Vite bundles the story LM's plain WASM build by itself.

import { copyFile, mkdir } from 'node:fs/promises';

const root = new URL('..', import.meta.url).pathname;
const ort = `${root}node_modules/onnxruntime-web/dist/`;
const pub = `${root}web/public/`;

await mkdir(`${pub}ort`, { recursive: true });
for (const variant of ['.asyncify']) {
  for (const ext of ['mjs', 'wasm']) {
    const f = `ort-wasm-simd-threaded${variant}.${ext}`;
    await copyFile(ort + f, `${pub}ort/${f}`);
  }
}
await copyFile(`${root}node_modules/coi-serviceworker/coi-serviceworker.min.js`, `${pub}coi-serviceworker.min.js`);
console.log('copied web assets to web/public/');
