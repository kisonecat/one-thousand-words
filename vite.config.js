import { defineConfig } from 'vite';
import { createReadStream, statSync } from 'node:fs';
import { resolve } from 'node:path';

const MODELS = resolve(import.meta.dirname, 'models');

// In development, serve ../models/titok-b64 at /models/titok-b64 so the app
// works before the ONNX files are published to the Hugging Face Hub.
function localModels() {
  return {
    name: 'local-models',
    configureServer(server) {
      server.middlewares.use('/models/', (req, res, next) => {
        const file = resolve(MODELS, decodeURIComponent(req.url.split('?')[0]).replace(/^\/+/, ''));
        if (!file.startsWith(MODELS + '/')) return next();
        try {
          const { size } = statSync(file);
          res.setHeader('Content-Length', size);
          res.setHeader('Content-Type', 'application/octet-stream');
          createReadStream(file).pipe(res);
        } catch {
          next();
        }
      });
    },
  };
}

// Multi-threaded WASM needs SharedArrayBuffer, which needs cross-origin
// isolation. The dev server sends the headers; on GitHub Pages the
// coi-serviceworker shim adds them.
const isolation = {
  'Cross-Origin-Opener-Policy': 'same-origin',
  'Cross-Origin-Embedder-Policy': 'require-corp',
};

export default defineConfig({
  root: 'web',
  base: './',
  plugins: [localModels()],
  resolve: {
    // The story LM must run the same WASM binary as in Node (the plain
    // ort-wasm-simd-threaded.wasm), not the WebGPU/JSEP build that the bare
    // 'onnxruntime-web' specifier picks in browsers. We use the "extern wasm"
    // build, which loads ort-wasm-simd-threaded.mjs/.wasm from
    // ort.env.wasm.wasmPaths (copied to public/ort/): the bundled variant
    // spawns its pthread workers from import.meta.url, which after bundling is
    // our lm-worker chunk, so threads never start. transformers.js imports
    // 'onnxruntime-web/webgpu' and is unaffected.
    alias: [
      {
        find: /^onnxruntime-web$/,
        replacement: resolve(import.meta.dirname, 'node_modules/onnxruntime-web/dist/ort.wasm.min.mjs'),
      },
    ],
  },
  server: { headers: isolation },
  preview: { headers: isolation },
  worker: { format: 'es' },
  optimizeDeps: { exclude: ['onnxruntime-web', '@huggingface/transformers'] },
  build: { outDir: '../dist', emptyOutDir: true, target: 'es2022' },
});
