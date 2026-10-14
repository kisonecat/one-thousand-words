// Minimal incremental decoder for an ONNX causal LM (SmolLM2 / Qwen2.5
// exports) on onnxruntime-web's WASM backend. Runs in Node and the browser
// (src/node.js loads models from disk; web/ fetches them).
//
// Determinism matters more than speed here: the encoder and decoder must see
// bit-identical logits, so we pin the backend to WASM (CPU).

import * as ort from 'onnxruntime-web';
import { Vocab } from './vocab.js';

// 4 threads unless LM_THREADS says otherwise; scripts/thread_test.mjs verified
// that the thread count does not change the logits.
ort.env.wasm.numThreads = Number(globalThis.process?.env?.LM_THREADS ?? 4);

export { ort };

export class LM {
  // modelBytes: Uint8Array of the .onnx file; config and tokenizerJson are
  // the parsed config.json and tokenizer.json.
  static async create(modelBytes, config, tokenizerJson) {
    const session = await ort.InferenceSession.create(modelBytes, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
    });
    return new LM(session, config, tokenizerJson);
  }

  constructor(session, config, tokenizerJson) {
    this.session = session;
    this.config = config;
    this.vocab = new Vocab(tokenizerJson);
    this.layers = config.num_hidden_layers;
    this.kvHeads = config.num_key_value_heads;
    this.headDim = config.hidden_size / config.num_attention_heads;
    this.reset();
  }

  reset() {
    this.past = null;
    this.pos = 0;
  }

  // Snapshot/restore of the KV cache, so a shared prompt is computed once.
  save() {
    return { past: this.past, pos: this.pos };
  }
  restore(s) {
    this.past = s.past;
    this.pos = s.pos;
  }

  // Feed token ids; returns logits (Float32Array) for the last position.
  async feed(ids) {
    const n = ids.length;
    const total = this.pos + n;
    const feeds = {
      input_ids: new ort.Tensor('int64', BigInt64Array.from(ids, BigInt), [1, n]),
      attention_mask: new ort.Tensor('int64', new BigInt64Array(total).fill(1n), [1, total]),
      position_ids: new ort.Tensor(
        'int64',
        BigInt64Array.from({ length: n }, (_, i) => BigInt(this.pos + i)),
        [1, n],
      ),
    };
    for (let l = 0; l < this.layers; l++) {
      for (const kv of ['key', 'value']) {
        const name = `past_key_values.${l}.${kv}`;
        feeds[name] = this.past
          ? this.past[name]
          : new ort.Tensor('float32', new Float32Array(0), [1, this.kvHeads, 0, this.headDim]);
      }
    }
    const out = await this.session.run(feeds);
    const past = {};
    for (let l = 0; l < this.layers; l++) {
      for (const kv of ['key', 'value']) {
        past[`past_key_values.${l}.${kv}`] = out[`present.${l}.${kv}`];
      }
    }
    this.past = past;
    this.pos = total;
    const logits = out.logits.data;
    const V = out.logits.dims[2];
    return logits.slice((n - 1) * V, n * V);
  }
}
