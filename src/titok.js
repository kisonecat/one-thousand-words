// TiTok image tokenizer (ONNX export from scripts/export_titok.py).
//
// A 256x256 RGB image becomes 64 codes of 12 bits each (768 bits). Unlike
// the language model, the image codec does not need bit-exact determinism:
// small float differences only nudge pixels.

import * as ort from 'onnxruntime-web';

export const SIZE = 256;
export const NUM_CODES = 64;
export const CODE_BITS = 12;
export const PAYLOAD_BITS = NUM_CODES * CODE_BITS;

export class TiTok {
  // variant: '' (fp32), '_fp16', '_int8' or '_fp8' (scripts/quantize_titok.py)
  static load(dir, { encoder = true, decoder = true, variant = '_int8' } = {}) {
    return TiTok.create({
      encoder: encoder && `${dir}/encoder${variant}.onnx`,
      decoder: decoder && `${dir}/decoder${variant}.onnx`,
    });
  }

  // encoder/decoder: a path, URL or Uint8Array of the .onnx file (or falsy
  // to skip that half).
  static async create({ encoder, decoder }) {
    const opts = { executionProviders: ['wasm'], graphOptimizationLevel: 'all' };
    const t = new TiTok();
    if (encoder) t.enc = await ort.InferenceSession.create(encoder, opts);
    if (decoder) t.dec = await ort.InferenceSession.create(decoder, opts);
    return t;
  }

  // rgb: Float32Array [3*256*256] planar, values in [0,1] -> Int array [64]
  async encode(rgb) {
    const out = await this.enc.run({ image: new ort.Tensor('float32', rgb, [1, 3, SIZE, SIZE]) });
    return Array.from(out.codes.data, Number);
  }

  async decode(codes) {
    const t = new ort.Tensor('int64', BigInt64Array.from(codes, BigInt), [codes.length]);
    const out = await this.dec.run({ codes: t });
    return out.image.data.slice(); // Float32Array planar [3*256*256] in [0,1]
  }
}

export function codesToBits(codes) {
  const bits = [];
  for (const c of codes) for (let i = CODE_BITS - 1; i >= 0; i--) bits.push((c >> i) & 1);
  return bits;
}

export function bitsToCodes(bits) {
  const codes = [];
  for (let k = 0; k < NUM_CODES; k++) {
    let c = 0;
    for (let i = 0; i < CODE_BITS; i++) c = c * 2 + (bits[k * CODE_BITS + i] ?? 0);
    codes.push(c);
  }
  return codes;
}
