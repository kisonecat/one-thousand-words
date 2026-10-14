// Run each quantized TiTok variant end to end in onnxruntime-web (the
// browser runtime) and save the reconstructions for a visual check.
//
//   node scripts/check_titok_variants.mjs PHOTO

import * as ort from 'onnxruntime-web';
import { loadSquare, savePng } from '../src/image-node.js';
import { SIZE } from '../src/titok.js';

const root = new URL('..', import.meta.url).pathname;
const dir = `${root}models/titok-b64`;
const photo = process.argv[2];
const x = await loadSquare(photo);

async function run(file, feeds) {
  const s = await ort.InferenceSession.create(`${dir}/${file}`, { executionProviders: ['wasm'] });
  const out = await s.run(feeds);
  await s.release();
  return Object.values(out)[0].data;
}

const encodeWith = async (suffix) =>
  Array.from(await run(`encoder${suffix}.onnx`, { image: new ort.Tensor('float32', x, [1, 3, SIZE, SIZE]) }), Number);
const decodeWith = (suffix, codes) =>
  run(`decoder${suffix}.onnx`, { codes: new ort.Tensor('int64', BigInt64Array.from(codes, BigInt), [codes.length]) });

const refCodes = await encodeWith('');
const refImg = await decodeWith('', refCodes);
for (const mode of ['fp16', 'int8', 'fp8']) {
  try {
    const t0 = performance.now();
    const codes = await encodeWith(`_${mode}`);
    const img = await decodeWith(`_${mode}`, codes);
    let mse = 0;
    for (let i = 0; i < img.length; i++) mse += (img[i] - refImg[i]) ** 2;
    const psnr = 10 * Math.log10(img.length / mse);
    const match = codes.filter((c, i) => c === refCodes[i]).length;
    await savePng(`${root}out/fowler_titok_${mode}.png`, img);
    console.log(
      `${mode}: codes ${match}/64 match fp32, end-to-end PSNR vs fp32 ${psnr.toFixed(1)} dB, ` +
        `${((performance.now() - t0) / 1000).toFixed(1)}s`,
    );
  } catch (e) {
    console.log(`${mode}: FAILED in onnxruntime-web: ${e.message.split('\n')[0]}`);
  }
}
