// Node-side image I/O (the browser will use a canvas instead): load a JPEG or
// PNG, center-crop to a square, area-resample to 256x256 planar float RGB.

import { readFile, writeFile } from 'node:fs/promises';
import jpeg from 'jpeg-js';
import { PNG } from 'pngjs';
import { SIZE } from './titok.js';

export async function loadSquare(path) {
  const buf = await readFile(path);
  const isPng = buf[0] === 0x89 && buf[1] === 0x50;
  const { width, height, data } = isPng
    ? PNG.sync.read(buf)
    : jpeg.decode(buf, { useTArray: true, maxMemoryUsageInMB: 2048 });

  const side = Math.min(width, height);
  const x0 = (width - side) / 2;
  const y0 = (height - side) / 2;
  const scale = side / SIZE;
  const out = new Float32Array(3 * SIZE * SIZE);
  // Box filter: average all source pixels whose centers fall in each cell.
  for (let y = 0; y < SIZE; y++) {
    const sy0 = Math.floor(y0 + y * scale);
    const sy1 = Math.max(sy0 + 1, Math.floor(y0 + (y + 1) * scale));
    for (let x = 0; x < SIZE; x++) {
      const sx0 = Math.floor(x0 + x * scale);
      const sx1 = Math.max(sx0 + 1, Math.floor(x0 + (x + 1) * scale));
      let r = 0, g = 0, b = 0, n = 0;
      for (let sy = sy0; sy < sy1; sy++) {
        for (let sx = sx0; sx < sx1; sx++) {
          const i = (sy * width + sx) * 4;
          r += data[i];
          g += data[i + 1];
          b += data[i + 2];
          n++;
        }
      }
      const o = y * SIZE + x;
      out[o] = r / n / 255;
      out[SIZE * SIZE + o] = g / n / 255;
      out[2 * SIZE * SIZE + o] = b / n / 255;
    }
  }
  return out;
}

export async function savePng(path, rgb) {
  const png = new PNG({ width: SIZE, height: SIZE });
  const N = SIZE * SIZE;
  for (let i = 0; i < N; i++) {
    for (let c = 0; c < 3; c++) {
      png.data[i * 4 + c] = Math.round(Math.min(1, Math.max(0, rgb[c * N + i])) * 255);
    }
    png.data[i * 4 + 3] = 255;
  }
  await writeFile(path, PNG.sync.write(png));
}
