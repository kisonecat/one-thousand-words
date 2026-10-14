// Browser image helpers: center-crop a photo to 256x256 planar float RGB
// (TiTok's input), and paint planar RGB back onto a canvas.

const SIZE = 256; // TiTok's input size (src/titok.js)

export async function fileToSquare(file) {
  const bmp = await createImageBitmap(file, { imageOrientation: 'from-image' });
  const side = Math.min(bmp.width, bmp.height);
  const canvas = new OffscreenCanvas(SIZE, SIZE);
  const ctx = canvas.getContext('2d');
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(bmp, (bmp.width - side) / 2, (bmp.height - side) / 2, side, side, 0, 0, SIZE, SIZE);
  bmp.close();
  const { data } = ctx.getImageData(0, 0, SIZE, SIZE);
  const N = SIZE * SIZE;
  const rgb = new Float32Array(3 * N);
  for (let i = 0; i < N; i++) {
    rgb[i] = data[i * 4] / 255;
    rgb[N + i] = data[i * 4 + 1] / 255;
    rgb[2 * N + i] = data[i * 4 + 2] / 255;
  }
  return rgb;
}

export function paint(canvas, rgb) {
  const N = SIZE * SIZE;
  const img = new ImageData(SIZE, SIZE);
  for (let i = 0; i < N; i++) {
    for (let c = 0; c < 3; c++) {
      img.data[i * 4 + c] = Math.round(Math.min(1, Math.max(0, rgb[c * N + i])) * 255);
    }
    img.data[i * 4 + 3] = 255;
  }
  canvas.getContext('2d').putImageData(img, 0, 0);
}
