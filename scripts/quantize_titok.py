"""Weight-only quantization of the TiTok ONNX graphs, to shrink the download.

    nix-shell --run 'python scripts/quantize_titok.py [model_dir]'

Large weight initializers are stored as fp16, int8 or fp8 (e4m3) with a
per-output-channel scale, and expanded back to fp32 by a Cast or
DequantizeLinear node; all arithmetic stays fp32. Writes
{encoder,decoder}_{fp16,int8,fp8}.onnx next to the fp32 originals and
reports how far each strays from fp32 on a test photo.
"""
import sys
from pathlib import Path

import ml_dtypes
import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, numpy_helper, version_converter

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from titok_recon import load_square, to_tensor  # noqa: E402

DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "models/titok-b64"
MIN_SIZE = 65536  # leave small tensors (codebook, norms, biases) in fp32
TEST_PHOTO = Path.home() / "downloads/fowler.jpg"


def quantize(model, mode):
    model = version_converter.convert_version(model, 19)  # fp8 DequantizeLinear
    g = model.graph
    keep, new_nodes = [], []
    for init in g.initializer:
        w = numpy_helper.to_array(init)
        if w.dtype != np.float32 or w.ndim < 2 or w.size < MIN_SIZE:
            keep.append(init)
            continue
        name = init.name
        if mode == "fp16":
            keep.append(numpy_helper.from_array(w.astype(np.float16), name + "_fp16"))
            new_nodes.append(helper.make_node("Cast", [name + "_fp16"], [name], to=onnx.TensorProto.FLOAT))
            continue
        absmax = np.abs(w.reshape(w.shape[0], -1)).max(axis=1)
        if mode == "int8":
            scale = np.maximum(absmax / 127, 1e-12).astype(np.float32)
            q = np.clip(np.round(w / scale.reshape(-1, *[1] * (w.ndim - 1))), -127, 127).astype(np.int8)
        else:  # fp8 e4m3: max finite value 448
            scale = np.maximum(absmax / 448, 1e-12).astype(np.float32)
            q = (w / scale.reshape(-1, *[1] * (w.ndim - 1))).astype(ml_dtypes.float8_e4m3fn)
        keep.append(numpy_helper.from_array(q, name + "_q"))
        keep.append(numpy_helper.from_array(scale, name + "_scale"))
        new_nodes.append(helper.make_node("DequantizeLinear", [name + "_q", name + "_scale"], [name], axis=0))
    del g.initializer[:]
    g.initializer.extend(keep)
    nodes = new_nodes + list(g.node)
    del g.node[:]
    g.node.extend(nodes)
    return model


def main():
    x = to_tensor(load_square(TEST_PHOTO)).numpy()
    run = lambda path, feeds: ort.InferenceSession(str(path)).run(None, feeds)[0]
    ref_codes = run(DIR / "encoder.onnx", {"image": x})
    ref_img = run(DIR / "decoder.onnx", {"codes": ref_codes})

    for mode in ("fp16", "int8", "fp8"):
        sizes = []
        for part in ("encoder", "decoder"):
            out = DIR / f"{part}_{mode}.onnx"
            onnx.save(quantize(onnx.load(DIR / f"{part}.onnx"), mode), out)
            sizes.append(out.stat().st_size / 1e6)
        codes = run(DIR / f"encoder_{mode}.onnx", {"image": x})
        img = run(DIR / f"decoder_{mode}.onnx", {"codes": ref_codes})
        psnr = 10 * np.log10(1 / np.mean((img - ref_img) ** 2))
        print(f"{mode}: encoder {sizes[0]:.0f} MB, decoder {sizes[1]:.0f} MB, "
              f"codes {int((codes == ref_codes).sum())}/{codes.size} match fp32, "
              f"decoder PSNR vs fp32 {psnr:.1f} dB")


if __name__ == "__main__":
    main()
