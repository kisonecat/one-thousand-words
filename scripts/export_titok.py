"""Export a TiTok tokenizer to two ONNX graphs for the browser.

    nix-shell --run 'python scripts/export_titok.py [hub_id] [outdir]'

encoder.onnx: image float32 [1,3,256,256] in [0,1] -> codes int64 [64]
decoder.onnx: codes int64 [64] -> image float32 [1,3,256,256] in [0,1]

Then checks the ONNX graphs against PyTorch on a test photo.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "vendor/1d-tokenizer"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import onnxruntime as ort
import torch
import torch.nn.functional as F
from modeling.titok import TiTok
from titok_recon import load_square, to_tensor, to_image

HUB_ID = sys.argv[1] if len(sys.argv) > 1 else "yucornetto/tokenizer_titok_b64_imagenet"
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else ROOT / "models/titok-b64")
TEST_PHOTO = Path.home() / "downloads/fowler.jpg"


class Encoder(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, image):
        z = self.m.encoder(pixel_values=image, latent_tokens=self.m.latent_tokens)
        z = z.reshape(z.shape[1], -1).t()  # [tokens, token_size]
        q = self.m.quantize
        if q.use_l2_norm:
            # For unit vectors, nearest neighbour == largest dot product.
            scores = F.normalize(z, dim=-1) @ F.normalize(q.embedding.weight, dim=-1).t()
            return scores.argmax(dim=1)
        d = (z**2).sum(1, keepdim=True) + (q.embedding.weight**2).sum(1) - 2 * z @ q.embedding.weight.t()
        return d.argmin(dim=1)


class Decoder(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, codes):
        return self.m.decode_tokens(codes.view(1, 1, -1)).clamp(0, 1)


def main():
    torch.set_grad_enabled(False)
    m = TiTok.from_pretrained(HUB_ID).eval()
    n = m.num_latent_tokens
    OUT.mkdir(parents=True, exist_ok=True)

    x = to_tensor(load_square(TEST_PHOTO))
    ref_codes = m.encode(x)[1]["min_encoding_indices"].reshape(-1)
    enc, dec = Encoder(m).eval(), Decoder(m).eval()
    assert torch.equal(enc(x), ref_codes), "wrapper disagrees with TiTok.encode"
    ref_img = dec(ref_codes)

    torch.onnx.export(enc, (x,), OUT / "encoder.onnx", input_names=["image"],
                      output_names=["codes"], dynamo=False, opset_version=17)
    torch.onnx.export(dec, (ref_codes,), OUT / "decoder.onnx", input_names=["codes"],
                      output_names=["image"], dynamo=False, opset_version=17)

    so = ort.SessionOptions()
    e = ort.InferenceSession(OUT / "encoder.onnx", so)
    d = ort.InferenceSession(OUT / "decoder.onnx", so)
    codes = e.run(None, {"image": x.numpy()})[0]
    img = d.run(None, {"codes": ref_codes.numpy()})[0]
    same = int((codes == ref_codes.numpy()).sum())
    print(f"codes: {same}/{n} match PyTorch")
    print(f"image: max abs diff {np.abs(img - ref_img.numpy()).max():.2e}")
    to_image(torch.from_numpy(img)).save(OUT / "test_decode.png")
    for f in ("encoder.onnx", "decoder.onnx"):
        print(f, f"{(OUT / f).stat().st_size / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
