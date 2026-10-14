"""Reconstruct photos through TiTok tokenizers to judge quality per bit.

    nix-shell --run 'python scripts/titok_recon.py PHOTO [PHOTO...]'

Writes out/<name>_titok.png: a grid with one row per photo (color input) and
one row per photo (grayscale input); columns are the original 256x256 crop
followed by each tokenizer's reconstruction.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "vendor/1d-tokenizer"))

import torch
from PIL import Image, ImageDraw, ImageOps
from modeling.titok import TiTok

MODELS = [  # (hub id, bits per image)
    ("yucornetto/tokenizer_titok_s128_imagenet", 128 * 12),
    ("yucornetto/tokenizer_titok_b64_imagenet", 64 * 12),
]


def load_square(path, size=256):
    im = Image.open(path)
    im = ImageOps.exif_transpose(im).convert("RGB")
    return ImageOps.fit(im, (size, size), Image.Resampling.LANCZOS)


def to_tensor(im):
    return torch.frombuffer(bytearray(im.tobytes()), dtype=torch.uint8).view(256, 256, 3).permute(2, 0, 1).float().div(255)[None]


def to_image(x):
    x = x[0].clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).contiguous()
    return Image.frombytes("RGB", (256, 256), x.numpy().tobytes())


def main(paths):
    torch.set_grad_enabled(False)
    models = []
    for hub_id, bits in MODELS:
        m = TiTok.from_pretrained(hub_id).eval()
        models.append((hub_id.split("_")[2].upper(), bits, m))

    for path in paths:
        color = load_square(path)
        gray = ImageOps.grayscale(color).convert("RGB")
        rows = []
        for im in (color, gray):
            x = to_tensor(im)
            row = [("original", im)]
            for name, bits, m in models:
                codes = m.encode(x)[1]["min_encoding_indices"]
                rec = to_image(m.decode_tokens(codes))
                row.append((f"{name}: {bits} bits", rec))
                print(f"{Path(path).name} {name}: codes {tuple(codes.shape)}", flush=True)
            rows.append(row)
        W, H, pad = 256, 256, 22
        grid = Image.new("RGB", (W * len(rows[0]), (H + pad) * len(rows)), "white")
        d = ImageDraw.Draw(grid)
        for r, row in enumerate(rows):
            for c, (label, im) in enumerate(row):
                grid.paste(im, (c * W, r * (H + pad) + pad))
                d.text((c * W + 6, r * (H + pad) + 5), label, fill="black")
        out = Path("out") / f"{Path(path).stem}_titok.png"
        grid.save(out)
        print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1:])
