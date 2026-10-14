---
license: apache-2.0
library_name: onnx
pipeline_tag: image-feature-extraction
tags:
  - image-tokenization
  - titok
  - onnx
  - transformers.js
base_model: yucornetto/tokenizer_titok_b64_imagenet
---

# TiTok-B-64 tokenizer, ONNX (int8 weights)

An ONNX export of ByteDance's [TiTok-B-64](https://huggingface.co/yucornetto/tokenizer_titok_b64_imagenet)
image tokenizer from [*An Image is Worth 32 Tokens for Reconstruction and Generation*](https://arxiv.org/abs/2406.07550)
(Yu et al., 2024; code at [bytedance/1d-tokenizer](https://github.com/bytedance/1d-tokenizer)), split into an
encoder and a decoder for use in the browser with [onnxruntime-web](https://onnxruntime.ai/).

It was made for *A Picture Is Worth 1,000 Words*, a web page that turns a photo into a short story and back.

## Files

| File | Input | Output |
|---|---|---|
| `encoder_int8.onnx` (87 MB) | `image`: float32 `[1, 3, 256, 256]`, RGB in `[0, 1]` | `codes`: int64 `[64]`, each in `[0, 4096)` |
| `decoder_int8.onnx` (121 MB) | `codes`: int64 `[64]` | `image`: float32 `[1, 3, 256, 256]`, RGB in `[0, 1]` |

A 256x256 image becomes 64 codes of 12 bits each: 768 bits per image.

## How it was made

1. `export_titok.py`: exports the PyTorch model (opset 17) as two graphs. The encoder includes the
   nearest-codebook lookup; the decoder includes TiTok's MaskGIT-VQGAN pixel decoder. Checked against PyTorch:
   all 64 codes identical, max pixel difference 6e-6.
2. `quantize_titok.py`: weight-only quantization. Every weight tensor with at least 65,536 elements is
   stored as int8 with a symmetric per-output-channel scale and expanded back to float32 by a
   `DequantizeLinear` node (opset 19); all arithmetic stays float32. On a test portrait, the int8 encoder
   matched the float32 codes in 63 of 64 positions, and the end-to-end reconstruction was 35.6 dB PSNR from
   the float32 pipeline.

## License

Apache-2.0, as for the original TiTok weights and code. Copyright for the model belongs to ByteDance Ltd. and
its affiliates; this repository only converts the format.
