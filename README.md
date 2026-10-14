# A Picture Is Worth 1,000 Words

Turn a photo into a short story, and the story back into the photo. The picture is encoded in the words
themselves (actually just a few hundred words are enough).

Everything runs in the browser:

- **Image codec:** [TiTok-B-64](https://github.com/bytedance/1d-tokenizer), exported to ONNX with int8
  weights ([kisonecat/titok-b64-int8-onnx](https://huggingface.co/kisonecat/titok-b64-int8-onnx)). A 256x256
  photo becomes 64 codes, 768 bits, plus a 16-bit CRC.
- **Captioner:** [Florence-2-base](https://huggingface.co/onnx-community/Florence-2-base-ft) describes the
  photo; the caption steers the story's opening sentences.
- **Language model:** [SmolLM2-135M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct),
  pinned to one commit, on onnxruntime-web's WASM backend. Arithmetic coding lets the photo's bits choose
  each word; decoding reruns the model and recovers the bits. Encoder and decoder must compute bit-identical
  probabilities, so the model, the WASM binary and the probability quantization are all fixed.

## Development

```sh
npm install
npm run dev        # http://localhost:5173 (serves TiTok from ./models)
npm run build      # static site in dist/
```

Command-line versions of both directions (models in `./models`):

```sh
node scripts/photo2story.mjs photo.jpg > story.txt
node scripts/story2photo.mjs story.txt photo.png
```

Python tooling (TiTok export and quantization) runs in `nix-shell`; see `scripts/export_titok.py`,
`scripts/quantize_titok.py` and `scripts/upload_titok.py`.
