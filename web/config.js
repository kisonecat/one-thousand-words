// Where the browser gets its models. The story LM is pinned to an exact
// commit: encoder and decoder must use bit-identical weights forever.

const LM_REPO = 'HuggingFaceTB/SmolLM2-135M-Instruct';
const LM_REVISION = '12fd25f77366fa6b3b4b768ec3050bf629380bac';

export const LM = {
  repo: LM_REPO,
  revision: LM_REVISION,
  base: `https://huggingface.co/${LM_REPO}/resolve/${LM_REVISION}/`,
  model: 'onnx/model_quantized.onnx',
  approxMB: 131,
};

// Our int8 TiTok export (scripts/upload_titok.py), pinned to a commit. The
// dev server serves the local copy in ../models instead; VITE_TITOK_BASE
// overrides both.
const TITOK_HUB = 'https://huggingface.co/kisonecat/titok-b64-int8-onnx/resolve/cd546ee5b42e0c714a1be535ed952e16e486b7eb/';

export const TITOK = {
  // appBase: the page's base URL (workers can't use their own location).
  base: (appBase) =>
    import.meta.env.VITE_TITOK_BASE ||
    (import.meta.env.DEV ? new URL('models/titok-b64/', appBase).href : TITOK_HUB),
  encoder: 'encoder_int8.onnx',
  decoder: 'decoder_int8.onnx',
  encoderMB: 87,
  decoderMB: 121,
};

export const CAPTIONER = {
  repo: 'onnx-community/Florence-2-base-ft',
  approxMB: 214,
};
