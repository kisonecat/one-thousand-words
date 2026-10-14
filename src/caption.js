// Captions a photo to use as the story's title, with Florence-2-base (q4,
// ~214 MB). Only the encoder runs this, so it needs no determinism; the
// title travels in plain sight at the top of the story.

import { AutoProcessor, Florence2ForConditionalGeneration, RawImage } from '@huggingface/transformers';
import { toTitle } from './title.js';

export { toTitle };

const MODEL = 'onnx-community/Florence-2-base-ft';
const DTYPE = { embed_tokens: 'quantized', vision_encoder: 'q4', encoder_model: 'q4', decoder_model_merged: 'q4' };

export class Captioner {
  // opts: extra from_pretrained options (progress_callback, device, ...)
  static async load(opts = {}) {
    const [model, processor] = await Promise.all([
      Florence2ForConditionalGeneration.from_pretrained(MODEL, { dtype: DTYPE, ...opts }),
      AutoProcessor.from_pretrained(MODEL, { progress_callback: opts.progress_callback }),
    ]);
    return new Captioner(model, processor);
  }

  constructor(model, processor) {
    this.model = model;
    this.processor = processor;
  }

  // image: a RawImage, or a path/URL RawImage.read() understands.
  async caption(image) {
    if (!(image instanceof RawImage)) image = await RawImage.read(image);
    const task = '<CAPTION>';
    const inputs = await this.processor(image, this.processor.construct_prompts(task));
    const ids = await this.model.generate({ ...inputs, max_new_tokens: 40 });
    const raw = this.processor.batch_decode(ids, { skip_special_tokens: false })[0];
    return toTitle(this.processor.post_process_generation(raw, task, image.size)[task]);
  }
}
