// Caption photos with small vision-language models, to pick a title writer.
//
//   node scripts/caption.mjs PHOTO [PHOTO...]

import {
  env,
  AutoProcessor,
  AutoModelForVision2Seq,
  Florence2ForConditionalGeneration,
  RawImage,
} from '@huggingface/transformers';

env.cacheDir = new URL('../models/hf-cache', import.meta.url).pathname;

const photos = process.argv.slice(2);

const CAPTIONERS = {
  async 'Florence-2-base-ft (q4)'() {
    const id = 'onnx-community/Florence-2-base-ft';
    const dtype = { embed_tokens: 'quantized', vision_encoder: 'q4', encoder_model: 'q4', decoder_model_merged: 'q4' };
    const model = await Florence2ForConditionalGeneration.from_pretrained(id, { dtype });
    const processor = await AutoProcessor.from_pretrained(id);
    return async (image) => {
      const task = '<CAPTION>';
      const inputs = await processor(image, processor.construct_prompts(task));
      const ids = await model.generate({ ...inputs, max_new_tokens: 40 });
      const text = processor.batch_decode(ids, { skip_special_tokens: false })[0];
      return processor.post_process_generation(text, task, image.size)[task];
    };
  },

  async 'SmolVLM-256M-Instruct (q4)'() {
    const id = 'HuggingFaceTB/SmolVLM-256M-Instruct';
    const dtype = { embed_tokens: 'quantized', vision_encoder: 'q4', decoder_model_merged: 'q4' };
    const model = await AutoModelForVision2Seq.from_pretrained(id, { dtype });
    const processor = await AutoProcessor.from_pretrained(id);
    return async (image) => {
      const messages = [
        {
          role: 'user',
          content: [
            { type: 'image' },
            { type: 'text', text: 'Write a short title for this photo, like a caption in a museum.' },
          ],
        },
      ];
      const prompt = processor.apply_chat_template(messages, { add_generation_prompt: true });
      const inputs = await processor(prompt, [image]);
      const ids = await model.generate({ ...inputs, max_new_tokens: 30, do_sample: false });
      const out = processor.batch_decode(ids.slice(null, [inputs.input_ids.dims.at(-1), null]), {
        skip_special_tokens: true,
      });
      return out[0].trim();
    };
  },
};

const images = await Promise.all(photos.map((p) => RawImage.read(p)));
for (const [name, load] of Object.entries(CAPTIONERS)) {
  let t0 = performance.now();
  const caption = await load();
  console.log(`## ${name} (loaded in ${((performance.now() - t0) / 1000).toFixed(1)}s)`);
  for (let i = 0; i < photos.length; i++) {
    t0 = performance.now();
    const text = await caption(images[i]);
    console.log(`  ${photos[i].split('/').pop()}: "${text}" (${((performance.now() - t0) / 1000).toFixed(1)}s)`);
  }
}
