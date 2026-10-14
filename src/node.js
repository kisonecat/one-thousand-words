// Node-only loaders for the models in ../models (the browser fetches them
// from the Hugging Face Hub instead; see web/config.js).

import { readFile } from 'node:fs/promises';
import { dirname, basename } from 'node:path';
import { AutoTokenizer, env } from '@huggingface/transformers';
import { LM } from './lm.js';

export async function loadLM(dir, file = 'onnx/model_quantized.onnx') {
  const [config, tokenizer, modelBytes] = await Promise.all([
    readFile(`${dir}/config.json`, 'utf8').then(JSON.parse),
    readFile(`${dir}/tokenizer.json`, 'utf8').then(JSON.parse),
    readFile(`${dir}/${file}`),
  ]);
  return LM.create(modelBytes, config, tokenizer);
}

export async function loadTokenizer(dir) {
  env.localModelPath = dirname(dir) + '/';
  env.allowRemoteModels = false;
  return AutoTokenizer.from_pretrained(basename(dir));
}
