// Do logits stay bit-identical across WASM thread counts?
//
//   node scripts/thread_test.mjs            (spawns one run per thread count)
//
// Each child greedily generates N tokens and prints a hash of every step's
// logits; the parent compares the hashes and timings.

import { spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';

const N = 200;
const root = new URL('..', import.meta.url).pathname;

if (process.env.CHILD) {
  const { loadLM, loadTokenizer } = await import('../src/node.js');
  const { subjectPromptIds: promptIds } = await import('../src/prompt.js');
  const dir = `${root}models/smollm2-135m`;
  const [lm, tok] = await Promise.all([loadLM(dir), loadTokenizer(dir)]);
  let logits = await lm.feed(promptIds(tok, 'A smiling man in front of a chalkboard'));
  const hashes = [];
  const t0 = performance.now();
  for (let i = 0; i < N; i++) {
    hashes.push(createHash('sha1').update(Buffer.from(logits.buffer)).digest('hex').slice(0, 12));
    let best = 0;
    for (let j = 1; j < logits.length; j++) if (logits[j] > logits[best]) best = j;
    logits = await lm.feed([best]);
  }
  console.log(JSON.stringify({ hashes, tokPerSec: N / ((performance.now() - t0) / 1000) }));
  process.exit(0);
}

const runs = {};
for (const threads of [1, 2, 4, 8]) {
  const r = spawnSync(process.execPath, [new URL(import.meta.url).pathname], {
    env: { ...process.env, CHILD: '1', LM_THREADS: String(threads) },
    encoding: 'utf8',
  });
  if (r.status !== 0) {
    console.log(`${threads} threads: failed\n${r.stderr}`);
    continue;
  }
  runs[threads] = JSON.parse(r.stdout.trim().split('\n').pop());
  const ref = runs[1].hashes;
  const firstDiff = runs[threads].hashes.findIndex((h, i) => h !== ref[i]);
  console.log(
    `${threads} threads: ${runs[threads].tokPerSec.toFixed(1)} tok/s, ` +
      (firstDiff < 0 ? `all ${N} steps bit-identical to 1 thread` : `logits differ from step ${firstDiff}`),
  );
}
