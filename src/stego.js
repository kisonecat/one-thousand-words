// Arithmetic-coding steganography: hide a bitstream in LM-generated text.
//
// The encoder treats the secret bits as an arithmetic-coded message and
// "decompresses" it through the model's next-token distribution; the decoder
// re-runs the model over the text and "compresses" the observed tokens to
// recover the bits (Ziegler et al. 2019, "Neural Linguistic Steganography").
//
// Everything that shapes the distribution is computed with integer or
// IEEE-exact (+, -, *, /) arithmetic so encoder and decoder agree bit for bit,
// given identical logits.

const PREC = 32;
const MASK = 2 ** PREC - 1; // inclusive upper bound of the coding interval
const HALF = 2 ** (PREC - 1);
const FREQ_TOTAL = 1 << 16;
const LOGIT_SCALE = 16; // logits are rounded to 1/16 before use

export const DEFAULTS = {
  temperature: 1.0,
  topK: 64,
  topP: 0.95,
};

// --- deterministic exp -------------------------------------------------------

const LN2 = 0.6931471805599453;
const POW2_NEG = [1];
for (let i = 1; i <= 1100; i++) POW2_NEG.push(POW2_NEG[i - 1] * 0.5);

// exp(x) for x <= 0 using only correctly rounded operations, so every JS
// engine returns the same double (Math.exp is not specified to be exact).
function detExp(x) {
  if (x < -700) return 0;
  const k = Math.round(x / LN2); // <= 0
  const r = x - k * LN2; // |r| <= ~0.35
  let term = 1;
  let sum = 1;
  for (let i = 1; i <= 14; i++) {
    term = (term * r) / i;
    sum += term;
  }
  return sum * POW2_NEG[-k];
}

// --- next-token distribution ------------------------------------------------

function isPrefix(a, b) {
  // true if byte array a is a prefix of b
  if (a.length > b.length) return false;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
  return true;
}

// Returns { ids, cum } with cum.length === ids.length + 1, cum[0] = 0.
// The candidate set is prefix-free as byte strings, which lets the decoder
// parse the text greedily without depending on the tokenizer. `allowed` is a
// per-token 0/1 mask (vocab.allowed, or a stricter one from vocab.js).
export function distribution(logits, vocab, params = DEFAULTS, allowed = vocab.allowed) {
  const { temperature, topK, topP } = params;

  let qmax = -Infinity;
  for (let i = 0; i < allowed.length; i++) {
    if (allowed[i]) {
      const q = Math.round(logits[i] * LOGIT_SCALE);
      if (q > qmax) qmax = q;
    }
  }
  const floor = qmax - 30 * LOGIT_SCALE;
  let cand = [];
  for (let i = 0; i < allowed.length; i++) {
    if (allowed[i]) {
      const q = Math.round(logits[i] * LOGIT_SCALE);
      if (q >= floor) cand.push({ id: i, q });
    }
  }
  cand.sort((a, b) => b.q - a.q || a.id - b.id);
  cand = cand.slice(0, topK);

  const denom = LOGIT_SCALE * temperature;
  for (const c of cand) c.w = detExp((c.q - qmax) / denom);

  // Nucleus cut.
  let total = 0;
  for (const c of cand) total += c.w;
  let acc = 0;
  let keep = 0;
  while (keep < cand.length) {
    acc += cand[keep].w;
    keep++;
    if (acc >= topP * total) break;
  }
  cand = cand.slice(0, keep);

  // Greedy prefix-free filter, most probable first.
  const kept = [];
  for (const c of cand) {
    const b = vocab.bytes[c.id];
    if (kept.every((k) => !isPrefix(k.b, b) && !isPrefix(b, k.b))) {
      kept.push({ ...c, b });
    }
  }

  let wsum = 0;
  for (const c of kept) wsum += c.w;
  const ids = new Array(kept.length);
  const cum = new Array(kept.length + 1);
  cum[0] = 0;
  kept.forEach((c, i) => {
    ids[i] = c.id;
    cum[i + 1] = cum[i] + Math.max(1, Math.floor((c.w / wsum) * FREQ_TOTAL));
  });
  return { ids, cum };
}

// Sub-interval boundaries for the current coding interval [low, high]
// (inclusive). Tokens whose slice rounds to zero width are dropped.
function slices(low, high, cum) {
  const range = high - low + 1; // <= 2^32, so range * cum < 2^53
  const total = cum[cum.length - 1];
  const b = new Array(cum.length);
  for (let i = 0; i < cum.length; i++) {
    b[i] = low + Math.floor((range * cum[i]) / total);
  }
  return b; // token i owns [b[i], b[i+1] - 1] when b[i+1] > b[i]
}

// Shift out the leading bits that low and high already agree on.
function* settle(state) {
  while (state.low >= HALF || state.high < HALF) {
    const bit = state.low >= HALF ? 1 : 0;
    state.low = (state.low * 2) % (MASK + 1);
    state.high = ((state.high * 2) % (MASK + 1)) + 1;
    yield bit;
  }
}

export function endsSentence(text) {
  return /[.!?]["')]?$/.test(text);
}

// --- encoder / decoder ------------------------------------------------------

// lm: object with async feed(ids) -> logits and vocab; it must already have
// consumed the prompt, and `logits` is the distribution for the first token.
// opts.onToken(text, settledBits) is called after every token.
export async function encode(lm, logits, bits, params = DEFAULTS, opts = {}) {
  const maxTokens = opts.maxTokens ?? 4000;
  const tail = opts.tail ?? 60; // extra tokens allowed to finish a sentence
  const bitAt = (i) => (i < bits.length ? bits[i] : 0);

  const state = { low: 0, high: MASK };
  let value = 0;
  let read = 0;
  for (; read < PREC; read++) value = value * 2 + bitAt(read);

  let settled = 0;
  const tokens = [];
  let text = '';
  let doneAt = null;
  while (tokens.length < maxTokens) {
    const { ids, cum } = distribution(logits, lm.vocab, params);
    const b = slices(state.low, state.high, cum);
    let pick = -1;
    for (let i = 0; i < ids.length; i++) {
      if (b[i + 1] > b[i] && value >= b[i] && value < b[i + 1]) {
        pick = i;
        break;
      }
    }
    if (pick < 0) throw new Error('encoder: value outside interval');
    state.low = b[pick];
    state.high = b[pick + 1] - 1;
    for (const _ of settle(state)) {
      value = (value * 2) % (MASK + 1) + bitAt(read++);
      settled++;
    }
    const id = ids[pick];
    tokens.push(id);
    text += lm.vocab.text([id]);
    opts.onToken?.(text, Math.min(settled, bits.length));
    if (doneAt === null && settled >= bits.length) doneAt = tokens.length;
    if (doneAt !== null && (endsSentence(text) || tokens.length - doneAt >= tail)) break;
    logits = await lm.feed([id]);
  }
  if (doneAt === null) throw new Error('encoder: ran out of tokens');
  return { tokens, text, payloadTokens: doneAt };
}

// Returns every bit the text determines (callers truncate using a header).
// opts.onProgress(bytesDone, bytesTotal) is called after every token.
export async function decode(lm, logits, text, params = DEFAULTS, opts = {}) {
  const bytes = new TextEncoder().encode(text);
  const state = { low: 0, high: MASK };
  const out = [];
  let pos = 0;
  while (pos < bytes.length) {
    const { ids, cum } = distribution(logits, lm.vocab, params);
    const b = slices(state.low, state.high, cum);
    let pick = -1;
    for (let i = 0; i < ids.length; i++) {
      if (b[i + 1] > b[i]) {
        const tb = lm.vocab.bytes[ids[i]];
        if (pos + tb.length <= bytes.length && tb.every((x, j) => x === bytes[pos + j])) {
          pick = i;
          break;
        }
      }
    }
    if (pick < 0) {
      const err = new Error(`decoder: no candidate matches text at byte ${pos}`);
      err.bits = out;
      err.pos = pos;
      throw err;
    }
    state.low = b[pick];
    state.high = b[pick + 1] - 1;
    for (const bit of settle(state)) out.push(bit);
    const id = ids[pick];
    pos += lm.vocab.bytes[id].length;
    opts.onProgress?.(pos, bytes.length);
    if (pos < bytes.length) logits = await lm.feed([id]);
  }
  return out;
}
