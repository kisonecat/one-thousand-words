// Byte-level view of a GPT-2 style BPE vocabulary (as used by Qwen2.5).
//
// Each token id maps to the raw bytes it stands for. Tokens we never let the
// encoder emit (special tokens, non-ASCII, whitespace that may not survive
// copy/paste) are marked as disallowed.

function byteDecoder() {
  // Inverse of GPT-2's bytes_to_unicode().
  const bs = [];
  for (let b = 0x21; b <= 0x7e; b++) bs.push(b);
  for (let b = 0xa1; b <= 0xac; b++) bs.push(b);
  for (let b = 0xae; b <= 0xff; b++) bs.push(b);
  const cs = bs.slice();
  let n = 0;
  for (let b = 0; b < 256; b++) {
    if (!bs.includes(b)) {
      bs.push(b);
      cs.push(256 + n);
      n++;
    }
  }
  const dec = new Map();
  bs.forEach((b, i) => dec.set(String.fromCodePoint(cs[i]), b));
  return dec;
}

// Printable ASCII only, so the text survives being pasted around. No token
// may end in a space (prevents doubled or trailing spaces, which editors and
// HTML collapse) and none may contain two consecutive spaces.
function isAllowed(bytes) {
  if (bytes.length === 0) return false;
  for (const b of bytes) if (b < 0x20 || b > 0x7e) return false;
  if (bytes[bytes.length - 1] === 0x20) return false;
  for (let i = 1; i < bytes.length; i++) {
    if (bytes[i] === 0x20 && bytes[i - 1] === 0x20) return false;
  }
  return true;
}

export class Vocab {
  constructor(tokenizerJson) {
    const dec = byteDecoder();
    const vocab = tokenizerJson.model.vocab;
    let size = 0;
    for (const id of Object.values(vocab)) size = Math.max(size, id + 1);
    for (const t of tokenizerJson.added_tokens) size = Math.max(size, t.id + 1);
    this.bytes = new Array(size).fill(null);
    this.allowed = new Uint8Array(size);
    for (const [str, id] of Object.entries(vocab)) {
      const b = Uint8Array.from([...str], (ch) => dec.get(ch));
      this.bytes[id] = b;
      this.allowed[id] = isAllowed(b) ? 1 : 0;
    }
    // Added/special tokens stay disallowed.
    for (const t of tokenizerJson.added_tokens) {
      this.bytes[t.id] = new TextEncoder().encode(t.content);
      this.allowed[t.id] = 0;
    }
    // While writing a story's opening (see opening.js), sentence-ending
    // punctuation may only be a token's last byte, so the decoder can find
    // where the opening ends by counting [.!?]. The very first token must not
    // start with a space, so trimming the story is harmless.
    const isEnd = (b) => b === 0x2e || b === 0x21 || b === 0x3f;
    this.allowedOpening = this.allowed.map((a, id) => {
      if (!a) return 0;
      const b = this.bytes[id];
      for (let i = 0; i < b.length - 1; i++) if (isEnd(b[i])) return 0;
      return 1;
    });
    this.allowedOpeningFirst = this.allowedOpening.map((a, id) => (a && this.bytes[id][0] !== 0x20 ? 1 : 0));
  }

  text(ids) {
    const parts = ids.map((id) => this.bytes[id]);
    const out = new Uint8Array(parts.reduce((n, p) => n + p.length, 0));
    let o = 0;
    for (const p of parts) {
      out.set(p, o);
      o += p.length;
    }
    return new TextDecoder().decode(out);
  }
}
