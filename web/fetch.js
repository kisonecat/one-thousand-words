// Download a file with progress, keeping a copy in the Cache API so models
// are fetched once per browser.

const CACHE = 'one-thousand-words-models-v1';

async function openCache() {
  try {
    return await caches.open(CACHE);
  } catch {
    return null; // e.g. private browsing
  }
}

// onProgress(loadedBytes, totalBytes | 0)
export async function fetchBytes(url, onProgress = () => {}) {
  const cache = await openCache();
  const hit = await cache?.match(url);
  if (hit) {
    const buf = new Uint8Array(await hit.arrayBuffer());
    onProgress(buf.length, buf.length);
    return buf;
  }

  const res = await fetch(url);
  if (!res.ok) throw new Error(`download failed (${res.status}): ${url}`);
  const total = Number(res.headers.get('Content-Length')) || 0;
  const reader = res.body.getReader();
  const chunks = [];
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.length;
    onProgress(loaded, total);
  }
  const buf = new Uint8Array(loaded);
  let o = 0;
  for (const c of chunks) {
    buf.set(c, o);
    o += c.length;
  }
  onProgress(loaded, loaded);
  try {
    await cache?.put(url, new Response(buf));
  } catch {
    // Quota exceeded: fine, we just download again next time.
  }
  return buf;
}

export async function fetchJson(url) {
  return JSON.parse(new TextDecoder().decode(await fetchBytes(url)));
}
