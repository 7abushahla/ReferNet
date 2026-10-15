// site/js/runner.js
export const pickProviders = (nav) => (nav && nav.gpu ? ["webgpu", "wasm"] : ["wasm"]);

export class RunGate {
  #busy = false;
  tryEnter() { if (this.#busy) return false; this.#busy = true; return true; }
  leave() { this.#busy = false; }
}

/** Download with progress, then create the session with the first provider that works; reports which one. */
export async function loadSession(ort, url, providers, onProgress) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`model download failed (${res.status})`);
  const total = Number(res.headers.get("content-length")) || 0;
  const reader = res.body.getReader(), chunks = [];
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value); got += value.length; onProgress(total ? got / total : 0);
  }
  const bytes = new Uint8Array(got);
  let o = 0;
  for (const c of chunks) { bytes.set(c, o); o += c.length; }
  for (const p of providers) {
    try { return { session: await ort.InferenceSession.create(bytes, { executionProviders: [p] }), backend: p }; }
    catch (e) { console.warn(`provider ${p} failed`, e); }
  }
  throw new Error("no execution provider could load the model");
}

export async function runOnce(session, feeds) {
  const t0 = performance.now();
  const outputs = await session.run(feeds);
  return { outputs, ms: performance.now() - t0 };
}
