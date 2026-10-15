// site/tests/runner.test.mjs
import { test } from "node:test";
import assert from "node:assert/strict";
import { pickProviders, RunGate } from "../js/runner.js";

test("WebGPU first when available, WebAssembly otherwise", () => {
  assert.deepEqual(pickProviders({ gpu: {} }), ["webgpu", "wasm"]);
  assert.deepEqual(pickProviders({}), ["wasm"]);
});

test("one run at a time", () => {
  const g = new RunGate();
  assert.equal(g.tryEnter(), true); assert.equal(g.tryEnter(), false);
  g.leave(); assert.equal(g.tryEnter(), true);
});

test("a provider whose warm-up run fails falls back to the next one", async () => {
  const { loadSession } = await import("../js/runner.js");
  globalThis.fetch = async () => new Response(new Uint8Array([1, 2, 3]));
  const ort = { InferenceSession: { create: async (_bytes, { executionProviders: [p] }) => ({
    run: async () => { if (p === "webgpu") throw new Error("kernel not supported"); return { ok: true }; } }) } };
  const warm = () => ({ x: 1 });
  const { backend } = await loadSession(ort, "model.onnx", ["webgpu", "wasm"], () => {}, warm);
  assert.equal(backend, "wasm");
});
