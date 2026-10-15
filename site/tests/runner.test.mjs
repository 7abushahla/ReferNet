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
