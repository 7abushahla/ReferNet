// site/tests/tensors.test.mjs
import { test } from "node:test";
import assert from "node:assert/strict";
import { rgbToNormalizedCHW, buildQuery, buildOverlap } from "../js/tensors.js";

test("normalizes like the paper (ImageNet mean/std, RGB order, CHW)", () => {
  const rgba = new Uint8Array([255, 0, 128, 255, 0, 255, 64, 255]); // w=2, h=1
  const t = rgbToNormalizedCHW(rgba, 2, 1);
  assert.equal(t.length, 6);
  assert.ok(Math.abs(t[0] - (1 - 0.485) / 0.229) < 1e-6);
  assert.ok(Math.abs(t[1] - (0 - 0.485) / 0.229) < 1e-6);
  assert.ok(Math.abs(t[2] - (0 - 0.456) / 0.224) < 1e-6);
  assert.ok(Math.abs(t[5] - (64 / 255 - 0.406) / 0.225) < 1e-6);
});

test("query takes the zone from R as its 4th channel; overlap comes from G", () => {
  const t = buildQuery(new Uint8Array([10, 20, 30, 255]), new Uint8Array([255, 0, 0, 255]), 1, 1);
  assert.equal(t.length, 4); assert.equal(t[3], 1);
  assert.deepEqual(Array.from(buildOverlap(new Uint8Array([255, 0, 0, 255]), 1, 1)), [0]);
  assert.deepEqual(Array.from(buildOverlap(new Uint8Array([0, 255, 0, 255]), 1, 1)), [1]);
});
