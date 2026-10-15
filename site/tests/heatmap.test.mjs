// site/tests/heatmap.test.mjs
import { test } from "node:test";
import assert from "node:assert/strict";
import { differenceMap, downsampleOverlap } from "../js/heatmap.js";

test("head-weighted channel sum, zero outside the overlap, scaled to [0,1]", () => {
  const N = 6400, local = new Float32Array(32 * N);
  local[5] = 2; local[N + 7] = 1;                    // channel 0 at pixel 5, channel 1 at pixel 7
  const ov = new Uint8Array(N); ov[5] = 1; ov[7] = 1;
  const head = { w_mean: Array(32).fill(0), w_max: Array(32).fill(0), bias: 0 };
  head.w_max[0] = 1; head.w_max[1] = 1;
  const m = differenceMap(local, ov, head);
  assert.equal(m[5], 1); assert.equal(m[7], 0.5); assert.equal(m[6], 0);
  ov[5] = 0; assert.equal(differenceMap(local, ov, head)[5], 0);
});

test("overlap downsampling is nearest with stride 8", () => {
  const big = new Float32Array(512 * 800); big[8 * 800 + 16] = 1;
  const small = downsampleOverlap(big);
  assert.equal(small.length, 6400); assert.equal(small[100 + 2], 1); assert.equal(small[0], 0);
});
