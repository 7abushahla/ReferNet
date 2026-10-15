// site/tests/decision.test.mjs
import { test } from "node:test";
import assert from "node:assert/strict";
import { sigmoid, decide, decideMismatch, percent } from "../js/decision.js";

test("refers above 0.5 and clears at or below (ties genuine, as in the paper)", () => {
  assert.equal(decide(0.51), "REFER"); assert.equal(decide(0.5), "CLEAR"); assert.equal(decide(0.2), "CLEAR");
  assert.ok(Math.abs(sigmoid(0) - 0.5) < 1e-12);
});
test("a document of another design is always referred", () => {
  assert.deepEqual(decideMismatch(), { decision: "REFER", reason: "no reference context, sent to the second line" });
});
test("percent formatting", () => {
  assert.equal(percent(0.968), "97%"); assert.equal(percent(0.002), "<1%"); assert.equal(percent(0.9995), ">99%");
});
