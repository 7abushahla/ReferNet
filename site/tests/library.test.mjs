// site/tests/library.test.mjs
import { test } from "node:test";
import assert from "node:assert/strict";
import { documentOptions, findSample } from "../js/library.js";

const manifest = {
  designs: [{ id: "a", name: "A passport" }, { id: "b", name: "B card" }],
  samples: [
    { id: "a__genuine__1", design: "a", family: "genuine", family_name: "Genuine", n: 1, label: 0 },
    { id: "a__text_inpaint__1", design: "a", family: "text_inpaint", family_name: "Inpaint-and-rewrite", n: 1, label: 1 },
    { id: "b__genuine__1", design: "b", family: "genuine", family_name: "Genuine", n: 1, label: 0 }],
  mismatch: [{ sample: "b__genuine__1", label: "B card · genuine", fits: { a: false } },
             { sample: "a__genuine__1", label: "A passport · genuine", fits: { b: false } }],
};

test("options list the chosen design, then documents of other designs only", () => {
  const g = documentOptions(manifest, "a");
  assert.equal(g[0].group, "A passport");
  assert.deepEqual(g[0].items.map((i) => i.text), ["Genuine · #1", "Inpaint-and-rewrite · #1"]);
  assert.equal(g[1].group, "Another design (no reference match)");
  assert.deepEqual(g[1].items.map((i) => i.text), ["B card · genuine"]);
});

test("findSample resolves both kinds", () => {
  assert.equal(findSample(manifest, "a__text_inpaint__1").kind, "sample");
  const m = findSample(manifest, "mismatch:b__genuine__1");
  assert.equal(m.kind, "mismatch"); assert.equal(m.sample.id, "b__genuine__1");
});
