// site/js/verify.js
import { buildQuery, buildOverlap, rgbToNormalizedCHW, decodePng } from "./tensors.js";
import { sigmoid, decide } from "./decision.js";
import { pickProviders, loadSession, runOnce } from "./runner.js";

const W = 800, H = 512;
const png = async (u) => decodePng(await (await fetch(u)).arrayBuffer(), window.UPNG);

async function main() {
  window.ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";
  const m = await (await fetch("manifest.json")).json();
  const T = window.ort.Tensor, prov = new URLSearchParams(location.search).get("wasm") ? ["wasm"] : pickProviders(navigator);
  const sessions = {};
  let backend = "";
  for (const arm of ["on", "off"]) {
    const s = await loadSession(window.ort, `model/refernet_${arm}.onnx`, prov, () => {});
    sessions[arm] = s.session; backend = s.backend;
  }
  let bad = 0, n = 0;
  for (const s of m.samples) {
    const [q, mk, r] = await Promise.all([png(s.query), png(s.masks), png(s.reference)]);
    const qt = buildQuery(q.rgba, mk.rgba, W, H), ov = buildOverlap(mk.rgba, W, H), rt = rgbToNormalizedCHW(r.rgba, W, H);
    for (const arm of ["on", "off"]) {
      const feeds = { query: new T("float32", qt, [1, 4, H, W]) };
      if (arm === "on") { feeds.reference = new T("float32", rt, [1, 3, H, W]); feeds.overlap = new T("float32", ov, [1, 1, H, W]); }
      const { outputs } = await runOnce(sessions[arm], feeds);
      const p = sigmoid(outputs.query_logit.data[0] + (arm === "on" ? outputs.correction.data[0] : 0));
      const ref = s.pytorch[arm].p, d = Math.abs(p - ref), ok = d <= 1e-3 && decide(p) === decide(ref);
      bad += ok ? 0 : 1; n++;
      document.getElementById("rows").insertAdjacentHTML("beforeend",
        `<tr style="color:${ok ? "inherit" : "#c8102e"}"><td>${s.id}</td><td>${arm}</td><td>${p.toFixed(5)}</td><td>${ref.toFixed(5)}</td><td>${d.toExponential(1)}</td><td>${decide(p)}</td></tr>`);
    }
  }
  document.getElementById("summary").textContent = bad ? `FAIL (${bad} of ${n} mismatches)` : `PASS (${n} runs, backend ${backend})`;
}

main();
