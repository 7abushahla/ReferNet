// site/js/main.js
import { buildQuery, buildOverlap, rgbToNormalizedCHW, decodePng } from "./tensors.js";
import { sigmoid, decide, decideMismatch, percent, showMap } from "./decision.js";
import { differenceMap, downsampleOverlap } from "./heatmap.js";
import { documentOptions, findSample } from "./library.js";
import { pickProviders, RunGate, loadSession, runOnce } from "./runner.js";

const W = 800, H = 512, ORT_DIST = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";
const $ = (id) => document.getElementById(id);
const state = { manifest: null, sessions: {}, loading: {}, withRef: true, busy: false, gate: new RunGate() };

async function png(url) { return decodePng(await (await fetch(url)).arrayBuffer(), window.UPNG); }

function fillDesigns() {
  for (const d of state.manifest.designs) $("design").add(new Option(`${d.name} · ${d.dataset}`, d.id));
  $("design").onchange = onDesign;
  onDesign();
}

function onDesign() {
  const d = state.manifest.designs.find((x) => x.id === $("design").value);
  $("ref-img").src = d.reference; $("zone-img").src = d.zone;
  const sel = $("document"); sel.innerHTML = "";
  for (const g of documentOptions(state.manifest, d.id)) {
    const og = document.createElement("optgroup"); og.label = g.group;
    for (const it of g.items) og.appendChild(new Option(it.text, it.value));
    sel.appendChild(og);
  }
  sel.onchange = onDocument;
  onDocument();
}

function onDocument() {
  $("doc-img").src = findSample(state.manifest, $("document").value).sample.query;
  clearResult();
}

function clearResult() {
  $("result").className = "empty"; $("result").textContent = "Run ReferNet to see its decision.";
  $("truth-hint").textContent = "Ground truth hidden until the run";
  const c = $("heat"); c.getContext("2d").clearRect(0, 0, c.width, c.height);
  $("map-wrap").hidden = true;
}

function setCondition(withRef) {
  state.withRef = withRef;
  $("with").classList.toggle("on", withRef); $("without").classList.toggle("on", !withRef);
  $("with").setAttribute("aria-checked", String(withRef)); $("without").setAttribute("aria-checked", String(!withRef));
  $("schematic").classList.toggle("off", !withRef);
  clearResult();
  updateRun();
  if (!withRef) ensureSession("off");
}

const currentArm = () => (state.withRef ? "on" : "off");

/** Run is usable only when the current condition's model is loaded and no run is in flight. */
function updateRun() {
  const ready = !!state.sessions[currentArm()];
  $("run").disabled = !ready || state.busy;
  $("run").textContent = ready ? "▶ Run" : "Loading model…";
}

/** Inputs stay fixed during a run, so a result always belongs to the document and condition on screen. */
function setBusy(busy) {
  state.busy = busy;
  for (const id of ["design", "document", "with", "without"]) $(id).disabled = busy;
  clearTimeout(state.flowTimer);
  if (busy) {
    state.runStart = performance.now();
    $("schematic").classList.add("running");
  } else {   // a WebGPU run can take a quarter of a second: keep the fast flow on screen for at least a second
    state.flowTimer = setTimeout(() => $("schematic").classList.remove("running"),
      Math.max(0, 1000 - (performance.now() - state.runStart)));
  }
  updateRun();
}

function feeds(arm, q, r, ov) {
  const T = window.ort.Tensor, f = { query: new T("float32", q, [1, 4, H, W]) };
  if (arm === "on") { f.reference = new T("float32", r, [1, 3, H, W]); f.overlap = new T("float32", ov, [1, 1, H, W]); }
  return f;
}

/** Loads (and warms up) one condition's model once; on failure it says so, forgets the attempt so a later
 *  click retries, and leaves the other condition usable. Resolves to the session, or null on failure. */
function ensureSession(arm) {
  if (!state.loading[arm]) {
    state.loading[arm] = (async () => {
      updateRun(); $("bar").style.width = "0";
      try {
        const warm = () => feeds(arm, new Float32Array(4 * H * W), new Float32Array(3 * H * W), new Float32Array(H * W).fill(1));
        const { session, backend } = await loadSession(window.ort, `model/refernet_${arm}.onnx`, pickProviders(navigator),
          (f) => ($("bar").style.width = `${Math.round(f * 100)}%`), warm);
        state.sessions[arm] = session;
        $("backend").textContent = `runs in your browser · ${backend === "webgpu" ? "WebGPU" : "WebAssembly"}`;
        return session;
      } catch (e) {
        delete state.loading[arm];
        if (currentArm() === arm) {
          $("result").className = "";
          $("result").textContent = `The ReferNet model could not load (${e.message}). Check your connection and choose the condition again to retry.`;
        }
        return null;
      } finally {
        updateRun();
      }
    })();
  }
  return state.loading[arm];
}

function drawHeat(map, querySrc) {
  $("map-img").src = querySrc; $("map-wrap").hidden = false;
  const c = $("heat"); c.width = 100; c.height = 64;
  const ctx = c.getContext("2d"), img = ctx.createImageData(100, 64);
  for (let i = 0; i < 6400; i++) {
    const v = map[i];
    img.data[4 * i] = 255; img.data[4 * i + 1] = Math.round(220 * (1 - v)); img.data[4 * i + 2] = 0;
    img.data[4 * i + 3] = v > 0.1 ? Math.round(60 + 150 * v) : 0;
  }
  ctx.putImageData(img, 0, 0);
}

function showResult({ p, ms, label, mismatch, arm, noMapNote }) {
  const truth = label === 1 ? "altered" : "genuine";
  $("result").className = "";
  if (mismatch) {
    $("result").innerHTML = `<div class="verdict REFER">REFER</div><p class="hint">${decideMismatch().reason}</p>
      <div class="kv"><span>Ground truth</span><span>${truth} · document of another design</span></div>`;
  } else {
    const d = decide(p, state.manifest.threshold), correct = (d === "REFER") === (label === 1);
    $("result").innerHTML = `<p class="hint">${arm === "on" ? "With reference" : "Without reference"}</p>
      <div class="pct" style="color:var(--${d === "REFER" ? "red" : "green"})">${percent(p)}<small>likelihood of alteration</small></div>
      <div class="verdict ${d}">${d}</div>
      <div class="kv"><span>Inference time</span><b>${Math.round(ms)} ms</b></div>
      <div class="kv"><span>Ground truth</span><span>${truth} · <span class="${correct ? "ok" : "bad"}">${correct ? "correct ✓" : "wrong ✗"}</span></span></div>`
      + (noMapNote ? `<p class="hint">The reference found no difference that raised the score, so no map is shown.</p>` : "");
  }
  $("truth-hint").textContent = `Ground truth: ${truth}`;
}

async function onRun() {
  if (!state.gate.tryEnter()) return;
  clearResult(); setBusy(true);
  try {
    const f = findSample(state.manifest, $("document").value);
    if (f.kind === "mismatch") { showResult({ label: f.sample.label, mismatch: true }); return; }
    const s = f.sample, arm = currentArm();
    const session = await ensureSession(arm);
    if (!session) return;
    const [q, m, r] = await Promise.all([png(s.query), png(s.masks), arm === "on" ? png(s.reference) : null]);
    const qt = buildQuery(q.rgba, m.rgba, W, H), ov = buildOverlap(m.rgba, W, H);
    const rt = r ? rgbToNormalizedCHW(r.rgba, W, H) : null;
    const { outputs, ms } = await runOnce(session, feeds(arm, qt, rt, ov));
    const correction = arm === "on" ? outputs.correction.data[0] : 0;
    const mapShown = arm === "on" && showMap(correction);
    showResult({ p: sigmoid(outputs.query_logit.data[0] + correction), ms, label: s.label, arm, noMapNote: arm === "on" && !mapShown });
    if (mapShown) drawHeat(differenceMap(outputs.local.data, downsampleOverlap(ov), state.manifest.head), s.query);
  } catch (e) {
    $("result").className = "";
    $("result").textContent = `This browser could not run the model (${e.message}). Try a recent Chrome, Edge, Firefox or Safari.`;
  } finally {
    state.gate.leave(); setBusy(false);
  }
}

async function init() {
  if (!window.ort || !window.UPNG) {
    $("result").className = ""; $("result").textContent = "The model runtime could not load. Try a recent Chrome, Edge, Firefox or Safari.";
    return;
  }
  window.ort.env.wasm.wasmPaths = ORT_DIST;
  state.manifest = await (await fetch("manifest.json")).json();
  fillDesigns();
  $("with").onclick = () => setCondition(true);
  $("without").onclick = () => setCondition(false);
  $("run").onclick = onRun;
  await ensureSession("on");
}

init();
