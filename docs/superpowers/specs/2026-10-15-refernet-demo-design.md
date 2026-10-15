# ReferNet demo website: design

Date: 15 October 2026. Status: approved in brainstorming, awaiting spec review.

## Purpose

A public demo of ReferNet, hosted on GitHub Pages from this repository (to be renamed `ReferNet` and made public
in a separate task). It shows how the system would behave at a first-line counter: a visitor picks a design
reference and a presented document from a fixed library, chooses whether ReferNet uses the reference, runs the
paper's deployment model in the browser, and sees one result, a percentage, Refer or Clear, and the inference time.

Audience: the GDRFA award judges and readers of the paper who follow its code link. Success means the demo runs
the exact deployment model of the paper (scores match PyTorch), works without a server, and is honest about what
is precomputed and what is not.

## Decisions taken

| Topic | Decision |
|---|---|
| Hosting | GitHub Pages, deployed from `site/` by a GitHub Actions workflow |
| What runs live | The network only, in the browser with ONNX Runtime Web (WebGPU, else WebAssembly) |
| What is precomputed | Registration (SIFT + RANSAC) of every sample to its design reference, with the paper's code |
| Model | Fold-0 deployment pair of Protocol A, one ONNX graph with a reference switch, FP32 (about 45 MB) |
| Sample library | All 18 designs of the paper (10 SIDTD, 8 IDNet); per design 2 genuine and 2 per attack family, picked by a fixed seed, about 110 samples, failures kept |
| Page layout | Left-to-right storyboard of four steps (stacked on phones) |
| Condition | Chosen by a toggle in step 3, "With reference" or "Without reference"; one condition per run |
| Result | One percentage (likelihood of alteration), Refer or Clear at 50%, inference time, ground truth with a tick or cross |
| Difference map | Shown only with the reference, computed live from the same run, labeled as not part of the paper |
| Stack | Plain HTML, CSS and JavaScript, no build step; ONNX Runtime Web from a CDN with a pinned version |

## Page

Header: "ReferNet · reference-assisted document forgery detection", one line of instructions, links to the paper,
the code, and "How it works".

1. **Design reference.** Drop-down of the 18 designs ("Azerbaijan passport · SIDTD", "Arizona license · IDNet",
   ...). Shows the design reference (median of genuine documents). Hover shows the personal-data zone.
2. **Presented document.** Drop-down grouped under the chosen design: Genuine #1–2, then each attack family #1–2
   (inpaint-and-rewrite, copy-and-replace, and for IDNet in-place text replacement). A last group, "Another design
   (no reference match)", lists a few documents of other designs. The ground truth stays hidden until the run.
3. **ReferNet.** A small animated schematic of the architecture (document and design reference into the shared
   early encoder, then "deeper stages, judge alone" and "compare document vs reference", meeting at a "+" before
   Refer / Clear). The toggle "With reference / Without reference" greys out the reference branch when off. During
   a run the active arrows animate. Below: the Run button and the backend in use ("runs in your browser · WebGPU").
4. **Decision.** The condition used, a large percentage ("97% likelihood of alteration"), the REFER or CLEAR banner,
   the inference time, and the ground truth with a tick or cross. With the reference, the difference map is
   overlaid on the document, followed by the altered-field box of the ground truth. No logits or loss terms are shown.

**How it works.** Three short paragraphs from the paper: the design reference and the personal-data zone; the
two paths and how the reference corrects the single-image decision; what is precomputed (registration) and what
runs live (the network). A note that times are measured on the visitor's device, whereas the paper's 280 ms is end
to end on a laptop CPU. A highlighted note: "The map in step 4 is not part of the paper. It is shown for
demonstration only. It is computed in your browser from the same run, from the comparison features weighted by the
correction layer, to indicate which region drove the reference's correction, and it has not been evaluated as a
localization method." No architecture figure in this section.

**Footer.** Dataset credits and licenses (SIDTD CC BY 4.0, MIDV-2020 images CC BY-SA 2.5, IDNet CC0), "registration
precomputed with the paper's code", "fold-0 deployment model, FP32", "scores verified against PyTorch".

## Components

### `demo/export_onnx.py`

Loads the fold-0 OFF and ON checkpoints with `build_direct_reference_model` (as in
`scripts/bench_cpu_deploy.py`) and exports one ONNX graph per condition, or one graph with a boolean input that
bypasses the local path, whichever exports cleanly. Outputs: `query_logit` (ℓ_q), `correction` (Δℓ, zero without
the reference), and `comparison_map` (the 32×64×100 local features before masked pooling). Inputs: the
four-channel query `[x; z]`, the four-channel reference `[r; z]`, and the overlap mask, all 512×800. The OFF
condition uses the OFF checkpoint, so its scores equal the paper's ReferNet (OFF).

### `demo/prepare_demo.py`

Runs where the data are (SIDTD locally; IDNet on the AUS cluster). For each of the 18 designs:

- Builds the design reference as in the paper: SIDTD from the fold-0 training documents of the design (Protocol A),
  IDNet from the 100 genuine images of its Protocol C reference set.
- Picks samples with a fixed seed from images the deployment model never trained on (SIDTD fold-0 test partition;
  IDNet images outside the reference set), 2 genuine and 2 per text attack family.
- Equalizes compression as in the paper, letterboxes to 512×800, registers each sample to its reference, and writes
  `query.png`, `reference.png` (warped into the query frame), `zone.png`, `overlap.png`.
- Tries each "another design" sample against the selected references and records the registration outcome.
- Scores every sample in PyTorch (both conditions) and stores ℓ_q, Δℓ and p in `manifest.json` for verification.

`manifest.json` lists designs (name, dataset, document type, reference image, zone image) and samples (design,
family, index, ground truth, altered-field box, file paths, registration status, PyTorch scores).

### `site/`

`index.html`, `style.css`, `app.js`, `manifest.json`, `model/refernet*.onnx`, `samples/<design>/<id>/*.png`.
`app.js` loads the manifest, populates the drop-downs, assembles the input tensors with the paper's ImageNet
normalization, runs ONNX Runtime Web, times the run with `performance.now()` after one silent warm-up run, and
draws the result and the difference map.

**Difference map.** For the comparison map F (32×64×100) and the correction head weights w = [w_mean; w_max]
(64 values), the map is M(x, y) = Σ_c (w_mean,c / |Ω| + w_max,c) F_c(x, y) inside the overlap Ω, clipped at zero,
upsampled to the document, and overlaid as a heat map. The head weights are exported in the manifest.

### `site/verify.html`

Runs every sample in the browser for both conditions and checks that p matches the PyTorch value in the manifest
to within 1e-3 and that every Refer/Clear decision is identical. Prints a pass/fail table.

### `.github/workflows/pages.yml`

On push to `main`, uploads `site/` as the Pages artifact and deploys it. The model is committed as a regular file
(under GitHub's 100 MB limit); Git LFS is not used, because Pages does not serve LFS content.

## Error handling

- No WebGPU: fall back to WebAssembly and show the backend actually used.
- First visit: progress bar while the model downloads; Run stays disabled until the model is loaded and warmed up.
- Document of another design: registration failed, so there is no reference context. Both toggle states show
  REFER, "no reference context, sent to the second line", as the deployed system does (the zone is an input of
  both conditions).
- ONNX Runtime Web unavailable: a plain message naming supported browsers.

## Testing

1. Parity: `verify.html` passes on all samples in Chrome before deployment.
2. Builder: a pytest checks the manifest schema, that every file exists, the sample counts per design, and that no
   sample comes from a training partition or a reference set.
3. Manual: Chrome, Safari, and Firefox on desktop and on a phone; steps stack on mobile; the mismatch path works.

## Out of scope

Uploading one's own image; live registration in the browser; showing both conditions side by side; INT8 models.

## Related task (separate)

Rename this repository to `ReferNet`, remove third-party paper texts (`evidence/*.txt`) and internal notes from the
history, make it public, enable Pages, and update `\repourl` in the paper.
