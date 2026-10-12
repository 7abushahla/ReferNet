# Claude consultation on the direct reference pilot

Date, 12 October 2026. Proposal reviewed at commit `b6848cc`.

The user requested an actual Claude consultation before continuation. Claude Code ran a fresh read-only session using only Read, Grep and Glob. The reported model was `claude-opus-5-5`; the session completed successfully in 255 seconds with 18 turns and no permission denials. No implementation or cluster submission was performed by the reviewer. The runtime implementation worker was held at a safe checkpoint during consultation.

## Coordinator assessment

The review supports provisional advancement of D, not final architecture selection or proven reference benefit. Its verdict is APPROVE WITH AMENDMENTS. This record does not silently amend the governing proposal or authorize additional jobs.

### Findings accepted

- The proposed adapter contains a local query readout that is absent from the reference-off arm (proposal lines 43-57). Therefore the currently proposed contrast estimates reference plus adapter, not reference information alone. A more tightly matched control retains the local adapter with structural zeros for unavailable reference/difference channels. This avoids passing a blank image through the encoder. Reference computation and functional input channels still differ, so do not claim identical active capacity or compute. Both arms would require fresh matched training.
- Query batch-statistics versus reference running-statistics normalization during training creates a potential comparison mismatch (proposal line 37). Even equal inputs need not give equal features. This is a real design concern, not empirical proof of a failure. The proposed query-statistics reuse is one candidate remedy, not an already validated implementation. It requires correct per-layer variance semantics, affine/epsilon handling, query-buffer preservation, mode restoration, and forward/gradient tests. Equal-input feature agreement would validate a mechanical property, not forgery detection.
- Compute and authenticate one image-only registration table per query/bank, reused across arms. Keep phase-specific membership controls and do not let validation queries fit shared bank products or acceptance thresholds. Coordinate validity and failure decisions must agree between arms. This avoids repeated registration without treating hashes as evidence of alignment quality.
- Choose and document one warp direction before implementation. Query-to-fold-safe-bank registration is closer to the measured D input route. Inverse context registration preserves query pixels but is not established as superior. Neither is a proven cause of better performance. Both require coordinate and visual checks.
- Diagnose reference use separately from accuracy, with predefined interventions and no test-driven model selection. Adapter magnitude alone does not establish useful reference information, because a local adapter can also learn from the query. A wrong-design intervention that changes alignment/zone as well as RGB is not a reference-only intervention and must be labeled accordingly.

### Qualifications and recommendations not accepted as stated

- Do not lower the bridge gate from the agreed accuracy/AUC targets by adding three errors merely because one existing learning curve fluctuates by that amount. Epoch-to-epoch variation is not a confidence interval or an estimate of independent training variability. Borderline outcomes warrant review or planned replication, not post hoc acceptance-threshold relaxation.
- Similar validation performance does not establish statistical non-inferiority. That requires a justified predeclared margin, an uncertainty method, and adequate evaluation evidence. Use descriptive wording such as comparable observed fold-0 validation scores until then. Reference sensitivity is not equivalent to reference benefit or design understanding.
- McNemar calculations on repeatedly inspected, checkpoint-selected validation predictions are descriptive only, not confirmatory model-comparison inference. Parameter count does not establish CPU latency; R's fewer input pixels also matter. Claims based on padding comments, all attacks being within zones, or median background content remain hypotheses unless checked against actual products.
- Parallel seed-42 on/off pilots may save time after a shared contract passes gates, but this changes the currently sequential launch scope. The suggested additional seed-43 pair is not automatically authorized and is not part of this consultation. No launches occurred.
- The claim that reference benefit would matter only in Protocol B is too strong. A rigorously demonstrated A benefit could matter, but it would not establish unseen-design generalization. Conversely, no detectable A advantage near ceiling would not refute the study hypothesis.

### Resulting direction

Retain D as the provisional family, the undifferenced query path, and the three-input study concept. Revise the comparison to control the added local query head, settle the normalization and coordinate policy, and freeze shared registration decisions before implementation. Keep the native Protocol A anchor and encoding limitations explicit. Do not restart a broad architecture search or advance to later datasets from this review alone. The next artifact is a bounded specification amendment, followed by existing synthetic and allocated-GPU gates, not an immediate sweep.

## Claude review, verbatim

Code and evidence have been checked; next I'm writing up the review. One finding from the D log: training BCE is still 0.068 at epoch 100, and validation accuracy moves between 0.964 and 0.977 across epochs 79–100. So the query path hasn't saturated, and selected-epoch scores carry about ±3 images of noise.

# Review: Direct Reference Pilot proposal (b6848cc)

**Verdict: APPROVE WITH AMENDMENTS.** Choosing D is a reasonable engineering bet. The residual design is close to minimal. But four items should change before any code is written: the comparison isn't symmetric during training, the reference arm gets extra query-only capacity, the bridge gate depends on a single image, and the run order costs time for no gain. None of these adds a week.

## 1. Should D be advanced?

**What the evidence shows.** On fold 0, seed 42, at the selected epochs, D made 5 errors, N made 11 and R made 0, out of 222 images. A rough paired comparison of discordant errors from the reviewed error lists:
- **D vs N:** 3 errors only D made, 9 only N made (D and N share the aze_97 and srb_87 copy misses). Exact McNemar p ≈ 0.15.
- **R vs D:** 0 vs 5, p ≈ 0.06.
- **R vs N:** 0 vs 11, p ≈ 0.001.

So R is very probably better than N. Neither D>N nor R>D is established on this fold. The epoch-to-epoch swing in D's log (`log.csv:80`, `:101`, 0.977 at epoch 79 vs 0.964 at epoch 100) is about the same size as the D–R gap. Picking the best of 100 validation epochs also inflates every row.

**Why D is still the right bet, on engineering grounds:**
- D already reads a 512×800 query plus zone. The bank products use exactly that grid (`bank_similarity_v1`, canvas `[512, 800]`, readiness doc lines 61–63).
- R's 299-square input squeezes a roughly 1.6:1 document in different amounts per axis. Extending R would need a new zone/reference geometry and a new two-class correction interface. That's more code in a 3-day window.
- R has twice the parameters (23.5M vs 11.3M), which matters for the CPU latency claim.
- R's perfect score doesn't speak to architecture. SIDTD has a deterministic link between JPEG encoding and label (`paper-plan.md:300`), and R also differs from D in recipe, resolution, initialization and inputs.

N's 0 false positives vs D's 2 out of 100 genuine images isn't a meaningful difference. N would need a reference route and normalization design built from scratch.

**The claim to make:** "provisionally advanced for extension readiness". Not "D is the strongest family". The proposal's §1 wording is already appropriately careful.

## 2. Is the residual design sound?

### Must fix before implementation

**F1. The reference comparison isn't reflexive during training.**
- During training, the query runs with train-mode BatchNorm, so stem, layer1 and layer2 are normalized by statistics of the current 8-image batch. The reference uses the running buffers (spec lines 37–38).
- So if the reference were an exact copy of the query, the stride-8 features would still differ during training, but would match at evaluation. The adapter learns from a difference channel containing a batch-dependent per-channel offset that vanishes at test time.
- I'm not calling this an established bug. But the project has already shown BN-statistics mismatch to be severe with this very encoder: control AUC moved from 0.4586 to 0.9789 when only the normalization context changed (`paper-plan.md:305`).
- **Fix (recommended):** in training, normalize the reference pass with the query's per-module batch mean and variance captured during the forward pass, treated as constants, with no buffer update. Keep running buffers for both streams at evaluation.
  - This leaves D's query computation exactly as it is.
  - It makes the two streams apply the same function in both phases.
  - Reference-off is unaffected.
- **Alternative:** put stem through layer2 BN in eval mode for both arms. That's simpler, but it changes the base recipe and leaves LEVIR statistics frozen in those layers.
- **New gate:** feed reference = query and require the adapter's |difference| ≈ 0 in a train-mode forward pass. The current design would fail this.

**F2. Reference-on gets extra query-only capacity.**
- The adapter reads the projected query features at stride 8, followed by two conv+GN blocks and masked max pooling. That is a second, local query readout that the off arm doesn't have.
- An improvement could therefore come from multiscale capacity rather than from the reference image. That conflicts with "the reference is the only difference" (`paper-plan.md:83`).
- **Fix:** give the off arm the same adapter, with the projected-reference and difference channels held structurally at zero. These are constants at the adapter input, not a blank image pushed through BatchNorm, so N's failure mode doesn't recur.
  - Both arms then start with identical logits, since the correction is zero-initialized in both.
  - They differ only in the reference channels.
  - This adds no runs. It does mean the bridge is "D + geometry package + local head", which matters little because the bridge is a package confirmation already (§3, line 76).
- If you reject this, state the contrast honestly as "reference + local adapter".

**F3. Freeze one transform table and share it between arms.**
- Registration should be precomputed once per (query, bank): training queries against their opposite half, validation queries against `enrollment`. Hash it, and have both arms read the same table.
- This guarantees the same alignment decision and the same failure population by construction.
- It also avoids SIFT+RANSAC on every training step (about 178k fits without a cache).
- And it removes any dependence on RANSAC determinism across data-loader workers. Verify that rather than assume it.

### Choices that are fine, with caveats (not bugs)

- **Zero-initialized final linear:** standard and sound. Gradients reach the hidden layers after the first step, and the proposal's check for that is right.
- **Reference neglect:** less acute than I first expected. D's training BCE is still 0.068 at epoch 100 under augmentation, so the query path isn't saturated and the adapter does get gradient. It still needs predeclared diagnostics (see §6).
- **GroupNorm over zeroed cells:** per-image statistics then depend on overlap and letterbox fraction. For SIDTD the padding is a few pixels (`rectify.py:3-4`), so this is minor. Masked mean+max excluding invalid cells is correct.
- **Masking scope:** "mask comparison features" (line 45) is ambiguous. Say explicitly whether projected-query channels outside the overlap are masked.
- **Shared gradients:** reference gradients will change the shared stem through layer2. That's expected and is part of the intervention, which is why disabling the reference in a trained model stays a secondary ablation.
- **Legitimate personal variation:** the zone channel is the right tool. Note that every SIDTD attack sits inside zones, where the median only carries background or guilloche under blurred text. Plausible reference signal is limited to background continuity. Don't expect much gain on copy attacks.
- **Bank size shift:** training-half banks have 21–37 contributors versus 56–68 for `enrollment`, so training medians are noisier. Record this as a known train/validation shift.

## 3. Bridge geometry

The fold-safe bank target, zones and coherent jitter are **necessary, not bundled redesign**. The legacy `register_design` builds its target from all genuine images, including validation and test (`register.py:74-83`), and the reference arm needs a jittered reference to stay aligned with the query.

The **warp direction** is a matter of judgment. "Preserve query pixels" is a deployment-elegance preference with no evidence behind it. D's evidence was produced with a resampled query, because legacy canvases were warped.

**Lower-risk alternative:** warp the query (and its validity mask) into the selected bank's frame with the same SIFT similarity, and use the bank's median, zone and valid planes untouched.
- It's closer to D's evidence.
- It touches 2 planes instead of 4.
- The median isn't resampled a second time.
- It matches the paper's "rectify to canvas" story for FantasyID.

For SIDTD the transforms are near-identity (`paper-plan.md:301`), so either direction gives nearly the same tensors. **Pick one today and don't revisit it.** I lean towards the forward query warp for deadline risk. Either needs the same synthetic coordinate tests and a visual overlay check of about 10 per design.

## 4. Fix now, gate cheaply, defer

**Before implementation:** F1, F2, F3, and the warp direction. Also replace the bridge gate.
- Accuracy ≥ 0.97 means at most 6 errors out of 222. D had 5, and its own epoch-to-epoch swing is about 3 images. A 1-image margin is noise.
- **Suggested gate:** AUC ≥ 0.99, errors ≤ D's 5 + 3, no copy-AUC collapse, and registration failures ≤ a small predeclared count. Treat it as a review input, not an automatic stop.

**Cheap gates, in one allocation:**
- The proposal's G4–G9 essentials.
- The reflexivity test (F1).
- Bit-identical initial logits for on vs off (F2).
- Transform-table hash parity (F3).
- 16-image overfit in both arms.
- 32-image train/validation smoke plus fresh-process replay.
- The visual overlay check.

Reuse the existing read-guard infrastructure. Don't build new phase-specific instrumentation unless it's nearly free. That is where §5 drifts towards bureaucracy.

**Defer:**
- Auxiliary loss on the adapter alone (add only if neglect is observed; it changes the objective).
- Multilevel taps.
- Specimen vs median comparison.
- Training for missing context.
- CPU timing of the adapter (parallel track).

## 5. Sequential or parallel?

**Run both arms in parallel** from one frozen source, after both pass the shared gates. Sequential order saves no GPUs that matter (GPUs are available) and costs about 4–5 hours plus review latency on a 3-day clock.

Parallel launch stays scientifically clean if the interpretation rule is written down before launch:
- If the bridge fails competence, the reference-on result is reported as uninterpretable for the reference contribution. It is never used to rescue the family.

This is two bounded runs, not a sweep. Even so, the parallel-launch change and the seed pair below both need the user's explicit approval before submission.

If you approve it, a paired seed-43 on/off run launched at the same time (4 GPUs total) would do more for interpretability than anything else. The spec plans that confirmation anyway (classification-centered spec, line 202).

## 6. What the results can and cannot show

**Can establish (fold-0 development evidence only):**
- Whether the fold-safe query+zone bridge keeps competence.
- Whether adding the reference image, at matched capacity, changes source-validation errors.
- Whether the model actually uses the reference, via predeclared checks on the selected on-arm checkpoint:
  1. distribution of |correction|;
  2. correction under a wrong-design bank and an alternate same-design training-half bank (genuine training material only, so fold-safe);
  3. correction-zeroed ablation.

With a 5-error baseline, the realistic outcome is that the arms can't be told apart. Predeclare that result as **"non-inferior, reliance = X"**, not "the reference doesn't help". Superiority would only mean anything on unseen designs (Protocol B).

**Cannot establish:**
- Seed or fold stability.
- Copy-attack performance (16 cases; 0–3 misses spans a wide interval).
- Independence from the encoding shortcut. Both arms read the same class-linked encoding, so a reference gain doesn't show that encoding was removed or that the model relies only on alteration evidence.
- Component-held-out generalization (183 of 222 validation images share a component with training).
- Test performance or CPU latency.

**Smallest interpretable next step:** implement with F1–F3 and the chosen warp direction, pass the single-allocation gates, then run on/off at seed 42 in parallel with the predeclared interpretation table and reliance diagnostics.
- **Success:** the bridge meets the revised gate, the on arm is non-inferior, and the reliance checks show a nonzero, design-specific correction.
- **Failure:** the bridge loses competence, or the on arm degrades. Either way, inspect before running anything else. Don't automatically swap to another family.
