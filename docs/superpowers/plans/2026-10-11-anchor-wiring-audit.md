# Protocol A anchor wiring audit

**Goal:** Verify that the borrowed ChangerEx components and our classifier are connected as intended before interpreting the running anchor pilots.

**Study authority:** Section 3.2 of `/Users/hamza/Documents/GitHub/GDRFA-Award-Paper/main.tex`, starting at line 308, is the manuscript identified by the user. `refer-or-clear` is the implementation repository. The four strands guide the audit, but they are not four architectures to implement. Single-image benchmarks establish the anchor; reference-conditioned work motivates the subsequent matched comparison; deployment work informs the readout and eventual cost measurement; change detection supplies the implemented backbone and fusion.

**Scope:** Native execution, no subagents. No architecture replacement, dataset change, training submission, test-set evaluation or fold sweep. The four running pilots remain untouched. A short synthetic-input diagnostic runs on one GPU compute node from an immutable source package.

## Checks and acceptance criteria

- [x] Compare the loaded three-channel, two-output official Open-CD modules against our four-channel, one-output wrapper. Both use the same LEVIR checkpoint and fixed running BN statistics. Compare decoder outputs on identical features separately from the complete backbone-plus-decoder path, at 128 by 160 and 512 by 800. Predeclare float32 tolerances of `atol=1e-4, rtol=1e-4`; save differences before asserting. This tests weight conversion, not the subsequent training-time BN-policy change.
- [x] Require precisely the four referral parameters to be missing from LEVIR loading, no unexpected keys, and a zero-initialized fourth-channel stem. Require optimizer groups to cover trainable parameters exactly once.
- [x] Verify that the exchange-off control ignores all supplied reference planes. Measure image-only and dense-only gradient connectivity separately. The expected image path includes encoder, decoder, map projection and referral head; the expected dense path excludes the referral head. No optimizer steps are allowed. Verify unchanged parameters and buffers.
- [x] Test the existing claim that identical inputs necessarily yield zero FDAF differences using the official warp and a nonzero displacement. If false, correct the explanatory claim, not the running model.
- [x] Record software/source/checkpoint hashes and synthetic-only scope. Commit and push the diagnostic and its evidence. Keep the existing classification-only and input-preparation pilots as the performance evidence.

## Files and execution

Create `scripts/audit_anchor_wiring.py`, a standalone assertion-based diagnostic, and `scripts/cluster/audit_anchor_wiring.sh`, following the existing diagnostic Slurm resource pattern. Source modules remain unmodified except for any demonstrably false explanatory text. Parse Python and check shell syntax locally. Run the full numerical assertions on a GPU node. Save results under `results/diagnostics/anchor-wiring-<job>/summary.json` even if a check fails.

## Interpretation boundary

A passing conversion check rules out a specific implementation error, not an inadequate architecture or training recipe. Gradient connectivity is not evidence that joint optimization helps. Synthetic inputs cannot establish document accuracy, foreground localization quality, or a causal explanation for the copy-and-replace deficit. Compare external classifier designs only to identify a specific next ablation after the current pilots finish.

## Execution record

The user reaffirmed that the anchor remains the priority. This audit deliberately excludes broader literature expansion, transfer-protocol implementation and additional training. Existing branch-calibration and dense-gradient diagnostics are not repeated.

Job 37938 passed checkpoint-key and optimizer coverage checks, then stopped in the diagnostic constructor because the upstream backbone's `train` override makes `eval()` return `None`. The harness incorrectly chained `.eval().cuda()` on that object. The correction separates device placement from `eval()`. This is a diagnostic-harness error, not a failure in the training model, which wraps the backbone and does not depend on its evaluation method's return value. The failed report is preserved; numerical parity remains pending until the rerun.

Rerun **37942** completed in 58 seconds with exit code 0:0, using source `a64542e62e9e295fcfa17161d426e2e454f8f208`. All checks passed. At both resolutions, backbone features were bit-identical; maximum dense-logit conversion differences were `1.34e-5` and `1.91e-5`, below the predeclared tolerance. The control's outputs were bit-identical after replacing supplied reference planes. Image loss reached all 96 trainable parameter tensors; dense loss reached the 92 encoder/decoder tensors and none of the four referral tensors. Parameters and buffers remained unchanged. The warp counterexample disproves the old unconditional zero-difference explanation. Only explanatory text was corrected in the production model.

Results are in `results/diagnostics/anchor-wiring-37942/summary.json`; the failed harness report is retained in `results/diagnostics/anchor-wiring-37938/summary.json`. The source comparison requested by the user is `evidence/section-3-2-architecture-lineage-audit-2026-10-11.md`, including the supplied CheckSim PDF. No model training or dataset changes resulted from this audit.
