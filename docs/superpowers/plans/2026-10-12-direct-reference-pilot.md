# Direct Reference Pilot Implementation Brief

> **For agentic workers:** Use `superpowers:executing-plans` within assigned ownership. This brief publishes the pinned model and data boundaries first. Coordinator owns the subsequent isolated runner and controlled-execution plan. No API changes without coordination. Maxwell owns the separately approved pure-library Task 1.

**Goal:** Implement the approved direct-query classifier with a matched optional-reference local correction and a support-safe query-canvas bridge.

**Architecture:** Fresh verified D-running pilot computation is an internal base only. A shared stride-eight projection and local head exist in both modes. The bridge preserves query pixels and inverse-warps authenticated training context into query coordinates.

**Tech Stack:** Existing PyTorch, Open-CD, OpenCV, pandas, and pytest; no dependency changes.

**Spec:** `docs/superpowers/specs/2026-10-12-direct-reference-pilot-design.md` and the coordinator's governing amendment. Existing provenance gates remain in `docs/anchor-reference-integration-readiness-2026-10-12.md` and the matched-reference preparation document.

## Global Constraints

- Only assigned files may change. Preserve existing `D`, `D_running`, `D_running_pilot`, N, and R contracts/checkpoints.
- Public architecture is `anchor_direct_reference_ia18_v1`; reject legacy checkpoint identities. No local real weights/images, model runs, SSH, downloads, submissions, or commits.
- Production input is 512 by 800; model-only synthetic grids divisible by 32 are allowed. Actual installed-stack geometry/gradients/BN/replay require an approved GPU allocation.
- Off keeps the same query projection/local head, zero R/difference slices, and no optional-reference inspection. Both modes start with identical base/adapter tensors and zero correction.
- Training uses sequential full-query then available-reference-prefix shared BN; evaluation freezes both. Labels/annotations/provenance never enter forward.
- Keep official membership, raw-R distinctions, strict attack probability `> 0.5`, genuine ties, original selection/scheduler/metrics, and `training_bit_reproducible=false`.
- Only bounded preparation/library testing is presently authorized. Integrated diagnostics and either 100-epoch run require separate scope approval and independent evidence review.

## Review Focus

1. Malformed optional reference in off must remain uninspected; invalid common tensors must fail before BN mutation (Task 1).
2. Empty overlap/unavailable rows must yield correction zero without encoding reference or permitting automatic clearance (Tasks 1/2).
3. Shared training BN must update the reference prefix only, while companion evaluation and checkpoint replay leave all buffers immutable (Task 1 and coordinator integration).
4. Untrusted registration JSON, stale query bytes, relocated paths, and wrong-half banks must fail closed; caches retain transitive exclusions (Task 2).
5. Stage-one validation pixel opens and legacy/mixed-arm checkpoints must be rejected, even when approved train/val files are already staged (Task 2 and coordinator integration).

## Task 1. Pure Model and Synthetic Tests

**Owner:** Maxwell. **Create only:** `src/refer_or_clear/models/direct_reference.py`, `tests/test_direct_reference_model.py`. Do not modify `anchor_changer.py` or its tests.

**Authoritative interfaces**

```python
DirectReferenceClassifier(base: AnchorChanger, *, reference_enabled: bool, adapter_seed: int = 42)
build_direct_reference_model(levir_path: Path, expected_sha256: str, *, reference_enabled: bool, head_seed: int = 42, adapter_seed: int = 42)
forward(query, *, overlap, reference, context_available) -> dict[str, Tensor]
```

- Factory calls verified `build_anchor_changer("D_running_pilot", levir_path, expected_sha256, head_seed)` internally. Preserve official hash verification/restricted loading and isolated initialization RNG. Expose new public architecture, not an old execution/checkpoint contract.
- Inputs are FP32 query `(B,4,H,W)`, binary FP32 overlap `(B,1,H,W)`, bool availability `(B,)`, and optional normalized FP32 reference `(B,3,H,W)`. Outputs `attack_logit`, `query_logit`, `correction_logit` are `(B,)`. Append query zone internally to reference. Validate before training traversal; off never examines reference, on ignores unavailable rows and validates required reference before mutation.
- Capture query stage two and deepest feature during one full encoder traversal. Reference traverses only shared stem/layer1/layer2 on available rows with nonempty feature overlap. Use the approved 128→32 projection, 98→32→32 GN8/ReLU head, masked mean/max64, and zero final64→1. Mask Q/R/difference outside overlap. Off R/difference are zeros, not `abs(Q)`.
- `parameter_groups` retains the base encoder group and includes base classifier plus projection/local-head parameters exactly once in the new group, identically in both modes. Disclose inactive off input slices rather than excluding the adapter.

- [ ] Write failing synthetic tests for matched base/adapter initialization, unchanged caller RNG, initial score identity, off local-head execution/garbage-reference immunity, masked channels, empty support, all-unavailable and mixed batches, output shapes, and rejection before any BN mutation.
- [ ] Run `OMP_NUM_THREADS=1 /private/tmp/roc-anchor-dev/bin/python -m pytest tests/test_direct_reference_model.py -q -k 'not actual_direct_reference_gpu_integration'`; capture the specific missing implementation or behavioral failure before implementation.
- [ ] Implement only the pinned wrapper/factory. Assert stage-two shape `(B,128,H/8,W/8)` on synthetic fixtures. Test sequential BN counters, no layer3/4 reference pass, finite first-step final gradients and second-backward hidden/reference activation after one optimizer step.
- [ ] Repeat the same focused command. Verify eight queries at evaluation batch sizes 1/2/4/8 with varied companions/positions, immutable buffers, optimizer coverage, and independent matched storage. Optional equal-full-batch-prefix equality is a mechanics test only.
- [ ] Add `test_actual_direct_reference_gpu_integration` for later allocated execution at 512×800, including official fresh initialization, 64×100 stage two, forward/backward, BN counters, invariance, and checkpoint replay. Do not run or claim this GPU gate locally. Report fresh red/green counts and freeze ownership for review.

## Task 2. Authenticated Bridge and Dataset Tests

**Create only:** `src/refer_or_clear/data/direct_reference.py`, `tests/test_direct_reference_data.py`. Task 2 can proceed alongside Task 1 after coordinator confirms ownership; it consumes only the pinned tensor boundary.

**Chosen data interfaces**

```python
build_registration_table(rows: pd.DataFrame, raw_root: Path, enrollment_manifest: dict, *, preprocess: dict) -> dict
load_registration_table(path: Path, expected_sha256: str, *, rows: pd.DataFrame, enrollment_manifest: dict, preprocess: dict) -> dict
DirectReferenceDataset(rows: pd.DataFrame, raw_root: Path, enrollment_manifest: dict, registration_table: dict, *, reference_enabled: bool, train: bool, augment: dict)
```

- `rows` are authenticated phase-authorized membership from `load_role_rows`, not arbitrary registry-supplied membership. Coordinator phase guards supply train-only rows first and allow declared smoke validation rows only after training gates pass. Metadata loading/staging is not pixel-read permission. Never permit test/evaluation roles through this new consumer, even though the existing bank resolver accepts them.
- `preprocess.phase_authority` supplies an external `{path, sha256}` binding to `direct_reference_phase_v1`. Bind phase, source/recipe/records/split/enrollment hashes and sorted disjoint authorized train/validation IDs; selected rows must belong to those roles. Coordinate/overfit authority cannot contain validation IDs. Smoke authority contains exactly32train/32val and binds both-mode `direct_reference_training_gates_v1` receipts with matching identities/official weights, correct arm, training-only access, passing coordinate/model gates, and the inherited16image overfit thresholds. Training authority additionally binds both-mode `direct_reference_smoke_gates_v1` receipts with matching identities,32train/32val, one epoch and passing fresh-process replay. Authenticate these receipts before any pixel reads. This is a runner-to-loader phase boundary, not a replacement for independent scientific review or coordinator launch authorization; a self-declared dataset boolean is insufficient.
- Manifest/products must pass existing `verify_products` against pinned manifest SHA `7691452ecb4d160e8767d277ba3ccf209355f16b3f49f85e52ff596c8a2719e8`. Historical product paths are relocated only under explicitly authenticated output-root authority. Dataset validates supplied table structure/bindings again; a plain dictionary is not evidence of authentication.
- Table schema `direct_reference_registration_v1` binds records/split and runtime/preprocess/OpenCV/SIFT versions, query ID/byte SHA, fold/design/role/bank, selected product hashes, fit status/reason/inliers, and finite validated query-canvas→bank-target similarity. Check every selected query's byte hash before use. Cache failed fits. Generation is separate from augmentation because existing `_fit` changes OpenCV RNG; do not lazily fit in dataset workers.
- Reuse `to_canvas`, BGR `_features`, `estimate_similarity`, existing `MIN_INLIERS=40`, and similarity validation. Both canvases are 512×800. Invert the canvas transform directly; target versus median and `target_valid` versus `valid` remain distinct. No new quality bound, identity/global fallback, or query warp.
- Train bank lookup supplies the training component and uses its opposite half. Validation supplies only enrolled design and `training_component=None`; query labels/attack family/source/donor/annotations cannot route or fit context. Transitive exclusion applies to the table and all context caches.
- Each item returns `query` FP32 `(4,512,800)`, `overlap` binary FP32 `(1,512,800)`, scalar-bool `context_available`, integer `label`, and `meta`. On also returns `reference` FP32 `(3,512,800)`; unavailable on rows may carry an inert transport tensor that the model never encodes. Off omits this key and performs no median pixel reads. Default batching produces the pinned model shapes.
- Metadata contains image/design ID, role, offline attack family, fold/bank, registration key/table hash, availability reason, and bound product/query hashes. Keep loss/evaluation metadata outside forward. Empty pixel or stride-eight overlap zeros zone and availability in both arms, retains the original example/label, and records referral.
- Apply the same isolated query RNG draws and coherent D affine/jitter to query/zone/reference/overlap. Query photometry remains strength0.2; no extra reference photometry. Keep BGR through geometry, convert to RGB for normalization afterward. Do not wrap `AnchorQueryDataset`, whose legacy geometry/global RNG differs.

- [ ] Write failing tests for inverse translation/rotation/scale, non-square letterboxing, padding, partial/empty overlap, BGR grayscale semantics, all-plane affine/jitter coherence, and exact paired query/zone/overlap tensors.
- [ ] Write read-spy/mutation tests for authenticated relocation, forged/stale JSON/query/product hashes, wrong-half/held-out transitive dependencies, no annotation-based validation routing, cached failures, no off median opens, and unchanged query RNG across modes.
- [ ] Run `OMP_NUM_THREADS=1 /private/tmp/roc-anchor-dev/bin/python -m pytest tests/test_direct_reference_data.py -q` and record red evidence. Implement the minimum isolated module, then repeat for green. Use synthetic images/products only.
- [ ] Add phase read-guard tests showing staged validation files receive zero pixel opens during coordinate/overfit and only declared validation opens during gated smoke. Test roles rejected before pixel access. Coordinator supplies the runner transition, not a dataset boolean waiver. Report counts and freeze for review.

## Task 3. Isolated Runner and Checkpoint Contract

**Coordinator-owned integration. Create:** `src/refer_or_clear/direct_reference_pilot.py`, `configs/direct_reference_pilot.yaml`, and `tests/test_direct_reference_pilot.py`. These consume the reviewed Task 1 and Task 2 interfaces. Do not expand old pilot architecture identifiers to accept this model.

- [ ] Pin a new `direct_reference_pilot_v1` execution/checkpoint schema. Bind architecture, arm, original initialization, source/configuration, records/split, enrollment manifest, registration table, authorized phase membership, and preprocessing hashes. Reject legacy, mixed-arm, missing-authority, and diagnostic-to-long-training checkpoint reuse before loading tensors. Replay alone may load the selected same-contract checkpoint.
- [ ] Preserve D's classification recipe: seed42, batch8, evaluation batch1, 100 epochs, FP32 without AMP/TF32, AdamW backbone1e-4/new1e-3 with weight decay0.05, existing warmup/cosine rule, clip norm5, no accumulation, training drop-last and complete validation. Keep classification-only loss. Coherent geometry is the already declared bridge change, not a stream-matched claim against historical D.
- [ ] Import `make_pilot_optimizer`, `pilot_scheduler`, `summarize_predictions`, `select_phase_rows`, and `compare_replay` from `pilots`, plus `selection_key` from `train`, wherever their contract-independent behavior applies. Explicitly name the borrowed D recipe without assigning the new run an old D architecture/checkpoint identity. Implement fresh four-input dispatch, phase checks, checkpoint validation, and the short training/evaluation loop. Do not call the old single-input runner or reinterpret its gates.
- [ ] Write failing synthetic tests for both-mode dispatch, finite image loss/gradients, sample-weighted accounting, step/scheduler counts, accuracy-first/AUC-second/earliest-tie selection, strict altered probability `>0.5`, unavailable examples retained, and complete all-image metrics. Availability and subgroup summaries supplement rather than filter the primary result.
- [ ] Test the phase state machine before any pixel access. Coordinate/overfit use training membership only. Smoke permits exactly the declared32train/32val only after both-mode training gates pass. No test role is accepted. Fresh initialization, optimizer, and RNG reset between phases/modes must be asserted, not inferred from filenames.
- [ ] Implement only after RED. Verify selected checkpoint replay, immutable evaluation buffers, identical prediction membership/order, reference-off optional-input isolation, and rejection of old/mixed-mode checkpoint envelopes. Run the new tests with existing selection/metric tests. Record separate implementation and actual-execution status.

## Task 4. Allocated Gates and Submission Packaging

**Create:** `scripts/cluster/diagnose_direct_reference.sh` and `tests/test_direct_reference_cluster.py`. Copy the successful `scripts/cluster/train_anchor_pilot.sh` bootstrap/allocation pattern rather than inventing Slurm headers. Centralized submission and its ledger remain coordinator-only.

- [ ] Test fail-closed GPU allocation, exact `/shared/b00090279/conda/envs/refer-or-clear` interpreter, immutable source/configuration bindings, scratch staging and protected output paths. Require the configured GPU runtime; do not execute models on the login node. Only `ssh b00090279@hpc-gpu.aus.edu` is used for access.
- [ ] Package a diagnostic-only entry point. It cannot start100epoch training. Run the actual512by800 model integration test in both modes, training-only all-design coordinate probes, and the inherited16image overfit gate (at least15correct, image loss below0.15, no more than500updates). Include stride-eight geometry, reference-prefix BN counters, post-update hidden/reference gradients, companion invariance and immutable buffers. A skipped installed-stack test is not a pass.
- [ ] Only successful training gates unlock fresh32train/32val one-epoch smokes and fresh-process replay for both modes. The fixed registration policy/cache is reused, not tuned from validation outcomes. Report registration failures, referral availability, false positives, both attack subsets, exact phase pixel accesses, hashes and exit statuses.
- [ ] Run synthetic wrapper/runner tests, shell syntax checks and independent review before requesting diagnostic allocation scope. After approval, record one source revision/archive, configuration/data hashes, environment, output directory and Slurm job ID in the central ledger. Review actual results before requesting the fresh off100epoch pilot. Reference-on remains subsequent and separately gated.

## Scientific Interpretation

D is the next candidate to test, not a demonstrated winner over R. A small query-only performance tradeoff could be worthwhile if the matched reference-enabled method improves the intended unseen-design setting. That is a hypothesis, not a current result or permission to relax Protocol A gates. Keep R's stronger observed query-only result, uncertainty, native encoding limitations, and measured deployment costs visible. Do not select a family by repeatedly examining Protocol B or later test outcomes.

## Coordinator Integration Handoff

The remaining work is narrow isolated runner/config/diagnostic-wrapper wiring, not another architecture design. Reuse existing optimizer, scheduler, selection, metrics, checkpoint identity and replay helpers where their interfaces apply; legacy single-query loops and hard-coded gates need explicit four-input dispatch, not silent reuse. Do not modify old pilot contracts or reinterpret D-running checkpoints.

After separately approved diagnostic scope, use one GPU allocation and existing upfront authorized train/val staging. First run training-only coordinate and both-mode overfit gates, then fresh32train/32val one-epoch smoke and fresh-process replay in each mode. No test staging or new validation cutoffs. Independently review complete evidence before requesting the first fresh off bridge100epoch; require selected accuracy≥0.97 and AUC≥0.99 before separately requesting on. Reuse that off run as control; no redundant control, extra seed, N/ResNet job, or sweep is automatic. CPU engineering remains separately scoped and nonblocking. Coordinator completes executable packaging/launch details before any allocation.
