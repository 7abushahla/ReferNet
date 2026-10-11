# Protocol A Classification-Centered Pilots Implementation Plan

> **For agentic workers:** Use the agreed file-boundary delegation under Codex coordination. Steps use checkbox (`- [ ]`) syntax for tracking. Read the specification and this plan before execution. Workers edit only their assigned files and return test evidence; only the coordinator commits, pushes and submits jobs.

**Goal:** Establish a credible SIDTD query-only foundation by comparing three isolated fold-0 tracks, prepare fold-safe reference support and an early encoding-header audit, then decide the matched reference extension before any sweep or downstream model evaluation.

**Architecture:** Add an isolated pilot runtime rather than changing historical training or checkpoint interpretation. R is a raw-RGB ResNet50 classifier, D is a direct current-encoder readout, and N is the current decoder classifier with a single query traversal and explicit missing reference features. A separate enrollment builder owns support provenance, while a metadata-only FantasyID audit operates independently.

**Tech Stack:** Python 3.10, existing PyTorch 2.1.2 and torchvision 0.16.2 cluster stack, Open-CD/OpenMMLab at the installed and recorded revisions, NumPy, OpenCV, pandas, scikit-learn, PyYAML, Pillow for header parsing, pytest, and Slurm. Reuse the existing environment. No automatic dependency upgrades or new architecture packages.

**Spec:** `docs/superpowers/specs/2026-10-11-classification-centered-anchor-design.md`.

**Status:** Execution authorized on 11 October 2026, with the reviewed critical-path simplification. Interface definitions land first. Workers deliver tested milestones for independent review. A separate coordinator submission starts each 100-epoch run only after that track's smoke/replay outputs have been reviewed. The original step checklist defines the execution sequence. The task-status table and execution ledger distinguish completed implementation from pending runtime and scientific checks.

## Execution Snapshot, 12 October 2026

| Task | Current state | Remaining work |
|---|---|---|
| 1, data/configuration contracts | Implemented, tested and reviewed | No new contract change proposed |
| 2, R classifier | Job 37967 completed 100 epochs; selected epoch 55 reports validation AUC/accuracy 1.0 and passing replay | Independently review selected-validation artifacts; measure preliminary CPU runtime |
| 3, D/N routes | N job 37971 training; original D failed overfit; bounded D-running job 37979 completed with passing overfit/smoke/replay | Independently review D-running evidence; any full-training contract needs a separate decision |
| 4, isolated training/replay | Implemented, tested and reviewed | Inspect completed pilot artifacts rather than treating smoke as competence |
| 5, fold-safe support | CPU job 37973 completed; metadata independently cleared; integrity job 37978 and its 270 matching hashes independently reviewed | Detector consumption and registration quality remain uncleared |
| 6, early header audit | CPU job 37972 completed; health passed; artifacts archived | Preserve its bounded metadata interpretation, not detector-transfer evidence |
| 7, correctness/shared runtime | R/N launch gates passed; original D context independently reviewed; D-running gates report passing; timing tools reviewed | Review actual D-running gates and perform actual common CPU timing |
| 8, packaging/report/decision | Allocated suite 37977 passed 558 tests with no failures/errors/skips on its frozen revision | Verify subsequent changes separately; complete selected-validation report, family decision and conditional continuation |

All detector work remains within query-only fold-0 Protocol A pilots. No reference extension, sweep or B-D detector evaluation has started. See `docs/anchor-pilot-execution-2026-10-11.md` for exact revisions, hashes, jobs, failures and independent reviews. Running epochs and intermediate metrics are not final acceptance results.

## Global Constraints

- Protocol A comes first. B, FantasyID model evaluation in C, and IDNet in D wait for its anchor and matched reference contract.
- The early FantasyID header/membership audit is permitted alongside A after execution approval. No target pixel decoding, inference, training, threshold calibration or family selection.
- Official fold 0, seed 42, 1,778 training images and 222 validation images, 100 epochs, no early stopping. Membership and labels remain official. Wrong-design fraction is zero.
- Select by validation accuracy, then validation AUC, retaining the earlier epoch on exact ties. Save selected-checkpoint results separately from last-epoch curves.
- New pilots never open official test pixels or make test predictions. Inspect test membership/provenance offline only to validate splits and exclude enrollment contamination.
- FP32, AMP off, cuDNN and matrix-multiplication TF32 off. Preserve the existing deterministic training setup and `CUBLAS_WORKSPACE_CONFIG=:4096:8`.
- Native source files remain unchanged. No JPEG augmentation or equalization, regenerated forgeries, revised historical masks or silent raw/canvas fallback.
- D and N are labeled `legacy_geometry_diagnostic`. They cannot become the final strictly train-only control without confirmation using authorized support products.
- Use new architecture identifiers, checkpoint schema, versioned directories and immutable source snapshots. Never overwrite historical artifacts or reinterpret old checkpoints.
- Heavy preparation, model tests, inference and timing run in allocated compute jobs, never on the login node. One GPU per pilot, up to three concurrent initial training jobs.
- Preserve existing jobs. No automatic sweep, auxiliary-loss follow-up, second seed/fold, reference training or B-D launch is authorized by this implementation plan alone.
- Study acceptance targets remain aggregate AUC at least 0.99 and accuracy at least 0.97. They are not guaranteed outcomes or a complete family-selection rule.
- Reviewed implementation milestones are committed and pushed. Data, checkpoints, credentials and large source archives remain untracked.

## Review Focus

1. **Absent reference changes shared state.** R/D/N never encode reference RGB; N does not run a blank through the shared decoder projection. Smoke and replay must expose companion-dependent scores or unexpected BN updates.
2. **Train-only path still opens target files.** RGB R works without zones, references or masks. D never reads masks. Test image access is blocked even when files exist. N reads masks for training loss only.
3. **Compound contrasts appear causal.** R versus D/N changes several factors. D retains declared settings and initialization but is explicitly non-stream-matched to its historical comparator. N is an absent-feature-plus-normalization package, not a proven exclusive cause.
4. **Support contains forbidden identities transitively.** Geometry, medians, zones, coverage and specimens all depend on the same authorized bank. Evaluation lookup cannot require query provenance or forgery annotations.
5. **Target audit or CPU timing overstates evidence.** Missing headers are reported, approximate quality is not a recovered encoder setting, and preliminary processed-input latency is not full officer-device latency.

## File Structure and Boundaries

| Files | Responsibility |
|---|---|
| `src/refer_or_clear/data/query.py` | Official role joins, raw RGB and legacy query-only datasets, deterministic augmentation replay |
| `src/refer_or_clear/models/anchor_rgb.py` | Offline, fail-closed ImageNet ResNet50 query classifier |
| `src/refer_or_clear/models/anchor_changer.py` | Single-query encoder traversal, D and N routes, LEVIR load audit |
| `src/refer_or_clear/pilots.py` | Versioned config/checkpoint contract, training, validation, replay, metrics and gate orchestration |
| `configs/anchor_pilot_r.yaml`, `configs/anchor_pilot_d.yaml`, `configs/anchor_pilot_n.yaml` | Fully resolved, separate pilot recipes |
| `src/refer_or_clear/data/enrollment.py`, `scripts/prepare_anchor_enrollment.py` | Eligible enrollment, cross-fitted banks, bank-scoped geometry/zones/medians/specimens |
| `scripts/audit_encoding_headers.py` | Metadata-only audit of exact existing target membership and original encoding headers |
| `scripts/check_anchor_pilots.py`, `scripts/bench_anchor_pilots.py` | GPU correctness gates and shared preliminary CPU measurement |
| `scripts/cluster/train_anchor_pilot.sh`, `scripts/cluster/prepare_anchor_enrollment.sh`, `scripts/cluster/audit_encoding_headers.sh`, `scripts/cluster/bench_anchor_pilots.sh` | Existing Slurm patterns, immutable source and data staging, isolated outputs |
| `scripts/report_anchor_pilots.py`, `tests/test_anchor_*.py`, `tests/test_enrollment.py`, `tests/test_encoding_headers.py` | Selected-validation decision report and regression checks |

Historical `train.py`, `evaluate.py`, `build.py`, `PairDataset`, split construction and scorer behavior stay unchanged. Import stable helpers where appropriate. Do not integrate new identifiers into legacy entry points merely to avoid an isolated runner. Add interfaces to new files only unless a regression demonstrates a necessary shared-helper change.

## Dependency and Scheduling Map

- Task 1's stable data/config interfaces and fixtures land first. Tasks 2 and 3 then develop independently while the coordinator completes loaders and shared runtime integration.
- Tasks 5 and 6 are independent of model construction. Prepare them in parallel with Tasks 2-4.
- Each track's critical path is its Task 1 data route, Task 2 or 3 model, Task 4 integration, Task 7 correctness gates and Task 8 training wrapper. Task 7 runtime and Task 8 report tooling can finish while pilots train.
- Task 8 owns cluster orchestration and evidence reporting. The audit and enrollment scripts can be submitted independently once their own tests pass and execution is approved. They do not wait for all model tasks.
- R, D and N each pass their own smoke and independent gate review before their own full pilot. The coordinator alone submits each job and owns the source/config/data/output/job-ID ledger. A failed or unfinished track does not block a correct independent track.
- Support preparation initially builds fold 0 only. The other nine folds are not required before these query diagnostics.
- A substantive reference pilot follows a focused extension design and approval, after query competence and support isolation are established. It does not wait for a ten-fold query sweep.

## Task 1. Explicit Pilot Data and Configuration Contracts

**Files:** Create `src/refer_or_clear/data/query.py`, `src/refer_or_clear/pilots.py` contract definitions, the three `configs/anchor_pilot_*.yaml` files, `tests/test_anchor_data.py`, and `tests/test_anchor_config.py`.

**Interfaces:**
- `load_role_rows(records_csv: Path, split_csv: Path, role: str) -> pd.DataFrame`, accepts `train` or `val` only, returns an image-ID-sorted one-to-one join with label/design/family/path/component columns and validates official membership without opening test images.
- `AnchorQueryDataset(rows: pd.DataFrame, raw_root: Path, legacy_root: Path | None, track: str, train: bool, augment: dict)` returns `query`, `label` and `meta`. R has three channels at 299 square. D/N have four channels at 512 by 800. N training additionally returns historical `mask` and `valid`. Labels are integer 0/1 before the loss chooses dtype.
- `resolve_pilot_config(config_path: Path) -> dict` returns the full effective recipe with `schema_version="anchor_pilot_v1"`, `track`, `architecture_id`, `data_contract`, normalization, precision, optimizer, schedule, selection and supervision. No permissive fallback for unknown fields/track combinations.
- New architecture IDs are `anchor_rgb_r50_v1`, `anchor_direct_ia18_v1`, and `anchor_absent_ia18_v1`. A pilot checkpoint contains schema, architecture ID, effective config, epoch, validation selection key, state dict and provenance hashes.

- [ ] **Step 1. Add failing data/config tests.** `test_rgb_needs_only_raw_image_and_label` deletes auxiliary directories; `test_test_pixels_are_never_opened` installs an image-open guard; `test_join_rejects_duplicate_missing_or_mismatched_ids` rejects silent inner-join loss; `test_no_wrong_design_relabeling` verifies all original labels. Assert D works without masks and N evaluation works without masks. Missing required training N masks and corrupt images raise errors containing the image ID.
- [ ] **Step 2. Run `python -m pytest tests/test_anchor_data.py tests/test_anchor_config.py -q`.** Expect failures for missing new interfaces before implementation, not import failures in unrelated dependencies.
- [ ] **Step 3. Implement the dataset and recipes.** R validation exactly matches `scripts/score_sidtd_checkpoints.py:preprocess`, OpenCV BGR-to-RGB, one bilinear resize, ImageNet normalization. Implement the training transform as a named `raw_rgb_299_v1` NumPy/OpenCV preset, crop scale 0.8-1.0, log-uniform ratio 0.75-4/3 with ten attempts and bounded center fallback, H/V flip probabilities 0.5, brightness/contrast probability 0.2, contrast factor uniform 0.8-1.2 and brightness offset uniform -0.2 to 0.2 times 255, clipped to uint8. Serialize interpolation and clipping semantics. This is a benchmark-adjacent transform family, not an exact historical Albumentations replay. No new dependency is required.
- [ ] **Step 4. Preserve D/N declared query augmentation settings.** Reuse `PairDataset`'s query geometry, jitter and photometric primitives without reference or forgery-file reads. Do not reproduce unused reference draws or historical RNG trajectories. Store effective settings and deterministic seeds. D inherits job 37932's declared recipe, N job 37877's. Both have no crop/JPEG and affine limit eight pixels; existing default query jitter is two pixels at probability 0.25. Persist those defaults explicitly. Preserve legacy `drop_last=True` for D/N training, with omitted IDs reported; R retains the final partial batch. Validation always retains every image.
- [ ] **Step 5. Add `test_rgb_preprocess_matches_scorer` with exact float32 tensor equality on a synthetic non-square color image.** Add deterministic-seed, binary-mask geometry and query-jitter alignment checks for the reused query primitives. Assert val has no augmentation, and no model input contains attack family, component, design ID or true altered-field location. Do not claim historical stream matching.
- [ ] **Step 6. Run the focused tests to pass and commit.** Include a machine-readable table of exact resolved recipe values. No new image preprocessing artifacts or jobs are created by these unit tests.

## Task 2. Pilot R, Fail-Closed Raw RGB Classifier

**Files:** Create `src/refer_or_clear/models/anchor_rgb.py` and `tests/test_anchor_rgb.py`.

**Interfaces:**
- `build_anchor_rgb(weights_path: Path, expected_sha256: str, head_seed: int) -> nn.Module` loads offline ImageNet ResNet50 `IMAGENET1K_V1` tensors, verifies hash/key coverage, then replaces the 1,000-way head with a seeded 2-class linear head.
- `forward(query: Tensor) -> dict[str, Tensor]` returns `class_logits` of shape `(B, 2)` and `attack_logit=class_logits[:, 1]-class_logits[:, 0]` of shape `(B,)`. No map is returned.
- `parameter_groups() -> tuple[list[nn.Parameter], list[nn.Parameter]]` separates encoder and new head for the bounded overfit probe; R's actual SGD uses one common LR.

- [ ] **Step 1. Add tests `test_weights_missing_or_hash_wrong_fail_closed`, `test_rgb_has_no_auxiliary_inputs`, `test_rgb_softmax_sigmoid_and_argmax_ties`, and `test_rgb_frozen_bn_companion_invariance`.** Missing weights must fail before training, never warn and continue randomly. All pretrained tensors, including BN buffers, must load. At equal two-class logits, argmax selects genuine, matching acceptance at bona fide score at least 0.5.
- [ ] **Step 2. Run `python -m pytest tests/test_anchor_rgb.py -q` and verify failure on missing implementation.** Use a tiny fake classifier/fixture for offline contract tests. Actual ResNet/weight tests run on the GPU allocation with the staged official checkpoint.
- [ ] **Step 3. Implement the classifier.** Keep conventional BN training updates and frozen-buffer evaluation. Download/stage requested weights before compute execution if needed, with provenance recorded; do not fetch silently inside a compute job. Record loaded tensor hashes before head initialization.
- [ ] **Step 4. Verify image loss reaches encoder and head, all active parameters occur exactly once, no decoder/map parameters exist, and fresh-process state loading reconstructs identical architecture.** Verify evaluation buffers remain byte-identical and logits meet the Task 7 invariance tolerance.
- [ ] **Step 5. Run focused tests to pass, document the pretrained source and hash, and commit.**

## Task 3. Pilots D and N, Isolated Current-Encoder Routes

**Files:** Create `src/refer_or_clear/models/anchor_changer.py` and `tests/test_anchor_changer.py`.

**Interfaces:**
- `encode_query(backbone: nn.Module, query: Tensor) -> list[Tensor]` traverses the installed IA_ResNetV1c stem/maxpool and four residual stages once, exchanges disabled, returning widths 64/128/256/512. Verify actual installed stage names against its pinned source before writing the traversal. Do not emulate a different ResNet stem.
- `build_anchor_changer(track: str, levir_path: Path, expected_sha256: str, head_seed: int) -> nn.Module`, accepts D/N, loads the same LEVIR surgeries and allowlisted tensors as the historical implementation, fails on missing weights/unexpected keys.
- D `forward(query)` returns `attack_logit` `(B,)` from GAP of deepest query features and a new 512-to-128-to-1 ReLU MLP, without a map.
- N `forward(query)` returns `attack_logit` `(B,)` and `map_logit` `(B,1,H/4,W/4)` using one decoder `base_forward`, a zero second projected tensor, then existing FDAF/MixFFN/map/referral modules.

- [ ] **Step 1. Add failing tests `test_query_features_match_legacy_exchange_off`, `test_direct_route_has_no_decoder_or_map_ancestor`, `test_absent_route_calls_encoder_and_base_forward_once`, and `test_absent_reference_never_updates_shared_bn`.** Check tensor shapes and expected BN `num_batches_tracked` increments by one per active module for N training, zero in evaluation. A mocked second decoder call that throws must never execute.
- [ ] **Step 2. Run `python -m pytest tests/test_anchor_changer.py -q` and verify the intended failures.** Keep pure tensor/mocked graph checks available locally; installed Open-CD/pretrained integration checks are mandatory on the allocated GPU, not silently skipped at launch.
- [ ] **Step 3. Implement D.** Initialize/load the historical modules before retaining the encoder, so query tensor starting hashes can be compared with job 37932's declared initialization. Isolate MLP initialization with a forked RNG stream. Apply encoder `batch_stats` only, retain affine parameters, evaluate one image at a time. The active model has no decoder, map or reference traversal. Compare query tensors/initial features with the historical single-no-exchange route.
- [ ] **Step 4. Implement N.** Keep `bn_policy=running` throughout active encoder/decoder. Compute projected query features once, synthesize `zeros_like(projected_query)` only after projection, and retain the existing map-dependent readout. Inspect every normalization/dropout module, including fusion blocks. Do not load a trained batch-stat checkpoint or add a BN recalibration stage.
- [ ] **Step 5. Add gradient/reference-invariance and save/load tests.** D classification loss reaches encoder/MLP only. N classification reaches its active readout/map route; N's dense target affects the declared training loss but never inference. Both ignore arbitrary legacy reference planes in the test adapter. Inactive modules are not counted as active parameters or added to optimizer groups.
- [ ] **Step 6. Run focused tests to pass and commit.** Record that N changes both absence representation and normalization. Do not promote it as a demonstrated normalization-only fix.

## Task 4. Validation-Only Training, Accumulation and Replay

**Files:** Implement `src/refer_or_clear/pilots.py`; create `tests/test_anchor_training.py` and `tests/test_anchor_replay.py`.

**Interfaces:**
- `run_pilot(config_path: Path, raw_root: Path, legacy_root: Path | None, run_dir: Path, device: str, phase: str) -> dict`, phases `overfit`, `smoke`, `train`, `replay`; training always uses train/val only.
- `evaluate_pilot(model: nn.Module, loader: DataLoader, track: str, device: str) -> tuple[dict, list[dict]]` returns sample-weighted classification loss, AUC/accuracy, class counts, genuine false positives, family metrics and per-image logits/scores.
- CLI `python -m refer_or_clear.pilots --config PATH --raw-root PATH [--legacy-root PATH] --run-dir PATH --device cuda --phase PHASE [--checkpoint PATH]`. Overfit/smoke use their declared fixed limits. There is no test-role switch in this development CLI.
- Output files are `train.json`, `log.csv`, `val_predictions.csv`, `val_summary.json`, `replay.json`, `gate.json` and checkpoints below the new run directory. Refuse any nonempty run directory unless executing a declared read-only replay.

- [ ] **Step 1. Add failing tests for validation-only file access, deterministic selection ties, sample-weighted validation CE, fresh-process replay, N-only dense target use, sampler/head RNG independence and uninitialized-weight rejection.** Add `test_partial_accumulation_matches_full_window_gradient` using unequal microbatches and a final one-microbatch window.
- [ ] **Step 2. Run `python -m pytest tests/test_anchor_training.py tests/test_anchor_replay.py -q`.** Expected failures identify unimplemented behavior, not unrelated historical tests.
- [ ] **Step 3. Implement R's recipe.** SGD LR 0.01, momentum 0.9, weight decay zero, Nesterov off, microbatch 32, accumulation two, final partial batches retained. Normalize each accumulation window by its actual total example count, not a fixed two or full-batch size. Step ReduceLROnPlateau once per epoch on deterministic sample-weighted validation CE, factor 0.9, patience one, relative threshold 1e-4, cooldown zero, min LR zero, eps 1e-6. Do not add gradient clipping to R's benchmark-adjacent recipe.
- [ ] **Step 4. Implement D/N recipes.** AdamW encoder/pretrained LR 1e-4, new MLP/referral LR 1e-3, weight decay 0.05, batch eight, one step per batch, existing gradient norm clip five. Import `train.make_scheduler` with 1,000 warm-up steps for the full pilot and `train.selection_key` for every track; do not duplicate these rules. D image BCE only. N imports `referral_loss` with weight three and immutable legacy masks. Use the helper's bounded warm-up during the shorter smoke, record actual values. Each trainable parameter appears once.
- [ ] **Step 5. Define score/export/selection accounting.** `attack_probability=sigmoid(attack_logit)`, `bonafide_score=1-attack_probability`, labels one altered. Reuse existing bona fide-oriented metric functions. Two-class ties accept genuine. Save both raw logits and probabilities to expose saturation. Family AUC uses the same genuine validation population for each attack; missing/single-class metrics are null with counts, never zero or invented. No localization metrics for R/D; N's box-target metrics are identified as such.
- [ ] **Step 6. Preserve stochastic provenance.** Seed sampler/worker streams independently of new-head initialization and record Python/NumPy/Torch versions and effective worker behavior. Verify reproducibility of the new route under its own declared seed. Label D's historical contrast non-stream-matched from the start.
- [ ] **Step 7. Add fresh-process replay.** Reconstruct only from the versioned effective checkpoint contract, reject mismatched architecture/schema/data hashes, leave state unchanged, and compare per-image validation logits using Task 7's tolerance. Never consume overfit/smoke state in long-run initialization. Set no-early-stop epochs 100 even if a curve fluctuates.
- [ ] **Step 8. Run focused tests and existing `tests/test_metrics.py`, `tests/test_classification_loss.py`, `tests/test_train_config.py` to pass, and commit.** Record all precision flags and active/total parameter counts in every run.

## Task 5. Fold-Safe Enrollment and Support Products

**Files:** Create `src/refer_or_clear/data/enrollment.py`, `scripts/prepare_anchor_enrollment.py`, and `tests/test_enrollment.py`.

**Interfaces:**
- `make_enrollment_manifest(records: pd.DataFrame, split: pd.DataFrame, fold: int, raw_root: Path) -> dict` records official split hash and eligible training-genuine IDs, excluding any component present in val/test. Its named banks are `train_half0`, `train_half1`, `enrollment`.
- `build_enrollment_products(manifest: dict, raw_root: Path, output_root: Path, specimen_root: Path | None = None) -> dict` reads authorized bank images/layout entries only and returns product paths/hashes and preparation counts.
- `resolve_enrollment_bank(manifest: dict, design_id: str, role: str, training_component: str | None = None) -> dict` selects the opposite stable component-hash half for train and the fixed enrollment bank for val/evaluation. Evaluation lookup has no query identity, label, source or donor argument.
- CLI `python scripts/prepare_anchor_enrollment.py --records PATH --split PATH --fold 0 --raw-root PATH --out PATH [--specimen-root PATH]`; existing output directory is a hard error.

- [ ] **Step 1. Add failing tests for source/donor exclusion, empty-bank failure, unchanged official classifier membership, same-component training exclusion and evaluation lookup with provenance/forgery fields stripped.** Include a donor-linked component fixture, not identity-only exclusion.
- [ ] **Step 2. Add transitive isolation tests.** Freeze partition/component membership, then mutate excluded genuine pixels/layout entries and the unused training half. Rebuild and require corresponding target/median/zone/coverage/specimen output hashes unchanged. Mutating an included image must change its contributor hash. Shared VIA files require hashes of the included parsed layout entry, not an artifact dependency on excluded entries in the entire file. Record complete file hashes separately for archival provenance without using them to fit banks.
- [ ] **Step 3. Run `python -m pytest tests/test_enrollment.py -q` and verify intended failures.** Tests use tiny synthetic images and annotations, not production data.
- [ ] **Step 4. Implement the manifest.** Validate the frozen graph from `provenance_components`, use existing `fold_of` parity for training halves, preserve all classifier rows and labels, and sort support IDs. Store per-contributor raw and parsed genuine-layout hashes, component IDs, fold/bank/version, source revision and output hashes. Keep independent blank specimen source/hash explicit. Empty support or unknown designs fail closed; no global median fallback.
- [ ] **Step 5. Build each bank independently.** Start from letterboxed support images, choose a deterministic authorized anchor, fit SIFT/RANSAC similarity using existing constants, record failed support registrations, and form a validity-aware median. Create zones from all genuine personal-data rectangles in that bank, never altered-field annotations. Transform rectangles as polygons before rasterization, apply the existing four-pixel zone dilation in bank coordinates, and retain overlap/coverage. Align an optional independently sourced specimen only to that bank's target. No test/val query processing belongs to this builder.
- [ ] **Step 6. Pin coordinate contracts.** Products live in `data/processed_anchor_v1/fold0/<design>/<bank>/` with a manifest and explicit bank-coordinate transforms. Registration/RANSAC seeds derive deterministically from fold/design/bank, with OpenCV worker settings recorded. Future reference-to-query transformation must pass synthetic coordinate/nearest-neighbor mask tests before consumption. A failed transformation marks context unavailable; do not silently substitute a global product or clear a query.
- [ ] **Step 7. Run isolation tests and existing reference/mask/split tests to pass, and commit.** The metadata feasibility counts in the spec are checks to compare against the actual builder, not asserted successful output counts. A production fold-0 build runs later under Task 8 allocation.

## Task 6. Early Exact-Membership Encoding-Header Audit

**Files:** Create `scripts/audit_encoding_headers.py`, `tests/test_encoding_headers.py`, and the Task 8 audit wrapper. Do not change `scripts/audit_jpeg.py` or replace the old saved audit.

**Interfaces:**
- `read_encoding_header(path: Path) -> dict` returns actual payload format, dimensions, component sampling/color markers, full quantization tables/signatures, exact IJG match candidates and the separately labeled legacy luminance-based estimate. It never calls pixel decoding or `Image.load`.
- `audit_target_headers(records_csv: Path, split_csv: Path, raw_root: Path, output_dir: Path) -> dict` audits FantasyID rows with roles `test` and `reference_only`, and records exact membership plus omissions. Other target training attacks may be inventoried in a separately labeled corpus section only, not inserted into evaluation membership.
- CLI `python scripts/audit_encoding_headers.py --records PATH --split PATH --raw-root PATH --out PATH` freezes manifests and emits per-image CSV, JSON summary and Markdown report in a new directory.

- [ ] **Step 1. Add failing tests `test_no_pixel_decode_or_model_import`, `test_png_payload_under_jpg_extension`, `test_exact_tables_are_distinct_from_quality_estimate`, `test_missing_file_is_counted_not_dropped`, and `test_membership_join_is_one_to_one`.** Monkeypatch `Image.Image.load` and OpenCV decoders to throw. Corrupt headers and duplicate records are explicit errors/counts, never silent exclusions.
- [ ] **Step 2. Run `python -m pytest tests/test_encoding_headers.py -q` and verify intended failures.** Fixtures include genuine/attack encodings at several qualities, a nonstandard table, and sampling differences at fixed tables.
- [ ] **Step 3. Implement header parsing.** Use Pillow header inspection with file handles closed, record content signature instead of filename suffix. Compare available luma/chroma tables against standard IJG tables for qualities 1-100; retain multiple matches and custom-table status rather than force one recovered setting. Record quality estimate separately. Metadata association is not direct CNN metadata access or proof of mechanism.
- [ ] **Step 4. Freeze the current membership candidate.** Copy/hash `records.csv` and `splits/protocol_c.csv` metadata before reading headers. Identify `heldout`, `attack1`, the complete official test and their design/attack/capture strata; report genuine and altered counts, table/sampling signatures, predeclared `estimated_quality < 85` rule accuracy and fixed-direction `-estimated_quality` AUC where both classes exist. Do not optimize a target threshold or orientation. Do not attach all genuine test images to an attack-only subset without a predeclared comparison definition. Single-class subsets report no AUC; any attack1-versus-genuine comparison identifies the different design populations explicitly.
- [ ] **Step 5. Audit proposed support membership separately.** Existing `reference_only` rows are a candidate support inventory, not an approved support pool for all held-out designs. Report identity/provenance overlaps and designs without separate support. Label the report `membership_candidate_v1`, not a completed locked Protocol C loader. If later support policy changes membership, the later protocol must rerun the audit.
- [ ] **Step 6. Report completeness and interpretation.** Missing/corrupt files produce an incomplete report with affected IDs and forbid exhaustive/perfect-separation claims. Deterministic association limits shortcut-independent interpretation but does not invalidate every transfer result. No pixel statistics, recompression, blockiness, weights, inference or data regeneration. Run focused tests to pass and commit.

## Task 7. Pretraining Gates and Shared Preliminary Runtime

**Files:** Create `scripts/check_anchor_pilots.py`, `scripts/bench_anchor_pilots.py`, and `tests/test_anchor_gates.py`, `tests/test_anchor_runtime.py`.

**Interfaces:**
- `check_inference_contract(model: nn.Module, queries: Tensor, track: str) -> dict` records hooks, BN/dropout policies, state hashes and logits across batch placements, with FP32 tolerances `atol=1e-5`, `rtol=1e-4`.
- `check_pilot_gates(run_dir: Path, phase: str) -> dict` validates initialized tensor hashes, optimizer coverage, actual train/val membership, no test access, precision, overfit/smoke outcomes and replay.
- CLI `python scripts/check_anchor_pilots.py --config PATH --phase pretrain|selected --run-dir PATH --raw-root PATH [--legacy-root PATH] --device cuda` exits nonzero on any required gate failure.
- CLI `python scripts/bench_anchor_pilots.py --runs R_PATH D_PATH N_PATH --runtime-manifest PATH --raw-root PATH --legacy-root PATH --out PATH` times all three serially in one CPU allocation, no predictions used for selection outside source validation.

- [ ] **Step 1. Add failing gate tests.** N/R are evaluated alone and in batch sizes two, four and eight at first/middle/last positions with distinct source companions; scores must meet the declared tolerance and buffers remain identical. D is deliberately scored singly and any request for batched production evaluation is rejected, not presented as invariant. Add a deliberately batch-dependent mock to prove the N gate catches it. Hooks require one N encoder traversal and one shared decoder projection. Test unknown schema/random initialization/forbidden target dependency failures.
- [ ] **Step 2. Implement mandatory pretraining phases.** Store 16 deterministic train IDs, eight genuine, six inpainting and two copy, augmentation off, minibatch eight, diagnostic AdamW LR 1e-3/no decay, at most 500 updates, evaluation every 50. Require at least 15/16 correct and mean image classification loss below 0.15 under each track's inference policy; also report N total/dense loss without replacing that common criterion. Failure blocks the track. Then run one epoch on 32 fixed train and 32 fixed val IDs under the real recipe and fresh-process replay. For N, batch/companion checks run before and after this smoke, not after 100 epochs only.
- [ ] **Step 3. Enforce resets.** Full training reconstructs from original pretrained tensors, newly seeded head, pristine BN, optimizer/scheduler and sampler RNG. Compare initialization hashes against the pre-probe ledger. Overfit/smoke checkpoints cannot be supplied to `--phase train`. Repeat invariance/replay on the selected long-run checkpoint.
- [ ] **Step 4. Pin shared runtime contract `anchor_cpu_v1`.** Freeze the first 200 lexicographically sorted fold-0 train IDs and record their hashes. Use the same physical allocated host for all tracks, PyTorch CPU FP32 eager/inference mode, batch one, Torch intra-op four threads, inter-op one, OpenCV one, OMP/MKL four, 20 untimed warm-ups per track and one timed pass over the same 200 documents in fixed R/D/N order. Set thread values before model execution, no concurrent benchmark workload, no ONNX/quantization. Record actual CPU model/hostname, affinity, OS/library versions and source/checkpoint hashes in the runtime manifest before timing. If a common host or at least four allocated CPUs is unavailable, report not-comparable rather than substitute settings.
- [ ] **Step 5. Define timing boundaries.** Network-only timing uses ready tensors and includes score materialization. Input-preparation timing reads/decode/transforms each track's declared source artifact. R reads native raw RGB; D/N read existing registered PNG plus zone. Report median/p95, all per-image durations, shape, active/total parameters and checkpoint size. D/N preparation excludes the upstream generation of legacy canvases, explicitly labeled `processed_input_only`. These are comparable recipe-level preliminary measurements, not an isolated network-capacity contrast or an end-to-end officer-device claim. Full geometry/reference cached/uncached latency remains a later matched-family measurement.
- [ ] **Step 6. Test timing accounting with a fake clock, manifest/host mismatch rejection, absence of raw-to-canvas fallback and exactly 200 timed documents.** Run `python -m pytest tests/test_anchor_gates.py tests/test_anchor_runtime.py -q` to pass and commit. A runtime fixture test does not execute a real benchmark locally.

## Task 8. Cluster Packaging, Independent Launch Gates and Decision Report

**Files:** Create the four cluster wrappers in the file map, `scripts/report_anchor_pilots.py`, `tests/test_anchor_cluster.py`, and `tests/test_anchor_report.py`.

**Interfaces:**
- Training wrapper `sbatch scripts/cluster/train_anchor_pilot.sh PHASE SOURCE_DIR CONFIG_PATH RAW_ROOT LEGACY_ROOT RECORDS SPLIT WEIGHTS WEIGHTS_SHA256 OUTPUT_ROOT [REVIEW_JSON]`; the config selects exactly one of R/D/N. `diagnostic` exits after its checks. `train` requires a separate matching independent review record. Explicit metadata and offline weight arguments bind the reviewed inputs. R ignores LEGACY_ROOT and fails if auxiliary data becomes required.
- Preparation wrapper `sbatch scripts/cluster/prepare_anchor_enrollment.sh SOURCE_DIR RECORDS SPLIT RAW_ROOT OUTPUT_ROOT`, builds fold zero only.
- Audit wrapper `sbatch scripts/cluster/audit_encoding_headers.sh SOURCE_DIR RECORDS SPLIT RAW_ROOT OUTPUT_ROOT`, performs headers/membership only.
- Benchmark wrapper `sbatch scripts/cluster/bench_anchor_pilots.sh SOURCE_DIR R_RUN D_RUN N_RUN RAW_ROOT LEGACY_ROOT OUTPUT_ROOT`; all three measurements stay on one host.
- Report CLI `python scripts/report_anchor_pilots.py --runs R_PATH D_PATH N_PATH --out PATH` consumes selected-validation artifacts and ledgers only, preserving incomplete/failed-track status.

- [ ] **Step 1. Add failing wrapper/report tests.** Assert allocated-node checks, required positional paths, one GPU per training track, no `srun`/Python model work on login, immutable code/config resolution, shell failure propagation and no automatic sweep command. Test incomplete audits/runs, missing replay, mixed val/test metrics and wrong checkpoint selection are not summarized as successful candidates.
- [ ] **Step 2. Copy existing Slurm patterns.** Use `train_classification_anchor.sh` for environment/account/staging and `diagnose_copy_gradients.sh` for short diagnostics. Retain account `acc-mialhajri`, four CPUs, absolute conda activation behavior, scratch/TMPDIR and deterministic flags. Model diagnostics and training request one GPU on `gpu`. CPU-only enrollment, header audit and CPU timing request the `cpu` partition without a GPU reservation, as explicitly required by the user. Pin resolved environment prefix and executable in the ledger instead of assuming a globally active env. Adjust job name/time/memory only with a recorded resource contract and observed smoke needs, never change the scientific batch/precision recipe on OOM.
- [ ] **Step 3. Package immutable source.** Commit tested code first, archive the exact commit to an untracked versioned source directory, record SHA256/revision/environment snapshot. Stage official raw train/val files for R, frozen legacy inputs/targets for D/N, and original metadata for audits. Exclude `._*`. Validate hashes before using data. Avoid staging irrelevant clips/videos or rebuilding old processed data. Scripts do not pull changing source during a run.
- [ ] **Step 4. Prepare the centralized launch ledger.** Record three proposed pilot tracks, independent support and header audit jobs, estimated resources from smoke and unique output paths. Verify `python -m pytest --version` in the active cluster environment before relying on pytest. Codex alone records source hash, configuration hash, data hashes, output directory and job ID for each submission. Preserve prior running jobs.
- [ ] **Step 5. Submit overfit/smoke diagnostics independently, then provide each gate bundle for independent review.** A diagnostic job must exit after its checks and fresh-process replay. The coordinator makes a separate 100-epoch submission only after review clears that track's gates; no worker or diagnostic script automatically starts it. Record job IDs/status and copy ledgers/results back without checkpoints/data entering git. Support/audit allocation uses its own focused tests and can run in parallel. If resources permit fewer than three training jobs, schedule the approved tracks without changing them or broadening the search.
- [ ] **Step 6. Generate selected-validation reports.** For each track show chosen epoch, overall AUC/accuracy/loss, per-family AUC/counts, genuine false positives and misses, initialization/replay/invariance status, active/total parameters and shared runtime availability. Keep curves and historical test evidence in separate sections. Report no causal or optimality conclusion from a compound contrast or 16-copy subset. Do not use the FantasyID audit to tune a detector or choose a backbone.
- [ ] **Step 7. Produce the 13 October decision note.** State actual completions, failures and support readiness. A passing query candidate is provisional. If R/D leads, write the focused same-family zone-only/reference-adapter design next; if N leads, specify its reference-on route and normalization next. Establish matched query pixels, zone, geometry, optimizer/supervision and a separately trained no-reference control. No full sweep precedes this check. If no credible candidate exists, report the anchor gap and propose a bounded source-validation investigation for approval.
- [ ] **Step 8. Run shell syntax and focused report tests, then the full repository suite on the allocated compatible environment.** Commands are `bash -n scripts/cluster/train_anchor_pilot.sh scripts/cluster/prepare_anchor_enrollment.sh scripts/cluster/audit_encoding_headers.sh scripts/cluster/bench_anchor_pilots.sh`, `python -m pytest tests/test_anchor_cluster.py tests/test_anchor_report.py -q`, and `python -m pytest tests/ -q`. Record unrelated pre-existing failures rather than repairing them silently. Commit/push reviewed milestones and the decision note, not source archives or model weights.

## Conditional Continuation, Outside the Initial Launch

1. A substantive fold-0 reference comparison starts after candidate competence and fold-safe support readiness, following a focused approved adapter/geometry design. Interface preparation may occur earlier; a smoke alone does not pick the family.
2. Confirm the candidate and matched contract on fold zero seed 43 and preselected fold one seed 42, validation only, with a repeated nearest comparator if claiming superiority. These require a separate continuation decision.
3. Approve a sweep only after those gates. Lock architecture, recipes, enrollment, scoring and operating-threshold policy before final A test evaluation. Disclose the prior exploratory use of historical test evidence.
4. Advance B only after A. Advance C afterward with a resolved independent genuine support/evaluation contract and an updated header audit if membership differs. Native transfer and encoding sensitivity remain distinct claims. D follows the study-wide sequence and actual available time.
5. Report what is complete on 15 October. Do not retrofit the planned architecture into the manuscript as an implemented result, hide unresolved anchor failures or promise unexecuted protocols.

## Self-Review and Execution Handoff

- Specification coverage is assigned to Tasks 1-4 for the three query tracks, Task 5 for transitive support safety, Task 6 for the early bounded target audit, Task 7 for mandatory N batch invariance/shared timing, and Task 8 for immutable cluster execution/validation reporting. The reference adapter is deliberately a conditional focused design, not an underspecified implemented deliverable.
- All model/loader/training interfaces use the new `anchor_pilot_v1` contract. Legacy checkpoints, datasets and run directories remain unchanged.
- Correctness gates, scientific advancement and job/sweep authorization are separate. No test-pixel access or B-D performance-selection route is provided.
- Execution uses parallel workers for R, D/N and the header audit, with support construction independent. Codex owns shared interfaces, loaders, runtime integration and every submission. The other agent reviews milestone diffs and gate outputs without editing the repository.

Proceed within the authorized scope. Publish completed milestone diffs and focused test evidence for independent review. Hold each 100-epoch submission until its gate bundle has been reviewed.
