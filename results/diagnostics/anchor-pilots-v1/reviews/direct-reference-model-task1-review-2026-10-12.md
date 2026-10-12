# Direct reference model Task 1 independent review

Date, 12 October 2026.

## Verdict and scope

**CHANGES REQUIRED against the pinned Task 1 contract.** The central shared-encoder, sequential-BN and learned reference-off computations passed the independently executed synthetic checks. Passing those checks does not close the contract discrepancies below or establish actual Open-CD/GPU, data-consumer, pipeline, diagnostic or training clearance.

Only this review file was written. No production source or test edits, real weights or images, SSH, jobs, submissions or commits were performed. Test fixtures generated synthetic weights in temporary directories and replaced the historical Open-CD constructor with `TinyHistorical`.

The coordinator subsequently authorized Maxwell to add the missing opt-in GPU test and qualify the model docstring. Those additions were not yet present at this review snapshot. Their future presence, guard behavior and coverage require inspection; no future test result is implied here.

## Findings

### F1. Unavailable rows still execute the local adapter

`src/refer_or_clear/models/direct_reference.py:164` projects every query row, and line 175 runs the local head and pooling for the complete batch. Line 176 zeros unavailable outputs only afterward. The reference encoder correctly excludes unavailable rows, but the local adapter does not.

This conflicts with the spec at `docs/superpowers/specs/2026-10-12-direct-reference-pilot-design.md:57`, which requires unavailable or empty-overlap rows to skip reference/local-adapter computation. Independent hook probes observed projection, local head and final correction calls on both rows of an all-unavailable batch. A mixed batch projected both query rows and ran the local head on both, despite only one available row. Final unavailable corrections were correctly zero and invalid optional reference rows remained ignored. This is a computation/contract discrepancy, not evidence that those reference pixels contaminated scores.

Required resolution, process only available rows through the local adapter and scatter their corrections into zeros, including a genuine all-unavailable early bypass. Add hooks asserting no adapter calls for all-unavailable/empty-support inputs and available-only batch sizes for mixed inputs in both modes. Preserve the full-query BN traversal and optimizer coverage.

### F2. The pinned FP32 overlap boundary is not enforced

`src/refer_or_clear/models/direct_reference.py:135` validates overlap shape/device and binary values but not dtype. Line 141 silently converts it to FP32. The pinned API explicitly requires binary FP32 overlap, both in the spec at line 69 and the plan at line 44.

Independent synthetic probes accepted bool, int64, float64 and complex64 binary overlap, followed by two shared-prefix BN updates in reference-on training. Complex conversion emitted a warning. `tests/test_direct_reference_model.py:398` positively tests bool acceptance rather than the declared FP32 boundary. Thus malformed-by-contract inputs do not fail before training mutation.

Required resolution, enforce FP32 overlap before traversal and replace the bool-acceptance case with pre-mutation rejection tests for non-FP32 dtypes. Any broader accepted-dtype contract instead requires an explicit coordinator-approved specification amendment, not an inferred API relaxation.

### F3. Planned installed-stack test was absent

The reviewed test file had 17 test functions, 50 parameterized cases, and ended at line 403. It contained no `test_actual_direct_reference_gpu_integration`, despite the explicit deliverable at `docs/superpowers/plans/2026-10-12-direct-reference-pilot.md:52`.

The inherited `tests/test_anchor_changer.py:777` integration test exercises the legacy anchor families, not this wrapper. Its skip cannot represent an implemented or passed direct-reference installed-stack gate. The local fixtures have **14 encoder BN modules, eight in the reference prefix**, rather than the actual architecture's expected 22 total BN modules.

Required resolution, inspect the authorized new opt-in test for a guard before any actual stack/weight/GPU access, official hash-bound fresh initialization, both modes, 512-by-800 inputs and 128-by-64-by-100 stage two, forward/backward, expected BN accounting, companion invariance, immutable evaluation buffers and checkpoint replay. Implementing or locally skipping that test still does not pass it on an allocated GPU.

### F4. Capacity wording overstates the match

The module docstring at `src/refer_or_clear/models/direct_reference.py:1` says “capacity-matched.” The spec at line 57 deliberately claims identical local-head tensors while disclosing inactive reference/difference input slices in off and additional reference computation in on. Qualify the wording as a matched local head, not exact functional capacity. This is a documentation finding, separate from F1/F2.

## Independently verified positive evidence

- Query executes once through stem and all four stages. Available reference rows execute the same parameter objects through stem/layer1/layer2 only, after the full query. Tests confirm prefix counters advance twice when reference executes and deeper counters once; off/all-unavailable prefix counters advance once. No temporary reference eval, BN copies, borrowed statistics, detachment or `no_grad` occur in the reference route.
- Factory delegates to verified `build_anchor_changer("D_running_pilot", ...)`. Existing hash verification/restricted loading and official-hash exception remain inherited, not replaced. The wrapper exposes `anchor_direct_reference_ia18_v1` and deep-copies base provenance while binding adapter seed, mode, normalization policy and adapter tensor hash. This is code-path verification, not local verification of official weight bytes.
- On/off base and adapter tensors initialize identically with caller Python/NumPy/CPU-Torch RNG preserved. Additional independent storage checks found no shared tensor storage between separately constructed arms. Changing adapter seed leaves base tensors unchanged.
- Zero final correction preserves base logits exactly on the synthetic train/eval fixtures in both modes. With a learned/nonzero final head, off retains query projection/local-head computation and structural-zero reference/difference slices. Garbage optional reference does not change off outputs, gradients or buffers.
- Available reference inputs append the corresponding query zone. Mixed batches ignore nonfinite unavailable reference rows. All-unavailable/empty stride-eight support tolerates an invalid reference object and produces finite base scores with zero correction. Common invalid inputs and needed invalid references covered by the suite fail before BN mutation, except the dtype omission in F2.
- Shared projection 128-to-32, 98-to-32-to-32 GN8/ReLU, clipped zone, overlap-masked Q/R/difference channels and valid-only mean/max pooling match the declared dimensions. Invalid cells are excluded from the reductions. Ordinary dense convolution/GN is not represented as masked normalization.
- First-step final gradients were finite and active. An independent correction-only loss probe after one adapter SGD update produced finite nonzero shared stem-convolution, BN-affine and reference-input gradients, avoiding confusion with gradients from the base query classifier.
- Optimizer groups contain every trainable parameter exactly once, with shared backbone in the pretrained group and base classifier plus the complete adapter in the new group, in both modes.
- Synthetic evaluation tests use a nonzero correction head and eight examples at batch sizes 1/2/4/8 with varied positions/companions, at `atol=1e-5`, `rtol=1e-4`. Buffers remain byte-identical. Strict same-process synthetic state-dictionary replay reproduces outputs exactly and rejects a bare legacy base state dictionary.

The equal-full-batch training-prefix equality test is a deterministic no-dropout mechanics check. It is not a universal published-Siamese training reflexivity requirement. Matching architecture/state keys alone also cannot authenticate on/off checkpoint authority; integrated checkpoint metadata must enforce the intended arm and configuration.

## Fresh test results

Environment, `/private/tmp/roc-anchor-dev/bin/python`, Torch 2.1.2, `OMP_NUM_THREADS=1`, `PYTHONDONTWRITEBYTECODE=1`. Pytest cache writing was disabled.

1. `python -m pytest tests/test_direct_reference_model.py -q -p no:cacheprovider -k 'not actual_direct_reference_gpu_integration'`, **50 passed, zero skipped/deselected, one TypedStorage warning**, exit 0, 39.30 seconds. The selector excluded nothing because the named integration test was absent.
2. `python -m pytest tests/test_direct_reference_model.py tests/test_anchor_changer.py -q -p no:cacheprovider -k 'not actual_opencd_gpu_integration and not actual_direct_reference_gpu_integration'`, **131 passed, one deselected, one TypedStorage warning**, exit 0, 108.13 seconds. The deselected case was the inherited anchor GPU integration test. No GPU test was executed.
3. Additional independent in-memory synthetic hook/dtype/storage/gradient probes completed with exit 0 and reproduced F1/F2. These observations supplement the authored tests; they are not new repository test files or a red/green implementation history.

## Reviewed byte identities

| File | SHA256 |
| --- | --- |
| `src/refer_or_clear/models/direct_reference.py` | `e0031535571a50d049c5364b55306eb76599b3b95388c80ad8f97e7fe356b5a0` |
| `tests/test_direct_reference_model.py` | `6e5a565005838def48e7f26a02b6dd14978329498b5ea6a9a88847a80733dc77` |
| `src/refer_or_clear/models/anchor_changer.py` | `e30e60dfff1f11548e1d3cf601c23fbc475b2aa84855fdba74795aedc45c2c0a` |
| `tests/test_anchor_changer.py` | `cc08f971f55a14c0668fb930f3c89e9329431d139d1d7c076d794c83ef48e446` |
| Design spec | `4c0da6c4ac45a8e716785a9b2cf6b989c8804affbbb16bf914cf3da80083b8a1` |
| Plan, latest observed snapshot | `c4bfc42b0b4e467de2a7ea62c674a1cb6f74e40749af8d0efeb1fbd1fcab47f4` |

The live plan changed during review; the quoted Task 1 clauses remained present on reinspection. Model and test hashes above remained unchanged through the final pre-write check. This review does not bind additions that were not yet inspected.

## Limits

No actual Open-CD backbone, official LEVIR bytes, real images, allocated-GPU tensors, actual 22-BN accounting, dataset/registration authority, installed-stack replay, fresh-process direct-reference replay, source archive or diagnostic artifacts were exercised. Synthetic tests cannot establish those properties. No diagnostic launcher, full training, validation outcome, family selection, sweep or downstream/reference run is approved by this review.

## Narrow re-review disposition, 12 October 2026

**Bounded PASS for the corrected Task 1 pure model and synthetic tests. F1–F4 are closed on the updated byte identities below.** This disposition supersedes the original CHANGES REQUIRED verdict only for this updated implementation snapshot. The original findings, reproductions and byte identities above are preserved as the historical review record. No new blocker was identified in this narrow re-review.

### Finding disposition

- **F1 closed.** `src/refer_or_clear/models/direct_reference.py:168` initializes a zero correction for the full population. Lines 169–182 enter the adapter only when effective availability is nonempty, subset query features before projection, execute reference only for that same subset, and scatter the differentiable corrections back with `index_copy`. All queries still traverse the original base. The ten regression cases at `tests/test_direct_reference_model.py:418` cover both modes across all-unavailable, empty pixel overlap, empty stride-eight overlap, mixed availability and mixed overlap. Hooks verify no adapter calls for empty support and available-only adapter batch sizes otherwise. They also verify exact zero unavailable corrections, no adapter gradients for all-unavailable batches, and unchanged intended BN counter accounting.
- **F2 closed.** `src/refer_or_clear/models/direct_reference.py:139` rejects non-FP32 overlap before query traversal. The eight cases at `tests/test_direct_reference_model.py:406` reject bool/int64/float64/complex64 in both modes and assert byte-identical buffers after rejection. The former positive bool-overlap test has been replaced by the valid FP32 alternate-grid case. No pinned API relaxation was introduced.
- **F3 closed as a test-implementation gap, not as an actual-GPU result.** `test_actual_direct_reference_gpu_integration` now exists at `tests/test_direct_reference_model.py:508` and calls `_actual_direct_reference_prerequisites` before any model construction or weight read. Static inspection confirms the real test uses no synthetic checkpoint fixture or monkeypatched historical backbone. Its declared checks cover both modes, official hash-bound fresh builds, matched initial state and preserved RNG, 512-by-800 FP32 tensors, `(2,128,64,100)` stage-two geometry, initial base-logit identity, 22 running-BN modules, sequential traversal/prefix versus deeper counters over two forwards, first/second backward adapter/reference gradients, evaluation companions at batch sizes 1/2/4/8, immutable evaluation buffers and strict state-dictionary replay. The companion GPU check tracks one query across positions with seven possible companions; the retained synthetic test checks all eight queries. The GPU replay is same-process, not the separately required future fresh-process pipeline replay.
- **F4 closed.** The revised module docstring at lines 1–4 explicitly distinguishes identical local-head tensors and inactive off input slices from exact functional capacity, and discloses extra on-mode encoder computation.

### Opt-in guard and skip semantics

The coordinator-approved existing flag is **`ROC_ANCHOR_GPU_INTEGRATION=1`**, with `ROC_LEVIR_CKPT` and `ROC_LEVIR_SHA256`. There is no separate DIRECT flag in this test. A later wrapper must use the same existing flag and require the actual test to execute without skips; this review does not inspect or clear that wrapper.

At `tests/test_direct_reference_model.py:490`, lack of opt-in is the only explicit skip path. Once opted in, absent CUDA, missing checkpoint environment or a nonofficial declared digest raises an assertion. Missing Open-CD raises an import error through `_require_opencd`, rather than `importorskip`. The inherited builder subsequently checks actual checkpoint bytes against the declared digest. Its byte-loading behavior was inspected, not exercised on official weights.

An additional independent in-memory probe replaced `Path.read_bytes` with a forbidden-read sentinel and simulated five prerequisite cases. Unset opt-in produced the expected skip; missing CUDA, missing environment and wrong hash failed closed; simulated missing installed stack raised `ImportError`. **Zero weight reads** occurred in all five cases. No real CUDA allocation or official weight loading was performed by these probes. The authored four guard cases also passed in the focused suite. A locally skipped integration case is explicitly unrun, never an installed-stack pass.

### Independent fresh GREEN results

Both commands explicitly unset all three GPU/checkpoint environment variables using `env -u`, set `OMP_NUM_THREADS=1` and `PYTHONDONTWRITEBYTECODE=1`, and use `/private/tmp/roc-anchor-dev/bin/python`. Pytest cache writing was disabled.

1. `python -m pytest tests/test_direct_reference_model.py -q -rs -p no:cacheprovider`, **72 passed, one skipped, one TypedStorage warning**, 48.68 seconds.
2. `python -m pytest tests/test_direct_reference_model.py tests/test_anchor_changer.py -q -rs -p no:cacheprovider`, **153 passed, two skipped, one TypedStorage warning**, 130.11 seconds. The combined shell command completed with exit 0. The two skips explicitly identify the new direct-reference integration test and inherited anchor integration test, respectively; neither executed a GPU gate.
3. The independent prerequisite/forbidden-read probe completed with exit 0. Maxwell's reported 18 RED regressions were not independently reproduced by reverting code; the fresh GREEN results and current regression behavior were independently verified.

### Updated snapshot binding

| File | SHA256 |
| --- | --- |
| `src/refer_or_clear/models/direct_reference.py` | `9839ffee680af418d6af4c1dd7b1f7c23aeaa4b73ae3555aa1a5cf628a98ce3e` |
| `tests/test_direct_reference_model.py` | `f674caae1d67578c7b43ac89d9bf766180135a7bf55576be82588eca58d84d09` |

Both hashes were checked before and during the re-review. The base model, inherited tests, design spec and latest plan retain their previously recorded hashes. The shared sequential BN policy, zero initial identity, learned off local head, gradient connectivity, optimizer uniqueness/provenance and synthetic evaluation invariance remain covered by the passing retained tests. The review does not reinterpret identical initial on/off tensors as equivalent training-state evolution.

### Continuing limits

Only this review disposition was appended. No production edits, real weights/images, SSH, jobs or commits were performed. The new GPU test is implemented and guard-reviewed but **unrun**. Actual 22-BN/installed-stack behavior, official byte identities, allocated-GPU invariance/replay, fresh-process replay, data/registration authority and integrated runner/launcher gates remain separate requirements. This bounded Task 1 PASS does not authorize an actual diagnostic allocation, full training, family selection, sweep or downstream run.
