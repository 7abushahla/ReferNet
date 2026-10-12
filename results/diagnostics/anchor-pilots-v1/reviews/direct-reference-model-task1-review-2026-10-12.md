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
