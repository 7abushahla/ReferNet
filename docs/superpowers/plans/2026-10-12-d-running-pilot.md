# Guarded D-running Validation Pilot Implementation Plan

> **For agentic workers:** Use superpowers:subagent-driven-development with file-boundary ownership under Codex coordination. Only the coordinator commits, pushes and submits. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute the user-approved single fold-0, seed-42, 100-epoch D-running validation pilot only after fresh same-source correctness gates and independent review.

**Architecture:** Preserve the original training-only `D_running` contract and its full-training prohibition. Add `D_running_pilot` with identifier `anchor_direct_running_ia18_pilot_v1`, using identical computation, initialization and training recipe but the normal official train/validation pilot scope. This is an execution-contract distinction, not a new network architecture. Do not reinterpret or initialize from diagnostic checkpoints.

**Tech Stack:** Existing Python 3.10, PyTorch/Open-CD project environment, pytest and reviewed Slurm wrapper. No dependency upgrades.

**Spec:** `docs/superpowers/specs/2026-10-11-classification-centered-anchor-design.md`, including the user-approved guarded continuation recorded in the execution ledger.

## Global Constraints

- One full-training job, official fold 0, seed 42, 100 epochs, no early stopping or sweep.
- Fresh official LEVIR initialization, direct encoder/pooling/MLP classification, running BN, localization weight zero. Preserve the original D-running query geometry, zone dependency and declared augmentations.
- Full membership is 1,778 training and 222 validation images. No official test pixels. Legacy geometry remains diagnostic and cannot become a final support-safe arm without confirmation.
- Select by validation accuracy, then AUC, keeping the earlier exact tie. Keep subsets, false positives and final selected validation separate from curves.
- Preserve original `D_running` diagnostic-only staging/evaluation and full-training rejection. Existing R/D/N jobs and artifacts are unchanged.
- New full-pilot smoke may evaluate 32 validation images under the newly authorized validation scope. Its overfit remains the fixed 16-image training probe. Neither phase initializes full training.
- Fresh actual-stack integration, overfit, smoke, invariance and fresh-process replay must share the eventual training source/config/data authorities. Job 37979's gate cannot authorize the new source or identifier.
- All model/GPU work uses allocated compute and `/shared/b00090279/conda/envs/refer-or-clear`. Access only through `ssh b00090279@hpc-gpu.aus.edu`. One GPU per job. No login-node inference.
- No reference extension, new fold/seed, family selection or B-D detector evaluation follows automatically.

## Review Focus

1. Adding full-pilot support must not unlock the old training-only contract or allow checkpoint reinterpretation.
2. New identifier must execute the same direct running-BN model, not fall into N's decoder path or read localization targets.
3. Diagnostic and training staging must bind the same new source/config/data while excluding test files.
4. Fresh phases must retain original official initialization and head tensors; neither job 37979 nor smoke can supply training weights/buffers.
5. Validation scope, subset uncertainty and legacy-geometry limits must be explicit; passing gates must not be presented as competence.

## Task 1. Stable Contract and Recipe

**Owner:** Coordinator. Files `anchor_contract.py`, `pilot_config.py`, `pilots.py`, `configs/anchor_pilot_d_running_pilot.yaml`, `tests/test_anchor_running_pilot.py`.

- [x] Add failing tests for the new contract, identical resolved recipe except track/identifier, and preserved old prohibition.
- [x] Add the contract, official weight authority and isolated recipe without changing old defaults.
- [x] Run focused configuration/runtime tests before model and packaging integration.

## Task 2. Identical Direct Model and Installed-stack Gate

**Owner:** Archimedes. Files `models/anchor_changer.py`, `tests/test_anchor_changer.py`.

- [x] Add failing assertions for new identifier routing, shared initialization and checkpoint isolation.
- [x] Route the new track through direct pooling/MLP with retained running BN, not the decoder.
- [x] Include the new track in actual installed-stack verification of 22 BN modules, training updates, evaluation invariance and replay.
- [x] Run synthetic model regressions and report GPU-only checks separately.

## Task 3. Packaging and Training Authorization

**Owner:** Wegener. Files `scripts/anchor_job_manifest.py`, `scripts/cluster/train_anchor_pilot.sh`, `tests/test_anchor_cluster.py`.

- [x] Add failing staging/wrapper tests for the new track, train/validation membership without dense targets/test files, and mandatory actual-stack gate.
- [x] Reuse normal reviewed training orchestration and identity-bound review validation. Keep old D-running hard-blocked.
- [x] Verify no full training occurs without explicit review and a passing matching gate.
- [x] Run wrapper/helper tests and shell syntax.

## Task 4. Integration, Review and Diagnostic Allocation

**Owner:** Coordinator with fresh read-only review.

- [x] Add smoke/replay tests proving train/validation evaluation with no targets and batched invariance, preserving bounded-track training-only regressions.
- [x] Run the combined focused suite, inspect diff and obtain independent implementation review.
- [ ] Commit/push, archive the exact revision, verify authorities and submit one fresh diagnostic allocation.
- [ ] Inspect actual stack XML, probe counts/loss, smoke scope, pristine fingerprints, normalization increments, replay and source/config/data/checkpoint hashes. Obtain independent actual-gate review.

## Task 5. Single Full Pilot

**Owner:** Coordinator only.

- [ ] Create an identity-bound review record for one full pilot after actual gates pass.
- [ ] Recheck cluster authorities and submit one 100-epoch training job from the identical frozen source/config/data and original offline initialization.
- [ ] Record job ID, output root, revision/archive/config/data/gate/review hashes and scheduler state. Commit/push the ledger.
- [ ] Stop at this authorized scope. Continue monitoring N independently; do not automatically sweep or add reference training.
