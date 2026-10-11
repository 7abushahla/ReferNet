# Protocol A Input Preparation Implementation Plan

> Execute natively with the executing-plans workflow. No subagents, commits or broad fold sweep.

**Goal:** Isolate direct resize, letterboxing and registration effects on the fold-0 no-reference control.

**Architecture:** Keep ChangerEx `single_noex`, LEVIR initialization and the existing corrected `batch_stats` encoder policy. Train at batch eight and validate each image independently. This policy is a fixed experimental control, not a claim of optimal normalization. Do not introduce diagnostic branch-calibration wrappers into the comparison.

**Tech Stack:** Existing PyTorch/Open-CD stack, OpenCV preprocessing, Slurm GPU nodes.

**Spec:** `docs/anchor-gap-investigation-2026-10-11.md`, approved input-preparation sequence.

## Global constraints

- SIDTD official fold 0, seed 42, image labels and accuracy-first checkpoint selection unchanged.
- Full 512 by 800 inputs, area interpolation for both direct resize and letterboxing. This is not the separate 299 by 299 published recipe.
- Identical native-image augmentation, including the verified query-jitter target correction. No JPEG re-encoding.
- Strict FP32 with TF32 disabled and no AMP, recorded in metadata.
- Validation-only pilots. Do not construct test pairs, invoke test inference or tune on test metrics.
- Separate output directories and fresh tags. Never overwrite existing processed data or completed runs.
- One-GPU smoke first. Only after all conditions pass, submit a bounded three-condition fold-0 comparison, not a fold sweep.

## Review focus

- Raw-to-output transforms must also map localization masks and zone coordinates correctly.
- Direct-resize caches must not silently reuse letterboxed pixels or registration transforms.
- All conditions must contain exactly the same train/validation IDs and labels and exclude wrong-design training pairs.
- Validation-only runs must never create test loaders. Strict-FP32 settings must survive checkpoint reload.
- Batch-normalization training/inference differences remain fixed across conditions. Older results are not matched controls because precision and jitter handling differ.

## Tasks

- [x] Add tested direct resizing to the preprocessing utility, and an explicit input-mode option in the data builder. Require direct resizing without registration. Record/validate the preprocessing mode for cache reuse.
- [x] Add tested validation-only and strict-FP32 run options. Keep defaults backward compatible; store the settings and enforce strict precision when reloading those checkpoints.
- [x] Add a GPU preparation/smoke script using the existing Slurm headers. Build a new direct-resize directory, verify all input conditions against the official fold, and train/evaluate a small smoke for each condition.
- [x] Run local relevant tests, then one GPU smoke. Inspect saved metadata and finite validation metrics before any longer training.
- [x] Submit the three matched 100-epoch fold-0 pilots only after the smoke succeeds. Use unique tags and the same training command except data root/tag. Report job IDs and archive the exact configuration.

## Interpretation

Compare validation accuracy and AUC, including copy-and-replace with its small sample count explicitly noted. This single-fold, single-seed development ablation cannot establish a general superiority claim. Later augmentation experiments hold the selected input preparation fixed. Wider replication and held-out reporting follow a frozen configuration and separate sweep approval.

## Execution ledger

Submitted preparation and three-condition smoke as **job 37919**, one GPU. The job builds `data/processed_direct_ablation` without modifying the two existing processed directories. It verifies every training/validation image shape and nonempty forgery mask, all fold IDs/roles/labels, and exact direct-resize/letterbox pixels against raw inputs. It then runs three one-epoch, 32-image smokes and reloads each checkpoint in a fresh process for validation only.

Local input-preparation, lossless-canvas, no-registration-build, training-config, BN-policy, fold-tool, target-alignment, official-anchor-split and smoke-acceptance tests pass. The new feature tests failed before implementation. Smoke acceptance rejects duplicate input directories, test metrics, mismatched configurations and lost precision settings. Full training remains gated on GPU smoke success.

Ruling: retain the corrected `batch_stats` policy with forced single-image inference for this ablation rather than introduce post-hoc branch calibration into training. This isolates input preparation with an already functioning control. It leaves normalization optimization for a separate experiment and does not claim the policy is optimal.

Ruling: job 37919 was deliberately canceled after 8 minutes 22 seconds during generic reference preparation. Its image generation and startup tests had succeeded, but the generic builder would generate unused reference banks for all ten folds. Added `--anchor-fold 0` to build only the median banks needed by this experiment and skip unused global/specimen banks. Replacement job **37920** reuses the verified direct-resize image cache and repeats the smoke gates. Image transforms, labels and selected-fold reference construction are unchanged. This is a compute-saving change, not a failed model experiment.

A submission watcher waits for job 37920 to reach `COMPLETED` and verifies `smokes.json` plus the matched-input audit before submitting exactly three fold-0 pilots. It stops on failure/cancellation and uses an exclusive submission lock to prevent duplicate launches. The live ledger is `/shared/b00090279/refer-or-clear/results/diagnostics/input-smoke-37920/submissions.json`. The watcher performs only Slurm polling and JSON checks on the login node; model computation remains on GPU nodes. Each pilot uses `scripts/cluster/train_input_ablation.sh`, checks the successful smoke gate and unchanged data manifests, and performs validation-only training and checkpoint replay. No broad fold sweep is authorized or scheduled.

Job 37920 completed in 12 minutes 14 seconds with exit code 0:0. All three smoke runs passed, including fresh-process validation replay and strict-FP32 metadata checks. The exact source archive is `results/diagnostics/input-smoke-37920/source.tar.gz`; smoke configurations are in `smokes.json` in that directory.

The watcher submitted direct-resize job **37922**, then recorded a broken-pipe error and stopped. Queue and accounting checks confirmed that neither remaining condition had been submitted. Recovery under a separate exclusive lock submitted only letterbox job **37923** and registered-letterbox job **37924**. The submission ledger preserves the original error and all three job IDs. Each job requests one GPU. Training completion and scientific interpretation remain pending; smoke success is not evidence of benchmark performance.
