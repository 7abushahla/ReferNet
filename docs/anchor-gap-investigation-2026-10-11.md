# Remaining Protocol A control gap

## Starting evidence and authorized scope

The corrected fold-0 control selected epoch 49 and scored 0.9441 AUC and 0.9279 accuracy on the native test set. The user authorized investigation and targeted corrections before a wider sweep. The 0.99 AUC target is an experimental acceptance criterion, not a promised outcome. Preserve the official folds and the study's no-reference/reference comparison.

## First diagnostic

`scripts/diagnose_anchor_gap.py` runs on a GPU node through `scripts/cluster/diagnose_anchor_gap.sh`.

1. Rebuild the selected model in a fresh process and verify that unchanged test inference reproduces the recorded AUC and accuracy. This test pass is verification only.
2. Score all training and validation images at batch size one using both best and last checkpoints. Record logits, attack family, map peak and target area. Check masks for empty forged targets and compare classification with the localization head's ranking.
3. Probe validation at batch size eight to quantify remaining normalization dependence. These cohort-dependent scores are diagnostic only.
4. Probe the selected model's validation predictions under the existing geometry, photometric, query-jitter and full augmentation operations, with a fixed random seed and unchanged weights.
5. Score the released fold-0 ResNet50 and EfficientNet-B3 checkpoints on the same official validation image IDs in four forms. Use original decoded pixels, a lossless PNG round trip, resize-only canvases and registered canvases. Assert exact PNG pixel and prediction equality. These are distribution-shift diagnostics, not newly trained benchmark results.

All diagnostic variants use training or validation images. Preserve per-image predictions and checkpoint hashes. Record source-code and split hashes. No JPEG re-encoding is performed. Diagnostic inference does not update model state.

## Code observations awaiting measurement

- Training applies an affine transform with rotation in the range plus or minus one degree on every example. Setting `affine_px=0` alone would still rotate images, so it is not a switch for disabling geometric augmentation.
- Query-only jitter shifts the RGB query but leaves its localization target fixed. This creates a small supervision misalignment in approximately one quarter of training samples. Its effect on the classification gap is not established.
- The encoder uses batch and spatial statistics during training (batch size eight) and spatial statistics during inference (batch size one). The magnitude of the remaining difference must be measured.
- Classification is trained jointly with a weighted localization loss. Low total training loss does not by itself establish attack-family generalization.
- The released baseline dataset code uses random resized crops, flips and brightness/contrast for training. Its baseline training script also requests the training transform for validation. These source observations do not establish the historical execution configuration, and our preprocessing and training recipe should not be described as identical to the paper's implementation.

Choose the next minimal correction from these measurements. Do not promote the current policy to all arms, alter the test split, or launch the full fold sweep solely to meet a target number.

## Investigation progress

The diagnostic was submitted as job **37902**, one GPU. Node startup and data staging precede inference. Its code uses the original augmentation implementation to characterize the completed pilot.

The official fold contains 808 genuine, 862 inpainting and 108 copy-and-replace training images. Validation contains 100 genuine, 106 inpainting and 16 copy-and-replace images. This family imbalance is an observation, not proof that sampling causes the remaining gap.

Comparing source and donor annotation values for copy-and-replace images finds identical stored values in 19 training examples and four validation examples. Two of these entries are signatures with empty text annotations. Equality of an annotation value does not establish equality of the rendered pixels or absence of manipulation traces. No examples are removed or relabeled.

`tests/test_augmentation_alignment.py` reproduced a concrete defect with a synthetic image and its exact mask. Query jitter shifted the visible region while leaving the mask behind. The correction shifts the target and valid region with the query, preserving the reference's position. The same test passes after the correction. Job 37902 retained the original code throughout its measurement. The correction was synchronized to the cluster only after that job completed. This defect does not yet explain the magnitude of the classification gap.

## Official reference verified

The user confirmed [Oriolrt/SIDTD_Dataset](https://github.com/Oriolrt/SIDTD_Dataset) as the reference. On 11 October, the upstream `main` commit was `ca17b286a1e6c7ae439bfda35abd5382079f4725`. Our local source directory is an archive inside the project, not a separate Git checkout. Byte comparison against that upstream commit confirms identical contents for `Baseline/dataset/custom_dataset.py`, `Baseline/training_kfold_baseline.py`, `Baseline/test_kfold_baseline.py` and `Baseline/_utils.py`.

The executable baseline entry points import the ordinary dataset from `SIDTD/utils/batch_generator.py` and the transforms from `Baseline/_utils.py`. The similarly named `Baseline/dataset/custom_dataset.py` is not the dataset imported by these entry points. The ordinary batch generator accepts the `dataset_crop` argument used in the training script.

| Item | Released baseline source | Current control pilot |
|---|---|---|
| Data and fold | SIDTD official template fold | Same fold-0 image IDs and labels |
| Test input | 299 by 299 resize, ImageNet normalization | Registered 512 by 800 canvas, ImageNet normalization |
| Training geometry | Random resized crop and flips | Small affine and query jitter |
| Validation transform | Training transform requested by released trainer | Deterministic full canvas |
| Initialization | ImageNet for the CNN baselines | LEVIR-CD for ChangerEx |
| Loss | Image classification cross entropy | Image BCE plus weighted dense BCE and Dice |
| Optimizer | SGD with momentum and loss-driven learning-rate reduction | AdamW with warmup and cosine schedule |
| Selection | Strict improvement in validation accuracy | Validation accuracy, with AUC breaking ties |

These are source-code differences, not a reconstruction of every historical training setting. Published checkpoint scoring is the empirical check of the baseline inference path. Relevant pinned files are [the trainer](https://github.com/Oriolrt/SIDTD_Dataset/blob/ca17b286a1e6c7ae439bfda35abd5382079f4725/SIDTD/models/Baseline/training_kfold_baseline.py), [transforms and model setup](https://github.com/Oriolrt/SIDTD_Dataset/blob/ca17b286a1e6c7ae439bfda35abd5382079f4725/SIDTD/models/Baseline/_utils.py), and [the test entry point](https://github.com/Oriolrt/SIDTD_Dataset/blob/ca17b286a1e6c7ae439bfda35abd5382079f4725/SIDTD/models/Baseline/test_kfold_baseline.py).

The fresh-process checkpoint check in job 37902 exactly reproduced the pilot's test AUC (0.9440635451505016) and accuracy (0.9279279279279279). Remaining diagnostic conditions use training and validation data only.

## Completed diagnostic 37902

Slurm reports completion with exit code 0:0 after 11 minutes 26 seconds. The complete summary and per-image predictions are archived under `results/diagnostics/anchor-gap-37902/`.

| Selected checkpoint condition | Overall validation AUC | Copy-and-replace AUC | Inpainting AUC |
|---|---:|---:|---:|
| Native, batch one | 0.975410 | 0.818125 | 0.999151 |
| Native, batch eight (diagnostic only) | 0.984180 | 0.883125 | 0.999434 |
| Geometry probe | 0.964754 | 0.749375 | 0.997264 |
| Photometric probe | 0.966639 | 0.779375 | 0.994906 |
| Query jitter probe | 0.982541 | 0.866875 | 1.000000 |
| Full augmentation probe | 0.956393 | 0.692500 | 0.996226 |

Each family AUC compares that family's forgeries against the same 100 genuine validation images. Copy-and-replace contains only 16 validation forgeries. The selected checkpoint's training AUC is 0.983693 overall and 0.856734 for copy-and-replace, so the weakness is not restricted to held-out examples. No forged training or validation examples have empty localization targets. The last checkpoint has validation AUC 0.959303 at batch one and 0.985820 at batch eight. Batch-dependent scores cannot be promoted to operational results.

| Published fold-0 checkpoint | Original pixels | PNG round trip | Resize-only canvas | Registered canvas |
|---|---:|---:|---:|---:|
| ResNet50 | 1.000000 | 1.000000 | 0.959672 | 0.958033 |
| EfficientNet-B3 | 1.000000 | 1.000000 | 0.949467 | 0.946803 |

These are validation AUCs from frozen published checkpoints, not retraining results. Lossless round trips preserve pixels and predictions exactly. Most of the measured degradation occurs before registration. Both models still discriminate copy-and-replace on registered canvases (AUC 0.969375 and 0.950000). This weakens the explanation that registration removed all useful class information. It does not isolate compression from other resampling effects or establish an achievable ceiling for a canvas-trained model.

Augmented-input probes test sensitivity of an already trained checkpoint. They do not show that training augmentation is harmful or that no augmentation is optimal. The user's intended policy remains realistic, evidence-preserving augmentation with consistent targets. Training ablations must isolate transformation families and strengths, using validation rather than test performance for selection.

## Next bounded normalization diagnostic

The user authorized continuation before any sweep. `scripts/diagnose_branch_calibration.py` tests whether train-only, branch-specific population moments can remove cohort dependence without discarding the selected checkpoint's discrimination. No weights are trained or changed. There is no test evaluation.

1. Reload the selected fold-0 checkpoint and reproduce native validation inference.
2. On all 1,778 unaugmented training images, collect encoder BN input moments separately for query and reference calls, using batches of eight. Freeze those moments and evaluate validation images individually.
3. Repeat as an isolated intervention covering all BN modules, including the decoder's shared per-branch convolutions. Installed Open-CD source confirms the decoder calls these modules separately for query and reference, so the earlier encoder-only correction leaves a possible additional shared-buffer issue.
4. Check batch-one versus batch-eight logits on the same first 16 validation images within numerical tolerance. Verify complete model-state restoration between interventions and an unchanged source checkpoint hash.
5. Preserve per-image predictions, calibration image IDs and branch moments. Do not deploy the diagnostic wrappers or start a training sweep on these results alone.

Calibration uses batch-normalized upstream activations. Switching to fixed moments can itself shift deeper activation distributions. A negative result therefore would not rule out branch-specific normalization trained consistently from the start. This is a bounded diagnostic, not proof of the optimal normalization design.

`tests/test_branch_calibration.py` verifies branch separation with analytically known moments, sample-independent output, rejection of validation/test calibration and restoration after exceptions. The test failed before implementation and passes afterward. The query-jitter target correction remains separate from this normalization experiment, whose inputs are unaugmented.

Submitted as job **37911**, one GPU with a one-hour limit. The output directory is `results/diagnostics/branch-bn-37911/`. Local branch-calibration, augmentation-alignment, data-pair, BN-policy and control-diagnostic tests all pass. No new training run or fold sweep was submitted with this diagnostic.

## Consistency-check failure and follow-up

Job 37911 failed after 4 minutes 33 seconds at the encoder-plus-head batch-one versus batch-eight logit check. Five of 16 values exceeded the existing tolerance (`rtol=1e-4`, `atol=1e-4`). The maximum absolute difference was 0.007088 and maximum relative difference was 0.000347. The encoder-only condition passed that check and recorded validation AUC 0.978361, but this remains a diagnostic result. Complete-run acceptance was not reached. The failed run's predictions, moments and original diagnostic source are archived under `results/diagnostics/branch-bn-37911/`.

The original script requested deterministic cuDNN execution but did not explicitly disable TF32. Changing batch dimensions can select different numerical implementations even when no operation combines samples. This is a hypothesis for the failure, not an established explanation.

Job **37914** repeats the diagnostic with additional instrumentation. It keeps the tolerance unchanged, records the actual cuDNN and matrix-multiply TF32 flags, compares individual versus batched inference, and separately changes companion images while keeping batch size and anchor position fixed. It repeats these checks with both TF32 flags disabled. Calibration and model weights remain fixed within each precision comparison. Evidence is saved before a failed check terminates the run. Strict-FP32 validation scores are also saved when the checks pass. Tests verify that the probe accepts a sample-independent function, rejects an explicitly batch-dependent one and restores precision flags.

## Approved sequence after the failure is resolved

1. Diagnose the consistency failure and establish a defensible normalization policy. Post-hoc calibration is not automatically the final training policy.
2. Conduct matched fold-0 training comparisons of direct resizing to 512 by 800, aspect-preserving letterboxing to that size, and letterboxing plus registration. Use identical architecture, initialization, normalization, schedule, augmentation policy and checkpoint-selection rule. Transform masks and auxiliary spatial inputs consistently. Existing resize-only and corrected registered runs differ in normalization and do not constitute this ablation.
3. Isolate augmentation families and strengths after input preparation is fixed. Apply the verified target-alignment correction throughout. A no-augmentation condition is a diagnostic baseline, not the assumed optimum.
4. Use validation for these development decisions. Avoid automatic test evaluation on every pilot. Freeze the configuration before the final held-out evaluation.
5. Obtain approval before the wider fold sweep and run sufficiently matched reference-arm comparisons before proceeding to Protocol B.

The user approved this sequence and explicitly required resolving the diagnostic failure before continuing. No preprocessing comparison or training sweep is launched while that gate remains unresolved.

## Consistency failure resolved, job 37914

Job 37914 completed successfully in 5 minutes 11 seconds with exit code 0:0. Its summary records `complete=true` and no failed strict-FP32 scopes. Artifacts are archived in `results/diagnostics/branch-bn-37914/`.

The rerun reproduces the previous encoder-plus-head discrepancy exactly with cuDNN TF32 enabled. Holding the model weights, calibration moments and inputs fixed and disabling TF32 reduces the discrepancy below the unchanged tolerance. Matrix-multiply TF32 was already disabled. Changing companion images at fixed batch shape produces zero logit difference for all four tested anchors in either precision condition.

| Calibration scope | Maximum batch-size logit difference, TF32 | Maximum difference, strict FP32 | Validation AUC, strict FP32 | Copy-and-replace AUC, strict FP32 |
|---|---:|---:|---:|---:|
| Encoder only | 0.00132751 | 0.00000763 | 0.978361 | 0.836250 |
| Encoder and head | 0.00708771 | 0.00000858 | 0.978279 | 0.836875 |

Both strict-FP32 checks pass at the original `rtol=1e-4`, `atol=1e-4`. The earlier assertion failure is attributable to TF32-related batch-shape numerics in this experiment, not demonstrated cross-image dependence in the calibrated model. This bounded finding does not remove the genuine batch dependence of the uncalibrated `batch_stats` policy at evaluation batch sizes greater than one. That existing policy must still score images individually.

The modest change from baseline validation AUC 0.975410 to approximately 0.9783 is not a closed anchor gap. Extending calibration into the head adds no material AUC improvement on this fold. Post-hoc calibration remains a diagnostic rather than an automatically adopted production policy. Numerical precision should be explicit and identical across subsequent matched input experiments. None of this supplies evidence that a registered canvas improves the no-reference control.

The failure gate is now resolved. The matched input-preparation ablation remains the next development experiment, with normalization held fixed across conditions. No test images were used in jobs 37911 or 37914. No new training or fold sweep was submitted during this failure investigation.

## Matched input preparation underway

The implementation plan is `docs/superpowers/plans/2026-10-11-input-ablation.md`. Preparation and validation-only smoke checks were submitted as **job 37919**, one GPU. The ablation holds the corrected encoder policy fixed (`batch_stats`, batch-eight training, single-image validation) and explicitly disables TF32 and AMP throughout. Post-hoc calibration remains a separate diagnostic. All conditions include the same corrected augmentation code.

The direct-resize condition uses area interpolation to 512 by 800 with no padding or registration. Letterboxing and registered letterboxing use the same output resolution. This controls output resolution and interpolation choice but is not the published 299-square input recipe. Three fresh runs are necessary because the earlier registered and resize-only runs do not share this full setup. They will not be treated as matched comparison rows.

New `--validation-only` training skips constructing test pairs and produces no test metrics. The smoke also checks fresh-process validation replay and checkpoint precision settings. The three 100-epoch fold-0 pilots remain gated on the smoke, and the broad fold sweep remains unapproved.

Job 37919 was deliberately stopped during reference preparation to avoid building unused ten-fold reference banks. Replacement **37920** uses the new `--anchor-fold 0` preparation option and reuses the completed 2,222 direct-resize images. It still builds the same selected-fold, training-only median banks and repeats every smoke gate. No model result is inferred from this scheduling optimization.

Job 37920 completed successfully in 12 minutes 14 seconds. All three smoke conditions passed the input, configuration, precision and checkpoint-replay gates. The approved 100-epoch, validation-only pilots are **37922** (direct resize), **37923** (letterbox) and **37924** (registered letterbox), one GPU each. A broken pipe stopped the submission watcher after the first submission; queue/accounting verification preceded submitting only the two missing conditions. The complete submission ledger and source archive are under `results/diagnostics/input-smoke-37920/`. No wider fold sweep was submitted, and these runs have not yet established which input preparation is preferable.

## Parallel copy-and-replace investigation

The user requested independent work while the input pilots train and explicitly approved one training-only GPU gradient diagnostic. Job **37930** runs a six-image smoke followed by all 108 fold-0 training copy forgeries, 108 inpainting examples and 108 genuine examples. The latter groups are sampled deterministically without replacement to match the copy examples' design counts. No validation or test pairs are constructed. Only new diagnostic files were synchronized; the three running pilots and their inputs remain unchanged.

`scripts/diagnose_copy_gradients.py` loads the completed corrected control's selected checkpoint, disables TF32, and measures image-classification and weighted-localization gradients separately through the backbone and decoder. It uses single-image evaluation mode, disables augmentation and performs no optimizer steps. Per-image gradient cosine similarities and norm ratios are saved with image IDs, losses and checkpoint/split hashes. Parameter buffers and the source checkpoint must remain unchanged. Tests first failed because the implementation was absent, then passed for analytic opposite-gradient geometry, zero-gradient handling, deterministic design matching and rejection of non-training selection.

This is a fixed-point sensitivity diagnostic, not a reconstruction of batch-eight augmented training. Opposing gradients can motivate a controlled loss ablation but do not prove that localization harmed learning. Absence of opposition at this checkpoint would not exclude conflict earlier in training.

Reanalysis of job 37902's saved predictions at the fixed zero-logit threshold gives the following counts. No new inference or test inspection was needed.

| Role and family | Examples | Missed forgeries | Image AUC | Map-peak AUC |
|---|---:|---:|---:|---:|
| Training copy | 108 | 48 | 0.856734 | 0.849927 |
| Training inpainting | 862 | 4 | 0.999599 | 0.999618 |
| Validation copy | 16 | 10 | 0.818125 | 0.806250 |
| Validation inpainting | 106 | 0 | 0.999151 | 0.999151 |

There are 14 false positives among 808 genuine training images and two among 100 genuine validation images. Copy errors occur across designs, not only in one nationality. The weak map-peak ranking does not support a simple claim that the localization branch already detects these examples reliably and only the referral head discards that information. Peak ranking alone does not measure spatial localization quality.

A separate annotation audit identified a candidate target-construction issue. `records.py` combines `field` and `second_field`; `forgeries.py` then marks both field names on the source image. These names differ in 16 training copy examples and two validation copy examples. The released generator describes the fields on separate source/donor documents, so treating both as altered regions on one output warrants pixel-level verification. Only three of those 16 training examples and one of those two validation examples are missed at zero logit. Therefore this candidate issue cannot account directly for most observed missed examples. No masks, labels or ongoing runs were changed. The released generator also has incompatible return interfaces, so its source alone is not proof of the historical generation behavior.

## Recipient-mask correction

Pixel checks on all 18 training/validation examples with different field names support the recipient-only interpretation. Each forged image was compared with its annotation's `src`, not a source inferred from its filename. Image dimensions matched. Both decoded images were Gaussian-smoothed with sigma 1 solely for this numerical comparison; no images were saved or re-encoded. Mean absolute channel-averaged differences inside the recipient rectangle ranged from 6.131 to 71.069 intensity levels. Differences at the donor-named rectangle on the recipient ranged from 1.056 to 3.919. Across these examples, the conspicuous change is in the recipient field, not both fields.

The user approved an isolated correction, regression tests and a separate corrected-mask directory, without changing ongoing jobs or launching another training run. The regression test reproduced the extra donor target before the fix. New records now retain only `field` in `altered_fields`; donor identity and the raw annotation remain available. The mask builder reads the recipient field from the raw annotation, so stale records containing both names cannot silently recreate the error. Missing recipient fields or missing recipient rectangles raise errors rather than falling back to donor geometry. Three recipient-mask tests and four adjacent test scripts pass. The correction remains local until deliberately deployed; the three input pilots and gradient diagnostic keep their original data and code.

This corrects supervision, not a measured performance outcome. Its effect requires a controlled follow-up. Most existing copy misses are outside these 18 affected examples.

### Isolated mask preparation verified

`scripts/prepare_copy_mask_overlay.py` prepared `data/interim/copy_target_correction_f0_20261011/`. It contains corrected train/validation records and 1,092 forged-image masks, not a complete training dataset. Exactly 18 masks change (16 training, two validation); the other 1,074 remain pixel-identical. Every changed mask removes pixels only, with no added target pixels, and every forged mask remains nonempty. No test masks are prepared. Hash checks confirm that the original records, split, transforms and all 1,092 source masks are unchanged. The running jobs have not been restarted or modified.

The reproducible audit and per-image old/new hashes are at `results/diagnostics/copy-target-audit-20261011/`. The overlay preparation test first failed before implementation and now passes for exclusion of test rows, rejection of inconsistent labels and rejection of existing or source-overlapping output directories. The mask-builder correction was committed and pushed as `a28c495` before preparing the overlay.

## Completed parallel gradient diagnostic, job 37930

The job completed in 4 minutes 37 seconds with exit code 0:0 after its six-image smoke. It scored 324 distinct training images at the epoch-49 checkpoint, 108 per family, with no optimizer steps and verified unchanged model state. Full results are under `results/diagnostics/copy-gradient-37930/`; smoke evidence is in `results/diagnostics/copy-gradient-smoke-37930/`.

| Training subset | Examples | Negative shared-gradient cosine | Median cosine | Median localization/image gradient norm ratio |
|---|---:|---:|---:|---:|
| Copy, missed at zero logit | 48 | 14/48 | 0.090502 | 0.048381 |
| Copy, correctly classified | 60 | 4/60 | 0.538380 | 297.307776 |

The shared gradient covers the backbone and decoder. Gradients are generally more aligned on correctly classified copy examples, but most missed copy examples also have nonnegative alignment. For missed copies, the median classification-gradient norm is 60.622 versus 3.076 for the already-weighted localization objective. This does not support localization dominance as a general explanation of the copy failures at this checkpoint. Across inpainting examples, 21 of 103 nonzero shared-gradient pairs have negative cosine; five image gradients are zero and have undefined cosine. Negative alignment is therefore not unique to the weak family.

Ratios on confidently correct examples are large partly because their classification gradients approach zero. They must not be interpreted as evidence of harmful loss weighting. These measurements concern unaugmented, per-image evaluation-mode gradients at one checkpoint, not the gradients of the historical augmented batch-eight training process. They neither prove nor exclude earlier training conflict. A blanket removal of localization is not justified by this diagnostic.

The immediate verified correction is the recipient mask. Input-preparation pilots remain in progress. A subsequent training comparison must use a separate corrected-data directory and compare against an otherwise matched run. Neither an additional training run nor a broader fold sweep was submitted with this diagnostic or mask correction.
