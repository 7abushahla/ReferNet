# Protocol A control diagnostic

## Approved scope

Inspect existing fold-0 checkpoints on one GPU before choosing another training experiment. Keep the architecture, training runs, official folds, preprocessing and published benchmark comparison unchanged. Do not submit a wider sweep or Protocol B.

## Question

Does the large training/evaluation discrepancy in the blank-reference control involve BatchNorm running statistics, rather than necessarily indicating that preprocessing removed useful image information?

The installed Open-CD encoder applies its shared stem and residual layers separately to the query and reference. The control supplies a blank RGB reference with the query zone. Both branches update the same BN buffers. Disabling exchange does not change this behavior. This is a hypothesis about the failure, not an established cause.

## Checkpoints and inputs

- Exchange-off aligned control, fold 0, 100-epoch run, best and last checkpoints.
- Median-reference aligned model, fold 0, 100-epoch run, best checkpoint as a contrast.
- Wait for both training jobs to finish before copying and hashing their checkpoints.
- Score the same 256 class-balanced training images and all 222 validation images in each mode, in a fixed shuffled order, batch size 8, FP32.
- No augmentation, new compression, wrong-design pairs or test evaluation.
- Recalibration uses 512 training images only. No optimizer or gradient updates.

## Four conditions

1. Original evaluation with saved running statistics.
2. Encoder BN uses current-batch statistics without updating buffers. Other layers, including dropout, remain in evaluation mode.
3. All BN layers use current-batch statistics without updating buffers. Dropout remains disabled.
4. All BN running statistics are re-estimated on unaugmented training images, then ordinary evaluation is used. Learnable parameters remain unchanged.

The first three conditions must leave the complete model state unchanged. The fourth is reverted before finishing. Source checkpoint hashes must remain unchanged. The diagnostic saves per-image logits, the existing bona fide score, image BCE, class-wise score distributions, and both logit and probability-based AUC. It also records inputs to the first shared BN for its query and reference calls.

## Interpretation and limits

- A large recovery on the same training images under BN-only probes supports a train/evaluation normalization mismatch. Validation recovery strengthens its relevance to generalization.
- Recovery under train-only recalibration supports a running-statistics problem, but does not identify the blank branch as the exclusive cause.
- Good training discrimination but poor validation discrimination across conditions points toward generalization problems rather than only an evaluation-mode failure.
- Failure under every condition leaves preprocessing, training augmentation, optimization and the control's feature path unresolved.
- A gap between raw-logit and exported-score AUC indicates probability saturation or ranking ties and should be investigated separately.
- Batch-statistic validation probes depend on the other validation images in a batch. They are diagnostic only, not corrected benchmark results or deployable improvements.
- Recalibration still pools shared BN statistics across the two branches. Its failure cannot rule out branch-distribution mismatch.
- A later correction requires a justified fold-0 training pilot and explicit review before a broad sweep.

## Execution

`scripts/cluster/diagnose_control.sh` uses the existing GPU-job template. `scripts/diagnose_control.py` writes a separate `results/diagnostics/control-bn-<jobid>` directory and never replaces training checkpoints or main result tables. `tests/test_diagnose_control.py` verifies state isolation, dropout isolation, train-only recalibration and metric polarity.

Submitted as Slurm job **37873**, one GPU, with `afterany:37860:37861` dependencies. The two existing aligned pilots finish before snapshots are taken. No retraining or additional sweep was submitted. Diagnostic results belong in `results/diagnostics/control-bn-37873/`.

Local verification passed the four diagnostic tests, the existing metric checks, training-configuration tests and SIDTD-scorer tests, Python compilation and shell syntax checks. The GPU job repeats the diagnostic unit tests before scoring.

## Result of job 37873

The GPU job completed successfully. Checkpoint hashes and in-memory parameter state were unchanged. The full per-image diagnostic is at `results/diagnostics/control-bn-37873/summary.md`.

| Model and checkpoint | Condition | Validation AUC | Validation image BCE |
|---|---|---:|---:|
| Exchange-off control, epoch 100 | Ordinary evaluation | 0.4586 | 4.3544 |
| Exchange-off control, epoch 100 | Encoder BN current-batch statistics | 0.9855 | 0.1755 |
| Exchange-off control, epoch 100 | All BN current-batch statistics | 0.9817 | 0.1974 |
| Exchange-off control, epoch 100 | BN recalibrated on 512 training images | 0.4575 | 3.6141 |
| Median-reference model, best epoch 80 | Ordinary evaluation | 0.9670 | 0.1700 |
| Median-reference model, best epoch 80 | Encoder BN current-batch statistics | 0.9713 | 0.1707 |

The late control learned a discriminative rule but its stored BN evaluation path destroys it. The same control reaches 0.9986 training AUC with encoder batch statistics, versus 0.4837 with stored statistics. The median-reference model is much less sensitive. This does not yet establish whether current-batch BN is suitable at deployment batch size one. It uses other validation images to determine the normalization statistics at batch size eight. A one-image-per-batch validation probe is the next diagnostic before selecting a trainable fix.

The first batch-one submission, job 37874, exited before scoring because the diagnostic argument check incorrectly required a batch size of at least two. The check has been corrected to allow a positive batch size. Job 37874 produced no diagnostic scores and must not be interpreted as an experimental result.

## Batch-one probe and selected correction

Job **37875** completed successfully on one GPU. It scores the same 222 validation images one at a time without fitting or evaluating on the test split. Full logits are at `results/diagnostics/control-batch1-37875/`.

| Model and checkpoint | Condition | Validation AUC | Validation accuracy |
|---|---|---:|---:|
| Exchange-off control, epoch 100 | Saved BN statistics | 0.4586 | 0.4505 |
| Exchange-off control, epoch 100 | Encoder BN per-image spatial statistics | 0.9789 | 0.9459 |
| Median-reference model, best epoch 80 | Saved BN statistics | 0.9670 | 0.9505 |
| Median-reference model, best epoch 80 | Encoder BN per-image spatial statistics | 0.9677 | 0.9505 |

The single-image recovery refutes the earlier claim that the aligned canvas necessarily erased all useful evidence. It supports an evaluation-mode failure caused by the encoder's shared running statistics. It does not prove that the blank branch alone caused the failure. A single-image normalization test does not depend on other validation images, but it is still a post hoc intervention on a selected checkpoint, not a corrected training result or a fold-averaged benchmark.

The one authorized correction is `bn_policy=batch_stats` in the shared ChangerEx encoder. It disables running-statistic buffers after loading the same LEVIR weights. During training, each branch uses statistics over its current minibatch and spatial dimensions. During validation and inference, each branch uses the individual image's spatial statistics because the evaluation batch size is one. The correction preserves affine parameters and the other model weights. Scores do not depend on which other images are scored. The same option must be applied to later reference-arm training to keep normalization consistent. The original running-statistic policy remains available and is the default for legacy checkpoints. Adopting the corrected policy as the study-wide default remains conditional on the pilot result.

Run a one-epoch smoke and one fold-0, 100-epoch control pilot on one GPU before any wider sweep. The pilot uses the official split, original aligned canvases, no JPEG augmentation, no wrong-design pairs and the same optimizer/schedule as the prior control. Do not treat the 0.9789 diagnostic AUC as the pilot's result; selection on validation accuracy, checkpoint loading and held-out test scoring must succeed independently.

The pilot was submitted as Slurm job **37877**. `scripts/cluster/train_control_bn_pilot.sh` first runs BN-policy, configuration and no-exchange tests on the GPU node, then a one-epoch limited-data smoke, then the full fold-0 control. All three checks and the smoke completed successfully; its native test evaluation executed without error. The full fold-0 run subsequently completed successfully. This is one GPU job, not a ten-fold sweep. Its result is recorded below.

## Pilot progress recorded on 11 October 2026

At approximately 13:25 Dubai time (09:25 UTC), job 37877 was running and had completed epoch 10 of 100. Validation AUC was 0.8846 and validation accuracy was 0.8604. Training loss had decreased from 3.8557 at epoch 1 to 0.7944 at epoch 10. Epochs were taking approximately 89 seconds. These are intermediate validation measurements, not final test results. The preceding resize-only diagnostic, job 37866, was still running.

The raw training configuration and CSV log snapshot are preserved at `results/diagnostics/control-pilot-37877-progress-20261011T0925Z/`. The completed diagnostic directories preserve per-image predictions, checkpoint hashes, configuration, metrics and state-isolation checks. No completed pilot test result or approved fold sweep is recorded yet.

At approximately 14:04 Dubai time (10:04 UTC), the corrected pilot had completed epoch 36 of 100. Its latest validation AUC was 0.9526. The highest validation AUC so far was 0.9589 at epoch 26. Under the configured accuracy-first selection rule, the currently selected checkpoint was epoch 25, with validation accuracy 0.9324 and AUC 0.9526. The log snapshot is at `results/diagnostics/control-pilot-37877-progress-20261011T1004Z/`. Test evaluation remained pending.

The earlier resize-only control, job 37866, completed successfully after 100 epochs. Its selected epoch was 46, validation AUC 0.7497, native test AUC 0.8144 and native test accuracy 0.7523. It retains the original running-statistic policy. This is a preprocessing diagnostic, not a matched comparison with the corrected aligned control, whose test result is pending. The archived run is `results/runs/A-cex-single_noex-median-levir-f0-s42-noreg/`.

At approximately 14:24 Dubai time (10:24 UTC), job 37877 had completed epoch 50 of 100. Its highest and latest validation AUC was 0.9807, with accuracy 0.9369. Accuracy-first checkpoint selection currently favored epoch 49, with validation accuracy 0.9459 and AUC 0.9754. Test evaluation remained pending. The log snapshot is at `results/diagnostics/control-pilot-37877-progress-20261011T1024Z/`.

At approximately 14:38 Dubai time (10:38 UTC), job 37877 had completed epoch 59 of 100. Its latest validation AUC was 0.9556 and accuracy was 0.8964. Validation performance had fluctuated below its earlier maximum of 0.9807 AUC. The accuracy-selected checkpoint remained epoch 49 (accuracy 0.9459, AUC 0.9754). Test evaluation remained pending. The log snapshot is at `results/diagnostics/control-pilot-37877-progress-20261011T1038Z/`.

At approximately 15:03 Dubai time (11:03 UTC), job 37877 had completed epoch 76 of 100. Its latest validation AUC was 0.9537 and accuracy was 0.9189. The highest AUC and selected checkpoint remained unchanged (epoch 50, AUC 0.9807; selected epoch 49, accuracy 0.9459 and AUC 0.9754). Training loss continued to decrease, but validation performance had not exceeded its earlier best. Test evaluation remained pending. The log snapshot is at `results/diagnostics/control-pilot-37877-progress-20261011T1103Z/`.

## Completed pilot result

By 15:43 Dubai time on 11 October, job **37877** had completed with Slurm exit code 0:0. It ran all 100 epochs and selected epoch 49 using validation accuracy, with AUC breaking ties. The complete configuration, log, metrics and Slurm output are archived at `results/runs/A-cex-single_noex-median-levir-f0-s42-bnbatch-e100/`.

| Measurement | Result |
|---|---:|
| Selected validation AUC | 0.9754 |
| Selected validation accuracy | 0.9459 |
| Native test AUC | 0.9441 |
| Native test accuracy | 0.9279 |
| Native test EER | 0.0989 |
| Native test samples | 92 genuine, 130 forged |
| Text-inpainting test AUC | 0.9929 (109 forged images) |
| Text-copy test AUC | 0.6905 (21 forged images) |

The normalization correction produces a functioning control and supports the earlier same-checkpoint diagnostic. It does not meet the planned 0.99 AUC anchor gate. The highest intermediate validation AUC, 0.9807, must not replace the selected checkpoint's held-out result. The two attack-family AUCs use the same 92 genuine test images; the smaller text-copy group accounts for much of the aggregate AUC deficit, but this analysis does not identify why those samples fail. This is one fold and one seed, not a comparison between fold-averaged results.

The archived metadata and CSV agree on all 100 epochs, the accuracy-first selection of epoch 49, encoder policy `batch_stats`, evaluation batch size one and 222 test samples. Training restored the selected checkpoint within its evaluation lifecycle. The separate fresh-process check in job 37902 subsequently reproduced the test AUC and accuracy exactly. The broad sweep remains unsubmitted. Further diagnosis should use training and validation evidence rather than selecting modifications by this test subset's performance.
