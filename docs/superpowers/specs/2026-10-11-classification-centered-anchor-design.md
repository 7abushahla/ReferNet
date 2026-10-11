# Classification-centered anchor and fold-safe enrollment design

Date: 2026-10-11.

Status: architectural direction approved in conversation. This written specification is awaiting user review. Implementation and new job submission follow written-spec and implementation-plan review, not this document's creation.

## 1. Approved objective

Develop and evaluate a reference-conditioned document alteration detector for first-line GDRFA triage that can generalize to designs and data sources absent from training. Measure whether a genuine same-design reference improves detection relative to the same model family without reference information, while meeting the officer-device runtime requirement. New-design enrollment supplies genuine design information without target forgery labels or target-specific parameter training.

Protocol A establishes a credible query-only control against the published SIDTD template benchmark and tests the initial reference contribution on familiar designs. Protocol B tests held-out SIDTD designs. Protocol C tests transfer to FantasyID, and Protocol D tests cross-design transfer within IDNet. A strong Protocol A anchor is necessary for the planned comparison, but does not by itself establish the study's generalization claim. The selected architecture family must support all four protocols.

Preserve the three-input contract of query RGB, reference RGB, and a personal-data zone. The zone is design context, not a forgery mask. The immediate pilots address the Protocol A foundation within this broader study objective.

The user explicitly authorizes replacing the existing all-genuine registration and zone construction. It is not a design constraint. Existing running jobs, archived data, and completed results remain immutable historical experiments. Available GPUs permit parallel independent pilots, not an uncontrolled search or an automatic full fold sweep.

Research basis and completed direct-resize result are recorded in `evidence/architecture-decision-review-2026-10-11.md`. The implementation architecture is not frozen merely because the manuscript currently depicts ChangerEx.

## 2. Decisions and bounded scope

1. A direct query classifier is the foundation. Its classification score must not require a predicted alteration map.
2. Reference evidence is an optional extension, not a prerequisite for the query path to work.
3. First implement two diagnostic pilots and isolated fold-safe preprocessing. Do not implement an entire forensic ensemble, cross-attention stack, or all B-D loaders in this increment.
4. Start the conventional candidate with ImageNet ResNet50 and 299-square raw RGB. Keep ConvNeXt-Tiny and local-field alternatives in reserve.
5. Preserve the current encoder for one readout diagnostic. Its old preprocessing is retained only to make that historical comparison interpretable, not as the new benchmark default.
6. Select with source validation. No new automatic test evaluation, recompression experiment, fold sweep, or B-D evaluation belongs to this increment.

A strong result is not guaranteed. The current aggregate Protocol A targets remain AUC at least 0.99 and accuracy at least 0.97. A single pilot meeting them is a reason to confirm the candidate, not a completed benchmark result.

## 3. Two distinct comparison contracts

### Published-benchmark anchor

Use official SIDTD template image membership and labels without regrouping its train, validation, or test files. A genuine RGB-only model receives no zone, specimen, reference-derived transform, forgery target, quality metadata, or design ID as a learned input. It reads the released image bytes and decodes them normally. No JPEG re-encoding is added.

Dataset metadata can supply evaluation strata and provenance audits without becoming a model input. The loader must work when all reference, zone, and localization files are unavailable. Pilot execution opens train and validation images only. Official test membership may be inspected for split-integrity assertions, but test pixels and predictions are not used.

### Matched reference contribution

Compare query plus zone against the same query plus the same zone and a genuine reference. Match query preprocessing, encoder initialization, classification readout, trainable objective, optimization recipe, and sample membership. Report the extra reference computation and parameters honestly.

This is not the same information condition as RGB-only. The study must not attribute zone or registration information to the reference RGB alone. Wrong-design genuine pairs labeled altered are excluded from the primary comparison. A separate enrollment-consistency task can be evaluated later without changing genuine/altered labels here.

## 4. Pilot D, current-encoder direct readout

Question: does replacing the decoder/map-dependent classification route improve learning under the existing representation and training conditions?

Comparator: job 37932, `A-classification-only-registered-f0-s42-fp32-jfix-e100`, once completed and replayed.

Keep the current IA_ResNetV1c-18 query encoder, LEVIR initialization, widened four-channel query input, exchange disabled, and the current `batch_stats` policy. Take its deepest undifferenced query feature, global-average pool its 512 channels, and classify with a new 512-to-128-to-1 ReLU MLP. No FDAF, decoder feature, or map maximum is an ancestor of the classification output. A localization prediction is absent rather than a fabricated trained output.

For this diagnostic only, preserve the old registered 512 by 800 query/zone artifacts and pair augmentation implementation. Label the run `legacy_geometry_diagnostic` in machine-readable metadata and tables. It is not eligible as the final strictly train-only anchor. Keeping these artifacts unchanged is necessary to avoid conflating the readout intervention with the enrollment correction.

| Item | Pilot D |
|---|---|
| Fold and seed | Official fold 0, seed 42 |
| Training examples | 1,778 train, 222 validation |
| Duration | 100 epochs, no early stopping |
| Objective | Image BCE only, no dense target access required by the loss |
| Optimizer | AdamW, encoder LR 1e-4, new-head LR 1e-3, weight decay 0.05 |
| Schedule | Existing 1,000-step warm-up and cosine decay |
| Batch sizes | Train 8, validation 1 |
| Precision | FP32, AMP off, cuDNN and matrix-multiplication TF32 off |
| Augmentation | Existing corrected paired transforms, affine limit 8 px, query jitter 2 px at probability 0.25, photometric jitter on, JPEG off, no crop |
| Selection | Validation accuracy first, AUC tie-break, earlier epoch on exact ties |

Initialize the encoder identically to the comparator. Isolate new-head initialization from data-sampler and augmentation random streams. The smoke audit must verify unchanged encoder starting tensors and the intended sample/augmentation replay contract. Do not claim a readout-only contrast if these gates fail. Report changed classifier capacity and active parameter counts. This experiment replaces the complete readout route, not only the map-peak scalar.

## 5. Pilot R, RGB-only ResNet50 anchor

Question: can a conventional classifier with paper-adjacent input and training establish the requested benchmark anchor?

Use torchvision ResNet50 with explicitly identified `IMAGENET1K_V1` initialization, adaptive average pooling, and a newly initialized two-class linear head. Preserve conventional train-time BN updates and frozen running-stat evaluation for this query-only pilot. The absent reference cannot update any statistics. This pilot's normalization is not presumed to be the final reference-arm policy.

At validation, decode with OpenCV, convert BGR to RGB, bilinearly resize once to 299 by 299, scale by 255, and apply ImageNet mean/std. Verify tensor agreement against `scripts/score_sidtd_checkpoints.py:preprocess`. Do not call the weight enum's default inference transform, which would introduce a different crop/resize policy.

| Item | Pilot R |
|---|---|
| Fold and seed | Official fold 0, seed 42 |
| Training examples | 1,778 train, 222 validation |
| Duration | 100 epochs, no early stopping |
| Objective | Unweighted two-class cross entropy, no localization loss |
| Optimizer | SGD, LR 0.01, momentum 0.9, weight decay 0, Nesterov off |
| Schedule | ReduceLROnPlateau on deterministic validation CE, factor 0.9, patience 1, threshold 1e-4 relative, cooldown 0, min LR 0, eps 1e-6 |
| Batches | Microbatch 32, two-step accumulation, include incomplete final batches and complete the final accumulation window |
| Validation | Deterministic, unaugmented, no shuffled sampling requirement, report every validation image |
| Precision | FP32, AMP and both TF32 modes off |
| Selection | Validation accuracy first, AUC tie-break, earlier epoch on exact ties |

Training transforms follow the released baseline's intended family: random resized crop to 299 square with scale 0.8-1.0 and aspect ratio 0.75-4/3, horizontal and vertical flips independently at probability 0.5, random brightness/contrast at probability 0.2 with each limit 0.2, then ImageNet normalization. Pin the transform implementation/version and serialize every effective parameter, including interpolation and brightness semantics. No JPEG augmentation, query-only registration jitter, additional affine warp, target-based crop selection, or generated forgeries are added. These augmentations are a benchmark-adjacent starting recipe, not a claim that flips or cropping are optimal for deployment.

Source locations are `data/raw/sidtd/SIDTD_Dataset/SIDTD/models/Baseline/_utils.py`, `training_kfold_baseline.py`, and `models/explore/options/baseline_options.py`. The options specify batch 32, accumulation 2, LR 0.01, and 100 epochs. The release also contains a training call that uses training augmentations for validation and incomplete-loop edge cases. Do not copy these behaviors into a deterministic development experiment. Use sample-correct loss accounting and accumulation. Record these differences; do not claim byte-for-byte reproduction of the authors' historical training.

Export attack probability as softmax class 1, and bona fide score as class 0. Validate equivalence with the logit difference if the common metric interface uses a sigmoid. Resolve threshold ties consistently with two-class argmax. The new runtime may reuse shared metric/checkpoint infrastructure, but must not fabricate masks or references merely to satisfy the old eight-channel loader.

Pilot R is a complete baseline-recipe candidate. A difference from Pilot D does not isolate backbone capacity, initialization, normalization, or optimizer. The controlled readout comparison is D versus job 37932.

## 6. Fold-safe preprocessing and support provenance

Build into a new versioned directory. Never overwrite `data/processed`, existing ablation directories, cluster scratch data, checkpoints, or original raw files. New runs must explicitly select a data contract; there is no silent fallback to legacy global products.

For an official Protocol A fold, only genuine images in its training partition may contribute to learned/estimated design products. An immutable, independently sourced blank specimen is a declared exception, with its source and hash recorded; its fitted alignment still uses only the authorized training support bank. Validation/test genuines must not influence registration targets, zone masks, reference pixels, coverage maps, or specimen-alignment targets. A path being under a training-specific directory is insufficient; record its actual contributing image IDs and hashes.

For each official fold, form a fixed eligible enrollment pool from training genuines whose provenance components do not occur anywhere in that fold's validation or test partitions. Source/donor annotations are used offline to audit this exclusion, not to route inference queries. Preserve every official training and evaluation image in its original partition; filtering enrollment membership does not filter the classifier's examples. A metadata-only audit on 2026-10-11 found 56-68 eligible genuine images per design in fold 0, and 45-75 across all ten folds. These counts establish feasibility, not successful artifact construction.

Construct one fixed full-enrollment bank per design for validation and test. Its selection depends only on the enrolled design and fold/version, never on a query identity, component, true label, or source/donor annotation. During training only, use deterministic component-cross-fitted halves of the eligible enrollment pool so that a training query cannot contribute to its own reference. Training annotations may support this exclusion. Apply the same exclusions to upstream geometry and zones as to the median. Record the different support counts for training halves and evaluation banks. Empty eligible support is a hard failure, never permission to use a global median.

Every support product must record fold, support IDs, component IDs, raw-file hashes, annotation sources, transforms, code revision, and output hash. Resolve every artifact by official fold, design, named bank (`train_half0`, `train_half1`, or `enrollment`), and geometry version, including zones and aligned specimens. Reference pixels, geometry target and zone for a bank must derive from the same authorized support contract. Fields are collected from genuine layout annotations only, never from which field was altered. Query-specific coordinate estimation may use the query image but not its forgery target or true label. Distinguish this inference-time operation from fitting a shared design target on evaluation samples.

A strict source-only RGB loader needs none of these products and therefore need not wait for this preprocessing build. The legacy readout diagnostic also need not wait, because it deliberately uses an immutable historical input contract. Fold-safe preprocessing can be prepared in parallel, but its results become mandatory for subsequent design-assisted/reference experiments.

For the selected family, prefer retaining the query classification pixels and transforming the reference/zone into the query coordinate system. Preserve a valid-overlap plane. Failed registration marks both the reference and its transformed zone unavailable. Both matched arms use the same geometry target, alignment decision, and failure population; failed images remain in evaluation. Any query-only fallback uses the same model's query path, never a silently substituted RGB-anchor checkpoint. It must be trained and calibrated on source-only missing-context cases before being deployed as a thresholded decision. Until that condition is satisfied, report unavailability and require referral rather than silently clearing the document. This geometry direction is a new implementation and must pass synthetic coordinate tests before it is adopted.

For B-D, reserve and record genuine enrollment/support material separately from target evaluation identities before official evaluation. Do not silently shrink a published test set to create support while claiming an unchanged benchmark. If a dataset lacks an independent support pool, resolve and disclose the altered support/evaluation contract before launching that protocol. This increment establishes the provenance interface, not the final C/D data policy.

## 7. Same-family reference extension

After a competent query candidate is confirmed, preserve its query encoder and direct classifier. Add a small local comparison adapter that can access query appearance, aligned reference features, their difference, zone context, and overlap validity. Keep the query representation available without forced subtraction or early exchange.

The initial reference extension must preserve the base classifier's score through a zero-initialized final correction. Its hidden computation must remain capable of learning; do not zero every layer and inadvertently suppress gradients. Reference-off mode skips reference-dependent computation and all reference-induced buffer updates. An inference ablation of a reference-trained model does not replace a separately trained no-reference control.

Use a query-plus-zone control before claiming reference RGB benefit. Match its active query path and query pixels to the reference arm. Introduce one reference type first, then the median/specimen contrast. Do not add dense localization, corrected masks, new augmentation, a different encoder, and reference conditioning simultaneously. Localization remains a separate comparison with recipient-corrected targets, and inference never requires those targets.

The exact adapter dimensions and final geometry/input choice follow the pilots and a focused extension design. They are not prerequisites for implementing the two query pilots, and are not silently treated as validated by them.

## 8. Acceptance and scheduling gates

Before long training, require all of the following.

- Verified nonrandom initialization, checkpoint hashes, explicit load-key allowlists, and failure on unavailable requested weights.
- Identical official train/validation IDs and labels, correct class polarity, no duplicate IDs within a partition, and recorded provenance overlaps without changing official folds.
- Correct intended gradient routes, each trainable parameter in the optimizer exactly once, no gradient dependence on forbidden reference or target inputs.
- Pilot R's raw RGB inference succeeds with no reference, zone, mask, or label files. Pilot D retains its declared legacy zone dependency but reference RGB and forgery targets cannot affect its classification outputs or gradients. Malformed required inputs fail with the affected ID rather than falling back.
- For the new support builder, with partition and provenance manifests fixed, mutating held-out or excluded-half pixels and layout annotations leaves the corresponding frozen support products unchanged. Assert source/donor-component exclusion through every transitive artifact dependency, reject global fallback, and verify that switching reference type preserves query pixels and shared geometry/zone products. Validation/test reference lookup and inference must work with query component, source/donor, and forgery-annotation fields removed.
- Reference-off invariance where the model API accepts auxiliary inputs, including output, gradients, and mutable buffers.
- Stable single-document inference. Frozen-BN ResNet outputs must be insensitive to companion images; the legacy batch-stat diagnostic is evaluated at batch one only.
- Unit checks for BCE/logit/softmax metric equivalence, deterministic checkpoint selection, final partial accumulation windows, and truthful absence of localization metrics.
- A bounded training-only overfit probe uses 16 deterministically selected images (eight genuine, six inpainting, two copy-and-replace), with IDs stored before execution, augmentation off, minibatch eight, and at most 500 optimizer updates. Use a separate diagnostic AdamW optimizer at LR 1e-3 with no weight decay; evaluate every 50 steps. Acceptance is at least 15/16 correct and mean image loss below 0.15 under the condition's inference policy. Failure blocks long training for inspection, not an automatic expansion of probe length or image selection. Follow with a one-epoch train/validation smoke under the actual pilot recipe and fresh-process checkpoint replay within declared numerical tolerances.
- Every long pilot starts in a new process with newly loaded declared initialization, a new head, pristine BN state, fresh optimizer/scheduler and reset random streams. Overfit and smoke weights, buffers, optimizer state, and scheduler state are never used to initialize a 100-epoch pilot.
- Full configuration, source archive/hash, raw/split hashes, weight hashes, actual precision flags, and submit IDs in an immutable ledger.

Acceptance gates apply to the condition that owns their inputs. The new enrollment-isolation checks gate fold-safe reference experiments, not the raw RGB pilot or explicitly legacy readout diagnostic. Both pilot smokes can run independently on separate GPU nodes. A successful condition can launch its own 100-epoch run without waiting for the other condition, once the approved implementation plan's gates pass. Request one GPU per training run. Heavy data preparation, model probes, inference, and timing run in allocated compute jobs, never on the login node. Preserve current jobs. Do not automatically change batch size or precision after an OOM; report and document a recipe revision first.

Reuse existing Slurm account/environment/staging patterns. Run each condition from an immutable committed source snapshot and a unique output directory. A local commit changing the working tree must not alter a running job. No new fold sweep is authorized by the availability of additional GPUs.

## 9. Interpretation and next decision

Evaluate the selected checkpoint, not the highest AUC encountered or the final epoch. Save per-image validation scores, attack family, genuine false positives and misses, overall AUC/accuracy, and copy/inpainting results. Explain the uncertainty associated with only 16 fold-0 copy forgeries.

If R succeeds and D remains weak, advance the classification-centered family. If D succeeds, retain the current encoder as a credible candidate rather than changing it gratuitously. If neither succeeds, inspect training fit, subgroup errors and the corresponding recipe before adding a new architecture. Missing the target does not itself establish a software defect.

A bridge run changes the successful conventional model to the candidate study input/zone contract while retaining its recipe. If using the new support-safe geometry, use that contract for both the zone-only and reference arms. Do not compare a legacy registered control with a new fold-safe reference arm as if reference presence were the only change.

Confirm the provisional family on fold 0 with seed 43 and preselected fold 1 with seed 42, using validation only. Repeat the nearest comparator if making a superiority claim. These are development checks, not independent final evidence, because the official folds reuse images. Obtain a separate decision before the full fold sweep.

Before reference-family replication, benchmark batch-one FP32 CPU latency on named hardware with fixed thread count, 20 warm-ups and 200 source documents. Report median and p95 end-to-end and network-only latency, cached and uncached reference handling, and preprocessing cost. The target is p95 under one second end-to-end on the designated desktop-class CPU. A cluster CPU measurement is not evidence about an unspecified officer device.

Freeze architecture, recipes, enrollment rules and source-derived operating thresholds before B-D. Do not use FantasyID-trained detectors or teachers to initialize a supposedly FantasyID-unseen model. Keep robustness to encoding interventions separate from the native benchmark; no compression equalization is introduced in this increment.

## 10. Deliverables and rollback

- Isolated, tested query-classification routes and deterministic training/evaluation support.
- A fold-safe enrollment/preprocessing builder and provenance audit, without modifying historical artifacts.
- Two smoke-gated fold-0 pilot jobs, independently schedulable, plus complete result ledgers.
- A validation-only comparison and explicit continuation decision, not a claim of optimality.
- Reviewed implementation milestones committed and pushed. No model weights, source data, credentials, or large source archives committed.

Old checkpoints retain their architecture identifiers and loaders. New identifiers and schemas must not reinterpret them. The current manuscript is not rewritten to claim an unmeasured architecture result. Its method description and the main study plan are synchronized after the family is selected.
