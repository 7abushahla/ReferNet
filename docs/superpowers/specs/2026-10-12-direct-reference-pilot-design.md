# Direct Reference Pilot Design Proposal

Date 12 October 2026. **Status, proposal for review only.** This document does not implement a model or authorize an allocation or training. D/N independent outcome reviews have passed; provisional advancement still requires explicit design and execution-scope approval.

## 1. Recommendation and Evidence

Recommend the direct running-BN IA-ResNet18 family for the next bounded reference experiment. Preserve R as the raw RGB benchmark fallback and N as the retained map-dependent comparator. This is an engineering recommendation, not an optimality claim or final family selection.

| Selected source-validation condition | Epoch | Accuracy | AUC | Copy AUC |
|---|---:|---:|---:|---:|
| D_running_pilot, job 37982 | 79 | 0.977477 | 0.996557 | 0.979375 |
| N, job 37971 | 69 | 0.950450 | 0.979754 | 0.886250 |
| R raw RGB, job 37967 | 55 | 1.000000 | 1.000000 | 1.000000 |

These values are read from the respective `results/runs/anchor-pilots-v1/anchor-pilot-train-<job>/pilot/val_summary.json` files, not independently recomputed here. Each has 222 validation images, including 100 genuine, 16 copy, and 106 inpainting examples. The [D_running_pilot outcome review](../../../results/diagnostics/anchor-pilots-v1/reviews/d-running-pilot-37982-outcome-review.md) and [N outcome review](../../../results/diagnostics/anchor-pilots-v1/reviews/n-37971-outcome-review.md) both **PASS** and are archived in commit `f700fb9`; reviewers independently reconstructed metrics and selection. R has its separate [selected-validation review](../../../results/diagnostics/anchor-pilots-v1/reviews/r-37967-outcome-review.md).

D/N PASS is metadata/source/available-artifact consistency evidence. Source/archive/CSV/metadata byte hashes were independently checked, but absent checkpoint, pretrained-weight, and production-image bytes were not independently rehashed locally. GPU tensors, gradients, BN updates, companion tests, and replay remain recorded allocated-GPU evidence, not independently repeated GPU measurements or pixel-access syscall audits. Neither review selects a final family or clears the new reference consumer.

D already has the desired four-channel input, a direct classification route, and stable stored-statistics inference. Its source-validation result motivates a small support-safe bridge before reference training. R's higher score makes it the credible benchmark fallback, not evidence that its unimplemented zone/reference extension is superior. The recipes and information conditions differ.

| Alternative | Benefit | Reason not to start there |
|---|---|---|
| Recommended D, context moved into the raw query canvas | Reuses the tested direct head and 512-by-800 field resolution; retains undifferenced query evidence. | New geometry and fold-safe zones still need bridge confirmation. |
| R, 299-square query with a newly designed zone/reference bridge | Strongest current raw benchmark and conventional classifier. | Requires an additional input/head bridge; existing perfect validation does not validate reference use or higher-resolution field comparison. Retain as fallback. |
| N, support-safe registered canvas and current decoder/referral route | Retains the existing map-dependent family as a comparator. | Lower current validation competence, localization supervision, and additional reference/normalization design make it a larger intervention. No new N run is proposed. |

A fold-safe canonical query-to-bank warp is another geometry option, but would resample classification pixels. Prefer inverse-warped context as specified below. Do not switch geometry automatically if the proposal fails its coordinate gates.

## 2. Concrete Model Candidate

The base path is unchanged in computation. `[normalized query RGB, query-coordinate zone]` gives four channels to the IA_ResNetV1c-18 encoder. Traverse it once, GAP its 512-channel deepest feature, then apply the existing 512-to-128-to-1 ReLU MLP. No decoder, FDAF, map peak, localization output, or map loss is added.

Use the existing verified offline LEVIR builder and head seed 42. Retain all 22 BN modules, including each module's running mean, running variance, batch counter, and initialized affine parameters. Construct additional layers in an isolated RNG context after the base, so their creation cannot change base tensors or sampler/worker streams. Both arms start from bit-identical fresh base parameters/buffers, not from epoch 79 or bridge/smoke weights.

**Reference path.** Input the inverse-warped median RGB plus the same query-coordinate zone to the same encoder parameters. Stop the reference traversal after `layer2`, the stride-eight, 128-channel stage. The query continues through all four stages for its original head. There is no second encoder parameter copy.

During training, execute the query first with its ordinary running-BN updates. For the reference pass only, temporarily put the traversed BN modules in stored-statistics evaluation mode, preserving and restoring each training flag even on error. Use their current stored buffers, which include that step's query update. Do not use `no_grad`, detach reference features, or update reference-pass buffers. Gradients can reach shared convolution and BN affine parameters from both paths once the correction learns. Evaluation freezes all BN and is per-document independent.

**Local adapter, proposed fixed dimensions.** At input 512 by 800, assert query/reference stage-two tensors have shape `(batch, 128, 64, 100)` on the installed stack.

| Operation | Output or setting |
|---|---|
| Shared 1-by-1 projection on query/reference stage-two features | 128 to 32 channels, stride one, bias enabled |
| Nearest-neighbor zone and overlap at stride eight | One channel each, same 64-by-100 grid |
| Concatenate projected query, projected reference, absolute difference, zone, overlap | 98 channels; mask comparison features outside overlap |
| 3-by-3 convolution, GroupNorm, ReLU | 98 to 32, stride one, padding one, no convolution bias; eight GN groups |
| 3-by-3 convolution, GroupNorm, ReLU | 32 to 32, otherwise identical settings |
| Valid-masked spatial mean and maximum, concatenated | 64 values; exclude invalid cells from both reductions |
| Linear final correction | 64 to one; weight and bias exactly zero initially |

Initialize other convolutions conventionally with Kaiming initialization, GN scale one/bias zero, and the projection bias zero. There is no adapter BN or dropout. The final attack logit is the base query logit plus the scalar correction. Empty valid pooling support gives no correction and explicit context unavailability, never a fabricated maximum or NaN.

The zero final layer preserves the base score for arbitrary valid references at initialization. On the first backward pass, final-layer gradients must be finite and active; hidden/reference gradients may initially be zero. On a deterministic nondegenerate fixture, the first optimizer update must enable nonzero hidden/projection/reference-path gradients on the next backward pass. Do not zero all adapter layers.

A 33-pixel field spans roughly four stride-eight cells. This motivates retaining local evidence before pooling rather than comparing only GAP vectors. It is an engineering hypothesis, not a proven optimal scale or an annotation-dependent inference rule. Convolutions and GN are ordinary dense operations, not claimed masked normalization. If actual feature geometry or CPU cost conflicts, propose a reviewed simplification rather than adding levels or conducting an automatic adapter search.

**Reference-off.** Skip the reference encoder and adapter entirely. Keep the same base model and prepared query/zone. Reference-off logits, query gradients, and common buffer updates cannot depend on optional reference contents. Exclude inactive adapter tensors from the control optimizer and report active versus total parameters. The separately trained control is primary; disabling reference in a reference-trained checkpoint is only a secondary ablation.

## 3. Fold-safe Query Canvas Bridge

Decode query and bank target/median with OpenCV as **BGR**. Reuse `to_canvas` to letterbox the query BGR into 512 by 800 with its native aspect ratio and validity plane. Retain that resulting query canvas pixel tensor. Do not register or re-encode it. Pass BGR query/target canvases to `_features`, whose grayscale conversion expects BGR, never silently RGB. Estimate **query-canvas-to-selected-bank-target** similarity from those image features only, then inverse-warp bank median, zone, and validity directly into the query canvas. Both grids are 512 by 800, so this operation needs no nominal native-letterbox inversion or rounding assumption. Keep color images in BGR throughout geometric/photometric primitives; convert query and aligned median to RGB only afterward for ImageNet normalization and model tensors.

Use `resolve_enrollment_bank` against the frozen `anchor_enrollment_v1` / `bank_similarity_v1` authority. Its 590 eligible genuine training contributors form 30 banks across ten designs. Train queries use the opposite component half for every context dependency; validation uses the fixed full `enrollment` bank by design/fold/version, with no query component supplied. Preserve all 1,778 train and 222 validation IDs and labels. Support eligibility does not filter classifier examples.

Pin the completed manifest hash `7691452ecb4d160e8767d277ba3ccf209355f16b3f49f85e52ff596c8a2719e8` and authenticate its selected products. The local bundle contains metadata, not PNGs. Actual consumption needs the declared cluster products and reviewed read/dependency trace. The existing integrity pass does not clear detector consumption.

### Registration and validity contract

- Reuse image-only SIFT with `N_FEATURES=3000`, Lowe ratio `0.75`, and RANSAC similarity reprojection threshold `3.0` pixels. `estimate_similarity` requires at least 12 keypoints/matches at its existing checks; accept a fit only with the existing `MIN_INLIERS=40` and a finite, invertible, orientation-preserving similarity matrix. Reuse the bank similarity-axis validation rather than accepting arbitrary affine matrices.
- Warp median BGR linearly and zone/valid masks with nearest-neighbor interpolation, constant-zero borders, and explicitly tested inverse direction. Define overlap as query validity intersected with inverse-warped bank `target_valid` and final `valid`. Clip transformed zone to this overlap. Reference-off uses exactly the same target, transform, zone, overlap, and availability decision.
- Training-half geometry metadata has 590 successful target-fit observations, with minimum 628, upper-middle observation 1,101, and maximum 2,283 inliers. These are support-to-bank registrations, not query fits or a validated acceptance range. No new scale, angle, translation, inlier-ratio, or overlap-fraction cutoff is invented from them. Record diagnostics in the training-only coordinate stage under the approved frozen rules. Visual coordinate-example review may occur after the gated validation smoke but must finish before long training. Do not tune new rejection cutoffs on validation.
- Too few matches/inliers, a failed/invalid fit, or empty pixel/feature overlap marks reference and zone unavailable. Compute stride-eight overlap availability before either mode's query forward. Zero the unavailable zone, skip reference computation, retain the query and original label, and record the reason. Unknown/missing/corrupt authenticated products are hard errors, not natural-registration failures or permission for global/identity-transform fallback.
- All failures remain in the evaluation population. Export diagnostic query-path scores for complete-population metrics alongside availability/referral counts, never as calibrated automatic clearance on unavailable context. No substitution of an R checkpoint. Thresholded missing-context clearance remains outside this proposal.
- Evaluation inference takes query pixels and an externally declared enrolled design only. Query annotations, identity/component/source/donor routing, true labels, attack family, forged fields, and masks are absent. Metric labels remain in a separate evaluation table. No official-test pixels are opened.

This bridge changes **geometry, fold-safe zones, and matched jitter together** relative to D's legacy globally prepared registered inputs. D versus bridge is a package confirmation, not a causal single-variable estimate. Only bridge reference-off versus reference-on isolates the declared incremental reference condition.

### Matched augmentation

Preserve D's draw distribution, no crop, affine translation up to eight pixels and rotation up to one degree, plus integer translation jitter up to two pixels with probability 0.25. Apply each geometric draw coherently to query, zone, aligned reference, and overlap. Unlike legacy `jitter_zone=false`, the bridge moves zone and context with query jitter. Use linear RGB/nearest mask interpolation; retain constant-zero affine borders and replicate RGB jitter borders, with constant-zero validity borders so replicated pixels are not valid comparison evidence.

Apply the existing query photometry with strength 0.2 in both arms and ImageNet RGB normalization. Initially apply no additional reference photometry. Reference RGB still receives the same geometric transforms. Keep query transform/photometry and sampler/worker random draws isolated from reference processing and identical between arms. No JPEG augmentation, mask-directed crop, independent reference jitter, or wrong-design pair is introduced. Evaluation is unaugmented.

## 4. Fixed Recipe and Selection

| Setting | Both fresh matched arms |
|---|---|
| Membership, duration | Official fold 0, seed 42, 100 epochs, no early stopping |
| Objective | Unweighted image BCE on attack logit; localization weight zero; no dense target access |
| Optimizer | AdamW, shared encoder LR 1e-4, head and active adapter LR 1e-3, weight decay 0.05, betas 0.9/0.999, epsilon 1e-8 |
| Schedule | Existing warmup-cosine helper, 1,000 warmup updates capped at 10% of total updates; gradient clip norm 5 |
| Batches | Train eight, accumulation one, preserve D's `drop_last=true`; validation one, no dropping; preserve all membership and record shuffled tail omissions |
| Precision | FP32, AMP/TF32 off, deterministic-algorithm flags as in D; `training_bit_reproducible=false`, no promise of bitwise training repeatability |
| Selection | Existing `train.selection_key`, validation accuracy first, AUC second, earlier exact tie |

Build both modes in the same frozen source before the bridge run. Reference-on later starts from identical original LEVIR/head initialization, not from the selected bridge checkpoint. The bridge run itself is the independently trained matched control and is not repeated just to label it a control.

Preserve `metrics.py` polarity and ties, genuine score `>= 0.5` is genuine, so attack probability must be **strictly greater than 0.5** to call altered. A zero attack logit is a genuine tie. Reuse canonical prediction families `text_copy`/`text_inpaint`, human subgroup keys `copy`/`inpainting`, and attack-versus-same-genuine subgroup AUC. Recompute selected-validation metrics and checkpoint/history/replay identities before interpretation.

## 5. Minimum Files and Gates

Prefer two focused production modules, proposed `data/direct_reference.py` for authenticated bridge/dataset/augmentation and `models/direct_reference.py` for the wrapper/local adapter. Reuse `anchor_changer` initialization and query traversal, enrollment resolution, registration/rectification primitives, metrics, selection, checkpoint and gate helpers. Add narrow contract/config/runner/packaging wiring and focused `test_direct_reference_data.py` / `test_direct_reference_model.py` tests. Do not rewrite legacy pairs, pilots, encoders, or existing model identities.

Model and bridge implementation can proceed in parallel **after approval**. Reuse the preparation plan's G1-G10 and support-readiness consumer tests by reference, with these concrete checks.

- Synthetic red/green coordinates must cover identity, inverse translation/rotation/scale, non-square source letterboxing, padding/partial overlap, and coherent affine/jitter. On/off query/zone tensors and selected product reads must match exactly. Repeat transitive component/unused-half exclusion through transforms and caches, and annotation-free evaluation.
- Preserve base initialization hashes and exactly one query traversal. Reference-on adds only stem/layer1/layer2 traversals. All 22 query BN counters increment once per training forward; reference adds zero increments and zero running-buffer changes. Restore temporary modes on exceptions. Assert no reference calls in off mode, optimizer uniqueness, zero-correction identity, and post-first-step reference gradients.
- Check eight distinct queries with batch sizes one, two, four, eight, first/middle/last positions and varied paired companions. Both modes and unavailable cases must agree within FP32 `atol=1e-5`, `rtol=1e-4`; all evaluation buffers stay unchanged. Repeat on saved checkpoints and fresh-process replay with identical IDs/order and pinned source/config/weight/bank hashes. Reject old D/N/diagnostic checkpoint identities.
- After explicit diagnostic-scope approval, propose **one allocation with two gated stages**, reusing approved train/validation staging upfront. Staging authorized files is not permission to read them; no deferred or dynamic validation staging is introduced. First run training-only coordinate checks and actual model/overfit gates in both modes, using the existing 16-image overfit criterion. Freeze IDs before pixel access and cover all designs in the coordinate sample. Phase-specific loader/read guards permit only training queries and genuine-training bank products; tests assert zero validation pixel opens. Failure stops the allocation before validation reads. No test pixels are staged in any phase.
- Only after the programmatic training gates pass, start fresh one-epoch actual-recipe smokes on the first 32 sorted train and first 32 sorted validation IDs, followed by fresh-process checkpoint replay, **for each mode**. Stage-two loader/read tests permit only the declared train/validation pixels. Reset original weights/head/BN, optimizer/scheduler, and RNG streams between phases/modes. Validation uses full genuine-training enrollment banks and cannot fit shared bank geometry, zones or support, or tune rejection rules. Image-only registration of each validation query to its frozen bank remains permitted and uses the same fixed registration algorithm. No test pixels are opened. Independently review the complete coordinate/overfit/smoke/replay outputs before any 100-epoch launch; no diagnostic state seeds long training. Execution still requires approval, not this proposal alone. The later 100-epoch wrapper likewise reuses upfront train/validation staging.

## 6. Small Sequential Experiment and Review Decisions

1. Review this concrete design alongside the passed D/N outcome reviews. Approve model/bridge implementation and integrated synthetic gates, not a new benchmark regime.
2. Separately approve the proposed single-allocation diagnostic scope. Independently review its training-only coordinate/overfit stage and gated fresh 32-train/32-validation smokes/replays in both modes, including registration failures and actual feature/BN/gradient behavior, before detector consumption in long training.
3. Request **one D query-plus-zone bridge**, fold 0 seed 42 for 100 epochs, as the first substantive experiment. Assess selected validation competence, all-image availability/referrals, false positives, both attack families, and stability. As a provisional source-validation advancement criterion, require accuracy at least 0.97 and AUC at least 0.99 without treating one fold as aggregate acceptance; registration problems can block advancement despite those numbers.
4. If the bridge is competent and reviewed, request one reference-on run from the same frozen code, original initialization, recipe, and data contract. Reuse the completed bridge as its control. If not, inspect the bridge rather than launch reference-on or automatically train another family. No extra N/ResNet pilot, repeat control, sweep, or B-D evaluation follows automatically.

CPU runtime engineering can proceed in parallel under its existing separately authorized same-host contract. Measure the incremental reference/adapter and alignment cost, including cached/uncached handling, before replication or deployment claims. Missing runtime is not measured. It does not block this design preparation or create another benchmark regime.

**Remaining review decisions are narrow.** Approve the candidate blocks, geometry/coherent-jitter bridge, product-consumption readiness, and the concrete two-stage diagnostic scope. The proposed sequence includes the governing train/validation smoke and replay requirement without a waiver or unresolved second gate decision. Coordinate/support construction remains training-only. New query-quality rejection bounds, if needed, require source-training evidence and approval before their use, never tuning on validation cases.

## 7. Scientific Limits and Governing Documents

Native SIDTD encoding associations remain a limitation. Higher AUC or copy performance does not establish alteration-specific causal evidence or encoding robustness. Preserve the native raw R benchmark and disclose historical test feedback; do not claim an untouched test or select on later-protocol results. Sixteen validation copy attacks imply substantial uncertainty. No reference gain, held-out-design competence, cross-source generalization, calibrated missing-context clearance, or officer-device latency is established yet.

This proposal refines the approved classification-centered spec's Sections 3 and 6-9. Provenance and consumer obligations are inherited from `docs/anchor-reference-integration-readiness-2026-10-12.md` and `docs/superpowers/plans/2026-10-12-matched-reference-extension-preparation.md`. Existing R/D/N and bounded D-running contracts, checkpoints, inputs, and running jobs remain immutable. Approval of this proposal is not a final family choice or training submission.
