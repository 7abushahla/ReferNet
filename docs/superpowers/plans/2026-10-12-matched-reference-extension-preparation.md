# Matched Reference Extension Preparation Plan

> **For agentic workers:** Use `superpowers:executing-plans` only after the family, extension design, interfaces, implementation ownership, and execution scope receive explicit approval. This document authorizes neither implementation nor execution. Checkboxes below describe future work, not work approved for this preparation task.

**Status:** Branch-conditional preparation requested on 12 October 2026. No family or fusion design is selected. No model run, reference training, sweep, submission, or commit is authorized by this document.

**Completed-pilot follow-up:** The reviewed D/N validation outcomes now motivate the concrete proposal in `docs/superpowers/specs/2026-10-12-direct-reference-pilot-design.md`. It specifies the provisional D route, local adapter, context-to-query bridge, and first matched experiment. That proposal retains the execution boundaries here and does not constitute a final family decision or launch approval.

**Goal:** Prepare a same-family comparison of query-plus-zone evidence against the identical query-plus-zone path with optional genuine same-design reference-conditioned evidence.

**Architecture:** Preserve the chosen family's query evidence and classification route. The direct-family branch must satisfy the approved local-comparison and zero-initialized final-correction requirements. The N branch requires its own focused extension design and cannot silently inherit either a direct-head adapter or the historical reference arm.

**Tech Stack:** Existing PyTorch, torchvision, Open-CD, OpenCV, pandas, pytest, and reviewed allocated-compute packaging. No dependency changes are proposed.

**Spec:** `docs/superpowers/specs/2026-10-11-classification-centered-anchor-design.md`, especially Sections 3, 6, 7, 8, and 9. Read alongside `docs/anchor-pilot-interfaces.md`, the executable contracts, and the separately approved `docs/superpowers/plans/2026-10-12-d-running-pilot.md`. The older interface note does not enumerate the later D-running contracts; executable identifiers remain authoritative.

## 1. Authorization and Scientific Boundaries

- Preparation owns this document only. Production files, configurations, tests, inputs, existing artifacts, and running jobs remain unchanged.
- Select a provisional family from reviewed selected-source-validation evidence, genuine false positives, attack subsets, stability, support-safe readiness, and runtime. No automatic highest-AUC winner, and no family choice is made here.
- Preserve official Protocol A classifier membership and labels, including images whose provenance excludes them from enrollment. Fold 0 has 1,778 train, 222 validation, and 222 test members. Test membership is metadata-only during development; test pixels and predictions are prohibited.
- Do not use Protocol B reference gaps, FantasyID, IDNet, historical test scores, or test-based thresholds as the selector. Historical test feedback already exists and must remain disclosed, not described as untouched final evidence.
- The primary contrast is `query + zone` versus `query + identical zone + reference RGB` within one approved family and recipe. It is not R raw RGB versus a design-assisted reference model.
- Exclude wrong-design hard negatives, relabeling of genuine examples, new generated forgeries, compression equalization, extra localization, and additional augmentation from the initial matched comparison.
- Use one approved reference type first. Median versus independently sourced specimen is a subsequent contrast, not another change bundled into the first comparison.
- The aggregate Protocol A targets remain AUC at least 0.99 and accuracy at least 0.97. One fold, a smoke, or support-integrity verification cannot establish those aggregate targets or reference superiority.
- Implementation approval, actual correctness-gate review, substantive matched-run approval, confirmation approval, and final-test approval are separate decisions. Available allocations do not authorize any of them.

## 2. Branch Conditions, Not Family Selection

| Reviewed outcome | Conditional implementation path | Required bridge and unresolved work |
|---|---|---|
| R is provisionally competent and approved | Preserve its ResNet50 query encoder, pooling, and two-class classifier. Implement only the subsequently approved small reference extension. | Establish a support-safe query-plus-zone control while retaining R's recipe. Approve zone encoding, input geometry, feature taps, correction-to-class-logit interface, and reference normalization. R raw RGB remains a separate baseline. |
| A direct IA-ResNet18 candidate is provisionally competent and approved | Preserve the query-only encoder traversal, deepest-feature GAP, and 512-to-128-to-1 MLP. Reference evidence cannot replace the undifferenced query path. | Replace legacy support/geometry with a reviewed bridge in both arms. Freeze the exact starting direct contract and normalization. `D_running_pilot` is not automatically selected by its training authorization. |
| Only legacy D is a credible candidate | Request an explicit normalization and support-safe confirmation decision before designing a matched arm. | D's batch-statistics singleton policy does not satisfy ordinary batched-companion invariance. Changing it to running BN is a new declared condition, not a reinterpretation of D weights or evidence. |
| N is provisionally competent and approved | Retain its explicitly tested query-driven decoder/map/referral route as the base evidence. Prepare a focused same-family reference-on and normalization design. | Approve the actual reference route, missing-stream semantics, initial score-preservation mechanism, and matched localization supervision. Do not replace N with a direct MLP or assume historical FDAF reference-on is validated. |
| Several candidates are competent | Compare extension readiness, stability, subgroup uncertainty, genuine false positives, and measured CPU cost before approval. | Query-only rank is insufficient to select a reference family. No architectural branch executes until the decision is recorded. |
| No candidate is competent, or evidence is incomplete | Hold substantive reference training. Inspect the bounded source-only evidence and recipe. | This preparation remains reusable. It does not authorize a new backbone, an ensemble, or a longer diagnostic. |

The original `D_running` identifier remains diagnostic-only and full-training blocked. The separately authorized `D_running_pilot` uses the same computation but a distinct execution identity. Neither legacy geometry nor a diagnostic checkpoint becomes a support-safe reference contract through renaming.

## 3. Decisions the Spec Does Not Lock

Each row needs a recorded choice and approval. Recommendations are planning rationale only, not defaults that an implementer may silently apply.

| Decision | Options and tradeoffs | Recommendation for later approval |
|---|---|---|
| Family and base route | R direct, IA direct, or N map-dependent. Existing pilots differ in initialization, losses, geometry, and optimization. | Do not rank here. Require selected-validation evidence and the branch-specific readiness checks above. |
| Classification geometry and size | Preserve query pixels and move context into query coordinates, as preferred by the spec; alternatively approve a new shared registered canvas. The latter changes query evidence and adds resampling. | Prefer the spec's context-to-query direction. Freeze dimensions, resize/crop conventions, padding, and coordinate composition after the family decision. Existing 299-square R and 512-by-800 legacy inputs are not an automatic extension contract. |
| Zone integration | An additional stem channel versus a separate context input to the approved query path. Stem widening changes the encoder interface; a separate input needs an approved consumption rule. | Keep tensors separate at the data boundary so both choices can be audited. Do not choose where or how the zone enters the network in this plan. Require zero-preserving initialization if a widened stem is chosen. |
| Direct comparison adapter | One selected feature level versus an approved small multilevel local comparison. Resolution, field-scale evidence, capacity, and CPU cost differ. | Approve the smallest local adapter justified by field-scale coverage and runtime. The spec requires access to query appearance, aligned reference features, difference, zone, and overlap, but does not fix levels, widths, operators, pooling, normalization, or hidden layers. No new fusion block is specified here. |
| N reference-on route | Audit the historical current-family reference path against the new requirements, or commission a focused replacement design within N's family. Historical reuse reduces new code but does not guarantee initial score identity or correct normalization. | Require the focused N design first. Any route must explicitly satisfy the common evidence and missing-context gates, or obtain a spec amendment before implementation. |
| Reference encoder and normalization | Share encoder parameters with an isolated reference-statistics policy, or use a separately declared encoder instance. Separate weights/buffers increase parameters and may change optimization; shared training BN can contaminate the query path. | Prefer preserving the base query normalization and preventing reference passes from updating its BN buffers, subject to approval of the actual mechanism. Freeze versus independently updating reference statistics, parameter sharing, and reference gradient policy remain unresolved. No joint query/reference BN batch is assumed. |
| Initial state of matched arms | Fresh identical official offline initialization and seed, or an explicitly approved common compatible bridge checkpoint for both arms. A bridge warm start adds a supervised selection stage. | Prefer fresh declared initialization under the existing long-pilot freshness rule. A common warm start needs explicit approval of the changed training contract and history. Never initialize one arm from a better validation checkpoint or from diagnostic/smoke weights. |
| First reference type | Train-only median already has a builder; independent specimen requires an authenticated source and successful bank alignment. | Consider median first to avoid an additional sourcing intervention. This is not a selected reference policy. Approve the type before any run. |
| Missing-context learning and clearance | Train/calibrate the same query path on declared source-only missing-context cases, or provide an availability/referral result without automatic clearance. Dropout frequency and loss handling change the recipe. | Refer on unavailability until the missing-context training and source-derived threshold policy are approved and verified. No silent switch to the R checkpoint. |
| Objective and score representation | Preserve R's CE and two-class polarity, IA direct BCE, or N's image plus matched localization objective. A uniform replacement objective is another intervention. | Retain the selected recipe in both arms. Freeze how any correction maps to the existing score interface, threshold ties, and any N target correction before training. |

**Coordinator action required:** The approved spec defines evidence access and comparison constraints, not actual network blocks. It does not lock the zone-consumption block, adapter dimensions/operators, reference feature taps, weight/statistics sharing, N reference-on computation, final geometry, or missing-context curriculum. Those are explicit design-review prerequisites. This plan cannot be treated as architecture approval.

## 4. Proposed Shared Query-plus-Zone Bridge Contract

These are logical interfaces for later review, not existing APIs. They deliberately keep the model-family decision outside the loader. Final names and identifiers must be approved before implementation.

### Common sample boundary

| Field | Contract |
|---|---|
| `query_rgb` | Float32 RGB tensor with shape `(batch, 3, height, width)`, normalized under the approved family recipe. Both arms receive identical query tensors for the same image and transform key. No extra query registration or JPEG re-encoding is introduced in one arm. |
| `zone` | Float32 design-context plane `(batch, 1, height, width)`, derived solely from authorized genuine layouts. It is not an altered-field or forgery mask. Its eventual network encoding remains a design decision. |
| `overlap_valid` | Binary plane `(batch, 1, height, width)` describing valid shared geometric support. It is not a training localization target. Invalid padding cannot become usable comparison evidence. |
| `context_available` | Boolean `(batch,)` with an audited reason for unavailable context. A zero zone with availability false is not confused with an available empty field region. |
| `reference_rgb` | Optional same-design reference RGB in the identical coordinate system, with its own declared validity when needed. Absent in reference-off model execution. It is never synthesized from a query forgery or evaluation population. |
| `label` and optional `localization_target` | Training-loss or offline evaluation fields only. They are not model arguments. Direct branches need no localization target. N training supervision requires separate approval and identical treatment in both arms. |
| `audit_meta` | Image ID, enrolled design key, fold/bank/geometry versions, artifact hashes, transform, alignment result, and metric strata. Design key is for external support lookup, not a learned design-ID embedding. Source/donor/component provenance is offline/training-only authority. |

Proposed boundary operations, to be defined in the bridge task, are `prepare_query_context(query_rgb, enrolled_design, authenticated_bank, transform_key) -> common_sample` and a model wrapper consuming only the permitted tensor fields plus an explicit `reference_enabled` mode. `authenticated_bank` means a resolved bank whose products were checked against an externally pinned built-manifest hash. The loader does not infer a bank from the query's true label or attack family.

### Geometry and alignment

1. Decode the query once. Fit any query-to-bank coordinate estimate from query pixels and the frozen bank target only. Do not read its annotations, source/donor IDs, forged-field names, or true label.
2. Existing bank transforms map native contributor coordinates to bank coordinates. For a new query, estimate its own image-based transform. Do not look up an annotation-derived contributor transform as an inference shortcut.
3. If the preferred context-to-query design is approved, compose `bank_to_model = query_native_to_model @ inverse(query_native_to_bank)`. If estimation uses a temporary letterbox, include its native-to-canvas transform in `query_native_to_bank`. Query classification pixels follow the agreed query recipe, not the temporary alignment canvas.
4. Warp reference RGB with the approved linear interpolation and masks with nearest-neighbor interpolation. Record axes, transform direction, half-pixel/rounding conventions, border fill, coverage intersection, and composition order. Test these before trusting coordinates.
5. Both arms use the same frozen target, alignment decision, zone, overlap rule, and failure population. Query-plus-zone is already design assisted, including any common registration-target information. Only the additional reference RGB/features are credited to the reference-on contrast.
6. Freeze the common bridge independently of reference type. Switching median to specimen cannot rebuild the target or zone or change query pixels. Any specimen-specific validity is additional reference metadata, not permission to rederive the shared zone. Differences in reference availability remain disclosed and all official examples remain accounted for.
7. Failed or empty-overlap alignment makes both the reference and transformed zone unavailable. Keep the image in evaluation. Corrupt authenticated products, unknown required banks, and malformed inputs are hard errors, not identity-transform/global-reference fallbacks. Unenrolled deployment designs require enrollment or referral.

### Matched stochastic preprocessing

Use one sampled geometry transform for all corresponding planes and masks. Keep query photometry and query/sampler/worker random streams identical across arms for paired seeds. Isolate reference-only augmentation draws so enabling reference processing cannot shift the next query's random draw. Serialize effective interpolation, normalization, crop, augmentation, missing-context policy, worker seeds, and final partial-batch behavior. Reusing legacy pair primitives does not establish stochastic-stream matching by itself.

## 5. Fold-safe Bank Interfaces and Dependency Closure

Reuse the following actual interfaces from `src/refer_or_clear/data/enrollment.py`.

```text
make_enrollment_manifest(records: pd.DataFrame, split: pd.DataFrame,
                         fold: int, raw_root: Path) -> dict
build_enrollment_products(manifest: dict, raw_root: Path, output_root: Path,
                          specimen_root: Path | None = None) -> dict
resolve_enrollment_bank(manifest: dict, design_id: str, role: str,
                         training_component: str | None = None) -> dict
```

- Manifest creation audits the transitive identity/source/donor graph and official fold binding. Eligible support is genuine training material whose entire provenance component is absent from validation/test. Do not change the official classifier split to improve support counts.
- Training resolution uses `train_half{1 - fold_of(training_component)}`. The excluded query component cannot contribute to the median, geometry anchor/target, zone, coverage, valid maps, specimen-alignment target, or any derived feature cache. The graph is global provenance authority; these halves are not the official classifier folds.
- Validation and later authorized evaluation use the fixed full `enrollment` bank for the enrolled design/fold/version. `training_component=None` is required. Evaluation resolution explicitly rejects query component provenance. Source/donor IDs, identity IDs, labels, attack strata, and query annotations cannot route support.
- Built banks record authorized and successfully registered contributors separately. Failed support contributors are recorded, not identity-warped into products. Empty authorized or successfully registered support fails closed.
- The bank schema is `anchor_enrollment_v1`, geometry `bank_similarity_v1`, named banks `train_half0`, `train_half1`, and `enrollment`. Current products include target, target coverage/valid, median, coverage/valid, zone, geometry JSON, and per-bank dependency manifest. Specimen and specimen validity are optional declared products.
- Support construction may use authorized genuine layout annotations offline. A shared VIA file's unrelated held-out entries are archival metadata, not bank dependencies. Distinguish whole annotation-file archival hashes from canonical authorized-layout-entry hashes when testing isolation.
- External blanks require `external_blank_v1` declarations, source/hash/independence evidence, complete-inventory exclusion checks, and bank-specific alignment using authorized support. Do not silently use a dataset image, alias/hardlink, donor, global template, or undeclared blank as independent material.
- Reuse `scripts/prepare_anchor_enrollment.py` for construction and `scripts/verify_anchor_enrollment_products.py` for the existing read-only integrity layer. Its `verify_products(manifest_path, expected_manifest_sha256, *, output_root=None)` checks externally authenticated product bytes/JSON without decoding pixels. It expressly does not grant detector clearance and currently binds fold 0. Fold-1 confirmation requires a reviewed extension of that verifier's fold authority, not bypassing the check.
- A resolver alone does not authenticate file bytes or construct query-coordinate tensors. The new loader must verify the pinned manifest/product authority before reading pixels, retain path containment under explicit relocation, and reject stale/mixed fold, bank, geometry, or cache identifiers.
- Feature-cache keys must bind family/architecture, weight and buffer state, preprocessing, fold/design/bank/reference type, and all input product hashes. During trainable encoder updates, cached features are invalid unless an explicitly approved frozen-reference contract makes them stable.

## 6. Matched Computation, Initialization, Optimization, and Selection

### Computation and evidence

- Execute the same query-plus-zone path exactly once in each arm. Preserve undifferenced query evidence. Reference-on adds only the approved reference traversal and comparison computation; report its additional active/total parameters, FLOPs if measured, and latency honestly.
- For a direct branch, the approved contract is base query-plus-zone classification plus a final correction initialized to zero. Check identity for arbitrary valid reference inputs before training, including any two-class logits used by R. Hidden adapter layers must be capable of learning. Zeroing every layer is prohibited because it can suppress gradients.
- At initial zero correction, require finite nonzero gradients in the final correction parameters. Hidden-layer/reference gradients may initially be zero through that final zero layer; require the intended routes to become finite and active after the correction learns in the bounded probe. Do not report those initial zeros as a broken route without this distinction.
- N must demonstrate the agreed initial score preservation using its approved map-dependent route. Merely inserting a real reference into the historical decoder does not prove it. Keep query evidence and reference-conditioned evidence separately observable for audit without fabricating localization outputs for direct branches.
- Reference-off skips reference encoders, projections, adapters, and reference-induced mutable-state updates. A blank image/feature sent through shared normalization is not reference-off. The query path may use common approved zone/geometry information, which is held fixed in reference-off invariance tests.
- Independently train the reference-off control. Disabling reference at inference in a reference-trained model is a secondary ablation, not that control. Any prediction-decomposition diagnostic is secondary to the two separately trained selected-checkpoint arms.

### Initialization and recipe

- Both arms begin with bit-identical active query encoder, query classifier, and approved common zone-path parameters and buffers at the same seed. Serialize named-state hashes, original offline weight hashes, full load-key/shape/dtype audits, and new-head seed. State sharing across arms must not alias live tensors.
- Reuse authenticated offline builders, not downloads or random fallbacks. R uses its explicit ImageNet V1 authority; IA branches use the official LEVIR authority and strict key audit. Any reference-branch initialization and parameter sharing are separately declared. Never repurpose a failed D, legacy N, bounded diagnostic, or smoke checkpoint as matched initialization.
- Preserve the chosen recipe in both arms, including image loss, any approved dense loss, optimizer groups/LRs, schedule, batch sizes, accumulation, clipping, augmentation, epochs, and sample membership. Additional reference parameters may require a declared group, but not an unreported query LR or optimization change.
- Compare R recipe changes through a separate bridge, not a backbone-only contrast. If N needs recipient-corrected localization targets, establish a new matched control with exactly those targets in both arms first. Do not silently reuse or correct historical targets in only one arm.
- Both matched arms need the same trained missing-context policy before thresholded fallback. Until approved, missing context remains availability/referral evidence, not an automatic bona fide decision.

### Selection and reporting

Reuse `train.selection_key("accuracy", ...)` and update only on a strictly greater key. Accuracy wins first, AUC breaks ties, and an exact tie keeps the earlier epoch. Do not replace this helper with a new selector.

Use the existing metrics and `pilots.summarize_predictions` where its schema applies. Prediction taxonomy is `text_copy` and `text_inpaint`; human subgroup keys remain `copy` and `inpainting`. Each subgroup AUC uses that attack family plus the same split's genuine images. Report the small fold-0 copy population of 16, genuine false positives, misses, selected epoch, actual role/counts, context failures/coverage, and both overall and subgroup metrics.

Recompute selected-validation CSV metrics and verify selected history, checkpoint hash/epoch/config, identities, fresh-process replay, and invariance before accepting a row. Existing reporting is useful validation logic, not an existing matched-arm family selector. Incomplete, unknown, mixed-test, mixed-bank, or incompatible-input evidence cannot produce a winner. Missing CPU runtime is `not measured`; measurements from different physical hosts are `not comparable`.

## 7. Exact Correctness Gates After Implementation Approval

| Gate | Fixture or allocated evidence | Required result |
|---|---|---|
| G1, official membership and polarity | Validated metadata joins using `load_role_rows`, both matched arms, fixed fold. | Identical sorted train/validation IDs and labels; no duplicate IDs, extra wrong-design rows, altered-label conversion, or test pixel access. Every geometry failure retains its official image ID. |
| G2, authenticated support and transitive exclusion | Synthetic provenance chain linking a genuine to a held-out source/donor through an intermediate forgery; fixed membership with held-out/excluded-half pixel and genuine-layout-entry mutations. | Entire connected component excluded from support. Rebuilding into a new directory leaves every permitted bank product hash unchanged, including target, target coverage/valid, median, coverage/valid, zone, geometry, dependency manifest, and aligned specimen/valid when present. Archival whole-file hashes may change. An included-contributor mutation is a positive control that changes its dependency/hash evidence. No test pixels are decoded. |
| G3, annotation-free lookup and inference | Physically absent query annotation/target files and rows stripped of query component, source/donor, identity, true-label, and forged-field metadata. Supply only query pixels, enrolled design, and frozen bank authority to inference. | Evaluation resolves the same full bank and computes the same tensors/logits. Training-only component routing remains isolated. Unknown/empty/mixed bank authority fails without global fallback. Labels can remain in a separate metric table, not the inference request. |
| G4, coordinate and failure behavior | Synthetic identity, translation, rotation, scale, non-square query, padding, crop/resize composition, partial/empty overlap, failed feature matching, invalid transform, and median/specimen switch. | Known landmarks/polygons map to expected pixels under the recorded transform and mask interpolation. Reference-type switching preserves query/zone/shared geometry. Alignment failure produces explicit common unavailability, never an identity-warp success or omitted image. |
| G5, identity-preserving initialization | Same-seed fresh states, arbitrary valid references, strict load audit, wrong architecture/arm/data/bank IDs and original-weight hash corruption. | Exact equality of common initial named parameters/buffers and independent tensor storage. Direct correction starts at zero and base outputs match within FP32 tolerances. N satisfies its separately approved identity condition. Incompatible checkpoints fail before execution. No generic unsafe checkpoint fallback or requested-weight substitution. |
| G6, traversal, gradients, and buffers | Hooks on every query/reference encoder and shared projection; parameter-ID optimizer audit; train/eval snapshots. Mutate optional reference/target arguments with common bridge tensors fixed. | Query path called once. Reference-off has zero reference-dependent traversals, identical outputs/query gradients/common-state updates, and no reference-branch BN counter changes. Reference-on gradients reach the approved routes after the zero-correction startup. Each active trainable parameter occurs exactly once. No inference dependence on target masks or labels. Record every encoder/projection/fusion/decoder normalization and dropout policy and its expected counter increments. |
| G7, companion independence | Eight distinct source queries, paired references/zones, singleton and batch sizes 1, 2, 4, 8, each query at first/middle/last positions among different companions; both modes and unavailable-context cases. | Attack logits, and class logits when exported, agree at `atol=1e-5`, `rtol=1e-4`. All evaluation buffers remain bit-identical and dropout is disabled. Query/reference companions never mix through training-mode BN. Legacy D singleton evidence is insufficient for this gate. |
| G8, bounded overfit | One frozen training-only 16-ID manifest, eight genuine, six `text_inpaint`, two `text_copy`; same IDs in both arms, augmentation off, minibatch eight. | Diagnostic AdamW LR 1e-3, weight decay zero, at most 500 updates, evaluate every 50. At least 15/16 correct and mean image loss below 0.15 under the declared inference policy. Record active gradient and availability evidence. Failure blocks substantive training, not an expanded probe or cherry-picked IDs. |
| G9, actual-recipe smoke and replay | New processes with fresh original initialization; one epoch on the deterministic first 32 sorted train IDs and first 32 sorted validation IDs, actual approved matched recipe. | All intended inputs/traversals/losses work. Sample-correct loss, final partial batches/accumulation, BN counters, and selection semantics pass. A fresh process replays the saved smoke checkpoint on exactly the same IDs/order, logits within FP32 tolerances, identical hashes/config/arm/bank authorities, and G7 repeated. No smoke state initializes the substantive run. |
| G10, selected-checkpoint readiness | Independently reviewed actual installed-stack evidence from the same immutable source/config/data/weights as the requested run. Repeat G3/G6/G7 and fresh replay on each eventual selected checkpoint. | Gates bind the exact architecture/arm, source, original weights, support manifests/products, bridge contract, split hashes, phase, and checkpoint. Passing synthetic gates alone, a different source revision, or the original D-running gate cannot clear the new matched arms. |

G1-G7 require meaningful synthetic red/green regressions and later actual installed-stack confirmation where model execution is involved. G8-G10 are future allocated-compute work only after explicit authorization and independent review. Passing them establishes implementation correctness, not generalization, reference benefit, or authorization for B-D.

For N, preserve and extend existing one-query-encoder/one-query-projection checks. The approved reference-on design must state the additional expected traversals and BN counters instead of reusing query-only counts blindly. Check grids/resampling backward behavior on the actual installed stack if that branch uses them.

## 8. Existing Reuse and Its Limits

| Existing code | Reuse | Not already supplied |
|---|---|---|
| `data/query.py` | `load_role_rows`, raw RGB decoding/normalization preset, reference-free pilot contracts. | A support-safe query/zone bridge, reference tensors, annotation-free evaluation row type, or reference-isolated augmentation streams. D/N loader still points at legacy geometry. |
| `data/records.py`, `data/references.py` | `provenance_components`, `fold_of`, and audited layout/provenance primitives. | Legacy `reference_path` routes by query component and is not the fixed evaluation-bank resolver. Do not reuse global reference/zone construction for the new contract. |
| `data/enrollment.py` and integrity verifier | Genuine-train-only banks, cross-fitting, provenance manifests, independent specimen checks, read-only hash/JSON verification. | Context-to-query inference geometry, model feature-cache validity, detector competence, or a currently verified fold-1 integration contract. |
| `data/rectify.py`, `data/register.py` | Coordinate/letterbox and image-based similarity primitives, with explicit transform composition. | Legacy `register_design` may use inappropriate global support and identity fallback. Reuse audited primitives, not that historical aggregate behavior. |
| `models/anchor_rgb.py` | Strict staged ResNet50 ImageNet initialization, isolated head seed, two-class score polarity, active parameter groups. | Zone ingestion, reference feature extraction, a local comparison adapter, or a matched reference policy. |
| `models/anchor_changer.py` | Audited LEVIR loading, `encode_query`, direct GAP/MLP and N base route, initialization provenance and BN checks. | A reference-on branch. Current builders/forward accept query only. The D-running pilot is still legacy geometry. |
| `models/changerex.py` | Current-family decoder/referral modules and historical reference computation as read-only design evidence. | Approval that the historical reference route preserves the new base score or matches normalization/support. |
| `models/build.py`, `models/fusion.py`, `data/pairs.py`, `data/dataset.py` | Inspect historical mechanics and reuse explicitly audited small primitives if approved. | `build_model` may fall back to random pretrained initialization; legacy pair construction defaults to wrong-design relabeling, query-component reference selection, and mask access. Existing difference/attention blocks are not the approved new adapter. Do not enable them as a shortcut. |
| `pilots.py`, `train.py`, metrics and pilot tests | Selection, sample accounting, optimizer coverage, bounded row selection, provenance, checkpoint/replay patterns, tolerances, normalization/traversal audits. | Recognition of proposed matched identifiers or reference inputs. Extend deliberately after approval, rather than presenting the current query-only gate as reference validation. |
| `scripts/report_anchor_pilots.py`, `scripts/bench_anchor_pilots.py` | Selected-artifact consistency checks and CPU timing/manifest discipline. | Matched-arm reporting, enrollment-aware availability metrics, full reference preprocessing, or cached/uncached reference timing. |

## 9. Minimum Implementation Tasks After Approval

The paths below are proposed future ownership boundaries. No change to them is authorized now. Family-specific files are conditional, and the approved decision must resolve their concrete blocks before Task 3 starts.

### Task 1. Freeze the Decision and Executable Contracts

**Proposed files:** `src/refer_or_clear/matched_contract.py`, `src/refer_or_clear/pilot_config.py`, family-specific matched YAML under `configs/`, and `tests/test_matched_contract.py`. Preserve every existing pilot identifier/configuration.

**Produces:** Exact family/base architecture, arm IDs, bridge version/dimensions, reference type, normalization/parameter-sharing policy, complete recipe, availability/clearance semantics, and authenticated initialization/support authorities.

- [ ] Record approvals for every Section 3 decision and the branch condition, including any departure from the existing freshness rule.
- [ ] Write and run failing table-driven tests for incompatible family/arm/checkpoint IDs, geometry or bank versions, wrong-design rows, requested-weight failure, original `D_running` full-training prohibition, and unresolved configuration fields.
- [ ] Implement only resolved contracts. Make unresolved choices configuration errors, not implicit defaults. Run focused contract regressions and independent review.

### Task 2. Authenticated Common Bridge and Matched Data

**Proposed files:** `src/refer_or_clear/data/query_context.py`, `src/refer_or_clear/data/matched.py`, `tests/test_query_context.py`, `tests/test_matched_data.py`, and focused additions to enrollment tests. Reuse the enrollment builder and verifier rather than rebuilding support policy.

**Consumes:** Task 1's frozen dimensions/geometry and authenticated built banks. **Produces:** The Section 4 tensor boundary, one common transform/availability result per query, and train-only component routing isolated from evaluation requests.

- [ ] Write failing G1-G4 regressions, including the multihop source/donor exclusion, excluded-half mutations, included-source positive control, forbidden-file read spies, coordinate fixtures, all-failure population, path relocation, and reference-type switching.
- [ ] Implement `prepare_query_context` and dataset boundaries under the approved geometry. Keep metadata/targets outside model inputs and isolate reference-only random draws.
- [ ] Run synthetic coordinate, membership, provenance, and artifact-authentication tests. Obtain review before any actual support/query preprocessing job. Generalize fold authority separately if fold-1 confirmation is approved.

### Task 3. Only the Approved Same-family Model Branch

**Proposed files:** `src/refer_or_clear/models/matched_rgb.py` if R, or `src/refer_or_clear/models/matched_changer.py` if IA direct/N; corresponding `tests/test_matched_rgb.py` or `tests/test_matched_changer.py`. Existing pilot models remain distinct.

**Consumes:** The approved block design and common tensor contract. **Produces:** Base query evidence, reference-conditioned evidence, declared final scores, initialization provenance, and exact active parameter groups. Do not invent localization outputs for direct branches.

- [ ] Write failing G5-G7 tests with small modules, exact common-state hashes/storage independence, arbitrary-reference zero-correction identity, delayed hidden-gradient activation, reference-off output/gradient/buffer invariance, and companion positions.
- [ ] For R or IA direct, implement only approved adapter dimensions/operators and the mandated final-zero correction. For N, implement only its separately reviewed route and initial score-preservation policy. Do not select historical difference/attention/FDAF behavior by convenience.
- [ ] Verify optimizer coverage, every traversal/counter, missing-context semantics, strict checkpoint identities, and the existing family regressions. Extend actual installed-stack tests, leaving hardware execution to the authorized coordinator.

### Task 4. Matched Training, Selection, Checkpoints, and Gates

**Proposed files:** `src/refer_or_clear/matched_pilots.py`, `tests/test_matched_training.py`, `tests/test_matched_gates.py`, `tests/test_matched_replay.py`. Reuse audited pilot/train helpers without altering old identifiers or execution permissions.

**Consumes:** Tasks 1-3. **Produces:** Independently trained-arm ledgers, selected-validation predictions, identity-bound G8-G10 records, and fresh-process replay.

- [ ] Write failing regressions for accuracy/AUC/earlier ties, R polarity/CE equivalence if applicable, canonical attack taxonomy, partial batches/accumulation, unavailable-context accounting, smoke-state contamination, mixed-bank checkpoints, and replay order/config/hash mismatch.
- [ ] Implement bounded overfit and one-epoch smoke/replay orchestration only under the approved scope. Bind original initialization and all support/bridge authorities. Explicitly prevent diagnostic weights, optimizer state, scheduler state, or buffers from seeding substantive training.
- [ ] Run synthetic focused tests and review. Submit nothing as part of preparation or implementation testing on the local machine.

### Task 5. Packaging, Read-only Reporting, and Runtime

**Proposed files:** Extend `scripts/anchor_job_manifest.py` with tests; add approved matched diagnostic/train wrapper under `scripts/cluster/`; add `scripts/report_matched_reference.py`, `scripts/bench_matched_reference.py`, and their tests. Reuse existing wrapper headers, explicit phase/path/hash/review contract, and absolute environment patterns.

**Produces:** Frozen source/input manifests, minimal role-specific staging, no-submission wrapper checks, validation-only comparison, and a distinct reference-runtime contract.

- [ ] Write failing staging/resource/read-spy tests that exclude test/query annotations and wrong-design rows, require authenticated support products, prevent stale gates or output reuse, and preserve old track prohibitions.
- [ ] Model diagnostics/training require separately authorized GPU allocations. CPU-only preprocessing/integrity/timing wrappers request partition `cpu`, no GPU, bounded resources, `/tmp/$USER` scratch with `ROC_SCRATCH_ROOT` override and a unique job directory, `/opt/miniconda3/etc/profile.d/conda.sh`, and the verified absolute shared conda environment. Reuse the current GPU wrapper bootstrap unchanged for GPU tasks.
- [ ] Extend selected-validation report checks without creating a family selector. Preserve raw-R, legacy-diagnostic, bridge-control, and reference-arm labels. Missing runtime remains unmeasured.
- [ ] Prepare a separately identified same-host batch-one FP32 CPU protocol with fixed thread settings, 20 warmups, and exactly the first 200 lexicographic training IDs shared by the arms. Hash IDs/source mappings, original weights/checkpoints, support/bridge artifacts, input shapes, hardware/allocation/affinity/OS/library versions, and cached/uncached timing boundaries before timing. Include query decode, bank access/cache state, alignment, transforms, network output materialization, median/p95 per image, and availability/failure counts. No officer-device compliance claim from unspecified cluster hardware.
- [ ] Run synthetic timing/resource/report regressions and shell syntax checks only after their implementation is approved. Actual timing remains separately authorized allocated work.

### Task 6. Actual Evidence Review and Conditional Advancement

**Owner:** Coordinator and independent reviewers. **Artifacts:** Same-source actual G1-G10 evidence, support-isolation/integrity report, bridge validation results, matched selected-validation reports, and measured runtime or an explicit unmeasured limitation.

- [ ] Obtain separate approval for actual diagnostic allocations. Run the installed-stack gates with the official offline weights and exact source/config/support authority, not just small-module fixtures.
- [ ] If the competent candidate survives its support-safe bridge and all correctness evidence passes independent review, request the substantive fold-0 matched comparison. Do not launch from this document or from passing smoke alone.
- [ ] After reviewing both selected-validation arms, request confirmation on fold 0 seed 43 and preselected fold 1 seed 42. Repeat the nearest comparator if making a superiority claim. Reused official-fold images make these development checks, not independent final evidence.
- [ ] Separately lock family, geometry, zone/reference/enrollment policy, recipe, selection, scoring, and source-derived operating thresholds before an approved final Protocol A test or B-D work. Unresolved C/D support/evaluation membership is not supplied by this plan. No automatic sweep or downstream evaluation follows.

## 10. What Current Pilots Cannot Validate

- Raw R cannot establish zone ingestion, registration fidelity, reference evidence use, reference encoder normalization, or the final support-safe input recipe. It remains the published-benchmark RGB-only condition and must work without support products.
- D, N, and the D-running continuation still use declared legacy geometry. Successful learning or replay there cannot certify strictly train-only support or a context-to-query bridge.
- D's batch-one scores do not certify batched inference independence. D-running's stored-statistics evidence certifies only its query-only direct route, not a shared reference encoder or new adapter.
- N's absence of a blank encoder pass certifies the absent-feature candidate, not its hypothetical reference-on counterpart. It does not isolate the cause of every historical error.
- Current model APIs consume one query tensor. Their outputs, gradients, initialization checks, and checkpoint replay contain no implemented optional reference contract.
- Bank construction and read-only integrity tests authenticate support dependency/bytes. They do not establish query alignment accuracy, model competence, calibrated fallback, reference benefit, encoding robustness, or deployment latency.
- Synthetic model tests and a local CPU timing fixture are not actual GPU gates or measured end-to-end reference runtime. The existing R/D/N preliminary timing cannot estimate uncached enrollment/alignment overhead.
- One selected fold-0 validation result, especially 16 copy forgeries, cannot establish aggregate Protocol A targets, causal shortcut independence, held-out-design benefit, or cross-source transfer.

## 11. Preparation Handoff

This document is implementation-ready at the dependency, contract, test-gate, and task-boundary level. Actual network-block implementation is intentionally blocked on Section 3 approval, not silently specified by the task names.

The coordinator should request a focused extension design after provisional family evidence is reviewed, resolve every open block/input/normalization decision, assign file ownership, and approve the resulting executable contracts. Until then, only this preparation document is complete. No source or input mutation, model run, job submission, or commit accompanies it.
