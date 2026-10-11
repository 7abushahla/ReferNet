# Anchor reference integration readiness, 2026-10-12

## Decision and scope

**Frozen fold-0 support has a completed byte-integrity pass. Detector consumption remains uncleared.** The remaining work is a versioned consumer bridge, its coordinate and availability contracts, integrated isolation tests, and a separately approved same-family reference extension. Hash agreement does not establish registration quality, zone correctness, learning competence, or reference benefit.

This document covers enrollment job `37973`, integrity job `37978`, current enrollment/dataset interfaces, and the classification-centered anchor specification. It does not select R, D, D_running, D_running_pilot, or N. Existing query pilots and their immutable legacy contracts remain unchanged. This preparation authorizes no pixel access, model import, allocation, training, detector evaluation, or official-test consumption.

## Frozen evidence and limits

The local evidence roots are concrete metadata bundles, not ready-to-load image datasets.

| Evidence | Existing local path | Binding or result |
|---|---|---|
| Build input ledger | `results/diagnostics/anchor-pilots-v1/anchor-enrollment-fold0-37973/input_manifest.json` | Frozen source parent `40e1528f6e6e7c6e5dc52614613cb86558d3ffda`; records and split bound below |
| Completed support authority | `results/diagnostics/anchor-pilots-v1/anchor-enrollment-fold0-37973/enrollment/manifest.json` | SHA256 `7691452ecb4d160e8767d277ba3ccf209355f16b3f49f85e52ff596c8a2719e8` |
| Integrity observations | `results/diagnostics/anchor-pilots-v1/anchor-enrollment-integrity-37978/report.json` | SHA256 `dd23746644c6235aeb806dfe68edf486d7052f13ef62a2f7272a79d5be311ab3`; `270/270` matched, zero errors |
| Verifier allocation and environment | `results/diagnostics/anchor-pilots-v1/anchor-enrollment-integrity-37978/launcher_manifest.json` | CPU partition, four CPUs, absolute `/shared/b00090279/conda/envs/refer-or-clear` environment |
| Completion | `results/diagnostics/anchor-pilots-v1/anchor-enrollment-integrity-37978/completion.json` | Return code `0`; same authority and checker hashes |
| Official records | `data/processed/records.csv` | SHA256 `afd661819fe9be97c83d2ee5123a7ce60869437c61d1e9029c4b8a96175aa276` |
| Official Protocol A fold 0 | `data/processed/splits/protocol_a_it0.csv` | SHA256 `985d10b47461e0fa41d5502fbb51cd6fc03f8c7e8ec8cd1040b21be318efc1d9` |

The previously reviewed integrity source revision is `982ebddb1bd32a824349da9f725f712a08e6d24d`, with source archive SHA256 `6fba2f336f667b9f9c9976c75b47cf84a38b9b54f842b347e14494d7ab821adc`. The current builder source matches its frozen implementation hash `4eb02e72170669174a29832768076768d6bbaac9b57361a0a9cbfe890f952e65`. The current checker matches the executed checker hash `fa6912ce4df26aa5b6c8bb12675b938e9070710178047738893cf8a1df3c511f`.

Fresh read-only standard-library checks for this document established the following.

- All `5,506` records remain in the complete inventory. The identity/source/donor graph partition agrees with the frozen component partition. All SIDTD classifier IDs remain in the official split, with explicit Protocol A and `it0` binding.
- Classifier membership remains `1,778` train, `222` validation, and `222` test. Exactly `590` eligible contributors are genuine official-training images whose transitive provenance components do not occur in validation or test.
- Ten designs have exactly three banks each. Authorized and registered membership agree in all 30 banks, with zero failed contributors. Full-enrollment uses total `590`; the two training halves together use `590`, giving `1,180` registered uses, not `1,180` unique images.
- The report exhaustively binds each unique `(design, bank, product)` to the completed manifest's exact declared path and SHA256. All 270 recorded actual hashes equal their expected hashes. There is no root relocation in job `37978`.
- All 60 local geometry/dependency JSON files were independently rehashed. Each bank dependency JSON equals its geometry JSON plus `output_sha256`, which binds all its declared products except the dependency manifest itself. That manifest's hash is bound by the completed global authority.
- The report explicitly records `scope=bank_product_integrity_only`, `detector_clearance=false`, `pixels_decoded=false`, and `raw_or_query_files_opened=false`.

**Independent verification limits remain explicit.** There are no PNGs in the local support bundle. The 210 PNG matches were checked as recorded cluster observations, not independently rehashed locally. Raw contributor-image hashes and parsed-layout hashes were not independently recomputed. Registered transforms and contributor consistency are metadata evidence, not validated pixel alignment. No model or image decoder was used for this document, and no integration tests were executed.

## Concrete bank products and coordinates

The declared cluster product root is

`/shared/b00090279/refer-or-clear/results/diagnostics/anchor-pilots-v1/anchor-enrollment-fold0-37973/enrollment`

Every bank lives below `fold0/<design_id>/<bank>/`. For example, the local metadata pair for one full-enrollment bank is

- `results/diagnostics/anchor-pilots-v1/anchor-enrollment-fold0-37973/enrollment/fold0/sidtd/alb_id/enrollment/geometry.json`
- `results/diagnostics/anchor-pilots-v1/anchor-enrollment-fold0-37973/enrollment/fold0/sidtd/alb_id/enrollment/manifest.json`

Its image declarations point to the corresponding cluster directory, not to local images. The authoritative lookup is `manifest["banks"][design_id][bank]["products"][product]`, whose entries contain `path` and `sha256`. Do not reconstruct legacy filenames or treat a metadata-only relocation as a complete product tree.

| Product key | File per bank | Meaning and consumer obligation |
|---|---|---|
| `target` | `target.png` | First-pass authorized-support median used as the fixed bank registration target. Query coordinate estimation must use this same target in both matched arms. |
| `target_coverage`, `target_valid` | `target_coverage.png`, `target_valid.png` | Initial target support coverage and validity. Neither describes query overlap. |
| `median` | `median.png` | Final validity-aware genuine reference RGB. This is the only reference type present in this build. |
| `coverage`, `valid` | `coverage.png`, `valid.png` | Final registered-support coverage and validity. Coverage is a capped uint8 count, not a calibrated quality probability. |
| `zone` | `zone.png` | Union of genuine personal-data rectangles transformed as polygons into bank coordinates, dilated by four bank pixels, and clipped to bank validity. |
| `geometry` | `geometry.json` | Bank identity, support membership, registrations, contributor provenance, and coordinate conventions. |
| `manifest` | `manifest.json` | Bank dependency hashes and the same geometry metadata. |

These are seven PNGs and two JSONs per bank, hence `210 PNG + 60 JSON = 270 products`. The optional `specimen` and `specimen_valid` keys are absent from all 30 banks. Legacy `single` reference selection is not an authorized product type here.

The frozen geometry is `bank_similarity_v1`, with bank canvas `[512, 800]` and axes `x=column,y=row`. `registrations[image_id]["native_to_bank"]` composes `canvas_to_bank @ native_to_canvas`. Those transforms map **support** coordinates into the bank. They do not supply a query-to-bank transform, its inverse, or query-valid overlap.

The specification prefers preserving query classification pixels and moving bank reference/zone into query coordinates. This is an unimplemented, unvalidated direction, not an accepted alignment result. A consumer must explicitly define transform direction, output dimensions, interpolation, overlap, and rejection criteria before use. RGB uses linear interpolation and validity/zone masks use nearest-neighbor interpolation under the bank contract. Query overlap must be derived from query validity and transformed bank validity, not inferred from a successful support build.

### Frozen contributor counts

| Design suffix under `sidtd/` | `train_half0` | `train_half1` | `enrollment` |
|---|---:|---:|---:|
| `alb_id` | 36 | 21 | 57 |
| `aze_passport` | 32 | 25 | 57 |
| `esp_id` | 33 | 25 | 58 |
| `est_id` | 29 | 29 | 58 |
| `fin_id` | 29 | 30 | 59 |
| `grc_passport` | 24 | 32 | 56 |
| `lva_passport` | 24 | 32 | 56 |
| `rus_internalpassport` | 28 | 30 | 58 |
| `srb_passport` | 31 | 37 | 68 |
| `svk_id` | 35 | 28 | 63 |

## Readiness matrix

“Established” below means evidence within the stated builder/metadata scope. It does not mean an integrated detector gate has passed.

| Gate | Existing evidence or interface | Remaining gap | Acceptance required before consumption |
|---|---|---|---|
| Authority and complete product binding | Completed manifest, integrity report, `scripts/verify_anchor_enrollment_products.py` | No consumer configuration pins this authority, bank schema, geometry version, and root together. Local PNG availability is absent. | A new explicit data contract verifies the pinned manifest and declared products, rejects stale/missing/mixed-root inputs, and records the actual root. No legacy-root fallback. |
| Training-only transitive support | `make_enrollment_manifest` in `src/refer_or_clear/data/enrollment.py`; frozen 590-member pool | Builder isolation does not prove every future consumer geometry/cache dependency selects the same authorized bank. | Trace target, zone, reference, validity, and cached transforms through the selected bank. No excluded component, unused half, global zone, or shared full-enrollment geometry may enter training context. |
| Component-cross-fitted training | `resolve_enrollment_bank(manifest, design_id, "train", training_component)` | Resolver is not wired into datasets/runners. | Every official training query, genuine or altered, uses the opposite `fold_of(component)` half for all context. Source/donor-linked query components are handled offline, not by identity-only exclusion. |
| Fixed evaluation bank and annotation-free inference | Resolver selects `enrollment` for `val`, `test`, or `evaluation` and rejects a supplied training component | Current dataset rows require labels/provenance metadata. No minimal inference request or bank-safe input adapter exists. | Inference works with query component, identity-based routing, source/donor, true label, attack family, query field annotations, and forgery masks removed. Only query pixels and declared enrolled design/fold/version determine coordinate estimation and bank selection. Metrics metadata stays outside inference. |
| Membership preservation and support separation | Official classifier counts unchanged; forbidden components excluded only from support | Future adapter could incorrectly filter classifier rows to 590 supports or add wrong-design altered labels. B-D support/evaluation policy remains separate. | Preserve all official IDs, roles, and labels. No support/evaluation component overlap in this fold. No primary wrong-design relabeling, support-driven sample dropping, or undisclosed benchmark shrinkage. |
| Query coordinate bridge and overlap | Support-to-bank matrices and interpolation metadata | No query registration, bank-to-query warp, coordinate-quality evidence, or query-overlap implementation | Approve direction and validity semantics, pass synthetic axis/composition/mask tests, then review a separately authorized training-only coordinate smoke. No claim of production alignment from hashes. |
| Reference-switch invariance | One fixed `target` and `zone` per bank; specification requires matched query-plus-zone control | Legacy pair augmentation is not an invariance contract. Actual specimen products are absent. | Reference on/off and any later reference-type switch preserve query pixels, geometry target/transform, zone, common overlap, alignment decision, and sample population. Reference-dependent validity is explicit and cannot silently alter the matched control. Isolate query RNG draws from reference draws. |
| Unavailability and failure population | Builder fails on empty banks and records failed registrations; resolver rejects unknown designs | No consumer availability/status interface, missing-context training/calibration, or referral handling | Registration failure makes both reference and transformed zone unavailable. Keep the query in both arms. Do not fabricate all-valid masks, substitute global products/checkpoints, or silently clear. Uncalibrated missing context requires referral. |
| Same-family input/model contract | `src/refer_or_clear/anchor_contract.py`, `docs/anchor-pilot-interfaces.md` describe current query-only tracks | No approved reference-enabled contract, adapter dimensions, normalization route, or runner wiring | Focused extension review after competent source-validation evidence. Same query path plus zone control; baseline-preserving final correction and reference-off output/gradient/buffer invariance must be tested separately. No family choice here. |
| Reproducible bounded integration | Frozen builder and checker identities; completed CPU ledgers | No integrated train-only read audit or adapter replay record | A separately approved smoke records source/config/authority hashes, training IDs, product-read inventory, availability counts, and replay tolerances. It does not authorize 100-epoch detector runs or official-test pixels. |

An independently sourced blank remains an explicit future exception, not a missing mandatory product. If added, it needs its own immutable `external_blank_v1` source/hash declaration and complete-dataset identity/source/donor/component/path/hardlink exclusion, including authoritative content hashes where available. It must align only to the selected bank target. The existing optional-specimen tests establish builder rejection behavior, not authorization of any actual specimen or a specimen comparison.

## Existing interfaces that must not be repointed

`src/refer_or_clear/data/enrollment.py` exposes three prepared APIs, `make_enrollment_manifest`, `build_enrollment_products`, and `resolve_enrollment_bank`. A source search finds no production consumer call to the resolver. The following contracts are not interchangeable with the frozen bank layout.

| Existing path/interface | Actual behavior | Integration consequence |
|---|---|---|
| `src/refer_or_clear/data/pairs.py`, `_RefPicker` / `build_pairs` | Uses `references/<design>_<group>_fold<opposite>.png`, global `zones/<design>.png`, and query component even for evaluation. Specimen filenames omit the official group. A `single` pool comes from all genuine records. Labels select mask paths; default training adds wrong-design negatives. | Not a safe bridge. Merely replacing `processed_root` cannot implement fixed evaluation enrollment, complete authority, or annotation-free inference. |
| `src/refer_or_clear/data/dataset.py`, `PairDataset` | Reads query/reference RGB and masks into a legacy eight-channel input. Optional validity can become all ones. Reference augmentation consumes RNG draws; query-only jitter does not move the zone with the query. | Do not assume query/reference switching preserves shared geometry or query streams. Define and test the new contract instead of silently reusing this item format. |
| `src/refer_or_clear/data/query.py`, `load_role_rows` / `AnchorQueryDataset` | Current pilots accept train/val metadata. Non-R queries use legacy canvas/global zone. N reads dense targets only during training. No enrollment authority, bank root, reference mode, overlap, or availability argument exists. | Preserve current D/N operation. Their target-free classification path is useful prior evidence but does not implement a bank-safe minimal inference API. |
| `src/refer_or_clear/anchor_contract.py`, `pilot_config.py`, `pilots.py` | R is raw RGB. All current non-R tracks declare `legacy_geometry_diagnostic`; the pilot runner constructs `AnchorQueryDataset`. | Add a separately approved, versioned contract later. Do not change existing track meanings or reinterpret checkpoints. |
| `src/refer_or_clear/evaluate.py`, `role_pairs`, and historical `train.py` | Historical pair route uses `build_pairs` and median/specimen/single reference naming. | These entry points do not become cleared by the CPU integrity report. Any future wiring needs independent integration review. |

The focused design must separate an inference request from its training/audit envelope. The inference side needs the query image or path, declared enrolled design, and frozen support-contract identity. The training-only envelope may carry the offline component needed for cross-fitting, labels, and optional dense supervision. Names, tensor shapes, overlap representation, cache keys, and availability/status fields are still interface decisions, not implemented APIs. Cache identity must include authority, fold/design/bank, geometry version, query preprocessing identity, and transform settings so a full-enrollment or legacy transform cannot be reused for a training half.

## Minimal planned training-only integration tests

These tests are **planned, not implemented or run**. A proposed isolated home is `tests/test_anchor_reference_integration.py`, alongside a separately owned future consumer adapter. Do not add them to or modify current production contracts as part of this document. Initial tests use synthetic arrays and metadata, not actual support/query pixels. Evaluation routing can be simulated without opening any validation/test image. A later actual-image smoke requires separate approval and predeclared official-training IDs only.

| Planned test | Fixture and concrete assertions |
|---|---|
| `test_pinned_authority_and_product_root` | Synthetic complete manifest/product tree. Require correct authority SHA, schema/fold/geometry, exact bank product keys, matching dependency hashes, and explicit root mapping. Reject missing files, mixed banks, bad SHA, traversal, symlinks, and metadata-only trees presented as image-complete. Preserve all inputs. |
| `test_train_component_crossfit_all_dependencies` | Genuine and altered training requests with transitive source/donor links. Assert opposite-half target/median/zone/valid/geometry paths, no query component in any contributor set, and no global/full-enrollment cache reads. Check all ten frozen design memberships using metadata only. |
| `test_inference_without_query_annotations` | Minimal synthetic inference request with design and frozen contract only. Remove label/component/source/donor/attack-family/field-layout/mask fields and deny annotation opens. Fixed `enrollment` selection and prepared inference inputs remain unchanged. Training targets stay in a separate envelope. |
| `test_bank_to_query_coordinates_and_overlap` | Synthetic landmarks, polygons, and binary validity planes with a known nonidentity transform. Assert direction, inverse/composition, axes, output shape, nearest-neighbor masks, and query-valid intersection. Preserve query pixels under the proposed query-coordinate direction. Reject singular/nonfinite transforms and empty overlap. |
| `test_reference_switch_preserves_query_zone_and_rng` | Fixed seed and synthetic median plus explicitly synthetic specimen fixture. Assert identical query tensor, shared transform/target/zone/common overlap, availability decision, and next-sample query RNG under reference on/off and type switches. Do not authorize absent specimen products from this fixture. |
| `test_missing_context_retains_population_and_refers` | Unknown bank, failed alignment, missing validity, empty overlap, and missing optional specimen. Assert explicit failure/unavailability, both reference and transformed-zone unavailable on registration failure, preserved sample IDs, and no global fallback, fabricated validity, or silent clearing. Distinguish malformed required products from ordinary per-query registration failure. |
| `test_classifier_membership_is_not_enrollment_membership` | Frozen CSV metadata and synthetic input reader. Preserve 1,778 training IDs with original labels, while support eligibility remains 590. No duplicates, wrong-design negatives, or loss of genuine/altered queries. Validation/test IDs remain metadata-only and unopened. |
| `test_selected_bank_read_allowlist_and_isolation` | Instrument file opens and cache lookups. Permit only the synthetic query and selected declared products. Deny raw contributor/layout/mask access during inference, unused halves, legacy roots, and all val/test image paths. Mutating unused synthetic banks cannot change prepared inputs; corrupting a selected bank fails closed. |

Retain the existing builder tests in `tests/test_enrollment.py`, especially `test_source_donor_transitive_exclusion_preserves_classifier`, `test_heldout_pixels_and_shared_via_entries_cannot_affect_products`, `test_unused_half_cannot_affect_target_median_zone_valid_or_specimen`, and `test_opposite_half_and_annotation_free_evaluation`. They prove upstream synthetic isolation and resolver behavior, not the proposed consumer bridge. The integrity tests in `tests/test_anchor_enrollment_integrity.py` and `tests/test_anchor_enrollment_integrity_cluster.py` prove verification/report protection, not detector semantics.

The model-free tests above are insufficient to approve a trained reference arm. Later same-family tests must separately cover initial baseline-score preservation, learning-capable correction gradients, optimizer ownership, and reference-off logits/gradients/mutable-buffer invariance. A separately trained query-plus-zone control is required; turning reference off in a reference-trained model is not that control. No model imports or adapter architecture choices are part of this preparation.

## Ordered handoff and clearance boundary

1. Preserve the completed authority and both ledgers. Resolve full product availability on the intended host with explicit paths and hashes. The local metadata bundle cannot satisfy image loading. Do not rebuild or modify `37973` merely to make legacy paths fit.
2. Obtain focused approval for the versioned consumer interface, query-coordinate direction, shared-zone/overlap invariants, cache identity, and missing-context/referral semantics. Exact adapter dimensions and family selection remain dependent on separately reviewed source-validation evidence.
3. Implement the isolated bridge under separate ownership. Pass the synthetic tests and existing builder/integrity regressions without changing historical pilot contracts. Independently review the complete read/dependency trace before actual images are permitted.
4. Request a bounded official-training-only integration smoke with frozen IDs and no validation/test decoding. Record declared product reads, coordinate checks, unavailable cases, query/reference-switch invariance, and unchanged classifier membership. This demonstrates software integration only, not scientifically validated alignment or detector performance.
5. Request separate detector-consumption and matched-family approvals. Establish a competent same-family member, approved input/zone control, training/calibrated missing-context behavior or explicit referral, and source-validation comparison protocol before a substantive fold-0 reference experiment.

No gate here clears detector 100-epoch runs, benchmark/runtime work owned elsewhere, official-test access, or Protocol B/C/D support policy. Those protocols need separate support/evaluation identity contracts and frozen loaders. In particular, FantasyID inventory coverage for specimen exclusion is not a completed Protocol C reference bank or evaluation authorization.

## Governing contracts

- `docs/superpowers/specs/2026-10-11-classification-centered-anchor-design.md`, matched reference contribution and Sections 6-9.
- `docs/superpowers/plans/2026-10-11-classification-centered-anchor.md`, Task 5 bank APIs, isolation, and future-coordinate gate.
- `docs/anchor-pilot-interfaces.md`, current query-only dataset and model boundaries.
- `src/refer_or_clear/data/enrollment.py`, actual authority, resolver, optional-specimen rejection, and bank-product implementation.
- `scripts/verify_anchor_enrollment_products.py` and `scripts/cluster/verify_anchor_enrollment.sh`, byte-integrity-only verification scope and CPU environment contract.

**Bounded finding.** Integrity and authorized support membership are established within the stated evidence limits. The consumer bridge, query-coordinate behavior, availability interface, and matched reference extension remain unimplemented or unapproved. No detector-consumption clearance is issued.
