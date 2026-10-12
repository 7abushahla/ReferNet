# Validation-only anchor candidate comparison readiness

Date, October 12, 2026. Status, selected metadata available for all three candidates; comparison readiness remains incomplete. No family selection or additional training is authorized by this document.

## Scope and authority

This preparation follows the [classification-centered anchor specification](superpowers/specs/2026-10-11-classification-centered-anchor-design.md) and the [guarded D-running pilot plan](superpowers/plans/2026-10-12-d-running-pilot.md). The comparison concerns official fold 0, seed 42, 1,778 training images and 222 validation images, with 100 epochs and selected-checkpoint validation. Selection maximizes accuracy, then AUC, retaining the earlier epoch on an exact tie. Loss is not a selection criterion.

Completed R37967, N37971 and D_running_pilot37982 metadata and selected validation predictions were evaluated here. The coordinator reports N37971 and D37982 as `COMPLETED`, exit `0:0`, with elapsed times 5:02:55 and 3:18:13, respectively. Scheduler state was not queried locally. The coordinator has confirmed N's independent bounded metadata outcome PASS, with no metadata blockers, in the [N37971 outcome review](../results/diagnostics/anchor-pilots-v1/reviews/n-37971-outcome-review.md), and has read Galileo's [D_running_pilot37982 outcome review](../results/diagnostics/anchor-pilots-v1/reviews/d-running-pilot-37982-outcome-review.md), which reports bounded PASS after 624 assertions and no substantive discrepancy. Both clearances retain their byte/GPU, legacy-geometry and scientific-interpretation limitations. Neither selects a family or authorizes another experiment. No training curves, smoke results, diagnostic checkpoints or historical test results substitute for selected-validation outcomes.

## Candidate readiness

| Track | Job | Candidate contract | Current evidence |
| --- | --- | --- | --- |
| R | 37967 | Raw-RGB ResNet50 pilot | Completed selected-validation metadata independently checked; checkpoint bytes omitted from local transfer |
| N | 37971 | Legacy IA_ResNet18 decoder-dependent classification pilot | Independent bounded metadata outcome PASS confirmed; no metadata blockers; no local checkpoint bytes |
| D_running_pilot | 37982 | `anchor_direct_running_ia18_pilot_v1` | Independent bounded selected-validation PASS, 624 assertions and no substantive discrepancy; no local checkpoint bytes |

The new D-running pilot changes the execution contract, not the network architecture, relative to the training-only `D_running` diagnostic. The original `D_running` identifier remains diagnostic-only and prohibited from full training. Neither that diagnostic nor the original batch-statistics D condition can fill the new candidate row.

| Track | Selected epoch | Accuracy | AUC | Image cross-entropy | Active / total parameters | Common-host CPU runtime |
| --- | --- | --- | --- | --- | --- | --- |
| R37967 | 55 | 1.000000 | 1.000000 | 0.01514829081474795 | 23,512,130 / 23,512,130 | Not measured |
| N37971, metadata-reviewed | 69 | 0.950450 | 0.979754 | 0.19787706323135812 | 11,407,874 / 11,407,874 | Not measured |
| D_running_pilot37982, metadata-reviewed | 79 | 0.977477 | 0.996557 | 0.09837372227219977 | 11,261,825 / 11,261,825 | Not measured |

The R row is authoritative for the transferred, selected-validation evidence. It is not a complete local executable report-tool result. The current tool correctly marks this transfer incomplete because `pilot/checkpoints/best.pt` is absent.

The same checkpoint-byte limitation applies to N and D_running_pilot. All three remain incomplete under strict report-tool requirements. The completed metadata do not establish measurement eligibility, authorize model execution or select a family.

## Selected R evidence

Evidence root, [`anchor-pilot-train-37967`](../results/runs/anchor-pilots-v1/anchor-pilot-train-37967/). The relevant files are `input_manifest.json`, `inputs/`, and `pilot/{train.json,log.csv,val_summary.json,val_predictions.csv,gate.json,replay.json}`.

| Evaluation population | Genuine | Altered | Accuracy | AUC | Genuine false positives | Altered misses |
| --- | --- | --- | --- | --- | --- | --- |
| Overall validation | 100 | 122 | 1.000000 | 1.000000 | 0 / 100 | 0 / 122 |
| Copy subgroup, `text_copy` | 100 | 16 | 1.000000 | 1.000000 | 0 / 100 | 0 / 16 |
| Inpainting subgroup, `text_inpaint` | 100 | 106 | 1.000000 | 1.000000 | 0 / 100 | 0 / 106 |

Both subgroup evaluations reuse the same 100 genuine images. Their denominators must not be added as disjoint populations. The false-positive and miss ID lists are empty. No localization metric is applicable to R.

- All 100 epochs are present in order. There are 31 exact ties at the best accuracy/AUC pair. Epoch 55 is the earliest and agrees with the saved selection key, summary, gate and replay. A later lower loss does not replace it.
- The prediction CSV has 222 unique IDs. Joining those IDs to the hash-verified official split establishes validation membership; the prediction labels and attack families match the declared validation rows. Independent accuracy and pairwise AUC recomputation both yield 1.0. Independent float64 cross-entropy is 0.01514829009980999, within 1e-6 of the recorded value.
- Fresh-process replay reports a distinct process, 222 validation predictions and maximum logit difference 0.0. This review checked its metadata and identity bindings, not a new checkpoint replay.
- Before-training, after-training and replay invariance records cover batch sizes 1, 2, 4 and 8, with 72 comparisons each. Maximum companion differences are approximately 3.5763e-7 before training and 9.5367e-7 afterward and on replay, within the declared absolute 1e-5 and relative 1e-4 tolerances. Evaluation buffers remain unchanged.
- All 53 BatchNorm modules use running statistics at evaluation. Each epoch records the expected 56 training-forward counter increments, 28 optimizer steps and 1,778 training samples. These checks support training and evaluation consistency, not causal interpretation of perfect validation scores.
- Initialization and optimizer records bind fresh offline ImageNet initialization rather than diagnostic weights. The optimizer covers 161 tensors exactly once, including 159 pretrained tensors and two head tensors. Its initial state is empty, with SGD learning rate 0.01 and weight decay zero. Fresh initialization identities agree with the reviewed R37962 gate.
- Manifest inventories contain the 2,000 declared train/validation image paths. The R staging evidence excludes masks, zones, references and test paths. Gate and summary metadata declare `no_test_open` and train/validation-only opening. No image pixels were opened during this review.

### R provenance authorities

| Authority | Value | Local verification boundary |
| --- | --- | --- |
| Approved source revision | `0959335bd9a6a22510f6706c49c62d84560cd524` | Bound by copied approval and manifest metadata |
| Source archive SHA-256 | `4413528c1afd15f6c420763287bc57d9eb523df2a69138d5b472f0b4b2092ba4` | Recorded authority; archive bytes not present |
| Raw R configuration SHA-256 | `ed5c22565f764449e4375be3da2f0747986e1c952febe3d16725c93f2bbb4467` | Copied configuration bytes rehashed |
| Effective configuration SHA-256 | `2860b05436a0fcf890b61a5124d717be925be95a710be28beb74414fa9d8030b` | Checked across runtime evidence bindings |
| Runtime source SHA-256 | `2f8a833e5cb3450fdccd8c58dba8f07f58af330747a69cca7363d1a21528da9a` | Checked across gate, training and replay bindings |
| Records SHA-256 | `afd661819fe9be97c83d2ee5123a7ce60869437c61d1e9029c4b8a96175aa276` | Copied records bytes rehashed |
| Split SHA-256 | `985d10b47461e0fa41d5502fbb51cd6fc03f8c7e8ec8cd1040b21be318efc1d9` | Copied split bytes rehashed |
| Offline ImageNet weights SHA-256 | `0676ba61b6795bbe1773cffd859882e5e297624d384b6993f7c9e683e722fb8a` | Metadata binding checked; weights bytes not rehashed locally |
| Selected checkpoint SHA-256 | `6acd6886030a4ff5281e171798e572240fda338ae9c0ebbba0bedf44229a66b2` | Training, gate and replay agree; checkpoint bytes absent |
| Reviewed R37962 pretraining gate SHA-256 | `a1da74419bcb9a93ac4484ef88dfbe74e2a9d7880f10fceb69a0da2f652e23fa` | Copied gate hash and review binding checked |

Raw configuration, resolved configuration, source archive and runtime source identities have different hashing domains. They are not interchangeable. The copied source-file inventory also has its own identity. Missing checkpoint bytes are an intentional transfer limitation, not evidence that training failed. Conversely, matching recorded hashes alone cannot establish the contents of an absent checkpoint.

## Boundedly reviewed N and D-running evidence

Evidence roots, [`anchor-pilot-train-37971`](../results/runs/anchor-pilots-v1/anchor-pilot-train-37971/) and [`anchor-pilot-train-37982`](../results/runs/anchor-pilots-v1/anchor-pilot-train-37982/). Each transfer contains `input_manifest.json` and `pilot/{train.json,log.csv,val_summary.json,val_predictions.csv,gate.json,replay.json}`. Unlike R, their configuration/initialization input bytes are not included in these local transfers.

| Candidate and population | Genuine | Altered | Accuracy | AUC | Genuine false positives | Altered misses |
| --- | --- | --- | --- | --- | --- | --- |
| N overall | 100 | 122 | 0.950450 | 0.979754 | 0 / 100 | 11 / 122 |
| N copy, `text_copy` | 100 | 16 | 0.939655 | 0.886250 | 0 / 100 | 7 / 16 |
| N inpainting, `text_inpaint` | 100 | 106 | 0.980583 | 0.993868 | 0 / 100 | 4 / 106 |
| D_running_pilot overall | 100 | 122 | 0.977477 | 0.996557 | 2 / 100 | 3 / 122 |
| D_running_pilot copy, `text_copy` | 100 | 16 | 0.956897 | 0.979375 | 2 / 100 | 3 / 16 |
| D_running_pilot inpainting, `text_inpaint` | 100 | 106 | 0.990291 | 0.999151 | 2 / 100 | 0 / 106 |

Each subgroup reuses the common 100 genuine images. D's two genuine false positives consequently appear in both subgroup denominators; they are not four distinct errors. The frozen `metrics.deepid_acc` rule predicts genuine when `bonafide_score >= 0.5`, including ties, and altered when `bonafide_score < 0.5`. With `bonafide_score = 1 - attack_probability`, altered requires attack probability strictly greater than 0.5, corresponding to a strictly positive logit under the sigmoid mapping. The serialized bona fide score is the authoritative decision input. All three actual prediction files have no score/probability threshold ties or zero logits, so correcting the tie description changes no reported metrics or error IDs. No threshold was tuned here.

- Both logs contain epochs 1 through 100. N's unique best accuracy/AUC pair occurs at epoch 69; D_running_pilot's occurs at epoch 79. Recomputed selection agrees with training and summary metadata. Curves were used only to verify selection, not as outcome rows.
- Each prediction CSV has 222 unique IDs. An exact join to R's hash-verified official records/split bytes establishes identical validation IDs, labels, design IDs and canonical attack families across all three candidates. Each training ledger's validation membership also matches R's. Prediction files themselves have no role column; the official join and summary role `val` establish evaluation scope.
- Independently recomputed N image cross-entropy is 0.1978770629349067; D's is 0.09837371880136948. Both agree with saved losses within 1e-6. Aggregate/subgroup accuracy and pairwise AUC agree with the summaries within numerical precision. Error ID lists also agree exactly.
- Manifest, training, selected gate and replay identities agree per candidate. Both candidates bind the same hash-verified official records and split as R. Their selected gates and fresh-process replay metadata declare success, distinct replay processes, 222 predictions and maximum replay logit difference 0.0. These are checked records, not locally repeated checkpoint execution. Both bounded independent outcome reviews now report PASS.
- Replay companion-invariance metadata reports maximum differences approximately 9.5367e-6 for N and 1.5259e-5 for D, with unchanged buffers and declared absolute 1e-5 plus relative 1e-4 tolerances. The combined tolerance is not an absolute-only limit. The linked reviews cover metadata consistency and interpretation against frozen source, not independently regenerated GPU execution. D's review additionally reports conservative combined-tolerance checks without rerunning companion pairs.
- Both training ledgers declare train/validation-only opening and `no_test_open`; selected summaries declare role `val`. Positive integer active/total parameter counts were checked directly, not borrowed from an earlier D diagnostic. Localization metrics are null in both selected summaries; N's training localization objective does not supply a selected localization outcome.

### Recorded N/D provenance bindings

| Candidate | Authority | Recorded value |
| --- | --- | --- |
| N | Raw configuration SHA-256 | `458b664a051097037adec83c8cb61466e702eba02ee471260be01a5950da5e40` |
| N | Effective configuration SHA-256 | `2f852c2715c780d6901b0d3e5c0068bbeb4f51a1c0f0f09fa33f40bf567ee1b9` |
| N | Runtime source SHA-256 | `2f8a833e5cb3450fdccd8c58dba8f07f58af330747a69cca7363d1a21528da9a` |
| N | Selected checkpoint SHA-256 | `a8d62fcbb98c80535ff23c4e629261091b98db72105838335f0cef1cb446d9f4` |
| D_running_pilot | Raw configuration SHA-256 | `604d888d046d1903ef1741476e0a3170bde05ec9cc8abef9482b43fe71072f79` |
| D_running_pilot | Effective configuration SHA-256 | `5672a841eea3e683cdf76892d828c015711d500187e56fbe9c0a8aa86f30fdf9` |
| D_running_pilot | Runtime source SHA-256 | `af7073f3dd12299d9836f665cab5307dd5fed4a22882cc9fbbef317b8a38ae59` |
| D_running_pilot | Selected checkpoint SHA-256 | `dec4df1aa360c445185f17ee53c7e1275ea7a24d9fb721dc2740a65d86659844` |

Both declare the official LEVIR initialization-weight SHA-256 `da3f569306dadd1fac5b64abc2a3d484571cd0ecf6410e1f152275cfe49a3618`. N shares R's runtime source identity. D retains its different scientific source and exact pilot identifier; it is not relabeled as legacy D. This comparison update checked recorded bindings rather than rehashing configuration/initialization/checkpoint bytes. The linked independent reviews separately verified available frozen source/archive/configuration and approval authorities. Initialization-weight and selected-checkpoint bytes remain unverified locally; recorded hashes do not replace those byte checks.

### Error IDs verified against the official join

D_running_pilot genuine false positives are `sidtd_aze_passport_77` and `sidtd_srb_passport_36`. Its three altered misses, all copy, are `sidtd_aze_passport_97_fake_3_115`, `sidtd_lva_passport_51_fake_0_114` and `sidtd_srb_passport_87_fake_5_100`.

N has no genuine false positives. Its seven copy misses are `sidtd_aze_passport_00_fake_3_105`, `sidtd_aze_passport_97_fake_3_115`, `sidtd_est_id_62_fake_9_113`, `sidtd_fin_id_18_fake_7_110`, `sidtd_grc_passport_30_fake_4_108`, `sidtd_srb_passport_87_fake_5_100` and `sidtd_svk_id_34_fake_8_103`. Its four inpainting misses are `sidtd_fin_id_11_fake_7_101`, `sidtd_fin_id_30_fake_7_15`, `sidtd_fin_id_74_fake_7_40` and `sidtd_rus_internalpassport_49_fake_1_83`.

## Interpretation limits and compound recipes

These are comparisons between compound recipes, not isolated backbone ablations.

| Dimension | R | D_running_pilot | N |
| --- | --- | --- | --- |
| Input | Raw 299 × 299 RGB | Legacy registered 512 × 800 query plus zone | Legacy registered 512 × 800 query plus zone |
| Initialization | ImageNet ResNet50 | Official LEVIR IA_ResNet18 | Official LEVIR IA_ResNet18 |
| Classification route | Conventional image head | Direct encoder, global pooling and MLP | Decoder/map-dependent route |
| BN policy | Running statistics | Running statistics | Running statistics |
| Localization loss weight | 0 | 0 | 3, with training-only dense targets |
| Optimization | SGD and plateau scheduler | Declared legacy AdamW/cosine recipe | Declared legacy AdamW/cosine recipe |
| Augmentation | Declared raw-RGB crop, flips and photometric augmentation | Declared legacy geometry and augmentation | Declared legacy geometry and augmentation |

N's tested absent-reference policy is not evidence of a beneficial reference extension. D/N zone information and geometry differ from R's raw-RGB inputs. R's success does not validate a support-safe reference adapter. Any later reference comparison needs matched query pixels, zone, geometry and preprocessing, with only the reference condition changing. Legacy geometry needs explicit support-safe confirmation before serving as a final matched arm.

An independent metadata comparison confirms exact equality of N/D's 2,010 recorded canvas/zone inventory entries, including paths, sizes and declared SHA-256 hashes, comprising 2,000 canvases and 10 zones. N's complete inventory additionally contains training targets and is not identical to D's complete inventory. Together with the observed validation separation, the shared canvas/zone representation is not devoid of discriminative signal in these runs. This does not identify which artifact or feature carries that signal, prove compression causality, establish shortcut independence or validate upstream geometry provenance. No image or zone bytes were rehashed or decoded for this check.

The copy result rests on only 16 altered examples. Zero observed errors is not proof of zero population risk. Single-fold, single-seed selected validation is development evidence, particularly after repeated recipe investigation. It does not establish independent generalization, robustness or superiority. Encoding or acquisition shortcuts remain a possible explanation for separation. Document-identity/source/donor graph semantics are accepted under the verified records field contract; historical completeness of source provenance remains a limitation, not an unresolved code-field interpretation.

An independent join of the hash-verified official records and fold-0 split confirms 864 training components and 200 validation components, with 163 recorded component IDs shared between them. Consequently, 183 of the 222 validation images share a recorded provenance component with a training image, although train/validation image IDs are disjoint. This limitation applies to the shared official split for all three candidates and agrees with the linked D review. The authorized official benchmark is used as given; this finding neither changes the protocol nor authorizes a replacement split. These results must not be described as identity/source/donor-component-held-out generalization or component-independent evidence.

The specification's aggregate study targets of accuracy at least 0.97 and AUC at least 0.99 are not guaranteed per-fold acceptance criteria or a reason to select R automatically. Correctness gates permit an approved pilot; advancing a family also requires subgroup interpretation, genuine false positives, stability, matched-reference readiness and defensible CPU cost. Historical test evidence remains separate and is not consumed here.

## Exact missing evidence checklist

For N37971 and D_running_pilot37982, metadata availability and final scientific review clearance are separate gates. The following records distinguish completed local checks from remaining evidence.

- [x] Coordinator-reported completion and exit status, complete 100-epoch logs and recomputed accuracy-first/AUC-second selection. Earliest exact ties are respected. Partial curves and final epochs are not outcome substitutes.
- [x] Transferred selected metadata and predictions, exact official validation joins, common labels/designs/attack families and 100 genuine/16 copy/106 inpainting counts. Recomputed aggregate/subgroup accuracy, AUC and loss, with verified false-positive and miss IDs.
- [x] Positive integer active/total parameter counts from each completed training ledger.
- [x] Coordinator confirmation of N's bounded independent metadata outcome PASS, with no metadata blockers, linked above. Its byte/GPU, legacy-geometry and scientific-interpretation limitations remain applicable.
- [x] Galileo's independent D_running_pilot outcome review, read by the coordinator, reports bounded PASS after 624 assertions with no substantive discrepancy. This accepts completed selected-validation evidence only, not family selection or another launch.
- [x] Linked independent reviews check available immutable source/archive/configuration, official records/split and approval bindings through the original diagnostics, training, selection and recorded replay. Their unavailable checkpoint/weight/pixel-byte and independently regenerated GPU-execution boundaries remain explicit.
- [x] Bounded review of recorded fresh same-source diagnostic/actual-stack, initialization, accounting, invariance and replay evidence against frozen code. D's authority is its reviewed 37980 diagnostic, not original D_running or job 37979. No diagnostic checkpoint is accepted as a full-training initializer, and no GPU check was rerun for this comparison.
- [ ] Fresh full-training initialization, empty optimizer state, intended optimizer parameter coverage without duplicates, expected BN policy/counters, sample coverage and update counts. Check that N alone consumes authorized dense training targets and that selected inference requires no targets. Check staging/opened-role manifests for no test pixels.
- [ ] Selected checkpoint bytes and independently verified SHA-256, configuration, epoch and selection-key bindings for strict report-tool completion. Obtain an authorized recorded replay or separately authorized replay if evidence is insufficient; do not perform a local model run as part of this preparation.

Shared remaining evidence and decisions are also required.

- [ ] R selected checkpoint bytes if a complete local report-tool result is required. Its current metadata-only row must retain that qualification. Archive and initialization-weight byte verification require those artifacts if independently recomputed authority hashes are requested.
- [ ] Separately authorized common-host CPU measurements for all actual candidate identifiers. Pin one named host, thread settings, batch one, FP32, identical declared document IDs, 20 warm-ups and 200 documents, and preparation/network/total timing boundaries. Report median and p95 alongside active/total parameters. A legacy D timing row cannot represent D_running_pilot. Reference construction and officer-device end-to-end timing remain separate requirements.
- [ ] Explicit matched-reference/support readiness assessment and any approved geometry bridge. No family decision, seed/fold confirmation, reference experiment, sweep or final test evaluation is authorized by this report.

## Minimal report-tool compatibility proposal

Inspection covered [`report_anchor_pilots.py`](../scripts/report_anchor_pilots.py) and [`test_anchor_report.py`](../tests/test_anchor_report.py). No implementation change was made.

The current CLI accepts exactly three run directories in R/D/N order. `build_report` constructs those fixed identifiers and emits `anchor_report_v1`. Generic `inspect_run` already resolves a track contract, but that does not make the fixed report entry point accept the new candidate. Existing complete candidates must share runtime source, records, split and validation-membership identities. The new D pilot uses a newly reviewed source, so merely replacing the D directory or relaxing the source check is unsafe.

The smallest bounded extension is an explicit opt-in comparison contract, for example `--comparison-contract running-pilot-v1`, with the following requirements.

1. Keep the default legacy R/D/N order, schema, output and mixed-source rejection unchanged. The opt-in contract maps exactly R/D_running_pilot/N and emits a distinct contract/schema identity and unambiguous Markdown labels. Never alias the new candidate to D or unlock diagnostic-only D_running.
2. Supply explicit approved per-candidate source/configuration authorities for the opt-in comparison. Validate each run against its own immutable reviewed source, including its diagnostics and replay. Permit only the declared source difference required for the new contract, not arbitrary mixed-source evidence. Still require identical records, split, validation IDs, labels and strata across candidates. Record the recipe/source differences rather than asserting code equivalence.
3. Retain all generic selected-checkpoint, full-history, metric, gate, fresh-replay, initialization and parameter-count checks. Keep absent checkpoint bytes incomplete, missing outcomes pending/incomplete, `winner` null, family selection unauthorized and curves restricted to selection verification. Do not add a metadata-only completion bypass to accommodate this transfer.
4. Leave new-contract CPU runtime unmeasured until independently reviewed, separately authorized measurements exist. Additive `anchor_candidate_cpu_v1` setup is now implemented and under independent review; preparation is not measurement. The old report consumer remains unchanged. Reject legacy `anchor_cpu_v1` D rows as substitutes and do not silently reinterpret measurements or attach the additive schema to the legacy report.
5. Add synthetic regressions for unchanged legacy output, valid opt-in identifiers, rejection of unknown/diagnostic/mismatched identifiers, approved versus unapproved source differences, membership/label/stratum mismatches, earliest-tie selection, missing checkpoints and incompatible runtime attachment. Existing synthetic fixture class counts are not the actual R class distribution.

These are proposed compatibility changes only. They do not authorize implementation, benchmarking or additional model execution.

## Checks performed for this preparation

The earlier R-only verification used `OMP_NUM_THREADS=1 /private/tmp/roc-anchor-dev/bin/python -B`, checked copied configuration/records/split hashes and gate/replay/initialization bindings and BN/update records, and reproduced selection and metrics with the existing report helpers. `inspect_run(..., "R")` returned incomplete specifically for the absent `pilot/checkpoints/best.pt`, as expected. That qualification remains unchanged.

The current update used the same interpreter with standard-library-only CSV/JSON/hash/math verification. All three 100-epoch selections and official validation joins passed. Independent pairwise AUC, classification decisions, cross-entropy, subgroup counts and error lists agree with saved summaries. Per-candidate manifest/training/gate/replay identities and common official records/split hashes agree. All three local selected checkpoint paths remain absent. Follow-up checks verified the genuine-on-tie rule against both frozen scientific versions, absence of actual threshold ties, unchanged FP/miss lists, exact recorded canvas/zone inventory equality, and the official split's 163 shared components and 183 affected validation images. Both linked bounded outcome reviews report PASS and have been acknowledged by the coordinator. No strict report execution or checkpoint replay was performed.

No checkpoints were loaded, model inference was run, image pixels were decoded, test outcomes were consumed, production code was changed, or cluster operations were performed for this document update. Only this document was updated. Comparison readiness remains incomplete, all CPU runtime rows remain unmeasured, and no candidate family is selected.
