# Validation-only anchor candidate comparison readiness

Date, October 12, 2026. Status, incomplete candidate comparison. No family selection or additional training is authorized by this document.

## Scope and authority

This preparation follows the [classification-centered anchor specification](superpowers/specs/2026-10-11-classification-centered-anchor-design.md) and the [guarded D-running pilot plan](superpowers/plans/2026-10-12-d-running-pilot.md). The comparison concerns official fold 0, seed 42, 1,778 training images and 222 validation images, with 100 epochs and selected-checkpoint validation. Selection maximizes accuracy, then AUC, retaining the earlier epoch on an exact tie. Loss is not a selection criterion.

Only completed R37967 metadata and validation predictions were evaluated here. N37971 and D_running_pilot37982 are pending according to the coordinator. Their selected artifacts are not present locally. No training curves, smoke results, diagnostic checkpoints or historical test results substitute for their selected-validation outcomes. Scheduler completion for R37967 was reported by the coordinator, not queried in this review.

## Candidate readiness

| Track | Job | Candidate contract | Current evidence |
| --- | --- | --- | --- |
| R | 37967 | Raw-RGB ResNet50 pilot | Completed selected-validation metadata independently checked; checkpoint bytes omitted from local transfer |
| N | 37971 | Legacy IA_ResNet18 decoder-dependent classification pilot | Pending selected-validation artifacts; no outcome entered |
| D_running_pilot | 37982 | `anchor_direct_running_ia18_pilot_v1` | Pending selected-validation artifacts; no outcome entered |

The new D-running pilot changes the execution contract, not the network architecture, relative to the training-only `D_running` diagnostic. The original `D_running` identifier remains diagnostic-only and prohibited from full training. Neither that diagnostic nor the original batch-statistics D condition can fill the new candidate row.

| Track | Selected epoch | Accuracy | AUC | Image cross-entropy | Active / total parameters | Common-host CPU runtime |
| --- | --- | --- | --- | --- | --- | --- |
| R37967 | 55 | 1.000000 | 1.000000 | 0.01514829081474795 | 23,512,130 / 23,512,130 | Not measured |
| N37971 | Pending | Pending | Pending | Pending | Pending | Not measured |
| D_running_pilot37982 | Pending | Pending | Pending | Pending | Pending | Not measured |

The R row is authoritative for the transferred, selected-validation evidence. It is not a complete local executable report-tool result. The current tool correctly marks this transfer incomplete because `pilot/checkpoints/best.pt` is absent.

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

The copy result rests on only 16 altered examples. Zero observed errors is not proof of zero population risk. Single-fold, single-seed selected validation is development evidence, particularly after repeated recipe investigation. It does not establish independent generalization, robustness or superiority. Encoding or acquisition shortcuts remain a possible explanation for separation. Document-identity/source/donor graph semantics are accepted under the verified records field contract; historical completeness of source provenance remains a limitation, not an unresolved code-field interpretation.

The specification's aggregate study targets of accuracy at least 0.97 and AUC at least 0.99 are not guaranteed per-fold acceptance criteria or a reason to select R automatically. Correctness gates permit an approved pilot; advancing a family also requires subgroup interpretation, genuine false positives, stability, matched-reference readiness and defensible CPU cost. Historical test evidence remains separate and is not consumed here.

## Exact missing evidence checklist

For each of N37971 and D_running_pilot37982, collect and independently check all of the following before replacing its pending row.

- [ ] Completion and exit status from the coordinator, followed by the complete 100-epoch log. Recompute accuracy-first/AUC-second selection and earliest exact tie. Do not report partial curves or the final epoch as selected outcomes.
- [ ] Selected `train.json`, `val_summary.json`, `val_predictions.csv`, `gate.json` and `replay.json`. Verify exactly 222 unique validation IDs with the shared labels and attack families, 100 genuine, 16 copy and 106 inpainting. Recompute aggregate/subgroup accuracy, AUC and loss, and enumerate genuine false-positive and altered-miss IDs.
- [ ] Immutable source revision/archive, raw and effective configuration, records, split, offline initialization weights and selected checkpoint authorities. Bind them through staging, review, pretraining diagnostics, training, selection and replay. Check the new pilot's exact identifier rather than substituting an older D identifier.
- [ ] Independent review and passing fresh same-source actual-stack, overfit, smoke, invariance and replay evidence. For the new D pilot, old D_running or job 37979 evidence cannot authorize or initialize this run.
- [ ] Fresh full-training initialization, empty optimizer state, intended optimizer parameter coverage without duplicates, expected BN policy/counters, sample coverage and update counts. Check that N alone consumes authorized dense training targets and that selected inference requires no targets. Check staging/opened-role manifests for no test pixels.
- [ ] Selected checkpoint bytes and independently verified SHA-256, configuration, epoch and selection-key bindings for strict report-tool completion. Obtain an authorized recorded replay or separately authorized replay if evidence is insufficient; do not perform a local model run as part of this preparation.
- [ ] Active and total parameter counts as positive integers, excluding booleans, with declared counting conventions. Do not infer counts from another track or reuse diagnostic counts as selected-checkpoint evidence.

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
4. Leave new-contract CPU runtime unmeasured until an explicitly compatible identity-bound measurement contract exists. Reject legacy `anchor_cpu_v1` D rows as substitutes. The current runtime validator is also fixed to R/D/N; do not silently reinterpret its measurements or expand benchmark scope in this report change.
5. Add synthetic regressions for unchanged legacy output, valid opt-in identifiers, rejection of unknown/diagnostic/mismatched identifiers, approved versus unapproved source differences, membership/label/stratum mismatches, earliest-tie selection, missing checkpoints and incompatible runtime attachment. Existing synthetic fixture class counts are not the actual R class distribution.

These are proposed compatibility changes only. They do not authorize implementation, benchmarking or additional model execution.

## Checks performed for this preparation

Read-only local verification used `OMP_NUM_THREADS=1 /private/tmp/roc-anchor-dev/bin/python -B`. It independently recomputed R metrics and selection, checked membership and copied configuration/records/split hashes, and checked gate/replay/initialization bindings and BN/update records. All assertions passed. Current report helpers reproduced the selected epoch and metrics. `inspect_run(..., "R")` returned incomplete specifically for the absent `pilot/checkpoints/best.pt`, as expected. Neither pending job's selected artifact directory was present.

No checkpoints were loaded, model inference was run, image pixels were decoded, test evidence was consumed, production code was changed, or cluster operations were performed. Only this document was added. Comparison readiness remains incomplete, and no candidate family is selected.
