# Independent N 37971 outcome review

## Bounded result

**PASS for final metadata outcome consistency of the authorized N legacy-geometry fold-0, seed-42, 100-epoch pilot. No metadata blocker was found.** This is not family selection, support-safe detector clearance, sweep approval, reference-extension approval, or official-test authorization.

The coordinator reports scheduler state `COMPLETED`, exit `0:0`, and elapsed `05:02:55`. Scheduler status was not independently queried. All seven requested local metadata files were present when audited, and their hashes remained unchanged across the checks. No transient transfer absence was classified as job failure.

The review used standard-library metadata calculations and frozen source/script reading only. It imported no project/model modules, read no image/checkpoint/weight bytes, decoded no pixels, ran no inference or GPU work, used no SSH, and submitted or committed nothing.

## Independently recomputed validation outcome

The prediction CSV contains exactly the 222 official fold-0 validation IDs, without duplicates or test IDs. Labels, designs, and attack families agree with the authoritative records/split join. There are 100 genuine and 122 altered documents. The fixed rule predicts genuine when `bonafide_score >= 0.5`, including ties. Bona fide scores equal `1 - attack_probability`; sigmoid probabilities agree with serialized logits within `1e-7`.

| Population | Genuine | Altered | Correct / total | Accuracy | AUC | Genuine false positives | Altered misses |
|---|---:|---:|---:|---:|---:|---:|---:|
| Overall | 100 | 122 | 211 / 222 | 0.9504504504504504 | 0.9797540983606557 | 0 | 11 |
| Copy-and-replace plus all validation genuines | 100 | 16 | 109 / 116 | 0.9396551724137931 | 0.88625 | 0 | 7 |
| Inpainting plus all validation genuines | 100 | 106 | 202 / 206 | 0.9805825242718447 | 0.993867924528302 | 0 | 4 |

AUC was independently computed from all genuine-versus-altered bona fide score comparisons, with half credit for exact score ties. The corresponding pair credits are `11953/12200`, `1418/1600`, and `10535/10600`. The last subgroup differs from the stored decimal by one floating-point rounding unit, not a substantive discrepancy.

Mean image BCE recomputed with numerically stable double-precision arithmetic from the saved logits is `0.1978770629349067`. The recorded FP32 evaluation result is `0.19787706323135812`, an absolute difference below `3e-10`. The selected epoch's logged validation loss also agrees. `localization_metrics` is correctly `null`; training map losses are not validation localization evidence.

All 100 logged epochs were scanned using the lexicographic selection key `(validation accuracy, validation AUC)`. The unique maximum occurs at **epoch 69**, with key `[0.9504504504504504, 0.9797540983606557]`. The frozen runner replaces the best checkpoint only on a strictly greater key, preserving the earlier epoch on exact ties. There is no exact winning tie in this actual log. The final epoch 100 has accuracy `0.9414414414414415` and AUC `0.9747540983606557`; it is not the selected outcome.

The eleven altered-miss IDs exactly match `val_summary.json` and the replay summary.

```text
sidtd_aze_passport_00_fake_3_105
sidtd_aze_passport_97_fake_3_115
sidtd_est_id_62_fake_9_113
sidtd_fin_id_11_fake_7_101
sidtd_fin_id_18_fake_7_110
sidtd_fin_id_30_fake_7_15
sidtd_fin_id_74_fake_7_40
sidtd_grc_passport_30_fake_4_108
sidtd_rus_internalpassport_49_fake_1_83
sidtd_srb_passport_87_fake_5_100
sidtd_svk_id_34_fake_8_103
```

The copy subgroup has only 16 altered examples, and detection at this fixed threshold is `9/16`. Its accuracy includes 100 correctly classified genuine examples. Do not interpret its subgroup accuracy or the overall result as uniformly strong attack detection. Zero observed genuine false positives is a validation observation at this threshold, not a deployment guarantee or a calibrated referral policy.

## Original scientific source and approval binding

The run binds to original source revision `0959335bd9a6a22510f6706c49c62d84560cd524`, not later enrollment or D-running changes. The local frozen snapshot is `/private/tmp/roc-anchor-source-0959335bd9a6a22510f6706c49c62d84560cd524/code`.

| Identity | Independently checked binding |
|---|---|
| Original source archive | Local `source.tar` SHA256 `4413528c1afd15f6c420763287bc57d9eb523df2a69138d5b472f0b4b2092ba4` |
| Complete source inventory | All 136 paths, sizes, and byte hashes independently recomputed from the frozen snapshot. The exact inventory equals `input_manifest.json` and hashes to `cb64905ab51a0c5fcc4600148765eb2e27db9cb5b860c4b2630593c617d37318`. |
| Runtime scientific source subset | Canonical source-path/hash map gives `2f8a833e5cb3450fdccd8c58dba8f07f58af330747a69cca7363d1a21528da9a`, matching manifest, training, pretrain gate, selected gate, and replay. |
| Frozen N YAML | SHA256 `458b664a051097037adec83c8cb61466e702eba02ee471260be01a5950da5e40` |
| Effective configuration | Canonical JSON SHA256 `2f852c2715c780d6901b0d3e5c0068bbeb4f51a1c0f0f09fa33f40bf567ee1b9`; exact configuration equals the input ledger and approved N diagnostic smoke configuration. |
| Records | Local `data/processed/records.csv` SHA256 `afd661819fe9be97c83d2ee5123a7ce60869437c61d1e9029c4b8a96175aa276` |
| Official Protocol A fold 0 split | Local `data/processed/splits/protocol_a_it0.csv` SHA256 `985d10b47461e0fa41d5502fbb51cd6fc03f8c7e8ec8cd1040b21be318efc1d9`; explicit Protocol A / `it0` membership |
| Prior bounded N approval | `results/diagnostics/anchor-pilots-v1/reviews/n-37964-review.json` SHA256 `fb45dd63d7d231c3ef66123ec258b6591bb91aaaac7a4e0830218a7f44f68848` |
| Completed N pretrain gate | `results/diagnostics/anchor-pilots-v1/anchor-pilot-diagnostic-37964/pilot/gate.json` SHA256 `32389c00f173c6c18b3f21d1e34a9759e7db82248a9959c91cfb004aab40378a` |
| Declared data inventory | Canonical metadata digest `a93245fe8e8ffe9b343abab5dd803c11d7ee311b88fc7661b5e4d9e56741e647`, identical to the approved diagnostic inventory. Actual image bytes were not rehashed. |

The complete source-inventory hash, runtime source-subset hash, YAML byte hash, and effective-config hash have different documented scopes. Their differences are expected, not identity failures. The run's full provenance equals the approved N review and diagnostic input ledger. All five runtime identities agree across the input ledger, training record, diagnostic gate, final selected gate, and replay.

## Execution-contract checks

| Check | Finding |
|---|---|
| Official membership | Exact sorted training and validation membership joins, including label, design, attack family, component, and raw path. Counts remain 1,778 train / 222 validation / 222 test. Only train and validation appear in run membership and predictions. |
| Authorized input inventory | Exactly 4,758 unique declared legacy files, comprising 2,000 train/validation canvases, 1,778 training validity masks, 970 altered-training target masks, and 10 design zones. The inventory equals an independently derived path allowlist. No reference, raw-image, validation-target-mask, or test-canvas entry is declared. |
| Mask-target authorization | Frozen `data/query.py` opens validity and forgery masks only for N training. Genuine training masks are generated zeros. Validation/replay classification receives query RGB plus legacy zone, not forgery targets. The runner calls `model(query)` before using targets in the training loss. The same inventory as the approved diagnostic binds the original legacy targets; no corrected-mask migration is declared. |
| Recipe | `anchor_absent_ia18_v1`, `legacy_geometry_diagnostic`, fold 0, seed 42, 100 epochs, train/eval batch eight, no wrong-design negatives, image loss plus map BCE/Dice at weight 3. AdamW base LRs `1e-4` and `1e-3`, weight decay `0.05`, warmup/cosine, clipping norm 5. Effective historical job-37877 settings are consistent with the retained recipe; N's running-BN and absent-feature changes are explicit. |
| Query preprocessing | ImageNet RGB normalization, 512 by 800 query plus zone, full canvas, affine eight pixels, photometry enabled, query jitter two pixels at probability 0.25, no JPEG augmentation. The zone is intentionally not query-jittered under this historical diagnostic contract. This is not a corrected geometry experiment or a historically stream-matched replay. |
| Fresh initialization | Recorded full-state, buffer, and pretrained-tensor hashes equal the completed pretrain gate's pristine hashes. `checkpoint_supplied=false`, source `original_offline_weights_and_seeded_head`, empty optimizer state, isolated head RNG. No diagnostic trained checkpoint is recorded as initialization. |
| Weight/load audit | Recorded LEVIR weight SHA256 `da3f569306dadd1fac5b64abc2a3d484571cd0ecf6410e1f152275cfe49a3618`; 173 unique loaded/active-pretrained keys, no unexpected keys, only the four new referral-head tensors missing. Historical and current recorded pretrained-tensor hashes agree. Weight/tensor bytes remain independently unverified. |
| Optimizer ownership | Recorded groups contain 92 pretrained and four new tensors, with 96 unique names and no name duplication. Coverage records agree between initialization and training. Active/total parameter counts are both 11,407,874. Live parameter ownership was not re-executed. |
| Complete training | Epoch IDs are exactly 1 through 100. Every log row records 1,776 consumed images and 222 optimizer steps, giving 22,200 steps total. Training membership remains 1,778; `drop_last=true` accounts for two recorded training-ID omissions per epoch. All omission IDs belong to official training. |
| Objective accounting | Every epoch's total loss agrees with `image_loss + 3 * (bce_map + dice_map)` within `6e-9`. All logged numeric values are finite. Scheduler records 22,200 iterations and 1,000 warmup iterations. |
| Training normalization | All 100 BN-update records contain 27 modules, each incrementing exactly 222 times. This is consistent with one query-driven training traversal per step, not an extra blank/reference traversal. |
| Evaluation normalization | Before training, on the selected checkpoint, and in replay, all 31 audited normalization/dropout modules are in evaluation mode. All 27 BN modules use stored statistics; the one InstanceNorm module has no running statistics. Dropout modules are inactive. |
| Recorded inference invariance | Each phase records 72 companion/position comparisons across batch sizes 1, 2, 4, and 8, tolerance `atol=1e-5`, `rtol=1e-4`, unchanged buffer hashes, one query stem call, one shared projection, and unchanged traversal state. |
| Selected gate and replay | Final gate passes with no failures. Selected checkpoint SHA256 `a8d62fcbb98c80535ff23c4e629261091b98db72105838335f0cef1cb446d9f4` agrees in train/gate/replay. Recorded replay uses a distinct child PID, matches all 222 predictions with maximum difference `0.0`, and repeats the selected inference gate. |
| Precision | Recorded FP32, AMP off, both TF32 modes off, deterministic algorithms with N's declared warn-only CUDA grid-sample-backward policy. Training bit reproducibility is explicitly false. |

The pristine state SHA256 is `4dc5c40189721778f4fd598b65c24b62cb6f7d2414a092edbc86a40733c9e837`, pristine buffer SHA256 is `c4bc9ea7a71b381e91b4d2262bf58215791611e8d003cd2573a2b4a5ac1e84b7`, and pretrained-tensor SHA256 is `1283f34343b41b2a5e27d55ec9654bf1e7dae47e1959adab4f403c4e0884ef44`. These were compared as metadata, not recomputed from tensors.

Recorded companion-invariance maximum absolute logit differences are `5.681067705154419e-08` before training and `9.5367431640625e-06` on the selected checkpoint and replay. Replay reports a separate prediction-replay difference of exactly `0.0`. These are different checks. Their flags, tolerances, state hashes, and cross-file bindings are consistent; GPU computations were not independently repeated.

The frozen N implementation encodes the query once, projects query features once, and creates `zeros_like(projected)` afterward for the absent stream. Its public forward input is a four-channel query tensor only. Dense targets enter the supervised loss, not the inference interface. This supports the declared candidate-package interpretation, not an isolated causal claim about normalization or the blank pass.

## Local metadata fingerprints

All paths below are relative to `results/runs/anchor-pilots-v1/anchor-pilot-train-37971/`.

| File | Independently recomputed SHA256 |
|---|---|
| `input_manifest.json` | `ab9a83e73327a6e01afe41a779add4caadb0816ec0b09c224936ea4f3ae8ff51` |
| `pilot/train.json` | `c764baf54e563617a39300e06f7daefd1c3ddb723ea8ef4926dd820459d9487d` |
| `pilot/log.csv` | `8249563505f66122eb79c9bbb201a8f2686c338323f4dcb92e961417c6155509` |
| `pilot/val_predictions.csv` | `c02bfdf0b4b57915227e75dc77adde4ce5c325bc47de595d80526b582178825e` |
| `pilot/val_summary.json` | `a27f9b215e65dd927ec736f6c4272c10c10bbd75e62930d028545996c12e729c` |
| `pilot/gate.json` | `b5442281ed58e74db5a77076676a78d0a1332d5df8fa60e64259835bade23e05` |
| `pilot/replay.json` | `32f2f711c00839318808db69290afc63849308400ca2ae94f44c1920ae980306` |

## Limitations and clearance boundary

- Checkpoint, pretrained weight, image, zone, validity, and target-mask bytes were not transferred or independently verified. Local source/archive/CSV/metadata hashing cannot replace those byte checks. No actual GPU tensor, gradient, BN update, sampler sequence, or replay was independently regenerated.
- The staged input inventory, frozen loader/runner, and `opened_roles` / `no_test_open` records are consistent with no official-test pixel or prediction access. There is no independent process-level file-open trace. Whole-split metadata inspection is not pixel access.
- The global legacy canvases/zones and original target masks are the authorized diagnostic inputs. This audit does not establish train-only upstream geometry provenance, recipient-corrected copy masks, pixel alignment, or localization quality. The completed enrollment/integrity jobs do not change N's declared input contract.
- This is one fold, one seed, and a source-validation-selected checkpoint. It does not establish cross-fold robustness, shortcut-independent generalization, deployment calibration, or superiority of a model family. No other candidate outcome or benchmark was reviewed here.

**Final finding.** The transferred outcome is numerically and internally consistent with the original bounded N approval, official membership, complete 100-epoch record, and recorded fresh replay/invariance gates. Accept it as a legacy-geometry diagnostic result with the byte/GPU limitations above. Family selection, sweeps, reference launches, and detector-consumption authorization remain outside this review.
