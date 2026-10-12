# Independent final outcome review of D_running_pilot 37982

Reviewed on October 12, 2026 by the independent Codex assistant in this session.

## Verdict and boundary

**PASS for completed, selected-checkpoint Protocol A validation evidence only.** The main independent audit passed 624 assertions, including all 165 frozen source files and four accounting/schedule/curve checks for each of the 100 epochs. Supplemental conservative inference-tolerance checks also passed. No substantive execution, authority, selection, or metric discrepancy was found.

This accepts the outcome of the already authorized single `D_running_pilot`, fold zero, seed 42, 100-epoch job. It is not authorization for another job, family-based selection, a sweep, another fold/seed, reference training, official test evaluation, or downstream detector evaluation. Original `D_running` remains diagnostic-only. Legacy geometry remains a limitation, not a final support-safe claim.

The coordinator reports job 37982 `COMPLETED`, exit `0:0`, elapsed `3:18:13`. The local manifest identifies GPU node `gpu-dy-g5-0-18`. Scheduler status was not independently queried because SSH was prohibited.

## Exact authorities and approval chain

| Authority | Exact value |
| --- | --- |
| Frozen source revision | `bef7544a64f36a166f76eeef0d3130e353bd3203` |
| Source archive SHA256 | `00d0b99e635536d27a01c6f4f39bd94e14e98409dc5616c722f7dd4db3f8d5de` |
| Full source inventory SHA256 | `1dc29e362523ab3bcfdfaa794764dd85f2c8c1043fe3acae3381b89bec1bc69a` |
| Runtime source SHA256 | `af7073f3dd12299d9836f665cab5307dd5fed4a22882cc9fbbef317b8a38ae59` |
| Configuration bytes SHA256 | `604d888d046d1903ef1741476e0a3170bde05ec9cc8abef9482b43fe71072f79` |
| Effective configuration SHA256 | `5672a841eea3e683cdf76892d828c015711d500187e56fbe9c0a8aa86f30fdf9` |
| Records SHA256 | `afd661819fe9be97c83d2ee5123a7ce60869437c61d1e9029c4b8a96175aa276` |
| Official fold-0 split SHA256 | `985d10b47461e0fa41d5502fbb51cd6fc03f8c7e8ec8cd1040b21be318efc1d9` |
| Official LEVIR weights SHA256 | `da3f569306dadd1fac5b64abc2a3d484571cd0ecf6410e1f152275cfe49a3618` |
| Data inventory SHA256 | `4276e25e9310ef5b76a9358f28a33fd261ea3c2eb2bbfc5f98cba82860b23680` |
| Approved 37980 diagnostic gate SHA256 | `90644094ef97f7867d842c52a3a2be7243001d1628a7b8b43eb6db8f5a06cc54` |
| Approved 37980 diagnostic manifest SHA256 | `78d0a6242865639c96402e0b1686c1cc7a1c84078e8079b6256812fdf4499d03` |
| Independent launch-approval JSON SHA256 | `baff93fcc127a5c1811ad93489b283ca6b51035e771a6b385203b9c7c3e1ff22` |
| Selected checkpoint SHA256, recorded only | `dec4df1aa360c445185f17ee53c7e1275ea7a24d9fb721dc2740a65d86659844` |

The source archive revision header and digest were independently verified. All 165 training-manifest source entries match both archive contents and extracted frozen files. Full-source and runtime-source hashes were recomputed. Raw YAML and resolved configuration hashes match the approved authorities. The source files, data entries, effective configuration, complete provenance, and runtime identities are identical to diagnostic 37980.

The available local launch-approval JSON was independently hashed and matches the training manifest's `review_sha256`. It explicitly approves one fold-0, seed-42, 100-epoch validation pilot, and binds this exact source/archive, provenance, diagnostic manifest, and gate. The local diagnostic manifest/gate were rehashed against that record. Diagnostic 37980's actual-stack XML contains one pass, no skip, no failure, and no error. Training, selected gate, and replay all bind the same runtime identities and selected checkpoint digest.

## Official metadata join and exclusions

Hash-verified local records/split copies from diagnostic 37979 were used for an independent official join. The records table contains 5,506 unique metadata rows, including 2,222 SIDTD and 3,284 FantasyID rows. The Protocol A split covers exactly the 2,222 SIDTD rows, with unique IDs and matching labels, designs, and component IDs. Metadata for other datasets does not imply access to their pixels.

Training membership is exactly 1,778 official training IDs. Validation membership and prediction order are exactly the 222 official validation IDs. The 222 official test IDs are excluded. There are no duplicate validation predictions or train/validation image-ID overlaps. All train/validation manifest row fields agree with the hash-verified records.

The unchanged staged inventory contains exactly 2,000 train/validation canvases and ten zones, all of kind `legacy`. It has no test canvases, reference files, raw images, masks, or valid-target planes. The frozen direct-track loader reads only canvas/zone inputs and does not enter N's dense-target path. Runtime records opened roles `train` and `val`, evaluation role `val`, and `no_test_open=true`.

**Important inherited split limitation.** The official split is not component-held-out. Training has 864 component IDs and validation has 200, with 163 shared IDs. Consequently, 183 of the 222 validation images share a recorded provenance component with some training image. This is not a deviation introduced by 37982. The frozen Protocol A implementation explicitly uses the supplied official folds as given. The result must not be described as identity/source/donor-component-held-out generalization or analyzed as component-independent evidence.

## Fresh initialization and actual recipe

After removing the training-specific pretrain-gate digest field, the training pristine initialization dictionary exactly equals the approved parent gate and both diagnostic phases. Training records no checkpoint initialization and an empty optimizer. Thus the recorded chain supports fresh official initialization, not continuation of diagnostic weights or BN buffers.

| Pristine fingerprint | SHA256 |
| --- | --- |
| Complete state | `32e2fcde137e5dac4c537dc5e18ba3ff02cf8d12a72311e0177aa28c076cfec4` |
| Buffers | `bc4e5ed3fe4e999395997ded7981a9bd9a0acd198e21c5b342f348713aa64dad` |
| Active pretrained tensors | `9ab72d0bbe12d81de9034b1b44e5d037576a8e83f45addf0ba9d339382e59c42` |

Initialization provenance equals diagnostic 37980's provenance except for the allocation-specific staged weight path. It declares the official hash-bound pickle policy, head seed 42, 173 loaded historical keys, 132 active backbone keys, no unexpected keys, and all 66 retained running-BN buffer keys. There are 22 running BN modules and 11,261,825 active parameters, with complete optimizer coverage of 70 tensors. The historical pretrained fingerprint remains `1283f34343b41b2a5e27d55ec9654bf1e7dae47e1959adab4f403c4e0884ef44`.

The actual phase recipe has batch eight, accumulation one, drop-last enabled, 100 epochs, the declared query augmentations, image-only loss, and gradient clipping at five. Configuration retains AdamW base rates 0.0001/0.001, weight decay 0.05, strict FP32 without AMP/TF32, fold zero, and seed 42. Runtime warns against claiming bit-reproducible training.

The optimizer metadata rates of approximately 1e-7/1e-6 are not a recipe mismatch. The source creates `LambdaLR` before serializing that metadata, applying the initial 1/1000 warmup multiplier to the configured base rates. The independently reproduced epoch-end rates match the declared 1,000-iteration warmup and cosine schedule at every logged epoch.

## Complete training and BN accounting

- Log epochs are exactly 1 through 100, without gaps or early stopping.
- Each epoch processes 1,776 images in 222 complete batches and executes 222 optimizer steps. All 22 BN counters increment by exactly 222 each epoch.
- Totals are 22,200 training forwards/optimizer steps, 22,200 increments per BN counter over the logged training trajectory, and 177,600 training-image exposures.
- Exactly two distinct official training IDs are recorded as omitted each epoch. The 200 omission events cover 193 unique IDs, with maximum frequency two. The 1,778-image membership count and 1,776-image per-epoch processing count are therefore consistent.
- Scheduler total iterations are 22,200. All 100 epoch-end learning-rate pairs independently reproduce the frozen warmup/cosine formula. Both rates end at zero.
- All logged training/validation scalar metrics are finite. Training image loss equals total loss and dense-loss columns remain empty.

This checks recorded accounting and named omissions. It does not independently reconstruct the entire sampler stream, every training exposure, intermediate tensor state, or optimizer update.

## Independent metric and checkpoint-selection reconstruction

Validation predictions were joined to official labels/designs/families and checked for finite logits, valid probabilities, and exact complementary bona fide scores. Sigmoid probabilities agree with logits within FP32 rounding. Accuracy uses the frozen threshold convention, and AUC was independently computed by pairwise genuine-versus-altered ranking with half-credit ties.

| Selected epoch-79 population | Genuine | Altered | Correct | Accuracy | Recomputed AUC |
| --- | --- | --- | --- | --- | --- |
| All validation | 100 | 122 | 217/222 | 0.9774774774774775 | 0.9965573770491803 |
| Copy, with the same genuine set | 100 | 16 | 111/116 | 0.9568965517241379 | 0.979375 |
| Inpainting, with the same genuine set | 100 | 106 | 204/206 | 0.9902912621359223 | 0.9991509433962265 |

Recorded overall AUC is 0.9965573770491802, differing from pairwise arithmetic only at floating-point rounding. Overall stable float64 BCE from logits is 0.09837371880136948. Recorded FP32 BCE is 0.09837372227219977, a difference of approximately 3.47e-9. All subgroup metrics, counts, and error lists match.

Across all 100 logged epochs, the maximum validation accuracy occurs at epochs 77 and 79. Epoch 77 has AUC 0.9964754098360656, while epoch 79 has 0.9965573770491802. Therefore epoch 79 is the unique lexicographic accuracy/AUC maximum. Independently choosing the earliest occurrence of the maximum key gives epoch 79; there is no winning exact tie to resolve in this run. The strict-greater checkpoint update in frozen source retains the earlier epoch for any exact tie. Selection metadata, selected summary, predictions, and epoch-79 curve agree.

The epoch-100 curve is distinct from the selected outcome. It has accuracy 0.963963963963964, AUC 0.9947540983606556, and validation BCE 0.1293172345187765. Training completed 100 epochs; selected validation evidence is from epoch 79, not epoch 100.

The two genuine false positives are `sidtd_aze_passport_77` and `sidtd_srb_passport_36`. The three altered misses are `sidtd_aze_passport_97_fake_3_115`, `sidtd_lva_passport_51_fake_0_114`, and `sidtd_srb_passport_87_fake_5_100`. All three misses are `text_copy`. Inpainting has zero altered misses; its two subgroup errors are the shared genuine false positives.

## Inference stability and fresh selected replay

Before training, after restoring the selected checkpoint, and during fresh replay, each audit records 72 companion comparisons at batch sizes 1/2/4/8 using the first eight validation queries. All 22 BN modules are in evaluation mode with running statistics. Buffer hashes are unchanged within every audit.

| Audit | Maximum absolute companion-logit difference |
| --- | --- |
| Before training | 9.313225746154785e-9 |
| Selected checkpoint | 1.52587890625e-5 |
| Fresh replay | 1.52587890625e-5 |

The latter differences exceed the standalone absolute tolerance 1e-5 but do not contradict the actual gate, which uses `atol=1e-5` plus `rtol=1e-4`, not an absolute-only rule. Frozen source confirms this combined test. The minimum combined tolerance from the selected prediction logits for those first eight queries is 0.0006951613521575928, well above the recorded maximum difference. The pre-training maximum is below the absolute tolerance alone. Individual companion pairs were not rerun locally.

The selected-checkpoint buffer fingerprint is `d264f427cef4c7c98547996c838524f6895718b5ad20074a2140b789114e85df`, identical after selection and in replay. The change from pristine buffers occurs during training; evaluation does not change them according to recorded audits. Cumulative 100-epoch BN increments refer to the training trajectory, whereas selected-state inference uses epoch-79 checkpoint buffers.

Replay child PID 33962 differs from its invocation parent PID 33952. This parent is a later selected-checkpoint verification process, not original training PID 4476. The wrapper explicitly runs selected verification after training, so the different parent is not a discrepancy. Replay binds the selected checkpoint digest and identical runtime authorities, records all 222 validation images, and reports exactly zero replay logit difference. Its complete metrics/error/subgroup summary equals the selected validation summary. The selected gate passes with no failures.

## Evidence limitations and scope

Available local evidence consists of the seven unmodified metadata/CSV artifacts listed below, the frozen source archive, earlier metadata authorities, and launch-approval record. No model, weights, checkpoint bytes, or production pixels were loaded or inferred over locally. No SSH, submissions, commits, or subagents were used. Only this requested Markdown was written; no optional JSON was created.

Source/config/available metadata/approval/artifact byte hashes were independently recomputed. Checkpoint and weight digests are hash-bound recorded evidence, not independent local rehashes of absent checkpoint bytes. Pixel hashes and staging exclusions are supported by matching manifests and verified copy/loader code, not an independent local pixel-byte audit or GPU syscall trace. Fresh initialization, BN updates, inference buffer immutability, and replay are allocated-GPU output evidence interpreted against frozen source, not independently repeated GPU measurements. The seven-artifact bundle does not contain a second replay prediction CSV; zero replay difference is the recorded fresh-process comparison result.

The selected 222-image validation set also supplies checkpoint selection over 100 epochs. This is not an untouched test estimate. The official component overlap and fixed single fold/seed further restrict generalization and uncertainty claims. Both family summaries reuse the same 100 genuine examples; their altered populations are unequal, particularly the 16-example copy subset. This outcome review alone does not approve final family selection or additional runs; provisional candidate choice must consider the specified comparative evidence. These results alone do not establish reference benefit or BN as the exclusive cause of all historical D failures.

## Outcome artifact fingerprints

Paths below are relative to `results/runs/anchor-pilots-v1/anchor-pilot-train-37982`.

| Artifact | SHA256 |
| --- | --- |
| `input_manifest.json` | `350854e17dd8deff5abecdbdd4fb9127effbb6edd0c7370dd0a279e6165b1742` |
| `pilot/train.json` | `efe1112b09c3af69089cb62fd4b47373c33083a2ec434d77d9dab2b17a23f957` |
| `pilot/log.csv` | `098d14d35945ff4aca98e9cd8a5f1284967a6174b381ef890907c6af5da11352` |
| `pilot/val_predictions.csv` | `3eb408122583894330e59fd010e7db6798887ecf6c2371699d2b6c242f62a295` |
| `pilot/val_summary.json` | `7af70bf6619d9950d466ff4c36c6f0d0a94c70ff4c8f2e4d7fa07660a4824b32` |
| `pilot/gate.json` | `3765173504825db3be23aa87878247b0bbeac2e65768e6b9e5fbcbb46c2e8846` |
| `pilot/replay.json` | `2189a4ff8cab67c58ba1efedf4efa10fdfd3f2c8e479d4e2db645eddf7962ed0` |
