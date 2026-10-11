# Independent actual-gate review of D_running_pilot 37980

Reviewed on October 12, 2026. Reviewer is the independent Codex assistant in this session.

## Verdict and authorization boundary

**CLEARED for exactly one user-approved D_running_pilot fold-0, seed-42, 100-epoch validation pilot, subject to the identity-bound packaging requirements below. No blockers were found in the reviewed actual diagnostic evidence.**

This clearance applies only to the frozen source, configuration, data authorities, official initialization, and diagnostic gate recorded here. It does not authorize a sweep, another fold or seed, reference training, test evaluation, downstream detector evaluation, or another track. The original `D_running` remains diagnostic-only and full-training blocked. The failed original D and bounded diagnostic 37979 are not training authorities. Source095 remains unchanged as the authority for the earlier R/N/D-context work.

The coordinator reports Slurm job 37980 `COMPLETED`, exit `0:0`, elapsed `4:29`. This reviewer did not query Slurm or use SSH. The coordinator must retain its scheduler verification and bind this review to the exact gate and manifest hashes before submitting the single training job.

## Immutable authorities

| Authority | Exact value |
| --- | --- |
| Source revision | `bef7544a64f36a166f76eeef0d3130e353bd3203` |
| Source archive SHA256 | `00d0b99e635536d27a01c6f4f39bd94e14e98409dc5616c722f7dd4db3f8d5de` |
| Full source inventory SHA256 | `1dc29e362523ab3bcfdfaa794764dd85f2c8c1043fe3acae3381b89bec1bc69a` |
| Runtime source SHA256 | `af7073f3dd12299d9836f665cab5307dd5fed4a22882cc9fbbef317b8a38ae59` |
| Configuration bytes SHA256 | `604d888d046d1903ef1741476e0a3170bde05ec9cc8abef9482b43fe71072f79` |
| Effective configuration SHA256 | `5672a841eea3e683cdf76892d828c015711d500187e56fbe9c0a8aa86f30fdf9` |
| Records SHA256 | `afd661819fe9be97c83d2ee5123a7ce60869437c61d1e9029c4b8a96175aa276` |
| Fold-0 split SHA256 | `985d10b47461e0fa41d5502fbb51cd6fc03f8c7e8ec8cd1040b21be318efc1d9` |
| Official LEVIR weights SHA256 | `da3f569306dadd1fac5b64abc2a3d484571cd0ecf6410e1f152275cfe49a3618` |
| Data inventory SHA256 | `4276e25e9310ef5b76a9358f28a33fd261ea3c2eb2bbfc5f98cba82860b23680` |
| Input manifest SHA256 | `78d0a6242865639c96402e0b1686c1cc7a1c84078e8079b6256812fdf4499d03` |
| Parent gate SHA256 | `90644094ef97f7867d842c52a3a2be7243001d1628a7b8b43eb6db8f5a06cc54` |

The local archive revision header and archive digest were verified. All 165 manifest source files match both archive contents and extracted frozen files. The full source inventory, runtime source, effective configuration, and data inventory hashes were independently recomputed. The raw configuration hash matches the frozen YAML. Available local records/split copies from 37979 were independently rehashed against the identical authorities; no staged inputs were requested or copied for this review.

The new architecture identifier is `anchor_direct_running_ia18_pilot_v1`. Its resolved configuration differs from bounded `D_running` only in track and architecture identifier. The frozen implementation routes it through the same direct encoder, global pooling, and seeded MLP with running BN, not N's decoder. It retains the original geometry, zone dependency, augmentations, and zero localization weight.

## Membership and exclusions

The packaged manifest has exactly 1,778 training and 222 validation IDs. These sets match the official split independently parsed from the hash-verified metadata. Its 2,010 data entries are exactly 2,000 corresponding canvases and ten design zones. The complete data-file entries also equal the original D37963 train/validation inventory, including their recorded hashes and sizes.

There are no official test canvases, reference files, raw images, masks, or valid-target planes in this inventory. Frozen staging code selects train/validation only for the new track. The query loader reads canvases/zones only for this direct track; dense-target loading is restricted to N training. This is manifest and code evidence of exclusion, not an independent GPU syscall trace or local pixel-byte audit.

Overfit opens only 16 training rows. Smoke opens 32 training and 32 distinct validation rows and evaluates validation, unlike bounded 37979's training-only smoke. Replay selects the same validation rows under the frozen role and membership checks. Prediction IDs, labels, designs, attack families, and order match the appropriate metadata memberships. All phases record `no_test_open=true`.

## Actual stack and pristine initialization

Actual-stack XML on `gpu-dy-g5-0-16` records one `test_actual_opencd_gpu_integration` case, zero failures, zero errors, and zero skips. The frozen test explicitly includes the new track, compares its parameter tensors with the original direct model, checks all 22 running BN modules and official stored backbone buffers, verifies one query traversal and counter increment, checks finite gradients, companion invariance, immutable evaluation state, and state replay. The log reports one pass and six non-failing dependency/cache warnings.

Both actual phases record official hash-bound LEVIR initialization, 173 historical loaded keys, 132 active backbone keys including all 66 BN buffer keys, no unexpected keys, and head seed 42. All 11,261,825 parameters are active and the optimizer covers 70 tensors. Both optimizers start empty. Both phases report no checkpoint initialization.

Parent, overfit, smoke, and bounded 37979 pristine initialization dictionaries are identical.

| Pristine fingerprint | SHA256 |
| --- | --- |
| Complete state | `32e2fcde137e5dac4c537dc5e18ba3ff02cf8d12a72311e0177aa28c076cfec4` |
| Buffers | `bc4e5ed3fe4e999395997ded7981a9bd9a0acd198e21c5b342f348713aa64dad` |
| Active pretrained tensors | `9ab72d0bbe12d81de9034b1b44e5d037576a8e83f45addf0ba9d339382e59c42` |

The complete historical pretrained fingerprint remains `1283f34343b41b2a5e27d55ec9654bf1e7dae47e1959adab4f403c4e0884ef44`. These tensor fingerprints and official-weight loading are GPU output evidence interpreted against verified source. This reviewer did not load or independently hash checkpoint tensors.

## Independently recomputed predictions

CSV metrics were recomputed with standard-library arithmetic. Stable float64 BCE from logits differs from recorded FP32 BCE only by less than 1.1e-8. Accuracy, pairwise AUC, subgroup populations/metrics, false-positive IDs, and missed-alteration IDs agree, allowing ordinary floating-point rounding.

| Evaluation | Population | Correct | AUC | Recomputed float64 BCE | Recorded FP32 BCE |
| --- | --- | --- | --- | --- | --- |
| Overfit, singleton training evaluation | 8 genuine, 6 text_inpaint, 2 text_copy | 16/16 | 1.0 | 0.004217774654371644 | 0.004217768659145804 |
| Smoke, singleton validation evaluation | 14 genuine, 17 text_inpaint, 1 text_copy | 14/32 | 0.3333333333333333 | 0.6946411559102179 | 0.6946411523967981 |

The overfit IDs exactly equal both original D37963 and bounded 37979. Its prediction CSV is byte-identical to bounded 37979. It passes the declared criterion of at least 15 correct and mean image loss below 0.15 after 50 optimizer updates, within the 500-update bound. The overfit recipe uses two eight-image groups, AdamW at 0.001 with zero weight decay, no augmentation, scheduler, or clipping.

Smoke executes one actual recipe epoch, batch eight, accumulation one, four AdamW updates, pretrained/new learning rates 0.0001/0.001, weight decay 0.05, declared query augmentations, and gradient clipping at five. Its four-iteration warmup/cosine schedule has zero effective warmup and ends at zero learning rate. No training rows are dropped in this 32-row smoke.

All 18 smoke alterations are missed and no genuine validation example is falsely rejected. Copy subgroup AUC is 0.7857142857142857 with only one altered example; inpainting subgroup AUC is 0.3067226890756303 with 17 altered examples. These small, unfavorable smoke results are not evidence of validation competence or a family-selection basis. They do not contradict the explicitly bounded execution/correctness criterion.

## BN counters, inference, and selected-checkpoint replay

Every one of the 22 BN counters increments by two in each of 25 overfit training blocks, totaling 50 training forwards and 50 optimizer steps. Each increments by four during smoke, matching four training forwards and four optimizer steps. Accumulation is one, so forwards and steps coincide here; counters are not evidence of extra calibration or silent BN retraining.

Before smoke training, after selected-checkpoint restoration, and during fresh-process replay, each inference audit checks 72 companion comparisons at batch sizes 1/2/4/8. All 22 BN modules are in evaluation mode with running statistics. Before/after buffer fingerprints are equal within each audit.

| Audit | Maximum absolute companion difference |
| --- | --- |
| Before training | 9.313225746154785e-9 |
| After training | 1.30385160446167e-8 |
| Fresh replay | 1.30385160446167e-8 |

The post-smoke/replay buffer fingerprint is `8c0c84fe682f98944f1976d8988b78ab9b24abae0ebc01def8b57e0cc48236ef`. Its change from pristine initialization occurs during training; it stays unchanged during inference. This is recorded GPU immutability evidence, not an independently repeated GPU test.

Fresh-process replay PID 3574 differs from parent PID 3529. It binds the selected smoke checkpoint, identical runtime identities, all 32 validation predictions, and tolerances `atol=1e-5`, `rtol=1e-4`. Reported replay logit difference is exactly zero. Its complete summary matches the selected validation metrics. Smoke selected epoch is one and its key is validation accuracy followed by AUC.

Checkpoint metadata binds overfit SHA256 `ce7821292c30a11c42cf45ec6b967834c23f25f2a1f83f905ea635d910257d4e` and smoke SHA256 `6db02f09b4a8c810b01fbae1d9a14f9823a6e3d65fd31cce699c335bfd30ec18`. Checkpoint bytes were intentionally not fetched or independently rehashed locally.

## Conditions for the single continuation

1. Reuse the exact frozen `bef7544a64f36a166f76eeef0d3130e353bd3203` archive and configuration above, even if main changes. Bind the coordinator's independent review record to this gate, manifest, source/config/data/weights provenance, and scheduler verification. Adding this report must not cause the scientific source archive to be rebuilt from a later revision.
2. Start full training from fresh official LEVIR weights and the seeded head with empty optimizer state. Do not continue weights or BN buffers from overfit, smoke, 37979, or failed D37963. The runtime compares pristine state/buffer/pretrained fingerprints against this matching pretrain gate before training.
3. Execute exactly 100 epochs, fold zero, seed 42, with 1,778 training and 222 validation membership and no early stopping. The frozen batch-eight/drop-last recipe processes 1,776 training rows per epoch and records the two omitted IDs. This implies 222 training forwards/optimizer steps per epoch, not 223. Selection is validation accuracy, then AUC, retaining the earlier exact tie.
4. Preserve original `D_running` full-training rejection and strict checkpoint architecture/config/source identity separation. Maintain test/reference/dense-target exclusions. Keep the selected 222-image validation result, error lists, subgroup uncertainty, and curves distinct, and perform selected-checkpoint replay/review after completion.
5. Treat this as a legacy-geometry validation pilot, not a final support-safe arm or proof that BN was the sole cause of all original D failures. No additional training, test evaluation, reference extension, or downstream work is approved by this review.

## Evidence hashes and review method

All 244 source/artifact assertions passed. This review used local Python JSON/CSV/XML parsing, SHA256 hashing, archive inspection, frozen-source reading, and arithmetic only. No model inference, model import/loading, checkpoint loading, production-pixel access, SSH, submissions, commits, or subagents occurred. The only repository write is this review Markdown under `reviews/`.

| Artifact relative to diagnostic 37980 | SHA256 |
| --- | --- |
| `actual-stack-test.xml` | `78d89b6f14265da2c52d01621bceda5ef40695db487a5d5d60acaa16a8aa0fcc` |
| `initialization-provenance.json` | `fefafc6b695e4d40def79e369b33349308b7766689a9439200316a2c043329b0` |
| `pilot/overfit/train.json` | `2d2cd8d3f66194a0ffb5cb4091c722c2f4a7082f9f14bf4003cf25b5b5bd94d6` |
| `pilot/overfit/val_predictions.csv` | `26f4db5eb59659e7d38734ae1e57c6eb9e7f0025339757a57c173a9f6b6ca287` |
| `pilot/overfit/val_summary.json` | `01f7f4e24eb6ebd39cdceddff0692a30d1c7e9166896c04e52a58218cca98679` |
| `pilot/smoke/train.json` | `3aefe1411853146cd16a77338d7519724523c5782327ff7154c00cd4b698662d` |
| `pilot/smoke/val_predictions.csv` | `8ecee915c97cd79e9f1aa7fdda3609779c73b2d077a351d9ee9fd6267a91a3ba` |
| `pilot/smoke/val_summary.json` | `9ee6bba9fc528cfcb131856aa1a2f5160d29b933e3916fe447736c2fbeaa7c80` |
| `pilot/smoke/replay.json` | `35c3e2b3856b6499c15d706b850a742a080dd392e37bfbed6a52a18465380697` |
