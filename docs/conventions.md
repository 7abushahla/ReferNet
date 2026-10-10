# Conventions we adopt from published code (10 October 2026)

Four read-only code audits were run before writing any training code: SIDTD (`evidence/code-audit-sidtd-2026-10-10.md`), DeepID and Sunlight (`evidence/code-audit-deepid-2026-10-10.md`), Open-CD / ChangerEx and the two reference-based document baselines (`evidence/code-audit-opencd-2026-10-10.md`), and FLiD, FakeIDet2 and the ISO/IEC 30107-3 metric libraries (`evidence/code-audit-metrics-flid-fakeidet-2026-10-10.md`). This note records what we copy, what we deliberately do differently, and why. The audits are the evidence; this is the decision.

## 1. Scores, labels and thresholds

| Item | Convention adopted | Source | Why |
|---|---|---|---|
| Score orientation | One float in [0, 1]; 1 = bona fide, 0 = altered. Internally our label column is 1 = altered, so `score = 1 - p_altered` at export. | DeepID baseline, FantasyID notebook, bob, PyPAD | Both challenge APIs and the Idiap tooling take bona fide as the positive, high-scoring class. FLiD and FakeIDet2 use the opposite orientation; AUC is invariant, thresholds are not. |
| Fixed operating point | 0.5 on the bona fide score, with `>=` as bona fide | DeepID | Required to put our number next to the DeepID rows. |
| ISO rates | APCER = fraction of altered scored bona fide; BPCER = fraction of bona fide scored altered | ISO/IEC 30107-3 via bob.measure (`farfrr`) | In our vocabulary APCER is the miss rate and BPCER is the genuine-referral rate. The paper says "miss" and "referral" and defines them once as APCER and BPCER. |
| Threshold at a miss budget | `far_threshold` semantics: the largest threshold whose realized APCER on the calibration set is at most the target; no grid, no interpolation | bob.measure (BSD-3), copied by the FantasyID evaluation notebook | FLiD does the same on a 5,000-point grid; results are equal at these sample sizes but the observed-score sweep is the standard. |
| Where the threshold is chosen | On the source validation split (Protocol A: MIDV validation designs; Protocol B: the same; Protocol C: the source state's validation), then frozen, then applied to the unseen designs | FantasyID notebook (`far_value=0.1` on the validation protocol) and our dossier's "transported threshold" | FLiD and FakeIDet2 pick EER on the test fold. We state this difference explicitly in the literature table. |
| Miss budget | 5 percent (BPCER20 in ISO naming) as the primary policy point; 1 percent (BPCER100) as secondary | ISO naming, FLiD reports BPCER10 only | Five percent matches the dossier's policy framing and the FantasyID TruFor row (FPR 4.9). |
| EER | Midpoint `(APCER + BPCER) / 2` at the observed-score threshold closest to equality | bob `eer_threshold`, pyeer, FLiD | Reported for comparability with FakeIDet2 and FLiD, never used as our operating point. |
| HTER | `(APCER + BPCER) / 2` at the frozen transported threshold | FantasyID notebook | Only meaningful with a frozen threshold; PyPAD's warning that ACER is not an ISO metric is noted. |
| AUC | `sklearn.metrics.roc_auc_score` with bona fide positive, stated once | all | Orientation stated in the metrics section. |
| Tie handling | bob rule: a score exactly on the threshold is accepted as bona fide | bob | Stated once; irrelevant in practice with float scores. |
| Localization | Per-image F1 on the predicted mask, class-dependent polarity as DeepID: for altered images the positive class is the altered pixels; for bona fide images a correct all-clear mask scores 1 | DeepID | Required to put our number next to the DeepID localization rows. |
| Weighted F1 at 0.5 | Reported next to the DeepID detection rows | DeepID | Same reason. |

Implementation: `src/refer_or_clear/policy_metrics.py` already holds the transported-threshold and DeepID F1 logic. The bob functions (`farfrr`, `far_threshold`, `eer_threshold`, `min_hter_threshold`) are reimplemented in a small `iso_metrics.py` with attribution rather than importing bob, to keep the training environment free of the bob dependency tree. The 25-line PAI-species aggregation from bob.pad.base (GPL-3) is reimplemented from the definition, not copied.

## 2. Splits

| Item | Convention adopted | Source | Why |
|---|---|---|---|
| Identity grouping | Every split is grouped by identity; SIDTD provenance components (source plus donor of crop-and-replace) stay together | FLiD (`StratifiedGroupKFold` by `face_id`), dossier | Prevents the same face or source document appearing on both sides. |
| Design (template) disjointness | Protocol A and C test designs never appear in training; Protocol B trains on no FantasyID design at all | FakeIDet2 (Spanish v1 held out), SIDTD few-shot split, IDNet cross-type | This is the claim of the paper; FLiD does not enforce it and we say so. |
| SIDTD protocol | The official `split_shot` files (train quadruple `rus_aze_lva_fin` held out) | SIDTD repository | Keeps the published rows as anchors. We do not regenerate forgeries or invent a split. |
| FantasyID protocol | The official `train/` and `test/` folders, unpooled | FantasyID paper, DeepID | FLiD pools and re-splits by identity and loses the design hold-out; we keep the official folders so the published rows stay anchors. |
| Validation | 10 percent of training components by hash | ours | Model selection and threshold calibration never touch a test design. |

## 3. Data pipeline

| Item | Convention adopted | Source | Why |
|---|---|---|---|
| Geometry | Letterbox to a fixed 512x800 canvas with the transform saved; no per-field crops | ours (DeepID runs at native size; Open-CD crops 512x512) | The model consumes a whole document pair; the canvas keeps the design's layout so the median reference aligns. |
| JPEG equalization | Every canvas is re-encoded at quality 95 after rescale; training applies random JPEG at quality 60 to 100 to both inputs; test is reported on native canvases and on canvases re-encoded at a fixed quality 75 ("equalized") | our audit, DeepID report | The native files carry a perfect class shortcut, see Section 5. |
| Augmentation applied jointly | Crop, flip and affine are drawn once per pair; photometric jitter is drawn per image | Open-CD `MultiImg*` transforms | Joint geometry keeps the pair registered; independent photometrics mimic two captures. |
| Reference channel | Design median across the other identity fold | ours | Removes personal content and approximates a template-library specimen; CheckID and Chen et al. use random genuine references, CoAARC a random genuine of any nationality. |
| Zone channel | Union of genuine field rectangles, dilated 4 px, as a fourth channel on both inputs | ours (dossier B1) | The single-image control carries the same channel, so the reference is the only difference between method and control. |
| Masks | SIDTD: template field rectangle of the altered field dilated 8 px; FantasyID: altered region rectangles | SIDTD annotations, FantasyID JSON | Box-level masks are what both datasets supply; we do not claim pixel-accurate localization. |

## 4. Model and training (Open-CD)

| Item | Convention adopted | Source | Why |
|---|---|---|---|
| Framework | Open-CD 1.1.0 (Apache-2.0), six registered subclasses, no edits to Open-CD | audit A.5 | Environment is solved (prebuilt mmcv wheels); re-implementation costs 1.5 to 2 days and risks silent divergence (FDAF `align_corners=True`, column-only spatial exchange, IN+GELU in FDAF). |
| Stack | torch 2.1.2 cu121, torchvision 0.16.2, `mmcv==2.1.0` from the cu121/torch2.1 wheel index, `mmengine==0.10.4`, `mmsegmentation==1.2.2`, `mmdet==3.3.0` | audit A.4 | Inside `opencd/__init__.py` asserted ranges; mmdet is a hard import; `mmcv < 2.2.0`. |
| Config | `configs/changer/changer_ex_r18_512x512_40k_levircd.py` as base; `norm_cfg=BN` (single GPU); `backbone.in_channels=4`, `backbone_inchannels=4` | audit A.1, A.5 | SyncBN has no single-GPU fallback in `tools/train.py`. |
| Initialization | LEVIR-CD checkpoint `ChangerEx_r18-512x512_40k_levircd_20221223_120511.pth`; stem `(32,3,3,3)` zero-padded to `(32,4,3,3)`; everything else loads strictly | audit A.3 | Paper states: LEVIR-pretrained, not ImageNet (trained from random init), and the released checkpoint was selected on the LEVIR test split. |
| Loss | Open-CD softmax CE with OHEM (`thresh=0.7, min_kept=100000`) on the change map, plus BCE on a pooled image-level logit; map weight 3, image weight 1 | Open-CD config; TruFor/EdgeDoc weighting | Keeps the published ChangerEx loss for the map; the image head is ours. |
| Optimizer | AdamW lr 5e-3 for new modules, 1e-4 for pretrained, weight decay 0.05, linear warm-up 1,000 iterations then poly decay | Open-CD config, scaled for fine-tuning | Open-CD's own schedule, with the lower pretrained rate because we fine-tune. |
| Batch, input | Batch 8, pairs stacked along channels (8 channels: A, zone, B, zone), ImageNet mean/std on RGB, mask channel mean 0 std 1 | Open-CD preprocessor | Preprocessor BGR swap index becomes `[2,1,0,3,6,5,4,7]`. |
| Seeds | 42, 43, 44; mean and std over seeds | ours | FLiD reports std over folds only; we report over seeds at a fixed split. |
| Model selection | Best validation AUC on the source validation split; never on a test design | FLiD `train_innerval`, FantasyID notebook | FLiD's library path early-stops on the test fold; we do not reuse it. |

## 5. The JPEG shortcut (measured, 10 October 2026)

`scripts/audit_jpeg.py` estimates the encoder quality of every file from its luminance quantization table and then asks how well that single number separates the classes.

| Split | Genuine median QF | Altered median QF | AUC of "lower QF = altered" |
|---|---|---|---|
| Protocol A train / val / test | 92 | 75 | 1.000 / 1.000 / 1.000 |
| Protocol B train / val | 92 | 75 | 1.000 / 1.000 |
| Protocol B test, held-out designs | 93.5 | 75 | 1.000 |

In both datasets the quantization table alone classifies every image correctly. A detector trained on the native files can reach a perfect score without reading the document, and the shortcut survives into the held-out designs, so it would also fake "generalization". DeepID reported the FantasyID train-val part of this (QF 95 against 75); SIDTD has the same structure (92 against 75) and nobody has reported it.

Does our pipeline remove it? The letterbox rescale plus the QF 95 re-encode destroys the 8-pixel grid trace: an 8-pixel blockiness statistic has AUC 0.971 on the raw files and 0.513 on our canvases. That is one hand-made statistic; a CNN may find subtler traces, so the pipeline also applies random JPEG recompression (quality 60 to 100) to both inputs in training, and every headline number is reported twice, on native canvases and on canvases re-encoded at a fixed quality 75 for both classes. If the two numbers differ materially the model learned compression, not alteration, and the equalized number is the one that stands.

This audit table goes into the paper's data section. It is also a point against the published single-image rows trained on these datasets, which we state without claiming to know what each published model learned.
