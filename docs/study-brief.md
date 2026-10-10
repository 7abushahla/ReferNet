# Study brief: Refer or Clear (10 October 2026)

What the published literature does, what we do, why, and whether the data suffice. Every number below was read from the cited paper on 10 October 2026, not from a secondary source. Dataset facts are in `data-decision.md`.

## 1. The problem in one paragraph

A first-line officer decides in seconds whether to clear a travel document or refer it to the Document Examination Centre. Commercial readers compare the document against a template library of specimens, but no vendor publishes error rates on designs outside that library, and the published learned detectors take a single image and memorize the designs they were trained on. The paper asks one question: when the detector is also given a genuine specimen of the same design (the reference), how much of the loss on never-seen designs is recovered, and what referral rate does that imply at a fixed miss budget?

## 2. What the published literature does (2024 to 2026)

| Work | Task | Input | Protocol | Result on unseen designs | Code |
|---|---|---|---|---|---|
| SIDTD, Scientific Data 2024 | altered vs genuine, 10 MIDV-2020 designs | single image | few-shot: 6 nationalities train, 4 test, on cropped video clips (unbalanced partition) | ResNet50 90.6 acc / 0.963 AUC; EfficientNet-B3 82.0 / 0.899; ViT-L 91.0 / 0.969; TransFG 83.6 / 0.907; CoAARC 62.8 / 0.675. Within-design k-fold: 96 to 99 | generator and loader public |
| IDNet, arXiv 2408.01690 v2 (2024) | six fraud types, 20 synthetic designs | single image | train on one design, test on another (Tables 14 to 20) | WV to AZ: face morph 50.6, portrait swap 71.2, mixed 71.2, text 60.9; GRC to ALB: 45.2 / 55.6 / 54.7 / 35.6; within-design 98 to 100 | data public (Zenodo, reduced), no training code |
| FantasyID, IJCB 2025 | face swap and text inpainting on printed and captured fantasy cards | single image | pretrained forensic detectors, zero-shot; test designs absent from train | TruFor AUC 93.5 overall, 99.5 text, 55.8 face, HTER 33.4, FNR 62 at FPR 10; MMFusion 94.4 / 99.8 / 61.5, HTER 25.8 | data public |
| DeepID challenge, ICCV-W 2025 | same, plus a private set of 20K real IDs | single image | 26 teams; test includes text (FA2) and face (FA3) attacks on card templates unseen in training | Winner Sunlight (Re-MTKD/Cue-Net): F1 0.991 on fantasy cards, 0.719 on real IDs; best on real IDs UAM-Biometrics 0.788; all but one entry built on TruFor; "generalization to unseen manipulations and unseen card types remains challenging" | Sunlight weights public |
| DocForge-Bench, arXiv 2603.01433 (2026) | zero-shot localization across 8 document datasets incl. FantasyID | single image | frozen published weights, 14 methods | best Pixel-F1 on FantasyID 0.296 (TruFor); Pixel-AUC moderate, F1 near zero: a calibration failure | benchmark public |
| FLiD, arXiv 2605.09089 (2026) | field-localized detection on FantasyID | single image, YOLO field crops | identity-disjoint 5-fold, no unseen design | face EER 24.7, BPCER10 43.0 | public |
| FakeIDet2 (Inf. Fusion 2026) and FakeIDet3-DB (arXiv 2607.26641, 2026) | PAD and digital attacks on 47 real Spanish ID types | privacy patches | patch benchmark, held-out ID templates mixed with unseen attack settings | best detection EER 32.5 (Re-MTKD); localization AUC 83.5 | patches public, documents not |
| Das, Kumar, Ramachandra survey, arXiv 2607.01442 (2026) | zero-shot stress test on 140 synthesized cards of unseen templates, printed on PVC | single image | public detectors and multimodal models at threshold 0.5 | APCER 24 to 37 percent for every model; EER 13.8 (GPT-5) to 50.7 (Sunlight); "substantial limits in cross-domain generalisation" | none |
| Nieto-Hidalgo, Espin, Tapia, arXiv 2608.16591 (2026) | presentation attacks (print, screen, PVC), not alterations | single image plus 4 bona fide samples of the target card type at inference, as prototypes | train on one country, test on unseen countries; private data and DLC-2021 | EER 9.07 with prototypes vs 26.56 with softmax, vs CLIP zero-shot 20.7 | none |
| Third competition on document forgery detection, arXiv 2607.15734 (2026) | PAD and composite physical forgeries, 27K sequestered real images | single image | injection attacks excluded | Track 1 winner Incode | none |
| Li et al., CVPR-W 2026 (Persona) | layout embedding for open-set fraud discovery | single image | proprietary US and Canadian IDs | 99.8 layout classification; 276 fraud cases surfaced | none |
| IDSPACE, arXiv 2609.03052 (2026) | generator and 359,240-image dataset, 10 EU designs, scanned and mobile renders | n/a | n/a | n/a | Apache-2.0 on Hugging Face |

Reference-based detectors that exist: CheckSim (ICDAR-W 2021, patch similarity against an aligned reference, private data, one design); Chen et al. (TIFS 2022, triplet network with a same-template reference, but for recapture, not alteration); Al-Ghadi et al. 2022 and FMIDV (MMSP 2023, Siamese with a random genuine of the same country, copy-move of guilloche only; the follow-up that dropped the reference at test fell to AUC about 0.5); Neitthoffer et al. (ICDAR-W 2023, template through attention works, channel stacking fails). None is evaluated leave-design-out on alterations with public data. The 2026 survey does not treat reference conditioning as a paradigm at all; it mentions references only for recapture detection and for a method that "eliminates the need for reference templates".

Industry: Regula (16,500 templates), Keesing, Smart Engines, Incode, Thales, and the Idakto edge pipeline (SSIM template matching against a library of 128 countries). None publishes cross-design error rates.

What this table says. (1) Every published detector, including all 26 DeepID entries and every model in the 2026 survey, is single-image. (2) Every measurement on unseen designs is poor: 35 to 71 percent accuracy on IDNet, AUC 56 to 62 on FantasyID faces, APCER above 24 percent in the survey, a drop from 0.99 to 0.72 F1 from fantasy cards to real IDs. (3) The one 2026 method that improves generalization across countries, Nieto-Hidalgo et al., does so by giving the model genuine samples of the target card at inference. That is the reference idea, applied to presentation attacks with whole-image prototypes and private data. Nobody has applied it to alterations with localization on public data.

## 3. Our method, and what is and is not new

Method. A two-input detector. Input one is the rectified query document. Input two is the design-median reference, the pixel-wise median over genuine documents of the same design, which removes personal data and keeps the printed design. A third channel is the zone mask marking personal-data fields. The network is ChangerEx from Open-CD, a published change-detection architecture, with two heads: a pixel map of altered regions and a global referral score. Two variants: difference fusion (the standard one) and attention fusion (query attends to the reference, with the zone mask as an attention bias). A wrong-design gate flags a query that does not match its reference.

Not new: the architecture (Fang et al. 2023), the median-reference idea (template libraries do it by hand), the datasets.

New: (1) the first learned reference-conditioned alteration detector evaluated leave-design-out on public data, with masks, against the published single-image numbers; (2) the measurement itself, which no vendor and no paper provides; (3) the referral framing, referral rate at a miss budget with the Centre's own prior, which turns a detector score into the officer's decision; (4) the released protocol, splits, references and code, so GDRFA can repeat the measurement on its own template library.

Positioning sentence for the paper: published detectors memorize designs; template libraries generalize by hand; we make the template library an input to the learned detector and measure what that buys on designs the detector has never seen.

## 4. Protocols: what we train on, what we test on, and why

| Protocol | Train | Test | Why this test | Published comparison |
|---|---|---|---|---|
| A: unseen design, same source | SIDTD, 6 nationalities (1,328 images) | SIDTD, 4 held-out nationalities (894 images) | the only public split with published numbers on unseen designs for text alterations | SIDTD Table 5 few-shot rows; note they are on cropped video clips, ours on templates, so we also train our own single-image ResNet-50 and EfficientNet-B3 on the identical template split as the controlled comparison |
| B: unseen source, generator, capture, script | SIDTD all 10 designs plus IDNet AZ and WV (adds face morphing and portrait substitution) | FantasyID official test: 300 genuine plus Attack-2 (face) and Attack-3 (text) on the 3 designs unseen in train-val (Chinese, Dutch, Singaporean); Attack-1 (text on the 10 train-val designs, which include Arabic) reported separately with the Arabic cards broken out; our model never trains on FantasyID, so all 13 designs are unseen to it | the only public test with printed and camera-captured cards, a different generator, face attacks, pixel masks and a document quad per image | FantasyID Table 3 (TruFor, MMFusion); DocForge-Bench Pixel-F1; DeepID FA2 and FA3 |
| C: unseen design, published pair | IDNet WV, four attack folders as positives (about 10,000 images after balancing) | IDNet AZ, 1,000 test identities per class, per task; and the reverse | direct rerun of the published failure we cite in the introduction | IDNet Table 14 |

In every protocol the reference for a test design is built from genuine images of that design that are never test samples and never training samples (leave-identity-out median for A and B; the 4,979 non-test genuine images for C). The detector never trains on a test design. The single-image baseline uses the same code with the reference input removed, so the reference is the only difference.

## 5. Is there enough data?

| Protocol | Training images | Training pairs after augmentation | Test images | Comparable published training size |
|---|---|---|---|---|
| A | 600 genuine, 728 altered | about 40,000 (affine, JPEG, photometric, reference re-draws) | 400 genuine, 494 altered | SIDTD baselines trained on the same 1,328 |
| B | 1,000 genuine, 1,222 altered (SIDTD) plus 2,000 genuine, 4,000 altered sampled from IDNet AZ and WV | about 150,000 | 300 genuine, 1,085 altered | TruFor was pretrained on 876K natural-image forgeries, then applied zero-shot; the DeepID winner fine-tuned on 786 genuine and 1,572 altered FantasyID train images |
| C | 4,979 genuine, 4,979 altered (balanced across four tasks) | about 100,000 | 1,000 genuine, 4,000 altered | IDNet used 4,979 per class |

Yes, for a fine-tuned ResNet-18 change detector initialized from the LEVIR checkpoint. Protocol A is small, but the published baselines used the same split and the result is a mean over three seeds with bootstrap intervals. Pixel masks exist for all three protocols: SIDTD and IDNet from field rectangles, FantasyID from its per-image JSON with altered-region provenance.

## 6. Metrics

AUC, EER, APCER at BPCER 1 and 5 percent, HTER at the FantasyID operating point, accuracy at 0.5 where the published row uses it, pixel F1 and IoU where masks exist, and the referral rate at miss budgets of 1 and 5 percent with the Centre's prior of 4.4 percent among referred documents.

## 7. Comparability caveats to print

1. SIDTD's few-shot rows are on cropped video clips; ours are on templates. The controlled comparison is our own single-image run on the same split.
2. IDNet's Zenodo release is 795 x 487 JPEG; the paper's numbers were on the originals.
3. IDNet trained one detector per task; we train one on all four and report per task.
4. Published rows are quoted, not rerun, except our own single-image baselines.
5. Training forgeries come from two generators; capture is tested only by FantasyID.

## 8. What would make the paper fail, and the answer

- The reference does not help on Protocol A. Then the paper reports a negative result against the published rows and the controlled baseline; the referral analysis still stands on the best single-image model. The FMIDV precedent (AUC 0.5 without the reference) makes this unlikely for text alterations.
- Registration fails on FantasyID photos. The wrong-design gate catches it; the misregistration sweep quantifies the degradation; the model still produces a single-image score when the gate trips.
- Capture gap sinks Protocol B. Augmentation first; SIDTD's cropped clips (28.5 GB, forged captures included) second; reported as a limitation if neither suffices.

## 9. Open items for the authors

Researcher 3; job titles, degrees, phone numbers; native-speaker check of the Arabic title and abstract; confirmation of the torch 2.1.2 and mmcv stack on the cluster.
