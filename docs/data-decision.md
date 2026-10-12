# Dataset decision for the paper (10 October 2026)

This note fixes which public datasets the study uses, what each one is for, and what each one cannot do. Everything below was checked against the servers and the papers on 10 October 2026, not against secondary descriptions.

## Current execution qualification, 13 October 2026

The protocol numbering and architecture changed after this historical dataset survey. Current A is the official familiar-design SIDTD template anchor, B is SIDTD held-out nationalities, C is frozen SIDTD-to-FantasyID transfer, and D is the original WV/AZ IDNet pair. Follow `docs/superpowers/plans/2026-10-13-parallel-study-execution.md` and the execution ledger rather than older B/C numbering or localization claims below. The current model produces scalar classification scores, not pixel masks.

SIDTD classifier training uses the original official folded image populations. Trusted geometry, zones, and median support exclude each fold's validation/test components. B and C reserve deterministic target genuine components for enrollment and exclude their linked queries/attacks from both arms. C's actual classifier source is A fold zero, 1,778 training and 222 validation images, not the legacy C source split. Its frozen 599 target images become 66 genuine support, 66 excluded altered derivatives, and 467 queries comprising 234 genuine, 113 face swaps, and 120 text inpaints. Target outcomes do not select models or thresholds. The exact reduced C cohort has deterministic class-associated quantization dictionaries, documented in `evidence/protocol-c-locked-header-audit-2026-10-13.md`; this is a measured dataset association, not a demonstrated exclusive CNN mechanism.

Already downloaded FantasyID and AZ were transferred to the cluster. These transport receipts do not establish original Protocol D readiness. WV is not verified locally or on the cluster, and unresolved IDNet combined-morph donor provenance remains fail-closed. Do not download new archives autonomously, claim D completion, or silently substitute an AZ-only experiment. Later statements about exact published-population comparison, Arabic evaluation, face-task training, and localization metrics describe earlier proposals unless specifically implemented and authenticated under the current plan.

## 1. Decision

| Dataset | Role | Download | Size |
|---|---|---|---|
| SIDTD (built on MIDV-2020) | Classifier source for A and B. A uses official template folds; B holds out nationalities. A-selected models deploy unchanged in C. | `templates.zip`, `few_shot_split_templates.zip`, `kfold_split_templates.zip` from `datasets.cvc.uab.es/SIDTD/`; verified locally on 10 October 2026 at `~/Downloads/Datasets/SIDTD` (1,000 reals, 1,222 fakes, 10 real and 1,222 fake annotation files, the few-shot and 10-fold split CSVs, the generator repository) | 1.3 GB |
| FantasyID | C test, unseen designs and captured-source transfer with target genuine enrollment; face swaps reported separately from text inpainting. No classifier training on FantasyID. | `FantasyID.tgz`, Zenodo 17063366 | 2.6 GB |
| IDNet (Zenodo release) | Planned D, the original West Virginia/Arizona cross-type pair. Not implemented current results or B training. | `AZ.zip` (Zenodo 13852757), `WV.zip` (Zenodo 13854938); AZ exists, WV remains unverified | 45 GB planned |
| IDNet, optional | Second published pair: Finland to Spain (Table 15, cheapest, shares designs with SIDTD) or Greece to Albania (Table 16, the worst published numbers). | `FIN.zip`, `ESP.zip` (13852734) or `GRC.zip`, `ALB.zip` (13854938, 13854897) | 13.6 GB or 34.5 GB |

Total for the core study: about 49 GB. Nothing else is downloaded.

## 2. What each dataset is

### SIDTD

SIDTD is MIDV-2020 plus generated forgeries. The archive holds the 1,000 MIDV-2020 genuine templates (10 designs, 100 identities each) and 1,222 forgeries made by the released generator with two operations, crop-and-replace (a field copied from another document) and inpaint-and-rewrite (a field erased and rewritten). Each forgery record names the source template and the altered field, so the pixel mask is derivable from the field rectangle. A random sample of 80 forgery records: 78 alter a text field, 2 alter the photo region. SIDTD is therefore a text-tampering dataset. The official few-shot split trains on six designs (Albania, Spain, Estonia, Greece, Serbia, Slovakia) and tests on four (Azerbaijan, Finland, Latvia, Russia). The server has this split under two file names (`split_shot.zip`, `few_shot_split_templates.zip`); their contents are the same.


What the published few-shot numbers actually are (audit of the SIDTD code, 10 Oct 2026). They are not a classifier trained on six nationalities and tested on four. The scripts `models/explore/train_fsl.py` and `test_fsl.py` run a prototypical network: a frozen ImageNet backbone with a small trainable head, trained by 2-way, 5-shot, 5-query episodes on the cropped video clips of six nationalities, and tested by 100 episodes on the four held-out nationalities, where every test episode receives 5 labelled genuine and 5 labelled forged clips of the test nationality as its support set. The four held-out nationalities are drawn ten times from a hard-coded list of quadruples, and the README's mean and standard deviation are over those ten partitions. Checkpoints are selected on episodes drawn from the held-out nationalities (a leak the paper does not mention), no seed is set, and no validation set exists. The split file on the server (`few_shot_split_templates.zip`, test = Azerbaijan, Finland, Latvia, Russia) is one of the ten published quadruples (`rus_aze_lva_fin`). Metrics: accuracy by argmax and ROC AUC by scikit-learn on the pooled probability of "fake"; their FPR and FNR treat genuine as the positive class, like FantasyID.

Consequence (revised 10 Oct, spec docs/superpowers/specs/2026-10-10-sidtd-anchor-protocols-design.md). Protocol A follows the paper's template 10-fold protocol exactly, so our control can be checked against the published composite-detection rows; Protocol B uses the paper's ten nationality groups on templates, with no labelled forgeries, compared against that control. The few-shot rows on clips stay out of our tables; the clips were downloaded and audited (section below) and are not used.

A second fact from the generator (`DataGenerator/Midv.py`): inpaint-and-rewrite erases the field and redraws the same original value in a DejaVu font, so those 1,078 forgeries change the rendering, not the content. Crop-and-replace pastes the field of another document of the same nationality with a random pixel shift of up to 10. Neither stores a pixel mask; the altered region is the source template's field rectangle plus the shift, which is why our masks dilate it. The committed generator code no longer runs as shipped, which is one more reason not to regenerate forgeries.

Consequences. MIDV-2020 is not downloaded: its templates are inside SIDTD, and its photos and scans are all genuine. SIDTD and MIDV-2020 count as one source and are cited together. The official SIDTD downloader is not used: it builds split-file links without the server address and mirrors the whole folder, which includes 54 GB of video.

### FantasyID

Thirteen fictional designs in ten scripts. Cards were printed on PVC and captured with an iPhone 15 Pro, a Huawei Mate 30 and a Kyocera office scanner (about 3,000 x 1,900 px from the phones, 2,000 x 1,250 from the scanner). The complete archive (2.4 GB unpacked) was inventoried locally on 10 October 2026 at `~/Downloads/Datasets/FantasyID`; every image has a JSON beside it.

| Folder | Designs | Images | Content |
|---|---|---|---|
| `train/bonafide/{huawei,iphone15pro,scan}` | the 10 train-val designs: Arabic, Chinese, Hindi (indian), French, Persian, Portuguese, Russian, Turkish, Ukrainian, English (usa) | 633 (211 cards x 3 devices) | genuine captures |
| `train/attack/digital_1` | same 10 | 633 | InSwapper face swap plus DiffSTE text rewrite |
| `train/attack/digital_2` | same 10 | 633 | FaceDancer face swap plus TextDiffuser2 text rewrite |
| `test/bonafide/{huawei,iphone15,scan}` | the 3 held-out designs: chinese2, netherland, singapore | 300 (100 cards x 3) | genuine captures |
| `test/attack/digital_3` | the 10 train-val designs | 786 (262 x 3) | Attack-1: text rewrite with fine-tuned TextDiffuser2, always 3 fields (dob, doe, a name field) |
| `test/attack/facedancer` | the 3 held-out designs | 150 | Attack-2: FaceDancer face swap |
| `test/attack/textdiffuserft_bfei` | the 3 held-out designs | 149 | Attack-3: text rewrite as in Attack-1 |

Totals match the paper: 786 plus 1,266 in train-val, 300 plus 1,085 in test. The paper's Attack-1, Attack-2, Attack-3 map to the folders `digital_3`, `facedancer`, `textdiffuserft_bfei`; DeepID's FA1, FA2, FA3 are Attack-1, Attack-3, Attack-2 respectively.

Annotations: one JSON per image in VIA style with every field rectangle, its value, language and `region_provenance` in {original, altered}. Face swaps mark the `face` rectangle as altered (two per image, the portrait and the ghost image). So pixel masks are derivable for every attack, and the zone mask for every design comes from the original rectangles. `cropping_info` gives the document quadrilateral and the homography used to crop, so rectification is provided.

Consequences for the protocols. (1) Our model never trains on FantasyID, so every FantasyID design, Arabic included, is unseen in Protocol B. (2) The Arabic design is one of FantasyID's ten train-val designs, so it appears in the test set only inside Attack-1 (text rewrites) and in the train-val face swaps, not in the official out-of-domain test. The headline Protocol B number is on the official test (Attack-2 and Attack-3 plus the 300 bona fide on the three held-out designs), to match the published tables; Attack-1 is a second result, with the Arabic cards broken out. The published zero-shot baselines (TruFor, MMFusion) had not seen Arabic either, but they report Attack-1 as one number, so the Arabic row compares only with our own TruFor rerun on that subset. (3) Pixel F1 is available on Protocol B.

### IDNet

The arXiv paper (2408.01690 v2) and the BigData 2024 paper both describe 837,060 images of 20 designs (10 US states, 10 European countries), with six fraud types per design. The Zenodo release (eight records, CC0) ships one zip per design. Inside `SD.zip`, read directly from Zenodo without downloading: seven folders of 5,979 images each (`positive`, `fraud1_copy_and_move`, `fraud2_face_morphing`, `fraud3_face_replacement`, `fraud4_combined`, `fraud5_inpaint_and_rewrite`, `fraud6_crop_and_replace`), plus per-image field annotations with bounding boxes, original and new values, fonts and sizes.

The published storage sizes are not the Zenodo sizes. Table 1 of the paper lists South Dakota at 32 GB; the Zenodo zip is 4.9 GB. The Zenodo images are 795 x 487 JPEG files of about 118 KB (the `.png` names are JPEG data). The release is a reduced-resolution, recompressed version of what the paper describes. Counts match (5,979 per class). The sample images carry no visible "For Research Purposes" overlay.

Annotation mapping, verified by file-name overlap: `fraud1_copy_and_move`, `fraud2_face_morphing`, `fraud3_face_replacement` and `fraud4_combined` reuse the original file names and are described by `*_textfieldreplacement_annotation.json`, `*_facemorph_annotation.json`, `*_facereplacement_annotation.json` and `*_combined_fraud_annotation.json`. The paper's "Text Replacement" task is therefore `fraud1_copy_and_move`. `fraud5` and `fraud6` have their own `_fake_` file names and their own JSON files. To confirm visually on the first download.

Which designs, and why. The paper's cross-type experiments (arXiv v2, Section 6.3) use six designs in pairs: West Virginia and Arizona in both directions with all four tasks (Table 14), Finland to Spain (Table 15), Greece to Albania (Table 16), Arizona to Spain and Albania (Table 17), Finland to Albania and Arizona (Table 18), Greece to Spain and Arizona (Table 19), and face morphing across Arizona, Greece and Finland (Table 20); the within-design face-morphing study (Table 13) adds Nevada. We take West Virginia and Arizona because Table 14 is the only pair reported in both directions with all four tasks, it is the pair our introduction quotes, and the two designs lie outside SIDTD's design family, so Protocol C tests designs that nothing else in the study has seen. Cost: 45 GB. The cheapest second pair is Finland to Spain (Table 15, one direction, 13.6 GB); it also shares its designs with SIDTD, which allows a check of IDNet's synthetic renders against SIDTD's templates. Greece to Albania (Table 16, 34.5 GB) carries the worst published numbers. Both are optional flags in the download script.

Published cross-type results (arXiv v2, Section 6.3, Tables 14 to 20; absent from the BigData version). Accuracy of a detector trained on one design and tested on another, MixFaceNet-S for face tasks and a five-layer CNN for text:

| Train | Test | Face morphing | Portrait substitution | Mixed | Text replacement |
|---|---|---|---|---|---|
| WV | AZ | 50.55 | 71.20 | 71.15 | 60.92 |
| AZ | WV | 54.10 | 80.05 | 71.10 | 50.00 |
| GRC | ALB | 45.15 | 55.55 | 54.65 | 35.56 |
| FIN | ESP | 78.80 | 55.25 | 49.00 | 50.01 |

Split used by the paper for face tasks: the first 4,979 identities train, the remaining 1,000 test. The ordering is not stated; we sort file names and take the last 1,000 as test, and say so.

Consequences. IDNet is the dataset in which the problem the paper addresses was measured, so Protocol C compares directly with the numbers the introduction cites. It supplies face morphing and portrait substitution, which SIDTD lacks and FantasyID tests. It is fully synthetic, so it says nothing about capture. It has no official split beyond the sentence above. Results are on the released 795 x 487 version, not the originals the paper used.

The Hugging Face release IDNet-2025 is not used. It covers the same 10 European designs as SIDTD with only the two SIDTD-style text attacks, plus simulated scans. It adds no face attacks and no new designs.

## 3. Datasets considered and left out

| Dataset | Why not |
|---|---|
| MIDV-2020 (full) | All genuine; templates already in SIDTD. Photos and scans optional for a robustness check (`WITH_MIDV=1`). |
| MIDV-Holo | Hologram copies: a presentation attack, not altered content. |
| DLC-2021 | Colour copies and screen recaptures: presentation attack. |
| MIDV-500, MIDV-2019 | Older genuine-only sets. |
| IDNet-2025 (Hugging Face) | Same designs and text attacks as SIDTD; no face attacks. |
| DeepID test set, IDNow private set | Not public. |

## 4. Protocols

| Protocol | Train | Test | Compared with |
|---|---|---|---|
| A: anchor, the paper's setting | SIDTD templates, the fold's training file (10 official folds) | the fold's test file | SIDTD Table 5 composite-detection rows (all five models) |
| B: unseen design, same source | SIDTD templates, 6 nationalities (10 paper groups) | the group's 4 held-out nationalities | the control validated in A |
| C: unseen source, generator, capture, script | SIDTD (all 10 designs) plus IDNet AZ and WV (face attacks) | FantasyID official test; Arabic design alone | FantasyID published TruFor and MMFusion rows |
| D: unseen design, published pair | IDNet WV (all four task folders as positives) | IDNet AZ, per task; and the reverse | arXiv Table 14 |

Reference for every protocol: the pixel-wise median over genuine documents of the test design, built from genuine images that are not test samples. For Protocol D the reference uses the 4,979 non-test genuine images of the target design and nothing else from it; the detector never trains on the target design.

Difference from the published Protocol C setup, to state in the paper: the paper trained one detector per task; we train one detector on all four tasks and report per-task accuracy from the same model, at threshold 0.5, plus AUC.

## 5. Limitations to state

1. All training forgeries come from two generators (SIDTD's and IDNet's); FantasyID is the only test beyond them.
2. SIDTD and IDNet are digital renders; capture is tested only by FantasyID.
3. IDNet results are on the reduced Zenodo release.
4. Training designs are European and US; the Arabic design appears only at test.
5. The Centre's 902 irregularities in 20,307 examinations is a ratio among examined documents, not a prevalence among travellers; workload projections use assumed prevalence scenarios.

## JPEG quality is a class label in both datasets (audit of 10 October 2026)

`scripts/audit_jpeg.py` reads the luminance quantization table of every file. SIDTD reals are encoded at quality 92 and every SIDTD forgery at 75. FantasyID train bona fide are at 95 and train attacks at 75; test bona fide at 93.5, the held-out-design attacks at 75, and Attack-1 at 95. On every split of Protocols A and B the quantization table alone classifies every image correctly (AUC 1.000). The canvases we train on are rescaled and re-encoded at quality 95, which removes the measurable block-grid trace (blockiness AUC 0.971 on raw files, 0.513 on canvases); random JPEG recompression on both inputs stays in training and every headline number is also reported on test canvases re-encoded at a fixed quality 75 for both classes. Details and the adopted rule are in `docs/conventions.md`, Section 5.

## Blank specimens: who has them (checked 10 October 2026)

The deployment reference is the blank specimen of a design, as a template library holds it. No forgery dataset we test on ships one. MIDV-2020's `templates.tar` holds the 1,000 filled renders; FantasyID distributes captures and forgeries, not its 13 designs; IDNet's own paper states that "high-quality, blank templates of real-world documents are generally unavailable" and builds its 20 templates by Stable Diffusion inpainting of one public sample per design (US DMV sample licences; MIDV specimens for the ten European designs). The Zenodo zips list positives, fraud folders and metadata; whether a template file is inside AZ.zip and WV.zip is checked when they are staged. Two consequences:

- IDNet's ten European designs are the ten MIDV-2020 designs, so training on IDNet EU zips would leak the SIDTD held-out designs (rus, aze, lva, fin). Protocol B therefore uses only the US zips (AZ, WV); the EU zips stay optional and only for Protocol C variants that never touch Protocol A's test designs.
- IDNet documents are generated from one fixed template with identical geometry, so the median over IDNet genuine images is the template itself outside the fields. Protocol C is therefore the closest we get to a true specimen reference without a template file; Protocols A and B use the registered median as the stand-in and the paper says so.

Blank templates do exist for DocXPand's nine fictitious designs (SVG, generator MIT, data CC BY-NC-SA), which ship no forgeries; rendering the true blank there and comparing it with the median is future work.

## Blank specimens recovered from the IDNet paper (10 October 2026, afternoon)

The IDNet paper figure embeds its twenty Stable Diffusion-inpainted blank templates as separate images; extracted from the PDF they come without the "FOR RESEARCH PURPOSES" overlay, at 238 to 1,334 px width (Arizona is the smallest at 238 px, West Virginia 1,013 px, the ten European designs 931 to 1,334 px). Registered to our SIDTD design medians, all ten European blanks give near-identity transforms (rotation within 0.01 degrees, scale 0.993 to 1.006, 190 to 1,316 inliers): they are the SIDTD designs. Unlike the median they carry no ghost face and no residual text. They are staged as `data/raw/idnet_templates/<design>.jpg` with `index.csv`, written by `build_specimens` as `references/<design>_specimen.jpg`, and enter the reference ablation as the "specimen" variant for Protocol A (all ten designs) and Protocol C (WV at full quality, AZ at figure resolution). The inpainted field interiors are hallucinated background, so the zone channel still marks them. Provenance is stated in the paper and the files are cited to the IDNet paper; the IDNet data release is CC0, the figure is the authors'. FantasyID has no blank, so Protocol B stays on the registered median.

## SIDTD cropped clips: audit after download (10 Oct 2026)

`clips_cropped.zip` (28,514,444,140 bytes, byte-exact) and `few_shot_split_clips_cropped.zip` are on the cluster under `data/raw/sidtd/`.

| Check | Result |
|---|---|
| Forged clips | 7,214, matches the paper |
| Genuine clips | 66,823; the paper states 68,409. The split files and the archive agree with each other: every split path exists and every image is in a split |
| Annotations | forged clips only (7,214 JSON files) |
| Clip split test nationalities | alb, fin, srb, svk. This differs from the template split (aze, fin, lva, rus) and is not one of the ten quadruples hard-coded in `test_fsl.py`, which ignores the CSV |
| Forged videos | 191 in total, 16 to 32 per nationality, 28 to 48 clips per video |
| Genuine videos | 100 identities per nationality, from MIDV-2020 |

Capture confound. Genuine clips are MIDV-2020 frames; forged clips were printed on a laser printer, laminated and filmed by CVC staff on their own phones (paper, Methods). The two classes come from different capture pipelines. File size alone (bytes of the JPEG) separates them with AUC 1.000 within every nationality and 0.983 pooled; median 0.23 MB genuine vs 1.21 MB forged. Any classifier can therefore score near 1.0 on the clip protocols from capture cues without seeing the alteration, which is consistent with the near-perfect AUC reported by Talarmain et al. (ICDAR 2024) on this protocol. The episode sampler also draws support and query clips from about 19 forged videos per nationality, so query frames frequently share a video with a support frame.

## The file-format cue in SIDTD, traced to its source (11 Oct 2026)

| Subset | Genuine files | Forged files | Where the forged format comes from |
|---|---|---|---|
| Templates | JPEG quality 92, all 1,000 | JPEG quality 75, all 1,222 | the released generator saves forgeries with `PIL.Image.save(path)` and no quality argument (`SIDTD/data/explore/fake_id_ex.py:135`); PIL's JPEG default is quality 75 |
| Cropped clips | JPEG quality 95 (300 of 300 sampled) | PNG, saved under a `.jpg` name (100 of 100 sampled) | not documented; it explains the file-size gap (median 0.23 MB against 1.21 MB) recorded in the clip audit above |

The label is therefore written into the file format in both subsets. The paper's released template checkpoints, scored on the paper's folds, reproduce Table 5 exactly (ResNet50 98.1 percent, EfficientNet-B3 99.4 percent, AUC 1.000) and fall to AUC 0.655 and 0.539 when every test file is re-encoded at one quality (`results/tables/sidtd_checkpoints.md`). Published work on SIDTD: the dataset paper (templates and clips, raw files); Talarmain et al., ICDAR 2024 (clips, AUC about 1.00); the eigenface saliency PAD paper, arXiv 2606.13528 (the 2,222 raw templates); none mentions compression or file format. DocVerify (Pattern Recognition Letters 2026) tests on SIDTD after training on FantasyID and reports near chance; its preprocessing is not in the abstract and the full text is paywalled.
