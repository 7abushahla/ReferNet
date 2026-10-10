# Dataset decision for the paper (10 October 2026)

This note fixes which public datasets the study uses, what each one is for, and what each one cannot do. Everything below was checked against the servers and the papers on 10 October 2026, not against secondary descriptions.

## 1. Decision

| Dataset | Role | Download | Size |
|---|---|---|---|
| SIDTD (built on MIDV-2020) | Training source for Protocols A and B. Protocol A test (nationality hold-out). | `templates.zip`, `few_shot_split_templates.zip`, `kfold_split_templates.zip` from `datasets.cvc.uab.es/SIDTD/` | 1.3 GB |
| FantasyID | Protocol B test: unseen designs, unseen generator, printed and captured, Arabic design. | `FantasyID.tgz`, Zenodo 17063366 | 2.6 GB |
| IDNet (Zenodo release) | Protocol C: the published cross-type pair West Virginia and Arizona, both directions. Face attacks for Protocol B training. | `AZ.zip` (Zenodo 13852757), `WV.zip` (Zenodo 13854938) | 45 GB |
| IDNet, optional | Second published pair, Greece to Albania (the worst published numbers). | `GRC.zip`, `ALB.zip` (Zenodo 13854938, 13854897) | 35 GB |

Total for the core study: about 49 GB. Nothing else is downloaded.

## 2. What each dataset is

### SIDTD

SIDTD is MIDV-2020 plus generated forgeries. The archive holds the 1,000 MIDV-2020 genuine templates (10 designs, 100 identities each) and 1,222 forgeries made by the released generator with two operations, crop-and-replace (a field copied from another document) and inpaint-and-rewrite (a field erased and rewritten). Each forgery record names the source template and the altered field, so the pixel mask is derivable from the field rectangle. A random sample of 80 forgery records: 78 alter a text field, 2 alter the photo region. SIDTD is therefore a text-tampering dataset. The official few-shot split trains on six designs (Albania, Spain, Estonia, Greece, Serbia, Slovakia) and tests on four (Azerbaijan, Finland, Latvia, Russia). The server has this split under two file names (`split_shot.zip`, `few_shot_split_templates.zip`); their contents are the same.

Consequences. MIDV-2020 is not downloaded: its templates are inside SIDTD, and its photos and scans are all genuine. SIDTD and MIDV-2020 count as one source and are cited together. The official SIDTD downloader is not used: it builds split-file links without the server address and mirrors the whole folder, which includes 54 GB of video.

### FantasyID

Thirteen fictional designs in ten scripts. Cards were printed on PVC and captured with an iPhone 15 Pro, a Huawei Mate 30 and a Kyocera office scanner (about 3,000 x 1,900 px from the phones, 2,000 x 1,250 from the scanner). Inspected on a local copy on 10 October 2026 (`~/Downloads/Datasets/FantasyID`, 938 MB, incomplete: only `test/attack/digital_3`, `test/attack/facedancer` and `examples` are present; the 300 test bona fide images, Attack-3, and the whole train-val set are missing, so the full 2.6 GB Zenodo archive is still needed).

Structure of the test set (paper Section 3.2 and the local files):

| Part | Designs | Images | Manipulation |
|---|---|---|---|
| Attack-1 (`digital_3`) | the 10 train-val designs: Arabic, Chinese, Hindi, French, Persian, Portuguese, Russian, Turkish, Ukrainian, English (USA) | 786 (262 cards x 3 devices) | text fields rewritten with fine-tuned TextDiffuser2, SAM-extracted and alpha-blended; always 3 fields (dob, doe, a name field) |
| Attack-2 (`facedancer`) | 3 designs unseen in train-val: Chinese (second design), Dutch, Singaporean | 137 locally (paper says 150) | face swap with FaceDancer |
| Attack-3 | the same 3 unseen designs | 149 | text rewrite as in Attack-1 |
| Test bona fide | the 3 unseen designs | 300 (100 cards x 3 devices) | none |

Annotations: one JSON per image in VIA style with every field rectangle, its value, language and `region_provenance` in {original, altered}. Face swaps mark the `face` rectangle as altered (two per image, the portrait and the ghost image). So pixel masks are derivable for every attack, and the zone mask for every design comes from the original rectangles. `cropping_info` gives the document quadrilateral and the homography used to crop, so rectification is provided.

Consequences for the protocols. (1) Our model never trains on FantasyID, so every FantasyID design, Arabic included, is unseen in Protocol B. (2) The Arabic design is one of FantasyID's ten train-val designs, so it appears in the test set only inside Attack-1 (text rewrites) and in the train-val face swaps, not in the official out-of-domain test. The headline Protocol B number is on the official test (Attack-2 and Attack-3 plus the 300 bona fide on the three held-out designs), to match the published tables; Attack-1 is a second result, with the Arabic cards broken out. The published zero-shot baselines (TruFor, MMFusion) had not seen Arabic either, but they report Attack-1 as one number, so the Arabic row compares only with our own TruFor rerun on that subset. (3) Pixel F1 is available on Protocol B.

### IDNet

The arXiv paper (2408.01690 v2) and the BigData 2024 paper both describe 837,060 images of 20 designs (10 US states, 10 European countries), with six fraud types per design. The Zenodo release (eight records, CC0) ships one zip per design. Inside `SD.zip`, read directly from Zenodo without downloading: seven folders of 5,979 images each (`positive`, `fraud1_copy_and_move`, `fraud2_face_morphing`, `fraud3_face_replacement`, `fraud4_combined`, `fraud5_inpaint_and_rewrite`, `fraud6_crop_and_replace`), plus per-image field annotations with bounding boxes, original and new values, fonts and sizes.

The published storage sizes are not the Zenodo sizes. Table 1 of the paper lists South Dakota at 32 GB; the Zenodo zip is 4.9 GB. The Zenodo images are 795 x 487 JPEG files of about 118 KB (the `.png` names are JPEG data). The release is a reduced-resolution, recompressed version of what the paper describes. Counts match (5,979 per class). The sample images carry no visible "For Research Purposes" overlay.

Annotation mapping, verified by file-name overlap: `fraud1_copy_and_move`, `fraud2_face_morphing`, `fraud3_face_replacement` and `fraud4_combined` reuse the original file names and are described by `*_textfieldreplacement_annotation.json`, `*_facemorph_annotation.json`, `*_facereplacement_annotation.json` and `*_combined_fraud_annotation.json`. The paper's "Text Replacement" task is therefore `fraud1_copy_and_move`. `fraud5` and `fraud6` have their own `_fake_` file names and their own JSON files. To confirm visually on the first download.

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
| A: unseen design, same source | SIDTD, 6 nationalities | SIDTD, 4 held-out nationalities | SIDTD published few-shot rows |
| B: unseen source, generator, capture, script | SIDTD (all 10 designs) plus IDNet AZ and WV (face attacks) | FantasyID official test; Arabic design alone | FantasyID published TruFor and MMFusion rows |
| C: unseen design, published pair | IDNet WV (all four task folders as positives) | IDNet AZ, per task; and the reverse | arXiv Table 14 |

Reference for every protocol: the pixel-wise median over genuine documents of the test design, built from genuine images that are not test samples. For Protocol C the reference uses the 4,979 non-test genuine images of the target design and nothing else from it; the detector never trains on the target design.

Difference from the published Protocol C setup, to state in the paper: the paper trained one detector per task; we train one detector on all four tasks and report per-task accuracy from the same model, at threshold 0.5, plus AUC.

## 5. Limitations to state

1. All training forgeries come from two generators (SIDTD's and IDNet's); FantasyID is the only test beyond them.
2. SIDTD and IDNet are digital renders; capture is tested only by FantasyID.
3. IDNet results are on the reduced Zenodo release.
4. Training designs are European and US; the Arabic design appears only at test.
5. The Centre's 902 irregularities in 20,307 examinations is a ratio among examined documents, not a prevalence among travellers; workload projections use assumed prevalence scenarios.
