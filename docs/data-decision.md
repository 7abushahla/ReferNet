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

Thirteen fictional designs in ten scripts, including Arabic. Cards were printed on PVC and captured with two phones and a scanner. Manipulations are face swaps (InSwapper, FaceDancer) and text edits (DiffSTE, TextDiffuser2). The official test split holds 300 genuine and 1,085 manipulated images on designs absent from training. Published baselines: TruFor AUC 93.5 overall, 99.5 on text, 55.8 on faces; MMFusion AUC 94.4 overall, 61.5 on faces. The validation set with real faces is restricted and is not used.

Consequences. FantasyID is the only test of capture conditions and of a forgery process unlike the training generator. The face-swap results are reported as a zero-shot attack type.

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
5. The 4.4 percent prior in the referral analysis is among documents already referred to the Document Examination Centre, not among all travellers.
