# Design package: Refer or Clear (reference-conditioned forgery triage)

Written 8 October 2026 for Hamza and Prof. AlHajri; revised the same day: the model runs on the counter's computer fed by the existing document scanner, so there is no hardware target and no quantization. The paper follows the structure of the group's Muharaf writer-identification paper (IJDAR 2026): a substantial data contribution, the first benchmark on public data under two protocols with the gap quantified, a proposed architecture built on a published framework with named modifications including attention, and a configuration grid with ablations over backbones and transfer-learning regimes at three seeds; and the contribution reframed as the public measurement an operator cannot get from its vendor. Grounded in the evidence files in `exa-results/` dated 2026-10-07 and 2026-10-08. Nothing has been downloaded or trained yet. This is the design to approve before the proposal text is revised and before any code is written.

---

## 1. What the paper claims, in one paragraph

Published identity-document forgery detectors are single-image classifiers. They score 96 to 100 percent on document designs they were trained on and fall to 35 to 62 percent, chance or worse, on designs they were not (IDNet, SIDTD, FantasyID, DeepID 2025). Deployed border systems (Regula, Keesing, Smart Engines, Thales) do the opposite: they compare the presented document against a stored specimen of the same design, by rules written per design, with no learned generalization and no published accuracy. We build the learned version of what the industry already does: a two-input network that receives the presented data page and a clean reference of the same design, is trained to score the deviation between them on a subset of designs, and is tested on designs, datasets and scripts it has never seen. We report its output as a first-line referral decision (referral rate at a fixed miss budget, against GDRFA's published 4.4 percent hit rate among referrals). GDRFA already operates a commercial reader with a template library of more than 80 countries; what neither it nor any operator has is a number for how such a system behaves on a design outside its library, because no vendor publishes one. The contribution is therefore the public measurement and the open baseline: the formulation, the leave-design-out protocol on public data, the model and code an operator can run against its own template library, and the evidence. The architecture components are known (Siamese difference networks from change detection; registration against a specimen).

---

## 2. State of the art, plainly

### 2.1 Published single-image detectors (what they are, what they learn)

| Family | Representative | What it learns | Cross-design result |
|---|---|---|---|
| Natural-image forensics applied to IDs | TruFor (CVPR 2023, 68.7M params), MMFusion, Cue-Net/Re-MTKD (DeepID 2025 winner) | camera-noise residual inconsistency, JPEG traces, pasted-region edges | FantasyID: AUC 99 on text edits, 56 to 62 on face swaps; DeepID: F1 0.99 on fantasy cards, 0.72 on 20K real IDs; near chance on small Poisson-blended text edits |
| Dataset-paper CNN baselines | SIDTD EfficientNet-B3 / ResNet50; IDNet EfficientNet/ResNet | template-specific appearance | SIDTD held-out nationalities: 82 to 91 (vs 97 to 100 seen); IDNet cross-template: 35 to 62 |
| Patch and field detectors | FakeIDet2 (Inf. Fusion 2026, DINOv2 patches), FLiD (MobileNetV3 field crops) | local texture on patches; face/text field cues | FakeIDet2 is PAD (print/screen), 9.6% EER cross-dataset; FLiD single dataset, folds overlap |
| Few-shot / prototype | Talarmain (ICDAR 2024), Nieto-Hidalgo (DAS 2026) | prototypes from a few target-design samples at inference | 9% EER on unseen country only with 4 target bona fide shots at inference |

Why they fail across designs: the cue is tied to the data-generating process. Noise and compression cues need the same capture chain; font and layout cues are template-specific; synthetic sets teach the generator's signature (the DeepID winner explicitly exploited a JPEG quality difference between genuine and forged files).

### 2.2 Industry (what border readers actually do)

Every documented product is reference-based: a template database (Regula 16,500 templates; Keesing; Smart Engines 4,715; Incode 4,900 types), pattern position/shape/colour checks against the stored specimen under white, UV and IR, MRZ and chip consistency, plus undisclosed learned models. No vendor publishes attack-stratified error rates. Frontex's own officer-plus-machine exercises concluded no system authenticates all features. The only externally comparable vendor number is Incode's DeepID rank 2, behind an academic model trained on public data.

### 2.3 Learned reference-based methods (the three that exist)

| Work | Reference input | Task | Data | Unseen designs | Result |
|---|---|---|---|---|---|
| CheckSim, ICDAR-W 2021 | predefined reference patch of the document model | alteration, patch level | private (4 French types) | no | numbers paywalled |
| Chen et al., TIFS 2022 | genuine reference of same template as triplet anchor | recapture (PAD), not alteration | private | cross-document-type: 6.9% APCER at 5% BPCER | reference cancels template content |
| Al-Ghadi 2022 / FMIDV 2023 | random genuine of same country | guilloche copy-move | MIDV-2020 + FMIDV | no | with reference TAR 0.92+; follow-up REMOVED the reference at test and fell to AUC ~0.5 |
| Neitthoffer, ICDAR-W 2023 | template image via attention | OCR on unseen ID types | MIDV-2020 leave-type-out | yes | unseen CER 0.31 with attention fusion, 0.81 with channel stacking |

Two facts from this table drive the design. The FMIDV negative result is published evidence that the reference is what carries the generalization. The Neitthoffer ablation says the reference must be fused as a difference, not stacked as extra channels.

### 2.4 The exact novelty claim (and what is not claimed)

Claimed, in the four parts the Muharaf paper has:
1. **Data.** The first paired, design-referenced forgery dataset built from public sources: design-median references and zone masks for 23 designs, hand-annotated field layouts for FantasyID's 13 designs, about 5,000 generated forgeries with exact masks, identity-grouped splits. Released.
2. **Benchmark.** The first evaluation of alteration detection under two protocols, unseen design within dataset (SIDTD nationality hold-out) and unseen design across dataset, generator, capture and script (FantasyID), with the gap between them quantified, for single-image and reference-conditioned detectors on identical samples.
3. **Architecture.** A proposed reference-conditioned architecture built on a published change-detection framework with named modifications: zone-conditioned fusion, one-sided query concatenation, a cross-attention fusion variant in which the query features attend over the reference features (the "improved" variant, parallel to the attention path in the Muharaf model), a global referral head, and a loss over both score and mask.
4. **Ablations.** A configuration grid over backbone (four), fusion (three), reference kind (two), and transfer regime (three), three seeds each, plus the referral-rate analysis.
Not claimed: a new network primitive. Siamese difference networks, registration against a specimen, attention fusion, and two-head outputs all exist; the published base framework is ChangerEx in Open-CD, as Deep-TEN via Chammas et al. was the base in the Muharaf paper.

---

## 3. Dataset design

### 3.1 Datasets in scope (all one-click, all openly licensed)

| Dataset | Role | Size to download | Why |
|---|---|---|---|
| MIDV-2020 (ftp://smartengines.com/midv-2020/dataset/, CC-BY-SA 2.5) | bona fide designs; field rectangles; capture variants | templates.tar 0.86 GB; photo.tar 4.0 GB; scan_upright 1.15 GB (clips optional) | 10 designs x 100 identities; VIA JSON with photo, face, signature and every text field rectangle; scans and phone photos of the same documents with doc_quad for rectification |
| SIDTD (http://datasets.cvc.uab.es/SIDTD/, CC-BY-SA) | forged versions of MIDV-2020 templates; official splits; generator code | templates.zip 1.27 GB; split zips | 1,222 forged templates (crop-and-replace, inpaint-and-rewrite) each naming its clean `src` and modified `field`; nationality hold-out split (6 train / 4 test) for direct comparison with published numbers; generator released (`fake_id_ex.py`) |
| FantasyID (Zenodo 17063366, CC-BY-4.0) | unseen-design, unseen-generator, printed-and-photographed test | 2.6 GB | 13 fictional designs in 10 scripts incl. Arabic; cards printed on PVC and captured on two phones and a scanner; face swaps (InSwapper, FaceDancer) and text edits (DiffSTE, TextDiffuser2); official test split with 3 designs absent from train |
| IDNet-2025 (HF cactuslab/IDNet-2025, CC-BY-4.0), optional | extra alteration types and volume for the same 10 EU designs | 2 to 4 country archives (EST 1.35, SVK 2.5, FIN 2.6, ESP 2.9 GB) | only if day 1 download succeeds; fully synthetic, so it is a capture-domain test, not a design test |

Out of scope for the 8-day paper (future work): MIDV-Holo and DLC-2021 (physical copies and recaptures; a different task, PAD).

### 3.2 Label harmonization

One record per sample: `image`, `design_id` (MIDV-2020 code: alb_id, aze_passport, esp_id, est_id, fin_id, grc_passport, lva_passport, rus_internalpassport, srb_passport, svk_id; FantasyID: its 13 template names; IDNet: `idnet:<LOC>` linked to the MIDV design family), `identity_id` (MIDV `<code>/<NN>`; SIDTD via `src`; FantasyID card id, three captures kept together), `label` (0 genuine, 1 altered), `fraud_type` (controlled vocabulary: crop_replace, inpaint_rewrite, face_swap, text_inpaint, plus generator name), `capture` (render, scan, phone, device), `mask` (altered-region mask where derivable, else null), `fields` (rectangles for photo, face, signature, text fields: native for MIDV/SIDTD; hand-annotated once per design for FantasyID's 13 layouts, about one hour of work, no new data collection).

### 3.3 Reference construction (the design decision that matters most)

The reference must be what an officer could have: the design, not the person. Using the same identity's clean counterpart would reduce the task to a pixel diff and would not exist at a border. Two reference kinds, both produced from bona fide documents of the same design:

- **Design median (deployment reference).** Pixel-wise median over all bona fide templates of design d after rectification. Personal content varies across identities and is removed by the median; static design content (guilloche, headings, field labels, borders) survives. This approximates the blank specimen a template library holds. For FantasyID, the median over the bona fide cards of each design (about 26 train cards per design; for the 3 test designs, the 100 test bona fide cards, used only as reference, never as test samples of the same identity as the query).
- **Masked exemplar (robustness reference).** One randomly chosen bona fide document of design d, identity j not equal to the query's identity, with personal-data zones (field rectangles) filled with the design-median values.

A third input channel, the **zone mask** (1 inside personal-data fields, 0 elsewhere), tells the network where content is expected to differ. Inside zones the network must find inconsistency with the design (font, baseline, colour, guilloche continuity across the field edge); outside zones any difference is suspicious. Without the zone mask, every genuine pair would show large differences in the fields and the network would have to learn the mask implicitly; the ablation measures this.

### 3.4 Pairs and labels

- Positive pair: (altered document, reference of the same design) with the derived mask of the altered field (SIDTD `field` rectangle on `src`, dilated by 8 px for the unrecorded shift; FantasyID masks if shipped, else the face box or the annotated field).
- Negative pair: (bona fide document, reference of the same design, different identity). Negatives include scans and phone photos of the bona fide documents rectified to the template canvas (MIDV-2020 photo and scan sets), so capture-domain difference is learned as "not an alteration".
- Hard negatives: bona fide with a wrong-design reference, labelled as a separate "design mismatch" outcome (see 4.4).
- Extra positives: our own forgeries generated with the released SIDTD generator on MIDV-2020 templates (the generator records the field, so masks are exact), to raise positives from 1,222 to about 5,000 and to cover every identity.

### 3.5 Splits (the protocol)

| Level | Train | Test | Compare against |
|---|---|---|---|
| L1: unseen design, same dataset | SIDTD/MIDV-2020, 6 nationalities (the official few-shot split's meta-train set) | the 4 held-out nationalities | SIDTD published few-shot: EfficientNet 82.0 / AUC 0.899, ResNet50 90.6 / 0.963, CoAARC 62.8 |
| L2: unseen dataset, generator and capture | all 10 MIDV designs (SIDTD + own generated + MIDV photos/scans) | FantasyID, full official test split (300 bona fide, 1,085 altered) | FantasyID published: TruFor AUC 93.5 all / 55.8 face, FNR ~50 to 62 at 10% FPR; MMFusion HTER 25.8; our own TruFor zero-shot run |
| L3: unseen script | same as L2 | FantasyID Arabic design only, reported separately, with 0 and 10 calibration cards | none published |
| L2b (optional) | add IDNet EU archives to L2 training | FantasyID | does volume help |

All splits are identity-grouped. FantasyID's three captures of a card stay together. L1 uses SIDTD's own `split_shot` CSV so the number is directly comparable.

---

## 4. Architecture

### 4.1 Base framework and the two proposed variants

Base framework: **ChangerEx** (Fang, Li, Li, IEEE TGRS 2023) from the Open-CD toolbox (Apache-2.0), a Siamese change-detection network with feature exchange inside the backbone and Flow Dual-Alignment Fusion, which tolerates residual misregistration. Pretrained LEVIR-CD and S2Looking checkpoints at 512 by 512. This plays the role Chammas et al.'s Deep-TEN framework played in the Muharaf paper.

Two proposed variants, both end to end, sharing every tensor size:

**Standard variant (difference fusion).** Shared encoder on query Q and reference R; at each scale the absolute difference |f(Q) - f(R)| is concatenated with f(Q) (one-sided artifacts live only in the query) and with the downsampled zone mask Z; the ChangerEx decoder emits the alteration mask M; a new global head emits the score s.

**Improved variant (attention fusion).** Three attention blocks inserted without changing tensor sizes, mirroring the Muharaf design: (1) a self-attention block over the query's deepest feature map so distant regions of the page attend to each other (a replaced field and its guilloche surroundings); (2) a **cross-attention block in which the query features are the queries and the reference features are the keys and values**, so each query location reads what the design should contain at that location, which is the learned form of "compare against the specimen" and the fusion the ICDAR 2023 unseen-ID-type study found necessary where channel stacking failed; (3) a cross-attention block from the pooled global descriptor over the fused local features before the score head. Z enters as an additive bias on the cross-attention logits, so attention inside personal-data zones is discounted. Six heads, key dimension 32, as in the Muharaf blocks.

Named modifications to the base framework, in the paper's own words: zone-conditioned fusion; one-sided query concatenation; cross-attention reference fusion; global referral head (global average pool of the deepest fused features concatenated with max(M), two dense layers, one logit); joint loss BCE(s) + 3 (BCE + Dice)(M); design-median reference construction; wrong-design training pairs.

Why ChangerEx and not a forensics network: TruFor and Cue-Net are single-image and are the baselines. Why not FC-Siam-diff: it was chosen earlier only for a microcontroller operator set, which no longer applies, and it is the weakest of the family.

### 4.2 Inputs and outputs

- Inputs: query Q (3 x H x W), reference R (3 x H x W), zone mask Z (1 x H x W), H x W = 512 x 800 (ID-1 aspect), passports cropped to the data page on the same canvas; both rectified to the design canvas.
- Outputs: alteration mask M (sigmoid) and one score s. Binary, genuine vs altered. Fraud type is reported, not predicted.

### 4.3 Configuration grid (the fourteen-configurations analogue)

| Axis | Values |
|---|---|
| Backbone (interchangeable feature extractor) | ResNet-18, ResNet-50, ResNeSt-50, MiT-B1 (SegFormer); all available in Open-CD |
| Fusion | plain single-image (Q only, same encoder and heads), channel-stacked early fusion (7 channels), difference fusion (standard), attention fusion (improved) |
| Reference kind | design median, masked exemplar |
| Transfer regime | frozen backbone, partial fine-tuning (last two stages), full fine-tuning, from scratch; initialization from LEVIR change-detection weights vs ImageNet |
| Seeds | 3 |

Main table: four backbones x four fusion settings on the design-median reference with full fine-tuning (16 configurations, three seeds). Transfer-regime and reference-kind ablations on the best backbone only. Published anchors alongside: SIDTD's EfficientNet-B3 and ResNet50 few-shot numbers on Level 1; TruFor zero-shot (public weights) on Level 2, with our own run of TruFor to validate the pipeline against the printed AUC of 93.5.

### 4.4 Registration and the wrong-reference case

- Rectification: document quadrilateral from MIDV/SIDTD annotations where present; for FantasyID photos, quad detection (contour plus Hough) then ORB/SIFT features against the design median with RANSAC homography; ECC refinement. Residual registration error kept as a scalar and reported.
- Training augmentation: random affine of plus or minus 8 px and plus or minus 1.5 degrees between Q and R; photometric jitter; JPEG quality 60 to 100 on both inputs independently so compression cannot be the cue (the DeepID lesson).
- Wrong reference: design identification before the network (nearest design median by normalized cross-correlation on static zones, inlier-ratio threshold). A mismatch returns "unknown design", which at a border means refer. Training includes wrong-design pairs labelled altered so the network fails safe; the experiment reports the gate's accuracy and the network's behaviour without it.

### 4.5 Loss and training recipe

- Loss = BCE(s) + 3 x (BCE + Dice)(M).
- AdamW, lr 1e-4 backbone and 1e-3 new blocks, cosine schedule, batch 8 at 512 x 800, 40 epochs, mixed precision; learning-rate scheduler and early stopping on identity-disjoint, design-seen validation AUC, three seeds, mean and standard deviation reported, as in the Muharaf tables.
- Metrics: image-level AUC, EER, APCER at BPCER 1 and 5 percent (ISO/IEC 30107-3 vocabulary); pixel F1 and IoU for the mask at a fixed threshold; the referral curve.

### 4.6 Referral framing

Score s, threshold t chosen on a calibration split with no test design (for L3, with 0 or 10 cards of the Arabic design). Report referral rate R(t) at miss rate M(t) of 1 and 5 percent. Prevalence p for the x-axis is GDRFA's published 4.4 percent among referrals, stated as a referral-set prior, not a population one.

## 5. Experiments and what each one proves

| # | Experiment | Proves |
|---|---|---|
| E1 | Protocol A (Level 1): main grid, 4 backbones x 4 fusion settings, SIDTD nationality hold-out, three seeds | whether the reference recovers the cross-design loss, and which fusion carries it; direct comparison with SIDTD's published 62.8 to 91.0 |
| E2 | Protocol B (Level 2): the same grid trained on all 10 MIDV designs, tested on FantasyID's official test split; TruFor zero-shot alongside | transfer across dataset, generator, capture and script; comparison with the FantasyID baselines; the Protocol A to B gap quantified, as line-level vs page-disjoint was |
| E3 | Level 3: Arabic design, 0 and 10 calibration cards | the Dubai case; calibration cost |
| E4 | Transfer-regime ablation on the best backbone: frozen, partial, full, scratch; LEVIR vs ImageNet init | the Muharaf transfer-learning analysis, repeated here |
| E5 | Reference and zone ablations: design median vs masked exemplar; zone mask on/off; difference-only vs difference-plus-query; attention bias on/off | which part of the method carries the gain |
| E6 | Robustness: wrong-design reference with and without the gate; residual misregistration sweep 0 to 12 px | failure modes and the fail-safe |
| E7 | Referral curves on Levels 1 to 3; referral rate at miss 1 and 5 percent; per-fraud-type breakdown | the operator metric |
| E8 (optional) | Level 0: fine-tune on FantasyID train, score on its test with F1 at 0.5 | the one column comparable to the public half of the DeepID leaderboard |
| E9 (optional) | IDNet EU archives added to Protocol B training | volume effect |

Minimum publishable result (if time runs short): E1 and E2 with two backbones (ResNet-18, ResNet-50) and all four fusion settings, plus E7.

## 6. Risks, stated

- The reference may not help enough: Al-Ghadi's with-reference results were good but on an easy task; CoAARC, which pairs arbitrary images, was the worst SIDTD baseline. Mitigation: the zone mask and design median are the differences from those attempts; E4 isolates them; a negative result on L2 with a positive on L1 is still a publishable finding with the referral framing.
- Public forgeries are composites and generative edits, not professional counterfeits (the reality gap named by the 2025 and 2026 surveys). The paper says what a leave-design-out number measures.
- FantasyID masks and layout files may not ship; mitigation: hand-annotate 13 layouts; masks from face box and annotated text fields.
- Registration on phone photos may fail on some cards; mitigation: report the failure rate; failures count as referrals.
- IDNet download is large and its releases are inconsistent; it is optional.
- The donor's own licence is permissive (Open-CD, Apache-2.0); MIDV-2020 and SIDTD are CC-BY-SA, so released derived data must be share-alike; FantasyID CC-BY-4.0.
- Overfitting: about 6,000 positive pairs against an 11M-parameter backbone; mitigation: pretrained initialization, strong augmentation, early stopping, and the BIT and TinyCD fallbacks if the validation gap says so.

---

## 7. Day-by-day plan (8 to 15 October)

| Day | Work | Output |
|---|---|---|
| 8 | Start downloads (MIDV-2020 templates/photo/scan, SIDTD templates and splits, FantasyID; IDNet EU archives in background). Read SIDTD JSON and FantasyID tgz layout. Build the unified loader and the design-median references for the 10 MIDV designs. | loader; references; first pair batch visualized |
| 9 | Rectification pipeline; FantasyID layout annotation (13 designs); zone masks; own forgeries with the SIDTD generator; identity-grouped splits written to CSV. | dataset v1 with counts table |
| 10 | Set up Open-CD; implement the standard variant (zone channel, query concat, global head) and the improved variant (three attention blocks); plain and stacked settings; launch E1 grid on the cluster (Slurm array over backbone x fusion x seed). | E1 table |
| 11 | Protocol B grid (E2) and E3; TruFor zero-shot on FantasyID validated against the printed AUC. | E2, E3 tables |
| 12 | E4 transfer regimes, E5 reference and zone ablations, E6 robustness; referral curves E7. | ablation tables; referral figure |
| 13 | Optional E8 and E9; otherwise buffer for reruns and figures from real outputs (deviation maps on FantasyID, the Protocol A to B gap chart). | optional tables |
| 14 | Write results into the eleven-section template; figures from real outputs replace the illustrative ones; internal review. | draft |
| 15 | Final checks, Word and PDF, Arabic abstract, submit. | submission |

Compute: one GPU node on the cluster suffices; each training run is a few hours at 512 x 800 with ResNet-18.

---

## 8. What changes in the proposal document once this is approved

- Title and framing: drop "quantized for the handheld"; the subtitle becomes "reference-conditioned forgery detection for unseen document designs, a public measurement for the first line". Section 1 gains a paragraph on what GDRFA already operates (Retro Check readers, remote examination devices, the 80-country template system) and the point that no vendor publishes a cross-design error rate; the contribution becomes the public measurement and open baseline.
- All quantization, device and latency content is removed (RQ3, the quantization paragraph, the device row of Table 1, the hardware figure); the group's prior work is cited for the cross-dataset protocol only.
- Scope: MIDV-Holo and DLC-2021 move from "RQ4 control" to future work; IDNet becomes optional.
- The reference is defined as the design median with a zone mask, not a per-person clean counterpart, and the paper says why.
- Table 1 adds Al-Ghadi/FMIDV with the negative result, and CoAARC as the pairing-without-reference baseline.
- Models section is rewritten as "Proposed architecture": base framework ChangerEx, standard variant (difference fusion) and improved variant (attention fusion), the named modifications, both heads, the configuration grid, with TruFor and the SIDTD baselines as published anchors.
- Experiments table becomes E1 to E8 as above.
- Figure 2 row C gains the zone mask and the design-median construction.
