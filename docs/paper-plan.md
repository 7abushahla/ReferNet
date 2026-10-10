---
title: Refer or Clear Paper Plan, Reference-Conditioned Detection of Altered Travel Documents on Unseen Designs
date: 2026-10-10
updated: 2026-10-10
status: Scope, datasets, protocols and model fixed on 2026-10-10 after the data inspection and the external dossier review. No training run yet. The claim chain below is the contract for the manuscript and the repository.
aliases:
  - Refer or Clear paper plan
  - GDRFA award paper plan
tags:
  - publication
  - document-forensics
  - gdrfa
related:
  - "[[02-Notes/general/2026-10-07 01 First-Line Forgery Triage on the Officer's Device, GDRFA Award Proposal/STUDY-BRIEF]]"
  - "[[02-Notes/general/2026-10-07 01 First-Line Forgery Triage on the Officer's Device, GDRFA Award Proposal/DATA-DECISION]]"
  - "[[02-Notes/general/2026-10-07 01 First-Line Forgery Triage on the Officer's Device, GDRFA Award Proposal/DOSSIER-REVIEW]]"
  - "[[02-Notes/general/2026-10-07 01 First-Line Forgery Triage on the Officer's Device, GDRFA Award Proposal/DESIGN]]"
---
# Refer or Clear Paper Plan, Reference-Conditioned Detection of Altered Travel Documents on Unseen Designs

A paper for the GDRFA Dubai Scientific Research Award, second cycle, Digital Innovation Award, Artificial Intelligence theme. The manuscript lives in the proposal folder under `submission/main.tex` (LuaLaTeX, the award's eleven-section structure). Code, data tooling and results live in the private repository `7abushahla/refer-or-clear`. This plan is the contract: the claim chain in Section 2 fixes what every experiment must show and the order in which the paper says it.

## Contents

- [[#1. Venue facts]]
- [[#2. The claim, one chain]]
- [[#3. The four protocols]]
- [[#4. Paper structure and word budget]]
- [[#5. Experimental protocol]]
- [[#6. Training and timing recipe]]
- [[#7. What the paper may and may not claim]]
- [[#8. Milestones and gates]]
- [[#9. Risks]]
- [[#10. Open items]]
- [[#11. Decision log]]
- [[#12. References]]

## 1. Venue facts

| Item | Value | Source |
|---|---|---|
| Award | GDRFA Dubai Scientific Research Award, second cycle; Digital Innovation Award; theme Artificial Intelligence, smart innovations and advanced data analytics | official guide |
| Deadline | 15 October 2026; internal review 14 October | official guide |
| Length | 4,000 to 8,000 words including tables, figures, appendices and references | official guide |
| Files | MS Word and PDF | official guide |
| Front matter | title in Arabic and English; names, workplace, job title, degree, email, phone; Arabic and English abstracts of at most 250 words each | official guide |
| Structure | title page, Arabic abstract, English abstract, introduction, research problem and objectives, framework and literature, methodology (population, samples, tools), results and discussion, conclusions and recommendations, references, appendices | official guide |
| Eligibility | original, unpublished, not submitted to another award this cycle; one entry per team; the award screens for originality, so the repository stays private until the decision | official guide |
| Criteria | eight: scientific value; originality and novelty; theoretical framework; language and presentation; methodology; integrity and citation; data analysis and results (recommendations linked to outputs); applied and innovative impact (translation into practical solutions and policies) | official guide |
| Submission | PDF uploaded through the researcher control panel plus the Word file; electronic nomination form and declarations; automated formal check, then the committee | official guide |
| Reader | GDRFA evaluators, not a computer-vision program committee; the problem must be theirs, the recommendations must follow from results, the deployment story must be concrete | the criteria above |

## 2. The claim, one chain

The paper is a deployment story told as a measurement. A detector is only useful to GDRFA if it answers the first-line question, on the first line's hardware, inside the first line's time, on designs the first line has never trained on. Every item below carries that thread. Each item is a statement the paper makes, the evidence that carries it, and the place it appears; the order is the order of exposition.

**The setting, and the deployment constraints it imposes.**

1. Dubai's border checks travel documents in two lines. About 1,600 first-line officers see every document for seconds, at a counter, with a document reader connected to a computer, and refer anything doubtful to the Document Examination Centre of about 50 experts. The Centre reported 902 irregularities in 20,307 examinations between January 2025 and June 2026. The ratio is among examined documents, not a prevalence among travellers, and it says nothing about documents never referred. Evidence: the Gulf News report, ICAO Doc 9303's inspection levels, Frontex's quick-check cards ("only seconds"). Place: introduction. This item fixes three deployment constraints the rest of the paper must respect: the decision is binary (refer or clear), it is made in seconds, and it is made on the counter computer with the scanner as the only sensor.
2. The first line already operates readers with a template library of specimens; a new design is handled by adding its specimen, not by retraining anything. That operating model is the one a learned detector must fit: no new data collection per design, no retraining per design, a specimen per design. What no vendor and no paper publishes is the error rate on a design outside the library or outside the training set, the number an operator would need before trusting a clear. Evidence: vendor documentation (Regula, Keesing, Smart Engines); the absence of any published cross-design figure. Place: introduction and problem statement, phrased as "we found no published", never "no vendor can".

**The gap, stated exactly.**

3. The failure is measured, not hypothesized. Four public benchmarks already evaluate on designs absent from training, and every detector they evaluate collapses there: SIDTD's few-shot split (82 to 91 percent accuracy, from 97 to 100 within design), IDNet's cross-type tables (35 to 71 percent), FantasyID's held-out designs (AUC 56 to 62 on faces; 62 percent of genuine cards referred to miss 5 percent of attacks), DeepID's private real IDs (F1 0.99 to 0.72), and the 2026 survey's zero-shot set (attack error above 24 percent for every public model, GPT-5 and Gemini included). So the gap is not that nobody tests on unseen designs. They test, and they fail. Evidence: those papers' tables. Place: literature.
4. The detectors in those benchmarks cannot use the specimen the counter already holds. All but one take a single image and nothing else (the five SIDTD baselines less one, IDNet's baselines, FantasyID's four, all 26 DeepID entries, DocForge-Bench's 14, the survey's set). The one exception, SIDTD's CoAARC, pairs the query with a randomly drawn genuine document of any nationality, resized to a square, compared by recurrent glimpses, and it scores worst of the five (62.8 percent, AUC 0.675). Pairing with an arbitrary genuine image does not help. No public benchmark has ever seen a registered specimen of the query's own design as an input. Evidence: the SIDTD paper and its released batcher code; the other benchmark papers. Place: literature.
5. Reference-conditioned learned detectors exist outside the benchmarks, and none answers the question. CheckSim: patch similarity to an aligned reference, four French document types, private data, within-type thresholds. Chen et al.: a same-template reference, but for recapture, not alteration. Al-Ghadi et al. and FMIDV: a random genuine of the same country, guilloche copy-move only, one dataset; the follow-up that dropped the reference fell to AUC 0.5. Neitthoffer et al.: the template through attention, within design. Nieto-Hidalgo, Espin and Tapia 2026: four target bona fide samples as prototypes, presentation attacks, private data, no localization, error on unseen countries from 26.6 to 9.1 percent. Commercial readers compare against a template library by rules with no published error rate. Evidence: those papers and vendor pages. Place: literature.

**The gap in one sentence, for the abstract and the problem statement.** The collapse of learned alteration detectors on unseen designs is measured in four public benchmarks; the remedy the counter already uses, a specimen of the same design, has never been given to a learned alteration detector and measured on those same benchmarks against the same detector without it. That measurement is the contribution. The deployment framing, the referral decision at a miss budget and the time on a counter CPU, is what makes the measurement usable to GDRFA; it is not the gap itself.

**The idea, shaped by the operating model.**

6. Make the specimen an input. The reference is the design median, the pixel-wise median over genuine documents of the design, which removes personal content and keeps the printed design, with a zone mask marking where personal data may differ. The detector looks for inconsistency with the design, not for the signature of a training generator. Deploying it is what the counter does today: enrol a specimen per design, look it up by the design the reader already identifies. Evidence: the idea figure and the reference figure. Place: framework.
7. The architecture is a published change-detection network, ChangerEx from Open-CD, with a pixel head and a referral head, chosen because it is small enough for a CPU and because change detection is the exact shape of the task (two aligned images, where do they differ). The architecture is not new and the paper says so; the adaptation is the design-median reference, the zone channel, the wrong-design gate and the referral head. Place: methodology.

**The controlled measurement.**

8. The reference is the only difference between method and baseline. The single-image control uses the same code, the same zone mask and the same training files with the reference input removed; it gets no wrong-design pairs, which only make sense with a reference. In Protocol A the control must land on the published single-image rows (the gate), which validates our data, split, scoring and network against prior work. Place: methodology and every results table.
9. Protocol A, the anchor: the SIDTD paper's template 10-fold protocol used as given (raw files, the official folds, accuracy and ROC AUC over folds), every nationality seen in training. Our control is compared with the paper's five Table 5 composite-detection rows; the reference arms are compared with the validated control. Place: results, first table.
10. Protocol B, unseen design within one source: the paper's ten held-out nationality groups, same recipe as A, mean and sd over groups and per nationality. No published number exists on templates without labelled forgeries; the comparison is the control validated in A. The paper's few-shot rows are on cropped clips with labelled forgeries of the test nationality and stay out of our tables (the clips carry a capture confound, docs/data-decision.md). Place: results.
11. Protocol C, unseen source, generator, capture and script: train on SIDTD plus IDNet's face attacks, test zero-shot on FantasyID's printed and camera-captured cards. The headline is the official held-out designs with face swaps and text rewrites; Attack-1 on the ten other designs is a second result with the Arabic cards broken out. Our model never trains on FantasyID, so all thirteen designs are unseen to it. Place: results.
12. Protocol D, the published pair: train on IDNet West Virginia with all four attack folders, test on Arizona per task, and the reverse, against IDNet's Table 14. This reruns the failure cited in item 4 with the reference added. Place: results, last priority.

**From a score to the officer's decision.**

12. A score becomes a decision through a referral threshold. Calibrate it on the training source at a 5 percent miss target, freeze it, and report the realized miss and referral rates on the unseen designs: that is what an operator would see after enrolling a new design without retraining. Project workload under assumed prevalence scenarios (0.1, 1 and 5 percent), never from the Centre's ratio. Evidence: the referral curves from the saved scores. Place: results and recommendations.
13. The decision must arrive inside the seconds the officer has. Measure inference time and throughput on a CPU, batch size one, FP32, with and without cached reference features (the library lookup), and report them beside the accuracy: pre-processing, network, end to end, FPS, parameters, file size, on a cluster CPU node and a counter-class computer. Target under one second end to end. Quantization and execution on the scanner itself are future work. No paper in this literature reports a measured inference time: EdgeDoc, the Idakto edge framework and FLiD describe themselves as lightweight or near real-time and give no latency, FPS or CPU figure; DeepID's only runtime is container wall time on an RTX 3090; vendors claim millisecond recognition with no error rate. The timing table is therefore the first measured time budget for an identity-document alteration detector that is also evaluated on unseen designs, stated as "we found no published measurement". Place: methodology (deployment measurement), results (timing table), discussion.

**What GDRFA can do with it.**

14. Each recommendation names the result it follows from: request cross-design error rates from the vendor using the released protocol (items 2 and 8 to 11); calibrate the threshold on the Centre's own template library (item 12); pilot the detector as a referral suggestion beside the existing checks on the counter computer (items 12 and 13); track referral and confirmed-hit rates per design as a KPI (item 12). Place: conclusions and recommendations.

**The chain against the award's eight criteria.**

| Criterion | Carried by |
|---|---|
| 1 Scientific value | items 2, 3, 8: a measurement nobody publishes, on a failure everybody measures |
| 2 Originality | items 4 to 6: the specimen of the same design as an input to a learned alteration detector, measured on the benchmarks where the failure was measured |
| 3 Theoretical framework | items 1, 5, 6, 12: inspection levels, reference conditioning as change detection, the referral policy |
| 4 Language and presentation | the award structure, tables for every result, figures 1 to 8 |
| 5 Methodology | items 8 to 12, Section 5: controlled baseline, four protocols, the paper's folds and nationality groups, one seed per fold or group |
| 6 Integrity and citation | literature table separate from results, every number sourced, no "first" claims |
| 7 Analysis and results | items 12 and 13, intervals, per-design and per-attack rows, recommendations tied to results |
| 8 Applied impact | items 1, 2, 13, 14: the operating model, the time budget, the released protocol |

## 2b. Positioning, the intersection

Four strands of literature each hold one side of the problem, and the literature review is written strand by strand, each closing with what it lacks. None holds all four, and the paper sits where they meet.

| Side | What it has | What it lacks | Representative work |
|---|---|---|---|
| The public benchmarks | leave-design-out evaluation of alteration detection, published numbers, masks, public data | every evaluated detector is single-image, or pairs a random genuine of any design (CoAARC); none can use the specimen the counter holds; none reports a referral policy or a time budget | SIDTD, IDNet, FantasyID, DeepID, DocForge-Bench, the 2026 survey |
| Reference-conditioned detectors | a genuine reference as a second input | within design, private data, or another task (recapture, guilloche copy-move, presentation attacks); never measured on the benchmarks above; no masks on alterations; no time budget | CheckSim, Chen et al., Al-Ghadi and FMIDV, Neitthoffer et al., Nieto-Hidalgo et al. 2026 |
| Deployment-oriented systems | lightweight architectures, edge or mobile framing, vendor readers with template libraries | no measured inference time or FPS anywhere; single-image or rule-based; no published cross-design error rate | EdgeDoc, Idakto's edge framework, FLiD, Regula and Keesing readers |
| The architecture's source fields | a dense reference-versus-query comparison with alignment, proven on golden-sample inspection and bitemporal change detection; the trusted-reference reading of differential morph detection | never applied to documents; no notion of legitimate variation (a board must match everywhere, a document's fields must differ) | Daudt et al. 2018, Changer 2023, ChangeChip 2021, CADNet 2025, Visual ChangeNet, Scherhag et al. 2020 |

Where we sit: a registered specimen of the query's own design, given to a learned alteration detector with a map head, measured leave-design-out on the public benchmarks where the single-image failure was measured, against the same detector without the specimen, with the output read as the first-line decision (referral at a miss budget) and timed on a counter-class CPU. Each of the three sides contributes one ingredient; the intersection is empty in the literature.

**Contributions, stated with their boundaries.**

| | Contribution | Is | Is not |
|---|---|---|---|
| C1 | The measurement | the first leave-design-out measurement of specimen conditioning for alteration detection on public data, with the specimen as the only difference from the baseline, on three protocols against the published rows | not the first reference-based detector (CheckSim, Chen) and not a new benchmark dataset |
| C2 | The method | the design-median reference, the zone channel, the wrong-design gate and the referral head on a published change-detection network, in a standard and an attention variant | not a new architecture; the network is ChangerEx |
| C3 | The decision | a referral policy calibrated on the source at a miss budget, transported to unseen designs, with realized miss and referral rates and prevalence scenarios; the first such reading for a learned document detector | not a workload saving at GDRFA and not a prevalence estimate |
| C4 | The time budget | measured inference time and FPS on a counter-class CPU for every variant, with and without cached reference features; the first measured time for an alteration detector also evaluated on unseen designs | not an on-sensor or quantized deployment, which is future work |
| C5 | The release | references, zone masks, box masks, provenance-grouped splits, the protocol and the code, so GDRFA can rerun the measurement on its own template library | not real documents and not a vendor evaluation |

## 2c. Why the architecture should work here, and where it may not

**The failure is design-tied, so the fix must carry the design.** On IDNet the same generator gives 98 to 100 percent within a design and 35 to 71 across designs; the generator did not change, the design did. A single-image detector must learn two things at once, what this design looks like and how an alteration deviates from it, and only the first fails to transfer. A registered specimen supplies the first at inference. What remains to learn, how an altered field deviates from its design (a texture discontinuity at the field edge, a broken guilloche, a font, kerning or baseline that disagrees with the printed labels, halftone or noise statistics that differ inside and outside the field), is the same function on every design. That is also why golden-sample inspection generalizes to a new board layout without retraining: the reference carries the layout.

**Four independent precedents say reference information is what carries generalization.** Differential morph detection outperforms single-image morph detection because the trusted reference supplies the baseline (Scherhag et al. 2020). Supplying four target bona fide samples at inference cut the error on unseen countries from 26.6 to 9.1 percent (Nieto-Hidalgo et al. 2026). Removing the reference from the FMIDV Siamese dropped it to chance. CADNet reports generalization from few labels precisely because it learns feature differences rather than defect appearance. The contrary precedent is CoAARC, whose partner is a random genuine of any design; it shows that an unregistered, wrong-design partner does not help, which is the condition our design removes.

**What each component is for.**

| Component | The failure it addresses | Why it should help |
|---|---|---|
| Shared encoder, absolute feature difference | design-specific appearance | the difference is near zero on unchanged static design and non-zero at an altered field, on any design; shared weights force a design-agnostic feature space |
| Page features kept beside the difference | legitimate variation in fields | inside a field the question is how the content differs, not whether; an inpainted name shows blended edges and a wrong font against the label, which needs the page's own features |
| Zone channel | every genuine pair differs in its fields | tells the network where difference is expected and where it is forbidden; difference outside zones is almost a direct signal, inside zones the task is consistency; without it the network must infer the layout from one example |
| Flow dual-alignment fusion | residual misregistration | pixel-level reference comparison fails first on misalignment (the recurring complaint in inspection work); the fusion aligns features before differencing; the misregistration sweep measures it |
| Feature exchange | each stream blind to the other until fusion | parameter-free early interaction, shown to help most on the harder change-detection set |
| Cross-attention, page queries reference | fields inpainted without a seam | each location reads what the specimen contains there (label font, spacing); the attention route Neitthoffer et al. found necessary where stacking failed |
| Map head with box masks, plus score head | small training sets | dense supervision from every altered pixel, the EdgeDoc and TruFor recipe proven on FantasyID; the score head is the decision |
| Change-detection pretraining (LEVIR checkpoint) | few thousand pairs | the encoder already knows how to compare two registered images and find differences, the same reason TruFor transfers from natural-image forensics |
| Design-median reference | no blank specimen in public data | the median over genuine documents removes personal content and keeps the printed design, the nearest public analog of the specimen a template library holds |

**Why it fits the competition.** It mirrors what the counter already does (a specimen per design, lookup by the design the reader identifies), it is small enough for a CPU (ResNet-18 ChangerEx, about 12 M parameters, to be timed), and it hands the officer a map and a decision, not a probability.

**Where it may not hold, stated before the results.**

1. Face swaps. The design median has no face; the portrait sits inside a zone, so the difference says nothing about whether the face was swapped. Detection there rests on the page's own blending cues, as for every published detector: TruFor and MMFusion reach AUC 55.8 and 61.5 on FantasyID's face-only attack while scoring 99 on text, DeepID's teams fell on the face-only subset, FLiD's dedicated face-crop classifiers reach AUC 0.83 to 0.85 (EER 22 to 25 percent) within design, and IDNet's face-crop morph detectors fall from 98 to 50 to 62 percent across designs. The literature's answer to faces at a border is a different reference, the traveller's live face: differential morphing-attack detection compares the document photo with a trusted live capture (Scherhag et al. 2020, Borghi et al. 2021), and document-to-live face comparison is already standard at the counter under ICAO Doc 9303. The paper therefore scopes the specimen hypothesis to the printed design and the personal text fields, reports face swaps separately, trains with IDNet's face attacks so the single-image cues are at least learned, and names the live-face branch as the natural second reference for future work.
2. Capture gap. The encoder trains on clean SIDTD and IDNet renders; FantasyID queries are photographs. The reference for each FantasyID design is built from captured genuine cards, so the reference carries the capture domain, which helps; the encoder may still shift. Augmentation first, SIDTD's captured clips second.
3. Registration. A failed quad or a wrong design yields spurious difference everywhere; the gate catches the detectable cases and the sweep measures the rest.
4. The bet itself. FMIDV's Siamese was weak and CoAARC's partner was wrong; ours wagers that a registered same-design reference, a zone channel and dense supervision are what was missing. The controlled baseline makes the result informative either way: if the specimen does not help, the paper says so against the published rows.

**Expectations, not promises.** Protocol A: the published single-image rows sit at 82 to 91 percent and within-design at 97 to 100; if the hypothesis holds, the reference recovers most of that gap. Protocol B text: single-image AUC is already near 99, so the gain to measure is the referral rate at a 5 percent miss budget (62 percent of genuine cards for TruFor) and the localization. Protocol B faces: uncertain. Protocol C: the published 50 to 71 percent is the number to move.

## 3. The four protocols

| Protocol | Train | Test | Reference for a test design | Published comparison | Priority |
|---|---|---|---|---|---|
| A (anchor) | SIDTD, the paper's 10 official template folds, all nationalities seen, raw files | the fold's test file, about 222 images; mean and sd over the 10 folds | per-fold median over the design's genuine templates in that fold's training file, leaving out the query's component | SIDTD Table 5, Composite Detection (K-fold, Balanced), all five models; our no-reference control must reach the published range (gate) | 1 |
| B | SIDTD, six nationalities, for each of the paper's 10 nationality groups | the group's four held-out nationalities; mean and sd over groups, plus per nationality | median over the unseen design's genuine templates, leaving out the query's component | none published on templates; compared against the control validated in A | 1 |
| C | SIDTD all 10 designs plus IDNet AZ and WV face attacks | FantasyID official test: 300 genuine, Attack-2 face swaps and Attack-3 text rewrites on the 3 held-out designs; Attack-1 separately with Arabic broken out | leave-identity-out median over the design's genuine captures | FantasyID Table 3; DeepID FA2 and FA3; DocForge-Bench pixel F1; our own TruFor rerun on the Arabic subset if time allows | 2 |
| D | IDNet WV, four attack folders | IDNet AZ, last 1,000 identities per class, per task; and the reverse | median over the target's 4,979 non-test genuine images | IDNet Table 14 | 3 |

Protocols A and B: docs/superpowers/specs/2026-10-10-sidtd-anchor-protocols-design.md. C and D are revisited in their own plan.

Every split is grouped by provenance component: a forgery stays with its source document and, for crop-and-replace, its donor document. The detector never trains on a test design and the reference never contains the query's identity.

## 4. Paper structure and word budget

| Section (award numbering) | Content from the chain | Words |
|---|---|---|
| Title page, both titles, authors | — | 150 |
| Arabic abstract, English abstract | items 1, 3, 6, 9 to 13 in 250 words each | 500 |
| 4 Introduction | items 1, 2, one sentence on 6, the contributions, Figures 1 and 2 | 600 |
| 5 Research problem, significance, objectives | problem statement, significance for GDRFA and the field, RQ1 to RQ5, contributions | 500 |
| 6 Framework and literature | the referral and reference principles; four strands (single-image benchmarks, reference-conditioned document detectors, deployment-oriented systems, the architecture's source fields: inspection and change detection), each closing with what it lacks; positioning paragraph; Table 1 | 1,100 |
| 7 Methodology | population and samples (data table, FantasyID facts), tools (pipeline, reference, masks, architecture figure, grid), protocols and metrics (Table 3), deployment measurement | 1,600 |
| 8 Results and discussion | Protocol A table, Protocol B table, Protocol C table if run, ablations, referral curves, timing table, discussion and limitations | 1,300 |
| 9 Conclusions and recommendations | item 14, one recommendation per result | 450 |
| 10 References | about 40 entries | 800 |
| 11 Appendix | licences and access; the released protocol files | 200 |
| Total | | about 7,100 |

Figures: attacks and the two-line check (1), the idea across designs (2), the project at a glance (3, landscape), the system (4), the templates (5), the architecture (6), the protocols (7), the referral curves (8, measured), the Gantt is dropped from the submission. Every result is a table or a figure; prose says what it means.

## 5. Experimental protocol

**Data.** SIDTD templates (1,000 MIDV-2020 genuine, 1,222 forgeries, official few-shot and k-fold splits); FantasyID (2.6 GB Zenodo archive; the local copy is partial); IDNet Zenodo AZ and WV (45 GB; 795 x 487 JPEG, a reduced release). Nothing else. Facts and caveats in the data decision note.

**Preparation.** Rectify every document to a 512 x 800 canvas by homography (SIDTD and IDNet are already flat; FantasyID ships the document quad per image). Zone mask per design from the original field rectangles of its genuine documents. Ground-truth mask per forgery from the altered rectangles (SIDTD field on the source, dilated 8 px; IDNet bounding boxes; FantasyID altered regions). Reference per design as above. Pairs: positive (forgery, reference), negative (genuine, reference of another identity), hard negative (genuine, wrong-design reference) labelled design mismatch.

**Models.** ChangerEx, ResNet-18 backbone initialized from the LEVIR checkpoint. Four settings: single image plus zones (B1), stacked six-channel plus zones (B2), difference fusion (B3, the standard variant), attention fusion (B4, the improved variant). Fallback if mmcv fails on the cluster on day 1: a ResNet-18 Siamese with stage 2 to 4 features, absolute differences and a light FPN map head, same four settings.


**Lineage of the architecture (every block has a source; ChangerEx has not been used for documents before, and the paper says so).**

| Block | Source | Note |
|---|---|---|
| Shared encoder, absolute feature difference | FC-Siam-diff, Daudt et al. ICIP 2018 | founding Siamese change-detection design; same reading as differential morph detection (Scherhag et al. TIFS 2020) |
| Feature exchange between streams; flow dual-alignment fusion | Changer, Fang et al. TGRS 2023; Open-CD toolbox | remote-sensing origin; alignment fusion absorbs residual misregistration of a photographed card |
| Page features kept beside the difference | FC-Siam-conc variant; artifacts are one-sided | our choice of which stream to keep |
| Cross-attention, page queries the reference | Neitthoffer et al. ICDAR-W 2023 (attention worked, stacking failed); Changer's aggregation-distribution interaction | the "improved variant" of the writer-identification structure |
| Joint score and mask heads; BCE plus 3 x (BCE plus Dice) | TruFor (CVPR 2023), EdgeDoc (DeepID 2025) | the two document-adapted DeepID detectors |
| Zone channel, design-median reference, wrong-design gate, referral head | ours | exist because the second input is a specimen of a design, not a second observation of the same scene |
| The class itself: Siamese comparison against a golden reference with a dense difference map | automated optical inspection (ChangeChip, PAINE 2021; CADNet, Scientific Reports 2025; NVIDIA Visual ChangeNet; CoNet 2023) and remote-sensing change detection | in identity-document work the comparison exists only at score or patch level (CheckSim, Chen, Al-Ghadi, CoAARC, Neitthoffer, prototypes); documents differ from boards in that the fields are supposed to differ, hence the zone mask |

**Metrics.** Image level: AUC, EER, APCER at BPCER 1 and 5 percent, HTER at FantasyID's operating point, accuracy at 0.5 where the published row uses it. Pixel level: F1 and IoU against box masks. Policy: transported threshold miss and referral (primary), referral at test miss 1 and 5 percent (secondary), prevalence scenarios. Deployment: median and p95 time per document for pre-processing, network (with and without cached reference features) and end to end, FPS, parameters, file size, on a cluster CPU node and one laptop or desktop.

**Statistics.** Three seeds per setting; mean and standard deviation; paired component bootstrap for the reference-minus-baseline difference with 95 percent intervals; published rows in a separate literature table.

**Audits.** Cross-tabulate label against JPEG quality and dimensions before training; misregistration sweep 0 to 12 px; wrong-design reference with and without the gate; reference construction ablation (median vs single specimen, with and without zone mask).

## 6. Training and timing recipe

Every row below was checked against published code before it was fixed; the sources and the reasons are in `docs/conventions.md`, the audits in `evidence/code-audit-*.md`.

| Setting | Value |
|---|---|
| Framework | Open-CD 1.1.0, ChangerEx r18 config, six registered subclasses (dataset, loader, packer, preprocessor, detector, head), no edits to Open-CD; ResNet-18 Siamese in plain PyTorch only if the mmcv wheels fail on the cluster |
| Stack | torch 2.1.2 cu121, mmcv 2.1.0 (prebuilt cu121/torch2.1 wheel), mmengine 0.10.4, mmsegmentation 1.2.2, mmdet 3.3.0 |
| Registration | every image registered to its design (SIFT, Lowe ratio 0.75, RANSAC similarity, 3 px, two passes, floor 40 inliers); inlier count logged and used by the wrong-design gate |
| Reference variants | registered design median (leave-identity-out, default); blank specimen (IDNet paper templates, Protocols A and C); single genuine card of another identity (the CheckID choice); unregistered median as the negative control |
| Input | 512 x 800 canvas; 8 channels, query RGB plus zone, reference RGB plus zone; ImageNet mean and std on RGB, zone channel mean 0 std 1; BN not SyncBN |
| Initialization | LEVIR-CD ChangerEx checkpoint (trained from random init, selected on the LEVIR test split; both stated in the paper); stem padded from 3 to 4 input channels with zeros |
| Loss | Open-CD softmax cross-entropy with OHEM on the change map (weight 3) plus BCE on the pooled referral logit (weight 1) |
| Optimizer | AdamW, pretrained modules 1e-4, new modules 5e-3, weight decay 0.05, 1,000-iteration linear warm-up then poly decay |
| Batch | 8, mixed precision |
| Epochs | 40 (A, B), 20 (C); best checkpoint by source validation AUC, never by a test design |
| Augmentation | geometry drawn once per pair (affine up to 8 px, alignment jitter up to 2 px with p 0.25), photometric drawn per image, JPEG quality 60 to 100 drawn per image; no flips, no crops that remove the altered field |
| Compression control | raw files read as distributed, canvases lossless (PNG), no JPEG augmentation in Protocols A and B (the paper's way); every headline number also reported with test canvases re-encoded at quality 75 for both classes (the "equalized" column); the quantization-table audit table is printed in the data section |
| Scores | exported as bona fide probability (1 = genuine), threshold 0.5 with `>=` for the DeepID rows; miss = APCER, referral = BPCER; transported threshold chosen on the source validation split at 5 percent miss by the bob `far_threshold` rule and frozen; EER and HTER reported for FLiD and FakeIDet2 comparability only |
| Seeds | 42, 43, 44; mean and std over seeds |
| Timing | batch 1, FP32, CPU, 4 threads, 20 warm-ups, 200 timed documents; reference features cached per design; ONNX Runtime as the fallback runtime |
| Cluster | account acc-mialhajri, partition gpu, A10G; `scripts/make_sweep.py` launchers; data rsynced to node-local scratch |

## 7. What the paper may and may not claim

May: a leave-design-out measurement of reference conditioning on public data, with masks, against the published single-image rows and a controlled baseline; a transported referral policy and its realized rates; CPU time and throughput on named machines; a released protocol GDRFA can rerun on its own library.

Scope of the claim: alterations of the printed design and the personal text fields, where a specimen of the design is the right reference. Face swaps are run under the same protocols as the published rows (FantasyID face-only; IDNet face morphing and portrait substitution), with both the single-image baseline and the reference-conditioned model, and reported in their own row beside the published numbers; no gain is claimed there, the discussion states why (the specimen is blind inside the photo box by construction) and recommends the live-face reference as the second input. IDNet's face attacks stay in training. Every results table has three rows, text alterations, faces, all attacks; the contributions rest on the first. No sentence claims the method detects forged documents in general.

May not: "first Siamese reference detector" (CheckSim, Chen); "first lightweight field detector" (FLiD); any vendor failure; any real-document or traveller-population accuracy; workload savings at GDRFA; chip, UV or physical-security verification (an RGB model cannot see them); a DeepID leaderboard position (the private set is unavailable and the cohorts differ); exact pixel localization (masks are boxes).

## 8. Milestones and gates

| Date | Milestone | Gate |
|---|---|---|
| 10 Oct | M0 cluster environment; downloads started; loader, references, splits for A, B, C; Sections 4 to 7 of the template drafted from the proposal | one reference and one forged pair plotted per source; splits CSV with provenance components; mmcv import passes or the fallback model is chosen |
| 11 Oct | M1 ChangerEx trains on one batch; four settings; Protocol A (12 jobs) and C (12 jobs) launched | jobs queued; one epoch completes without NaN |
| 12 Oct | M2 Protocol B launched (12 jobs); evaluation script with all metrics, referral policy and timing; the CPU benchmark run on finished checkpoints | metrics.json per run; timing JSON per variant |
| 13 Oct | M3 collect A, B, C; rerun failures; figures from real outputs; second-machine timing | final tables and figures; no `\todo` in results |
| 14 Oct | M4 results, discussion, conclusions and recommendations written; internal review by the supervisor | full draft; word count inside 8,000 |
| 15 Oct | M5 Arabic abstract checked by a native speaker; Word and PDF; declarations; submission | submitted |

Fallback ladder: drop Protocol C first (it costs 12 training runs, not data: Protocol B already needs IDNet Arizona), then the attention and stacked settings, then the second timing machine. Priority 1 (A and B with B1 and B3, 12 runs) answers RQ1, RQ2, RQ4 and RQ5 on its own. Download order on the cluster: SIDTD and FantasyID (small, verified locally), then IDNet Arizona alone (13 GB, enough for Protocol B's face attacks), then West Virginia (needed only by C). If Arizona arrives late, Protocol B starts on SIDTD alone and the IDNet-augmented run follows as a second row, reported as an ablation of face attacks in training.

## 9. Risks

| Risk | Answer |
|---|---|
| mmcv or Open-CD fails on the cluster | the ResNet-18 Siamese fallback, decided by noon on 11 October |
| The reference does not help on Protocol A | report the negative result against the published rows and the controlled baseline; the referral analysis stands on the best model; the FMIDV precedent makes this unlikely for text alterations |
| Capture gap sinks Protocol B | augmentation first; SIDTD cropped clips (28.5 GB, forged captures included) second; stated as a limitation if neither suffices |
| Registration fails on FantasyID photos | the shipped quad is used; the wrong-design gate catches failures; the misregistration sweep quantifies the rest |
| IDNet download or schema surprises | Protocol C is last priority; face attacks for B need only AZ |
| Word count overflows | references trimmed to about 40; the Gantt and the second architecture figure dropped |

## 10. Open items

- Researcher 3; job titles, degrees, phone numbers; Arabic title and abstract by a native speaker.
- Which second machine for timing: the Intel Mac is the candidate, since it is a counter-class computer.
- Whether FantasyID's face-swap masks should include FaceDancer's blended margin (DocForge-Bench used the boxes; we follow them).
- The TruFor rerun on the Arabic subset, only if the cluster is idle on 13 October.

## 11. Decision log

| Date | Decision | Reason |
|---|---|---|
| 7 Oct | Event-camera liveness proposal withdrawn; restart from a documented operator problem | staffed touchpoints and iris gates leave no need; see the feedback memory |
| 7 Oct | "Refer or Clear": reference-conditioned alteration detection on unseen designs, built on the Centre's own figures | the one documented first-line problem with public data to measure it |
| 8 Oct | Public data only; the Muharaf-paper framing (data, two protocols, base framework plus modifications, standard and attention variants, grid, three seeds) | the group's established paper shape |
| 9 Oct | All hardware and quantization removed; the model runs on the counter computer with the scanner as input | the author's instruction; the paper is about documents, not devices |
| 10 Oct | Datasets fixed: SIDTD, FantasyID, IDNet AZ and WV; MIDV-2020, IDNet-2025, MIDV-Holo, DLC-2021 out | data decision note; SIDTD contains MIDV-2020; IDNet-2025 adds nothing; the others are presentation attacks |
| 10 Oct | SIDTD's official few-shot split used; published rows noted as measured on clips | the split files were found and verified identical |
| 10 Oct | FantasyID inspected: boxes with altered flags and quads per image; Arabic is a train-val design; headline on the three held-out designs | the files |
| 10 Oct | From the external dossier: polarity fix, zone-aware control, provenance components, transported threshold, prevalence scenarios, mock wording, no "first" claims, JPEG audit | dossier review note |
| 10 Oct | Rejected from the dossier: regenerated SIDTD with a home-made split, dropping IDNet and C, replacing ChangerEx, adversarial and INT8 audits | same note |
| 10 Oct | CPU inference time and FPS promoted to a reported result (RQ5, E10) | the first line has seconds per document; the deployment story is part of the paper |
| 10 Oct | Conventions fixed from four code audits (SIDTD, DeepID, Open-CD, FLiD and ISO metric libraries): bona fide-high scores, bob `far_threshold` rule on the source validation split, DeepID F1 polarity, official SIDTD and FantasyID folders kept, Open-CD stack pinned | `docs/conventions.md` |
| 10 Oct | JPEG quality is a perfect class label in SIDTD (92 vs 75) and FantasyID (95 or 93.5 vs 75) on every split; canvases re-encoded at 95, random JPEG on both inputs, every headline number also reported on quality-75 equalized test canvases | `scripts/audit_jpeg.py`, `results/tables/jpeg_audit.md` |
| 10 Oct | Registration added to the data stage: SIFT + RANSAC similarity of every image to its design, two passes, annotation-free; FantasyID captures carried up to 2.8 degrees of rotation and 24 px of shift on the canvas, which blurred every median; SIDTD transforms are sub-pixel | the reference is the method; a blurred specimen makes the difference channel useless; `data/processed/counts.md` registration table |
| 10 Oct | Blank specimens recovered from the IDNet paper figure for the ten SIDTD designs and AZ, WV; registered to the design medians with identity transforms; added as the "specimen" reference variant (Protocols A and C); FantasyID stays on the registered median | no public forgery dataset ships blanks; IDNet built them by inpainting one public sample per design; `docs/data-decision.md` |
| 10 Oct | Protocol A final results (18 runs): median reference 0.83 AUC vs 0.53 control vs 0.72 specimen; attention variant 0.03 worse, kept as ablation; copy-move undetectable by construction; threshold transported from seen designs refers nearly all genuines | native and QF75 agree within 0.01; SIDTD published few-shot rows are on Clip-cropped images with 5 labelled forgeries per test episode, so not yet comparable to our template runs; direct comparison requires the clip partition (corrected the same day, an earlier JPEG explanation was wrong); see docs/results-protocol-a.md |
| 10 Oct | Protocols redesigned around the SIDTD paper: A is the anchor (the paper's template 10-fold protocol, raw files, official folds, accuracy and ROC AUC over folds; control, median and specimen arms; published numbers trusted as reported, released checkpoints rescored only on suspicion; gate on the control), B holds out nationalities with the paper's 10 groups and the same recipe; FantasyID becomes C, IDNet D; earlier A runs superseded | the no-reference control must reproduce published numbers before any reference gain is claimed; spec docs/superpowers/specs/2026-10-10-sidtd-anchor-protocols-design.md |

## 12. References

Primary sources read on 10 October 2026 unless stated. Full entries in `references.bib`.

1. GDRFA Dubai Scientific Research Award, official guide. researchaward.gdrfad.gov.ae/guide-en.html
2. Gulf News, "Dubai detects over 900 forged documents using smart system", 2 July 2026.
3. Boned et al., SIDTD, Scientific Data 2024. doi 10.1038/s41597-024-04160-9; repository github.com/Oriolrt/SIDTD_Dataset; files datasets.cvc.uab.es/SIDTD/
4. Bulatov et al., MIDV-2020.
5. Guan et al., IDNet, arXiv 2408.01690 v2 (cross-type Tables 14 to 20); Xie et al., IDNet, IEEE BigData 2024; Zenodo records 13852757 (AZ), 13854938 (WV, GRC), 13854897 (ALB)
6. Korshunov et al., FantasyID, IJCB 2025, arXiv 2507.20808; Zenodo 17063366
7. Korshunov et al., DeepID challenge, ICCV Workshops 2025, pp. 521 to 530
8. Yu et al., Sunlight solution, github.com/ZeqinYu/ICCV-DeepID2025-Sunlight (CC BY-NC)
9. George and Marcel, EdgeDoc, arXiv 2508.16284
10. Guillaro et al., TruFor, CVPR 2023
11. Zhao et al., DocForge-Bench, arXiv 2603.01433
12. Kumar et al., FLiD, arXiv 2605.09089
13. Muñoz-Haro et al., FakeIDet3-DB, arXiv 2607.26641; FakeIDet2, Information Fusion 2026
14. Das, Kumar and Ramachandra, survey, arXiv 2607.01442
15. Nieto-Hidalgo, Espin and Tapia, arXiv 2608.16591
16. Tapia et al., third competition on document forgery detection, arXiv 2607.15734
17. Li et al., layout-aware open-set ID fraud discovery, CVPR Workshops 2026
18. IDSPACE, arXiv 2609.03052
19. Ghanmi et al., CheckSim, ICDAR Workshops 2021
20. Chen et al., recapture authentication, IEEE TIFS 2022
21. Al-Ghadi et al., guilloche, 2022 and 2024; FMIDV, MMSP 2023
22. Neitthoffer et al., ICDAR Workshops 2023
23. Fang et al., Changer, IEEE TGRS 2023; Open-CD
24. ICAO Doc 9303; Frontex quick check cards; NIST SP 800-63A-4; ISO/IEC 30107-3
