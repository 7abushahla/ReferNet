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
- [[#3. The three protocols]]
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

**The gap, read as a deployment failure.**

3. Every published learned detector is single-image. All 26 entries of the ICCV 2025 DeepID challenge and every model in the July 2026 survey take one document image and nothing else, so none can use the specimen the counter already has. Evidence: the DeepID report, the survey. Place: literature.
4. Every measurement on unseen designs is poor, and poor in the way that matters at a counter. IDNet: 35 to 71 percent accuracy from one design to another. FantasyID: AUC 56 to 62 on faces for the best pretrained detectors, and at the published operating point TruFor misses 5 percent of attacks while referring 62 percent of genuine cards, a referral load no first line could absorb. DeepID: F1 0.99 on fantasy cards falls to 0.72 on real IDs. The 2026 survey: attack error above 24 percent for every public model including GPT-5 and Gemini. DocForge-Bench: best pixel F1 0.30. Evidence: the papers' own tables, in a literature table, never mixed with ours. Place: literature.
5. The one 2026 method that improves generalization across countries, Nieto-Hidalgo, Espin and Tapia, does so by giving the model genuine samples of the target card at inference, cutting the error on unseen countries from 26.6 to 9.1 percent. That is the specimen idea applied to presentation attacks with whole-image prototypes and private data. Earlier reference-based detectors exist (CheckSim, Chen et al., Al-Ghadi and FMIDV, Neitthoffer et al.); none was evaluated leave-design-out on alterations with public data, and the FMIDV follow-up that dropped the reference fell to chance. Evidence: those papers. Place: literature, closing with the gap statement: nobody has measured what the specimen the counter already holds buys a learned alteration detector on designs it has never seen.

**The idea, shaped by the operating model.**

6. Make the specimen an input. The reference is the design median, the pixel-wise median over genuine documents of the design, which removes personal content and keeps the printed design, with a zone mask marking where personal data may differ. The detector looks for inconsistency with the design, not for the signature of a training generator. Deploying it is what the counter does today: enrol a specimen per design, look it up by the design the reader already identifies. Evidence: the idea figure and the reference figure. Place: framework.
7. The architecture is a published change-detection network, ChangerEx from Open-CD, with a pixel head and a referral head, chosen because it is small enough for a CPU and because change detection is the exact shape of the task (two aligned images, where do they differ). The architecture is not new and the paper says so; the adaptation is the design-median reference, the zone channel, the wrong-design gate and the referral head. Place: methodology.

**The controlled measurement.**

8. The reference is the only difference between method and baseline. The single-image baseline uses the same code, the same zone mask and the same training data with the reference input removed. The published single-image rows are anchors beside it, not substitutes for it. Place: methodology and every results table.
9. Protocol A, unseen design within one source: train on six SIDTD nationalities, test on four, the official few-shot split, against the published rows (measured on cropped video frames, ours on templates, stated). Place: results.
10. Protocol B, unseen source, generator, capture and script: train on SIDTD plus IDNet's face attacks, test zero-shot on FantasyID's printed and camera-captured cards. The headline is the official held-out designs with face swaps and text rewrites; Attack-1 on the ten other designs is a second result with the Arabic cards broken out. Our model never trains on FantasyID, so all thirteen designs are unseen to it; this is the test closest to a counter meeting a design it has not seen. Place: results.
11. Protocol C, the published pair: train on IDNet West Virginia with all four attack folders, test on Arizona per task, and the reverse, against IDNet's Table 14. This reruns the failure cited in item 4 with the reference added. Place: results, last priority.

**From a score to the officer's decision.**

12. A score becomes a decision through a referral threshold. Calibrate it on the training source at a 5 percent miss target, freeze it, and report the realized miss and referral rates on the unseen designs: that is what an operator would see after enrolling a new design without retraining. Project workload under assumed prevalence scenarios (0.1, 1 and 5 percent), never from the Centre's ratio. Evidence: the referral curves from the saved scores. Place: results and recommendations.
13. The decision must arrive inside the seconds the officer has. Measure inference time and throughput on a CPU, batch size one, FP32, with and without cached reference features (the library lookup), and report them beside the accuracy: pre-processing, network, end to end, FPS, parameters, file size, on a cluster CPU node and a counter-class computer. Target under one second end to end. Quantization and execution on the scanner itself are future work. Place: methodology (deployment measurement), results (timing table), discussion.

**What GDRFA can do with it.**

14. Each recommendation names the result it follows from: request cross-design error rates from the vendor using the released protocol (items 2 and 8 to 11); calibrate the threshold on the Centre's own template library (item 12); pilot the detector as a referral suggestion beside the existing checks on the counter computer (items 12 and 13); track referral and confirmed-hit rates per design as a KPI (item 12). Place: conclusions and recommendations.

**The chain against the award's eight criteria.**

| Criterion | Carried by |
|---|---|
| 1 Scientific value | items 2, 4, 8: a measurement nobody publishes |
| 2 Originality | items 3, 5, 6: the specimen as an input, evaluated leave-design-out on public data |
| 3 Theoretical framework | items 1, 5, 6, 12: inspection levels, reference conditioning as change detection, the referral policy |
| 4 Language and presentation | the award structure, tables for every result, figures 1 to 8 |
| 5 Methodology | items 8 to 11, Section 5: controlled baseline, three protocols, provenance-grouped splits, three seeds |
| 6 Integrity and citation | literature table separate from results, every number sourced, no "first" claims |
| 7 Analysis and results | items 12 and 13, intervals, per-design and per-attack rows, recommendations tied to results |
| 8 Applied impact | items 1, 2, 13, 14: the operating model, the time budget, the released protocol |

## 3. The three protocols

| Protocol | Train | Test | Reference for a test design | Published comparison | Priority |
|---|---|---|---|---|---|
| A | SIDTD, 6 nationalities, 1,328 images | SIDTD, 4 held-out nationalities, 894 images | leave-identity-out median over the design's 100 genuine templates | SIDTD Table 5 few-shot rows (on clips; stated) | 1 |
| B | SIDTD all 10 designs plus IDNet AZ and WV face attacks | FantasyID official test: 300 genuine, Attack-2 face swaps and Attack-3 text rewrites on the 3 held-out designs; Attack-1 separately with Arabic broken out | leave-identity-out median over the design's genuine captures | FantasyID Table 3; DeepID FA2 and FA3; DocForge-Bench pixel F1; our own TruFor rerun on the Arabic subset if time allows | 1 |
| C | IDNet WV, four attack folders | IDNet AZ, last 1,000 identities per class, per task; and the reverse | median over the target's 4,979 non-test genuine images | IDNet Table 14 | 3 |

Every split is grouped by provenance component: a forgery stays with its source document and, for crop-and-replace, its donor document. The detector never trains on a test design and the reference never contains the query's identity.

## 4. Paper structure and word budget

| Section (award numbering) | Content from the chain | Words |
|---|---|---|
| Title page, both titles, authors | — | 150 |
| Arabic abstract, English abstract | items 1, 3, 6, 9 to 13 in 250 words each | 500 |
| 4 Introduction | items 1, 2, one sentence on 6, the contributions, Figures 1 and 2 | 600 |
| 5 Research problem, significance, objectives | problem statement, significance for GDRFA and the field, RQ1 to RQ5, contributions | 500 |
| 6 Framework and literature | items 3 to 5, the referral and reference principles, the literature table | 1,000 |
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

**Metrics.** Image level: AUC, EER, APCER at BPCER 1 and 5 percent, HTER at FantasyID's operating point, accuracy at 0.5 where the published row uses it. Pixel level: F1 and IoU against box masks. Policy: transported threshold miss and referral (primary), referral at test miss 1 and 5 percent (secondary), prevalence scenarios. Deployment: median and p95 time per document for pre-processing, network (with and without cached reference features) and end to end, FPS, parameters, file size, on a cluster CPU node and one laptop or desktop.

**Statistics.** Three seeds per setting; mean and standard deviation; paired component bootstrap for the reference-minus-baseline difference with 95 percent intervals; published rows in a separate literature table.

**Audits.** Cross-tabulate label against JPEG quality and dimensions before training; misregistration sweep 0 to 12 px; wrong-design reference with and without the gate; reference construction ablation (median vs single specimen, with and without zone mask).

## 6. Training and timing recipe

| Setting | Value |
|---|---|
| Input | 512 x 800 RGB query, reference, zone channel |
| Loss | BCE on the referral head plus 3 x (BCE plus Dice) on the map head |
| Optimizer | AdamW, backbone 1e-4, new layers 1e-3, weight decay 1e-4 |
| Batch | 8, mixed precision |
| Epochs | 40 (A, B), 20 (C); best checkpoint by source validation AUC |
| Augmentation | affine up to 8 px, independent JPEG 60 to 100 on both inputs, photometric; alignment jitter up to 2 px with p 0.25; no flips, no crops that remove the altered field |
| Seeds | 42, 43, 44 |
| Timing | batch 1, FP32, CPU, 4 threads, 20 warm-ups, 200 timed documents; ONNX Runtime as the fallback runtime |
| Cluster | account acc-mialhajri, partition gpu, A10G; `scripts/make_sweep.py` launchers; data rsynced to node-local scratch |

## 7. What the paper may and may not claim

May: a leave-design-out measurement of reference conditioning on public data, with masks, against the published single-image rows and a controlled baseline; a transported referral policy and its realized rates; CPU time and throughput on named machines; a released protocol GDRFA can rerun on its own library.

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

Fallback ladder: drop Protocol C first, then the attention and stacked settings, then the second timing machine. Priority 1 (A and B with B1 and B3, 12 runs) answers RQ1, RQ2, RQ4 and RQ5 on its own.

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
