# Manuscript status and consistency notes (10 October 2026)

Written with the first draft of Sections 1 to 4 (the guide's sections 4 to 7). The plan (`docs/paper-plan.md`) stays the contract; this note records what the draft changed, what in the repository is stale or inconsistent, and the decisions the authors still need to make.

## 00. Update of 11 Oct (Methodology and Results moved to the redesigned protocols, after commit c9cb1d8)

Base: `GDRFA-Award-Paper` at b672780, which holds Hamza's edits of 10 Oct evening. Sections 1 to 3 read well (Ariel, 11 Oct) and were kept, with only these factual fixes:
- the result sentences of both abstracts (the superseded 0.53 to 0.83 result is gone);
- RQ2 now names Protocols B and C;
- two guidance notes, one proposing a contribution on the compression cue and one on the SIDTD paragraph of the literature review.

The first contribution and Table 1 still say "three protocols". This is correct, since they count the held-out-design protocols B to D.

What changed in Sections 4 and 5:
- **Four protocols.**
  - A: anchor, SIDTD's ten official template folds, every design seen in training, with the gate;
  - B: the ten held-out nationality groups;
  - C: FantasyID;
  - D: IDNet.
- **Pipeline.**
  - Lossless PNG canvases and no JPEG augmentation.
  - The checkpoint is chosen by best validation accuracy, with ties broken by AUC.
  - Medians exclude the query's identity (per fold in A).
  - The control receives no wrong-design pairs.
- **Training table corrected.**
  - The anchor runs have `amp: false` in `train.json`, so "mixed precision" was removed.
  - Seed 42, one run per fold.
- **Control caveat.** The paper says that the reported control still had the feature exchange on (11c9cc3), and that the fair control (`single_noex`) and the with-cue control (e6b986d) are pending.
- **Protocol A table.** Table 5 layout: the five published rows, the two rescored released checkpoints, and our three arms, each on raw files and equalized at quality 75.
- **Protocol A text.** Gate FAIL, the diagnostic, per attack (inpaint 0.995 against copy 0.48), per design, map F1, and a rewritten referral analysis (median refers 54% of genuine documents at a 5.5% miss rate).
- **Earlier configuration (18 runs).** Kept only in the text:
  - as a preliminary indication for Protocol B (median 0.83, blank 0.72; the control gain is not reported);
  - as the attention ablation (0.80 against 0.83), as the spec allows.

Open for the authors:
1. **Equalized columns.** e6b986d removed `equalized_qf75` from the protocol configs, but `docs/conventions.md` (JPEG handling row) still says that every test is also reported equalized. Keep the equalized columns for our Protocol A arms or drop them?
2. **Official folds.** Do SIDTD's official folds keep a forgery in the same split as its source document? The paper now claims identity grouping only for B to D.
3. **Proposed contribution** on the compression cue (guidance note after the contributions).
4. **Literature review.** SIDTD's seen-designs result above 96% and its clip results are both confounded by file format, so the Introduction, the literature review and panel A of `fig_idea` may need a sentence.
5. **Pending runs.** Fair and with-cue controls, Protocol B (waiting for the gate decision), the plans for C and D, and the paired bootstrap.
6. **Unverified candidate sources.** `docs/study-brief.md` (line 51) says that references are "never test samples", but the spec builds the Protocol B median from the test design's other identities, and the paper follows the spec. Three sources named in the 11 Oct data note are not read or cited yet and need dated records before any citation:
   - Talarmain et al., ICDAR 2024;
   - arXiv 2606.13528;
   - DocVerify, PRL 2026.

## 0. Update of 10 Oct, afternoon (after commit 498c548)

Decisions taken in the draft. Each is reversible.

1. **Cuts.**
   - The project-at-a-glance figure (`fig_pipeline`) and the template-diversity figure (`fig_templates`) are commented out in `main.tex`; their files stay.
   - Five citations that carried no claim of their own were removed: the duplicate Gulf Today article, the Click and Travel release, the Smart Engines manual, the Transformer and AdamW references.
   - Sections 1 to 4 and the limitations were tightened.
2. **Attack-1 genuine counterparts.** Attack-1, with the Arabic cards broken out, is reported twice. The first comparison is against the 300 held-out genuine cards, as published. The second is against the genuine cards of its own ten designs from FantasyID's training folder, which our model never sees. **The evaluation code must implement the second comparison.**
3. **Published rows** stay in the results tables as anchors, below their own rule and labeled with their conditions (plan item 8). The plan's Section 5 sentence "published rows in a separate literature table" is superseded.
4. **Arabic abstract** drafted as a translation of the English abstract, with the same placeholders; **a native speaker must check it.**
5. **Title page:** Researcher 3, and the job titles, degrees and phone numbers of all four researchers, are still missing. Only the authors can supply them.

The methodology now follows commit 498c548:
- registration by SIFT and RANSAC (FantasyID up to 2.8 degrees and 24 px, SIDTD sub-pixel);
- reference variants: registered median (default), the IDNet authors' blank specimens for Protocols A and C, a single genuine card, and the unregistered median as the control;
- the wrong-design gate by registration inliers (floor 40);
- SIDTD forgery counts 145 crop-and-replace and 1,077 inpaint-and-rewrite;
- FMIDV's single-image test (AUC 50 to 56%), restored after checking the published PDF.

`fig_references` shows registered medians and two blank specimens.

Word count after these changes (build of 10:37Z, guidance hidden): 8,187 in the PDF count (23 pages), including the Arabic abstract (about 190 words) and about 480 words inside figures. That is about 7,700 in a Word-style count. Results (about 800 words of prose) and Conclusions (about 350) remain, so roughly 850 more words must go before submission; candidates are listed in Section 3 below.

## 1. What the draft changed

- `paper/main.tex` (identical copy in `GDRFA-Award-Paper/main.tex`):
  - Introduction, research problem and objectives (RQ1 to RQ5, contributions C1 to C5 with their boundaries), framework and the four-strand literature review with Table 1, and methodology (population and samples, data preparation, architecture, protocols and metrics, experiments, deployment measurement) are drafted from the plan, in the style of the authors' IJDAR 2026 paper: American spelling, Oxford comma, percent signs.
  - The English abstract is drafted, with placeholders for the results.
  - The keyword "presentation attack detection" was removed, since the scope is digital alteration.
  - A limitations paragraph and the dataset-license appendix are drafted.
  - The methodology follows `docs/conventions.md` and the Open-CD audit (commit 32a7bda):
    - The architecture is described as ChangerEx actually is: a shared ResNet-18 with parameter-free exchange, per-image multi-scale fusion, then flow dual-alignment fusion (warp and subtract). The proposal's "|f(x) − f(r)| ⊕ f(x) ⊕ z at each scale" is gone.
    - The zone mask is a fourth input channel of both streams.
    - The loss is cross-entropy with OHEM on the map (weight 3) plus BCE on the score.
    - The optimizer and schedule are Open-CD's.
    - The LEVIR-CD checkpoint's random initialization and test-split selection are stated.
    - The JPEG audit table is in the data section, with the quality-95 re-encode and the equalized quality-75 test.
    - The improved (attention) variant is described at the level of the plan, with a guidance note to confirm its insertion points once implemented.
  - The literature review notes that the published single-image rows on SIDTD and FantasyID may partly reflect the JPEG shortcut, without claiming what each model learned.
  - Results tables: the Protocol A caption now states that the published SIDTD rows are five-shot models on clips (code audit, 10 Oct). The Protocol B table now separates Attack-1, Attack-2 and Attack-3 (FantasyID Table 3: TruFor 99.5, 55.8, 99.9; MMFusion 99.8, 61.5, 99.1 and FNR 47.7).
- Figures:
  - `fig_idea`, `fig_pipeline` and `fig_referral` were regenerated by `paper/figures/gen/make_figures.py` to match the plan. The old versions showed FantasyID in training, "40 designs", SIDTD-generator forgeries, a four-backbone grid, two protocols, "20,307 referred, 902 altered (4.4%)", "against referring all" and "about half of alterations missed", the last from the metric polarity that the dossier review corrected.
  - `fig_references` is new: real design medians from the data stage (`results/figures/data_check_references.jpg`).
  - The architecture figure now shows the real median of FantasyID's Arabic design (`arch_reference_median.png`) instead of the same person's clean card, which contradicted the rule that the reference never contains the query's identity.
- `paper/references.bib`: seven entries were added (IDNet arXiv v2, EdgeDoc, ChangeChip, LEVIR-CD, ResNet, Transformer, AdamW). `chen2022domain` now carries its TIFS title and DOI; the preprint title differs.
- `docs/reference-timeline.md`: a dated ledger of every cited source, with what was checked and which peer-reviewed PDFs are needed.
- `scripts/sync_paper_repo.sh` no longer copies `figures/gen` into the paper-only repository.

## 2. Stale or inconsistent items found (not changed)

1. The plan says the manuscript lives under `submission/main.tex` in the proposal folder. It lives in `paper/main.tex` here and in `main.tex` in `GDRFA-Award-Paper`.
2. The plan's item 3 gives SIDTD "97 to 100 within design". The SIDTD article text says "above 96%" for most models (k-fold, templates), and the study brief says 96 to 99. The draft uses "above 96% on templates".
3. The plan's "IDNet cross-type 35 to 71 percent" does not cover the pairs in the data decision note, which range from 35.56 to 80.05 (Arizona to West Virginia, portrait substitution: 80.05; Finland to Spain, face morphing: 78.80). The draft quotes the West Virginia to Arizona row (50.6 face morphing, 60.9 text), which is Protocol C's pair. The idea figure now says "51 to 71 % (IDNet, West Virginia to Arizona)".
4. FantasyID's "AUC text 99.5" in the template was Attack-1, text on the ten training designs. Attack-3, on the held-out designs, is 99.9 for TruFor and 99.1 for MMFusion. The published "AUC all" pools all three attacks, so the headline on the held-out designs compares per attack. Fixed in the table.
5. FantasyID's published operating point is already a threshold set on validation data (FPR 10%) and carried to the unseen test designs. Contribution C3 in the plan calls the transported threshold "the first such reading for a learned document detector". The draft makes no "first" claim for C3 and credits FantasyID's operating point.
6. FantasyID's Attack-1 has no genuine test cards of its own ten designs: the published Attack-1 numbers pair these attacks with the genuine cards of the three held-out designs. For the Arabic breakdown we must choose the genuine counterparts. FantasyID's train-val bona fide cards are unseen by our model and could serve. Decided on 10 Oct: both comparisons (Section 0).
7. `experiments/README.md` (E3, "0 and 10 calibration cards") and `configs/protocol_b.yaml` (`calibration_cards: [0, 10]`) predate commit 7aac471, which dropped the calibration split. E4 (transfer regimes) and E8 (fine-tuning on FantasyID) are no longer in the plan, and the draft does not mention them.
8. `paper/README.md` says that the Protocol A and B tables are pre-filled with the published rows, while the dossier review (item 9) puts published rows in a separate literature table. Decided on 10 Oct: they stay as labeled anchors (Section 0).
9. `proposal/` (8 Oct) is superseded by the plan. Examples: "the first paired dataset", "extra forgeries with the SIDTD generator", the 4.4% prior, "refer everything as today's second line", the four-backbone grid. Nothing should be copied from it without checking the plan.
10. The plan cites ICAO Doc 9303's inspection levels. The wording was checked on 10 Oct against the official PDF of Part 2 (Levels 1 to 3), and the draft quotes it; the Schengen Handbook's definition of second-line checks was checked the same day.
11. The Gulf News article of 2 July 2026 names the unit "Documents Inspection Center" and gives no officer or examiner counts. The counts come from UAE Journal (25 Jan 2025: more than 1,600 officers, about 50 examiners); The National (3 Jun 2024) gave about 1,500 and about 30. The draft cites UAE Journal for the counts.

## 3. Word count (built locally with TeX Live 2026, 10 Oct ~10:11Z)

| Count | Words |
|---|---|
| Whole PDF without guidance, excluding the title page (the rule of `wordcount.sh`) | 8,738 |
| of which text inside figures (`fig_idea` 278, `fig_pipeline` 185, the two TikZ figures about 115, others 100) | about 680 |
| of which the reference list (IEEE "et al." after six authors) | 1,260 |
| English abstract (placeholders empty) | 251 |

Results and Conclusions are still to be written, about 1,150 words of prose beyond the result tables already counted. To finish within 8,000 (a Word-style count, which ignores text inside images), about 1,200 words must go. In order of cost to the paper:
1. Rasterize or drop `fig_pipeline` (185 words inside it) and drop `fig_templates`.
2. Drop citations that carry no claim of their own (the duplicate Gulf Today article, the Click and Travel release, the vendor manual, the standard ResNet/Transformer/AdamW references): about 200 words.
3. Shorten the literature strands and Table 1.
4. Keep Results to about 800 words of prose and Conclusions to about 350.

## 4. Two repositories

The authors said on 10 Oct that `GDRFA-Award-Paper` (synced with Overleaf) is where the paper is edited. Today's edits were made identically in both repositories. `scripts/sync_paper_repo.sh` copies `refer-or-clear/paper` over the paper repository with `rsync --delete`, so pull the paper repository and copy `main.tex` back before running it, as `paper/README.md` says; otherwise edits made in Overleaf are overwritten. After a push to `GDRFA-Award-Paper`, use Overleaf's Menu > GitHub > Pull.
