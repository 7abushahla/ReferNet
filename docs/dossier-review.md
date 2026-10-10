# Review of the "GDRFA Final Revision, DeepID and FantasyID" package (10 October 2026)

The package (dossier, change log, experiment spec, manuscript sections, policy helper) was written without access to the repository or the data. It is a careful critique. Where it reasons from the papers it is mostly right; where it guesses about the data we have since inspected the files, and our facts stand. Below: what we adopt, what we reject, and what changes in the plan.

## 1. Adopt (these change our plan)

1. **FantasyID metric polarity.** FantasyID treats bona fide as the positive class. Its published FPR is the attack-miss rate and its FNR is the genuine-referral rate. So TruFor's row "FPR 4.9, FNR 62.0" means: at an operating point that misses about 5 percent of attacks, TruFor refers 62 percent of genuine cards. We had described 62 as a miss rate. Corrected everywhere. This strengthens the paper's framing: the published detector already pays a 62 percent referral rate for a 5 percent miss budget, which is the trade-off we measure.
2. **Zone-aware single-image control.** The right baseline is query plus zone mask (their B1), not query alone, so the reference is the only difference between baseline and method. Our "single" setting now carries the zone channel; a plain query-only run (B0) is optional.
3. **Provenance components.** SIDTD crop-and-replace forgeries have a donor document (`second_src`). Identity-grouped splits must group source and donor together. Added to the split rules.
4. **Transported threshold as the primary policy.** Calibrate the referral threshold on source data at a 5 percent miss target, freeze it, and report the realized miss and referral on the unseen designs. Our "referral rate at miss 1 and 5 percent" on the test set is an ROC summary and is kept as a secondary, descriptive number. Their `policy_metrics.py` implements this and the DeepID-style F1; its self-tests pass, and it is now in the repo.
5. **The motivating statistic.** The Gulf News figures (20,307 examinations, 902 irregularities) do not establish prevalence among travellers, nor that the rest were clean, nor a "refer everything" practice. The paper states the ratio as reported, drops "nineteen of twenty referrals are clean", and projects workload only under assumed prevalence scenarios (0.1, 1, 5 percent).
6. **Negative-class wording.** SIDTD negatives are MIDV-2020 mock documents that already carry digital editing from dataset creation. Call them "baseline-unmodified mock" images; FantasyID negatives are "bona fide printed-card captures".
7. **No "first" claims.** CheckSim and Chen et al. are earlier reference-based Siamese detectors; FLiD is earlier lightweight field analysis; TruFor is earlier localization. Our novelty is scoped: to our knowledge no reference-conditioned alteration detector has been evaluated leave-design-out on public data. Phrase it as a gap, not a first.
8. **Compression audit.** DeepID reported class-correlated JPEG quality in FantasyID (QF95 bona fide vs QF75 altered). Cross-tabulate label against format, dimensions and quantization table before training; keep the JPEG augmentation on both classes; report native and recompressed results.
9. **Published rows live in a literature table**, not in our results tables. Our result tables hold only numbers we produced, with the controlled baseline.
10. **"No vendor publishes" becomes "we found no published"**, and no claim about GDRFA workload savings.

## 2. Reject (with the reason)

1. **Regenerating SIDTD forgeries and inventing a 6/4 split with seed 42.** The package could not find SIDTD's split files. We found them on the server under two names, verified they are identical, and they are the published few-shot protocol. Using them keeps the published rows as anchors. Regenerating would discard that.
2. **Dropping IDNet and Protocol C.** The package forbids fetching IDNet for time reasons. Protocol C is the direct rerun of the cross-type failure our introduction cites, and IDNet supplies the only face-attack training data. It stays, as the last priority: A and B run first, C runs if the cluster has capacity on 12 and 13 October.
3. **Replacing ChangerEx with a hand-built ResNet-18 classifier without a map head.** We have box masks for all three protocols, and localization is what DeepID and DocForge-Bench score. ChangerEx is a published architecture with a map head. It stays, with the package's ResNet-18 Siamese as the fallback if the mmcv environment fails on the cluster on day 1.
4. **Its data guesses.** "Do not infer digital_3 from numbers", "do not infer 10 plus 3 designs", "FantasyID validation record is restricted" and so on. We inspected the files: `digital_3` is Attack-1 (786 text rewrites on the 10 train-val designs), `facedancer` is Attack-2 (face swaps on the 3 held-out designs), every image carries field rectangles with altered flags and the document quad. Our data-decision document supersedes its guesses.
5. **Adversarial evasion audit, INT8, Grad-CAM sanity checks, human-AI design.** Out of scope for 15 October. CPU timing is kept and promoted to a reported result (RQ5, E10): the first line has seconds per document, so the deployment story is part of the paper. Misregistration and JPEG stress (our E6) stay.
6. **Single enrolled specimen as the reference.** Their reference is one genuine card of another identity. Ours is the design median with leave-identity-out, which removes personal content and approximates the blank specimen a template library holds. Keep the median; run the single-specimen reference as the ablation we already planned.

## 3. Net effect on the run budget

| Priority | Runs | Content |
|---|---|---|
| 1 | 12 | Protocols A and B, zone-aware single image (B1) and difference fusion (B3), 3 seeds |
| 2 | 12 | Protocols A and B, stacked (B2) and attention (B4), 3 seeds |
| 3 | 12 | Protocol C, both directions, B1 and B3, 3 seeds |

Priority 1 alone answers RQ1 and RQ4. Priority 3 is dropped first if the cluster is slow.

## 4. Corrections applied today

- Study brief, data decision, the submission template's Protocol B table and the Protocol B config: FantasyID's FNR column renamed to genuine-referral rate at the published operating point.
- `src/refer_or_clear/policy_metrics.py` added from the package (tested).
- Split rules: provenance components include donors.
- Wording: baseline-unmodified mock; gap instead of first; prevalence scenarios instead of a 4.4 percent prior.
