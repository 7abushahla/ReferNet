> **Superseded (10 Oct 2026).** These runs held out the official few-shot nationalities with re-encoded images and
> JPEG augmentation, and the control received wrong-design pairs. Protocol A is now the SIDTD paper's template
> 10-fold anchor (docs/superpowers/specs/2026-10-10-sidtd-anchor-protocols-design.md). The runs are kept in
> results/superseded/2026-10-10-heldout-reencoded/.

# Protocol A results (final runs, 2026-10-10)

Source: `results/tables/protocol_a.md`, aggregated by `scripts/aggregate_runs.py` from the
18 runs in `results/runs/A-cex-*-s4[234]/` (cluster jobs 37697 to 37714). Recipe: ChangerEx r18,
LEVIR init, fine-tuned end to end, full 512 x 800 canvas, 40 epochs, no early stopping,
best-validation checkpoint, seeds 42 to 44. Train 6 SIDTD designs, test the official
few-shot quadruple (aze, fin, lva, rus), 400 genuine and 494 altered.

## Headline (test, mean +- sd over 3 seeds)

| Arm | AUC native | AUC QF75 | EER native | Loc F1 | Val AUC |
|---|---|---|---|---|---|
| Standard, control (no reference) | 0.530 +- 0.028 | 0.521 +- 0.035 | 0.469 | 0.454 | 0.604 |
| Standard, median reference | 0.827 +- 0.002 | 0.824 +- 0.005 | 0.217 | 0.775 | 0.934 |
| Standard, blank specimen | 0.717 +- 0.012 | 0.716 +- 0.012 | 0.352 | 0.582 | 0.889 |
| Improved, control | 0.533 +- 0.010 | 0.522 +- 0.011 | 0.485 | 0.459 | 0.579 |
| Improved, median reference | 0.799 +- 0.018 | 0.803 +- 0.018 | 0.274 | 0.658 | 0.926 |
| Improved, blank specimen | 0.684 +- 0.029 | 0.677 +- 0.019 | 0.367 | 0.596 | 0.845 |

## Reading

1. The reference is the effect. Without a reference the same network is at chance on
   unseen designs (0.53). With the registered design median it reaches 0.83 with a seed
   spread of 0.002. Localisation F1 moves from 0.45 (empty maps) to 0.78.
2. Native and QF75-equalized numbers agree within 0.01 for every arm, so the result is
   not the JPEG quality-factor shortcut. This matters for the comparison below.
3. The blank specimen is a usable reference (0.72) but worse than the median by 0.11.
   The specimens are scans from a different capture pipeline (paper figure crops), the
   median is built from the same pipeline as the test images. The paper reports both
   and states this.
4. The attention variant does not help. It is 0.03 lower in AUC, three times the seed
   spread, and lower in localisation. It is reported as a negative ablation, not as
   the method. The standard difference fusion is the method.
5. Per attack family (median arm): text inpainting AUC 0.88, EER 0.17; copy-move
   (68 test images) AUC 0.51 for every arm. Copy-move pastes a field from another
   document of the same design, so the pasted region matches the reference in font,
   colour and layout. A reference comparison cannot see it by construction. State as
   a limitation, do not hide it.
6. Per design (median arm): lva 0.93, fin 0.86, aze 0.88, rus 0.65. The Russian
   internal passport is the weak design; its genuine images vary most in print
   (coverage of the median is lowest after fold split). Report per-design.
7. Threshold transport fails. The threshold chosen on the validation designs at 5
   percent miss (about 0.94) refers essentially every genuine of an unseen design
   (referral 0.99 to 1.00) while missing nothing. Scores on unseen designs are lower
   overall, so a threshold set on seen designs is too strict. On the test designs
   themselves, referral at 5 percent miss is 0.85 and at 10 percent miss 0.55. The
   operating point is the honest weakness of the first line and belongs in the paper
   as such; a per-design calibration from a handful of genuine samples (the
   Nieto-Hidalgo setting) is the obvious fix and is future work, not something to add
   now.

## Comparison with the SIDTD few-shot baselines

Corrected 10 Oct after re-reading the paper (Boned et al., Scientific Data 11:1356, Tables 4 and 5).
The published few-shot rows (AUC 0.675 to 0.969) are the column "Source Detection, Few-shot,
Unbalanced". Table 4 states that this partition (`unbalanced_few_shot_split`) uses Clip-cropped
images: frames of smartphone videos of printed documents, cropped and dewarped. Our runs use
the Templates images and the balanced `few_shot_split`. The paper reports no number on the
template few-shot split. The published model is also a 2-way 5-shot prototypical network that
sees five labelled genuine and five labelled forged clips of the test nationality in every
test episode (code audit, docs/data-decision.md). The nationalities and the quadruple are the
same; the images and the information budget are not. There is therefore no published number
on our exact setting yet.

The earlier explanation in this note, that the published rows reflect the JPEG quality-factor
shortcut, was wrong. The quality-factor split (92 vs 75) was measured on templates only, and
the published rows are not on templates.

To obtain a direct comparison: (1) run our method on the Clip-cropped unbalanced few-shot
partition under the published protocol, and (2) run the published prototypical network on
our template split.

## Decisions taken from these results

- Method for Protocols B and C: standard ChangerEx r18 difference fusion with the median
  reference. Attention is an ablation in A only.
- Specimen arm stays in A as the "true blank" comparison; not carried to B or C.
- Protocol B sweeps (12 jobs) should be reduced to the standard arm only (6 jobs:
  median x 3 seeds plus control x 3 seeds) unless the user wants the attention rows.
