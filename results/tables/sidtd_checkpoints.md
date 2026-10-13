# SIDTD released template checkpoints scored on our copy of the paper's folds

| Model | Published acc (%) | Published AUC | Ours, raw: acc (%) | Ours, raw: AUC | Equalized QF75: acc (%) | Equalized QF75: AUC | folds |
|---|---|---|---|---|---|---|---|
| resnet50 | 98.1 ± 1.2 | 0.99 ± 0.01 | 98.1 ± 1.1 | 1.000 ± 0.001 | 57.2 ± 3.7 | 0.655 ± 0.037 | 10 |
| efficientnet-b3 | 99.4 ± 1.0 | 1.00 ± 0.00 | 99.4 ± 1.0 | 1.000 ± 0.000 | 55.2 ± 2.8 | 0.539 ± 0.049 | 10 |

## Our reproduction of the published recipe (R pilot, job 37967: ImageNet ResNet50, raw files resized to 299, fold 0)

Scored on fold-0 validation (222 images: 100 genuine, 106 inpaint, 16 copy) with the selected epoch-55 checkpoint, by `scripts/score_r_pilot_reencoded.py` (job 38308, 13 Oct 2026). Raw outputs in `results/diagnostics/compression-cue/`.

| Files | Accuracy (%) | AUC | Copy AUC | Inpaint AUC | Genuine flagged |
|---|---|---|---|---|---|
| native | 100.0 | 1.000 | 1.000 | 1.000 | 0 / 100 |
| re-encoded IJG Q75 | 55.0 | 0.692 | 0.598 | 0.707 | 100 / 100 |

Under one quantization table the reproduction flags every validation image as altered (accuracy equals the altered fraction, 122/222).
