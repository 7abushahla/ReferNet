# Control checkpoint diagnostic

Batch-statistic probes are batch-dependent diagnostics, not benchmark results.

No test images or test labels are used. No optimizer or gradient updates are run.

| Checkpoint | Epoch | Mode | Split | n | AUC logits | AUC pipeline | Accuracy | Image BCE |
|---|---:|---|---|---:|---:|---:|---:|---:|
| A-cex-single_noex-median-levir-f0-s42-e100/last | 100 | eval | val | 222 | 0.4586 | 0.4586 | 0.4505 | 4.3544 |
| A-cex-single_noex-median-levir-f0-s42-e100/last | 100 | backbone_batch | val | 222 | 0.9789 | 0.9789 | 0.9459 | 0.2299 |
| A-cex-single_noex-median-levir-f0-s42-e100/last | 100 | all_batch | val | 222 | 0.9725 | 0.9725 | 0.9324 | 0.3180 |
| A-cex-difference-median-levir-f0-s42-e100/best | 80 | eval | val | 222 | 0.9670 | 0.9670 | 0.9505 | 0.1700 |
| A-cex-difference-median-levir-f0-s42-e100/best | 80 | backbone_batch | val | 222 | 0.9677 | 0.9677 | 0.9505 | 0.1806 |
| A-cex-difference-median-levir-f0-s42-e100/best | 80 | all_batch | val | 222 | 0.9642 | 0.9642 | 0.9234 | 0.2367 |
