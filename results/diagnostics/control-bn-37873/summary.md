# Control checkpoint diagnostic

Batch-statistic probes are batch-dependent diagnostics, not benchmark results.

No test images or test labels are used. No optimizer or gradient updates are run.

| Checkpoint | Epoch | Mode | Split | n | AUC logits | AUC pipeline | Accuracy | Image BCE |
|---|---:|---|---|---:|---:|---:|---:|---:|
| A-cex-single_noex-median-levir-f0-s42-e100/best | 2 | eval | train | 256 | 0.4648 | 0.4648 | 0.5000 | 0.6934 |
| A-cex-single_noex-median-levir-f0-s42-e100/best | 2 | eval | val | 222 | 0.4369 | 0.4369 | 0.5495 | 0.6919 |
| A-cex-single_noex-median-levir-f0-s42-e100/best | 2 | backbone_batch | train | 256 | 0.5238 | 0.5238 | 0.4883 | 0.6929 |
| A-cex-single_noex-median-levir-f0-s42-e100/best | 2 | backbone_batch | val | 222 | 0.5220 | 0.5220 | 0.5450 | 0.6920 |
| A-cex-single_noex-median-levir-f0-s42-e100/best | 2 | all_batch | train | 256 | 0.5325 | 0.5325 | 0.4844 | 0.6929 |
| A-cex-single_noex-median-levir-f0-s42-e100/best | 2 | all_batch | val | 222 | 0.5137 | 0.5137 | 0.5495 | 0.6920 |
| A-cex-single_noex-median-levir-f0-s42-e100/best | 2 | train_recalibrated | train | 256 | 0.5124 | 0.5124 | 0.4961 | 0.6932 |
| A-cex-single_noex-median-levir-f0-s42-e100/best | 2 | train_recalibrated | val | 222 | 0.4475 | 0.4475 | 0.5315 | 0.6929 |
| A-cex-single_noex-median-levir-f0-s42-e100/last | 100 | eval | train | 256 | 0.4837 | 0.4837 | 0.5039 | 3.8731 |
| A-cex-single_noex-median-levir-f0-s42-e100/last | 100 | eval | val | 222 | 0.4586 | 0.4586 | 0.4505 | 4.3544 |
| A-cex-single_noex-median-levir-f0-s42-e100/last | 100 | backbone_batch | train | 256 | 0.9986 | 0.9986 | 0.9922 | 0.0293 |
| A-cex-single_noex-median-levir-f0-s42-e100/last | 100 | backbone_batch | val | 222 | 0.9855 | 0.9855 | 0.9459 | 0.1755 |
| A-cex-single_noex-median-levir-f0-s42-e100/last | 100 | all_batch | train | 256 | 0.9993 | 0.9993 | 0.9922 | 0.0329 |
| A-cex-single_noex-median-levir-f0-s42-e100/last | 100 | all_batch | val | 222 | 0.9817 | 0.9817 | 0.9505 | 0.1974 |
| A-cex-single_noex-median-levir-f0-s42-e100/last | 100 | train_recalibrated | train | 256 | 0.4982 | 0.4982 | 0.5039 | 3.2142 |
| A-cex-single_noex-median-levir-f0-s42-e100/last | 100 | train_recalibrated | val | 222 | 0.4575 | 0.4575 | 0.4505 | 3.6141 |
| A-cex-difference-median-levir-f0-s42-e100/best | 80 | eval | train | 256 | 0.9975 | 0.9975 | 0.9805 | 0.0648 |
| A-cex-difference-median-levir-f0-s42-e100/best | 80 | eval | val | 222 | 0.9670 | 0.9670 | 0.9505 | 0.1700 |
| A-cex-difference-median-levir-f0-s42-e100/best | 80 | backbone_batch | train | 256 | 0.9973 | 0.9973 | 0.9844 | 0.0560 |
| A-cex-difference-median-levir-f0-s42-e100/best | 80 | backbone_batch | val | 222 | 0.9713 | 0.9713 | 0.9505 | 0.1707 |
| A-cex-difference-median-levir-f0-s42-e100/best | 80 | all_batch | train | 256 | 0.9971 | 0.9971 | 0.9844 | 0.0636 |
| A-cex-difference-median-levir-f0-s42-e100/best | 80 | all_batch | val | 222 | 0.9728 | 0.9728 | 0.9505 | 0.1628 |
| A-cex-difference-median-levir-f0-s42-e100/best | 80 | train_recalibrated | train | 256 | 0.9973 | 0.9973 | 0.9844 | 0.0635 |
| A-cex-difference-median-levir-f0-s42-e100/best | 80 | train_recalibrated | val | 222 | 0.9626 | 0.9626 | 0.9414 | 0.1959 |
