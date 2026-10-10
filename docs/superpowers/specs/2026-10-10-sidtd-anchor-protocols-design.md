# SIDTD protocols A (anchor) and B (held-out nationalities): design

Date: 10 October 2026. Status: approved in conversation, awaiting review of this written spec.

## 1. Goal

Protocol A reproduces the SIDTD dataset paper's template protocol exactly, so that our no-reference control lands on the published numbers and validates our data, split and scoring against prior work. The reference arms are then compared with that validated control. Protocol B keeps everything from Protocol A and changes only the split, holding out nationalities, to test the unseen-design claim within one dataset before FantasyID and IDNet.

Reference paper: C. Boned et al., "Synthetic dataset of ID and Travel Documents," Scientific Data 11:1356, 2024, Table 5, column "Composite Detection (K-fold, Balanced)", and its code at github.com/Oriolrt/SIDTD_Dataset.

## 2. Principle

The data, the split files and the scoring follow the paper. Only the method differs. Every method-level difference is listed in Section 5 and stated in the manuscript.

## 3. Protocol A: the anchor

| Item | Specification |
|---|---|
| Images | Raw SIDTD template files: 1,000 genuine, 1,222 forged (`templates.zip`), read as distributed. No re-encoding at any stage |
| Split | The official 10 folds, `kfold_split_templates/split_kfold/{train,val,test}_split_SIDTD_it_{0..9}.csv`, used as given |
| Arms | (1) control: single image, no reference; (2) median: query plus design median; (3) specimen: query plus blank specimen. All three use standard (difference) fusion |
| Runs | 3 arms x 10 folds, one run per fold, seed 42: 30 runs |
| Training recipe | The previous recipe (ChangerEx ResNet-18, LEVIR initialization, 40 epochs, no early stopping, batch 8, learning rates 1e-4 backbone and 1e-3 new layers, weight decay 0.05, 1,000 warm-up iterations, mask loss weight 3, wrong-design pairs at 10 percent for the referral head, mixed precision), with JPEG augmentation removed. Small affine shifts (8 px) and photometric jitter stay |
| Checkpoint | Best validation accuracy in each fold, as the paper selects; ties broken by validation ROC AUC |
| Headline metrics | Accuracy at the 0.5 threshold and ROC AUC on the fold's test file; mean and standard deviation over the 10 folds. Per-fold values are stored, so either standard-deviation convention can be reported |
| Secondary metrics | EER, localization F1 |
| Median reference | Per fold: pixel-wise median of the registered genuine templates of the query's design taken from that fold's training file, excluding the query's own identity |
| Specimen reference | The registered blank specimen of the design (`data/raw/idnet_templates/`) |

## 4. Protocol B: held-out nationalities

| Item | Specification |
|---|---|
| Images and recipe | Identical to Protocol A |
| Split | The paper's 10 nationality groups, as hard-coded in `SIDTD/models/fsl_models/test_fsl.py`: svk_alb_aze_lva, svk_rus_esp_aze, grc_esp_fin_srb, svk_alb_grc_est, alb_grc_rus_lva, alb_grc_est_rus, svk_rus_lva_fin, rus_aze_lva_fin, svk_alb_lva_fin, svk_alb_rus_fin. In each group the four named nationalities are the test set and the other six train |
| Validation | Held-out identities (about 10 percent, grouped so that a genuine and its forgeries stay together) from the six training nationalities, seed 42 |
| Arms | The same three arms as Protocol A |
| Runs | 3 arms x 10 groups, seed 42: 30 runs |
| Reported as | Mean and standard deviation over the 10 groups, plus results per test nationality |
| Median reference | Pixel-wise median of the registered genuine templates of the unseen design, excluding the query's own identity. The network never trains on this design; it receives the median only as input at test time, and never sees a forgery of that design |
| Specimen reference | The registered blank specimen of the unseen design |
| Comparison | Against the control validated in Protocol A. No retrained paper models |

## 5. Method-level differences from the paper (stated in the manuscript)

| Item | Paper | Ours | Reason |
|---|---|---|---|
| Network | EfficientNet-B3, ResNet50, ViT-L/16, TransFG, CoAARN | ChangerEx ResNet-18 change-detection network with image and pixel heads | the contribution |
| Input | resized to 299 x 299 (224 for ViT and CoAARN), aspect ratio not kept | letterboxed to 512 x 800, aspect kept, registered to the design. The same canvas is used for every dataset in the study | keeps about 1.65 times more pixels per altered field (median field height 33 px against 20 px), avoids squeezing text, gives every document of a design one coordinate grid for the comparison with the reference |
| Augmentation | random resized crop (0.8 to 1.0), horizontal and vertical flips, brightness and contrast | small shifts and photometric jitter, no flips | flips would break alignment with the reference |
| Schedule | 100 epochs, learning rate 0.01, batch 32 (paper defaults) | Section 3 | part of the method |

## 6. Checks (gates)

1. Reproduction of the paper. Download the five released template checkpoints (`http://datasets.cvc.uab.es/SIDTD/balanced_templates_SIDTD/{efficientnet-b3,resnet50,vit_large_patch16_224,trans_fg,coatten_fcn_model}_trained_models.zip`, about 26 GB) and score each fold's checkpoint on that fold's test file with the paper's preprocessing (`Baseline/test_kfold_baseline.py`: resize to 299, or 224 for ViT; ImageNet normalization; accuracy from argmax; ROC AUC from the softmax probability of "fake"; CoAARN through its own script `arc_pytorch/test_coAttn.py`). Pass: each model's mean accuracy and mean AUC lie within the larger of one published standard deviation and 0.005 of Table 5. Fail: stop and find the difference in data or scoring before any further run.
2. Validation of the control. Pass: our control's mean AUC is at least 0.99 and mean accuracy at least 0.97, the range of the paper's ResNet50 and EfficientNet-B3. Fail: fix our pipeline before the control is used as the bridge.
3. Compression check (reported, not a gate). Every model, ours and the paper's checkpoints, is also scored on a copy of each test file re-encoded at JPEG quality 75, so that genuine and forged images share one compression level. The main numbers stay on the raw files; this table is reported beside them.

Protocol B starts only after gates 1 and 2 pass.

## 7. Code changes

1. Canvases written losslessly (PNG) instead of JPEG quality 95 in `scripts/prepare_data.py`; registration, references and masks rebuilt from them.
2. JPEG augmentation and canvas re-encoding switched off in the training configuration.
3. Split builders for the 10 official folds (from the CSV files) and the 10 nationality groups (with the identity-grouped validation set).
4. Evaluator reports accuracy at 0.5 and ROC AUC per fold or group, and the mean and standard deviation; the equalized-test variant re-encodes raw test files at quality 75 before preprocessing.
5. A separate script scores the paper's released checkpoints with their own preprocessing, raw and equalized.
6. Aggregation script produces the Protocol A table in the layout of the paper's Table 5 (models as rows, accuracy and ROC AUC as columns), with our arms added as rows.

## 8. Compute

60 training runs (30 per protocol), spread over the available GPUs, plus checkpoint scoring (no training). Protocol B runs are submitted only after the gates pass.

## 9. Consequences

- The 18 earlier Protocol A runs (held-out nationalities, re-encoded images, `results/runs/A-cex-*`) are superseded. They stay in the repository and leave the main tables.
- Protocol renumbering: A is the anchor, B the held-out nationalities, C the FantasyID protocol (formerly B), D the IDNet protocol (formerly C). Configuration files, the paper plan and the manuscript change accordingly; the manuscript is being edited in another session, so the renumbering is coordinated there.
- The SIDTD clips (downloaded and audited, `docs/data-decision.md`) are not used.

## 10. Out of scope

- Input-size ablation (299 x 299 against 512 x 800): decided after the Protocol A results.
- The paper's clip protocols, retraining the paper's models, the attention fusion (kept as an ablation from the earlier runs).
- Protocols C and D: planned after this spec is implemented.
