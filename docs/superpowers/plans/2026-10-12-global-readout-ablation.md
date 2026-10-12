# Global-readout comparison on the current Protocol A inputs

The user approved the focused comparison after reviewing the current OFF performance gap. Its purpose is to test whether training the added local query branch contributes to that gap. This is one fold-zero, seed-42 development comparison. It does not test unseen nationalities or authorize a sweep.

## Fixed comparison

Use the original 38041 scientific source, raw input bytes, official train/validation membership, fold-safe support products, saved registration matrices, augmentation streams, sampler, initial base tensors, and 100-epoch FP32 training recipe. Restore the authenticated archived inputs at the original absolute paths on another allocated GPU node. Do not rebuild registration or modify the running control.

Construct the original fresh OFF model, authenticate its complete initial state, then expose its global query classifier through a separately identified external wrapper. Execute the backbone once, preserve the four-channel query and stored-statistics evaluation, and return the global query logit as the attack logit with a zero correction. Do not execute the local adapter. Freeze its retained initial parameters and exclude them from optimization; retaining them in storage permits direct initialization authentication. Reuse the base parameter groups, original loss, optimizer, scheduler, selection, and evaluation helpers. The backbone and global classifier both train. Do not load a trained D or OFF checkpoint to initialize this comparison.

The original OFF supervises a global-plus-local sum. This comparison supervises the global logit alone. It therefore tests the local training path, including its gradients and participation in clipping, under the same current data pipeline. It is not an inference-time deletion from an already jointly trained checkpoint.

## Implementation and launch order

1. Feynman owns `scripts/train_direct_reference_global.py` and `tests/test_direct_reference_global.py`. Publish a separate external request, checkpoint, and result schema. Reuse the pinned original-source authentication helper. Implement `smoke`, `train`, and fresh-process `replay` phases. Preserve current scientific modules and earlier execution contracts.
2. Turing owns `scripts/cluster/train_direct_reference_global.sh` and `tests/test_direct_reference_global_cluster.py`. Reuse the reviewed original-input archive restoration, environment, allocation, and protected-output checks. Coordinator alone deploys and submits.
3. Synthetic verification covers exact initial global outputs, absent local execution/gradients, optimizer membership, one backbone traversal and BN updates, evaluation invariance, unchanged original inputs, deterministic sampler/augmentation initialization, partial-batch accounting, selection, and checkpoint replay. Run tests before claiming readiness.
4. Independently review the new external runner and wrapper. Execute one fresh 32-train/32-validation epoch on an allocated GPU, with gradient/update and companion-invariance checks plus fresh-process selected replay. All smokes start from original initialization. The full training phase must reset model, optimizer, scheduler, and dataset RNGs.
5. Review the completed original OFF result and the actual smoke evidence. The intended full run is one 100-epoch fold-zero comparison, with the original 22,200-step scheduler and accuracy-first selection. Launch only after the new path passes its checks and the original result still warrants isolation. A completed control that meets the agreed targets would trigger reconsideration rather than automatic extra training.
6. Record source/helper/launcher/request/data hashes, job IDs, selected validation predictions, query/correction outputs, attack subsets, genuine false positives, and fresh-process replay. Interpretation compares this trained global-only model with the original trained global-plus-local OFF model. It does not make either model a demonstrated final family.

## Scope of the decision

If global-only competence improves substantially, the local training path becomes a supported explanation to investigate and the reference-extension design can be reconsidered around the competent query route. If it remains weak, changes shared by both current models, including input geometry, fold-safe zones, and augmentation, remain candidates. One seed does not establish a universal causal effect. Do not alter thresholds, tune on test data, or weaken support isolation to reproduce the historical score.
