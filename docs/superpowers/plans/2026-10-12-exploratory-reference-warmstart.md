# Exploratory reference adapter pilot while the control trains

The user authorized parallel reference training from the best validated control checkpoint on 12 October. This pilot answers whether the local adapter can learn useful reference information from the existing query representation before the control finishes. It complements the approved formal experiment sequence. Its results are exploratory and do not substitute for independently trained matched arms.

## Fixed pilot

- Capture the current best OFF checkpoint and contemporaneous validation history as read-only, hash-bound files. Verify that its selected epoch matches the accuracy-first, AUC-second, earliest-tie rule over that captured history.
- Restore the original 38041 code, input authorities, and support products at their original absolute paths on a different GPU node. Reuse the original registration and phase authority. Preserve the running allocation and its files.
- Load all saved model tensors into reference-on mode. Freeze the complete base model, including BatchNorm statistics, while training only the projection and local correction adapter.
- Run ten additional epochs on all 1,778 official training rows, with batch eight and the original drop-last rule. Use a fresh adapter-only AdamW optimizer at learning rate 0.001 and the existing warmup/cosine scheduler for 2,220 updates. Preserve the original query/reference geometry, augmentation, FP32 precision, and classification objective.
- Evaluate the original OFF checkpoint and initial ON configuration on all 222 validation rows before adaptation. Evaluate each adapted epoch and select by validation accuracy, then AUC, retaining the earliest exact tie, including epoch zero. Report attack subsets, genuine false positives, and context availability. Never consume test pixels.
- Save the selected exploratory state and complete continuation state under a distinct schema and output directory. Compare its selected ON scores with its reference-disabled inference scores as an additional diagnostic. That comparison is not an independently trained reference-off arm.

## Implementation and verification

1. The coordinator captures inputs and maintains the submission ledger. Feynman owns the external training script and its synthetic tests. Turing owns the Slurm wrapper and restore tests. Galileo and Maxwell independently review the respective changes.
2. Reuse the authenticated scientific modules and training/evaluation helpers from the original immutable source. Keep the external launcher outside that source. Authenticate the checkpoint, captured history, configuration, registration, phase authority, and helper at startup.
3. Verify strict state loading, original checkpoint selection and identities, adapter-only optimizer membership, frozen base parameters and buffers after training, reference-path gradients, finite updates, and the validation-only reporting contract with synthetic tests.
4. Commit and push reviewed code, record its hashes, and launch one GPU job in the established project environment. Record job ID, request, checkpoint, input/source identities, and output directory. Inspect startup and early training output.
5. Review the completed validation evidence alongside the ongoing control. The pilot provides direction for further reference work. Formal reference-on training and the fold sweep remain separate decisions under the existing plan.

The ten-epoch bound is a practical exploration budget, not an accuracy guarantee. A frozen backbone limits what the adapter can learn. A weak result would not establish that end-to-end reference training cannot help.
