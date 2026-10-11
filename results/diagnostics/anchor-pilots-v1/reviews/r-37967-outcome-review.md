# R37967 selected-validation evidence review

Independent read-only reviewer Wegener found no blocking inconsistency in the transferred evidence on 12 October 2026.

- Scheduler records completion with exit 0:0 in 1:17:01.
- Scientific source remains `0959335bd9a6a22510f6706c49c62d84560cd524`, with the original approved R recipe.
- All 100 epochs are present. Accuracy-first, AUC-second selection chooses epoch 55, the earliest of 31 exact ties. Lower loss at epoch 100 does not override selection.
- Recomputed 222 validation predictions, with 100 genuine and 122 altered images, yield accuracy 1.0, AUC 1.0 and image CE approximately 0.01514829.
- Copy-and-replace has 16 altered images and inpainting has 106. Both subgroups have accuracy and AUC 1.0 against the genuine pool. There are no genuine false positives or altered misses.
- Metadata/configuration identities, source inventories, initialization hashes and approval binding agree with R37962. Staged paths contain train/validation membership only.
- Recorded counters show 56 training forwards per epoch across all 53 BN modules, 28 optimizer steps per epoch and no omitted images. Optimizer coverage and scheduler progression agree with the recipe.
- Selected gate/replay identities agree. Recorded fresh-process replay difference is zero; companion invariance and unchanged evaluation buffers pass.

## Limits

The reviewer did not receive checkpoint or source-archive bytes and did not independently rehash or replay them locally. Pixel-access conclusions rely on staging/gate evidence. These are native-release, single-seed, selected-validation results, not aggregate test generalization, architecture superiority or evidence for reference conditioning. No family or sweep is selected by this review.
