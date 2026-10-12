# Compute plan (10 October 2026, deadline 15 October)

> Historical plan. The estimates and Protocol A–D definitions below were superseded. For current execution, use `anchor-pilot-execution-2026-10-11.md` and the 12 October priorities immediately below. Do not submit the historical campaigns or use their runtime estimates.

## Active priorities, 12 October

1. Complete the existing direct-reference OFF pilot, job 38041. It uses the official fold-zero train/validation population, 100 epochs, and the current direct classifier with its local query adapter. At 15:01 UTC, 50 epochs had completed. Do not restart or hot-patch this allocation. Inspect the selected validation checkpoint, both attack subsets, genuine false positives, context availability, and fresh-process replay when complete. The agreed competence criteria remain selected accuracy at least 0.97 and AUC at least 0.99.
2. Input-pipeline profiling is complete. The completed 38062 benchmark measured 6.437 end-to-end images/s at batch eight, versus 36.736 images/s for a preloaded batch. The reviewed immutable binary-mask cache is implemented, with per-access authentication and synthetic tensor/RNG equivalence retained. Completed GPU job 38088 measured 6.794 end-to-end images/s, a descriptive 5.55 percent improvement, still only a modest gain. Its rebuilt registration prevents attributing the difference exclusively to caching. Function-level profile 38102 identified repeated manifest/preprocessing serialization as a measured optimization candidate. Its immutable-snapshot design remains unimplemented and requires an explicit source/registration compatibility strategy. Neither benchmark changes the scientific batch size or justifies restarting the current control.
3. Reference-on packaging and original-input preservation are prepared and independently reviewed. CPU job 38089 verified all 2,474 archived input files. The external launcher reuses the original scientific implementation, initialization, membership, support products, and exact saved registration. Independently rebuilding registration in a new allocation is not automatically matched because registration keys include source/path-bound preprocessing. Actual ON execution, completed-control competence, explicit authorization, and a suitable measured wall limit remain outstanding. No ON100 or sweep is submitted.
4. Remaining-fold support preparation is complete. CPU jobs 38079 through 38087 built folds one through nine; CPU checks 38091 through 38099 verified all 2,430 products. Genuine-training-only eligibility and original split sizes remain unchanged, including fold four's 1776/223/223 and fold nine's 1777/222/223 membership. Detailed membership counts are in `../evidence/protocol-a-support-preflight-2026-10-12.md`. These banks are prepared inputs, not model results or authorization for a sweep. The current model training service still requires deliberate extension beyond fold zero.
5. Produce a decision immediately after the control completes. If it clears the agreed gate, the next scientific experiment is the matched reference-on pilot. If it does not, use its completed validation evidence to identify the remaining limitation before a sweep. Prepare collection and comparison scripts during training so result interpretation adds minutes rather than another implementation cycle.
6. Exploratory reference adapter pilot 38105 completed all ten epochs. It selected epoch zero under the declared accuracy-first rule, so adapter training did not improve the selected familiar-design validation result. This does not test unseen-nationality reference benefit. Original registration and support products were reused and the base/BatchNorm remained frozen. The implementation plan is `superpowers/plans/2026-10-12-exploratory-reference-warmstart.md`; its completed outcome is recorded in the execution ledger.
7. Prepare the user-approved global-readout comparison while the original control finishes. Keep its current data, initialization, augmentation, and training recipe; train only the backbone/global classifier and bypass the added local branch. The epoch-50 decomposition and historical D comparison identify copy-and-replace as the principal observed gap but do not establish its cause. Implement and independently review the isolated runner, then run its smoke and replay checks before the one-fold training decision. The concrete plan is `superpowers/plans/2026-10-12-global-readout-ablation.md`. This is development work within the A anchor, not a reference-benefit experiment or a sweep.

The immediate deliverables are a completed control verdict, measured input-pipeline improvements, and a reviewable reference-on launch path. Full Protocol A folds and later protocols depend on that result. Training remains on allocated GPU nodes; CPU-only work uses the CPU partition. Access the cluster through `ssh b00090279@hpc-gpu.aus.edu`.

## Superseded plan retained for history

Cluster: AWS SLURM, account `acc-mialhajri`, partition `gpu`, one A10G (24 GB) per job, 4 CPUs, 8 GB RAM per job. Local GPU for smoke tests: the lab laptop RTX 3080 (16 GB, `aric@192.168.1.202`).

## Cost of one run (to be measured in the smoke test, placeholders until then)

| Item | Estimate | Basis |
|---|---|---|
| Protocol A train set | 1,184 pairs (540 genuine, 644 altered) + 10 percent wrong-design negatives | `counts.md` |
| Batch 8 at 448 x 704 crop, ResNet-18 Siamese, AMP | about 150 iterations per epoch, 0.25 s per iteration on an A10G | measured on the 3080 first |
| 40 epochs with early stopping (patience 8) | 20 to 40 minutes per run, plus 3 minutes evaluation | |
| ChangerEx r18 (Open-CD) | about 1.5x the above | 11.4 M params, FDAF and exchange layers |
| Protocol B train set | 1,981 pairs (SIDTD + IDNet AZ/WV when staged) | 1.7x Protocol A |
| Protocol C | 2 directions x 4 tasks x about 5,000 training pairs | 20 epochs, about 1.5 hours per run |

## Protocol A (approved 10 October, 18 runs, submitted 16:40)

Two architectures times three arms, three seeds each. One network (ChangerEx r18, LEVIR init, fine-tuned end to end), full 512 x 800 canvas, 40 epochs with early stopping, batch 8, AMP. Dropped by decision: the ResNet-18 fallback as a headline arm, the ImageNet-encoder row, the 100-epoch row, the validation grid, the training crop.

| Architecture | Arm | Second input | Jobs |
|---|---|---|---|
| standard (exchange + FDAF) | control | content-free canvas + query zone | 3 |
| standard | method, median | registered design median, other provenance fold | 3 |
| standard | method, specimen | IDNet blank registered to the fold median | 3 |
| improved (+ self/cross-attention at strides 16, 32, global-over-local) | control | content-free canvas + query zone | 3 |
| improved | method, median | registered design median | 3 |
| improved | method, specimen | IDNet blank | 3 |

About 45 minutes of wall time on 18 A10Gs; aggregation by `scripts/aggregate_runs.py --protocol A`.

Optional one-seed ablations, not in the campaign: single genuine card as reference (CheckID's choice), unregistered median, no zone channel.

## Protocol B (priority 3, tomorrow)

A1, A2, A4 and A5 trained on SIDTD (+ IDNet AZ/WV when staged), 3 seeds each: 12 runs, about 1 hour each. Evaluation on the three held-out FantasyID designs with cross-fitted enrollment (fold references by component), native and equalized columns, text, face and all rows; Attack-1 as a miss-rate-only row.

## Protocol C (priority 4, if the cluster has capacity on 12 to 13 October)

A2 and A1 in both directions, 3 seeds: 12 runs, about 1.5 hours each.

## Timing (E10)

CPU, batch 1, FP32, 4 threads, 20 warm-ups, 200 documents, on the cluster login node class CPU and on the Intel Mac; reference features cached per design; reported as ms per document and documents per second, with the registration step timed separately.

## GPU budget

| Day | Jobs | GPU hours |
|---|---|---|
| 10 Oct (today) | Protocol A priority 1, 15 runs | about 12 |
| 11 Oct | Protocol A ablations (9) + Protocol B (12) | about 18 |
| 12 to 13 Oct | Protocol C (12) + reruns | about 20 |
| Total | about 48 runs | about 50 GPU hours |

Eight concurrent A10Gs is the request; four is the minimum that still lands Protocol A today.
