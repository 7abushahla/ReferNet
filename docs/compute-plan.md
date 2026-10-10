# Compute plan (10 October 2026, deadline 15 October)

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

## Protocol A today (priority 1)

| Arm | Architecture | Fusion | Reference | Seeds | Runs | Answers |
|---|---|---|---|---|---|---|
| A1 | siamese_r18 | single (zone-aware control) | none | 42, 43, 44 | 3 | the control |
| A2 | siamese_r18 | difference | registered median | 42, 43, 44 | 3 | RQ1 (reference helps) |
| A3 | siamese_r18 | difference | blank specimen (IDNet) | 42, 43, 44 | 3 | reference quality |
| A4 | changerex_r18 | difference (FDAF) | registered median | 42, 43, 44 | 3 | the published architecture |
| A5 | changerex_r18 | single | none | 42, 43, 44 | 3 | its control |

15 runs, one GPU each, about 40 minutes each: with 8 concurrent A10Gs the sweep finishes in about 1.5 hours of wall time; with 4, in about 3 hours. Submit A1 and A2 first (6 jobs), then A3, then A4 and A5 once the Open-CD environment passes `verify_gpu.sh`.

## Ablations on Protocol A (priority 2, one seed unless marked)

| Arm | Change | Runs |
|---|---|---|
| A6 | fusion stacked and attention, median reference | 2 (3 seeds for attention if time: 6) |
| A7 | no zone channel (model and control) | 2 |
| A8 | reference = single genuine card of another component (the CheckID choice) | 1 |
| A9 | unregistered median (negative control) | 1 |
| A10 | median-trained model evaluated with the specimen and vice versa (inference-only swap, no training) | 0 |
| A11 | init scratch vs ImageNet vs LEVIR (ChangerEx) | 2 |

About 9 runs, 6 hours of GPU time.

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
