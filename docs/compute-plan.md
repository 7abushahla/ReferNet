# Compute plan (10 October 2026, deadline 15 October)

> Historical plan. The estimates and Protocol A–D definitions below were superseded. For current execution, use `anchor-pilot-execution-2026-10-11.md` and the 12 October priorities immediately below. Do not submit the historical campaigns or use their runtime estimates.

## Active priorities, 12 October

1. Complete the existing direct-reference OFF pilot, job 38041. It uses the official fold-zero train/validation population, 100 epochs, and the current direct classifier with its local query adapter. At 13:45 UTC, 38 epochs had completed. The approximate remaining time is six hours at the observed cadence, subject to actual progress. Inspect the selected validation checkpoint, both attack subsets, genuine false positives, context availability, and fresh-process replay when complete. The agreed competence criteria remain selected accuracy at least 0.97 and AUC at least 0.99.
2. Use the waiting interval to reduce repeat input processing for subsequent experiments. The completed 38062 benchmark measured 6.437 end-to-end images/s at batch eight, versus 36.736 images/s for a preloaded batch. Batch 32 improved end-to-end speed by only 9.36 percent. The current loader correction does not justify restarting 38041. The next bounded correction caches validated immutable binary support masks while authenticating their bytes on every access. Preserve the image tensors, augmentation stream, sample order, batch size, and model recipe. Check equivalence and measure any gain before using it for training.
3. Prepare reference-on execution while the control trains. Reuse the same scientific implementation, original initialization, official membership, support products, and registration as the control. Independently rebuilding a registration table in a new allocation is not automatically a matched comparison because the present registration key includes path-bound preprocessing. Resolve snapshot/restaging and selected-checkpoint replay before launch. Preparation does not waive the agreed completed-control gate or authorize ON100 or a sweep.
4. Prepare the remaining nine folds' support banks independently on CPU nodes. The existing builder already supports all ten folds. Preserve genuine-training-only eligibility and the original split sizes, including fold four's 1776/223/223 and fold nine's 1777/222/223 train/validation/test membership. Metadata preflight found nonempty design halves in every fold. This data preparation does not launch a training sweep. Detailed counts are in `../evidence/protocol-a-support-preflight-2026-10-12.md`.
5. Produce a same-day decision from the completed control. If it clears the agreed gate, the next scientific experiment is the matched reference-on pilot. If it does not, use its completed validation evidence to identify the remaining limitation before a sweep. Prepare collection and comparison scripts during training so result interpretation adds minutes rather than another implementation cycle.

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
