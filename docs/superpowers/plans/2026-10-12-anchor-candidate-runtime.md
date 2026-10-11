# Candidate-Aware Anchor Runtime Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans task by task with isolated file ownership and coordinator review. Preparation does not authorize measurements or submissions. No commits, pushes, SSH, launches, model imports, or production-code changes are part of this planning task.

**Goal:** Prepare an additive runner for preliminary CPU measurements of completed R, D_running_pilot, and N candidates without replacing their scientific sources or weakening the original R/D/N runner.

**Architecture:** A measurement-side coordinator freezes explicit candidate bindings and a shared measurement manifest. Fresh subprocesses load each candidate exclusively through its own immutable scientific source. Reviewed measurement-side helpers provide inspection and timing, not replacement model, loader, configuration, or checkpoint semantics.

**Tech Stack:** Existing Python 3.10 project environment, PyTorch/Open-CD, NumPy/OpenCV, pytest, and the existing allocated-CPU Slurm pattern. No dependency upgrades.

**Spec:** `docs/superpowers/specs/2026-10-11-classification-centered-anchor-design.md`, especially selected-checkpoint evidence, preliminary CPU timing, and the approved D_running_pilot continuation. Preserve Task 7 of `docs/superpowers/plans/2026-10-11-classification-centered-anchor.md` and `docs/superpowers/plans/2026-10-12-d-running-pilot.md`.

## Observed Constraints

- `scripts/bench_anchor_pilots.py` requires exactly three runs in R/D/N order. `prepare_runs` requires each candidate's scientific `source_sha256` to equal the imported runtime's `source_sha256()`, then requires common records, split, and scientific-source identities. Its `validate_measurement` and the old report consumer also require R/D/N. Do not remove these checks or extend the old CLI implicitly.
- R job 37967 and N job 37971 use scientific revision `0959335bd9a6a22510f6706c49c62d84560cd524`. R's local metadata records 100 completed epochs, selected epoch 55, and passing selected/replay gates. This planning inspection did not load or independently rehash its checkpoint bytes.
- D_running_pilot job 37982 uses scientific revision `bef7544a64f36a166f76eeef0d3130e353bd3203`, identifier `anchor_direct_running_ia18_pilot_v1`. Original D37963 failed overfit and is not a substitute candidate. Bounded `D_running` and its smoke checkpoints are also ineligible.
- N and D_running_pilot are still training according to the coordinator's current request. The execution ledger's scheduler observations are historical snapshots, not a live completion check. No selected outcome or common runtime result is claimed for either.
- The local 095 snapshot has the required `refer_or_clear.pilots` helpers but no `scripts/report_anchor_pilots.py`. Measurement-side inspection must therefore be separately versioned, not assumed to exist inside every scientific snapshot.
- Existing benchmark imports eagerly load scientific modules and adjust `sys.path`. Importing it before establishing the worker's scientific package would risk measuring current code against an old checkpoint. A fresh process and import-origin checks are mandatory.

## Global Constraints

- Implementation preparation may proceed while the training jobs continue. It must not inspect changing checkpoints, run models, decode timing inputs, or gate/interfere with those jobs.
- Full measurement requires all three requested candidates to have completed, immutable, selected-validation evidence and independent completion review. Missing or failed completion produces explicit unmeasured status, never partial three-way comparability.
- Preserve actual track names and architecture identifiers. Never rename D_running_pilot to D, rewrite checkpoint identities, or require different scientific sources to hash equally.
- Keep source/configuration/weights/selection identities per candidate. Require common official records/split hashes and the same first 200 training IDs, not common architecture/configuration/source hashes.
- Timing uses training documents only. No validation/test pixel access, dense targets, references, support regeneration, training, optimization, model selection, compilation, quantization, or reference-architecture extension.
- Report compound-recipe preliminary CPU cost, not causal backbone superiority, support-safe geometry, officer-device acceptance, or a winner. Full geometry and cached/uncached reference timing remain separate later requirements.

## Pinned Measurement Contract

Use schema/contract name `anchor_candidate_cpu_v1`, with `base_timing_contract=anchor_cpu_v1` and fixed actual execution order `[R, D_running_pilot, N]`. Only the contract name and track-order identifiers differ from the existing `CONTRACT`. A regression must compare all remaining fields exactly against `scripts/bench_anchor_pilots.py::CONTRACT`.

| Field | Unchanged requirement |
| --- | --- |
| Membership | Official Protocol A, fold 0, first 200 lexicographically sorted IDs from the full 1,778 training-image membership; freeze ordered IDs and canonical SHA256 |
| Host | Same physical allocated host, boot ID, CPU model, affinity, job ID, OS and library/environment identity throughout all workers |
| Resources | One CPU allocation, at least four allocated CPUs and four affinity entries; no GPU or concurrent benchmark workload |
| Threads | Torch intra-op 4, inter-op 1; OpenCV 1; OMP/MKL 4, established before model execution |
| Execution | Batch 1, CPU FP32, eager `torch.inference_mode()`, AMP/TF32 off; no altered architecture or normalization policy |
| Warm-up/repetitions | First 20 selected documents as untimed warm-ups per candidate; one timed pass over all 200 in their frozen order |
| Preparation boundary | Declared source read, decode and deterministic augmentation-disabled transform; no upstream geometry generation |
| Network boundary | Ready tensor forward through attack-score scalar materialization, including `.item()` |
| Statistics | Per-image preparation/network/prepared-input-total milliseconds; median and NumPy p95; total equals preparation plus network |
| Inputs | R uses native raw RGB, `[1,3,299,299]`; D_running_pilot/N use existing registered canvas plus zone, `[1,4,512,800]` |
| Scope labels | R `raw_input`; D_running_pilot/N `processed_input_only`; these totals are not full end-to-end document processing |

Retain the existing source-artifact preparation and warm-up behavior. Do not flush caches or label these readings uncached. Source inventory construction already reads/decodes artifacts before timing. No performance interpretation may hide this boundary.

R preparation includes native RGB read/decode, resize to 299 square, and its declared normalization. D_running_pilot/N preparation starts at precomputed registered canvases and zones. Offline source-to-canvas rectification/registration, geometry-target construction and zone preparation are excluded, not assigned zero cost. Reference retrieval, preparation, registration/alignment and cached/uncached reference handling are also unmeasured. These measurements must never be labeled full source-to-decision deployment latency. A common host controls measurement conditions; it does not make different input pipelines, recipes or model families a backbone-only comparison.

## Unresolved Hardware and Execution Facts

- The future CPU allocation's node, physical CPU model, boot ID, affinity, frequency behavior, SMT/NUMA placement and co-resident workload are unknown. GPU training-node names are not CPU-benchmark hardware authority.
- An allocation of four CPUs is not evidence of whole-node exclusivity. Record the actual allocation and any available exclusivity/placement evidence. The runner can prohibit its own concurrent workers, not certify absence of unrelated tenants. Disclose that limitation.
- The designated officer desktop CPU and its deployment environment have not been supplied. The p95-under-one-second end-to-end target cannot be evaluated by this preliminary cluster measurement.
- Existing CPU wrapper requests 16 GiB and two hours. Adequacy for all three CPU forwards is not established here. An OOM/time-limit failure produces incomplete, non-comparable evidence; review resource needs rather than changing batch, precision, sample count or repetitions.
- Selected checkpoint availability, completion-review hashes and final N/D_running_pilot artifact hashes remain pending. Resolve exact paths from completed run authorities, never from a glob selecting the newest directory.

## Isolated Files

Create only these implementation files after plan approval:

- `scripts/bench_anchor_candidates.py`, stdlib-only metadata preparation/coordinator and new-schema validation. Importing it must not import Torch, OpenCV or `refer_or_clear`.
- `scripts/bench_anchor_candidate_worker.py`, allocated-only fresh-process scientific import, candidate inspection/preparation, and one-candidate timing.
- `scripts/cluster/bench_anchor_candidates.sh`, one allocated CPU wrapper, copied from the existing benchmark wrapper's resource/environment pattern.
- `tests/test_anchor_candidate_runtime.py`, synthetic binding, worker-isolation, timing and aggregation regressions.
- `tests/test_anchor_candidate_cluster.py`, synthetic shell/environment/argument/staging regressions.

Leave `scripts/bench_anchor_pilots.py`, `scripts/report_anchor_pilots.py`, their tests/wrapper, all `src/`, configurations, training artifacts and scientific snapshots unchanged. Do not attach the new schema to the old report consumer in this increment.

## Review Focus

1. Different scientific revisions must remain independently authenticated, not coerced into one source or substituted through an already imported package.
2. Incomplete runs, failed original D, bounded diagnostics and mismatched checkpoint epochs must not start timing or masquerade as completed candidates.
3. Shared IDs must be the same actual first 200 training IDs, not merely a self-consistent rewritten timing manifest.
4. Source fallback, target access, normalization-buffer mutation and timing-boundary drift must fail closed.
5. Partial workers, changed hosts, output traversal/symlinks and non-CPU allocations must not yield comparable evidence or mutate inputs.

## Task 1. Explicit Candidate Bindings and Metadata Readiness

**Files:** New coordinator and `tests/test_anchor_candidate_runtime.py`.

**Interfaces:** `prepare_bindings(catalogue: dict) -> dict`; CLI `--phase prepare|measure --candidates PATH --candidates-sha256 SHA --run-dir NEW_PATH`. The wrapper supplies a separately authenticated measurement-tool directory. No model-bearing import occurs in the preparation path.

- [ ] **RED.** Add synthetic tests accepting R095/N095/D_running_pilot-bef as distinct identities, rejecting a swapped source, rewritten config/track, original-D probe, duplicate candidates, forbidden roles, foreign/not-first-200 IDs and a changed externally hashed catalogue. Assert unfinished candidates return `status=waiting-for-completion`, `comparable=false`, no durations, and no worker invocation. A malformed or failed binding returns `status=not-comparable`.
- [ ] Run `OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /private/tmp/roc-anchor-dev/bin/python -m pytest -p no:cacheprovider tests/test_anchor_candidate_runtime.py -q` and observe the expected missing implementation failures.
- [ ] **GREEN.** Define catalogue schema `anchor_candidate_bindings_v1`. Each entry supplies exact track, architecture ID, run/parent-manifest paths and their external hashes, scientific revision/archive hash, scientific source directory, original identities, original-weight path/hash, records/split paths/hashes, declared raw/legacy roots, selected checkpoint/hash and selected-evidence/review hashes. Pending entries explicitly omit unavailable selected authorities; they cannot authorize execution. Measurement tooling has its own reviewed file-hash inventory, separate from every scientific identity.
- [ ] Validate duplicate-free JSON and safe absolute paths. Freeze common ordered IDs from complete training metadata, requiring full membership, matching labels/design/component/path metadata and common records/split hashes. Do not decode inputs or open changing checkpoints during `prepare`. A readiness result is metadata-only, not selected-checkpoint verification.
- [ ] Resolve missing staged paths only through an explicitly bound parent manifest or explicit hash-checked relocation. Never substitute raw inputs with canvases, infer latest checkpoints, rewrite stored paths/identities, or regenerate legacy products. Before writing, reject symlinks on the original requested output path, resolve its actual destination, then protect every input/source/run/data tree and existing artifact. Use exclusive creation.
- [ ] Rerun the focused tests. Exit 2 for waiting, 1 for invalid/non-comparable, 0 for metadata readiness; readiness never means measured.

## Task 2. Per-Candidate Scientific Worker and Shared Manifest

**Files:** New worker/coordinator and runtime tests.

**Interfaces:** Worker CLI `--operation inspect|prepare|measure --request PATH --request-sha256 SHA --out NEW_PATH`. `inspect` validates selected evidence/checkpoint without decoding images or constructing a model; `prepare` inventories the 200 authorized input artifacts; `measure` consumes the frozen shared runtime manifest/hash. Coordinator calls workers synchronously with the exact environment interpreter, never a subagent or concurrent process pool.

- [ ] **RED.** Add two synthetic scientific packages with deliberately different builders/contracts. Prove each fresh subprocess uses its own package, despite a misleading measurement-side `src` tree or poisoned inherited `PYTHONPATH`. Assert a worker fails on import-origin drift, unknown architecture, source hash mismatch, wrong checkpoint/config/identity/epoch/key, missing completion review and mutated source/weights. Prove that an absent report script inside historical source 095 is accepted when the separately authenticated measurement-side inspector is available, without changing scientific imports or synthesizing historical files. Missing or hash-mismatched measurement-side tooling must fail. Assert no preparation/model construction occurs before all candidates pass inspection.
- [ ] Run the new subprocess regressions and observe failures before implementing worker isolation.
- [ ] **GREEN.** Authenticate the measurement script inventory and each scientific source inventory before imports. Recompute the scientific runtime digest using the existing convention, all `src/**/*.py` plus `scripts/check_anchor_pilots.py`, and compare with that candidate's unchanged ledger. Keep archive/source-inventory hashes distinct from this runtime digest.
- [ ] Start a new interpreter for every worker operation. Clear inherited `PYTHONPATH`, insert only that candidate's scientific `src` first, and preload its `refer_or_clear` package, contract, pilots and query loader. Load the separately hashed measurement-side `report_anchor_pilots.py` and `bench_anchor_pilots.py` only afterward. Their path insertion must not replace the scientific package. Assert the package search path and all loaded `refer_or_clear.*` origins remain inside the candidate snapshot before and after inspection/model construction.
- [ ] Reuse measurement-side `inspect_run(run_dir, actual_track)` for complete selected-validation checks, using the preloaded candidate-specific scientific helpers. It must preserve its accuracy-first/AUC-second/earliest-tie verification, full 100-epoch evidence, selected gate/replay, initialization and checkpoint validation. Reuse frozen `read_pilot_checkpoint`, `build_initialized_model`, `load_role_rows`, `AnchorQueryDataset`, and `contract_for`; never reconstruct architecture or preprocessing in the sidecar. Independent review records bind final selected evidence, not just pretraining launch approval. Failing acceptance targets do not by themselves invalidate a completed candidate's timing eligibility.
- [ ] After **all** inspections pass, prepare source inventories using existing `runtime_train_rows`, `source_artifacts` and metadata/weight resolution helpers. Recheck the first 200 training IDs and paths against each ledger. Preserve the current augmentation-disabled evaluation view of those training rows, including the loader's internal `role=val` view convention, while recording their true membership as `train`; never call `load_role_rows(..., 'val')` for timing. N receives no masks/valid targets or references.
- [ ] Record host/thread/precision identity, input hashes/shapes, selected epoch/key, parameter counts, checkpoint bytes, per-run scientific inventories/identities and separate measurement-tool hashes. Freeze the shared `anchor_candidate_cpu_v1` manifest and digest before any warm-up. No new wrapper may call old `prepare_runs`, `main` or `validate_measurement` to bypass their R/D/N constraints.
- [ ] Rerun the subprocess/binding tests. Verify old `tests/test_anchor_runtime.py` still passes unchanged.

## Task 3. Timing, Immutability and Explicit Comparability

**Files:** New worker/coordinator and runtime tests.

**Interfaces:** Worker returns per-candidate durations and frozen provenance; `validate_candidate_measurement(measurement: dict, bindings: dict) -> None` checks the new schema only.

- [ ] **RED.** Use the existing fake-clock pattern to assert 220 score materializations, exactly 20 untimed warm-ups, 200 timed records, and scalar materialization inside network timing. Test wrong shape/precision/mode, changed host/affinity/environment, mismatched IDs/source mapping, nonfinite/negative durations, bad total/percentiles, mutated weights/buffers, and failure of a later worker after an earlier one succeeded. Partial data must remain non-comparable.
- [ ] Run the new timing/aggregation regressions and observe failures.
- [ ] **GREEN.** Reuse `configure_threads`, `host_info`, `validate_host`, `measure_track`, `stats`, and source-artifact verification from the authenticated measurement-side benchmark module. Use each worker's own scientific builder and strict checkpoint load. Preserve initialized-tensor provenance checks, actual parameter counts and the candidate's public forward/normalization policy. Place the model on CPU/FP32 and set evaluation mode without replacing its buffers.
- [ ] Run R, D_running_pilot, N serially in the same CPU allocation. Before/after each candidate, verify common host/boot/affinity/thread/precision/environment identity and rehash immutable tooling, source, checkpoint, evidence, metadata and timing input artifacts. Hash model parameters/buffers before warm-up and after timing. No backward, optimizer, buffer update, saved training checkpoint or cached-image substitution is permitted.
- [ ] Validate results against the shared manifest and original candidate bindings, not only worker-supplied fields. Recompute statistics and source/ID relationships. Do not compare each scientific source hash with the measurement-source hash or with another candidate's scientific hash.
- [ ] Produce `runtime-manifest.json`, `runtime.json`, `runtime.md`, per-worker request/result/log files, and separate measurement/per-candidate scientific provenance inventories. The aggregate is `status=measured, comparable=true` only when all three complete and revalidate. Otherwise record affected candidates/reasons with `comparable=false`; any retained partial durations are labeled diagnostic-only and excluded from the comparison table. Include `winner=null`, `family_selection=not-authorized`, `detector_clearance=false`, and the raw-versus-processed-input limitations. Add per-candidate `excluded_or_unmeasured_costs` entries for offline canvas/zone preparation and reference retrieval/preparation/alignment/cache handling, using `not_measured` or `not_applicable_to_query_only_candidate`, never numeric zero. Mark `full_source_to_decision_latency=not_measured` and reproduce these exclusions alongside the Markdown timing table.
- [ ] Rerun new and unchanged timing tests. Synthetic results are not actual CPU measurements.

## Task 4. Allocated CPU Wrapper and Review Gates

**Files:** New CPU wrapper and `tests/test_anchor_candidate_cluster.py`.

**Interface:** `scripts/cluster/bench_anchor_candidates.sh MEASUREMENT_SOURCE_DIR CANDIDATES_JSON CANDIDATES_SHA256 OUTPUT_ROOT`. This wrapper executes one measurement allocation; it does not submit, poll training jobs or schedule a later measurement automatically.

- [ ] **RED.** Add synthetic wrapper tests rejecting missing Slurm metadata, login/head/frontend/lgn nodes, invalid job IDs, non-`cpu` partitions, fewer than four CPUs, wrong environment prefix, bad catalogue/tool hashes and existing/overlapping outputs before worker/model execution. Cover relative output/scratch normalization, symlink/traversal protection, quoted paths with spaces, missing candidate completion, worker failure propagation, serial order and no GPU reservation/submission commands.
- [ ] Run `python -m pytest tests/test_anchor_candidate_cluster.py -q` and observe failures before creating the wrapper.
- [ ] **GREEN.** Copy `scripts/cluster/bench_anchor_pilots.sh` resource pattern, account `acc-mialhajri`, partition `cpu`, one node/task, four CPUs, 16 GiB, two hours, no GPU. Reuse `/opt/miniconda3/etc/profile.d/conda.sh`, absolute activation `/shared/b00090279/conda/envs/refer-or-clear`, and that prefix's `bin/python`. Enforce the resolved prefix, CPU partition and allocation guards before heavy work.
- [ ] Reuse `/tmp/$USER` scratch by default with quoted `ROC_SCRATCH_ROOT` override, isolated TMPDIR, bytecode disabled, hash seed 42, OMP/MKL 4, TF32 overrides off, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, and empty `CUDA_VISIBLE_DEVICES`. Stage only the authenticated measurement script inventory separately from scientific snapshots. Exclude `._*`; never merge measurement code into a scientific snapshot or set measurement `src` as the scientific `PYTHONPATH`.
- [ ] Snapshot environment, wrapper/tool hashes, allocation, candidate catalogue hash and immutable relocation bindings before inspection. Reference or hash-verify readonly per-candidate scientific sources/checkpoints and stage only the pinned training inputs needed for timing. Preserve original candidate files and all existing GPU jobs.
- [ ] Verify `bash -n scripts/cluster/bench_anchor_candidates.sh`; run new runtime/wrapper tests plus unchanged `tests/test_anchor_runtime.py` and `tests/test_anchor_report.py`. Local checks use synthetic fixtures only. Record unrelated failures without repairing other workers' files.
- [ ] Independent implementation review checks import isolation, 095 inspector compatibility, completion gating, unchanged old semantics, contract parity and output protection. Coordinator then verifies completed selected evidence/reviews for every candidate and resolves hardware/allocation facts before any measurement authorization. Preparation alone does not authorize a launch.
- [ ] After a separately authorized CPU allocation, independent actual-evidence review verifies host consistency, all 600 timed records, 60 warm-ups, immutable fingerprints, selected checkpoint bindings and reconstructed statistics. Missing/failed evidence stays unavailable. Do not append a winner, unlock reference replication, or interpret a cluster p95 as desktop acceptance.

## Handoff and Stop Boundary

This document authorizes no code execution or submission itself. Implementation is additive and bounded to the five new files above after coordinator review. Keep readiness evidence separate from the immutable manifest for an actual measurement. Do not change scientific recipes, original checkpoints, the failed-D outcome, old benchmark/report behavior, running jobs, or the reference architecture.
