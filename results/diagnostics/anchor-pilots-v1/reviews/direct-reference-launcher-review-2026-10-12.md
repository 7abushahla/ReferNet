# Direct-reference diagnostic launcher review

Date, 12 October 2026.

## Verdict and scope

**Bounded PASS for the wrapper and synthetic orchestration tests.** No blocking contradiction with the Task 4 launcher contract was found. This is not review clearance of `check_direct_reference.py` or its integrated diagnostic engine, actual installed-stack evidence, an allocation, or either long training run.

Reviewed SHA256 values:

| File | SHA256 |
|---|---|
| `scripts/cluster/diagnose_direct_reference.sh` | `3effc5278069e09f61057645b8087b7ed85b0a13c65b19bf72c8e64dae9be680` |
| `tests/test_direct_reference_cluster.py` | `b42f8afc303a4082994ce8d6df89748094e7b6349180d5447924c3841531e009` |

The invoked real stager was inspected at SHA256 `1a2ac07d989ca4b7f2c63993c0efbf60c821fcd3f81859ea6417aec20efd1662`, matching the corrected staging review.

## Verified boundaries

- Headers reuse `train_anchor_pilot.sh` account/allocation/logging settings. They request account `acc-mialhajri`, partition `gpu`, one node/task/GPU, four CPUs, 8 GB, and six hours. Runtime guards require a non-login allocated node, positive numeric job ID, at least four CPUs, GPU allocation evidence, and visible CUDA devices before bootstrap (`diagnose_direct_reference.sh:2`, `diagnose_direct_reference.sh:17`).
- GPU bootstrap remains `/opt/miniconda/etc/profile.d/conda.sh`. The absolute environment `/shared/b00090279/conda/envs/refer-or-clear` is activated, `CONDA_PREFIX` is checked, and its absolute interpreter checks `sys.prefix` with an explicit conditional that survives Python optimization. No environment fallback is present (`diagnose_direct_reference.sh:36`).
- The exact five arguments are `SOURCE_DIR SOURCE_SHA256 REQUEST_JSON REQUEST_SHA256 OUTPUT_ROOT`. Source and request authorities are forwarded unchanged to the reviewed stager. Relative paths are resolved from the submit directory. Scratch defaults to `/scratch/$USER` with `ROC_SCRATCH_ROOT` override; both scratch and output use `direct-reference-diagnostic-$SLURM_JOB_ID`. Existing/protected output rejection and verified immutable copying are delegated to the stager (`diagnose_direct_reference.sh:46`, `diagnose_direct_reference.sh:62`).
- The original frozen request bytes and external SHA are rechecked before deriving the staged official-weight binding. The diagnostic receives that same frozen request and original SHA, not a rewritten request. Model/test imports occur from the staged source, with staged weights and scratch temporary storage (`diagnose_direct_reference.sh:65`, `diagnose_direct_reference.sh:92`).
- Inherited scientific paths and GPU integration flags are cleared. `ROC_ANCHOR_GPU_INTEGRATION=1` is scoped only to the selected actual-stack pytest command, not the stager, JUnit checker, or diagnostic engine. The unused direct-reference integration flag remains unset (`diagnose_direct_reference.sh:36`, `diagnose_direct_reference.sh:101`).
- Pytest selects exactly `test_actual_direct_reference_gpu_integration`. The frozen real stager then checks the named, correctly classed, non-skipped case and consistent counters before diagnostic dispatch. `set -euo pipefail` preserves failures across the logging pipeline. Independent corrected-stager tests accompany the wrapper's mocked orchestration checks (`diagnose_direct_reference.sh:101`, `direct_reference_job.py:174`).
- Final dispatch is only `check_direct_reference.py --request ... --request-sha256 ... --run-dir ... --device cuda`. The wrapper has no train-phase selector, epoch override, automatic chaining, submission, or SSH command (`diagnose_direct_reference.sh:105`). What the diagnostic engine executes remains its separate ongoing review.

## Fresh validation

1. `bash -n scripts/cluster/diagnose_direct_reference.sh`, passed.
2. `PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 /private/tmp/roc-anchor-dev/bin/python -m pytest -p no:cacheprovider tests/test_direct_reference_cluster.py tests/test_direct_reference_job.py -q`, **73 passed in 29.63 seconds**, comprising 51 wrapper and 22 corrected-stager regressions.

The wrapper fixtures mock conda, the absolute interpreter, CUDA probing, pytest, and diagnostic dispatch. They do not execute an actual scientific stack. Interpreter mocks correctly use `command` to bypass the named shell function. The combined stager fixtures use synthetic bytes only. No production images/weights, GPU execution, SSH, jobs, or commits were used. Production and test files were not edited; this review note is the only authored artifact.
