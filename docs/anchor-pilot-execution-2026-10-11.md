# Protocol A pilot execution ledger

Plan `docs/superpowers/plans/2026-10-11-classification-centered-anchor.md`.

Starting revision `60a3adf`. Implementation and parallel delegation are authorized. Codex owns integration, commits, pushes and all submissions. External review owns milestone/gate assessment and makes no repository changes. Existing jobs and historical artifacts remain untouched.

## Milestones

| Milestone | State | Evidence |
|---|---|---|
| M0, reviewed amendments and stable interfaces | Committed and pushed | `61833c9`; three contract tests passed; D is non-stream-matched; per-track launch path; one coordinator |
| M1-R, raw RGB classifier and loader | Implemented, review in progress | R historical BN-counter compatibility finding is being corrected; loader/configuration focused tests pass |
| M1-DN, single-query encoder routes | Implemented, integration pending | Worker reports fixture tests; actual installed Open-CD/LEVIR checks require an allocation |
| M1-audit, exact-membership headers | Implemented, independent review in progress | 19 focused tests; Claude read-only review requested; real-corpus audit not yet run |
| M2, isolated training and smoke gates | Pending | |
| M3, fold-safe support | Pending | |
| M4, selected-validation reports and common runtime | Pending | |

## Decisions

- The authorized file-boundary workflow uses the existing main working tree, with disjoint write scopes and coordinator-only commits. Do not create competing branches or let workers submit jobs.
- Interface definitions land before workers implement their routes. Reuse the existing selection, scheduler, metrics and loss helpers.
- D compares a new readout with declared historical settings and identical initialization, without historical RNG replay. Its contrast is non-stream-matched.
- Diagnostic scripts stop after overfit, smoke and replay. Each 100-epoch run needs an independent gate review followed by a separate coordinator submission.
- Preliminary local preflight found system Python 3.14 without pytest. No scientific test result is inferred from that environment. The cluster environment must be checked independently.
- Execution workers use `gpt-6.1-sol` at the user's request to limit cost. The coordinator retains integration, review routing, commits and submissions. Independent write scopes cover R, D/N, data/configuration, runtime/gates, enrollment and packaging. Completed scopes release their agent slot for review or the next bounded task.
- Ruling: configuration resolution lives in `pilot_config.py` and is imported by `pilots.py`. This separates the data/configuration worker from the runtime worker without changing the public contract.
- Local verification uses `/private/tmp/roc-anchor-dev/bin/python`, Python 3.10.20, torch 2.1.2, torchvision 0.16.2 and pytest 9.1.1. The first combined contract/R/header-audit run passed 37 tests with one TypedStorage deprecation warning. R tests use tiny fixtures, not official weights. These results do not clear GPU gates.
- Always access the cluster with `ssh b00090279@hpc-gpu.aus.edu`, as explicitly required by the user. Use the same hostname for file transfers. Do not substitute historical or resolved IP addresses when DNS fails; report the connectivity failure instead. Slurm executables are under `/opt/slurm/bin`. The active environment is `/shared/b00090279/conda/envs/refer-or-clear`; its pytest version check returned 9.1.1. No model inference or training is performed on the login node.
- The staged LEVIR checkpoint SHA256 is `da3f569306dadd1fac5b64abc2a3d484571cd0ecf6410e1f152275cfe49a3618`. Official R weights still need staging and real-weight validation. Installed Open-CD encoder/decoder source was copied read-only for implementation inspection.
- Update: official torchvision R50 V1 weights were downloaded from `https://download.pytorch.org/models/resnet50-0676ba61.pth` to the cluster's `checkpoints/pretrained` directory. Their SHA256 is `0676ba61b6795bbe1773cffd859882e5e297624d384b6993f7c9e683e722fb8a`. Download/hash verification is not model validation. The R review identified an overly strict rejection of historical checkpoints without BN counters; the correction must be explicitly allowlisted and recorded, not silently fill arbitrary missing weights.
- Coordinator verification of the new data/configuration tests returned 85 passed. The actual local fold-0 metadata join returned 1,778 training images (808 genuine, 970 altered) and 222 validation images (100 genuine, 122 altered), without opening image pixels.
- Claude is authorized as an independent adviser/reviewer. The named `$claude` skill was not exposed in this session, so the available Claude CLI was used for a read-only review of the completed header-audit files, with only Read/Grep/Glob tools, no MCP servers, Sonnet and a USD 3 budget cap. Findings remain subject to coordinator verification. Claude does not edit or submit jobs.

## Submission ledger

No new jobs submitted. Each entry will record track/phase, source revision and archive SHA256, effective config hash, data-manifest hashes, output directory, resource request, job ID, gate result and independent-review status.
