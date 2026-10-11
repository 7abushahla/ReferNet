# Protocol A pilot execution ledger

Plan `docs/superpowers/plans/2026-10-11-classification-centered-anchor.md`.

Starting revision `60a3adf`. Implementation and parallel delegation are authorized. Codex owns integration, commits, pushes and all submissions. External review owns milestone/gate assessment and makes no repository changes. Existing jobs and historical artifacts remain untouched.

## Milestones

| Milestone | State | Evidence |
|---|---|---|
| M0, reviewed amendments and stable interfaces | In progress | D is non-stream-matched; per-track launch path; one coordinator |
| M1-R, raw RGB classifier and loader | Pending | |
| M1-DN, single-query encoder routes | Pending | |
| M1-audit, exact-membership headers | Pending | |
| M2, isolated training and smoke gates | Pending | |
| M3, fold-safe support | Pending | |
| M4, selected-validation reports and common runtime | Pending | |

## Decisions

- The authorized file-boundary workflow uses the existing main working tree, with disjoint write scopes and coordinator-only commits. Do not create competing branches or let workers submit jobs.
- Interface definitions land before workers implement their routes. Reuse the existing selection, scheduler, metrics and loss helpers.
- D compares a new readout with declared historical settings and identical initialization, without historical RNG replay. Its contrast is non-stream-matched.
- Diagnostic scripts stop after overfit, smoke and replay. Each 100-epoch run needs an independent gate review followed by a separate coordinator submission.
- Preliminary local preflight found system Python 3.14 without pytest. No scientific test result is inferred from that environment. The cluster environment must be checked independently.

## Submission ledger

No new jobs submitted. Each entry will record track/phase, source revision and archive SHA256, effective config hash, data-manifest hashes, output directory, resource request, job ID, gate result and independent-review status.
