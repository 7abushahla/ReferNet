# Anchor candidate runtime Task 3 review

## Scope and disposition

Independent bounded review of the uncommitted `scripts/bench_anchor_candidates.py`, new `scripts/bench_anchor_candidate_worker.py`, and `tests/test_anchor_candidate_runtime.py` against Tasks 2 and 3 of `docs/superpowers/plans/2026-10-12-anchor-candidate-runtime.md`.

One concrete P2 finding remains. Task 3 is not cleared by this review. No P0 or P1 finding was identified within this scope. This is not an implementation-completion claim, a measurement authorization, or a clearance of actual candidate execution.

Production files, architecture documentation, scientific models, and existing tests were not changed. No SSH, cluster submissions, real model construction, real checkpoint loading, or image decoding occurred. The focused reproduction used the existing synthetic fixtures and temporary synthetic artifacts only.

## P2. Preserve the non-comparable report when a parent authority changes

**Location.** `scripts/bench_anchor_candidates.py:921`, specifically the authenticated parent read at line 923, outside the failure handler at lines 915–919 and before the report writes at lines 925–927.

**Failure condition.** An immutable parent manifest changes or becomes unavailable during measurement. Runtime revalidation correctly rejects the change. The exception handler constructs a non-comparable result. Publication then rereads every parent manifest to build `scientific-source-inventories.json`. That read encounters the same failed authority and raises again, before either `runtime.json` or `runtime.md` is written.

**Observed evidence.** A focused synthetic reproduction reused `bench`, `worker`, `catalogue`, `runtime_catalogue`, `timing_setup`, and `measurement_runner` from the current test file. Immediately after the synthetic R measurement worker wrote its receipt, the runner appended a newline to the synthetic R parent manifest without updating its bound hash. The coordinator produced the following observations.

```text
RAISED: ValueError scientific parent SHA256 mismatch: /private/tmp/roc-task3-review-dhdy5bst/run-R/input_manifest.json
runtime.json exists: False
runtime.md exists: False
R measured receipt exists: True
imported scientific production package: False
```

The reproduction ran with `OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /private/tmp/roc-anchor-dev/bin/python` and used no production scientific imports. Its artifacts remain under `/private/tmp/roc-task3-review-dhdy5bst`.

**Impact.** This does not incorrectly publish `comparable=true`. It prevents the explicit aggregate failure record required by Task 3, leaving worker artifacts without the required non-comparable report and potentially replacing the original failure with a secondary inventory-publication error. Parent loss is precisely an immutability failure the coordinator must be able to report.

**Required correction.** Capture the authenticated per-candidate inventories before execution, or make optional inventory publication independently failure-safe. Always persist `runtime.json` and `runtime.md` with `status=not-comparable`, `comparable=false`, the original failure reason, and any previously accepted durations confined to `diagnostic_only`. Do not authenticate changed parent bytes as replacement provenance.

**Test gap.** `tests/test_anchor_candidate_runtime.py:1525` covers a later worker exception while all authority files remain readable and unchanged. `tests/test_anchor_candidate_runtime.py:1549` covers request mutation, not parent-authority loss during report publication. Add synthetic cases for parent hash drift and disappearance after a worker returns, including a later-worker failure after R succeeds. Require both failure reports to exist and forbid comparison-table rows.

## Bounded contract checks

- Measurement CLI is enabled at `scripts/bench_anchor_candidates.py:940`. It calls the allocated measurement path rather than the metadata-only path. Missing completion bindings return before worker invocation. Complete metadata alone does not bypass scientific inspection.
- The coordinator completes all three inspections before preparation at `scripts/bench_anchor_candidates.py:793`. Worker preparation and measurement require the three-inspection bundle at `scripts/bench_anchor_candidate_worker.py:292`.
- Fresh isolated worker interpreters, candidate-specific scientific origins, separately authenticated tooling, and unchanged checkpoint/configuration/evidence identities remain enforced. Scientific-source hashes are not required to equal one another or the measurement-tool hash.
- Preparation and measurement select the first 200 training rows. The internal `role=val` evaluation view is applied only after training membership checks. N receives the augmentation-disabled query-only view, not training masks, valid targets, or references.
- The worker reuses the existing timer. The current synthetic timer regression checks 20 untimed warm-ups, 200 timed records, 220 score materializations, and scalar materialization inside the network boundary.
- CPU FP32, batch one, inference/evaluation mode, Torch 4/1 threads, OpenCV 1, common host identity, and parameter/buffer fingerprints are checked before and after timing. The aggregate revalidates authorities, frozen inputs, durations, totals, percentiles, requests, and receipts.
- Ordinary later-worker failure remains non-comparable, with partial durations excluded from comparison tables. Actual `D_running_pilot` identifiers remain distinct from legacy D. The report retains raw-versus-processed input boundaries and does not authorize a winner, detector clearance, family selection, or full source-to-decision latency.

These checks are bounded static review and synthetic evidence, not verification of actual CPU models, checkpoints, images, allocation exclusivity, or deployment performance.

## Validation state

The coordinator reports fresh validation of 212 tests in 166.60 seconds across `tests/test_anchor_candidate_runtime.py`, `tests/test_anchor_runtime.py`, and `tests/test_anchor_report.py`, plus 46 cluster tests in 4.84 seconds and successful `bash -n`. Those suites were not duplicated by this reviewer. The initial incorrect-filename invocation was corrected and is not treated as a test failure. The focused authority-drift reproduction above exposes a coverage gap despite the reported passing suites.

Reviewed file SHA256 values remained unchanged across the audit.

| File | SHA256 |
| --- | --- |
| `scripts/bench_anchor_candidates.py` | `527d95170ca5a5a9b2aff9c31a3d808075b0b39ccb53eeb977dab6d5459d5a58` |
| `scripts/bench_anchor_candidate_worker.py` | `abd8d28dbe50d88baad73b68418cbaea5e9a50b02674da348da46b61e85ba456` |
| `tests/test_anchor_candidate_runtime.py` | `db002b23ab9f96413c73a5d8558c0d7d692c78f967733bb5c931a3d18ada0e0b` |

## Narrow re-review after the coordinator fix

**Disposition.** The original inventory-publication exception is corrected. Clearance remains pending because the independently rerun focused regression has one failure concerning retention of the authority-error reason. The original finding above is preserved as historical evidence. No production or test edits were made by this reviewer.

The reviewed coordinator now authenticates and caches the original scientific inventories inside the protected block at `scripts/bench_anchor_candidates.py:888`, before measurement. Publication at `scripts/bench_anchor_candidates.py:926` uses that cache and does not reread parent authorities. Measurement revalidation remains active and does not accept a changed parent as replacement provenance.

The independent command was

```text
OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /private/tmp/roc-anchor-dev/bin/python -m pytest -p no:cacheprovider -q tests/test_anchor_candidate_runtime.py -k parent_authority_loss
```

Its result was `1 failed, 3 passed, 165 deselected in 30.38s`, with exit code 1. The failing case was `test_parent_authority_loss_preserves_failure_reports[R-drift]`, at `tests/test_anchor_candidate_runtime.py:1570`. It expected the original SHA256-mismatch reason in the aggregate errors.

**Remaining P2 diagnostic issue.** `scripts/bench_anchor_candidates.py:633` reduces the failed readiness result to `all completed selected reviews required`, discarding its detailed authority errors. Consequently, R parent drift and disappearance both persist that generic message rather than the actual SHA256 mismatch or missing-parent reason. This masking is in runtime revalidation, not the repaired publication code. Preserve the underlying readiness errors in the propagated failure. Do not weaken the new drift assertion merely to accept the generic message. The missing-parent case also needs an explicit original-cause assertion.

After the focused run, this reviewer independently inspected all four generated JSON reports, Markdown reports, and cached inventories under `/private/var/folders/sd/hj30l0b96zj8lbk9ktvcxvpr0000gn/T/pytest-of-hamza/pytest-272`. The observations were

| Case | Both reports exist | Aggregate | Diagnostic-only tracks | Error retained in JSON and Markdown |
| --- | --- | --- | --- | --- |
| R return, parent drift | Yes | `not-comparable`, `comparable=false`, empty `tracks` | None | Generic readiness message, original cause lost |
| R return, parent missing | Yes | `not-comparable`, `comparable=false`, empty `tracks` | None | Generic readiness message, original cause lost |
| Later D exception, parent drift | Yes | `not-comparable`, `comparable=false`, empty `tracks` | R only | Original synthetic later-worker exception |
| Later D exception, parent missing | Yes | `not-comparable`, `comparable=false`, empty `tracks` | R only | Original synthetic later-worker exception |

All four Markdown reports omit the comparison table. For every candidate in every case, the canonical hash of the published cached `source_files` equals the original externally bound `scientific_source.inventory_sha256`. The cache therefore retains original authenticated inventories, not replacement parent provenance. No N measurement is attempted in these failures.

Reviewed snapshot identities for this narrow re-review were

| File | SHA256 |
| --- | --- |
| `scripts/bench_anchor_candidates.py` | `b13b2b60d1c01f5a197410aa0b9c2ce74a5f68f4725f597152ee036a3d6430a1` |
| `scripts/bench_anchor_candidate_worker.py` | `abd8d28dbe50d88baad73b68418cbaea5e9a50b02674da348da46b61e85ba456` |
| `tests/test_anchor_candidate_runtime.py` | `25fad200731eef2cc0ccc5ec248bbf301b167c287ba2af2f5baec322efa61f45` |

The coordinator's wider regression run is outside this narrow re-review and is not claimed to have passed. No actual CPU candidate execution occurred.

## Final narrow P2 disposition

**PASS, P2 closed within this bounded re-review.** The original finding and intermediate unsuccessful re-review remain above as historical evidence. This disposition does not claim full Task 3 completion or authorize measurements.

The current `scripts/bench_anchor_candidates.py:633` preserves `readiness.errors` in the revalidation exception. The protected inventory cache at `scripts/bench_anchor_candidates.py:889` and cache-only publication at `scripts/bench_anchor_candidates.py:927` remain intact. Changed parent authorities still fail revalidation and are not substituted for original provenance. The new R-drift assertion was not weakened.

This reviewer independently reran the same focused command documented above against the current bytes. It returned exit code 0 with `4 passed, 165 deselected in 36.17s`.

The reviewer then independently checked the artifacts from all four cases under `/private/var/folders/sd/hj30l0b96zj8lbk9ktvcxvpr0000gn/T/pytest-of-hamza/pytest-277`.

- Both `runtime.json` and `runtime.md` exist in every case. Each aggregate has `status=not-comparable`, `comparable=false`, `measured=false`, and empty `tracks`.
- The R-drift reports preserve the SHA256-mismatch cause. The R-disappearance reports preserve the missing-file cause. Both later-D failures preserve the original synthetic worker exception. Every recorded error appears in both JSON and Markdown.
- An already accepted R measurement is retained only in `diagnostic_only` for the later-D failures. No R duration is accepted when authority loss occurs immediately after the R worker returns. No failure report contains a comparison table.
- For all candidates in all four cases, the canonical hash of the published cached `source_files` equals the original catalogue-bound `scientific_source.inventory_sha256`. No changed parent provenance is accepted.

The coordinator's full-suite rerun remains separate and is not claimed to have passed here. This reviewer modified only this review file. No production or test edits, real-model or real-checkpoint execution, image decoding, SSH, jobs, or commits occurred.

File hashes stayed unchanged across this final narrow review.

| File | SHA256 |
| --- | --- |
| `scripts/bench_anchor_candidates.py` | `7a0519ac27bc44f9649066d01c438fe91c951e7ad9a01670b7e96806b0246da8` |
| `scripts/bench_anchor_candidate_worker.py` | `abd8d28dbe50d88baad73b68418cbaea5e9a50b02674da348da46b61e85ba456` |
| `tests/test_anchor_candidate_runtime.py` | `25fad200731eef2cc0ccc5ec248bbf301b167c287ba2af2f5baec322efa61f45` |
