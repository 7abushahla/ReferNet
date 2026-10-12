# Direct-reference staging review

Date, 12 October 2026.

## Verdict

**BLOCKED on two fail-closed defects.** The 15 existing focused synthetic tests pass, but independent synthetic probes reproduce acceptance of unauthenticated frozen source and invalid actual-stack evidence. No allocation is cleared.

Reviewed SHA256 values:

| File | SHA256 |
|---|---|
| `scripts/direct_reference_job.py` | `3aa86e6ba5729b190b7c6ea70cadf20b5f930f5aae80106dd5ad7e18314c7401` |
| `tests/test_direct_reference_job.py` | `cfde3d208354992406fc6b8a256b0d2d84a4c3a1f2601e2492d3859afb4689b6` |

## Blocking findings

### 1. Copied source need not match the external source authority

`scripts/direct_reference_job.py:90` authenticates one source scan. `scripts/direct_reference_job.py:119` separately scans the files used for copying. The final check at `scripts/direct_reference_job.py:137` hashes the original source again, not the frozen code. The copied entry list is never compared with the external digest.

An A-to-B-to-A source change therefore passes. The independently reproduced fixture sequence was:

1. Use the existing `tests/test_direct_reference_job.py` synthetic setup.
2. After the initial `source_digest` returns the authorized A digest, change `src/dummy.py` from `value = 1` to `value = 999`.
3. Let the second scan and verified copy capture B. Restore A immediately after copying that dummy file.
4. Let the final source check observe A.

`prepare_job` succeeds and records the authorized A digest, but its frozen `code/src/dummy.py` contains `value = 999`; hashing the frozen code disagrees with the recorded authority. Thus a later launcher can execute B while its ledger claims A.

Minimum correction, authenticate and reuse one captured entry list and verify the completed frozen code digest against the external source authority before returning success. Add a synthetic regression for the demonstrated change/revert sequence. No new staging framework is needed.

### 2. The actual-stack checker accepts absent, unrelated, or skipped cases

`scripts/direct_reference_job.py:173` checks suite aggregate attributes only. It does not require the actual installed-stack testcase or inspect testcase failure/error/skip elements. The positive fixture at `tests/test_direct_reference_job.py:144` itself has no testcase.

Each of these independent XML probes returned `True`:

```xml
<testsuites><testsuite tests="1" failures="0" errors="0" skipped="0"/></testsuites>
<testsuites><testsuite tests="1" failures="0" errors="0" skipped="0"><testcase name="test_unrelated"/></testsuite></testsuites>
<testsuites><testsuite tests="1" failures="0" errors="0" skipped="0"><testcase name="test_actual_direct_reference_gpu_integration"><skipped/></testcase></testsuite></testsuites>
```

Minimum correction, require the expected `test_actual_direct_reference_gpu_integration` testcase, reject missing/unexpected cases and testcase skip/failure/error elements, and validate aggregate counts consistently. The existing `anchor_job_manifest.py:196` provides the narrow named-case precedent. The current wrapper selects the correct node explicitly, but this independent evidence validator still cannot distinguish an actual pass from these invalid reports.

## Other inspected boundaries

The stager and invoked helper/verifier imports are standard-library-only; no scientific configuration resolver or model runner is called. Allocation checks precede staging. Request/configuration/records/split/official-weight/enrollment bindings are checked. Existing `source_entries`, `selected_rows(..., "R")`, and `data_entries` are reused to select train/validation raw files and exclude test/reference/clip paths. Counts are required to be 1,778/222. Declared enrollment products are hash-verified before and after relocation; the original manifest and request bytes are copied unchanged. Input/output overlap, existing destinations, and symlinks are rejected. Copied files and input directories are made read-only. These inspected behaviors do not repair the two blockers above and are not detector or phase-runner clearance.

## Fresh validation and scope

`PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 /private/tmp/roc-anchor-dev/bin/python -m pytest -p no:cacheprovider tests/test_direct_reference_job.py -q` produced **15 passed in 9.66 seconds**.

Independent probes used only temporary synthetic fixtures, XML strings, and inert source text. No production weights/images, scientific model imports, GPU execution, SSH, jobs, or commits occurred. Production/test files were not edited. This review note is the only authored artifact.

## Narrow fix rereview

**Bounded PASS for the corrected stager.** This supersedes the initial blocked verdict above. Both reproduced blockers are resolved; no allocation, wrapper, complete phase runner, or detector execution is cleared by this rereview.

Corrected SHA256 values:

| File | SHA256 |
|---|---|
| `scripts/direct_reference_job.py` | `1a2ac07d989ca4b7f2c63993c0efbf60c821fcd3f81859ea6417aec20efd1662` |
| `tests/test_direct_reference_job.py` | `4bf69813ccc50d2ce03e2622340676a2234c30685a10efd314a0a8535518ff74` |

- Source entries are captured once, authenticated against the external authority, and reused for verified copying (`direct_reference_job.py:90`). Original and completed frozen source digests are both checked (`direct_reference_job.py:137`). The corrected regression restores the original only after the copied modified candidate, matching the previously reproduced timing window (`test_direct_reference_job.py:140`).
- JUnit validation requires exactly one correctly named installed-stack case with the expected final classname module, rejects testcase status children independently of aggregate counters, and checks suite totals (`direct_reference_job.py:174`). The five absent/unrelated/skipped/failure/error regressions pass.
- Fresh focused command, unchanged from above, produced **22 passed in 19.78 seconds**. Independent XML probes rejected all five malformed cases. An additional temporary synthetic probe tampered only with the completed copied source after `copy_verified`; the frozen digest rejected it and no successful input manifest was emitted.

The coordinator reported proper RED failures for the original source race and all five JUnit cases. This rereview independently verifies current GREEN behavior; it does not claim another historical RED execution. No production/test edits, scientific model execution, SSH, jobs, or commits occurred. Only this review note was appended.
