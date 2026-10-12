# Direct-reference pilot core review

Date, 12 October 2026.

## Verdict

**Bounded PASS for the frozen Task 3 core.** No blocking defect found in the reviewed classification epoch, evaluation, configuration, or selected-checkpoint replay primitives. This is not clearance of the complete Task 3 phase runner, diagnostic packaging, an allocation, or either long run.

Reviewed files and SHA256 values are:

| File | SHA256 |
|---|---|
| `src/refer_or_clear/direct_reference_pilot.py` | `66a4bfebf919647af74f13e60a3b71b94f97dc3ca1f34b8bf75d51d10bf4ee01` |
| `configs/direct_reference_pilot.yaml` | `a89c6213a1877632510f66f49cf843c7e0c5a9e585b434da08abbecb8abf7b39` |
| `tests/test_direct_reference_pilot.py` | `cb56ebd46c0b7bcb2dbf49ed97d2fa78ac1b71cdd7880ddd30d1b4d98d767a5c` |

## Verified boundaries

- Four-input dispatch keeps labels and metric metadata outside forward. Off-mode optional reference is not inspected. Classification-only BCE ignores dense outputs. FP32 checks, finite losses/gradients, clip norm 5, one optimizer/scheduler step per batch, and sample-weighted accounting are covered (`direct_reference_pilot.py:45`, `direct_reference_pilot.py:89`).
- Evaluation retains unavailable examples in full-population and subgroup metrics, exports referral information, rejects duplicate IDs, and checks immutable evaluation buffers. Genuine threshold ties and strict altered probability greater than 0.5 pass synthetic checks (`direct_reference_pilot.py:116`).
- The configuration retains the declared D recipe without an old public architecture. Optimizer/scheduler and accuracy-first, AUC-second, earliest-exact-tie selection reuse the existing helpers (`direct_reference_pilot.yaml:1`, `test_direct_reference_pilot.py:222`).
- The checkpoint envelope binds the new schema, architecture, arm, effective configuration, source/data/phase hashes, and official initialization provenance. Legacy, mixed-arm, incomplete, and incompatible envelopes are rejected before model state loading. Only the replay primitive loads a checkpoint into the model; strict loading, prediction order, logits, and unchanged evaluation buffers are checked (`direct_reference_pilot.py:154`, `direct_reference_pilot.py:199`, `direct_reference_pilot.py:218`).

## Integration still pending

These primitives do not implement a phase CLI or authenticate the external phase authority and gate receipts. The coordinator must enforce authorized membership before pixel access, both-mode training gates before validation, smoke/replay before long training, fresh initialization between phases, and replay-only checkpoint loading. A hash field alone does not authenticate its receipt contents. The core training function accepts a caller-supplied model and therefore does not independently prohibit training a previously replayed model.

The inherited overfit probe uses no gradient clipping (`pilots.py:698`), whereas the reviewed actual-recipe epoch function always clips at 5. Future overfit orchestration must preserve that distinction instead of silently reusing this epoch function unchanged. This is a handoff requirement, not a defect in the scoped actual-recipe core.

## Fresh validation

Commands ran with `PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1` and `/private/tmp/roc-anchor-dev/bin/python`:

1. `-m pytest -p no:cacheprovider tests/test_direct_reference_pilot.py -q`, **34 passed in 5.20 seconds**.
2. `-m pytest -p no:cacheprovider tests/test_direct_reference_pilot.py tests/test_anchor_training.py tests/test_train_config.py::test_selection_key -q`, **60 passed in 11.66 seconds**. This includes one selection regression beyond the reported 59-test combination.
3. `tests/test_metrics.py`, **all five standalone checks passed**.

Both pytest runs emitted one PyTorch TypedStorage deprecation warning. No production weights/images, GPU model execution, SSH, jobs, or commits were used. Production and test files were not edited. This review note is the only authored artifact.
