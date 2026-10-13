# Fast matched OFF/ON sweep (13 Oct 2026, deadline plan)

## Diagnosis that this plan acts on

1. Branch co-adaptation. The current loss supervises only `query_logit + correction_logit`.
   In the selected OFF38041 checkpoint the correction logit carries the decision and the
   global query logit is negative for both classes (query-only AUC .70). The separately
   trained global-only run is better (.955/.985 vs .928/.975). B0 confirms the same
   architecture trains fine on another source split (OFF .96/.98, ON .98/.998) and shows
   the headline effect on unseen nationalities (OFF AUC .61, ON .94).
2. ON-arm BatchNorm contamination. The reference prefix runs in train mode and updates the
   shared stem/layer1/layer2 running statistics with reference-image statistics.
3. Throughput. GPU-only 37 img/s, loader 8 img/s, 0% GPU utilization on both running jobs.
   The fold-aware loader caches planes but still re-reads and SHA-256 hashes six files per
   sample on every cache hit (twice in Protocol A), with zero workers.
4. The published 98-99% SIDTD numbers ride on the Q92/Q75 JPEG confound (released checkpoints
   fall to AUC .65/.54 under re-encoding). A shortcut-resistant canvas pipeline tops out near
   .95-.98 accuracy on fold 0. That is the honest anchor and belongs in the paper.

## User decisions (13 Oct)

- OFF = global query classifier (adapter constructed with the same seed for matched initial
  tensors, but frozen and bypassed). ON = global + reference correction, trained with
  auxiliary BCE on `query_logit` (weight .5) so the global branch stays competent.
- Loader: authenticate on build, not on every cache hit. Fingerprint checks on authority files
  stay per access. Deterministic pixels and private augmentation streams are unchanged.

## Changes

| File | Change |
|---|---|
| `src/refer_or_clear/data/immutable_array_cache.py` | `fetch(..., validate_hits=True)`; when False, `validate` runs only before a cold build. |
| `src/refer_or_clear/data/protocol_a_reference.py`, `protocol_b_reference.py` | call with `validate_hits=False`, drop the post-fetch revalidation. Policy hash changes, so new registration tables are rebuilt by the smokes. |
| `src/refer_or_clear/models/direct_reference.py` | `local_enabled` (bypass adapter, freeze its tensors) and `reference_bn` in {`shared`, `stored`} (stored: reference prefix uses running statistics without updating them). Recorded in provenance. |
| `src/refer_or_clear/direct_reference_pilot.py` | `train_direct_reference_epoch(..., auxiliary_query_weight)`: `loss = BCE(attack) + w*BCE(query)`; `image_loss` stays the fused BCE; `aux_query_loss` logged. |
| `protocol_a_train.py`, `protocol_b_train.py` | base config may carry `arm_recipe: {off: {...}, on: {...}}`; effective config gets `local_branch`, `auxiliary_query_weight`, `reference_bn` for the arm. Model built from the effective config. |
| `scripts/protocol_a_study.py`, `protocol_b_study.py` | `fresh_model(request, config)`; cache cap 8 GiB. |
| `protocol_a_receipts.py`, `protocol_b_receipts.py` | cross-arm provenance equality ignores the arm recipe keys; history check allows `loss = image_loss + aux`. |
| `configs/direct_reference_study_v2.yaml` | the v2 recipe with `arm_recipe`. |
| `scripts/cluster/protocol_{a,b}_study.sh` | `--mem=14G`. |

## Fold-0 contrast set (parallel, same new source)

| Run | OFF | ON |
|---|---|---|
| v2 | global-only | adapter, aux .5, shared BN |
| v2-storedbn | (same OFF, not retrained) | adapter, aux .5, stored BN |
| v2-storedbn-noaux | - | adapter, aux 0, stored BN |

Existing 38146 (global-only, old loader) and 38152 (adapter, aux 0, shared BN) complete the
2x2 on the ON side. Predeclared rule: fold-0 validation accuracy, then AUC, earliest epoch;
copy-attack recall and genuine false positives reported. Then all A folds and B groups 1-9
run with the selected recipe in one wave. C evaluation uses the frozen fold-0 models.
