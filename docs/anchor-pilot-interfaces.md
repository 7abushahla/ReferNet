# Anchor pilot interfaces

These contracts unblock independent workers. `anchor_contract.py` is the executable source for track IDs, shapes, batch sizes, loss weight and normalization. The coordinator owns this file and integration. Model workers own only their assigned model and test files. No worker submits jobs or commits another worker's changes.

## Dataset and configuration

`load_role_rows(records_csv: Path, split_csv: Path, role: str) -> pd.DataFrame` accepts only train/val and returns an image-ID-sorted validated join. Output includes path, label, image_id, design_id, attack_family and component_id. Join discrepancies are errors. Test pixels are never opened.

`AnchorQueryDataset(rows, raw_root, legacy_root, track, train, augment)` returns a dictionary. `query` is a float32 tensor at the track's input shape; `label` is a scalar integer 0/1; `meta` contains image_id, design_id and attack_family. Training N also returns float32 `mask` and `valid` shaped `(1,512,800)`. No target mask is needed for validation classification. Validation metadata may carry labels for metrics; no metadata or label is passed into model.forward.

R uses raw RGB only. D/N use legacy registered query PNG plus its design zone. All tracks omit reference image access. No caller constructs legacy eight-channel model inputs. D is non-stream-matched to its historical comparator.

`pilot_config.resolve_pilot_config(config_path: Path) -> dict`, imported by `pilots.py`, produces schema_version, track, architecture_id, data_contract, seed, fold, epochs, batch_size, eval_batch_size, mask_weight, bn_policy, optimizer/scheduler and augment dictionaries, precision and selection. Exact scientific settings follow the spec. File roots and offline weights are runtime paths, with their hashes saved separately in the run ledger.

## Model interface

Each builder returns an `nn.Module` with attributes `architecture_id: str`, `track: str`, and `initialization_provenance: dict`. Its `forward(query: Tensor) -> dict[str, Tensor]` receives one tensor only.

| Track | Builder | Outputs |
|---|---|---|
| R | `build_anchor_rgb(weights_path: Path, expected_sha256: str, head_seed: int)` | `class_logits (B,2)` and `attack_logit (B,)` |
| D | `build_anchor_changer(track="D", levir_path: Path, expected_sha256: str, head_seed: int)` | `attack_logit (B,)` |
| N | `build_anchor_changer(track="N", levir_path: Path, expected_sha256: str, head_seed: int)` | `attack_logit (B,)` and `map_logit (B,1,H/4,W/4)` |

Every model implements `parameter_groups() -> tuple[list[nn.Parameter], list[nn.Parameter]]`, pretrained active tensors followed by new active tensors. Each active trainable parameter appears exactly once. R's actual SGD uses the same LR for both groups. A missing map is absent, never a fabricated zero output.

`initialization_provenance` records requested initialization, checkpoint path/hash, loaded/missing/unexpected tensor keys and a SHA256 of sorted pretrained tensor names/shapes/dtypes/bytes before head training. Fail on unavailable or mismatched requested weights. No downloads occur inside model construction. The hash verification helper may live in the worker's file initially; coordinator can consolidate later if justified.

R replaces its pretrained 1,000-class head after loading it. D retains only the initialized query encoder and direct MLP. N keeps its active decoder and referral head, calling shared projection once and creating the zero absent stream afterward. D uses encoder batch statistics and evaluation batch one; R/N use frozen running statistics in evaluation. None accepts reference tensors, zones outside the declared query channels, labels or masks as separate forward arguments.

## Integration and evidence

The runner imports `train.selection_key`, `train.make_scheduler` for D/N, the existing metrics and `referral_loss`. R supplies CE plus sample-correct two-microbatch accumulation. N supplies dense targets only for training loss. Seeded construction must not alter the externally owned sampler/worker RNG stream.

Inference invariance and checkpoint replay use FP32 `atol=1e-5`, `rtol=1e-4`. R/N are checked across batch companions and positions; D is evaluated singly. A diagnostic exits after smoke/replay. A separate coordinator job starts long training only after gate review.
