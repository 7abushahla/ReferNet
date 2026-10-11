"""Versioned, fail-closed scientific recipes, independent of runtime weight paths."""
from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path

import yaml

from .anchor_contract import SCHEMA_VERSION, contract_for


def augmentation_defaults(track: str) -> dict:
    contract_for(track)
    if track == "R":
        return dict(
            preset="raw_rgb_299_v1", enabled=True, crop_scale=[0.8, 1.0],
            crop_ratio=[0.75, 4 / 3], crop_ratio_sampling="log_uniform", crop_attempts=10,
            crop_fallback="bounded_center", horizontal_flip_p=0.5, vertical_flip_p=0.5,
            brightness_contrast_p=0.2, contrast_factor=[0.8, 1.2], brightness_offset=[-0.2, 0.2],
            brightness_units="255", contrast_semantics="multiply_pixels", clip="uint8_0_255",
            interpolation="opencv_inter_linear", jpeg_quality=None,
        )
    return dict(
        preset="legacy_query_v1", enabled=True, crop=None, affine_px=8, rotation_degrees=1.0,
        jitter_px=2, jitter_p=0.25, photometric=True, photometric_strength=0.2, jpeg_quality=None,
        interpolation="opencv_inter_linear", binary_interpolation="opencv_inter_nearest",
        affine_border="constant_zero", jitter_rgb_border="replicate", jitter_target_border="constant_zero",
        jitter_zone=False,
    )


def _merge(defaults: dict, supplied: dict, location: str = "config") -> dict:
    if not isinstance(supplied, dict):
        raise ValueError(f"{location} must be a mapping")
    unknown = supplied.keys() - defaults.keys()
    if unknown:
        raise ValueError(f"Unknown {location} fields {sorted(unknown, key=str)}")
    result = deepcopy(defaults)
    for key, value in supplied.items():
        expected = defaults[key]
        field = f"{location}.{key}"
        if isinstance(expected, dict):
            result[key] = _merge(expected, value, field)
            continue
        if expected is None:
            valid = value is None
        elif isinstance(expected, bool):
            valid = type(value) is bool
        elif isinstance(expected, int):
            valid = type(value) is int
        elif isinstance(expected, float):
            valid = type(value) in (int, float) and math.isfinite(value)
        elif isinstance(expected, list):
            valid = isinstance(value, list) and len(value) == len(expected)
            if valid:
                valid = all(type(item) in (int, float) and math.isfinite(item) for item in value)
        else:
            valid = isinstance(value, type(expected))
        if not valid:
            raise ValueError(f"Invalid type or value for {field}")
        result[key] = deepcopy(value)
    return result


def resolve_augmentation(track: str, supplied: dict) -> dict:
    """Resolve partial transform settings, including explicit diagnostic disablement."""
    defaults = augmentation_defaults(track)
    result = _merge(defaults, supplied, "augment")
    adjustable = {"enabled"}
    if track == "R":
        adjustable |= {"crop_scale", "crop_ratio", "horizontal_flip_p", "vertical_flip_p",
                       "brightness_contrast_p", "contrast_factor", "brightness_offset"}
        for key in ("crop_scale", "crop_ratio", "contrast_factor", "brightness_offset"):
            low, high = result[key]
            if low > high or (key != "brightness_offset" and low <= 0):
                raise ValueError(f"Invalid augment.{key} interval")
        if result["crop_scale"][1] > 1:
            raise ValueError("augment.crop_scale cannot exceed one")
    else:
        adjustable |= {"affine_px", "jitter_px", "jitter_p", "photometric"}
        if result["affine_px"] < 0 or result["jitter_px"] < 0:
            raise ValueError("Pixel limits cannot be negative")
    for key, value in result.items():
        if key.endswith("_p") and not 0 <= value <= 1:
            raise ValueError(f"augment.{key} must be a probability")
        if key not in adjustable and value != defaults[key]:
            raise ValueError(f"Unsupported transform semantics for augment.{key}")
    return result


def _defaults(track: str) -> dict:
    contract = contract_for(track)
    config = dict(
        schema_version=SCHEMA_VERSION, track=track, architecture_id=contract.architecture_id,
        data_contract=contract.data_contract, seed=42, fold=0, epochs=100,
        batch_size=contract.batch_size, eval_batch_size=contract.eval_batch_size,
        mask_weight=contract.mask_weight, bn_policy=contract.bn_policy,
        records_csv="data/processed/records.csv", split_csv="data/processed/splits/protocol_a_it0.csv",
        normalization=dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        precision=dict(dtype="float32", amp=False, cudnn_tf32=False, matmul_tf32=False),
        selection=dict(primary="accuracy", tie_break="auc", exact_tie="earlier_epoch"),
        accumulation_steps=2 if track == "R" else 1, drop_last=track != "R", eval_drop_last=False,
        wrong_design_fraction=0.0, augment=augmentation_defaults(track),
        comparison="non_stream_matched" if track in ("D", "D_running", "D_running_pilot") else "candidate_package",
    )
    if track == "R":
        config["optimizer"] = dict(name="sgd", lr=0.01, momentum=0.9, weight_decay=0.0, nesterov=False)
        config["scheduler"] = dict(name="reduce_on_plateau", monitor="val_ce", mode="min", factor=0.9,
                                   patience=1, threshold=1e-4, threshold_mode="rel", cooldown=0,
                                   min_lr=0.0, eps=1e-6)
    else:
        config["optimizer"] = dict(name="adamw", lr_backbone=1e-4, lr_new=1e-3, weight_decay=0.05,
                                   betas=[0.9, 0.999], eps=1e-8)
        config["scheduler"] = dict(name="warmup_cosine", warmup_iters=1000, warmup_cap_fraction=0.1)
    return config


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _unique_mapping(loader, node):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node)
        if not isinstance(key, str):
            raise ValueError("Configuration keys must be strings")
        if key in result:
            raise ValueError(f"duplicate configuration key {key!r}")
        result[key] = loader.construct_object(value_node)
    return result


_UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _unique_mapping)


def resolve_pilot_config(config_path: Path) -> dict:
    """Resolve a flat recipe; paths remain relative to the runtime working directory.

    Scientific settings are pinned to the authorized pilot recipes. Seed, fold and
    epoch counts are typed explicitly; diagnostic overrides belong to the runtime.
    """
    with Path(config_path).open() as stream:
        try:
            supplied = yaml.load(stream, Loader=_UniqueKeyLoader)
        except yaml.YAMLError as error:
            raise ValueError(f"Invalid pilot YAML in {config_path}") from error
    if not isinstance(supplied, dict) or not isinstance(supplied.get("track"), str):
        raise ValueError("Pilot config must be a mapping with an explicit track")
    defaults = _defaults(supplied["track"])
    config = _merge(defaults, supplied)
    if config["seed"] < 0 or not 0 <= config["fold"] < 10 or config["epochs"] <= 0:
        raise ValueError("Require nonnegative seed, official fold 0-9 and positive epochs")
    for key in ("records_csv", "split_csv"):
        if not config[key].strip():
            raise ValueError(f"{key} cannot be empty")
    if "split_csv" not in supplied:
        config["split_csv"] = f"data/processed/splits/protocol_a_it{config['fold']}.csv"
    variable = {"seed", "fold", "epochs", "records_csv", "split_csv"}
    for key, expected in defaults.items():
        if key not in variable and config[key] != expected:
            raise ValueError(f"{key} conflicts with the authorized {config['track']} pilot recipe")
    return config
