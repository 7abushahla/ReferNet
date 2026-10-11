"""Official train/validation membership and reference-free pilot query tensors."""
from __future__ import annotations

import math
from pathlib import Path
import random

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ..anchor_contract import contract_for
from ..pilot_config import resolve_augmentation
from .dataset import IMAGENET_MEAN, IMAGENET_STD, PairDataset, _normalize, _photometric

_RECORD_COLUMNS = ("image_id", "path", "label", "design_id", "attack_family", "component_id")
_SPLIT_COLUMNS = ("image_id", "label", "design_id", "component_id", "role")
_ROLE_COUNTS = {"train": 1778, "val": 222, "test": 222}


def _validate_rows(frame: pd.DataFrame, columns, name: str):
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"{name} missing columns {sorted(missing)}")
    if frame.image_id.duplicated().any():
        duplicate = frame.loc[frame.image_id.duplicated(), "image_id"].iloc[0]
        raise ValueError(f"{name} duplicate image_id {duplicate}")
    for column in columns:
        if column != "attack_family" and (frame[column].isna() | frame[column].astype(str).str.strip().eq("")).any():
            raise ValueError(f"{name} empty {column}")
    labels = pd.to_numeric(frame.label, errors="coerce")
    if not labels.isin([0, 1]).all():
        raise ValueError(f"{name} labels must be scalar binary integers")


def load_role_rows(records_csv: Path, split_csv: Path, role: str) -> pd.DataFrame:
    """Validate the complete official metadata join without inspecting any image."""
    if role not in ("train", "val"):
        raise ValueError("Pilot role must be train or val, never test")
    records = pd.read_csv(records_csv, keep_default_na=False, dtype={
        "image_id": str, "design_id": str, "component_id": str, "path": str, "attack_family": str,
    })
    split = pd.read_csv(split_csv, keep_default_na=False, dtype={
        "image_id": str, "design_id": str, "component_id": str, "role": str,
    })
    _validate_rows(records, _RECORD_COLUMNS, "records")
    _validate_rows(split, _SPLIT_COLUMNS, "split")
    if "dataset" in records:
        records = records.loc[records.dataset == "sidtd"].copy()
    if "protocol" in split and not split.protocol.eq("A").all():
        raise ValueError("Pilot membership must use official Protocol A")
    counts = split.role.value_counts().to_dict()
    if counts != _ROLE_COUNTS:
        raise ValueError(f"Official membership counts must be {_ROLE_COUNTS}, got {counts}")
    record_ids, split_ids = set(records.image_id), set(split.image_id)
    if record_ids != split_ids:
        raise ValueError(f"Membership discrepancy, missing records {sorted(split_ids - record_ids)[:3]}, "
                         f"missing split IDs {sorted(record_ids - split_ids)[:3]}")
    joined = split.merge(records[list(_RECORD_COLUMNS)], on="image_id", how="outer",
                         validate="one_to_one", suffixes=("", "_record"), indicator=True)
    if not joined["_merge"].eq("both").all():
        raise ValueError("Official membership join is not one-to-one")
    for column in ("label", "design_id", "component_id"):
        if not joined[column].eq(joined[f"{column}_record"]).all():
            image_id = joined.loc[~joined[column].eq(joined[f"{column}_record"]), "image_id"].iloc[0]
            raise ValueError(f"{image_id} has mismatched {column}")
    joined = joined.drop(columns=["_merge", "label_record", "design_id_record", "component_id_record"])
    joined["label"] = joined.label.astype(int)
    return joined.loc[joined.role == role].sort_values("image_id").reset_index(drop=True)


def raw_rgb_299_v1(bgr: np.ndarray, augment: dict) -> np.ndarray:
    """Named NumPy/OpenCV training preset, not historical Albumentations replay."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    height, width = rgb.shape[:2]
    window = None
    for _ in range(augment["crop_attempts"]):
        area = height * width * random.uniform(*augment["crop_scale"])
        ratio = math.exp(random.uniform(*(math.log(value) for value in augment["crop_ratio"])))
        crop_width, crop_height = round(math.sqrt(area * ratio)), round(math.sqrt(area / ratio))
        if 0 < crop_width <= width and 0 < crop_height <= height:
            top, left = random.randint(0, height - crop_height), random.randint(0, width - crop_width)
            window = top, left, crop_height, crop_width
            break
    if window is None:
        low, high = augment["crop_ratio"]
        if width / height < low:
            crop_width, crop_height = width, round(width / low)
        elif width / height > high:
            crop_height, crop_width = height, round(height * high)
        else:
            crop_height, crop_width = height, width
        crop_height, crop_width = max(1, min(height, crop_height)), max(1, min(width, crop_width))
        window = (height - crop_height) // 2, (width - crop_width) // 2, crop_height, crop_width
    top, left, crop_height, crop_width = window
    size = contract_for("R").width
    rgb = cv2.resize(rgb[top:top + crop_height, left:left + crop_width], (size, size), interpolation=cv2.INTER_LINEAR)
    if random.random() < augment["horizontal_flip_p"]:
        rgb = rgb[:, ::-1]
    if random.random() < augment["vertical_flip_p"]:
        rgb = rgb[::-1]
    if random.random() < augment["brightness_contrast_p"]:
        contrast = random.uniform(*augment["contrast_factor"])
        brightness = random.uniform(*augment["brightness_offset"]) * 255
        rgb = np.clip(rgb.astype(np.float32) * contrast + brightness, 0, 255).astype(np.uint8)
    normalized = (rgb.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    return normalized.transpose(2, 0, 1)


def _read(path: Path, image_id: str, flags: int, shape=None) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"{image_id} required image missing at {path}")
    image = cv2.imread(str(path), flags)
    if image is None:
        raise ValueError(f"{image_id} cannot decode image at {path}")
    if shape is not None and image.shape[:2] != shape:
        raise ValueError(f"{image_id} expected shape {shape}, got {image.shape[:2]} at {path}")
    return image


def _relative_path(root: Path, relative: str, image_id: str) -> Path:
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{image_id} image paths must remain within the declared root")
    return root / path


class AnchorQueryDataset(Dataset):
    """Query only; D never reads dense targets and N reads them only for training.

    Geometry and query photometry reuse legacy primitives, without unused reference
    RNG draws. D's historical comparison is consequently non-stream-matched.
    """

    def __init__(self, rows: pd.DataFrame, raw_root: Path, legacy_root: Path | None,
                 track: str, train: bool, augment: dict):
        self.contract = contract_for(track)
        _validate_rows(rows, _RECORD_COLUMNS + ("role",), "dataset")
        expected_role = "train" if train else "val"
        if not rows.role.eq(expected_role).all():
            raise ValueError(f"Dataset role must be {expected_role}, test pixels are forbidden")
        if track != "R" and legacy_root is None:
            raise ValueError(f"Track {track} requires legacy_root")
        self.rows = rows.to_dict("records")
        self.raw_root, self.legacy_root = Path(raw_root), Path(legacy_root) if legacy_root is not None else None
        self.track, self.train = track, bool(train)
        self.augment = resolve_augmentation(track, augment)
        self.crop = self.augment.get("crop")
        self.affine_px = self.augment.get("affine_px", 0)

    def __len__(self):
        return len(self.rows)

    def _geometry(self, height, width):
        return PairDataset._geometry(self, height, width)

    def _legacy_query(self, row):
        image_id, design_id = row["image_id"], row["design_id"]
        shape = self.contract.height, self.contract.width
        query = _read(_relative_path(self.legacy_root, f"canvases/{image_id}.png", image_id), image_id,
                      cv2.IMREAD_COLOR, shape)
        zone_path = _relative_path(self.legacy_root, f"zones/{design_id.replace('/', '_')}.png", image_id)
        zone = (_read(zone_path, image_id, cv2.IMREAD_GRAYSCALE, shape) > 127).astype(np.uint8)
        dense = {}
        if self.track == "N" and self.train:
            valid_path = _relative_path(self.legacy_root, f"valid/{image_id}.png", image_id)
            dense["valid"] = (_read(valid_path, image_id, cv2.IMREAD_GRAYSCALE, shape) > 127).astype(np.uint8)
            if int(row["label"]):
                mask_path = _relative_path(self.legacy_root, f"masks/{image_id}.png", image_id)
                dense["mask"] = (_read(mask_path, image_id, cv2.IMREAD_GRAYSCALE, shape) > 127).astype(np.uint8)
            else:
                dense["mask"] = np.zeros(shape, np.uint8)
        if self.train and self.augment["enabled"]:
            matrix, window = self._geometry(*shape)
            query = PairDataset._warp(query, matrix, window, False)
            zone = PairDataset._warp(zone, matrix, window, True)
            dense = {key: PairDataset._warp(plane, matrix, window, True) for key, plane in dense.items()}
            jitter_px = self.augment["jitter_px"]
            if jitter_px and random.random() < self.augment["jitter_p"]:
                jitter = np.float32([[1, 0, random.randint(-jitter_px, jitter_px)],
                                     [0, 1, random.randint(-jitter_px, jitter_px)]])
                dimensions = query.shape[1], query.shape[0]
                query = cv2.warpAffine(query, jitter, dimensions, flags=cv2.INTER_LINEAR,
                                       borderMode=cv2.BORDER_REPLICATE)
                dense = {key: cv2.warpAffine(plane, jitter, dimensions, flags=cv2.INTER_NEAREST,
                                            borderMode=cv2.BORDER_CONSTANT) for key, plane in dense.items()}
            if self.augment["photometric"]:
                query = _photometric(query, self.augment["photometric_strength"])
        tensor = np.concatenate([_normalize(query), zone[None].astype(np.float32)], axis=0)
        return tensor, {key: torch.from_numpy(plane[None].astype(np.float32)) for key, plane in dense.items()}

    def __getitem__(self, index):
        row = self.rows[index]
        if self.track == "R":
            image = _read(_relative_path(self.raw_root, row["path"], row["image_id"]), row["image_id"], cv2.IMREAD_COLOR)
            if self.train and self.augment["enabled"]:
                query = raw_rgb_299_v1(image, self.augment)
            else:
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                rgb = cv2.resize(rgb, (self.contract.width, self.contract.height), interpolation=cv2.INTER_LINEAR)
                query = ((rgb.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1)
            dense = {}
        else:
            query, dense = self._legacy_query(row)
        return dict(query=torch.from_numpy(np.ascontiguousarray(query)), label=int(row["label"]),
                    meta={key: row[key] for key in ("image_id", "design_id", "attack_family")}, **dense)
