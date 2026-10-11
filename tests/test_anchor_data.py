from importlib import import_module, util
from pathlib import Path
import random
import sys

import cv2
import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))


@pytest.fixture
def query_module():
    assert util.find_spec("refer_or_clear.data.query") is not None, "pilot query interface missing"
    return import_module("refer_or_clear.data.query")


@pytest.fixture
def manifests(tmp_path):
    roles = ["train"] * 1778 + ["val"] * 222 + ["test"] * 222
    records = pd.DataFrame([
        dict(dataset="sidtd", image_id=f"image_{index:04d}", path=f"images/{index}.png",
             design_id="sidtd/design", component_id=f"component_{index // 2}",
             label=index % 2, attack_family="text_copy" if index % 2 else "")
        for index in range(len(roles))
    ])
    split = records[["image_id", "design_id", "component_id", "label"]].copy()
    split["role"], split["protocol"] = roles, "A"
    records_path, split_path = tmp_path / "records.csv", tmp_path / "split.csv"
    records.to_csv(records_path, index=False)
    split.sample(frac=1, random_state=42).to_csv(split_path, index=False)
    return records, split, records_path, split_path


def sample_rows(role="val", label=1):
    return pd.DataFrame([dict(image_id="sample", design_id="sidtd/design", path="sample.png",
                             label=label, component_id="component", attack_family="text_copy",
                             role=role)])


def write_raw(root):
    height, width = 37, 83
    grid = np.arange(height * width, dtype=np.uint16).reshape(height, width)
    image = np.stack([grid % 256, (grid * 3) % 256, (grid * 7) % 256], axis=2).astype(np.uint8)
    assert cv2.imwrite(str(root / "sample.png"), image)
    return image


def write_legacy(root, dense=False):
    plane = np.zeros((512, 800), np.uint8)
    plane[140:240, 210:330] = 255
    for name in ("canvases", "zones") + (("masks", "valid") if dense else ()):
        (root / name).mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(root / "canvases/sample.png"), np.repeat(plane[..., None], 3, axis=2))
    assert cv2.imwrite(str(root / "zones/sidtd_design.png"), plane)
    if dense:
        assert cv2.imwrite(str(root / "masks/sample.png"), plane)
        assert cv2.imwrite(str(root / "valid/sample.png"), np.full_like(plane, 255))
    return plane


def test_rgb_needs_only_raw_image_and_label(query_module, tmp_path):
    write_raw(tmp_path)
    item = query_module.AnchorQueryDataset(sample_rows(), tmp_path, None, "R", False, {})[0]
    assert set(item) == {"query", "label", "meta"}
    assert item["query"].shape == (3, 299, 299)
    assert item["query"].dtype == torch.float32
    assert type(item["label"]) is int and item["label"] == 1
    assert item["meta"] == dict(image_id="sample", design_id="sidtd/design", attack_family="text_copy")


def test_rgb_preprocess_matches_scorer(query_module, tmp_path):
    from score_sidtd_checkpoints import preprocess
    image = write_raw(tmp_path)
    item = query_module.AnchorQueryDataset(sample_rows(), tmp_path, None, "R", False, {})[0]
    assert torch.equal(item["query"], torch.from_numpy(preprocess(image, 299)))


def test_test_pixels_are_never_opened(query_module, manifests, monkeypatch, tmp_path):
    _, _, records_path, split_path = manifests
    monkeypatch.setattr(cv2, "imread", lambda *args, **kwargs: pytest.fail("metadata join opened pixels"))
    rows = query_module.load_role_rows(records_path, split_path, "val")
    assert len(rows) == 222 and set(rows.role) == {"val"}
    with pytest.raises(ValueError, match="role|test"):
        query_module.load_role_rows(records_path, split_path, "test")
    with pytest.raises(ValueError, match="role|test"):
        query_module.AnchorQueryDataset(sample_rows("test"), tmp_path, None, "R", False, {})


@pytest.mark.parametrize("damage", ["duplicate_record", "duplicate_split", "missing_record", "missing_split",
                                   "unknown_id", "label", "design_id", "component_id", "bad_label",
                                   "wrong_counts", "protocol", "missing_column"])
def test_join_rejects_duplicate_missing_or_mismatched_ids(query_module, manifests, damage):
    records, split, records_path, split_path = manifests
    if damage == "duplicate_record":
        records = pd.concat([records, records.iloc[:1]])
    elif damage == "duplicate_split":
        split = pd.concat([split, split.iloc[:1]])
    elif damage == "missing_record":
        records = records.iloc[1:]
    elif damage == "missing_split":
        split = split.iloc[1:]
    elif damage == "unknown_id":
        split.loc[0, "image_id"] = "unknown"
    elif damage in ("label", "design_id", "component_id"):
        split.loc[0, damage] = 1 if damage == "label" else "incorrect"
    elif damage == "bad_label":
        records["label"] = records.label.astype(float)
        split["label"] = split.label.astype(float)
        records.loc[0, "label"] = split.loc[0, "label"] = 0.5
    elif damage == "wrong_counts":
        split.loc[0, "role"] = "val"
    elif damage == "protocol":
        split.loc[0, "protocol"] = "B"
    else:
        records = records.drop(columns="attack_family")
    records.to_csv(records_path, index=False)
    split.to_csv(split_path, index=False)
    with pytest.raises(ValueError):
        query_module.load_role_rows(records_path, split_path, "train")


def test_no_wrong_design_relabeling(query_module, manifests):
    records, split, records_path, split_path = manifests
    rows = query_module.load_role_rows(records_path, split_path, "train")
    expected = records[records.image_id.isin(split[split.role == "train"].image_id)].sort_values("image_id")
    assert rows.image_id.tolist() == sorted(expected.image_id)
    assert rows.label.tolist() == expected.label.tolist()
    assert rows.design_id.tolist() == expected.design_id.tolist()
    assert rows.attack_family.tolist() == expected.attack_family.tolist()


@pytest.mark.parametrize("track,train", [("D", True), ("D", False), ("N", False)])
def test_legacy_classification_never_opens_dense_or_reference_files(query_module, tmp_path, monkeypatch, track, train):
    write_legacy(tmp_path)
    original = cv2.imread
    def guarded(path, flags):
        assert Path(path).parent.name in {"canvases", "zones"}, path
        return original(path, flags)
    monkeypatch.setattr(cv2, "imread", guarded)
    item = query_module.AnchorQueryDataset(sample_rows("train" if train else "val"), tmp_path,
                                          tmp_path, track, train, {"enabled": False})[0]
    assert set(item) == {"query", "label", "meta"}
    assert item["query"].shape == (4, 512, 800)
    assert set(item["query"][3].unique().tolist()) <= {0.0, 1.0}


@pytest.mark.parametrize("missing", ["canvases/sample.png", "zones/sidtd_design.png",
                                     "masks/sample.png", "valid/sample.png"])
def test_missing_required_n_training_files_fail_with_id(query_module, tmp_path, missing):
    write_legacy(tmp_path, dense=True)
    (tmp_path / missing).unlink()
    dataset = query_module.AnchorQueryDataset(sample_rows("train"), tmp_path, tmp_path, "N", True, {})
    with pytest.raises((ValueError, FileNotFoundError), match="sample"):
        dataset[0]


def test_corrupt_rgb_fails_with_id(query_module, tmp_path):
    (tmp_path / "sample.png").write_bytes(b"not an image")
    dataset = query_module.AnchorQueryDataset(sample_rows(), tmp_path, None, "R", False, {})
    with pytest.raises(ValueError, match="sample"):
        dataset[0]


def test_legacy_rejects_wrong_canvas_shape(query_module, tmp_path):
    write_legacy(tmp_path)
    cv2.imwrite(str(tmp_path / "canvases/sample.png"), np.zeros((30, 50, 3), np.uint8))
    dataset = query_module.AnchorQueryDataset(sample_rows(), tmp_path, tmp_path, "D", False, {})
    with pytest.raises(ValueError, match="sample"):
        dataset[0]


def test_genuine_n_uses_zero_target_without_mask_read(query_module, tmp_path):
    write_legacy(tmp_path, dense=True)
    (tmp_path / "masks/sample.png").unlink()
    item = query_module.AnchorQueryDataset(sample_rows("train", label=0), tmp_path, tmp_path,
                                          "N", True, {"enabled": False})[0]
    assert item["mask"].shape == item["valid"].shape == (1, 512, 800)
    assert item["mask"].dtype == item["valid"].dtype == torch.float32
    assert not item["mask"].any() and item["valid"].all()


@pytest.mark.parametrize("track", ["R", "D", "N"])
def test_augmentation_is_seeded_and_validation_ignores_it(query_module, tmp_path, track):
    write_raw(tmp_path)
    write_legacy(tmp_path, dense=True)
    dataset = query_module.AnchorQueryDataset(sample_rows("train"), tmp_path, tmp_path, track, True, {})
    random.seed(42)
    first = dataset[0]
    random.seed(42)
    second = dataset[0]
    assert all(torch.equal(first[key], second[key]) for key in ("query", "mask", "valid") if key in first)
    validation = query_module.AnchorQueryDataset(sample_rows(), tmp_path, tmp_path, track, False, {})
    state = random.getstate()
    first_val = validation[0]
    assert random.getstate() == state
    random.seed(89)
    assert torch.equal(first_val["query"], validation[0]["query"])


def test_query_jitter_moves_targets_but_not_design_zone(query_module, tmp_path, monkeypatch):
    plane = write_legacy(tmp_path, dense=True)
    identity = np.float32([[1, 0, 0], [0, 1, 0]])
    monkeypatch.setattr(query_module.AnchorQueryDataset, "_geometry",
                        lambda self, height, width: (identity, (slice(0, height), slice(0, width))))
    offsets = iter([2, -1])
    monkeypatch.setattr(random, "randint", lambda *args: next(offsets))
    item = query_module.AnchorQueryDataset(sample_rows("train"), tmp_path, tmp_path, "N", True,
                                          {"jitter_p": 1.0, "photometric": False})[0]
    expected = cv2.warpAffine(plane, np.float32([[1, 0, 2], [0, 1, -1]]), (800, 512), flags=cv2.INTER_NEAREST)
    assert torch.equal(item["mask"][0] > 0, torch.from_numpy(expected > 0))
    assert torch.equal(item["query"][0] > 0, item["mask"][0] > 0)
    assert torch.equal(item["query"][3] > 0, torch.from_numpy(plane > 0))
    assert not item["valid"][0, :, :2].any() and not item["valid"][0, -1].any()


def test_affine_geometry_keeps_binary_targets_aligned(query_module, tmp_path, monkeypatch):
    write_legacy(tmp_path, dense=True)
    translation = np.float32([[1, 0, 5], [0, 1, -3]])
    monkeypatch.setattr(query_module.AnchorQueryDataset, "_geometry",
                        lambda self, height, width: (translation, (slice(0, height), slice(0, width))))
    item = query_module.AnchorQueryDataset(sample_rows("train"), tmp_path, tmp_path, "N", True,
                                          {"jitter_px": 0, "photometric": False})[0]
    assert torch.equal(item["query"][0] > 0, item["mask"][0] > 0)
    assert torch.equal(item["query"][3], item["mask"][0])
    assert set(item["mask"].unique().tolist()) == {0.0, 1.0}


def test_no_unused_reference_photometric_rng_draws(query_module, tmp_path, monkeypatch):
    write_legacy(tmp_path)
    identity = np.float32([[1, 0, 0], [0, 1, 0]])
    monkeypatch.setattr(query_module.AnchorQueryDataset, "_geometry",
                        lambda self, height, width: (identity, (slice(0, height), slice(0, width))))
    dataset = query_module.AnchorQueryDataset(sample_rows("train"), tmp_path, tmp_path, "D", True,
                                             {"jitter_px": 0, "photometric": True})
    random.seed(42)
    dataset[0]
    actual = random.getstate()
    random.seed(42)
    for _ in range(3):
        random.uniform(0.8, 1.2)
    assert random.getstate() == actual


def test_raw_crop_center_fallback_is_bounded(query_module, tmp_path):
    image = np.full((1, 4096, 3), 137, np.uint8)
    cv2.imwrite(str(tmp_path / "sample.png"), image)
    item = query_module.AnchorQueryDataset(sample_rows("train"), tmp_path, None, "R", True,
                                          {"brightness_contrast_p": 0.0})[0]
    assert item["query"].shape == (3, 299, 299) and torch.isfinite(item["query"]).all()


@pytest.mark.parametrize("file", ["zones/sidtd_design.png", "masks/sample.png", "valid/sample.png"])
def test_malformed_dense_planes_fail_with_id(query_module, tmp_path, file):
    write_legacy(tmp_path, dense=True)
    cv2.imwrite(str(tmp_path / file), np.zeros((30, 50), np.uint8))
    dataset = query_module.AnchorQueryDataset(sample_rows("train"), tmp_path, tmp_path, "N", True, {})
    with pytest.raises(ValueError, match="sample"):
        dataset[0]


def test_raw_training_photometry_clips_to_uint8(query_module, tmp_path, monkeypatch):
    cv2.imwrite(str(tmp_path / "sample.png"), np.full((299, 299, 3), 240, np.uint8))
    dataset = query_module.AnchorQueryDataset(sample_rows("train"), tmp_path, None, "R", True,
                                            {"crop_scale": [1.0, 1.0], "crop_ratio": [1.0, 1.0],
                                             "horizontal_flip_p": 0.0, "vertical_flip_p": 0.0,
                                             "brightness_contrast_p": 1.0, "contrast_factor": [1.2, 1.2],
                                             "brightness_offset": [0.2, 0.2]})
    item = dataset[0]
    expected = torch.tensor([(1.0 - 0.485) / 0.229, (1.0 - 0.456) / 0.224, (1.0 - 0.406) / 0.225])
    torch.testing.assert_close(item["query"], expected[:, None, None].expand_as(item["query"]))


@pytest.mark.parametrize("invalid", [{"crop": [400, 600]}, {"jpeg_quality": [60, 100]},
                                     {"jitter_p": 1.1}, {"jitter_px": -1}, {"reference": "median"}])
def test_invalid_legacy_augmentation_fails_closed(query_module, tmp_path, invalid):
    with pytest.raises(ValueError):
        query_module.AnchorQueryDataset(sample_rows("train"), tmp_path, tmp_path, "D", True, invalid)


def test_dataset_rejects_unsafe_path_with_image_id(query_module, tmp_path):
    rows = sample_rows()
    rows.loc[0, "path"] = "../test/sample.png"
    dataset = query_module.AnchorQueryDataset(rows, tmp_path, None, "R", False, {})
    with pytest.raises(ValueError, match="sample"):
        dataset[0]


def test_validation_does_not_read_existing_corrupt_dense_files(query_module, tmp_path):
    write_legacy(tmp_path, dense=True)
    (tmp_path / "masks/sample.png").write_bytes(b"corrupt forgery target")
    (tmp_path / "valid/sample.png").write_bytes(b"corrupt valid plane")
    item = query_module.AnchorQueryDataset(sample_rows(), tmp_path, tmp_path, "N", False, {})[0]
    assert set(item) == {"query", "label", "meta"} and torch.isfinite(item["query"]).all()
