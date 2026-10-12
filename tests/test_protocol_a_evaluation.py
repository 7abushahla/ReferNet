import csv
import hashlib
import importlib
import json
from pathlib import Path
import shutil
import sys

import cv2
import numpy as np
import pytest
import torch

from test_anchor_enrollment_integrity import bundle as integrity_bundle

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def binding(path):
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True))
    return binding(path)


@pytest.fixture
def api():
    assert (ROOT / "src/refer_or_clear/data/protocol_a_evaluation.py").exists(), "evaluation consumer missing"
    return importlib.import_module("refer_or_clear.data.protocol_a_evaluation")


@pytest.fixture
def inputs(api, integrity_bundle, tmp_path, monkeypatch):
    manifest_path, manifest = integrity_bundle
    raw = tmp_path / "raw"
    raw.mkdir()
    records = []
    split = []
    for identifier, role in (("support_0", "train"), ("support_1", "train"), ("val0", "val"), ("test0", "test"), ("test1", "test")):
        label = int(identifier == "test1")
        row = dict(image_id=identifier, dataset="sidtd", path=f"{identifier}.png", design_id="sidtd/example", label=label,
                   attack_family="text_copy" if label else "", component_id=identifier, annotation="DO-NOT-OPEN.json")
        records.append(row)
        split.append({key: row[key] for key in ("image_id", "design_id", "label", "component_id")})
        split[-1].update(role=role, protocol="A", ref_group="it0")
        if role == "test":
            assert cv2.imwrite(str(raw / row["path"]), np.full((64, 100, 3), (11, 66, 199), np.uint8))
    authorities = {}
    for name, rows in (("records", records), ("split", split)):
        path = tmp_path / f"{name}.csv"
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        authorities[name] = binding(path)
    import pandas as pd
    from refer_or_clear.data.enrollment import _table_hash
    manifest["official_split_sha256"] = _table_hash(pd.read_csv(authorities["split"]["path"], keep_default_na=False))
    for bank in manifest["banks"]["sidtd/example"].values():
        products = bank["products"]
        for name in ("target", "median", "target_coverage", "target_valid", "coverage", "valid", "zone"):
            image = np.full((512, 800, 3), (22, 77, 188), np.uint8) if name in ("target", "median") else np.full((512, 800), 255, np.uint8)
            assert cv2.imwrite(products[name]["path"], image)
            products[name] = binding(Path(products[name]["path"]))
        geometry_path = Path(products["geometry"]["path"])
        geometry = json.loads(geometry_path.read_text())
        geometry["official_split_sha256"] = manifest["official_split_sha256"]
        products["geometry"] = write_json(geometry_path, geometry)
        products["manifest"] = write_json(Path(products["manifest"]["path"]), {
            **geometry, "output_sha256": {name: product["sha256"] for name, product in products.items() if name != "manifest"}})
    authorities["enrollment"] = write_json(manifest_path, manifest)
    monkeypatch.setattr(api, "ENROLLMENT_SHA256", authorities["enrollment"]["sha256"])
    monkeypatch.setattr(api, "ROLE_COUNTS", {"train": 2, "val": 1, "test": 2})
    authorities["root"] = write_json(tmp_path / "root.json", dict(schema_version="direct_reference_product_root_v1",
        original_root=manifest["output_root"], relocated_root=manifest["output_root"],
        enrollment_manifest_sha256=authorities["enrollment"]["sha256"]))
    checkpoints = {}
    for arm in ("off", "on"):
        path = tmp_path / f"{arm}.pt"
        path.write_bytes(f"synthetic selected checkpoint {arm}".encode())
        checkpoints[arm] = binding(path)
    evaluation = dict(schema_version="protocol_a_evaluation_v1", protocol="A", fold=0,
        records_sha256=authorities["records"]["sha256"], split_sha256=authorities["split"]["sha256"],
        enrollment_manifest_sha256=authorities["enrollment"]["sha256"], test_image_ids=["test0", "test1"],
        source_validation_selected=True, checkpoints=checkpoints)
    authorities["evaluation"] = write_json(tmp_path / "evaluation.json", evaluation)
    monkeypatch.setattr(api.direct, "_fit", lambda source, target, seed: (np.eye(3), 100))
    monkeypatch.setattr(api.direct, "_features", lambda image: None)
    return dict(authorities=authorities, evaluation=evaluation, raw=raw, tmp=tmp_path, manifest=manifest)


def context(api, inputs):
    authorities = inputs["authorities"]
    return api.load_evaluation_context(authorities["records"], authorities["split"], authorities["enrollment"],
        authorities["root"], inputs["raw"], evaluation_binding=authorities["evaluation"])


def test_context_requires_binding_before_any_pixel_read(api, inputs, monkeypatch):
    monkeypatch.setattr(api.direct, "_image", lambda *args: pytest.fail("metadata authorization decoded pixels"))
    value = context(api, inputs)
    assert list(value.rows.image_id) == ["test0", "test1"]
    assert set(value.rows.role) == {"test"}
    assert not {"component_id", "annotation"} & set(value.rows)
    inputs["authorities"]["evaluation"] = None
    with pytest.raises(ValueError):
        context(api, inputs)


@pytest.mark.parametrize("field,value", [("protocol", "B"), ("fold", True), ("fold", 1),
    ("source_validation_selected", False), ("source_validation_selected", 1),
    ("test_image_ids", ["test1", "test0"]), ("test_image_ids", ["test0", "val0"]),
    ("records_sha256", "a" * 64), ("enrollment_manifest_sha256", "b" * 64),
    ("checkpoints", {})])
def test_invalid_authorization_blocks_pixels(api, inputs, monkeypatch, field, value):
    inputs["evaluation"][field] = value
    inputs["authorities"]["evaluation"] = write_json(inputs["tmp"] / "evaluation.json", inputs["evaluation"])
    monkeypatch.setattr(api.direct, "_image", lambda *args: pytest.fail("unauthorized pixels"))
    with pytest.raises(ValueError):
        context(api, inputs)


def test_off_on_same_query_no_median_off_and_saved_registration_replay(api, inputs, monkeypatch):
    value = context(api, inputs)
    path = inputs["tmp"] / "registration.json"
    table = api.build_evaluation_registration(value, path)
    assert json.loads(path.read_text())["schema_version"] == "protocol_a_evaluation_registration_v1"
    loaded = api.load_evaluation_registration(value, binding(path))
    original = api.direct._image
    reads = []

    def audited_image(authority, flags):
        reads.append(authority["path"])
        return original(authority, flags)

    monkeypatch.setattr(api.direct, "_image", audited_image)
    off = api.ProtocolAEvaluationDataset(value, table, reference_enabled=False)[0]
    assert not any(Path(name).name in ("median.png", "target.png", "DO-NOT-OPEN.json") for name in reads)
    on = api.ProtocolAEvaluationDataset(value, loaded, reference_enabled=True)[0]
    assert torch.equal(off["query"], on["query"])
    assert torch.equal(off["overlap"], on["overlap"])
    assert tuple(on["query"].shape) == (4, 512, 800)
    assert tuple(on["reference"].shape) == (3, 512, 800)
    assert bool(on["context_available"])
    assert not on["meta"]["requires_referral"]
    assert "reference" not in off


def test_registration_failure_retained_for_referral(api, inputs, monkeypatch):
    value = context(api, inputs)
    monkeypatch.setattr(api.direct, "_fit", lambda *args: (None, 0))
    table = api.build_evaluation_registration(value, inputs["tmp"] / "failed.json")
    sample = api.ProtocolAEvaluationDataset(value, table, reference_enabled=True)[0]
    assert not bool(sample["context_available"])
    assert sample["meta"]["requires_referral"]
    assert torch.count_nonzero(sample["reference"]) == 0
    assert torch.count_nonzero(sample["query"][3]) == 0


@pytest.mark.parametrize("authority", ["evaluation", "root", "records", "split", "enrollment", "off", "on"])
def test_changed_authority_rejected_before_getter_pixels(api, inputs, monkeypatch, authority):
    value = context(api, inputs)
    table = api.build_evaluation_registration(value, inputs["tmp"] / "registration.json")
    dataset = api.ProtocolAEvaluationDataset(value, table, reference_enabled=False)
    changed = inputs["evaluation"]["checkpoints"][authority] if authority in ("off", "on") else inputs["authorities"][authority]
    with Path(changed["path"]).open("ab") as stream:
        stream.write(b"changed")
    monkeypatch.setattr(api.direct, "_image", lambda *args: pytest.fail("pixels opened after authority change"))
    with pytest.raises(ValueError):
        dataset[0]


def test_plain_or_mutated_registration_rejected(api, inputs):
    value = context(api, inputs)
    path = inputs["tmp"] / "registration.json"
    table = api.build_evaluation_registration(value, path)
    with pytest.raises(ValueError):
        api.ProtocolAEvaluationDataset(value, json.loads(path.read_text()), reference_enabled=False)
    document = json.loads(path.read_text())
    document["entries"]["test0"]["query_canvas_to_bank"][0][0] = 2
    write_json(path, document)
    with pytest.raises(ValueError):
        api.ProtocolAEvaluationDataset(value, table, reference_enabled=False)


def test_original_enrollment_hash_not_replaceable(api, inputs, monkeypatch):
    monkeypatch.setattr(api, "ENROLLMENT_SHA256", "a" * 64)
    with pytest.raises(ValueError, match="enrollment"):
        context(api, inputs)


def test_original_manifest_relocation_without_path_rewrite(api, inputs):
    original = Path(inputs["manifest"]["output_root"])
    relocated = inputs["tmp"] / "relocated"
    shutil.copytree(original, relocated)
    for path in original.rglob("*.png"):
        path.write_bytes(b"original product location is not the authorized relocation")
    inputs["authorities"]["root"] = write_json(inputs["tmp"] / "root.json", dict(
        schema_version="direct_reference_product_root_v1", original_root=str(original), relocated_root=str(relocated),
        enrollment_manifest_sha256=inputs["authorities"]["enrollment"]["sha256"]))
    value = context(api, inputs)
    table = api.build_evaluation_registration(value, inputs["tmp"] / "relocated-registration.json")
    sample = api.ProtocolAEvaluationDataset(value, table, reference_enabled=True)[0]
    assert all(Path(product["path"]).is_relative_to(relocated) for product in sample["meta"]["products"].values())
    assert bool(sample["context_available"])


def test_off_never_reads_median_bytes_and_test_component_does_not_route(api, inputs, monkeypatch):
    original_bank = api.direct.resolve_enrollment_bank
    calls = []

    def audited_bank(manifest, design, role, component):
        calls.append((role, component))
        return original_bank(manifest, design, role, component)

    monkeypatch.setattr(api.direct, "resolve_enrollment_bank", audited_bank)
    value = context(api, inputs)
    assert calls == [("test", None), ("test", None)]
    table = api.build_evaluation_registration(value, inputs["tmp"] / "registration.json")
    dataset = api.ProtocolAEvaluationDataset(value, table, reference_enabled=False)
    original_bytes = api.direct._bytes

    def audited_bytes(authority):
        assert Path(authority["path"]).name != "median.png"
        assert Path(authority["path"]).name != "DO-NOT-OPEN.json"
        return original_bytes(authority)

    monkeypatch.setattr(api.direct, "_bytes", audited_bytes)
    dataset[0]


def test_frozen_inverse_warp_and_normalization_geometry(api, inputs, monkeypatch):
    value = context(api, inputs)
    matrix = np.array([[.9, -.1, 20], [.1, .9, 15], [0, 0, 1]], np.float64)
    monkeypatch.setattr(api.direct, "_fit", lambda *args: (matrix, 100))
    table = api.build_evaluation_registration(value, inputs["tmp"] / "registration.json")
    sample = api.ProtocolAEvaluationDataset(value, table, reference_enabled=True)[0]
    raw = cv2.imread(str(inputs["raw"] / "test0.png"))
    canvas, valid, _ = api.direct.to_canvas(raw)
    inverse = np.linalg.inv(matrix)
    mask = cv2.warpAffine(np.full((512, 800), 255, np.uint8), inverse[:2], (800, 512), flags=cv2.INTER_NEAREST,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    overlap = ((mask > 0) & (valid > 0)).astype(np.uint8)
    expected_query = np.concatenate([api.direct._normalize(canvas), overlap[None].astype(np.float32)])
    assert np.array_equal(sample["query"].numpy(), expected_query)
    assert np.array_equal(sample["overlap"].numpy(), overlap[None].astype(np.float32))
    median = np.full((512, 800, 3), (22, 77, 188), np.uint8)
    expected_reference = api.direct._normalize(api.direct._warp(median, inverse))
    assert np.array_equal(sample["reference"].numpy(), expected_reference)


def test_changed_mask_and_query_content_rejected(api, inputs):
    value = context(api, inputs)
    table = api.build_evaluation_registration(value, inputs["tmp"] / "registration.json")
    dataset = api.ProtocolAEvaluationDataset(value, table, reference_enabled=False)
    zone = Path(inputs["manifest"]["banks"]["sidtd/example"]["enrollment"]["products"]["zone"]["path"])
    original = zone.read_bytes()
    zone.write_bytes(b"corrupt mask")
    with pytest.raises(ValueError, match="SHA256"):
        dataset[0]
    zone.write_bytes(original)
    (inputs["raw"] / "test0.png").write_bytes(b"corrupt query")
    with pytest.raises(ValueError, match="SHA256"):
        dataset[0]


def test_caller_binding_alias_is_detached_before_validation(api, inputs, monkeypatch):
    original = api.direct._json
    expected = dict(inputs["authorities"]["evaluation"])
    revoked = dict(inputs["evaluation"], source_validation_selected=False)
    replacement = write_json(inputs["tmp"] / "not-authorized.json", revoked)

    def mutate_original_after_read(authority):
        value = original(authority)
        if authority.get("path") == expected["path"]:
            inputs["authorities"]["evaluation"].update(replacement)
        return value

    monkeypatch.setattr(api.direct, "_json", mutate_original_after_read)
    value = context(api, inputs)
    state = json.loads(value.payload)
    assert state["evaluation_sha256"] == expected["sha256"]
    assert expected in [item["binding"] for item in state["authorities"]]
