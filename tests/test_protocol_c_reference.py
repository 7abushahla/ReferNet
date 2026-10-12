import copy
import hashlib
import importlib
import json
from pathlib import Path
import random
import shutil

import cv2
import numpy as np
import pandas as pd
import pytest
import torch

from test_protocol_c_contract import ROOT, contract_api, load, metadata, sha


def bind(path):
    return dict(path=str(path), sha256=sha(path))


def write(path, document):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.chmod(0o644)
    path.write_text(json.dumps(document, sort_keys=True, allow_nan=False))
    return bind(path)


@pytest.fixture
def api():
    assert (ROOT / "src/refer_or_clear/data/protocol_c_reference.py").exists(), "C evaluation consumer missing"
    return importlib.import_module("refer_or_clear.data.protocol_c_reference")


@pytest.fixture
def inputs(api, contract_api, metadata, tmp_path, monkeypatch):
    producer = importlib.import_module("refer_or_clear.data.protocol_c_enrollment")
    for row in metadata["records"]:
        row["attack_family"] = "text_inpaint" if row["label"] else ""
    pd.DataFrame(metadata["records"]).to_csv(metadata["paths"]["records"], index=False)
    monkeypatch.setattr(contract_api, "RECORDS_SHA256", sha(metadata["paths"]["records"]))
    contract = load(contract_api, metadata)
    target_binding = write(tmp_path / "target-enrollment.json", contract_api.reserve_target_enrollment(contract))
    target = contract_api.load_target_enrollment(contract, target_binding["path"], expected_sha256=target_binding["sha256"])
    raw = tmp_path / "raw"
    raw.mkdir()
    identifiers = set(target.support_ids + target.query_ids)
    canvas = np.full((64, 96, 3), (11, 66, 199), np.uint8)
    for row in metadata["records"]:
        if row["image_id"] not in identifiers:
            continue
        path = raw / row["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        assert cv2.imwrite(str(path), canvas)
        if row["image_id"] in target.support_ids:
            write(raw / row["annotation"], dict(regions=[dict(shape_attributes=dict(name="rect", x=8, y=10, width=20, height=12),
                  region_attributes=dict(field_name="name"))]))
    monkeypatch.setattr(producer.enrollment, "_features", lambda *args: None)
    monkeypatch.setattr(producer.enrollment, "_fit", lambda *args: (np.eye(3), 100))
    manifest = producer.make_protocol_c_enrollment_manifest(contract, target, raw)
    built = producer.build_protocol_c_enrollment_products(manifest, raw, tmp_path / "products")
    enrollment_binding = bind(tmp_path / "products/manifest.json")
    checkpoints = {}
    for arm in ("off", "on"):
        path = tmp_path / f"selected-a-{arm}.pt"
        path.write_bytes(f"synthetic selected A {arm}".encode())
        checkpoints[arm] = bind(path)
    evaluation = dict(schema_version="protocol_c_evaluation_v1", protocol="C",
        deployment="locked_a_selected_models_no_target_training", source_protocol="A", source_fold=0,
        source_validation_selected=True, records_sha256=contract.records_sha256,
        source_split_sha256=contract.source_split_sha256, target_split_sha256=contract.target_split_sha256,
        target_enrollment_sha256=target_binding["sha256"], enrollment_manifest_sha256=enrollment_binding["sha256"],
        authorized_image_ids=sorted(target.query_ids), checkpoints=checkpoints)
    evaluation_binding = write(tmp_path / "evaluation.json", evaluation)
    monkeypatch.setattr(api.direct, "_features", lambda *args: None)
    monkeypatch.setattr(api.direct, "_fit", lambda *args: (np.eye(3), 100))
    return dict(contract=contract, target=target, raw=raw, paths=metadata["paths"], metadata=metadata,
        target_binding=target_binding, enrollment_binding=enrollment_binding, evaluation_binding=evaluation_binding,
        evaluation=evaluation, manifest=built, tmp=tmp_path, root_binding=None)


def context(api, inputs):
    return api.load_reference_context(inputs["paths"]["records"], inputs["paths"]["source"], inputs["paths"]["target"],
        records_sha256=inputs["contract"].records_sha256, target_binding=inputs["target_binding"],
        enrollment_binding=inputs["enrollment_binding"], raw_root=inputs["raw"],
        evaluation_binding=inputs["evaluation_binding"], root_binding=inputs["root_binding"])


def rebind_evaluation(inputs):
    inputs["evaluation_binding"] = write(inputs["tmp"] / "evaluation.json", inputs["evaluation"])


def rebind_enrollment(inputs):
    inputs["enrollment_binding"] = write(Path(inputs["enrollment_binding"]["path"]), inputs["manifest"])
    inputs["evaluation"]["enrollment_manifest_sha256"] = inputs["enrollment_binding"]["sha256"]
    rebind_evaluation(inputs)


def test_context_authenticates_c_only_before_decoding_queries(api, inputs, monkeypatch):
    monkeypatch.setattr(api.direct, "_image", lambda *args: pytest.fail("context decoded pixels"))
    value = context(api, inputs)
    assert value.rows().image_id.tolist() == sorted(inputs["target"].query_ids)
    assert set(value.rows().role) == {"test"}
    assert not {"annotation", "component_id", "identity_id", "source_id", "donor_id"} & set(value.rows())
    with pytest.raises(ValueError):
        value.rows("train")
    with pytest.raises(ValueError):
        value.rows("val")
    assert json.loads(value.payload)["identities"]["protocol"] == "C"


@pytest.mark.parametrize("field,value", [("schema_version", "protocol_b_evaluation_v1"), ("protocol", "A"),
    ("source_protocol", "B"), ("source_fold", True), ("source_fold", 1), ("source_validation_selected", False),
    ("source_validation_selected", 1), ("deployment", "target_training"), ("authorized_image_ids", []),
    ("records_sha256", "f" * 64), ("source_split_sha256", "f" * 64), ("target_split_sha256", "f" * 64),
    ("target_enrollment_sha256", "f" * 64), ("checkpoints", {}), ("fold", 0), ("group", 0)])
def test_bad_evaluation_authority_never_decodes_target(api, inputs, monkeypatch, field, value):
    inputs["evaluation"][field] = value
    rebind_evaluation(inputs)
    monkeypatch.setattr(api.direct, "_image", lambda *args: pytest.fail("bad authority decoded pixels"))
    with pytest.raises(ValueError):
        context(api, inputs)


@pytest.mark.parametrize("arm", ["off", "on"])
def test_selected_checkpoint_bytes_are_required_before_contract_or_queries(api, inputs, monkeypatch, arm):
    Path(inputs["evaluation"]["checkpoints"][arm]["path"]).write_bytes(b"changed")
    monkeypatch.setattr(api, "load_protocol_c_contract", lambda *args, **kwargs: pytest.fail("unchecked source checkpoints"))
    with pytest.raises(ValueError, match="checkpoint.*SHA256"):
        context(api, inputs)


@pytest.mark.parametrize("failure", ["schema", "fold", "bank", "support", "contributor", "path", "product-hash", "geometry", "dependency"])
def test_banks_are_canonical_c_full_support_only(api, inputs, monkeypatch, failure):
    manifest = inputs["manifest"]
    design = inputs["contract"].target_designs[0]
    bank = manifest["banks"][design]["enrollment"]
    if failure == "schema":
        manifest["schema_version"] = "protocol_b_enrollment_v1"
    elif failure == "fold":
        bank["fold"] = 0
    elif failure == "bank":
        manifest["banks"][design]["train_half0"] = copy.deepcopy(bank)
    elif failure == "support":
        bank["support_ids"].append(inputs["target"].query_ids[0])
    elif failure == "contributor":
        next(iter(bank["contributors"].values()))["identity_id"] = "foreign"
    elif failure == "path":
        bank["products"]["median"] = manifest["banks"][inputs["contract"].target_designs[1]]["enrollment"]["products"]["median"]
    elif failure == "product-hash":
        bank["products"]["median"]["sha256"] = "f" * 64
    elif failure in ("geometry", "dependency"):
        name = "geometry" if failure == "geometry" else "manifest"
        path = Path(bank["products"][name]["path"])
        document = json.loads(path.read_text())
        if failure == "geometry":
            document["protocol"] = "A"
        else:
            document["output_sha256"]["zone"] = "f" * 64
        bank["products"][name] = write(path, document)
    rebind_enrollment(inputs)
    monkeypatch.setattr(api.direct, "_image", lambda *args: pytest.fail("bad support decoded"))
    with pytest.raises(ValueError):
        context(api, inputs)


def test_registration_off_on_shared_exact_remaining_ids_and_no_median_off(api, inputs, monkeypatch):
    value = context(api, inputs)
    path = inputs["tmp"] / "registration.json"
    table = api.build_protocol_c_registration(value, output_path=path)
    document = json.loads(path.read_text())
    assert document["schema_version"] == "protocol_c_registration_v1"
    assert set(document["entries"]) == set(inputs["target"].query_ids)
    assert not set(document["entries"]) & set(inputs["target"].support_ids + inputs["target"].excluded_ids)
    assert "fold" not in document and "group" not in document
    loaded = api.load_protocol_c_registration(value, bind(path))
    original = api.direct._image
    opened = []

    def image(binding, flags):
        opened.append(binding["path"])
        assert not binding["path"].endswith(".json")
        return original(binding, flags)

    monkeypatch.setattr(api.direct, "_image", image)
    monkeypatch.setattr(api.direct, "_fit", lambda *args: pytest.fail("dataset fitted lazily"))
    off = api.ProtocolCReferenceDataset(value, table, reference_enabled=False)[0]
    assert not any(Path(path).name in ("median.png", "target.png") for path in opened)
    on = api.ProtocolCReferenceDataset(value, loaded, reference_enabled=True)[0]
    assert torch.equal(off["query"], on["query"]) and torch.equal(off["overlap"], on["overlap"])
    assert off["meta"]["protocol"] == "C" and "fold" not in off["meta"] and "group" not in off["meta"]
    assert off["meta"]["image_id"] == on["meta"]["image_id"] == sorted(inputs["target"].query_ids)[0]
    assert tuple(on["query"].shape) == (4, 512, 800) and tuple(on["reference"].shape) == (3, 512, 800)
    assert "reference" not in off and bool(on["context_available"])
    assert not on["meta"]["requires_referral"]


def test_inverse_warp_masks_normalization_and_no_rng_consumption(api, inputs, monkeypatch):
    value = context(api, inputs)
    matrix = np.array([[1., 0., 12.], [0., 1., -4.], [0., 0., 1.]])
    monkeypatch.setattr(api.direct, "_fit", lambda *args: (matrix, 100))
    table = api.build_protocol_c_registration(value, output_path=inputs["tmp"] / "translated.json")
    dataset = api.ProtocolCReferenceDataset(value, table, reference_enabled=True)
    python_state, numpy_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
    sample = dataset[0]
    state = json.loads(value.payload)
    row = state["rows"][0]
    products = state["banks"][row["design_id"]]["products"]
    query, valid, unused = api.direct.to_canvas(cv2.imread(str(inputs["raw"] / row["path"])))
    masks = {name: cv2.imread(products[name]["path"], cv2.IMREAD_GRAYSCALE) for name in ("target_valid", "valid", "zone")}
    inverse, overlap = api.direct._overlap(valid, masks, matrix)
    zone = (api.direct._warp(masks["zone"], inverse, True) > 0).astype(np.uint8) * overlap
    median = cv2.imread(products["median"]["path"])
    assert torch.equal(sample["query"][:3], torch.from_numpy(api.direct._normalize(query)))
    assert np.array_equal(sample["query"][3].numpy(), zone.astype(np.float32))
    assert np.array_equal(sample["overlap"][0].numpy(), overlap.astype(np.float32))
    assert torch.equal(sample["reference"], torch.from_numpy(api.direct._normalize(api.direct._warp(median, inverse))))
    assert random.getstate() == python_state
    assert np.array_equal(np.random.get_state()[1], numpy_state[1])
    assert torch.equal(torch.get_rng_state(), torch_state)


@pytest.mark.parametrize("empty", [False, True])
def test_failed_or_empty_overlap_queries_stay_with_zero_context_and_referral(api, inputs, monkeypatch, empty):
    value = context(api, inputs)
    monkeypatch.setattr(api.direct, "_fit", lambda *args: (np.eye(3), 100) if empty else (None, 0))
    if empty:
        monkeypatch.setattr(api.direct, "_overlap", lambda *args: (np.eye(3), np.zeros((512, 800), np.uint8)))
    table = api.build_protocol_c_registration(value, output_path=inputs["tmp"] / "failed.json")
    dataset = api.ProtocolCReferenceDataset(value, table, reference_enabled=True)
    assert len(dataset) == len(inputs["target"].query_ids)
    sample = dataset[0]
    assert not bool(sample["context_available"]) and sample["meta"]["requires_referral"]
    assert torch.count_nonzero(sample["reference"]) == torch.count_nonzero(sample["query"][3]) == 0
    assert torch.count_nonzero(sample["overlap"]) == 0


@pytest.mark.parametrize("failure", ["schema", "support-id", "annotation", "component", "fold", "matrix", "plain", "bytes"])
def test_frozen_registration_rejects_foreign_or_mutated_inputs(api, inputs, failure):
    value = context(api, inputs)
    path = inputs["tmp"] / "registration.json"
    table = api.build_protocol_c_registration(value, output_path=path)
    document = json.loads(path.read_text())
    first = next(iter(document["entries"].values()))
    if failure == "plain":
        with pytest.raises(ValueError):
            api.ProtocolCReferenceDataset(value, document, reference_enabled=False)
        return
    if failure == "schema":
        document["schema_version"] = "protocol_a_evaluation_registration_v1"
    elif failure == "support-id":
        document["entries"][inputs["target"].support_ids[0]] = copy.deepcopy(first)
    elif failure in ("annotation", "component", "fold"):
        first[{"annotation": "annotation", "component": "component_id", "fold": "fold"}[failure]] = "forbidden"
    elif failure == "matrix":
        first["query_canvas_to_bank"][0][0] = 2.
    else:
        path.write_text("changed")
        with pytest.raises(ValueError):
            api.ProtocolCReferenceDataset(value, table, reference_enabled=False)
        return
    authority = write(path, document)
    with pytest.raises(ValueError):
        api.load_protocol_c_registration(value, authority)


@pytest.mark.parametrize("authority", ["records", "source", "target", "evaluation", "enrollment", "checkpoint", "query", "zone"])
def test_changed_files_fail_before_getter_decoding(api, inputs, monkeypatch, authority):
    value = context(api, inputs)
    table = api.build_protocol_c_registration(value, output_path=inputs["tmp"] / "registration.json")
    dataset = api.ProtocolCReferenceDataset(value, table, reference_enabled=False)
    row = value.rows().iloc[0]
    if authority in ("records", "source", "target"):
        path = inputs["paths"][authority]
    elif authority in ("evaluation", "enrollment"):
        path = Path(inputs[f"{authority}_binding"]["path"])
    elif authority == "checkpoint":
        path = Path(inputs["evaluation"]["checkpoints"]["on"]["path"])
    elif authority == "query":
        path = inputs["raw"] / row["path"]
    else:
        path = Path(inputs["manifest"]["banks"][row.design_id]["enrollment"]["products"]["zone"]["path"])
    path.chmod(0o644)
    with path.open("ab") as stream:
        stream.write(b"changed")
    monkeypatch.setattr(api.direct, "_image", lambda *args: pytest.fail("mutation reached decode"))
    with pytest.raises(ValueError):
        dataset[0]


def test_relocated_readonly_products_do_not_rewrite_original_manifest(api, inputs, monkeypatch):
    original = Path(inputs["manifest"]["output_root"])
    relocated = inputs["tmp"] / "relocated"
    shutil.copytree(original, relocated)
    manifest_bytes = Path(inputs["enrollment_binding"]["path"]).read_bytes()
    for path in relocated.rglob("*"):
        if path.is_file():
            path.chmod(0o444)
    inputs["root_binding"] = write(inputs["tmp"] / "product-root.json", dict(schema_version="protocol_c_product_root_v1",
        protocol="C", original_root=str(original), relocated_root=str(relocated),
        enrollment_manifest_sha256=inputs["enrollment_binding"]["sha256"]))
    value = context(api, inputs)
    state = json.loads(value.payload)
    assert all(Path(binding["path"]).is_relative_to(relocated)
               for bank in state["banks"].values() for binding in bank["products"].values())
    table = api.build_protocol_c_registration(value, output_path=inputs["tmp"] / "relocated-registration.json")
    assert api.ProtocolCReferenceDataset(value, table, reference_enabled=True)[0]["meta"]["protocol"] == "C"
    assert Path(inputs["enrollment_binding"]["path"]).read_bytes() == manifest_bytes


@pytest.mark.parametrize("location", ["raw", "products", "existing"])
def test_registration_output_protection_precedes_query_reads(api, inputs, monkeypatch, location):
    value = context(api, inputs)
    path = inputs["raw"] / "new.json" if location == "raw" else Path(inputs["manifest"]["output_root"]) / "new.json"
    if location == "existing":
        path = inputs["tmp"] / "keep.json"
        path.write_text("keep")
    monkeypatch.setattr(api.direct, "_image", lambda *args: pytest.fail("unsafe output decoded query"))
    with pytest.raises(ValueError):
        api.build_protocol_c_registration(value, output_path=path)
    if location == "existing":
        assert path.read_text() == "keep"
