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

from test_enrollment import inventory
from test_protocol_a_contract import ROOT, contract_api, sha


def bind(path):
    return dict(path=str(Path(path).resolve()), sha256=sha(path))


def write(path, document):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, sort_keys=True, allow_nan=False))
    return bind(path)


@pytest.fixture
def api():
    assert (ROOT / "src/refer_or_clear/data/protocol_a_reference.py").exists(), "fold-aware A consumer missing"
    return importlib.import_module("refer_or_clear.data.protocol_a_reference")


@pytest.fixture
def inputs(api, contract_api, inventory, tmp_path, monkeypatch):
    raw, records, split, annotation, texture = inventory
    fold = 4
    split["ref_group"] = f"it{fold}"
    records["attack_family"] = records.label.map({0: "", 1: "text_copy"})
    paths = dict(records=tmp_path / "records.csv", split=tmp_path / "split.csv")
    records.to_csv(paths["records"], index=False)
    split.to_csv(paths["split"], index=False)
    monkeypatch.setattr(contract_api, "RECORDS_SHA256", sha(paths["records"]))
    monkeypatch.setattr(contract_api, "REVIEWED_SPLIT_SHA256", {fold: sha(paths["split"])})
    enrollment = importlib.import_module("refer_or_clear.data.enrollment")
    monkeypatch.setattr(enrollment, "_source", lambda: dict(implementation_sha256=sha(Path(enrollment.__file__)),
        helpers={name: sha(Path(enrollment.__file__).with_name(name))
                 for name in ("records.py", "rectify.py", "references.py", "register.py")}, revision="synthetic"))
    monkeypatch.setattr(enrollment, "_features", lambda *args: None)
    monkeypatch.setattr(enrollment, "_fit", lambda *args: (np.eye(3), 100))
    manifest = enrollment.make_enrollment_manifest(records, split, fold, raw)
    manifest = enrollment.build_enrollment_products(manifest, raw, tmp_path / "products")
    manifest["input_files"] = {name: bind(path) for name, path in paths.items()}
    enrollment_binding = write(tmp_path / "products/manifest.json", manifest)
    for row in records.itertuples():
        path = raw / row.path
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            assert cv2.imwrite(str(path), texture)
    monkeypatch.setattr(api, "_features", lambda *args: None)
    monkeypatch.setattr(api, "_fit", lambda *args: (np.eye(3), 100))
    return dict(paths=paths, records=records, split=split, raw=raw, manifest=manifest,
        fold=fold, enrollment_binding=enrollment_binding, root_binding=None,
        metadata_relocation_binding=None, evaluation_binding=None, tmp=tmp_path)


def context(api, inputs, **overrides):
    arguments = dict(fold=inputs["fold"], records_sha256=sha(inputs["paths"]["records"]),
        enrollment_binding=inputs["enrollment_binding"], raw_root=inputs["raw"],
        root_binding=inputs["root_binding"], metadata_relocation_binding=inputs["metadata_relocation_binding"],
        evaluation_binding=inputs["evaluation_binding"])
    arguments.update(overrides)
    return api.load_reference_context(inputs["paths"]["records"], inputs["paths"]["split"], **arguments)


def table(api, inputs, value=None, roles=("train", "val")):
    return api.build_protocol_a_registration(value or context(api, inputs), roles=roles,
        output_path=inputs["tmp"] / "registration.json")


def dataset(api, value, registration, enabled, role="train", cache_bytes=0, augment=None):
    return api.ProtocolAReferenceDataset(value, registration, role=role, reference_enabled=enabled,
        augment=augment or {"enabled": False}, cache_bytes=cache_bytes)


def rebind(inputs):
    inputs["enrollment_binding"] = write(Path(inputs["enrollment_binding"]["path"]), inputs["manifest"])


def test_context_preserves_published_overlap_excludes_support_without_query_reads(api, inputs, monkeypatch):
    monkeypatch.setattr(api, "_image", lambda *args: pytest.fail("context decoded pixels"))
    value = context(api, inputs)
    assert value.contract.fold == 4 and not hasattr(value.contract, "group")
    assert value.rows("train").image_id.tolist() == sorted(inputs["split"].query("role == 'train'").image_id)
    assert tuple(row.image_id for row in value.contract.eligible_support) == ("g1", "g2", "g3", "g4")
    assert not {"annotation", "identity_id", "component_id", "source_id", "donor_id"} & set(value.rows("val"))
    with pytest.raises(ValueError, match="test"):
        value.rows("test")
    detached = value.manifest
    detached["fold"] = 9
    assert value.manifest["fold"] == 4
    rows = value.rows("train")
    rows.loc[0, "path"] = "changed"
    assert value.rows("train").loc[0, "path"] != "changed"


@pytest.mark.parametrize("change", ["fold", "binding", "counts", "eligible", "forbidden", "contributors", "bank-support", "bank-path", "geometry", "dependency"])
def test_authenticated_manifest_still_rejects_invalid_bank_authority(api, inputs, change, monkeypatch):
    manifest = inputs["manifest"]
    bank = manifest["banks"]["sidtd/alb_id"]["enrollment"]
    if change == "fold": manifest["fold"] = 9
    elif change == "binding": manifest["official_fold_binding"]["fold"] = 9
    elif change == "counts": manifest["classifier_counts"]["val"] = 222
    elif change == "eligible": manifest["eligible_ids"].append("g0")
    elif change == "forbidden": manifest["forbidden_components"] = []
    elif change == "contributors": manifest["contributors"]["g1"]["component_id"] = "foreign"
    elif change == "bank-support": bank["support_ids"].append("g0")
    elif change == "bank-path": bank["products"]["median"]["path"] = bank["products"]["target"]["path"]
    else:
        name = "geometry" if change == "geometry" else "manifest"
        product = Path(bank["products"][name]["path"])
        document = json.loads(product.read_text())
        document["fold"] = 9
        bank["products"][name] = write(product, document)
    rebind(inputs)
    monkeypatch.setattr(api, "_image", lambda *args: pytest.fail("bad metadata decoded pixels"))
    with pytest.raises(ValueError): context(api, inputs)


def test_metadata_relocation_explicit_and_original_manifest_unchanged(api, inputs):
    original = Path(inputs["enrollment_binding"]["path"]).read_bytes()
    old_paths = inputs["paths"].copy()
    for name, path in old_paths.items():
        destination = inputs["tmp"] / "durable" / path.name
        destination.parent.mkdir(exist_ok=True)
        shutil.copyfile(path, destination)
        inputs["paths"][name] = destination
        path.unlink()
    with pytest.raises(ValueError, match="relocation"):
        context(api, inputs)
    document = dict(schema_version="protocol_a_metadata_relocation_v1", protocol="A", fold=4,
        enrollment_manifest_sha256=inputs["enrollment_binding"]["sha256"],
        relocations={name: dict(original_path=str(old_paths[name]), **bind(path)) for name, path in inputs["paths"].items()})
    inputs["metadata_relocation_binding"] = write(inputs["tmp"] / "relocation.json", document)
    value = context(api, inputs)
    assert value.contract.fold == 4
    assert Path(inputs["enrollment_binding"]["path"]).read_bytes() == original
    document["relocations"]["split"]["original_path"] = "/arbitrary"
    inputs["metadata_relocation_binding"] = write(inputs["tmp"] / "relocation.json", document)
    with pytest.raises(ValueError): context(api, inputs)


def test_obsolete_cpu_tmp_namespace_is_not_resolved_on_consumer_host(api, inputs):
    for name in ("records", "split"):
        inputs["manifest"]["input_files"][name]["path"] = f"/tmp/retired-cpu-job/inputs/{name}.csv"
    rebind(inputs)
    document = dict(schema_version="protocol_a_metadata_relocation_v1", protocol="A", fold=4,
        enrollment_manifest_sha256=inputs["enrollment_binding"]["sha256"],
        relocations={name: dict(original_path=inputs["manifest"]["input_files"][name]["path"], **bind(path))
                     for name, path in inputs["paths"].items()})
    inputs["metadata_relocation_binding"] = write(inputs["tmp"] / "relocation.json", document)
    assert context(api, inputs).contract.fold == 4


def test_root_relocation_exact_and_no_original_product_open(api, inputs):
    old_root = Path(inputs["manifest"]["output_root"])
    relocated = inputs["tmp"] / "relocated"
    shutil.copytree(old_root, relocated)
    for path in (old_root / "fold4").rglob("*"):
        if path.is_file(): path.unlink()
    document = dict(schema_version="protocol_a_product_root_v1", protocol="A", fold=4,
        enrollment_manifest_sha256=inputs["enrollment_binding"]["sha256"],
        original_root=str(old_root), relocated_root=str(relocated))
    inputs["root_binding"] = write(inputs["tmp"] / "root.json", document)
    value = context(api, inputs)
    registration = table(api, inputs, value)
    assert all(Path(product["path"]).is_relative_to(relocated)
        for entry in registration["entries"].values() for product in entry["products"].values())
    assert dataset(api, value, registration, True)[0]["query"].shape == (4, 512, 800)
    document["fold"] = 9
    inputs["root_binding"] = write(inputs["tmp"] / "root.json", document)
    with pytest.raises(ValueError): context(api, inputs)


@pytest.mark.parametrize("enabled", [False, True])
def test_cached_uncached_augmented_tensors_private_rng_and_mutation(api, inputs, enabled):
    value = context(api, inputs)
    registration = table(api, inputs, value)
    augment = dict(enabled=True, seed=42, photometric=True, jitter_px=3, jitter_p=1.0)
    plain = dataset(api, value, registration, enabled, augment=augment)
    cached = dataset(api, value, registration, enabled, cache_bytes=64 * 1024**2, augment=augment)
    before = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
    for index in [0, 1, 2, 0, 1, 2]:
        first, second = plain[index], cached[index]
        assert first.keys() == second.keys()
        for key in first:
            if torch.is_tensor(first[key]): assert torch.equal(first[key], second[key])
            else: assert first[key] == second[key]
        assert plain.geometry_rng.getstate() == cached.geometry_rng.getstate()
        assert plain.photometry_rng.getstate() == cached.photometry_rng.getstate()
        second["query"].fill_(999)
    assert random.getstate() == before[0] and torch.equal(torch.get_rng_state(), before[2])
    assert np.array_equal(np.random.get_state()[1], before[1][1])
    query = value.raw_root / value.rows("train").loc[0, "path"]
    query.write_bytes(b"changed after warm cache")
    assert cached[0]["query"].shape == first["query"].shape
    with pytest.raises(ValueError): plain[0]


def test_off_on_same_query_masks_labels_and_off_no_median(api, inputs, monkeypatch):
    value = context(api, inputs)
    registration = table(api, inputs, value)
    on = dataset(api, value, registration, True)
    off = dataset(api, value, registration, False)
    observed = []
    original = api._bytes
    def audited(binding):
        observed.append(Path(binding["path"]).name)
        return original(binding)
    monkeypatch.setattr(api, "_bytes", audited)
    for index in range(len(off)):
        first = off[index]
        assert "median.png" not in observed
        observed.clear()
        second = on[index]
        for key in ("query", "overlap", "context_available"):
            assert torch.equal(first[key], second[key])
        assert first["label"] == second["label"] and first["meta"] == second["meta"]
        assert "reference" not in first and second["reference"].shape == (3, 512, 800)
        observed.clear()
    assert dataset(api, value, registration, True, role="val")[0]["meta"]["bank"] == "enrollment"


@pytest.mark.parametrize("failure", ["fit", "empty"])
def test_failures_remain_population_and_zero_reference(api, inputs, monkeypatch, failure):
    if failure == "fit": monkeypatch.setattr(api, "_fit", lambda *args: (None, 0))
    else: monkeypatch.setattr(api, "_available", lambda *args: False)
    value = context(api, inputs)
    registration = table(api, inputs, value)
    on = dataset(api, value, registration, True)
    assert len(on) == len(value.contract.source_train)
    for index in range(len(on)):
        result = on[index]
        assert not result["context_available"] and result["meta"]["requires_referral"]
        assert torch.count_nonzero(result["reference"]) == 0
        assert torch.count_nonzero(result["overlap"]) == 0


def test_table_immutable_no_refit_and_guarded_membership(api, inputs, monkeypatch):
    value = context(api, inputs)
    registration = table(api, inputs, value)
    detached = registration["entries"]
    detached.clear()
    monkeypatch.setattr(api, "_fit", lambda *args: pytest.fail("getter fitted registration"))
    assert dataset(api, value, registration, True)[0]["query"].shape == (4, 512, 800)
    document = json.loads(registration.serialized)
    document["entries"].pop(next(iter(document["entries"])))
    binding = write(inputs["tmp"] / "other-table.json", document)
    with pytest.raises(ValueError): api.load_protocol_a_registration(value, binding, roles=["train", "val"])
    with pytest.raises(ValueError): table(api, inputs, value, roles=["test"])
    with pytest.raises(ValueError): table(api, inputs, value)


@pytest.mark.parametrize("mutation", ["authority", "table"])
def test_warm_cache_revalidates_authority_files(api, inputs, mutation):
    value = context(api, inputs)
    registration = table(api, inputs, value)
    warm = dataset(api, value, registration, True, cache_bytes=64 * 1024**2)
    warm[0]
    path = Path(inputs["enrollment_binding"]["path"]) if mutation == "authority" else inputs["tmp"] / "registration.json"
    path.write_bytes(b"changed")
    with pytest.raises(ValueError): warm[0]


@pytest.mark.parametrize("mutation", ["zone", "geometry", "manifest", "median"])
def test_product_files_are_authenticated_at_build_not_on_warm_hits(api, inputs, mutation):
    value = context(api, inputs)
    registration = table(api, inputs, value)
    cold = dataset(api, value, registration, True, cache_bytes=0)
    warm = dataset(api, value, registration, True, cache_bytes=64 * 1024**2)
    expected = warm[0]
    entry = registration["entries"][value.rows("train").loc[0, "image_id"]]
    Path(entry["products"][mutation]["path"]).write_bytes(b"changed")
    assert torch.equal(warm[0]["query"], expected["query"])
    with pytest.raises(ValueError): cold[0]


def test_test_authority_separate_both_checkpoints_no_provenance_routing(api, inputs):
    value = context(api, inputs)
    checkpoints = {}
    for arm in ("off", "on"):
        path = inputs["tmp"] / f"{arm}.pt"
        path.write_bytes(b"synthetic binding, not completion proof")
        checkpoints[arm] = bind(path)
    document = dict(schema_version="protocol_a_evaluation_v1", **json.loads(value.identities),
        source_validation_selected=True, authorized_image_ids=sorted(row.image_id for row in value.contract.target_test),
        checkpoints=checkpoints)
    inputs["evaluation_binding"] = write(inputs["tmp"] / "evaluation.json", document)
    authorized = context(api, inputs)
    assert not {"identity_id", "component_id", "annotation"} & set(authorized.rows("test"))
    registration = table(api, inputs, authorized, roles=["test"])
    assert all(entry["bank"] == "enrollment" and "training_component" not in entry for entry in registration["entries"].values())
    assert dataset(api, authorized, registration, True, role="test")[0]["meta"]["role"] == "test"
    Path(checkpoints["off"]["path"]).write_bytes(b"changed")
    with pytest.raises(ValueError): context(api, inputs)


def test_fold9_native_consumer_paths_and_typed_registration(api, inputs, contract_api, monkeypatch):
    inputs["fold"] = 9
    inputs["split"]["ref_group"] = "it9"
    inputs["split"].to_csv(inputs["paths"]["split"], index=False)
    monkeypatch.setattr(contract_api, "REVIEWED_SPLIT_SHA256", {9: sha(inputs["paths"]["split"])})
    manifest = inputs["manifest"]
    manifest.update(fold=9, official_fold_binding=dict(protocol="A", fold=9, ref_group="it9"),
        official_split_sha256=api.enrollment._table_hash(inputs["split"]))
    manifest["input_files"]["split"] = bind(inputs["paths"]["split"])
    root = Path(manifest["output_root"])
    (root / "fold4").rename(root / "fold9")
    for bank in manifest["banks"]["sidtd/alb_id"].values():
        bank["fold"] = 9
        for product in bank["products"].values():
            product["path"] = str(root / "fold9" / Path(product["path"]).relative_to(root / "fold4"))
        path = Path(bank["products"]["geometry"]["path"])
        geometry = json.loads(path.read_text())
        geometry.update(fold=9, official_split_sha256=manifest["official_split_sha256"])
        bank["products"]["geometry"] = write(path, geometry)
        path = Path(bank["products"]["manifest"]["path"])
        bank["products"]["manifest"] = write(path, dict(geometry,
            output_sha256={name: product["sha256"] for name, product in bank["products"].items() if name != "manifest"}))
    rebind(inputs)
    value = context(api, inputs)
    registration = table(api, inputs, value)
    assert json.loads(value.identities)["fold"] == 9
    assert all(entry["fold"] == 9 and "group" not in entry for entry in registration["entries"].values())
    assert dataset(api, value, registration, True)[0]["meta"]["fold"] == 9


def test_evaluation_augmentation_and_symlink_query_denied(api, inputs):
    value = context(api, inputs)
    registration = table(api, inputs, value)
    with pytest.raises(ValueError, match="augmentation"):
        dataset(api, value, registration, True, role="val", augment={"enabled": True})
    cold = dataset(api, value, registration, True, cache_bytes=0)
    cold[0]
    query = value.raw_root / value.rows("train").loc[0, "path"]
    retained = query.with_suffix(".retained")
    query.rename(retained)
    query.symlink_to(retained)
    with pytest.raises(ValueError, match="symlink"): cold[0]


@pytest.mark.parametrize("change", ["bank", "fold", "train-component", "val-component", "val-annotation", "matrix", "roles", "query-hash"])
def test_table_semantic_authority_cannot_be_forged_with_new_file_hash(api, inputs, change):
    value = context(api, inputs)
    registration = table(api, inputs, value)
    document = json.loads(registration.serialized)
    first = document["entries"][value.rows("train").loc[0, "image_id"]]
    if change == "bank": first["bank"] = "enrollment"
    elif change == "fold": first["fold"] = 9
    elif change == "train-component": first["training_component"] = "foreign"
    elif change == "val-component": document["entries"][value.rows("val").loc[0, "image_id"]]["training_component"] = "hidden"
    elif change == "val-annotation": document["entries"][value.rows("val").loc[0, "image_id"]]["annotation"] = "hidden.json"
    elif change == "matrix": first["query_canvas_to_bank"][0][0] = -1
    elif change == "roles": document["roles"] = ["train"]
    else: first["query_sha256"] = "f" * 64
    binding = write(inputs["tmp"] / "forged-table.json", document)
    with pytest.raises(ValueError): api.load_protocol_a_registration(value, binding, roles=["train", "val"])


def test_all_bank_planes_use_inverse_query_geometry_without_annotation_routing(api, inputs, monkeypatch):
    query_to_bank = np.array([[1., 0., 14.], [0., 1., -7.], [0., 0., 1.]])
    monkeypatch.setattr(api, "_fit", lambda *args: (query_to_bank, 100))
    zone = np.zeros((512, 800), np.uint8)
    zone[100:180, 150:250] = 255
    banks = inputs["manifest"]["banks"]["sidtd/alb_id"]
    for bank in banks.values():
        products = bank["products"]
        assert cv2.imwrite(products["zone"]["path"], zone)
        products["zone"] = bind(products["zone"]["path"])
        geometry = json.loads(Path(products["geometry"]["path"]).read_bytes())
        products["manifest"] = write(Path(products["manifest"]["path"]), dict(geometry,
            output_sha256={name: product["sha256"] for name, product in products.items() if name != "manifest"}))
    rebind(inputs)
    value = context(api, inputs)
    registration = table(api, inputs, value)
    row = value.rows("val").iloc[0]
    Path(inputs["raw"] / inputs["records"].set_index("image_id").loc[row.image_id, "annotation"]).parent.mkdir(parents=True, exist_ok=True)
    entry = registration["entries"][row.image_id]
    inverse = np.linalg.inv(query_to_bank)[:2]
    target_valid = cv2.imread(entry["products"]["target_valid"]["path"], cv2.IMREAD_GRAYSCALE)
    valid = cv2.imread(entry["products"]["valid"]["path"], cv2.IMREAD_GRAYSCALE)
    expected_overlap = ((cv2.warpAffine(target_valid, inverse, (800, 512), flags=cv2.INTER_NEAREST) > 0)
                        & (cv2.warpAffine(valid, inverse, (800, 512), flags=cv2.INTER_NEAREST) > 0)).astype(np.uint8)
    expected_zone = (cv2.warpAffine(zone, inverse, (800, 512), flags=cv2.INTER_NEAREST) > 0).astype(np.uint8) * expected_overlap
    sample = dataset(api, value, registration, True, role="val")[0]
    assert np.array_equal(sample["overlap"].numpy()[0], expected_overlap)
    assert np.array_equal(sample["query"].numpy()[3], expected_zone)
    median = cv2.imread(entry["products"]["median"]["path"], cv2.IMREAD_COLOR)
    expected_reference = cv2.warpAffine(median, inverse, (800, 512), flags=cv2.INTER_LINEAR)
    assert np.array_equal(sample["reference"].numpy(), api._normalize(expected_reference))


def test_cache_hit_reads_no_input_files_cold_build_reads_all(api, inputs, monkeypatch):
    value = context(api, inputs)
    registration = table(api, inputs, value)
    warm = dataset(api, value, registration, True, cache_bytes=64 * 1024**2)
    warm[0]
    observed = []
    original = api._bytes
    def audited(binding):
        observed.append(binding["path"])
        return original(binding)
    monkeypatch.setattr(api, "_bytes", audited)
    warm[0]
    assert observed == []
    cold = dataset(api, value, registration, True, cache_bytes=0)
    cold[0]
    row = value.rows("train").iloc[0]
    products = registration["entries"][row.image_id]["products"]
    expected = [str(value.raw_root / row.path)] + [products[name]["path"]
        for name in ("geometry", "manifest", "target_valid", "valid", "zone", "median")]
    assert observed == expected
