import copy
from dataclasses import asdict
import hashlib
import importlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from refer_or_clear import protocol_b_contract as contract_api
from refer_or_clear.data import enrollment
from refer_or_clear.data.references import fold_of
from refer_or_clear.data.splits import FSL_GROUPS, NATIONALITY


@pytest.fixture
def api():
    assert importlib.util.find_spec("refer_or_clear.data.protocol_b_enrollment"), "Protocol B enrollment layer is missing"
    return importlib.import_module("refer_or_clear.data.protocol_b_enrollment")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def inventory(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    target_designs = tuple(sorted(NATIONALITY[name] for name in FSL_GROUPS[0].split("_")))
    source_designs = tuple(sorted(set(NATIONALITY.values()) - set(target_designs)))
    source_train, source_val, targets, records = [], [], [], []
    for design in source_designs + target_designs:
        for index in range(10 if design in target_designs else 8):
            identity = f"{design}_{index:02d}"
            image_id = identity.replace("/", "_")
            member = contract_api.MembershipRow(image_id, identity, identity, design, 0)
            attack = contract_api.MembershipRow(image_id + "_attack", identity, identity, design, 1)
            members = targets if design in target_designs else source_train
            members.extend((member, attack))
            for row in (member, attack):
                records.append(dict(asdict(row), dataset="sidtd", source_id=identity if row.label else "",
                                    donor_id="", path=f"{design}/{row.image_id}.png",
                                    annotation=f"{design}/layout.json", width=96, height=64))
        if design in source_designs:
            identity = f"{design}_20"
            row = contract_api.MembershipRow(identity.replace("/", "_"), identity, identity, design, 0)
            source_val.append(row)
            records.append(dict(asdict(row), dataset="sidtd", source_id="", donor_id="",
                                path=f"{design}/{row.image_id}.png", annotation=f"{design}/layout.json",
                                width=96, height=64))
    records_path = tmp_path / "records.csv"
    pd.DataFrame(records).to_csv(records_path, index=False)
    contract = contract_api.ProtocolBContract(0, FSL_GROUPS[0], sha(records_path),
                    contract_api.REVIEWED_SPLIT_SHA256[0], source_designs, target_designs,
                    tuple(source_train), tuple(source_val), tuple(targets))
    reservation = contract_api.reserve_target_enrollment(contract)
    manifest_path = tmp_path / "target.json"
    manifest_path.write_text(json.dumps(reservation))
    target = contract_api.load_target_enrollment(contract, manifest_path, expected_sha256=sha(manifest_path))
    permitted = {row.image_id for row in source_train if row.label == 0} | set(target.support_ids)
    texture = np.random.default_rng(42).integers(0, 256, (64, 96, 3), dtype=np.uint8)
    for design in source_designs + target_designs:
        directory = raw / design
        directory.mkdir(parents=True)
        metadata = {}
        for record in records:
            if record["design_id"] != design or record["label"] != 0:
                continue
            index = record["identity_id"].rsplit("_", 1)[-1]
            metadata[index] = {"filename": index + ".jpg", "regions": [{
                "shape_attributes": {"name": "rect", "x": 8, "y": 10, "width": 20, "height": 12},
                "region_attributes": {"field_name": "name"}}]}
            if record["image_id"] in permitted:
                assert cv2.imwrite(str(raw / record["path"]), texture)
        (directory / "layout.json").write_text(json.dumps({"_via_img_metadata": metadata}))
    return dict(raw=raw, records_path=records_path, records=records, contract=contract,
                target=target, permitted=permitted)


def make(api, inventory):
    return api.make_protocol_b_enrollment_manifest(inventory["records_path"], inventory["contract"],
                                                   inventory["target"], inventory["raw"])


def test_manifest_is_metadata_only_and_never_pretends_to_be_a(api, inventory, monkeypatch):
    monkeypatch.setattr(enrollment, "_contributor", lambda *args: pytest.fail("manifest read contributor pixels/layout"))
    before = inventory["records_path"].read_bytes()
    manifest = make(api, inventory)
    assert inventory["records_path"].read_bytes() == before
    assert manifest["schema_version"] == "protocol_b_enrollment_v1"
    assert manifest["protocol"] == "B" and manifest["group"] == 0 and "fold" not in manifest
    assert manifest["canvas_hw"] == [512, 800]
    assert manifest["target_enrollment"] == inventory["target"].as_dict()
    assert set(manifest["eligible_ids"]) == inventory["permitted"]
    for design in inventory["contract"].source_designs:
        assert set(manifest["banks"][design]) == {"train_half0", "train_half1", "enrollment"}
    for design in inventory["contract"].target_designs:
        assert set(manifest["banks"][design]) == {"enrollment"}


def test_source_training_uses_opposite_half_and_val_never_contributes(api, inventory):
    manifest = make(api, inventory)
    source_val_components = {row.component_id for row in inventory["contract"].source_val}
    assert not source_val_components & {row["component_id"] for row in manifest["contributors"].values()}
    for row in inventory["contract"].source_train:
        selected = api.resolve_protocol_b_bank(manifest, row.design_id, "train", row.component_id)
        assert selected["bank"] == f"train_half{1 - fold_of(row.component_id)}"
        assert row.component_id not in selected["component_ids"]
    for row in inventory["contract"].source_val:
        assert api.resolve_protocol_b_bank(manifest, row.design_id, "val")["bank"] == "enrollment"


def test_target_resolution_is_fixed_annotation_free_and_not_training(api, inventory):
    manifest = make(api, inventory)
    for design in inventory["contract"].target_designs:
        selected = api.resolve_protocol_b_bank(manifest, design, "test")
        assert selected == api.resolve_protocol_b_bank(manifest, design, "evaluation")
        expected = {row.image_id for row in inventory["contract"].target_test
                    if row.design_id == design and row.image_id in inventory["target"].support_ids}
        assert set(selected["support_ids"]) == expected
        with pytest.raises(ValueError):
            api.resolve_protocol_b_bank(manifest, design, "train", "query-component")
        with pytest.raises(ValueError, match="component"):
            api.resolve_protocol_b_bank(manifest, design, "test", "query-component")


@pytest.mark.parametrize("mutation", ["legacy", "group", "contributor", "half", "target-half", "target-query"])
def test_corrupt_manifest_fails_before_any_pixel_read(api, inventory, tmp_path, monkeypatch, mutation):
    manifest = make(api, inventory)
    if mutation == "legacy":
        manifest["schema_version"] = "anchor_enrollment_v1"
    elif mutation == "group":
        manifest["group"] = 1
    elif mutation == "contributor":
        manifest["eligible_ids"].append(inventory["contract"].source_val[0].image_id)
    elif mutation == "half":
        design = inventory["contract"].source_designs[0]
        manifest["banks"][design]["train_half0"]["support_ids"] = manifest["banks"][design]["train_half1"]["support_ids"]
    elif mutation == "target-half":
        design = inventory["contract"].target_designs[0]
        manifest["banks"][design]["train_half0"] = copy.deepcopy(manifest["banks"][design]["enrollment"])
    else:
        manifest["target_enrollment"]["query_ids"].append(manifest["target_enrollment"]["support_ids"][0])
    monkeypatch.setattr(cv2, "imread", lambda *args: pytest.fail("corrupt manifest opened pixels"))
    with pytest.raises(ValueError):
        api.build_protocol_b_enrollment_products(manifest, inventory["raw"], tmp_path / "products")


def test_product_build_reuses_original_geometry_reads_only_support_and_has_b_schema(api, inventory, tmp_path, monkeypatch):
    manifest = make(api, inventory)
    permitted_paths = {str(inventory["raw"] / row["path"]) for row in inventory["records"]
                       if row["image_id"] in inventory["permitted"]}
    original_read = cv2.imread
    opened = []

    def support_only(path, flags):
        assert path in permitted_paths, "query/validation/altered pixels were decoded"
        opened.append(path)
        return original_read(path, flags)

    monkeypatch.setattr(cv2, "imread", support_only)
    monkeypatch.setattr(enrollment, "_features", lambda image: ([], None))
    monkeypatch.setattr(enrollment, "_fit", lambda *args: (np.eye(3), 20))
    previous_threads, previous_opencl = cv2.getNumThreads(), cv2.ocl.useOpenCL()
    output = tmp_path / "products"
    result = api.build_protocol_b_enrollment_products(manifest, inventory["raw"], output)
    assert set(opened) == permitted_paths
    assert result["counts"]["banks"] == 22
    assert cv2.getNumThreads() == previous_threads and cv2.ocl.useOpenCL() == previous_opencl
    assert json.loads((output / "manifest.json").read_text())["schema_version"] == "protocol_b_enrollment_v1"
    for design, banks in result["banks"].items():
        for bank, selected in banks.items():
            for name in ("geometry", "manifest"):
                binding = selected["products"][name]
                document = json.loads(Path(binding["path"]).read_text())
                assert document["schema_version"] == "protocol_b_enrollment_v1"
                assert document["protocol"] == "B" and document["group"] == 0 and "fold" not in document
                assert binding["sha256"] == sha(Path(binding["path"]))
                assert document["canvas_hw"] == [512, 800]
            assert Path(selected["products"]["target"]["path"]).parent == output / "group0" / design / bank
    with pytest.raises(FileExistsError):
        api.build_protocol_b_enrollment_products(manifest, inventory["raw"], output)


def test_changed_records_authority_fails_before_products(api, inventory, tmp_path):
    manifest = make(api, inventory)
    inventory["records_path"].write_bytes(inventory["records_path"].read_bytes() + b"\n")
    with pytest.raises(ValueError, match="hash"):
        api.build_protocol_b_enrollment_products(manifest, inventory["raw"], tmp_path / "products")


@pytest.mark.parametrize("mutation", ["query-as-support", "foreign-components", "group-bool"])
def test_resolver_rejects_corrupt_membership_without_reading_query_metadata(api, inventory, mutation):
    manifest = make(api, inventory)
    design = inventory["contract"].target_designs[0]
    selected = manifest["banks"][design]["enrollment"]
    if mutation == "query-as-support":
        selected["support_ids"].append(next(row.image_id for row in inventory["contract"].target_test
                                          if row.design_id == design and row.image_id in inventory["target"].query_ids))
    elif mutation == "foreign-components":
        selected["component_ids"] = ["foreign"]
    else:
        manifest["group"] = False
    with pytest.raises(ValueError):
        api.resolve_protocol_b_bank(manifest, design, "test")


def test_missing_explicit_reservation_fails_closed(api, inventory):
    with pytest.raises(ValueError):
        api.make_protocol_b_enrollment_manifest(inventory["records_path"], inventory["contract"],
                                                None, inventory["raw"])


def test_failed_bank_build_restores_opencv_state_and_never_falls_back(api, inventory, tmp_path, monkeypatch):
    manifest = make(api, inventory)
    previous_threads, previous_opencl = cv2.getNumThreads(), cv2.ocl.useOpenCL()

    def fail_bank(*args):
        raise ValueError("empty successfully registered support")

    monkeypatch.setattr(enrollment, "_build_bank", fail_bank)
    with pytest.raises(ValueError, match="registered support"):
        api.build_protocol_b_enrollment_products(manifest, inventory["raw"], tmp_path / "products")
    assert cv2.getNumThreads() == previous_threads and cv2.ocl.useOpenCL() == previous_opencl


def test_product_dependency_hashes_match_rewritten_b_geometry(api, inventory, tmp_path, monkeypatch):
    manifest = make(api, inventory)
    design = inventory["contract"].target_designs[0]
    real_builder = enrollment._build_bank

    def one_bank(manifest, selected, root, directory, blank, archives):
        if selected["design_id"] == design:
            return real_builder(manifest, selected, root, directory, blank, archives)
        products = {}
        directory.mkdir(parents=True)
        geometry = dict(schema_version="anchor_enrollment_v1", fold=selected["fold"], registration={})
        products["geometry"] = enrollment._write_json(directory / "geometry.json", geometry)
        products["manifest"] = enrollment._write_json(directory / "manifest.json", {})
        contributors = {identifier: manifest["contributors"][identifier] for identifier in selected["support_ids"]}
        return dict(selected, products=products, contributors=contributors)

    monkeypatch.setattr(enrollment, "_build_bank", one_bank)
    monkeypatch.setattr(enrollment, "_features", lambda image: ([], None))
    monkeypatch.setattr(enrollment, "_fit", lambda *args: (np.eye(3), 20))
    result = api.build_protocol_b_enrollment_products(manifest, inventory["raw"], tmp_path / "products")
    bank = result["banks"][design]["enrollment"]
    dependency = json.loads(Path(bank["products"]["manifest"]["path"]).read_text())
    assert dependency["output_sha256"] == {name: binding["sha256"] for name, binding in bank["products"].items()
                                           if name != "manifest"}


def test_resolver_returns_detached_bank_and_no_global_fallback(api, inventory):
    manifest = make(api, inventory)
    design = inventory["contract"].target_designs[0]
    selected = api.resolve_protocol_b_bank(manifest, design, "test")
    selected["support_ids"].clear()
    assert manifest["banks"][design]["enrollment"]["support_ids"]
    del manifest["banks"][design]
    with pytest.raises(ValueError, match="no global fallback"):
        api.resolve_protocol_b_bank(manifest, design, "test")
