import copy
import importlib
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from test_protocol_c_contract import ROOT, contract_api, load, metadata, sha


@pytest.fixture
def api():
    assert (ROOT / "src/refer_or_clear/data/protocol_c_enrollment.py").exists(), "C CPU support producer missing"
    return importlib.import_module("refer_or_clear.data.protocol_c_enrollment")


@pytest.fixture
def inventory(api, contract_api, metadata):
    contract = load(contract_api, metadata)
    document = contract_api.reserve_target_enrollment(contract)
    path = metadata["tmp"] / "reserved.json"
    path.write_text(json.dumps(document))
    target = contract_api.load_target_enrollment(contract, path, expected_sha256=sha(path))
    raw = metadata["tmp"] / "raw"
    raw.mkdir()
    permitted = set(target.support_ids)
    texture = np.random.default_rng(42).integers(0, 256, (64, 96, 3), dtype=np.uint8)
    for row in metadata["records"]:
        if row["image_id"] not in permitted:
            continue
        image = raw / row["path"]
        image.parent.mkdir(parents=True, exist_ok=True)
        assert cv2.imwrite(str(image), texture)
        (raw / row["annotation"]).write_text(json.dumps({"regions": [{
            "shape_attributes": dict(name="rect", x=8, y=10, width=20, height=12),
            "region_attributes": {"field_name": "name"}}]}))
    return dict(metadata=metadata, contract=contract, target=target, raw=raw, permitted=permitted)


def make(api, inventory):
    return api.make_protocol_c_enrollment_manifest(inventory["contract"], inventory["target"], inventory["raw"])


def test_manifest_metadata_only_full_target_banks_no_checkpoint_gate(api, inventory, monkeypatch):
    monkeypatch.setattr(api.enrollment, "_contributor", lambda *args: pytest.fail("metadata decoded support"))
    manifest = make(api, inventory)
    assert manifest["schema_version"] == "protocol_c_enrollment_v1"
    assert manifest["protocol"] == "C" and "fold" not in manifest and "group" not in manifest
    assert set(manifest["eligible_ids"]) == inventory["permitted"]
    assert len(manifest["banks"]) == 3
    assert all(set(banks) == {"enrollment"} for banks in manifest["banks"].values())
    assert all(item["dataset"] == "fantasyid" for item in manifest["contributors"].values())
    assert not set(manifest["contributors"]) & {row.image_id for row in inventory["contract"].source_train}


@pytest.mark.parametrize("mutation", ["schema", "eligible", "target-half", "query", "path", "dimensions"])
def test_corrupt_authority_blocks_pixels_and_output(api, inventory, tmp_path, monkeypatch, mutation):
    manifest = make(api, inventory)
    design = inventory["contract"].target_designs[0]
    if mutation == "schema":
        manifest["schema_version"] = "anchor_enrollment_v1"
    elif mutation == "eligible":
        manifest["eligible_ids"].append(inventory["target"].query_ids[0])
    elif mutation == "target-half":
        manifest["banks"][design]["train_half0"] = copy.deepcopy(manifest["banks"][design]["enrollment"])
    elif mutation == "query":
        manifest["target_enrollment"]["query_ids"].append(inventory["target"].support_ids[0])
    else:
        first = next(iter(manifest["contributors"].values()))
        first["path" if mutation == "path" else "width"] = "../escape.png" if mutation == "path" else 95
    monkeypatch.setattr(cv2, "imread", lambda *args: pytest.fail("invalid manifest decoded pixels"))
    output = tmp_path / "products"
    with pytest.raises(ValueError):
        api.build_protocol_c_enrollment_products(manifest, inventory["raw"], output)
    assert not output.exists()


def test_build_reads_only_reserved_genuine_and_reuses_geometry(api, inventory, tmp_path, monkeypatch):
    manifest = make(api, inventory)
    paths = {str(inventory["raw"] / row["path"]) for row in inventory["metadata"]["records"] if row["image_id"] in inventory["permitted"]}
    original = cv2.imread
    opened = []

    def only_support(path, flags):
        assert path in paths, "unreserved/altered/source pixels decoded"
        opened.append(path)
        return original(path, flags)

    monkeypatch.setattr(cv2, "imread", only_support)
    monkeypatch.setattr(api.enrollment, "_features", lambda *args: None)
    monkeypatch.setattr(api.enrollment, "_fit", lambda *args: (np.eye(3), 100))
    previous = cv2.getNumThreads(), cv2.ocl.useOpenCL()
    output = tmp_path / "products"
    result = api.build_protocol_c_enrollment_products(manifest, inventory["raw"], output)
    assert set(opened) == paths and len(opened) == len(paths)
    assert (cv2.getNumThreads(), cv2.ocl.useOpenCL()) == previous
    assert result["counts"]["designs"] == result["counts"]["banks"] == 3
    for design, banks in result["banks"].items():
        bank = banks["enrollment"]
        assert "fold" not in bank and bank["protocol"] == "C"
        assert len(bank["products"]) == 9
        for product, authority in bank["products"].items():
            expected = output / "groupC" / design / "enrollment" / f"{product}{'.json' if product in ('geometry','manifest') else '.png'}"
            assert Path(authority["path"]) == expected
            assert sha(expected) == authority["sha256"]
        geometry = json.loads(Path(bank["products"]["geometry"]["path"]).read_text())
        assert geometry["schema_version"] == "protocol_c_enrollment_v1"
        assert "fold" not in geometry and "official_split_sha256" not in geometry
        assert geometry["registration"]["seed_slot_meaning"] == "protocol_c_fixed_geometry_slot_not_fold"
        assert geometry["axes"] == "x=column,y=row; native_to_bank=canvas_to_bank@native_to_canvas"
        dependency = json.loads(Path(bank["products"]["manifest"]["path"]).read_text())
        assert {key:value for key,value in dependency.items() if key != "output_sha256"} == geometry
    assert json.loads((output / "manifest.json").read_text())["target_enrollment"] == inventory["target"].as_dict()


def test_failed_bank_not_reported_as_completed(api, inventory, tmp_path, monkeypatch):
    manifest = make(api, inventory)
    monkeypatch.setattr(api.enrollment, "_features", lambda *args: None)
    monkeypatch.setattr(api.enrollment, "_fit", lambda *args: (None, 0))
    output = tmp_path / "failed"
    with pytest.raises(ValueError, match="empty successfully registered support"):
        api.build_protocol_c_enrollment_products(manifest, inventory["raw"], output)
    assert not (output / "manifest.json").exists()


def test_output_cannot_overlap_inputs(api, inventory, monkeypatch):
    monkeypatch.setattr(cv2, "imread", lambda *args: pytest.fail("unsafe output opened pixels"))
    with pytest.raises(ValueError):
        api.build_protocol_c_enrollment_products(make(api, inventory), inventory["raw"], inventory["raw"] / "products")


def test_script_metadata_phase_without_pixel_reads(api, inventory, tmp_path, monkeypatch):
    script = ROOT / "scripts/prepare_protocol_c.py"
    assert script.exists(), "C preparation CLI missing"
    spec = importlib.util.spec_from_file_location("prepare_c_test", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(cv2, "imread", lambda *args: pytest.fail("metadata phase opened pixels"))
    paths = inventory["metadata"]["paths"]
    result = module.prepare(paths["records"], sha(paths["records"]), paths["source"], paths["target"],
                            inventory["raw"], tmp_path / "preparation", build=False)
    assert result["phase"] == "metadata" and result["enrollment"] is None
    assert not result["target_query_pixels_opened"]
    assert result["n_train"] == 2 and result["n_val"] == 1


def test_manifest_input_alias_detached_before_build_validation(api, inventory, tmp_path, monkeypatch):
    manifest = make(api, inventory)
    original = api._validate_manifest
    expected = set(manifest["eligible_ids"])

    def mutate_input_after_validation(owned, root):
        original(owned, root)
        manifest["eligible_ids"].append(inventory["target"].query_ids[0])

    monkeypatch.setattr(api, "_validate_manifest", mutate_input_after_validation)
    monkeypatch.setattr(api.enrollment, "_features", lambda *args: None)
    monkeypatch.setattr(api.enrollment, "_fit", lambda *args: (np.eye(3), 100))
    result = api.build_protocol_c_enrollment_products(manifest, inventory["raw"], tmp_path / "detached")
    assert set(result["eligible_ids"]) == expected


def test_changed_reservation_rejected_before_build_pixels(api, inventory, tmp_path, monkeypatch):
    manifest = make(api, inventory)
    with Path(inventory["target"].path).open("ab") as stream:
        stream.write(b"changed")
    monkeypatch.setattr(cv2, "imread", lambda *args: pytest.fail("changed reservation decoded pixels"))
    with pytest.raises(ValueError, match="hash"):
        api.build_protocol_c_enrollment_products(manifest, inventory["raw"], tmp_path / "rejected")
    assert not (tmp_path / "rejected").exists()


def test_support_symlink_rejected_at_metadata_boundary(api, inventory):
    path = inventory["raw"] / next(row["path"] for row in inventory["metadata"]["records"] if row["image_id"] in inventory["permitted"])
    original = path.with_suffix(".original")
    path.rename(original)
    path.symlink_to(original)
    with pytest.raises(ValueError, match="symlink"):
        make(api, inventory)


def test_script_products_phase_needs_no_models_or_checkpoints(api, inventory, tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("prepare_c_build_test", ROOT / "scripts/prepare_protocol_c.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(api.enrollment, "_features", lambda *args: None)
    monkeypatch.setattr(api.enrollment, "_fit", lambda *args: (np.eye(3), 100))
    paths = inventory["metadata"]["paths"]
    result = module.prepare(paths["records"], sha(paths["records"]), paths["source"], paths["target"],
                            inventory["raw"], tmp_path / "cpu-preparation", build=True)
    assert result["phase"] == "products" and result["passed"]
    assert not result["checkpoint_evaluation_authorized"]
    assert not result["target_query_pixels_opened"] and not result["source_pixels_opened"]
    assert sha(Path(result["enrollment"]["path"])) == result["enrollment"]["sha256"]
    completed = json.loads(Path(result["enrollment"]["path"]).read_text())
    assert completed["counts"]["products"] == 27
