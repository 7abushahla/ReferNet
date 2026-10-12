import copy
import csv
import hashlib
import importlib
import json
from pathlib import Path
import random
import sys

import cv2
import numpy as np
import pandas as pd
import pytest

from test_anchor_enrollment_integrity import bundle as integrity_bundle


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def authority(path):
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True, allow_nan=False))
    return authority(path)


@pytest.fixture
def api():
    assert (ROOT / "src/refer_or_clear/data/direct_reference.py").is_file(), "direct reference data bridge missing"
    return importlib.import_module("refer_or_clear.data.direct_reference")


@pytest.fixture
def inputs(api, integrity_bundle, tmp_path, monkeypatch):
    manifest_path, manifest = integrity_bundle
    manifest["forbidden_components"] = []
    for bank in manifest["banks"]["sidtd/example"].values():
        for identifier, contributor in bank["contributors"].items():
            contributor["component_id"] = "component0" if identifier == "support_0" else "component1"
        bank["component_ids"] = sorted({item["component_id"] for item in bank["contributors"].values()})
        products = bank["products"]
        for name in ("target", "median", "target_coverage", "target_valid", "coverage", "valid", "zone"):
            image = np.full((512, 800, 3), (11, 66, 199), np.uint8) if name in ("target", "median") else np.full((512, 800), 255, np.uint8)
            assert cv2.imwrite(products[name]["path"], image)
            products[name]["sha256"] = authority(Path(products[name]["path"]))["sha256"]
        geometry_path = Path(products["geometry"]["path"])
        geometry = json.loads(geometry_path.read_text())
        geometry["contributors"] = bank["contributors"]
        products["geometry"] = write_json(geometry_path, geometry)
        products["manifest"] = write_json(Path(products["manifest"]["path"]), {
            **geometry, "output_sha256": {name: binding["sha256"] for name, binding in products.items() if name != "manifest"}})
    enrollment_authority = write_json(manifest_path, manifest)
    monkeypatch.setattr(api, "ENROLLMENT_SHA256", enrollment_authority["sha256"])
    raw = tmp_path / "raw"
    raw.mkdir()
    records, split = [], []
    identifiers = ["query0", "query1", "support_0", "support_1"] + [f"train{index:04d}" for index in range(1774)] + [f"val{index:04d}" for index in range(222)] + [f"test{index:04d}" for index in range(222)]
    for index, identifier in enumerate(identifiers):
        role = "train" if index < 1778 else "val" if index < 2000 else "test"
        component = "component0" if identifier == "support_0" else "component1" if identifier == "support_1" else identifier
        label = 0 if identifier.startswith("support") else index % 2
        records.append(dict(image_id=identifier, dataset="sidtd", design_id="sidtd/example", component_id=component,
                            label=label, path=f"{identifier}.png", attack_family="text_copy" if label else ""))
        split.append(dict(image_id=identifier, design_id="sidtd/example", component_id=component, label=label,
                          role=role, protocol="A", ref_group="it0"))
    bindings = {}
    for name, content in (("records", records), ("split", split)):
        path = tmp_path / f"{name}.csv"
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(content[0]))
            writer.writeheader()
            writer.writerows(content)
        bindings[name] = authority(path)
    phase = dict(schema_version="direct_reference_phase_v1", phase="coordinate", source_sha256="a" * 64,
                 recipe_sha256="b" * 64, records_sha256=bindings["records"]["sha256"],
                 split_sha256=bindings["split"]["sha256"], enrollment_manifest_sha256=enrollment_authority["sha256"],
                 authorized_image_ids={"train": ["query0", "query1"], "val": []})
    preprocess = dict(enrollment_authority=enrollment_authority, phase_authority=write_json(tmp_path / "phase.json", phase),
                      product_root_authority=write_json(tmp_path / "root.json", dict(
                          schema_version="direct_reference_product_root_v1", original_root=manifest["output_root"],
                          relocated_root=manifest["output_root"], enrollment_manifest_sha256=enrollment_authority["sha256"])),
                      records_authority=bindings["records"], split_authority=bindings["split"], raw_root=str(raw),
                      source_sha256="a" * 64, recipe_sha256="b" * 64,
                      weights_sha256="da3f569306dadd1fac5b64abc2a3d484571cd0ecf6410e1f152275cfe49a3618")
    for identifier in ("query0", "query1", "val0000"):
        assert cv2.imwrite(str(raw / f"{identifier}.png"), np.full((512, 800, 3), (11, 66, 199), np.uint8))
    frame = pd.DataFrame(records).merge(pd.DataFrame(split)[["image_id", "role"]], on="image_id")
    monkeypatch.setattr(api, "_fit", lambda source, target, seed: (np.eye(3), 100))
    return dict(rows=frame[frame.image_id.isin(["query0", "query1"])].reset_index(drop=True), all_rows=frame,
                raw=raw, manifest=manifest, preprocess=preprocess, phase=phase, tmp=tmp_path)


def build(api, inputs, rows=None):
    return api.build_registration_table(inputs["rows"] if rows is None else rows, inputs["raw"], inputs["manifest"], preprocess=inputs["preprocess"])


def dataset(api, inputs, table, enabled, rows=None, train=True, augment=None):
    return api.DirectReferenceDataset(inputs["rows"] if rows is None else rows, inputs["raw"], inputs["manifest"], table,
                                      reference_enabled=enabled, train=train, augment=augment or {"enabled": False})


def permit(inputs, phase):
    inputs["preprocess"]["phase_authority"] = write_json(inputs["tmp"] / "phase.json", phase)


def gates(inputs, smoke=False):
    identities = {key: inputs["phase"][key] for key in ("source_sha256", "recipe_sha256", "records_sha256", "split_sha256", "enrollment_manifest_sha256")}
    result = {}
    for arm, enabled in (("off", False), ("on", True)):
        gate = dict(schema_version="direct_reference_smoke_gates_v1" if smoke else "direct_reference_training_gates_v1",
                    **identities, weights_sha256=inputs["preprocess"]["weights_sha256"], reference_enabled=enabled,
                    passed=True, no_test_open=True)
        if smoke:
            gate.update(n_images={"train": 32, "val": 32}, epochs_completed=1, replay={"passed": True, "fresh_process": True})
        else:
            gate.update(opened_roles=["train"], coordinate_gate={"passed": True}, model_gate={"passed": True},
                        overfit={"n_images": 16, "n_correct": 15, "image_loss": 0.14, "updates": 500})
        result[arm] = write_json(inputs["tmp"] / f'{arm}-{"smoke" if smoke else "training"}.json', gate)
    return result


def smoke_phase(inputs):
    phase = copy.deepcopy(inputs["phase"])
    phase.update(phase="smoke", authorized_image_ids={
        role: sorted(inputs["all_rows"].loc[inputs["all_rows"].role == role, "image_id"])[:32] for role in ("train", "val")}, training_gates=gates(inputs))
    return phase


def test_paired_tensors_and_off_never_reads_median(api, inputs, monkeypatch):
    table = build(api, inputs)
    on = dataset(api, inputs, table, True)[0]
    original = api._image
    def guarded(binding, flags):
        assert Path(binding["path"]).name != "median.png"
        return original(binding, flags)
    monkeypatch.setattr(api, "_image", guarded)
    off = dataset(api, inputs, table, False)[0]
    assert "reference" not in off
    assert on["reference"].shape == (3, 512, 800)
    assert on["query"].shape == (4, 512, 800)
    for key in ("query", "overlap", "context_available"):
        assert np.array_equal(on[key].numpy(), off[key].numpy())
    assert on["context_available"].item() is True
    assert on["meta"]["bank"] == "train_half1"


@pytest.mark.parametrize("phase", ["coordinate", "overfit"])
def test_phase_guard_denies_staged_validation_before_any_pixels(api, inputs, phase, monkeypatch):
    inputs["phase"]["phase"] = phase
    permit(inputs, inputs["phase"])
    monkeypatch.setattr(api, "_image", lambda *args: pytest.fail("forbidden pixel read"))
    val = inputs["all_rows"].query("image_id == 'val0000'")
    with pytest.raises(ValueError): build(api, inputs, val)


@pytest.mark.parametrize("mutation", ["missing", "hash", "val-authorized", "duplicate", "overlap", "source", "unsorted"])
def test_phase_authority_is_not_a_boolean_waiver(api, inputs, mutation, monkeypatch):
    phase = copy.deepcopy(inputs["phase"])
    if mutation == "missing": del inputs["preprocess"]["phase_authority"]
    elif mutation == "hash": inputs["preprocess"]["phase_authority"]["sha256"] = "0" * 64
    else:
        if mutation == "val-authorized": phase["authorized_image_ids"]["val"] = ["val0000"]
        if mutation == "duplicate": phase["authorized_image_ids"]["train"] = ["query0", "query0"]
        if mutation == "overlap": phase["authorized_image_ids"]["val"] = ["query0"]
        if mutation == "source": phase["source_sha256"] = "c" * 64
        if mutation == "unsorted": phase["authorized_image_ids"]["train"].reverse()
        permit(inputs, phase)
    monkeypatch.setattr(api, "_image", lambda *args: pytest.fail("pixel read before phase validation"))
    with pytest.raises(ValueError): build(api, inputs)


@pytest.mark.parametrize("mutation", ["missing-on", "failed", "wrong-arm", "wrong-weights", "val-opened", "correct14", "loss", "nan", "updates0", "updates501", "bool-count"])
def test_smoke_requires_both_authenticated_training_gates(api, inputs, mutation, monkeypatch):
    phase = smoke_phase(inputs)
    if mutation == "missing-on": del phase["training_gates"]["on"]
    else:
        binding = phase["training_gates"]["on"]
        gate = json.loads(Path(binding["path"]).read_text())
        changes = {"failed": ("passed", False), "wrong-arm": ("reference_enabled", False),
                   "wrong-weights": ("weights_sha256", "0" * 64), "val-opened": ("opened_roles", ["train", "val"])}
        if mutation in changes:
            key, value = changes[mutation]; gate[key] = value
        else:
            key, value = {"correct14": ("n_correct", 14), "loss": ("image_loss", 0.15), "nan": ("image_loss", float("nan")),
                          "updates0": ("updates", 0), "updates501": ("updates", 501), "bool-count": ("n_images", True)}[mutation]
            gate["overfit"][key] = value
        path = Path(binding["path"])
        path.write_text(json.dumps(gate))
        phase["training_gates"]["on"] = authority(path)
    permit(inputs, phase)
    monkeypatch.setattr(api, "_image", lambda *args: pytest.fail("pixel read before gates"))
    with pytest.raises(ValueError): build(api, inputs)


def test_gated_validation_never_routes_by_component_or_annotations(api, inputs):
    permit(inputs, smoke_phase(inputs))
    rows = inputs["all_rows"].query("image_id == 'val0000'").drop(columns="component_id")
    rows = rows.assign(source_id="forbidden", donor_id="forbidden", annotation="must-not-open")
    table = build(api, inputs, rows)
    item = dataset(api, inputs, table, False, rows, train=False)[0]
    assert item["meta"]["bank"] == "enrollment"
    assert item["label"] == int(rows.iloc[0].label)


@pytest.mark.parametrize("mutation", ["missing-smokes", "failed-replay", "not-fresh", "two-epochs", "wrong-count"])
def test_train_requires_authenticated_smoke_receipts(api, inputs, mutation, monkeypatch):
    phase = smoke_phase(inputs)
    phase["phase"] = "train"
    phase["smoke_gates"] = gates(inputs, smoke=True)
    if mutation == "missing-smokes": del phase["smoke_gates"]
    else:
        binding = phase["smoke_gates"]["off"]
        gate = json.loads(Path(binding["path"]).read_text())
        if mutation == "failed-replay": gate["replay"]["passed"] = False
        if mutation == "not-fresh": gate["replay"]["fresh_process"] = False
        if mutation == "two-epochs": gate["epochs_completed"] = 2
        if mutation == "wrong-count": gate["n_images"]["val"] = 31
        phase["smoke_gates"]["off"] = write_json(Path(binding["path"]), gate)
    permit(inputs, phase)
    monkeypatch.setattr(api, "_image", lambda *args: pytest.fail("premature training pixels"))
    with pytest.raises(ValueError): build(api, inputs)


@pytest.mark.parametrize("matrix", [np.array([[1,0,16],[0,1,8],[0,0,1]]),
                                      np.array([[0,-1,800],[1,0,0],[0,0,1]]),
                                      np.array([[0.8,0,30],[0,0.8,20],[0,0,1]])])
def test_inverse_context_preserves_query_pixels(api, inputs, monkeypatch, matrix):
    monkeypatch.setattr(api, "_fit", lambda *args: (matrix, 100))
    table = build(api, inputs)
    item = dataset(api, inputs, table, True)[0]
    bank = inputs["manifest"]["banks"]["sidtd/example"]["train_half1"]
    median = cv2.imread(bank["products"]["median"]["path"])
    expected = cv2.warpAffine(median, np.linalg.inv(matrix)[:2], (800,512), flags=cv2.INTER_LINEAR)
    from refer_or_clear.data.dataset import _normalize
    assert np.array_equal(item["reference"].numpy(), _normalize(expected))
    original = cv2.imread(str(inputs["raw"] / "query0.png"))
    assert np.array_equal(item["query"][:3].numpy(), _normalize(original))


def test_letterbox_padding_and_overlap(api, inputs):
    assert cv2.imwrite(str(inputs["raw"] / "query0.png"), np.full((200,800,3), (11,66,199), np.uint8))
    item = dataset(api, inputs, build(api, inputs), False)[0]
    assert item["overlap"].sum().item() == 200 * 800
    assert item["query"][3, :156].sum().item() == 0


@pytest.mark.parametrize("matrix,inliers,reason", [(None,12,"registration_failed"), (np.eye(3),39,"registration_failed"),
                                                  (np.array([[1,0,2000],[0,1,0],[0,0,1]]),100,"empty_overlap")])
def test_unavailable_context_retains_label_and_referral(api, inputs, monkeypatch, matrix, inliers, reason):
    monkeypatch.setattr(api, "_fit", lambda *args: (matrix,inliers))
    table = build(api, inputs)
    monkeypatch.setattr(api, "_fit", lambda *args: pytest.fail("lazy refitting"))
    for enabled in (False,True):
        item = dataset(api, inputs, table, enabled)[0]
        assert not item["context_available"].item()
        assert item["meta"]["requires_referral"] is True
        assert item["meta"]["availability_reason"] == reason
        assert item["label"] == int(inputs["rows"].iloc[0].label)
        assert item["query"][3].sum().item() == item["overlap"].sum().item() == 0


def test_table_file_authentication_and_mutable_dict_rejection(api, inputs):
    table = build(api, inputs)
    binding = write_json(inputs["tmp"] / "table.json", table)
    loaded = api.load_registration_table(Path(binding["path"]), binding["sha256"], rows=inputs["rows"], enrollment_manifest=inputs["manifest"], preprocess=inputs["preprocess"])
    assert dataset(api, inputs, loaded, False)[0]["context_available"].item()
    with pytest.raises(ValueError): dataset(api, inputs, dict(loaded), False)
    Path(binding["path"]).write_text("{}")
    with pytest.raises(ValueError): api.load_registration_table(Path(binding["path"]), binding["sha256"], rows=inputs["rows"], enrollment_manifest=inputs["manifest"], preprocess=inputs["preprocess"])


@pytest.mark.parametrize("mutation", ["query", "product", "table", "manifest", "row-label", "test-role", "symlink"])
def test_stale_or_forged_inputs_fail_closed(api, inputs, mutation):
    table = build(api, inputs)
    if mutation == "query": (inputs["raw"] / "query0.png").write_bytes(b"changed")
    if mutation == "product": Path(inputs["manifest"]["banks"]["sidtd/example"]["train_half1"]["products"]["zone"]["path"]).write_bytes(b"changed")
    if mutation == "table": table["entries"]["query0"]["bank"] = "enrollment"
    if mutation == "manifest": inputs["manifest"]["fold"] = 1
    if mutation == "row-label": inputs["rows"].loc[0,"label"] = 1-int(inputs["rows"].iloc[0].label)
    if mutation == "test-role": inputs["rows"].loc[0,"role"] = "test"
    if mutation == "symlink":
        path=inputs["raw"] / "query0.png"; path.rename(inputs["raw"] / "other.png"); path.symlink_to(inputs["raw"] / "other.png")
    with pytest.raises((ValueError,OSError)):
        dataset(api, inputs, table, False)[0]


def test_augmentation_is_matched_coherent_and_rng_isolated(api, inputs):
    table = build(api, inputs)
    state = random.getstate()
    augment = dict(enabled=True, affine_px=8, jitter_px=2, jitter_p=1.0, photometric=True)
    on, off = dataset(api, inputs, table, True, augment=augment), dataset(api, inputs, table, False, augment=augment)
    for index in (0,1,0):
        first, second = on[index], off[index]
        for key in ("query","overlap","context_available"):
            assert np.array_equal(first[key].numpy(),second[key].numpy())
        assert np.array_equal(first["query"][3].numpy(),first["overlap"][0].numpy())
    assert random.getstate()==state


@pytest.mark.parametrize("enabled", [False, True])
def test_actual_config_augmentation_and_default_collation(api, inputs, enabled):
    import yaml
    from torch.utils.data import DataLoader
    augment = yaml.safe_load((ROOT / "configs/direct_reference_pilot.yaml").read_text())["augment"]
    consumer = dataset(api, inputs, build(api, inputs), enabled, augment=augment)
    batch = next(iter(DataLoader(consumer, batch_size=2)))
    assert batch["query"].shape == (2, 4, 512, 800)
    assert batch["overlap"].shape == (2, 1, 512, 800)
    assert batch["context_available"].shape == (2,)
    assert batch["meta"]["image_id"] == ["query0", "query1"]
    assert len(batch["meta"]["products"]["zone"]["sha256"]) == 2
    assert ("reference" in batch) is enabled
    if enabled: assert batch["reference"].shape == (2, 3, 512, 800)


def test_authorized_superset_table_reuse_opens_selected_query_only(api, inputs, monkeypatch):
    table = build(api, inputs)
    binding = write_json(inputs["tmp"] / "union-table.json", table)
    monkeypatch.setattr(api, "_fit", lambda *args: pytest.fail("subset must not refit"))
    original = api._verifier._read_file
    for identifier in ("query0", "query1"):
        selected = inputs["rows"].query("image_id == @identifier")
        def guarded(path, *args, **kwargs):
            if Path(path).parent == inputs["raw"]:
                assert Path(path).name == f"{identifier}.png"
            return original(path, *args, **kwargs)
        monkeypatch.setattr(api._verifier, "_read_file", guarded)
        loaded = api.load_registration_table(Path(binding["path"]), binding["sha256"], rows=selected,
                    enrollment_manifest=inputs["manifest"], preprocess=inputs["preprocess"])
        assert dataset(api, inputs, loaded, False, selected)[0]["meta"]["image_id"] == identifier


def test_forward_uses_cached_metadata_and_no_registration_target(api, inputs, monkeypatch):
    consumer = dataset(api, inputs, build(api, inputs), False)
    monkeypatch.setattr(api, "load_role_rows", lambda *args: pytest.fail("metadata parsed per image"))
    original = api._verifier._read_file
    def guarded(path, *args, **kwargs):
        assert Path(path).name not in ("target.png", "median.png")
        return original(path, *args, **kwargs)
    monkeypatch.setattr(api._verifier, "_read_file", guarded)
    assert consumer[0]["context_available"].item()


def test_cached_metadata_authority_invalidation_precedes_pixels(api, inputs, monkeypatch):
    consumer = dataset(api, inputs, build(api, inputs), False)
    Path(inputs["preprocess"]["phase_authority"]["path"]).write_text("{}")
    monkeypatch.setattr(api, "_image", lambda *args: pytest.fail("pixels before authority invalidation"))
    with pytest.raises(ValueError): consumer[0]


@pytest.mark.parametrize("field,value", [("preset", "legacy_query_v1"), ("reference_photometric", True),
                                         ("reference_photometric", 0), ("jitter_zone", False), ("jitter_zone", 1),
                                         ("rotation_degrees", 2), ("jitter_p", 1.1), ("unknown", True)])
def test_bridge_augmentation_rejects_semantic_weakening(api, inputs, field, value):
    import yaml
    augment = yaml.safe_load((ROOT / "configs/direct_reference_pilot.yaml").read_text())["augment"]
    augment[field] = value
    with pytest.raises(ValueError): dataset(api, inputs, build(api, inputs), False, augment=augment)


def test_bridge_geometry_moves_every_plane_and_photometry_query_only(api, inputs):
    import yaml
    augment = yaml.safe_load((ROOT / "configs/direct_reference_pilot.yaml").read_text())["augment"]
    augment.update(jitter_p=1.0, photometric=False)
    table = build(api, inputs)
    item = dataset(api, inputs, table, True, augment=augment)[0]
    assert np.array_equal(item["query"][:3].numpy(), item["reference"].numpy())
    assert np.array_equal(item["query"][3].numpy(), item["overlap"][0].numpy())
    augment["photometric"] = True
    photo = dataset(api, inputs, table, True, augment=augment)[0]
    assert np.array_equal(photo["reference"].numpy(), item["reference"].numpy())
    assert np.array_equal(photo["overlap"].numpy(), item["overlap"].numpy())
    assert not np.array_equal(photo["query"][:3].numpy(), item["query"][:3].numpy())


def test_failed_fit_does_not_inherit_previous_query_inliers(api, inputs, monkeypatch):
    calls = []
    def fit(*args):
        calls.append(True)
        if len(calls) == 1: return np.eye(3), 100
        raise ValueError("synthetic fit failure")
    monkeypatch.setattr(api, "_fit", fit)
    table = build(api, inputs)
    assert table["entries"]["query1"]["inliers"] == 0


def test_dataset_rejects_table_mutation_after_construction(api, inputs, monkeypatch):
    consumer = dataset(api, inputs, build(api, inputs), False)
    consumer.table["entries"]["query0"]["query_canvas_to_bank"][0][2] = 5
    monkeypatch.setattr(api, "_image", lambda *args: pytest.fail("mutated fit consumed pixels"))
    with pytest.raises(ValueError): consumer[0]


def test_authenticated_relocation_preserves_original_manifest_bytes(api, inputs):
    import shutil
    original = Path(inputs["manifest"]["output_root"])
    relocated = inputs["tmp"] / "relocated-enrollment"
    before = Path(inputs["preprocess"]["enrollment_authority"]["path"]).read_bytes()
    shutil.copytree(original, relocated)
    staged_manifest = inputs["tmp"] / "enrollment.json"
    staged_manifest.write_bytes(before)
    inputs["preprocess"]["enrollment_authority"] = authority(staged_manifest)
    root = dict(schema_version="direct_reference_product_root_v1", original_root=str(original),
                relocated_root=str(relocated), enrollment_manifest_sha256=inputs["preprocess"]["enrollment_authority"]["sha256"])
    inputs["preprocess"]["product_root_authority"] = write_json(inputs["tmp"] / "root.json", root)
    shutil.rmtree(original)
    table = build(api, inputs)
    for enabled in (False, True):
        item = dataset(api, inputs, table, enabled)[0]
        assert item["context_available"].item()
        assert all(Path(binding["path"]).is_relative_to(relocated) for binding in item["meta"]["products"].values())
    assert Path(inputs["preprocess"]["enrollment_authority"]["path"]).read_bytes() == before


def test_smoke_union_table_reused_by_role_subsets_without_other_query_reads(api, inputs, monkeypatch):
    phase = smoke_phase(inputs)
    permit(inputs, phase)
    identifiers = phase["authorized_image_ids"]["train"] + phase["authorized_image_ids"]["val"]
    rows = inputs["all_rows"].loc[inputs["all_rows"].image_id.isin(identifiers)].copy()
    for identifier in identifiers:
        assert cv2.imwrite(str(inputs["raw"] / f"{identifier}.png"), np.full((512,800,3), (11,66,199), np.uint8))
    monkeypatch.setattr(api, "_features", lambda *args: None)
    table = build(api, inputs, rows)
    assert len(table["entries"]) == 64
    monkeypatch.setattr(api, "_fit", lambda *args: pytest.fail("smoke/replay must reuse frozen fit"))
    original = api._verifier._read_file
    for role in ("train", "val"):
        selected = rows.loc[rows.role == role]
        consumers = [dataset(api, inputs, table, enabled, selected, train=role == "train") for enabled in (False,True)]
        selected_id = selected.iloc[0].image_id
        def guarded(path, *args, **kwargs):
            if Path(path).parent == inputs["raw"]: assert Path(path).name == f"{selected_id}.png"
            return original(path, *args, **kwargs)
        monkeypatch.setattr(api._verifier, "_read_file", guarded)
        for consumer in consumers:
            assert consumer[0]["meta"]["role"] == role


def test_off_never_reads_median_bytes_and_tiny_overlap_refers(api, inputs, monkeypatch):
    monkeypatch.setattr(api, "_fit", lambda *args: (np.array([[1,0,-799],[0,1,-511],[0,0,1]]),100))
    table = build(api, inputs)
    consumer = dataset(api, inputs, table, False)
    original = api._verifier._read_file
    def guarded(path, *args, **kwargs):
        assert Path(path).name != "median.png"
        return original(path, *args, **kwargs)
    monkeypatch.setattr(api._verifier, "_read_file", guarded)
    item = consumer[0]
    assert not item["context_available"].item()
    assert item["meta"]["requires_referral"]
    assert item["overlap"].sum().item() == 0


def test_registration_features_use_bgr_grayscale(api, monkeypatch):
    observed = []
    class Sift:
        def detectAndCompute(self, gray, mask):
            observed.append(gray.copy())
            return [], None
    monkeypatch.setattr(cv2, "SIFT_create", lambda count: Sift())
    bgr = np.full((3,5,3), (11,66,199), np.uint8)
    api._features(bgr)
    assert np.array_equal(observed[0], cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
    assert not np.array_equal(observed[0], cv2.cvtColor(bgr, cv2.COLOR_RGB2GRAY))


@pytest.mark.parametrize("mutation", ["foreign-entry", "nan-matrix", "shear-matrix", "validation-component"])
def test_externally_hashed_forged_table_still_rejected(api, inputs, mutation, monkeypatch):
    table = dict(build(api, inputs))
    entry = table["entries"]["query0"]
    if mutation == "foreign-entry": table["entries"]["test0000"] = copy.deepcopy(entry)
    if mutation == "nan-matrix": entry["query_canvas_to_bank"][0][0] = float("nan")
    if mutation == "shear-matrix": entry["query_canvas_to_bank"][0][1] = .5
    if mutation == "validation-component":
        entry["role"] = "val"
        entry["training_component"] = "forbidden"
    path = inputs["tmp"] / "forged-table.json"
    path.write_text(json.dumps(table))
    binding = authority(path)
    monkeypatch.setattr(api, "_image", lambda *args: pytest.fail("forged table pixel access"))
    with pytest.raises(ValueError):
        api.load_registration_table(path, binding["sha256"], rows=inputs["rows"], enrollment_manifest=inputs["manifest"], preprocess=inputs["preprocess"])


def test_component_and_transitive_support_exclusion(api, inputs, monkeypatch):
    context = api._authorities(inputs["manifest"], inputs["preprocess"])
    row = inputs["rows"].iloc[0].to_dict()
    own_component = copy.deepcopy(inputs["manifest"]["banks"]["sidtd/example"]["train_half1"])
    own_component["component_ids"].append(row["component_id"])
    monkeypatch.setattr(api, "resolve_enrollment_bank", lambda *args: own_component)
    with pytest.raises(ValueError): api._bank(inputs["manifest"], row, context)
    forbidden = copy.deepcopy(inputs["manifest"])
    forbidden["forbidden_components"] = ["component1"]
    monkeypatch.setattr(api, "resolve_enrollment_bank", lambda *args: forbidden["banks"]["sidtd/example"]["train_half1"])
    with pytest.raises(ValueError): api._bank(forbidden, row, context)


def test_metadata_join_consumes_authenticated_bytes_not_reopened_paths(api, inputs, monkeypatch):
    original = api.load_role_rows
    path = Path(inputs["preprocess"]["records_authority"]["path"])
    before = path.read_bytes()
    def racing(records, split, role):
        path.write_bytes(before.replace(b"query0.png", b"foreign.png"))
        try:
            return original(records, split, role)
        finally:
            path.write_bytes(before)
    monkeypatch.setattr(api, "load_role_rows", racing)
    assert build(api, inputs)["entries"]["query0"]["path"] == "query0.png"
