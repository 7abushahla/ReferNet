import copy
import hashlib
import importlib
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from refer_or_clear.data.records import provenance_components
from refer_or_clear.data.references import fold_of


def api():
    assert importlib.util.find_spec("refer_or_clear.data.enrollment"), "fold-safe enrollment implementation is missing"
    return importlib.import_module("refer_or_clear.data.enrollment")


@pytest.fixture
def inventory(tmp_path):
    raw = tmp_path / "raw"
    base = raw / "sidtd/templates"
    (base / "Images/reals").mkdir(parents=True)
    (base / "Annotations/reals").mkdir(parents=True)
    texture = np.random.default_rng(17).integers(0, 256, (128, 200, 3), dtype=np.uint8)
    rows = []
    metadata = {}
    for number in range(8):
        name = f"alb_id_{number:02d}.png"
        identity = f"sidtd/alb_id_{number:02d}"
        assert cv2.imwrite(str(base / "Images/reals" / name), texture)
        metadata[str(number)] = {
            "filename": f"{number:02d}.jpg", "size": number,
            "regions": [{"shape_attributes": {"name": "rect", "x": 20, "y": 30, "width": 25, "height": 12},
                         "region_attributes": {"field_name": "name"}}],
        }
        rows.append(dict(dataset="sidtd", image_id=f"g{number}", identity_id=identity,
                         source_id="", donor_id="", design_id="sidtd/alb_id", label=0,
                         path=f"sidtd/templates/Images/reals/{name}",
                         annotation="sidtd/templates/Annotations/reals/alb_id.json", width=200, height=128))
    rows.extend([
        dict(rows[0], image_id="copy", label=1, source_id=rows[0]["identity_id"],
             donor_id=rows[6]["identity_id"], path="sidtd/templates/Images/fakes/copy.png",
             annotation="sidtd/templates/Annotations/fakes/copy.json"),
        dict(rows[6], image_id="bridge", label=1, source_id=rows[6]["identity_id"],
             donor_id=rows[7]["identity_id"], path="sidtd/templates/Images/fakes/bridge.png",
             annotation="sidtd/templates/Annotations/fakes/bridge.json"),
    ])
    annotation = base / "Annotations/reals/alb_id.json"
    annotation.write_text(json.dumps({"_via_img_metadata": metadata}))
    records = pd.DataFrame(rows)
    records["component_id"] = provenance_components(records)
    records.attrs["root"] = str(raw)
    split = records[["image_id", "component_id", "design_id", "label"]].copy()
    split["role"] = "train"
    split.loc[split.image_id == "copy", "role"] = "val"
    split.loc[split.image_id == "g5", "role"] = "test"
    split["protocol"] = "A"
    split["ref_group"] = "it0"
    return raw, records, split, annotation, texture


def mutate(inventory, image_id):
    raw, records, _, annotation, texture = inventory
    row = records.set_index("image_id").loc[image_id]
    assert cv2.imwrite(str(raw / row.path), np.roll(texture, 7, axis=1))
    via = json.loads(annotation.read_text())
    entry = via["_via_img_metadata"][str(int(image_id[1:]))]
    entry["regions"][0]["shape_attributes"]["x"] += 13
    annotation.write_text(json.dumps(via))


def bank_hashes(products, bank):
    return {key: item["sha256"] for key, item in products["banks"]["sidtd/alb_id"][bank]["products"].items()}


def test_source_donor_transitive_exclusion_preserves_classifier(inventory):
    raw, records, split, _, _ = inventory
    before_records, before_split = records.copy(deep=True), split.copy(deep=True)
    manifest = api().make_enrollment_manifest(records, split, 0, raw)
    assert manifest["eligible_ids"] == ["g1", "g2", "g3", "g4"]
    assert manifest["classifier_counts"] == {"train": 8, "val": 1, "test": 1}
    pd.testing.assert_frame_equal(records, before_records)
    pd.testing.assert_frame_equal(split, before_split)
    for component in records.loc[records.image_id.isin(["g0", "g6", "g7"]), "component_id"]:
        assert component in manifest["forbidden_components"]


def test_opposite_half_and_annotation_free_evaluation(inventory):
    raw, records, split, _, _ = inventory
    manifest = api().make_enrollment_manifest(records, split, 0, raw)
    for row in records.itertuples():
        selected = api().resolve_enrollment_bank(manifest, row.design_id, "train", row.component_id)
        assert selected["bank"] == f"train_half{1 - fold_of(row.component_id)}"
        assert row.component_id not in selected["component_ids"]
    stripped = {key: manifest[key] for key in ("schema_version", "geometry_version", "fold", "banks")}
    assert api().resolve_enrollment_bank(stripped, "sidtd/alb_id", "val")["support_ids"] == ["g1", "g2", "g3", "g4"]
    assert api().resolve_enrollment_bank(stripped, "sidtd/alb_id", "test") == api().resolve_enrollment_bank(stripped, "sidtd/alb_id", "evaluation")
    with pytest.raises(ValueError, match="component"):
        api().resolve_enrollment_bank(manifest, "sidtd/alb_id", "val", "secret-query-provenance")


def test_heldout_pixels_and_shared_via_entries_cannot_affect_products(inventory, tmp_path):
    raw, records, split, _, _ = inventory
    manifest = api().make_enrollment_manifest(records, split, 0, raw)
    before = api().build_enrollment_products(manifest, raw, tmp_path / "first")
    for image_id in ("g0", "g5", "g6", "g7"):
        mutate(inventory, image_id)
    after = api().build_enrollment_products(manifest, raw, tmp_path / "second")
    for bank in ("train_half0", "train_half1", "enrollment"):
        assert bank_hashes(before, bank) == bank_hashes(after, bank)
    assert before["archive_metadata"] != after["archive_metadata"]
    assert manifest["eligible_ids"] == api().make_enrollment_manifest(records, split, 0, raw)["eligible_ids"]


def test_unused_half_cannot_affect_target_median_zone_valid_or_specimen(inventory, tmp_path):
    raw, records, split, _, texture = inventory
    specimen = tmp_path / "specimens"
    specimen.mkdir()
    assert cv2.imwrite(str(specimen / "blank.png"), texture)
    digest = hashlib.sha256((specimen / "blank.png").read_bytes()).hexdigest()
    (specimen / "provenance.json").write_text(json.dumps({"schema_version": "external_blank_v1", "specimens": {
        "sidtd/alb_id": {"path": "blank.png", "sha256": digest, "source": "independent synthetic blank", "independent": True}}}))
    manifest = api().make_enrollment_manifest(records, split, 0, raw)
    before = api().build_enrollment_products(manifest, raw, tmp_path / "before", specimen)
    included = manifest["banks"]["sidtd/alb_id"]["train_half0"]["support_ids"]
    unused = manifest["banks"]["sidtd/alb_id"]["train_half1"]["support_ids"]
    for image_id in unused:
        mutate(inventory, image_id)
    after = api().build_enrollment_products(manifest, raw, tmp_path / "after", specimen)
    assert bank_hashes(before, "train_half0") == bank_hashes(after, "train_half0")
    assert {"target", "median", "zone", "coverage", "valid", "geometry", "specimen", "specimen_valid"} <= set(bank_hashes(before, "train_half0"))
    mutate(inventory, included[0])
    changed = api().build_enrollment_products(manifest, raw, tmp_path / "changed", specimen)
    previous = after["banks"]["sidtd/alb_id"]["train_half0"]["contributors"][included[0]]
    current = changed["banks"]["sidtd/alb_id"]["train_half0"]["contributors"][included[0]]
    assert previous["raw_sha256"] != current["raw_sha256"]
    assert previous["layout_sha256"] != current["layout_sha256"]


def test_empty_unknown_and_stale_graph_fail_closed(inventory, tmp_path):
    raw, records, split, _, _ = inventory
    broken = split.copy()
    broken["role"] = "test"
    with pytest.raises(ValueError, match="empty|support"):
        api().make_enrollment_manifest(records, broken, 0, raw)
    manifest = api().make_enrollment_manifest(records, split, 0, raw)
    with pytest.raises(ValueError, match="design"):
        api().resolve_enrollment_bank(manifest, "missing/design", "val")
    altered = records.copy()
    altered.loc[altered.image_id == "bridge", "donor_id"] = records.iloc[1].identity_id
    with pytest.raises(ValueError, match="component|graph"):
        api().make_enrollment_manifest(altered, split, 0, raw)
    empty = copy.deepcopy(manifest)
    empty["banks"]["sidtd/alb_id"]["train_half0"]["support_ids"] = []
    with pytest.raises(ValueError, match="empty|support"):
        api().build_enrollment_products(empty, raw, tmp_path / "empty")


@pytest.mark.parametrize("failure", ["root", "dataset", "escape", "filename", "coordinates", "fold", "duplicate", "label"])
def test_exact_input_contracts(inventory, tmp_path, failure):
    raw, records, split, annotation, _ = inventory
    if failure == "root":
        records.attrs["root"] = str(tmp_path / "elsewhere")
    elif failure == "dataset":
        records.loc[records.image_id == "g1", "path"] = "fantasyid/same.png"
    elif failure == "escape":
        records.loc[records.image_id == "g1", "path"] = "../same.png"
    elif failure == "filename":
        via = json.loads(annotation.read_text())
        via["_via_img_metadata"]["1"]["filename"] = "wrong.jpg"
        annotation.write_text(json.dumps(via))
    elif failure == "coordinates":
        via = json.loads(annotation.read_text())
        via["_via_img_metadata"]["1"]["regions"][0]["shape_attributes"]["x"] = 300
        annotation.write_text(json.dumps(via))
    elif failure == "fold":
        split["ref_group"] = "it1"
    elif failure == "duplicate":
        split = pd.concat([split, split.iloc[:1]])
    elif failure == "label":
        split.loc[split.image_id == "g1", "label"] = 1
    with pytest.raises((ValueError, FileNotFoundError)):
        api().make_enrollment_manifest(records, split, 0, raw)


def test_output_is_new_and_lookup_returns_bank_product_paths(inventory, tmp_path):
    raw, records, split, _, _ = inventory
    manifest = api().make_enrollment_manifest(records, split, 0, raw)
    out = tmp_path / "products"
    products = api().build_enrollment_products(manifest, raw, out)
    selected = api().resolve_enrollment_bank(products, "sidtd/alb_id", "val")
    for item in selected["products"].values():
        assert Path(item["path"]).is_file()
        assert hashlib.sha256(Path(item["path"]).read_bytes()).hexdigest() == item["sha256"]
    assert "/fold0/sidtd/alb_id/enrollment/" in selected["products"]["median"]["path"]
    with pytest.raises(FileExistsError):
        api().build_enrollment_products(manifest, raw, out)
    with pytest.raises(ValueError, match="raw|root"):
        api().build_enrollment_products(manifest, tmp_path, tmp_path / "badroot")


def test_validity_median_and_polygon_coordinates_are_not_bounding_boxes():
    module = api()
    first = np.full((4, 4, 3), 100, dtype=np.uint8)
    second = np.full((4, 4, 3), 200, dtype=np.uint8)
    valid_first = np.full((4, 4), 255, dtype=np.uint8)
    valid_second = valid_first.copy()
    valid_second[:, :2] = 0
    valid_first[0, 0] = 0
    median, coverage = module._valid_median([first, second], [valid_first, valid_second])
    assert median[1, 0].tolist() == [100, 100, 100]
    assert median[1, 3].tolist() == [150, 150, 150]
    assert median[0, 0].tolist() == [0, 0, 0]
    assert coverage[0, 0] == 0 and coverage[1, 0] == 1 and coverage[1, 3] == 2
    transform = np.array([[1., -1., 100.], [1., 1., 100.], [0., 0., 1.]])
    zone = module._polygon_mask([["name", 0, 0, 20, 20]], transform)
    assert zone[120, 100] == 255
    assert zone[101, 81] == 0
    assert zone[100, 100] == 255
    with pytest.raises(ValueError, match="transform|similarity"):
        module._polygon_mask([["name", 0, 0, 20, 20]], np.zeros((3, 3)))


def test_support_only_reads_and_registration_failure_have_no_fallback(inventory, tmp_path):
    raw, records, split, _, _ = inventory
    for row in records[records.image_id.isin(["g0", "g5", "g6", "g7"])].itertuples():
        (raw / row.path).unlink()
    manifest = api().make_enrollment_manifest(records, split, 0, raw)
    products = api().build_enrollment_products(manifest, raw, tmp_path / "no-query-files")
    assert products["counts"]["eligible"] == 4
    for row in records[records.image_id.isin(manifest["eligible_ids"])].itertuples():
        assert cv2.imwrite(str(raw / row.path), np.zeros((128, 200, 3), np.uint8))
    with pytest.raises(ValueError, match="registered|unavailable"):
        api().build_enrollment_products(manifest, raw, tmp_path / "failed-registration")
    assert not (tmp_path / "failed-registration/manifest.json").exists()


def test_source_to_bank_axes_and_nearest_neighbor_validity(inventory):
    module = api()
    texture = inventory[-1]
    source, _, _ = module.to_canvas(texture)
    truth = np.array([[1., 0., 7.], [0., 1., -3.], [0., 0., 1.]])
    target = module._warp(source, truth)
    estimated, inliers = module._fit(module._features(source), module._features(target), 71)
    assert inliers >= 40
    assert estimated is not None
    np.testing.assert_allclose(estimated[:2], truth[:2], atol=0.1)
    marker = np.zeros((512, 800), np.uint8)
    marker[15, 10] = 255
    moved = module._warp(marker, truth, mask=True)
    assert moved[12, 17] == 255 and np.count_nonzero(moved) == 1
    fractional = truth.copy()
    fractional[0, 2] = 7.3
    assert set(np.unique(module._warp(marker, fractional, mask=True))) == {0, 255}


def test_build_records_its_current_source_not_manifest_creation_source(inventory, tmp_path):
    raw, records, split, _, _ = inventory
    manifest = api().make_enrollment_manifest(records, split, 0, raw)
    manifest["source"] = {"revision": "old-manifest-source"}
    products = api().build_enrollment_products(manifest, raw, tmp_path / "current-source")
    geometry = products["banks"]["sidtd/alb_id"]["enrollment"]["products"]["geometry"]
    assert json.loads(Path(geometry["path"]).read_text())["source"] == products["source"]


def test_unrelated_revision_change_does_not_change_frozen_bank_source(inventory, tmp_path, monkeypatch):
    raw, records, split, _, _ = inventory
    module = api()
    manifest = module.make_enrollment_manifest(records, split, 0, raw)
    current_source = dict(manifest["source"], revision="unrelated-integration-commit")
    monkeypatch.setattr(module, "_source", lambda: current_source)
    products = module.build_enrollment_products(manifest, raw, tmp_path / "unrelated-commit")
    geometry = products["banks"]["sidtd/alb_id"]["enrollment"]["products"]["geometry"]
    assert json.loads(Path(geometry["path"]).read_text())["source"] == manifest["source"]
    assert products["execution_source"] == current_source


def test_component_graph_validation_is_invariant_to_union_representative(inventory):
    raw, records, split, _, _ = inventory
    reverse = records[records.image_id == "copy"].copy()
    reverse["image_id"] = "reverse-copy"
    reverse["identity_id"] = "sidtd/alb_id_06"
    reverse["source_id"] = "sidtd/alb_id_06"
    reverse["donor_id"] = "sidtd/alb_id_00"
    shuffled = pd.concat([reverse, records], ignore_index=True)
    shuffled.attrs["root"] = str(raw)
    extra = split[split.image_id == "bridge"].copy()
    extra["image_id"] = "reverse-copy"
    extended = pd.concat([extra, split], ignore_index=True)
    manifest = api().make_enrollment_manifest(shuffled, extended, 0, raw)
    assert manifest["eligible_ids"] == ["g1", "g2", "g3", "g4"]


def test_external_blank_hash_and_independent_declaration_fail_closed(inventory, tmp_path):
    raw, records, split, _, texture = inventory
    specimen = tmp_path / "specimen"
    specimen.mkdir()
    blank_path = specimen / "blank.png"
    assert cv2.imwrite(str(blank_path), texture)
    manifest = api().make_enrollment_manifest(records, split, 0, raw)
    declaration = {"path": "blank.png", "sha256": hashlib.sha256(blank_path.read_bytes()).hexdigest(),
                   "source": "independent synthetic blank", "independent": False}
    provenance = specimen / "provenance.json"
    provenance.write_text(json.dumps({"schema_version": "external_blank_v1", "specimens": {"sidtd/alb_id": declaration}}))
    with pytest.raises(ValueError, match="independent"):
        api().build_enrollment_products(manifest, raw, tmp_path / "undeclared", specimen)
    declaration["independent"] = True
    provenance.write_text(json.dumps({"schema_version": "external_blank_v1", "specimens": {"sidtd/alb_id": declaration}}))
    assert cv2.imwrite(str(blank_path), np.roll(texture, 9, axis=0))
    with pytest.raises(ValueError, match="hash|immutable"):
        api().build_enrollment_products(manifest, raw, tmp_path / "mutable", specimen)
    assert not (tmp_path / "mutable").exists()


def declare_external_specimen(root, path, **metadata):
    declaration = {"path": path.relative_to(root).as_posix(),
                   "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                   "source": "independent synthetic blank", "independent": True, **metadata}
    (root / "provenance.json").write_text(json.dumps({"schema_version": "external_blank_v1",
                                                     "specimens": {"sidtd/alb_id": declaration}}))


@pytest.mark.parametrize("image_id", ["g1", "g0", "g5", "copy", "reference-candidate"])
def test_external_specimen_rejects_complete_dataset_inventory(inventory, tmp_path, monkeypatch, image_id):
    raw, records, split, _, texture = inventory
    candidate = dict(records.iloc[0], image_id="reference-candidate", dataset="fantasyid",
                     design_id="fantasyid/reference", identity_id="fantasyid/reference-01",
                     component_id="fantasyid/reference-01", source_id="", donor_id="",
                     path="fantasyid/references/candidate.png", annotation="", role="reference_only")
    records = pd.concat([records, pd.DataFrame([candidate])], ignore_index=True)
    row = records.set_index("image_id").loc[image_id]
    path = raw / row.path
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), texture)
    manifest = api().make_enrollment_manifest(records, split, 0, raw)
    declare_external_specimen(raw, path)
    original_hash = api()._file_hash

    def no_dataset_specimen_hash(candidate_path):
        assert candidate_path != path
        return original_hash(candidate_path)

    monkeypatch.setattr(api(), "_file_hash", no_dataset_specimen_hash)
    with pytest.raises(ValueError, match="inventory|independent"):
        api().build_enrollment_products(manifest, raw, tmp_path / "dataset-specimen", raw)
    assert not (tmp_path / "dataset-specimen").exists()


@pytest.mark.parametrize("provenance", ["image_id", "identity_id", "source_id", "donor_id", "component_id", "raw_sha256"])
def test_external_specimen_rejects_authoritative_metadata_overlap(inventory, tmp_path, provenance):
    raw, records, split, _, texture = inventory
    specimen = tmp_path / "external"
    specimen.mkdir()
    path = specimen / "renamed.png"
    assert cv2.imwrite(str(path), texture)
    metadata = {}
    if provenance == "raw_sha256":
        records["raw_sha256"] = ""
        records.loc[records.image_id == "g5", "raw_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    else:
        heldout = records.set_index("image_id").loc["g5"]
        metadata[provenance] = "g5" if provenance == "image_id" else heldout.identity_id
    manifest = api().make_enrollment_manifest(records, split, 0, raw)
    declare_external_specimen(specimen, path, **metadata)
    with pytest.raises(ValueError, match="inventory|independent"):
        api().build_enrollment_products(manifest, raw, tmp_path / "metadata-specimen", specimen)
    assert not (tmp_path / "metadata-specimen").exists()


def test_inventory_authority_is_metadata_only_and_covers_unassigned_candidates(inventory, monkeypatch):
    raw, records, split, _, _ = inventory
    candidate = dict(records.iloc[0], image_id="reference-candidate", dataset="fantasyid",
                     design_id="fantasyid/reference", identity_id="fantasyid/reference-01",
                     component_id="fantasyid/reference-01", source_id="", donor_id="",
                     path="fantasyid/references/missing.png", annotation="", role="reference_only")
    records = pd.concat([records, pd.DataFrame([candidate])], ignore_index=True)
    for row in records[~records.image_id.isin(["g1", "g2", "g3", "g4"])].itertuples():
        (raw / row.path).unlink(missing_ok=True)
    original_hash = api()._file_hash
    authorized = {raw / records.set_index("image_id").loc[image_id, "path"] for image_id in ("g1", "g2", "g3", "g4")}
    annotation = raw / "sidtd/templates/Annotations/reals/alb_id.json"

    def support_only_hash(path):
        assert path in authorized or path == annotation or not path.is_relative_to(raw)
        return original_hash(path)

    monkeypatch.setattr(api(), "_file_hash", support_only_hash)
    manifest = api().make_enrollment_manifest(records, split, 0, raw)
    authority = manifest["dataset_inventory"]
    assert authority["records_sha256"] == api()._table_hash(records)
    entries = {entry["image_id"]: entry for entry in authority["entries"]}
    assert set(entries) == set(records.image_id)
    assert entries["g5"]["role"] == "test"
    assert entries["copy"]["role"] == "val"
    assert entries["reference-candidate"]["role"] == "reference_only"
    assert entries["copy"]["source_id"] == "sidtd/alb_id_00"
    assert entries["copy"]["donor_id"] == "sidtd/alb_id_06"
    assert authority["entries_sha256"] == api()._digest(authority["entries"])


@pytest.mark.parametrize("failure", ["protocol", "binding", "wrong_protocol", "wrong_fold", "incomplete", "incomplete_unmarked"])
def test_official_protocol_fold_and_complete_membership_are_required(inventory, failure):
    raw, records, split, _, _ = inventory
    if failure in ("protocol", "incomplete_unmarked"):
        split = split.drop(columns="protocol")
    if failure in ("binding", "incomplete_unmarked"):
        split = split.drop(columns="ref_group")
    if failure == "wrong_protocol":
        split["protocol"] = "B"
    if failure == "wrong_fold":
        split["fold"] = 9
    if failure in ("incomplete", "incomplete_unmarked"):
        split = split[split.image_id != "g5"]
    with pytest.raises(ValueError, match="protocol|Protocol|fold|SIDTD|binding"):
        api().make_enrollment_manifest(records, split, 0, raw)


def test_explicit_numeric_fold_binding_is_supported(inventory):
    raw, records, split, _, _ = inventory
    split = split.drop(columns="ref_group")
    split["fold"] = 0
    manifest = api().make_enrollment_manifest(records, split, 0, raw)
    assert manifest["official_fold_binding"] == {"protocol": "A", "fold": 0, "ref_group": "it0"}


@pytest.mark.parametrize("failure", ["missing", "changed_entries"])
def test_external_specimen_requires_frozen_inventory_authority(inventory, tmp_path, failure):
    raw, records, split, _, texture = inventory
    manifest = api().make_enrollment_manifest(records, split, 0, raw)
    if failure == "missing":
        manifest.pop("dataset_inventory", None)
    else:
        manifest["dataset_inventory"]["entries"] = []
    specimen = tmp_path / "external"
    specimen.mkdir()
    path = specimen / "blank.png"
    assert cv2.imwrite(str(path), texture)
    declare_external_specimen(specimen, path)
    with pytest.raises(ValueError, match="inventory|authority"):
        api().build_enrollment_products(manifest, raw, tmp_path / "missing-authority", specimen)


def test_external_specimen_rejects_hardlinked_dataset_image(inventory, tmp_path):
    raw, records, split, _, _ = inventory
    manifest = api().make_enrollment_manifest(records, split, 0, raw)
    specimen = tmp_path / "external"
    specimen.mkdir()
    path = specimen / "blank.png"
    path.hardlink_to(raw / records.set_index("image_id").loc["g5", "path"])
    declare_external_specimen(specimen, path)
    with pytest.raises(ValueError, match="inventory|independent"):
        api().build_enrollment_products(manifest, raw, tmp_path / "hardlinked-specimen", specimen)
    assert not (tmp_path / "hardlinked-specimen").exists()


@pytest.mark.parametrize("failure", ["missing", "fold", "protocol"])
def test_build_requires_frozen_official_fold_binding(inventory, tmp_path, failure):
    raw, records, split, _, _ = inventory
    manifest = api().make_enrollment_manifest(records, split, 0, raw)
    if failure == "missing":
        manifest.pop("official_fold_binding")
    elif failure == "fold":
        manifest["official_fold_binding"]["fold"] = 9
    else:
        manifest["official_fold_binding"]["protocol"] = "B"
    with pytest.raises(ValueError, match="official|fold|binding"):
        api().build_enrollment_products(manifest, raw, tmp_path / "unbound")
    assert not (tmp_path / "unbound").exists()


def test_failed_contributors_are_recorded_not_identity_warped(inventory, tmp_path):
    raw, records, split, _, _ = inventory
    manifest = api().make_enrollment_manifest(records, split, 0, raw)
    path = raw / records.set_index("image_id").loc["g4", "path"]
    assert cv2.imwrite(str(path), np.zeros((128, 200, 3), np.uint8))
    products = api().build_enrollment_products(manifest, raw, tmp_path / "partial")
    bank = products["banks"]["sidtd/alb_id"]["enrollment"]
    assert bank["counts"]["authorized"] == 4
    assert bank["counts"]["registered"] == 3
    assert bank["counts"]["failed"] == 1
    assert bank["registered_ids"] == ["g1", "g2", "g3"]
    geometry = json.loads(Path(bank["products"]["geometry"]["path"]).read_text())
    failure = geometry["registrations"]["g4"]
    assert not failure["target_applied"]
    assert failure["native_to_bank"] is None and failure["canvas_to_bank"] is None
    assert failure["failure"]
    assert set(geometry["contributors"]) == {"g1", "g2", "g3", "g4"}


def test_cli_builds_synthetic_only_and_refuses_existing_output(inventory, tmp_path):
    _, records, split, _, _ = inventory
    records_path, split_path = tmp_path / "records.csv", tmp_path / "split.csv"
    records.to_csv(records_path, index=False)
    split.to_csv(split_path, index=False)
    command = [sys.executable, "scripts/prepare_anchor_enrollment.py", "--records", str(records_path),
               "--split", str(split_path), "--fold", "0", "--raw-root", str(inventory[0]), "--out", str(tmp_path / "cli")]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "cli/manifest.json").is_file()
    manifest = json.loads((tmp_path / "cli/manifest.json").read_text())
    assert manifest["input_files"]["records"]["sha256"] == hashlib.sha256(records_path.read_bytes()).hexdigest()
    assert manifest["input_files"]["split"]["sha256"] == hashlib.sha256(split_path.read_bytes()).hexdigest()
    again = subprocess.run(command, capture_output=True, text=True)
    assert again.returncode != 0
