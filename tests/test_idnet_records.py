import hashlib
import importlib
import json
from pathlib import Path

import pytest


GROUPS = {
    "positive": "AZ_original_annotation.json",
    "fraud1_copy_and_move": "AZ_textfieldreplacement_annotation.json",
    "fraud2_face_morphing": "AZ_facemorph_annotation.json",
    "fraud3_face_replacement": "AZ_facereplacement_annotation.json",
    "fraud4_combined": "AZ_combined_fraud_annotation.json",
}


@pytest.fixture
def parser(monkeypatch):
    path = Path(__file__).resolve().parents[1] / "src/refer_or_clear/data/idnet_records.py"
    assert path.exists(), "AZ metadata parser is not implemented"
    monkeypatch.syspath_prepend(str(path.parents[2]))
    return importlib.import_module("refer_or_clear.data.idnet_records")


@pytest.fixture
def archive(tmp_path):
    metadata = tmp_path / "meta/detailed_with_fraud_info"
    metadata.mkdir(parents=True)
    names = ["document-a.png", "document-b.png"]
    annotations = {group: {} for group in GROUPS}
    for index, name in enumerate(names):
        donor = names[1 - index]
        annotations["positive"][name] = {"fraud": "False"}
        annotations["fraud1_copy_and_move"][name] = {"fraud": "True", "last_name": {"original_value": "OLD", "new_value": "NEW"}}
        annotations["fraud2_face_morphing"][name] = {"baseface": f"portrait-{index}", "overlayface": f"portrait-{1-index}"}
        annotations["fraud3_face_replacement"][name] = {"originalface": name, "newface": donor}
        annotations["fraud4_combined"][name] = {"originalface": name, "face_morphing": "False", "face_replacement": "True", "replaced_new_face": donor}
    for group, filename in GROUPS.items():
        (metadata / filename).write_text(json.dumps(annotations[group]))
        (tmp_path / group).mkdir()
        for name in names:
            (tmp_path / group / name).touch()
    return tmp_path, annotations


def rewrite(archive, group, name, field, value):
    root, annotations = archive
    annotations[group][name][field] = value
    (root / "meta/detailed_with_fraud_info" / GROUPS[group]).write_text(json.dumps(annotations[group]))


def test_explicit_lineage_no_pixels_no_roles_or_guessed_geometry(parser, archive, monkeypatch):
    original = Path.read_bytes

    def metadata_only(path):
        assert path.suffix == ".json", "image bytes must never be read"
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", metadata_only)
    result = parser.idnet_records(archive[0])
    assert len(result) == 10
    assert result.image_id.is_unique
    assert not {"width", "height", "role", "split"} & set(result.columns)
    morph = result[result.attack_source == "fraud2_face_morphing"].iloc[0]
    assert morph.identity_id == morph.source_id == "idnet/AZ/document-a.png"
    assert morph.donor_id == "idnet/AZ/document-b.png"
    assert result.component_id.nunique() == 1
    text = result[result.attack_source == "fraud1_copy_and_move"]
    assert set(text.donor_id) == {""}
    assert set(text.donor_status) == {"not_declared_by_annotation"}
    assert len(result.attrs["annotation_sha256"]) == 5
    filename = GROUPS["positive"]
    assert result.attrs["annotation_sha256"][filename] == hashlib.sha256(original(archive[0] / "meta/detailed_with_fraud_info" / filename)).hexdigest()


@pytest.mark.parametrize("group,field", [("fraud3_face_replacement", "newface"), ("fraud4_combined", "replaced_new_face")])
@pytest.mark.parametrize("token", ["generated.photos_v3_0950632_0222045_0671460.png", "generated.photos_0997964.png"])
def test_external_portrait_namespace_keeps_raw_token_without_document_edge(parser, archive, monkeypatch, group, field, token):
    original = Path.read_bytes

    def metadata_only(path):
        assert path.suffix == ".json", "image bytes must never be read"
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", metadata_only)
    for name in archive[1]["fraud2_face_morphing"]:
        index = 0 if name == "document-a.png" else 1
        rewrite(archive, "fraud2_face_morphing", name, "overlayface", f"portrait-{index}")
        rewrite(archive, "fraud3_face_replacement", name, "newface", name)
        rewrite(archive, "fraud4_combined", name, "replaced_new_face", name)
    for name in archive[1][group]:
        rewrite(archive, group, name, field, token)
    result = parser.idnet_records(archive[0])
    changed = result[result.attack_source == group]
    assert set(changed.donor_id) == {""}
    assert set(changed.donor_token) == {token}
    assert set(changed.donor_namespace) == {"generated.photos"}
    assert set(changed.donor_status) == {"external_portrait_unresolved_to_document"}
    assert result.component_id.nunique() == 2
    assert not {"width", "height", "role", "split"} & set(result.columns)


def test_replacement_uses_explicit_baseface_map_for_document_edges(parser, archive):
    rewrite(archive, "fraud3_face_replacement", "document-a.png", "newface", "portrait-1")
    result = parser.idnet_records(archive[0])
    replaced = result[(result.attack_source == "fraud3_face_replacement") & (result.annotation_key == "document-a.png")].iloc[0]
    assert replaced.donor_id == "idnet/AZ/document-b.png"
    assert replaced.donor_token == "portrait-1"
    assert replaced.donor_namespace == "baseface"
    assert replaced.donor_status == "declared"


def test_generated_photos_document_token_is_still_a_document_when_explicitly_in_inventory(parser, archive):
    root, annotations = archive
    old_name = "document-b.png"
    new_name = "generated.photos_v3_0950632.png"
    for group, filename in GROUPS.items():
        annotations[group][new_name] = annotations[group].pop(old_name)
        (root / group / old_name).rename(root / group / new_name)
        for entry in annotations[group].values():
            for field in ("originalface", "newface", "replaced_new_face"):
                if entry.get(field) == old_name:
                    entry[field] = new_name
        (root / "meta/detailed_with_fraud_info" / filename).write_text(json.dumps(annotations[group]))
    result = parser.idnet_records(root)
    replaced = result[(result.attack_source == "fraud3_face_replacement") & (result.annotation_key == "document-a.png")].iloc[0]
    assert replaced.donor_id == f"idnet/AZ/{new_name}"
    assert replaced.donor_token == new_name and replaced.donor_namespace == "document"
    assert replaced.donor_status == "declared" and result.component_id.nunique() == 1


@pytest.mark.parametrize("group,field,value,match", [
    ("fraud2_face_morphing", "overlayface", "unknown", "unresolved"),
    ("fraud2_face_morphing", "baseface", "portrait-1", "ambiguous"),
    ("fraud3_face_replacement", "newface", "outside.png", "unresolved"),
    ("fraud3_face_replacement", "newface", "generated.photos_v3_.png", "unresolved"),
    ("fraud3_face_replacement", "newface", "../generated.photos_v3_0950632.png", "unresolved"),
    ("fraud3_face_replacement", "newface", None, "unresolved"),
    ("fraud3_face_replacement", "originalface", "document-b.png", "source"),
    ("fraud4_combined", "replaced_new_face", "none", "unresolved"),
    ("fraud4_combined", "face_morphing", "True", "combined morph"),
    ("fraud4_combined", "face_replacement", True, "flags"),
    ("positive", "fraud", "True", "polarity"),
    ("fraud1_copy_and_move", "fraud", False, "polarity"),
])
def test_fails_closed_unknown_or_ambiguous_lineage(parser, archive, group, field, value, match):
    if group == "fraud4_combined" and field == "face_morphing":
        rewrite(archive, group, "document-a.png", "face_replacement", "False")
    rewrite(archive, group, "document-a.png", field, value)
    with pytest.raises(ValueError, match=match):
        parser.idnet_records(archive[0])


def test_inventory_mismatch_is_not_silently_dropped(parser, archive):
    (archive[0] / "positive/unannotated.png").touch()
    with pytest.raises(ValueError, match="inventory"):
        parser.idnet_records(archive[0])


def test_duplicate_annotation_key_rejected(parser, archive):
    (archive[0] / "meta/detailed_with_fraud_info" / GROUPS["positive"]).write_text('{"document-a.png":{},"document-a.png":{}}')
    with pytest.raises(ValueError, match="duplicate"):
        parser.idnet_records(archive[0])


def test_annotation_traversal_rejected(parser, archive):
    root, annotations = archive
    annotations["positive"]["../escape.png"] = annotations["positive"].pop("document-a.png")
    (root / "meta/detailed_with_fraud_info" / GROUPS["positive"]).write_text(json.dumps(annotations["positive"]))
    with pytest.raises(ValueError, match="filename"):
        parser.idnet_records(root)


def test_image_symlink_rejected_without_opening(parser, archive):
    path = archive[0] / "positive/document-a.png"
    path.unlink()
    path.symlink_to(archive[0] / "positive/document-b.png")
    with pytest.raises(ValueError, match="symlink"):
        parser.idnet_records(archive[0])


def test_external_portrait_does_not_waive_ambiguous_combined_morph(parser, archive):
    token = "generated.photos_v3_0950632_0222045_0671460.png"
    rewrite(archive, "fraud4_combined", "document-a.png", "face_morphing", "True")
    rewrite(archive, "fraud4_combined", "document-a.png", "face_replacement", "False")
    rewrite(archive, "fraud4_combined", "document-a.png", "replaced_new_face", token)
    with pytest.raises(ValueError, match="combined morph"):
        parser.idnet_records(archive[0])
