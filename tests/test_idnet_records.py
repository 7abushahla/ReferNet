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


@pytest.mark.parametrize("group,field,value,match", [
    ("fraud2_face_morphing", "overlayface", "unknown", "unresolved"),
    ("fraud2_face_morphing", "baseface", "portrait-1", "ambiguous"),
    ("fraud3_face_replacement", "newface", "outside.png", "unresolved"),
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
