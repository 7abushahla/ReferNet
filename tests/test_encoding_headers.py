import builtins
import csv
import errno
import hashlib
import importlib.util
import json
import subprocess
import struct
import sys
import types
import zlib
from pathlib import Path

import pytest
from PIL import Image


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/audit_encoding_headers.py"


@pytest.fixture
def audit():
    assert SCRIPT.exists(), "header-only audit is not implemented"
    spec = importlib.util.spec_from_file_location("encoding_headers", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def image(path, quality=75, subsampling=2, **kwargs):
    Image.new("RGB", (19, 13), (100, 50, 25)).save(
        path, format="JPEG", quality=quality, subsampling=subsampling, **kwargs
    )


def manifests(tmp_path, rows):
    records = tmp_path / "records.csv"
    split = tmp_path / "protocol_c.csv"
    fields = ["image_id", "dataset", "path", "label", "design_id", "identity_id",
              "source_id", "donor_id", "component_id", "attack_family", "capture"]
    with records.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
    with split.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["image_id", "role", "subset", "label"])
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in writer.fieldnames})
    return records, split


def row(identifier, label=0, role="test", subset="heldout", **kwargs):
    return dict(image_id=identifier, dataset="fantasyid", path=f"{identifier}.jpg",
                label=label, role=role, subset=subset, design_id="fantasyid/a",
                identity_id=identifier, **kwargs)


def test_no_pixel_decode_or_model_import(audit, tmp_path, monkeypatch):
    path = tmp_path / "sample.jpg"
    image(path)
    original_import = builtins.__import__

    def forbid(*args, **kwargs):
        raise AssertionError("pixel decoding forbidden")

    def guarded_import(name, *args, **kwargs):
        if name.split(".")[0] in {"torch", "torchvision", "cv2", "refer_or_clear"}:
            raise AssertionError(f"model/decoder import forbidden: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(Image.Image, "load", forbid)
    decoder_spies = types.ModuleType("cv2")
    decoder_spies.imread = forbid
    decoder_spies.imdecode = forbid
    monkeypatch.setitem(sys.modules, "cv2", decoder_spies)
    monkeypatch.setattr(builtins, "__import__", guarded_import)
    spec = importlib.util.spec_from_file_location("fresh_header_audit", SCRIPT)
    fresh = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fresh)
    assert fresh.read_encoding_header(path)["format"] == "JPEG"
    records, split = manifests(tmp_path, [row("sample")])
    assert fresh.audit_target_headers(records, split, tmp_path, tmp_path / "out")["counts"]["ok"] == 1


def test_png_payload_under_jpg_extension(audit, tmp_path, monkeypatch):
    path = tmp_path / "disguised.jpg"
    Image.new("RGBA", (9, 7)).save(path, format="PNG")
    monkeypatch.setattr(Image.Image, "load", lambda *args: pytest.fail("decoded pixels"))
    header = audit.read_encoding_header(path)
    assert (header["format"], header["width"], header["height"]) == ("PNG", 9, 7)
    assert header["color_markers"]["color_type"] == 6
    assert header["quantization_tables"] == {}
    assert header["estimated_quality"] is None
    assert header["exact_ijg_candidates"] == []


@pytest.mark.parametrize("quality", [1, 40, 75, 95, 100])
def test_exact_tables_are_distinct_from_quality_estimate(audit, tmp_path, quality):
    path = tmp_path / "standard.jpg"
    image(path, quality=quality)
    header = audit.read_encoding_header(path)
    assert quality in header["exact_ijg_candidates"]
    assert header["table_status"] == "standard_ijg_match"
    assert len(header["quantization_tables"]["0"]) == 64
    assert len(header["quantization_tables"]["1"]) == 64
    assert header["estimate_method"] == "legacy_luminance_median_ratio"
    assert header["estimated_quality"] is not None


def test_custom_chroma_and_sampling_are_not_luma_quality(audit, tmp_path):
    base = tmp_path / "base.jpg"
    image(base, subsampling=0)
    with Image.open(base) as opened:
        tables = opened.quantization
    tables[1][0] += 1
    custom = tmp_path / "custom.jpg"
    image(custom, quality=-1, qtables=tables)
    header = audit.read_encoding_header(custom)
    assert header["exact_ijg_candidates"] == []
    assert header["table_status"] == "custom_or_unmatched"
    assert header["estimated_quality"] == audit.read_encoding_header(base)["estimated_quality"]
    second = tmp_path / "second.jpg"
    image(second, subsampling=2)
    first_header, second_header = audit.read_encoding_header(base), audit.read_encoding_header(second)
    assert first_header["table_signature"] == second_header["table_signature"]
    assert first_header["sampling_signature"] != second_header["sampling_signature"]


def test_missing_file_is_counted_not_dropped(audit, tmp_path):
    image(tmp_path / "good.jpg", quality=95)
    (tmp_path / "bad.jpg").write_bytes(b"not an image")
    records, split = manifests(tmp_path, [row("good"), row("missing", 1), row("bad", 1)])
    result = audit.audit_target_headers(records, split, tmp_path, tmp_path / "out")
    assert result["counts"] == {"total": 3, "genuine": 1, "altered": 2, "ok": 1, "missing": 1,
                                "corrupt": 1, "unsupported": 0, "access_error": 0, "error": 0}
    assert result["groups"]["heldout"]["metrics"]["eligible"] == 1
    assert result["groups"]["heldout"]["metrics"]["auc_negative_estimate"] is None
    with (tmp_path / "out/per_image.csv").open() as stream:
        assert len(list(csv.DictReader(stream))) == 3
    assert (tmp_path / "out/records.csv").read_bytes() == records.read_bytes()
    assert (tmp_path / "out/protocol_c.csv").read_bytes() == split.read_bytes()
    assert json.loads((tmp_path / "out/summary.json").read_text()) == result
    with pytest.raises(FileExistsError):
        audit.audit_target_headers(records, split, tmp_path, tmp_path / "out")


@pytest.mark.parametrize("problem", ["duplicate_record", "duplicate_split", "unknown", "label_conflict"])
def test_membership_join_is_one_to_one(audit, tmp_path, problem):
    records, split = manifests(tmp_path, [row("one")])
    if problem.startswith("duplicate"):
        path = records if problem == "duplicate_record" else split
        with path.open("a") as stream:
            stream.write(path.read_text().splitlines()[1] + "\n")
    elif problem == "unknown":
        split.write_text(split.read_text().replace("one,", "unknown,"))
    else:
        split.write_text(split.read_text().replace("heldout,0", "heldout,1"))
    with pytest.raises(ValueError):
        audit.audit_target_headers(records, split, tmp_path, tmp_path / "out")


def test_fixed_direction_subsets_and_support_gaps(audit, tmp_path):
    rows = [row("g"), row("a", 1),
            row("attack1", 1, subset="attack1"), row("support", role="reference_only"),
            row("excluded", 1, role="train")]
    rows[0]["identity_id"] = "shared"
    rows[2]["design_id"] = "fantasyid/b"
    rows[3]["source_id"] = "shared"
    for item in rows:
        image(tmp_path / item["path"], quality=75 if item["label"] == 0 else 95)
    records, split = manifests(tmp_path, rows)
    result = audit.audit_target_headers(records, split, tmp_path, tmp_path / "out")
    assert result["membership_version"] == "membership_candidate_v1"
    assert result["counts"]["total"] == 4
    assert result["groups"]["heldout"]["metrics"]["auc_negative_estimate"] == 0.0
    assert result["groups"]["heldout"]["metrics"]["accuracy_estimate_lt_85"] == 0.0
    assert result["groups"]["attack1"]["counts"]["genuine"] == 0
    assert result["groups"]["attack1"]["metrics"]["auc_negative_estimate"] is None
    assert result["support"]["designs_without_separate_support"] == ["fantasyid/a", "fantasyid/b"]
    assert result["support"]["provenance_overlap_tokens"] == ["shared"]
    assert result["support"]["approved"] is False
    assert result["groups"]["candidate_test"]["counts"]["total"] == 3


def test_cli(audit, tmp_path):
    image(tmp_path / "one.jpg")
    records, split = manifests(tmp_path, [row("one")])
    completed = subprocess.run([sys.executable, str(SCRIPT), "--records", str(records),
                                "--split", str(split), "--raw-root", str(tmp_path),
                                "--out", str(tmp_path / "cli")], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    stdout = json.loads(completed.stdout)
    assert stdout["schema_version"] == 2
    assert stdout["completeness"]["exhaustive_test_inventory_confirmed"] is False
    assert "membership_candidate_v1" in (tmp_path / "cli/report.md").read_text()


def test_unknown_evaluation_identity_cannot_verify_separate_support(audit, tmp_path):
    rows = [row("evaluation"), row("support", role="reference_only")]
    rows[0]["identity_id"] = ""
    for item in rows:
        image(tmp_path / item["path"])
    records, split = manifests(tmp_path, rows)
    result = audit.audit_target_headers(records, split, tmp_path, tmp_path / "out")
    assert result["support"]["unknown_identity_evaluation_image_ids"] == ["evaluation"]
    assert result["support"]["designs_without_separate_support"] == ["fantasyid/a"]


def test_provenance_overlap_is_transitive(audit, tmp_path):
    rows = [row("evaluation"), row("support", role="reference_only"), row("bridge", 1, role="train")]
    rows[2]["source_id"] = "evaluation"
    rows[2]["donor_id"] = "support"
    for item in rows[:2]:
        image(tmp_path / item["path"])
    records, split = manifests(tmp_path, rows)
    result = audit.audit_target_headers(records, split, tmp_path, tmp_path / "out")
    assert result["support"]["overlapping_support_image_ids"] == ["support"]
    assert result["support"]["designs_without_separate_support"] == ["fantasyid/a"]
    assert result["counts"]["missing"] == 0


def test_metadata_is_frozen_before_headers_and_official_omissions_reported(audit, tmp_path, monkeypatch):
    rows = [row("evaluation"), row("omitted", role="train")]
    for item in rows:
        item["path"] = f"fantasyid/test/bonafide/{item['image_id']}.jpg"
    records, split = manifests(tmp_path, rows)
    frozen_records = records.read_bytes()
    output = tmp_path / "out"
    calls = []

    def inspect(path):
        assert (output / "records.csv").read_bytes() == frozen_records
        assert json.loads((output / "membership.json").read_text())[0]["image_id"] == "evaluation"
        calls.append(path)
        records.write_text("changed after snapshot")
        raise FileNotFoundError("synthetic missing header")

    monkeypatch.setattr(audit, "read_encoding_header", inspect)
    result = audit.audit_target_headers(records, split, tmp_path, output)
    assert len(calls) == 1
    assert result["official_test_membership"]["missing_from_candidate"] == ["omitted"]
    assert result["counts"]["missing"] == 1
    assert result["manifests"]["records"]["sha256"] == hashlib.sha256(frozen_records).hexdigest()


def test_unsafe_dimensions_are_reported_as_corrupt_headers(audit, tmp_path):
    path = tmp_path / "unsafe.jpg"
    Image.new("RGB", (1, 1)).save(path, format="PNG")
    payload = bytearray(path.read_bytes())
    payload[16:24] = struct.pack(">II", 100000, 100000)
    payload[29:33] = struct.pack(">I", zlib.crc32(payload[12:29]))
    path.write_bytes(payload)
    records, split = manifests(tmp_path, [row("unsafe")])
    result = audit.audit_target_headers(records, split, tmp_path, tmp_path / "out")
    assert result["counts"]["corrupt"] == 1
    assert "DecompressionBombError" in result["omissions"][0]["error"]


def test_incomplete_banner_and_metric_coverage(audit, tmp_path):
    records, split = manifests(tmp_path, [row("missing"), row("omitted", role="train")])
    result = audit.audit_target_headers(records, split, tmp_path, tmp_path / "out")
    assert result["completeness"]["complete"] is False
    assert result["completeness"]["candidate_headers_complete"] is False
    assert result["completeness"]["exhaustive_test_inventory_confirmed"] is False
    report = (tmp_path / "out/report.md").read_text()
    assert "INCOMPLETE" in report
    assert "Estimate excluded" in report
    assert "complete_official_test" not in result["groups"]


def test_optional_disk_inventory_does_not_open_unselected_images(audit, tmp_path, monkeypatch):
    rows = [row("selected"), row("not_on_disk", role="train")]
    for item in rows:
        item["path"] = f"fantasyid/test/bonafide/{item['image_id']}.jpg"
    folder = tmp_path / "fantasyid/test/bonafide"
    folder.mkdir(parents=True)
    image(folder / "selected.jpg")
    (folder / "unrecorded.JPEG").write_bytes(b"never inspect this unselected payload")
    (folder / "annotation.json").write_text("{}")
    original = audit.read_encoding_header
    calls = []

    def inspect(path):
        calls.append(path.name)
        return original(path)

    monkeypatch.setattr(audit, "read_encoding_header", inspect)
    records, split = manifests(tmp_path, rows)
    result = audit.audit_target_headers(records, split, tmp_path, tmp_path / "out", inventory_test_directory=True)
    inventory = result["disk_inventory"]
    assert calls == ["selected.jpg"]
    assert inventory["disk_images_absent_records"] == ["fantasyid/test/bonafide/unrecorded.JPEG"]
    assert inventory["records_absent_disk"] == ["fantasyid/test/bonafide/not_on_disk.jpg"]
    assert inventory["disk_image_count"] == 2
    assert result["completeness"]["complete"] is False


def test_positive_clean_separate_support_and_confirmed_inventory(audit, tmp_path):
    rows = [row("evaluation", capture="scan", component_id="eval-component"),
            row("support", role="reference_only", capture="scan", component_id="support-component")]
    rows[0]["path"] = "fantasyid/test/bonafide/evaluation.jpg"
    (tmp_path / "fantasyid/test/bonafide").mkdir(parents=True)
    for item in rows:
        image(tmp_path / item["path"])
    records, split = manifests(tmp_path, rows)
    result = audit.audit_target_headers(records, split, tmp_path, tmp_path / "out", inventory_test_directory=True)
    assert result["support"]["overlapping_support_image_ids"] == []
    assert result["support"]["designs_without_separate_support"] == []
    assert result["support"]["separate_support_image_ids"] == ["support"]
    assert result["support"]["approved"] is False
    assert result["completeness"]["complete"] is True
    assert result["completeness"]["exhaustive_test_inventory_confirmed"] is True
    assert "INCOMPLETE" not in (tmp_path / "out/report.md").read_text()


def test_signature_disjointness_is_independent_of_estimate_direction(audit, tmp_path):
    rows = [row("genuine"), row("altered", 1), row("png", 1)]
    image(tmp_path / "genuine.jpg", quality=75, subsampling=0)
    image(tmp_path / "altered.jpg", quality=95, subsampling=2)
    Image.new("RGB", (2, 2)).save(tmp_path / "png.jpg", format="PNG")
    records, split = manifests(tmp_path, rows)
    result = audit.audit_target_headers(records, split, tmp_path, tmp_path / "out")
    group = result["groups"]["candidate_test"]
    assert group["metrics"]["auc_negative_estimate"] == 0.0
    for kind in ("quantization", "sampling"):
        association = group["signature_associations"][kind]
        assert association["eligible"] == 2
        assert association["excluded"] == 1
        assert association["eligible_class_counts"] == {"genuine": 1, "altered": 1}
        assert association["class_disjoint"] is True
        assert association["full_coverage"] is False
        assert association["population_class_disjoint"] is None
        assert len(association["signatures"]) == 2
    assert "Signature associations" in (tmp_path / "out/report.md").read_text()


def test_shared_and_single_class_signatures(audit):
    rows = [dict(label=0, table_signature="same", status="ok"),
            dict(label=1, table_signature="same", status="ok")]
    shared = audit._signature_association(rows, "table_signature")
    assert shared["class_disjoint"] is False
    assert shared["population_class_disjoint"] is False
    assert shared["signatures"] == [{"signature": "same", "genuine": 1, "altered": 1}]
    assert audit._signature_association(rows[:1], "table_signature")["class_disjoint"] is None


def test_sampling_signature_ignores_component_and_table_ids(audit, tmp_path):
    original = tmp_path / "original.jpg"
    changed = tmp_path / "changed.jpg"
    image(original)
    payload = bytearray(original.read_bytes())
    frame = payload.index(b"\xff\xc0")
    scan = payload.index(b"\xff\xda")
    for position in range(payload[frame + 9]):
        payload[frame + 10 + 3 * position] += 10
        payload[frame + 12 + 3 * position] ^= 1
        payload[scan + 5 + 2 * position] += 10
    changed.write_bytes(payload)
    first, second = audit.read_encoding_header(original), audit.read_encoding_header(changed)
    assert first["components"] != second["components"]
    assert first["sampling_signature"] == second["sampling_signature"]


def test_duplicate_resolved_paths_and_populated_missing_metadata(audit, tmp_path):
    rows = [row("one"), row("two", 1)]
    rows[1]["path"] = "./one.jpg"
    image(tmp_path / "one.jpg")
    records, split = manifests(tmp_path, rows)
    result = audit.audit_target_headers(records, split, tmp_path, tmp_path / "out")
    assert result["duplicate_selected_paths"][0]["image_ids"] == ["one", "two"]
    assert result["metadata_quality"]["missing_values"]["capture"] == ["one", "two"]
    assert result["metadata_quality"]["missing_values"]["source_id"] == ["two"]
    assert result["metadata_quality"]["missing_values"]["attack_family"] == ["two"]
    assert result["completeness"]["complete"] is False


def test_identity_and_component_tokens_cannot_collide(audit, tmp_path):
    rows = [row("evaluation", component_id="shared"), row("support", role="reference_only")]
    rows[1]["identity_id"] = "component:shared"
    for item in rows:
        image(tmp_path / item["path"])
    records, split = manifests(tmp_path, rows)
    result = audit.audit_target_headers(records, split, tmp_path, tmp_path / "out")
    assert result["support"]["overlapping_support_image_ids"] == []
    assert result["support"]["designs_without_separate_support"] == []


def test_provenance_values_are_identity_ids_not_image_ids(audit, tmp_path):
    rows = [row("evaluation"), row("support", role="reference_only"), row("bridge", 1, role="train")]
    rows[0]["identity_id"] = "fantasyid/eval-document"
    rows[1]["identity_id"] = "fantasyid/support-document"
    rows[2]["source_id"] = "fantasyid/eval-document"
    rows[2]["donor_id"] = "fantasyid/support-document"
    for item in rows[:2]:
        image(tmp_path / item["path"])
    records, split = manifests(tmp_path, rows)
    result = audit.audit_target_headers(records, split, tmp_path, tmp_path / "out")
    assert result["support"]["overlapping_support_image_ids"] == ["support"]
    assert result["support"]["provenance_semantics"] == "source_id and donor_id are identity IDs, not image IDs"


def test_unsupported_payload_is_not_corrupt(audit, tmp_path):
    Image.new("RGB", (2, 2)).save(tmp_path / "gif.jpg", format="GIF")
    records, split = manifests(tmp_path, [row("gif")])
    result = audit.audit_target_headers(records, split, tmp_path, tmp_path / "out")
    assert result["counts"]["unsupported"] == 1
    assert result["counts"]["corrupt"] == 0
    assert result["omissions"][0]["format"] == "GIF"


@pytest.mark.parametrize("failure,status", [(PermissionError(errno.EACCES, "denied"), "access_error"),
                                             (OSError(errno.EIO, "I/O failure"), "access_error"),
                                             (KeyError("bad header metadata"), "error"),
                                             (RuntimeError("parser failure"), "error")])
def test_ordinary_per_file_errors_do_not_abort_inventory(audit, tmp_path, monkeypatch, failure, status):
    image(tmp_path / "good.jpg")
    records, split = manifests(tmp_path, [row("bad"), row("good")])
    original = audit.read_encoding_header

    def inspect(path):
        if path.name == "bad.jpg":
            raise failure
        return original(path)

    monkeypatch.setattr(audit, "read_encoding_header", inspect)
    result = audit.audit_target_headers(records, split, tmp_path, tmp_path / "out")
    assert result["counts"][status] == 1
    assert result["counts"]["ok"] == 1
    assert result["completeness"]["candidate_headers_complete"] is False


@pytest.mark.parametrize("failure", [KeyboardInterrupt, SystemExit])
def test_process_control_exceptions_are_not_swallowed(audit, tmp_path, monkeypatch, failure):
    records, split = manifests(tmp_path, [row("one")])

    def inspect(path):
        raise failure()

    monkeypatch.setattr(audit, "read_encoding_header", inspect)
    with pytest.raises(failure):
        audit.audit_target_headers(records, split, tmp_path, tmp_path / "out")


def test_mpo_jpeg_family_header_without_decoding(audit, tmp_path, monkeypatch):
    path = tmp_path / "family.mpo"
    Image.new("RGB", (7, 5)).save(path, format="MPO", save_all=True,
                                  append_images=[Image.new("RGB", (7, 5))], quality=75)
    monkeypatch.setattr(Image.Image, "load", lambda *args: pytest.fail("decoded pixels"))
    header = audit.read_encoding_header(path)
    assert header["format"] == "MPO"
    assert header["width"] == 7
    assert header["quantization_tables"]
    assert 75 in header["exact_ijg_candidates"]


@pytest.mark.parametrize("scale,expected", [(1, 50.0), (2, 25.0)])
def test_legacy_estimate_both_scaling_branches(audit, tmp_path, scale, expected):
    path = tmp_path / "scaled.jpg"
    Image.new("L", (5, 5)).save(path, format="JPEG", quality=-1,
                               qtables=[[value * scale for value in audit.STD_LUMA]])
    assert audit.read_encoding_header(path)["estimated_quality"] == expected


def test_auc_ties_order_and_strict_threshold_are_fixed(audit):
    rows = [dict(label=0, estimated_quality=85), dict(label=1, estimated_quality=85),
            dict(label=1, estimated_quality=None)]
    metrics = audit._metrics(rows)
    assert metrics["auc_negative_estimate"] == 0.5
    assert metrics["accuracy_estimate_lt_85"] == 0.5
    assert metrics["eligible"] == 2
    assert metrics["excluded"] == 1
    assert audit._metrics(list(reversed(rows))) == metrics


def test_signature_disjointness_with_equal_estimates_and_full_coverage(audit, tmp_path):
    image(tmp_path / "genuine.jpg", quality=75, subsampling=0)
    with Image.open(tmp_path / "genuine.jpg") as opened:
        tables = opened.quantization
    tables[1][0] += 1
    image(tmp_path / "altered.jpg", quality=-1, qtables=tables, subsampling=2)
    records, split = manifests(tmp_path, [row("genuine"), row("altered", 1)])
    result = audit.audit_target_headers(records, split, tmp_path, tmp_path / "out")
    group = result["groups"]["candidate_test"]
    assert group["metrics"]["auc_negative_estimate"] == 0.5
    for kind in ("quantization", "sampling"):
        association = group["signature_associations"][kind]
        assert association["full_coverage"] is True
        assert association["population_class_disjoint"] is True
    assert group["signature_associations"]["format"]["population_class_disjoint"] is False


def test_path_resolution_access_error_is_reported_without_opening(audit, tmp_path, monkeypatch):
    records, split = manifests(tmp_path, [row("bad")])
    original = Path.resolve

    def resolve(path, *args, **kwargs):
        if path.name == "bad.jpg":
            raise PermissionError(errno.EACCES, "cannot resolve header path")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", resolve)
    monkeypatch.setattr(audit, "read_encoding_header", lambda path: pytest.fail("unresolved path was opened"))
    result = audit.audit_target_headers(records, split, tmp_path, tmp_path / "out")
    assert result["counts"]["access_error"] == 1
    assert result["completeness"]["candidate_headers_complete"] is False


@pytest.mark.parametrize("include_missing", [False, True])
def test_format_disjointness_and_header_coverage(audit, tmp_path, include_missing):
    rows = [row("genuine"), row("altered", 1)]
    image(tmp_path / "genuine.jpg")
    Image.new("RGB", (2, 2)).save(tmp_path / "altered.jpg", format="PNG")
    if include_missing:
        rows.append(row("missing", 1))
    records, split = manifests(tmp_path, rows)
    result = audit.audit_target_headers(records, split, tmp_path, tmp_path / "out")
    association = result["groups"]["candidate_test"]["signature_associations"]["format"]
    assert association["eligible"] == 2
    assert association["excluded"] == int(include_missing)
    assert association["eligible_class_counts"] == {"genuine": 1, "altered": 1}
    assert association["coverage"] == 2 / len(rows)
    assert association["full_coverage"] is (not include_missing)
    assert association["class_disjoint"] is True
    assert association["population_class_disjoint"] is (None if include_missing else True)
    assert association["signatures"] == [
        {"signature": "JPEG", "genuine": 1, "altered": 0},
        {"signature": "PNG", "genuine": 0, "altered": 1},
    ]
    report = (tmp_path / "out/report.md").read_text()
    assert '"format": {' in report
    if include_missing:
        assert "INCOMPLETE" in report


def test_metadata_populated_and_distinct_counts_include_blank_optional_fields(audit):
    rows = [row("genuine", component_id="shared", capture="scan"),
            row("altered", 1, component_id="shared", capture="scan",
                source_id="genuine", donor_id="   ", attack_family="replacement"),
            row("excluded", 1, role="train", donor_id="unused")]
    metadata = audit._metadata_quality(rows, rows[:2])
    assert metadata["populated_counts"] == {
        "identity_id": 2, "source_id": 1, "donor_id": 0, "component_id": 2,
        "design_id": 2, "attack_family": 1, "capture": 2,
    }
    assert metadata["distinct_counts"] == {
        "identity_id": 2, "source_id": 1, "donor_id": 0, "component_id": 1,
        "design_id": 1, "attack_family": 1, "capture": 1,
    }
    assert not any(metadata["missing_values"].values())
    assert "donor_id" not in metadata["missing_values"]
    empty = audit._metadata_quality(rows, [])
    assert all(count == 0 for count in empty["populated_counts"].values())
    assert all(count == 0 for count in empty["distinct_counts"].values())
