import csv
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/audit_transfer_membership.py"


@pytest.fixture
def audit():
    assert SCRIPT.exists(), "metadata transfer audit is not implemented"
    spec = importlib.util.spec_from_file_location("transfer_membership_audit", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


@pytest.fixture
def membership(tmp_path):
    records = [
        dict(image_id="source", dataset="sidtd", design_id="sidtd/design", label="0",
             component_id="sidtd/source", identity_id="sidtd/source", source_id="", donor_id="", path="never-open.jpg"),
        dict(image_id="validation", dataset="sidtd", design_id="sidtd/design", label="1",
             component_id="sidtd/validation", identity_id="sidtd/validation", source_id="sidtd/validation", donor_id="", path="never-open.jpg"),
        dict(image_id="support", dataset="fantasyid", design_id="fantasyid/seen", label="0",
             component_id="shared", identity_id="shared", source_id="", donor_id="", path="never-open.jpg"),
        dict(image_id="attack", dataset="fantasyid", design_id="fantasyid/seen", label="1",
             component_id="shared", identity_id="shared", source_id="shared", donor_id="", path="never-open.jpg"),
        dict(image_id="heldout", dataset="fantasyid", design_id="fantasyid/unseen", label="0",
             component_id="unseen", identity_id="unseen", source_id="", donor_id="", path="never-open.jpg"),
    ]
    roles = ["train", "val", "reference_only", "test", "test"]
    split = [dict(image_id=row["image_id"], protocol="C", role=role,
                  subset="attack1" if row["image_id"] == "attack" else "heldout" if role == "test" else "",
                  component_id=row["component_id"], design_id=row["design_id"], label=row["label"])
             for row, role in zip(records, roles)]
    return dict(records=write_csv(tmp_path / "records.csv", records),
                split=write_csv(tmp_path / "protocol_c.csv", split),
                record_rows=records, split_rows=split, d_split=tmp_path / "protocol_d.csv")


def test_freezes_existing_roles_and_reports_gaps_without_adopting_support(audit, membership):
    result = audit.audit_membership(membership["records"], membership["split"], d_split=membership["d_split"])
    assert result["schema_version"] == "transfer_membership_readiness_v1"
    assert result["source"]["datasets"] == {"sidtd": 1}
    assert result["membership"]["reference_only"] == ["support"]
    assert result["support"]["missing_designs_by_subset"] == {"attack1": [], "heldout": ["fantasyid/unseen"]}
    assert result["support"]["shared_component_ids"] == ["shared"]
    assert result["support"]["shared_identity_ids"] == ["shared"]
    assert result["support"]["per_query_exclusions"]["attack"]["remaining_declared_candidates"] == 0
    assert result["d"]["split_status"] == "absent"
    assert result["d"]["idnet_record_count"] == 0
    assert result["pixel_reads"] == 0
    assert result["support_policy"] == "approved_reservation_not_applied_existing_roles_only"
    assert result["lineage_metadata"]["recorded_token_components_consistent"] is True


@pytest.mark.parametrize("shared", ["identity_id", "component_id"])
def test_identity_and_component_exclusions_are_both_reported(audit, membership, shared):
    membership["record_rows"][3]["identity_id"] = "other"
    membership["record_rows"][3]["component_id"] = "other"
    membership["record_rows"][3][shared] = "shared"
    membership["split_rows"][3]["component_id"] = membership["record_rows"][3]["component_id"]
    write_csv(membership["records"], membership["record_rows"])
    write_csv(membership["split"], membership["split_rows"])
    result = audit.audit_membership(membership["records"], membership["split"])
    excluded = result["support"]["per_query_exclusions"]["attack"]
    assert excluded["excluded_union"] == 1
    assert excluded["identity_matches"] == (1 if shared == "identity_id" else 0)
    assert excluded["component_matches"] == (1 if shared == "component_id" else 0)
    assert result["d"]["split_status"] == "not_inspected"


@pytest.mark.parametrize("mutation", ["duplicate-record", "duplicate-split", "foreign-id", "label", "component", "protocol", "role", "altered-support"])
def test_invalid_membership_is_rejected(audit, membership, mutation):
    if mutation == "duplicate-record": membership["record_rows"].append(membership["record_rows"][0])
    elif mutation == "duplicate-split": membership["split_rows"].append(membership["split_rows"][0])
    elif mutation == "foreign-id": membership["split_rows"][0]["image_id"] = "foreign"
    elif mutation == "label": membership["split_rows"][0]["label"] = "1"
    elif mutation == "component": membership["split_rows"][0]["component_id"] = "foreign"
    elif mutation == "protocol": membership["split_rows"][0]["protocol"] = "D"
    elif mutation == "role": membership["split_rows"][0]["role"] = "enrollment"
    else:
        membership["record_rows"][2]["label"] = "1"
        membership["split_rows"][2]["label"] = "1"
    write_csv(membership["records"], membership["record_rows"])
    write_csv(membership["split"], membership["split_rows"])
    with pytest.raises(ValueError): audit.audit_membership(membership["records"], membership["split"])


def test_reads_only_csv_bytes_and_enforces_optional_pinned_hashes(audit, membership, monkeypatch):
    allowed = {membership["records"], membership["split"]}
    original = Path.read_bytes
    def guarded(path):
        assert path in allowed, "nonmetadata read"
        return original(path)
    monkeypatch.setattr(Path, "read_bytes", guarded)
    result = audit.audit_membership(membership["records"], membership["split"],
                                   records_sha256=hashlib.sha256(original(membership["records"])).hexdigest())
    assert result["inputs"]["records"]["sha256"] == hashlib.sha256(original(membership["records"])).hexdigest()
    with pytest.raises(ValueError, match="SHA256"):
        audit.audit_membership(membership["records"], membership["split"], split_sha256="0" * 64)


def test_cli_emits_json_and_does_not_create_output_files(audit, membership):
    before = set(membership["records"].parent.iterdir())
    process = subprocess.run([sys.executable, "-B", str(SCRIPT), "--records", str(membership["records"]),
                              "--split", str(membership["split"]), "--d-split", str(membership["d_split"])],
                             capture_output=True, text=True)
    assert process.returncode == 0, process.stderr
    assert json.loads(process.stdout)["d"]["split_status"] == "absent"
    assert set(membership["records"].parent.iterdir()) == before


def test_existing_d_split_is_presence_only_not_an_invented_protocol(audit, membership):
    membership["d_split"].write_text("unreviewed remote placeholder\n")
    result = audit.audit_membership(membership["records"], membership["split"], d_split=membership["d_split"])
    assert result["d"]["split_status"] == "present_unreviewed"
    assert result["d"]["ready"] is False


def test_lineage_conflicts_and_missing_donor_evidence_are_not_hidden(audit, membership):
    membership["record_rows"][3]["component_id"] = "different-component"
    membership["split_rows"][3]["component_id"] = "different-component"
    write_csv(membership["records"], membership["record_rows"])
    write_csv(membership["split"], membership["split_rows"])
    result = audit.audit_membership(membership["records"], membership["split"])
    assert result["lineage_metadata"]["recorded_token_components_consistent"] is False
    assert result["lineage_metadata"]["token_component_conflicts"]
    assert result["lineage_metadata"]["fantasyid"]["populated"]["donor_id"] == 0
    assert result["lineage_metadata"]["complete_donor_provenance_proven"] is False
