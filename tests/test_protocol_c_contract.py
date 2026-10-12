from dataclasses import FrozenInstanceError
import hashlib
import importlib
import json
from pathlib import Path
import sys

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def contract_api():
    assert (ROOT / "src/refer_or_clear/protocol_c_contract.py").exists(), "C metadata authority missing"
    return importlib.import_module("refer_or_clear.protocol_c_contract")


@pytest.fixture
def metadata(contract_api, tmp_path, monkeypatch):
    records, source, target = [], [], []
    for identifier, role, identity in (("source0", "train", "sidtd/source0"), ("source1", "train", "sidtd/source1"),
                                       ("validation", "val", "sidtd/source0"), ("source-test", "test", "sidtd/test")):
        records.append(dict(image_id=identifier, dataset="sidtd", design_id="sidtd/example", identity_id=identity,
            component_id=identity, label=0, source_id="", donor_id="", path=f"sidtd/{identifier}.png",
            annotation=f"sidtd/{identifier}.json", width=96, height=64))
        source.append(dict(image_id=identifier, protocol="A", ref_group="it0", role=role,
            design_id="sidtd/example", component_id=identity, label=0))
        target.append(dict(image_id=identifier, protocol="C", subset="", role="val" if role == "test" else "train",
            design_id="sidtd/example", component_id=identity, label=0))
    for design in contract_api.TARGET_DESIGNS:
        for index in range(5):
            identity = f"{design}/{index}"
            for capture in range(2):
                for label in (0, 1):
                    identifier = f"{design.replace('/', '_')}_{index}_{capture}_{label}"
                    records.append(dict(image_id=identifier, dataset="fantasyid", design_id=design,
                        identity_id=identity, component_id=identity, label=label, source_id=identity if label else "",
                        donor_id="", path=f"{design}/{identifier}.png", annotation=f"{design}/{identifier}.json", width=96, height=64))
                    target.append(dict(image_id=identifier, protocol="C", subset="heldout", role="test",
                        design_id=design, component_id=identity, label=label))
    paths = {}
    for name, rows in (("records", records), ("source", source), ("target", target)):
        paths[name] = tmp_path / f"{name}.csv"
        pd.DataFrame(rows).to_csv(paths[name], index=False)
    monkeypatch.setattr(contract_api, "RECORDS_SHA256", sha(paths["records"]))
    monkeypatch.setattr(contract_api, "SOURCE_SPLIT_SHA256", sha(paths["source"]))
    monkeypatch.setattr(contract_api, "TARGET_SPLIT_SHA256", sha(paths["target"]))
    monkeypatch.setattr(contract_api, "SOURCE_COUNTS", {"train": 2, "val": 1, "test": 1})
    monkeypatch.setattr(contract_api, "TARGET_COUNT", 60)
    return dict(paths=paths, records=records, source=source, target=target, tmp=tmp_path)


def load(api, metadata):
    paths = metadata["paths"]
    return api.load_protocol_c_contract(paths["records"], paths["source"], paths["target"], records_sha256=sha(paths["records"]))


def test_metadata_freezes_actual_a_source_and_only_c_headline(contract_api, metadata, monkeypatch):
    original = Path.read_bytes

    def csv_only(path):
        assert path.suffix == ".csv"
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", csv_only)
    contract = load(contract_api, metadata)
    assert len(contract.source_train) == 2 and len(contract.source_val) == 1
    assert len(contract.target_test) == 60
    assert {row.image_id for row in contract.source_val} == {"validation"}
    assert contract.source_train[0].component_id == contract.source_val[0].component_id
    with pytest.raises(FrozenInstanceError):
        contract.target_test[0].label = 1


def test_component_reservation_capture_and_attack_exclusion(contract_api, metadata):
    contract = load(contract_api, metadata)
    result = contract_api.reserve_target_enrollment(contract)
    assert result == contract_api.reserve_target_enrollment(contract)
    assert result["schema_version"] == "protocol_c_target_enrollment_v1"
    assert result["selection"]["namespace"] == "protocol_c_target_support_v1"
    assert len(result["support_ids"]) == 12
    assert len(result["excluded_ids"]) == 12
    assert len(result["query_ids"]) == 36
    assert result["published_target_count"] == 60 and not result["published_test_population_preserved"]
    members = {row.image_id: row for row in contract.target_test}
    support = [members[identifier] for identifier in result["support_ids"]]
    queries = [members[identifier] for identifier in result["query_ids"]]
    assert all(row.label == 0 for row in support)
    for field in ("identity_id", "component_id"):
        assert not {getattr(row, field) for row in support} & {getattr(row, field) for row in queries}
    assert set(result["support_ids"] + result["query_ids"] + result["excluded_ids"]) == set(members)
    assert result["lineage"]["donor_unknown_altered_count"] == 30
    assert not result["lineage"]["complete_donor_disjointness_claimed"]


def test_reservation_requires_external_hash_and_exact_body(contract_api, metadata):
    contract = load(contract_api, metadata)
    document = contract_api.reserve_target_enrollment(contract)
    path = metadata["tmp"] / "target.json"
    path.write_text(json.dumps(document))
    reserved = contract_api.load_target_enrollment(contract, path, expected_sha256=sha(path))
    assert reserved.as_dict() == document
    document["query_ids"].append(document["support_ids"][0])
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError):
        contract_api.load_target_enrollment(contract, path, expected_sha256=sha(path))


@pytest.mark.parametrize("name", ["records", "source", "target"])
def test_known_byte_pins_fail_before_parse(contract_api, metadata, monkeypatch, name):
    with metadata["paths"][name].open("ab") as stream:
        stream.write(b"\n")
    monkeypatch.setattr(pd, "read_csv", lambda *args, **kwargs: pytest.fail("unapproved CSV parsed"))
    with pytest.raises(ValueError, match="hash|SHA256|pin"):
        load(contract_api, metadata)


@pytest.mark.parametrize("mutation", ["label", "component", "identity", "donor", "duplicate"])
def test_matching_hash_does_not_waive_metadata_validation(contract_api, metadata, monkeypatch, mutation):
    frame = pd.read_csv(metadata["paths"]["records"], keep_default_na=False)
    if mutation == "label":
        frame.loc[0, "label"] = 1
    elif mutation == "component":
        frame.loc[0, "component_id"] = "foreign"
    elif mutation == "identity":
        frame.loc[0, "identity_id"] = ""
    elif mutation == "donor":
        frame.loc[0, "donor_id"] = "unknown"
    else:
        frame = pd.concat([frame, frame.iloc[:1]])
    frame.to_csv(metadata["paths"]["records"], index=False)
    monkeypatch.setattr(contract_api, "RECORDS_SHA256", sha(metadata["paths"]["records"]))
    with pytest.raises(ValueError):
        load(contract_api, metadata)


def test_actual_known_metadata_snapshot(contract_api):
    contract = contract_api.load_protocol_c_contract(ROOT / "data/processed/records.csv",
        ROOT / "data/processed/splits/protocol_a_it0.csv", ROOT / "data/processed/splits/protocol_c.csv",
        records_sha256=contract_api.RECORDS_SHA256)
    assert (len(contract.source_train), len(contract.source_val), len(contract.target_test)) == (1778, 222, 599)
    result = contract_api.reserve_target_enrollment(contract)
    assert len(result["support_ids"]) == 66
    assert len(result["query_ids"]) == 467 and len(result["excluded_ids"]) == 66
    assert hashlib.sha256(json.dumps(result, sort_keys=True, separators=(",", ":")).encode()).hexdigest() == "a745dca5f1304be8a5feeeb689d8cbf53790c5bd5d1c41da93a82af7accf397a"


def test_ancestor_symlink_metadata_authority_rejected(contract_api, metadata):
    alias = metadata["tmp"] / "alias"
    alias.symlink_to(metadata["tmp"], target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        contract_api.load_protocol_c_contract(alias / "records.csv", metadata["paths"]["source"],
            metadata["paths"]["target"], records_sha256=sha(metadata["paths"]["records"]))


def test_small_population_fails_instead_of_reserving_all_queries(contract_api, metadata, monkeypatch):
    records = pd.read_csv(metadata["paths"]["records"], keep_default_na=False)
    target = pd.read_csv(metadata["paths"]["target"], keep_default_na=False)
    design = contract_api.TARGET_DESIGNS[0]
    remove = set(records.loc[(records.design_id == design) & ~records.identity_id.isin((f"{design}/0", f"{design}/1")), "image_id"])
    records = records.loc[~records.image_id.isin(remove)]
    target = target.loc[~target.image_id.isin(remove)]
    records.to_csv(metadata["paths"]["records"], index=False)
    target.to_csv(metadata["paths"]["target"], index=False)
    monkeypatch.setattr(contract_api, "RECORDS_SHA256", sha(metadata["paths"]["records"]))
    monkeypatch.setattr(contract_api, "TARGET_SPLIT_SHA256", sha(metadata["paths"]["target"]))
    monkeypatch.setattr(contract_api, "TARGET_COUNT", 48)
    with pytest.raises(ValueError, match="no remaining query"):
        contract_api.reserve_target_enrollment(load(contract_api, metadata))
