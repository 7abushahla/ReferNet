import copy
from dataclasses import FrozenInstanceError
import hashlib
import importlib
import json
from pathlib import Path
import sys

import pandas as pd
import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


@pytest.fixture
def api():
    assert importlib.util.find_spec("refer_or_clear.protocol_b_contract") is not None, "Protocol B shared contract is missing"
    return importlib.import_module("refer_or_clear.protocol_b_contract")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def contract(api):
    records = ROOT / "data/processed/records.csv"
    return api.load_protocol_b_contract(records, ROOT / "data/processed/splits/protocol_b_g0.csv",
                                        group=0, records_sha256=sha(records))


@pytest.fixture
def recipe():
    return yaml.safe_load((ROOT / "configs/direct_reference_pilot.yaml").read_text())


def write_manifest(tmp_path, manifest):
    path = tmp_path / "target-enrollment.json"
    path.write_text(json.dumps(manifest, sort_keys=True))
    return path


@pytest.mark.parametrize("group", range(10))
def test_all_reviewed_splits_are_byte_bound_and_immutable(api, group):
    records = ROOT / "data/processed/records.csv"
    split = ROOT / f"data/processed/splits/protocol_b_g{group}.csv"
    result = api.load_protocol_b_contract(records, split, group=group, records_sha256=sha(records))
    assert result.split_sha256 == sha(split) == api.REVIEWED_SPLIT_SHA256[group]
    assert len(result.source_train) + len(result.source_val) + len(result.target_test) == 2222
    assert len(result.source_designs) == 6 and len(result.target_designs) == 4
    assert not set(result.source_designs) & set(result.target_designs)
    assert {row.design_id for row in result.target_test} == set(result.target_designs)
    for source in (result.source_train, result.source_val):
        assert not {row.component_id for row in source} & {row.component_id for row in result.target_test}
    with pytest.raises(FrozenInstanceError):
        result.group = 8
    with pytest.raises(FrozenInstanceError):
        result.target_test[0].label = 1
    with pytest.raises(TypeError):
        api.REVIEWED_SPLIT_SHA256[group] = "a" * 64


@pytest.mark.parametrize("group", [True, -1, 10, 0.0, "0"])
def test_invalid_group_fails(api, group):
    records = ROOT / "data/processed/records.csv"
    with pytest.raises(ValueError, match="group"):
        api.load_protocol_b_contract(records, ROOT / "data/processed/splits/protocol_b_g0.csv",
                                     group=group, records_sha256=sha(records))


def test_split_byte_mutation_fails_before_csv_parsing(api, tmp_path, monkeypatch):
    records = ROOT / "data/processed/records.csv"
    split = tmp_path / "split.csv"
    split.write_bytes((ROOT / "data/processed/splits/protocol_b_g0.csv").read_bytes() + b"\n")
    monkeypatch.setattr(pd, "read_csv", lambda *args, **kwargs: pytest.fail("unauthenticated CSV parsed"))
    with pytest.raises(ValueError, match="split.*hash"):
        api.load_protocol_b_contract(records, split, group=0, records_sha256=sha(records))


def test_records_hash_mismatch(api):
    with pytest.raises(ValueError, match="records.*hash"):
        api.load_protocol_b_contract(ROOT / "data/processed/records.csv",
                                     ROOT / "data/processed/splits/protocol_b_g0.csv", group=0,
                                     records_sha256="a" * 64)


@pytest.mark.parametrize("mutation", ["label", "component", "identity", "missing", "duplicate", "donor"])
def test_authenticated_records_still_require_matching_metadata(api, tmp_path, mutation):
    records = pd.read_csv(ROOT / "data/processed/records.csv", keep_default_na=False)
    if mutation == "label":
        records.loc[0, "label"] = 1
    elif mutation == "component":
        records.loc[0, "component_id"] = "foreign"
    elif mutation == "identity":
        records.loc[0, "identity_id"] = ""
    elif mutation == "missing":
        records = records.iloc[1:]
    elif mutation == "duplicate":
        records = pd.concat([records, records.iloc[:1]], ignore_index=True)
    else:
        records.loc[0, "donor_id"] = records.loc[150, "identity_id"]
    path = tmp_path / "records.csv"
    records.to_csv(path, index=False)
    with pytest.raises(ValueError):
        api.load_protocol_b_contract(path, ROOT / "data/processed/splits/protocol_b_g0.csv", group=0,
                                     records_sha256=sha(path))


def test_accounting_uses_group_population_and_current_recipe(api, contract, recipe):
    result = api.training_accounting(contract, recipe)
    assert result.n_train == 1201 and result.n_val == 130 and result.n_target_test == 891
    assert result.batches_per_epoch == 150 and result.images_per_epoch == 1200
    assert result.omitted_per_epoch == 1 and result.total_optimizer_steps == 15000
    assert result.effective_warmup_iters == 1000
    records = ROOT / "data/processed/records.csv"
    other = api.load_protocol_b_contract(records, ROOT / "data/processed/splits/protocol_b_g9.csv",
                                         group=9, records_sha256=sha(records))
    assert api.training_accounting(other, recipe).total_optimizer_steps == 14700
    assert api.training_accounting(other, recipe).omitted_per_epoch == 0


@pytest.mark.parametrize("field,value", [("epochs", True), ("epochs", 40), ("batch_size", 4),
                                         ("accumulation_steps", 2), ("drop_last", False),
                                         ("eval_drop_last", True), ("seed", 43)])
def test_accounting_rejects_recipe_changes(api, contract, recipe, field, value):
    recipe[field] = value
    with pytest.raises(ValueError, match="recipe"):
        api.training_accounting(contract, recipe)


def test_scheduler_reuses_existing_helper(api, contract, recipe):
    from refer_or_clear.pilots import pilot_scheduler
    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=1e-3)
    expected_optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=1e-3)
    actual = api.make_protocol_b_scheduler(optimizer, contract, recipe)
    expected = pilot_scheduler(expected_optimizer, "D_running_pilot", 15000)
    for step in (0, 1, 999, 1000, 7500, 14999, 15000):
        assert actual.lr_lambdas[0](step) == expected.lr_lambdas[0](step)


def test_explicit_manifest_is_mandatory(api, contract):
    with pytest.raises(ValueError, match="manifest"):
        api.load_target_enrollment(contract, None, expected_sha256=None)


def test_reservation_is_deterministic_disjoint_and_disclosed(api, contract, tmp_path):
    manifest = api.reserve_target_enrollment(contract)
    assert manifest == api.reserve_target_enrollment(contract)
    assert manifest["selection"]["seed"] == 42 and manifest["selection"]["fraction"] == 0.2
    assert manifest["selection"]["rounding"] == "ceil"
    path = write_manifest(tmp_path, manifest)
    result = api.load_target_enrollment(contract, path, expected_sha256=sha(path))
    assert result.as_dict() == manifest
    support = set(manifest["support_ids"])
    queries = set(manifest["query_ids"])
    excluded = set(manifest["excluded_ids"])
    assert support and queries and excluded
    assert not support & queries and not support & excluded and not queries & excluded
    assert support | queries | excluded == {row.image_id for row in contract.target_test}
    support_rows = [row for row in contract.target_test if row.image_id in support]
    query_rows = [row for row in contract.target_test if row.image_id in queries]
    assert all(row.label == 0 for row in support_rows)
    assert not {row.identity_id for row in support_rows} & {row.identity_id for row in query_rows}
    assert not {row.component_id for row in support_rows} & {row.component_id for row in query_rows}
    assert set(manifest["roles"]) == {row.image_id for row in contract.source_train + contract.source_val + contract.target_test}
    assert all(manifest["roles"][row.image_id] == "source_train" for row in contract.source_train)
    assert all(manifest["roles"][row.image_id] == "source_val" for row in contract.source_val)
    assert manifest["published_target_count"] == 891 and manifest["query_count"] == len(queries) < 891
    assert set(manifest["excluded_reasons"]) == support | excluded
    with pytest.raises(FrozenInstanceError):
        result.query_ids = ()


@pytest.mark.parametrize("field,value", [("schema_version", "legacy"), ("group", 1), ("group", False),
                                         ("records_sha256", "b" * 64), ("split_sha256", "c" * 64)])
def test_manifest_identity_rejection(api, contract, tmp_path, field, value):
    manifest = api.reserve_target_enrollment(contract)
    manifest[field] = value
    path = write_manifest(tmp_path, manifest)
    with pytest.raises(ValueError):
        api.load_target_enrollment(contract, path, expected_sha256=sha(path))


@pytest.mark.parametrize("mutation", ["duplicate", "overlap", "foreign", "unaccounted", "retuned", "reason"])
def test_manifest_cannot_change_reserved_population(api, contract, tmp_path, mutation):
    manifest = api.reserve_target_enrollment(contract)
    if mutation == "duplicate":
        manifest["support_ids"].append(manifest["support_ids"][0])
    elif mutation == "overlap":
        manifest["query_ids"].append(manifest["support_ids"][0])
    elif mutation == "foreign":
        manifest["query_ids"].append(contract.source_train[0].image_id)
    elif mutation == "unaccounted":
        manifest["excluded_ids"].pop()
    elif mutation == "retuned":
        manifest["selection"]["seed"] = 43
    else:
        manifest["excluded_reasons"][manifest["support_ids"][0]] = "unexplained"
    path = write_manifest(tmp_path, manifest)
    with pytest.raises(ValueError):
        api.load_target_enrollment(contract, path, expected_sha256=sha(path))


def test_manifest_byte_mutation_rejected(api, contract, tmp_path):
    path = write_manifest(tmp_path, api.reserve_target_enrollment(contract))
    expected = sha(path)
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="manifest.*hash"):
        api.load_target_enrollment(contract, path, expected_sha256=expected)


def synthetic_contract(api, *, linked=False, small=False):
    designs = ("sidtd/alb_id", "sidtd/aze_passport", "sidtd/lva_passport", "sidtd/svk_id")
    targets = []
    for design in designs:
        for index in range(2 if small else 10):
            identity = f"{design}/{index}"
            targets.append(api.MembershipRow(f"{design}_g{index}", identity, identity, design, 0))
            targets.append(api.MembershipRow(f"{design}_a{index}", identity, identity, design, 1))
    if linked:
        reserved = sorted({row.component_id for row in targets if row.design_id == designs[0]},
                          key=lambda component: hashlib.sha256(f"protocol_b_target_support_v1:42:{designs[0]}:{component}".encode()).hexdigest())[:2]
        targets.append(api.MembershipRow("donor-linked-attack", "different-identity", reserved[0], designs[0], 1))
    return api.ProtocolBContract(0, "svk_alb_aze_lva", "a" * 64, "b" * 64,
                                 ("source",), designs, (), (), tuple(sorted(targets, key=lambda row: row.image_id)))


def test_transitively_donor_linked_attack_is_excluded(api):
    contract = synthetic_contract(api, linked=True)
    result = api.reserve_target_enrollment(contract)
    assert "donor-linked-attack" in result["excluded_ids"]
    assert result["excluded_reasons"]["donor-linked-attack"] == "reserved_component_derived_attack"
    for design in contract.target_designs:
        support_components = {row.component_id for row in contract.target_test
                              if row.image_id in result["support_ids"] and row.design_id == design}
        assert len(support_components) == 2


def test_minimum_support_cannot_empty_remaining_design_queries(api):
    with pytest.raises(ValueError, match="remaining query"):
        api.reserve_target_enrollment(synthetic_contract(api, small=True))


def test_support_query_identity_overlap_even_with_distinct_components_fails(api):
    original = synthetic_contract(api)
    manifest = api.reserve_target_enrollment(original)
    support = next(row for row in original.target_test if row.image_id in manifest["support_ids"])
    query = next(row for row in original.target_test if row.image_id in manifest["query_ids"] and row.label == 1)
    rows = tuple(api.MembershipRow(row.image_id, support.identity_id, row.component_id, row.design_id, row.label)
                 if row.image_id == query.image_id else row for row in original.target_test)
    altered = copy.copy(original)
    object.__setattr__(altered, "target_test", rows)
    with pytest.raises(ValueError, match="identity"):
        api.reserve_target_enrollment(altered)


def test_multi_hop_donor_connectivity_excludes_all_linked_attacks(api):
    from refer_or_clear.data.records import provenance_components

    original = synthetic_contract(api)
    manifest = api.reserve_target_enrollment(original)
    support = next(row for row in original.target_test if row.image_id in manifest["support_ids"])
    records = [dict(image_id=row.image_id, identity_id=row.identity_id, component_id=row.component_id,
                    design_id=row.design_id, label=row.label, source_id="", donor_id="")
               for row in original.target_test]
    records.extend([
        dict(image_id="bridge", identity_id=support.identity_id, design_id=support.design_id,
             label=1, source_id=support.identity_id, donor_id="middle"),
        dict(image_id="middle-attack", identity_id="middle", design_id=support.design_id,
             label=1, source_id="middle", donor_id="leaf"),
        dict(image_id="leaf-attack", identity_id="leaf", design_id=support.design_id,
             label=1, source_id="leaf", donor_id=""),
    ])
    frame = pd.DataFrame(records)
    frame["component_id"] = provenance_components(frame)
    rows = tuple(api.MembershipRow(row.image_id, row.identity_id, row.component_id, row.design_id, row.label)
                 for row in frame.sort_values("image_id").itertuples())
    altered = copy.copy(original)
    object.__setattr__(altered, "target_test", rows)
    result = api.reserve_target_enrollment(altered)
    assert {"bridge", "middle-attack", "leaf-attack"} <= set(result["excluded_ids"])
    assert not {"bridge", "middle-attack", "leaf-attack"} & set(result["query_ids"])


def test_contract_and_reservation_read_only_metadata(api, tmp_path, monkeypatch):
    records = ROOT / "data/processed/records.csv"
    split = ROOT / "data/processed/splits/protocol_b_g0.csv"
    expected = sha(records)
    original_read = Path.read_bytes
    opened = []

    def read_metadata(path):
        assert path in (records, split, tmp_path / "target-enrollment.json"), "unexpected non-metadata read"
        opened.append(path)
        return original_read(path)

    monkeypatch.setattr(Path, "read_bytes", read_metadata)
    contract = api.load_protocol_b_contract(records, split, group=0, records_sha256=expected)
    manifest = api.reserve_target_enrollment(contract)
    path = write_manifest(tmp_path, manifest)
    api.load_target_enrollment(contract, path, expected_sha256=sha(path))
    assert set(opened) == {records, split, path}


@pytest.mark.parametrize("field,value", [("warmup_iters", True), ("warmup_iters", 999),
                                         ("warmup_cap_fraction", 0.2), ("name", "different")])
def test_scheduler_recipe_cannot_silently_drift(api, contract, recipe, field, value):
    recipe["scheduler"][field] = value
    with pytest.raises(ValueError, match="recipe"):
        api.training_accounting(contract, recipe)


@pytest.mark.parametrize("field", ["source_id", "donor_id"])
def test_unknown_nonempty_provenance_tokens_fail_even_with_fresh_records_hash(api, tmp_path, field):
    records = pd.read_csv(ROOT / "data/processed/records.csv", keep_default_na=False)
    records.loc[0, field] = "sidtd_alb_id_00_fake_6_25unknownidentity"
    path = tmp_path / "records.csv"
    records.to_csv(path, index=False)
    with pytest.raises(ValueError, match="unknown.*identity"):
        api.load_protocol_b_contract(path, ROOT / "data/processed/splits/protocol_b_g0.csv", group=0,
                                     records_sha256=sha(path))
