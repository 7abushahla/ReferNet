import hashlib
import importlib
from pathlib import Path
import sys

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture
def contract_api():
    assert (ROOT / "src/refer_or_clear/protocol_a_contract.py").exists(), "A contract missing"
    return importlib.import_module("refer_or_clear.protocol_a_contract")


def official(api, fold):
    return api.load_protocol_a_contract(ROOT / "data/processed/records.csv",
        ROOT / f"data/processed/splits/protocol_a_it{fold}.csv", fold=fold,
        records_sha256=sha(ROOT / "data/processed/records.csv"))


@pytest.mark.parametrize("fold", range(10))
def test_official_membership_support_and_dynamic_accounting(contract_api, fold):
    contract = official(contract_api, fold)
    expected = {4: (1776, 223, 223), 9: (1777, 222, 223)}.get(fold, (1778, 222, 222))
    assert tuple(map(len, (contract.source_train, contract.source_val, contract.target_test))) == expected
    assert contract.fold == fold and contract.ref_group == f"it{fold}"
    assert not hasattr(contract, "group")
    assert len(contract.source_designs) == 10
    assert set(row.component_id for row in contract.source_train) & set(row.component_id for row in contract.source_val)
    forbidden = {row.component_id for row in contract.source_val + contract.target_test}
    eligible = tuple(row for row in contract.source_train if row.label == 0 and row.component_id not in forbidden)
    assert eligible == contract.eligible_support
    assert len(eligible) == [590, 587, 585, 600, 587, 570, 596, 604, 610, 578][fold]
    assert list(row.image_id for row in contract.source_train) == sorted(row.image_id for row in contract.source_train)
    recipe = dict(epochs=100, batch_size=8, eval_batch_size=1, accumulation_steps=1,
        drop_last=True, eval_drop_last=False, seed=42,
        scheduler=dict(name="warmup_cosine", warmup_iters=1000, warmup_cap_fraction=0.1))
    accounting = contract_api.training_accounting(contract, recipe)
    assert accounting.n_train == expected[0] and accounting.n_val == expected[1]
    assert accounting.total_optimizer_steps == 22200
    assert accounting.omitted_per_epoch == expected[0] - 1776
    assert accounting.effective_warmup_iters == 1000
    recipe["batch_size"] = 16
    with pytest.raises(ValueError):
        contract_api.training_accounting(contract, recipe)


@pytest.mark.parametrize("fold", [True, False, -1, 10, "4", 4.0, None])
def test_invalid_fold_before_read(contract_api, fold, monkeypatch):
    monkeypatch.setattr(Path, "read_bytes", lambda *args: pytest.fail("invalid fold read metadata"))
    with pytest.raises(ValueError, match="fold"):
        contract_api.load_protocol_a_contract("absent", "absent", fold=fold, records_sha256="a" * 64)


def test_wrong_official_fold_and_records_pin_fail_closed(contract_api, tmp_path):
    with pytest.raises(ValueError, match="split.*hash"):
        contract_api.load_protocol_a_contract(ROOT / "data/processed/records.csv",
            ROOT / "data/processed/splits/protocol_a_it4.csv", fold=9, records_sha256=contract_api.RECORDS_SHA256)
    path = tmp_path / "records.csv"
    path.write_bytes((ROOT / "data/processed/records.csv").read_bytes() + b"\n")
    with pytest.raises(ValueError, match="records"):
        contract_api.load_protocol_a_contract(path, ROOT / "data/processed/splits/protocol_a_it4.csv",
            fold=4, records_sha256=sha(path))


@pytest.mark.parametrize("change", ["protocol", "ref_group", "component_id", "label", "role", "duplicate"])
def test_hash_bound_split_still_requires_semantic_join(contract_api, tmp_path, monkeypatch, change):
    split = pd.read_csv(ROOT / "data/processed/splits/protocol_a_it4.csv", keep_default_na=False)
    if change == "duplicate":
        split.loc[1, "image_id"] = split.loc[0, "image_id"]
    else:
        split.loc[0, change] = {"protocol": "B", "ref_group": "it9", "component_id": "foreign",
                              "label": 7, "role": "other"}[change]
    path = tmp_path / "split.csv"
    split.to_csv(path, index=False)
    monkeypatch.setattr(contract_api, "REVIEWED_SPLIT_SHA256", {4: sha(path)})
    with pytest.raises(ValueError):
        contract_api.load_protocol_a_contract(ROOT / "data/processed/records.csv", path, fold=4,
            records_sha256=contract_api.RECORDS_SHA256)
