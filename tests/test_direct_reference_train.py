import copy
import csv
from contextlib import contextmanager
import importlib
import json
from pathlib import Path
import sys

import pandas as pd
import pytest
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from test_direct_reference_diagnostic import payload, write_receipt
from test_direct_reference_pilot import SyntheticClassifier
from refer_or_clear import direct_reference_diagnostic as diagnostic
from refer_or_clear.data import direct_reference as bridge
from refer_or_clear.pilots import json_sha256, tensor_sha256


@pytest.fixture
def service():
    assert importlib.util.find_spec("refer_or_clear.direct_reference_train") is not None, "full training service missing"
    return importlib.import_module("refer_or_clear.direct_reference_train")


@pytest.fixture
def setup(service, tmp_path, monkeypatch):
    config = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs/direct_reference_pilot.yaml").read_text())
    identities = dict(source_sha256="a"*64, recipe_sha256=json_sha256(config), records_sha256="c"*64,
                      split_sha256="d"*64, enrollment_manifest_sha256=bridge.ENROLLMENT_SHA256)
    context = dict(config=config, identities=identities, weights_sha256=bridge.WEIGHTS_SHA256,
                   device=torch.device("cpu"), run_dir=tmp_path, request=dict(path=str(tmp_path / "request.json"), sha256="e"*64),
                   manifest=dict(authorities={}, paths=dict(raw_root=str(tmp_path / "raw"))))
    context["manifest"]["authorities"] = {name: write_receipt(tmp_path / f"{name}.json", {})
                                            for name in ("records", "split", "enrollment", "weights")}
    context["manifest"]["authorities"]["weights"]["sha256"] = bridge.WEIGHTS_SHA256
    context["manifest"]["product_root_authority"] = write_receipt(tmp_path / "root.json", {})
    rows = {role: pd.DataFrame([dict(image_id=f"{role}{index:04d}", role=role, path=f"{role}{index:04d}.png",
            design_id=f"design{index % 10}", label=index % 2, attack_family="text_copy" if index % 2 else "",
            component_id=f"component{index}") for index in range(count)]) for role,count in (("train",1778),("val",222))}
    events = []
    monkeypatch.setattr(service, "load_context", lambda *args: copy.deepcopy(context))
    monkeypatch.setattr(service, "load_role_rows", lambda records, split, role: rows[role].copy())
    training, smoke = {}, {}
    for arm, enabled in (("off",False),("on",True)):
        receipt = payload(context, enabled)
        model = SyntheticClassifier(enabled)
        model.initialization_provenance["base"]["checkpoint_path"] = context["manifest"]["authorities"]["weights"]["path"]
        receipt["initialization"] = dict(state_sha256=tensor_sha256(model.state_dict()),
                                          initialization_sha256=json_sha256(model.initialization_provenance),
                                          provenance=copy.deepcopy(model.initialization_provenance))
        training[arm] = write_receipt(tmp_path / f"training-{arm}.json", receipt)
        smoke[arm] = write_receipt(tmp_path / f"smoke-{arm}.json", payload(context, enabled, True))
    gate = dict(schema_version="direct_reference_diagnostic_v1", diagnostic_only=True, passed=True,
                no_test_open=True, identities=identities, training_gates=training, smoke_gates=smoke)
    gate_binding = write_receipt(tmp_path / "gate.json", gate)
    audit = {}
    @contextmanager
    def pixels(context, allowed):
        audit["opened"] = []
        yield audit["opened"]
    monkeypatch.setattr(service,"pixel_audit",pixels)
    def build(selected, raw, enrollment, *, preprocess):
        events.append("registration")
        audit["opened"].extend(dict(image_id=row.image_id,role=row.role,path=str(raw / row.path)) for row in selected.itertuples())
        assert selected.role.tolist() == ["train"]*1778 + ["val"]*222
        phase = bridge._json(preprocess["phase_authority"])
        assert phase["phase"] == "train"
        assert phase["training_gates"] == training and phase["smoke_gates"] == smoke
        return dict(entries={identifier: {} for identifier in selected.image_id}, preprocess=preprocess)
    def load(path, sha, *, rows, enrollment_manifest, preprocess):
        value = bridge._json(dict(path=str(path), sha256=sha))
        assert set(value["entries"]) == set(rows.image_id)
        return value
    monkeypatch.setattr(bridge, "build_registration_table", build)
    monkeypatch.setattr(bridge, "load_registration_table", load)
    def dataset(context, prepared, selected, enabled, augmented=False):
        events.append(("dataset",enabled,selected.role.iloc[0],len(selected),augmented))
        return list(range(len(selected)))
    monkeypatch.setattr(service, "_dataset", dataset)
    models = []
    def model(context, enabled):
        events.append(("fresh-model",enabled))
        value = SyntheticClassifier(enabled)
        value.initialization_provenance["base"]["checkpoint_path"] = context["manifest"]["authorities"]["weights"]["path"]
        models.append(value)
        return value
    monkeypatch.setattr(service, "_model", model)
    updates, orders, summaries = [], [], []
    def train(model, loader, optimizer, scheduler, device):
        assert loader.batch_size == 8 and loader.drop_last and len(loader) == 222
        order = list(loader.sampler)
        assert sorted(order) == list(range(1778))
        orders.append(order)
        for unused in range(len(loader)):
            optimizer.step()
            scheduler.step()
        updates.append(scheduler.last_epoch)
        model.encoder.data.add_(.001)
        return dict(loss=.2,image_loss=.2,n_images=1776,optimizer_steps=222,bce_map=None,dice_map=None)
    monkeypatch.setattr(service, "train_direct_reference_epoch", train)
    def evaluate(model, loader, device):
        assert loader.batch_size == 1 and not loader.drop_last and len(loader) == 222
        epoch = len(updates)
        summary = dict(n_images=222,accuracy=.8 if epoch == 1 else .85,auc=.9 if epoch < 3 else .91,image_loss=.2)
        summaries.append(summary)
        predictions = [dict(image_id=row.image_id,design_id=row.design_id,attack_family=row.attack_family,label=row.label,
                             attack_probability=.8 if row.label else .2,bonafide_score=.2 if row.label else .8)
                       for row in rows["val"].itertuples()]
        return summary, predictions
    monkeypatch.setattr(service, "evaluate_direct_reference", evaluate)
    def replay(path, model, loader, device, *, config, identities, expected_predictions):
        from refer_or_clear.direct_reference_pilot import read_direct_reference_checkpoint
        checkpoint = read_direct_reference_checkpoint(path, config, identities)
        assert checkpoint["phase"] == "train" and checkpoint["epoch"] == 3
        assert tensor_sha256(model.state_dict()) == json.loads(Path(training["on" if model.reference_enabled else "off"]["path"]).read_text())["initialization"]["state_sha256"]
        assert len(loader) == len(expected_predictions) == 222
        model.load_state_dict(checkpoint["state_dict"])
        return summaries[2], expected_predictions, dict(passed=True)
    monkeypatch.setattr(service, "replay_direct_reference_checkpoint", replay)
    return dict(context=context,gate=gate,gate_binding=gate_binding,rows=rows,events=events,models=models,
                updates=updates,orders=orders,tmp=tmp_path)


@pytest.mark.parametrize("enabled", [False, True])
def test_full_recipe_selection_replay_and_fresh_state(service, setup, enabled):
    result = service.run_training(setup["context"], enabled, setup["gate_binding"])
    assert result["epochs_completed"] == 100 and result["selected_epoch"] == 3
    assert result["optimizer_steps"] == 22200 and setup["updates"][-1] == 22200
    assert result["scheduler"] == dict(total_updates=22200,warmup_updates=1000)
    assert result["role"] == "val" and result["replay"]["passed"]
    assert result["replay"]["fresh_process"] is False
    assert len(result["omitted_train_ids"]) == 100
    for epoch, omission in enumerate(result["omitted_train_ids"]):
        assert omission["image_ids"] == setup["rows"]["train"].iloc[setup["orders"][epoch][1776:]].image_id.tolist()
    assert len(set(map(tuple,setup["orders"]))) > 1
    assert len(setup["models"]) == 2 and setup["models"][0] is not setup["models"][1]
    assert setup["events"].count("registration") == 1
    with Path(result["predictions"]["path"]).open(newline="") as stream:
        assert len(list(csv.DictReader(stream))) == 222


@pytest.mark.parametrize("failure", ["bundle-hash","failed","old-source","missing-on","training-failed","smoke-not-fresh","smoke-hash","context-drift"])
def test_gates_and_source_fail_before_registration_dataset_or_model(service, setup, failure):
    binding = setup["gate_binding"]
    gate = copy.deepcopy(setup["gate"])
    if failure == "bundle-hash": binding = dict(binding,sha256="0"*64)
    elif failure == "context-drift": setup["context"]["identities"]["source_sha256"] = "f"*64
    else:
        if failure == "failed": gate["passed"] = False
        if failure == "old-source": gate["identities"]["source_sha256"] = "f"*64
        if failure == "missing-on": del gate["training_gates"]["on"]
        if failure in ("training-failed","smoke-not-fresh","smoke-hash"):
            key = "training_gates" if failure == "training-failed" else "smoke_gates"
            child = gate[key]["on"]
            receipt = json.loads(Path(child["path"]).read_text())
            if failure == "training-failed": receipt["passed"] = False
            if failure == "smoke-not-fresh": receipt["replay"]["fresh_process"] = False
            gate[key]["on"] = write_receipt(Path(child["path"]),receipt)
            if failure == "smoke-hash": gate[key]["on"]["sha256"] = "0"*64
        binding = write_receipt(Path(binding["path"]),gate)
    with pytest.raises(ValueError): service.run_training(setup["context"], False, binding)
    assert not setup["events"]


@pytest.mark.parametrize("role,count", [("train",1777),("val",221)])
def test_full_membership_required_before_pixels(service, setup, role, count):
    setup["rows"][role] = setup["rows"][role].iloc[:count]
    with pytest.raises(ValueError): service.run_training(setup["context"], False,setup["gate_binding"])
    assert not setup["events"]


def test_shared_full_table_reused_by_both_arms(service, setup):
    prepared = service.prepare_training(setup["context"],setup["gate_binding"])
    assert setup["events"] == ["registration"]
    arm_orders = []
    for enabled in (False, True):
        setup["updates"].clear()
        setup["orders"].clear()
        service.run_training(setup["context"],enabled,setup["gate_binding"],prepared=prepared)
        arm_orders.append(copy.deepcopy(setup["orders"]))
    assert setup["events"].count("registration") == 1
    assert arm_orders[0] == arm_orders[1]


def test_diagnostic_subset_table_cannot_be_reused_for_full_train(service, setup):
    prepared = service.prepare_training(setup["context"],setup["gate_binding"])
    table = bridge._json(prepared["table_binding"])
    table["entries"].pop(next(iter(table["entries"])))
    prepared["table_binding"] = write_receipt(setup["tmp"] / "diagnostic-subset.json",table)
    setup["events"].clear()
    with pytest.raises(ValueError): service.run_training(setup["context"],False,setup["gate_binding"],prepared=prepared)
    assert not setup["events"]


@pytest.mark.parametrize("key,value", [("epochs",1),("batch_size",7),("drop_last",False),("seed",43),
                                      ("eval_batch_size",2),("mask_weight",1.0),("accumulation_steps",2)])
def test_noncanonical_recipe_rejected_before_pixels(service, setup, key, value):
    setup["context"]["config"][key] = value
    with pytest.raises(ValueError): service.run_training(setup["context"],False,setup["gate_binding"])
    assert not setup["events"]


def test_no_diagnostic_warmstart_or_epoch_override_interface(service, setup):
    for extra in (dict(checkpoint="smoke.pt"),dict(epochs=1),dict(resume="last.pt")):
        with pytest.raises(TypeError): service.run_training(setup["context"],False,setup["gate_binding"],**extra)
    assert not setup["events"]


@pytest.mark.parametrize("enabled", [False, True])
def test_real_core_first_epoch_and_actual_scheduler_with_synthetic_tensors(service, setup, monkeypatch, enabled):
    from refer_or_clear.direct_reference_pilot import train_direct_reference_epoch
    class Samples:
        def __init__(self, rows, enabled):
            self.rows,self.enabled = rows,enabled
        def __len__(self): return len(self.rows)
        def __getitem__(self,index):
            row = self.rows.iloc[index]
            query = torch.zeros(4,32,32)
            query[0].fill_(1. if row.label else -1.)
            result = dict(query=query,overlap=torch.ones(1,32,32),context_available=torch.tensor(True),label=int(row.label),
                          meta=dict(image_id=row.image_id,design_id=row.design_id,attack_family=row.attack_family))
            if self.enabled: result["reference"] = torch.zeros(3,32,32)
            return result
    monkeypatch.setattr(service,"_dataset",lambda context,prepared,rows,enabled,augmented=False: Samples(rows,enabled))
    stub = service.train_direct_reference_epoch
    def train(model,loader,optimizer,scheduler,device):
        if not setup["updates"]:
            assert not optimizer.state
            assert optimizer.defaults["weight_decay"] == .05
            assert optimizer.defaults["betas"] == (.9,.999) and optimizer.defaults["eps"] == 1e-8
            assert [group["initial_lr"] for group in optimizer.param_groups] == [1e-4,1e-3]
            stats = train_direct_reference_epoch(model,loader,optimizer,scheduler,device)
            assert optimizer.state and stats["optimizer_steps"] == 222 and stats["n_images"] == 1776
            setup["orders"].append(list(loader.sampler))
            setup["updates"].append(scheduler.last_epoch)
            return stats
        return stub(model,loader,optimizer,scheduler,device)
    monkeypatch.setattr(service,"train_direct_reference_epoch",train)
    assert service.run_training(setup["context"],enabled,setup["gate_binding"])["optimizer_steps"] == 22200


@pytest.mark.parametrize("failure", ["missing","duplicate","test"])
def test_registration_query_read_ledger_must_cover_only_full_membership(service, setup, monkeypatch, failure):
    original = service.pixel_audit
    @contextmanager
    def corrupt(context,rows):
        with original(context,rows) as opened:
            yield opened
        if failure == "missing": opened.pop()
        if failure == "duplicate": opened[-1] = opened[0]
        if failure == "test": opened[-1]["role"] = "test"
    monkeypatch.setattr(service,"pixel_audit",corrupt)
    with pytest.raises(ValueError): service.prepare_training(setup["context"],setup["gate_binding"])


def test_selected_replay_rejects_smoke_phase_before_loading_model_state(service, setup, monkeypatch):
    original = service.save_direct_reference_checkpoint
    def save(path,model,config,identities,epoch,key,phase):
        return original(path,model,config,identities,epoch,key,"smoke" if path.name == "best.pt" else phase)
    monkeypatch.setattr(service,"save_direct_reference_checkpoint",save)
    with pytest.raises(ValueError,match="selected training checkpoint"):
        service.run_training(setup["context"],False,setup["gate_binding"])
    assert len(setup["models"]) == 1


@pytest.mark.parametrize("enabled", [False,True])
def test_fresh_model_allows_only_authenticated_allocation_location_change(service, setup, enabled):
    arm = "on" if enabled else "off"
    binding = setup["gate"]["training_gates"][arm]
    receipt = json.loads(Path(binding["path"]).read_text())
    initial = receipt["initialization"]
    initial["provenance"]["base"]["checkpoint_path"] = str(setup["tmp"] / "diagnostic-11/inputs/weights.pth")
    initial["initialization_sha256"] = json_sha256(initial["provenance"])
    setup["gate"]["training_gates"][arm] = write_receipt(Path(binding["path"]),receipt)
    before = copy.deepcopy(receipt)
    model = service._fresh_model(setup["context"],enabled,setup["gate"])
    assert model.initialization_provenance["base"]["checkpoint_path"] == setup["context"]["manifest"]["authorities"]["weights"]["path"]
    assert bridge._json(setup["gate"]["training_gates"][arm]) == before


@pytest.mark.parametrize("mutation", ["legacy","payload-hash","state","weight","head-seed","adapter-seed","bn-policy","arm","current-path"])
def test_location_normalization_preserves_all_other_guards(service, setup, monkeypatch, mutation):
    binding = setup["gate"]["training_gates"]["off"]
    receipt = json.loads(Path(binding["path"]).read_text())
    initial = receipt["initialization"]
    if mutation == "legacy": del initial["provenance"]
    elif mutation == "payload-hash": initial["provenance"]["base"]["checkpoint_path"] = "/different/weights.pth"
    elif mutation == "state": initial["state_sha256"] = "0"*64
    elif mutation != "current-path":
        prior = initial["provenance"]
        if mutation == "weight": prior["base"]["checkpoint_sha256"] = "0"*64
        if mutation == "head-seed": prior["base"]["head_seed"] = 43
        if mutation == "adapter-seed": prior["adapter_seed"] = 43
        if mutation == "bn-policy": prior["base"]["bn_policy"] = "different"
        if mutation == "arm": prior["reference_enabled"] = True
        initial["initialization_sha256"] = json_sha256(prior)
    else:
        original = service._model
        def wrong_path(context,enabled):
            model = original(context,enabled)
            model.initialization_provenance["base"]["checkpoint_path"] = "/unbound/weights.pth"
            return model
        monkeypatch.setattr(service,"_model",wrong_path)
    setup["gate"]["training_gates"]["off"] = write_receipt(Path(binding["path"]),receipt)
    with pytest.raises(ValueError): service._fresh_model(setup["context"],False,setup["gate"])


@pytest.mark.parametrize("mutation", ["legacy","payload-hash"])
def test_initialization_payload_authenticated_for_both_arms_before_pixels(service, setup, monkeypatch, mutation):
    gate = copy.deepcopy(setup["gate"])
    binding = gate["training_gates"]["on"]
    child = bridge._json(binding)
    if mutation == "legacy": del child["initialization"]["provenance"]
    else: child["initialization"]["provenance"]["base"]["checkpoint_path"] = "/altered/weights.pth"
    gate["training_gates"]["on"] = write_receipt(Path(binding["path"]),child)
    gate_binding = write_receipt(Path(setup["gate_binding"]["path"]),gate)
    monkeypatch.setattr(bridge,"build_registration_table",lambda *args,**kwargs: pytest.fail("registration pixels before provenance authentication"))
    with pytest.raises(ValueError): service.run_training(setup["context"],False,gate_binding)
    assert not setup["events"]
