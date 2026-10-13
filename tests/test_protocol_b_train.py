import copy
import csv
import hashlib
import importlib
import json
from pathlib import Path
import sys

import pytest
import torch
import yaml

from test_direct_reference_pilot import SyntheticClassifier
from test_protocol_b_receipts import make_gate_bundle


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from refer_or_clear.data import protocol_b_reference as bridge
from refer_or_clear.pilots import json_sha256, tensor_sha256
from refer_or_clear.protocol_b_contract import MembershipRow, ProtocolBContract


def write(path, document):
    path.write_text(json.dumps(document, sort_keys=True))
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def test_module_exists():
    assert (ROOT / "src/refer_or_clear/protocol_b_train.py").is_file(), "Protocol B training service missing"


@pytest.fixture
def service():
    return importlib.import_module("refer_or_clear.protocol_b_train")


@pytest.fixture
def setup(service, tmp_path, monkeypatch):
    config = yaml.safe_load((ROOT / "configs/direct_reference_pilot.yaml").read_text())
    bindings = [write(tmp_path / f"{name}.json", {"synthetic": name})
                for name in ("records", "split", "target", "enrollment")]
    rows = {}
    for role, count in (("train", 35), ("val", 33), ("test", 4)):
        rows[role] = [dict(image_id=f"{role}{index:03d}", identity_id=f"identity-{role}{index}",
                           component_id=f"component-{role}{index}", design_id="design",
                           label=index % 2, attack_family="text_copy" if index % 2 else "",
                           path=f"{role}{index}.png", role=role) for index in range(count)]
    members = {role: tuple(MembershipRow(**{key: row[key] for key in
               ("image_id", "identity_id", "component_id", "design_id", "label")})
               for row in selected) for role, selected in rows.items()}
    contract = ProtocolBContract(3, "synthetic-group", bindings[0]["sha256"], bindings[1]["sha256"],
                                ("design",), ("target-design",), members["train"], members["val"], members["test"])
    metadata = dict(protocol="B", group=3, records_sha256=contract.records_sha256,
                    split_sha256=contract.split_sha256, target_enrollment_sha256=bindings[2]["sha256"],
                    enrollment_manifest_sha256=bindings[3]["sha256"])
    context = bridge.ReferenceContext(contract, None, tmp_path / "raw",
        json.dumps(dict(output_root=str(tmp_path / "banks"))), json.dumps(rows),
        json.dumps(bindings), json.dumps(metadata), False)
    weights_path = tmp_path / "synthetic-weights.bin"
    weights_path.write_bytes(b"mock official initialization")
    weights = dict(path=str(weights_path), sha256=config["weights_sha256"])
    output = tmp_path / "output"
    output.mkdir()
    registration = write(output / "registration.json", {"synthetic": "train-val"})
    initial = SyntheticClassifier(False)
    initial.initialization_provenance["base"]["checkpoint_path"] = str(weights_path)
    bundle = make_gate_bundle(output, context, config, weights, registration,
        fresh_state=tensor_sha256(initial.state_dict()), provenance=initial.initialization_provenance)
    gate, gate_binding = bundle["gate"], bundle["gate_binding"]
    from refer_or_clear import protocol_b_receipts
    monkeypatch.setattr(protocol_b_receipts, "read_direct_reference_checkpoint",
                        lambda path, config, ids: copy.deepcopy(bundle["headers"][str(path)]))
    events, models, orders, updates = [], [], [], []
    original_bytes = service._bytes

    def authenticated(binding):
        if binding == weights:
            events.append("weights-authenticated")
            return weights_path.read_bytes()
        return original_bytes(binding)

    monkeypatch.setattr(service, "_bytes", authenticated)

    def load(context, binding, *, roles):
        events.append("registration")
        assert roles == ["train", "val"]
        assert binding == registration
        return object()

    monkeypatch.setattr(service, "load_protocol_b_registration", load)

    def dataset(context, table, *, role, reference_enabled, augment, cache_bytes):
        events.append(("dataset", role, reference_enabled, copy.deepcopy(augment), cache_bytes))
        assert role != "test"
        return list(range(len(context.rows(role))))

    monkeypatch.setattr(service, "ProtocolBReferenceDataset", dataset)

    def fresh(path, sha, *, reference_enabled, head_seed, adapter_seed, local_enabled=True, reference_bn="shared", fused_query_weight=1.):
        events.append("fresh-model")
        assert Path(path) == weights_path and sha == weights["sha256"]
        assert head_seed == adapter_seed == 42
        model = SyntheticClassifier(reference_enabled)
        model.initialization_provenance["base"]["checkpoint_path"] = str(path)
        models.append(model)
        return model

    monkeypatch.setattr(service, "build_direct_reference_model", fresh)

    def train(model, loader, optimizer, scheduler, device, auxiliary_query_weight=0.):
        assert device == "cpu" and loader.batch_size == 8 and loader.drop_last
        assert loader.num_workers == 0 and len(loader) == 4
        order = list(loader.sampler)
        assert sorted(order) == list(range(35))
        orders.append(order)
        for unused in range(len(loader)):
            optimizer.step()
            scheduler.step()
        updates.append(scheduler.last_epoch)
        model.encoder.data.add_(.001)
        return dict(loss=.2, image_loss=.2, n_images=32, optimizer_steps=4, bce_map=None, dice_map=None)

    monkeypatch.setattr(service, "train_direct_reference_epoch", train)

    def evaluate(model, loader, device):
        assert loader.batch_size == 1 and loader.num_workers == 0 and not loader.drop_last
        assert len(loader) == 33
        epoch = len(orders)
        summary = dict(n_images=33, accuracy=.8 if epoch == 1 else .9,
                       auc=.85 if epoch < 3 else .95, image_loss=.2, buffers_unchanged=True)
        predictions = [dict(image_id=row["image_id"], design_id=row["design_id"], label=row["label"],
                       attack_family=row["attack_family"], attack_probability=.2 + .6 * row["label"],
                       attack_logit=float(row["label"]), bonafide_score=.8 - .6 * row["label"])
                       for row in rows["val"]]
        return summary, predictions

    monkeypatch.setattr(service, "evaluate_direct_reference", evaluate)

    def replay(path, model, loader, device, *, config, identities, expected_predictions):
        checkpoint = service.read_direct_reference_checkpoint(path, config, identities)
        assert checkpoint["epoch"] == 3
        assert tensor_sha256(model.state_dict()) == tensor_sha256(SyntheticClassifier(model.reference_enabled).state_dict())
        assert model is not models[0]
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        summary = dict(n_images=33, accuracy=.9, auc=.95, image_loss=.2, buffers_unchanged=True)
        return summary, expected_predictions, dict(passed=True)

    monkeypatch.setattr(service, "replay_direct_reference_checkpoint", replay)
    return dict(context=context, config=config, weights=weights, registration=registration,
                gate=gate, gate_binding=gate_binding, rows=rows, events=events, models=models,
                orders=orders, updates=updates, output=output, tmp=tmp_path, bundle=bundle)


def run(service, setup, enabled=False, **changes):
    arguments = dict(reference_enabled=enabled, source_sha256="a" * 64,
                     gate_binding=setup["gate_binding"], device="cpu", cache_bytes=128)
    arguments.update(changes)
    return service.run_training(setup["context"], setup["config"], setup["weights"],
                                setup["registration"], setup["output"], **arguments)


@pytest.mark.parametrize("enabled", [False, True])
def test_dynamic_accounting_selection_and_fresh_replay(service, setup, enabled):
    receipt = run(service, setup, enabled)
    assert receipt["schema_version"] == "protocol_b_train_v1"
    assert receipt["protocol"] == "B" and receipt["group"] == 3 and "fold" not in receipt
    assert receipt["epochs_completed"] == 100 and receipt["selected_epoch"] == 3
    assert receipt["optimizer_steps"] == 400 and setup["updates"][-1] == 400
    assert receipt["scheduler"] == dict(total_updates=400, warmup_updates=40)
    assert receipt["n_train"] == 35 and receipt["n_val"] == 33
    assert receipt["images_per_epoch"] == 32 and receipt["batches_per_epoch"] == 4
    assert receipt["replay"]["passed"] is True and receipt["replay"]["fresh_process"] is False
    assert receipt["role"] == "val" and receipt["no_test_open"] is True
    assert len(setup["models"]) == 2 and setup["events"].count("registration") == 1
    expected_sampler = torch.Generator().manual_seed(1043)
    for epoch, omission in enumerate(receipt["omitted_train_ids"]):
        expected_order = torch.randperm(35, generator=expected_sampler).tolist()
        assert setup["orders"][epoch] == expected_order
        assert omission["image_ids"] == [setup["rows"]["train"][index]["image_id"] for index in expected_order[32:]]
    datasets = [entry for entry in setup["events"] if isinstance(entry, tuple)]
    assert [entry[1] for entry in datasets] == ["train", "val"]
    assert datasets[0][3] == dict(setup["config"]["augment"], seed=42)
    assert datasets[1][3] == dict(setup["config"]["augment"], enabled=False, seed=42)
    with Path(receipt["predictions"]["path"]).open() as stream:
        assert len(list(csv.DictReader(stream))) == 33
    checkpoint = service.read_direct_reference_checkpoint(Path(receipt["checkpoint"]["path"]),
                service.effective_config(setup["config"], setup["context"], enabled), receipt["identities"])
    assert service.identities(checkpoint) == receipt["identities"]
    assert receipt["identities"]["phase_authority_sha256"] == setup["gate_binding"]["sha256"]
    assert receipt["fresh_initial_state_sha256"] != tensor_sha256(checkpoint["state_dict"])


def test_effective_config_is_detached_protocol_b_identity(service, setup):
    original = copy.deepcopy(setup["config"])
    result = service.effective_config(original, setup["context"], True)
    assert original == setup["config"] and "fold" not in result
    assert result["protocol"] == "B" and result["group"] == 3
    assert result["data_contract"] == "protocol_b_query_canvas_v1"
    assert result["enrollment_sha256"] == setup["gate"]["enrollment_sha256"]
    assert result["records_sha256"] == setup["context"].contract.records_sha256
    assert result["split_sha256"] == setup["context"].contract.split_sha256
    assert result["arm"] == "on" and result["reference_enabled"] is True
    result["augment"]["affine_px"] = 99
    assert original["augment"]["affine_px"] == 8


@pytest.mark.parametrize("field", ["passed", "no_test_open", "both_arm_smoke", "replay", "batch_invariance"])
@pytest.mark.parametrize("value", [False, 1, None])
def test_gate_boolean_checks_fail_before_pixels_or_model(service, setup, field, value):
    gate = dict(setup["gate"], **{field: value})
    gate_binding = write(setup["tmp"] / "bad-gate.json", gate)
    with pytest.raises(ValueError):
        run(service, setup, gate_binding=gate_binding)
    assert not setup["events"] and not (setup["output"] / "training").exists()


@pytest.mark.parametrize("field", ["source_sha256", "records_sha256", "split_sha256", "enrollment_sha256",
                                  "registration_sha256", "weights_sha256", "recipe_sha256", "schema_version", "group", "protocol"])
def test_gate_identity_mismatch_is_failclosed(service, setup, field):
    gate = dict(setup["gate"], **{field: "f" * 64})
    with pytest.raises(ValueError):
        run(service, setup, gate_binding=write(setup["tmp"] / "wrong-gate.json", gate))
    assert not setup["events"]


def test_gate_file_hash_required(service, setup):
    with pytest.raises(ValueError, match="SHA256"):
        run(service, setup, gate_binding=dict(setup["gate_binding"], sha256="0" * 64))
    assert not setup["events"]


@pytest.mark.parametrize("field,value", [("epochs", 10), ("seed", 43), ("head_seed", 43),
    ("adapter_seed", 43), ("batch_size", 16), ("eval_batch_size", 8), ("drop_last", False),
    ("eval_drop_last", True), ("accumulation_steps", 2), ("mask_weight", 1.),
    ("clip_grad_norm", 1.), ("input_shape", [4, 32, 32]), ("initialization", "resume"),
    ("bn_policy", "other"), ("resume", "old.pt"), ("checkpoint", "old.pt")])
def test_changed_recipe_or_warmstart_rejected_before_pixels(service, setup, field, value):
    setup["config"][field] = value
    with pytest.raises(ValueError):
        run(service, setup)
    assert not setup["events"]


@pytest.mark.parametrize("field,value", [("precision", {"amp": True}), ("optimizer", {"name": "sgd"}),
    ("selection", {"primary": "auc"}), ("normalization", {"mean": [0., 0., 0.]}),
    ("augment", {"enabled": False})])
def test_nested_recipe_changes_rejected(service, setup, field, value):
    setup["config"][field] = value
    with pytest.raises(ValueError):
        run(service, setup)
    assert not setup["events"]


@pytest.mark.parametrize("failure", ["unsorted", "duplicate", "wrong-label", "test-row", "wrong-role", "missing-row", "overlap"])
def test_membership_rejected_before_registration(service, setup, failure):
    rows = copy.deepcopy(setup["rows"])
    if failure == "unsorted":
        rows["train"].reverse()
    elif failure == "duplicate":
        rows["train"][1] = dict(rows["train"][0])
    elif failure == "wrong-label":
        rows["train"][0]["label"] = 1
    elif failure == "test-row":
        rows["train"][0] = dict(rows["test"][0], role="train")
    elif failure == "wrong-role":
        rows["val"][0]["role"] = "test"
    elif failure == "missing-row":
        rows["val"].pop()
    else:
        rows["val"][0]["image_id"] = rows["train"][0]["image_id"]
    object.__setattr__(setup["context"], "_rows", json.dumps(rows))
    with pytest.raises(ValueError):
        run(service, setup)
    assert not setup["events"]


def test_existing_arm_output_rejected_without_touching_inputs(service, setup):
    directory = setup["output"] / "training/off"
    directory.mkdir(parents=True)
    sentinel = directory / "keep.txt"
    sentinel.write_text("untouched")
    with pytest.raises(ValueError):
        run(service, setup)
    assert sentinel.read_text() == "untouched" and not setup["events"]


@pytest.mark.parametrize("protected", ["raw", "banks"])
def test_output_cannot_overlap_protected_inputs(service, setup, protected):
    setup["output"] = setup["tmp"] / protected
    with pytest.raises(ValueError):
        run(service, setup)
    assert not setup["events"]


def test_output_parent_symlink_rejected_before_inputs(service, setup):
    setup["output"].mkdir(exist_ok=True)
    bank_root = setup["tmp"] / "banks"
    bank_root.mkdir()
    (setup["output"] / "training").symlink_to(bank_root, target_is_directory=True)
    with pytest.raises(ValueError):
        run(service, setup)
    assert not setup["events"] and not (bank_root / "off").exists()


@pytest.mark.parametrize("failure", ["validation-membership", "epoch-accounting", "replay"])
def test_failed_actual_execution_never_writes_success_receipt(service, setup, monkeypatch, failure):
    if failure == "validation-membership":
        original = service.evaluate_direct_reference

        def evaluate(*args):
            summary, predictions = original(*args)
            predictions[0]["image_id"] = "test000"
            return summary, predictions

        monkeypatch.setattr(service, "evaluate_direct_reference", evaluate)
    elif failure == "epoch-accounting":
        monkeypatch.setattr(service, "train_direct_reference_epoch", lambda *args, **kwargs: dict(n_images=17, optimizer_steps=2))
    else:
        original = service.replay_direct_reference_checkpoint

        def replay(*args, **kwargs):
            summary, predictions, result = original(*args, **kwargs)
            return summary, predictions, dict(result, passed=False)

        monkeypatch.setattr(service, "replay_direct_reference_checkpoint", replay)
    with pytest.raises(ValueError):
        run(service, setup)
    assert not (setup["output"] / "training/off/train.json").exists()


def test_identities_reject_legacy_checkpoint(service):
    with pytest.raises(ValueError):
        service.identities(dict(schema_version="pilot_v1", identities={}))


def test_cache_budget_shared_between_train_and_validation(service, setup):
    run(service, setup)
    datasets = [entry for entry in setup["events"] if isinstance(entry, tuple)]
    assert sum(entry[4] for entry in datasets) == 128
    assert all(entry[4] > 0 for entry in datasets)


def test_matched_arms_share_registration_gate_recipe_and_initial_tensors(service, setup):
    off = run(service, setup)
    off_order = copy.deepcopy(setup["orders"])
    setup["orders"].clear()
    setup["updates"].clear()
    on = run(service, setup, True)
    assert off_order == setup["orders"]
    assert off["registration"] == on["registration"] == setup["registration"]
    assert off["diagnostic_gate"] == on["diagnostic_gate"] == setup["gate_binding"]
    assert off["fresh_initial_state_sha256"] == on["fresh_initial_state_sha256"]
    for field in ("source_sha256", "weights_sha256", "records_sha256", "split_sha256", "enrollment_sha256",
                  "registration_sha256", "phase_authority_sha256", "preprocess_sha256"):
        assert off["identities"][field] == on["identities"][field]
    assert off["identities"]["config_sha256"] != on["identities"]["config_sha256"]


def test_changed_metadata_file_fails_before_registration(service, setup):
    Path(setup["context"].bindings[0]["path"]).write_text("mutated synthetic metadata")
    with pytest.raises(ValueError, match="SHA256"):
        run(service, setup)
    assert not setup["events"]


@pytest.mark.parametrize("budget", [-1, True, 1.5])
def test_invalid_cache_budget_fails_before_inputs(service, setup, budget):
    with pytest.raises(ValueError):
        run(service, setup, cache_bytes=budget)
    assert not setup["events"]


@pytest.mark.parametrize("failure", ["mixed-arm", "old-protocol", "fold", "initialization", "phase"])
def test_checkpoint_identity_helper_rejects_foreign_contracts(service, setup, failure):
    receipt = run(service, setup)
    checkpoint = torch.load(receipt["checkpoint"]["path"], map_location="cpu", weights_only=True)
    if failure == "mixed-arm":
        checkpoint["arm"] = "on"
    elif failure == "old-protocol":
        checkpoint["config"]["protocol"] = "A"
    elif failure == "fold":
        checkpoint["config"]["fold"] = 0
    elif failure == "initialization":
        checkpoint["initialization_provenance"]["base"]["requested_initialization"] = "warmstart"
    else:
        checkpoint["phase"] = "smoke"
    with pytest.raises(ValueError):
        service.identities(checkpoint)


def test_claimed_gate_without_nested_receipts_cannot_train(service, setup):
    gate = copy.deepcopy(setup["gate"])
    del gate["arm_reports"]
    binding = write(Path(setup["gate_binding"]["path"]), gate)
    with pytest.raises(ValueError):
        run(service, setup, gate_binding=binding)
    assert not setup["events"]


def test_changed_smoke_bytes_cannot_train(service, setup):
    Path(setup["gate"]["arm_reports"]["on"]["smoke"]["path"]).write_text("changed")
    with pytest.raises(ValueError):
        run(service, setup)
    assert not setup["events"]


@pytest.mark.parametrize("failure", ["initial-state", "initial-provenance"])
def test_training_initialization_must_equal_smoke_before_dataset(service, setup, monkeypatch, failure):
    original = service.build_direct_reference_model

    def changed(*args, **kwargs):
        model = original(*args, **kwargs)
        if failure == "initial-state":
            model.encoder.data.add_(.001)
        else:
            model.initialization_provenance["adapter_tensor_sha256"] = "f" * 64
        return model

    monkeypatch.setattr(service, "build_direct_reference_model", changed)
    with pytest.raises(ValueError):
        run(service, setup)
    assert not setup["orders"] and not any(isinstance(event, tuple) for event in setup["events"])


def test_training_receipt_binds_history_summary_parent_and_process(service, setup):
    receipt = run(service, setup)
    assert type(receipt["pid"]) is int and receipt["pid"] > 0
    assert receipt["request"] == setup["bundle"]["request_binding"]
    for name, filename in (("history", "log.csv"), ("summary", "val_summary.json"), ("in_process_replay", "replay.json")):
        path = setup["output"] / "training/off" / filename
        assert receipt[name] == dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def test_arm_recipe_enters_effective_config_and_models(service, setup):
    config = copy.deepcopy(setup["config"])
    config["arm_recipe"] = dict(off=dict(local_branch=False), on=dict(auxiliary_query_weight=.5, reference_bn="stored"))
    off = service.effective_config(config, setup["context"], False)
    on = service.effective_config(config, setup["context"], True)
    assert "arm_recipe" not in off and "arm_recipe" not in on
    assert off["local_branch"] is False and off["auxiliary_query_weight"] == 0. and off["reference_bn"] == "shared"
    assert on["local_branch"] is True and on["auxiliary_query_weight"] == .5 and on["reference_bn"] == "stored"
    plain = service.effective_config(setup["config"], setup["context"], True)
    assert plain["local_branch"] is True and plain["auxiliary_query_weight"] == 0. and plain["reference_bn"] == "shared"
    config["arm_recipe"] = dict(on=dict(local_branch=False))
    with pytest.raises(ValueError):
        service.effective_config(config, setup["context"], True)
