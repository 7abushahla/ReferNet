import copy
from contextlib import contextmanager
import hashlib
import importlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pandas as pd
import pytest
import torch
from torch import nn
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from refer_or_clear.data.direct_reference import _gate, WEIGHTS_SHA256
from refer_or_clear.direct_reference_pilot import read_direct_reference_checkpoint, save_direct_reference_checkpoint
from refer_or_clear.pilots import json_sha256, tensor_sha256


@pytest.fixture
def diagnostic():
    class Diagnostic:
        def __getattr__(self, name):
            assert importlib.util.find_spec("refer_or_clear.direct_reference_diagnostic") is not None, "diagnostic sequencing is not implemented"
            return getattr(importlib.import_module("refer_or_clear.direct_reference_diagnostic"), name)
    return Diagnostic()


@pytest.fixture
def context():
    return {"identities": {"source_sha256": "a" * 64, "recipe_sha256": "b" * 64,
            "records_sha256": "c" * 64, "split_sha256": "d" * 64,
            "enrollment_manifest_sha256": "7691452ecb4d160e8767d277ba3ccf209355f16b3f49f85e52ff596c8a2719e8"},
            "weights_sha256": WEIGHTS_SHA256}


def payload(context, enabled, smoke=False):
    receipt = dict(context["identities"], schema_version="direct_reference_smoke_gates_v1" if smoke else "direct_reference_training_gates_v1",
                   weights_sha256=context["weights_sha256"], reference_enabled=enabled, passed=True, no_test_open=True)
    if smoke:
        receipt.update(n_images={"train": 32, "val": 32}, epochs_completed=1,
                       replay={"passed": True, "fresh_process": True})
    else:
        receipt.update(opened_roles=["train"], coordinate_gate={"passed": True}, model_gate={"passed": True},
                       overfit={"n_images": 16, "n_correct": 15, "image_loss": .14, "updates": 500})
    return receipt


def write_receipt(path, receipt):
    content = (json.dumps(receipt, sort_keys=True, allow_nan=False) + "\n").encode()
    path.write_bytes(content)
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(content).hexdigest()}


def callbacks(tmp_path, context, events):
    receipts = {"training": {}, "smoke": {}}

    def training(enabled):
        assert type(enabled) is bool
        events.append(("training", enabled))
        arm = "on" if enabled else "off"
        binding = write_receipt(tmp_path / f"training-{arm}.json", payload(context, enabled))
        receipts["training"][arm] = binding
        return binding

    def smoke(enabled, training_gates):
        assert type(enabled) is bool
        events.append(("smoke", enabled))
        assert set(training_gates) == {"off", "on"}
        assert training_gates == receipts["training"]
        for arm, active in (("off", False), ("on", True)):
            _gate(training_gates[arm], context["identities"], context["weights_sha256"], active)
        arm = "on" if enabled else "off"
        binding = write_receipt(tmp_path / f"smoke-{arm}.json", payload(context, enabled, smoke=True))
        receipts["smoke"][arm] = binding
        return binding

    return training, smoke, receipts


@pytest.mark.parametrize("enabled,smoke", [(False, False), (True, False), (False, True), (True, True)])
def test_synthetic_receipts_satisfy_actual_data_gate(tmp_path, context, enabled, smoke):
    binding = write_receipt(tmp_path / "gate.json", payload(context, enabled, smoke))
    _gate(binding, context["identities"], context["weights_sha256"], enabled, smoke)


def test_both_mode_training_precedes_any_validation_and_returns_bound_receipts(diagnostic, tmp_path, context):
    events = []
    original = copy.deepcopy(context)
    training, smoke, receipts = callbacks(tmp_path, context, events)
    result = diagnostic.run_diagnostic_stages(context, training, smoke)
    assert events == [("training", False), ("training", True), ("smoke", False), ("smoke", True)]
    assert result["passed"] is True
    assert result["training_gates"] == receipts["training"] and result["smoke_gates"] == receipts["smoke"]
    assert context == original
    for arm, enabled in (("off", False), ("on", True)):
        _gate(result["training_gates"][arm], context["identities"], context["weights_sha256"], enabled)
        _gate(result["smoke_gates"][arm], context["identities"], context["weights_sha256"], enabled, smoke=True)


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("failure", ["failed", "wrong_arm", "wrong_weights", "wrong_schema", "source_drift", "hash_mutation", "validation_opened", "overfit_failed"])
def test_invalid_training_receipt_blocks_every_smoke_callback(diagnostic, tmp_path, context, enabled, failure):
    events = []
    training, smoke, _ = callbacks(tmp_path, context, events)

    def invalid_training(active):
        binding = training(active)
        if active is not enabled:
            return binding
        path = Path(binding["path"])
        receipt = json.loads(path.read_text())
        if failure == "failed":
            receipt["passed"] = False
        elif failure == "wrong_arm":
            receipt["reference_enabled"] = not active
        elif failure == "wrong_weights":
            receipt["weights_sha256"] = "0" * 64
        elif failure == "wrong_schema":
            receipt["schema_version"] = "direct_reference_smoke_gates_v1"
        elif failure == "source_drift":
            receipt["source_sha256"] = "e" * 64
        elif failure == "validation_opened":
            receipt["opened_roles"] = ["train", "val"]
        elif failure == "overfit_failed":
            receipt["overfit"]["n_correct"] = 14
        else:
            path.write_text(path.read_text() + " ")
            return binding
        return write_receipt(path, receipt)

    with pytest.raises(ValueError):
        diagnostic.run_diagnostic_stages(context, invalid_training, smoke)
    assert events and all(stage == "training" for stage, _ in events)


def test_later_training_callback_mutating_earlier_receipt_cannot_unlock_validation(diagnostic, tmp_path, context):
    events = []
    training, smoke, receipts = callbacks(tmp_path, context, events)

    def mutate_earlier(enabled):
        binding = training(enabled)
        if enabled:
            path = Path(receipts["training"]["off"]["path"])
            receipt = json.loads(path.read_text())
            receipt["passed"] = False
            write_receipt(path, receipt)
        return binding

    with pytest.raises(ValueError):
        diagnostic.run_diagnostic_stages(context, mutate_earlier, smoke)
    assert events == [("training", False), ("training", True)]


def test_callback_cannot_replace_source_authority_mid_diagnostic(diagnostic, tmp_path, context):
    events = []
    training, smoke, _ = callbacks(tmp_path, context, events)

    def drift_context(enabled):
        if enabled:
            context["identities"]["source_sha256"] = "e" * 64
        return training(enabled)

    with pytest.raises(ValueError):
        diagnostic.run_diagnostic_stages(context, drift_context, smoke)
    assert events and all(stage == "training" for stage, _ in events)


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("failure", ["failed", "wrong_arm", "wrong_weights", "wrong_schema", "source_drift", "hash_mutation", "replay_failed", "not_fresh", "wrong_count", "unsafe_test_access"])
def test_invalid_smoke_receipt_never_returns_passed(diagnostic, tmp_path, context, enabled, failure):
    events = []
    training, smoke, _ = callbacks(tmp_path, context, events)

    def invalid_smoke(active, training_gates):
        binding = smoke(active, training_gates)
        if active is not enabled:
            return binding
        path = Path(binding["path"])
        receipt = json.loads(path.read_text())
        if failure == "failed":
            receipt["passed"] = False
        elif failure == "wrong_arm":
            receipt["reference_enabled"] = not active
        elif failure == "wrong_weights":
            receipt["weights_sha256"] = "0" * 64
        elif failure == "wrong_schema":
            receipt["schema_version"] = "direct_reference_training_gates_v1"
        elif failure == "source_drift":
            receipt["source_sha256"] = "e" * 64
        elif failure == "replay_failed":
            receipt["replay"]["passed"] = False
        elif failure == "not_fresh":
            receipt["replay"]["fresh_process"] = False
        elif failure == "wrong_count":
            receipt["n_images"]["val"] = 31
        elif failure == "unsafe_test_access":
            receipt["no_test_open"] = False
        else:
            path.write_text(path.read_text() + " ")
            return binding
        return write_receipt(path, receipt)

    with pytest.raises(ValueError):
        diagnostic.run_diagnostic_stages(context, training, invalid_smoke)
    assert events[:2] == [("training", False), ("training", True)]
    assert ("smoke", enabled) in events


def test_training_callback_exception_never_opens_smoke(diagnostic, tmp_path, context):
    events = []
    _, smoke, _ = callbacks(tmp_path, context, events)

    def failed_training(enabled):
        events.append(("training", enabled))
        raise RuntimeError("training stage failed")

    with pytest.raises(RuntimeError, match="training stage failed"):
        diagnostic.run_diagnostic_stages(context, failed_training, smoke)
    assert events == [("training", False)]


def test_previous_success_cannot_supply_gates_for_a_later_failed_run(diagnostic, tmp_path, context):
    events = []
    training, smoke, _ = callbacks(tmp_path, context, events)
    assert diagnostic.run_diagnostic_stages(context, training, smoke)["passed"] is True
    events.clear()

    def failed_training(enabled):
        events.append(("training", enabled))
        receipt = payload(context, enabled)
        receipt["passed"] = False
        return write_receipt(tmp_path / "reset-failed.json", receipt)

    with pytest.raises(ValueError):
        diagnostic.run_diagnostic_stages(context, failed_training, smoke)
    assert events == [("training", False)]


def test_later_smoke_mutation_of_earlier_receipt_blocks_final_pass(diagnostic, tmp_path, context):
    events = []
    training, smoke, receipts = callbacks(tmp_path, context, events)

    def mutate_earlier(enabled, training_gates):
        binding = smoke(enabled, training_gates)
        if enabled:
            path = Path(receipts["smoke"]["off"]["path"])
            receipt = json.loads(path.read_text())
            receipt["replay"]["fresh_process"] = False
            write_receipt(path, receipt)
        return binding

    with pytest.raises(ValueError):
        diagnostic.run_diagnostic_stages(context, training, mutate_earlier)
    assert events == [("training", False), ("training", True), ("smoke", False), ("smoke", True)]


def test_training_pixel_audit_denies_validation_before_decode_and_restores_reader(diagnostic, tmp_path, monkeypatch):
    module = importlib.import_module("refer_or_clear.direct_reference_diagnostic")
    root = tmp_path / "raw"
    root.mkdir()
    binding = write_receipt(root / "val.json", {"synthetic_receipt_not_image": True})
    context = {"manifest": {"paths": {"raw_root": str(root)}}}
    allowed = pd.DataFrame([{"path": "train.json", "image_id": "train-only", "role": "train"}])
    original = module.bridge._image

    def forbidden_decode(*args, **kwargs):
        raise AssertionError("validation reached image decoding")

    monkeypatch.setattr(module.cv2, "imdecode", forbidden_decode)
    with pytest.raises(ValueError, match="outside current phase"):
        with module.pixel_audit(context, allowed) as opened:
            module.bridge._image(binding, module.cv2.IMREAD_COLOR)
    assert opened == [] and module.bridge._image is original


def test_gated_pixel_audit_allows_only_declared_validation_membership(diagnostic, tmp_path):
    module = importlib.import_module("refer_or_clear.direct_reference_diagnostic")
    root = tmp_path / "raw"
    root.mkdir()
    binding = write_receipt(root / "declared-val.json", {"synthetic_receipt_not_image": True})
    foreign = write_receipt(root / "foreign-val.json", {"synthetic_receipt_not_image": True})
    context = {"manifest": {"paths": {"raw_root": str(root)}}}
    allowed = pd.DataFrame([{"path": "declared-val.json", "image_id": "declared-val", "role": "val"}])
    original = module.bridge._image
    with module.pixel_audit(context, allowed) as opened:
        with pytest.raises(ValueError, match="cannot decode"):
            module.bridge._image(binding, module.cv2.IMREAD_COLOR)
        with pytest.raises(ValueError, match="outside current phase"):
            module.bridge._image(foreign, module.cv2.IMREAD_COLOR)
    assert opened == [{"image_id": "declared-val", "role": "val", "path": binding["path"]}]
    assert module.bridge._image is original


class CoreClassifier(nn.Module):
    architecture_id = "anchor_direct_reference_ia18_v1"

    def __init__(self, enabled):
        super().__init__()
        self.reference_enabled = enabled
        self.encoder = nn.Parameter(torch.tensor(1.))
        self.head = nn.Parameter(torch.tensor(0.))
        self.register_buffer("updates", torch.tensor(0))
        self.initialization_provenance = {
            "base": {"requested_initialization": "levir", "checkpoint_sha256": WEIGHTS_SHA256,
                     "pretrained_tensor_sha256": "b" * 64},
            "architecture_id": self.architecture_id, "reference_enabled": enabled,
            "adapter_seed": 42, "adapter_tensor_sha256": "c" * 64,
        }

    def parameter_groups(self):
        return [self.encoder], [self.head]

    def forward(self, query, *, overlap, reference, context_available):
        if self.training:
            self.updates.add_(1)
        query_logit = query[:, 0, 0, 0] * self.encoder + self.head
        correction = torch.zeros_like(query_logit)
        if self.reference_enabled:
            correction[context_available] = self.head * reference[context_available, 0, 0, 0] * overlap[context_available, 0, 0, 0]
        return {"attack_logit": query_logit + correction, "query_logit": query_logit, "correction_logit": correction}


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("replay_failure", ["none", "bool_only", "checkpoint", "identities", "count", "order"])
def test_engine_smoke_runs_real_core_recipe_selection_and_new_checkpoint(diagnostic, tmp_path, context, monkeypatch, enabled, replay_failure):
    module = importlib.import_module("refer_or_clear.direct_reference_diagnostic")
    config = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs/direct_reference_pilot.yaml").read_text())
    context.update(config=config, device=torch.device("cpu"), run_dir=tmp_path,
                   request=write_receipt(tmp_path / "request.json", {"synthetic": True}),
                   manifest={"authorities": {"records": {"path": "synthetic-records"}, "split": {"path": "synthetic-split"}}})
    context["identities"]["recipe_sha256"] = json_sha256(config)
    membership = {}
    for role in ("train", "val"):
        membership[role] = pd.DataFrame([{"image_id": f"{role}-{index:02d}", "role": role, "path": f"{role}-{index:02d}.json",
            "label": index % 2, "design_id": "design", "component_id": f"component-{index}",
            "attack_family": "text_copy" if index % 4 == 1 else "text_inpaint" if index % 2 else "real"} for index in range(32)])
    monkeypatch.setattr(module, "load_role_rows", lambda records, split, role: membership[role].copy())
    training_gates = {}
    for arm, active in (("off", False), ("on", True)):
        receipt = payload(context, active)
        pristine = CoreClassifier(active)
        receipt["initialization"] = {"state_sha256": tensor_sha256(pristine.state_dict()),
                                     "initialization_sha256": json_sha256(pristine.initialization_provenance)}
        training_gates[arm] = write_receipt(tmp_path / f"training-{arm}.json", receipt)

    def prepare(actual_context, rows, phase, gates):
        assert actual_context is context and phase == "smoke" and gates == training_gates
        assert rows.role.value_counts().to_dict() == {"train": 32, "val": 32}
        directory = tmp_path / "diagnostic" / "smoke"
        directory.mkdir(parents=True)
        phase_authority = write_receipt(directory / "phase.json", {"schema_version": "direct_reference_phase_v1", "phase": phase})
        table = write_receipt(directory / "table.json", {"synthetic": True})
        reads = [{"image_id": row.image_id, "role": row.role, "path": row.path} for row in rows.itertuples()]
        ledger = write_receipt(directory / "registration-query-reads.json", {"schema_version": "direct_reference_query_reads_v1",
                               "phase": "smoke", "shared": True, "query_opens": reads, "registration": table})
        return {"rows": rows.copy(), "directory": directory, "preprocess": {"phase_authority": phase_authority},
                "table_binding": table, "registration_pixel_opens": reads, "registration_query_reads": ledger}

    models, dataset_calls, replay_commands = [], [], []

    def build_model(actual_context, active):
        assert actual_context is context
        model = CoreClassifier(active)
        models.append(model)
        return model

    def dataset(actual_context, prepared, rows, active, augmented=False):
        assert actual_context["config"] == config
        dataset_calls.append((rows.role.iloc[0], active, augmented))
        items = []
        for index, row in enumerate(rows.to_dict("records")):
            query = torch.zeros(4, 32, 32)
            query[0].fill_(2. if row["label"] else -2.)
            item = {"query": query, "overlap": torch.ones(1, 32, 32), "context_available": torch.tensor(index % 4 != 0),
                    "label": row["label"], "meta": {key: row[key] for key in ("image_id", "design_id", "attack_family")}}
            if active:
                item["reference"] = torch.ones(3, 32, 32)
            items.append(item)
        return items

    @contextmanager
    def no_pixel_reads(actual_context, rows):
        assert actual_context is context and len(rows) == 64
        yield []

    def replay_process(command, **kwargs):
        replay_commands.append(command)
        arm = command[command.index("--arm") + 1]
        directory = tmp_path / "diagnostic" / "smoke" / arm
        state_path = directory / "replay-state.json"
        state = json.loads(state_path.read_text())
        report = {"passed": True, "fresh_process": True, "checkpoint": state["checkpoint"],
                  "identities": state["identities"], "val_ids": state["val_ids"], "summary": {"n_images": 32},
                  "replay_state": module.bind(state_path), "pixel_opens": []}
        if replay_failure == "bool_only":
            report = {"passed": True, "fresh_process": True}
        elif replay_failure == "checkpoint":
            report["checkpoint"]["sha256"] = "0" * 64
        elif replay_failure == "identities":
            report["identities"]["source_sha256"] = "e" * 64
        elif replay_failure == "count":
            report["summary"]["n_images"] = 1
        elif replay_failure == "order":
            report["val_ids"] = list(reversed(report["val_ids"]))
        write_receipt(directory / "replay.json", report)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(module, "_prepare_table", prepare)
    monkeypatch.setattr(module, "_model", build_model)
    monkeypatch.setattr(module, "_dataset", dataset)
    monkeypatch.setattr(module, "_invariance", lambda model, data, device: {"passed": True})
    monkeypatch.setattr(module, "pixel_audit", no_pixel_reads)
    monkeypatch.setattr(module.subprocess, "run", replay_process)
    engine = module.Diagnostic(context)
    if replay_failure != "none":
        with pytest.raises(ValueError, match="replay"):
            engine.smoke(enabled, training_gates)
        assert not (tmp_path / "diagnostic" / "smoke" / ("on" if enabled else "off") / "smoke-gate.json").exists()
        return
    binding = engine.smoke(enabled, training_gates)
    _gate(binding, context["identities"], context["weights_sha256"], enabled, smoke=True)
    receipt = json.loads(Path(binding["path"]).read_text())
    arm = "on" if enabled else "off"
    assert receipt["epochs_completed"] == 1 and receipt["n_images"] == {"train": 32, "val": 32}
    assert receipt["training"]["n_images"] == 32 and receipt["training"]["optimizer_steps"] == 4
    assert receipt["validation"]["n_images"] == 32 and receipt["validation"]["accuracy"] == 1.
    assert receipt["validation"]["n_context_unavailable"] == 8
    assert receipt["initial_state_sha256"] == json.loads(Path(training_gates[arm]["path"]).read_text())["initialization"]["state_sha256"]
    assert dataset_calls == [("train", enabled, True), ("val", enabled, False)]
    assert models[0].updates.item() == 4
    assert len(replay_commands) == 1 and replay_commands[0][replay_commands[0].index("--arm") + 1] == arm
    assert "--replay-state" in replay_commands[0] and "--replay-state-sha256" in replay_commands[0]
    state = json.loads((tmp_path / "diagnostic" / "smoke" / arm / "replay-state.json").read_text())
    checkpoint = read_direct_reference_checkpoint(Path(state["checkpoint"]["path"]), state["config"], state["identities"])
    assert checkpoint["schema_version"] == "direct_reference_pilot_v1"
    assert checkpoint["architecture_id"] == "anchor_direct_reference_ia18_v1" and checkpoint["arm"] == arm
    assert checkpoint["epoch"] == 1 and checkpoint["selection_key"] == [1., 1.]
    assert receipt["registration_query_reads"] == engine.prepared_smoke["registration_query_reads"]
    assert len(json.loads(Path(receipt["registration_query_reads"]["path"]).read_text())["query_opens"]) == 64
    assert receipt["pixel_opens_scope"] == "query_reads"
    if not enabled:
        other = engine.smoke(True, training_gates)
        other_receipt = json.loads(Path(other["path"]).read_text())
        assert other_receipt["registration_query_reads"] == receipt["registration_query_reads"]


@pytest.mark.parametrize("phase", ["coordinate", "smoke"])
def test_prepare_table_persists_shared_registration_query_reads(diagnostic, tmp_path, context, monkeypatch, phase):
    module = importlib.import_module("refer_or_clear.direct_reference_diagnostic")
    enrollment = write_receipt(tmp_path / "enrollment.json", {"synthetic": True})
    context.update(run_dir=tmp_path, manifest={"authorities": {"enrollment": enrollment, "records": enrollment, "split": enrollment},
                   "product_root_authority": enrollment, "paths": {"raw_root": str(tmp_path / "raw")}})
    rows = pd.DataFrame([{"image_id": f"{role}-{index:02d}", "role": role, "path": f"{role}/{index:02d}.json"}
                         for role in (("train",) if phase == "coordinate" else ("train", "val")) for index in range(32)])
    reads = [{"image_id": row.image_id, "role": row.role, "path": str(tmp_path / "raw" / row.path)} for row in rows.itertuples()]

    @contextmanager
    def audit(actual_context, selected):
        assert actual_context is context and selected.equals(rows)
        yield reads

    monkeypatch.setattr(module, "pixel_audit", audit)
    monkeypatch.setattr(module.bridge, "build_registration_table", lambda *args, **kwargs: {"synthetic": True})
    monkeypatch.setattr(module.bridge, "load_registration_table", lambda *args, **kwargs: {"synthetic": True})
    prepared = module._prepare_table(context, rows, phase, {} if phase == "smoke" else None)
    binding = prepared["registration_query_reads"]
    ledger = module.bridge._json(binding)
    assert ledger["schema_version"] == "direct_reference_query_reads_v1" and ledger["shared"] is True
    assert ledger["phase"] == phase and ledger["query_opens"] == reads
    assert ledger["registration"] == prepared["table_binding"]
    assert ledger["identities"] == context["identities"]
    assert {row["role"] for row in ledger["query_opens"]} == ({"train"} if phase == "coordinate" else {"train", "val"})
    assert len(ledger["query_opens"]) == (32 if phase == "coordinate" else 64)


def replay_fixture(tmp_path, context, module, monkeypatch):
    config = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs/direct_reference_pilot.yaml").read_text())
    config.update(arm="off", reference_enabled=False)
    directory = tmp_path / "diagnostic" / "smoke" / "off"
    directory.mkdir(parents=True)
    authority = write_receipt(tmp_path / "metadata.json", {"synthetic": True})
    context.update(config=config, device=torch.device("cpu"), run_dir=tmp_path,
                   manifest={"authorities": {"enrollment": authority, "records": authority, "split": authority},
                             "product_root_authority": authority, "paths": {"raw_root": str(tmp_path / "raw")}})
    context["identities"].update(recipe_sha256=json_sha256(config), records_sha256=authority["sha256"], split_sha256=authority["sha256"])
    rows = {role: pd.DataFrame([{"image_id": f"{role}-{index:02d}", "role": role, "path": f"{role}/{index:02d}.json",
            "design_id": "design", "label": index % 2, "attack_family": "text_copy" if index % 2 else "real"} for index in range(32)])
            for role in ("train", "val")}
    monkeypatch.setattr(module, "load_role_rows", lambda records, split, role: rows[role].copy())
    gates = {}
    model = CoreClassifier(False)
    for arm, enabled in (("off", False), ("on", True)):
        receipt = payload(context, enabled)
        pristine = CoreClassifier(enabled)
        receipt["initialization"] = {"state_sha256": tensor_sha256(pristine.state_dict()),
                                     "initialization_sha256": json_sha256(pristine.initialization_provenance)}
        gates[arm] = write_receipt(tmp_path / "diagnostic" / f"training-{arm}.json", receipt)
    phase = write_receipt(directory.parent / "phase-authority.json", {"schema_version": "direct_reference_phase_v1", "phase": "smoke",
                **context["identities"], "authorized_image_ids": {role: frame.image_id.tolist() for role, frame in rows.items()}, "training_gates": gates})
    preprocess = {"phase_authority": phase, "enrollment_authority": authority,
                  "product_root_authority": authority, "records_authority": authority, "split_authority": authority,
                  "raw_root": context["manifest"]["paths"]["raw_root"], "source_sha256": context["identities"]["source_sha256"],
                  "recipe_sha256": context["identities"]["recipe_sha256"], "weights_sha256": context["weights_sha256"]}
    table = write_receipt(directory.parent / "registration.json", {"synthetic": True})
    identities = {"source_sha256": context["identities"]["source_sha256"], "config_sha256": json_sha256(config),
                  "weights_sha256": context["weights_sha256"], "records_sha256": authority["sha256"], "split_sha256": authority["sha256"],
                  "enrollment_sha256": module.bridge.ENROLLMENT_SHA256, "registration_sha256": table["sha256"],
                  "phase_authority_sha256": phase["sha256"], "preprocess_sha256": json_sha256(preprocess),
                  "initialization_sha256": json_sha256(model.initialization_provenance)}
    checkpoint = directory / "best.pt"
    save_direct_reference_checkpoint(checkpoint, model, config, identities, 1, (1., 1.), "smoke")
    predictions = directory / "predictions.csv"
    predictions.write_text("image_id,label,attack_logit\n" + "".join(f"{row.image_id},{row.label},0\n" for row in rows["val"].itertuples()))
    state = {"config": config, "identities": identities, "preprocess": preprocess, "checkpoint": module.bind(checkpoint),
             "predictions": module.bind(predictions), "table": table, "val_ids": rows["val"].image_id.tolist()}
    return directory / "replay-state.json", state


@pytest.mark.parametrize("mutation", ["source", "recipe", "records", "split", "arm", "checkpoint", "preprocess", "initialization", "state_hash", "order"])
def test_child_rebinds_external_state_before_validation_iteration(diagnostic, tmp_path, context, monkeypatch, mutation):
    module = importlib.import_module("refer_or_clear.direct_reference_diagnostic")
    path, state = replay_fixture(tmp_path, context, module, monkeypatch)
    if mutation in ("source", "records", "split", "initialization"):
        state["identities"][f"{mutation}_sha256"] = "e" * 64
    elif mutation == "recipe":
        state["config"]["seed"] = 43
        state["identities"]["config_sha256"] = json_sha256(state["config"])
    elif mutation == "arm":
        state["config"].update(arm="on", reference_enabled=True)
    elif mutation == "checkpoint":
        state["checkpoint"] = write_receipt(tmp_path / "unrelated.json", {"synthetic": True})
    elif mutation == "preprocess":
        state["preprocess"]["raw_root"] = str(tmp_path / "unrelated")
    elif mutation == "order":
        state["val_ids"] = list(reversed(state["val_ids"]))
    binding = write_receipt(path, state)
    if mutation == "state_hash":
        path.write_text(path.read_text() + " ")

    def forbidden(*args, **kwargs):
        raise AssertionError("invalid replay reached dataset/model construction")

    monkeypatch.setattr(module, "_dataset", forbidden)
    monkeypatch.setattr(module, "_model", forbidden)
    monkeypatch.setattr(module.bridge, "load_registration_table", forbidden)
    with pytest.raises(ValueError):
        module.replay(context, "off", state_binding=binding)


def test_replay_cli_requires_external_state_before_loading_context(diagnostic, monkeypatch):
    module = importlib.import_module("refer_or_clear.direct_reference_diagnostic")

    def forbidden(*args, **kwargs):
        raise AssertionError("unbound replay loaded allocation context")

    monkeypatch.setattr(module, "load_context", forbidden)
    with pytest.raises(ValueError, match="replay"):
        module.main(["--request", "/synthetic/request.json", "--request-sha256", "a" * 64,
                     "--run-dir", "/synthetic/run", "--phase", "replay", "--arm", "off"])


def test_child_validated_replay_uses_context_and_records_separate_query_reads(diagnostic, tmp_path, context, monkeypatch):
    module = importlib.import_module("refer_or_clear.direct_reference_diagnostic")
    path, state = replay_fixture(tmp_path, context, module, monkeypatch)
    binding = write_receipt(path, state)
    opened = []

    @contextmanager
    def audit(actual_context, rows):
        assert actual_context is context and len(rows) == 32 and rows.role.eq("val").all()
        yield opened

    def dataset(actual_context, prepared, rows, enabled, augmented=False):
        assert actual_context is context and enabled is False and augmented is False
        assert prepared["preprocess"] == state["preprocess"]
        records = rows.to_dict("records")

        class Samples:
            def __len__(self):
                return len(records)

            def __getitem__(self, index):
                row = records[index]
                opened.append({"image_id": row["image_id"], "role": "val", "path": row["path"]})
                return {"query": torch.zeros(4, 32, 32), "overlap": torch.ones(1, 32, 32),
                        "context_available": torch.tensor(False), "label": row["label"],
                        "meta": {key: row[key] for key in ("image_id", "design_id", "attack_family")}}
        return Samples()

    def table(path, expected_sha256, *, rows, enrollment_manifest, preprocess):
        assert expected_sha256 == state["table"]["sha256"] and len(rows) == 32
        assert preprocess == state["preprocess"]
        return {"synthetic": True}

    monkeypatch.setattr(module.bridge, "load_registration_table", table)
    monkeypatch.setattr(module, "_dataset", dataset)
    monkeypatch.setattr(module, "_model", lambda actual_context, enabled: CoreClassifier(enabled))
    monkeypatch.setattr(module, "pixel_audit", audit)
    monkeypatch.setattr(module, "_invariance", lambda model, data, device: {"passed": True})
    report = module.replay(context, "off", state_binding=binding)
    assert report["passed"] is True and report["replay_state"] == binding
    assert report["identities"] == state["identities"] and report["checkpoint"] == state["checkpoint"]
    assert report["summary"]["n_images"] == 32 and report["val_ids"] == state["val_ids"]
    assert report["pixel_opens_scope"] == "query_reads" and report["query_read_stage"] == "replay"
    assert report["pixel_opens"] == opened and len(opened) == 32
    assert [row["image_id"] for row in opened] == state["val_ids"]


def test_replay_cli_forwards_parent_held_state_binding(diagnostic, monkeypatch):
    module = importlib.import_module("refer_or_clear.direct_reference_diagnostic")
    context = {"authenticated": True}
    observed = []
    monkeypatch.setattr(module, "load_context", lambda *args: context)

    def replay(actual_context, arm, *, state_binding):
        observed.append((actual_context, arm, state_binding))
        return {"passed": True}

    monkeypatch.setattr(module, "replay", replay)
    result = module.main(["--request", "/synthetic/request.json", "--request-sha256", "a" * 64,
                          "--run-dir", "/synthetic/run", "--phase", "replay", "--arm", "on",
                          "--replay-state", "/synthetic/run/diagnostic/smoke/on/replay-state.json",
                          "--replay-state-sha256", "b" * 64])
    assert result["passed"] is True
    assert observed == [(context, "on", {"path": "/synthetic/run/diagnostic/smoke/on/replay-state.json", "sha256": "b" * 64})]
@pytest.mark.parametrize("role,expected", [("train", True), ("val", False)])
def test_engine_passes_native_role_boolean_to_authenticated_dataset(diagnostic, monkeypatch, role, expected):
    import pandas as pd
    from refer_or_clear import direct_reference_diagnostic as module

    def consumer(*args, **kwargs):
        assert type(kwargs["train"]) is bool
        assert kwargs["train"] is expected
        return "validated"

    monkeypatch.setattr(module.bridge, "DirectReferenceDataset", consumer)
    context = {"config": {"augment": {"enabled": True}}}
    prepared = {"preprocess": {"raw_root": "/synthetic/raw"}, "enrollment": {}, "table": {}}
    rows = pd.DataFrame({"role": [role]})
    assert module._dataset(context, prepared, rows, False) == "validated"


@pytest.mark.parametrize("enabled", [False,True])
def test_initialization_receipt_emits_full_hash_bound_detached_provenance(diagnostic, enabled):
    model = CoreClassifier(enabled)
    model.initialization_provenance["base"]["checkpoint_path"] = "/synthetic/diagnostic-11/inputs/weights.pth"
    expected = copy.deepcopy(model.initialization_provenance)
    receipt = diagnostic._initialization_receipt(model)
    assert receipt["provenance"] == expected
    assert receipt["initialization_sha256"] == json_sha256(expected)
    assert receipt["state_sha256"] == tensor_sha256(model.state_dict())
    model.initialization_provenance["base"]["checkpoint_path"] = "/synthetic/control-22/inputs/weights.pth"
    assert receipt["provenance"] == expected
    assert receipt["initialization_sha256"] == json_sha256(receipt["provenance"])
