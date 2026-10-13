import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from test_protocol_a_reference import api, inputs, contract_api, inventory


def test_study_cli_exists():
    assert (Path(__file__).resolve().parents[1] / "scripts/protocol_a_study.py").is_file()


@pytest.fixture
def cli():
    path = Path(__file__).resolve().parents[1] / "scripts/protocol_a_study.py"
    specification = importlib.util.spec_from_file_location("_protocol_a_study_test", path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


@pytest.mark.parametrize("environment", [{}, {"SLURM_JOB_ID": "1", "SLURMD_NODENAME": "login", "SLURM_JOB_PARTITION": "gpu"},
                                         {"SLURM_JOB_ID": "1", "SLURMD_NODENAME": "compute", "SLURM_JOB_PARTITION": "cpu"}])
def test_gpu_execution_requires_allocated_gpu_not_login(cli, monkeypatch, environment):
    for name in ("SLURM_JOB_ID", "SLURMD_NODENAME", "SLURM_JOB_PARTITION", "CUDA_VISIBLE_DEVICES"):
        monkeypatch.delenv(name, raising=False)
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    with pytest.raises(ValueError):
        cli.require_gpu_allocation()


def test_missing_request_fails_before_source_or_model(cli, tmp_path):
    with pytest.raises((ValueError, FileNotFoundError)):
        cli.load_request(dict(path=str(tmp_path / "missing.json"), sha256="0" * 64))


@pytest.mark.parametrize("failure", ["missing", "unofficial", "byte-drift"])
def test_official_weights_are_authenticated_before_context_or_query_pixels(cli, tmp_path, monkeypatch, failure):
    import yaml

    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({"weights_sha256": cli.service._FIXED["weights_sha256"]}))
    weights = tmp_path / "weights.bin"
    weights.write_bytes(b"not official bytes")
    request = dict(schema_version="protocol_a_request_v1", protocol="A", fold=0,
                   source_sha256=cli.object_sha256([]), config=cli.bind(config))
    if failure != "missing":
        request["weights"] = dict(path=str(weights), sha256="0" * 64 if failure == "unofficial" else cli.service._FIXED["weights_sha256"])
    original_json = cli._json
    monkeypatch.setattr(cli, "_json", lambda binding: request if binding.get("path") == "request" else original_json(binding))
    monkeypatch.setattr(cli, "source_entries", lambda root: [])
    monkeypatch.setattr(cli.bridge, "load_reference_context", lambda *args, **kwargs: pytest.fail("weights failure opened context"))
    with pytest.raises(ValueError):
        cli.load_request(dict(path="request", sha256="a" * 64))


def test_failed_source_arm_blocks_all_target_registration_and_authorization(cli, tmp_path, monkeypatch):
    output = tmp_path / "run"
    output.mkdir()
    monkeypatch.setattr(cli, "load_request", lambda *args, **kwargs: ({}, object(), {}, output))
    visited = []

    def completion(request, context, base, root, arm, *args):
        visited.append(arm)
        if arm == "on":
            raise ValueError("source training incomplete")
        return ({}, {})

    monkeypatch.setattr(cli, "completed_arm", completion)
    monkeypatch.setattr(cli, "pinned_gate", lambda *args: {})
    monkeypatch.setattr(cli.bridge, "build_protocol_a_registration", lambda *args, **kwargs: pytest.fail("target pixels opened"))
    with pytest.raises(ValueError, match="source training incomplete"):
        cli.evaluate({"path": "unused", "sha256": "0" * 64})
    assert visited == ["off", "on"]
    assert not (output / "evaluation").exists()


@pytest.mark.parametrize("flag", ["passed", "no_test_open", "epochs_completed"])
def test_incomplete_training_receipt_is_rejected_before_checkpoint_inference(cli, tmp_path, monkeypatch, flag):
    import json

    directory = tmp_path / "training/off"
    directory.mkdir(parents=True)
    value = dict(schema_version="protocol_a_train_v1", protocol="A", fold=0, passed=True, no_test_open=True,
                 arm="off", epochs_completed=100, optimizer_steps=15000)
    value[flag] = 99 if flag == "epochs_completed" else False
    (directory / "train.json").write_text(json.dumps(value))
    context = SimpleNamespace(contract=SimpleNamespace(fold=0))
    monkeypatch.setattr(cli.service, "effective_config", lambda *args: {})
    monkeypatch.setattr(cli, "training_accounting", lambda *args: SimpleNamespace(total_optimizer_steps=15000))
    monkeypatch.setattr(cli.core, "read_direct_reference_checkpoint", lambda *args: pytest.fail("unsafe checkpoint opened"))
    with pytest.raises(ValueError, match="complete matched source training"):
        cli.completed_arm({}, context, {}, tmp_path, "off")

@pytest.mark.parametrize("fold", [True, -1, 10, "1", None])
def test_invalid_fold_is_rejected_before_source_or_weights(cli, monkeypatch, fold):
    request = dict(schema_version="protocol_a_request_v1", protocol="A", fold=fold)
    monkeypatch.setattr(cli, "_json", lambda binding: request)
    monkeypatch.setattr(cli, "source_entries", lambda root: pytest.fail("invalid fold reached source"))
    with pytest.raises(ValueError, match="Protocol A"):
        cli.load_request(dict(path="request", sha256="a" * 64))


@pytest.mark.parametrize("field", ["group", "target_enrollment", "resume", "checkpoint"])
def test_foreign_authorities_fail_before_source_or_pixels(cli, monkeypatch, field):
    request = dict(schema_version="protocol_a_request_v1", protocol="A", fold=1)
    request[field] = 0
    monkeypatch.setattr(cli, "_json", lambda binding: request)
    monkeypatch.setattr(cli, "source_entries", lambda root: pytest.fail("foreign authority reached source"))
    with pytest.raises(ValueError, match="request"):
        cli.load_request(dict(path="request", sha256="a" * 64))


@pytest.mark.parametrize("arm", ["off", "on"])
def test_training_dispatch_forwards_current_source_registration_and_gate(cli, tmp_path, monkeypatch, arm):
    gate_binding = dict(path="external-gate", sha256="b" * 64)
    binding = dict(path="request", sha256="a" * 64)
    request = dict(weights={"path": "weights", "sha256": "c" * 64}, source_sha256="d" * 64, cache_bytes=128)
    context, base = object(), object()
    registration = dict(path="registration", sha256="e" * 64)
    events = []
    monkeypatch.setattr(cli, "load_request", lambda *args: (request, context, base, tmp_path))
    monkeypatch.setattr(cli, "pinned_gate", lambda *args: gate_binding)
    monkeypatch.setattr(cli, "_json", lambda authority: dict(request=binding, registration=registration))
    def run(context_arg, base_arg, weights_arg, table_arg, root, **kwargs):
        assert (context_arg, base_arg, weights_arg, table_arg, root) == (context, base, request["weights"], registration, tmp_path)
        assert kwargs == dict(reference_enabled=arm == "on", source_sha256=request["source_sha256"],
                              gate_binding=gate_binding, device="cuda", cache_bytes=128)
        events.append("train")
        return dict(passed=True)
    monkeypatch.setattr(cli.service, "run_training", run)
    monkeypatch.setattr(cli, "child_replay", lambda *args: events.append(args))
    assert cli.train(binding, arm, "b" * 64) == dict(passed=True)
    assert events == ["train", (binding, arm, "train", "b" * 64)]


def test_child_replay_uses_current_interpreter_and_external_gate(cli, monkeypatch):
    calls = []
    monkeypatch.setattr(cli.subprocess, "run", lambda arguments, **kwargs: calls.append((arguments, kwargs)))
    cli.child_replay(dict(path="/request path.json", sha256="a" * 64), "on", "train", "b" * 64)
    arguments, kwargs = calls[0]
    assert arguments == [cli.sys.executable, str(Path(cli.__file__).resolve()), "replay",
                         "--request", "/request path.json", "--request-sha256", "a" * 64,
                         "--arm", "on", "--phase", "train", "--gate-sha256", "b" * 64]
    assert kwargs == dict(check=True)


@pytest.fixture
def synthetic_run(cli, tmp_path, monkeypatch):
    import copy
    import json
    import pandas as pd
    import torch
    import yaml
    from test_direct_reference_pilot import SyntheticClassifier

    base = yaml.safe_load((Path(cli.ROOT) / "configs/direct_reference_pilot.yaml").read_text())
    base["fold"] = 4
    rows = {role: pd.DataFrame([dict(image_id=f"{role}{index:03d}", design_id="design",
            identity_id=f"identity{index}", component_id=f"component{index}", label=index % 2,
            attack_family="text_copy" if index % 2 else "", role=role, path=f"{role}{index}.png")
            for index in range(count)]) for role, count in (("train", 35), ("val", 223), ("test", 223))}
    bindings = []
    for name in ("records", "split", "enrollment", "root"):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(dict(synthetic=name)))
        bindings.append(cli.bind(path))
    metadata = dict(protocol="A", fold=4, records_sha256=bindings[0]["sha256"],
                    split_sha256=bindings[1]["sha256"], enrollment_manifest_sha256=bindings[2]["sha256"],
                    product_root_sha256=bindings[3]["sha256"])
    contract = SimpleNamespace(fold=4, records_sha256=metadata["records_sha256"], split_sha256=metadata["split_sha256"],
        source_train=tuple(SimpleNamespace(**row) for row in rows["train"].to_dict("records")),
        source_val=tuple(SimpleNamespace(**row) for row in rows["val"].to_dict("records")),
        target_test=tuple(SimpleNamespace(**row) for row in rows["test"].to_dict("records")))
    context = SimpleNamespace(contract=contract, identities=json.dumps(metadata), rows=lambda role: rows[role].copy(deep=True),
        bindings=bindings, raw_root=tmp_path / "raw", manifest=dict(output_root=str(tmp_path / "banks")),
        product_root=str(tmp_path / "relocated-banks"))
    output = tmp_path / "output"
    weights = tmp_path / "weights.bin"
    weights.write_bytes(b"synthetic initialization")
    request = dict(schema_version="protocol_a_request_v1", protocol="A", fold=4,
        source_sha256="a" * 64, records=bindings[0], split=bindings[1], enrollment=bindings[2], root=bindings[3],
        raw_root=str(context.raw_root), output_root=str(output), cache_bytes=128,
        weights=dict(path=str(weights), sha256=base["weights_sha256"]))
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(base))
    request["config"] = cli.bind(config_path)
    binding = cli.write(tmp_path / "request.json", request)
    models, events = [], []

    def effective(config, context_arg, enabled):
        assert context_arg is context
        return dict(copy.deepcopy(config), protocol="A", fold=context.contract.fold,
                    data_contract="protocol_a_query_canvas_v1", reference_enabled=enabled, arm="on" if enabled else "off")

    def fresh(request_arg, enabled, **unused_recipe):
        assert request_arg == request
        model = SyntheticClassifier(enabled)
        model.initialization_provenance["base"]["checkpoint_path"] = str(weights)
        models.append(model)
        events.append(("fresh", enabled))
        return model

    def dataset(context_arg, table, *, role, reference_enabled, augment, cache_bytes):
        assert context_arg is context and table == {"synthetic": "registration"}
        events.append(("dataset", role, reference_enabled, augment, cache_bytes))
        samples = []
        for row in rows[role].to_dict("records"):
            query = torch.zeros(4, 32, 32)
            query[0].fill_(2. if row["label"] else -2.)
            sample = dict(query=query, overlap=torch.ones(1, 32, 32), label=row["label"],
                context_available=torch.tensor(True), meta={key: row[key] for key in ("image_id", "design_id", "attack_family")})
            if reference_enabled:
                sample["reference"] = torch.zeros(3, 32, 32)
            samples.append(sample)
        return samples

    def registration(context_arg, *, roles, output_path):
        assert context_arg is context
        events.append(("registration", roles))
        cli.write(output_path, {"synthetic": "registration"})
        return {"synthetic": "registration"}

    original_train = cli.core.train_direct_reference_epoch
    original_evaluate = cli.core.evaluate_direct_reference
    original_replay = cli.core.replay_direct_reference_checkpoint
    original_inputs = cli.core._batch_inputs
    original_loader = cli.load_request
    monkeypatch.setattr(cli.core, "train_direct_reference_epoch", lambda model, loader, optimizer, scheduler, device, auxiliary_query_weight=0.:
        original_train(model, loader, optimizer, scheduler, "cpu"))
    monkeypatch.setattr(cli.core, "evaluate_direct_reference", lambda model, loader, device: original_evaluate(model, loader, "cpu"))
    monkeypatch.setattr(cli.core, "replay_direct_reference_checkpoint", lambda path, model, loader, device, **kwargs:
        original_replay(path, model, loader, "cpu", **kwargs))
    monkeypatch.setattr(cli.core, "_batch_inputs", lambda model, batch, device: original_inputs(model, batch, "cpu"))
    monkeypatch.setattr(cli, "load_request", lambda *args, **kwargs: (request, context, base, output))
    monkeypatch.setattr(cli, "fresh_model", fresh)
    monkeypatch.setattr(cli.service, "effective_config", effective)
    monkeypatch.setattr(cli.bridge, "ProtocolAReferenceDataset", dataset)
    monkeypatch.setattr(cli.bridge, "build_protocol_a_registration", registration)
    monkeypatch.setattr(cli.bridge, "load_protocol_a_registration", lambda *args, **kwargs: {"synthetic": "registration"})
    monkeypatch.setattr(cli, "training_accounting", lambda *args: SimpleNamespace(total_optimizer_steps=400))
    monkeypatch.setattr(cli, "make_protocol_a_scheduler", lambda optimizer, *args:
                        torch.optim.lr_scheduler.LambdaLR(optimizer, lambda iteration: 1.))
    monkeypatch.setattr(cli.receipts, "validate_training_gate", lambda *args, **kwargs: events.append("gate-validated"))
    return dict(request=request, binding=binding, context=context, base=base, output=output, models=models,
                events=events, rows=rows, original_loader=original_loader)


def test_real_synthetic_smoke_saves_both_fresh_arms_and_replays_all_32(cli, synthetic_run, monkeypatch):
    run = synthetic_run
    replays = []

    def child(binding, arm, phase):
        replays.append((binding, arm, phase))
        cli.replay(binding, arm, phase)

    monkeypatch.setattr(cli, "child_replay", child)
    gate = cli._json(cli.smoke(run["binding"]))
    assert gate["protocol"] == "A" and gate["fold"] == 4 and "group" not in gate
    assert gate["passed"] and gate["no_test_open"] and gate["request"] == run["binding"]
    assert gate["accounting"]["total_optimizer_steps"] == 400
    assert replays == [(run["binding"], arm, "smoke") for arm in ("off", "on")]
    smokes = [cli._json(gate["arm_reports"][arm]["smoke"]) for arm in ("off", "on")]
    assert smokes[0]["fresh_initial_state_sha256"] == smokes[1]["fresh_initial_state_sha256"]
    for arm, smoke in zip(("off", "on"), smokes):
        assert smoke["stats"]["n_images"] == 32 and smoke["stats"]["optimizer_steps"] == 4
        assert smoke["source_val_ids"] == run["rows"]["val"].head(32).image_id.tolist()
        assert smoke["batch_invariance"]["passed"] and smoke["batch_invariance"]["buffers_unchanged"]
        fresh = cli._json(gate["arm_reports"][arm]["replay"])
        assert fresh["n_images"] == 32 and fresh["parent_result"] == gate["arm_reports"][arm]["smoke"]
        assert fresh["comparison"]["passed"] and fresh["comparison"]["n_images"] == 32
    assert not any(event[:2] == ("dataset", "test") for event in run["events"] if isinstance(event, tuple))
    assert run["events"][-1] == "gate-validated"


@pytest.mark.parametrize("test_count", [222, 223])
def test_evaluation_authenticates_both_completed_arms_before_shared_full_population_registration(cli, synthetic_run, monkeypatch, test_count):
    import copy

    run = synthetic_run
    context = run["context"]
    context.contract.target_test = context.contract.target_test[:test_count]
    run["rows"]["test"] = run["rows"]["test"].head(test_count)
    checkpoint = run["output"].parent / "selected.pt"
    checkpoint.write_bytes(b"mock selected checkpoint")
    checkpoint_binding = cli.bind(checkpoint)
    parents = {}
    for arm in ("off", "on"):
        config = cli.service.effective_config(run["base"], context, arm == "on")
        parents[arm] = (dict(checkpoint=checkpoint_binding, identities={}, selected_epoch=69), config)

    def completed(request, source_context, base, output, arm, gate, binding):
        assert binding == run["binding"] and source_context is context
        run["events"].append(("completed", arm))
        assert not (output / "evaluation").exists()
        return parents[arm]

    monkeypatch.setattr(cli, "completed_arm", completed)
    monkeypatch.setattr(cli, "pinned_gate", lambda *args: {"sha256": "b" * 64})
    monkeypatch.setattr(cli.core, "read_direct_reference_checkpoint", lambda *args:
        dict(state_dict=copy.deepcopy(run["models"][-1].state_dict())))
    result = cli._json(cli.evaluate(run["binding"], "b" * 64))
    assert run["events"][:3] == [("completed", "off"), ("completed", "on"), ("registration", ["test"])]
    authority = cli._json(result["authorization"])
    assert authority["authorized_image_ids"] == run["rows"]["test"].image_id.tolist()
    assert authority["checkpoints"] == {arm: checkpoint_binding for arm in ("off", "on")}
    assert authority["source_validation_selected"] is True and authority["fold"] == 4
    for arm in ("off", "on"):
        summary = cli._json(result["summaries"][arm])
        assert summary["n_images"] == summary["query_count"] == summary["published_target_count"] == test_count
        assert summary["published_test_population_preserved"] is True
        assert summary["n_target_support"] == summary["n_target_excluded"] == 0
        assert summary["selected_epoch"] == 69 and summary["source_checkpoint"] == checkpoint_binding
        assert len(cli.predictions(summary["predictions"])) == test_count


def test_completed_arm_requires_fresh_receipt_validation_not_parent_only(cli, tmp_path, monkeypatch):
    directory = tmp_path / "training/off"
    value = dict(schema_version="protocol_a_train_v1", protocol="A", fold=9, passed=True, no_test_open=True,
                 arm="off", epochs_completed=100, optimizer_steps=400)
    cli.write(directory / "train.json", value)
    context = SimpleNamespace(contract=SimpleNamespace(fold=9))
    monkeypatch.setattr(cli.service, "effective_config", lambda *args: {})
    monkeypatch.setattr(cli, "training_accounting", lambda *args: SimpleNamespace(total_optimizer_steps=400))
    calls = []
    monkeypatch.setattr(cli.receipts, "validate_completed_training", lambda *args, **kwargs: calls.append((args, kwargs)))
    cli.completed_arm({}, context, {}, tmp_path, "off", {"gate": "pinned"}, {"request": "pinned"})
    assert calls[0][1] == {}
    assert calls[0][0][-2:] == ({"gate": "pinned"}, {"request": "pinned"})


def authenticated_request(cli, run, monkeypatch, **changes):
    request = dict(run["request"], source_sha256=cli.object_sha256([]), **changes)
    monkeypatch.setattr(cli, "source_entries", lambda root: [])
    original_bytes = cli._bytes
    monkeypatch.setattr(cli, "_bytes", lambda binding:
        b"synthetic authenticated official weights" if binding == request["weights"] else original_bytes(binding))
    calls = []

    def context(records, split, **kwargs):
        calls.append((records, split, kwargs))
        return run["context"]

    monkeypatch.setattr(cli.bridge, "load_reference_context", context)
    binding = cli.write(run["output"].parent / "actual-request.json", request)
    return binding, request, calls


def test_request_authenticates_fold_native_enrollment_and_durable_root(cli, synthetic_run, monkeypatch):
    run = synthetic_run
    binding, request, calls = authenticated_request(cli, run, monkeypatch)
    authority = dict(path="test-authority", sha256="f" * 64)
    actual = run["original_loader"](binding, evaluation_binding=authority)
    assert actual == (request, run["context"], run["base"], run["output"])
    assert calls == [(Path(request["records"]["path"]), Path(request["split"]["path"]), dict(
        fold=4, records_sha256=request["records"]["sha256"], enrollment_binding=request["enrollment"],
        root_binding=request["root"], raw_root=Path(request["raw_root"]), evaluation_binding=authority))]


def test_request_forwards_explicit_metadata_relocation_authority(cli, synthetic_run, monkeypatch):
    run = synthetic_run
    relocation = dict(path="relocation-ledger", sha256="e" * 64)
    binding, request, calls = authenticated_request(cli, run, monkeypatch, metadata_relocation=relocation)
    run["original_loader"](binding)
    assert calls[0][2]["metadata_relocation_binding"] == relocation


@pytest.mark.parametrize("protected", ["raw", "banks", "relocated", "source"])
@pytest.mark.parametrize("direction", ["parent", "child"])
def test_request_output_cannot_overlap_scientific_inputs(cli, synthetic_run, monkeypatch, protected, direction):
    run = synthetic_run
    root = dict(raw=run["context"].raw_root, banks=Path(run["context"].manifest["output_root"]),
                relocated=Path(run["context"].product_root), source=Path(cli.ROOT))[protected]
    output = root.parent if direction == "parent" else root / "nested"
    binding, request, calls = authenticated_request(cli, run, monkeypatch, output_root=str(output))
    with pytest.raises(ValueError, match="overlap"):
        run["original_loader"](binding)
    assert not run["output"].exists()


@pytest.mark.parametrize("drift", ["source", "config", "split", "cache"])
def test_request_drift_fails_before_registration_or_model(cli, synthetic_run, monkeypatch, drift):
    run = synthetic_run
    changes = dict(cache_bytes=-1) if drift == "cache" else {}
    binding, request, calls = authenticated_request(cli, run, monkeypatch, **changes)
    if drift == "source":
        monkeypatch.setattr(cli, "source_entries", lambda root: [dict(path="changed.py", sha256="f" * 64)])
    elif drift == "config":
        Path(request["config"]["path"]).write_text("changed: true\n")
    elif drift == "split":
        run["context"].contract.split_sha256 = "f" * 64
    with pytest.raises(ValueError):
        run["original_loader"](binding)
    assert run["events"] == [] and run["models"] == []


@pytest.mark.parametrize("expected", [None, "", "b" * 63, "B" * 64, "g" * 64])
def test_gate_requires_external_lowercase_sha(cli, tmp_path, expected):
    with pytest.raises(ValueError, match="externally pinned"):
        cli.pinned_gate(tmp_path, expected)


def test_training_wrong_request_gate_never_calls_service(cli, tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "load_request", lambda *args: ({}, object(), {}, tmp_path))
    monkeypatch.setattr(cli, "pinned_gate", lambda *args: {"sha256": "b" * 64})
    monkeypatch.setattr(cli, "_json", lambda *args: dict(request={"path": "different-request"}))
    monkeypatch.setattr(cli.service, "run_training", lambda *args, **kwargs: pytest.fail("foreign gate opened training"))
    with pytest.raises(ValueError, match="different request"):
        cli.train({"path": "request", "sha256": "a" * 64}, "off", "b" * 64)


@pytest.mark.parametrize("arm", ["off", "on"])
def test_train_replay_authenticates_parent_before_registration_or_model(cli, synthetic_run, monkeypatch, arm):
    run = synthetic_run
    cli.write(run["output"] / "training" / arm / "train.json", {"passed": True})
    monkeypatch.setattr(cli, "pinned_gate", lambda *args: {"sha256": "b" * 64})

    def reject(*args, **kwargs):
        assert kwargs == dict(require_fresh=False)
        raise ValueError("invalid completed parent")

    monkeypatch.setattr(cli.receipts, "validate_completed_training", reject)
    monkeypatch.setattr(cli.bridge, "load_protocol_a_registration", lambda *args, **kwargs:
        pytest.fail("invalid parent opened registration"))
    with pytest.raises(ValueError, match="invalid completed parent"):
        cli.replay(run["binding"], arm, "train", "b" * 64)
    assert not run["models"] and not run["events"]


@pytest.mark.parametrize("val_count", [222, 223])
@pytest.mark.parametrize("arm", ["off", "on"])
def test_full_training_replay_uses_exact_fold_validation_count_and_selected_phase(cli, synthetic_run, monkeypatch, val_count, arm):
    run = synthetic_run
    context = run["context"]
    run["rows"]["val"] = run["rows"]["val"].head(val_count)
    context.contract.source_val = context.contract.source_val[:val_count]
    cli.bridge.build_protocol_a_registration(context, roles=["train", "val"], output_path=run["output"] / "registration.json")
    registration = cli.bind(run["output"] / "registration.json")
    gate = dict(path="external-gate", sha256="b" * 64)
    config = cli.service.effective_config(run["base"], context, arm == "on")
    model = cli.fresh_model(run["request"], arm == "on")
    identities = cli.checkpoint_ids(run["request"], context, config, registration, gate, model)
    directory = run["output"] / "training" / arm
    directory.mkdir(parents=True)
    key = cli.selection_key("accuracy", 1., 1., .1, .2)
    cli.core.save_direct_reference_checkpoint(directory / "best.pt", model, config, identities, 69, key, "train")
    data = cli.bridge.ProtocolAReferenceDataset(context, {"synthetic": "registration"}, role="val",
        reference_enabled=arm == "on", augment=dict(config["augment"], enabled=False), cache_bytes=0)
    summary, expected = cli.core.evaluate_direct_reference(model, cli.DataLoader(data, batch_size=1), "cuda")
    cli._write_csv(directory / "val_predictions.csv", expected)
    parent = dict(passed=True, identities=identities, checkpoint=cli.bind(directory / "best.pt"),
                  predictions=cli.bind(directory / "val_predictions.csv"))
    parent_binding = cli.write(directory / "train.json", parent)
    parent_calls = []

    def completed(*args, **kwargs):
        parent_calls.append((args, kwargs))
        return parent, config

    monkeypatch.setattr(cli.receipts, "validate_completed_training", completed)
    monkeypatch.setattr(cli, "pinned_gate", lambda *args: gate)
    replay = cli._json(cli.replay(run["binding"], arm, "train", gate["sha256"]))
    assert parent_calls[0][1] == dict(require_fresh=False)
    assert replay["parent_result"] == parent_binding and replay["request"] == run["binding"]
    assert replay["n_images"] == replay["comparison"]["n_images"] == val_count
    assert replay["comparison"]["max_abs_difference"] == 0. and replay["no_test_open"]
    assert replay["summary"] == summary and replay["identities"] == identities


def test_fresh_factory_isolated_between_arms_with_official_fixed_seeds(cli, monkeypatch):
    captures = []
    request = dict(weights={"path": "/synthetic/official.bin", "sha256": "a" * 64})
    monkeypatch.setattr(cli, "_bytes", lambda binding: b"synthetic weights")
    monkeypatch.setattr(cli, "configure_precision", lambda device, track: captures.append((device, track)))
    monkeypatch.setattr(cli.torch.cuda, "manual_seed_all", lambda seed: captures.append(("cuda-seed", seed)))

    def factory(path, sha, **kwargs):
        captures.append((path, sha, kwargs, cli.random.random(), float(cli.np.random.rand()), float(cli.torch.rand(()))))
        return SimpleNamespace(cuda=lambda: object())

    monkeypatch.setattr(cli, "build_direct_reference_model", factory)
    first = cli.fresh_model(request, False)
    second = cli.fresh_model(request, True)
    factories = [entry for entry in captures if isinstance(entry[0], Path)]
    assert first is not second and len(factories) == 2
    for enabled, invocation in zip((False, True), factories):
        assert invocation[:3] == (Path(request["weights"]["path"]), "a" * 64,
                                 dict(reference_enabled=enabled, head_seed=42, adapter_seed=42,
                                      local_enabled=True, reference_bn="shared"))
    assert factories[0][3:] == factories[1][3:]


@pytest.mark.parametrize("command", ["smoke", "train", "replay", "evaluate"])
def test_main_dispatches_authenticated_cli_arguments_without_new_recipe_flags(cli, tmp_path, monkeypatch, capsys, command):
    events = []
    monkeypatch.setattr(cli, "require_gpu_allocation", lambda: events.append("allocated"))
    for name in ("smoke", "train", "replay", "evaluate"):
        monkeypatch.setattr(cli, name, lambda *args, phase_name=name: events.append((phase_name, args)) or dict(passed=True))
    request_path = tmp_path / "request.json"
    arguments = [command, "--request", str(request_path), "--request-sha256", "a" * 64,
                 "--arm", "on", "--gate-sha256", "b" * 64]
    assert cli.main(arguments) == 0
    binding = dict(path=str(request_path.resolve()), sha256="a" * 64)
    expected = dict(smoke=(binding,), train=(binding, "on", "b" * 64),
                    replay=(binding, "on", "train", "b" * 64), evaluate=(binding, "b" * 64))
    assert events == ["allocated", (command, expected[command])]
    assert '"passed": true' in capsys.readouterr().out


@pytest.mark.parametrize("relocate_metadata", [False, True])
def test_actual_consumer_service_and_parent_request_share_durable_bindings(cli, api, inputs, tmp_path, monkeypatch, relocate_metadata):
    import copy
    import shutil
    import yaml

    config = dict(copy.deepcopy(cli.service._FIXED), data_contract="direct_reference_canvas_v1")
    config_path = tmp_path / "recipe.yaml"
    config_path.write_text(yaml.safe_dump(config))
    original_root = inputs["manifest"]["output_root"]
    root = cli.write(tmp_path / "product-root.json", dict(schema_version="protocol_a_product_root_v1", protocol="A",
        fold=inputs["fold"], enrollment_manifest_sha256=inputs["enrollment_binding"]["sha256"],
        original_root=original_root, relocated_root=original_root))
    weights_path = tmp_path / "synthetic-official-weights.bin"
    weights_path.write_bytes(b"synthetic initialization, not an actual model")
    weights = dict(path=str(weights_path), sha256=config["weights_sha256"])
    relocation = None
    if relocate_metadata:
        previous = inputs["paths"].copy()
        for name, path in previous.items():
            destination = tmp_path / "durable" / path.name
            destination.parent.mkdir(exist_ok=True)
            shutil.copyfile(path, destination)
            inputs["paths"][name] = destination
            path.unlink()
        relocation = cli.write(tmp_path / "metadata-relocation.json", dict(
            schema_version="protocol_a_metadata_relocation_v1", protocol="A", fold=inputs["fold"],
            enrollment_manifest_sha256=inputs["enrollment_binding"]["sha256"],
            relocations={name: dict(original_path=str(previous[name]), **cli.bind(path))
                         for name, path in inputs["paths"].items()}))
    request = dict(schema_version="protocol_a_request_v1", protocol="A", fold=inputs["fold"],
        source_sha256=cli.object_sha256([]), records=cli.bind(inputs["paths"]["records"]),
        split=cli.bind(inputs["paths"]["split"]), enrollment=inputs["enrollment_binding"], root=root,
        config=cli.bind(config_path), weights=weights, raw_root=str(inputs["raw"]),
        output_root=str(tmp_path / "study"), cache_bytes=128)
    if relocation is not None:
        request["metadata_relocation"] = relocation
    binding = cli.write(tmp_path / "study-request.json", request)
    original_bytes = cli._bytes
    monkeypatch.setattr(cli, "_bytes", lambda authority: b"synthetic official bytes"
                        if authority == weights else original_bytes(authority))
    monkeypatch.setattr(cli, "source_entries", lambda source: [])
    actual, context, base, output = cli.load_request(binding)
    assert actual == request and context.bindings[:3] == [request[name] for name in ("records", "split", "enrollment")]
    assert root in context.bindings[3:]
    if relocation is not None:
        assert relocation in context.bindings[3:]
    assert cli.service.effective_config(base, context, False)["fold"] == inputs["fold"]
    verified, root_path = cli.receipts._request(binding, context, base, weights, request["source_sha256"])
    assert verified == request and root_path == output


def test_reencoded_evaluation_requires_native_evaluation_first(cli, tmp_path, monkeypatch):
    output = tmp_path / "run"
    output.mkdir()
    monkeypatch.setattr(cli, "load_request", lambda binding, evaluation_binding=None: (dict(raw_root=str(tmp_path)), None, {}, output))
    with pytest.raises((ValueError, FileNotFoundError)):
        cli.evaluate_reencoded(dict(path=str(tmp_path / "request.json"), sha256="a" * 64), "b" * 64, 75)
    assert not (output / "evaluation-q75").exists()
