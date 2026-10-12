import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_study_cli_exists():
    assert (Path(__file__).resolve().parents[1] / "scripts/protocol_b_study.py").is_file()


@pytest.fixture
def cli():
    path = Path(__file__).resolve().parents[1] / "scripts/protocol_b_study.py"
    specification = importlib.util.spec_from_file_location("_protocol_b_study_test", path)
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
    request = dict(schema_version="protocol_b_request_v1", protocol="B", group=0,
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
    monkeypatch.setattr(cli.bridge, "build_protocol_b_registration", lambda *args, **kwargs: pytest.fail("target pixels opened"))
    with pytest.raises(ValueError, match="source training incomplete"):
        cli.evaluate({"path": "unused", "sha256": "0" * 64})
    assert visited == ["off", "on"]
    assert not (output / "evaluation").exists()


@pytest.mark.parametrize("flag", ["passed", "no_test_open", "epochs_completed"])
def test_incomplete_training_receipt_is_rejected_before_checkpoint_inference(cli, tmp_path, monkeypatch, flag):
    import json

    directory = tmp_path / "training/off"
    directory.mkdir(parents=True)
    value = dict(schema_version="protocol_b_train_v1", protocol="B", group=0, passed=True, no_test_open=True,
                 arm="off", epochs_completed=100, optimizer_steps=15000)
    value[flag] = 99 if flag == "epochs_completed" else False
    (directory / "train.json").write_text(json.dumps(value))
    context = SimpleNamespace(contract=SimpleNamespace(group=0))
    monkeypatch.setattr(cli.service, "effective_config", lambda *args: {})
    monkeypatch.setattr(cli, "training_accounting", lambda *args: SimpleNamespace(total_optimizer_steps=15000))
    monkeypatch.setattr(cli.core, "read_direct_reference_checkpoint", lambda *args: pytest.fail("unsafe checkpoint opened"))
    with pytest.raises(ValueError, match="complete matched source training"):
        cli.completed_arm({}, context, {}, tmp_path, "off")
