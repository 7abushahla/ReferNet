import importlib
import json
from pathlib import Path
import sys

import pytest
import torch
from torch import nn
from test_anchor_training import pilots, runtime_fixture

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@pytest.fixture
def gates():
    if importlib.util.find_spec("refer_or_clear.pilots") is not None:
        return importlib.import_module("refer_or_clear.pilots")
    class Runtime:
        def __getattr__(self, name):
            assert importlib.util.find_spec("refer_or_clear.pilots") is not None, "pilot gates are not implemented"
            return getattr(importlib.import_module("refer_or_clear.pilots"), name)
    return Runtime()


class InferencePilot(nn.Module):
    def __init__(self, dependent=False, mutate=False):
        super().__init__()
        self.bn = nn.BatchNorm2d(1)
        self.dropout = nn.Dropout(.5)
        self.dependent = dependent
        self.mutate = mutate
        self.register_buffer("calls", torch.zeros(()))

    def forward(self, query):
        output = self.dropout(self.bn(query)).mean((1, 2, 3))
        if self.dependent:
            output = output + query.mean()
        if self.mutate:
            self.calls.add_(1)
        return {"attack_logit": output}


def test_companions_positions_and_unchanged_buffers(gates):
    model = InferencePilot().train()
    result = gates.check_inference_contract(model, torch.arange(8.).reshape(8, 1, 1, 1), "R")
    assert result["passed"]
    assert result["batch_sizes"] == [1, 2, 4, 8]
    assert result["comparisons"] >= 8 * 7
    assert result["buffers_unchanged"]
    assert model.training
    assert result["atol"] == 1e-5
    assert result["rtol"] == 1e-4


@pytest.mark.parametrize("dependent,mutate", [(True, False), (False, True)])
def test_gate_catches_companion_dependence_and_buffer_mutation(gates, dependent, mutate):
    result = gates.check_inference_contract(InferencePilot(dependent, mutate), torch.arange(8.).reshape(8, 1, 1, 1), "N")
    assert not result["passed"]


def test_d_checks_singletons_without_claiming_companion_invariance(gates):
    result = gates.check_inference_contract(InferencePilot(dependent=True), torch.arange(8.).reshape(8, 1, 1, 1), "D")
    assert result["passed"]
    assert result["batch_sizes"] == [1]
    assert result["companion_invariance"] is None


def test_gate_rejects_missing_evidence(gates, tmp_path):
    result = gates.check_pilot_gates(tmp_path, "pretrain")
    assert result["passed"] is False
    assert result["failures"]


def test_pretrain_script_stops_after_overfit_smoke_and_replay(gates, monkeypatch, tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts" / "check_anchor_pilots.py"
    assert script.is_file(), "pretrain checker is not implemented"
    specification = importlib.util.spec_from_file_location("anchor_checks", script)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    phases = []
    def run(**options):
        phases.append(options["phase"])
        return {"passed": True}
    monkeypatch.setattr(module, "run_pilot", run)
    monkeypatch.setattr(module, "fresh_process_replay", lambda **options: {"passed": True})
    monkeypatch.setattr(module, "check_pilot_gates", lambda *args: {"passed": True, "failures": [], "identities": {}})
    result = module.main(["--config", "fixture.yaml", "--raw-root", str(tmp_path), "--run-dir", str(tmp_path / "diag"),
                          "--device", "cpu", "--weights", "offline.pth", "--weights-sha256", "a" * 64, "--phase", "pretrain"])
    assert result["passed"]
    assert phases == ["overfit", "smoke"]


class Projection(nn.Module):
    def base_forward(self, features):
        return features


class TraversalPilot(nn.Module):
    def __init__(self, duplicate=False):
        super().__init__()
        self.backbone = nn.Module()
        self.backbone.stem = nn.Identity()
        self.decode_head = Projection()
        self.duplicate = duplicate

    def forward(self, query):
        features = self.backbone.stem(query)
        projected = self.decode_head.base_forward(features)
        if self.duplicate:
            self.decode_head.base_forward(features)
        return {"attack_logit": projected.mean((1, 2, 3))}


def test_n_gate_requires_one_encoder_and_one_projection(gates):
    queries = torch.arange(8.).reshape(8, 1, 1, 1)
    good = gates.check_inference_contract(TraversalPilot(), queries, "N")
    assert good["passed"]
    assert good["traversal"]["calls"] == {"query_stem": 1, "shared_projection": 1}
    bad = gates.check_inference_contract(TraversalPilot(True), queries, "N")
    assert not bad["passed"]


def test_failed_pretrain_writes_gate_and_never_starts_smoke(gates, monkeypatch, tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts" / "check_anchor_pilots.py"
    specification = importlib.util.spec_from_file_location("failing_anchor_checks", script)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    def broken(**options):
        raise ValueError("uninitialized weights")
    monkeypatch.setattr(module, "run_pilot", broken)
    result = module.main(["--config", "fixture.yaml", "--raw-root", str(tmp_path), "--run-dir", str(tmp_path / "diag"),
                          "--device", "cpu", "--weights", "missing.pth", "--weights-sha256", "a" * 64, "--phase", "pretrain"])
    assert not result["passed"]
    assert (tmp_path / "diag" / "gate.json").is_file()
    assert not (tmp_path / "diag" / "smoke").exists()


def test_gate_revalidates_provenance_not_just_passed_flag(gates, runtime_fixture, tmp_path):
    options, _ = runtime_fixture
    root = tmp_path / "diagnostic"
    gates.run_pilot(**options, run_dir=root / "overfit", phase="overfit")
    info = gates.run_pilot(**options, run_dir=root / "smoke", phase="smoke")
    gates.write_json(root / "smoke" / "replay.json", {
        "passed": True, "fresh_process": True, "inference": {"passed": True},
        "identities": info["identities"], "checkpoint_sha256": info["selected_checkpoint_sha256"],
    })
    initial_gate = gates.check_pilot_gates(root, "pretrain")
    assert initial_gate["passed"], initial_gate["failures"]
    path = root / "smoke" / "train.json"
    tampered = json.loads(path.read_text())
    tampered["initialization"]["provenance"]["requested_initialization"] = "scratch"
    gates.write_json(path, tampered)
    assert not gates.check_pilot_gates(root, "pretrain")["passed"]
