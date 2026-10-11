"""State isolation, normalization probes and score orientation for the control diagnostic."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
SCRIPT = ROOT / "scripts" / "diagnose_control.py"


def load_diagnostic():
    assert SCRIPT.is_file(), "the state-isolated control diagnostic is not implemented"
    spec = importlib.util.spec_from_file_location("diagnose_control", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(nn.BatchNorm2d(1), nn.Dropout(0.8))
        self.head = nn.BatchNorm2d(1)

    def forward(self, inputs):
        features = self.head(self.backbone(inputs))
        return {"logit": features.mean((1, 2, 3))}


def test_probe_isolates_bn_and_restores_state_on_error():
    diagnostic = load_diagnostic()
    model = TinyModel().train()
    original = {name: value.clone() for name, value in model.state_dict().items()}
    original_flags = [module.training for module in model.modules()]
    inputs = torch.arange(16, dtype=torch.float32).reshape(4, 1, 2, 2)
    for mode in ("eval", "backbone_batch", "all_batch"):
        try:
            with diagnostic.bn_probe(model, mode):
                assert not model.backbone[1].training, "dropout must stay disabled"
                assert model.backbone[0].training == (mode != "eval")
                assert model.head.training == (mode == "all_batch")
                model(inputs)
                assert all(torch.equal(original[name], value) for name, value in model.state_dict().items())
                raise RuntimeError("probe interrupted")
        except RuntimeError as error:
            assert str(error) == "probe interrupted"
        assert [module.training for module in model.modules()] == original_flags
        assert model.backbone[0].track_running_stats
        assert model.head.track_running_stats
        assert all(torch.equal(original[name], value) for name, value in model.state_dict().items())


def test_probe_changes_batch_statistics_not_weights():
    diagnostic = load_diagnostic()
    model = TinyModel().eval()
    inputs = torch.arange(16, dtype=torch.float32).reshape(4, 1, 2, 2) + 20
    with diagnostic.bn_probe(model, "eval"):
        native = model(inputs)["logit"]
    with diagnostic.bn_probe(model, "all_batch"):
        batch = model(inputs)["logit"]
    assert abs(float(batch.mean())) < 1e-5
    assert float(native.mean()) > 20
    assert all(parameter.grad is None for parameter in model.parameters())


def test_recalibration_is_train_only_and_preserves_parameters():
    diagnostic = load_diagnostic()
    model = TinyModel().eval()
    parameters = {name: value.clone() for name, value in model.named_parameters()}
    inputs = torch.arange(16, dtype=torch.float32).reshape(4, 1, 2, 2) + 20
    loader = [{"x": inputs}]
    for role in ("val", "test"):
        try:
            diagnostic.recalibrate(model, loader, torch.device("cpu"), role=role)
        except ValueError:
            pass
        else:
            raise AssertionError("calibration accepted non-training data")
    assert model.backbone[0].num_batches_tracked.item() == 0
    diagnostic.recalibrate(model, loader, torch.device("cpu"), role="train")
    assert model.backbone[0].num_batches_tracked.item() == 1
    assert torch.allclose(model.backbone[0].running_mean, torch.tensor([27.5]))
    assert all(torch.equal(parameters[name], value) for name, value in model.named_parameters())
    assert not any(module.training for module in model.modules())


def test_score_summary_preserves_polarity_and_logit_auc():
    diagnostic = load_diagnostic()
    summary = diagnostic.summarize(np.array([0, 1]), np.array([-2., 2.]))
    assert summary["auc_logit"] == 1.0
    assert summary["accuracy"] == 1.0
    assert summary["genuine"]["n"] == summary["forged"]["n"] == 1
    assert abs(summary["bce_image"] - 0.126928011) < 1e-7
    assert diagnostic.summarize(np.array([0, 1]), np.array([0., 0.]))["predicted_forged"] == 0
    saturated = diagnostic.summarize(np.array([0, 1]), np.array([80., 100.]))
    assert saturated["auc_logit"] == 1.0
    assert saturated["auc_pipeline_score"] == 0.5


if __name__ == "__main__":
    torch.set_num_threads(1)
    test_probe_isolates_bn_and_restores_state_on_error()
    test_probe_changes_batch_statistics_not_weights()
    test_recalibration_is_train_only_and_preserves_parameters()
    test_score_summary_preserves_polarity_and_logit_auc()
    print("ok: 4 control diagnostic tests")
