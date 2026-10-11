"""Train-only branch calibration, independent predictions and complete state restoration."""
from pathlib import Path
import sys

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


class TwinModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(nn.BatchNorm2d(1, track_running_stats=False))
        self.head = nn.BatchNorm2d(1)

    def forward(self, inputs):
        query = self.backbone(inputs[:, :1])
        reference = self.backbone(inputs[:, 1:])
        return self.head(query - reference)


def load_context():
    assert (ROOT / "scripts/diagnose_branch_calibration.py").exists(), "branch calibration is not implemented"
    from diagnose_branch_calibration import calibrated_branches
    return calibrated_branches


def test_branch_moments_and_independent_predictions_restore_state():
    context = load_context()
    model = TwinModel().train()
    original_module = model.backbone[0]
    state = {name: value.clone() for name, value in model.state_dict().items()}
    flags = [module.training for module in model.modules()]
    query = torch.tensor([0., 2., 4., 6.]).reshape(1, 1, 2, 2)
    inputs = torch.cat([query, query + 100], 1)
    with context(model, [{"x": inputs}], torch.device("cpu"), role="train", scope="backbone") as report:
        moments = report["backbone.0"]
        assert moments["calls_per_forward"] == 2
        assert moments["mean"] == [[3.], [103.]]
        assert moments["variance"] == [[5.], [5.]]
        assert moments["count"] == [4, 4]
        with torch.no_grad():
            shifted = inputs.clone()
            shifted[:, :1] += 2
            alone = model(shifted)
            together = model(torch.cat([shifted, inputs + 500]))[:1]
        assert torch.allclose(alone, together, atol=1e-6)
        assert torch.allclose(alone, torch.full_like(alone, 2 / (5.00001 ** .5) / (1.00001 ** .5)), atol=1e-6)
    assert model.backbone[0] is original_module
    assert [module.training for module in model.modules()] == flags
    assert all(torch.equal(value, state[name]) for name, value in model.state_dict().items())


def test_train_only_and_exception_restoration():
    context = load_context()
    model = TwinModel().eval()
    original_module = model.backbone[0]
    inputs = torch.arange(32, dtype=torch.float32).reshape(2, 2, 2, 4)
    for role in ("val", "test"):
        try:
            with context(model, [{"x": inputs}], torch.device("cpu"), role=role, scope="all"):
                pass
        except ValueError:
            pass
        else:
            raise AssertionError("non-training calibration accepted")
    try:
        with context(model, [{"x": inputs}], torch.device("cpu"), role="train", scope="all") as report:
            assert report["head"]["calls_per_forward"] == 1
            raise RuntimeError("interrupted")
    except RuntimeError as error:
        assert str(error) == "interrupted"
    assert model.backbone[0] is original_module
    assert not model._forward_pre_hooks and not model._forward_hooks


def test_consistency_probe_distinguishes_companion_dependence():
    import diagnose_branch_calibration as diagnostic
    assert hasattr(diagnostic, "consistency_probe"), "precision-controlled consistency probe is missing"

    class Independent(nn.Module):
        def forward(self, inputs):
            return {"logit": inputs.mean((1, 2, 3))}

    class Dependent(nn.Module):
        def forward(self, inputs):
            return {"logit": inputs.mean((1, 2, 3)) - inputs.mean()}

    inputs = torch.arange(64, dtype=torch.float32).reshape(16, 1, 2, 2)
    settings = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    independent = diagnostic.consistency_probe(Independent(), inputs)
    dependent = diagnostic.consistency_probe(Dependent(), inputs)
    assert (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32) == settings
    for condition in ("original_precision", "strict_fp32"):
        assert independent[condition]["batch_size_pass"]
        assert independent[condition]["companions_pass"]
        assert not dependent[condition]["batch_size_pass"]
        assert not dependent[condition]["companions_pass"]
    assert not independent["strict_fp32"]["cudnn_tf32"]
    assert not independent["strict_fp32"]["matmul_tf32"]


if __name__ == "__main__":
    torch.set_num_threads(1)
    test_branch_moments_and_independent_predictions_restore_state()
    test_train_only_and_exception_restoration()
    test_consistency_probe_distinguishes_companion_dependence()
    print("ok: 3 branch calibration tests")
