import hashlib
import importlib
import inspect
from collections import OrderedDict
from pathlib import Path
import subprocess
import sys

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def test_rgb_builder_is_available():
    assert importlib.util.find_spec("refer_or_clear.models.anchor_rgb") is not None, "RGB builder is not implemented"


class TinyResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 4, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(4)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(4, 1000)

    def forward(self, query):
        features = self.bn1(self.conv1(query)).relu()
        return self.fc(self.avgpool(features).flatten(1))


@pytest.fixture
def rgb(monkeypatch):
    assert importlib.util.find_spec("refer_or_clear.models.anchor_rgb") is not None, "RGB builder is not implemented"
    module = importlib.import_module("refer_or_clear.models.anchor_rgb")
    import torchvision.models

    def offline_resnet50(*, weights):
        assert weights is None, "Construction must never request a download"
        return TinyResNet()

    monkeypatch.setattr(torchvision.models, "resnet50", offline_resnet50)
    return module


@pytest.fixture
def checkpoint(tmp_path):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(12)
        source = TinyResNet()
    source.bn1.num_batches_tracked.fill_(17)
    source.bn1.running_mean.fill_(0.3)
    path = tmp_path / "imagenet.pth"
    torch.save(source.state_dict(), path)
    return path, hashlib.sha256(path.read_bytes()).hexdigest(), source.state_dict()


def test_weights_missing_or_hash_wrong_fail_closed(rgb, checkpoint, tmp_path):
    path, digest, _ = checkpoint
    with pytest.raises(FileNotFoundError):
        rgb.build_anchor_rgb(tmp_path / "missing.pth", digest, 42)
    with pytest.raises(ValueError, match="SHA256"):
        rgb.build_anchor_rgb(path, "0" * 64, 42)
    path.write_bytes(b"not a torch checkpoint")
    with pytest.raises(ValueError, match="checkpoint"):
        rgb.build_anchor_rgb(path, hashlib.sha256(path.read_bytes()).hexdigest(), 42)


@pytest.mark.parametrize("mutation", ["missing_buffer", "missing_head", "unexpected", "shape", "dtype", "not_tensor", "wrapped"])
def test_pretraining_state_keys_and_shapes_fail_closed(rgb, checkpoint, mutation):
    path, _, state = checkpoint
    if mutation == "missing_buffer":
        del state["bn1.num_batches_tracked"]
    elif mutation == "missing_head":
        del state["fc.weight"]
    elif mutation == "unexpected":
        state["unknown.weight"] = torch.ones(1)
    elif mutation == "shape":
        state["fc.weight"] = torch.zeros(2, 4)
    elif mutation == "dtype":
        state["conv1.weight"] = state["conv1.weight"].double()
    elif mutation == "not_tensor":
        state["bn1.num_batches_tracked"] = 17
    else:
        state = {"state_dict": state}
    torch.save(state, path)
    with pytest.raises(ValueError, match="keys|shape|dtype|tensor"):
        rgb.build_anchor_rgb(path, hashlib.sha256(path.read_bytes()).hexdigest(), 42)


@pytest.mark.parametrize("metadata_version", [None, 1])
def test_official_v1_historical_counters_are_explicitly_migrated(rgb, checkpoint, monkeypatch, metadata_version):
    path, _, original = checkpoint
    state = OrderedDict((name, value) for name, value in original.items() if name != "bn1.num_batches_tracked")
    if metadata_version is not None:
        state._metadata = {"bn1": {"version": metadata_version}}
    torch.save(state, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(rgb, "_IMAGENET1K_V1_SHA256", digest, raising=False)
    with torch.random.fork_rng(devices=[]):
        reference = TinyResNet()
    result = reference.load_state_dict(state, strict=True)
    assert result.missing_keys == []
    assert result.unexpected_keys == []
    assert reference.bn1.num_batches_tracked.item() == 0
    assert "bn1.num_batches_tracked" not in state
    for name, value in state.items():
        assert torch.equal(reference.state_dict()[name], value)
    rng = torch.get_rng_state().clone()
    model = rgb.build_anchor_rgb(path, digest, 42)
    assert torch.equal(rng, torch.get_rng_state())
    assert model.backbone.bn1.num_batches_tracked.item() == 0
    for name, value in state.items():
        if not name.startswith("fc."):
            assert torch.equal(model.backbone.state_dict()[name], value)
    provenance = model.initialization_provenance
    assert provenance["checkpoint_sha256"] == digest
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    assert provenance["loaded_keys"] == sorted(state)
    assert provenance["missing_keys"] == []
    assert provenance["source_missing_keys"] == ["bn1.num_batches_tracked"]
    assert provenance["unexpected_keys"] == []
    assert provenance["synthesized_keys"] == ["bn1.num_batches_tracked"]
    assert provenance["synthesized_values"] == {"bn1.num_batches_tracked": 0}
    assert provenance["migration_policy"] == "official_imagenet1k_v1_missing_bn_counters_zero_v1"
    assert provenance["loaded_tensor_sha256"] == rgb._tensor_sha256(state)
    assert provenance["normalized_tensor_sha256"] == rgb._tensor_sha256(reference.state_dict())
    assert provenance["loaded_tensor_sha256"] != provenance["normalized_tensor_sha256"]
    assert provenance["checkpoint_sha256"] != provenance["normalized_tensor_sha256"]
    assert provenance["pretrained_tensor_sha256"] == rgb._tensor_sha256({
        name: value for name, value in reference.state_dict().items() if not name.startswith("fc.")
    })


def test_untrusted_historical_counters_fail_closed(rgb, checkpoint):
    path, _, original = checkpoint
    state = {name: value for name, value in original.items() if name != "bn1.num_batches_tracked"}
    torch.save(state, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="keys"):
        rgb.build_anchor_rgb(path, digest, 42)
    with pytest.raises(ValueError, match="SHA256"):
        rgb.build_anchor_rgb(path, "0676ba61b6795bbe1773cffd859882e5e297624d384b6993f7c9e683e722fb8a", 42)


def test_modern_partial_counter_state_fails_even_with_trusted_hash(rgb, checkpoint, monkeypatch):
    path, _, state = checkpoint
    assert state._metadata["bn1"]["version"] == 2
    del state["bn1.num_batches_tracked"]
    torch.save(state, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(rgb, "_IMAGENET1K_V1_SHA256", digest, raising=False)
    with pytest.raises(ValueError, match="keys"):
        rgb.build_anchor_rgb(path, digest, 42)


@pytest.mark.parametrize("mutation", ["running_mean", "running_var", "affine", "head", "unexpected", "shape", "dtype", "not_tensor"])
def test_trusted_historical_migration_rejects_other_corruptions(rgb, checkpoint, monkeypatch, mutation):
    path, _, original = checkpoint
    state = {name: value for name, value in original.items() if name != "bn1.num_batches_tracked"}
    if mutation in {"running_mean", "running_var"}:
        del state[f"bn1.{mutation}"]
    elif mutation == "affine":
        del state["bn1.weight"]
    elif mutation == "head":
        del state["fc.weight"]
    elif mutation == "unexpected":
        state["unknown.num_batches_tracked"] = torch.tensor(0)
    elif mutation == "shape":
        state["bn1.running_mean"] = torch.zeros(1)
    elif mutation == "dtype":
        state["bn1.running_mean"] = state["bn1.running_mean"].double()
    else:
        state["bn1.running_mean"] = 0
    torch.save(state, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(rgb, "_IMAGENET1K_V1_SHA256", digest, raising=False)
    rng = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="keys|shape|dtype|tensor"):
        rgb.build_anchor_rgb(path, digest, 42)
    assert torch.equal(rng, torch.get_rng_state())


def test_rgb_has_no_auxiliary_inputs(rgb, checkpoint):
    path, digest, _ = checkpoint
    model = rgb.build_anchor_rgb(path, digest, 42)
    assert list(inspect.signature(model.forward).parameters) == ["query"]
    outputs = model(torch.randn(2, 3, 7, 9))
    assert set(outputs) == {"class_logits", "attack_logit"}
    assert outputs["class_logits"].shape == (2, 2)
    assert outputs["attack_logit"].shape == (2,)
    assert model.track == "R"
    assert model.architecture_id == "anchor_rgb_r50_v1"


def test_rgb_softmax_sigmoid_and_argmax_ties(rgb, checkpoint):
    path, digest, _ = checkpoint
    model = rgb.build_anchor_rgb(path, digest, 42).eval()
    query = torch.randn(3, 3, 7, 9)
    outputs = model(query)
    logits = outputs["class_logits"]
    torch.testing.assert_close(outputs["attack_logit"], logits[:, 1] - logits[:, 0])
    torch.testing.assert_close(logits.softmax(1)[:, 1], outputs["attack_logit"].sigmoid())
    _, head = model.parameter_groups()
    with torch.no_grad():
        for parameter in head:
            parameter.zero_()
    outputs = model(query)
    assert outputs["class_logits"].argmax(1).tolist() == [0, 0, 0]
    assert ((1 - outputs["attack_logit"].sigmoid()) >= 0.5).all()


def test_gradients_and_parameter_groups(rgb, checkpoint):
    path, digest, _ = checkpoint
    model = rgb.build_anchor_rgb(path, digest, 42)
    pretrained, head = model.parameter_groups()
    active = [parameter for parameter in model.parameters() if parameter.requires_grad]
    assert len(head) == 2
    assert len(pretrained + head) == len(active)
    assert len({id(parameter) for parameter in pretrained + head}) == len(active)
    assert {id(parameter) for parameter in pretrained + head} == {id(parameter) for parameter in active}
    before = {name: value.clone() for name, value in model.named_buffers()}
    logits = model(torch.randn(4, 3, 7, 9))["class_logits"]
    nn.functional.cross_entropy(logits, torch.tensor([0, 1, 1, 0])).backward()
    assert all(parameter.grad is not None for parameter in active)
    assert any(parameter.grad.abs().sum() > 0 for parameter in pretrained)
    assert any(parameter.grad.abs().sum() > 0 for parameter in head)
    after = dict(model.named_buffers())
    assert any(not torch.equal(before[name], after[name]) for name in before)


def test_seed_rng_isolation_and_pretraining_provenance(rgb, checkpoint):
    path, digest, state = checkpoint
    torch.manual_seed(987)
    rng = torch.get_rng_state().clone()
    first = rgb.build_anchor_rgb(path, digest, 42)
    assert torch.equal(rng, torch.get_rng_state())
    second = rgb.build_anchor_rgb(path, digest, 42)
    third = rgb.build_anchor_rgb(path, digest, 43)
    assert all(torch.equal(first.state_dict()[name], value) for name, value in second.state_dict().items())
    assert not torch.equal(first.parameter_groups()[1][0], third.parameter_groups()[1][0])
    loaded = {name.removeprefix("backbone."): value for name, value in first.state_dict().items()}
    for name, value in state.items():
        if not name.startswith("fc."):
            assert torch.equal(loaded[name], value)
    provenance = first.initialization_provenance
    assert provenance["requested_initialization"] == "IMAGENET1K_V1"
    assert provenance["checkpoint_sha256"] == digest
    assert provenance["loaded_keys"] == sorted(state)
    assert provenance["missing_keys"] == []
    assert provenance["source_missing_keys"] == []
    assert provenance["unexpected_keys"] == []
    assert provenance["synthesized_keys"] == []
    assert provenance["synthesized_values"] == {}
    assert provenance["migration_policy"] is None
    assert provenance["loaded_tensor_sha256"] == provenance["normalized_tensor_sha256"]
    assert len(provenance["pretrained_tensor_sha256"]) == 64
    assert provenance["pretrained_tensor_sha256"] == third.initialization_provenance["pretrained_tensor_sha256"]
    state["conv1.weight"] += 1
    torch.save(state, path)
    changed = rgb.build_anchor_rgb(path, hashlib.sha256(path.read_bytes()).hexdigest(), 42)
    assert provenance["pretrained_tensor_sha256"] != changed.initialization_provenance["pretrained_tensor_sha256"]


def test_rgb_frozen_bn_companion_invariance(rgb, checkpoint):
    path, digest, _ = checkpoint
    model = rgb.build_anchor_rgb(path, digest, 42).eval()
    before = {name: value.clone() for name, value in model.named_buffers()}
    query = torch.randn(1, 3, 7, 9)
    companion = torch.randn(1, 3, 7, 9) * 100
    with torch.no_grad():
        single = model(query)
        for position in (0, 1):
            batch = torch.cat((query, companion) if position == 0 else (companion, query))
            outputs = model(batch)
            for name, value in single.items():
                torch.testing.assert_close(outputs[name][position:position + 1], value, atol=1e-5, rtol=1e-4)
    assert all(torch.equal(before[name], value) for name, value in model.named_buffers())


def test_fresh_process_checkpoint_replay(rgb, checkpoint, tmp_path):
    path, digest, _ = checkpoint
    model = rgb.build_anchor_rgb(path, digest, 42).eval()
    query = torch.randn(2, 3, 7, 9)
    replay = tmp_path / "replay.pth"
    torch.save({"state": model.state_dict(), "query": query, "logits": model(query)["class_logits"].detach()}, replay)
    script = """
import sys
import torch
import torchvision.models
sys.path.insert(0, sys.argv[1])
sys.path.insert(0, sys.argv[2])
from test_anchor_rgb import TinyResNet
from refer_or_clear.models.anchor_rgb import build_anchor_rgb
torchvision.models.resnet50 = lambda *, weights: TinyResNet()
model = build_anchor_rgb(sys.argv[3], sys.argv[4], 999).eval()
saved = torch.load(sys.argv[5], weights_only=True)
model.load_state_dict(saved['state'], strict=True)
torch.testing.assert_close(model(saved['query'])['class_logits'], saved['logits'], atol=1e-5, rtol=1e-4)
print('fresh-process replay passed')
"""
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, "-c", script, str(root / "src"), str(root / "tests"), str(path), digest, str(replay)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "fresh-process replay passed" in result.stdout
