import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import random
import subprocess
import sys

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as functional

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from refer_or_clear.losses import referral_loss
from refer_or_clear.models import changerex


class TinyResidual(nn.Module):
    def __init__(self, incoming, outgoing, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(incoming, outgoing, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(outgoing)
        self.conv2 = nn.Conv2d(outgoing, outgoing, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(outgoing)
        self.downsample = (
            nn.Sequential(nn.Conv2d(incoming, outgoing, 1, stride, bias=False), nn.BatchNorm2d(outgoing))
            if incoming != outgoing or stride != 1 else nn.Identity()
        )

    def forward(self, inputs):
        residual = self.downsample(inputs)
        outputs = functional.relu(self.bn1(self.conv1(inputs)))
        return functional.relu(self.bn2(self.conv2(outputs)) + residual)


class TinyBackbone(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.deep_stem = True
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, 2, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(),
        )
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        self.res_layers = ["layer1", "layer2", "layer3", "layer4"]
        self.out_indices = (0, 1, 2, 3)
        self.ccs = nn.ModuleList([nn.Identity() for _ in range(4)])
        incoming = 64
        for name, outgoing, stride in zip(self.res_layers, (64, 128, 256, 512), (1, 2, 2, 2)):
            setattr(self, name, TinyResidual(incoming, outgoing, stride))
            incoming = outgoing

    def forward(self, query, reference):
        query = self.maxpool(self.stem(query))
        reference = self.maxpool(self.stem(reference))
        outputs = []
        for name in self.res_layers:
            layer = getattr(self, name)
            query, reference = layer(query), layer(reference)
            outputs.append(torch.cat((query, reference), dim=1))
        return tuple(outputs)


class TinyFDAF(nn.Module):
    def __init__(self):
        super().__init__()
        self.flow_make = nn.Sequential(
            nn.Conv2d(128, 128, 5, padding=2, groups=128), nn.InstanceNorm2d(128),
            nn.GELU(), nn.Conv2d(128, 4, 1, bias=False),
        )

    @staticmethod
    def warp(inputs, flow):
        batch, channels, height, width = inputs.shape
        rows = torch.linspace(-1, 1, width, device=inputs.device, dtype=inputs.dtype)
        columns = torch.linspace(-1, 1, height, device=inputs.device, dtype=inputs.dtype)
        grid = torch.stack(torch.meshgrid(columns, rows, indexing="ij")[::-1], dim=-1)
        norm = inputs.new_tensor([width, height])
        grid = grid.unsqueeze(0).expand(batch, -1, -1, -1) + flow.permute(0, 2, 3, 1) / norm
        return functional.grid_sample(inputs, grid, align_corners=True)

    def forward(self, query, absent, fusion_policy):
        assert fusion_policy == "concat"
        query_flow, absent_flow = self.flow_make(torch.cat((query, absent), dim=1)).chunk(2, dim=1)
        return torch.cat((self.warp(query, query_flow) - absent, self.warp(absent, absent_flow) - query), dim=1)


class TinyMixFFN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(128, 128, 1), nn.Conv2d(128, 128, 3, padding=1, groups=128),
            nn.GELU(), nn.Dropout(0), nn.Conv2d(128, 128, 1), nn.Dropout(0),
        )
        self.dropout_layer = nn.Identity()

    def forward(self, inputs):
        return inputs + self.dropout_layer(self.layers(inputs))


class TinyDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Sequential(nn.Conv2d(width, 128, 1, bias=False), nn.BatchNorm2d(128), nn.ReLU())
            for width in (64, 128, 256, 512)
        ])
        self.fusion_conv = nn.Sequential(nn.Conv2d(512, 64, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU())
        self.neck_layer = TinyFDAF()
        self.discriminator = TinyMixFFN()
        self.dropout = nn.Dropout2d(0.1)
        self.conv_seg = nn.Conv2d(128, 1, 1)

    def base_forward(self, inputs):
        outputs = [
            functional.interpolate(conv(inputs[index]), inputs[0].shape[-2:], mode="bilinear", align_corners=False)
            for index, conv in enumerate(self.convs)
        ]
        return self.fusion_conv(torch.cat(outputs, dim=1))

    def cls_seg(self, features):
        return self.conv_seg(self.dropout(features))

    def forward(self, features):
        halves = [feature.chunk(2, dim=1) for feature in features]
        query = self.base_forward([pair[0] for pair in halves])
        reference = self.base_forward([pair[1] for pair in halves])
        fused = self.discriminator(self.neck_layer(query, reference, "concat"))
        return fused, self.cls_seg(fused)


class TinyHistorical(nn.Module):
    def __init__(self, fusion="single_noex", in_channels=4):
        super().__init__()
        assert fusion == "single_noex"
        self.in_channels = in_channels
        self.backbone = TinyBackbone(in_channels)
        self.decode_head = TinyDecoder()
        self.referral = changerex.ReferralHead(128)
        self.attention = None

    def init_opencd_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def anchor():
    return importlib.import_module("refer_or_clear.models.anchor_changer")


@pytest.fixture
def checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(changerex, "ChangerEx", TinyHistorical)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(11)
        source = TinyHistorical(in_channels=3)
        source.init_opencd_weights()
        state = {name: value.clone() for name, value in source.state_dict().items() if not name.startswith("referral.")}
        state["decode_head.conv_seg.weight"] = torch.cat((torch.zeros_like(state["decode_head.conv_seg.weight"]), state["decode_head.conv_seg.weight"]), dim=0)
        state["decode_head.conv_seg.bias"] = torch.tensor([0.25, 0.75])
    path = tmp_path / "levir.pth"
    torch.save({"state_dict": state}, path)
    return path, hashlib.sha256(path.read_bytes()).hexdigest(), state


def build(anchor, checkpoint, track):
    path, digest, state = checkpoint
    return anchor.build_anchor_changer(track, path, digest, 42)


def query_tensor(batch=2):
    generator = torch.Generator().manual_seed(19)
    return torch.randn(batch, 4, 32, 48, generator=generator)


@pytest.mark.parametrize("policy", ["running", "batch_stats"])
def test_query_features_match_legacy_exchange_off(anchor, policy):
    historical = TinyHistorical()
    changerex.set_encoder_bn_policy(historical, policy)
    backbone = historical.backbone.eval()
    query = query_tensor()
    with torch.no_grad():
        legacy = backbone(query, torch.randn_like(query))
        actual = anchor.encode_query(backbone, query)
    assert len(actual) == 4
    assert [tuple(feature.shape[1:]) for feature in actual] == [(64, 8, 12), (128, 4, 6), (256, 2, 3), (512, 1, 2)]
    for actual_stage, legacy_stage in zip(actual, legacy):
        torch.testing.assert_close(actual_stage, legacy_stage.chunk(2, dim=1)[0], atol=0, rtol=0)


def test_direct_route_has_no_decoder_or_map_ancestor(anchor, checkpoint):
    model = build(anchor, checkpoint, "D").train()
    assert not hasattr(model, "decode_head") and not hasattr(model, "referral")
    assert all(not name.startswith(("decode_head.", "referral.", "attention.")) for name in model.state_dict())
    output = model(query_tensor())
    assert set(output) == {"attack_logit"} and output["attack_logit"].shape == (2,)
    linears = [module for module in model.modules() if isinstance(module, nn.Linear)]
    assert [(module.in_features, module.out_features) for module in linears] == [(512, 128), (128, 1)]
    functional.binary_cross_entropy_with_logits(output["attack_logit"], torch.tensor([0., 1.])).backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters())
    assert model.backbone.stem[0].weight.grad.abs().sum() > 0
    assert linears[-1].weight.grad.abs().sum() > 0


def test_absent_route_calls_encoder_and_base_forward_once(anchor, checkpoint, monkeypatch):
    model = build(anchor, checkpoint, "N").train()
    calls = {}
    handles = []
    for name, module in [("stem", model.backbone.stem), *[(name, getattr(model.backbone, name)) for name in model.backbone.res_layers]]:
        def count(module, inputs, output, name=name):
            calls[name] = calls.get(name, 0) + 1
        handles.append(module.register_forward_hook(count))
    base_forward = model.decode_head.base_forward
    projections = []
    def project_once(features):
        assert not projections, "absent reference was passed through shared projection"
        result = base_forward(features)
        projections.append(result)
        return result
    monkeypatch.setattr(model.decode_head, "base_forward", project_once)
    observed = []
    handles.append(model.decode_head.neck_layer.register_forward_pre_hook(lambda module, inputs: observed.append(inputs)))
    output = model(query_tensor())
    for handle in handles:
        handle.remove()
    assert calls == {"stem": 1, "layer1": 1, "layer2": 1, "layer3": 1, "layer4": 1}
    assert len(projections) == 1
    projected, absent, policy = observed[0]
    assert projected is projections[0] and policy == "concat"
    assert absent.shape == projected.shape and absent.dtype == projected.dtype
    assert absent.device == projected.device and absent.count_nonzero() == 0
    assert not absent.requires_grad and absent.data_ptr() != projected.data_ptr()
    assert set(output) == {"attack_logit", "map_logit"}
    assert output["attack_logit"].shape == (2,) and output["map_logit"].shape == (2, 1, 8, 12)


def test_absent_reference_never_updates_shared_bn(anchor, checkpoint):
    model = build(anchor, checkpoint, "N").train()
    batch_norms = [module for module in model.modules() if isinstance(module, nn.modules.batchnorm._BatchNorm)]
    assert batch_norms and all(module.track_running_stats for module in batch_norms)
    before = [module.num_batches_tracked.clone() for module in batch_norms]
    model(query_tensor())
    assert all(module.num_batches_tracked == previous + 1 for module, previous in zip(batch_norms, before))
    model.eval()
    assert all(not module.training for module in model.modules())
    snapshot = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    with torch.no_grad():
        model(query_tensor())
    assert all(torch.equal(tensor, snapshot[name]) for name, tensor in model.state_dict().items())


def test_direct_uses_encoder_batch_stats_and_enforces_eval_batch_one(anchor, checkpoint):
    model = build(anchor, checkpoint, "D")
    norms = [module for module in model.modules() if isinstance(module, nn.modules.batchnorm._BatchNorm)]
    assert norms and all(not module.track_running_stats for module in norms)
    assert all(module.running_mean is None and module.running_var is None and module.num_batches_tracked is None for module in norms)
    assert all(module.weight.requires_grad and module.bias.requires_grad for module in norms)
    model.train()
    model(query_tensor())
    model.eval()
    with pytest.raises(ValueError, match="batch.*one|batch.*1"):
        model(query_tensor())
    assert model(query_tensor(1))["attack_logit"].shape == (1,)
    norms[0].track_running_stats = True
    norms[0].running_mean = torch.zeros_like(norms[0].weight)
    norms[0].running_var = torch.ones_like(norms[0].weight)
    with pytest.raises((ValueError, RuntimeError), match="BN|batch|normalization"):
        model(query_tensor(1))


@pytest.mark.parametrize("track", ["D", "N"])
def test_optimizer_groups_cover_only_active_trainable_tensors(anchor, checkpoint, track):
    model = build(anchor, checkpoint, track)
    pretrained, new = model.parameter_groups()
    assert pretrained and new
    identifiers = [id(parameter) for parameter in pretrained + new]
    assert len(identifiers) == len(set(identifiers))
    assert set(identifiers) == {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    assert {id(parameter) for parameter in pretrained} == {
        id(parameter) for name, parameter in model.named_parameters() if name.startswith(("backbone.", "decode_head."))
    }
    pretrained[0].requires_grad_(False)
    assert id(pretrained[0]) not in {id(parameter) for group in model.parameter_groups() for parameter in group}


def test_absent_classification_gradients_and_dense_target_affect_loss_not_inference(anchor, checkpoint):
    model = build(anchor, checkpoint, "N").train()
    output = model(query_tensor())
    output["map_logit"].retain_grad()
    functional.binary_cross_entropy_with_logits(output["attack_logit"], torch.tensor([0., 1.])).backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
    assert output["map_logit"].grad.abs().sum() > 0
    assert model.backbone.stem[0].weight.grad.abs().sum() > 0
    assert model.decode_head.conv_seg.weight.grad.abs().sum() > 0
    mask = torch.zeros(2, 1, 32, 48)
    valid = torch.ones_like(mask)
    genuine_target, _ = referral_loss(output["attack_logit"], output["map_logit"], torch.tensor([0., 1.]), mask, valid, mask_weight=3)
    attacked_target, _ = referral_loss(output["attack_logit"], output["map_logit"], torch.tensor([0., 1.]), 1 - mask, valid, mask_weight=3)
    assert not torch.isclose(genuine_target, attacked_target)
    with pytest.raises(TypeError):
        model(query_tensor(), mask)


@pytest.mark.parametrize("track", ["D", "N"])
def test_reference_planes_are_not_model_inputs(anchor, checkpoint, track):
    model = build(anchor, checkpoint, track).eval()
    query = query_tensor(1)
    legacy = torch.cat((query, torch.randn_like(query)), dim=1)
    changed = legacy.clone()
    changed[:, 4:] = float("nan")
    with torch.no_grad():
        first, second = model(legacy[:, :4]), model(changed[:, :4])
    for name in first:
        torch.testing.assert_close(first[name], second[name], atol=0, rtol=0)
    with pytest.raises(ValueError, match="four|4|channel"):
        model(legacy)


def test_absent_running_bn_companion_invariance(anchor, checkpoint):
    model = build(anchor, checkpoint, "N").train()
    model(query_tensor())
    model.eval()
    query = query_tensor(1)
    companions = torch.randn(2, 4, 32, 48) * 5
    with torch.no_grad():
        solo = model(query)
        first = model(torch.cat((query, companions), dim=0))
        last = model(torch.cat((companions, query), dim=0))
    for name in solo:
        torch.testing.assert_close(solo[name][0], first[name][0], atol=1e-5, rtol=1e-4)
        torch.testing.assert_close(solo[name][0], last[name][-1], atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("track", ["D", "N"])
def test_checkpoint_replay(anchor, checkpoint, track):
    model = build(anchor, checkpoint, track).train()
    model(query_tensor())
    model.eval()
    stream = io.BytesIO()
    torch.save(model.state_dict(), stream)
    stream.seek(0)
    replay = build(anchor, checkpoint, track).eval()
    replay.load_state_dict(torch.load(stream, weights_only=True), strict=True)
    with torch.no_grad():
        actual, expected = replay(query_tensor(1)), model(query_tensor(1))
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("track", ["D", "N"])
def test_fresh_process_checkpoint_replay(anchor, checkpoint, tmp_path, track):
    model = build(anchor, checkpoint, track).train()
    model(query_tensor())
    model.eval()
    query = query_tensor(1)
    with torch.no_grad():
        expected = model(query)
    replay_input = tmp_path / "replay-input.pth"
    replay_output = tmp_path / "replay-output.pth"
    torch.save({"state_dict": model.state_dict(), "query": query}, replay_input)
    program = """
import runpy
import sys
import torch
fixtures = runpy.run_path(sys.argv[1])
from refer_or_clear.models import anchor_changer, changerex
changerex.ChangerEx = fixtures['TinyHistorical']
torch.set_num_threads(1)
model = anchor_changer.build_anchor_changer(sys.argv[2], sys.argv[3], sys.argv[4], 42).eval()
saved = torch.load(sys.argv[5], map_location='cpu', weights_only=True)
model.load_state_dict(saved['state_dict'], strict=True)
with torch.no_grad():
    torch.save(model(saved['query']), sys.argv[6])
"""
    completed = subprocess.run(
        [sys.executable, "-c", program, str(Path(__file__).resolve()), track, str(checkpoint[0]), checkpoint[1], str(replay_input), str(replay_output)],
        capture_output=True, text=True, timeout=90,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    actual = torch.load(replay_output, map_location="cpu", weights_only=True)
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("track", ["D", "N"])
def test_construction_preserves_rng_and_declared_historical_initialization(anchor, checkpoint, track):
    torch.manual_seed(83)
    random.seed(91)
    np.random.seed(37)
    torch_before, python_before, numpy_before = torch.get_rng_state().clone(), random.getstate(), np.random.get_state()
    model = build(anchor, checkpoint, track)
    assert torch.equal(torch_before, torch.get_rng_state())
    assert python_before == random.getstate()
    assert all(np.array_equal(before, after) for before, after in zip(numpy_before, np.random.get_state()))
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(42)
        historical = TinyHistorical()
        historical.init_opencd_weights()
        changerex.load_levir(historical, checkpoint[0], verbose=False)
    for name, parameter in model.backbone.named_parameters():
        torch.testing.assert_close(parameter, dict(historical.backbone.named_parameters())[name], atol=0, rtol=0)
    if track == "N":
        for name, tensor in historical.state_dict().items():
            torch.testing.assert_close(model.state_dict()[name], tensor, atol=0, rtol=0)
    repeated = build(anchor, checkpoint, track)
    assert all(torch.equal(tensor, repeated.state_dict()[name]) for name, tensor in model.state_dict().items())


@pytest.mark.parametrize("track", ["D", "N"])
def test_levir_surgery_and_initialization_provenance(anchor, checkpoint, track):
    model = build(anchor, checkpoint, track)
    path, digest, source = checkpoint
    assert model.architecture_id == {"D": "anchor_direct_ia18_v1", "N": "anchor_absent_ia18_v1"}[track]
    assert model.track == track
    provenance = model.initialization_provenance
    assert provenance["requested_initialization"] == "levir"
    assert provenance["checkpoint_path"] == str(path.resolve())
    assert provenance["checkpoint_sha256"] == digest
    assert set(provenance["loaded_keys"]) == set(source)
    assert set(provenance["missing_keys"]) == {f"referral.{layer}.{kind}" for layer in ("fc1", "fc2") for kind in ("weight", "bias")}
    assert provenance["unexpected_keys"] == []
    assert len(provenance["pretrained_tensor_sha256"]) == 64
    torch.testing.assert_close(model.backbone.stem[0].weight[:, :3], source["backbone.stem.0.weight"], atol=0, rtol=0)
    assert model.backbone.stem[0].weight[:, 3:].count_nonzero() == 0
    if track == "N":
        torch.testing.assert_close(model.decode_head.conv_seg.weight, source["decode_head.conv_seg.weight"][1:] - source["decode_head.conv_seg.weight"][:1], atol=0, rtol=0)
        torch.testing.assert_close(model.decode_head.conv_seg.bias, torch.tensor([0.5]), atol=0, rtol=0)
    alternate = anchor.build_anchor_changer(track, path, digest, 43)
    assert alternate.initialization_provenance["pretrained_tensor_sha256"] == provenance["pretrained_tensor_sha256"]
    assert any(not torch.equal(first, second) for first, second in zip(model.parameter_groups()[1], alternate.parameter_groups()[1]))


@pytest.mark.parametrize("track", ["D", "N"])
def test_active_pretrained_hash_matches_retained_state_independently(anchor, checkpoint, track):
    model = build(anchor, checkpoint, track)
    state = model.state_dict()
    prefixes = ("backbone.", "decode_head.") if track == "N" else ("backbone.",)
    retained_keys = sorted(name for name in state if name.startswith(prefixes))
    declared_keys = model.initialization_provenance["active_pretrained_keys"]
    assert declared_keys == retained_keys
    buffer_suffixes = ("running_mean", "running_var", "num_batches_tracked")
    if track == "D":
        assert not any(name.endswith(buffer_suffixes) for name in declared_keys)
    else:
        assert any(name.endswith("num_batches_tracked") for name in declared_keys)
    digest = hashlib.sha256()
    for name in declared_keys:
        tensor = state[name].detach().cpu().contiguous()
        descriptor = json.dumps([name, list(tensor.shape), str(tensor.dtype)], separators=(",", ":"))
        digest.update(descriptor.encode("utf-8") + b"\n")
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    assert digest.hexdigest() == model.initialization_provenance["pretrained_tensor_sha256"]


@pytest.mark.parametrize("buffer", ["running_mean", "running_var", "num_batches_tracked"])
def test_direct_discarded_bn_buffers_change_only_historical_fingerprint(anchor, checkpoint, buffer):
    original = build(anchor, checkpoint, "D").eval()
    path, digest, source = checkpoint
    changed = dict(source)
    key = f"backbone.stem.1.{buffer}"
    changed[key] = source[key] + 1
    torch.save({"state_dict": changed}, path)
    changed_digest = hashlib.sha256(path.read_bytes()).hexdigest()
    modified = anchor.build_anchor_changer("D", path, changed_digest, 42).eval()
    assert modified.initialization_provenance["historical_pretrained_tensor_sha256"] != original.initialization_provenance["historical_pretrained_tensor_sha256"]
    assert modified.initialization_provenance["pretrained_tensor_sha256"] == original.initialization_provenance["pretrained_tensor_sha256"]
    assert modified.initialization_provenance["active_pretrained_keys"] == original.initialization_provenance["active_pretrained_keys"]
    assert key not in modified.initialization_provenance["active_pretrained_keys"]
    assert all(torch.equal(tensor, modified.state_dict()[name]) for name, tensor in original.state_dict().items())
    query = query_tensor(1)
    with torch.no_grad():
        torch.testing.assert_close(modified(query)["attack_logit"], original(query)["attack_logit"], atol=0, rtol=0)


def test_direct_active_weight_changes_initialization_fingerprint(anchor, checkpoint):
    original = build(anchor, checkpoint, "D")
    path, digest, source = checkpoint
    changed = dict(source)
    key = "backbone.stem.0.weight"
    changed[key] = source[key] + 1
    torch.save({"state_dict": changed}, path)
    changed_digest = hashlib.sha256(path.read_bytes()).hexdigest()
    modified = anchor.build_anchor_changer("D", path, changed_digest, 42)
    assert key in modified.initialization_provenance["active_pretrained_keys"]
    assert modified.initialization_provenance["pretrained_tensor_sha256"] != original.initialization_provenance["pretrained_tensor_sha256"]
    assert not torch.equal(modified.backbone.stem[0].weight, original.backbone.stem[0].weight)


def test_provenance_hashes_active_pretrained_tensors_and_audits_fusion_normalization(anchor, checkpoint):
    direct = build(anchor, checkpoint, "D")
    absent = build(anchor, checkpoint, "N")
    assert direct.initialization_provenance["historical_pretrained_tensor_sha256"] == absent.initialization_provenance["historical_pretrained_tensor_sha256"]
    for model in (direct, absent):
        pretrained, new = model.parameter_groups()
        provenance = model.initialization_provenance
        assert provenance["active_parameter_count"] == sum(parameter.numel() for parameter in pretrained + new)
        assert provenance["new_parameter_count"] == sum(parameter.numel() for parameter in new)
    audit = absent.initialization_provenance["normalization_audit"]
    assert any(entry["name"].startswith("decode_head.neck_layer.") and entry["type"] == "InstanceNorm2d" and entry["track_running_stats"] is False for entry in audit)
    assert any(entry["name"] == "decode_head.dropout" and entry["probability"] == 0.1 for entry in audit)
    assert all(entry["track_running_stats"] for entry in audit if entry["type"] == "BatchNorm2d")
    path, digest, source = checkpoint
    changed = dict(source)
    changed["decode_head.conv_seg.bias"] = source["decode_head.conv_seg.bias"] + torch.tensor([0., 1.])
    torch.save({"state_dict": changed}, path)
    changed_digest = hashlib.sha256(path.read_bytes()).hexdigest()
    modified_direct = anchor.build_anchor_changer("D", path, changed_digest, 42)
    modified_absent = anchor.build_anchor_changer("N", path, changed_digest, 42)
    assert modified_direct.initialization_provenance["pretrained_tensor_sha256"] == direct.initialization_provenance["pretrained_tensor_sha256"]
    assert modified_absent.initialization_provenance["pretrained_tensor_sha256"] != absent.initialization_provenance["pretrained_tensor_sha256"]


@pytest.mark.parametrize("track", ["D", "N"])
def test_input_contract_rejects_extra_inputs_and_invalid_query_tensors(anchor, checkpoint, track):
    model = build(anchor, checkpoint, track)
    for query in (torch.randn(2, 3, 32, 48), torch.randn(2, 4, 32), torch.randn(0, 4, 32, 48), query_tensor().double()):
        with pytest.raises(ValueError):
            model(query)
    with pytest.raises(TypeError):
        model(query_tensor(), label=torch.tensor([0., 1.]))


@pytest.mark.parametrize("fault", ["missing", "hash", "invalid_hash"])
def test_weights_missing_or_hash_wrong_fail_before_construction(anchor, checkpoint, monkeypatch, fault):
    def forbidden_construction(*args, **kwargs):
        raise AssertionError("model constructed before checkpoint verification")
    monkeypatch.setattr(changerex, "ChangerEx", forbidden_construction)
    path, digest, state = checkpoint
    if fault == "missing":
        path = path.with_name("absent.pth")
    elif fault == "hash":
        digest = "0" * 64
    else:
        digest = "not-a-sha256"
    with pytest.raises((FileNotFoundError, ValueError)):
        anchor.build_anchor_changer("D", path, digest, 42)


@pytest.mark.parametrize("fault", ["missing_encoder", "missing_decoder", "missing_bn", "unexpected", "trained_referral", "folded_head", "wrong_stem", "wrong_shape", "wrong_dtype"])
def test_levir_key_allowlist_and_shapes_fail_closed(anchor, checkpoint, fault):
    path, digest, source = checkpoint
    modified = dict(source)
    if fault == "missing_encoder":
        modified.pop("backbone.layer4.conv1.weight")
    elif fault == "missing_decoder":
        modified.pop("decode_head.fusion_conv.0.weight")
    elif fault == "missing_bn":
        modified.pop("backbone.stem.1.num_batches_tracked")
    elif fault == "unexpected":
        modified["auxiliary_head.conv.weight"] = torch.ones(1)
    elif fault == "trained_referral":
        modified["referral.fc2.bias"] = torch.ones(1)
    elif fault == "folded_head":
        modified["decode_head.conv_seg.weight"] = modified["decode_head.conv_seg.weight"][:1]
        modified["decode_head.conv_seg.bias"] = modified["decode_head.conv_seg.bias"][:1]
    elif fault == "wrong_stem":
        modified["backbone.stem.0.weight"] = modified["backbone.stem.0.weight"][:, :2]
    elif fault == "wrong_shape":
        modified["backbone.layer4.conv1.weight"] = torch.ones(1)
    else:
        modified["backbone.layer4.conv1.weight"] = modified["backbone.layer4.conv1.weight"].double()
    torch.save({"state_dict": modified}, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises((ValueError, RuntimeError), match="key|shape|stem|head|dtype|tensor"):
        anchor.build_anchor_changer("D", path, digest, 42)


def test_unknown_track_is_rejected(anchor, checkpoint):
    with pytest.raises(ValueError, match="track|D|N"):
        build(anchor, checkpoint, "R")


def test_official_hash_bound_checkpoint_allows_numpy_metadata(anchor, checkpoint, monkeypatch):
    path, _, state = checkpoint
    torch.save({"state_dict": state, "meta": {"best_score": np.float64(.99)}}, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(anchor, "_TRUSTED_LEVIR_SHA256", digest, raising=False)
    model = anchor.build_anchor_changer("D", path, digest, 42)
    assert model.initialization_provenance["checkpoint_loading_policy"] == "official_hash_bound_pickle_v1"
    assert set(model.initialization_provenance["loaded_keys"]) == set(state)


def test_nonallowlisted_numpy_checkpoint_has_no_unrestricted_fallback(anchor, checkpoint, monkeypatch):
    path, _, state = checkpoint
    torch.save({"state_dict": state, "meta": {"best_score": np.float64(.99)}}, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    original = torch.load
    policies = []
    def inspect_policy(*args, **kwargs):
        policies.append(kwargs.get("weights_only"))
        return original(*args, **kwargs)
    monkeypatch.setattr(torch, "load", inspect_policy)
    with pytest.raises(ValueError, match="metadata|allowlist|official"):
        anchor.build_anchor_changer("D", path, digest, 42)
    assert policies == [True]


def test_checkpoint_deserializes_the_same_bytes_it_verified(anchor, tmp_path, monkeypatch):
    path = tmp_path / "immutable-load.pth"
    torch.save({"state_dict": {"probe": torch.tensor([1.])}}, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(anchor, "_TRUSTED_LEVIR_SHA256", digest)
    original = torch.load
    def replace_source_before_loading(source, **kwargs):
        torch.save({"state_dict": {"probe": torch.tensor([2.])}}, path)
        return original(source, **kwargs)
    monkeypatch.setattr(torch, "load", replace_source_before_loading)
    state, actual_digest = anchor._verified_checkpoint(path, digest)
    assert actual_digest == digest
    torch.testing.assert_close(state["probe"], torch.tensor([1.]))


def test_actual_opencd_gpu_integration():
    if os.environ.get("ROC_ANCHOR_GPU_INTEGRATION") != "1":
        pytest.skip("Opt-in actual Open-CD/LEVIR GPU integration is unrun locally; set ROC_ANCHOR_GPU_INTEGRATION=1 with ROC_LEVIR_CKPT and ROC_LEVIR_SHA256 in an allocation")
    assert torch.cuda.is_available(), "the integration gate requires an allocated GPU"
    changerex._require_opencd()
    anchor = importlib.import_module("refer_or_clear.models.anchor_changer")
    path = Path(os.environ["ROC_LEVIR_CKPT"])
    digest = os.environ["ROC_LEVIR_SHA256"]
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    query = torch.randn(2, 4, 512, 800, device="cuda")
    for track in ("D", "N"):
        cpu_rng = torch.get_rng_state().clone()
        cuda_rng = [state.clone() for state in torch.cuda.get_rng_state_all()]
        model = anchor.build_anchor_changer(track, path, digest, 42)
        assert torch.equal(cpu_rng, torch.get_rng_state())
        assert all(torch.equal(before, after) for before, after in zip(cuda_rng, torch.cuda.get_rng_state_all()))
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(42)
            historical = changerex.ChangerEx(fusion="single_noex", in_channels=4)
            historical.init_opencd_weights()
            changerex.load_levir(historical, path, verbose=False)
        for name, parameter in model.backbone.named_parameters():
            torch.testing.assert_close(parameter, dict(historical.backbone.named_parameters())[name], atol=0, rtol=0)
        changerex.set_encoder_bn_policy(historical, model.bn_policy)
        historical.cuda().eval()
        with torch.no_grad():
            single_features = anchor.encode_query(historical.backbone, query[:1])
            legacy_features = historical.backbone(query[:1], torch.zeros_like(query[:1]))
        for single, paired in zip(single_features, legacy_features):
            torch.testing.assert_close(single, paired.chunk(2, dim=1)[0], atol=1e-5, rtol=1e-4)
        del historical, single_features, legacy_features
        model.cuda().train()
        counts = {}
        handles = []
        for name in ("stem", *model.backbone.res_layers):
            def count(module, inputs, output, name=name):
                counts[name] = counts.get(name, 0) + 1
            handles.append(getattr(model.backbone, name).register_forward_hook(count))
        norms = [module for module in model.modules() if isinstance(module, nn.modules.batchnorm._BatchNorm)]
        before = [module.num_batches_tracked.clone() for module in norms] if track == "N" else []
        projections = []
        if track == "N":
            base_forward = model.decode_head.base_forward
            def project_once(features):
                assert not projections, "absent reference was projected"
                projected = base_forward(features)
                projections.append(projected)
                return projected
            model.decode_head.base_forward = project_once
        output = model(query)
        if track == "N":
            assert len(projections) == 1
            model.decode_head.base_forward = base_forward
            del projections
        for handle in handles:
            handle.remove()
        assert all(counts[name] == 1 for name in ("stem", *model.backbone.res_layers))
        if track == "N":
            assert output["map_logit"].shape == (2, 1, 128, 200)
            assert all(module.num_batches_tracked == previous + 1 for module, previous in zip(norms, before))
        output["attack_logit"].sum().backward()
        assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters())
        model.eval()
        snapshot = {name: tensor.clone() for name, tensor in model.state_dict().items()}
        with torch.no_grad():
            solo = model(query[:1])
            if track == "N":
                together = model(query)
                reversed_batch = model(query.flip(0))
                for name in solo:
                    torch.testing.assert_close(solo[name][0], together[name][0], atol=1e-5, rtol=1e-4)
                    torch.testing.assert_close(solo[name][0], reversed_batch[name][-1], atol=1e-5, rtol=1e-4)
        assert all(torch.equal(tensor, snapshot[name]) for name, tensor in model.state_dict().items())
        del output, snapshot
        stream = io.BytesIO()
        torch.save(model.state_dict(), stream)
        del model
        stream.seek(0)
        replay = anchor.build_anchor_changer(track, path, digest, 42).cuda().eval()
        replay.load_state_dict(torch.load(stream, map_location="cpu", weights_only=True), strict=True)
        with torch.no_grad():
            replayed = replay(query[:1])
        for name in solo:
            torch.testing.assert_close(solo[name], replayed[name], atol=1e-5, rtol=1e-4)
        del replay, replayed, solo
