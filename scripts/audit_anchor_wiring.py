#!/usr/bin/env python3
"""Synthetic-input checks of the Protocol A model adaptation, without training."""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path
import sys
import traceback

import torch
from torch.nn import functional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from refer_or_clear.models.changerex import (
    Changer, ChangerEx, IA_ResNetV1c, INTERACTION_CFG, _quiet,
    load_levir, set_encoder_bn_policy,
)


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state_hash(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(str((tensor.shape, tensor.dtype)).encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def compare(report, name, actual, expected):
    difference = (actual - expected).abs()
    passed = torch.allclose(actual, expected, atol=1e-4, rtol=1e-4)
    report[name] = dict(passed=passed, max_absolute_difference=difference.max().item(),
                        mean_absolute_difference=difference.mean().item(),
                        atol=1e-4, rtol=1e-4)
    assert passed, f"{name}: {report[name]}"


def optimizer_coverage(model):
    pretrained, added = model.parameter_groups()
    actual = [id(parameter) for parameter in pretrained + added]
    expected = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    assert len(actual) == len(set(actual)), "duplicate optimizer parameter"
    assert set(actual) == expected, "missing or unexpected optimizer parameter"
    return dict(pretrained_tensors=len(pretrained), added_tensors=len(added),
                trainable_parameters=sum(parameter.numel() for parameter in model.parameters()
                                         if parameter.requires_grad))


def gradient_routes(model, inputs):
    output = model(inputs)
    image_loss = functional.binary_cross_entropy_with_logits(
        output["logit"], torch.zeros_like(output["logit"]))
    dense_loss = functional.binary_cross_entropy_with_logits(
        output["map"], torch.zeros_like(output["map"]))
    named = list(model.named_parameters())
    parameters = [parameter for _, parameter in named]
    result = {}
    for objective, loss in (("image_only", image_loss), ("dense_only", dense_loss)):
        gradients = torch.autograd.grad(loss, parameters, allow_unused=True,
                                        retain_graph=objective == "image_only")
        groups = {}
        for name, gradient in zip((name for name, _ in named), gradients):
            group = ("map_projection" if name.startswith("decode_head.conv_seg.")
                     else name.split(".")[0])
            entry = groups.setdefault(group, dict(tensors=0, connected=0, nonzero=0, norm_square=0.))
            entry["tensors"] += 1
            if gradient is not None:
                assert torch.isfinite(gradient).all(), f"nonfinite {objective} gradient: {name}"
                entry["connected"] += 1
                entry["nonzero"] += int(torch.count_nonzero(gradient).item() > 0)
                entry["norm_square"] += gradient.double().square().sum().item()
        for group, entry in groups.items():
            entry["norm"] = entry.pop("norm_square") ** 0.5
            expected_connected = 0 if objective == "dense_only" and group == "referral" else entry["tensors"]
            assert entry["connected"] == expected_connected, (objective, group, entry)
        assert groups["map_projection"]["norm"] > 0
        result[objective] = groups
    return result


def run_checks(args, report):
    assert torch.cuda.is_available(), "run this diagnostic on a GPU compute node"
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    report["environment"] = dict(torch=torch.__version__, cuda=torch.version.cuda,
                                gpu=torch.cuda.get_device_name(0), amp=False,
                                matmul_tf32=False, cudnn_tf32=False)
    report["checkpoint"] = dict(path=str(args.checkpoint), sha256=file_hash(args.checkpoint))
    report["sources"] = {str(path): file_hash(path) for path in (
        Path(__file__), ROOT / "src/refer_or_clear/models/changerex.py",
        Path(inspect.getfile(Changer)), Path(inspect.getfile(IA_ResNetV1c)))}
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    weights = checkpoint.get("state_dict", checkpoint)
    model = ChangerEx(fusion="difference").eval().cuda()
    missing, unexpected = load_levir(model, args.checkpoint, verbose=False)
    report["loading"] = dict(missing=missing, unexpected=unexpected)
    assert set(missing) == {f"referral.{layer}.{kind}" for layer in ("fc1", "fc2")
                            for kind in ("weight", "bias")}
    assert not unexpected
    assert torch.count_nonzero(model.backbone.stem[0].weight[:, 3:]).item() == 0
    report["optimizer"] = optimizer_coverage(model)
    before = state_hash(model)
    with _quiet():
        original_backbone = IA_ResNetV1c(
            depth=18, in_channels=3, interaction_cfg=INTERACTION_CFG, num_stages=4,
            out_indices=(0, 1, 2, 3), dilations=(1, 1, 1, 1), strides=(1, 2, 2, 2),
            norm_cfg=dict(type="BN", requires_grad=True), norm_eval=False,
            style="pytorch", contract_dilation=True).cuda()
        original_head = Changer(
            in_channels=[64, 128, 256, 512], in_index=[0, 1, 2, 3], channels=128,
            dropout_ratio=0.1, num_classes=2, out_channels=2,
            norm_cfg=dict(type="BN", requires_grad=True), align_corners=False).cuda()
    original_backbone.eval()
    original_head.eval()
    original_backbone.load_state_dict({name.removeprefix("backbone."): value
                                      for name, value in weights.items() if name.startswith("backbone.")},
                                     strict=True)
    original_head.load_state_dict({name.removeprefix("decode_head."): value
                                  for name, value in weights.items() if name.startswith("decode_head.")},
                                 strict=True)
    report["parity"] = {}
    with torch.no_grad():
        for height, width in ((128, 160), (512, 800)):
            inputs = torch.randn(1, 8, height, width, device="cuda")
            query, reference = model.split(inputs)
            original_features = original_backbone(query[:, :3], reference[:, :3])
            adapted_features = model.backbone(query, reference)
            original_logits = original_head(original_features)
            _, adapted_logits = model.decode_head(adapted_features)
            same_feature_logits = original_head(adapted_features)
            compare(report["parity"], f"decoder_{height}x{width}", adapted_logits,
                    same_feature_logits[:, 1:2] - same_feature_logits[:, :1])
            compare(report["parity"], f"end_to_end_{height}x{width}", adapted_logits,
                    original_logits[:, 1:2] - original_logits[:, :1])
            for stage, (actual, expected) in enumerate(zip(adapted_features, original_features)):
                compare(report["parity"], f"backbone_{height}x{width}_stage{stage}", actual, expected)
        ramp = torch.arange(80, device="cuda", dtype=torch.float32).reshape(1, 1, 8, 10)
        zero_flow = torch.zeros(1, 2, 8, 10, device="cuda")
        displacement = torch.ones_like(zero_flow)
        zero_difference = model.decode_head.neck_layer.warp(ramp, zero_flow) - ramp
        shifted_difference = model.decode_head.neck_layer.warp(ramp, displacement) - ramp
        report["identical_feature_warp"] = dict(
            zero_flow_max_difference=zero_difference.abs().max().item(),
            nonzero_flow_max_difference=shifted_difference.abs().max().item())
        assert zero_difference.abs().max() < 1e-4
        assert shifted_difference.abs().max() > 1
    assert before == state_hash(model), "parity checks changed model state"
    del original_head, original_backbone, model
    torch.cuda.empty_cache()
    control = ChangerEx(fusion="single_noex").eval().cuda()
    load_levir(control, args.checkpoint, verbose=False)
    set_encoder_bn_policy(control, "batch_stats")
    report["control_optimizer"] = optimizer_coverage(control)
    before = state_hash(control)
    inputs = torch.randn(1, 8, 128, 160, device="cuda")
    alternate = inputs.clone()
    alternate[:, 4:] = torch.randn_like(alternate[:, 4:]) * 10
    report["control_reference_invariance"] = {}
    with torch.no_grad():
        original_output, changed_output = control(inputs), control(alternate)
        for name in ("logit", "map"):
            compare(report["control_reference_invariance"], name, original_output[name], changed_output[name])
            assert torch.equal(original_output[name], changed_output[name])
    report["control_gradients"] = gradient_routes(control, inputs)
    report["control_state_unchanged"] = before == state_hash(control)
    assert report["control_state_unchanged"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    report = dict(complete=False, scope="synthetic inputs only; no dataset or optimizer steps")
    try:
        run_checks(args, report)
        report["complete"] = True
    except Exception:
        report["error"] = traceback.format_exc()
        raise
    finally:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(json.dumps(report, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
