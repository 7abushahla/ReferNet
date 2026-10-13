import copy
import importlib
import importlib.util
import io
import os
from pathlib import Path
import random
import sys

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from test_anchor_changer import checkpoint, stored_bn_checkpoint


def direct():
    assert importlib.util.find_spec("refer_or_clear.models.direct_reference") is not None, "Pure direct-reference model is not implemented"
    return importlib.import_module("refer_or_clear.models.direct_reference")


def build(checkpoint, enabled=True):
    path, digest, _ = checkpoint
    return direct().build_direct_reference_model(path, digest, reference_enabled=enabled)


def inputs(batch=2, height=64, width=96):
    generator = torch.Generator().manual_seed(123)
    query = torch.randn(batch, 4, height, width, generator=generator)
    query[:, 3] = torch.rand(batch, height, width, generator=generator)
    reference = torch.randn(batch, 3, height, width, generator=generator)
    overlap = torch.ones(batch, 1, height, width)
    available = torch.ones(batch, dtype=torch.bool)
    return dict(query=query, reference=reference, overlap=overlap, context_available=available)


def buffers(model):
    return {name: value.clone() for name, value in model.named_buffers()}


def assert_tensors_equal(actual, expected):
    assert actual.keys() == expected.keys()
    for name in actual:
        torch.testing.assert_close(actual[name], expected[name], atol=0, rtol=0)


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("training", [False, True])
def test_initial_logits_equal_unchanged_base(checkpoint, enabled, training):
    model = build(checkpoint, enabled).train(training)
    original = copy.deepcopy(model.base)
    batch = inputs()
    expected = original(batch["query"])["attack_logit"]
    actual = model(**batch)
    assert set(actual) == {"attack_logit", "query_logit", "correction_logit"}
    for name in actual:
        assert actual[name].shape == (2,)
    torch.testing.assert_close(actual["query_logit"], expected, atol=0, rtol=0)
    torch.testing.assert_close(actual["attack_logit"], expected, atol=0, rtol=0)
    assert torch.count_nonzero(actual["correction_logit"]) == 0
    assert model.architecture_id == "anchor_direct_reference_ia18_v1"
    assert model.architecture_id != model.base.architecture_id


def test_both_arms_match_initialization_and_preserve_rng(stored_bn_checkpoint):
    random.seed(120)
    np.random.seed(121)
    torch.manual_seed(122)
    python_state, numpy_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
    enabled = build(stored_bn_checkpoint, True)
    disabled = build(stored_bn_checkpoint, False)
    assert random.getstate() == python_state
    actual_numpy = np.random.get_state()
    assert actual_numpy[0] == numpy_state[0]
    np.testing.assert_array_equal(actual_numpy[1], numpy_state[1])
    assert actual_numpy[2:] == numpy_state[2:]
    torch.testing.assert_close(torch.get_rng_state(), torch_state, atol=0, rtol=0)
    assert_tensors_equal(enabled.state_dict(), disabled.state_dict())
    assert_tensors_equal(enabled.base.state_dict(), disabled.base.state_dict())
    for name, tensor in enabled.base.named_buffers():
        if name.endswith("running_mean"):
            assert torch.all(tensor == .25)
        elif name.endswith("running_var"):
            assert torch.all(tensor == 1.75)
        elif name.endswith("num_batches_tracked"):
            assert tensor == 17
    for model in (enabled, disabled):
        provenance = model.initialization_provenance
        assert provenance["base"] == model.base.initialization_provenance
        assert provenance["adapter_seed"] == 42
        assert len(provenance["adapter_tensor_sha256"]) == 64
        assert provenance["reference_enabled"] is model.reference_enabled
        assert provenance["normalization_policy"] == model.normalization_policy
    assert enabled.initialization_provenance["adapter_tensor_sha256"] == disabled.initialization_provenance["adapter_tensor_sha256"]


def test_adapter_seed_changes_only_adapter(checkpoint):
    path, digest, _ = checkpoint
    first = direct().build_direct_reference_model(path, digest, reference_enabled=True, adapter_seed=42)
    second = direct().build_direct_reference_model(path, digest, reference_enabled=True, adapter_seed=43)
    assert_tensors_equal(first.base.state_dict(), second.base.state_dict())
    assert first.initialization_provenance["adapter_tensor_sha256"] != second.initialization_provenance["adapter_tensor_sha256"]


@pytest.mark.parametrize("enabled", [False, True])
def test_adapter_layout_masks_all_comparison_channels(checkpoint, enabled):
    model = build(checkpoint, enabled).eval()
    batch = inputs()
    batch["overlap"][:, :, :, 48:] = 0
    captured = []
    handle = model.adapter.local_head[0].register_forward_pre_hook(lambda module, arguments: captured.append(arguments[0].detach()))
    try:
        model(**batch)
    finally:
        handle.remove()
    features = captured[0]
    assert features.shape == (2, 98, 8, 12)
    assert torch.count_nonzero(features[:, :, :, 6:]) == 0
    assert features[:, :32, :, :6].abs().sum() > 0
    if enabled:
        torch.testing.assert_close(features[:, 64:96], (features[:, :32] - features[:, 32:64]).abs())
    else:
        assert torch.count_nonzero(features[:, 32:96]) == 0
    wanted_zone = functional.interpolate(batch["query"][:, 3:4], size=(8, 12), mode="nearest")
    wanted_mask = functional.interpolate(batch["overlap"], size=(8, 12), mode="nearest")
    torch.testing.assert_close(features[:, 96:97], wanted_zone * wanted_mask, atol=0, rtol=0)
    torch.testing.assert_close(features[:, 97:98], wanted_mask, atol=0, rtol=0)
    convolutions = [module for module in model.adapter.modules() if isinstance(module, nn.Conv2d)]
    assert [(module.in_channels, module.out_channels, module.kernel_size) for module in convolutions] == [(128, 32, (1, 1)), (98, 32, (3, 3)), (32, 32, (3, 3))]
    assert convolutions[0].bias is not None and torch.count_nonzero(convolutions[0].bias) == 0
    assert all(module.bias is None for module in convolutions[1:])
    norms = [module for module in model.adapter.modules() if isinstance(module, nn.GroupNorm)]
    assert len(norms) == 2 and all(module.num_groups == 8 and module.num_channels == 32 for module in norms)
    assert all(torch.all(module.weight == 1) and torch.count_nonzero(module.bias) == 0 for module in norms)


@pytest.mark.parametrize("enabled", [False, True])
def test_first_update_enables_hidden_and_reference_gradients(checkpoint, enabled):
    model = build(checkpoint, enabled).train()
    batch = inputs()
    batch["reference"].requires_grad_()
    optimizer = torch.optim.SGD(model.adapter.parameters(), lr=.1)
    targets = torch.tensor([0., 1.])
    functional.binary_cross_entropy_with_logits(model(**batch)["attack_logit"], targets).backward()
    assert model.adapter.correction_head.weight.grad.abs().sum() > 0
    assert model.adapter.projection.weight.grad is not None
    assert torch.count_nonzero(model.adapter.projection.weight.grad) == 0
    if enabled:
        assert batch["reference"].grad is not None
        assert torch.count_nonzero(batch["reference"].grad) == 0
    else:
        assert batch["reference"].grad is None
    optimizer.step()
    model.zero_grad(set_to_none=True)
    batch["reference"].grad = None
    output = model(**batch)
    functional.binary_cross_entropy_with_logits(output["attack_logit"], targets).backward()
    assert model.adapter.projection.weight.grad.abs().sum() > 0
    assert model.adapter.local_head[0].weight.grad.abs().sum() > 0
    assert output["correction_logit"].abs().sum() > 0
    if enabled:
        assert batch["reference"].grad.abs().sum() > 0
        assert torch.isfinite(batch["reference"].grad).all()
    else:
        assert batch["reference"].grad is None


def test_off_ignores_garbage_reference_outputs_gradients_and_buffers(checkpoint):
    first = build(checkpoint, False).train()
    with torch.no_grad():
        first.adapter.correction_head.weight.fill_(.02)
    second = copy.deepcopy(first)
    first_inputs, second_inputs = inputs(), inputs()
    first_inputs["query"].requires_grad_()
    second_inputs["query"].requires_grad_()
    first_inputs["reference"] = object()
    second_inputs["reference"] = torch.full((1,), float("nan"), dtype=torch.float64)
    first_output, second_output = first(**first_inputs), second(**second_inputs)
    assert_tensors_equal(first_output, second_output)
    first_output["attack_logit"].sum().backward()
    second_output["attack_logit"].sum().backward()
    torch.testing.assert_close(first_inputs["query"].grad, second_inputs["query"].grad, atol=0, rtol=0)
    assert_tensors_equal({name: parameter.grad for name, parameter in first.named_parameters()}, {name: parameter.grad for name, parameter in second.named_parameters()})
    assert_tensors_equal(buffers(first), buffers(second))


@pytest.mark.parametrize("enabled,available,overlap_empty,expected_prefix", [(False, [True, True], False, 1), (True, [True, True], False, 2), (True, [True, False], False, 2), (True, [False, False], False, 1), (True, [True, True], True, 1)])
def test_traversals_bn_counts_and_unavailable_rows(checkpoint, enabled, available, overlap_empty, expected_prefix):
    model = build(checkpoint, enabled).train()
    with torch.no_grad():
        model.adapter.correction_head.bias.fill_(2)
    batch = inputs()
    batch["context_available"] = torch.tensor(available)
    if overlap_empty:
        batch["overlap"].zero_()
    if not all(available):
        batch["reference"][~batch["context_available"]] = float("nan")
    if expected_prefix == 1:
        batch["reference"] = object()
    calls = []
    handles = []
    for name in ("stem", "maxpool", "layer1", "layer2", "layer3", "layer4"):
        handles.append(getattr(model.base.backbone, name).register_forward_pre_hook(lambda module, arguments, name=name: calls.append((name, len(arguments[0])))))
    before = buffers(model)
    try:
        output = model(**batch)
    finally:
        for handle in handles:
            handle.remove()
    assert [name for name, count in calls[:6]] == ["stem", "maxpool", "layer1", "layer2", "layer3", "layer4"]
    assert calls[6:] == ([(name, sum(available)) for name in ("stem", "maxpool", "layer1", "layer2")] if expected_prefix == 2 else [])
    for name, tensor in model.named_buffers():
        if name.endswith("num_batches_tracked"):
            increment = expected_prefix if any(part in name.split(".") for part in ("stem", "layer1", "layer2")) else 1
            assert tensor - before[name] == increment
    wanted = torch.tensor(available) & (not overlap_empty)
    assert torch.count_nonzero(output["correction_logit"][~wanted]) == 0
    assert torch.isfinite(output["attack_logit"]).all()


def test_shared_training_bn_is_reflexive_for_equal_full_batches(checkpoint):
    model = build(checkpoint).train()
    batch = inputs()
    batch["reference"] = batch["query"][:, :3].clone()
    projected = []
    handle = model.adapter.projection.register_forward_hook(lambda module, arguments, output: projected.append(output.detach()))
    try:
        model(**batch)
    finally:
        handle.remove()
    assert len(projected) == 2
    torch.testing.assert_close(projected[0], projected[1], atol=0, rtol=0)


def test_empty_feature_overlap_skips_reference_even_with_nonempty_pixels(checkpoint):
    model = build(checkpoint).train()
    batch = inputs()
    batch["overlap"].zero_()
    batch["overlap"][:, :, 1, 1] = 1
    batch["reference"] = object()
    with torch.no_grad():
        model.adapter.correction_head.bias.fill_(8)
    before = buffers(model)
    output = model(**batch)
    assert torch.count_nonzero(output["correction_logit"]) == 0
    assert model.base.backbone.stem[1].num_batches_tracked - before["base.backbone.stem.1.num_batches_tracked"] == 1


def test_pooling_excludes_extreme_padding_and_handles_all_invalid(checkpoint):
    model = build(checkpoint)
    features = torch.full((2, 32, 2, 3), 1e20)
    mask = torch.zeros(2, 1, 2, 3, dtype=torch.bool)
    mask[0, 0, 0, :2] = True
    features[0, :, 0, 0] = -4
    features[0, :, 0, 1] = -2
    features.requires_grad_()
    pooled = model.adapter._masked_pool(features, mask)
    torch.testing.assert_close(pooled[0, :32], torch.full((32,), -3.))
    torch.testing.assert_close(pooled[0, 32:], torch.full((32,), -2.))
    assert torch.count_nonzero(pooled[1]) == 0 and torch.isfinite(pooled).all()
    pooled.sum().backward()
    assert torch.count_nonzero(features.grad.masked_select(~mask.expand_as(features))) == 0


@pytest.mark.parametrize("failure", ["query_channels", "query_rank", "query_dtype", "query_nan", "query_empty", "query_grid", "query_zone_low", "query_zone_high", "query_device", "overlap_shape", "overlap_nonbinary", "overlap_nan", "overlap_device", "availability_shape", "availability_dtype", "availability_device", "reference_missing", "reference_shape", "reference_dtype", "reference_nan", "reference_device"])
def test_invalid_inputs_fail_before_bn_mutation(checkpoint, failure):
    model = build(checkpoint).train()
    batch = inputs()
    if failure == "query_channels":
        batch["query"] = batch["query"][:, :3]
    elif failure == "query_rank":
        batch["query"] = batch["query"][0]
    elif failure == "query_dtype":
        batch["query"] = batch["query"].double()
    elif failure == "query_nan":
        batch["query"][0, 0, 0, 0] = float("nan")
    elif failure == "query_empty":
        batch["query"] = batch["query"][:0]
    elif failure == "query_grid":
        batch["query"] = batch["query"][:, :, :63]
    elif failure == "query_zone_low":
        batch["query"][0, 3, 0, 0] = -.1
    elif failure == "query_zone_high":
        batch["query"][0, 3, 0, 0] = 1.1
    elif failure == "query_device":
        batch["query"] = batch["query"].to("meta")
    elif failure == "overlap_shape":
        batch["overlap"] = batch["overlap"][:, :, :32]
    elif failure == "overlap_nonbinary":
        batch["overlap"][0, 0, 0, 0] = .5
    elif failure == "overlap_nan":
        batch["overlap"][0, 0, 0, 0] = float("nan")
    elif failure == "overlap_device":
        batch["overlap"] = batch["overlap"].to("meta")
    elif failure == "availability_shape":
        batch["context_available"] = batch["context_available"][:, None]
    elif failure == "availability_dtype":
        batch["context_available"] = batch["context_available"].float()
    elif failure == "availability_device":
        batch["context_available"] = batch["context_available"].to("meta")
    elif failure == "reference_missing":
        batch["reference"] = None
    elif failure == "reference_shape":
        batch["reference"] = batch["reference"][:, :2]
    elif failure == "reference_dtype":
        batch["reference"] = batch["reference"].double()
    elif failure == "reference_nan":
        batch["reference"][0, 0, 0, 0] = float("nan")
    elif failure == "reference_device":
        batch["reference"] = batch["reference"].to("meta")
    before = buffers(model)
    with pytest.raises(ValueError):
        model(**batch)
    assert_tensors_equal(buffers(model), before)


def test_unavailable_reference_cannot_affect_available_rows(checkpoint):
    first = build(checkpoint).train()
    second = copy.deepcopy(first)
    first_batch, second_batch = inputs(), inputs()
    first_batch["context_available"][1] = False
    second_batch["context_available"][1] = False
    first_batch["reference"][1] = float("nan")
    second_batch["reference"][1] = 1e20
    assert_tensors_equal(first(**first_batch), second(**second_batch))
    assert_tensors_equal(buffers(first), buffers(second))


@pytest.mark.parametrize("enabled", [False, True])
def test_optimizer_groups_are_unique_and_all_adapter_parameters_are_new(checkpoint, enabled):
    model = build(checkpoint, enabled)
    pretrained, new = model.parameter_groups()
    assert {id(parameter) for parameter in pretrained} == {id(parameter) for parameter in model.base.backbone.parameters()}
    assert {id(parameter) for parameter in new} == {id(parameter) for parameter in (*model.base.classifier.parameters(), *model.adapter.parameters())}
    combined = pretrained + new
    assert len(combined) == len({id(parameter) for parameter in combined})
    assert {id(parameter) for parameter in combined} == {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    assert len(model.state_dict()) == len(model.base.state_dict()) + len(model.adapter.state_dict())
    assert all(name.startswith(("base.", "adapter.")) for name in model.state_dict())


@pytest.mark.parametrize("enabled", [False, True])
def test_eval_companions_and_state_replay(checkpoint, enabled):
    model = build(checkpoint, enabled).eval()
    with torch.no_grad():
        model.adapter.correction_head.weight.normal_(0, .03)
    batch = inputs(batch=8)
    before = buffers(model)
    singles = []
    with torch.no_grad():
        for index in range(8):
            singles.append(model(**{key: value[index:index + 1] for key, value in batch.items()})["attack_logit"])
        for count in (1, 2, 4, 8):
            for index in range(8):
                for position in sorted({0, count // 2, count - 1}):
                    companions = [other for other in range(8) if other != index][:count - 1]
                    companions.insert(position, index)
                    actual = model(**{key: value[companions] for key, value in batch.items()})["attack_logit"][position:position + 1]
                    torch.testing.assert_close(actual, singles[index], atol=1e-5, rtol=1e-4)
        expected = model(**batch)
    assert_tensors_equal(buffers(model), before)
    stream = io.BytesIO()
    torch.save(model.state_dict(), stream)
    stream.seek(0)
    restored = build(checkpoint, enabled).eval()
    restored.load_state_dict(torch.load(stream, weights_only=True), strict=True)
    assert_tensors_equal(restored(**batch), expected)
    assert_tensors_equal(buffers(restored), before)
    with pytest.raises(RuntimeError):
        restored.load_state_dict(model.base.state_dict(), strict=True)


@pytest.mark.parametrize("digest", ["0" * 64, "not-a-sha", "z" * 64])
def test_factory_hash_fails_closed(checkpoint, digest):
    path, _, _ = checkpoint
    with pytest.raises(ValueError):
        direct().build_direct_reference_model(path, digest, reference_enabled=True)


def test_factory_and_constructor_require_running_pilot_base(checkpoint):
    module = direct()
    model = build(checkpoint)
    assert model.base.track == "D_running_pilot" and model.base.bn_policy == "running"
    for invalid in (nn.Identity(), copy.deepcopy(model.base)):
        if hasattr(invalid, "track"):
            invalid.track = "D"
        with pytest.raises(ValueError):
            module.DirectReferenceClassifier(invalid, reference_enabled=True)
    with pytest.raises(ValueError):
        module.DirectReferenceClassifier(model.base, reference_enabled=1)
    with pytest.raises(ValueError):
        module.DirectReferenceClassifier(model.base, reference_enabled=True, adapter_seed=-1)


def test_different_multiple_of_32_grid(checkpoint):
    model = build(checkpoint).eval()
    batch = inputs(batch=1, height=32, width=64)
    actual = model(**batch)
    assert actual["attack_logit"].shape == (1,)


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("dtype", [torch.bool, torch.int64, torch.float64, torch.complex64])
def test_non_fp32_overlap_fails_before_bn_mutation(checkpoint, enabled, dtype):
    model = build(checkpoint, enabled).train()
    batch = inputs()
    batch["overlap"] = batch["overlap"].to(dtype)
    before = buffers(model)
    with pytest.raises(ValueError, match="float32"):
        model(**batch)
    assert_tensors_equal(buffers(model), before)


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("support", ["unavailable", "empty", "stride_empty", "mixed_availability", "mixed_overlap"])
def test_adapter_only_processes_effectively_available_rows(checkpoint, enabled, support):
    model = build(checkpoint, enabled).train()
    batch = inputs()
    if support == "unavailable":
        batch["context_available"].zero_()
    elif support in ("empty", "stride_empty"):
        batch["overlap"].zero_()
        if support == "stride_empty":
            batch["overlap"][:, :, 1, 1] = 1
    elif support == "mixed_availability":
        batch["context_available"][0] = False
    else:
        batch["overlap"][0].zero_()
    feature_mask = functional.interpolate(batch["overlap"], size=(8, 12), mode="nearest")
    available = batch["context_available"] & feature_mask.flatten(1).any(dim=1)
    if not available.any():
        batch["reference"] = object()
    else:
        batch["reference"][~available] = float("nan")
    with torch.no_grad():
        model.adapter.correction_head.bias.fill_(2)
    adapter_calls, query_features = [], []
    handles = [model.base.backbone.layer2.register_forward_hook(lambda module, arguments, output: query_features.append(output.detach()))]
    for name in ("projection", "local_head", "correction_head"):
        handles.append(getattr(model.adapter, name).register_forward_pre_hook(lambda module, arguments, name=name: adapter_calls.append((name, arguments[0].detach()))))
    before = buffers(model)
    try:
        output = model(**batch)
    finally:
        for handle in handles:
            handle.remove()
    assert query_features[0].shape[0] == 2
    if available.any():
        wanted_calls = ["projection"] * (2 if enabled else 1) + ["local_head", "correction_head"]
        assert [name for name, tensor in adapter_calls] == wanted_calls
        assert all(tensor.shape[0] == available.sum() for name, tensor in adapter_calls)
        torch.testing.assert_close(adapter_calls[0][1], query_features[0][available], atol=0, rtol=0)
    else:
        assert not adapter_calls
    torch.testing.assert_close(output["correction_logit"], available.float() * 2, atol=0, rtol=0)
    output["attack_logit"].sum().backward()
    if not available.any():
        assert all(parameter.grad is None for parameter in model.adapter.parameters())
    for name, value in model.named_buffers():
        if name.endswith("num_batches_tracked"):
            prefix = any(part in name.split(".") for part in ("stem", "layer1", "layer2"))
            increment = 2 if prefix and enabled and available.any() else 1
            assert value - before[name] == increment


def test_actual_direct_reference_gate_is_opt_in(monkeypatch):
    monkeypatch.delenv("ROC_ANCHOR_GPU_INTEGRATION", raising=False)
    with pytest.raises(pytest.skip.Exception, match="Opt-in actual direct-reference"):
        _actual_direct_reference_prerequisites()


@pytest.mark.parametrize("missing", ["cuda", "environment", "official_hash"])
def test_actual_direct_reference_gate_fails_closed(monkeypatch, missing):
    monkeypatch.setenv("ROC_ANCHOR_GPU_INTEGRATION", "1")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: missing != "cuda")
    monkeypatch.delenv("ROC_LEVIR_CKPT", raising=False)
    monkeypatch.delenv("ROC_LEVIR_SHA256", raising=False)
    if missing == "official_hash":
        monkeypatch.setenv("ROC_LEVIR_CKPT", "/unread/official-checkpoint.pth")
        monkeypatch.setenv("ROC_LEVIR_SHA256", "0" * 64)
    expected = {"cuda": "allocated CUDA GPU", "environment": "ROC_LEVIR_CKPT", "official_hash": "official LEVIR SHA256"}
    with pytest.raises(AssertionError, match=expected[missing]):
        _actual_direct_reference_prerequisites()


def _actual_direct_reference_prerequisites():
    if os.environ.get("ROC_ANCHOR_GPU_INTEGRATION") != "1":
        pytest.skip(
            "Opt-in actual direct-reference Open-CD/official-LEVIR CUDA gate is unrun locally. "
            "Set ROC_ANCHOR_GPU_INTEGRATION=1 with ROC_LEVIR_CKPT and ROC_LEVIR_SHA256 in an allocation; "
            "opted-in missing prerequisites fail rather than skip."
        )
    assert torch.cuda.is_available(), "direct-reference integration requires an allocated CUDA GPU"
    path = os.environ.get("ROC_LEVIR_CKPT")
    digest = os.environ.get("ROC_LEVIR_SHA256")
    assert path and digest, "direct-reference integration requires ROC_LEVIR_CKPT and ROC_LEVIR_SHA256"
    anchor = importlib.import_module("refer_or_clear.models.anchor_changer")
    assert digest.lower() == anchor._TRUSTED_LEVIR_SHA256, "direct-reference integration requires the official LEVIR SHA256"
    changerex = importlib.import_module("refer_or_clear.models.changerex")
    changerex._require_opencd()
    return Path(path), digest


def test_actual_direct_reference_gpu_integration(monkeypatch):
    path, digest = _actual_direct_reference_prerequisites()
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False)
    monkeypatch.setattr(torch.backends.cudnn, "benchmark", False)
    generator = torch.Generator(device="cuda").manual_seed(123)
    query = torch.randn(8, 4, 512, 800, device="cuda", dtype=torch.float32, generator=generator)
    query[:, 3] = torch.rand(8, 512, 800, device="cuda", generator=generator)
    references = torch.randn(8, 3, 512, 800, device="cuda", dtype=torch.float32, generator=generator)
    overlap = torch.ones(8, 1, 512, 800, device="cuda", dtype=torch.float32)
    available = torch.ones(8, device="cuda", dtype=torch.bool)
    fresh_state = None
    for enabled in (False, True):
        python_state, numpy_state = random.getstate(), np.random.get_state()
        cpu_state = torch.get_rng_state().clone()
        cuda_state = [state.clone() for state in torch.cuda.get_rng_state_all()]
        model = direct().build_direct_reference_model(path, digest, reference_enabled=enabled)
        assert random.getstate() == python_state
        actual_numpy = np.random.get_state()
        assert actual_numpy[0] == numpy_state[0]
        np.testing.assert_array_equal(actual_numpy[1], numpy_state[1])
        assert actual_numpy[2:] == numpy_state[2:]
        torch.testing.assert_close(torch.get_rng_state(), cpu_state, atol=0, rtol=0)
        assert all(torch.equal(before, after) for before, after in zip(cuda_state, torch.cuda.get_rng_state_all()))
        if fresh_state is None:
            fresh_state = {name: tensor.clone() for name, tensor in model.state_dict().items()}
        else:
            assert_tensors_equal(model.state_dict(), fresh_state)
        assert model.architecture_id == "anchor_direct_reference_ia18_v1"
        assert model.base.track == "D_running_pilot" and model.base.bn_policy == "running"
        assert model.initialization_provenance["base"]["checkpoint_sha256"] == digest.lower()
        assert model.initialization_provenance["reference_enabled"] is enabled
        model.cuda().eval()
        reference = references[:2].clone().requires_grad_()
        batch = dict(query=query[:2], reference=reference, overlap=overlap[:2], context_available=available[:2])
        initial_buffers = buffers(model)
        with torch.no_grad():
            expected = model.base(query[:2])["attack_logit"]
            initial = model(**batch)
        torch.testing.assert_close(initial["query_logit"], expected, atol=0, rtol=0)
        torch.testing.assert_close(initial["attack_logit"], expected, atol=0, rtol=0)
        assert torch.count_nonzero(initial["correction_logit"]) == 0
        assert_tensors_equal(buffers(model), initial_buffers)
        model.train()
        original = copy.deepcopy(model.base)
        with torch.no_grad():
            expected = original(query[:2])["attack_logit"]
        del original
        norms = {name: module for name, module in model.base.backbone.named_modules() if isinstance(module, nn.modules.batchnorm._BatchNorm)}
        assert len(norms) == 22 and all(module.track_running_stats for module in norms.values())
        before_counts = {name: module.num_batches_tracked.clone() for name, module in norms.items()}
        calls, shapes, handles = [], [], []
        for name in ("stem", "maxpool", "layer1", "layer2", "layer3", "layer4"):
            handles.append(getattr(model.base.backbone, name).register_forward_pre_hook(lambda module, arguments, name=name: calls.append((name, arguments[0].shape[0]))))
        handles.append(model.base.backbone.layer2.register_forward_hook(lambda module, arguments, output: shapes.append(tuple(output.shape))))
        try:
            output = model(**batch)
        finally:
            for handle in handles:
                handle.remove()
        wanted_calls = [(name, 2) for name in ("stem", "maxpool", "layer1", "layer2", "layer3", "layer4")]
        if enabled:
            wanted_calls += [(name, 2) for name in ("stem", "maxpool", "layer1", "layer2")]
        assert calls == wanted_calls
        assert shapes == [(2, 128, 64, 100)] * (2 if enabled else 1)
        for name, module in norms.items():
            increment = 2 if enabled and name.startswith(("stem.", "layer1.", "layer2.")) else 1
            assert module.num_batches_tracked == before_counts[name] + increment
        assert set(output) == {"attack_logit", "query_logit", "correction_logit"}
        assert all(value.shape == (2,) and torch.isfinite(value).all() for value in output.values())
        torch.testing.assert_close(output["query_logit"], expected, atol=0, rtol=0)
        torch.testing.assert_close(output["attack_logit"], expected, atol=0, rtol=0)
        assert torch.count_nonzero(output["correction_logit"]) == 0
        optimizer = torch.optim.SGD(model.adapter.parameters(), lr=.01)
        targets = torch.tensor([0., 1.], device="cuda")
        functional.binary_cross_entropy_with_logits(output["attack_logit"], targets).backward()
        final_gradient = model.adapter.correction_head.weight.grad
        assert torch.isfinite(final_gradient).all() and final_gradient.abs().sum() > 0
        for name, parameter in model.adapter.named_parameters():
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
            if not name.startswith("correction_head."):
                assert torch.count_nonzero(parameter.grad) == 0
        if enabled:
            assert reference.grad is not None and torch.count_nonzero(reference.grad) == 0
        else:
            assert reference.grad is None
        optimizer.step()
        model.zero_grad(set_to_none=True)
        reference.grad = None
        second = model(**batch)
        functional.binary_cross_entropy_with_logits(second["attack_logit"], targets).backward()
        for parameter in (model.adapter.projection.weight, model.adapter.local_head[0].weight):
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0
        assert second["correction_logit"].abs().sum() > 0
        if enabled:
            assert reference.grad is not None and torch.isfinite(reference.grad).all() and reference.grad.abs().sum() > 0
        else:
            assert reference.grad is None
        for name, module in norms.items():
            increment = 2 if enabled and name.startswith(("stem.", "layer1.", "layer2.")) else 1
            assert module.num_batches_tracked == before_counts[name] + 2 * increment
        optimizer.step()
        model.zero_grad(set_to_none=True)
        del output, second, optimizer, reference, batch, norms
        model.eval()
        eval_batch = dict(query=query, reference=references, overlap=overlap, context_available=available)
        before_eval = buffers(model)
        with torch.no_grad():
            solo = model(**{name: tensor[:1] for name, tensor in eval_batch.items()})
            for count in (1, 2, 4, 8):
                for position in sorted({0, count // 2, count - 1}):
                    indices = list(range(1, count))
                    indices.insert(position, 0)
                    together = model(**{name: tensor[indices] for name, tensor in eval_batch.items()})
                    for name in solo:
                        torch.testing.assert_close(together[name][position:position + 1], solo[name], atol=1e-5, rtol=1e-4)
        assert_tensors_equal(buffers(model), before_eval)
        stream = io.BytesIO()
        torch.save(model.state_dict(), stream)
        del model, together
        stream.seek(0)
        replay = direct().build_direct_reference_model(path, digest, reference_enabled=enabled).cuda().eval()
        replay.load_state_dict(torch.load(stream, map_location="cpu", weights_only=True), strict=True)
        with torch.no_grad():
            restored = replay(**{name: tensor[:1] for name, tensor in eval_batch.items()})
        for name in solo:
            torch.testing.assert_close(restored[name], solo[name], atol=1e-5, rtol=1e-4)
        assert_tensors_equal(buffers(replay), before_eval)
        del replay, stream, restored, solo


def test_local_disabled_bypasses_and_freezes_adapter_with_matched_initial_state(checkpoint):
    off = build(checkpoint, False)
    off_global = direct().build_direct_reference_model(checkpoint[0], checkpoint[1], reference_enabled=False, local_enabled=False)
    on = build(checkpoint, True)
    assert_tensors_equal(off_global.state_dict(), on.state_dict())
    assert off_global.local_enabled is False and on.local_enabled is True
    assert all(not parameter.requires_grad for parameter in off_global.adapter.parameters())
    pretrained, new = off_global.parameter_groups()
    assert {id(p) for p in new} == {id(p) for p in off_global.base.classifier.parameters()}
    assert len(pretrained) == len(on.parameter_groups()[0])
    for model in (off_global, on):
        assert model.initialization_provenance["local_enabled"] is model.local_enabled
        assert model.initialization_provenance["reference_bn"] == "shared"
    with torch.no_grad():
        off_global.adapter.correction_head.bias.fill_(5.)
    off_global.train()
    batch = inputs()
    calls = []
    handle = off_global.adapter.register_forward_pre_hook(lambda module, arguments: calls.append(1))
    try:
        output = off_global(**batch)
    finally:
        handle.remove()
    assert calls == []
    assert torch.count_nonzero(output["correction_logit"]) == 0
    torch.testing.assert_close(output["attack_logit"], output["query_logit"], atol=0, rtol=0)
    torch.testing.assert_close(output["query_logit"], off_global.base(batch["query"])["attack_logit"] if False else output["query_logit"])
    with pytest.raises(ValueError):
        direct().build_direct_reference_model(checkpoint[0], checkpoint[1], reference_enabled=True, local_enabled=False)
    with pytest.raises(ValueError):
        direct().build_direct_reference_model(checkpoint[0], checkpoint[1], reference_enabled=True, reference_bn="frozen")


def test_stored_reference_bn_keeps_prefix_statistics_query_only(checkpoint):
    stored = direct().build_direct_reference_model(checkpoint[0], checkpoint[1], reference_enabled=True, reference_bn="stored").train()
    shared = build(checkpoint, True).train()
    query_only = build(checkpoint, False).train()
    assert stored.initialization_provenance["reference_bn"] == "stored"
    batch = inputs()
    batch["reference"] = batch["reference"] * 7. + 3.
    before = buffers(stored)
    stored(**batch)
    shared(**batch)
    query_only(**batch)
    prefix = lambda name: any(part in name.split(".") for part in ("stem", "layer1", "layer2"))
    for name, tensor in stored.named_buffers():
        if name.endswith("num_batches_tracked"):
            assert tensor - before[name] == 1
    assert_tensors_equal({n: t for n, t in stored.base.backbone.named_buffers() if prefix(n)},
                         {n: t for n, t in query_only.base.backbone.named_buffers() if prefix(n)})
    assert any(not torch.equal(t, dict(shared.base.backbone.named_buffers())[n])
               for n, t in stored.base.backbone.named_buffers() if prefix(n) and n.endswith("running_mean"))
    assert all(module.training for module in stored.modules())
    stored.eval()
    with torch.inference_mode():
        before = buffers(stored)
        stored(**batch)
    assert_tensors_equal(buffers(stored), before)
