import copy
import importlib
import importlib.util
import io
from pathlib import Path
import random
import sys

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from test_anchor_changer import checkpoint
from test_direct_reference_model import assert_tensors_equal, buffers, build, inputs


def regularized():
    assert importlib.util.find_spec("refer_or_clear.models.direct_reference_regularized") is not None, "regularized model module is missing"
    return importlib.import_module("refer_or_clear.models.direct_reference_regularized")


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("probability", [0.0, 0.1, 0.3])
def test_install_preserves_instance_parameters_state_and_rng(checkpoint, enabled, probability):
    model = build(checkpoint, enabled)
    original = copy.deepcopy(model.initialization_provenance)
    state = copy.deepcopy(model.state_dict())
    parameters = {name: id(parameter) for name, parameter in model.named_parameters()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    grouped = [[id(parameter) for parameter in group] for group in model.parameter_groups()]
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state().clone()
    result = regularized().regularize_model(model, probability)
    assert result is model
    assert parameters == {name: id(parameter) for name, parameter in model.named_parameters()}
    assert grouped == [[id(parameter) for parameter in group] for group in model.parameter_groups()]
    assert {id(parameter) for group in optimizer.param_groups for parameter in group["params"]} == set(parameters.values())
    assert_tensors_equal(model.state_dict(), state)
    assert random.getstate() == python_state
    observed_numpy = np.random.get_state()
    assert observed_numpy[0] == numpy_state[0]
    np.testing.assert_array_equal(observed_numpy[1], numpy_state[1])
    assert observed_numpy[2:] == numpy_state[2:]
    assert torch.equal(torch.get_rng_state(), torch_state)
    assert isinstance(model.global_dropout, nn.Dropout)
    assert isinstance(model.local_dropout, nn.Dropout)
    assert model.global_dropout.p == model.local_dropout.p == probability
    assert model.architecture_id == "anchor_direct_reference_ia18_head_dropout_v1"
    assert model.initialization_provenance["architecture_id"] == model.architecture_id
    assert model.initialization_provenance["regularization"] == {
        "policy": "head_readout_dropout_v1", "p": probability,
        "global_placement": "after_relu_before_final_linear",
        "local_placement": "after_masked_pool_before_correction_linear",
    }
    preserved = copy.deepcopy(model.initialization_provenance)
    preserved.pop("regularization")
    preserved["architecture_id"] = original["architecture_id"]
    assert preserved == original


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("training", [False, True])
def test_zero_probability_is_exact_parity(checkpoint, enabled, training):
    original = build(checkpoint, enabled).train(training)
    changed = copy.deepcopy(original)
    regularized().regularize_model(changed, 0.0)
    batch = inputs()
    torch_state = torch.get_rng_state().clone()
    expected = original(**batch)
    actual = changed(**batch)
    assert_tensors_equal(actual, expected)
    assert_tensors_equal(buffers(changed), buffers(original))
    assert torch.equal(torch.get_rng_state(), torch_state)


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("probability", [0.1, 0.3])
def test_evaluation_is_exact_and_dropout_inherits_existing_mode(checkpoint, enabled, probability):
    original = build(checkpoint, enabled).eval()
    with torch.no_grad():
        original.adapter.correction_head.weight.fill_(.01)
    changed = copy.deepcopy(original)
    regularized().regularize_model(changed, probability)
    assert not changed.global_dropout.training and not changed.local_dropout.training
    before = buffers(changed)
    torch_state = torch.get_rng_state().clone()
    batch = inputs()
    with torch.no_grad():
        assert_tensors_equal(changed(**batch), original(**batch))
    assert torch.equal(torch.get_rng_state(), torch_state)
    assert_tensors_equal(buffers(changed), before)
    changed.train()
    assert changed.global_dropout.training and changed.local_dropout.training
    changed.eval()
    assert not changed.global_dropout.training and not changed.local_dropout.training


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("probability", [0.1, 0.3])
def test_both_route_inputs_are_dropped_at_exact_readouts(checkpoint, enabled, probability):
    model = regularized().regularize_model(build(checkpoint, enabled), probability).train()
    captured = {name: [] for name in ("global", "local")}
    hidden = torch.arange(1, 129, dtype=torch.float32).repeat(8, 1)
    pooled = torch.arange(1, 65, dtype=torch.float32).repeat(8, 1)
    global_handle = model.base.classifier[2].register_forward_pre_hook(lambda module, args: captured["global"].append(args[0].detach().clone()))
    local_handle = model.adapter.correction_head.register_forward_pre_hook(lambda module, args: captured["local"].append(args[0].detach().clone()))
    for unused in range(2):
        model.base.classifier[2](hidden)
        model.adapter.correction_head(pooled)
    global_handle.remove()
    local_handle.remove()
    for name, original in (("global", hidden), ("local", pooled)):
        for observed in captured[name]:
            assert torch.count_nonzero(observed) > 0
            assert torch.count_nonzero(observed == 0) > 0
            torch.testing.assert_close(observed[observed != 0], (original / (1 - probability))[observed != 0])
        assert not torch.equal(*captured[name])
    torch.testing.assert_close(hidden, torch.arange(1, 129, dtype=torch.float32).repeat(8, 1))
    torch.testing.assert_close(pooled, torch.arange(1, 65, dtype=torch.float32).repeat(8, 1))


@pytest.mark.parametrize("enabled", [False, True])
def test_training_outputs_and_all_active_gradients(checkpoint, enabled):
    model = regularized().regularize_model(build(checkpoint, enabled), .3).train()
    with torch.no_grad():
        model.adapter.correction_head.weight.fill_(.01)
    batch = inputs()
    batch["reference"].requires_grad_()
    first = model(**batch)
    second = model(**batch)
    assert not torch.equal(first["query_logit"], second["query_logit"])
    assert not torch.equal(first["correction_logit"], second["correction_logit"])
    functional.binary_cross_entropy_with_logits(second["attack_logit"], torch.tensor([0., 1.])).backward()
    for module in (model.base.backbone, model.base.classifier, model.adapter.projection, model.adapter.local_head, model.adapter.correction_head):
        gradients = [parameter.grad for parameter in module.parameters()]
        assert all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients)
        assert sum(gradient.abs().sum() for gradient in gradients) > 0
    if enabled:
        assert batch["reference"].grad is not None and batch["reference"].grad.abs().sum() > 0
    else:
        assert batch["reference"].grad is None


@pytest.mark.parametrize("training", [False, True])
def test_off_ignores_malformed_reference_with_identical_rng(checkpoint, training):
    original = regularized().regularize_model(build(checkpoint, False), .3).train(training)
    changed = copy.deepcopy(original)
    batch = inputs()
    state = torch.get_rng_state().clone()
    expected = original(**batch)
    torch.set_rng_state(state)
    actual = changed(**dict(batch, reference=object()))
    assert_tensors_equal(actual, expected)
    assert_tensors_equal(buffers(changed), buffers(original))


@pytest.mark.parametrize("enabled", [False, True])
def test_eval_companions_and_state_reconstruction(checkpoint, enabled):
    model = regularized().regularize_model(build(checkpoint, enabled), .1).eval()
    with torch.no_grad():
        model.adapter.correction_head.weight.fill_(.01)
    saved = io.BytesIO()
    torch.save(model.state_dict(), saved)
    saved.seek(0)
    restored = regularized().regularize_model(build(checkpoint, enabled), .1).eval()
    restored.load_state_dict(torch.load(saved, weights_only=True), strict=True)
    assert restored.initialization_provenance == model.initialization_provenance
    assert_tensors_equal(restored.state_dict(), model.state_dict())
    batch = inputs(batch=8)
    before = buffers(model)
    with torch.no_grad():
        expected = model(**{name: value[:1] for name, value in batch.items()})
        for count in (1, 2, 4, 8):
            subset = {name: value[:count] for name, value in batch.items()}
            output = model(**subset)
            assert_tensors_equal(restored(**subset), output)
            for name in expected:
                torch.testing.assert_close(output[name][:1], expected[name], atol=1e-5, rtol=1e-4)
    assert_tensors_equal(buffers(model), before)


@pytest.mark.parametrize("probability", [True, False, None, "0.1", -0.1, 0.2, 1.0, float("nan"), float("inf")])
def test_invalid_rate_rejected_without_mutation(checkpoint, probability):
    model = build(checkpoint, False)
    provenance = copy.deepcopy(model.initialization_provenance)
    state = copy.deepcopy(model.state_dict())
    with pytest.raises(ValueError):
        regularized().regularize_model(model, probability)
    assert not hasattr(model, "global_dropout") and not hasattr(model, "local_dropout")
    assert model.initialization_provenance == provenance
    assert_tensors_equal(model.state_dict(), state)


def test_duplicate_install_rejected_without_additional_hooks(checkpoint):
    model = regularized().regularize_model(build(checkpoint, False), .1)
    provenance = copy.deepcopy(model.initialization_provenance)
    with pytest.raises(ValueError, match="installed"):
        regularized().regularize_model(model, .3)
    assert model.initialization_provenance == provenance
    assert model.global_dropout.p == model.local_dropout.p == .1
    assert len(model.base.classifier[2]._forward_pre_hooks) == 1
    assert len(model.adapter.correction_head._forward_pre_hooks) == 1


@pytest.mark.parametrize("enabled", [False, True])
def test_original_epoch_and_eval_core_accept_regularized_model(checkpoint, enabled):
    from refer_or_clear.direct_reference_pilot import evaluate_direct_reference, train_direct_reference_epoch
    from refer_or_clear.pilots import make_pilot_optimizer

    model = regularized().regularize_model(build(checkpoint, enabled), .1)
    batch = inputs()
    batch.update(label=torch.tensor([0, 1]), meta={
        "image_id": ["genuine", "altered"], "design_id": ["design", "design"],
        "attack_family": ["", "text_inpaint"],
    })
    optimizer = make_pilot_optimizer(model, {"track": "D_running_pilot"}, "train")
    assert [group["lr"] for group in optimizer.param_groups] == [.0001, .001]
    assert all(group["weight_decay"] == .05 for group in optimizer.param_groups)
    statistics = train_direct_reference_epoch(model, [batch], optimizer, None, "cpu")
    assert statistics["n_images"] == 2 and statistics["optimizer_steps"] == 1
    assert np.isfinite(statistics["loss"])
    before = buffers(model)
    summary, predictions = evaluate_direct_reference(model, [batch], "cpu")
    assert summary["n_images"] == 2 and summary["buffers_unchanged"]
    assert not model.global_dropout.training and not model.local_dropout.training
    assert [row["image_id"] for row in predictions] == ["genuine", "altered"]
    assert_tensors_equal(buffers(model), before)


def test_foreign_model_rejected():
    with pytest.raises(ValueError):
        regularized().regularize_model(nn.Linear(2, 1), .1)
