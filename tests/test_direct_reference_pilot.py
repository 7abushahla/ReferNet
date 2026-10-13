import copy
import importlib
from pathlib import Path
import sys

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from refer_or_clear import pilots
from refer_or_clear.train import selection_key


@pytest.fixture
def runtime():
    class Runtime:
        def __getattr__(self, name):
            assert importlib.util.find_spec("refer_or_clear.direct_reference_pilot") is not None, "direct-reference execution primitives are missing"
            return getattr(importlib.import_module("refer_or_clear.direct_reference_pilot"), name)
    return Runtime()


class SyntheticClassifier(nn.Module):
    architecture_id = "anchor_direct_reference_ia18_v1"

    def __init__(self, enabled=False):
        super().__init__()
        self.reference_enabled = enabled
        self.encoder = nn.Parameter(torch.tensor(1.))
        self.head = nn.Parameter(torch.tensor(0.))
        self.register_buffer("counter", torch.tensor(0))
        self.initialization_provenance = {
            "base": {"requested_initialization": "levir", "checkpoint_sha256": pilots.OFFICIAL_WEIGHT_SHA256["D_running_pilot"],
                     "pretrained_tensor_sha256": "b" * 64},
            "architecture_id": self.architecture_id, "reference_enabled": enabled,
            "adapter_seed": 42, "adapter_tensor_sha256": "c" * 64,
        }
        self.bad_output = False
        self.mutate_eval = False

    def parameter_groups(self):
        return [self.encoder], [self.head]

    def forward(self, query, *, overlap, reference, context_available):
        if self.training or self.mutate_eval:
            self.counter.add_(1)
        query_logit = query[:, 0, 0, 0] * self.encoder + self.head
        correction = torch.zeros_like(query_logit)
        if self.reference_enabled and context_available.any():
            correction[context_available] = reference[context_available, 0, 0, 0] * overlap[context_available, 0, 0, 0]
        attack = query_logit + correction
        if self.bad_output:
            attack = attack * float("inf")
        return {"attack_logit": attack, "query_logit": query_logit, "correction_logit": correction}


def samples(enabled=False):
    rows = []
    for index, (logit, label, available) in enumerate([(-2., 0, True), (2., 1, False), (0., 0, True), (0., 1, False), (1., 1, True)]):
        query = torch.zeros(4, 32, 32)
        query[0].fill_(logit)
        row = {"query": query, "overlap": torch.ones(1, 32, 32), "context_available": torch.tensor(available),
               "label": label, "meta": {"image_id": str(index), "design_id": "design",
               "attack_family": "text_copy" if index == 1 else "text_inpaint" if label else "real",
               "availability_reason": "available" if available else "registration_failed"}}
        if enabled:
            row["reference"] = torch.zeros(3, 32, 32)
        rows.append(row)
    return rows


@pytest.fixture
def config():
    path = Path(__file__).resolve().parents[1] / "configs/direct_reference_pilot.yaml"
    assert path.exists(), "direct-reference config is missing"
    return yaml.safe_load(path.read_text())


def identities(config, model):
    result = {key: "a" * 64 for key in ("source_sha256", "records_sha256", "split_sha256", "enrollment_sha256",
              "registration_sha256", "phase_authority_sha256", "preprocess_sha256")}
    result.update(config_sha256=pilots.json_sha256(config), weights_sha256=pilots.OFFICIAL_WEIGHT_SHA256["D_running_pilot"],
                  initialization_sha256=pilots.json_sha256(model.initialization_provenance))
    return result


@pytest.mark.parametrize("enabled", [False, True])
def test_training_dispatch_and_sample_weighted_accounting(runtime, enabled):
    model = SyntheticClassifier(enabled)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda iteration: 1.)
    result = runtime.train_direct_reference_epoch(model, DataLoader(samples(enabled), batch_size=2), optimizer, scheduler, "cpu")
    wanted = nn.functional.binary_cross_entropy_with_logits(torch.tensor([-2., 2., 0., 0., 1.]), torch.tensor([0., 1., 0., 1., 1.])).item()
    assert result["image_loss"] == pytest.approx(wanted)
    assert result["loss"] == pytest.approx(wanted)
    assert result["n_images"] == 5 and result["optimizer_steps"] == 3
    assert scheduler.last_epoch == 3 and model.counter.item() == 3
    assert result["bce_map"] is None and result["dice_map"] is None


@pytest.mark.parametrize("enabled", [False, True])
def test_evaluation_keeps_unavailable_examples_and_genuine_ties(runtime, enabled):
    model = SyntheticClassifier(enabled)
    summary, predictions = runtime.evaluate_direct_reference(model, DataLoader(samples(enabled), batch_size=2), "cpu")
    assert summary["n_images"] == 5 and summary["n_genuine"] == 2 and summary["n_altered"] == 3
    assert summary["accuracy"] == pytest.approx(.8)
    assert summary["auc"] == pytest.approx(11 / 12)
    assert summary["subgroups"]["copy"]["auc"] == 1.
    assert summary["subgroups"]["inpainting"]["auc"] == pytest.approx(.875)
    assert summary["n_context_available"] == 3 and summary["n_context_unavailable"] == 2
    assert summary["referral_ids"] == ["1", "3"]
    assert summary["altered_misses"] == ["3"] and summary["genuine_false_positives"] == []
    assert [row["image_id"] for row in predictions] == ["0", "1", "2", "3", "4"]
    assert predictions[2]["attack_probability"] == .5 and predictions[3]["bonafide_score"] == .5
    assert predictions[1]["context_available"] is False
    assert predictions[1]["availability_reason"] == "registration_failed"
    assert summary["buffers_unchanged"] and model.counter.item() == 0
    assert model.encoder.grad is None


def test_on_dispatch_uses_reference_and_overlap(runtime):
    rows = samples(True)
    rows[0]["reference"].fill_(3.)
    rows[2]["reference"].fill_(2.)
    rows[2]["overlap"].zero_()
    rows[1]["reference"].fill_(float("nan"))
    _, predictions = runtime.evaluate_direct_reference(SyntheticClassifier(True), DataLoader(rows, batch_size=5), "cpu")
    assert predictions[0]["attack_logit"] == 1.
    assert predictions[0]["query_logit"] == -2. and predictions[0]["correction_logit"] == 3.
    assert predictions[1]["attack_logit"] == 2. and predictions[2]["attack_logit"] == 0.


def test_off_never_inspects_optional_reference(runtime):
    batch = next(iter(DataLoader(samples(), batch_size=5)))
    batch["reference"] = object()
    model = SyntheticClassifier()
    result = runtime.train_direct_reference_epoch(model, [batch], torch.optim.SGD(model.parameters(), lr=0.), None, "cpu")
    assert result["n_images"] == 5
    _, predictions = runtime.evaluate_direct_reference(model, [batch], "cpu")
    assert predictions[0]["attack_logit"] == -2.


def test_training_clips_gradients_before_update(runtime):
    rows = samples()[:1]
    rows[0]["query"][0].fill_(1000.)
    model = SyntheticClassifier()
    before = torch.stack([parameter.detach().clone() for parameter in model.parameters()])
    runtime.train_direct_reference_epoch(model, DataLoader(rows, batch_size=1), torch.optim.SGD(model.parameters(), lr=.1), None, "cpu")
    after = torch.stack([parameter.detach() for parameter in model.parameters()])
    assert torch.linalg.vector_norm(before - after).item() == pytest.approx(.5, abs=1e-6)


@pytest.mark.parametrize("failure", ["loss", "gradient"])
def test_nonfinite_training_stops_before_optimizer_and_scheduler(runtime, failure):
    model = SyntheticClassifier()
    model.bad_output = failure == "loss"
    if failure == "gradient":
        model.encoder.register_hook(lambda gradient: torch.full_like(gradient, float("nan")))
    optimizer = torch.optim.SGD(model.parameters(), lr=.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda iteration: 1.)
    with pytest.raises(ValueError, match="non-finite"):
        runtime.train_direct_reference_epoch(model, DataLoader(samples(), batch_size=5), optimizer, scheduler, "cpu")
    assert model.encoder.item() == 1. and model.head.item() == 0. and scheduler.last_epoch == 0


@pytest.mark.parametrize("failure", ["model_dtype", "query_dtype", "autocast"])
def test_strict_fp32_rejects_before_model_mutation(runtime, failure):
    model = SyntheticClassifier()
    rows = samples()
    if failure == "model_dtype":
        model.double()
    if failure == "query_dtype":
        rows[0]["query"] = rows[0]["query"].double()
    with torch.autocast("cpu", enabled=failure == "autocast"):
        with pytest.raises(ValueError, match="FP32"):
            runtime.train_direct_reference_epoch(model, DataLoader(rows, batch_size=5), torch.optim.SGD(model.parameters(), lr=.1), None, "cpu")
    assert model.counter.item() == 0


@pytest.mark.parametrize("failure", ["empty", "duplicate", "buffers", "logits"])
def test_evaluation_rejects_invalid_population_or_mutation(runtime, failure):
    rows = samples()
    model = SyntheticClassifier()
    if failure == "empty":
        rows = []
    if failure == "duplicate":
        rows[1]["meta"]["image_id"] = "0"
    model.mutate_eval = failure == "buffers"
    model.bad_output = failure == "logits"
    with pytest.raises(ValueError):
        runtime.evaluate_direct_reference(model, DataLoader(rows, batch_size=5), "cpu")


def test_empty_training_fails(runtime):
    model = SyntheticClassifier()
    with pytest.raises(ValueError, match="no.*images|no.*batches"):
        runtime.train_direct_reference_epoch(model, [], torch.optim.SGD(model.parameters(), lr=.1), None, "cpu")


def test_classification_only_evaluation_does_not_materialize_dense_outputs(runtime):
    class ExtraOutputClassifier(SyntheticClassifier):
        def forward(self, query, **kwargs):
            outputs = super().forward(query, **kwargs)
            outputs["map_logit"] = object()
            return outputs
    summary, predictions = runtime.evaluate_direct_reference(ExtraOutputClassifier(), DataLoader(samples(), batch_size=5), "cpu")
    assert summary["n_images"] == 5 and "map_logit" not in predictions[0]


def test_saved_default_config_is_a_usable_checkpoint_contract(runtime, config, tmp_path):
    model = SyntheticClassifier()
    authority = identities(config, model)
    path = tmp_path / "default.pt"
    runtime.save_direct_reference_checkpoint(path, model, config, authority, 1, (.8, .9), "smoke")
    assert runtime.read_direct_reference_checkpoint(path, config, authority)["arm"] == "off"


def test_recipe_reuses_original_optimizer_scheduler_and_selection(runtime, config):
    model = SyntheticClassifier()
    optimizer = runtime.make_pilot_optimizer(model, config, "train")
    scheduler = runtime.pilot_scheduler(optimizer, config["track"], 100)
    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.defaults["weight_decay"] == .05
    assert [group["initial_lr"] for group in optimizer.param_groups] == [1e-4, 1e-3]
    assert scheduler.lr_lambdas[0](0) == .1 and scheduler.lr_lambdas[0](100) == 0.
    assert runtime.selection_key is selection_key
    assert runtime.summarize_predictions is pilots.summarize_predictions
    keys = [runtime.selection_key("accuracy", accuracy, auc, 1., 1.) for accuracy, auc in [(0.8, .99), (.9, .8), (.9, .9), (.9, .9)]]
    selected = 0
    for index in range(1, len(keys)):
        if keys[index] > keys[selected]:
            selected = index
    assert selected == 2
    assert config["schema_version"] == "direct_reference_pilot_v1" and config["arm"] == "off"
    assert config["architecture_id"] == model.architecture_id
    assert (config["fold"], config["seed"], config["epochs"], config["batch_size"], config["eval_batch_size"]) == (0, 42, 100, 8, 1)
    assert config["augment"]["jitter_zone"] is True
    assert config["accumulation_steps"] == 1 and config["mask_weight"] == 0.
    assert config["precision"]["dtype"] == "float32" and config["precision"]["amp"] is False


def save_checkpoint(runtime, config, tmp_path, enabled=False):
    config = copy.deepcopy(config)
    config.update(arm="on" if enabled else "off", reference_enabled=enabled)
    model = SyntheticClassifier(enabled)
    authority = identities(config, model)
    path = tmp_path / "selected.pt"
    runtime.save_direct_reference_checkpoint(path, model, config, authority, 4, (.8, .9), "smoke")
    return path, model, config, authority


@pytest.mark.parametrize("enabled", [False, True])
def test_selected_checkpoint_roundtrip_and_replay(runtime, config, tmp_path, enabled):
    path, model, config, authority = save_checkpoint(runtime, config, tmp_path, enabled)
    checkpoint = runtime.read_direct_reference_checkpoint(path, config, authority)
    assert checkpoint["schema_version"] == "direct_reference_pilot_v1"
    assert checkpoint["arm"] == ("on" if enabled else "off")
    assert checkpoint["selected"] is True and checkpoint["epoch"] == 4
    assert checkpoint["selection_key"] == [.8, .9]
    loader = DataLoader(samples(enabled), batch_size=2)
    _, expected = runtime.evaluate_direct_reference(model, loader, "cpu")
    restored = SyntheticClassifier(enabled)
    restored.head.data.fill_(7.)
    summary, predictions, replay = runtime.replay_direct_reference_checkpoint(path, restored, loader, "cpu", config=config, identities=authority, expected_predictions=expected)
    assert summary["n_images"] == 5 and predictions == expected and replay["passed"]
    assert restored.head.item() == 0.


@pytest.mark.parametrize("mutation", ["legacy_schema", "architecture", "arm", "config", "identities", "missing_authority", "initialization", "unselected", "epoch", "phase"])
def test_checkpoint_rejects_legacy_mixed_or_incomplete_contract(runtime, config, tmp_path, mutation):
    path, model, config, authority = save_checkpoint(runtime, config, tmp_path)
    checkpoint = torch.load(path, weights_only=True)
    if mutation == "legacy_schema":
        checkpoint["schema_version"] = "anchor_pilot_v1"
    elif mutation == "architecture":
        checkpoint["architecture_id"] = "anchor_direct_running_ia18_pilot_v1"
    elif mutation == "arm":
        checkpoint["arm"] = "on"
    elif mutation == "config":
        checkpoint["config"]["seed"] = 43
    elif mutation == "identities":
        checkpoint["identities"]["source_sha256"] = "d" * 64
    elif mutation == "missing_authority":
        del checkpoint["identities"]["phase_authority_sha256"]
    elif mutation == "initialization":
        checkpoint["initialization_provenance"]["adapter_seed"] = 43
    elif mutation == "unselected":
        checkpoint["selected"] = False
    elif mutation == "epoch":
        checkpoint["epoch"] = 1.5
    else:
        checkpoint["phase"] = "overfit"
    torch.save(checkpoint, path)
    with pytest.raises(ValueError):
        runtime.read_direct_reference_checkpoint(path, config, authority)


def test_checkpoint_missing_expected_authority_and_wrong_model_fail(runtime, config, tmp_path):
    model = SyntheticClassifier()
    authority = identities(config, model)
    del authority["registration_sha256"]
    with pytest.raises(ValueError):
        runtime.save_direct_reference_checkpoint(tmp_path / "missing.pt", model, config, authority, 1, (.9, .9), "smoke")
    authority = identities(config, model)
    with pytest.raises(ValueError):
        runtime.save_direct_reference_checkpoint(tmp_path / "mixed.pt", SyntheticClassifier(True), config, authority, 1, (.9, .9), "smoke")


def test_replay_rejects_wrong_contract_before_loading_and_wrong_order(runtime, config, tmp_path):
    path, model, config, authority = save_checkpoint(runtime, config, tmp_path)
    loader = DataLoader(samples(), batch_size=5)
    _, expected = runtime.evaluate_direct_reference(model, loader, "cpu")
    wrong_model = SyntheticClassifier(True)
    wrong_model.head.data.fill_(7.)
    with pytest.raises(ValueError):
        runtime.replay_direct_reference_checkpoint(path, wrong_model, loader, "cpu", config=config, identities=authority, expected_predictions=expected)
    assert wrong_model.head.item() == 7.
    with pytest.raises(ValueError, match="replay"):
        runtime.replay_direct_reference_checkpoint(path, model, loader, "cpu", config=config, identities=authority, expected_predictions=list(reversed(expected)))


def test_auxiliary_query_loss_is_added_to_fused_loss_and_logged(runtime):
    rows = samples(True)
    rows[0]["reference"].fill_(3.)
    model = SyntheticClassifier(True)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.)
    result = runtime.train_direct_reference_epoch(model, DataLoader(rows, batch_size=5), optimizer, None, "cpu",
                                                  auxiliary_query_weight=.5)
    labels = torch.tensor([0., 1., 0., 1., 1.])
    query = torch.tensor([-2., 2., 0., 0., 1.])
    fused = query.clone()
    fused[0] += 3.
    image = nn.functional.binary_cross_entropy_with_logits(fused, labels).item()
    auxiliary = .5 * nn.functional.binary_cross_entropy_with_logits(query, labels).item()
    assert result["image_loss"] == pytest.approx(image)
    assert result["aux_query_loss"] == pytest.approx(auxiliary)
    assert result["loss"] == pytest.approx(image + auxiliary)
    plain = runtime.train_direct_reference_epoch(SyntheticClassifier(True), DataLoader(rows, batch_size=5),
                                                 optimizer, None, "cpu")
    assert plain["aux_query_loss"] == 0. and plain["loss"] == pytest.approx(plain["image_loss"])
    for weight in (-1., 11., True, "0.5"):
        with pytest.raises(ValueError):
            runtime.train_direct_reference_epoch(SyntheticClassifier(True), DataLoader(rows, batch_size=5),
                                                 optimizer, None, "cpu", auxiliary_query_weight=weight)


@pytest.mark.parametrize("enabled", [False, True])
def test_arm_recipe_defaults_and_overrides(runtime, enabled):
    assert runtime.arm_recipe({}, enabled) == dict(local_branch=True, auxiliary_query_weight=0., reference_bn="shared",
                                                   fused_query_weight=1.)
    base = dict(arm_recipe=dict(off=dict(local_branch=False), on=dict(auxiliary_query_weight=.5, reference_bn="stored",
                                                                       fused_query_weight=0.)))
    expected = dict(local_branch=True, auxiliary_query_weight=.5, reference_bn="stored", fused_query_weight=0.) if enabled else \
        dict(local_branch=False, auxiliary_query_weight=0., reference_bn="shared", fused_query_weight=1.)
    assert runtime.arm_recipe(base, enabled) == expected
    with pytest.raises(ValueError):
        runtime.arm_recipe(dict(arm_recipe=dict(off=dict(local_branch=False, fused_query_weight=0.))), enabled)
    for bad in (dict(arm_recipe=dict(both=dict())), dict(arm_recipe=dict(off=dict(dropout=.1))),
                dict(arm_recipe=dict(off=dict(reference_bn="x"))), dict(arm_recipe=dict(off=dict(auxiliary_query_weight=-1.))),
                dict(arm_recipe=dict(off=dict(local_branch=1))), dict(arm_recipe=[])):
        with pytest.raises(ValueError):
            runtime.arm_recipe(bad, enabled)
    with pytest.raises(ValueError):
        runtime.arm_recipe(dict(arm_recipe=dict(on=dict(local_branch=False))), enabled)


@pytest.mark.parametrize("name", ["direct_reference_study_v2.yaml", "direct_reference_study_v2_storedbn.yaml",
                                  "direct_reference_study_v2_storedbn_noaux.yaml", "direct_reference_study_v2_corronly.yaml",
                                  "direct_reference_study_v2_corronly_noaux.yaml"])
def test_v2_configs_resolve_arm_recipes_after_yaml_parsing(runtime, name):
    base = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs" / name).read_text())
    off, on = runtime.arm_recipe(base, False), runtime.arm_recipe(base, True)
    assert off == dict(local_branch=False, auxiliary_query_weight=0., reference_bn="shared", fused_query_weight=1.)
    assert on["local_branch"] is True
    assert on["reference_bn"] == ("stored" if "storedbn" in name else "shared")
    assert on["auxiliary_query_weight"] == (0. if name.endswith("noaux.yaml") else .5)
    assert on["fused_query_weight"] == (0. if "corronly" in name else 1.)
    assert runtime.arm_recipe(dict(arm_recipe={False: dict(local_branch=False), True: {}}), False)["local_branch"] is False
