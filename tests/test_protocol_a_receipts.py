import copy
import csv
import hashlib
import importlib
import json
import math
from pathlib import Path
import sys

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from refer_or_clear import protocol_a_train as service
from refer_or_clear.data.protocol_a_reference import ReferenceContext, _policy
from refer_or_clear.pilots import ATOL, RTOL, json_sha256, summarize_predictions
from refer_or_clear.protocol_a_contract import MembershipRow, ProtocolAContract
from refer_or_clear.train import selection_key


def bind(path):
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def write(path, document):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, sort_keys=True, allow_nan=False))
    return bind(path)


def make_gate_bundle(output, context, config, weights, registration, *, fresh_state="c" * 64, provenance=None):
    output.mkdir(parents=True, exist_ok=True)
    config_path = output.parent / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    request = dict(schema_version="protocol_a_request_v1", protocol="A", fold=context.contract.fold,
        source_sha256="a" * 64, records=context.bindings[0], split=context.bindings[1],
        enrollment=context.bindings[2], root=context.bindings[3], config=bind(config_path),
        weights=weights, raw_root=str(context.raw_root), output_root=str(output), cache_bytes=128)
    request_binding = write(output.parent / "request.json", request)
    metadata = json.loads(context.identities)
    reports, headers, smokes, replays = {}, {}, {}, {}
    predictions = []
    for row in context.rows("val").head(32).to_dict("records"):
        logit = 2. if row["label"] else -2.
        probability = 1. / (1. + math.exp(-logit))
        predictions.append(dict(image_id=row["image_id"], design_id=row["design_id"], label=row["label"],
            attack_family=row["attack_family"], attack_logit=logit, query_logit=logit, correction_logit=0.,
            attack_probability=probability, bonafide_score=1. - probability,
            context_available=True, referral_required=False, availability_reason="fit"))
    summary = dict(summarize_predictions(predictions), image_loss=math.log1p(math.exp(-2.)),
        buffers_unchanged=True, n_context_available=32, n_context_unavailable=0, referral_ids=[])
    for arm, enabled in (("off", False), ("on", True)):
        directory = output / "gate" / arm
        directory.mkdir(parents=True)
        effective = service.effective_config(config, context, enabled)
        initial = copy.deepcopy(provenance) if provenance is not None else dict(
            architecture_id=effective["architecture_id"], adapter_seed=42, adapter_tensor_sha256="d" * 64,
            base=dict(requested_initialization="levir", checkpoint_sha256=weights["sha256"],
                      checkpoint_path=weights["path"], pretrained_tensor_sha256="e" * 64))
        initial["reference_enabled"] = enabled
        ids = dict(source_sha256="a" * 64, config_sha256=json_sha256(effective), weights_sha256=weights["sha256"],
            records_sha256=metadata["records_sha256"], split_sha256=metadata["split_sha256"],
            enrollment_sha256=metadata["enrollment_manifest_sha256"], registration_sha256=registration["sha256"],
            phase_authority_sha256=request_binding["sha256"], initialization_sha256=json_sha256(initial),
            preprocess_sha256=json_sha256(dict(identities=metadata, policy=_policy())))
        checkpoint_path = directory / "selected.pt"
        checkpoint_path.write_bytes(f"synthetic checkpoint {arm}".encode())
        headers[str(checkpoint_path)] = dict(schema_version="direct_reference_pilot_v1",
            architecture_id=effective["architecture_id"], config=effective, config_sha256=json_sha256(effective),
            arm=arm, identities=ids, phase="smoke", selected=True, epoch=1,
            selection_key=[summary["accuracy"], summary["auc"]], initialization_provenance=initial)
        predictions_path = directory / "val_predictions.csv"
        with predictions_path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(predictions[0]))
            writer.writeheader()
            writer.writerows(predictions)
        smoke = dict(passed=True, no_test_open=True, pid=101, request=request_binding, config=effective,
            identities=ids, stats=dict(loss=.2, image_loss=.2, n_images=32, optimizer_steps=4),
            initialization=initial, fresh_initial_state_sha256=fresh_state, summary=summary,
            source_train_ids=context.rows("train").head(32).image_id.tolist(),
            source_val_ids=context.rows("val").head(32).image_id.tolist(),
            checkpoint=bind(checkpoint_path), predictions=bind(predictions_path),
            batch_invariance=dict(passed=True, n_images=4, buffers_unchanged=True, max_abs_difference=0.))
        smoke_binding = write(directory / "smoke.json", smoke)
        fresh = dict(passed=True, fresh_process=True, no_test_open=True, pid=202, n_images=32,
            request=request_binding, parent_result=smoke_binding, checkpoint=smoke["checkpoint"],
            identities=ids, summary=summary,
            comparison=dict(passed=True, n_images=32, atol=ATOL, rtol=RTOL, max_abs_difference=0.))
        reports[arm] = dict(smoke=smoke_binding, replay=write(directory / "fresh-replay.json", fresh))
        smokes[arm], replays[arm] = smoke, fresh
    gate = dict(schema_version="protocol_a_training_gate_v1", protocol="A", fold=context.contract.fold,
        source_sha256="a" * 64, records_sha256=metadata["records_sha256"], split_sha256=metadata["split_sha256"],
        enrollment_sha256=metadata["enrollment_manifest_sha256"], weights_sha256=weights["sha256"],
        registration_sha256=registration["sha256"], recipe_sha256=json_sha256(service.effective_config(config, context, False)),
        passed=True, no_test_open=True, both_arm_smoke=True, replay=True, batch_invariance=True,
        request=request_binding, registration=registration, arm_reports=reports)
    gate_binding = write(output / "gate/gate.json", gate)
    return dict(gate=gate, gate_binding=gate_binding, request=request, request_binding=request_binding,
                headers=headers, smokes=smokes, replays=replays, output=output)


def test_module_exists():
    assert (ROOT / "src/refer_or_clear/protocol_a_receipts.py").is_file(), "shared receipt validator missing"


@pytest.fixture
def api():
    return importlib.import_module("refer_or_clear.protocol_a_receipts")


@pytest.fixture
def setup(api, tmp_path, monkeypatch, request):
    config = yaml.safe_load((ROOT / "configs/direct_reference_pilot.yaml").read_text())
    fold = getattr(request, "param", 3)
    config["fold"] = fold
    bindings = [write(tmp_path / f"{name}.json", dict(synthetic=name))
                for name in ("records", "split", "enrollment", "root")]
    bindings[3] = write(tmp_path / "root.json", dict(schema_version="protocol_a_product_root_v1"))
    rows = {role: [dict(image_id=f"{role}{index:03d}", identity_id=f"{role}-identity{index}",
             component_id=f"{role}-component{index}", design_id="design", label=index % 2,
             attack_family="text_copy" if index % 2 else "", path=f"{role}{index}.png", role=role)
             for index in range(count)] for role, count in (("train", 35), ("val", 223 if fold == 4 else 33), ("test", 4))}
    members = {role: tuple(MembershipRow(**{key: row[key] for key in
               ("image_id", "identity_id", "component_id", "design_id", "label")}) for row in selected)
               for role, selected in rows.items()}
    contract = ProtocolAContract(fold, f"it{fold}", bindings[0]["sha256"], bindings[1]["sha256"],
                                ("design",), members["train"], members["val"], members["test"], (), ())
    metadata = dict(protocol="A", fold=fold, records_sha256=contract.records_sha256,
        split_sha256=contract.split_sha256, enrollment_manifest_sha256=bindings[2]["sha256"])
    context = ReferenceContext(contract, tmp_path / "raw", json.dumps(dict(output_root=str(tmp_path / "banks"))),
                               json.dumps(rows), json.dumps(bindings), json.dumps(metadata), str(tmp_path / "banks"), False)
    output = tmp_path / "output"
    output.mkdir()
    registration = write(output / "registration.json", dict(synthetic="registration"))
    weights_path = tmp_path / "weights.bin"
    weights_path.write_bytes(b"mock official weights")
    weights = dict(path=str(weights_path), sha256=config["weights_sha256"])
    bundle = make_gate_bundle(output, context, config, weights, registration)
    reads = []

    def reader(path, effective, ids):
        reads.append(str(path))
        return copy.deepcopy(bundle["headers"][str(path)])

    monkeypatch.setattr(api, "read_direct_reference_checkpoint", reader)
    return dict(bundle, context=context, config=config, weights=weights, registration=registration,
                rows=rows, reads=reads, tmp=tmp_path)


def validate(api, setup, **changes):
    arguments = dict(request_binding=setup["request_binding"])
    arguments.update(changes)
    return api.validate_training_gate(setup["gate_binding"], setup["context"], setup["config"],
                                      setup["weights"], setup["registration"], "a" * 64, **arguments)


def update_child(setup, arm, kind):
    field = "smokes" if kind == "smoke" else "replays"
    name = "smoke.json" if kind == "smoke" else "fresh-replay.json"
    child = write(setup["output"] / "gate" / arm / name, setup[field][arm])
    setup["gate"]["arm_reports"][arm][kind] = child
    if kind == "smoke":
        setup["replays"][arm]["parent_result"] = child
        update_child(setup, arm, "replay")
    setup["gate_binding"] = write(setup["output"] / "gate/gate.json", setup["gate"])


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return bind(path)


@pytest.fixture
def completed(setup, request):
    arm = getattr(request, "param", "off")
    context = setup["context"]
    directory = setup["output"] / "training" / arm
    directory.mkdir(parents=True)
    config = service.effective_config(setup["config"], context, arm == "on")
    smoke = setup["smokes"][arm]
    ids = dict(smoke["identities"], phase_authority_sha256=setup["gate_binding"]["sha256"])
    predictions = []
    for row in context.rows("val").to_dict("records"):
        logit = 2. if row["label"] else -2.
        probability = 1. / (1. + math.exp(-logit))
        predictions.append(dict(image_id=row["image_id"], design_id=row["design_id"], label=row["label"],
            attack_family=row["attack_family"], attack_logit=logit, query_logit=logit, correction_logit=0.,
            attack_probability=probability, bonafide_score=1. - probability,
            context_available=True, referral_required=False, availability_reason="fit"))
    summary = dict(summarize_predictions(predictions), image_loss=math.log1p(math.exp(-2.)),
        buffers_unchanged=True, n_context_available=len(predictions), n_context_unavailable=0, referral_ids=[],
        role="val", protocol="A", fold=context.contract.fold, selected_epoch=1)
    history = [dict(epoch=epoch, n_images=32, optimizer_steps=4, loss=.2, image_loss=.2,
        bce_map=None, dice_map=None, val_accuracy=summary["accuracy"], val_auc=summary["auc"],
        val_image_loss=summary["image_loss"], lr_pretrained=.0001, lr_new=.001) for epoch in range(1, 101)]
    checkpoint = directory / "best.pt"
    checkpoint.write_bytes(b"synthetic full selected checkpoint")
    setup["headers"][str(checkpoint)] = dict(schema_version="direct_reference_pilot_v1",
        architecture_id=config["architecture_id"], config=config, config_sha256=json_sha256(config),
        arm=arm, identities=ids, phase="train", selected=True, epoch=1,
        selection_key=[summary["accuracy"], summary["auc"]], initialization_provenance=smoke["initialization"])
    replay = dict(passed=True, fresh_process=False, phase="train", n_images=len(predictions))
    result = dict(schema_version="protocol_a_train_v1", protocol="A", fold=context.contract.fold,
        passed=True, no_test_open=True, phase="train", diagnostic_only=False,
        arm=arm, reference_enabled=arm == "on", role="val", epochs_completed=100, selected_epoch=1,
        optimizer_steps=400, n_train=35, n_val=len(predictions), batches_per_epoch=4, images_per_epoch=32,
        scheduler=dict(total_updates=400, warmup_updates=40), cache_bytes=dict(total=128, train=66, val=62),
        fresh_initial_state_sha256=smoke["fresh_initial_state_sha256"], initialization=smoke["initialization"],
        identities=ids, selection_key=[summary["accuracy"], summary["auc"]], pid=303,
        omitted_train_ids=[dict(epoch=epoch, image_ids=[f"train{index:03d}" for index in (32, 33, 34)])
                           for epoch in range(1, 101)],
        checkpoint=bind(checkpoint), predictions=write_csv(directory / "val_predictions.csv", predictions),
        history=write_csv(directory / "log.csv", history), summary=write(directory / "val_summary.json", summary),
        replay=replay, in_process_replay=write(directory / "replay.json", replay),
        registration=setup["registration"], diagnostic_gate=setup["gate_binding"], request=setup["request_binding"])
    parent = write(directory / "train.json", result)
    fresh = dict(passed=True, no_test_open=True, fresh_process=True, pid=404, n_images=len(predictions),
        request=setup["request_binding"], parent_result=parent, identities=ids, checkpoint=result["checkpoint"],
        comparison=dict(passed=True, n_images=len(predictions), atol=ATOL, rtol=RTOL, max_abs_difference=0.), summary=summary)
    write(directory / "fresh-replay.json", fresh)
    return dict(setup, directory=directory, result=result, fresh=fresh, history=history,
                full_predictions=predictions, summary=summary)


def validate_completed(api, completed, **changes):
    arguments = dict(context=completed["context"], request=completed["request"], base=completed["config"],
        arm=completed["directory"].name, output=completed["output"], gate_binding=completed["gate_binding"],
        request_binding=completed["request_binding"])
    arguments.update(changes)
    return api.validate_completed_training(**arguments)


def update_completed(completed):
    completed["fresh"]["parent_result"] = write(completed["directory"] / "train.json", completed["result"])
    write(completed["directory"] / "fresh-replay.json", completed["fresh"])


@pytest.mark.parametrize("completed", ["off", "on"], indirect=True)
def test_completed_training_authenticates_full_source_validation_and_earliest_tie(api, completed):
    result, config = validate_completed(api, completed)
    assert result == completed["result"]
    assert config == service.effective_config(completed["config"], completed["context"], completed["directory"].name == "on")
    assert len(completed["reads"]) == 3


def test_completed_accuracy_before_auc_then_earliest_tie(api, completed):
    completed["history"][0].update(val_accuracy=.99, val_auc=1.)
    completed["result"]["selected_epoch"] = 2
    completed["summary"]["selected_epoch"] = 2
    completed["result"]["history"] = write_csv(completed["directory"] / "log.csv", completed["history"])
    completed["result"]["summary"] = write(completed["directory"] / "val_summary.json", completed["summary"])
    completed["fresh"]["summary"] = completed["summary"]
    completed["headers"][completed["result"]["checkpoint"]["path"]]["epoch"] = 2
    update_completed(completed)
    result, unused_config = validate_completed(api, completed)
    assert result["selected_epoch"] == 2


def test_replay_parent_validation_can_precede_fresh_child_but_not_evaluation(api, completed):
    (completed["directory"] / "fresh-replay.json").unlink()
    result, unused_config = validate_completed(api, completed, require_fresh=False)
    assert result == completed["result"]
    with pytest.raises((ValueError, FileNotFoundError)):
        validate_completed(api, completed)


def test_parent_only_replay_validation_still_requires_full_accounting(api, completed):
    (completed["directory"] / "fresh-replay.json").unlink()
    completed["history"].pop()
    completed["result"]["history"] = write_csv(completed["directory"] / "log.csv", completed["history"])
    write(completed["directory"] / "train.json", completed["result"])
    with pytest.raises(ValueError):
        validate_completed(api, completed, require_fresh=False)


@pytest.mark.parametrize("field,value", [("passed", False), ("no_test_open", False), ("diagnostic_only", True),
    ("phase", "smoke"), ("arm", "on"), ("role", "test"), ("fold", True), ("epochs_completed", 99),
    ("optimizer_steps", 200), ("n_train", 1778), ("n_val", 222), ("images_per_epoch", 16),
    ("batches_per_epoch", 2), ("pid", 0), ("selected_epoch", 2), ("selection_key", [.5, .5]),
    ("fresh_initial_state_sha256", "f" * 64), ("request", {}), ("diagnostic_gate", {}), ("registration", {})])
def test_completed_training_requires_actual_current_contract(api, completed, field, value):
    completed["result"][field] = value
    update_completed(completed)
    with pytest.raises(ValueError):
        validate_completed(api, completed)


@pytest.mark.parametrize("failure", ["missing", "extra", "wrong-epoch", "images", "updates", "nonfinite", "map-loss",
                                     "loss", "later-better", "selected-loss", "auc", "scheduler"])
def test_completed_history_accounting_and_earliest_selection_audited(api, completed, failure):
    history = completed["history"]
    if failure == "missing":
        history.pop()
    elif failure == "extra":
        history.append(dict(history[-1], epoch=101))
    elif failure == "scheduler":
        completed["result"]["scheduler"]["warmup_updates"] = 1000
    else:
        fields = {"wrong-epoch": ("epoch", 2), "images": ("n_images", 31), "updates": ("optimizer_steps", 3),
            "nonfinite": ("loss", float("nan")), "map-loss": ("bce_map", .1), "loss": ("image_loss", .8),
            "later-better": ("val_accuracy", .5), "selected-loss": ("val_image_loss", .8), "auc": ("val_auc", 1.1)}
        field, value = fields[failure]
        history[0][field] = value
    completed["result"]["history"] = write_csv(completed["directory"] / "log.csv", history)
    update_completed(completed)
    with pytest.raises(ValueError):
        validate_completed(api, completed)


@pytest.mark.parametrize("field,value", [("fresh_process", False), ("no_test_open", False), ("pid", 303),
    ("n_images", 32), ("parent_result", {}), ("request", {}), ("checkpoint", {}), ("identities", {})])
def test_completed_fresh_replay_must_bind_full_parent_and_population(api, completed, field, value):
    completed["fresh"][field] = value
    write(completed["directory"] / "fresh-replay.json", completed["fresh"])
    with pytest.raises(ValueError):
        validate_completed(api, completed)


@pytest.mark.parametrize("failure", ["source", "phase", "init", "label", "summary", "shadow-checkpoint", "checkpoint-hash",
                                     "history-bytes", "registration-bytes", "initial-provenance", "omissions"])
def test_completed_nested_authorities_cannot_be_substituted(api, completed, failure):
    result = completed["result"]
    if failure in ("source", "phase", "init"):
        field = dict(source="source_sha256", phase="phase_authority_sha256", init="initialization_sha256")[failure]
        result["identities"][field] = "f" * 64
    elif failure == "label":
        completed["full_predictions"][0]["label"] = 1
        result["predictions"] = write_csv(completed["directory"] / "val_predictions.csv", completed["full_predictions"])
    elif failure == "summary":
        result["summary"] = write(completed["directory"] / "val_summary.json", dict(completed["summary"], accuracy=.5))
    elif failure == "shadow-checkpoint":
        path = completed["directory"] / "shadow.pt"
        path.write_bytes(Path(result["checkpoint"]["path"]).read_bytes())
        result["checkpoint"] = bind(path)
    elif failure == "checkpoint-hash":
        Path(result["checkpoint"]["path"]).write_bytes(b"changed")
    elif failure == "history-bytes":
        Path(result["history"]["path"]).write_text("changed")
    elif failure == "registration-bytes":
        Path(result["registration"]["path"]).write_text("changed")
    elif failure == "initial-provenance":
        result["initialization"] = dict(result["initialization"], adapter_tensor_sha256="f" * 64)
    else:
        result["omitted_train_ids"][0]["image_ids"] = ["test000", "train001", "train002"]
    update_completed(completed)
    with pytest.raises(ValueError):
        validate_completed(api, completed)


def test_authenticates_both_arms_without_model_or_pixels(api, setup):
    assert validate(api, setup) == setup["gate"]
    assert len(setup["reads"]) == 2
    assert validate(api, setup, request_binding=None) == setup["gate"]


@pytest.mark.parametrize("field,value", [("adapter_seed", 43), ("adapter_tensor_sha256", "bad"),
                                        ("base", None)])
def test_smoke_initialization_provenance_fields_are_bound(api, setup, field, value):
    smoke = setup["smokes"]["off"]
    smoke["initialization"][field] = value
    smoke["identities"]["initialization_sha256"] = json_sha256(smoke["initialization"])
    update_child(setup, "off", "smoke")
    with pytest.raises(ValueError):
        validate(api, setup)


def test_smoke_cannot_substitute_boolean_recipe_value(api, setup):
    setup["smokes"]["off"]["config"]["accumulation_steps"] = True
    update_child(setup, "off", "smoke")
    with pytest.raises(ValueError):
        validate(api, setup)


def test_off_on_smoke_start_from_same_tensors(api, setup):
    setup["smokes"]["on"]["fresh_initial_state_sha256"] = "f" * 64
    update_child(setup, "on", "smoke")
    with pytest.raises(ValueError):
        validate(api, setup)


@pytest.mark.parametrize("failure", ["arm-reports", "missing-on", "request-hash", "registration-hash", "outer-hash"])
def test_missing_or_changed_authorities_rejected(api, setup, failure):
    if failure == "arm-reports":
        del setup["gate"]["arm_reports"]
    elif failure == "missing-on":
        del setup["gate"]["arm_reports"]["on"]
    elif failure == "request-hash":
        setup["gate"]["request"]["sha256"] = "f" * 64
    elif failure == "registration-hash":
        Path(setup["registration"]["path"]).write_text("changed")
    else:
        setup["gate_binding"]["sha256"] = "f" * 64
        with pytest.raises(ValueError):
            validate(api, setup)
        return
    setup["gate_binding"] = write(setup["output"] / "gate/gate.json", setup["gate"])
    with pytest.raises(ValueError):
        validate(api, setup)


@pytest.mark.parametrize("kind", ["smoke", "replay"])
def test_nested_receipt_hash_is_not_self_rebound(api, setup, kind):
    binding = setup["gate"]["arm_reports"]["on"][kind]
    Path(binding["path"]).write_text("changed receipt")
    with pytest.raises(ValueError):
        validate(api, setup)


@pytest.mark.parametrize("field,value", [("passed", False), ("no_test_open", 1),
    ("fresh_initial_state_sha256", "bad"), ("source_train_ids", []), ("source_val_ids", ["test000"]),
    ("pid", 0)])
def test_smoke_evidence_required(api, setup, field, value):
    setup["smokes"]["on"][field] = value
    update_child(setup, "on", "smoke")
    with pytest.raises(ValueError):
        validate(api, setup)


@pytest.mark.parametrize("field,value", [("n_images", 31), ("n_images", True), ("optimizer_steps", 3),
                                         ("loss", float("nan"))])
def test_exact_smoke_updates_required(api, setup, field, value):
    setup["smokes"]["off"]["stats"][field] = value
    if isinstance(value, float) and math.isnan(value):
        with pytest.raises(ValueError):
            write(setup["output"] / "gate/off/smoke.json", setup["smokes"]["off"])
        return
    update_child(setup, "off", "smoke")
    with pytest.raises(ValueError):
        validate(api, setup)


@pytest.mark.parametrize("field,value", [("passed", False), ("n_images", 3), ("buffers_unchanged", False),
                                         ("max_abs_difference", -1.)])
def test_existing_invariance_evidence_required(api, setup, field, value):
    setup["smokes"]["off"]["batch_invariance"][field] = value
    update_child(setup, "off", "smoke")
    with pytest.raises(ValueError):
        validate(api, setup)


def test_hash_consistent_excessive_invariance_difference_is_rejected(api, setup):
    setup["smokes"]["off"]["batch_invariance"]["max_abs_difference"] = ATOL + RTOL * 2. + 1e-6
    update_child(setup, "off", "smoke")
    with pytest.raises(ValueError, match="invariance.*tolerance"):
        validate(api, setup)


def test_invariance_difference_at_authenticated_singleton_bound_is_accepted(api, setup):
    setup["smokes"]["off"]["batch_invariance"]["max_abs_difference"] = ATOL + RTOL * 2.
    update_child(setup, "off", "smoke")
    assert validate(api, setup) == setup["gate"]


@pytest.mark.parametrize("field,value", [("fresh_process", False), ("pid", 0), ("pid", True), ("pid", 101),
    ("n_images", 31), ("no_test_open", False), ("parent_result", {}), ("checkpoint", {}), ("request", {})])
def test_child_freshness_and_parent_binding_required(api, setup, field, value):
    setup["replays"]["on"][field] = value
    update_child(setup, "on", "replay")
    with pytest.raises(ValueError):
        validate(api, setup)


@pytest.mark.parametrize("field,value", [("passed", False), ("n_images", 31), ("atol", 1.),
                                         ("rtol", 1.), ("max_abs_difference", -1.)])
def test_replay_logit_comparison_contract_required(api, setup, field, value):
    setup["replays"]["on"]["comparison"][field] = value
    update_child(setup, "on", "replay")
    with pytest.raises(ValueError):
        validate(api, setup)


@pytest.mark.parametrize("field", ["source_sha256", "records_sha256", "split_sha256", "enrollment_sha256",
                                  "registration_sha256", "phase_authority_sha256", "preprocess_sha256", "initialization_sha256"])
def test_receipt_ids_derived_from_current_authorities(api, setup, field):
    setup["smokes"]["on"]["identities"][field] = "f" * 64
    update_child(setup, "on", "smoke")
    with pytest.raises(ValueError):
        validate(api, setup)


@pytest.mark.parametrize("field,value", [("image_id", "test000"), ("design_id", "wrong"), ("label", 1),
    ("attack_family", "text_inpaint"), ("attack_probability", ".9"), ("bonafide_score", ".9"),
    ("referral_required", "True"), ("attack_logit", "nan")])
def test_csv_authenticated_but_wrong_metadata_or_scores_rejected(api, setup, field, value):
    smoke = setup["smokes"]["on"]
    path = Path(smoke["predictions"]["path"])
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    rows[0][field] = value
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    smoke["predictions"] = bind(path)
    update_child(setup, "on", "smoke")
    with pytest.raises(ValueError):
        validate(api, setup)


@pytest.mark.parametrize("kind", ["smoke", "replay"])
@pytest.mark.parametrize("field,value", [("accuracy", .5), ("auc", .5), ("n_images", 31),
    ("image_loss", 1.), ("buffers_unchanged", False), ("referral_ids", ["val000"])])
def test_summary_reproduced_from_predictions(api, setup, kind, field, value):
    document = setup["smokes" if kind == "smoke" else "replays"]["off"]
    document["summary"][field] = value
    update_child(setup, "off", kind)
    with pytest.raises(ValueError):
        validate(api, setup)


@pytest.mark.parametrize("field,value", [("phase", "train"), ("epoch", 2), ("arm", "off"),
                                         ("selection_key", [.5, .5])])
def test_selected_smoke_checkpoint_header_required(api, setup, field, value):
    path = setup["smokes"]["on"]["checkpoint"]["path"]
    setup["headers"][path][field] = value
    with pytest.raises(ValueError):
        validate(api, setup)


def test_initialization_provenance_identity_required(api, setup):
    setup["smokes"]["off"]["initialization"]["base"]["requested_initialization"] = "warmstart"
    update_child(setup, "off", "smoke")
    with pytest.raises(ValueError):
        validate(api, setup)


@pytest.mark.parametrize("remainder", [0, 1, 2])
def test_completed_drop_last_omissions_follow_current_a_population(api, completed, remainder):
    context = completed["context"]
    object.__setattr__(context.contract, "source_train", context.contract.source_train[:32 + remainder])
    rows = json.loads(context._rows)
    rows["train"] = rows["train"][:32 + remainder]
    object.__setattr__(context, "_rows", json.dumps(rows))
    completed["result"]["n_train"] = 32 + remainder
    for omission in completed["result"]["omitted_train_ids"]:
        omission["image_ids"] = [f"train{index:03d}" for index in range(32, 32 + remainder)]
    update_completed(completed)
    result, unused_config = validate_completed(api, completed)
    assert result["n_train"] == 32 + remainder
    assert all(len(omission["image_ids"]) == remainder for omission in result["omitted_train_ids"])


@pytest.mark.parametrize("setup", [4], indirect=True)
def test_fold_four_full_223_validation_receipt_and_fresh_replay(api, completed):
    result, config = validate_completed(api, completed)
    assert config["fold"] == result["fold"] == 4
    assert result["n_val"] == 223 and completed["fresh"]["n_images"] == 223
    assert len(completed["full_predictions"]) == 223
    completed["fresh"]["n_images"] = 222
    write(completed["directory"] / "fresh-replay.json", completed["fresh"])
    with pytest.raises(ValueError):
        validate_completed(api, completed)


@pytest.mark.parametrize("failure", ["omitted", "foreign", "metadata-as-root"])
def test_parent_request_cannot_omit_or_substitute_relocation_authority(api, setup, failure):
    request = copy.deepcopy(setup["request"])
    if failure == "omitted":
        request.pop("root")
    elif failure == "foreign":
        request["root"] = setup["registration"]
    else:
        request["root"] = request["enrollment"]
    with pytest.raises(ValueError):
        api._request(write(setup["tmp"] / "wrong-request.json", request), setup["context"],
                     setup["config"], setup["weights"], "a" * 64)


def test_unrelocated_a_context_needs_no_fake_target_or_root_authority(api, setup):
    request = copy.deepcopy(setup["request"])
    request.pop("root")
    object.__setattr__(setup["context"], "_bindings", json.dumps(setup["context"].bindings[:3]))
    supplied, unused_root = api._request(write(setup["tmp"] / "original-root-request.json", request),
        setup["context"], setup["config"], setup["weights"], "a" * 64)
    assert supplied == request and "target_enrollment" not in supplied



def _fused_rows(context, fused_query_weight):
    rows = context.rows("val").head(4).to_dict("records")
    predictions = []
    for index, row in enumerate(rows):
        available = index % 2 == 0
        query, correction = (3. if row["label"] else -3.), (1.5 if row["label"] else -1.5)
        logit = (fused_query_weight if available else 1.) * query + correction
        probability = 1. / (1. + math.exp(-logit))
        predictions.append(dict(image_id=row["image_id"], design_id=row["design_id"], label=row["label"],
            attack_family=row["attack_family"], attack_logit=logit, query_logit=query, correction_logit=correction,
            attack_probability=probability, bonafide_score=1. - probability,
            context_available=available, referral_required=not available, availability_reason="fit" if available else "no_fit"))
    return predictions


@pytest.mark.parametrize("weight", [0., .5])
def test_read_predictions_applies_fused_query_weight_only_with_context(api, setup, tmp_path, weight):
    context = setup["context"]
    predictions = _fused_rows(context, weight)
    path = tmp_path / "fused_predictions.csv"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(predictions[0]))
        writer.writeheader()
        writer.writerows(predictions)
    rows = context.rows("val").head(4)
    accepted = api.read_predictions(bind(path), rows, weight)
    assert [row["attack_logit"] for row in accepted] == [row["attack_logit"] for row in predictions]
    with pytest.raises(ValueError, match="score/logit/referral"):
        api.read_predictions(bind(path), rows)
    with pytest.raises(ValueError, match="fused_query_weight"):
        api.read_predictions(bind(path), rows, 1.5)
