import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))
SCRIPT = ROOT / "scripts/score_protocol_c_v2.py"
GENERATOR = ROOT / "scripts/make_protocol_c_v2_request.py"
WRAPPER = ROOT / "scripts/cluster/score_protocol_c_v2.sh"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def put(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
    return dict(path=str(path), sha256=sha(path))


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def module():
    assert SCRIPT.is_file(), "Protocol C v2 scorer is missing"
    return load(SCRIPT, "protocol_c_v2_score_under_test")


def rows():
    frame = []
    for index in range(467):
        label = index % 2
        frame.append(dict(image_id=f"query-{index:04d}", design_id=f"fantasyid/design-{index % 3}",
                          attack_family=("face_swap" if index % 4 == 1 else "text_inpaint") if label else "", label=label))
    return pd.DataFrame(frame)


@pytest.fixture
def ready(module, tmp_path, monkeypatch):
    from test_direct_reference_pilot import SyntheticClassifier

    monkeypatch.setattr(module, "source_entries", lambda root: [])
    source_sha = module.object_sha256([])
    tmp = tmp_path
    a_output = tmp / "a-study"
    a_bindings = {name: put(tmp / f"a-{name}.json", {"synthetic": name}) for name in ("records", "split", "enrollment", "root", "config")}
    weights = tmp / "weights.bin"
    weights.write_bytes(b"synthetic official initialization")
    a_request = dict(schema_version="protocol_a_request_v1", protocol="A", fold=0, source_sha256=source_sha,
                     config=a_bindings["config"], weights=dict(path=str(weights), sha256=sha(weights)),
                     records=a_bindings["records"], split=a_bindings["split"], enrollment=a_bindings["enrollment"],
                     root=a_bindings["root"], raw_root=str(tmp / "a-raw"), output_root=str(a_output), cache_bytes=128)
    a_binding = put(tmp / "a-request.json", a_request)
    gate_binding = put(a_output / "gate/gate.json", dict(schema_version="protocol_a_training_gate_v1", passed=True,
                                                         request=a_binding))
    fresh_state = module.pilots.tensor_sha256(SyntheticClassifier(False).state_dict())
    identities = {name: "a" * 64 for name in module.MATCHED_IDENTITIES}
    receipts, configs = {}, {}
    for arm in ("off", "on"):
        directory = a_output / "training" / arm
        directory.mkdir(parents=True)
        (directory / "best.pt").write_bytes(f"synthetic {arm} checkpoint".encode())
        checkpoint = dict(path=str(directory / "best.pt"), sha256=sha(directory / "best.pt"))
        receipts[arm] = dict(schema_version="protocol_a_train_v1", passed=True, arm=arm, selected_epoch=7 if arm == "off" else 9,
                             selection_key=[1., 1.], checkpoint=checkpoint, fresh_initial_state_sha256=fresh_state,
                             identities=dict(identities, config_sha256=("b" if arm == "off" else "c") * 64,
                                             initialization_sha256="d" * 64))
        put(directory / "train.json", receipts[arm])
        put(directory / "fresh-replay.json", dict(passed=True, fresh_process=True, checkpoint=checkpoint, request=a_binding))
        configs[arm] = dict(reference_enabled=arm == "on", arm=arm, local_branch=True, reference_bn="shared")
    members = lambda role, count: tuple(SimpleNamespace(image_id=f"{role}{index}") for index in range(count))
    context = SimpleNamespace(contract=SimpleNamespace(fold=0, source_train=members("train", 1778), source_val=members("val", 222)),
                              raw_root=tmp / "a-raw", manifest=dict(output_root=str(tmp / "a-banks")),
                              product_root=str(tmp / "a-relocated"))
    base = {"synthetic": "base"}
    events = []

    def load_request(binding, **kwargs):
        assert binding == a_binding and kwargs == {}
        events.append("a-request")
        return copy.deepcopy(a_request), context, base, a_output

    def completed_arm(request, context_arg, base_arg, output, arm, gate, binding):
        assert request == a_request and context_arg is context and base_arg is base
        assert output == a_output and gate == gate_binding and binding == a_binding
        events.append(("completed", arm))
        return copy.deepcopy(receipts[arm]), copy.deepcopy(configs[arm])

    monkeypatch.setattr(module.study, "load_request", load_request)
    monkeypatch.setattr(module.study, "completed_arm", completed_arm)
    frame = rows()
    target_split = put(tmp / "c-target-split.json", {"synthetic": "target split"})
    monkeypatch.setattr(module.contract_api, "TARGET_SPLIT_SHA256", target_split["sha256"])
    target = put(tmp / "c-target.json", {"query_ids": frame.image_id.tolist()})
    # the C enrollment manifest was built from its own copies of the A records and split (same content, other paths)
    inputs = {name: put(tmp / f"c-inputs/{name}.json", {"synthetic": name}) for name in ("records", "split")}
    enrollment = put(tmp / "c-enrollment.json", {"output_root": str(tmp / "c-original"),
                                                 "contract": {"records_path": inputs["records"]["path"],
                                                              "source_split_path": inputs["split"]["path"]}})
    root = put(tmp / "c-root.json", {"original_root": str(tmp / "c-original"), "relocated_root": str(tmp / "c-relocated")})
    request = dict(schema_version="protocol_c_v2_score_v1", protocol="C", seed=42, eval_batch_size=1,
                   source_sha256=source_sha, a_request=a_binding, a_gate_sha256=gate_binding["sha256"],
                   target_split=target_split, target_enrollment=target, enrollment=enrollment, root=root,
                   raw_root=str(tmp / "never-open-c-raw"), output_root=str(tmp / "c-score"))
    binding = put(tmp / "c-request.json", request)
    monkeypatch.setattr(module.contract_api, "load_protocol_c_contract", lambda *args, **kwargs: pytest.fail("C contract opened"))
    monkeypatch.setattr(module.consumer, "load_reference_context", lambda *args, **kwargs: pytest.fail("C context opened"))
    return dict(module=module, request=request, binding=binding, a_request=a_request, a_binding=a_binding,
                a_output=a_output, gate=gate_binding, receipts=receipts, events=events, rows=frame, tmp=tmp)


def rebind(ready):
    ready["binding"] = put(Path(ready["binding"]["path"]), ready["request"])
    return ready["binding"]


def test_interface_exists(module):
    assert all(callable(getattr(module, name)) for name in ("load_request", "authenticate_arms", "score", "main"))
    assert module.validate_predictions is module.current.validate_predictions
    assert module.evaluate is module.current.evaluate


def test_allocation_guard_precedes_request_unless_cpu_allowed(module, monkeypatch):
    for name in ("SLURM_JOB_ID", "SLURMD_NODENAME", "SLURM_JOB_PARTITION", "CUDA_VISIBLE_DEVICES"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(module, "load_request", lambda *args: pytest.fail("request opened without allocation"))
    with pytest.raises(ValueError, match="allocated"):
        module.main(["--request", "/never-open.json", "--request-sha256", "a" * 64])
    seen = []
    monkeypatch.setattr(module, "score", lambda binding, device: seen.append((binding, device)))
    module.main(["--request", "/never-open.json", "--request-sha256", "a" * 64, "--allow-cpu"])
    assert seen == [(dict(path="/never-open.json", sha256="a" * 64), "cpu")]


@pytest.mark.parametrize("change", [
    lambda r: r.update(protocol="A"), lambda r: r.update(seed=43), lambda r: r.update(eval_batch_size=True),
    lambda r: r.update(fold=0), lambda r: r.pop("a_gate_sha256"), lambda r: r.update(a_gate_sha256="A" * 64),
    lambda r: r.update(source_sha256="a" * 64), lambda r: r.update(a_request="/a-request.json"),
    lambda r: r.update(raw_root="relative/raw"), lambda r: r.update(schema_version="protocol_c_current_score_v1")])
def test_strict_request_fields(ready, change):
    change(ready["request"])
    with pytest.raises(ValueError):
        ready["module"].load_request(rebind(ready))
    assert ready["events"] == []


def test_fold_zero_study_required(ready, monkeypatch):
    module = ready["module"]
    study = module.study.load_request

    def fold_one(binding, **kwargs):
        request, context, base, output = study(binding, **kwargs)
        request["fold"] = 1
        return request, context, base, output

    monkeypatch.setattr(module.study, "load_request", fold_one)
    with pytest.raises(ValueError, match="fold-0"):
        module.score(ready["binding"], "cpu")
    assert ("completed", "off") not in ready["events"]
    assert not Path(ready["request"]["output_root"]).exists()


def test_refuses_existing_evaluation(ready):
    (Path(ready["request"]["output_root"]) / "evaluation").mkdir(parents=True)
    with pytest.raises(ValueError, match="already exists"):
        ready["module"].load_request(ready["binding"])
    assert ready["events"] == []


def test_both_arms_authenticated_before_any_c_authority(ready, monkeypatch):
    module = ready["module"]
    original = module.study.completed_arm

    def fail_on(request, context, base, output, arm, gate, binding):
        result = original(request, context, base, output, arm, gate, binding)
        if arm == "on":
            raise ValueError("source training incomplete")
        return result

    monkeypatch.setattr(module.study, "completed_arm", fail_on)
    monkeypatch.setattr(module.contract_api, "load_target_enrollment", lambda *args, **kwargs: pytest.fail("C target opened"))
    with pytest.raises(ValueError, match="source training incomplete"):
        module.score(ready["binding"], "cpu")
    assert ready["events"] == ["a-request", ("completed", "off"), ("completed", "on")]
    assert not Path(ready["request"]["output_root"]).exists()


@pytest.mark.parametrize("defect", ["fresh_process", "passed", "checkpoint", "request", "missing"])
def test_fresh_replay_receipt_required(ready, defect):
    path = ready["a_output"] / "training/on/fresh-replay.json"
    if defect == "missing":
        path.unlink()
    else:
        replay = json.loads(path.read_text())
        replay[defect] = dict(path="/elsewhere", sha256="e" * 64) if defect in ("checkpoint", "request") else False
        put(path, replay)
    with pytest.raises((ValueError, FileNotFoundError)):
        ready["module"].score(ready["binding"], "cpu")


@pytest.mark.parametrize("name", [*("source_sha256", "weights_sha256", "records_sha256", "split_sha256", "enrollment_sha256",
                                   "registration_sha256", "phase_authority_sha256", "preprocess_sha256"), "fresh_initial_state_sha256"])
def test_matched_arm_identities_required(ready, name):
    receipt = ready["receipts"]["on"]
    if name == "fresh_initial_state_sha256":
        receipt[name] = "f" * 64
    else:
        receipt["identities"][name] = "f" * 64
    put(ready["a_output"] / "training/on/train.json", receipt)
    with pytest.raises(ValueError, match="matched"):
        ready["module"].score(ready["binding"], "cpu")


def wire_c_side(ready, monkeypatch, defect=None):
    from test_direct_reference_pilot import SyntheticClassifier

    module = ready["module"]
    frame = ready["rows"]
    contract = SimpleNamespace(records_sha256=ready["a_request"]["records"]["sha256"],
                               source_split_sha256=ready["a_request"]["split"]["sha256"],
                               target_split_sha256=ready["request"]["target_split"]["sha256"])
    target = SimpleNamespace(support_ids=tuple(f"support-{index}" for index in range(66)),
                             query_ids=tuple(frame.image_id), excluded_ids=tuple(f"excluded-{index}" for index in range(66)))
    monkeypatch.setattr(module.contract_api, "load_protocol_c_contract", lambda *args, **kwargs: contract)
    monkeypatch.setattr(module.contract_api, "load_target_enrollment", lambda *args, **kwargs: target)
    built = []

    def fresh(a_request, config, device):
        assert a_request == ready["a_request"] and device == "cpu"
        built.append(("fresh", config["reference_enabled"]))
        return SyntheticClassifier(config["reference_enabled"])

    def read_checkpoint(path, config, identities):
        arm = config["arm"]
        assert Path(path) == ready["a_output"] / "training" / arm / "best.pt"
        assert identities == ready["receipts"][arm]["identities"]
        built.append(("checkpoint", arm))
        model = SyntheticClassifier(config["reference_enabled"])
        with torch.no_grad():
            model.head.fill_(.5 if arm == "on" else 0.)
        return dict(state_dict=model.state_dict(), phase="train", epoch=ready["receipts"][arm]["selected_epoch"],
                    selection_key=[1., 1.] if defect != "epoch" else [0., 0.])

    monkeypatch.setattr(module, "fresh_model", fresh)
    monkeypatch.setattr(module.core, "_validate_model", lambda *args: None)
    monkeypatch.setattr(module.core, "read_direct_reference_checkpoint", read_checkpoint)

    def context(records, split, target_split, **kwargs):
        manifest = module.driver.document(ready["request"]["enrollment"])["contract"]
        assert records == manifest["records_path"] and split == manifest["source_split_path"]
        assert records != ready["a_request"]["records"]["path"] and split != ready["a_request"]["split"]["path"]
        authority = module.driver.document(kwargs["evaluation_binding"])
        assert authority["authorized_image_ids"] == frame.image_id.tolist()
        assert authority["checkpoints"] == {arm: ready["receipts"][arm]["checkpoint"] for arm in ("off", "on")}
        assert authority["selected_epochs"] == {"off": 7, "on": 9} and authority["source_fold"] == 0
        assert kwargs["root_binding"] == ready["request"]["root"] and kwargs["raw_root"] == Path(ready["request"]["raw_root"])
        built.append("context")
        return SimpleNamespace(rows=lambda: frame.iloc[::-1] if defect == "membership" else frame)

    monkeypatch.setattr(module.consumer, "load_reference_context", context)

    def build(context, *, output_path):
        put(output_path, {"synthetic": "registration"})
        built.append("registration")
        return SimpleNamespace(path=str(output_path))

    monkeypatch.setattr(module.consumer, "build_protocol_c_registration", build)
    shared = object()
    monkeypatch.setattr(module.consumer, "load_protocol_c_registration", lambda *args: shared)

    def dataset(context, table, *, reference_enabled):
        assert table is shared
        built.append(("dataset", reference_enabled))
        samples = []
        for row in frame.to_dict("records"):
            query = torch.zeros(4, 16, 16)
            query[0].fill_(2. if row["label"] else -2.)
            sample = dict(query=query, overlap=torch.ones(1, 16, 16), label=row["label"],
                          context_available=torch.tensor(True),
                          meta={key: row[key] for key in ("image_id", "design_id", "attack_family")})
            if reference_enabled:
                sample["reference"] = torch.zeros(3, 16, 16)
            samples.append(sample)
        return samples

    monkeypatch.setattr(module.consumer, "ProtocolCReferenceDataset", dataset)
    return built


def test_end_to_end_synthetic_scoring(ready, monkeypatch):
    module = ready["module"]
    built = wire_c_side(ready, monkeypatch)
    result = module.score(ready["binding"], "cpu")
    assert built == ["fresh" and ("fresh", False), ("checkpoint", "off"), ("fresh", True), ("checkpoint", "on"),
                     "context", "registration", ("dataset", False), ("dataset", True)]
    assert ready["events"] == ["a-request", ("completed", "off"), ("completed", "on")]
    output = Path(ready["request"]["output_root"]) / "evaluation"
    assert {path.name for path in output.iterdir()} == {"authorization.json", "registration.json", "result.json", "off", "on"}
    assert result["passed"] and result["n_images"] == 467 and result["protocol"] == "C" and result["source_fold"] == 0
    assert result["published_target_count"] == 599 and result["n_target_support"] == result["n_target_excluded"] == 66
    assert result["learning_performed"] is False and result["threshold_tuning"] is False and result["bonafide_threshold"] == .5
    assert result["a_request"] == ready["a_binding"] and result["a_gate"] == ready["gate"]
    assert json.loads((output / "result.json").read_text()) == result
    for arm, epoch in (("off", 7), ("on", 9)):
        report = result["arms"][arm]
        assert report["checkpoint"] == ready["receipts"][arm]["checkpoint"] and report["selected_epoch"] == epoch
        assert report["fresh_replay"]["path"] == str(ready["a_output"] / "training" / arm / "fresh-replay.json")
        summary = json.loads((output / arm / "test_summary.json").read_text())
        assert summary["n_images"] == 467 and summary["selected_epoch"] == epoch and summary["role"] == "test"
        assert summary["source_checkpoint"] == report["checkpoint"] and summary["threshold_policy"] == "fixed_bonafide_threshold_0.5"
        assert set(summary["subgroups"]) == {"face_swap", "text_inpaint"} and summary["accuracy"] == 1.
        predictions = pd.read_csv(output / arm / "test_predictions.csv")
        assert predictions.image_id.tolist() == ready["rows"].image_id.tolist()
        assert (predictions.correction_logit == 0.).all()
    off = pd.read_csv(output / "off/test_predictions.csv")
    on = pd.read_csv(output / "on/test_predictions.csv")
    assert (on.attack_logit - off.attack_logit).round(6).eq(.5).all(), "selected checkpoint weights were loaded per arm"


@pytest.mark.parametrize("defect", ["membership", "epoch", "overlap", "changed_input"])
def test_defects_leave_no_result(ready, monkeypatch, defect):
    module = ready["module"]
    wire_c_side(ready, monkeypatch, defect)
    if defect == "overlap":
        ready["request"]["output_root"] = str(ready["a_output"] / "c-nested")
        rebind(ready)
    if defect == "changed_input":
        original = module.consumer.load_protocol_c_registration

        def mutate(*args):
            Path(ready["request"]["target_enrollment"]["path"]).write_text("changed after authorization\n")
            return original(*args)

        monkeypatch.setattr(module.consumer, "load_protocol_c_registration", mutate)
    with pytest.raises(ValueError):
        module.score(ready["binding"], "cpu")
    assert not (Path(ready["request"]["output_root"]) / "evaluation/result.json").exists()


def test_request_generator_copies_reviewed_c_bindings(ready, tmp_path, monkeypatch):
    generator = load(GENERATOR, "make_protocol_c_v2_request_under_test")
    assert not any(line.startswith(("import torch", "from torch", "from refer_or_clear")) for line in GENERATOR.read_text().splitlines())
    monkeypatch.setattr(generator, "source_entries", lambda root: [])
    template = json.loads((ROOT / "results/runs/protocol-c-current/requests/score-template-4aeb0d2.json").read_text())
    output = tmp_path / "requests/c-v2.json"
    source = tmp_path / "frozen-source"
    source.mkdir()
    code = generator.main([str(source), generator.object_sha256([]), ready["a_binding"]["path"], ready["gate"]["path"],
                           str(tmp_path / "c-v2-output"), "--output", str(output)])
    assert code == 0
    written = json.loads(output.read_text())
    assert written["schema_version"] == "protocol_c_v2_score_v1" and written["a_request"] == ready["a_binding"]
    assert written["a_gate_sha256"] == ready["gate"]["sha256"] and written["output_root"] == str(tmp_path / "c-v2-output")
    assert all(written[name] == template[name] for name in ("target_split", "target_enrollment", "enrollment", "root", "raw_root"))
    assert set(written) == set(ready["module"].FIELDS) | {"schema_version", "protocol", "seed", "eval_batch_size"}
    with pytest.raises(FileExistsError):
        generator.main([str(source), generator.object_sha256([]), ready["a_binding"]["path"], ready["gate"]["path"],
                        str(tmp_path / "c-v2-output"), "--output", str(output)])
    with pytest.raises(ValueError, match="overlap"):
        generator.build(source, generator.object_sha256([]), ready["a_binding"]["path"], ready["gate"]["path"],
                        str(ready["a_output"] / "nested"))


def test_wrapper_matches_study_wrapper_guards():
    assert WRAPPER.is_file(), "C v2 wrapper missing"
    content = WRAPPER.read_text()
    assert subprocess.run(["bash", "-n", str(WRAPPER)], capture_output=True).returncode == 0
    assert "scripts/score_protocol_c_v2.py" in content and "--allow-cpu" not in content
    assert "#SBATCH --job-name=protocol-c-v2-score" in content and "#SBATCH --partition=gpu" in content
    study = (ROOT / "scripts/cluster/protocol_a_study.sh").read_text()
    for guard in ("Compute work is forbidden on login nodes", "Allocated GPU visibility required",
                  "readonly non-symlink scientific source required", "conda activate", 'export PYTHONPATH="$source_dir/src:$source_dir/scripts"'):
        assert guard in study and guard in content


@pytest.mark.parametrize("name", ["records", "split"])
def test_consumer_inputs_must_hash_to_the_a_study(ready, monkeypatch, name):
    """The manifest's contract paths are used verbatim by the consumer, so they must carry the A study's bytes."""
    module = ready["module"]
    wire_c_side(ready, monkeypatch)
    manifest_path = Path(ready["request"]["enrollment"]["path"])
    manifest = json.loads(manifest_path.read_text())
    key = "records_path" if name == "records" else "source_split_path"
    Path(manifest["contract"][key]).write_text(json.dumps({"synthetic": "other content"}))
    with pytest.raises(ValueError, match=f"C enrollment {key} must hash to the A study {name}"):
        module.score(ready["binding"], device="cpu")


def test_scoring_source_may_differ_from_the_training_source(ready, monkeypatch):
    """A later frozen source may score, bound by scoring_source_sha256; source_sha256 stays the A training source."""
    module = ready["module"]
    ready["request"]["scoring_source_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="frozen scientific source differs"):
        module.load_request(rebind(ready))
    monkeypatch.setattr(module, "object_sha256", lambda entries: "b" * 64)
    request = module.load_request(rebind(ready))
    assert request["scoring_source_sha256"] == "b" * 64 and request["source_sha256"] == ready["a_request"]["source_sha256"]
    ready["request"]["unknown_field"] = 1
    with pytest.raises(ValueError, match="strict C v2 scoring request fields"):
        module.load_request(rebind(ready))


def test_request_generator_records_the_scoring_source(ready, tmp_path, monkeypatch):
    generator = load(GENERATOR, "make_protocol_c_v2_request_scoring_source")
    source = tmp_path / "frozen-source"; source.mkdir()
    scoring = tmp_path / "scoring-source"; (scoring / "scripts").mkdir(parents=True)
    # the training source inventories to [] (its hash is the A request's source); the scoring source to a different inventory
    monkeypatch.setattr(generator, "source_entries", lambda root: [] if Path(root).resolve() == source.resolve() else ["scorer"])
    with pytest.raises(ValueError, match="must hold the C v2 scorer"):
        generator.build(source, generator.object_sha256([]), ready["a_binding"]["path"], ready["gate"]["path"],
                        str(tmp_path / "c-v2-output"), scoring_source=scoring)
    (scoring / "scripts/score_protocol_c_v2.py").write_text("")
    output = tmp_path / "requests/c-v2.json"
    assert generator.main([str(source), generator.object_sha256([]), ready["a_binding"]["path"], ready["gate"]["path"],
                           str(tmp_path / "c-v2-output"), "--output", str(output), "--scoring-source", str(scoring)]) == 0
    written = json.loads(output.read_text())
    assert written["scoring_source_sha256"] == generator.object_sha256(["scorer"]) != written["source_sha256"]
    assert written["source_sha256"] == generator.object_sha256([])
