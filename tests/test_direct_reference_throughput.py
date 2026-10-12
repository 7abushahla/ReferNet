from contextlib import contextmanager
import importlib.util
import json
from pathlib import Path
import random
from types import SimpleNamespace

import pandas as pd
import pytest
import torch

from test_direct_reference_pilot import SyntheticClassifier
from refer_or_clear import direct_reference_diagnostic as diagnostic
from refer_or_clear import direct_reference_pilot as core
from refer_or_clear import pilots


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/benchmark_direct_reference_throughput.py"


def test_script_exists():
    assert SCRIPT.is_file(), "bounded direct-reference throughput script missing"


@pytest.fixture
def benchmark():
    assert SCRIPT.is_file(), "bounded direct-reference throughput script missing"
    spec = importlib.util.spec_from_file_location("throughput_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def probe(monkeypatch, tmp_path):
    rows = pd.DataFrame([dict(image_id=f"train{index:04d}", role="train", path=f"{index}.png")
                         for index in range(1778)])
    context = dict(device=torch.device("cpu"), run_dir=tmp_path, config=dict(seed=42, input_shape=[4, 32, 32]),
                   identities=dict(source_sha256="a" * 64), request=dict(path=str(tmp_path / "request"), sha256="b" * 64))
    events, models, samples = [], [], []

    class ToyDataset:
        def __init__(self, selected):
            self.rows = selected.to_dict("records")
            self.geometry_rng = random.Random(42)

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, index):
            row = self.rows[index]
            samples.append(row["image_id"])
            query = torch.zeros(4, 32, 32)
            query[0].fill_(self.geometry_rng.random())
            return dict(query=query, overlap=torch.ones(1, 32, 32), context_available=torch.tensor(False),
                        label=index % 2, meta=dict(image_id=row["image_id"], role=row["role"]))

    def dataset(context, prepared, selected, enabled, augmented=False):
        assert not enabled and augmented and selected.role.eq("train").all()
        events.append(("dataset", selected.image_id.tolist()))
        return ToyDataset(selected)

    def model(context, enabled):
        assert not enabled
        diagnostic._seed(42)
        value = SyntheticClassifier(False)
        models.append(value)
        return value

    @contextmanager
    def audit(context, selected):
        assert selected.role.eq("train").all()
        yield []

    for name in ("synchronize", "reset_peak_memory_stats", "empty_cache"):
        monkeypatch.setattr(torch.cuda, name, lambda name=name: events.append(name))
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 1024)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: 2048)
    api = SimpleNamespace(torch=torch, core=core, pilots=pilots,
                          diagnostic=SimpleNamespace(_dataset=dataset, _model=model, _seed=diagnostic._seed,
                                                     pixel_audit=audit))
    return dict(api=api, context=context, prepared={}, rows=rows, events=events, models=models, samples=samples)


def test_three_modes_count_updates_and_reset_fresh_state(benchmark, probe):
    selected = probe["rows"].iloc[:64]
    result = benchmark.measure_candidate(probe["context"], probe["prepared"], selected, 8, 1, 2, probe["api"])
    assert result["status"] == "completed" and result["batch_size"] == 8
    assert set(result["modes"]) == {"loader_only", "gpu_only", "end_to_end"}
    for mode, report in result["modes"].items():
        assert report["warmup_batches"] == 1 and report["timed_batches"] == 2 and report["timed_images"] == 16
        assert report["seconds"] > 0 and report["images_per_second"] > 0
        assert report["optimizer_steps"] == (0 if mode == "loader_only" else 3)
        assert report["clip_norm"] == (None if mode == "loader_only" else 5.0)
    assert len(probe["models"]) == 2 and probe["models"][0] is not probe["models"][1]
    assert result["modes"]["gpu_only"]["initial_state_sha256"] == result["modes"]["end_to_end"]["initial_state_sha256"]
    assert result["modes"]["gpu_only"]["peak_allocated_bytes"] == 1024
    assert probe["events"].count("synchronize") >= 6
    assert probe["samples"][:24] == probe["samples"][-24:]
    assert not any(key in result for key in ("accuracy", "auc", "loss", "predictions", "checkpoint"))


@pytest.mark.parametrize("batch_size", [8, 16, 32])
def test_candidate_batches_remain_bounded_and_training_only(benchmark, probe, batch_size):
    result = benchmark.measure_candidate(probe["context"], {}, probe["rows"].iloc[:64], batch_size, 1, 2, probe["api"])
    assert result["modes"]["end_to_end"]["timed_images"] == 2 * batch_size
    assert all(identifier.startswith("train") for identifier in probe["samples"])
    assert result["execution_probe_only"] is True


@pytest.mark.parametrize("role", ["val", "test"])
def test_selected_forbidden_roles_rejected_before_dataset_or_model(benchmark, probe, role):
    rows = probe["rows"].iloc[:64].copy()
    rows.loc[rows.index[0], "role"] = role
    with pytest.raises(ValueError, match="TRAIN"):
        benchmark.measure_candidate(probe["context"], {}, rows, 8, 1, 2, probe["api"])
    assert probe["events"] == [] and not probe["models"]


def test_nonfinite_gradients_fail_in_real_training_core(benchmark, probe, monkeypatch):
    original = probe["api"].diagnostic._model

    def invalid(context, enabled):
        model = original(context, enabled)
        model.encoder.register_hook(lambda gradient: gradient * float("nan"))
        return model

    monkeypatch.setattr(probe["api"].diagnostic, "_model", invalid)
    with pytest.raises(ValueError, match="gradient"):
        benchmark.measure_candidate(probe["context"], {}, probe["rows"].iloc[:64], 8, 1, 2, probe["api"])


def test_only_cuda_oom_is_reported_not_other_failures(benchmark, probe, monkeypatch):
    def oom(*args, **kwargs):
        raise torch.cuda.OutOfMemoryError("synthetic capacity limit")

    monkeypatch.setattr(probe["api"].diagnostic, "_model", oom)
    result = benchmark.measure_candidate(probe["context"], {}, probe["rows"].iloc[:64], 32, 1, 2, probe["api"])
    assert result["status"] == "cuda_oom" and result["modes"]["loader_only"]["timed_images"] == 64


def test_prepare_registers_all_train_in_private_namespace(benchmark, probe, tmp_path):
    context = probe["context"]
    context["manifest"] = dict(authorities={"records": {}, "split": {}}, paths={"raw_root": str(tmp_path / "raw")})
    prepared_contexts = []

    def write(path, payload):
        path.write_text(json.dumps(payload))
        return dict(path=str(path), sha256="c" * 64)

    def preprocess(context, binding):
        return dict(phase_authority=binding, enrollment_authority={})

    def build(rows, raw, manifest, *, preprocess):
        assert len(rows) == 1778 and rows.role.eq("train").all()
        phase = json.loads(Path(preprocess["phase_authority"]["path"]).read_text())
        assert phase["phase"] == "overfit" and phase["authorized_image_ids"]["val"] == []
        assert len(phase["authorized_image_ids"]["train"]) == 1778
        prepared_contexts.append(preprocess)
        return dict(entries={identifier: {} for identifier in rows.image_id})

    probe["api"].bridge = SimpleNamespace(_bytes=lambda binding: b"metadata", _json=lambda binding: {},
                                         build_registration_table=build,
                                         load_registration_table=lambda *args, **kwargs: pytest.fail("unnecessary second query scan"))
    probe["api"].diagnostic._preprocess = preprocess
    probe["api"].diagnostic.write_receipt = write
    probe["api"].load_role_rows = lambda records, split, role: probe["rows"].copy() if role == "train" else pytest.fail("val join")
    prepared, rows = benchmark.prepare(context, tmp_path / "throughput", probe["api"])
    assert len(rows) == 1778 and prepared["preprocess"] == prepared_contexts[0]
    assert (tmp_path / "throughput/phase-authority.json").is_file()
    assert (tmp_path / "throughput/registration.json").is_file()
    assert not (tmp_path / "diagnostic").exists() and not (tmp_path / "training").exists()
    with pytest.raises(FileExistsError): benchmark.prepare(context, tmp_path / "throughput", probe["api"])


@pytest.mark.parametrize("flag,value", [("--batch-sizes", "64"), ("--warmup-batches", "0"),
                                      ("--timed-batches", "17"), ("--sample-count", "2"),
                                      ("--torch-threads", "8"), ("--device", "cpu"), ("--num-workers", "1")])
def test_cli_rejects_unbounded_or_unapproved_settings_before_imports(benchmark, monkeypatch, flag, value):
    monkeypatch.setattr(benchmark, "runtime", lambda: pytest.fail("runtime before CLI validation"))
    with pytest.raises(SystemExit) as stopped:
        benchmark.main(["--request", "/request", "--request-sha256", "a" * 64,
                        "--run-dir", "/run", "--device", "cuda", flag, value])
    assert stopped.value.code == 2


@pytest.mark.parametrize("name,value", [("SLURMD_NODENAME", "login01"), ("SLURM_JOB_PARTITION", "cpu"),
                                      ("SLURM_CPUS_PER_TASK", "3"), ("CUDA_VISIBLE_DEVICES", "-1")])
def test_allocation_guards_precede_scientific_imports(benchmark, monkeypatch, name, value):
    environment = dict(SLURM_JOB_ID="123", SLURMD_NODENAME="gpu01", SLURM_JOB_PARTITION="gpu",
                       SLURM_CPUS_PER_TASK="4", SLURM_GPUS_ON_NODE="1", CUDA_VISIBLE_DEVICES="0")
    for key, item in environment.items():
        monkeypatch.setenv(key, item)
    monkeypatch.setenv(name, value)
    monkeypatch.setattr(benchmark, "runtime", lambda: pytest.fail("scientific import on invalid allocation"))
    with pytest.raises(ValueError):
        benchmark.main(["--request", "/request", "--request-sha256", "a" * 64, "--run-dir", "/run", "--device", "cuda"])


def test_import_does_not_load_scientific_runtime_or_execute(benchmark):
    assert callable(benchmark.runtime) and callable(benchmark.main)
    assert not hasattr(benchmark, "torch") and not hasattr(benchmark, "service")


@pytest.mark.parametrize("setting", ["prefix", "interpreter", "conda", "zero-job", "no-gpu"])
def test_strict_environment_and_resource_guards(benchmark, monkeypatch, setting):
    for name, value in dict(SLURM_JOB_ID="123", SLURMD_NODENAME="gpu01", SLURM_JOB_PARTITION="gpu",
                            SLURM_CPUS_PER_TASK="4", SLURM_GPUS_ON_NODE="1", CUDA_VISIBLE_DEVICES="0",
                            CONDA_PREFIX=benchmark.ENVIRONMENT).items():
        monkeypatch.setenv(name, value)
    for name in ("SLURM_GPUS", "SLURM_JOB_GPUS", "SLURM_STEP_GPUS"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(benchmark.sys, "prefix", benchmark.ENVIRONMENT)
    monkeypatch.setattr(benchmark.sys, "executable", benchmark.ENVIRONMENT + "/bin/python")
    if setting == "prefix":
        monkeypatch.setattr(benchmark.sys, "prefix", "/wrong")
    elif setting == "interpreter":
        monkeypatch.setattr(benchmark.sys, "executable", "/wrong/bin/python")
    else:
        monkeypatch.setenv({"conda": "CONDA_PREFIX", "zero-job": "SLURM_JOB_ID", "no-gpu": "SLURM_GPUS_ON_NODE"}[setting],
                           "/wrong" if setting == "conda" else "0")
    with pytest.raises(ValueError):
        benchmark.allocation_guard()


def test_authentication_failure_precedes_preparation_or_timing(benchmark, probe, monkeypatch):
    monkeypatch.setattr(benchmark, "allocation_guard", lambda: None)
    monkeypatch.setattr(benchmark, "runtime", lambda: probe["api"])
    monkeypatch.setattr(torch, "set_num_threads", lambda count: None)
    monkeypatch.setattr(torch, "set_num_interop_threads", lambda count: None)

    def refused(*args):
        raise ValueError("source or request authority mismatch")

    probe["api"].diagnostic.load_context = refused
    with pytest.raises(ValueError, match="authority"):
        benchmark.main(["--request", "/request", "--request-sha256", "a" * 64,
                        "--run-dir", str(probe["context"]["run_dir"]), "--device", "cuda"])
    assert not probe["samples"] and not probe["models"]
    assert not (probe["context"]["run_dir"] / "throughput").exists()
