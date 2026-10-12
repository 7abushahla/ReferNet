import hashlib
from contextlib import contextmanager
import importlib.util
import json
from pathlib import Path
import random
from types import SimpleNamespace

import pandas as pd
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/profile_direct_reference_getter.py"


@pytest.fixture
def profiler():
    assert SCRIPT.is_file(), "external getter profiler missing"
    spec = importlib.util.spec_from_file_location("getter_profile_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def binding(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value).encode() if not isinstance(value, bytes) else value
    path.write_bytes(payload)
    return dict(path=str(path), sha256=hashlib.sha256(payload).hexdigest())


@pytest.fixture
def frozen(profiler, monkeypatch, tmp_path):
    stage = tmp_path / "stage"
    source = stage / "code"
    run = tmp_path / "run"
    request = binding(stage / "inputs/request.json", {})
    entries = []
    for name in ("scripts/benchmark_direct_reference_throughput.py", "scripts/direct_reference_job.py"):
        value = binding(source / name, b"original frozen code")
        entries.append(dict(path=name, sha256=value["sha256"], size=20))
    source_sha = hashlib.sha256(json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    manifest = dict(source_files=entries, source_sha256=source_sha, request=request,
                    paths=dict(source_dir=str(source), stage_dir=str(stage), run_dir=str(run)), authorities={})
    manifest_binding = binding(run / "input_manifest.json", manifest)
    report = dict(completed=True, execution_probe_only=True, scientific_pilot=False, no_val_test_open=True,
                  arm="off", recipe_batch_size=8, request=request, identities=dict(source_sha256=source_sha))
    report_binding = binding(run / "throughput/report.json", report)
    for name, value in dict(SOURCE_SHA=source_sha, REQUEST_SHA=request["sha256"], REPORT_SHA=report_binding["sha256"],
                            MANIFEST_SHA=manifest_binding["sha256"], STAGE=stage).items():
        monkeypatch.setattr(profiler, name, value)
    monkeypatch.setenv("SLURMD_NODENAME", "gpu-dy-g5-0-17")
    monkeypatch.setenv("OMP_NUM_THREADS", "4")
    monkeypatch.setenv("MKL_NUM_THREADS", "4")
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    events = []
    context = dict(manifest=manifest, request=request, identities=report["identities"])
    api = SimpleNamespace(diagnostic=SimpleNamespace(load_context=lambda *args: context))
    benchmark = SimpleNamespace(allocation_guard=lambda: events.append("guard"), runtime=lambda: events.append("runtime") or api)
    staging = SimpleNamespace(source_digest=lambda source: source_sha, read_json=lambda path: json.loads(path.read_bytes()))
    monkeypatch.setattr(profiler, "load_script", lambda path: events.append(path.name) or
                        (benchmark if path.name.startswith("benchmark") else staging))
    args = SimpleNamespace(source_dir=str(source), request=request["path"], request_sha256=request["sha256"],
                           run_dir=str(run), report=report_binding["path"], report_sha256=report_binding["sha256"],
                           output_root=str(tmp_path / "new-profile"))
    return args, events, source, stage, run


def test_authenticated_bootstrap_guards_before_runtime(profiler, frozen):
    args, events, *_ = frozen
    profiler.bootstrap(args)
    assert events.index("guard") < events.index("runtime")


@pytest.mark.parametrize("failure", ["source", "report", "request", "manifest", "gone", "node", "overlap", "symlink", "extra_source", "bytecode"])
def test_bootstrap_failures_prevent_scientific_imports(profiler, frozen, monkeypatch, failure):
    args, events, source, stage, run = frozen
    if failure == "source":
        (source / "scripts/direct_reference_job.py").write_bytes(b"changed")
    elif failure == "report":
        Path(args.report).write_text("{}")
    elif failure == "request":
        Path(args.request).write_text("changed")
    elif failure == "manifest":
        (run / "input_manifest.json").write_text("{}")
    elif failure == "gone":
        args.source_dir = str(stage / "missing")
    elif failure == "node":
        monkeypatch.setenv("SLURMD_NODENAME", "gpu-dy-g5-0-18")
    elif failure == "overlap":
        args.output_root = str(stage / "profile")
    elif failure == "symlink":
        linked = stage.parent / "linked"
        linked.symlink_to(source, target_is_directory=True)
        args.source_dir = str(linked)
    elif failure == "bytecode":
        binding(source / "src/torch.pyc", b"undeclared executable bytecode")
    else:
        original = profiler.load_script
        def changed_inventory(path):
            module = original(path)
            if path.name == "direct_reference_job.py":
                module.source_digest = lambda source: "0" * 64
            return module
        monkeypatch.setattr(profiler, "load_script", changed_inventory)
    with pytest.raises((ValueError, OSError)):
        profiler.bootstrap(args)
    assert "runtime" not in events


@pytest.fixture
def fake_probe():
    rows = pd.DataFrame([dict(image_id=f"train{index:04d}", role="train") for index in range(48)])
    events, accessed = [], []
    class Dataset:
        geometry_rng = random.Random(42)
        photometry_rng = random.Random(43)
        def __getitem__(self, index):
            accessed.append(index)
            self.geometry_rng.random()
            self.photometry_rng.random()
            return dict(meta=dict(image_id=rows.iloc[index].image_id, role="train"))
    def dataset(context, prepared, selected, enabled, augmented):
        assert not enabled and augmented and selected.equals(rows)
        events.append("constructor")
        return Dataset()
    def collate(samples):
        return dict(query=SimpleNamespace(shape=[8, 4, 512, 800]),
                    meta=dict(image_id=[sample["meta"]["image_id"] for sample in samples], role=["train"] * 8))
    def step(model, batches, optimizer, scheduler, device):
        assert len(tuple(batches)) == 1
        events.append("step")
        return dict(n_images=8, optimizer_steps=1)
    api = SimpleNamespace(
        diagnostic=SimpleNamespace(_dataset=dataset, _model=lambda context, enabled: events.append("model") or object()),
        torch=SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda: events.append("sync")),
                              utils=SimpleNamespace(data=SimpleNamespace(default_collate=collate))),
        pilots=SimpleNamespace(make_pilot_optimizer=lambda *args: object(), pilot_scheduler=lambda *args: object()),
        core=SimpleNamespace(train_direct_reference_epoch=step))
    return rows, api, events, accessed


def test_one_constructor_48_unique_getters_six_discarded_steps(profiler, fake_probe):
    rows, api, events, accessed = fake_probe
    result, profile = profiler.measure(dict(device="cuda"), {}, rows, api)
    assert accessed == list(range(48))
    assert events.count("constructor") == 1 and events.count("model") == 1 and events.count("step") == 6
    assert result["warmup_images"] == 16 and result["timed_images"] == 32
    assert result["optimizer_steps"] == 6 and result["input_shape"] == [8, 4, 512, 800]
    assert all(result[name] > 0 for name in ("constructor_seconds", "getter_seconds", "gpu_step_seconds", "end_to_end_seconds"))
    assert len(result["rng_final_sha256"]["geometry_rng"]) == 64
    profile.create_stats()
    calls = sum(value[1] for key, value in profile.stats.items() if key[0] == __file__ and key[2] == "__getitem__")
    assert calls == 32
    for name, seed in (("geometry_rng", 42), ("photometry_rng", 43)):
        expected = random.Random(seed)
        for _ in range(48): expected.random()
        assert result["rng_final_sha256"][name] == hashlib.sha256(repr(expected.getstate()).encode()).hexdigest()


def test_real_exhaustive_inventory_rejects_undeclared_shadow_before_import(profiler, frozen, monkeypatch):
    args, events, source, _, run = frozen
    root = SCRIPT.parents[1]
    for name in ("anchor_job_manifest.py", "direct_reference_job.py"):
        (source / "scripts" / name).write_bytes((root / "scripts" / name).read_bytes())
    spec = importlib.util.spec_from_file_location("synthetic_frozen_staging", source / "scripts/direct_reference_job.py")
    staging = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(staging)
    entries = staging.helpers().source_entries(source)
    source_sha = staging.helpers().object_sha256(entries)
    manifest = json.loads((run / "input_manifest.json").read_bytes())
    manifest.update(source_files=entries, source_sha256=source_sha)
    manifest_binding = binding(run / "input_manifest.json", manifest)
    report = json.loads(Path(args.report).read_bytes())
    report["identities"]["source_sha256"] = source_sha
    report_binding = binding(Path(args.report), report)
    args.report_sha256 = report_binding["sha256"]
    for name, value in dict(SOURCE_SHA=source_sha, REPORT_SHA=report_binding["sha256"], MANIFEST_SHA=manifest_binding["sha256"]).items():
        monkeypatch.setattr(profiler, name, value)
    original = profiler.load_script
    monkeypatch.setattr(profiler, "load_script", lambda path: staging if path.name == "direct_reference_job.py" else original(path))
    binding(source / "src/torch.py", b"raise RuntimeError('shadow executed')")
    with pytest.raises(ValueError, match="undeclared source file"):
        profiler.bootstrap(args)
    assert "runtime" not in events


@pytest.mark.parametrize("failure", ["val", "duplicate", "count"])
def test_bad_selection_prevents_model_and_constructor(profiler, fake_probe, failure):
    rows, api, events, _ = fake_probe
    if failure == "val": rows.loc[0, "role"] = "val"
    elif failure == "duplicate": rows.loc[0, "image_id"] = rows.loc[1, "image_id"]
    else: rows = rows.iloc[:32]
    with pytest.raises(ValueError):
        profiler.measure(dict(device="cuda"), {}, rows, api)
    assert not events


@pytest.mark.parametrize("failure", [None, "table", "ledger", "drift", "threads", "existing_output"])
def test_run_reuses_full_table_and_publishes_only_authenticated_success(profiler, fake_probe, tmp_path, monkeypatch, failure):
    selected, api, events, accessed = fake_probe
    rows = pd.DataFrame([dict(image_id=f"train{index:04d}", role="train") for index in range(1778)])
    run = tmp_path / "old-run"
    run.mkdir()
    request = binding(tmp_path / "request.json", {})
    registration = binding(run / "registration.json", {})
    phase = binding(run / "phase.json", {})
    manifest = dict(authorities={name: dict(path=str(tmp_path / name), sha256="a" * 64) for name in ("records", "split")})
    manifest_binding = binding(run / "input_manifest.json", manifest)
    report = dict(selected_train_ids=rows.image_id.iloc[:128].tolist(), registration=registration, phase_authority=phase)
    report_binding = binding(run / "report.json", report)
    context = dict(manifest=manifest, identities=dict(source_sha256="a" * 64), request=request, run_dir=run, device="cuda")
    args = SimpleNamespace(request=request["path"], request_sha256=request["sha256"], run_dir=str(run),
                           report=report_binding["path"], report_sha256=report_binding["sha256"], output_root=str(tmp_path / "profile"))
    monkeypatch.setattr(profiler, "REQUEST_SHA", request["sha256"])
    monkeypatch.setattr(profiler, "REPORT_SHA", report_binding["sha256"])
    monkeypatch.setattr(profiler, "MANIFEST_SHA", manifest_binding["sha256"])
    monkeypatch.setattr(profiler, "bootstrap", lambda args: (api, context, report))
    monkeypatch.setenv("OMP_NUM_THREADS", "4")
    monkeypatch.setenv("MKL_NUM_THREADS", "4")
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    api.torch.set_num_threads = lambda count: events.append(("torch_threads", count))
    api.torch.set_num_interop_threads = lambda count: events.append(("interop_threads", count))
    api.cv2 = SimpleNamespace(setNumThreads=lambda count: events.append(("opencv_threads", count)))
    api.load_role_rows = lambda *args: rows
    enrollment = object()
    preprocess = dict(enrollment_authority={})
    api.diagnostic._preprocess = lambda context, authority: preprocess if authority == phase else pytest.fail("wrong phase")
    api.diagnostic.load_context = lambda *args: {**context, "identities": {"source_sha256": "changed"}} if failure == "drift" else context
    def load_table(path, sha, **kwargs):
        assert dict(path=path, sha256=sha) == registration
        assert kwargs["rows"].equals(rows) and len(kwargs["rows"]) == 1778
        assert kwargs["enrollment_manifest"] is enrollment and kwargs["preprocess"] is preprocess
        events.append("load_original_table")
        if failure == "table": raise ValueError("original registration rejected")
        return object()
    api.bridge = SimpleNamespace(_bytes=lambda binding: b"", _json=lambda binding: enrollment, load_registration_table=load_table)
    @contextmanager
    def audit(context, allowed):
        assert allowed.equals(selected)
        opened = []
        yield opened
        opened.extend(dict(image_id=rows.iloc[index].image_id, role="train", path=f"raw/{index}") for index in accessed)
        if failure == "ledger": opened[-1]["role"] = "val"
    api.diagnostic.pixel_audit = audit
    def receipt(path, value):
        with path.open("x") as stream: json.dump(value, stream)
    api.diagnostic.write_receipt = receipt
    if failure == "threads": monkeypatch.setenv("OMP_NUM_THREADS", "1")
    if failure == "existing_output": Path(args.output_root).mkdir()
    if failure:
        with pytest.raises((ValueError, FileExistsError)):
            profiler.run(args)
        assert not (Path(args.output_root) / "report.json").exists()
        if failure in ("table", "threads", "existing_output"): assert "model" not in events
    else:
        result = profiler.run(args)
        output = Path(args.output_root)
        assert set(path.name for path in output.iterdir()) == {"raw.pstats", "report.json"}
        assert result["registration_population"] == 1778 and len(result["query_reads"]) == 48
        assert result["registration"] == registration and result["phase_authority"] == phase
        assert result["identities"] == context["identities"] and result["instrumentation"]["path"] == str(SCRIPT)
        assert result["raw_pstats"]["sha256"] == hashlib.sha256((output / "raw.pstats").read_bytes()).hexdigest()
        assert events.index("load_original_table") < events.index("constructor")
        assert events.count("constructor") == 1 and events.count("step") == 6
        assert result["top_self"] and result["top_cumulative"]
