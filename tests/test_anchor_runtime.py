import importlib.util
import json
import os
import copy
from pathlib import Path
import subprocess

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def bench():
    path = ROOT / "scripts/bench_anchor_pilots.py"
    assert path.is_file(), "anchor benchmark is not implemented"
    spec = importlib.util.spec_from_file_location("anchor_bench", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def host():
    return {"hostname": "allocated01", "boot_id": "boot01", "cpu_model": "fixture CPU",
            "affinity": [0, 1, 2, 3], "allocated_cpus": 4, "job_id": "123",
            "os": "fixture", "versions": {"torch": "fixture"}}


def test_fake_clock_includes_preparation_and_materialized_score(bench):
    events = []
    ticks = iter(float(index) for index in range(400 * 3))
    class Dataset:
        def __getitem__(self, index):
            events.append(("prepare", index))
            return {"query": torch.zeros(3, 1, 1), "meta": {"image_id": f"{index:04d}"}}
    class Score:
        def reshape(self, *shape):
            return self
        def __getitem__(self, index):
            return self
        def item(self):
            events.append(("materialize", None))
            return 0.
    class Model:
        def __call__(self, query):
            events.append(("network", None))
            return {"attack_logit": Score()}
    ids = [f"{index:04d}" for index in range(200)]
    result = bench.measure_track(Model(), Dataset(), ids, clock=lambda: next(ticks))
    assert len(result["per_image"]) == 200
    assert len([event for event in events if event[0] == "materialize"]) == 220
    assert result["warmups"] == 20
    assert all(row["preparation_ms"] == 1000 and row["network_ms"] == 1000 for row in result["per_image"])
    assert result["timing"]["prepared_input_total_ms"] == {"median": 2000., "p95": 2000.}


@pytest.mark.parametrize("count", [199, 201])
def test_exactly_200_ids_required(bench, count):
    with pytest.raises(ValueError, match="200"):
        bench.measure_track(None, None, [str(index) for index in range(count)])


def test_manifest_contract_hash_and_host_guards(bench, tmp_path):
    ids = [f"{index:04d}" for index in range(200)]
    manifest = bench.make_manifest(ids, host(), {track: {"checkpoint_sha256": track} for track in "RDN"})
    path = tmp_path / "runtime.json"
    bench.freeze_manifest(path, manifest)
    bench.freeze_manifest(path, manifest)
    changed = dict(manifest, host=dict(host(), hostname="allocated02"))
    with pytest.raises(ValueError, match="manifest"):
        bench.freeze_manifest(path, changed)
    assert json.loads(path.read_text())["contract"]["track_order"] == ["R", "D", "N"]
    with pytest.raises(ValueError, match="host|CPU"):
        bench.make_manifest(ids, dict(host(), allocated_cpus=3), {})
    with pytest.raises(ValueError, match="lexicographic"):
        bench.make_manifest(list(reversed(ids)), host(), {})


def test_original_offline_weights_recovered_and_sha_checked(bench, tmp_path):
    run = tmp_path / "pilot"
    run.mkdir()
    weights = tmp_path / "original.pth"
    weights.write_bytes(b"offline fixture")
    info = {"paths": {"weights": str(tmp_path / "gone/weights.pth")},
            "identities": {"weights_sha256": bench.file_sha256(weights)}}
    (tmp_path / "input_manifest.json").write_text(json.dumps({"paths": {"weights": str(weights)}}))
    assert bench.resolve_weights(run, info) == weights
    weights.write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA256"):
        bench.resolve_weights(run, info)


def test_raw_missing_never_falls_back_to_canvas(bench, tmp_path):
    row = {"image_id": "id", "path": "missing.jpg", "design_id": "design"}
    legacy = tmp_path / "legacy"
    (legacy / "canvases").mkdir(parents=True)
    (legacy / "canvases/id.png").write_bytes(b"not a raw image")
    with pytest.raises(FileNotFoundError, match="missing.jpg"):
        bench.source_artifacts(row, "R", tmp_path / "raw", legacy)


def test_unallocated_host_reports_not_comparable_without_model(bench, monkeypatch, tmp_path):
    monkeypatch.setattr(bench, "host_info", lambda: dict(host(), job_id=None))
    monkeypatch.setattr(bench, "build_initialized_model", lambda *args: pytest.fail("must not build a model"))
    output = tmp_path / "timing.json"
    result = bench.main(["--runs", "r", "d", "n", "--runtime-manifest", str(tmp_path / "manifest.json"),
                         "--raw-root", "raw", "--legacy-root", "legacy", "--out", str(output)])
    assert result["status"] == "not-comparable"
    assert result["tracks"] == {}


def test_wrapper_is_one_allocation_and_has_no_submission():
    path = ROOT / "scripts/cluster/bench_anchor_pilots.sh"
    assert path.is_file(), "anchor benchmark wrapper is not implemented"
    script = path.read_text()
    assert "SLURMD_NODENAME" in script and "SLURM_CPUS_PER_TASK" in script
    assert "/shared/b00090279/conda/envs/refer-or-clear" in script
    assert '"$python_bin"' in script and "--runs" in script
    assert script.count("bench_anchor_pilots.py") == 1
    assert "sbatch " not in script and "srun " not in script
    assert "#SBATCH --partition=cpu" in script
    assert "#SBATCH --gres" not in script and "#SBATCH --gpus" not in script
    assert "source /opt/miniconda3/etc/profile.d/conda.sh" in script
    assert "source /opt/miniconda/etc/profile.d/conda.sh" not in script
    assert 'conda activate "$environment_path"' in script


@pytest.mark.parametrize("partition", ["gpu", "hpc", ""])
def test_wrapper_rejects_non_cpu_allocation_before_environment_or_inputs(partition, tmp_path):
    environment = dict(os.environ, SLURM_JOB_ID="123", SLURMD_NODENAME="allocated01",
                       SLURM_SUBMIT_DIR=str(tmp_path), SLURM_CPUS_PER_TASK="4", SLURM_JOB_PARTITION=partition)
    result = subprocess.run(["bash", str(ROOT / "scripts/cluster/bench_anchor_pilots.sh"),
                             "source", "r", "d", "n", "raw", "legacy", str(tmp_path / "output")],
                            env=environment, capture_output=True, text=True)
    assert result.returncode == 2
    assert "Allocated CPU partition required" in result.stderr
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("override", [None, "/custom scratch"])
def test_cpu_wrapper_scratch_defaults_to_tmp_and_preserves_override(override):
    script = (ROOT / "scripts/cluster/bench_anchor_pilots.sh").read_text()
    assignment = next(line for line in script.splitlines() if line.startswith("scratch="))
    environment = dict(os.environ, USER="fixture-user", SLURM_JOB_ID="123")
    environment.pop("ROC_SCRATCH_ROOT", None)
    if override is not None:
        environment["ROC_SCRATCH_ROOT"] = override
    result = subprocess.run(["bash", "-c", assignment + '\nprintf "%s" "$scratch"'],
                            env=environment, capture_output=True, text=True, check=True)
    assert result.stdout == f"{override or '/tmp/fixture-user'}/anchor-cpu-v1-123"


def test_all_tracks_fixed_order_and_manifest_precedes_warmups(bench, monkeypatch, tmp_path):
    ids = [f"{index:04d}" for index in range(200)]
    manifest_path = tmp_path / "manifest.json"
    events = []
    metadata = {track: {"checkpoint_sha256": track, "n_active_parameters": 1, "n_total_parameters": 1}
                for track in "RDN"}
    prepared = {track: ({"config": {"track": track}, "identities": {"weights_sha256": "a" * 64}},
                        "weights", "checkpoint", "dataset") for track in "RDN"}
    class FakeModel(torch.nn.Module):
        def __init__(self, track):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))
            self.track = track
            self.initialization_provenance = {"pretrained_tensor_sha256": "b" * 64}
    monkeypatch.setattr(bench, "host_info", host)
    monkeypatch.setattr(bench, "configure_threads", lambda: None)
    monkeypatch.setattr(bench, "prepare_runs", lambda *args: (ids, metadata, prepared))
    monkeypatch.setattr(bench, "verify_source_artifacts", lambda *args: None)
    original_hash = bench.file_sha256
    monkeypatch.setattr(bench, "file_sha256", lambda path: "checkpoint" if path == "checkpoint" else original_hash(path))
    for track in "RDN":
        metadata[track]["checkpoint_sha256"] = "checkpoint"
    def builder(config, *args):
        assert manifest_path.is_file()
        events.append(("build", config["track"]))
        return FakeModel(config["track"])
    monkeypatch.setattr(bench, "build_initialized_model", builder)
    monkeypatch.setattr(bench, "read_pilot_checkpoint", lambda *args: {
        "state_dict": {"weight": torch.ones(1)}, "initialization_provenance": {"pretrained_tensor_sha256": "b" * 64}})
    def measure(model, dataset, image_ids):
        assert not model.training and image_ids == ids
        events.append(("time", model.track))
        return {"warmups": 20, "n_documents": 200}
    monkeypatch.setattr(bench, "measure_track", measure)
    result = bench.main(["--runs", "r", "d", "n", "--runtime-manifest", str(manifest_path),
                         "--raw-root", "raw", "--legacy-root", "legacy", "--out", str(tmp_path / "out.json")])
    assert result["status"] == "measured", result
    assert events == [(action, track) for track in "RDN" for action in ("build", "time")]


def test_manifest_sources_must_not_change_during_timing(bench, tmp_path):
    source = tmp_path / "raw.png"
    source.write_bytes(b"original")
    track = {"sources": [{"image_id": "id", "artifacts": [{"path": str(source), "sha256": bench.file_sha256(source)}]}]}
    bench.verify_source_artifacts(track)
    source.write_bytes(b"changed")
    with pytest.raises(ValueError, match="source.*SHA256"):
        bench.verify_source_artifacts(track)


def test_not_enough_affinity_or_login_host_rejected(bench):
    with pytest.raises(ValueError, match="CPUs"):
        bench.validate_host(dict(host(), affinity=[0, 1]))
    with pytest.raises(ValueError, match="login"):
        bench.validate_host(dict(host(), hostname="login01"))


@pytest.fixture
def measurement_fixture(bench, tmp_path):
    ids = [f"{index:04d}" for index in range(200)]
    (tmp_path / "checkpoints").mkdir()
    checkpoint = tmp_path / "checkpoints/best.pt"
    checkpoint.write_bytes(b"selected checkpoint fixture")
    checkpoint_sha = bench.file_sha256(checkpoint)
    identities = {"source_sha256": "a" * 64}
    train_rows = [{"image_id": f"{index:04d}", "path": f"native/{index:04d}.jpg", "design_id": "design"}
                  for index in range(1778)]
    (tmp_path / "train.json").write_text(json.dumps({"membership": {"train": train_rows}}))
    candidates = {track: {"status": "complete", "identities": identities, "run_dir": str(tmp_path),
                          "n_active_parameters": 1, "n_total_parameters": 2} for track in "RDN"}
    metadata = {track: {"checkpoint_sha256": checkpoint_sha, "identities": identities,
                        "input_shape": [1, *bench.contract_for(track).input_shape],
                        "n_active_parameters": 1, "n_total_parameters": 2,
                        "preparation_scope": "raw_input" if track == "R" else "processed_input_only"} for track in "RDN"}
    for track in "RDN":
        source_root = tmp_path / ("raw" if track == "R" else "legacy")
        metadata[track]["source_roots"] = {"raw": str(tmp_path / "raw"), "legacy": str(tmp_path / "legacy")}
        sources = []
        for row in train_rows[:200]:
            relatives = [row["path"]] if track == "R" else [f"canvases/{row['image_id']}.png", "zones/design.png"]
            sources.append({"image_id": row["image_id"], "artifacts": [
                {"path": str(source_root / relative), "sha256": "b" * 64, "shape": [512, 800, 3] if index == 0 else [512, 800]}
                for index, relative in enumerate(relatives)]})
        metadata[track].update(sources=sources, sources_sha256=bench.json_sha256(sources))
    manifest = bench.make_manifest(ids, host(), metadata)
    durations = [{"image_id": image_id, "preparation_ms": 1., "network_ms": 2., "prepared_input_total_ms": 3.}
                 for image_id in ids]
    result = {"schema_version": "anchor_cpu_v1", "status": "measured", "manifest": manifest,
              "manifest_sha256": manifest["manifest_sha256"],
              "tracks": {track: dict(metadata[track], host=host(), image_ids_sha256=manifest["image_ids_sha256"],
                                      warmups=20, n_documents=200, per_image=durations,
                                      timing={key: bench.stats([row[key] for row in durations])
                                              for key in ("preparation_ms", "network_ms", "prepared_input_total_ms")})
                         for track in "RDN"}}
    return result, candidates


def test_measured_artifact_revalidates_common_host_counts_and_statistics(bench, measurement_fixture):
    result, candidates = measurement_fixture
    bench.validate_measurement(result, candidates)
    result["tracks"]["N"]["host"] = dict(host(), hostname="different-host")
    with pytest.raises(ValueError, match="host"):
        bench.validate_measurement(result, candidates)
    result["tracks"]["N"]["host"] = host()
    result["tracks"]["R"]["timing"]["network_ms"]["p95"] = 999.
    with pytest.raises(ValueError, match="summary"):
        bench.validate_measurement(result, candidates)
    result["tracks"]["R"]["timing"]["network_ms"]["p95"] = 2.
    result["tracks"]["D"]["input_shape"] = [1, 8, 512, 800]
    with pytest.raises(ValueError, match="shape"):
        bench.validate_measurement(result, candidates)


@pytest.mark.parametrize("damage", ["foreign_ids", "not_first_200", "wrong_raw_mapping", "wrong_zone_mapping", "result_sources"])
def test_runtime_must_bind_first_200_train_ids_and_declared_sources(bench, measurement_fixture, damage):
    original, candidates = measurement_fixture
    result = copy.deepcopy(original)
    manifest = result["manifest"]
    if damage in ("foreign_ids", "not_first_200"):
        ids = [f"foreign{index:04d}" for index in range(200)] if damage == "foreign_ids" else [f"{index:04d}" for index in range(200, 400)]
        manifest.update(image_ids=ids, image_ids_sha256=bench.json_sha256(ids))
        for track in "RDN":
            measured = result["tracks"][track]
            measured["image_ids_sha256"] = manifest["image_ids_sha256"]
            for index, image_id in enumerate(ids):
                manifest["tracks"][track]["sources"][index]["image_id"] = image_id
                measured["per_image"][index]["image_id"] = image_id
            measured["sources"] = copy.deepcopy(manifest["tracks"][track]["sources"])
    elif damage == "wrong_raw_mapping":
        manifest["tracks"]["R"]["sources"][0]["artifacts"][0]["path"] = "/elsewhere/raw/another.jpg"
        result["tracks"]["R"]["sources"] = copy.deepcopy(manifest["tracks"]["R"]["sources"])
    elif damage == "wrong_zone_mapping":
        manifest["tracks"]["D"]["sources"][0]["artifacts"][1]["path"] = "/elsewhere/legacy/zones/other.png"
        result["tracks"]["D"]["sources"] = copy.deepcopy(manifest["tracks"]["D"]["sources"])
    else:
        result["tracks"]["N"]["sources"] = []
    for track in "RDN":
        manifest["tracks"][track]["sources_sha256"] = bench.json_sha256(manifest["tracks"][track]["sources"])
        result["tracks"][track]["sources_sha256"] = bench.json_sha256(result["tracks"][track]["sources"])
    unsigned = dict(manifest)
    unsigned.pop("manifest_sha256")
    manifest["manifest_sha256"] = result["manifest_sha256"] = bench.json_sha256(unsigned)
    with pytest.raises(ValueError, match="train IDs|source"):
        bench.validate_measurement(result, candidates)
