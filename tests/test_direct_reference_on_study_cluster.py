import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile

import pytest

from test_direct_reference_cluster import ENV_PATH, ROOT, calls, cluster_fixture, sha256
from test_direct_reference_on_cluster import on_fixture


WRAPPER = ROOT / "scripts/cluster/train_direct_reference_on_study.sh"
STAGE = "/scratch/b00090279/direct-reference-control-38041"
SOURCE_SHA = "a5ff28ae2a332b977551241f2d07641d51df78d54a4cc1b58169c7338ce11292"
LAUNCHER_SHA = "b129f650da89f047473c0bd4a42f5ddcfcbd1ef97552fabb6fcfbe325c7efc04"
REQUEST_SHA = "2f6d479835003242790f6d10f44249d73aa9a43f6a465132e618bc99c24c87b9"
MANIFEST_SHA = "dbe33a68c945cb5f6b5c4140fb6000e293342150537bef4d09ff2304d46d613a"


def write_json(path, value):
    path.write_text(json.dumps(value))
    return {"path": str(path), "sha256": sha256(path)}


def pack(fixture, extra=None):
    with tarfile.open(fixture["archive"], "w") as archive:
        for root in ("code", "inputs", "raw", "enrollment"):
            archive.add(fixture["snapshot"] / root, arcname=root)
        if extra is not None:
            member, content = extra
            archive.addfile(member, io.BytesIO(content) if member.isfile() else None)


@pytest.fixture
def study_fixture(on_fixture):
    fixture = on_fixture
    authority = json.loads(fixture["authority"].read_text())
    authority.update(schema_version="direct_reference_on_study_request_v1",
                     anchor_policy="report_observed_no_performance_gate")
    completed_path = Path(authority["off_completion"]["path"])
    completed = json.loads(completed_path.read_text())
    completed["passed"] = True
    train_path = Path(completed["train_result"]["path"])
    selected = json.loads(train_path.read_text())
    selected["epochs_completed"] = 100
    completed["train_result"] = write_json(train_path, selected)
    replay_path = Path(completed["fresh_replay"]["path"])
    replay = json.loads(replay_path.read_text())
    replay["train_result"] = completed["train_result"]
    completed["fresh_replay"] = write_json(replay_path, replay)
    authority["off_completion"] = write_json(completed_path, completed)
    write_json(fixture["authority"], authority)
    fixture["manifest_hash"] = sha256(fixture["run"] / "input_manifest.json")
    fixture["request_hash"] = sha256(fixture["snapshot"] / "inputs/request.json")
    return fixture


def run_study(fixture, arguments=None, **changes):
    assert WRAPPER.is_file(), "ON cluster wrapper is not implemented"
    isolated = fixture["tmp_path"] / "wrapper-under-test.sh"
    isolated.write_text(WRAPPER.read_text().replace(STAGE, str(fixture["stage"]))
                        .replace(LAUNCHER_SHA, fixture["launcher_hash"])
                        .replace(REQUEST_SHA, fixture["request_hash"])
                        .replace(MANIFEST_SHA, fixture["manifest_hash"]))
    values = dict(fixture, **changes)
    arguments = arguments if arguments is not None else [str(values["archive"]), values.get("archive_hash", sha256(values["archive"])),
                 str(values["inventory"]), values.get("inventory_hash", sha256(values["inventory"])), str(values["launcher"]),
                 str(values["authority"]), values.get("authority_hash", sha256(values["authority"])), str(values["run"])]
    return subprocess.run(["bash", str(isolated), *arguments], env=values["environment"], capture_output=True, text=True, timeout=20)


def rejected_before_restore(fixture, result):
    assert result.returncode != 0
    assert not fixture["stage"].exists()
    assert calls(fixture) == []


def test_snapshot_restores_exact_bytes_without_shared_output_changes(study_fixture):
    protected = {path: path.read_bytes() for path in study_fixture["run"].rglob("*") if path.is_file()}
    result = run_study(study_fixture)
    assert result.returncode == 0, result.stderr
    for original in study_fixture["snapshot"].rglob("*"):
        restored = study_fixture["stage"] / original.relative_to(study_fixture["snapshot"])
        if original.is_file():
            assert restored.read_bytes() == original.read_bytes()
            assert restored.stat().st_mode & 0o222 == 0
    assert (study_fixture["stage"] / "tmp").is_dir()
    assert all(path.read_bytes() == content for path, content in protected.items())
    observed = calls(study_fixture)
    assert len(observed) == 1 and observed[0]["operation"] == "on"
    assert observed[0]["arguments"] == dict(source_dir=str(study_fixture["stage"] / "code"),
        request=str(study_fixture["stage"] / "inputs/request.json"),
        request_sha256=sha256(study_fixture["snapshot"] / "inputs/request.json"),
        on_request=str(study_fixture["authority"]), on_request_sha256=sha256(study_fixture["authority"]),
        run_dir=str(study_fixture["run"]), device="cuda")
    environment = observed[0]["environment"]
    for name, value in {"OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                        "NVIDIA_TF32_OVERRIDE": "0", "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE": "0", "PYTHONHASHSEED": "42",
                        "PYTHONDONTWRITEBYTECODE": "1", "CONDA_PREFIX": ENV_PATH,
                        "TMPDIR": str(study_fixture["stage"] / "tmp")}.items():
        assert environment[name] == value
    assert all(name not in environment for name in ("PYTHONPATH", "PYTHONHOME", "ROC_ANCHOR_GPU_INTEGRATION",
               "ROC_DIRECT_REFERENCE_GPU_INTEGRATION", "ROC_LEVIR_CKPT", "ROC_LEVIR_SHA256"))


@pytest.mark.parametrize("mutation", ["archive-hash", "inventory-hash", "authority-hash", "launcher-drift",
    "missing-off", "off-malformed", "replay-failed", "existing-on", "table-drift", "phase-drift", "manifest-path"])
def test_untrusted_snapshot_or_incomplete_off_blocks_restore(study_fixture, mutation):
    changes = {}
    if mutation.endswith("-hash"):
        changes[mutation.replace("-", "_")] = "0" * 64
    elif mutation == "launcher-drift":
        study_fixture["launcher"].write_text("raise AssertionError('must not run')")
    elif mutation == "existing-on":
        (study_fixture["run"] / "training/on").mkdir()
    elif mutation == "missing-off":
        (study_fixture["run"] / "training/off/control.json").unlink()
    elif mutation in ("table-drift", "phase-drift"):
        name = "registration.json" if mutation == "table-drift" else "phase-authority.json"
        (study_fixture["run"] / "training/shared" / name).write_text("{}")
    elif mutation == "manifest-path":
        path = study_fixture["run"] / "input_manifest.json"
        value = json.loads(path.read_text())
        value["paths"]["stage_dir"] = "/some/other/stage"
        write_json(path, value)
    else:
        authority = json.loads(study_fixture["authority"].read_text())
        root = Path(authority["off_completion"]["path"])
        completed = json.loads(root.read_text())
        if mutation == "off-failed":
            completed["passed"] = "false"
        else:
            path = Path(completed["fresh_replay"]["path"])
            value = json.loads(path.read_text())
            value["passed"] = False
            completed["fresh_replay"] = write_json(path, value)
        authority["off_completion"] = write_json(root, completed)
        write_json(study_fixture["authority"], authority)
    rejected_before_restore(study_fixture, run_study(study_fixture, **changes))


@pytest.mark.parametrize("field,value", [("arm", "off"), ("epochs", True), ("epochs", 99), ("fold", 1),
    ("seed", 43), ("source_sha256", "0" * 64), ("schema_version", "other"), ("resume", True), ("anchor_policy", "require_competence"),
    ("schema_version", "direct_reference_on_request_v1")])
def test_fixed_on_authority_only(study_fixture, field, value):
    authority = json.loads(study_fixture["authority"].read_text())
    authority[field] = value
    write_json(study_fixture["authority"], authority)
    rejected_before_restore(study_fixture, run_study(study_fixture))


@pytest.mark.parametrize("kind", ["traversal", "absolute", "unexpected", "symlink", "hardlink", "fifo", "duplicate"])
def test_unsafe_archive_members_are_rejected_before_any_extract(study_fixture, kind):
    name = {"traversal": "raw/../escape", "absolute": "/tmp/escape", "unexpected": "training/off/train.json"}.get(kind, "raw/unsafe")
    member = tarfile.TarInfo(name)
    content = b"not allowed"
    member.size = len(content)
    if kind in ("symlink", "hardlink", "fifo"):
        member.type = {"symlink": tarfile.SYMTYPE, "hardlink": tarfile.LNKTYPE, "fifo": tarfile.FIFOTYPE}[kind]
        member.linkname = "raw/train.bin"
        member.size = 0
    if kind == "duplicate":
        member.name = "raw/train.bin"
    pack(study_fixture, (member, content))
    rejected_before_restore(study_fixture, run_study(study_fixture))


@pytest.mark.parametrize("kind", ["missing", "extra", "wrong-bytes", "traversal", "duplicate"])
def test_file_manifest_must_match_archive_exactly(study_fixture, kind):
    path = study_fixture["inventory"]
    lines = path.read_text().splitlines()
    if kind == "missing":
        lines = lines[1:]
    elif kind == "extra":
        lines.append("0" * 64 + "  raw/extra.bin")
    elif kind == "wrong-bytes":
        lines[0] = "0" * 64 + lines[0][64:]
    elif kind == "traversal":
        lines.append("0" * 64 + "  raw/../escape")
    else:
        lines.append(lines[0])
    path.write_text("\n".join(lines) + "\n")
    rejected_before_restore(study_fixture, run_study(study_fixture))


def test_existing_stage_is_never_overwritten(study_fixture):
    study_fixture["stage"].mkdir(parents=True)
    sentinel = study_fixture["stage"] / "original"
    sentinel.write_bytes(b"untouched")
    result = run_study(study_fixture)
    assert result.returncode != 0 and calls(study_fixture) == []
    assert sentinel.read_bytes() == b"untouched"


def test_symlink_stage_parent_is_rejected(study_fixture):
    actual = study_fixture["tmp_path"] / "other"
    actual.mkdir()
    study_fixture["stage"].parent.symlink_to(actual, target_is_directory=True)
    rejected_before_restore(study_fixture, run_study(study_fixture))


@pytest.mark.parametrize("name,value", [("SLURM_JOB_PARTITION", "cpu"), ("SLURMD_NODENAME", "login01"),
    ("SLURM_JOB_ID", "0"), ("SLURM_CPUS_PER_TASK", "3"), ("SLURM_GPUS_ON_NODE", "0"),
    ("CUDA_VISIBLE_DEVICES", "-1"), ("MOCK_PREFIX", "/wrong"), ("MOCK_INTERPRETER_PREFIX", "/wrong"), ("SLURMD_NODENAME", "gpu-dy-g5-0-16")])
def test_wrong_allocation_or_environment_blocks_before_restore(study_fixture, name, value):
    study_fixture["environment"][name] = value
    rejected_before_restore(study_fixture, run_study(study_fixture))


def test_launcher_failure_propagates_without_completion_write(study_fixture):
    study_fixture["environment"]["ON_EXIT"] = "17"
    result = run_study(study_fixture)
    assert result.returncode == 17
    assert not (study_fixture["run"] / "training/on").exists()


@pytest.mark.parametrize("target", ["launcher", "authority"])
def test_launch_authorities_reauthenticated_after_restore(study_fixture, target):
    study_fixture["environment"]["LAUNCH_BOUNDARY_DRIFT"] = str(study_fixture[target])
    result = run_study(study_fixture)
    assert result.returncode != 0
    assert calls(study_fixture) == []
    assert study_fixture["stage"].is_dir()


def test_fixed_slurm_header_and_no_optional_recipe_arguments(study_fixture):
    assert WRAPPER.is_file(), "ON cluster wrapper is not implemented"
    options = dict(line.removeprefix("#SBATCH --").split("=", 1) for line in WRAPPER.read_text().splitlines()
                   if line.startswith("#SBATCH --") and "=" in line)
    assert {name: options[name] for name in ("account", "partition", "nodes", "ntasks", "gres", "cpus-per-task", "mem")} == dict(
        account="acc-mialhajri", partition="gpu", nodes="1", ntasks="1", gres="gpu:1", **{"cpus-per-task": "4", "mem": "8G"})
    assert options["time"] == "18:00:00" and options["job-name"] == "direct-reference-on-study"
    assert options["exclude"] == "gpu-dy-g5-0-16" and "nodelist" not in options
    assert subprocess.run(["bash", "-n", str(WRAPPER)], capture_output=True).returncode == 0
    rejected_before_restore(study_fixture, run_study(study_fixture, arguments=["--epochs", "1"]))


def test_verified_restore_reuse_preserves_original_bytes_and_modes(study_fixture):
    fixture = study_fixture
    first = run_study(fixture)
    assert first.returncode == 0, first.stderr
    before = {path: (path.read_bytes(), path.stat().st_mtime_ns, path.stat().st_mode)
              for path in fixture["stage"].rglob("*") if path.is_file()}
    second = run_study(fixture)
    assert second.returncode == 0, second.stderr
    assert all((path.read_bytes(), path.stat().st_mtime_ns, path.stat().st_mode) == state
               for path, state in before.items())


@pytest.mark.parametrize("mutation", ["modified", "extra", "writable", "symlink"])
def test_reuse_rejects_changed_scratch_without_repair(study_fixture, mutation):
    fixture = study_fixture
    assert run_study(fixture).returncode == 0
    observed = calls(fixture)
    target = fixture["stage"] / "raw/train.bin"
    if mutation in ("modified", "writable"):
        target.chmod(0o644)
        if mutation == "modified":
            target.write_bytes(b"changed")
            target.chmod(0o444)
    else:
        target.parent.chmod(0o755)
        if mutation == "symlink":
            target.unlink()
            target.symlink_to(fixture["snapshot"] / "raw/train.bin")
        else:
            (target.parent / "extra").write_bytes(b"unapproved")
    assert run_study(fixture).returncode != 0
    assert calls(fixture) == observed


@pytest.mark.parametrize("field,value", [("epochs_completed", 99), ("epochs_completed", True),
                                        ("phase", "smoke"), ("arm", "on"), ("passed", False)])
def test_incomplete_or_wrong_off_completion_blocks_study(study_fixture, field, value):
    authority = json.loads(study_fixture["authority"].read_text())
    path = Path(authority["off_completion"]["path"])
    value_doc = json.loads(path.read_text())
    value_doc[field] = value
    authority["off_completion"] = write_json(path, value_doc)
    write_json(study_fixture["authority"], authority)
    rejected_before_restore(study_fixture, run_study(study_fixture))


def test_unapproved_launcher_digest_is_failclosed(study_fixture):
    assert WRAPPER.is_file()
    script = study_fixture["tmp_path"] / "pending-pin-wrapper.sh"
    script.write_text(WRAPPER.read_text().replace(STAGE, str(study_fixture["stage"]))
                      .replace(REQUEST_SHA, study_fixture["request_hash"])
                      .replace(MANIFEST_SHA, study_fixture["manifest_hash"]))
    args = [str(study_fixture["archive"]), sha256(study_fixture["archive"]),
            str(study_fixture["inventory"]), sha256(study_fixture["inventory"]),
            str(study_fixture["launcher"]), str(study_fixture["authority"]),
            sha256(study_fixture["authority"]), str(study_fixture["run"])]
    result = subprocess.run(["bash", str(script), *args], env=study_fixture["environment"],
                            capture_output=True, text=True, timeout=20)
    rejected_before_restore(study_fixture, result)
