import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile

import pytest

from test_direct_reference_cluster import ENV_PATH, ROOT, calls, cluster_fixture, sha256


WRAPPER = ROOT / "scripts/cluster/train_direct_reference_on.sh"
STAGE = "/scratch/b00090279/direct-reference-control-38041"
SOURCE_SHA = "a5ff28ae2a332b977551241f2d07641d51df78d54a4cc1b58169c7338ce11292"
LAUNCHER_SHA = "f2697cbf85067baeba354077dec2bc0427e39fcf71b0005396f0f265a054c012"


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
def on_fixture(cluster_fixture):
    fixture = cluster_fixture
    base = fixture["tmp_path"]
    stage = base / "restored scratch/direct-reference-control-38041"
    snapshot = base / "readonly snapshot"
    for name in ("code", "inputs", "raw", "enrollment"):
        (snapshot / name).mkdir(parents=True)
    (snapshot / "code/source.txt").write_text("synthetic source, never imported")
    (snapshot / "raw/train.bin").write_bytes(b"synthetic TRAIN data, not an image")
    (snapshot / "raw/val.bin").write_bytes(b"synthetic VAL data, not an image")
    (snapshot / "enrollment/bank.bin").write_bytes(b"synthetic fold-safe bank")
    config = snapshot / "inputs/config.yaml"
    config.write_text("epochs: 100\nfold: 0\nseed: 42\n")
    request = snapshot / "inputs/request.json"
    write_json(request, dict(schema_version="direct_reference_diagnostic_request_v1",
                            source_sha256=SOURCE_SHA, config={"path": "/original/config.yaml", "sha256": sha256(config)}))
    run = base / "shared run"
    (run / "training/off").mkdir(parents=True)
    (run / "training/shared").mkdir()
    table = write_json(run / "training/shared/registration.json", {"entries": {"synthetic": "not refitted"}})
    phase = write_json(run / "training/shared/phase-authority.json", {"phase": "train"})
    manifest = dict(schema_version="direct_reference_job_v1", diagnostic_only=True, source_sha256=SOURCE_SHA,
                    request={"path": str(stage / "inputs/request.json"), "sha256": sha256(request)},
                    paths=dict(source_dir=str(stage / "code"), stage_dir=str(stage), run_dir=str(run),
                               raw_root=str(stage / "raw"), product_root=str(stage / "enrollment")),
                    authorities={"config": {"path": str(stage / "inputs/config.yaml"), "sha256": sha256(config)}})
    write_json(run / "input_manifest.json", manifest)
    control = write_json(base / "off authority.json", dict(schema_version="direct_reference_control_request_v1",
                        source_sha256=SOURCE_SHA, config_sha256=sha256(config),
                        diagnostic_request_sha256=sha256(request), arm="off", epochs=100, fold=0, seed=42))
    selected = write_json(run / "training/off/train.json", {"passed": True, "phase": "train", "arm": "off"})
    replay = write_json(run / "training/off/fresh-replay.json", dict(schema_version="direct_reference_control_replay_v1",
                        passed=True, fresh_process=True, arm="off", phase="train", train_result=selected, control_request=control))
    completed = write_json(run / "training/off/control.json", dict(schema_version="direct_reference_control_result_v1",
                         passed=True, arm="off", phase="train", epochs_completed=100,
                         control_request=control, train_result=selected, fresh_replay=replay))
    launcher = base / "external launcher.py"
    launcher.write_text('''import argparse, json, os
from pathlib import Path
parser = argparse.ArgumentParser(allow_abbrev=False)
for name in ('source-dir', 'request', 'request-sha256', 'on-request', 'on-request-sha256', 'run-dir', 'device'):
    parser.add_argument('--'+name, required=True)
args = parser.parse_args()
assert Path(args.source_dir).is_dir()
assert Path(args.request).is_file()
assert not (Path(args.run_dir) / 'training/on').exists()
with Path(os.environ['CALLS']).open('a') as stream:
    stream.write(json.dumps(dict(operation='on', arguments=vars(args), environment=dict(os.environ), cwd=os.getcwd()))+'\\n')
raise SystemExit(int(os.environ.get('ON_EXIT', '0')))
''')
    authority = base / "on authority.json"
    write_json(authority, dict(schema_version="direct_reference_on_request_v1", arm="on", epochs=100, fold=0, seed=42,
               source_sha256=SOURCE_SHA, config_sha256=sha256(config), diagnostic_request_sha256=sha256(request),
               launcher_sha256=sha256(launcher), off_completion=completed, registration=table, phase_authority=phase))
    inventory = base / "files.sha256"
    inventory.write_text("".join(f"{sha256(path)}  {path.relative_to(snapshot).as_posix()}\n"
                                 for path in sorted(snapshot.rglob("*")) if path.is_file()))
    fixture.update(stage=stage, snapshot=snapshot, archive=base / "inputs.tar", inventory=inventory,
                   launcher=launcher, launcher_hash=sha256(launcher), authority=authority, run=run)
    pack(fixture)
    interpreter = base / "interpreter-stub.py"
    interpreter.write_text(interpreter.read_text().replace("arguments = sys.argv[1:]", f"sys.executable = {ENV_PATH + '/bin/python'!r}\narguments = sys.argv[1:]"))
    interpreter.write_text(interpreter.read_text() + '''
if arguments[0] == '-' and os.environ.get('LAUNCH_BOUNDARY_DRIFT'):
    path = Path(os.environ['LAUNCH_BOUNDARY_DRIFT'])
    path.write_bytes(path.read_bytes() + b'\\n')
''')
    fixture["environment"].pop("ON_EXIT", None)
    fixture["environment"].pop("LAUNCH_BOUNDARY_DRIFT", None)
    return fixture


def run_on(fixture, arguments=None, **changes):
    assert WRAPPER.is_file(), "ON cluster wrapper is not implemented"
    isolated = fixture["tmp_path"] / "wrapper-under-test.sh"
    isolated.write_text(WRAPPER.read_text().replace(STAGE, str(fixture["stage"]))
                        .replace(LAUNCHER_SHA, fixture["launcher_hash"]))
    values = dict(fixture, **changes)
    arguments = arguments if arguments is not None else [str(values["archive"]), values.get("archive_hash", sha256(values["archive"])),
                 str(values["inventory"]), values.get("inventory_hash", sha256(values["inventory"])), str(values["launcher"]),
                 str(values["authority"]), values.get("authority_hash", sha256(values["authority"])), str(values["run"])]
    return subprocess.run(["bash", str(isolated), *arguments], env=values["environment"], capture_output=True, text=True, timeout=20)


def rejected_before_restore(fixture, result):
    assert result.returncode != 0
    assert not fixture["stage"].exists()
    assert calls(fixture) == []


def test_snapshot_restores_exact_bytes_without_shared_output_changes(on_fixture):
    protected = {path: path.read_bytes() for path in on_fixture["run"].rglob("*") if path.is_file()}
    result = run_on(on_fixture)
    assert result.returncode == 0, result.stderr
    for original in on_fixture["snapshot"].rglob("*"):
        restored = on_fixture["stage"] / original.relative_to(on_fixture["snapshot"])
        if original.is_file():
            assert restored.read_bytes() == original.read_bytes()
            assert restored.stat().st_mode & 0o222 == 0
    assert (on_fixture["stage"] / "tmp").is_dir()
    assert all(path.read_bytes() == content for path, content in protected.items())
    observed = calls(on_fixture)
    assert len(observed) == 1 and observed[0]["operation"] == "on"
    assert observed[0]["arguments"] == dict(source_dir=str(on_fixture["stage"] / "code"),
        request=str(on_fixture["stage"] / "inputs/request.json"),
        request_sha256=sha256(on_fixture["snapshot"] / "inputs/request.json"),
        on_request=str(on_fixture["authority"]), on_request_sha256=sha256(on_fixture["authority"]),
        run_dir=str(on_fixture["run"]), device="cuda")
    environment = observed[0]["environment"]
    for name, value in {"OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                        "NVIDIA_TF32_OVERRIDE": "0", "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE": "0", "PYTHONHASHSEED": "42",
                        "PYTHONDONTWRITEBYTECODE": "1", "CONDA_PREFIX": ENV_PATH,
                        "TMPDIR": str(on_fixture["stage"] / "tmp")}.items():
        assert environment[name] == value
    assert all(name not in environment for name in ("PYTHONPATH", "PYTHONHOME", "ROC_ANCHOR_GPU_INTEGRATION",
               "ROC_DIRECT_REFERENCE_GPU_INTEGRATION", "ROC_LEVIR_CKPT", "ROC_LEVIR_SHA256"))


@pytest.mark.parametrize("mutation", ["archive-hash", "inventory-hash", "authority-hash", "launcher-drift",
    "missing-off", "off-failed", "replay-failed", "existing-on", "table-drift", "phase-drift", "manifest-path"])
def test_untrusted_snapshot_or_incomplete_off_blocks_restore(on_fixture, mutation):
    changes = {}
    if mutation.endswith("-hash"):
        changes[mutation.replace("-", "_")] = "0" * 64
    elif mutation == "launcher-drift":
        on_fixture["launcher"].write_text("raise AssertionError('must not run')")
    elif mutation == "existing-on":
        (on_fixture["run"] / "training/on").mkdir()
    elif mutation == "missing-off":
        (on_fixture["run"] / "training/off/control.json").unlink()
    elif mutation in ("table-drift", "phase-drift"):
        name = "registration.json" if mutation == "table-drift" else "phase-authority.json"
        (on_fixture["run"] / "training/shared" / name).write_text("{}")
    elif mutation == "manifest-path":
        path = on_fixture["run"] / "input_manifest.json"
        value = json.loads(path.read_text())
        value["paths"]["stage_dir"] = "/some/other/stage"
        write_json(path, value)
    else:
        authority = json.loads(on_fixture["authority"].read_text())
        root = Path(authority["off_completion"]["path"])
        completed = json.loads(root.read_text())
        if mutation == "off-failed":
            completed["passed"] = False
        else:
            path = Path(completed["fresh_replay"]["path"])
            value = json.loads(path.read_text())
            value["passed"] = False
            completed["fresh_replay"] = write_json(path, value)
        authority["off_completion"] = write_json(root, completed)
        write_json(on_fixture["authority"], authority)
    rejected_before_restore(on_fixture, run_on(on_fixture, **changes))


@pytest.mark.parametrize("field,value", [("arm", "off"), ("epochs", True), ("epochs", 99), ("fold", 1),
    ("seed", 43), ("source_sha256", "0" * 64), ("schema_version", "other"), ("resume", True)])
def test_fixed_on_authority_only(on_fixture, field, value):
    authority = json.loads(on_fixture["authority"].read_text())
    authority[field] = value
    write_json(on_fixture["authority"], authority)
    rejected_before_restore(on_fixture, run_on(on_fixture))


@pytest.mark.parametrize("kind", ["traversal", "absolute", "unexpected", "symlink", "hardlink", "fifo", "duplicate"])
def test_unsafe_archive_members_are_rejected_before_any_extract(on_fixture, kind):
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
    pack(on_fixture, (member, content))
    rejected_before_restore(on_fixture, run_on(on_fixture))


@pytest.mark.parametrize("kind", ["missing", "extra", "wrong-bytes", "traversal", "duplicate"])
def test_file_manifest_must_match_archive_exactly(on_fixture, kind):
    path = on_fixture["inventory"]
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
    rejected_before_restore(on_fixture, run_on(on_fixture))


def test_existing_stage_is_never_overwritten(on_fixture):
    on_fixture["stage"].mkdir(parents=True)
    sentinel = on_fixture["stage"] / "original"
    sentinel.write_bytes(b"untouched")
    result = run_on(on_fixture)
    assert result.returncode != 0 and calls(on_fixture) == []
    assert sentinel.read_bytes() == b"untouched"


def test_symlink_stage_parent_is_rejected(on_fixture):
    actual = on_fixture["tmp_path"] / "other"
    actual.mkdir()
    on_fixture["stage"].parent.symlink_to(actual, target_is_directory=True)
    rejected_before_restore(on_fixture, run_on(on_fixture))


@pytest.mark.parametrize("name,value", [("SLURM_JOB_PARTITION", "cpu"), ("SLURMD_NODENAME", "login01"),
    ("SLURM_JOB_ID", "0"), ("SLURM_CPUS_PER_TASK", "3"), ("SLURM_GPUS_ON_NODE", "0"),
    ("CUDA_VISIBLE_DEVICES", "-1"), ("MOCK_PREFIX", "/wrong"), ("MOCK_INTERPRETER_PREFIX", "/wrong")])
def test_wrong_allocation_or_environment_blocks_before_restore(on_fixture, name, value):
    on_fixture["environment"][name] = value
    rejected_before_restore(on_fixture, run_on(on_fixture))


def test_launcher_failure_propagates_without_completion_write(on_fixture):
    on_fixture["environment"]["ON_EXIT"] = "17"
    result = run_on(on_fixture)
    assert result.returncode == 17
    assert not (on_fixture["run"] / "training/on").exists()


@pytest.mark.parametrize("target", ["launcher", "authority"])
def test_launch_authorities_reauthenticated_after_restore(on_fixture, target):
    on_fixture["environment"]["LAUNCH_BOUNDARY_DRIFT"] = str(on_fixture[target])
    result = run_on(on_fixture)
    assert result.returncode != 0
    assert calls(on_fixture) == []
    assert on_fixture["stage"].is_dir()


def test_fixed_slurm_header_and_no_optional_recipe_arguments(on_fixture):
    assert WRAPPER.is_file(), "ON cluster wrapper is not implemented"
    options = dict(line.removeprefix("#SBATCH --").split("=", 1) for line in WRAPPER.read_text().splitlines()
                   if line.startswith("#SBATCH --") and "=" in line)
    assert {name: options[name] for name in ("account", "partition", "nodes", "ntasks", "gres", "cpus-per-task", "mem")} == dict(
        account="acc-mialhajri", partition="gpu", nodes="1", ntasks="1", gres="gpu:1", **{"cpus-per-task": "4", "mem": "8G"})
    rejected_before_restore(on_fixture, run_on(on_fixture, arguments=["--epochs", "1"]))
