import json
import subprocess
import tarfile

import pytest

from test_direct_reference_cluster import ENV_PATH, ROOT, calls, cluster_fixture, sha256
from test_direct_reference_on_cluster import on_fixture, pack, write_json


WRAPPER = ROOT / "scripts/cluster/train_direct_reference_warmstart.sh"
STAGE = "/scratch/b00090279/direct-reference-control-38041"
OUTPUT = "/shared/b00090279/refer-or-clear/results/exploratory"
HELPER_SHA = "f2697cbf85067baeba354077dec2bc0427e39fcf71b0005396f0f265a054c012"
REQUEST_SHA = "2f6d479835003242790f6d10f44249d73aa9a43f6a465132e618bc99c24c87b9"
MANIFEST_SHA = "dbe33a68c945cb5f6b5c4140fb6000e293342150537bef4d09ff2304d46d613a"


@pytest.fixture
def warmstart_fixture(on_fixture):
    fixture = on_fixture
    base = fixture["tmp_path"]
    fixture["helper"] = fixture["launcher"]
    fixture["helper_hash"] = sha256(fixture["helper"])
    fixture["original_request_hash"] = sha256(fixture["snapshot"] / "inputs/request.json")
    fixture["original_manifest_hash"] = sha256(fixture["run"] / "input_manifest.json")
    launcher = base / "exploratory launcher.py"
    launcher.write_text('''import argparse, hashlib, json, os
from pathlib import Path
parser = argparse.ArgumentParser(allow_abbrev=False)
for name in ('authority', 'authority-sha256', 'source-dir', 'on-helper', 'output-root'):
    parser.add_argument('--'+name, required=True)
args = parser.parse_args()
assert hashlib.sha256(Path(args.authority).read_bytes()).hexdigest() == args.authority_sha256
assert Path(args.source_dir).is_dir() and Path(args.on_helper).is_file()
assert not Path(args.output_root).exists()
with Path(os.environ['CALLS']).open('a') as stream:
    stream.write(json.dumps(dict(operation='warmstart', arguments=vars(args), environment=dict(os.environ)))+'\\n')
raise SystemExit(int(os.environ.get('WARMSTART_EXIT', '0')))
''')
    fixture["launcher"] = launcher
    fixture["authority"] = base / "warmstart request.json"
    write_json(fixture["authority"], dict(original_run_dir=str(fixture["run"]), launcher_sha256=sha256(launcher)))
    fixture["output_base"] = base / "new exploratory outputs"
    fixture["output"] = fixture["output_base"] / "direct-reference-warmstart-1234"
    fixture["environment"]["SLURMD_NODENAME"] = "gpu-dy-g5-0-17"
    for name in ("WARMSTART_EXIT", "ON_EXIT", "LAUNCH_BOUNDARY_DRIFT"):
        fixture["environment"].pop(name, None)
    for name in ("control.json", "fresh-replay.json", "train.json"):
        (fixture["run"] / "training/off" / name).unlink()
    return fixture


def run_warmstart(fixture, arguments=None, **changes):
    assert WRAPPER.is_file(), "warmstart cluster wrapper missing"
    script = fixture["tmp_path"] / "warmstart-wrapper-under-test.sh"
    script.write_text(WRAPPER.read_text().replace(STAGE, str(fixture["stage"]))
                      .replace(OUTPUT, str(fixture["output_base"]))
                      .replace(HELPER_SHA, fixture["helper_hash"])
                      .replace(MANIFEST_SHA, fixture["original_manifest_hash"])
                      .replace(REQUEST_SHA, fixture["original_request_hash"]))
    values = dict(fixture, **changes)
    arguments = arguments if arguments is not None else [str(values["archive"]), values.get("archive_hash") or sha256(values["archive"]),
        str(values["inventory"]), values.get("inventory_hash") or sha256(values["inventory"]), str(values["launcher"]),
        str(values["helper"]), str(values["authority"]), values.get("authority_hash", sha256(values["authority"]))]
    return subprocess.run(["bash", str(script), *arguments], env=values["environment"], capture_output=True, text=True, timeout=20)


def rejected_before_restore(fixture, result):
    assert result.returncode != 0
    assert not fixture["stage"].exists()
    assert calls(fixture) == []


def test_restores_exact_original_paths_without_completion_gate_or_shared_writes(warmstart_fixture):
    fixture = warmstart_fixture
    before = {path: path.read_bytes() for path in fixture["run"].rglob("*") if path.is_file()}
    result = run_warmstart(fixture)
    assert result.returncode == 0, result.stderr
    for original in fixture["snapshot"].rglob("*"):
        if original.is_file():
            restored = fixture["stage"] / original.relative_to(fixture["snapshot"])
            assert restored.read_bytes() == original.read_bytes()
            assert restored.stat().st_mode & 0o222 == 0
    assert (fixture["stage"] / "tmp").is_dir()
    assert all(path.read_bytes() == content for path, content in before.items())
    observed = calls(fixture)
    assert len(observed) == 1 and observed[0]["operation"] == "warmstart"
    assert observed[0]["arguments"] == dict(authority=str(fixture["authority"]), authority_sha256=sha256(fixture["authority"]),
        source_dir=str(fixture["stage"] / "code"), on_helper=str(fixture["helper"]), output_root=str(fixture["output"]))
    environment = observed[0]["environment"]
    for name, value in dict(OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", CUBLAS_WORKSPACE_CONFIG=":4096:8",
        NVIDIA_TF32_OVERRIDE="0", TORCH_ALLOW_TF32_CUBLAS_OVERRIDE="0", PYTHONHASHSEED="42", PYTHONDONTWRITEBYTECODE="1",
        CONDA_PREFIX=ENV_PATH, TMPDIR=str(fixture["stage"] / "tmp")).items():
        assert environment[name] == value
    assert all(name not in environment for name in ("PYTHONPATH", "PYTHONHOME", "ROC_ANCHOR_GPU_INTEGRATION", "ROC_DIRECT_REFERENCE_GPU_INTEGRATION"))


@pytest.mark.parametrize("name,value", [("SLURM_JOB_PARTITION", "cpu"), ("SLURMD_NODENAME", "login01"),
    ("SLURMD_NODENAME", "gpu-dy-g5-0-16"), ("SLURM_JOB_ID", "0"),
    ("SLURM_CPUS_PER_TASK", "3"), ("SLURM_GPUS_ON_NODE", "0"), ("CUDA_VISIBLE_DEVICES", "-1"),
    ("MOCK_PREFIX", "/wrong"), ("MOCK_INTERPRETER_PREFIX", "/wrong")])
def test_allocation_and_environment_fail_before_restore(warmstart_fixture, name, value):
    warmstart_fixture["environment"][name] = value
    rejected_before_restore(warmstart_fixture, run_warmstart(warmstart_fixture))


def test_other_allocated_gpu_node_accepts_exact_path_restore(warmstart_fixture):
    warmstart_fixture["environment"]["SLURMD_NODENAME"] = "gpu-dy-g5-0-18"
    result = run_warmstart(warmstart_fixture)
    assert result.returncode == 0, result.stderr
    assert calls(warmstart_fixture)[0]["operation"] == "warmstart"


@pytest.mark.parametrize("mutation", ["archive-hash", "inventory-hash", "authority-hash", "launcher", "helper", "manifest-path", "request"])
def test_hash_and_path_guards_precede_restore(warmstart_fixture, mutation):
    fixture = warmstart_fixture
    changes = {}
    if mutation.endswith("-hash"):
        changes[mutation.replace("-", "_")] = "0" * 64
    elif mutation in ("launcher", "helper"):
        fixture[mutation].write_text("raise AssertionError('unverified execution')")
    elif mutation == "manifest-path":
        target = fixture["run"] / "input_manifest.json"
        value = json.loads(target.read_text())
        value["paths"]["stage_dir"] = "/wrong/stage"
        write_json(target, value)
    else:
        (fixture["snapshot"] / "inputs/request.json").write_text("{}")
        pack(fixture)
    rejected_before_restore(fixture, run_warmstart(fixture, **changes))


@pytest.mark.parametrize("kind", ["traversal", "absolute", "unexpected", "symlink", "hardlink", "fifo", "duplicate"])
def test_tar_members_rejected_before_extraction(warmstart_fixture, kind):
    fixture = warmstart_fixture
    member = tarfile.TarInfo({"traversal": "raw/../escape", "absolute": "/tmp/escape", "unexpected": "training/off/best.pt"}.get(kind, "raw/unsafe"))
    content = b"not allowed"
    member.size = len(content)
    if kind in ("symlink", "hardlink", "fifo"):
        member.type = {"symlink": tarfile.SYMTYPE, "hardlink": tarfile.LNKTYPE, "fifo": tarfile.FIFOTYPE}[kind]
        member.linkname, member.size = "raw/train.bin", 0
    if kind == "duplicate": member.name = "raw/train.bin"
    pack(fixture, (member, content))
    rejected_before_restore(fixture, run_warmstart(fixture))


@pytest.mark.parametrize("kind", ["missing", "extra", "wrong-bytes", "traversal", "duplicate"])
def test_manifest_matches_archive_exactly(warmstart_fixture, kind):
    fixture = warmstart_fixture
    lines = fixture["inventory"].read_text().splitlines()
    if kind == "missing": lines = lines[1:]
    elif kind == "extra": lines.append("0" * 64 + "  raw/extra.bin")
    elif kind == "wrong-bytes": lines[0] = "0" * 64 + lines[0][64:]
    elif kind == "traversal": lines.append("0" * 64 + "  raw/../escape")
    else: lines.append(lines[0])
    fixture["inventory"].write_text("\n".join(lines) + "\n")
    rejected_before_restore(fixture, run_warmstart(fixture))


@pytest.mark.parametrize("target", ["archive", "inventory"])
def test_nonregular_snapshot_inputs_fail_without_blocking(warmstart_fixture, target):
    fixture = warmstart_fixture
    checksum = sha256(fixture[target])
    fixture[target].unlink()
    fixture[target].mkdir()
    rejected_before_restore(fixture, run_warmstart(fixture, **{target + "_hash": checksum}))


def test_existing_stage_and_output_never_overwritten(warmstart_fixture):
    fixture = warmstart_fixture
    fixture["stage"].mkdir(parents=True)
    sentinel = fixture["stage"] / "untouched"
    sentinel.write_bytes(b"original")
    assert run_warmstart(fixture).returncode != 0 and calls(fixture) == []
    assert sentinel.read_bytes() == b"original"
    fixture["stage"].rename(fixture["stage"].with_name("preserved"))
    fixture["output"].mkdir(parents=True)
    rejected_before_restore(fixture, run_warmstart(fixture))


@pytest.mark.parametrize("target", ["launcher", "helper", "authority"])
def test_launch_boundary_reauthenticates_external_inputs(warmstart_fixture, target):
    fixture = warmstart_fixture
    fixture["environment"]["LAUNCH_BOUNDARY_DRIFT"] = str(fixture[target])
    result = run_warmstart(fixture)
    assert result.returncode != 0 and calls(fixture) == []
    assert fixture["stage"].is_dir()


def test_external_failure_propagates(warmstart_fixture):
    warmstart_fixture["environment"]["WARMSTART_EXIT"] = "17"
    assert run_warmstart(warmstart_fixture).returncode == 17
    assert not (warmstart_fixture["run"] / "training/on").exists()


def test_fixed_resources_eight_arguments_and_shell_syntax(warmstart_fixture):
    assert WRAPPER.is_file(), "warmstart cluster wrapper missing"
    options = dict(line.removeprefix("#SBATCH --").split("=", 1) for line in WRAPPER.read_text().splitlines() if line.startswith("#SBATCH --"))
    assert {name: options[name] for name in ("account", "partition", "nodes", "ntasks", "gres", "cpus-per-task", "mem", "time", "exclude")} == dict(
        account="acc-mialhajri", partition="gpu", nodes="1", ntasks="1", gres="gpu:1", **{"cpus-per-task": "4", "mem": "8G", "time": "3:00:00", "exclude": "gpu-dy-g5-0-16"})
    assert "nodelist" not in options
    assert subprocess.run(["bash", "-n", str(WRAPPER)], capture_output=True).returncode == 0
    rejected_before_restore(warmstart_fixture, run_warmstart(warmstart_fixture, arguments=["--epochs", "1"]))
