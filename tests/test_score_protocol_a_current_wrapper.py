import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "scripts/cluster/score_protocol_a_current.sh"
ENV_PATH = "/shared/b00090279/conda/envs/refer-or-clear"
STAGE = "/scratch/b00090279/direct-reference-control-38041/inputs"
NAMES = {"enrollment.json", "weights.pth", "split.csv", "product-root.json", "request.json", "config.yaml", "records.csv"}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def job(tmp_path):
    source = tmp_path / "frozen EEC overlay"
    (source / "scripts").mkdir(parents=True)
    (source / "src").mkdir()
    (source / "src/never-import.py").write_text("raise RuntimeError('scientific code must not run')\n")
    (source / "scripts/score_protocol_a_current.py").write_text('''import argparse, hashlib, json, os
from pathlib import Path
parser = argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument('--request', required=True)
parser.add_argument('--request-sha256', required=True)
args = parser.parse_args()
assert hashlib.sha256(Path(args.request).read_bytes()).hexdigest() == args.request_sha256
call = dict(arguments=vars(args), script=str(Path(__file__).resolve()),
            environment={name: os.environ.get(name) for name in
                         ('CONDA_PREFIX', 'PYTHONPATH', 'PYTHONHOME', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                          'NVIDIA_TF32_OVERRIDE', 'TORCH_ALLOW_TF32_CUBLAS_OVERRIDE', 'CUBLAS_WORKSPACE_CONFIG', 'PYTHONHASHSEED')})
Path(os.environ['CALL_PATH']).write_text(json.dumps(call))
raise SystemExit(int(os.environ.get('SCORER_EXIT', '0')))
''')
    for path in source.rglob("*"):
        path.chmod(0o555 if path.is_dir() else 0o444)
    source.chmod(0o555)
    inputs = tmp_path / "durable original inputs"
    inputs.mkdir()
    for name in NAMES:
        (inputs / name).write_bytes(f"synthetic {name}; no weights or pixels loaded".encode())
    inventory = tmp_path / "inputs.sha256"
    inventory.write_text("".join(f"{sha(inputs / name)}  {name}\n" for name in sorted(NAMES)))
    request = tmp_path / "score request.json"
    request.write_text('{"schema_version":"synthetic_no_scientific_execution"}')
    calls = tmp_path / "call.json"
    driver = tmp_path / "python driver.py"
    driver.write_text('''import os, runpy, sys
sys.prefix = os.environ.get('MOCK_PREFIX', '/shared/b00090279/conda/envs/refer-or-clear')
sys.executable = os.environ.get('MOCK_EXECUTABLE', sys.prefix + '/bin/python')
arguments = sys.argv[1:]
if arguments[0] in ('-', '-c'):
    code = sys.stdin.read() if arguments[0] == '-' else arguments[1]
    sys.argv = [arguments[0], *arguments[1 if arguments[0] == '-' else 2:]]
    exec(code, {'__name__': '__main__'})
else:
    sys.argv = arguments
    runpy.run_path(arguments[0], run_name='__main__')
''')
    bootstrap = tmp_path / "bash_env"
    bootstrap.write_text(f'''source() {{ [[ "$1" == /opt/miniconda/etc/profile.d/conda.sh ]]; }}
conda() {{
    [[ "$1" == activate && "$2" == {ENV_PATH} ]] || return 98
    export CONDA_PREFIX="${{MOCK_CONDA_PREFIX:-$2}}"
}}
function {ENV_PATH}/bin/python {{ command {shlex.quote(sys.executable)} {shlex.quote(str(driver))} "$@"; }}
''')
    environment = dict(os.environ, BASH_ENV=str(bootstrap), CALL_PATH=str(calls), SLURM_JOB_ID="1234",
                       SLURMD_NODENAME="gpu-dy-g5-0-12", SLURM_JOB_PARTITION="gpu", SLURM_CPUS_PER_TASK="4",
                       SLURM_GPUS_ON_NODE="1", CUDA_VISIBLE_DEVICES="0", SLURM_SUBMIT_DIR=str(tmp_path),
                       PYTHONPATH="/foreign", PYTHONHOME="/foreign")
    for name in ("SLURM_GPUS", "SLURM_JOB_GPUS", "SLURM_STEP_GPUS", "MOCK_PREFIX", "MOCK_EXECUTABLE", "MOCK_CONDA_PREFIX", "SCORER_EXIT"):
        environment.pop(name, None)
    return dict(source=source, inputs=inputs, inventory=inventory, request=request, calls=calls, environment=environment,
                stage=tmp_path / "restored scratch/inputs", arguments=[source, request, sha(request), inputs, inventory, sha(inventory)])


def run(job, arguments=None):
    assert WRAPPER.is_file(), "A current score wrapper missing"
    isolated = job["source"].parent / "wrapper-under-test.sh"
    isolated.write_text(WRAPPER.read_text().replace(STAGE, str(job["stage"])))
    return subprocess.run(["bash", str(isolated), *map(str, job["arguments"] if arguments is None else arguments)],
                          env=job["environment"], capture_output=True, text=True, timeout=30)


def test_header_is_bounded_gpu_scoring_only():
    assert WRAPPER.is_file(), "A current score wrapper missing"
    content = WRAPPER.read_text()
    options = dict(line.removeprefix("#SBATCH --").split("=", 1) for line in content.splitlines()
                   if line.startswith("#SBATCH --") and "=" in line)
    assert {key: options[key] for key in ("account", "partition", "nodes", "ntasks", "gres", "cpus-per-task", "mem", "time", "exclude")} == {
        "account": "acc-mialhajri", "partition": "gpu", "nodes": "1", "ntasks": "1", "gres": "gpu:1",
        "cpus-per-task": "4", "mem": "12G", "time": "1:00:00", "exclude": "gpu-dy-g5-0-16,gpu-dy-g5-0-3"}
    assert "source /opt/miniconda/etc/profile.d/conda.sh" in content
    assert not any(token in content for token in ("sbatch ", "ssh ", "git ", "tarfile", "train_direct_reference"))
    assert subprocess.run(["bash", "-n", str(WRAPPER)], capture_output=True).returncode == 0


def test_only_exact_authenticated_inputs_restored_readonly_and_scoring_dispatched(job):
    result = run(job)
    assert result.returncode == 0, result.stderr
    assert {path.name for path in job["stage"].iterdir()} == NAMES
    for name in NAMES:
        restored = job["stage"] / name
        assert restored.read_bytes() == (job["inputs"] / name).read_bytes()
        assert restored.stat().st_mode & 0o222 == 0
    assert {path.name for path in job["stage"].parent.iterdir()} == {"inputs"}
    call = json.loads(job["calls"].read_text())
    assert call["arguments"] == dict(request=str(job["request"]), request_sha256=sha(job["request"]))
    assert call["script"] == str(job["source"] / "scripts/score_protocol_a_current.py")
    assert call["environment"] == dict(CONDA_PREFIX=ENV_PATH, PYTHONPATH=f"{job['source']}/src:{job['source']}/scripts", PYTHONHOME=None,
        OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", NVIDIA_TF32_OVERRIDE="0", TORCH_ALLOW_TF32_CUBLAS_OVERRIDE="0",
        CUBLAS_WORKSPACE_CONFIG=":4096:8", PYTHONHASHSEED="42")


def test_matching_existing_inputs_reused_without_writes(job):
    first = run(job)
    assert first.returncode == 0, first.stderr
    before = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in job["stage"].iterdir()}
    second = run(job)
    assert second.returncode == 0, second.stderr
    assert all((path.read_bytes(), path.stat().st_mtime_ns) == value for path, value in before.items())


def test_relative_request_keeps_submit_directory_resolution(job):
    job["arguments"][1] = job["request"].name
    result = run(job)
    assert result.returncode == 0, result.stderr


def test_mismatching_existing_inputs_fail_without_overwrite_or_partial_restore(job):
    job["stage"].mkdir(parents=True)
    sentinel = job["stage"] / "weights.pth"
    sentinel.write_bytes(b"existing mismatch")
    result = run(job)
    assert result.returncode != 0 and not job["calls"].exists()
    assert sentinel.read_bytes() == b"existing mismatch"
    assert {path.name for path in job["stage"].iterdir()} == {"weights.pth"}


@pytest.mark.parametrize("mutation", ["request-hash", "inventory-hash", "content", "missing", "extra", "traversal", "absolute", "duplicate", "symlink", "hardlink", "writable-source"])
def test_untrusted_inputs_rejected_before_restore_or_scoring(job, mutation):
    if mutation in ("request-hash", "inventory-hash"):
        job["arguments"][2 if mutation == "request-hash" else 5] = "a" * 64
    elif mutation == "content":
        (job["inputs"] / "config.yaml").write_text("mutated")
    elif mutation == "missing":
        (job["inputs"] / "records.csv").unlink()
    elif mutation == "extra":
        (job["inputs"] / "extra.txt").write_text("unapproved")
    elif mutation in ("traversal", "absolute", "duplicate"):
        text = job["inventory"].read_text()
        if mutation == "duplicate":
            text += text.splitlines()[0] + "\n"
        else:
            text = text.replace("  config.yaml", "  ../config.yaml" if mutation == "traversal" else "  /config.yaml")
        job["inventory"].write_text(text)
        job["arguments"][5] = sha(job["inventory"])
    elif mutation == "writable-source":
        job["source"].chmod(0o755)
    else:
        original = job["inputs"] / "weights.pth"
        elsewhere = job["inputs"].parent / "elsewhere"
        original.rename(elsewhere)
        if mutation == "symlink":
            original.symlink_to(elsewhere)
        else:
            os.link(elsewhere, original)
    result = run(job)
    assert result.returncode != 0 and not job["calls"].exists() and not job["stage"].exists()


@pytest.mark.parametrize("name,value", [("SLURM_JOB_ID", ""), ("SLURM_JOB_ID", "0"), ("SLURMD_NODENAME", "LOGIN01"),
    ("SLURMD_NODENAME", "head01"), ("SLURMD_NODENAME", "frontend01"), ("SLURMD_NODENAME", "lgn01"),
    ("SLURMD_NODENAME", "gpu-dy-g5-0-16"), ("SLURMD_NODENAME", "gpu-dy-g5-0-3"), ("SLURM_JOB_PARTITION", "cpu"),
    ("SLURM_CPUS_PER_TASK", "3"), ("SLURM_GPUS_ON_NODE", "0"), ("CUDA_VISIBLE_DEVICES", ""),
    ("CUDA_VISIBLE_DEVICES", "-1"), ("CUDA_VISIBLE_DEVICES", "NoDevFiles"), ("SLURM_SUBMIT_DIR", ""),
    ("MOCK_CONDA_PREFIX", "/wrong"), ("MOCK_PREFIX", "/wrong"), ("MOCK_EXECUTABLE", "/wrong/bin/python")])
def test_allocation_and_environment_errors_block_restore(job, name, value):
    job["environment"][name] = value
    result = run(job)
    assert result.returncode != 0 and not job["calls"].exists() and not job["stage"].exists()


@pytest.mark.parametrize("count", [0, 5, 7])
def test_exact_six_arguments(job, count):
    arguments = job["arguments"][:count] if count < 6 else [*job["arguments"], "extra"]
    assert run(job, arguments).returncode != 0 and not job["stage"].exists()


def test_scoring_failure_propagates(job):
    job["environment"]["SCORER_EXIT"] = "37"
    assert run(job).returncode == 37
