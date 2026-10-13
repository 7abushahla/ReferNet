import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "scripts/cluster/protocol_a_study.sh"
ENV_PATH = "/shared/b00090279/conda/envs/refer-or-clear"


@pytest.fixture
def job(tmp_path):
    source = tmp_path / "frozen source with spaces"
    for name in ("scripts", "src"):
        (source / name).mkdir(parents=True)
    (source / "scripts/protocol_a_study.py").write_text("raise RuntimeError('scientific code must not run in wrapper tests')\n")
    (source / "src/frozen.py").write_text("version = 1\n")
    for path in source.rglob("*"):
        path.chmod(0o555 if path.is_dir() else 0o444)
    source.chmod(0o555)
    request = tmp_path / "request with spaces.json"
    request.write_text("{}")
    calls = tmp_path / "dispatch.json"
    driver = tmp_path / "python driver.py"
    driver.write_text('''import json, os, sys
from pathlib import Path
arguments = sys.argv[1:]
if arguments[0] == '-c':
    sys.prefix = os.environ.get('MOCK_INTERPRETER_PREFIX', '/shared/b00090279/conda/envs/refer-or-clear')
    sys.executable = os.environ.get('MOCK_EXECUTABLE', '/shared/b00090279/conda/envs/refer-or-clear/bin/python')
    sys.argv = ['-c', *arguments[2:]]
    exec(arguments[1], {'__name__': '__main__'})
else:
    call = dict(arguments=arguments, cwd=str(Path.cwd()), environment={name: os.environ.get(name) for name in
        ('CONDA_PREFIX', 'PYTHONPATH', 'PYTHONHOME', 'PYTHONHASHSEED', 'CUBLAS_WORKSPACE_CONFIG',
         'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NVIDIA_TF32_OVERRIDE', 'TORCH_ALLOW_TF32_CUBLAS_OVERRIDE',
         'PYTHONDONTWRITEBYTECODE', 'PYTHONUNBUFFERED', 'ROC_LEVIR_CKPT', 'ROC_LEVIR_SHA256',
         'ROC_ANCHOR_GPU_INTEGRATION', 'ROC_DIRECT_REFERENCE_GPU_INTEGRATION')})
    Path(os.environ['CALL_PATH']).write_text(json.dumps(call))
    if arguments[1] == 'smoke':
        if '--gate-sha256' in arguments:
            raise SystemExit('smoke must not receive a gate hash')
    else:
        if '--gate-sha256' not in arguments:
            raise SystemExit('external gate hash missing')
        gate_sha = arguments[arguments.index('--gate-sha256') + 1]
        if gate_sha != 'b' * 64:
            raise SystemExit('external gate hash mismatch')
    raise SystemExit(int(os.environ.get('BUILDER_EXIT', '0')))
''')
    bootstrap = tmp_path / "bash_env"
    bootstrap.write_text(f'''source() {{
    [[ "$1" == /opt/miniconda/etc/profile.d/conda.sh ]] || return 99
}}
conda() {{
    [[ "$1" == activate && "$2" == {ENV_PATH} ]] || return 98
    export CONDA_PREFIX="${{MOCK_PREFIX:-$2}}"
}}
function {ENV_PATH}/bin/python {{
    command {shlex.quote(sys.executable)} {shlex.quote(str(driver))} "$@"
}}
''')
    environment = dict(os.environ, BASH_ENV=str(bootstrap), CALL_PATH=str(calls), SLURM_JOB_ID="1234",
                       SLURMD_NODENAME="gpu-dy-g5-0-12", SLURM_SUBMIT_DIR=str(tmp_path),
                       SLURM_JOB_PARTITION="gpu", SLURM_CPUS_PER_TASK="4", SLURM_GPUS_ON_NODE="1",
                       CUDA_VISIBLE_DEVICES="0", PYTHONPATH="/foreign/package", PYTHONHOME="/foreign/python",
                       ROC_LEVIR_CKPT="/foreign/weights", ROC_LEVIR_SHA256="foreign",
                       ROC_ANCHOR_GPU_INTEGRATION="1", ROC_DIRECT_REFERENCE_GPU_INTEGRATION="1")
    for name in ("SLURM_GPUS", "SLURM_JOB_GPUS", "SLURM_STEP_GPUS", "MOCK_PREFIX", "MOCK_EXECUTABLE",
                 "MOCK_INTERPRETER_PREFIX", "BUILDER_EXIT"):
        environment.pop(name, None)
    return dict(source=source, request=request, calls=calls, environment=environment,
                arguments=[source, request, "a" * 64, "train", "off", "b" * 64])


def run(job, arguments=None):
    assert WRAPPER.is_file(), "Protocol A GPU study wrapper is missing"
    return subprocess.run(["bash", str(WRAPPER), *map(str, job["arguments"] if arguments is None else arguments)],
                          env=job["environment"], text=True, capture_output=True, timeout=30)


def test_header_matches_approved_gpu_resources_and_no_submission():
    assert WRAPPER.is_file(), "Protocol A GPU study wrapper is missing"
    content = WRAPPER.read_text()
    options = dict(line.removeprefix("#SBATCH --").split("=", 1) for line in content.splitlines()
                   if line.startswith("#SBATCH --") and "=" in line)
    assert {key: options[key] for key in ("account", "partition", "nodes", "ntasks", "gres", "cpus-per-task", "mem", "time", "exclude")} == {
        "account": "acc-mialhajri", "partition": "gpu", "nodes": "1", "ntasks": "1", "gres": "gpu:1",
        "cpus-per-task": "4", "mem": "14G", "time": "12:00:00", "exclude": "gpu-dy-g5-0-16"}
    assert "source /opt/miniconda/etc/profile.d/conda.sh" in content
    assert "sbatch " not in content and "ssh " not in content and "git " not in content
    assert subprocess.run(["bash", "-n", str(WRAPPER)], capture_output=True).returncode == 0


@pytest.mark.parametrize("command", ["smoke", "train", "replay", "evaluate"])
@pytest.mark.parametrize("arm", ["off", "on"])
def test_dispatches_pinned_command_and_arm_without_staging_or_recipe_overrides(job, command, arm):
    gate_sha = "-" if command == "smoke" else "b" * 64
    arguments = [*job["arguments"][:3], command, arm, gate_sha]
    result = run(job, arguments)
    assert result.returncode == 0, result.stderr
    call = json.loads(job["calls"].read_text())
    expected_arguments = [str(job["source"] / "scripts/protocol_a_study.py"), command,
                          "--request", str(job["request"]), "--request-sha256", "a" * 64, "--arm", arm]
    if command != "smoke":
        expected_arguments += ["--gate-sha256", gate_sha]
    assert call["arguments"] == expected_arguments
    assert call["environment"] == dict(CONDA_PREFIX=ENV_PATH,
        PYTHONPATH=f"{job['source']}/src:{job['source']}/scripts", PYTHONHOME=None, PYTHONHASHSEED="42",
        CUBLAS_WORKSPACE_CONFIG=":4096:8", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", NVIDIA_TF32_OVERRIDE="0",
        TORCH_ALLOW_TF32_CUBLAS_OVERRIDE="0", PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1",
        ROC_LEVIR_CKPT=None, ROC_LEVIR_SHA256=None, ROC_ANCHOR_GPU_INTEGRATION=None,
        ROC_DIRECT_REFERENCE_GPU_INTEGRATION=None)
    assert not (job["source"].parent / "scratch").exists()
    assert (job["source"] / "scripts/protocol_a_study.py").stat().st_mode & 0o222 == 0


@pytest.mark.parametrize("name,value", [("SLURM_JOB_ID", ""), ("SLURM_JOB_ID", "0"), ("SLURM_JOB_ID", "12/x"),
    ("SLURMD_NODENAME", ""), ("SLURMD_NODENAME", "LOGIN01"), ("SLURMD_NODENAME", "head01"),
    ("SLURMD_NODENAME", "frontend01"), ("SLURMD_NODENAME", "lgn01"), ("SLURMD_NODENAME", "gpu-dy-g5-0-16"),
    ("SLURM_SUBMIT_DIR", ""), ("SLURM_JOB_PARTITION", "cpu"), ("SLURM_CPUS_PER_TASK", "3"),
    ("SLURM_CPUS_PER_TASK", "four"), ("SLURM_GPUS_ON_NODE", "0"), ("CUDA_VISIBLE_DEVICES", ""),
    ("CUDA_VISIBLE_DEVICES", "-1"), ("CUDA_VISIBLE_DEVICES", "NoDevFiles"), ("MOCK_PREFIX", "/wrong"),
    ("MOCK_INTERPRETER_PREFIX", "/wrong"), ("MOCK_EXECUTABLE", "/wrong/bin/python")])
def test_allocation_and_exact_environment_guards_precede_scientific_dispatch(job, name, value):
    job["environment"][name] = value
    result = run(job)
    assert result.returncode != 0 and not job["calls"].exists()


@pytest.mark.parametrize("count", [0, 5, 7])
def test_exact_six_arguments_required(job, count):
    arguments = [] if count == 0 else job["arguments"][:count] if count == 5 else [*job["arguments"], "extra"]
    result = run(job, arguments)
    assert result.returncode != 0 and not job["calls"].exists()


@pytest.mark.parametrize("index,value", [(2, "bad-sha"), (3, "test"), (3, "evaluation"), (4, "median"), (4, "")])
def test_invalid_interface_values_fail_before_dispatch(job, index, value):
    job["arguments"][index] = value
    result = run(job)
    assert result.returncode != 0 and not job["calls"].exists()


@pytest.mark.parametrize("command", ["train", "replay", "evaluate"])
def test_missing_external_gate_hash_fails_before_dispatch(job, command):
    job["arguments"][3] = command
    result = run(job, job["arguments"][:5])
    assert result.returncode != 0 and not job["calls"].exists()


@pytest.mark.parametrize("command", ["train", "replay", "evaluate"])
@pytest.mark.parametrize("gate_sha", ["-", "", "B" * 64, "b" * 63, "g" * 64])
def test_invalid_external_gate_hash_fails_before_dispatch(job, command, gate_sha):
    job["arguments"][3] = command
    job["arguments"][5] = gate_sha
    result = run(job)
    assert result.returncode != 0 and not job["calls"].exists()
    assert "gate" in result.stderr.lower()


@pytest.mark.parametrize("command", ["train", "replay", "evaluate"])
def test_wrong_expected_gate_hash_is_forwarded_and_cli_rejection_propagates(job, command):
    job["arguments"][3] = command
    job["arguments"][5] = "c" * 64
    result = run(job)
    assert result.returncode != 0
    assert "external gate hash mismatch" in result.stderr
    call = json.loads(job["calls"].read_text())
    assert call["arguments"][-2:] == ["--gate-sha256", "c" * 64]


@pytest.mark.parametrize("target", ["root", "script", "module"])
def test_frozen_source_must_be_readonly(job, target):
    path = {"root": job["source"], "script": job["source"] / "scripts/protocol_a_study.py",
            "module": job["source"] / "src/frozen.py"}[target]
    path.chmod(0o755 if path.is_dir() else 0o644)
    result = run(job)
    assert result.returncode != 0 and not job["calls"].exists()


def test_dispatch_failure_propagates(job):
    job["environment"]["BUILDER_EXIT"] = "37"
    result = run(job)
    assert result.returncode == 37
