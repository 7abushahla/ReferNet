import json
import subprocess

import pytest

from test_protocol_a_study_wrapper import ENV_PATH, ROOT, job


WRAPPER = ROOT / "scripts/cluster/probe_direct_reference_regularization.sh"


@pytest.fixture
def probe_job(job):
    source = job["source"]
    source.chmod(0o755)
    (source / "scripts").chmod(0o755)
    runner = source / "scripts/probe_direct_reference_regularization.py"
    runner.write_text("raise RuntimeError('scientific code must not run in wrapper tests')\n")
    runner.chmod(0o444)
    (source / "scripts").chmod(0o555)
    source.chmod(0o555)
    driver = source.parent / "python driver.py"
    driver.write_text(driver.read_text().replace("if arguments[1] == 'smoke':", "if arguments[1] != 'train':"))
    job["arguments"] = [source, job["request"], "a" * 64, "train", "b" * 64]
    return job


def run(probe_job, arguments=None):
    assert WRAPPER.is_file(), "Regularization GPU wrapper is missing"
    return subprocess.run(
        ["bash", str(WRAPPER), *map(str, probe_job["arguments"] if arguments is None else arguments)],
        env=probe_job["environment"], text=True, capture_output=True, timeout=30,
    )


def test_approved_resource_contract_and_shell_syntax():
    assert WRAPPER.is_file(), "Regularization GPU wrapper is missing"
    options = dict(line.removeprefix("#SBATCH --").split("=", 1) for line in WRAPPER.read_text().splitlines()
                   if line.startswith("#SBATCH --") and "=" in line)
    assert {name: options[name] for name in ("account", "partition", "nodes", "ntasks", "gres",
            "cpus-per-task", "mem", "time", "exclude", "job-name")} == {
        "account": "acc-mialhajri", "partition": "gpu", "nodes": "1", "ntasks": "1", "gres": "gpu:1",
        "cpus-per-task": "4", "mem": "12G", "time": "12:00:00", "exclude": "gpu-dy-g5-0-16",
        "job-name": "a-regularization",
    }
    assert subprocess.run(["bash", "-n", str(WRAPPER)], capture_output=True).returncode == 0


@pytest.mark.parametrize("command", ["smoke", "train", "replay"])
def test_dispatches_exactly_one_requested_phase_and_external_train_gate(probe_job, command):
    gate = "b" * 64 if command == "train" else "-"
    result = run(probe_job, [*probe_job["arguments"][:3], command, gate])
    assert result.returncode == 0, result.stderr
    call = json.loads(probe_job["calls"].read_text())
    expected = [str(probe_job["source"] / "scripts/probe_direct_reference_regularization.py"), command,
                "--request", str(probe_job["request"]), "--request-sha256", "a" * 64]
    if command == "train":
        expected += ["--gate-sha256", "b" * 64]
    assert call["arguments"] == expected
    assert call["cwd"] == str(probe_job["source"].parent)
    assert call["environment"] == dict(
        CONDA_PREFIX=ENV_PATH, PYTHONPATH=f"{probe_job['source']}/src:{probe_job['source']}/scripts",
        PYTHONHOME=None, PYTHONHASHSEED="42", CUBLAS_WORKSPACE_CONFIG=":4096:8",
        OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", NVIDIA_TF32_OVERRIDE="0",
        TORCH_ALLOW_TF32_CUBLAS_OVERRIDE="0", PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1",
        ROC_LEVIR_CKPT=None, ROC_LEVIR_SHA256=None, ROC_ANCHOR_GPU_INTEGRATION=None,
        ROC_DIRECT_REFERENCE_GPU_INTEGRATION=None,
    )
    assert not (probe_job["source"].parent / "scratch").exists()
    assert (probe_job["source"] / "scripts/probe_direct_reference_regularization.py").stat().st_mode & 0o222 == 0


@pytest.mark.parametrize("count", [0, 4, 6])
def test_wrong_argument_count_rejects_before_dispatch(probe_job, count):
    arguments = probe_job["arguments"][:count] if count < 5 else [*probe_job["arguments"], "extra"]
    result = run(probe_job, arguments)
    assert result.returncode != 0 and not probe_job["calls"].exists()


@pytest.mark.parametrize("index,value", [
    (2, "bad-sha"), (2, "A" * 64), (3, "evaluate"), (3, "test"),
    (4, "-"), (4, ""), (4, "B" * 64), (4, "b" * 63), (4, "g" * 64),
])
def test_invalid_command_or_train_hash_rejects_before_dispatch(probe_job, index, value):
    probe_job["arguments"][index] = value
    result = run(probe_job)
    assert result.returncode != 0 and not probe_job["calls"].exists()


@pytest.mark.parametrize("command", ["smoke", "replay"])
def test_nontraining_gate_hash_cannot_be_silently_ignored(probe_job, command):
    result = run(probe_job, [*probe_job["arguments"][:3], command, "b" * 64])
    assert result.returncode != 0 and not probe_job["calls"].exists()


@pytest.mark.parametrize("name,value", [
    ("SLURM_JOB_ID", ""), ("SLURM_JOB_ID", "0"), ("SLURM_JOB_ID", "12/x"),
    ("SLURMD_NODENAME", ""), ("SLURMD_NODENAME", "LOGIN01"), ("SLURMD_NODENAME", "head01"),
    ("SLURMD_NODENAME", "frontend01"), ("SLURMD_NODENAME", "lgn01"), ("SLURMD_NODENAME", "gpu-dy-g5-0-16"),
    ("SLURM_SUBMIT_DIR", ""), ("SLURM_JOB_PARTITION", "cpu"), ("SLURM_CPUS_PER_TASK", "3"),
    ("SLURM_CPUS_PER_TASK", "four"), ("SLURM_GPUS_ON_NODE", "0"), ("CUDA_VISIBLE_DEVICES", ""),
    ("CUDA_VISIBLE_DEVICES", "-1"), ("CUDA_VISIBLE_DEVICES", "NoDevFiles"), ("MOCK_PREFIX", "/wrong"),
    ("MOCK_INTERPRETER_PREFIX", "/wrong"), ("MOCK_EXECUTABLE", "/wrong/bin/python"),
])
def test_allocation_and_exact_environment_guards_precede_dispatch(probe_job, name, value):
    probe_job["environment"][name] = value
    result = run(probe_job)
    assert result.returncode != 0 and not probe_job["calls"].exists()


@pytest.mark.parametrize("target", ["root", "script", "module"])
def test_writable_source_rejects_before_dispatch(probe_job, target):
    path = {"root": probe_job["source"],
            "script": probe_job["source"] / "scripts/probe_direct_reference_regularization.py",
            "module": probe_job["source"] / "src/frozen.py"}[target]
    path.chmod(0o755 if path.is_dir() else 0o644)
    result = run(probe_job)
    assert result.returncode != 0 and not probe_job["calls"].exists()


def test_wrong_pinned_gate_is_forwarded_and_runner_failure_propagates(probe_job):
    probe_job["arguments"][4] = "c" * 64
    result = run(probe_job)
    assert result.returncode != 0 and "external gate hash mismatch" in result.stderr
    assert json.loads(probe_job["calls"].read_text())["arguments"][-2:] == ["--gate-sha256", "c" * 64]


def test_runner_failure_propagates(probe_job):
    probe_job["environment"]["BUILDER_EXIT"] = "37"
    assert run(probe_job).returncode == 37
