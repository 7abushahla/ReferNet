from pathlib import Path

import pytest

import test_direct_reference_cluster as diagnostic
from test_direct_reference_cluster import cluster_fixture


ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "scripts/cluster/benchmark_direct_reference.sh"


@pytest.fixture
def throughput_fixture(cluster_fixture, monkeypatch):
    monkeypatch.setattr(diagnostic, "WRAPPER", WRAPPER)
    scripts = cluster_fixture["source"] / "scripts"
    original = (scripts / "check_direct_reference.py").read_text()
    (scripts / "benchmark_direct_reference_throughput.py").write_text(
        original.replace("operation='diagnostic'", "operation='throughput'")
    )
    return cluster_fixture


def test_benchmark_uses_allocated_staged_source_and_exact_environment(throughput_fixture):
    result = diagnostic.run_wrapper(throughput_fixture)
    assert result.returncode == 0, result.stderr
    observed = diagnostic.calls(throughput_fixture)
    assert [entry["operation"] for entry in observed] == ["prepare", "throughput"]
    scratch = throughput_fixture["tmp_path"] / "scratch with spaces/direct-reference-throughput-1234"
    output = throughput_fixture["tmp_path"] / throughput_fixture["output"] / "direct-reference-throughput-1234"
    measured = observed[-1]
    assert measured["script"] == str(scratch / "code/scripts/benchmark_direct_reference_throughput.py")
    assert measured["arguments"] == dict(
        request=str(scratch / "inputs/request.json"),
        request_sha256=diagnostic.sha256(throughput_fixture["request"]),
        run_dir=str(output), device="cuda",
    )
    assert measured["cwd"] == str(scratch / "code")
    assert measured["environment"]["CONDA_PREFIX"] == diagnostic.ENV_PATH
    assert measured["environment"]["PYTHONPATH"] == str(scratch / "code/src")
    assert measured["environment"]["ROC_LEVIR_CKPT"] == str(scratch / "inputs/weights.pth")
    assert measured["environment"]["TMPDIR"] == str(scratch / "tmp")


@pytest.mark.parametrize("name,value", [
    ("SLURMD_NODENAME", "login01"), ("SLURM_JOB_PARTITION", "cpu"),
    ("SLURM_JOB_ID", "0"), ("SLURM_CPUS_PER_TASK", "3"),
    ("CUDA_VISIBLE_DEVICES", ""), ("SLURM_GPUS_ON_NODE", "0"),
])
def test_benchmark_rejects_invalid_allocation_before_bootstrap(throughput_fixture, name, value):
    throughput_fixture["environment"][name] = value
    assert diagnostic.run_wrapper(throughput_fixture).returncode != 0
    assert diagnostic.calls(throughput_fixture, "INTERPRETER_CALLS") == []


@pytest.mark.parametrize("name", ["MOCK_PREFIX", "MOCK_INTERPRETER_PREFIX"])
def test_benchmark_rejects_wrong_environment_before_staging(throughput_fixture, name):
    throughput_fixture["environment"][name] = "/wrong/environment"
    assert diagnostic.run_wrapper(throughput_fixture).returncode != 0
    assert diagnostic.calls(throughput_fixture) == []


@pytest.mark.parametrize("name,code,operations", [
    ("PREPARE_EXIT", 19, ["prepare"]),
    ("CUDA_PROBE_EXIT", 20, ["prepare"]),
    ("DIAGNOSTIC_EXIT", 23, ["prepare", "throughput"]),
])
def test_benchmark_propagates_failure(throughput_fixture, name, code, operations):
    throughput_fixture["environment"][name] = str(code)
    result = diagnostic.run_wrapper(throughput_fixture)
    assert result.returncode == code
    assert [entry["operation"] for entry in diagnostic.calls(throughput_fixture)] == operations


def test_benchmark_is_bounded_and_cannot_submit_full_training():
    script = WRAPPER.read_text()
    original = (ROOT / "scripts/cluster/diagnose_direct_reference.sh").read_text()
    for line in original.splitlines():
        if line.startswith("#SBATCH") and "--job-name" not in line and "--time" not in line:
            assert line in script
    assert "#SBATCH --time=0:30:00" in script
    assert "source /opt/miniconda/etc/profile.d/conda.sh" in script
    assert "--epochs" not in script and "train_direct_reference_control.py" not in script
    assert "sbatch " not in script and "ssh " not in script
