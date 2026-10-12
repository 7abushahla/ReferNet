import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import pytest

from test_anchor_candidate_runtime import catalogue as metadata_catalogue, complete


ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "scripts/cluster/bench_anchor_candidates.sh"
ENV_PATH = "/shared/b00090279/conda/envs/refer-or-clear"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def fixture(tmp_path):
    source = tmp_path / "measurement source"
    tools = source / "scripts"
    tools.mkdir(parents=True)
    coordinator = tools / "bench_anchor_candidates.py"
    coordinator.write_text('''import argparse, json, os
from pathlib import Path
parser = argparse.ArgumentParser()
parser.add_argument('--phase', choices=('prepare', 'measure'), required=True)
parser.add_argument('--candidates', required=True)
parser.add_argument('--candidates-sha256', required=True)
parser.add_argument('--run-dir', required=True)
args = parser.parse_args()
with Path(os.environ['CALLS']).open('a') as stream:
    stream.write(json.dumps(dict(arguments=vars(args), script=__file__,
                                environment=dict(os.environ))) + '\\n')
if args.phase != 'prepare':
    raise SystemExit('measurement must not be requested')
out = Path(args.run_dir)
out.mkdir(exist_ok=False)
status = os.environ.get('READINESS', 'metadata-ready')
(out / 'readiness.json').write_text(json.dumps(dict(
    status=status, comparable=False, execution_authorized=False)))
raise SystemExit(int(os.environ.get('COORDINATOR_EXIT', '0')))
''')
    for name in ("bench_anchor_pilots.py", "report_anchor_pilots.py", "bench_anchor_candidate_worker.py"):
        (tools / name).write_text("raise RuntimeError('must not import or execute this tool')\n")
    (source / "src").mkdir()
    (source / "src/poison.py").write_text("raise RuntimeError('measurement src forbidden')\n")
    (tools / "._unbound.py").write_text("forbidden")
    metadata = tmp_path / "wrapper metadata"
    metadata.mkdir()
    candidates = []
    for track in ("R", "D_running_pilot", "N"):
        parent = metadata / f"{track}.json"
        parent.write_text(json.dumps({"paths": {
            "source_dir": str(tmp_path / f"scientific-{track}"),
            "raw_root": str(tmp_path / f"raw-{track}"),
            "legacy_root": str(tmp_path / f"legacy-{track}"),
            "config": str(tmp_path / f"scientific-{track}/config.yaml"),
        }}))
        candidates.append({"track": track, "run_dir": str(tmp_path / f"run-{track}"),
                           "parent_manifest": {"path": str(parent), "sha256": sha256(parent)},
                           "scientific_source": {"directory": str(tmp_path / f"scientific-{track}")},
                           "relocations": {"records": {
                               "from": str(tmp_path / "original metadata/records.csv"),
                               "to": str(tmp_path / "relocated metadata/records.csv")}}})
    catalogue = metadata / "candidates.json"
    catalogue.write_text(json.dumps({
        "measurement_tools": {"directory": str(tools), "files": {
            path.name: sha256(path) for path in tools.glob("*.py") if not path.name.startswith("._")}},
        "candidates": candidates,
    }))
    interpreter = tmp_path / "interpreter_stub.py"
    interpreter.write_text(f'''import os, runpy, sys
sys.prefix = os.environ.get('MOCK_INTERPRETER_PREFIX', {ENV_PATH!r})
sys.argv = sys.argv[1:]
if sys.argv[0] == '-':
    exec(compile(sys.stdin.read(), '<wrapper-bootstrap>', 'exec'))
else:
    runpy.run_path(sys.argv[0], run_name='__main__')
''')
    bash_env = tmp_path / "bash-env.sh"
    bash_env.write_text(f'''source() {{
    [[ "$1" == /opt/miniconda3/etc/profile.d/conda.sh ]] || return 99
}}
conda() {{
    [[ "$1" == activate && "$2" == {ENV_PATH} ]] || return 98
    export CONDA_PREFIX="${{MOCK_PREFIX:-$2}}"
}}
function {ENV_PATH}/bin/python {{
    if [[ "$1" == -m && "$2" == pip ]]; then printf 'synthetic-environment\\n'; return "${{PIP_EXIT:-0}}"; fi
    command {shlex.quote(sys.executable)} -B {shlex.quote(str(interpreter))} "$@"
}}
''')
    environment = dict(os.environ, BASH_ENV=str(bash_env), SLURM_JOB_ID="1234",
                       SLURMD_NODENAME="cpu01", SLURM_JOB_PARTITION="cpu", SLURM_CPUS_PER_TASK="4",
                       SLURM_SUBMIT_DIR=str(tmp_path), USER="fixture-user",
                       ROC_SCRATCH_ROOT="scratch with spaces", CALLS=str(tmp_path / "calls.jsonl"),
                       PYTHONPATH=str(source / "src"), CUDA_VISIBLE_DEVICES="0")
    for name in ("MOCK_PREFIX", "MOCK_INTERPRETER_PREFIX", "READINESS", "COORDINATOR_EXIT", "PIP_EXIT", "SLURM_GPUS",
                 "SLURM_GPUS_ON_NODE", "SLURM_JOB_GPUS", "SLURM_STEP_GPUS"):
        environment.pop(name, None)
    return dict(source=source, tools=tools, catalogue=catalogue, environment=environment,
                output=Path("output with spaces"), tmp_path=tmp_path)


def run(fixture, **changes):
    assert WRAPPER.is_file(), "candidate CPU wrapper is not implemented"
    values = dict(fixture, **changes)
    return subprocess.run(["bash", str(WRAPPER), str(values["source"]), str(values["catalogue"]),
                           values.get("hash", sha256(values["catalogue"])), str(values["output"])],
                          env=values["environment"], capture_output=True, text=True)


def calls(fixture):
    path = Path(fixture["environment"]["CALLS"])
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def test_preparation_stages_only_authenticated_tools_and_never_measures(fixture):
    result = run(fixture)
    assert result.returncode == 0, result.stderr
    observed = calls(fixture)
    assert len(observed) == 1
    invocation = observed[0]
    output = fixture["tmp_path"] / fixture["output"] / "anchor-candidate-cpu-v1-1234"
    assert invocation["arguments"] == dict(phase="prepare", candidates=str(fixture["catalogue"]),
                                           candidates_sha256=sha256(fixture["catalogue"]),
                                           run_dir=str(output / "preparation"))
    scratch = fixture["tmp_path"] / "scratch with spaces/anchor-candidate-cpu-v1-1234"
    assert Path(invocation["script"]) == scratch / "tools/bench_anchor_candidates.py"
    assert sorted(path.name for path in (scratch / "tools").iterdir()) == [
        "bench_anchor_candidate_worker.py", "bench_anchor_candidates.py", "bench_anchor_pilots.py",
        "report_anchor_pilots.py"]
    environment = invocation["environment"]
    assert environment.get("PYTHONPATH", "") == ""
    assert environment["CUDA_VISIBLE_DEVICES"] == ""
    assert environment["CONDA_PREFIX"] == ENV_PATH
    assert environment["TMPDIR"] == str(scratch / "tmp")
    for key, expected in {"OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "PYTHONHASHSEED": "42",
                          "PYTHONDONTWRITEBYTECODE": "1", "NVIDIA_TF32_OVERRIDE": "0",
                          "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE": "0",
                          "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}.items():
        assert environment[key] == expected
    snapshot = json.loads((output / "allocation.json").read_text())
    assert snapshot["catalogue_sha256"] == sha256(fixture["catalogue"])
    assert snapshot["wrapper_sha256"] == sha256(WRAPPER)
    assert snapshot["interpreter_prefix"] == ENV_PATH
    assert snapshot["candidate_order"] == ["R", "D_running_pilot", "N"]
    assert snapshot["execution_authorized"] is False
    assert snapshot["candidates"][0]["relocations"]["records"]["from"].endswith("original metadata/records.csv")
    assert json.loads((output / "preparation/readiness.json").read_text())["execution_authorized"] is False


@pytest.mark.parametrize("key", ["SLURM_JOB_ID", "SLURMD_NODENAME", "SLURM_SUBMIT_DIR", "SLURM_CPUS_PER_TASK"])
def test_missing_allocation_metadata_blocks_execution(fixture, key):
    fixture["environment"].pop(key)
    assert run(fixture).returncode != 0
    assert calls(fixture) == []


@pytest.mark.parametrize("key,value", [
    ("SLURMD_NODENAME", "login01"), ("SLURMD_NODENAME", "head01"),
    ("SLURMD_NODENAME", "frontend01"), ("SLURMD_NODENAME", "lgn01"),
    ("SLURM_JOB_ID", "job-123"), ("SLURM_JOB_ID", "0"),
    ("SLURM_JOB_PARTITION", "gpu"), ("SLURM_JOB_PARTITION", ""),
    ("SLURM_CPUS_PER_TASK", "3"), ("SLURM_CPUS_PER_TASK", "four"),
    ("MOCK_PREFIX", "/wrong/environment"),
])
def test_invalid_allocation_or_environment_blocks_execution(fixture, key, value):
    fixture["environment"][key] = value
    assert run(fixture).returncode != 0
    assert calls(fixture) == []
    assert not (fixture["tmp_path"] / fixture["output"]).exists()


@pytest.mark.parametrize("key,value", [("SLURM_GPUS", "1"), ("SLURM_GPUS_ON_NODE", "1"),
                                       ("SLURM_JOB_GPUS", "0"), ("SLURM_STEP_GPUS", "0")])
def test_cpu_partition_with_gpu_allocation_is_rejected(fixture, key, value):
    fixture["environment"][key] = value
    result = run(fixture)
    assert result.returncode == 2
    assert calls(fixture) == []


def test_wrong_interpreter_prefix_is_rejected_despite_claimed_conda_prefix(fixture):
    fixture["environment"]["MOCK_INTERPRETER_PREFIX"] = "/wrong/interpreter"
    result = run(fixture)
    assert result.returncode != 0
    assert "interpreter prefix mismatch" in result.stderr
    assert calls(fixture) == []
    assert not (fixture["tmp_path"] / fixture["output"]).exists()
    assert not (fixture["tmp_path"] / "scratch with spaces").exists()


@pytest.mark.parametrize("mutation", ["catalogue-hash", "tool-hash", "coordinator-unbound", "outside-tools", "tool-traversal"])
def test_unauthenticated_catalogue_or_tool_blocks_execution(fixture, mutation):
    catalogue = json.loads(fixture["catalogue"].read_text())
    overrides = {}
    if mutation == "catalogue-hash":
        overrides["hash"] = "0" * 64
    elif mutation == "tool-hash":
        (fixture["tools"] / "bench_anchor_candidates.py").write_text("raise RuntimeError('changed')\n")
    elif mutation == "coordinator-unbound":
        del catalogue["measurement_tools"]["files"]["bench_anchor_candidates.py"]
    elif mutation == "outside-tools":
        overrides["source"] = fixture["tmp_path"] / "other-source"
        overrides["source"].mkdir()
    else:
        catalogue["measurement_tools"]["files"]["../escape.py"] = "0" * 64
    fixture["catalogue"].write_text(json.dumps(catalogue))
    assert run(fixture, **overrides).returncode != 0
    assert calls(fixture) == []
    assert not (fixture["tmp_path"] / fixture["output"]).exists()


@pytest.mark.parametrize("kind", ["existing-output", "existing-scratch", "source", "scientific", "raw", "relocation", "ancestor", "symlink", "traversal", "scratch-overlap"])
def test_output_and_scratch_protection_precedes_execution(fixture, kind):
    base = fixture["tmp_path"]
    output = fixture["output"]
    if kind == "existing-output":
        (base / output / "anchor-candidate-cpu-v1-1234").mkdir(parents=True)
    elif kind == "existing-scratch":
        (base / "scratch with spaces/anchor-candidate-cpu-v1-1234").mkdir(parents=True)
    elif kind in ("source", "scientific", "raw", "relocation"):
        output = {"source": fixture["source"], "scientific": base / "scientific-R",
                  "raw": base / "raw-R", "relocation": base / "original metadata"}[kind]
    elif kind == "ancestor":
        parent = base / "parent/anchor-candidate-cpu-v1-1234"
        parent.mkdir(parents=True)
        nested = parent / "metadata"
        fixture["catalogue"].rename(nested.with_suffix(".json"))
        fixture["catalogue"] = nested.with_suffix(".json")
        output = base / "parent"
    elif kind == "symlink":
        (base / "link").symlink_to(fixture["source"], target_is_directory=True)
        output = base / "link"
    elif kind == "traversal":
        output = base / "unused/../escape"
    else:
        fixture["environment"]["ROC_SCRATCH_ROOT"] = str(fixture["source"])
    assert run(fixture, output=output).returncode != 0
    assert calls(fixture) == []


@pytest.mark.parametrize("destination", ["output", "scratch"])
def test_line_break_paths_are_rejected_before_creating_destinations(fixture, destination):
    overrides = {}
    if destination == "output":
        overrides["output"] = Path("output\nwith newline")
    else:
        fixture["environment"]["ROC_SCRATCH_ROOT"] = "scratch\nwith newline"
    assert run(fixture, **overrides).returncode != 0
    assert calls(fixture) == []
    assert not (fixture["tmp_path"] / overrides.get("output", fixture["output"])).exists()


@pytest.mark.parametrize("status,exit_code", [("waiting-for-completion", 2), ("not-comparable", 1),
                                              ("metadata-ready", 23)])
def test_coordinator_failure_or_missing_completion_is_preserved(fixture, status, exit_code):
    fixture["environment"].update(READINESS=status, COORDINATOR_EXIT=str(exit_code))
    result = run(fixture)
    assert result.returncode == exit_code, result.stderr
    assert [item["arguments"]["phase"] for item in calls(fixture)] == ["prepare"]


def test_environment_snapshot_failure_blocks_coordinator(fixture):
    fixture["environment"]["PIP_EXIT"] = "19"
    assert run(fixture).returncode == 19
    assert calls(fixture) == []


@pytest.mark.parametrize("state,exit_code", [("pending", 2), ("complete", 0), ("failed", 1)])
def test_real_prepare_interface_never_turns_metadata_into_execution(fixture, metadata_catalogue, state, exit_code):
    tools = Path(metadata_catalogue["measurement_tools"]["directory"])
    coordinator = tools / "bench_anchor_candidates.py"
    coordinator.write_bytes((ROOT / "scripts/bench_anchor_candidates.py").read_bytes())
    metadata_catalogue["measurement_tools"]["files"][coordinator.name] = sha256(coordinator)
    worker = tools / "bench_anchor_candidate_worker.py"
    worker.write_text("raise RuntimeError('worker execution is not authorized')\n")
    metadata_catalogue["measurement_tools"]["files"][worker.name] = sha256(worker)
    if state == "complete":
        for candidate in metadata_catalogue["candidates"]:
            complete(candidate)
    elif state == "failed":
        metadata_catalogue["candidates"][0]["status"] = "failed"
    fixture["catalogue"].write_text(json.dumps(metadata_catalogue))
    result = run(fixture, source=tools)
    assert result.returncode == exit_code, result.stderr
    output = fixture["tmp_path"] / fixture["output"] / "anchor-candidate-cpu-v1-1234"
    readiness = json.loads((output / "preparation/readiness.json").read_text())
    assert readiness["status"] == {"pending": "waiting-for-completion", "complete": "metadata-ready",
                                   "failed": "not-comparable"}[state]
    assert readiness["comparable"] is False
    assert readiness["execution_authorized"] is False
    assert "durations" not in readiness
    assert not (output / "runtime.json").exists()
    assert calls(fixture) == []


def test_resource_directives_and_default_scratch_are_cpu_only():
    assert WRAPPER.is_file(), "candidate CPU wrapper is not implemented"
    script = WRAPPER.read_text()
    directives = [line for line in script.splitlines() if line.startswith("#SBATCH")]
    for option in ("--account=acc-mialhajri", "--partition=cpu", "--nodes=1", "--ntasks=1",
                   "--cpus-per-task=4", "--mem=16G", "--time=2:00:00"):
        assert f"#SBATCH {option}" in directives
    assert not any("--gres" in line or "--gpu" in line for line in directives)
    assert "sbatch " not in script and "srun " not in script and "ssh " not in script
    assignment = next(line for line in script.splitlines() if line.startswith("scratch="))
    environment = dict(os.environ, USER="fixture-user", SLURM_JOB_ID="1234")
    environment.pop("ROC_SCRATCH_ROOT", None)
    result = subprocess.run(["bash", "-c", assignment + '\nprintf "%s" "$scratch"'],
                            env=environment, capture_output=True, text=True, check=True)
    assert result.stdout == "/tmp/fixture-user/anchor-candidate-cpu-v1-1234"
