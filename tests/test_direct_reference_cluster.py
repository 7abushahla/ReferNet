import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "scripts/cluster/diagnose_direct_reference.sh"
ENV_PATH = "/shared/b00090279/conda/envs/refer-or-clear"
GPU_TEST = "tests/test_direct_reference_model.py::test_actual_direct_reference_gpu_integration"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def cluster_fixture(tmp_path):
    source = tmp_path / "scientific source"
    scripts = source / "scripts"
    scripts.mkdir(parents=True)
    (source / "src").mkdir()
    (source / "tests").mkdir()
    (source / "tests/test_direct_reference_model.py").write_text("raise AssertionError('no actual models in wrapper tests')\n")
    source_authority = source / "source-authority.txt"
    source_authority.write_text("synthetic immutable source inventory\n")
    stager = scripts / "direct_reference_job.py"
    stager.write_text('''import argparse, hashlib, json, os, shutil, sys
from pathlib import Path
import xml.etree.ElementTree as ET
parser = argparse.ArgumentParser()
subcommands = parser.add_subparsers(dest='operation', required=True)
prepare = subcommands.add_parser('prepare')
for name in ('source-dir', 'source-sha256', 'request', 'request-sha256', 'stage-dir', 'run-dir'):
    prepare.add_argument('--' + name, required=True)
stack = subcommands.add_parser('check-stack')
stack.add_argument('--report', required=True)
args = parser.parse_args()
with Path(os.environ['CALLS']).open('a') as stream:
    stream.write(json.dumps(dict(operation=args.operation, arguments=vars(args), script=__file__,
                                cwd=os.getcwd(), environment=dict(os.environ))) + '\\n')
if args.operation == 'check-stack':
    cases = ET.parse(args.report).getroot().findall('.//testcase')
    assert len(cases) == 1 and cases[0].get('name') == 'test_actual_direct_reference_gpu_integration'
    assert not any(cases[0].find(tag) is not None for tag in ('skipped', 'failure', 'error'))
    raise SystemExit(int(os.environ.get('STACK_EXIT', '0')))
raise_code = int(os.environ.get('PREPARE_EXIT', '0'))
if raise_code:
    raise SystemExit(raise_code)
source = Path(args.source_dir)
request = Path(args.request)
assert hashlib.sha256((source / 'source-authority.txt').read_bytes()).hexdigest() == args.source_sha256, 'source SHA256 mismatch'
content = request.read_bytes()
assert hashlib.sha256(content).hexdigest() == args.request_sha256, 'request SHA256 mismatch'
stage, output = Path(args.stage_dir), Path(args.run_dir)
for destination in (stage, output):
    assert destination.is_absolute() and '..' not in destination.parts
    assert not any(path.is_symlink() for path in (destination, *destination.parents))
    assert not destination.exists()
    assert not destination.is_relative_to(source) and not source.is_relative_to(destination)
    assert not destination.is_relative_to(request.parent) and not request.parent.is_relative_to(destination)
assert not stage.is_relative_to(output) and not output.is_relative_to(stage)
Path(os.environ['DATA_TOUCHES']).write_text('synthetic staging input reads authorized')
stage.mkdir(parents=True)
output.mkdir(parents=True)
shutil.copytree(source, stage / 'code')
(stage / 'inputs').mkdir()
(stage / 'tmp').mkdir()
(stage / 'inputs/request.json').write_bytes(content + (b'\\n' if os.environ.get('STAGED_REQUEST_DRIFT') else b''))
(stage / 'inputs/weights.pth').write_bytes(b'synthetic weights, not loaded')
''')
    (scripts / "check_direct_reference.py").write_text('''import argparse, hashlib, json, os
from pathlib import Path
parser = argparse.ArgumentParser()
for name in ('request', 'request-sha256', 'run-dir', 'device'):
    parser.add_argument('--' + name, required=True)
args = parser.parse_args()
assert args.device == 'cuda'
assert hashlib.sha256(Path(args.request).read_bytes()).hexdigest() == args.request_sha256
with Path(os.environ['CALLS']).open('a') as stream:
    stream.write(json.dumps(dict(operation='diagnostic', arguments=vars(args), script=__file__,
                                cwd=os.getcwd(), environment=dict(os.environ))) + '\\n')
raise SystemExit(int(os.environ.get('DIAGNOSTIC_EXIT', '0')))
''')
    metadata = tmp_path / "request metadata"
    metadata.mkdir()
    request = metadata / "request.json"
    request.write_text(json.dumps({"weights": {"sha256": hashlib.sha256(b"synthetic weights, not loaded").hexdigest()}}))
    interpreter = tmp_path / "interpreter-stub.py"
    interpreter.write_text(f'''import json, os, runpy, sys
from pathlib import Path
sys.prefix = os.environ.get('MOCK_INTERPRETER_PREFIX', {ENV_PATH!r})
arguments = sys.argv[1:]
with Path(os.environ['INTERPRETER_CALLS']).open('a') as stream:
    stream.write(json.dumps(dict(arguments=arguments, environment=dict(os.environ))) + '\\n')
if arguments[0] == '-':
    sys.argv = arguments
    exec(compile(sys.stdin.read(), '<wrapper-bootstrap>', 'exec'))
elif arguments[0] == '-c':
    sys.argv = arguments[1:]
    if arguments[1].startswith('import torch;'):
        raise SystemExit(int(os.environ.get('CUDA_PROBE_EXIT', '0')))
    exec(compile(arguments[1], '<wrapper-prefix-guard>', 'exec'))
elif arguments[:2] == ['-m', 'pytest']:
    if '--version' in arguments:
        print('pytest synthetic')
        raise SystemExit(0)
    with Path(os.environ['CALLS']).open('a') as stream:
        stream.write(json.dumps(dict(operation='pytest', arguments=arguments, cwd=os.getcwd(),
                                    environment=dict(os.environ))) + '\\n')
    outcome = os.environ.get('TEST_OUTCOME', 'pass')
    report = arguments[arguments.index('--junitxml') + 1]
    name = 'wrong_test' if outcome == 'wrong-name' else 'test_actual_direct_reference_gpu_integration'
    child = '<' + outcome + '/>' if outcome in ('skipped', 'failure', 'error') else ''
    case = '<testcase name="' + name + '">' + child + '</testcase>'
    if outcome == 'multiple':
        case += case
    if outcome != 'missing':
        Path(report).write_text('not XML' if outcome == 'malformed' else '<testsuites><testsuite>' + case + '</testsuite></testsuites>')
    raise SystemExit(int(os.environ.get('PYTEST_EXIT', '0')))
else:
    sys.argv = arguments
    runpy.run_path(arguments[0], run_name='__main__')
''')
    bash_env = tmp_path / "bash-env.sh"
    bash_env.write_text(f'''source() {{
    [[ "$1" == /opt/miniconda/etc/profile.d/conda.sh ]] || return 99
}}
conda() {{
    [[ "$1" == activate && "$2" == {ENV_PATH} ]] || return 98
    export CONDA_PREFIX="${{MOCK_PREFIX:-$2}}"
}}
function {ENV_PATH}/bin/python {{
    command {shlex.quote(sys.executable)} -B {shlex.quote(str(interpreter))} "$@"
}}
''')
    environment = dict(os.environ, BASH_ENV=str(bash_env), SLURM_JOB_ID="1234", SLURMD_NODENAME="gpu01",
                       SLURM_JOB_PARTITION="gpu", SLURM_CPUS_PER_TASK="4", SLURM_GPUS_ON_NODE="1",
                       SLURM_SUBMIT_DIR=str(tmp_path), CUDA_VISIBLE_DEVICES="0", USER="fixture-user",
                       ROC_SCRATCH_ROOT="scratch with spaces", CALLS=str(tmp_path / "calls.jsonl"),
                       INTERPRETER_CALLS=str(tmp_path / "interpreter-calls.jsonl"), DATA_TOUCHES=str(tmp_path / "data-touches"),
                       PYTHONPATH="/poisoned/scientific/source", ROC_LEVIR_SHA256="0" * 64,
                       ROC_LEVIR_CKPT="/poisoned/weights.pth", ROC_DIRECT_REFERENCE_GPU_INTEGRATION="1",
                       ROC_ANCHOR_GPU_INTEGRATION="1")
    for name in ("MOCK_PREFIX", "MOCK_INTERPRETER_PREFIX", "PREPARE_EXIT", "STACK_EXIT", "DIAGNOSTIC_EXIT",
                 "CUDA_PROBE_EXIT", "PYTEST_EXIT", "TEST_OUTCOME", "STAGED_REQUEST_DRIFT", "PYTHONHOME",
                 "SLURM_GPUS", "SLURM_JOB_GPUS", "SLURM_STEP_GPUS"):
        environment.pop(name, None)
    return dict(source=source, source_authority=source_authority, request=request, environment=environment,
                output=Path("output with spaces"), tmp_path=tmp_path)


def run_wrapper(fixture, arguments=None, **changes):
    assert WRAPPER.is_file(), "direct-reference diagnostic wrapper is not implemented"
    values = dict(fixture, **changes)
    arguments = arguments if arguments is not None else [str(values["source"]), values.get("source_hash", sha256(values["source_authority"])),
                                                       str(values["request"]), values.get("request_hash", sha256(values["request"])),
                                                       str(values["output"])]
    return subprocess.run(["bash", str(WRAPPER), *arguments], env=values["environment"], capture_output=True, text=True, timeout=15)


def calls(fixture, name="CALLS"):
    path = Path(fixture["environment"][name])
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def test_allocated_diagnostic_uses_frozen_scratch_and_absolute_interpreter(cluster_fixture):
    result = run_wrapper(cluster_fixture)
    assert result.returncode == 0, result.stderr
    observed = calls(cluster_fixture)
    assert [call["operation"] for call in observed] == ["prepare", "pytest", "check-stack", "diagnostic"]
    scratch = cluster_fixture["tmp_path"] / "scratch with spaces/direct-reference-diagnostic-1234"
    output = cluster_fixture["tmp_path"] / cluster_fixture["output"] / "direct-reference-diagnostic-1234"
    request_hash = sha256(cluster_fixture["request"])
    assert observed[0]["arguments"] == dict(operation="prepare", source_dir=str(cluster_fixture["source"]),
        source_sha256=sha256(cluster_fixture["source_authority"]), request=str(cluster_fixture["request"]),
        request_sha256=request_hash, stage_dir=str(scratch), run_dir=str(output))
    assert observed[0]["script"] == str(cluster_fixture["source"] / "scripts/direct_reference_job.py")
    selector = str(scratch / "code" / GPU_TEST.split("::")[0]) + "::" + GPU_TEST.split("::")[1]
    assert observed[1]["arguments"][2] == selector
    assert observed[2]["arguments"] == dict(operation="check-stack", report=str(output / "actual-stack-test.xml"))
    assert observed[2]["script"] == str(scratch / "code/scripts/direct_reference_job.py")
    assert observed[3]["arguments"] == dict(request=str(scratch / "inputs/request.json"), request_sha256=request_hash,
                                            run_dir=str(output), device="cuda")
    assert observed[3]["script"] == str(scratch / "code/scripts/check_direct_reference.py")
    for call in observed[1:]:
        assert call["cwd"] == str(scratch / "code")
        environment = call["environment"]
        assert environment["PYTHONPATH"] == str(scratch / "code/src")
        assert environment.get("ROC_ANCHOR_GPU_INTEGRATION") == ("1" if call["operation"] == "pytest" else None)
        assert "ROC_DIRECT_REFERENCE_GPU_INTEGRATION" not in environment
        assert environment["ROC_LEVIR_CKPT"] == str(scratch / "inputs/weights.pth")
        assert environment["ROC_LEVIR_SHA256"] == json.loads(cluster_fixture["request"].read_text())["weights"]["sha256"]
        for name, value in {"OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                            "NVIDIA_TF32_OVERRIDE": "0", "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE": "0", "PYTHONHASHSEED": "42",
                            "PYTHONDONTWRITEBYTECODE": "1", "CONDA_PREFIX": ENV_PATH}.items():
            assert environment[name] == value
        assert environment["TMPDIR"] == str(scratch / "tmp")
    assert observed[0]["environment"].get("PYTHONPATH", "") == ""
    assert "ROC_ANCHOR_GPU_INTEGRATION" not in observed[0]["environment"]
    assert "ROC_DIRECT_REFERENCE_GPU_INTEGRATION" not in observed[0]["environment"]
    assert (output / "actual-stack-test.log").is_file()
    assert (output / "pytest-version.txt").read_text().strip() == "pytest synthetic"


@pytest.mark.parametrize("name", ["SLURM_JOB_ID", "SLURMD_NODENAME", "SLURM_SUBMIT_DIR", "SLURM_CPUS_PER_TASK"])
def test_missing_allocation_fails_before_bootstrap(cluster_fixture, name):
    cluster_fixture["environment"].pop(name)
    assert run_wrapper(cluster_fixture).returncode != 0
    assert calls(cluster_fixture, "INTERPRETER_CALLS") == []


@pytest.mark.parametrize("name,value", [
    ("SLURMD_NODENAME", "login01"), ("SLURMD_NODENAME", "head01"), ("SLURMD_NODENAME", "frontend01"),
    ("SLURMD_NODENAME", "lgn01"), ("SLURM_JOB_PARTITION", "cpu"), ("SLURM_JOB_PARTITION", ""),
    ("SLURM_JOB_ID", "0"), ("SLURM_JOB_ID", "job123"), ("SLURM_CPUS_PER_TASK", "3"),
    ("SLURM_CPUS_PER_TASK", "four"), ("CUDA_VISIBLE_DEVICES", ""), ("CUDA_VISIBLE_DEVICES", "-1"),
    ("CUDA_VISIBLE_DEVICES", "NoDevFiles"), ("SLURM_GPUS_ON_NODE", "0"),
])
def test_invalid_gpu_allocation_fails_before_bootstrap(cluster_fixture, name, value):
    cluster_fixture["environment"][name] = value
    assert run_wrapper(cluster_fixture).returncode != 0
    assert calls(cluster_fixture, "INTERPRETER_CALLS") == []


@pytest.mark.parametrize("name,value", [("SLURM_GPUS", "1"), ("SLURM_JOB_GPUS", "0"), ("SLURM_STEP_GPUS", "0")])
def test_gpu_allocation_evidence_variants(cluster_fixture, name, value):
    cluster_fixture["environment"].pop("SLURM_GPUS_ON_NODE")
    cluster_fixture["environment"][name] = value
    result = run_wrapper(cluster_fixture)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("name", ["MOCK_PREFIX", "MOCK_INTERPRETER_PREFIX"])
def test_exact_environment_and_interpreter_guard_precedes_staging(cluster_fixture, name):
    cluster_fixture["environment"][name] = "/wrong/environment"
    result = run_wrapper(cluster_fixture)
    assert result.returncode != 0
    assert calls(cluster_fixture) == []
    assert not Path(cluster_fixture["environment"]["DATA_TOUCHES"]).exists()
    assert not (cluster_fixture["tmp_path"] / "scratch with spaces").exists()


def test_interpreter_guard_remains_enforced_with_python_optimization(cluster_fixture):
    cluster_fixture["environment"].update(MOCK_INTERPRETER_PREFIX="/wrong/environment", PYTHONOPTIMIZE="1")
    assert run_wrapper(cluster_fixture).returncode != 0
    assert calls(cluster_fixture) == []
    assert not Path(cluster_fixture["environment"]["DATA_TOUCHES"]).exists()


@pytest.mark.parametrize("mutation", ["source-hash", "request-hash", "source-drift"])
def test_stager_authenticates_bindings_before_data_reads(cluster_fixture, mutation):
    changes = {}
    if mutation == "source-hash":
        changes["source_hash"] = "0" * 64
    elif mutation == "request-hash":
        changes["request_hash"] = "0" * 64
    else:
        changes["source_hash"] = sha256(cluster_fixture["source_authority"])
        cluster_fixture["source_authority"].write_text("changed source\n")
    assert run_wrapper(cluster_fixture, **changes).returncode != 0
    assert [call["operation"] for call in calls(cluster_fixture)] == ["prepare"]
    assert not Path(cluster_fixture["environment"]["DATA_TOUCHES"]).exists()


@pytest.mark.parametrize("kind", ["source", "metadata", "existing-output", "existing-scratch", "symlink", "traversal"])
def test_protected_outputs_and_scratch_block_execution(cluster_fixture, kind):
    base = cluster_fixture["tmp_path"]
    changes = {}
    if kind in ("source", "metadata"):
        changes["output"] = cluster_fixture["source"] if kind == "source" else cluster_fixture["request"].parent
    elif kind == "existing-output":
        (base / cluster_fixture["output"] / "direct-reference-diagnostic-1234").mkdir(parents=True)
    elif kind == "existing-scratch":
        (base / "scratch with spaces/direct-reference-diagnostic-1234").mkdir(parents=True)
    elif kind == "symlink":
        link = base / "output link"
        link.symlink_to(cluster_fixture["source"], target_is_directory=True)
        changes["output"] = link
    else:
        changes["output"] = base / "unused/../escape"
    assert run_wrapper(cluster_fixture, **changes).returncode != 0
    assert calls(cluster_fixture)[-1]["operation"] == "prepare"
    assert not Path(cluster_fixture["environment"]["DATA_TOUCHES"]).exists()


def test_frozen_request_drift_blocks_gpu_and_diagnostic(cluster_fixture):
    cluster_fixture["environment"]["STAGED_REQUEST_DRIFT"] = "1"
    assert run_wrapper(cluster_fixture).returncode != 0
    assert [call["operation"] for call in calls(cluster_fixture)] == ["prepare"]
    assert not any(call["arguments"][:1] == ["-c"] and call["arguments"][1].startswith("import torch;")
                   for call in calls(cluster_fixture, "INTERPRETER_CALLS"))


@pytest.mark.parametrize("outcome", ["skipped", "failure", "error", "wrong-name", "multiple", "missing", "malformed"])
def test_exact_non_skipped_stack_gate_blocks_diagnostic(cluster_fixture, outcome):
    cluster_fixture["environment"]["TEST_OUTCOME"] = outcome
    assert run_wrapper(cluster_fixture).returncode != 0
    assert [call["operation"] for call in calls(cluster_fixture)] == ["prepare", "pytest", "check-stack"]


@pytest.mark.parametrize("name,code,operations", [
    ("PREPARE_EXIT", 19, ["prepare"]), ("CUDA_PROBE_EXIT", 20, ["prepare"]),
    ("PYTEST_EXIT", 21, ["prepare", "pytest"]), ("STACK_EXIT", 22, ["prepare", "pytest", "check-stack"]),
    ("DIAGNOSTIC_EXIT", 23, ["prepare", "pytest", "check-stack", "diagnostic"]),
])
def test_gate_failure_preserves_exit_and_blocks_later_operations(cluster_fixture, name, code, operations):
    cluster_fixture["environment"][name] = str(code)
    assert run_wrapper(cluster_fixture).returncode == code
    assert [call["operation"] for call in calls(cluster_fixture)] == operations


@pytest.mark.parametrize("arguments", [[], ["train"], ["train"] * 6])
def test_launcher_has_exact_five_argument_diagnostic_only_interface(cluster_fixture, arguments):
    assert run_wrapper(cluster_fixture, arguments=arguments).returncode == 2
    assert calls(cluster_fixture, "INTERPRETER_CALLS") == []


def test_gpu_headers_reuse_successful_launcher_without_long_training():
    assert WRAPPER.is_file(), "direct-reference diagnostic wrapper is not implemented"
    script = WRAPPER.read_text()
    baseline = (ROOT / "scripts/cluster/train_anchor_pilot.sh").read_text()
    headers = [line for line in baseline.splitlines() if line.startswith("#SBATCH") and "--job-name" not in line]
    assert all(header in script for header in headers)
    assert "source /opt/miniconda/etc/profile.d/conda.sh" in script
    assert f'python_bin="$environment_path/bin/python"' in script
    assert 'environment_path=' + ENV_PATH in script
    assert "--phase train" not in script and "--epochs" not in script
    assert "refer_or_clear.pilots" not in script and "sbatch " not in script and "ssh " not in script
