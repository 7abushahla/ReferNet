import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from test_direct_reference_cluster import GPU_TEST, ROOT, calls, cluster_fixture, sha256


WRAPPER = ROOT / "scripts/cluster/train_direct_reference_control.sh"


@pytest.fixture
def control_fixture(cluster_fixture):
    fixture = cluster_fixture
    config = fixture["source"] / "configs/direct_reference_pilot.yaml"
    config.parent.mkdir()
    config.write_text("epochs: 100\nfold: 0\nseed: 42\n")
    request = json.loads(fixture["request"].read_text())
    request.update(schema_version="direct_reference_diagnostic_request_v1",
                   source_sha256=sha256(fixture["source_authority"]),
                   config={"path": str(config), "sha256": sha256(config)})
    fixture["request"].write_text(json.dumps(request))
    authority_root = fixture["tmp_path"] / "control authorities"
    authority_root.mkdir()
    gate = authority_root / "diagnostic.json"
    gate.write_text(json.dumps({"schema_version": "direct_reference_diagnostic_v1", "passed": True,
                               "diagnostic_only": True, "no_test_open": True,
                               "identities": {"source_sha256": request["source_sha256"]}}))
    gate.chmod(0o444)
    control = authority_root / "control.json"
    control.write_text(json.dumps(dict(schema_version="direct_reference_control_request_v1",
        source_sha256=request["source_sha256"], config_sha256=sha256(config),
        diagnostic_request_sha256=sha256(fixture["request"]), arm="off", epochs=100, fold=0, seed=42,
        diagnostic_gate={"path": str(gate), "sha256": sha256(gate)})))
    control.chmod(0o444)
    fixture.update(control=control, gate=gate, config=config)
    (fixture["source"] / "scripts/train_direct_reference_control.py").write_text('''import argparse, hashlib, json, os
from pathlib import Path
parser = argparse.ArgumentParser(allow_abbrev=False)
for name in ('request', 'request-sha256', 'control-request', 'control-request-sha256', 'run-dir', 'device'):
    parser.add_argument('--' + name, required=True)
args = parser.parse_args()
assert args.device == 'cuda'
assert hashlib.sha256(Path(args.request).read_bytes()).hexdigest() == args.request_sha256
assert hashlib.sha256(Path(args.control_request).read_bytes()).hexdigest() == args.control_request_sha256
with Path(os.environ['CALLS']).open('a') as stream:
    stream.write(json.dumps(dict(operation='control', arguments=vars(args), script=__file__,
                                cwd=os.getcwd(), environment=dict(os.environ))) + '\\n')
raise SystemExit(int(os.environ.get('CONTROL_EXIT', '0')))
''')
    stager = fixture["source"] / "scripts/direct_reference_job.py"
    stager.write_text(stager.read_text() + '''
if os.environ.get('CONTROL_DRIFT_AFTER_STAGE'):
    control = Path(os.environ['CONTROL_PATH'])
    control.chmod(0o644)
    control.write_bytes(control.read_bytes() + b'\\n')
if os.environ.get('GATE_DRIFT_AFTER_STAGE'):
    gate = Path(os.environ['GATE_PATH'])
    gate.chmod(0o644)
    gate.write_bytes(gate.read_bytes() + b'\\n')
''')
    fixture["environment"].update(CONTROL_PATH=str(control), GATE_PATH=str(gate))
    interpreter = fixture["tmp_path"] / "interpreter-stub.py"
    interpreter.write_text(interpreter.read_text().replace("startswith('import torch;')", "startswith('import torch')"))
    for name in ("CONTROL_EXIT", "CONTROL_DRIFT_AFTER_STAGE", "GATE_DRIFT_AFTER_STAGE"):
        fixture["environment"].pop(name, None)
    return fixture


def run_control(fixture, arguments=None, **changes):
    assert WRAPPER.is_file(), "direct-reference control wrapper is not implemented"
    values = dict(fixture, **changes)
    arguments = arguments if arguments is not None else [str(values["source"]),
        values["source_hash"] if "source_hash" in values else sha256(values["source_authority"]), str(values["request"]),
        values.get("request_hash", sha256(values["request"])), str(values["control"]),
        values["control_hash"] if "control_hash" in values else sha256(values["control"]), str(values["output"])]
    return subprocess.run(["bash", str(WRAPPER), *arguments], env=values["environment"],
                          capture_output=True, text=True, timeout=15)


def rewrite_control(fixture, **changes):
    content = json.loads(fixture["control"].read_text())
    content.update(changes)
    fixture["control"].chmod(0o644)
    fixture["control"].write_text(json.dumps(content))
    fixture["control"].chmod(0o444)


def assert_preimport_rejection(fixture, result):
    assert result.returncode != 0
    assert calls(fixture) == []
    assert not Path(fixture["environment"]["DATA_TOUCHES"]).exists()
    assert not (fixture["tmp_path"] / "scratch with spaces").exists()
    assert not any(call["arguments"][:1] == ["-c"] and call["arguments"][1].startswith("import torch")
                   for call in calls(fixture, "INTERPRETER_CALLS"))


def test_control_launch_uses_only_explicit_off_authority_and_frozen_code(control_fixture):
    original = control_fixture["control"].read_bytes()
    result = run_control(control_fixture)
    assert result.returncode == 0, result.stderr
    observed = calls(control_fixture)
    assert [call["operation"] for call in observed] == ["prepare", "pytest", "check-stack", "control"]
    scratch = control_fixture["tmp_path"] / "scratch with spaces/direct-reference-control-1234"
    output = control_fixture["tmp_path"] / control_fixture["output"] / "direct-reference-control-1234"
    assert observed[0]["arguments"] == dict(operation="prepare", source_dir=str(control_fixture["source"]),
        source_sha256=sha256(control_fixture["source_authority"]), request=str(control_fixture["request"]),
        request_sha256=sha256(control_fixture["request"]), stage_dir=str(scratch), run_dir=str(output))
    assert observed[1]["arguments"][2] == str(scratch / "code" / GPU_TEST.split("::")[0]) + "::" + GPU_TEST.split("::")[1]
    assert observed[1]["arguments"][3:5] == ["-p", "no:cacheprovider"]
    assert observed[2]["arguments"] == dict(operation="check-stack", report=str(output / "actual-stack-test.xml"))
    assert observed[3]["arguments"] == dict(request=str(scratch / "inputs/request.json"),
        request_sha256=sha256(control_fixture["request"]), control_request=str(control_fixture["control"]),
        control_request_sha256=hashlib.sha256(original).hexdigest(), run_dir=str(output), device="cuda")
    assert observed[3]["script"] == str(scratch / "code/scripts/train_direct_reference_control.py")
    assert calls(control_fixture, "INTERPRETER_CALLS")[-1]["arguments"] == [
        str(scratch / "code/scripts/train_direct_reference_control.py"),
        "--request", str(scratch / "inputs/request.json"), "--request-sha256", sha256(control_fixture["request"]),
        "--control-request", str(control_fixture["control"]), "--control-request-sha256", hashlib.sha256(original).hexdigest(),
        "--run-dir", str(output), "--device", "cuda"]
    assert control_fixture["control"].read_bytes() == original
    for call in observed[1:]:
        assert call["cwd"] == str(scratch / "code")
        environment = call["environment"]
        assert environment["PYTHONPATH"] == str(scratch / "code/src")
        assert environment["ROC_LEVIR_CKPT"] == str(scratch / "inputs/weights.pth")
        assert environment["ROC_LEVIR_SHA256"] == json.loads(control_fixture["request"].read_text())["weights"]["sha256"]
        assert environment.get("ROC_ANCHOR_GPU_INTEGRATION") == ("1" if call["operation"] == "pytest" else None)
        assert "ROC_DIRECT_REFERENCE_GPU_INTEGRATION" not in environment
        for name, value in {"OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                            "NVIDIA_TF32_OVERRIDE": "0", "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE": "0",
                            "PYTHONHASHSEED": "42", "PYTHONDONTWRITEBYTECODE": "1"}.items():
            assert environment[name] == value
    assert all("ROC_ANCHOR_GPU_INTEGRATION" not in call["environment"]
               for call in calls(control_fixture, "INTERPRETER_CALLS") if call["arguments"][:1] == ["-"])


@pytest.mark.parametrize("field,value", [
    ("schema_version", "direct_reference_diagnostic_request_v1"), ("source_sha256", "0" * 64),
    ("config_sha256", "0" * 64), ("diagnostic_request_sha256", "0" * 64), ("arm", "on"),
    ("epochs", 1), ("epochs", True), ("epochs", 100.0), ("seed", 41), ("seed", "42"),
    ("fold", 1), ("fold", False), ("resume", "selected.pt"), ("sweep", ["off", "on"]),
])
def test_control_recipe_or_authority_mismatch_fails_before_staging(control_fixture, field, value):
    rewrite_control(control_fixture, **{field: value})
    assert_preimport_rejection(control_fixture, run_control(control_fixture))


@pytest.mark.parametrize("mutation", ["control-hash", "diag-hash", "missing-control", "missing-gate", "gate-hash", "duplicate", "nonfinite", "config-drift"])
def test_external_authorities_are_authenticated_before_staging(control_fixture, mutation):
    changes = {}
    if mutation == "control-hash":
        changes["control_hash"] = "0" * 64
    elif mutation == "diag-hash":
        changes["request_hash"] = "0" * 64
    elif mutation == "missing-control":
        changes["control_hash"] = sha256(control_fixture["control"])
        control_fixture["control"].unlink()
    elif mutation == "missing-gate":
        control_fixture["gate"].unlink()
    elif mutation == "gate-hash":
        rewrite_control(control_fixture, diagnostic_gate={"path": str(control_fixture["gate"]), "sha256": "0" * 64})
    elif mutation in ("duplicate", "nonfinite"):
        control_fixture["control"].chmod(0o644)
        content = control_fixture["control"].read_text()
        suffix = ', "arm": "off"}' if mutation == "duplicate" else ', "unexpected": NaN}'
        control_fixture["control"].write_text(content[:-1] + suffix)
    else:
        control_fixture["config"].write_text("epochs: 1\n")
    assert_preimport_rejection(control_fixture, run_control(control_fixture, **changes))


@pytest.mark.parametrize("field,value", [
    ("schema_version", "anchor_diagnostic_v1"), ("passed", False), ("passed", 1),
    ("no_test_open", False), ("diagnostic_only", False),
    ("identities", {"source_sha256": "0" * 64}), ("identities", None),
])
def test_failed_or_old_source_diagnostic_gate_blocks_staging(control_fixture, field, value):
    content = json.loads(control_fixture["gate"].read_text())
    content[field] = value
    control_fixture["gate"].chmod(0o644)
    control_fixture["gate"].write_text(json.dumps(content))
    rewrite_control(control_fixture, diagnostic_gate={"path": str(control_fixture["gate"]), "sha256": sha256(control_fixture["gate"])})
    assert_preimport_rejection(control_fixture, run_control(control_fixture))


@pytest.mark.parametrize("mutation", ["source-hash", "source-drift", "missing-source"])
def test_stager_rejects_source_loss_before_data_or_models(control_fixture, mutation):
    source_hash = sha256(control_fixture["source_authority"])
    if mutation == "source-hash":
        source_hash = "0" * 64
        request = json.loads(control_fixture["request"].read_text())
        request["source_sha256"] = source_hash
        control_fixture["request"].write_text(json.dumps(request))
        gate = json.loads(control_fixture["gate"].read_text())
        gate["identities"]["source_sha256"] = source_hash
        control_fixture["gate"].chmod(0o644)
        control_fixture["gate"].write_text(json.dumps(gate))
        rewrite_control(control_fixture, source_sha256=source_hash,
                        diagnostic_request_sha256=sha256(control_fixture["request"]),
                        diagnostic_gate={"path": str(control_fixture["gate"]), "sha256": sha256(control_fixture["gate"])})
    elif mutation == "source-drift":
        control_fixture["source_authority"].write_text("changed source inventory\n")
    else:
        control_fixture["source_authority"].unlink()
    assert run_control(control_fixture, source_hash=source_hash).returncode != 0
    assert [call["operation"] for call in calls(control_fixture)] == ["prepare"]
    assert not Path(control_fixture["environment"]["DATA_TOUCHES"]).exists()


@pytest.mark.parametrize("kind", ["output-control", "scratch-control", "output-gate", "scratch-gate", "ancestor", "symlink", "traversal"])
def test_outputs_cannot_modify_control_input_trees(control_fixture, kind):
    base = control_fixture["tmp_path"]
    changes = {}
    if kind.endswith("gate"):
        gate_root = base / "independent gate authority"
        gate_root.mkdir()
        gate = gate_root / "diagnostic.json"
        gate.write_bytes(control_fixture["gate"].read_bytes())
        rewrite_control(control_fixture, diagnostic_gate={"path": str(gate), "sha256": sha256(gate)})
        root = gate_root
    else:
        root = control_fixture["control"].parent
    if kind.startswith("scratch"):
        control_fixture["environment"]["ROC_SCRATCH_ROOT"] = str(root)
    elif kind == "ancestor":
        changes["output"] = base
    elif kind == "symlink":
        link = base / "output link"
        link.symlink_to(root, target_is_directory=True)
        changes["output"] = link
    elif kind == "traversal":
        changes["output"] = base / "unused/../escape"
    else:
        changes["output"] = root
    if kind == "ancestor":
        control = base / "output with spaces/direct-reference-control-1234/authority/control.json"
        control.parent.mkdir(parents=True)
        control.write_bytes(control_fixture["control"].read_bytes())
        changes["control"] = control
        changes["output"] = base / "output with spaces"
    assert_preimport_rejection(control_fixture, run_control(control_fixture, **changes))


@pytest.mark.parametrize("name", ["CONTROL_DRIFT_AFTER_STAGE", "GATE_DRIFT_AFTER_STAGE", "STAGED_REQUEST_DRIFT"])
def test_authority_drift_after_staging_blocks_first_model_import(control_fixture, name):
    control_fixture["environment"][name] = "1"
    result = run_control(control_fixture)
    assert result.returncode != 0
    assert [call["operation"] for call in calls(control_fixture)] == ["prepare"]
    assert not any(call["arguments"][:1] == ["-c"] and call["arguments"][1].startswith("import torch")
                   for call in calls(control_fixture, "INTERPRETER_CALLS"))


@pytest.mark.parametrize("name,value", [
    ("SLURMD_NODENAME", "login01"), ("SLURMD_NODENAME", "head01"), ("SLURMD_NODENAME", "frontend01"),
    ("SLURMD_NODENAME", "lgn01"), ("SLURM_JOB_PARTITION", "cpu"), ("SLURM_JOB_ID", "0"),
    ("SLURM_CPUS_PER_TASK", "3"), ("CUDA_VISIBLE_DEVICES", ""), ("CUDA_VISIBLE_DEVICES", "-1"),
    ("SLURM_GPUS_ON_NODE", "0"),
])
def test_invalid_allocation_blocks_bootstrap(control_fixture, name, value):
    control_fixture["environment"][name] = value
    assert run_control(control_fixture).returncode != 0
    assert calls(control_fixture, "INTERPRETER_CALLS") == []


@pytest.mark.parametrize("name", ["MOCK_PREFIX", "MOCK_INTERPRETER_PREFIX"])
def test_exact_environment_guard_precedes_authority_reads(control_fixture, name):
    control_fixture["environment"].update({name: "/wrong/environment", "PYTHONOPTIMIZE": "1"})
    assert_preimport_rejection(control_fixture, run_control(control_fixture))


@pytest.mark.parametrize("outcome", ["skipped", "failure", "error", "wrong-name", "multiple", "missing", "malformed"])
def test_nonpassing_actual_stack_blocks_control(control_fixture, outcome):
    control_fixture["environment"]["TEST_OUTCOME"] = outcome
    assert run_control(control_fixture).returncode != 0
    assert [call["operation"] for call in calls(control_fixture)] == ["prepare", "pytest", "check-stack"]


@pytest.mark.parametrize("name,code,operations", [
    ("PREPARE_EXIT", 19, ["prepare"]), ("CUDA_PROBE_EXIT", 20, ["prepare"]),
    ("PYTEST_EXIT", 21, ["prepare", "pytest"]), ("STACK_EXIT", 22, ["prepare", "pytest", "check-stack"]),
    ("CONTROL_EXIT", 23, ["prepare", "pytest", "check-stack", "control"]),
])
def test_failure_preserves_exit_status_and_blocks_later_steps(control_fixture, name, code, operations):
    control_fixture["environment"][name] = str(code)
    assert run_control(control_fixture).returncode == code
    assert [call["operation"] for call in calls(control_fixture)] == operations


@pytest.mark.parametrize("arguments", [[], ["train"], ["authority"] * 5, ["authority"] * 6, ["authority"] * 8])
def test_launcher_requires_exact_seven_arguments(control_fixture, arguments):
    assert run_control(control_fixture, arguments=arguments).returncode == 2
    assert calls(control_fixture, "INTERPRETER_CALLS") == []


def test_headers_reuse_reviewed_gpu_resources_without_submission():
    assert WRAPPER.is_file(), "direct-reference control wrapper is not implemented"
    script = WRAPPER.read_text()
    baseline = (ROOT / "scripts/cluster/diagnose_direct_reference.sh").read_text()
    headers = [line for line in baseline.splitlines() if line.startswith("#SBATCH") and "--job-name" not in line]
    assert [line for line in script.splitlines() if line.startswith("#SBATCH") and "--job-name" not in line] == headers
    assert "#SBATCH --job-name=direct-reference-control" in script
    assert "sbatch " not in script and "ssh " not in script
    assert "check_direct_reference.py" not in script
