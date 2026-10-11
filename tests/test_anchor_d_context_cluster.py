import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "scripts/cluster/check_anchor_d_context.sh"
ENV_PATH = "/shared/b00090279/conda/envs/refer-or-clear"
SOURCE = "/shared/b00090279/refer-or-clear/results/source/anchor-0959335bd9a6a22510f6706c49c62d84560cd524/code"
SOURCE_SHA = "2f8a833e5cb3450fdccd8c58dba8f07f58af330747a69cca7363d1a21528da9a"
SCRIPT_SHA = "4253fc893e41d8a31cc95c35e5b170f1553b102eb53f2a7134f3faa8da7114b4"
CHECKPOINT_SHA = "c78d0b4a06b5dbde434d77f34ffa14ea437dbde4e1d3eafbd2d8aadfff72c47f"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def context_job(tmp_path):
    tooling = tmp_path / "tooling with spaces"
    for folder, name in (("scripts", "check_anchor_d_context.py"), ("tests", "test_anchor_d_context.py")):
        (tooling / folder).mkdir(parents=True)
        shutil.copyfile(ROOT / folder / name, tooling / folder / name)
    (tooling / "src").mkdir()
    (tooling / "src/forbidden.py").write_text("raise RuntimeError('wrong scientific source')\n")
    inputs = tmp_path / "authorities with spaces"
    inputs.mkdir()
    paths = {}
    for name, contents in (("config", "track: D\n"), ("records", "image_id,label\nprobe,0\n"),
                           ("split", "image_id,role\nprobe,train\n"), ("weights", "synthetic offline weights"),
                           ("checkpoint", "synthetic pinned D checkpoint"),
                           ("predictions", "image_id,label,attack_logit\nprobe,0,0.1\n")):
        paths[name] = inputs / name
        paths[name].write_text(contents)
    provenance = {f"{name}_sha256": digest(paths[name]) for name in ("config", "records", "split", "weights")}
    provenance["source_sha256"] = hashlib.sha256(b"[]").hexdigest()
    parent = dict(schema_version="anchor_job_manifest_v1", phase="diagnostic", track="D",
                  data_scope="train_val_only", allocation={"job_id": "37963"}, source_files=[],
                  paths={"source_dir": SOURCE}, provenance=provenance,
                  runtime_identities={"source_sha256": SOURCE_SHA})
    train = dict(schema_version="anchor_pilot_v1", phase="overfit", track="D", opened_roles=["train"],
                 no_test_open=True, n_images={"train": 16}, selected_checkpoint_sha256=CHECKPOINT_SHA)
    for name, payload in (("parent_manifest", parent), ("train_json", train)):
        paths[name] = inputs / f"{name}.json"
        paths[name].write_text(json.dumps(payload))
    raw, legacy = tmp_path / "raw", tmp_path / "legacy"
    raw.mkdir()
    legacy.mkdir()
    driver = tmp_path / "python driver.py"
    driver.write_text(f'''import hashlib, importlib.util, json, os, shutil, sys
from pathlib import Path
sys.prefix = os.environ.get('MOCK_PREFIX', {ENV_PATH!r})
sys.executable = os.environ.get('MOCK_EXECUTABLE', {str(Path(ENV_PATH) / 'bin/python')!r})
original_sha256 = hashlib.sha256
class SyntheticCheckpointHash:
    def __init__(self, data=b''):
        self.checksum = original_sha256(data)
        self.contents = bytearray(data)
    def update(self, data):
        self.checksum.update(data)
        self.contents.extend(data)
    def hexdigest(self):
        if self.contents == b'synthetic offline weights' and os.environ.get('MUTATE_TOOLING'):
            Path(os.environ['MUTATE_TOOLING']).write_text("raise RuntimeError('script changed after initial verification')")
        if self.contents == b'synthetic pinned D checkpoint':
            return {CHECKPOINT_SHA!r}
        return self.checksum.hexdigest()
hashlib.sha256 = SyntheticCheckpointHash
original_copy = shutil.copyfile
def copyfile(source, destination, *args, **kwargs):
    result = original_copy(source, destination, *args, **kwargs)
    if Path(destination).name == os.environ.get('COPY_DAMAGE'):
        Path(destination).write_bytes(b'changed during transfer')
    return result
shutil.copyfile = copyfile
if sys.argv[1] == '-':
    sys.argv = sys.argv[1:]
    exec(compile(sys.stdin.read(), '<launcher-preparation>', 'exec'))
elif sys.argv[1] == '-c':
    assert sys.argv[2].startswith('import torch;'), 'unexpected executable phase'
    if os.environ.get('MOCK_GPU_AVAILABLE', '1') != '1':
        raise SystemExit(19)
    print('synthetic allocated GPU')
else:
    script = Path(sys.argv[1])
    assert script.name == 'check_anchor_d_context.py', 'unrelated executable invoked'
    assert original_sha256(script.read_bytes()).hexdigest() == {SCRIPT_SHA!r}
    spec = importlib.util.spec_from_file_location('reviewed_context_parser', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    args = module.parse_args(sys.argv[2:])
    run = args.run_dir
    assert not run.exists(), 'diagnostic output must be new'
    run.mkdir()
    (run / 'invocation.json').write_text(json.dumps(dict(
        arguments={{key: str(value) for key, value in vars(args).items()}}, script=str(script.resolve()),
        cwd=str(Path.cwd()), environment={{key: os.environ.get(key) for key in
            ('PYTHONPATH', 'TMPDIR', 'CONDA_PREFIX', 'CUBLAS_WORKSPACE_CONFIG', 'PYTHONHASHSEED',
             'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NVIDIA_TF32_OVERRIDE', 'TORCH_ALLOW_TF32_CUBLAS_OVERRIDE')}})))
    raise SystemExit(int(os.environ.get('RUNNER_EXIT', '0')))
''')
    bash_env = tmp_path / "bash_env"
    bash_env.write_text(f'''source() {{
    [[ "$1" == /opt/miniconda/etc/profile.d/conda.sh ]] || return 99
}}
conda() {{
    [[ "$1" == activate && "$2" == {ENV_PATH} ]] || return 98
    export CONDA_PREFIX="${{MOCK_CONDA_PREFIX:-$2}}"
}}
function {ENV_PATH}/bin/python {{
    command {shlex.quote(sys.executable)} {shlex.quote(str(driver))} "$@"
}}
''')
    environment = dict(os.environ, BASH_ENV=str(bash_env), SLURM_JOB_ID="1234", SLURMD_NODENAME="gpu01",
                       SLURM_JOB_PARTITION="gpu", SLURM_CPUS_PER_TASK="4", SLURM_SUBMIT_DIR=str(tmp_path),
                       ROC_SCRATCH_ROOT="scratch with spaces", PYTHONDONTWRITEBYTECODE="1")
    for key in ("MOCK_PREFIX", "MOCK_EXECUTABLE", "MOCK_CONDA_PREFIX", "COPY_DAMAGE", "RUNNER_EXIT",
                "MOCK_GPU_AVAILABLE", "MUTATE_TOOLING"):
        environment.pop(key, None)
    arguments = [tooling, paths["parent_manifest"], digest(paths["parent_manifest"]), paths["train_json"],
                 digest(paths["train_json"]), paths["config"], paths["records"], paths["split"], paths["weights"],
                 paths["checkpoint"], paths["predictions"], digest(paths["predictions"]), raw, legacy, "outputs"]
    return dict(tooling=tooling, paths=paths, raw=raw, legacy=legacy, environment=environment, arguments=arguments,
                run=tmp_path / "outputs/anchor-d-context-1234", stage=tmp_path / "scratch with spaces/anchor-d-context-1234")


def run_wrapper(job, arguments=None):
    assert WRAPPER.is_file(), "D-context cluster wrapper is missing"
    return subprocess.run(["bash", str(WRAPPER), *map(str, job["arguments"] if arguments is None else arguments)],
                          env=job["environment"], text=True, capture_output=True, timeout=30)


def test_only_reviewed_sidecar_executes_from_frozen_tooling_with_pinned_scientific_source(context_job):
    job = context_job
    originals = {name: path.read_bytes() for name, path in job["paths"].items()}
    result = run_wrapper(job)
    assert result.returncode == 0, result.stderr
    run = job["run"]
    invocation = json.loads((run / "context/invocation.json").read_text())
    frozen = run / "inputs"
    assert invocation["arguments"] == dict(source_dir=SOURCE, parent_manifest=str(frozen / "parent_manifest.json"),
                                          parent_manifest_sha256=job["arguments"][2], train_json=str(frozen / "train.json"),
                                          train_json_sha256=job["arguments"][4], config=str(frozen / "config.yaml"),
                                          records=str(frozen / "records.csv"), split=str(frozen / "split.csv"),
                                          weights=str(job["paths"]["weights"]), checkpoint=str(job["paths"]["checkpoint"]),
                                          predictions=str(frozen / "predictions.csv"), predictions_sha256=job["arguments"][11],
                                          raw_root=str(job["raw"]), legacy_root=str(job["legacy"]), run_dir=str(run / "context"),
                                          device="cuda", workers="0")
    assert invocation["script"] == str(run / "tooling/scripts/check_anchor_d_context.py")
    assert invocation["cwd"] == str(run / "tooling")
    assert invocation["environment"] == dict(PYTHONPATH=f"{SOURCE}/src", TMPDIR=str(job["stage"] / "tmp"),
                                               CONDA_PREFIX=ENV_PATH, CUBLAS_WORKSPACE_CONFIG=":4096:8",
                                               PYTHONHASHSEED="42", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4",
                                               NVIDIA_TF32_OVERRIDE="0", TORCH_ALLOW_TF32_CUBLAS_OVERRIDE="0")
    assert (job["stage"] / "tmp").is_dir()
    assert not (run / "tooling/src").exists()
    assert list(job["raw"].iterdir()) == list(job["legacy"].iterdir()) == []
    assert all(path.read_bytes() == originals[name] for name, path in job["paths"].items())
    ledger = json.loads((run / "launcher_manifest.json").read_text())
    assert ledger["schema_version"] == "anchor_d_context_launcher_v1"
    assert ledger["diagnostic_only"] is True
    assert ledger["scientific_gate_passed"] is None
    assert ledger["scientific_source"] == dict(path=SOURCE, runtime_sha256=SOURCE_SHA,
                                               inventory_sha256=hashlib.sha256(b"[]").hexdigest())
    assert ledger["diagnostic_script_sha256"] == SCRIPT_SHA
    assert ledger["launcher_sha256"] == digest(WRAPPER)
    assert ledger["allocation"]["job_id"] == "1234"
    assert ledger["environment"]["conda_prefix"] == ENV_PATH
    assert ledger["environment"]["python_prefix"] == ENV_PATH
    assert ledger["environment"]["python_executable"] == f"{ENV_PATH}/bin/python"
    assert ledger["inputs"]["checkpoint"]["sha256"] == CHECKPOINT_SHA
    for name in ("parent_manifest", "train_json", "predictions", "config", "records", "split", "weights"):
        assert ledger["inputs"][name]["sha256"] == digest(job["paths"][name])
    for directory in (run / "inputs", run / "tooling"):
        for path in directory.rglob("*"):
            assert path.stat().st_mode & 0o222 == 0


@pytest.mark.parametrize("guard", ["job", "node", "submit", "cpus", "login", "head", "frontend", "lgn",
                                   "partition", "job_id", "few_arguments", "extra_phase"])
def test_cli_and_allocation_guards_stop_before_preparation(context_job, guard):
    job = context_job
    arguments = job["arguments"]
    if guard in {"job", "node", "submit", "cpus"}:
        job["environment"].pop({"job": "SLURM_JOB_ID", "node": "SLURMD_NODENAME", "submit": "SLURM_SUBMIT_DIR",
                                "cpus": "SLURM_CPUS_PER_TASK"}[guard])
    elif guard in {"login", "head", "frontend", "lgn"}:
        job["environment"]["SLURMD_NODENAME"] = f"{guard.upper()}01"
    elif guard == "partition":
        job["environment"]["SLURM_JOB_PARTITION"] = "cpu"
    elif guard == "job_id":
        job["environment"]["SLURM_JOB_ID"] = "../invalid"
    elif guard == "few_arguments":
        arguments = arguments[:-1]
    else:
        arguments = ["train", *arguments]
    result = run_wrapper(job, arguments)
    assert result.returncode != 0
    assert not job["stage"].exists()
    assert not job["run"].exists()


@pytest.mark.parametrize("key", ["MOCK_PREFIX", "MOCK_EXECUTABLE", "MOCK_CONDA_PREFIX"])
def test_wrong_environment_is_rejected_before_any_staging(context_job, key):
    job = context_job
    job["environment"][key] = "/wrong/environment"
    result = run_wrapper(job)
    assert result.returncode != 0
    assert "environment" in result.stderr.lower()
    assert not job["stage"].exists()
    assert not job["run"].exists()


@pytest.mark.parametrize("damage", ["script", "parent_hash", "train_hash", "predictions_hash", "config", "records",
                                    "split", "weights", "checkpoint", "parent_phase", "train_phase", "source"])
def test_reviewed_tooling_and_authority_failures_stop_before_runner(context_job, damage):
    job = context_job
    if damage == "script":
        (job["tooling"] / "scripts/check_anchor_d_context.py").write_text("raise RuntimeError('unreviewed script')")
    elif damage in {"parent_hash", "train_hash", "predictions_hash"}:
        job["arguments"][{"parent_hash": 2, "train_hash": 4, "predictions_hash": 11}[damage]] = "0" * 64
    elif damage in {"config", "records", "split", "weights", "checkpoint"}:
        job["paths"][damage].write_bytes(b"changed input")
    else:
        name = "train_json" if damage == "train_phase" else "parent_manifest"
        path = job["paths"][name]
        payload = json.loads(path.read_text())
        if damage == "source":
            payload["paths"]["source_dir"] = "/changing/main/tree"
        else:
            payload["phase"] = "train"
        path.write_text(json.dumps(payload))
        job["arguments"][4 if name == "train_json" else 2] = digest(path)
    result = run_wrapper(job)
    assert result.returncode != 0
    assert not job["run"].exists()
    assert not job["stage"].exists()


@pytest.mark.parametrize("directory", ["run", "stage"])
def test_existing_job_directories_remain_untouched(context_job, directory):
    job = context_job
    job[directory].mkdir(parents=True)
    marker = job[directory] / "previous.txt"
    marker.write_text("preserve historical output")
    result = run_wrapper(job)
    assert result.returncode != 0
    assert "new" in result.stderr.lower()
    assert marker.read_text() == "preserve historical output"
    assert not (job["run"] / "context").exists()


@pytest.mark.parametrize("name", ["check_anchor_d_context.py", "records.csv", "parent_manifest.json"])
def test_changed_copies_fail_closed_before_diagnostic(context_job, name):
    context_job["environment"]["COPY_DAMAGE"] = name
    result = run_wrapper(context_job)
    assert result.returncode != 0
    assert "hash" in result.stderr.lower()
    assert not (context_job["run"] / "context").exists()


def test_no_available_gpu_stops_before_runner(context_job):
    context_job["environment"]["MOCK_GPU_AVAILABLE"] = "0"
    result = run_wrapper(context_job)
    assert result.returncode == 19, result.stderr
    assert not (context_job["run"] / "context").exists()


def test_sidecar_failure_propagates_without_training_or_followups(context_job):
    context_job["environment"]["RUNNER_EXIT"] = "23"
    result = run_wrapper(context_job)
    assert result.returncode == 23, result.stderr
    assert not (context_job["run"] / "pilot").exists()


def test_relative_relocation_bindings_remain_absolute_after_changing_to_frozen_tooling(context_job):
    job = context_job
    submit = Path(job["environment"]["SLURM_SUBMIT_DIR"])
    for index in (0, 1, 3, 5, 6, 7, 8, 9, 10, 12, 13):
        job["arguments"][index] = Path(job["arguments"][index]).relative_to(submit)
    result = run_wrapper(job)
    assert result.returncode == 0, result.stderr
    invocation = json.loads((job["run"] / "context/invocation.json").read_text())["arguments"]
    assert invocation["weights"] == str(job["paths"]["weights"])
    assert invocation["checkpoint"] == str(job["paths"]["checkpoint"])
    assert invocation["raw_root"] == str(job["raw"])
    assert invocation["legacy_root"] == str(job["legacy"])


def test_script_mutation_during_authority_hashing_cannot_replace_reviewed_execution(context_job):
    job = context_job
    job["environment"]["MUTATE_TOOLING"] = str(job["tooling"] / "scripts/check_anchor_d_context.py")
    result = run_wrapper(job)
    assert result.returncode != 0
    assert "hash" in result.stderr.lower()
    assert not (job["run"] / "context").exists()


def test_gpu_scratch_default_is_user_and_job_scoped_without_touching_host_scratch(context_job):
    job = context_job
    job["environment"].pop("ROC_SCRATCH_ROOT")
    job["environment"]["USER"] = "context-test-user"
    captured = job["tooling"].parent / "preparation_arguments.txt"
    with Path(job["environment"]["BASH_ENV"]).open("a") as stream:
        stream.write(f'''function {ENV_PATH}/bin/python {{
    printf '%s\\n' "$@" > {shlex.quote(str(captured))}
    return 37
}}
''')
    result = run_wrapper(job)
    assert result.returncode == 37, result.stderr
    assert captured.read_text().splitlines()[-4] == "/scratch/context-test-user/anchor-d-context-1234"
    assert not job["run"].exists()


def test_scheduler_request_has_exact_short_gpu_resource_contract():
    directives = [line for line in WRAPPER.read_text().splitlines() if line.startswith("#SBATCH")]
    for request in ("--account=acc-mialhajri", "--partition=gpu", "--nodes=1", "--ntasks=1", "--gres=gpu:1",
                    "--cpus-per-task=4", "--mem=8G", "--time=1:00:00"):
        assert f"#SBATCH {request}" in directives
