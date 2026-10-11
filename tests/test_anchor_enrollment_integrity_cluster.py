import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "scripts/cluster/verify_anchor_enrollment.sh"
ENV_PATH = "/shared/b00090279/conda/envs/refer-or-clear"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def integrity_job(tmp_path):
    source = tmp_path / "source with spaces"
    (source / "scripts").mkdir(parents=True)
    script = source / "scripts/verify_anchor_enrollment_products.py"
    script.write_text('''import argparse, hashlib, json, os
from pathlib import Path
parser = argparse.ArgumentParser()
parser.add_argument('--manifest', type=Path, required=True)
parser.add_argument('--manifest-sha256', required=True)
parser.add_argument('--report', type=Path, required=True)
args = parser.parse_args()
assert hashlib.sha256(args.manifest.read_bytes()).hexdigest() == args.manifest_sha256
assert not args.report.exists()
manifest = json.loads(args.manifest.read_text())
assert not args.report.resolve().is_relative_to(Path(manifest['output_root']).resolve())
call = dict(manifest=str(args.manifest), manifest_sha256=args.manifest_sha256,
            manifest_text=args.manifest.read_text(), script=str(Path(__file__).resolve()),
            environment={key: os.environ.get(key) for key in
                         ('CONDA_PREFIX', 'TMPDIR', 'PYTHONPATH', 'OMP_NUM_THREADS')})
args.report.write_text(json.dumps(call))
if os.environ.get('MUTATE_SNAPSHOT'):
    Path(__file__).chmod(0o644)
    Path(__file__).write_text('changed after invocation')
raise SystemExit(int(os.environ.get('VERIFIER_EXIT', '0')))
''')
    (source / "scripts/prepare_anchor_enrollment.py").write_text("raise RuntimeError('builder forbidden')")
    (source / "test.png").write_bytes(b"synthetic unstageable test placeholder")
    products = tmp_path / "bank products"
    products.mkdir()
    product = products / "median.png"
    product.write_bytes(b"synthetic binary product, not an image")
    manifest = products / "manifest.json"
    manifest.write_text(json.dumps(dict(output_root=str(products), banks={"design": {"enrollment": {
        "products": {"median": {"path": str(product), "sha256": digest(product)}}}}})))
    driver = tmp_path / "python driver.py"
    driver.write_text(f'''import os, runpy, shutil, sys
from pathlib import Path
script, *arguments = sys.argv[1:]
sys.argv = [script, *arguments]
if script == '-':
    sys.prefix = os.environ.get('MOCK_PREFIX', {ENV_PATH!r})
    sys.executable = os.environ.get('MOCK_EXECUTABLE', {ENV_PATH + '/bin/python'!r})
    original_copy = shutil.copyfile
    def copy(source, destination):
        result = original_copy(source, destination)
        if Path(destination).name == os.environ.get('COPY_DAMAGE'):
            Path(destination).write_bytes(b'copy damaged')
        if Path(destination).name == os.environ.get('MUTATE_SOURCE'):
            Path(source).write_bytes(b'source changed during copy')
        return result
    shutil.copyfile = copy
    exec(compile(sys.stdin.read(), '<inline-launcher>', 'exec'))
    if os.environ.get('MUTATE_LIVE') and len(arguments) == 8:
        Path(arguments[1]).write_text('changed live manifest')
        (Path(arguments[0]) / 'scripts/verify_anchor_enrollment_products.py').write_text('raise RuntimeError("live script")')
else:
    runpy.run_path(script, run_name='__main__')
''')
    bash_env = tmp_path / "bash_env"
    bash_env.write_text(f'''source() {{
    [[ "$1" == /opt/miniconda3/etc/profile.d/conda.sh ]] || return 99
}}
conda() {{
    [[ "$1" == activate && "$2" == {ENV_PATH} ]] || return 98
    export CONDA_PREFIX="${{MOCK_CONDA_PREFIX:-$2}}"
}}
function {ENV_PATH}/bin/python {{
    command {shlex.quote(sys.executable)} {shlex.quote(str(driver))} "$@"
}}
''')
    environment = dict(os.environ, BASH_ENV=str(bash_env), SLURM_JOB_ID="1234",
                       SLURMD_NODENAME="cpu01", SLURM_JOB_PARTITION="cpu", SLURM_CPUS_PER_TASK="4",
                       SLURM_SUBMIT_DIR=str(tmp_path), ROC_SCRATCH_ROOT="scratch with spaces",
                       PYTHONDONTWRITEBYTECODE="1", FUNCNEST="12")
    for key in ("MOCK_PREFIX", "MOCK_EXECUTABLE", "MOCK_CONDA_PREFIX", "COPY_DAMAGE", "MUTATE_SOURCE",
                "MUTATE_LIVE", "MUTATE_SNAPSHOT", "VERIFIER_EXIT"):
        environment.pop(key, None)
    return dict(source=source, script=script, manifest=manifest, product=product, environment=environment,
                arguments=[source, manifest, digest(manifest), "outputs with spaces"],
                run=tmp_path / "outputs with spaces/anchor-enrollment-integrity-1234",
                scratch=tmp_path / "scratch with spaces/anchor-enrollment-integrity-1234")


def run_wrapper(job, arguments=None):
    assert WRAPPER.is_file(), "integrity wrapper is missing"
    return subprocess.run(["bash", str(WRAPPER), *map(str, job["arguments"] if arguments is None else arguments)],
                          env=job["environment"], text=True, capture_output=True, timeout=30)


def test_cpu_resources_and_shell_syntax():
    assert WRAPPER.is_file(), "integrity wrapper is missing"
    content = WRAPPER.read_text()
    for directive in ("--account=acc-mialhajri", "--partition=cpu", "--nodes=1", "--ntasks=1",
                      "--cpus-per-task=4", "--mem=8G", "--time=1:00:00"):
        assert f"#SBATCH {directive}\n" in content
    assert "--gres" not in content and "--gpus" not in content
    assert "import torch" not in content and "anchor_job_manifest.py" not in content
    assert subprocess.run(["bash", "-n", str(WRAPPER)], capture_output=True).returncode == 0


def test_frozen_verifier_and_manifest_preserve_authorities_without_staging_products(integrity_job):
    job = integrity_job
    before = {name: digest(job[name]) for name in ("script", "manifest", "product")}
    result = run_wrapper(job)
    assert result.returncode == 0, result.stderr
    call = json.loads((job["run"] / "report.json").read_text())
    assert call["manifest_text"] == job["manifest"].read_text()
    assert call["manifest_sha256"] == before["manifest"]
    assert Path(call["script"]) == job["scratch"] / "code/verify_anchor_enrollment_products.py"
    assert Path(call["manifest"]) == job["scratch"] / "inputs/manifest.json"
    for path in (Path(call["script"]), Path(call["manifest"])):
        assert path.stat().st_mode & 0o777 == 0o444
        assert path.parent.stat().st_mode & 0o777 == 0o555
    assert sorted(path.name for path in job["scratch"].rglob("*") if path.is_file()) == [
        "manifest.json", "verify_anchor_enrollment_products.py"]
    assert all(digest(job[name]) == value for name, value in before.items())
    ledger = json.loads((job["run"] / "launcher_manifest.json").read_text())
    assert ledger["script"]["sha256"] == before["script"]
    assert ledger["authority"]["sha256"] == before["manifest"]
    assert ledger["environment"]["python_prefix"] == ENV_PATH
    assert ledger["environment"]["python_executable"] == ENV_PATH + "/bin/python"
    assert ledger["allocation"] == dict(job_id="1234", node="cpu01", partition="cpu", cpus="4")
    assert ledger["resource_contract"] == dict(partition="cpu", cpus=4, memory="8G", time="1:00:00")
    completion = json.loads((job["run"] / "completion.json").read_text())
    assert completion == dict(returncode=0, script_sha256=before["script"], manifest_sha256=before["manifest"])
    assert call["environment"]["CONDA_PREFIX"] == ENV_PATH
    assert call["environment"]["PYTHONPATH"] == ""


@pytest.mark.parametrize("key,value", [("SLURM_JOB_PARTITION", "gpu"), ("SLURM_JOB_ID", "../1234"),
    ("SLURMD_NODENAME", "LOGIN01"), ("SLURMD_NODENAME", "head01"),
    ("SLURMD_NODENAME", "frontend01"), ("SLURMD_NODENAME", "lgn01"), ("SLURM_CPUS_PER_TASK", "1"),
    ("MOCK_CONDA_PREFIX", "/wrong/env"), ("MOCK_PREFIX", "/wrong/env"),
    ("MOCK_EXECUTABLE", "/usr/bin/python")])
def test_allocation_and_environment_guards(integrity_job, key, value):
    job = integrity_job
    job["environment"][key] = value
    result = run_wrapper(job)
    assert result.returncode != 0
    assert not job["run"].exists() and not job["scratch"].exists()


@pytest.mark.parametrize("key", ["SLURM_JOB_ID", "SLURMD_NODENAME", "SLURM_SUBMIT_DIR", "SLURM_CPUS_PER_TASK"])
def test_missing_allocation_fails_closed(integrity_job, key):
    job = integrity_job
    job["environment"].pop(key)
    assert run_wrapper(job).returncode != 0
    assert not job["run"].exists()


@pytest.mark.parametrize("extra", [[], ["train"]])
def test_exact_four_argument_cli_rejects_phases(integrity_job, extra):
    job = integrity_job
    arguments = job["arguments"][:-1] if not extra else [*job["arguments"], *extra]
    result = run_wrapper(job, arguments)
    assert result.returncode == 2
    assert "Usage:" in result.stderr


@pytest.mark.parametrize("hash_value", ["0" * 64, "invalid"])
def test_authority_hash_mismatch_precedes_staging(integrity_job, hash_value):
    job = integrity_job
    job["arguments"][2] = hash_value
    result = run_wrapper(job)
    assert result.returncode != 0 and "manifest hash" in result.stderr
    assert not job["scratch"].exists() and not job["run"].exists()


@pytest.mark.parametrize("name", ["manifest.json", "verify_anchor_enrollment_products.py"])
@pytest.mark.parametrize("hook", ["COPY_DAMAGE", "MUTATE_SOURCE"])
def test_hash_changes_during_freezing_fail_before_invocation(integrity_job, name, hook):
    job = integrity_job
    job["environment"][hook] = name
    result = run_wrapper(job)
    assert result.returncode != 0 and "hash changed" in result.stderr
    assert not (job["run"] / "report.json").exists()


@pytest.mark.parametrize("destination", ["run", "scratch"])
def test_existing_output_or_scratch_is_preserved(integrity_job, destination):
    job = integrity_job
    job[destination].mkdir(parents=True)
    marker = job[destination] / "old.txt"
    marker.write_text("preserve")
    assert run_wrapper(job).returncode != 0
    assert marker.read_text() == "preserve"


@pytest.mark.parametrize("destination", ["run", "scratch"])
def test_dangling_directory_symlink_is_rejected(integrity_job, destination):
    job = integrity_job
    job[destination].parent.mkdir(parents=True)
    job[destination].symlink_to(job["source"].parent / "missing")
    assert run_wrapper(job).returncode != 0
    assert job[destination].is_symlink()


@pytest.mark.parametrize("protected", ["source", "manifest", "product"])
def test_output_must_not_touch_authoritative_directories(integrity_job, protected):
    job = integrity_job
    directory = job[protected] if protected == "source" else job[protected].parent
    job["arguments"][3] = directory
    result = run_wrapper(job)
    assert result.returncode != 0
    assert not (directory / "anchor-enrollment-integrity-1234").exists()


def test_relative_inputs_resolve_before_execution(integrity_job):
    job = integrity_job
    base = Path(job["environment"]["SLURM_SUBMIT_DIR"])
    job["arguments"][:2] = [job[key].relative_to(base) for key in ("source", "manifest")]
    result = run_wrapper(job)
    assert result.returncode == 0, result.stderr


def test_live_input_mutation_cannot_change_invoked_snapshots(integrity_job):
    job = integrity_job
    original = job["manifest"].read_text()
    job["environment"]["MUTATE_LIVE"] = "1"
    result = run_wrapper(job)
    assert result.returncode == 0, result.stderr
    assert json.loads((job["run"] / "report.json").read_text())["manifest_text"] == original
    assert job["manifest"].read_text() != original


def test_verifier_exit_status_is_preserved_and_recorded(integrity_job):
    job = integrity_job
    job["environment"]["VERIFIER_EXIT"] = "23"
    result = run_wrapper(job)
    assert result.returncode == 23, result.stderr
    assert json.loads((job["run"] / "completion.json").read_text())["returncode"] == 23


def test_post_invocation_snapshot_hash_is_checked(integrity_job):
    job = integrity_job
    job["environment"]["MUTATE_SNAPSHOT"] = "1"
    result = run_wrapper(job)
    assert result.returncode != 0 and "snapshot hash changed" in result.stderr


def test_default_scratch_is_tmp_user_unique_job(integrity_job):
    job = integrity_job
    job["environment"].pop("ROC_SCRATCH_ROOT")
    job["environment"]["USER"] = "integrity-test-user"
    captured = job["source"].parent / "arguments.txt"
    with Path(job["environment"]["BASH_ENV"]).open("a") as stream:
        stream.write(f'''function {ENV_PATH}/bin/python {{
    printf '%s\\n' "$@" > {shlex.quote(str(captured))}
    return 37
}}
''')
    assert run_wrapper(job).returncode == 37
    assert "/tmp/integrity-test-user/anchor-enrollment-integrity-1234" in captured.read_text().splitlines()


def test_absolute_python_mock_bypasses_matching_shell_function(monkeypatch, request):
    monkeypatch.setitem(globals(), "ENV_PATH", str(Path(sys.executable).parent.parent))
    job = request.getfixturevalue("integrity_job")
    probe = job["source"].parent / "portable-probe.py"
    probe.write_text('print("portable mock")\n')
    invocation = f"{shlex.quote(ENV_PATH + '/bin/python')} {shlex.quote(str(probe))}"
    result = subprocess.run(["bash", "-c", invocation], env=job["environment"], text=True,
                            capture_output=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "portable mock"
    bash_env = Path(job["environment"]["BASH_ENV"])
    bash_env.write_text(bash_env.read_text().replace(f"command {shlex.quote(sys.executable)}",
                                                   shlex.quote(sys.executable)))
    recursive = subprocess.run(["bash", "-c", invocation], env=job["environment"], text=True,
                               capture_output=True, timeout=10)
    assert recursive.returncode != 0
    assert "maximum function nesting level" in recursive.stderr
