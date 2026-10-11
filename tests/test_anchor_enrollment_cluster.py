import csv
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
WRAPPER = ROOT / "scripts/cluster/prepare_anchor_enrollment.sh"
ENV_PATH = "/shared/b00090279/conda/envs/refer-or-clear"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def enrollment_job(tmp_path):
    source = tmp_path / "source with spaces"
    (source / "scripts").mkdir(parents=True)
    (source / "src").mkdir()
    shutil.copyfile(ROOT / "scripts/anchor_job_manifest.py", source / "scripts/anchor_job_manifest.py")
    (source / "src/support.py").write_text("version = 'frozen'\n")
    (source / "src/._support.py").write_text("excluded sidecar")
    (source / "data").mkdir()
    (source / "data/test.png").write_bytes(b"not source code")
    (source / "scripts/prepare_anchor_enrollment.py").write_text('''import argparse, json, os
from pathlib import Path
parser = argparse.ArgumentParser()
for flag in ('records', 'split', 'raw-root', 'out'):
    parser.add_argument('--' + flag, type=Path, required=True)
parser.add_argument('--fold', type=int, choices=[0], required=True)
args = parser.parse_args()
if args.out.exists():
    parser.error('output already exists')
args.out.mkdir()
call = dict(arguments={key: str(value) for key, value in vars(args).items()},
            script=str(Path(__file__).resolve()), cwd=str(Path.cwd()),
            records= args.records.read_text(), split=args.split.read_text(),
            environment={key: os.environ.get(key) for key in
                         ('PYTHONPATH', 'CONDA_PREFIX', 'TMPDIR', 'PYTHONHASHSEED',
                          'CUBLAS_WORKSPACE_CONFIG', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                          'NVIDIA_TF32_OVERRIDE', 'TORCH_ALLOW_TF32_CUBLAS_OVERRIDE')})
(args.out / 'builder_call.json').write_text(json.dumps(call))
raise SystemExit(int(os.environ.get('BUILDER_EXIT', '0')))
''')
    for name in ("audit_encoding_headers.py", "check_anchor_pilots.py"):
        (source / "scripts" / name).write_text("raise RuntimeError('unrelated job invoked')\n")
    raw = tmp_path / "raw with spaces"
    raw.mkdir()
    records, split = tmp_path / "records with spaces.csv", tmp_path / "split with spaces.csv"
    rows = [dict(image_id=identifier, dataset="sidtd", path=f"{identifier}.png",
                 annotation=f"{identifier}.json", label=label, component_id=identifier,
                 design_id="sidtd/design", identity_id=f"sidtd/{identifier}", source_id="", donor_id="",
                 width=2, height=2)
            for identifier, label in [("genuine", 0), ("forged", 1), ("validation", 0), ("test", 0)]]
    with records.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with split.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["image_id", "role", "component_id", "design_id", "label",
                                                   "protocol", "fold", "ref_group"])
        writer.writeheader()
        writer.writerows(dict(image_id=row["image_id"], role=role, component_id=row["component_id"],
                              design_id=row["design_id"], label=row["label"], protocol="A", fold=0, ref_group="it0")
                         for row, role in zip(rows, ["train", "train", "val", "test"]))
    (raw / "genuine.png").write_bytes(b"synthetic genuine placeholder")
    (raw / "genuine.json").write_text('{"regions": []}')
    driver = tmp_path / "python driver.py"
    driver.write_text('''import os, runpy, shutil, sys
from pathlib import Path
script, *arguments = sys.argv[1:]
if Path(script).name == 'anchor_job_manifest.py' and arguments[0] == 'prepare':
    damage = os.environ.get('COPY_DAMAGE')
    original_copy = shutil.copyfile
    def damaged_copy(source, destination):
        result = original_copy(source, destination)
        if damage and Path(destination).name == damage:
            Path(destination).write_bytes(b'corrupted in transfer')
        return result
    shutil.copyfile = damaged_copy
sys.argv = [script, *arguments]
runpy.run_path(script, run_name='__main__')
if Path(script).name == 'anchor_job_manifest.py' and arguments[0] == 'prepare':
    if os.environ.get('MUTATE_LIVE_INPUTS') == '1':
        for flag in ('--records', '--split'):
            Path(arguments[arguments.index(flag) + 1]).write_text('changed after freezing')
        source = Path(arguments[arguments.index('--source-dir') + 1])
        (source / 'scripts/prepare_anchor_enrollment.py').write_text("raise RuntimeError('live builder invoked')")
    if os.environ.get('PRECREATE_PRODUCT') == '1':
        product = Path(arguments[arguments.index('--run-dir') + 1]) / 'enrollment'
        product.mkdir()
        (product / 'previous.txt').write_text('preserve previous product')
''')
    environment_file = tmp_path / "bash_env"
    environment_file.write_text(f'''source() {{
    [[ "$1" == /opt/miniconda3/etc/profile.d/conda.sh ]] || return 99
}}
conda() {{
    [[ "$1" == activate && "$2" == {ENV_PATH} ]] || return 98
    export CONDA_PREFIX="$2"
}}
function {ENV_PATH}/bin/python {{
    command {shlex.quote(sys.executable)} {shlex.quote(str(driver))} "$@"
}}
''')
    environment = dict(os.environ, BASH_ENV=str(environment_file), SLURM_JOB_ID="1234",
                       SLURMD_NODENAME="cpu01", SLURM_JOB_PARTITION="cpu", SLURM_CPUS_PER_TASK="4",
                       SLURM_SUBMIT_DIR=str(tmp_path), ROC_SCRATCH_ROOT="scratch with spaces",
                       PYTHONDONTWRITEBYTECODE="1")
    for key in ("COPY_DAMAGE", "BUILDER_EXIT", "MUTATE_LIVE_INPUTS", "PRECREATE_PRODUCT"):
        environment.pop(key, None)
    return dict(source=source, raw=raw, records=records, split=split, environment=environment,
                arguments=[source, records, split, raw, "outputs with spaces"],
                run=tmp_path / "outputs with spaces/anchor-enrollment-fold0-1234",
                stage=tmp_path / "scratch with spaces/anchor-enrollment-fold0-1234")


def run_wrapper(job, arguments=None):
    assert WRAPPER.is_file(), "fold-0 enrollment wrapper is missing"
    return subprocess.run(["bash", str(WRAPPER), *map(str, job["arguments"] if arguments is None else arguments)],
                          env=job["environment"], text=True, capture_output=True, timeout=30)


def test_default_scratch_uses_tmp_user_and_unique_job_directory(enrollment_job):
    job = enrollment_job
    job["environment"].pop("ROC_SCRATCH_ROOT")
    job["environment"]["USER"] = "enrollment-test-user"
    captured = job["source"].parent / "prepare_arguments.txt"
    with Path(job["environment"]["BASH_ENV"]).open("a") as stream:
        stream.write(f'''function {ENV_PATH}/bin/python {{
    printf '%s\\n' "$@" > {shlex.quote(str(captured))}
    return 37
}}
''')
    result = run_wrapper(job)
    assert result.returncode == 37, result.stderr
    arguments = captured.read_text().splitlines()
    assert arguments[arguments.index("--stage-dir") + 1] == "/tmp/enrollment-test-user/anchor-enrollment-fold0-1234"
    assert not job["run"].exists()


def test_enrollment_runs_frozen_fold0_builder_without_staging_pixels_or_launching_other_jobs(enrollment_job):
    job = enrollment_job
    original_records, original_split = job["records"].read_text(), job["split"].read_text()
    result = run_wrapper(job)
    assert result.returncode == 0, result.stderr
    call = json.loads((job["run"] / "enrollment/builder_call.json").read_text())
    stage = job["stage"]
    assert call["arguments"] == dict(records=str(stage / "inputs/records.csv"),
                                     split=str(stage / "inputs/split.csv"), fold="0",
                                     raw_root=str(job["raw"]), out=str(job["run"] / "enrollment"))
    assert call["script"] == str(stage / "code/scripts/prepare_anchor_enrollment.py")
    assert call["cwd"] == str(stage / "code")
    assert call["records"] == original_records
    assert call["split"] == original_split
    assert list((stage / "raw").iterdir()) == []
    assert list((stage / "legacy").iterdir()) == []
    assert not (stage / "code/data").exists()
    assert not (stage / "code/src/._support.py").exists()
    assert not (job["run"] / "audit").exists()
    assert not (job["run"] / "pilot").exists()
    assert call["environment"] == dict(PYTHONPATH=str(stage / "code/src"), CONDA_PREFIX=ENV_PATH,
                                       TMPDIR=str(stage / "tmp"), PYTHONHASHSEED="42",
                                       CUBLAS_WORKSPACE_CONFIG=":4096:8", OMP_NUM_THREADS="4",
                                       MKL_NUM_THREADS="4", NVIDIA_TF32_OVERRIDE="0",
                                       TORCH_ALLOW_TF32_CUBLAS_OVERRIDE="0")
    assert (stage / "tmp").is_dir()
    manifest = json.loads((job["run"] / "input_manifest.json").read_text())
    assert manifest["schema_version"] == "anchor_job_manifest_v1"
    assert manifest["phase"] == "audit"
    assert manifest["data_scope"] == "metadata_only"
    assert manifest["data_files"] == []
    assert manifest["review"] is None
    assert manifest["allocation"]["job_id"] == "1234"
    assert manifest["allocation"]["partition"] == "cpu"
    assert manifest["provenance"]["records_sha256"] == digest(job["records"])
    assert manifest["provenance"]["split_sha256"] == digest(job["split"])
    for entry in manifest["source_files"]:
        frozen = stage / "code" / entry["path"]
        assert digest(frozen) == entry["sha256"]
        assert frozen.stat().st_mode & 0o222 == 0
    for root in (stage / "inputs", job["run"] / "inputs"):
        for name in ("records.csv", "split.csv"):
            assert (root / name).stat().st_mode & 0o222 == 0


@pytest.mark.parametrize("guard", ["job", "node", "submit", "login", "head", "frontend", "lgn",
                                   "partition", "job_id", "few_arguments", "extra_arguments"])
def test_allocation_and_cli_guards_fail_before_preparation(enrollment_job, guard):
    job = enrollment_job
    arguments = job["arguments"]
    if guard in {"job", "node", "submit"}:
        job["environment"].pop({"job": "SLURM_JOB_ID", "node": "SLURMD_NODENAME",
                                "submit": "SLURM_SUBMIT_DIR"}[guard])
    elif guard in {"login", "head", "frontend", "lgn"}:
        job["environment"]["SLURMD_NODENAME"] = f"{guard.upper()}01"
    elif guard == "partition":
        job["environment"]["SLURM_JOB_PARTITION"] = "gpu"
    elif guard == "job_id":
        job["environment"]["SLURM_JOB_ID"] = "../invalid"
    elif guard == "few_arguments":
        arguments = arguments[:-1]
    else:
        arguments = [*arguments, "specimens"]
    result = run_wrapper(job, arguments)
    assert result.returncode != 0
    assert not job["stage"].exists()
    assert not job["run"].exists()


@pytest.mark.parametrize("existing", ["run", "stage"])
def test_existing_job_directories_are_not_reused(enrollment_job, existing):
    job = enrollment_job
    job[existing].mkdir(parents=True)
    marker = job[existing] / "previous.txt"
    marker.write_text("preserve previous job")
    result = run_wrapper(job)
    assert result.returncode != 0
    assert "new scratch and output directories required" in result.stderr
    assert marker.read_text() == "preserve previous job"
    assert not (job["run"] / "enrollment").exists()


@pytest.mark.parametrize("damage", ["support.py", "records.csv", "split.csv"])
def test_source_and_metadata_copy_hash_failures_stop_before_builder(enrollment_job, damage):
    job = enrollment_job
    job["environment"]["COPY_DAMAGE"] = damage
    result = run_wrapper(job)
    assert result.returncode != 0
    assert "hash changed during staging" in result.stderr
    assert not (job["run"] / "enrollment").exists()


def test_builder_failure_propagates_without_followup_jobs(enrollment_job):
    enrollment_job["environment"]["BUILDER_EXIT"] = "23"
    result = run_wrapper(enrollment_job)
    assert result.returncode == 23, result.stderr
    assert not (enrollment_job["run"] / "audit").exists()
    assert not (enrollment_job["run"] / "pilot").exists()


def test_builder_consumes_frozen_source_and_official_metadata_even_if_live_inputs_change(enrollment_job):
    job = enrollment_job
    records, split = job["records"].read_text(), job["split"].read_text()
    job["environment"]["MUTATE_LIVE_INPUTS"] = "1"
    result = run_wrapper(job)
    assert result.returncode == 0, result.stderr
    call = json.loads((job["run"] / "enrollment/builder_call.json").read_text())
    assert job["records"].read_text() != records
    assert job["split"].read_text() != split
    assert call["records"] == records
    assert call["split"] == split


def test_existing_builder_output_is_not_removed_or_reused(enrollment_job):
    job = enrollment_job
    job["environment"]["PRECREATE_PRODUCT"] = "1"
    result = run_wrapper(job)
    assert result.returncode != 0
    assert "output already exists" in result.stderr
    assert (job["run"] / "enrollment/previous.txt").read_text() == "preserve previous product"
    assert not (job["run"] / "enrollment/builder_call.json").exists()


def test_different_allocations_get_separate_versioned_outputs(enrollment_job):
    job = enrollment_job
    first = run_wrapper(job)
    assert first.returncode == 0, first.stderr
    original = (job["run"] / "enrollment/builder_call.json").read_bytes()
    job["environment"]["SLURM_JOB_ID"] = "1235"
    second = run_wrapper(job)
    assert second.returncode == 0, second.stderr
    assert (job["run"].with_name("anchor-enrollment-fold0-1235") / "enrollment/builder_call.json").is_file()
    assert (job["run"] / "enrollment/builder_call.json").read_bytes() == original
