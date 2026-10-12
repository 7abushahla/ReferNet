import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys

import pytest

from test_anchor_enrollment_cluster import enrollment_job


ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "scripts/cluster/prepare_protocol_b.sh"
ENV_PATH = "/shared/b00090279/conda/envs/refer-or-clear"


@pytest.fixture
def job(enrollment_job):
    job = enrollment_job
    builder = job["source"] / "scripts/prepare_protocol_b.py"
    builder.write_text('''import argparse, json, os
from pathlib import Path
parser = argparse.ArgumentParser()
for flag in ('records', 'records-sha256', 'split', 'raw-root', 'output-root'):
    parser.add_argument('--' + flag, required=True)
parser.add_argument('--group', type=int, choices=range(10), required=True)
parser.add_argument('--build', action='store_true')
args = parser.parse_args()
assert args.build, 'actual bank construction flag required'
output = Path(args.output_root)
assert not output.exists(), 'existing output must not be overwritten'
output.mkdir()
call = dict(arguments=vars(args), script=str(Path(__file__).resolve()), cwd=str(Path.cwd()),
            records=Path(args.records).read_text(), split=Path(args.split).read_text(),
            environment={key: os.environ.get(key) for key in
                         ('CONDA_PREFIX', 'PYTHONPATH', 'TMPDIR', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                          'PYTHONHASHSEED', 'CUDA_VISIBLE_DEVICES')})
(output / 'builder_call.json').write_text(json.dumps(call))
raise SystemExit(int(os.environ.get('BUILDER_EXIT', '0')))
''')
    checksum = hashlib.sha256(job["records"].read_bytes()).hexdigest()
    job["arguments"] = [job["source"], job["records"], checksum, job["split"], job["raw"],
                        "outputs with spaces", "0"]
    job["run"] = job["source"].parent / "outputs with spaces/protocol-b-group0-1234"
    job["stage"] = job["source"].parent / "scratch with spaces/protocol-b-group0-1234"
    environment_file = Path(job["environment"]["BASH_ENV"])
    driver = job["source"].parent / "python driver.py"
    with environment_file.open("a") as stream:
        stream.write(f'''function {ENV_PATH}/bin/python {{
    if [[ "$1" == -c ]]; then
        command {shlex.quote(sys.executable)} "$@"
    else
        command {shlex.quote(sys.executable)} {shlex.quote(str(driver))} "$@"
    fi
}}
''')
    for key in ("SLURM_JOB_GPUS", "SLURM_STEP_GPUS", "SLURM_GPUS"):
        job["environment"].pop(key, None)
    return job


def run(job, arguments=None):
    assert WRAPPER.is_file(), "Protocol B CPU preparation wrapper is missing"
    return subprocess.run(["bash", str(WRAPPER), *map(str, job["arguments"] if arguments is None else arguments)],
                          env=job["environment"], text=True, capture_output=True, timeout=30)


def test_cpu_header_and_bootstrap_are_the_reviewed_pattern():
    assert WRAPPER.is_file(), "Protocol B CPU preparation wrapper is missing"
    content = WRAPPER.read_text()
    assert "#SBATCH --partition=cpu" in content
    assert "#SBATCH --cpus-per-task=4" in content and "#SBATCH --mem=8G" in content
    assert "#SBATCH --nodes=1" in content and "#SBATCH --ntasks=1" in content
    assert "#SBATCH --gres" not in content
    assert "source /opt/miniconda3/etc/profile.d/conda.sh" in content
    assert ENV_PATH in content
    assert "sbatch " not in content and "ssh " not in content and "git " not in content


def test_build_dispatch_is_frozen_metadata_only_staging_and_group_bound(job):
    result = run(job)
    assert result.returncode == 0, result.stderr
    call = json.loads((job["run"] / "preparation/builder_call.json").read_text())
    assert call["arguments"] == dict(records=str(job["stage"] / "inputs/records.csv"),
                                    records_sha256=job["arguments"][2], split=str(job["stage"] / "inputs/split.csv"),
                                    raw_root=str(job["raw"]), output_root=str(job["run"] / "preparation"),
                                    group=0, build=True)
    assert call["script"] == str(job["stage"] / "code/scripts/prepare_protocol_b.py")
    assert call["cwd"] == str(job["stage"] / "code")
    assert call["records"] == job["records"].read_text() and call["split"] == job["split"].read_text()
    assert call["environment"] == dict(CONDA_PREFIX=ENV_PATH, PYTHONPATH=str(job["stage"] / "code/src"),
                                       TMPDIR=str(job["stage"] / "tmp"), OMP_NUM_THREADS="4", MKL_NUM_THREADS="4",
                                       PYTHONHASHSEED="42", CUDA_VISIBLE_DEVICES="")
    manifest = json.loads((job["run"] / "input_manifest.json").read_text())
    assert manifest["phase"] == "audit" and manifest["data_scope"] == "metadata_only"
    assert manifest["data_files"] == [] and manifest["provenance"]["records_sha256"] == job["arguments"][2]
    for entry in manifest["source_files"]:
        frozen = job["stage"] / "code" / entry["path"]
        assert hashlib.sha256(frozen.read_bytes()).hexdigest() == entry["sha256"]
        assert frozen.stat().st_mode & 0o222 == 0
    assert list((job["stage"] / "raw").iterdir()) == []
    assert list((job["stage"] / "legacy").iterdir()) == []


@pytest.mark.parametrize("group", [1, 9])
def test_group_propagates_to_unique_output_and_scratch(job, group):
    arguments = [*job["arguments"][:-1], str(group)]
    result = run(job, arguments)
    assert result.returncode == 0, result.stderr
    call = json.loads((job["run"].with_name(f"protocol-b-group{group}-1234") / "preparation/builder_call.json").read_text())
    assert call["arguments"]["group"] == group
    assert call["script"] == str(job["stage"].with_name(f"protocol-b-group{group}-1234") / "code/scripts/prepare_protocol_b.py")


@pytest.mark.parametrize("key,value", [("SLURM_JOB_ID", ""), ("SLURM_JOB_ID", "123/4"),
                                     ("SLURMD_NODENAME", ""), ("SLURMD_NODENAME", "CPU-login01"),
                                     ("SLURMD_NODENAME", "Head01"), ("SLURMD_NODENAME", "frontend01"),
                                     ("SLURMD_NODENAME", "lgn01"), ("SLURM_JOB_PARTITION", "gpu"),
                                     ("SLURM_JOB_PARTITION", "hpc"), ("SLURM_CPUS_PER_TASK", "3"),
                                     ("SLURM_CPUS_PER_TASK", "four"), ("SLURM_SUBMIT_DIR", ""),
                                     ("SLURM_JOB_GPUS", "0"), ("SLURM_STEP_GPUS", "0"),
                                     ("SLURM_GPUS", "1")])
def test_invalid_allocation_fails_before_staging(job, key, value):
    job["environment"][key] = value
    result = run(job)
    assert result.returncode != 0
    assert not job["stage"].exists() and not job["run"].exists()


@pytest.mark.parametrize("group", ["-1", "10", "true", "0.0", ""])
def test_invalid_group_fails_before_staging(job, group):
    result = run(job, [*job["arguments"][:-1], group])
    assert result.returncode != 0
    assert not job["run"].exists()


@pytest.mark.parametrize("checksum", ["invalid", "a" * 64])
def test_records_hash_is_checked_before_staging(job, checksum):
    job["arguments"][2] = checksum
    result = run(job)
    assert result.returncode != 0 and not job["stage"].exists() and not job["run"].exists()


def test_live_metadata_mutation_after_freeze_cannot_change_builder_inputs(job):
    original_records = job["records"].read_text()
    job["environment"]["MUTATE_LIVE_INPUTS"] = "1"
    result = run(job)
    assert result.returncode == 0, result.stderr
    call = json.loads((job["run"] / "preparation/builder_call.json").read_text())
    assert call["records"] == original_records and call["records"] != job["records"].read_text()


@pytest.mark.parametrize("location", ["run", "stage"])
def test_existing_output_or_scratch_is_not_overwritten(job, location):
    job[location].mkdir(parents=True)
    sentinel = job[location] / "previous.txt"
    sentinel.write_text("preserve")
    result = run(job)
    assert result.returncode != 0 and sentinel.read_text() == "preserve"


def test_wrong_conda_prefix_fails_before_staging(job):
    with Path(job["environment"]["BASH_ENV"]).open("a") as stream:
        stream.write('conda() { export CONDA_PREFIX=/wrong/environment; }\n')
    result = run(job)
    assert result.returncode != 0 and not job["stage"].exists()


def test_corrupt_frozen_source_copy_blocks_builder(job):
    job["environment"]["COPY_DAMAGE"] = "prepare_protocol_b.py"
    result = run(job)
    assert result.returncode != 0
    assert not (job["run"] / "preparation").exists()


def test_builder_exit_status_is_not_hidden(job):
    job["environment"]["BUILDER_EXIT"] = "37"
    result = run(job)
    assert result.returncode == 37


def test_default_cpu_scratch_is_tmp_user_and_job_unique(job):
    job["environment"].pop("ROC_SCRATCH_ROOT")
    job["environment"]["USER"] = "protocol-b-wrapper-test"
    capture = job["source"].parent / "arguments.txt"
    with Path(job["environment"]["BASH_ENV"]).open("a") as stream:
        stream.write(f'''function {ENV_PATH}/bin/python {{
    if [[ "$1" == -c ]]; then command {shlex.quote(sys.executable)} "$@"; return; fi
    printf '%s\\n' "$@" > {shlex.quote(str(capture))}
    return 37
}}
''')
    result = run(job)
    assert result.returncode == 37
    arguments = capture.read_text().splitlines()
    assert arguments[arguments.index("--stage-dir") + 1] == "/tmp/protocol-b-wrapper-test/protocol-b-group0-1234"
    assert not job["run"].exists()
