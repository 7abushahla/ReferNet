import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "scripts/cluster/prepare_protocol_c.sh"
ENV_PATH = "/shared/b00090279/conda/envs/refer-or-clear"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def job(tmp_path):
    source = tmp_path / "readonly source"
    (source / "src").mkdir(parents=True)
    (source / "scripts").mkdir()
    (source / "src/frozen.py").write_text("version = 1\n")
    (source / "scripts/prepare_protocol_c.py").write_text('''import argparse, json, os
from pathlib import Path
parser = argparse.ArgumentParser(allow_abbrev=False)
for name in ('records', 'records-sha256', 'source-split', 'target-split', 'raw-root', 'output-root'):
    parser.add_argument('--' + name, required=True)
parser.add_argument('--build', action='store_true')
args = parser.parse_args()
assert args.build
output = Path(args.output_root)
output.mkdir()
call = dict(arguments=vars(args), script=str(Path(__file__).resolve()),
            metadata={name: Path(getattr(args, name)).read_text() for name in ('records', 'source_split', 'target_split')},
            environment={name: os.environ.get(name) for name in
                         ('CONDA_PREFIX', 'PYTHONPATH', 'PYTHONHOME', 'TMPDIR', 'CUDA_VISIBLE_DEVICES', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')})
(output / 'call.json').write_text(json.dumps(call))
raise SystemExit(int(os.environ.get('PRODUCER_EXIT', '0')))
''')
    for path in source.rglob("*"):
        path.chmod(0o555 if path.is_dir() else 0o444)
    source.chmod(0o555)
    records = tmp_path / "records.csv"
    records.write_text("image_id,dataset\nsupport,fantasyid\n")
    source_split = tmp_path / "source.csv"
    source_split.write_text("image_id,protocol,role\nsource,A,train\n")
    target_split = tmp_path / "target.csv"
    target_split.write_text("image_id,protocol,role\nsupport,C,test\n")
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "unread-pixels.bin").write_bytes(b"never decoded")
    driver = tmp_path / "python driver.py"
    driver.write_text('''import os, runpy, sys
sys.prefix = os.environ.get('MOCK_PREFIX', '/shared/b00090279/conda/envs/refer-or-clear')
sys.executable = os.environ.get('MOCK_EXECUTABLE', sys.prefix + '/bin/python')
arguments = sys.argv[1:]
if arguments[0] in ('-', '-c'):
    code = sys.stdin.read() if arguments[0] == '-' else arguments[1]
    sys.argv = [arguments[0], *arguments[1 if arguments[0] == '-' else 2:]]
    exec(code, {'__name__': '__main__'})
else:
    sys.argv = arguments
    runpy.run_path(arguments[0], run_name='__main__')
''')
    bootstrap = tmp_path / "bash_env"
    bootstrap.write_text(f'''source() {{ [[ "$1" == /opt/miniconda3/etc/profile.d/conda.sh ]]; }}
conda() {{
    [[ "$1" == activate && "$2" == {ENV_PATH} ]] || return 98
    export CONDA_PREFIX="${{MOCK_CONDA_PREFIX:-$2}}"
}}
function {ENV_PATH}/bin/python {{ command {shlex.quote(sys.executable)} {shlex.quote(str(driver))} "$@"; }}
''')
    environment = dict(os.environ, BASH_ENV=str(bootstrap), SLURM_JOB_ID="1234", SLURMD_NODENAME="cpu01",
                       SLURM_JOB_PARTITION="cpu", SLURM_CPUS_PER_TASK="4", SLURM_SUBMIT_DIR=str(tmp_path),
                       ROC_SCRATCH_ROOT=str(tmp_path / "scratch"), PYTHONPATH="/foreign", PYTHONHOME="/foreign")
    for name in ("SLURM_GPUS", "SLURM_GPUS_ON_NODE", "SLURM_JOB_GPUS", "SLURM_STEP_GPUS",
                 "MOCK_PREFIX", "MOCK_EXECUTABLE", "MOCK_CONDA_PREFIX", "PRODUCER_EXIT"):
        environment.pop(name, None)
    return dict(source=source, records=records, source_split=source_split, target_split=target_split, raw=raw,
                environment=environment, arguments=[source, records, sha(records), source_split, target_split, raw, tmp_path / "outputs"],
                run=tmp_path / "outputs/protocol-c-support-1234", scratch=tmp_path / "scratch/protocol-c-support-1234")


def run(job, arguments=None):
    assert WRAPPER.is_file(), "C CPU wrapper missing"
    return subprocess.run(["bash", str(WRAPPER), *map(str, job["arguments"] if arguments is None else arguments)],
                          env=job["environment"], capture_output=True, text=True, timeout=30)


def test_cpu_header_and_bootstrap():
    assert WRAPPER.is_file(), "C CPU wrapper missing"
    content = WRAPPER.read_text()
    options = dict(line.removeprefix("#SBATCH --").split("=", 1) for line in content.splitlines()
                   if line.startswith("#SBATCH --") and "=" in line)
    assert {key: options[key] for key in ("account", "partition", "nodes", "ntasks", "cpus-per-task", "mem", "time")} == {
        "account": "acc-mialhajri", "partition": "cpu", "nodes": "1", "ntasks": "1", "cpus-per-task": "4", "mem": "8G", "time": "1:00:00"}
    assert "gres" not in options
    assert "source /opt/miniconda3/etc/profile.d/conda.sh" in content
    assert not any(token in content for token in ("sbatch ", "ssh ", "git ", "anchor_job_manifest.py"))
    assert subprocess.run(["bash", "-n", str(WRAPPER)], capture_output=True).returncode == 0


def test_build_uses_durable_readonly_c_metadata_and_original_readonly_source(job):
    result = run(job)
    assert result.returncode == 0, result.stderr
    call = json.loads((job["run"] / "preparation/call.json").read_text())
    assert call["arguments"] == dict(records=str(job["run"] / "inputs/records.csv"), records_sha256=sha(job["records"]),
        source_split=str(job["run"] / "inputs/source-split.csv"), target_split=str(job["run"] / "inputs/target-split.csv"),
        raw_root=str(job["raw"]), output_root=str(job["run"] / "preparation"), build=True)
    assert call["script"] == str(job["source"] / "scripts/prepare_protocol_c.py")
    assert call["metadata"] == {name: job[name].read_text() for name in ("records", "source_split", "target_split")}
    for name in ("records", "source_split", "target_split"):
        assert Path(call["arguments"][name]).stat().st_mode & 0o222 == 0
    assert call["environment"] == dict(CONDA_PREFIX=ENV_PATH, PYTHONPATH=str(job["source"] / "src"), PYTHONHOME=None,
        TMPDIR=str(job["scratch"] / "tmp"), CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
    assert not (job["run"] / "input_manifest.json").exists()
    assert not (job["scratch"] / "raw").exists()


@pytest.mark.parametrize("name,value", [("SLURM_JOB_ID", ""), ("SLURM_JOB_ID", "0"), ("SLURM_JOB_ID", "1/x"),
    ("SLURMD_NODENAME", "LOGIN01"), ("SLURMD_NODENAME", "head01"), ("SLURMD_NODENAME", "frontend01"), ("SLURMD_NODENAME", "lgn01"),
    ("SLURMD_NODENAME", ""), ("SLURM_SUBMIT_DIR", ""), ("SLURM_JOB_PARTITION", "gpu"), ("SLURM_JOB_PARTITION", "hpc"),
    ("SLURM_CPUS_PER_TASK", "3"), ("SLURM_CPUS_PER_TASK", "four"), ("SLURM_GPUS", "1"), ("SLURM_GPUS_ON_NODE", "1"),
    ("SLURM_JOB_GPUS", "0"), ("SLURM_STEP_GPUS", "0"), ("MOCK_CONDA_PREFIX", "/wrong"),
    ("MOCK_PREFIX", "/wrong"), ("MOCK_EXECUTABLE", "/wrong/bin/python")])
def test_allocation_and_environment_errors_block_staging(job, name, value):
    job["environment"][name] = value
    result = run(job)
    assert result.returncode != 0 and not job["run"].exists() and not job["scratch"].exists()


@pytest.mark.parametrize("value", ["bad", "A" * 64, "a" * 64])
def test_records_hash_rejected_before_staging(job, value):
    job["arguments"][2] = value
    result = run(job)
    assert result.returncode != 0 and not job["run"].exists() and not job["scratch"].exists()


@pytest.mark.parametrize("location", ["source", "run", "scratch"])
def test_writable_source_or_existing_destinations_rejected(job, location):
    if location == "source":
        job["source"].chmod(0o755)
    else:
        job[location].mkdir(parents=True)
        (job[location] / "sentinel").write_text("preserve")
    result = run(job)
    assert result.returncode != 0
    if location != "source":
        assert (job[location] / "sentinel").read_text() == "preserve"


@pytest.mark.parametrize("count", [0, 6, 8])
def test_exact_seven_arguments(job, count):
    arguments = job["arguments"][:count] if count < 7 else [*job["arguments"], "extra"]
    assert run(job, arguments).returncode != 0
    assert not job["run"].exists()


def test_producer_failure_propagates(job):
    job["environment"]["PRODUCER_EXIT"] = "37"
    assert run(job).returncode == 37


def test_default_scratch_is_tmp_user_job_unique(job):
    job["environment"].pop("ROC_SCRATCH_ROOT")
    job["environment"]["USER"] = "protocol-c-wrapper-test"
    capture = job["source"].parent / "captured.txt"
    with Path(job["environment"]["BASH_ENV"]).open("a") as stream:
        stream.write(f'''function {ENV_PATH}/bin/python {{ printf '%s\\n' "$@" > {shlex.quote(str(capture))}; return 37; }}\n''')
    assert run(job).returncode == 37
    assert "/tmp/protocol-c-wrapper-test/protocol-c-support-1234" in capture.read_text().splitlines()
