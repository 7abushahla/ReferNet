import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "scripts/anchor_job_manifest.py"
CLUSTER = ROOT / "scripts/cluster"
ENV_PATH = "/shared/b00090279/conda/envs/refer-or-clear"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def helper():
    assert HELPER.is_file(), "bounded input hashing/staging helper is missing"
    specification = importlib.util.spec_from_file_location("anchor_job_manifest", HELPER)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


@pytest.fixture
def inputs(tmp_path):
    source = tmp_path / "source with spaces"
    (source / "scripts").mkdir(parents=True)
    (source / "src").mkdir()
    (source / "src/model.py").write_text("model = 'frozen'\n")
    (source / "src/._model.py").write_text("excluded")
    if HELPER.exists():
        shutil.copyfile(HELPER, source / "scripts/anchor_job_manifest.py")
    runtime = '''import argparse, json, os
from pathlib import Path
parser = argparse.ArgumentParser()
for flag in ('config', 'raw-root', 'legacy-root', 'records', 'split', 'weights',
             'weights-sha256', 'run-dir', 'device', 'phase'):
    parser.add_argument('--' + flag, required=True)
parser.add_argument('--pretrain-gate')
args = parser.parse_args()
run = Path(args.run_dir)
if args.phase in ('pretrain', 'train'):
    if run.exists() and any(run.iterdir()):
        raise ValueError('scientific run directory must be empty')
    run.mkdir(parents=True, exist_ok=True)
if args.phase == 'train':
    if args.pretrain_gate is None or not Path(args.pretrain_gate).is_file():
        raise ValueError('train requires an existing --pretrain-gate')
calls = run / 'runtime_calls.json'
history = json.loads(calls.read_text()) if calls.exists() else []
history.append(vars(args))
calls.write_text(json.dumps(history))
if os.environ.get('RUNTIME_FAIL') == args.phase:
    raise SystemExit(23)
if args.phase == 'pretrain':
    manifest = json.loads((run.parent / 'input_manifest.json').read_text())
    (run / 'gate.json').write_text(json.dumps(dict(
        phase='pretrain', passed=True, identities=manifest['runtime_identities'])))
'''
    (source / "scripts/check_anchor_pilots.py").write_text(runtime)
    (source / "src/refer_or_clear").mkdir()
    (source / "src/refer_or_clear/__init__.py").write_text("")
    (source / "src/refer_or_clear/pilots.py").write_text(runtime)
    for filename in ("pilot_config.py", "anchor_contract.py"):
        shutil.copyfile(ROOT / "src/refer_or_clear" / filename, source / "src/refer_or_clear" / filename)
    config = tmp_path / "config.yaml"
    config.write_text("schema_version: anchor_pilot_v1\ntrack: R\n")
    raw, legacy = tmp_path / "raw", tmp_path / "legacy"
    raw.mkdir()
    legacy.mkdir()
    rows = []
    for identifier, role, label in [("a", "train", 0), ("b", "train", 1),
                                    ("c", "val", 1), ("d", "test", 1)]:
        (raw / f"{identifier}.jpg").write_bytes(f"raw-{identifier}".encode())
        rows.append(dict(image_id=identifier, dataset="sidtd", path=f"{identifier}.jpg",
                         design_id="sidtd/design", component_id=identifier,
                         label=label, attack_family="text_copy" if label else ""))
        for folder in ("canvases", "masks", "valid"):
            (legacy / folder).mkdir(exist_ok=True)
            (legacy / folder / f"{identifier}.png").write_bytes(f"{folder}-{identifier}".encode())
    (raw / "clip.mp4").write_bytes(b"forbidden")
    (raw / "._a.jpg").write_bytes(b"excluded")
    (legacy / "zones").mkdir()
    (legacy / "zones/sidtd_design.png").write_bytes(b"zone")
    (legacy / "references").mkdir()
    (legacy / "references/median.png").write_bytes(b"forbidden")
    records, split = tmp_path / "records.csv", tmp_path / "split.csv"
    write_csv(records, rows)
    write_csv(split, [dict(image_id=row["image_id"], role=role, label=row["label"])
                      for row, role in zip(rows, ["train", "train", "val", "test"])])
    weights = tmp_path / "weights.pth"
    weights.write_bytes(b"offline original pretrained tensors")
    return dict(source_dir=source, config=config, raw_root=raw, legacy_root=legacy,
                records=records, split=split, weights=weights,
                weights_sha256=digest(weights), stage_dir=tmp_path / "stage",
                run_dir=tmp_path / "run", phase="diagnostic", review=None)


def prepare(helper, inputs, **changes):
    return helper.prepare_job(**dict(inputs, **changes))


def staged_files(root):
    return sorted(path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file())


def test_r_stages_only_train_val_raw_and_freezes_inputs(helper, inputs):
    manifest = prepare(helper, inputs, legacy_root=inputs["legacy_root"] / "absent")
    stage = inputs["stage_dir"]
    assert staged_files(stage / "raw") == ["a.jpg", "b.jpg", "c.jpg"]
    assert staged_files(stage / "legacy") == []
    assert not (stage / "code/src/._model.py").exists()
    assert (stage / "inputs/config.yaml").read_bytes() == inputs["config"].read_bytes()
    assert (stage / "inputs/weights.pth").read_bytes() == inputs["weights"].read_bytes()
    assert (stage / "code/src/model.py").stat().st_mode & 0o222 == 0
    assert manifest["provenance"]["config_sha256"] == digest(inputs["config"])
    assert manifest["provenance"]["weights_sha256"] == inputs["weights_sha256"]
    assert len(manifest["provenance"]["source_sha256"]) == 64
    assert len(manifest["provenance"]["data_sha256"]) == 64
    assert manifest["membership"] == {"train": ["a", "b"], "val": ["c"]}


@pytest.mark.parametrize("track,expected", [
    ("D", ["canvases/a.png", "canvases/b.png", "canvases/c.png", "zones/sidtd_design.png"]),
    ("N", ["canvases/a.png", "canvases/b.png", "canvases/c.png", "masks/b.png",
           "valid/a.png", "valid/b.png", "zones/sidtd_design.png"]),
])
def test_legacy_stages_only_needed_query_and_training_targets(helper, inputs, track, expected):
    inputs["config"].write_text(f"schema_version: anchor_pilot_v1\ntrack: {track}\n")
    prepare(helper, inputs, raw_root=inputs["raw_root"] / "absent")
    assert staged_files(inputs["stage_dir"] / "raw") == []
    assert staged_files(inputs["stage_dir"] / "legacy") == expected


def test_d_running_stages_only_train_queries_without_raw_or_targets(helper, inputs, monkeypatch):
    inputs["config"].write_text("schema_version: anchor_pilot_v1\ntrack: D_running\n")
    original_hash = helper.file_sha256

    def guarded_hash(path):
        path = Path(path)
        if path.is_relative_to(inputs["raw_root"]):
            pytest.fail(f"D_running hashed raw pixels: {path}")
        if path.is_relative_to(inputs["legacy_root"]):
            relative = path.relative_to(inputs["legacy_root"]).as_posix()
            assert relative in {"canvases/a.png", "canvases/b.png", "zones/sidtd_design.png"}, relative
        return original_hash(path)

    monkeypatch.setattr(helper, "file_sha256", guarded_hash)
    manifest = prepare(helper, inputs)
    assert manifest["track"] == "D_running"
    assert manifest["data_scope"] == "train_only"
    assert manifest["membership"] == {"train": ["a", "b"]}
    assert staged_files(inputs["stage_dir"] / "raw") == []
    assert staged_files(inputs["stage_dir"] / "legacy") == [
        "canvases/a.png", "canvases/b.png", "zones/sidtd_design.png",
    ]


def test_d_running_pilot_stages_train_val_queries_without_raw_dense_or_test(helper, inputs, monkeypatch):
    inputs["config"].write_text("schema_version: anchor_pilot_v1\ntrack: D_running_pilot\n")
    original_hash = helper.file_sha256
    expected = {"canvases/a.png", "canvases/b.png", "canvases/c.png", "zones/sidtd_design.png"}

    def guarded_hash(path):
        path = Path(path)
        if path.is_relative_to(inputs["raw_root"]):
            pytest.fail(f"D_running_pilot hashed raw pixels: {path}")
        if path.is_relative_to(inputs["legacy_root"]):
            assert path.relative_to(inputs["legacy_root"]).as_posix() in expected
        return original_hash(path)

    monkeypatch.setattr(helper, "file_sha256", guarded_hash)
    manifest = prepare(helper, inputs)
    assert manifest["track"] == "D_running_pilot"
    assert manifest["effective_config"]["architecture_id"] == "anchor_direct_running_ia18_pilot_v1"
    assert manifest["data_scope"] == "train_val_only"
    assert manifest["membership"] == {"train": ["a", "b"], "val": ["c"]}
    assert staged_files(inputs["stage_dir"] / "raw") == []
    assert staged_files(inputs["stage_dir"] / "legacy") == sorted(expected)


def test_d_running_full_train_rejected_before_data_selection_or_staging(helper, inputs, monkeypatch):
    inputs["config"].write_text("schema_version: anchor_pilot_v1\ntrack: D_running\n")

    def forbidden(*args, **kwargs):
        pytest.fail("D_running full training selected or staged data")

    monkeypatch.setattr(helper, "selected_rows", forbidden)
    monkeypatch.setattr(helper, "copy_verified", forbidden)
    with pytest.raises(ValueError, match="D_running is diagnostic-only"):
        prepare(helper, inputs, phase="train")
    assert not inputs["stage_dir"].exists()
    assert not inputs["run_dir"].exists()


@pytest.mark.parametrize("damage", ["weights", "duplicate", "mismatch", "escape", "sidecar", "symlink"])
def test_invalid_inputs_fail_before_runtime(helper, inputs, damage):
    if damage == "weights":
        inputs["weights"].write_bytes(b"wrong weights")
    elif damage == "duplicate":
        with inputs["split"].open("a") as stream:
            stream.write("a,train,0\n")
    elif damage == "mismatch":
        inputs["split"].write_text(inputs["split"].read_text().replace("a,train,0", "a,train,1"))
    elif damage == "escape":
        inputs["records"].write_text(inputs["records"].read_text().replace("a.jpg", "../weights.pth"))
    elif damage == "sidecar":
        inputs["records"].write_text(inputs["records"].read_text().replace("a.jpg", "._a.jpg"))
    else:
        (inputs["raw_root"] / "a.jpg").unlink()
        (inputs["raw_root"] / "a.jpg").symlink_to(inputs["weights"])
    with pytest.raises((ValueError, FileNotFoundError), match="hash|duplicate|mismatch|path|symlink|sidecar"):
        prepare(helper, inputs)
    assert not (inputs["run_dir"] / "runtime_calls.json").exists()


def review_record(inputs, manifest, write_gate=True):
    gate = inputs["run_dir"] / "pilot/gate.json"
    gate.parent.mkdir(exist_ok=True)
    if write_gate:
        gate.write_text(json.dumps(dict(phase="pretrain", passed=True, identities=manifest["runtime_identities"])))
    review = inputs["run_dir"].parent / "review.json"
    review.write_text(json.dumps(dict(schema_version="anchor_review_v1", approved=True,
                                     reviewer="independent reviewer", reviewed_at="2026-10-11T10:00:00Z",
                                     gate_path=str(gate), gate_sha256=digest(gate),
                                     provenance=manifest["provenance"])))
    return review


@pytest.mark.parametrize("damage", ["missing", "unapproved", "gate_failed", "config", "data",
                                    "source", "weight", "gate_hash", "reviewer", "provenance"])
@pytest.mark.parametrize("track", ["R", "D_running_pilot"])
def test_train_requires_explicit_review_matching_all_inputs(helper, inputs, damage, track):
    inputs["config"].write_text(f"schema_version: anchor_pilot_v1\ntrack: {track}\n")
    manifest = prepare(helper, inputs)
    review = review_record(inputs, manifest)
    payload = json.loads(review.read_text())
    if damage == "missing":
        review = None
    elif damage == "unapproved":
        payload["approved"] = False
    elif damage == "gate_failed":
        gate = Path(payload["gate_path"])
        evidence = json.loads(gate.read_text())
        evidence["passed"] = False
        gate.write_text(json.dumps(evidence))
        payload["gate_sha256"] = digest(gate)
    elif damage == "config":
        inputs["config"].write_text(inputs["config"].read_text() + "seed: 43\n")
    elif damage == "data":
        data_path = inputs["raw_root"] / "a.jpg" if track == "R" else inputs["legacy_root"] / "canvases/a.png"
        data_path.write_bytes(b"changed data")
    elif damage == "source":
        (inputs["source_dir"] / "src/model.py").write_text("changed source")
    elif damage == "weight":
        inputs["weights"].write_bytes(b"another checkpoint")
        inputs["weights_sha256"] = digest(inputs["weights"])
    elif damage == "gate_hash":
        payload["gate_sha256"] = "0" * 64
    elif damage == "reviewer":
        payload["reviewer"] = ""
    else:
        payload["provenance"] = {}
    if review is not None:
        review.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="review|gate|provenance|approval"):
        prepare(helper, inputs, phase="train", review=review,
                stage_dir=inputs["stage_dir"].parent / "train-stage",
                run_dir=inputs["run_dir"].parent / "train-run")
    assert not (inputs["stage_dir"].parent / "train-stage").exists()
    assert not (inputs["run_dir"].parent / "train-run").exists()


def test_d_running_pilot_rejects_old_training_only_diagnostic_review(helper, inputs):
    inputs["config"].write_text("schema_version: anchor_pilot_v1\ntrack: D_running\n")
    diagnostic = prepare(helper, inputs)
    review = review_record(inputs, diagnostic)
    inputs["config"].write_text("schema_version: anchor_pilot_v1\ntrack: D_running_pilot\n")
    stage = inputs["stage_dir"].parent / "pilot-train-stage"
    run = inputs["run_dir"].parent / "pilot-train-run"
    with pytest.raises(ValueError, match="review|provenance|identities"):
        prepare(helper, inputs, phase="train", review=review, stage_dir=stage, run_dir=run)
    assert not stage.exists()
    assert not run.exists()


def test_d_running_pilot_matching_review_authorizes_separate_fresh_training(helper, inputs, shell_env, tmp_path):
    inputs["config"].write_text("schema_version: anchor_pilot_v1\ntrack: D_running_pilot\n")
    diagnostic = prepare(helper, inputs)
    review = review_record(inputs, diagnostic)
    output = tmp_path / "approved-training"
    result = run_wrapper("train_anchor_pilot.sh", pilot_args(inputs, output, "train", review), shell_env)
    assert result.returncode == 0, result.stderr
    run = output / "anchor-pilot-train-1234"
    manifest = json.loads((run / "input_manifest.json").read_text())
    assert manifest["provenance"] == diagnostic["provenance"]
    assert manifest["runtime_identities"] == diagnostic["runtime_identities"]
    assert manifest["data_files"] == diagnostic["data_files"]
    assert manifest["data_scope"] == "train_val_only"
    assert staged_files(Path(manifest["paths"]["stage_dir"]) / "raw") == []
    assert staged_files(Path(manifest["paths"]["stage_dir"]) / "legacy") == [
        "canvases/a.png", "canvases/b.png", "canvases/c.png", "zones/sidtd_design.png",
    ]
    calls = json.loads((run / "pilot/runtime_calls.json").read_text())
    assert [call["phase"] for call in calls] == ["train", "selected"]
    assert Path(calls[0]["pretrain_gate"]).read_bytes() == (inputs["run_dir"] / "pilot/gate.json").read_bytes()
    assert Path(calls[0]["weights"]).read_bytes() == inputs["weights"].read_bytes()


def test_test_pixels_are_not_hashed_and_outputs_cannot_be_reused(helper, inputs):
    first = prepare(helper, inputs)
    (inputs["raw_root"] / "d.jpg").unlink()
    (inputs["source_dir"] / "src/._model.py").write_text("changed sidecar")
    second = prepare(helper, inputs, stage_dir=inputs["stage_dir"].parent / "stage2",
                     run_dir=inputs["run_dir"].parent / "run2")
    assert first["provenance"] == second["provenance"]
    with pytest.raises(FileExistsError):
        prepare(helper, inputs)


def test_source_snapshot_cannot_smuggle_raw_or_reference_products(helper, inputs):
    (inputs["source_dir"] / "data/raw").mkdir(parents=True)
    (inputs["source_dir"] / "data/raw/test.jpg").write_bytes(b"forbidden test pixels")
    (inputs["source_dir"] / "results").mkdir()
    (inputs["source_dir"] / "results/reference.png").write_bytes(b"forbidden reference")
    (inputs["source_dir"] / "src/test.jpg").write_bytes(b"forbidden pixels disguised as source")
    prepare(helper, inputs)
    assert not (inputs["stage_dir"] / "code/data").exists()
    assert not (inputs["stage_dir"] / "code/results").exists()
    assert not (inputs["stage_dir"] / "code/src/test.jpg").exists()


def test_changed_copy_is_rejected_instead_of_recording_old_hash(helper, inputs, monkeypatch):
    original_copy = helper.shutil.copyfile

    def change_before_copy(source, destination):
        if Path(source) == inputs["raw_root"] / "a.jpg":
            Path(source).write_bytes(b"changed during transfer")
        return original_copy(source, destination)

    monkeypatch.setattr(helper.shutil, "copyfile", change_before_copy)
    with pytest.raises(ValueError, match="hash changed"):
        prepare(helper, inputs)


@pytest.mark.parametrize("path", ["clip.mp4", "test/a.jpg", "references/a.jpg", "d.jpg"])
def test_raw_selection_rejects_clips_reference_and_test_paths(helper, inputs, path):
    target = inputs["raw_root"] / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"forbidden")
    inputs["records"].write_text(inputs["records"].read_text().replace("a.jpg", path))
    with pytest.raises(ValueError, match="raw|test|reference|clip"):
        prepare(helper, inputs)


def test_config_change_between_parse_and_hash_is_rejected(helper, inputs, monkeypatch):
    original_load = helper.resolve_config

    def change_config(source, config):
        parsed = original_load(source, config)
        inputs["config"].write_text("schema_version: anchor_pilot_v1\ntrack: D\n")
        return parsed

    monkeypatch.setattr(helper, "resolve_config", change_config)
    with pytest.raises(ValueError, match="hash changed"):
        prepare(helper, inputs)


def test_packaged_identities_match_actual_runner_source_and_resolved_config(helper, inputs):
    sys.path.insert(0, str(ROOT / "src"))
    from refer_or_clear import pilots
    from refer_or_clear.pilot_config import resolve_pilot_config

    manifest = prepare(helper, inputs, source_dir=ROOT)
    assert manifest["runtime_identities"] == dict(
        source_sha256=pilots.source_sha256(), config_sha256=pilots.json_sha256(resolve_pilot_config(inputs["config"])),
        records_sha256=digest(inputs["records"]), split_sha256=digest(inputs["split"]),
        weights_sha256=digest(inputs["weights"]))
    assert manifest["runtime_identities"]["source_sha256"] != manifest["provenance"]["source_sha256"]
    assert manifest["runtime_identities"]["config_sha256"] != manifest["provenance"]["config_sha256"]


@pytest.mark.parametrize("damage", ["gate_identity", "diagnostic_data"])
@pytest.mark.parametrize("track", ["R", "D_running_pilot"])
def test_review_is_bound_to_runtime_gate_and_original_diagnostic_manifest(helper, inputs, damage, track):
    inputs["config"].write_text(f"schema_version: anchor_pilot_v1\ntrack: {track}\n")
    manifest = prepare(helper, inputs)
    review = review_record(inputs, manifest)
    payload = json.loads(review.read_text())
    if damage == "gate_identity":
        gate = Path(payload["gate_path"])
        evidence = json.loads(gate.read_text())
        evidence["identities"]["source_sha256"] = "0" * 64
        gate.write_text(json.dumps(evidence))
        payload["gate_sha256"] = digest(gate)
        review.write_text(json.dumps(payload))
    else:
        path = inputs["run_dir"] / "input_manifest.json"
        path.chmod(0o644)
        prior = json.loads(path.read_text())
        prior["provenance"]["data_sha256"] = "0" * 64
        path.write_text(json.dumps(prior))
    with pytest.raises(ValueError, match="identit|provenance"):
        prepare(helper, inputs, phase="train", review=review,
                stage_dir=inputs["stage_dir"].parent / "stage2", run_dir=inputs["run_dir"].parent / "run2")


@pytest.fixture
def shell_env(tmp_path):
    environment_file = tmp_path / "bash_env"
    environment_file.write_text(f'''source() {{
    [[ "$1" == /opt/miniconda/etc/profile.d/conda.sh && "$SLURM_JOB_PARTITION" == gpu ]] ||
        [[ "$1" == /opt/miniconda3/etc/profile.d/conda.sh && "$SLURM_JOB_PARTITION" == cpu ]] || return 99
}}
conda() {{
    [[ "$1" == activate && "$2" == {ENV_PATH} ]] || return 98
    export CONDA_PREFIX="$2"
}}
function {ENV_PATH}/bin/python {{
    if [[ "$1" == -c && "$2" == 'import torch;'* ]]; then return 0; fi
    command {sys.executable} "$@"
}}
''')
    return dict(os.environ, BASH_ENV=str(environment_file), SLURM_JOB_ID="1234",
                SLURMD_NODENAME="gpu01", SLURM_JOB_PARTITION="gpu", SLURM_CPUS_PER_TASK="4",
                SLURM_SUBMIT_DIR=str(ROOT), ROC_SCRATCH_ROOT=str(tmp_path / "scratch"))


def run_wrapper(name, arguments, environment):
    wrapper = CLUSTER / name
    assert wrapper.is_file(), f"{name} wrapper is missing"
    return subprocess.run(["bash", str(wrapper), *map(str, arguments)], env=environment,
                          text=True, capture_output=True)


def test_python_mock_bypasses_itself_when_environment_matches_interpreter(monkeypatch, request):
    monkeypatch.setattr(sys.modules[__name__], "ENV_PATH", str(Path(sys.executable).parent.parent))
    environment = request.getfixturevalue("shell_env")
    environment["FUNCNEST"] = "12"
    result = subprocess.run(["bash", "-c", f"{sys.executable} -c 'print(42)'"],
                            env=environment, text=True, capture_output=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "42"


def pilot_args(inputs, output, phase="diagnostic", review=None):
    args = [phase, inputs["source_dir"], inputs["config"], inputs["raw_root"], inputs["legacy_root"],
            inputs["records"], inputs["split"], inputs["weights"], inputs["weights_sha256"], output]
    return args + ([review] if review else [])


@pytest.mark.parametrize("guard", ["job", "node", "login", "partition", "arguments", "phase"])
def test_wrapper_rejects_unallocated_or_invalid_invocations(inputs, shell_env, tmp_path, guard):
    args = pilot_args(inputs, tmp_path / "out")
    if guard == "job":
        shell_env.pop("SLURM_JOB_ID")
    elif guard == "node":
        shell_env.pop("SLURMD_NODENAME")
    elif guard == "login":
        shell_env["SLURMD_NODENAME"] = "login01"
    elif guard == "partition":
        shell_env["SLURM_JOB_PARTITION"] = "cpu"
    elif guard == "arguments":
        args = args[:-1]
    else:
        args[0] = "smoke-and-train"
    result = run_wrapper("train_anchor_pilot.sh", args, shell_env)
    assert result.returncode != 0
    assert not (tmp_path / "scratch").exists()


def test_diagnostic_exits_without_training_and_approved_train_is_separate(inputs, shell_env, tmp_path):
    output = tmp_path / "out"
    result = run_wrapper("train_anchor_pilot.sh", pilot_args(inputs, output), shell_env)
    assert result.returncode == 0, result.stderr
    run = output / "anchor-pilot-diagnostic-1234"
    calls = json.loads((run / "pilot/runtime_calls.json").read_text())
    assert [call["phase"] for call in calls] == ["pretrain"]
    assert calls[0]["device"] == "cuda"
    for name in ("config", "records", "split", "weights", "raw_root", "legacy_root"):
        assert Path(calls[0][name]).is_relative_to(tmp_path / "scratch")
    manifest = json.loads((run / "input_manifest.json").read_text())
    assert manifest["environment"]["conda_prefix"] == ENV_PATH
    assert manifest["allocation"]["job_id"] == "1234"
    review = review_record(dict(inputs, run_dir=run), manifest)
    shell_env["SLURM_JOB_ID"] = "1235"
    result = run_wrapper("train_anchor_pilot.sh", pilot_args(inputs, output, "train", review), shell_env)
    assert result.returncode == 0, result.stderr
    trained = output / "anchor-pilot-train-1235"
    calls = json.loads((trained / "pilot/runtime_calls.json").read_text())
    assert [call["phase"] for call in calls] == ["train", "selected"]
    assert not any(call["phase"] == "pretrain" for call in calls)
    assert (trained / "inputs/review.json").read_bytes() == review.read_bytes()
    assert Path(calls[0]["pretrain_gate"]).read_bytes() == (run / "pilot/gate.json").read_bytes()
    assert Path(calls[0]["pretrain_gate"]).is_relative_to(tmp_path / "scratch")


def test_runtime_failure_propagates_without_selected_check(inputs, shell_env, tmp_path):
    output = tmp_path / "out"
    run_wrapper("train_anchor_pilot.sh", pilot_args(inputs, output), shell_env)
    run = output / "anchor-pilot-diagnostic-1234"
    manifest = json.loads((run / "input_manifest.json").read_text())
    review = review_record(dict(inputs, run_dir=run), manifest)
    shell_env.update(SLURM_JOB_ID="1235", RUNTIME_FAIL="train")
    result = run_wrapper("train_anchor_pilot.sh", pilot_args(inputs, output, "train", review), shell_env)
    assert result.returncode == 23
    calls = json.loads((output / "anchor-pilot-train-1235/pilot/runtime_calls.json").read_text())
    assert [call["phase"] for call in calls] == ["train"]


def test_relative_output_and_scratch_paths_are_resolved_before_runtime(inputs, shell_env, tmp_path):
    shell_env["SLURM_SUBMIT_DIR"] = str(tmp_path)
    shell_env["ROC_SCRATCH_ROOT"] = "relative-scratch"
    result = run_wrapper("train_anchor_pilot.sh", pilot_args(inputs, "relative-output"), shell_env)
    assert result.returncode == 0, result.stderr
    run = tmp_path / "relative-output/anchor-pilot-diagnostic-1234"
    calls = json.loads((run / "pilot/runtime_calls.json").read_text())
    assert Path(calls[0]["run_dir"]) == run / "pilot"
    assert Path(calls[0]["config"]).is_absolute()


def test_diagnostic_fails_if_success_exit_has_no_gate(inputs, shell_env, tmp_path):
    checker = inputs["source_dir"] / "scripts/check_anchor_pilots.py"
    checker.write_text(checker.read_text().replace("if args.phase == 'pretrain':", "if False:"))
    output = tmp_path / "out"
    result = run_wrapper("train_anchor_pilot.sh", pilot_args(inputs, output), shell_env)
    assert result.returncode != 0
    calls = json.loads((output / "anchor-pilot-diagnostic-1234/pilot/runtime_calls.json").read_text())
    assert [call["phase"] for call in calls] == ["pretrain"]


def test_audit_wrapper_freezes_metadata_but_never_stages_or_decodes_pixels(inputs, shell_env, tmp_path):
    shell_env["SLURM_JOB_PARTITION"] = "cpu"
    shell_env["SLURMD_NODENAME"] = "cpu01"
    audit_script = inputs["source_dir"] / "scripts/audit_encoding_headers.py"
    audit_script.write_text('''import argparse, json
from pathlib import Path
parser = argparse.ArgumentParser()
for name in ('records', 'split', 'raw-root', 'out'):
    parser.add_argument('--' + name, required=True)
parser.add_argument('--inventory-test-directory', action='store_true')
args = parser.parse_args()
out = Path(args.out)
out.mkdir()
(out / 'called.json').write_text(json.dumps(vars(args)))
(out / 'summary.json').write_text(json.dumps(dict(schema_version=2,
    groups=dict(candidate_test={}), disk_inventory=dict(scanned=args.inventory_test_directory),
    completeness=dict(complete=True, candidate_headers_complete=True,
        manifest_membership_complete=True, exhaustive_test_inventory_confirmed=True, reasons=[]))))
''')
    output = tmp_path / "audit-output"
    result = run_wrapper("audit_encoding_headers.sh", [inputs["source_dir"], inputs["records"],
                         inputs["split"], inputs["raw_root"], output], shell_env)
    assert result.returncode == 0, result.stderr
    run = output / "encoding-headers-1234"
    calls = json.loads((run / "audit/called.json").read_text())
    assert Path(calls["raw_root"]) == inputs["raw_root"]
    assert calls["inventory_test_directory"] is True
    assert Path(calls["records"]).is_relative_to(tmp_path / "scratch")
    manifest = json.loads((run / "input_manifest.json").read_text())
    assert manifest["data_scope"] == "metadata_only"
    assert not list((tmp_path / "scratch").rglob("*.jpg"))
    assert not (run / "runtime_calls.json").exists()
    assert json.loads((run / "audit/health.json").read_text())["passed"] is True


def test_incomplete_audit_retains_summary_and_reports_failed_health(inputs, shell_env, tmp_path):
    shell_env["SLURM_JOB_PARTITION"] = "cpu"
    shell_env["SLURMD_NODENAME"] = "cpu01"
    script = inputs["source_dir"] / "scripts/audit_encoding_headers.py"
    script.write_text('''import argparse, json
from pathlib import Path
parser = argparse.ArgumentParser()
for name in ('records', 'split', 'raw-root', 'out'):
    parser.add_argument('--' + name, required=True)
parser.add_argument('--inventory-test-directory', action='store_true')
args = parser.parse_args()
out = Path(args.out)
out.mkdir()
(out / 'summary.json').write_text(json.dumps(dict(schema_version=2,
    groups=dict(candidate_test={}), disk_inventory=dict(scanned=True),
    completeness=dict(complete=False, candidate_headers_complete=False,
        manifest_membership_complete=True, exhaustive_test_inventory_confirmed=False,
        reasons=['missing FantasyID test files']))))
''')
    output = tmp_path / "audit-output"
    result = run_wrapper("audit_encoding_headers.sh", [inputs["source_dir"], inputs["records"],
                         inputs["split"], inputs["raw_root"], output], shell_env)
    assert result.returncode != 0
    run = output / "encoding-headers-1234"
    assert (run / "audit/summary.json").exists()
    health = json.loads((run / "audit/health.json").read_text())
    assert health["passed"] is False
    assert "missing FantasyID test files" in health["reasons"]


def test_real_checker_and_runner_use_child_runs_matching_identities_and_staged_gate(inputs, shell_env, tmp_path):
    source = inputs["source_dir"]
    shutil.copytree(ROOT / "src", source / "src", dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("__pycache__", "._*"))
    shutil.copyfile(ROOT / "scripts/check_anchor_pilots.py", source / "scripts/check_anchor_pilots.py")
    (source / "tests").mkdir()
    shutil.copyfile(ROOT / "tests/test_anchor_training.py", source / "tests/fixture_training.py")
    bootstrap = source / "scripts/fixture_bootstrap.py"
    bootstrap.write_text('''import importlib.util, os, sys
from pathlib import Path
import pandas as pd
import torch
runtime_root = Path(os.environ['PYTHONPATH'].split(os.pathsep)[0]).parent
sys.path.insert(0, str(runtime_root / 'tests'))
from fixture_training import TinyPilot, FixtureDataset
from refer_or_clear import pilots
from refer_or_clear.data import query
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
pilots.OFFICIAL_WEIGHT_SHA256['R'] = sys.argv[sys.argv.index('--weights-sha256') + 1]
def builder(**kwargs):
    model = TinyPilot()
    model.initialization_provenance['checkpoint_sha256'] = kwargs['expected_sha256']
    model.initialization_provenance['checkpoint_path'] = str(kwargs['weights_path'])
    return model
def rows(records, split, role):
    assert role in ('train', 'val')
    count = 1778 if role == 'train' else 222
    return pd.DataFrame([dict(image_id=f'{role}-{index:04d}', path=f'{role}-{index}.png',
        label=index % 2, design_id='design', component_id=f'{role}-{index}', role=role,
        attack_family='' if index % 2 == 0 else ('text_copy' if index % 8 == 1 else 'text_inpaint'))
        for index in range(count)])
pilots._model_builder = lambda track: builder
query.AnchorQueryDataset = FixtureDataset
query.load_role_rows = rows
actual_launch = pilots.subprocess.run
def replay_launch(command, **kwargs):
    return actual_launch([command[0], __file__, *command[1:]], **kwargs)
pilots.subprocess.run = replay_launch
arguments = sys.argv[1:]
if arguments[0] == '-m':
    assert arguments[1] == 'refer_or_clear.pilots'
    arguments = arguments[2:]
    is_checker = False
else:
    checker_path = arguments.pop(0)
    is_checker = True
position = arguments.index('--device') + 1
assert arguments[position] in ('cuda', 'cpu')
arguments[position] = 'cpu'
if is_checker:
    specification = importlib.util.spec_from_file_location('real_checker', checker_path)
    checker = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(checker)
    result = checker.main(arguments)
else:
    result = pilots.main(arguments)
raise SystemExit(0 if result['passed'] else 1)
''')
    with Path(shell_env["BASH_ENV"]).open("a") as stream:
        stream.write(f'''function {ENV_PATH}/bin/python {{
    if [[ "$1" == -c && "$2" == 'import torch;'* ]]; then return 0; fi
    if [[ "$1" == */check_anchor_pilots.py || ( "$1" == -m && "$2" == refer_or_clear.pilots ) ]]; then
        command {sys.executable} "$ROC_TEST_BOOTSTRAP" "$@"
    else
        command {sys.executable} "$@"
    fi
}}
''')
    shell_env["ROC_TEST_BOOTSTRAP"] = str(bootstrap)
    output = tmp_path / "real-runtime-output"
    diagnostic = run_wrapper("train_anchor_pilot.sh", pilot_args(inputs, output), shell_env)
    assert diagnostic.returncode == 0, diagnostic.stderr
    run = output / "anchor-pilot-diagnostic-1234"
    manifest = json.loads((run / "input_manifest.json").read_text())
    gate = json.loads((run / "pilot/gate.json").read_text())
    assert gate["passed"] and gate["identities"] == manifest["runtime_identities"]
    replay = json.loads((run / "pilot/smoke/replay.json").read_text())
    assert replay["passed"] and replay["fresh_process"]
    review = review_record(dict(inputs, run_dir=run), manifest, write_gate=False)
    shell_env["SLURM_JOB_ID"] = "1235"
    training = run_wrapper("train_anchor_pilot.sh", pilot_args(inputs, output, "train", review), shell_env)
    assert training.returncode == 0, training.stderr
    trained = output / "anchor-pilot-train-1235"
    info = json.loads((trained / "pilot/train.json").read_text())
    assert info["epochs_completed"] == 100
    assert info["fresh_initialization"]["pretrain_gate_sha256"] == digest(run / "pilot/gate.json")
    selected = json.loads((trained / "pilot/gate.json").read_text())
    assert selected["passed"] and selected["phase"] == "selected"
    assert selected["identities"] == gate["identities"]


def test_d_running_wrapper_rejects_full_train_before_staging(inputs, shell_env, tmp_path):
    inputs["config"].write_text("schema_version: anchor_pilot_v1\ntrack: D_running\n")
    output = tmp_path / "out"
    review = tmp_path / "unused-review.json"
    result = run_wrapper("train_anchor_pilot.sh", pilot_args(inputs, output, "train", review), shell_env)
    assert result.returncode != 0
    assert "D_running is diagnostic-only" in result.stderr
    assert not Path(shell_env["ROC_SCRATCH_ROOT"]).exists()
    assert not output.exists()


@pytest.mark.parametrize("track", ["D", "D_running", "D_running_pilot", "N"])
@pytest.mark.parametrize("outcome", ["pass", "fail", "skip"])
def test_dn_diagnostic_requires_actual_stack_gate_before_checker(inputs, shell_env, tmp_path, track, outcome):
    inputs["config"].write_text(f"schema_version: anchor_pilot_v1\ntrack: {track}\n")
    tests_dir = inputs["source_dir"] / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_anchor_changer.py").write_text('''import hashlib, json, os
from pathlib import Path
import pytest
def test_actual_opencd_gpu_integration():
    assert os.environ['ROC_ANCHOR_GPU_INTEGRATION'] == '1'
    checkpoint = Path(os.environ['ROC_LEVIR_CKPT'])
    assert checkpoint.name == 'weights.pth'
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == os.environ['ROC_LEVIR_SHA256']
    Path(os.environ['ROC_TEST_STACK_OUT']).write_text(json.dumps(dict(
        checkpoint=str(checkpoint), sha256=os.environ['ROC_LEVIR_SHA256'])))
    if os.environ['ROC_TEST_STACK_OUTCOME'] == 'skip':
        pytest.skip('unexpected silent skip')
    assert os.environ['ROC_TEST_STACK_OUTCOME'] != 'fail', 'actual stack gate failure'
''')
    recorded = tmp_path / "stack-called.json"
    shell_env.update(ROC_TEST_STACK_OUT=str(recorded), ROC_TEST_STACK_OUTCOME=outcome)
    output = tmp_path / "out"
    result = run_wrapper("train_anchor_pilot.sh", pilot_args(inputs, output), shell_env)
    assert recorded.exists(), result.stderr
    run = output / "anchor-pilot-diagnostic-1234"
    assert (run / "pytest-version.txt").is_file()
    assert (run / "actual-stack-test.log").is_file()
    if outcome == "pass":
        assert result.returncode == 0, result.stderr
        calls = json.loads((run / "pilot/runtime_calls.json").read_text())
        assert [call["phase"] for call in calls] == ["pretrain"]
        if track == "D_running":
            manifest = json.loads((run / "input_manifest.json").read_text())
            assert manifest["data_scope"] == "train_only"
            assert manifest["membership"] == {"train": ["a", "b"]}
    else:
        assert result.returncode != 0
        assert not (run / "pilot").exists()


def test_slurm_requests_one_gpu_without_changing_resource_pattern():
    wrapper = CLUSTER / "train_anchor_pilot.sh"
    assert wrapper.exists(), "training wrapper missing"
    directives = [line for line in wrapper.read_text().splitlines() if line.startswith("#SBATCH")]
    for request in ("--account=acc-mialhajri", "--partition=gpu", "--nodes=1", "--ntasks=1",
                    "--gres=gpu:1", "--cpus-per-task=4", "--mem=8G", "--time=6:00:00"):
        assert f"#SBATCH {request}" in directives


def test_cpu_audit_default_scratch_uses_node_tmp_and_keeps_override():
    wrapper = (ROOT / "scripts/cluster/audit_encoding_headers.sh").read_text()
    assert 'scratch="${ROC_SCRATCH_ROOT:-/tmp/$USER}/encoding-headers-$SLURM_JOB_ID"' in wrapper
    assert "/scratch/$USER" not in wrapper


def test_header_audit_requests_cpu_partition_without_gpu():
    wrapper = CLUSTER / "audit_encoding_headers.sh"
    directives = [line for line in wrapper.read_text().splitlines() if line.startswith("#SBATCH")]
    assert "#SBATCH --partition=cpu" in directives
    assert not any("--gres" in line or "--gpus" in line for line in directives)
