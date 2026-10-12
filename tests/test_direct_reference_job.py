import csv
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]


def binding(path):
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


@pytest.fixture
def job():
    path = ROOT / "scripts/direct_reference_job.py"
    assert path.is_file(), "direct-reference diagnostic staging is missing"
    spec = importlib.util.spec_from_file_location("direct_reference_job", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def setup(job, tmp_path, monkeypatch):
    source = tmp_path / "source"
    (source / "scripts").mkdir(parents=True)
    (source / "scripts/anchor_job_manifest.py").write_bytes((ROOT / "scripts/anchor_job_manifest.py").read_bytes())
    (source / "scripts/verify_anchor_enrollment_products.py").write_text(
        'def verify_products(*args, **kwargs):\n    return {"passed": True, "counts": {"products_verified": 270}}\n')
    (source / "src").mkdir()
    (source / "src/dummy.py").write_text("value = 1\n")
    config = source / "configs/direct_reference_pilot.yaml"
    config.parent.mkdir()
    config.write_text("schema_version: direct_reference_pilot_v1\n")
    raw = tmp_path / "raw"
    raw.mkdir()
    rows = []
    for index in range(2222):
        role = "train" if index < 1778 else "val" if index < 2000 else "test"
        relative = f"sidtd/templates/Images/{index}.jpg"
        path = raw / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(str(index).encode())
        rows.append({"image_id": f"id{index:04d}", "dataset": "sidtd", "path": relative,
                     "role": role, "label": str(index % 2), "design_id": "sidtd/design", "component_id": f"c{index}"})
    authorities = {}
    for name, fields in (("records", ["image_id", "dataset", "path", "label", "design_id", "component_id"]),
                         ("split", ["image_id", "role", "label", "design_id", "component_id"])):
        path = tmp_path / f"{name}.csv"
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        authorities[name] = binding(path)
    products = tmp_path / "products"
    product = products / "fold0/sidtd/design/enrollment/zone.png"
    product.parent.mkdir(parents=True)
    product.write_bytes(b"synthetic product")
    enrollment = tmp_path / "enrollment.json"
    enrollment.write_text(json.dumps({"output_root": str(products), "banks": {
        "sidtd/design": {"enrollment": {"products": {"zone": binding(product)}}}}}))
    monkeypatch.setattr(job, "ENROLLMENT_SHA256", binding(enrollment)["sha256"])
    weights = tmp_path / "weights.pth"
    weights.write_bytes(b"synthetic weights")
    monkeypatch.setattr(job, "LEVIR_SHA256", binding(weights)["sha256"])
    source_hash = job.source_digest(source)
    request = tmp_path / "request.json"
    request.write_text(json.dumps({"schema_version": "direct_reference_diagnostic_request_v1",
        "source_sha256": source_hash, "config": binding(config), "weights": binding(weights),
        "enrollment": binding(enrollment), "product_root": str(products), "raw_root": str(raw), **authorities}))
    monkeypatch.setattr(job, "validate_allocation", lambda: None)
    return SimpleNamespace(source=source, source_hash=source_hash, request=request,
                           scratch=tmp_path / "scratch", run=tmp_path / "run", raw=raw, rows=rows)


def prepare(job, setup):
    return job.prepare_job(source_dir=setup.source, source_sha256=setup.source_hash,
        request=setup.request, request_sha256=binding(setup.request)["sha256"],
        stage_dir=setup.scratch, run_dir=setup.run)


def test_diagnostic_staging_preserves_authorities_and_excludes_test(job, setup):
    original = setup.request.read_bytes()
    result = prepare(job, setup)
    assert result["schema_version"] == "direct_reference_job_v1"
    assert result["diagnostic_only"] is True
    assert result["membership_counts"] == {"train": 1778, "val": 222}
    assert (setup.scratch / "inputs/request.json").read_bytes() == original
    assert len(result["query_files"]) == 2000
    for row in setup.rows[2000:]:
        assert not (setup.scratch / "raw" / row["path"]).exists()
    assert all(not path.stat().st_mode & 0o222 for path in (setup.scratch / "raw").rglob("*.jpg"))
    relocation = json.loads((setup.scratch / "inputs/product-root.json").read_text())
    assert relocation["enrollment_manifest_sha256"] == job.ENROLLMENT_SHA256
    assert relocation["relocated_root"] == str(setup.scratch / "enrollment")
    assert json.loads((setup.run / "input_manifest.json").read_text()) == result


@pytest.mark.parametrize("mutation", ["source", "request", "weights", "enrollment", "records", "split", "symlink", "output"])
def test_invalid_staging_fails_before_copying_queries(job, setup, mutation):
    if mutation == "source":
        (setup.source / "src/dummy.py").write_text("changed\n")
    elif mutation == "request":
        with pytest.raises(ValueError):
            job.prepare_job(source_dir=setup.source, source_sha256=setup.source_hash,
                request=setup.request, request_sha256="0" * 64, stage_dir=setup.scratch, run_dir=setup.run)
        assert not setup.scratch.exists()
        return
    elif mutation == "output":
        setup.scratch.mkdir()
    else:
        request = json.loads(setup.request.read_text())
        key = "weights" if mutation == "symlink" else mutation
        path = Path(request[key]["path"])
        if mutation == "symlink":
            replacement = path.with_name("replacement.pth")
            path.rename(replacement)
            path.symlink_to(replacement)
        else:
            path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises((ValueError, FileExistsError)):
        prepare(job, setup)
    assert not (setup.scratch / "raw").exists()


@pytest.mark.parametrize("partition,node", [("cpu", "cpu01"), ("gpu", "login01"), ("", "gpu01")])
def test_stager_requires_gpu_allocation(job, monkeypatch, partition, node):
    monkeypatch.setenv("SLURM_JOB_ID", "1234")
    monkeypatch.setenv("SLURM_JOB_PARTITION", partition)
    monkeypatch.setenv("SLURMD_NODENAME", node)
    with pytest.raises(ValueError):
        job.validate_allocation()


def test_source_cannot_change_and_revert_between_inventory_scans(job, setup, monkeypatch):
    common = job.helpers()
    original = common.source_entries
    original_copy = common.copy_verified
    candidate = setup.source / "src/dummy.py"
    before = candidate.read_bytes()
    scans = []

    def inventory(root):
        entries = original(root)
        if Path(root) == setup.source:
            scans.append(entries)
            if len(scans) == 1:
                candidate.write_bytes(b"changed during freeze\n")
        return entries

    def copy_file(source, target, digest):
        original_copy(source, target, digest)
        if Path(source) == candidate:
            candidate.write_bytes(before)

    monkeypatch.setattr(common, "source_entries", inventory)
    monkeypatch.setattr(common, "copy_verified", copy_file)
    monkeypatch.setattr(job, "helpers", lambda: common)
    with pytest.raises(ValueError, match="hash|SHA256"):
        prepare(job, setup)
    assert not (setup.run / "input_manifest.json").exists()


@pytest.mark.parametrize("case", ["absent", "unrelated", "skipped", "failure", "error"])
def test_required_actual_test_case_cannot_be_hidden_by_success_counters(job, tmp_path, case):
    name = "unrelated_test" if case == "unrelated" else "test_actual_direct_reference_gpu_integration"
    body = "" if case in ("absent", "unrelated") else f"<{case}/>"
    testcase = "" if case == "absent" else f'<testcase classname="tests.test_direct_reference_model" name="{name}">{body}</testcase>'
    path = tmp_path / "stack.xml"
    path.write_text(f'<testsuites><testsuite tests="1" failures="0" errors="0" skipped="0">{testcase}</testsuite></testsuites>')
    with pytest.raises(ValueError):
        job.check_stack_report(path)


@pytest.mark.parametrize("counts", [(1, 0, 0), (1, 1, 0), (1, 0, 1), (0, 0, 0)])
def test_actual_stack_gate_rejects_skips_failures_and_empty_reports(job, tmp_path, counts):
    tests, failures, skipped = counts
    path = tmp_path / "stack.xml"
    testcase = '<testcase classname="tests.test_direct_reference_model" name="test_actual_direct_reference_gpu_integration"/>' if tests else ""
    path.write_text(f'<testsuites><testsuite tests="{tests}" failures="{failures}" errors="0" skipped="{skipped}">{testcase}</testsuite></testsuites>')
    if counts == (1, 0, 0):
        assert job.check_stack_report(path) is True
    else:
        with pytest.raises(ValueError):
            job.check_stack_report(path)
