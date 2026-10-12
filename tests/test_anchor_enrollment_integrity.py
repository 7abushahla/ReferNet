import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/verify_anchor_enrollment_products.py"
PRODUCTS = ("target", "target_coverage", "target_valid", "median", "coverage", "valid", "zone")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, payload):
    path.write_text(json.dumps(payload, sort_keys=True) + "\n")


@pytest.fixture
def verifier():
    assert SCRIPT.is_file(), "enrollment product verifier is not implemented"
    spec = importlib.util.spec_from_file_location("enrollment_integrity_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def bundle(tmp_path):
    output = tmp_path / "support"
    output.mkdir()
    source = {"implementation_sha256": "a" * 64, "helpers": {}, "revision": "fixture"}
    contributors = {identifier: {"image_id": identifier, "design_id": "sidtd/example", "component_id": identifier,
                                  "raw_sha256": "b" * 64, "layout_sha256": "c" * 64}
                    for identifier in ("support_0", "support_1")}
    identity = [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]
    banks = {}
    for name, support in (("train_half0", ["support_0"]), ("train_half1", ["support_1"]),
                          ("enrollment", ["support_0", "support_1"])):
        directory = output / "fold0/sidtd/example" / name
        directory.mkdir(parents=True)
        products = {}
        for product in PRODUCTS:
            path = directory / f"{product}.png"
            path.write_bytes(f"not decoded, {name}, {product}".encode())
            products[product] = {"path": str(path), "sha256": digest(path)}
        geometry = {"schema_version": "anchor_enrollment_v1", "geometry_version": "bank_similarity_v1", "fold": 0,
                    "design_id": "sidtd/example", "bank": name, "anchor_id": support[0], "canvas_hw": [512, 800],
                    "axes": "x=column,y=row; native_to_bank=canvas_to_bank@native_to_canvas",
                    "support_ids": support, "registered_ids": support,
                    "contributors": {identifier: contributors[identifier] for identifier in support}, "source": source,
                    "official_split_sha256": "d" * 64, "valid_interpolation": "nearest", "rgb_interpolation": "linear",
                    "registrations": {identifier: {"anchor_applied": True, "anchor_inliers": None, "anchor_transform": identity,
                                                     "target_applied": True, "target_inliers": 40,
                                                     "canvas_to_bank": identity, "native_to_bank": identity} for identifier in support}}
        path = directory / "geometry.json"
        write_json(path, geometry)
        products["geometry"] = {"path": str(path), "sha256": digest(path)}
        dependency = {**geometry, "output_sha256": {key: entry["sha256"] for key, entry in products.items()}}
        path = directory / "manifest.json"
        write_json(path, dependency)
        products["manifest"] = {"path": str(path), "sha256": digest(path)}
        banks[name] = {"bank": name, "design_id": "sidtd/example", "fold": 0, "geometry_version": "bank_similarity_v1",
                       "support_ids": support, "registered_ids": support, "component_ids": support,
                       "contributors": geometry["contributors"], "products": products,
                       "counts": {"authorized": len(support), "registered": len(support), "failed": 0}}
    manifest = {"schema_version": "anchor_enrollment_v1", "geometry_version": "bank_similarity_v1", "fold": 0,
                "official_fold_binding": {"protocol": "A", "fold": 0, "ref_group": "it0"},
                "classifier_counts": {"train": 1778, "val": 222, "test": 222},
                "official_split_hash_format": "canonical_json_rows_v1", "official_split_sha256": "d" * 64,
                "output_root": str(output), "source": source, "contributors": contributors,
                "eligible_ids": list(contributors), "counts": {"designs": 1, "banks": 3, "eligible": 2},
                "banks": {"sidtd/example": banks}}
    path = output / "manifest.json"
    write_json(path, manifest)
    return path, manifest


def verify(verifier, bundle, **kwargs):
    path, manifest = bundle
    write_json(path, manifest)
    return verifier.verify_products(path, digest(path), **kwargs)


def set_bundle_fold(bundle, fold):
    path, manifest = bundle
    if fold:
        (path.parent / "fold0").rename(path.parent / f"fold{fold}")
    manifest["fold"] = fold
    manifest["official_fold_binding"] = dict(protocol="A", fold=fold, ref_group=f"it{fold}")
    manifest["classifier_counts"] = {4: dict(train=1776, val=223, test=223),
                                     9: dict(train=1777, val=222, test=223)}.get(fold, dict(train=1778, val=222, test=222))
    for bank in manifest["banks"]["sidtd/example"].values():
        bank["fold"] = fold
        for product in bank["products"].values():
            relative = Path(product["path"]).relative_to(path.parent / "fold0")
            product["path"] = str(path.parent / f"fold{fold}" / relative)
        geometry_path = Path(bank["products"]["geometry"]["path"])
        geometry = json.loads(geometry_path.read_text())
        geometry["fold"] = fold
        write_json(geometry_path, geometry)
        bank["products"]["geometry"]["sha256"] = digest(geometry_path)
        dependency_path = Path(bank["products"]["manifest"]["path"])
        write_json(dependency_path, dict(geometry, output_sha256={name: entry["sha256"]
                   for name, entry in bank["products"].items() if name != "manifest"}))
        bank["products"]["manifest"]["sha256"] = digest(dependency_path)
    write_json(path, manifest)


@pytest.mark.parametrize("fold", range(1, 10))
def test_explicit_nonzero_fold_verifies_all_bank_products(verifier, bundle, fold):
    set_bundle_fold(bundle, fold)
    result = verify(verifier, bundle, expected_fold=fold)
    assert result["passed"], result["errors"]
    assert result["expected_fold"] == fold
    assert result["counts"]["products_matched"] == 27
    assert result["counts"]["bank_json_pairs_checked"] == 3
    assert result["pixels_decoded"] is False and result["raw_or_query_files_opened"] is False


@pytest.mark.parametrize("actual,expected", [(1, 0), (0, 1), (4, 9), (9, 4)])
def test_wrong_expected_fold_fails_before_product_reads(verifier, bundle, monkeypatch, actual, expected):
    set_bundle_fold(bundle, actual)
    original = verifier._read_file
    def manifest_only(path, **kwargs):
        assert path == bundle[0], "wrong-fold bank products must not be opened"
        return original(path, **kwargs)
    monkeypatch.setattr(verifier, "_read_file", manifest_only)
    result = verify(verifier, bundle, expected_fold=expected)
    assert not result["passed"] and result["counts"]["products_hashed"] == 0


@pytest.mark.parametrize("fold", [True, False, -1, 10, 1.0, "1", None])
def test_invalid_expected_fold_rejected_before_file_reads(verifier, bundle, monkeypatch, fold):
    monkeypatch.setattr(verifier, "_read_file", lambda *args, **kwargs: pytest.fail("invalid fold opened files"))
    with pytest.raises(ValueError, match="expected_fold"):
        verifier.verify_products(bundle[0], digest(bundle[0]), expected_fold=fold)


@pytest.mark.parametrize("change", ["bank", "geometry", "dependency", "path", "binding", "counts"])
def test_nonzero_fold_checks_every_fold_binding(verifier, bundle, change):
    set_bundle_fold(bundle, 4)
    manifest = bundle[1]
    bank = manifest["banks"]["sidtd/example"]["enrollment"]
    if change == "bank": bank["fold"] = 0
    elif change == "binding": manifest["official_fold_binding"]["fold"] = 0
    elif change == "counts": manifest["classifier_counts"] = dict(train=1778, val=222, test=222)
    elif change == "path":
        bank["products"]["median"]["path"] = str(bundle[0].parent / "fold0/sidtd/example/enrollment/median.png")
    else:
        target = Path(bank["products"][change if change == "geometry" else "manifest"]["path"])
        document = json.loads(target.read_text())
        document["fold"] = 0
        write_json(target, document)
        bank["products"][change if change == "geometry" else "manifest"]["sha256"] = digest(target)
        if change == "geometry":
            target = Path(bank["products"]["manifest"]["path"])
            document = json.loads(target.read_text())
            document["fold"] = 0
            document["output_sha256"]["geometry"] = bank["products"]["geometry"]["sha256"]
            write_json(target, document)
            bank["products"]["manifest"]["sha256"] = digest(target)
    result = verify(verifier, bundle, expected_fold=4)
    assert not result["passed"]


@pytest.mark.parametrize("fold", [False, 0.0])
def test_dependency_fold_requires_an_explicit_integer(verifier, bundle, fold):
    bank = bundle[1]["banks"]["sidtd/example"]["enrollment"]
    dependency_path = Path(bank["products"]["manifest"]["path"])
    dependency = json.loads(dependency_path.read_text())
    dependency["fold"] = fold
    write_json(dependency_path, dependency)
    bank["products"]["manifest"]["sha256"] = digest(dependency_path)
    result = verify(verifier, bundle)
    assert not result["passed"]
    assert any(error["code"] == "dependency_fold" for error in result["errors"])


def test_nonzero_fold_cli_and_relocated_product_paths(verifier, bundle, tmp_path):
    import shutil
    set_bundle_fold(bundle, 9)
    relocated = tmp_path / "relocated"
    shutil.copytree(bundle[0].parent, relocated)
    report = tmp_path / "report.json"
    result = subprocess.run([sys.executable, str(SCRIPT), "--manifest", str(bundle[0]),
                            "--manifest-sha256", digest(bundle[0]), "--expected-fold", "9",
                            "--output-root", str(relocated), "--report", str(report)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(report.read_text())["expected_fold"] == 9


@pytest.mark.parametrize("fold", ["-1", "10", "True", "1.0"])
def test_cli_invalid_expected_fold_rejected_before_manifest_reads(verifier, tmp_path, fold):
    report = tmp_path / "report.json"
    with pytest.raises(SystemExit) as error:
        verifier.main(["--manifest", str(tmp_path / "missing.json"), "--manifest-sha256", "a" * 64,
                       "--report", str(report), "--expected-fold", fold])
    assert error.value.code == 2 and not report.exists()


def test_verifies_every_product_and_json_without_decoding_or_mutation(verifier, bundle):
    path, manifest = bundle
    protected = {product["path"]: digest(Path(product["path"])) for bank in manifest["banks"]["sidtd/example"].values()
                 for product in bank["products"].values()}
    result = verify(verifier, bundle)
    assert result["passed"]
    assert result["errors"] == []
    assert result["counts"]["banks_checked"] == 3
    assert result["counts"]["products_declared"] == 27
    assert result["counts"]["products_hashed"] == 27
    assert result["counts"]["products_matched"] == 27
    assert result["counts"]["bank_json_pairs_checked"] == 3
    assert result["counts"]["registered_supports"] == 4
    assert result["detector_clearance"] is False
    assert result["pixels_decoded"] is False
    assert protected == {filename: digest(Path(filename)) for filename in protected}


def test_manifest_authority_mismatch_never_opens_products(verifier, bundle, monkeypatch):
    path, manifest = bundle
    expected = digest(path)
    manifest["counts"]["eligible"] = 999
    write_json(path, manifest)
    real_open = Path.open

    def protected_open(self, *args, **kwargs):
        assert self == path, "product opened under modified manifest authority"
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", protected_open)
    result = verifier.verify_products(path, expected)
    assert not result["passed"]
    assert result["counts"]["products_hashed"] == 0
    assert any(error["code"] == "manifest_sha256" for error in result["errors"])


def test_reports_multiple_failures_and_continues_all_banks(verifier, bundle):
    banks = bundle[1]["banks"]["sidtd/example"]
    Path(banks["train_half0"]["products"]["median"]["path"]).write_bytes(b"corruption")
    Path(banks["train_half1"]["products"]["target"]["path"]).unlink()
    Path(banks["enrollment"]["products"]["zone"]["path"]).write_bytes(b"corruption")
    result = verify(verifier, bundle)
    assert not result["passed"]
    assert result["counts"]["banks_checked"] == 3
    assert result["counts"]["products_hashed"] == 26
    assert result["counts"]["products_matched"] == 24
    assert result["counts"]["products_missing"] == 1
    assert sum(error["code"] == "product_sha256" for error in result["errors"]) == 2
    assert result["error_count"] == len(result["errors"])


@pytest.mark.parametrize("change", ["fold", "boolean_fold", "schema", "geometry_version", "binding", "classifier_counts", "empty_banks"])
def test_unauthorized_manifest_fails_before_bank_reads(verifier, bundle, change):
    manifest = bundle[1]
    if change == "binding":
        manifest["official_fold_binding"]["ref_group"] = "it1"
    elif change == "classifier_counts":
        manifest["classifier_counts"]["test"] = 0
    elif change == "empty_banks":
        manifest["banks"] = {}
    else:
        manifest[{"schema": "schema_version", "boolean_fold": "fold"}.get(change, change)] = {
            "fold": 1, "boolean_fold": False, "schema": "legacy", "geometry_version": "legacy"}[change]
    result = verify(verifier, bundle)
    assert not result["passed"]
    assert result["counts"]["products_hashed"] == 0


@pytest.mark.parametrize("change", ["outside", "dotdot", "symlink", "parent_symlink", "wrong_bank_path", "relative"])
def test_unsafe_product_paths_are_not_read(verifier, bundle, tmp_path, change):
    bank = bundle[1]["banks"]["sidtd/example"]["train_half0"]
    product = bank["products"]["median"]
    original = Path(product["path"])
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"must not be read")
    if change == "outside":
        product["path"] = str(outside)
    elif change == "dotdot":
        product["path"] = str(original.parent / ".." / original.parent.name / original.name)
    elif change == "relative":
        product["path"] = "median.png"
    elif change == "wrong_bank_path":
        product["path"] = bundle[1]["banks"]["sidtd/example"]["enrollment"]["products"]["median"]["path"]
    elif change == "symlink":
        original.unlink()
        original.symlink_to(outside)
    else:
        relocated = original.parent.with_name("moved")
        original.parent.rename(relocated)
        original.parent.symlink_to(relocated, target_is_directory=True)
    result = verify(verifier, bundle)
    assert not result["passed"]
    assert any(error["code"] == "product_path" for error in result["errors"])
    target = next(record for record in result["products"] if record["bank"] == "train_half0" and record["product"] == "median")
    assert target["actual_sha256"] is None


@pytest.mark.parametrize("change", ["dependency_hash", "self_hash", "dependency_geometry", "geometry_bank", "geometry_membership",
                                    "geometry_source", "geometry_split", "geometry_canvas", "registered", "counts", "missing_product", "malformed_json"])
def test_json_and_manifest_inconsistency_rejected_even_with_updated_file_hash(verifier, bundle, change):
    bank = bundle[1]["banks"]["sidtd/example"]["enrollment"]
    geometry_path = Path(bank["products"]["geometry"]["path"])
    dependency_path = Path(bank["products"]["manifest"]["path"])
    geometry = json.loads(geometry_path.read_text())
    dependency = json.loads(dependency_path.read_text())
    if change == "dependency_hash":
        dependency["output_sha256"]["zone"] = "0" * 64
    elif change == "self_hash":
        dependency["output_sha256"]["manifest"] = bank["products"]["manifest"]["sha256"]
    elif change == "dependency_geometry":
        dependency["anchor_id"] = "other"
    elif change == "registered":
        bank["registered_ids"] = ["unauthorized"]
    elif change == "counts":
        bank["counts"]["registered"] = 999
    elif change == "missing_product":
        del bank["products"]["median"]
    elif change.startswith("geometry_"):
        field, value = {"geometry_bank": ("bank", "train_half0"), "geometry_membership": ("support_ids", ["wrong"]),
                        "geometry_source": ("source", {}), "geometry_split": ("official_split_sha256", "0" * 64),
                        "geometry_canvas": ("canvas_hw", [1, 1])}[change]
        geometry[field] = value
        dependency[field] = value
        write_json(geometry_path, geometry)
        bank["products"]["geometry"]["sha256"] = digest(geometry_path)
        dependency["output_sha256"]["geometry"] = digest(geometry_path)
    if change == "malformed_json":
        dependency_path.write_bytes(b"not JSON")
    else:
        write_json(dependency_path, dependency)
    bank["products"]["manifest"]["sha256"] = digest(dependency_path)
    result = verify(verifier, bundle)
    assert not result["passed"]
    assert result["errors"]
    assert result["counts"]["banks_checked"] == 3


def test_exact_original_root_can_be_explicitly_relocated(verifier, bundle):
    path, manifest = bundle
    original = Path("/cluster/frozen/enrollment")
    local = Path(manifest["output_root"])
    for bank in manifest["banks"]["sidtd/example"].values():
        for product in bank["products"].values():
            product["path"] = str(original / Path(product["path"]).relative_to(local))
    manifest["output_root"] = str(original)
    result = verify(verifier, bundle, output_root=local)
    assert result["passed"]
    assert result["declared_output_root"] == str(original)
    assert result["verification_output_root"] == str(local)


def test_cli_writes_separate_report_and_failure_exit(verifier, bundle, tmp_path):
    path, manifest = bundle
    report = tmp_path / "reports/integrity.json"
    command = [sys.executable, str(SCRIPT), "--manifest", str(path), "--manifest-sha256", digest(path), "--report", str(report)]
    completed = subprocess.run(command, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    result = json.loads(report.read_text())
    assert result["passed"] and result["detector_clearance"] is False
    before = digest(path)
    report.unlink()
    Path(manifest["banks"]["sidtd/example"]["enrollment"]["products"]["median"]["path"]).unlink()
    completed = subprocess.run(command, capture_output=True, text=True)
    assert completed.returncode == 1, completed.stderr
    assert not json.loads(report.read_text())["passed"]
    assert digest(path) == before


@pytest.mark.parametrize("destination", ["manifest", "product", "inside_output", "existing_report", "symlink_report", "hardlink_report"])
def test_cli_never_overwrites_protected_or_existing_paths(verifier, bundle, tmp_path, destination):
    path, manifest = bundle
    report = tmp_path / "report.json"
    if destination == "manifest":
        report = path
    elif destination == "product":
        report = Path(manifest["banks"]["sidtd/example"]["enrollment"]["products"]["median"]["path"])
    elif destination == "inside_output":
        report = path.parent / "new_report.json"
    elif destination == "existing_report":
        report.write_bytes(b"keep")
    elif destination == "symlink_report":
        report.symlink_to(path)
    else:
        report.hardlink_to(path)
    before = digest(path)
    existing = report.read_bytes() if report.exists() else None
    completed = subprocess.run([sys.executable, str(SCRIPT), "--manifest", str(path), "--manifest-sha256", digest(path),
                                "--report", str(report)], capture_output=True, text=True)
    assert completed.returncode != 0
    assert digest(path) == before
    if existing is not None:
        assert report.read_bytes() == existing
    else:
        assert not report.exists()


def test_cli_requires_external_manifest_authority(verifier, bundle, tmp_path):
    completed = subprocess.run([sys.executable, str(SCRIPT), "--manifest", str(bundle[0]), "--report", str(tmp_path / "report")],
                               capture_output=True, text=True)
    assert completed.returncode != 0
    assert not (tmp_path / "report").exists()


@pytest.mark.parametrize("field,value", [("official_fold_binding", None), ("official_fold_binding", []),
                                         ("eligible_ids", None), ("banks", []), ("counts", None)])
def test_malformed_metadata_returns_failure_report_not_exception(verifier, bundle, field, value):
    bundle[1][field] = value
    result = verify(verifier, bundle)
    assert not result["passed"]
    assert result["error_count"] == len(result["errors"])


def test_report_tree_protection_survives_wrong_manifest_authority(verifier, bundle, tmp_path):
    original, manifest = bundle
    relocated_manifest = tmp_path / "inputs/manifest.json"
    relocated_manifest.parent.mkdir()
    write_json(relocated_manifest, manifest)
    report = original.parent / "must_not_create.json"
    completed = subprocess.run([sys.executable, str(SCRIPT), "--manifest", str(relocated_manifest),
                                "--manifest-sha256", "0" * 64, "--report", str(report)], capture_output=True, text=True)
    assert completed.returncode != 0
    assert not report.exists()


def test_missing_geometry_does_not_hide_other_product_failures(verifier, bundle):
    bank = bundle[1]["banks"]["sidtd/example"]["train_half0"]
    Path(bank["products"]["geometry"]["path"]).unlink()
    Path(bank["products"]["median"]["path"]).write_bytes(b"corrupted")
    result = verify(verifier, bundle)
    assert not result["passed"]
    assert result["counts"]["products_hashed"] == 26
    assert any(error["code"] == "bank_json_unavailable" for error in result["errors"])
    assert any(error["code"] == "product_sha256" for error in result["errors"])


def test_file_reads_are_limited_to_authenticated_manifest_and_declared_bank_outputs(verifier, bundle, monkeypatch):
    path, manifest = bundle
    allowed = {path} | {Path(product["path"]) for bank in manifest["banks"]["sidtd/example"].values()
                        for product in bank["products"].values()}
    opened = []
    real_open = verifier.os.open

    def bounded_open(filename, flags, *args, **kwargs):
        opened.append(Path(filename))
        assert Path(filename) in allowed
        return real_open(filename, flags, *args, **kwargs)

    monkeypatch.setattr(verifier.os, "open", bounded_open)
    result = verify(verifier, bundle)
    assert result["passed"]
    assert set(opened) == allowed
    assert len(opened) == 28


@pytest.mark.parametrize("alias_kind", ["output_root", "manifest_parent"])
def test_report_cannot_enter_real_output_tree_through_input_alias(verifier, bundle, tmp_path, alias_kind):
    original, manifest = bundle
    alias = tmp_path / "alias"
    alias.symlink_to(original.parent, target_is_directory=True)
    if alias_kind == "output_root":
        local = original.parent
        manifest["output_root"] = str(alias)
        for bank in manifest["banks"]["sidtd/example"].values():
            for product in bank["products"].values():
                product["path"] = str(alias / Path(product["path"]).relative_to(local))
        manifest_path = tmp_path / "inputs/manifest.json"
        manifest_path.parent.mkdir()
        write_json(manifest_path, manifest)
        authority = digest(manifest_path)
    else:
        manifest_path = alias / original.name
        authority = digest(original)
    report = original.parent / "must_not_write.json"
    completed = subprocess.run([sys.executable, str(SCRIPT), "--manifest", str(manifest_path),
                                "--manifest-sha256", authority, "--report", str(report)], capture_output=True, text=True)
    assert completed.returncode != 0
    assert not report.exists()


def test_report_parent_traversal_cannot_enter_protected_output_tree(verifier, bundle, tmp_path):
    original, manifest = bundle
    outside = tmp_path / "outside"
    outside.mkdir()
    report = outside / ".." / original.parent.name / "must_not_write.json"
    completed = subprocess.run([sys.executable, str(SCRIPT), "--manifest", str(original),
                                "--manifest-sha256", digest(original), "--report", str(report)],
                               capture_output=True, text=True)
    assert completed.returncode != 0
    assert not (original.parent / "must_not_write.json").exists()


def test_report_normalization_does_not_hide_symlink_before_parent_traversal(verifier, bundle, tmp_path):
    original, manifest = bundle
    real = tmp_path / "real"
    real.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    report = alias / ".." / "must_not_write.json"
    completed = subprocess.run([sys.executable, str(SCRIPT), "--manifest", str(original),
                                "--manifest-sha256", digest(original), "--report", str(report)],
                               capture_output=True, text=True)
    assert completed.returncode != 0
    assert not (tmp_path / "must_not_write.json").exists()


def test_safe_report_parent_traversal_writes_only_normalized_destination(verifier, bundle, tmp_path):
    original, manifest = bundle
    outside = tmp_path / "outside"
    outside.mkdir()
    report = outside / ".." / "reports" / "integrity.json"
    completed = subprocess.run([sys.executable, str(SCRIPT), "--manifest", str(original),
                                "--manifest-sha256", digest(original), "--report", str(report)],
                               capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    assert json.loads((tmp_path / "reports/integrity.json").read_text())["passed"]
