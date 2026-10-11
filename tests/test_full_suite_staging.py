import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def staging():
    spec = importlib.util.spec_from_file_location("full_suite_staging", ROOT / "scripts/stage_full_suite_metadata.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def repository(tmp_path):
    root = tmp_path / "repository"
    metadata = ["data/processed/records.csv", "data/processed/splits/protocol_a_it0.csv"]
    metadata += [f"data/raw/sidtd/kfold_split_templates/split_kfold/{role}_split_SIDTD_it_{fold}.csv"
                 for fold in range(10) for role in ("train", "val", "test")]
    for relative in metadata:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("image_id,label\nfixture,0\n")
    images = root / "data/raw/sidtd/templates/Images"
    images.mkdir(parents=True)
    (images / "fixture.jpg").write_bytes(b"image payload must not be read or copied")
    return root


def test_stages_required_collection_metadata_and_no_pixel_bytes(staging, repository, tmp_path):
    code = tmp_path / "scratch/code"
    code.mkdir(parents=True)
    manifest = tmp_path / "metadata.json"
    staging.stage(repository, code, manifest)
    result = json.loads(manifest.read_text())
    assert len(result["metadata_files"]) == 32
    for entry in result["metadata_files"]:
        original = repository / entry["relative_path"]
        staged = code / entry["relative_path"]
        assert original.read_bytes() == staged.read_bytes()
        assert hashlib.sha256(staged.read_bytes()).hexdigest() == entry["sha256"]
    images = code / "data/raw/sidtd/templates/Images"
    assert images.is_symlink()
    assert images.resolve() == (repository / "data/raw/sidtd/templates/Images").resolve()
    assert all(entry["relative_path"].endswith(".csv") for entry in result["metadata_files"])


def test_missing_metadata_stops_before_partial_staging(staging, repository, tmp_path):
    (repository / "data/processed/records.csv").unlink()
    code = tmp_path / "code"
    code.mkdir()
    with pytest.raises(FileNotFoundError):
        staging.stage(repository, code, tmp_path / "metadata.json")
    assert not (code / "data").exists()


def test_rejects_repository_mutation_and_existing_staging(staging, repository, tmp_path):
    with pytest.raises(ValueError):
        staging.stage(repository, repository, tmp_path / "metadata.json")
    with pytest.raises(ValueError):
        staging.stage(repository, repository / "scratch/code", tmp_path / "metadata.json")
    code = tmp_path / "code"
    (code / "data/processed").mkdir(parents=True)
    (code / "data/processed/records.csv").write_text("existing")
    with pytest.raises(FileExistsError):
        staging.stage(repository, code, tmp_path / "metadata.json")
    assert (code / "data/processed/records.csv").read_text() == "existing"
