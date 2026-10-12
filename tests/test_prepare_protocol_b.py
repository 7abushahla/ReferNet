import importlib.util
import json
from pathlib import Path

import pytest

from test_protocol_b_enrollment import inventory


@pytest.fixture
def preparer():
    script = Path(__file__).resolve().parents[1] / "scripts/prepare_protocol_b.py"
    assert script.is_file(), "Protocol B preparer missing"
    specification = importlib.util.spec_from_file_location("_prepare_b_test", script)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_metadata_preparation_freezes_reservation_before_support_pixels(preparer, inventory, tmp_path, monkeypatch):
    monkeypatch.setattr(preparer, "load_protocol_b_contract", lambda *args, **kwargs: inventory["contract"])
    monkeypatch.setattr(preparer, "build_protocol_b_enrollment_products", lambda *args: pytest.fail("metadata mode opened support pixels"))
    output = tmp_path / "prepared"
    result = preparer.prepare(inventory["records_path"], inventory["contract"].records_sha256,
                              tmp_path / "split.csv", 0, inventory["raw"], output, build=False)
    assert result["protocol"] == "B" and result["phase"] == "metadata"
    assert result["passed"] is True
    target = json.loads((output / "target-enrollment.json").read_text())
    assert target["published_test_population_preserved"] is False
    assert set(target["support_ids"]).isdisjoint(target["query_ids"])
    assert (output / "authorized-enrollment.json").is_file()
    assert not (output / "enrollment").exists()
    assert result["target_enrollment"]["sha256"] == preparer.digest(output / "target-enrollment.json")


def test_existing_preparation_is_never_overwritten(preparer, inventory, tmp_path, monkeypatch):
    monkeypatch.setattr(preparer, "load_protocol_b_contract", lambda *args, **kwargs: inventory["contract"])
    output = tmp_path / "prepared"
    output.mkdir()
    sentinel = output / "previous.txt"
    sentinel.write_text("previous")
    with pytest.raises(FileExistsError):
        preparer.prepare(inventory["records_path"], inventory["contract"].records_sha256,
                         tmp_path / "split.csv", 0, inventory["raw"], output, build=False)
    assert sentinel.read_text() == "previous"
