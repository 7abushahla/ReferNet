# tests/demo/test_build_site_data.py
import csv, json, sys
from pathlib import Path
import pytest
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "demo"))
SITE = ROOT / "site"
pytestmark = pytest.mark.skipif(not (SITE / "manifest.json").exists(), reason="run demo/build_site_data.py first")


def load():
    return json.loads((SITE / "manifest.json").read_text())


def test_every_file_exists():
    m = load()
    for d in m["designs"]:
        assert (SITE / d["reference"]).exists() and (SITE / d["zone"]).exists()
    for s in m["samples"]:
        for k in ("query", "reference", "masks"):
            assert (SITE / s[k]).exists(), s[k]


def test_counts_names_and_wording():
    m = load()
    assert len(m["designs"]) == 18 and len(m["samples"]) == 61 and len(m["mismatch"]) == 3
    assert "enrol" not in json.dumps(m).lower()
    assert {s["family_name"] for s in m["samples"]} == {"Genuine", "Inpaint-and-rewrite", "Copy-and-replace",
                                                       "In-place text replacement"}


def test_sidtd_samples_are_test_documents():
    from select_samples import select_samples
    split = ROOT / "data/processed/splits/protocol_a_it0.csv"
    if not split.exists():
        pytest.skip("SIDTD split not present")
    roles = {r["image_id"]: r["role"] for r in csv.DictReader(open(split))}
    for s in select_samples():
        if s["dataset"] == "sidtd":
            assert roles[s["image_id"]] == "test", s["image_id"]


def test_pytorch_scores_present_and_in_range():
    for s in load()["samples"]:
        assert 0.0 <= s["pytorch"]["on"]["p"] <= 1.0 and 0.0 <= s["pytorch"]["off"]["p"] <= 1.0
