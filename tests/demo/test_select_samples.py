# tests/demo/test_select_samples.py
import sys
from collections import Counter
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "demo"))
from select_samples import DESIGNS, FAMILIES_OF, eligible_counts, select_mismatch, select_samples  # noqa: E402


def test_counts_follow_the_paper_population_and_are_deterministic():
    a, b = select_samples(), select_samples()
    assert a == b
    assert len(DESIGNS) == 18
    per, avail = Counter((s["design_id"], s["family"]) for s in a), eligible_counts()
    for d, meta in DESIGNS.items():
        for f in FAMILIES_OF[meta["dataset"]]:
            assert per[(d, f)] == min(1, avail.get((d, f), 0)), (d, f)
    assert per[("sidtd_fin_id", "text_copy")] == 0
    assert len(a) == 61
    assert len({s["image_id"] for s in a}) == len(a)


def test_labels_ids_and_paper_logits():
    for s in select_samples():
        assert s["label"] == (0 if s["family"] == "genuine" else 1)
        on = s["paper"]["on"]
        assert abs(on["query_logit"] + on["correction_logit"] - on["attack_logit"]) < 1e-3
        assert s["sample_id"].startswith(f"{s['design_id']}__{s['family']}__")


def test_mismatch_samples():
    m = select_mismatch(select_samples())
    assert [s["design_id"] for s in m] == ["sidtd_esp_id", "sidtd_srb_passport", "idnet_AZ"]
    assert all(s["family"] == "genuine" and s["mismatch_demo"] for s in m)
