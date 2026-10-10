"""Protocol configs: python tests/test_configs.py"""
from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load(letter):
    return yaml.safe_load(open(ROOT / "configs" / f"protocol_{letter}.yaml"))


def test_letters_match_files():
    for letter in "abcd":
        assert load(letter)["protocol"] == letter.upper(), letter


def test_anchor_and_heldout_recipe():
    for letter, split in (("a", "protocol_a_it{fold}.csv"), ("b", "protocol_b_g{fold}.csv")):
        c = load(letter)
        assert Path(c["data"]["split"]).name == split, c["data"]["split"]
        assert c["data"]["folds"] == 10
        assert c["augment"]["jpeg_quality"] is None
        assert "canvas_jpeg_quality" not in c["augment"]
        assert c["train"]["select"] == "accuracy"
        assert c["train"]["seeds"] == [42]
        assert c["train"]["epochs"] == 40
        assert c["eval"]["test_variants"] == ["native", "equalized_qf75"]


def test_fantasyid_and_idnet_moved():
    assert Path(load("c")["data"]["split"]).name == "protocol_c.csv"
    assert "FantasyID" in open(ROOT / "configs" / "protocol_c.yaml").read().splitlines()[0]
    assert Path(load("d")["data"]["split"]).name == "protocol_d.csv"
    assert "IDNet" in open(ROOT / "configs" / "protocol_d.yaml").read().splitlines()[0]


if __name__ == "__main__":
    test_letters_match_files()
    test_anchor_and_heldout_recipe()
    test_fantasyid_and_idnet_moved()
    print("ok")
