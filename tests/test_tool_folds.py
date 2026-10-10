"""Tools outside training pass the fold on: python tests/test_tool_folds.py"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import bench_cpu as B  # noqa: E402
from refer_or_clear import evaluate as E  # noqa: E402


class _Stop(Exception):
    pass


class _Model:
    def eval(self):
        return self

    def float(self):
        return self


def _ck(fold):
    return {"args": {"arch": "changerex_r18", "fusion": "single", "seed": 42, "reference": "median"},
            "config": {"protocol": "A", "fold": fold}}


def test_checkpoint_fold():
    assert E.checkpoint_fold(_ck(4)) == 4
    assert E.checkpoint_fold(_ck(4), 7) == 7
    assert E.checkpoint_fold({"args": {}}) is None


def test_bench_cpu_passes_the_checkpoint_fold():
    seen = {}

    def fake_split(cfg, root, config_path=None, fold=None):
        seen["fold"] = fold
        raise _Stop
    orig = (B.E.load_checkpoint, B.E.load_split)
    B.E.load_checkpoint, B.E.load_split = (lambda path, device: (_Model(), _ck(4))), fake_split
    try:
        B.main(["--checkpoint", "x/best.pt", "--config", str(ROOT / "configs" / "protocol_a.yaml")])
    except _Stop:
        pass
    finally:
        B.E.load_checkpoint, B.E.load_split = orig
    assert seen.get("fold") == 4, seen


def test_smoke_script_passes_a_fold():
    text = (ROOT / "scripts" / "smoke_test.sh").read_text()
    assert "--fold 0" in text, "smoke_test.sh trains Protocol A without a fold"


if __name__ == "__main__":
    test_checkpoint_fold()
    test_bench_cpu_passes_the_checkpoint_fold()
    test_smoke_script_passes_a_fold()
    print("ok")
