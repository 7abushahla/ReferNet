import importlib
import hashlib
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from test_anchor_training import TinyPilot, runtime_fixture


@pytest.fixture
def pilots():
    if importlib.util.find_spec("refer_or_clear.pilots") is not None:
        return importlib.import_module("refer_or_clear.pilots")
    class Runtime:
        def __getattr__(self, name):
            assert importlib.util.find_spec("refer_or_clear.pilots") is not None, "pilot replay is not implemented"
            return getattr(importlib.import_module("refer_or_clear.pilots"), name)
    return Runtime()


def test_checkpoint_rejects_schema_architecture_and_data_hashes(pilots, tmp_path):
    config = {"schema_version": "anchor_pilot_v1", "track": "R", "architecture_id": "anchor_rgb_r50_v1"}
    identities = {"records_sha256": "a" * 64, "split_sha256": "b" * 64}
    path = tmp_path / "selected.pt"
    pilots.save_pilot_checkpoint(path, TinyPilot(), config, identities, 1, (.75, .8), "smoke")
    assert pilots.read_pilot_checkpoint(path, config, identities)["phase"] == "smoke"
    with pytest.raises(ValueError, match="hash|identit"):
        pilots.read_pilot_checkpoint(path, config, {**identities, "split_sha256": "c" * 64})
    for key, value in (("schema_version", "legacy"), ("architecture_id", "old_model")):
        with pytest.raises(ValueError, match="schema|architecture|config"):
            pilots.read_pilot_checkpoint(path, {**config, key: value}, identities)


def test_logits_replay_rejects_reorder_missing_ids_and_tolerance_violation(pilots):
    predictions = [{"image_id": "a", "attack_logit": .3}, {"image_id": "b", "attack_logit": -.7}]
    assert pilots.compare_replay(predictions, predictions)["passed"]
    assert not pilots.compare_replay(predictions, predictions[::-1])["passed"]
    assert not pilots.compare_replay(predictions, predictions[:1])["passed"]
    changed = [{"image_id": "a", "attack_logit": .31}, predictions[1]]
    assert not pilots.compare_replay(predictions, changed)["passed"]


def test_real_fresh_process_replay_preserves_checkpoint_and_scores(pilots, runtime_fixture, monkeypatch, tmp_path):
    options, _ = runtime_fixture
    run_dir = tmp_path / "smoke"
    pilots.run_pilot(**options, run_dir=run_dir, phase="smoke")
    protected = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in
                 (run_dir / "train.json", run_dir / "val_predictions.csv", run_dir / "checkpoints" / "best.pt")}
    actual_run = subprocess.run
    bootstrap = """
import sys
from pathlib import Path
from test_anchor_training import FixtureDataset, fixture_rows, TinyPilot
from refer_or_clear import pilots
from refer_or_clear.data import query
def builder(**kwargs):
    model = TinyPilot()
    model.initialization_provenance['checkpoint_sha256'] = kwargs['expected_sha256']
    model.initialization_provenance['checkpoint_path'] = str(kwargs['weights_path'])
    return model
pilots._model_builder = lambda track: builder
pilots.OFFICIAL_WEIGHT_SHA256 = {'R': sys.argv[sys.argv.index('--weights-sha256') + 1]}
query.AnchorQueryDataset = FixtureDataset
query.load_role_rows = lambda records, split, role: fixture_rows(role)
pilots.main(sys.argv[1:])
"""
    def launch(command, **kwargs):
        environment = kwargs["env"].copy()
        environment["PYTHONPATH"] += os.pathsep + str(Path(__file__).parent)
        return actual_run([command[0], "-c", bootstrap, *command[3:]], **{**kwargs, "env": environment})
    monkeypatch.setattr(pilots.subprocess, "run", launch)
    result = pilots.fresh_process_replay(**options, run_dir=run_dir)
    assert result["passed"], result.get("stderr", result)
    assert result["fresh_process"]
    assert result["pid"] != os.getpid()
    assert result["n_images"] == 32
    for path in (run_dir / "train.json", run_dir / "val_predictions.csv", run_dir / "checkpoints" / "best.pt"):
        assert hashlib.sha256(path.read_bytes()).hexdigest() == protected[path.name]
