"""Mask preparation cannot alter source data or include held-out test rows."""
from pathlib import Path
import sys
import tempfile

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def test_development_rows_and_output_safety():
    assert (ROOT / "scripts/prepare_copy_mask_overlay.py").exists(), "overlay preparation missing"
    from prepare_copy_mask_overlay import development_records, validate_output
    records = pd.DataFrame([dict(image_id=name, label=label, dataset="sidtd")
                            for name, label in (("first", 0), ("second", 1), ("heldout", 1))])
    split = pd.DataFrame([dict(image_id=name, label=label, role=role)
                          for name, label, role in (("first", 0, "train"), ("second", 1, "val"), ("heldout", 1, "test"))])
    selected = development_records(records, split)
    assert set(selected.image_id) == {"first", "second"}
    assert set(selected.role) == {"train", "val"}
    split.loc[split.image_id == "second", "label"] = 0
    try:
        development_records(records, split)
    except ValueError:
        pass
    else:
        raise AssertionError("label mismatch accepted")
    with tempfile.TemporaryDirectory() as temporary:
        source = Path(temporary) / "processed"
        source.mkdir()
        for output in (source, source / "new", source.parent):
            try:
                validate_output(source, output)
            except ValueError:
                pass
            else:
                raise AssertionError("overlapping output accepted")
        validate_output(source, Path(temporary) / "overlay")


if __name__ == "__main__":
    test_development_rows_and_output_safety()
    print("ok: development-only mask overlay guards")
