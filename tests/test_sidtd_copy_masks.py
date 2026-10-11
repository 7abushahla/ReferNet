"""SIDTD copy targets describe the recipient, not the donor's field location."""
import json
from pathlib import Path
import sys
import tempfile

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from refer_or_clear.data.records import sidtd_records
from refer_or_clear.data.forgeries import altered_rects, build_masks


def fixture(root):
    base = root / "sidtd/templates"
    for directory in ("Images/reals", "Images/fakes", "Annotations/reals", "Annotations/fakes"):
        (base / directory).mkdir(parents=True)
    pixels = np.zeros((512, 800, 3), np.uint8)
    cv2.imwrite(str(base / "Images/reals/fin_id_00.jpg"), pixels)
    cv2.imwrite(str(base / "Images/fakes/fin_id_00_fake_0.jpg"), pixels)
    annotation = dict(src="fin_id_00.jpg", second_src="fin_id_01.jpg", field="name",
                      second_field="gender", ctype="Crop_and_Replace")
    (base / "Annotations/fakes/fin_id_00_fake_0.json").write_text(json.dumps(annotation))
    regions = [dict(shape_attributes=dict(name="rect", x=left, y=20, width=20, height=20),
                    region_attributes=dict(field_name=name)) for name, left in (("name", 20), ("gender", 200))]
    via = dict(_via_img_metadata={"00.jpg": dict(filename="00.jpg", regions=regions)})
    (base / "Annotations/reals/fin_id.json").write_text(json.dumps(via))
    return base


def test_records_keep_only_recipient_field():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        fixture(root)
        records = sidtd_records(root)
        forged = records[records.label == 1].iloc[0]
        assert forged.altered_fields == "name", "donor field incorrectly marked altered on recipient"
        assert forged.donor_id == "sidtd/fin_id_01"
        assert forged.source_id == "sidtd/fin_id_00"


def test_legacy_records_cannot_add_donor_region():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        fixture(root)
        records = sidtd_records(root)
        records.loc[records.label == 1, "altered_fields"] = "name,gender"
        forged = next(records[records.label == 1].itertuples(index=False))
        assert [region[0] for region in altered_rects(root, forged)] == ["name"]
        build_masks(records, root, {forged.image_id: np.eye(3)}, root / "masks")
        mask = cv2.imread(str(root / "masks" / f"{forged.image_id}.png"), cv2.IMREAD_GRAYSCALE)
        assert mask[30, 30] == 255 and mask[30, 210] == 0


def test_missing_target_does_not_fall_back_to_donor():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        base = fixture(root)
        path = base / "Annotations/fakes/fin_id_00_fake_0.json"
        annotation = json.loads(path.read_text())
        annotation["field"] = "None"
        path.write_text(json.dumps(annotation))
        try:
            sidtd_records(root)
        except ValueError:
            pass
        else:
            raise AssertionError("missing recipient field accepted")


if __name__ == "__main__":
    test_records_keep_only_recipient_field()
    test_legacy_records_cannot_add_donor_region()
    test_missing_target_does_not_fall_back_to_donor()
    print("ok: 3 SIDTD recipient-mask tests")
