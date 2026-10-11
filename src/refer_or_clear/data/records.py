"""Unified records for the public datasets.

One row per image with the same columns for every source, so references, masks, splits and the
training dataset never need to know which archive an image came from.

Columns
    dataset        sidtd | fantasyid | idnet
    image_id       unique, filesystem-safe (canvas and mask file names)
    path           source image, relative to the data root
    design_id      sidtd/alb_id, fantasyid/arabic, idnet/AZ
    identity_id    the document: sidtd/alb_id_27, fantasyid/arabic-003_03 (shared across devices)
    label          0 genuine, 1 altered
    attack_family  text_inpaint | text_copy | face_swap | face_morph | mixed | '' for genuine
    attack_source  raw group name from the release (ctype or folder)
    capture        template | scan | huawei | iphone15pro | iphone15 | render
    source_id      identity the forgery was made from ('' for genuine)
    donor_id       second document for copy-and-replace ('' otherwise)
    altered_fields comma-separated field names marked altered ('' for genuine)
    annotation     JSON with the field rectangles, relative to the data root
    width, height  native image size

Verified 10 Oct 2026. SIDTD: templates/Images/reals/{design}_{NN}.jpg, 100 per design, one size per
design; fakes {design}_{NN}_fake_{k}_{n}.jpg with a JSON naming ctype, src, second_src, field; genuine
field rectangles in one VIA file per design keyed by '{NN}.jpg<size>' with 14 regions.
FantasyID: images are already document crops; rectangles are in shipped coordinates; the card
basename is the identity and is shared across the three capture folders.
"""
from __future__ import annotations

import json
import os
import struct
from pathlib import Path

import pandas as pd

SIDTD_DESIGNS = ["alb_id", "aze_passport", "esp_id", "est_id", "fin_id", "grc_passport",
                 "lva_passport", "rus_internalpassport", "srb_passport", "svk_id"]
SIDTD_CTYPE = {"Inpaint_and_Rewrite": "text_inpaint", "Crop_and_Replace": "text_copy"}

FANTASY_GROUPS = {  # folder -> (label, attack_family, attack_source)
    "train/bonafide": (0, "", "bonafide"),
    "test/bonafide": (0, "", "bonafide"),
    "train/attack/digital_1": (1, "mixed", "digital_1 (InSwapper face + DiffSTE text)"),
    "train/attack/digital_2": (1, "mixed", "digital_2 (FaceDancer face + TextDiffuser2 text)"),
    "test/attack/digital_3": (1, "text_inpaint", "attack1 digital_3 (fine-tuned TextDiffuser2)"),
    "test/attack/facedancer": (1, "face_swap", "attack2 facedancer"),
    "test/attack/textdiffuserft_bfei": (1, "text_inpaint", "attack3 textdiffuserft_bfei"),
}


def jpeg_size(path: Path) -> tuple[int, int]:
    """(width, height) from the JPEG header without decoding the image."""
    b = path.read_bytes()[:400000]
    i = 2
    while i < len(b) - 9:
        if b[i] != 0xFF:
            i += 1
            continue
        marker = b[i + 1]
        if marker in (0xC0, 0xC1, 0xC2):
            h, w = struct.unpack(">HH", b[i + 5:i + 9])
            return w, h
        i += 2 + struct.unpack(">H", b[i + 2:i + 4])[0]
    raise ValueError(f"no SOF marker in {path}")


def sidtd_target_field(annotation: dict) -> str:
    """The recipient field is altered; second_field describes the donor document."""
    field = annotation.get("field")
    if not isinstance(field, str) or field in ("", "None"):
        raise ValueError("SIDTD annotation has no recipient field")
    return field


def sidtd_records(root: Path) -> pd.DataFrame:
    base = root / "sidtd" / "templates"
    rows = []
    size_cache: dict[str, tuple[int, int]] = {}
    for name in sorted(os.listdir(base / "Images" / "reals")):
        if not name.endswith(".jpg"):
            continue
        design, nn = name[:-4].rsplit("_", 1)
        if design not in size_cache:
            size_cache[design] = jpeg_size(base / "Images" / "reals" / name)
        w, h = size_cache[design]
        rows.append(dict(dataset="sidtd", image_id=f"sidtd_{name[:-4]}", path=f"sidtd/templates/Images/reals/{name}",
                         design_id=f"sidtd/{design}", identity_id=f"sidtd/{design}_{nn}", label=0,
                         attack_family="", attack_source="", capture="template", source_id="", donor_id="",
                         altered_fields="", annotation=f"sidtd/templates/Annotations/reals/{design}.json",
                         width=w, height=h))
    for name in sorted(os.listdir(base / "Images" / "fakes")):
        if not name.endswith(".jpg"):
            continue
        stem = name[:-4]
        meta = json.load(open(base / "Annotations" / "fakes" / f"{stem}.json"))
        src = meta["src"][:-4]
        design = src.rsplit("_", 1)[0]
        donor = meta.get("second_src") or "None"
        donor_id = "" if donor in ("None", "", None) else f"sidtd/{donor[:-4]}"
        if design not in size_cache:
            size_cache[design] = jpeg_size(base / "Images" / "fakes" / name)
        w, h = size_cache[design]
        field = sidtd_target_field(meta)
        rows.append(dict(dataset="sidtd", image_id=f"sidtd_{stem}", path=f"sidtd/templates/Images/fakes/{name}",
                         design_id=f"sidtd/{design}", identity_id=f"sidtd/{src}", label=1,
                         attack_family=SIDTD_CTYPE.get(meta["ctype"], meta["ctype"]), attack_source=meta["ctype"],
                         capture="template", source_id=f"sidtd/{src}", donor_id=donor_id,
                         altered_fields=field, annotation=f"sidtd/templates/Annotations/fakes/{stem}.json",
                         width=w, height=h))
    return pd.DataFrame(rows)


def fantasyid_records(root: Path) -> pd.DataFrame:
    base = root / "fantasyid"
    rows = []
    for group, (label, family, source) in FANTASY_GROUPS.items():
        gdir = base / group
        if not gdir.is_dir():
            continue
        for device in sorted(os.listdir(gdir)):
            ddir = gdir / device
            if not ddir.is_dir():
                continue
            for name in sorted(os.listdir(ddir)):
                if not name.lower().endswith((".jpg", ".png")):
                    continue
                stem = name.rsplit(".", 1)[0]
                design = stem.split("-", 1)[0]
                ann = ddir / f"{stem}.json"
                altered = ""
                if ann.is_file():
                    j = json.load(open(ann))
                    altered = ",".join(sorted({r["region_attributes"].get("field_name", "")
                                               for r in j.get("regions", [])
                                               if r["region_attributes"].get("region_provenance") == "altered"}))
                w, h = jpeg_size(ddir / name)
                rows.append(dict(dataset="fantasyid", image_id=f"fantasyid_{group.replace('/', '_')}_{device}_{stem}",
                                 path=f"fantasyid/{group}/{device}/{name}", design_id=f"fantasyid/{design}",
                                 identity_id=f"fantasyid/{stem}", label=label, attack_family=family,
                                 attack_source=source, capture=device,
                                 source_id=f"fantasyid/{stem}" if label else "", donor_id="",
                                 altered_fields=altered,
                                 annotation=f"fantasyid/{group}/{device}/{stem}.json" if ann.is_file() else "",
                                 width=w, height=h))
    return pd.DataFrame(rows)


def build_records(root: Path, datasets=("sidtd", "fantasyid")) -> pd.DataFrame:
    parts = []
    if "sidtd" in datasets and (root / "sidtd" / "templates").is_dir():
        parts.append(sidtd_records(root))
    if "fantasyid" in datasets and (root / "fantasyid").is_dir():
        parts.append(fantasyid_records(root))
    df = pd.concat(parts, ignore_index=True)
    assert df.image_id.is_unique, "image_id must be unique"
    return df


def provenance_components(df: pd.DataFrame) -> pd.Series:
    """Union-find over identity, source and donor so a forgery stays with the documents it was made from."""
    parent: dict[str, str] = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for r in df.itertuples(index=False):
        find(r.identity_id)
        for other in (r.source_id, r.donor_id):
            if other:
                union(r.identity_id, other)
    return df.identity_id.map(find)
