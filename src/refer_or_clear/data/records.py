"""Unified record per image across MIDV-2020, SIDTD and FantasyID (DESIGN.md 3.2).

Fields: image_path, dataset, design_id (MIDV codes alb_id ... svk_id; FantasyID template names),
identity_id, label (0 genuine, 1 altered), fraud_type (crop_replace, inpaint_rewrite, face_swap,
text_inpaint, plus generator name), capture (render, scan, phone, device), fields (rectangles),
mask_path (altered region, or None).
"""


def build_midv2020(root):
    raise NotImplementedError


def build_sidtd(root, midv_records):
    """Fakes resolve their clean source via the JSON `src` field and the altered field via `field`."""
    raise NotImplementedError


def build_fantasyid(root, layouts):
    """Layouts are annotated once by hand for the 13 designs (data/interim/fantasyid_layouts.json)."""
    raise NotImplementedError
