"""Training and test samples as (page, reference, zone mask, label, mask).

Genuine pages pair with a reference built without their identity. Wrong-design pairs are added as hard
negatives labelled altered, so the network fails safe.
"""


def build_pairs(records, references, wrong_design_fraction=0.1):
    raise NotImplementedError
