"""Design-median reference and zone mask per design (DESIGN.md 3.3).

The reference is the pixel-wise median over the genuine documents of a design after rectification,
built WITHOUT the query's own identity. It keeps the static design and removes personal content.
The zone mask marks personal-data fields (1 inside, 0 outside).
"""


def design_median(pages):
    raise NotImplementedError


def zone_mask(field_rectangles, canvas=(512, 800)):
    raise NotImplementedError


def masked_exemplar(page, field_rectangles, median):
    """Robustness reference: one genuine document of another identity, personal zones filled from the median."""
    raise NotImplementedError
