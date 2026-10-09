"""Rectify a page onto its design canvas (512 x 800, ID-1 aspect).

Annotated document quadrilateral where the dataset has one; otherwise ORB/SIFT matching against the
design median, RANSAC homography, ECC refinement. Returns the warped page and the residual error.
"""


def rectify(image, design_median, quad=None):
    raise NotImplementedError
