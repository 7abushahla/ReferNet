"""Inference-only JPEG re-encoding of decoded query images (compression-cue sensitivity analysis).

The published SIDTD files carry a class-linked encoding (genuine IJG Q92, altered IJG Q75).
``reencoded_queries`` temporarily wraps a consumer module's ``_image`` decoder so that every
colour image read from ``raw_root`` (the query files) is re-encoded as JPEG at ``quality`` and
decoded again before any geometry or normalization. Product images (medians, masks) outside
``raw_root`` are untouched. Nothing is written to disk and no training is affected.
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import cv2


@contextmanager
def reencoded_queries(module, raw_root, quality):
    if type(quality) is not int or not 1 <= quality <= 100:
        raise ValueError("JPEG quality must be an integer in [1, 100]")
    original = module._image
    root = Path(raw_root).resolve()

    def wrapped(binding, flags):
        image = original(binding, flags)
        path = Path(binding["path"]).resolve()
        if flags == cv2.IMREAD_COLOR and root in path.parents:
            ok, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
            if not ok:
                raise ValueError(f"cannot re-encode query {binding['path']}")
            image = cv2.imdecode(buffer, flags)
            if image is None:
                raise ValueError(f"cannot decode re-encoded query {binding['path']}")
        return image

    module._image = wrapped
    try:
        yield dict(kind="query_jpeg_reencode", quality=quality, scope="raw_root_color_queries_after_decode_before_canvas",
                   encoder=f"opencv_{cv2.__version__}_imencode_jpg")
    finally:
        module._image = original
