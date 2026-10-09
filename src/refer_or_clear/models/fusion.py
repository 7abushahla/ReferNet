"""Fusion settings compared on identical samples (DESIGN.md 4.1, 4.3).

single     : page only, same encoder and heads (same-capacity baseline)
stacked    : page, reference, zone as 7 channels into one encoder (negative control)
difference : |f(x) - f(r)| concatenated with f(x) and the downsampled zone mask (standard variant)
attention  : self-attention over f(x); cross-attention Q = f(x), K = V = f(r) with the zone mask as a
             logit bias; cross-attention from the global descriptor over fused local features
             (improved variant). Six heads, key dimension 32.
"""


def build_fusion(kind, channels):
    raise NotImplementedError
