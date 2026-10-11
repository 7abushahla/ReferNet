"""referral_loss = BCE(logit, label) + mask_weight * (masked BCE + soft Dice)(map inside valid).

The map logit (B, 1, h, w) is upsampled bilinearly to the mask size (B, 1, H, W). BCE on the map is
the mean over valid pixels; Dice treats all valid pixels of the batch as one set (smoothing 1.0), so
all-genuine batches still give a finite, non-trivial term. Wrong-design samples pass the valid region
as their mask (dataset side). Returns (total, {"bce_image", "bce_map", "dice_map"} as floats).
With mask_weight zero, only image BCE is evaluated and localization targets are never read.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def referral_loss(logit, map_logit, label, mask, valid, mask_weight=3.0):
    label = label.reshape(-1).to(logit.dtype)
    bce_image = F.binary_cross_entropy_with_logits(logit.reshape(-1), label)
    if mask_weight == 0:
        return bce_image, {"bce_image": bce_image.item(), "bce_map": 0., "dice_map": 0.}
    if map_logit.shape[-2:] != mask.shape[-2:]:
        map_logit = F.interpolate(map_logit, size=mask.shape[-2:], mode="bilinear", align_corners=False)
    mask, valid = mask.to(map_logit.dtype), valid.to(map_logit.dtype)
    n_valid = valid.sum().clamp(min=1.0)
    bce_map = (F.binary_cross_entropy_with_logits(map_logit, mask, reduction="none") * valid).sum() / n_valid
    p, t = torch.sigmoid(map_logit) * valid, mask * valid
    dice_map = 1.0 - (2.0 * (p * t).sum() + 1.0) / (p.sum() + t.sum() + 1.0)
    total = bce_image + mask_weight * (bce_map + dice_map)
    parts = {"bce_image": bce_image.item(), "bce_map": bce_map.item(), "dice_map": dice_map.item()}
    return total, parts
