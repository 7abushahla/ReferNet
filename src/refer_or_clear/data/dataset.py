"""PyTorch Dataset over pair lists.

Item: x (8, H, W) float32 = [query RGB, query zone, reference RGB, reference zone] with ImageNet
normalization on the RGB channels and zones in {0, 1}; mask (1, H, W) in {0, 1} (all zero for
genuine, the valid region for wrong-design pairs); valid (1, H, W) in {0, 1}; label float32 scalar
(1 = altered); meta dict. Training: geometry (random crop, small affine) drawn once per pair and
applied to every plane, alignment jitter on the query only, photometric jitter and JPEG re-encoding
drawn independently for query and reference. Test: full canvas, no augmentation, optional fixed
re-encoding of both inputs (`equalize_qf`). No flips. Files are BGR; RGB from here on.
"""
from __future__ import annotations

import random

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


def _jpeg(bgr: np.ndarray, quality: int) -> np.ndarray:
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    assert ok
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def _photometric(bgr: np.ndarray, strength: float = 0.2) -> np.ndarray:
    """Mild brightness, contrast and saturation jitter, each factor uniform in 1 +- strength."""
    b, c, s = (random.uniform(1 - strength, 1 + strength) for _ in range(3))
    x = bgr.astype(np.float32) * b
    x = (x - x.mean()) * c + x.mean()
    gray = cv2.cvtColor(np.clip(x, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)[..., None].astype(np.float32)
    x = gray + (x - gray) * s
    return np.clip(x, 0, 255).astype(np.uint8)


def _binary(path, shape) -> np.ndarray:
    if path is None:
        return np.ones(shape, np.uint8)
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return (m > 127).astype(np.uint8)


def _normalize(bgr: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return ((rgb - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1)


class PairDataset(Dataset):
    def __init__(self, pairs, train, crop=(448, 704), jpeg=(60, 100), affine_px=8, jitter_px=2,
                 jitter_p=0.25, photometric=True, equalize_qf=None):
        self.pairs, self.train = list(pairs), bool(train)
        self.crop, self.jpeg, self.affine_px = crop, jpeg, affine_px
        self.jitter_px, self.jitter_p, self.photometric, self.equalize_qf = jitter_px, jitter_p, photometric, equalize_qf

    def __len__(self):
        return len(self.pairs)

    def _geometry(self, h, w):
        """One affine (translation up to affine_px, rotation up to 1 degree) and one crop window per pair."""
        M = cv2.getRotationMatrix2D((w / 2, h / 2), random.uniform(-1, 1), 1.0)
        M[:, 2] += [random.uniform(-self.affine_px, self.affine_px) for _ in range(2)]
        ch, cw = self.crop if self.crop else (h, w)
        y0, x0 = random.randint(0, h - ch), random.randint(0, w - cw)
        return M, (slice(y0, y0 + ch), slice(x0, x0 + cw))

    @staticmethod
    def _warp(img, M, window, nearest):
        flags = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
        out = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=flags, borderMode=cv2.BORDER_CONSTANT)
        return out[window]

    def __getitem__(self, i):
        p = self.pairs[i]
        q, r = cv2.imread(p["path"], cv2.IMREAD_COLOR), cv2.imread(p["ref_path"], cv2.IMREAD_COLOR)
        h, w = q.shape[:2]
        zone = _binary(p["zone_path"], (h, w))
        valid = _binary(p["valid_path"], (h, w))
        if p["mask_path"]:
            mask = _binary(p["mask_path"], (h, w))
        elif p["wrong_design"]:
            mask = valid.copy()
        else:
            mask = np.zeros((h, w), np.uint8)

        if self.train:
            M, win = self._geometry(h, w)
            q, r = self._warp(q, M, win, False), self._warp(r, M, win, False)
            zone, mask, valid = (self._warp(m, M, win, True) for m in (zone, mask, valid))
            if self.jitter_px and random.random() < self.jitter_p:
                J = np.float32([[1, 0, random.randint(-self.jitter_px, self.jitter_px)],
                                [0, 1, random.randint(-self.jitter_px, self.jitter_px)]])
                q = cv2.warpAffine(q, J, (q.shape[1], q.shape[0]), borderMode=cv2.BORDER_REPLICATE)
            if self.photometric:
                q, r = _photometric(q), _photometric(r)
            if self.equalize_qf is None and self.jpeg:
                q, r = _jpeg(q, random.randint(*self.jpeg)), _jpeg(r, random.randint(*self.jpeg))
        if self.equalize_qf is not None:
            q, r = _jpeg(q, self.equalize_qf), _jpeg(r, self.equalize_qf)

        z = zone[None].astype(np.float32)
        x = np.concatenate([_normalize(q), z, _normalize(r), z], 0)
        return dict(x=torch.from_numpy(np.ascontiguousarray(x)),
                    mask=torch.from_numpy(mask[None].astype(np.float32)),
                    valid=torch.from_numpy(valid[None].astype(np.float32)),
                    label=torch.tensor(float(p["label"]), dtype=torch.float32),
                    meta=dict(image_id=p["image_id"], design_id=p["design_id"], subset=p["subset"],
                              attack_family=p["attack_family"], wrong_design=bool(p["wrong_design"])))


def make_loader(pairs, train, batch_size, workers, **kw) -> DataLoader:
    """DataLoader over PairDataset(pairs, train, **kw); shuffles and drops the last batch in training."""
    ds = PairDataset(pairs, train, **kw)
    return DataLoader(ds, batch_size=batch_size, shuffle=train, num_workers=workers, drop_last=train,
                      persistent_workers=workers > 0)
