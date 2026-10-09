"""PyTorch Dataset over pair lists, with augmentation: random affine up to 8 px between page and
reference, photometric jitter, and independent JPEG quality 60 to 100 on each input."""


class PairDataset:
    def __init__(self, pairs, train=True):
        raise NotImplementedError
