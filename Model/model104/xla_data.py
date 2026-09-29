"""Datasets for scripts/train_xla.py, at module level so DataLoader workers can be started with 'spawn'.

On the Kaggle TPU VM, torch_xla.launch runs the chip processes with the 'spawn' start method and their DataLoader
workers are spawned too, so a Dataset class defined inside a function cannot be pickled
("Can't get local object 'run.<locals>.TrainBags'", TPU run 2026-09-28 23:34). Everything a worker needs is passed in.
"""
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

from knee.dense import DenseCache
from knee.external import augment
from tpu_common import windows

MEAN = torch.tensor([.485, .456, .406]).view(1, 3, 1, 1)
STD = torch.tensor([.229, .224, .225]).view(1, 3, 1, 1)


class TrainBags(Dataset):
    """train_model16.Studies: n_windows random 3-slice windows of one study, resized to res, plus CPU augmentation."""

    def __init__(self, cache_root, ids, targets, n_windows, res, augment_on):
        self.cache = DenseCache(cache_root, ids)
        self.targets = np.asarray(targets, np.float32)
        assert len(self.targets) == len(self.cache)
        self.n_windows, self.res, self.augment_on = int(n_windows), int(res), bool(augment_on)

    def __len__(self):
        return len(self.cache)

    def __getitem__(self, i):
        volume, mask = self.cache[i]
        valid = np.flatnonzero(mask)
        if len(valid) < 3:
            valid = np.arange(3)
        lo, hi = int(valid.min()), int(valid.max())
        centers = list(range(lo + 1, hi)) or [max(1, min((lo + hi) // 2, len(volume) - 2))]
        n = len(centers)
        idx = torch.randperm(n).repeat((self.n_windows + n - 1) // n)[:self.n_windows]
        x = torch.from_numpy(np.stack([volume[centers[j] - 1:centers[j] + 2] for j in idx.tolist()]).astype(np.float32) / 255)
        if x.shape[-1] != self.res:
            x = F.interpolate(x, size=(self.res, self.res), mode='bilinear', align_corners=False)
        x = (x - MEAN) / STD
        if self.augment_on:
            x = augment(x[None])[0]          # same augmentation as train_model16 (there applied on the GPU batch)
        return x, torch.from_numpy(self.targets[i])


class EvalBags(Dataset):
    """Every window of one study, padded to max_eval with a validity mask (fixed shapes -> one XLA graph)."""

    def __init__(self, cache_root, ids, targets, res, max_eval):
        self.ids = list(ids)
        self.cache = DenseCache(cache_root, self.ids)
        self.targets = np.asarray(targets, np.float32)
        assert len(self.targets) == len(self.ids)
        self.res, self.max_eval = int(res), int(max_eval)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        volume, mask = self.cache[i]
        x = windows(volume, mask, self.res)[:self.max_eval]
        k = x.shape[0]
        padded = torch.zeros((self.max_eval,) + tuple(x.shape[1:]), dtype=x.dtype)
        padded[:k] = x
        valid = torch.zeros(self.max_eval, dtype=torch.bool)
        valid[:k] = True
        return padded, valid, torch.from_numpy(self.targets[i]), i
