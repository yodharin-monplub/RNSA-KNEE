"""External knee MRI supervision (MRNet, fastMRI+, KneeMRI) and GPU augmentation for the CoAtNet fine-tune.

External exams never touch the 12 competition heads: they train separate auxiliary heads on the shared trainable
backbone stage, so differing label definitions cannot corrupt the competition labels.
Exam-level auxiliary targets (AUX, 15 values, masked per source) plus, for fastMRI+, window-level targets from
the radiologist boxes (WINDOW, 10 values): a window is positive for a class when any of its 3 slices has a box.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

AUX = ['mrnet_abnormal', 'mrnet_acl', 'mrnet_meniscus',
       'fm_meniscus', 'fm_cart_partial', 'fm_cart_full', 'fm_effusion', 'fm_cyst', 'fm_acl_high', 'fm_acl_any',
       'fm_mcl', 'fm_fracture_contusion', 'fm_subchondral_edema',
       'km_acl_any', 'km_acl_complete']
FM = {'Meniscus Tear': 'fm_meniscus', 'Displaced Meniscal Tissue': 'fm_meniscus',
      'Cartilage - Partial Thickness loss/defect': 'fm_cart_partial',
      'Cartilage - Full Thickness loss/defect': 'fm_cart_full', 'Joint Effusion': 'fm_effusion',
      'Periarticular cysts': 'fm_cyst', 'Ligament - ACL High Grade Sprain': 'fm_acl_high',
      'Ligament - ACL Low Grade sprain': 'fm_acl_any', 'Ligament - MCL Low-Mod Grade Sprain': 'fm_mcl',
      'Ligament - MCL High Grade sprain': 'fm_mcl', 'Bone-Fracture/Contusion/dislocation': 'fm_fracture_contusion',
      'Bone- Subchondral edema': 'fm_subchondral_edema'}
WINDOW = [a for a in AUX if a.startswith('fm_')]
MEAN = torch.tensor([.485, .456, .406]).view(1, 3, 1, 1)
STD = torch.tensor([.229, .224, .225]).view(1, 3, 1, 1)


def unpack(root, scratch):
    """Expand every npz shard under root into one .npy per volume (memory-mappable); returns {key: path}."""
    scratch = Path(scratch); scratch.mkdir(parents=True, exist_ok=True)
    done = scratch / 'UNPACKED.json'
    if done.exists():
        return {k: Path(v) for k, v in json.loads(done.read_text()).items()}
    paths = {}
    for shard in sorted(Path(root).rglob('*.npz')):
        with np.load(shard, allow_pickle=False) as archive:
            for key in archive.files:
                target = scratch / f'{shard.stem[:2]}_{key}.npy'
                np.save(target, archive[key])
                paths[f'{shard.stem.split("_")[0]}:{key}'] = target
    done.write_text(json.dumps({k: str(v) for k, v in paths.items()}))
    return paths


class ExternalBags:
    """Infinite sampler of (windows[K,3,S,S], aux_target[15], aux_mask[15], window_target[K,10], window_mask)."""

    def __init__(self, roots, scratch, windows, size, seed):
        self.k, self.size, self.rng = windows, size, np.random.default_rng(seed)
        self.items = []
        paths = unpack(roots['mrnet'], Path(scratch) / 'mrnet')
        labels = json.loads(next(Path(roots['mrnet']).rglob('labels.json')).read_text())
        for exam, row in labels.items():
            if row['split'] != 'train':
                continue  # MRNet's radiologist-read validation exams stay untouched for evaluation
            target, mask = np.zeros(len(AUX), np.float32), np.zeros(len(AUX), np.float32)
            for name in ('abnormal', 'acl', 'meniscus'):
                target[AUX.index(f'mrnet_{name}')] = row[name]; mask[AUX.index(f'mrnet_{name}')] = 1
            planes = [paths[f'mrnet:{exam}_{p}'] for p in ('sagittal', 'coronal', 'axial')]
            self.items.append(('mrnet', planes, target, mask, None))
        paths = unpack(roots['fastmri'], Path(scratch) / 'fastmri')
        boxes = pd.read_csv(next(Path(roots['fastmri']).rglob('knee.csv')))
        boxes = boxes[boxes.label.isin(FM)]
        by_file = {f: g for f, g in boxes.groupby('file')}
        for key, path in paths.items():
            file = key.split(':', 1)[1]
            target, mask = np.zeros(len(AUX), np.float32), np.zeros(len(AUX), np.float32)
            mask[[AUX.index(a) for a in WINDOW]] = 1
            per_slice = {}
            for row in by_file.get(file, pd.DataFrame()).itertuples():
                column = FM[row.label]
                target[AUX.index(column)] = 1
                if column == 'fm_acl_high':
                    target[AUX.index('fm_acl_any')] = 1
                if row.slice == row.slice and str(row.study_level).strip().lower() != 'yes':  # study-level rows have no slice
                    per_slice.setdefault(int(row.slice), set()).add(column)
            self.items.append(('fastmri', [path], target, mask, per_slice))
        paths = unpack(roots['kneemri'], Path(scratch) / 'kneemri')
        labels = json.loads(next(Path(roots['kneemri']).rglob('labels.json')).read_text())
        for key, row in labels.items():
            target, mask = np.zeros(len(AUX), np.float32), np.zeros(len(AUX), np.float32)
            target[AUX.index('km_acl_any')] = row['aclDiagnosis'] >= 1
            target[AUX.index('km_acl_complete')] = row['aclDiagnosis'] == 2
            mask[[AUX.index('km_acl_any'), AUX.index('km_acl_complete')]] = 1
            self.items.append(('kneemri', [paths[f'kneemri:{key}']], target, mask, None))
        self.counts = pd.Series([i[0] for i in self.items]).value_counts().to_dict()

    def sample(self):
        source, planes, target, mask, per_slice = self.items[self.rng.integers(len(self.items))]
        volumes = [np.load(p, mmap_mode='r') for p in planes]
        centers = [(v, c) for v in range(len(volumes)) for c in range(1, len(volumes[v]) - 1)]
        chosen = self.rng.choice(len(centers), size=self.k, replace=len(centers) < self.k)
        stack, window_target = [], np.zeros((self.k, len(WINDOW)), np.float32)
        for row, index in enumerate(chosen):
            v, c = centers[index]
            stack.append(np.asarray(volumes[v][c - 1:c + 2], dtype=np.float32) / 255)
            if per_slice:
                for s in (c - 1, c, c + 1):
                    for column in per_slice.get(s, ()):
                        window_target[row, WINDOW.index(column)] = 1
                        if column == 'fm_acl_high':
                            window_target[row, WINDOW.index('fm_acl_any')] = 1
        images = [F.interpolate(torch.from_numpy(s)[None], size=(self.size, self.size), mode='bilinear',
                                align_corners=False)[0] for s in stack]
        images = (torch.stack(images) - MEAN) / STD
        return images, torch.from_numpy(target), torch.from_numpy(mask), torch.from_numpy(window_target), float(per_slice is not None)

    def batch(self, n):
        parts = [self.sample() for _ in range(n)]
        return (torch.stack([p[0] for p in parts]), torch.stack([p[1] for p in parts]), torch.stack([p[2] for p in parts]),
                torch.stack([p[3] for p in parts]), torch.tensor([p[4] for p in parts]))


class AuxHeads(nn.Module):
    """Own attention-MIL head for the exam-level external labels and a per-window linear head for fastMRI+ boxes."""

    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.att = nn.Sequential(nn.Linear(dim, 256), nn.Tanh(), nn.Linear(256, len(AUX)))
        self.weight = nn.Parameter(torch.zeros(len(AUX), dim)); self.bias = nn.Parameter(torch.zeros(len(AUX)))
        self.window = nn.Linear(dim, len(WINDOW))

    def forward(self, features):
        h = self.norm(features)
        attention = self.att(h).softmax(dim=1)
        pooled = torch.einsum('bkn,bkf->bnf', attention, h)
        return (pooled * self.weight).sum(-1) + self.bias, self.window(h)


def external_loss(heads, features, target, mask, window_target, window_mask):
    exam_logits, window_logits = heads(features.float())
    exam = (F.binary_cross_entropy_with_logits(exam_logits, target, reduction='none') * mask).sum() / mask.sum().clamp(min=1)
    weights = window_mask.view(-1, 1, 1).expand_as(window_target)
    window = (F.binary_cross_entropy_with_logits(window_logits, window_target, reduction='none') * weights).sum() / weights.sum().clamp(min=1)
    return exam + window, float(exam.detach()), float(window.detach())


def augment(images, generator=None):
    """Per-study random rotation (+-10 deg), scale (+-10%), shift (+-6%), brightness/contrast/gamma on (B,K,3,H,W).

    No flips: left/right flips swap medial and lateral anatomy. One transform per study keeps its windows consistent.
    """
    b, k, c, h, w = images.shape
    device = images.device
    angle = (torch.rand(b, device=device) * 2 - 1) * np.deg2rad(10)
    scale = 1 + (torch.rand(b, device=device) * 2 - 1) * .10
    shift = (torch.rand(b, 2, device=device) * 2 - 1) * .06
    cos, sin = torch.cos(angle) / scale, torch.sin(angle) / scale
    theta = torch.stack([torch.stack([cos, -sin, shift[:, 0]], 1), torch.stack([sin, cos, shift[:, 1]], 1)], 1)
    theta = theta.repeat_interleave(k, 0)
    flat = images.reshape(b * k, c, h, w)
    grid = F.affine_grid(theta, flat.shape, align_corners=False)
    flat = F.grid_sample(flat, grid, mode='bilinear', padding_mode='border', align_corners=False)
    mean, std = MEAN.to(device), STD.to(device)
    pixel = (flat * std + mean).clamp(0, 1).reshape(b, k, c, h, w)
    gamma = torch.exp((torch.rand(b, 1, 1, 1, 1, device=device) * 2 - 1) * .25)
    contrast = 1 + (torch.rand(b, 1, 1, 1, 1, device=device) * 2 - 1) * .15
    bright = (torch.rand(b, 1, 1, 1, 1, device=device) * 2 - 1) * .08
    pixel = ((pixel.clamp(min=1e-4) ** gamma - .5) * contrast + .5 + bright).clamp(0, 1)
    return ((pixel.reshape(b * k, c, h, w) - mean) / std).reshape(b, k, c, h, w)
