"""Self-contained copy of model12.inference's Classifier, windows() and label constants for the TPU trainer
(model12.inference imports OpenCV and pydicom at import time, which the TPU image may lack)."""
import numpy as np
import timm
import torch
from torch import nn
from torch.nn import functional as F

UID = 'StudyInstanceUID'
SLOTS = [('Sagittal', 1, 18), ('Sagittal', 0, 14), ('Coronal', 1, 12),
         ('Coronal', 0, 8), ('Axial', -1, 12)]
LABELS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA',
          'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']



class Classifier(nn.Module):
    def __init__(self, arch, img_size=None):
        super().__init__()
        # img_size is only passed when a config asks for it: CoAtNet rmlp backbones bake the relative
        # position bias for their native resolution (384 -> 576 tokens) and must be rebuilt for another one.
        extra = {'img_size': img_size} if img_size else {}
        self.backbone = timm.create_model(arch, pretrained=False, num_classes=0,
                                          in_chans=3, global_pool='avg', **extra)
        dim = self.backbone.num_features
        self.norm = nn.LayerNorm(dim)
        self.att = nn.Sequential(nn.Linear(dim, 256), nn.Tanh(), nn.Dropout(.2), nn.Linear(256, 12))
        self.clsW = nn.Parameter(torch.zeros(12, dim))
        self.clsb = nn.Parameter(torch.zeros(12))

    def predict_features(self, features):
        features = self.norm(features)
        attention = self.att(features).softmax(dim=1)
        pooled = torch.einsum('bkn,bkf->bnf', attention, features)
        return (pooled * self.clsW).sum(-1) + self.clsb


def windows(volume, mask, size):
    valid = np.flatnonzero(mask)
    if len(valid) < 3:
        valid = np.arange(3)
    lo, hi = int(valid.min()), int(valid.max())
    centers = list(range(lo+1, hi)) or [max(1, min((lo+hi)//2, len(volume)-2))]
    indices = np.arange(len(centers))
    slices = np.stack([volume[centers[i]-1:centers[i]+2] for i in indices]).astype(np.float32) / 255
    images = F.interpolate(torch.from_numpy(slices), size=(size,size), mode='bilinear', align_corners=False)
    return (images - torch.tensor([.485,.456,.406]).view(1,3,1,1)) / torch.tensor([.229,.224,.225]).view(1,3,1,1)


