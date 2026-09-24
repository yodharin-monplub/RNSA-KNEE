import torch
from torch import nn


class StudyHead(nn.Module):
    def __init__(self, dim, kind, dropout, n_labels=12):
        super().__init__()
        self.kind = kind
        if kind == 'linear':
            self.predict = nn.Sequential(nn.Dropout(dropout), nn.Linear(6 * dim + 6, n_labels))
        elif kind == 'attention':
            self.attention = nn.Sequential(nn.Linear(dim, 64), nn.Tanh(), nn.Linear(64, 1))
            self.predict = nn.Sequential(nn.LayerNorm(6 * dim + 6), nn.Dropout(dropout),
                                         nn.Linear(6 * dim + 6, 256), nn.GELU(),
                                         nn.Dropout(dropout), nn.Linear(256, n_labels))
        else:
            raise ValueError(kind)

    def forward(self, x, mask):
        if self.kind == 'linear':
            x = x.mean(dim=2)
        else:
            weights = self.attention(x).softmax(dim=2)
            x = (weights * x).sum(dim=2)
        x = x * mask[..., None]
        return self.predict(torch.cat([x.flatten(1), mask.float()], dim=1))


def confidence_bce(logits, targets, weights):
    loss = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    # Normalize per label so frequent report mentions do not dominate.
    denom = weights.sum(dim=0)
    known = denom > 0
    if not known.any():
        return logits.sum() * 0
    return ((loss * weights).sum(dim=0)[known] / denom[known]).mean()
