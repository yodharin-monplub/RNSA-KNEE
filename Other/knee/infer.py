"""Offline inference; no report, internet, or training labels are required."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from knee.data import UID
from knee.extract import encode_pixels, make_encoder
from knee.models import StudyHead
from knee.preprocess import preprocess_study


class TestStudies(Dataset):
    def __init__(self, root, ids, series):
        self.root, self.ids = root, list(ids)
        self.series = dict(tuple(series.groupby(UID, sort=False)))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        return preprocess_study(self.root, self.series[self.ids[index]])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--weights', required=True)
    parser.add_argument('--models', nargs='+', default=['model2', 'model3'])
    parser.add_argument('--output', default='submission.csv')
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    root, weights = Path(args.data), Path(args.weights)
    torch.set_num_threads(4)
    test = pd.read_csv(root / 'test.csv')
    sample = pd.read_csv(root / 'sample_submission.csv')
    series = pd.read_csv(root / 'test_series.csv')
    labels = list(sample.columns.drop(UID))
    assert test[UID].is_unique and sample[UID].is_unique
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    heads = []
    backbone = None
    encoder_paths = set()
    for name in args.models:
        ckpt = torch.load(weights / name / 'head.pt', map_location='cpu', weights_only=True)
        assert ckpt['labels'] == labels
        cfg = ckpt['config']
        if backbone is not None and cfg['backbone'] != backbone:
            raise ValueError('Ensemble requires the same encoder')
        backbone = cfg['backbone']
        own_encoder = weights / name / 'encoder.pt'
        encoder_paths.add(own_encoder if own_encoder.exists() else weights / 'encoder.pt')
        head = StudyHead(ckpt['dim'], cfg['head'], cfg['dropout'], len(labels)).to(device).eval()
        head.load_state_dict(ckpt['state_dict'])
        heads.append((head, ckpt['mean'].to(device), ckpt['std'].to(device)))
    if len(encoder_paths) != 1:
        raise ValueError('Run heads with different encoders separately, then blend their predictions')
    encoder = make_encoder(backbone, pretrained=False).to(device).eval()
    encoder.load_state_dict(torch.load(next(iter(encoder_paths)), map_location='cpu', weights_only=True))
    loader = DataLoader(TestStudies(root / 'test_series', test[UID], series),
                        batch_size=4, shuffle=False, num_workers=args.workers)
    predictions = []
    with torch.inference_mode():
        for pixels, mask in loader:
            features = torch.from_numpy(encode_pixels(encoder, pixels, mask, device)).to(device)
            mask = mask.to(device)
            probs = []
            for head, mean, std in heads:
                x = ((features - mean) / std) * mask[:, :, None, None]
                probs.append(head(x, mask).sigmoid())
            predictions.append(torch.stack(probs).mean(0).cpu().numpy())
    pred = np.concatenate(predictions)
    assert np.isfinite(pred).all() and ((pred >= 0) & (pred <= 1)).all()
    result = pd.DataFrame(pred, columns=labels)
    result.insert(0, UID, test[UID].to_numpy())
    submission = sample[[UID]].merge(result, on=UID, validate='one_to_one')[sample.columns]
    assert len(submission) == len(sample) == len(test)
    submission.to_csv(args.output, index=False)
    print(f'Wrote {len(submission)} studies to {args.output}')


if __name__ == '__main__':
    main()
