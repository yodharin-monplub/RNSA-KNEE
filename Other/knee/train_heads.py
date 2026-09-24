import argparse
import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from knee.data import UID
from knee.models import StudyHead, confidence_bce


def score(y, p):
    values = [float(roc_auc_score(y[:, j], p[:, j])) if len(np.unique(y[:, j])) == 2
              else None for j in range(y.shape[1])]
    return {'macro_auc': float(np.mean([v for v in values if v is not None])),
            'per_label_auc': values}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    random.seed(cfg['seed']); np.random.seed(cfg['seed']); torch.manual_seed(cfg['seed'])
    torch.set_num_threads(4)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parts = pd.read_csv('artifacts/partitions/partitions.csv')
    ids = pd.read_csv('artifacts/features/ids.csv')
    assert parts[UID].equals(ids[UID]), 'Feature/target alignment mismatch'
    labels = json.loads(Path('artifacts/partitions/labels.json').read_text())
    targets = np.load('artifacts/partitions/targets.npz')
    raw = np.load('artifacts/features/embeddings.npy')
    mask = np.load('artifacts/features/mask.npy')
    meta = json.loads(Path('artifacts/features/metadata.json').read_text())
    assert meta['backbone'] == cfg['backbone']
    tr = np.flatnonzero(parts.split.eq('train')); va = np.flatnonzero(parts.split.eq('weak_val'))
    gold = np.flatnonzero(parts.split.eq('gold'))
    present = raw[tr][mask[tr]].reshape(-1, raw.shape[-1])
    mean = present.mean(axis=0); std = np.maximum(present.std(axis=0), 1e-3)
    x = torch.from_numpy(((raw - mean) / std) * mask[:, :, None, None]).to(device)
    m = torch.from_numpy(mask.astype(np.float32)).to(device)
    y = torch.from_numpy(targets['weak']).to(device)
    w = torch.from_numpy(targets['confidence']).to(device)
    model = StudyHead(raw.shape[-1], cfg['head'], cfg['dropout'], len(labels)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg['epochs'])
    loader = DataLoader(TensorDataset(torch.from_numpy(tr)), batch_size=cfg['batch_size'], shuffle=True)
    best, best_epoch, best_state, stale = float('inf'), 0, None, 0
    history = []
    out = Path('artifacts') / cfg['name']; out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    for epoch in range(cfg['epochs']):
        model.train()
        for (index,) in loader:
            index = index.to(device)
            loss = confidence_bce(model(x[index], m[index]), y[index], w[index])
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
        scheduler.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(confidence_bce(model(x[va], m[va]), y[va], w[va]))
        history.append({'epoch': epoch+1, 'weak_val_bce': val_loss})
        if val_loss < best - 1e-5:
            best, best_epoch, stale = val_loss, epoch+1, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
        print(json.dumps({'model': cfg['name'], **history[-1]}), flush=True)
        if stale >= cfg['patience']:
            break
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        p = model(x[gold], m[gold]).sigmoid().cpu().numpy()
        vp = model(x[va], m[va]).sigmoid().cpu().numpy()
    metrics = score(targets['gold'][gold], p)
    metrics['per_label_auc'] = dict(zip(labels, metrics['per_label_auc']))
    metrics.update({'model': cfg['name'], 'gold_n': len(gold), 'training_n': len(tr),
                    'weak_validation_n': len(va), 'best_epoch': best_epoch,
                    'weak_val_bce': best, 'training_seconds': time.time()-started,
                    'evaluation': 'All gold studies held out; selected using weak validation only',
                    'leaderboard_score': None})
    rng = np.random.default_rng(2026)
    bootstrap = []
    for _ in range(300):
        sample = rng.integers(0, len(gold), len(gold))
        bootstrap.append(score(targets['gold'][gold][sample], p[sample])['macro_auc'])
    metrics['gold_bootstrap_95pct'] = np.quantile(bootstrap, [.025, .975]).tolist()
    for subset, pred, name in [(gold, p, 'gold_predictions.csv'), (va, vp, 'weak_val_predictions.csv')]:
        frame = pd.DataFrame(pred, columns=labels)
        frame.insert(0, UID, ids.iloc[subset][UID].to_numpy())
        frame.to_csv(out / name, index=False)
    checkpoint = {'config': cfg, 'state_dict': {k:v.cpu() for k,v in best_state.items()},
                  'mean': torch.from_numpy(mean), 'std': torch.from_numpy(std),
                  'dim': raw.shape[-1], 'labels': labels, 'best_epoch': best_epoch}
    torch.save(checkpoint, out / 'head.pt')
    (out / 'metrics.json').write_text(json.dumps(metrics, indent=2))
    (out / 'history.json').write_text(json.dumps(history, indent=2))
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == '__main__':
    main()
