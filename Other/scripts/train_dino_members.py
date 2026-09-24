"""Fine-tune the twenty public DINOv2 members on this project's labels.

Each member is rebuilt with the notebook's own architecture, checked against its stored fingerprint,
fine-tuned for a fixed number of epochs on a member-specific random subset of the non-gold studies
(keeping ensemble diversity), then written back in the same checkpoint format with a recomputed
fingerprint so the public ensemble's verification still passes. Gold58 is never trained on.
"""
import argparse
import copy
import hashlib
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from knee.dino_cache import DinoCache
from knee.dino_model import TARGETS, build_model, fingerprint

UID = 'StudyInstanceUID'


class Studies(Dataset):
    def __init__(self, cache, ids, targets):
        self.cache = DinoCache(cache, ids)
        self.targets = targets.loc[ids, TARGETS].to_numpy(dtype=np.float32)

    def __len__(self):
        return len(self.cache)

    def __getitem__(self, i):
        volume, mask = self.cache[i]
        return torch.from_numpy(volume), torch.from_numpy(mask), torch.from_numpy(self.targets[i])


@torch.inference_mode()
def predict(model, dataset, device, batch_size=4):
    model.eval()
    out = []
    for images, mask, _ in DataLoader(dataset, batch_size=batch_size, num_workers=0):
        with torch.autocast('cuda', dtype=torch.float16):
            logits = model(images.to(device), mask.to(device), 336)
        out.append(logits.float().sigmoid().cpu().numpy())
    return np.concatenate(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', required=True)
    parser.add_argument('--weights', required=True, help='directory holding manifest.json and m_*.pt')
    parser.add_argument('--config', default=str(ROOT / 'model47/config.json'))
    parser.add_argument('--output', default='artifacts/model47')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    assert torch.cuda.is_available()
    device = torch.device('cuda')
    torch.set_num_threads(4)
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)

    partition = pd.read_csv(ROOT / 'artifacts/partitions/partitions.csv').set_index(UID)
    targets = pd.read_csv(ROOT / config['targets']).set_index(UID)
    gold = pd.read_csv(ROOT / 'data/train.csv').set_index(UID)
    gold_ids = partition.index[partition.split == 'gold'].tolist()
    pool = [uid for uid in partition.index if uid not in set(gold_ids)]
    assert len(gold_ids) == 58 and len(pool) == 4349
    manifest = json.loads((Path(args.weights) / 'manifest.json').read_text())
    members = manifest['members']
    assert len(members) == 20
    if args.smoke:
        members = members[:1]
    gold_set = Studies(args.cache, gold_ids, gold)
    gold_y = gold.loc[gold_ids, TARGETS].to_numpy()
    macro = lambda p: float(np.mean([roc_auc_score(gold_y[:, j], p[:, j]) for j in range(12)]))
    report = []
    started = time.time()
    for position, member in enumerate(members):
        seed = config['seed'] + position
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        checkpoint = torch.load(Path(args.weights) / member['file'], map_location='cpu', weights_only=False)
        cfg = member['config']
        model = build_model(int(cfg['unfreeze_last']), variant=cfg['variant'], pool=cfg.get('pool', 'cls_mean'),
                            prior=bool(cfg.get('prior', False)))
        model.load_state_dict(checkpoint['model'])
        model = model.to(device)
        stored = np.asarray(checkpoint['fingerprint'], np.float32)
        before_fp = fingerprint(model, device, 336)
        drift = float(np.abs(before_fp - stored).max())
        assert drift < 0.002, f"{member['id']}: architecture mismatch, fingerprint drift {drift}"
        before = macro(predict(model, gold_set, device))

        rng = np.random.default_rng(seed)
        subset = sorted(rng.choice(np.asarray(pool), size=int(len(pool) * config['subset_fraction']),
                                   replace=False).tolist())
        if args.smoke:
            subset = subset[:config['batch_size'] * 2]
        train_set = Studies(args.cache, subset, targets)
        teacher = copy.deepcopy(model).eval().requires_grad_(False)
        encoder = [p for n, p in model.named_parameters() if n.startswith('backbone.') and p.requires_grad]
        head = [p for n, p in model.named_parameters() if not n.startswith('backbone.')]
        optimizer = torch.optim.AdamW([{'params': encoder, 'lr': config['encoder_lr']},
                                       {'params': head, 'lr': config['head_lr']}],
                                      weight_decay=config['weight_decay'])
        scaler = torch.amp.GradScaler('cuda')
        loader = DataLoader(train_set, batch_size=config['batch_size'], shuffle=True,
                            num_workers=0 if args.smoke else config['workers'], pin_memory=True, drop_last=True)
        model.train()
        for epoch in range(1, (1 if args.smoke else config['epochs']) + 1):
            for step, (images, mask, target) in enumerate(loader):
                images, mask, target = images.to(device), mask.to(device), target.to(device)
                confidence = 2 * (target - .5).abs()
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda', dtype=torch.float16):
                    logits = model(images, mask, 336).float()
                    supervised = (F.binary_cross_entropy_with_logits(logits, target, reduction='none')
                                  * confidence).sum() / confidence.sum().clamp(min=1)
                    loss = supervised
                    if config['teacher_weight'] > 0:
                        with torch.no_grad():
                            teacher_prob = teacher(images, mask, 336).float().sigmoid()
                        loss = loss + config['teacher_weight'] * F.binary_cross_entropy_with_logits(logits, teacher_prob)
                assert torch.isfinite(loss)
                scaler.scale(loss).backward(); scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(encoder + head, config['gradient_clip'], error_if_nonfinite=False)
                scaler.step(optimizer); scaler.update()
                if step % 100 == 0:
                    print(json.dumps({'member': member['id'], 'epoch': epoch, 'step': step,
                                      'loss': float(loss.detach())}), flush=True)
                if args.smoke and step >= 1:
                    break
        after = macro(predict(model, gold_set, device))
        new_fp = fingerprint(model, device, 336)
        state = {'model': {k: v.detach().cpu() for k, v in model.state_dict().items()},
                 'fingerprint': new_fp.tolist(), 'config': cfg, 'id': member['id'],
                 'source_member': member['file'], 'training_config': config}
        torch.save(state, out / member['file'])
        report.append({'id': member['id'], 'gold_before': before, 'gold_after': after,
                       'train_studies': len(subset), 'fingerprint_drift_vs_stored': drift})
        print(json.dumps(report[-1]), flush=True)
        (out / 'members_report.json').write_text(json.dumps(report, indent=2))
        del model, teacher, optimizer
        torch.cuda.empty_cache()
    summary = {'members': len(report), 'mean_gold_before': float(np.mean([r['gold_before'] for r in report])),
               'mean_gold_after': float(np.mean([r['gold_after'] for r in report])),
               'elapsed_seconds': time.time() - started, 'targets_sha256':
               hashlib.sha256((ROOT / config['targets']).read_bytes()).hexdigest(), 'smoke': args.smoke}
    (out / ('SMOKE_PASS.json' if args.smoke else 'metrics.json')).write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)
    if not args.smoke:
        new_manifest = dict(manifest)
        new_manifest['members'] = [dict(m, fingerprint=torch.load(out / m['file'], map_location='cpu',
                                                                 weights_only=False)['fingerprint'])
                                   for m in manifest['members']]
        (out / 'manifest.json').write_text(json.dumps(new_manifest))
        (out / 'TRAINING_COMPLETE').write_text('model47 members complete\n')


if __name__ == '__main__':
    main()
