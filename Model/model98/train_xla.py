"""DRAFT (untested on TPU): train_model16.py's from-ImageNet recipe on a Kaggle TPU v5e-8 with PyTorch/XLA (PJRT).

Same data (the verified dense cache), labels, partitions, model (model12 Classifier = timm backbone + Raptor MIL head),
loss (plain soft-label BCE), optimizer groups (encoder / head LR), warmup + cosine schedule and weak-val epoch
selection as scripts/train_model16.py. TPU-specific changes:
  - one process per TPU chip (torch_xla.launch); DistributedSampler; per-chip batch = batch_size / world (model75:
    8 studies -> 1 study of 16 windows per chip, the same global batch);
  - bf16 autocast instead of fp16 + GradScaler; gradients all-reduced (xm.reduce_gradients) BEFORE clipping;
  - augmentation on the host CPU inside the Dataset (knee.external.augment on CPU tensors), so the XLA graph stays
    small and static;
  - evaluation pads every study to MAX_EVAL windows with an attention mask (fixed shapes -> one compiled graph),
    shards studies across chips and gathers predictions; rank 0 selects, saves and logs.
Outputs match train_model16.py: best.pt (same checkpoint layout, loadable by Classifier / the fleet leg),
history.json, gold_predictions.csv, metrics.json, TRAINING_COMPLETE.

Usage (in a Kaggle TPU notebook, cache/targets/partitions attached):
  python train_xla.py --cache /kaggle/tmp/dense --config config.json --output /kaggle/working/model98 [--smoke]
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
MAX_EVAL = 62   # windows per study at evaluation (64-frame cache -> at most 62 centers)


def masked_predict(model, feats, mask):
    """Raptor MIL head with padded windows masked out of the attention softmax (feats: b,k,f; mask: b,k bool)."""
    import torch
    feats = model.norm(feats)
    logits = model.att(feats)                                   # b, k, n
    logits = logits.masked_fill(~mask[..., None], torch.finfo(logits.dtype).min)
    attention = logits.softmax(dim=1)
    pooled = torch.einsum('bkn,bkf->bnf', attention, feats)
    return (pooled * model.clsW).sum(-1) + model.clsb


def run(index, args):
    import torch
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl
    import torch_xla.runtime as xr
    import pandas as pd
    import timm
    from torch.nn import functional as F
    from torch.utils.data import DataLoader, Dataset
    from torch.utils.data.distributed import DistributedSampler
    from sklearn.metrics import roc_auc_score
    from knee.dense import DenseCache
    from knee.external import augment
    from tpu_common import Classifier, LABELS, UID, windows

    config = json.loads(Path(args.config).read_text())
    device = xm.xla_device()
    world, rank = xr.world_size(), xr.global_ordinal()
    torch.manual_seed(config['seed'] + rank); np.random.seed(config['seed'] + rank)
    res = int(config['res'])
    per_chip = max(1, int(config['batch_size']) // world)

    partition = pd.read_csv(ROOT / 'artifacts/partitions/partitions.csv').set_index(UID)
    targets = pd.read_csv(ROOT / config['targets']).set_index(UID)
    gold = pd.read_csv(ROOT / 'data/train.csv').set_index(UID)
    train_ids = partition.index[partition.split == 'train'].tolist()
    val_ids = partition.index[partition.split == 'weak_val'].tolist()
    gold_ids = partition.index[partition.split == 'gold'].tolist()
    if args.smoke:
        # Mechanics only (compile, steps, evaluation, saving) on whatever the (pilot) cache holds - not a quality run.
        cached = pd.read_csv(Path(args.cache) / 'index.csv')[UID].tolist()
        gold_cached = [u for u in gold_ids if u in set(cached)]
        train_ids = (cached * 4)[:per_chip * world * 3]
        val_ids = cached[:world * 2]
        gold_ids = gold_cached[:world * 2]

    class TrainBags(Dataset):   # train_model16.Studies with n_windows, plus CPU augmentation
        def __init__(self, ids):
            self.cache = DenseCache(args.cache, ids)
            self.targets = targets.loc[ids, LABELS].to_numpy(np.float32)

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
            idx = torch.randperm(n).repeat((config['train_windows'] + n - 1) // n)[:config['train_windows']]
            x = torch.from_numpy(np.stack([volume[centers[j] - 1:centers[j] + 2] for j in idx.tolist()]).astype(np.float32) / 255)
            if x.shape[-1] != res:
                x = F.interpolate(x, size=(res, res), mode='bilinear', align_corners=False)
            x = (x - torch.tensor([.485, .456, .406]).view(1, 3, 1, 1)) / torch.tensor([.229, .224, .225]).view(1, 3, 1, 1)
            if config.get('augment'):
                x = augment(x[None])[0]          # same augmentation as train_model16 (there applied on the GPU batch)
            return x, torch.from_numpy(self.targets[i])

    class EvalBags(Dataset):   # all windows, padded to MAX_EVAL with a mask
        def __init__(self, ids, labels_frame):
            self.ids = list(ids)
            self.cache = DenseCache(args.cache, self.ids)
            self.targets = labels_frame.loc[self.ids, LABELS].to_numpy(np.float32)

        def __len__(self):
            return len(self.ids)

        def __getitem__(self, i):
            volume, mask = self.cache[i]
            x = windows(volume, mask, res)[:MAX_EVAL]
            k = x.shape[0]
            padded = torch.zeros((MAX_EVAL,) + tuple(x.shape[1:]), dtype=x.dtype)
            padded[:k] = x
            valid = torch.zeros(MAX_EVAL, dtype=torch.bool); valid[:k] = True
            return padded, valid, torch.from_numpy(self.targets[i]), i

    model = Classifier(config['arch'], img_size=config.get('backbone_img_size'))
    overlay = {'file': str(ROOT / config['pretrained_file'])} if config.get('pretrained_file') else None
    model.backbone = timm.create_model(config['arch'], pretrained=True, num_classes=0, in_chans=3, global_pool='avg',
                                       pretrained_cfg_overlay=overlay,
                                       **({'img_size': config['backbone_img_size']} if config.get('backbone_img_size') else {}))
    model = model.to(device)
    encoder = list(model.backbone.parameters())
    head = [p for name, p in model.named_parameters() if not name.startswith('backbone.')]
    optimizer = torch.optim.AdamW([{'params': encoder, 'lr': config['encoder_lr']}, {'params': head, 'lr': config['head_lr']}],
                                  weight_decay=config['weight_decay'])
    train_set = TrainBags(train_ids)
    sampler = DistributedSampler(train_set, num_replicas=world, rank=rank, shuffle=True, seed=config['seed'], drop_last=True)
    loader = DataLoader(train_set, batch_size=per_chip, sampler=sampler, num_workers=config.get('workers', 4), drop_last=True)
    steps_per_epoch = len(loader)
    total = steps_per_epoch * (1 if args.smoke else config['epochs'])
    warm = int(float(config.get('warmup_ratio', 0)) * total)
    ratio = float(config.get('min_lr_ratio', 0.01))

    def lr_factor(step):
        if step < warm:
            return (step + 1) / warm
        t = min(step - warm, total - warm) / max(1, total - warm)
        return ratio + (1 - ratio) * .5 * (1 + np.cos(np.pi * t))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)

    def evaluate(ids, labels_frame):
        data = EvalBags(ids, labels_frame)
        shard = DistributedSampler(data, num_replicas=world, rank=rank, shuffle=False, drop_last=False)
        dl = pl.MpDeviceLoader(DataLoader(data, batch_size=1, sampler=shard, num_workers=2), device)
        model.eval()
        preds, rows = [], []
        with torch.no_grad():
            for x, valid, target, i in dl:
                with torch.autocast('xla', dtype=torch.bfloat16):
                    feats = model.backbone(x.flatten(0, 1)).reshape(1, MAX_EVAL, -1)
                    logits = masked_predict(model, feats.float(), valid)
                preds.append(logits.float().sigmoid()); rows.append(i)
        p = torch.cat(preds) if preds else torch.zeros((0, len(LABELS)), device=device)
        r = torch.cat(rows).to(device) if rows else torch.zeros((0,), dtype=torch.long, device=device)
        p_all = xm.all_gather(p, dim=0); r_all = xm.all_gather(r, dim=0)
        p_all, r_all = p_all.cpu().numpy(), r_all.cpu().numpy()
        full = np.zeros((len(data), len(LABELS)), np.float32)
        full[r_all] = p_all                       # DistributedSampler pads by repeating; duplicates overwrite equally
        return full, data.targets

    history, best, started = [], float('inf'), time.time()
    for epoch in range(1, (1 if args.smoke else config['epochs']) + 1):
        sampler.set_epoch(epoch)
        model.train()
        for step, (images, target) in enumerate(pl.MpDeviceLoader(loader, device)):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('xla', dtype=torch.bfloat16):
                b, k = images.shape[:2]
                feats = model.backbone(images.flatten(0, 1)).reshape(b, k, -1)
                logits = model.predict_features(feats.float()).float()
                loss = F.binary_cross_entropy_with_logits(logits, target)
            loss.backward()
            xm.reduce_gradients(optimizer)       # average gradients across chips before clipping
            torch.nn.utils.clip_grad_norm_(encoder + head, config['gradient_clip'])
            optimizer.step(); scheduler.step()
            xm.mark_step()
            if step % 100 == 0 and rank == 0:
                print(json.dumps({'epoch': epoch, 'step': step, 'loss': float(loss.detach().cpu())}), flush=True)
            if args.smoke and step >= 2:
                break
        pv, tv = evaluate(val_ids, targets)
        val_bce = float(np.mean(-(tv * np.log(np.clip(pv, 1e-6, 1)) + (1 - tv) * np.log(np.clip(1 - pv, 1e-6, 1)))))
        pg, tg = evaluate(gold_ids, gold)
        gold_auc = float(roc_auc_score(tg, pg)) if not args.smoke else None
        history.append({'epoch': epoch, 'weak_val_bce': val_bce, 'gold_auc_descriptive': gold_auc,
                        'elapsed_seconds': time.time() - started})
        if rank == 0:
            print(json.dumps(history[-1]), flush=True)
            (out / 'history.json').write_text(json.dumps(history, indent=2))
        if val_bce < best:
            best = val_bce
            state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            if rank == 0:
                torch.save({'model': state, 'arch': config['arch'], 'res': res, 'lab': LABELS, 'epoch': epoch,
                            'weak_val_bce': val_bce, 'training_config': config}, out / 'best.pt')
                pd.DataFrame(pg, index=pd.Index(gold_ids, name=UID), columns=LABELS).to_csv(out / 'gold_predictions.csv')
        xm.rendezvous(f'epoch{epoch}')
    if rank == 0:
        metrics = {'selected_epoch': int(min(history, key=lambda h: h['weak_val_bce'])['epoch']), 'weak_val_bce': best,
                   'elapsed_seconds': time.time() - started, 'trainer': 'train_xla.py (TPU v5e-8, bf16)'}
        (out / 'metrics.json').write_text(json.dumps(metrics, indent=2))
        (out / 'TRAINING_COMPLETE').write_text('complete\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    os.environ.setdefault('PJRT_DEVICE', 'TPU')
    import torch_xla
    if hasattr(torch_xla, 'launch'):
        torch_xla.launch(run, args=(args,))
    else:   # older torch_xla
        import torch_xla.distributed.xla_multiprocessing as xmp
        xmp.spawn(run, args=(args,))


if __name__ == '__main__':
    main()
