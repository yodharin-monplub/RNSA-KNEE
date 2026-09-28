"""train_model16.py's from-ImageNet recipe on a Kaggle TPU v5e-8 with PyTorch/XLA (PJRT).

Same data (the verified dense cache), labels, partitions, model (model12 Classifier = timm backbone + Raptor MIL head),
loss (plain soft-label BCE), optimizer groups (encoder / head LR), warmup + cosine schedule and weak-val epoch
selection as scripts/train_model16.py. TPU-specific changes:
  - one process per TPU chip (torch_xla.launch); DistributedSampler; per-chip batch = batch_size / world (model75:
    8 studies -> 1 study of 16 windows per chip, the same global batch);
  - bf16 autocast instead of fp16 + GradScaler; gradients all-reduced (xm.reduce_gradients) BEFORE clipping;
  - augmentation on the host CPU inside the Dataset (knee.external.augment on CPU tensors), so the XLA graph stays
    small and static;
  - evaluation pads every study to all window centers of the cache (62 for 64 frames, 94 for 96) with an attention
    mask (fixed shapes -> one compiled graph),
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
    from torch.utils.data import DataLoader
    from torch.utils.data.distributed import DistributedSampler
    from sklearn.metrics import roc_auc_score
    from knee.dense import DenseCache
    from tpu_common import Classifier, LABELS, UID
    from xla_data import EvalBags, TrainBags

    config = json.loads(Path(args.config).read_text())
    max_eval = DenseCache(args.cache).frames - 2   # every window center: 62 for the 64-frame cache, 94 for 96 frames
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
    fold = args.fold if args.fold is not None else config.get('fold')
    oof_ids = []
    if fold is not None and fold != '':
        # train_model16.py's k-fold: the non-gold studies split by report group (md5 of report_hash), train on k-1 folds
        # and predict the held-out fold with the LAST epoch (fixed schedule; the fold is monitored, never used to select).
        import hashlib
        assert config.get('select') == 'last'
        pool = train_ids + val_ids
        digest = {u: int(hashlib.md5(str(partition.loc[u, 'report_hash']).encode()).hexdigest(), 16) for u in pool}
        n_folds, k = int(config['n_folds']), int(fold)
        assert 0 <= k < n_folds
        train_ids = [u for u in pool if digest[u] % n_folds != k]
        oof_ids = val_ids = [u for u in pool if digest[u] % n_folds == k]
        assert train_ids and oof_ids and set(train_ids).isdisjoint(gold_ids)
        if rank == 0:
            print(json.dumps({'fold': k, 'train': len(train_ids), 'oof': len(oof_ids)}), flush=True)
    select_last = config.get('select') == 'last'
    if args.smoke:
        # Mechanics only (compile, steps, evaluation, saving) on whatever the (pilot) cache holds - not a quality run.
        cached = pd.read_csv(Path(args.cache) / 'index.csv')[UID].tolist()
        gold_cached = [u for u in gold_ids if u in set(cached)]
        train_ids = (cached * 4)[:per_chip * world * 3]
        val_ids = cached[:world * 2]
        gold_ids = gold_cached[:world * 2]
        oof_ids = val_ids if oof_ids else []

    model = Classifier(config['arch'], img_size=config.get('backbone_img_size'))
    overlay = {'file': str(ROOT / config['pretrained_file'])} if config.get('pretrained_file') else None
    model.backbone = timm.create_model(config['arch'], pretrained=True, num_classes=0, in_chans=3, global_pool='avg',
                                       pretrained_cfg_overlay=overlay,
                                       **({'img_size': config['backbone_img_size']} if config.get('backbone_img_size') else {}))
    if config.get('grad_checkpointing'):
        # timm's checkpoint_seq calls torch.utils.checkpoint.checkpoint. On XLA use torch_xla's version (its optimization
        # barriers stop the compiler from merging the recomputation back); it supports only the reentrant variant.
        import torch.utils.checkpoint as torch_checkpoint
        import torch_xla.utils.checkpoint as xla_checkpoint

        def _checkpoint(function, *inputs, use_reentrant=None, **kwargs):
            return xla_checkpoint.checkpoint(function, *inputs, use_reentrant=True, **kwargs)
        torch_checkpoint.checkpoint = _checkpoint
        model.backbone.set_grad_checkpointing(True)
    model = model.to(device)
    encoder = list(model.backbone.parameters())
    head = [p for name, p in model.named_parameters() if not name.startswith('backbone.')]
    optimizer = torch.optim.AdamW([{'params': encoder, 'lr': config['encoder_lr']}, {'params': head, 'lr': config['head_lr']}],
                                  weight_decay=config['weight_decay'])
    train_set = TrainBags(args.cache, train_ids, targets.loc[train_ids, LABELS].to_numpy(np.float32),
                          config['train_windows'], res, config.get('augment'))
    sampler = DistributedSampler(train_set, num_replicas=world, rank=rank, shuffle=True, seed=config['seed'], drop_last=True)
    workers = int(config.get('workers', 4))
    loader = DataLoader(train_set, batch_size=per_chip, sampler=sampler, num_workers=workers, drop_last=True,
                        persistent_workers=workers > 0)
    steps_per_epoch = len(loader)
    plan_epochs = 2 if args.smoke else int(config['epochs'])   # may shrink (or regrow up to config) to meet --stop-at
    total = steps_per_epoch * plan_epochs
    warm = int(float(config.get('warmup_ratio', 0)) * total)
    ratio = float(config.get('min_lr_ratio', 0.01))

    def lr_factor(step):
        if step < warm:
            return (step + 1) / warm
        t = min(step - warm, total - warm) / max(1, total - warm)
        return ratio + (1 - ratio) * .5 * (1 + np.cos(np.pi * t))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)

    def eval_loader(ids, labels_frame):
        data = EvalBags(args.cache, ids, labels_frame.loc[list(ids), LABELS].to_numpy(np.float32), res, max_eval)
        shard = DistributedSampler(data, num_replicas=world, rank=rank, shuffle=False, drop_last=False)
        n = int(config.get('eval_workers', 2))
        return data, DataLoader(data, batch_size=1, sampler=shard, num_workers=n, persistent_workers=n > 0)

    val_eval, gold_eval = eval_loader(val_ids, targets), eval_loader(gold_ids, gold)

    def evaluate(data, cpu_loader):
        dl = pl.MpDeviceLoader(cpu_loader, device)
        model.eval()
        preds, rows = [], []
        with torch.no_grad():
            for x, valid, target, i in dl:
                with torch.autocast('xla', dtype=torch.bfloat16):
                    feats = model.backbone(x.flatten(0, 1)).reshape(1, max_eval, -1)
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
    epoch = 0
    while epoch < plan_epochs:
        epoch += 1
        epoch_started = time.time()
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
        pv, tv = evaluate(*val_eval)
        val_bce = float(np.mean(-(tv * np.log(np.clip(pv, 1e-6, 1)) + (1 - tv) * np.log(np.clip(1 - pv, 1e-6, 1)))))
        pg, tg = evaluate(*gold_eval)
        gold_auc = float(roc_auc_score(tg, pg)) if not args.smoke else None
        # Epoch plan for the time limit: every chip proposes how many epochs fit before --stop-at (from this epoch's
        # duration) and all agree on the minimum. The cosine schedule follows the new total (lr_factor reads `total`).
        seconds = time.time() - epoch_started
        want = plan_epochs if args.smoke else int(config['epochs'])
        if args.stop_at:
            want = min(want, epoch + int(max(0.0, args.stop_at - time.time()) // (seconds * 1.05)))
        want = max(epoch, want)
        agreed = int(xm.all_reduce(xm.REDUCE_MIN, torch.tensor([float(want)], device=device)).cpu().item())
        if agreed != plan_epochs:
            if rank == 0:
                print(f'epoch plan {plan_epochs} -> {agreed} (epoch {seconds:.0f}s, stop-at in '
                      f'{args.stop_at - time.time():.0f}s)', flush=True)
            plan_epochs = agreed
            total = steps_per_epoch * plan_epochs
        history.append({'epoch': epoch, 'weak_val_bce': val_bce, 'gold_auc_descriptive': gold_auc,
                        'elapsed_seconds': time.time() - started, 'epoch_seconds': seconds, 'plan_epochs': plan_epochs})
        if rank == 0:
            print(json.dumps(history[-1]), flush=True)
            (out / 'history.json').write_text(json.dumps(history, indent=2))
        if (epoch == plan_epochs) if select_last else (val_bce < best):   # k-fold keeps the last epoch
            best = val_bce
            state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            if rank == 0:
                torch.save({'model': state, 'arch': config['arch'], 'res': res, 'lab': LABELS, 'epoch': epoch,
                            'weak_val_bce': val_bce, 'training_config': config}, out / 'best.pt')
                pd.DataFrame(pg, index=pd.Index(gold_ids, name=UID), columns=LABELS).to_csv(out / 'gold_predictions.csv')
                if oof_ids:
                    pd.DataFrame(pv, index=pd.Index(oof_ids, name=UID), columns=LABELS).to_csv(out / 'oof_predictions.csv')
        xm.rendezvous(f'epoch{epoch}')
    if rank == 0:
        selected = history[-1] if select_last else min(history, key=lambda h: h['weak_val_bce'])
        metrics = {'selected_epoch': int(selected['epoch']), 'weak_val_bce': selected['weak_val_bce'],
                   'gold_auc': selected['gold_auc_descriptive'], 'fold': fold, 'oof_written': len(oof_ids),
                   'elapsed_seconds': time.time() - started, 'trained_epochs': epoch, 'configured_epochs': config['epochs'],
                   'world_size': world, 'trainer': f"train_xla.py ({os.environ.get('PJRT_DEVICE')}, bf16)"}
        (out / 'metrics.json').write_text(json.dumps(metrics, indent=2))
        (out / 'TRAINING_COMPLETE').write_text('complete\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--fold', help="override config['fold'] (k-fold index; needs n_folds and select='last')")
    parser.add_argument('--stop-at', type=float, default=0.0,
                        help='UNIX time by which training must end (0 = none); the epoch plan shrinks to fit')
    args = parser.parse_args()
    os.environ.setdefault('PJRT_DEVICE', 'TPU')
    if os.environ['PJRT_DEVICE'] == 'TPU':
        # Kaggle's TPU image presets single-process values (TPU_PROCESS_ADDRESSES=local, CLOUD_TPU_TASK_ID). torch_xla
        # only fills these with setdefault, so the 8 per-chip processes failed: "Expected 8 worker addresses, got 1".
        for var in ('TPU_PROCESS_ADDRESSES', 'CLOUD_TPU_TASK_ID'):
            if var in os.environ:
                print(f'unset {var}={os.environ.pop(var)!r}', flush=True)
    import torch_xla
    if hasattr(torch_xla, 'launch'):
        torch_xla.launch(run, args=(args,))
    else:   # older torch_xla
        import torch_xla.distributed.xla_multiprocessing as xmp
        xmp.spawn(run, args=(args,))


if __name__ == '__main__':
    main()
