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
import contextlib
import gc
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
    if args.xla_cache:
        # Persistent compilation cache: a relaunched process (--resume) reuses the compiled graphs instead of spending
        # ~20 min recompiling. Must run before any XLA computation; only local chip 0 writes.
        try:
            xr.initialize_cache(args.xla_cache, readonly=index != 0)
        except Exception as exc:
            print('xla cache not enabled:', type(exc).__name__, str(exc)[:120], flush=True)
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
    full_data = bool(config.get('full_data'))
    if full_data:
        # Full-train recipe (forum; our pseudo-label students): every non-gold study trains (train + weak_val). weak_val
        # stays only as an IN-SAMPLE monitor, so the last epoch must be kept; gold remains the honest check.
        assert config.get('select') == 'last' and (fold is None or fold == ''), 'full_data needs select last, no fold'
        train_ids = train_ids + val_ids
        if rank == 0:
            print(json.dumps({'full_data': True, 'train': len(train_ids), 'monitor_in_sample': len(val_ids)}), flush=True)
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

    # Precision: fp32 by default. The first full TPU run used bf16 autocast (XLA then also runs BatchNorm/LayerNorm in
    # bf16) and learned clearly worse than the fp16 GPU run of the same recipe (weak-val BCE 0.374 vs 0.316 at epoch 13,
    # gold similar); the TPU is fast enough (~5 min/epoch in bf16) to afford fp32. config xla_bf16=true restores bf16.
    use_bf16 = bool(config.get('xla_bf16', False))

    def amp():
        return torch.autocast('xla', dtype=torch.bfloat16) if use_bf16 else contextlib.nullcontext()

    def eval_loader(ids, labels_frame):
        data = EvalBags(args.cache, ids, labels_frame.loc[list(ids), LABELS].to_numpy(np.float32), res, max_eval)
        shard = DistributedSampler(data, num_replicas=world, rank=rank, shuffle=False, drop_last=False)
        n = int(config.get('eval_workers', 1))   # not persistent: 8 chips x resident loader workers may have OOM-killed a chip
        return data, DataLoader(data, batch_size=1, sampler=shard, num_workers=n)

    val_eval, gold_eval = eval_loader(val_ids, targets), eval_loader(gold_ids, gold)

    def evaluate(data, cpu_loader):
        dl = pl.MpDeviceLoader(cpu_loader, device)
        model.eval()
        preds, rows = [], []
        with torch.no_grad():
            for x, valid, target, i in dl:
                with amp():
                    feats = model.backbone(x.flatten(0, 1)).reshape(1, max_eval, -1)
                    logits = masked_predict(model, feats.float(), valid)
                preds.append(logits.float().sigmoid()); rows.append(i)
        p = torch.cat(preds) if preds else torch.zeros((0, len(LABELS)), device=device)
        r = torch.cat(rows).to(device) if rows else torch.zeros((0,), dtype=torch.long, device=device)
        p_all = xm.all_gather(p, dim=0); r_all = xm.all_gather(r, dim=0)
        p_all, r_all = p_all.cpu().numpy(), r_all.cpu().numpy()
        full = np.zeros((len(data), len(LABELS)), np.float32)
        full[r_all] = p_all                       # DistributedSampler pads by repeating; duplicates overwrite equally
        del dl, preds, rows, p, r, p_all, r_all
        gc.collect()
        return full, data.targets

    history, best, started = [], float('inf'), time.time()
    epoch, epochs_here = 0, 0
    last_path = out / 'last.pt'   # full training state after every epoch, so a crashed run can continue (--resume)
    if args.resume and last_path.exists():
        ck = torch.load(last_path, map_location='cpu', weights_only=False)
        model.load_state_dict(ck['model'])
        optimizer.load_state_dict(ck['optimizer'])      # state tensors follow their (XLA) parameters
        scheduler.load_state_dict(ck['scheduler'])
        history, best, epoch, plan_epochs = ck['history'], ck['best'], ck['epoch'], ck['plan_epochs']
        total = steps_per_epoch * plan_epochs
        if rank == 0:
            print(f'resumed after epoch {epoch} of {plan_epochs} (best weak-val BCE {best:.4f})', flush=True)
    while epoch < plan_epochs:
        epoch += 1
        epoch_started = time.time()
        sampler.set_epoch(epoch)
        model.train()
        train_dl = pl.MpDeviceLoader(loader, device)
        for step, (images, target) in enumerate(train_dl):
            optimizer.zero_grad(set_to_none=True)
            with amp():
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
        del train_dl
        gc.collect()
        pv, tv = evaluate(*val_eval)
        # train_model16's weak-val BCE: each soft target weighted by its confidence 2|t - 0.5| (the plain mean used before
        # 2026-09-30 counted near-0.5 labels fully and read ~0.06 higher than the GPU runs for the same quality).
        conf = 2 * np.abs(tv - .5)
        bce = -(tv * np.log(np.clip(pv, 1e-6, 1)) + (1 - tv) * np.log(np.clip(1 - pv, 1e-6, 1)))
        val_bce = float((bce * conf).sum() / max(float(conf.sum()), 1e-9))
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
        try:
            import psutil
            host_free = round(psutil.virtual_memory().available / 1e9, 1)
        except Exception:
            host_free = None
        # Host memory breakdown (2026-10-01: MemAvailable fell 201 -> 114 GB across process relaunches in model110
        # fold 0 and the epochs crawled once it was low): shared memory, page cache, and the largest processes.
        mem = None
        if rank == 0:
            try:
                import shutil as _shutil
                import psutil
                vm = psutil.virtual_memory()
                procs = []
                for p in psutil.process_iter(['pid', 'name', 'memory_info']):
                    rss = p.info['memory_info'].rss if p.info.get('memory_info') else 0
                    procs.append((rss, p.info['pid'], p.info['name']))
                procs.sort(reverse=True)
                try:
                    shm = round(_shutil.disk_usage('/dev/shm').used / 1e9, 1)
                except OSError:
                    shm = None    # no /dev/shm (local Windows test)
                mem = {'shm_used_gb': shm,
                       'cached_gb': round(getattr(vm, 'cached', 0) / 1e9, 1), 'shared_gb': round(getattr(vm, 'shared', 0) / 1e9, 1),
                       'python_rss_gb': round(sum(r for r, _, n in procs if 'python' in (n or '')) / 1e9, 1),
                       'n_python': sum(1 for _, _, n in procs if 'python' in (n or '')),
                       'top': [(n, pid, round(r / 1e9, 1)) for r, pid, n in procs[:4]]}
            except Exception as exc:
                mem = {'error': f'{type(exc).__name__}: {str(exc)[:80]}'}
        # Diagnostics for the epoch-14 slowdown / crash of the first TPU runs: XLA compilations so far and device memory.
        try:
            import torch_xla.debug.metrics as met
            compile_data = met.metric_data('CompileTime')
            compiles = int(compile_data[0]) if compile_data else 0
        except Exception:
            compiles = None
        try:
            hbm = {k: round(v / 1e9, 2) for k, v in xm.get_memory_info(device).items() if isinstance(v, (int, float))}
        except Exception:
            hbm = None
        history.append({'epoch': epoch, 'weak_val_bce': val_bce, 'gold_auc_descriptive': gold_auc,
                        'elapsed_seconds': time.time() - started, 'epoch_seconds': seconds, 'plan_epochs': plan_epochs,
                        'host_free_gb': host_free, 'xla_compiles': compiles, 'device_memory_gb': hbm, 'host_memory': mem})
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
        if rank == 0 and (not args.smoke or config.get('smoke_saves_last')):
            opt = optimizer.state_dict()
            opt = {'state': {k: {n: v.detach().cpu() if torch.is_tensor(v) else v for n, v in s.items()}
                             for k, s in opt['state'].items()}, 'param_groups': opt['param_groups']}
            torch.save({'model': {k: v.detach().cpu() for k, v in model.state_dict().items()}, 'optimizer': opt,
                        'scheduler': scheduler.state_dict(), 'history': history, 'best': best, 'epoch': epoch,
                        'plan_epochs': plan_epochs}, out / 'last.tmp')
            os.replace(out / 'last.tmp', last_path)
        epochs_here += 1
        if args.epochs_per_process and epochs_here >= args.epochs_per_process and epoch < plan_epochs:
            # Fresh processes every few epochs: the first TPU runs degraded after ~1.5 h in the same processes (host
            # memory -42 GB and ~15x slower epochs from epoch 14; the bf16 run crashed there). Rank 0 leaves a RELAUNCH
            # marker next to last.pt and every chip returns; the kernel relaunches with --resume. All chips reach this
            # point at the same epoch.
            xm.rendezvous(f'relaunch{epoch}')
            if rank == 0:
                (out / 'RELAUNCH').write_text(f'after epoch {epoch}\n')
                print(f'leaving after epoch {epoch} for a process relaunch (--epochs-per-process {args.epochs_per_process})',
                      flush=True)
            return
        if os.environ.get('XLA_TEST_CRASH_AFTER_EPOCH') == str(epoch) and not args.resume:
            raise SystemExit(f'test crash after epoch {epoch}')   # local test of --resume only
        xm.rendezvous(f'epoch{epoch}')
    if rank == 0:
        selected = history[-1] if select_last else min(history, key=lambda h: h['weak_val_bce'])
        metrics = {'selected_epoch': int(selected['epoch']), 'weak_val_bce': selected['weak_val_bce'],
                   'gold_auc': selected['gold_auc_descriptive'], 'fold': fold, 'full_data': full_data, 'oof_written': len(oof_ids),
                   'elapsed_seconds': time.time() - started, 'trained_epochs': epoch, 'configured_epochs': config['epochs'],
                   'world_size': world,
                   'trainer': f"train_xla.py ({os.environ.get('PJRT_DEVICE')}, {'bf16' if use_bf16 else 'fp32'})"}
        (out / 'metrics.json').write_text(json.dumps(metrics, indent=2))
        (out / 'TRAINING_COMPLETE').write_text('complete\n')
        last_path.unlink(missing_ok=True)   # ~0.9 GB of optimizer state is not needed once training is complete
        (out / 'RELAUNCH').unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--fold', help="override config['fold'] (k-fold index; needs n_folds and select='last')")
    parser.add_argument('--resume', action='store_true', help='continue from <output>/last.pt after a crash')
    parser.add_argument('--epochs-per-process', type=int, default=0,
                        help='exit with code 3 after this many epochs in one process (the caller relaunches with --resume)')
    parser.add_argument('--xla-cache', default='', help='persistent XLA compilation cache folder (reused across relaunches)')
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
