"""Fine-tune the last CoAtNet stage on verified dense pixels and v4 weak targets."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
from knee.dense import DenseCache
from cloud.prepare_dense_cache import EXPECTED_GEOMETRY, SLOTS44, SLOTS128
from model12.inference import Classifier, LABELS, UID, windows


class Studies(Dataset):
    def __init__(self, cache, ids, targets, n_windows=None, size=384):
        self.size = size
        self.cache = DenseCache(cache, ids)
        self.targets = targets.loc[ids, LABELS].to_numpy(dtype=np.float32)
        self.n_windows = n_windows

    def __len__(self):
        return len(self.cache)

    def __getitem__(self, i):
        volume, mask = self.cache[i]
        if self.n_windows is None:
            images = windows(volume, mask, self.size)
        else:
            # Same centers and sampling as windows()+randperm, but only the chosen windows are built.
            valid = np.flatnonzero(mask)
            if len(valid) < 3:
                valid = np.arange(3)
            lo, hi = int(valid.min()), int(valid.max())
            centers = list(range(lo+1, hi)) or [max(1, min((lo+hi)//2, len(volume)-2))]
            n = len(centers)
            idx = torch.randperm(n)
            idx = idx.repeat((self.n_windows+n-1)//n)[:self.n_windows]
            slices = torch.from_numpy(np.stack([volume[centers[j]-1:centers[j]+2] for j in idx.tolist()]).astype(np.float32) / 255)
            if slices.shape[-1] != self.size:
                slices = F.interpolate(slices, size=(self.size, self.size), mode='bilinear', align_corners=False)
            images = (slices - torch.tensor([.485,.456,.406]).view(1,3,1,1)) / torch.tensor([.229,.224,.225]).view(1,3,1,1)
        return images, torch.from_numpy(self.targets[i])


CHUNK = 12  # windows per backbone call; config['chunk'] raises it for small BatchNorm backbones


def forward(model, images):
    b, k = images.shape[:2]
    encoded = torch.cat([model.backbone(chunk) for chunk in images.flatten(0, 1).split(CHUNK)])
    return model.predict_features(encoded.reshape(b, k, -1))


@torch.inference_mode()
def evaluate(model, dataset, device):
    model.eval()
    numerator = denominator = 0.0
    predictions = []
    for images, target in DataLoader(dataset, batch_size=1, num_workers=0):
        images, target = images.to(device), target.to(device)
        with torch.autocast('cuda', dtype=torch.float16):
            logits = forward(model, images)
        probability = logits.float().sigmoid()
        assert torch.isfinite(probability).all()
        confidence = 2*(target-.5).abs()
        numerator += float((F.binary_cross_entropy_with_logits(logits.float(), target, reduction='none')*confidence).sum())
        denominator += float(confidence.sum())
        predictions.append(probability.cpu().numpy()[0])
    assert denominator > 0
    return numerator/denominator, np.asarray(predictions)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', default='artifacts/model16')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--config', default=str(ROOT/'model16/config.json'))
    parser.add_argument('--cache-size', type=int, choices=[256,336,384], default=336)
    parser.add_argument('--span-start', type=float, default=.02)
    parser.add_argument('--span-end', type=float, default=.98)
    parser.add_argument('--slots', type=int, choices=[64, 44, 128], default=64)
    # model71: one config, five k-fold runs; the shell loop passes the fold index here.
    parser.add_argument('--fold', help="override config['fold'] (k-fold index, or the legacy A/B half)")
    args = parser.parse_args()
    config_path = Path(args.config)
    config = json.loads(config_path.read_text())
    random.seed(config['seed']); np.random.seed(config['seed']); torch.manual_seed(config['seed'])
    global CHUNK
    CHUNK = int(config.get('chunk', 12))
    assert torch.cuda.is_available(), 'Run GPU training on the authorized rental'
    device = torch.device('cuda')
    torch.set_num_threads(4)
    cache = DenseCache(args.cache)
    assert args.smoke or not cache.verified['pilot']
    expected_geometry = dict(EXPECTED_GEOMETRY)
    expected_geometry['size'] = args.cache_size
    assert (args.span_start,args.span_end) in ((.02,.98),(.06,.94))
    expected_geometry['span'] = [args.span_start,args.span_end]
    if args.slots == 44:
        expected_geometry['slots'] = SLOTS44
    if args.slots == 128:
        expected_geometry['slots'] = SLOTS128
    assert cache.verified['geometry'] == expected_geometry
    assert cache.verified['source_metadata_sha256'] == hashlib.sha256((ROOT/'data/train.csv').read_bytes()).hexdigest()
    partition = pd.read_csv(ROOT/'artifacts/partitions/partitions.csv').set_index(UID)
    targets = pd.read_csv(ROOT/config['targets']).set_index(UID)
    gold = pd.read_csv(ROOT/'data/train.csv').set_index(UID)
    assert partition.index.is_unique and targets.index.is_unique
    assert set(partition.index) == set(targets.index) == set(gold.index)
    assert np.isfinite(targets[LABELS]).all().all()
    assert ((targets[LABELS]>=0)&(targets[LABELS]<=1)).all().all()
    train_ids = partition.index[partition.split=='train'].tolist()
    val_ids = partition.index[partition.split=='weak_val'].tolist()
    gold_ids = partition.index[partition.split=='gold'].tolist()
    assert len(gold_ids)==58 and gold.loc[gold_ids,LABELS].notna().all().all()
    assert set(partition.loc[train_ids,'report_hash']).isdisjoint(partition.loc[val_ids+gold_ids,'report_hash'])
    fold = args.fold if args.fold is not None else config.get('fold')
    oof_ids = []
    if fold is not None and fold != '':
        # Folds of the non-gold studies, split by report group so duplicates of one report stay together.
        assert config.get('select') == 'last'
        pool = train_ids + val_ids
        digest = {uid: int(hashlib.md5(str(partition.loc[uid, 'report_hash']).encode()).hexdigest(), 16)
                  for uid in pool}
        if str(fold) in ('A', 'B'):
            # model68/69: train on one half, predict the other.
            want = 0 if fold == 'A' else 1
            train_ids = [uid for uid in pool if digest[uid] % 2 == want]
            oof_ids = [uid for uid in pool if digest[uid] % 2 != want]
        else:
            # model71: standard k-fold - train on the other k-1 folds, predict this one.
            n_folds, k = int(config['n_folds']), int(fold)
            assert 0 <= k < n_folds
            train_ids = [uid for uid in pool if digest[uid] % n_folds != k]
            oof_ids = [uid for uid in pool if digest[uid] % n_folds == k]
        assert train_ids and oof_ids and set(train_ids).isdisjoint(gold_ids)
        print(json.dumps({'fold': fold, 'train': len(train_ids), 'oof': len(oof_ids)}), flush=True)
    refit = config.get('select') == 'last'
    if refit and not fold:
        # Full-data refit: epoch count fixed in advance by an earlier validated run.
        assert config.get('include_weak_val_in_train') and config['epochs'] >= 1
        train_ids = train_ids + val_ids
        assert set(train_ids).isdisjoint(gold_ids)
    if args.smoke:
        # Fill one full batch so the smoke run exercises training-time memory.
        train_ids = [uid for uid in train_ids if uid in cache.index.index][:max(2, config['batch_size'])]
        assert train_ids
    else:
        assert set(cache.index.index) == set(partition.index)
    scratch = config.get('init') == 'imagenet'
    res = int(config.get('res', 384))
    train_set = Studies(args.cache, train_ids, targets, config['train_windows'], res)
    if scratch:
        # New model family: ImageNet backbone, every parameter trained, no teacher.
        import timm
        arch = config['arch']; student = Classifier(arch, img_size=config.get('backbone_img_size'))
        # Rentals may not reach the HF hub; weights come from a verified local file when configured.
        overlay = {'file': str(ROOT/config['pretrained_file'])} if config.get('pretrained_file') else None
        student.backbone = timm.create_model(arch, pretrained=True, num_classes=0, in_chans=3, global_pool='avg',
                                             pretrained_cfg_overlay=overlay,
                                             **({'img_size': config['backbone_img_size']}
                                                if config.get('backbone_img_size') else {}))
        assert student.backbone.num_features == student.clsW.shape[1]
        teacher = None
        student = student.to(device).requires_grad_(True)
        if config.get('grad_checkpointing'):
            student.backbone.set_grad_checkpointing(True)
        trainable_stages = student.backbone
    else:
        ck = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
        assert list(ck.get('lab', LABELS)) == LABELS and int(ck['res']) == 384
        arch = ck['arch']; student = Classifier(arch)
        student.load_state_dict(ck['model'], strict=True); del ck
        teacher = copy.deepcopy(student).to(device).eval().requires_grad_(False)
        student = student.to(device)
        student.backbone.requires_grad_(False)
        trainable_stages = student.backbone.stages[-int(config.get('unfreeze_stages', 1)):]
        trainable_stages.requires_grad_(True)
        if config.get('grad_checkpointing'):
            # model79: unfreezing every stage at 384 px needs the same memory saving as the scratch runs
            student.backbone.set_grad_checkpointing(True)
    external = aux_heads = None
    if config.get('external'):
        # Auxiliary supervision from MRNet / fastMRI+ / KneeMRI through separate heads (knee/external.py).
        from knee.external import AuxHeads, ExternalBags, external_loss
        spec = config['external']
        external = ExternalBags({k: ROOT/v for k, v in spec['roots'].items()}, ROOT/spec['scratch'],
                                config['train_windows'], res, config['seed'])
        aux_heads = AuxHeads(student.clsW.shape[1]).to(device)
        print(json.dumps({'external_exams': external.counts}), flush=True)
    encoder = [p for p in student.backbone.parameters() if p.requires_grad]
    head = [p for name,p in student.named_parameters() if not name.startswith('backbone.')]
    if aux_heads is not None:
        head = head + list(aux_heads.parameters())
    assert encoder and all(p.requires_grad for p in head)
    optimizer = torch.optim.AdamW([{'params':encoder,'lr':config['encoder_lr']},
                                  {'params':head,'lr':config['head_lr']}], weight_decay=config['weight_decay'])
    scheduler = None
    scaler = torch.amp.GradScaler('cuda')
    out = Path(args.output); out.mkdir(parents=True,exist_ok=True)
    if scratch:
        checkpoint_hash = 'timm-pretrained:' + arch
    else:
        with Path(args.checkpoint).open('rb') as stream:
            checkpoint_hash = hashlib.file_digest(stream,'sha256').hexdigest()
    provenance = {'initial_checkpoint_sha256':checkpoint_hash,
                  'targets_sha256':hashlib.sha256((ROOT/config['targets']).read_bytes()).hexdigest(),
                  'partition_sha256':hashlib.sha256((ROOT/'artifacts/partitions/partitions.csv').read_bytes()).hexdigest(),
                  'config':config,'cache':cache.verified,'smoke':args.smoke}
    (out/'provenance.json').write_text(json.dumps(provenance,indent=2))
    started = time.time(); history = []

    def save(epoch, loss):
        state = {'model':{k:v.detach().cpu() for k,v in student.state_dict().items()},
                 'arch':arch,'res':res,'lab':LABELS,'epoch':epoch,'weak_val_bce':loss,
                 'training_config':config}
        torch.save(state, out/'best.tmp'); (out/'best.tmp').replace(out/'best.pt')

    if not args.smoke:
        validation = Studies(args.cache,val_ids,targets,size=res)
        gold_set = Studies(args.cache,gold_ids,gold,size=res)
        if scratch:
            best = float('inf')
            history.append({'epoch':0,'weak_val_bce':None,'note':'untrained head; not a selection candidate'})
        else:
            baseline_loss, _ = evaluate(student, validation, device)
            _, baseline_gold = evaluate(student, gold_set, device)
            best = baseline_loss; save(0,best)
            history.append({'epoch':0,'weak_val_bce':best,'gold_auc':float(roc_auc_score(gold.loc[gold_ids,LABELS],baseline_gold))})
        print(json.dumps(history[-1]),flush=True)
    loader = DataLoader(train_set,batch_size=config['batch_size'],shuffle=True,
                        num_workers=0 if args.smoke else config['workers'],pin_memory=True)
    if 'min_lr_ratio' in config:
        total = len(loader)*config['epochs']
        ratio = float(config['min_lr_ratio'])
        warm = int(float(config.get('warmup_ratio', 0))*total)

        def lr_factor(step_index):
            if step_index < warm:
                return (step_index+1)/warm
            t = min(step_index-warm, total-warm)/max(1, total-warm)
            return ratio+(1-ratio)*.5*(1+np.cos(np.pi*t))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    for epoch in range(1, (1 if args.smoke else config['epochs'])+1):
        student.eval(); trainable_stages.train(); student.att.train()
        if scratch: student.train()
        for step,(images,target) in enumerate(loader):
            images,target = images.to(device),target.to(device)
            confidence = 2*(target-.5).abs()
            optimizer.zero_grad(set_to_none=True)
            if config.get('augment'):
                from knee.external import augment
                with torch.no_grad(): images = augment(images)
            with torch.autocast('cuda',dtype=torch.float16):
                logits = forward(student,images).float()
                if config.get('plain_bce'):
                    supervised = F.binary_cross_entropy_with_logits(logits,target)
                else:
                    supervised = (F.binary_cross_entropy_with_logits(logits,target,reduction='none')*confidence).sum()/confidence.sum().clamp(min=1)
                loss = supervised
                if teacher is not None and config['teacher_weight'] > 0:
                    with torch.no_grad(): teacher_prob = forward(teacher, images).float().sigmoid()
                    loss = loss + config['teacher_weight']*F.binary_cross_entropy_with_logits(logits,teacher_prob)
                if external is not None:
                    ext_images, ext_target, ext_mask, ext_window, ext_window_mask = [t.to(device) for t in external.batch(config['batch_size'])]
                    if config.get('augment'):
                        with torch.no_grad(): ext_images = augment(ext_images)
                    eb, ek = ext_images.shape[:2]
                    ext_features = torch.cat([student.backbone(chunk) for chunk in ext_images.flatten(0, 1).split(CHUNK)]).reshape(eb, ek, -1)
                    ext_loss, ext_exam, ext_win = external_loss(aux_heads, ext_features, ext_target, ext_mask, ext_window, ext_window_mask)
                    loss = loss + config['external']['weight']*ext_loss
                    if step % 100 == 0: print(json.dumps({'external_exam_bce': ext_exam, 'external_window_bce': ext_win}), flush=True)
            assert torch.isfinite(loss)
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            # fp16 overflow gives a non-finite norm; GradScaler then skips the step and lowers its scale.
            norm = nn.utils.clip_grad_norm_(encoder+head,config['gradient_clip'],error_if_nonfinite=False)
            assert teacher is None or all(p.grad is None for p in teacher.parameters())
            scaler.step(optimizer); scaler.update()
            if scheduler is not None: scheduler.step()
            if step % 100 == 0: print(json.dumps({'epoch':epoch,'step':step,'loss':float(loss.detach()),'gradient_norm':float(norm)}),flush=True)
            if args.smoke:
                (out/'SMOKE_PASS.json').write_text(json.dumps({'loss':float(loss.detach()),'weak_training_cases':train_ids,'trainable_parameters':sum(p.numel() for p in encoder+head)}))
                return
        val_loss,_ = evaluate(student,validation,device)
        history.append({'epoch':epoch,'weak_val_bce':val_loss,'weak_val_in_train':refit,
                        'elapsed_seconds':time.time()-started})
        if config.get('log_gold_each_epoch'):
            _, epoch_gold = evaluate(student, gold_set, device)
            history[-1]['gold_auc_descriptive'] = float(roc_auc_score(gold.loc[gold_ids,LABELS], epoch_gold))
        if refit or val_loss < best: best=val_loss;save(epoch,best)
        (out/'history.json').write_text(json.dumps(history,indent=2));print(json.dumps(history[-1]),flush=True)
    ck = torch.load(out/'best.pt',map_location='cpu',weights_only=True)
    student.load_state_dict(ck['model'],strict=True)
    _, pred = evaluate(student,gold_set,device)
    pd.DataFrame(pred,index=pd.Index(gold_ids,name=UID),columns=LABELS).to_csv(out/'gold_predictions.csv')
    if oof_ids:
        _, oof = evaluate(student, Studies(args.cache, oof_ids, targets, size=res), device)
        pd.DataFrame(oof, index=pd.Index(oof_ids, name=UID), columns=LABELS).to_csv(out/'oof_predictions.csv')
        print(json.dumps({'oof_written': len(oof_ids)}), flush=True)
    metrics={'selected_epoch':ck['epoch'],'weak_val_bce':best,'gold_auc':float(roc_auc_score(gold.loc[gold_ids,LABELS],pred)),
             'elapsed_seconds':time.time()-started,'notes':'Gold is descriptive development evidence; epoch chosen by weak BCE including baseline.'}
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2));print(json.dumps(metrics),flush=True)
    (out/'TRAINING_COMPLETE').write_text('Model16 complete; artifacts ready for backup.\n')


if __name__=='__main__': main()
