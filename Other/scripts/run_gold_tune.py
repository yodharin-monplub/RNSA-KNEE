"""Does a short fine-tune on the gold studies help? Five-fold test on Kaggle GPU (no rental cost).

Gold labels are the only ones with the same definitions as the hidden test. Each fold fine-tunes the
model38 checkpoint on 4/5 of the 58 gold studies and scores the held-out fold, against the same
checkpoint's score on that fold. Pixels come from the verified 2%-98% 336px pilot cache.
"""
import json
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / '.env')
os.environ.setdefault('KAGGLE_API_TOKEN', os.environ['KAGGLE_API_KEY'])
os.environ['KAGGLE_CONFIG_DIR'] = str(ROOT / '.kaggle')
from kaggle.api.kaggle_api_extended import KaggleApi

source = (ROOT / 'scripts/validate_dense_inference.py').read_text()
head = source[:source.index('REFERENCE_CSV = ')]

BODY = r'''
LR_SETTINGS = [{'tag': 'head_only', 'encoder_lr': 0.0, 'head_lr': 3e-4, 'epochs': 6},
               {'tag': 'stage_head', 'encoder_lr': 5e-6, 'head_lr': 3e-5, 'epochs': 4}]
WINDOWS = 12


def load_gold(cache, manifest, gold_ids):
    import hashlib
    volumes, masks = {}, {}
    for shard in manifest['shards']:
        path = cache / shard['file']
        with path.open('rb') as stream:
            assert hashlib.file_digest(stream, 'sha256').hexdigest() == shard['sha256']
        with np.load(path, allow_pickle=False) as archive:
            for i, uid in enumerate(archive['ids'].tolist()):
                if uid in gold_ids:
                    volumes[uid] = archive['volumes'][i]
                    masks[uid] = archive['masks'][i]
    return volumes, masks


def sample_windows(volume, mask, size, n, generator):
    valid = np.flatnonzero(mask)
    lo, hi = int(valid.min()), int(valid.max())
    centers = list(range(lo + 1, hi)) or [max(1, min((lo + hi) // 2, len(volume) - 2))]
    pick = torch.randperm(len(centers), generator=generator)
    pick = pick.repeat((n + len(centers) - 1) // len(centers))[:n]
    slices = np.stack([volume[centers[j] - 1:centers[j] + 2] for j in pick.tolist()]).astype(np.float32) / 255
    images = torch.from_numpy(slices)
    if images.shape[-1] != size:
        images = F.interpolate(images, size=(size, size), mode='bilinear', align_corners=False)
    return (images - torch.tensor([.485, .456, .406]).view(1, 3, 1, 1)) / torch.tensor([.229, .224, .225]).view(1, 3, 1, 1)


def forward_train(model, images):
    encoded = torch.cat([model.backbone(chunk) for chunk in images.split(12)])
    return model.predict_features(encoded.unsqueeze(0))


def main():
    import copy
    from sklearn.metrics import roc_auc_score
    started = time.time()
    device = torch.device('cuda')
    inputs = Path('/kaggle/input')
    cache = next(p.parent for p in inputs.rglob('cache_manifest.json'))
    manifest = json.loads((cache / 'cache_manifest.json').read_text())
    assert manifest['geometry']['span'] == [.02, .98] and manifest['geometry']['size'] == 336
    comp = next(p.parent for p in inputs.rglob('train.csv') if (p.parent / 'test_series.csv').exists())
    gold = pd.read_csv(comp / 'train.csv').dropna(subset=LABELS).set_index(UID)
    gold_ids = gold.index.tolist()
    volumes, masks = load_gold(cache, manifest, set(gold_ids))
    assert len(volumes) == 58
    weights = list(inputs.rglob('model38_best.pt'))
    assert len(weights) == 1, weights
    base = torch.load(weights[0], map_location='cpu', weights_only=True)
    arch, size = base['arch'], int(base['res'])
    y = gold.loc[gold_ids, LABELS].to_numpy()
    folds = {uid: i % 5 for i, uid in enumerate(sorted(gold_ids))}

    def evaluate(model, ids):
        model.eval()
        out = []
        for uid in ids:
            out.append(predict(model, windows(volumes[uid], masks[uid], size), device))
        return np.stack(out)

    model = Classifier(arch); model.load_state_dict(base['model'], strict=True); model = model.to(device)
    baseline = {uid: p for uid, p in zip(gold_ids, evaluate(model, gold_ids))}
    report = {'settings': []}
    for setting in LR_SETTINGS:
        tuned = {}
        for fold in range(5):
            train_ids = [u for u in gold_ids if folds[u] != fold]
            held = [u for u in gold_ids if folds[u] == fold]
            torch.manual_seed(2026 + fold)
            generator = torch.Generator().manual_seed(2026 + fold)
            model = Classifier(arch); model.load_state_dict(base['model'], strict=True); model = model.to(device)
            model.backbone.requires_grad_(setting['encoder_lr'] > 0)
            if setting['encoder_lr'] > 0:
                model.backbone.requires_grad_(False)
                model.backbone.stages[-1].requires_grad_(True)
            encoder = [p for p in model.backbone.parameters() if p.requires_grad]
            head_params = [p for n, p in model.named_parameters() if not n.startswith('backbone.')]
            groups = ([{'params': encoder, 'lr': setting['encoder_lr']}] if encoder else []) + [
                {'params': head_params, 'lr': setting['head_lr']}]
            optimizer = torch.optim.AdamW(groups, weight_decay=0.01)
            scaler = torch.amp.GradScaler('cuda')
            target_table = torch.from_numpy(gold.loc[train_ids, LABELS].to_numpy(np.float32))
            model.train() if encoder else model.eval()
            for epoch in range(setting['epochs']):
                order = torch.randperm(len(train_ids), generator=generator).tolist()
                for position in order:
                    uid = train_ids[position]
                    images = sample_windows(volumes[uid], masks[uid], size, WINDOWS, generator).to(device)
                    target = target_table[position].unsqueeze(0).to(device)
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast('cuda', dtype=torch.float16):
                        logits = forward_train(model, images).float()
                    loss = F.binary_cross_entropy_with_logits(logits, target)
                    assert torch.isfinite(loss)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(encoder + head_params, 1.0, error_if_nonfinite=False)
                    scaler.step(optimizer); scaler.update()
            for uid, prediction in zip(held, evaluate(model, held)):
                tuned[uid] = prediction
            print(json.dumps({'setting': setting['tag'], 'fold': fold, 'held': len(held),
                              'elapsed': round(time.time() - started)}), flush=True)
            del model, optimizer
            torch.cuda.empty_cache()
        before = np.stack([baseline[u] for u in gold_ids])
        after = np.stack([tuned[u] for u in gold_ids])
        macro = lambda p: float(np.mean([roc_auc_score(y[:, j], p[:, j]) for j in range(12)]))
        report['settings'].append({'tag': setting['tag'], 'epochs': setting['epochs'],
                                   'gold_auc_baseline': macro(before), 'gold_auc_tuned_oof': macro(after),
                                   'per_label_delta': [round(roc_auc_score(y[:, j], after[:, j])
                                                             - roc_auc_score(y[:, j], before[:, j]), 4)
                                                       for j in range(12)]})
        pd.DataFrame(after, index=pd.Index(gold_ids, name=UID), columns=LABELS).to_csv(
            f"/kaggle/working/gold_tuned_{setting['tag']}.csv")
        print(json.dumps(report['settings'][-1]), flush=True)
    report['elapsed_seconds'] = time.time() - started
    Path('/kaggle/working/gold_tune_report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


main()
'''

code = head + BODY
compile(code, 'gold_tune', 'exec')
api = KaggleApi(); api.authenticate()
out = ROOT / 'artifacts/gold_tune_notebook'
out.mkdir(parents=True, exist_ok=True)
(out / 'inference.py').write_text(code)
meta = {'id': 'yodharinmonplub/rsna-knee-gold-tune', 'title': 'RSNA Knee Gold Tune',
        'code_file': 'inference.py', 'language': 'python', 'kernel_type': 'script', 'is_private': True,
        'enable_gpu': True, 'machine_shape': 'NvidiaTeslaT4', 'enable_internet': False,
        'dataset_sources': ['yodharinmonplub/rsna-knee-model38-weights'],
        'competition_sources': ['rsna-knee-abnormality-detection'],
        'kernel_sources': ['yodharinmonplub/rsna-knee-dense-cache-pilot'], 'model_sources': []}
(out / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2))
print(api.kernels_push(str(out), acc='NvidiaTeslaT4'))
