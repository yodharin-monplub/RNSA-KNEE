"""Does training the CoAtNet head with the gold studies mixed in (heavily weighted) help? Free Kaggle test.

Uses model38's fine-tuned backbone as a frozen feature extractor over every 3-slice window of every
study (the 2%-98% 336px maxspan cache, four Kaggle parts), then trains the small attention head on
those features in seconds per run. Compares, by 5-fold cross-validation over the 58 gold studies:
  A0/A) head retrained on the Gemini relabels only (plain BCE / the project's confidence-weighted BCE);
  B-F) the same plus the gold fold-train studies at several weights, with or without the pairwise rank
     loss of the public 'DINOsaur V4' training notebook (luonmayman).
Held-out gold is never trained on in any arm. Also writes the features, the seed-averaged out-of-fold
predictions and a final head per arm. History: v1 ran out of GPU memory (no inference_mode); v2 crashed in
arm B on a 1-D/2-D weight mix-up after arm A had scored 0.9305 / 0.9332 against 0.9254 for model38's own
head - with a loss that the same mix-up had turned into plain BCE (kept here as arm A0).
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
MAX_WINDOWS = 62
GOLD_WEIGHT = 14.0
GOLD_REPEAT = 8


def extract_features(model, device, inputs, size):
    import hashlib
    manifests = sorted(p for p in inputs.rglob('cache_manifest.json') if 'dense-cache-part' in str(p) or 'dense_cache_part' in str(p) or 'rsna-knee-dense-cache-part' in str(p))
    if len(manifests) != 4:
        manifests = sorted(p for p in inputs.rglob('cache_manifest.json'))
    assert len(manifests) == 4, manifests
    ids, feats, masks = [], [], []
    for path in manifests:
        manifest = json.loads(path.read_text())
        assert manifest['geometry']['span'] == [.02, .98] and manifest['geometry']['size'] == 336
        for shard in manifest['shards']:
            shard_path = path.parent / shard['file']
            with shard_path.open('rb') as stream:
                assert hashlib.file_digest(stream, 'sha256').hexdigest() == shard['sha256']
            with np.load(shard_path, allow_pickle=False) as archive:
                for i, uid in enumerate(archive['ids'].tolist()):
                    images = windows(archive['volumes'][i], archive['masks'][i], size)
                    # inference_mode: without it autograd kept every chunk's graph alive (OOM on the T4).
                    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
                        encoded = torch.cat([model.backbone(chunk.to(device)) for chunk in images.split(12)])
                    n = min(len(encoded), MAX_WINDOWS)
                    block = np.zeros((MAX_WINDOWS, encoded.shape[1]), np.float16)
                    block[:n] = encoded[:n].float().cpu().numpy().astype(np.float16)
                    window_mask = np.zeros(MAX_WINDOWS, np.float32)
                    window_mask[:n] = 1
                    ids.append(uid); feats.append(block); masks.append(window_mask)
        print('features', len(ids), flush=True)
    return ids, np.stack(feats), np.stack(masks)


class Head(nn.Module):
    """model12's prediction head (norm + attention pooling + per-label linear), trainable alone."""
    def __init__(self, dim, state=None):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.att = nn.Sequential(nn.Linear(dim, 256), nn.Tanh(), nn.Dropout(.2), nn.Linear(256, 12))
        self.clsW = nn.Parameter(torch.zeros(12, dim))
        self.clsb = nn.Parameter(torch.zeros(12))
        if state is not None:
            self.load_state_dict(state, strict=True)

    def forward(self, features, mask):
        h = self.norm(features.float())
        attention = self.att(h).masked_fill(mask.unsqueeze(-1) < .5, -1e4).softmax(dim=1)
        pooled = torch.einsum('bkn,bkf->bnf', attention, h)
        return (pooled * self.clsW).sum(-1) + self.clsb


def rank_loss(logits, labels, rows):
    losses = []
    for j in range(12):
        positive = torch.nonzero(rows & (labels[:, j] > .5)).flatten()
        negative = torch.nonzero(rows & (labels[:, j] <= .5)).flatten()
        if len(positive) == 0 or len(negative) == 0:
            continue
        k = min(24, len(positive), len(negative))
        pos = positive[torch.randperm(len(positive), device=logits.device)[:k]]
        neg = negative[torch.randperm(len(negative), device=logits.device)[:k]]
        losses.append(F.softplus(-(logits[pos, j] - logits[neg, j])).mean())
    return torch.stack(losses).mean() if losses else logits.new_tensor(0.0)


def main():
    from sklearn.metrics import roc_auc_score
    started = time.time()
    device = torch.device('cuda')
    inputs = Path('/kaggle/input')
    comp = next(p.parent for p in inputs.rglob('train.csv') if (p.parent / 'test_series.csv').exists())
    weights = list(inputs.rglob('model38_best.pt'))
    assert len(weights) == 1, weights
    ck = torch.load(weights[0], map_location='cpu', weights_only=True)
    model = Classifier(ck['arch']); model.load_state_dict(ck['model'], strict=True)
    model = model.to(device).eval(); size = int(ck['res'])
    head_state = {k: v for k, v in ck['model'].items() if not k.startswith('backbone.')}
    feature_file = Path('/kaggle/working/window_features_fp16.npz')
    if feature_file.exists():
        with np.load(feature_file, allow_pickle=False) as archive:
            ids, feats, masks = archive['ids'].tolist(), archive['features'], archive['masks']
    else:
        ids, feats, masks = extract_features(model, device, inputs, size)
        np.savez(feature_file, ids=np.asarray(ids), features=feats, masks=masks)
    print('feature array', feats.shape, round(time.time() - started), flush=True)
    del model; torch.cuda.empty_cache()

    labels = pd.read_csv(next(inputs.rglob('targets_g38f_low.csv'))).set_index(UID)
    partition = pd.read_csv(next(inputs.rglob('partitions.csv'))).set_index(UID)
    gold = pd.read_csv(comp / 'train.csv').set_index(UID)
    position = {uid: i for i, uid in enumerate(ids)}
    gold_ids = sorted(u for u in partition.index[partition.split == 'gold'] if u in position)
    weak_ids = [u for u in partition.index if u in position and u not in set(gold_ids)]
    assert len(gold_ids) == 58
    y_weak = labels.loc[weak_ids, LABELS].to_numpy(np.float32)
    y_gold = gold.loc[gold_ids, LABELS].to_numpy(np.float32)
    weak_index = np.asarray([position[u] for u in weak_ids])
    gold_index = np.asarray([position[u] for u in gold_ids])
    folds = {u: i % 5 for i, u in enumerate(gold_ids)}
    feats_t = torch.from_numpy(feats)
    masks_t = torch.from_numpy(masks)

    # Reproduce model38's own head on the cached features as a sanity check.
    base = Head(feats.shape[-1], head_state).to(device).eval()
    with torch.no_grad():
        base_pred = torch.sigmoid(base(feats_t[gold_index].to(device), masks_t[gold_index].to(device))).cpu().numpy()
    macro = lambda p: float(np.mean([roc_auc_score(y_gold[:, j], p[:, j]) for j in range(12)]))
    report = {'model38_head_on_cached_features': macro(base_pred)}
    print(json.dumps(report), flush=True)

    def train_head(train_rows, train_y, train_w, gold_rows_mask, seed, epochs=6, use_rank=False, init=True):
        torch.manual_seed(seed); np.random.seed(seed)
        head = Head(feats.shape[-1], head_state if init else None).to(device)
        optimizer = torch.optim.AdamW(head.parameters(), lr=3e-4 if init else 7e-4, weight_decay=2e-3)
        rows = np.asarray(train_rows); ys = torch.from_numpy(train_y); ws = torch.from_numpy(train_w); gm = torch.from_numpy(gold_rows_mask)
        rng = np.random.default_rng(seed)
        for epoch in range(epochs):
            order = rng.permutation(len(rows))
            head.train()
            for start in range(0, len(order), 128):
                chunk = order[start:start + 128]
                f = feats_t[rows[chunk]].to(device); m = masks_t[rows[chunk]].to(device)
                target = ys[chunk].to(device); weight = ws[chunk].to(device); is_gold = gm[chunk].to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = head(f, m)
                bce = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
                # weight is per study AND per finding (n x 12). v2 of this script passed it 2-D into a 1-D
                # formula, which broadcast to (b, b, 12) and silently trained arm A with plain BCE.
                loss = (bce * weight).sum() / weight.sum().clamp(min=1)
                if use_rank:
                    loss = loss + 0.055 * rank_loss(logits, target, is_gold)
                loss.backward()
                nn.utils.clip_grad_norm_(head.parameters(), 2.0)
                optimizer.step()
        head.eval()
        return head

    def predict(head, index):
        with torch.no_grad():
            return torch.sigmoid(head(feats_t[index].to(device), masks_t[index].to(device))).cpu().numpy()

    conf_weak = (2 * np.abs(y_weak - .5)).astype(np.float32)
    weak_weights = {'conf': conf_weak, 'ones': np.ones_like(conf_weak)}
    # gold_weight = total loss weight of one gold study relative to one fully confident weak study; it is
    # spread over GOLD_REPEAT copies so that the pairwise rank loss sees gold pairs in most batches.
    settings = {
        'A0_weak_plain_bce': dict(weak='ones'),                      # what v2 trained by accident (0.9305 / 0.9332)
        'A_weak_confidence': dict(weak='conf'),                      # the project's usual loss
        'B_conf_gold14_rank': dict(weak='conf', gold_weight=14.0, use_rank=True),   # public DINOsaur recipe
        'C_conf_gold14': dict(weak='conf', gold_weight=14.0),
        'D_conf_gold4': dict(weak='conf', gold_weight=4.0),
        'E_conf_gold40': dict(weak='conf', gold_weight=40.0),
        'F_plain_gold14': dict(weak='ones', gold_weight=14.0),
    }
    SEEDS = 3

    def build(setting, train_gold):
        rows = list(weak_index); ys = [y_weak]; ws = [weak_weights[setting['weak']]]
        gmask = [np.zeros(len(weak_index), bool)]
        if 'gold_weight' in setting and len(train_gold):
            rep = np.repeat(gold_index[train_gold], GOLD_REPEAT)
            rows += list(rep)
            ys.append(np.repeat(y_gold[train_gold], GOLD_REPEAT, axis=0))
            ws.append(np.full((len(rep), 12), setting['gold_weight'] / GOLD_REPEAT, np.float32))
            gmask.append(np.ones(len(rep), bool))
        return rows, np.concatenate(ys), np.concatenate(ws), np.concatenate(gmask)

    out = Path('/kaggle/working')
    try:
        for name, setting in settings.items():
            scores, mean_oof = [], np.zeros_like(y_gold)
            for seed_offset in range(SEEDS):
                oof = np.zeros_like(y_gold)
                for fold in range(5):
                    held = [i for i, u in enumerate(gold_ids) if folds[u] == fold]
                    train_gold = [i for i, u in enumerate(gold_ids) if folds[u] != fold]
                    head = train_head(*build(setting, train_gold), seed=100 * seed_offset + fold,
                                      use_rank=setting.get('use_rank', False))
                    oof[held] = predict(head, gold_index[held])
                scores.append(macro(oof)); mean_oof += oof / SEEDS
                print(json.dumps({f'{name}_seed{seed_offset}': scores[-1], 'elapsed': round(time.time() - started)}), flush=True)
            report[name] = {'seeds': scores, 'mean': float(np.mean(scores)), 'std': float(np.std(scores)),
                            'seed_averaged_oof': macro(mean_oof),
                            'per_label': {c: float(roc_auc_score(y_gold[:, j], mean_oof[:, j])) for j, c in enumerate(LABELS)}}
            pd.DataFrame(mean_oof, index=gold_ids, columns=LABELS).rename_axis(UID).to_csv(out / f'oof_{name}.csv')
            # Final head on all the data this arm may use (weak only, or weak + all 58 gold), for a possible submission.
            final = train_head(*build(setting, list(range(len(gold_ids)))), seed=7, use_rank=setting.get('use_rank', False))
            torch.save({k: v.cpu() for k, v in final.state_dict().items()}, out / f'head_{name}.pt')
            if 'gold_weight' not in setting:
                report[name]['final_head_gold_auc'] = macro(predict(final, gold_index))
            print(json.dumps({name: report[name]}), flush=True)
    except Exception as exc:  # keep the kernel COMPLETE so the saved features stay usable as a data source
        import traceback
        report['error'] = traceback.format_exc()
        print(report['error'], flush=True)
    report['base_per_label'] = {c: float(roc_auc_score(y_gold[:, j], base_pred[:, j])) for j, c in enumerate(LABELS)}
    report['elapsed_seconds'] = time.time() - started
    (out / 'head_gold_cv.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


main()
'''

code = head + BODY
compile(code, 'head_gold_cv', 'exec')
api = KaggleApi(); api.authenticate()
out = ROOT / 'artifacts/head_gold_cv_notebook'
out.mkdir(parents=True, exist_ok=True)
(out / 'inference.py').write_text(code)
meta = {'id': 'yodharinmonplub/rsna-knee-head-gold-cv', 'title': 'RSNA Knee Head Gold CV',
        'code_file': 'inference.py', 'language': 'python', 'kernel_type': 'script', 'is_private': True,
        'enable_gpu': True, 'machine_shape': 'NvidiaTeslaT4', 'enable_internet': False,
        'dataset_sources': ['yodharinmonplub/rsna-knee-model38-weights', 'yodharinmonplub/rsna-knee-gemini-relabels'],
        'competition_sources': ['rsna-knee-abnormality-detection'],
        'kernel_sources': [f'yodharinmonplub/rsna-knee-dense-cache-part{i}' for i in range(4)], 'model_sources': []}
(out / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2))
print(api.kernels_push(str(out), acc='NvidiaTeslaT4'))
