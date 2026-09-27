"""Retrain only the prediction head of one fine-tuned CoAtNet arm on frozen full-bag features (free Kaggle T4).

Follow-up to scripts/run_head_gold_cv.py, where a head retrained with plain soft-label BCE on the Gemini relabels
(no gold involved) lifted the maxspan arm from 0.9254 to 0.9317 +/- 0.0010 gold AUC and the four-arm blend from
0.9305 to 0.9331. Recipe fixed there and NOT tuned here: start from the arm's own head, AdamW lr 3e-4, wd 2e-3,
batch 128 studies, all windows per study, 6 epochs, grad clip 2.0.

Per arm the kernel: encodes every window of every study once with the arm's frozen backbone (or reuses a features
file found in its inputs), checks that the arm's own head reproduces its known gold AUC from those features, trains
three seeds (gold AUC each, never used for training or selection) plus the final head (seed 7), and writes a full
checkpoint `model61_<arm>_best.pt` (same layout as the arm's checkpoint, head replaced) plus gold predictions.

Usage: python scripts/run_head_retrain.py maxspan|dense|v8|fine [TAG TARGETS_FILE]
  no TAG          -> kernel rsna-knee-head-retrain-<arm>, targets_g38f_low.csv, output model61_<arm>_best.pt
                     (the model61 runs; the dense/v8/fine kernels also saved their features)
  TAG TARGETS_FILE -> kernel rsna-knee-head-retrain-<TAG>-<arm>: same recipe on another targets file found in the
                     attached datasets (upload it to yodharinmonplub/rsna-knee-gemini-relabels first), features
                     reused from the arm's first kernel (~15 min instead of ~3 h), output <TAG>_<arm>_best.pt.
"""
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / '.env')
os.environ.setdefault('KAGGLE_API_TOKEN', os.environ.get('KAGGLE_API_KEY') or os.environ['KAGGLE_API_KEY_YODHAIRNMONPLUB'])
os.environ['KAGGLE_CONFIG_DIR'] = str(ROOT / '.kaggle')
from kaggle.api.kaggle_api_extended import KaggleApi

ARMS = {
    'maxspan': dict(weights='rsna-knee-model38-weights', file='model38_best.pt', cache='rsna-knee-dense-cache-part{}',
                    known_gold=0.925370, extra_kernels=['yodharinmonplub/rsna-knee-head-gold-cv']),
    'dense': dict(weights='rsna-knee-model39-weights', file='model39_best.pt', cache='rsna-knee-dense-cache-384-part{}',
                  known_gold=0.920484, extra_kernels=[]),
    'v8': dict(weights='rsna-knee-model40-weights', file='model40_best.pt', cache='rsna-knee-dense-cache-384-6-94-part{}-s44',
               known_gold=0.919671, extra_kernels=[]),
    'fine': dict(weights='rsna-knee-model41-weights', file='model41_best.pt', cache='rsna-knee-dense-cache-336-6-94-part{}',
                 known_gold=0.915969, extra_kernels=[]),
    # model97 = model75 (CoAtNet-2 384 px trained from ImageNet, 0.939 public alone) with its head retrained on full bags
    'model75': dict(weights='rsna-knee-fleet-weights', file='model75_best.pt', cache='rsna-knee-dense-cache-part{}',
                    known_gold=0.920393, extra_kernels=[], prefix='model97'),
}
arm = sys.argv[1]
spec = dict(ARMS[arm])
tag = sys.argv[2] if len(sys.argv) > 2 else ''
targets_file = sys.argv[3] if len(sys.argv) > 3 else 'targets_g38f_low.csv'
assert bool(tag) == (len(sys.argv) > 3), 'give TAG and TARGETS_FILE together'
if tag and not spec['extra_kernels']:
    spec['extra_kernels'] = [f'yodharinmonplub/rsna-knee-head-retrain-{arm}']  # saved window_features_fp16.npz
output_prefix = tag or spec.get('prefix', 'model61')

source = (ROOT / 'scripts/validate_dense_inference.py').read_text()
head = source[:source.index('REFERENCE_CSV = ')]

BODY = r'''
ARM = __ARM__
WEIGHT_FILE = __FILE__
KNOWN_GOLD = __KNOWN__
TARGETS_FILE = __TARGETS__
OUTPUT_PREFIX = __PREFIX__
MAX_WINDOWS = 62


def extract_features(model, device, inputs, size):
    import hashlib
    manifests = sorted(p for p in inputs.rglob('cache_manifest.json'))
    assert len(manifests) == 4, manifests
    ids, feats, masks = [], [], []
    for path in manifests:
        manifest = json.loads(path.read_text())
        print('cache geometry', manifest['geometry'], flush=True)
        for shard in manifest['shards']:
            shard_path = path.parent / shard['file']
            with shard_path.open('rb') as stream:
                assert hashlib.file_digest(stream, 'sha256').hexdigest() == shard['sha256']
            with np.load(shard_path, allow_pickle=False) as archive:
                for i, uid in enumerate(archive['ids'].tolist()):
                    images = windows(archive['volumes'][i], archive['masks'][i], size)
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
    def __init__(self, dim, state):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.att = nn.Sequential(nn.Linear(dim, 256), nn.Tanh(), nn.Dropout(.2), nn.Linear(256, 12))
        self.clsW = nn.Parameter(torch.zeros(12, dim))
        self.clsb = nn.Parameter(torch.zeros(12))
        self.load_state_dict(state, strict=True)

    def forward(self, features, mask):
        h = self.norm(features.float())
        attention = self.att(h).masked_fill(mask.unsqueeze(-1) < .5, -1e4).softmax(dim=1)
        pooled = torch.einsum('bkn,bkf->bnf', attention, h)
        return (pooled * self.clsW).sum(-1) + self.clsb


def main():
    from sklearn.metrics import roc_auc_score
    started = time.time()
    # Head training on cached features is small enough for a CPU kernel (used when the account's two GPU batch
    # slots are taken); only the feature extraction needs the GPU.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    inputs = Path('/kaggle/input')
    out = Path('/kaggle/working')
    comp = next(p.parent for p in inputs.rglob('train.csv') if (p.parent / 'test_series.csv').exists())
    weights = list(inputs.rglob(WEIGHT_FILE))
    assert len(weights) == 1, weights
    ck = torch.load(weights[0], map_location='cpu', weights_only=True)
    model = Classifier(ck['arch']); model.load_state_dict(ck['model'], strict=True)
    size = int(ck['res'])
    head_state = {k: v for k, v in ck['model'].items() if not k.startswith('backbone.')}
    cached = list(inputs.rglob('window_features_fp16.npz'))
    if cached:
        with np.load(cached[0], allow_pickle=False) as archive:
            ids, feats, masks = archive['ids'].tolist(), archive['features'], archive['masks']
        print('reused features from', cached[0], flush=True)
    else:
        assert device.type == 'cuda', 'feature extraction needs a GPU kernel'
        model = model.to(device).eval()
        ids, feats, masks = extract_features(model, device, inputs, size)
        np.savez(out / 'window_features_fp16.npz', ids=np.asarray(ids), features=feats, masks=masks)
        model = model.cpu()
    torch.cuda.empty_cache()
    print('feature array', feats.shape, round(time.time() - started), flush=True)

    target_paths = list(inputs.rglob(TARGETS_FILE))
    assert len(target_paths) == 1, target_paths
    labels = pd.read_csv(target_paths[0]).set_index(UID)
    assert labels.index.is_unique and np.isfinite(labels[LABELS].to_numpy(float)).all()
    partition = pd.read_csv(next(inputs.rglob('partitions.csv'))).set_index(UID)
    gold = pd.read_csv(comp / 'train.csv').set_index(UID)
    position = {uid: i for i, uid in enumerate(ids)}
    gold_ids = sorted(u for u in partition.index[partition.split == 'gold'] if u in position)
    weak_ids = [u for u in partition.index if u in position and u not in set(gold_ids)]
    assert len(gold_ids) == 58 and len(weak_ids) > 4000
    y_weak = torch.from_numpy(labels.loc[weak_ids, LABELS].to_numpy(np.float32))
    y_gold = gold.loc[gold_ids, LABELS].to_numpy(np.float32)
    weak_index = np.asarray([position[u] for u in weak_ids])
    gold_index = np.asarray([position[u] for u in gold_ids])
    feats_t, masks_t = torch.from_numpy(feats), torch.from_numpy(masks)
    macro = lambda p: float(np.mean([roc_auc_score(y_gold[:, j], p[:, j]) for j in range(12)]))

    def predict(head_module):
        head_module.eval()
        with torch.no_grad():
            return torch.sigmoid(head_module(feats_t[gold_index].to(device), masks_t[gold_index].to(device))).cpu().numpy()

    base = macro(predict(Head(feats.shape[-1], head_state).to(device)))
    report = {'arm': ARM, 'own_head_on_cached_features': base, 'known_gold_auc': KNOWN_GOLD}
    print(json.dumps(report), flush=True)
    assert abs(base - KNOWN_GOLD) < 0.002, 'cached features do not reproduce the arm'

    def train_head(seed, epochs=6):
        torch.manual_seed(seed)
        head_module = Head(feats.shape[-1], head_state).to(device)
        optimizer = torch.optim.AdamW(head_module.parameters(), lr=3e-4, weight_decay=2e-3)
        rng = np.random.default_rng(seed)
        for _ in range(epochs):
            order = rng.permutation(len(weak_index))
            head_module.train()
            for start in range(0, len(order), 128):
                chunk = order[start:start + 128]
                logits = head_module(feats_t[weak_index[chunk]].to(device), masks_t[weak_index[chunk]].to(device))
                loss = F.binary_cross_entropy_with_logits(logits, y_weak[chunk].to(device))
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(head_module.parameters(), 2.0)
                optimizer.step()
        return head_module

    seed_predictions = []
    for seed in (0, 1, 2):
        seed_predictions.append(predict(train_head(seed)))
        report[f'seed{seed}_gold_auc'] = macro(seed_predictions[-1])
        print(json.dumps({f'seed{seed}_gold_auc': report[f'seed{seed}_gold_auc']}), flush=True)
    report['seed_mean'] = float(np.mean([report[f'seed{s}_gold_auc'] for s in (0, 1, 2)]))
    final = train_head(7)
    final_prediction = predict(final)
    report['final_head_gold_auc'] = macro(final_prediction)
    report['final_per_label'] = {c: float(roc_auc_score(y_gold[:, j], final_prediction[:, j])) for j, c in enumerate(LABELS)}
    pd.DataFrame(final_prediction, index=gold_ids, columns=LABELS).rename_axis(UID).to_csv(out / f'gold_predictions_{ARM}.csv')

    # Full checkpoint in the arm's own layout, head replaced; verify it end to end through Classifier on gold features.
    new_state = dict(ck['model'])
    for key, value in final.state_dict().items():
        assert key in new_state and new_state[key].shape == value.shape, key
        new_state[key] = value.detach().cpu().to(new_state[key].dtype)
    checkpoint = dict(ck); checkpoint['model'] = new_state
    checkpoint['head_retrain'] = {'source': WEIGHT_FILE, 'targets': TARGETS_FILE,
                                  'recipe': 'plain BCE, lr 3e-4, wd 2e-3, bs 128, 6 epochs, seed 7'}
    report['targets'] = TARGETS_FILE
    torch.save(checkpoint, out / f'{OUTPUT_PREFIX}_{ARM}_best.pt')
    verify = Classifier(ck['arch']); verify.load_state_dict(new_state, strict=True); verify = verify.to(device).eval()
    with torch.no_grad():
        f = feats_t[gold_index[:8]].to(device).float(); m = masks_t[gold_index[:8]]
        rows = [torch.sigmoid(verify.predict_features(f[i:i + 1, :int(m[i].sum())])).cpu().numpy()[0] for i in range(8)]
    report['checkpoint_max_abs_diff'] = float(np.abs(np.stack(rows) - final_prediction[:8]).max())
    assert report['checkpoint_max_abs_diff'] < 1e-4, report['checkpoint_max_abs_diff']
    report['elapsed_seconds'] = time.time() - started
    (out / f'head_retrain_{ARM}.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


main()
'''

code = head + (BODY.replace('__ARM__', repr(arm)).replace('__FILE__', repr(spec['file']))
               .replace('__KNOWN__', repr(spec['known_gold'])).replace('__TARGETS__', repr(targets_file))
               .replace('__PREFIX__', repr(output_prefix)))
compile(code, 'head_retrain', 'exec')
api = KaggleApi(); api.authenticate()
suffix = f'{tag}-{arm}' if tag else arm
out = ROOT / f'artifacts/head_retrain_{suffix.replace("-", "_")}_notebook'
out.mkdir(parents=True, exist_ok=True)
(out / 'inference.py').write_text(code)
meta = {'id': f'yodharinmonplub/rsna-knee-head-retrain-{suffix}', 'title': f'RSNA Knee Head Retrain {suffix.replace("-", " ")}',
        'code_file': 'inference.py', 'language': 'python', 'kernel_type': 'script', 'is_private': True,
        'enable_gpu': not os.environ.get('HEAD_RETRAIN_CPU'), 'enable_internet': False,
        **({} if os.environ.get('HEAD_RETRAIN_CPU') else {'machine_shape': 'NvidiaTeslaT4'}),
        'dataset_sources': [f'yodharinmonplub/{spec["weights"]}', 'yodharinmonplub/rsna-knee-gemini-relabels'],
        'competition_sources': ['rsna-knee-abnormality-detection'],
        'kernel_sources': ([] if spec['extra_kernels'] else [f'yodharinmonplub/{spec["cache"].format(i)}' for i in range(4)])
                          + spec['extra_kernels'],
        'model_sources': []}
(out / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2))
if os.environ.get('HEAD_RETRAIN_DRY'):
    print('dry run: built', out, json.dumps(meta['kernel_sources']))
else:
    # HEAD_RETRAIN_CPU=1: CPU kernel (only valid when the features are reused, i.e. with a TAG).
    assert not os.environ.get('HEAD_RETRAIN_CPU') or spec['extra_kernels'], 'CPU mode needs cached features'
    print(api.kernels_push(str(out), acc=None if os.environ.get('HEAD_RETRAIN_CPU') else 'NvidiaTeslaT4'))
