"""Gold (58-case) test of horizontal-flip TTA for the fine-tuned maxspan CoAtNet (model16).

Windows whose centre frame lies in the coronal/axial slots (frame >= 32 of the 64-frame layout) are
flipped left-right (equivalent to the contralateral knee, which the model sees in training); a
second variant flips every window. Derived from scripts/validate_dense_inference.py.
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
body = r'''
def centers_of(mask):
    valid = np.flatnonzero(mask)
    if len(valid) < 3:
        valid = np.arange(3)
    lo, hi = int(valid.min()), int(valid.max())
    return list(range(lo+1, hi)) or [max(1, min((lo+hi)//2, 62))]


def main():
    import hashlib
    from sklearn.metrics import roc_auc_score
    started = time.time()
    device = torch.device('cuda')
    inputs = Path('/kaggle/input')
    cache = [p.parent for p in inputs.rglob('cache_manifest.json')]
    assert len(cache) == 1, cache
    cache = cache[0]
    manifest = json.loads((cache/'cache_manifest.json').read_text())
    assert manifest['geometry']['span'] == [.02, .98] and manifest['geometry']['size'] == 336
    comp = next(p for p in [inputs/'competitions/rsna-knee-abnormality-detection', inputs/'rsna-knee-abnormality-detection'] if (p/'train.csv').exists())
    gold = pd.read_csv(comp/'train.csv').dropna(subset=LABELS).set_index(UID)
    files = list(inputs.rglob('model16_best.pt'))
    assert len(files) == 1, files
    ck = torch.load(files[0], map_location='cpu', weights_only=True)
    model = Classifier(ck['arch']); model.load_state_dict(ck['model'], strict=True)
    model = model.to(device).eval(); size = int(ck['res']); del ck
    out = {k: {} for k in ['plain', 'flip_cor_ax', 'flip_all']}
    for shard in manifest['shards']:
        p = cache/shard['file']
        with np.load(p, allow_pickle=False) as a:
            ids = a['ids'].tolist(); volumes = a['volumes']; masks = a['masks']
        for i, uid in enumerate(ids):
            if uid not in gold.index:
                continue
            images = windows(volumes[i], masks[i], size)
            centers = np.asarray(centers_of(masks[i]))
            corax = torch.from_numpy(centers >= 32)
            flipped_ca = images.clone(); flipped_ca[corax] = flipped_ca[corax].flip(-1)
            out['plain'][uid] = predict(model, images, device)
            out['flip_cor_ax'][uid] = predict(model, flipped_ca, device)
            out['flip_all'][uid] = predict(model, images.flip(-1), device)
    ids = sorted(out['plain'])
    assert len(ids) == 58
    y = gold.loc[ids, LABELS].to_numpy()
    arr = {k: np.stack([v[u] for u in ids]) for k, v in out.items()}
    arr['avg_plain_corax'] = (arr['plain'] + arr['flip_cor_ax']) / 2
    arr['avg_plain_all'] = (arr['plain'] + arr['flip_all']) / 2
    report = {}
    for k, v in arr.items():
        report[k] = float(np.mean([roc_auc_score(y[:, j], v[:, j]) for j in range(12)]))
        pd.DataFrame(v, index=pd.Index(ids, name=UID), columns=LABELS).to_csv(f'/kaggle/working/{k}.csv')
    report['elapsed_seconds'] = time.time() - started
    Path('/kaggle/working/flip_tta.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


main()
'''
code = head + body
compile(code, 'flip', 'exec')
api = KaggleApi(); api.authenticate()
out = ROOT / 'artifacts/flip_tta_gold_notebook'
out.mkdir(parents=True, exist_ok=True)
(out / 'inference.py').write_text(code)
meta = {'id': 'yodharinmonplub/rsna-knee-flip-tta-gold', 'title': 'RSNA Knee Flip TTA Gold',
        'code_file': 'inference.py', 'language': 'python', 'kernel_type': 'script', 'is_private': True,
        'enable_gpu': True, 'machine_shape': 'NvidiaTeslaT4', 'enable_internet': False,
        'dataset_sources': ['yodharinmonplub/rsna-knee-model16-weights'],
        'competition_sources': ['rsna-knee-abnormality-detection'],
        'kernel_sources': ['yodharinmonplub/rsna-knee-dense-cache-pilot'], 'model_sources': []}
(out / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2))
print(api.kernels_push(str(out), acc='NvidiaTeslaT4'))
