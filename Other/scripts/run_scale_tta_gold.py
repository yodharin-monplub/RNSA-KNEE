"""Multi-scale test-time augmentation test for the best CoAtNet arm (free Kaggle GPU, 58 gold cases).

The arm normally evaluates 3-slice windows resized to 384px. This scores the same checkpoint at
336/384/448px and their averages, to see whether multi-scale TTA is worth its extra forward passes.
Derived from scripts/validate_dense_inference.py; gold is never trained on by model51.
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
def main():
    import hashlib
    from sklearn.metrics import roc_auc_score
    started = time.time()
    device = torch.device('cuda')
    inputs = Path('/kaggle/input')
    cache = next(p.parent for p in inputs.rglob('cache_manifest.json'))
    manifest = json.loads((cache / 'cache_manifest.json').read_text())
    assert manifest['geometry']['span'] == [.02, .98] and manifest['geometry']['size'] == 336
    comp = next(p.parent for p in inputs.rglob('train.csv') if (p.parent / 'test_series.csv').exists())
    gold = pd.read_csv(comp / 'train.csv').dropna(subset=LABELS).set_index(UID)
    files = list(inputs.rglob('model51_best.pt'))
    assert len(files) == 1, files
    ck = torch.load(files[0], map_location='cpu', weights_only=True)
    model = Classifier(ck['arch']); model.load_state_dict(ck['model'], strict=True)
    model = model.to(device).eval()
    print('checkpoint res', ck['res'], flush=True)
    scales = [336, 384, 448]
    out = {scale: {} for scale in scales}
    for shard in manifest['shards']:
        path = cache / shard['file']
        with path.open('rb') as stream:
            assert hashlib.file_digest(stream, 'sha256').hexdigest() == shard['sha256']
        with np.load(path, allow_pickle=False) as archive:
            ids, volumes, masks = archive['ids'].tolist(), archive['volumes'], archive['masks']
        for i, uid in enumerate(ids):
            if uid not in gold.index:
                continue
            for scale in scales:
                out[scale][uid] = predict(model, windows(volumes[i], masks[i], scale), device)
        print('cases', len(out[384]), round(time.time() - started), flush=True)
    ids = sorted(out[384])
    assert len(ids) == 58
    y = gold.loc[ids, LABELS].to_numpy()
    arrays = {f'scale_{scale}': np.stack([out[scale][u] for u in ids]) for scale in scales}
    arrays['mean_384_448'] = (arrays['scale_384'] + arrays['scale_448']) / 2
    arrays['mean_336_384_448'] = (arrays['scale_336'] + arrays['scale_384'] + arrays['scale_448']) / 3
    arrays['mean_336_384'] = (arrays['scale_336'] + arrays['scale_384']) / 2
    report = {k: float(np.mean([roc_auc_score(y[:, j], v[:, j]) for j in range(12)]))
              for k, v in arrays.items()}
    report['elapsed_seconds'] = time.time() - started
    for name, values in arrays.items():
        pd.DataFrame(values, index=pd.Index(ids, name=UID), columns=LABELS).to_csv(f'/kaggle/working/{name}.csv')
    Path('/kaggle/working/scale_tta.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


main()
'''

code = head + BODY
compile(code, 'scale_tta', 'exec')
api = KaggleApi(); api.authenticate()
out = ROOT / 'artifacts/scale_tta_notebook'
out.mkdir(parents=True, exist_ok=True)
(out / 'inference.py').write_text(code)
meta = {'id': 'yodharinmonplub/rsna-knee-scale-tta-gold', 'title': 'RSNA Knee Scale TTA Gold',
        'code_file': 'inference.py', 'language': 'python', 'kernel_type': 'script', 'is_private': True,
        'enable_gpu': True, 'machine_shape': 'NvidiaTeslaT4', 'enable_internet': False,
        'dataset_sources': ['yodharinmonplub/rsna-knee-model51-weights'],
        'competition_sources': ['rsna-knee-abnormality-detection'],
        'kernel_sources': ['yodharinmonplub/rsna-knee-dense-cache-pilot'], 'model_sources': []}
(out / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2))
print(api.kernels_push(str(out), acc='NvidiaTeslaT4'))
