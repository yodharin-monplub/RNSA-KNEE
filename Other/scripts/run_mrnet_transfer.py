"""Free transfer check: how well does model61 (maxspan arm, trained only on competition data) read MRNet?

Runs on a Kaggle T4. Each MRNet exam (sagittal, coronal, axial uint8 stacks, 256 px) becomes 3-slice windows
per plane at the checkpoint's resolution; the frozen model predicts the 12 competition findings; these are scored
against MRNet's own labels: ACL tear vs our ACL, meniscal tear vs max(medial, lateral meniscus), abnormal vs the
max over all 12. High AUCs mean the two domains are compatible, so MRNet labels are usable supervision; it also
writes per-exam window features (fp16) so later head-level experiments are free.
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
PER_PLANE = 22


def exam_windows(planes, size):
    """3-slice windows inside each plane (never across planes), evenly subsampled to PER_PLANE slices per plane."""
    stacks = []
    for volume in planes:
        if len(volume) > PER_PLANE:
            volume = volume[np.linspace(0, len(volume) - 1, PER_PLANE).round().astype(int)]
        if len(volume) < 3:
            volume = np.concatenate([volume] * 3)[:3]
        stacks.append(np.stack([volume[i - 1:i + 2] for i in range(1, len(volume) - 1)]))
    slices = torch.from_numpy(np.concatenate(stacks).astype(np.float32) / 255)
    images = F.interpolate(slices, size=(size, size), mode='bilinear', align_corners=False)
    return (images - torch.tensor([.485, .456, .406]).view(1, 3, 1, 1)) / torch.tensor([.229, .224, .225]).view(1, 3, 1, 1)


def main():
    from sklearn.metrics import roc_auc_score
    started = time.time()
    device = torch.device('cuda')
    inputs = Path('/kaggle/input')
    weights = list(inputs.rglob('model61_maxspan_best.pt'))
    assert len(weights) == 1, weights
    ck = torch.load(weights[0], map_location='cpu', weights_only=True)
    model = Classifier(ck['arch']); model.load_state_dict(ck['model'], strict=True)
    model = model.to(device).eval(); size = int(ck['res'])
    labels = json.loads(next(inputs.rglob('labels.json')).read_text())
    shards = sorted(p for p in inputs.rglob('mrnet_*.npz'))
    assert len(shards) == 14, shards
    rows, feats = [], {}
    for shard in shards:
        with np.load(shard, allow_pickle=False) as archive:
            exams = sorted({k.split('_')[0] for k in archive.files})
            for exam in exams:
                images = exam_windows([archive[f'{exam}_{p}'] for p in ('sagittal', 'coronal', 'axial')], size)
                with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
                    encoded = torch.cat([model.backbone(chunk.to(device)) for chunk in images.split(12)])
                    prob = model.predict_features(encoded.float().unsqueeze(0)).float().sigmoid()[0].cpu().numpy()
                feats[exam] = encoded.float().cpu().numpy().astype(np.float16)
                rows.append({'exam': exam, **labels[exam], **{l: float(p) for l, p in zip(LABELS, prob)}})
        print(shard.name, len(rows), round(time.time() - started), flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv('/kaggle/working/mrnet_predictions.csv', index=False)
    np.savez('/kaggle/working/mrnet_window_features_fp16.npz', **feats)
    frame['men'] = frame[['Medial Meniscus', 'Lateral Meniscus']].max(axis=1)
    frame['any'] = frame[LABELS].max(axis=1)
    report = {'exams': len(frame)}
    for split in ('train', 'valid', 'all'):
        part = frame if split == 'all' else frame[frame.split == split]
        report[split] = {'acl_vs_ACL': roc_auc_score(part.acl, part['ACL']),
                         'meniscus_vs_max_meniscus': roc_auc_score(part.meniscus, part.men),
                         'abnormal_vs_max_any': roc_auc_score(part.abnormal, part['any']),
                         'n': len(part), 'acl_pos': int(part.acl.sum()), 'men_pos': int(part.meniscus.sum())}
    report['seconds'] = time.time() - started
    Path('/kaggle/working/mrnet_transfer.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


main()
'''

code = head + BODY
compile(code, 'mrnet_transfer', 'exec')
api = KaggleApi(); api.authenticate()
out = ROOT / 'artifacts/mrnet_transfer_notebook'
out.mkdir(parents=True, exist_ok=True)
(out / 'inference.py').write_text(code)
meta = {'id': 'yodharinmonplub/rsna-knee-mrnet-transfer', 'title': 'RSNA Knee MRNet Transfer',
        'code_file': 'inference.py', 'language': 'python', 'kernel_type': 'script', 'is_private': True,
        'enable_gpu': True, 'machine_shape': 'NvidiaTeslaT4', 'enable_internet': False,
        'dataset_sources': ['yodharinmonplub/rsna-knee-external-mrnet-cache'],
        'competition_sources': [], 'kernel_sources': ['yodharinmonplub/rsna-knee-head-retrain-maxspan'],
        'model_sources': []}
(out / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2))
print(api.kernels_push(str(out), acc='NvidiaTeslaT4'))
