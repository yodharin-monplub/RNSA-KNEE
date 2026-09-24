"""Train RadImageNet heads on this project's labels (Kaggle GPU job).

The RadImageNet ResNet50 encoder is frozen in the public ensemble, so features are extracted once for
all 4,407 studies and only the small attention heads are trained. Head architecture and the bundle
contract are copied from the public notebook (mattiaangeli/bend-the-knee-to-the-dinosaurs); folds are
split by report group, gold58 is never trained on, so the reported gold AUC is honest for these heads.
Two bundles with different seeds are produced because the notebook loads two distinct head files.
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

NOTEBOOK = ROOT / ('artifacts/sources/public_notebooks/mattiaangeli__bend-the-knee-to-the-dinosaurs/'
                   'bend-the-knee-to-the-dinosaurs.ipynb')


def head_source():
    """Copy the _RadHead class and the encoder wrapper verbatim from the notebook."""
    cells = [''.join(c['source']) for c in json.loads(NOTEBOOK.read_text(encoding='utf-8'))['cells']]
    cell = next(c for c in cells if 'class _RadHead' in c)
    start = cell.index('class _RadEncoder')
    end = cell.index('def _rad_load_public_heads')
    block = cell[start:end]
    assert 'class _RadHead' in block and 'def forward' in block
    return block


BODY = '''
import hashlib, json, time
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch import nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

_rad_nn = nn
_rad_torch = torch
from torchvision.models import resnet50 as _rad_resnet50

_RAD_LABELS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA',
               'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
_RAD_TOKEN_DIM, _RAD_HEAD_DIM = 2048, 512
_RAD_ENCODER_SHA256 = '08629f7e7bd3e29b8ee9522ca3f65ce4d010a7ddf74f0ea3c7e3f3d0bbab0734'
N_SLOT, CACHE_SLICES = 3, 8
UID = 'StudyInstanceUID'
EPOCHS = 12
BATCH = 64

__HEAD_SOURCE__


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def load_cache():
    inputs = Path('/kaggle/input')
    manifests = sorted(inputs.rglob('cache_manifest.json'))
    assert len(manifests) == 4, manifests
    ids, feats, masks = [], [], []
    for path in manifests:
        manifest = json.loads(path.read_text())
        assert manifest['kind'] == 'rad_part' and manifest['geometry']['size'] == 224
        for shard in manifest['shards']:
            shard_path = path.parent / shard['file']
            assert sha256(shard_path) == shard['sha256']
            with np.load(shard_path, allow_pickle=False) as archive:
                ids.extend(archive['ids'].tolist())
                feats.append(archive['volumes'])
                masks.append(archive['masks'])
    return ids, feats, masks


def main():
    started = time.time()
    device = torch.device('cuda')
    inputs = Path('/kaggle/input')
    encoder_paths = [p for p in inputs.rglob('ResNet50.pt')]
    assert len(encoder_paths) == 1, encoder_paths
    assert sha256(encoder_paths[0]) == _RAD_ENCODER_SHA256
    encoder = _RadEncoder()
    encoder.load_state_dict(torch.load(encoder_paths[0], map_location='cpu', weights_only=True), strict=True)
    assert sum(p.numel() for p in encoder.parameters()) == 23_508_032
    encoder.eval().to(device).requires_grad_(False)

    ids, volume_blocks, mask_blocks = load_cache()
    print('studies', len(ids), 'elapsed', round(time.time() - started), flush=True)
    features = np.zeros((len(ids), N_SLOT * CACHE_SLICES, _RAD_TOKEN_DIM), np.float16)
    token_mask = np.zeros((len(ids), N_SLOT * CACHE_SLICES), np.float32)
    offset = 0
    with torch.inference_mode():
        for volumes, masks in zip(volume_blocks, mask_blocks):
            n = len(volumes)
            tokens = np.repeat(masks[:, :, None], CACHE_SLICES, axis=2).reshape(n, -1)
            token_mask[offset:offset + n] = tokens
            flat = volumes.reshape(-1, 224, 224)
            valid = np.flatnonzero(tokens.reshape(-1) > 0)
            block = np.zeros((n * N_SLOT * CACHE_SLICES, _RAD_TOKEN_DIM), np.float16)
            for start in range(0, len(valid), 96):
                index = valid[start:start + 96]
                image = torch.from_numpy(flat[index]).to(device).float().div_(127.5).sub_(1.0)
                image = image.unsqueeze(1).expand(-1, 3, -1, -1).contiguous()
                with torch.autocast('cuda'):
                    out = encoder(image)
                values = out.float().cpu().numpy()
                assert np.isfinite(values).all()
                block[index] = values.astype(np.float16)
            features[offset:offset + n] = block.reshape(n, N_SLOT * CACHE_SLICES, _RAD_TOKEN_DIM)
            offset += n
            print('encoded', offset, round(time.time() - started), flush=True)
    assert offset == len(ids)
    del volume_blocks, mask_blocks

    labels = pd.read_csv(next(inputs.rglob('targets_g38f_low.csv'))).set_index(UID)
    partition = pd.read_csv(next(inputs.rglob('partitions.csv'))).set_index(UID)
    comp = next(p.parent for p in inputs.rglob('train.csv') if (p.parent / 'test_series.csv').exists())
    gold_table = pd.read_csv(comp / 'train.csv').set_index(UID)
    position = {uid: i for i, uid in enumerate(ids)}
    gold_ids = [u for u in partition.index[partition.split == 'gold'] if u in position]
    pool = [u for u in partition.index if u not in set(gold_ids) and u in position]
    assert len(gold_ids) == 58 and len(pool) > 4000
    gold_y = gold_table.loc[gold_ids, _RAD_LABELS].to_numpy()
    groups = partition.loc[pool, 'report_hash'].astype(str)
    fold_of = {uid: int(hashlib.md5(groups[uid].encode()).hexdigest(), 16) % 5 for uid in pool}
    y = labels.loc[pool, _RAD_LABELS].to_numpy(np.float32)
    index = np.asarray([position[u] for u in pool])
    gold_index = np.asarray([position[u] for u in gold_ids])

    def batches(rows, size, shuffle, rng=None):
        # rows are positions into pool; features are addressed by the absolute cache index.
        order = rng.permutation(rows) if shuffle else rows
        for start in range(0, len(order), size):
            chunk = order[start:start + size]
            absolute = index[chunk]
            yield (torch.from_numpy(features[absolute]).to(device),
                   torch.from_numpy(token_mask[absolute]).to(device), chunk)

    def absolute_batches(absolute, size):
        for start in range(0, len(absolute), size):
            rows = absolute[start:start + size]
            yield (torch.from_numpy(features[rows]).to(device),
                   torch.from_numpy(token_mask[rows]).to(device))

    @torch.inference_mode()
    def infer(head, absolute):
        head.eval()
        out = []
        for image, mask in absolute_batches(absolute, 128):
            with torch.autocast('cuda'):
                out.append(torch.sigmoid(head(image, mask)).float().cpu().numpy())
        return np.concatenate(out)

    report = {'bundles': []}
    for bundle_index, seed in enumerate([20260930, 20261001]):
        folds, fold_scores = [], []
        oof = np.zeros((len(pool), len(_RAD_LABELS)), np.float32)
        for fold in range(5):
            torch.manual_seed(seed + fold); np.random.seed(seed + fold)
            rng = np.random.default_rng(seed + fold)
            train_rows = np.asarray([i for i, u in enumerate(pool) if fold_of[u] != fold])
            valid_rows = np.asarray([i for i, u in enumerate(pool) if fold_of[u] == fold])
            head = _RadHead().to(device)
            optimizer = torch.optim.AdamW(head.parameters(), lr=3e-4, weight_decay=0.01)
            steps = EPOCHS * max(1, len(train_rows) // BATCH)
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=3e-4, total_steps=steps,
                                                            pct_start=0.1)
            done = 0
            for epoch in range(EPOCHS):
                head.train()
                for image, mask, chunk in batches(train_rows, BATCH, True, rng):
                    target = torch.from_numpy(y[chunk]).to(device)
                    confidence = 2 * (target - .5).abs()
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast('cuda'):
                        logits = head(image, mask).float()
                    loss = (F.binary_cross_entropy_with_logits(logits, target, reduction='none')
                            * confidence).sum() / confidence.sum().clamp(min=1)
                    assert torch.isfinite(loss)
                    loss.backward()
                    nn.utils.clip_grad_norm_(head.parameters(), 1.0)
                    optimizer.step()
                    done += 1
                    if done < steps:
                        scheduler.step()
                    else:
                        break
            predictions = infer(head, index[valid_rows])
            oof[valid_rows] = predictions
            valid_y = y[valid_rows]
            usable = [j for j in range(12) if len(set((valid_y[:, j] > .5).astype(int))) == 2]
            fold_scores.append(float(np.mean([roc_auc_score(valid_y[:, j] > .5, predictions[:, j])
                                              for j in usable])))
            folds.append({'fold': fold, 'state_dict': {k: v.detach().cpu() for k, v in head.state_dict().items()}})
            print(json.dumps({'bundle': bundle_index, 'fold': fold, 'val_auc_vs_labels': fold_scores[-1]}), flush=True)
            del head
            torch.cuda.empty_cache()
        payload = {'version': 'v52-radimagenet-resnet50-official-1', 'targets': _RAD_LABELS,
                   'encoder_sha256': _RAD_ENCODER_SHA256,
                   'encoder_source_commit': '0ce16f7375db4236e646829d1eca61cdb4282133',
                   'img': 224, 'slices_per_plane': 8, 'feature': 'global_average_pool', 'folds': folds}
        name = f'v52_radimagenet_heads_b{bundle_index}.pt'
        torch.save(payload, f'/kaggle/working/{name}')
        gold_heads = []
        for record in folds:
            head = _RadHead().to(device)
            head.load_state_dict(record['state_dict'], strict=True)
            gold_heads.append(infer(head, gold_index))
            del head
        gold_mean = np.mean(gold_heads, axis=0)
        gold_auc = float(np.mean([roc_auc_score(gold_y[:, j], gold_mean[:, j]) for j in range(12)]))
        report['bundles'].append({'index': bundle_index, 'seed': seed, 'file': name,
                                  'sha256': sha256(f'/kaggle/working/{name}'),
                                  'fold_val_auc_vs_labels': fold_scores, 'gold_auc_five_head_mean': gold_auc})
        print(json.dumps(report['bundles'][-1]), flush=True)
    report['elapsed_seconds'] = time.time() - started
    Path('/kaggle/working/rad_heads_report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


main()
'''


def main():
    code = BODY.replace('__HEAD_SOURCE__', head_source())
    compile(code, 'rad_heads', 'exec')
    api = KaggleApi(); api.authenticate()
    out = ROOT / 'artifacts/rad_heads_notebook'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'inference.py').write_text(code)
    meta = {'id': 'yodharinmonplub/rsna-knee-rad-heads', 'title': 'RSNA Knee Rad Heads',
            'code_file': 'inference.py', 'language': 'python', 'kernel_type': 'script', 'is_private': True,
            'enable_gpu': True, 'machine_shape': 'NvidiaTeslaT4', 'enable_internet': False,
            'dataset_sources': ['marwanmath/resnet-50-radimagenet-marwan',
                                'yodharinmonplub/rsna-knee-gemini-relabels'],
            'competition_sources': ['rsna-knee-abnormality-detection'],
            'kernel_sources': [f'yodharinmonplub/rsna-knee-rad-cache-{i}' for i in range(4)],
            'model_sources': []}
    (out / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2))
    print(api.kernels_push(str(out), acc='NvidiaTeslaT4'))


if __name__ == '__main__':
    main()
