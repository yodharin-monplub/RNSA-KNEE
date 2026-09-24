"""Build the DINOv2-member training cache on Kaggle CPU, using the public notebook's own preprocessing.

Cells 1-7 of mattiaangeli/bend-the-knee-to-the-dinosaurs are embedded verbatim except that the GPU
probe and the memory-based cache sizing are replaced (CPU job, fixed one group of three slices), so
the pixels match what the released DINOv2 members were fitted on. Writes uint8 shards
[n, 6 slots, 3 slices, 336, 336] plus the slot mask, for one quarter of the training studies.

Usage: python scripts/run_dino_cache.py --part 0   (--part pilot for a 64-study trial)
"""
import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / '.env')
os.environ.setdefault('KAGGLE_API_TOKEN', os.environ['KAGGLE_API_KEY'])
os.environ['KAGGLE_CONFIG_DIR'] = str(ROOT / '.kaggle')
from kaggle.api.kaggle_api_extended import KaggleApi

SOURCE = ROOT / 'artifacts/sources/public_notebooks/mattiaangeli__bend-the-knee-to-the-dinosaurs'
NOTEBOOK = SOURCE / 'bend-the-knee-to-the-dinosaurs.ipynb'


def preprocessing_source():
    cells = [''.join(c['source']) for c in json.loads(NOTEBOOK.read_text(encoding='utf-8'))['cells']]
    first = cells[1]
    start, end = first.index('def _cuda_execution_probe(index):'), first.index('T0 = time.time()')
    first = first[:start] + "DEVS = []\n_STARTUP_GPU_IMPORT_S = 0.0\n" + first[end:]
    second = cells[2]
    second = second.replace("N_GROUP = plan_cache(len(pd.read_csv(ROOT / 'train.csv')), len(pd.read_csv(ROOT / 'test.csv')))",
                            'N_GROUP = N_GROUP_MAX')
    assert 'plan_cache(len' not in second and 'cuda' not in first
    return '\n'.join([first, second] + cells[3:8])


RAD_GEOMETRY = '''
# RadImageNet family geometry (notebook section _rad_*): three fat-suppressed planes, eight acquired
# slices each, full frame at 224px, legacy ordering/laterality/fill.
SLOTS = [('SAG_FS', 'Sagittal', None, True), ('COR_FS', 'Coronal', None, True), ('AX_FS', 'Axial', None, True)]
N_SLOT = len(SLOTS)
CACHE_SLICES = 8
IMG = CACHE_IMG = 224
CROP_MM = 10000.0
SLICE_BAND = (0.2, 0.8)
RULES = dict(RULES_LEGACY)
SLOT_SCHEME = 'rad'
GROUP = CACHE_SLICES
N_GROUP = 1
'''

BODY = '''
PART = __PART__
PARTS = 4


def main():
    import hashlib
    started = time.time()
    train = pd.read_csv(ROOT / 'train.csv')
    assert len(train) == 4407 and train.StudyInstanceUID.is_unique
    labels = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA',
              'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
    if PART == 'pilot':
        gold = train.dropna(subset=labels)
        ids = gold.StudyInstanceUID.tolist() + train[~train.StudyInstanceUID.isin(gold.StudyInstanceUID)].StudyInstanceUID.head(6).tolist()
    else:
        ids = np.array_split(train.StudyInstanceUID.to_numpy(), PARTS)[int(PART)].tolist()
    wanted = set(ids)
    series = pd.read_csv(ROOT / 'train_series.csv')
    plane_map = dict(zip(series.SeriesInstanceUID, series.Anatomical_Plane))
    headers = annotate(walk('train_series'))
    headers = headers[headers.StudyInstanceUID.isin(wanted)].reset_index(drop=True)
    log(f'part {PART}: {len(wanted)} studies, {len(headers)} series')
    slot_map = pick_slots(headers, plane_map)
    lat_map = lat_of(headers, 'train ')
    studies, cache, mask = build_cache(slot_map, plane_map, lat_map, f'train {PART}')
    assert cache.shape[1:] == (N_SLOT, CACHE_SLICES, IMG, IMG) and cache.dtype == np.uint8
    print('geometry', N_SLOT, CACHE_SLICES, IMG, CROP_MM, SLOT_SCHEME, flush=True)
    out = Path('/kaggle/working')
    manifest = {'kind': SLOT_SCHEME + '_part', 'part': PART, 'parts': PARTS, 'studies': len(studies),
                'geometry': {'slots': [s[0] for s in SLOTS], 'slot_scheme': SLOT_SCHEME, 'group': GROUP,
                             'n_group': N_GROUP, 'size': IMG, 'crop_mm': CROP_MM, 'band': list(SLICE_BAND),
                             'rules': RULES},
                'decode_failures': len(DECODE_FAILED), 'shards': []}
    for start in range(0, len(studies), 64):
        block = studies[start:start + 64]
        name = f'{SLOT_SCHEME}_{PART}_{start // 64:03d}.npz'
        path = out / name
        np.savez_compressed(path, ids=np.asarray(block), volumes=cache[start:start + 64],
                            masks=mask[start:start + 64])
        with np.load(path, allow_pickle=False) as check:
            assert check['ids'].tolist() == block
            assert np.array_equal(check['volumes'], cache[start:start + 64])
            assert np.array_equal(check['masks'], mask[start:start + 64])
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        manifest['shards'].append({'file': name, 'ids': block, 'bytes': path.stat().st_size, 'sha256': digest})
        log(f'wrote {name} ({start + len(block)}/{len(studies)})')
    manifest['compressed_bytes'] = sum(s['bytes'] for s in manifest['shards'])
    manifest['training_metadata_sha256'] = hashlib.sha256((ROOT / 'train.csv').read_bytes()).hexdigest()
    manifest['elapsed_seconds'] = time.time() - started
    (out / 'cache_manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: v for k, v in manifest.items() if k != 'shards'}), flush=True)


main()
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--part', required=True)
    parser.add_argument('--geometry', choices=['dino', 'rad'], default='dino')
    args = parser.parse_args()
    code = preprocessing_source() + (RAD_GEOMETRY if args.geometry == 'rad' else '') + BODY.replace('__PART__', repr(args.part))
    compile(code, 'dino_cache', 'exec')
    api = KaggleApi(); api.authenticate()
    tag = 'dino' if args.geometry == 'dino' else 'rad'
    out = ROOT / f'artifacts/{tag}_cache_{args.part}_notebook'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'inference.py').write_text(code)
    slug = f'rsna-knee-{tag}-cache-{args.part}'
    meta = {'id': f'yodharinmonplub/{slug}', 'title': f'RSNA Knee {tag.capitalize()} Cache {args.part}',
            'code_file': 'inference.py', 'language': 'python', 'kernel_type': 'script', 'is_private': True,
            'enable_gpu': False, 'enable_internet': False, 'dataset_sources': [],
            'competition_sources': ['rsna-knee-abnormality-detection'], 'kernel_sources': [], 'model_sources': ['metaresearch/dinov2/PyTorch/small/1']}
    (out / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2))
    print(api.kernels_push(str(out)))


if __name__ == '__main__':
    main()
