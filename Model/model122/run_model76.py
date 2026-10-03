"""model76: mattiaangeli's public "Bend the Knee to Speedy Raptors - The Original" (public LB 0.943) with ONE change:
its four Raptor CoAtNet views load this project's Gemini-fine-tuned, head-retrained checkpoints (model61's) instead
of the public dreaddevelopment weights.

Why: our edge over the old public base came entirely from those Raptor weights (Dinosaurs base 0.941 -> model43
Gemini-fine-tuned arms 0.942 -> model61 retrained heads 0.943). Speedy Raptors reached 0.943 by a different route
(two new CoAtNet readers, D4 depth-zone and Global96, blended into the Raptor side) and still ships the public Raptor
weights, so the two improvements should stack. Everything else - views, view weights, the 94-window evaluation, the
two-GPU schedule, the CoAt readers and the outer blend - is Speedy's, untouched.

Mapping (same as model61): maxspan-v5 and maxspan-v5-reverse -> model61_maxspan_best.pt, native384dense-v10 ->
model61_dense_best.pt, native384-v8 -> model61_v8_best.pt, from the private kernels
yodharinmonplub/rsna-knee-head-retrain-{maxspan,dense,v8}. Speedy resolves weights through a whitelist of the public
dataset folders (ASSET_ROOTS), so our three files get their own lookup. Its loader is strict (exact keys and shapes,
architecture equality) and logs a SHA-256 per loaded checkpoint, and weight lookup/loading sits outside any
try-block, so a wrong or missing file stops the run instead of silently falling back to public weights.

Usage: python scripts/run_model76.py [--push]
"""
import argparse
import copy
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / '.env')
os.environ.setdefault('KAGGLE_API_TOKEN', os.environ.get('KAGGLE_API_KEY_yodharinmonplub@actuarialtutor.org', os.environ.get('KAGGLE_API_KEY', '')))
os.environ['KAGGLE_CONFIG_DIR'] = str(ROOT / '.kaggle')

SOURCE = ROOT / 'artifacts/sources/public_notebooks/mattiaangeli__bend-the-knee-to-speedy-raptors-the-original'
NOTEBOOK = 'bend-the-knee-to-speedy-raptors-the-original.ipynb'
ANCHOR = "exec(compile(_KE_SRC, '<raptor>', 'exec'), _KE_NS)\n"
NAME = 'model76'
TITLE = 'RSNA Knee Model76 Speedy Raptors Our Heads'  # Kaggle caps titles at 50 characters
OUR_KERNELS = [f'yodharinmonplub/rsna-knee-head-retrain-{arm}' for arm in ('maxspan', 'dense', 'v8')]

PATCH = '''
# ---- model76: this project's Gemini-fine-tuned, head-retrained Raptor checkpoints (model61) ----
import glob as _m76_glob
_M76_ARM_FILE = {'maxspan-v5': 'model61_maxspan_best.pt', 'maxspan-v5-reverse': 'model61_maxspan_best.pt',
                 'native384dense-v10': 'model61_dense_best.pt', 'native384-v8': 'model61_v8_best.pt'}
_M76_SLUG = {'model61_maxspan_best.pt': 'rsna-knee-head-retrain-maxspan',
             'model61_dense_best.pt': 'rsna-knee-head-retrain-dense',
             'model61_v8_best.pt': 'rsna-knee-head-retrain-v8'}
assert sorted(a['name'] for a in _KE_NS['ARMS']) == sorted(_M76_ARM_FILE), [a['name'] for a in _KE_NS['ARMS']]
for _m76_arm in _KE_NS['ARMS']:
    _m76_arm['file'] = _M76_ARM_FILE[_m76_arm['name']]
_m76_public_find = _KE_NS['find_weight_file']

def _m76_find_weight_file(fname):
    if fname not in _M76_SLUG:
        return _m76_public_find(fname)
    slug = _M76_SLUG[fname]
    hits = sorted({p for pattern in (f'/kaggle/input/notebooks/*/{slug}/{fname}', f'/kaggle/input/{slug}/{fname}',
                                     f'/kaggle/input/*/{slug}/{fname}', f'/kaggle/input/*/*/{slug}/{fname}')
                   for p in _m76_glob.glob(pattern)})
    if len(hits) != 1:
        raise RuntimeError(f'model76: expected one {fname} under {slug}, found {hits}')
    return hits[0]

_KE_NS['find_weight_file'] = _m76_find_weight_file
print('[model76] Raptor arms:', [(a['name'], a['file'], a['w']) for a in _KE_NS['ARMS']], flush=True)
'''


def build():
    nb = json.loads((SOURCE / NOTEBOOK).read_text(encoding='utf-8'))
    meta = json.loads((SOURCE / 'kernel-metadata.json').read_text())
    hits = [c for c in nb['cells'] if c['cell_type'] == 'code' and ANCHOR in ''.join(c['source'])]
    assert len(hits) == 1, f'anchor found in {len(hits)} cells'
    text = ''.join(hits[0]['source'])
    assert text.count(ANCHOR) == 1
    hits[0]['source'] = text.replace(ANCHOR, ANCHOR + PATCH).splitlines(keepends=True)
    new_meta = copy.deepcopy(meta)
    new_meta.pop('id_no', None)
    new_meta.update({'id': f'yodharinmonplub/{TITLE.lower().replace(" ", "-")}', 'title': TITLE,
                     'code_file': 'notebook.ipynb', 'is_private': True})
    new_meta['dataset_sources'] = [s for s in meta['dataset_sources'] if s]
    new_meta['kernel_sources'] = list(meta.get('kernel_sources', [])) + OUR_KERNELS
    for folder in (ROOT / NAME, ROOT / f'artifacts/{NAME}_notebook'):
        folder.mkdir(parents=True, exist_ok=True)
        (folder / 'notebook.ipynb').write_text(json.dumps(nb, indent=1), encoding='utf-8')
        (folder / 'kernel-metadata.json').write_text(json.dumps(new_meta, indent=2))
    for cell in nb['cells']:          # compile check (IPython magics skipped)
        if cell['cell_type'] == 'code':
            src = ''.join(cell['source'])
            if not any(line.lstrip().startswith(('!', '%')) for line in src.splitlines()):
                compile(src, NAME, 'exec')
    return ROOT / f'artifacts/{NAME}_notebook', new_meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--push', action='store_true')
    args = parser.parse_args()
    folder, meta = build()
    print('built', folder, '| kernel', meta['id'])
    print('  datasets', len(meta['dataset_sources']), '| kernels', meta['kernel_sources'])
    if args.push:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi(); api.authenticate()
        result = api.kernels_push(str(folder), acc=meta.get('machine_shape', 'NvidiaTeslaT4'))
        (folder / 'run_response.txt').write_text(str(result))
        print(result)


if __name__ == '__main__':
    main()
