"""model76 (Speedy Raptors + our model61 Raptor weights, public 0.945) with chosen Raptor arms swapped for NEW
checkpoints from a private dataset - the deployment path for new Raptor weights (e.g. model79's full fine-tune).

Arms: maxspan -> maxspan-v5 and maxspan-v5-reverse (one checkpoint, forward + reversed windows, 0.60 + 0.10 of the
Raptor side), dense -> native384dense-v10 (0.10), v8 -> native384-v8 (0.20). Everything else is model76.
The strict loader, SHA-256 logging and the "[model76] Raptor arms:" line are model76's, so the same tick check applies.

Usage: python scripts/run_raptor_swap.py --name model87 --title "RSNA Knee Model87 Speedy Model79 Maxspan" \
         --swap maxspan=model79_best.pt --dataset yodharinmonplub/rsna-knee-model79-weights [--push]
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_model76 as base

ARMS = {'maxspan': ("'maxspan-v5': 'model61_maxspan_best.pt', 'maxspan-v5-reverse': 'model61_maxspan_best.pt'",
                    "'maxspan-v5': '{f}', 'maxspan-v5-reverse': '{f}'"),
        'dense': ("'native384dense-v10': 'model61_dense_best.pt'", "'native384dense-v10': '{f}'"),
        'v8': ("'native384-v8': 'model61_v8_best.pt'", "'native384-v8': '{f}'")}
SLUG_ANCHOR = "_M76_SLUG = {'model61_maxspan_best.pt': 'rsna-knee-head-retrain-maxspan',"


def build(name, title, swaps, dataset):
    folder76, meta76 = base.build()
    nb = json.loads((folder76 / 'notebook.ipynb').read_text(encoding='utf-8'))
    slug = dataset.split('/', 1)[1]
    hits = [c for c in nb['cells'] if c['cell_type'] == 'code' and SLUG_ANCHOR in ''.join(c['source'])]
    assert len(hits) == 1
    text = ''.join(hits[0]['source'])
    new_files = set()
    for arm, fname in swaps.items():
        old, new = ARMS[arm]
        assert text.count(old) == 1, arm
        text = text.replace(old, new.format(f=fname))
        new_files.add(fname)
    extra = ''.join(f"'{f}': '{slug}', " for f in sorted(new_files))
    text = text.replace(SLUG_ANCHOR, "_M76_SLUG = {" + extra + SLUG_ANCHOR[len("_M76_SLUG = {"):])
    hits[0]['source'] = text.splitlines(keepends=True)
    meta = dict(meta76)
    meta.update({'id': f'yodharinmonplub/{title.lower().replace(" ", "-")}', 'title': title})
    meta['dataset_sources'] = list(meta76['dataset_sources']) + [dataset]
    folder = base.ROOT / f'artifacts/{name}_notebook'
    model_dir = base.ROOT.parent / 'Model' / name
    for target in (folder, model_dir):
        target.mkdir(parents=True, exist_ok=True)
        (target / 'notebook.ipynb').write_text(json.dumps(nb, indent=1), encoding='utf-8')
        (target / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2))
    for script in ('run_model76.py', 'run_raptor_swap.py'):
        shutil.copy2(Path(__file__).resolve().parent / script, model_dir / script)
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            src = ''.join(cell['source'])
            if not any(line.lstrip().startswith(('!', '%')) for line in src.splitlines()):
                compile(src, name, 'exec')
    return folder, meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=True)
    parser.add_argument('--title', required=True)
    parser.add_argument('--swap', action='append', required=True, help='arm=file, arm in maxspan/dense/v8')
    parser.add_argument('--dataset', required=True, help='owner/slug of the private dataset holding the new files')
    parser.add_argument('--push', action='store_true')
    args = parser.parse_args()
    assert len(args.title) <= 50, 'Kaggle caps titles at 50 characters'
    swaps = dict(s.split('=', 1) for s in args.swap)
    assert set(swaps) <= set(ARMS), swaps
    folder, meta = build(args.name, args.title, swaps, args.dataset)
    print('built', folder, '| kernel', meta['id'], '| swaps', swaps)
    if args.push:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi(); api.authenticate()
        result = api.kernels_push(str(folder), acc=meta.get('machine_shape', 'NvidiaTeslaT4'))
        (folder / 'run_response.txt').write_text(str(result))
        print(result)


if __name__ == '__main__':
    main()
