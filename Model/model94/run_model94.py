"""model94: model93 (model76 + model75 as a fifth Raptor-side view, time-boxed and fail-safe) plus model78's final
FLEET LEG (fleet4 = model72, model71 x5, model73, model75; rank blend at w 0.30) - do the two model75 uses stack?

model78 (model76 + fleet4 leg) scored 0.946 public; model93 is scoring. Both edits are independent: the fifth view
changes Speedy's Raptor side mid-pipeline, the leg rewrites the finished submission. Runtime: Speedy ~7.1 h + fifth view
~10-15 min (allowance 1500 s) + fleet leg ~25 min, all under the leg's 9 h process-age guard with 30 min to spare; both
additions fall back to the result without them on any error or time shortfall.

Usage: python scripts/run_model94.py [--push]
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_model80 as leg
import run_model93 as m93

NAME = 'model94'
TITLE = 'RSNA Knee Model94 CoAt75 View Plus Fleet4'   # <= 50 characters


def build():
    folder93, meta93 = m93.build()
    nb = json.loads((folder93 / 'notebook.ipynb').read_text(encoding='utf-8'))
    last = ''.join(nb['cells'][-1]['source'])
    assert "print(f'COMPLETE: {_final}" in last, 'the last Speedy cell is no longer the submission writer'
    cell = leg.FLEET_LEG.lstrip('\n')
    assert cell.count('_M80_W = 0.30\n') == 1
    member73 = "              ('model73', ['model73_best.pt'])]"
    assert cell.count(member73) == 1
    cell = cell.replace(member73, leg.MODEL75)
    nb['cells'].append({'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [],
                        'source': cell.splitlines(keepends=True)})
    meta = dict(meta93)
    meta.update({'id': f'yodharinmonplub/{TITLE.lower().replace(" ", "-")}', 'title': TITLE})
    assert 'yodharinmonplub/rsna-knee-fleet-weights' in meta['dataset_sources']   # model93 already mounts it
    folder = m93.base.ROOT / f'artifacts/{NAME}_notebook'
    model_dir = m93.base.ROOT.parent / 'Model' / NAME
    for target in (folder, model_dir):
        target.mkdir(parents=True, exist_ok=True)
        (target / 'notebook.ipynb').write_text(json.dumps(nb, indent=1), encoding='utf-8')
        (target / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2))
    for script in ('run_model76.py', 'run_model77.py', 'run_model80.py', 'run_model93.py', 'run_model94.py'):
        shutil.copy2(Path(__file__).resolve().parent / script, model_dir / script)
    for c in nb['cells']:
        if c['cell_type'] == 'code':
            src = ''.join(c['source'])
            if not any(line.lstrip().startswith(('!', '%')) for line in src.splitlines()):
                compile(src, NAME, 'exec')
    return folder, meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--push', action='store_true')
    args = parser.parse_args()
    folder, meta = build()
    print('built', folder, '| kernel', meta['id'])
    if args.push:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi(); api.authenticate()
        result = api.kernels_push(str(folder), acc=meta.get('machine_shape', 'NvidiaTeslaT4'))
        (folder / 'run_response.txt').write_text(str(result))
        print(result)


if __name__ == '__main__':
    main()
