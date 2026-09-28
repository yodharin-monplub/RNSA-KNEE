"""model76 (Speedy + our model61 Raptor weights) + a final-level FLEET LEG built from fleet leg v2 (per-member reader
geometry), so the fleet may mix 64-frame members (model72/71/73/75, 's64') with 128-frame members (model86, 's128').
Same leg mechanics as model78 (rank blend after Speedy's submission.csv, both GPUs, fail-safe, 9 h process-age guard).

Usage: python scripts/run_speedy_fleet_v2.py --name model99 --title "RSNA Knee Model99 Speedy Fleet With Model86" \
         --weight 0.30 --fleet "[('model72', ['model72_best.pt'], 's64'), ..., ('model86', ['model86_best.pt'], 's128')]" [--push]
"""
import argparse
import ast
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_model76 as base
from fleet_leg_v2 import FLEET_LEG_V2

FLEET_SRC = """_M80_FLEET = [('model72', ['model72_best.pt'], 's64'),
              ('model71', [f'model71_fold{k}_best.pt' for k in range(5)], 's64'),
              ('model73', ['model73_best.pt'], 's64')]"""


def build(name, title, weight, fleet):
    members = ast.literal_eval(fleet)
    assert all(g in ('s64', 's128') and all(f.endswith('.pt') for f in files) for _, files, g in members)
    folder76, meta76 = base.build()
    nb = json.loads((folder76 / 'notebook.ipynb').read_text(encoding='utf-8'))
    assert "print(f'COMPLETE: {_final}" in ''.join(nb['cells'][-1]['source'])
    leg = FLEET_LEG_V2.lstrip('\n')
    assert leg.count('_M80_W = 0.30\n') == 1 and leg.count(FLEET_SRC) == 1
    leg = leg.replace('_M80_W = 0.30\n', f'_M80_W = {weight:.2f}\n').replace(FLEET_SRC, '_M80_FLEET = ' + repr(members))
    compile(leg, 'leg', 'exec')
    nb['cells'].append({'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [],
                        'source': leg.splitlines(keepends=True)})
    meta = dict(meta76)
    meta.update({'id': f'yodharinmonplub/{title.lower().replace(" ", "-")}', 'title': title})
    meta['dataset_sources'] = list(meta76['dataset_sources']) + ['yodharinmonplub/rsna-knee-fleet-weights']
    folder = base.ROOT / f'artifacts/{name}_notebook'
    model_dir = base.ROOT.parent / 'Model' / name
    for target in (folder, model_dir):
        target.mkdir(parents=True, exist_ok=True)
        (target / 'notebook.ipynb').write_text(json.dumps(nb, indent=1), encoding='utf-8')
        (target / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2))
    for script in ('run_model76.py', 'run_speedy_fleet_v2.py', 'fleet_leg_v2.py'):
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
    parser.add_argument('--weight', type=float, default=0.30)
    parser.add_argument('--fleet', required=True)
    parser.add_argument('--push', action='store_true')
    args = parser.parse_args()
    assert len(args.title) <= 50, 'Kaggle caps titles at 50 characters'
    folder, meta = build(args.name, args.title, args.weight, args.fleet)
    print('built', folder, '| kernel', meta['id'])
    if args.push:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi(); api.authenticate()
        result = api.kernels_push(str(folder), acc=meta.get('machine_shape', 'NvidiaTeslaT4'))
        (folder / 'run_response.txt').write_text(str(result))
        print(result)


if __name__ == '__main__':
    main()
