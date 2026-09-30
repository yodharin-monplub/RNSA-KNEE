"""Per-group public-LB readout (diagnostic submission, forum 744294): a copy of a scored submission notebook whose last
cell keeps only one group of 4 findings and sets the other 8 columns to 0.5. A constant column scores AUC 0.5 and the
metric is the mean of the 12 per-finding AUCs, so the public score reveals the group's mean AUC on the hidden set:

    group AUC = (12 * LB - 4) / 4        (+-0.0015 from the 3-decimal rounding)

Groups (the forum post's split): G1 ACL, MCL, Medial Meniscus, Lateral Meniscus | G2 Medial OA, Synovitis, Baker's,
Fracture | G3 Lateral OA, PF OA, Effusion, Contusion. Reading out two submissions (e.g. model103 with the CoAtNet leg and
model76 without) group by group shows where the leg helps -> per-group leg weights.

The readout cell is fail-safe: if it fails, the full submission stays (its "group AUC" then comes out > 1, i.e. obvious).

Usage: python scripts/make_group_readout.py --source model103 --group 1 --name model114
"""
import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GROUPS = {1: ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus'],
          2: ['Medial OA', 'Synovitis', "Baker's", 'Fracture'],
          3: ['Lateral OA', 'PF OA', 'Effusion', 'Contusion']}
CELL = '''# GROUP READOUT (diagnostic submission): keep only these findings, set the other 8 columns to 0.5 (a constant column
# scores AUC 0.5), so the public LB gives this group's mean AUC = (12 * LB - 4) / 4. Fail-safe: on any error the full
# submission stays.
try:
    import os as _os
    import pandas as _pd
    from pathlib import Path as _Path
    _KEEP = __KEEP__
    _sub_path = _Path('/kaggle/working/submission.csv')
    _sub = _pd.read_csv(_sub_path)
    _labels = [c for c in _sub.columns if c != 'StudyInstanceUID']
    assert len(_labels) == 12 and set(_KEEP) <= set(_labels), _labels
    for _c in _labels:
        if _c not in _KEEP:
            _sub[_c] = 0.5
    _tmp = _sub_path.with_suffix('.csv.tmp')
    _sub.to_csv(_tmp, index=False)
    _os.replace(_tmp, _sub_path)
    print('GROUP READOUT: kept', _KEEP, '| rows', len(_sub), flush=True)
except Exception as _exc:
    print('GROUP READOUT FAILED (full submission kept):', type(_exc).__name__, _exc, flush=True)
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True, help='scored model folder whose notebook is copied, e.g. model103')
    parser.add_argument('--group', type=int, choices=sorted(GROUPS), required=True)
    parser.add_argument('--name', required=True, help='new model folder, e.g. model114')
    args = parser.parse_args()
    src = ROOT.parent / 'Model' / args.source
    nb = json.loads((src / 'notebook.ipynb').read_text(encoding='utf-8'))
    meta = json.loads((src / 'kernel-metadata.json').read_text())
    assert 'GROUP READOUT' not in json.dumps(nb), 'source is already a readout'
    cell = CELL.replace('__KEEP__', repr(GROUPS[args.group]))
    compile(cell, 'readout', 'exec')
    nb['cells'].append({'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [],
                        'source': cell.splitlines(keepends=True)})
    num = args.name.replace('model', '')
    meta['id'] = f'yodharinmonplub/rsna-knee-model{num}-readout-g{args.group}-{args.source}'
    meta['title'] = f'RSNA Knee Model{num} Readout G{args.group} {args.source.capitalize()}'
    assert len(meta['title']) <= 50, meta['title']
    folder = ROOT / f'artifacts/{args.name}_notebook'
    model_dir = ROOT.parent / 'Model' / args.name
    for target in (folder, model_dir):
        target.mkdir(parents=True, exist_ok=True)
        (target / 'notebook.ipynb').write_text(json.dumps(nb, indent=1), encoding='utf-8')
        (target / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2))
    shutil.copy2(Path(__file__), model_dir / Path(__file__).name)
    print('built', folder, '|', meta['id'], '| keeps', GROUPS[args.group])


if __name__ == '__main__':
    main()
