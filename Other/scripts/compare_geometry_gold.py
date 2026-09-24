"""Compare Raptor geometry variants (scripts/run_geometry_gold.py outputs) on the 58 gold cases.

Base = model61 four-arm CoAtNet blend (scripts/blend_model61.py weights) with the tested arm's prediction
taken from the SAME geometry kernel at the notebook's current geometry, so decode path differences cancel.
Candidate = the same blend with that arm swapped to a variant geometry.
Pre-declared rule (2026-09-21, before any result): adopt a variant only if the blend gains >= +0.002, or the
arm gains >= +0.002 and the blend does not drop.
Usage: python scripts/compare_geometry_gold.py fine|maxspan
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

def rankdata(a):  # average ranks, 1-based; numpy only (scipy is blocked by this PC's app-control policy)
    a = np.asarray(a, float)
    order = a.argsort(kind='mergesort')
    ranks = np.empty(len(a), float)
    ranks[order] = np.arange(1, len(a) + 1)
    values = a[order]
    start = 0
    for i in range(1, len(a) + 1):
        if i == len(a) or values[i] != values[start]:
            if i - start > 1:
                ranks[order[start:i]] = (start + 1 + i) / 2
            start = i
    return ranks


ROOT = Path(__file__).resolve().parents[1]
L = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA',
     'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
UID = 'StudyInstanceUID'
W = {'maxspan': .70, 'dense': .15, 'v8': .30, 'fine': .15}
CURRENT = {'fine': 'cur_64_06_94', 'maxspan': 'cur_64_02_98'}
gold = pd.read_csv(ROOT.parent / 'Data/competition/train.csv').set_index(UID).dropna(subset=L)
y = gold[L].to_numpy(float)


def auc(truth, score):
    pos = truth > .5
    n_pos, n_neg = pos.sum(), (~pos).sum()
    return (rankdata(score)[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def macro(p, rows=slice(None)):
    return float(np.mean([auc(y[rows, j], p[rows, j]) for j in range(12)]))


def read(path):
    return pd.read_csv(path).set_index(UID).loc[gold.index, L].to_numpy(float)


arm = sys.argv[1]
if arm == 'finerev':
    # Base = model70 (fine arm forward view at 80 slices, weight 0.15). Candidate adds a reversed view at 0.05, fixed
    # before the result (the dense arm's forward:reverse ratio 0.10:0.05). Same rule as above.
    out = ROOT / 'artifacts/geometry_gold_finerev_notebook/output'
    heads = {a: read(ROOT / f'artifacts/head_retrain_{a}_notebook/output/gold_predictions_{a}.csv') for a in W}
    fwd, rev = read(out / 'raptor_80_02_98.csv'), read(out / 'raptor_80_02_98_rev.csv')
    blend_w = lambda parts, w: sum(w[a] * parts[a] for a in w) / sum(w.values())
    base = blend_w({**heads, 'fine': fwd}, W)
    both = (.15 * fwd + .05 * rev) / .20
    cand = blend_w({**heads, 'fine': both}, {**W, 'fine': .20})
    print(f'fine forward 80: {macro(fwd):.4f} | reverse 80: {macro(rev):.4f} | fwd+rev (3:1): {macro(both):.4f}')
    print(f'blend model70 {macro(base):.4f} -> with reverse view {macro(cand):.4f} ({macro(cand) - macro(base):+.4f})')
    rng = np.random.default_rng(20260922)
    diffs = []
    while len(diffs) < 3000:
        rows = rng.integers(0, len(y), len(y))
        if all(0 < y[rows, j].sum() < len(rows) for j in range(12)):
            diffs.append(macro(cand, rows) - macro(base, rows))
    arm_gain, blend_gain = macro(both) - macro(fwd), macro(cand) - macro(base)
    print(f'95% CI [{np.percentile(diffs, 2.5):+.4f}, {np.percentile(diffs, 97.5):+.4f}], P>0 {np.mean(np.array(diffs) > 0):.2f} | '
          f'{"ADOPT" if blend_gain >= .002 or (arm_gain >= .002 and blend_gain >= 0) else "no"}')
    sys.exit()
out = next(p for p in (ROOT / f'artifacts/geometry_gold_{arm}_notebook/output', ROOT / f'artifacts/geometry_gold_{arm}/output')
           if p.is_dir() and any(p.glob('*.csv')))  # scripts/koutput.py saves to the _notebook folder
heads = {a: read(ROOT / f'artifacts/head_retrain_{a}_notebook/output/gold_predictions_{a}.csv') for a in W}
variants = {p.stem: read(p) for p in sorted(out.glob('*.csv'))}
blend = lambda parts: sum(W[a] * parts[a] for a in W) / sum(W.values())
cur = variants[CURRENT[arm]]
base = blend({**heads, arm: cur})
print(f'saved head-retrain prediction for {arm}: {macro(heads[arm]):.4f} | same-kernel current geometry: {macro(cur):.4f}')
print(f'blend with same-kernel current geometry: {macro(base):.4f} (model61 reference 0.9326)')
rng = np.random.default_rng(20260921)
boot = []
while len(boot) < 3000:
    rows = rng.integers(0, len(y), len(y))
    if all(0 < y[rows, j].sum() < len(rows) for j in range(12)):
        boot.append(rows)
for name, pred in variants.items():
    if name == CURRENT[arm]:
        continue
    cand = blend({**heads, arm: pred})
    diffs = np.array([macro(cand, r) - macro(base, r) for r in boot])
    arm_gain, blend_gain = macro(pred) - macro(cur), macro(cand) - macro(base)
    adopt = blend_gain >= .002 or (arm_gain >= .002 and blend_gain >= 0)
    print(f'{name:18s} arm {macro(pred):.4f} ({arm_gain:+.4f}) | blend {macro(cand):.4f} ({blend_gain:+.4f}, '
          f'95% CI [{np.percentile(diffs, 2.5):+.4f}, {np.percentile(diffs, 97.5):+.4f}], P>0 {np.mean(diffs > 0):.2f}) '
          f'| {"ADOPT" if adopt else "no"}')
    worst = sorted(((auc(y[:, j], pred[:, j]) - auc(y[:, j], cur[:, j]), c) for j, c in enumerate(L)))
    print('    per finding (arm):', ', '.join(f'{c} {d:+.3f}' for d, c in worst[:3] + worst[-3:]))
