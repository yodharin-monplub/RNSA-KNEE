"""Pseudo-label targets for full-training students: report-derived soft labels mixed with out-of-fold image predictions.

targets = alpha * labels + (1 - alpha) * OOF for every non-gold study with an out-of-fold prediction; the OOF of several
k-fold runs (e.g. two 5-fold models) is averaged per study. Studies without OOF keep their labels, gold rows keep their
labels too (gold is never trained on).

Sanity check (as on 2026-09-24): the same mix on the 58 gold studies, with the runs' gold predictions averaged in place of
the OOF, scored against the gold reads (macro AUC over the 12 findings; the v1 labels alone score 0.9035).

Usage:
  python scripts/make_pseudo_targets.py --labels data/llm_relabel/targets_g38f_low.csv --alpha 0.5 \
      --runs artifacts/remote_model110/fold0 artifacts/remote_model110/fold1 ... \
      --out data/llm_relabel/targets_v1_oof110_50.csv [--allow-partial]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
UID = 'StudyInstanceUID'
LABELS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion',
          'Synovitis', "Baker's", 'Contusion', 'Fracture']


def macro_auc(truth, pred):
    scores = [roc_auc_score(truth[c], pred[c]) for c in LABELS if truth[c].nunique() == 2]
    return float(np.mean(scores)), len(scores)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--labels', required=True)
    parser.add_argument('--runs', nargs='+', required=True, help='run folders with oof_predictions.csv (+ gold_predictions.csv)')
    parser.add_argument('--alpha', type=float, default=0.5, help='weight of the report labels')
    parser.add_argument('--out', required=True)
    parser.add_argument('--allow-partial', action='store_true', help='accept OOF for only part of the non-gold studies')
    args = parser.parse_args()

    labels = pd.read_csv(ROOT / args.labels).set_index(UID)[LABELS]
    partition = pd.read_csv(ROOT / 'artifacts/partitions/partitions.csv').set_index(UID)
    gold_ids = partition.index[partition.split == 'gold']
    pool = partition.index[partition.split.isin(['train', 'weak_val'])]

    oof, gold_preds = [], []
    for run in args.runs:
        run = ROOT / run if not Path(run).is_absolute() else Path(run)
        frame = pd.read_csv(run / 'oof_predictions.csv').set_index(UID)[LABELS]
        assert frame.index.is_unique and not frame.index.isin(gold_ids).any(), run
        oof.append(frame)
        if (run / 'gold_predictions.csv').exists():
            gold_preds.append(pd.read_csv(run / 'gold_predictions.csv').set_index(UID)[LABELS])
        print(f'{run.name}: {len(frame)} OOF studies')
    stacked = pd.concat(oof)
    counts = stacked.groupby(level=0).size()
    mean_oof = stacked.groupby(level=0).mean()
    covered = mean_oof.index.intersection(pool)
    print(f'OOF coverage {len(covered)} of {len(pool)} non-gold studies; predictions per study {counts.min()}-{counts.max()}')
    if len(covered) < len(pool) and not args.allow_partial:
        raise SystemExit('partial OOF coverage - pass --allow-partial to mix only the covered studies')

    targets = labels.copy()
    targets.loc[covered] = args.alpha * labels.loc[covered].to_numpy() + (1 - args.alpha) * mean_oof.loc[covered].to_numpy()
    truth_all = pd.read_csv(ROOT / 'data/train.csv').set_index(UID)
    # how far the OOF moves the labels (mean absolute change per finding, covered studies)
    shift = (targets.loc[covered] - labels.loc[covered]).abs().mean()
    print('mean |change| per finding:', {c: round(float(v), 3) for c, v in shift.items()})

    truth = truth_all.loc[gold_ids, LABELS]
    base, n = macro_auc(truth, labels.loc[gold_ids])
    print(f'gold check ({len(gold_ids)} studies, {n} findings): labels alone {base:.4f}')
    if gold_preds:
        g = pd.concat(gold_preds).groupby(level=0).mean().loc[gold_ids]
        mixed = args.alpha * labels.loc[gold_ids] + (1 - args.alpha) * g
        print(f'  image models alone (mean of {len(gold_preds)} runs) {macro_auc(truth, g)[0]:.4f} | '
              f'mix alpha {args.alpha} {macro_auc(truth, mixed)[0]:.4f}')
    out = Path(args.out) if Path(args.out).is_absolute() else ROOT / args.out
    targets.round(4).reset_index().to_csv(out, index=False)
    print('wrote', out, len(targets), 'rows')


if __name__ == '__main__':
    main()
