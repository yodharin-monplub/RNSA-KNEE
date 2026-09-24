"""Compare fixed, gold-held-out models and their predefined equal ensemble."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata


def macro(y, p):
    scores = [roc_auc_score(y[:, j], p[:, j]) for j in range(y.shape[1])
              if len(np.unique(y[:, j])) == 2]
    return float(np.mean(scores))


def bootstrap_auc(y, p, indices):
    """Vectorized rank AUC, including tied predictions and single-class draws."""
    yy = y[indices]
    ranks = rankdata(p[indices], axis=1, method='average')
    positives = yy.sum(axis=1)
    negatives = yy.shape[1] - positives
    numerator = (ranks * yy).sum(axis=1) - positives * (positives + 1) / 2
    denominator = positives * negatives
    auc = np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 0)
    return np.nanmean(auc, axis=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--artifacts', default='artifacts/remote')
    args = parser.parse_args()
    artifacts = Path(args.artifacts)
    labels = json.loads((artifacts / 'partitions/labels.json').read_text())
    uid = 'StudyInstanceUID'
    gold = pd.read_csv('data/train.csv').dropna(subset=labels).set_index(uid)
    predictions = {}
    for name in ['model2', 'model3']:
        frame = pd.read_csv(artifacts / name / 'gold_predictions.csv').set_index(uid)
        assert frame.index.is_unique and set(frame.index) == set(gold.index)
        predictions[name] = frame.loc[gold.index, labels].to_numpy()
    predictions['model4'] = 0.5 * predictions['model2'] + 0.5 * predictions['model3']
    for name in ['model5', 'model6']:
        path = artifacts / name / 'gold_predictions.csv'
        if path.exists():
            frame = pd.read_csv(path).set_index(uid)
            assert frame.index.is_unique and set(frame.index) == set(gold.index)
            predictions[name] = frame.loc[gold.index, labels].to_numpy()
    y = gold[labels].to_numpy()
    rng = np.random.default_rng(2026)
    indices = rng.integers(0, len(y), (1000, len(y)))
    report = {}
    bootstraps = {}
    for name, pred in predictions.items():
        draws = bootstrap_auc(y, pred, indices)
        bootstraps[name] = draws
        report[name] = {'gold_macro_auc': macro(y, pred),
                        'bootstrap_95pct': np.quantile(draws, [.025, .975]).tolist(),
                        'gold_studies': len(y), 'leaderboard_score': None,
                        'per_label_auc': {label: float(roc_auc_score(y[:,j], pred[:,j]))
                                          for j,label in enumerate(labels)}}
    # Paired differences share the same bootstrap studies.
    report['ensemble_minus_linear_95pct'] = np.quantile(
        bootstraps['model4'] - bootstraps['model2'],
        [.025, .975]).tolist()
    report['notes'] = 'All 58 gold studies held out from training. Checkpoints selected by weak validation BCE. Ensemble fixed before gold inspection. Model1 uses a different evaluation protocol.'
    out = Path('artifacts/comparison'); out.mkdir(parents=True, exist_ok=True)
    (out / 'metrics.json').write_text(json.dumps(report, indent=2))
    frame = pd.DataFrame(predictions['model4'], columns=labels, index=gold.index)
    frame.to_csv(out / 'model4_gold_predictions.csv')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
