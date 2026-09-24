"""Audit data and evaluate a CPU metadata baseline. This is not an MRI model."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
UID = 'StudyInstanceUID'


def features(series, ids):
    # Use only acquisition fields available at inference, never UIDs or reports.
    frame = pd.get_dummies(series[['Fluid_Sensitive', 'Fat_Suppression',
                                   'Anatomical_Plane']], dtype=float)
    frame[UID] = series[UID]
    grouped = frame.groupby(UID).agg(['mean', 'sum'])
    grouped.columns = ['_'.join(c) for c in grouped.columns]
    grouped['series_count'] = series.groupby(UID).size()
    return grouped.reindex(ids).fillna(0)


def main():
    out = ROOT / 'artifacts/baseline'
    out.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(ROOT / 'data/train.csv')
    test = pd.read_csv(ROOT / 'data/test.csv')
    sample = pd.read_csv(ROOT / 'data/sample_submission.csv')
    labels = list(sample.columns.drop(UID))
    assert train[UID].is_unique and test[UID].is_unique
    assert set(train[UID]).isdisjoint(test[UID])
    for label in labels:
        assert set(train[label].dropna().unique()) <= {0, 1}
    labeled = train.loc[train[labels].notna().all(axis=1)].reset_index(drop=True)
    assert len(labeled) >= 10, 'Insufficient labeled studies for validation'
    x = features(pd.read_csv(ROOT / 'data/train_series.csv'), labeled[UID])
    xt = features(pd.read_csv(ROOT / 'data/test_series.csv'), test[UID])
    xt = xt.reindex(columns=x.columns, fill_value=0)
    # Exact normalized duplicate reports stay in one fold. Patient/site IDs
    # are absent from these CSVs, so this cannot establish site independence.
    groups = labeled.Report.fillna('').map(
        lambda s: hashlib.sha256(' '.join(s.lower().split()).encode()).hexdigest())
    splits = list(GroupKFold(n_splits=5).split(x, groups=groups))
    oof = np.zeros((len(labeled), len(labels)))
    fold_ids = np.full(len(labeled), -1)
    for fold, (tr, va) in enumerate(splits):
        fold_ids[va] = fold
        for j, label in enumerate(labels):
            y = labeled[label].to_numpy()
            if len(np.unique(y[tr])) < 2:
                oof[va, j] = (y[tr].sum() + 1) / (len(tr) + 2)
            else:
                model = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=1000))
                model.fit(x.iloc[tr], y[tr])
                oof[va, j] = model.predict_proba(x.iloc[va])[:, 1]
    scores = {label: float(roc_auc_score(labeled[label], oof[:, j]))
              if labeled[label].nunique() == 2 else None
              for j, label in enumerate(labels)}
    prediction = test[[UID]].copy()
    for label in labels:
        y = labeled[label]
        if y.nunique() < 2:
            prediction[label] = (y.sum() + 1) / (len(y) + 2)
        else:
            model = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=1000))
            model.fit(x, y)
            prediction[label] = model.predict_proba(xt)[:, 1]
    submission = sample[[UID]].merge(prediction, on=UID, validate='one_to_one')[sample.columns]
    assert submission.shape == sample.shape
    assert np.isfinite(submission[labels]).all().all()
    assert submission[labels].ge(0).all().all() and submission[labels].le(1).all().all()
    submission.to_csv(out / 'submission.csv', index=False)
    folds = labeled[[UID]].assign(fold=fold_ids, report_hash=groups)
    folds.to_csv(out / 'folds.csv', index=False)
    pd.DataFrame(oof, columns=labels).assign(**{UID: labeled[UID]}).to_csv(out / 'oof.csv', index=False)
    summary = {
        'training_studies': len(train), 'fully_labeled_studies': len(labeled),
        'test_studies': len(test), 'test_has_report': 'Report' in test,
        'label_counts': {c: {'known': int(train[c].notna().sum()),
                             'positive': int(train[c].sum())} for c in labels},
        'per_label_oof_auc': scores,
        'macro_oof_auc': float(np.mean([s for s in scores.values() if s is not None])),
        'validation': '5-fold GroupKFold on normalized report hashes; C=0.1 fixed before evaluation',
        'limitations': 'Only 58 labeled studies; no patient/site grouping available. Metadata may encode site. Local OOF AUC is not a leaderboard score. Reports excluded from predictors.'
    }
    (out / 'metrics.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
