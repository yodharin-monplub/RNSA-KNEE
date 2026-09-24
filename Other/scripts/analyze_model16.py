"""Descriptive model16 comparisons with fixed equal blends; no weight search."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from compare_models import bootstrap_auc
ROOT=Path(__file__).resolve().parents[1]
labels=list(pd.read_csv(ROOT/'data/sample_submission.csv').columns[1:])
gold=pd.read_csv(ROOT/'data/train.csv').dropna(subset=labels).set_index('StudyInstanceUID')
assert len(gold)==58 and gold.index.is_unique
predictions={}
for name,path in {
 'model12':'artifacts/coatnet_gold_notebook/output/model12_predictions.csv',
 'model16':'artifacts/remote_model16/model16/gold_predictions.csv',
 'model18':'artifacts/model18_development/predictions.csv',
}.items():
 f=pd.read_csv(ROOT/path).set_index('StudyInstanceUID')
 assert f.index.is_unique and set(f.index)==set(gold.index)
 p=f.loc[gold.index,labels].to_numpy()
 assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
 predictions[name]=p
predictions['equal_12_16']=(predictions['model12']+predictions['model16'])/2
predictions['equal_18_16']=(predictions['model18']+predictions['model16'])/2
y=gold[labels].to_numpy()
indices=np.random.default_rng(20260911).integers(0,len(y),(2000,len(y)))
draws={k:bootstrap_auc(y,p,indices) for k,p in predictions.items()}
report={'notes':'58 development cases, not an untouched holdout. Two equal blends specified before computing their scores; no weight optimization.', 'models':{}}
for k,p in predictions.items():
 report['models'][k]={'macro_auc':float(roc_auc_score(y,p)), 'paired_delta_vs_model12_95pct':np.quantile(draws[k]-draws['model12'],[.025,.975]).tolist(), 'per_label_auc':dict(zip(labels,roc_auc_score(y,p,average=None).tolist()))}
 print(k,report['models'][k]['macro_auc'],report['models'][k]['paired_delta_vs_model12_95pct'])
out=ROOT/'artifacts/model16_development';out.mkdir(exist_ok=True)
(out/'analysis.json').write_text(json.dumps(report,indent=2))
