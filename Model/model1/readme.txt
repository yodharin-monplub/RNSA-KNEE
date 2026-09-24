Model 1: acquisition metadata logistic regression
Status: trained locally; diagnostic baseline only.
Run: .venv/bin/python scripts/baseline.py
Result: local 5-fold OOF macro AUC 0.532676 on 58 labeled studies.
Artifacts: artifacts/baseline/{metrics.json,folds.csv,oof.csv,submission.csv}
Predictors: scan counts, anatomical plane, fluid sensitivity, fat suppression.
No reports or identifiers enter the predictor. No MRI pixels are used.
This is not a leaderboard score and has not been submitted.
Its cross-validation differs from models 2/3, which hold out all 58 gold cases.
