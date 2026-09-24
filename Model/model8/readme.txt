Model8: CoAtNet full window coverage
Status: offline Kaggle version 1 completed; submission 56141935 accepted.
Public AUC 0.927, COMPLETE, verified via Kaggle API on 2026-09-10.
Improvement over model7: +0.003 public AUC. Private score unavailable.
Visible-test output schema and probabilities validated.
Visible-test script time: 23.2 seconds vs model7's 20.3 seconds; three cases
including setup time, not a reliable hidden-test runtime estimate.
Parent: model7, reproduced public AUC 0.924.
Only prediction change: evaluate each three-slice window within the valid
stack extent once, up to 62 windows, instead of sampling 42 windows.
Hypothesis: additional slice coverage improves finding detection. Expected
backbone work rises by up to 62/42 (48%); this is an accuracy experiment.
The checkpoint, five-slot 64-slice geometry, normalization, attention pooling,
12-window GPU microbatches, and raw probability output are unchanged.
Weights: dreaddevelopment/raptor-knee-widedense/raptor_ft_coatnet_v4_full.pt.
Checkpoint credit: Johnathan Wagner / Dread Development; CC0 weights.
Architecture and preprocessing attribution: SOURCES.md and model7/readme.txt.
Our measured public AUC matches the publisher's reported 0.927 at 62 windows.
Discussion: https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/737696
No local training or new Vast.ai rental is required. Public leaderboard
evaluation is exploratory; private performance is unknown.
Run: .venv/bin/python scripts/run_model8.py

Gold development evaluation (2026-09-11): macro AUC 0.916053.
58 cases; checkpoint authors used these during development, so this is not an
untouched holdout. Paired uncertainty: artifacts/coatnet_gold_notebook/output/analysis.json.
