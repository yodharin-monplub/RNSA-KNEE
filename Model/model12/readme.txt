Model12: Max-Span CoAtNet checkpoint
Status: offline Kaggle version 1 COMPLETE; visible-test submission output
validated for schema, study order, finite values and bounds. Script runtime
22.0 seconds for three visible studies. Not yet submitted for scoring.
58-case gold development evaluation completed. Prioritize for the next scoring
slot: best single model in this comparison, with overlapping confidence intervals.
Weights: dreaddevelopment/raptor-knee-maxspan/raptor_ft_coatnet_v5_full_swa.pt.
Credit: Johnathan Wagner / Dread Development. License: CC0 per Kaggle metadata.
Geometry: 64 slices, 140mm crop, 336px intermediate images, 384px backbone,
2%-98% sampling and all valid-extent three-slice windows (up to 62).
The dataset card explicitly documents these parameters and reports 0.928
public AUC. This is a publisher claim, not yet reproduced by this project.
Architecture, ordering, normalization, five slots and microbatches match model8.
Gold checks are on the 58 labeled competition cases, which have also been used
by checkpoint authors for development; these are not an untouched test set.
Source: https://www.kaggle.com/datasets/dreaddevelopment/raptor-knee-maxspan
Metadata snapshot: artifacts/sources/maxspan/dataset-metadata.json.
Run inference: scripts/run_model12.py. Gold evaluation: scripts/run_coatnet_gold.py.

Gold development evaluation (2026-09-11): macro AUC 0.919929.
58 cases; checkpoint authors used these during development, so this is not an
untouched holdout. Paired uncertainty: artifacts/coatnet_gold_notebook/output/analysis.json.

Confirmed public AUC0.928, submission56153792, version1. Model16 fine-tuning of this checkpoint scored0.932.
