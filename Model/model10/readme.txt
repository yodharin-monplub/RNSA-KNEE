Model10: fine-spacing CoAtNet checkpoint with full coverage inference
Status: offline Kaggle version 1 COMPLETE; submission 56148775 accepted.
Public score pending. Strict state loading and output validation passed.
Visible-test script time: 21.8 seconds for three cases including setup.
Checkpoint metadata confirms architecture, resolution 384 and epoch 12;
its embedded gold_auc is publisher metadata, not our validation result.
Comparison: model8 public AUC 0.927, model9 public AUC 0.927.
Weights: dreaddevelopment/raptor-knee-finespacing/raptor_ft_coatnet_v9_full.pt.
Checkpoint credit: Johnathan Wagner / Dread Development; CC0 weights.
Dataset metadata snapshot: artifacts/sources/finespacing/dataset-metadata.json.
The publisher describes different training slice sampling but does not give
the full geometry in the dataset card. We deliberately evaluate the checkpoint
using model8's verified 64-slice, 140mm, 6%-94% pipeline and full window coverage.
This is a transfer experiment, not a verified reproduction of training geometry.
Model architecture, strict state loading, label order checks, microbatch size,
normalization, and probability output match model8. No new training required.
No independent gold-validation performance is claimed for these public weights.
Source: https://www.kaggle.com/datasets/dreaddevelopment/raptor-knee-finespacing
Run: .venv/bin/python scripts/run_model10.py

Gold development evaluation (2026-09-11): macro AUC 0.909689.
58 cases; checkpoint authors used these during development, so this is not an
untouched holdout. Paired uncertainty: artifacts/coatnet_gold_notebook/output/analysis.json.

Confirmed public score:0.929; submission56148775 COMPLETE. New campaign best.
