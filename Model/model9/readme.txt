Model9: SWA CoAtNet with full window coverage
Status: private offline Kaggle version 1 COMPLETE; competition submission
56143463 COMPLETE, public AUC 0.927, verified on 2026-09-11 via Kaggle API.
Ties model8 at the reported three-decimal precision. Output schema, study order, finite values,
and probability bounds validated. Visible-test script runtime: 21.5 seconds
for three cases including setup; hidden-test runtime is not available.
Parent: model8 (verified public AUC 0.927).
Only prediction change: use raptor_ft_coatnet_v4_full_swa.pt instead of
raptor_ft_coatnet_v4_full.pt from dreaddevelopment/raptor-knee-widedense.
Publisher describes SWA as averaging the three best training epochs.
Hypothesis: averaged weights improve generalization at unchanged model size
and window count. Public leaderboard evaluation is exploratory.
Preprocessing, up to 62 windows, attention pooling, 12-window microbatches,
and raw probabilities are identical to model8. No test-time augmentation.
Checkpoint credit: Johnathan Wagner / Dread Development; CC0 weights.
Architecture and geometry attribution: SOURCES.md and model7/readme.txt.
No new local training or Vast.ai rental. Private score remains unknown.
Run: .venv/bin/python scripts/run_model9.py

Gold development evaluation (2026-09-11): macro AUC 0.915364.
58 cases; checkpoint authors used these during development, so this is not an
untouched holdout. Paired uncertainty: artifacts/coatnet_gold_notebook/output/analysis.json.
