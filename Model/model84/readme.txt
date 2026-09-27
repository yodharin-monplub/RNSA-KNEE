Model84 (prepared, not yet run): model72 exactly - EfficientNet-B0 at 288 px trained from ImageNet, all layers, GPU
augmentation, plain soft-label BCE, 16 windows/study, batch 8, cosine, 25 epochs, epoch chosen by weak-validation
BCE, SAME seed 20261003 - with ONE change: the training targets are pseudo-labels,
  targets_v1_oof5f50.csv = 0.5 * Gemini v1 labels + 0.5 * model71's 5-fold out-of-fold predictions
  (gold rows keep v1; each study's OOF value comes from a fold model that never saw it).
As labels these targets agree with gold at 0.9316 macro AUC, v1 alone at 0.9035.

Why: pseudo-labels failed twice here, but only for FROZEN-feature head retrains of the Raptor arms (model61 heads),
where mixing image-model predictions into the labels removed signal the frozen features needed. The forum says top
teams pseudo-label with full image-model training (noisy student), which is untested here. Paired with model72
(gold 0.9143): only the targets differ, so the gold difference isolates their effect (still noisy on 58 cases; the
leaderboard test is a fleet-only submission with this model in place of model72).
Files: config.json (Other/model84/config.json), Other/cloud/run_model84.sh, cloud/model84_budget.json (cap 1.2 USD,
~2.3 h, ~0.4-0.5 USD), cloud/model84_ready.py (gate passed 2026-09-27). Launch after model75 (one instance at a
time): RSNA_CLOUD_RUN=model84 python cloud/launch_only.py, then cloud/deploy_run.py.
Score: pending.
