Model 4: equal-weight ensemble of model2 and model3
Status: evaluated. Gold macro AUC: 0.753488 (58 held-out studies).
No additional backbone training. Local results: artifacts/comparison/metrics.json.
Inference: average model2 and model3 probabilities, weight 0.5 each.
The blend is fixed before inspecting gold results; no label-specific tuning on
the small gold set. Both heads use the same frozen encoder for efficient inference.
Run: python -m knee.infer --data DATA_ROOT --weights WEIGHTS_ROOT
Evaluation: python scripts/compare_models.py
Gold evaluation has 58 cases; bootstrap intervals quantify some sampling noise,
but do not address site shift or development leakage in public weak labels.
No prize or leaderboard score is implied by a local result.
