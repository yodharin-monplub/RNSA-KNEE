Model148 (TPU training, planned): ROUND-2 pseudo-label student, seed 20261028.
Targets data/llm_relabel/targets_v2_oof147_110_35.csv = 0.35 x v1 Gemini labels + 0.65 x mean OOF of model147's 5
student folds and model110's 5 plain folds (scripts/make_pseudo_targets.py --alpha 0.35 --runs <10 fold dirs>).
Recipe = model113's (full data, 18 epochs, last epoch). Pushed by scratch round2_chain.py after model147's folds.
Config: Model/model148/config.json. RESULT 2026-10-12 07:14 (teammate TPU, queued 13:04 -> ran 04:24-07:14): gold 0.9347 (round-1 student model113 0.9357, plain model112 0.9319) - round-2 targets (gold 0.9441 vs 0.9423) give no visible single-model gain on the 58-study gold set. Checkpoint added to yodharinmonplub/rsna-knee-fleet-weights as model148_best.pt. No LB score yet (main GPU quota until 2026-10-17).
