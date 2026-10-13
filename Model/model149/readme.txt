Model149 (TPU training, planned): ROUND-2 pseudo-label student, seed 20261029.
Targets data/llm_relabel/targets_v2_oof147_110_35.csv = 0.35 x v1 Gemini labels + 0.65 x mean OOF of model147's 5
student folds and model110's 5 plain folds (scripts/make_pseudo_targets.py --alpha 0.35 --runs <10 fold dirs>).
Recipe = model113's (full data, 18 epochs, last epoch). Pushed by scratch round2_chain.py after model147's folds.
Config: Model/model149/config.json. RESULT 2026-10-13 00:37 (teammate TPU, pushed 22:07): gold 0.9341 (model148 0.9347, round-1 model113 0.9357) - round 2 gives no visible single-model gain on the 58-study gold set. Checkpoint added to yodharinmonplub/rsna-knee-fleet-weights as model149_best.pt. No LB score yet.
