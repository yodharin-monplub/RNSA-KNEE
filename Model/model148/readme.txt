Model148 (TPU training, planned): ROUND-2 pseudo-label student, seed 20261028.
Targets data/llm_relabel/targets_v2_oof147_110_35.csv = 0.35 x v1 Gemini labels + 0.65 x mean OOF of model147's 5
student folds and model110's 5 plain folds (scripts/make_pseudo_targets.py --alpha 0.35 --runs <10 fold dirs>).
Recipe = model113's (full data, 18 epochs, last epoch). Pushed by scratch round2_chain.py after model147's folds.
Config: Model/model148/config.json. Score: pending.
