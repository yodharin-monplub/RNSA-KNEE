Model138 (TPU training, planned): pseudo-label STUDENT on 96 frames with the gold-best target mix.
Recipe = model113's (CoAtNet-2 384 rmlp, 96-frame 336 px cache, full data = all 4,345 non-gold studies, 18 epochs, last
epoch) with targets data/llm_relabel/targets_v1_oof110_25.csv = 0.25 x v1 Gemini labels + 0.75 x model110's 5-fold OOF
(built 2026-10-05 by Other/scripts/make_pseudo_targets.py --alpha 0.25 --runs <model110 fold0..4>; gold check of the
targets: alpha 0.2 0.9432 / 0.25 0.9440 / 0.3 0.9434 / 0.35 0.9423 / 0.5 0.9398) and seed 20261021.
Why: the first student (model113, alpha 0.35) beat the same recipe on plain labels by ~+0.003 on gold (0.9347 vs
0.9317, mean of epochs 14-18), and 128 frames did not help on gold -> more 96-frame students with new seeds.
Session (main account, TPU session B): python scripts/run_tpu_train.py --name model138 --plan full --cache s96 --account yodharinmonplub --then model139 --then-ratio 1.0
Config: Model/model138/config.json (differs from model113's in seed, targets and description).
Score: pending.
Pushed 2026-10-05 17:13 on the main account: yodharinmonplub/rsna-knee-tpu-train-model138-then-model139.
