Model132 (planned): second PSEUDO-LABEL STUDENT - model113's recipe (full-data CoAtNet-2 384, 96-frame cache, 18 epochs,
last epoch) with targets_v1_oof110_50.csv = 0.5 x v1 Gemini labels + 0.5 x model110 5-fold OOF, seed 20261018.
Trained after model113 in the same TPU session (run_tpu_train.py --name model113 --plan full --cache s96 --then model132).
Purpose: LB comparison of the label/teacher mix (model113 = 0.35/0.65) and a second student for the leg.
Files: config.json (code: Model/model113 / Other/scripts).
Score: pending.
