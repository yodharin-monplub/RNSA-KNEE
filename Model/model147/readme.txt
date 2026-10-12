Model147 (TPU training, planned - pushed only if the student model113 beats model104 on the LB, Sat 2026-10-10):
5-FOLD pseudo-label STUDENT = model110's k-fold recipe on model113's student targets (alpha 0.35), seed 20261027.
Purpose: its out-of-fold predictions (5 student folds) form a stronger teacher than model110's OOF for a SECOND
pseudo-label round (round-2 targets -> round-2 full-data students).
Plan (next week's 40 h of TPU on both team accounts): folds split over the two accounts, e.g.
  run_tpu_train.py --name model147 --plan folds --folds 0,1,2 --cache s96 --account krittanutsomtuas
  run_tpu_train.py --name model147 --plan folds --folds 3,4 --cache s96 --account yodharinmonplub
then scripts/make_pseudo_targets.py on the 5 student folds (+ model110's) and two round-2 students.
Config: Model/model147/config.json (model110's config with model113's targets and a new seed).
RESULT 2026-10-11 (teammate TPU, folds 0-2 ran 17:23-01:14, folds 3-4 07:02-12:48): gold per fold 0.9308 / 0.9341 / 0.9343 / 0.9293 / 0.9331 (mean 0.932; plain model110 folds mean 0.928). OOF + model110 OOF -> targets_v2_oof147_110_35.csv (gold mix 0.9441). Fold checkpoints added to the fleet weights as model147_fold{k}_best.pt. No LB score yet.
