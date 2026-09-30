Model112 (planned): FULL-DATA CoAtNet-2 384 on the 96-frame 336 px cache - model104's recipe (model75's settings;
model104 reached gold 0.9306, our best own model) trained on ALL 4,345 non-gold studies (train + weak_val; weak_val is
only an in-sample monitor), seed 20261016, 18 epochs, the LAST epoch kept, v1 Gemini labels (targets_g38f_low.csv).
Purpose: a stronger single member for the CoAtNet-only leg (+18% training data vs model104, +25% vs a model110 fold) -
the forum's full-train recipe ("train on everything, evaluate on gold only").

Runs on Kaggle TPU v5e-8 (teammate account krittanutsomtuas, .env key KAGGLE_API_KEY_WARIN) as the second model of the
model110 folds-3,4 session, on the same unpacked cache:
  python scripts/build_trainer_kit.py --account krittanutsomtuas --push "full_data + model112"
  python scripts/run_tpu_train.py --name model110 --plan folds --folds 3,4 --cache s96 --account krittanutsomtuas --then model112 --push
The kernel (krittanutsomtuas/rsna-knee-tpu-train-model110-f34-then-model112) starts model112 only if ~1.25 x the last
fold's duration still fits before the 9 h session limit, so the 18-epoch plan is kept. Always exits 0 (a failed Kaggle
run keeps no outputs); the outcome is in the log ("PLAN ... OK/FAILED") and PLAN_STATUS.txt.
Collect: python scripts/collect_tpu_run.py model110-f34-then-model112 --account krittanutsomtuas (outputs under
model112/: best.pt, history.json, gold_predictions.csv, metrics.json with full_data true).

Files: config.json, train_xla.py (full_data option), xla_data.py, tpu_common.py, run_tpu_train.py (--then option,
exit-0 kernels), build_trainer_kit.py, collect_tpu_run.py, make_pseudo_targets.py (for the later pseudo-label student).
Score: pending.
