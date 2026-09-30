Model110 (queued): 5-fold CoAtNet-2 384 on the 96-frame 336 px cache - model104's recipe (model75's settings; model104
reached gold 0.9306, the best own model) with seed 20261014, 18 epochs per fold, the LAST epoch of each fold kept
(select 'last'), train_model16's report-group folds (md5(report_hash) % 5 over train + weak_val, gold never trained on),
out-of-fold predictions written per fold (oof_predictions.csv).
Purpose: five strong members for the CoAtNet-only leg (model103: model75 + model100 leg = public 0.947) and OOF
predictions from a strong teacher for a full-training pseudo-label round (the forum's 0.95 single-model recipe; our
earlier pseudo-label failures used a weak ResNet34 teacher and frozen-feature heads).
Runs on the teammate account's TPU (krittanutsomtuas; own 20 h/week quota): session 1 = folds 0,1,2
(krittanutsomtuas/rsna-knee-tpu-train-model110-f012, pushed 2026-09-30 13:34), session 2 = folds 3,4.
Files: config.json, train_xla.py, xla_data.py, tpu_common.py, run_tpu_train.py (--name model110 --plan smoke+folds
--folds 0,1,2 --cache s96 --account krittanutsomtuas), build_trainer_kit.py, collect_tpu_run.py.
Score: pending.
