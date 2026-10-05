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
SESSION 1 RESULT (2026-10-01 05:05): only fold 0 finished (cold /tmp cache slowed epochs 1-2, a late slowdown in
process 3 cut it to 16 epochs; the old kernel rule then skipped folds 1-2). Fold 0: gold AUC 0.9343 at epoch 16
(0.9363 at epoch 15) - best single model so far; OOF for 832 studies; fleet weights model110_fold0_best.pt
(sha256 b837d3c7...). Folds 1-4 (+ model112): next session, after the teammate TPU quota refresh (Sat 07:00),
kernel model110-f1234-then-model112 (cache in /dev/shm, steady-state fold estimate) pushed by scratch tpu_chain2.py.
Score: fold 0 gold 0.9343 (see score.txt).
2026-10-04 15:54 ALL FIVE FOLDS DONE: fold 1 0.9258 / fold 2 0.9178 / fold 3 0.9262 (TPU session 2,
krittanutsomtuas/rsna-knee-tpu-train-model110-f1234-then-model112) and fold 4 0.9339 (session 3,
krittanutsomtuas/rsna-knee-tpu-train-model110-f4-then-model112; 18 epochs, OOF 892, best.pt sha256 e521aa2d...).
Gold of the 5-fold mean: 0.9350. Complete OOF (4,345 non-gold studies) -> pseudo-label targets
data/llm_relabel/targets_v1_oof110_35.csv (gold 0.9423) and _50.csv (0.9398) for the students model113/132/135/137.
All five folds are in the fleet weights (model110_fold{0..4}_best.pt).
