Model109 (queued): a third seed of model75's CoAtNet-2 384 recipe (64-frame 336 px cache, gradient checkpointing,
16 windows/study, global batch 8, plain soft-label BCE on the Gemini v1 targets, augmentation, 18 epochs, epoch by the
confidence-weighted weak-val BCE of train_model16), seed 20261013, trained on a Kaggle TPU v5e-8 in fp32.
Purpose: another strong member for the CoAtNet-only leg (model103 = model75 + model100 leg, public 0.947) and the first
TPU run with the fixes for the epoch-14 slowdown: fresh processes every 7 epochs (resume from last.pt, RELAUNCH
marker, XLA persistent compile cache), per-epoch XLA compile count / device memory / host memory in history.json.
Files: config.json, train_xla.py + xla_data.py + tpu_common.py (trainer), run_tpu_train.py (kernel: --name model109
--plan smoke+full), build_trainer_kit.py (kit dataset), collect_tpu_run.py (downloads the outputs).
Pushed 2026-09-30 13:12 as yodharinmonplub/rsna-knee-tpu-train-model109 (batch TPU queue).
Score: pending.

RESULT 2026-09-30: ran on the MAIN account's TPU (queued before the 2026-09-30 submit-only rule; the user's cancel did
not take effect; started 19:50, finished 23:01 +07). PLAN smoke+full OK in 11,454 s (3.2 h): processes relaunched
after epochs 7 and 14 as designed, no epoch-14 slowdown (epoch times ~300 s; 462-905 s right after relaunches).
Selected epoch 18 (select by confidence-weighted weak-val BCE): BCE 0.3118, gold AUC 0.9126 (descriptive, 58 cases).
best.pt sha256 e7aa7efedef4f993ebee35aa95e42e1fed69ab127b63b8f91774b5bf7ebbe551 (artifacts/remote_model109/model109).
Collect: python scripts/collect_tpu_run.py model109 --account yodharinmonplub (download only).
Use in a leg: the user OK'd it (Discord DM 2026-10-01 00:42: "yes you can"); added to the fleet weights as model109_best.pt.
