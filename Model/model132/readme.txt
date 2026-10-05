Model132 (planned): second PSEUDO-LABEL STUDENT - model113's recipe (full-data CoAtNet-2 384, 96-frame cache, 18 epochs,
last epoch) with targets_v1_oof110_50.csv = 0.5 x v1 Gemini labels + 0.5 x model110 5-fold OOF, seed 20261018.
Trained after model113 in the same TPU session (run_tpu_train.py --name model113 --plan full --cache s96 --then model132).
Purpose: LB comparison of the label/teacher mix (model113 = 0.35/0.65) and a second student for the leg.
Files: config.json (code: Model/model113 / Other/scripts).
Score: pending.
2026-10-04 11:44: DEFERRED. TPU session 4 now trains model113 then model135 (model113's exact recipe on the new 128-frame
cache) instead of model132: the s96 vs s128 A/B answers a bigger question with the same TPU time. model132 (alpha 0.5,
seed 20261018) stays prepared for a later session.
2026-10-04 12:30: RESCHEDULED on the MAIN account's TPU (AGENTS.md now lets the main account train): model132 (s96) then
model137 (model135's recipe with seed 20261019, s128) in one session, pushed once the full 5-fold targets exist.
Pushed 2026-10-04 16:22 on the main account: yodharinmonplub/rsna-knee-tpu-train-model132-then-model137 (model132 first, then model137 on the 128-frame cache).
TRAINED 2026-10-05 (main-account TPU, yodharinmonplub/rsna-knee-tpu-train-model132-then-model137, ran 10:41-17:06 after
~18 h queued; 18 epochs): gold AUC 0.9336 at epoch 18 (mean of epochs 14-18 0.9328) - alpha 0.5 is ~0.002 below the
alpha 0.35 student model113 (0.9347) on gold. best.pt sha256 789e77fb..., fleet weights model132_best.pt. LB pending.
