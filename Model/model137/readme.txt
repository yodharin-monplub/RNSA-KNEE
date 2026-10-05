Model137 (TPU training, planned): second 128-FRAME pseudo-label student = model135's exact recipe with seed 20261019.
CoAtNet-2 384 rmlp, full data (all 4,345 non-gold studies), 18 epochs, last epoch; targets
data/llm_relabel/targets_v1_oof110_35.csv (0.35 x v1 Gemini labels + 0.65 x model110's 5-fold OOF); input = the 128-frame
336 px cache (krittanutsomtuas/rsna-knee-dense-cache-336-part{0..3}-s128).
Why: the 2026-10-04 AGENTS.md update lets the main account train, which adds its 20 h/week TPU. Main's first session
hedges the density question: model132 (alpha 0.5 student, s96, seed 20261018) then model137 (this, s128). With TPU
session 4 on the teammate account (model113 s96 + model135 s128) that gives two students per density.
Session (main account):
  python scripts/run_tpu_train.py --name model132 --plan full --cache s96 --account yodharinmonplub --then model137 --then-cache s128 --then-ratio 1.15
Config: Model/model137/config.json (differs from model135's only in seed and description).
In the fleet leg: geometry 's128x336' of fleet leg v5 (reader parity with the cache checked on 58 gold studies).
Score: pending.
Pushed 2026-10-04 16:22 on the main account: yodharinmonplub/rsna-knee-tpu-train-model132-then-model137 (model132 first, then model137 on the 128-frame cache).
