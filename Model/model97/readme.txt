Model97: model75 (CoAtNet-2 trained from ImageNet at 384 px; 0.939 public alone) with its prediction HEAD retrained on
frozen full-bag features - the recipe that lifted every Raptor arm (+0.002 to +0.005 gold, 12 of 12 seeds) and moved the
leaderboard 0.942 -> 0.943 (model61). model75 itself trained its head on 16 random windows per study; the retrain uses all
(up to 62) windows per study, plain soft-label BCE on the Gemini v1 labels, AdamW lr 3e-4, wd 2e-3, batch 128, 6 epochs,
starting from model75's own head (recipe fixed earlier, not tuned here); 3 seeds are scored on gold for information, the
final head (seed 7) is written as model97_model75_best.pt (model75's checkpoint layout, head replaced).
Kernel: yodharinmonplub/rsna-knee-head-retrain-model75 (Other/scripts/run_head_retrain.py, arm 'model75'; features
extracted on a Kaggle T4 from the 64-frame cache kernel outputs, ~3 h; the known-gold sanity check requires the cached
features to reproduce model75's gold 0.9204 before training). Pushed 2026-09-28 01:21 after the user allowed Kaggle GPU
use ("both ok").
Use: if the retrained head beats model75 on gold (and it is expected to), it replaces model75 in the fleet leg / as a
Raptor view.
Score: pending.
