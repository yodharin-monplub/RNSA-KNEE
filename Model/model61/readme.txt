Model61: model43 with only the prediction heads of the four CoAtNet arms retrained.

Why: the fine-tunes behind model43 (models 38-41) moved the public checkpoints' heads with lr 1e-5 for two epochs
on 12 random windows per study. scripts/run_head_gold_cv.py showed that retraining just the head on frozen
features of ALL windows (batch 128 studies, AdamW lr 3e-4, wd 2e-3, 6 epochs, grad clip 2) with PLAIN soft-label
BCE on the Gemini relabels lifts the maxspan arm from 0.9254 to 0.9317 +/- 0.0010 gold AUC (3 seeds) - without
using any gold study - while the project's usual confidence-weighted BCE gives 0.9281 and mixing the gold studies
into head training hurts at every weight (0.9248 / 0.9210 / 0.9163 for weights 4 / 14 / 40).

How: scripts/run_head_retrain.py <arm> (free Kaggle T4) encodes every window of all 4,407 studies with the arm's
frozen backbone, asserts that the arm's own head reproduces its known gold AUC from those features, trains the head
(recipe fixed above, never tuned on gold; 3 seeds for the noise estimate, final head seed 7, trained on all 4,349
non-gold studies) and writes model61_<arm>_best.pt = the arm's checkpoint with the head replaced, verified through
Classifier.predict_features (max abs diff < 5e-7).

Gold AUC (58 cases, never trained on), original -> retrained final head:
  maxspan (model38)        0.9254 -> 0.9308   (seeds 0.9300 / 0.9305 / 0.9313)
  native384dense (model39) 0.9205 -> 0.9236   (seeds 0.9242 / 0.9235 / 0.9239)
  native384-v8 (model40)   0.9197 -> 0.9218   (seeds 0.9223 / 0.9223 / 0.9224)
  fine-spacing (model41)   0.9160 -> see score.txt
CoAtNet-side blend (weights 0.70/0.15/0.30/0.15): 0.9305 -> see score.txt (scripts/blend_model61.py).

Submission: notebook rsna-knee-model61-retrained-heads version 1, competition submission 56354360 (2026-09-19).
Public AUC 0.943 - new project best (model43 with the original heads: 0.942). Scored in about 5.4 hours.
fine-spacing arm 0.9160 -> 0.9181; four-arm blend CV 0.9305 -> 0.9326 (details in score.txt).

Notebook: scripts/run_model23_24.py model61 = model43's exact path (same views, geometry, weights, grouped decode
with prefetch, public DINO/RadImageNet side untouched) plus HEAD_RETRAIN_ARMS, which points every view at the
checkpoint in its head-retrain kernel's output. Runtime is unchanged (same backbones and number of views).
