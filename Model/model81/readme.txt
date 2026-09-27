Model81: this project's fully trained fleet ALONE as a submission (no public models) - a leaderboard calibration.

fleet = mean probability of
  model72  EfficientNet-B0, 288 px, from ImageNet          gold AUC 0.9143
  model71  ResNet34, 256 px, 5 report-group folds (mean)   gold AUC 0.9138
  model73  CoAtNet-2, 256 px, from ImageNet                gold AUC 0.9122
  fleet                                                    gold AUC 0.9258 (58 cases)
Why: model80 blends this fleet into the public Speedy pipeline at w = 0.30 (the d4-blend recipe). How much weight the
fleet deserves depends on how good it is on the leaderboard, which the 58 gold cases cannot tell (a competitor's
gold 0.925 model scores 0.947 public; forum teams report 5-fold EfficientNet-B0 ensembles at 0.94+ public). This run
measures it directly and cheaply: it needs no public models and runs in minutes on the hidden set.

Code: the same cell as model80's fleet leg (this project's model12 reader geometry - 64 frames in the 18/14/12/8/12
slot layout, 2-98% span, 140 mm crop, 336 px - both GPUs, strict checkpoint loads with SHA-256), except that the
fleet's probabilities are the submission, in sample_submission's column order. A study the fleet cannot read gets the
fleet's mean over the readable studies; fewer than 97% readable studies, or any error, fails the run (there is
nothing to fall back to).
Files: run_model81.py (builder, the notebook cell is embedded), notebook.ipynb, kernel-metadata.json. Weights: private
dataset yodharinmonplub/rsna-knee-fleet-weights (Model/model80/upload_fleet_weights.py). Local CPU test with the real
checkpoints and a fake reader: passes (neutral fill, column order).
Kaggle: yodharinmonplub/rsna-knee-model81-fleet-only v1 (T4 x2). Dummy run verified (3/3 studies read, 7 checkpoints,
no errors); submitted 2026-09-27 07:29 (+07), submission 56593100.
SCORE: public 0.933 (scored ~25 min after submission, so the fleet leg costs well under 30 min on the hidden set).
Reading: our own fleet is 0.010 below the public Speedy stack (0.943). Gold 0.9258 -> LB 0.933, consistent with
model61's blend (gold 0.9326 -> LB 0.943). A 0.933 fleet is weaker than d4-blend's private fleet (which added +0.003 at
w 0.45), so model80 (w 0.30) decides whether it still helps through diversity; model83 (w 0.45) stays on hold.
