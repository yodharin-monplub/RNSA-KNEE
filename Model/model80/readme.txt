Model80: model76 (the public "Bend the Knee to Speedy Raptors" notebook with this project's model61 Raptor
weights) plus a FLEET LEG: this project's fully trained models blended into the FINAL submission in rank space at
one global weight fixed in advance, w = 0.30.

Why: pjmathematician's d4-blend (public 0.946) is the Speedy pipeline (0.943) plus its author's private fleet of
trained models blended at rank weight 0.45 after the public pipeline. That is the only public +0.003 over Speedy.
Our fleet:
  model72  EfficientNet-B0, 288 px, from ImageNet          gold AUC 0.9143
  model71  ResNet34, 256 px, 5 report-group folds (mean)   gold AUC 0.9138
  model73  CoAtNet-2, 256 px, from ImageNet                gold AUC 0.9122
  fleet = mean of the three probabilities                  gold AUC 0.9258, rank corr 0.935 vs the Raptor base
Gold check (58 cases, base = model61 Raptor arms 0.9326; final = (1-w) rank(base) + w rank(fleet)):
  w 0.15 +0.0005 | 0.25 +0.0013 | 0.35 +0.0012 | 0.45 +0.0015 (P>0 0.64-0.73, noise level).
w = 0.30 sits between the flat gold region and d4-blend's 0.45. This is a leaderboard probe; the 58 gold cases have
repeatedly failed to predict the leaderboard.

How it runs (last notebook cell, after Speedy has written submission.csv):
- Every test study is read with this project's own reader (model12: 64 frames in the fixed 18/14/12/8/12 slot
  layout, 2-98% span, 140 mm crop, 336 px). The fleet was trained and gold-evaluated on exactly this geometry.
- Both GPUs, two decode threads each; windows are built once per resolution (256, 288) and shared by the models.
- submission.csv = rank((1-w) rank(Speedy) + w rank(mean fleet probability)). A study the fleet cannot read keeps
  its Speedy rank.
- Fail-safe: Speedy's file is replaced only when every study was attempted inside the 9-hour limit (process age +
  30 min safety). The leg gives up early when its projected finish is past the limit. Any error keeps Speedy's file.
- Weights: private dataset yodharinmonplub/rsna-knee-fleet-weights (built by upload_fleet_weights.py). Every loaded
  checkpoint's SHA-256 is printed.

Files: run_model80.py (builder; imports run_model76.py), upload_fleet_weights.py, notebook.ipynb,
kernel-metadata.json. Rebuild and push: python scripts/run_model80.py --push (from Other/, Kaggle key set).
Local test (CPU, real checkpoints, fake reader): the fleet leg loads all 7 checkpoints strictly, blends with the
right column order, keeps the Speedy rank for an unreadable study, and leaves submission.csv untouched on a missing
checkpoint or a time shortfall.

Kaggle: yodharinmonplub/rsna-knee-model80-speedy-our-heads-plus-fleet, version 1 (T4 x2).
Dummy-data run VERIFIED 2026-09-27: our three model61 Raptor checkpoints loaded (no public Raptor weights), no Raptor
errors or fallbacks, "[model80] fleet leg blended at w=0.3: 3/3 studies read by the fleet, 0 failed ... 33s on 2
GPUs", 7 fleet checkpoints with SHA-256, submission.csv written (whole run ~6 min).
SUBMITTED 2026-09-27 07:26 (+07), submission 56593028.
SCORE: public 0.944 - 0.001 BELOW model76 (0.945). The 3-member fleet (0.933 alone, model81) is too weak to help at w 0.30.
Pre-declared rule: model83 (w 0.45) needed model80 > model76 or model81 >= 0.940 - neither held, so model83 is dropped.
Crude fit from this result: final ~ A - w(A-B) + w(1-w)D with A 0.945, B 0.933 gives a diversity term D ~ 0.012;
for the 4-member fleet (B 0.938, model88) it predicts ~0.9455 at w 0.2-0.3, i.e. about a tie with model76.
