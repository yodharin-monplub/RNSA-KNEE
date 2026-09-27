Model89: model75 ALONE (CoAtNet-2 at 384 px, trained from ImageNet by this project, gold 0.9204) - a leaderboard
calibration. model88 (the fleet with model75 as one of four members) scored 0.938, +0.005 over the same fleet without it
(model81, 0.933), far more than gold predicted. model89 measures model75 by itself: if it is ~0.94 alone, more
full-training CoAtNet-2 at 384 px is the best use of Vast money and the fleet should lean toward it.
Code: model81's cell with a one-member fleet (run_model81.py --name model89 --fleet "[('model75', ['model75_best.pt'])]"),
this project's model12 reader geometry (64 frames, 336 px cache, windows at 384 px), weights from
yodharinmonplub/rsna-knee-fleet-weights version 2.
Kaggle: yodharinmonplub/rsna-knee-model89-model75-alone v1. Dummy run verified (3/3 studies, 1 checkpoint); SUBMITTED
2026-09-27 ~14:30 (+07), submission 56602006.
SCORE: public 0.939 - model75 ALONE beats the 4-member fleet (model88 0.938) and the 3-member fleet (model81 0.933).
Full training of CoAtNet-2 at 384 px is this project's strongest own-model recipe; the EfficientNet/ResNet/256 px members
slightly dilute it. Gold (0.9204) understated it: gold ranks it barely above model72 (0.9143), the leaderboard by ~0.006+.
