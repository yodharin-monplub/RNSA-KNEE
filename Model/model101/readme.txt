Model101: model100 ALONE on the leaderboard - fleet-only notebook (no public models) with one member, model100
(CoAtNet-2 at 384 px trained from ImageNet with model75's recipe on the DENSE 96-frame 336 px cache, read at the 's96'
geometry: 24/18/18/18/18 slices per slot over 2-98% of each series, 140 mm crop).
model100 was stopped after 12 of 18 epochs when the shared Vast credit ran out; its selected epoch 11 (weak-val BCE
0.3115, gold AUC 0.9231) already matches model75 (0.3106 / 0.9204, LB 0.939 alone as model89).
Purpose: measure whether the denser cache helps the CoAtNet-2 recipe on the hidden set, as it did for EfficientNet-B0
(model92 0.920 -> model91 0.922). Decides whether the TPU / Vast runs that follow train on the 96-frame cache.
Built with Other/scripts/run_fleet_only.py --name model101 --fleet "[('model100', ['model100_best.pt'], 's96')]".
Weights: model100_best.pt in the private dataset yodharinmonplub/rsna-knee-fleet-weights
(sha256 07e6c2cccefac9b9511493aa2689a8550398184807afef3fa92e21d5cca92a79).
Dummy run VERIFIED 2026-09-29 05:17 (model100_best.pt sha256 07e6c2cc..., epoch 11, res 384, 3/3 studies read at s96,
19 s on 2 GPUs, no errors). Submit after the 07:00 quota reset.
SUBMITTED 2026-09-29 07:05 (+07), submission 56656173.
Score: public 0.937 (scored 2026-09-29 07:34) vs model75 alone 0.939 (model89): -0.002. Confounded - model100 stopped at
epoch 12 of 18 (cosine LR never annealed) while model75 trained fully (25 epochs, selected 16); its higher gold (0.9231 vs
0.9204) did not carry to the LB. The density question for CoAtNet-2 needs a fully trained 96-frame run.
