Model78 (built, not pushed yet): model80 with the 4-member fleet - model72 (EfficientNet-B0 288 px), model71 (ResNet34
256 px x5 folds), model73 (CoAtNet-2 256 px) and model75 (CoAtNet-2 384 px) - blended into model76 (Speedy + our model61
Raptor weights, public 0.945) in rank space at w = 0.30, the same weight as model80.
Why: the fleet alone improved from 0.933 (model81, without model75) to 0.938 (model88, with it), so if a fleet leg helps
Speedy at all, this one should help more. Everything else is model80 (final cell after Speedy's submission.csv, model12
reader geometry, both GPUs, fail-safe to Speedy's file, 9 h process-age guard); one line of the leg differs (member list).
Build: python Other/scripts/run_model80.py --name model78 --title "RSNA Knee Model78 Speedy Our Heads Fleet4"
--with-model75 [--push]. Weights: yodharinmonplub/rsna-knee-fleet-weights version 2.
Pushed 2026-09-27 after model80 scored 0.944 (fleet3 at w 0.30 = -0.001): same weight, stronger fleet, so model78 - model80
isolates the fleet change. Dummy run verified (our model61 Raptor weights, no public Raptor loads, fleet leg 3/3 studies,
8 checkpoints, 44 s); SUBMITTED ~15:25 (+07), submission 56603186. Prediction from model80's fit: ~0.9455.
SCORE: public 0.946 - NEW PROJECT BEST (+0.001 over model76; model80 with the 3-member fleet at the same weight = 0.944).
Adding model75 to the fleet (fleet alone 0.933 -> 0.938) turned the final-level leg from -0.001 into +0.001: the fleet leg
pays once the own side is strong enough, as the 0.946 public d4-blend shows with its private fleet. Rank 127 of 4,441.
