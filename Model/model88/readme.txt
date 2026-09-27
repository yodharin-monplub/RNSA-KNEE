Model88: this project's fully trained fleet ALONE, now with model75 - a leaderboard calibration like model81 (0.933).
fleet = mean probability of model72 (EfficientNet-B0 288 px, gold 0.9143), model71 (ResNet34 256 px, 5 folds, 0.9138),
model73 (CoAtNet-2 256 px, 0.9122) and model75 (CoAtNet-2 384 px, 0.9204); fleet gold 0.9265 (model81's fleet 0.9258).
Why: model80 blends the fleet into Speedy; if the fleet helps there, the next submission (model78) uses this 4-member
fleet, and model88 says whether model75 lifts the fleet itself on the leaderboard.
Code: model81's cell (run_model81.py --name model88 --fleet ...), identical except for the member list; weights from
yodharinmonplub/rsna-knee-fleet-weights version 2 (model75_best.pt added 2026-09-27).
Kaggle: yodharinmonplub/rsna-knee-model88-fleet4-with-model75 v1. Dummy run verified 2026-09-27 13:42 (3/3 studies, 8
checkpoints, 4 members, 46 s on 2 GPUs); SUBMITTED 13:43 (+07), submission 56600803.
SCORE: public 0.938 - +0.005 over model81 (0.933, the same fleet without model75). One CoAtNet-2 at 384 px as a quarter of
the fleet moved the leaderboard far more than gold predicted (fleet gold 0.9258 -> 0.9265): model75 is much stronger on the
hidden set than on the 58 gold cases. Next: model75's weight in the fleet, model75 inside Speedy (fleet leg / Raptor view),
and more full-training CoAtNet-2 at 384 px.
