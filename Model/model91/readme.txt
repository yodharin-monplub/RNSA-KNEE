Model91: model86 ALONE - the leaderboard half of the slice-density test (paired with model92 = model72 alone).
model86 = EfficientNet-B0 trained from ImageNet with model72's exact recipe and seed, on the denser cache: 128 frames per
study (32/24/24/24/24 slices per slot) at 256 px instead of 64 frames (18/14/12/8/12) at 336 px. Weak-validation BCE
0.3119 vs model72's 0.3174 (better at every epoch from 15 on); gold 0.9102 vs 0.9143 (noise level).
Code: scripts/run_fleet_only.py (fleet leg v2 in fleet-only form): the study is read at the member's own geometry
('s128' = 128 frames @ 256 px, this project's model12 reader generalised), windows at 256 px, the probability is the
submission; unreadable studies get the mean (neutral fill). CPU-tested with mixed geometries.
Build: python Other/scripts/run_fleet_only.py --name model91 --title "RSNA Knee Model91 Model86 Alone Dense128"
--fleet "[('model86', ['model86_best.pt'], 's128')]" [--push]. Weights: yodharinmonplub/rsna-knee-fleet-weights v3.
Kaggle: yodharinmonplub/rsna-knee-model91-model86-alone-dense128 v1 pushed 2026-09-27 ~16:55 (dummy run); submit after
the daily quota resets (00:00 UTC = 07:00 Bangkok).
Dummy run VERIFIED 2026-09-27 ~18:28 (from outputs - the kernel-status API was rate-limited): model86 read at 's128',
3/3 studies, 1 checkpoint, 14 s on 2 GPUs. Submit after 07:00.
Score: pending.
