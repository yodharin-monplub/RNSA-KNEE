Model92: model72 ALONE - the control for model91 (slice-density test). Same code path as model91 (scripts/run_fleet_only.py,
fleet leg v2 in fleet-only form) but the member is read at 's64' = 64 frames (18/14/12/8/12) @ 336 px, the geometry model72
was trained on, windows at 288 px. model72 = EfficientNet-B0 from ImageNet, gold 0.9143, weak-val BCE 0.3174.
model91 - model92 on the public leaderboard = the effect of the denser cache for the same recipe and seed.
Build: python Other/scripts/run_fleet_only.py --name model92 --title "RSNA Knee Model92 Model72 Alone"
--fleet "[('model72', ['model72_best.pt'], 's64')]" [--push].
Kaggle: yodharinmonplub/rsna-knee-model92-model72-alone v1 pushed 2026-09-27 ~16:55 (dummy run); submit after 07:00.
Dummy run VERIFIED 2026-09-27 ~18:30 (from outputs): model72 read at 's64', 3/3 studies, 1 checkpoint, 14 s. Submit
after 07:00.
Score: pending.
