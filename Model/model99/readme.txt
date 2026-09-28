Model99 (built, not pushed): model78's recipe with model86 added to the fleet - model76 (Speedy + our model61 Raptor
weights) + a final-level rank leg at w 0.30 over fleet5 = model72, model71 x5, model73, model75 (all read at 's64' =
64 frames @ 336 px) and model86 (EfficientNet-B0 trained on the 128-frame cache, read at 's128' = 128 frames @ 256 px).
Built with Other/scripts/run_speedy_fleet_v2.py (fleet leg v2 = per-member reader geometry; CPU-tested with mixed
geometries).
Decision rule, fixed now: push only if model91 (model86 alone) beats model92 (model72 alone) on the public leaderboard,
i.e. the denser cache helps a paired model; otherwise model86 stays out of the fleet.
Rule met 2026-09-28 07:27 (model91 0.922 > model92 0.920): dummy run pushed; verified 10:14 (fleet leg blended at w 0.30,
3/3 dummy studies read by all 5 members, geometries s64 + s128, 50 s on 2 GPUs, valid submission). Submit with the
next quota (after 07:00 Bangkok on 2026-09-29).
Score: pending.
