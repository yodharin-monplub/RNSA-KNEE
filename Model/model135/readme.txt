Model135 (TPU training, planned): A/B of INPUT DENSITY for the pseudo-label student. Exactly model113's recipe
(CoAtNet-2 384 rmlp, full data = all 4,345 non-gold studies, 18 epochs, last epoch, seed 20261017, targets
data/llm_relabel/targets_v1_oof110_35.csv = 0.35 x v1 Gemini labels + 0.65 x model110's 5-fold OOF) but on the
128-FRAME cache: 32/24/24/24/24 slices per series at 336 px (krittanutsomtuas/rsna-knee-dense-cache-336-part{0..3}-s128,
4,407 studies, 276 shards, 45.3 GB, 0 failed slices; built 2026-10-04 with
Other/scripts/run_dense_cache_part.py --part 0..3 --size 336 --slots 128) instead of 96 (24/18/18/18/18).
Why: the forum reports 24 -> 32 cached slices per series beat any label work; our own s64 -> s96 step (model75 0.939 ->
model104 0.942 LB) and EfficientNet s64 -> s128 (+0.002 LB) point the same way. Training samples 16 windows per study,
so training costs about the same as s96; evaluation/inference use 126 windows (x1.34 of s96).
Session: TPU session 4 on the teammate account, after model113 in the same kernel:
  python scripts/run_tpu_train.py --name model113 --plan full --cache s96 --account krittanutsomtuas --then model135 --then-cache s128 --then-ratio 1.15
(the s96 cache is freed from /dev/shm before the s128 cache is unpacked).
Config: Model/model135/config.json (differs from model113's only in its description).
In the fleet leg: geometry 's128x336' of fleet leg v5 (Other/scripts/fleet_leg_v5.py; build with run_speedy_fleet_v2.py --leg v5 and a fleet entry like ('model135', ['model135_best.pt'], 's128x336', 2)); ~23 min per member on the hidden set at full windows, ~11.5 at stride 2.
Score: pending.
Reader parity 2026-10-04: the leg v5 reader at 's128x336' rebuilds all 58 gold studies byte-identical to this cache (krittanutsomtuas/rsna-knee-cache-vs-leg-s128).
Pushed 2026-10-04 16:07 on the teammate account: krittanutsomtuas/rsna-knee-tpu-train-model113-then-model135 (model113 first, then model135 on the 128-frame cache).
