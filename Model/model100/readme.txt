Model100 (RUNNING on Vast since 2026-09-28 19:42): model75's CoAtNet-2 384 px recipe - full training from ImageNet-12k/1k
weights, encoder LR 2e-5 / head LR 1e-3, warmup 10%, gradient checkpointing, GPU augmentation, plain soft-label BCE on the
Gemini v1 labels, 16 windows per study, batch 8, cosine, epoch by weak-validation BCE - trained on the DENSE 96-frame cache
at 336 px (24/18/18/18/18 slices per slot) instead of 64 frames (18/14/12/8/12); new seed 20261012, 18 epochs.
Why: (1) model75 alone is our best own model (0.939 public, model89) and the fleet leg only paid once it joined (model78
0.946); (2) the denser cache helped the same EfficientNet-B0 recipe on the leaderboard (model92 0.920 -> model91 0.922).
Cache: built free on Kaggle CPU (scripts/run_dense_cache_part.py --size 336 --slots 96 --part 0..3 -> kernels
yodharinmonplub/rsna-knee-dense-cache-336-part{0..3}-s96, pushed 2026-09-28 ~07:35; ~8.6 GB compressed per part); the
pipeline (prepare_dense_cache.py / knee/dense.py / train_model16.py / download_dense.py) now accepts --slots 96 and was
tested on a synthetic pilot cache. Readiness gate: cloud/model100_ready.py (cloud/dense96_ready.py).
Run: Other/cloud/run_model100.sh; budget cloud/model100_budget.json (cap 2.8 USD, 120 GB disk, ~10 h).
Inference in the fleet leg needs a 96-frame 's96' reader (fleet leg v2 geometry table gets one entry).
LAUNCHED 2026-09-28 19:39 (deployed 19:42) on Vast instance 53178323 (RTX 3090, 0.18 USD/h, cap 2.8 USD, deadline
12.8 h) after model79 finished; readiness gate cloud/model100_ready.py passed at 17:3x. Guard: cloud/guard_once.py.
ATTEMPT 1 ABANDONED ~20:50: the host's RTX 3090 was power-capped at 160 W (~2x slower, ~16 h for 18 epochs);
destroyed and excluded. RELAUNCHED 20:52 on Vast 53189562 (RTX 3090 0.207 USD/h, 320 W - cloud/gpu_check.py OK),
deployed ~20:57; guard deadline 05:15 because the shared Vast credit (3.35 USD) runs out ~05:55 (top-up requested).
Score: pending.
