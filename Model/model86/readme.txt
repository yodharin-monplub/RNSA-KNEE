Model86 (prepared, not yet run): the SLICE-DENSITY test. model72's recipe (EfficientNet-B0 from ImageNet, all layers,
GPU augmentation, plain soft-label BCE on Gemini v1, 16 windows/study, batch 8, cosine, 25 epochs, epoch by
weak-validation BCE, same seed 20261003) trained on a DENSER cache:
  128 frames per study, slots 32/24/24/24/24 (sagittal fluid / sagittal other / coronal fluid / coronal other / axial),
  i.e. 24-32 slices per series, at 256 px - instead of 64 frames (18/14/12/8/12) at 336 px; model res 256 (was 288).

Why: our own fleet scores 0.933 public alone (model81), while forum teams report 5-fold EfficientNet-B0s at 0.94+ public
that read 24-32 slices per series, and one team found 24 -> 32 cached slices per series worth more than any label work.
Our 64-frame layout keeps only 8-18 slices per series (8 for coronal T1). Speedy's Raptor stage also reads ~96 slices.

Cache: built free on Kaggle CPU by scripts/run_dense_cache_part.py --size 256 --slots 128 --part 0..3 (kernels
yodharinmonplub/rsna-knee-dense-cache-256-part{0..3}-s128, pushed 2026-09-27; same model12 reader as the 64-frame cache,
builder generalised to any frame count - build_dense_cache_part.py). Training side (tested on a synthetic pilot cache):
prepare_dense_cache.py / knee/dense.py / train_model16.py accept --size/--cache-size 256 --slots 128;
download_dense.py maps model86 -> 'dense128'; cloud/dense128_ready.py gates the run on all four manifests.
Run: Other/cloud/run_model86.sh, budget cloud/model86_budget.json (cap 1.5 USD, 120 GB disk, ~2.5-3 h).
If it beats model72 (paired, same seed), the fleet leg needs the 128-frame reader for it and the next 5-fold runs move
to this cache.
Cache BUILT 2026-09-27: 4,407 studies, 276 shards, 26.93 GB compressed, 0 failed slices (35-46 min per part on free
Kaggle CPU); readiness gate cloud/model86_ready.py PASSES. Queued first after model75.
RESULT (2026-09-27 16:35): selected epoch 24 of 25, weak-validation BCE 0.3119 vs model72's 0.3174 - lower at EVERY epoch
from 15 on (the large weak-validation split favours the denser cache); gold 0.9102 vs model72 0.9143 (-0.0041, 95% CI
[-0.0195, +0.0108], P>0 0.29): noise level, with large per-finding swings (Medial Meniscus 0.903 vs 0.968, MCL 0.937 vs
0.961; Fracture 0.939 vs 0.856, Lateral OA 0.847 vs 0.828). Gold already understated model75 on the leaderboard, so the
leaderboard decides: model91 (model86 alone, fleet-only v2 with the 128-frame reader) vs model92 (model72 alone).
Ops: RTX 3090 0.18 USD/h, 3.19 h, ~0.57 USD; a ~35 min stall on Kaggle-API dataset downloads was fixed on the fly
(cloud/download_dense.py now skips the unused public checkpoint for this cache and reuses a verified backbone).
Weights added to yodharinmonplub/rsna-knee-fleet-weights (version 3) as model86_best.pt.
Score: see score.txt (leaderboard pending).
