Model75: model73's CoAtNet-2 full-training recipe at 384 px - the backbone's native size and the resolution our
best fine-tuned CoAtNet arms (gold 0.925) run at - instead of 256 px.

Recipe (config.json): coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k from ImageNet-12k/1k weights, every layer trained,
encoder LR 2e-5 / head LR 1e-3, 10% warmup then cosine, gradient checkpointing, chunk 16, GPU augmentation, plain
soft-label BCE on the Gemini v1 labels, 16 windows per study, batch 8, 25 epochs, epoch chosen by weak-validation
BCE (gold is never used for training or selection). At 384 px no position-bias rebuild is needed.

Why: model73 at 256 px reached gold 0.9122 and added +0.0018 to the blend the notebook's way - just under the
+0.002 bar. A fully trained CoAtNet is the only new model family that adds to the blend at all; the question is
whether native resolution lifts it clearly over. Pre-declared test: fifth Raptor-side view at the model77 share
(18.75% of the Raptor side), blended the notebook's way (probability mean), bar +0.002.

Run: Other/cloud/run_model75.sh on one Vast RTX 3090 (0.174 USD/h, cap 3.5 USD), launched 2026-09-27 00:38 after
one broken host (CDI GPU error) was abandoned. Expected ~12 h.
Build/train: python Other/scripts/train_model16.py --cache data/dense --config model75/config.json --checkpoint none
RESULT (2026-09-27 13:13): selected epoch 16 of 25 (weak-val BCE 0.3106), gold 0.9204 - the best single fully trained
model of the project (model73 at 256 px 0.9122, model72 EfficientNet-B0 0.9143). Per-epoch gold 0.707 (e1) -> 0.881 (e4)
-> 0.906 (e8) -> 0.920 (e16) -> 0.914 (e25). 11.85 h on one RTX 3090, ~2.56 USD; backed up and destroyed by the guard.
Pre-declared fifth-Raptor-view test (w 0.30, notebook-way probability mean): +0.0017 (P>0 0.79) -> NOT adopted as a view
(bar +0.002; model73 gave +0.0018). As a FLEET member it helps: fleet 72+71+73+75 alone 0.9265 vs 0.9258, final rank
blend +0.0016/+0.0020/+0.0022 at w 0.25/0.35/0.45 (fleet3: +0.0013/+0.0012/+0.0015). Weights added to the private dataset
yodharinmonplub/rsna-knee-fleet-weights (version 2) for the next fleet submissions (run_model80.py --with-model75).
Score: see score.txt (gold only; no leaderboard submission of model75 alone).
