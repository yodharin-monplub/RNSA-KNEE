Model79 (prepared, not yet run): a FULL fine-tune of the public Raptor maxspan checkpoint
(dreaddevelopment/raptor-knee-maxspan, raptor_ft_coatnet_v5_full_swa.pt, CoAtNet-2 at 384 px) on the Gemini v1
labels - all four CoAtNet stages trainable, plain soft-label BCE, GPU augmentation, no teacher distillation,
10 epochs, encoder LR 1e-5 / head LR 1e-4 with 10% warmup and cosine decay, gradient checkpointing (config.json).

Why: every leaderboard gain this project has had came from better Raptor WEIGHTS, +0.001 per step (weak-label
fine-tune, Gemini-label fine-tune, head retrain). model38 - the maxspan arm in our 0.943 blend - was a very light
adaptation: only the LAST stage, 2 epochs, LR 5e-6, distillation toward the public model and confidence-weighted
BCE (gold 0.9254; after a head retrain on full bags 0.9308). model73/model75 showed our full-training recipe is
stable once the encoder LR is low. A genuine full adaptation of the strongest public init is the next step of that
ladder. The untouched public checkpoint is the epoch-0 candidate and the epoch is chosen by weak-val BCE, so a
harmful fine-tune cannot be selected.

Code change for this run: scripts/train_model16.py now turns on gradient checkpointing on the fine-tune path when
the config asks for it (it was only wired into the from-ImageNet path).
Run: Other/cloud/run_model79.sh (cloud/model79_budget.json: cheapest 24 GB card, cap 2.5 USD, ~5-7 h). AGENTS.md
allows one Vast instance at a time, so it launches after model75 finishes. Readiness gate cloud/model79_ready.py
passed on 2026-09-27.
Priority raised 2026-09-27: model76 (Speedy + our model61 Raptor weights) scored 0.945, so better Raptor weights are
worth +0.002 even inside the strongest public pipeline. model79 runs FIRST after model75.
Deployment rule, fixed before the result: if model79's selected epoch scores >= 0.9254 on the 58 gold cases (model38's
level - the maxspan weights that already gained on the LB), its checkpoint replaces the maxspan arm (forward and reverse
views) of model76: model87, built with Other/scripts/run_raptor_swap.py --swap maxspan=model79_best.pt --dataset
yodharinmonplub/rsna-knee-model79-weights. The leaderboard decides; below 0.9254 it is not submitted.
LAUNCHED 2026-09-28 07:12 on Vast instance 53065087 (RTX 3090, 0.174 USD/h, cap 2.5 USD, deadline 11.4 h) after the user
topped up Vast; the self-imposed campaign rail was raised 25 -> 35 USD (cloud/model16_budget.json, reason recorded).
RESULT 2026-09-28 18:55 (Vast 53065087, 12.4 h, ~2.17 USD; backed up and destroyed by the guard at 19:38):
epoch  weak-val BCE  gold AUC (58)
  0      0.44384      0.9199   (the untouched public maxspan checkpoint)
  1      0.28177      0.9212
  2      0.28061      0.9261
  3      0.27779      0.9223
  4      0.27691      0.9246
  5      0.27641      0.9213
  6      0.27677      0.9244
  7      0.27730      0.9213
  8      0.27624      0.9219
  9      0.27629      0.9205
 10      0.27449      0.9242   <- selected (lowest weak-val BCE)
The full fine-tune learns the Gemini labels far better than the public checkpoint (weak-val BCE 0.444 -> 0.274), but
gold stays at 0.920-0.926 - below model61's maxspan arm (0.9308 after its head retrain) and below the pre-declared
deployment bar of 0.9254 (model38). RULE NOT MET -> model87 (swap into model76) is NOT built or submitted.
Weights kept locally: Other/artifacts/remote_model79/model79/best.pt (sha in provenance / guard backup).
Score: gold 0.9242 (selected epoch 10); not submitted (below the 0.9254 deployment bar).
