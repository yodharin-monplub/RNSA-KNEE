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
Score: pending.
