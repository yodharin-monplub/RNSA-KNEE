Model90 (prepared, not launched): model75's recipe - CoAtNet-2 at 384 px trained from ImageNet-12k/1k weights, all
layers, encoder LR 2e-5 / head LR 1e-3, warmup 10%, gradient checkpointing, GPU augmentation, plain soft-label BCE on the
Gemini v1 labels, 16 windows/study, batch 8, cosine, epoch chosen by weak-validation BCE - with a NEW SEED (20261009) and
18 instead of 25 epochs (model75's weak-val BCE plateaued from epoch 13 and bottomed at epoch 16, so a shorter cosine
should reach the same quality for ~30% less).
Why: model75 alone scored 0.939 public (model89) - this project's best own model - while a fleet of weaker members
dilutes it (0.938). A seed ensemble of CoAtNet-2 384 models is the most direct way to a stronger own model; a final-level
blend into model76 only pays once our own side is close to ~0.943 (model80's fit, EXPERIMENTS.md).
Files: config.json, run_model90.sh, model90_budget.json (cap 2.5 USD, ~9 h, 24 GB card), model90_ready.py, train_model16.py.
Launch blockers on 2026-09-27: (1) Vast credit ~1.6 USD < the 2.5 USD cap the launcher requires, shared with another
project's instance - top-up requested; (2) the self-imposed campaign rail in cloud/model16_budget.json (25 USD, recorded
spend 23.85) must be raised when the top-up arrives (it has been raised twice before with the user informed).
Score: pending.
