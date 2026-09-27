Model82 (prepared, not yet run): model72's recipe (EfficientNet-B0 at 288 px from ImageNet, all layers, GPU
augmentation, plain soft-label BCE on the Gemini v1 labels, 16 windows/study, batch 8, cosine, 25 epochs, epoch by
weak-validation BCE) as 5 report-group folds (scripts/train_model16.py --fold k), like model71 did for ResNet34.
Why: forum teams report 5-fold EfficientNet-B0 ensembles at 0.94+ public on their own, and folds bought +0.0034 gold
for model71. model82 is a stronger member for the fleet leg (model80/81) and gives out-of-fold predictions.
If model84 shows that full-training pseudo-labels help, model82 switches to those targets before launch.
Files: config.json (Other/model82/config.json, seed 20261008), Other/cloud/run_model82.sh (folds resume after a
restart), cloud/model82_budget.json (cap 2.5 USD, ~9-10 h, ~1.5-2 USD), cloud/model82_ready.py (gate passed
2026-09-27). Launch after model75 (one Vast instance at a time).
Score: pending.
