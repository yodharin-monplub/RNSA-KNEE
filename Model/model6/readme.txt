Model 6: weak-validation-selected blend of frozen and fine-tuned MRI models
Status: evaluated. Gold macro AUC: 0.762663 (58 held-out studies).
Selected fine-tuned weight: 0.5. Local artifacts: artifacts/remote/model6/.
Base prediction is the fixed model4 average of model2 and model3.
Select fine-tuned model5 weight from {0, .25, .5, .75, 1} using confidence-weighted
soft-label BCE on weak validation only. Gold results never select the blend.
Stores selected weight in artifacts/ensemble_config.json. Two encoder passes
are required if both branches have nonzero weight.
See artifacts/model6/metrics.json for actual gold evaluation and model readmes
for preprocessing, training, source attribution, and validation limitations.
