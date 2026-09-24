Model 3: frozen DINOv2-small with attention pooling and nonlinear classifier
Status: trained on Vast.ai. Gold macro AUC: 0.749041 (58 held-out studies).
Local artifacts: artifacts/remote/model3/metrics.json and head.pt.
Uses the same extracted MRI features and fixed data partition as model2.
A learned attention layer pools the three slice groups separately within each
scan slot; a nonlinear head combines slots to predict twelve abnormalities.
Absent series are explicitly masked. Dropout and weight decay regularize the
head; weak validation BCE selects the checkpoint without consulting gold AUC.
Run: python -m knee.train_heads --config model3/config.json
This is a distinct trainable head, not a separately fine-tuned MRI backbone.
Sources and licenses: SOURCES.md. No leaderboard result has been claimed.
