Model27: full-data refit of model16's max-span recipe.

Recipe: scripts/train_model16.py with model27/config.json: v5_full_swa max-span start, last stage
+ head trainable, v4 targets, teacher 0.3, 12 windows, batch 2, LR 5e-6/1e-5 (model16 values),
seed 20260918, but trained on train + weak_val (all non-gold studies, 4,345) for a FIXED single
epoch, the epoch model16 selected by weak-validation BCE. No checkpoint selection (select=last).
Gold58 excluded from training and used descriptively only.

Cloud: instance 51257213 (RTX3090, $0.172/h, fast host), 49 min wall, estimated compute $0.139
(credit delta not separable: model28 ran concurrently). Watchdog backed up and destroyed.
Weak BCE on weak_val (now in training, so optimistic) 0.366590. Gold development AUC 0.924035
(model16 0.923080, v5 start 0.919929). Checkpoint sha256 77135ccf...bc97ee.
Private dataset: yodharinmonplub/rsna-knee-model27-weights. Planned use: model29 maxspan forward arm.