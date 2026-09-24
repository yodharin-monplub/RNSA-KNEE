Model31: full-data refit of model28 (native384-v8 fine-tune).

Recipe: model28's settings (v8_full_swa start, s44 6%-94% 384px cache, last stage + head, v4
targets, teacher 0.3), trained on train + weak_val for a FIXED 2 epochs (model28's selected
epoch), seed 20260920, no checkpoint selection. Gold58 excluded from training.
Weak BCE on weak_val (in training, optimistic) 0.370144. Gold development AUC 0.917440
(model28 0.915729, v8 start 0.911638). Descriptive only.
Cloud: instance 51266662 (RTX3090 $0.172/h), 57 min, estimated $0.163; observed credit delta for
concurrent model31+model32 $0.476 (booked here). Watchdog backed up and destroyed.
Private dataset: yodharinmonplub/rsna-knee-model31-weights. Planned use: model33 v8 views.