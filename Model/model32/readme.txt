Model32: full-data refit of model20 (native384dense fine-tune).

Recipe: model20's settings (v10 native384dense start, 2%-98% 384px cache, last stage + head, v4
targets, teacher 0.3), trained on train + weak_val for a FIXED 1 epoch (model20's selected epoch),
seed 20260921, no checkpoint selection. Gold58 excluded from training.
Weak BCE on weak_val (in training, optimistic) 0.368905. Gold development AUC 0.918837
(model20 0.919209, v10 start 0.916975). Descriptive only.
Cloud: instance 51266707 (RTX3090 $0.176/h), 58 min, estimated $0.169 (credit delta booked under
model31). Watchdog backed up and destroyed.
Private dataset: yodharinmonplub/rsna-knee-model32-weights. Planned use: model33 dense views.