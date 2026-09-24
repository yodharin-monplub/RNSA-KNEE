Model28: weak-label fine-tune of the public native384-v8 CoAtNet (last untuned Raptor arm).

Start: dreaddevelopment/raptor-knee-native384 raptor_ft_coatnet_v8_full_swa.pt
(sha256 3db77388...1c21862). Geometry = public arm: 44 slots (Sag FS 12, Sag 10, Cor FS 8,
Cor 6, Ax 8), 6%-94% span, 140mm crop, 384px, all valid windows (42).
New private s44 cache (44 frames stored at the front of 64-frame arrays, rest zero/masked),
4 Kaggle CPU parts, 20.46 GB. Parity notebook with the public Raptor reader/eval code verbatim:
max pixel diff 0, max probability diff 0.0 on 58 development cases (artifacts/v8_parity_notebook).

Recipe: model16 config (last stage + head, v4 targets, teacher 0.3, 12 windows, 2 epochs,
LR 5e-6/1e-5), seed 20260919; selection by weak-validation BCE including epoch0.
Weak BCE: epoch0 0.464822, epoch1 0.372541, epoch2 0.370465 (selected).
Gold development AUC: 0.911638 (start) -> 0.915729 (selected). Descriptive only.
Cloud: instance 51259907 (RTX3090 $0.172/h), 57 min, estimated compute $0.164; observed credit
delta for model27+model28 together $0.548 (booked here). Watchdog backed up and destroyed.
Private dataset: yodharinmonplub/rsna-knee-model28-weights. Planned use: model29 v8 arm.