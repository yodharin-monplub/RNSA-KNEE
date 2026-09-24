Model34: second-seed repeat of model27 (full-data maxspan refit).

Identical to model27 except seed 20260922: v5_full_swa start, last stage + head, v4 targets,
teacher 0.3, train + weak_val, fixed 1 epoch, no selection. Gold58 excluded from training.
Weak BCE on weak_val (in training, optimistic) 0.362464. Gold development AUC 0.923966
(model27 0.924035; probability average of model27+model34 0.923816). Descriptive only.
Cloud: instance 51270809 (RTX3090 $0.172/h), 38 min, estimated $0.110, observed credit delta
$0.181. Watchdog backed up and destroyed.
Private dataset: yodharinmonplub/rsna-knee-model34-weights. Planned use: model35 (seed split of
the maxspan forward view, declared in advance).