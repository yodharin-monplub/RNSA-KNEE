Model66: Full-training test, arm B: model65 (same seed) plus auxiliary supervision from MRNet, fastMRI+ (exam + box-window labels) and KneeMRI through separate heads
(knee/external.py, weight 0.5, one external batch per step). Tests whether external data helps when the whole backbone is trained - the regime where it can matter
(it was neutral in the last-stage fine-tune, model64). Private caches only; research-use licences accepted by the account owner.
