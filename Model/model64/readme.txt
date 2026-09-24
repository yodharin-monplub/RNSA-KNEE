Model64: Arm B of the external-data test: model63 (same seed, same augmentation) plus auxiliary supervision from MRNet (train split),
fastMRI+ (exam labels and slice boxes as window labels) and KneeMRI through separate heads (knee/external.py), loss weight 0.5,
one external batch per step. External labels never touch the 12 competition heads. Compare with model63 to isolate the external data.
Licences: research-use / non-commercial, accepted by the account owner; caches are private Kaggle datasets, never redistributed.
