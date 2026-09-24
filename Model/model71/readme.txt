Model71: model65's ResNet34 recipe trained as 5 report-group folds (the forum top teams' recipe: small ImageNet
backbone, all layers trained, GPU augmentation, plain soft-label BCE on the Gemini v1 labels, 256 px, 16 windows
per study, batch 8, cosine schedule, fixed 25 epochs). Each fold trains on 4/5 of the non-gold studies, split by
report hash so duplicate reports stay on one side, and predicts its held-out fifth.

Built by: python Other/scripts/run_model23_24.py is NOT used here - training is Other/scripts/train_model16.py
--config model71/config.json --fold k (k = 0..4), driven by Other/cloud/run_model71.sh on one Vast rental.
config.json in this folder is the exact config; the k-fold split lives in train_model16.py (digest % n_folds).

Results (58 gold cases, never trained on, descriptive only):
  fold0 0.9031 | fold1 0.9024 | fold2 0.9023 | fold3 0.8937 | fold4 0.9065
  5-fold ensemble 0.9138   (model65, one model on all the data, 0.9104; a fine-tuned CoAtNet arm 0.925)
  As an extra arm in the model61 blend at the pre-declared weight 0.30: 0.9312 -> 0.9323 (+0.0011,
  95% CI [-0.0039, +0.0061], P>0 0.69) => NOT adopted. model67 already showed a single ResNet34 arm
  costs 0.001 on the public LB (0.942 vs 0.943).

The point of the run is the OUT-OF-FOLD predictions: 4,345 of the 4,349 non-gold studies, each predicted by a
fold that never saw it (the 4 partition-excluded studies keep their v1 labels). Mixed 50/50 with the Gemini v1
labels they agree with gold at 0.9316, against 0.9236 for the two half-data models whose pseudo-labels failed on
2026-09-20 (blend 0.9326 -> 0.9251) and 0.9384 for the CoAtNet blend (which is in-sample and cannot be used).
Files: data/llm_relabel/oof_resnet34_5fold.csv, targets_v1_oof5f50.csv, targets_v1_oof5f25.csv (scratchpad
build_oof5.py, same checks as scripts/build_oof_targets.py). Head-retrain test: free CPU kernels
rsna-knee-head-retrain-oof5f50-{maxspan,dense,v8,fine}.

Cost and history: one Vast RTX 3090 at 0.219 USD/h, 13.07 h, 2.86 USD total for all five folds (~63 min each).
Three earlier launch attempts cost 0.24 USD: one died because cloud/download_dense.py did not know the run name,
two on an RTX A4000 whose host could not pull the PyTorch image. The rental then stopped on its own when the Vast
account hit 0 credit with all five folds finished but nothing backed up; it was restarted after a top-up and the
artifacts recovered intact.

Weights: fold checkpoints best.pt (5 x 82 MB) in Other/artifacts/remote_model71/model71/fold*/ and uploaded as the
private Kaggle dataset yodharinmonplub/rsna-knee-model71-weights. Gold and OOF predictions sit next to them.
