Model98 (in progress): a SECOND CoAtNet-2 at 384 px trained from ImageNet with model75's recipe (model75 = 0.939 public
alone, our best own model), new seed 20261011, 18 epochs (model75's weak-val BCE plateaued from epoch 13), trained FOR
FREE on a Kaggle TPU v5e-8 (the user allowed the TPU quota on 2026-09-28; Vast credit is 0).
Trainer: Other/scripts/train_xla.py (PyTorch/XLA port of train_model16.py's from-ImageNet path: one process per chip,
per-chip batch 1 study x 16 windows = model75's global batch 8, bf16 autocast, gradients all-reduced before clipping,
augmentation on the host CPU, evaluation padded to 62 windows with a masked attention softmax, rank-0 selection by
weak-val BCE, same checkpoint layout). Kernel builder: Other/scripts/run_tpu_train.py; code + splits + targets in the
private dataset yodharinmonplub/rsna-knee-trainer-kit (config_model98.json).
Status: smoke test (pilot cache) pushed 2026-09-28 00:21 as yodharinmonplub/rsna-knee-tpu-train-model98-smoke; the full
run follows once the smoke passes.
Score: pending.
