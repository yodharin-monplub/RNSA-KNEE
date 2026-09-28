Model98 (in progress): a SECOND CoAtNet-2 at 384 px trained from ImageNet with model75's recipe (model75 = 0.939 public
alone, our best own model), new seed 20261011, 18 epochs (model75 selected epoch 16 of 25 by weak-val BCE), trained FOR
FREE on a Kaggle TPU v5e-8 (the user allowed the TPU quota on 2026-09-28).

Files here (copies; the working versions live in Other/scripts):
  config.json            the recipe (model75's: gradient checkpointing, 16 windows/study, global batch 8, plain soft-label
                         BCE on targets_g38f_low, augmentation, encoder LR 2e-5 / head LR 1e-3, warmup 10%, cosine)
  train_xla.py           PyTorch/XLA port of train_model16.py's from-ImageNet path: one process per TPU chip, per-chip
                         batch 1 study x 16 windows (= model75's global batch 8), bf16 autocast, gradients all-reduced
                         before clipping, torch_xla gradient checkpointing, augmentation on the host CPU, evaluation
                         padded to 62 windows with a masked attention softmax, rank-0 selection by weak-val BCE, same
                         checkpoint layout as train_model16.py (loadable by the fleet leg as an 's64' member).
                         --stop-at: the chips agree (all_reduce MIN) on how many epochs fit before the session limit.
  tpu_common.py          model12's Classifier / windows / labels without cv2/pydicom.
  run_tpu_train.py       builds the Kaggle kernel (plans smoke, full, smoke+full, cpu-test).
  build_trainer_kit.py   rebuilds the private dataset yodharinmonplub/rsna-knee-trainer-kit (code + splits + targets).

How to reproduce:
  1. python Other/scripts/build_trainer_kit.py --push "notes"
  2. python Other/scripts/run_tpu_train.py --name model98 --plan smoke+full --push
  3. Kaggle API pushes do not get a TPU (kaggle-cli issue #1197): open the kernel
     yodharinmonplub/rsna-knee-tpu-train-model98 in the Kaggle editor, choose TPU v5e-8, then "Save Version" ->
     "Save & Run All". Outputs: /kaggle/working/model98/best.pt, history.json, metrics.json, gold_predictions.csv.
  4. Add best.pt to yodharinmonplub/rsna-knee-fleet-weights (Other/scripts/upload_fleet_weights.py --add) and use it as
     an 's64' fleet member (Other/scripts/run_speedy_fleet_v2.py / run_fleet_only.py).

History:
  2026-09-28 09:40  first TPU smoke run (started by the user in the web editor): setup fine (torch 2.8.0, torch_xla
                    2.8.0, 8 devices, timm 1.0.27, 377 GB RAM) but the 8-process launch failed: "TPU initialization
                    failed: Invalid --..._slice_builder_worker_addresses specified. Expected 8 worker addresses, got 1".
                    Cause: Kaggle's TPU image presets TPU_PROCESS_ADDRESSES=local and CLOUD_TPU_TASK_ID for a single
                    process, and torch_xla only fills them with setdefault. Fix: train_xla.py unsets both before the
                    launch. Also: finding the cache manifest with rglob over the competition's DICOM tree took 11 min;
                    now a depth-limited search that skips the competition folder.
  2026-09-28 10:10  added torch_xla gradient checkpointing (model75 used it; 16 windows of 384 px per 16 GB chip would
                    not fit without it); CPU check of the trainer (PJRT_DEVICE=CPU on a free Kaggle CPU kernel,
                    yodharinmonplub/rsna-knee-tpu-train-model98-cpu-test) pushed; it never produced output (queued or
                    stuck; the kernel-status API is rate-limited so its state is unknown).
  2026-09-28 11:31  smoke+full kernel yodharinmonplub/rsna-knee-tpu-train-model98 pushed; the user was asked to start
                    it from the web editor (API pushes get no TPU).
  2026-09-28 12:20  local logic test PASSED: train_xla.py with a CPU stand-in for torch_xla (world size 1, batch 1 x 2
                    windows, pilot cache, CoAtNet-2 from the HF hub): 2 smoke epochs with gradient checkpointing,
                    gradient reduce, epoch-plan all_reduce, all_gather evaluation, rendezvous, best.pt / history /
                    metrics / TRAINING_COMPLETE written; best.pt loads strictly into tpu_common.Classifier.
  2026-09-28 15:00  kaggle-cli #1197 (maintainer): API pushes without a pinned docker_image DO get a TPU but wait in
                    the batch TPU queue (1 active-or-queued batch TPU session per user). v1 is presumably queued;
                    the user may cancel it and start from the editor instead (faster start).
Score: pending.
