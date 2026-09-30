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
  2026-09-29 16:23-18:13  FIRST FULL TPU RUN (v2, from the batch queue): smoke passed; full cache unpack 737 s;
                    epoch 1 1,559 s (XLA compilation), then ~285-300 s per epoch INCLUDING evaluation (~10x a 3090).
                    Crashed in epoch 14: one chip process was killed abruptly (BrokenProcessPool, no Python traceback -
                    probably the host OOM killer with 8 chips x ~10 resident loader workers). best.pt = epoch 13.
                    Quality problem: weak-val BCE 0.374 at epoch 13 vs 0.316 for model75 (GPU, same recipe) at epoch
                    13; gold similar (0.907 vs 0.910); gold predictions correlate 0.93-0.97 with model75's but several
                    labels are shifted (less converged). Suspected cause: bf16 autocast on XLA also runs BatchNorm /
                    LayerNorm in bf16 (CUDA fp16 autocast keeps their statistics in fp32).
                    Fixes (2026-09-29 evening): fp32 by default (config xla_bf16 restores bf16), per-epoch last.pt +
                    --resume (the kernel resumes a crashed run up to twice), eval loader workers not resident, host
                    free-memory logging per epoch. Epoch-13 weights kept in Other/artifacts/remote_model98 (not used).
  2026-09-30 05:06-12:40  v3 (fp32, resume-capable) COMPLETED: 17 epochs (plan shrank 18 -> 17 for the time limit),
                    selected epoch 16 (plain weak-val BCE 0.3700), gold AUC 0.9154 (epoch 13 had 0.9207); best.pt sha256
                    378563bb45129968a15e075ea5fdaccd3d163608b56781367d896fb50367353d, added to rsna-knee-fleet-weights.
                    Epochs 1-13 took ~305 s each (fp32 as fast as bf16), but epochs 14-17 took 4,500-6,800 s each and
                    free host memory fell 137 -> 94 GB at epoch 14 (the bf16 run crashed at the same epoch) - cause
                    unknown yet; per-epoch XLA compile counts and device memory are now logged.
                    METRIC MISMATCH found: train_model16 reports a CONFIDENCE-WEIGHTED weak-val BCE (weights 2|t-0.5|),
                    train_xla used a plain mean, which reads ~0.06 higher for the same quality - the "quality gap" to
                    the GPU runs was this, not bf16. Gold AUC (comparable) is on par with model75 (0.92). train_xla now
                    uses the weighted formula.
Score: pending (LB test planned after the Kaggle GPU quota refresh on 2026-10-03).
