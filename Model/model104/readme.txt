Model104 (queued): model100's exact recipe FULLY trained on a Kaggle TPU v5e-8 of the TEAMMATE account
(krittanutsomtuas; its own 20 h/week TPU quota and its own batch TPU slot, so it runs in parallel with model98).
Recipe (config.json): CoAtNet-2 at 384 px from ImageNet-12k/1k (model75's settings: gradient checkpointing, 16 windows
per study, global batch 8, plain soft-label BCE on the Gemini v1 labels, augmentation, encoder LR 2e-5 / head LR 1e-3,
warmup 10%, cosine), on the 96-frame 336 px cache ('s96'), seed 20261012, 18 epochs, epoch by weak-val BCE.
Why: model100 (same recipe on Vast) was stopped at epoch 12 of 18 when the credit ran out and scored 0.937 alone
(model101) vs model75 0.939; a full run settles whether the denser cache helps CoAtNet-2, and adds a strong fleet member.

Files here: config.json, train_xla.py + xla_data.py + tpu_common.py (the TPU trainer), run_tpu_train.py (kernel builder:
--account krittanutsomtuas --cache s96 --plan smoke+full), build_trainer_kit.py (kit dataset; --account for the copy).
The teammate account holds its own copies of the private datasets rsna-knee-trainer-kit and rsna-knee-backbone-coatnet2
(new private datasets are not visible across accounts); it attaches yodharinmonplub's cache KERNELS
(rsna-knee-dense-cache-336-part{0..3}-s96), which is allowed.
Pushed 2026-09-29 07:42 as krittanutsomtuas/rsna-knee-tpu-train-model104 (batch TPU queue; API push, no click needed).
Hang guard: the smoke run is killed after 45 min, the full run 15 min after its --stop-at deadline.
2026-09-29 18:36: v2 pushed with the fixed trainer (fp32 instead of bf16 - the first full TPU run of model98 learned worse in
bf16 -, per-epoch last.pt + automatic resume after a crash, non-resident eval workers); replaces the queued v1.
RESULT (v2, 2026-09-30 05:10-13:25 on the teammate TPU, fp32): smoke OK; the 48 GB 96-frame cache unpacked in 664 s;
epochs 1-2 took 4,895 s and 4,200 s (cold page cache - the first passes read the cache from disk), epochs 3-13 ~336 s,
then epochs 14-15 took 5,832 s and 9,304 s (the same epoch-14 slowdown as model98, host free memory 144 -> 103 GB), so
the time guard cut the plan to 15 epochs. Selected epoch 15 (plain weak-val BCE 0.3647): GOLD AUC 0.9306 - the best of
any model this project has trained (model75 0.9204, model100 0.9231, model79 0.9242). best.pt sha256
4d2670fccd7c5f8e6aa081c71ecb25ed7d4ea1b9df35f30df14099debf88d67c, added to yodharinmonplub/rsna-knee-fleet-weights.
(v1, bf16, ran 22:30-04:30 on 2026-09-29/30 and produced no output - probably preempted.)
Score: pending (LB tests after the Kaggle GPU quota refresh on 2026-10-03).
