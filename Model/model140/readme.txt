Model140 (TPU training, planned): DIVERSITY pseudo-label student with a different backbone family.
Recipe = model138's (96-frame 336 px cache, full data = all 4,345 non-gold studies, 18 epochs, last epoch, encoder lr
2e-5 / head lr 1e-3, 16 windows per study, gradient checkpointing; targets data/llm_relabel/targets_v1_oof110_25.csv =
0.25 x v1 Gemini labels + 0.75 x model110's 5-fold OOF) with backbone ConvNeXt-Small ImageNet-22k -> 1k at 384 px
(timm convnext_small.fb_in22k_ft_in1k_384, 49.5M params, ~half the FLOPs of CoAtNet-2 -> cheaper leg members), seed 20261023.
Backbone weights: private datasets yodharinmonplub/ and krittanutsomtuas/rsna-knee-backbone-convnext-s
(unmodified timm weights from huggingface.co/timm, sha256 603aaa31...; uploaded with Other/scripts/upload_backbone.py).
Why: on the gold studies, mixing models that differ from the students (plain-label CoAtNets) added more than more
CoAtNet students (students share one teacher); another architecture should diversify further.
Plan: CPU mechanics test first (run_tpu_train.py --name model140 --plan cpu-test --account krittanutsomtuas), then a
main-account TPU session after model138/139: run_tpu_train.py --name model140 --plan smoke+full --cache s96
--account yodharinmonplub --then model141 --then-ratio 1.0 (the smoke run catches TPU compile problems early).
run_tpu_train.py now attaches each config's own backbone dataset and copies its weights (2026-10-06).
Config: Model/model140/config.json (differs from model138's in seed, arch and backbone fields).
Score: pending.
2026-10-06: CPU mechanics test of model140 passed (2 smoke epochs on the pilot cache, loss 0.693 -> 0.532). Queued as main-account TPU session C after model138/139.
2026-10-06 09:50: pushed as main-account session yodharinmonplub/rsna-knee-tpu-train-model140-then-model138 (smoke+full; model138 follows in the same session).
TRAINED 2026-10-06 (main-account TPU, yodharinmonplub/rsna-knee-tpu-train-model140-then-model138, ran 15:31-20:22 after
~5.7 h queued; smoke OK; 18 epochs, ~323 s/epoch = 16% faster than CoAtNet-2): gold AUC 0.9267 at epoch 18 (mean of
epochs 14-18 0.9266) - clearly below the CoAtNet-2 students (0.933-0.935). Spearman vs model113 0.968 (more diverse than
another CoAtNet student, 0.988), but on gold adding it LOWERS the best ensemble (m113+m132+m112+m104 0.9375 -> 0.9356).
Possibly under-trained at CoAtNet's encoder lr 2e-5. best.pt sha256 19fdd5b1..., fleet weights model140_best.pt.
