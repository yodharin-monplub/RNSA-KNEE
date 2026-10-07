Model141 (TPU training, planned): DIVERSITY pseudo-label student with a different backbone family.
Recipe = model138's (96-frame 336 px cache, full data = all 4,345 non-gold studies, 18 epochs, last epoch, encoder lr
2e-5 / head lr 1e-3, 16 windows per study, gradient checkpointing; targets data/llm_relabel/targets_v1_oof110_25.csv =
0.25 x v1 Gemini labels + 0.75 x model110's 5-fold OOF) with backbone ConvNeXt-Small ImageNet-22k -> 1k at 384 px
(timm convnext_small.fb_in22k_ft_in1k_384, 49.5M params, ~half the FLOPs of CoAtNet-2 -> cheaper leg members), seed 20261024.
Backbone weights: private datasets yodharinmonplub/ and krittanutsomtuas/rsna-knee-backbone-convnext-s
(unmodified timm weights from huggingface.co/timm, sha256 603aaa31...; uploaded with Other/scripts/upload_backbone.py).
Why: on the gold studies, mixing models that differ from the students (plain-label CoAtNets) added more than more
CoAtNet students (students share one teacher); another architecture should diversify further.
Plan: CPU mechanics test first (run_tpu_train.py --name model140 --plan cpu-test --account krittanutsomtuas), then a
main-account TPU session after model138/139: run_tpu_train.py --name model140 --plan smoke+full --cache s96
--account yodharinmonplub --then model141 --then-ratio 1.0 (the smoke run catches TPU compile problems early).
run_tpu_train.py now attaches each config's own backbone dataset and copies its weights (2026-10-06).
Config: Model/model141/config.json (differs from model138's in seed, arch and backbone fields).
Score: pending.
2026-10-06: CPU mechanics test of model140 passed (2 smoke epochs on the pilot cache, loss 0.693 -> 0.532). Queued as main-account TPU session C after model138/139.
2026-10-06: model141 deferred (the 09:50 main session pairs model140 with model138 instead).
