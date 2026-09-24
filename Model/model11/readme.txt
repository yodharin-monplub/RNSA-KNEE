Model11: fixed CoAtNet probability ensemble
Status: private offline Kaggle version 1 COMPLETE and validated, not submitted.
Five submissions already made in the current UTC day; await available quota.
Visible-case runtime: 33.15 seconds; peak GPU memory 4.00 GB. These three cases
do not establish hidden-test runtime. Branch max absolute differences versus
standalone outputs: model8 0.00016945, model9 0.00015664, model10 0.00011980.
Initial 1e-5 equivalence tolerance failed; branches pass 1e-3 tolerance suitable
for float16 inference. Small differences are recorded, not bitwise equivalence.
Blend arithmetic agrees within 1.68e-8. Details: output/validation.json and
runtime.json under artifacts/model11_notebook. No leaderboard gain claimed.
Weights fixed before model10's score is available:
  0.25 model8 (wide-dense single epoch)
  0.25 model9 (wide-dense averaged weights)
  0.50 model10 (fine-spacing checkpoint)
This balances the two training variants; no leaderboard weight optimization.
All checkpoints share model8's full-coverage inference geometry. Decode each
study once and reuse its CPU images. GPU forward passes run sequentially with
12-window microbatches while all three model weights remain resident.
Raw per-label probabilities are averaged. No test labels or reports are used.
Store branch predictions and runtime.json for verification and runtime tracking.
Check branch equivalence against standalone model8/9/10 visible-test outputs
before submitting. Score and hidden-test runtime are not yet known.
Credits: Johnathan Wagner / Dread Development; CC0 public checkpoint weights.
Source details and geometry limitations: SOURCES.md, model8/9/10/readme.txt.
No new training or Vast.ai rental. Run: scripts/run_model11.py.

Gold development evaluation (2026-09-11): macro AUC 0.917394.
58 cases; checkpoint authors used these during development, so this is not an
untouched holdout. Paired uncertainty: artifacts/coatnet_gold_notebook/output/analysis.json.
