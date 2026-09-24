Model30: model29 plus reversed-slice views of two fine-tuned native384 arms, with a grouped
Raptor decode loop.

Changes versus model29 (declared before model29/model30 scores):
- add 'native384dense-v10-reverse' (model20_best.pt, reverse=True, w 0.05)
- add 'native384-v8-reverse' (model28_best.pt, reverse=True, w 0.10)
  (the public notebook's maxspan-v5-reverse view is the precedent)
- replace the Raptor main() with a grouped loop: arms sharing (img, slots, span, k_eval) are run
  on one decoded volume per study (the public loop re-decodes per arm). Per-arm inference,
  fallbacks, weights (renormalized, now over 1.30), rank output and file are unchanged.
Validation notebook yodharinmonplub/rsna-knee-model30-grouped-loop-check ran the grouped loop and
then the public loop (recording every infer_probs call): max abs per-arm probability diff 0.0.
Visible-test Raptor stage: 76 s for 7 views vs 67 s for model29's 5 arms (3 studies, includes
model loading); maxspan, dense and v8 each decode once for two views on the hidden test.

Built by scripts/run_model23_24.py (model30 / model30check). Kaggle:
yodharinmonplub/rsna-knee-model30-dinosaurs-grouped-reverse-views v1 COMPLETE, submitted 56292155.
Confirmed public AUC 0.941, submission 56292155, version 1. Ties models 24/26; scored before model29 despite being submitted 50 min later (grouped decode saves hidden-test runtime).
