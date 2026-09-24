Model46: self-distillation - maxspan recipe trained on targets blending labels and image predictions.

Targets (scripts/make_distill_targets.py): equal average of the Gemini relabels and the held-out
fold-teacher predictions (model44/model45) for all 4,345 non-gold studies; the 50/50 weight was fixed
in advance. Motivation: on the 58 gold cases, image+label blends agree with gold much better (0.936)
than labels alone (0.893) or images alone (0.927).

Result: CV 0.923974 versus 0.925370 for model38 (same recipe, labels only). No gain, so the
self-distilled checkpoint is not used in any ensemble. Cost ~0.3 USD.
Interpretation: the teacher was fine-tuned from a checkpoint that had already seen all studies, so its
predictions add little information beyond the labels, and averaging them softens the targets.