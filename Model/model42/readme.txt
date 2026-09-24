Model42: CoAtNet-2 trained from ImageNet-12k weights on the Gemini relabels - ABANDONED (diverged).

Attempt1 failed the smoke test: coatnet_rmlp_2_rw_384 builds its relative-position attention bias for
384px inputs, so a 336px input gives 441 tokens against the stored 576 (cost ~0.04 USD).
Attempt2 ran at 384px, batch 2, grad checkpointing, LR 5e-5 backbone / 1e-3 head, 5% warmup + cosine,
same recipe that worked for ConvNeXt-small (model36). It did not learn: weak-validation BCE stayed at
0.4946 / 0.4990 / 0.4954 (base-rate level) and descriptive gold AUC fell 0.579 -> 0.538 -> 0.515 over
epochs 1-3. Stopped after epoch 4 to save budget (estimated 0.426 USD).

Interpretation: these settings are unstable for the CoAtNet hybrid (attention blocks in fp16 with a
1e-3 head LR); a much lower head LR and longer warmup would be needed. Not pursued because the
from-scratch family line had already shown no ensemble value: model36 (ConvNeXt, CV 0.892) lowered the
blend at every weight tested. Diversity from new families is not worth the budget here; the fine-tuned
public checkpoints remain far stronger.