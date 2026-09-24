Model24: model23 public ensemble with this project's fine-tuned CoAtNet weights.

Single-variable change versus model23: the Raptor CoAtNet arm loads
- model16_best.pt (our weak-label fine-tune of maxspan v5_full_swa; public AUC
  0.932 alone vs 0.928 for the starting checkpoint) for the maxspan-v5 arm and
  the maxspan-v5-reverse arm;
- model20_best.pt (our weak-label fine-tune of native384dense v10; development
  AUC 0.919209 vs 0.916975 for the starting checkpoint) for native384dense-v10.
Arm weights, geometry (64 slices, 2%-98%, 140mm, 336/384 px, 62 windows),
native384-v8, CoAt resgated arm, transformer family and final blend are unchanged.
Implemented by patching the executed Raptor namespace (ARMS file names and
find_weight_file) right before main(); the rest of the notebook is byte-identical.

Checkpoints share the RaptorClassifier parameter names (strict load) and carry
arch/res metadata. Private datasets: yodharinmonplub/rsna-knee-model16-weights,
yodharinmonplub/rsna-knee-model20-weights.

Built by scripts/run_model23_24.py. Kaggle:
yodharinmonplub/rsna-knee-model24-dinosaurs-finetuned-coatnet, T4 x2.
Version 1 launched 2026-09-16. No score yet.

Confirmed public AUC 0.941, submission 56282567, version 1 (2026-09-17 local). New campaign best (+0.006 over model19/21 0.935). Model23 baseline score pending for the paired comparison.
