Model26: model24 plus a fifth Raptor CoAtNet arm from our fine-tuned model22.

Change versus model24: append arm "finespacing-ft22" = model22_best.pt
(weak-label fine-tune of dreaddevelopment raptor_ft_coatnet_v9_full, fine-spacing),
at the geometry used by our scored model10 (public 0.929): 64 slices, 6%-94% span,
140mm crop, 336px intermediate, 384px input, 62 windows. Arm weight 0.15 was fixed
before any model24/model26 score was available; arm weights are renormalized by the
notebook (0.6/0.1/0.1/0.2/0.15 over 1.15). Everything else identical to model24.

Motivation: model10 was the strongest standalone public checkpoint in this campaign
and is absent from the public ensemble, so it adds a third fine-tuned slice geometry.
model22 descriptive gold AUC 0.915761 vs 0.909689 for its starting checkpoint.

Built by scripts/run_model23_24.py (name model26). Kaggle:
yodharinmonplub/rsna-knee-model26-dinosaurs-five-arm-finetuned, T4 x2, version 1
launched 2026-09-17 local. No score yet.

Confirmed public AUC 0.941, submission 56285541, version 1 (scored after ~7.7h). Ties model24 at displayed precision; the extra fine-tuned fine-spacing arm did not change the rounded score.
