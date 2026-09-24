Model29: public Dinosaurs ensemble with every Raptor CoAtNet arm fine-tuned by this project.

Declared before model26/27/28 results (EXPERIMENTS.md). Raptor arms (weights unchanged,
renormalized over 1.15 as in model26):
- maxspan-v5 (0.60): model27_best.pt (full-data refit of model16's recipe)
- native384dense-v10 (0.10): model20_best.pt
- maxspan-v5-reverse (0.10): model16_best.pt
- native384-v8 (0.20): model28_best.pt
- finespacing-ft22 (0.15): model22_best.pt
CoAt resgated arm, transformer family and final blend unchanged from the public notebook.

Development AUCs of the substituted arms vs their public starts (58 cases, descriptive):
model27 0.9240 / model16 0.9231 (v5 0.9199); model20 0.9192 (v10 0.9170);
model28 0.9157 (v8 0.9116); model22 0.9158 (v9 0.9097).

Built by scripts/run_model23_24.py (name model29). Kaggle:
yodharinmonplub/rsna-knee-model29-dinosaurs-all-finetuned-arms, T4 x2, version 1 launched
2026-09-17 local. No score yet.
Submission 56291152 FAILED: 'Your submission notebook exceeded the allowed runtime.' The public per-arm Raptor loop re-decodes each arm; with five arms the hidden test exceeded the limit. Grouped decode (model30+) avoids this.
