Model23: exact reproduction of the public "Bend the Knee to the Dinosaurs" notebook.

Source: https://www.kaggle.com/code/mattiaangeli/bend-the-knee-to-the-dinosaurs
(Mattia Angeli; credits inside the notebook: Pilkwang, Sofia Anjenje, Antoine G.,
prvsiyan, Marwan Mahmoud, dreaddevelopment / Roman Tamrazov, renta.k, Anvith
Pothula, Evgen Dvorkin). Several public forks report public AUC 0.941.

Purpose: establish the 0.941 public baseline inside this account before testing
our fine-tuned CoAtNet checkpoints inside the same ensemble (model24).

Ensemble structure (unchanged):
- DINOv2/DINOv3/RadImageNet transformer family (20 fingerprint-checked DINOv2
  members plus RadImageNet heads), rank averaged.
- Raptor CoAtNet arm: maxspan-v5 0.60, native384dense-v10 0.10, maxspan-v5 on
  reversed windows 0.10, native384-v8 0.20 (probability mean, then rank).
- Mattia's CoAt resgated e4/e6/e8, rank-blended into the Raptor arm (alpha 0.4).
- Final per-finding rank blend: CoAtNet weight 0.60 (ACL/Lateral OA/Fracture
  0.75, Medial Meniscus 0.80, Lateral Meniscus 1.00).

Only change: private kernel id/title. Built by scripts/run_model23_24.py.
Kaggle: yodharinmonplub/rsna-knee-model23-public-dinosaurs-repro, T4 x2.
Version 1 launched 2026-09-16. No score yet.

Confirmed public AUC 0.940, submission 56282564, version 1 (public forks report 0.941). Model24 (fine-tuned CoAtNet swap) scored 0.941 on the same day: +0.001 at displayed precision.
