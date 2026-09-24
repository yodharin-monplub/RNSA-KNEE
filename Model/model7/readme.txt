Model 7: public CoAtNet checkpoint reproduction
Status: competition submission 56140057 COMPLETE; public AUC 0.924,
verified via Kaggle API on 2026-09-10. Not trained by this project.
Checkpoint author: Johnathan Wagner / Dread Development.
Weights: dreaddevelopment/raptor-knee-widedense, raptor_ft_coatnet_v4_full.pt.
Our reproduction matches the publisher's reported 0.924 public AUC.
The checkpoint is CC0; architecture/preprocessing attribution is in SOURCES.md.
Inference follows the published five-slot, 64-slice, wide-span geometry and
label-specific attention pooling. We use safe tensor-only checkpoint loading,
bounded backbone microbatches, raw probabilities, and explicit failure checks.
No local gold score is claimed for third-party weights; their development
history cannot be independently established from the release alone.
Private Kaggle run: scripts/run_model7.py (explicit T4, internet disabled).
