Model13: standalone public DINOv3 fold ensemble
Status: private offline Kaggle version 1 COMPLETE. All five full checkpoints
load strictly; fold probability and rank-aggregation checks passed on 3 cases.
Visible-test inference script: 14.73 seconds, 0.585 GB peak GPU allocation.
These figures exclude dependency installation and are not hidden-test runtime.
58-case gold development evaluation COMPLETE: macro AUC 0.868556.
Inferior to model12 on this development set; not prioritized for submission.
Source: Mattia Angeli, Bend the Knee to DinoV3 (ensembled), Apache-2.0 notebook.
Five public m_f0.pt through m_f4.pt files from mattiaangeli/knee-mri-fold-weights.
Dataset is listed CC0; underlying Meta DINOv3 license remains applicable.
Geometry from the source: six plane/fat-suppression slots, 16 slices each,
130mm crop, 12%-88% band, 336px, per-slice percentile 1%-99% normalization.
Architecture and input normalization come from checkpoint configuration.
Uses strict complete state loading and tensor-only checkpoint deserialization.
One study at a time, all five fold models; float32 inference on T4.
Mean fold ranks with average treatment of ties. Per-fold raw probabilities saved.
Changes from source: no arbitrary 256-file cap, no silent whole-study fallback,
bounded per-study memory, strict loading instead of permitting missing encoders,
float32 instead of bf16 autocast, tie-aware ranks instead of ordinal tie breaking.
No leaderboard-tuned parent blends or RadImageNet branches are included.
This project did not train these checkpoints. Training overlap with the 58 gold
cases must not be assumed absent; a gold score is development evidence only.
No new Vast.ai rental. See NOTICE.txt and SOURCES.md for attribution.
