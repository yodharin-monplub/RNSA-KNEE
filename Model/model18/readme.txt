Model18: fixed equal probability blend of model10, model12, model15.
Motivation: model10 public0.929 leads the campaign despite lower gold development AUC. Test complementary checkpoint predictions without tuning blend weights.
Weights one-third each, fixed before inspecting this blend development results or public score.
Each branch embeds its validated image preprocessing/inference source verbatim; sequential subprocesses release GPU memory.
Public CoAtNet checkpoints by Johnathan Wagner / Dread Development, CC0 Kaggle datasets raptor-knee-finespacing, raptor-knee-maxspan, raptor-knee-native384dense.
No public score yet. Private GPU inference validation pending.

Development macro AUC0.922287 across58 cases; not untouched and not a public score.
Local wrapper integration using real component predictions and AST embedded-source identity checks passed. Private T4 version1 accepted.

Private GPU version1 COMPLETE: 90.46 seconds,3 visible studies; exact three-way probabilities verified. Ready for leaderboard submission; no public score.

Public submission56158870 version1 scored0.933. Model19, which adds the
fine-tuned model16 branch, scored0.935; model18 remains useful comparison
evidence rather than the campaign best.
