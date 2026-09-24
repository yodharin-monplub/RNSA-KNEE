Model67: model61 (our best, public 0.943) plus one extra image arm from a different model family.

The extra arm is model65: a ResNet34 trained FULLY from ImageNet weights (all layers, 25 epochs, GPU augmentation, plain
soft-label BCE on the Gemini v1 labels, 256 px) - the first from-ImageNet run of this project that works (0.9104 on the 58
gold cases, 0.67 USD). It is weaker than a fine-tuned CoAtNet arm overall (0.925) but better on Baker's 0.971 vs 0.942,
Synovitis 0.873 vs 0.853, Lateral OA 0.838 vs 0.820 and Medial OA 0.978 vs 0.974, i.e. it makes different errors.

Everything else is byte-identical to model61 (scripts/run_model23_24.py model67: kind = model61 + RESNET_ARM).
The arm uses the maxspan geometry (336 px cache, 64 slots, 2-98% span, 62 windows), so it shares that decode group and
adds almost no runtime; it is resized to 256 px by the public eval_windows. Arm weight 0.30 against the seven CoAtNet views
(total 1.30), fixed before any leaderboard result: gold rank-blend tests with the 4-arm CoAtNet blend gave 0.9314 (w 0.2)
to 0.9326 (w 0.4) against 0.9305, inside noise on 58 cases, so the leaderboard is the real test.

Weights: private dataset yodharinmonplub/rsna-knee-model65-weights (sha256 1985e017...). Source: notebook.ipynb here.

2026-09-22: run from teammate account krittanutsomtuas (same team; yodharinmonplub's GPU quota used up). Version 1
failed because the public CoAtNet child hard-codes /kaggle/input/competitions/<comp> and that account mounts the
legacy /kaggle/input/<comp>; the build now tries both paths (COAT_ROOT_NEW in run_model23_24.py). Version 2 pushed.
