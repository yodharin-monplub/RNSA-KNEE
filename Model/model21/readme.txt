Model21: fixed equal probability blend of model16 and model20.

Each branch is a weak-label fine-tuned CoAtNet selected by confidence-weighted
weak-validation BCE. Model16 uses the max-span checkpoint with a 336-pixel
intermediate cache; model20 uses the complementary native384 checkpoint with a
direct 384-pixel cache. Both use 64 slices, the 2%-98% stack span and a 140mm
crop. The two exact component inference programs run sequentially so GPU memory
is released between branches.

Weights: model16 0.5; model20 0.5. This simple blend was specified before its
development score was computed. There was no continuous or per-label weight
search. On the 58 upstream development cases its macro AUC is 0.926461 versus
0.926221 for model19, a difference of 0.000240. The paired bootstrap 95% interval
for model21 minus model19 is [-0.006340, 0.006543], so the apparent improvement
is uncertain. These cases are not an untouched holdout.

Model20 selected epoch1: weak-validation BCE 0.368714 versus baseline 0.465390;
descriptive gold AUC 0.919209 versus model15 baseline 0.916975. Model20 checkpoint
SHA-256 is 5087c36765aeb9fdf19881c778edd905eb3e91ee1cd78defb8b90175e8924a75.

Architecture, original checkpoints and geometry: Johnathan Wagner / Dread
Development. Fine-tuned model16 and model20 checkpoints: this project. See
SOURCES.md.

Private Kaggle T4 version1 completed in66.29 seconds for three visible studies.
Schema, row alignment, finite/range values, exact embedded component source
identity, checkpoint metadata, and 0.5/0.5 arithmetic passed; maximum arithmetic
error was8.33e-17. Exact version1 was submitted as submission56164781 and
scored public AUC0.935, tying model19 at the displayed precision. It did not
raise the campaign best, but confirms that the complementary model20 branch can
replace model19's three public-checkpoint branches without reducing the score.
