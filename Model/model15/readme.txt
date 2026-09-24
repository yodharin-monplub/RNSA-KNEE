Model15: native384 dense CoAtNet
Status: visible-test Kaggle version 1 COMPLETE; output schema, UID order,
finite values and probability bounds validated. Script runtime 22.6 seconds
on three visible cases including setup. 58-case development evaluation COMPLETE:
model15 AUC0.916875 vs model12 0.919929; fixed equal probability blend0.921806.
Blend-minus-model12 paired95% interval [-0.00570,0.01007]; no robust gain claim.
Detailed analysis: artifacts/native384_gold_notebook/output/analysis.json.
Checkpoint: dreaddevelopment/raptor-knee-native384dense/raptor_ft_coatnet_v10_full.pt.
Credit: Johnathan Wagner / Dread Development; CC0 checkpoint dataset listing.
Documented geometry: 64 slices, 2%-98% span, 140mm crop, native384px images,
62 windows. Compared with model12, change the checkpoint and intermediate
pixel size from336 to384, matching this checkpoint's training representation.
Architecture, labels, slice ordering, normalization and GPU microbatching match
model12. This is a checkpoint/geometry comparison, not an isolated resize test.
Predeclare comparison against model12 and an equal probability blend; no weight
search. Gold development cases were used by checkpoint authors and are not an
untouched holdout. No leaderboard score is claimed until actually submitted.
Source: https://www.kaggle.com/datasets/dreaddevelopment/raptor-knee-native384dense
Metadata: artifacts/sources/raptor-knee-native384dense/dataset-metadata.json.
No new Vast.ai rental. Run: scripts/run_model15.py; scripts/run_native384_gold.py.
