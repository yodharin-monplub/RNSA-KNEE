Model39: model20 recipe retrained on the Gemini 3.8 Flash relabels.

Only change versus model20 : targets file data/llm_relabel/targets_g38f_low.csv (all 4,407 reports relabelled
with google/gemini-3.8-flash at low reasoning; those labels score 0.9035 macro AUC on the 58 gold cases
versus 0.8927 for the v4 labels, cost 6.32 USD). Same checkpoint, cache, geometry, recipe, selection.

Gold development AUC 0.9204840968112924 versus 0.919209 for model20. Weak-validation BCE is not comparable across label
sets. Private dataset: yodharinmonplub/rsna-knee-model39-weights.
