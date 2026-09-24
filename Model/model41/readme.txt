Model41: model22 recipe retrained on the Gemini 3.8 Flash relabels.

Only change versus model22 : targets file data/llm_relabel/targets_g38f_low.csv (all 4,407 reports relabelled
with google/gemini-3.8-flash, low reasoning; those labels score 0.9035 macro AUC against the 58 gold
cases versus 0.8927 for the v4 labels). Same checkpoint, cache, geometry, recipe and selection rule.

Gold development AUC 0.9159693634860112 versus 0.915761 for model22. Weak-validation BCE is not comparable across
label sets. Private dataset: yodharinmonplub/rsna-knee-model41-weights.
