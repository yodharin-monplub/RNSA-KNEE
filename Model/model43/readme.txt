Model43: the public 0.941 ensemble with every Raptor CoAtNet view retrained on the Gemini relabels.

Views (weights unchanged from model30, renormalized over 1.30):
maxspan-v5 model38 (0.60), maxspan-v5-reverse model38 (0.10), native384dense-v10 model39 (0.10),
native384dense-v10-reverse model39 (0.05), native384-v8 model40 (0.20), native384-v8-reverse
model40 (0.10), finespacing-ft22 model41 (0.15). Grouped decode with 6-thread prefetch
(validated identical to the public per-arm loop in model37check).

Labels: data/llm_relabel/targets_g38f_low.csv - all 4,407 reports re-extracted with
google/gemini-3.8-flash (low reasoning) using radiologist-style definitions; 0.9035 macro AUC
against the 58 gold cases versus 0.8927 for the v4 labels.

CV (58 gold cases): arms 0.9254 / 0.9205 / 0.9197 / 0.9160; their weighted blend 0.9301 versus
0.9273 for the previous (v4-label) arms. Old+new mixed blend was worse (0.9289), so only the new
arms are used.

Kaggle: yodharinmonplub/rsna-knee-model43-gemini-relabel-arms v1 COMPLETE (7 views, no fallbacks),
submitted 56313516.
Confirmed public AUC 0.942, submission 56313516: new campaign best (+0.001 over the 0.941 plateau). The Gemini relabels lifted CV 0.9273 -> 0.9301 and the leaderboard 0.941 -> 0.942.
