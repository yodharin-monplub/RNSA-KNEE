Model60: model38 recipe (model16's maxspan fine-tune, same seed 20260911) retrained on hybrid report labels.

Only change versus model38: targets file data/llm_relabel/targets_hybrid.csv, built by scripts/make_hybrid_targets.py.
It keeps the Gemini 3.8 Flash relabels for ten findings and takes Effusion and Baker's from the public Lee v4 labels
(data/weak_labels_v4/llm_labels_v4_blend.csv), the two findings where a published set beats the Gemini labels by at
least 0.02 AUC on the 58 gold cases (Effusion 0.877 vs 0.846, Baker's 0.944 vs 0.898). Hybrid labels score 0.9099
macro AUC on gold versus 0.9035 for Gemini alone, but that figure is selected on the same 58 cases (optimistic).

Why those two: Gemini scores reports that do not mention a Baker's cyst 0.02-0.20 (noise; the one silent gold
positive sits at 0.02, below explicit "no cyst" reports), and scores "mild/small effusion" about 0.45, tied with
silent reports, whereas the gold readers mark roughly half of the mild effusions and few silent reports positive.
Lee v4 ranks explicit negative < silent (0.25) < mentioned, which is the structure the gold shows.

Purpose: ~1 USD probe of whether the hybrid labels lift the maxspan arm's gold AUC above model38's 0.925370;
the other three arms are only retrained if it does. Same checkpoint, cache, geometry, recipe, selection as model38.

Result (2026-09-19, Vast instance 51482148, RTX 3090, estimated 0.234 USD): selected epoch 1, gold AUC 0.925523 vs
model38 0.925370 - a +0.0002 difference with paired bootstrap 95% CI [-0.0015, +0.0020], i.e. neutral. Baker's
+0.0054 and Effusion -0.0012 on gold: the image model already outranks the report labels on those findings
(Effusion 0.980 vs 0.846 for the labels), so fixing their label structure does not help it. Other arms not retrained;
nothing submitted. Weights kept locally in artifacts/remote_model60/model60/best.pt.
