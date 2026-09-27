Model76: the public notebook "Bend the Knee to Speedy Raptors - The Original" (mattiaangeli, public LB 0.943,
all public assets) with ONE change - its four Raptor CoAtNet views load this project's Gemini-fine-tuned,
head-retrained checkpoints (the model61 weights) instead of the public dreaddevelopment ones.

Why this should help: our improvement over the older public base came entirely from those Raptor weights
(Dinosaurs repro 0.940 -> model24 weak-label fine-tune 0.941 -> model43 Gemini-label fine-tune 0.942 -> model61
head retrain 0.943). Extra Raptor views never moved the leaderboard. Speedy Raptors reached 0.943 by a different
route (two new CoAtNet readers, D4 depth-zone and Global96, blended into the Raptor side, plus capacity-aware
dense slice sampling with ~94 windows) and still ships the public Raptor weights, so the two should stack.

Mapping (same as model61): maxspan-v5 and maxspan-v5-reverse -> model61_maxspan_best.pt; native384dense-v10 ->
model61_dense_best.pt; native384-v8 -> model61_v8_best.pt. Those files are the outputs of the private kernels
yodharinmonplub/rsna-knee-head-retrain-{maxspan,dense,v8} (Other/scripts/run_head_retrain.py). Speedy only looks
for weights inside a whitelist of the public dataset folders, so the patch gives our three filenames their own
lookup. Speedy's loader is strict (exact keys and shapes, architecture equality), logs a SHA-256 for every loaded
checkpoint, and weight lookup and loading sit outside any try-block - a wrong or missing file stops the run
instead of silently falling back to the public weights. Everything else (views, view weights 0.60/0.10/0.10/0.20,
dense sampling, the two-GPU schedule, the CoAt readers, the outer blend) is Speedy's, untouched.

Build: python Other/scripts/run_model76.py [--push]   (source copy in this folder; the public notebook is
Other/artifacts/sources/public_notebooks/mattiaangeli__bend-the-knee-to-speedy-raptors-the-original, pulled
2026-09-27). notebook.ipynb and kernel-metadata.json here are the exact pushed files.
Verification before submitting (scratchpad model76_tick.py): the "[model76] Raptor arms" line lists our three
files, each appears in a raptor_checkpoint load event, zero public Raptor checkpoints are loaded, no Raptor study
failures or neutral fills, submission.csv written. Pushed 2026-09-27 from yodharinmonplub (dummy-data run only).
Score: pending (see score.txt once scored).

SCORE: public 0.945 (submission 56587720) - NEW PROJECT BEST, +0.002 over both model61 (0.943, our own pipeline) and
the unmodified Speedy notebook (0.943). Our Raptor weights and Speedy's pipeline improvements stacked as expected.
Leaderboard 2026-09-27: rank 134 of 4,407 (public silver zone). Hidden rerun ~7.4 h including queue: Speedy is
slow on the ~1,300 hidden studies, which leaves ~1-1.5 h of the 9 h limit for anything added after it.
