Model25: deeper weak-label fine-tune of the max-span CoAtNet (model16 variant).

Recipe (scripts/train_model16.py with model25/config.json): same verified 2%-98% 336px
dense cache, Steven Lee v4 targets, report-disjoint partition and frozen-teacher penalty
0.3 as model16, but the last TWO CoAtNet stages are trainable, 3 epochs, encoder/head
LR 1e-5/2e-5 with cosine decay to 10%, seed 20260917. Selection by weak-validation BCE.

Cloud: attempt1 (instance 51235681) never left image loading; destroyed via
cloud/abandon.py and its machine excluded. Attempt2 (51237122, RTX3090, $0.182/h) was on
an I/O-starved host (~70 min/epoch). Weak-validation BCE: epoch0 0.457297, epoch1
0.369724, epoch2 0.365205. Epoch3 crashed: clip_grad_norm_(error_if_nonfinite=True)
raised on an fp16 overflow step that GradScaler would have skipped (now fixed in the
trainer). The epoch2 checkpoint was uploaded from the rental straight to the private
Kaggle dataset yodharinmonplub/rsna-knee-model25-weights (sha256 e4efa39f...), because the
host's link to the local machine was ~0.12 MB/s. Estimated compute $0.91; observed
credit decrease $2.08 (includes another project's concurrent instance; recorded
conservatively).

Development evaluation (58 upstream-development cases, cached pixels, Kaggle T4;
scripts/run_model25_gold.py + scripts/analyze_model25.py):
model25 0.919413, model16 0.923080, model16+model25 mean 0.921371, model12 0.919929.
model25 - model16 paired 95% interval [-0.01224, 0.00529].

Conclusion: the deeper fine-tune did not beat model16 on the pre-declared weak-BCE
criterion (0.365205 vs 0.364008) nor descriptively on gold. Not used in ensembles.
