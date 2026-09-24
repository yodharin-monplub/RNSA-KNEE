Model51 / Model52: maxspan recipe variants on the Gemini relabels (vs model38, CV 0.925370).

model51: 24 random windows per study instead of 12, 2 epochs. Selected epoch 2,
weak-validation BCE 0.286347, gold development AUC 0.927971 (+0.0026 over model38) - the best single
arm so far. More windows give a better gradient estimate per study at the same data cost.
model52: plain BCE instead of the confidence-weighted loss, 12 windows. Selected epoch 1,
weak BCE 0.295130, gold 0.925695 - neutral versus model38, so the confidence weighting is kept.

Both use the same checkpoint, cache, geometry and selection rule as model38; only the stated setting
differs. model51 weights are uploaded as yodharinmonplub/rsna-knee-model51-weights.