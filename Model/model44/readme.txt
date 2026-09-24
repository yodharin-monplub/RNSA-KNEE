Model44 / Model45: half-fold teachers for the self-distillation experiment (model46).

Each trains the model16/model38 maxspan recipe on one half of the 4,345 non-gold studies (split by
report_hash parity so duplicate reports stay together), with the Gemini relabels as targets, fixed
1 epoch, then writes out-of-sample predictions for the other half (oof_predictions.csv).
model44 fold A: 2,179 train / 2,166 held out, CV 0.923693. model45 fold B, CV 0.925448.
(model38, trained on all non-gold studies, is CV 0.925370.)
Caveat: the initial checkpoint was trained by its publisher on all studies, so these predictions are
not fully out-of-sample.