Model58: maxspan arm with TWO backbone stages trainable (model51 recipe otherwise).

24 windows per study, 2 epochs, Gemini relabels, encoder LR 3e-6 (lower for the larger trainable set),
head LR 1e-5. Selected epoch 1. Weak-validation BCE 0.282365 - the best of any arm - but gold
development AUC 0.9260775459634072, below model51's 0.927971 with one stage. Extra capacity fits the weak labels
better without generalising. Blends: maxspan->58 0.9286, mean(38,58) 0.9297, none beating the
base blend 0.9305. Not used. Estimated cost 0.22 USD.
