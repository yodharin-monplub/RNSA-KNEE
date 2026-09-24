Model69: model65 recipe (ResNet34 from ImageNet, all layers, 25 epochs, augmentation, plain BCE, Gemini v1 labels) trained on fold B
only - one half of the non-gold studies, split by report-hash parity so duplicate reports stay together - and used to predict the other half.
model68 + model69 together give a true out-of-fold image prediction for every training study (the forum's top teams report that such
predictions are closer to the radiologists than report labels). Uses: (1) pseudo-label targets = blend of report labels and OOF predictions,
checked for free with the head-retrain test; (2) their average as an extra ensemble member. Gold (58 cases) is never trained on.
