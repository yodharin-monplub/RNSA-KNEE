Model17: fixed equal probability blend of model12 Max-Span and model15 Native384 Dense.
Both checkpoints: Johnathan Wagner / Dread Development, public CC0 Kaggle datasets raptor-knee-maxspan and raptor-knee-native384dense.
Each branch retains its exact independently validated preprocessing and runs in a separate subprocess. Offline inference; no report labels used at inference.
58-case development AUC: 0.921806; model12: 0.919929. Paired 95% improvement interval includes zero. This is not an untouched holdout or a public leaderboard score.
Weights fixed before the development comparison; no weight search. Model12 remains first submission priority.
Status: private T4 validation version2 COMPLETE (59.38 seconds, three visible studies; exact equal blend verified); not submitted.
Version1 stopped after model12 predictions because the wrapper expected an absent runtime.json. Removed the optional rename. A local wrapper integration check using real visible prediction files confirms exact 50/50 output; embedded prediction code remains unchanged.
