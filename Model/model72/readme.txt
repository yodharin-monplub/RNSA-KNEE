Model72: the model65 recipe with only the backbone and the resolution changed - EfficientNet-B0 at 288 px instead of
ResNet34 at 256 px. Everything else identical: every layer trained from ImageNet weights, GPU augmentation, plain
soft-label BCE on the Gemini v1 labels, 16 windows per study, batch 8, cosine schedule, 25 epochs, epoch chosen by
weak-validation BCE (gold is never used for training or selection). This is the recipe Tucker Arrants (4th on the
public LB) reports: a small backbone at 224-288 px.

Result (58 gold cases, descriptive):
  model72 EfficientNet-B0 288   0.9143   <- best single image model this project has trained
  model65 ResNet34 256          0.9104
  model71 5-fold ResNet34 256   0.9138 (ensemble of five folds)
  a fine-tuned CoAtNet arm      0.925
  model61 four-arm blend        0.9312 (public LB 0.943)
Selected epoch 17 of 25 by weak-val BCE 0.3174; gold rose steadily to 0.9135 by epoch 24, so the recipe is stable -
no divergence, which is what killed the earlier from-ImageNet attempts that had no augmentation.

As an extra arm in the model61 blend at the pre-declared weight 0.30: 0.9312 -> 0.9328 (+0.0016, 95% CI
[-0.0033, +0.0068], P>0 0.75) => NOT adopted, below the +0.002 bar fixed before the test. Averaged with model71's
5-fold ensemble first, it is worse: +0.0008. For context, model67 put a single ResNet34 arm into the blend for
+0.0021 on gold and the public LB went 0.943 -> 0.942.

Per finding vs ResNet34: MCL +0.132, Lateral Meniscus +0.016, Medial Meniscus +0.016, Medial OA +0.008;
Fracture -0.051, Synovitis -0.038, Contusion -0.015, ACL -0.012. The gain is concentrated in MCL, where the
CoAtNet arms are already strong, which is why the blend barely moves.

=> NOT worth taking to 5 folds. model71 showed five folds buy about +0.0034 over one model of the same recipe, so a
5-fold EffNet-B0 would reach roughly 0.918 - still below a single fine-tuned CoAtNet arm (0.925), and its blend
contribution would stay inside the noise, for about 2.5 USD. Full training of SMALL CNNs plateaus around 0.914 for us.

Cost: one Vast RTX A4000 at 0.162 USD/h, 2.3 h, 0.37 USD. Two earlier rentals for this run were destroyed without
training (Vast machine 11183 cannot pull the PyTorch image; a third booted fine but my own ssh -F os.devnull bug
blocked the deploy from Git Bash - fixed in cloud/vast.py).
Weights: artifacts/remote_model72/model72/best.pt (17 MB) plus gold_predictions.csv, history.json, provenance.json.
Build: python Other/scripts/train_model16.py --cache data/dense --config model72/config.json --checkpoint none
(driven by Other/cloud/run_model72.sh); backbone dataset yodharinmonplub/rsna-knee-backbone-effb0.
