Model36: new model family - ConvNeXt-small (timm convnext_small.fb_in22k_ft_in1k_384) trained from
ImageNet weights on the verified 2%-98% 336px maxspan cache, in the Raptor classifier format
(attention pooling over 3-slice windows; loadable by the public Raptor loader via arch/res metadata).

Recipe (scripts/train_model16.py, model36/config.json): all parameters trained, 336px, 10 epochs,
batch 4 studies x 12 random windows, grad checkpointing, AdamW LR 5e-5 (backbone) / 1e-3 (head),
5% warmup + cosine to 1%, wd 0.05, v4 targets with confidence-weighted BCE, selection by weak BCE.
Pretrained weights from private dataset yodharinmonplub/rsna-knee-backbone-convnext-small (the
rental could not reach the HF hub on attempt1, which failed before training, cost ~$0.04).

Result: weak BCE 0.525 -> 0.408 (epoch 10 selected); gold development AUC by epoch
0.807, 0.851, 0.873, 0.873, 0.880, 0.883, 0.884, 0.887, 0.892, 0.892.
Blend with the fine-tuned CoAtNet side (0.9273): 0.9255 at 10%, 0.9237 at 30% -> no gain
(prediction correlation 0.895). Not used. Cloud: instance 51309915, 1.9 h, estimated $0.33.
Conclusion: a new family trained from ImageNet on these weak labels is far weaker than the public
checkpoints fine-tuned by this project; label quality is the binding constraint.