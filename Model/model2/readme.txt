Model 2: frozen DINOv2-small MRI encoder and regularized linear classifier
Status: trained on Vast.ai. Gold macro AUC: 0.749623 (58 held-out studies).
Local artifacts: artifacts/remote/model2/metrics.json and head.pt.
Inputs: six plane/fluid slots, three neighboring-slice RGB groups per slot.
Trainable predictor: slice-mean features per slot, masked absent slots, linear
12-label head. DINOv2 ImageNet normalization; encoder features standardized
using the training split only.
Training targets: public report-derived soft labels (original v1), confidence
weight 2*abs(p-0.5); unaddressed findings contribute zero loss.
All 58 gold-labeled cases and matching normalized reports are excluded from
training and weak validation. Early stopping uses grouped weak validation BCE.
Gold evaluation occurs after checkpoint selection. External report labels may
have been developed using gold examples; this remains a validation limitation.
Run on GPU: python -m knee.extract --cache data/pixels --out artifacts/features
Then: python -m knee.train_heads --config model2/config.json
Sources and licenses: SOURCES.md. No reports required at inference.
