# Data and pretrained weights

- Official metadata and gold labels: [RSNA Knee Abnormality Detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/data), competition rules and RSNA MIRA terms.
- MRI cache: Dmitrii Gluzdov, [RSNA Knee uint8 224 × 9 crop 130](https://www.kaggle.com/datasets/dmitriigluzdov/rsna-knee-uint8-224-9-c130), version 1. Derived competition pixels remain subject to competition/MIRA restrictions. Cache geometry credits Steven Lee's Apache-2.0 code. Preserve the downloaded NOTICE and LICENSE files. No blanket open-source license is asserted for MRI data.
- Weak targets: Steven Lee, [RSNA Knee LLM Report Labels](https://www.kaggle.com/datasets/stevenleehans/rsna-knee-llm-report-labels), version 6, `llm_labels_full.csv` (original v1), CC0. The publisher credits Pilkwang Kim, Karnakbayev Artur, Will/wguesdon, Alexandre Moritz, and Roman Rozen. These are report-derived probabilities, not independent clinical annotations. V2/v4 gold-informed refinements are not used in the initial pilot.
- Encoder for models2–6: Meta, [DINOv2](https://github.com/facebookresearch/dinov2), Apache-2.0 pretrained weights via timm, `vit_small_patch14_dinov2.lvd142m`. These models use no competition-fine-tuned third-party weights.

The competition permits freely and publicly available external data and pretrained models and requires offline notebook inference within nine hours: [official code requirements](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview/code-requirements).

Downloaded dataset metadata snapshots are in `artifacts/sources/`. The cache's SPEC.json supplies checksums and exact geometry. No training data or secrets are included in the code bundle intended for inference.

## Model7 public checkpoint reproduction

Johnathan Wagner / Dread Development publishes [CoAtNet weights](https://www.kaggle.com/datasets/dreaddevelopment/raptor-knee-widedense), version 2, under CC0. Model7 uses `raptor_ft_coatnet_v4_full.pt`. Our competition submission reproduced the publisher's 0.924 public AUC. Full coverage model8 and SWA model9 both scored 0.927.

Model10 uses the same author's [Fine Spacing checkpoint](https://www.kaggle.com/datasets/dreaddevelopment/raptor-knee-finespacing), `raptor_ft_coatnet_v9_full.pt`, CC0 as verified through Kaggle dataset metadata on 2026-09-11. Exact training sampling is incompletely documented; model10 tests it in model8's inference geometry.

Model12 uses the author's [Max-Span checkpoint](https://www.kaggle.com/datasets/dreaddevelopment/raptor-knee-maxspan), `raptor_ft_coatnet_v5_full_swa.pt`, CC0. The dataset card specifies 64 slices sampled at 2–98%, a 140mm crop, 336px intermediate size, and 62 windows. Metadata saved on 2026-09-11. The reported 0.928 public AUC has not yet been reproduced here.

Model13 adapts only the DINOv3 member from Mattia Angeli's [Apache-2.0 notebook](https://www.kaggle.com/code/mattiaangeli/bend-the-knee-to-dinov3-ensembled). Five fold checkpoints are from [knee-mri-fold-weights](https://www.kaggle.com/datasets/mattiaangeli/knee-mri-fold-weights), whose dataset listing is CC0. That listing does not replace the underlying [Meta DINOv3 license](https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md). Local attribution and modifications are recorded in model13/NOTICE.txt and readme.txt. The source's later leaderboard-tuned whole-ensemble weights are not used.

Model15 uses Dread Development's [native384dense checkpoint](https://www.kaggle.com/datasets/dreaddevelopment/raptor-knee-native384dense), `raptor_ft_coatnet_v10_full.pt`, CC0 dataset listing. It uses the documented native384 intermediate resolution with 64 slices at 2–98% and a 140mm crop. This changes both checkpoint and intermediate geometry relative to model12.

Model16's planned fine-tuning uses `llm_labels_v4_blend.csv` from Steven Lee's report-label dataset, combining that publisher's v2 with lixin73's labels. Unlike the initial DINO pilot's v1-only targets, v4 includes development-informed refinements. These are noisy report probabilities, not ground truth; neither gold nor weak validation is claimed untouched by upstream development.

Architecture and preprocessing are described in the author's [inference notebook](https://www.kaggle.com/code/dreaddevelopment/knee-mri-twelve-findings-from-a-single-model) and [Apache-2.0 training notebook](https://www.kaggle.com/code/dreaddevelopment/knee-mri-training-the-twelve-finding-model). Our compatible inference implementation uses tensor-only loading, bounded backbone microbatches, raw probabilities, shallow input discovery, and explicit data/prediction checks. It is kept separate from the locally trained DINO experiments; no independent gold-validation result is claimed for the public checkpoint.
