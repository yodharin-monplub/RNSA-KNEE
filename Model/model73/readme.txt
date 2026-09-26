Model73: FULL training of a CoAtNet-2 (coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k, 73M parameters) - the same backbone
family as every arm of our 0.943 blend, which are only fine-tuned from public Raptor weights. Recipe = model65/model72
(every layer trained from ImageNet-12k/1k weights, GPU augmentation, plain soft-label BCE on the Gemini v1 labels,
16 windows per study, batch 8, cosine, 25 epochs, epoch chosen by weak-validation BCE; gold never used for
training or selection), with three settings this backbone needed:
  - backbone rebuilt at 256 px (backbone_img_size): its relative-position bias is baked for 384 px, so a plain
    256 px input crashed. All 478 tensors of the 384-px checkpoint load with no shape mismatch (verified locally).
  - gradient checkpointing, chunk 32: plain settings ran out of memory on a 24 GB card.
  - encoder LR 2e-5 (head 1e-3), warmup 10%: at the small-CNN LR of 2e-4 the model never fit even its training set
    (five epochs, gold 0.562 -> 0.584, training loss flat near 0.5); at 2e-5 gold was 0.710 after one epoch.

Result (58 gold cases):
  model73 alone                        0.9122  (selected epoch 17, weak-val BCE 0.3115; peak gold 0.918 at epoch 19)
  model72 EfficientNet-B0 288          0.9143
  model65 ResNet34 256                 0.9104
  model61 four-arm blend               0.9312  (public LB 0.943)
  blend + model73 at pre-declared 0.30 0.9335  (+0.0023, P>0 0.81) with the RANK-mean approximation of the blend
  SAME TEST THE NOTEBOOK'S WAY (probability mean, view weights maxspan 0.70 / dense 0.15 / v8 0.30 / fine 0.15,
  base reproduces the canonical 0.9326):  0.9326 -> 0.9344 (+0.0018, 95% CI [-0.0027, +0.0064], P>0 0.78) => NOT adopted,
  0.0002 under the bar. (w=0.40 gives +0.0020, but choosing the weight after seeing it is not allowed.)
  Control: model72 at 0.30 the notebook's way: +0.0006.
  blend + mean(model73, model72)       0.9327  (+0.0016) => no
model73 is not a stronger single model than EfficientNet-B0, but it improves the blend more (+0.0018 vs +0.0006 the notebook's way):
its errors differ from the fine-tuned CoAtNet arms'. Caveat: +0.0023 is at the edge of noise on 58 cases, and
model67 (+0.0021 on gold) lost 0.001 on the public LB, so the leaderboard is the real test.

Cost: the successful run was 4.8 h on an RTX 3090 at 0.174 USD/h = 0.84 USD. All earlier attempts together cost
0.48 USD (position-bias crash, broken hosts, out-of-memory, the LR-2e-4 run stopped at epoch 5).
Weights: artifacts/remote_model73/model73/best.pt (293 MB) with gold_predictions.csv, history.json, provenance.json.
Build: python Other/scripts/train_model16.py --cache data/dense --config model73/config.json --checkpoint none
(driven by Other/cloud/run_model73.sh); backbone dataset yodharinmonplub/rsna-knee-backbone-coatnet2.

