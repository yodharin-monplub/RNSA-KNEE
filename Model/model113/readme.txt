Model113 (planned): PSEUDO-LABEL STUDENT - model112's full-data CoAtNet-2 384 recipe on the 96-frame 336 px cache (all
4,345 non-gold studies, 18 epochs, last epoch kept, seed 20261017) trained on targets_v1_oof110_50.csv = 0.5 x the v1
Gemini labels + 0.5 x model110's 5-fold OUT-OF-FOLD predictions (each study predicted by the fold model that never saw
it); gold rows keep their labels (gold is never trained on).
Why: the forum's main lever - Chris Deotte: single models trained on pseudo labels match the best ensemble's CV; 11th
place reports a 5-fold ResNet at 224 px = 0.954; Raymond Yuen (0.947): weak-label teachers -> pseudo labels -> final
model. Our earlier pseudo-label failures (2026-09-20/24) were frozen-feature HEAD retrains with a weak ResNet34 teacher
(gold 0.91); this is a full CoAtNet-2 model with a CoAtNet-2 5-fold teacher.
Steps once model110 folds 0-4 are collected (artifacts/remote_model110-f012/model110/fold{0,1,2},
artifacts/remote_model110-f34-then-model112/model110/fold{3,4}):
  python scripts/make_pseudo_targets.py --labels data/llm_relabel/targets_g38f_low.csv --alpha 0.5 \
     --runs artifacts/remote_model110-f012/model110/fold0 ... fold4 --out data/llm_relabel/targets_v1_oof110_50.csv
  (check: OOF coverage 4,345/4,345; gold sanity = labels 0.9035 vs the mix with the fold models' gold predictions)
  add 'targets_v1_oof110_50.csv' to LAYOUT and 'model113' to MODEL_CONFIGS in scripts/build_trainer_kit.py, then
  python scripts/build_trainer_kit.py --account krittanutsomtuas --push "pseudo-label targets + config_model113"
  python scripts/run_tpu_train.py --name model113 --plan full --cache s96 --account krittanutsomtuas --push
Files: config.json, train_xla.py, xla_data.py, tpu_common.py, run_tpu_train.py, build_trainer_kit.py, collect_tpu_run.py,
make_pseudo_targets.py, upload_fleet_weights.py (copies of Other/scripts at build time).
Score: pending.
