Model47: the twenty public DINOv2 members fine-tuned on this project's Gemini relabels.

Pipeline: a new cache built with the public notebook's own preprocessing cells (6 slots
SAG_FLUID_FS/COR_FLUID_FS/AX_FLUID_FS/SAG_FLUID_NOFS/COR_T1/SAG_T1, 3 slices, 336px, 130mm crop,
0.2-0.8 band, laterality normalisation), 4 Kaggle CPU parts, 4,407 studies, 72 shards
(scripts/run_dino_cache.py, cloud/prepare_dino_cache.py, knee/dino_cache.py).
Architecture regenerated verbatim from the notebook (scripts/make_dino_model_module.py ->
knee/dino_model.py); each member's stored fingerprint reproduced within 5e-6 before training, which
confirms the rebuild computes what the released weights were fitted to.
Training (scripts/train_dino_members.py, model47/config.json): per member, one epoch on a
member-specific random 80% of the non-gold studies (diversity preserved), confidence-weighted BCE on
the relabels, frozen-teacher penalty 0.3, LR 5e-6 backbone / 1e-5 head. Gold58 never trained on.
Each member is re-saved with a recomputed fingerprint so the ensemble's 20/20 check still passes.

IMPORTANT: these public members score mean 0.9314 on the 58 gold cases (single members up to 0.959)
versus 0.9301 for our honest CoAtNet blend, and their own manifest reports a 0.838 holdout. The gold
studies were therefore part of their training/selection data, so gold AUC cannot evaluate them;
mean gold after fine-tuning is 0.9307 (unchanged within noise) and carries no information.
Only the leaderboard can judge this change: model49 (members only) vs model23 0.940, and
model48 (members + our CoAtNet arms) vs model43.

Cloud: attempt1 failed at import (download_dense asserted a dense-cache run name), 0.01 USD.
Attempt2 instance 51369520, 20 members in 1,647 s, estimated 0.14 USD. The local backup captured only
8 of 20 files before the watchdog destroyed the rental (watchdog bug, now fixed to verify the backup),
but the members had already been uploaded from the rental to the private dataset
yodharinmonplub/rsna-knee-model47-members (20 distinct files + manifest, verified by listing).