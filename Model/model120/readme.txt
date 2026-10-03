Model120: model76 (Speedy Raptors + our model61 Raptor weights) + a final rank leg (w 0.30) built with FLEET LEG V4 at
window STRIDE 2 (every 2nd window: half the GPU time per member; on model104 x 58 gold studies stride-2 predictions
have Spearman 0.995 with all windows and gold 0.9309 vs 0.9281) over four ImageNet-initialised CoAtNet-2 384 models,
in priority order (the last is dropped first if the time guard runs short):
  model110f0 = model110 fold 0 (TPU, 96-frame cache, 80% of the non-gold data, 16 epochs, gold 0.9343; 0.936 at ep 15)
  model104   (TPU, 96 frames, gold 0.9306) | model75 (GPU, 64 frames, LB 0.939 alone) | model100 (GPU, 96 frames)
Question: does model110 fold 0 + stride 2 beat model107 (same leg without model110f0, all windows)?
Build: python scripts/run_speedy_fleet_v2.py --name model120 --title "RSNA Knee Model120 Speedy Leg Four Stride2" \
  --weight 0.30 --leg v4 --fleet "[('model110f0', ['model110_fold0_best.pt'], 's96', 2), ('model104', ['model104_best.pt'], 's96', 2), ('model75', ['model75_best.pt'], 's64', 1), ('model100', ['model100_best.pt'], 's96', 2)]"
Weights: yodharinmonplub/rsna-knee-fleet-weights (model110_fold0_best.pt sha256 b837d3c7...).
Dummy run + submission planned for Saturday 2026-10-03 (after the GPU quota refresh, 07:40 runner).
STRIDES (2026-10-02 GPU check, krittanutsomtuas/rsna-knee-leg-window-parity-gpu, fp16 like the leg): s96 members
lose ~nothing at stride 2 (model110f0 gold 0.9342 all -> 0.9333 even, Spearman 0.995) but s64 members lose ~0.003
(model109 0.9113 -> 0.9079, model98 0.9199 -> 0.9174, Spearman 0.991) -> model75 (s64, our strongest LB member) runs
all windows (stride 1); the other members stride 2.
Dummy run VERIFIED (all members, none dropped) and SUBMITTED 2026-10-03 07:47:13 (+07), submission 56785892.
Score: public 0.947 (scored 2026-10-03 15:42). Same as model107/121; model111 (full windows, model98 instead of
model110f0) scored 0.948 -> stride 2 may cost a little on the LB, or it is rounding noise.
