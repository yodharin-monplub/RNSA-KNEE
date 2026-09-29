Model103: model76 (Speedy Raptors + our model61 Raptor weights) + a final-level rank leg at w 0.30 over ONLY our two
CoAtNet-2 384 px models: model75 (64-frame cache, 's64'; LB 0.939 alone) and model100 (96-frame cache, 's96';
stopped at epoch 12 of 18, selected epoch 11, gold 0.9231).
Question: do the weaker fleet members (EfficientNet-B0 / ResNet34 / CoAtNet-2 at 256 px) dilute the leg? Evidence so
far: fleet4 alone 0.938 (model88) < model75 alone 0.939 (model89); fleet3 without model75 0.933 (model81).
Compare with model102 (same base and weight, fleet6 = all members incl. model100).
Built with Other/scripts/run_speedy_fleet_v2.py --name model103 --weight 0.30
  --fleet "[('model75', ['model75_best.pt'], 's64'), ('model100', ['model100_best.pt'], 's96')]".
Dummy run VERIFIED 2026-09-29 05:40 (our model61 Raptor weights, fleet leg with model75 + model100 = 2 checkpoints,
3/3 studies, no errors). Submit after the 07:00 quota reset.
SUBMITTED 2026-09-29 07:06 (+07), submission 56656251.
Score: public 0.947 (scored 2026-09-29 14:09) = NEW PROJECT BEST (+0.001 over model78/93/94 at 0.946; model102 with
the full fleet6 at the same weight 0.945). Only strong CoAtNet-2 384 members belong in the leg.
