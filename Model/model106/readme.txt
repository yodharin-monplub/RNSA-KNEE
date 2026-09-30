Model106: model103 (public 0.947, project best) + model79 as a third member of the CoAtNet-only leg - model76 (Speedy
Raptors + our model61 Raptor weights) + rank leg at w 0.30 over model75 ('s64', ImageNet init), model100 ('s96',
ImageNet init) and model79 ('s64'; full fine-tune of the public Raptor maxspan CoAtNet-2 384 checkpoint on the Gemini
labels, gold 0.9242, selected epoch 10 - it failed the bar to REPLACE the maxspan arm, but it is a strong CoAtNet-2 384
model trained on our 64-frame cache, so it qualifies for the leg).
Caveat: model79 shares its initialisation with the Raptor maxspan arm, so it is less independent of the Speedy side than
the ImageNet-initialised members.
Built with Other/scripts/run_speedy_fleet_v2.py --name model106 --weight 0.30 --fleet
  "[('model75', ['model75_best.pt'], 's64'), ('model100', ['model100_best.pt'], 's96'), ('model79', ['model79_best.pt'], 's64')]".
Weights: model79_best.pt added to yodharinmonplub/rsna-knee-fleet-weights (2026-09-29).
Dummy run VERIFIED 2026-09-29 16:32 (3 CoAtNet-2 384 checkpoints, 3/3 studies, no errors). Submit after the 07:00 reset.
SUBMITTED 2026-09-30 07:02 (+07), submission 56690928.
Score: public 0.946 (scored 2026-09-30 13:57) vs model103 (same leg without model79) 0.947 -> model79 hurts the leg: it shares
its initialisation with the Raptor maxspan arm already in Speedy, so it adds little new and dilutes. Dropped from the leg.
