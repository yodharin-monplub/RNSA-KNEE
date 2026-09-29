Model102: model99 + model100 in the fleet - model76 (Speedy Raptors + our model61 Raptor weights) + a final-level
rank leg at w 0.30 over fleet6 = model72, model71 x5, model73, model75 (all 's64'), model86 ('s128') and model100
(CoAtNet-2 384 on the 96-frame cache, 's96'; stopped at 12 of 18 epochs, selected epoch 11, gold 0.9231).
Rationale: the fleet leg paid only once a strong CoAtNet-2 member joined (model80 0.944 -> model78 0.946); model100 is
a second member of that class, trained on denser input.
Built with Other/scripts/run_speedy_fleet_v2.py --name model102 --weight 0.30 --fleet (see kernel metadata / notebook).
Dummy run VERIFIED 2026-09-29 06:06 (our model61 Raptor weights, fleet6 leg = 10 checkpoints incl. model100 at s96,
3/3 studies, no errors). Submit after the 07:00 quota reset.
SUBMITTED 2026-09-29 07:07 (+07), submission 56656299.
Score: public 0.945 (scored 2026-09-29 13:54) = model99 (fleet5, 0.945); model78 (fleet4) 0.946. model100 does not
lift the diluted fleet back up.
