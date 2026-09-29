Model105: model103 (public 0.947, project best) with the CoAtNet-only leg at weight 0.40 instead of 0.30 - model76
(Speedy Raptors + our model61 Raptor weights) + rank leg over model75 ('s64') and model100 ('s96').
Question: with only strong members in the leg, does it deserve more weight? (With the diluted fleet4, 0.40 was worse:
model95 0.945 vs model78 0.946.) Informs the weight of the final submission's leg.
Built with Other/scripts/run_speedy_fleet_v2.py --name model105 --weight 0.40
  --fleet "[('model75', ['model75_best.pt'], 's64'), ('model100', ['model100_best.pt'], 's96')]".
Dummy run VERIFIED 2026-09-29 16:21; SUBMITTED 16:22 (+07), submission 56670477.
Score: public 0.945 (scored 2026-09-30 00:26) vs model103 at w 0.30 = 0.947 -> -0.002. A heavier leg hurts even with only
strong members (as with the full fleet: model95 w 0.40 0.945 vs model78 w 0.30 0.946). Keep w 0.30 (lower untested).
