Model107: model103 (public 0.947, project best: model76 + rank leg at w 0.30 over model75 + model100) + model104 as a
third leg member - model104 is the TPU-trained CoAtNet-2 384 on the 96-frame cache (gold 0.9306, best own model).
Leg members: model75 ('s64'), model100 ('s96'), model104 ('s96'), all ImageNet-initialised CoAtNet-2 384 (model79,
which shares the Raptor init, diluted the leg: model106 0.946). Compare model111 (+ model98 as well).
Built with Other/scripts/run_speedy_fleet_v2.py --name model107 --weight 0.30 --fleet
  "[('model75', ['model75_best.pt'], 's64'), ('model100', ['model100_best.pt'], 's96'), ('model104', ['model104_best.pt'], 's96')]".
Dummy run and submission after the Kaggle GPU quota refresh (2026-10-03 07:00).
REBUILT 2026-09-30 with fleet leg v3 (Other/scripts/fleet_leg_v3.py, run_speedy_fleet_v2.py --leg v3): identical
predictions when time allows (one-pass reader, byte-identical volumes on 43 real studies), but on a projected overrun the
LAST fleet members are dropped first instead of abandoning the whole leg (v2 silently fell back to Speedy = model76).
Priority order = fleet order: model104, model75, model100.
Build: python scripts/run_speedy_fleet_v2.py --name model107 --title "RSNA Knee Model107 Speedy CoAtNet Leg Plus M104" --weight 0.30 --leg v3 --fleet "[('model104', ['model104_best.pt'], 's96'), ('model75', ['model75_best.pt'], 's64'), ('model100', ['model100_best.pt'], 's96')]"
Dummy run VERIFIED and SUBMITTED 2026-10-03 07:14:40 (+07), submission 56785116.
Score: public 0.947 (scored 2026-10-03 14:40) = ties the best (model103 0.947, leg model75 + model100);
adding model104 (alone 0.942) did not show at 3-decimal resolution.
