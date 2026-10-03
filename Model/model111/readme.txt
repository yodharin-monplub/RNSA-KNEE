Model111: model107 + model98 - model76 + rank leg at w 0.30 over four ImageNet-initialised CoAtNet-2 384 models:
model75 ('s64', GPU), model100 ('s96', GPU, 12 of 18 epochs), model104 ('s96', TPU, gold 0.9306), model98 ('s64',
TPU, second seed of model75's recipe, gold 0.9154 at its selected epoch).
Question: does a second 64-frame seed add to the leg beyond model107?
Built with Other/scripts/run_speedy_fleet_v2.py --name model111 --weight 0.30 --fleet
  "[('model75', ['model75_best.pt'], 's64'), ('model100', ['model100_best.pt'], 's96'), ('model104', ['model104_best.pt'], 's96'), ('model98', ['model98_best.pt'], 's64')]".
Dummy run and submission after the Kaggle GPU quota refresh (2026-10-03 07:00).
REBUILT 2026-09-30 with fleet leg v3 (Other/scripts/fleet_leg_v3.py, run_speedy_fleet_v2.py --leg v3): identical
predictions when time allows (one-pass reader, byte-identical volumes on 43 real studies), but on a projected overrun the
LAST fleet members are dropped first instead of abandoning the whole leg (v2 silently fell back to Speedy = model76).
Priority order = fleet order: model104, model75, model100, model98.
Build: python scripts/run_speedy_fleet_v2.py --name model111 --title "RSNA Knee Model111 Speedy CoAtNet Leg Four" --weight 0.30 --leg v3 --fleet "[('model104', ['model104_best.pt'], 's96'), ('model75', ['model75_best.pt'], 's64'), ('model100', ['model100_best.pt'], 's96'), ('model98', ['model98_best.pt'], 's64')]"
Dummy run VERIFIED and SUBMITTED 2026-10-03 07:21:27 (+07), submission 56785295.
Score: public 0.948 (scored 2026-10-03 15:26) = NEW PROJECT BEST (+0.001 over model103/107/121 at 0.947).
Four CoAtNet-2 384 members at FULL windows (leg v3), w 0.30; ran within the hidden-run time budget.
