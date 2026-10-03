Model130: model111 leg with model110 fold0 instead of model100 (leg v3 full windows: model110f0, model104, model75, model98) at w 0.40.
Why: model111 (full windows; model104, model75, model100, model98; w 0.30) is the best (0.948) while the stride-2 legs
scored 0.947; model110 fold 0 is our best model on gold (0.934 vs model100 0.923). Same GPU cost as model111 (fits the
hidden-run budget).
Build: run_speedy_fleet_v2.py --name model130 --weight 0.40 --leg v3 --fleet "[('model110f0', ['model110_fold0_best.pt'], 's96'), ('model104', ['model104_best.pt'], 's96'), ('model75', ['model75_best.pt'], 's64'), ('model98', ['model98_best.pt'], 's64')]"
Planned submission: Sunday 2026-10-04 07:01.
Score: pending.
