Model144: Same as model142 (students leg) at w 0.60: weight curve with the stronger leg. Leg = fleet leg v6 (one-pass reader, priority dropping, PARTIAL blend on a time shortfall), s96 members at window stride 2 (~8.5 min each on the hidden set; gold-neutral for s96); light dummy run (scripts/light_gate.py) on the main account.
Build (from Other/): python scripts/run_speedy_fleet_v2.py --name model144 --title "RSNA Knee Model144 Students Leg W60" --weight 0.60 --leg v6 --dummy light --fleet "[('model113', ['model113_best.pt'], 's96', 2), ('model104', ['model104_best.pt'], 's96', 2), ('model112', ['model112_best.pt'], 's96', 2), ('model132', ['model132_best.pt'], 's96', 2)]"
Planned: dummy run + submission Sat 2026-10-10 from 07:01 (main GPU quota refresh).
Score: pending.
