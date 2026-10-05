Model134 (diagnostic, dummy run only, not submitted): model111's exact notebook built with the LIGHT DUMMY-RUN GATE in
'light' mode (see Model/model133/readme.txt and Other/scripts/light_gate.py). On the 3-study commit run the 23 Speedy
cells are skipped, a stand-in submission.csv (0.5 everywhere) is written, and the fleet leg runs in full.
Checks: the log shows "light=True" and 23 "skipped (dummy run)" lines; the leg's member predictions (model80_fleet.npz)
equal model111's dummy-run ones; the session is ~2 GPU minutes instead of ~8.
Build: python scripts/run_speedy_fleet_v2.py --name model134 --title "RSNA Knee Model134 Gate Check Light" --weight 0.30 --leg v3 --dummy light --fleet "[('model104', ['model104_best.pt'], 's96'), ('model75', ['model75_best.pt'], 's64'), ('model100', ['model100_best.pt'], 's96'), ('model98', ['model98_best.pt'], 's64')]"
Pushed 2026-10-04 07:53 (+07) on the main account (yodharinmonplub/rsna-knee-model134-gate-check-light v1).
RESULT 2026-10-04 07:55: dummy run COMPLETE, log span 7-74 s (model133 gated-full: 8-426 s; model126 plain: 12-495 s). 23/23 Speedy cells skipped, stand-in written, leg 3/3 studies, all 4 members, dropped for time [], 0 tracebacks; the leg's member predictions (model80_fleet.npz) equal model111's dummy run exactly (max abs diff 0.0). => every submission from now on is built with --dummy light (~1.5 instead of ~8 main-account GPU minutes). Never submitted (never_submit.txt).
