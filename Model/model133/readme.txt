Model133 (diagnostic, dummy run only, not submitted): model111's exact notebook (project best 0.948: model76 + fleet leg
v3 at w 0.30 over model104, model75, model100, model98) built with the LIGHT DUMMY-RUN GATE in 'gated-full' mode.
What the gate is (Other/scripts/light_gate.py, 2026-10-04): every Speedy cell gets a first line "%%rsna_speedy_cell <i>"
and a new first code cell registers that cell magic. On Kaggle's commit run (public 3-study dummy test, no
KAGGLE_IS_COMPETITION_RERUN flag) a 'light' build skips the unchanged Speedy cells and feeds the leg a stand-in
submission.csv (0.5 everywhere): ~2 instead of ~8 GPU minutes per dummy run on the submit-only main account. The hidden
rerun executes every cell through IPython's own run_cell (same compiler, so __future__ flags carry over; errors re-raised).
'gated-full' = same gate with light mode disabled, so this dummy run executes the Speedy cells THROUGH the gate.
Check: its dummy-run submission_speedy.csv and submission.csv must equal model111's (same weights, same 3 studies).
Local tests (real IPython 9.17 kernel via nbclient; Other/tests/test_light_gate.py): gated-full outputs identical to the
plain notebook (incl. __future__ annotations across cells and last-expression display), errors still stop the
notebook before the leg, light mode skips + writes the stand-in, the rerun flag or > 10 test studies force full mode.
Build: python scripts/run_speedy_fleet_v2.py --name model133 --title "RSNA Knee Model133 Gate Check Full" --weight 0.30 --leg v3 --dummy gated-full --fleet "[('model104', ['model104_best.pt'], 's96'), ('model75', ['model75_best.pt'], 's64'), ('model100', ['model100_best.pt'], 's96'), ('model98', ['model98_best.pt'], 's64')]"
Pushed 2026-10-04 07:53 (+07) on the main account (yodharinmonplub/rsna-knee-model133-gate-check-full v1).
RESULT 2026-10-04 08:01: dummy run COMPLETE (log span 8-426 s). Gate line "light=False (allowed=False, rerun flag=False, test studies=3)"; 0 cells skipped, 0 tracebacks; submission_speedy.csv AND submission.csv BYTE-IDENTICAL to model111's dummy run -> the %%rsna_speedy_cell gate leaves Speedy unchanged on Kaggle (same docker image, T4 x2). Never submitted (never_submit.txt).
