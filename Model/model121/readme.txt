Model121: model120 with SIX leg members at stride 2 (fleet leg v4), priority order model110f0, model104, model75,
model109 (TPU, 64 frames, gold 0.9126; trained on the main account before the submit-only rule, used with the user's OK),
model100, model98 (TPU, 64 frames, gold 0.9154). Estimated leg time ~50 min (reading ~40 min + 6 members x 5.5-8.5 min
GPU in parallel); the v3/v4 time guard drops the last members first if the projection runs over.
Question: more strong members (at half the windows) vs the four of model120 / three of model107.
Build: python scripts/run_speedy_fleet_v2.py --name model121 --title "RSNA Knee Model121 Speedy Leg Six Stride2" \
  --weight 0.30 --leg v4 --fleet "[('model110f0', ['model110_fold0_best.pt'], 's96', 2), ('model104', ['model104_best.pt'], 's96', 2), ('model75', ['model75_best.pt'], 's64', 2), ('model109', ['model109_best.pt'], 's64', 2), ('model100', ['model100_best.pt'], 's96', 2), ('model98', ['model98_best.pt'], 's64', 2)]"
Dummy run + submission planned for Saturday 2026-10-03 (07:40 runner).
Score: pending.
