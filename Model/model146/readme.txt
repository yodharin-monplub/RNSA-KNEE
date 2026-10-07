Model146: model135 ALONE (fleet-only, 128-frame 336 px student, s128x336 geometry, full windows): paired with model143 (the same student recipe and seed on 96 frames) = the input-density A/B on the LB (gold: 0.9334 vs 0.9347).
Build (from Other/): python scripts/run_fleet_only.py --name model146 --title "RSNA Knee Model146 Model135 Alone" --leg v6 --fleet "[('model135', ['model135_best.pt'], 's128x336')]"
Planned: dummy run + submission Sat 2026-10-10 from 07:01 (main GPU quota refresh).
Score: pending.
