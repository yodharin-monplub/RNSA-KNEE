Model143: model113 ALONE (fleet-only, no Speedy): is the pseudo-label student better than model104 alone (public 0.942, model108)? Full windows, fleet-only notebook with leg v6 (scripts/run_fleet_only.py --leg v6). Decides next week's TPU plan (second pseudo-label round or not).
Build (from Other/): python scripts/run_fleet_only.py --name model143 --title "RSNA Knee Model143 Model113 Alone" --leg v6 --fleet "[('model113', ['model113_best.pt'], 's96')]"
Planned: dummy run + submission Sat 2026-10-10 from 07:01 (main GPU quota refresh).
Score: pending.
