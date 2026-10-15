Model174: heliosli 50/50 (0.954) + our leg (m113,m104,m112,m132 s96) at w 0.15
Base: artifacts/public/heliosli__blend956_compactv1 = heliosli's Raptor + S6 50/50 notebook reading the compact bundle 209a32c8b50a6b1c from heliosli/rsna-knee-compact-v1 (alone 0.954 = model167; + LEG4 at w .20/.30 = 0.955 = model168/169; w .45 = 0.954 = model171). See Model/model167/readme.txt.
Leg (fleet leg v6, last cell; partial blend on a time shortfall; base kept on any failure): [('model113', ['model113_best.pt'], 's96'), ('model104', ['model104_best.pt'], 's96'), ('model112', ['model112_best.pt'], 's96'), ('model132', ['model132_best.pt'], 's96')]\n", at w 0.15.
Why: on this base our legs add +0.001 at w .20-.30 (0.955, out of the 646-team tie at 0.954); w .45 is too heavy. These variants add the round-2 students (model148/149, gold 0.9347/0.9341; model147 5-fold) and probe w .15-.30, hoping for 0.956 (silver line 2026-10-13).
Build (from Other/): python scripts/run_base_fleet.py --base artifacts/public/heliosli__blend956_compactv1 --name model174 --title "RSNA Knee model174 h954x50 plus leg w15" --weight 0.15 --fleet "<members above>"
Dummy run 2026-10-13 13:01-13:07 VERIFIED; tools/sunday_submit.py armed for 2026-10-14 07:01.
Score: public 0.955 (submitted 2026-10-14 07:01). All five 2026-10-14 variants (w .15-.30, 4-10 members incl. round-2 students) = 0.955, same as model168-170.
