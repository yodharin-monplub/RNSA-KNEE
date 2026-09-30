Model108: model104 ALONE on the leaderboard - fleet-only notebook with one member, model104 (CoAtNet-2 384 trained on
a free Kaggle TPU with model75's recipe on the 96-frame 336 px cache, 15 epochs, gold AUC 0.9306 = best own model),
read at the 's96' geometry. Compare: model75 alone 0.939 (model89, 64 frames), model100 alone 0.937 (model101, same
recipe on Vast, cut at epoch 12 of 18).
Built with Other/scripts/run_fleet_only.py --name model108 --fleet "[('model104', ['model104_best.pt'], 's96')]".
Dummy run and submission after the Kaggle GPU quota refresh (2026-10-03 07:00).
Score: pending.
