Model108: model104 ALONE on the leaderboard - fleet-only notebook with one member, model104 (CoAtNet-2 384 trained on
a free Kaggle TPU with model75's recipe on the 96-frame 336 px cache, 15 epochs, gold AUC 0.9306 = best own model),
read at the 's96' geometry. Compare: model75 alone 0.939 (model89, 64 frames), model100 alone 0.937 (model101, same
recipe on Vast, cut at epoch 12 of 18).
Built with Other/scripts/run_fleet_only.py --name model108 --fleet "[('model104', ['model104_best.pt'], 's96')]".
Dummy run and submission after the Kaggle GPU quota refresh (2026-10-03 07:00).
Dummy run VERIFIED and SUBMITTED 2026-10-03 07:10:53 (+07), submission 56784991.
Score: public 0.942 (scored 2026-10-03 08:03, ~55 min after submission) = BEST OWN SINGLE MODEL on the LB
(model75 alone 0.939, model100 alone 0.937; the public Speedy Raptors notebook 0.943). TPU-trained CoAtNet-2 384 on
the 96-frame cache works on the hidden set with GPU fp16 inference.
