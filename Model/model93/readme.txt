Model93: model76 (Speedy Raptors + our model61 Raptor weights, public 0.945) plus model75 as a FIFTH Raptor-side view.
model75 = this project's CoAtNet-2 trained from ImageNet at 384 px: public 0.939 ALONE (model89), gold 0.9204.

Why: model75 is as strong as a Raptor arm on the leaderboard and takes exactly the maxspan arm's input (336 px crops,
windows at 384 px), so it belongs in the Raptor probability mean more naturally than in a final-level blend (model80:
the 3-member fleet at w 0.30 cost 0.001). Share fixed in advance = model77's: w 0.23 next to Speedy's view weights
0.60/0.10/0.10/0.20 -> 18.7% of the Raptor side. Gold (notebook-way probability mean on our base) +0.0017 at that share:
noise level, and gold understated model75 by a wide margin.

Code: model77 v3's fifth-view extension (Other/scripts/run_model77.py, CPU-tested) with only the checkpoint, dataset and
geometry changed (Other/scripts/run_model93.py): inserted right after Speedy's Raptor stage; applies Speedy's exact
neutral fill to raptor_raw.npz; model75 on both GPUs (studies split even/odd); time allowance 1500 s and never past
T0 + TIME_BUDGET - 30 min (a study not reached keeps its four-view value); any error keeps Speedy's four-view
_raptor.csv (the submission then equals model76). Estimated hidden cost ~10-15 min (1,300 studies x 94 windows at 384 px
on two T4s); Speedy itself runs ~7.1 h of the 8 h TIME_BUDGET.
Mocked-namespace test (Other scratch test_m93.py): normal, allowance cutting in (24/41 without a value), zero allowance,
checkpoint read failure (fallback), dummy-sized - all pass.
Weights: model75_best.pt from yodharinmonplub/rsna-knee-fleet-weights.
Kaggle: yodharinmonplub/rsna-knee-model93-speedy-our-heads-plus-coat75 v1 pushed 2026-09-27 ~17:55 (dummy run); submit
after the daily quota resets (07:00 Bangkok).
Dummy run VERIFIED 2026-09-27 ~18:59 (from outputs; the status API was rate-limited): our three model61 Raptor weights +
model75_best.pt loaded (SHA-256 events), no public Raptor weights, "fifth view model75 added: ... 0/3 studies without a
model75 value (0 past the 1500s allowance) ... 15s on 2 GPUs", no errors, submission.csv written. Submit after 07:00.
Score: pending.
