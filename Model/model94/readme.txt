Model94: model93 + model78 combined - model76 (Speedy + our model61 Raptor weights, 0.945) with BOTH model75 uses:
(1) model75 as a fifth Raptor-side view (model93's time-boxed, fail-safe extension; w 0.23 = 18.7% of the Raptor side), and
(2) the final-level fleet leg of model78 (fleet4 = model72, model71 x5, model73, model75; rank blend at w 0.30).
Why: model78 scored 0.946 (new best); the two edits act at different points (mid-pipeline Raptor side vs finished
submission), so they may stack. Runtime: Speedy ~7.1 h + fifth view ~10-15 min (allowance 1500 s) + leg ~25 min, under
the leg's 9 h process-age guard; each addition falls back to the result without it on any error or time shortfall.
Build: python Other/scripts/run_model94.py [--push] (imports run_model93.py and run_model80.py; files copied here).
Kaggle: yodharinmonplub/rsna-knee-model94-coat75-view-plus-fleet4 v1 pushed 2026-09-27 ~22:20 (dummy run); submit after
07:00 with model91/92/93/95.
Dummy run VERIFIED 2026-09-28 06:48 (from outputs): model61 Raptor weights + model75 (fifth view, 0/3 without a value) +
fleet4 leg (3/3 studies, 8 checkpoints), no public Raptor loads, no errors. Submit after 07:00.
SUBMITTED 2026-09-28 ~07:05 (+07), submission 56623156.
Score: public 0.946 (scored 2026-09-28 14:33) = same as model93 (fifth view only) and model78 (fleet leg only); the two
uses of model75 do not stack visibly at 3 decimals.
