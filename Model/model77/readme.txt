Model77: model76 (public Speedy Raptors + our model61 Raptor weights) plus ONE extra Raptor-side view - model73,
this project's fully trained CoAtNet-2 at 256 px (Model/model73).

Evidence: on the 58 gold cases, blended the notebook's way (probability mean of the Raptor views), model73 at the
pre-declared share of 18.75% of the Raptor side gave +0.0018 (0.9326 -> 0.9344, P>0 0.78) - just under this
project's +0.002 bar. It goes to the leaderboard anyway as a probe: the 58 cases have repeatedly failed to predict
the leaderboard (competitors report gold 0.925 models at 0.947 public) and model73 is the only new model family
that adds anything to the blend. The share is fixed in advance: w = 0.23 next to Speedy's four view weights
(0.60 / 0.10 / 0.10 / 0.20, sum 1.0), i.e. 0.23 / 1.23 = 18.7% of the Raptor side.

How: Speedy's Raptor stage writes the four views' raw probabilities to /kaggle/working/raptor_raw.npz and their
ranked probability mean to _raptor.csv, which is the only way the Raptor result reaches the rest of the pipeline.
Right after that stage completes, the extension loads model73 (backbone rebuilt at 256 px - its relative-position
bias is baked for 384 px), prepares windows with Speedy's own capacity-aware sampler (like every other view, so
~94 windows rather than the 62 of the gold test), recomputes the weighted probability mean with model73 as a fifth
view, re-ranks, and overwrites _raptor.csv. A study model73 cannot read keeps its four-view value.

Build: python Other/scripts/run_model77.py [--push]  (imports run_model76.py; both copied here). Weights: private
dataset rsna-knee-model73-weights (sha256 4a566636...), created under both yodharinmonplub and krittanutsomtuas
because new private datasets are not shared across the two team accounts. Pushed 2026-09-27 from krittanutsomtuas
(version 2; version 1 lacked access to the dataset), dummy-data run only.
Verification before submitting (scratchpad model77_tick.py): model76's checks plus the "[model77] fifth view
model73 added ... 0/3 studies unreadable" line and a raptor_checkpoint event for model73_best.pt.
VERSION 2 FAILED ON THE HIDDEN SET (submission 56587767, 2026-09-27: "Your notebook hit an unhandled error while
rerunning your code"). Cause, my bug: Speedy saves raptor_raw.npz BEFORE it neutral-fills the studies a view could not
read, so those rows are NaN in the file the extension reads; on the hidden set some studies are unreadable and the NaN
reached Speedy's strict finite check (rsna_rank01 -> rsna_finite raises). The 3 dummy studies are all readable, so the
dummy run passed. v2 state is archived in Other/artifacts/model77_notebook_krittanutsomtuas/v2_failed_hidden_rerun.

VERSION 3 (run_model77.py, tested locally against a mocked Speedy namespace, 5 cases):
- applies exactly Speedy's neutral fill (view mean over readable studies, else 0.5) before blending;
- runs model73 on BOTH GPUs (studies split even/odd, two model copies) - half the wall time;
- time-boxed: stops taking new studies after ALLOW_SECONDS (build option --allow-seconds; also never later than
  T0 + TIME_BUDGET - 30 min); a study without a model73 value keeps its four-view value (the blend then equals the
  four-view probability for it); runtime is binding on the ~1,300-study hidden set (model29 failed on time);
- fail-safe: any exception in the fifth view is caught, logged (rsna_event model77_fifth_view_failed) and Speedy's
  four-view _raptor.csv is kept, so the submission degrades to model76 instead of failing;
- removes its own (arm, study) identities from Speedy's preparation audit afterwards; writes _raptor.csv atomically.
The allowance is set from model76's hidden-set runtime before version 3 is pushed.
Score: pending (see score.txt once scored).
