# RSNA Knee Abnormality Detection

## Current status (2026-09-30)

Best public AUC **0.947** (model103), rank 114 of 4,642 on 2026-09-30 (gold line = rank 19 = 0.956, silver = top 5% =
rank 232 = 0.944, bronze = top 10% = 0.943; top 0.961). Competition deadline: 2026-10-22.
Kaggle keys in Other/.env are named after the owners' e-mail names (never rename them): KAGGLE_API_KEY_YODHAIRNMONPLUB =
yodharinmonplub (SUBMISSIONS ONLY, per AGENTS.md), KAGGLE_API_KEY_WARIN = krittanutsomtuas (training on its free TPU),
KAGGLE_API_KEY_KRITTANUT = actuarialtutor (new; GPU works, TPU via API not yet, not joined to the competition yet).
model103 = model76 (the public "Speedy Raptors" notebook, 0.943, with its four Raptor CoAtNet views switched to this
project's own weights - model61) + a final rank-blend leg (w 0.30) over ONLY our two CoAtNet-2 384 px models (model75
on the 64-frame cache, model100 on the 96-frame cache).

| Model | Content | Public AUC |
|---|---|---|
| model61 | our pipeline: public ensemble + our Gemini-fine-tuned, head-retrained Raptor arms | 0.943 |
| model76 | Speedy Raptors (public 0.943) + our model61 Raptor weights | 0.945 |
| model89 | model75 alone (CoAtNet-2 at 384 px, trained from ImageNet) - our best own model | 0.939 |
| model101 | model100 alone (the same recipe on the 96-frame cache, stopped at epoch 12 of 18) | 0.937 |
| model81 / model88 | our small-model fleet alone / with model75 | 0.933 / 0.938 |
| model78 | model76 + fleet4 leg (model72, model71 x5, model73, model75) at w 0.30 | 0.946 |
| model93 / model94 | model76 + model75 as a fifth Raptor view / + the fleet4 leg as well | 0.946 / 0.946 |
| model95 | model78 at leg weight 0.40 | 0.945 |
| model99 / model102 | model78 + model86 (dense EfficientNet) / + model86 and model100 (possibly leg TIMEOUTS, see below) | 0.945 / 0.945 |
| model103 | model76 + leg over ONLY model75 + model100 (both CoAtNet-2 384) at w 0.30 | **0.947** |
| model105 / model106 | model103 at w 0.40 / model103 leg + model79 (Raptor-init CoAtNet) | 0.945 / 0.946 |

Leg time budget: the leg runs after Speedy (~7.2-7.4 h into the hidden run) and must end by notebook age 8.5 h, so it
has ~40-70 min; fleet_leg_v2 is all-or-nothing (a projected overrun silently keeps Speedy's file = model76's 0.945).
fleet_leg_v3 (2026-09-30) reads each DICOM once for all geometries and drops the LAST members of the fleet list first
on an overrun (run_speedy_fleet_v2.py --leg v3; the fleet order is the priority).

### Final submission guidance

Pick **model103** (0.947) and, as the hedge, **model94** (0.946: model75 as a fifth Raptor view plus the fleet4 leg) or
**model78** (0.946). The public LB shows 3 decimals; ties cannot be separated. If a private-LB shake-up worry dominates
(most metals are copies of two public notebooks), prefer the candidate that leans more on our own models. Any later
CoAtNet-2 384 model (model98 / model104 on the TPU) joins the model103 leg, never the small-model fleet.

### In progress (2026-09-30)

- Kaggle GPU quota (needed for every submission's dummy run) is used up on both team accounts until 2026-10-03 07:00
  (+07). Armed for then: model108 (model104 alone), model107 (leg model75 + model100 + model104), model111 (+ model98);
  then per-group readouts model114-119 (scripts/make_group_readout.py).
- TPU lane (krittanutsomtuas): model110 = 5-fold CoAtNet-2 384 s96 (OOF teacher), session 1 = folds 0-2 queued; session
  2 = folds 3-4 + model112 (FULL-DATA CoAtNet-2 384, train_xla.py full_data) chained by scratch tpu_chain.py; then
  model113 = pseudo-label student (scripts/make_pseudo_targets.py: 0.5 labels + 0.5 model110 OOF) - the forum's lever.
- model104 (TPU, gold 0.9306) and model98 (TPU, gold 0.9154) finished; model109 (third seed) was queued on the main
  account before the submit-only rule and started on 2026-09-30 19:50 despite a cancel - the user was asked to stop it.
- Vast credit 0.25 USD: no Vast training until a top-up.

### What moved the score

- Better Raptor weights, every step: Gemini relabels (0.941 -> 0.942), head retraining on full bags (0.943), and the
  same weights inside the stronger Speedy pipeline (0.945).
- Neutral or harmful on the leaderboard: extra views and single fully trained arms inside our blend (model26/30/67),
  label-prompt variants, pseudo-labels for frozen-feature heads, external data, flip TTA.
- Own models: full training of CoAtNet-2 at 384 px is the best recipe (model75 alone 0.939); small CNNs dilute it.
  Blending own models into model76 needs an own side near ~0.943 to pay (model80: 0.933 fleet at w 0.30 = -0.001).
- Next (see EXPERIMENTS.md): model79 full fine-tune of the public maxspan Raptor checkpoint; model90 (a second
  CoAtNet-2 384 seed, 18 epochs); model86 (running) fleet model on a denser 128-frame cache; model84 pseudo-labels.
  All need Vast credit (top-up requested 2026-09-27).

## Status on 2026-09-19 (our own pipeline)

Best verified public AUC **0.943** (model61). The public ensemble everyone forks scores 0.940-0.941; our
gain comes from retraining its CoAtNet views on better training labels (0.942) and then retraining only
their prediction heads on full window bags with plain BCE (0.943).

| Model | Content | Public AUC |
|---|---|---|
| model23 | public ensemble, unchanged | 0.940 |
| model24 / model26 / model30 / model33 / model35 | our v4-label fine-tuned CoAtNet arms, variants | 0.941 |
| model43 | all CoAtNet views retrained on Gemini relabels | 0.942 |
| model48 | model43 + our fine-tuned DINOv2 members | 0.942 |
| model61 | model43 with the four heads retrained (frozen backbones, all windows, plain BCE) | **0.943** |
| model49 | public ensemble + our members only | 0.940 |
| model53 | model43 with CoAtNet blend weight 0.60 -> 0.70 | 0.942 |
| model29 | five per-arm decodes | FAILED (runtime limit) |

### What moved the score, and what did not

- **Better training labels (worked).** The report-derived labels agree with the 58 expert gold labels
  at only 0.893 AUC. Re-extracting all 4,407 reports with Gemini 3.8 Flash reached 0.9035, lifted the
  CoAtNet blend CV from 0.9273 to 0.9301 and the leaderboard from 0.941 to 0.942.
- **Head retraining (worked, free).** Retraining only each arm's head on frozen features of all windows
  with plain soft-label BCE (scripts/run_head_retrain.py, free Kaggle T4, no gold used) lifted every arm
  (+0.002 to +0.005 gold AUC, 12 of 12 seeds), the blend CV from 0.9305 to 0.9326 and the leaderboard
  from 0.942 to 0.943. Mixing the 58 gold studies into that training hurts at every weight.
- **Label-structure fixes (neutral).** Hybrid labels (model60) raised label-vs-gold agreement by 0.006 but
  not the image model (0.9255 vs 0.9254): it already outranks its labels on those findings.
- **More windows per study (better arm, neutral blend).** 24 instead of 12 windows gave our best
  single arm (model51, CV 0.9280 vs 0.9254) but no blend gain.
- **Neutral or harmful:** fine-tuning the public DINOv2 members (model47/48/49), self-distillation
  (model46), deeper fine-tuning (model25), new families trained from ImageNet (model36 ConvNeXt 0.892,
  model42 CoAtNet-2 diverged), flip TTA, cross-label recalibration, and fine-tuning on the gold
  studies themselves (5-fold CV: 0.9014-0.9142 vs 0.9254).

### Measurement caveats

CV here is macro AUC on the 58 fully labelled studies, which our own models never train on. The
public DINOv2 members score 0.959 on those same cases while reporting a 0.838 holdout, so the gold
set is inside their training/selection data: CV is **not** comparable across model families, and the
transformer half of the ensemble can only be judged on the leaderboard.

### Final submission guidance

Pick **model61** (0.943) as one final submission and **model43** (0.942, same backbones, original heads)
as the second, unless a later model beats them.

Open leads (from the Kaggle forum read on 2026-09-19, see EXPERIMENTS.md): relabel with the host's
official annotation criteria (prompt v2 scores 0.9099 vs 0.9035 on gold; any new label set can now be
scored for free by a head retrain on the saved features); a fully trained small model at 224-288 px with
augmentation, 5 folds and out-of-fold pseudo-labels (what the 0.94-0.95 single models use); the 80-slice
geometry of the fine-spacing arm; more axial coverage for PF OA.

## Earlier statusModels1 through19 each have readme.txt. Best verified public AUC was **0.935**
(model19), versus0.932 for model16 and0.929 for model10, versus0.927 for models8/9,0.924 for model7, and0.790 for model6.

Model16 cloud fine-tuning completed. Weak-validation BCE selected epoch1;
its58-case development AUC is0.923080, versus0.919929 for the starting model12.
These cases are development evidence, not an untouched holdout or public score.
Model16 passed private T4 inference v1 on all three visible studies in20seconds.
Model19 blends50% model16 with one-sixth each of models10/12/15; development
AUC0.926221, with a paired interval versus model12 that includes zero.
Its private GPU inference v1 passed all three visible cases and blend checks. Models17/18 also passed private tests.

All Vast rentals are destroyed; artifacts are backed up. Recorded total campaign
cost is approximately$0.73 against the authorized$20 cap, including the failed
first model16 download attempt. No GPU instance is currently required.
The September10 UTC submission quota is exhausted; the next window starts
September11 00:00 UTC (07:00 Bangkok). Candidates include model12v1, model16v1,
and model19v1 after validation. No public scores yet for these candidates.
See EXPERIMENTS.md for the full history and limitations.

## Run

```bash
uv venv .venv
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python scripts/download_data.py
.venv/bin/python scripts/baseline.py
```

The downloader loads credentials from the ignored `.env` without printing them.
It supports the existing `KAGGLE_API_KEY` name by mapping it to
`KAGGLE_API_TOKEN` when the latter is unset. It uses the `kagglehub` competition
download API from AGENTS.md with individual file paths. Metadata downloads are
the default because this machine has approximately 44 GB free and no detected
NVIDIA GPU. Download a particular image with `--file <competition-relative-path>`.

## Initial experiment: 2026-09-10

- 4,407 training studies and 24,371 training series.
- 58 studies have all 12 explicit labels; 4,349 do not have explicit labels.
- Reports exist for training studies; the three visible test studies have no reports.
- Acquisition metadata logistic regression: **0.532676 local OOF macro ROC AUC**.
- Five folds group exact normalized duplicate reports. Features exclude reports
  and identifiers. Regularization C=0.1 was fixed before evaluation.
- Missing labels are not treated as negatives.

This small validation set cannot establish reliable improvements. Patient and
site identifiers are absent from the downloaded CSVs; report grouping does not
prevent all patient/site leakage. Acquisition metadata can encode site effects.
This is a diagnostic baseline, not a competitive MRI model or leaderboard score.

Generated outputs are in `artifacts/baseline/`: `metrics.json`, `folds.csv`,
`oof.csv`, and `submission.csv`. The submission file has been checked for schema,
finite probabilities, and probability bounds. It covers only visible test rows;
competition execution must regenerate predictions for the hidden test set.

## GPU pilot

The CPU baseline is model1. Models 2 and 3 learn distinct classifiers over
frozen DINOv2-small MRI features. Model4 is their predefined equal ensemble.
They use six MRI scan slots and neighboring-slice inputs, with soft report-derived
training targets. All 58 gold studies are held out, along with four unlabeled
studies sharing a normalized report. Training has 3,691 studies and weak validation
has 654. Checkpoint selection uses weak validation BCE, never gold AUC.

The pinned 11.12 GiB derived pixel cache is downloaded and verified on the rental.
The full original MRI archive is not downloaded to this PC. Source attribution,
data terms, and limitations are documented in SOURCES.md.

Cloud commands (use the existing recorded instance; do not launch a duplicate):

```bash
.venv/bin/python cloud/vast.py status
.venv/bin/python cloud/remote.py 'cd /workspace/rsna && ls artifacts'
```

Results were copied into `artifacts/remote/`. The watchdog destroys the rental
only after the final ensemble validates on real DICOMs and artifacts are backed
up. At the time/credit limit it stops compute and retains storage for
recovery. A stopped instance still incurs storage charges; destroy it after
recovering artifacts. The watchdog must remain running until cleanup finishes.
Credentials stay in ignored files; only a Kaggle download credential is sent to
the rental, never the Vast.ai, RunPod, or OpenRouter API keys.

Inference package creation is gated on trained weights and successful real-DICOM
validation. It generates a private Kaggle dataset bundle and an offline notebook;
neither generation nor a visible-test smoke run is a leaderboard submission.

## Follow-up experiments

1. Audit report languages and derive weak training labels with explicit handling
   of negation and uncertainty. Evaluate the extraction on the labeled studies;
   keep validation labels out of extraction tuning for their own fold.
2. Prepare a bounded image sample and compact per-series volumes, retaining
   sagittal, coronal, and axial views plus fluid-sensitive sequences.
3. Train an MRI model with weak labels and evaluate held-out explicit labels.
   Use GPU compute for this stage. Reports must never be required at inference.
4. Compare checkpoints and ensembles on fixed folds; record per-label AUC,
   uncertainty, runtime, and any actual leaderboard results separately.

Official competition pages:
[data](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/data),
[evaluation](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview/evaluation).

September11: model16v1 scored0.932 (56153815), new campaign best.
Model12v1 and model19v1 remain pending.

September11 04:51UTC: model19v1 scored0.935, new campaign best.
