# Experiment record

## 2026-09-10: first MRI pilot

Difficulty assessment from updated AGENTS.md: 4/5 for the engineering pipeline;
5/5 for obtaining a prize. Main constraints: 58 gold-labeled studies, multilingual
weak labels, domain shift, and a small compute budget. Inspection confirmed those
constraints. High reasoning effort is appropriate; no model switch is requested.

Latest user budget: $20 for this round. First job capped at $5 and 10 hours.
Vast instance 50461091: 1 × RTX 3090, 24 GiB VRAM, 60 GB disk,
quoted compute plus storage $0.1477778/hour. Account credit at launch $33.14746;
the account balance is not permission to spend beyond the task budget.

| Model | Approach | Status | Local metric |
|---|---|---|---|
| model1 | Acquisition metadata logistic regression | Completed | 0.532676 OOF macro AUC |
| model2 | Frozen DINOv2-small, linear study head | Completed | See below |
| model3 | Frozen DINOv2-small, attention + nonlinear head | Completed | See below |
| model4 | Fixed equal probability ensemble of model2/3 | Completed | See below |
| model5 | Fine-tuned last two DINOv2 blocks | Completed | See below |
| model6 | Validation-selected frozen/fine-tuned blend | Completed | See below |

Completed pilot results (58 gold cases, all held out from training):

| Model | Gold macro AUC | Bootstrap 95% interval | Weak validation BCE |
|---|---:|---|---:|
| model2 | 0.749623 | 0.7006–0.7942 | 0.481905 |
| model3 | 0.749041 | 0.6992–0.7952 | 0.476309 |
| model4 | 0.753488 | 0.7062–0.7998 | 0.473637 |
| model5 | 0.767302 | 0.7217–0.8086 | 0.475110 |
| model6 | 0.762663 | 0.7137–0.8059 | 0.468124 |

Model5 unfreezes the last two encoder blocks; epoch 3 of 4 was selected. Its
initial pixel-based validation loss matched the frozen-feature pipeline within
0.000003, confirming preprocessing/normalization consistency. Model6 selects a
0.5 fine-tuned / 0.5 frozen-ensemble blend using weak validation BCE. It is the
planned first submission; gold AUC did not choose the blend. Model5's higher
gold AUC is a descriptive result, not evidence of a statistically reliable lead.

Real visible-test inference passed for frozen and fine-tuned encoders with HF
offline mode and an empty HF cache. Seven integration/unit checks passed on the
rental; a separate local bootstrap-AUC test matched scikit-learn, including ties
and single-class resamples. Full results: artifacts/comparison/metrics.json.

Models 2–4 hold out all 58 gold studies. Model1 uses five-fold validation over
those studies, so the metrics do not constitute a controlled architecture comparison.
3691 weak training studies; 654 weak validation; 58 gold; 4 excluded report duplicates.
Unmentioned findings (p=0.5) have zero loss weight. Model selection uses weak
validation BCE. Public weak labels may themselves reflect development against
gold examples, so the gold set is not guaranteed to be an untouched external test.

Completed checks: report-group separation; cache ID/mask alignment; short-stack
slice sampling; physical DICOM ordering; absent-slot invariance; zero gradient
for unmentioned labels; finite zero-confidence loss; actual GPU encoder forward.
All cache shards undergo SHA-256 checks. Full DICOM-to-submission validation runs
after training. Actual cloud logs/results are copied into artifacts/remote.

No leaderboard result or prize is claimed. The private offline notebook must
successfully execute on Kaggle before any competition submission is scored.

Cloud cleanup completed: instance 50461091 destroyed; subsequent account query
found zero instances. Observed credit decreased by $0.0903966 (about $0.09);
the time-based estimate was $0.09131. These are observed/estimated charges, not
a final billing invoice. The latest $20 round limit remains in effect.

Private Kaggle weights uploaded:
https://www.kaggle.com/datasets/yodharinmonplub/rsna-knee-dinov2-pilot-weights
Private offline notebook version 1 launched:
https://www.kaggle.com/code/yodharinmonplub/rsna-knee-dinov2-pilot
Offline execution subsequently completed successfully on T4 (version 3).

Competition submissions accepted on 2026-09-10:
- Model6 DINOv2 ensemble: notebook version 3, submission 56140055.
- Model7 public CoAtNet reproduction: notebook version 1, submission 56140057.
Both visible-test outputs passed column, study-order, finite-value, and probability
range checks. Both competition submissions completed successfully, verified via
Kaggle API on 2026-09-10. Model6 public AUC: 0.790. Model7 public AUC: 0.924.
Model7 reproduces the publisher's reported score and improves on Model6 by
0.134 AUC. These are public leaderboard scores; private scores are unavailable.
CoAtNet is the stronger starting point for the next experiment.

Model8: full coverage CoAtNet experiment (2026-09-10)
Difficulty remains 4/5 after inspection; high reasoning is appropriate. The
main uncertainty is generalization from sparse labels and public evaluations.
Use model7's exact checkpoint and preprocessing with every valid-extent
three-slice window (at most 62) instead of 42 sampled windows. No other
prediction change. This tests coverage at up to 48% more backbone work.
Publisher discussion motivates the hypothesis, but does not validate our run:
https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/737696
Private offline T4 notebook version 1 launched:
https://www.kaggle.com/code/yodharinmonplub/rsna-knee-model8-coatnet-full-coverage
No additional Vast.ai rental; the $20 spending cap remains in effect.
Model8 version 1 completed successfully. Submission schema, study order, and
finite [0,1] probabilities passed validation. Visible-test script runtime was
23.2 seconds vs model7's 20.3 seconds (three studies, includes initialization).
Competition submission 56141935 COMPLETE; public AUC 0.927, verified via Kaggle
API on 2026-09-10. Full coverage improved public AUC by 0.003 over model7.
Private score unavailable. No additional Vast.ai training spend.

Model9: SWA CoAtNet full coverage (2026-09-10)
Difficulty 3/5, unchanged after repository inspection; current reasoning is
sufficient. Generalization of averaged weights is the main uncertainty.
Compared with model8, only the checkpoint filename changes to the public
raptor_ft_coatnet_v4_full_swa.pt. No architecture or preprocessing changes.
Python syntax checks passed; reviewed diff confirms the single-variable test.
Private offline T4 notebook version 1 launched:
https://www.kaggle.com/code/yodharinmonplub/rsna-knee-model9-coatnet-swa-full-coverage
No new Vast.ai rental. Model8's 0.927 remains the comparison baseline.
Model9 version 1 completed successfully. Submission schema, study order, finite
values and probability bounds passed validation. Visible-test script time:
21.5 seconds (three cases, includes setup; not hidden-test runtime).
Competition submission 56143463 COMPLETE; public AUC 0.927, verified via Kaggle
API on 2026-09-11. Model9 ties model8 at the reported three-decimal precision;
no demonstrated public score improvement from the SWA checkpoint alone.

Model10: fine-spacing checkpoint transfer (2026-09-11)
Difficulty 4/5 before and after inspection; current high reasoning is appropriate.
Main uncertainty: generalization and incomplete training-geometry documentation.
Kaggle dataset metadata confirms CC0 raptor_ft_coatnet_v9_full.pt. Test it with
the verified model8 full-coverage inference pipeline. Strict state/label loading
and visible-test inference must pass before submission. No new Vast.ai rental.
Private offline T4 notebook version 1 launched:
https://www.kaggle.com/code/yodharinmonplub/rsna-knee-model10-coatnet-fine-spacing
The $20 campaign budget remains in force; public pretrained checkpoint work
does not consume that rental budget. No score improvement is yet claimed.
Model10 version 1 completed successfully with strict checkpoint loading and
valid study order, columns, finite values, and probability bounds. Script time
on the three visible studies: 21.8 seconds including setup. Checkpoint metadata
reports resolution 384, epoch 12 and gold_auc 0.9031125 (publisher metadata only).
Competition submission 56148775 accepted; public score pending.

Model11: fixed diversity blend (2026-09-11 local date)
Previous goal turn made progress: model10 was validated and submitted. Current
Kaggle query confirms submission 56148775 remains PENDING, with no error.
Five account submissions already fall on 2026-09-10 UTC; do not consume another
slot until quota is available. This does not block independent notebook work.
Model11 uses 0.25 model8 + 0.25 model9 + 0.50 model10 probabilities, fixed before
model10's score. Decode once per study, then run three encoders sequentially.
AST checks confirm unchanged architecture, preprocessing, windows and prediction
functions. Private offline T4 version 1 launched; branch-output equivalence and
runtime checks are required before competition submission. No new Vast rental.
Model11 version 1 COMPLETE. Visible-case runtime 33.15 seconds; peak GPU memory
4.00 GB. Branch differences from standalone outputs max 0.00016945, 0.00015664,
0.00011980. Initial 1e-5 tolerance failed; passed documented float16-scale 1e-3
tolerance. Blend arithmetic error <=1.68e-8, schema/order/probability checks pass.
Candidate ready for scoring once daily quota permits; not yet submitted.

Gold development evaluation and model12 (2026-09-11 local)
Previous goal turn made progress: model11 built and validated. Model10 remains
PENDING on the current authoritative Kaggle query. No duplicate submission.
Difficulty stays 4/5 after inspection; current high reasoning is appropriate.
Run four public checkpoints on all 58 gold training cases using raw DICOMs on
Kaggle T4: models8/9/10 at 6%-94%, model12 at documented 2%-98% sampling.
This is development evaluation, not an untouched holdout: publisher development
used these cases. Analyze paired bootstrap intervals, without per-label tuning.
Predefine a three-family probability blend before inspecting results: average
of (model8+model9)/2, model10 and model12. Compare with fixed model11 and singles.
Gold notebook: yodharinmonplub/rsna-knee-coatnet-gold-evaluation, version 1.
Model12 inference notebook launched separately for visible-test validation.
No new Vast.ai rental or spending. Gold notebook never writes submission.csv.

Gold evaluation COMPLETE: 58/58 studies, 488.22 seconds, 4.34 GB peak GPU memory.
The previous goal turn was a verified wait on this specific running notebook.
Downloaded predictions passed UID-set alignment, probability and blend checks.
Development macro AUC: model8 0.916053; model9 0.915364; model10 0.909689;
model12 0.919929; model11 0.917394; predeclared three-family blend 0.919484.
Model12 minus model8 paired bootstrap 95% interval: [-0.00753, 0.01527].
Every candidate's paired interval overlaps zero: no robust superiority claim.
Prioritize model12's already-validated notebook for the next available scoring
slot because it leads this development comparison without multi-model cost.
Model11 remains a secondary candidate; no weight tuning on these gold cases.
Detailed report: artifacts/coatnet_gold_notebook/output/analysis.json.
The runtime decode_seconds field covers the standard span only; total elapsed
includes both spans and all four checkpoints. Do not treat it as total I/O time.

Independent model-family reconnaissance: downloaded public Mattia Angeli
knee-mri-fold-weights metadata and original DINOv3 notebook under artifacts/sources/dinov3.
Five 94MB fold files and a timm wheel, CC0 dataset listing. Source author warns
that later whole-ensemble versions were tuned to small public-score movements;
inspect and validate the standalone DINOv3 member, not those tuned blend weights.
No DINOv3 model has yet been built or executed by this project.

Model13 DINOv3 standalone reproduction (2026-09-11 local)
Prior goal turn made progress: completed CoAtNet development analysis and chose
model12 as next scoring priority. Difficulty remains 4/5 after code inspection.
Adapted the original Apache-2.0 notebook's standalone DINOv3 member, with source
attribution and separate backbone license. Preserve six plane/fat-suppression
slots, 130mm crop, 12%-88%, 16 slices, 336px and configured pooling/normalization.
Use strict full checkpoint loading, float32, one study at a time and tie-aware
fold rank mean. No parent ensemble's tuned per-label or blend weights.
Private notebook yodharinmonplub/rsna-knee-model13-dinov3-folds version 1 COMPLETE.
All five fold probabilities and final rank average passed visible-test checks.
Runtime 14.73 seconds inside inference main; 0.585 GB peak GPU allocated.
The source/config label order was verified in all five loaded configurations.
Launched separate gold notebook yodharinmonplub/rsna-knee-dinov3-gold-evaluation
version 1. Evaluate model13 and a predeclared equal rank blend with model12;
do not search weights on the 58 cases. Training overlap remains unverified.

Cross-family evaluation COMPLETE (2026-09-11 local). Model13 development AUC
0.868556; model12 0.919929; fixed equal rank blend model14 0.913246. Paired
model13-minus-model12 interval [-0.08322,-0.02048]; blend interval
[-0.02284,0.00859]. Do not prioritize model13 or model14 for submission.
Model13 gold inference: 111.45 seconds, 0.616 GB peak GPU allocation.
Model14 offline version 1 COMPLETE; branch and blend checks passed. Model12
remains the next scoring priority. Model10 remains PENDING; daily quota used.

Training-cache investigation: public knee-raptor-corpus and -ext cover all4407
studies but have old 44-slice 15%-85% geometry, not the dense wider-span inputs.
Do not start a paid rental assuming this cache matches model12. Saved metadata
under artifacts/sources/raptor_corpus. The native384dense checkpoint instead
documents 64 slices, 2%-98%, 140mm and native384px; a compatible next candidate.
Metadata/file list saved under artifacts/sources/raptor-knee-native384dense.

Model15 native384dense (2026-09-11 local)
Previous goal turn made progress: rejected model13/14 on development evidence
and identified model15's documented geometry. Difficulty remains 4/5 after
inspection; current model and high reasoning are sufficient.
Model15 changes model12's checkpoint and intermediate size to384, retaining
64 slices, 2%-98%, 140mm and full coverage. Syntax and source-diff checks pass.
Visible-test notebook version 1 launched. Initial gold notebook launch was
rejected by Kaggle's two-batch-GPU-session limit with no kernel created; retry
only after a confirmed running notebook completes. No paid rental needed.
Model10 remains PENDING; five submissions already used on current UTC date.
Model15 visible-test version 1 COMPLETE; schema/order/probability checks pass,
22.6 seconds inside inference script. Gold launch retried after terminal status
and accepted as version 1: yodharinmonplub/rsna-knee-native384-gold-evaluation.
No duplicate evaluation run. Comparison script: scripts/analyze_native384.py.

Dense cache pilot preparation (2026-09-11 local)
Previous turn made progress: model15 validation and development launch. Current
authoritative state still RUNNING for native384 gold; model10 still PENDING.
Organizer's published efficiency table downloaded to artifacts/efficiency_leaderboard:
top-three snapshot public AUCs 0.954, 0.948, 0.944; our team not present. Table
does not expose runtimes and may lag current submissions. Accuracy still matters.
Built CPU cache pilot for58 gold +6 weak studies using source-identical model12
ordered_files/read_study and constants. Four compressed16-study shards, SHA256,
safe-load round-trip equality checks, preserved UID order and stored geometry.
Purpose: test exact dense cache generation before full-corpus preparation and
Vast.ai fine-tuning. No paid rental yet. Private CPU notebook launched:
yodharinmonplub/rsna-knee-dense-cache-pilot version1. No submission.csv output.
Native384 gold evaluation subsequently COMPLETE. Model15 AUC0.916875;
model12 0.919929; predeclared equal probability blend0.921806. Paired blend
delta95% interval [-0.00570,0.01007]. Model12 remains first scoring priority.
The blend is a secondary fixed candidate, not evidence of a reliable gain.

Dense cache pilot COMPLETE (2026-09-11 local)
64 studies, zero failed slices, 93.74 seconds CPU preprocessing/compression.
333197263 compressed bytes; extrapolated full size ~22.94GB (nonrepresentative
pilot estimate only). Downloaded all four shards locally and verified SHA256,
safe npz loading, volume dtype/shape, mask equivalence and unique UID order.
AST comparison confirms ordered_files/read_study identical to model12.
Full corpus partitioned into1102/1102/1102/1101 studies, disjoint exact coverage.
Four private CPU jobs accepted, all version1:
yodharinmonplub/rsna-knee-dense-cache-part0 through part3.
No paid instance launched. Use completed manifests to size rental disk.

Label audit: dreaddevelopment/rsna-knee-labels has4349 weak studies and excludes
gold. Compared with Steven Lee v1 on shared IDs, mean absolute difference0.18336.
Strong opposite Effusion probabilities occur in1134 studies. Inspected examples
include minimal effusion scored0.8 versus0.2, indicating severity semantics.
Do not blindly average these target sets. Metadata and source labels saved in
artifacts/sources/raptor_labels and data/raptor_labels. Training target scheme
still requires a reasoned choice; no label consensus is yet adopted.

Dense loader preparation (2026-09-11 local)
Previous goal turn made progress: pilot validation and four full CPU cache jobs.
Rechecked all four jobs: RUNNING. Difficulty remains4/5 after repository review;
current model/high reasoning is sufficient. No duplicate jobs or paid rental.
Implemented cloud/prepare_dense_cache.py: validate hashes, geometry, UIDs,
metadata fingerprint, masks and full4407-study coverage, then unpack to mmap
arrays. Ready marker removed before rebuild so interruption cannot leave stale
completion. Implemented bounded-LRU knee/dense.py with explicit requested order.
Real64-study pilot unpacked under /tmp/rsna-dense-pilot; reversed study lookups,
exact pixel/mask equality and mmap eviction checks passed.
Launched private T4 cache-vs-raw inference check:
yodharinmonplub/rsna-knee-dense-inference-validation version1. Compares all58
cached gold predictions with the recorded raw-DICOM model12 predictions using
float16-scale tolerance1e-3. Must pass before paid full-cache training.

Cache-vs-raw inference check COMPLETE: all58 cases matched within0.000289382
absolute probability difference, below predeclared1e-3 half-precision tolerance.
Report: artifacts/dense_inference_validation_notebook/output/cache_inference_validation.json.

Model16 fine-tuning preparation (2026-09-11 local)
Previous goal turn made progress: verified loader and launched inference parity
check. Difficulty remains4/5 after inspection; current high reasoning suitable.
Downloaded/validated published v4 labels: all4407 unique IDs, finite[0,1] targets.
Do not average Raptor and v1 labels with different severity conventions. Use the
published v4 scheme as a distinct target experiment, with a frozen baseline
teacher penalty0.3 and small last-stage/head learning rates5e-6/1e-5.
Two epochs,12 random windows/study,batch2. Existing report-group partitions.
Include unchanged epoch0 in weak-BCE checkpoint selection; gold is descriptive.
Source/target/partition/cache hashes recorded. Full training not yet run.
Prepared scripts/train_model16.py with two-weak-case GPU smoke mode. Syntax and
backbone-stage API inspection pass; actual GPU backward test now launched:
yodharinmonplub/rsna-knee-model16-smoke version1. No gold optimizer updates.
No paid instance launched; plan at most$3 for the next rental within total$20.

Model16 smoke COMPLETE: loss0.769859,23,143,240 trainable parameters, two weak
training studies and no gold updates. Target/partition/config hashes match local
inputs. Readiness gate correctly refuses launch until all full-cache manifests
are present. Four CPU cache jobs still RUNNING at the latest check.
Cloud plumbing prepared: separate model16 state/backup roots,100GB disk,$3 cap,
manifest identity/hash checks on download, mmap conversion, rental-local smoke,
and full training. Watchdog starts automatically at model16 rental creation and
backs up then destroys on TRAINING_COMPLETE or RUN_FAILED. Remote flock prevents
duplicate execution. Code syntax/bash checks pass; actual deployment not run.

Model17 prepared and launched as private offline T4 notebook version1:
yodharinmonplub/rsna-knee-model17-coatnet-equal-blend. Fixed equal probability
blend of model12/model15, preserving each complete inference component verbatim
(AST literal/source identity checked). Development AUC0.921806; paired delta
interval includes zero. No public submission yet; model12 stays first priority.

Full dense-cache parts0/1/2 COMPLETE; manifests downloaded only, with1102 studies
each. Compressed sizes5.733/5.732/5.753GB. Failed selected-slice decodes0/2/0;
the recorded masks retain exact inference fallback behavior. Part3 still RUNNING
on authoritative Kaggle status; no replacement or paid rental launched. Model10
submission56148775 remains PENDING; best confirmed public score0.927.

Model17 v1 terminal ERROR: wrapper renamed nonexistent runtime.json after the
model12 branch succeeded. Removed rename; version2 accepted and running. Local
wrapper integration with real visible branch predictions verifies exact equal
probability output. Neither embedded branch changed.

All full cache parts COMPLETE. Readiness PASS:4407 unique studies,22.994GB
compressed, exact source metadata/geometry and smoke provenance match.
Model17 version2 COMPLETE:59.38sec/3 visible studies; final CSV verified against
its two branch outputs as exact equal probability blend. No public score yet.

Model16 rental50515330 created and deployed:RTX3090,$0.166296/hour,100GB,
6-hour deadline,$3 run allocation within$20 campaign cap. Initial SSH registration
retry fixed to reuse an already registered public key; no duplicate rental.
Persistent watchdog exec session36260 is live and backs up to remote_model16.
Remote run deployed; actual training awaits install/download/cache verification.

Model10 submission56148775 COMPLETE with public0.929, new confirmed best
(previous0.927). Gold-development ranking did not predict the public ranking;
retain model12 as next planned submission and do not treat gold as untouched.

Model16 attempt1 failed during sequential Kaggle download with SSL record-layer
error, before training. Watchdog backed up logs and destroyed50515330. Estimated
compute$0.231744; observed credit decrease$0.277219. State/results archived in
cloud/model16_attempt1 and remote_model16_attempt1. Campaign cost~$0.368529.
Downloader now streams four independent shards with SHA/size checks, atomic
files, five retries, and verified-shard reuse. Live paginated discovery/manifest
identity and simulated SSL recovery/reuse checks pass. Retry50524988 created,
RTX3090,$0.174444/hour,6h,$2.70cap to retain original combined$3 allocation.
Watchdog now runs in the launch exec session1172, avoiding detached sandbox
process lifetime issues. Retry still installing/starting; no trained checkpoint.

Model18 fixed equal blend of model10/model12/model15 launched as private T4
version1:rsna-knee-model18-coatnet-triple-blend. Weights fixed before development
calculation; macroAUC0.922287 on58 development cases (not untouched). Source
identity and wrapper arithmetic verified using existing actual visible outputs.
Model16 retry live: dependency install complete and new downloader actively
transferring cache shards. No full training results yet.

Prepared model16 private checkpoint packaging and notebook runner while download
continues. Packaging requires TRAINING_COMPLETE, an epoch>0 selected by lower
weak BCE, and backed-up checkpoint. Negative readiness check correctly refuses
unfinished training. Inference functions/geometry match model12 by AST; only
checkpoint location changes. Upload and real GPU inference validation remain.

Model18 private T4 version1 COMPLETE; runtime90.46sec. Final CSV matches exact equal component probabilities. Ready candidate, not submitted.

Additional public checkpoint audit via live Kaggle metadata: raptor-knee-arms
(three-backbone publisher0.915), arms-x(0.917), native384(0.926), widefov/fullspan
(publisher negative results0.917), cnn336(old~0.88). These are publisher claims,
not our submitted results. Metadata saved under artifacts/sources/<slug>.
No new GPU reproduction scheduled: existing dense checkpoints and ongoing
fine-tuning are better supported candidates. Download retry part0_056 recovered
and passed SHA verification on the live rental.

Model16 retry completed all downloads and unpacked276 verified shards covering
all4407 studies. Rental-local two-weak-case smoke PASS:loss0.786648,23143240
trainable parameters. Full trainer PID7600 is live, GPU100%, currently in initial
baseline evaluation (no epoch0 metrics yet). No full-run optimizer progress or
new validation score claimed. Remote provenance config matches local config.

Model16 full-run baseline: weak-validation BCE0.4572973400, gold-development
AUC0.9199286631 (matches prior model12 development AUC). Epoch0 checkpoint
backed up locally. Epoch1 confirmed beyondstep700 with finite loss/gradients,
GPU100%; two additional trainer processes are configured data-loader workers.
No epoch1 validation or higher-score claim yet.

Model16 epoch1 weak-validation BCE improved0.45729734 ->0.36400766.
History and selected checkpoint backed up locally. Epoch2 confirmed beyond
step1000; final gold-development AUC not evaluated yet. Lower weak BCE is
checkpoint-selection evidence, not proof of higher public ROC AUC.


## Model16 completed and model19 candidate — September10,22:50UTC

Model16 attempt2 completed both epochs and selected epoch1 by weak-validation
BCE: baseline0.4572973400, epoch1 0.3640076564, epoch2 0.3665926500.
Selected checkpoint development AUC0.9230802364 versus model12 0.9199286631.
Paired2000-bootstrap 95% difference interval[-0.0044945,0.0103964] includes zero.
Gold58 did not receive optimizer updates here, but upstream publishers used
these development cases; this is not untouched validation.

Vast50524988 was destroyed after backup, with empty live inventory confirmed.
Attempt2 credit decrease$0.36036955; prior recorded runs$0.36852915; total
approximately$0.72889870. Full artifacts: artifacts/remote_model16/model16.
Private model16 weights uploaded. Kaggle model16v1 COMPLETE: T4,3 visible
cases,20seconds, all12 output labels finite and in bounds.

Two fixed equal blends were evaluated without weight search. Model12+16
AUC0.9230078674; model18+16 AUC0.9262210133. The latter becomes model19:
weights model16=.5 and model10/model12/model15 each1/6. Paired95% interval
versus model12[-0.0007795,0.0138176] includes zero. GPU validation launched.
Analysis script scripts/analyze_model16.py; output artifacts/model16_development/analysis.json.
Leaderboard confirmation is pending; best confirmed score remains model10 .929.


September11 00:00 UTC quota reset: submitted tested v1 model12 (56153792), model16 (56153815), and model19 (56153836). All accepted; public scores pending. Two daily slots reserved for evidence-guided follow-up.


September11 01:20UTC: model16 submission56153815 COMPLETE, public AUC0.932. New campaign best, +0.003 versus model10 .929. Model12 and model19 remain PENDING. No new rental needed while these results are pending.

September11 01:31UTC: model12v1 COMPLETE, public AUC0.928. Model16v1 .932 improves its direct starting checkpoint by0.004. Model19 remains pending.

September11 04:51UTC: model19v1 COMPLETE, public AUC0.935. New best, +0.003 over model16 .932 and +0.006 over previous model10 .929. All three queued submissions scored. Two daily slots remain.

September11 05:02UTC: submitted model18v1 (56158870), fixed equal models10/12/15 ensemble. This measures the non-fine-tuned branch of model19 .935 against model16 .932. One daily slot remains; reserve until this comparison scores.

September11 09:05UTC: model18v1 completed at public AUC0.933, below model19
0.935 but above model16 0.932. Model20 begins exact-native384 fine-tuning of
model15. Its64-study384px Kaggle CPU pilot completed in120.4seconds and passed
local manifest, SHA, shape and unpack checks. Estimated full compressed cache
is29.38GB; compressed plus unpacked payload is70.97GB. Launched cached-versus-
raw GPU parity, two-case model20 GPU smoke, and four full CPU cache parts.
No Vast rental is active; paid training remains gated on all six jobs.


## 2026-09-16 round (20-hour session, $10 Vast cap per updated AGENTS.md)

Difficulty 5/5 for a prize after inspection: top public 0.957, rank-20 0.952,
3,832 teams, deadline 2026-10-22. Our best public remains 0.935 (model19/21).
Public notebooks now reach 0.941; nearly all fork mattiaangeli/bend-the-knee-to-the-dinosaurs
(DINO/RadImageNet family + Raptor CoAtNet arm + CoAt resgated, per-finding rank blend).
Its Raptor arm uses maxspan-v5 (0.6 + 0.1 reversed) and native384dense-v10 (0.1),
exactly the two checkpoints our model16/model20 fine-tuned.

Environment moved to Windows: new .venv, scripts/send_email.py (Purelymail SMTP),
kstatus/kwait/koutput/ksubmit helpers. Cloud tooling ported (os.devnull ssh config,
private-ACL key copy, msvcrt lock, ssh+tar backup instead of rsync); launch guard
now ignores other projects' instances on the shared Vast account (label prefix
rsna-knee). Campaign cap in model22 budget lowered to $10.

Model23: exact public notebook reproduction, private T4x2 v1 COMPLETE, submitted 56282564.
Model24: model23 with model16_best.pt in both maxspan arms and model20_best.pt in the
native384dense arm (namespace patch before main; rest byte-identical). v1 COMPLETE,
log confirms the substituted files, submitted 56282567. Scores pending.
Model22 Vast rental 51226491 (RTX3090, $0.174/h, 6h deadline, $3 cap) deployed from
Windows; watchdog running locally in the launch process.

Model22 COMPLETE (2026-09-17 local): weak BCE 0.472695 -> 0.380618 (epoch1 selected),
gold AUC 0.909689 -> 0.915761. Watchdog destroyed 51226491 after backup. Estimated
compute $0.342843; observed credit decrease $0.468578. Weights uploaded privately.
Model25 attempt1 (51235681) stuck in image loading ~25 min on machine 68093:
destroyed via cloud/abandon.py (archived, machine excluded). Attempt2 51237122,
RTX3090 $0.182/h, deployed. Config: last two CoAtNet stages, 3 epochs, cosine LR
(1e-5/2e-5 -> 10%), teacher 0.3, same v4 targets/partition/model16 cache.
Model26 = model24 + model22 fifth arm (w 0.15 fixed a priori), v1 COMPLETE with all
five arms loaded, submitted 56285541. Model23/24 still PENDING (~3h).

Model24 COMPLETE: public AUC 0.941 (56282567), new campaign best, +0.006 over 0.935.
Model23 (unmodified public baseline) still pending, so the effect of the fine-tuned
swap is not yet isolated at leaderboard precision.
Model25 attempt2 FAILED in epoch3: clip_grad_norm_(error_if_nonfinite=True) raised on an
fp16-overflow step (GradScaler would have skipped it). Fixed for future runs
(error_if_nonfinite=False). Epoch2 checkpoint (weak BCE 0.365205 vs model16 0.364008;
epoch1 0.369724) was saved. Watchdog full backup timed out repeatedly (host upload
~0.12 MB/s), so the rental uploaded the checkpoint directly to the private Kaggle dataset
yodharinmonplub/rsna-knee-model25-weights (190 s, sha256 e4efa39f...) via
cloud/remote_upload.py; then cloud/finish_manual.py destroyed 51237122. Estimated compute
$0.911540; observed credit decrease $2.081936 (includes another project's concurrent
instance model124-us2, recorded conservatively). A partial local best.pt from the timed-out
tar extraction was deleted. Gold predictions for model25 launched on Kaggle T4.

Model25 development (58 cases): model25 0.919413 vs model16 0.923080 (paired 95% CI
[-0.01224, 0.00529]); not used. Model27 = model16 recipe refit on train+weak_val, fixed
1 epoch (from model16's validated selection), seed 20260918. Trainer gained select=last /
include_weak_val_in_train. Cloud: host floors (disk_bw>=1000, inet_up>=300) in budget;
watchdog now auto-falls back to rental->Kaggle upload after two failed full backups, and
subtracts other projects' running instance cost (integrated dph) from the credit delta,
with this rental's own time estimate as a floor (model124-us2 at $0.445/h would otherwise
trip the $1.75 stop in ~2.8h). Model27 instance 51257213, RTX3090 $0.172/h, deployed.

Model23 COMPLETE: public 0.940 (56282564). Model24 0.941 vs its exact baseline 0.940: the fine-tuned model16/model20 swap is +0.001 at displayed precision (small, consistent with development gains). Model26 still PENDING.

Model28 = model16 recipe (seed 20260919) on public native384-v8 (raptor_ft_coatnet_v8_full_swa,
sha256 3db77388...), the last untuned Raptor arm. New s44 cache: 44 slots stored at the front of
the usual 64-frame arrays (frames 44+ zero/masked), 6%-94%, 384px; runners got --slots 44 (Kaggle
resolved slugs rsna-knee-dense-cache-384-6-94-{pilot,partN}-s44). Pilot 64 studies 72.7s; four
full parts COMPLETE, 20.46 GB compressed. Parity notebook (public Raptor reader/eval code verbatim)
on 58 cases: max pixel diff 0, max probability diff 0.0 (public windows and our training windows),
v8 development AUC 0.911638 -> PASS. prepare_dense_cache/train_model16 gained --slots 44.
Launch guard now allows concurrent project runs with distinct labels if prior spend + all active
caps + new cap <= $10 (3.66 + 2 + 2). Model28 instance 51259907 ($0.172/h) deployed.
Pre-declared model29 (before model26/27/28 results): model26 with model27 in the maxspan forward
arm (reverse arm keeps model16) and model28 in the native384-v8 arm; all weights unchanged.

Model27 COMPLETE: fixed 1-epoch full-data refit, gold dev AUC 0.924035 (model16 0.923080),
estimated compute $0.139, uploaded (rsna-knee-model27-weights).
Model28 COMPLETE: epoch2 selected (weak BCE 0.464822 -> 0.372541 -> 0.370465), gold dev AUC
0.911638 -> 0.915729, estimated $0.164; credit delta for the concurrent model27+28 window $0.548
booked to model28. Uploaded (rsna-knee-model28-weights). Conservative campaign spend ~$4.35.
Model29 v1 COMPLETE (all five arms loaded as declared), submitted 56291152.
Model26 still PENDING after ~8.5h (5-arm Raptor adds runtime; watch for timeout).

Model26 COMPLETE: public 0.941 (ties model24). Five-arm notebook scored in ~7.7h (no timeout).
Model30 = model29 + reversed native384dense (w .05) and v8 (w .10) views, grouped Raptor decode loop validated identical to the public loop (max diff 0.0). Submitted 56292155.

Model31 (v8 full-data refit, fixed 2 epochs, seed 20260920) and model32 (native384dense full-data
refit, fixed 1 epoch, seed 20260921) launched concurrently (instances 51266662, 51266707; $1 caps;
conservative prior spend $4.35). Pre-declared model33 (before model29/30/31/32 results): model30
with model31 in both v8 views and model32 in both dense views; all weights unchanged.

Model31 COMPLETE (v8 refit, 2 fixed epochs): gold dev AUC 0.917440; est $0.163.
Model32 COMPLETE (dense refit, 1 fixed epoch): gold dev AUC 0.918837; est $0.169.
Concurrent credit delta $0.476 booked to model31. Conservative campaign spend ~$4.99.
Both uploaded. Model33 v1 COMPLETE and submitted 56293478.
Pending: model29 (56291152), model30 (56292155), model33 (56293478).

Model34 = second-seed (20260922) repeat of model27's full-data maxspan refit, $1 cap. Pre-declared model35 (before any model29-34 score): model33 with the maxspan forward weight 0.60 split equally between model27 and model34 (two views in the same decode group).

Model34 COMPLETE (seed-2 maxspan refit): gold dev AUC 0.923966; est $0.110, observed $0.181. Uploaded. Conservative campaign spend ~$5.17.
Model35 v1 COMPLETE (8 views), submitted 56294688. Pending: model29, model30, model33, model35.

Model30 COMPLETE: public 0.941 (56292155), ties 24/26; finished before model29 (grouped decode reduces hidden-test runtime). Model29 still PENDING at ~8.9h.

Session wrap-up (2026-09-17 17:45 local, ~19.5 h of the 20 h limit)
Confirmed public: model23 0.940, model24 0.941, model26 0.941, model30 0.941 (best 0.941).
Still pending at wrap-up: model29 (56291152, ~9.2 h, possible runtime-limit risk because the
public per-arm loop re-decodes every arm), model33 (56293478), model35 (56294688).
If model29 errors, keep using the grouped-decode notebooks (model30/33/35). Final-selection
candidates: model30 (confirmed 0.941), then model33/model35 once scored.
No Vast rentals running; account credit $21.65; conservative campaign spend ~$5.17 of $10.
Takeaway: fine-tuned Raptor arms give consistent development gains (+0.002 to +0.006 AUC each)
but only +0.001 on the public ensemble. The ~0.016 gap to the top needs new strong model
families, not more CoAtNet arm tuning.

2026-09-17 19:30: model29 submission FAILED with exceeded runtime (5 per-arm decodes). Runtime is a binding constraint; all later notebooks use the grouped decode. New goal from user: push toward 0.96 CV and public LB; run until told to stop.

2026-09-18: user raised the Vast.ai campaign cap to 20 USD (message in session). New budget files use campaign_cap_usd 20; conservative spend before model36 ~5.17.

Model37check: grouped loop with 6-thread chunked DICOM prefetch reproduces the public per-arm loop exactly (max diff 0.0). Use GROUPED_MAIN_PREFETCH for new notebooks.
Model36 launched (51307509): ConvNeXt-small (fb_in22k_ft_in1k_384) trained from ImageNet at 336px on the maxspan cache, 10 epochs, bs 4x12 windows, grad checkpointing, LR 5e-5/1e-3 warmup+cosine, v4 targets, selection by weak BCE, gold logged per epoch.

Model33 COMPLETE 0.941 (56293478); model35 COMPLETE 0.941 (56294688). Every CoAtNet-side variant ties at 0.941: plateau.
Model36 attempt1 failed: rental could not reach the HF hub for ImageNet weights (cost ~0.04). Weights now come from private dataset yodharinmonplub/rsna-knee-backbone-convnext-small (sha256 17e4aa3c...); trainer loads via pretrained_cfg_overlay; downloader verifies sha. Attempt2 instance 51309915 deployed.

2026-09-18 findings: v4 report labels vs gold AUC only 0.893 (image models 0.927). Gold labels are image reads; label quality is the bottleneck. LOO cross-label recalibration on gold hurts (0.84-0.88). OpenRouter relabel tests on gold (scripts/llm_relabel.py): Sonnet 5 0.901, Gemini 3.8 Flash 0.909 (default), 0.903 (low effort), 0.896 (minimal), DeepSeek 0.786, GLM-5.2 free 0.874. New OpenRouter key: 10 USD limit per updated AGENTS.md. Full relabel with Gemini 3.8 Flash low effort running (all_g38f_low).
Model36 COMPLETE: ConvNeXt-small from ImageNet, gold 0.892; blending lowers CV; not used. Est $0.33.

Flip TTA gold test (model16, 58 cases, scripts/run_flip_tta_gold.py): plain 0.923109 (matches model16), flip coronal/axial 0.920214, average 0.922523, flip all 0.907717. No gain; not used.

Full Gemini 3.8 Flash relabel (low reasoning) of all 4,407 reports COMPLETE: gold AUC 0.9035 vs v4 0.8927 (+0.011); cost 6.32 USD of the 10 USD OpenRouter limit. Targets file data/llm_relabel/targets_g38f_low.csv.
Launched four label-swap runs (same recipes as the originals, only targets changed): model38 (maxspan, vs model16 0.9231), model39 (native384dense, vs model20 0.9192), model40 (v8, vs model28 0.9157), model41 (fine-spacing, vs model22 0.9158). Instances 51336366/51339034/51339059/51339073; caps 1.0-2.98; prior spend 5.55.

Label-swap results (CV, 58 gold): model38 maxspan 0.925370 (model16 0.923080), model39 dense 0.920484 (model20 0.919209), model40 v8 0.919671 (model28 0.915729), model41 finespacing 0.915969 (model22 0.915761). Weighted blend of the new arms 0.9301 vs 0.9273 for the old arms; old+new mixed 0.9289 (worse). All four uploaded.
Model42 attempt1 FAILED in smoke: CoAtNet relative-position bias is fixed to 384px inputs (441 vs 576 tokens) - retrying at res 384, batch 2. Cost 0.09.
Model43 (all views = Gemini-relabel retrains) v1 COMPLETE, submitted 56313516.

Self-distillation experiment: model44/model45 train on complementary halves of the non-gold studies (split by report_hash parity, fixed 1 epoch, Gemini targets) and write out-of-sample predictions for the other half. Planned model46: retrain on targets blending those predictions with the Gemini labels (on the 58 gold cases, image+label blends scored 0.936 vs 0.927 image-only and 0.893 labels-only). Caveat: the starting checkpoint was trained by its publisher on all studies, so the predictions are not fully out-of-sample.
Model42 attempt2 (CoAtNet-2 from ImageNet-12k at 384px, Gemini labels) running as instance 51348240.

Model42 ABANDONED: CoAtNet-2 from ImageNet diverged (BCE ~0.495, gold 0.58->0.52 over 3 epochs) with the recipe that worked for ConvNeXt; stopped at epoch 4, est 0.43. From-scratch families dropped as a direction.

Model44 (fold A) CV 0.923693, model45 (fold B) CV 0.925448. Model46 self-distillation CV 0.923974 vs model38 0.925370: no gain, not used. Averaging maxspan variants (38/44/45/46) in the blend also lowers CV (0.9294 vs 0.9305), so model38 alone stays.

IMPORTANT (2026-09-18): the public DINOv2 members score 0.9589 macro AUC on the 58 gold cases, far above our honest CoAtNet CV (0.9301). Their manifest reports a 0.838 holdout, so the gold studies were part of their training/selection data. Gold CV is therefore contaminated for that family and cannot be compared across families; our CoAtNet numbers stay honest (gold never trained on). Consequence: model47 (fine-tuning those members on the Gemini relabels) cannot be judged on gold - only on the leaderboard.
Model47 attempt1 failed at import (download_dense asserted a dense-cache run name; now checked lazily), cost 0.01. Attempt2 instance 51369520: cache prepared (4,407 studies, 72 shards), member fingerprints reproduce within 5e-6, smoke PASS.

Model47 COMPLETE: 20 members fine-tuned (mean gold 0.9314 -> 0.9307, uninformative because gold is in their training data), uploaded as rsna-knee-model47-members. Model48 (members + Gemini arms) submitted 56318312; model49 (members only) submitted 56318305 to isolate the effect against model23 0.940.

Watchdog fix: a full backup is now confirmed against a remote file listing (name+size) before the rental is destroyed - tar can exit 0 on a truncated stream, which is how model47 lost 12 of 20 member files locally (they survived only because the rental had already uploaded them to Kaggle).

Model43 COMPLETE: public 0.942 (56313516) - NEW BEST, first move off the 0.941 plateau. Better training labels improved both CV (0.9273 -> 0.9301) and LB (0.941 -> 0.942).

Rad head experiment (scripts/run_rad_heads.py): built a 224px 3-plane/8-slice cache with the notebook's own Rad geometry, extracted frozen RadImageNet ResNet50 features once for all 4,407 studies, and trained two 5-fold _RadHead bundles on the Gemini relabels (folds by report group, gold excluded). Honest gold AUC of the five-head mean: 0.8778 (seed A) and 0.8752 (seed B); fold AUC vs labels 0.866-0.882. These are well below our CoAtNet blend (0.930) and cannot be compared fairly with the public heads (unknown training data, probably gold-inclusive), so swapping them in is a gamble - deferred until model49 shows whether replacing public components with ours helps.

Model48 COMPLETE 0.942 (= model43) and model49 COMPLETE 0.940 (= model23 baseline): fine-tuning the public DINOv2 members on our labels is neutral. All measured gains come from the CoAtNet arms trained on better labels. Direction closed: do not replace further public transformer components (the trained Rad heads are kept but unused).

Gold fine-tune experiment (scripts/run_gold_tune.py, free Kaggle GPU, 5-fold over the 58 gold cases): fine-tuning model38 on 4/5 of gold and scoring the held-out fold gives 0.9014 (head only, 6 epochs) and 0.9142 (last stage + head, 4 epochs) against 0.9254 for the untouched checkpoint. 46 studies overfit; direction closed.

Model53 = model43 with base CoAtNet blend weight 0.60 -> 0.70 (published per-finding exceptions kept), submitted 56327619. Single pre-declared step, no weight search.
Model51 (24 windows/study, 2 epochs) and model52 (plain BCE) launched as maxspan recipe variants vs model38 CV 0.925370. Model52 attempt1 was stuck loading and abandoned (0.13 est); attempt2 running as 51414824 after rebuilding its state file by hand (interrupted archive left a stale instance id).

Recipe variants: model51 (24 windows/study, 2 epochs) CV 0.927971 vs model38 0.925370 (+0.0026, best single arm); model52 (plain BCE) 0.925695 (neutral). Retraining the other three arms with 24 windows as model54 (dense), model55 (v8), model56 (fine-spacing).

2026-09-18: Vast.ai account credit reached ZERO; the provider stopped every instance on the shared account (ours plus two other projects'). model54 (dense, 24 windows) and model56 (fine-spacing, 24 windows) were killed mid-training and only history/provenance survived; we destroyed the stopped rentals. model51 (maxspan 24w, CV 0.927971) and model55 (v8 24w, CV 0.920749) completed and are uploaded. Our own conservative spend estimate is ~10.5 USD of the authorized 20.

Arm-swap evaluation on CV: base (model38/39/40/41) 0.9305; maxspan->model51 0.9294; v8->model55 0.9310; both 0.9296; maxspan as mean(38,51) 0.9298. Paired 95% CI for maxspan->51 minus base is [-0.0037, 0.0034]. Individually better arms do not improve the blend, so model43's arm set is kept and no submission slot was spent on a swap.

Model53 COMPLETE 0.942 (= model43): raising the base CoAtNet blend weight 0.60 -> 0.70 is neutral; published weights kept.
Vast.ai credit restored to 11.24 USD, so training can resume (our conservative campaign spend is ~10.5 of the authorised 20).

Multi-scale TTA is not possible for this CoAtNet: the relative-position attention bias is fixed to 384px inputs (576 tokens), so evaluating the same checkpoint at 336 or 448 raises a shape error (same constraint that broke model42). Idea closed.

Model58 (two trainable stages, 24 windows): weak BCE 0.282365 (best) but gold 0.926078 (< model51 0.927971); no blend containing it beats the base 0.9305. Capacity is not the limit. Launching model59: one stage, 24 windows, 4 epochs with cosine decay - the last untested dimension (training length).

Model59 (4-epoch cosine, 24 windows) was stopped by our budget guard after epoch 2 (cap 1.0 USD inherited from model51). Weak BCE flat across epochs 1-2 (0.287046 / 0.287058), so training length is not the limit either. Rental destroyed; est 0.79 USD. Conservative campaign spend ~11.4 of 20 USD.

Public-vs-ours blend test (free, 58 cases): public checkpoints alone 0.9252 (v5 0.9199, v10 0.9170, v8 0.9116, v9 0.9097), ours alone 0.9305, mixtures 0.9298-0.9303. Keeping the public checkpoints alongside ours does not help either. The CoAtNet side is saturated: every variant tried (labels aside) lands within ~0.002 CV and 0.000 LB.

2026-09-19 research pass (user request: other ways to raise the score - forum, public code, web - plus a bug review).
- Web/GitHub write-ups of this competition describe the same lineage as ours (LLM report labels + CoAtNet multi-window MIL, ~0.93 gold / 0.939 LB); no technique we have not tried. One public repo warns that sending report text to hosted LLM APIs may conflict with the competition data rules; the public pilkwang label dataset was itself produced with a hosted Claude labeler (api_labeler.py), so the practice is common, but the rules page still needs reading (Kaggle forum/rules/data pages are client-rendered and need browser tools, not enabled this session).
- luonmayman's public "v5-train" notebook (the DINOsaur V4 training code) trains a slot head on frozen DINO features with the gold studies mixed in at weight 14, a pairwise rank loss, a nested per-label blend and 3-source ensemble labels. Gold-in-training is the one cheap idea we have not tested on our arms: launched rsna-knee-head-gold-cv (free Kaggle T4) = model38 backbone frozen, all 62 windows/study encoded once, then the attention head retrained with 5-fold CV over the 58 gold studies (A: Gemini labels only; B/C/D: plus fold-train gold at weight 14 with/without rank loss, weight 4). Held-out gold never trained on.
- Label sources vs the 58 gold studies (scripts/compare_label_sources.py): gemini_low 0.9035, lee_v4 0.8927, lee_v2 0.8873, lee_v1 0.8780, gpt56sol 0.8352 (pilkwang_v2 covers 57/58 and raptor excludes gold - dropped). Probability or rank averages never beat Gemini alone (best gemini+gpt56sol 0.9037).
- Per-finding hybrid (scripts/make_hybrid_targets.py; a finding switches source only if another set is >= 0.02 better on gold): Effusion -> lee_v4 (0.877 vs 0.846), Baker's -> lee_v2 (0.946 vs 0.898); hybrid 0.9101 vs 0.9035. Selected on the same 58 cases, so optimistic; written to data/llm_relabel/targets_hybrid.csv, not trained on yet.
- Bug review: scripts/train_model16.py read end to end (masking, window sampling, confidence weights, teacher term, refit/fold logic, selection, metrics) and the notebook builder patches - no defects found; the grouped decode was already validated bit-identical to the public loop.

Why the two hybrid findings differ (report inspection, 58 gold): Gemini scores reports that never mention a Baker's cyst 0.02-0.20 (noise; the one silent gold positive sits at 0.02, below explicit "no cyst" reports), and scores "mild/small effusion" ~0.45, tied with silent reports, because the prompt says small effusions may be judged physiologic - but the gold readers mark about half of the mild effusions and few silent reports positive. Lee v4 ranks explicit negative < silent (0.25) < mentioned, the structure the gold shows. Tie-break added to make_hybrid_targets.py: prefer lee_v4 over lee_v2 when within 0.005 (v2 gives silent reports 0.50 = zero confidence weight). Final hybrid: Gemini + Lee v4 Effusion and Baker's, gold 0.9099.
2026-09-19 Kaggle forum/rules/data pages read through the browser tools (user enabled /chrome). Findings:
- RULES: host's pinned thread "Use of Commercially Hosted LLMs" (discussion 733965) explicitly permits sending report text to hosted LLM APIs for label extraction ("You can use LLM API, such as those from OpenAI, to read the reports to generate the labels"); not treated as prohibited private sharing (rules 2.6.b). External public knee-MRI datasets with click-through research licences (MRNet, OAI, fastMRI+, SKM-TEA) are also allowed by the host. Two final submissions; winners must release training code + weights; hand-labelling of TEST data is forbidden (training data is not mentioned).
- OFFICIAL ANNOTATION CRITERIA (host overview, discussion 733343; two MSK radiologists + adjudicator, image reads, "on the fence" = NEGATIVE): ACL = high-grade partial (>50% fibres) or complete tear, mucoid/degeneration/thickening negative; MCL = high-grade partial or complete ACUTE tear, low-grade sprains and chronic changes NEGATIVE; meniscus = signal definitely reaching the surface on >=2 images or morphologic abnormality, intrasubstance degeneration negative; OA (each compartment) = moderate/large area (~>=1 cm) of high-grade (>50% thickness) cartilage loss; Effusion = MODERATE or LARGE; Synovitis = inflammation and thickening of the synovium; Baker's = MODERATE or LARGE collection; Contusion = marrow oedema from impact without a fracture line; Fracture = ACUTE cortical break or fracture line. Our Gemini prompt deviates: it counts MCL sprains grade 1-3, any-size Baker's cysts, "meaningful" chondromalacia/osteophytes for OA, and any fracture type incl. subchondral insufficiency. Forum caveat (T. Aindow): readers are noisy near the thresholds ("small effusion" is positive ~40% of the time on gold), so the LLM should estimate the probability that the image readers marked the finding, informed by the criteria, not apply them as hard rules.
- Silence statistics (S. Lee, discussion 733932), gold-positive rate when the report is silent vs when it speaks: Synovitis 0.34/0.76, PF OA 0.21/0.41, Baker's 0.03/0.44, Medial OA 0.00/0.36; the Effusion field predicts gold Synovitis better than the Synovitis field.
- WHAT THE TOP TEAMS DO ("Best single-model score", 735304; "Plateau at 0.903", 740610): single small models reach 0.938-0.950 LB - Scott Willis (1st) small ResNet, single fold 0.938, 5-fold 0.947, 5-minute inference; Archit Konde (23rd) single-fold 2.5D CoAtNet at 224px 0.950 after label work ("my OOF predictions were already closer to the radiologists than the extracted labels"); Tucker Arrants (4th) 0.943 5-fold at 224px; tennogh 0.942 at 288px with OOF pseudo-labels; Yann Majewski 0.936 small ResNet ("label work +0.015, encoder pre-training, training longer with more regularization"); Raymond Yuen 0.938 full-train CoAtNet 288 ("pseudo labeling and blending different labels"). Tucker's recipe: ResNet34/EffNet-B0 at 224/288, simple pooling, validate 5-fold against the report labels (noise threshold ~0.003, not the 58 gold, not the LB), then ablate the standard and MRI-specific augmentation stacks one at a time, then backbones/resolution; he doubts DINO inits beat ImageNet CNNs. => Our from-ImageNet attempts (ConvNeXt 0.892, CoAtNet diverged) failed on recipe (no augmentation, short schedule), not because full training is expensive; none of our runs used augmentation.
- GEOMETRY BUG ON OUR SIDE: the Raptor author now documents the fine-spacing corpus (v9) as 80 slices/study over 2-98%, 140 mm, 336 px, slots 22 sag-fluid/18 sag/15 cor-fluid/10 cor/15 axial, 78 windows, LB 0.932 only at 78 windows; "density is the lever" (44 slices over 2-98% 0.917, 64 slices 0.928, 80 slices 0.932). Our fine-spacing arm (model10/22/41, notebook arm finespacing-ft22) runs at 64 slices, 6-94%, 62 windows, because the dataset page omitted the geometry when model10 was built (see SOURCES.md). No newer Raptor weights have been published (all dreaddevelopment datasets date from August; the 0.936-0.938 single model is unreleased).
- No DICOM-metadata shortcut (733517: metadata-only 0.65 random folds, 0.60 scanner-grouped). A few training studies contain both knees.

Model60 = model38 recipe and seed with the hybrid targets (only the targets differ; ready gates check that the other ten findings are identical to model38's). Probe for ~1 USD: retrain the other three arms only if model60 beats model38's gold 0.925370. Instance 51482148 (RTX 3090, $0.189/h, cap 1.0), prior spend 11.38 of 20; account credit only 5.60 USD because two other projects' rentals are running on the shared account.

2026-09-19 browser access (rules + forum read; details in docs/official_label_criteria.md). Rules 2.6: external data allowed if publicly available and equally accessible at no/minimal cost. Pinned host thread 733965: hosted commercial LLMs explicitly permitted for processing reports. Host overview thread 733343 publishes the annotators' criteria; our v1 relabel prompt contradicted them (MCL any sprain vs high-grade acute only; Baker's any size vs moderate/large; effusion, ACL, OA thresholds; acute fracture; borderline = negative).
Prompt v2 (scripts/llm_relabel.py --prompt v2, written from the official text, not tuned on gold): Gemini 3.8 Flash low effort on the 58 gold reports scores 0.9099 vs v1 0.9035 (ACL .996, MCL .989, LatOA .849, PFOA .949, Effusion .872, Synovitis .827, Fracture .915 up; MedMen .953, MedOA .921 down; Baker's flat .897). Cost 0.05 USD. Full relabel (~6.5 USD) needs more OpenRouter credit - asked the user by email.
New user goal: find external knee MRI data (user offers 100+ USD for training if data is found); dataset access terms being verified.

Model60 COMPLETE (session 9f0f45): hybrid labels are NEUTRAL on the maxspan arm. Selected epoch 1 (weak BCE 0.459834 -> 0.307343 -> 0.309642), gold AUC 0.925523 vs model38 0.925370 (+0.0002, paired bootstrap 95% CI [-0.0015, +0.0020]; scripts/compare_gold_predictions.py model38 model60). Per finding: Baker's +0.0054 (0.9420 -> 0.9475), Lateral Meniscus +0.0025, Effusion -0.0012, Synovitis -0.0024, rest within 0.0015. A +0.0064 gain in label-vs-gold agreement on two findings did not move the image model, so the other three arms are not retrained and nothing is submitted. Rental 51482148 destroyed after verified backup; estimated compute 0.234 USD (conservative campaign spend ~11.6 of 20). Weights kept locally only (artifacts/remote_model60).
Diagnostic from the same table (Tucker Arrants' test: where does the image model fail to beat its own labels?): image model vs Gemini-label agreement on gold - PF OA 0.855 vs 0.929, Lateral OA 0.820 vs 0.824, Lateral Meniscus 0.894 vs 0.890, Synovitis 0.853 vs 0.815, Fracture 0.919 vs 0.899; everything else the image model wins by 0.03-0.13 (Effusion 0.980 vs 0.846, Baker's 0.942 vs 0.898). So PF OA (and the lateral compartment) is modelling/geometry headroom - the Raptor geometry gives the axial plane only 12 of 64 slots - while Effusion/Baker's label fixes cannot help an image model that already outranks those labels.

Head gold-CV kernel (rsna-knee-head-gold-cv, free Kaggle T4), session 9f0f45: v1 died of GPU OOM (feature extraction ran without inference_mode - my bug). v2 extracted all features (4,407 x 62 windows x 1024, 2.9 h), reproduced model38's gold AUC exactly from the cached features (0.925370 - sanity check PASS), then crashed in arm B on a second bug of mine (per-finding 2-D weights for the weak rows concatenated with 1-D gold weights). Before the crash, arm A (head only, retrained from model38's head on the Gemini labels, frozen features, all 62 windows per study, batch 128, 6 epochs, lr 3e-4) scored 0.9305 and 0.9332 (two seeds) against 0.9254 for model38's own head - and because of the same weight bug its loss was effectively PLAIN soft-label BCE, not the confidence-weighted BCE. No gold is used in that arm, so the +0.005-0.008 is an honest reading (58 cases, so still noisy): the fine-tunes only moved the public head at lr 1e-5 for two epochs on 12 random windows; a properly retrained head on full bags looks better. A failed kernel keeps no output files, so the features were lost. v3 pushed: weights fixed (n x 12 everywhere), arms A0 plain BCE / A confidence-weighted / B gold14+rank (DINOsaur recipe) / C gold14 / D gold4 / E gold40 / F plain+gold14, three seeds each, seed-averaged OOF predictions and a final head per arm saved, experiments wrapped in try/except so the kernel completes and its features stay usable as a data source. If A0/A holds up, the same head retrain can be applied to all four arms for free.

Head gold-CV v3 COMPLETE (58 gold, mean of 3 seeds +/- std; gold arms are 5-fold CV over gold, weak-only arms never see gold): model38's own head 0.9254 | A0 weak, plain BCE 0.9317 +/- 0.0010 (final all-weak head 0.9308) | A weak, confidence-weighted 0.9281 +/- 0.0007 | D conf + gold weight 4 0.9248 | F plain + gold 14 0.9229 | C conf + gold 14 0.9210 | B conf + gold 14 + rank loss (DINOsaur recipe) 0.9197 | E conf + gold 40 0.9163.
=> (1) Mixing the gold studies into head training HURTS, monotonically in the gold weight, with or without the rank loss; direction closed (consistent with the earlier gold fine-tune overfit). (2) Retraining only the head on full bags helps, and plain BCE beats the confidence-weighted BCE the whole project has used: A0 per finding vs model38: Synovitis +0.022, Lateral OA +0.021, Baker's +0.018, Lateral Meniscus +0.009, MCL +0.007; Medial Meniscus -0.005, Contusion/Fracture -0.003.
Blend check (scripts/blend_head_test.py; CoAtNet side = weighted probability mean 0.70/0.15/0.30/0.15 of models 38/39/40/41, reproduces 0.9305): maxspan head -> A0 gives 0.9331 (paired bootstrap 95% CI of the difference [-0.0015, +0.0076]); -> A 0.9322; -> D 0.9295. Single-arm CI for A0 vs model38 [-0.0015, +0.0152]. Promising, not proven on 58 cases - same size as the Gemini-label step that moved the LB by +0.001.
Follow-up (free Kaggle T4, recipe frozen as A0, nothing tuned on gold): scripts/run_head_retrain.py <arm> encodes all windows with the arm's frozen backbone, asserts that the arm's own head reproduces its known gold AUC, trains 3 seeds + a final head (seed 7) on all non-gold studies, and writes a full checkpoint model61_<arm>_best.pt with the head replaced (verified through Classifier.predict_features). Pushed: rsna-knee-head-retrain-maxspan (reuses the v3 features) and -dense; v8 and fine follow. Planned model61 = model43 with the four retrained heads, to be submitted only if the four-arm blend CV beats 0.9305 by a clear margin.
Head retrain, maxspan (rsna-knee-head-retrain-maxspan v1, 14 min with reused features): own head reproduces 0.925370 exactly; seeds 0.9300 / 0.9305 / 0.9313 (mean 0.9306); final head (seed 7) 0.930821 - identical to the v3 final head, so the recipe is deterministic; checkpoint model61_maxspan_best.pt verified through Classifier (max abs diff 3.6e-7). Kernel -v8 pushed; -dense still encoding.
Head retrain, v8 (rsna-knee-head-retrain-v8 v1, 2.3 h): own head reproduces 0.919671; seeds 0.92228 / 0.92230 / 0.92236 (mean 0.9223); final head 0.921834 (+0.0022 vs model40, smaller than maxspan's +0.0054 but the same sign on every seed); checkpoint verified (2.1e-7). Kernel -fine pushed.
Head retrain, dense (rsna-knee-head-retrain-dense v1, 3.6 h): own head reproduces 0.920484; seeds 0.9242 / 0.9235 / 0.9239 (mean 0.9239); final head 0.923640 (+0.0032 vs model39); checkpoint verified (4.9e-7).
Interim blend (scripts/blend_model61.py, final heads for maxspan/dense/v8, original model41 for fine-spacing): 0.9305 -> 0.9331, +0.0026, paired bootstrap 95% CI [-0.0023, +0.0084], P(diff>0) 0.84. By finding: Synovitis 0.852 -> 0.876, Medial OA +0.005, Baker's +0.005, Lateral OA/Lateral Meniscus +0.004; Contusion -0.008, Effusion/Medial Meniscus -0.003. The two extra retrained arms add nothing beyond the maxspan head alone (0.9331 either way).
Model61 builder added to scripts/run_model23_24.py (model43 path + HEAD_RETRAIN_ARMS: every view loads model61_<arm>_best.pt from /kaggle/input/notebooks/yodharinmonplub/rsna-knee-head-retrain-<arm>/); builds and compiles; not pushed until the fine-spacing kernel finishes and the user has been emailed.
Head retrain, fine-spacing (rsna-knee-head-retrain-fine v1, 2.8 h): own head reproduces 0.915969; seeds 0.9182 / 0.9178 / 0.9178; final head 0.918063 (+0.0021 vs model41); checkpoint verified (3.0e-7).
ALL FOUR ARMS: maxspan +0.0054, dense +0.0032, v8 +0.0022, fine +0.0021 - 12 of 12 seeds above their arm's original head. Final four-arm blend (scripts/blend_model61.py): 0.9305 -> 0.9326, +0.0020, paired bootstrap 95% CI [-0.0029, +0.0078], P(diff>0) 0.78; Synovitis 0.852 -> 0.875, Medial OA/Baker's +0.005, Contusion -0.011, Fracture/MCL -0.003. Not significant on 58 cases, the same order as the Gemini-label step (+0.0028 CV -> +0.001 LB). Model61 notebook v1 pushed (rsna-knee-model61-retrained-heads); user emailed before submission per AGENTS.md.
Model61 v1 COMPLETE on the visible test (69 s Raptor side): log shows the four model61_<arm>_best.pt files found under /kaggle/input/notebooks/yodharinmonplub/rsna-knee-head-retrain-<arm>/, all seven views mapped to them, four decode groups, no FALLBACK lines; submission.csv 3 rows x 13 columns, values in [1/3, 1]. No stop from the user; SUBMITTED as 56354360 (2026-09-19 16:10 local). Score pending (~8 h).
MODEL61 COMPLETE: public AUC 0.943 (56354360, scored in ~5.4 h) - NEW PROJECT BEST, +0.001 over model43/48 (0.942). Second lever that moved the leaderboard after the Gemini relabel; CV predicted it (+0.0020 blend, every arm and seed up). Cost 0 USD (free Kaggle T4). README.md status table and final-submission guidance updated (model61 + model43). The saved per-arm features (kernel outputs rsna-knee-head-gold-cv / rsna-knee-head-retrain-{dense,v8,fine}: window_features_fp16.npz) make any further head-level experiment (new labels, external-data heads, loss variants) a ~15-minute free run.

2026-09-19 external data (session 841c25, user goal: find external knee MRI data; user budget 18 USD for steps 1-2; OpenRouter route dropped by the user, so the v2 full relabel is NOT run).
Sources in data/external (git-ignored; all research / no-redistribution licences, accepted by the user; host thread 733965 does not exclude click-through licences): MRNet 1,250 exams x 3 planes (Redivis API, md5-verified; labels abnormal/ACL/meniscus); KneeMRI Rijeka 917 sagittal exams (Zenodo 14789903, md5 verified; ACL 3 grades); fastMRI+ boxes for 1,173 coronal exams (meniscus tear 663 exams, cartilage partial 588 / full 122, effusion 142, periarticular cysts 161, ACL high 101 / low 153, MCL 125, fracture-or-contusion 119); OAI MOAKS tables (10,325 knee-visits read: effusion-synovitis 0-3, popliteal cyst, ACL, cartilage per subregion, meniscus) - OAI images not fetched (800-980 GB per visit package; needs a custom NDA package).
Model62 (not a model; cache build): Vast 51553607 RTX 3090, 3.2 h, est 0.53 USD. Streams the two fastMRI single-coil knee .tar.xz archives (78.0 + 16.1 GB) through a Range-resuming reader (first attempt died when S3 closed the stream), keeps the annotated exams' reconstruction_esc as uint8 320x320 stacks: 1,172 of 1,173 kept -> private Kaggle dataset yodharinmonplub/rsna-knee-external-fastmri-plus-cache. MRNet repacked to npz shards -> private yodharinmonplub/rsna-knee-external-mrnet-cache (first attempt dropped one exam per split because Redivis exports the first CSV row as a header; count assert caught it before upload).
KneeMRI: 917 volumes unpacked with a numpy-only restricted unpickler, uint8 shards -> private yodharinmonplub/rsna-knee-external-kneemri-cache (ACL 690 healthy / 172 partial / 55 complete, matches the paper).
MRNet zero-shot transfer (scripts/run_mrnet_transfer.py, free T4, 29 min; model61 maxspan arm, never trained on MRNet; 22 slices per plane, 3-slice windows inside each plane): ACL vs MRNet ACL-tear 0.908 all / 0.977 on the 120 radiologist-read validation exams (MRNet paper ~0.965); max(medial, lateral meniscus) vs meniscal tear 0.872 / 0.870 (paper ~0.847); max of all 12 vs abnormal 0.901 / 0.930 (paper ~0.937). => domains are compatible, MRNet labels are usable supervision, but our model already matches MRNet-trained models on ACL/meniscus, so MRNet/KneeMRI are unlikely to add much; the external value should be in fastMRI+ boxes (cartilage, effusion, cysts, marrow edema) and OAI MOAKS (effusion-synovitis, Baker's, per-compartment cartilage). MRNet window features saved as kernel output mrnet_window_features_fp16.npz. External-data spend so far 0.53 of the user's 18 USD.

2026-09-20 external-data step 2 (scripts/compare_step2.py; knee/external.py; trainer flags augment / external). Same recipe and seed as model38 (maxspan CoAtNet, last stage + head, Gemini labels, 2 epochs, weak-BCE selection).
Model63 = + GPU augmentation (rot +-10, scale +-10%, shift +-6%, gamma/contrast/brightness, no flips): gold 0.925133 vs model38 0.925370, diff -0.0002, CI [-0.0031, +0.0028]. Augmentation is neutral for this light fine-tune. Vast est 0.26 USD.
Model64 = model63 + auxiliary supervision from MRNet train (1,130), fastMRI+ (1,172, exam labels + box-derived window labels) and KneeMRI (917) through separate heads, weight 0.5, one external batch per step. External tasks were learned (exam BCE 0.69 -> 0.30, window BCE 0.12) and weak-val BCE improved slightly (0.2877 vs 0.2919) but gold 0.923925: -0.0012 vs model63, CI [-0.0067, +0.0044]; -0.0014 vs model38. Per finding: Lateral OA +0.004, Medial Meniscus +0.003, Contusion +0.005, ACL +0.002; Synovitis -0.007, Fracture -0.006, Lateral Meniscus -0.005, MCL -0.009. Mean(model38, model64) 0.9256. Vast est 0.44 USD.
=> In the last-stage fine-tune regime neither augmentation nor 3,200 externally labelled exams move gold AUC; this regime is saturated (every variant since model16 lands within +-0.003). External data could only matter in a regime that actually re-learns features (full training), which is untested. External-data spend so far 1.23 of 18 USD (model62 0.53, model63 0.26, model64 0.44).
Note: background watchdogs were killed by host memory pressure (another project using 13 GB); replaced by cloud/guard_once.py run from a 20-minute tick.

2026-09-20 full-training test (scripts/compare_fulltrain.py). ResNet34 a1_in1k, 256 px, all layers, 25 epochs, GPU augmentation, plain soft-label BCE on the Gemini labels, 16 windows x batch 8, cosine, selection by weak BCE; gold logged per epoch.
Model65 (competition data only): gold 0.7290 (ep1) -> 0.9028 (ep15) -> 0.9104 (ep25, selected). First from-ImageNet run that works (model36 ConvNeXt without augmentation 0.892; model42 diverged): the forum recipe is right that augmentation + long schedule was what we lacked. 3.3 h on a 3090, est 0.67 USD. Per finding vs model38: better on Baker's 0.971 vs 0.942, Synovitis 0.873 vs 0.853, Lateral OA 0.838 vs 0.820, Medial OA 0.978 vs 0.974; much worse on MCL 0.830 vs 0.968, Lateral Meniscus 0.858 vs 0.894, Contusion, Effusion, Fracture.
Model66 (same seed + MRNet / fastMRI+ / KneeMRI aux heads, weight 0.5): gold 0.8962 (ep23); -0.0142 vs model65, 95% CI [-0.0313, +0.0012]; worse on 10 of 12 findings (MCL 0.776, Lateral Meniscus 0.815, Fracture 0.867). At matched epochs it was never ahead after epoch 10. Est 0.72 USD.
Blend on gold (rank average with the model43 4-arm CoAtNet blend 0.9305): + 0.2 model65 0.9314, + 0.2 mean(65,66) 0.9320 (diff +0.0014, CI [-0.0036, +0.0060]), + 0.4 model65 0.9326. Small, inside noise.
DECISION for the user's question "should we start a 100 USD training job?": NO. (1) External data is neutral in the fine-tune regime (model64) and harmful in the full-training regime (model66): the auxiliary tasks compete for a small backbone's capacity and their label definitions/sequences differ from the competition's. (2) Full training works but costs 0.7 USD per model, so even a 5-fold x 2-backbone programme is ~7 USD - there is nothing a 100 USD job would buy. (3) The measured bottleneck remains label quality and per-finding geometry (MCL, PF OA), not compute. External-data phase spend: 2.62 of 18 USD (model62 0.53, model63 0.26, model64 0.44, model65 0.67, model66 0.72).

2026-09-20 (session 9f0f45) head-retrain tooling generalised for other label sets: scripts/run_head_retrain.py <arm> <TAG> <TARGETS_FILE> builds kernel rsna-knee-head-retrain-<TAG>-<arm>, reuses the arm's saved window_features_fp16.npz (so ~15 min instead of ~3 h), trains the frozen A0 recipe on TARGETS_FILE (found in the attached datasets) and writes <TAG>_<arm>_best.pt; HEAD_RETRAIN_DRY=1 builds without pushing. Waiting for the official-criteria relabel (targets_g38f_low_v2.csv, peer session, OpenRouter topped up by the user). Pre-declared rule for a v2-head submission: four-arm blend CV >= 0.9326 + 0.002, or all four arms above their model61 heads (0.9308 / 0.9236 / 0.9218 / 0.9181); otherwise logged as neutral.

2026-09-20 layout change (AGENTS.md): root = AGENTS.md only; Data/{competition,labels,external}; Model/modelN; Other/ = everything else. ROOT for all code = Other/; Other/data/* and Other/modelN are junctions/hard links (scripts/layout_data_links.py; Other/REPRODUCE.md).
2026-09-20 relabel v2 COMPLETE (user reversed the earlier decision and topped up OpenRouter to 20 USD total): all 4,407 reports re-read by Gemini 3.8 Flash (low reasoning) with the host's official criteria (scripts/llm_relabel.py --prompt v2; 64 rate-limit failures re-run, 0 v1 fills). File data/llm_relabel/targets_g38f_low_v2.csv (scripts/finish_relabel_v2.py). Cost 6.66 USD (OpenRouter usage 13.08 total, 6.92 left).
Gold agreement v1 -> v2: macro 0.9035 -> 0.9099. Up: Lateral OA .824->.849, Effusion .846->.872, PF OA .929->.949, Fracture .899->.915, Synovitis .815->.827, MCL .982->.989, ACL .991->.996. Down: Medial Meniscus .969->.953, Medial OA .935->.921. Flat: Lateral Meniscus, Baker's, Contusion. The stricter criteria lower the mean probability a lot (macro 0.263 -> 0.177; MCL 0.156 -> 0.054, PF OA 0.38 -> 0.22, Baker's 0.29 -> 0.16); v1~v2 correlation is lowest for MCL 0.69, Baker's 0.78, PF/Lateral/Medial OA 0.85-0.88.
Handed to the other RSNA session for the frozen head-retrain evaluation (A0 recipe, four arms; pre-declared rule: submit only if the four-arm blend CV beats 0.9326 by >= +0.002 or all four arms improve).

Relabel v2 head-retrain evaluation (session 9f0f45; free Kaggle CPU kernels rsna-knee-head-retrain-v2-<arm>, ~10-14 min each - the account's two GPU batch slots were held by another project, so scripts/run_head_retrain.py gained HEAD_RETRAIN_CPU=1; features reused; A0 recipe unchanged). Final head gold AUC, model61 (v1 labels) -> v2 labels: maxspan 0.9308 -> 0.9264 (seeds 0.9265/0.9251/0.9256), dense 0.9236 -> 0.9181, v8 0.9218 -> 0.9132, fine 0.9181 -> 0.9178. Four-arm blend 0.9326 -> 0.9262 (-0.0063, paired bootstrap 95% CI [-0.0187, +0.0030], P(diff>0) 0.12; scripts/blend_heads_compare.py v2). By finding in the blend: MCL 0.9705 -> 0.9433 and Baker's 0.9746 -> 0.9130 carry the loss; PF OA +0.008, Fracture +0.006, Contusion +0.005, Effusion +0.001; rest flat.
=> RESULT: the official-criteria labels agree better with gold as LABELS (+0.0064) but train WORSE image heads (-0.0063 blend). Turning low-grade MCL sprains and small Baker's cysts into confident negatives removes exactly the visual concept the model needs to learn; at test time ranking by severity does the rest. Label-vs-gold agreement is therefore not a reliable proxy for label usefulness - the head retrain (free, ~15 min) is. Rule outcome: no v2 submission.
Follow-up (free): label ensemble mean(v1, v2) (scripts/make_v1v2_mean_targets.py; label-vs-gold 0.9113 - best label set so far; per finding between or above both, e.g. Fracture 0.924, Baker's 0.909, Lateral Meniscus 0.896) uploaded as targets_g38f_low_v1v2mean.csv; kernels rsna-knee-head-retrain-v12-<arm> running under the same rule.
Label ensemble mean(v1, v2) RESULT (first push errored: kernels started before the dataset version with the new file existed; re-pushed): final head gold AUC model61 -> v12: maxspan 0.9308 -> 0.9299, dense 0.9236 -> 0.9233, v8 0.9218 -> 0.9216, fine 0.9181 -> 0.9193; blend 0.9326 -> 0.9323 (-0.0003, CI [-0.0036, +0.0028], P(diff>0) 0.44); mean of both heads per arm 0.9327. NEUTRAL - no submission.
=> Label work via the LLM prompt is saturated for this pipeline: hybrid (model60), official-criteria v2 and mean(v1, v2) all raise label-vs-gold agreement (0.9035 -> 0.9099-0.9113) and none improves the image heads (0.9255 / 0.9262 / 0.9323 vs 0.9326). v1 Gemini labels + retrained heads (model61, LB 0.943) stay the best. Remaining levers are on the image side (fully trained small models - model65 0.910 single, per-finding geometry for PF OA / MCL, the 80-slice fine-spacing geometry) and out-of-fold pseudo-labels from the image models themselves rather than better report reading.

2026-09-20 user chose option 1 for the Kaggle GPU-quota block (30 h/week account quota exhausted, shared with the user's ARC project; model67 = model61 + model65 arm w0.30 is built but cannot be saved until quota returns): keep preparing candidates on Vast.
Model68 / model69 = model65 recipe (ResNet34 from ImageNet, 25 fixed epochs, augmentation, plain BCE, Gemini v1 labels) on fold A / fold B of the non-gold studies (report-hash parity), each predicting the other half: gold 0.8664 / 0.8903 (half data; est 0.34 + 0.54 USD). True out-of-fold predictions for 4,345 studies (4 partition-excluded studies keep v1 labels) -> data/llm_relabel/oof_resnet34.csv (scripts/build_oof_targets.py). OOF vs v1 labels: AUC 0.79 (MCL) - 0.91 (Effusion, Medial Meniscus), r 0.44-0.80, means match (0.27).
Gold sanity (not used for training): labels alone 0.9035; 0.5 labels + 0.5 mean(68,69) 0.9236; 0.5 labels + 0.5 CoAtNet blend 0.9384 -> image evidence adds a lot to the report labels. As ensemble members the fold models add nothing (coat + 0.2 mean(65,68,69) 0.9310 vs 0.9305; 0.4 -> 0.9265).
Pseudo-label targets data/llm_relabel/targets_v1_oof50.csv = 0.5 v1 + 0.5 OOF (gold rows = v1) uploaded to rsna-knee-gemini-relabels; four CPU head-retrain kernels pushed (scripts/run_head_retrain.py <arm> oof50 targets_v1_oof50.csv, HEAD_RETRAIN_CPU=1): rsna-knee-head-retrain-oof50-{maxspan,dense,v8,fine}. Pre-declared rule unchanged: use only if the four-arm blend beats model61's 0.9326 by >= +0.002 or all four arms improve (model61 heads: maxspan 0.9308, dense 0.9236, v8 0.9218, fine 0.9181). The other RSNA session has exited; this session runs the check.

2026-09-20 OOF pseudo-label head retrain (targets_v1_oof50 = 0.5 Gemini v1 + 0.5 out-of-fold ResNet34; CPU kernels rsna-knee-head-retrain-oof50-<arm>; scripts/blend_heads_compare.py oof50): FAILS the pre-declared rule. Final head gold AUC model61 -> oof50: maxspan 0.9308 -> 0.9250, dense 0.9236 -> 0.9175, v8 0.9218 -> 0.9180, fine 0.9181 -> 0.9140; blend 0.9326 -> 0.9251 (CI [-0.0160, +0.0004], P(diff>0) 0.03). Loss concentrated where the ResNet34 teacher is weak: MCL 0.9705 -> 0.9410, Fracture 0.9208 -> 0.8917, PF OA, ACL, Lateral Meniscus; only Effusion/Lateral OA up marginally.
Lesson: a pseudo-label teacher must be at least as strong as the student on every finding. Half-data ResNet34s (0.87-0.89 gold, MCL 0.83) drag 0.92-0.93 CoAtNet heads down, even though label+OOF blends score higher on gold than labels alone (0.9236 vs 0.9035) - the second time label-side gold agreement failed to predict student quality (after v2). The forum recipe works for teams whose OOF models ARE their best models. Direction closed for this teacher; a usable OOF teacher would need 5-fold CoAtNet-class models trained from ImageNet (~10+ USD, uncertain). Phase spend 3.50 of 18 USD; nothing running on Vast. model61 (0.943) remains best; model67 waits for the Kaggle GPU quota.

2026-09-22 Geometry gold test (scripts/run_geometry_gold.py, CPU kernels rsna-knee-geometry-gold-<arm>; scripts/compare_geometry_gold.py; pre-declared rule: adopt if blend >= +0.002, or arm >= +0.002 with blend not worse). Geometry is inference-only; model61 head-retrained checkpoints re-run on re-decoded gold DICOMs.
MAXSPAN arm (model61_maxspan_best.pt): same-kernel control at the notebook geometry (64 slices, 2-98%, 62 windows) = 0.9308, identical to the saved head-retrain prediction (test path validated); blend 0.9327.
- raptor_80_02_98 (slots 22/18/15/10/15, 78 windows): arm 0.9311 (+0.0003), blend 0.9337 (+0.0011, 95% CI [-0.0037, +0.0066], P>0 0.65). Per finding: Synovitis +0.018, MCL +0.011, Baker's +0.011, Lateral OA -0.021, Lateral Meniscus -0.016. => NO (below rule).
- ax80_02_98 (extra 16 slices on axial, 18/14/12/8/28): arm 0.9287 (-0.0022), blend 0.9322 (-0.0005); PF OA -0.009. => NO; more axial slices do not fix PF OA for this arm.
Fine-spacing arm kernel still running.

2026-09-22 FINE-SPACING arm geometry (same test, model61_fine_best.pt): same-kernel control at the notebook geometry (64 slices, 6-94%, 62 windows) = 0.9181, identical to the saved head-retrain prediction; blend 0.9326.
- raptor_80_02_98 (the author's documented v9 geometry: 22/18/15/10/15, 2-98%, 78 windows): arm 0.9252 (+0.0071), blend 0.9340 (+0.0015, 95% CI [-0.0007, +0.0042], P>0 0.91). Per finding: MCL +0.057, Lateral Meniscus +0.029, Fracture +0.025, Medial OA -0.008. => ADOPT (arm >= +0.002, blend not worse). The "geometry bug" was real for this arm.
- s64_02_98 (only the span widened): arm 0.9162 (-0.0019) => no; the gain comes from density, not span.
- ax80_02_98 (80 slices, extra on axial): arm 0.9151 (-0.0030), blend +0.0010 => no.
=> model70 = model61 with the fine-spacing arm switched to 80 slices / 2-98% / 78 windows (scripts/run_model23_24.py FINE80). Built 2026-09-22; pushed from teammate account krittanutsomtuas when a GPU slot is free (dummy-test commit run only).
model67 v1 on the teammate account failed: the public CoAtNet child hard-codes /kaggle/input/competitions/<comp>, the teammate account gets the legacy /kaggle/input/<comp> mount. Build script now tries both (COAT_ROOT_NEW); model67 v2 pushed.

2026-09-23 LEADERBOARD RESULTS: model67 (model61 + full-trained ResNet34 arm w0.30) public 0.942, BELOW model61's 0.943 - the extra family does not help the public ensemble; not a final pick. model70 (fine-spacing arm at the author's 80-slice geometry) public 0.943, TIES model61 despite +0.0014 on the gold blend (0.9326 -> 0.9340). Public LB prints 3 decimals, so a true gain under ~0.0005 is invisible; gold moves of ~0.0015 on 58 cases keep failing to show on the LB (4th time).
Fine-spacing reverse view (weight 0.05 fixed before the test, dense arm's 0.10:0.05 ratio; kernel rsna-knee-geometry-gold-finerev): forward 0.9252, reverse 0.9216, 3:1 mix 0.9247; blend 0.9340 -> 0.9349 (+0.0009, 95% CI [-0.0007, +0.0028], P>0 0.87) => NO under the rule.
Note: scipy is now blocked on this PC by an Application Control policy; scripts/compare_geometry_gold.py uses a numpy rankdata instead.

2026-09-24 MODEL71 = model65's ResNet34 recipe as 5 report-group folds (Vast RTX 3090, 13.07 h, 2.86 USD total; k-fold support added to scripts/train_model16.py --fold k). Fold gold AUC 0.9031 / 0.9024 / 0.9023 / 0.8937 / 0.9065; 5-fold ensemble 0.9138 (model65 single full-data 0.9104). As an extra blend arm at the pre-declared w=0.30: 0.9312 -> 0.9323 (+0.0011, 95% CI [-0.0039, +0.0061], P>0 0.69) => NOT adopted (and model67 showed a single ResNet arm costs ~0.001 LB).
Out-of-fold predictions for 4,345 of 4,349 non-gold studies -> data/llm_relabel/oof_resnet34_5fold.csv, targets_v1_oof5f50.csv (0.5 v1 + 0.5 OOF) and targets_v1_oof5f25.csv. Pseudo-label sanity on gold: labels alone 0.9035, 0.5 labels + 0.5 OOF(5-fold) 0.9316 (the failed 2026-09-20 teacher gave 0.9236; CoAtNet blend, unusable because in-sample, 0.9384). Four free CPU head-retrain kernels pushed: rsna-knee-head-retrain-oof5f50-{maxspan,dense,v8,fine}. Pre-declared rule unchanged: adopt only if the blend beats 0.9326 by >= +0.002 or all four arms improve (maxspan 0.9308, dense 0.9236, v8 0.9218, fine 0.9181).
Operational: the Vast account hit 0 credit mid-run and the rental stopped with all five folds trained but NOTHING backed up - the 20-minute guard dies when the Claude app is closed. Recovered by restarting the stopped instance after a top-up. Lesson: a stopped Vast instance keeps its disk, but only an OS-level scheduled task would back up while the app is shut.

2026-09-24 OOF PSEUDO-LABEL RETEST WITH THE 5-FOLD TEACHER (targets_v1_oof5f50 = 0.5 Gemini v1 + 0.5 model71 out-of-fold; free CPU kernels rsna-knee-head-retrain-oof5f50-<arm>; scripts/blend_heads_compare.py oof5f50): FAILS the pre-declared rule again. Final head gold AUC model61 -> oof5f50: maxspan 0.9308 -> 0.9289, dense 0.9236 -> 0.9197, v8 0.9218 -> 0.9200, fine 0.9181 -> 0.9166; blend 0.9326 -> 0.9281 (-0.0044, 95% CI [-0.0112, +0.0017], P(diff>0) 0.075). Mean of both heads per arm 0.9309, also below 0.9326.
=> The better teacher does reduce the damage (old half-data teacher -0.0075, 5-fold teacher -0.0044, and mixed-label-vs-gold agreement rose 0.9236 -> 0.9316), but the sign never turns. Extrapolating to the 0.75/0.25 mix (targets_v1_oof5f25.csv is built and uploaded) gives roughly -0.002, so that variant was NOT run. DIRECTION CLOSED for head retraining: mixing image-model predictions into the report labels hurts heads trained on frozen CoAtNet features, at every teacher quality we can produce. Note the top teams pseudo-label a full image model, not a frozen-feature head - that would need a full retrain, and model71 already showed a ResNet34 arm is not worth adding to this blend (+0.0011).

2026-09-25 INFRASTRUCTURE: the project folder moved to "coding\RSNA Knee Abnormality Detection". All 74 directory junctions (Other/model* -> Model/model*, Other/data/* -> Data/*) still pointed at the old resources\kaggle path and were dead; repointed all 74. Other/model12 (the inference package every notebook imports) was missing after the move and was restored from the new GitHub backup (private repo yodharin-monplub/RNSA-KNEE, deploy key in ~/.ssh/rsna_knee_deploy, .gitignore excludes data/artifacts/weights/secrets).
cloud/vast.py ssh_command passed os.devnull ('nul' on Windows) to ssh -F. That works from PowerShell but under Git Bash ssh reads it as a relative filename and every deploy dies with "Can't open user config file nul"; model72's third rental booted and sat idle through 25 minutes of retries. Fixed to use .cache/ssh/empty_config. Two earlier model72 rentals hit Vast machine 11183, whose Docker Hub login is broken ("unauthorized") so it never pulls the image - excluded per run via cloud/model72_exclude.json.
Communication moved from email to Discord (bot RNSA in #rnsa): scripts/discord_notify.py and scripts/discord_check.py, token Discord_RNSA_BOT in .env. scripts/send_email.py still works as a fallback.

2026-09-25 MODEL72 = model65's recipe with only backbone+resolution changed: EfficientNet-B0 at 288 px (Vast RTX A4000, 2.3 h, 0.37 USD). Gold 0.9143 at selected epoch 17 (weak-val BCE 0.3174), the best SINGLE image model of the project: model65 ResNet34 256 = 0.9104, model71 5-fold ResNet34 = 0.9138, a fine-tuned CoAtNet arm = 0.925. Training was stable throughout (no divergence; gold still climbing at epoch 24 = 0.9135).
As a blend arm at the pre-declared w=0.30: 0.9312 -> 0.9328 (+0.0016, 95% CI [-0.0033, +0.0068], P>0 0.75) => NOT adopted. With model71's 5-fold ensemble averaged in first: +0.0008, worse. Per finding vs ResNet34: MCL +0.132, Lat/Med Meniscus +0.016, Medial OA +0.008; Fracture -0.051, Synovitis -0.038, Contusion -0.015, ACL -0.012 - the gain sits in MCL where the CoAtNet arms are already strong, so the blend barely moves.
=> DECISION: do not take EffNet-B0 to 5 folds. Folds bought +0.0034 for model71, so a 5-fold EffNet-B0 lands near 0.918, still under one CoAtNet arm (0.925), for ~2.5 USD. Full training of SMALL CNNs plateaus at ~0.914 for this pipeline; the blend is CoAtNet-driven and only a model in that class can move it. Untested big lever remaining: full-train a CoAtNet (backbone yodharinmonplub/rsna-knee-backbone-coatnet2 is already uploaded) with the model65 recipe - Raymond Yuen reports full-trained CoAtNet at 288 px scoring 0.938 LB, and our own from-ImageNet CoAtNet attempt failed BEFORE augmentation existed in the recipe.

2026-09-26 MODEL73 = FULL training of CoAtNet-2 (coatnet_rmlp_2_rw_384, 73M) with the model65/72 recipe at 256 px. Needed: backbone_img_size=256 (position bias baked for 384 px), grad checkpointing + chunk 32 (OOM at 24 GB), and encoder LR 2e-5 / warmup 10% (at 2e-4 it underfit: gold 0.584 after 5 epochs, training loss flat ~0.5; stopped at 0.44 USD). Final run 4.8 h, RTX 3090 0.174 USD/h, 0.84 USD. Gold 0.9122 at selected epoch 17 (weak-val BCE 0.3115; peak 0.918 at epoch 19) - not better alone than EffNet-B0 (0.9143).
BLEND at the pre-declared w=0.30: 0.9312 -> 0.9335 (+0.0023, 95% CI [-0.0033, +0.0080], P>0 0.81) => ADOPT - first new arm to pass the rule since model61. mean(model73, model72) as the arm: +0.0016 => no. model73 helps the blend more than the stronger EffNet-B0 does, i.e. the full-trained CoAtNet makes different errors from the fine-tuned ones. CAVEAT: model67 had +0.0021 gold and lost 0.001 LB; the LB is the test. Next: model74 = model61 + model73 arm at 0.30, dummy-test run, submit.

2026-09-27 CORRECTION to model73's verdict. The arm tests (model67/71/72/73) used a RANK mean of the four head-retrained arms with weights .6/.1/.1/.15 (base 0.9312). The notebook actually blends Raptor arms by a weighted PROBABILITY mean with view weights maxspan 0.70 / dense 0.15 / v8 0.30 / fine 0.15 (base 0.9326, reproduced exactly). Re-tested faithfully (scratchpad eval_model73_notebook_way.py): model73 at the pre-declared w=0.30 -> 0.9344 (+0.0018, 95% CI [-0.0027, +0.0064], P>0 0.78) => NOT ADOPTED, 0.0002 under the bar; w=0.20 +0.0007, w=0.40 +0.0020 (not selectable after the fact); control model72 at 0.30 +0.0006. model74 (model61 + model73 arm) is therefore NOT built or submitted. Lesson: future arm tests must use the notebook's own blend (probability mean, notebook weights), not the rank approximation - the approximation overstated model73 by 0.0005.

2026-09-27 PUBLIC NOTEBOOKS OVERTOOK US; GRAFTS ONTO "SPEEDY RAPTORS". LB (4390 teams): top 0.959, ~400 teams tied at 0.943 (ranks ~201-600), we are rank 394 at 0.943; silver cutoff (rank 219) is the top of that tie, so 0.944 = silver zone. A 76th-place competitor reports gold 0.925 -> LB 0.947 and "trusts +0.002 LB changes" - our own history agrees the 58 gold cases barely predict the LB.
- pjmathematician/rsna-knee-d4-blend (public 0.946) = Speedy Raptors + its author's PRIVATE model fleet (datasets rsna-knee-eff6-assets / rsna-knee-ens14-assets, hidden as blank sources) rank-blended at 0.45; without them it "SOFT FAIL -> public only". d4-lite (0.945) differs by one line (fleet size). Not reproducible for us.
- mattiaangeli/bend-the-knee-to-speedy-raptors-the-original (public 0.943, all public) = our base family ("Bend the Knee to the Dinosaurs") + two new CoAtNet readers (D4 depth-zone SWA3, Global96) + a restructured Raptor stage: 4 views on 2 GPUs, capacity-aware dense slice sampling (~96 slices, ~94 windows), strict checkpoint loader with SHA-256 logging, weights resolved only inside a whitelist of public dataset folders. It still uses the PUBLIC Raptor weights.
- Our LB ladder came entirely from Raptor WEIGHTS: repro 0.940 -> model24 weak-label fine-tune 0.941 -> model43 Gemini fine-tune 0.942 -> model61 head retrain 0.943; extra views (model26 fifth arm, model30 reverse views) never moved the LB.
=> model76 = Speedy + our model61 Raptor weights (only change; own lookup for our three files). model77 = model76 + model73 as a fifth Raptor-side view at the pre-declared share 18.75% (w 0.23), inserted after the Raptor stage via raptor_raw.npz -> re-ranked _raptor.csv (LB probe; gold +0.0018 the notebook's way). model76 queued on yodharinmonplub, model77 (v2) on krittanutsomtuas - both waiting for GPUs (likely weekly quota; reset 00:00 UTC). model79 prepared: full fine-tune of the public Raptor maxspan checkpoint (all stages, plain BCE, augmentation, 10 epochs, LR 1e-5/1e-4); train_model16.py fine-tune path now honours grad_checkpointing. Also examined and set aside: d4-blend's public-side tweaks (Fracture routed through RadImageNet, CoAtNet weight 0.60 vs 0.70 - LB-tuned, and our own 0.60->0.70 test was neutral); retraining the E11/E13 RadImageNet heads (GPU, and they already train on three other LLM label sets); the new CoAt readers' gold58 files (their epochs were picked on gold, so gold tests with them are biased).

2026-09-27 (morning) MODEL77 V2 FAILED ON THE HIDDEN SET (submission 56587767, "unhandled error while rerunning"). Cause, my bug: Speedy writes raptor_raw.npz BEFORE neutral-filling the studies a view could not read; on the hidden set some studies are unreadable, the NaN rows reached rsna_rank01 -> rsna_finite and it raised. The 3 dummy studies are all readable, so the dummy run passed. Lesson: every extension that reads an intermediate file must survive NaN/unreadable rows, and a hidden-set failure shows up as a COMPLETE submission with errorDescription and no score - the tick scripts now read error_description and report SUBMISSION_FAILED (the old ones printed "no score yet" forever).
Model77 v3 (scripts/run_model77.py): exact Speedy neutral fill before blending; fifth view on both GPUs (even/odd studies, two model copies); time-boxed (--allow-seconds; a study not reached keeps its four-view value, which makes its blend equal the four-view probability); fail-safe (any exception keeps Speedy's four-view _raptor.csv); own identities removed from Speedy's preparation audit; atomic write. Mocked-namespace test, 5 cases pass (normal with one failing study, allowance cutting in, zero allowance, checkpoint read error, dummy-sized). Allowance to be set from model76's hidden runtime (runtime is binding: model29 failed on time; Speedy's later CoAt children and rsna_deadline need the 8 h TIME_BUDGET).
Speedy facts found while fixing: _ke_prepare_windows ignores k_eval (always ~96 capacity-aware slices -> ~94 windows); the maxspan arm's preparation (img 336, SLOTS64, span 2-98%) equals ours at another res; final blend = transformer side vs CoAtNet side at 0.70-1.00 per label, and the CoAtNet side = 0.6 public-Raptor rank + 0.4 CoAt-family rank, so the Raptor side is ~35-60% of the final.
d4-blend facts: hidden test ~1,300 studies ("on the hidden set (~1300)"); its private fleet (arms S_dist / S_coat / S_cnxl, max 36 windows) runs after the public pipeline with the remaining time of a 9 h limit (process age) and is rank-blended with ONE global weight 0.45 ("tuning per-label weights in-sample would have claimed +0.0026 out of pure noise"); fallback to public-only on any failure; they measured rank correlation 0.75-0.89 between their legs and the public output.
FLEET LEG gold check (scratchpad eval_fleet_leg.py; base = model61 Raptor arms, probability mean 0.9326; final = (1-w) rank(base) + w rank(fleet)): fleet3 = mean prob of model72 + model71 (5 folds) + model73 alone 0.9258, rank corr 0.935 with the base; w 0.15/0.25/0.35/0.45 -> +0.0005/+0.0013/+0.0012/+0.0015 (P>0 0.65/0.73/0.65/0.64). fleet2 (72+73) and fleet4 (+model65) weaker (<= +0.0007). Noise level on 58 cases, but it is the mechanism behind d4-blend's +0.003 LB.
=> MODEL80 = model76 + fleet leg (fleet3) at w = 0.30 fixed in advance, as a final notebook cell after Speedy's submission.csv: model12's own reader (the geometry the fleet was trained/gold-evaluated on), both GPUs with two decode threads each, windows per resolution shared by the models, a study the fleet cannot read keeps its Speedy rank, fail-safe (replaced only when all studies were attempted inside 9 h minus 30 min by process age; gives up early on a projected overrun; any error keeps Speedy's file). CPU test with the real checkpoints and a fake reader passes (7 strict loads, column order, unreadable-study rule, missing-checkpoint and time-guard fallbacks). Weights: private dataset yodharinmonplub/rsna-knee-fleet-weights (scripts/upload_fleet_weights.py). Kernel yodharinmonplub/rsna-knee-model80-speedy-our-heads-plus-fleet v1 pushed for the dummy-data run.
2026-09-27 public-notebook sweep: no public notebook beats Speedy's 0.943 without private assets. evgendvorkin/rsna-versia-5 (225 votes; annotated Speedy) best 0.942; kolyaflexcu/rsna-knee-speedy947 ("taken apart" Speedy) says public 0.942; romantamrazov/rsna-knee-dinosaur-v5 (Speedy + a Synovitis specialist from the RadImageNet ranks) 0.943. Only pjmathematician's d4-blend 0.946 / d4-lite 0.945 are higher - via a private fleet, i.e. the model80 route.
Forum (2026-09-27): Chris Deotte (87th) reports a 0.945 LB ensemble of his own models without public notebook models; several teams report 5-fold EfficientNet-B0 (+TTA) at 0.94+ public with 0.88-AUC extraction labels; a single fold with 0.908-AUC labels scored 0.935; resolution 224-288 px is enough; one team found 24 -> 32 cached slices per series worth more than any label work; Prateek Grover cut a CoAtNet's hidden-set inference from ~40 to <10 min (one model per GPU, 16 reader threads - a single reader gets ~59 MB/s from /kaggle/input, 16 threads ~372 MB/s - header parsing without pydicom, pixels via np.frombuffer, strided percentiles); top-50 are all > 0.95 with their own models; the silver/bronze pile-up comes from shared notebooks, so a private-LB shake-up is likely.
MODEL81 SCORED: public 0.933 (56593100) - this project's fully trained fleet ALONE (mean prob of model72 EffNet-B0 288, model71 ResNet34 x5 folds, model73 CoAtNet-2 256; gold 0.9258). Scored ~25 min after submission, so the fleet leg is cheap on the hidden set. Our own models sit 0.010 below the public Speedy stack (0.943); gold->LB is consistent with model61 (gold 0.9326 -> 0.943). Pre-declared rule: model83 (fleet w 0.45) only if model80 beats model76 or model81 >= 0.940 - the second condition failed, so it waits on model80.
2026-09-27 fleet-strengthening plan after model81 = 0.933 (our own models are 0.010 below Speedy). Prepared (not run; one Vast instance at a time, after model75; Vast credit 4.36 USD, top-up requested):
- model84 = model72 exactly (same seed) on pseudo-label targets targets_v1_oof5f50 (0.5 v1 + 0.5 model71 OOF; label-vs-gold 0.9316 vs 0.9035). Pseudo-labels only ever failed for frozen-feature head retrains; full-training pseudo-labels are untested here.
- model86 = model72 recipe (same seed) on a DENSER cache: 128 frames (32/24/24/24/24, i.e. 24-32 slices per series) at 256 px instead of 64 frames (18/14/12/8/12, 8-18 per series) at 336 px; res 256. Cache building now on free Kaggle CPU (yodharinmonplub/rsna-knee-dense-cache-256-part{0..3}-s128). Builder/verifier/loader/trainer generalised to 128 frames and tested on a synthetic pilot cache (the test caught a third hard-coded 64 in the verifier's mask check); the 64-frame path is unchanged.
- model82 = 5-fold EfficientNet-B0 (64-frame cache, v1 labels) - switches targets/cache to whichever of model84/model86 wins.
- model83 = model80 at w 0.45 - waits on model80 vs model76.
Tooling: scripts/run_model80.py and scripts/run_model81.py now take --name/--title and --weight/--with-model75 (fleet leg) or --fleet (fleet-only member list); default rebuilds are byte-identical to the submitted model80/model81 notebooks. scripts/fleet_leg_v2.py = the same leg with a per-member reader geometry ('s64' 64 frames @ 336 px, 's128' 128 frames @ 256 px) for fleets that mix model86-style members; CPU-tested with mixed geometries, an unreadable study, a missing checkpoint and the time guard. scripts/upload_fleet_weights.py --add-model75 publishes model75 as a new dataset version once it finishes.
128-FRAME CACHE BUILT (free Kaggle CPU, yodharinmonplub/rsna-knee-dense-cache-256-part{0..3}-s128): 4,407 studies, 276 shards, 26.93 GB compressed, 0 failed slices, 35-46 min per part; manifests in artifacts/dense_cache_256_s128_part{p}_notebook/output. cloud/model86_ready.py (via dense128_ready.py) PASSES. Queue after model75: model86 (density) then model84 (pseudo-labels), then the 5-fold of whichever wins.
MODEL76 SCORED: public 0.945 (56587720) - NEW PROJECT BEST. Speedy Raptors (0.943) with our model61 Raptor weights swapped in: +0.002 over both parents (model61 0.943, Speedy 0.943). The Raptor-weight ladder (0.940 -> 0.941 -> 0.942 -> 0.943) carries over to the stronger pipeline. Leaderboard 2026-09-27 02:29 UTC: rank 134 / 4,407 (rank 100 = 0.947, 200 = 0.944, 220 = 0.943; 165 teams >= 0.945). Hidden rerun ~7.4 h including queue -> Speedy's own runtime is ~7 h; an extension inside the pipeline (model77) has little room before TIME_BUDGET 8 h, a final-cell leg (model80) has ~1-1.5 h under its 9 h guard.
Implication: better Raptor weights are worth the most per step -> model79 (full fine-tune of the public maxspan Raptor checkpoint) moves up the queue.
MODEL75 DONE (CoAtNet-2 full training at 384 px, model73's recipe at native resolution): selected epoch 16/25 (weak-val BCE 0.3106), gold 0.9204 - best single fully trained model so far (model73 256 px 0.9122, model72 0.9143). 11.85 h RTX 3090, ~2.56 USD. Pre-declared fifth-Raptor-view test (w 0.30, probability mean): +0.0017 (P>0 0.79) -> not adopted as a view (bar +0.002). As a fleet member: fleet4 = 72+71+73+75 alone 0.9265 (fleet3 0.9258), final rank blend +0.0016/+0.0020/+0.0022 at w 0.25/0.35/0.45. Uploading as fleet-weights dataset version 2.
Vast after model75 (13:30): account credit 1.94 USD, and ANOTHER project's instance (alpha-connectome, RTX 4070, 0.19 USD/h, 12 h so far) draws on the same credit. model79 (5-7 h) would risk emptying the account mid-run, which would also stop the other project's instance. So only a short capped run now: model86 (density test) launched on an RTX 3090 at 0.18 USD/h, cap 1.0 USD, deadline extended from the formula's 2.78 h to 4.0 h (<= 0.72 USD); model79 waits for a top-up (requested).
MODEL88 SCORED: public 0.938 (56600803) = fleet ALONE with model75 (72 + 71x5 + 73 + 75, mean probability). model81 (same without model75) = 0.933 -> +0.005 from one CoAtNet-2 384 px member at 1/4 weight, while gold moved only 0.9258 -> 0.9265. model75 is far stronger on the hidden set than the 58 gold cases show (the gold set saturates / is noisy for this model family). This makes full-training CoAtNet-2 at 384 px the strongest own-model direction and makes a model75-heavy fleet worth testing.
model86 ops (2026-09-27 14:25): the 27 GB cache downloaded fine (~40 min via kernel-output URLs), but Kaggle-API DATASET downloads crawled on this host (~37 KB/s): the 21 MB EfficientNet backbone took >10 min and the next step - the 300 MB public maxspan checkpoint, which from-ImageNet runs never use - would have taken ~2 h and blown the deadline. Fixed cloud/download_dense.py: dense128 runs skip that checkpoint, and an already-verified pretrained backbone is reused instead of force-re-downloaded. Recovery without a new rental: SIGKILL the run script first (so its ERR trap does not write RUN_FAILED), then the downloader; pipe the verified backbone and the patched downloader over ssh; restart run_model86.sh detached. It resumed at the cache unpack. Cost of the stall ~35 min (~0.10 USD).
MODEL89 SCORED: public 0.939 (56602006) = model75 ALONE (CoAtNet-2 384 px, full training). Better than the fleet with it (model88 0.938) and far above the fleet without it (model81 0.933): the other members dilute it. Own-model ladder on the LB: fleet3 0.933 < fleet4 0.938 < model75 0.939. Gold ranked model75 only slightly above the small CNNs (0.9204 vs 0.9143) - gold understates the 384 px CoAtNet. => More CoAtNet-2 384 full-training runs (seeds / folds, ~2.5 USD and ~12 h each on a 3090) are the best own-model investment; a model75-heavy fleet leg is the natural candidate for blending into Speedy.
MODEL80 SCORED: public 0.944 (56593028) = model76 + fleet3 leg at w 0.30 -> -0.001 vs model76 (0.945). A 0.933 fleet is too weak to add at 0.30. model83 (w 0.45) dropped by its pre-declared rule. Crude fit final ~ A - w(A-B) + w(1-w)D (A 0.945, B 0.933, w 0.3 -> D ~ 0.012) predicts ~0.9455 for fleet4 (B 0.938) at w 0.2-0.3: final-level blending of our current own models is worth at most ~+0.0005; it needs a stronger own model (B >= ~0.943) to matter. Today's last submission goes to model78 (fleet4, same w 0.30) to measure the fleet change at fixed weight.
MODEL86 DONE (density test, EfficientNet-B0, same seed/recipe as model72, 128-frame 256 px cache): weak-val BCE 0.3119 (epoch 24) vs 0.3174 - lower at every epoch from 15 on; gold 0.9102 vs 0.9143 (-0.0041, CI [-0.0195, +0.0108], P>0 0.29 - noise; Medial Meniscus -0.065, MCL -0.024, Fracture +0.083). 3.19 h, ~0.57 USD. LB test tomorrow: model91 = model86 alone (scripts/run_fleet_only.py, fleet leg v2 with the 's128' reader) vs model92 = model72 alone.
Vast credit after model86: 0.72 USD with the other project's instance (alpha-connectome, 0.19 USD/h) still running -> ~3.8 h of runway for it; user alerted by DM + email. RSNA has no instance up. Further training (model79, model90, model84) blocked on a top-up.
model93 prepared (2026-09-27 evening): model76 + model75 as a fifth Raptor-side view (model77 v3's time-boxed, fail-safe extension; w 0.23 = 18.7% of the Raptor side, fixed in advance), because model75 alone = 0.939 LB (on par with a Raptor arm) and shares the maxspan arm's exact input geometry. Mocked test passes (5 cases). Dummy run pushed; submit after 07:00 with model91/model92 (density test).
Ops 2026-09-27 evening: Kaggle's kernel-STATUS endpoint (GetKernelSessionStatus) returned 429 Too Many Requests for 40+ minutes after heavy polling; the OUTPUT endpoint kept working. Tick scripts now fall back to fetching outputs (a finished log + submission.csv = complete) and the in-turn monitor polls every ~170 s. model91/model92 dummy runs verified that way; model93's dummy run had no outputs yet at 18:30 (queued or running).
MODEL78 SCORED: public 0.946 (56603186) - NEW PROJECT BEST. model76 + fleet4 (with model75; 0.938 alone) at w 0.30: +0.001 over model76, +0.002 over model80 (fleet3 0.933 at the same weight = 0.944). The final-level fleet leg pays once the own fleet is strong enough (~0.938+), matching d4-blend's public 0.946. Leaderboard 2026-09-27 15:07 UTC: rank 127 / 4,441 (rank 100 = 0.947, 150 = 0.945, 220 = 0.943; gold line rank 20 = 0.955).
Next: a stronger fleet (more CoAtNet-2 384 models) and the weight - model95 = model78 at w 0.40; model94 = model93 (model75 as a Raptor view) + the fleet4 leg.
2026-09-27 night, Vast credit 0 (the other project's instance was stopped by Vast at 0 credit; user alerted twice). Free work only:
- Five LB tests verified and queued for 07:04 (quota reset): model91/92 (density: model86 vs model72 alone), model93 (model75 as a fifth Raptor view), model94 (model93 + fleet4 leg), model95 (model78 at w 0.40).
- model96 built (fleet4 with model75 x2 inside the fleet; gold noise-level +0.0001); run_model80.py gained --member-weights.
- Local CPU gold test of the reversed-window view (Speedy's maxspan-reverse trick) for model72/model75 on the pilot cache (58 gold studies at the training geometry) running.
- Asked the user whether the Kaggle TPU quota (separate from GPU) may be used to train more CoAtNet-2 384 models; TPU port of train_model16.py scoped (8-core data parallel, bf16, fixed shapes, eval shape handling), not started without an answer.
TPU route (drafted, NOT run - waiting for the user's OK on TPU quota): Kaggle gives 20 TPU-h/week (separate from the 30 GPU-h), sessions up to 9 h, now TPU v5e-8 (8 chips, ~197 bf16 TFLOPS each, 16 GB HBM each); TPU notebooks cannot be SUBMISSIONS in code competitions, but training there and inferring on GPU is the documented pattern. scripts/train_xla.py = train_model16.py's from-ImageNet recipe on PyTorch/XLA: one process per chip (torch_xla.launch), DistributedSampler, per-chip batch 1 study x 16 windows (global 8 = model75), bf16 autocast, gradients all-reduced before clipping, augmentation on the host CPU, evaluation padded to 62 windows with a masked attention softmax (fixed shapes), predictions all-gathered, rank-0 selection/saving in train_model16's checkpoint layout. Rough compute: model75's recipe should take well under an hour per run on v5e-8 vs ~12 h on an RTX 3090.
Reversed-window test-time view (2026-09-28 night, local CPU on the 58 gold studies at the training geometry): model72 forward 0.9141 (reproduces saved 0.9143 within fp16 noise) | reverse 0.9154 | mean 0.9163 (+0.0020); model75 rerunning (the first run hit a 7000 s timeout at 51/58 studies; ~118 s/study on this CPU). model98 = model78 + the reversed view on every fleet checkpoint, built (run_model80.py --reverse-tta).
2026-09-28 00:20 - user replied "both ok" (Discord DM): Kaggle TPU quota and ~4 h of Kaggle GPU may be used.
Quota facts (api.quota_view(), week refreshing 2026-10-03 00:00 UTC): GPU is only 6 h/week per account - main 31 min used (5.5 h left), teammate 5.3 h used (0.7 h left; the user's other competition); TPU 20 h/week per account, 0 used on both (40 TPU-h/week total).
- model97 (model75's head retrain, ~3 h T4): both accounts' two batch-GPU slots are occupied (main: model94/95 dummy runs stuck in the T4 queue; teammate: other work); retrying the push on the main account every 15 min. Kit for the teammate route uploaded (krittanutsomtuas/rsna-knee-head-retrain-kit).
- TPU: trainer kit dataset yodharinmonplub/rsna-knee-trainer-kit (flat files + layout.json: train_xla.py, tpu_common.py [model12's Classifier/windows without the OpenCV/pydicom imports], knee/dense.py, knee/external.py, prepare_dense_cache.py, partitions, labels, targets, config_model98.json = model75's recipe, seed 20261011, 18 epochs, no grad checkpointing); scripts/run_tpu_train.py builds the TPU kernel (restores the layout, copies train.csv + CoAtNet-2 ImageNet weights, unpacks the cache, runs train_xla.py). Smoke test (pilot cache) pushed as yodharinmonplub/rsna-knee-tpu-train-model98-smoke with machine shape TpuV5E8 (accepted by the API).
Reversed-window view, final: model72 forward 0.9141 / reverse 0.9154 / mean 0.9163 (+0.0020); model75 forward 0.9205 (reproduces saved 0.9204) / reverse 0.9155 / mean 0.9182 (-0.0023) / 0.9 fwd + 0.1 rev 0.9199. Opposite signs on two models, both inside 58-case noise -> DROPPED (no fleet reverse-TTA). The built variant is archived in Other/artifacts/dropped_reverse_tta_variant; the name model98 now belongs to the TPU-trained second CoAtNet-2 384 seed.
MODEL97 (model75 head retrain, Kaggle T4 3.2 h): cached features reproduce model75 (0.9205 vs 0.9204); retrained heads 0.9169/0.9170/0.9165, final 0.9165 = -0.004 on every seed -> NOT adopted. The head-retrain trick worked for the lightly fine-tuned Raptor arms (heads that had seen 12-16 windows for 2 epochs), not for a head trained jointly with its backbone for 16 epochs. GPU quota left on the main account this week ~1.8 h (dummy runs only).
TPU blocker (2026-09-28 06:45): known Kaggle bug (kaggle-cli issue #1197) - kernels pushed via the API with enable_tpu / machine_shape TpuV5E8 (or Tpu1VmV38) are accepted but not provisioned with the TPU image; our smoke kernel sat "queued" 6 h with 0 s of TPU used and no output. Workaround: start the run from the web editor (Settings -> Accelerator: TPU v5e-8 -> Save & Run All). The only connected Chrome is a Linux machine that is not this PC, so the user was asked (Discord) to click it or allow browser use.
2026-09-28 07:05 - five LB tests submitted: model91 56622724 (model86 alone), model92 56622827 (model72 alone), model93 56623019 (model75 as a fifth Raptor view), model94 56623156 (model93 + fleet4 leg), model95 56623256 (model78 at w 0.40).
User topped up Vast ("vast top up if you needed"; credit 9.56 USD, shared with alpha-connectome 0.17/h and arc-model39-cv 0.41/h). Campaign rail raised 25 -> 35 USD. model79 (full fine-tune of the public maxspan Raptor checkpoint, all stages, Gemini labels, 10 epochs) launched on Vast 53065087 (RTX 3090 0.174 USD/h, cap 2.5 USD). Next on Vast: model90 (second CoAtNet-2 384 seed) unless the TPU route works.
MODEL92 SCORED: public 0.920 = model72 ALONE (EfficientNet-B0 288 px, gold 0.9143). model75 (CoAtNet-2 384) alone = 0.939: +0.019 on the LB for a 0.006 gold gap - the CoAtNet-2 384 recipe dominates our small CNNs; gold compresses the difference between model families.
MODEL91 SCORED: public 0.922 = model86 ALONE (EfficientNet-B0 on the 128-frame 256 px cache) vs model92 0.920 (same recipe/seed, 64-frame 336 px cache) -> DENSITY HELPS (+0.002 LB), matching the weak-val BCE gap (0.3119 vs 0.3174). model99 (model76 + fleet5 with model86 read at 's128', w 0.30) qualifies; dummy run pushed. Next big own model: model75's CoAtNet-2 384 recipe on a dense 336 px cache (96 frames: 24/18/18/18/18) - cache build on free Kaggle CPU.
Dense 336 px cache for CoAtNet-2: 96 frames (24/18/18/18/18) at 336 px, 4 free Kaggle CPU kernels rsna-knee-dense-cache-336-part{0..3}-s96 pushed ~07:35 (pipeline extended to --slots 96, synthetic-tested). model100 = model75's recipe on it (seed 20261012, 18 epochs), queued on Vast after model79.
96-FRAME 336 px CACHE BUILT (rsna-knee-dense-cache-336-part{0..3}-s96): 4,407 studies, 276 shards, 33.94 GB compressed, 0 failed slices, 23-38 min per part (free Kaggle CPU). model100 readiness gate checked.
model79 progress: epoch 0 (untouched public maxspan checkpoint) weak-val BCE 0.4438 / gold 0.9199; epoch 1 BCE 0.2818 / gold 0.9212. Epochs take 78 min (all stages at 384 px, gradient checkpointing) -> 10 epochs ~13 h; rental limits extended to 13.8 h / 2.8 USD cap (~2.4 USD expected). Vast credit 7.80 USD, total burn 0.34 USD/h (alpha-connectome + model79).
TPU: the web-editor start (the user clicked Save & Run All; version 1 pushed via the API never got a TPU) is RUNNING - 822 s of TPU used at 09:44.
TPU SMOKE RUN #1 (2026-09-28 09:40, started by the user in the web editor, ~13 min of TPU): environment fine (torch 2.8.0+cpu, torch_xla 2.8.0, 8 devices, timm 1.0.27, 377 GB RAM). Two problems: (1) 11 min lost before the cache step - rglob('cache_manifest.json') walked the competition's DICOM tree -> depth-limited search that skips /kaggle/input/competitions; (2) the 8-process launch failed "TPU initialization failed: Invalid --..._slice_builder_worker_addresses specified. Expected 8 worker addresses, got 1" (libtpu: `tpu_process_addresses`="local") - Kaggle's TPU image presets TPU_PROCESS_ADDRESSES=local and CLOUD_TPU_TASK_ID for single-process use and torch_xla's configure_topology only uses setdefault -> train_xla.py now unsets both before torch_xla.launch. Also added: torch_xla gradient checkpointing (model75 used it; 16 windows of 384 px per 16 GB v5e chip would not fit without), --stop-at (chips agree via all_reduce MIN on how many epochs fit before the 9 h session limit; the cosine follows the new total), 2-epoch smoke, eval length = all window centers of the cache (62 for 64 frames, 94 for 96), --cache s96 option (rsna-knee-dense-cache-336-part{0..3}-s96), plans smoke / full / smoke+full / cpu-test. Kit rebuilt by scripts/build_trainer_kit.py. CPU check (PJRT_DEVICE=CPU on a free Kaggle CPU kernel, batch 1 x 2 windows) pushed 10:10 before asking for another TPU click.
model99 dummy run verified 10:14 (fleet5 leg incl. model86 at s128, 3/3 read, 50 s on 2 GPUs) -> submit with the next quota.
LEADERBOARD 2026-09-28 10:15: 4,474 teams; we (team "KrittanutSomtuas") rank 131 at 0.946. Lines: gold (top 18) 0.955, rank 100 0.947, silver (top 5% = 223) 0.944, bronze (top 10% = 447) 0.943; 707 teams >= 0.943 (Speedy clones). Best public notebooks now 0.945 (pjmathematician/rsna-knee-d4-lite, aastikrajan15/knee-s75-w60; both use private datasets), so our margin over copied notebooks is only ~0.001.
MODEL95 SCORED 2026-09-28 13:50: public 0.945 = model78 (model76 + fleet4 leg) at w 0.40 instead of 0.30 (0.946) -> -0.001; heavier fleet weight hurts, keep 0.30. model79: e3 BCE 0.2778 gold 0.9223, e4 0.2769 / 0.9246, e5 0.2764 / 0.9213 (BCE still falling, gold noisy around the 0.9254 bar).
MODEL93 SCORED 2026-09-28 14:26: public 0.946 = model76 (0.945) + model75 as a fifth Raptor view (w 0.23 of the Raptor side) -> +0.001, same as model78's fleet4 leg. model94 (both) pending.
MODEL94 SCORED 2026-09-28 14:33: public 0.946 = model93 + fleet4 leg (w 0.30). model78 / model93 / model94 all 0.946 (model76 0.945, model95 at w 0.40 0.945): at 3-decimal rounding the public LB can no longer separate these; the next visible step needs stronger own models (TPU CoAtNet-2 seeds, dense96) or better Raptor weights (model79).
TPU trainer k-fold support (2026-09-28 14:57): train_xla.py --fold k with config n_folds + select 'last' = train_model16.py's report-group folds (md5(report_hash) % n_folds over train+weak_val; fold 0 = 3,513 train / 832 OOF), last epoch kept, oof_predictions.csv per fold; run_tpu_train.py plans folds / smoke+folds (--folds 0,1,2: one cache unpack, a fold starts only if the previous fold's duration still fits). Local logic tests (CPU stand-in torch_xla) pass for the plain and the fold path. Purpose: 5-fold CoAtNet-2 384 on the TPU = a 5-model fleet member AND a strong OOF teacher for full-training pseudo-labels (the forum's 0.95 single-model recipe; our earlier pseudo-label failures used a weak ResNet34 teacher and frozen-feature heads).
TPU API PUSH FINDING (2026-09-28 15:00, kaggle-cli issue #1197, maintainer reply): API pushes DO get a TPU when kernel-metadata.json pins no CPU/GPU docker_image (ours pins none; acc=TpuV5E8). They become BATCH TPU sessions: background QUEUED state that can last hours, and only 1 active-or-queued batch TPU session per user ('Maximum batch TPU session count of 1'). So model98 v1 (API, 11:31) is most likely waiting in the TPU queue - it also blocks a second batch session (a web-editor Save & Run All). The morning's 6 h 'stuck' API smoke push was the same queue, not a missing image. Consequence: later TPU runs (k-fold) can be launched unattended via the API; only the queue wait is uncertain. Forum (743746): GPU/CPU batch kernels are also stuck in QUEUED for hours this week; advice is re-push. CPU test kernel re-pushed as v3 at 14:36.
MODEL79 COMPLETE (2026-09-28 18:55; Vast 12.4 h, ~2.17 USD; guard backed up + destroyed 19:38): full fine-tune of the public maxspan Raptor checkpoint on Gemini labels. Weak-val BCE 0.4438 (epoch 0 = public checkpoint) -> 0.2745 (epoch 10, selected); gold 0.9199 -> 0.9242 (range 0.9205-0.9261 over epochs, no trend after epoch 2). Below the pre-declared 0.9254 bar (and below model61's maxspan arm 0.9308) -> model87 NOT built. Lesson: fitting the LLM labels better (BCE) does not move gold for the public Raptor init; the head-retrained light fine-tune stays best. Vast slot released -> model100 (CoAtNet-2 384, 96-frame cache) launching 19:40.
model79 extra-view diagnostic (58 gold, notebook-way probability mean, bootstrap 3000): as an extra Raptor view at the pre-declared w 0.30 +0.0002 (CI [-0.0038, +0.0041]) -> no; w 0.15 -0.0002; together with model75 +0.0015 vs model75 alone +0.0017; replacing the maxspan arm -0.0020. model79 CLOSED (no use in the submission).
MODEL100 LAUNCHED 2026-09-28 19:39 (deployed 19:42): Vast 53178323, RTX 3090 0.18 USD/h, cap 2.8 USD, deadline 12.8 h; model75's CoAtNet-2 384 recipe on the 96-frame 336 px cache, seed 20261012, 18 epochs.
RUNTIME OPTION (2026-09-28 19:45): jiweiliu/rsna-knee-fast-2xt4-inference v11 (public 0.943, 191 votes) = Mattia Angeli's Speedy Raptors v34 recipe with inference-only speedups (20 DINO members share the frozen DINO prefix, overlapped CPU prep, shared MaxSpan decoding, cached DICOM order, parallel CoAt readers); same Raptor files (v5 maxspan fwd/rev, v10 native384dense, v8 native384). Pulled to artifacts/public_fast2xt4. If its hidden runtime is far below Speedy's ~7.2 h, it becomes the better base once our fleet needs more than ~1 h of GPU (a 5-fold CoAtNet-2 384 at 94 windows is ~25-45 min; current additions fit). Grafting model61 weights needs a new builder (its Raptor cell differs from Speedy's). Not started. Other public notebooks checked: evgendvorkin/rsna-versia-5 and romantamrazov/rsna-knee-dinosaur-v5 (0.943) are the same model family.
VAST CREDIT 2026-09-28 19:47: 3.86 USD, burn 0.347 USD/h (model100 0.18 + alpha-connectome 0.167) -> ~11 h; model100 needs ~12 h. Top-up requested by DM.
MODEL100 HOST ABANDONED 2026-09-28 ~20:50: Vast machine 67866 (RTX 3090, 0.18 USD/h) had the GPU power limit set to 160 W (normal 350 W); SM clock ~450 of 2100 MHz at 100% utilization -> ~6.5 s/step (~50 min/epoch + eval, 18 epochs ~16 h) vs ~3 s/step for model75 on a normal 3090. Vast's dlperf score (43.2) did not reveal it. Destroyed after ~1.4 h (~0.25 USD), machine excluded (cloud/model100_exclude.json), logs archived to artifacts/remote_model100_attempt1_throttled160W, state to artifacts/cloud/model100_attempt1. Relaunched; new check cloud/gpu_check.py (power limit vs card minimum) runs right after every deploy.
MODEL100 RELAUNCHED 2026-09-28 20:52 (first retry: rent HTTP 400, offer taken; second: rented, launch_only died on a Vast 429 during polling, deploy_run.py retried by hand, deployed ~20:57): Vast 53189562, RTX 3090 0.207 USD/h, power limit 320 W (cloud/gpu_check.py OK). Credit 3.35 USD at 0.373 USD/h account burn -> guard deadline set to 05:15 (backup + stop ~40 min before the credit runs out); restore after a top-up.
TPU RUN STARTED ~21:00 2026-09-28 (quota timeUsed 782 -> 882 s by 21:02): yodharinmonplub/rsna-knee-tpu-train-model98 (smoke+full).
model100 (relaunched host): training started ~21:43; epoch 1 = 2,162 s (36 min incl. eval; weak-val BCE 0.4573, gold 0.7103 in warmup) -> 18 epochs end ~08:30 2026-09-29. The credit-based guard deadline (05:15) would stop it after ~12 epochs unless the Vast credit is topped up (requested 19:47).
2026-09-29 00:08: model98 TPU kernel running since 23:34 (lastRunTime 16:34:53Z; TPU timeUsed 1771 -> 3795 s) - the queued API version started by itself; smoke presumably passed (session still alive after 34 min). The 21:00-21:18 TPU use was an interactive editor session on the old smoke notebook (no committed output). The user's connected Chrome is not signed in to Kaggle -> not usable for the private notebook (no sign-in by me).
TPU RUN 2026-09-28 23:34 (queued API v1, ~2 h of TPU until Kaggle ended it ~01:40): the launch fix WORKS (log: unset TPU_PROCESS_ADDRESSES='local'; 8 chip processes initialized; env TPU_ACCELERATOR_TYPE v5litepod-8, TPU_CHIPS_PER_HOST_BOUNDS 2,4,1, TPU_HOST_BOUNDS 1,1,1; host 96 CPUs, 377 GB RAM, /dev/shm 164 GB, /tmp 1.1 TB free; setup to pilot cache 74 s). NEW BUG at the first training batch (133 s): DataLoader workers inside the chip processes start with 'spawn', so the function-local Dataset could not be pickled ('Can't get local object run.<locals>.TrainBags'); the MpDeviceLoader thread died and the processes hung until the session ended (no outputs besides the log). My local test had used workers=0 and missed it. FIX: datasets moved to scripts/xla_data.py (module level, all inputs passed in), persistent workers, eval loaders built once; local test now runs train_xla.py as the main script with 2 spawned loader workers (Windows = spawn, same as the TPU).
2026-09-29 05:0x-05:40: model101 (model100 alone, s96) and model103 (model76 + model75/model100 CoAtNet-only leg) dummy runs VERIFIED; model102 (model76 + fleet6) dummy run pushed 05:18, verification pending (Kaggle network timeouts). Submission runner (scratchpad submit_morning.py) submits model99, model101, model103, model102 after 07:02.
2026-09-29 07:02-07:07 SUBMITTED: model99 56655870 (model76 + fleet5 incl. model86), model101 56656173 (model100 alone, s96), model103 (model76 + model75/model100 CoAtNet-only leg), model102 (model76 + fleet6 incl. model100). One slot left today.
MODEL101 SCORED 2026-09-29 07:34: public 0.937 = model100 ALONE (CoAtNet-2 384 on the 96-frame cache, stopped at epoch 12 of 18) vs model75 alone 0.939 -> -0.002; gold (0.9231 vs 0.9204) again did not predict the LB. Confounded by the unfinished schedule; a fully trained 96-frame CoAtNet is needed to settle density for this recipe.
2026-09-29 07:42: TEAMMATE TPU LANE - model104 (model100 recipe on the 96-frame cache, 18 epochs) pushed as krittanutsomtuas/rsna-knee-tpu-train-model104 (queued). Teammate copies of rsna-knee-trainer-kit and rsna-knee-backbone-coatnet2 created (build_trainer_kit.py / upload_backbone.py --account); main's cache kernels attach fine. run_tpu_train.py: --account, and a hang guard (process-group kill: smoke after 45 min, full 15 min past --stop-at).
LEADERBOARD 2026-09-29 08:07: 4,540 teams; us rank 137 at 0.946 (37 subs). Gold (18) 0.956, rank 50 0.952, rank 100 0.948, silver (227) 0.944, bronze (454) 0.943; >=0.946: 149 teams, >=0.943: 775.
MODEL99 SCORED 2026-09-29 13:50: public 0.945 = model78 (0.946) + model86 in the fleet -> -0.001. Weak members do not help; model102 (fleet6 with model100) / model103 (model75+model100 only) pending.
MODEL102 SCORED 2026-09-29 13:54: public 0.945 (fleet6 = fleet5 + model100) = model99; model78 (fleet4) stays 0.946. model103 (CoAtNet-only leg) pending.
MODEL103 SCORED 2026-09-29 14:09: public 0.947 = NEW PROJECT BEST. model76 + leg of ONLY our two CoAtNet-2 384 models (model75 s64 + model100 s96) at w 0.30. Same base/weight with the full fleet6 (model102) = 0.945, fleet4 (model78) = 0.946. => The weak members (EffNet-B0 x2, ResNet34 x5, CoAtNet-2 256) diluted the leg; model100 helps inside the pair despite 0.937 alone. Rule from now on: the leg holds only strong CoAtNet-2 384 models (model98 TPU seed, model104 dense96 full run next).
MODEL105 SUBMITTED 2026-09-29 16:22 (56670477): model103 with the CoAtNet-only leg at w 0.40. All 5 of today's slots used.
2026-09-29 17:25: model98 TPU v2 (queued 05:06, started ~16:23) passed the smoke (still running past the 45-min smoke cap: TPU timeUsed 9134 -> 12846 s) -> full training in progress. First working run of the TPU pipeline.
MODEL98 FIRST FULL TPU RUN (2026-09-29 16:23-18:13): smoke OK; ~285-300 s/epoch incl. eval after a 1,559 s first epoch (compile) = ~10x a 3090. Crashed in epoch 14 (BrokenProcessPool: one chip process killed abruptly, likely host OOM). Quality gap vs GPU: weak-val BCE 0.374 (epoch 13) vs model75 0.316 at epoch 13, gold similar (0.907 vs 0.910) -> bf16 autocast (BatchNorm/LayerNorm in bf16 on XLA) suspected. Fixes: fp32 default, last.pt/--resume with automatic kernel retry, non-resident eval workers, memory logging.
2026-09-29 18:36: TPU runs re-queued with the fixed trainer (fp32, last.pt/--resume, non-resident eval workers; local crash/resume test passed): model98 v3 (main, s64, seed 20261011, 18 epochs) and model104 v2 (teammate, s96, seed 20261012, 18 epochs; replaces the queued bf16 v1).
MODEL105 SCORED 2026-09-30 00:26: public 0.945 = model103 with the CoAtNet-only leg at w 0.40 (model103 at w 0.30 = 0.947) -> the leg weight optimum is <= 0.30.
2026-09-30 00:30: Kaggle GPU quota exhausted on BOTH accounts (yodharinmonplub 26,495 s, krittanutsomtuas 45,654 s of 21,600 s/week; refresh 2026-10-03 00:00 UTC). No new dummy runs -> no new submissions until then except the verified model106. TPU quota unaffected (main 15,807 s, teammate ~260 s of 72,000 s).
MODEL106 SUBMITTED 2026-09-30 07:02 (56690928): model76 + CoAtNet-only leg with model75, model100, model79 at w 0.30. GPU quota exhausted on both accounts until 2026-10-03 07:00 -> no further dummy runs/submissions until then.
MODEL98 TPU v3 COMPLETE (2026-09-30 05:06-12:40, fp32): 17 epochs, selected epoch 16 (plain BCE 0.3700, gold 0.9154; epoch 13 gold 0.9207). Epochs 1-13 ~305 s, epochs 14-17 4,500-6,800 s with host free memory 137 -> 94 GB at epoch 14 (bf16 v2 crashed at epoch 14) -> unexplained slowdown, diagnostics added (XLA compile count, device memory). KEY: the weak-val BCE gap to GPU was a METRIC MISMATCH (train_model16 = confidence-weighted BCE; train_xla used plain mean, ~0.06 higher); gold is on par (0.92) -> TPU models are fine. train_xla switched to the weighted formula. model98 best.pt added to the fleet weights.
2026-09-30 13:12: model109 (third model75-recipe seed, s64, fp32) queued on the main TPU with the relaunch fix (every 7 epochs; local relaunch test passed) and the confidence-weighted weak-val BCE. model104 (teammate) still running the previous code.
MODEL104 TPU COMPLETE (teammate, 2026-09-30 05:10-13:25, fp32, 96-frame cache): 15 epochs (plan cut by the time guard), selected epoch 15, GOLD 0.9306 = best own model so far. Epochs 1-2 slow (cold page cache, ~70-80 min), 3-13 ~336 s, 14-15 5,832/9,304 s -> the epoch-14 slowdown is STEP-based (epoch 14 began 3.6 h into training here vs 1.5 h for model98), with host free memory falling ~2.5 GB/epoch then -41 GB at epoch 14. Leading hypothesis: a per-epoch device-memory leak fills HBM around step 6,000 and the runtime starts swapping (the bf16 run crashed there). model109 (queued) relaunches processes every 7 epochs and logs device memory. model104 best.pt added to the fleet weights.
2026-09-30 13:34: model110 = 5-fold CoAtNet-2 384 on the 96-frame cache (seed 20261014, 18 epochs, select last, OOF) - session 1 (folds 0,1,2) queued on the teammate TPU (krittanutsomtuas/rsna-knee-tpu-train-model110-f012).
MODEL106 SCORED 2026-09-30 13:57: public 0.946 = model103 leg (model75 + model100) + model79 -> -0.001 vs model103 (0.947). model79 (fine-tuned from the public Raptor maxspan checkpoint) dilutes the leg - it overlaps the Speedy Raptor side. Leg members must be independent strong CoAtNets (ImageNet-initialised).
2026-09-30 14:05: Saturday (after the 2026-10-03 07:00 GPU quota refresh) candidates built and armed (scratchpad saturday_runner.py: push dummy runs, verify, submit): model108 = model104 alone (fleet-only s96); model107 = model76 + leg [model75 s64, model100 s96, model104 s96] w 0.30; model111 = model107 + model98 (s64). model79 dropped from legs (model106 0.946).
2026-09-30 19:20 KAGGLE KEYS: the .env key names are the owners' EMAIL names (the user: "it's the email name, don't change it" - never rename .env keys): KAGGLE_API_KEY_YODHAIRNMONPLUB = yodharinmonplub (submission only), KAGGLE_API_KEY_WARIN = krittanutsomtuas (training lane; was KAGGLE_API_KEY_KRITTANUTSOMTUNG), KAGGLE_API_KEY_KRITTANUT = actuarialtutor (new). KEYS dicts in run_tpu_train / build_trainer_kit / collect_tpu_run / upload_backbone / upload_fleet_weights (and the Model copies) updated; new scripts/lb_rank.py (read-only LB snapshot with a training key). Access test (artifacts/accel_test_actuarialtutor): actuarialtutor GPU OK (2x Tesla T4, 13 s used); TPU via API NOT granted - 3 pushes (default image -> CPU image; 'latest' pinning; TPU-VM image pinned -> torch_xla present but device kind CPU) all ran on a CPU machine, 0 s TPU used, while the same push queues for a real v5e-8 on the other two accounts. actuarialtutor has not joined the competition (ListSubmissions 403). Asked the user for one web-editor TPU run on it, and to join the competition + our team before it touches competition-derived data (our caches).
LEADERBOARD 2026-09-30 19:10: 4,642 teams; us rank 114 at 0.947 (tied block 114-129). Gold (19) 0.956, silver (232) 0.944, bronze (464) 0.943; top 0.961.
2026-09-30 20:10 FORUM (discussion 735304 'Best single-model score', 742327, 744294): 11th place reports a 5-fold ResNet at 224 px = 0.954; 52nd a Qwen 3.5 2B VLM (LoRA + vision encoder, 384 px) single fold 0.950; Chris Deotte: single models trained on PSEUDO labels match the best ensemble CV; Raymond Yuen (80th, 0.947): several weak-label sources -> teacher models -> pseudo-labels -> final model, "train on everything, evaluate on gold only", labels cost < 5 USD. Per-group LB readout trick (744294): keep 4 findings, set 8 to 0.5 -> group AUC = (12 LB - 4)/4 (+-0.0015); one team's CoAtNet 0.940: ACL/MCL/menisci 0.962, Synovitis/Baker's/Medial OA/Fracture 0.941, Effusion/Contusion/PF OA/Lateral OA 0.914.
=> Pseudo-label full training is the lever. Pipeline prepared: (1) train_xla.py 'full_data' option (train + weak_val = 4,345 studies, select last; local smoke PASS); (2) scripts/make_pseudo_targets.py (alpha*labels + (1-alpha)*OOF, OOF of several k-fold runs averaged, gold sanity check; reproduces the 2026-09-24 numbers exactly with model71: labels 0.9035, mix 0.9316); (3) run_tpu_train.py --then NAME2 (second model on the same cache after the folds if ~1.25x the last fold still fits; plan logic simulated with a fake clock) and kernels now ALWAYS exit 0 + PLAN_STATUS.txt (a failed Kaggle run keeps no outputs - one failing fold would have discarded the finished ones; model110-f012, already queued, still has the old exit code); (4) collect_tpu_run.py reports every run folder (folds, --then). Trainer kit (krittanutsomtuas) pushed with full_data + config_model112.
Plan: model110-f012 (queued) -> session 2 = model110 folds 3,4 + model112 (FULL-DATA CoAtNet-2 384 s96, v1 labels, seed 20261016) -> model113 = pseudo-label student (targets 0.5 v1 + 0.5 model110 OOF, full data).
2026-09-30 20:40 GROUP READOUTS prepared (scripts/make_group_readout.py; Model/model114-119): model103 (0.947) and model76 (0.945) each read out in G1 = ACL/MCL/menisci, G2 = Medial OA/Synovitis/Baker's/Fracture, G3 = Lateral OA/PF OA/Effusion/Contusion (last cell keeps 4 findings, 8 columns 0.5; group AUC = (12 LB - 4)/4 +-0.0015; fail-safe, tested on a fake submission). Tick: scratchpad readout_tick.py (checks the readout took effect: 8 columns exactly 0.5). Each needs a main-account GPU dummy run -> from Saturday 2026-10-03, after model108/107/111, using spare daily slots unless TPU models (model110 folds) arrive first.
2026-09-30 20:30 LEG TIME BUDGET (review of fleet_leg_v2): the leg starts after Speedy (~7.2-7.4 h into the hidden run) and must finish by notebook age 8.5 h (9 h - 30 min safety) -> ~40-70 min. It is ALL-OR-NOTHING: when the projected finish passes the deadline it silently keeps Speedy's file, i.e. the submission scores like model76 (0.945). It also re-reads every study's DICOMs once PER GEOMETRY (s64 + s96 = two full header scans + ~160 decodes per study). => model102 (fleet6, 0.945) and model99 (fleet5 incl. s128, 0.945) may have TIMED OUT rather than been diluted (both equal model76 exactly); model106 (3 CoAtNet-384, 0.946 != 0.945) did run. Saturday's model111 (4 CoAtNet-384 members) is at risk.
FLEET LEG V3 (scripts/fleet_leg_v3.py, generated by scratch make_fleet_leg_v3.py from v2): (1) one-pass reader - geometries share the plane/fluid slot sequence, so each slot's series is chosen once, its DICOM order scanned once and each slice decoded at most once (identical volumes by construction; real-DICOM check running on a free CPU kernel, krittanutsomtuas/rsna-knee-leg-reader-bench); (2) priority-based member dropping - on a projected overrun the LAST members of _M80_FLEET are dropped (partial predictions discarded) and the projection re-measured over 40 new studies; only if the first member alone cannot finish is the leg abandoned; (3) reading follows the active members' geometries. run_speedy_fleet_v2.py --leg v3 (default stays v2: a v2 rebuild of model107 is byte-identical to the armed notebook).
FORUM 743374 (Prateek Grover, 173rd): CoAtNet inference ~40 min -> <10 min on 2xT4 by: one model copy per GPU with its own study shard (we do this); read each DICOM ONCE, concurrently - one threaded pass per series for ordering + pixels, 1 reader ~59 MB/s from /kaggle/input vs 16 threads ~372 MB/s (our leg: 4 reader threads, 2 passes); parse headers directly (0.31 -> 0.06 ms/file); pixels via np.frombuffer at the PixelData offset (bit-identical, 1.22 -> 0.33 ms/slice); percentiles on a strided subsample (np.percentile ~60% of decode time - would change our normalisation, so not for us); fp16 weights + half inputs without autocast = 2.08x over autocast on T4; uint8 to GPU, normalise on device, pinned memory. => candidate leg v4 speedups: more reader threads, windows built on the GPU from uint8, pure fp16 (check prediction drift in the dummy run).
LEG v3 CPU TESTS PASS (scripts/test_fleet_leg_v3.py; real checkpoints on CPU + fake reader): (A) normal blend 5/6 studies, members kept, reads only the requested geometries; (B) missing checkpoint -> FAILED, Speedy file untouched; (C) deadline passed -> TimeoutError, kept; (D) 200 studies, tight budget -> 'dropped model73 after 40/200', model72 + model71 blend 199/200; (E) first member alone too slow -> model73 and model71 dropped, then abandoned, Speedy kept.
T4 BENCHMARK (actuarialtutor free GPU, random weights, no competition data): CoAtNet-2 384 = ~60 img/s per T4 for autocast AND pure fp16, batch 12/24/47, cudnn.benchmark on/off (the forum's 2x fp16 gain does not reproduce on torch 2.10); windows() on the CPU = 0.26 s per s96 study. => hidden set (1,300 studies, 650 per T4): ~17 min per s96 member (94 windows), ~11 min per s64 member (62 windows), + ~5 min CPU windows for s64 + s96. Estimated legs: model103 ~33 min (ran, 0.947), model106 ~50 min (ran, 0.946), model107 ~51 min, model111 ~62 min (borderline -> v3 drops model98 first). The leg's lever is WINDOWS per study, not precision.
LEG READER BENCHMARKS (free CPU kernels on krittanutsomtuas, 2026-09-30 20:00-20:25): (1) krittanutsomtuas/rsna-knee-leg-reader-bench: v3's one-pass reader gives BYTE-IDENTICAL s64 + s96 volumes and masks to v2 on 43 real studies (3 test + 40 train), 0 mismatches, 0 errors. (2) leg-reader-threads: v3 s64+s96 wall per study with 4 / 8 / 12 / 16 reader threads = 1.84 / 1.70 / 2.00 / 1.95 s -> no gain from threads (bandwidth-bound: ~50 studies x ~82 MB in 92 s = ~45 MB/s from /kaggle/input); v3 keeps 2 readers per GPU, prefetch 4. (3) leg-reader-profile (30 cold studies, 1 thread, 4.6 s/study): 59% ordered_files = the header scan that opens EVERY file of each chosen series (~164 files/study; first touch of a file is the expensive part), 13% np.percentile, 17% pixel decode + dcmread parsing, 6% clip + resize. All series are Explicit VR Little Endian 16-bit with non-unit RescaleSlope. Speedy keeps its own header caches but orders and selects series differently -> not reusable without changing our volumes. => The leg's reading floor is ~1.8 s/study x 1,300 = ~40 min of the leg budget; the next lever is GPU time per member (window stride, parity kernel running).
2026-09-30 20:35: full leg v3 test suite PASSES on the final generated version (2 readers/GPU, prefetch 4). model107 and model111 REBUILT with --leg v3 (only the leg cell differs from the armed v2 notebooks; kernel metadata identical); priority = fleet order: model104, model75, model100 (, model98). Their ticks now also require 'dropped for time []' in the dummy-run log. Saturday runner 2 (scratch saturday_runner2.py, 07:40) will push the model114/model115 readouts (model103 G1/G2) into the two spare slots.
2026-09-30 20:36 model110-f012 (folds 0-2, smoke first) STARTED on the teammate TPU after ~7 h in the batch queue (teammate TPU timeUsed 50,987 -> 51,157 s). Chained (scratch tpu_chain.py): on completion collect -> DM -> push model110-f34-then-model112. model113 (pseudo-label student) prepared in Model/model113 (config + steps; waits for all 5 folds' OOF).
2026-09-30 23:07 MODEL109 COMPLETE on the MAIN account's TPU (queued 13:12 before the submit-only rule; the user's cancel did not take; it started 19:50 and finished 23:01): PLAN smoke+full OK, session 11,454 s = 3.2 h for smoke + unpack + 18 epochs. THE RELAUNCH FIX WORKS: processes restarted after epochs 7 and 14 (resume from last.pt); epoch times 299-302 s steady, epoch 14 462 s, 15-16 (first after the relaunch, compile) 778 / 905 s, 17 447 s, 18 302 s - no more 15-25x epoch-14 slowdown; XLA compiles 8-10 per process; device memory flat. Results (s64, seed 20261013, model75 recipe, fp32, confidence-weighted weak-val BCE): epoch 18 BCE 0.3118 (best), gold 0.9126 (epochs 11-18 gold 0.905-0.914). => a full 18-epoch TPU run costs ~3.2 h (s64), so the teammate's 20 h/week = ~5-6 models/week. Main-account TPU total now 54,691 s; nothing else will ever be queued on main.
2026-09-30 23:30 TPU SESSION TIMING (model110-f012 live log with event times): after the smoke (445 s) and the s96 cache unpack to /tmp (1,188 s), fold 0 epoch 1 = 5,594 s (steps 0-100 2,259 s incl. compile, then 700-880 s per 100 steps), epoch 2 = 294 -> 467 -> 659 -> 807 s per 100 steps (slower as it goes), vs ~65 s per 100 steps in steady state (model104 epochs 4-13 = 337 s/epoch). The trainer's --stop-at guard shrank fold 0's plan 18 -> 4 after epoch 1 (it projects from the last epoch); it regrows once epochs are fast and the warmup (1.8 epochs) hides most of it, but the kernel's "next fold only if the last fold's duration fits" rule then used the inflated first-fold time and would skip folds 1-2 in this session and fold 4 + model112 in the next. FIXES in run_tpu_train.py (kernel): (1) steady_need(): the next run's need = median epoch time after epoch 2 x epochs + 1.5 epochs per process start/relaunch + 10 min (simulated with a fake clock: session 2 now runs fold 3 AND fold 4); (2) the FULL cache is unpacked to /dev/shm (RAM, 164 GB; need x 1.5 headroom) instead of /tmp, whose slow disk is the likely cause of the crawling first epochs. model109 (s64) epoch 1 = 2,567 s, epoch 2 = 726 s - the s64 cache is smaller, the s96 one hurts most. The running session keeps the old kernel (likely fold 0 only); scratch tpu_chain2.py then builds the next kernel for whichever model110 folds are still missing (+ --then model112) and pushes it.
FLEET LEG V4 (scripts/fleet_leg_v4.py, generated by scratch make_fleet_leg_v4.py from v3): optional 4th element per fleet entry = window stride (('model104', ['model104_best.pt'], 's96', 2) feeds every 2nd window -> half that member's GPU time, ~8.5 instead of ~17 min on the hidden set). Tests (scripts/test_fleet_leg_v4.py): all strides 1 -> byte-identical submission and member predictions to v3; model73 at stride 2 -> only model73's predictions change. run_speedy_fleet_v2.py --leg v4 accepts 4-tuples (v3 rebuilds of model107/111 still byte-identical). Whether stride 2 is safe for real models waits for the window-parity kernel (model104 on the 58 gold studies: all vs even vs odd windows).
2026-10-01 00:32 WINDOW PARITY (krittanutsomtuas/rsna-knee-leg-window-parity, free CPU, model104 on the 58 gold studies through fleet leg v3's own reader/windows/predict, CPU fp32): gold AUC TPU training eval 0.9306 | leg all 94 windows 0.9281 | even windows (stride 2) 0.9309 | odd 0.9233 | mean(even, odd) 0.9285; stride-2 vs all windows: Spearman 0.9955 (even) / 0.9957 (odd), mean |diff| 0.011 -> STRIDE 2 IS SAFE (use leg v4 with stride 2 to halve a member's GPU time). BUT the leg path does NOT reproduce the TPU evaluation: mean |prob diff| 0.018, max 0.251 (gold -0.0025). The cache builder's reader (scratch build_dense_cache_part.orig.py) is logically identical to the leg's and stores lossless npz, and windows()/masked_predict match -> suspects: (a) different volumes from library versions (pydicom/cv2) or DICOM glob order ties, (b) TPU numerics (PyTorch/XLA's default matmul precision on TPU may be bf16 even for fp32 models). Byte check of cache vs leg volumes for the gold studies running (krittanutsomtuas/rsna-knee-cache-vs-leg).
2026-10-01 00:50 CACHE vs LEG (krittanutsomtuas/rsna-knee-cache-vs-leg, free CPU): all 58 gold studies' s96 TRAINING-cache volumes and masks are BYTE-IDENTICAL to fleet leg v3's reader output (max pixel diff 0; Kaggle image pydicom 3.0.2, cv2 4.13.0, numpy 2.0.2; cache geometry 24/18/18/18/18, span 2-98%, 140 mm, 336 px). => the leg's inputs are exact; the 0.018 mean prediction gap between the TPU evaluation and the leg path is numerical (TPU matmuls at XLA's default bf16 precision during training/eval vs fp32/fp16 at inference); gold 0.9306 vs 0.9281 is within noise for 58 cases. Saturday's model108 (model104 alone, GPU fp16 inference) is the real test of TPU-trained models on the LB. Stride-2 windows (leg v4) are cleared for future legs with more members.
2026-10-01 05:05 MODEL110 SESSION 1 (krittanutsomtuas/rsna-knee-tpu-train-model110-f012) COMPLETE, 30,269 s: smoke OK; fold 0 only. Fold 0 epochs: 1 = 5,594 s and 2 = 3,308 s (cold /tmp cache; plan 18 -> 4 -> 7), 3 = 913 s (plan -> 18), 4-14 = 343-428 s, relaunches after 7 and 14 OK; epoch 15 = 3,743 s and 16 = 10,038 s (host MemAvailable 166 -> 114 GB inside process 3) -> time guard cut the plan to 16 epochs; "fold 1 not started" (old kernel rule). Fold 0 result: gold 0.9343 at epoch 16 (0.9357 ep 14, 0.9363 ep 15) = our BEST single model on gold, OOF for 832 studies, best.pt sha256 b837d3c7... -> fleet weights as model110_fold0_best.pt. The teammate's TPU is now over its weekly 20 h (push refused: "Maximum weekly TPU quota of 20.00 hours reached") until Sat 07:00; scratch tpu_chain2.py keeps retrying the push of model110-f1234-then-model112 (built with the steady-estimate + /dev/shm kernel). Trainer: per-epoch host memory breakdown added to history (shm, page cache, python RSS, top processes) to find what eats ~50 GB in late processes.
SATURDAY PLAN (5 slots): 07:01 runner: model108 (model104 alone), model107 (leg 3 members, v3), model111 (leg 4, v3); 07:40 runner: model120 (leg v4 stride 2: model110f0, model104, model75, model100) and model121 (leg v4 stride 2, six: + model109 + model98). Group readouts model114-119 move to Sunday+.
2026-10-02 11:30 AGENTS.md: "You can use the rest of kaggle api gpu (except for KAGGLE_API_KEY_YODHAIRNMONPLUB) for training as much as possible". .env key names unchanged; mapping unchanged (WARIN = krittanutsomtuas, KRITTANUT = actuarialtutor). Quotas now: krittanutsomtuas GPU 13,736 / 21,600 s, TPU 81,354 / 72,000 s (over, until Sat); actuarialtutor GPU 40,834 / 21,600 s (over - other project), TPU 0, still not joined to RSNA (403). Vast credit 0 (balance -4.33 USD). T4x2 trains CoAtNet-2 384 ~6x slower than TPU v5e-8 (~8-10 h per model) -> Kaggle GPU hours go to short jobs (head retrains, pseudo-label tests); TPU stays the main lane.
2026-10-02 11:51 GPU STRIDE CHECK (krittanutsomtuas/rsna-knee-leg-window-parity-gpu, T4 fp16 autocast = the leg's numerics, 58 gold, leg v3 code): model110f0 (s96) gold all 0.9342 / even 0.9333 / TPU eval 0.9343 (mean |all - TPU| 0.008), Spearman even-vs-all 0.995; model109 (s64) 0.9113 / 0.9079 / 0.9126 (Spearman 0.992); model98 (s64) 0.9199 / 0.9174 / 0.9154 (0.991). => on GPU fp16 the leg reproduces the TPU evaluation closely; stride 2 is ~free for s96 (94 -> 47 windows) but costs ~0.003 gold for s64 (62 -> 31). model120/model121 rebuilt with model75 (s64, strongest LB member) at stride 1, all others stride 2. GPU time 171 s (s96) / ~122 s (s64) for 58 studies x (all + even windows).
2026-10-03 07:10-07:47 SUBMITTED (all dummy runs verified, leg 'dropped for time []'): model108 56784991 (model104 alone), model107 56785116 (leg v3 model104+model75+model100), model111 56785295 (+model98), model120 56785892 (leg v4: model110f0/2, model104/2, model75/1, model100/2), model121 56785905 (+model109/2, model98/2). Main GPU after 3 dummy runs: 872 s of 21,600 (~5 min each). TPU session 2 (model110-f1234-then-model112) pushed 07:00 by tpu_chain2, queued.
MODEL108 SCORED 2026-10-03 08:03: public 0.942 = model104 ALONE (TPU CoAtNet-2 384, 96 frames, gold 0.9306) -> best own single model on the LB (model75 0.939, model100 0.937 partial, Speedy public 0.943). TPU training + GPU fp16 inference transfer fine; s96 > s64 for this recipe. The leg results (model107/111/120/121) arrive later (~7 h hidden runs).
2026-10-03 10:40 .env KEYS RENAMED by the user to KAGGLE_API_KEY_<e-mail>: yodharinmonplub@actuarialtutor.org -> yodharinmonplub (main, submit only), warinaeinwandee@... -> krittanutsomtuas, krittanutsomtung@... -> actuarialtutor, aikayui@... -> aikayui (NEW 4th account; not joined to RSNA - 403; GPU busy with other projects; TPU never used). AGENTS.md now names the main key by e-mail. 114 .py files (Other/scripts, cloud, Model copies, scratch tools) sed-renamed, all compile; tick + lb_rank verified. LB 10:40: rank 133 at 0.947 (tied block of 26).
MODEL107 SCORED 2026-10-03 14:40: public 0.947 = model103 (0.947) + model104 in the leg (v3, w 0.30) -> tie at 3 decimals.
MODEL121 SCORED 2026-10-03 14:55: public 0.947 (six-member leg v4 stride 2, w 0.30; no timeout). 2, 3 and 6 strong members all 0.947 at w 0.30 -> the weight, not the membership, is the next lever.
MODEL111 SCORED 2026-10-03 15:26: public 0.948 = NEW PROJECT BEST. model76 + leg v3 (full windows) [model104, model75, model100, model98] at w 0.30. (model107 3 members 0.947, model121 6 members stride 2 0.947, model103 2 members 0.947.)
SUNDAY 2026-10-04 07:01 ARMED (scratch sunday_runner.py): model126 = model111's leg @ w 0.40, model127 = @ w 0.50, model129 = model111's four + model110f0 (leg v4, s96 at stride 2, s64 full) @ w 0.30, model130 = model129 @ w 0.40, model123 = six-member stride-2 leg @ w 0.40. Questions: leg weight now that our side is strong (model104 alone 0.942), and whether model110f0 adds to the best leg. LB 15:26: rank 118 at 0.948; silver line now 0.945 (rank 248), bronze 0.943, gold 0.957.
MODEL120 SCORED 2026-10-03 15:42: public 0.947 (leg v4: model110f0/2, model104/2, model75/1, model100/2 at w 0.30). Saturday summary: 108 0.942 | 107 0.947 | 111 0.948 (best; full windows) | 120 0.947 | 121 0.947 -> full-window legs >= stride-2 legs (maybe noise).
SUNDAY PLAN REVISED 15:50 (stride-2 legs 0.947 vs the full-window model111 0.948): runner re-armed with model126 (model111 leg @ 0.40), model127 (@ 0.50), model131 (@ 0.35), model129 (model111 leg with model110f0 instead of model100, full windows, @ 0.30), model130 (same @ 0.40). model122-125 (stride-2 weight variants) built but not scheduled.
2026-10-03 16:06 TPU SESSION 2 (model110-f1234-then-model112, started 15:01 after ~8 h queued): /dev/shm FIX WORKS - cache unpacked in 778 s (was 1,188 s on /tmp), fold 1 epoch 1 = 668 s (compile; was 5,594 s), epochs 2-5 = 343-346 s (steady from epoch 2, plan stays 18). Memory breakdown: shm 47.8 GB (the cache), 8 chip processes ~28-29 GB RSS each (python RSS 215 -> 226 GB over 5 epochs, +~0.3 GB/epoch/process), MemAvailable 127.5 -> 117 GB; relaunches every 7 epochs should reset the growth. Fold 1: 3,494 train / 851 OOF.
2026-10-03 18:43 SESSION 2 PROGRESS: fold 1 COMPLETE, 18 epochs (gold 0.9258 at epoch 18; 0.9284 ep 12) in 8,822 s; epochs 342-383 s, 637-673 s right after the relaunches; process 3 crashed once after epoch 15 (one chip process got SIGTERM -> BrokenProcessPool; MemAvailable was 118 GB) and the kernel resumed from last.pt (retry 1) -> epochs 16-18 done. MemAvailable resets at each relaunch (112 -> 152 GB), no late slowdown. Fold 2 running (350-360 s/epoch). steady_need() estimates 8,904 s per fold -> this session should finish folds 1-3; fold 4 + model112 next: scratch tpu_chain2.py re-armed to collect session 2 and push model110-f4-then-model112.
2026-10-03 23:10 SESSION 2 COMPLETE (PLAN OK): model110 folds 1-3 done, 18 epochs each: gold fold1 0.9258, fold2 0.9178, fold3 0.9262 (fold0 0.9343, 16 epochs); OOF 851 / 865 / 905 (+832). Session 3 (fold 4 + model112) pushed 23:05 by tpu_chain2.
PSEUDO-LABEL MIX ON GOLD (folds 0-3; teacher = mean of the 4 fold models' gold predictions; scripts/make_pseudo_targets.py): labels alone 0.9035 | teacher alone 0.9319 | alpha (label weight) 0.1 0.9396 | 0.2 0.9419 | 0.3 0.9429 | 0.4 0.9408 | 0.5 0.9386. The CoAtNet-2 teacher lifts the targets far more than the old ResNet34 teacher (mix 0.9316). Choice: model113 alpha 0.35 (OOF per study comes from one fold model, noisier than the 4-model mean), model132 = second student at alpha 0.5 (seed 20261018), both trained in session 4 (--then).
2026-10-04 07:20-07:29 SUBMITTED (dummy runs verified): model126 56809947 | model127 56809960 | model131 56809974 | model129 56809987 | model130 56810110 (model126/127/131 = model111 leg at w 0.40/0.50/0.35; model129/130 = model110f0 instead of model100 at w 0.30/0.40).
