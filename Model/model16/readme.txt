Model16: conservative last-stage CoAtNet fine-tuning
Status: full two-epoch cloud training completed on attempt2, instance50524988.
Selected epoch1 by weak-validation BCE: 0.3640076564 versus baseline0.4572973400
and epoch2 0.3665926500. Gold58 development AUC0.9230802364 versus starting
model12 0.9199286631. These are development results, not leaderboard scores.
Checkpoint and all artifacts backed up; instance destroyed and empty Vast
inventory confirmed. Attempt2 observed credit decrease$0.360370.
Private Kaggle weights uploaded; GPU inference v1 COMPLETE, three visible
studies passed output schema and finite-probability checks. Public score pending.
Initialize from model12 Max-Span checkpoint. Use exact verified dense cache.
Train only the last backbone stage and prediction head, with 12 random windows
per study, 2 epochs, small learning rates and a frozen model12 teacher penalty.
Targets: published Steven Lee v4 report probabilities. Confidence2*abs(p-0.5).
Do not directly average the audited Raptor/v1 targets with differing severity
conventions. V4 is a separate published target scheme, not an assertion of truth.
Retain existing normalized-report-group train/weak_val/gold partition.
Gold58 cases and their report duplicates receive no optimizer updates here.
The pretrained checkpoint and label publishers used development cases; neither
weak validation nor gold should be represented as untouched external evidence.
Select checkpoint by weak-validation BCE, including unchanged epoch0. Gold
evaluation is descriptive only. Preserve the baseline if fine-tuning hurts.
Frozen teacher on the same sampled windows limits forgetting; weight fixed0.3.
Before training: full-cache hashes/IDs/geometry, cached-vs-raw inference, GPU
forward/backward smoke on weak training cases, and remaining rental budget.
Initial paid-run allocation at most$3 within the existing total$20 cap; rental
disk/runtime guard must be configured before launch. No automatic purchases.
Cloud launch uses RSNA_CLOUD_RUN=model16, separate state and backup directories,
automatic watchdog startup, manifest/coverage/smoke readiness gates and a100GB
disk request. Remote execution is protected against duplicate launches by flock.
Source and third-party provenance: SOURCES.md, model12/readme.txt.

Attempt1: download SSL failure before training; instance50515330 destroyed after backup. Observed cost$0.277219. Retry instance50524988 with improved per-shard downloader,$2.70cap, quote$0.174444/hour.

Confirmed Kaggle public AUC: 0.932, submission56153815, version1.
Improves previous campaign best model10 0.929 by0.003.
