Model14: fixed equal rank blend of model12 and model13
Status: private offline Kaggle version 1 COMPLETE and output validated.
58-case development AUC 0.913246 versus model12 0.919929. Not prioritized
for leaderboard submission. Paired difference interval overlaps zero.
Weight fixed before DINOv3 gold evaluation finishes: 0.5 CoAtNet ranks,
0.5 DINOv3 ensemble ranks, computed per label across available test studies.
No weight optimization. Check alignment and branch equivalence before scoring.
Embedded component scripts run sequentially in separate processes; CoAtNet
runs before DINOv3 installs its pinned timm wheel. All input preprocessing is
inherited from model12 and model13. Component outputs retained for validation.
Attribution: model12/readme.txt, model13/NOTICE.txt, SOURCES.md. Full Apache-2.0
and DINOv3 license texts are embedded and emitted with notebook outputs.
No leaderboard submission or new training. Gold development is not an untouched
holdout; public checkpoint authors used these cases and overlap is unverified.
