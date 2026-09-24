Model22: conservative last-stage fine-tuning of model10 fine-spacing CoAtNet.

Goal: apply the model16/model20 weak-label fine-tuning recipe to the public
checkpoint that produced the strongest standalone public result in this
campaign (model10, public AUC0.929), then test complementary blends with the
current0.935 models. Initial checkpoint:
dreaddevelopment/raptor-knee-finespacing/raptor_ft_coatnet_v9_full.pt.

Geometry follows the model10 transfer pipeline used for its public score:
64 slices, 6%-94% stack span, 140mm crop, 336x336 intermediate pixels, then
384x384 model input. The publisher did not fully document the original training
geometry, so this is a fine-tuning experiment on the exact geometry used by our
scored model10 inference. A new private cache must pass cached-versus-raw parity
before paid training; the existing2%-98% cache is not valid for this model.

Training recipe: fixed report-disjoint partition, Steven Lee v4 weak
probabilities, confidence-weighted BCE, frozen-teacher penalty0.3,12 random
windows per study, two epochs, last backbone stage plus prediction head
trainable. Select by weak-validation BCE including epoch0. Gold58 is descriptive
only and is excluded from optimizer updates and checkpoint selection.

Budget: target$3 or less within the existing$20 Vast.ai campaign authorization.
Conservative campaign spending before model22 is about$1.0602. No local GPU.
Pilot cache, cached/raw parity, GPU smoke, full-manifest coverage, disk/transfer,
budget watchdog, artifact backup and rental cleanup gates are required.

Exact6%-94% pilot cache version1 completed. All64 study IDs, four shard hashes,
uint8 shapes, masks, source metadata and geometry passed local unpack validation.
Cached-versus-raw parity PASS on58 cases with maximum absolute probability
difference0.0005463 under the predeclared0.001 tolerance. Two-case GPU training
smoke PASS with finite loss0.784561 and23,143,240 trainable parameters. Initial
checkpoint SHA-256:
0850149a98cf04c7bacc1ea99b3af048102a256de14766b129b1d7255c02de7b.
All four full-cache part jobs completed. Their manifests cover4,407 unique UIDs
and276 distinct shards with23.363GB compressed and31.842GB unpacked. Exact
geometry, part counts, metadata hashes, disk, transfer and campaign-budget gates
passed. Paid launch code is prepared and syntax-validated.

The first launch request created no rental because the account already has an
unrelated running instance,50610014, labeled arc-model28-partial-unfreeze-retry.
The safety guard correctly refused to create overlapping compute. Model22 will
wait for the account inventory to clear; it will not stop or modify that other
workload.

2026-09-16/17 run (Windows-ported tooling): Vast instance 51226491, RTX3090,
$0.174444/h quote. The launch guard now ignores other projects' instances. Download,
unpack, rental-local smoke and two epochs completed; watchdog backed up and destroyed.
Selected epoch1 by weak-validation BCE: 0.472695 (epoch0) -> 0.380618 (epoch1),
0.382169 (epoch2). Descriptive gold AUC 0.909689 (epoch0) -> 0.915761 (selected).
Estimated compute $0.342843; observed credit decrease $0.468578.
Artifacts: artifacts/remote_model22/model22. Weights packaged as private dataset
yodharinmonplub/rsna-knee-model22-weights.
