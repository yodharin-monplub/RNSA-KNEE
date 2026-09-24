Model20: conservative last-stage fine-tuning of model15 native384 CoAtNet.

Goal: reproduce model16's successful weak-label fine-tuning recipe on the
complementary native384 checkpoint, then test its blend with model16/model19.
Initial checkpoint: dreaddevelopment/raptor-knee-native384dense,
raptor_ft_coatnet_v10_full.pt. Architecture and geometry credit: Johnathan
Wagner / Dread Development; see SOURCES.md.

Geometry: 64 slices, 2%-98% stack span, 140mm crop, direct 384x384 intermediate
pixels. A new private four-part cache is built on Kaggle CPU and must pass a
cached-versus-raw prediction parity gate before paid training. Do not train this
checkpoint on the existing 336-pixel cache.

Training recipe: same fixed partition, Steven Lee v4 weak probabilities,
confidence-weighted BCE, frozen teacher penalty 0.3, 12 random windows per
study, two epochs, last backbone stage plus prediction head trainable. Select
by weak-validation BCE including epoch zero. Gold58 is descriptive only and is
not used for optimizer updates or checkpoint selection. Upstream labels and
checkpoints used these development cases, so it is not untouched validation.

Budget: up to $3 within the existing $20 Vast.ai campaign limit. No local GPU.
Download, manifest, geometry, checkpoint, smoke, budget watchdog, backup, and
rental cleanup gates are required before or during launch. No public score yet.

Pilot status: exact384 cache pilot COMPLETE. All64 studies, four shard hashes,
shapes, masks and metadata passed local unpack validation. Estimated full cache
is29.38GB compressed and70.97GB including unpacked arrays.

GPU gates: cached-versus-raw parity PASS on58 development cases, maximum
absolute probability difference0.0002834204 under the predeclared0.001 limit.
Two-case forward/backward smoke PASS with finite loss0.7767370 and23,143,240
trainable parameters. Initial checkpoint SHA-256:
5add8816a47e040efca89bb44480e1cbd01f92db1253831ff7ceb6c65ee02b1d.
The four full384 cache-part jobs completed on Kaggle CPU; their manifests and
combined UID/hash/disk-budget validation passed before rental creation.

Full cache gate passed:4,407 unique studies,276 unique shards,29.523GB
compressed and71.112GB compressed plus unpacked payload. All four metadata
hashes, part counts, geometry records and UID coverage match.

Vast attempt1 used instance50576292 with one RTX3090,120GB disk and a quoted
$0.18/hour rate. It failed before data download or training because the host's
DNS temporarily could not resolve the Python package index. The watchdog backed
up its logs and destroyed the rental. Observed credit decrease was$0.010060
(conservative timed estimate$0.012406). The retry reserves at most$2.98 so the
combined model20 attempts remain within the authorized$3 allocation. The
installer now makes five bounded attempts with longer network retries.

Vast attempt2 uses instance50577604 with one RTX3090,120GB disk, a quoted
$0.166667/hour rate, a six-hour deadline, and a hard$2.98 retry cap. Local
watchdog exec session69633 will back up artifacts and destroy the rental after
completion, failure, or the cap. Campaign spending before this retry is
$0.741305 on the conservative accounting used by the readiness gate.

Attempt2 completed successfully and the watchdog verified the backup before
destroying instance50577604. Epoch1 was selected: weak-validation BCE improved
from0.465390 at epoch0 to0.368714, while epoch2 reached0.371890. Descriptive
gold AUC increased from0.916974 at epoch0 to0.919209 for the selected checkpoint.
Checkpoint SHA-256:
5087c36765aeb9fdf19881c778edd905eb3e91ee1cd78defb8b90175e8924a75.
Attempt2 estimated compute was$0.232232; observed credit decrease including
data transfer was$0.318864. Conservative campaign spending through model20 is
about$1.0602, well under the authorized$20 cap.

Fixed-blend development analysis found model16/model20 equal averaging at
macro AUC0.926461, compared with0.926221 for model19. Its paired bootstrap
95% delta interval versus model19 is[-0.006340,0.006543], so the apparent gain
is uncertain. Model21 packages this equal blend for private GPU validation.
