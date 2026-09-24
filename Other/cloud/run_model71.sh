#!/usr/bin/env bash
set -euo pipefail
cd /workspace/rsna
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export RSNA_CLOUD_RUN=model71
mkdir -p artifacts
exec 9>artifacts/run.lock
flock -n 9 || exit 0
if [[ -e artifacts/RUN_FAILED || -e artifacts/model71/TRAINING_COMPLETE ]]; then
  exit 0
fi
trap 'touch artifacts/RUN_FAILED' ERR
: > artifacts/install.log
installed=0
for attempt in 1 2 3 4 5; do
  if python -m pip install --retries 10 --timeout 60 -r requirements-model16.txt >> artifacts/install.log 2>&1; then
    installed=1
    break
  fi
  printf 'Dependency install attempt %s failed; retrying.\n' "$attempt" >> artifacts/install.log
  sleep $((attempt * 15))
done
if [[ "$installed" -ne 1 ]]; then
  printf 'Dependency installation failed after five attempts.\n' >> artifacts/install.log
  exit 1
fi
python -m pip freeze > artifacts/environment.txt
python cloud/download_dense.py > artifacts/download.log 2>&1
python cloud/prepare_dense_cache.py --manifest data/dense_shards/part*/cache_manifest.json --destination data/dense > artifacts/cache_prepare.log 2>&1
python scripts/train_model16.py --cache data/dense --config model71/config.json --fold 0 --checkpoint none --output artifacts/smoke --smoke > artifacts/smoke.log 2>&1
# Five folds, one after another on the same rental; each writes its own checkpoint and out-of-fold predictions.
# A fold already carrying TRAINING_COMPLETE is skipped, so the run resumes if the rental is stopped and restarted.
for fold in 0 1 2 3 4; do
  if [[ -e "artifacts/model71/fold${fold}/TRAINING_COMPLETE" ]]; then
    continue
  fi
  python scripts/train_model16.py --cache data/dense --config model71/config.json --fold "${fold}" \
    --checkpoint none --output "artifacts/model71/fold${fold}" > "artifacts/train_fold${fold}.log" 2>&1
done
touch artifacts/model71/TRAINING_COMPLETE
