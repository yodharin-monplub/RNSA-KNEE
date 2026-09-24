#!/usr/bin/env bash
set -euo pipefail
cd /workspace/rsna
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export RSNA_CLOUD_RUN=model47
mkdir -p artifacts
exec 9>artifacts/run.lock
flock -n 9 || exit 0
if [[ -e artifacts/RUN_FAILED || -e artifacts/model47/TRAINING_COMPLETE ]]; then
  exit 0
fi
trap 'touch artifacts/RUN_FAILED' ERR
: > artifacts/install.log
installed=0
for attempt in 1 2 3 4 5; do
  if python -m pip install --retries 10 --timeout 60 -r requirements-model47.txt >> artifacts/install.log 2>&1; then
    installed=1
    break
  fi
  sleep $((attempt * 15))
done
if [[ "$installed" -ne 1 ]]; then exit 1; fi
python -m pip freeze > artifacts/environment.txt
python cloud/download_dino.py > artifacts/download.log 2>&1
python cloud/prepare_dino_cache.py --manifest data/dino_shards/part*/cache_manifest.json --destination data/dino > artifacts/cache_prepare.log 2>&1
python scripts/train_dino_members.py --cache data/dino --weights data/dino_members --config model47/config.json --output artifacts/smoke --smoke > artifacts/smoke.log 2>&1
python scripts/train_dino_members.py --cache data/dino --weights data/dino_members --config model47/config.json --output artifacts/model47 > artifacts/train.log 2>&1