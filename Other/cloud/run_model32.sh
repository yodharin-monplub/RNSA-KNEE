#!/usr/bin/env bash
set -euo pipefail
cd /workspace/rsna
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export RSNA_CLOUD_RUN=model32
mkdir -p artifacts
exec 9>artifacts/run.lock
flock -n 9 || exit 0
if [[ -e artifacts/RUN_FAILED || -e artifacts/model32/TRAINING_COMPLETE ]]; then
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
python cloud/prepare_dense_cache.py --manifest data/dense_shards/part*/cache_manifest.json --destination data/dense --size 384 > artifacts/cache_prepare.log 2>&1
python scripts/train_model16.py --cache data/dense --cache-size 384 --config model32/config.json --checkpoint data/weights/raptor_ft_coatnet_v10_full.pt --output artifacts/smoke --smoke > artifacts/smoke.log 2>&1
python scripts/train_model16.py --cache data/dense --cache-size 384 --config model32/config.json --checkpoint data/weights/raptor_ft_coatnet_v10_full.pt --output artifacts/model32 > artifacts/train.log 2>&1

