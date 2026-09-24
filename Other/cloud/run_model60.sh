#!/usr/bin/env bash
set -euo pipefail
cd /workspace/rsna
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
mkdir -p artifacts
export RSNA_CLOUD_RUN=model60
exec 9>artifacts/run.lock
flock -n 9 || exit 0
if [[ -e artifacts/RUN_FAILED || -e artifacts/model60/TRAINING_COMPLETE ]]; then
  exit 0
fi
trap 'touch artifacts/RUN_FAILED' ERR
python -m pip install -r requirements-model16.txt > artifacts/install.log 2>&1
python -m pip freeze > artifacts/environment.txt
python cloud/download_dense.py > artifacts/download.log 2>&1
python cloud/prepare_dense_cache.py --manifest data/dense_shards/part*/cache_manifest.json --destination data/dense > artifacts/cache_prepare.log 2>&1
python scripts/train_model16.py --cache data/dense --checkpoint data/weights/raptor_ft_coatnet_v5_full_swa.pt --config model60/config.json --output artifacts/smoke --smoke > artifacts/smoke.log 2>&1
python scripts/train_model16.py --cache data/dense --checkpoint data/weights/raptor_ft_coatnet_v5_full_swa.pt --config model60/config.json --output artifacts/model60 > artifacts/train.log 2>&1
