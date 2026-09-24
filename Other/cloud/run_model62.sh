#!/usr/bin/env bash
set -euo pipefail
cd /workspace/rsna
export PYTHONUNBUFFERED=1
mkdir -p artifacts
export RSNA_CLOUD_RUN=model62
exec 9>artifacts/run.lock
flock -n 9 || exit 0
if [[ -e artifacts/RUN_FAILED || -e artifacts/model62/TRAINING_COMPLETE ]]; then
  exit 0
fi
trap 'touch artifacts/RUN_FAILED' ERR
python -m pip install h5py kaggle numpy requests > artifacts/install.log 2>&1
# The two archives decompress independently (xz is single-threaded), so run them side by side.
python cloud/build_fastmri_cache.py knee_singlecoil_val > artifacts/build_val.log 2>&1 &
VAL=$!
python cloud/build_fastmri_cache.py knee_singlecoil_train > artifacts/build_train.log 2>&1 &
TRAIN=$!
wait $VAL
wait $TRAIN
python cloud/build_fastmri_cache.py merge > artifacts/build.log 2>&1
python cloud/upload_fastmri_cache.py > artifacts/upload.log 2>&1
cp artifacts/model62/manifest.json artifacts/model62/history.json
touch artifacts/model62/TRAINING_COMPLETE
