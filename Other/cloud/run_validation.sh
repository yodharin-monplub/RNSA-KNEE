#!/usr/bin/env bash
set -euo pipefail
cd /workspace/rsna
export PYTHONUNBUFFERED=1
python cloud/download_test.py > artifacts/test_download.log 2>&1
while [ ! -f artifacts/PILOT_COMPLETE ]; do sleep 15; done
python cloud/validate_gpu.py > artifacts/inference_validation.log 2>&1
touch artifacts/INFERENCE_VALIDATED
