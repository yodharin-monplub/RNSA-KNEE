#!/usr/bin/env bash
set -euo pipefail
cd /workspace/rsna
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export HF_HOME=/workspace/rsna/.cache/huggingface
export KAGGLEHUB_CACHE=/workspace/rsna/.cache/kagglehub
mkdir -p artifacts
python -m pip install -r requirements-gpu.txt > artifacts/install.log 2>&1
python -m pip freeze > artifacts/environment.txt
python cloud/download_pixels.py > artifacts/download.log 2>&1
python -m knee.extract > artifacts/extract.log 2>&1
python -m knee.train_heads --config model2/config.json > artifacts/model2.log 2>&1
python -m knee.train_heads --config model3/config.json > artifacts/model3.log 2>&1
touch artifacts/PILOT_COMPLETE
