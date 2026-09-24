#!/usr/bin/env bash
set -euo pipefail
cd /workspace/rsna
export PYTHONUNBUFFERED=1
while [ ! -f artifacts/PILOT_COMPLETE ]; do sleep 15; done
python -m knee.finetune --config model5/config.json > artifacts/model5.log 2>&1
touch artifacts/FINETUNE_COMPLETE
while [ ! -f artifacts/INFERENCE_VALIDATED ]; do sleep 15; done
python cloud/finish_ensemble.py > artifacts/ensemble_validation.log 2>&1
touch artifacts/ENSEMBLE_VALIDATED
