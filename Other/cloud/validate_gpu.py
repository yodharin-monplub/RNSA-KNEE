"""Run after training: validate real DICOM inference and make a weights bundle."""
import json
import os
import shutil
import subprocess
from pathlib import Path

out = Path('artifacts/inference_bundle')
out.mkdir(parents=True, exist_ok=True)
shutil.copy('artifacts/features/encoder.pt', out / 'encoder.pt')
for name in ['model2', 'model3']:
    (out / name).mkdir(exist_ok=True)
    shutil.copy(f'artifacts/{name}/head.pt', out / name / 'head.pt')
env = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
           HF_HOME='/tmp/rsna-offline-empty-cache')
subprocess.run(['python', '-m', 'knee.infer', '--data', 'data', '--weights', str(out),
                '--output', 'artifacts/submission.csv'], check=True, env=env)
subprocess.run(['python', '-m', 'unittest', 'discover', '-s', 'tests', '-v'], check=True)
print('INFERENCE_VALIDATED', flush=True)
