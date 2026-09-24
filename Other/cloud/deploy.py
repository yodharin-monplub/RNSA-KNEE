"""Upload only the project bundle and a Kaggle-only credential to our rental."""
import json
import os
import shlex
import subprocess
import tarfile

from vast import ROOT, STATE, ssh_command, status


def main():
    state = status(json.loads((STATE / 'instance.json').read_text()))
    command = ssh_command(state)
    subprocess.run(command + ['mkdir -p /workspace/rsna'], check=True, timeout=30)
    archive = STATE / 'pilot.tar.gz'
    with tarfile.open(archive, 'w:gz') as tar:
        for name in ['knee', 'model1', 'model2', 'model3', 'requirements-gpu.txt', 'SOURCES.md',
                     'cloud/download_pixels.py', 'cloud/run_pilot.sh',
                     'data/train.csv', 'data/train_series.csv', 'data/test.csv',
                     'data/test_series.csv', 'data/sample_submission.csv',
                     'data/weak_labels/llm_labels_full.csv', 'artifacts/partitions']:
            tar.add(ROOT / name, arcname=name, filter=lambda info: None if '__pycache__' in info.name else info)
    with archive.open('rb') as stream:
        subprocess.run(command + ['tar xzf - -C /workspace/rsna'], stdin=stream, check=True, timeout=120)
    # No Vast/OpenRouter/RunPod keys leave the local workspace.
    token = os.environ.get('KAGGLE_API_TOKEN') or os.environ['KAGGLE_API_KEY']
    if '\n' in token or '\r' in token:
        raise ValueError('Unexpected credential format')
    subprocess.run(command + ['umask 077; cat > /workspace/rsna/.env'],
                   input=('KAGGLE_API_TOKEN=' + token + '\n').encode(), check=True, timeout=30)
    subprocess.run(command + ['command -v rsync || (apt-get update -qq && apt-get install -y -qq rsync)'],
                   check=True, timeout=180)
    start = "import subprocess; f=open('artifacts/pilot.log','ab'); subprocess.Popen(['bash','cloud/run_pilot.sh'],stdin=subprocess.DEVNULL,stdout=f,stderr=f,start_new_session=True)"
    subprocess.run(command + ['cd /workspace/rsna && python -c ' + shlex.quote(start)], check=True, timeout=30)
    print('Pilot deployed and started. Logs are copied to artifacts/remote by the watchdog.')


if __name__ == '__main__':
    main()
