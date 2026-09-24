"""Deploy a bounded managed training run (RSNA_CLOUD_RUN=modelNN); only the Kaggle token leaves this machine."""
import json
import os
import shlex
import subprocess
import tarfile

from vast import ROOT, STATE, RUN_NAME, MANAGED, ssh_command, status


def main():
    assert MANAGED and (ROOT / f'cloud/run_{RUN_NAME}.sh').exists(), 'Set RSNA_CLOUD_RUN to a prepared run'
    state = status(json.loads((STATE / 'instance.json').read_text()))
    assert state['actual_status'] == 'running'
    command = ssh_command(state)
    subprocess.run(command + ['mkdir -p /workspace/rsna/artifacts'], check=True, timeout=30)
    archive = STATE / f'{RUN_NAME}.tar.gz'
    with tarfile.open(archive, 'w:gz') as tar:
        names = ['knee', 'model12', RUN_NAME, 'requirements-model16.txt', 'SOURCES.md',
                 'scripts/train_model16.py', 'cloud/download_dense.py', 'cloud/prepare_dense_cache.py',
                 'cloud/remote_upload.py',
                 f'cloud/run_{RUN_NAME}.sh', 'data/train.csv', 'data/sample_submission.csv',
                 'data/weak_labels_v4/llm_labels_v4_blend.csv', 'artifacts/partitions']
        if RUN_NAME == 'model47':
            # DINO member fine-tuning needs its own modules, requirements and pinned backbone config.
            names += ['cloud/download_dino.py', 'cloud/prepare_dino_cache.py', 'cloud/download_dense.py',
                      'scripts/train_dino_members.py', 'requirements-model47.txt', 'third_party/dinov2']
        if RUN_NAME in ('model64', 'model66'):
            names.append('cloud/download_external.py')
        if RUN_NAME == 'model62':
            # External-data cache build: only the builder, the two signed fastMRI links and the annotations.
            names = [RUN_NAME, 'cloud/build_fastmri_cache.py', 'cloud/upload_fastmri_cache.py',
                     f'cloud/run_{RUN_NAME}.sh', 'data/external/fastmri/links_knee.json',
                     'data/external/fastmri_plus']
        run_targets = json.loads((ROOT / RUN_NAME / 'config.json').read_text()).get('targets')
        if run_targets and run_targets not in names:
            names.append(run_targets)
        for name in names:
            tar.add(ROOT / name, arcname=name,
                    filter=lambda info: None if '__pycache__' in info.name else info)
        tar.add(STATE / 'ready.json', arcname=f'cloud/{RUN_NAME}_ready.json')
    with archive.open('rb') as stream:
        subprocess.run(command + ['tar xzf - -C /workspace/rsna'], stdin=stream, check=True, timeout=120)
    token = os.environ.get('KAGGLE_API_TOKEN') or os.environ['KAGGLE_API_KEY']
    assert '\n' not in token and '\r' not in token
    subprocess.run(command + ['umask 077; cat > /workspace/rsna/.env'],
                   input=('KAGGLE_API_TOKEN=' + token + '\n').encode(), check=True, timeout=30)
    start = ("import subprocess; f=open('artifacts/launcher.log','ab'); "
             f"subprocess.Popen(['bash','cloud/run_{RUN_NAME}.sh'],stdin=subprocess.DEVNULL,"
             "stdout=f,stderr=f,start_new_session=True)")
    subprocess.run(command + ['cd /workspace/rsna && python -c ' + shlex.quote(start)], check=True, timeout=30)
    print(f'{RUN_NAME} deployed. Watchdog backs up to artifacts/remote_{RUN_NAME}.')


if __name__ == '__main__':
    main()
