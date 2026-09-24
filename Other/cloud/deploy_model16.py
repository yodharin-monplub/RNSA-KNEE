"""Deploy the bounded model16 run; only the Kaggle token leaves this machine."""
import json
import os
import shlex
import subprocess
import tarfile

from vast import ROOT, STATE, RUN_NAME, ssh_command, status


def main():
    assert RUN_NAME == 'model16', 'Set RSNA_CLOUD_RUN=model16'
    state = status(json.loads((STATE/'instance.json').read_text()))
    assert state['actual_status']=='running'
    command = ssh_command(state)
    subprocess.run(command+['mkdir -p /workspace/rsna/artifacts'],check=True,timeout=30)
    archive=STATE/'model16.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        names=['knee','model12','model16','requirements-model16.txt','SOURCES.md',
               'scripts/train_model16.py','cloud/download_dense.py','cloud/prepare_dense_cache.py',
               'cloud/run_model16.sh','data/train.csv','data/sample_submission.csv',
               'data/weak_labels_v4/llm_labels_v4_blend.csv','artifacts/partitions']
        for name in names:
            tar.add(ROOT/name,arcname=name,filter=lambda info:None if '__pycache__' in info.name else info)
        tar.add(STATE/'ready.json',arcname='cloud/model16_ready.json')
    with archive.open('rb') as stream:
        subprocess.run(command+['tar xzf - -C /workspace/rsna'],stdin=stream,check=True,timeout=120)
    token=os.environ.get('KAGGLE_API_TOKEN') or os.environ['KAGGLE_API_KEY']
    assert '\n' not in token and '\r' not in token
    subprocess.run(command+['umask 077; cat > /workspace/rsna/.env'],
                   input=('KAGGLE_API_TOKEN='+token+'\n').encode(),check=True,timeout=30)
    subprocess.run(command+['command -v rsync || (apt-get update -qq && apt-get install -y -qq rsync)'],
                   check=True,timeout=180)
    start="import subprocess; f=open('artifacts/launcher.log','ab'); subprocess.Popen(['bash','cloud/run_model16.sh'],stdin=subprocess.DEVNULL,stdout=f,stderr=f,start_new_session=True)"
    subprocess.run(command+['cd /workspace/rsna && python -c '+shlex.quote(start)],check=True,timeout=30)
    print('Model16 deployed. Watchdog backs up to artifacts/remote_model16.')


if __name__=='__main__':main()
