"""Bounded, atomic, hash-verified cache downloads with per-shard retries."""
import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from dotenv import load_dotenv
ROOT = Path(__file__).resolve().parents[1]
RUN_NAME = os.environ.get('RSNA_CLOUD_RUN', 'model16')
# Later runs reuse an earlier run's verified cache and initial checkpoint.
DATA_RUN = {'model100': 'dense96', 'model90': 'model16', 'model86': 'dense128', 'model84': 'model16', 'model82': 'model16', 'model79': 'model16', 'model75': 'model16', 'model73': 'model16', 'model72': 'model16', 'model71': 'model16', 'model68': 'model16', 'model69': 'model16', 'model65': 'model16', 'model66': 'model16', 'model63': 'model16', 'model64': 'model16', 'model25': 'model16', 'model27': 'model16', 'model34': 'model16', 'model36': 'model16', 'model38': 'model16', 'model60': 'model16', 'model59': 'model16', 'model58': 'model16', 'model56': 'model22', 'model55': 'model28', 'model54': 'model20', 'model52': 'model16', 'model51': 'model16', 'model46': 'model16', 'model45': 'model16', 'model44': 'model16', 'model42': 'model16', 'model41': 'model22', 'model40': 'model28', 'model39': 'model20', 'model31': 'model28', 'model32': 'model20'}.get(RUN_NAME, RUN_NAME)
load_dotenv(ROOT/'.env')
os.environ.setdefault('KAGGLE_API_TOKEN', os.environ.get('KAGGLE_API_KEY', ''))
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest


def output_urls(api, slug):
    result = {}
    token = None
    with api.build_kaggle_client() as client:
        while True:
            req = ApiListKernelSessionOutputRequest()
            req.user_name = 'yodharinmonplub'; req.kernel_slug = slug
            api._set_paging(req, 100, token)
            page = client.kernels.kernels_api_client.list_kernel_session_output(req)
            for item in page.files or []:
                assert Path(item.file_name).name == item.file_name
                result[item.file_name] = item.url
            token = page.next_page_token
            if not token: return result


def verified(path, size, digest):
    if not path.is_file() or path.stat().st_size != size: return False
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest() == digest


def fetch(url, path, size, digest):
    if verified(path, size, digest): return
    temp = path.with_suffix(path.suffix+'.partial')
    for attempt in range(5):
        try:
            h = hashlib.sha256(); count = 0; started = time.monotonic()
            with requests.get(url, stream=True, timeout=(20, 60)) as response:
                response.raise_for_status()
                with temp.open('wb') as stream:
                    for chunk in response.iter_content(1024*1024):
                        if time.monotonic()-started > 600:
                            raise TimeoutError('Shard exceeded ten-minute transfer limit')
                        count += len(chunk)
                        if count > size: raise ValueError('Unexpected file size')
                        stream.write(chunk); h.update(chunk)
            if count != size or h.hexdigest() != digest:
                raise ValueError('Downloaded shard identity mismatch')
            temp.replace(path)
            print('Verified', path.name, size, flush=True)
            return
        except (requests.RequestException, TimeoutError, ValueError) as exc:
            # Never log signed URLs or exception bodies containing credentials.
            print('Retry', path.name, attempt+1, type(exc).__name__, flush=True)
            temp.unlink(missing_ok=True)
            if attempt == 4: raise RuntimeError('Shard download failed: '+path.name) from None
            time.sleep(min(2**attempt, 10))


def main():
    # Checked here, not at import, so other runs (e.g. model47) can reuse fetch/output_urls.
    assert DATA_RUN in ('model16','model20','model22','model28','dense128','dense96')
    api = KaggleApi(); api.authenticate()
    ready = json.loads((ROOT/f'cloud/{RUN_NAME}_ready.json').read_text())
    slug_template = {'model16':'rsna-knee-dense-cache-part{}',
                     'model20':'rsna-knee-dense-cache-384-part{}',
                     'model22':'rsna-knee-dense-cache-336-6-94-part{}',
                     'model28':'rsna-knee-dense-cache-384-6-94-part{}-s44',
                     # Kaggle derived these slugs from the kernel titles
                     'dense128':'rsna-knee-dense-cache-256-part{}-s128',
                     'dense96':'rsna-knee-dense-cache-336-part{}-s96'}[DATA_RUN]
    for part, expected in enumerate(ready['manifests']):
        folder = ROOT/f'data/dense_shards/part{part}'
        folder.mkdir(parents=True,exist_ok=True)
        urls = output_urls(api, slug_template.format(part))
        response = requests.get(urls['cache_manifest.json'], timeout=(20,60))
        response.raise_for_status()
        actual = response.json()
        assert actual == expected, 'Cache output changed since readiness verification'
        (folder/'cache_manifest.json').write_text(json.dumps(actual,indent=2))
        def download(shard):
            fetch(urls[shard['file']],folder/shard['file'],shard['bytes'],shard['sha256'])
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(download, actual['shards']))
        print('Downloaded/verified cache part',part,flush=True)
    weights=ROOT/'data/weights';weights.mkdir(exist_ok=True)
    run_config = ROOT/RUN_NAME/'config.json'
    config = json.loads(run_config.read_text()) if run_config.exists() else {}
    if config.get('pretrained_dataset'):
        target = ROOT/config['pretrained_file']
        target.parent.mkdir(parents=True, exist_ok=True)
        already = target.is_file() and hashlib.sha256(target.read_bytes()).hexdigest() == config['pretrained_sha256']
        if already:
            print('Verified pretrained backbone (already present)', target.name, flush=True)
        for attempt in range(0 if already else 5):
            try:
                api.dataset_download_file(config['pretrained_dataset'], target.name, path=str(target.parent),
                                          force=True, quiet=True)
                with target.open('rb') as stream:
                    assert hashlib.file_digest(stream,'sha256').hexdigest()==config['pretrained_sha256']
                print('Verified pretrained backbone', target.name, flush=True)
                break
            except Exception as exc:
                print('Backbone retry',attempt+1,type(exc).__name__,flush=True)
                if attempt==4: raise RuntimeError('Backbone download failed') from None
                time.sleep(5)
    if DATA_RUN in ('dense128', 'dense96'):
        # Only from-ImageNet runs use this cache; the fine-tune checkpoint is not needed, and Kaggle-API dataset
        # downloads crawl on some hosts (model86: ~37 KB/s -> ~2 h for this 300 MB file).
        return
    if DATA_RUN == 'model16':
        dataset, name = 'dreaddevelopment/raptor-knee-maxspan', 'raptor_ft_coatnet_v5_full_swa.pt'
    elif DATA_RUN == 'model20':
        dataset, name = 'dreaddevelopment/raptor-knee-native384dense', 'raptor_ft_coatnet_v10_full.pt'
    elif DATA_RUN == 'model28':
        dataset, name = 'dreaddevelopment/raptor-knee-native384', 'raptor_ft_coatnet_v8_full_swa.pt'
    else:
        dataset, name = 'dreaddevelopment/raptor-knee-finespacing', 'raptor_ft_coatnet_v9_full.pt'
    for attempt in range(5):
        try:
            api.dataset_download_file(dataset,name,path=str(weights),force=True,quiet=True)
            with (weights/name).open('rb') as stream:
                assert hashlib.file_digest(stream,'sha256').hexdigest()==ready['initial_checkpoint_sha256']
            return
        except Exception as exc:
            print('Checkpoint retry',attempt+1,type(exc).__name__,flush=True)
            if attempt==4: raise RuntimeError('Checkpoint download failed') from None
            time.sleep(5)


if __name__=='__main__':main()
