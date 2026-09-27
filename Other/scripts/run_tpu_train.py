"""Build and push a Kaggle TPU (v5e-8) training kernel for scripts/train_xla.py.

The kernel restores this project's layout from the private trainer kit (yodharinmonplub/rsna-knee-trainer-kit, flat files
+ layout.json), copies the competition train.csv and the CoAtNet-2 ImageNet weights (yodharinmonplub/rsna-knee-backbone-
coatnet2), unpacks the dense cache from its kernel outputs (pilot cache for --smoke, the 4-part 64-frame 336 px cache
otherwise) and runs train_xla.py. TPU notebooks cannot be competition submissions; the trained best.pt is used from a GPU
notebook (fleet leg) as usual.

Usage: python scripts/run_tpu_train.py --name model98 [--smoke] [--acc TpuV5E8] [--push]
"""
import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / '.env')
os.environ.setdefault('KAGGLE_API_TOKEN', os.environ.get('KAGGLE_API_KEY_YODHAIRNMONPLUB', ''))
os.environ['KAGGLE_CONFIG_DIR'] = str(ROOT / '.kaggle')

KERNEL = r'''
import json, os, shutil, subprocess, sys, time
from pathlib import Path
started = time.time()
SMOKE = __SMOKE__
NAME = __NAME__
IN = Path('/kaggle/input')
ROOT = Path('/kaggle/working/rsna')
kit = next(IN.rglob('layout.json')).parent
for name, rel in json.loads((kit / 'layout.json').read_text()).items():
    (ROOT / rel).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(kit / name, ROOT / rel)
comp = next(p.parent for p in IN.rglob('train.csv') if (p.parent / 'test_series.csv').exists())
(ROOT / 'data').mkdir(exist_ok=True); shutil.copy2(comp / 'train.csv', ROOT / 'data/train.csv')
bb = next(IN.rglob('coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k.safetensors'))
(ROOT / 'data/weights').mkdir(parents=True, exist_ok=True); shutil.copy2(bb, ROOT / 'data/weights' / bb.name)

def sh(cmd):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print('$', cmd, '\n', (r.stdout + r.stderr)[-3000:], flush=True)
    return r.returncode
sh('python -c "import torch, torch_xla; print(\'torch\', torch.__version__, \'torch_xla\', torch_xla.__version__)"')
sh('python -c "import torch_xla.runtime as xr; print(\'devices\', xr.global_runtime_device_count())"')
if sh('python -c "import timm; print(\'timm\', timm.__version__)"') != 0:
    sh('pip install -q timm==1.0.26 safetensors')
sh('python -c "import timm; print(\'timm\', timm.__version__)"')
sh('nproc; free -g | head -2; df -h /kaggle/working /kaggle/tmp /tmp 2>/dev/null')

sys.path.insert(0, str(ROOT))
from cloud.prepare_dense_cache import prepare
manifests = sorted(p for p in IN.rglob('cache_manifest.json'))
dest = Path('/kaggle/tmp/dense') if Path('/kaggle/tmp').exists() else Path('/tmp/dense')
print('manifests', [str(m) for m in manifests], '->', dest, flush=True)
print(prepare(manifests, ROOT / 'data/train.csv', dest, pilot=SMOKE), flush=True)
print('cache ready', round(time.time() - started), 's', flush=True)
out = Path('/kaggle/working') / (NAME + ('_smoke' if SMOKE else ''))
cmd = [sys.executable, str(ROOT / 'scripts/train_xla.py'), '--cache', str(dest), '--config', str(ROOT / NAME / 'config.json'),
       '--output', str(out)] + (['--smoke'] if SMOKE else [])
env = dict(os.environ, PJRT_DEVICE='TPU')
rc = subprocess.run(cmd, env=env).returncode
print('train_xla exit code', rc, 'elapsed', round(time.time() - started), 's', flush=True)
shutil.rmtree(dest, ignore_errors=True)   # never ship cache pixels as notebook output
sys.exit(rc)
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='model98')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--acc', default='TpuV5E8')
    parser.add_argument('--push', action='store_true')
    args = parser.parse_args()
    suffix = f'{args.name}-smoke' if args.smoke else args.name
    folder = ROOT / f'artifacts/tpu_{suffix.replace("-", "_")}_notebook'
    folder.mkdir(parents=True, exist_ok=True)
    code = KERNEL.replace('__SMOKE__', repr(bool(args.smoke))).replace('__NAME__', repr(args.name))
    compile(code, 'tpu_kernel', 'exec')
    (folder / 'train_tpu.py').write_text(code, encoding='utf-8')
    cache = (['yodharinmonplub/rsna-knee-dense-cache-pilot'] if args.smoke
             else [f'yodharinmonplub/rsna-knee-dense-cache-part{i}' for i in range(4)])
    meta = {'id': f'yodharinmonplub/rsna-knee-tpu-train-{suffix}', 'title': f'RSNA Knee TPU Train {suffix.replace("-", " ")}',
            'code_file': 'train_tpu.py', 'language': 'python', 'kernel_type': 'script', 'is_private': True,
            'enable_gpu': False, 'enable_tpu': True, 'enable_internet': True,
            'dataset_sources': ['yodharinmonplub/rsna-knee-trainer-kit', 'yodharinmonplub/rsna-knee-backbone-coatnet2'],
            'competition_sources': ['rsna-knee-abnormality-detection'], 'kernel_sources': cache, 'model_sources': []}
    (folder / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2))
    print('built', folder, '| kernel', meta['id'])
    if args.push:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi(); api.authenticate()
        result = api.kernels_push(str(folder.resolve()), acc=args.acc)
        (folder / 'run_response.txt').write_text(str(result))
        print(result)


if __name__ == '__main__':
    main()
