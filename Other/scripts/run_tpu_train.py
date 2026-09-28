"""Build and push a Kaggle TPU (v5e-8) training kernel for scripts/train_xla.py, or a free CPU kernel that runs the same
trainer on torch_xla's CPU backend (PJRT_DEVICE=CPU) to check the code before spending TPU time.

The kernel restores this project's layout from the private trainer kit (yodharinmonplub/rsna-knee-trainer-kit, flat files
+ layout.json; rebuild it with scripts/build_trainer_kit.py), copies the competition train.csv and the CoAtNet-2 ImageNet
weights (yodharinmonplub/rsna-knee-backbone-coatnet2), unpacks the dense cache from its kernel outputs and runs
train_xla.py. Plans:
  smoke       pilot cache, 2 short epochs (mechanics only);
  full        the 4-part 336 px cache (--cache s64 = 64 frames, s96 = 96 frames), the configured recipe, stopped by
              --stop-at before the session limit;
  smoke+full  smoke first, then the full run in the same session only if the smoke run completed;
  folds       k-fold runs (config n_folds + select 'last'; --folds 0,1,2): one cache unpack, folds in order, each fold
              only if the previous fold's duration still fits before the session limit; OOF predictions per fold;
  smoke+folds smoke (with the first fold's split) first, then the folds;
  cpu-test    (CPU kernel, no accelerator) installs the torch/torch_xla 2.8 CPU wheels and runs the smoke plan with batch 1
              and 2 windows on PJRT_DEVICE=CPU (1 process) - catches code errors for free.
TPU notebooks cannot be competition submissions; the trained best.pt is used from a GPU notebook (fleet leg) as usual.
Kaggle API pushes do not get a TPU (kaggle-cli #1197): push, then start the run with "Save & Run All" in the web editor.

Usage: python scripts/run_tpu_train.py --name model98 --plan smoke+full [--push]
       python scripts/run_tpu_train.py --name model98 --plan cpu-test --push
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

SESSION_LIMIT_S = 9 * 3600   # Kaggle TPU sessions stop at 9 h; outputs are only kept if the script ends by itself
MARGIN_S = 40 * 60           # time left after training for the last save and the output upload

KERNEL = r'''
import json, os, shutil, subprocess, sys, time
from pathlib import Path
started = time.time()
PLAN = __PLAN__
NAME = __NAME__
CACHE = __CACHE__
FOLDS = __FOLDS__
STOP_AT = started + __TRAIN_SECONDS__
IN = Path('/kaggle/input')
ROOT = Path('/kaggle/working/rsna')


def find(name, max_depth=5):
    """Files called `name` under /kaggle/input without walking the competition's DICOM tree (a full rglob took 11 min)."""
    hits, base = [], len(IN.parts)
    for folder, dirs, files in os.walk(IN):
        depth = len(Path(folder).parts) - base
        if depth >= max_depth or (depth == 1 and Path(folder).name == 'competitions'):
            dirs[:] = []
        if name in files:
            hits.append(Path(folder) / name)
    return sorted(hits)


def sh(cmd):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print('$', cmd, '\n', (r.stdout + r.stderr)[-3000:], flush=True)
    return r.returncode


kit = find('layout.json')[0].parent
for name, rel in json.loads((kit / 'layout.json').read_text()).items():
    (ROOT / rel).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(kit / name, ROOT / rel)
comp = next(Path(p) for p in ('/kaggle/input/competitions/rsna-knee-abnormality-detection',
                              '/kaggle/input/rsna-knee-abnormality-detection') if Path(p, 'train.csv').exists())
(ROOT / 'data').mkdir(exist_ok=True); shutil.copy2(comp / 'train.csv', ROOT / 'data/train.csv')
bb = find('coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k.safetensors')[0]
(ROOT / 'data/weights').mkdir(parents=True, exist_ok=True); shutil.copy2(bb, ROOT / 'data/weights' / bb.name)
print('inputs ready', round(time.time() - started), 's', flush=True)

config_path = ROOT / NAME / 'config.json'
env = dict(os.environ)
if PLAN == 'cpu-test':
    sh('pip install -q torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu')
    sh('pip install -q --no-deps torch_xla==2.8.0')
    config = json.loads(config_path.read_text())
    config.update(batch_size=1, train_windows=2, workers=2)       # CPU memory; same code paths
    config_path = ROOT / NAME / 'config_cpu_test.json'
    config_path.write_text(json.dumps(config, indent=2))
    env['PJRT_DEVICE'] = 'CPU'
else:
    env['PJRT_DEVICE'] = 'TPU'
    print({k: v for k, v in sorted(os.environ.items()) if k.startswith(('TPU', 'XLA', 'PJRT', 'CLOUD_TPU', 'LIBTPU'))},
          flush=True)
sh('python -c "import torch, torch_xla; print(torch.__version__, torch_xla.__version__)"')
if sh('python -c "import timm; print(timm.__version__)"') != 0:
    sh('pip install -q timm==1.0.26 safetensors')
sh('nproc; free -g | head -2; df -h /tmp /dev/shm /kaggle/working 2>/dev/null | cat')

sys.path.insert(0, str(ROOT))
from cloud.prepare_dense_cache import prepare
manifests = find('cache_manifest.json')
print('manifests', [str(m) for m in manifests], flush=True)


def space(path):
    try:
        return shutil.disk_usage(path).free
    except OSError:
        return 0


def unpack(pilot):
    """Unpack the pilot cache or the full cache (geometry CACHE) to local disk; returns its folder."""
    geometry = 'pilot' if pilot else CACHE
    chosen = [m for m in manifests if {'pilot': 'pilot' in str(m), 's96': '-s96' in str(m),
                                       's64': 'pilot' not in str(m) and '-s96' not in str(m)}[geometry]]
    need = {'pilot': 8e9, 's64': 40e9, 's96': 55e9}[geometry]   # 4,407 studies unpack to ~32 GB (64 frames) / ~48 GB (96)
    base = next((d for d in ('/kaggle/tmp', '/tmp', '/dev/shm') if space(d) > need), '/tmp')
    dest = Path(base) / ('dense_pilot' if pilot else 'dense')
    t0 = time.time()
    print('cache', [str(m) for m in chosen], '->', dest, f'free {space(base) / 1e9:.0f} GB', flush=True)
    print(prepare(chosen, ROOT / 'data/train.csv', dest, pilot=pilot, size=336, slots=96 if geometry == 's96' else 64),
          flush=True)
    print('cache ready', round(time.time() - t0), 's', flush=True)
    return dest


def run_trainer(dest, out, stop_at, smoke=False, fold=None):
    """One train_xla.py run; True when it completed (exit code 0 and TRAINING_COMPLETE written)."""
    cmd = [sys.executable, str(ROOT / 'scripts/train_xla.py'), '--cache', str(dest), '--config', str(config_path),
           '--output', str(out), '--stop-at', str(stop_at)] + (['--smoke'] if smoke else [])
    cmd += ['--fold', str(fold)] if fold is not None else []
    t0 = time.time()
    rc = subprocess.run(cmd, env=env).returncode
    print('train_xla', 'smoke' if smoke else 'full', '' if fold is None else f'fold {fold}', 'exit code', rc, 'train',
          round(time.time() - t0), 's, session', round(time.time() - started), 's', flush=True)
    return rc == 0 and (out / 'TRAINING_COMPLETE').exists()


ok = True
if PLAN in ('smoke', 'smoke+full', 'cpu-test', 'smoke+folds'):
    dest = unpack(True)
    smoke = Path('/kaggle/working') / (NAME + '_smoke')
    ok = run_trainer(dest, smoke, min(STOP_AT, time.time() + 3 * 3600), smoke=True, fold=FOLDS[0] if FOLDS else None)
    shutil.rmtree(dest, ignore_errors=True)   # never ship cache pixels as notebook output
    if ok:
        print('smoke history', (smoke / 'history.json').read_text(), flush=True)
        (smoke / 'best.pt').unlink(missing_ok=True)   # mechanics only: keep the logs, not 300 MB of weights
if ok and PLAN in ('full', 'smoke+full'):
    dest = unpack(False)
    ok = run_trainer(dest, Path('/kaggle/working') / NAME, STOP_AT)
    shutil.rmtree(dest, ignore_errors=True)
if ok and PLAN in ('folds', 'smoke+folds'):
    # k-fold: one cache unpack, folds in order; a fold starts only if the previous fold's duration still fits.
    dest = unpack(False)
    last = None
    for k in FOLDS:
        if last is not None and STOP_AT - time.time() < last * 1.1:
            print(f'fold {k} not started: {STOP_AT - time.time():.0f}s left, last fold took {last:.0f}s', flush=True)
            break
        t0 = time.time()
        fold_ok = run_trainer(dest, Path('/kaggle/working') / NAME / f'fold{k}', STOP_AT, fold=k)
        last = time.time() - t0
        ok = ok and fold_ok
        if not fold_ok:
            break
    shutil.rmtree(dest, ignore_errors=True)
shutil.rmtree(ROOT, ignore_errors=True)           # the kit copy and backbone weights are inputs, not outputs
print('PLAN', PLAN, 'OK' if ok else 'FAILED', 'session', round(time.time() - started), 's', flush=True)
sys.exit(0 if ok else 1)
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='model98')
    parser.add_argument('--plan', choices=['smoke', 'full', 'smoke+full', 'cpu-test', 'folds', 'smoke+folds'],
                        default='smoke+full')
    parser.add_argument('--cache', choices=['s64', 's96'], default='s64')
    parser.add_argument('--folds', default='', help='k-fold plans: fold indices to run in this session, e.g. 0,1,2')
    parser.add_argument('--acc', default='TpuV5E8')
    parser.add_argument('--push', action='store_true')
    args = parser.parse_args()
    cpu = args.plan == 'cpu-test'
    folds = [int(k) for k in args.folds.split(',') if k.strip()]
    assert bool(folds) == args.plan.endswith('folds'), 'folds plans need --folds (and only they take it)'
    suffix = {'smoke': f'{args.name}-smoke', 'cpu-test': f'{args.name}-cpu-test'}.get(args.plan, args.name)
    if folds:
        suffix += '-f' + ''.join(str(k) for k in folds)   # one kernel per fold set, e.g. model101-f012
    folder = ROOT / f'artifacts/tpu_{suffix.replace("-", "_")}_notebook'
    folder.mkdir(parents=True, exist_ok=True)
    code = (KERNEL.replace('__PLAN__', repr(args.plan)).replace('__NAME__', repr(args.name))
            .replace('__CACHE__', repr(args.cache)).replace('__FOLDS__', repr(folds))
            .replace('__TRAIN_SECONDS__', str(SESSION_LIMIT_S - MARGIN_S)))
    compile(code, 'tpu_kernel', 'exec')
    (folder / 'train_tpu.py').write_text(code, encoding='utf-8')
    cache = []
    if args.plan in ('smoke', 'smoke+full', 'cpu-test', 'smoke+folds'):
        cache.append('yodharinmonplub/rsna-knee-dense-cache-pilot')
    if args.plan in ('full', 'smoke+full', 'folds', 'smoke+folds'):
        slug = 'rsna-knee-dense-cache-336-part{}-s96' if args.cache == 's96' else 'rsna-knee-dense-cache-part{}'
        cache += [f'yodharinmonplub/{slug.format(i)}' for i in range(4)]
    meta = {'id': f'yodharinmonplub/rsna-knee-tpu-train-{suffix}', 'title': f'RSNA Knee TPU Train {suffix.replace("-", " ")}',
            'code_file': 'train_tpu.py', 'language': 'python', 'kernel_type': 'script', 'is_private': True,
            'enable_gpu': False, 'enable_tpu': not cpu, 'enable_internet': True,
            'dataset_sources': ['yodharinmonplub/rsna-knee-trainer-kit', 'yodharinmonplub/rsna-knee-backbone-coatnet2'],
            'competition_sources': ['rsna-knee-abnormality-detection'], 'kernel_sources': cache, 'model_sources': []}
    assert len(meta['title']) <= 50, meta['title']
    (folder / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2))
    print('built', folder, '| kernel', meta['id'], '| plan', args.plan, '| cache', args.cache)
    if args.push:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi(); api.authenticate()
        result = api.kernels_push(str(folder.resolve()), acc=None if cpu else args.acc)
        (folder / 'run_response.txt').write_text(str(result))
        print(result)


if __name__ == '__main__':
    main()
