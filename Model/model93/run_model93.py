"""model93: model76 (Speedy + our model61 Raptor weights, public 0.945) plus model75 - this project's CoAtNet-2 trained
from ImageNet at 384 px, public 0.939 ALONE (model89) - as a FIFTH Raptor-side view, with model77 v3's tested code
(exact Speedy neutral fill, both GPUs, time allowance, fail-safe to Speedy's four-view result).

Why: model75 is as strong as a Raptor arm on the leaderboard (0.939 alone) and uses exactly the maxspan arm's input
(336 px crops, windows at 384 px), so it belongs in the Raptor probability mean more naturally than in a final-level
blend (model80's 3-member fleet at w 0.30 cost 0.001). Pre-declared share = model77's: w 0.23 next to Speedy's four
view weights (sum 1.0) = 18.7% of the Raptor side. Gold (notebook-way probability mean on our base) +0.0017 at that
share - noise level, and gold already understated model75 by a wide margin (0.9204 gold vs 0.939 LB).
Runtime: ~1,300 hidden studies x 94 windows of CoAtNet-2 at 384 px on two T4s ~ 10-15 min; allowance 1500 s, and
never past T0 + TIME_BUDGET - 30 min. Weights: model75_best.pt from yodharinmonplub/rsna-knee-fleet-weights.

Usage: python scripts/run_model93.py [--push] [--allow-seconds N]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_model76 as base
import run_model77 as m77

NAME = 'model93'
TITLE = 'RSNA Knee Model93 Speedy Our Heads Plus CoAt75'   # <= 50 characters
ALLOW_SECONDS = 1500
DATASET = 'yodharinmonplub/rsna-knee-fleet-weights'


def view_text(allow_seconds):
    v = m77.EXTRA_VIEW.replace('__M77_ALLOW_SECONDS__', str(int(allow_seconds)))
    reps = [
        ("# ---- model77: model73 (this project's fully trained CoAtNet-2, 256 px) as a fifth Raptor-side view ----",
         "# ---- model93: model75 (this project's fully trained CoAtNet-2, 384 px; 0.939 public alone) as a fifth Raptor-side view ----\n"
         "# (model77 v3's code; only the checkpoint, its dataset and the 384 px geometry differ)"),
        ("    hits = sorted({p for pattern in ('/kaggle/input/datasets/*/rsna-knee-model73-weights/model73_best.pt',\n"
         "                                     '/kaggle/input/rsna-knee-model73-weights/model73_best.pt',\n"
         "                                     '/kaggle/input/*/rsna-knee-model73-weights/model73_best.pt')\n"
         "                   for p in _m77_glob.glob(pattern)})\n"
         "    if len(hits) != 1:\n"
         "        raise RuntimeError(f'model77: expected one model73_best.pt, found {hits}')",
         "    hits = sorted({_ke_os.path.realpath(p) for pattern in ('/kaggle/input/datasets/*/rsna-knee-fleet-weights/model75_best.pt',\n"
         "                                     '/kaggle/input/rsna-knee-fleet-weights/model75_best.pt',\n"
         "                                     '/kaggle/input/*/rsna-knee-fleet-weights/model75_best.pt',\n"
         "                                     '/kaggle/input/*/*/rsna-knee-fleet-weights/model75_best.pt')\n"
         "                   for p in _m77_glob.glob(pattern)})\n"
         "    if len(hits) != 1:\n"
         "        raise RuntimeError(f'model93: expected one model75_best.pt, found {hits}')"),
        ("    if arch != 'coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k' or int(ck.get('res', 0)) != 256:\n"
         "        raise RuntimeError(f'model77: unexpected checkpoint arch={arch} res={ck.get(\"res\")}')",
         "    if arch != 'coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k' or int(ck.get('res', 0)) != 384:\n"
         "        raise RuntimeError(f'model93: unexpected checkpoint arch={arch} res={ck.get(\"res\")}')"),
        ("        backbone = timm.create_model(arch, pretrained=False, num_classes=0, in_chans=3, global_pool='avg', img_size=256)",
         "        backbone = timm.create_model(arch, pretrained=False, num_classes=0, in_chans=3, global_pool='avg')   # native 384"),
        ("    rsna_event('raptor_checkpoint', file=str(hits[0]), sha256=rsna_sha(hits[0]), arch=arch, res=256)",
         "    rsna_event('raptor_checkpoint', file=str(hits[0]), sha256=rsna_sha(hits[0]), arch=arch, res=384)"),
        ("    # Same preparation as the maxspan arm (336 px crops, 64-slot proportions, 2-98% span), evaluated at 256 px.\n"
         "    arm = {\"name\": \"coatnet2-fulltrain-73\", \"file\": \"model73_best.pt\", \"arch\": arch, \"res\": 256, \"img\": 336,",
         "    # Exactly the maxspan arm's preparation: 336 px crops, 64-slot proportions, 2-98% span, windows at 384 px.\n"
         "    arm = {\"name\": \"coatnet2-fulltrain-75\", \"file\": \"model75_best.pt\", \"arch\": arch, \"res\": 384, \"img\": 336,"),
        ("    p73 = _ke_np.full(", "    p73 = _ke_np.full("),   # variable names kept (p73 = the fifth view's probabilities)
        ("    print(f\"[model77] fifth view model73 added:", "    print(f\"[model77] fifth view model75 added:"),
        ("        print(f\"[model77] no model73 value for any of", "        print(f\"[model77] no model75 value for any of"),
        ("f\"{int(missing.sum())}/{len(uids)} studies without a model73 value", "f\"{int(missing.sum())}/{len(uids)} studies without a model75 value"),
    ]
    for old, new in reps:
        assert v.count(old) == 1, old[:90]
        v = v.replace(old, new)
    assert 'model73_best' not in v and "img_size=256" not in v
    return v


def build(allow_seconds=ALLOW_SECONDS):
    folder76, meta76 = base.build()
    nb = json.loads((folder76 / 'notebook.ipynb').read_text(encoding='utf-8'))
    hits = [c for c in nb['cells'] if c['cell_type'] == 'code' and m77.PHASE in ''.join(c['source'])]
    assert len(hits) == 1
    text = ''.join(hits[0]['source'])
    assert text.count(m77.PHASE) == 1
    hits[0]['source'] = text.replace(m77.PHASE, m77.PHASE + view_text(allow_seconds)).splitlines(keepends=True)
    meta = dict(meta76)
    meta.update({'id': f'yodharinmonplub/{TITLE.lower().replace(" ", "-")}', 'title': TITLE})
    meta['dataset_sources'] = list(meta76['dataset_sources']) + [DATASET]
    folder = base.ROOT / f'artifacts/{NAME}_notebook'
    model_dir = base.ROOT.parent / 'Model' / NAME
    for target in (folder, model_dir):
        target.mkdir(parents=True, exist_ok=True)
        (target / 'notebook.ipynb').write_text(json.dumps(nb, indent=1), encoding='utf-8')
        (target / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2))
    import shutil
    for script in ('run_model76.py', 'run_model77.py', 'run_model93.py'):
        shutil.copy2(Path(__file__).resolve().parent / script, model_dir / script)
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            src = ''.join(cell['source'])
            if not any(line.lstrip().startswith(('!', '%')) for line in src.splitlines()):
                compile(src, NAME, 'exec')
    return folder, meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--push', action='store_true')
    parser.add_argument('--allow-seconds', type=int, default=ALLOW_SECONDS)
    args = parser.parse_args()
    folder, meta = build(args.allow_seconds)
    print('built', folder, '| kernel', meta['id'], '| allowance', args.allow_seconds)
    if args.push:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi(); api.authenticate()
        result = api.kernels_push(str(folder), acc=meta.get('machine_shape', 'NvidiaTeslaT4'))
        (folder / 'run_response.txt').write_text(str(result))
        print(result)


if __name__ == '__main__':
    main()
