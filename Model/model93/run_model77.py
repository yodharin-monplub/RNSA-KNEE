"""model77: model76 (public Speedy Raptors + our model61 Raptor weights) plus ONE extra view - model73, this
project's fully trained CoAtNet-2 at 256 px - blended into the Raptor side.

Evidence: on the 58 gold cases, blended the notebook's way (probability mean of the Raptor views), model73 at the
pre-declared share 0.30/1.60 = 18.75% of the Raptor side gave +0.0018 (0.9326 -> 0.9344, P>0 0.78) - just under
this project's +0.002 bar. It is submitted anyway as a leaderboard probe because the 58 cases have repeatedly failed
to predict the leaderboard (competitors report gold 0.925 models scoring 0.947 public) and model73 is the only new
model family that adds anything to the blend. The share is fixed in advance: 18.75% of the Raptor side, i.e.
w = 0.23 next to Speedy's four view weights (0.60 / 0.10 / 0.10 / 0.20, sum 1.0).

How: Speedy's Raptor stage writes the four views' raw probabilities to /kaggle/working/raptor_raw.npz and their
ranked probability mean to _raptor.csv, which is the only way the Raptor result reaches the rest of the pipeline.
Right after that stage completes, the extension loads model73 (backbone rebuilt at 256 px - its relative-position
bias is baked for 384 px), prepares windows with Speedy's own capacity-aware sampler (same as every other view),
recomputes the weighted probability mean with model73 as a fifth view, re-ranks, and overwrites _raptor.csv.
A study model73 cannot read keeps its four-view value. Weights: private dataset yodharinmonplub/rsna-knee-model73-weights.

Usage: python scripts/run_model77.py [--push]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_model76 as base

NAME = 'model77'
TITLE = 'RSNA Knee Model77 Speedy Our Heads Plus CoAt73'   # Kaggle caps titles at 50 characters
PHASE = "rsna_phase('public_raptor', 'COMPLETE')\n"
# Wall-clock seconds the fifth view may use on the hidden set (see readme: set from model76's hidden runtime).
ALLOW_SECONDS = 2700

EXTRA_VIEW = '''
# ---- model77: model73 (this project's fully trained CoAtNet-2, 256 px) as a fifth Raptor-side view ----
# Time-boxed and fail-safe: the view runs on both GPUs, stops taking new studies after _M77_ALLOW_SECONDS (a study
# without a model73 value keeps its four-view probability), and any error keeps Speedy's four-view _raptor.csv.
_M77_ALLOW_SECONDS = __M77_ALLOW_SECONDS__


def _m77_extra_view():
    import glob as _m77_glob
    from concurrent.futures import ThreadPoolExecutor as _M77Pool
    torch, timm = _KE_NS['torch'], _KE_NS['timm']
    started = time.time()
    stop_at = min(started + _M77_ALLOW_SECONDS, T0 + TIME_BUDGET - 1800)
    raw = _ke_np.load('/kaggle/working/raptor_raw.npz', allow_pickle=False)
    probs = raw['raw_probabilities'].astype(_ke_np.float32)
    # raptor_raw.npz is written BEFORE Speedy neutral-fills the studies a view could not read, so those rows are NaN
    # here. Apply exactly Speedy's fill (the view's mean over readable studies, else 0.5). Without it the hidden-set
    # rerun failed on model77 v2 (submission 56587767): NaN reached the rank check and raised.
    nan_filled = 0
    for view in probs:
        bad = ~_ke_np.isfinite(view).all(axis=1)
        if bad.any():
            fill = (_ke_np.nanmean(_ke_np.where(_ke_np.isfinite(view), view, _ke_np.nan), axis=0)
                    if (~bad).any() else _ke_np.full(view.shape[1], .5, view.dtype))
            view[bad] = _ke_np.where(_ke_np.isfinite(fill), fill, .5)
            nan_filled += int(bad.sum())
    probs = _ke_np.clip(probs, 0, 1)
    uids = [str(u) for u in raw['study_uids']]
    root = _KE_NS['find_test_root']()
    series_root = root + '/test_series'
    if not _ke_os.path.isdir(series_root):
        series_root = root + '/test_images'
    test_series = _ke_pd.read_csv(root + '/test_series.csv')
    test_series['StudyInstanceUID'] = test_series['StudyInstanceUID'].astype(str)
    test_series['SeriesInstanceUID'] = test_series['SeriesInstanceUID'].astype(str)
    series = {key: frame.to_dict('records') for key, frame in test_series.groupby('StudyInstanceUID')}
    hits = sorted({p for pattern in ('/kaggle/input/datasets/*/rsna-knee-model73-weights/model73_best.pt',
                                     '/kaggle/input/rsna-knee-model73-weights/model73_best.pt',
                                     '/kaggle/input/*/rsna-knee-model73-weights/model73_best.pt')
                   for p in _m77_glob.glob(pattern)})
    if len(hits) != 1:
        raise RuntimeError(f'model77: expected one model73_best.pt, found {hits}')
    ck = torch.load(hits[0], map_location='cpu', weights_only=False)
    arch = ck.get('arch')
    if arch != 'coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k' or int(ck.get('res', 0)) != 256:
        raise RuntimeError(f'model77: unexpected checkpoint arch={arch} res={ck.get("res")}')
    models = []
    for gpu in range(2):
        backbone = timm.create_model(arch, pretrained=False, num_classes=0, in_chans=3, global_pool='avg', img_size=256)
        model = _KE_NS['RaptorClassifier'](backbone, F_dim=backbone.num_features)
        rsna_strict_load(model, ck['model'], hits[0])
        models.append(model.eval().to(torch.device(f'cuda:{gpu}')))
    rsna_event('raptor_checkpoint', file=str(hits[0]), sha256=rsna_sha(hits[0]), arch=arch, res=256)
    del ck
    # Same preparation as the maxspan arm (336 px crops, 64-slot proportions, 2-98% span), evaluated at 256 px.
    arm = {"name": "coatnet2-fulltrain-73", "file": "model73_best.pt", "arch": arch, "res": 256, "img": 336,
           "slots": _KE_NS['_SLOTS64'], "span": (0.02, 0.98), "k_eval": 62, "reverse": False, "w": 0.23}
    p73 = _ke_np.full((len(uids), probs.shape[2]), _ke_np.nan, _ke_np.float32)
    late = [0, 0]

    def run_gpu(gpu):
        device = torch.device(f'cuda:{gpu}')
        index = list(range(gpu, len(uids), 2))
        windows = _ke_prefetched_windows(arm, [uids[i] for i in index], series, series_root, _KE_NS['_make_reader']())
        try:
            with torch.cuda.device(device):
                for local, uid, future in windows:
                    if time.time() > stop_at:
                        late[gpu] = len(index) - local
                        break
                    try:
                        p73[index[local]] = _ke_infer_input(models[gpu], future.result(), device)
                    except Exception as error:
                        rsna_event('model77_study_failed', study=str(uid),
                                   error=f'{type(error).__name__}: {str(error)[:300]}')
        finally:
            windows.close()

    with _M77Pool(max_workers=2) as pool:
        for job in [pool.submit(run_gpu, gpu) for gpu in range(2)]:
            job.result()
    del models
    with _ke_input_lock:   # keep Speedy's preparation audit about its own four views
        _ke_input_ids.difference_update({(arm['name'], u) for u in uids})
    missing = ~_ke_np.isfinite(p73).all(axis=1)
    rsna_event('model77_view', studies=len(uids), without_value=int(missing.sum()), past_allowance=int(sum(late)),
               seconds=round(time.time() - started, 1), four_view_rows_neutral_filled=nan_filled)
    if missing.all():
        print(f"[model77] no model73 value for any of {len(uids)} studies ({sum(late)} past the time allowance); "
              f"_raptor.csv unchanged", flush=True)
        return
    w4 = _ke_np.asarray([float(a['w']) for a in _KE_NS['ARMS']], _ke_np.float64)
    four_view = _ke_np.tensordot(w4 / w4.sum(), probs, axes=(0, 0))
    p73[missing] = four_view[missing]        # no model73 value -> the study keeps its four-view value
    weights = _ke_np.append(w4, arm['w'])
    blend = _ke_np.tensordot(weights / weights.sum(), _ke_np.concatenate([probs, _ke_np.clip(p73, 0, 1)[None]]),
                             axes=(0, 0))
    rsna_finite(blend, 'model77 five-view Raptor probabilities', probability=True)
    ranks = _KE_NS['rankpct'](blend)
    current = _ke_pd.read_csv('/kaggle/working/_raptor.csv', dtype={'StudyInstanceUID': str})
    if current['StudyInstanceUID'].tolist() != uids:
        raise RuntimeError('model77: _raptor.csv study order differs from raptor_raw.npz')
    rsna_finite(ranks, 'model77 five-view Raptor ranks', probability=True)
    current[_KE_NS['LAB']] = ranks.astype(_ke_np.float32)
    current.to_csv('/kaggle/working/._raptor_model77.csv', index=False)
    _ke_os.replace('/kaggle/working/._raptor_model77.csv', '/kaggle/working/_raptor.csv')
    _ke_np.savez_compressed('/kaggle/working/model77_view.npz', study_uids=_ke_np.asarray(uids), model73=p73,
                            missing=missing)
    print(f"[model77] fifth view model73 added: w={arm['w']} = {arm['w'] / weights.sum():.4f} of the Raptor side; "
          f"{int(missing.sum())}/{len(uids)} studies without a model73 value ({sum(late)} past the "
          f"{_M77_ALLOW_SECONDS}s allowance) -> four-view value; {nan_filled} four-view rows neutral-filled as "
          f"Speedy does; {time.time() - started:.0f}s on 2 GPUs; rewrote _raptor.csv", flush=True)


try:
    _m77_extra_view()
except Exception as _m77_error:
    rsna_event('model77_fifth_view_failed', error=f'{type(_m77_error).__name__}: {str(_m77_error)[:500]}')
    print(f'[model77] fifth view FAILED ({type(_m77_error).__name__}: {str(_m77_error)[:300]}); '
          f'Speedy four-view _raptor.csv kept', flush=True)
finally:
    _ke_gc.collect()
    for _m77_gpu in range(_KE_NS['torch'].cuda.device_count()):
        with _KE_NS['torch'].cuda.device(_m77_gpu):
            _KE_NS['torch'].cuda.empty_cache()
'''


def build(allow_seconds=ALLOW_SECONDS):
    folder76, meta76 = base.build()
    nb = json.loads((folder76 / 'notebook.ipynb').read_text(encoding='utf-8'))
    hits = [c for c in nb['cells'] if c['cell_type'] == 'code' and PHASE in ''.join(c['source'])]
    assert len(hits) == 1, f'phase line found in {len(hits)} cells'
    text = ''.join(hits[0]['source'])
    assert text.count(PHASE) == 1
    view = EXTRA_VIEW.replace('__M77_ALLOW_SECONDS__', str(int(allow_seconds)))
    assert '__M77' not in view.replace('_M77_ALLOW_SECONDS =', '')
    hits[0]['source'] = text.replace(PHASE, PHASE + view).splitlines(keepends=True)
    meta = dict(meta76)
    meta.update({'id': f'yodharinmonplub/{TITLE.lower().replace(" ", "-")}', 'title': TITLE})
    meta['dataset_sources'] = list(meta76['dataset_sources']) + ['yodharinmonplub/rsna-knee-model73-weights']
    for folder in (base.ROOT / NAME, base.ROOT / f'artifacts/{NAME}_notebook'):
        folder.mkdir(parents=True, exist_ok=True)
        (folder / 'notebook.ipynb').write_text(json.dumps(nb, indent=1), encoding='utf-8')
        (folder / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2))
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            src = ''.join(cell['source'])
            if not any(line.lstrip().startswith(('!', '%')) for line in src.splitlines()):
                compile(src, NAME, 'exec')
    return base.ROOT / f'artifacts/{NAME}_notebook', meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--push', action='store_true')
    parser.add_argument('--allow-seconds', type=int, default=ALLOW_SECONDS)
    args = parser.parse_args()
    folder, meta = build(args.allow_seconds)
    print('built', folder, '| kernel', meta['id'], '| datasets', len(meta['dataset_sources']))
    if args.push:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi(); api.authenticate()
        result = api.kernels_push(str(folder), acc=meta.get('machine_shape', 'NvidiaTeslaT4'))
        (folder / 'run_response.txt').write_text(str(result))
        print(result)


if __name__ == '__main__':
    main()
