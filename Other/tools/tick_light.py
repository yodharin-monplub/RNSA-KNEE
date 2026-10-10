"""One tick of a LIGHT-gated Speedy + fleet-leg notebook (scripts/light_gate.py) on the main account: verify the
3-study commit run, then submit once and read the score. Works for any artifacts/<model>_notebook built with
run_speedy_fleet_v2.py --dummy light (optionally + a group-readout cell after the leg).

State in Other/artifacts/<model>_notebook/: run_response.txt, output/, verify.txt, submitted.txt, score.txt, failed.txt.
Prints one STATE line: RUNNING / FAILED / VERIFY_FAILED / VERIFIED / SUBMITTED / SUBMISSION_FAILED / SCORED.
Usage: tick_light.py MODEL "SUBMISSION MESSAGE" [--no-submit]"""
import ast
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]   # the project's Other/ folder
os.environ['KAGGLE_API_TOKEN'] = dotenv_values(ROOT / '.env')['KAGGLE_API_KEY_yodharinmonplub@actuarialtutor.org']
os.environ['KAGGLE_CONFIG_DIR'] = str(ROOT / '.kaggle')
from kaggle.api.kaggle_api_extended import KaggleApi

COMP = 'rsna-knee-abnormality-detection'
MODEL, MESSAGE = sys.argv[1], sys.argv[2]
D = ROOT / f'artifacts/{MODEL}_notebook'
REF = json.loads((D / 'kernel-metadata.json').read_text())['id']
NB = json.loads((D / 'notebook.ipynb').read_text(encoding='utf-8'))
SRC = [''.join(c['source']) for c in NB['cells'] if c['cell_type'] == 'code']
# two kinds: a LIGHT-gated Speedy + leg notebook, or a FLEET-ONLY notebook (our models alone, one cell, no Speedy)
FLEET_ONLY = SRC[0].startswith('# ---- fleet ONLY')
# third kind (2026-10-09, scripts/run_base_fleet.py): a public base notebook (e.g. Versia_5) + our leg as the last cell;
# the base's own dummy run is short, so it has no light gate
BASE = SRC[-1].startswith("# ---- this project's fleet leg on a public base")
# fourth kind (2026-10-10): the public base ALONE (run_base_fleet.py --fleet none), a last cell that checks submission.csv
BASE_ONLY = SRC[-1].startswith('# ---- base ONLY')
assert FLEET_ONLY or BASE or BASE_ONLY or (SRC[0].startswith('# ---- light dummy-run gate') and '_RSNA_LIGHT_ALLOWED = True' in SRC[0]), \
    'neither a light build, a fleet-only notebook nor a base + leg notebook'
N_GATED = sum(1 for s in SRC if s.startswith('%%rsna_speedy_cell'))
LEG = '_M80_FLEET = []' if BASE_ONLY else SRC[0] if FLEET_ONLY else SRC[-1] if BASE else \
    next(s for s in SRC if s.startswith('# ---- model80:'))
FLEET = ast.literal_eval(re.search(r'^_M80_FLEET = (.+)$', LEG, re.M).group(1))
N_CKPT = len({f for m in FLEET for f in m[1]})
READOUT = any('GROUP READOUT' in s for s in SRC)

api = KaggleApi(); api.authenticate()


def version():
    p = D / 'run_response.txt'
    m = re.search(r'"versionNumber":\s*(\d+)', p.read_text()) if p.exists() else None
    return int(m.group(1)) if m else None


def fetch(out):
    pattern = r'(\.log$|submission\.csv$|model80_fleet\.npz$)'
    token, pages = None, 0
    while True:
        result = api.kernels_output(REF, path=str(out), file_pattern=pattern, force=True, quiet=True, page_token=token,
                                    page_size=50)
        token = result[1] if isinstance(result, tuple) else None
        pages += 1
        if not token or pages > 40:
            break


if (D / 'score.txt').exists():
    print('STATE SCORED |', (D / 'score.txt').read_text().strip()); raise SystemExit

if (D / 'submitted.txt').exists():
    for s in api.competition_submissions(COMP) or []:
        if (getattr(s, 'description', '') or '') == MESSAGE:
            st, pub = str(getattr(s, 'status', '')), getattr(s, 'public_score', None)
            err = (getattr(s, 'error_description', '') or '').strip()
            if err:   # a hidden-set rerun that errored is COMPLETE with no score
                (D / 'failed.txt').write_text(f'{st} | {err}')
                print('STATE SUBMISSION_FAILED |', st, '|', err[:200])
            elif pub not in (None, ''):
                (D / 'score.txt').write_text(f'public {pub} | status {st}')
                print('STATE SCORED | public', pub, '|', st)
            else:
                print('STATE SUBMITTED | status', st, '| no score yet')
            break
    else:
        print('STATE SUBMITTED | submission not listed yet')
    raise SystemExit

out = D / 'output'
verified = (D / 'verify.txt').exists() and (D / 'verify.txt').read_text().startswith('OK ')
if not verified:
    try:
        st = str(api.kernels_status(REF).status)
    except Exception as exc:
        print('STATE RUNNING | status unavailable', type(exc).__name__); raise SystemExit
    if 'COMPLETE' not in st:
        if 'ERROR' in st or 'CANCEL' in st:
            out.mkdir(exist_ok=True)
            try:
                fetch(out)
            except Exception as e:
                print('log download failed', str(e)[:150])
            print('STATE FAILED |', st)
        else:
            print('STATE RUNNING |', st, '| version', version())
        raise SystemExit
    out.mkdir(exist_ok=True)
    fetch(out)
    text, span = '', None
    for p in out.rglob('*.log'):
        entries = json.loads(p.read_text(encoding='utf-8'))
        text += ''.join(e.get('data', '') for e in entries)
        if entries:
            span = round(entries[-1]['time'] - entries[0]['time'])
    lines = text.splitlines()
    ckpts = text.count('[model80] checkpoint ')
    if FLEET_ONLY:   # no gate, no Speedy: the one cell writes the fleet's mean probability
        gate_ok = True
        skipped = 0
        leg = next((l for l in lines if l.startswith('[fleet-only] submission written')), '')
        leg_ok = ' 3/3 studies read by the fleet, 0 failed' in leg and ckpts == N_CKPT
    elif BASE_ONLY:   # the base as published: only its own submission file and our checking cell
        gate_ok, skipped = True, 0
        leg = next((l for l in lines if l.startswith('[base-only] submission.csv')), '')
        leg_ok = leg.startswith('[base-only] submission.csv (3, 13)')
    elif BASE:   # the base runs in full on the dummy data; check our leg's line (base tracebacks may be handled ones)
        gate_ok, skipped = True, 0
        leg = next((l for l in lines if l.startswith('[model80] fleet leg blended')), '')
        leg_ok = (' 3/3 studies read by the fleet, 0 failed' in leg and 'dropped for time []' in leg and ckpts == N_CKPT
                  and 'fleet leg FAILED' not in text)
    else:
        gate_ok = any(l.startswith('[light] dummy-run gate: light=True') for l in lines)
        skipped = sum(1 for l in lines if l.startswith('[light] Speedy cell ') and l.endswith('skipped (dummy run)'))
        leg = next((l for l in lines if l.startswith('[model80] fleet leg blended')), '')
        leg_ok = (' 3/3 studies read by the fleet, 0 failed' in leg and 'dropped for time []' in leg and ckpts == N_CKPT
                  and 'fleet leg FAILED' not in text)
    readout_ok = (not READOUT) or ('GROUP READOUT: kept' in text and 'GROUP READOUT FAILED' not in text)
    tracebacks = sum(1 for l in lines if 'Traceback' in l)
    sub_ok, sub_info = False, 'missing'
    sub = next(out.rglob('submission.csv'), None)
    if sub is not None:
        d = pd.read_csv(sub)
        v = d.iloc[:, 1:].to_numpy(float)
        sub_ok = d.shape == (3, 13) and np.isfinite(v).all() and v.min() >= 0 and v.max() <= 1
        sub_info = f'{d.shape} finite in [0,1]: {sub_ok}'
    ok = gate_ok and skipped == N_GATED and leg_ok and readout_ok and (tracebacks == 0 or BASE or BASE_ONLY) and sub_ok
    report = (f'{"OK" if ok else "BAD"} gate_light={gate_ok} skipped={skipped}/{N_GATED} leg_ok={leg_ok} '
              f'checkpoints={ckpts}/{N_CKPT} readout_ok={readout_ok} tracebacks={tracebacks} submission={sub_info} '
              f'log_span={span}s\n{leg}\n')
    (D / 'verify.txt').write_text(report)
    if not ok:
        print('STATE VERIFY_FAILED |', report[:900]); raise SystemExit
if '--no-submit' in sys.argv or (D / 'never_submit.txt').exists():   # diagnostics (e.g. model133/134) are never submitted
    print('STATE VERIFIED (not submitted) |', (D / 'verify.txt').read_text().splitlines()[0]); raise SystemExit

result = api.competition_submit_code('submission.csv', MESSAGE, competition=COMP, kernel=REF, kernel_version=version(),
                                     quiet=False)
(D / 'submitted.txt').write_text(f'{time.strftime("%Y-%m-%d %H:%M:%S")} version {version()}\n{result}\n')
print('STATE SUBMITTED | version', version(), '|', (D / 'verify.txt').read_text().splitlines()[0], '|', str(result)[:200])
