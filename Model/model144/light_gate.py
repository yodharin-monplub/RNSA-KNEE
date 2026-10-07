"""Light dummy run for the Speedy + fleet-leg submission notebooks (2026-10-04).

Kaggle's commit run of a code-competition notebook sees only the public 3-study dummy test, and none of this project's
submissions change anything above the final fleet-leg cell. The light gate lets that commit run skip the unchanged
Speedy cells and feed the leg a stand-in submission.csv (0.5 everywhere, like sample_submission.csv), so a dummy run
costs ~2 instead of ~8 GPU minutes on the submit-only account. The leg itself runs in full.

The hidden rerun executes every cell: each Speedy cell is handed to IPython's own run_cell (the shell's compiler, so
__future__ flags carry over exactly as in a plain notebook) and any error is re-raised, so the notebook still fails
exactly where it would have failed. Light mode needs ALL of: light allowed at build time, no
KAGGLE_IS_COMPETITION_RERUN flag, a competition root found, and <= 10 test studies.

apply_light_gate(nb, allow_light) prefixes every code cell BEFORE the fleet-leg cell (the Speedy cells) with
%%rsna_speedy_cell <i> and inserts the gate cell before the first code cell; the leg and any later cell (e.g. a group
readout) stay as they are. allow_light=False ('gated-full') runs the gated notebook in
full on the dummy data too; used once to check that the gate leaves Speedy's output unchanged.
"""

LIGHT_GATE = '''
# ---- light dummy-run gate (this project, 2026-10-04; see Other/scripts/light_gate.py) ----
# On Kaggle's commit run (public 3-study dummy test) the Speedy cells below are skipped and a stand-in
# submission.csv (0.5 everywhere) feeds the final fleet-leg cell, which runs in full. The hidden rerun executes every
# cell unchanged (each Speedy cell through IPython's own run_cell; any error is re-raised).
_RSNA_LIGHT_ALLOWED = True


def _rsna_light_decide():
    import os
    from pathlib import Path
    import pandas as pd
    root = None
    for value in [os.environ.get('RSNA_COMP_ROOT', ''), '/kaggle/input/competitions/rsna-knee-abnormality-detection',
                  '/kaggle/input/rsna-knee-abnormality-detection']:
        if value and (Path(value) / 'test.csv').is_file():
            root = value
            break
    rerun = bool(os.environ.get('KAGGLE_IS_COMPETITION_RERUN'))
    n_test = None
    if root is not None:
        n_test = int(pd.read_csv(Path(root) / 'test.csv', dtype=str)['StudyInstanceUID'].nunique())
    light = bool(_RSNA_LIGHT_ALLOWED and not rerun and n_test is not None and n_test <= 10)
    return light, root, rerun, n_test


_RSNA_LIGHT, _RSNA_LIGHT_ROOT, _rsna_rerun_flag, _rsna_n_test = _rsna_light_decide()
print(f'[light] dummy-run gate: light={_RSNA_LIGHT} (allowed={_RSNA_LIGHT_ALLOWED}, rerun flag={_rsna_rerun_flag}, '
      f'test studies={_rsna_n_test}, root={_RSNA_LIGHT_ROOT})', flush=True)

from IPython.core.magic import register_cell_magic as _rsna_register_cell_magic


@_rsna_register_cell_magic
def rsna_speedy_cell(line, cell):
    if _RSNA_LIGHT:
        print(f'[light] Speedy cell {line.strip()} skipped (dummy run)', flush=True)
        return None
    result = get_ipython().run_cell(cell)
    if not result.success:
        error = result.error_in_exec if result.error_in_exec is not None else result.error_before_exec
        if error is None:
            error = RuntimeError(f'Speedy cell {line.strip()} failed')
        raise error
    return None


if _RSNA_LIGHT:
    def _rsna_light_standin():
        from pathlib import Path
        import pandas as pd
        root = Path(_RSNA_LIGHT_ROOT)
        test = pd.read_csv(root / 'test.csv', dtype={'StudyInstanceUID': str})
        labels = [c for c in pd.read_csv(root / 'sample_submission.csv', nrows=1).columns if c != 'StudyInstanceUID']
        sub = pd.DataFrame({'StudyInstanceUID': test['StudyInstanceUID'].drop_duplicates().tolist()})
        for label in labels:
            sub[label] = 0.5
        sub.to_csv('/kaggle/working/submission.csv', index=False)
        print(f'[light] stand-in Speedy submission.csv: {len(sub)} studies x {len(labels)} labels at 0.5', flush=True)

    _rsna_light_standin()
    _KE_NS = {'find_test_root': lambda: _RSNA_LIGHT_ROOT}   # the one Speedy helper the fleet leg uses
'''

GATE_PREFIX = '%%rsna_speedy_cell'


def apply_light_gate(nb, allow_light=True):
    """Gate every code cell before the fleet-leg cell; insert the gate cell before the first code cell."""
    cells = nb['cells']
    code = [i for i, c in enumerate(cells) if c['cell_type'] == 'code']
    legs = [i for i in code if ''.join(cells[i]['source']).startswith('# ---- model80:')]
    assert len(legs) == 1, f'expected exactly one fleet-leg cell, found {legs}'
    assert 'light dummy-run gate' not in ''.join(''.join(c['source']) for c in cells), 'already gated'
    for i in [i for i in code if i < legs[0]]:
        src = ''.join(cells[i]['source'])
        assert not src.lstrip().startswith('%%'), f'cell {i} already starts with a cell magic'
        cells[i]['source'] = (f'{GATE_PREFIX} {i}\n' + src).splitlines(keepends=True)
    gate = LIGHT_GATE.lstrip('\n')
    assert gate.count('_RSNA_LIGHT_ALLOWED = True\n') == 1
    if not allow_light:
        gate = gate.replace('_RSNA_LIGHT_ALLOWED = True\n', '_RSNA_LIGHT_ALLOWED = False\n')
    compile(gate, 'light_gate', 'exec')
    cells.insert(code[0], {'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [],
                           'source': gate.splitlines(keepends=True)})
    return nb


def gated_body(src):
    """The original source of a gated cell (for compile checks)."""
    first, _, rest = src.partition('\n')
    return rest if first.startswith(GATE_PREFIX) else src
