"""Execute toy notebooks with a real IPython kernel (nbclient) to check scripts/light_gate.py:
plain vs gated-full give the same cell outputs (incl. __future__ annotations carried across cells and last-expression
display), errors still stop the notebook, light mode skips the 'Speedy' cells and writes the stand-in, and the rerun
flag / a large test set force full mode.
Needs ipykernel + nbclient + pandas (not in requirements.txt; a separate venv works): python tests/test_light_gate.py"""
import copy
import json
import os
import sys
import tempfile
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import light_gate

TMP = Path(tempfile.mkdtemp(prefix='lg_'))
WORK = TMP / 'working'
WORK.mkdir()
light_gate.LIGHT_GATE = light_gate.LIGHT_GATE.replace('/kaggle/working/', WORK.as_posix() + '/')


def make_root(n):
    root = TMP / f'root{n}'
    root.mkdir(exist_ok=True)
    uids = [f'1.2.{k}' for k in range(n)]
    (root / 'test.csv').write_text('StudyInstanceUID\n' + '\n'.join(uids) + '\n')
    (root / 'sample_submission.csv').write_text('StudyInstanceUID,ACL,MCL\n' + ''.join(f'{u},0.5,0.5\n' for u in uids))
    return root


CELLS = [
    ('markdown', '# toy Speedy'),
    ('code', 'from __future__ import annotations\nimport os\nX = 41\n\ndef f(a: NotDefinedAnywhere) -> NotDefinedAnywhere:\n'
             '    return a\nprint("cell1", X, f.__annotations__)'),
    ('code', 'def g(b: AlsoUndefined):\n    return b\nY = X + 1\n_KE_NS = {"find_test_root": lambda: "speedy-root"}\n'
             'print("cell2", Y, g.__annotations__)\nY * 10'),
    ('code', 'print("cell3", Y)\n__ERROR__'),
    ('code', '# ---- model80: toy leg\nimport os\nprint("LEG", _KE_NS["find_test_root"](), Y if "Y" in globals() else None)\n'
             'p = os.path.join(r"' + str(WORK) + '", "submission.csv")\nprint("LEG sees submission", os.path.exists(p))'),
]


def notebook(error=False):
    nb = nbformat.v4.new_notebook()
    for kind, src in CELLS:
        src = src.replace('__ERROR__', 'raise ValueError("boom in cell3")' if error else 'pass')
        nb.cells.append(nbformat.v4.new_markdown_cell(src) if kind == 'markdown' else nbformat.v4.new_code_cell(src))
    return nb


def run(nb, env):
    for k in ('KAGGLE_IS_COMPETITION_RERUN', 'RSNA_COMP_ROOT'):
        os.environ.pop(k, None)
    os.environ.update(env)
    for f in WORK.glob('*'):
        f.unlink()
    nb = copy.deepcopy(nb)
    client = NotebookClient(nb, timeout=120, kernel_name='python3', resources={'metadata': {'path': str(TMP)}})
    error = None
    try:
        client.execute()
    except CellExecutionError as exc:
        error = str(exc).splitlines()[-1][:120] if str(exc) else 'CellExecutionError'
    outs = []
    for cell in nb.cells:
        if cell.cell_type != 'code':
            continue
        text = ''
        for o in cell.get('outputs', []):
            if o.get('output_type') == 'stream':
                text += o.get('text', '')
            elif o.get('output_type') == 'execute_result':
                text += 'OUT:' + o['data'].get('text/plain', '') + '\n'
            elif o.get('output_type') == 'error':
                text += f"ERR:{o.get('ename')}:{o.get('evalue')}\n"
        outs.append(text)
    return outs, error, sorted(p.name for p in WORK.glob('*'))


def gated(nb, allow_light):
    d = json.loads(nbformat.writes(nb))
    light_gate.apply_light_gate(d, allow_light=allow_light)
    return nbformat.reads(json.dumps(d), as_version=4)


results = {}
root3, root20 = make_root(3), make_root(20)
plain, plain_out_err = notebook(), None
base_outs, base_err, _ = run(plain, {'RSNA_COMP_ROOT': str(root3)})
print('PLAIN', base_err, base_outs)
assert base_err is None and "'a': 'NotDefinedAnywhere'" in base_outs[0] and "'b': 'AlsoUndefined'" in base_outs[1]
assert 'OUT:420' in base_outs[1] and 'LEG speedy-root 42' in base_outs[3]

# gated-full on the dummy-sized test: the 'Speedy' cells' outputs equal the plain run's
outs, err, files = run(gated(plain, False), {'RSNA_COMP_ROOT': str(root3)})
print('GATED-FULL', err, outs)
assert err is None and 'light=False (allowed=False' in outs[0]
assert outs[1:] == base_outs, (outs[1:], base_outs)
assert files == []

# light build on the dummy-sized test: Speedy skipped, stand-in written, leg runs with the stand-in root
outs, err, files = run(gated(plain, True), {'RSNA_COMP_ROOT': str(root3)})
print('LIGHT', err, outs, files)
assert err is None and 'light=True' in outs[0] and 'stand-in Speedy submission.csv: 3 studies x 2 labels' in outs[0]
assert all('skipped (dummy run)' in o for o in outs[1:4])
assert ('LEG ' + str(root3)) in outs[4] and 'LEG sees submission True' in outs[4] and files == ['submission.csv']
sub = (WORK / 'submission.csv').read_text() if (WORK / 'submission.csv').exists() else ''

# light build but the rerun flag set -> full
outs, err, files = run(gated(plain, True), {'RSNA_COMP_ROOT': str(root3), 'KAGGLE_IS_COMPETITION_RERUN': 'Batch'})
print('LIGHT+RERUN', err, outs)
assert err is None and 'light=False (allowed=True, rerun flag=True' in outs[0] and outs[1:] == base_outs

# light build with a 20-study test -> full
outs, err, files = run(gated(plain, True), {'RSNA_COMP_ROOT': str(root20)})
print('LIGHT+20', err, outs)
assert err is None and 'light=False' in outs[0] and 'test studies=20' in outs[0] and outs[1:] == base_outs

# errors: plain and gated-full both stop at cell 3; the leg never runs
bad = notebook(error=True)
p_outs, p_err, _ = run(bad, {'RSNA_COMP_ROOT': str(root3)})
g_outs, g_err, _ = run(gated(bad, False), {'RSNA_COMP_ROOT': str(root3)})
l_outs, l_err, _ = run(gated(bad, True), {'RSNA_COMP_ROOT': str(root20)})
print('ERROR plain', p_err, p_outs)
print('ERROR gated-full', g_err, g_outs)
print('ERROR light-build/20 studies', l_err, l_outs)
assert p_err and g_err and l_err
assert 'boom in cell3' in p_err and 'boom in cell3' in g_err and 'boom in cell3' in l_err
assert p_outs[3] == '' and g_outs[4] == '' and l_outs[4] == ''   # leg cell never ran
print('ALL LIGHT-GATE TESTS PASS')
