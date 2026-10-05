"""Derive scripts/fleet_leg_v5.py from fleet_leg_v4.py: one more reader geometry, 's128x336' = 128 frames
(32/24/24/24/24 slices per series) at 336 px - the 128-frame training cache of model135
(krittanutsomtuas/rsna-knee-dense-cache-336-part{0..3}-s128). Everything else (one-pass reader, priority dropping,
per-member window stride) is v4's; a fleet without 's128x336' members writes exactly v4's submission.
Usage: python scripts/make_fleet_leg_v5.py"""
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
t = (SCRIPTS / 'fleet_leg_v4.py').read_text(encoding='utf-8')


def sub(old, new, count=1):
    global t
    assert t.count(old) == count, (t.count(old), old[:90])
    t = t.replace(old, new)


sub('"""Fleet leg v4 = v3 + an optional per-member window stride',
    '"""Fleet leg v5 = v4 + reader geometry \'s128x336\' (128 frames, 32/24/24/24/24 at 336 px: model135\'s cache).\n'
    'Generated from fleet_leg_v4.py by scripts/make_fleet_leg_v5.py.\n'
    'Fleet leg v4 = v3 + an optional per-member window stride')
sub("FLEET_LEG_V4 = '''\n", "FLEET_LEG_V5 = '''\n")
sub("""                  's96': ([('Sagittal', 1, 24), ('Sagittal', 0, 18), ('Coronal', 1, 18), ('Coronal', 0, 18),
                           ('Axial', -1, 18)], 336)}
""", """                  's96': ([('Sagittal', 1, 24), ('Sagittal', 0, 18), ('Coronal', 1, 18), ('Coronal', 0, 18),
                           ('Axial', -1, 18)], 336),
                  's128x336': ([('Sagittal', 1, 32), ('Sagittal', 0, 24), ('Coronal', 1, 24), ('Coronal', 0, 24),
                                ('Axial', -1, 24)], 336)}
""")
compile(t, 'fleet_leg_v5', 'exec')
(SCRIPTS / 'fleet_leg_v5.py').write_text(t, encoding='utf-8')
print('wrote fleet_leg_v5.py', len(t))
