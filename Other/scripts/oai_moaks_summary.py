"""How many OAI knee-visits carry MOAKS readings, and how do they map to our findings?"""
from pathlib import Path

import pandas as pd

root = Path('data/external/oai/mri_assessments')
frames = []
for path in sorted(root.rglob('kMRI_SQ_MOAKS_BICL[0-9][0-9].txt')):
    frame = pd.read_csv(path, sep='|', low_memory=False, encoding='latin-1')
    frame.columns = [c.upper() for c in frame.columns]
    visit = path.stem[-2:]
    frame = frame.rename(columns={c: c.replace(f'V{visit}', 'V') for c in frame.columns})
    frame['VISIT'] = visit
    frames.append(frame)
    print(path.name, 'rows', len(frame), 'subjects', frame['ID'].nunique())
data = pd.concat(frames, ignore_index=True)
data = data.drop_duplicates(['ID', 'SIDE', 'VISIT', 'READPRJ']) if 'READPRJ' in data else data
knees = data.drop_duplicates(['ID', 'SIDE', 'VISIT'])
print('\nknee-visits with a MOAKS read:', len(knees), '| distinct knees:', len(knees.drop_duplicates(['ID', 'SIDE'])),
      '| subjects:', knees['ID'].nunique())
print(knees.groupby('VISIT').size().to_string())
columns = [c for c in data.columns if c.startswith('VM')]
print('\nMOAKS variables:', len(columns))
for key, label in [('EFF', 'effusion-synovitis'), ('HOF', 'Hoffa synovitis'), ('POP', 'popliteal (Baker) cyst'),
                   ('ACL', 'ACL'), ('MTR', 'meniscus tear type (MTRx)'), ('MCM', 'cartilage size (MCMx)')]:
    found = [c for c in columns if key in c]
    print(f'{label:32s}', found[:8])
    for c in found[:2]:
        print('   ', c, knees[c].astype(str).value_counts().to_dict())
