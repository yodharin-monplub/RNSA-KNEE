"""Public-LB snapshot: our team's rank and score, the medal lines, and the top scores.
Usage: python scripts/lb_rank.py [--account krittanutsomtuas]   (read-only; uses a training key, not the submit key)"""
import argparse, io, os, zipfile
from pathlib import Path

import pandas as pd
from dotenv import dotenv_values

OTHER = Path(__file__).resolve().parents[1]
KEYS = {'yodharinmonplub': 'KAGGLE_API_KEY_yodharinmonplub@actuarialtutor.org', 'krittanutsomtuas': 'KAGGLE_API_KEY_warinaeinwandee@actuarialtutor.org', 'actuarialtutor': 'KAGGLE_API_KEY_krittanutsomtung@actuarialtutor.org', 'aikayui': 'KAGGLE_API_KEY_aikayui@actuarialtutor.org'}
COMP = 'rsna-knee-abnormality-detection'
TEAM = 'KrittanutSomtuas'

parser = argparse.ArgumentParser()
parser.add_argument('--account', default='krittanutsomtuas', choices=sorted(KEYS))
args = parser.parse_args()
os.environ['KAGGLE_API_TOKEN'] = dotenv_values(OTHER / '.env')[KEYS[args.account]]
os.environ['KAGGLE_CONFIG_DIR'] = str(OTHER / '.kaggle')
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()
out = OTHER / 'artifacts/leaderboard'
out.mkdir(parents=True, exist_ok=True)
api.competition_leaderboard_download(COMP, str(out), quiet=True)
zp = max(out.glob('*.zip'), key=lambda p: p.stat().st_mtime)
with zipfile.ZipFile(zp) as z:
    lb = pd.read_csv(io.BytesIO(z.read(z.namelist()[0])))
lb = lb.sort_values(['Score', 'LastSubmissionDate'], ascending=[False, True]).reset_index(drop=True)
lb['Rank'] = lb['Score'].rank(method='min', ascending=False).astype(int)
n = len(lb)
lines = {'gold (top 10 + 0.2%)': 10 + int(0.002 * n), 'silver (top 5%)': int(0.05 * n), 'bronze (top 10%)': int(0.10 * n)}
print(f'teams {n} | top scores {lb.Score.head(5).tolist()}')
for name, r in lines.items():
    print(f'{name}: rank {r} -> score {lb.Score.iloc[r - 1]:.3f}')
me = lb[lb.TeamName.str.lower() == TEAM.lower()]
if len(me):
    r = me.iloc[0]
    tied = int((lb.Score == r.Score).sum())
    print(f'us ({TEAM}): score {r.Score:.3f} | rank {r.Rank} (tied block of {tied}, rows {lb.index[lb.Score == r.Score].min() + 1}-{lb.index[lb.Score == r.Score].max() + 1})')
else:
    print('team not found')
print('score counts near us:', lb.Score.value_counts().sort_index(ascending=False).head(12).to_dict())
