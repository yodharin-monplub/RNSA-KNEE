"""Scan the staged git diff for any Other/.env value (and a few token shapes). Prints counts only, never the values.
Exit 1 if anything is found."""
import re
import subprocess
import sys
from pathlib import Path

from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[2]   # the project root
GIT = r'C:\Program Files\Git\cmd\git.exe'
raw_env = dotenv_values(ROOT / 'Other/.env')
env = {k: v for k, v in raw_env.items() if v and len(v) >= 12}
# a line missing its '=' puts the token into the KEY (2026-10-04: one such line) -> scan those key names too
env.update({f'malformed_key_{i}': k for i, k in enumerate(raw_env) if 'KGAT_' in k or re.search(r'[0-9a-f]{24,}', k)})
diff = subprocess.run([GIT, 'diff', '--cached', '--no-color', '-U0'], cwd=ROOT, capture_output=True, text=True,
                      encoding='utf-8', errors='replace').stdout
names = subprocess.run([GIT, 'diff', '--cached', '--name-only'], cwd=ROOT, capture_output=True, text=True).stdout.split()
hits = {k: diff.count(v) for k, v in env.items() if diff.count(v)}
patterns = {'kaggle_token': r'KGAT_[A-Za-z0-9]{20,}', 'openrouter': r'sk-or-v1-[a-f0-9]{20,}', 'discord_bot': r'[MN][A-Za-z\d]{23}\.[\w-]{6}\.[\w-]{27}',
            'private_key': r'-----BEGIN [A-Z ]*PRIVATE KEY-----'}
shape_hits = {k: len(re.findall(p, diff)) for k, p in patterns.items() if re.findall(p, diff)}
bad_files = [n for n in names if any(s in n.lower() for s in ('.env', 'email.txt', '.kaggle', 'id_rsa', 'deploy'))]
print(f'staged files {len(names)} | diff chars {len(diff)} | env values checked {len(env)} | env hits {hits} | '
      f'token-shape hits {shape_hits} | suspicious file names {bad_files}')
sys.exit(1 if hits or shape_hits or bad_files else 0)
