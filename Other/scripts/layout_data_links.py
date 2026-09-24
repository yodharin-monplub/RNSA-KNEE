"""Split Data/ into competition / labels / external and rebuild Other/data as a folder of links (idempotent).

AGENTS.md layout: Data/ (competition + external data in subfolders), Model/modelN, Other/ (everything else).
All code still addresses ROOT/data/... with ROOT = Other/, so Other/data holds junctions (directories) and hard
links (the five competition CSVs) that point into Data/. Removing a junction never touches its target.
"""
import os
import subprocess
from pathlib import Path

OTHER = Path(__file__).resolve().parents[1]
DATA = OTHER.parent / 'Data'
links = OTHER / 'data'
if links.exists() and os.path.isjunction(links):
    os.rmdir(links)  # removes only the junction
links.mkdir(exist_ok=True)
plan = {'competition': ['sample_submission.csv', 'test.csv', 'test_series.csv', 'train.csv', 'train_series.csv', '.complete', 'pixel_spec'],
        'labels': ['llm_relabel', 'raptor_labels', 'weak_labels', 'weak_labels_v4'],
        '': ['external']}
for folder, names in plan.items():
    (DATA / folder).mkdir(exist_ok=True)
    for name in names:
        target = DATA / folder / name
        if not target.exists() and (DATA / name).exists():
            (DATA / name).rename(target)
        assert target.exists(), target
        link = links / name
        if link.exists():
            continue
        if target.is_dir():
            subprocess.run(['cmd', '/c', 'mklink', '/J', str(link), str(target)], check=True, capture_output=True)
        else:
            os.link(target, link)
print('Data:', sorted(p.name for p in DATA.iterdir()))
print('Other/data:', sorted(p.name for p in links.iterdir()))
