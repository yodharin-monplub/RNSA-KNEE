"""Readiness gates for model62 (external fastMRI cache build)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def validate_ready():
    links = json.loads((ROOT / 'data/external/fastmri/links_knee.json').read_text())
    assert set(links) == {'knee_singlecoil_train', 'knee_singlecoil_val'}
    assert all(v.startswith('https://fastmri-dataset.s3.amazonaws.com/') for v in links.values())
    files = (ROOT / 'data/external/fastmri_plus/knee_file_list.csv').read_text().split()
    assert len(set(files)) > 1100
    prior_cost = 0.0
    for path in (ROOT / 'artifacts/cloud').rglob('finished.json'):
        previous = json.loads(path.read_text())
        prior_cost += max(previous.get('observed_credit_decrease_usd', 0), previous['estimated_compute_usd'])
    budget = json.loads((ROOT / 'cloud/model62_budget.json').read_text())
    assert prior_cost + budget['pilot_cap_usd'] <= budget['campaign_cap_usd']
    return {'prior_cost_usd': prior_cost, 'annotated_exams': len(set(files))}


if __name__ == '__main__':
    print(json.dumps(validate_ready(), indent=2))
