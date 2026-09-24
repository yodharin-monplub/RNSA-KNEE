"""Destroy a run's rental that never became usable and archive its state for a retry.

Records the machine id so the relaunch can exclude that host (cloud/<run>_exclude.json).
"""
import json
import shutil
import time

from vast import ROOT, STATE, RUN_NAME, api


def main():
    state = json.loads((STATE / 'instance.json').read_text())
    data = api('GET', f"instances/{state['instance_id']}/")['instances']
    data = data[0] if isinstance(data, list) and data else data
    machine = data.get('machine_id') if data else None
    print('instance', state['instance_id'], 'machine', machine, 'status',
          data and data.get('actual_status'), data and data.get('status_msg'))
    if data:
        print(api('DELETE', f"instances/{state['instance_id']}/"))
    credit = api('GET', 'users/current/').get('credit')
    elapsed = time.time() - state['created_at']
    (STATE / 'finished.json').write_text(json.dumps({
        'reason': 'abandoned_not_running', 'elapsed_seconds': elapsed,
        'estimated_compute_usd': elapsed / 3600 * state['quoted_hourly_usd'],
        'observed_credit_decrease_usd': max(0.0, state['initial_credit'] - credit)}))
    exclude_path = ROOT / f'cloud/{RUN_NAME}_exclude.json'
    excluded = json.loads(exclude_path.read_text()) if exclude_path.exists() else []
    if machine is not None and machine not in excluded:
        excluded.append(machine)
    exclude_path.write_text(json.dumps(excluded))
    n = 1
    while (STATE.parent / f'{RUN_NAME}_attempt{n}').exists():
        n += 1
    shutil.move(str(STATE), str(STATE.parent / f'{RUN_NAME}_attempt{n}'))
    print('archived to', f'{RUN_NAME}_attempt{n}', 'excluded machines', excluded)


if __name__ == '__main__':
    main()
