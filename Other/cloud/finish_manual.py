"""Manual completion when the checkpoint was delivered another way (e.g. remote Kaggle upload).

Backs up small artifacts, destroys the rental, and records finished.json with costs.
"""
import json
import sys
import time

from vast import ROOT, STATE, RUN_NAME, api, ssh_command, status
from watchdog import backup


def main():
    reason = sys.argv[1] if len(sys.argv) > 1 else 'manual'
    state = status(json.loads((STATE / 'instance.json').read_text()))
    destination = ROOT / f'artifacts/remote_{RUN_NAME}'
    destination.mkdir(parents=True, exist_ok=True)
    if state.get('actual_status') == 'running':
        print('small backup ok:', backup(ssh_command(state), destination, small_only=True))
    response = api('DELETE', f"instances/{state['instance_id']}/")
    assert response.get('success'), response
    time.sleep(5)
    credit = api('GET', 'users/current/').get('credit')
    elapsed = time.time() - state['created_at']
    record = {'reason': reason, 'elapsed_seconds': elapsed,
              'estimated_compute_usd': elapsed / 3600 * state['quoted_hourly_usd'],
              'observed_credit_decrease_usd': max(0.0, state['initial_credit'] - credit)}
    (STATE / 'finished.json').write_text(json.dumps(record))
    print(record)
    print('remaining instances:', [(i['id'], i.get('label'), i.get('actual_status'))
                                   for i in api('GET', 'instances/').get('instances', [])])


if __name__ == '__main__':
    main()
