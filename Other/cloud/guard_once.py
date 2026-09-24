"""One foreground pass of the rental guard (used when the long-running watchdog process cannot be kept alive).

RSNA_CLOUD_RUN=modelNN python cloud/guard_once.py
Backs up small artifacts, and: on completion/failure -> full verified backup, destroy, write finished.json;
on cap/deadline -> backup, stop, write guard_stopped.json. Prints one JSON status line.
"""
import json
import time

from vast import ROOT, STATE, RUN_NAME, api, ssh_command, status
from watchdog import backup, backup_complete

if (STATE / 'finished.json').exists() or (STATE / 'guard_stopped.json').exists():
    print(json.dumps({'run': RUN_NAME, 'state': 'already closed'})); raise SystemExit
state = status(json.loads((STATE / 'instance.json').read_text()))
destination = ROOT / f'artifacts/remote_{RUN_NAME}'
destination.mkdir(parents=True, exist_ok=True)
elapsed = time.time() - state['created_at']
spent = elapsed / 3600 * state['quoted_hourly_usd']
report = {'run': RUN_NAME, 'status': state.get('actual_status'), 'hours': round(elapsed / 3600, 2), 'est_usd': round(spent, 3)}
if state.get('actual_status') == 'running' and state.get('ssh_host'):
    command = ssh_command(state)
    backup(command, destination, small_only=True)
    complete = (destination / f'{RUN_NAME}/TRAINING_COMPLETE').exists()
    failed = (destination / 'RUN_FAILED').exists()
    report.update(complete=complete, failed=failed)
    if complete or failed:
        if backup(command, destination, small_only=False) and backup_complete(command, destination):
            assert api('DELETE', f"instances/{state['instance_id']}/").get('success')
            (STATE / 'finished.json').write_text(json.dumps({'reason': 'failed' if failed else 'complete',
                                                             'elapsed_seconds': elapsed, 'estimated_compute_usd': spent}))
            report['action'] = 'backed up and destroyed'
        else:
            report['action'] = 'full backup incomplete; rental kept'
if 'action' not in report and (time.time() >= state['deadline'] or spent >= state['pilot_cap_usd'] - 0.25):
    try:
        backup(ssh_command(state), destination, small_only=False)
    except Exception:
        pass
    assert api('PUT', f"instances/{state['instance_id']}/", {'state': 'stopped'}).get('success')
    (STATE / 'guard_stopped.json').write_text(json.dumps({'reason': 'pilot_limit', 'credit_delta': spent}))
    report['action'] = 'cap/deadline reached; stopped'
print(json.dumps(report))
