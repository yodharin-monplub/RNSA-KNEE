"""Create the rental for RSNA_CLOUD_RUN, wait until it runs, deploy, and exit (no long-lived watchdog process).

The guard is then cloud/guard_once.py, run periodically in the foreground. Used because the host PC kills
long-running background shells under memory pressure.
"""
import json
import subprocess
import time

import vast
import deploy_run

vast.launch()
deadline = time.time() + 25 * 60
while time.time() < deadline:
    state = vast.status(json.loads((vast.STATE / 'instance.json').read_text()))
    print(time.strftime('%H:%M:%S'), state.get('actual_status'), flush=True)
    if state.get('actual_status') == 'running' and state.get('ssh_host'):
        try:
            deploy_run.main()
            break
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, AssertionError) as exc:
            print('deploy attempt failed:', type(exc).__name__, flush=True)
    time.sleep(20)
else:
    raise SystemExit('rental never became reachable; inspect and abandon')
