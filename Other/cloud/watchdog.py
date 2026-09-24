"""Local persistent rental guard. Back up outputs and stop on a budget deadline."""
import json
import os
import subprocess
import time
from pathlib import Path

from vast import ROOT, STATE, RUN_NAME, MANAGED, api, ssh_command, status


def acquire_lock(handle):
    try:
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except OSError:
        return False


def backup(command, destination, small_only):
    """Stream /workspace/rsna/artifacts into destination via ssh + tar (no rsync needed)."""
    excludes = " --exclude='*.pt' --exclude='*.npy' --exclude='*.npz' --exclude='*.whl'" if small_only else ''
    remote = f"tar czf - -C /workspace/rsna/artifacts{excludes} ."
    sender = subprocess.Popen(command + [remote], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    try:
        receiver = subprocess.run(['tar', 'xzf', '-', '-C', str(destination)], stdin=sender.stdout,
                                  capture_output=True, timeout=60 if small_only else 900)
        sender.stdout.close()
        return sender.wait(timeout=60) == 0 and receiver.returncode == 0
    except subprocess.TimeoutExpired:
        # A slow uplink must count as a failed backup so the Kaggle-upload fallback can take over.
        sender.kill()
        return False


def backup_complete(command, destination):
    """True only when every remote artifact file exists locally with the same size.

    tar can exit cleanly on a truncated stream, so a successful-looking copy is verified explicitly.
    """
    listing = subprocess.run(command + ["cd /workspace/rsna/artifacts && find . -type f -printf '%P\\t%s\\n'"],
                             capture_output=True, timeout=120)
    if listing.returncode != 0:
        return False
    missing = []
    for line in listing.stdout.decode('utf-8', 'replace').splitlines():
        name, _, size = line.rpartition('\t')
        if not name:
            continue
        local = destination / name
        if not local.is_file() or local.stat().st_size != int(size):
            missing.append(name)
    if missing:
        print(f'Backup incomplete: {len(missing)} file(s) missing, e.g. {missing[:3]}', flush=True)
    return not missing


def main():
    lock = (STATE / 'watchdog.lock').open('a+')
    if not acquire_lock(lock):
        print('Another watchdog already owns this project.', flush=True)
        return
    state = json.loads((STATE / 'instance.json').read_text())
    destination = ROOT / (f'artifacts/remote_{RUN_NAME}' if RUN_NAME else 'artifacts/remote')
    destination.mkdir(parents=True, exist_ok=True)
    failures = full_failures = 0
    others_cost = 0.0
    last_poll = max(time.time(), state['created_at'])
    if time.time() - state['created_at'] > 120:
        # Restarted watchdog: assume other instances ran the whole time so far (conservative for them,
        # generous to us only up to their listed rates).
        others_cost = sum(float(i.get('dph_total') or 0) for i in api('GET', 'instances/').get('instances', [])
                          if i.get('id') != state['instance_id'] and i.get('actual_status') == 'running'
                          ) * (time.time() - state['created_at']) / 3600
    while True:
        try:
            if (STATE / 'finished.json').exists():
                print('Rental cleanup already recorded; watchdog exiting.', flush=True)
                return
            state = status(state)
            elapsed = time.time() - state['created_at']
            if state.get('actual_status') == 'running' and state.get('ssh_host'):
                command = ssh_command(state)
                # Metrics/logs arrive promptly even when checkpoints are large.
                backup(command, destination, small_only=True)
                complete = ((destination / f'{RUN_NAME}/TRAINING_COMPLETE').exists()
                            if MANAGED else (destination / 'ENSEMBLE_VALIDATED').exists())
                failed = MANAGED and (destination / 'RUN_FAILED').exists()
                uploaded = MANAGED and (destination / f'{RUN_NAME}/UPLOAD_DONE').exists()
                if complete or failed:
                    backed_up = uploaded or (backup(command, destination, small_only=False)
                                             and backup_complete(command, destination))
                    if not backed_up:
                        full_failures += 1
                        if full_failures == 2 and MANAGED and (destination / f'{RUN_NAME}/history.json').exists():
                            # Slow uplink: let the rental push the checkpoint to Kaggle instead.
                            start = (f'cd /workspace/rsna && nohup python cloud/remote_upload.py {RUN_NAME} '
                                     '> artifacts/upload.log 2>&1 &')
                            subprocess.run(command + [start], timeout=60)
                            print('Full backup failing; started remote Kaggle upload.', flush=True)
                    if backed_up:
                        response = api('DELETE', f"instances/{state['instance_id']}/")
                        if not response.get('success'):
                            raise RuntimeError('Destroy request was not confirmed')
                        (STATE / 'finished.json').write_text(json.dumps({
                            'reason': 'failed' if failed else 'complete', 'elapsed_seconds': elapsed,
                            'estimated_compute_usd': elapsed / 3600 * state['quoted_hourly_usd']}))
                        print('Artifacts copied; finished instance destroyed.', flush=True)
                        return
            credit = api('GET', 'users/current/').get('credit')
            # The account is shared with other projects: remove their running cost from the credit
            # delta, but never count less than this rental's own time-based estimate.
            now = time.time()
            others = [i for i in api('GET', 'instances/').get('instances', [])
                      if i.get('id') != state['instance_id'] and i.get('actual_status') == 'running']
            others_cost += sum(float(i.get('dph_total') or 0) for i in others) * (now - last_poll) / 3600
            last_poll = now
            own_estimate = elapsed / 3600 * state['quoted_hourly_usd']
            delta = max(0, (state.get('initial_credit') or credit or 0) - (credit or 0))
            spent = max(own_estimate, delta - others_cost)
            if time.time() >= state['deadline'] or spent >= state['pilot_cap_usd'] - 0.25:
                try:
                    if state.get('actual_status') == 'running' and state.get('ssh_host'):
                        backup(ssh_command(state), destination, small_only=False)
                except Exception:
                    pass
                response = api('PUT', f"instances/{state['instance_id']}/", {'state': 'stopped'})
                if not response.get('success'):
                    raise RuntimeError('Stop request was not confirmed')
                (STATE / 'guard_stopped.json').write_text(json.dumps({'reason': 'pilot_limit', 'credit_delta': spent}))
                print('Pilot limit reached; compute stopped. Storage retained for artifact recovery.', flush=True)
                return
            failures = 0
        except Exception as exc:
            failures += 1
            print(f'Watchdog retry {failures}: {type(exc).__name__}: {str(exc)[:200]}', flush=True)
        time.sleep(30)


if __name__ == '__main__':
    main()
