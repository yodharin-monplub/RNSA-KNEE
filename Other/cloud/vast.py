"""Small Vast API client. Credentials are never printed or stored in state."""
import argparse
import json
import importlib
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
RUN_NAME = os.environ.get('RSNA_CLOUD_RUN', '')
MANAGED = bool(re.fullmatch(r'model\d+', RUN_NAME))
if RUN_NAME and not (MANAGED and (ROOT / f'cloud/{RUN_NAME}_budget.json').exists()):
    raise ValueError('Unknown cloud run name')
STATE = ROOT / 'artifacts/cloud' / RUN_NAME
STATE.mkdir(parents=True, exist_ok=True)
load_dotenv(ROOT / '.env')


def api(method, path, payload=None):
    response = requests.request(method, 'https://console.vast.ai/api/v0/' + path,
                                headers={'Authorization': 'Bearer ' + os.environ['VAST_AI_API_KEY']},
                                json=payload, timeout=45)
    if not response.ok:
        raise RuntimeError(f'Vast {method} {path}: HTTP {response.status_code}')
    return response.json()


def ssh_command(state):
    return ['ssh', '-F', os.devnull, '-i', str(ssh_key()), '-p', str(state['ssh_port']),
            '-o', 'IdentitiesOnly=yes', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
            '-o', 'StrictHostKeyChecking=accept-new', '-o', 'ServerAliveInterval=30',
            '-o', 'UserKnownHostsFile="' + str(ROOT / '.cache/ssh/known_hosts') + '"',
            'root@' + state['ssh_host']]


def ssh_key():
    """Windows OpenSSH rejects keys readable by other users; use a private copy there."""
    key = ROOT / '.cache/ssh/pilot'
    if os.name != 'nt':
        return key
    private = Path.home() / '.ssh' / 'rsna_pilot'
    if not private.exists() or private.read_bytes() != key.read_bytes():
        private.parent.mkdir(parents=True, exist_ok=True)
        private.write_bytes(key.read_bytes())
        user = os.environ['USERNAME']
        subprocess.run(['icacls', str(private), '/inheritance:r', '/grant:r', f'{user}:F'],
                       check=True, capture_output=True)
    return private


def status(state):
    data = api('GET', f"instances/{state['instance_id']}/")['instances']
    if not data:
        state['actual_status'] = 'destroyed' if (STATE / 'finished.json').exists() else 'not_found'
        (STATE / 'instance.json').write_text(json.dumps(state, indent=2))
        return state
    if isinstance(data, list):
        data = data[0]
    for field in ['ssh_host', 'ssh_port', 'actual_status', 'dph_total', 'gpu_name',
                  'start_date', 'disk_space', 'inet_up_billed', 'inet_down_billed']:
        if field in data:
            state[field] = data[field]
    direct = data.get('ports', {}).get('22/tcp', [])
    if direct and data.get('public_ipaddr'):
        state['ssh_host'] = data['public_ipaddr']
        state['ssh_port'] = int(direct[0]['HostPort'])
    (STATE / 'instance.json').write_text(json.dumps(state, indent=2))
    return state


def launch():
    if (STATE / 'instance.json').exists():
        raise RuntimeError('A run is already recorded. Archive its cloud state and results before launching another run; never overwrite an active run.')
    budget_path = ROOT / (f'cloud/{RUN_NAME}_budget.json' if MANAGED else 'cloud/budget.json')
    budget = json.loads(budget_path.read_text())
    if MANAGED:
        ready = importlib.import_module(f'{RUN_NAME}_ready').validate_ready()
        (STATE / 'ready.json').write_text(json.dumps(ready, indent=2))
    instances = api('GET', 'instances/').get('instances', [])
    # Other projects share this account; only this project's rentals block a launch.
    # The credit-delta budget guard stays conservative because it counts all spending.
    project = [i for i in instances if str(i.get('label') or '').startswith('rsna-knee')]
    if any(i.get('label') == budget.get('label') for i in project):
        raise RuntimeError('This run already has a rental; inspect before creating another')
    if project:
        # Concurrent project runs: reserve every active run's full cap against the campaign cap.
        active_caps = 0.0
        for other in (ROOT / 'artifacts/cloud').iterdir():
            path = other / 'instance.json'
            if other.is_dir() and path.exists() and not (other / 'finished.json').exists():
                active_caps += float(json.loads(path.read_text()).get('pilot_cap_usd', 0))
        prior = ready.get('prior_cost_usd', float('inf')) if MANAGED else float('inf')
        if prior + active_caps + budget['pilot_cap_usd'] > budget['campaign_cap_usd']:
            raise RuntimeError('Concurrent rsna-knee rentals would exceed the campaign cap')
        print('Concurrent project rentals:', [i.get('label') for i in project],
              'reserved', active_caps, flush=True)
    if instances:
        print('Ignoring non-project instances:', [(i['id'], i.get('label'), i.get('actual_status'))
                                                   for i in instances], flush=True)
    key = ROOT / '.cache/ssh/pilot'
    key.parent.mkdir(parents=True, exist_ok=True)
    if not key.exists():
        subprocess.run(['ssh-keygen', '-q', '-t', 'ed25519', '-N', '', '-C', 'rsna-pilot', '-f', str(key)], check=True)
    public_key = key.with_suffix('.pub').read_text().strip()
    registered = api('GET', 'ssh/')
    if not isinstance(registered, list):
        raise RuntimeError('Unexpected SSH key inventory response')
    if not any((row.get('public_key') or '').split()[:2] == public_key.split()[:2]
               for row in registered):
        result = api('POST', 'ssh/', {'ssh_key': public_key})
        if not result.get('success'):
            raise RuntimeError('SSH public key registration failed')
    query = {'verified': {'eq': True}, 'rentable': {'eq': True}, 'rented': {'eq': False},
             # AGENTS.md 2026-09-23: prefer affordable GPUs, longer training time is fine, so a run's budget
             # file may widen the card list and lower the VRAM floor.
             'num_gpus': {'eq': 1},
             'gpu_name': {'in': budget.get('gpu_names', ['A40', 'RTX 3090', 'RTX 4090'])},
             'gpu_ram': {'gte': budget.get('min_gpu_ram', 24000)}, 'disk_space': {'gte': budget['disk_gb']},
             'inet_down': {'gte': 500}, 'inet_down_cost': {'lte': budget['max_transfer_usd_per_gb']},
             'inet_up_cost': {'lte': budget['max_transfer_usd_per_gb']},
             'reliability': {'gte': budget.get('min_reliability', 0.99)}, 'direct_port_count': {'gte': 1},
             'cuda_max_good': {'gte': 12.4}, 'cpu_ram': {'gte': 24000},
             'type': 'ondemand', 'limit': 20, 'order': [['dph_total', 'asc']],
             'allocated_storage': budget['disk_gb'], 'dph_total': {'lte': budget['max_hourly_usd']}}
    # Optional host-quality floors (model25 lost hours to an I/O- and uplink-starved host).
    for key in ('disk_bw', 'inet_up'):
        if f'min_{key}' in budget:
            query[key] = {'gte': budget[f'min_{key}']}
    exclude_path = ROOT / f'cloud/{RUN_NAME}_exclude.json'
    if exclude_path.exists():
        query['machine_id'] = {'notin': json.loads(exclude_path.read_text())}
    offers = api('POST', 'bundles/', query)['offers']
    if 'machine_id' in query:
        offers = [o for o in offers if o.get('machine_id') not in query['machine_id']['notin']]
    if not offers:
        raise RuntimeError('No offer within configured budget')
    # Prefer 48 GB A40 when close in price; larger batches speed extraction.
    offers.sort(key=lambda x: x['dph_total'] / (1.4 if x['gpu_name'] == 'A40' else 1))
    offer = offers[0]
    price = float(offer['dph_total'])
    hours = min(budget['pilot_max_hours'], (budget['pilot_cap_usd'] - 0.5) / price)
    initial_credit = api('GET', 'users/current/').get('credit')
    if initial_credit is None or initial_credit < budget['pilot_cap_usd']:
        raise RuntimeError('Insufficient verified account credit for this capped run')
    payload = {'client_id': 'me', 'image': 'pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime',
               'disk': budget['disk_gb'], 'runtype': 'ssh_direct',
               'label': budget.get('label', 'rsna-knee-pilot-model2-model3'), 'cancel_unavail': True}
    result = api('PUT', f"asks/{offer['id']}/", payload)
    if not result.get('success') or not result.get('new_contract'):
        raise RuntimeError('Rental was not created')
    state = {'instance_id': result['new_contract'], 'offer_id': offer['id'],
             'created_at': time.time(), 'deadline': time.time() + hours * 3600,
             'quoted_hourly_usd': price, 'pilot_cap_usd': budget['pilot_cap_usd'],
             'initial_credit': initial_credit,
             'gpu_name': offer['gpu_name'], 'ssh_key_id': result.get('key', {}).get('id')}
    (STATE / 'instance.json').write_text(json.dumps(state, indent=2))
    print(json.dumps(state, indent=2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('action', choices=['launch', 'status', 'stop', 'destroy'])
    args = p.parse_args()
    if args.action == 'launch':
        launch()
        if MANAGED:
            # Keep the guard in the managed exec session; detached children may
            # be terminated when their sandbox's foreground command exits.
            from watchdog import main
            main()
    else:
        state = json.loads((STATE / 'instance.json').read_text())
        if args.action == 'status':
            print(json.dumps(status(state), indent=2))
        elif args.action == 'stop':
            print(api('PUT', f"instances/{state['instance_id']}/", {'state': 'stopped'}))
        elif args.action == 'destroy':
            print(api('DELETE', f"instances/{state['instance_id']}/"))
