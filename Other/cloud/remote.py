"""Execute a command on the recorded rental, without exposing credentials."""
import argparse
import json
import subprocess

from vast import STATE, ssh_command

p = argparse.ArgumentParser()
p.add_argument('command')
p.add_argument('--timeout', type=int, default=60)
args = p.parse_args()
state = json.loads((STATE / 'instance.json').read_text())
raise SystemExit(subprocess.run(ssh_command(state) + [args.command], timeout=args.timeout).returncode)
