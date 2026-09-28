"""Check the rented GPU is not power-capped (2026-09-28: a Vast RTX 3090 host capped at 160 W ran ~2x slower at full
price). RSNA_CLOUD_RUN=modelNN python cloud/gpu_check.py -> prints power limit / clocks; exit 1 below MIN_WATTS."""
import json
import subprocess
import sys

from vast import STATE, ssh_command, status

MIN_WATTS = {'RTX 3090': 300, 'RTX 3090 Ti': 400, 'RTX 4090': 400, 'RTX A5000': 200, 'L40S': 300}
state = status(json.loads((STATE / 'instance.json').read_text()))
query = ('nvidia-smi --query-gpu=name,power.limit,power.default_limit,power.max_limit,clocks.sm,clocks.max.sm,'
         'temperature.gpu,utilization.gpu --format=csv,noheader,nounits')
out = subprocess.run(ssh_command(state) + [query], capture_output=True, text=True, timeout=60).stdout.strip()
print(out)
name, limit = out.split(',')[0].strip().replace('NVIDIA GeForce ', '').replace('NVIDIA ', ''), float(out.split(',')[1])
need = MIN_WATTS.get(name, 0)
print(f'{name}: power limit {limit:.0f} W (need >= {need} W) -> {"OK" if limit >= need else "CAPPED"}')
sys.exit(0 if limit >= need else 1)
